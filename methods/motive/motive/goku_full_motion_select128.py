"""Select and atomically materialize an exact full-motion editing dataset.

The selector is intentionally the last fail-closed stage.  It trusts neither
the pending generation manifest, Wan's generated index, nor postcheck verdict
prose.  It replays their deterministic validators, rehashes every selected
byte, recomputes each visual aggregate, and only then copies an exact number of
samples into a create-only dataset directory.

Selection is deterministic: three-dynamic-unit clips precede two-unit clips,
which precede one-unit clips; the finalizer's primary order and IID break ties.
By default the output contains exactly 128 samples and at least 32 multi-unit
samples.  An insufficient hard-pass pool is an error, never a smaller dataset.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

from . import goku_full_motion_finalize as finalizer
from . import goku_full_motion_postcheck as postcheck
from . import goku_full_motion_shard_manifest as shard_manifest


SELECT_SCHEMA = "motive-goku-full-motion-select128-v6"
DATASET_ROW_SCHEMA = "motive-goku-full-motion-dataset-row-v6"
SUMMARY_SCHEMA = "motive-goku-full-motion-dataset-summary-v1"
DONE_SCHEMA = "motive-goku-full-motion-dataset-done-v1"
ARTIFACT_SCHEMA = "motive-goku-full-motion-dataset-artifact-v1"

DEFAULT_EXACT_SIZE = 128
DEFAULT_MIN_MULTI_UNIT = 32
MANIFEST_NAME = "dataset_manifest.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
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
_GENERATION_SCHEMA_V6 = "motive-goku-full-motion-generation-v6"
_MOTION_SPEC_SCHEMA_V6 = "motive-goku-full-motion-generation-spec-v6"
_QWEN_EVIDENCE_SCHEMA_V6 = "motive-goku-full-motion-qwen-evidence-v6"
_QWEN_RECORD_SCHEMA_V6 = "goku-full-motion-qwen-record-v6"
_QWEN_HARD_GATE_SCHEMA_V6 = "goku-full-motion-hard-gate-v6"
_POSTCHECK_SCHEMA_V6 = "motive-goku-full-motion-postcheck-v6"
_POSTCHECK_QWEN_BINDING_SCHEMA_V6 = (
    "motive-goku-full-motion-postcheck-qwen-evidence-binding-v6"
)
_SOURCE_INVENTORY_ALIGNMENT_SCHEMA_V4 = (
    "motive-goku-full-motion-source-inventory-alignment-v4"
)
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
_QWEN_HARD_GATE_KEYS = {
    "schema_version",
    "change_region_proposals_sha256",
    "coverage_authority_sha256",
    "coverage_authority_inventory_sha256",
    "coverage_authority_assignments_sha256",
    "coverage_authority_alignment_sha256",
    "i0_grounding_sha256",
    "source_census_sha256",
    "source_census_canonicalization_sha256",
    "secondary_source_census_sha256",
    "secondary_source_census_canonicalization_sha256",
    "source_inventory_alignment_sha256",
    "target_plan_canonicalization_sha256",
    "decision",
    "risk_codes",
}
_MIN_GENERATION_KEYS = {
    "schema_version",
    "iid",
    "group_id",
    "family",
    "source_video",
    "resolved_source_video",
    "source_video_sha256",
    "anchor_image",
    "resolved_anchor_image",
    "anchor_sha256",
    "edit_instruction",
    "edit_instruction_sha256",
    "motion_spec",
    "motion_spec_sha256",
    "action_change_substantive",
    "manifest_role",
    "human_review_status",
    "generation_authorized",
    "production_eligible",
    "approval",
    "annotation_source",
    "human_reviewed",
}
_REQUIRED_CONDITIONING_FIELDS = {
    "conditioning_anchor_original",
    "conditioning_frame0_float32",
    "conditioning_frame0_png",
}


class GokuFullMotionSelect128Error(RuntimeError):
    """An input closure, quota, byte binding, or publication is invalid."""


def _reject_constant(value: str) -> None:
    raise GokuFullMotionSelect128Error(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GokuFullMotionSelect128Error(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GokuFullMotionSelect128Error(
            "value is not finite canonical JSON"
        ) from error


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GokuFullMotionSelect128Error(
            "value is not finite JSON"
        ) from error


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _ordered_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(
        b"".join(value.encode("utf-8") + b"\n" for value in values)
    ).hexdigest()


def _stable_read(path: Path, *, context: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise GokuFullMotionSelect128Error(
            f"{context} must be a non-symlink regular file: {path}"
        )
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
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
    if before_identity != after_identity or len(raw) != before.st_size:
        raise GokuFullMotionSelect128Error(
            f"{context} changed while being read: {path}"
        )
    return raw


def _file_digest(path: Path, *, context: str) -> str:
    return hashlib.sha256(_stable_read(path, context=context)).hexdigest()


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GokuFullMotionSelect128Error(
            f"{context} is not UTF-8"
        ) from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise GokuFullMotionSelect128Error(
            f"{context} is not strict JSON"
        ) from error


def _json_object(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    raw = _stable_read(path, context=context)
    value = _parse_json(raw, context=context)
    if not isinstance(value, dict):
        raise GokuFullMotionSelect128Error(f"{context} must be one object")
    return value, raw


def _jsonl_rows_from_bytes(
    raw: bytes, *, context: str, allow_empty: bool = False
) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GokuFullMotionSelect128Error(
            f"{context} is not UTF-8"
        ) from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise GokuFullMotionSelect128Error(
                f"{context} has a blank line at {number}"
            )
        value = _parse_json(
            line.encode("utf-8"), context=f"{context} line {number}"
        )
        if not isinstance(value, dict):
            raise GokuFullMotionSelect128Error(
                f"{context} line {number} is not an object"
            )
        rows.append(value)
    if not rows and not allow_empty:
        raise GokuFullMotionSelect128Error(f"{context} is empty")
    return rows


def _regular_file(path: str | Path, *, context: str) -> Path:
    try:
        candidate = Path(path).expanduser()
    except (TypeError, ValueError, OSError) as error:
        raise GokuFullMotionSelect128Error(
            f"{context} is not a valid file path"
        ) from error
    if candidate.is_symlink() or not candidate.is_file():
        raise GokuFullMotionSelect128Error(
            f"{context} must be a non-symlink regular file: {candidate}"
        )
    return candidate.resolve(strict=True)


def _regular_directory(path: str | Path, *, context: str) -> Path:
    try:
        candidate = Path(path).expanduser()
    except (TypeError, ValueError, OSError) as error:
        raise GokuFullMotionSelect128Error(
            f"{context} is not a valid directory path"
        ) from error
    if candidate.is_symlink() or not candidate.is_dir():
        raise GokuFullMotionSelect128Error(
            f"{context} must be a non-symlink directory: {candidate}"
        )
    return candidate.resolve(strict=True)


def _resolve_file(value: Any, root: Path, *, context: str) -> Path:
    text = _text(value, context=context)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return _regular_file(candidate, context=context)


def _text(value: Any, *, context: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise GokuFullMotionSelect128Error(
            f"{context} must be non-empty trimmed text"
        )
    if "\x00" in value:
        raise GokuFullMotionSelect128Error(f"{context} contains NUL")
    return value


def _digest(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GokuFullMotionSelect128Error(
            f"{context} must be a lowercase SHA-256"
        )
    return value


def _safe_iid(value: Any, *, context: str = "iid") -> str:
    iid = _text(value, context=context)
    if _SAFE_IID_RE.fullmatch(iid) is None or iid in {".", ".."}:
        raise GokuFullMotionSelect128Error(f"{context} is unsafe")
    return iid


def _safe_basename(value: Any, *, context: str) -> str:
    name = _text(value, context=context)
    if Path(name).name != name or name in {".", ".."}:
        raise GokuFullMotionSelect128Error(
            f"{context} must be one safe basename"
        )
    return name


def _same_scalar(value: Any, expected: Any) -> bool:
    return type(value) is type(expected) and value == expected


def _validate_generation_row(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one exact, closed finalizer generation row."""

    if not isinstance(value, Mapping):
        raise GokuFullMotionSelect128Error("generation row is not an object")
    row = dict(value)
    if row.get("schema_version") != _GENERATION_SCHEMA_V6:
        raise GokuFullMotionSelect128Error("generation row schema differs")
    missing = _MIN_GENERATION_KEYS - set(row)
    if missing:
        raise GokuFullMotionSelect128Error(
            f"generation row lacks required fields: {sorted(missing)}"
        )
    spec = row.get("motion_spec")
    if not isinstance(spec, Mapping) or set(spec) != _MOTION_SPEC_KEYS:
        raise GokuFullMotionSelect128Error("motion_spec is not closed")
    if spec.get("schema_version") != _MOTION_SPEC_SCHEMA_V6:
        raise GokuFullMotionSelect128Error("motion_spec schema differs")
    if _object_digest(spec) != row.get("motion_spec_sha256"):
        raise GokuFullMotionSelect128Error("motion_spec SHA differs")
    evidence = row.get("qwen_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != _QWEN_EVIDENCE_KEYS:
        raise GokuFullMotionSelect128Error("qwen_evidence is not closed")
    if (
        evidence.get("schema_version") != _QWEN_EVIDENCE_SCHEMA_V6
        or evidence.get("record_schema_version") != _QWEN_RECORD_SCHEMA_V6
    ):
        raise GokuFullMotionSelect128Error(
            "Qwen v6 evidence/record schema differs"
        )
    hard_gate = evidence.get("hard_gate")
    if (
        not isinstance(hard_gate, Mapping)
        or set(hard_gate) != _QWEN_HARD_GATE_KEYS
        or hard_gate.get("schema_version") != _QWEN_HARD_GATE_SCHEMA_V6
    ):
        raise GokuFullMotionSelect128Error("Qwen v6 hard gate is not closed")

    try:
        validated = finalizer.validate_generation_row(row)
    except finalizer.GokuFullMotionFinalizeError as error:
        raise GokuFullMotionSelect128Error(
            f"generation row validation failed: {error}"
        ) from error
    result = dict(validated)
    spec = dict(result["motion_spec"])
    evidence = dict(result["qwen_evidence"])
    hard_gate = dict(evidence["hard_gate"])
    try:
        from .goku_full_motion_contract import (
            validate_source_census,
            validate_source_inventory_alignment,
        )
        from .goku_full_motion_qwen import (
            _RECORD_KEYS,
            build_coverage_authority,
            validate_change_region_proposals,
            validate_coverage_authority,
            validate_coverage_authority_assignments,
            validate_coverage_authority_alignment,
            validate_coverage_authority_inventory,
            validate_i0_grounding,
            validate_source_census_i0_binding,
        )

        proposals = validate_change_region_proposals(
            spec["change_region_proposals"], expected_iid=str(result["iid"])
        )
        authority = validate_coverage_authority(
            spec["coverage_authority"],
            expected_iid=str(result["iid"]),
            change_region_proposals=proposals,
        )
        authority_inventory = validate_coverage_authority_inventory(
            authority["inventory"], expected_iid=str(result["iid"])
        )
        authority_assignments = validate_coverage_authority_assignments(
            authority["assignments"],
            expected_iid=str(result["iid"]),
            coverage_authority_inventory=authority_inventory,
            change_region_proposals=proposals,
        )
        if authority != build_coverage_authority(
            coverage_authority_inventory=authority_inventory,
            coverage_authority_assignments=authority_assignments,
            change_region_proposals=proposals,
        ):
            raise GokuFullMotionSelect128Error(
                "coverage authority A0a/A0b composition differs"
            )
        grounding = validate_i0_grounding(
            spec["i0_grounding"], expected_iid=str(result["iid"])
        )
        primary = validate_source_census(spec["source_census"])
        primary = validate_source_census_i0_binding(primary, grounding)
        secondary = validate_source_census(spec["secondary_source_census"])
        secondary = validate_source_census_i0_binding(secondary, grounding)
        alignment = validate_source_inventory_alignment(
            spec["source_inventory_alignment"],
            primary=primary,
            secondary=secondary,
        )
        if alignment.get("schema_version") != _SOURCE_INVENTORY_ALIGNMENT_SCHEMA_V4:
            raise GokuFullMotionSelect128Error(
                "source inventory alignment v4 schema differs"
            )
        authority_alignment = validate_coverage_authority_alignment(
            spec["coverage_authority_alignment"],
            coverage_authority=authority,
            change_region_proposals=proposals,
            i0_grounding=grounding,
            primary=primary,
            secondary=secondary,
            source_inventory_alignment=alignment,
        )
    except Exception as error:
        raise GokuFullMotionSelect128Error(
            f"generation v6 A0/exact-I0/source closure differs: {error}"
        ) from error
    canonicalization_fields = (
        "source_census_canonicalization",
        "secondary_source_census_canonicalization",
        "target_plan_canonicalization",
    )
    if any(
        not isinstance(evidence.get(field), Mapping)
        for field in canonicalization_fields
    ):
        raise GokuFullMotionSelect128Error(
            "Qwen v6 canonicalization evidence is malformed"
        )
    proposals_sha = _object_digest(proposals)
    authority_inventory_sha = _object_digest(authority_inventory)
    authority_assignments_sha = _object_digest(authority_assignments)
    authority_sha = _object_digest(authority)
    authority_alignment_sha = _object_digest(authority_alignment)
    grounding_sha = _object_digest(grounding)
    primary_sha = _object_digest(primary)
    secondary_sha = _object_digest(secondary)
    alignment_sha = _object_digest(alignment)
    source_receipt_sha = _object_digest(
        evidence["source_census_canonicalization"]
    )
    secondary_receipt_sha = _object_digest(
        evidence["secondary_source_census_canonicalization"]
    )
    target_receipt_sha = _object_digest(
        evidence["target_plan_canonicalization"]
    )
    expected_hard_gate = {
        "schema_version": _QWEN_HARD_GATE_SCHEMA_V6,
        "change_region_proposals_sha256": proposals_sha,
        "coverage_authority_inventory_sha256": authority_inventory_sha,
        "coverage_authority_assignments_sha256": authority_assignments_sha,
        "coverage_authority_sha256": authority_sha,
        "coverage_authority_alignment_sha256": authority_alignment_sha,
        "i0_grounding_sha256": grounding_sha,
        "source_census_sha256": primary_sha,
        "source_census_canonicalization_sha256": source_receipt_sha,
        "secondary_source_census_sha256": secondary_sha,
        "secondary_source_census_canonicalization_sha256": (
            secondary_receipt_sha
        ),
        "source_inventory_alignment_sha256": alignment_sha,
        "target_plan_canonicalization_sha256": target_receipt_sha,
        "decision": "pass",
        "risk_codes": [],
    }
    if (
        evidence.get("result_digest") != spec.get("qwen_result_digest")
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
        or evidence.get("i0_grounding_digest") != grounding_sha
        or evidence.get("source_census_digest") != primary_sha
        or evidence.get("secondary_source_census_digest") != secondary_sha
        or evidence.get("source_inventory_alignment_digest") != alignment_sha
        or evidence.get("source_census_canonicalization_digest")
        != source_receipt_sha
        or evidence.get("secondary_source_census_canonicalization_digest")
        != secondary_receipt_sha
        or evidence.get("target_plan_canonicalization_digest")
        != target_receipt_sha
        or evidence.get("target_plan_digest")
        != _object_digest(spec["target_plan"])
        or evidence.get("compiled_instruction_digest")
        != _object_digest(spec["compiled_instruction"])
        or evidence.get("coverage_critic_digest")
        != _object_digest(spec["coverage_critic"])
        or evidence.get("full_motion_contract_digest")
        != _object_digest(spec["full_motion_contract"])
        or hard_gate != expected_hard_gate
    ):
        raise GokuFullMotionSelect128Error(
            "Qwen v6 A0/exact-I0/evidence/hard-gate binding differs"
        )
    record_payload = evidence.get("qwen_record_payload")
    if (
        not isinstance(record_payload, Mapping)
        or set(record_payload) != _RECORD_KEYS
        or record_payload.get("schema_version") != _QWEN_RECORD_SCHEMA_V6
        or record_payload.get("iid") != result.get("iid")
        or record_payload.get("status") != "ok"
        or record_payload.get("pipeline_decision") != "pass"
        or record_payload.get("change_region_proposals") != proposals
        or record_payload.get("coverage_authority") != authority
        or record_payload.get("coverage_authority_inventory_prompt_digest")
        != evidence.get("coverage_authority_inventory_prompt_digest")
        or record_payload.get(
            "coverage_authority_inventory_visual_input_digest"
        )
        != evidence.get("coverage_authority_inventory_visual_input_digest")
        or record_payload.get("coverage_authority_inventory_digest")
        != authority_inventory_sha
        or record_payload.get("coverage_authority_assignments_prompt_digest")
        != evidence.get("coverage_authority_assignments_prompt_digest")
        or record_payload.get(
            "coverage_authority_assignments_visual_input_digest"
        )
        != evidence.get("coverage_authority_assignments_visual_input_digest")
        or record_payload.get("coverage_authority_assignments_digest")
        != authority_assignments_sha
        or record_payload.get("coverage_authority_digest") != authority_sha
        or record_payload.get("hard_gate") != hard_gate
        or record_payload.get("result_digest") != evidence.get("result_digest")
        or record_payload.get("provenance_digest")
        != evidence.get("provenance_digest")
    ):
        raise GokuFullMotionSelect128Error(
            "Qwen v6 complete record payload binding differs"
        )
    _safe_iid(result.get("iid"), context="generation iid")
    finalization = result.get("full_motion_finalization")
    if not isinstance(finalization, Mapping):
        raise GokuFullMotionSelect128Error(
            "generation row lacks full_motion_finalization"
        )
    if finalization.get("selection_bucket") != "primary":
        raise GokuFullMotionSelect128Error(
            "selector input must contain primary generation rows only"
        )
    return result


def _validate_finalizer_parent(
    manifest_path: Path,
    *,
    manifest_raw: bytes,
    manifest_rows: int,
    done_path: Path | None = None,
) -> dict[str, Any]:
    parent = manifest_path.parent
    done_file = _regular_file(
        done_path or parent / finalizer.DONE_NAME,
        context="finalizer done",
    )
    if done_file.parent != parent:
        raise GokuFullMotionSelect128Error(
            "finalizer done must be beside generation manifest"
        )
    done, done_raw = _json_object(done_file, context="finalizer done")
    if (
        done.get("schema_version") != finalizer.DONE_SCHEMA
        or done.get("status") != "complete"
    ):
        raise GokuFullMotionSelect128Error("finalizer done is not complete")
    stored_done_digest = _digest(
        done.get("done_digest"), context="finalizer done digest"
    )
    done_payload = dict(done)
    del done_payload["done_digest"]
    if _object_digest(done_payload) != stored_done_digest:
        raise GokuFullMotionSelect128Error("finalizer done digest differs")
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise GokuFullMotionSelect128Error("finalizer artifacts are malformed")
    if done.get("artifact_digest") != _object_digest(artifacts):
        raise GokuFullMotionSelect128Error(
            "finalizer aggregate artifact digest differs"
        )
    actual_entries = {entry.name for entry in parent.iterdir()}
    expected_entries = set(artifacts) | {done_file.name}
    if actual_entries != expected_entries:
        raise GokuFullMotionSelect128Error(
            "finalizer directory closure differs: "
            f"{sorted(actual_entries ^ expected_entries)}"
        )
    artifact_records: dict[str, dict[str, Any]] = {}
    for raw_name, raw_metadata in artifacts.items():
        name = _safe_basename(raw_name, context="finalizer artifact name")
        metadata = raw_metadata
        if not isinstance(metadata, Mapping):
            raise GokuFullMotionSelect128Error(
                f"finalizer artifact metadata is malformed: {name}"
            )
        path = _regular_file(parent / name, context=f"finalizer artifact {name}")
        raw = _stable_read(path, context=f"finalizer artifact {name}")
        expected_sha = _digest(
            metadata.get("sha256"), context=f"finalizer artifact {name} SHA"
        )
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise GokuFullMotionSelect128Error(
                f"finalizer artifact SHA differs: {name}"
            )
        if type(metadata.get("bytes")) is not int or metadata["bytes"] != len(raw):
            raise GokuFullMotionSelect128Error(
                f"finalizer artifact byte count differs: {name}"
            )
        artifact_records[name] = {
            "path": str(path),
            "sha256": expected_sha,
            "bytes": len(raw),
            "rows": metadata.get("rows"),
        }
    manifest_metadata = artifacts.get(manifest_path.name)
    if not isinstance(manifest_metadata, Mapping):
        raise GokuFullMotionSelect128Error(
            "finalizer done does not bind generation manifest"
        )
    if (
        manifest_metadata.get("sha256")
        != hashlib.sha256(manifest_raw).hexdigest()
        or manifest_metadata.get("bytes") != len(manifest_raw)
        or manifest_metadata.get("rows") != manifest_rows
    ):
        raise GokuFullMotionSelect128Error(
            "generation manifest differs from finalizer done"
        )
    return {
        "manifest": artifact_records[manifest_path.name],
        "done": {
            "path": str(done_file),
            "sha256": hashlib.sha256(done_raw).hexdigest(),
            "done_digest": stored_done_digest,
        },
        "artifacts": artifact_records,
        "artifact_digest": done["artifact_digest"],
        "input_digest": done.get("input_digest"),
        "implementation_digest": done.get("implementation_digest"),
    }


def load_generation_manifest(
    generation_manifest: str | Path,
    *,
    finalizer_done: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _regular_file(
        generation_manifest, context="primary generation manifest"
    )
    raw = _stable_read(path, context="primary generation manifest")
    raw_rows = _jsonl_rows_from_bytes(
        raw, context="primary generation manifest"
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_row in raw_rows:
        row = _validate_generation_row(raw_row)
        iid = _safe_iid(row.get("iid"), context="generation iid")
        if iid in seen:
            raise GokuFullMotionSelect128Error(
                f"duplicate generation IID: {iid}"
            )
        seen.add(iid)
        rows.append(row)
    closure = _validate_finalizer_parent(
        path,
        manifest_raw=raw,
        manifest_rows=len(rows),
        done_path=(Path(finalizer_done) if finalizer_done is not None else None),
    )
    closure["manifest"]["row_object_digest"] = _object_digest(rows)
    return rows, closure


_SHARD_DESCRIPTOR_KEYS = {
    "shard_index",
    "shard_id",
    "path",
    "root_row_start_zero_based",
    "root_row_end_exclusive",
    "root_row_indices_zero_based",
    "rows",
    "bytes",
    "sha256",
    "ordered_iids",
    "ordered_iids_sha256",
    "ordered_row_sha256",
}


def load_generation_shard_manifest(
    shard_manifest_dir: str | Path,
    *,
    generation_manifest_path: Path,
    generation_rows: Sequence[Mapping[str, Any]],
    finalizer_closure: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[Path, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Validate the exact 32x8 descriptor closure over ``primary_256``."""

    root = _regular_directory(
        shard_manifest_dir, context="generation shard-manifest directory"
    )
    expected_root_entries = {
        shard_manifest.SHARDS_NAME,
        shard_manifest.JOBS_NAME,
        shard_manifest.SUMMARY_NAME,
        shard_manifest.DONE_NAME,
    }
    if {entry.name for entry in root.iterdir()} != expected_root_entries:
        raise GokuFullMotionSelect128Error(
            "generation shard-manifest root closure differs"
        )
    shards_dir = _regular_directory(
        root / shard_manifest.SHARDS_NAME,
        context="generation shard directory",
    )
    expected_shard_names = {
        f"shard_{index:03d}.jsonl"
        for index in range(shard_manifest.SHARD_COUNT)
    }
    if {entry.name for entry in shards_dir.iterdir()} != expected_shard_names:
        raise GokuFullMotionSelect128Error(
            "generation shard file closure differs"
        )

    summary, summary_raw = _json_object(
        root / shard_manifest.SUMMARY_NAME,
        context="generation shard summary",
    )
    done, done_raw = _json_object(
        root / shard_manifest.DONE_NAME,
        context="generation shard done",
    )
    expected_done_keys = {
        "schema_version",
        "status",
        "policy_version",
        "implementation",
        "implementation_digest",
        "source",
        "input_digest",
        "artifacts",
        "artifact_digest",
        "done_digest",
    }
    if (
        set(done) != expected_done_keys
        or done.get("schema_version") != shard_manifest.DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("policy_version") != shard_manifest.POLICY_VERSION
    ):
        raise GokuFullMotionSelect128Error(
            "generation shard done schema/status differs"
        )
    done_payload = dict(done)
    stored_done_digest = _digest(
        done_payload.pop("done_digest", None),
        context="generation shard done digest",
    )
    if stored_done_digest != _object_digest(done_payload):
        raise GokuFullMotionSelect128Error(
            "generation shard done digest differs"
        )
    expected_summary_keys = {
        "schema_version",
        "status",
        "policy_version",
        "authorization_semantics",
        "source",
        "input_digest",
        "layout",
        "shards",
        "shards_digest",
        "jobs",
        "implementation",
        "implementation_digest",
    }
    if (
        set(summary) != expected_summary_keys
        or summary.get("schema_version") != shard_manifest.SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("policy_version") != shard_manifest.POLICY_VERSION
        or summary.get("source") != done.get("source")
        or summary.get("input_digest") != done.get("input_digest")
        or summary.get("implementation") != done.get("implementation")
        or summary.get("implementation_digest")
        != done.get("implementation_digest")
    ):
        raise GokuFullMotionSelect128Error(
            "generation shard summary/done binding differs"
        )
    implementation = done.get("implementation")
    if (
        not isinstance(implementation, Mapping)
        or dict(implementation) != shard_manifest._implementation_bundle()
        or done.get("implementation_digest") != _object_digest(implementation)
    ):
        raise GokuFullMotionSelect128Error(
            "generation shard implementation closure differs"
        )
    source = done.get("source")
    if not isinstance(source, Mapping) or done.get("input_digest") != _object_digest(
        source
    ):
        raise GokuFullMotionSelect128Error(
            "generation shard source closure differs"
        )
    primary_path = _regular_file(
        source.get("primary_path"), context="shard source primary manifest"
    )
    generation_path = _regular_file(
        generation_manifest_path, context="primary generation manifest"
    )
    primary_raw = _stable_read(
        generation_path, context="primary generation manifest shard binding"
    )
    primary_lines = primary_raw.splitlines(keepends=True)
    if (
        primary_path != generation_path
        or len(generation_rows) != shard_manifest.ROOT_ROWS
        or len(primary_lines) != shard_manifest.ROOT_ROWS
        or source.get("generation_schema") != finalizer.GENERATION_SCHEMA
        or source.get("primary_sha256")
        != hashlib.sha256(primary_raw).hexdigest()
        or source.get("primary_bytes") != len(primary_raw)
        or source.get("primary_rows") != len(generation_rows)
        or source.get("finalizer_dir") != str(generation_path.parent)
        or source.get("done_path") != finalizer_closure["done"]["path"]
        or source.get("done_sha256") != finalizer_closure["done"]["sha256"]
        or source.get("done_digest")
        != finalizer_closure["done"]["done_digest"]
        or source.get("implementation_digest")
        != finalizer_closure.get("implementation_digest")
        or source.get("input_digest") != finalizer_closure.get("input_digest")
        or source.get("artifact_digest")
        != finalizer_closure.get("artifact_digest")
    ):
        raise GokuFullMotionSelect128Error(
            "generation shard source does not close over primary_256"
        )
    root_iids = [str(row["iid"]) for row in generation_rows]
    root_row_digests = [_object_digest(dict(row)) for row in generation_rows]
    if (
        source.get("root_ordered_iids_sha256") != _ordered_digest(root_iids)
        or source.get("root_ordered_row_sha256")
        != _ordered_digest(root_row_digests)
    ):
        raise GokuFullMotionSelect128Error(
            "generation shard root ordering digest differs"
        )
    layout = summary.get("layout")
    if layout != {
        "strategy": "ordered_contiguous_fixed_size",
        "root_rows": shard_manifest.ROOT_ROWS,
        "rows_per_shard": shard_manifest.ROWS_PER_SHARD,
        "shard_count": shard_manifest.SHARD_COUNT,
        "row_index_basis": "zero_based_end_exclusive",
        "complete_nonoverlapping_coverage": True,
    }:
        raise GokuFullMotionSelect128Error(
            "generation shard layout differs from exact 32x8"
        )
    authorization = summary.get("authorization_semantics")
    if authorization != {
        "root_manifest_is_authorization_object": True,
        "shards_are_contiguous_byte_exact_slices": True,
        "descriptor_grants_authorization": False,
        "signed_release_must_independently_verify_contiguity": True,
    }:
        raise GokuFullMotionSelect128Error(
            "generation shard authorization semantics differ"
        )
    descriptors_value = summary.get("shards")
    if (
        not isinstance(descriptors_value, list)
        or len(descriptors_value) != shard_manifest.SHARD_COUNT
        or summary.get("shards_digest") != _object_digest(descriptors_value)
    ):
        raise GokuFullMotionSelect128Error(
            "generation shard descriptors differ"
        )
    descriptors: list[dict[str, Any]] = []
    by_iid: dict[str, dict[str, Any]] = {}
    by_path: dict[Path, dict[str, Any]] = {}
    reconstructed: list[bytes] = []
    for shard_index, raw_descriptor in enumerate(descriptors_value):
        if not isinstance(raw_descriptor, Mapping):
            raise GokuFullMotionSelect128Error(
                f"generation shard descriptor {shard_index} is malformed"
            )
        descriptor = dict(raw_descriptor)
        if set(descriptor) != _SHARD_DESCRIPTOR_KEYS:
            raise GokuFullMotionSelect128Error(
                f"generation shard descriptor {shard_index} schema differs"
            )
        start = shard_index * shard_manifest.ROWS_PER_SHARD
        end = start + shard_manifest.ROWS_PER_SHARD
        relative = f"shards/shard_{shard_index:03d}.jsonl"
        manifest_path = _regular_file(
            root / relative,
            context=f"generation shard manifest {shard_index}",
        )
        raw = _stable_read(
            manifest_path, context=f"generation shard manifest {shard_index}"
        )
        expected_raw = b"".join(primary_lines[start:end])
        expected_rows = [dict(row) for row in generation_rows[start:end]]
        parsed_rows = _jsonl_rows_from_bytes(
            raw, context=f"generation shard manifest {shard_index}"
        )
        expected_iids = root_iids[start:end]
        if (
            descriptor.get("shard_index") != shard_index
            or descriptor.get("shard_id") != f"shard_{shard_index:03d}"
            or descriptor.get("path") != relative
            or descriptor.get("root_row_start_zero_based") != start
            or descriptor.get("root_row_end_exclusive") != end
            or descriptor.get("root_row_indices_zero_based")
            != list(range(start, end))
            or descriptor.get("rows") != shard_manifest.ROWS_PER_SHARD
            or descriptor.get("bytes") != len(raw)
            or descriptor.get("sha256") != hashlib.sha256(raw).hexdigest()
            or descriptor.get("ordered_iids") != expected_iids
            or descriptor.get("ordered_iids_sha256")
            != _ordered_digest(expected_iids)
            or descriptor.get("ordered_row_sha256")
            != _ordered_digest(root_row_digests[start:end])
            or raw != expected_raw
            or parsed_rows != expected_rows
        ):
            raise GokuFullMotionSelect128Error(
                f"generation shard {shard_index} does not match root slice"
            )
        binding = {
            "descriptor": descriptor,
            "descriptor_digest": _object_digest(descriptor),
            "manifest_path": manifest_path,
            "manifest_sha256": descriptor["sha256"],
            "manifest_bytes": descriptor["bytes"],
            "rows": expected_rows,
        }
        descriptors.append(descriptor)
        by_path[manifest_path] = binding
        for shard_row_index, (iid, root_index) in enumerate(
            zip(expected_iids, range(start, end), strict=True)
        ):
            if iid in by_iid:
                raise GokuFullMotionSelect128Error(
                    f"generation shard IID is duplicated: {iid}"
                )
            by_iid[iid] = {
                **binding,
                "shard_index": shard_index,
                "shard_row_index": shard_row_index,
                "root_row_index": root_index,
            }
        reconstructed.append(raw)
    if b"".join(reconstructed) != primary_raw or set(by_iid) != set(root_iids):
        raise GokuFullMotionSelect128Error(
            "generation shard set does not reconstruct primary_256"
        )

    jobs_path = _regular_file(
        root / shard_manifest.JOBS_NAME, context="generation shard jobs.tsv"
    )
    jobs_raw = _stable_read(jobs_path, context="generation shard jobs.tsv")
    if jobs_raw != shard_manifest._jobs_bytes(descriptors):
        raise GokuFullMotionSelect128Error(
            "generation shard jobs.tsv differs from descriptors"
        )
    artifacts = done.get("artifacts")
    expected_artifact_names = {
        shard_manifest.JOBS_NAME,
        shard_manifest.SUMMARY_NAME,
        *(
            f"shards/shard_{index:03d}.jsonl"
            for index in range(shard_manifest.SHARD_COUNT)
        ),
    }
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != expected_artifact_names
        or done.get("artifact_digest") != _object_digest(artifacts)
    ):
        raise GokuFullMotionSelect128Error(
            "generation shard artifact closure differs"
        )
    raw_by_name = {
        shard_manifest.JOBS_NAME: jobs_raw,
        shard_manifest.SUMMARY_NAME: summary_raw,
        **{
            descriptor["path"]: reconstructed[index]
            for index, descriptor in enumerate(descriptors)
        },
    }
    for name, raw in raw_by_name.items():
        metadata = artifacts.get(name)
        expected_rows = (
            shard_manifest.SHARD_COUNT
            if name == shard_manifest.JOBS_NAME
            else 1
            if name == shard_manifest.SUMMARY_NAME
            else shard_manifest.ROWS_PER_SHARD
        )
        if metadata != {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": expected_rows,
        }:
            raise GokuFullMotionSelect128Error(
                f"generation shard artifact metadata differs: {name}"
            )
    if summary.get("jobs") != {
        "path": shard_manifest.JOBS_NAME,
        "sha256": hashlib.sha256(jobs_raw).hexdigest(),
        "bytes": len(jobs_raw),
        "rows_excluding_header": shard_manifest.SHARD_COUNT,
    }:
        raise GokuFullMotionSelect128Error(
            "generation shard jobs binding differs"
        )
    closure = {
        "root": str(root),
        "summary": {
            "path": str(root / shard_manifest.SUMMARY_NAME),
            "sha256": hashlib.sha256(summary_raw).hexdigest(),
        },
        "done": {
            "path": str(root / shard_manifest.DONE_NAME),
            "sha256": hashlib.sha256(done_raw).hexdigest(),
            "done_digest": stored_done_digest,
        },
        "input_digest": done["input_digest"],
        "implementation_digest": done["implementation_digest"],
        "artifact_digest": done["artifact_digest"],
        "shards_digest": summary["shards_digest"],
    }
    return by_iid, by_path, descriptors, closure


def _load_wan_runs(
    roots: Sequence[str | Path],
    *,
    generated_manifests: Sequence[str | Path] | None,
    generation_shards_by_path: Mapping[Path, Mapping[str, Any]],
    generation_shards_by_iid: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not roots:
        raise GokuFullMotionSelect128Error("at least one Wan run root is required")
    if generated_manifests is not None and len(generated_manifests) != len(roots):
        raise GokuFullMotionSelect128Error(
            "wan_generated_manifests must correspond one-to-one with roots"
        )
    if len(roots) != shard_manifest.SHARD_COUNT:
        raise GokuFullMotionSelect128Error(
            "Wan run-root count must close the exact 32x8 topology"
        )
    by_iid: dict[str, dict[str, Any]] = {}
    closures: list[dict[str, Any]] = []
    used_shards: set[int] = set()
    for index, raw_root in enumerate(roots):
        root = _regular_directory(raw_root, context=f"Wan run root {index}")
        contract_path = _regular_file(
            root / "run_contract.json", context=f"Wan run contract {index}"
        )
        contract_preflight, _ = _json_object(
            contract_path, context=f"Wan run contract {index}"
        )
        manifest_binding = contract_preflight.get("manifest")
        if not isinstance(manifest_binding, Mapping):
            raise GokuFullMotionSelect128Error(
                f"Wan run contract {index} manifest binding is malformed"
            )
        bound_shard_path = _regular_file(
            manifest_binding.get("path"),
            context=f"Wan run contract {index} shard manifest",
        )
        generation_shard = generation_shards_by_path.get(bound_shard_path)
        if generation_shard is None:
            raise GokuFullMotionSelect128Error(
                f"Wan run contract {index} binds an unknown shard manifest"
            )
        descriptor = generation_shard["descriptor"]
        shard_index = int(descriptor["shard_index"])
        if shard_index in used_shards:
            raise GokuFullMotionSelect128Error(
                f"multiple Wan roots bind generation shard {shard_index}"
            )
        if (
            manifest_binding.get("sha256")
            != generation_shard["manifest_sha256"]
            or manifest_binding.get("bytes")
            != generation_shard["manifest_bytes"]
            or manifest_binding.get("row_count") != shard_manifest.ROWS_PER_SHARD
        ):
            raise GokuFullMotionSelect128Error(
                f"Wan run contract shard binding differs for shard {shard_index}"
            )
        run_contract, run_contract_sha = postcheck._validate_run_contract(
            root,
            manifest_path=bound_shard_path,
            manifest_sha256=generation_shard["manifest_sha256"],
            manifest_rows=shard_manifest.ROWS_PER_SHARD,
        )
        generated_path = _regular_file(
            (
                generated_manifests[index]
                if generated_manifests is not None
                else root / "generated_manifest.jsonl"
            ),
            context=f"Wan generated manifest {index}",
        )
        generated_rows, generated_sha, complete, complete_sha = (
            postcheck._validate_generated_manifest(
                root,
                generated_manifest_path=generated_path,
                generation_rows=generation_shard["rows"],
                input_manifest_sha256=generation_shard["manifest_sha256"],
                run_contract=run_contract,
            )
        )
        complete_path = root / "run_complete.json"
        contract_raw = _stable_read(
            contract_path, context=f"Wan run contract {index} closure"
        )
        generated_raw = _stable_read(
            generated_path, context=f"Wan generated manifest {index} closure"
        )
        complete_raw = _stable_read(
            complete_path, context=f"Wan run completion {index} closure"
        )
        if (
            hashlib.sha256(contract_raw).hexdigest() != run_contract_sha
            or hashlib.sha256(generated_raw).hexdigest() != generated_sha
            or hashlib.sha256(complete_raw).hexdigest() != complete_sha
        ):
            raise GokuFullMotionSelect128Error(
                f"Wan shard {index} changed during validation"
            )
        closure = {
            "root": str(root),
            "generation_shard": {
                "shard_index": shard_index,
                "manifest_path": str(bound_shard_path),
                "manifest_sha256": generation_shard["manifest_sha256"],
                "manifest_bytes": generation_shard["manifest_bytes"],
                "descriptor_digest": generation_shard["descriptor_digest"],
                "ordered_iids": list(descriptor["ordered_iids"]),
                "root_row_indices_zero_based": list(
                    descriptor["root_row_indices_zero_based"]
                ),
            },
            "run_contract": {
                "path": str(contract_path),
                "sha256": run_contract_sha,
                "bytes": len(contract_raw),
                "contract_digest": run_contract["contract_digest"],
            },
            "generated_manifest": {
                "path": str(generated_path),
                "sha256": generated_sha,
                "bytes": len(generated_raw),
                "rows": len(generated_rows),
            },
            "run_complete": {
                "path": str(complete_path),
                "sha256": complete_sha,
                "bytes": len(complete_raw),
                "complete_digest": complete["complete_digest"],
            },
        }
        closures.append(closure)
        generated_iids = [str(row.get("iid")) for row in generated_rows]
        if generated_iids != descriptor["ordered_iids"]:
            raise GokuFullMotionSelect128Error(
                f"Wan generated IID order differs for shard {shard_index}"
            )
        for generated_row in generated_rows:
            iid = _safe_iid(generated_row.get("iid"), context="generated iid")
            if iid in by_iid:
                raise GokuFullMotionSelect128Error(
                    f"IID generated by multiple Wan roots: {iid}"
                )
            iid_shard = generation_shards_by_iid.get(iid)
            if iid_shard is None or iid_shard["shard_index"] != shard_index:
                raise GokuFullMotionSelect128Error(
                    f"Wan generated IID/root-index binding differs: {iid}"
                )
            by_iid[iid] = {
                "generated_row": dict(generated_row),
                "root": root,
                "run_contract": run_contract,
                "run_contract_sha256": run_contract_sha,
                "closure": closure,
                "generation_shard": iid_shard,
            }
        used_shards.add(shard_index)
    if used_shards != set(range(shard_manifest.SHARD_COUNT)) or set(by_iid) != set(
        generation_shards_by_iid
    ):
        raise GokuFullMotionSelect128Error(
            "Wan run roots do not close all 32 generation shards"
        )
    closures.sort(
        key=lambda value: int(value["generation_shard"]["shard_index"])
    )
    return by_iid, closures


def _postcheck_config_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {
        "schema_version",
        "status",
        "assigned_iids",
        "output",
        "receipt_digest",
    }
    return {key: value for key, value in receipt.items() if key not in reserved}


def _validate_postcheck_semantics(
    record: Mapping[str, Any],
    *,
    generation_row: Mapping[str, Any],
    generation_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if record.get("schema_version") != _POSTCHECK_SCHEMA_V6:
        raise GokuFullMotionSelect128Error(
            f"postcheck v6 schema differs for iid={record.get('iid')}"
        )
    try:
        normalized = postcheck._normalize_contract(
            generation_row,
            manifest_root=generation_manifest_path.parent,
        )
    except Exception as error:
        raise GokuFullMotionSelect128Error(
            "postcheck generation/Qwen-record closure differs for "
            f"iid={record.get('iid')}: {error}"
        ) from error
    digest_fields = {
        "change_region_proposals_digest": "change_region_proposals_digest",
        "coverage_authority_inventory_digest": (
            "coverage_authority_inventory_digest"
        ),
        "coverage_authority_assignments_digest": (
            "coverage_authority_assignments_digest"
        ),
        "coverage_authority_digest": "coverage_authority_digest",
        "coverage_authority_alignment_digest": (
            "coverage_authority_alignment_digest"
        ),
        "source_census_digest": "source_census_digest",
        "target_plan_digest": "target_plan_digest",
        "motion_spec_digest": "motion_spec_digest",
        "compiled_instruction_digest": "compiled_instruction_digest",
        "coverage_critic_digest": "coverage_critic_digest",
        "full_motion_contract_digest": "full_motion_contract_digest",
        "qwen_result_digest": "qwen_result_digest",
        "qwen_provenance_digest": "qwen_provenance_digest",
        "qwen_record_payload_sha256": "qwen_record_payload_sha256",
        "qwen_evidence_binding": "qwen_evidence_binding",
        "instruction_sha256": "instruction_sha256",
    }
    for record_field, normalized_field in digest_fields.items():
        if record.get(record_field) != normalized.get(normalized_field):
            raise GokuFullMotionSelect128Error(
                f"postcheck {record_field} differs for iid={record.get('iid')}"
            )
    binding = normalized.get("qwen_evidence_binding")
    generation_evidence = generation_row.get("qwen_evidence")
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema_version")
        != _POSTCHECK_QWEN_BINDING_SCHEMA_V6
        or binding.get("record_schema_version") != _QWEN_RECORD_SCHEMA_V6
        or binding.get("hard_gate_schema_version")
        != _QWEN_HARD_GATE_SCHEMA_V6
        or not isinstance(generation_evidence, Mapping)
        or binding.get("change_region_proposals_digest")
        != normalized.get("change_region_proposals_digest")
        or binding.get("change_region_proposals_digest")
        != generation_evidence.get("change_region_proposals_digest")
        or binding.get("coverage_authority_inventory_digest")
        != normalized.get("coverage_authority_inventory_digest")
        or binding.get("coverage_authority_inventory_digest")
        != generation_evidence.get("coverage_authority_inventory_digest")
        or binding.get("coverage_authority_assignments_digest")
        != normalized.get("coverage_authority_assignments_digest")
        or binding.get("coverage_authority_assignments_digest")
        != generation_evidence.get("coverage_authority_assignments_digest")
        or binding.get("coverage_authority_digest")
        != normalized.get("coverage_authority_digest")
        or binding.get("coverage_authority_digest")
        != generation_evidence.get("coverage_authority_digest")
        or binding.get("coverage_authority_alignment_digest")
        != normalized.get("coverage_authority_alignment_digest")
        or binding.get("coverage_authority_alignment_digest")
        != generation_evidence.get("coverage_authority_alignment_digest")
        or binding.get("i0_grounding_digest")
        != normalized.get("i0_grounding_digest")
        or binding.get("i0_grounding_digest")
        != generation_evidence.get("i0_grounding_digest")
        or binding.get("qwen_record_payload_sha256")
        != normalized.get("qwen_record_payload_sha256")
        or binding.get("qwen_record_payload_sha256")
        != _object_digest(generation_evidence.get("qwen_record_payload"))
    ):
        raise GokuFullMotionSelect128Error(
            f"postcheck Qwen-v6 A0/exact-I0 binding differs for iid={record.get('iid')}"
        )
    census = postcheck.validate_target_census(
        record.get("target_census")
    )
    judgment = postcheck.validate_clause_judgment(
        record.get("clause_judgment"),
        expected_dynamic_units=normalized["dynamic_units"],
        expected_static_ids=normalized["static_ids"],
    )
    expected_aggregate = postcheck.aggregate_postcheck(
        census,
        judgment,
        expected_contract=postcheck._expected_judge_contract(normalized),
    )
    if record.get("aggregate") != expected_aggregate:
        raise GokuFullMotionSelect128Error(
            f"postcheck aggregate differs for iid={record.get('iid')}"
        )
    if (
        record.get("decision") != expected_aggregate["decision"]
        or record.get("eligible") is not expected_aggregate["eligible"]
    ):
        raise GokuFullMotionSelect128Error(
            f"postcheck top-level decision differs for iid={record.get('iid')}"
        )
    return normalized, expected_aggregate


def _load_postcheck_shards(
    outputs: Sequence[str | Path],
    *,
    receipts: Sequence[str | Path] | None,
    generation_rows_by_iid: Mapping[str, Mapping[str, Any]],
    generation_shards_by_iid: Mapping[str, Mapping[str, Any]],
    generation_shards_by_path: Mapping[Path, Mapping[str, Any]],
    wan_by_iid: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not outputs:
        raise GokuFullMotionSelect128Error(
            "at least one postcheck shard output is required"
        )
    if receipts is not None and len(receipts) != len(outputs):
        raise GokuFullMotionSelect128Error(
            "postcheck receipts must correspond one-to-one with outputs"
        )
    if len(outputs) != shard_manifest.SHARD_COUNT:
        raise GokuFullMotionSelect128Error(
            "postcheck output count must close the exact 32x8 topology"
        )
    by_iid: dict[str, dict[str, Any]] = {}
    closures: list[dict[str, Any]] = []
    used_shards: set[int] = set()
    for index, raw_output in enumerate(outputs):
        output = _regular_file(
            raw_output, context=f"postcheck output {index}"
        )
        receipt_path = _regular_file(
            (
                receipts[index]
                if receipts is not None
                else postcheck.shard_receipt_path(output)
            ),
            context=f"postcheck receipt {index}",
        )
        receipt, receipt_raw = _json_object(
            receipt_path, context=f"postcheck receipt {index}"
        )
        assigned = receipt.get("assigned_iids")
        if not isinstance(assigned, list) or any(
            type(iid) is not str for iid in assigned
        ):
            raise GokuFullMotionSelect128Error(
                f"postcheck receipt {index} assigned_iids is malformed"
            )
        config = _postcheck_config_binding(receipt)
        postcheck.validate_shard_receipt(
            receipt,
            output=output,
            assigned_iids=assigned,
            config_binding=config,
        )
        bound_manifest = _regular_file(
            config.get("manifest"),
            context=f"postcheck shard {index} bound manifest",
        )
        generation_shard = generation_shards_by_path.get(bound_manifest)
        if generation_shard is None:
            raise GokuFullMotionSelect128Error(
                "postcheck shard binds an unknown generation shard manifest"
            )
        descriptor = generation_shard["descriptor"]
        shard_index = int(descriptor["shard_index"])
        if shard_index in used_shards:
            raise GokuFullMotionSelect128Error(
                f"multiple postcheck outputs bind shard {shard_index}"
            )
        if (
            config.get("manifest_sha256")
            != generation_shard["manifest_sha256"]
            or assigned != descriptor["ordered_iids"]
        ):
            raise GokuFullMotionSelect128Error(
                f"postcheck generation-shard binding differs: {shard_index}"
            )
        shard_wan = [wan_by_iid[iid] for iid in assigned if iid in wan_by_iid]
        if len(shard_wan) != shard_manifest.ROWS_PER_SHARD or any(
            item["generation_shard"]["shard_index"] != shard_index
            for item in shard_wan
        ):
            raise GokuFullMotionSelect128Error(
                f"postcheck/Wan IID topology differs for shard {shard_index}"
            )
        expected_wan = shard_wan[0]
        if any(item["root"] != expected_wan["root"] for item in shard_wan):
            raise GokuFullMotionSelect128Error(
                f"postcheck shard spans multiple Wan roots: {shard_index}"
            )
        if (
            _regular_directory(
                config.get("generation_root"),
                context=f"postcheck shard {shard_index} generation root",
            )
            != expected_wan["root"]
            or config.get("run_contract_sha256")
            != expected_wan["run_contract_sha256"]
            or _regular_file(
                config.get("generated_manifest"),
                context=f"postcheck shard {shard_index} generated manifest",
            )
            != Path(
                expected_wan["closure"]["generated_manifest"]["path"]
            )
            or config.get("generated_manifest_sha256")
            != expected_wan["closure"]["generated_manifest"]["sha256"]
            or config.get("run_complete_sha256")
            != expected_wan["closure"]["run_complete"]["sha256"]
        ):
            raise GokuFullMotionSelect128Error(
                f"postcheck/Wan byte binding differs for shard {shard_index}"
            )
        output_raw = _stable_read(output, context=f"postcheck output {index}")
        rows = _jsonl_rows_from_bytes(
            output_raw,
            context=f"postcheck output {index}",
            allow_empty=True,
        )
        if [row.get("iid") for row in rows] != assigned:
            raise GokuFullMotionSelect128Error(
                f"postcheck output/receipt IID order differs at shard {index}"
            )
        closure = {
            "generation_shard": {
                "shard_index": shard_index,
                "manifest_path": str(bound_manifest),
                "manifest_sha256": generation_shard["manifest_sha256"],
                "descriptor_digest": generation_shard["descriptor_digest"],
                "ordered_iids": list(descriptor["ordered_iids"]),
                "root_row_indices_zero_based": list(
                    descriptor["root_row_indices_zero_based"]
                ),
            },
            "output": {
                "path": str(output),
                "sha256": hashlib.sha256(output_raw).hexdigest(),
                "bytes": len(output_raw),
                "rows": len(rows),
            },
            "receipt": {
                "path": str(receipt_path),
                "sha256": hashlib.sha256(receipt_raw).hexdigest(),
                "bytes": len(receipt_raw),
                "receipt_digest": receipt["receipt_digest"],
            },
            "config_digest": config.get("config_digest"),
            "assigned_iids": list(assigned),
        }
        closures.append(closure)
        for row in rows:
            iid = _safe_iid(row.get("iid"), context="postcheck iid")
            if iid in by_iid:
                raise GokuFullMotionSelect128Error(
                    f"duplicate postcheck IID across shards: {iid}"
                )
            generation_row = generation_rows_by_iid.get(iid)
            if generation_row is None:
                raise GokuFullMotionSelect128Error(
                    f"postcheck IID is absent from generation manifest: {iid}"
                )
            postcheck._validate_output_record(
                row,
                expected_row=generation_row,
                config_binding=config,
            )
            if row.get("status") != "ok":
                raise GokuFullMotionSelect128Error(
                    f"terminal postcheck shard contains error iid={iid}"
                )
            normalized, aggregate = _validate_postcheck_semantics(
                row,
                generation_row=generation_row,
                generation_manifest_path=bound_manifest,
            )
            iid_shard = generation_shards_by_iid.get(iid)
            if iid_shard is None or iid_shard["shard_index"] != shard_index:
                raise GokuFullMotionSelect128Error(
                    f"postcheck IID/root-index binding differs: {iid}"
                )
            by_iid[iid] = {
                "record": dict(row),
                "normalized": normalized,
                "aggregate": aggregate,
                "closure": closure,
                "generation_shard": iid_shard,
            }
        if (
            _stable_read(output, context=f"postcheck output {index} closure")
            != output_raw
            or _stable_read(
                receipt_path, context=f"postcheck receipt {index} closure"
            )
            != receipt_raw
        ):
            raise GokuFullMotionSelect128Error(
                f"postcheck shard {index} changed during validation"
            )
        used_shards.add(shard_index)
    if used_shards != set(range(shard_manifest.SHARD_COUNT)) or set(by_iid) != set(
        generation_shards_by_iid
    ):
        raise GokuFullMotionSelect128Error(
            "postcheck outputs do not close all 32 generation shards"
        )
    closures.sort(
        key=lambda value: int(value["generation_shard"]["shard_index"])
    )
    return by_iid, closures


def select_exact_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    exact_size: int = DEFAULT_EXACT_SIZE,
    min_multi_unit: int = DEFAULT_MIN_MULTI_UNIT,
) -> list[dict[str, Any]]:
    """Return exact deterministic selection, preferring 3 then 2 units."""

    if type(exact_size) is not int or exact_size <= 0:
        raise GokuFullMotionSelect128Error("exact_size must be positive")
    if (
        type(min_multi_unit) is not int
        or min_multi_unit < 0
        or min_multi_unit > exact_size
    ):
        raise GokuFullMotionSelect128Error("min_multi_unit is invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in candidates:
        candidate = dict(raw)
        iid = _safe_iid(candidate.get("iid"), context="candidate iid")
        if iid in seen:
            raise GokuFullMotionSelect128Error(
                f"duplicate selection candidate: {iid}"
            )
        seen.add(iid)
        unit_count = candidate.get("dynamic_unit_count")
        primary_index = candidate.get("primary_index")
        if type(unit_count) is not int or not 1 <= unit_count <= 3:
            raise GokuFullMotionSelect128Error(
                f"candidate dynamic_unit_count is invalid: {iid}"
            )
        if type(primary_index) is not int or primary_index < 0:
            raise GokuFullMotionSelect128Error(
                f"candidate primary_index is invalid: {iid}"
            )
        normalized.append(candidate)
    multi_available = sum(
        candidate["dynamic_unit_count"] >= 2 for candidate in normalized
    )
    if multi_available < min_multi_unit:
        raise GokuFullMotionSelect128Error(
            "insufficient multi-unit postcheck passes: "
            f"required={min_multi_unit} available={multi_available}"
        )
    if len(normalized) < exact_size:
        raise GokuFullMotionSelect128Error(
            "insufficient postcheck passes for exact dataset: "
            f"required={exact_size} available={len(normalized)}"
        )
    ordered = sorted(
        normalized,
        key=lambda item: (
            -int(item["dynamic_unit_count"]),
            int(item["primary_index"]),
            str(item["iid"]),
        ),
    )
    selected = ordered[:exact_size]
    selected_multi = sum(
        candidate["dynamic_unit_count"] >= 2 for candidate in selected
    )
    if selected_multi < min_multi_unit:
        raise GokuFullMotionSelect128Error(
            "deterministic exact selection missed multi-unit quota"
        )
    return selected


def _artifact_record(
    *,
    final_path: Path,
    relative_path: Path,
    sha256: str,
    byte_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "path": str(final_path),
        "relative_path": relative_path.as_posix(),
        "sha256": _digest(sha256, context="materialized artifact SHA"),
        "bytes": byte_count,
    }


def _copy_verified(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    context: str,
) -> tuple[str, int]:
    """Copy one regular file while binding the open inode before and after."""

    expected = _digest(expected_sha256, context=f"{context} expected SHA")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, flags)
    except OSError as error:
        raise GokuFullMotionSelect128Error(
            f"cannot safely open {context}: {source}"
        ) from error
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise GokuFullMotionSelect128Error(
                f"{context} is not a regular file: {source}"
            )
        with destination.open("xb") as output:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(source_fd)
    finally:
        os.close(source_fd)
    path_after = source.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    observed = digest.hexdigest()
    if (
        identity_before != identity_after
        or identity_after != path_identity
        or byte_count != before.st_size
        or observed != expected
    ):
        raise GokuFullMotionSelect128Error(
            f"{context} changed or differs from expected SHA: {source}"
        )
    return observed, byte_count


def _write_new(path: Path, raw: bytes) -> tuple[str, int]:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    else:  # pragma: no cover - production and CI use Linux/macOS.
        raise GokuFullMotionSelect128Error(
            "platform lacks atomic no-replace directory rename"
        )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(output)
    raise OSError(number, os.strerror(number), str(output))


def _conditioning_sources(candidate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    generated = candidate["wan"]["generated_row"]
    root = candidate["wan"]["root"]
    iid = str(candidate["iid"])
    fields = {
        key
        for key, value in generated.items()
        if key.startswith("conditioning_")
        and not key.endswith("_sha256")
        and type(value) is str
    }
    missing = _REQUIRED_CONDITIONING_FIELDS - fields
    if missing:
        raise GokuFullMotionSelect128Error(
            f"Wan generated row lacks conditioning artifacts for {iid}: "
            f"{sorted(missing)}"
        )
    sources: dict[str, dict[str, Any]] = {}
    expected_parent = (root / "samples" / iid).resolve(strict=True)
    for field in sorted(fields):
        digest_field = f"{field}_sha256"
        expected_sha = _digest(
            generated.get(digest_field),
            context=f"{iid} {digest_field}",
        )
        path = _resolve_file(
            generated[field], root, context=f"{iid} {field}"
        )
        if path.parent != expected_parent:
            raise GokuFullMotionSelect128Error(
                f"conditioning artifact escapes sample directory: {path}"
            )
        name = _safe_basename(path.name, context=f"{iid} conditioning basename")
        if name in sources:
            raise GokuFullMotionSelect128Error(
                f"duplicate conditioning basename for {iid}: {name}"
            )
        if _file_digest(path, context=f"{iid} {field}") != expected_sha:
            raise GokuFullMotionSelect128Error(
                f"conditioning artifact SHA differs: {iid} {field}"
            )
        sources[name] = {
            "field": field,
            "path": path,
            "sha256": expected_sha,
        }
    return sources


def _materialize_sample(
    candidate: Mapping[str, Any],
    *,
    selection_rank: int,
    staging_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    iid = str(candidate["iid"])
    row = candidate["generation_row"]
    record = candidate["postcheck"]["record"]
    media = candidate["verified_media"]
    sample_relative = Path("samples") / iid
    staging_sample = staging_root / sample_relative
    final_sample = final_root / sample_relative
    staging_sample.mkdir(parents=False, exist_ok=False)

    artifacts: dict[str, Any] = {}

    def copy_role(
        role: str,
        source: Path,
        filename: str,
        expected_sha: str,
    ) -> None:
        relative = sample_relative / filename
        observed, size = _copy_verified(
            source,
            staging_root / relative,
            expected_sha256=expected_sha,
            context=f"iid={iid} {role}",
        )
        artifacts[role] = _artifact_record(
            final_path=final_root / relative,
            relative_path=relative,
            sha256=observed,
            byte_count=size,
        )

    source_binding = media.get("source")
    target_binding = media.get("target")
    result_binding = media.get("sample_result")
    if not all(
        isinstance(value, Mapping)
        for value in (source_binding, target_binding, result_binding)
    ):
        raise GokuFullMotionSelect128Error(
            f"verified media binding is incomplete for {iid}"
        )
    copy_role(
        "source",
        _regular_file(source_binding["path"], context=f"{iid} source"),
        "source.mp4",
        source_binding["sha256"],
    )
    copy_role(
        "target",
        _regular_file(target_binding["path"], context=f"{iid} target"),
        "target.mp4",
        target_binding["sha256"],
    )
    copy_role(
        "result",
        _regular_file(result_binding["path"], context=f"{iid} result"),
        "result.json",
        result_binding["sha256"],
    )

    instruction_raw = str(row["edit_instruction"]).encode("utf-8")
    instruction_relative = sample_relative / "edit_instruction.txt"
    instruction_sha, instruction_bytes = _write_new(
        staging_root / instruction_relative, instruction_raw
    )
    if instruction_sha != row["edit_instruction_sha256"]:
        raise GokuFullMotionSelect128Error(
            f"instruction bytes differ for {iid}"
        )
    artifacts["edit_instruction"] = _artifact_record(
        final_path=final_root / instruction_relative,
        relative_path=instruction_relative,
        sha256=instruction_sha,
        byte_count=instruction_bytes,
    )

    generated_objects = {
        "generation_row": (
            sample_relative / "generation_row.json",
            _pretty_bytes(row),
        ),
        "motion_spec": (
            sample_relative / "motion_spec.json",
            _pretty_bytes(row["motion_spec"]),
        ),
        "qwen_record_payload": (
            sample_relative / "qwen_record_payload.json",
            _pretty_bytes(row["qwen_evidence"]["qwen_record_payload"]),
        ),
        "postcheck": (
            sample_relative / "postcheck.json",
            _pretty_bytes(record),
        ),
    }
    for role, (relative, raw) in generated_objects.items():
        sha, size = _write_new(staging_root / relative, raw)
        artifacts[role] = _artifact_record(
            final_path=final_root / relative,
            relative_path=relative,
            sha256=sha,
            byte_count=size,
        )

    conditioning: dict[str, dict[str, Any]] = {}
    for name, source in _conditioning_sources(candidate).items():
        if name in {
            "source.mp4",
            "target.mp4",
            "edit_instruction.txt",
            "generation_row.json",
            "motion_spec.json",
            "qwen_record_payload.json",
            "postcheck.json",
            "result.json",
        }:
            raise GokuFullMotionSelect128Error(
                f"conditioning basename collides for {iid}: {name}"
            )
        relative = sample_relative / name
        observed, size = _copy_verified(
            source["path"],
            staging_root / relative,
            expected_sha256=source["sha256"],
            context=f"iid={iid} conditioning {name}",
        )
        conditioning[source["field"]] = _artifact_record(
            final_path=final_root / relative,
            relative_path=relative,
            sha256=observed,
            byte_count=size,
        )
    artifacts["conditioning"] = dict(sorted(conditioning.items()))
    _fsync_directory(staging_sample)

    normalized = candidate["postcheck"]["normalized"]
    finalization = row["full_motion_finalization"]
    generation_shard = candidate["generation_shard"]
    dataset_row: dict[str, Any] = {
        "schema_version": DATASET_ROW_SCHEMA,
        "iid": iid,
        "selection_rank": selection_rank,
        "primary_index": candidate["primary_index"],
        "group_id": row["group_id"],
        "family": row["family"],
        "dynamic_unit_count": candidate["dynamic_unit_count"],
        "multi_unit": candidate["dynamic_unit_count"] >= 2,
        "target_action_signatures": list(
            finalization["target_action_signatures"]
        ),
        "sample_dir": str(final_sample),
        "edit_instruction_sha256": row["edit_instruction_sha256"],
        "motion_spec_object_sha256": row["motion_spec_sha256"],
        "change_region_proposals_sha256": normalized[
            "change_region_proposals_digest"
        ],
        "coverage_authority_inventory_sha256": normalized[
            "coverage_authority_inventory_digest"
        ],
        "coverage_authority_assignments_sha256": normalized[
            "coverage_authority_assignments_digest"
        ],
        "coverage_authority_sha256": normalized[
            "coverage_authority_digest"
        ],
        "coverage_authority_alignment_sha256": normalized[
            "coverage_authority_alignment_digest"
        ],
        "i0_grounding_sha256": normalized["i0_grounding_digest"],
        "qwen_evidence_sha256": _object_digest(row["qwen_evidence"]),
        "qwen_record_payload_sha256": normalized[
            "qwen_record_payload_sha256"
        ],
        "qwen_hard_gate_sha256": _object_digest(
            row["qwen_evidence"]["hard_gate"]
        ),
        "postcheck_result_digest": record["result_digest"],
        "generation_binding": {
            "manifest": candidate["generation_manifest"]["path"],
            "manifest_sha256": candidate["generation_manifest"]["sha256"],
            "row_digest": _object_digest(row),
            "motion_spec_sha256": normalized["motion_spec_digest"],
            "change_region_proposals_sha256": normalized[
                "change_region_proposals_digest"
            ],
            "coverage_authority_inventory_sha256": normalized[
                "coverage_authority_inventory_digest"
            ],
            "coverage_authority_assignments_sha256": normalized[
                "coverage_authority_assignments_digest"
            ],
            "coverage_authority_sha256": normalized[
                "coverage_authority_digest"
            ],
            "coverage_authority_alignment_sha256": normalized[
                "coverage_authority_alignment_digest"
            ],
            "i0_grounding_sha256": normalized["i0_grounding_digest"],
            "qwen_evidence_sha256": _object_digest(row["qwen_evidence"]),
            "qwen_record_payload_sha256": normalized[
                "qwen_record_payload_sha256"
            ],
            "qwen_hard_gate_sha256": _object_digest(
                row["qwen_evidence"]["hard_gate"]
            ),
            "shard_manifest": str(generation_shard["manifest_path"]),
            "shard_manifest_sha256": generation_shard["manifest_sha256"],
            "shard_descriptor_digest": generation_shard[
                "descriptor_digest"
            ],
            "shard_index": generation_shard["shard_index"],
            "shard_row_index": generation_shard["shard_row_index"],
            "root_row_index": generation_shard["root_row_index"],
        },
        "wan_binding": {
            "root": str(candidate["wan"]["root"]),
            "generated_manifest": candidate["wan"]["closure"][
                "generated_manifest"
            ],
            "run_contract": candidate["wan"]["closure"]["run_contract"],
            "run_complete": candidate["wan"]["closure"]["run_complete"],
            "generated_row_digest": _object_digest(
                candidate["wan"]["generated_row"]
            ),
        },
        "postcheck_binding": {
            "output": candidate["postcheck"]["closure"]["output"],
            "receipt": candidate["postcheck"]["closure"]["receipt"],
            "record_digest": record["result_digest"],
            "aggregate_digest": _object_digest(record["aggregate"]),
            "change_region_proposals_sha256": normalized[
                "change_region_proposals_digest"
            ],
            "coverage_authority_inventory_sha256": normalized[
                "coverage_authority_inventory_digest"
            ],
            "coverage_authority_assignments_sha256": normalized[
                "coverage_authority_assignments_digest"
            ],
            "coverage_authority_sha256": normalized[
                "coverage_authority_digest"
            ],
            "coverage_authority_alignment_sha256": normalized[
                "coverage_authority_alignment_digest"
            ],
            "qwen_record_payload_sha256": normalized[
                "qwen_record_payload_sha256"
            ],
        },
        "artifacts": artifacts,
    }
    dataset_row["artifact_digest"] = _object_digest(artifacts)
    return dataset_row


def _dataset_row_artifact_map(row: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = row["artifacts"]
    flat: dict[str, Any] = {}
    for role, value in artifacts.items():
        if role == "conditioning":
            for field, artifact in value.items():
                flat[f"conditioning:{field}"] = {
                    "relative_path": artifact["relative_path"],
                    "sha256": artifact["sha256"],
                    "bytes": artifact["bytes"],
                }
        else:
            flat[role] = {
                "relative_path": value["relative_path"],
                "sha256": value["sha256"],
                "bytes": value["bytes"],
            }
    return dict(sorted(flat.items()))


def _publish_dataset(
    selected: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    exact_size: int,
    min_multi_unit: int,
    finalizer_closure: Mapping[str, Any],
    generation_shard_closure: Mapping[str, Any],
    wan_closures: Sequence[Mapping[str, Any]],
    postcheck_closures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = select_exact_candidates(
        selected,
        exact_size=exact_size,
        min_multi_unit=min_multi_unit,
    )
    if [item["iid"] for item in ordered] != [item.get("iid") for item in selected]:
        raise GokuFullMotionSelect128Error(
            "publication selection is not in deterministic selection order"
        )
    raw_output = Path(output_dir).expanduser()
    if raw_output.name in {"", ".", ".."}:
        raise GokuFullMotionSelect128Error("output directory name is unsafe")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    parent = _regular_directory(raw_output.parent, context="output parent")
    output = parent / raw_output.name
    if os.path.lexists(output):
        raise FileExistsError(output)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=parent
        )
    )
    try:
        (staging / "samples").mkdir()
        dataset_rows = [
            _materialize_sample(
                candidate,
                selection_rank=index,
                staging_root=staging,
                final_root=output,
            )
            for index, candidate in enumerate(selected, start=1)
        ]
        manifest_raw = b"".join(
            _canonical_bytes(row) + b"\n" for row in dataset_rows
        )
        manifest_sha, manifest_bytes = _write_new(
            staging / MANIFEST_NAME, manifest_raw
        )
        selected_iids = [row["iid"] for row in dataset_rows]
        sample_artifacts = {
            row["iid"]: _dataset_row_artifact_map(row) for row in dataset_rows
        }
        sample_artifact_digest = _object_digest(sample_artifacts)
        unit_counts = {
            str(count): sum(
                row["dynamic_unit_count"] == count for row in dataset_rows
            )
            for count in (1, 2, 3)
        }
        implementation_sha = _file_digest(
            Path(__file__).resolve(strict=True),
            context="selector implementation",
        )
        input_closure = {
            "finalizer": dict(finalizer_closure),
            "generation_shards": dict(generation_shard_closure),
            "wan_runs": [dict(item) for item in wan_closures],
            "postcheck_shards": [
                dict(item) for item in postcheck_closures
            ],
        }
        input_digest = _object_digest(input_closure)
        summary: dict[str, Any] = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "complete",
            "policy": {
                "exact_size": exact_size,
                "min_multi_unit": min_multi_unit,
                "selection_order": (
                    "dynamic_unit_count_desc_then_primary_index_then_iid"
                ),
                "postcheck_requirement": (
                    "status_ok_aggregate_pass_decision_pass_eligible_true"
                ),
            },
            "counts": {
                "selected": len(dataset_rows),
                "multi_unit": sum(row["multi_unit"] for row in dataset_rows),
                "single_unit": sum(
                    not row["multi_unit"] for row in dataset_rows
                ),
                "by_dynamic_unit_count": unit_counts,
            },
            "selection_iids": selected_iids,
            "selection_digest": _object_digest(selected_iids),
            "sample_artifact_digest": sample_artifact_digest,
            "dataset_manifest": {
                "path": str(output / MANIFEST_NAME),
                "sha256": manifest_sha,
                "bytes": manifest_bytes,
                "rows": len(dataset_rows),
            },
            "inputs": input_closure,
            "input_digest": input_digest,
            "implementation": {
                "path": str(Path(__file__).resolve(strict=True)),
                "sha256": implementation_sha,
            },
        }
        summary_raw = _pretty_bytes(summary)
        summary_sha, summary_bytes = _write_new(
            staging / SUMMARY_NAME, summary_raw
        )
        published_artifacts = {
            MANIFEST_NAME: {
                "sha256": manifest_sha,
                "bytes": manifest_bytes,
                "rows": len(dataset_rows),
            },
            SUMMARY_NAME: {
                "sha256": summary_sha,
                "bytes": summary_bytes,
                "rows": 1,
            },
        }
        done_payload: dict[str, Any] = {
            "schema_version": DONE_SCHEMA,
            "status": "complete",
            "selector_schema_version": SELECT_SCHEMA,
            "config": summary["policy"],
            "counts": summary["counts"],
            "selection_iids": selected_iids,
            "selection_digest": summary["selection_digest"],
            "sample_artifact_digest": sample_artifact_digest,
            "inputs": input_closure,
            "input_digest": input_digest,
            "implementation": summary["implementation"],
            "artifacts": published_artifacts,
            "artifact_digest": _object_digest(published_artifacts),
        }
        done = dict(done_payload)
        done["done_digest"] = _object_digest(done_payload)
        _write_new(staging / DONE_NAME, _pretty_bytes(done))
        _fsync_directory(staging / "samples")
        _fsync_directory(staging)
        _publish_directory_noreplace(staging, output)
        _fsync_directory(parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    validate_materialized_dataset(output)
    return summary


def _validate_artifact_record(
    value: Any,
    *,
    root: Path,
    expected_sample: Path,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GokuFullMotionSelect128Error(f"{context} is not an object")
    artifact = dict(value)
    expected_keys = {
        "schema_version",
        "path",
        "relative_path",
        "sha256",
        "bytes",
    }
    if set(artifact) != expected_keys or artifact.get("schema_version") != (
        ARTIFACT_SCHEMA
    ):
        raise GokuFullMotionSelect128Error(f"{context} schema differs")
    relative = Path(_text(artifact.get("relative_path"), context=context))
    if relative.is_absolute() or ".." in relative.parts:
        raise GokuFullMotionSelect128Error(f"{context} relative path is unsafe")
    path = _regular_file(root / relative, context=context)
    if path.parent != expected_sample:
        raise GokuFullMotionSelect128Error(f"{context} escapes sample directory")
    if str(path) != artifact.get("path"):
        raise GokuFullMotionSelect128Error(f"{context} absolute path differs")
    expected_sha = _digest(artifact.get("sha256"), context=f"{context} SHA")
    raw = _stable_read(path, context=context)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise GokuFullMotionSelect128Error(f"{context} file SHA differs")
    if type(artifact.get("bytes")) is not int or artifact["bytes"] != len(raw):
        raise GokuFullMotionSelect128Error(f"{context} bytes differ")
    return artifact


def validate_materialized_dataset(output_dir: str | Path) -> dict[str, Any]:
    root = _regular_directory(output_dir, context="materialized dataset")
    actual_root_entries = {entry.name for entry in root.iterdir()}
    expected_root_entries = {
        "samples",
        MANIFEST_NAME,
        SUMMARY_NAME,
        DONE_NAME,
    }
    if actual_root_entries != expected_root_entries:
        raise GokuFullMotionSelect128Error(
            "materialized root closure differs: "
            f"{sorted(actual_root_entries ^ expected_root_entries)}"
        )
    samples = _regular_directory(root / "samples", context="samples directory")
    done, _ = _json_object(root / DONE_NAME, context="dataset done")
    expected_done_keys = {
        "schema_version",
        "status",
        "selector_schema_version",
        "config",
        "counts",
        "selection_iids",
        "selection_digest",
        "sample_artifact_digest",
        "inputs",
        "input_digest",
        "implementation",
        "artifacts",
        "artifact_digest",
        "done_digest",
    }
    if set(done) != expected_done_keys:
        raise GokuFullMotionSelect128Error("dataset done schema differs")
    if done.get("schema_version") != DONE_SCHEMA or done.get("status") != "complete":
        raise GokuFullMotionSelect128Error("dataset done is not complete")
    if done.get("selector_schema_version") != SELECT_SCHEMA:
        raise GokuFullMotionSelect128Error("dataset selector schema differs")
    stored_done_digest = _digest(done.get("done_digest"), context="done digest")
    done_payload = dict(done)
    del done_payload["done_digest"]
    if _object_digest(done_payload) != stored_done_digest:
        raise GokuFullMotionSelect128Error("dataset done digest differs")
    artifacts = done.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != {MANIFEST_NAME, SUMMARY_NAME}
        or done.get("artifact_digest") != _object_digest(artifacts)
    ):
        raise GokuFullMotionSelect128Error("dataset root artifact digest differs")
    for name in (MANIFEST_NAME, SUMMARY_NAME):
        metadata = artifacts.get(name)
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "sha256",
            "bytes",
            "rows",
        }:
            raise GokuFullMotionSelect128Error(f"done lacks {name}")
        raw = _stable_read(root / name, context=name)
        if (
            hashlib.sha256(raw).hexdigest() != metadata.get("sha256")
            or len(raw) != metadata.get("bytes")
        ):
            raise GokuFullMotionSelect128Error(f"done binding differs for {name}")
    manifest_raw = _stable_read(root / MANIFEST_NAME, context="dataset manifest")
    rows = _jsonl_rows_from_bytes(manifest_raw, context="dataset manifest")
    config = done.get("config")
    if not isinstance(config, Mapping):
        raise GokuFullMotionSelect128Error("dataset done config is malformed")
    if (
        set(config)
        != {
            "exact_size",
            "min_multi_unit",
            "selection_order",
            "postcheck_requirement",
        }
        or config.get("selection_order")
        != "dynamic_unit_count_desc_then_primary_index_then_iid"
        or config.get("postcheck_requirement")
        != "status_ok_aggregate_pass_decision_pass_eligible_true"
    ):
        raise GokuFullMotionSelect128Error("dataset selection policy differs")
    inputs = done.get("inputs")
    if not isinstance(inputs, Mapping) or done.get("input_digest") != _object_digest(
        inputs
    ):
        raise GokuFullMotionSelect128Error("dataset input closure digest differs")
    exact_size = config.get("exact_size")
    if type(exact_size) is not int or len(rows) != exact_size:
        raise GokuFullMotionSelect128Error("dataset manifest is not exact-size")
    if artifacts[MANIFEST_NAME].get("rows") != len(rows) or artifacts[
        SUMMARY_NAME
    ].get("rows") != 1:
        raise GokuFullMotionSelect128Error("dataset root artifact row count differs")
    if done.get("selection_iids") != [row.get("iid") for row in rows]:
        raise GokuFullMotionSelect128Error("dataset selection IID order differs")
    if done.get("selection_digest") != _object_digest(done["selection_iids"]):
        raise GokuFullMotionSelect128Error("dataset selection digest differs")
    expected_sample_dirs: set[str] = set()
    sample_artifacts: dict[str, Any] = {}
    observed_multi = 0
    unit_counts = {"1": 0, "2": 0, "3": 0}
    for selection_rank, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise GokuFullMotionSelect128Error("dataset row is not an object")
        row = dict(raw_row)
        required = {
            "schema_version",
            "iid",
            "selection_rank",
            "primary_index",
            "group_id",
            "family",
            "dynamic_unit_count",
            "multi_unit",
            "target_action_signatures",
            "sample_dir",
            "edit_instruction_sha256",
            "motion_spec_object_sha256",
            "change_region_proposals_sha256",
            "coverage_authority_inventory_sha256",
            "coverage_authority_assignments_sha256",
            "coverage_authority_sha256",
            "coverage_authority_alignment_sha256",
            "i0_grounding_sha256",
            "qwen_evidence_sha256",
            "qwen_record_payload_sha256",
            "qwen_hard_gate_sha256",
            "postcheck_result_digest",
            "generation_binding",
            "wan_binding",
            "postcheck_binding",
            "artifacts",
            "artifact_digest",
        }
        if set(row) != required or row.get("schema_version") != DATASET_ROW_SCHEMA:
            raise GokuFullMotionSelect128Error("dataset row schema differs")
        iid = _safe_iid(row.get("iid"), context="dataset iid")
        if iid in expected_sample_dirs:
            raise GokuFullMotionSelect128Error(
                f"duplicate materialized IID: {iid}"
            )
        if row.get("selection_rank") != selection_rank:
            raise GokuFullMotionSelect128Error("dataset selection rank differs")
        if type(row.get("primary_index")) is not int or row["primary_index"] < 0:
            raise GokuFullMotionSelect128Error(
                f"primary index differs for {iid}"
            )
        _text(row.get("group_id"), context=f"{iid} group_id")
        _text(row.get("family"), context=f"{iid} family")
        unit_count = row.get("dynamic_unit_count")
        if type(unit_count) is not int or not 1 <= unit_count <= 3:
            raise GokuFullMotionSelect128Error(
                f"dynamic unit count differs for {iid}"
            )
        if type(row.get("multi_unit")) is not bool or row["multi_unit"] is not (
            unit_count >= 2
        ):
            raise GokuFullMotionSelect128Error(
                f"multi-unit flag differs for {iid}"
            )
        observed_multi += int(row["multi_unit"])
        unit_counts[str(unit_count)] += 1
        signatures = row.get("target_action_signatures")
        if (
            not isinstance(signatures, list)
            or len(signatures) != unit_count
            or any(type(value) is not str or not value for value in signatures)
        ):
            raise GokuFullMotionSelect128Error(
                f"target action signatures differ for {iid}"
            )
        _digest(
            row.get("edit_instruction_sha256"),
            context=f"{iid} instruction SHA",
        )
        _digest(
            row.get("motion_spec_object_sha256"),
            context=f"{iid} motion-spec SHA",
        )
        for field in (
            "change_region_proposals_sha256",
            "coverage_authority_inventory_sha256",
            "coverage_authority_assignments_sha256",
            "coverage_authority_sha256",
            "coverage_authority_alignment_sha256",
            "i0_grounding_sha256",
            "qwen_evidence_sha256",
            "qwen_record_payload_sha256",
            "qwen_hard_gate_sha256",
        ):
            _digest(row.get(field), context=f"{iid} {field}")
        _digest(
            row.get("postcheck_result_digest"),
            context=f"{iid} postcheck digest",
        )
        expected_sample_dirs.add(iid)
        sample = _regular_directory(samples / iid, context=f"sample {iid}")
        if str(sample) != row.get("sample_dir"):
            raise GokuFullMotionSelect128Error(f"sample path differs for {iid}")
        row_artifacts = row.get("artifacts")
        if not isinstance(row_artifacts, Mapping):
            raise GokuFullMotionSelect128Error(f"artifacts malformed for {iid}")
        required_roles = {
            "source",
            "target",
            "result",
            "edit_instruction",
            "generation_row",
            "motion_spec",
            "qwen_record_payload",
            "postcheck",
            "conditioning",
        }
        if set(row_artifacts) != required_roles:
            raise GokuFullMotionSelect128Error(f"artifact roles differ for {iid}")
        expected_role_names = {
            "source": "source.mp4",
            "target": "target.mp4",
            "result": "result.json",
            "edit_instruction": "edit_instruction.txt",
            "generation_row": "generation_row.json",
            "motion_spec": "motion_spec.json",
            "qwen_record_payload": "qwen_record_payload.json",
            "postcheck": "postcheck.json",
        }
        for role in required_roles - {"conditioning"}:
            artifact = _validate_artifact_record(
                row_artifacts[role],
                root=root,
                expected_sample=sample,
                context=f"{iid} {role}",
            )
            if Path(artifact["relative_path"]).name != expected_role_names[role]:
                raise GokuFullMotionSelect128Error(
                    f"standard artifact filename differs for {iid} {role}"
                )
        if (
            row_artifacts["edit_instruction"]["sha256"]
            != row.get("edit_instruction_sha256")
        ):
            raise GokuFullMotionSelect128Error(
                f"instruction artifact binding differs for {iid}"
            )
        motion_spec_value = _parse_json(
            _stable_read(
                Path(row_artifacts["motion_spec"]["path"]),
                context=f"{iid} motion_spec",
            ),
            context=f"{iid} motion_spec",
        )
        if (
            not isinstance(motion_spec_value, Mapping)
            or set(motion_spec_value) != _MOTION_SPEC_KEYS
            or motion_spec_value.get("schema_version")
            != _MOTION_SPEC_SCHEMA_V6
            or _object_digest(motion_spec_value)
            != row.get("motion_spec_object_sha256")
            or _object_digest(
                motion_spec_value.get("change_region_proposals")
            )
            != row.get("change_region_proposals_sha256")
            or _object_digest(motion_spec_value.get("coverage_authority"))
            != row.get("coverage_authority_sha256")
            or _object_digest(
                motion_spec_value.get("coverage_authority", {}).get(
                    "inventory"
                )
            )
            != row.get("coverage_authority_inventory_sha256")
            or _object_digest(
                motion_spec_value.get("coverage_authority", {}).get(
                    "assignments"
                )
            )
            != row.get("coverage_authority_assignments_sha256")
            or _object_digest(
                motion_spec_value.get("coverage_authority_alignment")
            )
            != row.get("coverage_authority_alignment_sha256")
            or _object_digest(motion_spec_value.get("i0_grounding"))
            != row.get("i0_grounding_sha256")
        ):
            raise GokuFullMotionSelect128Error(
                f"motion_spec object binding differs for {iid}"
            )
        generation_row_value = _parse_json(
            _stable_read(
                Path(row_artifacts["generation_row"]["path"]),
                context=f"{iid} generation_row",
            ),
            context=f"{iid} generation_row",
        )
        if not isinstance(generation_row_value, Mapping):
            raise GokuFullMotionSelect128Error(
                f"generation row artifact is malformed for {iid}"
            )
        try:
            validated_generation_row = _validate_generation_row(
                generation_row_value
            )
        except GokuFullMotionSelect128Error as error:
            raise GokuFullMotionSelect128Error(
                f"generation row artifact validation failed for {iid}: {error}"
            ) from error
        qwen_record_payload_value = _parse_json(
            _stable_read(
                Path(row_artifacts["qwen_record_payload"]["path"]),
                context=f"{iid} qwen_record_payload",
            ),
            context=f"{iid} qwen_record_payload",
        )
        if (
            validated_generation_row.get("iid") != iid
            or validated_generation_row.get("motion_spec") != motion_spec_value
            or validated_generation_row.get("motion_spec_sha256")
            != row.get("motion_spec_object_sha256")
            or validated_generation_row.get("edit_instruction_sha256")
            != row.get("edit_instruction_sha256")
            or _object_digest(validated_generation_row.get("qwen_evidence"))
            != row.get("qwen_evidence_sha256")
            or not isinstance(qwen_record_payload_value, Mapping)
            or qwen_record_payload_value
            != validated_generation_row.get("qwen_evidence", {}).get(
                "qwen_record_payload"
            )
            or _object_digest(qwen_record_payload_value)
            != row.get("qwen_record_payload_sha256")
            or validated_generation_row.get("qwen_evidence", {}).get(
                "change_region_proposals_digest"
            )
            != row.get("change_region_proposals_sha256")
            or validated_generation_row.get("qwen_evidence", {}).get(
                "coverage_authority_inventory_digest"
            )
            != row.get("coverage_authority_inventory_sha256")
            or validated_generation_row.get("qwen_evidence", {}).get(
                "coverage_authority_assignments_digest"
            )
            != row.get("coverage_authority_assignments_sha256")
            or validated_generation_row.get("qwen_evidence", {}).get(
                "coverage_authority_digest"
            )
            != row.get("coverage_authority_sha256")
            or validated_generation_row.get("qwen_evidence", {}).get(
                "coverage_authority_alignment_digest"
            )
            != row.get("coverage_authority_alignment_sha256")
            or _object_digest(
                validated_generation_row.get("qwen_evidence", {}).get(
                    "hard_gate"
                )
            )
            != row.get("qwen_hard_gate_sha256")
        ):
            raise GokuFullMotionSelect128Error(
                f"generation row artifact binding differs for {iid}"
            )
        postcheck_value = _parse_json(
            _stable_read(
                Path(row_artifacts["postcheck"]["path"]),
                context=f"{iid} postcheck",
            ),
            context=f"{iid} postcheck",
        )
        if (
            not isinstance(postcheck_value, Mapping)
            or postcheck_value.get("result_digest")
            != row.get("postcheck_result_digest")
            or postcheck_value.get("change_region_proposals_digest")
            != row.get("change_region_proposals_sha256")
            or postcheck_value.get("coverage_authority_digest")
            != row.get("coverage_authority_sha256")
            or postcheck_value.get("coverage_authority_inventory_digest")
            != row.get("coverage_authority_inventory_sha256")
            or postcheck_value.get("coverage_authority_assignments_digest")
            != row.get("coverage_authority_assignments_sha256")
            or postcheck_value.get("coverage_authority_alignment_digest")
            != row.get("coverage_authority_alignment_sha256")
            or postcheck_value.get("qwen_record_payload_sha256")
            != row.get("qwen_record_payload_sha256")
            or postcheck_value.get("qwen_evidence_binding", {}).get(
                "qwen_record_payload_sha256"
            )
            != row.get("qwen_record_payload_sha256")
        ):
            raise GokuFullMotionSelect128Error(
                f"postcheck record binding differs for {iid}"
            )
        postcheck_payload = dict(postcheck_value)
        postcheck_digest = postcheck_payload.pop("result_digest", None)
        if (
            _digest(postcheck_digest, context=f"{iid} postcheck digest")
            != _object_digest(postcheck_payload)
        ):
            raise GokuFullMotionSelect128Error(
                f"postcheck self digest differs for {iid}"
            )
        result_value = _parse_json(
            _stable_read(
                Path(row_artifacts["result"]["path"]),
                context=f"{iid} result",
            ),
            context=f"{iid} result",
        )
        if not isinstance(result_value, Mapping):
            raise GokuFullMotionSelect128Error(
                f"result record is malformed for {iid}"
            )
        result_payload = dict(result_value)
        result_digest = result_payload.pop("result_digest", None)
        if (
            _digest(result_digest, context=f"{iid} result digest")
            != _object_digest(result_payload)
        ):
            raise GokuFullMotionSelect128Error(
                f"result self digest differs for {iid}"
            )
        conditioning = row_artifacts["conditioning"]
        if not isinstance(conditioning, Mapping) or not (
            _REQUIRED_CONDITIONING_FIELDS <= set(conditioning)
        ):
            raise GokuFullMotionSelect128Error(
                f"conditioning closure differs for {iid}"
            )
        for field, artifact in conditioning.items():
            _validate_artifact_record(
                artifact,
                root=root,
                expected_sample=sample,
                context=f"{iid} {field}",
            )
        expected_sample_files = {
            Path(artifact["relative_path"]).name
            for role, artifact in row_artifacts.items()
            if role != "conditioning"
        }
        expected_sample_files.update(
            Path(artifact["relative_path"]).name
            for artifact in conditioning.values()
        )
        actual_sample_files = {entry.name for entry in sample.iterdir()}
        if actual_sample_files != expected_sample_files or any(
            entry.is_symlink() or not entry.is_file() for entry in sample.iterdir()
        ):
            raise GokuFullMotionSelect128Error(
                f"sample file closure differs for {iid}"
            )
        if row.get("artifact_digest") != _object_digest(row_artifacts):
            raise GokuFullMotionSelect128Error(
                f"sample artifact digest differs for {iid}"
            )
        generation_binding = row.get("generation_binding")
        if (
            not isinstance(generation_binding, Mapping)
            or set(generation_binding)
            != {
                "manifest",
                "manifest_sha256",
                "row_digest",
                "motion_spec_sha256",
                "change_region_proposals_sha256",
                "coverage_authority_inventory_sha256",
                "coverage_authority_assignments_sha256",
                "coverage_authority_sha256",
                "coverage_authority_alignment_sha256",
                "i0_grounding_sha256",
                "qwen_evidence_sha256",
                "qwen_record_payload_sha256",
                "qwen_hard_gate_sha256",
                "shard_manifest",
                "shard_manifest_sha256",
                "shard_descriptor_digest",
                "shard_index",
                "shard_row_index",
                "root_row_index",
            }
            or generation_binding.get("motion_spec_sha256")
            != row["motion_spec_object_sha256"]
            or generation_binding.get("change_region_proposals_sha256")
            != row["change_region_proposals_sha256"]
            or generation_binding.get("coverage_authority_sha256")
            != row["coverage_authority_sha256"]
            or generation_binding.get("coverage_authority_inventory_sha256")
            != row["coverage_authority_inventory_sha256"]
            or generation_binding.get("coverage_authority_assignments_sha256")
            != row["coverage_authority_assignments_sha256"]
            or generation_binding.get(
                "coverage_authority_alignment_sha256"
            )
            != row["coverage_authority_alignment_sha256"]
            or generation_binding.get("i0_grounding_sha256")
            != row["i0_grounding_sha256"]
            or generation_binding.get("qwen_evidence_sha256")
            != row["qwen_evidence_sha256"]
            or generation_binding.get("qwen_record_payload_sha256")
            != row["qwen_record_payload_sha256"]
            or generation_binding.get("qwen_hard_gate_sha256")
            != row["qwen_hard_gate_sha256"]
            or generation_binding.get("root_row_index") != row["primary_index"]
            or type(generation_binding.get("shard_index")) is not int
            or not 0 <= generation_binding["shard_index"] < 32
            or type(generation_binding.get("shard_row_index")) is not int
            or not 0 <= generation_binding["shard_row_index"] < 8
            or generation_binding["root_row_index"]
            != (
                generation_binding["shard_index"] * 8
                + generation_binding["shard_row_index"]
            )
        ):
            raise GokuFullMotionSelect128Error(
                f"generation binding differs for {iid}"
            )
        _text(generation_binding.get("manifest"), context=f"{iid} manifest")
        _text(
            generation_binding.get("shard_manifest"),
            context=f"{iid} shard manifest",
        )
        for field in (
            "manifest_sha256",
            "row_digest",
            "motion_spec_sha256",
            "change_region_proposals_sha256",
            "coverage_authority_inventory_sha256",
            "coverage_authority_assignments_sha256",
            "coverage_authority_sha256",
            "coverage_authority_alignment_sha256",
            "i0_grounding_sha256",
            "qwen_evidence_sha256",
            "qwen_record_payload_sha256",
            "qwen_hard_gate_sha256",
            "shard_manifest_sha256",
            "shard_descriptor_digest",
        ):
            _digest(
                generation_binding.get(field),
                context=f"{iid} generation {field}",
            )
        if generation_binding.get("row_digest") != _object_digest(
            validated_generation_row
        ):
            raise GokuFullMotionSelect128Error(
                f"generation row object digest differs for {iid}"
            )
        wan_binding = row.get("wan_binding")
        if not isinstance(wan_binding, Mapping) or set(wan_binding) != {
            "root",
            "generated_manifest",
            "run_contract",
            "run_complete",
            "generated_row_digest",
        }:
            raise GokuFullMotionSelect128Error(f"Wan binding differs for {iid}")
        _text(wan_binding.get("root"), context=f"{iid} Wan root")
        _digest(
            wan_binding.get("generated_row_digest"),
            context=f"{iid} generated-row digest",
        )
        for field in ("generated_manifest", "run_contract", "run_complete"):
            if not isinstance(wan_binding.get(field), Mapping):
                raise GokuFullMotionSelect128Error(
                    f"Wan {field} binding differs for {iid}"
                )
        post_binding = row.get("postcheck_binding")
        if (
            not isinstance(post_binding, Mapping)
            or set(post_binding)
            != {
                "output",
                "receipt",
                "record_digest",
                "aggregate_digest",
                "change_region_proposals_sha256",
                "coverage_authority_inventory_sha256",
                "coverage_authority_assignments_sha256",
                "coverage_authority_sha256",
                "coverage_authority_alignment_sha256",
                "qwen_record_payload_sha256",
            }
            or post_binding.get("record_digest")
            != row["postcheck_result_digest"]
            or post_binding.get("change_region_proposals_sha256")
            != row["change_region_proposals_sha256"]
            or post_binding.get("coverage_authority_sha256")
            != row["coverage_authority_sha256"]
            or post_binding.get("coverage_authority_inventory_sha256")
            != row["coverage_authority_inventory_sha256"]
            or post_binding.get("coverage_authority_assignments_sha256")
            != row["coverage_authority_assignments_sha256"]
            or post_binding.get("coverage_authority_alignment_sha256")
            != row["coverage_authority_alignment_sha256"]
            or post_binding.get("qwen_record_payload_sha256")
            != row["qwen_record_payload_sha256"]
        ):
            raise GokuFullMotionSelect128Error(
                f"postcheck binding differs for {iid}"
            )
        for field in (
            "record_digest",
            "aggregate_digest",
            "change_region_proposals_sha256",
            "coverage_authority_inventory_sha256",
            "coverage_authority_assignments_sha256",
            "coverage_authority_sha256",
            "coverage_authority_alignment_sha256",
            "qwen_record_payload_sha256",
        ):
            _digest(
                post_binding.get(field), context=f"{iid} postcheck {field}"
            )
        for field in ("output", "receipt"):
            if not isinstance(post_binding.get(field), Mapping):
                raise GokuFullMotionSelect128Error(
                    f"postcheck {field} binding differs for {iid}"
                )
        sample_artifacts[iid] = _dataset_row_artifact_map(row)
    actual_sample_dirs = {entry.name for entry in samples.iterdir()}
    if actual_sample_dirs != expected_sample_dirs or any(
        entry.is_symlink() or not entry.is_dir() for entry in samples.iterdir()
    ):
        raise GokuFullMotionSelect128Error("sample directory closure differs")
    if done.get("sample_artifact_digest") != _object_digest(sample_artifacts):
        raise GokuFullMotionSelect128Error(
            "dataset aggregate sample artifact digest differs"
        )
    expected_counts = {
        "selected": len(rows),
        "multi_unit": observed_multi,
        "single_unit": len(rows) - observed_multi,
        "by_dynamic_unit_count": unit_counts,
    }
    if done.get("counts") != expected_counts:
        raise GokuFullMotionSelect128Error("dataset counts differ")
    minimum_multi = config.get("min_multi_unit")
    if (
        type(minimum_multi) is not int
        or minimum_multi < 0
        or observed_multi < minimum_multi
    ):
        raise GokuFullMotionSelect128Error("dataset multi-unit quota differs")
    summary, _ = _json_object(root / SUMMARY_NAME, context="dataset summary")
    expected_summary_keys = {
        "schema_version",
        "status",
        "policy",
        "counts",
        "selection_iids",
        "selection_digest",
        "sample_artifact_digest",
        "dataset_manifest",
        "inputs",
        "input_digest",
        "implementation",
    }
    if (
        set(summary) != expected_summary_keys
        or summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("selection_iids") != done.get("selection_iids")
        or summary.get("sample_artifact_digest")
        != done.get("sample_artifact_digest")
        or summary.get("policy") != done.get("config")
        or summary.get("counts") != done.get("counts")
        or summary.get("inputs") != done.get("inputs")
        or summary.get("input_digest") != done.get("input_digest")
        or summary.get("implementation") != done.get("implementation")
        or summary.get("dataset_manifest")
        != {
            "path": str(root / MANIFEST_NAME),
            "sha256": artifacts[MANIFEST_NAME]["sha256"],
            "bytes": artifacts[MANIFEST_NAME]["bytes"],
            "rows": artifacts[MANIFEST_NAME]["rows"],
        }
    ):
        raise GokuFullMotionSelect128Error("dataset summary/done binding differs")
    return {
        "root": str(root),
        "rows": len(rows),
        "selection_iids": list(done["selection_iids"]),
        "done_digest": stored_done_digest,
        "manifest_sha256": artifacts[MANIFEST_NAME]["sha256"],
        "sample_artifact_digest": done["sample_artifact_digest"],
    }


def select_and_materialize_exact128(
    *,
    generation_manifest: str | Path,
    generation_shard_manifest_dir: str | Path,
    wan_run_roots: Sequence[str | Path],
    postcheck_outputs: Sequence[str | Path],
    output_dir: str | Path,
    finalizer_done: str | Path | None = None,
    wan_generated_manifests: Sequence[str | Path] | None = None,
    postcheck_receipts: Sequence[str | Path] | None = None,
    exact_size: int = DEFAULT_EXACT_SIZE,
    min_multi_unit: int = DEFAULT_MIN_MULTI_UNIT,
    ffprobe: str = "ffprobe",
    ffmpeg: str = "ffmpeg",
    frame0_max_mae: float = postcheck.DEFAULT_FRAME0_MAX_MAE,
    frame0_outlier_threshold: int = postcheck.DEFAULT_FRAME0_OUTLIER_THRESHOLD,
    frame0_max_outlier_fraction: float = (
        postcheck.DEFAULT_FRAME0_MAX_OUTLIER_FRACTION
    ),
    media_validator: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate all upstream closures and publish one exact dataset."""

    generation_path = _regular_file(
        generation_manifest, context="primary generation manifest"
    )
    generation_rows, finalizer_closure = load_generation_manifest(
        generation_path, finalizer_done=finalizer_done
    )
    generation_by_iid = {str(row["iid"]): row for row in generation_rows}
    (
        generation_shards_by_iid,
        generation_shards_by_path,
        _generation_shard_descriptors,
        generation_shard_closure,
    ) = load_generation_shard_manifest(
        generation_shard_manifest_dir,
        generation_manifest_path=generation_path,
        generation_rows=generation_rows,
        finalizer_closure=finalizer_closure,
    )
    wan_by_iid, wan_closures = _load_wan_runs(
        wan_run_roots,
        generated_manifests=wan_generated_manifests,
        generation_shards_by_path=generation_shards_by_path,
        generation_shards_by_iid=generation_shards_by_iid,
    )
    postcheck_by_iid, postcheck_closures = _load_postcheck_shards(
        postcheck_outputs,
        receipts=postcheck_receipts,
        generation_rows_by_iid=generation_by_iid,
        generation_shards_by_iid=generation_shards_by_iid,
        generation_shards_by_path=generation_shards_by_path,
        wan_by_iid=wan_by_iid,
    )
    if set(postcheck_by_iid) != set(wan_by_iid):
        raise GokuFullMotionSelect128Error(
            "Wan-generated and postchecked IID sets differ: "
            f"{sorted(set(postcheck_by_iid) ^ set(wan_by_iid))}"
        )

    candidates: list[dict[str, Any]] = []
    for primary_index, row in enumerate(generation_rows):
        iid = str(row["iid"])
        post = postcheck_by_iid.get(iid)
        wan = wan_by_iid.get(iid)
        if post is None or wan is None:
            continue
        generation_shard = generation_shards_by_iid[iid]
        if generation_shard["root_row_index"] != primary_index:
            raise GokuFullMotionSelect128Error(
                f"primary root index differs for {iid}"
            )
        record = post["record"]
        if (
            record.get("status") != "ok"
            or record.get("decision") != "pass"
            or record.get("eligible") is not True
            or post["aggregate"].get("decision") != "pass"
            or post["aggregate"].get("eligible") is not True
            or post["aggregate"].get("failure_codes") != []
        ):
            continue
        if (
            _regular_directory(
                record.get("generation_root"),
                context=f"postcheck generation root for {iid}",
            )
            != wan["root"]
            or record.get("run_contract_sha256")
            != wan["run_contract_sha256"]
            or _regular_file(
                record.get("generated_manifest"),
                context=f"postcheck generated manifest for {iid}",
            )
            != Path(wan["closure"]["generated_manifest"]["path"])
            or record.get("generated_manifest_sha256")
            != wan["closure"]["generated_manifest"]["sha256"]
            or record.get("run_complete_sha256")
            != wan["closure"]["run_complete"]["sha256"]
        ):
            raise GokuFullMotionSelect128Error(
                f"postcheck/Wan run binding differs for {iid}"
            )
        unit_count = len(post["normalized"]["dynamic_units"])
        finalization_count = row["full_motion_finalization"][
            "dynamic_unit_count"
        ]
        if unit_count != finalization_count:
            raise GokuFullMotionSelect128Error(
                f"dynamic-unit count differs for {iid}"
            )
        candidates.append(
            {
                "iid": iid,
                "primary_index": primary_index,
                "dynamic_unit_count": unit_count,
                "generation_row": row,
                "generation_manifest": finalizer_closure["manifest"],
                "generation_shard": generation_shard,
                "wan": wan,
                "postcheck": post,
            }
        )
    selected = select_exact_candidates(
        candidates,
        exact_size=exact_size,
        min_multi_unit=min_multi_unit,
    )
    validate_media = media_validator or postcheck.validate_generated_sample
    for candidate in selected:
        wan = candidate["wan"]
        actual_media = dict(
            validate_media(
                candidate["generation_row"],
                generated_row=wan["generated_row"],
                contract=candidate["postcheck"]["normalized"],
                manifest_path=candidate["generation_shard"]["manifest_path"],
                manifest_sha256=candidate["generation_shard"][
                    "manifest_sha256"
                ],
                run_contract=wan["run_contract"],
                run_contract_sha256=wan["run_contract_sha256"],
                generation_root=wan["root"],
                ffprobe=ffprobe,
                ffmpeg=ffmpeg,
                frame0_max_mae=frame0_max_mae,
                frame0_outlier_threshold=frame0_outlier_threshold,
                frame0_max_outlier_fraction=frame0_max_outlier_fraction,
            )
        )
        if actual_media != candidate["postcheck"]["record"].get(
            "media_binding"
        ):
            raise GokuFullMotionSelect128Error(
                f"media revalidation differs from postcheck for {candidate['iid']}"
            )
        candidate["verified_media"] = actual_media
    return _publish_dataset(
        selected,
        output_dir=output_dir,
        exact_size=exact_size,
        min_multi_unit=min_multi_unit,
        finalizer_closure=finalizer_closure,
        generation_shard_closure=generation_shard_closure,
        wan_closures=wan_closures,
        postcheck_closures=postcheck_closures,
    )


def discover_generation_shard_roots(index_dir: str | Path) -> list[Path]:
    """Resolve a fail-closed directory whose immediate children are Wan runs."""

    index = _regular_directory(index_dir, context="generation shard index dir")
    roots: list[Path] = []
    required_files = {
        "run_contract.json",
        "generated_manifest.jsonl",
        "run_complete.json",
    }
    for child in sorted(index.iterdir(), key=lambda value: value.name):
        if child.is_symlink():
            raise GokuFullMotionSelect128Error(
                f"generation shard index contains a symlink: {child}"
            )
        if not child.is_dir():
            continue
        root = child.resolve(strict=True)
        missing = [
            name
            for name in sorted(required_files)
            if (root / name).is_symlink() or not (root / name).is_file()
        ]
        samples = root / "samples"
        if samples.is_symlink() or not samples.is_dir():
            missing.append("samples/")
        if missing:
            raise GokuFullMotionSelect128Error(
                f"generation shard directory is incomplete: {root} "
                f"missing={missing}"
            )
        roots.append(root)
    if not roots:
        raise GokuFullMotionSelect128Error(
            "generation shard index contains no Wan run roots"
        )
    return roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select and materialize an exact full-motion dataset"
    )
    parser.add_argument("--generation-manifest", required=True, type=Path)
    parser.add_argument("--finalizer-done", type=Path)
    parser.add_argument(
        "--generation-shard-manifest-dir",
        "--shard-manifest-dir",
        dest="generation_shard_manifest_dir",
        required=True,
        type=Path,
        help=(
            "closed goku_full_motion_shard_manifest output containing the "
            "32 contiguous eight-row manifests and descriptors"
        ),
    )
    shard_source = parser.add_mutually_exclusive_group(required=True)
    shard_source.add_argument(
        "--wan-run-root",
        "--generation-shard-root",
        dest="wan_run_root",
        action="append",
        type=Path,
        help=(
            "repeat once per independent Wan generation shard root; each root "
            "must contain its own run contract, generated manifest, completion, "
            "and sample results"
        ),
    )
    shard_source.add_argument(
        "--generation-shard-index-dir",
        "--wan-shard-root-dir",
        dest="generation_shard_index_dir",
        type=Path,
        help=(
            "directory whose immediate child directories are independent Wan "
            "generation shard roots (normally WAN_OUTPUT_ROOT/wan_shards)"
        ),
    )
    parser.add_argument(
        "--wan-generated-manifest", action="append", type=Path
    )
    parser.add_argument(
        "--postcheck-output", required=True, action="append", type=Path
    )
    parser.add_argument("--postcheck-receipt", action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--exact-size", type=int, default=DEFAULT_EXACT_SIZE)
    parser.add_argument(
        "--min-multi-unit", type=int, default=DEFAULT_MIN_MULTI_UNIT
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--frame0-max-mae",
        type=float,
        default=postcheck.DEFAULT_FRAME0_MAX_MAE,
    )
    parser.add_argument(
        "--frame0-outlier-threshold",
        type=int,
        default=postcheck.DEFAULT_FRAME0_OUTLIER_THRESHOLD,
    )
    parser.add_argument(
        "--frame0-max-outlier-fraction",
        type=float,
        default=postcheck.DEFAULT_FRAME0_MAX_OUTLIER_FRACTION,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wan_run_roots = (
        args.wan_run_root
        if args.wan_run_root is not None
        else discover_generation_shard_roots(args.generation_shard_index_dir)
    )
    summary = select_and_materialize_exact128(
        generation_manifest=args.generation_manifest,
        generation_shard_manifest_dir=args.generation_shard_manifest_dir,
        finalizer_done=args.finalizer_done,
        wan_run_roots=wan_run_roots,
        wan_generated_manifests=args.wan_generated_manifest,
        postcheck_outputs=args.postcheck_output,
        postcheck_receipts=args.postcheck_receipt,
        output_dir=args.output_dir,
        exact_size=args.exact_size,
        min_multi_unit=args.min_multi_unit,
        ffprobe=args.ffprobe,
        ffmpeg=args.ffmpeg,
        frame0_max_mae=args.frame0_max_mae,
        frame0_outlier_threshold=args.frame0_outlier_threshold,
        frame0_max_outlier_fraction=args.frame0_max_outlier_fraction,
    )
    print(
        "[motive-goku-full-motion-select128] complete "
        f"selected={summary['counts']['selected']} "
        f"multi={summary['counts']['multi_unit']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
