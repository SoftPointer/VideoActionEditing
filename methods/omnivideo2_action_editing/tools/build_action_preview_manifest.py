#!/usr/bin/env python3
"""Join v17 Qwen, Wan, and optional natural-label preview artifacts.

This utility deliberately publishes *preview metadata only*.  It verifies the
committed artifacts it references, but it cannot approve a generated video or
authorize training.  Missing in-progress Wan/natural outputs are skipped;
committed artifacts with inconsistent hashes or authorization claims fail
closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence


QWEN_PASSED_SCHEMA = "motive-goku-full-motion-qwen-v16-passed-v1"
WAN_GENERATED_SCHEMA = "motive-wan22-i2v-generated-target-v1"
WAN_RESULT_SCHEMA = "motive-wan22-i2v-sample-v1"
WAN_COMPLETE_SCHEMA = "motive-wan22-i2v-batch-complete-v1"
NATURAL_RESULT_SCHEMA = "motive-goku-natural-motion-result-v1"
NATURAL_RECEIPT_SCHEMA = "motive-goku-natural-motion-receipt-v1"
NATURAL_DATASET_ROW_SCHEMA = "motive-goku-natural-motion-dataset-row-v1"
NATURAL_VERIFY_SUMMARY_SCHEMA = "motive-goku-natural-motion-verify-summary-v1"
OUTPUT_ROW_SCHEMA = "omnivideo2-action-preview-row-v1"
OUTPUT_SUMMARY_SCHEMA = "omnivideo2-action-preview-summary-v1"

SELECTION_POLICY_STRICT = "strict_single_actor"
SELECTION_POLICY_NATURAL_RELEASE = "natural_release_all"
SELECTION_POLICIES = (
    SELECTION_POLICY_STRICT,
    SELECTION_POLICY_NATURAL_RELEASE,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class PreviewManifestError(RuntimeError):
    """Fail-closed preview-manifest error."""


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
        raise PreviewManifestError(f"value is not canonical JSON: {error}") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object_sha256_without(value: Mapping[str, Any], field: str) -> str:
    candidate = dict(value)
    candidate.pop(field, None)
    return _object_sha256(candidate)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreviewManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, context: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, PreviewManifestError) as error:
        raise PreviewManifestError(f"invalid JSON in {context}: {error}") from error


def _require_plain_file(path: Path, *, context: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise PreviewManifestError(f"missing {context}: {path}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise PreviewManifestError(f"{context} is not a plain file: {path}")
    return path


def _require_plain_dir(path: Path, *, context: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise PreviewManifestError(f"missing {context}: {path}") from error
    if not stat.S_ISDIR(mode) or path.is_symlink():
        raise PreviewManifestError(f"{context} is not a plain directory: {path}")
    return path


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    _require_plain_file(path, context=context)
    value = _parse_json(path.read_bytes(), context=context)
    if not isinstance(value, dict):
        raise PreviewManifestError(f"{context} must contain one JSON object")
    return value


def _load_one_jsonl(path: Path, *, context: str) -> dict[str, Any]:
    _require_plain_file(path, context=context)
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise PreviewManifestError(f"{context} must end with a newline")
    lines = payload.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise PreviewManifestError(f"{context} must contain exactly one JSONL row")
    value = _parse_json(lines[0], context=context)
    if not isinstance(value, dict):
        raise PreviewManifestError(f"{context} row must be an object")
    return value


def _load_jsonl(path: Path, *, context: str) -> list[dict[str, Any]]:
    """Load a non-empty JSONL file with duplicate-key rejection."""

    _require_plain_file(path, context=context)
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise PreviewManifestError(f"{context} must end with a newline")
    lines = payload.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise PreviewManifestError(f"{context} must contain non-blank JSONL rows")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        value = _parse_json(line, context=f"{context} line {line_number}")
        if not isinstance(value, dict):
            raise PreviewManifestError(
                f"{context} line {line_number} must be an object"
            )
        rows.append(value)
    return rows


def _sha_field(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PreviewManifestError(f"invalid SHA-256 in {context}")
    return value


def _text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PreviewManifestError(f"invalid text in {context}")
    return value


def _iid(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _IID_RE.fullmatch(value) is None:
        raise PreviewManifestError(f"invalid IID in {context}: {value!r}")
    return value


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreviewManifestError(f"{context} must be an object")
    return dict(value)


def _list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise PreviewManifestError(f"{context} must be a list")
    return value


def _assert_file_hash(
    path: Path,
    declared_sha256: Any,
    *,
    context: str,
    declared_bytes: Any = None,
) -> str:
    _require_plain_file(path, context=context)
    expected = _sha_field(declared_sha256, context=f"{context} declared hash")
    actual = _file_sha256(path)
    if actual != expected:
        raise PreviewManifestError(
            f"{context} hash mismatch: expected={expected} actual={actual} path={path}"
        )
    if declared_bytes is not None:
        if type(declared_bytes) is not int or declared_bytes < 0:
            raise PreviewManifestError(f"invalid byte count in {context}")
        if path.stat().st_size != declared_bytes:
            raise PreviewManifestError(
                f"{context} byte count mismatch: expected={declared_bytes} "
                f"actual={path.stat().st_size}"
            )
    return actual


def _same_file_path(declared: Any, expected: Path, *, context: str) -> Path:
    text = _text(declared, context=context)
    declared_path = Path(text).expanduser()
    _require_plain_file(declared_path, context=context)
    _require_plain_file(expected, context=f"expected {context}")
    if declared_path.resolve(strict=True) != expected.resolve(strict=True):
        raise PreviewManifestError(
            f"{context} path escapes IID commit: declared={declared_path} expected={expected}"
        )
    return expected


def _validate_qwen_row(path: Path) -> dict[str, Any]:
    row = _load_one_jsonl(path, context=f"Qwen passed row {path}")
    if row.get("schema_version") != QWEN_PASSED_SCHEMA:
        raise PreviewManifestError(f"Qwen schema differs in {path}")
    iid = _iid(row.get("iid"), context=f"Qwen row {path}")
    if path.stem != iid:
        raise PreviewManifestError(f"Qwen filename/IID mismatch: {path.name} vs {iid}")
    if (
        row.get("generation_authorized") is not False
        or row.get("production_eligible") is not False
        or row.get("human_review_status") != "pending"
    ):
        raise PreviewManifestError(f"Qwen row makes a non-preview claim: {iid}")
    for field in (
        "action_change_substantive",
        "all_dynamic_subjects_covered",
        "camera_covered",
    ):
        if row.get(field) is not True:
            raise PreviewManifestError(f"Qwen row {iid} has false {field}")

    instruction = _text(row.get("edit_instruction"), context=f"Qwen instruction {iid}")
    instruction_sha = _sha_field(
        row.get("edit_instruction_sha256"), context=f"Qwen instruction {iid}"
    )
    if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != instruction_sha:
        raise PreviewManifestError(f"Qwen instruction hash mismatch: {iid}")

    census = _mapping(row.get("source_census"), context=f"source census {iid}")
    plan = _mapping(row.get("target_plan"), context=f"target plan {iid}")
    compiled = _mapping(
        row.get("compiled_instruction"), context=f"compiled instruction {iid}"
    )
    if census.get("iid") != iid or plan.get("iid") != iid or compiled.get("iid") != iid:
        raise PreviewManifestError(f"Qwen nested IID mismatch: {iid}")
    if compiled.get("instruction") != instruction or compiled.get(
        "instruction_sha256"
    ) != instruction_sha:
        raise PreviewManifestError(f"Qwen compiled instruction differs: {iid}")
    if compiled.get("source_census_sha256") != _object_sha256(census):
        raise PreviewManifestError(f"Qwen source-census hash mismatch: {iid}")
    if compiled.get("target_plan_sha256") != _object_sha256(plan):
        raise PreviewManifestError(f"Qwen target-plan hash mismatch: {iid}")
    _sha_field(row.get("qwen_record_digest"), context=f"Qwen record digest {iid}")
    _sha_field(row.get("source_video_sha256"), context=f"Qwen source video {iid}")
    _sha_field(row.get("anchor_sha256"), context=f"Qwen anchor {iid}")
    return row


def _selection_gates(row: Mapping[str, Any]) -> tuple[dict[str, bool], list[str]]:
    iid = str(row["iid"])
    census = _mapping(row.get("source_census"), context=f"source census {iid}")
    plan = _mapping(row.get("target_plan"), context=f"target plan {iid}")
    subjects = _list(census.get("dynamic_subjects"), context=f"dynamic subjects {iid}")
    targets = _list(
        plan.get("dynamic_subject_targets"), context=f"dynamic subject targets {iid}"
    )
    source_camera = _mapping(census.get("camera"), context=f"source camera {iid}")
    target_camera = _mapping(plan.get("camera_target"), context=f"target camera {iid}")
    gates = {
        "single_dynamic_actor": (
            len(subjects) == 1
            and subjects[0].get("dynamic") is True
            and len(targets) == 1
            and targets[0].get("subject_id") == subjects[0].get("subject_id")
        ),
        "source_camera_locked_off": source_camera.get("motion_class") == "locked_off",
        "target_camera_locked_off": target_camera.get("motion_class") == "locked_off",
        "target_camera_preserve_static": target_camera.get("relation") == "preserve_static",
        "source_census_high_confidence": census.get("confidence") == "high",
        "target_plan_high_confidence": plan.get("confidence") == "high",
    }
    return gates, [name for name, passed in gates.items() if not passed]


def _load_natural_release(natural_root: Path) -> dict[str, Any]:
    """Validate the completed natural-v5 release and index its accepted rows.

    The release contains every semantically accepted natural instruction, not
    just the narrower single-actor cohort used by the original OmniVideo2
    overfit study.  This function binds an opt-in broad preview join to the
    publisher's manifest and verification receipt instead of inferring success
    from directory contents.
    """

    summary_path = natural_root / "verification_summary.json"
    manifest_path = natural_root / "natural_edit_instruction_manifest.jsonl"
    summary = _load_json(summary_path, context="natural release verification summary")
    if summary.get("schema_version") != NATURAL_VERIFY_SUMMARY_SCHEMA:
        raise PreviewManifestError("natural release verification schema differs")
    _validate_self_digest(
        summary,
        "summary_digest",
        context="natural release verification summary",
    )
    _same_file_path(
        summary.get("dataset_manifest_path"),
        manifest_path,
        context="natural release dataset manifest",
    )
    manifest_sha = _assert_file_hash(
        manifest_path,
        summary.get("dataset_manifest_sha256"),
        context="natural release dataset manifest",
    )
    expected_rows = summary.get("expected_rows")
    terminal_rows = summary.get("terminal_rows")
    ok_rows = summary.get("ok_rows")
    error_rows = summary.get("error_rows")
    if (
        type(expected_rows) is not int
        or type(terminal_rows) is not int
        or type(ok_rows) is not int
        or type(error_rows) is not int
        or expected_rows <= 0
        or terminal_rows != expected_rows
        or ok_rows <= 0
        or error_rows < 0
        or ok_rows + error_rows != terminal_rows
    ):
        raise PreviewManifestError("natural release verification counts differ")

    rows = _load_jsonl(manifest_path, context="natural release dataset manifest")
    if len(rows) != ok_rows:
        raise PreviewManifestError(
            f"natural release row count differs: manifest={len(rows)} summary={ok_rows}"
        )
    indexed: dict[str, dict[str, Any]] = {}
    row_file_hashes: dict[str, str] = {}
    for row in rows:
        if row.get("schema_version") != NATURAL_DATASET_ROW_SCHEMA:
            raise PreviewManifestError("natural release dataset row schema differs")
        iid = _iid(row.get("iid"), context="natural release dataset IID")
        if iid in indexed:
            raise PreviewManifestError(f"duplicate natural release IID: {iid}")
        instruction = _text(
            row.get("natural_edit_instruction"),
            context=f"natural release instruction {iid}",
        )
        instruction_sha = _sha_field(
            row.get("natural_edit_instruction_sha256"),
            context=f"natural release instruction {iid}",
        )
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != instruction_sha:
            raise PreviewManifestError(
                f"natural release instruction hash mismatch: {iid}"
            )
        semantic_audit = _mapping(
            row.get("semantic_audit"), context=f"natural release audit {iid}"
        )
        diagnostics = _mapping(
            semantic_audit.get("model_reported_diagnostics"),
            context=f"natural release diagnostics {iid}",
        )
        if (
            row.get("label_status")
            != "structured_plan_semantic_audit_passed_video_audit_pending"
            or semantic_audit.get("effective_verdict") != "pass"
            or diagnostics.get("confidence") != "high"
        ):
            raise PreviewManifestError(
                f"natural release row is not a high-confidence semantic pass: {iid}"
            )
        _sha_field(
            row.get("source_passed_sha256"),
            context=f"natural release Qwen hash {iid}",
        )
        _sha_field(
            row.get("result_sha256"),
            context=f"natural release result hash {iid}",
        )
        indexed[iid] = row
        row_file_hashes[iid] = hashlib.sha256(
            canonical_json_bytes(row) + b"\n"
        ).hexdigest()
    return {
        "summary_path": summary_path,
        "summary_sha256": _file_sha256(summary_path),
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "expected_rows": expected_rows,
        "ok_rows": ok_rows,
        "error_rows": error_rows,
        "iid_set_sha256": hashlib.sha256(
            "".join(f"{iid}\n" for iid in sorted(indexed)).encode("utf-8")
        ).hexdigest(),
        "rows": indexed,
        "row_file_hashes": row_file_hashes,
    }


def _verify_qwen_media(row: Mapping[str, Any]) -> dict[str, Path]:
    iid = str(row["iid"])
    source = Path(
        _text(row.get("resolved_source_video"), context=f"Qwen source path {iid}")
    )
    anchor = Path(
        _text(row.get("resolved_anchor_image"), context=f"Qwen anchor path {iid}")
    )
    _assert_file_hash(
        source, row.get("source_video_sha256"), context=f"Qwen source video {iid}"
    )
    _assert_file_hash(anchor, row.get("anchor_sha256"), context=f"Qwen anchor {iid}")
    return {"source": source, "anchor": anchor}


def _validate_self_digest(value: Mapping[str, Any], field: str, *, context: str) -> str:
    declared = _sha_field(value.get(field), context=f"{context}.{field}")
    actual = _object_sha256_without(value, field)
    if declared != actual:
        raise PreviewManifestError(
            f"{context} {field} mismatch: expected={declared} actual={actual}"
        )
    return declared


def _verify_wan_commit(
    wan_root: Path, row: Mapping[str, Any], qwen_media: Mapping[str, Path]
) -> Optional[dict[str, Any]]:
    iid = str(row["iid"])
    wrapper = wan_root / "samples" / iid
    generated_path = wrapper / "generated_manifest.jsonl"
    complete_path = wrapper / "run_complete.json"
    contract_path = wrapper / "run_contract.json"
    commit_paths = (generated_path, complete_path, contract_path)
    if not all(path.exists() or path.is_symlink() for path in commit_paths):
        return None

    generated = _load_one_jsonl(
        generated_path, context=f"Wan generated manifest {iid}"
    )
    complete = _load_json(complete_path, context=f"Wan completion {iid}")
    contract = _load_json(contract_path, context=f"Wan contract {iid}")
    if generated.get("schema_version") != WAN_GENERATED_SCHEMA:
        raise PreviewManifestError(f"Wan generated schema differs: {iid}")
    if complete.get("schema_version") != WAN_COMPLETE_SCHEMA:
        raise PreviewManifestError(f"Wan completion schema differs: {iid}")
    if generated.get("iid") != iid:
        raise PreviewManifestError(f"Wan generated IID differs: {iid}")
    if complete.get("generated_manifest") != generated_path.name:
        raise PreviewManifestError(f"Wan completion manifest path differs: {iid}")
    generated_sha = _file_sha256(generated_path)
    if complete.get("generated_manifest_sha256") != generated_sha:
        raise PreviewManifestError(f"Wan completion manifest hash mismatch: {iid}")
    _validate_self_digest(complete, "complete_digest", context=f"Wan completion {iid}")
    contract_digest = _validate_self_digest(
        contract, "contract_digest", context=f"Wan contract {iid}"
    )
    if complete.get("contract_digest") != contract_digest:
        raise PreviewManifestError(f"Wan completion/contract binding differs: {iid}")
    if (
        complete.get("selected_sample_count") != 1
        or complete.get("completed_sample_count") != 1
    ):
        raise PreviewManifestError(f"Wan per-IID completion count differs: {iid}")

    sample_dir = wrapper / "samples" / iid
    expected_paths = {
        "source_video": sample_dir / "source_video.mp4",
        "target_preview_mp4": sample_dir / "preview.mp4",
        "edit_instruction_file": sample_dir / "edit_instruction.txt",
        "conditioning_anchor_original": sample_dir / "conditioning_anchor_original.png",
        "conditioning_frame0_float32": sample_dir / "conditioning_frame0_float32.npy",
        "conditioning_frame0_png": sample_dir / "conditioning_frame0.png",
        "result_json": sample_dir / "result.json",
    }
    verified_paths: dict[str, Path] = {}
    for key, expected in expected_paths.items():
        verified_paths[key] = _same_file_path(
            generated.get(key), expected, context=f"Wan {key} {iid}"
        )
    for key in (
        "source_video",
        "target_preview_mp4",
        "edit_instruction_file",
        "conditioning_anchor_original",
        "conditioning_frame0_float32",
        "conditioning_frame0_png",
    ):
        _assert_file_hash(
            verified_paths[key],
            generated.get(f"{key}_sha256"),
            context=f"Wan {key} {iid}",
            declared_bytes=generated.get(f"{key}_bytes"),
        )

    result = _load_json(verified_paths["result_json"], context=f"Wan result {iid}")
    if result.get("schema_version") != WAN_RESULT_SCHEMA or result.get("iid") != iid:
        raise PreviewManifestError(f"Wan result identity differs: {iid}")
    result_digest = _validate_self_digest(result, "result_digest", context=f"Wan result {iid}")
    if generated.get("result_digest") != result_digest:
        raise PreviewManifestError(f"Wan generated/result digest differs: {iid}")
    if complete.get("sample_result_digests") != [result_digest]:
        raise PreviewManifestError(f"Wan completion/result digest differs: {iid}")
    if result.get("contract_digest") != contract_digest:
        raise PreviewManifestError(f"Wan result/contract digest differs: {iid}")

    for artifact, result_key in (
        ("source_video", "source_video"),
        ("target_preview_mp4", "preview_mp4"),
        ("edit_instruction_file", "edit_instruction_file"),
        ("conditioning_anchor_original", "conditioning_anchor_original"),
        ("conditioning_frame0_float32", "conditioning_frame0_float32"),
        ("conditioning_frame0_png", "conditioning_frame0_png"),
    ):
        outputs = _mapping(result.get("outputs"), context=f"Wan outputs {iid}")
        if outputs.get(f"{result_key}_sha256") != generated.get(f"{artifact}_sha256"):
            raise PreviewManifestError(f"Wan output hash binding differs for {artifact}: {iid}")

    structured = str(row["edit_instruction"])
    structured_sha = str(row["edit_instruction_sha256"])
    if (
        generated.get("edit_instruction") != structured
        or generated.get("edit_instruction_sha256") != structured_sha
        or verified_paths["edit_instruction_file"].read_bytes() != structured.encode("utf-8")
    ):
        raise PreviewManifestError(f"Wan generation instruction differs from Qwen: {iid}")
    prompt = _mapping(result.get("prompt"), context=f"Wan prompt {iid}")
    if (
        prompt.get("field") != "edit_instruction"
        or prompt.get("text") != structured
        or prompt.get("sha256") != structured_sha
    ):
        raise PreviewManifestError(f"Wan result prompt differs from Qwen: {iid}")

    if generated.get("source_video_sha256") != row.get("source_video_sha256"):
        raise PreviewManifestError(f"Wan/Qwen source binding differs: {iid}")
    if generated.get("conditioning_anchor_original_sha256") != row.get("anchor_sha256"):
        raise PreviewManifestError(f"Wan/Qwen anchor binding differs: {iid}")
    if _file_sha256(qwen_media["source"]) != generated.get("source_video_sha256"):
        raise PreviewManifestError(f"Wan committed source differs from Qwen source: {iid}")
    if _file_sha256(qwen_media["anchor"]) != generated.get(
        "conditioning_anchor_original_sha256"
    ):
        raise PreviewManifestError(f"Wan committed anchor differs from Qwen anchor: {iid}")

    preview_bindings = _mapping(
        generated.get("preview_bindings"), context=f"Wan preview bindings {iid}"
    )
    if (
        preview_bindings.get("iid") != iid
        or preview_bindings.get("source_census_sha256")
        != _object_sha256(row["source_census"])
        or preview_bindings.get("target_plan_sha256")
        != _object_sha256(row["target_plan"])
        or preview_bindings.get("qwen_record_digest") != row.get("qwen_record_digest")
    ):
        raise PreviewManifestError(f"Wan preview/Qwen provenance differs: {iid}")

    if (
        generated.get("production_eligible") is not False
        or generated.get("generation_authorized") is not False
        or generated.get("human_review_status") != "pending"
        or generated.get("production_use_forbidden") is not True
        or result.get("production_eligible") is not False
        or result.get("generation_authorized_in_manifest") is not False
        or result.get("human_review_status_at_generation") != "pending"
        or result.get("production_use_forbidden") is not True
        or contract.get("production_use_forbidden") is not True
    ):
        raise PreviewManifestError(f"Wan commit makes a non-preview claim: {iid}")

    return {
        "generated": generated,
        "generated_path": generated_path,
        "generated_sha256": generated_sha,
        "complete_path": complete_path,
        "complete_sha256": _file_sha256(complete_path),
        "contract_path": contract_path,
        "contract_sha256": _file_sha256(contract_path),
        "result": result,
        "result_path": verified_paths["result_json"],
        "result_sha256": _file_sha256(verified_paths["result_json"]),
        "paths": verified_paths,
    }


def _verify_natural_commit(
    natural_root: Path, qwen_path: Path, row: Mapping[str, Any]
) -> Optional[dict[str, Any]]:
    iid = str(row["iid"])
    result_path = natural_root / "rows" / iid / "result.json"
    instruction_path = natural_root / "instructions" / iid / "natural_edit_instruction.txt"
    receipt_path = natural_root / "terminal" / f"{iid}.receipt.json"
    if not receipt_path.exists() and not receipt_path.is_symlink():
        return None
    receipt = _load_json(receipt_path, context=f"natural receipt {iid}")
    result = _load_json(result_path, context=f"natural result {iid}")
    if (
        receipt.get("schema_version") != NATURAL_RECEIPT_SCHEMA
        or receipt.get("iid") != iid
        or receipt.get("status") not in {"ok", "error"}
    ):
        raise PreviewManifestError(f"natural receipt identity/status differs: {iid}")
    _validate_self_digest(receipt, "receipt_digest", context=f"natural receipt {iid}")
    _same_file_path(receipt.get("result_path"), result_path, context=f"natural result path {iid}")
    _assert_file_hash(
        result_path, receipt.get("result_sha256"), context=f"natural result {iid}"
    )
    if (
        result.get("schema_version") != NATURAL_RESULT_SCHEMA
        or result.get("iid") != iid
        or result.get("status") != receipt.get("status")
    ):
        raise PreviewManifestError(f"natural result identity/status differs: {iid}")
    _validate_self_digest(result, "record_digest", context=f"natural result {iid}")
    qwen_sha = _file_sha256(qwen_path)
    if result.get("source_passed_sha256") != qwen_sha:
        raise PreviewManifestError(f"natural/Qwen file hash differs: {iid}")
    _same_file_path(
        result.get("source_passed_path"), qwen_path, context=f"natural Qwen path {iid}"
    )
    if (
        result.get("generation_prompt") != row.get("edit_instruction")
        or result.get("generation_prompt_sha256") != row.get("edit_instruction_sha256")
        or result.get("source_census_sha256") != _object_sha256(row["source_census"])
        or result.get("target_plan_sha256") != _object_sha256(row["target_plan"])
    ):
        raise PreviewManifestError(f"natural/Qwen semantic lineage differs: {iid}")
    if receipt["status"] == "error":
        if (
            receipt.get("instruction_path") is not None
            or receipt.get("instruction_sha256") is not None
            or instruction_path.exists()
            or instruction_path.is_symlink()
            or result.get("natural_edit_instruction") is not None
            or result.get("natural_edit_instruction_sha256") is not None
        ):
            raise PreviewManifestError(
                f"natural error receipt unexpectedly binds an instruction: {iid}"
            )
        return {
            "status": "error",
            "result_path": result_path,
            "result_sha256": _file_sha256(result_path),
            "receipt_path": receipt_path,
            "receipt_sha256": _file_sha256(receipt_path),
        }

    _require_plain_file(instruction_path, context=f"natural instruction {iid}")
    _same_file_path(
        receipt.get("instruction_path"),
        instruction_path,
        context=f"natural instruction path {iid}",
    )
    instruction_file_sha = _assert_file_hash(
        instruction_path,
        receipt.get("instruction_sha256"),
        context=f"natural instruction {iid}",
    )
    instruction = _text(
        result.get("natural_edit_instruction"), context=f"natural instruction text {iid}"
    )
    instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    if result.get("natural_edit_instruction_sha256") != instruction_sha:
        raise PreviewManifestError(f"natural instruction text hash differs: {iid}")
    if instruction_path.read_bytes() != (instruction + "\n").encode("utf-8"):
        raise PreviewManifestError(f"natural instruction sidecar differs: {iid}")
    audit = _mapping(result.get("audit"), context=f"natural audit {iid}")
    diagnostics = _mapping(
        audit.get("model_reported_diagnostics"), context=f"natural diagnostics {iid}"
    )
    if audit.get("effective_verdict") != "pass" or diagnostics.get("confidence") != "high":
        raise PreviewManifestError(f"natural audit is not high-confidence pass: {iid}")
    return {
        "status": "ok",
        "instruction": instruction,
        "instruction_sha256": instruction_sha,
        "instruction_file_sha256": instruction_file_sha,
        "instruction_path": instruction_path,
        "result_path": result_path,
        "result_sha256": _file_sha256(result_path),
        "receipt_path": receipt_path,
        "receipt_sha256": _file_sha256(receipt_path),
    }


def _preview_row(
    qwen_path: Path,
    qwen: Mapping[str, Any],
    gates: Mapping[str, bool],
    wan: Mapping[str, Any],
    *,
    instruction_source: str,
    natural: Optional[Mapping[str, Any]],
    natural_release: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    iid = str(qwen["iid"])
    if instruction_source == "structured":
        instruction = str(qwen["edit_instruction"])
        instruction_sha = str(qwen["edit_instruction_sha256"])
    else:
        assert natural is not None
        instruction = str(natural["instruction"])
        instruction_sha = str(natural["instruction_sha256"])
    generated = wan["generated"]
    provenance: dict[str, Any] = {
        "qwen_passed_path": str(qwen_path),
        "qwen_passed_sha256": _file_sha256(qwen_path),
        "qwen_record_digest": qwen["qwen_record_digest"],
        "wan_generated_manifest_path": str(wan["generated_path"]),
        "wan_generated_manifest_sha256": wan["generated_sha256"],
        "wan_run_complete_path": str(wan["complete_path"]),
        "wan_run_complete_sha256": wan["complete_sha256"],
        "wan_run_contract_path": str(wan["contract_path"]),
        "wan_run_contract_sha256": wan["contract_sha256"],
        "wan_result_path": str(wan["result_path"]),
        "wan_result_sha256": wan["result_sha256"],
        "wan_result_digest": wan["result"]["result_digest"],
    }
    if natural is not None:
        provenance.update(
            {
                "natural_result_path": str(natural["result_path"]),
                "natural_result_sha256": natural["result_sha256"],
                "natural_receipt_path": str(natural["receipt_path"]),
                "natural_receipt_sha256": natural["receipt_sha256"],
                "natural_instruction_path": str(natural["instruction_path"]),
                "natural_instruction_file_sha256": natural[
                    "instruction_file_sha256"
                ],
            }
        )
    if natural_release is not None:
        provenance.update(
            {
                "natural_release_summary_path": str(
                    natural_release["summary_path"]
                ),
                "natural_release_summary_sha256": natural_release[
                    "summary_sha256"
                ],
                "natural_release_manifest_path": str(
                    natural_release["manifest_path"]
                ),
                "natural_release_manifest_sha256": natural_release[
                    "manifest_sha256"
                ],
                "natural_release_row_file_sha256": natural_release[
                    "row_file_sha256"
                ],
            }
        )
    result: dict[str, Any] = {
        "schema_version": OUTPUT_ROW_SCHEMA,
        "iid": iid,
        "group_id": qwen.get("group_id"),
        "family": qwen.get("family"),
        "source_video_path": str(wan["paths"]["source_video"]),
        "source_video_sha256": generated["source_video_sha256"],
        "target_video_path": str(wan["paths"]["target_preview_mp4"]),
        "target_video_sha256": generated["target_preview_mp4_sha256"],
        "edit_instruction": instruction,
        "edit_instruction_sha256": instruction_sha,
        "instruction_source": instruction_source,
        "generation_instruction": qwen["edit_instruction"],
        "generation_instruction_sha256": qwen["edit_instruction_sha256"],
        "source_census": qwen["source_census"],
        "target_plan": qwen["target_plan"],
        "selection_gates": dict(gates),
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "production_eligible": False,
        "post_video_acceptance": "pending",
        "provenance": provenance,
    }
    result["row_digest"] = _object_sha256(result)
    return result


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


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


def _prepare_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_plain_dir(path.parent, context=f"output parent for {path}")
    if path.exists() or path.is_symlink():
        raise PreviewManifestError(f"create-only output already exists: {path}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError:
        return False
    return True


def _publish_create_only(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(path))
        except FileExistsError as error:
            raise PreviewManifestError(f"create-only output already exists: {path}") from error
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_preview_manifest(
    *,
    qwen_passed_dir: Path,
    wan_root: Path,
    output_manifest: Path,
    summary_output: Path,
    instruction_source: str,
    natural_root: Optional[Path] = None,
    selection_policy: str = SELECTION_POLICY_STRICT,
) -> dict[str, Any]:
    qwen_passed_dir = _require_plain_dir(
        qwen_passed_dir.expanduser().resolve(strict=True), context="Qwen passed directory"
    )
    wan_root = _require_plain_dir(
        wan_root.expanduser().resolve(strict=True), context="Wan root"
    )
    if instruction_source not in {"structured", "natural"}:
        raise PreviewManifestError("instruction_source must be structured or natural")
    if selection_policy not in SELECTION_POLICIES:
        raise PreviewManifestError(
            f"selection_policy must be one of {SELECTION_POLICIES}"
        )
    if instruction_source == "natural" and natural_root is None:
        raise PreviewManifestError("--natural-root is required for natural instructions")
    if (
        selection_policy == SELECTION_POLICY_NATURAL_RELEASE
        and instruction_source != "natural"
    ):
        raise PreviewManifestError(
            "natural_release_all selection requires natural instructions"
        )
    if natural_root is not None:
        natural_root = _require_plain_dir(
            natural_root.expanduser().resolve(strict=True), context="natural root"
        )
    natural_release: Optional[dict[str, Any]] = None
    if selection_policy == SELECTION_POLICY_NATURAL_RELEASE:
        assert natural_root is not None
        natural_release = _load_natural_release(natural_root)
    output_manifest = output_manifest.expanduser().absolute()
    summary_output = summary_output.expanduser().absolute()
    if output_manifest == summary_output:
        raise PreviewManifestError("manifest and summary outputs must differ")
    readonly_roots = [qwen_passed_dir, wan_root]
    if natural_root is not None:
        readonly_roots.append(natural_root)
    for output in (output_manifest, summary_output):
        for readonly_root in readonly_roots:
            if _is_within(output, readonly_root):
                raise PreviewManifestError(
                    f"output must be outside read-only input root {readonly_root}: {output}"
                )
    _prepare_output(output_manifest)
    _prepare_output(summary_output)

    qwen_paths = sorted(qwen_passed_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not qwen_paths:
        raise PreviewManifestError(f"no Qwen passed JSONL files in {qwen_passed_dir}")
    qwen_rows: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    qwen_index: list[dict[str, str]] = []
    for path in qwen_paths:
        row = _validate_qwen_row(path)
        iid = str(row["iid"])
        if iid in seen:
            raise PreviewManifestError(f"duplicate Qwen IID: {iid}")
        seen.add(iid)
        qwen_rows.append((path, row))
        qwen_index.append({"iid": iid, "sha256": _file_sha256(path)})
    if natural_release is not None:
        release_iids = set(natural_release["rows"])
        missing_qwen = sorted(release_iids - seen)
        if missing_qwen:
            raise PreviewManifestError(
                f"natural release contains IIDs absent from Qwen passed rows: "
                f"{missing_qwen[:8]}"
            )

    gate_rejections: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    eligible = 0
    strict_gate_eligible = 0
    qwen_strict_gate_eligible = 0
    output_rows: list[dict[str, Any]] = []
    for qwen_path, qwen in qwen_rows:
        gates, failed = _selection_gates(qwen)
        iid = str(qwen["iid"])
        if failed:
            gate_rejections.update(failed)
        else:
            qwen_strict_gate_eligible += 1
        release_row: Optional[dict[str, Any]] = None
        if natural_release is not None:
            release_row = natural_release["rows"].get(iid)
            if release_row is None:
                skipped["natural_release_not_accepted"] += 1
                continue
        elif failed:
            skipped["selection_gate_failed"] += 1
            continue
        eligible += 1
        if not failed:
            strict_gate_eligible += 1
        wrapper = wan_root / "samples" / iid
        commit_paths = (
            wrapper / "generated_manifest.jsonl",
            wrapper / "run_complete.json",
            wrapper / "run_contract.json",
        )
        if not all(path.exists() or path.is_symlink() for path in commit_paths):
            skipped["wan_commit_missing_or_in_progress"] += 1
            continue
        qwen_media = _verify_qwen_media(qwen)
        wan = _verify_wan_commit(wan_root, qwen, qwen_media)
        if wan is None:  # Defensive; the pre-check above avoids expensive source hashes.
            skipped["wan_commit_missing_or_in_progress"] += 1
            continue
        natural: Optional[dict[str, Any]] = None
        if instruction_source == "natural":
            assert natural_root is not None
            natural = _verify_natural_commit(natural_root, qwen_path, qwen)
            if natural is None:
                skipped["natural_commit_missing_or_in_progress"] += 1
                continue
            if natural["status"] == "error":
                skipped["natural_terminal_error"] += 1
                continue
        release_binding: Optional[dict[str, Any]] = None
        if release_row is not None:
            assert natural is not None and natural_release is not None
            if (
                natural["status"] != "ok"
                or release_row.get("natural_edit_instruction")
                != natural["instruction"]
                or release_row.get("natural_edit_instruction_sha256")
                != natural["instruction_sha256"]
                or release_row.get("source_passed_sha256")
                != _file_sha256(qwen_path)
                or release_row.get("result_sha256") != natural["result_sha256"]
            ):
                raise PreviewManifestError(
                    f"natural release/terminal binding differs: {iid}"
                )
            _same_file_path(
                release_row.get("result_path"),
                Path(natural["result_path"]),
                context=f"natural release result {iid}",
            )
            release_binding = {
                "summary_path": natural_release["summary_path"],
                "summary_sha256": natural_release["summary_sha256"],
                "manifest_path": natural_release["manifest_path"],
                "manifest_sha256": natural_release["manifest_sha256"],
                "row_file_sha256": natural_release["row_file_hashes"][iid],
            }
        output_rows.append(
            _preview_row(
                qwen_path,
                qwen,
                gates,
                wan,
                instruction_source=instruction_source,
                natural=natural,
                natural_release=release_binding,
            )
        )

    if natural_release is not None:
        release_iids = set(natural_release["rows"])
        output_iids = {str(row["iid"]) for row in output_rows}
        if output_iids != release_iids:
            missing = sorted(release_iids - output_iids)
            extra = sorted(output_iids - release_iids)
            raise PreviewManifestError(
                "natural_release_all join is incomplete: "
                f"missing={missing[:8]} extra={extra[:8]} skipped={dict(skipped)}"
            )

    output_rows.sort(key=lambda row: str(row["iid"]))
    manifest_payload = _jsonl_bytes(output_rows)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    summary: dict[str, Any] = {
        "schema_version": OUTPUT_SUMMARY_SCHEMA,
        "qwen_passed_dir": str(qwen_passed_dir),
        "qwen_passed_rows": len(qwen_rows),
        "qwen_passed_index_sha256": _object_sha256(qwen_index),
        "wan_root": str(wan_root),
        "natural_root": str(natural_root) if natural_root is not None else None,
        "instruction_source": instruction_source,
        "selection_policy": selection_policy,
        "selection_eligible_rows": eligible,
        "strict_gate_eligible_rows": strict_gate_eligible,
        "qwen_strict_gate_eligible_rows": qwen_strict_gate_eligible,
        "preview_rows": len(output_rows),
        "gate_rejections": dict(sorted(gate_rejections.items())),
        "skipped_rows": dict(sorted(skipped.items())),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": manifest_sha,
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "production_eligible": False,
        "post_video_acceptance": "pending",
    }
    if natural_release is not None:
        summary.update(
            {
                "natural_release_manifest_path": str(
                    natural_release["manifest_path"]
                ),
                "natural_release_manifest_sha256": natural_release[
                    "manifest_sha256"
                ],
                "natural_release_summary_path": str(
                    natural_release["summary_path"]
                ),
                "natural_release_summary_sha256": natural_release[
                    "summary_sha256"
                ],
                "natural_release_ok_rows": natural_release["ok_rows"],
                "natural_release_expected_rows": natural_release["expected_rows"],
                "natural_release_error_rows": natural_release["error_rows"],
                "natural_release_iid_set_sha256": natural_release[
                    "iid_set_sha256"
                ],
            }
        )
    summary["summary_digest"] = _object_sha256(summary)
    _publish_create_only(output_manifest, manifest_payload)
    _publish_create_only(summary_output, _pretty_bytes(summary))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a hash-verified, non-training action preview manifest."
    )
    parser.add_argument("--qwen-passed-dir", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, required=True)
    parser.add_argument("--natural-root", type=Path)
    parser.add_argument(
        "--instruction-source",
        choices=("structured", "natural"),
        default="structured",
    )
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default=SELECTION_POLICY_STRICT,
        help=(
            "strict_single_actor keeps the original narrow OmniVideo2 cohort; "
            "natural_release_all binds to every accepted row in the completed "
            "natural-label release while retaining the narrow gates as diagnostics"
        ),
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        build_preview_manifest(
            qwen_passed_dir=args.qwen_passed_dir,
            wan_root=args.wan_root,
            natural_root=args.natural_root,
            instruction_source=args.instruction_source,
            selection_policy=args.selection_policy,
            output_manifest=args.output_manifest,
            summary_output=args.summary_output,
        )
    except PreviewManifestError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
