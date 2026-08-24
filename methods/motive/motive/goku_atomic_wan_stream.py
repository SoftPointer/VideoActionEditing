"""Immutable batch admissions for streaming atomic labels into Wan2.2.

The atomic planner/label pipeline grows a cumulative, parent-ordered manifest.
This module turns each newly appended suffix into one frozen Wan dispatch
batch.  A dispatch batch is created immediately after an atomic verify phase;
it never waits for the final dataset target.  The executable Wan prompt stays
the deeply validated planner ``edit_instruction`` while the independent
``atomic_action_instruction`` remains the primary training label.

The existing v16 Wan watcher owns GPU scheduling and retry claims.  This
module supplies the missing append-only layer around that fixed-input watcher:

* ``admit-batch`` publishes one immutable input subset and per-IID admissions;
* ``materialize-metadata`` validates a completed watcher batch and copies all
  instruction/provenance sidecars into each successful sample directory;
* ``inspect`` reports cumulative Wan terminal counts without mutating state;
* ``publish-terminal`` closes the complete multi-batch stream exactly once.

Every write is create-only.  ``--resume`` permits only byte-identical existing
artifacts, so a controller restart cannot dispatch an IID twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


ADMISSION_BATCH_SCHEMA = "motive-goku-atomic-wan-admission-batch-v1"
ADMISSION_SCHEMA = "motive-goku-atomic-wan-admission-v1"
METADATA_SCHEMA = "motive-goku-atomic-wan-sample-metadata-v1"
METADATA_BATCH_SCHEMA = "motive-goku-atomic-wan-metadata-batch-v1"
STREAM_TERMINAL_SCHEMA = "motive-goku-atomic-wan-stream-terminal-v1"
ATOMIC_DATASET_SCHEMA = "motive-goku-atomic-motion-dataset-row-v1"
TOPUP_PROGRESS_SCHEMA = "motive-goku-atomic-topup-progress-v1"
V16_TERMINAL_SCHEMA = "motive-full-motion-v16-wan-stream-terminal-v1"

_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TAG_RE = re.compile(r"batch_([0-9]{4,})\Z")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_VIDEO_RE = re.compile(r"source_video\.[A-Za-z0-9]{1,16}\Z")


class AtomicWanStreamError(RuntimeError):
    """An admission, watcher result, or copied metadata binding differs."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_digest(value: Mapping[str, Any], *, omit: str) -> str:
    copy = dict(value)
    copy.pop(omit, None)
    return _sha_bytes(_canonical(copy))


def _plain_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except FileNotFoundError:
        return False


def _plain_dir(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode) and not path.is_symlink()
    except FileNotFoundError:
        return False


def _require_file(path: Path, *, context: str) -> Path:
    if not path.is_absolute() or not _plain_file(path):
        raise AtomicWanStreamError(f"{context} is not a plain absolute file: {path}")
    return path


def _require_dir(path: Path, *, context: str) -> Path:
    if not path.is_absolute() or not _plain_dir(path):
        raise AtomicWanStreamError(f"{context} is not a plain absolute directory: {path}")
    return path


def _publish(path: Path, raw: bytes, *, resume: bool) -> None:
    if not path.is_absolute():
        raise AtomicWanStreamError(f"output path is not absolute: {path}")
    if path.exists() or path.is_symlink():
        if not resume or not _plain_file(path) or path.read_bytes() != raw:
            raise AtomicWanStreamError(f"existing create-only artifact differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    _require_file(path, context=context)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AtomicWanStreamError(f"{context} JSON root is not an object")
    return value


def _load_jsonl(
    path: Path, *, context: str, allow_empty: bool = False
) -> tuple[list[dict[str, Any]], list[bytes]]:
    _require_file(path, context=context)
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise AtomicWanStreamError(f"{context} is not newline-terminated JSONL")
    if not raw and not allow_empty:
        raise AtomicWanStreamError(f"{context} is empty")
    rows: list[dict[str, Any]] = []
    lines = raw.splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AtomicWanStreamError(f"{context} row {number} is not an object")
        rows.append(value)
    return rows, lines


def _safe_iid(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _IID_RE.fullmatch(value) is None:
        raise AtomicWanStreamError(f"{context} has an unsafe IID")
    return value


def _sha(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise AtomicWanStreamError(f"{context} is not a SHA-256 digest")
    return value


def _text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise AtomicWanStreamError(f"{context} is not nonempty text")
    return value


def _validate_atomic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    iid = _safe_iid(value.get("iid"), context="atomic row")
    if value.get("schema_version") != ATOMIC_DATASET_SCHEMA:
        raise AtomicWanStreamError(f"atomic row schema differs iid={iid}")
    if (
        value.get("label_status")
        != "atomic_plan_and_instruction_audits_passed_video_audit_pending"
        or value.get("primary_training_label_field")
        != "atomic_action_instruction"
    ):
        raise AtomicWanStreamError(f"atomic row is not a final label pass iid={iid}")
    index = value.get("original_candidate_index")
    if type(index) is not int or index < 0:
        raise AtomicWanStreamError(f"atomic parent index differs iid={iid}")
    for field in (
        "atomic_action_instruction",
        "camera_instruction",
        "preservation_instruction",
        "full_edit_instruction",
    ):
        content = _text(value.get(field), context=f"atomic row {iid} {field}")
        expected = _sha(value.get(f"{field}_sha256"), context=f"{iid} {field} SHA")
        if _sha_bytes(content.encode("utf-8")) != expected:
            raise AtomicWanStreamError(f"atomic field digest differs iid={iid} field={field}")
    if value["full_edit_instruction"] != (
        f"{value['atomic_action_instruction']} {value['camera_instruction']} "
        f"{value['preservation_instruction']}"
    ):
        raise AtomicWanStreamError(f"atomic composite instruction differs iid={iid}")
    result = Path(_text(value.get("result_path"), context=f"{iid} result path"))
    _require_file(result, context=f"atomic result iid={iid}")
    if _sha_file(result) != _sha(value.get("result_sha256"), context=f"{iid} result SHA"):
        raise AtomicWanStreamError(f"atomic result bytes differ iid={iid}")
    result_value = _load_json(result, context=f"atomic result iid={iid}")
    if result_value.get("iid") != iid or result_value.get("status") != "ok":
        raise AtomicWanStreamError(f"atomic result is not ok iid={iid}")
    for field in (
        "atomic_action_instruction",
        "atomic_action_instruction_sha256",
        "camera_instruction",
        "camera_instruction_sha256",
        "preservation_instruction",
        "preservation_instruction_sha256",
        "full_edit_instruction",
        "full_edit_instruction_sha256",
    ):
        if result_value.get(field) != value.get(field):
            raise AtomicWanStreamError(f"atomic result field differs iid={iid} field={field}")
    provenance = value.get("source_generation_provenance")
    if not isinstance(provenance, Mapping):
        raise AtomicWanStreamError(f"atomic generation provenance is absent iid={iid}")
    prompt = _text(
        provenance.get("frame_gridded_prompt"),
        context=f"atomic generation prompt iid={iid}",
    )
    prompt_sha = _sha(
        provenance.get("frame_gridded_prompt_sha256"),
        context=f"atomic generation prompt SHA iid={iid}",
    )
    if _sha_bytes(prompt.encode("utf-8")) != prompt_sha:
        raise AtomicWanStreamError(f"atomic generation prompt digest differs iid={iid}")
    return value


def _validate_progress(path: Path, atomic_manifest: Path) -> dict[str, Any]:
    value = _load_json(path, context="top-up progress")
    if value.get("schema_version") != TOPUP_PROGRESS_SCHEMA:
        raise AtomicWanStreamError("top-up progress schema differs")
    if value.get("progress_digest") != _object_digest(value, omit="progress_digest"):
        raise AtomicWanStreamError("top-up progress digest differs")
    if (
        value.get("atomic_manifest") != str(atomic_manifest)
        or value.get("atomic_manifest_sha256") != _sha_file(atomic_manifest)
    ):
        raise AtomicWanStreamError("top-up progress does not bind the atomic manifest")
    return value


def _validate_admission_batch(path: Path) -> dict[str, Any]:
    value = _load_json(path, context="Wan admission batch")
    required = {
        "schema_version",
        "batch_tag",
        "batch_index",
        "progress_receipt",
        "progress_receipt_sha256",
        "atomic_manifest",
        "atomic_manifest_sha256",
        "planner_input",
        "planner_input_sha256",
        "planner_root",
        "wan_input",
        "wan_input_sha256",
        "wan_batch_root",
        "previous_admission_batch",
        "previous_admission_batch_sha256",
        "batch_iids",
        "batch_parent_indices",
        "batch_rows",
        "cumulative_iids",
        "cumulative_rows",
        "dispatch_policy",
        "generation_prompt_field",
        "primary_training_label_field",
        "receipt_digest",
    }
    if set(value) != required or value.get("schema_version") != ADMISSION_BATCH_SCHEMA:
        raise AtomicWanStreamError("Wan admission batch schema differs")
    if value.get("receipt_digest") != _object_digest(value, omit="receipt_digest"):
        raise AtomicWanStreamError("Wan admission batch digest differs")
    match = _TAG_RE.fullmatch(str(value.get("batch_tag", "")))
    if match is None or int(match.group(1)) != value.get("batch_index"):
        raise AtomicWanStreamError("Wan admission batch tag/index differs")
    batch_iids = value.get("batch_iids")
    cumulative = value.get("cumulative_iids")
    indices = value.get("batch_parent_indices")
    if (
        not isinstance(batch_iids, list)
        or not isinstance(cumulative, list)
        or not isinstance(indices, list)
        or any(_IID_RE.fullmatch(str(iid)) is None for iid in batch_iids + cumulative)
        or len(batch_iids) != len(set(batch_iids))
        or len(cumulative) != len(set(cumulative))
        or value.get("batch_rows") != len(batch_iids)
        or value.get("cumulative_rows") != len(cumulative)
    ):
        raise AtomicWanStreamError("Wan admission batch IID closure differs")
    if batch_iids and cumulative[-len(batch_iids) :] != batch_iids:
        raise AtomicWanStreamError("Wan admission batch is not a cumulative suffix")
    if len(indices) != len(batch_iids) or any(type(index) is not int for index in indices):
        raise AtomicWanStreamError("Wan admission parent indices differ")
    for field, sha_field in (
        ("progress_receipt", "progress_receipt_sha256"),
        ("atomic_manifest", "atomic_manifest_sha256"),
        ("planner_input", "planner_input_sha256"),
        ("wan_input", "wan_input_sha256"),
    ):
        artifact = _require_file(
            Path(str(value.get(field, ""))), context=f"admission {field}"
        )
        if _sha_file(artifact) != _sha(
            value.get(sha_field), context=f"admission {sha_field}"
        ):
            raise AtomicWanStreamError(f"Wan admission {field} bytes differ")
    _require_dir(Path(str(value.get("planner_root", ""))), context="admission planner root")
    wan_batch_root = Path(str(value.get("wan_batch_root", "")))
    if not wan_batch_root.is_absolute():
        raise AtomicWanStreamError("admission Wan batch root is not absolute")
    return value


def _validate_admission(path: Path, *, batch: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(path, context="Wan IID admission")
    required = {
        "schema_version",
        "iid",
        "batch_tag",
        "batch_index",
        "parent_index",
        "batch_admission_receipt",
        "batch_admission_receipt_sha256",
        "planner_passed",
        "planner_passed_sha256",
        "atomic_result",
        "atomic_result_sha256",
        "source_video",
        "source_video_sha256",
        "wan_generation_prompt",
        "wan_generation_prompt_sha256",
        "atomic_action_instruction",
        "atomic_action_instruction_sha256",
        "camera_instruction",
        "camera_instruction_sha256",
        "preservation_instruction",
        "preservation_instruction_sha256",
        "full_edit_instruction",
        "full_edit_instruction_sha256",
        "primary_training_label_field",
        "wan_prompt_role",
        "admission_digest",
    }
    if set(value) != required or value.get("schema_version") != ADMISSION_SCHEMA:
        raise AtomicWanStreamError("Wan IID admission schema differs")
    if value.get("admission_digest") != _object_digest(value, omit="admission_digest"):
        raise AtomicWanStreamError("Wan IID admission digest differs")
    iid = _safe_iid(value.get("iid"), context="Wan IID admission")
    if (
        value.get("batch_tag") != batch["batch_tag"]
        or value.get("batch_index") != batch["batch_index"]
    ):
        raise AtomicWanStreamError(f"Wan admission batch identity differs iid={iid}")
    receipt_path = Path(_text(value.get("batch_admission_receipt"), context="batch receipt path"))
    _require_file(receipt_path, context=f"batch receipt iid={iid}")
    if (
        _sha_file(receipt_path)
        != _sha(value.get("batch_admission_receipt_sha256"), context="batch receipt SHA")
        or _load_json(receipt_path, context="batch receipt") != dict(batch)
    ):
        raise AtomicWanStreamError(f"Wan admission batch receipt differs iid={iid}")
    for path_field, sha_field in (
        ("planner_passed", "planner_passed_sha256"),
        ("atomic_result", "atomic_result_sha256"),
        ("source_video", "source_video_sha256"),
    ):
        artifact = Path(_text(value.get(path_field), context=f"{iid} {path_field}"))
        _require_file(artifact, context=f"{iid} {path_field}")
        if _sha_file(artifact) != _sha(value.get(sha_field), context=f"{iid} {sha_field}"):
            raise AtomicWanStreamError(f"Wan admission artifact differs iid={iid} field={path_field}")
    for field in (
        "wan_generation_prompt",
        "atomic_action_instruction",
        "camera_instruction",
        "preservation_instruction",
        "full_edit_instruction",
    ):
        content = _text(value.get(field), context=f"{iid} {field}")
        if _sha_bytes(content.encode("utf-8")) != _sha(
            value.get(f"{field}_sha256"), context=f"{iid} {field} SHA"
        ):
            raise AtomicWanStreamError(f"Wan admission text digest differs iid={iid} field={field}")
    if (
        value.get("primary_training_label_field") != "atomic_action_instruction"
        or value.get("wan_prompt_role")
        != "planner_private_trajectory_generation_only"
    ):
        raise AtomicWanStreamError(f"Wan admission label/prompt roles differ iid={iid}")
    return value


def _planner_rows(path: Path, expected_sha: str) -> tuple[list[dict[str, Any]], list[bytes], dict[str, int]]:
    _require_file(path, context="planner input")
    if _sha_file(path) != _sha(expected_sha, context="planner input SHA"):
        raise AtomicWanStreamError("planner input bytes differ")
    rows, lines = _load_jsonl(path, context="planner input")
    rank: dict[str, int] = {}
    for index, row in enumerate(rows):
        iid = _safe_iid(row.get("iid"), context=f"planner input row {index}")
        if iid in rank:
            raise AtomicWanStreamError(f"duplicate planner input IID: {iid}")
        rank[iid] = index
    return rows, lines, rank


def admit_batch(args: argparse.Namespace) -> int:
    planner_input = args.planner_input.expanduser().resolve(strict=True)
    planner_root = args.planner_root.expanduser().resolve(strict=True)
    atomic_manifest = args.atomic_manifest.expanduser().resolve(strict=True)
    progress_path = args.progress.expanduser().resolve(strict=True)
    wan_root = args.wan_root.expanduser().resolve(strict=True)
    output_input = args.output_input.expanduser().resolve()
    output_receipt = args.output_receipt.expanduser().resolve()
    admission_root = args.admission_root.expanduser().resolve(strict=True)
    wan_batch_root = args.wan_batch_root.expanduser().resolve()
    _require_dir(planner_root, context="planner root")
    _require_dir(wan_root, context="Wan stream root")
    _require_dir(admission_root, context="Wan admission root")
    _require_dir(output_input.parent, context="Wan input directory")
    _require_dir(output_receipt.parent, context="Wan admission batch directory")
    if not wan_batch_root.is_absolute() or wan_batch_root.parent != wan_root / "batches":
        raise AtomicWanStreamError("Wan batch root is outside the stream batch directory")
    match = _TAG_RE.fullmatch(args.batch_tag)
    if match is None:
        raise AtomicWanStreamError("unsafe Wan batch tag")
    batch_index = int(match.group(1))

    _, planner_lines, parent_rank = _planner_rows(
        planner_input, args.planner_input_sha256
    )
    progress = _validate_progress(progress_path, atomic_manifest)
    atomic_rows_raw, _ = _load_jsonl(
        atomic_manifest, context="atomic manifest", allow_empty=True
    )
    atomic_rows = [_validate_atomic_row(row) for row in atomic_rows_raw]
    atomic_iids = [row["iid"] for row in atomic_rows]
    if (
        len(atomic_iids) != len(set(atomic_iids))
        or any(iid not in parent_rank for iid in atomic_iids)
        or atomic_iids != sorted(atomic_iids, key=parent_rank.__getitem__)
        or any(row["original_candidate_index"] != parent_rank[row["iid"]] for row in atomic_rows)
        or progress.get("atomic_ok_rows") != len(atomic_rows)
    ):
        raise AtomicWanStreamError("atomic manifest is not the verified parent-order subset")

    previous_path: Path | None = None
    previous_iids: list[str] = []
    if args.previous_admission_batch is not None:
        previous_path = args.previous_admission_batch.expanduser().resolve(strict=True)
        previous = _validate_admission_batch(previous_path)
        if previous["batch_index"] + 1 != batch_index:
            raise AtomicWanStreamError("Wan admission batches are not contiguous")
        previous_iids = list(previous["cumulative_iids"])
        if atomic_iids[: len(previous_iids)] != previous_iids:
            raise AtomicWanStreamError("atomic pass prefix regressed before Wan admission")
    elif batch_index != 0:
        raise AtomicWanStreamError("nonzero Wan admission batch lacks its predecessor")

    delta_rows = atomic_rows[len(previous_iids) :]
    delta_iids = [row["iid"] for row in delta_rows]
    delta_indices = [parent_rank[iid] for iid in delta_iids]
    payload = b"".join(planner_lines[index] for index in delta_indices)
    _publish(output_input, payload, resume=args.resume)

    batch: dict[str, Any] = {
        "schema_version": ADMISSION_BATCH_SCHEMA,
        "batch_tag": args.batch_tag,
        "batch_index": batch_index,
        "progress_receipt": str(progress_path),
        "progress_receipt_sha256": _sha_file(progress_path),
        "atomic_manifest": str(atomic_manifest),
        "atomic_manifest_sha256": _sha_file(atomic_manifest),
        "planner_input": str(planner_input),
        "planner_input_sha256": args.planner_input_sha256,
        "planner_root": str(planner_root),
        "wan_input": str(output_input),
        "wan_input_sha256": _sha_bytes(payload),
        "wan_batch_root": str(wan_batch_root),
        "previous_admission_batch": str(previous_path) if previous_path else None,
        "previous_admission_batch_sha256": _sha_file(previous_path) if previous_path else None,
        "batch_iids": delta_iids,
        "batch_parent_indices": delta_indices,
        "batch_rows": len(delta_iids),
        "cumulative_iids": atomic_iids,
        "cumulative_rows": len(atomic_iids),
        "dispatch_policy": "immediate_verified_atomic_suffix_parent_order",
        "generation_prompt_field": "planner_passed.edit_instruction",
        "primary_training_label_field": "atomic_action_instruction",
        "receipt_digest": None,
    }
    batch["receipt_digest"] = _object_digest(batch, omit="receipt_digest")
    batch_raw = _pretty(batch)
    _publish(output_receipt, batch_raw, resume=args.resume)
    receipt_sha = _sha_bytes(batch_raw)

    try:
        from motive.goku_atomic_motion_qwen import validate_passed_row
    except Exception as error:  # pragma: no cover - import failure is runtime-specific
        raise AtomicWanStreamError(f"atomic planner adapter import failed: {error}") from error
    atomic_by_iid = {row["iid"]: row for row in delta_rows}
    for iid in delta_iids:
        atomic = atomic_by_iid[iid]
        passed_path = planner_root / "passed" / f"{iid}.jsonl"
        passed_rows, passed_lines = _load_jsonl(
            passed_path, context=f"planner passed iid={iid}"
        )
        if len(passed_rows) != 1:
            raise AtomicWanStreamError(f"planner passed fragment is not one row iid={iid}")
        passed = validate_passed_row(passed_rows[0])
        prompt = _text(passed.get("edit_instruction"), context=f"planner prompt iid={iid}")
        prompt_sha = _sha(passed.get("edit_instruction_sha256"), context=f"planner prompt SHA iid={iid}")
        provenance = atomic["source_generation_provenance"]
        if (
            passed.get("iid") != iid
            or _sha_bytes(prompt.encode("utf-8")) != prompt_sha
            or prompt != provenance.get("frame_gridded_prompt")
            or prompt_sha != provenance.get("frame_gridded_prompt_sha256")
            or passed.get("resolved_source_video") != atomic.get("source_video")
            or passed.get("source_video_sha256") != atomic.get("source_video_sha256")
        ):
            raise AtomicWanStreamError(f"planner/atomic generation binding differs iid={iid}")
        source = Path(str(passed["resolved_source_video"]))
        _require_file(source, context=f"source video iid={iid}")
        if _sha_file(source) != passed["source_video_sha256"]:
            raise AtomicWanStreamError(f"source video bytes differ iid={iid}")
        admission: dict[str, Any] = {
            "schema_version": ADMISSION_SCHEMA,
            "iid": iid,
            "batch_tag": args.batch_tag,
            "batch_index": batch_index,
            "parent_index": parent_rank[iid],
            "batch_admission_receipt": str(output_receipt),
            "batch_admission_receipt_sha256": receipt_sha,
            "planner_passed": str(passed_path),
            "planner_passed_sha256": _sha_bytes(b"".join(passed_lines)),
            "atomic_result": atomic["result_path"],
            "atomic_result_sha256": atomic["result_sha256"],
            "source_video": str(source),
            "source_video_sha256": passed["source_video_sha256"],
            "wan_generation_prompt": prompt,
            "wan_generation_prompt_sha256": prompt_sha,
            "atomic_action_instruction": atomic["atomic_action_instruction"],
            "atomic_action_instruction_sha256": atomic["atomic_action_instruction_sha256"],
            "camera_instruction": atomic["camera_instruction"],
            "camera_instruction_sha256": atomic["camera_instruction_sha256"],
            "preservation_instruction": atomic["preservation_instruction"],
            "preservation_instruction_sha256": atomic["preservation_instruction_sha256"],
            "full_edit_instruction": atomic["full_edit_instruction"],
            "full_edit_instruction_sha256": atomic["full_edit_instruction_sha256"],
            "primary_training_label_field": "atomic_action_instruction",
            "wan_prompt_role": "planner_private_trajectory_generation_only",
            "admission_digest": None,
        }
        admission["admission_digest"] = _object_digest(
            admission, omit="admission_digest"
        )
        _publish(admission_root / f"{iid}.json", _pretty(admission), resume=args.resume)

    # Validate after publishing so a crash between the batch receipt and the
    # per-IID receipts is repaired (only with --resume) rather than dispatched.
    observed = _validate_admission_batch(output_receipt)
    if observed != batch:
        raise AtomicWanStreamError("published Wan admission batch differs")
    for iid in delta_iids:
        _validate_admission(admission_root / f"{iid}.json", batch=batch)
    print(
        f"{len(delta_iids)}\t{output_input}\t{output_receipt}\t{wan_batch_root}",
        flush=True,
    )
    return 0


def _validate_v16_terminal(batch: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(batch["wan_batch_root"])
    terminal_path = root / "watcher_terminal.json"
    contract_path = root / "watch_contract.json"
    terminal = _load_json(terminal_path, context="v16 Wan watcher terminal")
    required = {
        "schema_version",
        "status",
        "watch_contract_sha256",
        "expected_iids",
        "qwen_ok_iids",
        "qwen_error_iids",
        "wan_success_iids",
        "wan_error_iids",
        "completed_at_utc",
        "terminal_digest",
    }
    if set(terminal) != required or terminal.get("schema_version") != V16_TERMINAL_SCHEMA:
        raise AtomicWanStreamError("v16 Wan terminal schema differs")
    if terminal.get("terminal_digest") != _object_digest(terminal, omit="terminal_digest"):
        raise AtomicWanStreamError("v16 Wan terminal digest differs")
    _require_file(contract_path, context="v16 Wan watch contract")
    if terminal.get("watch_contract_sha256") != _sha_file(contract_path):
        raise AtomicWanStreamError("v16 Wan terminal contract binding differs")
    expected = list(batch["batch_iids"])
    success = terminal.get("wan_success_iids")
    errors = terminal.get("wan_error_iids")
    if (
        terminal.get("expected_iids") != expected
        or terminal.get("qwen_ok_iids") != expected
        or terminal.get("qwen_error_iids") != []
        or not isinstance(success, list)
        or not isinstance(errors, list)
        or set(success).intersection(errors)
        or set(success + errors) != set(expected)
        or [iid for iid in expected if iid in set(success)] != success
        or [iid for iid in expected if iid in set(errors)] != errors
        or terminal.get("status")
        != ("complete" if not errors else "complete_with_wan_errors")
    ):
        raise AtomicWanStreamError("v16 Wan terminal partition/order differs")
    return terminal


def _write_sample_metadata(
    *,
    sample: Path,
    admission_path: Path,
    admission: Mapping[str, Any],
    resume: bool,
) -> tuple[Path, str]:
    _require_dir(sample, context=f"Wan committed sample iid={admission['iid']}")
    source_candidates = [
        path for path in sample.iterdir() if _SOURCE_VIDEO_RE.fullmatch(path.name)
    ]
    if len(source_candidates) != 1:
        raise AtomicWanStreamError(f"Wan sample source-video closure differs iid={admission['iid']}")
    source_copy = _require_file(source_candidates[0], context="copied source video")
    preview = _require_file(sample / "preview.mp4", context="Wan target preview")
    result = _require_file(sample / "result.json", context="Wan sample result")
    edit_file = _require_file(sample / "edit_instruction.txt", context="Wan prompt file")
    if (
        _sha_file(source_copy) != admission["source_video_sha256"]
        or edit_file.read_bytes() != admission["wan_generation_prompt"].encode("utf-8")
        or _sha_file(edit_file) != admission["wan_generation_prompt_sha256"]
    ):
        raise AtomicWanStreamError(f"Wan copied input bytes differ iid={admission['iid']}")

    sidecars: dict[str, bytes] = {
        "wan_generation_prompt.txt": admission["wan_generation_prompt"].encode("utf-8"),
        "atomic_action_instruction.txt": admission["atomic_action_instruction"].encode("utf-8"),
        "camera_instruction.txt": admission["camera_instruction"].encode("utf-8"),
        "preservation_instruction.txt": admission["preservation_instruction"].encode("utf-8"),
        "full_edit_instruction.txt": admission["full_edit_instruction"].encode("utf-8"),
        "planner_passed.jsonl": Path(admission["planner_passed"]).read_bytes(),
        "atomic_result.json": Path(admission["atomic_result"]).read_bytes(),
        "atomic_admission.json": admission_path.read_bytes(),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, raw in sidecars.items():
        target = sample / name
        _publish(target, raw, resume=resume)
        artifacts[name] = {
            "path": str(target),
            "sha256": _sha_bytes(raw),
            "bytes": len(raw),
        }
    artifacts.update(
        {
            source_copy.name: {
                "path": str(source_copy),
                "sha256": _sha_file(source_copy),
                "bytes": source_copy.stat().st_size,
            },
            "edit_instruction.txt": {
                "path": str(edit_file),
                "sha256": _sha_file(edit_file),
                "bytes": edit_file.stat().st_size,
            },
            "preview.mp4": {
                "path": str(preview),
                "sha256": _sha_file(preview),
                "bytes": preview.stat().st_size,
            },
            "result.json": {
                "path": str(result),
                "sha256": _sha_file(result),
                "bytes": result.stat().st_size,
            },
        }
    )
    metadata: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA,
        "iid": admission["iid"],
        "primary_training_label_field": "atomic_action_instruction",
        "wan_generation_prompt_field": "planner_passed.edit_instruction",
        "wan_generation_prompt_is_training_label": False,
        "edit_instruction_txt_role": "generation_only_not_training_label",
        "wan_generation_prompt_txt_role": "generation_only_not_training_label",
        "atomic_action_instruction": admission["atomic_action_instruction"],
        "atomic_action_instruction_sha256": admission["atomic_action_instruction_sha256"],
        "camera_instruction": admission["camera_instruction"],
        "camera_instruction_sha256": admission["camera_instruction_sha256"],
        "preservation_instruction": admission["preservation_instruction"],
        "preservation_instruction_sha256": admission["preservation_instruction_sha256"],
        "full_edit_instruction": admission["full_edit_instruction"],
        "full_edit_instruction_sha256": admission["full_edit_instruction_sha256"],
        "wan_generation_prompt": admission["wan_generation_prompt"],
        "wan_generation_prompt_sha256": admission["wan_generation_prompt_sha256"],
        "source_video_sha256": admission["source_video_sha256"],
        "artifacts": artifacts,
        "metadata_digest": None,
    }
    metadata["metadata_digest"] = _object_digest(metadata, omit="metadata_digest")
    target = sample / "atomic_sample_metadata.json"
    raw = _pretty(metadata)
    _publish(target, raw, resume=resume)
    return target, _sha_bytes(raw)


def materialize_metadata(args: argparse.Namespace) -> int:
    batch_path = args.admission_batch.expanduser().resolve(strict=True)
    admission_root = args.admission_root.expanduser().resolve(strict=True)
    batch = _validate_admission_batch(batch_path)
    if batch["batch_rows"] < 1:
        raise AtomicWanStreamError("an empty admission batch has no Wan metadata phase")
    terminal = _validate_v16_terminal(batch)
    batch_root = Path(batch["wan_batch_root"])
    records: list[dict[str, Any]] = []
    success_set = set(terminal["wan_success_iids"])
    for iid in batch["batch_iids"]:
        admission_path = admission_root / f"{iid}.json"
        admission = _validate_admission(admission_path, batch=batch)
        if iid in success_set:
            sample = batch_root / "samples" / iid / "samples" / iid
            metadata, metadata_sha = _write_sample_metadata(
                sample=sample,
                admission_path=admission_path,
                admission=admission,
                resume=args.resume,
            )
            records.append(
                {
                    "iid": iid,
                    "status": "success",
                    "admission": str(admission_path),
                    "admission_sha256": _sha_file(admission_path),
                    "sample_dir": str(sample),
                    "sample_metadata": str(metadata),
                    "sample_metadata_sha256": metadata_sha,
                }
            )
        else:
            records.append(
                {
                    "iid": iid,
                    "status": "error",
                    "admission": str(admission_path),
                    "admission_sha256": _sha_file(admission_path),
                    "sample_dir": None,
                    "sample_metadata": None,
                    "sample_metadata_sha256": None,
                }
            )
    value: dict[str, Any] = {
        "schema_version": METADATA_BATCH_SCHEMA,
        "batch_tag": batch["batch_tag"],
        "admission_batch": str(batch_path),
        "admission_batch_sha256": _sha_file(batch_path),
        "watcher_terminal": str(batch_root / "watcher_terminal.json"),
        "watcher_terminal_sha256": _sha_file(batch_root / "watcher_terminal.json"),
        "expected_iids": list(batch["batch_iids"]),
        "wan_success_iids": list(terminal["wan_success_iids"]),
        "wan_error_iids": list(terminal["wan_error_iids"]),
        "records": records,
        "metadata_batch_digest": None,
    }
    value["metadata_batch_digest"] = _object_digest(
        value, omit="metadata_batch_digest"
    )
    output = batch_root / "atomic_metadata_status.json"
    _publish(output, _pretty(value), resume=args.resume)
    print(
        f"{len(terminal['wan_success_iids'])}\t{len(terminal['wan_error_iids'])}\t{output}",
        flush=True,
    )
    return 0


def _admission_chain(latest: Path) -> list[tuple[Path, dict[str, Any]]]:
    chain: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    path: Path | None = latest
    while path is not None:
        path = path.resolve(strict=True)
        if path in seen:
            raise AtomicWanStreamError("Wan admission batch chain contains a cycle")
        seen.add(path)
        value = _validate_admission_batch(path)
        chain.append((path, value))
        previous = value["previous_admission_batch"]
        if previous is None:
            path = None
        else:
            previous_path = Path(previous)
            if _sha_file(previous_path) != value["previous_admission_batch_sha256"]:
                raise AtomicWanStreamError("previous Wan admission batch bytes differ")
            path = previous_path
    chain.reverse()
    if [value["batch_index"] for _, value in chain] != list(range(len(chain))):
        raise AtomicWanStreamError("Wan admission batch chain indices are not contiguous")
    prior: list[str] = []
    for _, value in chain:
        if value["cumulative_iids"][: len(prior)] != prior:
            raise AtomicWanStreamError("Wan cumulative admission prefix regressed")
        if value["cumulative_iids"][len(prior) :] != value["batch_iids"]:
            raise AtomicWanStreamError("Wan admission delta differs from cumulative suffix")
        prior = list(value["cumulative_iids"])
    return chain


def _stream_records(latest: Path) -> tuple[list[str], list[dict[str, Any]]]:
    chain = _admission_chain(latest)
    records: list[dict[str, Any]] = []
    for batch_path, batch in chain:
        if batch["batch_rows"] == 0:
            continue
        status_path = Path(batch["wan_batch_root"]) / "atomic_metadata_status.json"
        status = _load_json(status_path, context="Wan atomic metadata batch")
        if (
            status.get("schema_version") != METADATA_BATCH_SCHEMA
            or status.get("metadata_batch_digest")
            != _object_digest(status, omit="metadata_batch_digest")
            or status.get("admission_batch") != str(batch_path)
            or status.get("admission_batch_sha256") != _sha_file(batch_path)
            or status.get("expected_iids") != batch["batch_iids"]
        ):
            raise AtomicWanStreamError("Wan atomic metadata batch binding differs")
        batch_records = status.get("records")
        if (
            not isinstance(batch_records, list)
            or [row.get("iid") for row in batch_records] != batch["batch_iids"]
            or any(row.get("status") not in {"success", "error"} for row in batch_records)
        ):
            raise AtomicWanStreamError("Wan atomic metadata records differ")
        for row in batch_records:
            admission_path = Path(str(row.get("admission", "")))
            admission = _validate_admission(admission_path, batch=batch)
            if row.get("admission_sha256") != _sha_file(admission_path):
                raise AtomicWanStreamError("Wan stream admission hash differs")
            if row["status"] == "success":
                metadata_path = Path(str(row.get("sample_metadata", "")))
                _require_file(metadata_path, context="Wan sample metadata")
                if row.get("sample_metadata_sha256") != _sha_file(metadata_path):
                    raise AtomicWanStreamError("Wan sample metadata bytes differ")
                sample_dir = Path(str(row.get("sample_dir", "")))
                _require_dir(sample_dir, context="Wan sample directory")
            records.append({**row, "batch_tag": batch["batch_tag"], "admission_value": admission})
    expected = list(chain[-1][1]["cumulative_iids"])
    if [row["iid"] for row in records] != expected:
        raise AtomicWanStreamError("Wan stream records do not close cumulative admissions")
    return expected, records


def inspect_stream(args: argparse.Namespace) -> int:
    latest = args.latest_admission_batch.expanduser().resolve(strict=True)
    expected, records = _stream_records(latest)
    success = sum(row["status"] == "success" for row in records)
    errors = sum(row["status"] == "error" for row in records)
    print(f"{len(expected)}\t{success}\t{errors}", flush=True)
    return 0


def publish_terminal(args: argparse.Namespace) -> int:
    latest = args.latest_admission_batch.expanduser().resolve(strict=True)
    atomic_manifest = args.atomic_manifest.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    atomic_rows, _ = _load_jsonl(
        atomic_manifest, context="final atomic manifest", allow_empty=True
    )
    atomic_iids = [_validate_atomic_row(row)["iid"] for row in atomic_rows]
    expected, records = _stream_records(latest)
    if atomic_iids != expected:
        raise AtomicWanStreamError("final atomic manifest differs from Wan admissions")
    success_iids = [row["iid"] for row in records if row["status"] == "success"]
    error_iids = [row["iid"] for row in records if row["status"] == "error"]
    public_records = [
        {key: value for key, value in row.items() if key != "admission_value"}
        for row in records
    ]
    value: dict[str, Any] = {
        "schema_version": STREAM_TERMINAL_SCHEMA,
        "status": "complete" if not error_iids else "complete_with_wan_errors",
        "latest_admission_batch": str(latest),
        "latest_admission_batch_sha256": _sha_file(latest),
        "atomic_manifest": str(atomic_manifest),
        "atomic_manifest_sha256": _sha_file(atomic_manifest),
        "expected_iids": expected,
        "wan_success_iids": success_iids,
        "wan_error_iids": error_iids,
        "records": public_records,
        "terminal_digest": None,
    }
    value["terminal_digest"] = _object_digest(value, omit="terminal_digest")
    _publish(output, _pretty(value), resume=args.resume)
    print(f"{len(expected)}\t{len(success_iids)}\t{len(error_iids)}\t{output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    admit = commands.add_parser("admit-batch")
    admit.add_argument("--planner-input", type=Path, required=True)
    admit.add_argument("--planner-input-sha256", required=True)
    admit.add_argument("--planner-root", type=Path, required=True)
    admit.add_argument("--atomic-manifest", type=Path, required=True)
    admit.add_argument("--progress", type=Path, required=True)
    admit.add_argument("--wan-root", type=Path, required=True)
    admit.add_argument("--wan-batch-root", type=Path, required=True)
    admit.add_argument("--admission-root", type=Path, required=True)
    admit.add_argument("--output-input", type=Path, required=True)
    admit.add_argument("--output-receipt", type=Path, required=True)
    admit.add_argument("--batch-tag", required=True)
    admit.add_argument("--previous-admission-batch", type=Path)
    admit.add_argument("--resume", action="store_true")
    admit.set_defaults(func=admit_batch)

    metadata = commands.add_parser("materialize-metadata")
    metadata.add_argument("--admission-batch", type=Path, required=True)
    metadata.add_argument("--admission-root", type=Path, required=True)
    metadata.add_argument("--resume", action="store_true")
    metadata.set_defaults(func=materialize_metadata)

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--latest-admission-batch", type=Path, required=True)
    inspect.set_defaults(func=inspect_stream)

    terminal = commands.add_parser("publish-terminal")
    terminal.add_argument("--latest-admission-batch", type=Path, required=True)
    terminal.add_argument("--atomic-manifest", type=Path, required=True)
    terminal.add_argument("--output", type=Path, required=True)
    terminal.add_argument("--resume", action="store_true")
    terminal.set_defaults(func=publish_terminal)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except AtomicWanStreamError as error:
        print(f"[atomic-wan-stream] ERROR: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
