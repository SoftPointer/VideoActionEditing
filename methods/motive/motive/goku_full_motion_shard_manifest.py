"""Materialize a verified full-motion primary manifest into contiguous shards.

The root manifest remains the authorization object.  This stage copies its
canonical row bytes into exactly 32 consecutive eight-row manifests without
reordering, filtering, or reserializing them.  The resulting descriptors are
audit evidence only; the signed-release verifier independently proves that a
submitted shard is one unique contiguous slice of the signed root.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

from . import goku_full_motion_finalize as _finalizer


SUMMARY_SCHEMA = "motive-goku-full-motion-shard-manifest-v1"
DONE_SCHEMA = "motive-goku-full-motion-shard-manifest-done-v1"
POLICY_VERSION = "full-motion-primary256-contiguous8-v1"
ROOT_ROWS = 256
ROWS_PER_SHARD = 8
SHARD_COUNT = 32

PRIMARY_NAME = "primary_256.jsonl"
RESERVE_NAME = "reserve_64.jsonl"
REVIEW_NAME = "review_candidates.jsonl"
PARENT_SUMMARY_NAME = "summary.json"
PARENT_DONE_NAME = "done.json"
SHARDS_NAME = "shards"
JOBS_NAME = "jobs.tsv"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"

_PARENT_ENTRIES = {
    PRIMARY_NAME,
    RESERVE_NAME,
    REVIEW_NAME,
    PARENT_SUMMARY_NAME,
    PARENT_DONE_NAME,
}
_PARENT_DONE_KEYS = {
    "schema_version",
    "status",
    "policy_version",
    "implementation",
    "implementation_digest",
    "inputs",
    "input_digest",
    "config",
    "artifacts",
    "artifact_digest",
    "done_digest",
}
_PARENT_SUMMARY_KEYS = {
    "schema_version",
    "status",
    "policy_version",
    "semantics",
    "config",
    "counts",
    "rejection_counts",
    "diversity",
    "selection",
    "implementation",
    "implementation_digest",
    "inputs",
    "input_digest",
    "artifacts",
}
_PARENT_CONFIG = {
    "qwen_shards": 8,
    "primary_size": 256,
    "reserve_size": 64,
    "min_primary_multi_dynamic": 64,
    "target_signature_cap": 32,
    "family_cap": 32,
    "selection_order": "candidate_manifest_order_with_multi_unit_quota_first",
    "caps_apply_to": "primary_plus_reserve",
}
_ARTIFACT_METADATA_KEYS = {"sha256", "bytes", "rows"}
_CANDIDATE_INPUT_KEYS = {"path", "sha256", "bytes", "rows"}
_QWEN_INPUT_KEYS = {
    "shard_index",
    "output_path",
    "output_sha256",
    "output_bytes",
    "output_rows",
    "receipt_path",
    "receipt_sha256",
    "receipt_bytes",
    "receipt_digest",
    "assigned_iids",
}


class GokuFullMotionShardManifestError(RuntimeError):
    """The parent finalizer or requested contiguous shard set is invalid."""


def _reject_constant(value: str) -> None:
    raise GokuFullMotionShardManifestError(
        f"non-finite JSON constant: {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GokuFullMotionShardManifestError(
                f"duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GokuFullMotionShardManifestError(
            f"{context} is not UTF-8"
        ) from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, GokuFullMotionShardManifestError):
            raise
        raise GokuFullMotionShardManifestError(
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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_digest(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _ordered_digest(values: Sequence[str]) -> str:
    return _sha256(
        b"".join(value.encode("utf-8") + b"\n" for value in values)
    )


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    if set(value) != expected:
        raise GokuFullMotionShardManifestError(
            f"{context} keys differ: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )


def _stable_read(path: Path, *, context: str) -> bytes:
    unresolved = Path(os.path.abspath(path.expanduser()))
    if unresolved.is_symlink() or not unresolved.is_file():
        raise GokuFullMotionShardManifestError(
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
        raise GokuFullMotionShardManifestError(
            f"{context} changed while it was read: {unresolved}"
        )
    return raw


def _strict_object(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    raw = _stable_read(path, context=context)
    value = _parse_json(raw, context=context)
    if not isinstance(value, dict):
        raise GokuFullMotionShardManifestError(
            f"{context} must contain one object"
        )
    return value, raw


def _strict_jsonl(
    path: Path,
    *,
    context: str,
    allow_empty: bool = False,
    require_canonical: bool = True,
) -> tuple[list[dict[str, Any]], list[bytes], bytes]:
    raw = _stable_read(path, context=context)
    if not raw:
        if allow_empty:
            return [], [], raw
        raise GokuFullMotionShardManifestError(f"{context} is empty")
    if not raw.endswith(b"\n"):
        raise GokuFullMotionShardManifestError(
            f"{context} must be newline terminated"
        )
    rows: list[dict[str, Any]] = []
    lines: list[bytes] = []
    for line_number, bare in enumerate(raw.splitlines(), start=1):
        if not bare:
            raise GokuFullMotionShardManifestError(
                f"{context} has a blank line at {line_number}"
            )
        value = _parse_json(bare, context=f"{context}:{line_number}")
        if not isinstance(value, dict):
            raise GokuFullMotionShardManifestError(
                f"{context}:{line_number} is not an object"
            )
        if require_canonical and bare != _canonical_bytes(value):
            raise GokuFullMotionShardManifestError(
                f"{context}:{line_number} is not canonical JSON"
            )
        rows.append(value)
        lines.append(bare + b"\n")
    return rows, lines, raw


def _text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GokuFullMotionShardManifestError(
            f"{context} must be non-empty text"
        )
    return value


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GokuFullMotionShardManifestError(
            f"{context} must be an integer >= {minimum}"
        )
    return value


def _digest(value: Any, *, context: str) -> str:
    text = _text(value, context=context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise GokuFullMotionShardManifestError(
            f"{context} is not a lowercase SHA-256"
        )
    return text


def _resolve_bound_file(
    metadata: Mapping[str, Any],
    *,
    path_field: str,
    sha_field: str,
    bytes_field: str,
    context: str,
) -> tuple[Path, bytes]:
    path_value = _text(metadata.get(path_field), context=f"{context}.{path_field}")
    path = Path(path_value)
    if not path.is_absolute():
        raise GokuFullMotionShardManifestError(
            f"{context}.{path_field} must be absolute"
        )
    raw = _stable_read(path, context=context)
    if _sha256(raw) != _digest(
        metadata.get(sha_field), context=f"{context}.{sha_field}"
    ):
        raise GokuFullMotionShardManifestError(f"{context} SHA differs")
    if len(raw) != _integer(
        metadata.get(bytes_field), context=f"{context}.{bytes_field}"
    ):
        raise GokuFullMotionShardManifestError(f"{context} byte count differs")
    return path.resolve(strict=True), raw


def _validate_parent_inputs(inputs: Any) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise GokuFullMotionShardManifestError(
            "finalizer inputs must be an object"
        )
    _exact_keys(
        inputs, {"candidate_manifest", "qwen_shards"}, context="finalizer inputs"
    )
    candidate = inputs.get("candidate_manifest")
    if not isinstance(candidate, Mapping):
        raise GokuFullMotionShardManifestError(
            "candidate input binding is malformed"
        )
    _exact_keys(candidate, _CANDIDATE_INPUT_KEYS, context="candidate input")
    candidate_path, candidate_raw = _resolve_bound_file(
        candidate,
        path_field="path",
        sha_field="sha256",
        bytes_field="bytes",
        context="candidate manifest",
    )
    candidate_rows, _candidate_lines, _ = _strict_jsonl(
        candidate_path,
        context="candidate manifest",
        require_canonical=False,
    )
    if len(candidate_rows) != _integer(
        candidate.get("rows"), context="candidate input rows", minimum=1
    ):
        raise GokuFullMotionShardManifestError(
            "candidate input row count differs"
        )
    candidate_iids: list[str] = []
    for row in candidate_rows:
        iid = _text(row.get("iid"), context="candidate IID")
        if iid in candidate_iids:
            raise GokuFullMotionShardManifestError(
                f"duplicate candidate IID: {iid}"
            )
        candidate_iids.append(iid)

    shards = inputs.get("qwen_shards")
    if not isinstance(shards, list) or len(shards) != 8:
        raise GokuFullMotionShardManifestError(
            "finalizer must bind exactly eight Qwen shards"
        )
    assigned_all: list[str] = []
    normalized: list[dict[str, Any]] = []
    for expected_index, item in enumerate(shards):
        if not isinstance(item, Mapping):
            raise GokuFullMotionShardManifestError(
                "Qwen input descriptor is malformed"
            )
        _exact_keys(item, _QWEN_INPUT_KEYS, context="Qwen input descriptor")
        if item.get("shard_index") != expected_index:
            raise GokuFullMotionShardManifestError(
                "Qwen input shard indices are not exactly 0..7"
            )
        output_path, output_raw = _resolve_bound_file(
            item,
            path_field="output_path",
            sha_field="output_sha256",
            bytes_field="output_bytes",
            context=f"Qwen shard {expected_index} output",
        )
        output_rows, _output_lines, _ = _strict_jsonl(
            output_path,
            context=f"Qwen shard {expected_index} output",
            allow_empty=True,
            require_canonical=False,
        )
        assigned = item.get("assigned_iids")
        if (
            not isinstance(assigned, list)
            or any(not isinstance(iid, str) or not iid for iid in assigned)
            or len(set(assigned)) != len(assigned)
        ):
            raise GokuFullMotionShardManifestError(
                f"Qwen shard {expected_index} assigned IID list is malformed"
            )
        if (
            len(output_rows)
            != _integer(item.get("output_rows"), context="Qwen output rows")
            or [row.get("iid") for row in output_rows] != assigned
        ):
            raise GokuFullMotionShardManifestError(
                f"Qwen shard {expected_index} output coverage differs"
            )
        receipt_path, receipt_raw = _resolve_bound_file(
            item,
            path_field="receipt_path",
            sha_field="receipt_sha256",
            bytes_field="receipt_bytes",
            context=f"Qwen shard {expected_index} receipt",
        )
        receipt = _parse_json(receipt_raw, context="Qwen shard receipt")
        if not isinstance(receipt, dict):
            raise GokuFullMotionShardManifestError(
                "Qwen shard receipt must be an object"
            )
        stored_receipt_digest = _digest(
            item.get("receipt_digest"), context="Qwen receipt digest binding"
        )
        payload = dict(receipt)
        receipt_digest = payload.pop("receipt_digest", None)
        if (
            receipt_digest != stored_receipt_digest
            or _object_digest(payload) != stored_receipt_digest
            or receipt.get("status") != "complete"
            or receipt.get("shard_index") != expected_index
            or receipt.get("assigned_iids") != assigned
            or not isinstance(receipt.get("output"), Mapping)
            or receipt["output"].get("sha256") != _sha256(output_raw)
        ):
            raise GokuFullMotionShardManifestError(
                f"Qwen shard {expected_index} terminal receipt differs"
            )
        assigned_all.extend(assigned)
        normalized.append(dict(item))
    if len(set(assigned_all)) != len(assigned_all) or set(assigned_all) != set(
        candidate_iids
    ):
        raise GokuFullMotionShardManifestError(
            "Qwen shard union does not exactly cover candidate IIDs"
        )
    if len(candidate_raw) != candidate["bytes"]:
        raise AssertionError("candidate stable-read byte drift")
    return {
        "candidate_manifest": dict(candidate),
        "qwen_shards": normalized,
    }


def _validate_artifact_metadata(
    value: Any,
    *,
    expected_raw: bytes,
    expected_rows: int,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GokuFullMotionShardManifestError(f"{context} is malformed")
    _exact_keys(value, _ARTIFACT_METADATA_KEYS, context=context)
    expected = {
        "sha256": _sha256(expected_raw),
        "bytes": len(expected_raw),
        "rows": expected_rows,
    }
    if dict(value) != expected:
        raise GokuFullMotionShardManifestError(f"{context} differs")
    return expected


def _validate_generation_rows(
    rows: Sequence[dict[str, Any]],
    *,
    context: str,
    expected_bucket: str | None,
) -> None:
    seen_iids: set[str] = set()
    seen_groups: set[str] = set()
    prior_candidate_rank = 0
    prior_review_rank = 0
    for index, raw_row in enumerate(rows):
        try:
            row = _finalizer.validate_generation_row(raw_row)
        except Exception as error:
            raise GokuFullMotionShardManifestError(
                f"{context} row {index} validation failed: {error}"
            ) from error
        iid = _text(row.get("iid"), context=f"{context} IID")
        group = _text(row.get("group_id"), context=f"{context} group_id")
        if iid in seen_iids or group in seen_groups:
            raise GokuFullMotionShardManifestError(
                f"{context} IID/group_id values are not unique"
            )
        finalization = row["full_motion_finalization"]
        bucket = finalization["selection_bucket"]
        if expected_bucket is not None and bucket != expected_bucket:
            raise GokuFullMotionShardManifestError(
                f"{context} row {iid} selection bucket differs"
            )
        candidate_rank = int(finalization["candidate_rank"])
        review_rank = int(finalization["review_rank"])
        if candidate_rank <= prior_candidate_rank or review_rank <= prior_review_rank:
            raise GokuFullMotionShardManifestError(
                f"{context} is not in candidate/review order"
            )
        prior_candidate_rank = candidate_rank
        prior_review_rank = review_rank
        seen_iids.add(iid)
        seen_groups.add(group)


def _diversity_counts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    signatures: dict[str, int] = {}
    families: dict[str, int] = {}
    for row in rows:
        finalization = row["full_motion_finalization"]
        family = str(row["family"]).casefold()
        families[family] = families.get(family, 0) + 1
        unique_signatures = dict.fromkeys(
            str(value).casefold()
            for value in finalization["target_action_signatures"]
        )
        for signature in unique_signatures:
            signatures[signature] = signatures.get(signature, 0) + 1
    return dict(sorted(signatures.items())), dict(sorted(families.items()))


def _validate_parent(
    finalizer_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[bytes],
    bytes,
    dict[str, Any],
]:
    unresolved = Path(os.path.abspath(finalizer_dir.expanduser()))
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise GokuFullMotionShardManifestError(
            "finalizer_dir must be a non-symlink directory"
        )
    root = unresolved.resolve(strict=True)
    if {entry.name for entry in root.iterdir()} != _PARENT_ENTRIES:
        raise GokuFullMotionShardManifestError(
            "finalizer directory artifact closure differs"
        )

    done, done_raw = _strict_object(root / PARENT_DONE_NAME, context="finalizer done")
    _exact_keys(done, _PARENT_DONE_KEYS, context="finalizer done")
    if (
        done.get("schema_version") != _finalizer.DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("policy_version") != _finalizer.POLICY_VERSION
    ):
        raise GokuFullMotionShardManifestError("finalizer done identity differs")
    done_payload = dict(done)
    stored_done_digest = done_payload.pop("done_digest", None)
    if _digest(stored_done_digest, context="finalizer done digest") != _object_digest(
        done_payload
    ):
        raise GokuFullMotionShardManifestError("finalizer done digest differs")
    implementation = done.get("implementation")
    if not isinstance(implementation, Mapping):
        raise GokuFullMotionShardManifestError(
            "finalizer implementation binding is malformed"
        )
    current_implementation = _finalizer._implementation_bundle()
    if (
        dict(implementation) != current_implementation
        or done.get("implementation_digest") != _object_digest(implementation)
    ):
        raise GokuFullMotionShardManifestError(
            "finalizer implementation closure differs from current bytes"
        )
    inputs = _validate_parent_inputs(done.get("inputs"))
    if done.get("input_digest") != _object_digest(inputs):
        raise GokuFullMotionShardManifestError("finalizer input digest differs")
    config = done.get("config")
    if not isinstance(config, Mapping):
        raise GokuFullMotionShardManifestError("finalizer config is malformed")
    required_iids = config.get("required_iids")
    if (
        {key: config.get(key) for key in _PARENT_CONFIG} != _PARENT_CONFIG
        or not isinstance(required_iids, list)
        or _finalizer.DEFAULT_CANARY_IID not in required_iids
        or len(set(required_iids)) != len(required_iids)
    ):
        raise GokuFullMotionShardManifestError(
            "finalizer config is not strict primary256/reserve64"
        )

    primary_rows, primary_lines, primary_raw = _strict_jsonl(
        root / PRIMARY_NAME, context=PRIMARY_NAME
    )
    reserve_rows, _reserve_lines, reserve_raw = _strict_jsonl(
        root / RESERVE_NAME, context=RESERVE_NAME
    )
    review_rows, _review_lines, review_raw = _strict_jsonl(
        root / REVIEW_NAME, context=REVIEW_NAME
    )
    if len(primary_rows) != ROOT_ROWS or len(reserve_rows) != 64:
        raise GokuFullMotionShardManifestError(
            "finalizer primary/reserve row counts differ"
        )
    _validate_generation_rows(
        primary_rows, context=PRIMARY_NAME, expected_bucket="primary"
    )
    _validate_generation_rows(
        reserve_rows, context=RESERVE_NAME, expected_bucket="reserve"
    )
    _validate_generation_rows(
        review_rows, context=REVIEW_NAME, expected_bucket=None
    )
    primary_iids = [str(row["iid"]) for row in primary_rows]
    reserve_iids = [str(row["iid"]) for row in reserve_rows]
    primary_groups = [str(row["group_id"]) for row in primary_rows]
    reserve_groups = [str(row["group_id"]) for row in reserve_rows]
    if (
        set(primary_iids) & set(reserve_iids)
        or set(primary_groups) & set(reserve_groups)
    ):
        raise GokuFullMotionShardManifestError(
            "finalizer primary/reserve identities overlap"
        )
    review_by_iid = {str(row["iid"]): row for row in review_rows}
    if len(review_by_iid) != len(review_rows):
        raise GokuFullMotionShardManifestError("review IID values are not unique")
    for row in [*primary_rows, *reserve_rows]:
        if review_by_iid.get(str(row["iid"])) != row:
            raise GokuFullMotionShardManifestError(
                "primary/reserve row differs from review_candidates"
            )

    artifacts = done.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        PRIMARY_NAME,
        RESERVE_NAME,
        REVIEW_NAME,
        PARENT_SUMMARY_NAME,
    }:
        raise GokuFullMotionShardManifestError(
            "finalizer done artifact set differs"
        )
    summary, summary_raw = _strict_object(
        root / PARENT_SUMMARY_NAME, context="finalizer summary"
    )
    _validate_artifact_metadata(
        artifacts[PRIMARY_NAME],
        expected_raw=primary_raw,
        expected_rows=ROOT_ROWS,
        context="finalizer primary artifact",
    )
    _validate_artifact_metadata(
        artifacts[RESERVE_NAME],
        expected_raw=reserve_raw,
        expected_rows=64,
        context="finalizer reserve artifact",
    )
    _validate_artifact_metadata(
        artifacts[REVIEW_NAME],
        expected_raw=review_raw,
        expected_rows=len(review_rows),
        context="finalizer review artifact",
    )
    _validate_artifact_metadata(
        artifacts[PARENT_SUMMARY_NAME],
        expected_raw=summary_raw,
        expected_rows=1,
        context="finalizer summary artifact",
    )
    if done.get("artifact_digest") != _object_digest(artifacts):
        raise GokuFullMotionShardManifestError(
            "finalizer artifact aggregate digest differs"
        )

    _exact_keys(summary, _PARENT_SUMMARY_KEYS, context="finalizer summary")
    if (
        summary.get("schema_version") != _finalizer.FINALIZE_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("policy_version") != _finalizer.POLICY_VERSION
        or summary.get("implementation") != implementation
        or summary.get("implementation_digest")
        != done.get("implementation_digest")
        or summary.get("inputs") != inputs
        or summary.get("input_digest") != done.get("input_digest")
        or summary.get("config") != config
    ):
        raise GokuFullMotionShardManifestError(
            "finalizer summary/done binding differs"
        )
    expected_summary_artifacts = {
        name: artifacts[name]
        for name in (PRIMARY_NAME, RESERVE_NAME, REVIEW_NAME)
    }
    if summary.get("artifacts") != expected_summary_artifacts:
        raise GokuFullMotionShardManifestError(
            "finalizer summary artifact bindings differ"
        )
    counts = summary.get("counts")
    selection = summary.get("selection")
    diversity = summary.get("diversity")
    if not all(isinstance(value, Mapping) for value in (counts, selection, diversity)):
        raise GokuFullMotionShardManifestError(
            "finalizer summary count/selection/diversity is malformed"
        )
    primary_multi = sum(
        row["full_motion_finalization"]["dynamic_unit_count"] >= 2
        for row in primary_rows
    )
    if (
        counts.get("primary_rows") != ROOT_ROWS
        or counts.get("reserve_rows") != 64
        or counts.get("review_rows") != len(review_rows)
        or counts.get("primary_multi_dynamic_rows") != primary_multi
        or primary_multi < 64
        or selection.get("primary_iids") != primary_iids
        or selection.get("reserve_iids") != reserve_iids
        or any(iid not in primary_iids for iid in required_iids)
        or selection.get("required_iids_in_primary") != required_iids
    ):
        raise GokuFullMotionShardManifestError(
            "finalizer selection/count closure differs"
        )
    signature_counts, family_counts = _diversity_counts(
        [*primary_rows, *reserve_rows]
    )
    if (
        diversity.get("primary_multi_dynamic_rows") != primary_multi
        or diversity.get("target_signature_counts") != signature_counts
        or diversity.get("family_counts") != family_counts
        or any(value > 32 for value in signature_counts.values())
        or any(value > 32 for value in family_counts.values())
    ):
        raise GokuFullMotionShardManifestError(
            "finalizer diversity closure differs"
        )
    parent = {
        "finalizer_dir": str(root),
        "done_path": str(root / PARENT_DONE_NAME),
        "done_sha256": _sha256(done_raw),
        "done_digest": stored_done_digest,
        "summary_path": str(root / PARENT_SUMMARY_NAME),
        "summary_sha256": _sha256(summary_raw),
        "implementation_digest": done["implementation_digest"],
        "input_digest": done["input_digest"],
        "artifact_digest": done["artifact_digest"],
        "primary_path": str(root / PRIMARY_NAME),
        "primary_sha256": _sha256(primary_raw),
        "primary_bytes": len(primary_raw),
        "primary_rows": ROOT_ROWS,
    }
    return primary_rows, primary_lines, primary_raw, parent


def _implementation_bundle() -> dict[str, str]:
    directory = Path(__file__).resolve(strict=True).parent
    files = {
        "shard_manifest": "goku_full_motion_shard_manifest.py",
        "finalizer": "goku_full_motion_finalize.py",
        "signed_release": "wan22_full_motion_signed_release.py",
    }
    return {
        key: _sha256(_stable_read(directory / name, context=f"{key} implementation"))
        for key, name in files.items()
    }


def _jobs_bytes(shards: Sequence[Mapping[str, Any]]) -> bytes:
    columns = [
        "shard_index",
        "shard_id",
        "manifest_relpath",
        "root_row_start_zero_based",
        "root_row_end_exclusive",
        "row_count",
        "manifest_sha256",
        "manifest_bytes",
        "ordered_iids_sha256",
        "ordered_row_sha256",
        "ordered_iids_json",
    ]
    lines = ["\t".join(columns)]
    for shard in shards:
        values = [
            str(shard["shard_index"]),
            str(shard["shard_id"]),
            str(shard["path"]),
            str(shard["root_row_start_zero_based"]),
            str(shard["root_row_end_exclusive"]),
            str(shard["rows"]),
            str(shard["sha256"]),
            str(shard["bytes"]),
            str(shard["ordered_iids_sha256"]),
            str(shard["ordered_row_sha256"]),
            json.dumps(shard["ordered_iids"], ensure_ascii=False, separators=(",", ":")),
        ]
        if any("\t" in value or "\n" in value for value in values):
            raise GokuFullMotionShardManifestError(
                "jobs.tsv value contains a tab or newline"
            )
        lines.append("\t".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


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
    else:  # pragma: no cover
        raise GokuFullMotionShardManifestError(
            "platform lacks atomic no-replace directory rename"
        )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(output)
    raise OSError(number, os.strerror(number), str(output))


def materialize_full_motion_shards(
    *, finalizer_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Verify one strict finalizer and atomically publish 32 contiguous shards."""

    rows, root_lines, root_raw, parent = _validate_parent(Path(finalizer_dir))
    output = Path(output_dir).expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise GokuFullMotionShardManifestError(
            "output parent is not a plain directory"
        )

    shard_raw: dict[str, bytes] = {}
    descriptors: list[dict[str, Any]] = []
    root_row_sha = [_object_digest(row) for row in rows]
    for shard_index in range(SHARD_COUNT):
        start = shard_index * ROWS_PER_SHARD
        end = start + ROWS_PER_SHARD
        name = f"shard_{shard_index:03d}.jsonl"
        relative = f"{SHARDS_NAME}/{name}"
        raw = b"".join(root_lines[start:end])
        iids = [str(row["iid"]) for row in rows[start:end]]
        descriptor = {
            "shard_index": shard_index,
            "shard_id": f"shard_{shard_index:03d}",
            "path": relative,
            "root_row_start_zero_based": start,
            "root_row_end_exclusive": end,
            "root_row_indices_zero_based": list(range(start, end)),
            "rows": ROWS_PER_SHARD,
            "bytes": len(raw),
            "sha256": _sha256(raw),
            "ordered_iids": iids,
            "ordered_iids_sha256": _ordered_digest(iids),
            "ordered_row_sha256": _ordered_digest(root_row_sha[start:end]),
        }
        shard_raw[relative] = raw
        descriptors.append(descriptor)
    if b"".join(shard_raw[item["path"]] for item in descriptors) != root_raw:
        raise AssertionError("contiguous shard bytes do not reconstruct root")

    jobs_raw = _jobs_bytes(descriptors)
    implementation = _implementation_bundle()
    implementation_digest = _object_digest(implementation)
    source = {
        **parent,
        "generation_schema": _finalizer.GENERATION_SCHEMA,
        "root_ordered_iids_sha256": _ordered_digest(
            [str(row["iid"]) for row in rows]
        ),
        "root_ordered_row_sha256": _ordered_digest(root_row_sha),
    }
    input_digest = _object_digest(source)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "policy_version": POLICY_VERSION,
        "authorization_semantics": {
            "root_manifest_is_authorization_object": True,
            "shards_are_contiguous_byte_exact_slices": True,
            "descriptor_grants_authorization": False,
            "signed_release_must_independently_verify_contiguity": True,
        },
        "source": source,
        "input_digest": input_digest,
        "layout": {
            "strategy": "ordered_contiguous_fixed_size",
            "root_rows": ROOT_ROWS,
            "rows_per_shard": ROWS_PER_SHARD,
            "shard_count": SHARD_COUNT,
            "row_index_basis": "zero_based_end_exclusive",
            "complete_nonoverlapping_coverage": True,
        },
        "shards": descriptors,
        "shards_digest": _object_digest(descriptors),
        "jobs": {
            "path": JOBS_NAME,
            "sha256": _sha256(jobs_raw),
            "bytes": len(jobs_raw),
            "rows_excluding_header": SHARD_COUNT,
        },
        "implementation": implementation,
        "implementation_digest": implementation_digest,
    }
    summary_raw = _pretty_bytes(summary)
    artifacts: dict[str, dict[str, Any]] = {
        relative: {
            "sha256": descriptor["sha256"],
            "bytes": descriptor["bytes"],
            "rows": ROWS_PER_SHARD,
        }
        for relative, descriptor in (
            (item["path"], item) for item in descriptors
        )
    }
    artifacts[JOBS_NAME] = {
        "sha256": _sha256(jobs_raw),
        "bytes": len(jobs_raw),
        "rows": SHARD_COUNT,
    }
    artifacts[SUMMARY_NAME] = {
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
        "source": source,
        "input_digest": input_digest,
        "artifacts": artifacts,
        "artifact_digest": _object_digest(artifacts),
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
        shards_dir = staging / SHARDS_NAME
        shards_dir.mkdir()
        for descriptor in descriptors:
            relative = str(descriptor["path"])
            _write_new(staging / relative, shard_raw[relative])
        _fsync_directory(shards_dir)
        _write_new(staging / JOBS_NAME, jobs_raw)
        _write_new(staging / SUMMARY_NAME, summary_raw)
        _write_new(staging / DONE_NAME, done_raw)
        _fsync_directory(staging)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        _publish_directory_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalizer-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = materialize_full_motion_shards(
        finalizer_dir=args.finalizer_dir,
        output_dir=args.output_dir,
    )
    print(
        "[goku-full-motion-shard-manifest] "
        f"root_rows={summary['layout']['root_rows']} "
        f"shards={summary['layout']['shard_count']} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
