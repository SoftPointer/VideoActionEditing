#!/usr/bin/env python3
"""Build the detached, zero-authority review packet for SAIC T2V Job 132254.

The completed generation bank is immutable input.  This builder verifies its
sealed spec, master receipt, per-attempt receipts, source bindings, and all 60
candidate MP4 hashes; copies the bound media into one fresh review root; and
runs ``saic_exact81_media_diagnostics_v1`` for every source/candidate pair.

Machine camera, technical, and temporal measurements are diagnostics only.
They never fill semantic labels, verify an event or identity, select a seed,
create a training target, or authorize an optimizer/parameter update.  The two
observer files emitted here are sealed *blank templates*.  Independent human
observers must work on copies outside this immutable packet, and no human
assessment is claimed by this builder.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import saic_exact81_media_diagnostics_v1 as diagnostics  # noqa: E402
import saic_pure_t2v_event_bank_v1 as event_contract  # noqa: E402


SCHEMA_VERSION = "bernini-saic-t2v-event-bank-detached-review-manifest-v1"
RECEIPT_SCHEMA_VERSION = (
    "bernini-saic-t2v-event-bank-detached-review-receipt-v1"
)
OBSERVER_TEMPLATE_SCHEMA_VERSION = (
    "bernini-saic-t2v-event-bank-independent-observer-blank-template-v1"
)
OBSERVER_PROTOCOL_SCHEMA_VERSION = (
    "bernini-saic-t2v-event-bank-independent-full81-observer-protocol-v1"
)
MASTER_SCHEMA_VERSION = "bernini-saic-pure-t2v-event-bank-receipt-v1"
ATTEMPT_SCHEMA_VERSION = "bernini-saic-pure-t2v-event-generation-receipt-v1"
PACKET_ID = "t2v-events-v1-533074b-r3-detached-review-v1"
MASTER_BASENAME = "saic-pure-t2v-event-bank-receipt.json"
SOURCE_MANIFEST_BASENAME = "sealed-saic-source-manifest.json"
EVENT_SPEC_BASENAME = "sealed-saic-t2v-event-spec.json"
ATTEMPT_RECEIPT_BASENAME = "saic-event-generation-receipt.json"
FRAME_COUNT = 81
FPS = 25
TRANSITION_COUNT = 80
SEMANTIC_STATUS = "UNASSESSED"
BRANCH_ORDER = ("forward", "reverse", "noop")
EXPECTED_COUNTS = {
    "rows": 8,
    "seed_cells": 20,
    "candidates": 60,
    "sources": 8,
    "observer_templates": 2,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")

_MASTER_FIELDS = {
    "schema_version",
    "bank_id",
    "root_spec_raw_sha256",
    "source_manifest_content_sha256",
    "topology",
    "sampling_contract",
    "semantic_input_closure",
    "geometry_proxy_contract",
    "artifact_authority",
    "attempt_count",
    "row_count",
    "seed_cell_count",
    "branch_order",
    "same_seed_official_gaussian_proofs",
    "attempts",
    "detached_full81_event_review_complete",
    "event_verified",
    "identity_preservation_verified",
    "seed_selection_authorized",
    "training_target_authorized",
    "optimizer_or_parameter_update_authorized",
    "receipt_digest",
}
_MASTER_ATTEMPT_FIELDS = {
    "candidate_id",
    "row_id",
    "iid",
    "analysis_split",
    "branch",
    "seed",
    "receipt_path",
    "receipt_sha256",
    "receipt_digest",
    "mp4_path",
    "mp4_sha256",
    "event_audit_status",
}


class DetachedReviewError(RuntimeError):
    """An input, media, packet, or authority invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise DetachedReviewError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DetachedReviewError(message)


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DetachedReviewError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise DetachedReviewError(f"{label} must be path-safe")
    return value


def _closed(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise DetachedReviewError(
            f"{label} keys differ: observed={observed!r}, expected={sorted(expected)!r}"
        )
    return value


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DetachedReviewError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _plain_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise DetachedReviewError(f"{label} must be absolute")
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise DetachedReviewError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DetachedReviewError(f"{label} must be a plain non-symlink file")
    return candidate.resolve(strict=True)


def _plain_dir(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise DetachedReviewError(f"{label} must be absolute")
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise DetachedReviewError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise DetachedReviewError(f"{label} must be a plain non-symlink directory")
    return candidate.resolve(strict=True)


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _plain_file(path, label=label)
    try:
        value = json.loads(
            source.read_text(encoding="ascii"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DetachedReviewError(f"cannot decode {label}") from error
    if type(value) is not dict:
        raise DetachedReviewError(f"{label} root must be one object")
    return value


def _load_sealed(path: str | Path, *, label: str) -> tuple[dict[str, Any], str]:
    value = _load_json(path, label=label)
    unsigned = dict(value)
    declared = _sha(unsigned.pop("receipt_digest", None), label=f"{label} digest")
    if object_sha256(unsigned) != declared:
        raise DetachedReviewError(f"{label} digest differs")
    return value, declared


def _load_canonical_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _plain_file(path, label=label)
    raw = source.read_bytes()
    value = _load_json(source, label=label)
    if raw != canonical_json_bytes(value) + b"\n":
        raise DetachedReviewError(f"{label} is not canonical JSON bytes")
    return value


def _write_bytes_create_only(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    if not path.is_absolute() or not path.parent.is_dir() or path.parent.is_symlink():
        raise DetachedReviewError(f"unsafe output path: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as error:
        raise DetachedReviewError(f"refusing to overwrite output: {path}") from error
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_json_bytes(value) + b"\n"
    _write_bytes_create_only(path, raw)
    return hashlib.sha256(raw).hexdigest()


def _copy_verified_create_only(
    source: Path, destination: Path, expected_sha256: str
) -> dict[str, Any]:
    source = _plain_file(source, label="copy source")
    expected = _sha(expected_sha256, label="copy source hash")
    _require(file_sha256(source) == expected, f"copy source hash differs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o444)
    except OSError as error:
        raise DetachedReviewError(f"refusing to overwrite copy: {destination}") from error
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fchmod(output_handle.fileno(), 0o444)
            os.fsync(output_handle.fileno())
    except BaseException:
        raise
    observed = file_sha256(destination)
    _require(observed == expected, f"portable copy hash differs: {destination}")
    return {
        "path": str(destination.resolve(strict=True)),
        "sha256": observed,
        "bytes": destination.stat().st_size,
    }


def _false_authority() -> dict[str, bool]:
    return {
        "machine_diagnostics_have_semantic_authority": False,
        "event_verified": False,
        "identity_preservation_verified": False,
        "candidate_selection_allowed": False,
        "seed_selection_allowed": False,
        "training_target_allowed": False,
        "training_allowed": False,
        "optimizer_step_allowed": False,
        "parameter_update_allowed": False,
        "absolute_action_editing_success_claimed": False,
    }


def _flatten_candidates(spec: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    flattened: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group in spec["groups"]:
        for candidate in group["candidates"]:
            flattened.append((dict(group), dict(candidate)))
    return flattened


def _validate_master(
    master: Mapping[str, Any],
    *,
    master_digest: str,
    spec: Mapping[str, Any],
    manifest_summary: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    row = _closed(dict(master), _MASTER_FIELDS, label="master receipt")
    _require(row["schema_version"] == MASTER_SCHEMA_VERSION, "master schema differs")
    _require(row["receipt_digest"] == master_digest, "master seal differs")
    _require(row["bank_id"] == spec["bank_id"], "master bank id differs")
    _require(
        row["source_manifest_content_sha256"]
        == manifest_summary["manifest_content_sha256"],
        "master source manifest content binding differs",
    )
    for field in (
        "sampling_contract",
        "semantic_input_closure",
        "geometry_proxy_contract",
        "artifact_authority",
        "branch_order",
    ):
        _require(row[field] == spec[field], f"master {field} differs from spec")
    _require(row["branch_order"] == list(BRANCH_ORDER), "master branch order differs")
    _require(row["attempt_count"] == EXPECTED_COUNTS["candidates"], "master attempt count differs")
    _require(row["row_count"] == EXPECTED_COUNTS["rows"], "master row count differs")
    _require(row["seed_cell_count"] == EXPECTED_COUNTS["seed_cells"], "master seed count differs")
    for field in (
        "detached_full81_event_review_complete",
        "event_verified",
        "identity_preservation_verified",
        "seed_selection_authorized",
        "training_target_authorized",
        "optimizer_or_parameter_update_authorized",
    ):
        _require(row[field] is False, f"input master unexpectedly authorizes {field}")
    attempts = row["attempts"]
    _require(
        type(attempts) is list and len(attempts) == EXPECTED_COUNTS["candidates"],
        "master attempts differ",
    )
    for index, attempt in enumerate(attempts):
        _closed(attempt, _MASTER_ATTEMPT_FIELDS, label=f"master attempt {index}")
    return attempts


def _validate_attempt_receipt(
    path: Path,
    *,
    expected_sha256: str,
    expected_digest: str,
    candidate: Mapping[str, Any],
    group: Mapping[str, Any],
    root_spec_sha256: str,
) -> dict[str, Any]:
    _require(file_sha256(path) == expected_sha256, "attempt receipt file hash differs")
    receipt, digest = _load_sealed(path, label="attempt receipt")
    _require(digest == expected_digest, "attempt receipt digest differs")
    _require(receipt.get("schema_version") == ATTEMPT_SCHEMA_VERSION, "attempt schema differs")
    _require(receipt.get("root_spec_raw_sha256") == root_spec_sha256, "attempt spec binding differs")
    _require(receipt.get("candidate") == candidate, "attempt candidate binding differs")
    _require(receipt.get("group_id") == group["group_id"], "attempt group differs")
    _require(receipt.get("actor_family") == group["actor_family"], "attempt actor differs")
    _require(receipt.get("visible_gpus") == group["visible_gpus"], "attempt GPU topology differs")
    _require(receipt.get("event_audit_status") == "pending_detached_full81_review", "attempt event state differs")
    for field in (
        "event_verified",
        "identity_preservation_verified",
        "seed_selection_authorized",
        "training_target_authorized",
        "optimizer_or_parameter_update_authorized",
    ):
        _require(receipt.get(field) is False, f"attempt unexpectedly authorizes {field}")
    artifacts = receipt.get("artifacts")
    _require(type(artifacts) is dict and type(artifacts.get("mp4")) is dict, "attempt MP4 binding absent")
    return receipt


def _load_and_validate_inputs(input_root: Path) -> dict[str, Any]:
    root = _plain_dir(input_root, label="event-bank input root")
    master_path = _plain_file(root / MASTER_BASENAME, label="master receipt")
    source_manifest_path = _plain_file(
        root / SOURCE_MANIFEST_BASENAME, label="sealed source manifest"
    )
    event_spec_path = _plain_file(root / EVENT_SPEC_BASENAME, label="sealed event spec")

    source_manifest = source_set.load_manifest(source_manifest_path)
    manifest_summary = source_set.validate_manifest(source_manifest)
    master, master_digest = _load_sealed(master_path, label="master receipt")
    spec, spec_raw_sha256 = event_contract.load_sealed_spec(
        event_spec_path,
        expected_raw_sha256=_sha(
            master.get("root_spec_raw_sha256"), label="master root spec hash"
        ),
        source_manifest_path=source_manifest_path,
    )
    attempts = _validate_master(
        master,
        master_digest=master_digest,
        spec=spec,
        manifest_summary=manifest_summary,
    )
    flattened = _flatten_candidates(spec)
    _require(len(flattened) == EXPECTED_COUNTS["candidates"], "spec candidate count differs")
    _require(
        [item[1]["candidate_id"] for item in flattened]
        == [item["candidate_id"] for item in attempts],
        "master attempts reordered spec candidates",
    )
    source_rows = source_manifest.get("rows")
    _require(type(source_rows) is list and len(source_rows) == EXPECTED_COUNTS["sources"], "source row count differs")
    sources_by_iid = {str(row["iid"]): row for row in source_rows}
    _require(len(sources_by_iid) == EXPECTED_COUNTS["sources"], "duplicate source iid")

    candidate_rows: list[dict[str, Any]] = []
    for ordinal, ((group, candidate), master_attempt) in enumerate(
        zip(flattened, attempts), start=1
    ):
        candidate_id = _safe_id(candidate["candidate_id"], label="candidate id")
        _require(master_attempt["row_id"] == candidate["row_id"], "master row id differs")
        _require(master_attempt["iid"] == candidate["iid"], "master iid differs")
        _require(master_attempt["analysis_split"] == candidate["analysis_split"], "master split differs")
        _require(master_attempt["branch"] == candidate["branch"], "master branch differs")
        _require(master_attempt["seed"] == candidate["seed"], "master seed differs")
        _require(master_attempt["event_audit_status"] == "pending_detached_full81_review", "master event state differs")
        source = sources_by_iid.get(candidate["iid"])
        _require(source is not None, "candidate source row absent")
        _require(source["row_id"] == candidate["row_id"], "candidate/source row binding differs")
        _require(
            source["source_video_sha256"]
            == candidate["source_media_sha256_for_nonuse_audit"],
            "candidate/source media hash binding differs",
        )
        source_video = _plain_file(source["source_video"], label="source video")
        source_sha = _sha(source["source_video_sha256"], label="source video hash")
        _require(file_sha256(source_video) == source_sha, "source video hash differs")

        expected_attempt_path = (
            root / "attempts" / candidate_id / ATTEMPT_RECEIPT_BASENAME
        ).resolve()
        attempt_path = _plain_file(master_attempt["receipt_path"], label="attempt receipt")
        _require(attempt_path == expected_attempt_path, "attempt receipt escaped canonical root")
        attempt_receipt = _validate_attempt_receipt(
            attempt_path,
            expected_sha256=_sha(master_attempt["receipt_sha256"], label="attempt receipt hash"),
            expected_digest=_sha(master_attempt["receipt_digest"], label="attempt receipt digest"),
            candidate=candidate,
            group=group,
            root_spec_sha256=spec_raw_sha256,
        )
        expected_mp4_path = (root / "attempts" / candidate_id / "t2v.mp4").resolve()
        mp4_path = _plain_file(master_attempt["mp4_path"], label="candidate MP4")
        _require(mp4_path == expected_mp4_path, "candidate MP4 escaped canonical root")
        mp4_sha = _sha(master_attempt["mp4_sha256"], label="candidate MP4 hash")
        _require(file_sha256(mp4_path) == mp4_sha, "candidate MP4 hash differs")
        _require(
            attempt_receipt["artifacts"]["mp4"].get("path") == str(mp4_path)
            and attempt_receipt["artifacts"]["mp4"].get("sha256") == mp4_sha,
            "attempt/master MP4 binding differs",
        )
        candidate_rows.append(
            {
                "registered_candidate_index": ordinal,
                "candidate_id": candidate_id,
                "row_id": candidate["row_id"],
                "iid": candidate["iid"],
                "analysis_split": candidate["analysis_split"],
                "actor_family": candidate["actor_family"],
                "action_family_id": candidate["action_family_id"],
                "branch": candidate["branch"],
                "seed": candidate["seed"],
                "initial_state_type": candidate["initial_state_type"],
                "terminal_state_type": candidate["terminal_state_type"],
                "branch_start_state_caption": candidate["branch_start_state_caption"],
                "branch_instruction": candidate["branch_instruction"],
                "full_t2v_caption": candidate["full_t2v_caption"],
                "source_input_path": str(source_video),
                "source_sha256": source_sha,
                "candidate_input_path": str(mp4_path),
                "candidate_sha256": mp4_sha,
                "attempt_receipt_input_path": str(attempt_path),
                "attempt_receipt_sha256": master_attempt["receipt_sha256"],
                "attempt_receipt_digest": master_attempt["receipt_digest"],
                "semantic_status": SEMANTIC_STATUS,
                "event_verified": False,
                "identity_preservation_verified": False,
            }
        )
    return {
        "input_root": root,
        "master": master,
        "master_digest": master_digest,
        "master_path": master_path,
        "source_manifest": source_manifest,
        "source_manifest_path": source_manifest_path,
        "source_manifest_summary": manifest_summary,
        "event_spec": spec,
        "event_spec_path": event_spec_path,
        "event_spec_raw_sha256": spec_raw_sha256,
        "candidate_rows": candidate_rows,
    }


def _diagnostic_worker(task: Mapping[str, str]) -> dict[str, str]:
    value = diagnostics.build_diagnostic(
        source_video=task["source_video"],
        expected_source_sha256=task["source_sha256"],
        candidate_video=task["candidate_video"],
        expected_candidate_sha256=task["candidate_sha256"],
    )
    file_hash = diagnostics.write_diagnostic_create_only(task["output"], value)
    return {
        "candidate_id": task["candidate_id"],
        "diagnostic_digest": value["diagnostic_digest"],
        "diagnostic_file_sha256": file_hash,
    }


def _run_diagnostics(tasks: Sequence[Mapping[str, str]], *, workers: int) -> dict[str, dict[str, str]]:
    if workers == 1:
        rows = [_diagnostic_worker(task) for task in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_diagnostic_worker, task): task for task in tasks}
            for future in as_completed(futures):
                rows.append(future.result())
    _require(len(rows) == EXPECTED_COUNTS["candidates"], "diagnostic count differs")
    indexed = {row["candidate_id"]: row for row in rows}
    _require(len(indexed) == len(rows), "duplicate diagnostic result")
    return indexed


def _validate_diagnostic_file(
    path: Path,
    *,
    source_path: Path,
    source_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
) -> dict[str, Any]:
    value = diagnostics.load_canonical_diagnostic(path)
    unsigned = {key: item for key, item in value.items() if key != "diagnostic_digest"}
    _require(
        object_sha256(unsigned) == value.get("diagnostic_digest"),
        "diagnostic object seal differs",
    )
    _require(value.get("authority") == {
        "measurement_runtime_qualified": False,
        "candidate_selection_allowed": False,
        "training_allowed": False,
        "optimizer_step_allowed": False,
        "absolute_action_editing_success_claimed": False,
    }, "diagnostic authority differs")
    expected_availability = {
        "identity": "unavailable",
        "appearance": "unavailable",
        "background": "unavailable",
        "non_target": "unavailable",
        "event": "unavailable",
        "source_bind": "unavailable",
        "inverse": "unavailable",
        "camera": "diagnostic_only",
        "technical": "diagnostic_only",
        "temporal_consistency": "diagnostic_only",
    }
    _require(value.get("availability") == expected_availability, "diagnostic availability differs")
    media = value.get("media", {})
    for label, expected_path, expected_sha in (
        ("source", source_path, source_sha256),
        ("candidate", candidate_path, candidate_sha256),
    ):
        binding = media.get(label, {})
        _require(binding.get("path") == str(expected_path), f"diagnostic {label} path differs")
        _require(binding.get("sha256") == expected_sha, f"diagnostic {label} hash differs")
        decode = binding.get("decode", {})
        _require(
            decode.get("frame_count") == FRAME_COUNT
            and decode.get("fps_numerator") == FPS
            and decode.get("fps_denominator") == 1,
            f"diagnostic {label} is not exact81@25fps",
        )
    _require(
        value.get("source", {}).get("motion_summary", {}).get("transition_count")
        == TRANSITION_COUNT,
        "source transition count differs",
    )
    _require(
        value.get("candidate", {}).get("motion_summary", {}).get("transition_count")
        == TRANSITION_COUNT,
        "candidate transition count differs",
    )
    return value


def _observer_protocol(*, review_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item_set_digest = object_sha256(list(review_items))
    body = {
        "schema_version": OBSERVER_PROTOCOL_SCHEMA_VERSION,
        "packet_id": PACKET_ID,
        "protocol_timing": "SEALED_AFTER_GENERATION_BEFORE_ANY_DETACHED_HUMAN_LABELS",
        "post_generation_protocol_cannot_claim_pre_generation_preregistration": True,
        "review_item_set_digest": item_set_digest,
        "media_contract": {
            "source_and_candidate_shown_side_by_side": True,
            "candidate_frame_count": FRAME_COUNT,
            "source_frame_count": FRAME_COUNT,
            "fps": FPS,
            "all_frames_must_be_viewed_at_normal_speed_at_least_once": True,
            "frame_scrubbing_and_replay_after_first_view_allowed": True,
            "candidate_id_seed_and_machine_metrics_hidden_in_human_stage": True,
        },
        "stage_order": [
            {
                "stage": 1,
                "artifact": "blind-review.html",
                "rule": "Each observer independently completes and seals all 60 human responses before opening any machine diagnostic artifact.",
            },
            {
                "stage": 2,
                "artifact": "index.html and diagnostics/*.json",
                "rule": "Machine diagnostics may be inspected only after both independent human response artifacts are immutable; they cannot revise human labels.",
            },
        ],
        "branch_specific_full81_criteria": {
            "forward": {
                "start": "Required initial state is visibly correct throughout frames 0-15 without an identity discontinuity.",
                "transition": "A directionally correct, continuous transition toward the registered terminal state is visibly present during frames 16-72.",
                "terminal": "The registered terminal state is reached no later than frame 72 and remains visibly held without reversal through every frame 73-80.",
                "event_pass": "start AND transition AND terminal must all be true.",
            },
            "reverse": {
                "start": "Registered reverse branch start state is visibly correct throughout frames 0-15 without an identity discontinuity.",
                "transition": "A directionally correct, continuous inverse transition toward the registered original state is visibly present during frames 16-72.",
                "terminal": "The registered original state is recovered no later than frame 72 and remains visibly held without reversal through every frame 73-80.",
                "event_pass": "start AND transition AND terminal must all be true.",
            },
            "noop": {
                "start": "Registered initial state is visibly correct at frame 0.",
                "full81_hold": "The same initial state remains visible for all frames 0-80, allowing only breathing or tiny natural motion, and the target terminal event never occurs at any frame.",
                "event_pass": "start AND full81_hold must both be true.",
            },
        },
        "shared_full81_axes": {
            "identity_preserved_full81": "The same source subject remains recognizable with no identity swap, replacement, or material identity morph at every frame 0-80.",
            "camera_locked_full81": "No cut, camera move, reframing, or discontinuous viewpoint change occurs across frames 0-80.",
            "technical_quality_acceptable_full81": "No ghosting, duplication, tearing, disappearance, or corruption obscures the state/event judgment at any frame.",
        },
        "observer_contract": {
            "minimum_independent_observers": 2,
            "observer_kind": "independent_human_full81_review",
            "different_people_required": True,
            "communication_or_label_sharing_before_seal_forbidden": True,
            "distinct_observer_identity_and_authority_artifacts_required": True,
            "same_preparer_must_not_act_as_either_observer": True,
            "one_person_filling_both_templates_forbidden": True,
        },
        "aggregation_rule": {
            "majority_vote_allowed": False,
            "tie_break_or_adjudication_inside_v1_allowed": False,
            "missing_response_result": SEMANTIC_STATUS,
            "observer_disagreement_result": SEMANTIC_STATUS,
            "agreed_positive_result": "AGREED_POSITIVE_PENDING_SEPARATE_EXTERNAL_SEAL",
            "agreed_negative_result": "AGREED_NEGATIVE_PENDING_SEPARATE_EXTERNAL_SEAL",
            "event_verified_may_be_set_by_this_packet": False,
            "identity_verified_may_be_set_by_this_packet": False,
            "separate_versioned_aggregator_required": True,
        },
        "machine_diagnostic_contract": {
            "human_labels_must_precede_machine_diagnostic_access": True,
            "machine_camera_or_technical_thresholds_calibrated": False,
            "machine_diagnostics_may_fill_or_change_human_labels": False,
            "machine_diagnostics_have_semantic_authority": False,
            "machine_diagnostics_may_select_seed_or_training_target": False,
        },
        "authority": _false_authority(),
    }
    return {**body, "protocol_digest": object_sha256(body)}


def _observer_template(
    *,
    slot: int,
    review_items: Sequence[Mapping[str, Any]],
    protocol_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _require(slot in (1, 2), "observer slot differs")
    responses = []
    for item in review_items:
        responses.append(
            {
                "review_item_id": item["review_item_id"],
                "start_state_correct_frames_0_15": None,
                "directional_transition_observed_frames_16_72": None,
                "required_terminal_reached_by_frame_72": None,
                "required_terminal_held_frames_73_80": None,
                "noop_initial_state_held_full81_and_target_absent": None,
                "event_reaches_required_terminal_state": None,
                "event_holds_terminal_state": None,
                "noop_avoids_target_event": None,
                "event_branch_pass": None,
                "identity_preserved_full81": None,
                "camera_locked_full81": None,
                "technical_quality_acceptable_full81": None,
                "observer_notes": None,
            }
        )
    body = {
        "schema_version": OBSERVER_TEMPLATE_SCHEMA_VERSION,
        "packet_id": PACKET_ID,
        "observer_slot": slot,
        "template_only": True,
        "semantic_status": SEMANTIC_STATUS,
        "observer_id": None,
        "observer_kind": None,
        "observer_authority_artifact": None,
        "observer_protocol_artifact": dict(protocol_binding),
        "completed_at": None,
        "independent_observer_required": True,
        "same_person_must_not_fill_both_slots": True,
        "copy_outside_sealed_packet_before_completion": True,
        "blindness_or_independence_established_by_template": False,
        "review_item_set_digest": object_sha256(list(review_items)),
        "responses": responses,
        "authority": _false_authority(),
    }
    return {**body, "template_digest": object_sha256(body)}


def _url(path: str) -> str:
    return quote(path, safe="/:._-")


def _render_blind_html(
    items: Sequence[Mapping[str, Any]], *, job_id: str, protocol_digest: str
) -> str:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item["row_id"], []).append(item)
    sections = []
    for row_index, row_items in enumerate(grouped.values(), start=1):
        first = row_items[0]
        cards = []
        for item in row_items:
            cards.append(
                f'''<article class="card"><header><span class="eyebrow">{html.escape(item['review_item_id'])} · {html.escape(item['branch'].upper())}</span><h3>Registered branch criterion</h3><p>{html.escape(item['branch_start_state_caption'])}</p><p>{html.escape(item['branch_instruction'])}</p></header><video controls muted playsinline preload="metadata" src="{_url(item['portable_candidate'])}"></video></article>'''
            )
        sections.append(
            f'''<section class="sample"><h2>Blind source row {row_index:02d}</h2><p class="muted">Candidate IDs, seeds, and machine measurements are absent from this page.</p><div class="source"><article class="card source-card"><header><span class="eyebrow">SOURCE REFERENCE</span><h3>Hash-bound exact81 source</h3></header><video controls muted playsinline preload="metadata" src="{_url(first['portable_source'])}"></video></article></div><div class="grid">{''.join(cards)}</div></section>'''
        )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAIC T2V blind human review · Job {html.escape(job_id)}</title><style>
:root{{--ink:#18211d;--paper:#f3efe7;--panel:#fffdf7;--line:#cdc6b8;--muted:#68716c;--accent:#166953;--warn:#873d1b;--warnbg:#ffefdc}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1880px;margin:auto;padding:24px}}.hero,.sample{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:17px}}h1{{font-size:clamp(29px,4vw,52px);line-height:1.03;margin:7px 0 13px}}h2,h3,p{{margin-top:0}}.eyebrow{{font-size:11px;font-weight:800;color:var(--accent);letter-spacing:.1em}}.warning{{background:var(--warnbg);border:1px solid #dda171;border-radius:10px;padding:13px;color:var(--warn)}}.muted,.card p{{color:var(--muted)}}.source{{max-width:330px;margin:12px 0}}.grid{{display:grid;grid-template-columns:repeat(6,minmax(195px,1fr));gap:9px}}.card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#faf8f2}}.source-card{{border-color:#6b9f8b;background:#edf7f2}}.card header{{padding:9px;min-height:150px}}video{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#0c0e0d}}@media(max-width:1200px){{.grid{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:700px){{main{{padding:8px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><section class="hero"><span class="eyebrow">INDEPENDENT HUMAN STAGE 1 · AUH {html.escape(job_id)}</span><h1>Blind full81 review</h1><p class="warning"><strong>Do not open index.html or diagnostics/*.json yet.</strong> First watch every source/candidate pair through all 81 frames at normal speed, complete one assigned observer template independently, and seal that response outside this packet. Machine diagnostics cannot revise a human label.</p><p>Protocol digest: <code>{html.escape(protocol_digest)}</code>. Candidate IDs, seeds, and all machine metrics are absent from this page.</p></section>{''.join(sections)}</main></body></html>'''


def _render_html(manifest: Mapping[str, Any], *, job_id: str) -> str:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in manifest["items"]:
        grouped.setdefault(item["row_id"], []).append(item)
    sections: list[str] = []
    for row_index, (row_id, items) in enumerate(grouped.items(), start=1):
        first = items[0]
        source = first["portable_source"]
        cards = []
        for item in items:
            diagnostic = item["diagnostic_summary"]
            technical = diagnostic["technical"]
            camera = diagnostic["camera"]
            cards.append(
                f'''<article class="card" data-cell="{html.escape(item['review_item_id'])}"><header><span class="eyebrow">{html.escape(item['branch'].upper())} · SEED {item['seed']}</span><h3>{html.escape(item['review_item_id'])}</h3><p>semantic: <strong>UNASSESSED</strong> · 81 frames / 80 transitions</p></header><video controls muted playsinline preload="metadata" src="{_url(item['portable_candidate'])}"></video><details><summary>Prompt and zero-authority diagnostics</summary><p>{html.escape(item['branch_instruction'])}</p><dl><dt>Technical geometric mean</dt><dd>{technical['geometric_mean_technical_diagnostic']:.6g}</dd><dt>Camera endpoint L2</dt><dd>{camera['cumulative_global_endpoint_l2_difference']:.6g}</dd></dl><p class="fine">No calibrated pass threshold; no event/identity authority.</p></details></article>'''
            )
        sections.append(
            f'''<section class="sample" id="row-{row_index}"><div class="row-head"><div><span class="row-number">{row_index:02d}</span><h2>{html.escape(row_id)}</h2><p>{html.escape(first['analysis_split'])} · {html.escape(first['actor_family'])} · {len(items)//3} seeds × 3 fixed branches</p></div><div class="controls"><button data-play="{row_index}">Play source + visible candidates</button><button data-reset="{row_index}">Reset row</button></div></div><div class="source"><article class="card source-card"><header><span class="eyebrow">HASH-BOUND SOURCE</span><h3>{html.escape(first['iid'])}</h3><p>exact81 / 25 fps · reference only</p></header><video data-row="{row_index}" data-source controls muted playsinline preload="metadata" src="{_url(source)}"></video></article></div><div class="grid" data-row-grid="{row_index}">{''.join(cards)}</div></section>'''
        )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAIC T2V 60 detached full81 review · Job {html.escape(job_id)}</title>
<style>
:root{{--ink:#17211d;--paper:#f2efe7;--panel:#fffdf7;--line:#cfc8b9;--muted:#69726c;--accent:#176953;--warn:#8b3f1d;--warnbg:#fff0df}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#e7eee9,#f2efe7 30rem);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1900px;margin:auto;padding:24px}}.hero,.sample{{background:rgba(255,253,247,.96);border:1px solid var(--line);border-radius:17px;box-shadow:0 12px 32px rgba(25,45,35,.08)}}.hero{{padding:26px;margin-bottom:18px}}h1{{font-size:clamp(30px,4.4vw,58px);line-height:1;margin:8px 0 14px;letter-spacing:-.04em}}h2,h3,p{{margin-top:0}}h2{{display:inline;font-size:20px}}h3{{margin:4px 0;font-size:14px}}.kicker,.eyebrow{{color:var(--accent);font-size:11px;font-weight:800;letter-spacing:.11em;text-transform:uppercase}}.warning{{max-width:1250px;padding:14px 16px;background:var(--warnbg);border:1px solid #dda476;border-radius:10px;color:var(--warn)}}.facts{{display:flex;flex-wrap:wrap;gap:8px}}.fact{{border:1px solid var(--line);border-radius:999px;padding:7px 10px;background:#f8f6ef}}.sample{{padding:16px;margin-bottom:18px}}.row-head{{display:flex;justify-content:space-between;gap:15px}}.row-number{{display:inline-block;margin-right:9px;background:var(--accent);color:white;border-radius:999px;padding:3px 8px;font-weight:800}}.controls{{display:flex;gap:7px}}button{{border:1px solid #a7aea8;border-radius:8px;background:#f6f2e9;padding:8px 11px;cursor:pointer;font-weight:650}}.source{{max-width:330px;margin:12px 0}}.grid{{display:grid;grid-template-columns:repeat(6,minmax(195px,1fr));gap:9px}}.card{{border:1px solid var(--line);border-radius:11px;overflow:hidden;background:#faf9f4}}.source-card{{border-color:#6fa08e;background:#eef7f2}}.card header{{padding:9px;min-height:91px}}.card header p,.fine{{font-size:12px;color:var(--muted);margin-bottom:0}}video{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#0c0e0d}}details{{padding:8px}}details p{{margin:7px 0}}summary{{cursor:pointer;font-weight:700}}dl{{display:grid;grid-template-columns:1fr auto;gap:3px 8px;font-size:12px}}dt,dd{{margin:0}}a{{color:var(--accent)}}footer{{color:var(--muted);padding:10px 2px 30px}}@media(max-width:1350px){{.grid{{grid-template-columns:repeat(4,minmax(190px,1fr))}}}}@media(max-width:780px){{main{{padding:9px}}.grid{{grid-template-columns:1fr}}.row-head{{display:block}}.controls{{margin:8px 0}}}}
</style></head><body><main><section class="hero"><span class="kicker">AUH job {html.escape(job_id)} · detached review packet · HUMAN STAGE 2 ONLY</span><h1>60 fixed T2V proposals, full81 machine diagnostics.</h1><p class="warning"><strong>Do not open this page before both independent human response files are immutable.</strong> Human stage 1 uses <a href="blind-review.html">blind-review.html</a>. Diagnostics are not semantic evidence: camera, technical, and temporal measurements have permanently zero authority. Event and identity status remain <strong>UNASSESSED</strong>; no seed is ranked or selected; no training or optimizer update is authorized.</p><div class="facts"><span class="fact">8 source rows</span><span class="fact">20 fixed seed cells</span><span class="fact">60 proposals</span><span class="fact">81 frames / 80 transitions each</span><a class="fact" href="observer-protocol.json">sealed protocol</a><a class="fact" href="review-manifest.json">sealed manifest</a><a class="fact" href="observer-templates/observer-1-blank.json">observer 1 blank</a><a class="fact" href="observer-templates/observer-2-blank.json">observer 2 blank</a></div></section>{''.join(sections)}<footer>Independent observers must fill separate copies outside this sealed packet. This page assigns no semantic label.</footer></main><script>
function rowVideos(n){{return [...document.querySelectorAll(`[data-row="${{n}}"], [data-row-grid="${{n}}"] video`)]}}async function reset(n){{for(const v of rowVideos(n)){{v.pause();try{{v.currentTime=0}}catch(_e){{}}}}}}async function play(n){{document.querySelectorAll('video').forEach(v=>v.pause());const vs=rowVideos(n).filter(v=>v.offsetParent!==null);await reset(n);await Promise.allSettled(vs.map(v=>v.play()))}}document.querySelectorAll('[data-play]').forEach(b=>b.onclick=()=>play(b.dataset.play));document.querySelectorAll('[data-reset]').forEach(b=>b.onclick=()=>reset(b.dataset.reset));
</script></body></html>'''


def build_review(
    *,
    input_root: str | Path,
    output_root: str | Path,
    job_id: str,
    workers: int = 16,
) -> dict[str, Any]:
    """Build one fresh immutable review packet; never modify the event bank."""

    _require(type(workers) is int and 1 <= workers <= 32, "workers must be in [1, 32]")
    output = Path(output_root)
    _require(output.is_absolute() and output != Path("/"), "output root must be absolute and non-root")
    output = output.resolve()
    _require(not output.exists() and not output.is_symlink(), "output root must be fresh")
    parent = _plain_dir(output.parent, label="audit parent")
    _require(output == parent / output.name, "output root must be canonical")
    _safe_id(output.name, label="output basename")
    validated = _load_and_validate_inputs(Path(input_root))
    input_root_path = validated["input_root"]
    _require(output != input_root_path and input_root_path not in output.parents, "audit root must be outside input root")

    try:
        os.mkdir(output, 0o755)
    except OSError as error:
        raise DetachedReviewError("cannot create fresh audit root") from error
    for relative in (
        "evidence/attempts",
        "media/sources",
        "media/candidates",
        "diagnostics",
        "observer-templates",
    ):
        (output / relative).mkdir(parents=True, exist_ok=False)

    evidence_bindings = {}
    for key, source_path, basename in (
        ("master_receipt", validated["master_path"], MASTER_BASENAME),
        ("source_manifest", validated["source_manifest_path"], SOURCE_MANIFEST_BASENAME),
        ("event_spec", validated["event_spec_path"], EVENT_SPEC_BASENAME),
    ):
        binding = _copy_verified_create_only(
            source_path, output / "evidence" / basename, file_sha256(source_path)
        )
        binding["portable_path"] = f"evidence/{basename}"
        evidence_bindings[key] = binding

    source_portable: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    diagnostic_tasks: list[dict[str, str]] = []
    for row in validated["candidate_rows"]:
        iid = row["iid"]
        if iid not in source_portable:
            portable = f"media/sources/{row['row_id']}.mp4"
            copied = _copy_verified_create_only(
                Path(row["source_input_path"]), output / portable, row["source_sha256"]
            )
            copied["portable_path"] = portable
            source_portable[iid] = copied
        candidate_portable = (
            f"media/candidates/{row['registered_candidate_index']:04d}-{row['candidate_id']}.mp4"
        )
        candidate_copy = _copy_verified_create_only(
            Path(row["candidate_input_path"]),
            output / candidate_portable,
            row["candidate_sha256"],
        )
        attempt_portable = f"evidence/attempts/{row['candidate_id']}.json"
        attempt_copy = _copy_verified_create_only(
            Path(row["attempt_receipt_input_path"]),
            output / attempt_portable,
            row["attempt_receipt_sha256"],
        )
        diagnostic_portable = f"diagnostics/{row['candidate_id']}.json"
        review_item_id = f"review-{row['registered_candidate_index']:04d}"
        item = {
            **row,
            "review_item_id": review_item_id,
            "portable_source": source_portable[iid]["portable_path"],
            "portable_candidate": candidate_portable,
            "portable_attempt_receipt": attempt_portable,
            "portable_diagnostic": diagnostic_portable,
            "portable_source_bytes": source_portable[iid]["bytes"],
            "portable_candidate_bytes": candidate_copy["bytes"],
            "portable_attempt_receipt_bytes": attempt_copy["bytes"],
        }
        items.append(item)
        diagnostic_tasks.append(
            {
                "candidate_id": row["candidate_id"],
                "source_video": str(output / item["portable_source"]),
                "source_sha256": row["source_sha256"],
                "candidate_video": str(output / candidate_portable),
                "candidate_sha256": row["candidate_sha256"],
                "output": str(output / diagnostic_portable),
            }
        )

    diagnostic_results = _run_diagnostics(diagnostic_tasks, workers=workers)
    review_items_for_observers = []
    for item in items:
        diagnostic_path = output / item["portable_diagnostic"]
        diagnostic = _validate_diagnostic_file(
            diagnostic_path,
            source_path=output / item["portable_source"],
            source_sha256=item["source_sha256"],
            candidate_path=output / item["portable_candidate"],
            candidate_sha256=item["candidate_sha256"],
        )
        result = diagnostic_results[item["candidate_id"]]
        _require(result["diagnostic_digest"] == diagnostic["diagnostic_digest"], "worker diagnostic digest differs")
        _require(result["diagnostic_file_sha256"] == file_sha256(diagnostic_path), "worker diagnostic file hash differs")
        item["diagnostic_digest"] = diagnostic["diagnostic_digest"]
        item["diagnostic_file_sha256"] = result["diagnostic_file_sha256"]
        item["diagnostic_summary"] = {
            "camera": diagnostic["comparisons"]["camera_trajectory"],
            "technical": diagnostic["comparisons"]["technical"],
            "scene_cut_ratio_absolute_difference": diagnostic["comparisons"]["scene_cut_ratio_absolute_difference"],
            "temporal_energy_cv_absolute_difference": diagnostic["comparisons"]["temporal_energy_cv_absolute_difference"],
            "semantic_status": SEMANTIC_STATUS,
            "authority": "diagnostic_only",
        }
        review_items_for_observers.append(
            {
                "review_item_id": item["review_item_id"],
                "candidate_media_sha256": item["candidate_sha256"],
                "source_media_sha256": item["source_sha256"],
                "branch": item["branch"],
                "seed": item["seed"],
            }
        )

    protocol = _observer_protocol(review_items=review_items_for_observers)
    protocol_file_sha = _write_json_create_only(
        output / "observer-protocol.json", protocol
    )
    protocol_binding = {
        "portable_path": "observer-protocol.json",
        "file_sha256": protocol_file_sha,
        "protocol_digest": protocol["protocol_digest"],
        "review_item_set_digest": protocol["review_item_set_digest"],
    }
    observer_bindings = []
    for slot in (1, 2):
        template = _observer_template(
            slot=slot,
            review_items=review_items_for_observers,
            protocol_binding=protocol_binding,
        )
        relative = f"observer-templates/observer-{slot}-blank.json"
        file_hash = _write_json_create_only(output / relative, template)
        observer_bindings.append(
            {
                "observer_slot": slot,
                "portable_path": relative,
                "file_sha256": file_hash,
                "template_digest": template["template_digest"],
                "semantic_status": SEMANTIC_STATUS,
                "template_only": True,
            }
        )

    manifest_body = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": PACKET_ID,
        "job_id": str(job_id),
        "input_root": str(input_root_path),
        "input_bank_receipt_digest": validated["master_digest"],
        "input_event_spec_raw_sha256": validated["event_spec_raw_sha256"],
        "input_bindings": evidence_bindings,
        "row_count": EXPECTED_COUNTS["rows"],
        "source_count": len(source_portable),
        "seed_cell_count": EXPECTED_COUNTS["seed_cells"],
        "candidate_count": len(items),
        "machine_diagnostic_count": len(items),
        "frame_count_per_media": FRAME_COUNT,
        "transition_count_per_media": TRANSITION_COUNT,
        "fps": FPS,
        "branch_order": list(BRANCH_ORDER),
        "machine_diagnostic_axes": ["camera", "technical", "temporal_consistency"],
        "machine_diagnostic_authority": "ZERO_AUTHORITY_DIAGNOSTIC_ONLY",
        "semantic_status": SEMANTIC_STATUS,
        "detached_human_review_complete": False,
        "observer_protocol": protocol_binding,
        "observer_template_count": len(observer_bindings),
        "observer_templates": observer_bindings,
        "candidate_ranking_or_selection_performed": False,
        "authority": _false_authority(),
        "items": items,
    }
    _require(manifest_body["source_count"] == EXPECTED_COUNTS["sources"], "portable source count differs")
    _require(manifest_body["candidate_count"] == EXPECTED_COUNTS["candidates"], "portable candidate count differs")
    manifest = {**manifest_body, "manifest_digest": object_sha256(manifest_body)}
    manifest_file_sha = _write_json_create_only(output / "review-manifest.json", manifest)

    blind_html_text = _render_blind_html(
        items, job_id=str(job_id), protocol_digest=protocol["protocol_digest"]
    )
    _require(
        blind_html_text.count("<video ")
        == EXPECTED_COUNTS["sources"] + EXPECTED_COUNTS["candidates"],
        "blind HTML video closure differs",
    )
    _require(
        "machine measurements are absent" in blind_html_text,
        "blind HTML disclosure differs",
    )
    blind_html_raw = blind_html_text.encode("utf-8")
    _write_bytes_create_only(output / "blind-review.html", blind_html_raw)
    blind_html_sha = hashlib.sha256(blind_html_raw).hexdigest()

    html_text = _render_html(manifest, job_id=str(job_id))
    _require(html_text.count("<video ") == EXPECTED_COUNTS["sources"] + EXPECTED_COUNTS["candidates"], "HTML video closure differs")
    _require(html_text.count("semantic: <strong>UNASSESSED</strong>") == EXPECTED_COUNTS["candidates"], "HTML semantic closure differs")
    html_raw = html_text.encode("utf-8")
    _write_bytes_create_only(output / "index.html", html_raw)
    html_sha = hashlib.sha256(html_raw).hexdigest()

    receipt_body = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "packet_id": PACKET_ID,
        "job_id": str(job_id),
        "input_bank_receipt": {
            "path": str(validated["master_path"]),
            "file_sha256": file_sha256(validated["master_path"]),
            "receipt_digest": validated["master_digest"],
        },
        "review_manifest": {
            "path": str(output / "review-manifest.json"),
            "file_sha256": manifest_file_sha,
            "manifest_digest": manifest["manifest_digest"],
        },
        "html_review": {
            "path": str(output / "index.html"),
            "file_sha256": html_sha,
        },
        "blind_human_review": {
            "path": str(output / "blind-review.html"),
            "file_sha256": blind_html_sha,
            "machine_diagnostics_exposed": False,
        },
        "observer_protocol": {
            "path": str(output / "observer-protocol.json"),
            "file_sha256": protocol_file_sha,
            "protocol_digest": protocol["protocol_digest"],
        },
        "row_count": EXPECTED_COUNTS["rows"],
        "source_count": EXPECTED_COUNTS["sources"],
        "seed_cell_count": EXPECTED_COUNTS["seed_cells"],
        "candidate_count": EXPECTED_COUNTS["candidates"],
        "machine_diagnostic_count": EXPECTED_COUNTS["candidates"],
        "exact81_machine_diagnostics_complete": True,
        "full80_transition_machine_diagnostics_complete": True,
        "machine_diagnostics_zero_authority": True,
        "detached_full81_event_review_complete": False,
        "semantic_status": SEMANTIC_STATUS,
        "event_verified": False,
        "identity_preservation_verified": False,
        "candidate_ranking_or_selection_performed": False,
        "seed_selection_authorized": False,
        "training_target_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_step_authorized": False,
        "parameter_update_authorized": False,
        "observer_template_count": EXPECTED_COUNTS["observer_templates"],
        "observer_labels_present": False,
        "authority": _false_authority(),
    }
    receipt = {**receipt_body, "receipt_digest": object_sha256(receipt_body)}
    _write_json_create_only(output / "detached-review-receipt.json", receipt)

    for artifact in (path for path in output.rglob("*") if path.is_file()):
        os.chmod(artifact, 0o444)
    for directory in sorted(
        (path for path in output.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        os.chmod(directory, 0o555)
    os.chmod(output, 0o555)
    return receipt


def _verify_object_seal(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    declared = _sha(value.get(field), label=f"{label} {field}")
    body = {key: item for key, item in value.items() if key != field}
    _require(object_sha256(body) == declared, f"{label} object seal differs")
    return declared


def validate_packet(output_root: str | Path) -> dict[str, Any]:
    """Validate the published packet without replaying expensive diagnostics."""

    output = _plain_dir(output_root, label="detached review packet")
    _require(
        stat.S_IMODE(output.stat().st_mode) == 0o555,
        "packet root mode must be 0555",
    )
    for path in output.rglob("*"):
        if path.is_symlink():
            raise DetachedReviewError(f"packet contains symlink: {path}")
        expected_mode = 0o555 if path.is_dir() else 0o444
        _require(
            stat.S_IMODE(path.stat().st_mode) == expected_mode,
            f"packet mode differs: {path}",
        )

    receipt_path = output / "detached-review-receipt.json"
    manifest_path = output / "review-manifest.json"
    protocol_path = output / "observer-protocol.json"
    receipt = _load_canonical_json(receipt_path, label="detached review receipt")
    manifest = _load_canonical_json(manifest_path, label="review manifest")
    protocol = _load_canonical_json(protocol_path, label="observer protocol")
    _verify_object_seal(receipt, field="receipt_digest", label="detached review receipt")
    _verify_object_seal(manifest, field="manifest_digest", label="review manifest")
    _verify_object_seal(protocol, field="protocol_digest", label="observer protocol")

    _require(receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION, "receipt schema differs")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "manifest schema differs")
    _require(protocol.get("schema_version") == OBSERVER_PROTOCOL_SCHEMA_VERSION, "protocol schema differs")
    _require(
        receipt.get("packet_id") == manifest.get("packet_id") == protocol.get("packet_id") == PACKET_ID,
        "packet id closure differs",
    )
    _require(receipt.get("semantic_status") == SEMANTIC_STATUS, "receipt semantic status differs")
    _require(manifest.get("semantic_status") == SEMANTIC_STATUS, "manifest semantic status differs")
    for field in (
        "detached_full81_event_review_complete",
        "event_verified",
        "identity_preservation_verified",
        "candidate_ranking_or_selection_performed",
        "seed_selection_authorized",
        "training_target_authorized",
        "training_performed",
        "optimizer_created",
        "optimizer_step_authorized",
        "parameter_update_authorized",
        "observer_labels_present",
    ):
        _require(receipt.get(field) is False, f"receipt unexpectedly authorizes {field}")
    _require(receipt.get("machine_diagnostics_zero_authority") is True, "machine authority differs")
    _require(receipt.get("authority") == _false_authority(), "receipt authority contract differs")

    manifest_binding = receipt.get("review_manifest", {})
    _require(manifest_binding.get("path") == str(manifest_path), "receipt manifest path differs")
    _require(manifest_binding.get("file_sha256") == file_sha256(manifest_path), "receipt manifest hash differs")
    _require(manifest_binding.get("manifest_digest") == manifest["manifest_digest"], "receipt manifest digest differs")
    for field, basename in (
        ("html_review", "index.html"),
        ("blind_human_review", "blind-review.html"),
    ):
        binding = receipt.get(field, {})
        artifact = output / basename
        _require(binding.get("path") == str(artifact), f"{field} path differs")
        _require(binding.get("file_sha256") == file_sha256(artifact), f"{field} hash differs")
    _require(receipt["blind_human_review"].get("machine_diagnostics_exposed") is False, "blind page exposure differs")
    protocol_binding = receipt.get("observer_protocol", {})
    _require(protocol_binding.get("path") == str(protocol_path), "receipt protocol path differs")
    _require(protocol_binding.get("file_sha256") == file_sha256(protocol_path), "receipt protocol hash differs")
    _require(protocol_binding.get("protocol_digest") == protocol["protocol_digest"], "receipt protocol digest differs")

    items = manifest.get("items")
    _require(type(items) is list and len(items) == EXPECTED_COUNTS["candidates"], "manifest item closure differs")
    _require(len({item.get("review_item_id") for item in items}) == len(items), "review item ids differ")
    _require(len({item.get("candidate_id") for item in items}) == len(items), "candidate ids differ")
    source_paths = {item.get("portable_source") for item in items}
    _require(len(source_paths) == EXPECTED_COUNTS["sources"], "source path closure differs")
    for item in items:
        _require(item.get("semantic_status") == SEMANTIC_STATUS, "item semantic status differs")
        _require(item.get("event_verified") is False, "item event status differs")
        _require(item.get("identity_preservation_verified") is False, "item identity status differs")
        for path_field, sha_field in (
            ("portable_source", "source_sha256"),
            ("portable_candidate", "candidate_sha256"),
            ("portable_attempt_receipt", "attempt_receipt_sha256"),
            ("portable_diagnostic", "diagnostic_file_sha256"),
        ):
            artifact = _plain_file(output / item[path_field], label=f"item {path_field}")
            _require(output in artifact.parents, f"item artifact escaped packet: {artifact}")
            _require(file_sha256(artifact) == item[sha_field], f"item {path_field} hash differs")
        diagnostic = _load_canonical_json(
            output / item["portable_diagnostic"], label="item diagnostic"
        )
        _verify_object_seal(diagnostic, field="diagnostic_digest", label="item diagnostic")
        _require(diagnostic.get("authority", {}).get("candidate_selection_allowed") is False, "diagnostic selection authority differs")
        _require(diagnostic.get("availability", {}).get("event") == "unavailable", "diagnostic event availability differs")

    manifest_protocol = manifest.get("observer_protocol", {})
    _require(manifest_protocol.get("file_sha256") == file_sha256(protocol_path), "manifest protocol hash differs")
    _require(manifest_protocol.get("protocol_digest") == protocol["protocol_digest"], "manifest protocol digest differs")
    _require(protocol.get("review_item_set_digest") == object_sha256([
        {
            "review_item_id": item["review_item_id"],
            "candidate_media_sha256": item["candidate_sha256"],
            "source_media_sha256": item["source_sha256"],
            "branch": item["branch"],
            "seed": item["seed"],
        }
        for item in items
    ]), "protocol review item closure differs")
    _require(protocol.get("aggregation_rule", {}).get("observer_disagreement_result") == SEMANTIC_STATUS, "protocol disagreement rule differs")
    _require(protocol.get("aggregation_rule", {}).get("event_verified_may_be_set_by_this_packet") is False, "protocol event authority differs")

    templates = manifest.get("observer_templates")
    _require(type(templates) is list and len(templates) == EXPECTED_COUNTS["observer_templates"], "template binding closure differs")
    for expected_slot, binding in enumerate(templates, start=1):
        template_path = output / binding["portable_path"]
        template = _load_canonical_json(template_path, label="observer blank template")
        _verify_object_seal(template, field="template_digest", label="observer blank template")
        _require(template.get("observer_slot") == expected_slot, "observer slot differs")
        _require(template.get("template_only") is True, "observer artifact is not blank template")
        _require(template.get("observer_id") is None and template.get("completed_at") is None, "observer template is already filled")
        _require(template.get("observer_protocol_artifact") == manifest_protocol, "template protocol binding differs")
        _require(binding.get("file_sha256") == file_sha256(template_path), "template file hash differs")
        _require(len(template.get("responses", [])) == EXPECTED_COUNTS["candidates"], "template response closure differs")
        _require(all(response.get("event_branch_pass") is None for response in template["responses"]), "template contains event labels")

    _require(len(list((output / "diagnostics").glob("*.json"))) == EXPECTED_COUNTS["candidates"], "diagnostic file count differs")
    _require(len(list((output / "observer-templates").glob("*.json"))) == EXPECTED_COUNTS["observer_templates"], "observer template file count differs")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--input-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--job-id", required=True)
    build.add_argument("--workers", type=int, default=16)
    validate = commands.add_parser("validate")
    validate.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        receipt = build_review(
            input_root=args.input_root,
            output_root=args.output_root,
            job_id=args.job_id,
            workers=args.workers,
        )
    else:
        receipt = validate_packet(args.output_root)
    print(
        canonical_json_bytes(
            {
                "candidate_count": receipt["candidate_count"],
                "machine_diagnostic_count": receipt["machine_diagnostic_count"],
                "receipt_digest": receipt["receipt_digest"],
                "semantic_status": receipt["semantic_status"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DetachedReviewError",
    "EXPECTED_COUNTS",
    "FRAME_COUNT",
    "FPS",
    "OBSERVER_PROTOCOL_SCHEMA_VERSION",
    "OBSERVER_TEMPLATE_SCHEMA_VERSION",
    "PACKET_ID",
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SEMANTIC_STATUS",
    "TRANSITION_COUNT",
    "build_review",
    "canonical_json_bytes",
    "file_sha256",
    "main",
    "object_sha256",
    "validate_packet",
]
