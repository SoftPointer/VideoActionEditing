"""Select the lowest-ranked eight pending rows from one Goku finalizer.

This module is deliberately an authorization-free bridge.  It verifies one
complete :mod:`motive.goku_action_anchor_finalize` output directory and emits
an exact-eight, still-pending generation manifest plus a provenance receipt.
It never invokes Wan, signs a release, or changes any authorization flag.

The parent finalizer remains the source of truth:

* ``done.json`` must bind every expected finalizer output;
* ``summary.json`` must independently bind every non-summary output;
* the recorded finalizer implementation must equal the current sibling source;
* review and generation JSONL inputs must be byte-canonical; and
* every generation instruction must equal the frozen review-row ``prompt``.

Selection is deterministic: the eight generation rows whose corresponding
review rows have the lowest unique positive ``review_rank`` are copied
byte-for-byte, in ascending rank order.
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
import stat
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FINALIZER_REVIEW_SCHEMA = "motive-goku-action-anchor-final-row-v8"
FINALIZER_GENERATION_SCHEMA = "motive-goku-action-anchor-generation-v9"
FINALIZER_SUMMARY_SCHEMA = "motive-goku-action-anchor-finalize-v8"
FINALIZER_DONE_SCHEMA = "motive-goku-action-anchor-finalize-done-v8"
FINALIZER_POLICY_VERSION = "goku-action-anchor-strict-continuity-v8"

RECEIPT_SCHEMA = "motive-wan22-exact8-selection-receipt-v1"
SELECTION_POLICY = "lowest_finalizer_review_rank_exact8"
SELECTED_ROW_COUNT = 8

REVIEW_NAME = "review_candidates.jsonl"
PROPOSED_NAME = "proposed_128.jsonl"
RESERVE_NAME = "reserve_32.jsonl"
PARENT_GENERATION_NAME = "generation_manifest.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"

OUTPUT_MANIFEST_NAME = "generation_manifest.jsonl"
OUTPUT_RECEIPT_NAME = "selection_receipt.json"

_FINALIZER_HASHED_OUTPUTS = (
    REVIEW_NAME,
    PROPOSED_NAME,
    RESERVE_NAME,
    PARENT_GENERATION_NAME,
    SUMMARY_NAME,
)
_SUMMARY_HASHED_OUTPUTS = (
    REVIEW_NAME,
    PROPOSED_NAME,
    RESERVE_NAME,
    PARENT_GENERATION_NAME,
)

_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

_PENDING_SEMANTICS = {
    "manifest_role": "review_proposal",
    "production_eligible": False,
    "human_review_status": "pending",
    "generation_authorized": False,
    "approval": None,
    "authorization_interface_available": False,
}
_INSTRUCTION_CONTRACT = {
    "sole_candidate_instruction_field": "edit_instruction",
    "candidate_instruction_source": "frozen_selected_prompt",
    "writer_proposal_payload_included": False,
    "writer_proposals_executable": False,
    "requires_future_signed_release_verifier": True,
}


class Wan22Exact8SelectionError(RuntimeError):
    """The parent finalizer or requested exact-eight output is invalid."""


@dataclass(frozen=True)
class _CanonicalRow:
    value: dict[str, Any]
    raw_line: bytes
    line_number: int


def _reject_constant(value: str) -> None:
    raise Wan22Exact8SelectionError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Wan22Exact8SelectionError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _parse_finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise Wan22Exact8SelectionError(
            f"non-finite JSON number is forbidden: {value}"
        )
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Wan22Exact8SelectionError(
            f"{context} is not UTF-8"
        ) from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except Wan22Exact8SelectionError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise Wan22Exact8SelectionError(
            f"{context} is not strict JSON: {error}"
        ) from error


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
        raise Wan22Exact8SelectionError(
            f"value is not canonical JSON: {error}"
        ) from error


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Wan22Exact8SelectionError(f"{context} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise Wan22Exact8SelectionError(
            f"{context} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _string(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise Wan22Exact8SelectionError(
            f"{context} must be one canonical non-empty string"
        )
    return value


def _iid(value: Any, *, context: str) -> str:
    result = _string(value, context=context)
    if _IID_RE.fullmatch(result) is None:
        raise Wan22Exact8SelectionError(f"{context} is unsafe")
    return result


def _digest(value: Any, *, context: str) -> str:
    result = _string(value, context=context)
    if _SHA_RE.fullmatch(result) is None:
        raise Wan22Exact8SelectionError(
            f"{context} must be a lowercase SHA-256"
        )
    return result


def _stable_read(path: Path, *, context: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Wan22Exact8SelectionError(
            f"{context} is missing or is not a readable non-symlink file: "
            f"{path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Wan22Exact8SelectionError(
                f"{context} is not a regular file: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
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
        raw = b"".join(chunks)
        if before_identity != after_identity or len(raw) != after.st_size:
            raise Wan22Exact8SelectionError(
                f"{context} changed while it was read: {path}"
            )
        return raw
    finally:
        os.close(descriptor)


def _load_json_object(raw: bytes, *, context: str) -> dict[str, Any]:
    value = _parse_json(raw, context=context)
    if not isinstance(value, dict):
        raise Wan22Exact8SelectionError(
            f"{context} top level must be an object"
        )
    return value


def _load_canonical_jsonl(
    raw: bytes,
    *,
    context: str,
) -> list[_CanonicalRow]:
    if not raw or not raw.endswith(b"\n"):
        raise Wan22Exact8SelectionError(
            f"{context} must be non-empty and newline-terminated"
        )
    rows: list[_CanonicalRow] = []
    for line_number, bare in enumerate(raw.splitlines(), start=1):
        if not bare:
            raise Wan22Exact8SelectionError(
                f"{context}:{line_number} is blank"
            )
        value = _parse_json(
            bare,
            context=f"{context}:{line_number}",
        )
        if not isinstance(value, dict):
            raise Wan22Exact8SelectionError(
                f"{context}:{line_number} is not an object"
            )
        if bare != _canonical_bytes(value):
            raise Wan22Exact8SelectionError(
                f"{context}:{line_number} is not canonical JSON"
            )
        rows.append(
            _CanonicalRow(
                value=value,
                raw_line=bare + b"\n",
                line_number=line_number,
            )
        )
    return rows


def _validate_parent_hashes(
    *,
    finalizer_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, bytes],
    bytes,
    str,
]:
    done_raw = _stable_read(
        finalizer_dir / DONE_NAME,
        context=DONE_NAME,
    )
    done = _load_json_object(done_raw, context=DONE_NAME)
    _exact_keys(
        done,
        {
            "schema_version",
            "status",
            "summary_sha256",
            "implementation_sha256",
            "output_sha256",
        },
        context=DONE_NAME,
    )
    if done.get("schema_version") != FINALIZER_DONE_SCHEMA:
        raise Wan22Exact8SelectionError("done.json schema differs")
    if done.get("status") != "complete":
        raise Wan22Exact8SelectionError(
            "done.json status is not complete"
        )

    done_outputs = _mapping(
        done.get("output_sha256"),
        context="done.json.output_sha256",
    )
    _exact_keys(
        done_outputs,
        set(_FINALIZER_HASHED_OUTPUTS),
        context="done.json.output_sha256",
    )
    raw_outputs = {
        name: _stable_read(
            finalizer_dir / name,
            context=f"finalizer output {name}",
        )
        for name in _FINALIZER_HASHED_OUTPUTS
    }
    for name, raw in raw_outputs.items():
        expected = _digest(
            done_outputs.get(name),
            context=f"done.json.output_sha256[{name!r}]",
        )
        if _sha256(raw) != expected:
            raise Wan22Exact8SelectionError(
                f"done.json hash differs for {name}"
            )

    summary_raw = raw_outputs[SUMMARY_NAME]
    summary_sha = _digest(
        done.get("summary_sha256"),
        context="done.json.summary_sha256",
    )
    if summary_sha != _sha256(summary_raw):
        raise Wan22Exact8SelectionError(
            "done.json summary_sha256 differs"
        )
    summary = _load_json_object(summary_raw, context=SUMMARY_NAME)
    _exact_keys(
        summary,
        {
            "schema_version",
            "policy_version",
            "seed",
            "input",
            "hard_gate",
            "diversity",
            "selection",
            "semantics",
            "implementation_sha256",
            "output_sha256",
        },
        context=SUMMARY_NAME,
    )
    if summary.get("schema_version") != FINALIZER_SUMMARY_SCHEMA:
        raise Wan22Exact8SelectionError("summary.json schema differs")
    if summary.get("policy_version") != FINALIZER_POLICY_VERSION:
        raise Wan22Exact8SelectionError("summary.json policy differs")

    implementation_path = Path(__file__).resolve(strict=True).with_name(
        "goku_action_anchor_finalize.py"
    )
    if implementation_path.is_symlink():
        raise Wan22Exact8SelectionError(
            "finalizer implementation sibling must not be a symlink"
        )
    implementation_raw = _stable_read(
        implementation_path,
        context="finalizer implementation sibling",
    )
    implementation_sha = _sha256(implementation_raw)
    done_implementation = _digest(
        done.get("implementation_sha256"),
        context="done.json.implementation_sha256",
    )
    summary_implementation = _digest(
        summary.get("implementation_sha256"),
        context="summary.json.implementation_sha256",
    )
    if (
        done_implementation != implementation_sha
        or summary_implementation != implementation_sha
    ):
        raise Wan22Exact8SelectionError(
            "recorded finalizer implementation does not match sibling source"
        )

    summary_outputs = _mapping(
        summary.get("output_sha256"),
        context="summary.json.output_sha256",
    )
    _exact_keys(
        summary_outputs,
        set(_SUMMARY_HASHED_OUTPUTS),
        context="summary.json.output_sha256",
    )
    for name in _SUMMARY_HASHED_OUTPUTS:
        expected = _digest(
            summary_outputs.get(name),
            context=f"summary.json.output_sha256[{name!r}]",
        )
        actual = _sha256(raw_outputs[name])
        if expected != actual or done_outputs.get(name) != expected:
            raise Wan22Exact8SelectionError(
                f"summary.json hash differs for {name}"
            )

    semantics = _mapping(
        summary.get("semantics"),
        context="summary.json.semantics",
    )
    expected_summary_semantics = {
        **_PENDING_SEMANTICS,
        "human_labels_asserted": False,
    }
    if dict(semantics) != expected_summary_semantics:
        raise Wan22Exact8SelectionError(
            "summary.json does not assert exact pending semantics"
        )
    return done, summary, raw_outputs, done_raw, implementation_sha


def _validate_review_rows(
    rows: Sequence[_CanonicalRow],
) -> dict[str, tuple[_CanonicalRow, int]]:
    by_iid: dict[str, tuple[_CanonicalRow, int]] = {}
    groups: set[str] = set()
    ranks: set[int] = set()
    for record in rows:
        row = record.value
        context = f"{REVIEW_NAME}:{record.line_number}"
        iid = _iid(row.get("iid"), context=f"{context}.iid")
        group = _string(
            row.get("group_id"),
            context=f"{context}.group_id",
        )
        prompt = _string(
            row.get("prompt"),
            context=f"{context}.prompt",
        )
        if iid in by_iid:
            raise Wan22Exact8SelectionError(
                f"duplicate review iid: {iid}"
            )
        if group in groups:
            raise Wan22Exact8SelectionError(
                f"duplicate review group_id: {group}"
            )
        finalization = _mapping(
            row.get("action_anchor_finalization"),
            context=f"{context}.action_anchor_finalization",
        )
        if finalization.get("schema_version") != FINALIZER_REVIEW_SCHEMA:
            raise Wan22Exact8SelectionError(
                f"{context} finalization schema differs"
            )
        if finalization.get("policy_version") != FINALIZER_POLICY_VERSION:
            raise Wan22Exact8SelectionError(
                f"{context} finalization policy differs"
            )
        if finalization.get("hard_gate_passed") is not True:
            raise Wan22Exact8SelectionError(
                f"{context} did not pass the hard gate"
            )
        failures = finalization.get("hard_gate_failures")
        if not isinstance(failures, list) or failures:
            raise Wan22Exact8SelectionError(
                f"{context} has hard-gate failures"
            )
        rank = finalization.get("review_rank")
        if type(rank) is not int or rank <= 0:
            raise Wan22Exact8SelectionError(
                f"{context} review_rank must be a positive integer"
            )
        if rank in ranks:
            raise Wan22Exact8SelectionError(
                f"duplicate review_rank: {rank}"
            )
        bucket = finalization.get("selection_bucket")
        if bucket not in {"proposed", "reserve", "review_only"}:
            raise Wan22Exact8SelectionError(
                f"{context} has an invalid selection bucket"
            )
        for field, expected in _PENDING_SEMANTICS.items():
            if finalization.get(field) != expected:
                raise Wan22Exact8SelectionError(
                    f"{context}.{field} is not the exact pending value"
                )
        if finalization.get("human_label") is not False:
            raise Wan22Exact8SelectionError(
                f"{context} unexpectedly asserts a human label"
            )
        # Force evaluation here so later comparison cannot accept an exotic
        # non-string value that happened to survive object lookup.
        if prompt != row["prompt"]:
            raise AssertionError("unreachable canonical prompt mismatch")
        by_iid[iid] = (record, rank)
        groups.add(group)
        ranks.add(rank)
    return by_iid


def _validate_generation_rows(
    rows: Sequence[_CanonicalRow],
    *,
    review_by_iid: Mapping[str, tuple[_CanonicalRow, int]],
) -> list[tuple[int, _CanonicalRow]]:
    ranked: list[tuple[int, _CanonicalRow]] = []
    iids: set[str] = set()
    groups: set[str] = set()
    for record in rows:
        row = record.value
        context = f"{PARENT_GENERATION_NAME}:{record.line_number}"
        if row.get("schema_version") != FINALIZER_GENERATION_SCHEMA:
            raise Wan22Exact8SelectionError(
                f"{context} schema differs"
            )
        iid = _iid(row.get("iid"), context=f"{context}.iid")
        group = _string(
            row.get("group_id"),
            context=f"{context}.group_id",
        )
        if iid in iids:
            raise Wan22Exact8SelectionError(
                f"duplicate generation iid: {iid}"
            )
        if group in groups:
            raise Wan22Exact8SelectionError(
                f"duplicate generation group_id: {group}"
            )
        for field, expected in _PENDING_SEMANTICS.items():
            if row.get(field) != expected:
                raise Wan22Exact8SelectionError(
                    f"{context}.{field} is not the exact pending v9 value"
                )
        if row.get("action_change_substantive") != "yes":
            raise Wan22Exact8SelectionError(
                f"{context} action change is not substantive"
            )
        if "_authorization_mode" in row:
            raise Wan22Exact8SelectionError(
                f"{context} contains a forbidden authorization marker"
            )
        if (
            "absolute_target_prompt" in row
            or "writer_absolute_target_prompt" in row
        ):
            raise Wan22Exact8SelectionError(
                f"{context} contains a forbidden writer prompt"
            )
        instruction = _string(
            row.get("edit_instruction"),
            context=f"{context}.edit_instruction",
        )
        instruction_sha = _digest(
            row.get("edit_instruction_sha256"),
            context=f"{context}.edit_instruction_sha256",
        )
        if instruction_sha != _sha256(instruction.encode("utf-8")):
            raise Wan22Exact8SelectionError(
                f"{context} edit_instruction SHA differs"
            )
        if row.get("source_instruction_provenance") != instruction:
            raise Wan22Exact8SelectionError(
                f"{context} instruction provenance differs"
            )
        if row.get("source_edited_caption_provenance_role") != (
            "non_executable_provenance"
        ):
            raise Wan22Exact8SelectionError(
                f"{context} edited-caption provenance is executable"
            )
        contract = _mapping(
            row.get("instruction_contract"),
            context=f"{context}.instruction_contract",
        )
        if dict(contract) != _INSTRUCTION_CONTRACT:
            raise Wan22Exact8SelectionError(
                f"{context} instruction contract differs"
            )

        match = review_by_iid.get(iid)
        if match is None:
            raise Wan22Exact8SelectionError(
                f"{context} has no matching review row"
            )
        review_record, rank = match
        review = review_record.value
        finalization = _mapping(
            review["action_anchor_finalization"],
            context=f"review iid={iid} finalization",
        )
        if finalization.get("selection_bucket") != "proposed":
            raise Wan22Exact8SelectionError(
                f"generation iid={iid} is not in the proposed bucket"
            )
        if review.get("group_id") != group:
            raise Wan22Exact8SelectionError(
                f"generation iid={iid} group differs from review"
            )
        if review.get("prompt") != instruction:
            raise Wan22Exact8SelectionError(
                f"generation iid={iid} instruction differs from frozen prompt"
            )
        ranked.append((rank, record))
        iids.add(iid)
        groups.add(group)
    if len(ranked) < SELECTED_ROW_COUNT:
        raise Wan22Exact8SelectionError(
            "fewer than eight valid proposed generation rows"
        )
    return ranked


def _validate_summary_counts(
    summary: Mapping[str, Any],
    *,
    review_count: int,
    generation_count: int,
) -> None:
    selection = _mapping(
        summary.get("selection"),
        context="summary.json.selection",
    )
    for field, expected in (
        ("review_rows", review_count),
        ("generation_rows", generation_count),
        ("proposed_rows", generation_count),
    ):
        value = selection.get(field)
        if type(value) is not int or value != expected:
            raise Wan22Exact8SelectionError(
                f"summary.json.selection.{field} differs"
            )


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory_noreplace(staging: Path, output: Path) -> None:
    """Atomically publish ``staging`` without replacing ``output``."""

    source = os.fsencode(staging)
    destination = os.fsencode(output)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,  # AT_FDCWD
            source,
            -100,  # AT_FDCWD
            destination,
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            source,
            destination,
            0x00000004,  # RENAME_EXCL
        )
    else:
        raise Wan22Exact8SelectionError(
            "this platform lacks an atomic no-replace directory rename"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(output)
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(output),
    )


def select_exact8(
    *,
    finalizer_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify ``finalizer_dir`` and atomically publish an exact-eight subset."""

    parent = Path(finalizer_dir).expanduser()
    if parent.is_symlink() or not parent.is_dir():
        raise Wan22Exact8SelectionError(
            f"finalizer_dir must be a non-symlink directory: {parent}"
        )
    parent = parent.resolve(strict=True)
    output = Path(output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    (
        _done,
        summary,
        raw_outputs,
        done_raw,
        finalizer_implementation_sha,
    ) = _validate_parent_hashes(
        finalizer_dir=parent,
    )
    review_rows = _load_canonical_jsonl(
        raw_outputs[REVIEW_NAME],
        context=REVIEW_NAME,
    )
    generation_rows = _load_canonical_jsonl(
        raw_outputs[PARENT_GENERATION_NAME],
        context=PARENT_GENERATION_NAME,
    )
    review_by_iid = _validate_review_rows(review_rows)
    ranked_generation = _validate_generation_rows(
        generation_rows,
        review_by_iid=review_by_iid,
    )
    _validate_summary_counts(
        summary,
        review_count=len(review_rows),
        generation_count=len(generation_rows),
    )

    ranked_generation.sort(key=lambda item: item[0])
    selected = ranked_generation[:SELECTED_ROW_COUNT]
    selected_payload = b"".join(
        record.raw_line for _rank, record in selected
    )
    selected_iids = [
        _iid(
            record.value.get("iid"),
            context="selected generation iid",
        )
        for _rank, record in selected
    ]
    selected_ranks = [rank for rank, _record in selected]
    selected_sha = _sha256(selected_payload)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "policy": SELECTION_POLICY,
        "parent": {
            "done_sha256": _sha256(done_raw),
            "summary_sha256": _sha256(raw_outputs[SUMMARY_NAME]),
            "review_candidates_sha256": _sha256(
                raw_outputs[REVIEW_NAME]
            ),
            "generation_manifest_sha256": _sha256(
                raw_outputs[PARENT_GENERATION_NAME]
            ),
            "finalizer_implementation_sha256": (
                finalizer_implementation_sha
            ),
        },
        "selection": {
            "row_count": SELECTED_ROW_COUNT,
            "ordered_iids": selected_iids,
            "ordered_review_ranks": selected_ranks,
            "output_file": OUTPUT_MANIFEST_NAME,
            "output_sha256": selected_sha,
            "output_bytes": len(selected_payload),
        },
    }
    receipt_payload = _canonical_bytes(receipt) + b"\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        _write_new(
            staging / OUTPUT_MANIFEST_NAME,
            selected_payload,
        )
        _write_new(
            staging / OUTPUT_RECEIPT_NAME,
            receipt_payload,
        )
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
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a completed Goku action-anchor finalizer and select the "
            "eight lowest-ranked still-pending generation rows."
        )
    )
    parser.add_argument(
        "--finalizer-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = select_exact8(
        finalizer_dir=args.finalizer_dir,
        output_dir=args.output_dir,
    )
    print(
        "[wan22-select-exact8] "
        f"rows={receipt['selection']['row_count']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
