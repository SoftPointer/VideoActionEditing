#!/usr/bin/env python3
"""Build the exact644 source+instruction catalogue without opening target media.

The input preview JSONL and raw parquet are immutable upstream authorities.  The
preview rows contain target *metadata*, but this extractor never opens, hashes,
decodes, or copies a target media path.  It cross-joins only the source role and
instruction against a selected-column raw-parquet projection, verifies every
source MP4 as exact81/25fps, and publishes a create-only source catalogue.

The script is a derivation tool, not standalone authority: its receipt records
that an external release must pin the final script and output bytes before a
trainer may consume them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, NoReturn, Optional, Sequence, Tuple


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full644_target_free_preference_v1 as target_free  # noqa: E402


RECEIPT_SCHEMA = "bernini-full644-target-free-source-catalog-receipt-v1"
PREVIEW_ROW_SCHEMA = "omnivideo2-action-preview-row-v1"
NATURAL_ROW_SCHEMA = "motive-goku-natural-motion-dataset-row-v1"
CATALOG_FILENAME = "source_catalog.json"
RECEIPT_FILENAME = "source_catalog_receipt.json"
STRICT_SOURCE_COUNT = 359
BROAD_SOURCE_COUNT = 285
RAW_ROW_SCHEMA = "bernini-r-action-raw-row-v2"
EXPECTED_RAW_PARQUET_COLUMNS = (
    "schema_version",
    "inputs",
    "videos",
    "iid",
    "group_id",
    "family",
    "edit_instruction_sha256",
    "source_video_path",
    "source_video_declared_path",
    "source_video_sha256",
    "target_video_path",
    "target_video_declared_path",
    "target_video_sha256",
    "shared_i0_path",
    "shared_i0_sha256",
    "preview_manifest_path",
    "preview_manifest_sha256",
    "preview_row_digest",
    "preview_row_file_sha256",
    "experimental_inclusion_policy",
    "selection_gates_json",
    "strict_selection_gates_all_true",
    "upstream_authorization_json",
    "preview_only",
    "training_authorized",
    "training_use_forbidden",
    "production_eligible",
    "post_video_acceptance",
    "experimental_training_acknowledged",
    "production_claim_forbidden",
    "renderer_row_digest",
)
RAW_SAFE_COLUMNS = (
    "schema_version",
    "iid",
    "group_id",
    "family",
    "inputs",
    "videos",
    "source_video_path",
    "source_video_declared_path",
    "source_video_sha256",
    "edit_instruction_sha256",
    "preview_manifest_sha256",
    "preview_row_digest",
    "preview_row_file_sha256",
    "strict_selection_gates_all_true",
    "preview_only",
    "training_authorized",
    "training_use_forbidden",
    "production_eligible",
    "post_video_acceptance",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")


class SourceCatalogExtractionError(RuntimeError):
    """Raised before an unbound source role or target-media read can publish."""


def fail(message: str) -> NoReturn:
    raise SourceCatalogExtractionError(message)


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be one lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        fail(f"{label} must be one safe identifier")
    return value


def _absolute_lexical_path(value: Any, *, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        fail(f"{label} must be one path string")
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or any(
        part in ("", ".", "..") for part in path.parts[1:]
    ):
        fail(f"{label} must be one absolute lexical path")
    return path


def _upstream_canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise SourceCatalogExtractionError(
            "upstream row is not canonical finite UTF-8 JSON"
        ) from error


def _parse_jsonl(raw: bytes, *, label: str) -> list[Mapping[str, Any]]:
    if not raw or not raw.endswith(b"\n"):
        fail(f"{label} must be non-empty newline-terminated JSONL")
    lines = raw.splitlines()
    if any(not line for line in lines):
        fail(f"{label} contains a blank row")
    return [
        target_free._strict_json(line, label=f"{label} line {index}")
        for index, line in enumerate(lines, start=1)
    ]


def _iid_set_sha256(iids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{iid}\n" for iid in sorted(iids)).encode("utf-8")
    ).hexdigest()


def _instruction_from_inputs(value: Any, *, iid: str) -> str:
    if type(value) is not str:
        fail(f"raw inputs differ for {iid}")

    def reject_constant(constant: str) -> None:
        fail(f"raw inputs contain non-finite constant for {iid}: {constant}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                fail(f"raw inputs contain duplicate key for {iid}: {key}")
            result[key] = item
        return result

    try:
        messages = json.loads(
            value,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SourceCatalogExtractionError(f"raw inputs are invalid for {iid}") from error
    if (
        type(messages) is not list
        or len(messages) != 3
        or type(messages[0]) is not dict
        or set(messages[0]) != {"type", "has_loss"}
        or messages[0].get("type") != "video"
        or type(messages[0].get("has_loss")) is not int
        or messages[0].get("has_loss") != 0
        or type(messages[1]) is not dict
        or set(messages[1]) != {"type", "text", "has_loss"}
        or messages[1].get("type") != "text"
        or type(messages[1].get("has_loss")) is not int
        or messages[1].get("has_loss") != 0
        or type(messages[1].get("text")) is not str
        or type(messages[2]) is not dict
        or set(messages[2]) != {"type", "has_loss"}
        or messages[2].get("type") != "video_gen"
        or type(messages[2].get("has_loss")) is not int
        or messages[2].get("has_loss") != 1
    ):
        fail(f"raw source/instruction message ABI differs for {iid}")
    return str(messages[1]["text"])


def _video_role_paths(value: Any, *, iid: str) -> tuple[str, str]:
    """Bind Bernini's positional video ABI without touching either media path."""

    if type(value) is not list or len(value) != 2:
        fail(f"raw Bernini video-role sequence differs for {iid}")
    paths: list[str] = []
    for index, record in enumerate(value):
        if (
            type(record) is not dict
            or set(record) != {"video_path"}
            or type(record.get("video_path")) is not str
        ):
            fail(f"raw Bernini video role {index} differs for {iid}")
        paths.append(
            str(
                _absolute_lexical_path(
                    record["video_path"],
                    label=f"raw Bernini video role {index} for {iid}",
                )
            )
        )
    return paths[0], paths[1]


def _validate_preview_rows(raw: bytes) -> Mapping[str, Mapping[str, Any]]:
    rows = _parse_jsonl(raw, label="pinned preview manifest")
    if len(rows) != target_free.SOURCE_COUNT:
        fail("preview manifest must contain exact644 rows")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != PREVIEW_ROW_SCHEMA:
            fail("preview row schema differs")
        iid = _safe_id(row.get("iid"), label="preview IID")
        if iid in indexed:
            fail(f"duplicate preview IID {iid}")
        declared_digest = _sha(row.get("row_digest"), label=f"preview row {iid} digest")
        unsigned = dict(row)
        unsigned.pop("row_digest", None)
        if hashlib.sha256(_upstream_canonical_json_bytes(unsigned)).hexdigest() != declared_digest:
            fail(f"preview row digest differs for {iid}")
        instruction = row.get("edit_instruction")
        instruction_sha = _sha(
            row.get("edit_instruction_sha256"), label=f"preview instruction {iid} SHA"
        )
        if (
            type(instruction) is not str
            or not instruction.strip()
            or hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            != instruction_sha
        ):
            fail(f"preview instruction differs for {iid}")
        _safe_id(row.get("group_id"), label=f"preview group {iid}")
        _safe_id(row.get("family"), label=f"preview family {iid}")
        _absolute_lexical_path(row.get("source_video_path"), label=f"preview source {iid}")
        _sha(row.get("source_video_sha256"), label=f"preview source {iid} SHA")
        # Target fields are validated as immutable metadata only.  No filesystem
        # operation in this module receives this path.
        _absolute_lexical_path(row.get("target_video_path"), label=f"preview target {iid}")
        _sha(row.get("target_video_sha256"), label=f"preview target {iid} SHA")
        if (
            row.get("preview_only") is not True
            or row.get("training_authorized") is not False
            or row.get("training_use_forbidden") is not True
            or row.get("production_eligible") is not False
            or row.get("post_video_acceptance") != "pending"
        ):
            fail(f"preview authorization state differs for {iid}")
        indexed[iid] = row
    if [str(row["iid"]) for row in rows] != sorted(indexed):
        fail("preview rows are not ASCII IID ordered")
    return indexed


def _validate_natural_rows(
    raw: bytes, *, preview_by_iid: Mapping[str, Mapping[str, Any]]
) -> None:
    rows = _parse_jsonl(raw, label="pinned natural manifest")
    if len(rows) != target_free.SOURCE_COUNT:
        fail("natural manifest must contain exact644 accepted rows")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != NATURAL_ROW_SCHEMA:
            fail("natural row schema differs")
        iid = _safe_id(row.get("iid"), label="natural IID")
        instruction = row.get("natural_edit_instruction")
        instruction_sha = _sha(
            row.get("natural_edit_instruction_sha256"),
            label=f"natural instruction {iid} SHA",
        )
        if (
            iid in indexed
            or type(instruction) is not str
            or hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            != instruction_sha
        ):
            fail(f"natural row differs for {iid}")
        preview = preview_by_iid.get(iid)
        if (
            preview is None
            or preview.get("edit_instruction") != instruction
            or preview.get("edit_instruction_sha256") != instruction_sha
        ):
            fail(f"natural/preview instruction join differs for {iid}")
        indexed[iid] = row
    if set(indexed) != set(preview_by_iid):
        fail("natural/preview IID set differs")
    expected = target_free.PINNED_FULL644_SOURCE_AUTHORITY["sorted_iid_set_sha256"]
    if _iid_set_sha256(tuple(indexed)) != expected:
        fail("natural exact644 IID-set digest differs")


def _load_raw_source_projection(raw: bytes) -> list[Mapping[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise SourceCatalogExtractionError("pyarrow is required to read raw parquet") from error
    try:
        schema = pq.read_schema(pa.BufferReader(raw))
        if tuple(schema.names) != EXPECTED_RAW_PARQUET_COLUMNS:
            fail("raw parquet schema field order differs")
        table = pq.read_table(pa.BufferReader(raw), columns=list(RAW_SAFE_COLUMNS))
        rows = table.to_pylist()
    except Exception as error:
        raise SourceCatalogExtractionError(
            "cannot read held raw source-only column projection"
        ) from error
    if len(rows) != target_free.SOURCE_COUNT:
        fail("raw source projection must contain exact644 rows")
    return rows


def _probe_exact81_25(
    path: Path, *, ffprobe: str, pass_fds: Sequence[int] = ()
) -> Mapping[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v",
        "-count_frames",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=tuple(pass_fds),
            env={"LANG": "C", "LC_ALL": "C"},
        )
        value = json.loads(completed.stdout.decode("utf-8", errors="strict"))
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError) as error:
        raise SourceCatalogExtractionError(f"ffprobe failed for {path}") from error
    streams = value.get("streams") if isinstance(value, Mapping) else None
    if type(streams) is not list or len(streams) != 1 or type(streams[0]) is not dict:
        fail(f"source video stream closure differs for {path}")
    stream = streams[0]
    count_text = stream.get("nb_read_frames")
    try:
        frame_count = int(count_text)
        avg_numerator, avg_denominator = str(stream.get("avg_frame_rate")).split(
            "/", 1
        )
        real_numerator, real_denominator = str(stream.get("r_frame_rate")).split(
            "/", 1
        )
        fps = int(avg_numerator) / int(avg_denominator)
        real_fps = int(real_numerator) / int(real_denominator)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise SourceCatalogExtractionError(f"source media probe differs for {path}") from error
    declared_frames = stream.get("nb_frames")
    if (
        frame_count != target_free.FRAME_COUNT
        or fps != target_free.FPS
        or real_fps != target_free.FPS
        or (
            declared_frames not in (None, "N/A")
            and int(declared_frames) != target_free.FRAME_COUNT
        )
    ):
        fail(f"source is not exact81/25fps: {path}")
    return {"frame_count": frame_count, "fps": float(fps)}


def _read_and_probe_stable_source(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    ffprobe: str,
    ffprobe_pass_fds: Sequence[int] = (),
) -> tuple[bytes, Mapping[str, Any]]:
    """Hash and ffprobe the same retained file description, then replay it."""

    expected = _sha(expected_sha256, label=f"{label} SHA")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourceCatalogExtractionError(f"cannot safely open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            fail(f"{label} must be one regular nlink1 source file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        if len(payload) != int(before.st_size) or hashlib.sha256(payload).hexdigest() != expected:
            fail(f"{label} bytes differ")
        fd_root = Path("/proc/self/fd")
        if not fd_root.is_dir():  # macOS unit/development fallback
            fd_root = Path("/dev/fd")
        probe = _probe_exact81_25(
            fd_root / str(descriptor),
            ffprobe=ffprobe,
            pass_fds=tuple(dict.fromkeys((*ffprobe_pass_fds, descriptor))),
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        replay = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            replay.extend(block)
        after = os.fstat(descriptor)
        identity_before = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mode),
            int(before.st_nlink),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        )
        identity_after = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mode),
            int(after.st_nlink),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
        )
        if (
            identity_before != identity_after
            or bytes(replay) != payload
            or hashlib.sha256(replay).hexdigest() != expected
        ):
            fail(f"{label} changed across held-FD decode")
        return payload, probe
    finally:
        os.close(descriptor)


SourceLoader = Callable[[Path, str, str], Tuple[bytes, Mapping[str, Any]]]


def _fd_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _read_fd_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    return b"".join(chunks)


def _linux_fd_executable_alias(descriptor: int) -> Path:
    root = Path("/proc/self/fd")
    if not root.is_dir():
        fail("production extractor requires Linux /proc/self/fd execution")
    return root / str(descriptor)


def _open_verified_executable(
    path: Path, *, expected_sha256: str, label: str
) -> tuple[int, Mapping[str, Any], tuple[int, ...]]:
    expected = _sha(expected_sha256, label=f"{label} SHA")
    canonical = path.resolve(strict=True)
    if canonical != path:
        fail(f"{label} path must be absolute and canonical")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourceCatalogExtractionError(f"cannot safely open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or stat.S_IMODE(before.st_mode) & 0o111 == 0
        ):
            fail(f"{label} must be one executable regular nlink1 file")
        payload = _read_fd_bytes(descriptor)
        after = os.fstat(descriptor)
        identity = _fd_identity(before)
        if (
            len(payload) != int(before.st_size)
            or hashlib.sha256(payload).hexdigest() != expected
            or _fd_identity(after) != identity
        ):
            fail(f"{label} executable bytes differ")
        binding = {
            "path": str(path),
            "sha256": expected,
            "size": len(payload),
            "mode": stat.S_IMODE(before.st_mode),
            "nlink": int(before.st_nlink),
            "held_fd_execution": True,
            "clean_environment": {"LANG": "C", "LC_ALL": "C"},
        }
        return descriptor, binding, identity
    except Exception:
        os.close(descriptor)
        raise


def _replay_verified_executable(
    descriptor: int,
    *,
    path: Path,
    expected_sha256: str,
    expected_identity: tuple[int, ...],
    label: str,
) -> None:
    payload = _read_fd_bytes(descriptor)
    after = os.fstat(descriptor)
    try:
        live = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise SourceCatalogExtractionError(f"{label} pathname disappeared: {error}") from error
    if (
        _fd_identity(after) != expected_identity
        or hashlib.sha256(payload).hexdigest() != expected_sha256
        or int(live.st_dev) != int(after.st_dev)
        or int(live.st_ino) != int(after.st_ino)
    ):
        fail(f"{label} changed across exact644 extraction")


def _build_catalog_value_with_source_loader_for_tests_v1(
    *,
    preview_raw: bytes,
    natural_raw: bytes,
    raw_rows: Sequence[Mapping[str, Any]],
    source_loader: SourceLoader,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Pure join helper; its injected-loader evidence is never authoritative."""

    preview_by_iid = _validate_preview_rows(preview_raw)
    _validate_natural_rows(natural_raw, preview_by_iid=preview_by_iid)
    if type(raw_rows) not in (list, tuple) or len(raw_rows) != target_free.SOURCE_COUNT:
        fail("raw source projection must contain exact644 rows")
    raw_by_iid: dict[str, Mapping[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            fail("raw source projection row must be an object")
        iid = _safe_id(raw.get("iid"), label="raw IID")
        if iid in raw_by_iid:
            fail(f"duplicate raw IID {iid}")
        raw_by_iid[iid] = raw
    if set(raw_by_iid) != set(preview_by_iid):
        fail("raw/preview IID set differs")

    output_rows: list[Mapping[str, Any]] = []
    source_bytes = 0
    strict_source_count = 0
    source_paths: set[str] = set()
    target_paths = {
        str(preview_by_iid[iid]["target_video_path"]) for iid in preview_by_iid
    }
    target_shas = {
        str(preview_by_iid[iid]["target_video_sha256"]) for iid in preview_by_iid
    }
    for iid in sorted(preview_by_iid):
        preview = preview_by_iid[iid]
        raw = raw_by_iid[iid]
        instruction = _instruction_from_inputs(raw.get("inputs"), iid=iid)
        source_path = _absolute_lexical_path(
            raw.get("source_video_path"), label=f"raw source {iid}"
        )
        raw_index0_path, raw_index1_path = _video_role_paths(
            raw.get("videos"), iid=iid
        )
        if (
            raw.get("schema_version") != RAW_ROW_SCHEMA
            or raw.get("group_id") != preview.get("group_id")
            or raw.get("family") != preview.get("family")
            or raw_index0_path != str(source_path)
            or raw_index1_path != preview.get("target_video_path")
            or raw.get("source_video_declared_path") != str(source_path)
            or str(source_path) != preview.get("source_video_path")
            or raw.get("source_video_sha256") != preview.get("source_video_sha256")
            or raw.get("edit_instruction_sha256")
            != preview.get("edit_instruction_sha256")
            or instruction != preview.get("edit_instruction")
            or raw.get("preview_manifest_sha256")
            != target_free.PINNED_FULL644_SOURCE_AUTHORITY["preview_manifest_sha256"]
            or raw.get("preview_row_digest") != preview.get("row_digest")
            or type(raw.get("strict_selection_gates_all_true")) is not bool
            or raw.get("preview_only") is not True
            or raw.get("training_authorized") is not False
            or raw.get("training_use_forbidden") is not True
            or raw.get("production_eligible") is not False
            or raw.get("post_video_acceptance") != "pending"
        ):
            fail(f"raw/preview source-role join differs for {iid}")
        strict_source_count += int(raw["strict_selection_gates_all_true"])
        row_file_sha = hashlib.sha256(
            _upstream_canonical_json_bytes(preview) + b"\n"
        ).hexdigest()
        if raw.get("preview_row_file_sha256") != row_file_sha:
            fail(f"raw/preview row-file join differs for {iid}")
        source_sha = _sha(raw.get("source_video_sha256"), label=f"raw source {iid} SHA")
        if str(source_path) in target_paths or source_sha in target_shas:
            fail(f"source aliases an upstream target role for {iid}")
        payload, probe = source_loader(
            source_path, source_sha, f"source video {iid}"
        )
        if (
            type(payload) is not bytes
            or hashlib.sha256(payload).hexdigest() != source_sha
        ):
            fail(f"source loader returned unbound bytes for {iid}")
        source_bytes += len(payload)
        if (
            type(probe.get("frame_count")) is not int
            or probe.get("frame_count") != target_free.FRAME_COUNT
            or type(probe.get("fps")) is not float
            or probe.get("fps") != target_free.FPS
        ):
            fail(f"source media exact81/25 closure differs for {iid}")
        source_paths.add(str(source_path))
        unsigned = {
            "schema_version": target_free.SOURCE_ROW_SCHEMA,
            "row_id": iid,
            "group_id": str(raw["group_id"]),
            "action_family": str(raw["family"]),
            "source_video_path": str(source_path),
            "source_video_sha256": source_sha,
            "source_frame_count": target_free.FRAME_COUNT,
            "source_fps": target_free.FPS,
            "instruction": instruction,
            "instruction_sha256": str(raw["edit_instruction_sha256"]),
            "upstream_preview_row_digest": str(preview["row_digest"]),
        }
        output_rows.append({**unsigned, "row_digest": target_free.object_sha256(unsigned)})
    if len(source_paths) != target_free.SOURCE_COUNT:
        fail("source path inventory is not exact644 unique")
    if (
        strict_source_count != STRICT_SOURCE_COUNT
        or target_free.SOURCE_COUNT - strict_source_count != BROAD_SOURCE_COUNT
    ):
        fail("full644 strict/broad source partition differs")
    unsigned_root = {
        "schema_version": target_free.SOURCE_SCHEMA,
        "training_mode": target_free.TRAINING_MODE,
        "source_count": target_free.SOURCE_COUNT,
        "action_family_count": target_free.ACTION_FAMILY_COUNT,
        "rows": output_rows,
        "source_authority": dict(target_free.PINNED_FULL644_SOURCE_AUTHORITY),
        "row_order": "ascii_ascending_row_id",
        "input_closure": dict(target_free.SOURCE_INPUT_CLOSURE),
    }
    catalog = {
        **unsigned_root,
        "manifest_digest": target_free.object_sha256(unsigned_root),
    }
    catalog_sha = hashlib.sha256(target_free.canonical_json_bytes(catalog)).hexdigest()
    target_free.validate_source_catalog_value(
        catalog, manifest_sha256=catalog_sha, require_source_files=False
    )
    evidence = {
        "source_file_open_count": target_free.SOURCE_COUNT,
        "source_file_total_bytes": source_bytes,
        "strict_source_count": strict_source_count,
        "broad_source_count": target_free.SOURCE_COUNT - strict_source_count,
        "access_ledger_authoritative": False,
        "source_loader_contract": "INJECTED_TEST_JOIN_HELPER_NOT_FOR_PUBLICATION",
        "raw_columns_read": list(RAW_SAFE_COLUMNS),
        "raw_videos_role_column_read": True,
        "raw_target_path_metadata_read_for_role_rejection": True,
        "source_paths_unique": True,
        "source_target_path_disjoint": True,
        "source_probe_inventory_digest": target_free.object_sha256(
            [
                {
                    "row_id": row["row_id"],
                    "source_video_sha256": row["source_video_sha256"],
                    "source_frame_count": row["source_frame_count"],
                    "source_fps": row["source_fps"],
                }
                for row in output_rows
            ]
        ),
    }
    return catalog, evidence


def _create_only_file_at(
    directory_fd: int, name: str, payload: bytes
) -> Mapping[str, Any]:
    if type(name) is not str or not name or "/" in name or name in (".", ".."):
        fail("create-only filename is not one basename")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise SourceCatalogExtractionError(f"cannot create {name}: {error}") from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                fail(f"short create-only write for {name}")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        replay = _read_fd_bytes(descriptor)
        final_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or int(info.st_nlink) != 1
            or stat.S_IMODE(info.st_mode) != 0o444
            or _fd_identity(info) != _fd_identity(final_info)
            or replay != payload
        ):
            fail(f"published file identity or bytes differ: {name}")
        return {
            "name": name,
            "sha256": hashlib.sha256(replay).hexdigest(),
            "size": len(replay),
            "mode": stat.S_IMODE(final_info.st_mode),
            "nlink": int(final_info.st_nlink),
            "device": int(final_info.st_dev),
            "inode": int(final_info.st_ino),
            "same_fd_replay_verified": True,
        }
    finally:
        os.close(descriptor)


def _open_canonical_parent(path: Path) -> tuple[int, tuple[int, int]]:
    canonical = path.resolve(strict=True)
    if canonical != path:
        fail("output parent must be one canonical directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourceCatalogExtractionError(f"cannot hold output parent: {error}") from error
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        fail("output parent is not one directory")
    return descriptor, (int(info.st_dev), int(info.st_ino))


def _create_output_directory_at(parent_fd: int, name: str) -> tuple[int, tuple[int, int]]:
    if type(name) is not str or not name or "/" in name or name in (".", ".."):
        fail("output root basename differs")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise SourceCatalogExtractionError(f"cannot create held output root: {error}") from error
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        os.close(descriptor)
        fail("fresh output directory identity differs")
    return descriptor, (int(info.st_dev), int(info.st_ino))


def _seal_output_directory(
    *,
    parent_fd: int,
    parent_path: Path,
    expected_parent_identity: tuple[int, int],
    output_fd: int,
    output_name: str,
    expected_output_identity: tuple[int, int],
    expected_file_bindings: Sequence[Mapping[str, Any]],
) -> None:
    expected_names = {CATALOG_FILENAME, RECEIPT_FILENAME}
    if (
        type(expected_file_bindings) not in (list, tuple)
        or len(expected_file_bindings) != 2
        or {binding.get("name") for binding in expected_file_bindings}
        != expected_names
        or set(os.listdir(output_fd)) != expected_names
    ):
        fail("published output directory entries differ")
    retained_files: list[tuple[int, Mapping[str, Any]]] = []
    try:
        for binding in expected_file_bindings:
            name = str(binding["name"])
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(name, flags, dir_fd=output_fd)
            except OSError as error:
                raise SourceCatalogExtractionError(
                    f"cannot replay published file {name}: {error}"
                ) from error
            retained_files.append((descriptor, binding))
            info = os.fstat(descriptor)
            try:
                named = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
            except OSError as error:
                raise SourceCatalogExtractionError(
                    f"cannot stat published file {name}: {error}"
                ) from error
            replay = _read_fd_bytes(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or int(info.st_nlink) != 1
                or stat.S_IMODE(info.st_mode) != 0o444
                or len(replay) != binding.get("size")
                or hashlib.sha256(replay).hexdigest() != binding.get("sha256")
                or int(info.st_dev) != binding.get("device")
                or int(info.st_ino) != binding.get("inode")
                or int(named.st_dev) != int(info.st_dev)
                or int(named.st_ino) != int(info.st_ino)
            ):
                fail(f"published file replay differs: {name}")
        os.fchmod(output_fd, 0o555)
        os.fsync(output_fd)
        if set(os.listdir(output_fd)) != expected_names:
            fail("post-seal output directory entries differ")
        output_info = os.fstat(output_fd)
        parent_info = os.fstat(parent_fd)
        try:
            parent_live = os.stat(parent_path, follow_symlinks=False)
            output_live = os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise SourceCatalogExtractionError(
                f"published path replay failed: {error}"
            ) from error
        if (
            (int(parent_info.st_dev), int(parent_info.st_ino))
            != expected_parent_identity
            or (int(parent_live.st_dev), int(parent_live.st_ino))
            != expected_parent_identity
            or (int(output_info.st_dev), int(output_info.st_ino))
            != expected_output_identity
            or (int(output_live.st_dev), int(output_live.st_ino))
            != expected_output_identity
            or stat.S_IMODE(output_info.st_mode) != 0o555
            or stat.S_IMODE(output_live.st_mode) != 0o555
        ):
            fail("published directory identity replay differs")
        for descriptor, binding in retained_files:
            info = os.fstat(descriptor)
            named = os.stat(
                str(binding["name"]), dir_fd=output_fd, follow_symlinks=False
            )
            replay = _read_fd_bytes(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or int(info.st_nlink) != 1
                or stat.S_IMODE(info.st_mode) != 0o444
                or len(replay) != binding.get("size")
                or hashlib.sha256(replay).hexdigest() != binding.get("sha256")
                or int(info.st_dev) != binding.get("device")
                or int(info.st_ino) != binding.get("inode")
                or int(named.st_dev) != int(info.st_dev)
                or int(named.st_ino) != int(info.st_ino)
            ):
                fail(f"post-seal file replay differs: {binding['name']}")
        os.fsync(parent_fd)
    finally:
        for descriptor, _binding in retained_files:
            os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-manifest", required=True)
    parser.add_argument("--natural-manifest", required=True)
    parser.add_argument("--raw-parquet", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--ffprobe-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    preview_path = _absolute_lexical_path(args.preview_manifest, label="preview manifest")
    natural_path = _absolute_lexical_path(args.natural_manifest, label="natural manifest")
    raw_path = _absolute_lexical_path(args.raw_parquet, label="raw parquet")
    output_root = _absolute_lexical_path(args.output_root, label="output root")
    ffprobe_path = _absolute_lexical_path(args.ffprobe, label="ffprobe executable")
    ffprobe_sha = _sha(args.ffprobe_sha256, label="ffprobe executable SHA")
    parent_fd, parent_identity = _open_canonical_parent(output_root.parent)
    ffprobe_fd = -1
    output_fd = -1
    try:
        try:
            os.stat(output_root.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            fail("output root must be a fresh path")
        ffprobe_fd, ffprobe_binding, ffprobe_identity = (
            _open_verified_executable(
                ffprobe_path,
                expected_sha256=ffprobe_sha,
                label="pinned ffprobe executable",
            )
        )
        ffprobe_alias = _linux_fd_executable_alias(ffprobe_fd)
        authority = target_free.PINNED_FULL644_SOURCE_AUTHORITY
        preview_raw = target_free._read_stable_file(
            preview_path,
            expected_sha256=str(authority["preview_manifest_sha256"]),
            label="pinned preview manifest",
        )
        natural_raw = target_free._read_stable_file(
            natural_path,
            expected_sha256=str(authority["natural_manifest_sha256"]),
            label="pinned natural manifest",
        )
        raw_bytes = target_free._read_stable_file(
            raw_path,
            expected_sha256=str(authority["raw_parquet_sha256"]),
            label="pinned raw parquet",
        )
        raw_rows = _load_raw_source_projection(raw_bytes)

        def source_loader(
            path: Path, expected_sha: str, label: str
        ) -> tuple[bytes, Mapping[str, Any]]:
            return _read_and_probe_stable_source(
                path,
                expected_sha256=expected_sha,
                label=label,
                ffprobe=str(ffprobe_alias),
                ffprobe_pass_fds=(ffprobe_fd,),
            )

        catalog, join_evidence = _build_catalog_value_with_source_loader_for_tests_v1(
            preview_raw=preview_raw,
            natural_raw=natural_raw,
            raw_rows=raw_rows,
            source_loader=source_loader,
        )
        _replay_verified_executable(
            ffprobe_fd,
            path=ffprobe_path,
            expected_sha256=ffprobe_sha,
            expected_identity=ffprobe_identity,
            label="pinned ffprobe executable",
        )
        evidence = {
            **join_evidence,
            "access_ledger_authoritative": True,
            "source_loader_contract": (
                "INTERNAL_HELD_SOURCE_FD_AND_PINNED_HELD_FFPROBE_ONLY"
            ),
            "extractor_owned_target_media_open_count": 0,
            "legacy_pair_shard_open_count": 0,
            "target_media_payload_read": False,
        }
        output_fd, output_identity = _create_output_directory_at(
            parent_fd, output_root.name
        )
        catalog_path = output_root / CATALOG_FILENAME
        receipt_path = output_root / RECEIPT_FILENAME
        catalog_bytes = target_free.canonical_json_bytes(catalog)
        catalog_binding = _create_only_file_at(
            output_fd, CATALOG_FILENAME, catalog_bytes
        )
        catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()
        if catalog_binding["sha256"] != catalog_sha:
            fail("published catalog binding differs")
        self_path = Path(__file__).resolve(strict=True)
        self_bytes = self_path.read_bytes()
        unsigned_receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "SOURCE_ONLY_EXACT644_CATALOG_COMPLETE",
            "training_mode": target_free.TRAINING_MODE,
            "catalog_path": str(catalog_path),
            "catalog_sha256": catalog_sha,
            "catalog_size": len(catalog_bytes),
            "catalog_digest": str(catalog["manifest_digest"]),
            "catalog_file_binding": catalog_binding,
            "source_count": len(catalog["rows"]),
            "action_family_count": len(
                {str(row["action_family"]) for row in catalog["rows"]}
            ),
            "preview_manifest_sha256": str(authority["preview_manifest_sha256"]),
            "natural_manifest_sha256": str(authority["natural_manifest_sha256"]),
            "raw_parquet_sha256": str(authority["raw_parquet_sha256"]),
            "sorted_iid_set_sha256": str(authority["sorted_iid_set_sha256"]),
            "ffprobe_binding": ffprobe_binding,
            "ffprobe_held_fd_prepost_replay_verified": True,
            "extractor_self_sha256": hashlib.sha256(self_bytes).hexdigest(),
            "extractor_self_size": len(self_bytes),
            "self_pin_is_observation_not_standalone_authority": True,
            "trainer_consumption_requires_external_release_pin": True,
            "target_media_used": False,
            "paired_edited_target_present": False,
            "evidence": evidence,
        }
        receipt = {
            **unsigned_receipt,
            "receipt_digest": target_free.object_sha256(unsigned_receipt),
        }
        receipt_bytes = target_free.canonical_json_bytes(receipt)
        receipt_binding = _create_only_file_at(
            output_fd, RECEIPT_FILENAME, receipt_bytes
        )
        _seal_output_directory(
            parent_fd=parent_fd,
            parent_path=output_root.parent,
            expected_parent_identity=parent_identity,
            output_fd=output_fd,
            output_name=output_root.name,
            expected_output_identity=output_identity,
            expected_file_bindings=(catalog_binding, receipt_binding),
        )
        print(receipt_bytes.decode("ascii"))
        return 0
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        if ffprobe_fd >= 0:
            os.close(ffprobe_fd)
        os.close(parent_fd)


if __name__ == "__main__":
    raise SystemExit(main())
