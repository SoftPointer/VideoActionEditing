#!/usr/bin/env python3
"""Freeze the v17 Qwen passes used by natural motion-instruction rewriting.

The source candidate manifest defines order.  Only existing, deeply validated
``passed/<iid>.jsonl`` fragments are selected.  This helper never writes below
either source root; materialization is create-only in a separate destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

from motive.goku_natural_motion_instruction import (
    INPUT_SCHEMA,
    _object_digest,
    _validate_passed_row,
)


SCHEMA = "motive-goku-natural-motion-input-provenance-v1"
_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class MaterializationError(RuntimeError):
    """The immutable source set or create-only destination is invalid."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MaterializationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_object(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                MaterializationError(f"non-finite JSON in {context}: {item}")
            ),
            object_pairs_hook=_closed_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError(f"invalid JSON in {context}") from error
    if not isinstance(value, dict):
        raise MaterializationError(f"{context} must be a JSON object")
    return value


def _plain_file(path: Path, *, context: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise MaterializationError(f"missing {context}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise MaterializationError(f"{context} is not a plain file: {path}")
    return info


def _plain_directory(path: Path, *, context: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise MaterializationError(f"missing {context}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise MaterializationError(f"{context} is not a plain directory: {path}")
    return info


def _read_stable(path: Path, *, context: str) -> bytes:
    before = _plain_file(path, context=context)
    raw = path.read_bytes()
    after = _plain_file(path, context=context)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise MaterializationError(f"{context} changed while being read: {path}")
    return raw


def _read_jsonl(path: Path, *, context: str) -> tuple[bytes, list[dict[str, Any]]]:
    raw = _read_stable(path, context=context)
    if not raw or not raw.endswith(b"\n"):
        raise MaterializationError(f"{context} must be nonempty newline-terminated JSONL")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise MaterializationError(f"blank line {line_number} in {context}")
        rows.append(_json_object(line, context=f"{context} line {line_number}"))
    return raw, rows


def _validate_source_link(candidate: Mapping[str, Any], passed: Mapping[str, Any]) -> None:
    iid = str(candidate["iid"])
    for candidate_field, passed_field in (
        ("group_id", "group_id"),
        ("family", "family"),
        ("source_video_sha256", "source_video_sha256"),
        ("anchor_sha256", "anchor_sha256"),
    ):
        if candidate_field in candidate and candidate[candidate_field] != passed[passed_field]:
            raise MaterializationError(
                f"candidate/pass {candidate_field} differs for iid={iid}"
            )
    if "resolved_src_video" in candidate:
        candidate_source = str(Path(str(candidate["resolved_src_video"])).resolve())
        passed_source = str(Path(str(passed["resolved_source_video"])).resolve())
        if candidate_source != passed_source:
            raise MaterializationError(f"candidate/pass source path differs for iid={iid}")
    if "resolved_anchor_image" in candidate:
        candidate_anchor = str(Path(str(candidate["resolved_anchor_image"])).resolve())
        passed_anchor = str(Path(str(passed["resolved_anchor_image"])).resolve())
        if candidate_anchor != passed_anchor:
            raise MaterializationError(f"candidate/pass anchor path differs for iid={iid}")


def inspect_sources(
    *,
    candidates: Path,
    passed_root: Path,
    expected_candidates: int,
    expected_passed: int,
    expected_candidates_sha256: str,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Return provenance plus full/smoke JSONL bytes without writing anything."""

    if not candidates.is_absolute() or not passed_root.is_absolute():
        raise MaterializationError("source paths must be absolute")
    passed_before = _plain_directory(passed_root, context="passed root")
    candidate_raw, candidate_rows = _read_jsonl(
        candidates, context="candidate manifest"
    )
    observed_candidates_sha = _sha256(candidate_raw)
    if observed_candidates_sha != expected_candidates_sha256:
        raise MaterializationError("candidate manifest SHA-256 differs")
    if len(candidate_rows) != expected_candidates:
        raise MaterializationError(
            f"expected {expected_candidates} candidates, observed {len(candidate_rows)}"
        )
    candidate_iids: list[str] = []
    for index, row in enumerate(candidate_rows):
        iid = row.get("iid")
        if not isinstance(iid, str) or _IID_RE.fullmatch(iid) is None:
            raise MaterializationError(f"unsafe candidate IID at row {index}")
        candidate_iids.append(iid)
    if len(set(candidate_iids)) != expected_candidates:
        raise MaterializationError("candidate IIDs are duplicated")

    fragment_entries = list(passed_root.iterdir())
    fragment_names: set[str] = set()
    for path in fragment_entries:
        _plain_file(path, context="passed-root entry")
        if path.suffix != ".jsonl" or _IID_RE.fullmatch(path.stem) is None:
            raise MaterializationError(f"unexpected passed-root entry: {path.name}")
        fragment_names.add(path.name)
    if len(fragment_names) != expected_passed:
        raise MaterializationError(
            f"expected {expected_passed} passed fragments, observed {len(fragment_names)}"
        )

    selected_rows: list[dict[str, Any]] = []
    fragments: list[dict[str, str]] = []
    selected_iids: list[str] = []
    for candidate_index, candidate in enumerate(candidate_rows):
        iid = str(candidate["iid"])
        name = f"{iid}.jsonl"
        if name not in fragment_names:
            continue
        fragment_path = passed_root / name
        fragment_raw, fragment_rows = _read_jsonl(
            fragment_path, context=f"passed fragment iid={iid}"
        )
        if len(fragment_rows) != 1:
            raise MaterializationError(f"passed fragment must contain one row iid={iid}")
        passed = _validate_passed_row(fragment_rows[0], expected_iid=iid)
        if passed["iid"] != iid:
            raise MaterializationError(f"passed fragment IID differs iid={iid}")
        _validate_source_link(candidate, passed)
        selected_iids.append(iid)
        wrapper: dict[str, Any] = {
            "schema_version": INPUT_SCHEMA,
            "iid": iid,
            "original_candidate_index": candidate_index,
            "candidates_path": str(candidates.resolve()),
            "candidates_sha256": observed_candidates_sha,
            "source_passed_path": str(fragment_path.resolve()),
            "source_passed_sha256": _sha256(fragment_raw),
            "passed_row": passed,
            "row_digest": None,
        }
        wrapper["row_digest"] = _object_digest(wrapper, omit="row_digest")
        selected_rows.append(wrapper)
        fragments.append({"iid": iid, "sha256": _sha256(fragment_raw)})
    if len(selected_iids) != expected_passed or set(fragment_names) != {
        f"{iid}.jsonl" for iid in selected_iids
    }:
        raise MaterializationError("passed fragments do not close against candidate order")
    passed_after = _plain_directory(passed_root, context="passed root")
    before_identity = (
        passed_before.st_dev,
        passed_before.st_ino,
        passed_before.st_mtime_ns,
        passed_before.st_ctime_ns,
    )
    after_identity = (
        passed_after.st_dev,
        passed_after.st_ino,
        passed_after.st_mtime_ns,
        passed_after.st_ctime_ns,
    )
    final_names = {path.name for path in passed_root.iterdir()}
    if before_identity != after_identity or final_names != fragment_names:
        raise MaterializationError("passed root changed while being inspected")

    full_raw = b"".join(_canonical(row) + b"\n" for row in selected_rows)
    smoke_raw = b"".join(_canonical(row) + b"\n" for row in selected_rows[:8])
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA,
        "candidates": str(candidates.resolve()),
        "candidates_sha256": observed_candidates_sha,
        "candidate_count": len(candidate_rows),
        "passed_root": str(passed_root.resolve()),
        "passed_count": len(selected_iids),
        "passed_set_sha256": _sha256(_canonical(fragments)),
        "full_input_sha256": _sha256(full_raw),
        "smoke8_input_sha256": _sha256(smoke_raw),
        "smoke_count": min(8, len(selected_iids)),
        "ordered_iids_sha256": _sha256(_canonical(selected_iids)),
        "fragments": fragments,
    }
    return provenance, full_raw, smoke_raw


def _write_create_only(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    provenance, full_raw, smoke_raw = inspect_sources(
        candidates=args.candidates,
        passed_root=args.passed_root,
        expected_candidates=args.expected_candidates,
        expected_passed=args.expected_passed,
        expected_candidates_sha256=args.expected_candidates_sha256,
    )
    if provenance["passed_set_sha256"] != args.expected_passed_set_sha256:
        raise MaterializationError("passed fragment-set SHA-256 differs")
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        raise MaterializationError("output directory must be absolute")
    try:
        output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as error:
        raise MaterializationError(
            f"create-only output directory already exists: {output_dir}"
        ) from error
    _write_create_only(output_dir / "full673.jsonl", full_raw)
    _write_create_only(output_dir / "smoke8.jsonl", smoke_raw)
    payload = json.dumps(
        provenance, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    _write_create_only(output_dir / "provenance.json", payload)
    directory_fd = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "materialize"))
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--passed-root", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=1000)
    parser.add_argument("--expected-passed", type=int, default=673)
    parser.add_argument("--expected-candidates-sha256", required=True)
    parser.add_argument("--expected-passed-set-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--output-format",
        choices=("json", "passed-set-sha256"),
        default="json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _SHA256_RE.fullmatch(args.expected_candidates_sha256) is None:
        raise MaterializationError("expected candidates SHA-256 is invalid")
    if args.expected_candidates < 1 or args.expected_passed < 8:
        raise MaterializationError("invalid expected source counts")
    if args.mode == "inspect":
        if args.output_dir is not None or args.expected_passed_set_sha256 is not None:
            raise MaterializationError("inspect mode is read-only")
        provenance, _, _ = inspect_sources(
            candidates=args.candidates,
            passed_root=args.passed_root,
            expected_candidates=args.expected_candidates,
            expected_passed=args.expected_passed,
            expected_candidates_sha256=args.expected_candidates_sha256,
        )
    else:
        if args.output_dir is None:
            raise MaterializationError("materialize mode requires --output-dir")
        if (
            not isinstance(args.expected_passed_set_sha256, str)
            or _SHA256_RE.fullmatch(args.expected_passed_set_sha256) is None
        ):
            raise MaterializationError(
                "materialize mode requires a valid --expected-passed-set-sha256"
            )
        provenance = materialize(args)
    if args.output_format == "passed-set-sha256":
        sys.stdout.write(str(provenance["passed_set_sha256"]) + "\n")
    else:
        sys.stdout.write(
            json.dumps(
                provenance, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
