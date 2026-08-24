#!/usr/bin/env python3
"""Copy source videos beside committed Wan2.2 outputs without changing signed files."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "motive-wan22-source-copy-v1"
PAIR_SCHEMA_VERSION = "motive-wan22-local-source-target-pair-v1"
SAMPLE_SCHEMA_VERSION = "motive-wan22-i2v-sample-v1"
IID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class MaterializationError(RuntimeError):
    """Raised when provenance or no-clobber validation fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_object_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MaterializationError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_bytes(payload: Any) -> bytes:
    return canonical_object_bytes(payload) + b"\n"


def object_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_object_bytes(payload)).hexdigest()


def reject_constant(value: str) -> None:
    raise MaterializationError(f"non-finite JSON constant is forbidden: {value}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise MaterializationError(f"duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def strict_json_loads(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise MaterializationError(f"invalid JSON in {label}: {exc}") from exc


def require_regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise MaterializationError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MaterializationError(
            f"{label} must be a regular non-symlink file: {path}"
        )


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_regular_file(path, label)
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"), label)
    except (OSError, UnicodeDecodeError) as exc:
        raise MaterializationError(f"invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MaterializationError(f"{label} must contain a JSON object: {path}")
    return payload


def load_manifest(path: Path, raw: bytes | None = None) -> list[dict[str, Any]]:
    require_regular_file(path, "generation manifest")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if raw is None:
        raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MaterializationError(f"manifest is not valid UTF-8: {path}: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = strict_json_loads(line, f"manifest line {line_number}")
        if not isinstance(row, dict):
            raise MaterializationError(
                f"manifest line {line_number} must be a JSON object"
            )
        iid = row.get("iid")
        if not isinstance(iid, str) or not IID_RE.fullmatch(iid):
            raise MaterializationError(
                f"manifest line {line_number} has an unsafe iid: {iid!r}"
            )
        if iid in seen:
            raise MaterializationError(f"duplicate manifest iid: {iid}")
        seen.add(iid)
        source = row.get("resolved_source_video")
        expected_sha = row.get("source_video_sha256")
        if not isinstance(source, str) or not Path(source).is_absolute():
            raise MaterializationError(
                f"manifest iid {iid} lacks an absolute resolved_source_video"
            )
        if not isinstance(expected_sha, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha
        ):
            raise MaterializationError(
                f"manifest iid {iid} has an invalid source_video_sha256"
            )
        instruction = row.get("edit_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise MaterializationError(
                f"manifest iid {iid} lacks a non-empty edit_instruction"
            )
        rows.append(row)
    if not rows:
        raise MaterializationError(f"generation manifest is empty: {path}")
    return rows


def atomic_copy_no_clobber(source: Path, destination: Path, expected_sha: str) -> str:
    """Copy and verify through a same-directory temporary file, then publish once."""
    if destination.exists() or destination.is_symlink():
        require_regular_file(destination, "existing source copy")
        actual_sha = sha256_file(destination)
        if actual_sha != expected_sha:
            raise MaterializationError(
                f"refusing to overwrite mismatched source copy: {destination}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        return "already_present"

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".source.mp4.copying-", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(file_descriptor, "wb") as target, source.open("rb") as origin:
            while True:
                chunk = origin.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
            target.flush()
            os.fsync(target.fileno())
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise MaterializationError(
                f"source changed or failed SHA verification while copying {source}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        os.chmod(temporary_path, 0o644)
        try:
            os.link(temporary_path, destination)
            status = "copied"
        except FileExistsError:
            require_regular_file(destination, "raced source copy")
            raced_sha = sha256_file(destination)
            if raced_sha != expected_sha:
                raise MaterializationError(
                    f"raced source copy has a mismatched SHA: {destination}"
                )
            status = "already_present"
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return status


def atomic_json_no_clobber(path: Path, payload: dict[str, Any]) -> None:
    expected = canonical_json_bytes(payload)
    atomic_bytes_no_clobber(path, expected, "source-copy sidecar")


def atomic_bytes_no_clobber(path: Path, expected: bytes, label: str) -> None:
    if path.exists() or path.is_symlink():
        require_regular_file(path, f"existing {label}")
        if path.read_bytes() != expected:
            raise MaterializationError(
                f"refusing to overwrite mismatched {label}: {path}"
            )
        return
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.writing-", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            require_regular_file(path, f"raced {label}")
            if path.read_bytes() != expected:
                raise MaterializationError(
                    f"raced {label} differs: {path}"
                )
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def atomic_replace_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        require_regular_file(path, "replaceable ancillary output")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.writing-", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def validate_plain_filename(value: Any, label: str, iid: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise MaterializationError(
            f"result for {iid} has an unsafe {label} filename: {value!r}"
        )
    return value


@contextmanager
def exclusive_output_lock(output_root: Path) -> Iterator[None]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise MaterializationError(
            f"output root must be an existing non-symlink directory: {output_root}"
        )
    lock_path = output_root / ".source_materialization.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o644)
    except OSError as exc:
        raise MaterializationError(f"cannot open materialization lock: {lock_path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise MaterializationError(
                f"materialization lock is not a regular file: {lock_path}"
            )
        os.fchmod(descriptor, 0o644)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def validate_committed_sample(
    row: dict[str, Any],
    sample_index: int,
    samples_root: Path,
    manifest_sha: str,
) -> dict[str, Any] | None:
    iid = row["iid"]
    sample_dir = samples_root / iid
    result_path = sample_dir / "result.json"
    if not sample_dir.exists() and not sample_dir.is_symlink():
        return None
    if sample_dir.is_symlink() or not sample_dir.is_dir():
        raise MaterializationError(
            f"existing sample path must be a non-symlink directory: {sample_dir}"
        )
    require_regular_file(result_path, f"result for {iid}")

    result = load_json(result_path, f"result for {iid}")
    result_digest = result.get("result_digest")
    if not isinstance(result_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", result_digest
    ):
        raise MaterializationError(f"invalid result_digest for iid {iid}")
    bound_result = dict(result)
    del bound_result["result_digest"]
    actual_result_digest = object_digest(bound_result)
    if actual_result_digest != result_digest:
        raise MaterializationError(
            f"result_digest mismatch for iid {iid}: expected {result_digest}, "
            f"got {actual_result_digest}"
        )

    expected_bindings = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "iid": iid,
        "group_id": row.get("group_id"),
        "sample_index": sample_index,
        "manifest_sha256": manifest_sha,
        "manifest_row_digest": object_digest(row),
        "action_change_substantive": row.get("action_change_substantive"),
    }
    for field, expected in expected_bindings.items():
        if result.get(field) != expected:
            raise MaterializationError(
                f"result/manifest binding mismatch for iid {iid}, field {field}: "
                f"expected {expected!r}, got {result.get(field)!r}"
            )

    instruction = row["edit_instruction"]
    instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    prompt = result.get("prompt")
    if not isinstance(prompt, dict) or prompt != {
        "field": "edit_instruction",
        "sha256": instruction_sha,
        "text": instruction,
    }:
        raise MaterializationError(f"result prompt differs from manifest for iid {iid}")

    inputs = result.get("inputs")
    outputs = result.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise MaterializationError(f"result lacks inputs/outputs objects: {result_path}")

    expected_source_sha = row["source_video_sha256"]
    source_path = Path(row["resolved_source_video"])
    if inputs.get("source_video_sha256") != expected_source_sha:
        raise MaterializationError(f"result/manifest source SHA mismatch for iid {iid}")
    if inputs.get("source_video_resolved_path") != str(source_path):
        raise MaterializationError(f"result/manifest source path mismatch for iid {iid}")
    require_regular_file(source_path, f"source video for {iid}")
    actual_source_sha = sha256_file(source_path)
    if actual_source_sha != expected_source_sha:
        raise MaterializationError(
            f"source SHA mismatch for iid {iid}: expected {expected_source_sha}, "
            f"got {actual_source_sha}"
        )

    preview_name = validate_plain_filename(
        outputs.get("preview_mp4"), "preview_mp4", iid
    )
    preview_path = sample_dir / preview_name
    require_regular_file(preview_path, f"generated preview for {iid}")
    expected_preview_sha = outputs.get("preview_mp4_sha256")
    if not isinstance(expected_preview_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_preview_sha
    ):
        raise MaterializationError(f"invalid preview SHA in result for iid {iid}")
    actual_preview_sha = sha256_file(preview_path)
    if actual_preview_sha != expected_preview_sha:
        raise MaterializationError(
            f"generated preview SHA mismatch for iid {iid}: "
            f"expected {expected_preview_sha}, got {actual_preview_sha}"
        )

    return {
        "expected_preview_sha": expected_preview_sha,
        "expected_source_sha": expected_source_sha,
        "iid": iid,
        "preview_path": preview_path,
        "result": result,
        "result_path": result_path,
        "row": row,
        "sample_dir": sample_dir,
        "source_path": source_path,
    }


def _materialize_locked(
    manifest_path: Path,
    output_root: Path,
    expected_manifest_sha256: str | None,
    require_all: bool,
) -> dict[str, Any]:
    manifest_path = manifest_path.absolute()
    output_root = output_root.absolute()
    require_regular_file(manifest_path, "generation manifest")
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        raise MaterializationError(
            f"manifest SHA mismatch: expected {expected_manifest_sha256}, got {manifest_sha}"
        )
    rows = load_manifest(manifest_path, manifest_raw)
    samples_root = output_root / "samples"
    if samples_root.is_symlink() or not samples_root.is_dir():
        raise MaterializationError(
            f"samples root must be an existing non-symlink directory: {samples_root}"
        )

    prepared: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample_index, row in enumerate(rows):
        validated = validate_committed_sample(
            row, sample_index, samples_root, manifest_sha
        )
        if validated is None:
            missing.append(row["iid"])
        else:
            prepared.append(validated)

    if require_all and missing:
        raise MaterializationError(
            f"only validated {len(prepared)}/{len(rows)} samples; missing: "
            + ", ".join(missing)
        )

    pairs: list[dict[str, Any]] = []
    copied = 0
    already_present = 0
    for sample in prepared:
        iid = sample["iid"]
        sample_dir = sample["sample_dir"]
        source_path = sample["source_path"]
        expected_source_sha = sample["expected_source_sha"]
        preview_path = sample["preview_path"]
        expected_preview_sha = sample["expected_preview_sha"]
        result = sample["result"]
        result_path = sample["result_path"]
        row = sample["row"]
        destination = sample_dir / "source.mp4"
        copy_status = atomic_copy_no_clobber(
            source_path, destination, expected_source_sha
        )
        if copy_status == "copied":
            copied += 1
        else:
            already_present += 1

        source_bytes = destination.stat().st_size
        sidecar = {
            "destination_bytes": source_bytes,
            "destination_filename": destination.name,
            "destination_sha256": expected_source_sha,
            "iid": iid,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "original_source_path": str(source_path),
            "result_digest": result.get("result_digest"),
            "schema_version": SCHEMA_VERSION,
        }
        atomic_json_no_clobber(sample_dir / "source_copy.json", sidecar)
        instruction = row["edit_instruction"]
        instruction_path = sample_dir / "edit_instruction.txt"
        instruction_payload = instruction.encode("utf-8") + b"\n"
        atomic_bytes_no_clobber(
            instruction_path,
            instruction_payload,
            "edit-instruction file",
        )

        pairs.append(
            {
                "edit_instruction": instruction,
                "edit_instruction_file": str(instruction_path),
                "edit_instruction_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
                "generated_video": str(preview_path),
                "generated_video_sha256": expected_preview_sha,
                "iid": iid,
                "result_json": str(result_path),
                "schema_version": PAIR_SCHEMA_VERSION,
                "source_video": str(destination),
                "source_video_bytes": source_bytes,
                "source_video_sha256": expected_source_sha,
            }
        )

    index_path = output_root / "source_target_pairs.jsonl"
    index_payload = b"".join(canonical_json_bytes(pair) for pair in pairs)
    atomic_replace_bytes(index_path, index_payload)

    summary = {
        "already_present": already_present,
        "copied": copied,
        "index_path": str(index_path),
        "instruction_file_count": len(pairs),
        "manifest_count": len(rows),
        "manifest_sha256": manifest_sha,
        "materialized_count": len(pairs),
        "missing_iids": missing,
        "output_root": str(output_root),
        "schema_version": SCHEMA_VERSION,
    }
    atomic_replace_bytes(
        output_root / "source_materialization_status.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )
    return summary


def materialize(
    manifest_path: Path,
    output_root: Path,
    expected_manifest_sha256: str | None,
    require_all: bool,
) -> dict[str, Any]:
    absolute_output_root = output_root.absolute()
    with exclusive_output_lock(absolute_output_root):
        return _materialize_locked(
            manifest_path,
            absolute_output_root,
            expected_manifest_sha256,
            require_all,
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail before writing unless every manifest row is committed",
    )
    args = parser.parse_args(argv)
    if args.expected_manifest_sha256 and not re.fullmatch(
        r"[0-9a-f]{64}", args.expected_manifest_sha256
    ):
        parser.error("--expected-manifest-sha256 must be 64 lowercase hex digits")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        summary = materialize(
            args.manifest,
            args.output_root,
            args.expected_manifest_sha256,
            args.require_all,
        )
    except MaterializationError as exc:
        print(f"materialize_wan22_sources: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
