"""Create immutable deterministic epochs from a high-recall Goku manifest.

Rows are validated but never re-serialized: each epoch's ``selected.jsonl`` is
the byte-for-byte concatenation of a contiguous range of parent lines.  The
parent SHA, half-open row interval, row count, and output SHA are bound into
both ``summary.json`` and ``done.json`` for every epoch.  IID and group IDs are
required to be globally unique before any output is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "motive-goku-atomic-candidate-epoch-slicer-v1"
EPOCH_SUMMARY_SCHEMA = "motive-goku-atomic-candidate-epoch-summary-v1"
EPOCH_DONE_SCHEMA = "motive-goku-atomic-candidate-epoch-done-v1"
ROOT_SUMMARY_SCHEMA = "motive-goku-atomic-candidate-epochs-summary-v1"
ROOT_DONE_SCHEMA = "motive-goku-atomic-candidate-epochs-done-v1"
SELECTED_NAME = "selected.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
EPOCH_ENTRIES = frozenset({SELECTED_NAME, SUMMARY_NAME, DONE_NAME})


class EpochSlicerError(RuntimeError):
    """Fail-closed epoch slicing error."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical_json(value).encode("utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha_field(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EpochSlicerError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_row(raw: bytes, *, row_number: int) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise EpochSlicerError(
            f"invalid parent JSON at row {row_number}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise EpochSlicerError(f"parent row {row_number} is not a JSON object")
    return value


def _read_bound_parent(
    parent_selected: str | Path,
    *,
    expected_parent_sha256: str,
) -> tuple[Path, bytes, list[bytes], list[dict[str, Any]], str]:
    path = Path(parent_selected).expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise EpochSlicerError(f"parent selected must be a plain file: {path}")
    raw = path.read_bytes()
    if not raw:
        raise EpochSlicerError("parent selected is empty")
    actual_sha = _sha256(raw)
    expected_sha = _sha_field(
        expected_parent_sha256, name="expected parent selected SHA"
    )
    if actual_sha != expected_sha:
        raise EpochSlicerError(
            f"parent selected SHA differs: expected={expected_sha} actual={actual_sha}"
        )
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise EpochSlicerError(
            "every parent row must end with LF so epoch rows can remain byte-exact"
        )
    rows: list[dict[str, Any]] = []
    seen_iids: dict[str, int] = {}
    seen_groups: dict[str, int] = {}
    for row_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EpochSlicerError(f"blank parent row: {row_number}")
        row = _parse_row(line, row_number=row_number)
        iid = row.get("iid")
        group = row.get("group_id")
        if not isinstance(iid, str) or not iid.strip():
            raise EpochSlicerError(f"parent row {row_number} has invalid iid")
        if not isinstance(group, str) or not group.strip():
            raise EpochSlicerError(f"parent row {row_number} has invalid group_id")
        if iid in seen_iids:
            raise EpochSlicerError(
                f"duplicate IID across parent rows {seen_iids[iid]} and {row_number}: {iid}"
            )
        if group in seen_groups:
            raise EpochSlicerError(
                "duplicate group_id across parent rows "
                f"{seen_groups[group]} and {row_number}: {group}"
            )
        seen_iids[iid] = row_number
        seen_groups[group] = row_number
        rows.append(row)
    return path, raw, lines, rows, actual_sha


def _write_bytes(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory(output_dir: Path, writer: Any) -> None:
    target = output_dir.expanduser().resolve(strict=False)
    if os.path.lexists(target):
        raise FileExistsError(f"create-only output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        writer(stage, target)
        if os.path.lexists(target):
            raise FileExistsError(f"output appeared during publication: {target}")
        os.replace(stage, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _epoch_binding(
    *,
    parent_path: Path,
    parent_sha: str,
    parent_rows: int,
    epoch_index: int,
    epoch_name: str,
    start: int,
    end: int,
    output_sha: str,
) -> dict[str, Any]:
    """Build the exact binding copied into both epoch summary and done."""

    binding = {
        "parent_selected_path": str(parent_path),
        "parent_selected_sha256": parent_sha,
        "parent_rows": parent_rows,
        "epoch_index": epoch_index,
        "epoch_name": epoch_name,
        "interval_semantics": "zero_based_half_open",
        "start": start,
        "end": end,
        "start_row_one_based": start + 1,
        "end_row_one_based_inclusive": end,
        "rows": end - start,
        "output_sha256": output_sha,
        "row_bytes_preserved": True,
    }
    binding["binding_digest"] = _digest(binding)
    return binding


def slice_epochs(
    *,
    parent_selected: str | Path,
    expected_parent_sha256: str,
    output_dir: str | Path,
    epoch_size: int = 2000,
    min_epochs: int = 8,
) -> dict[str, Any]:
    """Atomically publish contiguous byte-exact epochs and audit receipts."""

    if epoch_size <= 0:
        raise EpochSlicerError("epoch_size must be positive")
    if min_epochs <= 0:
        raise EpochSlicerError("min_epochs must be positive")
    parent_path, parent_raw, lines, rows, parent_sha = _read_bound_parent(
        parent_selected,
        expected_parent_sha256=expected_parent_sha256,
    )
    epoch_count = math.ceil(len(rows) / epoch_size)
    if epoch_count < min_epochs:
        raise EpochSlicerError(
            f"parent yields {epoch_count} epochs, fewer than required {min_epochs}"
        )
    implementation = {
        "path": str(Path(__file__).resolve(strict=True)),
        "sha256": _file_sha256(Path(__file__).resolve(strict=True)),
    }
    target = Path(output_dir).expanduser().resolve(strict=False)

    def writer(stage: Path, final_output: Path) -> None:
        epoch_records: list[dict[str, Any]] = []
        concatenated: list[bytes] = []
        width = max(4, len(str(epoch_count)))
        for zero_index, start in enumerate(range(0, len(rows), epoch_size)):
            end = min(start + epoch_size, len(rows))
            epoch_index = zero_index + 1
            epoch_name = f"epoch_{epoch_index:0{width}d}"
            epoch_dir = stage / epoch_name
            epoch_dir.mkdir()
            selected_raw = b"".join(lines[start:end])
            concatenated.append(selected_raw)
            output_sha = _sha256(selected_raw)
            binding = _epoch_binding(
                parent_path=parent_path,
                parent_sha=parent_sha,
                parent_rows=len(rows),
                epoch_index=epoch_index,
                epoch_name=epoch_name,
                start=start,
                end=end,
                output_sha=output_sha,
            )
            _write_bytes(epoch_dir / SELECTED_NAME, selected_raw)
            summary = {
                "schema_version": EPOCH_SUMMARY_SCHEMA,
                "status": "complete",
                "binding": binding,
                # Duplicate the required scheduling coordinates at top level
                # so shell supervisors need not interpret nested provenance.
                "parent_sha256": parent_sha,
                "start": start,
                "end": end,
                "rows": end - start,
                "output_sha256": output_sha,
                "first_iid": rows[start]["iid"],
                "last_iid": rows[end - 1]["iid"],
                "iid_digest": _digest([row["iid"] for row in rows[start:end]]),
                "group_id_digest": _digest(
                    [row["group_id"] for row in rows[start:end]]
                ),
                "implementation": implementation,
            }
            summary_raw = _json_bytes(summary)
            _write_bytes(epoch_dir / SUMMARY_NAME, summary_raw)
            artifacts = {
                SELECTED_NAME: output_sha,
                SUMMARY_NAME: _sha256(summary_raw),
            }
            done = {
                "schema_version": EPOCH_DONE_SCHEMA,
                "status": "complete",
                "binding": binding,
                "parent_sha256": parent_sha,
                "start": start,
                "end": end,
                "rows": end - start,
                "output_sha256": output_sha,
                "implementation_sha256": implementation["sha256"],
                "artifacts": artifacts,
                "artifact_digest": _digest(artifacts),
            }
            done_raw = _json_bytes(done)
            _write_bytes(epoch_dir / DONE_NAME, done_raw)
            if {path.name for path in epoch_dir.iterdir()} != EPOCH_ENTRIES:
                raise EpochSlicerError(f"epoch closure differs: {epoch_name}")
            epoch_records.append(
                {
                    **binding,
                    "selected_path": f"{epoch_name}/{SELECTED_NAME}",
                    "summary_sha256": artifacts[SUMMARY_NAME],
                    "done_sha256": _sha256(done_raw),
                }
            )

        if b"".join(concatenated) != parent_raw:
            raise EpochSlicerError("epoch byte concatenation differs from parent")
        root_summary = {
            "schema_version": ROOT_SUMMARY_SCHEMA,
            "status": "complete",
            "parent_selected_path": str(parent_path),
            "parent_selected_sha256": parent_sha,
            "parent_rows": len(rows),
            "parent_bytes": len(parent_raw),
            "epoch_size": epoch_size,
            "min_epochs": min_epochs,
            "epoch_count": epoch_count,
            "iid_global_unique": True,
            "group_id_global_unique": True,
            "row_order": "parent_manifest_contiguous",
            "row_bytes_preserved": True,
            "concatenated_output_sha256": _sha256(b"".join(concatenated)),
            "epochs": epoch_records,
            "implementation": implementation,
        }
        root_summary_raw = _json_bytes(root_summary)
        _write_bytes(stage / SUMMARY_NAME, root_summary_raw)
        epoch_done_sha = {
            record["epoch_name"]: record["done_sha256"] for record in epoch_records
        }
        root_artifacts = {
            SUMMARY_NAME: _sha256(root_summary_raw),
            "epoch_done_sha256": epoch_done_sha,
        }
        root_done = {
            "schema_version": ROOT_DONE_SCHEMA,
            "status": "complete",
            "parent_selected_sha256": parent_sha,
            "parent_rows": len(rows),
            "epoch_size": epoch_size,
            "epoch_count": epoch_count,
            "implementation_sha256": implementation["sha256"],
            "artifacts": root_artifacts,
            "artifact_digest": _digest(root_artifacts),
        }
        _write_bytes(stage / DONE_NAME, _json_bytes(root_done))
        expected_root = {SUMMARY_NAME, DONE_NAME} | {
            record["epoch_name"] for record in epoch_records
        }
        if {path.name for path in stage.iterdir()} != expected_root:
            raise EpochSlicerError("root epoch closure differs")

    _publish_directory(target, writer)
    return json.loads((target / SUMMARY_NAME).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slice a high-recall Goku selected manifest into epochs."
    )
    parser.add_argument("--parent-selected", required=True, type=Path)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epoch-size", type=int, default=2000)
    parser.add_argument("--min-epochs", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = slice_epochs(
        parent_selected=args.parent_selected,
        expected_parent_sha256=args.expected_parent_sha256,
        output_dir=args.output_dir,
        epoch_size=args.epoch_size,
        min_epochs=args.min_epochs,
    )
    print(_canonical_json(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
