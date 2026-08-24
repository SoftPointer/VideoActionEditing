#!/usr/bin/env python3
"""Render or verify a released shard of the prospective factorial branch bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import prospective_factorial_branch_manifest_v1 as branch_manifest  # noqa: E402


ENTRY_RECEIPT_SCHEMA = "bernini-prospective-factorial-branch-entry-receipt-v1"
RELEASE_RECEIPT_SCHEMA = "bernini-prospective-factorial-branch-release-receipt-v1"
INFERENCE_RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-inference-receipt-v1"
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CELL = re.compile(r"(?P<source>[0-9a-f]{16}):(?P<seed>[0-9]+)\Z")


class FactorialBranchRunError(RuntimeError):
    """Raised before incomplete or unbound branch output is accepted."""


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise FactorialBranchRunError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise FactorialBranchRunError(f"{label} must be an absolute plain directory")
    return path.resolve(strict=True)


def _executable(value: str | Path, *, label: str) -> Path:
    path = _plain_file(value, label=label)
    if not stat.S_ISREG(path.stat().st_mode) or not os.access(path, os.X_OK):
        raise FactorialBranchRunError(f"{label} must be executable")
    return path


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorialBranchRunError(f"cannot read {label}") from error
    if type(value) is not dict:
        raise FactorialBranchRunError(f"{label} must contain one object")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FactorialBranchRunError(f"refusing to overwrite {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(branch_manifest.canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_manifest(path_value: str | Path, expected_sha256: str) -> tuple[dict[str, Any], Path]:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise FactorialBranchRunError("expected manifest SHA-256 differs")
    path = _plain_file(path_value, label="branch manifest")
    if branch_manifest.file_sha256(path) != expected_sha256:
        raise FactorialBranchRunError("branch manifest raw SHA-256 differs")
    value = _read_object(path, label="branch manifest")
    branch_manifest.validate_manifest(value)
    return value, path


def _cells(values: Sequence[str]) -> list[tuple[str, int]]:
    if not values:
        raise FactorialBranchRunError("at least one released cell is required")
    parsed: list[tuple[str, int]] = []
    for value in values:
        match = _CELL.fullmatch(value)
        if match is None:
            raise FactorialBranchRunError(f"invalid released cell: {value}")
        cell = (match.group("source"), int(match.group("seed")))
        if cell in parsed:
            raise FactorialBranchRunError("duplicate released cell")
        parsed.append(cell)
    if parsed != sorted(parsed):
        raise FactorialBranchRunError("released cells must be sorted")
    return parsed


def _released_entries(
    manifest: Mapping[str, Any], cells: Sequence[tuple[str, int]], split: str
) -> list[dict[str, Any]]:
    if split != "fit":
        raise FactorialBranchRunError(
            "v1 runtime releases fit only; calibration/confirmation remain sealed"
        )
    cell_set = set(cells)
    rows = [
        dict(row)
        for row in manifest["entries"]
        if (row["source_id"], row["seed"]) in cell_set
    ]
    observed = {(row["source_id"], row["seed"]) for row in rows}
    if observed != cell_set or any(row["analysis_split"] != split for row in rows):
        raise FactorialBranchRunError("released cell does not belong to the fit split")
    for cell in cells:
        cell_rows = [
            row for row in rows if (row["source_id"], row["seed"]) == cell
        ]
        if {row["branch"] for row in cell_rows} != set(branch_manifest.BRANCHES):
            raise FactorialBranchRunError("released factorial cell is incomplete")
    return rows


def _entry_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["entry_id"]): dict(row) for row in rows}


def _verify_source(row: Mapping[str, Any]) -> Path:
    path = _plain_file(row["source_video"], label=f"{row['entry_id']} source")
    if branch_manifest.file_sha256(path) != row["source_video_sha256"]:
        raise FactorialBranchRunError(f"{row['entry_id']} source digest differs")
    return path


def _verify_inference_receipt(
    receipt_path: Path,
    row: Mapping[str, Any],
    output_path: Path,
    *,
    method_source_revision: str,
    method_source_archive_sha256: str,
) -> dict[str, Any]:
    receipt = _read_object(
        _plain_file(receipt_path, label="inference receipt"), label="inference receipt"
    )
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest", None)
    if type(digest) is not str or branch_manifest.object_sha256(unsigned) != digest:
        raise FactorialBranchRunError("inference receipt digest differs")
    input_row = receipt.get("input")
    preprocessing = receipt.get("preprocessing")
    sampling = receipt.get("sampling")
    output = receipt.get("output")
    adapter = receipt.get("adapter")
    if (
        receipt.get("schema_version") != INFERENCE_RECEIPT_SCHEMA
        or receipt.get("method_source_revision") != method_source_revision
        or receipt.get("method_source_archive_sha256")
        != method_source_archive_sha256
        or receipt.get("bernini_commit") != BERNINI_COMMIT
        or receipt.get("veomni_commit") != VEOMNI_COMMIT
        or receipt.get("checkpoint_tree_sha256") != CHECKPOINT_TREE_SHA256
        or not isinstance(input_row, Mapping)
        or input_row.get("source_video_path") != str(Path(row["source_video"]).resolve())
        or input_row.get("source_video_sha256") != row["source_video_sha256"]
        or input_row.get("instruction_utf8_sha256")
        != row["instruction_utf8_sha256"]
        or input_row.get("target_accessed_by_inference") is not False
        or not isinstance(preprocessing, Mapping)
        or preprocessing.get("frame_count") != 81
        or float(preprocessing.get("fps", -1)) != 25.0
        or not isinstance(sampling, Mapping)
        or sampling.get("seed") != row["seed"]
        or sampling.get("num_inference_steps") != 40
        or sampling.get("ulysses_size") != 4
        or not isinstance(output, Mapping)
        or output.get("path") != str(output_path)
        or output.get("frame_count") != 81
        or float(output.get("fps", -1)) != 25.0
        or adapter
        != {
            "enabled": False,
            "mode": "frozen_base_no_adapter",
            "strictly_reloaded": False,
            "safe_merged_for_inference": False,
            "tensor_count": 0,
        }
    ):
        raise FactorialBranchRunError("frozen Bernini inference contract differs")
    media = _plain_file(output_path, label="generated branch output")
    if branch_manifest.file_sha256(media) != output.get("sha256"):
        raise FactorialBranchRunError("generated output digest differs")
    return receipt


def _run_entry(
    row: Mapping[str, Any],
    *,
    output_root: Path,
    method_root: Path,
    python_bin: Path,
    bernini_root: Path,
    veomni_root: Path,
    checkpoint: Path,
    master_port: int,
    method_source_revision: str,
    method_source_archive_sha256: str,
    manifest_sha256: str,
    manifest_digest: str,
) -> dict[str, Any]:
    source = _verify_source(row)
    entry_root = output_root / "entries" / row["entry_id"]
    entry_root.mkdir(mode=0o700)
    output = entry_root / "output.mp4"
    inference_receipt = output.with_name("output.mp4.receipt.json")
    if row["executor"] == "exact_source_copy":
        with source.open("rb") as reader, output.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        output_sha = branch_manifest.file_sha256(output)
        if output_sha != row["source_video_sha256"]:
            raise FactorialBranchRunError("noop copy differs from source")
        native_digest = None
    else:
        runner = _plain_file(method_root / "infer_lora.py", label="Bernini runner")
        command = [
            str(python_bin),
            "-B",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=4",
            "--master_addr=127.0.0.1",
            f"--master_port={master_port}",
            str(runner),
            "--bernini-root",
            str(bernini_root),
            "--veomni-root",
            str(veomni_root),
            "--checkpoint",
            str(checkpoint),
            "--base-only",
            "--source-video",
            str(source),
            "--instruction",
            row["instruction"],
            "--output",
            str(output),
            "--num-inference-steps",
            "40",
            "--seed",
            str(row["seed"]),
            "--expected-bernini-commit",
            BERNINI_COMMIT,
            "--expected-veomni-commit",
            VEOMNI_COMMIT,
            "--expected-checkpoint-tree-sha256",
            CHECKPOINT_TREE_SHA256,
            "--method-source-revision",
            method_source_revision,
            "--method-source-archive-sha256",
            method_source_archive_sha256,
        ]
        subprocess.run(command, check=True)
        native = _verify_inference_receipt(
            inference_receipt,
            row,
            output,
            method_source_revision=method_source_revision,
            method_source_archive_sha256=method_source_archive_sha256,
        )
        output_sha = native["output"]["sha256"]
        native_digest = native["receipt_digest"]
    unsigned = {
        "schema_version": ENTRY_RECEIPT_SCHEMA,
        "status": "branch_rendered_not_training_target_not_method_success",
        "manifest_sha256": manifest_sha256,
        "manifest_digest": manifest_digest,
        "entry_id": row["entry_id"],
        "source_id": row["source_id"],
        "analysis_split": row["analysis_split"],
        "seed": row["seed"],
        "branch": row["branch"],
        "executor": row["executor"],
        "source_video_sha256": row["source_video_sha256"],
        "instruction_utf8_sha256": row["instruction_utf8_sha256"],
        "output_path": str(output),
        "output_sha256": output_sha,
        "native_inference_receipt_digest": native_digest,
        "same_source_seed_cell_required": True,
        "training_target_authorized": False,
        "optimizer_step_authorized": False,
        "method_success_claimed": False,
    }
    receipt = {**unsigned, "receipt_digest": branch_manifest.object_sha256(unsigned)}
    _write_create_only(entry_root / "entry.receipt.json", receipt)
    return receipt


def _verify_entry(
    row: Mapping[str, Any], output_root: Path, manifest_sha256: str
) -> dict[str, Any]:
    root = _plain_directory(
        output_root / "entries" / row["entry_id"], label="entry output root"
    )
    receipt = _read_object(
        _plain_file(root / "entry.receipt.json", label="entry receipt"),
        label="entry receipt",
    )
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest", None)
    if (
        type(digest) is not str
        or branch_manifest.object_sha256(unsigned) != digest
        or receipt.get("schema_version") != ENTRY_RECEIPT_SCHEMA
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("entry_id") != row["entry_id"]
        or receipt.get("source_video_sha256") != row["source_video_sha256"]
        or receipt.get("instruction_utf8_sha256")
        != row["instruction_utf8_sha256"]
        or receipt.get("training_target_authorized") is not False
        or receipt.get("optimizer_step_authorized") is not False
        or receipt.get("method_success_claimed") is not False
    ):
        raise FactorialBranchRunError("entry receipt differs")
    media = _plain_file(root / "output.mp4", label="entry output")
    if branch_manifest.file_sha256(media) != receipt.get("output_sha256"):
        raise FactorialBranchRunError("entry output digest differs")
    return receipt


def command_run_lane(args: argparse.Namespace) -> int:
    manifest, _ = _load_manifest(args.manifest, args.expected_manifest_sha256)
    cells = _cells(args.cell)
    rows = _released_entries(manifest, cells, args.split)
    if args.lane_count != 2 or args.lane_index not in (0, 1):
        raise FactorialBranchRunError("runtime requires exactly two isolated SP4 lanes")
    lane_cells = set(cells[args.lane_index :: args.lane_count])
    lane_rows = [
        row for row in rows if (row["source_id"], row["seed"]) in lane_cells
    ]
    output_root = _plain_directory(args.output_root, label="release output root")
    method_root = _plain_directory(args.method_root, label="method root")
    python_bin = _executable(args.python_bin, label="Python")
    if _SHA1.fullmatch(args.method_source_revision) is None or _SHA256.fullmatch(
        args.method_source_archive_sha256
    ) is None:
        raise FactorialBranchRunError("method source identity differs")
    results = []
    for row in lane_rows:
        results.append(
            _run_entry(
                row,
                output_root=output_root,
                method_root=method_root,
                python_bin=python_bin,
                bernini_root=_plain_directory(args.bernini_root, label="Bernini root"),
                veomni_root=_plain_directory(args.veomni_root, label="VeOmni root"),
                checkpoint=_plain_directory(args.checkpoint, label="checkpoint"),
                master_port=args.master_port,
                method_source_revision=args.method_source_revision,
                method_source_archive_sha256=args.method_source_archive_sha256,
                manifest_sha256=args.expected_manifest_sha256,
                manifest_digest=manifest["manifest_digest"],
            )
        )
    lane_receipt = {
        "schema_version": RELEASE_RECEIPT_SCHEMA,
        "status": "lane_complete_not_training_target_not_method_success",
        "manifest_sha256": args.expected_manifest_sha256,
        "split": args.split,
        "lane_index": args.lane_index,
        "lane_count": args.lane_count,
        "released_cells": [f"{source}:{seed}" for source, seed in cells],
        "entry_receipt_digests": [row["receipt_digest"] for row in results],
        "training_target_authorized": False,
        "optimizer_step_authorized": False,
        "method_success_claimed": False,
    }
    lane_receipt["receipt_digest"] = branch_manifest.object_sha256(lane_receipt)
    _write_create_only(
        output_root / f"lane-{args.lane_index}.receipt.json", lane_receipt
    )
    print(json.dumps({"lane": args.lane_index, "entries": len(results)}, sort_keys=True))
    return 0


def command_verify_release(args: argparse.Namespace) -> int:
    manifest, _ = _load_manifest(args.manifest, args.expected_manifest_sha256)
    cells = _cells(args.cell)
    rows = _released_entries(manifest, cells, args.split)
    output_root = _plain_directory(args.output_root, label="release output root")
    receipts = [_verify_entry(row, output_root, args.expected_manifest_sha256) for row in rows]
    branches = {branch: sum(row["branch"] == branch for row in receipts) for branch in branch_manifest.BRANCHES}
    if set(branches.values()) != {len(cells)}:
        raise FactorialBranchRunError("released branch balance differs")
    unsigned = {
        "schema_version": RELEASE_RECEIPT_SCHEMA,
        "status": "released_fit_cells_complete_review_pending_not_training_target",
        "manifest_sha256": args.expected_manifest_sha256,
        "split": args.split,
        "released_cells": [f"{source}:{seed}" for source, seed in cells],
        "entry_count": len(receipts),
        "branch_counts": branches,
        "entry_receipt_digests": sorted(row["receipt_digest"] for row in receipts),
        "decoded_semantic_review_required": True,
        "training_target_authorized": False,
        "optimizer_step_authorized": False,
        "method_success_claimed": False,
    }
    result = {**unsigned, "receipt_digest": branch_manifest.object_sha256(unsigned)}
    _write_create_only(Path(args.output_receipt), result)
    print(json.dumps({"entries": len(receipts), "receipt": args.output_receipt}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--split", default="fit")
    parser.add_argument("--cell", action="append", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-lane")
    run.add_argument("--lane-index", type=int, required=True)
    run.add_argument("--lane-count", type=int, default=2)
    run.add_argument("--master-port", type=int, required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--method-root", required=True)
    run.add_argument("--python-bin", required=True)
    run.add_argument("--bernini-root", required=True)
    run.add_argument("--veomni-root", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--method-source-revision", required=True)
    run.add_argument("--method-source-archive-sha256", required=True)
    verify = sub.add_parser("verify-release")
    verify.add_argument("--output-root", required=True)
    verify.add_argument("--output-receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-lane":
        return command_run_lane(args)
    return command_verify_release(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FactorialBranchRunError as error:
        print(f"[factorial-branch-run] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
