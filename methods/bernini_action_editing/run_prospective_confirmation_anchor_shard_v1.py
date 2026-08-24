#!/usr/bin/env python3
"""Render forward/noop shards from the sealed prospective confirmation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

BASE_NAME = "run_prospective_factorial_branch_shard_v1.py"
BASE_SHA256 = "dc76f9906589a720404af183332e8fe453b2785df1e740d0ec0dbfb6b78b3662"
BASE_PATH = METHOD_ROOT / BASE_NAME
if (
    not BASE_PATH.is_file() or BASE_PATH.is_symlink()
    or hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256
):
    raise RuntimeError("pinned factorial branch runtime dependency differs")

import run_prospective_factorial_branch_shard_v1 as base  # noqa: E402


SCHEMA_VERSION = "bernini-prospective-confirmation-anchor-shard-v1"
RELEASE_BRANCHES = ("noop", "forward")
AUTHORITY = {
    "confirmation_generation_only": True,
    "decoded_review_required": True,
    "representation_reselection_authorized": False,
    "threshold_recalibration_authorized": False,
    "training_target_authorized": False,
    "optimizer_step_authorized": False,
    "method_success_claimed": False,
}


class ConfirmationAnchorShardError(RuntimeError):
    """Raised before an incomplete confirmation shard can be accepted."""


def released_rows(
    manifest: Mapping[str, Any], cells: Sequence[tuple[str, int]]
) -> list[dict[str, Any]]:
    cell_set = set(cells)
    rows = [
        dict(row) for row in manifest["entries"]
        if (row["source_id"], row["seed"]) in cell_set
        and row["branch"] in RELEASE_BRANCHES
    ]
    by_cell = {
        cell: {row["branch"] for row in rows if (row["source_id"], row["seed"]) == cell}
        for cell in cells
    }
    if (
        any(branches != set(RELEASE_BRANCHES) for branches in by_cell.values())
        or any(row["analysis_split"] != "confirmation" for row in rows)
        or len(rows) != 2 * len(cells)
    ):
        raise ConfirmationAnchorShardError("confirmation forward/noop closure differs")
    order = {branch: index for index, branch in enumerate(RELEASE_BRANCHES)}
    return sorted(rows, key=lambda row: (row["source_id"], row["seed"], order[row["branch"]]))


def receipt(
    *, manifest_sha256: str, manifest_digest: str,
    cells: Sequence[tuple[str, int]], entry_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "confirmation_forward_noop_complete_decoded_review_pending",
        "manifest_sha256": manifest_sha256,
        "manifest_digest": manifest_digest,
        "analysis_split": "confirmation",
        "released_cells": [f"{source}:{seed}" for source, seed in cells],
        "branches": list(RELEASE_BRANCHES),
        "entry_count": len(entry_receipts),
        "entry_receipt_digests": [row["receipt_digest"] for row in entry_receipts],
        "authority": dict(AUTHORITY),
    }
    return {**unsigned, "receipt_digest": base.branch_manifest.object_sha256(unsigned)}


def command_run(args: argparse.Namespace) -> int:
    manifest, _ = base._load_manifest(args.manifest, args.expected_manifest_sha256)
    cells = base._cells(args.cell)
    rows = released_rows(manifest, cells)
    output_root = base._plain_directory(args.output_root, label="output root")
    entries_root = base._plain_directory(output_root / "entries", label="entries root")
    if any(entries_root.iterdir()):
        raise ConfirmationAnchorShardError("entries root must be fresh")
    receipts = [
        base._run_entry(
            row,
            output_root=output_root,
            method_root=base._plain_directory(args.method_root, label="method root"),
            python_bin=base._executable(args.python_bin, label="Python"),
            bernini_root=base._plain_directory(args.bernini_root, label="Bernini root"),
            veomni_root=base._plain_directory(args.veomni_root, label="VeOmni root"),
            checkpoint=base._plain_directory(args.checkpoint, label="checkpoint"),
            master_port=args.master_port,
            method_source_revision=args.method_source_revision,
            method_source_archive_sha256=args.method_source_archive_sha256,
            manifest_sha256=args.expected_manifest_sha256,
            manifest_digest=manifest["manifest_digest"],
        )
        for row in rows
    ]
    value = receipt(
        manifest_sha256=args.expected_manifest_sha256,
        manifest_digest=manifest["manifest_digest"],
        cells=cells,
        entry_receipts=receipts,
    )
    base._write_create_only(output_root / "confirmation-anchor-shard.receipt.json", value)
    print(json.dumps({"cells": len(cells), "entries": len(receipts)}, sort_keys=True))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    manifest, _ = base._load_manifest(args.manifest, args.expected_manifest_sha256)
    cells = base._cells(args.cell)
    rows = released_rows(manifest, cells)
    output_root = base._plain_directory(args.output_root, label="output root")
    receipts = [
        base._verify_entry(row, output_root, args.expected_manifest_sha256) for row in rows
    ]
    expected = receipt(
        manifest_sha256=args.expected_manifest_sha256,
        manifest_digest=manifest["manifest_digest"],
        cells=cells,
        entry_receipts=receipts,
    )
    observed = base._read_object(
        base._plain_file(
            output_root / "confirmation-anchor-shard.receipt.json",
            label="confirmation anchor shard receipt",
        ),
        label="confirmation anchor shard receipt",
    )
    if observed != expected:
        raise ConfirmationAnchorShardError("confirmation anchor shard receipt differs")
    print(json.dumps({"verified_cells": len(cells)}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--cell", action="append", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--master-port", type=int, required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--method-root", required=True)
    run.add_argument("--python-bin", required=True)
    run.add_argument("--bernini-root", required=True)
    run.add_argument("--veomni-root", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--method-source-revision", required=True)
    run.add_argument("--method-source-archive-sha256", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return command_run(args) if args.command == "run" else command_verify(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfirmationAnchorShardError, base.FactorialBranchRunError) as error:
        print(f"[confirmation-anchor-shard] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
