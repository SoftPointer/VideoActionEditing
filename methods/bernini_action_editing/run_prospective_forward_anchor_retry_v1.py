#!/usr/bin/env python3
"""Render disclosed fit-only retry seeds for failed forward-anchor cells.

Retries keep the frozen source and instruction from the prospective manifest.
Only the sampling seed changes.  Outputs remain decoded-review candidates: this
program grants neither representation-selection nor optimizer authority.
"""

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
    not BASE_PATH.is_file()
    or BASE_PATH.is_symlink()
    or hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256
):
    raise RuntimeError("pinned factorial branch runtime dependency differs")

import run_prospective_factorial_branch_shard_v1 as base  # noqa: E402


SCHEMA_VERSION = "bernini-prospective-forward-anchor-retry-v1"
AUTHORITY = {
    "fit_retry_generation_only": True,
    "decoded_review_required": True,
    "post_review_seed_expansion_disclosed": True,
    "representation_selection_authorized": False,
    "training_target_authorized": False,
    "optimizer_step_authorized": False,
    "method_success_claimed": False,
}


class ForwardAnchorRetryError(RuntimeError):
    """Raised before an invalid retry candidate is rendered or accepted."""


def retry_row(
    manifest: Mapping[str, Any], source_id: str, retry_seed: int
) -> tuple[dict[str, Any], list[int]]:
    if type(source_id) is not str or len(source_id) != 16:
        raise ForwardAnchorRetryError("source ID syntax differs")
    if type(retry_seed) is not int or not 0 <= retry_seed < 2**63:
        raise ForwardAnchorRetryError("retry seed differs")
    rows = [
        dict(row)
        for row in manifest["entries"]
        if row["source_id"] == source_id
        and row["analysis_split"] == "fit"
        and row["branch"] == "forward"
    ]
    if len(rows) != 2:
        raise ForwardAnchorRetryError("source does not have two registered fit rows")
    registered_seeds = sorted(row["seed"] for row in rows)
    if retry_seed in registered_seeds:
        raise ForwardAnchorRetryError("retry seed aliases a registered seed")
    template = rows[0]
    invariant = (
        "action_family",
        "analysis_split",
        "branch",
        "executor",
        "instruction",
        "instruction_utf8_sha256",
        "source_id",
        "source_video",
        "source_video_sha256",
    )
    if any(row[key] != template[key] for row in rows[1:] for key in invariant):
        raise ForwardAnchorRetryError("registered forward rows differ beyond seed")
    template["seed"] = retry_seed
    template["entry_id"] = f"{source_id}-retry-s{retry_seed}-forward"
    return template, registered_seeds


def retry_receipt(
    *, manifest_sha256: str, manifest_digest: str, row: Mapping[str, Any],
    registered_seeds: Sequence[int], entry_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "fit_retry_rendered_decoded_review_pending",
        "manifest_sha256": manifest_sha256,
        "manifest_digest": manifest_digest,
        "source_id": row["source_id"],
        "action_family": row["action_family"],
        "retry_seed": row["seed"],
        "registered_seeds": list(registered_seeds),
        "instruction_utf8_sha256": row["instruction_utf8_sha256"],
        "source_video_sha256": row["source_video_sha256"],
        "entry_receipt_digest": entry_receipt["receipt_digest"],
        "authority": dict(AUTHORITY),
    }
    return {**unsigned, "receipt_digest": base.branch_manifest.object_sha256(unsigned)}


def _row_and_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    manifest, _ = base._load_manifest(args.manifest, args.expected_manifest_sha256)
    row, registered = retry_row(manifest, args.source_id, args.retry_seed)
    return manifest, row, registered


def command_run(args: argparse.Namespace) -> int:
    manifest, row, registered = _row_and_manifest(args)
    output_root = base._plain_directory(args.output_root, label="output root")
    entries_root = base._plain_directory(output_root / "entries", label="entries root")
    if any(entries_root.iterdir()):
        raise ForwardAnchorRetryError("entries root must be fresh")
    entry = base._run_entry(
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
    receipt = retry_receipt(
        manifest_sha256=args.expected_manifest_sha256,
        manifest_digest=manifest["manifest_digest"],
        row=row,
        registered_seeds=registered,
        entry_receipt=entry,
    )
    base._write_create_only(output_root / "forward-anchor-retry.receipt.json", receipt)
    print(json.dumps({"source_id": row["source_id"], "retry_seed": row["seed"]}, sort_keys=True))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    manifest, row, registered = _row_and_manifest(args)
    output_root = base._plain_directory(args.output_root, label="output root")
    entry = base._verify_entry(row, output_root, args.expected_manifest_sha256)
    expected = retry_receipt(
        manifest_sha256=args.expected_manifest_sha256,
        manifest_digest=manifest["manifest_digest"],
        row=row,
        registered_seeds=registered,
        entry_receipt=entry,
    )
    observed = base._read_object(
        base._plain_file(
            output_root / "forward-anchor-retry.receipt.json",
            label="forward anchor retry receipt",
        ),
        label="forward anchor retry receipt",
    )
    if observed != expected:
        raise ForwardAnchorRetryError("forward anchor retry receipt differs")
    print(json.dumps({"verified_source_id": row["source_id"], "retry_seed": row["seed"]}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--retry-seed", type=int, required=True)
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
    except (ForwardAnchorRetryError, base.FactorialBranchRunError) as error:
        print(f"[forward-anchor-retry] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
