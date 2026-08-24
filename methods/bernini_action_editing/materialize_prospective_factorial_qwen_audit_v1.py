#!/usr/bin/env python3
"""Expose a verified factorial release to the pinned Qwen-v6 triage reader.

The adapter creates byte-identical, read-only audit views for the six branches
understood by ``audit_saic_t2v_branch_semantics_qwen_v1.py``.  Wrong-owner is
deliberately left for a dedicated owner-aware evaluator and human review.
Nothing produced here grants target, training, optimizer, or selection
authority.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import prospective_factorial_branch_manifest_v1 as branch_manifest  # noqa: E402
import run_prospective_factorial_branch_shard_v1 as runner  # noqa: E402


SCHEMA_VERSION = "bernini-prospective-factorial-qwen-audit-view-v1"
ANCHOR_BRANCHES = frozenset({"forward", "reverse", "noop"})
COUNTERFACTUAL_BRANCHES = frozenset(
    {"incomplete", "camera_only", "appearance_only"}
)
SUPPORTED_BRANCHES = ANCHOR_BRANCHES | COUNTERFACTUAL_BRANCHES
OMITTED_BRANCHES = frozenset({"wrong_actor_or_object"})
AUTHORITY = {
    "data_selection": False,
    "human_review": False,
    "optimizer": False,
    "scientific_claim": False,
    "training": False,
    "training_target": False,
}


class FactorialQwenViewError(RuntimeError):
    """Raised before a partial or mutable audit view can be published."""


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_create_only(source: Path, destination: Path) -> None:
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _actor_family(row: Mapping[str, Any]) -> str:
    family = str(row["action_family"])
    if family.startswith("dog-"):
        return "dog"
    if family.startswith("human-"):
        return "human"
    raise FactorialQwenViewError("action family has no Qwen-v6 actor mapping")


def materialize(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    cells: Sequence[str],
    output_root: str | Path,
    audit_root: str | Path,
) -> dict[str, Any]:
    try:
        manifest, _ = runner._load_manifest(
            manifest_path, expected_manifest_sha256
        )
        parsed_cells = runner._cells(cells)
        rows = runner._released_entries(manifest, parsed_cells, "fit")
        released_root = runner._plain_directory(
            output_root, label="factorial release output root"
        )
        receipts = {
            row["entry_id"]: runner._verify_entry(
                row, released_root, expected_manifest_sha256
            )
            for row in rows
        }
    except (
        runner.FactorialBranchRunError,
        branch_manifest.FactorialBranchManifestError,
    ) as error:
        raise FactorialQwenViewError(str(error)) from error

    destination = Path(audit_root)
    if not destination.is_absolute() or destination == Path("/"):
        raise FactorialQwenViewError("audit root must be an absolute non-root path")
    if destination.exists() or destination.is_symlink():
        raise FactorialQwenViewError("audit root must be fresh")
    destination.mkdir(mode=0o700)
    attempts = destination / "attempts"
    attempts.mkdir(mode=0o700)

    published: list[dict[str, Any]] = []
    for row in rows:
        branch = str(row["branch"])
        if branch in OMITTED_BRANCHES:
            continue
        if branch not in SUPPORTED_BRANCHES:
            raise FactorialQwenViewError("factorial branch support differs")
        entry_id = str(row["entry_id"])
        source_video = released_root / "entries" / entry_id / "output.mp4"
        attempt = attempts / entry_id
        attempt.mkdir(mode=0o700)
        audit_video = attempt / "t2v.mp4"
        _copy_create_only(source_video, audit_video)
        if branch_manifest.file_sha256(audit_video) != receipts[entry_id]["output_sha256"]:
            raise FactorialQwenViewError("audit video copy digest differs")
        receipt_name = (
            "saic-event-generation-receipt.json"
            if branch in ANCHOR_BRANCHES
            else "saic-event-topup-generation-receipt.json"
        )
        candidate = {
            "candidate_id": entry_id,
            "iid": str(row["source_id"]),
            "branch": branch,
            "analysis_split": "fit",
            "actor_family": _actor_family(row),
            "action_family_id": str(row["action_family"]),
            "seed": int(row["seed"]),
            "branch_start_state_caption": str(
                manifest["source_states"][row["source_id"]]["initial_state"]
            ),
            "branch_instruction": str(row["instruction"]),
            "event_verified": False,
            "event_audit_status": "pending_detached_full81_review",
        }
        audit_receipt = {
            "schema_version": SCHEMA_VERSION,
            "candidate": candidate,
            "factorial_entry_receipt_digest": receipts[entry_id]["receipt_digest"],
            "factorial_manifest_sha256": expected_manifest_sha256,
            "byte_identical_read_only_view": True,
            "authority": dict(AUTHORITY),
        }
        _write_create_only(attempt / receipt_name, audit_receipt)
        published.append(
            {
                "entry_id": entry_id,
                "branch": branch,
                "video_sha256": receipts[entry_id]["output_sha256"],
                "receipt": receipt_name,
            }
        )

    expected_count = len(parsed_cells) * len(SUPPORTED_BRANCHES)
    if len(published) != expected_count:
        raise FactorialQwenViewError("published Qwen audit view is incomplete")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_qwen_v6_diagnostic_view_not_training_authority",
        "manifest_sha256": expected_manifest_sha256,
        "released_cells": [f"{source}:{seed}" for source, seed in parsed_cells],
        "published_entry_count": len(published),
        "supported_branches": sorted(SUPPORTED_BRANCHES),
        "omitted_branches": sorted(OMITTED_BRANCHES),
        "entries": published,
        "authority": dict(AUTHORITY),
    }
    result = {**unsigned, "receipt_digest": branch_manifest.object_sha256(unsigned)}
    _write_create_only(destination / "materialization.receipt.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--audit-root", required=True)
    args = parser.parse_args(argv)
    result = materialize(
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        cells=args.cell,
        output_root=args.output_root,
        audit_root=args.audit_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FactorialQwenViewError as error:
        print(f"[factorial-qwen-view] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
