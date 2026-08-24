#!/usr/bin/env python3
"""Exact47 diagnostic-only DINO evaluation for the historical SAIC r4 bank.

This is a fail-closed specialization of the audited partial28 evaluator.  It
pins that implementation byte-for-byte, changes only the exact bank geometry
and schema namespace, and emits an exact47 aggregate with no scientific,
identity, event, ranking, selection, training, or optimizer authority.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


_BASE_BASENAME = "diagnose_saic_partial28_dinov2_temporal_v1.py"
_BASE_SHA256 = "213e408295610b5a0dd2e1eeb54f406c19a1985fb1ff290f89522fd38b4aaf4d"
_BASE_PATH = Path(__file__).resolve().with_name(_BASE_BASENAME)
if not _BASE_PATH.is_file() or _BASE_PATH.is_symlink():
    raise RuntimeError("pinned partial28 evaluator dependency is absent or not a plain file")
if hashlib.sha256(_BASE_PATH.read_bytes()).hexdigest() != _BASE_SHA256:
    raise RuntimeError("pinned partial28 evaluator dependency SHA-256 differs")

import diagnose_saic_partial28_dinov2_temporal_v1 as core  # noqa: E402


SCHEMA_VERSION = "bernini-saic-partial47-dinov2-temporal-diagnostic-v1"
INPUT_SCHEMA = f"{SCHEMA_VERSION}-input"
SHARD_SCHEMA = f"{SCHEMA_VERSION}-shard"
AGGREGATE_SCHEMA = f"{SCHEMA_VERSION}-aggregate"
PREFLIGHT_SCHEMA = f"{SCHEMA_VERSION}-preflight"
EXPECTED_ATTEMPT_COUNT = 47
EXPECTED_WORLD_SIZE = 8
EXPECTED_PARTITION_SIZES = (6, 6, 6, 6, 6, 6, 6, 5)
AUTHORITY_CLOSURE = dict(core.AUTHORITY_CLOSURE)
Partial47DINOError = core.Partial28DINOError
_base_partition_indices = core.partition_indices


def _configure_core() -> None:
    # Every inherited function resolves these names in the core module globals.
    # Point self-authentication at this exact47 source while retaining the
    # byte-pinned implementation for all validation and measurement behavior.
    core.__file__ = __file__
    core.SCHEMA_VERSION = SCHEMA_VERSION
    core.INPUT_SCHEMA = INPUT_SCHEMA
    core.SHARD_SCHEMA = SHARD_SCHEMA
    core.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
    core.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    core.EXPECTED_ATTEMPT_COUNT = EXPECTED_ATTEMPT_COUNT
    core.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE


_configure_core()


def partition_indices(count: int, rank: int, world_size: int) -> tuple[int, ...]:
    indices = _base_partition_indices(count, rank, world_size)
    if tuple(len(_base_partition_indices(count, item, world_size)) for item in range(world_size)) != EXPECTED_PARTITION_SIZES:
        raise Partial47DINOError("partial47 partition sizes differ")
    return indices


# Make inherited worker/preflight/shard validation use the checked specialization.
core.partition_indices = partition_indices


def aggregate(args: Any) -> int:
    source_sha = core._verify_self(args.expected_source_sha256)
    manifest, manifest_sha = core.load_input_manifest(
        args.input_manifest,
        expected_sha256=args.expected_input_manifest_sha256,
        expected_source_sha256=source_sha,
    )
    output_root = core._plain_directory(args.output_root, label="output root")
    shards: list[dict[str, Any]] = []
    by_index: dict[int, Mapping[str, Any]] = {}
    for rank in range(EXPECTED_WORLD_SIZE):
        path = output_root / f"shard-{rank:02d}-of-{EXPECTED_WORLD_SIZE:02d}.json"
        value, raw_sha = core._strict_json(path, expected_sha256=None, label=f"shard {rank}")
        checked = core._validate_shard(
            value, rank=rank, source_sha=source_sha, manifest_sha=manifest_sha,
        )
        shards.append({
            "rank": rank,
            "path": str(path.resolve(strict=True)),
            "sha256": raw_sha,
            "receipt_digest": checked["receipt_digest"],
        })
        for index, result in zip(checked["partition_indices"], checked["candidate_results"]):
            if index in by_index:
                raise Partial47DINOError("shard partition overlaps")
            by_index[index] = result
    if set(by_index) != set(range(EXPECTED_ATTEMPT_COUNT)):
        raise Partial47DINOError("shards do not cover exact partial47")
    ordered = [by_index[index] for index in range(EXPECTED_ATTEMPT_COUNT)]
    expected_ids = [row["candidate_id"] for row in manifest["attempts"]]
    if [row.get("candidate_id") for row in ordered] != expected_ids:
        raise Partial47DINOError("aggregate candidate order differs")
    unsigned = {
        "schema_version": AGGREGATE_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "world_size": EXPECTED_WORLD_SIZE,
        "candidate_count": EXPECTED_ATTEMPT_COUNT,
        "coverage": "exactly_once_complete_partial47",
        "candidate_order": expected_ids,
        "shards": shards,
        "candidate_results": ordered,
        "interpretation": {
            "measurement": "same-video frozen-DINO temporal appearance stability proxy plus exact81 decode evidence",
            "no_source_comparison": True,
            "no_event_measurement": True,
            "no_threshold_or_ranking": True,
            "cannot_verify_identity_or_action_editing_success": True,
        },
        "authority": dict(AUTHORITY_CLOSURE),
    }
    receipt = {**unsigned, "receipt_digest": core.object_sha256(unsigned)}
    core._write_create_only(output_root / "aggregate-receipt.json", receipt)
    return 0


core.aggregate = aggregate


def main(argv: Sequence[str] | None = None) -> int:
    _configure_core()
    core.partition_indices = partition_indices
    core.aggregate = aggregate
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
