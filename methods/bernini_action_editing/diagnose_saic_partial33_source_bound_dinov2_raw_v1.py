#!/usr/bin/env python3
"""Exact33 specialization of the frozen source-bound DINO raw diagnostic."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


_BASE_BASENAME = "diagnose_saic_partial47_source_bound_dinov2_raw_v1.py"
_BASE_SHA256 = "ffbc9ba149d1ddadf704dd8258678a8893235e328da4c7601e98d63ba37aa7a2"
_BASE_PATH = Path(__file__).resolve().with_name(_BASE_BASENAME)
if not _BASE_PATH.is_file() or _BASE_PATH.is_symlink():
    raise RuntimeError("pinned exact47 source-bound evaluator is absent or not a plain file")
if hashlib.sha256(_BASE_PATH.read_bytes()).hexdigest() != _BASE_SHA256:
    raise RuntimeError("pinned exact47 source-bound evaluator SHA-256 differs")

import diagnose_saic_partial47_source_bound_dinov2_raw_v1 as core  # noqa: E402


SCHEMA_VERSION = "bernini-saic-partial33-source-bound-dinov2-raw-v1"
INPUT_SCHEMA = f"{SCHEMA_VERSION}-input"
SHARD_SCHEMA = f"{SCHEMA_VERSION}-shard"
AGGREGATE_SCHEMA = f"{SCHEMA_VERSION}-aggregate"
PREFLIGHT_SCHEMA = f"{SCHEMA_VERSION}-preflight"
EXPECTED_ATTEMPT_COUNT = 33
EXPECTED_WORLD_SIZE = 8
EXPECTED_PARTITION_SIZES = (5, 4, 4, 4, 4, 4, 4, 4)
AUTHORITY_CLOSURE = dict(core.AUTHORITY_CLOSURE)
SourceBoundRaw33Error = core.SourceBoundRawError


def _configure_core() -> None:
    # The exact47 implementation delegates self-verification and evaluator
    # helpers to its pinned partial28 module.  Both module-global layers must
    # therefore see this exact33 executable identity and geometry.
    core.__file__ = __file__
    core.SCHEMA_VERSION = SCHEMA_VERSION
    core.INPUT_SCHEMA = INPUT_SCHEMA
    core.SHARD_SCHEMA = SHARD_SCHEMA
    core.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
    core.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    core.EXPECTED_ATTEMPT_COUNT = EXPECTED_ATTEMPT_COUNT
    core.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE
    core.AUTHORITY_CLOSURE = AUTHORITY_CLOSURE
    core.core.__file__ = __file__
    core.core.SCHEMA_VERSION = SCHEMA_VERSION
    core.core.INPUT_SCHEMA = INPUT_SCHEMA
    core.core.SHARD_SCHEMA = SHARD_SCHEMA
    core.core.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
    core.core.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    core.core.EXPECTED_ATTEMPT_COUNT = EXPECTED_ATTEMPT_COUNT
    core.core.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE
    core.core.AUTHORITY_CLOSURE = AUTHORITY_CLOSURE


_configure_core()


def partition_indices(count: int, rank: int, world_size: int) -> tuple[int, ...]:
    if count != EXPECTED_ATTEMPT_COUNT or world_size != EXPECTED_WORLD_SIZE:
        raise SourceBoundRaw33Error("partial33 source-bound partition geometry differs")
    if type(rank) is not int or rank < 0 or rank >= world_size:
        raise SourceBoundRaw33Error("rank is outside the fixed partial33 world")
    indices = tuple(range(rank, count, world_size))
    sizes = tuple(len(tuple(range(item, count, world_size))) for item in range(world_size))
    if sizes != EXPECTED_PARTITION_SIZES:
        raise SourceBoundRaw33Error("partial33 source-bound partition sizes differ")
    return indices


core.partition_indices = partition_indices


def aggregate(args: Any) -> int:
    source_sha = core.core._verify_self(args.expected_source_sha256)
    manifest, manifest_sha = core.load_input_manifest(
        args.input_manifest,
        expected_sha256=args.expected_input_manifest_sha256,
        expected_source_sha256=source_sha,
    )
    output_root = core.core._plain_directory(args.output_root, label="output root")
    shards: list[dict[str, Any]] = []
    by_index: dict[int, Mapping[str, Any]] = {}
    for rank in range(EXPECTED_WORLD_SIZE):
        path = output_root / f"shard-{rank:02d}-of-{EXPECTED_WORLD_SIZE:02d}.json"
        value, raw_sha = core.core._strict_json(
            path, expected_sha256=None, label=f"shard {rank}",
        )
        unsigned = dict(value)
        declared = core.core._sha256(
            unsigned.pop("receipt_digest", None), label="shard digest",
        )
        indices = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, EXPECTED_WORLD_SIZE)
        results = value.get("candidate_results")
        if (
            value.get("schema_version") != SHARD_SCHEMA
            or value.get("diagnostic_source_sha256") != source_sha
            or value.get("input_manifest_sha256") != manifest_sha
            or value.get("rank") != rank
            or value.get("world_size") != EXPECTED_WORLD_SIZE
            or value.get("partition_indices") != list(indices)
            or value.get("candidate_count") != len(indices)
            or not isinstance(results, list)
            or len(results) != len(indices)
            or value.get("authority") != AUTHORITY_CLOSURE
            or declared != core.core.object_sha256(unsigned)
        ):
            raise SourceBoundRaw33Error(f"shard {rank} contract differs")
        shards.append({
            "rank": rank, "path": str(path.resolve(strict=True)),
            "sha256": raw_sha, "receipt_digest": declared,
        })
        for index, result in zip(indices, results):
            if index in by_index:
                raise SourceBoundRaw33Error("shard partition overlaps")
            by_index[index] = result
    if set(by_index) != set(range(EXPECTED_ATTEMPT_COUNT)):
        raise SourceBoundRaw33Error("shards do not cover exact partial33")
    ordered = [by_index[index] for index in range(EXPECTED_ATTEMPT_COUNT)]
    expected_ids = [row["candidate_id"] for row in manifest["attempts"]]
    if [row.get("candidate_id") for row in ordered] != expected_ids:
        raise SourceBoundRaw33Error("aggregate candidate order differs")
    unsigned = {
        "schema_version": AGGREGATE_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "world_size": EXPECTED_WORLD_SIZE,
        "candidate_count": EXPECTED_ATTEMPT_COUNT,
        "coverage": "exactly_once_complete_partial33_source_bound_raw",
        "candidate_order": expected_ids,
        "shards": shards,
        "candidate_results": ordered,
        "interpretation": {
            "measurement": "raw frozen-DINO candidate/correct/wrong source proxies and source-self upper bounds",
            "wrong_source_preregistered_without_candidate_metrics": True,
            "no_absolute_preservation_claim": True,
            "no_event_measurement": True,
            "no_threshold_or_ranking": True,
        },
        "authority": dict(AUTHORITY_CLOSURE),
    }
    core.core._write_create_only(
        output_root / "aggregate-receipt.json",
        {**unsigned, "receipt_digest": core.core.object_sha256(unsigned)},
    )
    return 0


core.aggregate = aggregate


def main(argv: Sequence[str] | None = None) -> int:
    _configure_core()
    core.partition_indices = partition_indices
    core.aggregate = aggregate
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
