#!/usr/bin/env python3
"""R6 exact28 specialization of the frozen source-bound DINO raw diagnostic."""

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


SCHEMA_VERSION = "bernini-saic-r6-partial28-source-bound-dinov2-raw-v1"
INPUT_SCHEMA = f"{SCHEMA_VERSION}-input"
SHARD_SCHEMA = f"{SCHEMA_VERSION}-shard"
AGGREGATE_SCHEMA = f"{SCHEMA_VERSION}-aggregate"
PREFLIGHT_SCHEMA = f"{SCHEMA_VERSION}-preflight"
EXPECTED_ATTEMPT_COUNT = 28
EXPECTED_WORLD_SIZE = 8
EXPECTED_PARTITION_SIZES = (4, 4, 4, 4, 3, 3, 3, 3)

EXPECTED_RUN_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/runs/"
    "t2v-events-topup-r6-umaskfix-72f3a40-r1"
)
EXPECTED_ATTEMPTS_ROOT = f"{EXPECTED_RUN_ROOT}/attempts"
EXPECTED_ROOT_SPEC_PATH = f"{EXPECTED_RUN_ROOT}/sealed-saic-t2v-event-topup-v2-spec.json"
EXPECTED_ROOT_SPEC_SHA256 = "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145"
EXPECTED_ROOT_SPEC_CONTENT_SHA256 = "af2dfc387a96ade19518c5bb5313d9485683510cdbd80a4f63b1cb0746683065"
EXPECTED_BASE_SPEC_PATH = f"{EXPECTED_RUN_ROOT}/sealed-base-saic-t2v-event-v1-spec.json"
EXPECTED_BASE_SPEC_SHA256 = "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
EXPECTED_BASE_SPEC_CONTENT_SHA256 = "3920d5c121b75c6bbf984c24440c9773dfb49006778c61a671ae50963bb5456a"
EXPECTED_SOURCE_MANIFEST_PATH = f"{EXPECTED_RUN_ROOT}/sealed-saic-source-manifest.json"
EXPECTED_SOURCE_MANIFEST_SHA256 = "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256 = "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f"
EXPECTED_RECEIPT_BINDING_SHA256 = "f3e1717fb86298a5e0995d6a70322709c4b9df0614c22e7eb63fe927e35dcb92"

SOURCE_VIDEO_PREFIX = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/goku_action_wan22_20260730T043022Z/"
    "fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples"
)
EXPECTED_SOURCE_SHA256_BY_IID = {
    "311c82f83eca4a7f": "cc329d61942f02e9e58c43e455e95bf892dcf8b8cd5c2785c3fff1e2e6fbd798",
    "31c34509415745ca": "2c34ec7f74faa624909f966d3d98f485f4b92250905843e2a7dd67a44d24fda6",
    "6d346c38cf504493": "99483c3d0dacb86658841df54452c0a3492563f197813c4a2526e4f096dba14f",
    "6ea45d35943742bb": "a4f0567c8c7a63421a34099bd16c13f4ca77dd4bbe9b0ce3725a363e96b6d193",
    "7b88a1ca1f804f41": "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed",
    "841b5e0080a1441d": "5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a",
    "99cde432839f4240": "4fcd6d75b09e3b294d3dfff15fc7b523a536551e592484eed91de66e2e733a2c",
    "a35b590961d24694": "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed",
}
EXPECTED_ACTOR_BY_IID = {
    "311c82f83eca4a7f": "human", "31c34509415745ca": "human",
    "6d346c38cf504493": "human", "a35b590961d24694": "human",
    "6ea45d35943742bb": "dog", "7b88a1ca1f804f41": "dog",
    "841b5e0080a1441d": "dog", "99cde432839f4240": "dog",
}
EXPECTED_WRONG_IID_BY_IID = {
    "311c82f83eca4a7f": "31c34509415745ca",
    "31c34509415745ca": "6d346c38cf504493",
    "6d346c38cf504493": "a35b590961d24694",
    "a35b590961d24694": "311c82f83eca4a7f",
    "6ea45d35943742bb": "7b88a1ca1f804f41",
    "7b88a1ca1f804f41": "841b5e0080a1441d",
    "841b5e0080a1441d": "99cde432839f4240",
    "99cde432839f4240": "6ea45d35943742bb",
}

AUTHORITY_CLOSURE = dict(core.AUTHORITY_CLOSURE)
SourceBoundRaw28Error = core.SourceBoundRawError
_base_build_manifest = core.build_manifest
_base_load_input_manifest = core.load_input_manifest


def _configure_core() -> None:
    core.__file__ = __file__
    core.core.__file__ = __file__
    core.SCHEMA_VERSION = SCHEMA_VERSION
    core.INPUT_SCHEMA = INPUT_SCHEMA
    core.SHARD_SCHEMA = SHARD_SCHEMA
    core.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
    core.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    core.EXPECTED_ATTEMPT_COUNT = EXPECTED_ATTEMPT_COUNT
    core.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE
    core.EXPECTED_PARTITION_SIZES = EXPECTED_PARTITION_SIZES
    core.AUTHORITY_CLOSURE = AUTHORITY_CLOSURE
    core.core.SCHEMA_VERSION = SCHEMA_VERSION
    core.core.INPUT_SCHEMA = INPUT_SCHEMA
    core.core.SHARD_SCHEMA = SHARD_SCHEMA
    core.core.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
    core.core.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    core.core.EXPECTED_ATTEMPT_COUNT = EXPECTED_ATTEMPT_COUNT
    core.core.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE
    core.core.AUTHORITY_CLOSURE = AUTHORITY_CLOSURE
    core.core.partition_indices = partition_indices


def partition_indices(count: int, rank: int, world_size: int) -> tuple[int, ...]:
    if count != EXPECTED_ATTEMPT_COUNT or world_size != EXPECTED_WORLD_SIZE:
        raise SourceBoundRaw28Error("r6 partial28 source-bound partition geometry differs")
    core.core._rank(rank, world_size=world_size)
    indices = tuple(index for index in range(count) if index % world_size == rank)
    sizes = tuple(
        len(tuple(index for index in range(count) if index % world_size == item))
        for item in range(world_size)
    )
    if sizes != EXPECTED_PARTITION_SIZES:
        raise SourceBoundRaw28Error("r6 partial28 source-bound partition sizes differ")
    return indices


def _configure_partitions() -> None:
    core._base_partition_indices = partition_indices
    core.partition_indices = partition_indices
    core.core.partition_indices = partition_indices


def _source_path(iid: str) -> str:
    return f"{SOURCE_VIDEO_PREFIX}/{iid}/samples/{iid}/source_video.mp4"


def _receipt_binding_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    bindings = []
    for row in rows:
        candidate_id = row.get("candidate_id")
        receipt_sha256 = row.get("receipt_sha256")
        if not isinstance(candidate_id, str) or not isinstance(receipt_sha256, str):
            raise SourceBoundRaw28Error("r6 receipt binding fields differ")
        bindings.append({"candidate_id": candidate_id, "receipt_sha256": receipt_sha256})
    bindings.sort(key=lambda item: item["candidate_id"])
    return core.core.object_sha256(bindings)


def _validate_fixed_build_inputs(args: Any) -> None:
    if str(Path(args.attempts_root)) != EXPECTED_ATTEMPTS_ROOT:
        raise SourceBoundRaw28Error("r6 attempts-root lexical path differs")
    attempts_root = core.core._plain_directory(args.attempts_root, label="r6 attempts root")
    if str(attempts_root) != EXPECTED_ATTEMPTS_ROOT:
        raise SourceBoundRaw28Error("r6 attempts-root resolved path differs")
    if str(Path(args.source_manifest)) != EXPECTED_SOURCE_MANIFEST_PATH:
        raise SourceBoundRaw28Error("r6 source-manifest lexical path differs")
    if args.expected_root_spec_sha256 != EXPECTED_ROOT_SPEC_SHA256:
        raise SourceBoundRaw28Error("r6 root-spec SHA-256 argument differs")
    if args.expected_source_manifest_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise SourceBoundRaw28Error("r6 source-manifest SHA-256 argument differs")

    root_spec, _ = core.core._strict_json(
        EXPECTED_ROOT_SPEC_PATH,
        expected_sha256=EXPECTED_ROOT_SPEC_SHA256,
        label="r6 sealed top-up spec",
    )
    base_spec, _ = core.core._strict_json(
        EXPECTED_BASE_SPEC_PATH,
        expected_sha256=EXPECTED_BASE_SPEC_SHA256,
        label="r6 sealed base spec",
    )
    if (
        core.core.object_sha256(root_spec) != EXPECTED_ROOT_SPEC_CONTENT_SHA256
        or core.core.object_sha256(base_spec) != EXPECTED_BASE_SPEC_CONTENT_SHA256
        or root_spec.get("source_manifest_file_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or root_spec.get("source_manifest_content_sha256") != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or root_spec.get("base_v1_spec_raw_sha256") != EXPECTED_BASE_SPEC_SHA256
        or root_spec.get("base_v1_spec_content_sha256") != EXPECTED_BASE_SPEC_CONTENT_SHA256
        or root_spec.get("top_up_only") is not True
        or root_spec.get("artifact_authority") != core.core.topup_generate.contract.ARTIFACT_AUTHORITY
    ):
        raise SourceBoundRaw28Error("r6 sealed spec/source/authority closure differs")

    paths = sorted(
        attempts_root.rglob(core.core.ATTEMPT_BASENAME),
        key=lambda item: item.as_posix(),
    )
    if len(paths) != EXPECTED_ATTEMPT_COUNT:
        raise SourceBoundRaw28Error("r6 bank is not exact28")
    rows = []
    for path in paths:
        receipt, receipt_sha = core.core._strict_json(
            path, expected_sha256=None, label="r6 generation receipt",
        )
        candidate = receipt.get("candidate")
        rows.append({
            "candidate_id": candidate.get("candidate_id") if isinstance(candidate, Mapping) else None,
            "receipt_sha256": receipt_sha,
        })
    if _receipt_binding_digest(rows) != EXPECTED_RECEIPT_BINDING_SHA256:
        raise SourceBoundRaw28Error("r6 exact28 candidate/receipt frozen set differs")


def build_manifest(args: Any) -> int:
    _validate_fixed_build_inputs(args)
    return _base_build_manifest(args)


def _validate_source_binding(binding: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(binding, Mapping):
        raise SourceBoundRaw28Error(f"{label} source binding is absent")
    iid = binding.get("iid")
    if not isinstance(iid, str) or iid not in EXPECTED_SOURCE_SHA256_BY_IID:
        raise SourceBoundRaw28Error(f"{label} source IID differs")
    if (
        binding.get("actor_family") != EXPECTED_ACTOR_BY_IID[iid]
        or binding.get("source_video_sha256") != EXPECTED_SOURCE_SHA256_BY_IID[iid]
        or binding.get("source_video") != _source_path(iid)
    ):
        raise SourceBoundRaw28Error(f"{label} source path/SHA/actor binding differs")
    return binding


def load_input_manifest(
    path: str | Path, *, expected_sha256: str, expected_source_sha256: str,
) -> tuple[dict[str, Any], str]:
    value, raw_sha = _base_load_input_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_source_sha256=expected_source_sha256,
    )
    attempts = value.get("attempts")
    source_manifest = value.get("source_manifest")
    if (
        value.get("attempts_root") != EXPECTED_ATTEMPTS_ROOT
        or value.get("root_spec_raw_sha256") != EXPECTED_ROOT_SPEC_SHA256
        or not isinstance(attempts, list)
        or _receipt_binding_digest(attempts) != EXPECTED_RECEIPT_BINDING_SHA256
        or not isinstance(source_manifest, Mapping)
        or source_manifest.get("path") != EXPECTED_SOURCE_MANIFEST_PATH
        or source_manifest.get("raw_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or source_manifest.get("content_sha256") != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or source_manifest.get("wrong_source_policy") != core.WRONG_SOURCE_POLICY
    ):
        raise SourceBoundRaw28Error("r6 source-bound input identity differs")
    for row in attempts:
        if not isinstance(row, Mapping):
            raise SourceBoundRaw28Error("r6 attempt row differs")
        correct = _validate_source_binding(row.get("correct_source"), label="correct")
        wrong = _validate_source_binding(row.get("wrong_source"), label="wrong")
        if wrong.get("iid") != EXPECTED_WRONG_IID_BY_IID[correct["iid"]]:
            raise SourceBoundRaw28Error("r6 preregistered wrong-source policy differs")
    return value, raw_sha


def aggregate(args: Any) -> int:
    source_sha = core.core._verify_self(args.expected_source_sha256)
    manifest, manifest_sha = load_input_manifest(
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
            raise SourceBoundRaw28Error(f"shard {rank} contract differs")
        shards.append({
            "rank": rank, "path": str(path.resolve(strict=True)),
            "sha256": raw_sha, "receipt_digest": declared,
        })
        for index, result in zip(indices, results):
            if index in by_index:
                raise SourceBoundRaw28Error("shard partition overlaps")
            by_index[index] = result
    if set(by_index) != set(range(EXPECTED_ATTEMPT_COUNT)):
        raise SourceBoundRaw28Error("shards do not cover exact r6 partial28")
    ordered = [by_index[index] for index in range(EXPECTED_ATTEMPT_COUNT)]
    expected_ids = [row["candidate_id"] for row in manifest["attempts"]]
    if [row.get("candidate_id") for row in ordered] != expected_ids:
        raise SourceBoundRaw28Error("aggregate candidate order differs")
    unsigned = {
        "schema_version": AGGREGATE_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "world_size": EXPECTED_WORLD_SIZE,
        "candidate_count": EXPECTED_ATTEMPT_COUNT,
        "coverage": "exactly_once_complete_r6_partial28_source_bound_raw",
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


def _install_specialization() -> None:
    _configure_core()
    _configure_partitions()
    core.build_manifest = build_manifest
    core.load_input_manifest = load_input_manifest
    core.aggregate = aggregate


def main(argv: Sequence[str] | None = None) -> int:
    _install_specialization()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
