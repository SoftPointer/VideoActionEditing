#!/usr/bin/env python3
"""R8 exact60 DINO diagnostic against all three same-actor sources.

The negative universe is derived only from the sealed source-manifest order.
The completed cyclic r8 aggregate is a required, byte-pinned regression input:
its sixty candidate results and all eight shards are revalidated before any
all-three manifest is admitted.  This remains descriptive diagnostic evidence
and grants no scientific, ranking, selection, training, or optimizer authority.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
ALGORITHM_SOURCE_NAME = (
    "diagnose_saic_partial47_source_bound_dinov2_"
    "same_actor_all_three_negatives_raw_v1.py"
)
ALGORITHM_SOURCE_SHA256 = (
    "847f3cf31553cb8a73d276278026dbdd898ed4bd094fd973978ce1f143be50dd"
)
CYCLIC_SOURCE_NAME = "diagnose_saic_r8_exact60_source_bound_dinov2_raw_v1.py"
CYCLIC_SOURCE_SHA256 = (
    "2839641766b4605311f0c4e7a3ff41ea9a322ad44cf1d12d21fb7f0cca5ab24e"
)
SOURCE_VALIDATOR_NAME = "build_saic_reversible_source_set_v1.py"
SOURCE_VALIDATOR_SHA256 = (
    "0cf012adf25dd1afffb33d1e0c918630a574c9075e9aa293914e04890c71cf5b"
)
for _name, _sha256 in (
    (ALGORITHM_SOURCE_NAME, ALGORITHM_SOURCE_SHA256),
    (CYCLIC_SOURCE_NAME, CYCLIC_SOURCE_SHA256),
    (SOURCE_VALIDATOR_NAME, SOURCE_VALIDATOR_SHA256),
):
    _path = METHOD_ROOT / _name
    if (
        not _path.is_file()
        or _path.is_symlink()
        or hashlib.sha256(_path.read_bytes()).hexdigest() != _sha256
    ):
        raise RuntimeError(f"pinned r8 all-three dependency differs: {_name}")

import diagnose_saic_partial47_source_bound_dinov2_same_actor_all_three_negatives_raw_v1 as algorithm  # noqa: E402
import diagnose_saic_r8_exact60_source_bound_dinov2_raw_v1 as cyclic  # noqa: E402


frozen = algorithm.frozen
core = algorithm.core
SCHEMA_VERSION = (
    "bernini-saic-r8-exact60-source-bound-dinov2-"
    "same-actor-all-three-negatives-raw-v1"
)
INPUT_SCHEMA = f"{SCHEMA_VERSION}-input"
SHARD_SCHEMA = f"{SCHEMA_VERSION}-shard"
AGGREGATE_SCHEMA = f"{SCHEMA_VERSION}-aggregate"
PREFLIGHT_SCHEMA = f"{SCHEMA_VERSION}-preflight"
EXPECTED_ATTEMPT_COUNT = 60
EXPECTED_WORLD_SIZE = 8
EXPECTED_PARTITION_SIZES = (8, 8, 8, 8, 7, 7, 7, 7)
EXPECTED_SOURCE_COUNT = 8
EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE = 3
EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT = 180
EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT = 24
EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT = 24
EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT = 8
EXPECTED_MISSING_CORRECT_SOURCE_IIDS: frozenset[str] = frozenset()

EXPECTED_ATTEMPTS_ROOT = cyclic.EXPECTED_ATTEMPTS_ROOT
EXPECTED_ROOT_SPEC_SHA256 = cyclic.EXPECTED_ROOT_SPEC_SHA256
EXPECTED_SOURCE_MANIFEST_PATH = cyclic.EXPECTED_SOURCE_MANIFEST_PATH
EXPECTED_SOURCE_MANIFEST_SHA256 = cyclic.EXPECTED_SOURCE_MANIFEST_SHA256
EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256 = (
    cyclic.EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
)
EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256 = (
    "257d3aafaaee126ff2c1a061413d01bd0457676eb5d1ee027671221a5a794218"
)
EXPECTED_TERMINAL_EVIDENCE_PATH = cyclic.EXPECTED_TERMINAL_EVIDENCE_PATH
EXPECTED_EVALUATOR_SPEC_SHA256 = cyclic.core.EXPECTED_EVALUATOR_SPEC_SHA256
EXPECTED_VISUAL_SCORER_SHA256 = cyclic.core.EXPECTED_VISUAL_SCORER_SHA256
EXPECTED_VISUAL_CONTRACT_SHA256 = cyclic.core.EXPECTED_VISUAL_CONTRACT_SHA256
EXPECTED_VISUAL_EVALUATOR = cyclic.EXPECTED_VISUAL_EVALUATOR
EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256 = (
    cyclic.EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256
)

NEGATIVE_SOURCE_POLICY = (
    "same_actor_family_sealed_manifest_order_all_other_three_v1"
)
LEGACY_CYCLIC_POLICY = cyclic.core.WRONG_SOURCE_POLICY
FROZEN_CYCLIC_AGGREGATE_SCHEMA = cyclic.AGGREGATE_SCHEMA
FROZEN_CYCLIC_AUTHORITY_CLOSURE = dict(cyclic.AUTHORITY_CLOSURE)
EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/diagnostics/"
    "allocation-134939-r8-sourcebound60-dinov2-full-28396417-r1/"
    "input-manifest.json"
)
EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256 = (
    "28ff1e40f4dd314548616050013afdfb5e2a2a768aba9f0cbd4f00c9f6718c62"
)
EXPECTED_LEGACY_CYCLIC_INPUT_RECEIPT_DIGEST = (
    "a4ad17a09e46d549089356ecf86e21d5d8a6da2f41aaa92d218b058a9e28f378"
)
EXPECTED_LEGACY_CYCLIC_AGGREGATE_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/diagnostics/"
    "allocation-134939-r8-sourcebound60-dinov2-full-28396417-r1/"
    "aggregate-receipt.json"
)
EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256 = (
    "2e10fd8539d42aecb8872bba3e504e26d7e2dfb9a5120b1145080f8b463dc7fb"
)
EXPECTED_LEGACY_CYCLIC_AGGREGATE_RECEIPT_DIGEST = (
    "68c1670836f01ff2d147237ca70ee03914b4c8735438a5ffa9295621de4161e1"
)

AUTHORITY_CLOSURE = {
    **FROZEN_CYCLIC_AUTHORITY_CLOSURE,
    "multi_negative_proxy_authority": False,
    "formal_retained_source_fd_authority": False,
}
OPERATIONAL_LIMITATION = {
    "operational_diagnostic_only": True,
    "exact8_source_features_cached_per_rank": True,
    "exact8_source_files_retained_open_for_full_process_lifetime": False,
    "formal_source_retained_fd_closure_satisfied": False,
    "formal_or_training_admission_authorized": False,
}
AllThreeNegativeRawError = algorithm.AllThreeNegativeRawError

_base_source_closure = algorithm._source_closure
_base_worker_common = algorithm._worker_common
_base_preflight = algorithm.preflight
_base_worker = algorithm.worker
_base_cache_hash_map = algorithm._cache_hash_map
_base_validate_candidate_result = algorithm._validate_candidate_result


def partition_indices(count: int, rank: int, world_size: int) -> tuple[int, ...]:
    if count != EXPECTED_ATTEMPT_COUNT or world_size != EXPECTED_WORLD_SIZE:
        raise AllThreeNegativeRawError("r8 all-three partition geometry differs")
    if type(rank) is not int or rank < 0 or rank >= world_size:
        raise AllThreeNegativeRawError("rank is outside the fixed r8 world")
    indices = tuple(range(rank, count, world_size))
    sizes = tuple(len(tuple(range(item, count, world_size))) for item in range(world_size))
    if sizes != EXPECTED_PARTITION_SIZES:
        raise AllThreeNegativeRawError("r8 all-three partition sizes differ")
    return indices


def _configure_core() -> None:
    for module in (algorithm, frozen, core):
        module.__file__ = __file__
        module.SCHEMA_VERSION = SCHEMA_VERSION
        module.INPUT_SCHEMA = INPUT_SCHEMA
        module.SHARD_SCHEMA = SHARD_SCHEMA
        module.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
        module.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
        module.EXPECTED_ATTEMPT_COUNT = EXPECTED_ATTEMPT_COUNT
        module.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE
        module.AUTHORITY_CLOSURE = AUTHORITY_CLOSURE
        module.partition_indices = partition_indices
    algorithm.EXPECTED_PARTITION_SIZES = EXPECTED_PARTITION_SIZES
    algorithm.EXPECTED_SOURCE_COUNT = EXPECTED_SOURCE_COUNT
    algorithm.EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE = (
        EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE
    )
    algorithm.EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT = (
        EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT
    )
    algorithm.EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT = (
        EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT
    )
    algorithm.EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT = (
        EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT
    )
    algorithm.EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT = (
        EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT
    )
    algorithm.EXPECTED_MISSING_CORRECT_SOURCE_IID = None
    algorithm.EXPECTED_SOURCE_MANIFEST_SHA256 = EXPECTED_SOURCE_MANIFEST_SHA256
    algorithm.EXPECTED_EVALUATOR_SPEC_SHA256 = EXPECTED_EVALUATOR_SPEC_SHA256
    algorithm.EXPECTED_VISUAL_SCORER_SHA256 = EXPECTED_VISUAL_SCORER_SHA256
    algorithm.EXPECTED_VISUAL_CONTRACT_SHA256 = EXPECTED_VISUAL_CONTRACT_SHA256
    algorithm.NEGATIVE_SOURCE_POLICY = NEGATIVE_SOURCE_POLICY
    algorithm.LEGACY_CYCLIC_POLICY = LEGACY_CYCLIC_POLICY
    algorithm.FROZEN_CYCLIC_SOURCE_SHA256 = CYCLIC_SOURCE_SHA256
    algorithm.FROZEN_CYCLIC_AGGREGATE_SCHEMA = FROZEN_CYCLIC_AGGREGATE_SCHEMA
    algorithm.FROZEN_CYCLIC_AUTHORITY_CLOSURE = FROZEN_CYCLIC_AUTHORITY_CLOSURE
    algorithm.EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256 = (
        EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256
    )
    algorithm.AUTHORITY_CLOSURE = AUTHORITY_CLOSURE
    algorithm.OPERATIONAL_LIMITATION = OPERATIONAL_LIMITATION


def negative_design(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return algorithm.negative_design(sources)


def _source_closure(
    source_manifest_path: str | Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if str(Path(source_manifest_path)) != EXPECTED_SOURCE_MANIFEST_PATH:
        raise AllThreeNegativeRawError("r8 source-manifest lexical path differs")
    sources, evidence, design = _base_source_closure(
        source_manifest_path, expected_sha256,
    )
    if (
        evidence.get("path") != EXPECTED_SOURCE_MANIFEST_PATH
        or evidence.get("raw_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or evidence.get("content_sha256")
        != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or evidence.get("validator_summary_sha256")
        != EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256
        or evidence.get("bound_files_verified") is not True
        or evidence.get("negative_source_policy") != NEGATIVE_SOURCE_POLICY
    ):
        raise AllThreeNegativeRawError("r8 source manifest/content/validator closure differs")
    return sources, evidence, design


def _cyclic_input_and_terminal() -> tuple[dict[str, Any], str, dict[str, Any]]:
    cyclic._install_specialization()
    try:
        terminal = cyclic._validate_terminal_evidence(
            EXPECTED_TERMINAL_EVIDENCE_PATH
        )
        manifest, raw_sha = cyclic.load_input_manifest(
            EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_PATH,
            expected_sha256=EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256,
            expected_source_sha256=CYCLIC_SOURCE_SHA256,
        )
        if manifest.get("receipt_digest") != EXPECTED_LEGACY_CYCLIC_INPUT_RECEIPT_DIGEST:
            raise AllThreeNegativeRawError("frozen cyclic input digest differs")
        return manifest, raw_sha, terminal
    finally:
        _install_specialization()


def _validate_cyclic_shards(
    aggregate: Mapping[str, Any], manifest: Mapping[str, Any],
) -> None:
    shards = aggregate.get("shards")
    if not isinstance(shards, list) or len(shards) != EXPECTED_WORLD_SIZE:
        raise AllThreeNegativeRawError("frozen cyclic shard inventory differs")
    by_index: dict[int, Mapping[str, Any]] = {}
    for rank, evidence in enumerate(shards):
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "rank", "path", "sha256", "receipt_digest",
        }:
            raise AllThreeNegativeRawError("frozen cyclic shard evidence differs")
        expected_path = str(
            Path(EXPECTED_LEGACY_CYCLIC_AGGREGATE_PATH).with_name(
                f"shard-{rank:02d}-of-{EXPECTED_WORLD_SIZE:02d}.json"
            )
        )
        if evidence.get("rank") != rank or evidence.get("path") != expected_path:
            raise AllThreeNegativeRawError("frozen cyclic shard path/rank differs")
        shard, raw_sha = core._strict_json(
            expected_path,
            expected_sha256=evidence.get("sha256"),
            label=f"frozen cyclic shard {rank}",
        )
        unsigned = dict(shard)
        declared = core._sha256(
            unsigned.pop("receipt_digest", None), label="frozen cyclic shard digest",
        )
        indices = cyclic.partition_indices(EXPECTED_ATTEMPT_COUNT, rank, EXPECTED_WORLD_SIZE)
        results = shard.get("candidate_results")
        if (
            raw_sha != evidence.get("sha256")
            or declared != evidence.get("receipt_digest")
            or declared != core.object_sha256(unsigned)
            or set(shard) != {
                "schema_version", "diagnostic_source_sha256",
                "input_manifest_sha256", "rank", "world_size",
                "partition_indices", "candidate_count", "candidate_results",
                "visual_evaluator", "authority", "receipt_digest",
            }
            or shard.get("schema_version") != cyclic.SHARD_SCHEMA
            or shard.get("diagnostic_source_sha256") != CYCLIC_SOURCE_SHA256
            or shard.get("input_manifest_sha256")
            != EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256
            or shard.get("rank") != rank
            or shard.get("world_size") != EXPECTED_WORLD_SIZE
            or shard.get("partition_indices") != list(indices)
            or shard.get("candidate_count") != len(indices)
            or shard.get("authority") != FROZEN_CYCLIC_AUTHORITY_CLOSURE
            or not isinstance(results, list)
            or len(results) != len(indices)
        ):
            raise AllThreeNegativeRawError(f"frozen cyclic shard {rank} contract differs")
        cyclic._validate_visual_evaluator(shard.get("visual_evaluator"), rank=rank)
        for index, result in zip(indices, results):
            cyclic._validate_candidate_result(
                result, expected=manifest["attempts"][index],
            )
            if index in by_index:
                raise AllThreeNegativeRawError("frozen cyclic shards overlap")
            by_index[index] = result
    ordered = [by_index[index] for index in range(EXPECTED_ATTEMPT_COUNT)]
    if ordered != aggregate.get("candidate_results"):
        raise AllThreeNegativeRawError("frozen cyclic shard/aggregate results differ")


def _legacy_aggregate(
    path_value: str | Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        str(Path(path_value)) != EXPECTED_LEGACY_CYCLIC_AGGREGATE_PATH
        or expected_sha256 != EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256
    ):
        raise AllThreeNegativeRawError(
            "caller legacy cyclic aggregate path/SHA-256 differs from hard pin"
        )
    cyclic._install_specialization()
    try:
        terminal = cyclic._validate_terminal_evidence(
            EXPECTED_TERMINAL_EVIDENCE_PATH
        )
        manifest, manifest_sha = cyclic.load_input_manifest(
            EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_PATH,
            expected_sha256=EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256,
            expected_source_sha256=CYCLIC_SOURCE_SHA256,
        )
        value, raw_sha = core._strict_json(
            path_value,
            expected_sha256=EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256,
            label="required frozen r8 cyclic aggregate",
        )
        unsigned = dict(value)
        declared = core._sha256(
            unsigned.pop("receipt_digest", None), label="cyclic aggregate digest",
        )
        results = value.get("candidate_results")
        visual = cyclic._validate_visual_evaluator(
            value.get("visual_evaluator_evidence_projection"), rank=0,
        )
        rank_visual = value.get("per_rank_visual_evaluator_projection_receipts")
        if (
            set(value) != {
                "schema_version", "diagnostic_source_sha256",
                "input_manifest_sha256", "world_size", "candidate_count",
                "coverage", "candidate_order", "shards", "candidate_results",
                "visual_evaluator_evidence_projection",
                "visual_evaluator_evidence_projection_sha256",
                "per_rank_visual_evaluator_projection_receipts",
                "all8_visual_evaluator_projections_identical", "interpretation",
                "authority", "receipt_digest",
            }
            or raw_sha != EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256
            or declared != EXPECTED_LEGACY_CYCLIC_AGGREGATE_RECEIPT_DIGEST
            or declared != core.object_sha256(unsigned)
            or value.get("schema_version") != FROZEN_CYCLIC_AGGREGATE_SCHEMA
            or value.get("diagnostic_source_sha256") != CYCLIC_SOURCE_SHA256
            or value.get("input_manifest_sha256")
            != EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256
            or manifest_sha != EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256
            or value.get("world_size") != EXPECTED_WORLD_SIZE
            or value.get("candidate_count") != EXPECTED_ATTEMPT_COUNT
            or value.get("coverage")
            != "exactly_once_complete_r8_exact60_source_bound_raw"
            or value.get("candidate_order")
            != [row["candidate_id"] for row in manifest["attempts"]]
            or value.get("authority") != FROZEN_CYCLIC_AUTHORITY_CLOSURE
            or not isinstance(results, list)
            or len(results) != EXPECTED_ATTEMPT_COUNT
            or value.get("visual_evaluator_evidence_projection_sha256")
            != EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256
            or core.object_sha256(visual)
            != EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256
            or value.get("all8_visual_evaluator_projections_identical") is not True
            or not isinstance(rank_visual, list)
            or rank_visual != [
                {
                    "rank": rank,
                    "visual_evaluator_projection_sha256":
                        EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256,
                }
                for rank in range(EXPECTED_WORLD_SIZE)
            ]
        ):
            raise AllThreeNegativeRawError("required frozen cyclic aggregate contract differs")
        for result, expected in zip(results, manifest["attempts"]):
            cyclic._validate_candidate_result(result, expected=expected)
        _validate_cyclic_shards(value, manifest)
        evidence = {
            "path": EXPECTED_LEGACY_CYCLIC_AGGREGATE_PATH,
            "raw_sha256": raw_sha,
            "receipt_digest": declared,
            "schema_version": FROZEN_CYCLIC_AGGREGATE_SCHEMA,
            "diagnostic_source_sha256": CYCLIC_SOURCE_SHA256,
            "input_manifest_path": EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_PATH,
            "input_manifest_sha256": EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256,
            "input_manifest_receipt_digest":
                EXPECTED_LEGACY_CYCLIC_INPUT_RECEIPT_DIGEST,
            "candidate_count": EXPECTED_ATTEMPT_COUNT,
            "visual_evaluator_projection_sha256":
                EXPECTED_VISUAL_EVALUATOR_OBJECT_SHA256,
            "terminal_evidence": terminal,
            "all8_shards_and_all60_candidate_results_deep_validated": True,
            "required_for_aggregate_regression": True,
        }
        return dict(value), evidence
    finally:
        _install_specialization()


def build_manifest(args: Any) -> int:
    cyclic._install_specialization()
    try:
        cyclic._validate_fixed_build_inputs(args)
    finally:
        _install_specialization()
    if (
        args.expected_root_spec_sha256 != EXPECTED_ROOT_SPEC_SHA256
        or str(Path(args.attempts_root)) != EXPECTED_ATTEMPTS_ROOT
        or str(Path(args.source_manifest)) != EXPECTED_SOURCE_MANIFEST_PATH
        or args.expected_source_manifest_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise AllThreeNegativeRawError("r8 fixed build inputs differ")
    source_sha = core._verify_self(args.expected_source_sha256)
    attempts_root = core._plain_directory(args.attempts_root, label="r8 attempts root")
    output_root = Path(args.output_root)
    if (
        not output_root.is_absolute()
        or output_root == Path("/")
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise AllThreeNegativeRawError("output root must be fresh, absolute, and non-root")
    sources, source_evidence, design = _source_closure(
        args.source_manifest, args.expected_source_manifest_sha256,
    )
    _, legacy_evidence = _legacy_aggregate(
        args.legacy_cyclic_aggregate,
        args.expected_legacy_cyclic_aggregate_sha256,
    )
    paths = sorted(
        attempts_root.rglob(core.ATTEMPT_BASENAME),
        key=lambda item: item.as_posix(),
    )
    if len(paths) != EXPECTED_ATTEMPT_COUNT:
        raise AllThreeNegativeRawError("r8 bank is not exact60")
    rows = []
    for path in paths:
        row = core.validate_attempt_receipt(
            path, expected_root_spec_sha256=args.expected_root_spec_sha256,
        )
        receipt, _ = core._strict_json(
            path, expected_sha256=row["receipt_sha256"],
            label="r8 generation receipt",
        )
        candidate = receipt.get("candidate")
        iid = candidate.get("iid") if isinstance(candidate, Mapping) else None
        correct = sources.get(iid)
        if (
            correct is None
            or candidate.get("source_media_sha256_for_nonuse_audit")
            != correct["source_video_sha256"]
            or candidate.get("actor_family") != correct["actor_family"]
            or candidate.get("analysis_split") != correct["analysis_split"]
            or candidate.get("row_id") != correct["row_id"]
        ):
            raise AllThreeNegativeRawError("r8 candidate/source identity differs")
        negative_sources = [
            sources[negative_iid]
            for negative_iid in design["negative_iids_by_correct_iid"][iid]
        ]
        rows.append({
            **row,
            "correct_source": correct,
            "negative_sources": negative_sources,
            "legacy_cyclic_negative_iid":
                design["legacy_cyclic_negative_iid_by_correct_iid"][iid],
        })
    rows.sort(key=lambda row: row["candidate_id"])
    if (
        len({row["candidate_id"] for row in rows}) != EXPECTED_ATTEMPT_COUNT
        or sum(len(row["negative_sources"]) for row in rows)
        != EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT
    ):
        raise AllThreeNegativeRawError("r8 all-three candidate/pair closure differs")
    output_root.mkdir(mode=0o700)
    unsigned = {
        "schema_version": INPUT_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "attempts_root": str(attempts_root),
        "root_spec_raw_sha256": EXPECTED_ROOT_SPEC_SHA256,
        "attempt_count": EXPECTED_ATTEMPT_COUNT,
        "world_size": EXPECTED_WORLD_SIZE,
        "partition_rule": "candidate_order_index_modulo_world_size",
        "selected_frame_indices": list(core.EVAL_FRAME_INDICES),
        "source_manifest": source_evidence,
        "negative_design": design,
        "legacy_cyclic_regression": legacy_evidence,
        "attempts": rows,
        "operational_limitation": dict(OPERATIONAL_LIMITATION),
        "authority": dict(AUTHORITY_CLOSURE),
    }
    core._write_create_only(
        output_root / "input-manifest.json",
        {**unsigned, "receipt_digest": core.object_sha256(unsigned)},
    )
    return 0


def _collect_source_bindings(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for row in attempts:
        negatives = row.get("negative_sources")
        if not isinstance(negatives, list):
            raise AllThreeNegativeRawError("candidate negative-source list differs")
        for binding in [row.get("correct_source"), *negatives]:
            if not isinstance(binding, Mapping) or not isinstance(binding.get("iid"), str):
                raise AllThreeNegativeRawError("source binding differs")
            iid = binding["iid"]
            normalized = dict(binding)
            if iid in sources and sources[iid] != normalized:
                raise AllThreeNegativeRawError(
                    "source binding is inconsistent across candidates"
                )
            sources[iid] = normalized
    return sources


def load_input_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_source_sha256: str,
) -> tuple[dict[str, Any], str]:
    cyclic_manifest, _, _ = _cyclic_input_and_terminal()
    value, raw_sha = core._strict_json(
        path, expected_sha256=expected_sha256, label="r8 all-three input manifest",
    )
    core._closed(value, {
        "schema_version", "diagnostic_source_sha256", "attempts_root",
        "root_spec_raw_sha256", "attempt_count", "world_size",
        "partition_rule", "selected_frame_indices", "source_manifest",
        "negative_design", "legacy_cyclic_regression", "attempts",
        "operational_limitation", "authority", "receipt_digest",
    }, label="r8 all-three input manifest")
    unsigned = dict(value)
    declared = core._sha256(
        unsigned.pop("receipt_digest", None), label="r8 all-three input digest",
    )
    source = value.get("source_manifest")
    attempts = value.get("attempts")
    design = value.get("negative_design")
    legacy = value.get("legacy_cyclic_regression")
    if (
        value.get("schema_version") != INPUT_SCHEMA
        or value.get("diagnostic_source_sha256") != expected_source_sha256
        or value.get("attempts_root") != EXPECTED_ATTEMPTS_ROOT
        or value.get("root_spec_raw_sha256") != EXPECTED_ROOT_SPEC_SHA256
        or value.get("attempt_count") != EXPECTED_ATTEMPT_COUNT
        or value.get("world_size") != EXPECTED_WORLD_SIZE
        or value.get("partition_rule")
        != "candidate_order_index_modulo_world_size"
        or value.get("selected_frame_indices") != list(core.EVAL_FRAME_INDICES)
        or value.get("operational_limitation") != OPERATIONAL_LIMITATION
        or value.get("authority") != AUTHORITY_CLOSURE
        or not isinstance(source, Mapping)
        or source.get("path") != EXPECTED_SOURCE_MANIFEST_PATH
        or source.get("raw_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or source.get("content_sha256") != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or source.get("validator_summary_sha256")
        != EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256
        or source.get("bound_files_verified") is not True
        or source.get("negative_source_policy") != NEGATIVE_SOURCE_POLICY
        or not isinstance(design, Mapping)
        or not isinstance(legacy, Mapping)
        or legacy != _legacy_aggregate(
            EXPECTED_LEGACY_CYCLIC_AGGREGATE_PATH,
            EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256,
        )[1]
        or not isinstance(attempts, list)
        or len(attempts) != EXPECTED_ATTEMPT_COUNT
        or declared != core.object_sha256(unsigned)
    ):
        raise AllThreeNegativeRawError("r8 all-three fixed input identity differs")
    candidate_ids = [
        row.get("candidate_id") for row in attempts if isinstance(row, Mapping)
    ]
    if (
        len(candidate_ids) != EXPECTED_ATTEMPT_COUNT
        or len(set(candidate_ids)) != EXPECTED_ATTEMPT_COUNT
        or candidate_ids != sorted(candidate_ids)
    ):
        raise AllThreeNegativeRawError("r8 all-three candidate order differs")
    sources = _collect_source_bindings(attempts)
    stored_order = design.get("source_manifest_order")
    if (
        not isinstance(stored_order, list)
        or len(stored_order) != EXPECTED_SOURCE_COUNT
        or len(set(stored_order)) != EXPECTED_SOURCE_COUNT
        or set(stored_order) != set(sources)
    ):
        raise AllThreeNegativeRawError("r8 all-three stored source order differs")
    recomputed = negative_design({iid: sources[iid] for iid in stored_order})
    if (
        dict(design) != recomputed
        or source.get("source_manifest_order") != stored_order
        or source.get("negative_registration_sha256")
        != core.object_sha256(recomputed)
    ):
        raise AllThreeNegativeRawError("r8 all-three negative design differs")
    legacy_attempts = cyclic_manifest["attempts"]
    observed_pairs: set[tuple[str, str]] = set()
    observed_correct_iids: set[str] = set()
    pair_count = 0
    for row, legacy in zip(attempts, legacy_attempts):
        expected_binding = dict(legacy)
        wrong = expected_binding.pop("wrong_source")
        correct = expected_binding["correct_source"]
        negatives = row.get("negative_sources")
        if (
            row.get("candidate_id") != legacy.get("candidate_id")
            or {key: row.get(key) for key in expected_binding} != expected_binding
            or not isinstance(negatives, list)
            or len(negatives) != EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE
            or row.get("legacy_cyclic_negative_iid") != wrong.get("iid")
            or sum(item.get("iid") == wrong.get("iid") for item in negatives) != 1
            or row.get("correct_source") != correct
        ):
            raise AllThreeNegativeRawError("r8 all-three/cyclic candidate binding differs")
        iid = correct["iid"]
        expected_negative_iids = recomputed["negative_iids_by_correct_iid"][iid]
        if [item.get("iid") for item in negatives] != expected_negative_iids:
            raise AllThreeNegativeRawError("r8 all-three negative order differs")
        observed_correct_iids.add(iid)
        pair_count += len(negatives)
        observed_pairs.update((iid, item["iid"]) for item in negatives)
    expected_pairs = {
        (iid, negative_iid)
        for iid in observed_correct_iids
        for negative_iid in recomputed["negative_iids_by_correct_iid"][iid]
    }
    if (
        pair_count != EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT
        or len(observed_correct_iids) != EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT
        or set(stored_order) - observed_correct_iids
        != EXPECTED_MISSING_CORRECT_SOURCE_IIDS
        or len(observed_pairs) != EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT
        or observed_pairs != expected_pairs
        or recomputed.get("registered_directed_source_pair_count")
        != EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT
    ):
        raise AllThreeNegativeRawError(
            "r8 input 180/executed24/registered24/missing0 closure differs"
        )
    return value, raw_sha


def _worker_common(args: Any) -> tuple[Any, ...]:
    if (
        str(Path(args.visual_checkpoint)) != cyclic.EXPECTED_CHECKPOINT_ROOT
        or str(Path(args.visual_checkpoint_manifest))
        != cyclic.EXPECTED_CHECKPOINT_MANIFEST_PATH
        or str(Path(args.evaluator_spec)) != cyclic.EXPECTED_EVALUATOR_SPEC_PATH
        or str(Path(args.visual_scorer_source)) != cyclic.EXPECTED_VISUAL_SCORER_PATH
        or str(Path(args.visual_contract_source))
        != cyclic.EXPECTED_VISUAL_CONTRACT_PATH
        or args.expected_evaluator_spec_sha256 != EXPECTED_EVALUATOR_SPEC_SHA256
        or args.expected_visual_scorer_sha256 != EXPECTED_VISUAL_SCORER_SHA256
        or args.expected_visual_contract_sha256 != EXPECTED_VISUAL_CONTRACT_SHA256
        or core.file_sha256(args.visual_checkpoint_manifest)
        != cyclic.EXPECTED_CHECKPOINT_MANIFEST_SHA256
    ):
        raise AllThreeNegativeRawError("r8 frozen visual evaluator path/SHA differs")
    result = _base_worker_common(args)
    cyclic._validate_visual_evaluator(result[4], rank=-1)
    return result


preflight = _base_preflight
worker = _base_worker


def _cache_hash_map(
    summary: Any,
    *,
    expected_order: Sequence[str],
    expected_sources: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    hash_map = _base_cache_hash_map(
        summary,
        expected_order=expected_order,
        expected_sources=expected_sources,
    )
    entries = summary.get("entries") if isinstance(summary, Mapping) else None
    if not isinstance(entries, list):
        raise AllThreeNegativeRawError("rank source cache entries differ")
    for row in entries:
        iid = row.get("iid") if isinstance(row, Mapping) else None
        expected = expected_sources.get(iid) if isinstance(iid, str) else None
        geometry = row.get("feature_geometry") if isinstance(row, Mapping) else None
        if (
            not isinstance(expected, Mapping)
            or set(row) != {
                "iid", "actor_family", "source_video_sha256",
                "global_feature_sha256", "dense_feature_sha256", "decode",
                "feature_geometry",
            }
            or not isinstance(geometry, Mapping)
            or set(geometry) != {
                "selected_frame_count", "dense_grid_height", "dense_grid_width",
                "feature_dimension",
            }
            or dict(geometry) != {
                "selected_frame_count": 17,
                "dense_grid_height": 16,
                "dense_grid_width": 16,
                "feature_dimension": 768,
            }
        ):
            raise AllThreeNegativeRawError("rank source cache deep closure differs")
        cyclic._validate_decode_evidence(
            row.get("decode"),
            expected_artifact_sha256=expected["source_video_sha256"],
            label=f"cached source {iid}",
        )
    return hash_map


def _validate_candidate_result(
    result: Any,
    *,
    expected: Mapping[str, Any],
) -> set[tuple[str, str]]:
    if not isinstance(result, Mapping) or set(result) != {
        "candidate_id", "candidate_binding", "candidate_decode",
        "candidate_features", "source_features_served_from_exact8_rank_cache",
        "negative_results", "candidate_descriptive_summary",
        "operational_limitation", "authority",
    }:
        raise AllThreeNegativeRawError("all-three candidate result field closure differs")
    cyclic._validate_decode_evidence(
        result.get("candidate_decode"),
        expected_artifact_sha256=expected.get("mp4_sha256"),
        label="all-three candidate",
    )
    cyclic._validate_feature_evidence(
        result.get("candidate_features"), label="all-three candidate",
    )
    negatives = result.get("negative_results")
    if not isinstance(negatives, list):
        raise AllThreeNegativeRawError("all-three negative result list differs")
    for row in negatives:
        if not isinstance(row, Mapping) or set(row) != {
            "negative_ordinal_in_manifest_order", "correct_source_iid",
            "correct_source_video_sha256", "negative_source_iid",
            "negative_source_video_sha256", "is_legacy_cyclic_negative",
            "raw_metrics", "authority",
        }:
            raise AllThreeNegativeRawError("all-three negative result field closure differs")
    return _base_validate_candidate_result(result, expected=expected)


def _verify_legacy_regression(
    ordered: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    legacy_path: str | Path,
    expected_legacy_sha256: str,
) -> dict[str, Any]:
    legacy, evidence = _legacy_aggregate(legacy_path, expected_legacy_sha256)
    if evidence != manifest.get("legacy_cyclic_regression"):
        raise AllThreeNegativeRawError("aggregate cyclic evidence differs from preregistration")
    legacy_by_id = {row["candidate_id"]: row for row in legacy["candidate_results"]}
    matched_fields = 0
    projection_rows = []
    for result in ordered:
        old = legacy_by_id.get(result["candidate_id"])
        old_binding = old.get("candidate_binding") if isinstance(old, Mapping) else None
        new_binding = result["candidate_binding"]
        pairs = [
            row for row in result["negative_results"]
            if row.get("is_legacy_cyclic_negative") is True
        ]
        if (
            not isinstance(old_binding, Mapping)
            or old_binding.get("candidate_id") != new_binding["candidate_id"]
            or old_binding.get("receipt_sha256") != new_binding["receipt_sha256"]
            or old_binding.get("mp4_sha256") != new_binding["mp4_sha256"]
            or old_binding.get("correct_source") != new_binding["correct_source"]
            or old_binding.get("wrong_source", {}).get("iid")
            != new_binding["legacy_cyclic_negative_iid"]
            or len(pairs) != 1
        ):
            raise AllThreeNegativeRawError("cyclic regression candidate binding differs")
        old_metrics = old.get("raw_metrics")
        if not isinstance(old_metrics, Mapping):
            raise AllThreeNegativeRawError("cyclic regression raw metrics are absent")
        projected = {key: pairs[0]["raw_metrics"].get(key) for key in old_metrics}
        if projected != old_metrics:
            raise AllThreeNegativeRawError(
                f"cyclic raw-metric regression differs for {result['candidate_id']}"
            )
        matched_fields += len(old_metrics)
        projection_rows.append({
            "candidate_id": result["candidate_id"],
            "legacy_negative_iid": new_binding["legacy_cyclic_negative_iid"],
            "raw_metrics": projected,
        })
    return {
        **evidence,
        "comparison_mode": "exact_json_field_value_equality",
        "candidate_match_count": EXPECTED_ATTEMPT_COUNT,
        "raw_metric_field_value_match_count": matched_fields,
        "legacy_cyclic_projection_sha256": core.object_sha256(projection_rows),
        "all_60_legacy_cyclic_raw_metrics_exact": True,
    }


def aggregate(args: Any) -> int:
    source_sha = core._verify_self(args.expected_source_sha256)
    manifest, manifest_sha = load_input_manifest(
        args.input_manifest,
        expected_sha256=args.expected_input_manifest_sha256,
        expected_source_sha256=source_sha,
    )
    output_root = core._plain_directory(args.output_root, label="output root")
    expected_order = manifest["negative_design"]["source_manifest_order"]
    expected_sources = _collect_source_bindings(manifest["attempts"])
    shards: list[dict[str, Any]] = []
    by_index: dict[int, Mapping[str, Any]] = {}
    reference_hash_map: list[dict[str, str]] | None = None
    reference_visual: dict[str, Any] | None = None
    cache_receipts, visual_receipts = [], []
    for rank in range(EXPECTED_WORLD_SIZE):
        path = output_root / f"shard-{rank:02d}-of-{EXPECTED_WORLD_SIZE:02d}.json"
        value, raw_sha = core._strict_json(path, expected_sha256=None, label=f"shard {rank}")
        unsigned = dict(value)
        declared = core._sha256(unsigned.pop("receipt_digest", None), label="shard digest")
        indices = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, EXPECTED_WORLD_SIZE)
        results = value.get("candidate_results")
        if (
            set(value) != {
                "schema_version", "diagnostic_source_sha256",
                "input_manifest_sha256", "rank", "world_size",
                "partition_indices", "candidate_count",
                "candidate_negative_pair_count", "exact8_source_feature_cache",
                "candidate_results", "visual_evaluator", "operational_limitation",
                "authority", "receipt_digest",
            }
            or value.get("schema_version") != SHARD_SCHEMA
            or value.get("diagnostic_source_sha256") != source_sha
            or value.get("input_manifest_sha256") != manifest_sha
            or value.get("rank") != rank
            or value.get("world_size") != EXPECTED_WORLD_SIZE
            or value.get("partition_indices") != list(indices)
            or value.get("candidate_count") != len(indices)
            or value.get("candidate_negative_pair_count")
            != len(indices) * EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE
            or not isinstance(results, list)
            or len(results) != len(indices)
            or value.get("operational_limitation") != OPERATIONAL_LIMITATION
            or value.get("authority") != AUTHORITY_CLOSURE
            or declared != core.object_sha256(unsigned)
        ):
            raise AllThreeNegativeRawError(f"all-three shard {rank} contract differs")
        hash_map = _cache_hash_map(
            value.get("exact8_source_feature_cache"),
            expected_order=expected_order,
            expected_sources=expected_sources,
        )
        if reference_hash_map is None:
            reference_hash_map = hash_map
        elif hash_map != reference_hash_map:
            raise AllThreeNegativeRawError("source feature hashes differ across ranks")
        visual = cyclic._validate_visual_evaluator(value.get("visual_evaluator"), rank=rank)
        if reference_visual is None:
            reference_visual = visual
        else:
            cyclic._require_identical_visual_evaluator(reference_visual, visual, rank=rank)
        cache_receipts.append({"rank": rank, "feature_hash_map_sha256": core.object_sha256(hash_map)})
        visual_receipts.append({
            "rank": rank,
            "visual_evaluator_projection_sha256": core.object_sha256(visual),
        })
        shards.append({
            "rank": rank, "path": str(path.resolve(strict=True)),
            "sha256": raw_sha, "receipt_digest": declared,
        })
        for index, result in zip(indices, results):
            if index in by_index:
                raise AllThreeNegativeRawError("all-three shard partition overlaps")
            _validate_candidate_result(result, expected=manifest["attempts"][index])
            by_index[index] = result
    if set(by_index) != set(range(EXPECTED_ATTEMPT_COUNT)):
        raise AllThreeNegativeRawError("shards do not cover r8 exact60")
    ordered = [by_index[index] for index in range(EXPECTED_ATTEMPT_COUNT)]
    expected_ids = [row["candidate_id"] for row in manifest["attempts"]]
    if [row.get("candidate_id") for row in ordered] != expected_ids:
        raise AllThreeNegativeRawError("aggregate candidate order differs")
    observed_pairs: set[tuple[str, str]] = set()
    observed_correct_iids: set[str] = set()
    pair_count = 0
    for result, expected in zip(ordered, manifest["attempts"]):
        observed_pairs.update(_validate_candidate_result(result, expected=expected))
        observed_correct_iids.add(expected["correct_source"]["iid"])
        pair_count += len(result["negative_results"])
    missing = set(expected_order) - observed_correct_iids
    expected_pairs = {
        (iid, negative_iid)
        for iid in observed_correct_iids
        for negative_iid in manifest["negative_design"]["negative_iids_by_correct_iid"][iid]
    }
    if (
        pair_count != EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT
        or len(observed_correct_iids) != EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT
        or missing != EXPECTED_MISSING_CORRECT_SOURCE_IIDS
        or len(observed_pairs) != EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT
        or observed_pairs != expected_pairs
        or manifest["negative_design"].get("registered_directed_source_pair_count")
        != EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT
        or reference_visual is None
        or reference_hash_map is None
    ):
        raise AllThreeNegativeRawError(
            "aggregate 180/executed24/registered24/missing0 closure differs"
        )
    legacy = _verify_legacy_regression(
        ordered,
        manifest=manifest,
        legacy_path=args.legacy_cyclic_aggregate,
        expected_legacy_sha256=args.expected_legacy_cyclic_aggregate_sha256,
    )
    pair_rows = [
        {"correct_source_iid": correct, "negative_source_iid": negative}
        for correct, negative in sorted(observed_pairs)
    ]
    unsigned = {
        "schema_version": AGGREGATE_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "world_size": EXPECTED_WORLD_SIZE,
        "candidate_count": EXPECTED_ATTEMPT_COUNT,
        "candidate_negative_pair_count": EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT,
        "executed_correct_source_count": EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT,
        "executed_correct_source_iids": [
            iid for iid in expected_order if iid in observed_correct_iids
        ],
        "missing_correct_source_iids": [],
        "executed_directed_source_pair_count":
            EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT,
        "executed_directed_source_pairs": pair_rows,
        "executed_directed_source_pairs_sha256": core.object_sha256(pair_rows),
        "registered_directed_source_pair_universe_count":
            EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT,
        "registered_directed_source_pair_universe_sha256": core.object_sha256(
            manifest["negative_design"]["directed_source_pairs"]
        ),
        "coverage": "exactly_once_complete_r8_exact60_same_actor_all_three_negatives_raw",
        "candidate_order": expected_ids,
        "shards": shards,
        "cross_rank_source_feature_cache_consistency": {
            "source_count": EXPECTED_SOURCE_COUNT,
            "rank_count": EXPECTED_WORLD_SIZE,
            "per_source_feature_hashes": reference_hash_map,
            "per_rank_feature_hash_map_receipts": cache_receipts,
            "all_exact8_source_feature_hashes_identical_across_all8_ranks": True,
        },
        "visual_evaluator_evidence_projection": reference_visual,
        "visual_evaluator_evidence_projection_sha256": core.object_sha256(reference_visual),
        "per_rank_visual_evaluator_projection_receipts": visual_receipts,
        "all8_visual_evaluator_projections_identical": True,
        "legacy_cyclic_raw_metric_regression": legacy,
        "candidate_results": ordered,
        "interpretation": {
            "measurement": "raw frozen-DINO candidate/correct/all-three-same-actor-negative source proxies",
            "negative_map_preregistered_from_actor_and_manifest_order_only": True,
            "all8_registered_sources_observed_as_correct_sources": True,
            "executed_unique_directed_source_pairs": 24,
            "candidate_worst_and_median_are_descriptive_only": True,
            "thresholds": None,
            "no_absolute_preservation_claim": True,
            "no_event_measurement": True,
            "no_ranking_selection_or_training_admission": True,
            "operational_only_due_to_missing_full_lifetime_exact8_source_retained_fds": True,
        },
        "operational_limitation": dict(OPERATIONAL_LIMITATION),
        "authority": dict(AUTHORITY_CLOSURE),
    }
    core._write_create_only(
        output_root / "aggregate-receipt.json",
        {**unsigned, "receipt_digest": core.object_sha256(unsigned)},
    )
    return 0


def _visual_args(parser: Any) -> None:
    core._add_visual_arguments(parser)


def _legacy_args(parser: Any) -> None:
    parser.add_argument("--legacy-cyclic-aggregate", required=True)
    parser.add_argument("--expected-legacy-cyclic-aggregate-sha256", required=True)


def build_parser() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--attempts-root", required=True)
    build.add_argument("--expected-root-spec-sha256", required=True)
    build.add_argument("--source-manifest", required=True)
    build.add_argument("--expected-source-manifest-sha256", required=True)
    build.add_argument("--terminal-evidence", required=True)
    build.add_argument("--expected-source-sha256", required=True)
    _legacy_args(build)
    build.add_argument("--output-root", required=True)
    check = commands.add_parser("preflight")
    _visual_args(check)
    run = commands.add_parser("worker")
    _visual_args(run)
    run.add_argument("--world-size", required=True, type=int)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--input-manifest", required=True)
    combine.add_argument("--expected-input-manifest-sha256", required=True)
    combine.add_argument("--expected-source-sha256", required=True)
    _legacy_args(combine)
    combine.add_argument("--output-root", required=True)
    return parser


def _install_specialization() -> None:
    _configure_core()
    algorithm._source_closure = _source_closure
    algorithm._legacy_aggregate = _legacy_aggregate
    algorithm.build_manifest = build_manifest
    algorithm.load_input_manifest = load_input_manifest
    algorithm._worker_common = _worker_common
    algorithm._cache_hash_map = _cache_hash_map
    algorithm._collect_source_bindings = _collect_source_bindings
    algorithm._validate_candidate_result = _validate_candidate_result
    algorithm._verify_legacy_regression = _verify_legacy_regression
    algorithm.aggregate = aggregate
    algorithm.preflight = preflight
    algorithm.worker = worker
    frozen.build_manifest = build_manifest
    frozen.load_input_manifest = load_input_manifest
    frozen._worker_common = _worker_common
    frozen.aggregate = aggregate
    frozen.preflight = preflight
    frozen.worker = worker


def main(argv: Sequence[str] | None = None) -> int:
    _install_specialization()
    os.umask(0o077)
    args = build_parser().parse_args(argv)
    return {
        "build-manifest": build_manifest,
        "preflight": preflight,
        "worker": worker,
        "aggregate": aggregate,
    }[args.command](args)


_install_specialization()


if __name__ == "__main__":
    raise SystemExit(main())
