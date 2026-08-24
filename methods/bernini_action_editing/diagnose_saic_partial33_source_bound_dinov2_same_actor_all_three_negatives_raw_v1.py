#!/usr/bin/env python3
"""Operational r5 exact33 DINO diagnostic against all three same-actor sources."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
ALGORITHM_BASE_SOURCE_NAME = "diagnose_saic_partial47_source_bound_dinov2_raw_v1.py"
ALGORITHM_BASE_SOURCE_SHA256 = "ffbc9ba149d1ddadf704dd8258678a8893235e328da4c7601e98d63ba37aa7a2"
FROZEN_CYCLIC_SOURCE_NAME = "diagnose_saic_partial33_source_bound_dinov2_raw_v1.py"
FROZEN_CYCLIC_SOURCE_SHA256 = "3faad816ee0a4f320b83afb94377da7a437c535f60105d3fd8578b7068851600"
EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256 = "35d757481bc62f64dfea12d4924b555cc989d207d8b7a502a9d7f214ef815834"
EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256 = "2f85ca48b9338920f050bc7b6bfe222b7e324087fc6ff99ff020a9a3c600092c"
EXPECTED_LEGACY_CYCLIC_AGGREGATE_RECEIPT_DIGEST = "a586eb4776e7080a94ab225bc4d39e85324b6508310012221181dc689e301218"
_algorithm_path = METHOD_ROOT / ALGORITHM_BASE_SOURCE_NAME
if (
    not _algorithm_path.is_file()
    or _algorithm_path.is_symlink()
    or hashlib.sha256(_algorithm_path.read_bytes()).hexdigest() != ALGORITHM_BASE_SOURCE_SHA256
):
    raise RuntimeError("frozen exact47 all-three algorithm dependency differs")
_frozen_path = METHOD_ROOT / FROZEN_CYCLIC_SOURCE_NAME
if (
    not _frozen_path.is_file()
    or _frozen_path.is_symlink()
    or hashlib.sha256(_frozen_path.read_bytes()).hexdigest() != FROZEN_CYCLIC_SOURCE_SHA256
):
    raise RuntimeError("frozen r5 exact33 cyclic source-bound diagnostic differs")

import diagnose_saic_partial47_source_bound_dinov2_raw_v1 as frozen  # noqa: E402
import diagnose_saic_partial33_source_bound_dinov2_raw_v1 as cyclic_frozen  # noqa: E402


core = frozen.core
SCHEMA_VERSION = (
    "bernini-saic-r5-partial33-source-bound-dinov2-"
    "same-actor-all-three-negatives-raw-v1"
)
INPUT_SCHEMA = f"{SCHEMA_VERSION}-input"
SHARD_SCHEMA = f"{SCHEMA_VERSION}-shard"
AGGREGATE_SCHEMA = f"{SCHEMA_VERSION}-aggregate"
PREFLIGHT_SCHEMA = f"{SCHEMA_VERSION}-preflight"
EXPECTED_ATTEMPT_COUNT = 33
EXPECTED_WORLD_SIZE = 8
EXPECTED_PARTITION_SIZES = (5, 4, 4, 4, 4, 4, 4, 4)
EXPECTED_SOURCE_COUNT = 8
EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE = 3
EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT = 99
EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT = 24
EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT = 15
EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT = 5
EXPECTED_MISSING_CORRECT_SOURCE_IIDS = frozenset({
    "6ea45d35943742bb",
    "841b5e0080a1441d",
    "99cde432839f4240",
})
EXPECTED_SOURCE_MANIFEST_SHA256 = frozen.EXPECTED_SOURCE_MANIFEST_SHA256
EXPECTED_EVALUATOR_SPEC_SHA256 = frozen.EXPECTED_EVALUATOR_SPEC_SHA256
EXPECTED_VISUAL_SCORER_SHA256 = frozen.EXPECTED_VISUAL_SCORER_SHA256
EXPECTED_VISUAL_CONTRACT_SHA256 = frozen.EXPECTED_VISUAL_CONTRACT_SHA256
NEGATIVE_SOURCE_POLICY = "same_actor_family_sealed_manifest_order_all_other_three_v1"
LEGACY_CYCLIC_POLICY = frozen.WRONG_SOURCE_POLICY
FROZEN_CYCLIC_AGGREGATE_SCHEMA = cyclic_frozen.AGGREGATE_SCHEMA
FROZEN_CYCLIC_AUTHORITY_CLOSURE = dict(cyclic_frozen.AUTHORITY_CLOSURE)
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
AllThreeNegativeRawError = frozen.SourceBoundRawError
_base_partition_indices = frozen._base_partition_indices


def _configure_core() -> None:
    # Importing the frozen exact33 cyclic specialization rewrites both the
    # exact47 wrapper module and its nested partial28 implementation.  Restore
    # both layers to this executable identity before any validation or work.
    frozen.__file__ = __file__
    frozen.SCHEMA_VERSION = SCHEMA_VERSION
    frozen.INPUT_SCHEMA = INPUT_SCHEMA
    frozen.SHARD_SCHEMA = SHARD_SCHEMA
    frozen.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
    frozen.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    frozen.EXPECTED_ATTEMPT_COUNT = EXPECTED_ATTEMPT_COUNT
    frozen.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE
    frozen.AUTHORITY_CLOSURE = AUTHORITY_CLOSURE
    core.__file__ = __file__
    core.SCHEMA_VERSION = SCHEMA_VERSION
    core.INPUT_SCHEMA = INPUT_SCHEMA
    core.SHARD_SCHEMA = SHARD_SCHEMA
    core.AGGREGATE_SCHEMA = AGGREGATE_SCHEMA
    core.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    core.EXPECTED_ATTEMPT_COUNT = EXPECTED_ATTEMPT_COUNT
    core.EXPECTED_WORLD_SIZE = EXPECTED_WORLD_SIZE
    core.AUTHORITY_CLOSURE = AUTHORITY_CLOSURE


_configure_core()


def partition_indices(count: int, rank: int, world_size: int) -> tuple[int, ...]:
    indices = _base_partition_indices(count, rank, world_size)
    sizes = tuple(
        len(_base_partition_indices(count, item, world_size))
        for item in range(world_size)
    )
    if sizes != EXPECTED_PARTITION_SIZES:
        raise AllThreeNegativeRawError("r5 exact33 all-three partition sizes differ")
    return indices


def _configure_partitions() -> None:
    frozen.partition_indices = partition_indices
    core.partition_indices = partition_indices


_configure_partitions()


def negative_design(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Derive the full same-actor negative map from manifest order only."""

    manifest_order = list(sources)
    if len(manifest_order) != EXPECTED_SOURCE_COUNT or len(set(manifest_order)) != EXPECTED_SOURCE_COUNT:
        raise AllThreeNegativeRawError("source manifest order must contain exact8 unique IIDs")
    if len({sources[iid].get("source_video_sha256") for iid in manifest_order}) != EXPECTED_SOURCE_COUNT:
        raise AllThreeNegativeRawError("exact8 source videos must have distinct SHA-256 identities")
    by_actor: dict[str, list[str]] = {}
    for iid in manifest_order:
        row = sources[iid]
        if row.get("iid") != iid or row.get("actor_family") not in {"dog", "human"}:
            raise AllThreeNegativeRawError("source IID or actor-family binding differs")
        by_actor.setdefault(row["actor_family"], []).append(iid)
    if set(by_actor) != {"dog", "human"} or any(len(rows) != 4 for rows in by_actor.values()):
        raise AllThreeNegativeRawError("source actor-family closure must be four plus four")
    negative_iids: dict[str, list[str]] = {}
    directed_pairs: list[dict[str, Any]] = []
    manifest_index = {iid: index for index, iid in enumerate(manifest_order)}
    for iid in manifest_order:
        actor = sources[iid]["actor_family"]
        others = [item for item in by_actor[actor] if item != iid]
        if len(others) != EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE:
            raise AllThreeNegativeRawError("each source must have exactly three same-actor negatives")
        negative_iids[iid] = others
        for negative_iid in others:
            directed_pairs.append({
                "correct_source_iid": iid,
                "negative_source_iid": negative_iid,
                "actor_family": actor,
                "correct_manifest_index": manifest_index[iid],
                "negative_manifest_index": manifest_index[negative_iid],
            })
    pair_keys = {
        (row["correct_source_iid"], row["negative_source_iid"])
        for row in directed_pairs
    }
    if (
        len(directed_pairs) != EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT
        or len(pair_keys) != EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT
    ):
        raise AllThreeNegativeRawError("directed exact8 source-pair closure differs")
    legacy_cyclic = {}
    for rows in by_actor.values():
        ordered = sorted(rows)
        for index, iid in enumerate(ordered):
            legacy_cyclic[iid] = ordered[(index + 1) % len(ordered)]
    if any(legacy_cyclic[iid] not in negative_iids[iid] for iid in manifest_order):
        raise AllThreeNegativeRawError("legacy cyclic negative escaped all-three design")
    return {
        "policy": NEGATIVE_SOURCE_POLICY,
        "policy_inputs": ["sealed_source_manifest_actor_family", "sealed_source_manifest_row_order"],
        "candidate_metrics_consulted_during_registration": False,
        "source_manifest_order": manifest_order,
        "actor_family_manifest_order": {actor: list(rows) for actor, rows in by_actor.items()},
        "negative_iids_by_correct_iid": negative_iids,
        "directed_source_pairs": directed_pairs,
        "registered_directed_source_pair_count": EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT,
        "negative_count_per_candidate": EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE,
        "candidate_negative_pair_count": EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT,
        "legacy_cyclic_policy": LEGACY_CYCLIC_POLICY,
        "legacy_cyclic_negative_iid_by_correct_iid": legacy_cyclic,
    }


def _source_closure(
    source_manifest_path: str | Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sources, frozen_policy = frozen._source_closure(source_manifest_path, expected_sha256)
    design = negative_design(sources)
    evidence = dict(frozen_policy["evidence"])
    if evidence.pop("wrong_source_policy", None) != LEGACY_CYCLIC_POLICY:
        raise AllThreeNegativeRawError("frozen source-manifest validation evidence differs")
    evidence.update({
        "negative_source_policy": NEGATIVE_SOURCE_POLICY,
        "source_manifest_order": list(design["source_manifest_order"]),
        "negative_registration_sha256": core.object_sha256(design),
    })
    return sources, evidence, design


def _legacy_aggregate(
    path_value: str | Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if expected_sha256 != EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256:
        raise AllThreeNegativeRawError("caller legacy cyclic aggregate SHA-256 differs from hard pin")
    value, raw_sha = core._strict_json(
        path_value,
        expected_sha256=expected_sha256,
        label="required frozen cyclic aggregate",
    )
    unsigned = dict(value)
    declared = core._sha256(unsigned.pop("receipt_digest", None), label="cyclic aggregate digest")
    results = value.get("candidate_results")
    if (
        value.get("schema_version") != FROZEN_CYCLIC_AGGREGATE_SCHEMA
        or value.get("diagnostic_source_sha256") != FROZEN_CYCLIC_SOURCE_SHA256
        or value.get("input_manifest_sha256")
        != EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256
        or value.get("world_size") != EXPECTED_WORLD_SIZE
        or value.get("candidate_count") != EXPECTED_ATTEMPT_COUNT
        or value.get("coverage") != "exactly_once_complete_partial33_source_bound_raw"
        or value.get("authority") != FROZEN_CYCLIC_AUTHORITY_CLOSURE
        or not isinstance(results, list)
        or len(results) != EXPECTED_ATTEMPT_COUNT
        or len({row.get("candidate_id") for row in results if isinstance(row, Mapping)}) != EXPECTED_ATTEMPT_COUNT
        or declared != EXPECTED_LEGACY_CYCLIC_AGGREGATE_RECEIPT_DIGEST
        or declared != core.object_sha256(unsigned)
    ):
        raise AllThreeNegativeRawError("required frozen cyclic aggregate contract differs")
    path = core._plain_file(path_value, label="required frozen cyclic aggregate")
    evidence = {
        "path": str(path),
        "raw_sha256": raw_sha,
        "receipt_digest": declared,
        "schema_version": FROZEN_CYCLIC_AGGREGATE_SCHEMA,
        "diagnostic_source_sha256": FROZEN_CYCLIC_SOURCE_SHA256,
        "input_manifest_sha256": EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256,
        "candidate_count": EXPECTED_ATTEMPT_COUNT,
        "required_for_aggregate_regression": True,
    }
    return value, evidence


def build_manifest(args: Any) -> int:
    source_sha = core._verify_self(args.expected_source_sha256)
    attempts_root = core._plain_directory(args.attempts_root, label="attempts root")
    output_root = Path(args.output_root)
    if not output_root.is_absolute() or output_root == Path("/") or output_root.exists() or output_root.is_symlink():
        raise AllThreeNegativeRawError("output root must be fresh, absolute, and non-root")
    sources, source_evidence, design = _source_closure(
        args.source_manifest,
        args.expected_source_manifest_sha256,
    )
    # The negative map is sealed before any legacy candidate metric is opened.
    _, legacy_evidence = _legacy_aggregate(
        args.legacy_cyclic_aggregate,
        args.expected_legacy_cyclic_aggregate_sha256,
    )
    paths = sorted(attempts_root.rglob(core.ATTEMPT_BASENAME), key=lambda item: item.as_posix())
    if len(paths) != EXPECTED_ATTEMPT_COUNT:
        raise AllThreeNegativeRawError(f"r5 exact33 bank receipt count differs: {len(paths)}")
    rows = []
    for path in paths:
        row = core.validate_attempt_receipt(
            path,
            expected_root_spec_sha256=args.expected_root_spec_sha256,
        )
        receipt, _ = core._strict_json(
            path,
            expected_sha256=row["receipt_sha256"],
            label="generation receipt",
        )
        candidate = receipt.get("candidate")
        iid = candidate.get("iid") if isinstance(candidate, Mapping) else None
        correct = sources.get(iid)
        if correct is None or candidate.get("source_media_sha256_for_nonuse_audit") != correct["source_video_sha256"]:
            raise AllThreeNegativeRawError("candidate IID/source nonuse audit binding differs")
        if (
            candidate.get("actor_family") != correct["actor_family"]
            or candidate.get("analysis_split") != correct["analysis_split"]
            or candidate.get("row_id") != correct["row_id"]
        ):
            raise AllThreeNegativeRawError("candidate compound source identity differs")
        negative_sources = [
            sources[negative_iid]
            for negative_iid in design["negative_iids_by_correct_iid"][iid]
        ]
        rows.append({
            **row,
            "correct_source": correct,
            "negative_sources": negative_sources,
            "legacy_cyclic_negative_iid": design[
                "legacy_cyclic_negative_iid_by_correct_iid"
            ][iid],
        })
    rows.sort(key=lambda row: row["candidate_id"])
    if len({row["candidate_id"] for row in rows}) != EXPECTED_ATTEMPT_COUNT:
        raise AllThreeNegativeRawError("candidate ID closure differs")
    pair_count = sum(len(row["negative_sources"]) for row in rows)
    if pair_count != EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT:
        raise AllThreeNegativeRawError("candidate-negative pair count differs")
    output_root.mkdir(mode=0o700)
    unsigned = {
        "schema_version": INPUT_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "attempts_root": str(attempts_root),
        "root_spec_raw_sha256": core._sha256(
            args.expected_root_spec_sha256,
            label="root spec SHA-256",
        ),
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


def _collect_source_bindings(attempts: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for row in attempts:
        bindings = [row.get("correct_source"), *(row.get("negative_sources") or [])]
        for binding in bindings:
            if not isinstance(binding, Mapping) or not isinstance(binding.get("iid"), str):
                raise AllThreeNegativeRawError("source binding differs")
            iid = binding["iid"]
            normalized = dict(binding)
            if iid in sources and sources[iid] != normalized:
                raise AllThreeNegativeRawError("source binding is inconsistent across candidates")
            sources[iid] = normalized
    return sources


def load_input_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_source_sha256: str,
) -> tuple[dict[str, Any], str]:
    value, raw_sha = core._strict_json(
        path,
        expected_sha256=expected_sha256,
        label="all-three input manifest",
    )
    core._closed(value, {
        "schema_version", "diagnostic_source_sha256", "attempts_root",
        "root_spec_raw_sha256", "attempt_count", "world_size",
        "partition_rule", "selected_frame_indices", "source_manifest",
        "negative_design", "legacy_cyclic_regression", "attempts",
        "operational_limitation", "authority", "receipt_digest",
    }, label="all-three input manifest")
    unsigned = dict(value)
    declared = core._sha256(unsigned.pop("receipt_digest", None), label="input manifest digest")
    attempts = value.get("attempts")
    design = value.get("negative_design")
    source_manifest = value.get("source_manifest")
    legacy = value.get("legacy_cyclic_regression")
    if (
        value.get("schema_version") != INPUT_SCHEMA
        or value.get("diagnostic_source_sha256") != expected_source_sha256
        or value.get("attempt_count") != EXPECTED_ATTEMPT_COUNT
        or value.get("world_size") != EXPECTED_WORLD_SIZE
        or value.get("partition_rule") != "candidate_order_index_modulo_world_size"
        or value.get("selected_frame_indices") != list(core.EVAL_FRAME_INDICES)
        or value.get("operational_limitation") != OPERATIONAL_LIMITATION
        or value.get("authority") != AUTHORITY_CLOSURE
        or not isinstance(source_manifest, Mapping)
        or source_manifest.get("raw_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or source_manifest.get("bound_files_verified") is not True
        or source_manifest.get("negative_source_policy") != NEGATIVE_SOURCE_POLICY
        or not isinstance(design, Mapping)
        or not isinstance(legacy, Mapping)
        or legacy.get("raw_sha256") != EXPECTED_LEGACY_CYCLIC_AGGREGATE_SHA256
        or legacy.get("receipt_digest")
        != EXPECTED_LEGACY_CYCLIC_AGGREGATE_RECEIPT_DIGEST
        or legacy.get("diagnostic_source_sha256") != FROZEN_CYCLIC_SOURCE_SHA256
        or legacy.get("input_manifest_sha256")
        != EXPECTED_LEGACY_CYCLIC_INPUT_MANIFEST_SHA256
        or legacy.get("schema_version") != FROZEN_CYCLIC_AGGREGATE_SCHEMA
        or legacy.get("candidate_count") != EXPECTED_ATTEMPT_COUNT
        or legacy.get("required_for_aggregate_regression") is not True
        or not isinstance(attempts, list)
        or len(attempts) != EXPECTED_ATTEMPT_COUNT
        or declared != core.object_sha256(unsigned)
    ):
        raise AllThreeNegativeRawError("all-three input manifest contract differs")
    candidate_ids = [row.get("candidate_id") for row in attempts if isinstance(row, Mapping)]
    if len(candidate_ids) != EXPECTED_ATTEMPT_COUNT or len(set(candidate_ids)) != EXPECTED_ATTEMPT_COUNT or candidate_ids != sorted(candidate_ids):
        raise AllThreeNegativeRawError("all-three candidate order or uniqueness differs")
    sources = _collect_source_bindings(attempts)
    stored_order = design.get("source_manifest_order")
    if (
        not isinstance(stored_order, list)
        or len(stored_order) != EXPECTED_SOURCE_COUNT
        or len(set(stored_order)) != EXPECTED_SOURCE_COUNT
        or set(stored_order) != set(sources)
    ):
        raise AllThreeNegativeRawError("stored source manifest order differs")
    ordered_sources = {iid: sources[iid] for iid in stored_order}
    recomputed = negative_design(ordered_sources)
    if (
        dict(design) != recomputed
        or source_manifest.get("source_manifest_order") != stored_order
        or source_manifest.get("negative_registration_sha256")
        != core.object_sha256(recomputed)
    ):
        raise AllThreeNegativeRawError("preregistered all-three negative design differs")
    pair_count = 0
    observed_pairs = set()
    observed_correct_iids = set()
    for row in attempts:
        correct = row.get("correct_source")
        negatives = row.get("negative_sources")
        iid = correct.get("iid") if isinstance(correct, Mapping) else None
        expected_negative_iids = recomputed["negative_iids_by_correct_iid"].get(iid)
        if (
            not isinstance(negatives, list)
            or len(negatives) != EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE
            or [item.get("iid") for item in negatives if isinstance(item, Mapping)] != expected_negative_iids
            or row.get("legacy_cyclic_negative_iid")
            != recomputed["legacy_cyclic_negative_iid_by_correct_iid"].get(iid)
        ):
            raise AllThreeNegativeRawError("candidate all-three negative binding differs")
        pair_count += len(negatives)
        observed_correct_iids.add(iid)
        observed_pairs.update((iid, item["iid"]) for item in negatives)
    missing_correct_iids = set(stored_order) - observed_correct_iids
    expected_executed_pairs = {
        (iid, negative_iid)
        for iid in observed_correct_iids
        for negative_iid in recomputed["negative_iids_by_correct_iid"][iid]
    }
    if (
        pair_count != EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT
        or len(observed_correct_iids) != EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT
        or missing_correct_iids != EXPECTED_MISSING_CORRECT_SOURCE_IIDS
        or len(observed_pairs) != EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT
        or observed_pairs != expected_executed_pairs
        or recomputed.get("registered_directed_source_pair_count")
        != EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT
    ):
        raise AllThreeNegativeRawError("candidate executed or registered source-pair closure differs")
    return value, raw_sha


def _finite(value: Any, *, label: str) -> float:
    if type(value) not in {int, float}:
        raise AllThreeNegativeRawError(f"{label} is not a strict numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise AllThreeNegativeRawError(f"{label} is non-finite")
    return result


def raw_pair_metrics(
    candidate_global: Any,
    candidate_dense: Any,
    correct_global: Any,
    correct_dense: Any,
    wrong_global: Any,
    wrong_dense: Any,
) -> dict[str, Any]:
    """Extend the frozen cyclic raw fields without changing their values."""

    result = dict(frozen.raw_metrics(
        candidate_global,
        candidate_dense,
        correct_global,
        correct_dense,
        wrong_global,
        wrong_dense,
    ))
    global_denominator = _finite(
        result["global_source_self_upper_bound"] - result["global_candidate_wrong"],
        label="global normalized-contrast denominator",
    )
    dense_denominator = _finite(
        result["dense_source_self_upper_bound"] - result["dense_candidate_wrong"],
        label="dense normalized-contrast denominator",
    )
    global_contrast = (
        result["global_correct_minus_wrong_margin"] / global_denominator
        if global_denominator > 0.0
        else 0.0
    )
    dense_contrast = (
        result["dense_correct_minus_wrong_margin"] / dense_denominator
        if dense_denominator > 0.0
        else 0.0
    )
    result.update({
        **AUTHORITY_CLOSURE,
        "global_wrong_normalized_contrast_denominator": global_denominator,
        "global_wrong_normalized_contrast": _finite(global_contrast, label="global normalized contrast"),
        "dense_wrong_normalized_contrast_denominator": dense_denominator,
        "dense_wrong_normalized_contrast": _finite(dense_contrast, label="dense normalized contrast"),
        "normalized_contrast_zero_when_denominator_nonpositive": True,
        "descriptive_only": True,
        "operational_diagnostic_only": True,
        "multi_negative_proxy_authority": False,
        "formal_retained_source_fd_authority": False,
    })
    return result


_SUMMARY_FIELDS = (
    "global_correct_minus_wrong_margin",
    "global_wrong_normalized_contrast",
    "dense_correct_minus_wrong_margin",
    "dense_wrong_normalized_contrast",
)
RAW_METRIC_FIELDS = frozenset({
    "measurement_label",
    "global_candidate_correct", "global_candidate_wrong",
    "global_correct_minus_wrong_margin", "global_source_self_upper_bound",
    "dense_candidate_correct", "dense_candidate_wrong",
    "dense_correct_minus_wrong_margin", "dense_source_self_upper_bound",
    "thresholds",
    "global_wrong_normalized_contrast_denominator",
    "global_wrong_normalized_contrast",
    "dense_wrong_normalized_contrast_denominator",
    "dense_wrong_normalized_contrast",
    "normalized_contrast_zero_when_denominator_nonpositive",
    "descriptive_only", "operational_diagnostic_only",
    *AUTHORITY_CLOSURE.keys(),
})


def _metric_close(actual: Any, expected: float, *, label: str) -> None:
    value = _finite(actual, label=label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise AllThreeNegativeRawError(f"{label} arithmetic closure differs")


def _validate_raw_metric_closure(metrics: Mapping[str, Any]) -> None:
    if set(metrics) != RAW_METRIC_FIELDS:
        raise AllThreeNegativeRawError("raw metric field closure differs")
    if set(AUTHORITY_CLOSURE).difference(metrics):
        raise AllThreeNegativeRawError("raw metrics omit authority closure fields")
    if any(metrics.get(key) != value for key, value in AUTHORITY_CLOSURE.items()):
        raise AllThreeNegativeRawError("raw metrics authority closure differs")
    if (
        metrics.get("measurement_label")
        != "frozen_dinov2_source_bound_raw_proxy_only"
        or metrics.get("thresholds") is not None
        or metrics.get("normalized_contrast_zero_when_denominator_nonpositive") is not True
        or metrics.get("descriptive_only") is not True
        or metrics.get("operational_diagnostic_only") is not True
        or metrics.get("multi_negative_proxy_authority") is not False
        or metrics.get("formal_retained_source_fd_authority") is not False
    ):
        raise AllThreeNegativeRawError("raw metric descriptive/authority closure differs")
    global_correct = _finite(metrics.get("global_candidate_correct"), label="global correct")
    global_wrong = _finite(metrics.get("global_candidate_wrong"), label="global wrong")
    global_self = _finite(metrics.get("global_source_self_upper_bound"), label="global self")
    dense_correct = _finite(metrics.get("dense_candidate_correct"), label="dense correct")
    dense_wrong = _finite(metrics.get("dense_candidate_wrong"), label="dense wrong")
    dense_self = _finite(metrics.get("dense_source_self_upper_bound"), label="dense self")
    if not all(0.0 <= value <= 1.0 for value in (
        global_correct, global_wrong, global_self,
        dense_correct, dense_wrong, dense_self,
    )):
        raise AllThreeNegativeRawError("raw mapped-cosine metric escaped [0,1]")
    if global_self != 1.0 or dense_self != 1.0:
        raise AllThreeNegativeRawError("raw metric source-self closure differs")
    global_margin = global_correct - global_wrong
    dense_margin = dense_correct - dense_wrong
    global_denominator = global_self - global_wrong
    dense_denominator = dense_self - dense_wrong
    global_contrast = global_margin / global_denominator if global_denominator > 0.0 else 0.0
    dense_contrast = dense_margin / dense_denominator if dense_denominator > 0.0 else 0.0
    _metric_close(metrics.get("global_correct_minus_wrong_margin"), global_margin, label="global margin")
    _metric_close(metrics.get("dense_correct_minus_wrong_margin"), dense_margin, label="dense margin")
    _metric_close(metrics.get("global_wrong_normalized_contrast_denominator"), global_denominator, label="global denominator")
    _metric_close(metrics.get("dense_wrong_normalized_contrast_denominator"), dense_denominator, label="dense denominator")
    _metric_close(metrics.get("global_wrong_normalized_contrast"), global_contrast, label="global normalized contrast")
    _metric_close(metrics.get("dense_wrong_normalized_contrast"), dense_contrast, label="dense normalized contrast")


def descriptive_candidate_summary(pair_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(pair_metrics) != EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE:
        raise AllThreeNegativeRawError("candidate summary requires exactly three negative pairs")
    summaries = {}
    for field in _SUMMARY_FIELDS:
        values = sorted(_finite(row.get(field), label=field) for row in pair_metrics)
        summaries[field] = {"worst_minimum": values[0], "median": values[1]}
    return {
        "negative_count": EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE,
        "worst_definition": "minimum_of_three_preregistered_same_actor_pair_values",
        "statistics": summaries,
        "descriptive_only": True,
        "thresholds": None,
        "ranking_authorized": False,
        "selection_authorized": False,
        "training_target_authorized": False,
        "scientific_claim_authorized": False,
    }


def _source_feature_cache(
    manifest: Mapping[str, Any],
    *,
    scorer: Any,
    processor: Any,
    model: Any,
    device: Any,
    evaluator_spec: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sources = _collect_source_bindings(manifest["attempts"])
    order = manifest["negative_design"]["source_manifest_order"]
    if list(sources) != order:
        sources = {iid: sources[iid] for iid in order}
    cache: dict[str, dict[str, Any]] = {}
    entries = []
    for iid in order:
        binding = sources[iid]
        global_feature, dense_feature, evidence = frozen._features(
            binding,
            scorer=scorer,
            processor=processor,
            model=model,
            device=device,
            evaluator_spec=evaluator_spec,
        )
        feature = evidence["features"]
        entry = {
            "iid": iid,
            "actor_family": binding["actor_family"],
            "source_video_sha256": binding["source_video_sha256"],
            "global_feature_sha256": feature["global_feature_sha256"],
            "dense_feature_sha256": feature["dense_feature_sha256"],
            "decode": evidence["decode"],
            "feature_geometry": {
                "selected_frame_count": feature["selected_frame_count"],
                "dense_grid_height": feature["dense_grid_height"],
                "dense_grid_width": feature["dense_grid_width"],
                "feature_dimension": feature["feature_dimension"],
            },
        }
        cache[iid] = {
            "global": global_feature,
            "dense": dense_feature,
            "entry": entry,
        }
        entries.append(entry)
    hash_map = [{
        "iid": row["iid"],
        "source_video_sha256": row["source_video_sha256"],
        "global_feature_sha256": row["global_feature_sha256"],
        "dense_feature_sha256": row["dense_feature_sha256"],
    } for row in entries]
    summary = {
        "cache_scope": "one_rank_process",
        "source_count": len(entries),
        "source_manifest_order": list(order),
        "all_sources_warmed_before_candidate_decode": True,
        "cache_reused_for_all_candidate_measurements": True,
        "source_features_held_in_cpu_memory_until_worker_exit": True,
        "source_files_retained_open_until_worker_exit": False,
        "entries": entries,
        "feature_hash_map_sha256": core.object_sha256(hash_map),
        "operational_limitation": dict(OPERATIONAL_LIMITATION),
    }
    if len(cache) != EXPECTED_SOURCE_COUNT:
        raise AllThreeNegativeRawError("source feature cache did not warm exact8")
    return cache, summary


def _measure(
    row: Mapping[str, Any],
    *,
    evaluator: Mapping[str, Any],
    model: Any,
    device: Any,
    source_cache: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scorer, processor, spec = evaluator["scorer"], evaluator["processor"], evaluator["spec"]
    if core.file_sha256(row["receipt_path"]) != row["receipt_sha256"]:
        raise AllThreeNegativeRawError("generation receipt changed after input sealing")
    frames, candidate_decode = scorer.decode_exact81_rgb(
        row["mp4_path"],
        expected_sha256=row["mp4_sha256"],
    )
    _, pixels = scorer.preprocess_selected_rgb(frames, processor)
    candidate_global, candidate_dense, candidate_features = scorer.extract_features(
        model,
        pixels,
        device=device,
        num_register_tokens=spec["model"]["num_register_tokens"],
        evaluation_image_size=spec["model"]["preprocessor_golden_output_shape"][-1],
        patch_size=spec["model"]["patch_size"],
    )
    correct_iid = row["correct_source"]["iid"]
    correct = source_cache[correct_iid]
    negative_results = []
    for ordinal, binding in enumerate(row["negative_sources"]):
        negative_iid = binding["iid"]
        negative = source_cache[negative_iid]
        metrics = raw_pair_metrics(
            candidate_global,
            candidate_dense,
            correct["global"],
            correct["dense"],
            negative["global"],
            negative["dense"],
        )
        negative_results.append({
            "negative_ordinal_in_manifest_order": ordinal,
            "correct_source_iid": correct_iid,
            "correct_source_video_sha256": row["correct_source"]["source_video_sha256"],
            "negative_source_iid": negative_iid,
            "negative_source_video_sha256": binding["source_video_sha256"],
            "is_legacy_cyclic_negative": negative_iid == row["legacy_cyclic_negative_iid"],
            "raw_metrics": metrics,
            "authority": dict(AUTHORITY_CLOSURE),
        })
    if sum(row["is_legacy_cyclic_negative"] for row in negative_results) != 1:
        raise AllThreeNegativeRawError("candidate does not have exactly one legacy cyclic pair")
    summary = descriptive_candidate_summary([row["raw_metrics"] for row in negative_results])
    return {
        "candidate_id": row["candidate_id"],
        "candidate_binding": dict(row),
        "candidate_decode": candidate_decode,
        "candidate_features": candidate_features,
        "source_features_served_from_exact8_rank_cache": True,
        "negative_results": negative_results,
        "candidate_descriptive_summary": summary,
        "operational_limitation": dict(OPERATIONAL_LIMITATION),
        "authority": dict(AUTHORITY_CLOSURE),
    }


def _worker_common(
    args: Any,
) -> tuple[str, dict[str, Any], str, Mapping[str, Any], dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    source_sha = core._verify_self(args.expected_source_sha256)
    if (
        args.expected_evaluator_spec_sha256 != EXPECTED_EVALUATOR_SPEC_SHA256
        or args.expected_visual_scorer_sha256 != EXPECTED_VISUAL_SCORER_SHA256
        or args.expected_visual_contract_sha256 != EXPECTED_VISUAL_CONTRACT_SHA256
    ):
        raise AllThreeNegativeRawError("registered visual evaluator identity differs")
    manifest, manifest_sha = load_input_manifest(
        args.input_manifest,
        expected_sha256=args.expected_input_manifest_sha256,
        expected_source_sha256=source_sha,
    )
    evaluator, checkpoint = core._load_evaluator(args)
    device = core._configure_device()
    model, loading_counts = evaluator["scorer"].load_frozen_model(
        checkpoint,
        device=device,
    )
    checkpoint["root"] = str(checkpoint["root"])
    checkpoint["loading_counts"] = loading_counts
    checkpoint["frozen_eval"] = True
    checkpoint["trainable_parameter_tensors"] = 0
    checkpoint["identity_authority"] = False
    checkpoint["scientific_claim_authorized"] = False
    source_cache, cache_summary = _source_feature_cache(
        manifest,
        scorer=evaluator["scorer"],
        processor=evaluator["processor"],
        model=model,
        device=device,
        evaluator_spec=evaluator["spec"],
    )
    return (
        source_sha,
        manifest,
        manifest_sha,
        evaluator,
        checkpoint,
        (model, device),
        source_cache,
        cache_summary,
    )


def preflight(args: Any) -> int:
    source_sha, manifest, manifest_sha, evaluator, checkpoint, owned, cache, cache_summary = _worker_common(args)
    rank = core._rank(args.rank, world_size=EXPECTED_WORLD_SIZE)
    index = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, EXPECTED_WORLD_SIZE)[0]
    result = _measure(
        manifest["attempts"][index],
        evaluator=evaluator,
        model=owned[0],
        device=owned[1],
        source_cache=cache,
    )
    unsigned = {
        "schema_version": PREFLIGHT_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "rank": rank,
        "world_size": EXPECTED_WORLD_SIZE,
        "one_candidate_only": True,
        "exact8_source_feature_cache": cache_summary,
        "candidate_result": result,
        "visual_evaluator": checkpoint,
        "operational_limitation": dict(OPERATIONAL_LIMITATION),
        "authority": dict(AUTHORITY_CLOSURE),
    }
    core._write_create_only(
        core._plain_directory(args.output_root, label="output root")
        / f"preflight-rank-{rank:02d}.json",
        {**unsigned, "receipt_digest": core.object_sha256(unsigned)},
    )
    return 0


def worker(args: Any) -> int:
    source_sha, manifest, manifest_sha, evaluator, checkpoint, owned, cache, cache_summary = _worker_common(args)
    rank = core._rank(args.rank, world_size=args.world_size)
    if args.world_size != EXPECTED_WORLD_SIZE:
        raise AllThreeNegativeRawError("worker world size must be exactly eight")
    indices = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, args.world_size)
    results = [
        _measure(
            manifest["attempts"][index],
            evaluator=evaluator,
            model=owned[0],
            device=owned[1],
            source_cache=cache,
        )
        for index in indices
    ]
    unsigned = {
        "schema_version": SHARD_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "rank": rank,
        "world_size": EXPECTED_WORLD_SIZE,
        "partition_indices": list(indices),
        "candidate_count": len(results),
        "candidate_negative_pair_count": len(results) * EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE,
        "exact8_source_feature_cache": cache_summary,
        "candidate_results": results,
        "visual_evaluator": checkpoint,
        "operational_limitation": dict(OPERATIONAL_LIMITATION),
        "authority": dict(AUTHORITY_CLOSURE),
    }
    core._write_create_only(
        core._plain_directory(args.output_root, label="output root")
        / f"shard-{rank:02d}-of-{EXPECTED_WORLD_SIZE:02d}.json",
        {**unsigned, "receipt_digest": core.object_sha256(unsigned)},
    )
    return 0


def _cache_hash_map(
    summary: Any,
    *,
    expected_order: Sequence[str],
    expected_sources: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    if not isinstance(summary, Mapping) or summary.get("source_count") != EXPECTED_SOURCE_COUNT:
        raise AllThreeNegativeRawError("rank source cache summary differs")
    entries = summary.get("entries")
    if (
        summary.get("cache_scope") != "one_rank_process"
        or summary.get("source_manifest_order") != list(expected_order)
        or summary.get("all_sources_warmed_before_candidate_decode") is not True
        or summary.get("cache_reused_for_all_candidate_measurements") is not True
        or summary.get("source_features_held_in_cpu_memory_until_worker_exit") is not True
        or summary.get("source_files_retained_open_until_worker_exit") is not False
        or summary.get("operational_limitation") != OPERATIONAL_LIMITATION
        or not isinstance(entries, list)
        or len(entries) != EXPECTED_SOURCE_COUNT
        or [row.get("iid") for row in entries if isinstance(row, Mapping)] != list(expected_order)
    ):
        raise AllThreeNegativeRawError("rank exact8 source cache closure differs")
    hash_map = []
    for row in entries:
        iid = row.get("iid")
        expected = expected_sources.get(iid) if isinstance(iid, str) else None
        if (
            not isinstance(expected, Mapping)
            or row.get("actor_family") != expected.get("actor_family")
            or row.get("source_video_sha256") != expected.get("source_video_sha256")
        ):
            raise AllThreeNegativeRawError("rank cached source identity differs from manifest binding")
        hash_map.append({
            "iid": iid,
            "source_video_sha256": core._sha256(row.get("source_video_sha256"), label="cached source SHA-256"),
            "global_feature_sha256": core._sha256(row.get("global_feature_sha256"), label="cached global feature SHA-256"),
            "dense_feature_sha256": core._sha256(row.get("dense_feature_sha256"), label="cached dense feature SHA-256"),
        })
    if summary.get("feature_hash_map_sha256") != core.object_sha256(hash_map):
        raise AllThreeNegativeRawError("rank source cache feature hash-map digest differs")
    return hash_map


def _validate_candidate_result(
    result: Any,
    *,
    expected: Mapping[str, Any],
) -> set[tuple[str, str]]:
    if not isinstance(result, Mapping):
        raise AllThreeNegativeRawError("candidate result differs")
    negatives = result.get("negative_results")
    if (
        result.get("candidate_id") != expected["candidate_id"]
        or result.get("candidate_binding") != expected
        or result.get("source_features_served_from_exact8_rank_cache") is not True
        or result.get("operational_limitation") != OPERATIONAL_LIMITATION
        or result.get("authority") != AUTHORITY_CLOSURE
        or not isinstance(negatives, list)
        or len(negatives) != EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE
    ):
        raise AllThreeNegativeRawError("candidate result contract differs")
    expected_negative_iids = [row["iid"] for row in expected["negative_sources"]]
    if [row.get("negative_source_iid") for row in negatives if isinstance(row, Mapping)] != expected_negative_iids:
        raise AllThreeNegativeRawError("candidate negative result order differs")
    observed_pairs = set()
    for ordinal, row in enumerate(negatives):
        metrics = row.get("raw_metrics")
        expected_negative = expected["negative_sources"][ordinal]
        expected_is_legacy = (
            expected_negative["iid"] == expected["legacy_cyclic_negative_iid"]
        )
        if (
            row.get("negative_ordinal_in_manifest_order") != ordinal
            or row.get("correct_source_iid") != expected["correct_source"]["iid"]
            or row.get("correct_source_video_sha256")
            != expected["correct_source"]["source_video_sha256"]
            or row.get("negative_source_iid") != expected_negative["iid"]
            or row.get("negative_source_video_sha256")
            != expected_negative["source_video_sha256"]
            or row.get("is_legacy_cyclic_negative") is not expected_is_legacy
            or row.get("authority") != AUTHORITY_CLOSURE
            or not isinstance(metrics, Mapping)
        ):
            raise AllThreeNegativeRawError("candidate negative raw result differs")
        _validate_raw_metric_closure(metrics)
        observed_pairs.add((row["correct_source_iid"], row["negative_source_iid"]))
    legacy_rows = [row for row in negatives if row.get("is_legacy_cyclic_negative") is True]
    if (
        len(legacy_rows) != 1
        or legacy_rows[0].get("negative_source_iid")
        != expected["legacy_cyclic_negative_iid"]
    ):
        raise AllThreeNegativeRawError("candidate legacy cyclic marker count differs")
    expected_summary = descriptive_candidate_summary([row["raw_metrics"] for row in negatives])
    if result.get("candidate_descriptive_summary") != expected_summary:
        raise AllThreeNegativeRawError("candidate descriptive summary differs")
    return observed_pairs


def _verify_legacy_regression(
    ordered: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    legacy_path: str | Path,
    expected_legacy_sha256: str,
) -> dict[str, Any]:
    legacy, evidence = _legacy_aggregate(legacy_path, expected_legacy_sha256)
    if evidence != manifest["legacy_cyclic_regression"]:
        raise AllThreeNegativeRawError("aggregate cyclic regression input differs from preregistration")
    legacy_by_id = {row["candidate_id"]: row for row in legacy["candidate_results"]}
    matched_fields = 0
    for result in ordered:
        old = legacy_by_id.get(result["candidate_id"])
        if not isinstance(old, Mapping):
            raise AllThreeNegativeRawError("cyclic regression candidate is absent")
        old_binding = old.get("candidate_binding")
        new_binding = result["candidate_binding"]
        if (
            not isinstance(old_binding, Mapping)
            or old_binding.get("candidate_id") != new_binding["candidate_id"]
            or old_binding.get("receipt_sha256") != new_binding["receipt_sha256"]
            or old_binding.get("mp4_sha256") != new_binding["mp4_sha256"]
            or old_binding.get("correct_source", {}).get("iid") != new_binding["correct_source"]["iid"]
            or old_binding.get("wrong_source", {}).get("iid") != new_binding["legacy_cyclic_negative_iid"]
        ):
            raise AllThreeNegativeRawError("cyclic regression candidate/source binding differs")
        legacy_pairs = [
            row for row in result["negative_results"]
            if row.get("is_legacy_cyclic_negative") is True
        ]
        old_metrics = old.get("raw_metrics")
        if len(legacy_pairs) != 1 or not isinstance(old_metrics, Mapping):
            raise AllThreeNegativeRawError("cyclic regression raw metrics are absent")
        new_metrics = legacy_pairs[0]["raw_metrics"]
        projection = {key: new_metrics.get(key) for key in old_metrics}
        if projection != old_metrics:
            raise AllThreeNegativeRawError(
                f"cyclic raw-metric regression differs for {result['candidate_id']}"
            )
        matched_fields += len(old_metrics)
    return {
        **evidence,
        "comparison_mode": "exact_json_field_value_equality",
        "candidate_match_count": EXPECTED_ATTEMPT_COUNT,
        "raw_metric_field_value_match_count": matched_fields,
        "all_33_legacy_cyclic_raw_metrics_exact": True,
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
    shards, by_index, reference_hash_map = [], {}, None
    cross_rank_cache_receipts = []
    for rank in range(EXPECTED_WORLD_SIZE):
        path = output_root / f"shard-{rank:02d}-of-{EXPECTED_WORLD_SIZE:02d}.json"
        value, raw_sha = core._strict_json(path, expected_sha256=None, label=f"shard {rank}")
        unsigned = dict(value)
        declared = core._sha256(unsigned.pop("receipt_digest", None), label="shard digest")
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
            or value.get("candidate_negative_pair_count") != len(indices) * EXPECTED_NEGATIVE_COUNT_PER_CANDIDATE
            or not isinstance(results, list)
            or len(results) != len(indices)
            or value.get("operational_limitation") != OPERATIONAL_LIMITATION
            or value.get("authority") != AUTHORITY_CLOSURE
            or declared != core.object_sha256(unsigned)
        ):
            raise AllThreeNegativeRawError(f"shard {rank} contract differs")
        hash_map = _cache_hash_map(
            value.get("exact8_source_feature_cache"),
            expected_order=expected_order,
            expected_sources=expected_sources,
        )
        if reference_hash_map is None:
            reference_hash_map = hash_map
        elif hash_map != reference_hash_map:
            raise AllThreeNegativeRawError("source feature hashes differ across ranks")
        cross_rank_cache_receipts.append({
            "rank": rank,
            "feature_hash_map_sha256": core.object_sha256(hash_map),
        })
        shards.append({
            "rank": rank,
            "path": str(path.resolve(strict=True)),
            "sha256": raw_sha,
            "receipt_digest": declared,
        })
        for index, result in zip(indices, results):
            if index in by_index:
                raise AllThreeNegativeRawError("shard partition overlaps")
            by_index[index] = result
    if set(by_index) != set(range(EXPECTED_ATTEMPT_COUNT)):
        raise AllThreeNegativeRawError("shards do not cover r5 exact33")
    ordered = [by_index[index] for index in range(EXPECTED_ATTEMPT_COUNT)]
    expected_ids = [row["candidate_id"] for row in manifest["attempts"]]
    if [row.get("candidate_id") for row in ordered] != expected_ids:
        raise AllThreeNegativeRawError("aggregate candidate order differs")
    observed_pairs = set()
    observed_correct_iids = set()
    pair_count = 0
    for result, expected in zip(ordered, manifest["attempts"]):
        observed_pairs.update(_validate_candidate_result(result, expected=expected))
        observed_correct_iids.add(expected["correct_source"]["iid"])
        pair_count += len(result["negative_results"])
    missing_correct_iids = set(expected_order) - observed_correct_iids
    expected_executed_pairs = {
        (iid, negative_iid)
        for iid in observed_correct_iids
        for negative_iid in manifest["negative_design"]["negative_iids_by_correct_iid"][iid]
    }
    if (
        pair_count != EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT
        or len(observed_correct_iids) != EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT
        or missing_correct_iids != EXPECTED_MISSING_CORRECT_SOURCE_IIDS
        or len(observed_pairs) != EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT
        or observed_pairs != expected_executed_pairs
        or manifest["negative_design"].get("registered_directed_source_pair_count")
        != EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT
    ):
        raise AllThreeNegativeRawError("aggregate 99/executed15/registered24 pair closure differs")
    legacy_regression = _verify_legacy_regression(
        ordered,
        manifest=manifest,
        legacy_path=args.legacy_cyclic_aggregate,
        expected_legacy_sha256=args.expected_legacy_cyclic_aggregate_sha256,
    )
    cache_consistency = {
        "source_count": EXPECTED_SOURCE_COUNT,
        "rank_count": EXPECTED_WORLD_SIZE,
        "per_source_feature_hashes": reference_hash_map,
        "per_rank_feature_hash_map_receipts": cross_rank_cache_receipts,
        "all_exact8_source_feature_hashes_identical_across_all8_ranks": True,
    }
    executed_pair_rows = [
        {"correct_source_iid": correct_iid, "negative_source_iid": negative_iid}
        for correct_iid, negative_iid in sorted(observed_pairs)
    ]
    unsigned = {
        "schema_version": AGGREGATE_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "world_size": EXPECTED_WORLD_SIZE,
        "candidate_count": EXPECTED_ATTEMPT_COUNT,
        "candidate_negative_pair_count": EXPECTED_CANDIDATE_NEGATIVE_PAIR_COUNT,
        "executed_correct_source_count": EXPECTED_EXECUTED_CORRECT_SOURCE_COUNT,
        "executed_correct_source_iids": [iid for iid in expected_order if iid in observed_correct_iids],
        "missing_correct_source_iids": [
            iid for iid in expected_order
            if iid in EXPECTED_MISSING_CORRECT_SOURCE_IIDS
        ],
        "executed_directed_source_pair_count": EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT,
        "executed_directed_source_pairs": executed_pair_rows,
        "executed_directed_source_pairs_sha256": core.object_sha256(executed_pair_rows),
        "registered_directed_source_pair_universe_count": EXPECTED_REGISTERED_DIRECTED_SOURCE_PAIR_COUNT,
        "registered_directed_source_pair_universe_sha256": core.object_sha256(
            manifest["negative_design"]["directed_source_pairs"]
        ),
        "coverage": "exactly_once_complete_r5_partial33_same_actor_all_three_negatives_raw",
        "candidate_order": expected_ids,
        "shards": shards,
        "cross_rank_source_feature_cache_consistency": cache_consistency,
        "legacy_cyclic_raw_metric_regression": legacy_regression,
        "candidate_results": ordered,
        "interpretation": {
            "measurement": "raw frozen-DINO candidate/correct/all-three-same-actor-negative source proxies",
            "negative_map_preregistered_from_actor_and_manifest_order_only": True,
            "registered_pair_universe_24_is_not_claimed_as_executed": True,
            "executed_unique_directed_source_pairs": EXPECTED_EXECUTED_DIRECTED_SOURCE_PAIR_COUNT,
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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--attempts-root", required=True)
    build.add_argument("--expected-root-spec-sha256", required=True)
    build.add_argument("--source-manifest", required=True)
    build.add_argument("--expected-source-manifest-sha256", required=True)
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


def main(argv: Sequence[str] | None = None) -> int:
    _configure_core()
    _configure_partitions()
    os.umask(0o077)
    args = build_parser().parse_args(argv)
    return {
        "build-manifest": build_manifest,
        "preflight": preflight,
        "worker": worker,
        "aggregate": aggregate,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
