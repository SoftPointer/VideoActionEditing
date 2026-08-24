#!/usr/bin/env python3
"""Raw source-bound frozen-DINO proxies for the exact47 historical SAIC r4 bank."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
_PINNED = {
    "diagnose_saic_partial28_dinov2_temporal_v1.py": "213e408295610b5a0dd2e1eeb54f406c19a1985fb1ff290f89522fd38b4aaf4d",
    "build_saic_reversible_source_set_v1.py": "0cf012adf25dd1afffb33d1e0c918630a574c9075e9aa293914e04890c71cf5b",
}
for _name, _sha in _PINNED.items():
    _path = METHOD_ROOT / _name
    if not _path.is_file() or _path.is_symlink() or hashlib.sha256(_path.read_bytes()).hexdigest() != _sha:
        raise RuntimeError(f"pinned dependency differs: {_name}")

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import diagnose_saic_partial28_dinov2_temporal_v1 as core  # noqa: E402


SCHEMA_VERSION = "bernini-saic-partial47-source-bound-dinov2-raw-v1"
INPUT_SCHEMA = f"{SCHEMA_VERSION}-input"
SHARD_SCHEMA = f"{SCHEMA_VERSION}-shard"
AGGREGATE_SCHEMA = f"{SCHEMA_VERSION}-aggregate"
PREFLIGHT_SCHEMA = f"{SCHEMA_VERSION}-preflight"
EXPECTED_ATTEMPT_COUNT = 47
EXPECTED_WORLD_SIZE = 8
EXPECTED_PARTITION_SIZES = (6, 6, 6, 6, 6, 6, 6, 5)
EXPECTED_SOURCE_MANIFEST_SHA256 = "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
EXPECTED_EVALUATOR_SPEC_SHA256 = "6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736"
EXPECTED_VISUAL_SCORER_SHA256 = "9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39"
EXPECTED_VISUAL_CONTRACT_SHA256 = "183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a"
WRONG_SOURCE_POLICY = "same_actor_family_iid_lexical_cyclic_next_v1"
AUTHORITY_CLOSURE = {
    **core.AUTHORITY_CLOSURE,
    "absolute_preservation_authority": False,
    "source_bound_proxy_authority": False,
}
SourceBoundRawError = core.Partial28DINOError
_base_partition_indices = core.partition_indices


def _configure_core() -> None:
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
    sizes = tuple(len(_base_partition_indices(count, item, world_size)) for item in range(world_size))
    if sizes != EXPECTED_PARTITION_SIZES:
        raise SourceBoundRawError("partial47 source-bound partition sizes differ")
    return indices


core.partition_indices = partition_indices


def _source_closure(source_manifest_path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = core._plain_file(source_manifest_path, label="sealed source manifest")
    expected = core._sha256(expected_sha256, label="source manifest SHA-256")
    if expected != EXPECTED_SOURCE_MANIFEST_SHA256 or core.file_sha256(path) != expected:
        raise SourceBoundRawError("sealed source manifest SHA-256 differs")
    manifest = source_set.load_manifest(path)
    try:
        summary = dict(source_set.validate_manifest(manifest, verify_bound_files=True))
    except Exception as error:
        raise SourceBoundRawError(f"sealed source manifest failed validation: {error}") from error
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise SourceBoundRawError("source manifest must contain exactly eight rows")
    by_iid: dict[str, dict[str, Any]] = {}
    for row in rows:
        iid = row.get("iid")
        sha = row.get("source_video_sha256")
        if not isinstance(iid, str) or iid in by_iid:
            raise SourceBoundRawError("source IID closure differs")
        path_value, actual = core._stable_bound_file(row.get("source_video"), sha, label=f"source {iid}")
        by_iid[iid] = {
            "iid": iid, "row_id": row.get("row_id"),
            "analysis_split": row.get("analysis_split"),
            "actor_family": row.get("actor_family"),
            "actor_group_id": row.get("actor_group_id"),
            "scene_group_id": row.get("scene_group_id"),
            "source_video": str(path_value), "source_video_sha256": actual,
        }
    by_actor: dict[str, list[str]] = {}
    for iid, row in by_iid.items():
        by_actor.setdefault(row["actor_family"], []).append(iid)
    if set(by_actor) != {"dog", "human"} or any(len(items) != 4 for items in by_actor.values()):
        raise SourceBoundRawError("source actor-family closure differs")
    wrong: dict[str, str] = {}
    for items in by_actor.values():
        ordered = sorted(items)
        for index, iid in enumerate(ordered):
            wrong[iid] = ordered[(index + 1) % len(ordered)]
    if any(iid == wrong_iid or by_iid[iid]["source_video_sha256"] == by_iid[wrong_iid]["source_video_sha256"] for iid, wrong_iid in wrong.items()):
        raise SourceBoundRawError("wrong-source policy did not change IID and SHA")
    evidence = {
        "path": str(path), "raw_sha256": expected,
        "content_sha256": summary.get("manifest_content_sha256"),
        "validator_summary_sha256": core.object_sha256(summary),
        "bound_files_verified": summary.get("bound_files_verified"),
        "wrong_source_policy": WRONG_SOURCE_POLICY,
    }
    return by_iid, {"evidence": evidence, "wrong_by_iid": wrong}


def build_manifest(args: Any) -> int:
    source_sha = core._verify_self(args.expected_source_sha256)
    attempts_root = core._plain_directory(args.attempts_root, label="attempts root")
    output_root = Path(args.output_root)
    if not output_root.is_absolute() or output_root == Path("/") or output_root.exists() or output_root.is_symlink():
        raise SourceBoundRawError("output root must be fresh, absolute, and non-root")
    sources, source_policy = _source_closure(args.source_manifest, args.expected_source_manifest_sha256)
    paths = sorted(attempts_root.rglob(core.ATTEMPT_BASENAME), key=lambda item: item.as_posix())
    if len(paths) != EXPECTED_ATTEMPT_COUNT:
        raise SourceBoundRawError(f"exact47 bank receipt count differs: {len(paths)}")
    rows = []
    for path in paths:
        row = core.validate_attempt_receipt(path, expected_root_spec_sha256=args.expected_root_spec_sha256)
        receipt, _ = core._strict_json(path, expected_sha256=row["receipt_sha256"], label="generation receipt")
        candidate = receipt.get("candidate")
        iid = candidate.get("iid") if isinstance(candidate, Mapping) else None
        correct = sources.get(iid)
        if correct is None or candidate.get("source_media_sha256_for_nonuse_audit") != correct["source_video_sha256"]:
            raise SourceBoundRawError("candidate IID/source nonuse audit binding differs")
        if (
            candidate.get("actor_family") != correct["actor_family"]
            or candidate.get("analysis_split") != correct["analysis_split"]
            or candidate.get("row_id") != correct["row_id"]
        ):
            raise SourceBoundRawError("candidate compound source identity differs")
        wrong = sources[source_policy["wrong_by_iid"][iid]]
        rows.append({**row, "correct_source": correct, "wrong_source": wrong})
    if len({row["candidate_id"] for row in rows}) != EXPECTED_ATTEMPT_COUNT:
        raise SourceBoundRawError("candidate ID closure differs")
    rows.sort(key=lambda row: row["candidate_id"])
    output_root.mkdir(mode=0o700)
    unsigned = {
        "schema_version": INPUT_SCHEMA, "diagnostic_source_sha256": source_sha,
        "attempts_root": str(attempts_root),
        "root_spec_raw_sha256": core._sha256(args.expected_root_spec_sha256, label="root spec SHA-256"),
        "attempt_count": EXPECTED_ATTEMPT_COUNT, "world_size": EXPECTED_WORLD_SIZE,
        "partition_rule": "candidate_order_index_modulo_world_size",
        "selected_frame_indices": list(core.EVAL_FRAME_INDICES),
        "source_manifest": source_policy["evidence"], "attempts": rows,
        "authority": dict(AUTHORITY_CLOSURE),
    }
    core._write_create_only(output_root / "input-manifest.json", {**unsigned, "receipt_digest": core.object_sha256(unsigned)})
    return 0


def load_input_manifest(path: str | Path, *, expected_sha256: str, expected_source_sha256: str) -> tuple[dict[str, Any], str]:
    value, raw_sha = core._strict_json(path, expected_sha256=expected_sha256, label="source-bound input manifest")
    core._closed(value, {
        "schema_version", "diagnostic_source_sha256", "attempts_root",
        "root_spec_raw_sha256", "attempt_count", "world_size",
        "partition_rule", "selected_frame_indices", "source_manifest",
        "attempts", "authority", "receipt_digest",
    }, label="source-bound input manifest")
    unsigned = dict(value)
    declared = core._sha256(unsigned.pop("receipt_digest", None), label="input manifest digest")
    attempts = value.get("attempts")
    source_manifest = value.get("source_manifest")
    if (
        value.get("schema_version") != INPUT_SCHEMA
        or value.get("diagnostic_source_sha256") != expected_source_sha256
        or value.get("attempt_count") != EXPECTED_ATTEMPT_COUNT
        or value.get("world_size") != EXPECTED_WORLD_SIZE
        or value.get("partition_rule") != "candidate_order_index_modulo_world_size"
        or value.get("selected_frame_indices") != list(core.EVAL_FRAME_INDICES)
        or value.get("authority") != AUTHORITY_CLOSURE
        or not isinstance(source_manifest, Mapping)
        or source_manifest.get("raw_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or source_manifest.get("bound_files_verified") is not True
        or source_manifest.get("wrong_source_policy") != WRONG_SOURCE_POLICY
        or not isinstance(attempts, list) or len(attempts) != EXPECTED_ATTEMPT_COUNT
        or len({row.get("candidate_id") for row in attempts if isinstance(row, Mapping)}) != EXPECTED_ATTEMPT_COUNT
        or declared != core.object_sha256(unsigned)
    ):
        raise SourceBoundRawError("source-bound input manifest contract differs")
    for row in attempts:
        correct, wrong = row.get("correct_source"), row.get("wrong_source")
        if not isinstance(correct, Mapping) or not isinstance(wrong, Mapping) or correct.get("iid") == wrong.get("iid") or correct.get("source_video_sha256") == wrong.get("source_video_sha256"):
            raise SourceBoundRawError("correct/wrong source binding differs")
    return value, raw_sha


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SourceBoundRawError("source-bound metric is non-finite")
    return result


def raw_metrics(candidate_global: Any, candidate_dense: Any, correct_global: Any, correct_dense: Any, wrong_global: Any, wrong_dense: Any) -> dict[str, Any]:
    import torch
    def similarity(left: Any, right: Any, *, dense: bool) -> float:
        if tuple(left.shape) != tuple(right.shape):
            raise SourceBoundRawError("source-bound feature geometry differs")
        mapped = (((left * right).sum(dim=-1) + 1.0) * 0.5).clamp(0.0, 1.0)
        return _finite((mapped.reshape(-1).median() if dense else mapped.mean()).item())
    global_correct = similarity(candidate_global, correct_global, dense=False)
    global_wrong = similarity(candidate_global, wrong_global, dense=False)
    dense_correct = similarity(candidate_dense, correct_dense, dense=True)
    dense_wrong = similarity(candidate_dense, wrong_dense, dense=True)
    measured_global_self = similarity(correct_global, correct_global, dense=False)
    measured_dense_self = similarity(correct_dense, correct_dense, dense=True)
    if not math.isclose(measured_global_self, 1.0, rel_tol=0.0, abs_tol=1.0e-6) or not math.isclose(measured_dense_self, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise SourceBoundRawError("normalized source-self check differs from one")
    global_self = 1.0
    dense_self = 1.0
    return {
        "measurement_label": "frozen_dinov2_source_bound_raw_proxy_only",
        "global_candidate_correct": global_correct, "global_candidate_wrong": global_wrong,
        "global_correct_minus_wrong_margin": global_correct - global_wrong,
        "global_source_self_upper_bound": global_self,
        "dense_candidate_correct": dense_correct, "dense_candidate_wrong": dense_wrong,
        "dense_correct_minus_wrong_margin": dense_correct - dense_wrong,
        "dense_source_self_upper_bound": dense_self,
        "thresholds": None, "absolute_preservation_authority": False,
        "identity_authority": False, "event_authority": False,
        "scientific_claim_authorized": False, "ranking_authorized": False,
        "selection_authorized": False, "training_target_authorized": False,
    }


def _features(binding: Mapping[str, Any], *, scorer: Any, processor: Any, model: Any, device: Any, evaluator_spec: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    frames, decode = scorer.decode_exact81_rgb(binding["source_video"], expected_sha256=binding["source_video_sha256"])
    _, normalized = scorer.preprocess_selected_rgb(frames, processor)
    global_feature, dense_feature, feature = scorer.extract_features(
        model, normalized, device=device,
        num_register_tokens=evaluator_spec["model"]["num_register_tokens"],
        evaluation_image_size=evaluator_spec["model"]["preprocessor_golden_output_shape"][-1],
        patch_size=evaluator_spec["model"]["patch_size"],
    )
    return global_feature, dense_feature, {"decode": decode, "features": feature}


def _measure(row: Mapping[str, Any], *, evaluator: Mapping[str, Any], model: Any, device: Any) -> dict[str, Any]:
    scorer, processor, spec = evaluator["scorer"], evaluator["processor"], evaluator["spec"]
    if core.file_sha256(row["receipt_path"]) != row["receipt_sha256"]:
        raise SourceBoundRawError("generation receipt changed after input sealing")
    candidate_frames, candidate_decode = scorer.decode_exact81_rgb(row["mp4_path"], expected_sha256=row["mp4_sha256"])
    _, candidate_pixels = scorer.preprocess_selected_rgb(candidate_frames, processor)
    candidate_global, candidate_dense, candidate_features = scorer.extract_features(
        model, candidate_pixels, device=device, num_register_tokens=spec["model"]["num_register_tokens"],
        evaluation_image_size=spec["model"]["preprocessor_golden_output_shape"][-1], patch_size=spec["model"]["patch_size"],
    )
    correct_global, correct_dense, correct_evidence = _features(row["correct_source"], scorer=scorer, processor=processor, model=model, device=device, evaluator_spec=spec)
    wrong_global, wrong_dense, wrong_evidence = _features(row["wrong_source"], scorer=scorer, processor=processor, model=model, device=device, evaluator_spec=spec)
    return {
        "candidate_id": row["candidate_id"], "candidate_binding": dict(row),
        "candidate_decode": candidate_decode, "candidate_features": candidate_features,
        "correct_source_evidence": correct_evidence, "wrong_source_evidence": wrong_evidence,
        "raw_metrics": raw_metrics(candidate_global, candidate_dense, correct_global, correct_dense, wrong_global, wrong_dense),
        "authority": dict(AUTHORITY_CLOSURE),
    }


def _worker_common(args: Any) -> tuple[str, dict[str, Any], str, Mapping[str, Any], dict[str, Any], Any]:
    source_sha = core._verify_self(args.expected_source_sha256)
    if (
        args.expected_evaluator_spec_sha256 != EXPECTED_EVALUATOR_SPEC_SHA256
        or args.expected_visual_scorer_sha256 != EXPECTED_VISUAL_SCORER_SHA256
        or args.expected_visual_contract_sha256 != EXPECTED_VISUAL_CONTRACT_SHA256
    ):
        raise SourceBoundRawError("registered visual evaluator identity differs")
    manifest, manifest_sha = load_input_manifest(args.input_manifest, expected_sha256=args.expected_input_manifest_sha256, expected_source_sha256=source_sha)
    evaluator, checkpoint = core._load_evaluator(args)
    device = core._configure_device()
    model, loading_counts = evaluator["scorer"].load_frozen_model(checkpoint, device=device)
    checkpoint["root"] = str(checkpoint["root"])
    checkpoint["loading_counts"] = loading_counts
    checkpoint["frozen_eval"] = True
    checkpoint["trainable_parameter_tensors"] = 0
    checkpoint["identity_authority"] = False
    checkpoint["scientific_claim_authorized"] = False
    return source_sha, manifest, manifest_sha, evaluator, checkpoint, (model, device)


def preflight(args: Any) -> int:
    source_sha, manifest, manifest_sha, evaluator, checkpoint, owned = _worker_common(args)
    rank = core._rank(args.rank, world_size=EXPECTED_WORLD_SIZE)
    index = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, EXPECTED_WORLD_SIZE)[0]
    result = _measure(manifest["attempts"][index], evaluator=evaluator, model=owned[0], device=owned[1])
    unsigned = {"schema_version": PREFLIGHT_SCHEMA, "diagnostic_source_sha256": source_sha, "input_manifest_sha256": manifest_sha, "rank": rank, "world_size": EXPECTED_WORLD_SIZE, "one_candidate_only": True, "candidate_result": result, "visual_evaluator": checkpoint, "authority": dict(AUTHORITY_CLOSURE)}
    core._write_create_only(core._plain_directory(args.output_root, label="output root") / f"preflight-rank-{rank:02d}.json", {**unsigned, "receipt_digest": core.object_sha256(unsigned)})
    return 0


def worker(args: Any) -> int:
    source_sha, manifest, manifest_sha, evaluator, checkpoint, owned = _worker_common(args)
    rank = core._rank(args.rank, world_size=args.world_size)
    if args.world_size != EXPECTED_WORLD_SIZE:
        raise SourceBoundRawError("worker world size must be exactly eight")
    indices = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, args.world_size)
    results = [_measure(manifest["attempts"][index], evaluator=evaluator, model=owned[0], device=owned[1]) for index in indices]
    unsigned = {"schema_version": SHARD_SCHEMA, "diagnostic_source_sha256": source_sha, "input_manifest_sha256": manifest_sha, "rank": rank, "world_size": EXPECTED_WORLD_SIZE, "partition_indices": list(indices), "candidate_count": len(results), "candidate_results": results, "visual_evaluator": checkpoint, "authority": dict(AUTHORITY_CLOSURE)}
    core._write_create_only(core._plain_directory(args.output_root, label="output root") / f"shard-{rank:02d}-of-{EXPECTED_WORLD_SIZE:02d}.json", {**unsigned, "receipt_digest": core.object_sha256(unsigned)})
    return 0


def aggregate(args: Any) -> int:
    source_sha = core._verify_self(args.expected_source_sha256)
    manifest, manifest_sha = load_input_manifest(args.input_manifest, expected_sha256=args.expected_input_manifest_sha256, expected_source_sha256=source_sha)
    output_root = core._plain_directory(args.output_root, label="output root")
    shards, by_index = [], {}
    for rank in range(EXPECTED_WORLD_SIZE):
        path = output_root / f"shard-{rank:02d}-of-{EXPECTED_WORLD_SIZE:02d}.json"
        value, raw_sha = core._strict_json(path, expected_sha256=None, label=f"shard {rank}")
        unsigned = dict(value); declared = core._sha256(unsigned.pop("receipt_digest", None), label="shard digest")
        indices = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, EXPECTED_WORLD_SIZE)
        results = value.get("candidate_results")
        if value.get("schema_version") != SHARD_SCHEMA or value.get("diagnostic_source_sha256") != source_sha or value.get("input_manifest_sha256") != manifest_sha or value.get("rank") != rank or value.get("world_size") != EXPECTED_WORLD_SIZE or value.get("partition_indices") != list(indices) or value.get("candidate_count") != len(indices) or not isinstance(results, list) or len(results) != len(indices) or value.get("authority") != AUTHORITY_CLOSURE or declared != core.object_sha256(unsigned):
            raise SourceBoundRawError(f"shard {rank} contract differs")
        shards.append({"rank": rank, "path": str(path.resolve(strict=True)), "sha256": raw_sha, "receipt_digest": declared})
        for index, result in zip(indices, results):
            if index in by_index: raise SourceBoundRawError("shard partition overlaps")
            by_index[index] = result
    if set(by_index) != set(range(EXPECTED_ATTEMPT_COUNT)):
        raise SourceBoundRawError("shards do not cover exact partial47")
    ordered = [by_index[index] for index in range(EXPECTED_ATTEMPT_COUNT)]
    expected_ids = [row["candidate_id"] for row in manifest["attempts"]]
    if [row.get("candidate_id") for row in ordered] != expected_ids:
        raise SourceBoundRawError("aggregate candidate order differs")
    unsigned = {"schema_version": AGGREGATE_SCHEMA, "diagnostic_source_sha256": source_sha, "input_manifest_sha256": manifest_sha, "world_size": EXPECTED_WORLD_SIZE, "candidate_count": EXPECTED_ATTEMPT_COUNT, "coverage": "exactly_once_complete_partial47_source_bound_raw", "candidate_order": expected_ids, "shards": shards, "candidate_results": ordered, "interpretation": {"measurement": "raw frozen-DINO candidate/correct/wrong source proxies and source-self upper bounds", "wrong_source_preregistered_without_candidate_metrics": True, "no_absolute_preservation_claim": True, "no_event_measurement": True, "no_threshold_or_ranking": True}, "authority": dict(AUTHORITY_CLOSURE)}
    core._write_create_only(output_root / "aggregate-receipt.json", {**unsigned, "receipt_digest": core.object_sha256(unsigned)})
    return 0


def _visual_args(parser: Any) -> None:
    core._add_visual_arguments(parser)


def build_parser() -> Any:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--attempts-root", required=True); build.add_argument("--expected-root-spec-sha256", required=True)
    build.add_argument("--source-manifest", required=True); build.add_argument("--expected-source-manifest-sha256", required=True)
    build.add_argument("--expected-source-sha256", required=True); build.add_argument("--output-root", required=True)
    check = commands.add_parser("preflight"); _visual_args(check)
    run = commands.add_parser("worker"); _visual_args(run); run.add_argument("--world-size", required=True, type=int)
    combine = commands.add_parser("aggregate"); combine.add_argument("--input-manifest", required=True); combine.add_argument("--expected-input-manifest-sha256", required=True); combine.add_argument("--expected-source-sha256", required=True); combine.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    args = build_parser().parse_args(argv)
    return {"build-manifest": build_manifest, "preflight": preflight, "worker": worker, "aggregate": aggregate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
