"""Streaming, resumable rule and motion-feature cascade for Goku action edits."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np

from .archive import (
    build_feature_metadata,
    load_feature_archive,
    save_feature_archive,
)
from .descriptor import (
    DescriptorConfig,
    encode_action_descriptor,
    encode_factorized_action_delta,
)
from .geometry import MotionConfig, analyze_video
from .motion_features import extract_actor_motion_features
from .qwen_filter import (
    _object_digest as _qwen_object_digest,
    _validate_observation as _validate_qwen_observation,
    _validate_visual as _validate_qwen_visual,
)
from .rules import RULE_VERSION, score_action_rule, stable_group_split


CASCADE_SCHEMA = "motive-action-cascade-v1"
FEATURE_STAGE_VERSION = "goku-paired-motion-feature-v4"
SOURCE_GROUP_VERSION = "source-sampled-phash-v1"
BALANCED_SAMPLE_SCHEME = "family-tier-capped-round-robin-hash"
BALANCED_SAMPLE_VERSION = "v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@contextmanager
def _atomic_text_writer(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    with _atomic_text_writer(path) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _iter_chunks(
    rows: Iterable[dict[str, Any]],
    size: int,
) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def build_balanced_sample(args: argparse.Namespace) -> int:
    """Write a deterministic round-robin sample over family × rule tier."""

    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    sample_size = int(args.sample_size)
    max_per_bucket = int(args.max_per_bucket)
    seed = int(args.seed)
    if sample_size < 0:
        raise ValueError("--sample-size must be non-negative")
    if max_per_bucket < 0:
        raise ValueError("--max-per-bucket must be non-negative")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} exists; use --overwrite or a new output"
        )
    buckets: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {}
    total = 0
    for row in _iter_jsonl(input_path):
        total += 1
        rule = row.get("auto_rule", {})
        families = rule.get("action_families", [])
        family = str(families[0]) if families else "unknown"
        tier = str(rule.get("tier", "unknown"))
        bucket = f"{family}|{tier}"
        iid = str(row.get("iid") or row.get("id"))
        priority = hashlib.sha256(
            f"{seed}\0{iid}".encode("utf-8")
        ).hexdigest()
        # iid and canonical row content make the order deterministic even in
        # the practically unlikely event of a hash collision.
        buckets.setdefault(bucket, []).append(
            (priority, iid, _digest(row), row)
        )
    for values in buckets.values():
        values.sort(key=lambda item: item[:3])

    selected: list[tuple[str, int, dict[str, Any]]] = []
    selected_counts: Counter[str] = Counter()
    ordered_buckets = sorted(buckets)
    if sample_size:
        for rank in range(max_per_bucket):
            for bucket in ordered_buckets:
                values = buckets[bucket]
                if rank >= len(values):
                    continue
                selected.append((bucket, rank + 1, values[rank][3]))
                selected_counts[bucket] += 1
                if len(selected) >= sample_size:
                    break
            if len(selected) >= sample_size:
                break

    stratum_summary: dict[str, dict[str, Any]] = {}
    for bucket in ordered_buckets:
        population = len(buckets[bucket])
        selected_count = selected_counts[bucket]
        probability = selected_count / population
        stratum_summary[bucket] = {
            "population": population,
            "selected": selected_count,
            "inclusion_probability": probability,
        }

    with _atomic_text_writer(output_path) as handle:
        for bucket, within_stratum_rank, input_row in selected:
            row = dict(input_row)
            population = len(buckets[bucket])
            selected_count = selected_counts[bucket]
            probability = selected_count / population
            row["sampling_provenance"] = {
                "scheme": BALANCED_SAMPLE_SCHEME,
                "version": BALANCED_SAMPLE_VERSION,
                "seed": seed,
                "stratum": bucket,
                "stratum_population": population,
                "stratum_selected": selected_count,
                "inclusion_probability": probability,
                "inverse_probability_weight": 1.0 / probability,
                "within_stratum_rank": within_stratum_rank,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": CASCADE_SCHEMA,
        "stage": "balanced_sample",
        "input": str(input_path),
        "input_sha256": _file_digest(input_path),
        "total": total,
        "sample_size": len(selected),
        "requested_sample_size": sample_size,
        "max_per_bucket": max_per_bucket,
        "seed": seed,
        "buckets": dict(sorted(selected_counts.items())),
        "sampling_provenance": {
            "scheme": BALANCED_SAMPLE_SCHEME,
            "version": BALANCED_SAMPLE_VERSION,
            "seed": seed,
            "strata": stratum_summary,
        },
        "output_sha256": _file_digest(output_path),
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    print(
        f"[motive-cascade sample] total={total} selected={len(selected)} "
        f"buckets={len(selected_counts)} output={output_path}",
        flush=True,
    )
    return 0


def _normalized_goku_row(path: Path, raw: dict[str, Any], seed: int) -> dict[str, Any]:
    case_id = str(raw.get("case_id") or path.name.removesuffix("_all.json"))
    source_video = str(raw["source_video"])
    source_caption = str(raw.get("source_caption") or "")
    group_id, split = stable_group_split(
        source_video=source_video,
        source_caption=source_caption,
        seed=seed,
    )
    row = {
        "schema_version": CASCADE_SCHEMA,
        "iid": case_id,
        "prompt": str(raw.get("instruction_en") or raw.get("instruction") or ""),
        "src_video": source_video,
        "tgt_video": str(raw["edited_video"]),
        "source_caption": source_caption,
        "edited_caption": str(raw.get("edited_caption") or ""),
        "source_json": str(path),
        "group_id": group_id,
        "split": split,
        "split_provenance": {
            "version": "caption-or-path-fallback-v1",
            "seed": seed,
        },
    }
    row["input_digest"] = _digest(
        {
            key: row[key]
            for key in (
                "iid",
                "prompt",
                "src_video",
                "tgt_video",
                "source_caption",
                "edited_caption",
            )
        }
    )
    return row


def build_rule_manifest(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.expanduser()
    combined_dir = dataset_root / "jsons" / "combine_json"
    if not combined_dir.is_dir():
        raise FileNotFoundError(combined_dir)
    files = sorted(combined_dir.glob("*_all.json"))
    if not files:
        raise RuntimeError(f"no *_all.json files found in {combined_dir}")
    if args.sample_size is not None:
        if args.sample_size <= 0:
            raise ValueError("--sample-size must be positive")
        files = random.Random(args.seed).sample(
            files,
            min(args.sample_size, len(files)),
        )
    output_dir = args.output_dir.expanduser()
    candidates_path = output_dir / "candidates.jsonl"
    reject_path = output_dir / "reject_audit.jsonl"
    summary_path = output_dir / "summary.json"
    if not args.overwrite:
        for path in (candidates_path, reject_path, summary_path):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; use --overwrite or a new output directory"
                )

    include_tiers = set(args.include_tiers)
    labels: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    scanned = kept = reject_audit = errors = 0
    use_edited_caption = bool(
        getattr(args, "use_edited_caption_as_rule_evidence", False)
    )
    with _atomic_text_writer(candidates_path) as candidate_handle, _atomic_text_writer(
        reject_path
    ) as reject_handle:
        for path in files:
            scanned += 1
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                row = _normalized_goku_row(path, raw, args.seed)
                rule = score_action_rule(
                    row["prompt"],
                    source_caption=row["source_caption"],
                    edited_caption=row["edited_caption"],
                    use_edited_caption=use_edited_caption,
                )
                row["auto_rule"] = rule.to_dict()
                labels[rule.label] += 1
                tiers[rule.tier] += 1
                splits[row["split"]] += 1
                if rule.tier in include_tiers and rule.score >= args.min_score:
                    candidate_handle.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                    kept += 1
                else:
                    bucket = int(
                        hashlib.sha256(
                            f"{args.seed}\0{row['iid']}".encode("utf-8")
                        ).hexdigest()[:8],
                        16,
                    ) / 0xFFFFFFFF
                    if bucket < args.reject_audit_fraction:
                        reject_handle.write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )
                        reject_audit += 1
            except Exception as error:
                errors += 1
                if not args.continue_on_error:
                    raise
                error_row = {
                    "schema_version": CASCADE_SCHEMA,
                    "source_json": str(path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                reject_handle.write(json.dumps(error_row, ensure_ascii=False) + "\n")
            if scanned % 10000 == 0:
                print(
                    f"[motive-cascade rules] scanned={scanned}/{len(files)} "
                    f"kept={kept} reject_audit={reject_audit} errors={errors}",
                    flush=True,
                )

    summary = {
        "schema_version": CASCADE_SCHEMA,
        "stage": "rules",
        "rule_version": RULE_VERSION,
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "scanned": scanned,
        "kept": kept,
        "reject_audit": reject_audit,
        "errors": errors,
        "min_score": args.min_score,
        "include_tiers": list(args.include_tiers),
        "reject_audit_fraction": args.reject_audit_fraction,
        "use_edited_caption_as_rule_evidence": (
            use_edited_caption
        ),
        "labels": dict(sorted(labels.items())),
        "tiers": dict(sorted(tiers.items())),
        "splits": dict(sorted(splits.items())),
    }
    summary["config_digest"] = _digest(summary)
    _write_json(summary_path, summary)
    print(
        f"[motive-cascade rules] scanned={scanned} kept={kept} "
        f"reject_audit={reject_audit} errors={errors} output={output_dir}",
        flush=True,
    )
    return 0 if errors == 0 else 2


def _resolve_video(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _feature_score(
    *,
    row: dict[str, Any],
    source: Any,
    target: Any,
    source_actor: Any,
    target_actor: Any,
    descriptor_delta: float,
) -> tuple[float, bool, list[str]]:
    rule = row["auto_rule"]
    family = str(rule["label"])
    source_speed = float(source.metrics.residual_speed_p90)
    target_speed = float(target.metrics.residual_speed_p90)
    relevant_speed = source_speed if family == "motion_suppression" else target_speed
    actor_score = (
        source_actor.actor_likeness
        if family == "motion_suppression"
        else target_actor.actor_likeness
    )
    temporal_coverage = (
        source_actor.temporal_coverage
        if family == "motion_suppression"
        else target_actor.temporal_coverage
    )
    delta_component = min(descriptor_delta / 0.80, 1.0)
    speed_component = min(relevant_speed / 0.020, 1.0)
    feature_score = (
        0.30 * delta_component
        + 0.25 * speed_component
        + 0.30 * actor_score
        + 0.15 * temporal_coverage
    )
    reasons: list[str] = []
    invalid = {"camera_only", "cut_or_decode_artifact"}
    gate = source.label not in invalid and target.label not in invalid
    if not gate:
        reasons.append("invalid_camera_or_cut")
    if descriptor_delta < 0.25:
        gate = False
        reasons.append("small_motion_descriptor_delta")
    if family == "motion_suppression":
        if source_speed < 0.003 or source_speed < target_speed * 1.05:
            gate = False
            reasons.append("motion_not_reduced")
    elif target_speed < 0.004:
        gate = False
        reasons.append("target_motion_too_weak")
    if actor_score < 0.22:
        reasons.append("weak_actor_localization")
        feature_score *= 0.72
    if target_actor.spatial_energy_entropy > 0.92:
        reasons.append("diffuse_background_or_flicker_risk")
        feature_score *= 0.80
    return float(np.clip(feature_score, 0.0, 1.0)), gate, reasons


def _source_content_fingerprint(
    frames_gray: np.ndarray,
    *,
    max_frames: int = 6,
) -> dict[str, str]:
    """Build deterministic exact and coarse perceptual source fingerprints."""

    frames = np.asarray(frames_gray)
    if frames.ndim != 3 or len(frames) < 1:
        raise ValueError("frames_gray must have shape [T,H,W] with T >= 1")
    indices = np.rint(
        np.linspace(0, len(frames) - 1, num=min(max_frames, len(frames)))
    ).astype(np.int64)
    exact_hasher = hashlib.sha256()
    perceptual = bytearray()
    for frame_index in indices:
        frame = np.asarray(frames[int(frame_index)], dtype=np.uint8)
        compact = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        # Four-bit luminance reduces codec-noise sensitivity while remaining
        # substantially stronger than caption/path-based grouping.
        exact_hasher.update((compact >> 4).tobytes())
        coefficients = cv2.dct(compact.astype(np.float32) / 255.0)[:8, :8]
        low_frequency = coefficients.reshape(-1)[1:]
        median = float(np.median(low_frequency))
        perceptual.extend(
            np.packbits(
                (low_frequency >= median).astype(np.uint8),
                bitorder="little",
            ).tobytes()
        )
    return {
        "version": SOURCE_GROUP_VERSION,
        "sampled_frame_digest": exact_hasher.hexdigest(),
        "perceptual_hash": bytes(perceptual).hex(),
    }


def _analyze_pair(payload: dict[str, Any]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    row = payload["row"]
    root = Path(payload["root"])
    motion_config = MotionConfig(**payload["motion_config"])
    descriptor_config = DescriptorConfig(
        active_speed_threshold=motion_config.active_speed_threshold
    )
    source = analyze_video(_resolve_video(row["src_video"], root), motion_config)
    target = analyze_video(_resolve_video(row["tgt_video"], root), motion_config)
    source_fingerprint = _source_content_fingerprint(source.frames_gray)
    split_seed = int(
        (row.get("split_provenance") or {}).get("seed", 260108828)
    )
    source_group_id, source_split = stable_group_split(
        source_video=str(row["src_video"]),
        source_group_key=source_fingerprint["perceptual_hash"],
        seed=split_seed,
    )
    source_descriptor = encode_action_descriptor(
        source.residual_flows,
        source.frame_times,
        source.frames_gray.shape[2],
        global_flows=source.global_flows,
        config=descriptor_config,
        normalize=False,
    )
    target_descriptor = encode_action_descriptor(
        target.residual_flows,
        target.frame_times,
        target.frames_gray.shape[2],
        global_flows=target.global_flows,
        config=descriptor_config,
        normalize=False,
    )
    (
        delta_descriptor,
        descriptor_delta,
        camera_descriptor_delta,
    ) = encode_factorized_action_delta(
        source_descriptor,
        target_descriptor,
        camera_dims=8,
    )
    source_actor = extract_actor_motion_features(
        source,
        active_speed_threshold=motion_config.active_speed_threshold,
    )
    target_actor = extract_actor_motion_features(
        target,
        active_speed_threshold=motion_config.active_speed_threshold,
    )
    feature_score, gate, reasons = _feature_score(
        row=row,
        source=source,
        target=target,
        source_actor=source_actor,
        target_actor=target_actor,
        descriptor_delta=descriptor_delta,
    )
    rule_score = float(row["auto_rule"]["score"])
    combined = 0.45 * rule_score + 0.55 * feature_score
    if gate and combined >= 0.66 and not reasons:
        decision = "auto_keep"
    elif gate and combined >= 0.50:
        decision = "auto_keep" if "weak_actor_localization" not in reasons else "review"
    elif combined >= 0.38 and "invalid_camera_or_cut" not in reasons:
        decision = "review"
    else:
        decision = "auto_reject"
    return {
        "row": row,
        "descriptor": delta_descriptor,
        "split_update": {
            "group_id": source_group_id,
            "split": source_split,
            "split_provenance": {
                "seed": split_seed,
                **source_fingerprint,
            },
        },
        "auto_feature": {
            "schema_version": CASCADE_SCHEMA,
            "stage_version": FEATURE_STAGE_VERSION,
            "source_label": source.label,
            "target_label": target.label,
            "source_metrics": source.metrics.to_dict(),
            "target_metrics": target.metrics.to_dict(),
            "source_actor_features": source_actor.to_dict(),
            "target_actor_features": target_actor.to_dict(),
            "descriptor_delta_norm": descriptor_delta,
            "camera_descriptor_delta_norm": camera_descriptor_delta,
            "feature_score": feature_score,
            "gate_passed": gate,
            "reason_codes": reasons,
        },
        "auto_decision": {
            "decision": decision,
            "heuristic_score": float(np.clip(combined, 0.0, 1.0)),
            "score_kind": "heuristic_not_probability",
            "rule_feature_conflict": abs(rule_score - feature_score),
            "manual_review_pending": True,
        },
    }


def _feature_config(
    args: argparse.Namespace,
) -> tuple[MotionConfig, str, str]:
    config = MotionConfig(
        analysis_frames=args.analysis_frames,
        resize_width=args.resize_width,
        active_speed_threshold=args.active_speed_threshold,
        static_residual_p90=args.static_residual_p90,
        static_active_fraction=args.static_active_fraction,
        max_scene_cuts=args.max_scene_cuts,
    )
    config.validate()
    implementation_digest = _feature_implementation_digest()
    digest = _digest(
        {
            "stage_version": FEATURE_STAGE_VERSION,
            "motion_config": asdict(config),
            "implementation_digest": implementation_digest,
        }
    )
    return config, digest, implementation_digest


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _feature_implementation_digest() -> str:
    """Bind resumability to the exact feature implementation, not a checkout."""

    package_dir = Path(__file__).resolve().parent
    names = (
        "archive.py",
        "cascade.py",
        "descriptor.py",
        "geometry.py",
        "motion_features.py",
        "rules.py",
    )
    return _digest(
        {
            name: _file_digest(package_dir / name)
            for name in names
        }
    )


def run_feature_stage(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser()
    root = args.root.expanduser()
    output_dir = args.output_dir.expanduser()
    stage_dir = output_dir / "features"
    stage_dir.mkdir(parents=True, exist_ok=True)
    motion_config, config_digest, implementation_digest = _feature_config(args)
    shard_index = int(getattr(args, "shard_index", 0))
    num_shards = int(getattr(args, "num_shards", 1))
    total = successful = errors = skipped_shards = retried_shards = 0
    decisions: Counter[str] = Counter()

    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    for shard_id, rows in enumerate(
        _iter_chunks(_iter_jsonl(input_path), args.shard_size)
    ):
        if shard_id % num_shards != shard_index:
            continue
        prefix = f"shard-{shard_id:05d}"
        json_path = stage_dir / f"{prefix}.jsonl"
        archive_path = stage_dir / f"{prefix}.npz"
        done_path = stage_dir / f"{prefix}.done.json"
        shard_input_digest = _digest(
            [(row.get("iid"), row.get("input_digest")) for row in rows]
        )
        retrying_error_shard = False
        output_rows = [dict(row) for row in rows]
        shard_features: list[np.ndarray] = []
        shard_ids: list[str] = []
        process_indices = list(range(len(rows)))
        if done_path.exists() and args.resume:
            marker = json.loads(done_path.read_text(encoding="utf-8"))
            if (
                marker.get("config_digest") != config_digest
                or marker.get("input_digest") != shard_input_digest
            ):
                raise RuntimeError(
                    f"{done_path} exists but config/input digest changed"
                )
            if (
                not json_path.is_file()
                or _file_digest(json_path) != marker.get("json_sha256")
            ):
                raise RuntimeError(f"{done_path} JSON output checksum mismatch")
            expected_archive = marker.get("archive_sha256")
            if expected_archive is not None and (
                not archive_path.is_file()
                or _file_digest(archive_path) != expected_archive
            ):
                raise RuntimeError(f"{done_path} archive checksum mismatch")
            if expected_archive is None and archive_path.exists():
                raise RuntimeError(
                    f"{done_path} unexpectedly has an untracked feature archive"
                )
            marker_errors = int(marker["errors"])
            retry_errors = bool(getattr(args, "retry_errors", True))
            if marker_errors == 0 or not retry_errors:
                skipped_shards += 1
                total += int(marker["rows"])
                successful += int(marker["successful"])
                errors += marker_errors
                decisions.update(marker.get("decisions", {}))
                continue

            output_rows = list(_iter_jsonl(json_path))
            if len(output_rows) != len(rows):
                raise RuntimeError(
                    f"{done_path} row count disagrees with JSON output"
                )
            for index, (input_row, output_row) in enumerate(
                zip(rows, output_rows)
            ):
                if (
                    str(input_row.get("iid")) != str(output_row.get("iid"))
                    or input_row.get("input_digest")
                    != output_row.get("input_digest")
                ):
                    raise RuntimeError(
                        f"{done_path} row {index} no longer matches input"
                    )

            process_indices = [
                index
                for index, row in enumerate(output_rows)
                if (row.get("auto_feature") or {}).get("status") == "error"
            ]
            if len(process_indices) != marker_errors:
                raise RuntimeError(
                    f"{done_path} error count disagrees with JSON output"
                )
            if expected_archive is not None:
                matrix, identifiers, metadata = load_feature_archive(archive_path)
                provenance = metadata.get("provenance") or {}
                if (
                    provenance.get("feature_implementation_digest")
                    != implementation_digest
                ):
                    raise RuntimeError(
                        f"{done_path} feature implementation provenance changed"
                    )
                shard_features = [
                    np.asarray(feature, dtype=np.float32)
                    for feature in matrix
                ]
                shard_ids = [str(identifier) for identifier in identifiers]
            successful_indices: list[int] = []
            for index, row in enumerate(output_rows):
                auto_feature = row.get("auto_feature") or {}
                if auto_feature.get("status") == "error":
                    continue
                feature_index = auto_feature.get("feature_index")
                if not isinstance(feature_index, int):
                    raise RuntimeError(
                        f"{json_path}:{index + 1} lacks a valid feature_index"
                    )
                if not 0 <= feature_index < len(shard_ids):
                    raise RuntimeError(
                        f"{json_path}:{index + 1} feature_index is out of range"
                    )
                if shard_ids[feature_index] != str(row.get("iid")):
                    raise RuntimeError(
                        f"{json_path}:{index + 1} archive id mismatch"
                    )
                successful_indices.append(feature_index)
            if sorted(successful_indices) != list(range(len(shard_ids))):
                raise RuntimeError(
                    f"{done_path} archive indices are incomplete or duplicated"
                )
            if len(shard_ids) != int(marker["successful"]):
                raise RuntimeError(
                    f"{done_path} success count disagrees with feature archive"
                )
            retrying_error_shard = True
            retried_shards += 1
        if any(path.exists() for path in (json_path, archive_path, done_path)):
            if not retrying_error_shard:
                raise FileExistsError(
                    f"incomplete/existing shard {prefix}; use a new output "
                    "directory or remove only that verified incomplete shard"
                )

        payloads = [
            {
                "row": rows[index],
                "root": str(root),
                "motion_config": asdict(motion_config),
            }
            for index in process_indices
        ]
        outcomes: list[dict[str, Any] | Exception] = []
        if args.workers <= 1:
            for payload in payloads:
                try:
                    outcomes.append(_analyze_pair(payload))
                except Exception as error:
                    outcomes.append(error)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_analyze_pair, payload) for payload in payloads]
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except Exception as error:
                        outcomes.append(error)

        for index, outcome in zip(process_indices, outcomes):
            row = rows[index]
            output_row = dict(row)
            if isinstance(outcome, Exception):
                output_row["auto_feature"] = {
                    "schema_version": CASCADE_SCHEMA,
                    "stage_version": FEATURE_STAGE_VERSION,
                    "status": "error",
                    "error_type": type(outcome).__name__,
                    "error": str(outcome),
                }
                output_row["auto_decision"] = {
                    "decision": "review",
                    "heuristic_score": None,
                    "score_kind": "missing_due_to_error",
                    "manual_review_pending": True,
                }
            else:
                output_row.update(outcome.get("split_update", {}))
                output_row["auto_feature"] = outcome["auto_feature"]
                output_row["auto_decision"] = outcome["auto_decision"]
                feature_index = len(shard_features)
                output_row["auto_feature"]["feature_index"] = feature_index
                shard_features.append(
                    np.asarray(outcome["descriptor"], dtype=np.float32)
                )
                shard_ids.append(str(row["iid"]))
            output_rows[index] = output_row

        shard_success = shard_errors = 0
        shard_decisions: Counter[str] = Counter()
        with _atomic_text_writer(json_path) as handle:
            for output_row in output_rows:
                if (output_row.get("auto_feature") or {}).get("status") == "error":
                    shard_errors += 1
                else:
                    shard_success += 1
                decision = str(
                    (output_row.get("auto_decision") or {}).get(
                        "decision",
                        "review",
                    )
                )
                shard_decisions[decision] += 1
                handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

        if shard_features:
            matrix = np.stack(shard_features)
            descriptor_config = DescriptorConfig(
                active_speed_threshold=motion_config.active_speed_threshold
            )
            metadata = build_feature_metadata(
                feature_kind="geometry_action_delta",
                dimension=matrix.shape[1],
                provenance={
                    "descriptor_version": "camera_factorized_hoof_v3",
                    "descriptor_config": asdict(descriptor_config),
                    "motion_backend": "opencv_farneback_partial_affine_v1",
                    "motion_config": asdict(motion_config),
                    "feature_stage_version": FEATURE_STAGE_VERSION,
                    "feature_implementation_digest": implementation_digest,
                },
            )
            save_feature_archive(
                archive_path,
                features=matrix,
                ids=np.asarray(shard_ids),
                metadata=metadata,
            )
        marker = {
            "schema_version": CASCADE_SCHEMA,
            "stage": "features",
            "stage_version": FEATURE_STAGE_VERSION,
            "shard_id": shard_id,
            "rows": len(rows),
            "successful": shard_success,
            "errors": shard_errors,
            "decisions": dict(sorted(shard_decisions.items())),
            "config_digest": config_digest,
            "implementation_digest": implementation_digest,
            "input_digest": shard_input_digest,
            "json_sha256": _file_digest(json_path),
            "archive_sha256": (
                _file_digest(archive_path) if archive_path.exists() else None
            ),
        }
        _write_json(done_path, marker)
        total += len(rows)
        successful += shard_success
        errors += shard_errors
        decisions.update(shard_decisions)
        print(
            f"[motive-cascade features] shard={shard_id} rows={len(rows)} "
            f"ok={shard_success} errors={shard_errors} "
            f"decisions={dict(shard_decisions)}",
            flush=True,
        )

    summary = {
        "schema_version": CASCADE_SCHEMA,
        "stage": "features",
        "stage_version": FEATURE_STAGE_VERSION,
        "input": str(input_path),
        "root": str(root),
        "config_digest": config_digest,
        "implementation_digest": implementation_digest,
        "total": total,
        "successful": successful,
        "errors": errors,
        "skipped_shards": skipped_shards,
        "retried_shards": retried_shards,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "decisions": dict(sorted(decisions.items())),
    }
    summary_name = (
        "feature_summary.json"
        if num_shards == 1
        else f"feature_summary-{shard_index:05d}-of-{num_shards:05d}.json"
    )
    _write_json(output_dir / summary_name, summary)
    return 0 if errors == 0 or bool(getattr(args, "allow_errors", False)) else 2


def export_feature_results(args: argparse.Namespace) -> int:
    feature_dir = args.feature_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    keep_decisions = set(args.keep_decisions)
    counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    total = written = 0
    with _atomic_text_writer(output_dir / "selected.jsonl") as selected_handle, _atomic_text_writer(
        output_dir / "review.jsonl"
    ) as review_handle, _atomic_text_writer(output_dir / "all.jsonl") as all_handle:
        for path in sorted(feature_dir.glob("shard-*.jsonl")):
            for row in _iter_jsonl(path):
                total += 1
                decision = str(row.get("auto_decision", {}).get("decision", "review"))
                counts[decision] += 1
                families = row.get("auto_rule", {}).get("action_families", [])
                family_counts[str(families[0]) if families else "unknown"] += 1
                encoded = json.dumps(row, ensure_ascii=False) + "\n"
                all_handle.write(encoded)
                if decision in keep_decisions:
                    selected_handle.write(encoded)
                    written += 1
                if decision == "review":
                    review_handle.write(encoded)
    qwen_written = 0
    qwen_written_decisions: Counter[str] = Counter()
    qwen_budget = int(getattr(args, "qwen_budget", 0))
    if qwen_budget > 0:
        qwen_decisions = set(
            getattr(args, "qwen_decisions", ["auto_keep", "review"])
        )
        auto_keep_fraction = float(
            getattr(args, "qwen_auto_keep_fraction", 0.20)
        )
        seed = int(getattr(args, "seed", 260108828))
        decision_heaps: dict[
            str,
            list[tuple[float, int, str, str]],
        ] = {decision: [] for decision in qwen_decisions}
        for path in sorted(feature_dir.glob("shard-*.jsonl")):
            for row in _iter_jsonl(path):
                decision = str(
                    row.get("auto_decision", {}).get("decision", "review")
                )
                if decision not in qwen_decisions:
                    continue
                iid = str(row["iid"])
                families = row.get("auto_rule", {}).get("action_families", [])
                family = str(families[0]) if families else "unknown"
                family_rarity = 1.0 / np.sqrt(max(1, family_counts[family]))
                auto = row.get("auto_decision", {})
                score_value = auto.get("heuristic_score")
                score = float(score_value) if score_value is not None else 0.5
                uncertainty = 1.0 - min(abs(score - 0.5) / 0.5, 1.0)
                conflict = min(
                    float(auto.get("rule_feature_conflict") or 0.0),
                    1.0,
                )
                reasons = set(
                    row.get("auto_feature", {}).get("reason_codes", [])
                )
                priority = (
                    (1.0 if decision == "review" else 0.20)
                    + 0.35 * uncertainty
                    + 0.30 * conflict
                    + 0.20 * family_rarity
                    + (0.10 if "weak_actor_localization" in reasons else 0.0)
                )
                tie = int(
                    hashlib.sha256(
                        f"{seed}\0{iid}".encode("utf-8")
                    ).hexdigest()[:16],
                    16,
                )
                item = (
                    float(priority),
                    tie,
                    iid,
                    json.dumps(row, ensure_ascii=False) + "\n",
                )
                heap = decision_heaps[decision]
                if len(heap) < qwen_budget:
                    heapq.heappush(heap, item)
                elif item[:3] > heap[0][:3]:
                    heapq.heapreplace(heap, item)
        ranked = {
            decision: sorted(heap, reverse=True)
            for decision, heap in decision_heaps.items()
        }
        selected_items: list[tuple[float, int, str, str]] = []
        selected_ids: set[str] = set()

        def take(decision: str, count: int) -> None:
            for item in ranked.get(decision, []):
                if len(selected_items) >= qwen_budget or count <= 0:
                    break
                if item[2] in selected_ids:
                    continue
                selected_items.append(item)
                selected_ids.add(item[2])
                count -= 1

        if "auto_keep" in qwen_decisions and auto_keep_fraction > 0.0:
            keep_target = max(
                1,
                int(round(qwen_budget * auto_keep_fraction)),
            )
            take("auto_keep", keep_target)
        if "review" in qwen_decisions:
            take("review", qwen_budget - len(selected_items))
        remaining = sorted(
            (
                item
                for values in ranked.values()
                for item in values
                if item[2] not in selected_ids
            ),
            reverse=True,
        )
        for item in remaining:
            if len(selected_items) >= qwen_budget:
                break
            selected_items.append(item)
            selected_ids.add(item[2])
        queue_path = output_dir / "qwen_queue.jsonl"
        with _atomic_text_writer(queue_path) as qwen_handle:
            for _priority, _tie, _iid, encoded in sorted(
                selected_items,
                reverse=True,
            ):
                qwen_handle.write(encoded)
                qwen_written += 1
                selected_row = json.loads(encoded)
                qwen_written_decisions[
                    str(
                        selected_row.get("auto_decision", {}).get(
                            "decision",
                            "review",
                        )
                    )
                ] += 1
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": CASCADE_SCHEMA,
            "stage": "export",
            "total": total,
            "selected": written,
            "keep_decisions": sorted(keep_decisions),
            "decisions": dict(sorted(counts.items())),
            "action_families": dict(sorted(family_counts.items())),
            "qwen_budget": qwen_budget,
            "qwen_written": qwen_written,
            "qwen_written_decisions": dict(
                sorted(qwen_written_decisions.items())
            ),
            "qwen_decisions": sorted(
                set(getattr(args, "qwen_decisions", []))
            ),
            "qwen_auto_keep_fraction": (
                float(getattr(args, "qwen_auto_keep_fraction", 0.20))
                if qwen_budget > 0
                else None
            ),
            "qwen_queue_policy": (
                "review_then_uncertainty_conflict_rarity_v1"
                if qwen_budget > 0
                else None
            ),
        },
    )
    print(
        f"[motive-cascade export] total={total} selected={written} output={output_dir}",
        flush=True,
    )
    return 0


def _load_qwen_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    results: dict[str, dict[str, Any]] = {}
    expanded = path.expanduser()
    paths = sorted(expanded.glob("*.jsonl")) if expanded.is_dir() else [expanded]
    if not paths:
        raise RuntimeError(f"no Qwen JSONL results found in {expanded}")
    for result_path in paths:
        for row in _iter_jsonl(result_path):
            iid = str(row["iid"])
            if iid in results:
                raise ValueError(
                    f"duplicate Qwen result for iid={iid} in {expanded}"
                )
            results[iid] = row
    return results


def _ordinal_score(confidence: str, high: float, medium: float, low: float) -> float:
    return {"high": high, "medium": medium, "low": low}.get(confidence, medium)


def _qwen_visual_trust(record: dict[str, Any] | None) -> str:
    if record is None:
        return "not_available"
    if record.get("status") != "ok":
        return "status_error"
    if (
        record.get("observation_validated_from") != "original"
        or record.get("result_validated_from") != "original"
    ):
        return "manual_review_required"
    observation = record.get("observation")
    result = record.get("result")
    if not isinstance(observation, dict) or not isinstance(result, dict):
        return "original_malformed"
    try:
        _validate_qwen_observation(observation)
        _validate_qwen_visual(result, observation=observation)
    except (TypeError, ValueError):
        return "original_semantic_conflict"
    observation_digest = record.get("observation_digest")
    result_digest = record.get("result_digest")
    if not isinstance(observation_digest, str) or not isinstance(
        result_digest, str
    ):
        return "original_digest_missing"
    if observation_digest != _qwen_object_digest(observation):
        return "original_digest_mismatch"
    if result_digest != _qwen_object_digest(result):
        return "original_digest_mismatch"
    return "original_validated"


def _qwen_hard_reject_supported(record: dict[str, Any] | None) -> bool:
    """Require unusually strong blind evidence before automatic rejection."""

    if _qwen_visual_trust(record) != "original_validated":
        return False
    assert record is not None
    observation = record["observation"]
    result = record["result"]
    verdict = str(result["verdict"])
    source_motion = str(observation["source_actor_motion"])
    target_motion = str(observation["target_actor_motion"])
    if verdict in {"endpoint_only", "appearance_only", "static"}:
        # A moving source and still target can be a valid suppression edit.
        return source_motion == "none" and target_motion == "none"
    if verdict == "camera_motion":
        return (
            observation["camera_dominance"] == "high"
            and target_motion == "none"
        )
    if verdict == "background_motion":
        return (
            observation["background_dominance"] == "high"
            and target_motion == "none"
        )
    if verdict == "artifact":
        return (
            observation["artifact_level"] == "high"
            or observation["preservation_quality"] == "poor"
        )
    if verdict == "instruction_mismatch":
        return target_motion in {"clear", "weak"}
    return False


def _qwen_score(record: dict[str, Any] | None, mode: str) -> float | None:
    if not record or record.get("status") != "ok":
        return None
    if mode == "visual" and _qwen_visual_trust(record) != "original_validated":
        return 0.45
    result = record["result"]
    verdict = str(result.get("verdict"))
    confidence = str(result.get("confidence", "medium"))
    if mode == "text":
        if verdict in {"temporal_action", "motion_suppression"}:
            return _ordinal_score(confidence, 0.95, 0.80, 0.62)
        if verdict == "endpoint_only":
            return _ordinal_score(confidence, 0.08, 0.20, 0.35)
        if verdict == "non_action":
            return _ordinal_score(confidence, 0.02, 0.12, 0.30)
        return 0.45
    if verdict in {"valid_action", "valid_suppression"}:
        return _ordinal_score(confidence, 0.98, 0.85, 0.68)
    if verdict in {
        "endpoint_only",
        "appearance_only",
        "camera_motion",
        "background_motion",
        "static",
        "instruction_mismatch",
        "artifact",
    }:
        return _ordinal_score(confidence, 0.02, 0.14, 0.32)
    return 0.45


def fuse_results(args: argparse.Namespace) -> int:
    feature_dir = args.feature_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    text_results = _load_qwen_results(args.qwen_text)
    visual_results = _load_qwen_results(args.qwen_visual)
    counts: Counter[str] = Counter()
    visual_trust_counts: Counter[str] = Counter()
    total = 0
    with _atomic_text_writer(output_dir / "all.jsonl") as all_handle, _atomic_text_writer(
        output_dir / "selected.jsonl"
    ) as selected_handle, _atomic_text_writer(output_dir / "review.jsonl") as review_handle, _atomic_text_writer(
        output_dir / "rejected.jsonl"
    ) as reject_handle:
        for path in sorted(feature_dir.glob("shard-*.jsonl")):
            for row in _iter_jsonl(path):
                total += 1
                iid = str(row["iid"])
                rule_score = float(row["auto_rule"]["score"])
                feature_value = row.get("auto_feature", {}).get("feature_score")
                feature_score = (
                    float(feature_value) if feature_value is not None else 0.35
                )
                text_record = text_results.get(iid)
                visual_record = visual_results.get(iid)
                for source, record in (
                    ("text", text_record),
                    ("visual", visual_record),
                ):
                    if (
                        record is not None
                        and record.get("input_digest") != row.get("input_digest")
                    ):
                        raise ValueError(
                            f"stale Qwen {source} result for iid={iid}: "
                            "input digest differs"
                        )
                text_score = _qwen_score(text_record, "text")
                visual_score = _qwen_score(visual_record, "visual")
                components = [
                    ("rule", rule_score, 0.35),
                    ("feature", feature_score, 0.45),
                ]
                if text_score is not None:
                    components = [
                        ("rule", rule_score, 0.25),
                        ("feature", feature_score, 0.40),
                        ("qwen_text", text_score, 0.35),
                    ]
                if visual_score is not None:
                    components = [
                        ("rule", rule_score, 0.15),
                        ("feature", feature_score, 0.35),
                        ("qwen_visual", visual_score, 0.50),
                    ]
                    if text_score is not None:
                        components = [
                            ("rule", rule_score, 0.15),
                            ("feature", feature_score, 0.35),
                            ("qwen_text", text_score, 0.15),
                            ("qwen_visual", visual_score, 0.35),
                        ]
                weight_sum = sum(weight for _, _, weight in components)
                final_score = sum(
                    value * weight for _, value, weight in components
                ) / weight_sum
                gate_passed = bool(
                    row.get("auto_feature", {}).get("gate_passed", False)
                )
                base_decision = str(
                    row.get("auto_decision", {}).get("decision", "review")
                )
                visual_result = (
                    visual_record.get("result", {}) if visual_record else {}
                )
                visual_verdict = str(visual_result.get("verdict", ""))
                visual_confidence = str(visual_result.get("confidence", ""))
                visual_trust = _qwen_visual_trust(visual_record)
                visual_hard_reject_supported = (
                    _qwen_hard_reject_supported(visual_record)
                )
                visual_trust_counts[visual_trust] += 1
                if (
                    visual_record is not None
                    and visual_trust != "original_validated"
                ):
                    decision = "review"
                elif (
                    visual_verdict
                    in {
                        "endpoint_only",
                        "appearance_only",
                        "camera_motion",
                        "background_motion",
                        "static",
                        "instruction_mismatch",
                        "artifact",
                    }
                    and visual_confidence == "high"
                    and visual_hard_reject_supported
                ):
                    decision = "auto_reject"
                elif visual_verdict in {
                    "endpoint_only",
                    "appearance_only",
                    "camera_motion",
                    "background_motion",
                    "static",
                    "instruction_mismatch",
                    "artifact",
                }:
                    # A negative pseudo-label without strong blind evidence is
                    # useful for routing, never for silently dropping data.
                    decision = "review"
                elif (
                    gate_passed
                    and final_score >= args.keep_threshold
                    and base_decision != "auto_reject"
                ):
                    decision = "auto_keep"
                elif final_score >= args.review_threshold:
                    decision = "review"
                else:
                    decision = "auto_reject"
                qwen_evidence = {
                    "text": text_record,
                    "visual": visual_record,
                }
                action_signature = None
                for record in (visual_record, text_record):
                    if record and record.get("status") == "ok":
                        if (
                            record is visual_record
                            and visual_trust != "original_validated"
                        ):
                            continue
                        candidate = str(
                            record.get("result", {}).get(
                                "action_signature",
                                "",
                            )
                        ).strip()
                        if candidate and candidate != "unknown":
                            action_signature = candidate
                            break
                if action_signature is None:
                    families = row["auto_rule"].get("action_families", [])
                    action_signature = "+".join(families) if families else "unknown"
                row["qwen_evidence"] = qwen_evidence
                row["final_triage"] = {
                    "decision": decision,
                    "heuristic_score": float(final_score),
                    "score_kind": "heuristic_not_probability",
                    "components": {
                        name: {"value": value, "weight": weight}
                        for name, value, weight in components
                    },
                    "action_signature": action_signature,
                    "manual_review_pending": True,
                    "calibrator_id": None,
                    "qwen_visual_trust": visual_trust,
                    "qwen_hard_reject_supported": (
                        visual_hard_reject_supported
                    ),
                }
                counts[decision] += 1
                encoded = json.dumps(row, ensure_ascii=False) + "\n"
                all_handle.write(encoded)
                if decision == "auto_keep":
                    selected_handle.write(encoded)
                elif decision == "review":
                    review_handle.write(encoded)
                else:
                    reject_handle.write(encoded)
    summary = {
        "schema_version": CASCADE_SCHEMA,
        "stage": "fusion",
        "total": total,
        "decisions": dict(sorted(counts.items())),
        "qwen_text": str(args.qwen_text) if args.qwen_text else None,
        "qwen_visual": str(args.qwen_visual) if args.qwen_visual else None,
        "keep_threshold": args.keep_threshold,
        "review_threshold": args.review_threshold,
        "score_kind": "heuristic_not_probability",
        "manual_review_pending": True,
        "qwen_visual_trust": dict(sorted(visual_trust_counts.items())),
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        f"[motive-cascade fuse] total={total} decisions={dict(counts)} "
        f"output={output_dir}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rule/feature cascade for action-edit candidate mining."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample")
    sample.add_argument("--input", required=True, type=Path)
    sample.add_argument("--output", required=True, type=Path)
    sample.add_argument("--sample-size", type=int, default=2000)
    sample.add_argument("--max-per-bucket", type=int, default=100)
    sample.add_argument("--seed", type=int, default=260108828)
    sample.add_argument("--overwrite", action="store_true")
    sample.set_defaults(handler=build_balanced_sample)

    rules = subparsers.add_parser("rules")
    rules.add_argument("--dataset-root", required=True, type=Path)
    rules.add_argument("--output-dir", required=True, type=Path)
    rules.add_argument("--sample-size", type=int)
    rules.add_argument("--seed", type=int, default=260108828)
    rules.add_argument("--min-score", type=float, default=0.40)
    rules.add_argument("--include-tiers", nargs="+", default=["high", "possible"])
    rules.add_argument("--reject-audit-fraction", type=float, default=0.01)
    rules.add_argument(
        "--use-edited-caption-as-rule-evidence",
        action="store_true",
        help=(
            "Opt in to target-caption evidence. Disabled by default to avoid "
            "answer leakage during curation."
        ),
    )
    rules.add_argument("--continue-on-error", action="store_true")
    rules.add_argument("--overwrite", action="store_true")
    rules.set_defaults(handler=build_rule_manifest)

    features = subparsers.add_parser("features")
    features.add_argument("--input", required=True, type=Path)
    features.add_argument("--root", required=True, type=Path)
    features.add_argument("--output-dir", required=True, type=Path)
    features.add_argument("--workers", type=int, default=1)
    features.add_argument("--shard-size", type=int, default=256)
    features.add_argument("--shard-index", type=int, default=0)
    features.add_argument("--num-shards", type=int, default=1)
    features.add_argument("--resume", action="store_true")
    features.add_argument(
        "--no-retry-errors",
        action="store_false",
        dest="retry_errors",
        help=(
            "Keep completed error rows instead of retrying them on --resume. "
            "By default only prior error rows are retried."
        ),
    )
    features.set_defaults(retry_errors=True)
    features.add_argument(
        "--allow-errors",
        action="store_true",
        help=(
            "Return success after recording errors so an explicitly configured "
            "downstream audit/finalizer may run."
        ),
    )
    features.add_argument("--analysis-frames", type=int, default=20)
    features.add_argument("--resize-width", type=int, default=160)
    features.add_argument("--active-speed-threshold", type=float, default=0.005)
    features.add_argument("--static-residual-p90", type=float, default=0.003)
    features.add_argument("--static-active-fraction", type=float, default=0.025)
    features.add_argument("--max-scene-cuts", type=int, default=0)
    features.set_defaults(handler=run_feature_stage)

    export = subparsers.add_parser("export")
    export.add_argument("--feature-dir", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument("--keep-decisions", nargs="+", default=["auto_keep"])
    export.add_argument(
        "--qwen-budget",
        type=int,
        default=0,
        help="If positive, also write a bounded qwen_queue.jsonl.",
    )
    export.add_argument(
        "--qwen-decisions",
        nargs="+",
        default=["auto_keep", "review"],
    )
    export.add_argument(
        "--qwen-auto-keep-fraction",
        type=float,
        default=0.20,
        help="Reserve this fraction of the Qwen budget for auto-keep auditing.",
    )
    export.add_argument("--seed", type=int, default=260108828)
    export.set_defaults(handler=export_feature_results)

    fuse = subparsers.add_parser("fuse")
    fuse.add_argument("--feature-dir", required=True, type=Path)
    fuse.add_argument("--output-dir", required=True, type=Path)
    fuse.add_argument("--qwen-text", type=Path)
    fuse.add_argument("--qwen-visual", type=Path)
    fuse.add_argument("--keep-threshold", type=float, default=0.68)
    fuse.add_argument("--review-threshold", type=float, default=0.42)
    fuse.set_defaults(handler=fuse_results)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= getattr(args, "reject_audit_fraction", 0.0) <= 1.0:
        raise ValueError("--reject-audit-fraction must be in [0, 1]")
    if getattr(args, "shard_size", 1) <= 0:
        raise ValueError("--shard-size must be positive")
    if getattr(args, "num_shards", 1) <= 0:
        raise ValueError("--num-shards must be positive")
    if getattr(args, "qwen_budget", 0) < 0:
        raise ValueError("--qwen-budget must be non-negative")
    qwen_keep_fraction = getattr(args, "qwen_auto_keep_fraction", 0.20)
    if not 0.0 <= qwen_keep_fraction <= 1.0:
        raise ValueError("--qwen-auto-keep-fraction must be in [0, 1]")
    sample_size = getattr(args, "sample_size", 1)
    if sample_size is not None and sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if getattr(args, "max_per_bucket", 1) <= 0:
        raise ValueError("--max-per-bucket must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
