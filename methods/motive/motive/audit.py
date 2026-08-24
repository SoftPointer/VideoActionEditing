"""CLI for read-only motion screening and geometry descriptor extraction."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .archive import build_feature_metadata, save_feature_archive
from .descriptor import (
    DescriptorConfig,
    encode_action_delta,
    encode_action_descriptor,
)
from .geometry import MotionConfig, analyze_video
from .semantics import classify_instruction


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
}
DEFAULT_TARGET_KEYS = (
    "tgt_video",
    "target_video",
    "edited_video",
    "video",
    "video_path",
    "path",
)
DEFAULT_SOURCE_KEYS = (
    "src_video",
    "source_video",
    "original_video",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("data", "rows", "samples", "items"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise ValueError(f"{path} must contain a list of objects")
        return list(value)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"unsupported manifest format: {path}")


def _directory_rows(path: Path) -> list[dict[str, Any]]:
    videos = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS
    )
    return [
        {
            "iid": index,
            "video": str(video),
        }
        for index, video in enumerate(videos)
    ]


def _first_value(row: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _sample_id(row: dict[str, Any], index: int) -> str:
    for key in ("iid", "id", "sample_id", "uid", "name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return str(index)


def _analyze_one(payload: dict[str, Any]) -> dict[str, Any]:
    # Avoid CPU oversubscription when multiple process workers are used.
    cv2.setNumThreads(1)
    config = MotionConfig(**payload["config"])
    descriptor_config = DescriptorConfig(
        active_speed_threshold=config.active_speed_threshold
    )
    target = analyze_video(payload["target_path"], config)
    target_descriptor = encode_action_descriptor(
        target.residual_flows,
        target.frame_times,
        target.frames_gray.shape[2],
        global_flows=target.global_flows,
        config=descriptor_config,
    )
    semantics = classify_instruction(payload.get("instruction", ""))
    result: dict[str, Any] = {
        "id": payload["id"],
        "target_path": payload["target_path"],
        "target_label": target.label,
        "target_metrics": target.metrics.to_dict(),
        "feature": target_descriptor,
        "paired": False,
        "selected": target.label == "dynamic_object",
        "instruction": payload.get("instruction", ""),
        "instruction_semantics": semantics.to_dict(),
    }

    source_path = payload.get("source_path")
    if source_path:
        source = analyze_video(source_path, config)
        source_descriptor = encode_action_descriptor(
            source.residual_flows,
            source.frame_times,
            source.frames_gray.shape[2],
            global_flows=source.global_flows,
            config=descriptor_config,
        )
        raw_delta = target_descriptor - source_descriptor
        delta_norm = float(np.linalg.norm(raw_delta))
        delta_descriptor = encode_action_delta(
            source_descriptor,
            target_descriptor,
        )
        invalid_labels = {"cut_or_decode_artifact", "camera_only"}
        selected = (
            source.label not in invalid_labels
            and target.label not in invalid_labels
            and "dynamic_object" in {source.label, target.label}
            and delta_norm >= payload["min_descriptor_delta"]
        )
        allowed_semantics = payload.get("semantic_classes")
        if allowed_semantics:
            selected = selected and semantics.label in allowed_semantics
        minimum_action_speed = float(payload.get("min_action_residual_p90", 0.0))
        minimum_action_ratio = float(payload.get("min_action_motion_ratio", 0.0))
        minimum_action_gain = float(payload.get("min_action_motion_gain", 0.0))
        minimum_suppression_speed = float(
            payload.get("min_suppression_residual_p90", minimum_action_speed)
        )
        minimum_suppression_ratio = float(
            payload.get("min_suppression_motion_ratio", 1.0)
        )
        source_speed = source.metrics.residual_speed_p90
        target_speed = target.metrics.residual_speed_p90
        if semantics.label == "continuous_action":
            selected = selected and target_speed >= minimum_action_speed
            if minimum_action_ratio > 0.0 or minimum_action_gain > 0.0:
                selected = selected and (
                    target_speed >= source_speed * minimum_action_ratio
                    or target_speed - source_speed >= minimum_action_gain
                )
        elif semantics.label == "motion_suppression":
            selected = (
                selected
                and source_speed >= minimum_suppression_speed
                and source_speed >= target_speed * minimum_suppression_ratio
            )
        result.update(
            {
                "source_path": source_path,
                "source_label": source.label,
                "source_metrics": source.metrics.to_dict(),
                "descriptor_delta_norm": delta_norm,
                "feature": delta_descriptor,
                "paired": True,
                "selected": selected,
            }
        )
    return result


def _payloads(
    rows: list[dict[str, Any]],
    *,
    root: Path,
    target_key: str | None,
    source_key: str | None,
    config: MotionConfig,
    min_descriptor_delta: float,
    semantic_classes: tuple[str, ...] | None,
    min_action_residual_p90: float,
    min_action_motion_ratio: float,
    min_action_motion_gain: float,
    min_suppression_residual_p90: float,
    min_suppression_motion_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    originals: list[dict[str, Any]] = []
    target_keys = (target_key,) if target_key else DEFAULT_TARGET_KEYS
    source_keys = (source_key,) if source_key else DEFAULT_SOURCE_KEYS
    for index, row in enumerate(rows):
        target_value = _first_value(row, target_keys)
        if target_value is None:
            raise ValueError(
                f"row {index} has no target video key among {target_keys}"
            )
        source_value = _first_value(row, source_keys)
        instruction = str(
            _first_value(
                row,
                ("instruction_en", "prompt", "instruction", "edit_prompt"),
            )
            or ""
        )
        target_path = _resolve_path(target_value, root)
        source_path = (
            None if source_value is None else _resolve_path(source_value, root)
        )
        payloads.append(
            {
                "id": _sample_id(row, index),
                "target_path": str(target_path),
                "source_path": None if source_path is None else str(source_path),
                "config": asdict(config),
                "min_descriptor_delta": min_descriptor_delta,
                "instruction": instruction,
                "semantic_classes": semantic_classes,
                "min_action_residual_p90": min_action_residual_p90,
                "min_action_motion_ratio": min_action_motion_ratio,
                "min_action_motion_gain": min_action_motion_gain,
                "min_suppression_residual_p90": min_suppression_residual_p90,
                "min_suppression_motion_ratio": min_suppression_motion_ratio,
            }
        )
        originals.append(dict(row))
    return payloads, originals


def _ordered_map(
    payloads: list[dict[str, Any]],
    workers: int,
) -> Iterable[dict[str, Any] | Exception]:
    if workers <= 1:
        for payload in payloads:
            try:
                yield _analyze_one(payload)
            except Exception as error:  # recorded per sample by design
                yield error
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Keep only a small bounded window of decoded-video jobs in flight.
        # Submitting a future for every item is prohibitive on large manifests.
        iterator = iter(payloads)
        pending = deque()
        for _ in range(min(len(payloads), workers * 2)):
            pending.append(executor.submit(_analyze_one, next(iterator)))
        while pending:
            future = pending.popleft()
            try:
                yield future.result()
            except Exception as error:  # recorded per sample by design
                yield error
            try:
                payload = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(_analyze_one, payload))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only stage-0 motion audit. It writes manifests and descriptors "
            "but never moves, deletes, or rewrites source videos."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        help="Base directory for relative manifest paths (default: manifest parent).",
    )
    parser.add_argument("--target-key")
    parser.add_argument("--source-key")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--analysis-frames", type=int, default=32)
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--active-speed-threshold", type=float, default=0.005)
    parser.add_argument("--static-residual-p90", type=float, default=0.003)
    parser.add_argument("--static-active-fraction", type=float, default=0.025)
    parser.add_argument("--camera-explained-ratio", type=float, default=0.70)
    parser.add_argument("--max-scene-cut-ratio", type=float, default=0.15)
    parser.add_argument("--max-scene-cuts", type=int, default=0)
    parser.add_argument("--min-descriptor-delta", type=float, default=0.10)
    parser.add_argument(
        "--semantic-classes",
        nargs="+",
        help=(
            "For paired data, only select these instruction classes. "
            "Recommended action pool: continuous_action motion_suppression."
        ),
    )
    parser.add_argument("--min-action-residual-p90", type=float, default=0.0)
    parser.add_argument(
        "--min-action-motion-ratio",
        type=float,
        default=0.0,
        help="Optional target/source speed ratio (reversed for motion_suppression).",
    )
    parser.add_argument("--min-action-motion-gain", type=float, default=0.0)
    parser.add_argument(
        "--min-suppression-residual-p90",
        type=float,
        default=0.003,
    )
    parser.add_argument(
        "--min-suppression-motion-ratio",
        type=float,
        default=1.10,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser()
    if input_path.is_dir():
        rows = _directory_rows(input_path)
        default_root = input_path
    elif input_path.is_file():
        rows = _read_rows(input_path)
        default_root = input_path.parent
    else:
        raise FileNotFoundError(input_path)
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(rows)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no input videos were discovered")

    root = (args.root or default_root).expanduser()
    config = MotionConfig(
        analysis_frames=args.analysis_frames,
        resize_width=args.resize_width,
        active_speed_threshold=args.active_speed_threshold,
        static_residual_p90=args.static_residual_p90,
        static_active_fraction=args.static_active_fraction,
        camera_explained_ratio=args.camera_explained_ratio,
        max_scene_cut_ratio=args.max_scene_cut_ratio,
        max_scene_cuts=args.max_scene_cuts,
    )
    config.validate()
    payloads, originals = _payloads(
        rows,
        root=root,
        target_key=args.target_key,
        source_key=args.source_key,
        config=config,
        min_descriptor_delta=args.min_descriptor_delta,
        semantic_classes=(
            None if not args.semantic_classes else tuple(args.semantic_classes)
        ),
        min_action_residual_p90=args.min_action_residual_p90,
        min_action_motion_ratio=args.min_action_motion_ratio,
        min_action_motion_gain=args.min_action_motion_gain,
        min_suppression_residual_p90=args.min_suppression_residual_p90,
        min_suppression_motion_ratio=args.min_suppression_motion_ratio,
    )
    paired_flags = [payload.get("source_path") is not None for payload in payloads]
    if any(paired_flags) and not all(paired_flags):
        raise ValueError(
            "a single audit archive cannot mix paired delta descriptors with "
            "single-video descriptors; split the manifest by representation kind"
        )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    feature_ids: list[str] = []
    labels: Counter[str] = Counter()
    semantic_labels: Counter[str] = Counter()
    errors = 0

    for index, (original, payload, outcome) in enumerate(
        zip(originals, payloads, _ordered_map(payloads, args.workers)),
        start=1,
    ):
        row = dict(original)
        if isinstance(outcome, Exception):
            errors += 1
            row["motive_audit"] = {
                "status": "error",
                "error_type": type(outcome).__name__,
                "error": str(outcome),
                "target_path": payload["target_path"],
            }
        else:
            feature = outcome.pop("feature")
            selected = bool(outcome["selected"])
            label = (
                f"{outcome.get('source_label', 'single')}->{outcome['target_label']}"
                if outcome["paired"]
                else outcome["target_label"]
            )
            labels[label] += 1
            semantic_labels[
                str(outcome["instruction_semantics"]["label"])
            ] += 1
            row["motive_audit"] = {"status": "ok", **outcome}
            output_feature_index = len(features)
            row["motive_audit"]["feature_index"] = output_feature_index
            features.append(feature)
            feature_ids.append(outcome["id"])
            if selected:
                selected_rows.append(row)
        output_rows.append(row)
        if index % 50 == 0 or index == len(payloads):
            print(
                f"[motive-audit] processed={index}/{len(payloads)} "
                f"selected={len(selected_rows)} errors={errors}",
                flush=True,
            )

    _write_jsonl(output_dir / "audit.jsonl", output_rows)
    _write_jsonl(output_dir / "selected.jsonl", selected_rows)
    if features:
        feature_matrix = np.stack(features)
        feature_kind = (
            "geometry_action_delta"
            if any(payload.get("source_path") for payload in payloads)
            else "geometry_action_descriptor"
        )
        descriptor_config = DescriptorConfig(
            active_speed_threshold=config.active_speed_threshold
        )
        archive_metadata = build_feature_metadata(
            feature_kind=feature_kind,
            dimension=feature_matrix.shape[1],
            provenance={
                "descriptor_version": "camera_compensated_hoof_v2",
                "descriptor_config": asdict(descriptor_config),
                "motion_backend": "opencv_farneback_partial_affine_v1",
                "motion_config": asdict(config),
                "speed_units": "frame_width_per_second",
            },
        )
        save_feature_archive(
            output_dir / "descriptors.npz",
            features=feature_matrix,
            ids=np.asarray(feature_ids),
            metadata=archive_metadata,
        )
    summary = {
        "input": str(input_path),
        "root": str(root),
        "total": len(payloads),
        "successful": len(features),
        "selected": len(selected_rows),
        "errors": errors,
        "labels": dict(sorted(labels.items())),
        "instruction_semantics": dict(sorted(semantic_labels.items())),
        "selection_semantic_classes": args.semantic_classes,
        "config": asdict(config),
        "descriptor_semantics": (
            "paired target-minus-source geometry action delta"
            if any(payload.get("source_path") for payload in payloads)
            else "single-video camera-compensated geometry action descriptor"
        ),
        "archive_compatibility_digest": (
            archive_metadata["compatibility_digest"] if features else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[motive-audit] outputs={output_dir}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
