"""Evaluate action-edit inference with camera-compensated motion features.

The evaluator consumes one canonical ``samples.json`` written by
``lucy/infer_compare.py`` and one output directory per experimental arm.  Each
arm must contain::

    sample_000/source.mp4
    sample_000/target.mp4
    sample_000/step_000100.mp4

and so on for every canonical sample.  Missing or empty inputs fail the whole
evaluation before any output is replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .descriptor import DescriptorConfig, encode_action_delta, encode_action_descriptor
from .geometry import MotionAnalysis, MotionConfig, analyze_video
from .motion_features import ActorMotionFeatures, extract_actor_motion_features


SCHEMA = "motive-action-inference-evaluation-v1"
ALL_SPLIT = "__all__"
ARM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    directory: Path


@dataclass(frozen=True)
class VideoFeatures:
    analysis: MotionAnalysis
    descriptor: np.ndarray
    actor: ActorMotionFeatures


# direction is used only for improvement-oriented paired comparisons.
# ``None`` means that the raw paired difference is descriptive, not an
# automatic quality improvement.
METRICS: dict[str, tuple[str | None, str | None]] = {
    "target_descriptor_cosine": ("higher", "target_descriptor_cosine_valid"),
    "source_descriptor_cosine": (None, "source_descriptor_cosine_valid"),
    "delta_descriptor_cosine": ("higher", "delta_descriptor_cosine_valid"),
    "motion_magnitude_log_ratio_error": ("lower", None),
    "actor_likeness": ("higher", None),
    "actor_likeness_abs_error": ("lower", None),
    "camera_ratio": (None, None),
    "camera_ratio_abs_error": ("lower", None),
    "motion_label_match": ("higher", None),
    "static_dynamic_match": ("higher", "static_dynamic_match_valid"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_samples(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical samples JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"canonical samples JSON must be a non-empty list: {path}")
    if any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"every canonical sample must be a JSON object: {path}")
    return payload


def _parse_arm(value: str) -> ArmSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--arm must be NAME=DIRECTORY")
    name, raw_directory = value.split("=", 1)
    name = name.strip()
    raw_directory = raw_directory.strip()
    if not ARM_NAME_PATTERN.fullmatch(name):
        raise argparse.ArgumentTypeError(
            f"invalid arm name {name!r}; use letters, digits, '.', '_' or '-'"
        )
    if not raw_directory:
        raise argparse.ArgumentTypeError("arm directory must not be empty")
    return ArmSpec(name=name, directory=Path(raw_directory).expanduser())


def _validate_basename(value: str, option: str) -> str:
    path = Path(value)
    if not value or path.name != value or value in {".", ".."}:
        raise ValueError(f"{option} must be a filename, not a path: {value!r}")
    return value


def _nested_value(row: dict[str, Any], dotted_key: str) -> Any:
    current: Any = row
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _sample_split(row: dict[str, Any], split_key: str) -> str:
    value = _nested_value(row, split_key)
    if value is None or str(value).strip() == "":
        return "unspecified"
    if isinstance(value, (dict, list)):
        value = _canonical_json(value)
    result = str(value).strip()
    if result == ALL_SPLIT:
        raise ValueError(
            f"sample split {ALL_SPLIT!r} is reserved for aggregate summaries"
        )
    return result


def _sample_id(row: dict[str, Any], sample_index: int) -> str:
    for key in ("iid", "id", "sample_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"sample_{sample_index:03d}"


def _required_paths(
    arms: Sequence[ArmSpec],
    sample_count: int,
    prediction_name: str,
) -> list[tuple[str, Path]]:
    required: list[tuple[str, Path]] = []
    for arm in arms:
        required.append((f"arm directory {arm.name}", arm.directory))
        for sample_index in range(sample_count):
            sample_dir = arm.directory / f"sample_{sample_index:03d}"
            for filename in ("source.mp4", "target.mp4", prediction_name):
                required.append(
                    (
                        f"{arm.name} sample {sample_index:03d} {filename}",
                        sample_dir / filename,
                    )
                )
    return required


def _preflight(
    arms: Sequence[ArmSpec],
    samples: Sequence[dict[str, Any]],
    canonical_samples_json: Path,
    prediction_name: str,
    *,
    verify_arm_samples: bool,
) -> None:
    if not arms:
        raise ValueError("at least one --arm NAME=DIRECTORY is required")
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        raise ValueError(f"duplicate arm names: {duplicates}")

    failures: list[str] = []
    for label, path in _required_paths(arms, len(samples), prediction_name):
        if label.startswith("arm directory "):
            if not path.is_dir():
                failures.append(f"{label}: not a directory: {path}")
        elif not path.is_file():
            failures.append(f"{label}: missing: {path}")
        elif path.stat().st_size <= 0:
            failures.append(f"{label}: empty: {path}")

    canonical_payload = _canonical_json(samples)
    if verify_arm_samples:
        for arm in arms:
            arm_samples = arm.directory / "samples.json"
            if not arm_samples.exists():
                continue
            if not arm_samples.is_file():
                failures.append(f"{arm.name} samples.json is not a file: {arm_samples}")
                continue
            try:
                payload = json.loads(arm_samples.read_text(encoding="utf-8"))
            except Exception as error:
                failures.append(f"{arm.name} samples.json cannot be parsed: {error}")
                continue
            if _canonical_json(payload) != canonical_payload:
                failures.append(
                    f"{arm.name} samples.json differs from canonical {canonical_samples_json}"
                )

    if failures:
        preview = "\n".join(f"  - {message}" for message in failures[:50])
        remainder = len(failures) - min(len(failures), 50)
        suffix = f"\n  - ... and {remainder} more" if remainder else ""
        raise RuntimeError(
            "inference evaluation preflight failed; no outputs were written:\n"
            f"{preview}{suffix}"
        )


def _cosine(first: np.ndarray, second: np.ndarray, eps: float) -> tuple[float, bool]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not math.isfinite(denominator) or denominator <= eps:
        return 0.0, False
    value = float(np.dot(first, second) / denominator)
    if not math.isfinite(value):
        raise RuntimeError("non-finite descriptor cosine")
    return float(np.clip(value, -1.0, 1.0)), True


def _motion_class(label: str) -> str:
    if label == "dynamic_object":
        return "dynamic"
    if label == "static":
        return "static"
    return label


def _analyze(
    path: Path,
    motion_config: MotionConfig,
    descriptor_config: DescriptorConfig,
) -> VideoFeatures:
    analysis = analyze_video(path, motion_config)
    descriptor = encode_action_descriptor(
        analysis.residual_flows,
        analysis.frame_times,
        int(analysis.frames_gray.shape[2]),
        global_flows=analysis.global_flows,
        config=descriptor_config,
    )
    if not np.all(np.isfinite(descriptor)):
        raise RuntimeError(f"non-finite action descriptor: {path}")
    actor = extract_actor_motion_features(
        analysis,
        active_speed_threshold=descriptor_config.active_speed_threshold,
        minimum_frame_support=descriptor_config.minimum_active_fraction,
    )
    scalar_values = list(analysis.metrics.to_dict().values()) + list(
        actor.to_dict().values()
    )
    for value in scalar_values:
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite motion feature in {path}")
    return VideoFeatures(analysis=analysis, descriptor=descriptor, actor=actor)


def _evaluate_sample(
    *,
    arm: ArmSpec,
    sample: dict[str, Any],
    sample_index: int,
    split_key: str,
    prediction_name: str,
    motion_config: MotionConfig,
    descriptor_config: DescriptorConfig,
    cosine_eps: float,
    magnitude_eps: float,
    cache: dict[Path, VideoFeatures],
) -> dict[str, object]:
    sample_dir = arm.directory / f"sample_{sample_index:03d}"
    source_path = (sample_dir / "source.mp4").resolve()
    target_path = (sample_dir / "target.mp4").resolve()
    prediction_path = (sample_dir / prediction_name).resolve()

    def feature(path: Path) -> VideoFeatures:
        if path not in cache:
            cache[path] = _analyze(path, motion_config, descriptor_config)
        return cache[path]

    source = feature(source_path)
    target = feature(target_path)
    prediction = feature(prediction_path)

    target_cosine, target_cosine_valid = _cosine(
        prediction.descriptor, target.descriptor, cosine_eps
    )
    source_cosine, source_cosine_valid = _cosine(
        prediction.descriptor, source.descriptor, cosine_eps
    )
    target_delta = encode_action_delta(source.descriptor, target.descriptor, eps=cosine_eps)
    prediction_delta = encode_action_delta(
        source.descriptor, prediction.descriptor, eps=cosine_eps
    )
    delta_cosine, delta_cosine_valid = _cosine(
        prediction_delta, target_delta, cosine_eps
    )

    source_magnitude = float(source.analysis.metrics.residual_speed_mean)
    target_magnitude = float(target.analysis.metrics.residual_speed_mean)
    prediction_magnitude = float(prediction.analysis.metrics.residual_speed_mean)
    magnitude_log_ratio = math.log(
        (prediction_magnitude + magnitude_eps) / (target_magnitude + magnitude_eps)
    )

    source_label = source.analysis.label
    target_label = target.analysis.label
    prediction_label = prediction.analysis.label
    source_class = _motion_class(source_label)
    target_class = _motion_class(target_label)
    prediction_class = _motion_class(prediction_label)
    binary_classes = {"static", "dynamic"}
    # A camera-only or artifact prediction is a mismatch, not a reason to drop
    # an otherwise valid static/dynamic target from classification accuracy.
    static_dynamic_valid = target_class in binary_classes

    source_camera = float(source.analysis.metrics.camera_explained_ratio)
    target_camera = float(target.analysis.metrics.camera_explained_ratio)
    prediction_camera = float(prediction.analysis.metrics.camera_explained_ratio)
    source_actor = float(source.actor.actor_likeness)
    target_actor = float(target.actor.actor_likeness)
    prediction_actor = float(prediction.actor.actor_likeness)

    row: dict[str, object] = {
        "arm": arm.name,
        "sample_index": sample_index,
        "sample_id": _sample_id(sample, sample_index),
        "split": _sample_split(sample, split_key),
        "prompt": str(sample.get("prompt", "")),
        "source_path": str(source_path),
        "target_path": str(target_path),
        "prediction_path": str(prediction_path),
        "source_motion_label": source_label,
        "target_motion_label": target_label,
        "generated_motion_label": prediction_label,
        "source_motion_class": source_class,
        "target_motion_class": target_class,
        "motion_class": prediction_class,
        "motion_label_match": float(prediction_label == target_label),
        "static_dynamic_match": (
            float(prediction_class == target_class) if static_dynamic_valid else 0.0
        ),
        "static_dynamic_match_valid": static_dynamic_valid,
        "source_descriptor_norm": float(np.linalg.norm(source.descriptor)),
        "target_descriptor_norm": float(np.linalg.norm(target.descriptor)),
        "generated_descriptor_norm": float(np.linalg.norm(prediction.descriptor)),
        "target_delta_descriptor_norm": float(np.linalg.norm(target_delta)),
        "generated_delta_descriptor_norm": float(np.linalg.norm(prediction_delta)),
        "target_descriptor_cosine": target_cosine,
        "target_descriptor_cosine_valid": target_cosine_valid,
        "source_descriptor_cosine": source_cosine,
        "source_descriptor_cosine_valid": source_cosine_valid,
        "delta_descriptor_cosine": delta_cosine,
        "delta_descriptor_cosine_valid": delta_cosine_valid,
        "source_motion_magnitude": source_magnitude,
        "target_motion_magnitude": target_magnitude,
        "motion_magnitude": prediction_magnitude,
        "motion_magnitude_log_ratio": magnitude_log_ratio,
        "motion_magnitude_log_ratio_error": abs(magnitude_log_ratio),
        "source_actor_likeness": source_actor,
        "target_actor_likeness": target_actor,
        "actor_likeness": prediction_actor,
        "actor_likeness_abs_error": abs(prediction_actor - target_actor),
        "source_camera_ratio": source_camera,
        "target_camera_ratio": target_camera,
        "camera_ratio": prediction_camera,
        "camera_ratio_abs_error": abs(prediction_camera - target_camera),
        "source_active_fraction": float(source.actor.active_fraction),
        "target_active_fraction": float(target.actor.active_fraction),
        "generated_active_fraction": float(prediction.actor.active_fraction),
        "source_residual_speed_p90": float(
            source.analysis.metrics.residual_speed_p90
        ),
        "target_residual_speed_p90": float(
            target.analysis.metrics.residual_speed_p90
        ),
        "generated_residual_speed_p90": float(
            prediction.analysis.metrics.residual_speed_p90
        ),
        "source_scene_cut_ratio": float(source.analysis.metrics.scene_cut_ratio),
        "target_scene_cut_ratio": float(target.analysis.metrics.scene_cut_ratio),
        "generated_scene_cut_ratio": float(
            prediction.analysis.metrics.scene_cut_ratio
        ),
    }
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(
                f"non-finite per-sample value {key} for {arm.name} sample {sample_index}"
            )
    return row


def _bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> list[float] | None:
    if samples <= 0 or len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = np.asarray(
        [statistic(values[index]) for index in indices],
        dtype=np.float64,
    )
    tail = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1.0 - tail)),
    ]


def _derived_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _summarize_values(
    values: Iterable[object],
    *,
    total_n: int,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    result: dict[str, object] = {
        "n": int(len(finite)),
        "total_n": int(total_n),
        "mean": float(np.mean(finite)) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "std": float(np.std(finite)) if len(finite) else None,
        "min": float(np.min(finite)) if len(finite) else None,
        "max": float(np.max(finite)) if len(finite) else None,
    }
    if bootstrap_samples > 0 and len(finite):
        result["bootstrap_confidence"] = confidence
        result["mean_bootstrap_ci"] = _bootstrap_interval(
            finite,
            lambda array: float(np.mean(array)),
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        )
        result["median_bootstrap_ci"] = _bootstrap_interval(
            finite,
            lambda array: float(np.median(array)),
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seed ^ 0x9E3779B97F4A7C15,
        )
    return result


def _metric_values(
    rows: Sequence[dict[str, object]],
    metric: str,
    valid_key: str | None,
) -> list[object]:
    return [
        row[metric]
        for row in rows
        if metric in row and (valid_key is None or bool(row.get(valid_key)))
    ]


def _group_summaries(
    rows: Sequence[dict[str, object]],
    arms: Sequence[ArmSpec],
    *,
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    sample_splits = sorted({str(row["split"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm.name]
        for split in (ALL_SPLIT, *sample_splits):
            group = arm_rows if split == ALL_SPLIT else [
                row for row in arm_rows if row["split"] == split
            ]
            if not group:
                continue
            metric_summaries: dict[str, object] = {}
            for metric, (_direction, valid_key) in METRICS.items():
                metric_summaries[metric] = _summarize_values(
                    _metric_values(group, metric, valid_key),
                    total_n=len(group),
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    seed=_derived_seed(
                        bootstrap_seed, f"group:{arm.name}:{split}:{metric}"
                    ),
                )
            summaries.append(
                {
                    "arm": arm.name,
                    "split": split,
                    "n_samples": len(group),
                    "generated_motion_label_counts": dict(
                        sorted(Counter(str(row["generated_motion_label"]) for row in group).items())
                    ),
                    "target_motion_label_counts": dict(
                        sorted(Counter(str(row["target_motion_label"]) for row in group).items())
                    ),
                    "generated_motion_class_counts": dict(
                        sorted(Counter(str(row["motion_class"]) for row in group).items())
                    ),
                    "metrics": metric_summaries,
                }
            )
    return summaries


def _paired_comparisons(
    rows: Sequence[dict[str, object]],
    arms: Sequence[ArmSpec],
    reference_arm: str,
    *,
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    splits = sorted({str(row["split"]) for row in rows})
    by_arm = {
        arm.name: {int(row["sample_index"]): row for row in rows if row["arm"] == arm.name}
        for arm in arms
    }
    reference = by_arm[reference_arm]
    comparisons: list[dict[str, object]] = []
    for arm in arms:
        if arm.name == reference_arm:
            continue
        candidate = by_arm[arm.name]
        for split in (ALL_SPLIT, *splits):
            paired_indices = sorted(set(reference) & set(candidate))
            if split != ALL_SPLIT:
                paired_indices = [
                    index
                    for index in paired_indices
                    if reference[index]["split"] == split
                    and candidate[index]["split"] == split
                ]
            metric_results: dict[str, object] = {}
            for metric, (direction, valid_key) in METRICS.items():
                differences: list[float] = []
                improvements: list[float] = []
                for index in paired_indices:
                    reference_row = reference[index]
                    candidate_row = candidate[index]
                    if valid_key is not None and (
                        not bool(reference_row.get(valid_key))
                        or not bool(candidate_row.get(valid_key))
                    ):
                        continue
                    reference_value = float(reference_row[metric])
                    candidate_value = float(candidate_row[metric])
                    if not (
                        math.isfinite(reference_value)
                        and math.isfinite(candidate_value)
                    ):
                        continue
                    difference = candidate_value - reference_value
                    differences.append(difference)
                    if direction == "higher":
                        improvements.append(difference)
                    elif direction == "lower":
                        improvements.append(-difference)
                result: dict[str, object] = {
                    "direction": direction,
                    "candidate_minus_reference": _summarize_values(
                        differences,
                        total_n=len(paired_indices),
                        bootstrap_samples=bootstrap_samples,
                        confidence=confidence,
                        seed=_derived_seed(
                            bootstrap_seed,
                            f"paired-difference:{arm.name}:{split}:{metric}",
                        ),
                    ),
                }
                if direction is not None:
                    improvement_array = np.asarray(improvements, dtype=np.float64)
                    result["improvement"] = _summarize_values(
                        improvements,
                        total_n=len(paired_indices),
                        bootstrap_samples=bootstrap_samples,
                        confidence=confidence,
                        seed=_derived_seed(
                            bootstrap_seed,
                            f"paired-improvement:{arm.name}:{split}:{metric}",
                        ),
                    )
                    result["win_rate"] = (
                        float(np.mean(improvement_array > 0))
                        if len(improvement_array)
                        else None
                    )
                    result["tie_rate"] = (
                        float(np.mean(improvement_array == 0))
                        if len(improvement_array)
                        else None
                    )
                metric_results[metric] = result
            comparisons.append(
                {
                    "reference_arm": reference_arm,
                    "candidate_arm": arm.name,
                    "split": split,
                    "paired_sample_count": len(paired_indices),
                    "metrics": metric_results,
                }
            )
    return comparisons


def _write_atomic(
    path: Path,
    writer: Callable[[Any], None],
    *,
    newline: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=newline,
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, object]]) -> None:
    def writer(handle: Any) -> None:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    _write_atomic(path, writer)


def _write_csv_atomic(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty per-sample CSV")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("per-sample rows do not share one CSV schema")

    def writer(handle: Any) -> None:
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    _write_atomic(path, writer, newline="")


def _write_json_atomic(path: Path, payload: object) -> None:
    def writer(handle: Any) -> None:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _write_atomic(path, writer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize camera-compensated action metrics for matched "
            "lucy/infer_compare.py outputs."
        )
    )
    parser.add_argument(
        "--samples-json",
        type=Path,
        required=True,
        help="Canonical samples.json shared by every arm.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        type=_parse_arm,
        required=True,
        metavar="NAME=DIRECTORY",
        help="Inference output arm; repeat once per arm.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per_sample.jsonl, per_sample.csv and summary.json.",
    )
    parser.add_argument(
        "--prediction-name",
        default="step_000100.mp4",
        help="Generated video filename inside each sample directory.",
    )
    parser.add_argument(
        "--split-key",
        default="split",
        help="Canonical sample key (dotted keys supported) used for grouped summaries.",
    )
    parser.add_argument("--analysis-frames", type=int, default=32)
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--active-speed-threshold", type=float, default=0.005)
    parser.add_argument("--magnitude-eps", type=float, default=1e-6)
    parser.add_argument("--cosine-eps", type=float, default=1e-8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--reference-arm",
        default=None,
        help="Reference for paired comparisons (defaults to e1_plain_lora or first arm).",
    )
    parser.add_argument(
        "--verify-arm-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If an arm has samples.json, require it to match the canonical file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples_json = args.samples_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    arms = [
        ArmSpec(name=arm.name, directory=arm.directory.resolve())
        for arm in args.arm
    ]
    prediction_name = _validate_basename(args.prediction_name, "--prediction-name")
    if args.analysis_frames < 3:
        raise ValueError("--analysis-frames must be >= 3")
    if args.resize_width < 32:
        raise ValueError("--resize-width must be >= 32")
    if args.active_speed_threshold < 0:
        raise ValueError("--active-speed-threshold must be non-negative")
    if args.magnitude_eps <= 0 or args.cosine_eps <= 0:
        raise ValueError("--magnitude-eps and --cosine-eps must be positive")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be in (0, 1)")

    if not samples_json.is_file():
        raise FileNotFoundError(
            f"canonical samples JSON does not exist: {samples_json}"
        )
    samples_json_sha256 = _sha256(samples_json)
    samples = _load_samples(samples_json)
    if _sha256(samples_json) != samples_json_sha256:
        raise RuntimeError(
            f"canonical samples JSON changed while it was being read: {samples_json}"
        )
    for sample in samples:
        _sample_split(sample, args.split_key)
    _preflight(
        arms,
        samples,
        samples_json,
        prediction_name,
        verify_arm_samples=args.verify_arm_samples,
    )

    arm_names = [arm.name for arm in arms]
    reference_arm = args.reference_arm
    if reference_arm is None:
        reference_arm = (
            "e1_plain_lora" if "e1_plain_lora" in arm_names else arm_names[0]
        )
    if reference_arm not in arm_names:
        raise ValueError(
            f"--reference-arm {reference_arm!r} is not among {arm_names}"
        )

    motion_config = MotionConfig(
        analysis_frames=args.analysis_frames,
        resize_width=args.resize_width,
        active_speed_threshold=args.active_speed_threshold,
    )
    descriptor_config = DescriptorConfig(
        active_speed_threshold=args.active_speed_threshold,
    )
    cache: dict[Path, VideoFeatures] = {}
    rows: list[dict[str, object]] = []
    for arm in arms:
        for sample_index, sample in enumerate(samples):
            row = _evaluate_sample(
                arm=arm,
                sample=sample,
                sample_index=sample_index,
                split_key=args.split_key,
                prediction_name=prediction_name,
                motion_config=motion_config,
                descriptor_config=descriptor_config,
                cosine_eps=args.cosine_eps,
                magnitude_eps=args.magnitude_eps,
                cache=cache,
            )
            rows.append(row)
            print(
                f"[eval] arm={arm.name} sample={sample_index:03d} "
                f"target_cos={float(row['target_descriptor_cosine']):.6f} "
                f"delta_cos={float(row['delta_descriptor_cosine']):.6f} "
                f"motion={row['motion_class']}",
                flush=True,
            )

    groups = _group_summaries(
        rows,
        arms,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    paired = _paired_comparisons(
        rows,
        arms,
        reference_arm,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    if _sha256(samples_json) != samples_json_sha256:
        raise RuntimeError(
            f"canonical samples JSON changed during evaluation: {samples_json}"
        )
    summary = {
        "schema": SCHEMA,
        "complete": True,
        "samples_json": str(samples_json),
        "samples_json_sha256": samples_json_sha256,
        "sample_count": len(samples),
        "arms": [
            {"name": arm.name, "directory": str(arm.directory)}
            for arm in arms
        ],
        "prediction_name": prediction_name,
        "split_key": args.split_key,
        "reference_arm": reference_arm,
        "configuration": {
            "analysis_frames": args.analysis_frames,
            "resize_width": args.resize_width,
            "active_speed_threshold": args.active_speed_threshold,
            "magnitude_eps": args.magnitude_eps,
            "cosine_eps": args.cosine_eps,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "confidence": args.confidence,
            "verify_arm_samples": args.verify_arm_samples,
        },
        "metric_definitions": {
            "target_descriptor_cosine": (
                "cosine(generated action descriptor, target action descriptor)"
            ),
            "source_descriptor_cosine": (
                "cosine(generated action descriptor, source action descriptor); "
                "descriptive rather than higher-is-better"
            ),
            "delta_descriptor_cosine": (
                "cosine(unit(generated-source descriptor delta), "
                "unit(target-source descriptor delta))"
            ),
            "motion_magnitude_log_ratio_error": (
                "abs(log((generated residual_speed_mean + eps) / "
                "(target residual_speed_mean + eps)))"
            ),
            "actor_likeness": (
                "transparent coherent-actor proxy from generated "
                "camera-compensated residual flow"
            ),
            "camera_ratio": (
                "generated geometry.MotionMetrics.camera_explained_ratio"
            ),
            "static_dynamic_class": (
                "geometry label mapped only static->static and "
                "dynamic_object->dynamic; camera_only/artifact remain explicit"
            ),
        },
        "groups": groups,
        "paired_comparisons": paired,
    }

    # Summary is the commit marker and is replaced last.
    _write_jsonl_atomic(output_dir / "per_sample.jsonl", rows)
    _write_csv_atomic(output_dir / "per_sample.csv", rows)
    _write_json_atomic(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "complete": True,
                "sample_count": len(samples),
                "arm_count": len(arms),
                "per_sample_count": len(rows),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
