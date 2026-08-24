"""Build a high-quality, dynamic Goku source-anchor preselection.

The input is the 13k R7 Qwen ``fused.jsonl``.  Existing blind Qwen evidence is
used only as a conservative source-video gate; this stage does not treat the
paired target as ground truth.  Surviving source videos are decoded with the
shared camera-compensated geometry and actor-motion feature extractors.

Publication is create-only and directory-atomic.  A completed artifact contains
``evaluated.jsonl``, ``selected.jsonl``, canonical lossless first-frame PNGs,
``summary.json``, and ``done.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .geometry import MotionConfig, analyze_video
from .motion_features import extract_actor_motion_features
from .qwen_filter import _validate_observation as _validate_old_observation


SCHEMA_VERSION = "motive-goku-action-anchor-prefilter-v1"
EVALUATED_NAME = "evaluated.jsonl"
SELECTED_NAME = "selected.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
ANCHOR_DIR_NAME = "anchors"
OUTPUT_ENTRIES = frozenset(
    {
        EVALUATED_NAME,
        SELECTED_NAME,
        SUMMARY_NAME,
        DONE_NAME,
        ANCHOR_DIR_NAME,
    }
)
_SAFE_IID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class PrefilterConfig:
    """Thresholds for the cheap source-anchor eligibility stage."""

    sample_size: int = 768
    workers: int = 8
    max_per_family: int = 96
    analysis_frames: int = 32
    resize_width: int = 256
    active_speed_threshold: float = 0.005
    min_short_side: int = 480
    min_pixels: int = 832 * 480
    min_fps: float = 16.0
    max_fps: float = 60.0
    min_duration_seconds: float = 2.5
    max_duration_seconds: float = 10.0
    min_source_frames: int = 49
    min_residual_speed_p90: float = 0.005
    min_active_pixel_fraction: float = 0.010
    min_active_frame_fraction: float = 0.40
    min_actor_likeness: float = 0.25
    min_temporal_coverage: float = 0.40
    min_largest_component_share: float = 0.08
    max_spatial_energy_entropy: float = 0.94

    def validate(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.max_per_family <= 0:
            raise ValueError("max_per_family must be positive")
        if self.analysis_frames < 3:
            raise ValueError("analysis_frames must be at least 3")
        if self.resize_width < 32:
            raise ValueError("resize_width must be at least 32")
        if self.min_short_side <= 0 or self.min_pixels <= 0:
            raise ValueError("resolution thresholds must be positive")
        if not 0.0 < self.min_fps <= self.max_fps:
            raise ValueError("fps thresholds are invalid")
        if not 0.0 <= self.min_duration_seconds <= self.max_duration_seconds:
            raise ValueError("duration thresholds are invalid")
        if self.min_source_frames < 3:
            raise ValueError("min_source_frames must be at least 3")
        unit_interval = {
            "min_active_pixel_fraction": self.min_active_pixel_fraction,
            "min_active_frame_fraction": self.min_active_frame_fraction,
            "min_actor_likeness": self.min_actor_likeness,
            "min_temporal_coverage": self.min_temporal_coverage,
            "min_largest_component_share": self.min_largest_component_share,
            "max_spatial_energy_entropy": self.max_spatial_energy_entropy,
        }
        for name, value in unit_interval.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.active_speed_threshold <= 0.0:
            raise ValueError("active_speed_threshold must be positive")
        if self.min_residual_speed_p90 <= 0.0:
            raise ValueError("min_residual_speed_p90 must be positive")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return (
        "".join(_canonical_json(dict(row)) + "\n" for row in rows)
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _object_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _strict_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _is_plain_directory(path: Path) -> bool:
    try:
        return path.is_dir() and not path.is_symlink()
    except OSError:
        return False


def _read_fused(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"input fused manifest is not a regular file: {resolved}")
    raw = resolved.read_bytes()
    if not raw:
        raise ValueError("input fused manifest is empty")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(
                f"input fused manifest contains a blank line: {line_number}"
            )
        try:
            value = json.loads(
                raw_line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_strict_json_object,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"invalid JSON in fused manifest line {line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"fused manifest line {line_number} is not an object"
            )
        iid = str(value.get("iid") or "").strip()
        if not iid:
            raise ValueError(f"fused manifest line {line_number} has no iid")
        if not _SAFE_IID.fullmatch(iid) or iid in {".", ".."}:
            raise ValueError(f"unsafe iid in fused manifest: {iid!r}")
        if iid in seen:
            raise ValueError(f"duplicate iid in fused manifest: {iid}")
        seen.add(iid)
        row = dict(value)
        row["_source_line_number"] = line_number
        rows.append(row)
    return rows, raw


def _primary_family(row: Mapping[str, Any]) -> str:
    selection = row.get("r7_expansion_selection")
    if isinstance(selection, Mapping):
        value = selection.get("primary_family")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    rule = row.get("auto_rule")
    if isinstance(rule, Mapping):
        families = rule.get("action_families")
        if isinstance(families, Sequence) and not isinstance(
            families, (str, bytes)
        ):
            for value in families:
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
    return "unknown"


def _text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    return value.strip() if isinstance(value, str) else ""


def _old_qwen_sha256(value: Any, *, field: str, iid: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"iid={iid} old Qwen {field} must be a lowercase SHA-256 digest"
        )
    return value


def _old_qwen_text(value: Any, *, field: str, iid: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise ValueError(
            f"iid={iid} old Qwen {field} must be a canonical non-empty string"
        )
    return value


def _validate_old_qwen_source_evidence(
    row: Mapping[str, Any],
    *,
    iid: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the indivisibly signed old observation, but no result.

    The old Qwen schema placed source and target observations in one object and
    signed that complete object with one ``observation_digest``.  There is no
    independently verifiable source projection in the historical artifacts.
    Consequently the complete closed observation remains integrity-checked,
    while only its source fields are used by :func:`qwen_source_gate`.

    The paired alignment/result object and every result-side digest or repair
    trail are deliberately ignored.  A corrupt old target verdict must not
    exclude an otherwise usable source video.
    """

    evidence = row.get("qwen_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError(f"iid={iid} has no qwen_evidence object")
    visual = evidence.get("visual")
    if not isinstance(visual, dict):
        raise ValueError(f"iid={iid} has no qwen_evidence.visual object")
    if visual.get("iid") != iid:
        raise ValueError(f"iid={iid} disagrees with old Qwen evidence IID")

    input_digest = _old_qwen_sha256(
        row.get("input_digest"),
        field="input_digest",
        iid=iid,
    )
    if visual.get("input_digest") != input_digest:
        raise ValueError(f"iid={iid} old Qwen input_digest mismatch")
    if visual.get("status") != "ok" or visual.get("mode") != "visual":
        raise ValueError(
            f"iid={iid} old Qwen evidence is not a successful visual row"
        )

    observation = visual.get("observation")
    if not isinstance(observation, dict):
        raise ValueError(f"iid={iid} lacks an old Qwen observation object")
    try:
        _validate_old_observation(observation)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"iid={iid} old Qwen observation schema validation failed: {error}"
        ) from error
    observation_digest = _object_digest(observation)
    if visual.get("observation_digest") != observation_digest:
        raise ValueError(f"iid={iid} old Qwen observation_digest mismatch")

    repairs = visual.get("observation_repairs")
    if visual.get("observation_validated_from") != "original":
        raise ValueError(
            f"iid={iid} old Qwen observation provenance is not original"
        )
    if not isinstance(repairs, list) or repairs:
        raise ValueError(
            f"iid={iid} old Qwen original observation has a repair trail"
        )

    for field in (
        "visual_input_digest",
        "run_config_digest",
        "config_digest",
        "implementation_digest",
        "execution_manifest_sha256",
    ):
        _old_qwen_sha256(visual.get(field), field=field, iid=iid)
    for field in (
        "model_revision",
        "transformers_version",
        "execution_manifest",
    ):
        _old_qwen_text(visual.get(field), field=field, iid=iid)
    shard_index = visual.get("execution_shard_index")
    shard_count = visual.get("execution_shard_count")
    if (
        type(shard_count) is not int
        or shard_count != 8
        or type(shard_index) is not int
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError(
            f"iid={iid} old Qwen execution shard provenance is invalid"
        )
    return dict(visual), dict(observation)


def qwen_source_gate(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Apply the strict existing-Qwen source-only gates.

    The historical observation has an indivisible source+target digest, so its
    complete closed schema is still checked for integrity.  Only the source
    actor-motion field has gate semantics.  The paired result/target verdict is
    neither parsed nor trusted and cannot reject a source anchor.
    """

    reasons: list[str] = []
    iid = str(row.get("iid") or "")
    try:
        visual, observation = _validate_old_qwen_source_evidence(
            row,
            iid=iid,
        )
        evidence_error = None
    except Exception as error:
        evidence = row.get("qwen_evidence")
        visual_value = (
            evidence.get("visual") if isinstance(evidence, Mapping) else None
        )
        visual = dict(visual_value) if isinstance(visual_value, Mapping) else {}
        observation_value = visual.get("observation")
        observation = (
            dict(observation_value)
            if isinstance(observation_value, Mapping)
            else {}
        )
        evidence_error = f"{type(error).__name__}: {error}"
        reasons.append("invalid_r7_qwen_evidence")

    if visual.get("observation_validated_from") != "original":
        reasons.append("qwen_observation_not_original")
    if observation.get("source_actor_motion") != "clear":
        reasons.append("qwen_source_motion_not_clear")

    gate = {
        "status": visual.get("status"),
        "evidence_integrity": evidence_error is None,
        "evidence_error": evidence_error,
        "observation_validated_from": visual.get(
            "observation_validated_from"
        ),
        "source_actor_motion": observation.get("source_actor_motion"),
        "camera_dominance": observation.get("camera_dominance"),
        "background_dominance": observation.get("background_dominance"),
        "artifact_level": observation.get("artifact_level"),
        "preservation_quality": observation.get("preservation_quality"),
        "observed_source_action": observation.get("source_action"),
        "legacy_observation_digest": visual.get("observation_digest"),
        "legacy_result_ignored": True,
        "passed": not reasons,
    }
    return gate, reasons


def _resolve_source_video(value: str, video_root: Path) -> Path:
    if not value:
        raise ValueError("missing src_video")
    raw = Path(value).expanduser()
    if raw.is_absolute():
        candidate = raw
    else:
        candidate = video_root / raw
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(video_root)
    except ValueError as error:
        raise ValueError(
            f"source video escapes video_root: {resolved}"
        ) from error
    if not resolved.is_file():
        raise ValueError(f"source video is not a regular file: {resolved}")
    return resolved


def _base_evaluated_row(
    row: Mapping[str, Any],
    *,
    input_sha256: str,
) -> dict[str, Any]:
    iid = str(row["iid"])
    family = _primary_family(row)
    group_id = _text(row, "group_id") or iid
    gate, reasons = qwen_source_gate(row)
    required_text = {
        "prompt": _text(row, "prompt"),
        "source_caption": _text(row, "source_caption"),
        "edited_caption": _text(row, "edited_caption"),
        "src_video": _text(row, "src_video"),
    }
    for field, value in required_text.items():
        if not value:
            reasons.append(f"missing_{field}")
    if family == "unknown":
        reasons.append("unknown_action_family")
    return {
        "schema_version": SCHEMA_VERSION,
        "iid": iid,
        "group_id": group_id,
        "family": family,
        "source_line_number": int(row["_source_line_number"]),
        "input_manifest_sha256": input_sha256,
        "input_row_sha256": _object_digest(
            {
                key: value
                for key, value in row.items()
                if key != "_source_line_number"
            }
        ),
        **required_text,
        "tgt_video": _text(row, "tgt_video"),
        "qwen_source_gate": gate,
        "resolved_src_video": None,
        "media": None,
        "motion": None,
        "actor_motion": None,
        "score_components": None,
        "prefilter_score": None,
        "eligible": False,
        "rejection_reasons": reasons,
        "selected": False,
        "selection_rank": None,
        "within_family_rank": None,
        "anchor_image": None,
        "resolved_anchor_image": None,
        "anchor_sha256": None,
        "source_video_sha256": None,
    }


def _finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}")
    return result


def _analysis_payload(
    record: Mapping[str, Any],
    *,
    config: PrefilterConfig,
) -> dict[str, Any]:
    return {
        "iid": record["iid"],
        "video_path": record["resolved_src_video"],
        "analysis_frames": config.analysis_frames,
        "resize_width": config.resize_width,
        "active_speed_threshold": config.active_speed_threshold,
    }


def _probe_container_metadata(path: Path) -> dict[str, float | int]:
    """Read trustworthy container metadata without geometry's FPS fallback."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video metadata: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    values = {
        "fps": fps,
        "frame_count": frame_count_value,
        "width": width_value,
        "height": height_value,
    }
    invalid = [
        name
        for name, value in values.items()
        if not math.isfinite(value) or value <= 0
    ]
    if invalid:
        raise ValueError(
            "invalid or unavailable container metadata: "
            + ",".join(sorted(invalid))
        )
    return {
        "fps": fps,
        "frame_count": int(round(frame_count_value)),
        "width": int(round(width_value)),
        "height": int(round(height_value)),
    }


def _analyze_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Worker entry point; failures become explicit evaluated-row evidence."""

    cv2.setNumThreads(1)
    iid = str(payload["iid"])
    path = Path(str(payload["video_path"]))
    try:
        motion_config = MotionConfig(
            analysis_frames=int(payload["analysis_frames"]),
            resize_width=int(payload["resize_width"]),
            active_speed_threshold=float(payload["active_speed_threshold"]),
        )
        stat_before = path.stat()
        probed = _probe_container_metadata(path)
        analysis = analyze_video(path, motion_config)
        actor = extract_actor_motion_features(
            analysis,
            active_speed_threshold=float(payload["active_speed_threshold"]),
        )
        stat_after = path.stat()
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        ):
            raise RuntimeError("source video changed during analysis")
        metrics = analysis.metrics.to_dict()
        for field, metric_field in (
            ("frame_count", "source_frame_count"),
            ("width", "source_width"),
            ("height", "source_height"),
        ):
            if int(probed[field]) != int(metrics[metric_field]):
                raise RuntimeError(
                    f"container {field} changed or disagrees with analysis"
                )
        if not math.isclose(
            float(probed["fps"]),
            float(metrics["source_fps"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "container fps changed or geometry used an FPS fallback"
            )
        actor_values = actor.to_dict()
        media = {
            "width": int(metrics["source_width"]),
            "height": int(metrics["source_height"]),
            "pixels": int(metrics["source_width"])
            * int(metrics["source_height"]),
            "short_side": min(
                int(metrics["source_width"]),
                int(metrics["source_height"]),
            ),
            "fps": _finite_float(metrics["source_fps"], field="source_fps"),
            "frame_count": int(metrics["source_frame_count"]),
            "duration_seconds": _finite_float(
                metrics["duration_seconds"],
                field="duration_seconds",
            ),
            "file_size_bytes": int(stat_after.st_size),
            "mtime_ns_at_analysis": int(stat_after.st_mtime_ns),
        }
        motion = {
            "label": str(analysis.label),
            "raw_speed_mean": _finite_float(
                metrics["raw_speed_mean"], field="raw_speed_mean"
            ),
            "raw_speed_p90": _finite_float(
                metrics["raw_speed_p90"], field="raw_speed_p90"
            ),
            "residual_speed_mean": _finite_float(
                metrics["residual_speed_mean"],
                field="residual_speed_mean",
            ),
            "residual_speed_p90": _finite_float(
                metrics["residual_speed_p90"],
                field="residual_speed_p90",
            ),
            "residual_speed_p99": _finite_float(
                metrics["residual_speed_p99"],
                field="residual_speed_p99",
            ),
            "active_pixel_fraction": _finite_float(
                metrics["active_pixel_fraction"],
                field="active_pixel_fraction",
            ),
            "active_frame_fraction": _finite_float(
                metrics["active_frame_fraction"],
                field="active_frame_fraction",
            ),
            "camera_explained_ratio": _finite_float(
                metrics["camera_explained_ratio"],
                field="camera_explained_ratio",
            ),
            "affine_inlier_ratio": _finite_float(
                metrics["affine_inlier_ratio"],
                field="affine_inlier_ratio",
            ),
            "scene_cut_ratio": _finite_float(
                metrics["scene_cut_ratio"], field="scene_cut_ratio"
            ),
            "temporal_energy_cv": _finite_float(
                metrics["temporal_energy_cv"],
                field="temporal_energy_cv",
            ),
            "sampled_frames": int(metrics["sampled_frames"]),
        }
        clean_actor = {
            key: (
                value
                if isinstance(value, str)
                else _finite_float(value, field=f"actor_motion.{key}")
            )
            for key, value in actor_values.items()
        }
        return {
            "iid": iid,
            "ok": True,
            "media": media,
            "motion": motion,
            "actor_motion": clean_actor,
        }
    except Exception as error:  # one corrupt clip must remain auditable
        return {
            "iid": iid,
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _quality_score(
    media: Mapping[str, Any],
    motion: Mapping[str, Any],
    actor: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    resolution = (
        0.60 * _clip01(float(media["short_side"]) / 720.0)
        + 0.40 * _clip01(float(media["pixels"]) / float(1280 * 720))
    )
    fps = float(media["fps"])
    fps_quality = 1.0 if 16.0 <= fps <= 30.0 else 0.75
    duration = float(media["duration_seconds"])
    duration_quality = _clip01(1.0 - abs(duration - 4.0) / 8.0)
    media_quality = (
        0.65 * resolution
        + 0.20 * fps_quality
        + 0.15 * duration_quality
    )
    residual = _clip01(float(motion["residual_speed_p90"]) / 0.025)
    active_pixels = _clip01(
        float(motion["active_pixel_fraction"]) / 0.08
    )
    active_frames = _clip01(float(motion["active_frame_fraction"]))
    dynamics = (
        0.45 * residual
        + 0.25 * active_pixels
        + 0.30 * active_frames
    )
    component = _clip01(float(actor["largest_component_share"]) / 0.60)
    coherence = _clip01(
        (float(actor["adjacent_energy_coherence"]) + 1.0) / 2.0
    )
    actor_quality = (
        0.40 * _clip01(float(actor["actor_likeness"]))
        + 0.30 * _clip01(float(actor["temporal_coverage"]))
        + 0.18 * component
        + 0.12 * coherence
    )
    total = 0.25 * media_quality + 0.35 * dynamics + 0.40 * actor_quality
    components = {
        "media_quality": media_quality,
        "resolution_quality": resolution,
        "fps_quality": fps_quality,
        "duration_quality": duration_quality,
        "dynamics": dynamics,
        "residual_motion": residual,
        "active_pixels": active_pixels,
        "active_frames": active_frames,
        "actor_quality": actor_quality,
        "actor_component": component,
        "temporal_coherence": coherence,
    }
    return float(total), components


def _media_motion_reasons(
    result: Mapping[str, Any],
    config: PrefilterConfig,
) -> list[str]:
    media = result["media"]
    motion = result["motion"]
    actor = result["actor_motion"]
    reasons: list[str] = []
    if motion["label"] != "dynamic_object":
        reasons.append(f"motion_label_{motion['label']}")
    if int(media["short_side"]) < config.min_short_side:
        reasons.append("resolution_short_side_too_small")
    if int(media["pixels"]) < config.min_pixels:
        reasons.append("resolution_pixel_count_too_small")
    if not config.min_fps <= float(media["fps"]) <= config.max_fps:
        reasons.append("fps_out_of_range")
    if not (
        config.min_duration_seconds
        <= float(media["duration_seconds"])
        <= config.max_duration_seconds
    ):
        reasons.append("duration_out_of_range")
    if int(media["frame_count"]) < config.min_source_frames:
        reasons.append("too_few_source_frames")
    if (
        float(motion["residual_speed_p90"])
        < config.min_residual_speed_p90
    ):
        reasons.append("residual_motion_too_weak")
    if (
        float(motion["active_pixel_fraction"])
        < config.min_active_pixel_fraction
    ):
        reasons.append("active_pixel_fraction_too_small")
    if (
        float(motion["active_frame_fraction"])
        < config.min_active_frame_fraction
    ):
        reasons.append("active_frame_fraction_too_small")
    if float(actor["actor_likeness"]) < config.min_actor_likeness:
        reasons.append("actor_likeness_too_low")
    if float(actor["temporal_coverage"]) < config.min_temporal_coverage:
        reasons.append("actor_temporal_coverage_too_low")
    if (
        float(actor["largest_component_share"])
        < config.min_largest_component_share
    ):
        reasons.append("actor_motion_too_diffuse")
    if (
        float(actor["spatial_energy_entropy"])
        > config.max_spatial_energy_entropy
    ):
        reasons.append("spatial_motion_entropy_too_high")
    return reasons


def select_diverse(
    eligible: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    max_per_family: int,
) -> list[dict[str, Any]]:
    """Select score-ranked rows in deterministic family round-robin order."""

    if sample_size <= 0 or max_per_family <= 0:
        raise ValueError("selection sizes must be positive")
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        if not row.get("eligible"):
            raise ValueError("select_diverse received an ineligible row")
        by_family[str(row["family"])].append(row)
    for family in by_family:
        by_family[family].sort(
            key=lambda row: (-float(row["prefilter_score"]), str(row["iid"]))
        )
    family_order = sorted(
        by_family,
        key=lambda family: (
            -float(by_family[family][0]["prefilter_score"]),
            family,
        ),
    )
    offsets = {family: 0 for family in family_order}
    family_counts: Counter[str] = Counter()
    groups: set[str] = set()
    chosen: list[dict[str, Any]] = []
    while len(chosen) < sample_size:
        progress = False
        for family in family_order:
            if len(chosen) >= sample_size:
                break
            if family_counts[family] >= max_per_family:
                continue
            candidates = by_family[family]
            while offsets[family] < len(candidates):
                candidate = candidates[offsets[family]]
                offsets[family] += 1
                group_id = str(candidate["group_id"])
                if group_id in groups:
                    continue
                selected = dict(candidate)
                family_counts[family] += 1
                selected["within_family_rank"] = family_counts[family]
                chosen.append(selected)
                groups.add(group_id)
                progress = True
                break
        if not progress:
            break
    for rank, row in enumerate(chosen, start=1):
        row["selection_rank"] = rank
    return chosen


def _extract_anchor_png_bytes(path: Path) -> tuple[bytes, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open selected video: {path}")
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(f"could not decode canonical first frame: {path}")
    ok, encoded = cv2.imencode(
        ".png",
        frame,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not ok:
        raise RuntimeError(f"could not encode canonical first frame: {path}")
    height, width = frame.shape[:2]
    return encoded.tobytes(), int(width), int(height)


def _materialize_selected_source(
    row: dict[str, Any],
    *,
    stage: Path,
    final_output: Path,
) -> dict[str, Any]:
    video = Path(str(row["resolved_src_video"]))
    before = video.stat()
    source_sha256 = _file_sha256(video)
    anchor_bytes, anchor_width, anchor_height = _extract_anchor_png_bytes(video)
    after = video.stat()
    expected_size = int(row["media"]["file_size_bytes"])
    expected_mtime = int(row["media"]["mtime_ns_at_analysis"])
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or after.st_size != expected_size
        or after.st_mtime_ns != expected_mtime
    ):
        raise RuntimeError(
            f"selected source video changed after analysis: {row['iid']}"
        )
    relative = Path(ANCHOR_DIR_NAME) / f"{row['iid']}.png"
    anchor_path = stage / relative
    anchor_path.write_bytes(anchor_bytes)
    updated = dict(row)
    updated["selected"] = True
    updated["anchor_image"] = relative.as_posix()
    updated["resolved_anchor_image"] = str(
        (final_output / relative).resolve(strict=False)
    )
    updated["anchor_sha256"] = _sha256_bytes(anchor_bytes)
    updated["source_video_sha256"] = source_sha256
    updated["media"] = {
        **dict(row["media"]),
        "anchor_width": anchor_width,
        "anchor_height": anchor_height,
        "anchor_frame_index": 0,
        "anchor_encoding": "lossless_png",
    }
    return updated


def _publish_directory(
    output_dir: Path,
    *,
    writer: Any,
) -> None:
    target = output_dir.expanduser().resolve(strict=False)
    if os.path.lexists(target):
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not _is_plain_directory(target.parent):
        raise ValueError(f"output parent is not a plain directory: {target.parent}")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=str(target.parent),
        )
    )
    try:
        writer(stage, target)
        if set(path.name for path in stage.iterdir()) != OUTPUT_ENTRIES:
            raise RuntimeError("staging artifact closure differs")
        if os.path.lexists(target):
            raise FileExistsError(
                f"output appeared during publication: {target}"
            )
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def run_prefilter(
    *,
    input_fused: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    config: PrefilterConfig | None = None,
) -> dict[str, Any]:
    """Evaluate source anchors and atomically publish a diverse preselection."""

    config = config or PrefilterConfig()
    config.validate()
    input_path = Path(input_fused).expanduser().resolve(strict=True)
    root = Path(video_root).expanduser().resolve(strict=True)
    if not _is_plain_directory(root):
        raise ValueError(f"video_root is not a plain directory: {root}")
    target = Path(output_dir).expanduser().resolve(strict=False)
    if os.path.lexists(target):
        raise FileExistsError(f"output already exists: {target}")

    raw_rows, input_raw = _read_fused(input_path)
    input_sha256 = _sha256_bytes(input_raw)
    evaluated = [
        _base_evaluated_row(row, input_sha256=input_sha256)
        for row in raw_rows
    ]
    by_iid = {str(row["iid"]): row for row in evaluated}

    payloads: list[dict[str, Any]] = []
    for record in evaluated:
        if record["rejection_reasons"]:
            continue
        try:
            resolved = _resolve_source_video(str(record["src_video"]), root)
        except Exception as error:
            record["rejection_reasons"].append(
                f"source_video_error:{type(error).__name__}"
            )
            record["source_video_error"] = str(error)
            continue
        record["resolved_src_video"] = str(resolved)
        payloads.append(_analysis_payload(record, config=config))

    if config.workers == 1:
        results = [_analyze_payload(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            results = list(executor.map(_analyze_payload, payloads))

    for result in results:
        record = by_iid[str(result["iid"])]
        if not result["ok"]:
            record["rejection_reasons"].append(
                f"analysis_error:{result['error_type']}"
            )
            record["analysis_error"] = result["error"]
            continue
        record["media"] = result["media"]
        record["motion"] = result["motion"]
        record["actor_motion"] = result["actor_motion"]
        record["rejection_reasons"].extend(
            _media_motion_reasons(result, config)
        )
        score, components = _quality_score(
            result["media"],
            result["motion"],
            result["actor_motion"],
        )
        record["score_components"] = components
        record["prefilter_score"] = score
        record["eligible"] = not record["rejection_reasons"]

    evaluated.sort(key=lambda row: str(row["iid"]))
    prelim = select_diverse(
        [row for row in evaluated if row["eligible"]],
        sample_size=config.sample_size,
        max_per_family=config.max_per_family,
    )
    selected_rank = {
        str(row["iid"]): (
            int(row["selection_rank"]),
            int(row["within_family_rank"]),
        )
        for row in prelim
    }
    for record in evaluated:
        ranks = selected_rank.get(str(record["iid"]))
        if ranks is not None:
            record["selection_rank"], record["within_family_rank"] = ranks

    def _writer(stage: Path, final_output: Path) -> None:
        (stage / ANCHOR_DIR_NAME).mkdir()
        materialized: list[dict[str, Any]] = []
        evaluated_by_iid = {
            str(record["iid"]): record for record in evaluated
        }
        for selected in prelim:
            complete = _materialize_selected_source(
                selected,
                stage=stage,
                final_output=final_output,
            )
            materialized.append(complete)
            evaluated_by_iid[str(complete["iid"])] = complete
        final_evaluated = sorted(
            evaluated_by_iid.values(),
            key=lambda row: str(row["iid"]),
        )
        materialized.sort(key=lambda row: int(row["selection_rank"]))
        evaluated_raw = _jsonl_bytes(final_evaluated)
        selected_raw = _jsonl_bytes(materialized)
        (stage / EVALUATED_NAME).write_bytes(evaluated_raw)
        (stage / SELECTED_NAME).write_bytes(selected_raw)

        rejection_counts: Counter[str] = Counter()
        qwen_passed = 0
        geometry_attempted = 0
        for record in final_evaluated:
            qwen_passed += bool(record["qwen_source_gate"]["passed"])
            geometry_attempted += record["media"] is not None
            rejection_counts.update(record["rejection_reasons"])
        eligible_rows = [
            record for record in final_evaluated if record["eligible"]
        ]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "input": {
                "path": str(input_path),
                "rows": len(raw_rows),
                "sha256": input_sha256,
            },
            "video_root": str(root),
            "output": str(final_output),
            "config": asdict(config),
            "counts": {
                "evaluated": len(final_evaluated),
                "qwen_source_gate_passed": qwen_passed,
                "geometry_attempted": geometry_attempted,
                "eligible": len(eligible_rows),
                "selected": len(materialized),
                "requested": config.sample_size,
                "selection_shortfall": max(
                    config.sample_size - len(materialized), 0
                ),
            },
            "eligible_family_counts": dict(
                sorted(
                    Counter(
                        str(record["family"]) for record in eligible_rows
                    ).items()
                )
            ),
            "selected_family_counts": dict(
                sorted(
                    Counter(
                        str(record["family"]) for record in materialized
                    ).items()
                )
            ),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "selection_policy": {
                "version": "family-round-robin-score-v1",
                "within_family_order": "prefilter_score_desc_then_iid",
                "family_order": "best_score_desc_then_family",
                "unique_group_id": True,
                "max_per_family": config.max_per_family,
                "pseudo_labels_only": True,
                "human_review_pending": True,
            },
            "media_motion_policy": {
                "bounds_are_inclusive": True,
                "minimums": {
                    "short_side": config.min_short_side,
                    "pixels": config.min_pixels,
                    "fps": config.min_fps,
                    "duration_seconds": config.min_duration_seconds,
                    "source_frames": config.min_source_frames,
                },
                "intentional_upper_bounds": {
                    "fps": config.max_fps,
                    "duration_seconds": config.max_duration_seconds,
                    "purpose": (
                        "exclude atypical high-rate or long clips from the "
                        "small controlled generation pilot"
                    ),
                },
                "invalid_container_metadata": "fail_closed",
                "source_path_scope": "must_resolve_within_video_root",
            },
            "production_eligible": False,
        }
        summary_raw = _json_bytes(summary)
        (stage / SUMMARY_NAME).write_bytes(summary_raw)
        anchors = {
            row["anchor_image"]: row["anchor_sha256"]
            for row in materialized
        }
        artifacts = {
            EVALUATED_NAME: _sha256_bytes(evaluated_raw),
            SELECTED_NAME: _sha256_bytes(selected_raw),
            SUMMARY_NAME: _sha256_bytes(summary_raw),
            ANCHOR_DIR_NAME: _object_digest(anchors),
        }
        done = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "input_sha256": input_sha256,
            "evaluated_rows": len(final_evaluated),
            "selected_rows": len(materialized),
            "artifacts": artifacts,
            "anchor_sha256": dict(sorted(anchors.items())),
            "artifact_digest": _object_digest(artifacts),
        }
        (stage / DONE_NAME).write_bytes(_json_bytes(done))

    _publish_directory(target, writer=_writer)
    return json.loads((target / SUMMARY_NAME).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    defaults = PrefilterConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Select high-resolution, clearly dynamic Goku source anchors from "
            "the R7 13k Qwen fused manifest."
        )
    )
    parser.add_argument(
        "--input-fused",
        "--input",
        dest="input_fused",
        required=True,
        type=Path,
    )
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sample-size", type=int, default=defaults.sample_size
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=defaults.workers,
    )
    parser.add_argument(
        "--max-per-family", type=int, default=defaults.max_per_family
    )
    parser.add_argument(
        "--analysis-frames", type=int, default=defaults.analysis_frames
    )
    parser.add_argument(
        "--resize-width", type=int, default=defaults.resize_width
    )
    parser.add_argument(
        "--active-speed-threshold",
        type=float,
        default=defaults.active_speed_threshold,
    )
    parser.add_argument(
        "--min-short-side", type=int, default=defaults.min_short_side
    )
    parser.add_argument(
        "--min-pixels", type=int, default=defaults.min_pixels
    )
    parser.add_argument(
        "--min-fps", type=float, default=defaults.min_fps
    )
    parser.add_argument(
        "--max-fps", type=float, default=defaults.max_fps
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=defaults.min_duration_seconds,
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=defaults.max_duration_seconds,
    )
    parser.add_argument(
        "--min-source-frames",
        type=int,
        default=defaults.min_source_frames,
    )
    parser.add_argument(
        "--min-residual-speed-p90",
        type=float,
        default=defaults.min_residual_speed_p90,
    )
    parser.add_argument(
        "--min-active-pixel-fraction",
        type=float,
        default=defaults.min_active_pixel_fraction,
    )
    parser.add_argument(
        "--min-active-frame-fraction",
        type=float,
        default=defaults.min_active_frame_fraction,
    )
    parser.add_argument(
        "--min-actor-likeness",
        type=float,
        default=defaults.min_actor_likeness,
    )
    parser.add_argument(
        "--min-temporal-coverage",
        type=float,
        default=defaults.min_temporal_coverage,
    )
    parser.add_argument(
        "--min-largest-component-share",
        type=float,
        default=defaults.min_largest_component_share,
    )
    parser.add_argument(
        "--max-spatial-energy-entropy",
        type=float,
        default=defaults.max_spatial_energy_entropy,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PrefilterConfig(
        sample_size=args.sample_size,
        workers=args.workers,
        max_per_family=args.max_per_family,
        analysis_frames=args.analysis_frames,
        resize_width=args.resize_width,
        active_speed_threshold=args.active_speed_threshold,
        min_short_side=args.min_short_side,
        min_pixels=args.min_pixels,
        min_fps=args.min_fps,
        max_fps=args.max_fps,
        min_duration_seconds=args.min_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
        min_source_frames=args.min_source_frames,
        min_residual_speed_p90=args.min_residual_speed_p90,
        min_active_pixel_fraction=args.min_active_pixel_fraction,
        min_active_frame_fraction=args.min_active_frame_fraction,
        min_actor_likeness=args.min_actor_likeness,
        min_temporal_coverage=args.min_temporal_coverage,
        min_largest_component_share=args.min_largest_component_share,
        max_spatial_energy_entropy=args.max_spatial_energy_entropy,
    )
    summary = run_prefilter(
        input_fused=args.input_fused,
        video_root=args.video_root,
        output_dir=args.output_dir,
        config=config,
    )
    print(
        "[motive-goku-action-anchor-prefilter] "
        f"evaluated={summary['counts']['evaluated']} "
        f"eligible={summary['counts']['eligible']} "
        f"selected={summary['counts']['selected']} "
        f"output={summary['output']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
