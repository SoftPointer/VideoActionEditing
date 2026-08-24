"""Build the provenance-bound R10B Bernini controlled-pilot manifest.

This module is intentionally split into two fail-closed stages:

``queue``
    Screen the R7 candidate/track commits with frozen feature gates and emit a
    component-disjoint queue for a source+target+instruction Qwen audit.

``finalize``
    Validate one Qwen record for every queued row, apply the frozen semantic
    precedence and quotas, and emit at most 20 visual rows.  Missing controls remain
    explicit shortfalls; rows are never duplicated or relabelled to fill a
    quota.

Only paths and cryptographic bindings are written.  Video bytes are neither
copied nor decoded here.  Qwen decisions are audit pseudo-labels, not human
labels or authorization for representation promotion, rendering, generation,
or training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .r10b_tangent_core import (
    R10BTangentError,
    SMOKE_ROW_SCHEMA,
    canonical_json,
    file_digest,
    object_digest,
    strict_atomic_family,
    track_delta_components,
    validate_smoke_rows,
    validate_track_cache_arrays,
)


QUEUE_ROW_SCHEMA = "motive-r10b-bernini-qwen-audit-queue-row-v1"
QUEUE_SUMMARY_SCHEMA = "motive-r10b-bernini-qwen-audit-queue-v1"
QUEUE_DONE_SCHEMA = "motive-r10b-bernini-qwen-audit-queue-done-v1"
AUDIT_ROW_SCHEMA = "motive-r10b-bernini-qwen-audit-record-v1"
FINAL_SUMMARY_SCHEMA = "motive-r10b-bernini-controlled-pilot-v1"
FINAL_DONE_SCHEMA = "motive-r10b-bernini-controlled-pilot-done-v1"
SHORTFALL_SCHEMA = "motive-r10b-bernini-controlled-pilot-shortfalls-v1"

QUEUE_NAME = "qwen_audit_queue.jsonl"
QUEUE_SUMMARY_NAME = "summary.json"
QUEUE_DONE_NAME = "done.json"
FINAL_MANIFEST_NAME = "manifest.jsonl"
FINAL_SHORTFALL_NAME = "shortfalls.json"
FINAL_SUMMARY_NAME = "summary.json"
FINAL_DONE_NAME = "done.json"

SELECTION_SEED = 260108847
MAX_FINAL_ROWS = 20
NOOP_PROMPT = "Keep the video unchanged."
CANONICAL_PROMPTS = {
    "wave": "Make the subject wave one forelimb toward the viewer.",
    "quadruped_lie_down": "Make the quadruped lie down.",
}
CROSS_FAMILY = {
    "wave": "quadruped_lie_down",
    "quadruped_lie_down": "wave",
}

ACTION_THRESHOLDS = {
    "edit_delta_p90_min": 0.0025,
    "target_stabilized_motion_p90_min": 0.001,
    "target_to_source_motion_ratio_min": 0.75,
    "paired_visibility_mean_min": 0.70,
    "camera_crossfit_residual_median_max": 0.001,
}
STATIC_THRESHOLDS = {
    "target_stabilized_motion_p90_max": 0.00075,
    "edit_delta_p90_max": 0.0015,
}
CAMERA_THRESHOLDS = {
    "target_raw_motion_p90_min": 0.003,
    "target_stabilized_motion_p90_max": 0.00075,
    "target_camera_residual_reduction_min": 0.65,
}
EFFECT_THRESHOLDS = {
    "target_stabilized_motion_p90_min": 0.001,
    "edit_delta_p90_min": 0.0025,
    "paired_visibility_effect_max": 0.75,
    "target_visibility_drop_min": 0.10,
    "target_acceleration_to_speed_p90_min": 1.75,
}
BOUNDED_ACTION_NEAR_MISS_TIER = "bounded_action_near_miss_v1"
BOUNDED_ACTION_NEAR_MISS_THRESHOLDS = {
    "edit_delta_p90_min": 0.002,
    "target_stabilized_motion_p90_min": 0.00075,
    "target_to_source_motion_ratio_min": 0.5,
    "paired_visibility_mean_min": 0.55,
    "camera_crossfit_residual_median_max": 0.0025,
}
_BOUNDED_ACTION_NEAR_MISS_POLICY = {
    "schema_version": "motive-r10b-bounded-action-near-miss-policy-v1",
    "tier": BOUNDED_ACTION_NEAR_MISS_TIER,
    "applies_only_when": {
        "strict_action_pass": False,
        "upstream_label_class": "positive",
        "atomic_families": ["wave", "quadruped_lie_down"],
    },
    "thresholds": dict(BOUNDED_ACTION_NEAR_MISS_THRESHOLDS),
    "audit_only": True,
    "final_pilot_eligible": False,
}
BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256 = object_digest(
    _BOUNDED_ACTION_NEAR_MISS_POLICY
)
_EXPANSION_EVIDENCE_SCHEMA = (
    "motive-r10b-bounded-action-near-miss-evidence-v1"
)
_EXPANSION_ROW_FIELDS = frozenset(
    {
        "candidate_expansion_tier",
        "candidate_expansion_policy_sha256",
        "candidate_expansion_check_evidence",
        "audit_only",
        "final_pilot_eligible",
    }
)

FINAL_QUOTAS = {
    "positive:wave:adult_human": 1,
    "positive:wave:child_human": 1,
    "positive:wave:character_or_nonhuman": 1,
    "positive:wave:additional_direct_nonreflection": 1,
    "positive:quadruped_lie_down:dog_or_bulldog": 2,
    "positive:quadruped_lie_down:cat": 1,
    "positive:quadruped_lie_down:other_quadruped": 1,
    "control:static:global": 4,
    "control:camera:global": 4,
    "control:effect:global": 4,
}
if sum(FINAL_QUOTAS.values()) != MAX_FINAL_ROWS:  # pragma: no cover
    raise RuntimeError("controlled-pilot quotas no longer sum to 20")

_SCREEN_CELL_QUOTAS = {
    "positive:wave": 4,
    "positive:quadruped_lie_down": 4,
    "static:global": 4,
    "camera:global": 4,
    "effect:global": 4,
}
_SCREEN_CELL_ORDER = tuple(_SCREEN_CELL_QUOTAS)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_NEGATIVE_ROLE_HINT = {
    "instruction_mismatch": "wrong",
    "static": "static",
    "camera_motion": "camera",
    "artifact": "effect",
    "appearance_only": "effect",
}
_AUTHORIZATION = {
    "human_label": False,
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}

_AUDIT_ENUMS = {
    "intended_atomic": frozenset(
        {"wave", "quadruped_lie_down", "other", "ambiguous"}
    ),
    "observed_atomic_or_none": frozenset(
        {"wave", "quadruped_lie_down", "none", "other", "ambiguous"}
    ),
    "subject_morphology": frozenset(
        {
            "adult_human",
            "child_human",
            "character_or_nonhuman",
            "dog",
            "bulldog",
            "cat",
            "other_quadruped",
            "other",
            "ambiguous",
        }
    ),
    "onset": frozenset({"clear", "weak", "none", "ambiguous"}),
    "periodicity": frozenset({"repeated", "single", "none", "ambiguous"}),
    "direction": frozenset(
        {"toward_viewer", "away", "lateral", "other", "none", "ambiguous"}
    ),
    "success": frozenset({"yes", "no", "ambiguous"}),
    "actor_motion": frozenset({"clear", "weak", "none", "ambiguous"}),
    "camera_motion": frozenset({"none", "low", "high", "ambiguous"}),
    "identity_appearance_change": frozenset(
        {"none", "low", "high", "ambiguous"}
    ),
    "nonphysical_effect": frozenset({"none", "low", "high", "ambiguous"}),
    "deformation": frozenset({"none", "low", "high", "ambiguous"}),
    "flicker": frozenset({"none", "low", "high", "ambiguous"}),
    "confidence": frozenset({"low", "medium", "high"}),
    "reflection_or_sunglasses_artifact": frozenset(
        {"none", "present", "ambiguous"}
    ),
    "secondary_action": frozenset(
        {"none", "head_tilt", "stretch", "other", "ambiguous"}
    ),
}
_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "iid",
        "queue_row_sha256",
        "qwen_model_id",
        "qwen_prompt_sha256",
        "source_state",
        "target_state",
        *_AUDIT_ENUMS,
    }
)


class R10BBerniniPilotError(ValueError):
    """A pilot queue, audit, binding, or quota contract is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise R10BBerniniPilotError(
            f"{field} must be one lowercase SHA-256 digest"
        )
    return value


def _plain_string(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise R10BBerniniPilotError(f"{field} must be one canonical string")
    return value


def _relative_media_path(value: Any, *, field: str) -> str:
    text = _plain_string(value, field=field)
    if "\\" in text:
        raise R10BBerniniPilotError(f"{field} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise R10BBerniniPilotError(
            f"{field} must be normalized below the bound data_root"
        )
    return text


def _data_root(value: Any, *, field: str) -> str:
    text = _plain_string(value, field=field)
    if (
        not os.path.isabs(text)
        or text.startswith("//")
        or os.path.normpath(text) != text
    ):
        raise R10BBerniniPilotError(
            f"{field} must be one normalized absolute path"
        )
    return text


def _regular_file(path: str | Path, *, field: str) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise R10BBerniniPilotError(
            f"{field} must be a regular non-symlink file: {expanded}"
        )
    return expanded.resolve(strict=True)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _load_jsonl(path: str | Path, *, field: str) -> tuple[list[dict[str, Any]], bytes]:
    resolved = _regular_file(path, field=field)
    raw = resolved.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise R10BBerniniPilotError(
            f"{field} must be non-empty canonical newline-terminated JSONL"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise R10BBerniniPilotError(
                f"{field}:{line_number} contains a blank line"
            )
        try:
            value = json.loads(
                line.decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise R10BBerniniPilotError(
                f"{field}:{line_number} is not strict JSON"
            ) from error
        if not isinstance(value, dict):
            raise R10BBerniniPilotError(
                f"{field}:{line_number} must contain one JSON object"
            )
        rows.append(value)
    return rows, raw


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
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
    return "".join(canonical_json(dict(row)) + "\n" for row in rows).encode(
        "utf-8"
    )


def _atomic_directory(output_dir: str | Path, files: Mapping[str, bytes]) -> None:
    output = Path(output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        for name, payload in files.items():
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _safe_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R10BBerniniPilotError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise R10BBerniniPilotError(f"{field} must be finite")
    return number


def _motion_p90(
    tracks: np.ndarray,
    visibility: np.ndarray,
) -> tuple[float, float]:
    coordinates = np.asarray(tracks, dtype=np.float64)
    visible = np.asarray(visibility, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise R10BBerniniPilotError("track coordinates must have shape [F,N,2]")
    if visible.shape != coordinates.shape[:2]:
        raise R10BBerniniPilotError("track visibility shape differs")
    transition_visibility = np.minimum(visible[:-1], visible[1:]).clip(0.0, 1.0)
    speed = np.linalg.norm(np.diff(coordinates, axis=0), axis=-1)
    weighted_speed = speed * transition_visibility
    acceleration = np.zeros_like(speed)
    if len(speed) > 1:
        acceleration[1:] = np.abs(np.diff(speed, axis=0))
        acceleration[0] = acceleration[1]
    weighted_acceleration = acceleration * transition_visibility
    return (
        float(np.quantile(weighted_speed, 0.90)),
        float(np.quantile(weighted_acceleration, 0.90)),
    )


def _row_metrics(
    arrays: Mapping[str, np.ndarray],
    cache_index: int,
) -> dict[str, float | bool]:
    source_stabilized = np.asarray(
        arrays["source_stabilized_tracks"][cache_index]
    )
    target_stabilized = np.asarray(
        arrays["target_stabilized_tracks"][cache_index]
    )
    source_visibility = np.asarray(arrays["source_visibility"][cache_index])
    target_visibility = np.asarray(arrays["target_visibility"][cache_index])
    source_p90, _source_accel = _motion_p90(
        source_stabilized, source_visibility
    )
    target_p90, target_accel = _motion_p90(
        target_stabilized, target_visibility
    )
    target_raw_p90, _target_raw_accel = _motion_p90(
        arrays["target_normalized_tracks"][cache_index],
        target_visibility,
    )
    _edit_velocity, edit_magnitude, _midpoint = track_delta_components(
        source_stabilized,
        target_stabilized,
        source_visibility,
        target_visibility,
    )
    edit_delta_p90 = float(np.quantile(edit_magnitude, 0.90))
    source_visibility_mean = float(np.mean(source_visibility))
    target_visibility_mean = float(np.mean(target_visibility))
    paired_visibility_mean = float(
        np.mean(np.minimum(source_visibility, target_visibility))
    )
    visibility_drop = max(
        0.0, source_visibility_mean - target_visibility_mean
    )
    camera_residual = max(
        float(
            arrays["source_camera_crossfit_residual_median"][cache_index]
        ),
        float(
            arrays["target_camera_crossfit_residual_median"][cache_index]
        ),
    )
    target_camera_reduction = float(
        arrays["target_camera_crossfit_residual_reduction"][cache_index]
    )
    acceleration_ratio = target_accel / max(target_p90, 1e-12)
    paired_valid = all(
        bool(arrays[name][cache_index])
        for name in (
            "source_track_valid",
            "target_track_valid",
            "source_camera_valid",
            "target_camera_valid",
            "source_camera_crossfit_valid",
            "target_camera_crossfit_valid",
        )
    )
    return {
        "paired_track_camera_crossfit_valid": paired_valid,
        "source_stabilized_motion_p90": source_p90,
        "target_stabilized_motion_p90": target_p90,
        "target_raw_motion_p90": target_raw_p90,
        "edit_delta_p90": edit_delta_p90,
        "source_visibility_mean": source_visibility_mean,
        "target_visibility_mean": target_visibility_mean,
        "paired_visibility_mean": paired_visibility_mean,
        "target_visibility_drop": visibility_drop,
        "camera_crossfit_residual_median_max": camera_residual,
        "target_camera_residual_reduction": target_camera_reduction,
        "target_acceleration_p90": target_accel,
        "target_acceleration_to_speed_p90": acceleration_ratio,
    }


def _screen_gates(metrics: Mapping[str, float | bool]) -> dict[str, Any]:
    valid = bool(metrics["paired_track_camera_crossfit_valid"])
    visibility = float(metrics["paired_visibility_mean"])
    residual = float(metrics["camera_crossfit_residual_median_max"])
    base = (
        valid
        and visibility >= ACTION_THRESHOLDS["paired_visibility_mean_min"]
        and residual
        <= ACTION_THRESHOLDS["camera_crossfit_residual_median_max"]
    )
    source = float(metrics["source_stabilized_motion_p90"])
    target = float(metrics["target_stabilized_motion_p90"])
    delta = float(metrics["edit_delta_p90"])
    action_checks = {
        "paired_track_camera_crossfit_valid": valid,
        "paired_visibility_mean": (
            visibility >= ACTION_THRESHOLDS["paired_visibility_mean_min"]
        ),
        "camera_crossfit_residual_median": (
            residual
            <= ACTION_THRESHOLDS["camera_crossfit_residual_median_max"]
        ),
        "edit_delta_p90": (
            delta >= ACTION_THRESHOLDS["edit_delta_p90_min"]
        ),
        "target_stabilized_motion_p90": (
            target
            >= ACTION_THRESHOLDS["target_stabilized_motion_p90_min"]
        ),
        "target_to_source_motion_ratio": (
            target
            >= ACTION_THRESHOLDS["target_to_source_motion_ratio_min"]
            * source
        ),
    }
    static_checks = {
        "base_quality": base,
        "target_stabilized_motion_p90": (
            target
            <= STATIC_THRESHOLDS["target_stabilized_motion_p90_max"]
        ),
        "edit_delta_p90": (
            delta <= STATIC_THRESHOLDS["edit_delta_p90_max"]
        ),
    }
    camera_checks = {
        "base_quality": base,
        "target_raw_motion_p90": (
            float(metrics["target_raw_motion_p90"])
            >= CAMERA_THRESHOLDS["target_raw_motion_p90_min"]
        ),
        "target_stabilized_motion_p90": (
            target
            <= CAMERA_THRESHOLDS["target_stabilized_motion_p90_max"]
        ),
        "target_camera_residual_reduction": (
            float(metrics["target_camera_residual_reduction"])
            >= CAMERA_THRESHOLDS[
                "target_camera_residual_reduction_min"
            ]
        ),
    }
    effect_proxy = (
        visibility <= EFFECT_THRESHOLDS["paired_visibility_effect_max"]
        or float(metrics["target_visibility_drop"])
        >= EFFECT_THRESHOLDS["target_visibility_drop_min"]
        or float(metrics["target_acceleration_to_speed_p90"])
        >= EFFECT_THRESHOLDS[
            "target_acceleration_to_speed_p90_min"
        ]
    )
    effect_checks = {
        "base_quality": base,
        "target_stabilized_motion_p90": (
            target
            >= EFFECT_THRESHOLDS["target_stabilized_motion_p90_min"]
        ),
        "edit_delta_p90": (
            delta >= EFFECT_THRESHOLDS["edit_delta_p90_min"]
        ),
        "effect_proxy": effect_proxy,
    }
    return {
        "action": {
            "applicable_to": ["positive", "wrong"],
            "checks": action_checks,
            "pass": all(action_checks.values()),
        },
        "static": {
            "applicable_to": ["static"],
            "checks": static_checks,
            "pass": all(static_checks.values()),
        },
        "camera": {
            "applicable_to": ["camera"],
            "checks": camera_checks,
            "pass": all(camera_checks.values()),
        },
        "effect": {
            "applicable_to": ["effect"],
            "checks": effect_checks,
            "pass": all(effect_checks.values()),
        },
    }


def _bounded_action_near_miss_checks(
    metrics: Mapping[str, float | bool],
    *,
    strict_action_pass: bool,
) -> dict[str, bool]:
    source = float(metrics["source_stabilized_motion_p90"])
    target = float(metrics["target_stabilized_motion_p90"])
    return {
        "strict_action_failed": not strict_action_pass,
        "paired_track_camera_crossfit_valid": bool(
            metrics["paired_track_camera_crossfit_valid"]
        ),
        "edit_delta_p90": (
            float(metrics["edit_delta_p90"])
            >= BOUNDED_ACTION_NEAR_MISS_THRESHOLDS["edit_delta_p90_min"]
        ),
        "target_stabilized_motion_p90": (
            target
            >= BOUNDED_ACTION_NEAR_MISS_THRESHOLDS[
                "target_stabilized_motion_p90_min"
            ]
        ),
        "target_to_source_motion_ratio": (
            target
            >= BOUNDED_ACTION_NEAR_MISS_THRESHOLDS[
                "target_to_source_motion_ratio_min"
            ]
            * source
        ),
        "paired_visibility_mean": (
            float(metrics["paired_visibility_mean"])
            >= BOUNDED_ACTION_NEAR_MISS_THRESHOLDS[
                "paired_visibility_mean_min"
            ]
        ),
        "camera_crossfit_residual_median": (
            float(metrics["camera_crossfit_residual_median_max"])
            <= BOUNDED_ACTION_NEAR_MISS_THRESHOLDS[
                "camera_crossfit_residual_median_max"
            ]
        ),
    }


def _candidate_expansion_evidence(
    metrics: Mapping[str, float | bool],
    *,
    strict_action_pass: bool,
) -> dict[str, Any]:
    checks = _bounded_action_near_miss_checks(
        metrics,
        strict_action_pass=strict_action_pass,
    )
    return {
        "schema_version": _EXPANSION_EVIDENCE_SCHEMA,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _candidate_family(row: Mapping[str, Any]) -> str | None:
    label = row.get("label")
    if not isinstance(label, Mapping):
        return None
    atomic = strict_atomic_family(
        str(label.get("primary_family", "")),
        str(row.get("prompt", "")),
    )
    if atomic == "wave":
        return "wave"
    if atomic == "lie_down":
        return "quadruped_lie_down"
    return None


def _role_hint(row: Mapping[str, Any]) -> str | None:
    label = row.get("label")
    if not isinstance(label, Mapping):
        return None
    if label.get("class") == "positive":
        return "positive"
    if label.get("class") != "negative":
        return None
    return _NEGATIVE_ROLE_HINT.get(str(label.get("negative_type", "")))


def _media_binding(
    row: Mapping[str, Any],
    *,
    iid: str,
) -> dict[str, Any]:
    source_bindings = row.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise R10BBerniniPilotError(f"iid={iid} lacks source_bindings")
    media = source_bindings.get("media")
    if not isinstance(media, Mapping):
        raise R10BBerniniPilotError(f"iid={iid} lacks media bindings")
    data_root = _data_root(media.get("data_root"), field=f"iid={iid} data_root")
    output: dict[str, Any] = {"data_root": data_root}
    for side, upstream in (("src_video", "src_video"), ("tgt_video", "tgt_video")):
        value = media.get(upstream)
        if not isinstance(value, Mapping):
            raise R10BBerniniPilotError(
                f"iid={iid} lacks {upstream} media binding"
            )
        output[side] = {
            "relative_path": _relative_media_path(
                value.get("relative_path"),
                field=f"iid={iid} {upstream}.relative_path",
            ),
            "sha256": _require_sha256(
                value.get("sha256"),
                field=f"iid={iid} {upstream}.sha256",
            ),
        }
    return output


def _validate_track_binding(
    *,
    candidate: Mapping[str, Any],
    track_row: Mapping[str, Any],
    cache_index: int,
    media: Mapping[str, Any],
) -> int:
    iid = str(candidate["iid"])
    if track_row.get("iid") != iid:
        raise R10BBerniniPilotError(f"iid={iid} track IID differs")
    input_index = track_row.get("input_index")
    if (
        isinstance(input_index, bool)
        or not isinstance(input_index, int)
        or input_index < 0
    ):
        raise R10BBerniniPilotError(
            f"iid={iid} track input_index is invalid"
        )
    if (
        "final_array_index" in track_row
        and track_row.get("final_array_index") != cache_index
    ):
        raise R10BBerniniPilotError(
            f"iid={iid} final_array_index/cache index differs"
        )
    input_row_sha = track_row.get("input_row_sha256")
    if input_row_sha is not None and input_row_sha != object_digest(candidate):
        raise R10BBerniniPilotError(
            f"iid={iid} candidate/track input-row digest differs"
        )
    if track_row.get("input_digest") not in {
        None,
        candidate.get("input_digest"),
    }:
        raise R10BBerniniPilotError(
            f"iid={iid} candidate/track input_digest differs"
        )
    for side, key in (("source", "src_video"), ("target", "tgt_video")):
        record = track_row.get(side)
        if isinstance(record, Mapping) and record.get("video_sha256") is not None:
            if record.get("video_sha256") != media[key]["sha256"]:
                raise R10BBerniniPilotError(
                    f"iid={iid} {side} media/track SHA-256 differs"
                )
    if track_row.get("paired_track_valid") is not True:
        raise R10BBerniniPilotError(f"iid={iid} paired tracks are invalid")
    if track_row.get("paired_camera_valid") is not True:
        raise R10BBerniniPilotError(f"iid={iid} paired camera is invalid")
    return int(input_index)


def _screen_rank_key(role: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["feature_metrics"]
    if role in {"positive", "wrong"}:
        quality = (
            -float(metrics["edit_delta_p90"]),
            -float(metrics["target_stabilized_motion_p90"]),
        )
    elif role == "static":
        quality = (
            float(metrics["target_stabilized_motion_p90"]),
            float(metrics["edit_delta_p90"]),
        )
    elif role == "camera":
        quality = (
            -float(metrics["target_raw_motion_p90"]),
            float(metrics["target_stabilized_motion_p90"]),
            -float(metrics["target_camera_residual_reduction"]),
        )
    else:
        quality = (
            -float(metrics["edit_delta_p90"]),
            -float(metrics["target_acceleration_to_speed_p90"]),
        )
    return (*quality, row["selection_key_sha256"], row["iid"])


def derive_qwen_audit_queue(
    *,
    candidate_manifest: str | Path,
    track_manifest: str | Path,
    track_cache: str | Path,
    qwen_model_id: str,
    qwen_prompt_sha256: str,
    audit_oversample: int = 4,
    seed: int = SELECTION_SEED,
    candidate_expansion_tier: str | None = None,
) -> dict[str, Any]:
    """Derive an auditable, component-disjoint Qwen queue without video I/O."""

    if (
        isinstance(audit_oversample, bool)
        or not isinstance(audit_oversample, int)
        or audit_oversample <= 0
    ):
        raise R10BBerniniPilotError("audit_oversample must be positive")
    if candidate_expansion_tier not in {
        None,
        BOUNDED_ACTION_NEAR_MISS_TIER,
    }:
        raise R10BBerniniPilotError(
            "candidate_expansion_tier must be None or "
            f"{BOUNDED_ACTION_NEAR_MISS_TIER!r}"
        )
    model_id = _plain_string(qwen_model_id, field="qwen_model_id")
    prompt_sha = _require_sha256(
        qwen_prompt_sha256, field="qwen_prompt_sha256"
    )
    candidate_path = _regular_file(
        candidate_manifest, field="candidate_manifest"
    )
    track_manifest_path = _regular_file(
        track_manifest, field="track_manifest"
    )
    track_cache_path = _regular_file(track_cache, field="track_cache")
    candidates, candidate_raw = _load_jsonl(
        candidate_path, field="candidate_manifest"
    )
    track_rows, track_raw = _load_jsonl(
        track_manifest_path, field="track_manifest"
    )
    if len(candidates) != len(track_rows):
        raise R10BBerniniPilotError(
            "candidate and track manifests must have identical row counts"
        )
    candidate_iids = [str(row.get("iid", "")) for row in candidates]
    if any(not iid for iid in candidate_iids) or len(set(candidate_iids)) != len(
        candidate_iids
    ):
        raise R10BBerniniPilotError(
            "candidate manifest has missing/duplicate IID"
        )
    track_by_iid: dict[str, Mapping[str, Any]] = {}
    for row in track_rows:
        iid = str(row.get("iid", ""))
        if not iid or iid in track_by_iid:
            raise R10BBerniniPilotError(
                "track manifest has missing/duplicate IID"
            )
        track_by_iid[iid] = row
    if set(track_by_iid) != set(candidate_iids):
        raise R10BBerniniPilotError(
            "candidate/track IID coverage differs"
        )

    required_arrays = {
        "input_indices",
        "source_normalized_tracks",
        "target_normalized_tracks",
        "source_stabilized_tracks",
        "target_stabilized_tracks",
        "source_visibility",
        "target_visibility",
        "source_track_valid",
        "target_track_valid",
        "source_camera_valid",
        "target_camera_valid",
        "source_camera_crossfit_valid",
        "target_camera_crossfit_valid",
        "source_camera_crossfit_residual_median",
        "target_camera_crossfit_residual_median",
        "target_camera_crossfit_residual_reduction",
    }
    with np.load(track_cache_path, allow_pickle=False) as archive:
        missing = sorted(required_arrays - set(archive.files))
        if missing:
            raise R10BBerniniPilotError(
                f"track cache lacks required arrays: {missing}"
            )
        arrays = {name: np.asarray(archive[name]) for name in required_arrays}
    try:
        validate_track_cache_arrays(arrays)
    except (KeyError, R10BTangentError) as error:
        raise R10BBerniniPilotError(str(error)) from error
    rows_in_cache = len(arrays["input_indices"])
    for name in required_arrays - {
        "source_normalized_tracks",
        "target_normalized_tracks",
        "source_stabilized_tracks",
        "target_stabilized_tracks",
        "source_visibility",
        "target_visibility",
    }:
        if np.asarray(arrays[name]).shape[0] != rows_in_cache:
            raise R10BBerniniPilotError(
                f"track cache array {name} row count differs"
            )
    input_to_cache = {
        int(input_index): cache_index
        for cache_index, input_index in enumerate(arrays["input_indices"])
    }

    source_digests = {
        "candidate_manifest": _sha256_bytes(candidate_raw),
        "track_manifest": _sha256_bytes(track_raw),
        "track_cache": file_digest(track_cache_path),
    }
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expansion_pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    for candidate_index, candidate in enumerate(candidates):
        iid = str(candidate["iid"])
        label = candidate.get("label")
        assignment = candidate.get("assignment")
        if not isinstance(label, Mapping) or not isinstance(
            assignment, Mapping
        ):
            exclusions["missing_label_or_assignment"] += 1
            continue
        if (
            assignment.get("fresh") is not True
            or assignment.get("split") == "test"
        ):
            exclusions["not_fresh_non_test"] += 1
            continue
        component_id = assignment.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            exclusions["missing_component_id"] += 1
            continue
        track_row = track_by_iid[iid]
        input_index = track_row.get("input_index")
        if (
            isinstance(input_index, bool)
            or not isinstance(input_index, int)
            or input_index not in input_to_cache
        ):
            exclusions["track_input_index_missing"] += 1
            continue
        cache_index = input_to_cache[input_index]
        try:
            media = _media_binding(candidate, iid=iid)
            track_input_index = _validate_track_binding(
                candidate=candidate,
                track_row=track_row,
                cache_index=cache_index,
                media=media,
            )
        except R10BBerniniPilotError:
            exclusions["binding_or_paired_validity_failed"] += 1
            continue
        metrics = _row_metrics(arrays, cache_index)
        gates = _screen_gates(metrics)
        intended_family = _candidate_family(candidate)
        upstream_role = _role_hint(candidate)
        hypotheses: list[tuple[str, str, str, str | None]] = []
        positive_hypothesis_added = False
        if intended_family is not None and upstream_role == "positive":
            if bool(gates["action"]["pass"]):
                hypotheses.append(
                    (
                        "positive",
                        intended_family,
                        f"positive:{intended_family}",
                        None,
                    )
                )
                positive_hypothesis_added = True
            else:
                exclusions["positive_feature_gate_failed"] += 1
                if (
                    candidate_expansion_tier
                    == BOUNDED_ACTION_NEAR_MISS_TIER
                ):
                    expansion_evidence = _candidate_expansion_evidence(
                        metrics,
                        strict_action_pass=False,
                    )
                    if expansion_evidence["pass"] is True:
                        hypotheses.append(
                            (
                                "positive",
                                intended_family,
                                f"positive:{intended_family}",
                                BOUNDED_ACTION_NEAR_MISS_TIER,
                            )
                        )
        # These are feature-screen hypotheses, not assigned labels.  Qwen's
        # frame-indexed blind audit must confirm the final nuisance role.
        # Assign one hypothesis with effect > camera > static precedence so a
        # camera-dominated clip is not consumed by the static queue merely
        # because stabilization removes its global motion.
        if not positive_hypothesis_added:
            nuisance_family = intended_family or "other"
            nuisance_role = next(
                (
                    role
                    for role in ("effect", "camera", "static")
                    if bool(gates[role]["pass"])
                ),
                None,
            )
            if nuisance_role is not None:
                hypotheses.append(
                    (
                        nuisance_role,
                        nuisance_family,
                        f"{nuisance_role}:global",
                        None,
                    )
                )
        if not hypotheses:
            exclusions["no_feature_screen_hypothesis"] += 1
            continue
        original_prompt = _plain_string(
            candidate.get("prompt"), field=f"iid={iid} original prompt"
        )
        input_digest = _require_sha256(
            candidate.get("input_digest"),
            field=f"iid={iid} input_digest",
        )
        for (
            role,
            hypothesis_family,
            screen_cell,
            expansion_tier,
        ) in hypotheses:
            if screen_cell not in _SCREEN_CELL_QUOTAS:
                raise RuntimeError(
                    f"internal unsupported screen cell: {screen_cell}"
                )
            selection_contract = {
                "schema_version": QUEUE_ROW_SCHEMA,
                "seed": int(seed),
                "iid": iid,
                "screen_cell": screen_cell,
                "component_id": component_id,
            }
            if expansion_tier is not None:
                selection_contract["candidate_expansion_tier"] = (
                    expansion_tier
                )
            selection_key = object_digest(selection_contract)
            if hypothesis_family in CROSS_FAMILY:
                canonical_prompt = CANONICAL_PROMPTS[hypothesis_family]
                cross_family = CROSS_FAMILY[hypothesis_family]
                cross_prompt = CANONICAL_PROMPTS[cross_family]
            else:
                canonical_prompt = original_prompt
                cross_family = "none"
                cross_prompt = NOOP_PROMPT
            queue_row = {
                "schema_version": QUEUE_ROW_SCHEMA,
                "iid": iid,
                "candidate_input_digest": input_digest,
                "candidate_row_index": int(candidate_index),
                "candidate_row_sha256": object_digest(candidate),
                "component_id": component_id,
                "source_split": str(assignment.get("split")),
                "fresh": True,
                "upstream_label": {
                    "class": str(label.get("class")),
                    "negative_type": label.get("negative_type"),
                    "primary_family": str(label.get("primary_family")),
                    "provenance_kind": str(
                        label.get("provenance_kind", "")
                    ),
                    "human_label": False,
                },
                "screen_role_hint": role,
                "screen_cell": screen_cell,
                "screen_role_is_feature_hypothesis": role
                in {"static", "camera", "effect"},
                "intended_family": hypothesis_family,
                "original_prompt": original_prompt,
                "canonical_prompt": canonical_prompt,
                "prompt_variants": {
                    "canonical": canonical_prompt,
                    "noop": NOOP_PROMPT,
                    "cross_family_shuffle": cross_prompt,
                    "cross_family_shuffle_family": cross_family,
                },
                "feature_metrics": metrics,
                "feature_gates": gates,
                "motion_gate_applicable": role == "positive",
                "motion_gate_pass": (
                    bool(gates["action"]["pass"])
                    if role == "positive"
                    else False
                ),
                "media_binding": media,
                "source_bindings": dict(candidate["source_bindings"]),
                "track_binding": {
                    "track_input_index": track_input_index,
                    "track_cache_index": int(cache_index),
                    "track_row_sha256": object_digest(track_row),
                    "track_input_row_sha256": track_row.get(
                        "input_row_sha256"
                    ),
                    "track_manifest_sha256": source_digests[
                        "track_manifest"
                    ],
                    "track_cache_sha256": source_digests["track_cache"],
                },
                "qwen_audit_binding": {
                    "schema_version": AUDIT_ROW_SCHEMA,
                    "qwen_model_id": model_id,
                    "qwen_prompt_sha256": prompt_sha,
                    "required_observation_fields": sorted(
                        {
                            "source_state",
                            "target_state",
                            *_AUDIT_ENUMS,
                        }
                    ),
                    "semantic_precedence": [
                        "effect",
                        "camera",
                        "wrong",
                        "static",
                        "positive_or_reject",
                    ],
                },
                "selection_seed": int(seed),
                "selection_key_sha256": selection_key,
                "screen_rank": 0,
                "authorization": dict(_AUTHORIZATION),
                "formal_evidence": False,
                "renderer_probe_authorized": False,
                "training_authorized": False,
            }
            if expansion_tier is not None:
                queue_row.update(
                    {
                        "candidate_expansion_tier": expansion_tier,
                        "candidate_expansion_policy_sha256": (
                            BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256
                        ),
                        "candidate_expansion_check_evidence": (
                            _candidate_expansion_evidence(
                                metrics,
                                strict_action_pass=bool(
                                    gates["action"]["pass"]
                                ),
                            )
                        ),
                        "audit_only": True,
                        "final_pilot_eligible": False,
                    }
                )
                expansion_pools[screen_cell].append(queue_row)
            else:
                pools[screen_cell].append(queue_row)

    selected: list[dict[str, Any]] = []
    used_components: set[str] = set()
    selected_by_cell: Counter[str] = Counter()
    regular_eligible_by_cell: dict[str, int] = {}
    for cell in _SCREEN_CELL_ORDER:
        role = cell.split(":", 1)[0]
        ranked = sorted(
            pools.get(cell, ()),
            key=lambda row: _screen_rank_key(role, row),
        )
        regular_eligible_by_cell[cell] = len(ranked)
        target = _SCREEN_CELL_QUOTAS[cell] * audit_oversample
        chosen: list[dict[str, Any]] = []
        for row in ranked:
            if row["component_id"] in used_components:
                continue
            row = dict(row)
            row["screen_rank"] = len(chosen) + 1
            chosen.append(row)
            used_components.add(str(row["component_id"]))
            if len(chosen) == target:
                break
        selected.extend(chosen)
        selected_by_cell[cell] += len(chosen)

    expansion_selected: list[dict[str, Any]] = []
    for cell in ("positive:wave", "positive:quadruped_lie_down"):
        target = _SCREEN_CELL_QUOTAS[cell] * audit_oversample
        missing = target - selected_by_cell[cell]
        if missing <= 0:
            continue
        ranked = sorted(
            expansion_pools.get(cell, ()),
            key=lambda row: _screen_rank_key("positive", row),
        )
        for row in ranked:
            if row["component_id"] in used_components:
                continue
            chosen = dict(row)
            chosen["screen_rank"] = selected_by_cell[cell] + 1
            expansion_selected.append(chosen)
            selected_by_cell[cell] += 1
            used_components.add(str(chosen["component_id"]))
            if selected_by_cell[cell] == target:
                break
    # Strict rows and nuisance controls retain priority, including component
    # ownership and queue order.  Audit-only expansions are appended last.
    selected.extend(expansion_selected)

    screen_shortfalls: dict[str, Any] = {}
    for cell in _SCREEN_CELL_ORDER:
        target = _SCREEN_CELL_QUOTAS[cell] * audit_oversample
        if selected_by_cell[cell] < target:
            eligible = regular_eligible_by_cell[cell]
            if cell.startswith("positive:"):
                eligible += len(expansion_pools.get(cell, ()))
            screen_shortfalls[cell] = {
                "target": target,
                "selected": selected_by_cell[cell],
                "eligible_before_component_dedup": eligible,
            }
    for queue_rank, row in enumerate(selected, 1):
        row["queue_rank"] = queue_rank
    queue_bytes = _jsonl_bytes(selected)
    return {
        "rows": selected,
        "summary": {
            "schema_version": QUEUE_SUMMARY_SCHEMA,
            "experiment_role": "qwen_audit_queue_only",
            "selection_seed": int(seed),
            "audit_oversample": int(audit_oversample),
            "rows": len(selected),
            "unique_components": len(used_components),
            "component_disjoint": len(used_components) == len(selected),
            "screen_cell_counts": dict(
                sorted(Counter(row["screen_cell"] for row in selected).items())
            ),
            "screen_shortfalls": screen_shortfalls,
            "thresholds": {
                "action": dict(ACTION_THRESHOLDS),
                "static": dict(STATIC_THRESHOLDS),
                "camera": dict(CAMERA_THRESHOLDS),
                "effect": dict(EFFECT_THRESHOLDS),
            },
            "candidate_expansion": {
                "tier": candidate_expansion_tier,
                "policy": (
                    dict(_BOUNDED_ACTION_NEAR_MISS_POLICY)
                    if candidate_expansion_tier is not None
                    else None
                ),
                "policy_sha256": (
                    BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256
                    if candidate_expansion_tier is not None
                    else None
                ),
                "eligible_before_component_dedup": sum(
                    len(rows) for rows in expansion_pools.values()
                ),
                "selected_rows": len(expansion_selected),
                "selected_cell_counts": dict(
                    sorted(
                        Counter(
                            row["screen_cell"]
                            for row in expansion_selected
                        ).items()
                    )
                ),
                "audit_only": True,
                "final_pilot_eligible": False,
            },
            "qwen_audit": {
                "schema_version": AUDIT_ROW_SCHEMA,
                "qwen_model_id": model_id,
                "qwen_prompt_sha256": prompt_sha,
                "semantic_precedence": [
                    "effect",
                    "camera",
                    "wrong",
                    "static",
                    "positive_or_reject",
                ],
            },
            "inputs": {
                "candidate_manifest": str(candidate_path),
                "candidate_manifest_sha256": source_digests[
                    "candidate_manifest"
                ],
                "track_manifest": str(track_manifest_path),
                "track_manifest_sha256": source_digests[
                    "track_manifest"
                ],
                "track_cache": str(track_cache_path),
                "track_cache_sha256": source_digests["track_cache"],
            },
            "queue_sha256": _sha256_bytes(queue_bytes),
            "exclusion_counts": dict(sorted(exclusions.items())),
            "video_bytes_copied": False,
            "authorization": dict(_AUTHORIZATION),
        },
    }


def write_qwen_audit_queue(
    payload: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    rows = list(payload["rows"])
    summary = dict(payload["summary"])
    queue_bytes = _jsonl_bytes(rows)
    summary_bytes = _pretty_bytes(summary)
    if summary.get("queue_sha256") != _sha256_bytes(queue_bytes):
        raise R10BBerniniPilotError("queue payload digest differs")
    done = {
        "schema_version": QUEUE_DONE_SCHEMA,
        "rows": len(rows),
        "files": {
            QUEUE_NAME: _sha256_bytes(queue_bytes),
            QUEUE_SUMMARY_NAME: _sha256_bytes(summary_bytes),
        },
        "authorization": dict(_AUTHORIZATION),
    }
    _atomic_directory(
        output_dir,
        {
            QUEUE_NAME: queue_bytes,
            QUEUE_SUMMARY_NAME: summary_bytes,
            QUEUE_DONE_NAME: _pretty_bytes(done),
        },
    )


def _load_json_object(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    resolved = _regular_file(path, field=field)
    raw = resolved.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise R10BBerniniPilotError(f"{field} is not strict JSON") from error
    if not isinstance(value, dict):
        raise R10BBerniniPilotError(f"{field} must be one JSON object")
    return value, raw


def _validate_candidate_expansion_row(
    row: Mapping[str, Any],
    *,
    index: int,
) -> bool:
    present = _EXPANSION_ROW_FIELDS.intersection(row)
    if not present:
        return False
    if present != _EXPANSION_ROW_FIELDS:
        raise R10BBerniniPilotError(
            f"queue row {index} candidate expansion fields are incomplete"
        )
    if (
        row.get("candidate_expansion_tier")
        != BOUNDED_ACTION_NEAR_MISS_TIER
        or row.get("candidate_expansion_policy_sha256")
        != BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256
        or row.get("audit_only") is not True
        or row.get("final_pilot_eligible") is not False
        or row.get("screen_role_hint") != "positive"
        or row.get("screen_cell")
        not in {"positive:wave", "positive:quadruped_lie_down"}
        or row.get("intended_family")
        not in {"wave", "quadruped_lie_down"}
    ):
        raise R10BBerniniPilotError(
            f"queue row {index} candidate expansion contract differs"
        )
    upstream = row.get("upstream_label")
    metrics = row.get("feature_metrics")
    gates = row.get("feature_gates")
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("class") != "positive"
        or not isinstance(metrics, Mapping)
        or not isinstance(gates, Mapping)
    ):
        raise R10BBerniniPilotError(
            f"queue row {index} candidate expansion applicability differs"
        )
    recomputed_gates = _screen_gates(metrics)
    if gates != recomputed_gates or recomputed_gates["action"]["pass"] is not False:
        raise R10BBerniniPilotError(
            f"queue row {index} strict feature gates differ"
        )
    expected_evidence = _candidate_expansion_evidence(
        metrics,
        strict_action_pass=False,
    )
    if (
        row.get("candidate_expansion_check_evidence") != expected_evidence
        or expected_evidence["pass"] is not True
        or row.get("motion_gate_applicable") is not True
        or row.get("motion_gate_pass") is not False
    ):
        raise R10BBerniniPilotError(
            f"queue row {index} candidate expansion evidence differs"
        )
    return True


def _load_queue_commit(
    queue_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    root = Path(queue_dir).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise R10BBerniniPilotError("queue_dir must be a non-symlink directory")
    expected_names = {
        QUEUE_NAME,
        QUEUE_SUMMARY_NAME,
        QUEUE_DONE_NAME,
    }
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != expected_names:
        raise R10BBerniniPilotError(
            "queue directory closure differs; expected exactly "
            f"{sorted(expected_names)}"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise R10BBerniniPilotError(
                "queue artifacts must be regular non-symlink files: "
                f"{entry}"
            )
    paths = {
        QUEUE_NAME: root / QUEUE_NAME,
        QUEUE_SUMMARY_NAME: root / QUEUE_SUMMARY_NAME,
        QUEUE_DONE_NAME: root / QUEUE_DONE_NAME,
    }
    rows, queue_raw = _load_jsonl(paths[QUEUE_NAME], field=QUEUE_NAME)
    summary, summary_raw = _load_json_object(
        paths[QUEUE_SUMMARY_NAME], field=QUEUE_SUMMARY_NAME
    )
    done, _done_raw = _load_json_object(
        paths[QUEUE_DONE_NAME], field=QUEUE_DONE_NAME
    )
    if summary.get("schema_version") != QUEUE_SUMMARY_SCHEMA:
        raise R10BBerniniPilotError("queue summary schema differs")
    if done.get("schema_version") != QUEUE_DONE_SCHEMA:
        raise R10BBerniniPilotError("queue done schema differs")
    expected_files = {
        QUEUE_NAME: _sha256_bytes(queue_raw),
        QUEUE_SUMMARY_NAME: _sha256_bytes(summary_raw),
    }
    if done.get("files") != expected_files:
        raise R10BBerniniPilotError("queue commit file digests differ")
    if (
        done.get("rows") != len(rows)
        or summary.get("rows") != len(rows)
        or summary.get("queue_sha256") != expected_files[QUEUE_NAME]
    ):
        raise R10BBerniniPilotError("queue row count/digest differs")
    seen_iids: set[str] = set()
    seen_components: set[str] = set()
    expansion_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("schema_version") != QUEUE_ROW_SCHEMA:
            raise R10BBerniniPilotError(f"queue row {index} schema differs")
        iid = _plain_string(row.get("iid"), field=f"queue row {index} IID")
        component = _plain_string(
            row.get("component_id"),
            field=f"queue row {index} component_id",
        )
        if iid in seen_iids or component in seen_components:
            raise R10BBerniniPilotError(
                "queue is not IID/component-disjoint"
            )
        seen_iids.add(iid)
        seen_components.add(component)
        if row.get("authorization") != _AUTHORIZATION:
            raise R10BBerniniPilotError(
                f"queue row {index} authorization differs"
            )
        if _validate_candidate_expansion_row(row, index=index):
            expansion_rows.append(row)
    expansion_summary = summary.get("candidate_expansion")
    if expansion_summary is None:
        if expansion_rows:
            raise R10BBerniniPilotError(
                "queue candidate expansion summary is missing"
            )
    else:
        if not isinstance(expansion_summary, Mapping):
            raise R10BBerniniPilotError(
                "queue candidate expansion summary differs"
            )
        tier = expansion_summary.get("tier")
        if tier not in {None, BOUNDED_ACTION_NEAR_MISS_TIER}:
            raise R10BBerniniPilotError(
                "queue candidate expansion tier differs"
            )
        expected_policy = (
            _BOUNDED_ACTION_NEAR_MISS_POLICY if tier is not None else None
        )
        expected_digest = (
            BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256
            if tier is not None
            else None
        )
        selected_counts = dict(
            sorted(
                Counter(
                    row["screen_cell"] for row in expansion_rows
                ).items()
            )
        )
        if (
            expansion_summary.get("policy") != expected_policy
            or expansion_summary.get("policy_sha256") != expected_digest
            or expansion_summary.get("selected_rows") != len(expansion_rows)
            or expansion_summary.get("selected_cell_counts")
            != selected_counts
            or expansion_summary.get("audit_only") is not True
            or expansion_summary.get("final_pilot_eligible") is not False
            or (expansion_rows and tier is None)
        ):
            raise R10BBerniniPilotError(
                "queue candidate expansion summary contract differs"
            )
    return rows, summary, expected_files


def _validate_audit_record(
    record: Mapping[str, Any],
    *,
    queue_row: Mapping[str, Any],
    model_id: str,
    prompt_sha256: str,
) -> None:
    iid = str(queue_row["iid"])
    if set(record) != _AUDIT_FIELDS:
        missing = sorted(_AUDIT_FIELDS - set(record))
        extra = sorted(set(record) - _AUDIT_FIELDS)
        raise R10BBerniniPilotError(
            f"iid={iid} audit fields differ; missing={missing}, extra={extra}"
        )
    if (
        record.get("schema_version") != AUDIT_ROW_SCHEMA
        or record.get("iid") != iid
        or record.get("queue_row_sha256") != object_digest(queue_row)
        or record.get("qwen_model_id") != model_id
        or record.get("qwen_prompt_sha256") != prompt_sha256
    ):
        raise R10BBerniniPilotError(f"iid={iid} audit binding differs")
    _plain_string(record.get("source_state"), field=f"iid={iid} source_state")
    _plain_string(record.get("target_state"), field=f"iid={iid} target_state")
    for field, values in _AUDIT_ENUMS.items():
        if record.get(field) not in values:
            raise R10BBerniniPilotError(
                f"iid={iid} audit enum {field} differs"
            )
    if record["intended_atomic"] != queue_row["intended_family"]:
        raise R10BBerniniPilotError(
            f"iid={iid} Qwen intended action differs from bound instruction"
        )


def _audit_role(
    audit: Mapping[str, Any],
    *,
    intended_family: str,
) -> tuple[str, str | None]:
    if any(
        audit[field] == "high"
        for field in (
            "identity_appearance_change",
            "nonphysical_effect",
            "deformation",
            "flicker",
        )
    ):
        return "effect", None
    if audit["camera_motion"] == "high":
        return "camera", None
    observed = str(audit["observed_atomic_or_none"])
    if (
        observed in {"wave", "quadruped_lie_down", "other"}
        and observed != intended_family
        and audit["actor_motion"] in {"clear", "weak"}
    ):
        return "wrong", intended_family
    if observed == "none" and audit["actor_motion"] == "none":
        return "static", None
    positive_common = (
        observed == intended_family
        and audit["success"] == "yes"
        and audit["actor_motion"] == "clear"
        and audit["onset"] == "clear"
        and audit["confidence"] in {"medium", "high"}
        and audit["camera_motion"] in {"none", "low"}
        and audit["identity_appearance_change"] in {"none", "low"}
        and audit["nonphysical_effect"] in {"none", "low"}
        and audit["deformation"] in {"none", "low"}
        and audit["flicker"] in {"none", "low"}
        and audit["reflection_or_sunglasses_artifact"] == "none"
        and audit["secondary_action"] == "none"
    )
    if not positive_common:
        return "reject", None
    if intended_family == "wave":
        if (
            audit["direction"] != "toward_viewer"
            or audit["periodicity"] != "repeated"
            or audit["subject_morphology"]
            not in {
                "adult_human",
                "child_human",
                "character_or_nonhuman",
            }
        ):
            return "reject", None
    else:
        if audit["subject_morphology"] not in {
            "dog",
            "bulldog",
            "cat",
            "other_quadruped",
        }:
            return "reject", None
    return "positive", intended_family


def _proxy_gate_for_role(queue_row: Mapping[str, Any], role: str) -> bool:
    gate_name = "action" if role in {"positive", "wrong"} else role
    gates = queue_row.get("feature_gates")
    return (
        isinstance(gates, Mapping)
        and isinstance(gates.get(gate_name), Mapping)
        and gates[gate_name].get("pass") is True
    )


def _final_rank(
    queue_row: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    role: str,
) -> tuple[Any, ...]:
    confidence_rank = {"high": 0, "medium": 1, "low": 2}[str(audit["confidence"])]
    key = object_digest(
        {
            "schema_version": FINAL_SUMMARY_SCHEMA,
            "seed": int(queue_row["selection_seed"]),
            "role": role,
            "iid": queue_row["iid"],
            "audit_sha256": object_digest(audit),
        }
    )
    return (
        confidence_rank,
        int(queue_row["screen_rank"]),
        key,
        str(queue_row["iid"]),
    )


def _select_cell(
    *,
    cell: str,
    candidates: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    count: int,
    used_components: set[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    role = cell.split(":", 2)[1] if cell.startswith("control:") else "positive"
    ranked = sorted(
        candidates,
        key=lambda pair: _final_rank(pair[0], pair[1], role=role),
    )
    selected = []
    for queue_row, audit in ranked:
        component = str(queue_row["component_id"])
        if component in used_components:
            continue
        selected.append((queue_row, audit))
        used_components.add(component)
        if len(selected) == count:
            break
    return selected


def finalize_controlled_pilot(
    *,
    queue_dir: str | Path,
    audit_records: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Finalize at most 20 audited visual rows; quota gaps remain explicit."""

    queue_rows, queue_summary, queue_files = _load_queue_commit(queue_dir)
    audit_path = _regular_file(audit_records, field="audit_records")
    audits, audit_raw = _load_jsonl(audit_path, field="audit_records")
    audit_by_iid: dict[str, dict[str, Any]] = {}
    for audit in audits:
        iid = str(audit.get("iid", ""))
        if not iid or iid in audit_by_iid:
            raise R10BBerniniPilotError(
                "audit records have missing/duplicate IID"
            )
        audit_by_iid[iid] = audit
    queue_by_iid = {str(row["iid"]): row for row in queue_rows}
    if set(audit_by_iid) != set(queue_by_iid):
        missing = sorted(set(queue_by_iid) - set(audit_by_iid))
        extra = sorted(set(audit_by_iid) - set(queue_by_iid))
        raise R10BBerniniPilotError(
            f"audit/queue IID coverage differs; missing={missing}, extra={extra}"
        )
    qwen = queue_summary["qwen_audit"]
    model_id = str(qwen["qwen_model_id"])
    prompt_sha = str(qwen["qwen_prompt_sha256"])

    pools: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    rejection_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()
    expansion_classification_counts: Counter[str] = Counter()
    expansion_rejection_count = 0
    for iid, queue_row in queue_by_iid.items():
        audit = audit_by_iid[iid]
        _validate_audit_record(
            audit,
            queue_row=queue_row,
            model_id=model_id,
            prompt_sha256=prompt_sha,
        )
        role, family = _audit_role(
            audit,
            intended_family=str(queue_row["intended_family"]),
        )
        classification_counts[role] += 1
        if (
            queue_row.get("candidate_expansion_tier")
            == BOUNDED_ACTION_NEAR_MISS_TIER
        ):
            expansion_classification_counts[role] += 1
            expansion_rejection_count += 1
            rejection_counts["candidate_expansion_audit_only"] += 1
            continue
        if role == "reject":
            rejection_counts["semantic_reject"] += 1
            continue
        if not _proxy_gate_for_role(queue_row, role):
            rejection_counts[f"{role}_proxy_gate_failed"] += 1
            continue
        if role == "positive":
            morphology = str(audit["subject_morphology"])
            if family == "wave":
                pools[f"positive:wave:{morphology}"].append(
                    (queue_row, audit)
                )
                pools[
                    "positive:wave:additional_direct_nonreflection"
                ].append((queue_row, audit))
            elif morphology in {"dog", "bulldog"}:
                pools[
                    "positive:quadruped_lie_down:dog_or_bulldog"
                ].append((queue_row, audit))
            else:
                pools[
                    f"positive:quadruped_lie_down:{morphology}"
                ].append((queue_row, audit))
        elif role == "wrong":
            # Wrong-instruction evidence is evaluated as a counterfactual on
            # each positive visual pair; it is not a unique-video quota.
            rejection_counts["wrong_is_prompt_counterfactual_only"] += 1
            continue
        elif role == "static":
            pools["control:static:global"].append((queue_row, audit))
        else:
            pools[f"control:{role}:global"].append((queue_row, audit))

    selected: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    used_components: set[str] = set()
    shortfalls: dict[str, Any] = {}
    for cell, target in FINAL_QUOTAS.items():
        chosen = _select_cell(
            cell=cell,
            candidates=pools.get(cell, ()),
            count=target,
            used_components=used_components,
        )
        selected.extend((cell, row, audit) for row, audit in chosen)
        if len(chosen) < target:
            shortfalls[cell] = {
                "required": int(target),
                "selected": len(chosen),
                "eligible_before_component_dedup": len(pools.get(cell, ())),
            }
    if len(selected) > MAX_FINAL_ROWS:  # pragma: no cover
        raise RuntimeError("internal pilot cap violation")

    final_rows: list[dict[str, Any]] = []
    cell_ranks: Counter[str] = Counter()
    for global_rank, (cell, queue_row, audit) in enumerate(selected, 1):
        cell_ranks[cell] += 1
        family = str(queue_row["intended_family"])
        media = queue_row["media_binding"]
        final_row = {
            "schema_version": SMOKE_ROW_SCHEMA,
            "iid": queue_row["iid"],
            "family": family,
            "primary_family": queue_row["upstream_label"][
                "primary_family"
            ],
            "prompt": queue_row["canonical_prompt"],
            "canonical_prompt": queue_row["canonical_prompt"],
            "original_prompt": queue_row["original_prompt"],
            "noop_prompt": queue_row["prompt_variants"]["noop"],
            "cross_family_shuffle_prompt": queue_row[
                "prompt_variants"
            ]["cross_family_shuffle"],
            "cross_family_shuffle_family": queue_row[
                "prompt_variants"
            ]["cross_family_shuffle_family"],
            "component_id": queue_row["component_id"],
            "source_split": queue_row["source_split"],
            "fresh": True,
            "candidate_row_index": queue_row["candidate_row_index"],
            "track_input_index": queue_row["track_binding"][
                "track_input_index"
            ],
            "track_cache_index": queue_row["track_binding"][
                "track_cache_index"
            ],
            "track_delta_p90": queue_row["feature_metrics"][
                "edit_delta_p90"
            ],
            "data_root": media["data_root"],
            "src_video": media["src_video"]["relative_path"],
            "tgt_video": media["tgt_video"]["relative_path"],
            "src_video_sha256": media["src_video"]["sha256"],
            "tgt_video_sha256": media["tgt_video"]["sha256"],
            "candidate_input_digest": queue_row[
                "candidate_input_digest"
            ],
            "candidate_row_sha256": queue_row[
                "candidate_row_sha256"
            ],
            "source_bindings": queue_row["source_bindings"],
            "track_binding": queue_row["track_binding"],
            "feature_metrics": queue_row["feature_metrics"],
            "feature_gates": queue_row["feature_gates"],
            "motion_gate_applicable": cell.startswith("positive:") or (
                cell.startswith("control:wrong:")
            ),
            "motion_gate_pass": (
                bool(queue_row["feature_gates"]["action"]["pass"])
                if (
                    cell.startswith("positive:")
                    or cell.startswith("control:wrong:")
                )
                else False
            ),
            "pilot_role": (
                "positive"
                if cell.startswith("positive:")
                else cell.split(":", 2)[1]
            ),
            "quota_cell": cell,
            "within_quota_rank": int(cell_ranks[cell]),
            "pilot_rank": int(global_rank),
            "selection_seed": int(queue_row["selection_seed"]),
            "qwen_audit_binding": {
                "qwen_model_id": model_id,
                "qwen_prompt_sha256": prompt_sha,
                "queue_row_sha256": object_digest(queue_row),
                "audit_record_sha256": object_digest(audit),
                "semantic_precedence": [
                    "effect",
                    "camera",
                    "wrong",
                    "static",
                    "positive_or_reject",
                ],
                "audit": audit,
            },
            "component_source": "r7_indexed_visual_component",
            "label_provenance": "qwen_controlled_pilot_audit",
            "human_label": False,
            "formal_evidence": False,
            "representation_promoted": False,
            "renderer_probe_authorized": False,
            "generation_authorized": False,
            "training_authorized": False,
            "authorization": dict(_AUTHORIZATION),
        }
        final_rows.append(final_row)
    try:
        validate_smoke_rows(final_rows)
    except R10BTangentError as error:
        if final_rows:
            raise R10BBerniniPilotError(str(error)) from error

    balanced_ready = (
        not shortfalls
        and len(final_rows) == MAX_FINAL_ROWS
        and len(used_components) == MAX_FINAL_ROWS
    )
    manifest_bytes = _jsonl_bytes(final_rows)
    shortfall_payload = {
        "schema_version": SHORTFALL_SCHEMA,
        "balanced_pilot_ready": balanced_ready,
        "shortfalls": shortfalls,
        "no_control_rows_fabricated": True,
        "row_reuse_allowed": False,
        "component_reuse_allowed": False,
    }
    shortfall_bytes = _pretty_bytes(shortfall_payload)
    summary = {
        "schema_version": FINAL_SUMMARY_SCHEMA,
        "experiment_role": (
            "balanced_controlled_engineering_pilot"
            if balanced_ready
            else "engineering_unbalanced_evidence_only"
        ),
        "balanced_pilot_ready": balanced_ready,
        "rows": len(final_rows),
        "maximum_rows": MAX_FINAL_ROWS,
        "unique_iids": len({row["iid"] for row in final_rows}),
        "unique_components": len(used_components),
        "component_disjoint": len(used_components) == len(final_rows),
        "quota_targets": dict(FINAL_QUOTAS),
        "quota_selected": dict(
            sorted(Counter(row["quota_cell"] for row in final_rows).items())
        ),
        "shortfalls": shortfalls,
        "classification_counts": dict(sorted(classification_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "candidate_expansion_audit": {
            "tier": queue_summary.get("candidate_expansion", {}).get(
                "tier"
            ),
            "rows": expansion_rejection_count,
            "classification_counts": dict(
                sorted(expansion_classification_counts.items())
            ),
            "rejection_count": expansion_rejection_count,
            "admitted_to_final_quota": 0,
            "audit_only": True,
            "final_pilot_eligible": False,
        },
        "selection_seed": int(queue_summary["selection_seed"]),
        "qwen_audit": {
            "schema_version": AUDIT_ROW_SCHEMA,
            "qwen_model_id": model_id,
            "qwen_prompt_sha256": prompt_sha,
            "audit_records_sha256": _sha256_bytes(audit_raw),
            "audit_records_rows": len(audits),
        },
        "inputs": {
            "queue_dir": str(Path(queue_dir).expanduser().resolve(strict=True)),
            "queue_files": queue_files,
            "queue_inputs": queue_summary["inputs"],
            "audit_records": str(audit_path),
            "audit_records_sha256": _sha256_bytes(audit_raw),
        },
        "outputs": {
            FINAL_MANIFEST_NAME: {
                "rows": len(final_rows),
                "sha256": _sha256_bytes(manifest_bytes),
            },
            FINAL_SHORTFALL_NAME: {
                "sha256": _sha256_bytes(shortfall_bytes),
            },
        },
        "video_bytes_copied": False,
        "controls_fabricated": False,
        "human_labels": False,
        "authorization": dict(_AUTHORIZATION),
    }
    summary_bytes = _pretty_bytes(summary)
    done = {
        "schema_version": FINAL_DONE_SCHEMA,
        "rows": len(final_rows),
        "balanced_pilot_ready": balanced_ready,
        "files": {
            FINAL_MANIFEST_NAME: _sha256_bytes(manifest_bytes),
            FINAL_SHORTFALL_NAME: _sha256_bytes(shortfall_bytes),
            FINAL_SUMMARY_NAME: _sha256_bytes(summary_bytes),
        },
        "authorization": dict(_AUTHORIZATION),
    }
    _atomic_directory(
        output_dir,
        {
            FINAL_MANIFEST_NAME: manifest_bytes,
            FINAL_SHORTFALL_NAME: shortfall_bytes,
            FINAL_SUMMARY_NAME: summary_bytes,
            FINAL_DONE_NAME: _pretty_bytes(done),
        },
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/finalize the R10B Bernini controlled pilot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    queue = subparsers.add_parser("queue")
    queue.add_argument("--candidate-manifest", type=Path, required=True)
    queue.add_argument("--track-manifest", type=Path, required=True)
    queue.add_argument("--track-cache", type=Path, required=True)
    queue.add_argument("--qwen-model-id", required=True)
    queue.add_argument("--qwen-prompt-sha256", required=True)
    queue.add_argument("--audit-oversample", type=int, default=4)
    queue.add_argument("--seed", type=int, default=SELECTION_SEED)
    queue.add_argument(
        "--candidate-expansion-tier",
        choices=("none", BOUNDED_ACTION_NEAR_MISS_TIER),
        default="none",
    )
    queue.add_argument("--output-dir", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--queue-dir", type=Path, required=True)
    finalize.add_argument("--audit-records", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "queue":
        payload = derive_qwen_audit_queue(
            candidate_manifest=args.candidate_manifest,
            track_manifest=args.track_manifest,
            track_cache=args.track_cache,
            qwen_model_id=args.qwen_model_id,
            qwen_prompt_sha256=args.qwen_prompt_sha256,
            audit_oversample=args.audit_oversample,
            seed=args.seed,
            candidate_expansion_tier=(
                None
                if args.candidate_expansion_tier == "none"
                else args.candidate_expansion_tier
            ),
        )
        write_qwen_audit_queue(payload, args.output_dir)
        result = {
            "output_dir": str(args.output_dir.resolve()),
            **payload["summary"],
        }
    else:
        result = finalize_controlled_pilot(
            queue_dir=args.queue_dir,
            audit_records=args.audit_records,
            output_dir=args.output_dir,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_THRESHOLDS",
    "AUDIT_ROW_SCHEMA",
    "BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256",
    "BOUNDED_ACTION_NEAR_MISS_THRESHOLDS",
    "BOUNDED_ACTION_NEAR_MISS_TIER",
    "CAMERA_THRESHOLDS",
    "CANONICAL_PROMPTS",
    "EFFECT_THRESHOLDS",
    "FINAL_QUOTAS",
    "MAX_FINAL_ROWS",
    "QUEUE_ROW_SCHEMA",
    "R10BBerniniPilotError",
    "SELECTION_SEED",
    "STATIC_THRESHOLDS",
    "derive_qwen_audit_queue",
    "finalize_controlled_pilot",
    "write_qwen_audit_queue",
]
