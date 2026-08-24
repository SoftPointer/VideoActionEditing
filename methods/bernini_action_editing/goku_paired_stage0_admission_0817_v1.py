#!/usr/bin/env python3
"""Fail-closed admission of GOKU paired stage-0 diagnostics.

This module is intentionally *not* a training-manifest builder.  It verifies a
complete ``motive.audit`` output against its pinned input, re-hashes the
physical source and target bytes, collapses aliases/seed variants into
effective source+instruction units, and emits only:

* a diagnostic candidate manifest; and
* a pending human-review queue.

No output of this program is qualified supervision.  In particular,
``qualification_status`` is always ``unqualified`` and
``training_authorized`` is always ``false``.  A later, separately reviewed
qualification authority must consume the review results and the frozen media
bytes before any D0 sampler can be constructed.

The implementation duplicates the small, frozen selection equation used by
the 2026-08-17 GOKU target-side stage-0 run.  This is necessary because the
current upstream ``motive.audit`` summary does not serialize all numeric
selection arguments.  The compiled policy and its digest are written into
every receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any

import numpy as np


SCHEMA_VERSION = "bernini-goku-paired-stage0-admission-0817-v1"
CANDIDATE_SCHEMA_VERSION = "bernini-goku-paired-stage0-candidate-0817-v1"
REVIEW_SCHEMA_VERSION = "bernini-goku-paired-stage0-review-task-0817-v1"
RECEIPT_SCHEMA_VERSION = "bernini-goku-paired-stage0-admission-receipt-0817-v1"
DONE_SCHEMA_VERSION = "bernini-goku-paired-stage0-admission-done-0817-v1"
EQUIVALENCE_SCHEMA_VERSION = "bernini-goku-paired-equivalence-authority-0817-v1"
FEATURE_SCHEMA_VERSION = "videoedit-motive-feature-v1"

ROLE = "diagnostic_candidate_only"
QUALIFICATION_STATUS = "unqualified"
TRAINING_AUTHORIZED = False
FORMAL_D0_COUNT_CONTRIBUTION = 0
REQUIRED_HUMAN_REVIEWS = 2

PARTITION_POLICY_VERSION = "actor-scene-connected-components-sha256-10000-v1"
PARTITIONS = (
    ("train_candidate", 0, 8000),
    ("calibration_candidate", 8000, 8800),
    ("promotion_candidate", 8800, 9400),
    ("locked_final_candidate", 9400, 10000),
)
FUTURE_D0_QUOTAS = {
    "general": 300,
    "strict_action": 100,
    "noop": 100,
}

EXPECTED_MOTION_CONFIG = {
    "analysis_frames": 32,
    "resize_width": 256,
    "farneback_pyr_scale": 0.5,
    "farneback_levels": 4,
    "farneback_winsize": 21,
    "farneback_iterations": 3,
    "farneback_poly_n": 7,
    "farneback_poly_sigma": 1.5,
    "active_speed_threshold": 0.005,
    "static_residual_p90": 0.003,
    "static_active_fraction": 0.025,
    "camera_raw_speed": 0.003,
    "camera_explained_ratio": 0.70,
    "camera_residual_multiplier": 1.75,
    "max_scene_cut_ratio": 0.15,
    "max_scene_cuts": 0,
    "scene_cut_luma_delta": 0.28,
    "min_frames": 3,
    "eps": 1e-8,
}
EXPECTED_DESCRIPTOR_CONFIG = {
    "temporal_bins": 4,
    "grid_rows": 2,
    "grid_cols": 2,
    "orientation_bins": 8,
    "active_speed_threshold": 0.005,
    "minimum_active_fraction": 0.001,
    "magnitude_clip_percentile": 95.0,
    "eps": 1e-8,
}

TARGET_KEYS = (
    "tgt_video",
    "target_video",
    "edited_video",
    "video",
    "video_path",
    "path",
)
SOURCE_KEYS = ("src_video", "source_video", "original_video")
INSTRUCTION_KEYS = ("instruction_en", "prompt", "instruction", "edit_prompt")
ID_KEYS = ("iid", "id", "sample_id", "uid", "name")
ACTION_FAMILY_KEYS = (
    "action_family",
    "family",
    "pilot_action_signature",
    "action_signature",
    "target_action_family",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class GokuStage0AdmissionError(RuntimeError):
    """Raised for any closure or trust-boundary violation."""


@dataclass(frozen=True)
class FrozenSelectionPolicy:
    """Exact policy used by the 0817 paired target-side diagnostic."""

    min_descriptor_delta: float = 0.35
    # The executed 0817 Stage-0 producer passed no semantic-class allowlist.
    # Upstream records that exact policy as JSON null and applies the motion
    # equations to every classified instruction.  Admission must replay those
    # bytes faithfully; semantic eligibility is a later human-review gate.
    semantic_classes: tuple[str, ...] | None = None
    min_action_residual_p90: float = 0.005
    min_action_motion_ratio: float = 0.0
    min_action_motion_gain: float = 0.0
    min_suppression_residual_p90: float = 0.003
    min_suppression_motion_ratio: float = 1.10

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": "goku-paired-stage0-node279-0817-v1",
            "min_descriptor_delta": self.min_descriptor_delta,
            "semantic_classes": (
                None
                if self.semantic_classes is None
                else list(self.semantic_classes)
            ),
            "min_action_residual_p90": self.min_action_residual_p90,
            "min_action_motion_ratio": self.min_action_motion_ratio,
            "min_action_motion_gain": self.min_action_motion_gain,
            "min_suppression_residual_p90": (
                self.min_suppression_residual_p90
            ),
            "min_suppression_motion_ratio": (
                self.min_suppression_motion_ratio
            ),
        }


SELECTION_POLICY = FrozenSelectionPolicy()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise GokuStage0AdmissionError(
            f"value is not canonical-JSON encodable: {error}"
        ) from error
    return text.encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GokuStage0AdmissionError(f"{field} must be lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GokuStage0AdmissionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GokuStage0AdmissionError(f"{label} is not UTF-8") from error
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except GokuStage0AdmissionError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise GokuStage0AdmissionError(f"invalid JSON in {label}: {error}") from error


def _require_regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise GokuStage0AdmissionError(f"missing {label}: {path}") from error
    if stat.S_ISLNK(before.st_mode):
        raise GokuStage0AdmissionError(f"{label} may not be a symlink: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise GokuStage0AdmissionError(f"{label} is not a regular file: {path}")
    if before.st_size <= 0:
        raise GokuStage0AdmissionError(f"{label} is empty: {path}")
    return before


def _hash_physical_file(path: Path, *, label: str) -> dict[str, Any]:
    """Hash an opened regular file and reject concurrent byte replacement."""

    _require_regular_file(path, label=label)
    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise GokuStage0AdmissionError(f"cannot open {label}: {resolved}") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise GokuStage0AdmissionError(
                f"{label} stopped being a non-empty regular file: {resolved}"
            )
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise GokuStage0AdmissionError(f"{label} changed while hashing: {resolved}")
    return {
        "resolved_path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": int(after.st_size),
    }


def _verify_pinned_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    expected = _valid_sha256(expected_sha256, field=f"expected {label} sha256")
    artifact = _hash_physical_file(path, label=label)
    if artifact["sha256"] != expected:
        raise GokuStage0AdmissionError(
            f"{label} SHA-256 mismatch: expected {expected}, "
            f"got {artifact['sha256']}"
        )
    return {
        "name": path.name,
        "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"],
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = _decode_json(path.read_bytes(), label=label)
    if not isinstance(value, dict):
        raise GokuStage0AdmissionError(f"{label} must be a JSON object")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise GokuStage0AdmissionError(f"{label} must end with one newline")
    lines = raw.splitlines()
    if not lines:
        raise GokuStage0AdmissionError(f"{label} has no rows")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise GokuStage0AdmissionError(
                f"{label}:{line_number} is blank; physical line closure failed"
            )
        value = _decode_json(line, label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise GokuStage0AdmissionError(
                f"{label}:{line_number} must be a JSON object"
            )
        rows.append(value)
    return rows


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GokuStage0AdmissionError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _require_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GokuStage0AdmissionError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise GokuStage0AdmissionError(f"{field} must be finite")
    return number


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GokuStage0AdmissionError(f"{field} must be a non-empty string")
    return value


def _row_id(row: Mapping[str, Any], index: int) -> str:
    for key in ID_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return str(index)


def _unique_declared_value(
    row: Mapping[str, Any],
    keys: Iterable[str],
    *,
    field: str,
) -> str:
    values = [str(row[key]) for key in keys if row.get(key) not in (None, "")]
    unique = list(dict.fromkeys(values))
    if not unique:
        raise GokuStage0AdmissionError(f"row has no {field}")
    if len(unique) != 1:
        raise GokuStage0AdmissionError(
            f"row has ambiguous {field} declarations: {unique}"
        )
    return unique[0]


def _resolve_manifest_path(value: str, root: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def _resolve_existing(path: Path, *, label: str) -> Path:
    try:
        return path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise GokuStage0AdmissionError(f"cannot resolve {label}: {path}") from error


def _row_without_current_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "motive_audit"}


def _selection_from_bound_audit(
    audit: Mapping[str, Any],
    *,
    policy: FrozenSelectionPolicy,
    row_id: str,
) -> bool:
    if audit.get("status") != "ok":
        raise GokuStage0AdmissionError(f"{row_id}: audit status is not ok")
    if audit.get("paired") is not True:
        raise GokuStage0AdmissionError(f"{row_id}: stage0 row is not paired")
    source_label = _require_nonempty_string(
        audit.get("source_label"), field=f"{row_id}.source_label"
    )
    target_label = _require_nonempty_string(
        audit.get("target_label"), field=f"{row_id}.target_label"
    )
    semantics = audit.get("instruction_semantics")
    if not isinstance(semantics, dict):
        raise GokuStage0AdmissionError(
            f"{row_id}.instruction_semantics must be an object"
        )
    semantic_label = _require_nonempty_string(
        semantics.get("label"), field=f"{row_id}.instruction_semantics.label"
    )
    source_metrics = audit.get("source_metrics")
    target_metrics = audit.get("target_metrics")
    if not isinstance(source_metrics, dict) or not isinstance(target_metrics, dict):
        raise GokuStage0AdmissionError(f"{row_id}: motion metrics are missing")
    source_speed = _require_number(
        source_metrics.get("residual_speed_p90"),
        field=f"{row_id}.source_metrics.residual_speed_p90",
    )
    target_speed = _require_number(
        target_metrics.get("residual_speed_p90"),
        field=f"{row_id}.target_metrics.residual_speed_p90",
    )
    delta_norm = _require_number(
        audit.get("descriptor_delta_norm"),
        field=f"{row_id}.descriptor_delta_norm",
    )
    invalid = {"cut_or_decode_artifact", "camera_only"}
    selected = (
        source_label not in invalid
        and target_label not in invalid
        and "dynamic_object" in {source_label, target_label}
        and delta_norm >= policy.min_descriptor_delta
        and (
            policy.semantic_classes is None
            or semantic_label in policy.semantic_classes
        )
    )
    if semantic_label == "continuous_action":
        selected = selected and target_speed >= policy.min_action_residual_p90
        if (
            policy.min_action_motion_ratio > 0.0
            or policy.min_action_motion_gain > 0.0
        ):
            selected = selected and (
                target_speed >= source_speed * policy.min_action_motion_ratio
                or target_speed - source_speed >= policy.min_action_motion_gain
            )
    elif semantic_label == "motion_suppression":
        selected = (
            selected
            and source_speed >= policy.min_suppression_residual_p90
            and source_speed
            >= target_speed * policy.min_suppression_motion_ratio
        )
    return bool(selected)


def _feature_metadata_digest(metadata: Mapping[str, Any]) -> str:
    compatibility = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_kind": metadata.get("feature_kind"),
        "dimension": metadata.get("dimension"),
        "provenance": metadata.get("provenance"),
    }
    return _object_sha256(compatibility)


def _load_feature_archive(
    path: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"features", "ids", "metadata_json"}
            if set(archive.files) != required:
                raise GokuStage0AdmissionError(
                    "descriptors.npz keys must be exactly "
                    f"{sorted(required)}, got {sorted(archive.files)}"
                )
            features_raw = np.asarray(archive["features"])
            ids_raw = np.asarray(archive["ids"])
            metadata_raw = np.asarray(archive["metadata_json"])
    except GokuStage0AdmissionError:
        raise
    except Exception as error:
        raise GokuStage0AdmissionError(
            f"cannot load descriptors.npz safely: {error}"
        ) from error
    if features_raw.dtype != np.dtype("float32"):
        raise GokuStage0AdmissionError("descriptor features must be float32")
    if features_raw.ndim != 2 or features_raw.shape[1] <= 0:
        raise GokuStage0AdmissionError("descriptor features must have shape [N,D]")
    if not np.isfinite(features_raw).all():
        raise GokuStage0AdmissionError("descriptor features contain NaN/Inf")
    if ids_raw.ndim != 1 or ids_raw.dtype.kind not in {"U", "S"}:
        raise GokuStage0AdmissionError("descriptor ids must be a 1-D string array")
    if len(ids_raw) != len(features_raw):
        raise GokuStage0AdmissionError("descriptor feature/id length mismatch")
    if metadata_raw.ndim != 0 or metadata_raw.dtype.kind not in {"U", "S"}:
        raise GokuStage0AdmissionError(
            "descriptor metadata_json must be a scalar string"
        )
    metadata_value = _decode_json(
        str(metadata_raw.item()).encode("utf-8"), label="descriptor metadata_json"
    )
    if not isinstance(metadata_value, dict):
        raise GokuStage0AdmissionError("descriptor metadata must be an object")
    metadata = dict(metadata_value)
    if metadata.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise GokuStage0AdmissionError("unexpected descriptor schema_version")
    if metadata.get("feature_kind") != "geometry_action_delta":
        raise GokuStage0AdmissionError(
            "paired stage0 requires geometry_action_delta descriptors"
        )
    if metadata.get("dimension") != int(features_raw.shape[1]):
        raise GokuStage0AdmissionError("descriptor dimension metadata mismatch")
    expected_compatibility = _feature_metadata_digest(metadata)
    if metadata.get("compatibility_digest") != expected_compatibility:
        raise GokuStage0AdmissionError(
            "descriptor compatibility digest does not close"
        )
    return features_raw, [str(value) for value in ids_raw.tolist()], metadata


def _counter_dict(value: Mapping[str, int]) -> dict[str, int]:
    return {key: int(value[key]) for key in sorted(value)}


def _assert_summary_closure(
    summary: Mapping[str, Any],
    audit_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    feature_ids: Sequence[str],
    feature_metadata: Mapping[str, Any],
    original_count: int,
) -> None:
    total = _require_int(summary.get("total"), field="summary.total", minimum=1)
    successful = _require_int(
        summary.get("successful"), field="summary.successful"
    )
    selected = _require_int(summary.get("selected"), field="summary.selected")
    errors = _require_int(summary.get("errors"), field="summary.errors")
    if total != original_count or total != len(audit_rows):
        raise GokuStage0AdmissionError("summary/original/audit row counts disagree")
    if errors != 0 or successful != total:
        raise GokuStage0AdmissionError(
            "partial stage0 output is rejected: errors must be 0 and all rows successful"
        )
    if successful != len(feature_ids):
        raise GokuStage0AdmissionError("summary successful count != descriptor rows")
    if selected != len(selected_rows):
        raise GokuStage0AdmissionError("summary selected count != selected.jsonl rows")
    if summary.get("descriptor_semantics") != (
        "paired target-minus-source geometry action delta"
    ):
        raise GokuStage0AdmissionError("summary descriptor semantics are not paired")
    if summary.get("archive_compatibility_digest") != feature_metadata.get(
        "compatibility_digest"
    ):
        raise GokuStage0AdmissionError(
            "summary/archive compatibility digest mismatch"
        )
    classes = summary.get("selection_semantic_classes")
    expected_classes = SELECTION_POLICY.semantic_classes
    classes_match = (
        classes is None
        if expected_classes is None
        else (
            isinstance(classes, list)
            and len(classes) == len(expected_classes)
            and set(classes) == set(expected_classes)
        )
    )
    if not classes_match:
        raise GokuStage0AdmissionError(
            "summary semantic selection classes do not match frozen 0817 policy"
        )
    if summary.get("config") != EXPECTED_MOTION_CONFIG:
        raise GokuStage0AdmissionError(
            "summary motion config does not match frozen node279 stage0 config"
        )
    provenance = feature_metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise GokuStage0AdmissionError("descriptor provenance must be an object")
    expected_provenance = {
        "descriptor_version": "camera_compensated_hoof_v2",
        "descriptor_config": EXPECTED_DESCRIPTOR_CONFIG,
        "motion_backend": "opencv_farneback_partial_affine_v1",
        "motion_config": EXPECTED_MOTION_CONFIG,
        "speed_units": "frame_width_per_second",
    }
    if provenance != expected_provenance:
        raise GokuStage0AdmissionError(
            "descriptor provenance does not match frozen node279 stage0 config"
        )


def _declared_sha_values(row: Mapping[str, Any], *, endpoint: str) -> list[str]:
    keys = (
        ("source_video_sha256", "source_sha256")
        if endpoint == "source"
        else ("target_video_sha256", "target_sha256", "edited_video_sha256")
    )
    values: list[str] = []
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            values.append(_valid_sha256(value, field=key))
    return list(dict.fromkeys(values))


def _first_nonempty(row: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _motion_stratum(semantic_label: str, row: Mapping[str, Any]) -> str:
    explicit = _first_nonempty(row, ("motion_stratum", "training_subset"))
    if semantic_label in {"noop", "no_op", "preservation"} or explicit in {
        "noop",
        "noop_preservation",
    }:
        return "noop"
    if semantic_label == "motion_suppression":
        return "suppression"
    if semantic_label == "continuous_action":
        return "dynamic"
    return "other"


def _action_family(row: Mapping[str, Any]) -> str:
    value = _first_nonempty(row, ACTION_FAMILY_KEYS)
    return "unresolved" if value is None else value.strip().lower()


def _load_equivalence_authority(
    path: Path | None,
    expected_sha256: str | None,
    *,
    input_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if (path is None) != (expected_sha256 is None):
        raise GokuStage0AdmissionError(
            "equivalence authority path and expected SHA-256 must be supplied together"
        )
    if path is None:
        return {}, None
    artifact = _verify_pinned_file(
        path,
        expected_sha256=str(expected_sha256),
        label="equivalence authority",
    )
    rows = _load_jsonl(path, label="equivalence authority")
    by_id: dict[str, dict[str, Any]] = {}
    required = {
        "schema_version",
        "iid",
        "source_sha256",
        "target_sha256",
        "canonical_source_id",
        "canonical_target_id",
        "instruction_equivalence_id",
        "upstream_group_id",
        "actor_scene_group_id",
    }
    for index, row in enumerate(rows):
        if set(row) != required:
            raise GokuStage0AdmissionError(
                f"equivalence authority row {index} fields must be exactly {sorted(required)}"
            )
        if row.get("schema_version") != EQUIVALENCE_SCHEMA_VERSION:
            raise GokuStage0AdmissionError(
                f"equivalence authority row {index} schema mismatch"
            )
        iid = _require_nonempty_string(row.get("iid"), field="authority.iid")
        if iid in by_id:
            raise GokuStage0AdmissionError(f"duplicate authority iid: {iid}")
        for field in ("source_sha256", "target_sha256"):
            _valid_sha256(row.get(field), field=f"authority.{field}")
        for field in (
            "canonical_source_id",
            "canonical_target_id",
            "instruction_equivalence_id",
            "upstream_group_id",
            "actor_scene_group_id",
        ):
            _require_nonempty_string(row.get(field), field=f"authority.{field}")
        by_id[iid] = dict(row)
    if set(by_id) != input_ids:
        missing = sorted(input_ids - set(by_id))[:10]
        extra = sorted(set(by_id) - input_ids)[:10]
        raise GokuStage0AdmissionError(
            f"equivalence authority IID closure failed; missing={missing}, extra={extra}"
        )
    return by_id, artifact


def _partition_for_component(component_id: str) -> tuple[str, int]:
    digest = hashlib.sha256(
        (PARTITION_POLICY_VERSION + "\x00" + component_id).encode("utf-8")
    ).hexdigest()
    bucket = int(digest[:16], 16) % 10000
    for name, lower, upper in PARTITIONS:
        if lower <= bucket < upper:
            return name, bucket
    raise AssertionError(bucket)


def _assert_group_disjoint(candidates: Sequence[Mapping[str, Any]]) -> None:
    owners: dict[str, str] = {}
    for candidate in candidates:
        partition = str(candidate["candidate_partition"])
        closure = candidate["group_closure"]
        assert isinstance(closure, dict)
        tokens = closure["component_tokens"]
        assert isinstance(tokens, list)
        for token in tokens:
            prior = owners.setdefault(str(token), partition)
            if prior != partition:
                raise GokuStage0AdmissionError(
                    f"group token leaks across candidate partitions: {token}"
                )


def _stable_rank(namespace: str, effective_row_id: str) -> str:
    return hashlib.sha256(
        (namespace + "\x00" + effective_row_id).encode("utf-8")
    ).hexdigest()


def _assign_review_quotas(
    candidates: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_id = {str(row["effective_row_id"]): row for row in candidates}
    remaining = set(by_id)
    assigned: dict[str, str] = {}

    def reserve(cell: str, eligible: Iterable[str]) -> None:
        ordered = sorted(
            set(eligible) & remaining,
            key=lambda row_id: (_stable_rank(f"quota:{cell}:v1", row_id), row_id),
        )
        for row_id in ordered[: FUTURE_D0_QUOTAS[cell]]:
            assigned[row_id] = cell
            remaining.remove(row_id)

    reserve(
        "noop",
        (
            row_id
            for row_id, row in by_id.items()
            if row["motion_stratum"] == "noop"
        ),
    )
    reserve(
        "strict_action",
        (
            row_id
            for row_id, row in by_id.items()
            if row["motion_stratum"] == "dynamic"
            and row["action_family"] != "unresolved"
        ),
    )
    reserve(
        "general",
        (
            row_id
            for row_id, row in by_id.items()
            if row["motion_stratum"] != "noop"
        ),
    )
    order = {"general": 0, "strict_action": 1, "noop": 2, "backlog": 3}
    review_rows: list[dict[str, Any]] = []
    for row_id, candidate in by_id.items():
        cell = assigned.get(row_id, "backlog")
        task_identity = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "effective_row_id": row_id,
            "source_sha256": candidate["source"]["sha256"],
            "target_sha256": candidate["target"]["sha256"],
            "instruction_sha256": candidate["instruction"]["sha256"],
            "required_human_reviews": REQUIRED_HUMAN_REVIEWS,
        }
        review_rows.append(
            {
                **task_identity,
                "review_task_id": _object_sha256(task_identity),
                "role": ROLE,
                "quota_cell": cell,
                "candidate_partition": candidate["candidate_partition"],
                "source": candidate["source"],
                "target": candidate["target"],
                "instruction": candidate["instruction"],
                "action_family": candidate["action_family"],
                "motion_stratum": candidate["motion_stratum"],
                "review_status": "pending",
                "reviewer_receipts": [],
                "review_rubric": {
                    "action_execution": "pending",
                    "actor_object_ownership": "pending",
                    "identity_preservation": "pending",
                    "background_preservation": "pending",
                    "camera_preservation": "pending",
                    "temporal_quality": "pending",
                    "watermark_or_artifact": "pending",
                },
                "qualification_status": QUALIFICATION_STATUS,
                "training_authorized": TRAINING_AUTHORIZED,
            }
        )
    review_rows.sort(
        key=lambda row: (
            order[str(row["quota_cell"])],
            _stable_rank("review-queue-v1", str(row["effective_row_id"])),
            str(row["effective_row_id"]),
        )
    )
    counts = Counter(str(row["quota_cell"]) for row in review_rows)
    return review_rows, _counter_dict(counts)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    with path.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _output_artifact(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "name": path.name,
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
        "line_count": len(raw.splitlines()),
    }


def admit_stage0(
    *,
    stage0_dir: Path,
    original_selected: Path,
    output_dir: Path,
    expected_original_sha256: str,
    expected_summary_sha256: str,
    expected_audit_sha256: str,
    expected_selected_sha256: str,
    expected_descriptors_sha256: str,
    equivalence_authority: Path | None = None,
    expected_equivalence_authority_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and atomically materialize diagnostic-only candidate outputs."""

    stage0_dir = stage0_dir.expanduser()
    original_selected = original_selected.expanduser()
    output_dir = output_dir.expanduser()
    if output_dir.exists():
        raise GokuStage0AdmissionError(f"output directory already exists: {output_dir}")
    if not stage0_dir.is_dir() or stage0_dir.is_symlink():
        raise GokuStage0AdmissionError(
            f"stage0_dir must be a physical directory: {stage0_dir}"
        )
    paths = {
        "summary": stage0_dir / "summary.json",
        "audit": stage0_dir / "audit.jsonl",
        "selected": stage0_dir / "selected.jsonl",
        "descriptors": stage0_dir / "descriptors.npz",
        "original": original_selected,
    }
    expected = {
        "summary": expected_summary_sha256,
        "audit": expected_audit_sha256,
        "selected": expected_selected_sha256,
        "descriptors": expected_descriptors_sha256,
        "original": expected_original_sha256,
    }
    input_artifacts = {
        name: _verify_pinned_file(
            paths[name], expected_sha256=expected[name], label=name
        )
        for name in ("original", "summary", "audit", "selected", "descriptors")
    }

    original_rows = _load_jsonl(paths["original"], label="original selected.jsonl")
    summary = _load_json_object(paths["summary"], label="summary.json")
    audit_rows = _load_jsonl(paths["audit"], label="audit.jsonl")
    selected_rows = _load_jsonl(paths["selected"], label="selected.jsonl")
    features, feature_ids, feature_metadata = _load_feature_archive(
        paths["descriptors"]
    )
    _assert_summary_closure(
        summary,
        audit_rows,
        selected_rows,
        feature_ids,
        feature_metadata,
        len(original_rows),
    )

    original_ids = [_row_id(row, index) for index, row in enumerate(original_rows)]
    if len(set(original_ids)) != len(original_ids):
        raise GokuStage0AdmissionError("original selected.jsonl has duplicate IDs")
    authority_by_id, authority_artifact = _load_equivalence_authority(
        equivalence_authority,
        expected_equivalence_authority_sha256,
        input_ids=set(original_ids),
    )

    summary_root_raw = summary.get("root")
    summary_root = Path(
        _require_nonempty_string(summary_root_raw, field="summary.root")
    ).expanduser()
    selected_expected: list[Mapping[str, Any]] = []
    media_cache: dict[str, dict[str, Any]] = {}
    physical_rows: list[dict[str, Any]] = []
    ok_ids: list[str] = []
    label_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    declared_source_sha_count = 0
    declared_target_sha_count = 0

    for index, (original, audited) in enumerate(zip(original_rows, audit_rows)):
        iid = original_ids[index]
        audited_id = _row_id(audited, index)
        if audited_id != iid:
            raise GokuStage0AdmissionError(
                f"row {index}: audit ID {audited_id!r} != original ID {iid!r}"
            )
        if _row_without_current_audit(audited) != _row_without_current_audit(original):
            raise GokuStage0AdmissionError(
                f"{iid}: audit row is not an exact copy of its pinned input row"
            )
        motive = audited.get("motive_audit")
        if not isinstance(motive, dict):
            raise GokuStage0AdmissionError(f"{iid}: motive_audit is missing")
        feature_index = _require_int(
            motive.get("feature_index"), field=f"{iid}.feature_index"
        )
        if feature_index != index or feature_index >= len(features):
            raise GokuStage0AdmissionError(
                f"{iid}: feature_index is not the exact successful-row index"
            )
        if feature_ids[feature_index] != iid:
            raise GokuStage0AdmissionError(
                f"{iid}: descriptor ID does not match feature_index"
            )
        ok_ids.append(iid)
        computed_selected = _selection_from_bound_audit(
            motive, policy=SELECTION_POLICY, row_id=iid
        )
        if motive.get("selected") is not computed_selected:
            raise GokuStage0AdmissionError(
                f"{iid}: motive selected flag disagrees with frozen policy recomputation"
            )
        if computed_selected:
            selected_expected.append(audited)

        source_value = _unique_declared_value(
            original, SOURCE_KEYS, field="source video path"
        )
        target_value = _unique_declared_value(
            original, TARGET_KEYS, field="target video path"
        )
        instruction = _unique_declared_value(
            original, INSTRUCTION_KEYS, field="instruction"
        )
        if not instruction.strip():
            raise GokuStage0AdmissionError(f"{iid}: instruction is empty")
        source_path = _resolve_manifest_path(source_value, summary_root)
        target_path = _resolve_manifest_path(target_value, summary_root)
        for endpoint, path in (("source", source_path), ("target", target_path)):
            cache_key = str(_resolve_existing(path, label=f"{iid} {endpoint} video"))
            if cache_key not in media_cache:
                media_cache[cache_key] = _hash_physical_file(
                    path, label=f"{iid} {endpoint} video"
                )
        source_media = dict(
            media_cache[str(_resolve_existing(source_path, label=f"{iid} source video"))]
        )
        target_media = dict(
            media_cache[str(_resolve_existing(target_path, label=f"{iid} target video"))]
        )
        motive_source_path = _resolve_existing(
            Path(str(motive.get("source_path"))), label=f"{iid} motive source_path"
        )
        motive_target_path = _resolve_existing(
            Path(str(motive.get("target_path"))), label=f"{iid} motive target_path"
        )
        if motive_source_path != Path(
            source_media["resolved_path"]
        ):
            raise GokuStage0AdmissionError(f"{iid}: motive source_path does not close")
        if motive_target_path != Path(
            target_media["resolved_path"]
        ):
            raise GokuStage0AdmissionError(f"{iid}: motive target_path does not close")
        if motive.get("instruction") != instruction:
            raise GokuStage0AdmissionError(
                f"{iid}: motive instruction does not match pinned input"
            )
        for declared in _declared_sha_values(original, endpoint="source"):
            declared_source_sha_count += 1
            if declared != source_media["sha256"]:
                raise GokuStage0AdmissionError(
                    f"{iid}: declared source SHA disagrees with physical bytes"
                )
        for declared in _declared_sha_values(original, endpoint="target"):
            declared_target_sha_count += 1
            if declared != target_media["sha256"]:
                raise GokuStage0AdmissionError(
                    f"{iid}: declared target SHA disagrees with physical bytes"
                )

        authority = authority_by_id.get(iid)
        if authority is not None:
            if authority["source_sha256"] != source_media["sha256"]:
                raise GokuStage0AdmissionError(
                    f"{iid}: authority source SHA disagrees with physical bytes"
                )
            if authority["target_sha256"] != target_media["sha256"]:
                raise GokuStage0AdmissionError(
                    f"{iid}: authority target SHA disagrees with physical bytes"
                )
            canonical_source_id = str(authority["canonical_source_id"])
            canonical_target_id = str(authority["canonical_target_id"])
            instruction_equivalence_id = str(
                authority["instruction_equivalence_id"]
            )
            upstream_group_id = str(authority["upstream_group_id"])
            actor_scene_group_id = str(authority["actor_scene_group_id"])
        else:
            canonical_source_id = (
                _first_nonempty(
                    original,
                    ("canonical_source_id", "source_id", "content_group_id"),
                )
                or f"source-bytes:{source_media['sha256']}"
            )
            canonical_target_id = (
                _first_nonempty(original, ("canonical_target_id", "target_id"))
                or f"target-bytes:{target_media['sha256']}"
            )
            instruction_equivalence_id = (
                _first_nonempty(original, ("instruction_equivalence_id",))
                or f"instruction-bytes:{_sha256_bytes(instruction.encode('utf-8'))}"
            )
            upstream_group_id = (
                _first_nonempty(
                    original,
                    ("upstream_group_id", "content_group_id", "canonical_source_id"),
                )
                or canonical_source_id
            )
            actor_scene_group_id = (
                _first_nonempty(
                    original,
                    ("actor_scene_group_id", "scene_id", "content_group_id"),
                )
                or canonical_source_id
            )

        semantics = motive["instruction_semantics"]
        semantic_label = str(semantics["label"])
        label_counts[f"{motive['source_label']}->{motive['target_label']}"] += 1
        semantic_counts[semantic_label] += 1
        physical_rows.append(
            {
                "iid": iid,
                "selected": computed_selected,
                "source": source_media,
                "target": target_media,
                "instruction": instruction,
                "instruction_sha256": _sha256_bytes(instruction.encode("utf-8")),
                "canonical_source_id": canonical_source_id,
                "canonical_target_id": canonical_target_id,
                "instruction_equivalence_id": instruction_equivalence_id,
                "upstream_group_id": upstream_group_id,
                "actor_scene_group_id": actor_scene_group_id,
                "semantic": semantics,
                "action_family": _action_family(original),
                "motion_stratum": _motion_stratum(semantic_label, original),
                "feature_index": feature_index,
                "motive": motive,
                "audit_row_digest": _object_sha256(audited),
                "original_row_digest": _object_sha256(original),
                "original": original,
            }
        )

    if ok_ids != feature_ids:
        raise GokuStage0AdmissionError("descriptor IDs do not close in audit order")
    if list(selected_rows) != selected_expected:
        raise GokuStage0AdmissionError(
            "selected.jsonl is not the exact ordered selected=True subset of audit.jsonl"
        )
    if _counter_dict(label_counts) != summary.get("labels"):
        raise GokuStage0AdmissionError("summary label counts do not close")
    if _counter_dict(semantic_counts) != summary.get("instruction_semantics"):
        raise GokuStage0AdmissionError("summary instruction semantic counts do not close")

    # Exact byte aliases must never acquire two identities, even when no
    # external equivalence authority is available.  Conversely, multiple byte
    # SHAs may intentionally share one canonical ID when a pinned authority (or
    # a diagnostic embedded group) identifies transcodes.
    source_identity_by_sha: dict[str, str] = {}
    target_identity_by_sha: dict[str, str] = {}
    actor_scene_by_source_identity: dict[str, str] = {}
    actor_scene_by_upstream_group: dict[str, str] = {}
    actor_scene_by_source_sha: dict[str, str] = {}
    for row in physical_rows:
        source_sha = str(row["source"]["sha256"])
        target_sha = str(row["target"]["sha256"])
        canonical_source_id = str(row["canonical_source_id"])
        canonical_target_id = str(row["canonical_target_id"])
        upstream_group_id = str(row["upstream_group_id"])
        actor_scene_group_id = str(row["actor_scene_group_id"])
        source_prior = source_identity_by_sha.setdefault(
            source_sha, canonical_source_id
        )
        target_prior = target_identity_by_sha.setdefault(
            target_sha, canonical_target_id
        )
        if source_prior != canonical_source_id:
            raise GokuStage0AdmissionError(
                "one physical source SHA maps to multiple canonical source IDs"
            )
        if target_prior != canonical_target_id:
            raise GokuStage0AdmissionError(
                "one physical target SHA maps to multiple canonical target IDs"
            )
        for mapping, key, label in (
            (
                actor_scene_by_source_identity,
                canonical_source_id,
                "canonical source ID",
            ),
            (
                actor_scene_by_upstream_group,
                upstream_group_id,
                "upstream group ID",
            ),
            (actor_scene_by_source_sha, source_sha, "physical source SHA"),
        ):
            prior_actor_scene = mapping.setdefault(key, actor_scene_group_id)
            if prior_actor_scene != actor_scene_group_id:
                raise GokuStage0AdmissionError(
                    f"one {label} maps to multiple actor-scene groups"
                )

    selected_physical = [row for row in physical_rows if row["selected"]]
    variants_by_effective: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_physical:
        identity = {
            "schema_version": "goku-effective-edit-row-identity-0817-v1",
            "canonical_source_id": row["canonical_source_id"],
            "instruction_equivalence_id": row["instruction_equivalence_id"],
        }
        effective_row_id = _object_sha256(identity)
        row["effective_row_id"] = effective_row_id
        variants_by_effective[effective_row_id].append(row)

    # One semantic unit may have seed/target/path variants.  They are aliases,
    # never extra N.  Conflicting bound semantics are rejected.
    effective_units: list[dict[str, Any]] = []
    for effective_row_id in sorted(variants_by_effective):
        variants = variants_by_effective[effective_row_id]
        semantic_digests = {_object_sha256(row["semantic"]) for row in variants}
        if len(semantic_digests) != 1:
            raise GokuStage0AdmissionError(
                f"{effective_row_id}: aliases have conflicting instruction semantics"
            )
        resolved_families = {
            str(row["action_family"])
            for row in variants
            if row["action_family"] != "unresolved"
        }
        if len(resolved_families) > 1:
            raise GokuStage0AdmissionError(
                f"{effective_row_id}: aliases have conflicting action families"
            )
        ranked = sorted(
            variants,
            key=lambda row: (
                _object_sha256(
                    {
                        "source_sha256": row["source"]["sha256"],
                        "target_sha256": row["target"]["sha256"],
                        "instruction_sha256": row["instruction_sha256"],
                        "iid": row["iid"],
                    }
                ),
                str(row["iid"]),
            ),
        )
        representative = ranked[0]
        group_tokens = sorted(
            {
                f"canonical-source:{row['canonical_source_id']}"
                for row in variants
            }
            | {f"upstream-group:{row['upstream_group_id']}" for row in variants}
            | {
                f"actor-scene-group:{row['actor_scene_group_id']}"
                for row in variants
            }
        )
        effective_units.append(
            {
                "effective_row_id": effective_row_id,
                "representative": representative,
                "variants": ranked,
                "group_tokens": group_tokens,
            }
        )

    # Partition directly from the canonical actor-scene group.  Unlike hashing
    # the set of rows currently present in a batch, this remains stable when a
    # future admission adds another source or instruction to the same group.
    # The consistency checks above make canonical-source and upstream groups
    # subordinate to exactly one actor-scene partition unit.
    units_by_actor_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in effective_units:
        actor_scene_ids = {
            str(row["actor_scene_group_id"]) for row in unit["variants"]
        }
        if len(actor_scene_ids) != 1:
            raise GokuStage0AdmissionError(
                f"{unit['effective_row_id']}: aliases span actor-scene groups"
            )
        actor_scene_id = next(iter(actor_scene_ids))
        unit["partition_actor_scene_group_id"] = actor_scene_id
        units_by_actor_scene[actor_scene_id].append(unit)
    component_data: dict[str, tuple[str, list[str], str, int]] = {}
    for actor_scene_id, units in units_by_actor_scene.items():
        tokens = sorted({str(token) for unit in units for token in unit["group_tokens"]})
        component_id = _object_sha256(
            {
                "schema_version": "goku-actor-scene-partition-unit-0817-v1",
                "actor_scene_group_id": actor_scene_id,
            }
        )
        partition, bucket = _partition_for_component(component_id)
        component_data[actor_scene_id] = (
            component_id,
            tokens,
            partition,
            bucket,
        )

    policy_object = SELECTION_POLICY.as_dict()
    policy_digest = _object_sha256(policy_object)
    candidates: list[dict[str, Any]] = []
    for unit in effective_units:
        representative = unit["representative"]
        actor_scene_id = str(unit["partition_actor_scene_group_id"])
        component_id, component_tokens, partition, bucket = component_data[
            actor_scene_id
        ]
        motive = representative["motive"]
        variants = unit["variants"]
        action_family = (
            next(
                (
                    str(row["action_family"])
                    for row in variants
                    if row["action_family"] != "unresolved"
                ),
                "unresolved",
            )
        )
        candidates.append(
            {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "role": ROLE,
                "effective_row_id": unit["effective_row_id"],
                "representative_iid": representative["iid"],
                "alias_iids": sorted({str(row["iid"]) for row in variants}),
                "physical_variant_count": len(variants),
                "variant_endpoint_digests": sorted(
                    {
                        _object_sha256(
                            {
                                "source_sha256": row["source"]["sha256"],
                                "target_sha256": row["target"]["sha256"],
                                "instruction_sha256": row["instruction_sha256"],
                            }
                        )
                        for row in variants
                    }
                ),
                "source": {
                    **representative["source"],
                    "canonical_source_id": representative["canonical_source_id"],
                },
                "target": {
                    **representative["target"],
                    "canonical_target_id": representative["canonical_target_id"],
                    "provenance": _first_nonempty(
                        representative["original"],
                        ("target_provenance", "provenance", "dataset"),
                    )
                    or "licensed-paired-candidate-unverified",
                },
                "instruction": {
                    "text": representative["instruction"],
                    "sha256": representative["instruction_sha256"],
                    "instruction_equivalence_id": representative[
                        "instruction_equivalence_id"
                    ],
                },
                "instruction_semantics": representative["semantic"],
                "action_family": action_family,
                "motion_stratum": representative["motion_stratum"],
                "group_closure": {
                    "upstream_group_ids": sorted(
                        {str(row["upstream_group_id"]) for row in variants}
                    ),
                    "actor_scene_group_ids": sorted(
                        {str(row["actor_scene_group_id"]) for row in variants}
                    ),
                    "connected_component_id": component_id,
                    "component_tokens": component_tokens,
                    "external_equivalence_authority": bool(authority_by_id),
                },
                "candidate_partition": partition,
                "partition_bucket": bucket,
                "partition_policy_version": PARTITION_POLICY_VERSION,
                "stage0_evidence": {
                    "audit_row_digest": representative["audit_row_digest"],
                    "original_row_digest": representative["original_row_digest"],
                    "feature_index": representative["feature_index"],
                    "feature_archive_sha256": input_artifacts["descriptors"]["sha256"],
                    "feature_archive_compatibility_digest": feature_metadata[
                        "compatibility_digest"
                    ],
                    "descriptor_delta_norm": motive["descriptor_delta_norm"],
                    "source_label": motive["source_label"],
                    "target_label": motive["target_label"],
                    "source_metrics_digest": _object_sha256(motive["source_metrics"]),
                    "target_metrics_digest": _object_sha256(motive["target_metrics"]),
                    "selection_equation_recomputed_from_bound_audit_fields": True,
                    "selection_policy_digest": policy_digest,
                },
                "human_review": "pending",
                "required_human_reviews": REQUIRED_HUMAN_REVIEWS,
                "qualification_status": QUALIFICATION_STATUS,
                "qualification_receipt": None,
                "training_authorized": TRAINING_AUTHORIZED,
                "formal_d0_count_contribution": FORMAL_D0_COUNT_CONTRIBUTION,
            }
        )
    candidates.sort(key=lambda row: str(row["effective_row_id"]))
    _assert_group_disjoint(candidates)
    review_rows, quota_reservations = _assign_review_quotas(candidates)

    candidate_partition_counts = Counter(
        str(row["candidate_partition"]) for row in candidates
    )
    candidate_semantic_counts = Counter(
        str(row["instruction_semantics"]["label"]) for row in candidates
    )
    candidate_family_counts = Counter(str(row["action_family"]) for row in candidates)
    candidate_stratum_counts = Counter(str(row["motion_stratum"]) for row in candidates)
    aliases_collapsed = len(selected_physical) - len(candidates)
    code_artifact = _hash_physical_file(Path(__file__), label="admission code")

    # The inputs were parsed after their first pin check.  Re-read their hashes
    # before materialization so a concurrent replacement cannot turn parsed
    # bytes into an output attributed to a different frozen artifact.
    for name in ("original", "summary", "audit", "selected", "descriptors"):
        _verify_pinned_file(
            paths[name], expected_sha256=expected[name], label=f"final {name}"
        )
    if equivalence_authority is not None:
        _verify_pinned_file(
            equivalence_authority,
            expected_sha256=str(expected_equivalence_authority_sha256),
            label="final equivalence authority",
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "sealed"
        staging.mkdir()
        candidate_path = staging / "candidate_manifest.jsonl"
        review_path = staging / "review_queue.jsonl"
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(review_path, review_rows)
        candidate_artifact = _output_artifact(candidate_path)
        review_artifact = _output_artifact(review_path)
        quota_availability = {
            cell: quota_reservations.get(cell, 0) for cell in FUTURE_D0_QUOTAS
        }
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "role": ROLE,
            "closure_status": "PASS",
            "code": {
                "name": Path(__file__).name,
                "sha256": code_artifact["sha256"],
                "size_bytes": code_artifact["size_bytes"],
            },
            "input_artifacts": input_artifacts,
            "equivalence_authority": authority_artifact,
            "selection_policy": policy_object,
            "selection_policy_digest": policy_digest,
            "feature_archive": {
                "feature_kind": feature_metadata["feature_kind"],
                "dimension": feature_metadata["dimension"],
                "compatibility_digest": feature_metadata[
                    "compatibility_digest"
                ],
                "rows": len(features),
                "feature_index_and_id_closure": True,
            },
            "physical_media_closure": {
                "source_and_target_bytes_rehashed": True,
                "trusted_declared_hashes": False,
                "declared_source_hashes_checked": declared_source_sha_count,
                "declared_target_hashes_checked": declared_target_sha_count,
                "unique_physical_paths_hashed": len(media_cache),
            },
            "counts": {
                "original_rows": len(original_rows),
                "audit_rows": len(audit_rows),
                "selected_physical_rows": len(selected_physical),
                "effective_diagnostic_candidates": len(candidates),
                "seed_path_endpoint_aliases_collapsed": aliases_collapsed,
                "formal_d0_qualified_rows": 0,
                "training_authorized_rows": 0,
            },
            "candidate_partition_counts": _counter_dict(candidate_partition_counts),
            "candidate_semantic_counts": _counter_dict(candidate_semantic_counts),
            "candidate_action_family_counts": _counter_dict(candidate_family_counts),
            "candidate_motion_stratum_counts": _counter_dict(candidate_stratum_counts),
            "future_d0_licensed500_review_plan": {
                "desired_quotas": FUTURE_D0_QUOTAS,
                "diagnostic_candidates_reserved": quota_availability,
                "availability_shortfall": {
                    cell: max(0, FUTURE_D0_QUOTAS[cell] - quota_availability[cell])
                    for cell in FUTURE_D0_QUOTAS
                },
                "human_qualified": {cell: 0 for cell in FUTURE_D0_QUOTAS},
                "qualification_shortfall": dict(FUTURE_D0_QUOTAS),
                "quota_gate_passed": False,
            },
            "outputs": {
                "candidate_manifest": candidate_artifact,
                "review_queue": review_artifact,
            },
            "human_review": {
                "status": "not_started",
                "required_reviews_per_candidate": REQUIRED_HUMAN_REVIEWS,
                "accepted_receipts": 0,
            },
            "qualification_status": QUALIFICATION_STATUS,
            "training_authorized": TRAINING_AUTHORIZED,
            "formal_d0_count_contribution": FORMAL_D0_COUNT_CONTRIBUTION,
            "blocking_reasons": [
                "stage0_is_mechanical_diagnostic_only",
                "double_human_review_receipts_absent",
                "target_qualification_receipts_absent",
                "future_licensed500_quota_gate_not_satisfied",
            ]
            + (
                []
                if authority_by_id
                else ["external_transcode_actor_scene_equivalence_authority_absent"]
            ),
        }
        receipt_path = staging / "admission_receipt.json"
        _write_json(receipt_path, receipt)
        receipt_artifact = _output_artifact(receipt_path)
        done = {
            "schema_version": DONE_SCHEMA_VERSION,
            "role": ROLE,
            "closure_status": "PASS",
            "candidate_manifest": candidate_artifact,
            "review_queue": review_artifact,
            "admission_receipt": receipt_artifact,
            "qualification_status": QUALIFICATION_STATUS,
            "training_authorized": TRAINING_AUTHORIZED,
            "formal_d0_count_contribution": FORMAL_D0_COUNT_CONTRIBUTION,
        }
        _write_json(staging / "DONE.json", done)
        if output_dir.exists():
            raise GokuStage0AdmissionError(
                f"output directory appeared during admission: {output_dir}"
            )
        os.replace(staging, output_dir)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify GOKU motive stage0 and emit diagnostic-only candidates; "
            "never authorizes training."
        )
    )
    parser.add_argument("--stage0-dir", required=True, type=Path)
    parser.add_argument("--original-selected", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-original-sha256", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--expected-selected-sha256", required=True)
    parser.add_argument("--expected-descriptors-sha256", required=True)
    parser.add_argument("--equivalence-authority", type=Path)
    parser.add_argument("--expected-equivalence-authority-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = admit_stage0(
            stage0_dir=args.stage0_dir,
            original_selected=args.original_selected,
            output_dir=args.output_dir,
            expected_original_sha256=args.expected_original_sha256,
            expected_summary_sha256=args.expected_summary_sha256,
            expected_audit_sha256=args.expected_audit_sha256,
            expected_selected_sha256=args.expected_selected_sha256,
            expected_descriptors_sha256=args.expected_descriptors_sha256,
            equivalence_authority=args.equivalence_authority,
            expected_equivalence_authority_sha256=(
                args.expected_equivalence_authority_sha256
            ),
        )
    except GokuStage0AdmissionError as error:
        print(f"goku stage0 admission rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
