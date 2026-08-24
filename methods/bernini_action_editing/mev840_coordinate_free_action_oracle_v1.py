#!/usr/bin/env python3
"""Coordinate-free MEV840 target-action oracle and post-generation scorer.

This module is deliberately outside the video generator.  It accepts only a
small, audited 21-phase relation/timing summary and compares it with the same
summary extracted from a generated candidate.  It cannot consume or emit RGB,
masks, spatial coordinates, model features, latents, Gaussian noise, or Q/K/V.

The real target is therefore an *evaluation/selection oracle*, never a model
condition or a supervision tensor.  Media paths and hashes belong in a
separate observer provenance receipt and are rejected from this ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = "mev840-coordinate-free-action-oracle-v1"
SCORE_SCHEMA = "mev840-coordinate-free-action-score-v1"
PHASE_COUNT = 21
ROLES = ("human_agent", "moving_object", "recipient", "head")
CHANNELS = (
    "object_motion_progress",
    "object_incremental_motion",
    "agent_object_gap_ratio",
    "object_recipient_gap_ratio",
    "agent_object_contact",
    "object_recipient_contact",
    "object_scale_log_ratio",
    "object_axis_change",
    "head_profile_change",
)
EVENTS = (
    "turn_onset",
    "turn_peak",
    "agent_object_contact_end",
    "recipient_contact_start",
    "release",
)

TOP_LEVEL_KEYS = {
    "schema",
    "case_id",
    "phase_count",
    "roles",
    "channels",
    "phase_relations",
    "events",
    "authority",
    "evidence_boundary",
    "representation_digest",
}
PHASE_KEYS = {"phase_index", "phase_time"} | set(CHANNELS)
AUTHORITY = {
    "selection_only": True,
    "generator_read_authorized": False,
    "renderer_condition_authorized": False,
    "training_authorized": False,
    "optimizer_updates": 0,
}

# Fail closed on both exact forbidden keys and appearance/spatial-model words
# embedded in longer names.  ``phase_relations`` is intentionally permitted;
# it carries only the fixed scalar channels above.
FORBIDDEN_KEY_PARTS = (
    "path",
    "rgb",
    "pixel",
    "mask",
    "bbox",
    "box_",
    "_xy",
    "coordinate",
    "flow",
    "feature",
    "embedding",
    "latent",
    "gaussian",
    "query",
    "key_tensor",
    "value_tensor",
    "hidden",
    "attention",
    "appearance",
    "sha256",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CoordinateFreeActionOracleError(RuntimeError):
    """An oracle representation or score violated its sealed ABI."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise CoordinateFreeActionOracleError("non-canonical JSON value") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoordinateFreeActionOracleError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CoordinateFreeActionOracleError(f"{label} is not finite")
    return result


def _forbid_leakage_keys(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise CoordinateFreeActionOracleError(
                    f"{location} contains a non-string key"
                )
            key = raw_key.lower()
            if any(part in key for part in FORBIDDEN_KEY_PARTS):
                raise CoordinateFreeActionOracleError(
                    f"forbidden leakage key at {location}.{raw_key}"
                )
            _forbid_leakage_keys(child, location=f"{location}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbid_leakage_keys(child, location=f"{location}[{index}]")
    elif isinstance(value, (str, int, float, bool)) or value is None:
        return
    else:
        raise CoordinateFreeActionOracleError(
            f"unsupported value type at {location}"
        )


def representation_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("representation_digest", None)
    return payload


def validate_representation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != TOP_LEVEL_KEYS:
        raise CoordinateFreeActionOracleError("representation top-level ABI differs")
    if value.get("schema") != SCHEMA or value.get("case_id") != "MEV840":
        raise CoordinateFreeActionOracleError("representation identity differs")
    if value.get("phase_count") != PHASE_COUNT:
        raise CoordinateFreeActionOracleError("phase count differs")
    if value.get("roles") != list(ROLES) or value.get("channels") != list(CHANNELS):
        raise CoordinateFreeActionOracleError("role/channel ABI differs")
    if value.get("authority") != AUTHORITY:
        raise CoordinateFreeActionOracleError("selection-only authority differs")
    boundary = value.get("evidence_boundary")
    if boundary not in {
        "human_reviewed_sparse_oracle_not_mechanical_tracking",
        "frozen_sam2_geometry_reduced_before_export",
    }:
        raise CoordinateFreeActionOracleError("evidence boundary differs")
    rows = value.get("phase_relations")
    if not isinstance(rows, list) or len(rows) != PHASE_COUNT:
        raise CoordinateFreeActionOracleError("phase relation count differs")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != PHASE_KEYS:
            raise CoordinateFreeActionOracleError(f"phase {index} ABI differs")
        if row.get("phase_index") != index:
            raise CoordinateFreeActionOracleError(f"phase {index} index differs")
        phase_time = _finite_number(row.get("phase_time"), label="phase time")
        if abs(phase_time - index / (PHASE_COUNT - 1)) > 1.0e-9:
            raise CoordinateFreeActionOracleError(f"phase {index} time differs")
        for channel in CHANNELS:
            item = _finite_number(row.get(channel), label=f"phase {index} {channel}")
            if channel in {
                "object_motion_progress",
                "object_incremental_motion",
                "agent_object_contact",
                "object_recipient_contact",
                "object_axis_change",
                "head_profile_change",
            } and not 0.0 <= item <= 1.0:
                raise CoordinateFreeActionOracleError(
                    f"phase {index} {channel} is outside [0,1]"
                )
            if channel in {"agent_object_gap_ratio", "object_recipient_gap_ratio"} and item < 0:
                raise CoordinateFreeActionOracleError(
                    f"phase {index} {channel} is negative"
                )
            if channel == "object_scale_log_ratio" and abs(item) > 4.0:
                raise CoordinateFreeActionOracleError(
                    f"phase {index} scale ratio is implausible"
                )
    events = value.get("events")
    if not isinstance(events, dict) or set(events) != set(EVENTS):
        raise CoordinateFreeActionOracleError("event ABI differs")
    for event in EVENTS:
        item = events[event]
        if item is not None:
            numeric = _finite_number(item, label=event)
            if not 0.0 <= numeric <= 1.0:
                raise CoordinateFreeActionOracleError(f"{event} is outside [0,1]")
    _forbid_leakage_keys(value)
    digest = value.get("representation_digest")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise CoordinateFreeActionOracleError("representation digest differs")
    if object_sha256(representation_payload(value)) != digest:
        raise CoordinateFreeActionOracleError("representation self-hash differs")
    return value


def make_representation(
    phase_relations: Sequence[Mapping[str, Any]],
    events: Mapping[str, Any],
    *,
    evidence_boundary: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "case_id": "MEV840",
        "phase_count": PHASE_COUNT,
        "roles": list(ROLES),
        "channels": list(CHANNELS),
        "phase_relations": [dict(item) for item in phase_relations],
        "events": dict(events),
        "authority": dict(AUTHORITY),
        "evidence_boundary": evidence_boundary,
    }
    payload["representation_digest"] = object_sha256(payload)
    return validate_representation(payload)


def read_representation(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CoordinateFreeActionOracleError("representation must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except Exception as error:
        raise CoordinateFreeActionOracleError("cannot parse representation") from error
    return validate_representation(value)


def _channel_error(target: Sequence[float], candidate: Sequence[float], scale: float) -> float:
    return sum(min(abs(float(a) - float(b)) / scale, 1.0) for a, b in zip(target, candidate)) / len(target)


def _event_error(target: Any, candidate: Any) -> float:
    if target is None and candidate is None:
        return 0.0
    if target is None or candidate is None:
        return 1.0
    return min(abs(float(target) - float(candidate)) / 0.25, 1.0)


def score_representations(target: Any, candidate: Any) -> dict[str, Any]:
    """Compare two sanitized summaries; no media/model state is accepted."""

    target = validate_representation(target)
    candidate = validate_representation(candidate)
    target_rows = target["phase_relations"]
    candidate_rows = candidate["phase_relations"]
    scales = {
        "object_motion_progress": 0.35,
        "object_incremental_motion": 0.25,
        "agent_object_gap_ratio": 0.35,
        "object_recipient_gap_ratio": 0.35,
        "agent_object_contact": 1.0,
        "object_recipient_contact": 1.0,
        "object_scale_log_ratio": 0.35,
        "object_axis_change": 0.40,
        "head_profile_change": 0.35,
    }
    channel_weights = {
        "object_motion_progress": 2.0,
        "object_incremental_motion": 1.0,
        "agent_object_gap_ratio": 1.5,
        "object_recipient_gap_ratio": 2.0,
        "agent_object_contact": 1.5,
        "object_recipient_contact": 2.0,
        "object_scale_log_ratio": 0.35,
        "object_axis_change": 0.50,
        "head_profile_change": 1.0,
    }
    channel_errors: dict[str, float] = {}
    for channel in CHANNELS:
        channel_errors[channel] = _channel_error(
            [row[channel] for row in target_rows],
            [row[channel] for row in candidate_rows],
            scales[channel],
        )
    trajectory_channels = (
        "object_motion_progress",
        "object_incremental_motion",
        "agent_object_gap_ratio",
        "object_recipient_gap_ratio",
        "object_scale_log_ratio",
        "object_axis_change",
    )
    contact_channels = ("agent_object_contact", "object_recipient_contact")
    trajectory_error = sum(
        channel_errors[name] * channel_weights[name] for name in trajectory_channels
    ) / sum(channel_weights[name] for name in trajectory_channels)
    contact_error = sum(
        channel_errors[name] * channel_weights[name] for name in contact_channels
    ) / sum(channel_weights[name] for name in contact_channels)
    head_turn_error = channel_errors["head_profile_change"]
    event_errors = {
        name: _event_error(target["events"][name], candidate["events"][name])
        for name in EVENTS
    }
    event_error = sum(event_errors.values()) / len(event_errors)
    total_error = (
        0.45 * trajectory_error
        + 0.25 * contact_error
        + 0.15 * head_turn_error
        + 0.15 * event_error
    )
    action_score = max(0.0, min(1.0, 1.0 - total_error))
    return {
        "schema": SCORE_SCHEMA,
        "case_id": "MEV840",
        "target_representation_digest": target["representation_digest"],
        "candidate_representation_digest": candidate["representation_digest"],
        "scores": {
            "action": action_score,
            "trajectory": 1.0 - trajectory_error,
            "contact_release": 1.0 - contact_error,
            "head_turn": 1.0 - head_turn_error,
            "event_timing": 1.0 - event_error,
        },
        "channel_errors": channel_errors,
        "event_errors": event_errors,
        "decision": {
            "action_gate_threshold": 0.78,
            "action_gate_passed": action_score >= 0.78,
            "appearance_quality_gate_external_required": True,
            "appearance_quality_gate_passed": None,
            "selection_authorized": False,
        },
        "authority": {
            "post_generation_scoring_only": True,
            "generator_read_authorized": False,
            "renderer_condition_authorized": False,
            "training_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("representation", type=Path)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--target", required=True, type=Path)
    score_parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        value = read_representation(args.representation)
        print(value["representation_digest"])
        return 0
    result = score_representations(
        read_representation(args.target), read_representation(args.candidate)
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
