#!/usr/bin/env python3
"""Sealed evaluator evidence for one native PAIR-v5 RV2V candidate.

The safe-Pareto selector consumes scalars, but those scalars are optimizer
authorization evidence only when they are cryptographically tied to the
physical candidate that was evaluated.  This module closes that boundary.
One packet binds the native rollout receipt and MP4, source/caption identity,
the complete evaluator/model registry, raw and reported scores, and every
registered hard-negative decision.

The packet contains no pixels, latents, noise, proposal, donor, mask, flow,
pose, track, or trajectory.  It is an offline evidence receipt only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping


REGISTRY_SCHEMA = "bernini-pair-v5-candidate-evaluator-registry-v1"
EVALUATOR_SCHEMA = "bernini-pair-v5-candidate-evaluator-binding-v1"
PACKET_SCHEMA = "bernini-pair-v5-candidate-evaluator-packet-v1"

EVALUATOR_AXES = ("action", "identity", "consistency", "quality", "hard_negative")
SCORE_AXES = ("action", "identity", "consistency", "quality")
HARD_NEGATIVE_FLAGS = (
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "blur",
    "camera",
    "appearance_contamination",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_EVALUATOR_FIELDS = frozenset(
    {"schema_version", "evaluator_id", "evaluator_sha256", "model_digest"}
)
_REGISTRY_FIELDS = frozenset(
    {"schema_version", "axis_order", "evaluators", "registry_digest"}
)
_PACKET_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "rollout_receipt_digest",
        "mp4_sha256",
        "source_video_sha256",
        "complete_caption_sha256",
        "evaluator_registry_digest",
        "upstream_evaluator_receipt_digest_by_axis",
        "raw_scores",
        "reported_scores",
        "hard_negative_flags",
        "input_closure",
        "packet_digest",
    }
)
_INPUT_CLOSURE = {
    "scalar_and_boolean_evidence_only": True,
    "candidate_media_bound_by_sha256_but_not_loaded": True,
    "t2v_proposal_media_consumed": False,
    "proposal_latent_or_noise_consumed": False,
    "donor_or_paired_target_consumed": False,
    "mask_flow_pose_track_trajectory_consumed": False,
}


class PairV5EvaluatorPacketError(ValueError):
    """A candidate evaluator registry or packet is not closed and replayable."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV5EvaluatorPacketError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV5EvaluatorPacketError(f"{label} must be a mapping")
    keys = set(value)
    if not all(isinstance(key, str) for key in keys) or keys != set(fields):
        raise PairV5EvaluatorPacketError(
            f"{label} closure differs; missing={sorted(fields - keys)}, "
            f"extra={sorted(repr(key) for key in keys - fields)}"
        )
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PairV5EvaluatorPacketError(f"{label} must be lowercase SHA-256")
    return value


def _slug(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
        raise PairV5EvaluatorPacketError(f"{label} must be a lowercase safe identifier")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairV5EvaluatorPacketError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PairV5EvaluatorPacketError(f"{label} must be finite numeric")
    return result


def _unit(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise PairV5EvaluatorPacketError(f"{label} must lie in [0,1]")
    return result


def _embedded_digest(value: Mapping[str, Any], field: str, *, label: str) -> str:
    declared = _sha256(value.get(field), label=f"{label} {field}")
    unsigned = dict(value)
    unsigned.pop(field)
    if object_sha256(unsigned) != declared:
        raise PairV5EvaluatorPacketError(f"{label} embedded digest differs")
    return declared


def make_evaluator_binding(
    evaluator_id: str, *, evaluator_sha256: str, model_digest: str
) -> dict[str, Any]:
    return validate_evaluator_binding(
        {
            "schema_version": EVALUATOR_SCHEMA,
            "evaluator_id": evaluator_id,
            "evaluator_sha256": evaluator_sha256,
            "model_digest": model_digest,
        }
    )


def validate_evaluator_binding(value: Any) -> dict[str, Any]:
    row = _closed(value, _EVALUATOR_FIELDS, label="evaluator binding")
    if row["schema_version"] != EVALUATOR_SCHEMA:
        raise PairV5EvaluatorPacketError("evaluator binding schema differs")
    return {
        "schema_version": EVALUATOR_SCHEMA,
        "evaluator_id": _slug(row["evaluator_id"], label="evaluator_id"),
        "evaluator_sha256": _sha256(
            row["evaluator_sha256"], label="evaluator_sha256"
        ),
        "model_digest": _sha256(row["model_digest"], label="model_digest"),
    }


def make_registry(evaluators: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checked = {
        axis: validate_evaluator_binding(evaluators[axis]) for axis in EVALUATOR_AXES
    } if isinstance(evaluators, Mapping) and set(evaluators) == set(EVALUATOR_AXES) else None
    if checked is None:
        raise PairV5EvaluatorPacketError("evaluator registry must cover every axis exactly")
    unsigned = {
        "schema_version": REGISTRY_SCHEMA,
        "axis_order": list(EVALUATOR_AXES),
        "evaluators": checked,
    }
    return validate_registry({**unsigned, "registry_digest": object_sha256(unsigned)})


def validate_registry(value: Any) -> dict[str, Any]:
    row = _closed(value, _REGISTRY_FIELDS, label="evaluator registry")
    if row["schema_version"] != REGISTRY_SCHEMA or row["axis_order"] != list(
        EVALUATOR_AXES
    ):
        raise PairV5EvaluatorPacketError("evaluator registry schema/order differs")
    evaluators = row["evaluators"]
    if not isinstance(evaluators, Mapping) or set(evaluators) != set(EVALUATOR_AXES):
        raise PairV5EvaluatorPacketError("evaluator registry axis closure differs")
    checked = {
        axis: validate_evaluator_binding(evaluators[axis]) for axis in EVALUATOR_AXES
    }
    normalized = {
        "schema_version": REGISTRY_SCHEMA,
        "axis_order": list(EVALUATOR_AXES),
        "evaluators": checked,
        "registry_digest": row["registry_digest"],
    }
    _embedded_digest(normalized, "registry_digest", label="evaluator registry")
    return normalized


def make_packet(
    candidate_id: str,
    *,
    rollout_receipt_digest: str,
    mp4_sha256: str,
    source_video_sha256: str,
    complete_caption_sha256: str,
    evaluator_registry_digest: str,
    upstream_evaluator_receipt_digest_by_axis: Mapping[str, str],
    raw_scores: Mapping[str, float],
    reported_scores: Mapping[str, float],
    hard_negative_flags: Mapping[str, bool],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": PACKET_SCHEMA,
        "candidate_id": candidate_id,
        "rollout_receipt_digest": rollout_receipt_digest,
        "mp4_sha256": mp4_sha256,
        "source_video_sha256": source_video_sha256,
        "complete_caption_sha256": complete_caption_sha256,
        "evaluator_registry_digest": evaluator_registry_digest,
        "upstream_evaluator_receipt_digest_by_axis": dict(
            upstream_evaluator_receipt_digest_by_axis
        ),
        "raw_scores": dict(raw_scores),
        "reported_scores": dict(reported_scores),
        "hard_negative_flags": dict(hard_negative_flags),
        "input_closure": dict(_INPUT_CLOSURE),
    }
    return validate_packet({**unsigned, "packet_digest": object_sha256(unsigned)})


def validate_packet(value: Any) -> dict[str, Any]:
    row = _closed(value, _PACKET_FIELDS, label="evaluator packet")
    if row["schema_version"] != PACKET_SCHEMA:
        raise PairV5EvaluatorPacketError("evaluator packet schema differs")
    upstream = row["upstream_evaluator_receipt_digest_by_axis"]
    raw_scores = row["raw_scores"]
    reported_scores = row["reported_scores"]
    flags = row["hard_negative_flags"]
    if not isinstance(upstream, Mapping) or set(upstream) != set(EVALUATOR_AXES):
        raise PairV5EvaluatorPacketError("upstream evaluator receipt closure differs")
    if not isinstance(raw_scores, Mapping) or set(raw_scores) != set(SCORE_AXES):
        raise PairV5EvaluatorPacketError("raw score closure differs")
    if not isinstance(reported_scores, Mapping) or set(reported_scores) != set(SCORE_AXES):
        raise PairV5EvaluatorPacketError("reported score closure differs")
    if not isinstance(flags, Mapping) or set(flags) != set(HARD_NEGATIVE_FLAGS):
        raise PairV5EvaluatorPacketError("hard-negative flag closure differs")
    checked_raw = {
        axis: _finite(raw_scores[axis], label=f"raw {axis} score") for axis in SCORE_AXES
    }
    checked_reported = {
        axis: _unit(reported_scores[axis], label=f"reported {axis} score")
        for axis in SCORE_AXES
    }
    # There is a registered calibration map only for action.  Preservation
    # scores therefore remain their transparent raw unit-interval values.
    if any(
        checked_raw[axis] != checked_reported[axis]
        for axis in ("identity", "consistency", "quality")
    ):
        raise PairV5EvaluatorPacketError(
            "preservation scores cannot change without a registered calibrator"
        )
    checked_flags: dict[str, bool] = {}
    for name in HARD_NEGATIVE_FLAGS:
        if type(flags[name]) is not bool:
            raise PairV5EvaluatorPacketError(f"hard-negative flag {name} must be boolean")
        checked_flags[name] = flags[name]
    normalized = {
        "schema_version": PACKET_SCHEMA,
        "candidate_id": _slug(row["candidate_id"], label="candidate_id"),
        "rollout_receipt_digest": _sha256(
            row["rollout_receipt_digest"], label="rollout_receipt_digest"
        ),
        "mp4_sha256": _sha256(row["mp4_sha256"], label="mp4_sha256"),
        "source_video_sha256": _sha256(
            row["source_video_sha256"], label="source_video_sha256"
        ),
        "complete_caption_sha256": _sha256(
            row["complete_caption_sha256"], label="complete_caption_sha256"
        ),
        "evaluator_registry_digest": _sha256(
            row["evaluator_registry_digest"], label="evaluator_registry_digest"
        ),
        "upstream_evaluator_receipt_digest_by_axis": {
            axis: _sha256(upstream[axis], label=f"{axis} upstream receipt digest")
            for axis in EVALUATOR_AXES
        },
        "raw_scores": checked_raw,
        "reported_scores": checked_reported,
        "hard_negative_flags": checked_flags,
        "input_closure": row["input_closure"],
        "packet_digest": row["packet_digest"],
    }
    if normalized["input_closure"] != _INPUT_CLOSURE:
        raise PairV5EvaluatorPacketError("evaluator packet admits a forbidden input")
    _embedded_digest(normalized, "packet_digest", label="evaluator packet")
    return normalized


def verify_packet(
    packet: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    safe_candidate: Mapping[str, Any],
    action_score_receipt: Mapping[str, Any],
    rollout_receipt_digest: str,
    mp4_sha256: str,
    source_video_sha256: str,
    complete_caption_sha256: str,
    expected_action_evaluator_sha256: str,
    expected_action_model_digest: str,
) -> dict[str, Any]:
    """Replay every packet-to-rollout, scorer, and selector commitment."""

    checked = validate_packet(packet)
    checked_registry = validate_registry(registry)
    action_binding = checked_registry["evaluators"]["action"]
    expected = {
        "candidate_id": safe_candidate.get("candidate_id"),
        "rollout_receipt_digest": rollout_receipt_digest,
        "mp4_sha256": mp4_sha256,
        "source_video_sha256": source_video_sha256,
        "complete_caption_sha256": complete_caption_sha256,
        "evaluator_registry_digest": checked_registry["registry_digest"],
    }
    if any(checked[name] != value for name, value in expected.items()):
        raise PairV5EvaluatorPacketError("evaluator packet physical candidate binding differs")
    if (
        safe_candidate.get("evaluator_packet_digest") != checked["packet_digest"]
        or safe_candidate.get("rollout_receipt_digest") != rollout_receipt_digest
        or checked["reported_scores"]
        != {
            axis: safe_candidate.get(f"{axis}_score") for axis in SCORE_AXES
        }
        or checked["hard_negative_flags"] != safe_candidate.get("hard_negative_flags")
    ):
        raise PairV5EvaluatorPacketError("safe candidate does not cover evaluator packet")
    if (
        checked["raw_scores"]["action"]
        != action_score_receipt.get("raw_candidate_own_score")
        or checked["reported_scores"]["action"]
        != action_score_receipt.get("calibrated_action_score")
        or checked["upstream_evaluator_receipt_digest_by_axis"]["action"]
        != action_score_receipt.get("candidate_evaluator_receipt_digest")
    ):
        raise PairV5EvaluatorPacketError("action score receipt is not the packet action evidence")
    if (
        action_binding["evaluator_sha256"]
        != _sha256(expected_action_evaluator_sha256, label="expected action evaluator SHA")
        or action_binding["model_digest"]
        != _sha256(expected_action_model_digest, label="expected action model digest")
    ):
        raise PairV5EvaluatorPacketError("action evaluator/model registry differs")
    return checked


__all__ = [
    "EVALUATOR_AXES",
    "EVALUATOR_SCHEMA",
    "HARD_NEGATIVE_FLAGS",
    "PACKET_SCHEMA",
    "PairV5EvaluatorPacketError",
    "REGISTRY_SCHEMA",
    "SCORE_AXES",
    "make_evaluator_binding",
    "make_packet",
    "make_registry",
    "object_sha256",
    "validate_evaluator_binding",
    "validate_packet",
    "validate_registry",
    "verify_packet",
]
