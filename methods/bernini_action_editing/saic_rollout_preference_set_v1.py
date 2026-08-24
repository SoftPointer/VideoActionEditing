#!/usr/bin/env python3
"""Fail-closed SAIC Stage-B on-policy preference-set admission.

This module is the trust boundary between decoded exact81 rollouts and the
rectified-flow preference objective.  Preference endpoints are *only* current
source-conditioned policy rollouts that have been decoded, codec-roundtripped,
and VAE re-encoded.  Pure-T2V media may calibrate an event evaluator upstream,
but no T2V pixel, latent, noise, or media digest has a route into an endpoint.

Admission is a hard partial order, never a scalar reward.  The chosen endpoint
must improve the event score and, independently on every preservation axis,
both endpoints must pass an absolute floor and the chosen endpoint must be
non-inferior to the rejected endpoint.  A dog and a human pair are jointly
required; otherwise the returned set authorizes exactly zero updates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any


SCHEMA_VERSION = "bernini-saic-rollout-preference-candidate-v1"
ROLLOUT_RECEIPT_SCHEMA_VERSION = "bernini-saic-on-policy-rollout-receipt-v1"
CODEC_RECEIPT_SCHEMA_VERSION = "bernini-saic-codec-reencoded-endpoint-v1"
PREFERENCE_SET_SCHEMA_VERSION = "bernini-saic-rollout-preference-set-v1"
PAIR_SCHEMA_VERSION = "bernini-saic-hard-pareto-pair-v1"

ARMS = ("dog", "human")
PRESERVATION_AXES = (
    "identity",
    "camera",
    "background",
    "non_target",
    "quality",
    "source_bind",
    "inverse",
)
EXACT40_UPDATE_INDICES = (4, 12, 20, 28, 33, 34, 35, 37)
EXACT40_FORBIDDEN_UPDATE_INDICES = (38, 39)
FRAME_COUNT = 81
FPS_NUMERATOR = 25
FPS_DENOMINATOR = 1
GENERATION_MODE = "current_policy_native_source_conditioned_rv2v"
ENDPOINT_ROLE = "codec_reencoded_on_policy_source_conditioned_endpoint"
PURE_T2V_ROLE = "event_evaluator_calibration_only_not_endpoint"
PAIR_ORDER = (
    "event_relative_gain_then_seven_axis_floor_and_noninferiority_no_compensation"
)

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "arm",
        "source_id",
        "instruction_id",
        "policy_sha256",
        "source_media_sha256",
        "output_media_sha256",
        "endpoint_latent_sha256",
        "declared_role",
        "event_score",
        "axis_scores",
        "rollout_receipt",
        "codec_reencode_receipt",
        "candidate_digest",
    }
)
_ROLLOUT_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "generation_mode",
        "on_policy",
        "weights_frozen_during_rollout",
        "source_conditioned",
        "policy_sha256",
        "source_id",
        "source_media_sha256",
        "output_media_sha256",
        "frame_count",
        "fps_numerator",
        "fps_denominator",
        "exact40_step_count",
        "preference_update_indices",
        "pure_t2v_media_read",
        "pure_t2v_latent_read",
        "pure_t2v_noise_read",
        "target_media_read",
        "paired_target_read",
        "receipt_digest",
    }
)
_CODEC_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "input_output_media_sha256",
        "decoded_rgb24_sha256",
        "codec_name",
        "codec_bitstream_sha256",
        "codec_decoded_rgb24_sha256",
        "vae_id",
        "vae_weights_sha256",
        "reencoded_latent_sha256",
        "frame_count",
        "fps_numerator",
        "fps_denominator",
        "endpoint_detached",
        "receipt_digest",
    }
)


class SAICRolloutPreferenceError(ValueError):
    """An endpoint or preference admission violates the Stage-B contract."""


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
        raise SAICRolloutPreferenceError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SAICRolloutPreferenceError(f"{label} must be a mapping")
    keys = set(value)
    if not all(type(key) is str for key in keys):
        raise SAICRolloutPreferenceError(f"{label} keys must be strings")
    if keys != fields:
        raise SAICRolloutPreferenceError(
            f"{label} schema differs; missing={sorted(fields - keys)}, "
            f"extra={sorted(keys - fields)}"
        )
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise SAICRolloutPreferenceError(f"{label} must be a canonical identifier")
    return value


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise SAICRolloutPreferenceError(f"{label} must be lowercase SHA-256")
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise SAICRolloutPreferenceError(f"{label} must be boolean")
    return value


def _exact_integer(value: Any, expected: int, *, label: str) -> int:
    if type(value) is not int or value != expected:
        raise SAICRolloutPreferenceError(f"{label} must be exact integer {expected}")
    return value


def _unit_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SAICRolloutPreferenceError(f"{label} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise SAICRolloutPreferenceError(f"{label} must lie in [0,1]")
    return score


def _axis_map(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(PRESERVATION_AXES):
        raise SAICRolloutPreferenceError(
            f"{label} must contain exactly the seven registered axes"
        )
    return {
        axis: _unit_score(value[axis], label=f"{label}.{axis}")
        for axis in PRESERVATION_AXES
    }


def _sealed(row: Mapping[str, Any], *, field: str, label: str) -> None:
    digest = _sha(row[field], label=f"{label}.{field}")
    body = {key: value for key, value in row.items() if key != field}
    if digest != object_sha256(body):
        raise SAICRolloutPreferenceError(f"{label} digest differs")


def validate_candidate(value: Any) -> dict[str, Any]:
    """Validate and return a deep canonical copy of one eligible endpoint."""

    row = _closed(value, _CANDIDATE_FIELDS, label="candidate")
    if row["schema_version"] != SCHEMA_VERSION:
        raise SAICRolloutPreferenceError("candidate schema_version differs")
    candidate_id = _safe_id(row["candidate_id"], label="candidate_id")
    arm = row["arm"]
    if arm not in ARMS:
        raise SAICRolloutPreferenceError("candidate arm must be dog or human")
    source_id = _safe_id(row["source_id"], label="source_id")
    _safe_id(row["instruction_id"], label="instruction_id")
    policy_sha = _sha(row["policy_sha256"], label="policy_sha256")
    source_sha = _sha(row["source_media_sha256"], label="source_media_sha256")
    output_sha = _sha(row["output_media_sha256"], label="output_media_sha256")
    endpoint_sha = _sha(row["endpoint_latent_sha256"], label="endpoint_latent_sha256")
    if row["declared_role"] != ENDPOINT_ROLE:
        raise SAICRolloutPreferenceError("candidate is not an on-policy endpoint")
    _unit_score(row["event_score"], label="event_score")
    _axis_map(row["axis_scores"], label="axis_scores")

    rollout = _closed(row["rollout_receipt"], _ROLLOUT_FIELDS, label="rollout receipt")
    if rollout["schema_version"] != ROLLOUT_RECEIPT_SCHEMA_VERSION:
        raise SAICRolloutPreferenceError("rollout receipt schema differs")
    _sealed(rollout, field="receipt_digest", label="rollout receipt")
    required_equal = {
        "candidate_id": candidate_id,
        "policy_sha256": policy_sha,
        "source_id": source_id,
        "source_media_sha256": source_sha,
        "output_media_sha256": output_sha,
    }
    for key, expected in required_equal.items():
        if rollout[key] != expected:
            raise SAICRolloutPreferenceError(f"rollout receipt {key} differs")
    if rollout["generation_mode"] != GENERATION_MODE:
        raise SAICRolloutPreferenceError("pure-T2V or non-native endpoint is forbidden")
    for key in ("on_policy", "weights_frozen_during_rollout", "source_conditioned"):
        if _boolean(rollout[key], label=f"rollout receipt {key}") is not True:
            raise SAICRolloutPreferenceError(f"rollout receipt {key} must be true")
    for key in (
        "pure_t2v_media_read",
        "pure_t2v_latent_read",
        "pure_t2v_noise_read",
        "target_media_read",
        "paired_target_read",
    ):
        if _boolean(rollout[key], label=f"rollout receipt {key}") is not False:
            raise SAICRolloutPreferenceError(f"rollout receipt {key} must be false")
    for key, expected in (
        ("frame_count", FRAME_COUNT),
        ("fps_numerator", FPS_NUMERATOR),
        ("fps_denominator", FPS_DENOMINATOR),
        ("exact40_step_count", 40),
    ):
        _exact_integer(rollout[key], expected, label=f"rollout receipt {key}")
    update_indices = rollout["preference_update_indices"]
    if (
        isinstance(update_indices, (str, bytes))
        or not isinstance(update_indices, Sequence)
        or any(type(index) is not int for index in update_indices)
        or tuple(update_indices) != EXACT40_UPDATE_INDICES
    ):
        raise SAICRolloutPreferenceError("rollout exact81/exact40/J contract differs")
    if any(index in update_indices for index in EXACT40_FORBIDDEN_UPDATE_INDICES):
        raise SAICRolloutPreferenceError("exact40 indices 38/39 cannot update")

    codec = _closed(
        row["codec_reencode_receipt"], _CODEC_FIELDS, label="codec receipt"
    )
    if codec["schema_version"] != CODEC_RECEIPT_SCHEMA_VERSION:
        raise SAICRolloutPreferenceError("codec receipt schema differs")
    _sealed(codec, field="receipt_digest", label="codec receipt")
    if codec["candidate_id"] != candidate_id:
        raise SAICRolloutPreferenceError("codec receipt candidate differs")
    if codec["input_output_media_sha256"] != output_sha:
        raise SAICRolloutPreferenceError("codec input is not the rollout output")
    if codec["reencoded_latent_sha256"] != endpoint_sha:
        raise SAICRolloutPreferenceError("endpoint is not the codec-reencoded latent")
    for key in (
        "decoded_rgb24_sha256",
        "codec_bitstream_sha256",
        "codec_decoded_rgb24_sha256",
        "vae_weights_sha256",
    ):
        _sha(codec[key], label=f"codec receipt {key}")
    _safe_id(codec["codec_name"], label="codec_name")
    _safe_id(codec["vae_id"], label="vae_id")
    for key, expected in (
        ("frame_count", FRAME_COUNT),
        ("fps_numerator", FPS_NUMERATOR),
        ("fps_denominator", FPS_DENOMINATOR),
    ):
        _exact_integer(codec[key], expected, label=f"codec receipt {key}")
    if _boolean(codec["endpoint_detached"], label="endpoint_detached") is not True:
        raise SAICRolloutPreferenceError("codec exact81/detached contract differs")
    _sealed(row, field="candidate_digest", label="candidate")
    return json.loads(canonical_json_bytes(row).decode("ascii"))


def _policy_axes(value: Any, *, label: str) -> dict[str, float]:
    return _axis_map(value, label=label)


def _pair_receipt(
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
    *,
    event_gain: float,
    floors: Mapping[str, float],
    margins: Mapping[str, float],
) -> dict[str, Any]:
    body = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "arm": chosen["arm"],
        "source_id": chosen["source_id"],
        "instruction_id": chosen["instruction_id"],
        "policy_sha256": chosen["policy_sha256"],
        "chosen_candidate_id": chosen["candidate_id"],
        "chosen_candidate_digest": chosen["candidate_digest"],
        "rejected_candidate_id": rejected["candidate_id"],
        "rejected_candidate_digest": rejected["candidate_digest"],
        "event_relative_gain": event_gain,
        "axis_floors": dict(floors),
        "axis_noninferiority_margins": dict(margins),
        "axis_chosen_minus_rejected": {
            axis: float(chosen["axis_scores"][axis])
            - float(rejected["axis_scores"][axis])
            for axis in PRESERVATION_AXES
        },
        "hard_order": PAIR_ORDER,
        "scalar_reward_or_weighted_compensation_used": False,
        "pure_t2v_endpoint_used": False,
        "preference_update_indices": list(EXACT40_UPDATE_INDICES),
    }
    return {**body, "pair_digest": object_sha256(body)}


@dataclass(frozen=True)
class SAICPreferenceSet:
    optimizer_step_allowed: bool
    authorized_pairs: tuple[Mapping[str, Any], ...]
    diagnostic_admissible_pair_count_by_arm: Mapping[str, int]
    zero_update_reason: str | None
    receipt: Mapping[str, Any]


def build_preference_set(
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_event_gain: float,
    axis_floors: Mapping[str, float],
    axis_noninferiority_margins: Mapping[str, float] | None = None,
) -> SAICPreferenceSet:
    """Select one deterministic hard-Pareto pair per arm or authorize zero.

    Candidates may be compared only within one arm/source/instruction/policy
    cell.  Both endpoints pass all seven floors.  No axis can compensate for
    another axis or for insufficient event gain.
    """

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise SAICRolloutPreferenceError("candidates must be a sequence")
    rows = tuple(validate_candidate(row) for row in candidates)
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise SAICRolloutPreferenceError("candidate IDs must be unique")
    if len({row["candidate_digest"] for row in rows}) != len(rows):
        raise SAICRolloutPreferenceError("candidate digests must be unique")
    gain_floor = _unit_score(minimum_event_gain, label="minimum_event_gain")
    if gain_floor <= 0.0:
        raise SAICRolloutPreferenceError("minimum_event_gain must be positive")
    floors = _policy_axes(axis_floors, label="axis_floors")
    margins = (
        {axis: 0.0 for axis in PRESERVATION_AXES}
        if axis_noninferiority_margins is None
        else _policy_axes(
            axis_noninferiority_margins, label="axis_noninferiority_margins"
        )
    )

    admissible: dict[str, list[tuple[dict[str, Any], dict[str, Any], float]]] = {
        arm: [] for arm in ARMS
    }
    for chosen in rows:
        for rejected in rows:
            if chosen["candidate_id"] == rejected["candidate_id"]:
                continue
            if any(
                chosen[key] != rejected[key]
                for key in ("arm", "source_id", "instruction_id", "policy_sha256")
            ):
                continue
            event_gain = float(chosen["event_score"]) - float(rejected["event_score"])
            if event_gain < gain_floor:
                continue
            if any(
                float(endpoint["axis_scores"][axis]) < floors[axis]
                for endpoint in (chosen, rejected)
                for axis in PRESERVATION_AXES
            ):
                continue
            if any(
                float(chosen["axis_scores"][axis]) + margins[axis]
                < float(rejected["axis_scores"][axis])
                for axis in PRESERVATION_AXES
            ):
                continue
            admissible[chosen["arm"]].append((chosen, rejected, event_gain))

    counts = {arm: len(admissible[arm]) for arm in ARMS}
    complete = all(counts[arm] > 0 for arm in ARMS)
    authorized: list[dict[str, Any]] = []
    if complete:
        for arm in ARMS:
            # Largest event separation, then stable endpoint IDs.
            chosen, rejected, event_gain = sorted(
                admissible[arm],
                key=lambda item: (
                    -item[2],
                    item[0]["candidate_id"],
                    item[1]["candidate_id"],
                ),
            )[0]
            authorized.append(
                _pair_receipt(
                    chosen,
                    rejected,
                    event_gain=event_gain,
                    floors=floors,
                    margins=margins,
                )
            )
    reason = None if complete else "missing_admissible_pair_in_one_or_more_required_arms"
    body = {
        "schema_version": PREFERENCE_SET_SCHEMA_VERSION,
        "candidate_count": len(rows),
        "candidate_digests": sorted(row["candidate_digest"] for row in rows),
        "required_arms": list(ARMS),
        "minimum_event_gain": gain_floor,
        "axis_floors": floors,
        "axis_noninferiority_margins": margins,
        "hard_order": PAIR_ORDER,
        "scalar_reward_or_weighted_compensation_used": False,
        "pure_t2v_role": PURE_T2V_ROLE,
        "pure_t2v_endpoint_used": False,
        "preference_update_indices": list(EXACT40_UPDATE_INDICES),
        "forbidden_update_indices": list(EXACT40_FORBIDDEN_UPDATE_INDICES),
        "diagnostic_admissible_pair_count_by_arm": counts,
        "authorized_pair_digests": [pair["pair_digest"] for pair in authorized],
        "optimizer_step_allowed": complete,
        "zero_update_reason": reason,
    }
    receipt = {**body, "preference_set_digest": object_sha256(body)}
    return SAICPreferenceSet(
        optimizer_step_allowed=complete,
        authorized_pairs=tuple(authorized),
        diagnostic_admissible_pair_count_by_arm=counts,
        zero_update_reason=reason,
        receipt=receipt,
    )


def validate_update_index(index: Any) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise SAICRolloutPreferenceError("exact40 update index must be an integer")
    if index in EXACT40_FORBIDDEN_UPDATE_INDICES:
        raise SAICRolloutPreferenceError("exact40 indices 38/39 are zero-update")
    if index not in EXACT40_UPDATE_INDICES:
        raise SAICRolloutPreferenceError("index is outside registered Stage-B J")
    return index


__all__ = [
    "ARMS",
    "CODEC_RECEIPT_SCHEMA_VERSION",
    "ENDPOINT_ROLE",
    "EXACT40_FORBIDDEN_UPDATE_INDICES",
    "EXACT40_UPDATE_INDICES",
    "FRAME_COUNT",
    "GENERATION_MODE",
    "PAIR_SCHEMA_VERSION",
    "PREFERENCE_SET_SCHEMA_VERSION",
    "PRESERVATION_AXES",
    "PURE_T2V_ROLE",
    "ROLLOUT_RECEIPT_SCHEMA_VERSION",
    "SAICPreferenceSet",
    "SAICRolloutPreferenceError",
    "SCHEMA_VERSION",
    "build_preference_set",
    "canonical_json_bytes",
    "object_sha256",
    "validate_candidate",
    "validate_update_index",
]
