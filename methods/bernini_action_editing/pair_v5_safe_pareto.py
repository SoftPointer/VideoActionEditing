"""Fail-closed PAIR-v5 safe-Pareto preference-pair selection.

This module is deliberately a small, pure-Python trust boundary between
native Bernini RV2V rollouts and a later preference trainer.  A candidate can
carry only its opaque ID, four calibrated scalar scores, the complete hard-
negative flag registry, the sealed evaluator-packet/rollout commitments, and
its canonical digest.  In particular, there is no
API or schema slot for a proposal/donor video, latent, noise, source, target,
mask, flow, pose, track, or trajectory.

Self-generated T2V action proposals may influence *offline calibration* of the
action scorer.  PAIR-v5 binds that fact through an opaque calibrator receipt
hash only.  Proposal media and tensors can never enter candidate comparison.

The selector has two monotone stages:

``safe_pareto_bootstrap``
    A winner must improve action by at least the registered delta, may degrade
    identity/consistency/quality by at most their registered epsilons, and may
    neither endpoint may carry any hard-negative flag.

``strict_feasible_only``
    As soon as *any* observed candidate passes the absolute A/I/C/Q thresholds
    and every hard-negative check, this stage is entered in that same event.
    It can never be left.  The relative safe-Pareto rules remain active and a
    selected winner must additionally be absolutely feasible.  A loser may be
    an infeasible near miss; ``strict-feasible`` describes the preference
    winner, which is the sample optimized by DPO.  The loser is a clean
    score-level near miss, never a wrong-actor/reverse/incomplete/blur/camera/
    appearance hard negative.

Every artifact uses a closed schema and an embedded SHA-256 over canonical
JSON.  Persistent callers must pin the state digest returned by the preceding
receipt.  :func:`validate_state_transition` additionally rejects a strict to
bootstrap transition even if an attacker recomputes JSON digests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from typing import Any


CANDIDATE_SCHEMA = "bernini-pair-v5-native-rv2v-candidate-v2"
POLICY_SCHEMA = "bernini-pair-v5-safe-pareto-policy-v1"
CALIBRATOR_PROVENANCE_SCHEMA = (
    "bernini-pair-v5-action-calibrator-provenance-v1"
)
STATE_SCHEMA = "bernini-pair-v5-selector-state-v1"
PAIR_SCHEMA = "bernini-pair-v5-selected-preference-pair-v1"
EVENT_SCHEMA = "bernini-pair-v5-selector-event-v1"
RECEIPT_SCHEMA = "bernini-pair-v5-selection-receipt-v1"
CANDIDATE_SET_SCHEMA = "bernini-pair-v5-candidate-set-commitment-v1"

BOOTSTRAP_STAGE = "safe_pareto_bootstrap"
STRICT_STAGE = "strict_feasible_only"
STAGES = (BOOTSTRAP_STAGE, STRICT_STAGE)

SCORE_AXES = ("action", "identity", "consistency", "quality")
PRESERVATION_AXES = ("identity", "consistency", "quality")
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

ACTION_SCORE_SEMANTICS = (
    "calibrated_unit_interval_candidate_own_coordinate_action_energy"
)
CALIBRATOR_ROLE = "offline_action_calibrator_only"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")

_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "action_score",
        "identity_score",
        "consistency_score",
        "quality_score",
        "hard_negative_flags",
        "evaluator_packet_digest",
        "rollout_receipt_digest",
        "candidate_digest",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "policy_id",
        "bootstrap_action_delta",
        "max_preservation_degradation",
        "absolute_thresholds",
        "hard_negative_flag_order",
        "policy_digest",
    }
)
_CALIBRATOR_FIELDS = frozenset(
    {
        "schema_version",
        "calibrator_id",
        "action_evaluator_sha256",
        "calibration_receipt_sha256",
        "calibration_receipt_digest",
        "action_score_semantics",
        "calibrator_role",
        "hard_negative_flag_order",
        "provenance_digest",
    }
)
_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "policy_digest",
        "stage",
        "revision",
        "strict_since_revision",
        "last_event_digest",
        "state_digest",
    }
)
_PAIR_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "winner_candidate_id",
        "winner_candidate_digest",
        "loser_candidate_id",
        "loser_candidate_digest",
        "action_improvement",
        "identity_degradation",
        "consistency_degradation",
        "quality_degradation",
        "winner_absolute_feasible",
        "winner_hard_negative_flags_all_false",
        "loser_hard_negative_flags_all_false",
        "safe_pareto_pass",
        "strict_feasible_pass",
        "pair_digest",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "policy_digest",
        "calibrator_provenance_digest",
        "state_before_digest",
        "candidate_set_digest",
        "candidate_count",
        "candidate_ids",
        "absolute_feasible_candidate_ids",
        "stage_before",
        "stage_after",
        "transitioned_to_strict",
        "eligible_pair_count",
        "selected_pair",
        "decision",
        "condition_closure",
        "event_digest",
        "next_state",
        "receipt_digest",
    }
)
_CONDITION_CLOSURE_FIELDS = frozenset(
    {
        "candidate_input_fields",
        "calibrator_provenance_fields",
        "selector_consumes_scores_only",
        "proposal_video_consumed",
        "proposal_latent_consumed",
        "proposal_noise_consumed",
        "donor_consumed",
        "source_or_target_media_consumed",
        "mask_flow_pose_track_trajectory_consumed",
        "both_pair_endpoints_hard_negative_free",
        "relative_safe_pareto_constraints_retained_in_strict",
        "strict_feasible_only_means_winner_eligibility",
    }
)


class PairV5ContractError(ValueError):
    """A PAIR-v5 artifact or state transition violates the closed contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode canonical UTF-8 JSON or fail on non-JSON/non-finite values."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PairV5ContractError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PairV5ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise PairV5ContractError(f"non-finite JSON number: {value}")


def parse_canonical_json_bytes(payload: Any) -> Any:
    """Parse only exact canonical JSON bytes, rejecting duplicate keys."""

    if type(payload) is not bytes:
        raise PairV5ContractError("canonical JSON payload must be exact bytes")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except PairV5ContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairV5ContractError("invalid UTF-8 JSON payload") from error
    if canonical_json_bytes(value) != payload:
        raise PairV5ContractError("JSON payload is not in canonical form")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV5ContractError(f"{label} must be a mapping")
    return value


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    row = _mapping(value, label=label)
    keys = set(row.keys())
    if not all(isinstance(key, str) for key in keys):
        raise PairV5ContractError(f"{label} keys must all be strings")
    missing = sorted(fields - keys)
    extra = sorted(keys - fields)
    if missing or extra:
        raise PairV5ContractError(
            f"{label} schema closure differs; missing={missing}, extra={extra}"
        )
    return row


def _schema(value: Any, expected: str, *, label: str) -> str:
    if value != expected:
        raise PairV5ContractError(f"{label} schema_version must be {expected!r}")
    return expected


def _slug(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SLUG_RE.fullmatch(value) is None:
        raise PairV5ContractError(f"{label} must be a canonical slug")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5ContractError(f"{label} must be lowercase SHA-256")
    return value


def _optional_sha256(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, label=label)


def _boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise PairV5ContractError(f"{label} must be boolean")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PairV5ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _unit_float(value: Any, *, label: str) -> float:
    # Exact floats avoid the canonical ambiguity between JSON ``1`` and
    # ``1.0`` and prevent booleans from silently becoming scores.
    if type(value) is not float or not math.isfinite(value):
        raise PairV5ContractError(f"{label} must be a finite JSON float")
    if value < 0.0 or value > 1.0:
        raise PairV5ContractError(f"{label} must remain in [0, 1]")
    return value


def _exact_string_list(value: Any, expected: Sequence[str], *, label: str) -> list[str]:
    if not isinstance(value, list) or value != list(expected):
        raise PairV5ContractError(f"{label} must equal the closed registered order")
    return list(value)


def _validate_embedded_digest(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    declared = _sha256(value.get(field), label=f"{label} {field}")
    unsigned = dict(value)
    unsigned.pop(field)
    if object_sha256(unsigned) != declared:
        raise PairV5ContractError(f"{label} embedded digest mismatch")
    return declared


def _seal(unsigned: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    value = dict(unsigned)
    if field in value:
        raise PairV5ContractError(f"unsigned artifact already contains {field}")
    value[field] = object_sha256(value)
    return value


def make_candidate(
    candidate_id: str,
    *,
    action_score: float,
    identity_score: float,
    consistency_score: float,
    quality_score: float,
    hard_negative_flags: Mapping[str, bool],
    evaluator_packet_digest: str,
    rollout_receipt_digest: str,
) -> dict[str, Any]:
    """Create one score record bound to evaluator evidence and native rollout."""

    unsigned = {
        "schema_version": CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "action_score": action_score,
        "identity_score": identity_score,
        "consistency_score": consistency_score,
        "quality_score": quality_score,
        "hard_negative_flags": dict(hard_negative_flags),
        "evaluator_packet_digest": evaluator_packet_digest,
        "rollout_receipt_digest": rollout_receipt_digest,
    }
    value = _seal(unsigned, field="candidate_digest")
    return validate_candidate(value)


def validate_candidate(value: Any) -> dict[str, Any]:
    row = _closed(value, _CANDIDATE_FIELDS, label="candidate")
    _schema(row["schema_version"], CANDIDATE_SCHEMA, label="candidate")
    candidate_id = _slug(row["candidate_id"], label="candidate_id")
    scores = {
        f"{axis}_score": _unit_float(
            row[f"{axis}_score"], label=f"candidate {axis}_score"
        )
        for axis in SCORE_AXES
    }
    flags = _closed(
        row["hard_negative_flags"],
        frozenset(HARD_NEGATIVE_FLAGS),
        label="candidate hard_negative_flags",
    )
    checked_flags = {
        name: _boolean(flags[name], label=f"hard-negative flag {name}")
        for name in HARD_NEGATIVE_FLAGS
    }
    digest = _validate_embedded_digest(
        row, field="candidate_digest", label="candidate"
    )
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        **scores,
        "hard_negative_flags": checked_flags,
        "evaluator_packet_digest": _sha256(
            row["evaluator_packet_digest"], label="candidate evaluator_packet_digest"
        ),
        "rollout_receipt_digest": _sha256(
            row["rollout_receipt_digest"], label="candidate rollout_receipt_digest"
        ),
        "candidate_digest": digest,
    }


def make_policy(
    policy_id: str,
    *,
    bootstrap_action_delta: float,
    max_identity_degradation: float,
    max_consistency_degradation: float,
    max_quality_degradation: float,
    min_action_score: float,
    min_identity_score: float,
    min_consistency_score: float,
    min_quality_score: float,
) -> dict[str, Any]:
    """Create the immutable relative and absolute selection policy."""

    unsigned = {
        "schema_version": POLICY_SCHEMA,
        "policy_id": policy_id,
        "bootstrap_action_delta": bootstrap_action_delta,
        "max_preservation_degradation": {
            "identity": max_identity_degradation,
            "consistency": max_consistency_degradation,
            "quality": max_quality_degradation,
        },
        "absolute_thresholds": {
            "action": min_action_score,
            "identity": min_identity_score,
            "consistency": min_consistency_score,
            "quality": min_quality_score,
        },
        "hard_negative_flag_order": list(HARD_NEGATIVE_FLAGS),
    }
    value = _seal(unsigned, field="policy_digest")
    return validate_policy(value)


def validate_policy(value: Any) -> dict[str, Any]:
    row = _closed(value, _POLICY_FIELDS, label="policy")
    _schema(row["schema_version"], POLICY_SCHEMA, label="policy")
    policy_id = _slug(row["policy_id"], label="policy_id")
    delta = _unit_float(
        row["bootstrap_action_delta"], label="bootstrap_action_delta"
    )
    if delta <= 0.0:
        raise PairV5ContractError("bootstrap_action_delta must be > 0")
    degradation = _closed(
        row["max_preservation_degradation"],
        frozenset(PRESERVATION_AXES),
        label="max_preservation_degradation",
    )
    checked_degradation = {
        axis: _unit_float(
            degradation[axis], label=f"max {axis} degradation"
        )
        for axis in PRESERVATION_AXES
    }
    thresholds = _closed(
        row["absolute_thresholds"],
        frozenset(SCORE_AXES),
        label="absolute_thresholds",
    )
    checked_thresholds = {
        axis: _unit_float(thresholds[axis], label=f"absolute {axis} threshold")
        for axis in SCORE_AXES
    }
    flag_order = _exact_string_list(
        row["hard_negative_flag_order"],
        HARD_NEGATIVE_FLAGS,
        label="policy hard_negative_flag_order",
    )
    digest = _validate_embedded_digest(row, field="policy_digest", label="policy")
    return {
        "schema_version": POLICY_SCHEMA,
        "policy_id": policy_id,
        "bootstrap_action_delta": delta,
        "max_preservation_degradation": checked_degradation,
        "absolute_thresholds": checked_thresholds,
        "hard_negative_flag_order": flag_order,
        "policy_digest": digest,
    }


def make_calibrator_provenance(
    calibrator_id: str,
    *,
    action_evaluator_sha256: str,
    calibration_receipt_sha256: str,
    calibration_receipt_digest: str,
) -> dict[str, Any]:
    """Bind an action calibrator without admitting its proposal artifacts."""

    unsigned = {
        "schema_version": CALIBRATOR_PROVENANCE_SCHEMA,
        "calibrator_id": calibrator_id,
        "action_evaluator_sha256": action_evaluator_sha256,
        "calibration_receipt_sha256": calibration_receipt_sha256,
        "calibration_receipt_digest": calibration_receipt_digest,
        "action_score_semantics": ACTION_SCORE_SEMANTICS,
        "calibrator_role": CALIBRATOR_ROLE,
        "hard_negative_flag_order": list(HARD_NEGATIVE_FLAGS),
    }
    value = _seal(unsigned, field="provenance_digest")
    return validate_calibrator_provenance(value)


def validate_calibrator_provenance(value: Any) -> dict[str, Any]:
    row = _closed(value, _CALIBRATOR_FIELDS, label="calibrator provenance")
    _schema(
        row["schema_version"],
        CALIBRATOR_PROVENANCE_SCHEMA,
        label="calibrator provenance",
    )
    calibrator_id = _slug(row["calibrator_id"], label="calibrator_id")
    evaluator = _sha256(
        row["action_evaluator_sha256"], label="action evaluator SHA-256"
    )
    receipt_file = _sha256(
        row["calibration_receipt_sha256"], label="calibration receipt SHA-256"
    )
    receipt_digest = _sha256(
        row["calibration_receipt_digest"], label="calibration receipt digest"
    )
    if row["action_score_semantics"] != ACTION_SCORE_SEMANTICS:
        raise PairV5ContractError("action_score_semantics is not the fixed contract")
    if row["calibrator_role"] != CALIBRATOR_ROLE:
        raise PairV5ContractError("calibrator_role is not offline-only")
    flag_order = _exact_string_list(
        row["hard_negative_flag_order"],
        HARD_NEGATIVE_FLAGS,
        label="calibrator hard_negative_flag_order",
    )
    digest = _validate_embedded_digest(
        row, field="provenance_digest", label="calibrator provenance"
    )
    return {
        "schema_version": CALIBRATOR_PROVENANCE_SCHEMA,
        "calibrator_id": calibrator_id,
        "action_evaluator_sha256": evaluator,
        "calibration_receipt_sha256": receipt_file,
        "calibration_receipt_digest": receipt_digest,
        "action_score_semantics": ACTION_SCORE_SEMANTICS,
        "calibrator_role": CALIBRATOR_ROLE,
        "hard_negative_flag_order": flag_order,
        "provenance_digest": digest,
    }


def initial_state(policy: Any) -> dict[str, Any]:
    checked_policy = validate_policy(policy)
    return _seal(
        {
            "schema_version": STATE_SCHEMA,
            "policy_digest": checked_policy["policy_digest"],
            "stage": BOOTSTRAP_STAGE,
            "revision": 0,
            "strict_since_revision": None,
            "last_event_digest": None,
        },
        field="state_digest",
    )


def validate_state(value: Any, policy: Any) -> dict[str, Any]:
    checked_policy = validate_policy(policy)
    row = _closed(value, _STATE_FIELDS, label="selector state")
    _schema(row["schema_version"], STATE_SCHEMA, label="selector state")
    policy_digest = _sha256(row["policy_digest"], label="state policy_digest")
    if policy_digest != checked_policy["policy_digest"]:
        raise PairV5ContractError("selector state is bound to a different policy")
    stage = row["stage"]
    if stage not in STAGES:
        raise PairV5ContractError("selector state has an unknown stage")
    revision = _integer(row["revision"], label="state revision")
    strict_since = row["strict_since_revision"]
    if stage == BOOTSTRAP_STAGE:
        if strict_since is not None:
            raise PairV5ContractError(
                "bootstrap state cannot have strict_since_revision"
            )
    else:
        strict_since = _integer(
            strict_since, label="strict_since_revision", minimum=1
        )
        if strict_since > revision:
            raise PairV5ContractError(
                "strict_since_revision cannot exceed state revision"
            )
    last_event = _optional_sha256(
        row["last_event_digest"], label="state last_event_digest"
    )
    if revision == 0 and last_event is not None:
        raise PairV5ContractError("initial state cannot have a last event")
    if revision > 0 and last_event is None:
        raise PairV5ContractError("advanced state must bind its last event")
    digest = _validate_embedded_digest(
        row, field="state_digest", label="selector state"
    )
    return {
        "schema_version": STATE_SCHEMA,
        "policy_digest": policy_digest,
        "stage": stage,
        "revision": revision,
        "strict_since_revision": strict_since,
        "last_event_digest": last_event,
        "state_digest": digest,
    }


def validate_state_transition(previous: Any, following: Any, policy: Any) -> None:
    """Validate only monotone state facts shared by every legal event."""

    before = validate_state(previous, policy)
    after = validate_state(following, policy)
    if after["revision"] != before["revision"] + 1:
        raise PairV5ContractError("state revision must advance by exactly one")
    if before["stage"] == STRICT_STAGE and after["stage"] != STRICT_STAGE:
        raise PairV5ContractError("strict-feasible-only state is irreversible")
    if before["stage"] == STRICT_STAGE:
        if after["strict_since_revision"] != before["strict_since_revision"]:
            raise PairV5ContractError("strict transition revision cannot change")
    elif after["stage"] == BOOTSTRAP_STAGE:
        if after["strict_since_revision"] is not None:
            raise PairV5ContractError("bootstrap stage has a strict marker")
    elif after["strict_since_revision"] != after["revision"]:
        raise PairV5ContractError("first strict marker must equal transition revision")


def _all_hard_negative_flags_false(candidate: Mapping[str, Any]) -> bool:
    return not any(candidate["hard_negative_flags"][name] for name in HARD_NEGATIVE_FLAGS)


def is_absolute_feasible(candidate: Any, policy: Any) -> bool:
    row = validate_candidate(candidate)
    checked_policy = validate_policy(policy)
    return _absolute_feasible_checked(row, checked_policy)


def _absolute_feasible_checked(
    candidate: Mapping[str, Any], policy: Mapping[str, Any]
) -> bool:
    if not _all_hard_negative_flags_false(candidate):
        return False
    return all(
        candidate[f"{axis}_score"] >= policy["absolute_thresholds"][axis]
        for axis in SCORE_AXES
    )


def _candidate_set(
    candidates: Any,
) -> tuple[list[dict[str, Any]], str]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise PairV5ContractError("candidates must be a sequence")
    if len(candidates) < 2:
        raise PairV5ContractError("at least two candidates are required")
    checked = [validate_candidate(candidate) for candidate in candidates]
    checked.sort(key=lambda item: item["candidate_id"])
    ids = [item["candidate_id"] for item in checked]
    if len(ids) != len(set(ids)):
        raise PairV5ContractError("candidate_id values must be unique")
    commitment = {
        "schema_version": CANDIDATE_SET_SCHEMA,
        "candidates": checked,
    }
    return checked, object_sha256(commitment)


def _pair_metrics(
    winner: Mapping[str, Any], loser: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "action_improvement": winner["action_score"] - loser["action_score"],
        **{
            f"{axis}_degradation": (
                loser[f"{axis}_score"] - winner[f"{axis}_score"]
            )
            for axis in PRESERVATION_AXES
        },
    }


def _safe_pareto_pass(
    winner: Mapping[str, Any],
    loser: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[bool, dict[str, float]]:
    metrics = _pair_metrics(winner, loser)
    passed = (
        _all_hard_negative_flags_false(winner)
        and _all_hard_negative_flags_false(loser)
        and metrics["action_improvement"] >= policy["bootstrap_action_delta"]
        and all(
            metrics[f"{axis}_degradation"]
            <= policy["max_preservation_degradation"][axis]
            for axis in PRESERVATION_AXES
        )
    )
    return passed, metrics


def _selected_pair(
    winner: Mapping[str, Any],
    loser: Mapping[str, Any],
    *,
    stage: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    safe, metrics = _safe_pareto_pass(winner, loser, policy)
    absolute = _absolute_feasible_checked(winner, policy)
    strict = safe and absolute
    unsigned = {
        "schema_version": PAIR_SCHEMA,
        "stage": stage,
        "winner_candidate_id": winner["candidate_id"],
        "winner_candidate_digest": winner["candidate_digest"],
        "loser_candidate_id": loser["candidate_id"],
        "loser_candidate_digest": loser["candidate_digest"],
        **metrics,
        "winner_absolute_feasible": absolute,
        "winner_hard_negative_flags_all_false": _all_hard_negative_flags_false(
            winner
        ),
        "loser_hard_negative_flags_all_false": _all_hard_negative_flags_false(
            loser
        ),
        "safe_pareto_pass": safe,
        "strict_feasible_pass": strict,
    }
    return _seal(unsigned, field="pair_digest")


def _eligible_pairs(
    candidates: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    policy: Mapping[str, Any],
) -> list[tuple[tuple[Any, ...], Mapping[str, Any], Mapping[str, Any]]]:
    eligible = []
    for winner in candidates:
        for loser in candidates:
            if winner["candidate_id"] == loser["candidate_id"]:
                continue
            safe, metrics = _safe_pareto_pass(winner, loser, policy)
            if not safe:
                continue
            absolute = _absolute_feasible_checked(winner, policy)
            if stage == STRICT_STAGE and not absolute:
                continue
            # Prefer a decisive action margin, then the smallest worst
            # preservation degradation, then robust preservation and stable
            # lexical IDs.  The key depends only on values, never input order.
            worst_degradation = max(
                metrics[f"{axis}_degradation"] for axis in PRESERVATION_AXES
            )
            weakest_preservation = min(
                winner[f"{axis}_score"] for axis in PRESERVATION_AXES
            )
            key = (
                -metrics["action_improvement"],
                worst_degradation,
                -weakest_preservation,
                winner["candidate_id"],
                loser["candidate_id"],
            )
            eligible.append((key, winner, loser))
    eligible.sort(key=lambda item: item[0])
    return eligible


def _condition_closure() -> dict[str, Any]:
    return {
        "candidate_input_fields": sorted(_CANDIDATE_FIELDS),
        "calibrator_provenance_fields": sorted(_CALIBRATOR_FIELDS),
        "selector_consumes_scores_only": True,
        "proposal_video_consumed": False,
        "proposal_latent_consumed": False,
        "proposal_noise_consumed": False,
        "donor_consumed": False,
        "source_or_target_media_consumed": False,
        "mask_flow_pose_track_trajectory_consumed": False,
        "both_pair_endpoints_hard_negative_free": True,
        "relative_safe_pareto_constraints_retained_in_strict": True,
        "strict_feasible_only_means_winner_eligibility": True,
    }


def advance_pair_selector(
    *,
    state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    calibrator_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Advance once and emit a replay-verifiable PAIR-v5 receipt.

    The stage decision is made before pair enumeration.  Thus the first event
    containing an absolute feasible candidate is already strict-feasible-only.
    """

    checked_policy = validate_policy(policy)
    checked_state = validate_state(state, checked_policy)
    provenance = validate_calibrator_provenance(calibrator_provenance)
    checked_candidates, candidate_set_digest = _candidate_set(candidates)

    feasible_ids = sorted(
        candidate["candidate_id"]
        for candidate in checked_candidates
        if _absolute_feasible_checked(candidate, checked_policy)
    )
    stage_before = checked_state["stage"]
    transitioned = stage_before == BOOTSTRAP_STAGE and bool(feasible_ids)
    stage_after = (
        STRICT_STAGE
        if (stage_before == STRICT_STAGE or transitioned)
        else BOOTSTRAP_STAGE
    )
    revision = checked_state["revision"] + 1

    eligible = _eligible_pairs(
        checked_candidates, stage=stage_after, policy=checked_policy
    )
    selected = None
    if eligible:
        _, winner, loser = eligible[0]
        selected = _selected_pair(
            winner, loser, stage=stage_after, policy=checked_policy
        )

    if selected is None:
        decision = "no_eligible_pair"
    elif stage_after == STRICT_STAGE:
        decision = "selected_strict_feasible_pair"
    else:
        decision = "selected_safe_pareto_bootstrap_pair"

    event_body = {
        "schema_version": EVENT_SCHEMA,
        "policy_digest": checked_policy["policy_digest"],
        "calibrator_provenance_digest": provenance["provenance_digest"],
        "state_before_digest": checked_state["state_digest"],
        "candidate_set_digest": candidate_set_digest,
        "stage_before": stage_before,
        "stage_after": stage_after,
        "transitioned_to_strict": transitioned,
        "absolute_feasible_candidate_ids": feasible_ids,
        "selected_pair_digest": (
            None if selected is None else selected["pair_digest"]
        ),
        "eligible_pair_count": len(eligible),
    }
    event_digest = object_sha256(event_body)
    strict_since = checked_state["strict_since_revision"]
    if transitioned:
        strict_since = revision
    next_state = _seal(
        {
            "schema_version": STATE_SCHEMA,
            "policy_digest": checked_policy["policy_digest"],
            "stage": stage_after,
            "revision": revision,
            "strict_since_revision": strict_since,
            "last_event_digest": event_digest,
        },
        field="state_digest",
    )
    validate_state_transition(checked_state, next_state, checked_policy)

    receipt = _seal(
        {
            "schema_version": RECEIPT_SCHEMA,
            "policy_digest": checked_policy["policy_digest"],
            "calibrator_provenance_digest": provenance["provenance_digest"],
            "state_before_digest": checked_state["state_digest"],
            "candidate_set_digest": candidate_set_digest,
            "candidate_count": len(checked_candidates),
            "candidate_ids": [
                candidate["candidate_id"] for candidate in checked_candidates
            ],
            "absolute_feasible_candidate_ids": feasible_ids,
            "stage_before": stage_before,
            "stage_after": stage_after,
            "transitioned_to_strict": transitioned,
            "eligible_pair_count": len(eligible),
            "selected_pair": selected,
            "decision": decision,
            "condition_closure": _condition_closure(),
            "event_digest": event_digest,
            "next_state": next_state,
        },
        field="receipt_digest",
    )
    return validate_selection_receipt(receipt, checked_policy)


def _validate_selected_pair(value: Any, *, expected_stage: str) -> dict[str, Any]:
    row = _closed(value, _PAIR_FIELDS, label="selected pair")
    _schema(row["schema_version"], PAIR_SCHEMA, label="selected pair")
    if row["stage"] != expected_stage:
        raise PairV5ContractError("selected pair stage differs from receipt stage")
    winner_id = _slug(row["winner_candidate_id"], label="winner candidate_id")
    loser_id = _slug(row["loser_candidate_id"], label="loser candidate_id")
    if winner_id == loser_id:
        raise PairV5ContractError("winner and loser candidate IDs must differ")
    winner_digest = _sha256(
        row["winner_candidate_digest"], label="winner candidate digest"
    )
    loser_digest = _sha256(
        row["loser_candidate_digest"], label="loser candidate digest"
    )
    metrics = {}
    for field in (
        "action_improvement",
        "identity_degradation",
        "consistency_degradation",
        "quality_degradation",
    ):
        value_float = row[field]
        if type(value_float) is not float or not math.isfinite(value_float):
            raise PairV5ContractError(f"selected pair {field} must be finite float")
        if value_float < -1.0 or value_float > 1.0:
            raise PairV5ContractError(f"selected pair {field} must remain in [-1, 1]")
        metrics[field] = value_float
    hard_clear = _boolean(
        row["winner_hard_negative_flags_all_false"],
        label="winner hard-negative closure",
    )
    loser_hard_clear = _boolean(
        row["loser_hard_negative_flags_all_false"],
        label="loser hard-negative closure",
    )
    safe = _boolean(row["safe_pareto_pass"], label="safe_pareto_pass")
    absolute = _boolean(
        row["winner_absolute_feasible"], label="winner_absolute_feasible"
    )
    strict = _boolean(row["strict_feasible_pass"], label="strict_feasible_pass")
    if not hard_clear or not loser_hard_clear or not safe:
        raise PairV5ContractError(
            "both pair endpoints must be hard-negative-free and safe-Pareto"
        )
    if strict != (safe and absolute):
        raise PairV5ContractError("strict_feasible_pass is inconsistent")
    if expected_stage == STRICT_STAGE and not strict:
        raise PairV5ContractError("strict stage selected a non-feasible winner")
    digest = _validate_embedded_digest(
        row, field="pair_digest", label="selected pair"
    )
    return {
        "schema_version": PAIR_SCHEMA,
        "stage": expected_stage,
        "winner_candidate_id": winner_id,
        "winner_candidate_digest": winner_digest,
        "loser_candidate_id": loser_id,
        "loser_candidate_digest": loser_digest,
        **metrics,
        "winner_absolute_feasible": absolute,
        "winner_hard_negative_flags_all_false": hard_clear,
        "loser_hard_negative_flags_all_false": loser_hard_clear,
        "safe_pareto_pass": safe,
        "strict_feasible_pass": strict,
        "pair_digest": digest,
    }


def validate_selection_receipt(value: Any, policy: Any) -> dict[str, Any]:
    """Validate a receipt's closed form; use replay for score-level proof."""

    checked_policy = validate_policy(policy)
    row = _closed(value, _RECEIPT_FIELDS, label="selection receipt")
    _schema(row["schema_version"], RECEIPT_SCHEMA, label="selection receipt")
    policy_digest = _sha256(
        row["policy_digest"], label="receipt policy_digest"
    )
    if policy_digest != checked_policy["policy_digest"]:
        raise PairV5ContractError("selection receipt binds a different policy")
    provenance_digest = _sha256(
        row["calibrator_provenance_digest"],
        label="receipt calibrator provenance digest",
    )
    state_before_digest = _sha256(
        row["state_before_digest"], label="receipt state_before_digest"
    )
    candidate_set_digest = _sha256(
        row["candidate_set_digest"], label="receipt candidate_set_digest"
    )
    candidate_count = _integer(
        row["candidate_count"], label="receipt candidate_count", minimum=2
    )
    candidate_ids = row["candidate_ids"]
    if not isinstance(candidate_ids, list):
        raise PairV5ContractError("receipt candidate_ids must be a list")
    checked_ids = [_slug(item, label="receipt candidate_id") for item in candidate_ids]
    if checked_ids != sorted(set(checked_ids)) or len(checked_ids) != candidate_count:
        raise PairV5ContractError("receipt candidate_ids must be sorted and unique")
    feasible_ids = row["absolute_feasible_candidate_ids"]
    if not isinstance(feasible_ids, list):
        raise PairV5ContractError(
            "absolute_feasible_candidate_ids must be a list"
        )
    checked_feasible = [
        _slug(item, label="absolute feasible candidate_id") for item in feasible_ids
    ]
    if checked_feasible != sorted(set(checked_feasible)):
        raise PairV5ContractError("absolute feasible IDs must be sorted and unique")
    if not set(checked_feasible).issubset(checked_ids):
        raise PairV5ContractError("absolute feasible IDs are outside candidate set")
    stage_before = row["stage_before"]
    stage_after = row["stage_after"]
    if stage_before not in STAGES or stage_after not in STAGES:
        raise PairV5ContractError("receipt contains an unknown stage")
    transitioned = _boolean(
        row["transitioned_to_strict"], label="transitioned_to_strict"
    )
    if stage_before == STRICT_STAGE and stage_after != STRICT_STAGE:
        raise PairV5ContractError("receipt attempts to leave strict stage")
    if (
        stage_before == BOOTSTRAP_STAGE
        and stage_after == STRICT_STAGE
        and not checked_feasible
    ):
        raise PairV5ContractError(
            "bootstrap cannot enter strict stage without absolute feasibility"
        )
    expected_transition = (
        stage_before == BOOTSTRAP_STAGE
        and stage_after == STRICT_STAGE
        and bool(checked_feasible)
    )
    if transitioned != expected_transition:
        raise PairV5ContractError("receipt strict transition marker is inconsistent")
    if (
        stage_before == BOOTSTRAP_STAGE
        and checked_feasible
        and stage_after != STRICT_STAGE
    ):
        raise PairV5ContractError("absolute feasibility must trigger strict stage")
    eligible_count = _integer(
        row["eligible_pair_count"], label="eligible_pair_count"
    )
    selected = row["selected_pair"]
    if selected is None:
        if eligible_count != 0 or row["decision"] != "no_eligible_pair":
            raise PairV5ContractError("empty selection decision is inconsistent")
        checked_selected = None
    else:
        if eligible_count < 1:
            raise PairV5ContractError("selected pair requires eligible_pair_count >= 1")
        checked_selected = _validate_selected_pair(
            selected, expected_stage=stage_after
        )
        expected_decision = (
            "selected_strict_feasible_pair"
            if stage_after == STRICT_STAGE
            else "selected_safe_pareto_bootstrap_pair"
        )
        if row["decision"] != expected_decision:
            raise PairV5ContractError("selection decision label is inconsistent")
        if checked_selected["winner_candidate_id"] not in checked_ids:
            raise PairV5ContractError("winner is outside candidate set")
        if checked_selected["loser_candidate_id"] not in checked_ids:
            raise PairV5ContractError("loser is outside candidate set")
        if (
            stage_after == STRICT_STAGE
            and checked_selected["winner_candidate_id"] not in checked_feasible
        ):
            raise PairV5ContractError("strict winner is not listed as absolute feasible")

    closure = _closed(
        row["condition_closure"],
        _CONDITION_CLOSURE_FIELDS,
        label="selection condition_closure",
    )
    if dict(closure) != _condition_closure():
        raise PairV5ContractError("selection condition closure is not exact")
    event_digest = _sha256(row["event_digest"], label="receipt event_digest")
    expected_event_digest = object_sha256(
        {
            "schema_version": EVENT_SCHEMA,
            "policy_digest": policy_digest,
            "calibrator_provenance_digest": provenance_digest,
            "state_before_digest": state_before_digest,
            "candidate_set_digest": candidate_set_digest,
            "stage_before": stage_before,
            "stage_after": stage_after,
            "transitioned_to_strict": transitioned,
            "absolute_feasible_candidate_ids": checked_feasible,
            "selected_pair_digest": (
                None
                if checked_selected is None
                else checked_selected["pair_digest"]
            ),
            "eligible_pair_count": eligible_count,
        }
    )
    if event_digest != expected_event_digest:
        raise PairV5ContractError("selection event digest mismatch")
    next_state = validate_state(row["next_state"], checked_policy)
    if next_state["stage"] != stage_after:
        raise PairV5ContractError("next state stage differs from receipt")
    if next_state["last_event_digest"] != event_digest:
        raise PairV5ContractError("next state does not bind receipt event")
    embedded = _validate_embedded_digest(
        row, field="receipt_digest", label="selection receipt"
    )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "policy_digest": policy_digest,
        "calibrator_provenance_digest": provenance_digest,
        "state_before_digest": state_before_digest,
        "candidate_set_digest": candidate_set_digest,
        "candidate_count": candidate_count,
        "candidate_ids": checked_ids,
        "absolute_feasible_candidate_ids": checked_feasible,
        "stage_before": stage_before,
        "stage_after": stage_after,
        "transitioned_to_strict": transitioned,
        "eligible_pair_count": eligible_count,
        "selected_pair": checked_selected,
        "decision": row["decision"],
        "condition_closure": dict(closure),
        "event_digest": event_digest,
        "next_state": next_state,
        "receipt_digest": embedded,
    }


def replay_and_verify_receipt(
    receipt: Any,
    *,
    state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    calibrator_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute selection and require byte-identical canonical receipt data."""

    observed = validate_selection_receipt(receipt, policy)
    expected = advance_pair_selector(
        state=state,
        candidates=candidates,
        policy=policy,
        calibrator_provenance=calibrator_provenance,
    )
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise PairV5ContractError("selection receipt does not replay exactly")
    return observed


__all__ = [
    "ACTION_SCORE_SEMANTICS",
    "BOOTSTRAP_STAGE",
    "CALIBRATOR_PROVENANCE_SCHEMA",
    "CALIBRATOR_ROLE",
    "CANDIDATE_SCHEMA",
    "HARD_NEGATIVE_FLAGS",
    "PAIR_SCHEMA",
    "POLICY_SCHEMA",
    "PairV5ContractError",
    "RECEIPT_SCHEMA",
    "SCORE_AXES",
    "STATE_SCHEMA",
    "STRICT_STAGE",
    "advance_pair_selector",
    "canonical_json_bytes",
    "initial_state",
    "is_absolute_feasible",
    "make_calibrator_provenance",
    "make_candidate",
    "make_policy",
    "object_sha256",
    "parse_canonical_json_bytes",
    "replay_and_verify_receipt",
    "validate_calibrator_provenance",
    "validate_candidate",
    "validate_policy",
    "validate_selection_receipt",
    "validate_state",
    "validate_state_transition",
]
