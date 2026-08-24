#!/usr/bin/env python3
"""Fail-closed PAIR-v5 global-action-energy calibration and event audit.

Version 3 deliberately calibrates ``raw_global_action_energy_score``: the
global MACE margin returned by ``CandidateActionEnergyResult.reward``.  It
never aliases that value with the phase-conjunctive minimum.  The latter is a
useful diagnostic for ordered interactions, but is an invalid positive gate
for actions such as stand-to-sit whose early phases legitimately depict the
pre-action state.

This module is scalar/receipt-only.  It has no path, media, tensor, source,
target, donor, mask, flow, pose, track, or trajectory input.  External manual
or VLM event judgments enter solely as detached booleans in a separately
sealed event-audit receipt.  A calibration can authorize optimization only
when every action branch shows the complete transition and terminal hold,
every hard-negative branch explicitly confirms the full target action false,
fit/confirmation actor, scene, and action-instance groups are disjoint, every
family/split has full ten-branch coverage, and both overall and per-family
confirmation metrics pass.  ``full_target_action_observed`` and
``full_target_action_false_confirmed`` may both be false for an ambiguous
audit (which fails closed), but may never both be true.

Pure T2V generations remain calibration/action-field evidence only.  Nothing
in this module permits using them as an RV2V target, donor, input, or noise.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from typing import Any


EVENT_AUDIT_SCHEMA = "bernini-pair-v5-detached-event-audit-receipt-v3"
SCORE_ROW_SCHEMA = "bernini-pair-v5-global-action-energy-row-v3"
PREREGISTRATION_SCHEMA = "bernini-pair-v5-global-action-energy-preregistration-v3"
CALIBRATION_RECEIPT_SCHEMA = "bernini-pair-v5-global-action-energy-calibration-receipt-v3"

FRAME_COUNT = 81
ANALYSIS_SPLITS = ("fit", "confirmation")
ACTION_BRANCH = "action"
NEGATIVE_BRANCHES = (
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)
BRANCH_ORDER = (ACTION_BRANCH, *NEGATIVE_BRANCHES)
GROUP_AXES = ("actor_group_id", "scene_group_id", "action_group_id")
AUDIT_SOURCE_KINDS = ("manual_detached", "vlm_detached")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

_EVENT_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "semantic_branch",
        "generation_receipt_digest",
        "audit_source_kind",
        "external_audit_artifact_sha256",
        "complete_target_transition_observed",
        "terminal_hold_observed",
        "full_target_action_observed",
        "full_target_action_false_confirmed",
        "event_qualified_action_positive",
        "external_labels_are_detached_booleans",
        "labels_used_as_model_condition",
        "audit_media_used_by_calibrator",
        "scientific_action_editing_claim",
        "receipt_digest",
    }
)
_SCORE_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "row_id",
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "semantic_branch",
        "raw_global_action_energy_score",
        "score_coordinate",
        "frame_count",
        "generation_receipt_digest",
        "frozen_scorer_receipt_digest",
        "event_audit_receipt_digest",
        "media_fields_present",
        "row_digest",
    }
)
_PREREG_FIELDS = frozenset(
    {
        "schema_version",
        "calibrator_id",
        "action_family_order",
        "analysis_split_order",
        "branch_order",
        "group_disjointness_axes",
        "minimum_rows_per_branch_per_family_per_split",
        "fit_positive_lower_quantile",
        "fit_negative_upper_quantile",
        "minimum_fit_anchor_gap",
        "decision_threshold",
        "minimum_confirmation_auroc",
        "minimum_confirmation_positive_recall",
        "minimum_confirmation_negative_specificity_by_branch",
        "require_overall_and_per_family_metrics",
        "score_field",
        "phase_conjunctive_role",
        "preregistration_digest",
    }
)


class PairV5EnergyCalibrationV3Error(ValueError):
    """A v3 artifact is malformed, ambiguous, or exceeds its authority."""


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
        raise PairV5EnergyCalibrationV3Error(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise PairV5EnergyCalibrationV3Error(
            f"{label} fields differ: missing={sorted(set(fields)-actual)}, "
            f"extra={sorted(actual-set(fields))}"
        )
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5EnergyCalibrationV3Error(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise PairV5EnergyCalibrationV3Error(f"{label} must be a canonical safe ID")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PairV5EnergyCalibrationV3Error(f"{label} must be a finite JSON float")
    return value


def _unit_float(value: Any, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise PairV5EnergyCalibrationV3Error(f"{label} must lie in [0,1]")
    return result


def _seal(unsigned: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    result = dict(unsigned)
    result[field] = object_sha256(result)
    return result


def _verify_embedded(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    digest = _sha256(value[field], label=f"{label} {field}")
    unsigned = dict(value)
    unsigned.pop(field)
    if object_sha256(unsigned) != digest:
        raise PairV5EnergyCalibrationV3Error(f"{label} embedded digest mismatch")
    return digest


def seal_event_audit_receipt(
    *,
    candidate_id: str,
    analysis_split: str,
    action_family_id: str,
    calibration_group_id: str,
    actor_group_id: str,
    scene_group_id: str,
    action_group_id: str,
    semantic_branch: str,
    generation_receipt_digest: str,
    audit_source_kind: str,
    external_audit_artifact_sha256: str,
    complete_target_transition_observed: bool,
    terminal_hold_observed: bool,
    full_target_action_observed: bool,
    full_target_action_false_confirmed: bool,
) -> dict[str, Any]:
    """Seal detached branch labels; no audit artifact content enters a model."""

    if analysis_split not in ANALYSIS_SPLITS:
        raise PairV5EnergyCalibrationV3Error("event audit analysis_split differs")
    if semantic_branch not in BRANCH_ORDER:
        raise PairV5EnergyCalibrationV3Error("event audit branch differs")
    if audit_source_kind not in AUDIT_SOURCE_KINDS:
        raise PairV5EnergyCalibrationV3Error("event audit source kind differs")
    for name, value in (
        ("complete_target_transition_observed", complete_target_transition_observed),
        ("terminal_hold_observed", terminal_hold_observed),
        ("full_target_action_observed", full_target_action_observed),
        ("full_target_action_false_confirmed", full_target_action_false_confirmed),
    ):
        if type(value) is not bool:
            raise PairV5EnergyCalibrationV3Error(f"{name} must be a detached boolean")
    if full_target_action_observed and full_target_action_false_confirmed:
        raise PairV5EnergyCalibrationV3Error(
            "target action cannot be both observed and confirmed false"
        )
    identity = {
        "candidate_id": candidate_id,
        "action_family_id": action_family_id,
        "calibration_group_id": calibration_group_id,
        "actor_group_id": actor_group_id,
        "scene_group_id": scene_group_id,
        "action_group_id": action_group_id,
    }
    for name, value in identity.items():
        _safe_id(value, label=name)
    event_positive = bool(
        semantic_branch == ACTION_BRANCH
        and complete_target_transition_observed
        and terminal_hold_observed
        and full_target_action_observed
    )
    unsigned = {
        "schema_version": EVENT_AUDIT_SCHEMA,
        **identity,
        "analysis_split": analysis_split,
        "semantic_branch": semantic_branch,
        "generation_receipt_digest": _sha256(
            generation_receipt_digest, label="generation receipt digest"
        ),
        "audit_source_kind": audit_source_kind,
        "external_audit_artifact_sha256": _sha256(
            external_audit_artifact_sha256, label="external audit artifact SHA-256"
        ),
        "complete_target_transition_observed": complete_target_transition_observed,
        "terminal_hold_observed": terminal_hold_observed,
        "full_target_action_observed": full_target_action_observed,
        "full_target_action_false_confirmed": full_target_action_false_confirmed,
        "event_qualified_action_positive": event_positive,
        "external_labels_are_detached_booleans": True,
        "labels_used_as_model_condition": False,
        "audit_media_used_by_calibrator": False,
        "scientific_action_editing_claim": False,
    }
    return _seal(unsigned, field="receipt_digest")


def validate_event_audit_receipt(value: Any) -> dict[str, Any]:
    row = dict(_closed(value, _EVENT_AUDIT_FIELDS, label="event audit receipt"))
    if row["schema_version"] != EVENT_AUDIT_SCHEMA:
        raise PairV5EnergyCalibrationV3Error("event audit schema differs")
    rebuilt = seal_event_audit_receipt(
        **{
            key: row[key]
            for key in (
                "candidate_id",
                "analysis_split",
                "action_family_id",
                "calibration_group_id",
                "actor_group_id",
                "scene_group_id",
                "action_group_id",
                "semantic_branch",
                "generation_receipt_digest",
                "audit_source_kind",
                "external_audit_artifact_sha256",
                "complete_target_transition_observed",
                "terminal_hold_observed",
                "full_target_action_observed",
                "full_target_action_false_confirmed",
            )
        }
    )
    if row != rebuilt:
        raise PairV5EnergyCalibrationV3Error("event audit semantic or digest closure differs")
    return row


def make_score_row(
    *,
    row_id: str,
    candidate_id: str,
    analysis_split: str,
    action_family_id: str,
    calibration_group_id: str,
    actor_group_id: str,
    scene_group_id: str,
    action_group_id: str,
    semantic_branch: str,
    raw_global_action_energy_score: float,
    generation_receipt_digest: str,
    frozen_scorer_receipt_digest: str,
    event_audit_receipt_digest: str,
) -> dict[str, Any]:
    """Seal one scalar-only global-MACE observation."""

    for name, value in (
        ("row_id", row_id),
        ("candidate_id", candidate_id),
        ("action_family_id", action_family_id),
        ("calibration_group_id", calibration_group_id),
        ("actor_group_id", actor_group_id),
        ("scene_group_id", scene_group_id),
        ("action_group_id", action_group_id),
    ):
        _safe_id(value, label=name)
    if analysis_split not in ANALYSIS_SPLITS or semantic_branch not in BRANCH_ORDER:
        raise PairV5EnergyCalibrationV3Error("score-row split or branch differs")
    unsigned = {
        "schema_version": SCORE_ROW_SCHEMA,
        "row_id": row_id,
        "candidate_id": candidate_id,
        "analysis_split": analysis_split,
        "action_family_id": action_family_id,
        "calibration_group_id": calibration_group_id,
        "actor_group_id": actor_group_id,
        "scene_group_id": scene_group_id,
        "action_group_id": action_group_id,
        "semantic_branch": semantic_branch,
        "raw_global_action_energy_score": _finite_float(
            raw_global_action_energy_score,
            label="raw_global_action_energy_score",
        ),
        "score_coordinate": "candidate_own_exact81_same_cell_official_gaussian",
        "frame_count": FRAME_COUNT,
        "generation_receipt_digest": _sha256(
            generation_receipt_digest, label="generation receipt digest"
        ),
        "frozen_scorer_receipt_digest": _sha256(
            frozen_scorer_receipt_digest, label="frozen scorer receipt digest"
        ),
        "event_audit_receipt_digest": _sha256(
            event_audit_receipt_digest, label="event audit receipt digest"
        ),
        "media_fields_present": False,
    }
    return _seal(unsigned, field="row_digest")


def validate_score_row(value: Any) -> dict[str, Any]:
    row = dict(_closed(value, _SCORE_ROW_FIELDS, label="score row"))
    if (
        row["schema_version"] != SCORE_ROW_SCHEMA
        or row["score_coordinate"]
        != "candidate_own_exact81_same_cell_official_gaussian"
        or row["frame_count"] != FRAME_COUNT
        or row["media_fields_present"] is not False
    ):
        raise PairV5EnergyCalibrationV3Error("score-row information-flow contract differs")
    _verify_embedded(row, field="row_digest", label="score row")
    # Re-run strict scalar/ID validators without permitting coercion.
    _finite_float(
        row["raw_global_action_energy_score"],
        label="raw_global_action_energy_score",
    )
    if row["analysis_split"] not in ANALYSIS_SPLITS or row["semantic_branch"] not in BRANCH_ORDER:
        raise PairV5EnergyCalibrationV3Error("score-row split/branch differs")
    for name in (
        "row_id",
        "candidate_id",
        "action_family_id",
        "calibration_group_id",
        *GROUP_AXES,
    ):
        _safe_id(row[name], label=name)
    for name in (
        "generation_receipt_digest",
        "frozen_scorer_receipt_digest",
        "event_audit_receipt_digest",
    ):
        _sha256(row[name], label=name)
    return row


def make_preregistration(
    calibrator_id: str,
    action_family_order: Sequence[str],
    *,
    minimum_rows_per_branch_per_family_per_split: int = 1,
    fit_positive_lower_quantile: float = 0.10,
    fit_negative_upper_quantile: float = 0.90,
    minimum_fit_anchor_gap: float = 1.0e-6,
    decision_threshold: float = 0.50,
    minimum_confirmation_auroc: float = 0.75,
    minimum_confirmation_positive_recall: float = 1.0,
    minimum_confirmation_negative_specificity_by_branch: float = 1.0,
) -> dict[str, Any]:
    _safe_id(calibrator_id, label="calibrator_id")
    families = list(action_family_order)
    if not families or len(set(families)) != len(families):
        raise PairV5EnergyCalibrationV3Error("action families must be nonempty and unique")
    for family in families:
        _safe_id(family, label="action family")
    minimum = minimum_rows_per_branch_per_family_per_split
    if type(minimum) is not int or minimum < 1:
        raise PairV5EnergyCalibrationV3Error("minimum branch rows must be >=1")
    unsigned = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "calibrator_id": calibrator_id,
        "action_family_order": families,
        "analysis_split_order": list(ANALYSIS_SPLITS),
        "branch_order": list(BRANCH_ORDER),
        "group_disjointness_axes": list(GROUP_AXES),
        "minimum_rows_per_branch_per_family_per_split": minimum,
        "fit_positive_lower_quantile": _unit_float(
            fit_positive_lower_quantile, label="fit positive lower quantile"
        ),
        "fit_negative_upper_quantile": _unit_float(
            fit_negative_upper_quantile, label="fit negative upper quantile"
        ),
        "minimum_fit_anchor_gap": _finite_float(
            minimum_fit_anchor_gap, label="minimum fit anchor gap"
        ),
        "decision_threshold": _unit_float(
            decision_threshold, label="decision threshold"
        ),
        "minimum_confirmation_auroc": _unit_float(
            minimum_confirmation_auroc, label="minimum confirmation AUROC"
        ),
        "minimum_confirmation_positive_recall": _unit_float(
            minimum_confirmation_positive_recall,
            label="minimum confirmation positive recall",
        ),
        "minimum_confirmation_negative_specificity_by_branch": _unit_float(
            minimum_confirmation_negative_specificity_by_branch,
            label="minimum confirmation negative specificity",
        ),
        "require_overall_and_per_family_metrics": True,
        "score_field": "raw_global_action_energy_score",
        "phase_conjunctive_role": "diagnostic_only_never_optimizer_gate",
    }
    if unsigned["minimum_fit_anchor_gap"] <= 0.0:
        raise PairV5EnergyCalibrationV3Error("minimum fit anchor gap must be positive")
    return _seal(unsigned, field="preregistration_digest")


def validate_preregistration(value: Any) -> dict[str, Any]:
    row = dict(_closed(value, _PREREG_FIELDS, label="preregistration"))
    _verify_embedded(row, field="preregistration_digest", label="preregistration")
    rebuilt = make_preregistration(
        row["calibrator_id"],
        row["action_family_order"],
        minimum_rows_per_branch_per_family_per_split=row[
            "minimum_rows_per_branch_per_family_per_split"
        ],
        fit_positive_lower_quantile=row["fit_positive_lower_quantile"],
        fit_negative_upper_quantile=row["fit_negative_upper_quantile"],
        minimum_fit_anchor_gap=row["minimum_fit_anchor_gap"],
        decision_threshold=row["decision_threshold"],
        minimum_confirmation_auroc=row["minimum_confirmation_auroc"],
        minimum_confirmation_positive_recall=row[
            "minimum_confirmation_positive_recall"
        ],
        minimum_confirmation_negative_specificity_by_branch=row[
            "minimum_confirmation_negative_specificity_by_branch"
        ],
    )
    if row != rebuilt:
        raise PairV5EnergyCalibrationV3Error("preregistration semantic closure differs")
    return row


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise PairV5EnergyCalibrationV3Error("cannot compute an empty quantile")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _map_score(value: float, lower: float, upper: float) -> float:
    if not upper > lower:
        return 0.0
    return float(min(1.0, max(0.0, (value - lower) / (upper - lower))))


def _auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return float(wins / (len(positives) * len(negatives)))


def _confirmation_metrics(
    rows: Sequence[Mapping[str, Any]],
    audits: Mapping[str, Mapping[str, Any]],
    mappings: Mapping[str, Mapping[str, float]],
    *,
    threshold: float,
) -> dict[str, Any]:
    labels: list[bool] = []
    scores: list[float] = []
    negative_by_branch: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        audit = audits[row["candidate_id"]]
        label = bool(audit["event_qualified_action_positive"])
        mapping = mappings[row["action_family_id"]]
        score = _map_score(
            row["raw_global_action_energy_score"],
            mapping["lower_raw_anchor"],
            mapping["upper_raw_anchor"],
        )
        predicted = score >= threshold
        labels.append(label)
        scores.append(score)
        if row["semantic_branch"] in NEGATIVE_BRANCHES:
            negative_by_branch[row["semantic_branch"]].append(not predicted)
    positive_predictions = [score >= threshold for label, score in zip(labels, scores) if label]
    recall = (
        sum(positive_predictions) / len(positive_predictions)
        if positive_predictions
        else 0.0
    )
    specificity = {
        branch: (
            sum(negative_by_branch[branch]) / len(negative_by_branch[branch])
            if negative_by_branch[branch]
            else 0.0
        )
        for branch in NEGATIVE_BRANCHES
    }
    return {
        "row_count": len(rows),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "auroc": _auroc(labels, scores),
        "positive_recall": float(recall),
        "negative_specificity_by_branch": specificity,
    }


def calibrate_global_action_energy(
    score_rows: Sequence[Mapping[str, Any]],
    event_audit_receipts: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    *,
    source_bank_spec_sha256: str,
    source_bank_receipt_digest: str,
) -> dict[str, Any]:
    """Fit on fit rows, evaluate once on confirmation, and fail closed."""

    prereg = validate_preregistration(preregistration)
    source_spec_digest = _sha256(
        source_bank_spec_sha256, label="source bank spec SHA-256"
    )
    source_bank_digest = _sha256(
        source_bank_receipt_digest, label="source bank receipt digest"
    )
    rows = [validate_score_row(value) for value in score_rows]
    audits_list = [validate_event_audit_receipt(value) for value in event_audit_receipts]
    if len({row["row_id"] for row in rows}) != len(rows):
        raise PairV5EnergyCalibrationV3Error("score row IDs repeat")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise PairV5EnergyCalibrationV3Error("score candidate IDs repeat")
    if len({row["candidate_id"] for row in audits_list}) != len(audits_list):
        raise PairV5EnergyCalibrationV3Error("event audit candidate IDs repeat")
    audits = {row["candidate_id"]: row for row in audits_list}
    if set(audits) != {row["candidate_id"] for row in rows}:
        raise PairV5EnergyCalibrationV3Error("score/event-audit candidate closure differs")

    identity_fields = (
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        *GROUP_AXES,
        "semantic_branch",
        "generation_receipt_digest",
    )
    for row in rows:
        audit = audits[row["candidate_id"]]
        if any(row[field] != audit[field] for field in identity_fields):
            raise PairV5EnergyCalibrationV3Error("score/event-audit identity binding differs")
        if row["event_audit_receipt_digest"] != audit["receipt_digest"]:
            raise PairV5EnergyCalibrationV3Error("score row binds a different event audit")

    families = prereg["action_family_order"]
    if set(row["action_family_id"] for row in rows) != set(families):
        raise PairV5EnergyCalibrationV3Error("observed action-family closure differs")
    failures: list[str] = []

    # Group disjointness is checked independently on the three sealed axes.
    disjoint_by_axis: dict[str, bool] = {}
    for axis in GROUP_AXES:
        fit = {row[axis] for row in rows if row["analysis_split"] == "fit"}
        confirmation = {
            row[axis] for row in rows if row["analysis_split"] == "confirmation"
        }
        disjoint_by_axis[axis] = not bool(fit & confirmation)
        if not disjoint_by_axis[axis]:
            failures.append(f"fit_confirmation_overlap:{axis}")

    minimum = prereg["minimum_rows_per_branch_per_family_per_split"]
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    coverage_complete = True
    for family in families:
        coverage[family] = {}
        for split in ANALYSIS_SPLITS:
            counts = {
                branch: sum(
                    row["action_family_id"] == family
                    and row["analysis_split"] == split
                    and row["semantic_branch"] == branch
                    for row in rows
                )
                for branch in BRANCH_ORDER
            }
            coverage[family][split] = counts
            for branch, count in counts.items():
                if count < minimum:
                    coverage_complete = False
                    failures.append(f"coverage:{family}:{split}:{branch}")

    action_event_contract_passed = True
    negative_event_contract_passed = True
    for row in rows:
        audit = audits[row["candidate_id"]]
        branch = row["semantic_branch"]
        if branch == ACTION_BRANCH:
            passed = (
                audit["complete_target_transition_observed"] is True
                and audit["terminal_hold_observed"] is True
                and audit["full_target_action_observed"] is True
                and audit["full_target_action_false_confirmed"] is False
                and audit["event_qualified_action_positive"] is True
            )
            if not passed:
                action_event_contract_passed = False
        else:
            passed = (
                audit["full_target_action_observed"] is False
                and audit["full_target_action_false_confirmed"] is True
                and audit["event_qualified_action_positive"] is False
            )
            if not passed:
                negative_event_contract_passed = False
        if not passed:
            failures.append(f"event_audit:{row['candidate_id']}:{branch}")
    event_contract_passed = bool(
        action_event_contract_passed and negative_event_contract_passed
    )

    mappings: dict[str, dict[str, Any]] = {}
    anchor_separation: dict[str, bool] = {}
    for family in families:
        fit_family = [
            row
            for row in rows
            if row["analysis_split"] == "fit" and row["action_family_id"] == family
        ]
        positives = [
            row["raw_global_action_energy_score"]
            for row in fit_family
            if audits[row["candidate_id"]]["event_qualified_action_positive"]
        ]
        negatives = [
            row["raw_global_action_energy_score"]
            for row in fit_family
            if row["semantic_branch"] in NEGATIVE_BRANCHES
            and audits[row["candidate_id"]]["full_target_action_observed"] is False
            and audits[row["candidate_id"]][
                "full_target_action_false_confirmed"
            ]
            is True
        ]
        lower = _quantile(negatives, prereg["fit_negative_upper_quantile"]) if negatives else 0.0
        upper = _quantile(positives, prereg["fit_positive_lower_quantile"]) if positives else 0.0
        separated = bool(
            positives
            and negatives
            and upper - lower >= prereg["minimum_fit_anchor_gap"]
        )
        anchor_separation[family] = separated
        if not separated:
            failures.append(f"fit_anchor_separation:{family}")
        mapping_unsigned = {
            "kind": "clipped_affine_fit_only",
            "score_field": "raw_global_action_energy_score",
            "lower_raw_anchor": float(lower),
            "upper_raw_anchor": float(upper),
            "clip_min": 0.0,
            "clip_max": 1.0,
            "fit_positive_count": len(positives),
            "fit_negative_count": len(negatives),
            "anchor_source_split": "fit",
        }
        mappings[family] = {
            **mapping_unsigned,
            "mapping_digest": object_sha256(mapping_unsigned),
        }

    confirmation = [row for row in rows if row["analysis_split"] == "confirmation"]
    overall_metrics = _confirmation_metrics(
        confirmation,
        audits,
        mappings,
        threshold=prereg["decision_threshold"],
    )
    per_family_metrics = {
        family: _confirmation_metrics(
            [row for row in confirmation if row["action_family_id"] == family],
            audits,
            mappings,
            threshold=prereg["decision_threshold"],
        )
        for family in families
    }

    def metrics_pass(metric: Mapping[str, Any]) -> bool:
        return bool(
            metric["auroc"] >= prereg["minimum_confirmation_auroc"]
            and metric["positive_recall"]
            >= prereg["minimum_confirmation_positive_recall"]
            and all(
                metric["negative_specificity_by_branch"][branch]
                >= prereg["minimum_confirmation_negative_specificity_by_branch"]
                for branch in NEGATIVE_BRANCHES
            )
        )

    overall_passed = metrics_pass(overall_metrics)
    per_family_passed = {
        family: metrics_pass(metric) for family, metric in per_family_metrics.items()
    }
    if not overall_passed:
        failures.append("confirmation_metrics:overall")
    for family, passed in per_family_passed.items():
        if not passed:
            failures.append(f"confirmation_metrics:{family}")

    gates = {
        "full_branch_coverage_by_family_and_split": coverage_complete,
        "fit_confirmation_group_disjoint_by_axis": disjoint_by_axis,
        "all_action_transitions_and_terminal_holds_event_qualified": action_event_contract_passed,
        "all_negative_branches_full_target_action_false": negative_event_contract_passed,
        "fit_anchor_separation_by_family": anchor_separation,
        "confirmation_overall": overall_passed,
        "confirmation_by_family": per_family_passed,
    }
    optimizer_authorized = bool(
        coverage_complete
        and all(disjoint_by_axis.values())
        and event_contract_passed
        and all(anchor_separation.values())
        and overall_passed
        and all(per_family_passed.values())
    )
    # A false gate must always have an explicit reason and vice versa.
    unique_failures = sorted(set(failures))
    if optimizer_authorized == bool(unique_failures):
        raise PairV5EnergyCalibrationV3Error("gate/failure-reason closure differs")

    fit_rows = sorted(
        (row for row in rows if row["analysis_split"] == "fit"),
        key=lambda row: row["row_id"],
    )
    confirmation_rows = sorted(
        (row for row in rows if row["analysis_split"] == "confirmation"),
        key=lambda row: row["row_id"],
    )
    unsigned = {
        "schema_version": CALIBRATION_RECEIPT_SCHEMA,
        "calibrator_id": prereg["calibrator_id"],
        "preregistration_digest": prereg["preregistration_digest"],
        "source_bank_spec_sha256": source_spec_digest,
        "source_bank_receipt_digest": source_bank_digest,
        "score_field": "raw_global_action_energy_score",
        "phase_conjunctive_score_used_for_calibration": False,
        "phase_conjunctive_role": "diagnostic_only_never_optimizer_gate",
        "frame_count": FRAME_COUNT,
        "action_family_order": list(families),
        "branch_order": list(BRANCH_ORDER),
        "fit_row_count": len(fit_rows),
        "confirmation_row_count": len(confirmation_rows),
        "fit_row_set_digest": object_sha256([row["row_digest"] for row in fit_rows]),
        "confirmation_row_set_digest": object_sha256(
            [row["row_digest"] for row in confirmation_rows]
        ),
        "event_audit_receipt_set_digest": object_sha256(
            sorted(audit["receipt_digest"] for audit in audits_list)
        ),
        "frozen_scorer_receipt_set_digest": object_sha256(
            sorted({row["frozen_scorer_receipt_digest"] for row in rows})
        ),
        "coverage_counts": coverage,
        "mapping_by_family": mappings,
        "confirmation_metrics": {
            "overall": overall_metrics,
            "by_family": per_family_metrics,
        },
        "raw_score_evidence_by_family": {
            family: {
                split: [
                    {
                        "candidate_id": row["candidate_id"],
                        "semantic_branch": row["semantic_branch"],
                        "raw_global_action_energy_score": row[
                            "raw_global_action_energy_score"
                        ],
                        "event_audit_receipt_digest": row[
                            "event_audit_receipt_digest"
                        ],
                        "generation_receipt_digest": row[
                            "generation_receipt_digest"
                        ],
                        "frozen_scorer_receipt_digest": row[
                            "frozen_scorer_receipt_digest"
                        ],
                        "score_row_digest": row["row_digest"],
                    }
                    for row in sorted(
                        (
                            item
                            for item in rows
                            if item["action_family_id"] == family
                            and item["analysis_split"] == split
                        ),
                        key=lambda item: (
                            BRANCH_ORDER.index(item["semantic_branch"]),
                            item["candidate_id"],
                        ),
                    )
                ]
                for split in ANALYSIS_SPLITS
            }
            for family in families
        },
        "decision_threshold": prereg["decision_threshold"],
        "confirmation_thresholds": {
            "minimum_auroc": prereg["minimum_confirmation_auroc"],
            "minimum_positive_recall": prereg[
                "minimum_confirmation_positive_recall"
            ],
            "minimum_negative_specificity_by_branch": prereg[
                "minimum_confirmation_negative_specificity_by_branch"
            ],
        },
        "gates": gates,
        "fit_event_qualified_action_candidate_ids": sorted(
            row["candidate_id"]
            for row in fit_rows
            if audits[row["candidate_id"]]["event_qualified_action_positive"]
        ),
        "confirmation_rows_consumed_by_optimizer": False,
        "t2v_media_consumed_by_calibrator": False,
        "t2v_media_as_rv2v_target_donor_input_or_noise": False,
        "optimizer_authorized": optimizer_authorized,
        "failure_reasons": unique_failures,
        "scientific_action_editing_claim": False,
    }
    return _seal(unsigned, field="receipt_digest")


__all__ = [
    "ACTION_BRANCH",
    "ANALYSIS_SPLITS",
    "AUDIT_SOURCE_KINDS",
    "BRANCH_ORDER",
    "CALIBRATION_RECEIPT_SCHEMA",
    "EVENT_AUDIT_SCHEMA",
    "FRAME_COUNT",
    "GROUP_AXES",
    "NEGATIVE_BRANCHES",
    "PREREGISTRATION_SCHEMA",
    "PairV5EnergyCalibrationV3Error",
    "SCORE_ROW_SCHEMA",
    "calibrate_global_action_energy",
    "canonical_json_bytes",
    "make_preregistration",
    "make_score_row",
    "object_sha256",
    "seal_event_audit_receipt",
    "validate_event_audit_receipt",
    "validate_preregistration",
    "validate_score_row",
]
