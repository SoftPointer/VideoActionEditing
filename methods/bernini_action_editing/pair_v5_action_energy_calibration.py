"""Fail-closed calibration of PAIR-v5's frozen T2V action critic.

The calibration bank contains *only scalar receipts* produced by evaluating
frozen Bernini T2V self-generations in each video's own exact-81 coordinate.
The T2V videos teach a robust scale for action evidence; they are never a
student target, donor, latent, noise, or condition.

Calibration is deliberately conservative:

* the branch registry is one ``action`` prompt plus nine hard negatives;
* fit and confirmation are disjoint by prompt and action-instance group;
* every registered action family must cover every branch in both splits;
* robust quantile anchors are fitted independently for every action family;
* confirmation AUROC, positive recall, and negative specificity must pass
  both overall and for every action family; and
* an action positive is ``branch == action AND event_qualified``.

Ordinary scientific failures (missing coverage, group leakage, overlapping
anchors, or failed confirmation metrics) produce a sealed receipt with
``optimizer_authorized == False``.  Malformed/tampered artifacts raise.  The
RV2V scoring API accepts only an opaque candidate ID, scalar candidate-own
energy, and receipt digests.  There is no media/tensor or privileged geometry
slot in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from typing import Any


ROW_SCHEMA = "bernini-pair-v5-t2v-action-calibration-row-v2"
PREREGISTRATION_SCHEMA = "bernini-pair-v5-action-calibration-preregistration-v2"
RECEIPT_SCHEMA = "bernini-pair-v5-action-calibration-receipt-v2"
PROVENANCE_SCHEMA = "bernini-pair-v5-action-calibration-provenance-v2"
RV2V_SCORE_SCHEMA = "bernini-pair-v5-rv2v-calibrated-action-score-v1"

GENERATION_MODE = "frozen_bernini_t2v_self_generated"
SCORE_COORDINATE = "candidate_own_exact81"
FRAME_COUNT = 81
SPLITS = ("fit", "confirmation")
GROUP_AXES = ("prompt_group", "action_family_group")
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
BRANCHES = (ACTION_BRANCH, *NEGATIVE_BRANCHES)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9._:-]{0,191}")

_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "row_id",
        "split",
        "action_family",
        "prompt_group",
        "action_family_group",
        "branch",
        "raw_phase_conjunctive_score",
        "event_qualified",
        "generation_mode",
        "score_coordinate",
        "frame_count",
        "frozen_generator_receipt_digest",
        "frozen_scorer_receipt_digest",
        "event_qualification_receipt_digest",
        "row_digest",
    }
)
_PREREG_FIELDS = frozenset(
    {
        "schema_version",
        "calibrator_id",
        "action_family_order",
        "action_branch",
        "negative_branch_order",
        "group_disjointness_axes",
        "minimum_rows_per_required_branch_per_family_per_split",
        "fit_positive_lower_quantile",
        "fit_negative_upper_quantile",
        "minimum_fit_anchor_gap",
        "decision_threshold",
        "minimum_confirmation_auroc",
        "minimum_confirmation_positive_recall",
        "minimum_confirmation_negative_specificity_by_branch",
        "require_metrics_per_family",
        "preregistration_digest",
    }
)
_MAPPING_FIELDS = frozenset(
    {
        "kind",
        "lower_raw_anchor",
        "upper_raw_anchor",
        "clip_min",
        "clip_max",
        "anchor_source_split",
        "anchor_statistic",
        "mapping_digest",
    }
)
_METRIC_FIELDS = frozenset(
    {"auroc", "positive_recall", "negative_specificity_by_branch"}
)
_GATE_FIELDS = frozenset(
    {
        "coverage_complete",
        "group_disjoint",
        "fit_anchor_separation_by_family",
        "confirmation_overall",
        "confirmation_by_family",
    }
)
_CLOSURE_FIELDS = frozenset(
    {
        "score_rows_only",
        "t2v_self_generations_are_calibration_only",
        "fit_anchors_frozen_before_confirmation",
        "confirmation_can_modify_mapping",
        "proposal_media_consumed",
        "proposal_latent_consumed",
        "proposal_noise_consumed",
        "source_or_target_media_consumed",
        "mask_flow_pose_track_trajectory_consumed",
        "usable_candidate_score_emitted_during_calibration",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "calibrator_id",
        "preregistration_digest",
        "frozen_generator_receipt_digest",
        "frozen_scorer_receipt_digest",
        "fit_row_set_digest",
        "confirmation_row_set_digest",
        "fit_row_count",
        "confirmation_row_count",
        "action_family_order",
        "positive_definition",
        "negative_definition",
        "mapping_by_family",
        "fit_quantiles_by_family",
        "confirmation_metrics",
        "gates",
        "optimizer_authorized",
        "failure_reasons",
        "input_closure",
        "receipt_digest",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "calibrator_id",
        "calibration_receipt_digest",
        "preregistration_digest",
        "frozen_generator_receipt_digest",
        "frozen_scorer_receipt_digest",
        "mapping_digest_by_family",
        "optimizer_authorized",
        "input_semantics",
        "proposal_visual_data_consumed",
        "privileged_visual_inputs_consumed",
        "provenance_digest",
    }
)
_RV2V_SCORE_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "action_family",
        "raw_candidate_own_score",
        "calibrated_action_score",
        "score_coordinate",
        "candidate_evaluator_receipt_digest",
        "calibration_receipt_digest",
        "frozen_scorer_receipt_digest",
        "proposal_visual_data_consumed",
        "privileged_visual_inputs_consumed",
        "score_digest",
    }
)


class PairV5CalibrationError(ValueError):
    """A calibration artifact violates the fail-closed contract."""


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
        raise PairV5CalibrationError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV5CalibrationError(f"{label} must be a mapping")
    keys = set(value)
    if not all(isinstance(key, str) for key in keys):
        raise PairV5CalibrationError(f"{label} keys must be strings")
    missing = sorted(fields - keys)
    extra = sorted(keys - fields)
    if missing or extra:
        raise PairV5CalibrationError(
            f"{label} closure differs; missing={missing}, extra={extra}"
        )
    return value


def _seal(unsigned: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    result = dict(unsigned)
    result[field] = object_sha256(result)
    return result


def _embedded_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    digest = _sha256(value[field], label=f"{label} {field}")
    unsigned = dict(value)
    unsigned.pop(field)
    if object_sha256(unsigned) != digest:
        raise PairV5CalibrationError(f"{label} embedded digest mismatch")
    return digest


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5CalibrationError(f"{label} must be lowercase SHA-256")
    return value


def _slug(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SLUG_RE.fullmatch(value) is None:
        raise PairV5CalibrationError(f"{label} must be a lowercase canonical slug")
    return value


def _float(value: Any, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PairV5CalibrationError(f"{label} must be a finite JSON float")
    return value


def _unit_float(value: Any, *, label: str) -> float:
    result = _float(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise PairV5CalibrationError(f"{label} must be in [0, 1]")
    return result


def _positive_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise PairV5CalibrationError(f"{label} must be an integer >= 1")
    return value


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PairV5CalibrationError(f"{label} must be an integer >= 0")
    return value


def _exact_string_list(value: Any, expected: Sequence[str], *, label: str) -> list[str]:
    if not isinstance(value, list) or value != list(expected):
        raise PairV5CalibrationError(f"{label} must equal the registered order")
    return list(value)


def _family_mapping(value: Any, families: Sequence[str], *, label: str) -> Mapping[str, Any]:
    return _closed(value, frozenset(families), label=label)


def make_score_row(
    row_id: str,
    *,
    split: str,
    action_family: str,
    prompt_group: str,
    action_family_group: str,
    branch: str,
    raw_phase_conjunctive_score: float,
    event_qualified: bool,
    frozen_generator_receipt_digest: str,
    frozen_scorer_receipt_digest: str,
    event_qualification_receipt_digest: str,
) -> dict[str, Any]:
    """Seal one scalar-only observation from the frozen exact-81 T2V bank."""

    unsigned = {
        "schema_version": ROW_SCHEMA,
        "row_id": row_id,
        "split": split,
        "action_family": action_family,
        "prompt_group": prompt_group,
        "action_family_group": action_family_group,
        "branch": branch,
        "raw_phase_conjunctive_score": raw_phase_conjunctive_score,
        "event_qualified": event_qualified,
        "generation_mode": GENERATION_MODE,
        "score_coordinate": SCORE_COORDINATE,
        "frame_count": FRAME_COUNT,
        "frozen_generator_receipt_digest": frozen_generator_receipt_digest,
        "frozen_scorer_receipt_digest": frozen_scorer_receipt_digest,
        "event_qualification_receipt_digest": event_qualification_receipt_digest,
    }
    return validate_score_row(_seal(unsigned, field="row_digest"))


def validate_score_row(value: Any) -> dict[str, Any]:
    row = _closed(value, _ROW_FIELDS, label="score row")
    if row["schema_version"] != ROW_SCHEMA:
        raise PairV5CalibrationError("score row schema_version is not registered")
    result = {
        "schema_version": ROW_SCHEMA,
        "row_id": _slug(row["row_id"], label="row_id"),
        "split": row["split"],
        "action_family": _slug(row["action_family"], label="action_family"),
        "prompt_group": _slug(row["prompt_group"], label="prompt_group"),
        "action_family_group": _slug(
            row["action_family_group"], label="action_family_group"
        ),
        "branch": row["branch"],
        "raw_phase_conjunctive_score": _float(
            row["raw_phase_conjunctive_score"],
            label="raw_phase_conjunctive_score",
        ),
        "event_qualified": row["event_qualified"],
        "generation_mode": row["generation_mode"],
        "score_coordinate": row["score_coordinate"],
        "frame_count": row["frame_count"],
        "frozen_generator_receipt_digest": _sha256(
            row["frozen_generator_receipt_digest"],
            label="frozen_generator_receipt_digest",
        ),
        "frozen_scorer_receipt_digest": _sha256(
            row["frozen_scorer_receipt_digest"],
            label="frozen_scorer_receipt_digest",
        ),
        "event_qualification_receipt_digest": _sha256(
            row["event_qualification_receipt_digest"],
            label="event_qualification_receipt_digest",
        ),
    }
    if result["split"] not in SPLITS:
        raise PairV5CalibrationError("score row split must be fit or confirmation")
    if result["branch"] not in BRANCHES:
        raise PairV5CalibrationError("score row branch is outside the closed registry")
    if type(result["event_qualified"]) is not bool:
        raise PairV5CalibrationError("event_qualified must be boolean")
    if result["generation_mode"] != GENERATION_MODE:
        raise PairV5CalibrationError("calibration row is not frozen Bernini T2V self-generation")
    if result["score_coordinate"] != SCORE_COORDINATE:
        raise PairV5CalibrationError("calibration score is not candidate-own exact81")
    if result["frame_count"] != FRAME_COUNT or type(result["frame_count"]) is not int:
        raise PairV5CalibrationError("calibration row must contain exactly 81 frames")
    result["row_digest"] = _embedded_digest(row, field="row_digest", label="score row")
    return result


def make_preregistration(
    calibrator_id: str,
    *,
    action_families: Sequence[str],
    fit_positive_lower_quantile: float = 0.10,
    fit_negative_upper_quantile: float = 0.90,
    minimum_fit_anchor_gap: float = 0.05,
    decision_threshold: float = 0.50,
    minimum_confirmation_auroc: float = 0.90,
    minimum_confirmation_positive_recall: float = 0.80,
    minimum_confirmation_negative_specificity: float | Mapping[str, float] = 0.80,
    minimum_rows_per_required_branch_per_family_per_split: int = 1,
) -> dict[str, Any]:
    families = list(action_families)
    if not families or len(families) != len(set(families)):
        raise PairV5CalibrationError("action_families must be non-empty and unique")
    families = [_slug(item, label="action family") for item in families]
    if type(minimum_confirmation_negative_specificity) is float:
        specificity = {
            branch: minimum_confirmation_negative_specificity
            for branch in NEGATIVE_BRANCHES
        }
    elif isinstance(minimum_confirmation_negative_specificity, Mapping):
        specificity = dict(minimum_confirmation_negative_specificity)
    else:
        raise PairV5CalibrationError(
            "minimum_confirmation_negative_specificity must be float or mapping"
        )
    unsigned = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "calibrator_id": calibrator_id,
        "action_family_order": families,
        "action_branch": ACTION_BRANCH,
        "negative_branch_order": list(NEGATIVE_BRANCHES),
        "group_disjointness_axes": list(GROUP_AXES),
        "minimum_rows_per_required_branch_per_family_per_split": (
            minimum_rows_per_required_branch_per_family_per_split
        ),
        "fit_positive_lower_quantile": fit_positive_lower_quantile,
        "fit_negative_upper_quantile": fit_negative_upper_quantile,
        "minimum_fit_anchor_gap": minimum_fit_anchor_gap,
        "decision_threshold": decision_threshold,
        "minimum_confirmation_auroc": minimum_confirmation_auroc,
        "minimum_confirmation_positive_recall": minimum_confirmation_positive_recall,
        "minimum_confirmation_negative_specificity_by_branch": specificity,
        "require_metrics_per_family": True,
    }
    return validate_preregistration(_seal(unsigned, field="preregistration_digest"))


def validate_preregistration(value: Any) -> dict[str, Any]:
    row = _closed(value, _PREREG_FIELDS, label="preregistration")
    if row["schema_version"] != PREREGISTRATION_SCHEMA:
        raise PairV5CalibrationError("preregistration schema_version is not registered")
    families_raw = row["action_family_order"]
    if not isinstance(families_raw, list) or not families_raw:
        raise PairV5CalibrationError("action_family_order must be a non-empty list")
    families = [_slug(item, label="action family") for item in families_raw]
    if len(families) != len(set(families)):
        raise PairV5CalibrationError("action_family_order contains duplicates")
    if row["action_branch"] != ACTION_BRANCH:
        raise PairV5CalibrationError("action branch definition may not change")
    _exact_string_list(
        row["negative_branch_order"], NEGATIVE_BRANCHES, label="negative branch order"
    )
    _exact_string_list(
        row["group_disjointness_axes"], GROUP_AXES, label="group disjointness axes"
    )
    if row["require_metrics_per_family"] is not True:
        raise PairV5CalibrationError("per-family confirmation metrics are mandatory")
    specificity_raw = _closed(
        row["minimum_confirmation_negative_specificity_by_branch"],
        frozenset(NEGATIVE_BRANCHES),
        label="negative specificity thresholds",
    )
    result = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "calibrator_id": _slug(row["calibrator_id"], label="calibrator_id"),
        "action_family_order": families,
        "action_branch": ACTION_BRANCH,
        "negative_branch_order": list(NEGATIVE_BRANCHES),
        "group_disjointness_axes": list(GROUP_AXES),
        "minimum_rows_per_required_branch_per_family_per_split": _positive_integer(
            row["minimum_rows_per_required_branch_per_family_per_split"],
            label="minimum rows per branch/family/split",
        ),
        "fit_positive_lower_quantile": _unit_float(
            row["fit_positive_lower_quantile"], label="fit positive lower quantile"
        ),
        "fit_negative_upper_quantile": _unit_float(
            row["fit_negative_upper_quantile"], label="fit negative upper quantile"
        ),
        "minimum_fit_anchor_gap": _float(
            row["minimum_fit_anchor_gap"], label="minimum fit anchor gap"
        ),
        "decision_threshold": _unit_float(
            row["decision_threshold"], label="decision threshold"
        ),
        "minimum_confirmation_auroc": _unit_float(
            row["minimum_confirmation_auroc"], label="minimum confirmation AUROC"
        ),
        "minimum_confirmation_positive_recall": _unit_float(
            row["minimum_confirmation_positive_recall"],
            label="minimum confirmation positive recall",
        ),
        "minimum_confirmation_negative_specificity_by_branch": {
            branch: _unit_float(
                specificity_raw[branch],
                label=f"minimum specificity for {branch}",
            )
            for branch in NEGATIVE_BRANCHES
        },
        "require_metrics_per_family": True,
    }
    if result["minimum_fit_anchor_gap"] <= 0.0:
        raise PairV5CalibrationError("minimum fit anchor gap must be positive")
    result["preregistration_digest"] = _embedded_digest(
        row, field="preregistration_digest", label="preregistration"
    )
    return result


def event_qualified_positive(row: Mapping[str, Any]) -> bool:
    """The only registered positive predicate."""

    checked = validate_score_row(row)
    return checked["branch"] == ACTION_BRANCH and checked["event_qualified"] is True


def _is_positive_checked(row: Mapping[str, Any]) -> bool:
    return row["branch"] == ACTION_BRANCH and row["event_qualified"] is True


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise PairV5CalibrationError("quantile requires a non-empty population")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _make_mapping(lower: float, upper: float, pos_q: float, neg_q: float) -> dict[str, Any]:
    return _seal(
        {
            "kind": "per_family_clipped_linear_monotone_increasing",
            "lower_raw_anchor": lower,
            "upper_raw_anchor": upper,
            "clip_min": 0.0,
            "clip_max": 1.0,
            "anchor_source_split": "fit_only",
            "anchor_statistic": (
                f"negative_q{neg_q:.6f}_to_event_positive_q{pos_q:.6f}"
            ),
        },
        field="mapping_digest",
    )


def _apply_mapping(raw_score: float, mapping: Mapping[str, Any]) -> float:
    scaled = (raw_score - mapping["lower_raw_anchor"]) / (
        mapping["upper_raw_anchor"] - mapping["lower_raw_anchor"]
    )
    return float(min(1.0, max(0.0, scaled)))


def _auroc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return float(wins / (len(positives) * len(negatives)))


def _metrics(scored_rows: Sequence[tuple[Mapping[str, Any], float]]) -> dict[str, Any]:
    positives = [score for row, score in scored_rows if _is_positive_checked(row)]
    negatives = [score for row, score in scored_rows if not _is_positive_checked(row)]
    if not positives or not negatives:
        raise PairV5CalibrationError("confirmation metric population is empty")
    threshold = 0.5  # replaced by _metrics_with_threshold caller
    return {
        "_positives": positives,
        "_negatives": negatives,
        "_scored_rows": scored_rows,
        "_threshold": threshold,
    }


def _metrics_with_threshold(
    scored_rows: Sequence[tuple[Mapping[str, Any], float]], threshold: float
) -> dict[str, Any]:
    state = _metrics(scored_rows)
    positives = state["_positives"]
    negatives = state["_negatives"]
    specificity: dict[str, float] = {}
    for branch in NEGATIVE_BRANCHES:
        values = [score for row, score in scored_rows if row["branch"] == branch]
        if not values:
            raise PairV5CalibrationError(f"confirmation lacks branch {branch}")
        specificity[branch] = float(sum(score < threshold for score in values) / len(values))
    return {
        "auroc": _auroc(positives, negatives),
        "positive_recall": float(
            sum(score >= threshold for score in positives) / len(positives)
        ),
        "negative_specificity_by_branch": specificity,
    }


def _metrics_pass(metrics: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    return (
        metrics["auroc"] >= policy["minimum_confirmation_auroc"]
        and metrics["positive_recall"]
        >= policy["minimum_confirmation_positive_recall"]
        and all(
            metrics["negative_specificity_by_branch"][branch]
            >= policy["minimum_confirmation_negative_specificity_by_branch"][branch]
            for branch in NEGATIVE_BRANCHES
        )
    )


def _row_set_digest(rows: Sequence[Mapping[str, Any]], split: str) -> str:
    return object_sha256(
        {
            "schema_version": "bernini-pair-v5-action-calibration-row-set-v2",
            "split": split,
            "row_digests": sorted(row["row_digest"] for row in rows),
        }
    )


def _input_closure() -> dict[str, bool]:
    return {
        "score_rows_only": True,
        "t2v_self_generations_are_calibration_only": True,
        "fit_anchors_frozen_before_confirmation": True,
        "confirmation_can_modify_mapping": False,
        "proposal_media_consumed": False,
        "proposal_latent_consumed": False,
        "proposal_noise_consumed": False,
        "source_or_target_media_consumed": False,
        "mask_flow_pose_track_trajectory_consumed": False,
        "usable_candidate_score_emitted_during_calibration": False,
    }


def calibrate_action_energy(
    rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    *,
    registered_preregistration_digest: str,
) -> dict[str, Any]:
    """Fit family anchors and independently audit them on held-out groups."""

    policy = validate_preregistration(preregistration)
    pinned = _sha256(
        registered_preregistration_digest,
        label="registered_preregistration_digest",
    )
    if pinned != policy["preregistration_digest"]:
        raise PairV5CalibrationError(
            "preregistration does not match the externally registered digest"
        )
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise PairV5CalibrationError("rows must be a sequence of scalar score rows")
    checked = [validate_score_row(row) for row in rows]
    if not checked:
        raise PairV5CalibrationError("calibration rows may not be empty")
    if len({row["row_id"] for row in checked}) != len(checked):
        raise PairV5CalibrationError("row_id values must be globally unique")
    if len({row["row_digest"] for row in checked}) != len(checked):
        raise PairV5CalibrationError("duplicate score-row digest")
    families = policy["action_family_order"]
    unknown_families = sorted({row["action_family"] for row in checked} - set(families))
    if unknown_families:
        raise PairV5CalibrationError(f"unregistered action families: {unknown_families}")
    generator_digests = {row["frozen_generator_receipt_digest"] for row in checked}
    scorer_digests = {row["frozen_scorer_receipt_digest"] for row in checked}
    if len(generator_digests) != 1 or len(scorer_digests) != 1:
        raise PairV5CalibrationError("rows must bind one frozen generator and scorer")

    by_split = {
        split: [row for row in checked if row["split"] == split] for split in SPLITS
    }
    failure_reasons: list[str] = []
    minimum_rows = policy[
        "minimum_rows_per_required_branch_per_family_per_split"
    ]
    for split in SPLITS:
        for family in families:
            family_rows = [
                row for row in by_split[split] if row["action_family"] == family
            ]
            positive_count = sum(_is_positive_checked(row) for row in family_rows)
            if positive_count < minimum_rows:
                failure_reasons.append(
                    f"coverage:{split}:{family}:event_qualified_action"
                )
            for branch in NEGATIVE_BRANCHES:
                count = sum(row["branch"] == branch for row in family_rows)
                if count < minimum_rows:
                    failure_reasons.append(f"coverage:{split}:{family}:{branch}")
    coverage_complete = not failure_reasons

    group_disjoint = True
    for axis in GROUP_AXES:
        leaked = sorted(
            {row[axis] for row in by_split["fit"]}
            & {row[axis] for row in by_split["confirmation"]}
        )
        if leaked:
            group_disjoint = False
            failure_reasons.append(f"group_leakage:{axis}:{','.join(leaked)}")

    quantiles_by_family: dict[str, Any] = {}
    mappings_by_family: dict[str, Any] = {}
    separation_gates = {family: False for family in families}
    if coverage_complete and group_disjoint:
        for family in families:
            fit_rows = [
                row for row in by_split["fit"] if row["action_family"] == family
            ]
            positives = [
                row["raw_phase_conjunctive_score"]
                for row in fit_rows
                if _is_positive_checked(row)
            ]
            negatives = [
                row["raw_phase_conjunctive_score"]
                for row in fit_rows
                if not _is_positive_checked(row)
            ]
            positive_lower = _quantile(
                positives, policy["fit_positive_lower_quantile"]
            )
            negative_upper = _quantile(
                negatives, policy["fit_negative_upper_quantile"]
            )
            gap = positive_lower - negative_upper
            quantiles_by_family[family] = {
                "positive_lower": positive_lower,
                "negative_upper": negative_upper,
                "anchor_gap": gap,
            }
            passed = gap >= policy["minimum_fit_anchor_gap"]
            separation_gates[family] = passed
            if passed:
                mappings_by_family[family] = _make_mapping(
                    negative_upper,
                    positive_lower,
                    policy["fit_positive_lower_quantile"],
                    policy["fit_negative_upper_quantile"],
                )
            else:
                failure_reasons.append(f"fit_anchor_gap_below_minimum:{family}")

    all_separated = all(separation_gates.values())
    mapping_value: dict[str, Any] | None = (
        mappings_by_family if coverage_complete and group_disjoint and all_separated else None
    )
    metrics_value: dict[str, Any] | None = None
    confirmation_overall_gate = False
    confirmation_family_gates = {family: False for family in families}
    if mapping_value is not None:
        scored = [
            (
                row,
                _apply_mapping(
                    row["raw_phase_conjunctive_score"],
                    mapping_value[row["action_family"]],
                ),
            )
            for row in by_split["confirmation"]
        ]
        overall = _metrics_with_threshold(scored, policy["decision_threshold"])
        by_family_metrics = {
            family: _metrics_with_threshold(
                [(row, score) for row, score in scored if row["action_family"] == family],
                policy["decision_threshold"],
            )
            for family in families
        }
        metrics_value = {"overall": overall, "by_family": by_family_metrics}
        confirmation_overall_gate = _metrics_pass(overall, policy)
        if not confirmation_overall_gate:
            failure_reasons.append("confirmation_metrics_failed:overall")
        for family in families:
            passed = _metrics_pass(by_family_metrics[family], policy)
            confirmation_family_gates[family] = passed
            if not passed:
                failure_reasons.append(f"confirmation_metrics_failed:{family}")

    gates = {
        "coverage_complete": coverage_complete,
        "group_disjoint": group_disjoint,
        "fit_anchor_separation_by_family": separation_gates,
        "confirmation_overall": confirmation_overall_gate,
        "confirmation_by_family": confirmation_family_gates,
    }
    authorized = (
        coverage_complete
        and group_disjoint
        and all_separated
        and confirmation_overall_gate
        and all(confirmation_family_gates.values())
    )
    unsigned = {
        "schema_version": RECEIPT_SCHEMA,
        "calibrator_id": policy["calibrator_id"],
        "preregistration_digest": policy["preregistration_digest"],
        "frozen_generator_receipt_digest": next(iter(generator_digests)),
        "frozen_scorer_receipt_digest": next(iter(scorer_digests)),
        "fit_row_set_digest": _row_set_digest(by_split["fit"], "fit"),
        "confirmation_row_set_digest": _row_set_digest(
            by_split["confirmation"], "confirmation"
        ),
        "fit_row_count": len(by_split["fit"]),
        "confirmation_row_count": len(by_split["confirmation"]),
        "action_family_order": families,
        "positive_definition": "branch==action AND event_qualified==true",
        "negative_definition": "NOT positive_definition",
        "mapping_by_family": mapping_value,
        "fit_quantiles_by_family": quantiles_by_family,
        "confirmation_metrics": metrics_value,
        "gates": gates,
        "optimizer_authorized": authorized,
        "failure_reasons": failure_reasons,
        "input_closure": _input_closure(),
    }
    return validate_calibration_receipt(_seal(unsigned, field="receipt_digest"))


def _validate_mapping(value: Any) -> dict[str, Any]:
    row = _closed(value, _MAPPING_FIELDS, label="family mapping")
    if row["kind"] != "per_family_clipped_linear_monotone_increasing":
        raise PairV5CalibrationError("mapping kind is not registered")
    lower = _float(row["lower_raw_anchor"], label="lower_raw_anchor")
    upper = _float(row["upper_raw_anchor"], label="upper_raw_anchor")
    if upper <= lower:
        raise PairV5CalibrationError("mapping upper anchor must exceed lower anchor")
    if row["clip_min"] != 0.0 or row["clip_max"] != 1.0:
        raise PairV5CalibrationError("mapping clip interval must be [0,1]")
    if row["anchor_source_split"] != "fit_only":
        raise PairV5CalibrationError("mapping anchors must come from fit only")
    if not isinstance(row["anchor_statistic"], str) or not row["anchor_statistic"].startswith(
        "negative_q"
    ):
        raise PairV5CalibrationError("mapping anchor statistic is invalid")
    digest = _embedded_digest(row, field="mapping_digest", label="family mapping")
    return {**dict(row), "lower_raw_anchor": lower, "upper_raw_anchor": upper, "mapping_digest": digest}


def _validate_metrics(value: Any, *, label: str) -> dict[str, Any]:
    row = _closed(value, _METRIC_FIELDS, label=label)
    specificity = _closed(
        row["negative_specificity_by_branch"],
        frozenset(NEGATIVE_BRANCHES),
        label=f"{label} specificity",
    )
    return {
        "auroc": _unit_float(row["auroc"], label=f"{label} AUROC"),
        "positive_recall": _unit_float(
            row["positive_recall"], label=f"{label} positive recall"
        ),
        "negative_specificity_by_branch": {
            branch: _unit_float(specificity[branch], label=f"{label} {branch} specificity")
            for branch in NEGATIVE_BRANCHES
        },
    }


def validate_calibration_receipt(value: Any) -> dict[str, Any]:
    row = _closed(value, _RECEIPT_FIELDS, label="calibration receipt")
    if row["schema_version"] != RECEIPT_SCHEMA:
        raise PairV5CalibrationError("calibration receipt schema is not registered")
    families_raw = row["action_family_order"]
    if not isinstance(families_raw, list) or not families_raw:
        raise PairV5CalibrationError("receipt action families must be non-empty")
    families = [_slug(item, label="receipt action family") for item in families_raw]
    if len(families) != len(set(families)):
        raise PairV5CalibrationError("receipt action families contain duplicates")
    for field in (
        "preregistration_digest",
        "frozen_generator_receipt_digest",
        "frozen_scorer_receipt_digest",
        "fit_row_set_digest",
        "confirmation_row_set_digest",
    ):
        _sha256(row[field], label=field)
    _slug(row["calibrator_id"], label="calibrator_id")
    _nonnegative_integer(row["fit_row_count"], label="fit_row_count")
    _nonnegative_integer(
        row["confirmation_row_count"], label="confirmation_row_count"
    )
    if row["fit_row_count"] + row["confirmation_row_count"] < 1:
        raise PairV5CalibrationError("receipt cannot represent an empty row set")
    if row["positive_definition"] != "branch==action AND event_qualified==true":
        raise PairV5CalibrationError("positive definition changed")
    if row["negative_definition"] != "NOT positive_definition":
        raise PairV5CalibrationError("negative definition changed")

    mappings = row["mapping_by_family"]
    if mappings is not None:
        mapping_rows = _family_mapping(mappings, families, label="mapping_by_family")
        for family in families:
            _validate_mapping(mapping_rows[family])
    quantiles = _family_mapping(
        row["fit_quantiles_by_family"],
        row["fit_quantiles_by_family"].keys()
        if isinstance(row["fit_quantiles_by_family"], Mapping)
        else (),
        label="fit_quantiles_by_family",
    )
    if set(quantiles) not in (set(), set(families)):
        raise PairV5CalibrationError("fit quantiles must be empty or cover every family")
    for family, values in quantiles.items():
        qrow = _closed(
            values,
            frozenset({"positive_lower", "negative_upper", "anchor_gap"}),
            label=f"fit quantiles {family}",
        )
        pos = _float(qrow["positive_lower"], label=f"{family} positive lower")
        neg = _float(qrow["negative_upper"], label=f"{family} negative upper")
        gap = _float(qrow["anchor_gap"], label=f"{family} anchor gap")
        if not math.isclose(gap, pos - neg, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise PairV5CalibrationError("anchor gap is inconsistent")

    metrics = row["confirmation_metrics"]
    if metrics is not None:
        metrics_row = _closed(metrics, frozenset({"overall", "by_family"}), label="metrics")
        _validate_metrics(metrics_row["overall"], label="overall metrics")
        family_metrics = _family_mapping(
            metrics_row["by_family"], families, label="metrics by family"
        )
        for family in families:
            _validate_metrics(family_metrics[family], label=f"metrics {family}")

    gates = _closed(row["gates"], _GATE_FIELDS, label="calibration gates")
    if type(gates["coverage_complete"]) is not bool or type(gates["group_disjoint"]) is not bool:
        raise PairV5CalibrationError("coverage/group gates must be booleans")
    separation = _family_mapping(
        gates["fit_anchor_separation_by_family"], families, label="separation gates"
    )
    confirmation_family = _family_mapping(
        gates["confirmation_by_family"], families, label="confirmation family gates"
    )
    if type(gates["confirmation_overall"]) is not bool or any(
        type(separation[family]) is not bool
        or type(confirmation_family[family]) is not bool
        for family in families
    ):
        raise PairV5CalibrationError("all calibration gates must be booleans")
    prerequisites = (
        gates["coverage_complete"]
        and gates["group_disjoint"]
        and all(separation.values())
    )
    if (mappings is not None) != prerequisites:
        raise PairV5CalibrationError("mapping presence disagrees with prerequisite gates")
    if (metrics is not None) != prerequisites:
        raise PairV5CalibrationError("metric presence disagrees with prerequisite gates")
    expected_authorized = (
        prerequisites
        and gates["confirmation_overall"]
        and all(confirmation_family.values())
    )
    if type(row["optimizer_authorized"]) is not bool or row["optimizer_authorized"] != expected_authorized:
        raise PairV5CalibrationError("optimizer authorization disagrees with gates")
    reasons = row["failure_reasons"]
    if not isinstance(reasons, list) or not all(isinstance(item, str) and item for item in reasons):
        raise PairV5CalibrationError("failure_reasons must be a string list")
    if expected_authorized == bool(reasons):
        raise PairV5CalibrationError("failure reasons disagree with optimizer authorization")
    closure = _closed(row["input_closure"], _CLOSURE_FIELDS, label="input closure")
    if dict(closure) != _input_closure():
        raise PairV5CalibrationError("input closure admits forbidden data or adaptation")
    _embedded_digest(row, field="receipt_digest", label="calibration receipt")
    return dict(row)


def apply_calibrator(
    raw_candidate_own_score: float,
    action_family: str,
    calibration_receipt: Mapping[str, Any],
    *,
    registered_calibration_receipt_digest: str,
) -> float:
    """Apply the pinned per-family map to one RV2V candidate scalar."""

    raw = _float(raw_candidate_own_score, label="raw_candidate_own_score")
    family = _slug(action_family, label="action_family")
    receipt = validate_calibration_receipt(calibration_receipt)
    pinned = _sha256(
        registered_calibration_receipt_digest,
        label="registered_calibration_receipt_digest",
    )
    if pinned != receipt["receipt_digest"]:
        raise PairV5CalibrationError(
            "calibration receipt does not match the externally registered digest"
        )
    if not receipt["optimizer_authorized"] or receipt["mapping_by_family"] is None:
        raise PairV5CalibrationError(
            "calibration failed; no usable RV2V candidate score may be emitted"
        )
    if family not in receipt["mapping_by_family"]:
        raise PairV5CalibrationError("RV2V candidate action family was not calibrated")
    return _apply_mapping(raw, receipt["mapping_by_family"][family])


def score_rv2v_candidate(
    candidate_id: str,
    *,
    action_family: str,
    raw_candidate_own_score: float,
    candidate_evaluator_receipt_digest: str,
    calibration_receipt: Mapping[str, Any],
    registered_calibration_receipt_digest: str,
) -> dict[str, Any]:
    """Seal a calibrated scalar score without accepting candidate media."""

    receipt = validate_calibration_receipt(calibration_receipt)
    score = apply_calibrator(
        raw_candidate_own_score,
        action_family,
        receipt,
        registered_calibration_receipt_digest=registered_calibration_receipt_digest,
    )
    unsigned = {
        "schema_version": RV2V_SCORE_SCHEMA,
        "candidate_id": _slug(candidate_id, label="candidate_id"),
        "action_family": _slug(action_family, label="action_family"),
        "raw_candidate_own_score": _float(
            raw_candidate_own_score, label="raw_candidate_own_score"
        ),
        "calibrated_action_score": score,
        "score_coordinate": SCORE_COORDINATE,
        "candidate_evaluator_receipt_digest": _sha256(
            candidate_evaluator_receipt_digest,
            label="candidate_evaluator_receipt_digest",
        ),
        "calibration_receipt_digest": receipt["receipt_digest"],
        "frozen_scorer_receipt_digest": receipt["frozen_scorer_receipt_digest"],
        "proposal_visual_data_consumed": False,
        "privileged_visual_inputs_consumed": False,
    }
    return validate_rv2v_candidate_score(_seal(unsigned, field="score_digest"))


def validate_rv2v_candidate_score(value: Any) -> dict[str, Any]:
    row = _closed(value, _RV2V_SCORE_FIELDS, label="RV2V action score")
    if row["schema_version"] != RV2V_SCORE_SCHEMA:
        raise PairV5CalibrationError("RV2V action score schema is not registered")
    _slug(row["candidate_id"], label="candidate_id")
    _slug(row["action_family"], label="action_family")
    _float(row["raw_candidate_own_score"], label="raw_candidate_own_score")
    _unit_float(row["calibrated_action_score"], label="calibrated_action_score")
    if row["score_coordinate"] != SCORE_COORDINATE:
        raise PairV5CalibrationError("RV2V action score is not candidate-own exact81")
    for field in (
        "candidate_evaluator_receipt_digest",
        "calibration_receipt_digest",
        "frozen_scorer_receipt_digest",
    ):
        _sha256(row[field], label=field)
    if row["proposal_visual_data_consumed"] is not False or row[
        "privileged_visual_inputs_consumed"
    ] is not False:
        raise PairV5CalibrationError("RV2V score admits forbidden visual inputs")
    _embedded_digest(row, field="score_digest", label="RV2V action score")
    return dict(row)


def make_calibrator_provenance(
    calibration_receipt: Mapping[str, Any],
    *,
    registered_calibration_receipt_digest: str,
) -> dict[str, Any]:
    """Emit optimizer provenance only for the exact pinned passing receipt."""

    receipt = validate_calibration_receipt(calibration_receipt)
    pinned = _sha256(
        registered_calibration_receipt_digest,
        label="registered_calibration_receipt_digest",
    )
    if pinned != receipt["receipt_digest"]:
        raise PairV5CalibrationError(
            "calibration receipt does not match the externally registered digest"
        )
    if not receipt["optimizer_authorized"] or receipt["mapping_by_family"] is None:
        raise PairV5CalibrationError("failed calibration cannot produce provenance")
    unsigned = {
        "schema_version": PROVENANCE_SCHEMA,
        "calibrator_id": receipt["calibrator_id"],
        "calibration_receipt_digest": receipt["receipt_digest"],
        "preregistration_digest": receipt["preregistration_digest"],
        "frozen_generator_receipt_digest": receipt[
            "frozen_generator_receipt_digest"
        ],
        "frozen_scorer_receipt_digest": receipt["frozen_scorer_receipt_digest"],
        "mapping_digest_by_family": {
            family: receipt["mapping_by_family"][family]["mapping_digest"]
            for family in receipt["action_family_order"]
        },
        "optimizer_authorized": True,
        "input_semantics": "rv2v_candidate_own_scalar_action_energy_only",
        "proposal_visual_data_consumed": False,
        "privileged_visual_inputs_consumed": False,
    }
    return validate_calibrator_provenance(_seal(unsigned, field="provenance_digest"))


def validate_calibrator_provenance(value: Any) -> dict[str, Any]:
    row = _closed(value, _PROVENANCE_FIELDS, label="calibrator provenance")
    if row["schema_version"] != PROVENANCE_SCHEMA:
        raise PairV5CalibrationError("provenance schema is not registered")
    _slug(row["calibrator_id"], label="calibrator_id")
    for field in (
        "calibration_receipt_digest",
        "preregistration_digest",
        "frozen_generator_receipt_digest",
        "frozen_scorer_receipt_digest",
    ):
        _sha256(row[field], label=field)
    mapping_digests = row["mapping_digest_by_family"]
    if not isinstance(mapping_digests, Mapping) or not mapping_digests:
        raise PairV5CalibrationError("provenance mapping digests must be non-empty")
    for family, digest in mapping_digests.items():
        _slug(family, label="provenance action family")
        _sha256(digest, label=f"mapping digest {family}")
    if row["optimizer_authorized"] is not True:
        raise PairV5CalibrationError("provenance must authorize the optimizer")
    if row["input_semantics"] != "rv2v_candidate_own_scalar_action_energy_only":
        raise PairV5CalibrationError("provenance input semantics changed")
    if row["proposal_visual_data_consumed"] is not False or row[
        "privileged_visual_inputs_consumed"
    ] is not False:
        raise PairV5CalibrationError("provenance admits forbidden visual inputs")
    _embedded_digest(row, field="provenance_digest", label="calibrator provenance")
    return dict(row)


__all__ = [
    "ACTION_BRANCH",
    "BRANCHES",
    "FRAME_COUNT",
    "GENERATION_MODE",
    "GROUP_AXES",
    "NEGATIVE_BRANCHES",
    "PairV5CalibrationError",
    "SCORE_COORDINATE",
    "apply_calibrator",
    "calibrate_action_energy",
    "canonical_json_bytes",
    "event_qualified_positive",
    "make_calibrator_provenance",
    "make_preregistration",
    "make_score_row",
    "object_sha256",
    "score_rv2v_candidate",
    "validate_calibration_receipt",
    "validate_calibrator_provenance",
    "validate_preregistration",
    "validate_rv2v_candidate_score",
    "validate_score_row",
]
