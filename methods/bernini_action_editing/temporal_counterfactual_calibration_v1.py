#!/usr/bin/env python3
"""Fit-only thresholds for temporal-counterfactual score vectors.

Each score dimension is already a within-candidate, action-vs-no-op temporal
contrast.  This calibrator therefore fits no affine cross-video mapping.  One
threshold per action family and temporal transform is set from the fit cell;
the disjoint confirmation cell is evaluated once.  Optimizer authorization is
a conjunction of every transform threshold, prompt/direction/rank hard gates,
detached event labels, branch specificity, positive recall, and AUROC.  The
diagnostic composite scalar can never authorize an optimizer by itself.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pair_v5_t2v_energy_calibration_v3 as detached_events
import temporal_counterfactual_contract_v1 as contract


PREREGISTRATION_SCHEMA = "bernini-temporal-counterfactual-preregistration-v1"
CALIBRATION_RECEIPT_SCHEMA = "bernini-temporal-counterfactual-calibration-v1"
CALIBRATOR_ID = "pair5-core4-v2-same-video-temporal-counterfactual-v1"
ACTION_FAMILY_ORDER = (
    "dog-sit-facing-camera",
    "human-rise-to-stand",
)
GROUP_AXES = ("actor_group_id", "scene_group_id", "action_group_id")
MINIMUM_FIT_GAP = 1.0e-6
MINIMUM_CONFIRMATION_AUROC = 0.75
MINIMUM_POSITIVE_RECALL = 1.0
MINIMUM_NEGATIVE_SPECIFICITY = 1.0

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SHA1_RE = re.compile(r"[0-9a-f]{40}")


class TemporalCounterfactualCalibrationError(ValueError):
    """The score/audit population or a preregistered gate failed closed."""


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise TemporalCounterfactualCalibrationError(f"{label} field closure differs")
    return dict(value)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TemporalCounterfactualCalibrationError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise TemporalCounterfactualCalibrationError(f"{label} must be lowercase SHA-1")
    return value


def _finite(value: Any, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise TemporalCounterfactualCalibrationError(f"{label} must be finite")
    return float(value)


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(unsigned)
    return {**value, "receipt_digest": contract.object_sha256(value)}


_PREREG_FIELDS = frozenset(
    {
        "schema_version",
        "calibrator_id",
        "source_population",
        "action_family_order",
        "analysis_split_order",
        "branch_order",
        "transform_order",
        "sigma_coordinate_digest",
        "fit_threshold_rule",
        "minimum_fit_gap",
        "candidate_prediction_rule",
        "required_candidate_hard_gates",
        "minimum_confirmation_auroc",
        "minimum_positive_recall",
        "minimum_negative_specificity_by_branch",
        "fit_confirmation_group_disjointness_axes",
        "confirmation_rows_used_for_threshold_fit_or_optimizer_gradient",
        "confirmation_metrics_used_once_for_optimizer_go_nogo",
        "single_scalar_can_authorize_optimizer",
        "t2v_calibration_only_never_rv2v_condition_target_donor_or_noise",
        "preregistration_digest",
    }
)


def make_preregistration() -> dict[str, Any]:
    unsigned = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "calibrator_id": CALIBRATOR_ID,
        "source_population": "sealed_pair5_t2v_core4_v2_exact40",
        "action_family_order": list(ACTION_FAMILY_ORDER),
        "analysis_split_order": list(contract.ANALYSIS_SPLITS),
        "branch_order": list(contract.BRANCH_ORDER),
        "transform_order": list(contract.COUNTERFACTUAL_TRANSFORMS),
        "sigma_coordinate_digest": contract.make_sigma_coordinate_receipt()[
            "receipt_digest"
        ],
        "fit_threshold_rule": (
            "per_family_per_transform_midpoint_between_fit_action_and_fit_max_negative"
        ),
        "minimum_fit_gap": MINIMUM_FIT_GAP,
        "candidate_prediction_rule": (
            "all_six_transform_margins_meet_fit_thresholds_and_all_prompt_direction_rank_hard_gates"
        ),
        "required_candidate_hard_gates": [
            "chronological_action_beats_noop_at_every_sigma",
            "reverse_action_direction_hard_gate_all_sigmas",
            "reverse_prompt_specific_hard_gate_all_sigmas",
            "chronological_rank1_among_multiset_controls_all_sigmas",
            "candidate_hard_gate_passed",
        ],
        "minimum_confirmation_auroc": MINIMUM_CONFIRMATION_AUROC,
        "minimum_positive_recall": MINIMUM_POSITIVE_RECALL,
        "minimum_negative_specificity_by_branch": MINIMUM_NEGATIVE_SPECIFICITY,
        "fit_confirmation_group_disjointness_axes": list(GROUP_AXES),
        "confirmation_rows_used_for_threshold_fit_or_optimizer_gradient": False,
        "confirmation_metrics_used_once_for_optimizer_go_nogo": True,
        "single_scalar_can_authorize_optimizer": False,
        "t2v_calibration_only_never_rv2v_condition_target_donor_or_noise": True,
    }
    return {**unsigned, "preregistration_digest": contract.object_sha256(unsigned)}


def validate_preregistration(value: Any) -> dict[str, Any]:
    row = _closed(value, _PREREG_FIELDS, label="preregistration")
    if row != make_preregistration():
        raise TemporalCounterfactualCalibrationError("preregistration differs")
    return row


def _event_positive(audit: Mapping[str, Any]) -> bool:
    return bool(audit["event_qualified_action_positive"])


def _event_contract_passes(audit: Mapping[str, Any]) -> bool:
    if audit["semantic_branch"] == contract.ACTION_BRANCH:
        return bool(
            audit["complete_target_transition_observed"] is True
            and audit["terminal_hold_observed"] is True
            and audit["full_target_action_observed"] is True
            and audit["full_target_action_false_confirmed"] is False
            and audit["event_qualified_action_positive"] is True
        )
    return bool(
        audit["complete_target_transition_observed"] is False
        and audit["terminal_hold_observed"] is False
        and audit["full_target_action_observed"] is False
        and audit["full_target_action_false_confirmed"] is True
        and audit["event_qualified_action_positive"] is False
    )


def _feature(row: Mapping[str, Any], transform_name: str) -> float:
    return float(
        row["transform_contributions"][transform_name][
            "minimum_prompt_specific_chronological_margin"
        ]
    )


def _hard_gate(row: Mapping[str, Any]) -> bool:
    expected = make_preregistration()["required_candidate_hard_gates"]
    return all(row["hard_gates"].get(name) is True for name in expected)


def _predict(
    row: Mapping[str, Any], thresholds: Mapping[str, float]
) -> tuple[bool, dict[str, bool], float]:
    by_transform = {
        name: _feature(row, name) >= float(thresholds[name])
        for name in contract.COUNTERFACTUAL_TRANSFORMS
    }
    composite_margin = min(
        _feature(row, name) - float(thresholds[name])
        for name in contract.COUNTERFACTUAL_TRANSFORMS
    )
    return bool(all(by_transform.values()) and _hard_gate(row)), by_transform, float(
        composite_margin
    )


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


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    audits: Mapping[str, Mapping[str, Any]],
    thresholds_by_family: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    labels: list[bool] = []
    diagnostics: list[float] = []
    predictions: list[bool] = []
    reports: list[dict[str, Any]] = []
    for row in rows:
        identity = row["candidate_identity"]
        candidate_id = identity["candidate_id"]
        family = identity["action_family_id"]
        predicted, threshold_pass, diagnostic = _predict(
            row, thresholds_by_family[family]
        )
        label = _event_positive(audits[candidate_id])
        labels.append(label)
        diagnostics.append(diagnostic)
        predictions.append(predicted)
        reports.append(
            {
                "candidate_id": candidate_id,
                "action_family_id": family,
                "semantic_branch": identity["semantic_branch"],
                "event_positive": label,
                "transform_margin_by_name": {
                    name: _feature(row, name)
                    for name in contract.COUNTERFACTUAL_TRANSFORMS
                },
                "transform_threshold_pass_by_name": threshold_pass,
                "all_candidate_hard_gates_passed": _hard_gate(row),
                "predicted_positive": predicted,
                "diagnostic_conjunctive_margin": diagnostic,
                "diagnostic_original_composite_score": row[
                    "diagnostic_composite_score"
                ],
                "single_scalar_used_as_prediction_rule": False,
                "score_receipt_digest": row["receipt_digest"],
                "event_audit_receipt_digest": audits[candidate_id]["receipt_digest"],
            }
        )
    positive_predictions = [
        prediction
        for label, prediction in zip(labels, predictions)
        if label
    ]
    recall = (
        sum(positive_predictions) / len(positive_predictions)
        if positive_predictions
        else 0.0
    )
    specificity = {}
    for branch in contract.NEGATIVE_BRANCHES:
        branch_predictions = [
            prediction
            for row, prediction in zip(rows, predictions)
            if row["candidate_identity"]["semantic_branch"] == branch
        ]
        specificity[branch] = (
            sum(not prediction for prediction in branch_predictions)
            / len(branch_predictions)
            if branch_predictions
            else 0.0
        )
    return {
        "row_count": len(rows),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "diagnostic_composite_auroc": _auroc(labels, diagnostics),
        "positive_recall": float(recall),
        "negative_specificity_by_branch": specificity,
        "candidate_reports": reports,
    }


def _metrics_pass(value: Mapping[str, Any]) -> bool:
    return bool(
        value["diagnostic_composite_auroc"] >= MINIMUM_CONFIRMATION_AUROC
        and value["positive_recall"] >= MINIMUM_POSITIVE_RECALL
        and all(
            value["negative_specificity_by_branch"][branch]
            >= MINIMUM_NEGATIVE_SPECIFICITY
            for branch in contract.NEGATIVE_BRANCHES
        )
    )


_CALIBRATION_FIELDS = frozenset(
    {
        "schema_version",
        "calibrator_id",
        "preregistration_digest",
        "source_bank_spec_sha256",
        "source_bank_receipt_digest",
        "source_bank_receipt_file_sha256",
        "source_group_receipt_digests_by_id",
        "source_group_candidate_order_digest",
        "scoring_source_binding",
        "calibrator_source_binding",
        "score_receipt_count",
        "event_audit_receipt_count",
        "fit_cell_count",
        "confirmation_cell_count",
        "action_family_order",
        "branch_order",
        "transform_order",
        "sigma_coordinate_digest",
        "score_receipt_set_digest",
        "event_audit_receipt_set_digest",
        "fit_thresholds_by_family_and_transform",
        "fit_metrics",
        "confirmation_metrics",
        "transform_contribution_report",
        "gates",
        "failure_reasons",
        "optimizer_authorization_rule",
        "diagnostic_composite_auroc_is_sufficient_for_optimizer",
        "single_scalar_authorizes_optimizer",
        "confirmation_rows_used_for_threshold_fit_or_optimizer_gradient",
        "confirmation_metrics_used_once_for_optimizer_go_nogo",
        "t2v_media_or_latent_consumed_by_calibrator",
        "t2v_media_or_latent_may_be_rv2v_condition_target_donor_or_noise",
        "training_performed",
        "optimizer_authorized",
        "scientific_action_editing_claim",
        "receipt_digest",
    }
)


def calibrate_temporal_counterfactual_scores(
    score_receipts: Sequence[Mapping[str, Any]],
    event_audit_receipts: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    group_receipts: Sequence[Mapping[str, Any]],
    *,
    source_bank_spec_sha256: str,
    source_bank_receipt_digest: str,
    calibrator_source_revision: str,
    calibrator_source_archive_sha256: str,
    expected_calibrator_source_sha256: str,
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    source_spec = _sha256(source_bank_spec_sha256, label="bank spec SHA-256")
    source_bank = _sha256(source_bank_receipt_digest, label="bank receipt digest")
    calibration_revision = _sha1(
        calibrator_source_revision, label="calibrator source revision"
    )
    calibration_archive_sha = _sha256(
        calibrator_source_archive_sha256,
        label="calibrator source archive SHA-256",
    )
    calibration_source_sha = _sha256(
        expected_calibrator_source_sha256,
        label="calibrator source SHA-256",
    )
    if (
        source_spec != contract.REQUIRED_CORE4_V2_SPEC_SHA256
        or source_bank != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST
        or calibration_source_sha
        != contract.file_sha256(Path(__file__).resolve())
    ):
        raise TemporalCounterfactualCalibrationError(
            "formal bank or calibrator source authority differs"
        )
    try:
        rows = [contract.validate_candidate_score_receipt(row) for row in score_receipts]
        groups = [contract.validate_group_receipt(row) for row in group_receipts]
        audits_list = [
            detached_events.validate_event_audit_receipt(row)
            for row in event_audit_receipts
        ]
    except (contract.TemporalCounterfactualContractError, detached_events.PairV5EnergyCalibrationV3Error) as error:
        raise TemporalCounterfactualCalibrationError(str(error)) from error
    if len(rows) != 40 or len(audits_list) != 40:
        raise TemporalCounterfactualCalibrationError("calibration requires exact 40/40 closure")
    score_by_id = {
        row["candidate_identity"]["candidate_id"]: row for row in rows
    }
    if len(score_by_id) != 40:
        raise TemporalCounterfactualCalibrationError("score candidate IDs repeat")
    group_by_id = {row["group_id"]: row for row in groups}
    if len(groups) != 2 or set(group_by_id) != {"sp4-a", "sp4-b"}:
        raise TemporalCounterfactualCalibrationError(
            "calibration requires exact sp4-a/sp4-b group receipts"
        )
    candidate_ids = []
    validated_groups = {}
    for group_id in ("sp4-a", "sp4-b"):
        group = group_by_id[group_id]
        try:
            joined = [score_by_id[candidate_id] for candidate_id in group["candidate_order"]]
        except KeyError as error:
            raise TemporalCounterfactualCalibrationError(
                "group receipt references an absent score candidate"
            ) from error
        validated_groups[group_id] = contract.validate_group_receipt(
            group, candidate_receipts=joined
        )
        candidate_ids.extend(group["candidate_order"])
    if (
        len(candidate_ids) != 40
        or len(set(candidate_ids)) != 40
        or set(candidate_ids) != set(score_by_id)
        or contract.object_sha256(candidate_ids)
        != contract.REQUIRED_CORE4_V2_CANDIDATE_ORDER_DIGEST
        or contract.object_sha256(
            [
                score_by_id[candidate_id]["candidate_identity"]
                for candidate_id in candidate_ids
            ]
        )
        != contract.REQUIRED_CORE4_V2_CANDIDATE_IDENTITY_DIGEST
    ):
        raise TemporalCounterfactualCalibrationError(
            "formal core4-v2 group/candidate order join differs"
        )
    rows = [score_by_id[candidate_id] for candidate_id in candidate_ids]
    audits = {row["candidate_id"]: row for row in audits_list}
    if len(audits) != 40 or set(audits) != set(candidate_ids):
        raise TemporalCounterfactualCalibrationError("score/audit population differs")
    if any(
        row["root_spec_raw_sha256"] != source_spec
        or row["bank_receipt_digest"] != source_bank
        or row["bank_receipt_file_sha256"]
        != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
        for row in rows
    ):
        raise TemporalCounterfactualCalibrationError("score bank authority differs")
    if any(
        group["root_spec_raw_sha256"] != source_spec
        or group["bank_receipt_digest"] != source_bank
        or group["bank_receipt_file_sha256"]
        != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
        for group in validated_groups.values()
    ):
        raise TemporalCounterfactualCalibrationError("group bank authority differs")
    scoring_source_bindings = {
        contract.object_sha256(
            {
                "method_source_revision": group["method_source_revision"],
                "method_source_archive_sha256": group[
                    "method_source_archive_sha256"
                ],
                "scorer_source_sha256": group["scorer_source_sha256"],
                "contract_source_sha256": group["contract_source_sha256"],
            }
        )
        for group in validated_groups.values()
    }
    if len(scoring_source_bindings) != 1:
        raise TemporalCounterfactualCalibrationError(
            "SP4 scoring source authority drifted"
        )
    first_group = validated_groups["sp4-a"]
    scoring_source_binding = {
        "method_source_revision": first_group["method_source_revision"],
        "method_source_archive_sha256": first_group[
            "method_source_archive_sha256"
        ],
        "scorer_source_sha256": first_group["scorer_source_sha256"],
        "contract_source_sha256": first_group["contract_source_sha256"],
    }
    method_root = Path(__file__).resolve().parent
    if (
        scoring_source_binding["scorer_source_sha256"]
        != contract.file_sha256(
            method_root / "temporal_counterfactual_action_scorer_v1.py"
        )
        or scoring_source_binding["contract_source_sha256"]
        != contract.file_sha256(
            method_root / "temporal_counterfactual_contract_v1.py"
        )
    ):
        raise TemporalCounterfactualCalibrationError(
            "group scoring source hashes do not match loaded v1 sources"
        )
    calibrator_source_binding = {
        "method_source_revision": calibration_revision,
        "method_source_archive_sha256": calibration_archive_sha,
        "calibrator_source_sha256": calibration_source_sha,
    }

    failures: list[str] = []
    identity_fields = (
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        *GROUP_AXES,
        "semantic_branch",
    )
    event_contract_passed = True
    for row in rows:
        identity = row["candidate_identity"]
        audit = audits[identity["candidate_id"]]
        if (
            any(identity[name] != audit[name] for name in identity_fields)
            or row["generation_binding"]["generation_receipt_digest"]
            != audit["generation_receipt_digest"]
        ):
            raise TemporalCounterfactualCalibrationError("score/audit identity differs")
        if not _event_contract_passes(audit):
            event_contract_passed = False
            failures.append(f"event_audit:{identity['candidate_id']}")

    # Exact four-cell, ten-branch topology.
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        identity = row["candidate_identity"]
        key = (
            identity["analysis_split"],
            identity["action_family_id"],
            identity["calibration_group_id"],
        )
        cells.setdefault(key, []).append(row)
    coverage_complete = bool(
        len(cells) == 4
        and {
            (split, family)
            for split, family, _cell in cells
        }
        == {
            (split, family)
            for split in contract.ANALYSIS_SPLITS
            for family in ACTION_FAMILY_ORDER
        }
        and all(
            len(cell_rows) == len(contract.BRANCH_ORDER)
            and [
                row["candidate_identity"]["semantic_branch"]
                for row in cell_rows
            ]
            == list(contract.BRANCH_ORDER)
            for cell_rows in cells.values()
        )
    )
    if not coverage_complete:
        failures.append("exact_four_cell_branch_coverage")

    target_binding_closed = True
    prompt_pair_binding_closed = True
    same_cell_gaussian_closed = True
    for cell_rows in cells.values():
        action_rows = [
            row
            for row in cell_rows
            if row["candidate_identity"]["semantic_branch"] == contract.ACTION_BRANCH
        ]
        if len(action_rows) != 1:
            target_binding_closed = False
            continue
        action_id = action_rows[0]["candidate_identity"]["candidate_id"]
        noop_rows = [
            row
            for row in cell_rows
            if row["candidate_identity"]["semantic_branch"] == "noop"
        ]
        if len(noop_rows) != 1:
            target_binding_closed = False
            continue
        noop_id = noop_rows[0]["candidate_identity"]["candidate_id"]
        expected = action_rows[0]["target_action_binding"]
        if (
            expected["target_action_candidate_id"] != action_id
            or expected["target_noop_candidate_id"] != noop_id
            or expected["target_action_caption_utf8_sha256"]
            != action_rows[0]["generation_binding"][
                "candidate_own_caption_utf8_sha256"
            ]
            or expected["target_noop_caption_utf8_sha256"]
            != noop_rows[0]["generation_binding"][
                "candidate_own_caption_utf8_sha256"
            ]
            or any(row["target_action_binding"] != expected for row in cell_rows)
        ):
            target_binding_closed = False
        expected_prompt_pair = action_rows[0]["prompt_binding"]["prompt_pair_digest"]
        if any(
            row["prompt_binding"]["prompt_pair_digest"] != expected_prompt_pair
            for row in cell_rows
        ):
            prompt_pair_binding_closed = False
        if len(
            {
                row["generation_binding"]["official_gaussian_tensor_sha256"]
                for row in cell_rows
            }
        ) != 1:
            same_cell_gaussian_closed = False
    if not target_binding_closed:
        failures.append("cell_fixed_target_action_binding")
    if not prompt_pair_binding_closed:
        failures.append("cell_fixed_action_noop_prompt_pair")
    if not same_cell_gaussian_closed:
        failures.append("same_cell_official_gaussian")

    disjoint_by_axis = {}
    for axis in GROUP_AXES:
        fit_values = {
            row["candidate_identity"][axis]
            for row in rows
            if row["candidate_identity"]["analysis_split"] == "fit"
        }
        confirmation_values = {
            row["candidate_identity"][axis]
            for row in rows
            if row["candidate_identity"]["analysis_split"] == "confirmation"
        }
        disjoint_by_axis[axis] = not bool(fit_values & confirmation_values)
        if not disjoint_by_axis[axis]:
            failures.append(f"fit_confirmation_overlap:{axis}")

    common_model_authority = len(
        {contract.object_sha256(row["model_binding"]) for row in rows}
    ) == 1
    if not common_model_authority:
        failures.append("frozen_model_authority_drift")

    thresholds: dict[str, dict[str, float]] = {}
    separation: dict[str, dict[str, bool]] = {}
    for family in ACTION_FAMILY_ORDER:
        fit_rows = [
            row
            for row in rows
            if row["candidate_identity"]["analysis_split"] == "fit"
            and row["candidate_identity"]["action_family_id"] == family
        ]
        positives = [
            row for row in fit_rows if _event_positive(audits[row["candidate_identity"]["candidate_id"]])
        ]
        negatives = [
            row
            for row in fit_rows
            if not _event_positive(audits[row["candidate_identity"]["candidate_id"]])
        ]
        thresholds[family] = {}
        separation[family] = {}
        for transform_name in contract.COUNTERFACTUAL_TRANSFORMS:
            positive_anchor = min((_feature(row, transform_name) for row in positives), default=0.0)
            negative_anchor = max((_feature(row, transform_name) for row in negatives), default=0.0)
            separated = bool(
                positives
                and negatives
                and positive_anchor - negative_anchor >= prereg["minimum_fit_gap"]
            )
            separation[family][transform_name] = separated
            thresholds[family][transform_name] = float(
                0.5 * (positive_anchor + negative_anchor)
            )
            if not separated:
                failures.append(f"fit_separation:{family}:{transform_name}")

    fit_rows_all = [
        row for row in rows if row["candidate_identity"]["analysis_split"] == "fit"
    ]
    confirmation_rows = [
        row
        for row in rows
        if row["candidate_identity"]["analysis_split"] == "confirmation"
    ]
    fit_metrics = {
        "overall": _metrics(fit_rows_all, audits, thresholds),
        "by_family": {
            family: _metrics(
                [
                    row
                    for row in fit_rows_all
                    if row["candidate_identity"]["action_family_id"] == family
                ],
                audits,
                thresholds,
            )
            for family in ACTION_FAMILY_ORDER
        },
    }
    confirmation_metrics = {
        "overall": _metrics(confirmation_rows, audits, thresholds),
        "by_family": {
            family: _metrics(
                [
                    row
                    for row in confirmation_rows
                    if row["candidate_identity"]["action_family_id"] == family
                ],
                audits,
                thresholds,
            )
            for family in ACTION_FAMILY_ORDER
        },
    }
    fit_metrics_passed = bool(
        _metrics_pass(fit_metrics["overall"])
        and all(_metrics_pass(value) for value in fit_metrics["by_family"].values())
    )
    confirmation_metrics_passed = bool(
        _metrics_pass(confirmation_metrics["overall"])
        and all(
            _metrics_pass(value)
            for value in confirmation_metrics["by_family"].values()
        )
    )
    if not fit_metrics_passed:
        failures.append("fit_vector_classification")
    if not confirmation_metrics_passed:
        failures.append("confirmation_vector_metrics")

    positive_hard_gates = {
        split: all(
            _hard_gate(row)
            for row in rows
            if row["candidate_identity"]["analysis_split"] == split
            and _event_positive(audits[row["candidate_identity"]["candidate_id"]])
        )
        for split in contract.ANALYSIS_SPLITS
    }
    for split, passed in positive_hard_gates.items():
        if not passed:
            failures.append(f"positive_candidate_hard_gates:{split}")

    gates = {
        "exact_four_cell_branch_coverage": coverage_complete,
        "detached_event_contract_complete": event_contract_passed,
        "cell_fixed_target_action_binding": target_binding_closed,
        "cell_fixed_action_noop_prompt_pair": prompt_pair_binding_closed,
        "same_cell_official_gaussian": same_cell_gaussian_closed,
        "fit_confirmation_group_disjoint_by_axis": disjoint_by_axis,
        "single_frozen_model_authority": common_model_authority,
        "fit_separation_by_family_and_transform": separation,
        "fit_vector_classification": fit_metrics_passed,
        "positive_candidate_hard_gates_by_split": positive_hard_gates,
        "confirmation_positive_recall_one": confirmation_metrics["overall"][
            "positive_recall"
        ]
        >= MINIMUM_POSITIVE_RECALL,
        "confirmation_every_negative_branch_specificity_one": all(
            confirmation_metrics["overall"]["negative_specificity_by_branch"][branch]
            >= MINIMUM_NEGATIVE_SPECIFICITY
            for branch in contract.NEGATIVE_BRANCHES
        ),
        "confirmation_auroc_at_least_point75": confirmation_metrics["overall"][
            "diagnostic_composite_auroc"
        ]
        >= MINIMUM_CONFIRMATION_AUROC,
        "confirmation_vector_metrics_overall_and_by_family": confirmation_metrics_passed,
        "t2v_is_calibration_only": True,
        "single_scalar_authorization_forbidden": True,
    }
    optimizer_authorized = bool(
        coverage_complete
        and event_contract_passed
        and target_binding_closed
        and prompt_pair_binding_closed
        and same_cell_gaussian_closed
        and all(disjoint_by_axis.values())
        and common_model_authority
        and all(all(values.values()) for values in separation.values())
        and fit_metrics_passed
        and all(positive_hard_gates.values())
        and confirmation_metrics_passed
    )
    unique_failures = sorted(set(failures))
    if optimizer_authorized == bool(unique_failures):
        raise TemporalCounterfactualCalibrationError("gate/failure closure differs")

    contribution_report = {
        transform_name: {
            split: {
                family: [
                    {
                        "candidate_id": row["candidate_identity"]["candidate_id"],
                        "semantic_branch": row["candidate_identity"]["semantic_branch"],
                        "minimum_prompt_specific_chronological_margin": _feature(
                            row, transform_name
                        ),
                        "score_receipt_digest": row["receipt_digest"],
                    }
                    for row in rows
                    if row["candidate_identity"]["analysis_split"] == split
                    and row["candidate_identity"]["action_family_id"] == family
                ]
                for family in ACTION_FAMILY_ORDER
            }
            for split in contract.ANALYSIS_SPLITS
        }
        for transform_name in contract.COUNTERFACTUAL_TRANSFORMS
    }
    unsigned = {
        "schema_version": CALIBRATION_RECEIPT_SCHEMA,
        "calibrator_id": CALIBRATOR_ID,
        "preregistration_digest": prereg["preregistration_digest"],
        "source_bank_spec_sha256": source_spec,
        "source_bank_receipt_digest": source_bank,
        "source_bank_receipt_file_sha256": (
            contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
        ),
        "source_group_receipt_digests_by_id": {
            group_id: validated_groups[group_id]["receipt_digest"]
            for group_id in ("sp4-a", "sp4-b")
        },
        "source_group_candidate_order_digest": contract.object_sha256(
            candidate_ids
        ),
        "scoring_source_binding": scoring_source_binding,
        "calibrator_source_binding": calibrator_source_binding,
        "score_receipt_count": len(rows),
        "event_audit_receipt_count": len(audits_list),
        "fit_cell_count": sum(key[0] == "fit" for key in cells),
        "confirmation_cell_count": sum(key[0] == "confirmation" for key in cells),
        "action_family_order": list(ACTION_FAMILY_ORDER),
        "branch_order": list(contract.BRANCH_ORDER),
        "transform_order": list(contract.COUNTERFACTUAL_TRANSFORMS),
        "sigma_coordinate_digest": contract.make_sigma_coordinate_receipt()[
            "receipt_digest"
        ],
        "score_receipt_set_digest": contract.object_sha256(
            [row["receipt_digest"] for row in rows]
        ),
        "event_audit_receipt_set_digest": contract.object_sha256(
            [audits[candidate_id]["receipt_digest"] for candidate_id in candidate_ids]
        ),
        "fit_thresholds_by_family_and_transform": thresholds,
        "fit_metrics": fit_metrics,
        "confirmation_metrics": confirmation_metrics,
        "transform_contribution_report": contribution_report,
        "gates": gates,
        "failure_reasons": unique_failures,
        "optimizer_authorization_rule": (
            "conjunction_of_six_transform_thresholds_prompt_noop_reverse_rank_event_fit_and_confirmation_gates"
        ),
        "diagnostic_composite_auroc_is_sufficient_for_optimizer": False,
        "single_scalar_authorizes_optimizer": False,
        "confirmation_rows_used_for_threshold_fit_or_optimizer_gradient": False,
        "confirmation_metrics_used_once_for_optimizer_go_nogo": True,
        "t2v_media_or_latent_consumed_by_calibrator": False,
        "t2v_media_or_latent_may_be_rv2v_condition_target_donor_or_noise": False,
        "training_performed": False,
        "optimizer_authorized": optimizer_authorized,
        "scientific_action_editing_claim": False,
    }
    return _seal(unsigned)


def validate_calibration_receipt(
    value: Any,
    *,
    score_receipts: Sequence[Mapping[str, Any]] | None = None,
    event_audit_receipts: Sequence[Mapping[str, Any]] | None = None,
    preregistration: Mapping[str, Any] | None = None,
    group_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a receipt, replaying every input before accepting GO authority.

    A sealed JSON object is not an authority by itself: anyone able to rewrite
    JSON can recompute its digest.  NO-GO receipts may be inspected standalone,
    but an ``optimizer_authorized=true`` receipt is valid only when the exact
    40 score receipts, 40 detached audits, two SP4 group receipts, and
    preregistration are supplied and deterministically reproduce it.
    """

    row = _closed(value, _CALIBRATION_FIELDS, label="calibration receipt")
    unsigned = dict(row)
    digest = _sha256(unsigned.pop("receipt_digest"), label="calibration digest")
    if contract.object_sha256(unsigned) != digest:
        raise TemporalCounterfactualCalibrationError("calibration digest differs")
    if type(row["optimizer_authorized"]) is not bool:
        raise TemporalCounterfactualCalibrationError(
            "optimizer authority must be a JSON boolean"
        )
    failure_reasons = row["failure_reasons"]
    if (
        type(failure_reasons) is not list
        or any(type(reason) is not str or not reason for reason in failure_reasons)
        or failure_reasons != sorted(set(failure_reasons))
    ):
        raise TemporalCounterfactualCalibrationError(
            "failure reasons must be a sorted unique list of nonempty strings"
        )
    group_digests = _closed(
        row["source_group_receipt_digests_by_id"],
        {"sp4-a", "sp4-b"},
        label="source group receipt digests",
    )
    for group_id, group_digest in group_digests.items():
        _sha256(group_digest, label=f"{group_id} group receipt digest")
    scoring_source = _closed(
        row["scoring_source_binding"],
        {
            "method_source_revision",
            "method_source_archive_sha256",
            "scorer_source_sha256",
            "contract_source_sha256",
        },
        label="scoring source binding",
    )
    _sha1(scoring_source["method_source_revision"], label="scoring source revision")
    for name in (
        "method_source_archive_sha256",
        "scorer_source_sha256",
        "contract_source_sha256",
    ):
        _sha256(scoring_source[name], label=f"scoring {name}")
    calibrator_source = _closed(
        row["calibrator_source_binding"],
        {
            "method_source_revision",
            "method_source_archive_sha256",
            "calibrator_source_sha256",
        },
        label="calibrator source binding",
    )
    _sha1(
        calibrator_source["method_source_revision"],
        label="calibrator source revision",
    )
    _sha256(
        calibrator_source["method_source_archive_sha256"],
        label="calibrator archive SHA-256",
    )
    _sha256(
        calibrator_source["calibrator_source_sha256"],
        label="calibrator source SHA-256",
    )
    if (
        row["schema_version"] != CALIBRATION_RECEIPT_SCHEMA
        or row["calibrator_id"] != CALIBRATOR_ID
        or row["preregistration_digest"]
        != make_preregistration()["preregistration_digest"]
        or row["score_receipt_count"] != 40
        or row["event_audit_receipt_count"] != 40
        or row["fit_cell_count"] != 2
        or row["confirmation_cell_count"] != 2
        or row["action_family_order"] != list(ACTION_FAMILY_ORDER)
        or row["branch_order"] != list(contract.BRANCH_ORDER)
        or row["transform_order"] != list(contract.COUNTERFACTUAL_TRANSFORMS)
        or row["sigma_coordinate_digest"]
        != contract.make_sigma_coordinate_receipt()["receipt_digest"]
        or row["source_bank_spec_sha256"]
        != contract.REQUIRED_CORE4_V2_SPEC_SHA256
        or row["source_bank_receipt_digest"]
        != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST
        or row["source_bank_receipt_file_sha256"]
        != contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
        or row["source_group_candidate_order_digest"]
        != contract.REQUIRED_CORE4_V2_CANDIDATE_ORDER_DIGEST
        or set(row["source_group_receipt_digests_by_id"])
        != {"sp4-a", "sp4-b"}
        or row["diagnostic_composite_auroc_is_sufficient_for_optimizer"] is not False
        or row["single_scalar_authorizes_optimizer"] is not False
        or row[
            "confirmation_rows_used_for_threshold_fit_or_optimizer_gradient"
        ]
        is not False
        or row["confirmation_metrics_used_once_for_optimizer_go_nogo"] is not True
        or row["t2v_media_or_latent_consumed_by_calibrator"] is not False
        or row[
            "t2v_media_or_latent_may_be_rv2v_condition_target_donor_or_noise"
        ]
        is not False
        or row["training_performed"] is not False
        or row["scientific_action_editing_claim"] is not False
        or row["optimizer_authorized"] == bool(row["failure_reasons"])
    ):
        raise TemporalCounterfactualCalibrationError("calibration semantics differ")
    replay_values = (
        score_receipts,
        event_audit_receipts,
        preregistration,
        group_receipts,
    )
    replay_supplied = [item is not None for item in replay_values]
    if any(replay_supplied) and not all(replay_supplied):
        raise TemporalCounterfactualCalibrationError(
            "calibration replay requires scores, audits, preregistration, and groups together"
        )
    if row["optimizer_authorized"] is True and not all(replay_supplied):
        raise TemporalCounterfactualCalibrationError(
            "GO receipt requires exact score/audit/preregistration replay"
        )
    if all(replay_supplied):
        expected = calibrate_temporal_counterfactual_scores(
            score_receipts,  # type: ignore[arg-type]
            event_audit_receipts,  # type: ignore[arg-type]
            preregistration,  # type: ignore[arg-type]
            group_receipts,  # type: ignore[arg-type]
            source_bank_spec_sha256=row["source_bank_spec_sha256"],
            source_bank_receipt_digest=row["source_bank_receipt_digest"],
            calibrator_source_revision=row["calibrator_source_binding"][
                "method_source_revision"
            ],
            calibrator_source_archive_sha256=row["calibrator_source_binding"][
                "method_source_archive_sha256"
            ],
            expected_calibrator_source_sha256=row["calibrator_source_binding"][
                "calibrator_source_sha256"
            ],
        )
        if row != expected:
            raise TemporalCounterfactualCalibrationError(
                "calibration receipt does not reproduce from exact inputs"
            )
    return row


__all__ = [
    "ACTION_FAMILY_ORDER",
    "CALIBRATION_RECEIPT_SCHEMA",
    "CALIBRATOR_ID",
    "MINIMUM_CONFIRMATION_AUROC",
    "MINIMUM_NEGATIVE_SPECIFICITY",
    "MINIMUM_POSITIVE_RECALL",
    "PREREGISTRATION_SCHEMA",
    "TemporalCounterfactualCalibrationError",
    "calibrate_temporal_counterfactual_scores",
    "make_preregistration",
    "validate_calibration_receipt",
    "validate_preregistration",
]
