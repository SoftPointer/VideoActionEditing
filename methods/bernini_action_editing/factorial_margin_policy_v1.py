#!/usr/bin/env python3
"""Fit-only authoring and frozen evaluation of factorial action margins.

The authoring input contains only prospective ``fit`` and ``calibration``
cells.  Each cell shares one source and seed across seven typed branches.  A
candidate representation is eligible only when, in every fit cell, the
forward action score is strictly above every negative branch and every
preservation axis is no worse than the matched noop branch.  These checks are
conjunctive: action gain cannot compensate identity, camera, background,
owner, or quality regression.

The fit winner is chosen lexicographically from the worst observed margins.
The calibration split cannot reselect the winner.  It only freezes one
threshold per action comparison and one absolute/relative threshold per
preservation axis using the worst observed calibration cell.  If any
calibration action margin is non-positive or any preservation delta is
negative, the result is a sealed zero-update receipt.

Confirmation is evaluated by a separate entry point.  It requires a frozen
policy digest and source/media disjointness, never changes a threshold, and
does not authorize a method-success claim.  The intentionally conservative
worst-observed rule is a canary contract, not a universal numerical threshold.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


POPULATION_SCHEMA = "bernini-factorial-margin-population-v1"
SCORE_CONTRACT_SCHEMA = "bernini-factorial-margin-score-contract-v1"
POLICY_SCHEMA = "bernini-factorial-margin-frozen-policy-v1"
CONFIRMATION_SCHEMA = "bernini-factorial-margin-confirmation-v1"
CONFIRMATION_RECEIPT_SCHEMA = "bernini-factorial-margin-confirmation-receipt-v1"

BRANCHES = (
    "forward",
    "noop",
    "reverse",
    "incomplete",
    "camera_only",
    "appearance_only",
    "wrong_actor_or_object",
)
NEGATIVE_BRANCHES = BRANCHES[1:]
CORE_NEGATIVES = ("noop", "reverse")
PRESERVATION_AXES = ("identity", "camera", "background", "owner", "quality")
METRIC_FIELDS = ("action", *PRESERVATION_AXES)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

_POPULATION_FIELDS = frozenset(
    {
        "schema_version",
        "population_id",
        "created_utc",
        "score_contract",
        "confirmation_registry_sha256",
        "fit_cells",
        "calibration_cells",
    }
)
_SCORE_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "evaluator_sha256",
        "action_score_semantics",
        "preservation_score_semantics",
        "score_range",
        "higher_is_better",
        "branch_order",
        "preservation_axis_order",
    }
)
_CELL_FIELDS = frozenset(
    {
        "cell_id",
        "source_id",
        "source_media_sha256",
        "seed",
        "action_family",
        "candidate_scores",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "population_id",
        "population_digest",
        "score_contract",
        "score_contract_digest",
        "confirmation_registry_sha256",
        "fit_source_ids",
        "fit_source_media_sha256s",
        "calibration_source_ids",
        "calibration_source_media_sha256s",
        "fit_candidate_summaries",
        "selected_candidate_id",
        "calibration_gate_policy",
        "optimizer_step_allowed",
        "confirmation_scores_consumed",
        "method_success_claimed",
        "policy_digest",
    }
)
_GATE_FIELDS = frozenset(
    {
        "candidate_id",
        "calibration_rule",
        "calibration_cell_ids",
        "action_margin_minimums",
        "preservation_absolute_minimums",
        "preservation_delta_vs_noop_minimums",
        "all_calibration_action_margins_strictly_positive",
        "all_calibration_preservation_deltas_nonnegative",
    }
)
_FIT_SUMMARY_FIELDS = frozenset(
    {
        "candidate_id",
        "fit_cell_ids",
        "minimum_action_margins",
        "minimum_core_forward_vs_noop_reverse_margin",
        "minimum_all_negative_margin",
        "minimum_forward_preservation_scores",
        "minimum_preservation_delta_vs_noop",
        "minimum_all_preservation_deltas",
        "all_action_margins_strictly_positive",
        "all_preservation_deltas_nonnegative",
        "fit_eligible",
    }
)
_CONFIRMATION_FIELDS = frozenset(
    {
        "schema_version",
        "policy_digest",
        "score_contract",
        "confirmation_registry_sha256",
        "cells",
    }
)


class FactorialMarginError(ValueError):
    """A factorial population, frozen policy, or evaluation violated closure."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FactorialMarginError("value is not finite canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorialMarginError(f"{label} must be a mapping")
    return value


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    row = _mapping(value, label=label)
    if set(row) != fields:
        raise FactorialMarginError(f"{label} field closure differs")
    return row


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise FactorialMarginError(f"{label} must be a sealed identifier")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise FactorialMarginError(f"{label} must be lowercase SHA-256")
    return value


def _sequence(value: Any, *, label: str, nonempty: bool = True) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FactorialMarginError(f"{label} must be a sequence")
    if nonempty and not value:
        raise FactorialMarginError(f"{label} must be non-empty")
    return value


def _unit_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactorialMarginError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise FactorialMarginError(f"{label} must be finite in [0,1]")
    return result


def validate_score_contract(value: Any) -> dict[str, Any]:
    row = _closed(value, _SCORE_CONTRACT_FIELDS, label="score contract")
    if row["schema_version"] != SCORE_CONTRACT_SCHEMA:
        raise FactorialMarginError("score contract schema differs")
    _sha256(row["evaluator_sha256"], label="evaluator SHA-256")
    for field in ("action_score_semantics", "preservation_score_semantics"):
        if type(row[field]) is not str or not row[field].strip() or "\x00" in row[field]:
            raise FactorialMarginError(f"{field} differs")
    if row["score_range"] != [0.0, 1.0] or row["higher_is_better"] is not True:
        raise FactorialMarginError("score scale/direction differs")
    if row["branch_order"] != list(BRANCHES):
        raise FactorialMarginError("branch order differs")
    if row["preservation_axis_order"] != list(PRESERVATION_AXES):
        raise FactorialMarginError("preservation axis order differs")
    return dict(row)


def _validate_branch_metrics(value: Any, *, label: str) -> dict[str, float]:
    row = _mapping(value, label=label)
    if set(row) != set(METRIC_FIELDS):
        raise FactorialMarginError(f"{label} metric closure differs")
    return {field: _unit_score(row[field], label=f"{label}.{field}") for field in METRIC_FIELDS}


def _validate_candidate_scores(value: Any, *, label: str) -> dict[str, dict[str, dict[str, float]]]:
    candidates = _mapping(value, label=label)
    if not candidates:
        raise FactorialMarginError(f"{label} must contain candidates")
    result: dict[str, dict[str, dict[str, float]]] = {}
    for candidate_id, raw_branches in candidates.items():
        candidate = _identifier(candidate_id, label="candidate ID")
        branches = _mapping(raw_branches, label=f"{label}.{candidate}")
        if set(branches) != set(BRANCHES):
            raise FactorialMarginError(f"{label}.{candidate} branch closure differs")
        result[candidate] = {
            branch: _validate_branch_metrics(
                branches[branch], label=f"{label}.{candidate}.{branch}"
            )
            for branch in BRANCHES
        }
    return result


def validate_cells(value: Any, *, label: str) -> list[dict[str, Any]]:
    raw_cells = _sequence(value, label=label)
    result: list[dict[str, Any]] = []
    cell_ids: set[str] = set()
    source_seed_pairs: set[tuple[str, int]] = set()
    candidate_ids: tuple[str, ...] | None = None
    for index, raw in enumerate(raw_cells):
        row = _closed(raw, _CELL_FIELDS, label=f"{label}[{index}]")
        cell_id = _identifier(row["cell_id"], label="cell ID")
        source_id = _identifier(row["source_id"], label="source ID")
        source_hash = _sha256(row["source_media_sha256"], label="source media SHA-256")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int) or not 0 <= row["seed"] < 2**63:
            raise FactorialMarginError("seed must be a non-negative signed-63 integer")
        family = _identifier(row["action_family"], label="action family")
        scores = _validate_candidate_scores(
            row["candidate_scores"], label=f"{label}[{index}].candidate_scores"
        )
        current_candidates = tuple(sorted(scores))
        if candidate_ids is None:
            candidate_ids = current_candidates
        elif current_candidates != candidate_ids:
            raise FactorialMarginError(f"{label} candidate registry differs across cells")
        if cell_id in cell_ids:
            raise FactorialMarginError(f"duplicate {label} cell ID")
        source_seed = (source_id, row["seed"])
        if source_seed in source_seed_pairs:
            raise FactorialMarginError(f"duplicate {label} source/seed cell")
        cell_ids.add(cell_id)
        source_seed_pairs.add(source_seed)
        result.append(
            {
                "cell_id": cell_id,
                "source_id": source_id,
                "source_media_sha256": source_hash,
                "seed": row["seed"],
                "action_family": family,
                "candidate_scores": scores,
            }
        )
    return result


def validate_population(value: Any) -> dict[str, Any]:
    row = _closed(value, _POPULATION_FIELDS, label="population")
    if row["schema_version"] != POPULATION_SCHEMA:
        raise FactorialMarginError("population schema differs")
    population_id = _identifier(row["population_id"], label="population ID")
    if type(row["created_utc"]) is not str or not row["created_utc"].endswith("Z"):
        raise FactorialMarginError("created_utc must be an explicit UTC string")
    contract = validate_score_contract(row["score_contract"])
    registry = _sha256(
        row["confirmation_registry_sha256"], label="confirmation registry SHA-256"
    )
    fit = validate_cells(row["fit_cells"], label="fit cells")
    calibration = validate_cells(row["calibration_cells"], label="calibration cells")
    fit_candidates = tuple(sorted(fit[0]["candidate_scores"]))
    calibration_candidates = tuple(sorted(calibration[0]["candidate_scores"]))
    if fit_candidates != calibration_candidates:
        raise FactorialMarginError("fit/calibration candidate registries differ")
    fit_ids = {cell["source_id"] for cell in fit}
    calibration_ids = {cell["source_id"] for cell in calibration}
    fit_hashes = {cell["source_media_sha256"] for cell in fit}
    calibration_hashes = {cell["source_media_sha256"] for cell in calibration}
    if fit_ids & calibration_ids or fit_hashes & calibration_hashes:
        raise FactorialMarginError("fit/calibration source or media leakage")
    return {
        "schema_version": POPULATION_SCHEMA,
        "population_id": population_id,
        "created_utc": row["created_utc"],
        "score_contract": contract,
        "confirmation_registry_sha256": registry,
        "fit_cells": fit,
        "calibration_cells": calibration,
    }


def _candidate_observations(cells: Sequence[Mapping[str, Any]], candidate_id: str) -> dict[str, Any]:
    action_margins = {branch: [] for branch in NEGATIVE_BRANCHES}
    preservation_absolute = {axis: [] for axis in PRESERVATION_AXES}
    preservation_delta = {axis: [] for axis in PRESERVATION_AXES}
    for cell in cells:
        branches = cell["candidate_scores"][candidate_id]
        forward = branches["forward"]
        noop = branches["noop"]
        for branch in NEGATIVE_BRANCHES:
            action_margins[branch].append(forward["action"] - branches[branch]["action"])
        for axis in PRESERVATION_AXES:
            preservation_absolute[axis].append(forward[axis])
            preservation_delta[axis].append(forward[axis] - noop[axis])
    return {
        "action_margins": action_margins,
        "preservation_absolute": preservation_absolute,
        "preservation_delta_vs_noop": preservation_delta,
    }


def _minimums(values: Mapping[str, Sequence[float]]) -> dict[str, float]:
    return {key: min(rows) for key, rows in values.items()}


def _fit_summary(cells: Sequence[Mapping[str, Any]], candidate_id: str) -> dict[str, Any]:
    observations = _candidate_observations(cells, candidate_id)
    action = _minimums(observations["action_margins"])
    absolute = _minimums(observations["preservation_absolute"])
    delta = _minimums(observations["preservation_delta_vs_noop"])
    core = min(action[branch] for branch in CORE_NEGATIVES)
    all_negative = min(action.values())
    worst_delta = min(delta.values())
    eligible = all_negative > 0.0 and worst_delta >= 0.0
    return {
        "candidate_id": candidate_id,
        "fit_cell_ids": [cell["cell_id"] for cell in cells],
        "minimum_action_margins": action,
        "minimum_core_forward_vs_noop_reverse_margin": core,
        "minimum_all_negative_margin": all_negative,
        "minimum_forward_preservation_scores": absolute,
        "minimum_preservation_delta_vs_noop": delta,
        "minimum_all_preservation_deltas": worst_delta,
        "all_action_margins_strictly_positive": all_negative > 0.0,
        "all_preservation_deltas_nonnegative": worst_delta >= 0.0,
        "fit_eligible": eligible,
    }


def _select_fit_candidate(summaries: Sequence[Mapping[str, Any]]) -> str | None:
    eligible = [row for row in summaries if row["fit_eligible"]]
    if not eligible:
        return None
    # Candidate ID is the deterministic final tie break; confirmation and
    # calibration scores have no route into selection.
    return max(
        eligible,
        key=lambda row: (
            row["minimum_core_forward_vs_noop_reverse_margin"],
            row["minimum_all_negative_margin"],
            row["minimum_all_preservation_deltas"],
            row["candidate_id"],
        ),
    )["candidate_id"]


def author_policy(population_value: Any) -> dict[str, Any]:
    population = validate_population(population_value)
    population_digest = object_sha256(population)
    candidate_ids = sorted(population["fit_cells"][0]["candidate_scores"])
    summaries = [
        _fit_summary(population["fit_cells"], candidate_id)
        for candidate_id in candidate_ids
    ]
    selected = _select_fit_candidate(summaries)
    gate_policy: dict[str, Any] | None = None
    status = "zero_update_no_fit_candidate"
    optimizer_allowed = False
    if selected is not None:
        calibration = _candidate_observations(population["calibration_cells"], selected)
        action = _minimums(calibration["action_margins"])
        absolute = _minimums(calibration["preservation_absolute"])
        delta = _minimums(calibration["preservation_delta_vs_noop"])
        action_pass = min(action.values()) > 0.0
        preservation_pass = min(delta.values()) >= 0.0
        gate_policy = {
            "candidate_id": selected,
            "calibration_rule": "worst_observed_prospective_calibration_cell_no_confirmation_access",
            "calibration_cell_ids": [cell["cell_id"] for cell in population["calibration_cells"]],
            "action_margin_minimums": action,
            "preservation_absolute_minimums": absolute,
            "preservation_delta_vs_noop_minimums": delta,
            "all_calibration_action_margins_strictly_positive": action_pass,
            "all_calibration_preservation_deltas_nonnegative": preservation_pass,
        }
        if action_pass and preservation_pass:
            status = "policy_frozen_optimizer_canary_allowed"
            optimizer_allowed = True
        else:
            status = "zero_update_calibration_failed"
    body = {
        "schema_version": POLICY_SCHEMA,
        "status": status,
        "population_id": population["population_id"],
        "population_digest": population_digest,
        "score_contract": population["score_contract"],
        "score_contract_digest": object_sha256(population["score_contract"]),
        "confirmation_registry_sha256": population["confirmation_registry_sha256"],
        "fit_source_ids": sorted({cell["source_id"] for cell in population["fit_cells"]}),
        "fit_source_media_sha256s": sorted(
            {cell["source_media_sha256"] for cell in population["fit_cells"]}
        ),
        "calibration_source_ids": sorted(
            {cell["source_id"] for cell in population["calibration_cells"]}
        ),
        "calibration_source_media_sha256s": sorted(
            {cell["source_media_sha256"] for cell in population["calibration_cells"]}
        ),
        "fit_candidate_summaries": summaries,
        "selected_candidate_id": selected,
        "calibration_gate_policy": gate_policy,
        "optimizer_step_allowed": optimizer_allowed,
        "confirmation_scores_consumed": False,
        "method_success_claimed": False,
    }
    return {**body, "policy_digest": object_sha256(body)}


def validate_policy(value: Any) -> dict[str, Any]:
    row = _closed(value, _POLICY_FIELDS, label="policy")
    if row["schema_version"] != POLICY_SCHEMA:
        raise FactorialMarginError("policy schema differs")
    unsigned = {key: row[key] for key in row if key != "policy_digest"}
    if row["policy_digest"] != object_sha256(unsigned):
        raise FactorialMarginError("policy digest differs")
    contract = validate_score_contract(row["score_contract"])
    if row["score_contract_digest"] != object_sha256(contract):
        raise FactorialMarginError("score contract digest differs")
    if row["confirmation_scores_consumed"] is not False or row["method_success_claimed"] is not False:
        raise FactorialMarginError("policy claims confirmation access or method success")
    _identifier(row["population_id"], label="policy population ID")
    _sha256(row["population_digest"], label="population digest")
    _sha256(
        row["confirmation_registry_sha256"],
        label="policy confirmation registry SHA-256",
    )
    source_registries = (
        ("fit_source_ids", _identifier),
        ("calibration_source_ids", _identifier),
        ("fit_source_media_sha256s", _sha256),
        ("calibration_source_media_sha256s", _sha256),
    )
    normalized_registries: dict[str, list[str]] = {}
    for field, validator in source_registries:
        values = list(_sequence(row[field], label=field))
        if values != sorted(set(values)):
            raise FactorialMarginError(f"{field} must be sorted and unique")
        normalized_registries[field] = [
            validator(item, label=f"{field} item") for item in values
        ]
    if set(normalized_registries["fit_source_ids"]) & set(
        normalized_registries["calibration_source_ids"]
    ):
        raise FactorialMarginError("policy fit/calibration source IDs overlap")
    if set(normalized_registries["fit_source_media_sha256s"]) & set(
        normalized_registries["calibration_source_media_sha256s"]
    ):
        raise FactorialMarginError("policy fit/calibration media overlap")
    summaries = _sequence(row["fit_candidate_summaries"], label="fit candidate summaries")
    summary_ids: list[str] = []
    for index, raw_summary in enumerate(summaries):
        summary = _closed(
            raw_summary, _FIT_SUMMARY_FIELDS, label=f"fit candidate summary {index}"
        )
        summary_ids.append(_identifier(summary["candidate_id"], label="summary candidate ID"))
        for field, keys in (
            ("minimum_action_margins", NEGATIVE_BRANCHES),
            ("minimum_forward_preservation_scores", PRESERVATION_AXES),
            ("minimum_preservation_delta_vs_noop", PRESERVATION_AXES),
        ):
            values = _mapping(summary[field], label=f"summary {field}")
            if set(values) != set(keys):
                raise FactorialMarginError(f"summary {field} closure differs")
            for key in keys:
                number = values[key]
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                    raise FactorialMarginError(f"summary {field}.{key} must be finite")
        if any(
            type(summary[field]) is not bool
            for field in (
                "all_action_margins_strictly_positive",
                "all_preservation_deltas_nonnegative",
                "fit_eligible",
            )
        ):
            raise FactorialMarginError("summary decisions must be boolean")
    if summary_ids != sorted(set(summary_ids)):
        raise FactorialMarginError("fit candidate summaries must be sorted and unique")
    allowed = row["optimizer_step_allowed"]
    if type(allowed) is not bool:
        raise FactorialMarginError("optimizer_step_allowed must be boolean")
    gate = row["calibration_gate_policy"]
    if gate is not None:
        gate = _closed(gate, _GATE_FIELDS, label="calibration gate policy")
        _identifier(gate["candidate_id"], label="gate candidate ID")
        if gate["calibration_rule"] != "worst_observed_prospective_calibration_cell_no_confirmation_access":
            raise FactorialMarginError("calibration rule differs")
        cell_ids = list(_sequence(gate["calibration_cell_ids"], label="calibration cell IDs"))
        if len(cell_ids) != len(set(cell_ids)):
            raise FactorialMarginError("calibration cell IDs must be unique")
        for cell_id in cell_ids:
            _identifier(cell_id, label="calibration cell ID")
        for field, keys in (
            ("action_margin_minimums", NEGATIVE_BRANCHES),
            ("preservation_absolute_minimums", PRESERVATION_AXES),
            ("preservation_delta_vs_noop_minimums", PRESERVATION_AXES),
        ):
            values = _mapping(gate[field], label=field)
            if set(values) != set(keys):
                raise FactorialMarginError(f"{field} closure differs")
            for key in keys:
                number = values[key]
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                    raise FactorialMarginError(f"{field}.{key} must be finite")
                numeric = float(number)
                if field == "action_margin_minimums" and not 0.0 < numeric <= 1.0:
                    raise FactorialMarginError("frozen action margins must be in (0,1]")
                if field != "action_margin_minimums" and not 0.0 <= numeric <= 1.0:
                    raise FactorialMarginError("frozen preservation gates must be in [0,1]")
        if any(
            type(gate[field]) is not bool
            for field in (
                "all_calibration_action_margins_strictly_positive",
                "all_calibration_preservation_deltas_nonnegative",
            )
        ):
            raise FactorialMarginError("calibration gate decisions must be boolean")
    if allowed:
        if gate is None or row["status"] != "policy_frozen_optimizer_canary_allowed":
            raise FactorialMarginError("allowed policy lacks a frozen gate")
        if gate["candidate_id"] != row["selected_candidate_id"]:
            raise FactorialMarginError("selected/gate candidate differs")
        if gate["all_calibration_action_margins_strictly_positive"] is not True:
            raise FactorialMarginError("allowed policy lacks positive action margins")
        if gate["all_calibration_preservation_deltas_nonnegative"] is not True:
            raise FactorialMarginError("allowed policy lacks preservation non-regression")
    else:
        if not str(row["status"]).startswith("zero_update_"):
            raise FactorialMarginError("disallowed policy lacks zero-update status")
    return dict(row)


def evaluate_confirmation(policy_value: Any, confirmation_value: Any) -> dict[str, Any]:
    policy = validate_policy(policy_value)
    if not policy["optimizer_step_allowed"]:
        raise FactorialMarginError("zero-update policy cannot open confirmation")
    row = _closed(confirmation_value, _CONFIRMATION_FIELDS, label="confirmation")
    if row["schema_version"] != CONFIRMATION_SCHEMA:
        raise FactorialMarginError("confirmation schema differs")
    if row["policy_digest"] != policy["policy_digest"]:
        raise FactorialMarginError("confirmation policy digest differs")
    contract = validate_score_contract(row["score_contract"])
    if object_sha256(contract) != policy["score_contract_digest"]:
        raise FactorialMarginError("confirmation score contract differs")
    registry = _sha256(
        row["confirmation_registry_sha256"], label="confirmation registry SHA-256"
    )
    if registry != policy["confirmation_registry_sha256"]:
        raise FactorialMarginError("confirmation registry differs")
    cells = validate_cells(row["cells"], label="confirmation cells")
    sealed_ids = set(policy["fit_source_ids"]) | set(policy["calibration_source_ids"])
    sealed_hashes = set(policy["fit_source_media_sha256s"]) | set(
        policy["calibration_source_media_sha256s"]
    )
    if any(cell["source_id"] in sealed_ids for cell in cells):
        raise FactorialMarginError("confirmation source ID leaked from fit/calibration")
    if any(cell["source_media_sha256"] in sealed_hashes for cell in cells):
        raise FactorialMarginError("confirmation media leaked from fit/calibration")
    candidate = policy["selected_candidate_id"]
    if any(candidate not in cell["candidate_scores"] for cell in cells):
        raise FactorialMarginError("frozen candidate missing from confirmation")
    gate = policy["calibration_gate_policy"]
    decisions = []
    for cell in cells:
        observations = _candidate_observations([cell], candidate)
        action_values = _minimums(observations["action_margins"])
        absolute_values = _minimums(observations["preservation_absolute"])
        delta_values = _minimums(observations["preservation_delta_vs_noop"])
        action_pass = {
            branch: action_values[branch] >= gate["action_margin_minimums"][branch]
            for branch in NEGATIVE_BRANCHES
        }
        absolute_pass = {
            axis: absolute_values[axis] >= gate["preservation_absolute_minimums"][axis]
            for axis in PRESERVATION_AXES
        }
        delta_pass = {
            axis: delta_values[axis] >= gate["preservation_delta_vs_noop_minimums"][axis]
            for axis in PRESERVATION_AXES
        }
        passed = all(action_pass.values()) and all(absolute_pass.values()) and all(delta_pass.values())
        decisions.append(
            {
                "cell_id": cell["cell_id"],
                "source_id": cell["source_id"],
                "action_family": cell["action_family"],
                "action_margin_values": action_values,
                "action_margin_pass": action_pass,
                "preservation_absolute_values": absolute_values,
                "preservation_absolute_pass": absolute_pass,
                "preservation_delta_vs_noop_values": delta_values,
                "preservation_delta_vs_noop_pass": delta_pass,
                "all_noncompensating_gates_pass": passed,
            }
        )
    body = {
        "schema_version": CONFIRMATION_RECEIPT_SCHEMA,
        "policy_digest": policy["policy_digest"],
        "score_contract_digest": policy["score_contract_digest"],
        "confirmation_registry_sha256": registry,
        "selected_candidate_id": candidate,
        "thresholds_frozen_before_confirmation": True,
        "confirmation_cell_decisions": decisions,
        "all_confirmation_cells_pass": all(row["all_noncompensating_gates_pass"] for row in decisions),
        "optimizer_step_allowed": False,
        "method_success_claimed": False,
    }
    return {**body, "receipt_digest": object_sha256(body)}


def _read_json(path_value: str) -> Any:
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise FactorialMarginError("input must be an absolute plain file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorialMarginError(f"cannot read JSON: {error}") from error


def _write_create_only(path_value: str, value: Any) -> None:
    path = Path(path_value)
    if not path.is_absolute() or path == Path("/"):
        raise FactorialMarginError("output must be absolute and non-root")
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise FactorialMarginError("output must be fresh with an existing parent")
    payload = canonical_json_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    author = sub.add_parser("author-policy")
    author.add_argument("--population", required=True)
    author.add_argument("--output", required=True)
    confirm = sub.add_parser("evaluate-confirmation")
    confirm.add_argument("--policy", required=True)
    confirm.add_argument("--confirmation", required=True)
    confirm.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "author-policy":
        result = author_policy(_read_json(args.population))
    else:
        result = evaluate_confirmation(
            _read_json(args.policy), _read_json(args.confirmation)
        )
    _write_create_only(args.output, result)
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
