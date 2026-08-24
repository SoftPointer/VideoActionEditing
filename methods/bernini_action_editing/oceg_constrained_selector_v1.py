#!/usr/bin/env python3
"""Fail-closed OCEG candidate selection against a matched Frozen Base.

This module deliberately does not implement a scalar reward.  It accepts a
pre-registered set of preservation gates and graph predicates, rejects every
candidate with a failed or uncertain required component, and only then uses
action margin plus Frozen-Base policy deviation to order feasible candidates.

The real target is outside this ABI.  A selected pair is an offline preference
pair only; it does not authorize training, decoding, routing, or a scientific
claim.  When no strict net gain over the matched Frozen Base is provable, the
selector returns the Frozen Base.  If the Frozen Base itself fails its absolute
safety gates, the selector abstains instead of pretending that a safe output
exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence


INPUT_SCHEMA = "oceg-constrained-selection-input-v1"
OUTPUT_SCHEMA = "oceg-constrained-selection-receipt-v1"
METHOD = "oceg-noncompensatory-frozen-base-selector-v1"

HARD_CATEGORIES = ("quality", "identity", "noncollapse")
GATE_STATUSES = ("pass", "fail", "uncertain")
PREDICATE_APPLICABILITY = ("required", "not_applicable")
PREDICATE_STATUSES = ("pass", "fail", "uncertain", "not_applicable")
BASE_ACTION_OUTCOMES = ("pass", "fail", "uncertain")

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class OCEGConstrainedSelectionError(ValueError):
    """Input or provenance violates the constrained-selection ABI."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise OCEGConstrainedSelectionError("value is not canonical finite JSON") from error


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OCEGConstrainedSelectionError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str, *, nonempty: bool = False) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OCEGConstrainedSelectionError(f"{label} must be a sequence")
    if nonempty and len(value) == 0:
        raise OCEGConstrainedSelectionError(f"{label} must be nonempty")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise OCEGConstrainedSelectionError(f"{label} is not a canonical identifier")
    return value


def _boolean(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise OCEGConstrainedSelectionError(f"{label} must be {expected}")


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OCEGConstrainedSelectionError(f"{label} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise OCEGConstrainedSelectionError(f"{label} must be a finite nonnegative number")
    return result


def _seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OCEGConstrainedSelectionError(f"{label} must be a nonnegative integer")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise OCEGConstrainedSelectionError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _validate_gate_registry(value: Any) -> dict[str, tuple[str, ...]]:
    registry = _mapping(value, "gate_registry")
    _exact_keys(registry, set(HARD_CATEGORIES), "gate_registry")
    result: dict[str, tuple[str, ...]] = {}
    for category in HARD_CATEGORIES:
        rows = tuple(
            _identifier(item, f"gate_registry.{category} gate")
            for item in _sequence(registry[category], f"gate_registry.{category}", nonempty=True)
        )
        if len(set(rows)) != len(rows):
            raise OCEGConstrainedSelectionError(
                f"gate_registry.{category} contains duplicate gates"
            )
        result[category] = rows
    return result


def _validate_hard_gates(
    value: Any,
    registry: Mapping[str, tuple[str, ...]],
    label: str,
) -> tuple[bool, list[str]]:
    categories = _mapping(value, label)
    _exact_keys(categories, set(HARD_CATEGORIES), label)
    reasons: list[str] = []
    for category in HARD_CATEGORIES:
        gates = _mapping(categories[category], f"{label}.{category}")
        _exact_keys(gates, set(registry[category]), f"{label}.{category}")
        for gate_id in registry[category]:
            status = gates[gate_id]
            if status not in GATE_STATUSES:
                raise OCEGConstrainedSelectionError(
                    f"{label}.{category}.{gate_id} status differs"
                )
            if status != "pass":
                reasons.append(f"{category}:{gate_id}:{status}")
    return not reasons, reasons


def _validate_predicate_registry(value: Any) -> tuple[dict[str, Any], ...]:
    rows = _sequence(value, "predicate_registry", nonempty=True)
    result = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"predicate_registry[{index}]")
        _exact_keys(
            row,
            {"predicate_id", "applicability", "preregistered"},
            f"predicate_registry[{index}]",
        )
        predicate_id = _identifier(row["predicate_id"], "predicate_id")
        if predicate_id in seen:
            raise OCEGConstrainedSelectionError("predicate_registry contains duplicate IDs")
        seen.add(predicate_id)
        if row["applicability"] not in PREDICATE_APPLICABILITY:
            raise OCEGConstrainedSelectionError("predicate applicability differs")
        _boolean(row["preregistered"], True, "predicate preregistered")
        result.append(
            {
                "predicate_id": predicate_id,
                "applicability": row["applicability"],
                "preregistered": True,
            }
        )
    return tuple(result)


def _validate_graph_predicates(
    value: Any,
    registry: Sequence[Mapping[str, Any]],
    label: str,
) -> tuple[bool, list[str]]:
    rows = _sequence(value, label, nonempty=True)
    by_id: dict[str, str] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{label}[{index}]")
        _exact_keys(row, {"predicate_id", "status"}, f"{label}[{index}]")
        predicate_id = _identifier(row["predicate_id"], "predicate_id")
        if predicate_id in by_id:
            raise OCEGConstrainedSelectionError(f"{label} contains duplicate predicates")
        if row["status"] not in PREDICATE_STATUSES:
            raise OCEGConstrainedSelectionError(f"{label} predicate status differs")
        by_id[predicate_id] = row["status"]
    registered_ids = {row["predicate_id"] for row in registry}
    if set(by_id) != registered_ids:
        raise OCEGConstrainedSelectionError(f"{label} does not exactly cover predicate registry")
    reasons: list[str] = []
    for row in registry:
        predicate_id = row["predicate_id"]
        status = by_id[predicate_id]
        if row["applicability"] == "required":
            if status != "pass":
                reasons.append(f"graph:{predicate_id}:{status}")
        elif status != "not_applicable":
            # An absent/offscreen edge is removable only through the sealed
            # registry.  It cannot later be relabelled as observed success.
            reasons.append(f"graph:{predicate_id}:expected_not_applicable_got_{status}")
    return not reasons, reasons


def _validate_provenance(
    value: Mapping[str, Any],
    *,
    label: str,
    case_id: str,
    seed: int,
) -> None:
    if value.get("case_id") != case_id or value.get("seed") != seed:
        raise OCEGConstrainedSelectionError(f"{label} is not case/seed matched")
    _boolean(value.get("target_inputs_consumed"), False, f"{label}.target_inputs_consumed")
    _boolean(value.get("real_target_decoded"), False, f"{label}.real_target_decoded")


def select_oceg_candidate(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    """Select a strict net gain or safely fall back/abstain.

    Failed and uncertain hard gates are intentionally never converted to a
    number.  Numeric action margin is consulted only after all Boolean gates
    pass, so it cannot compensate for blur, identity loss, collapse, or a
    missing required graph predicate.
    """

    root = _mapping(bundle, "bundle")
    if root.get("schema_version") != INPUT_SCHEMA:
        raise OCEGConstrainedSelectionError("input schema differs")
    case_id = _identifier(root.get("case_id"), "case_id")
    seed = _seed(root.get("seed"), "seed")
    registry = _validate_gate_registry(root.get("gate_registry"))
    predicate_registry = _validate_predicate_registry(root.get("predicate_registry"))
    threshold = _finite_nonnegative(root.get("action_margin_min"), "action_margin_min")

    base = _mapping(root.get("frozen_base"), "frozen_base")
    base_id = _identifier(base.get("record_id"), "frozen_base.record_id")
    _validate_provenance(base, label="frozen_base", case_id=case_id, seed=seed)
    _boolean(base.get("base_frozen"), True, "frozen_base.base_frozen")
    _boolean(
        base.get("graph_observation_supplied"),
        False,
        "frozen_base.graph_observation_supplied",
    )
    if base.get("graph_success") is not None:
        raise OCEGConstrainedSelectionError("Frozen Base graph_success must remain null")
    _boolean(
        base.get("used_as_graph_positive"),
        False,
        "frozen_base.used_as_graph_positive",
    )
    if base.get("action_outcome") not in BASE_ACTION_OUTCOMES:
        raise OCEGConstrainedSelectionError("frozen_base.action_outcome differs")
    base_safe, base_reasons = _validate_hard_gates(
        base.get("hard_gates"), registry, "frozen_base.hard_gates"
    )

    candidates = _sequence(root.get("candidates"), "candidates", nonempty=True)
    candidate_rows: list[dict[str, Any]] = []
    seen_ids = {base_id}
    feasible: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        candidate = _mapping(raw, f"candidates[{index}]")
        candidate_id = _identifier(candidate.get("candidate_id"), "candidate_id")
        if candidate_id in seen_ids:
            raise OCEGConstrainedSelectionError("candidate/base IDs are not unique")
        seen_ids.add(candidate_id)
        _validate_provenance(
            candidate,
            label=f"candidate {candidate_id}",
            case_id=case_id,
            seed=seed,
        )
        _boolean(candidate.get("base_checkpoint_frozen"), True, f"{candidate_id}.base_checkpoint_frozen")
        _boolean(candidate.get("selected_without_real_target"), True, f"{candidate_id}.selected_without_real_target")
        hard_pass, hard_reasons = _validate_hard_gates(
            candidate.get("hard_gates"), registry, f"candidate {candidate_id}.hard_gates"
        )
        graph_pass, graph_reasons = _validate_graph_predicates(
            candidate.get("graph_predicates"),
            predicate_registry,
            f"candidate {candidate_id}.graph_predicates",
        )
        action_margin = _finite_nonnegative(
            candidate.get("action_margin_over_registered_floor"),
            f"{candidate_id}.action_margin_over_registered_floor",
        )
        policy_deviation = _finite_nonnegative(
            candidate.get("frozen_base_policy_deviation"),
            f"{candidate_id}.frozen_base_policy_deviation",
        )
        reasons = [*hard_reasons, *graph_reasons]
        if action_margin < threshold:
            reasons.append("action:margin_below_preregistered_threshold")
        if base.get("action_outcome") != "fail":
            reasons.append(
                "net_gain:not_provable_from_frozen_base_"
                + str(base.get("action_outcome"))
            )
        eligible = hard_pass and graph_pass and action_margin >= threshold and (
            base.get("action_outcome") == "fail"
        )
        row = {
            "candidate_id": candidate_id,
            "eligible": eligible,
            "hard_gates_passed": hard_pass,
            "critical_graph_passed": graph_pass,
            "strict_net_gain_over_frozen_base": eligible,
            "action_margin_over_registered_floor": action_margin,
            "frozen_base_policy_deviation": policy_deviation,
            "rejection_reasons": reasons,
        }
        candidate_rows.append(row)
        if eligible:
            feasible.append(row)

    if not base_safe:
        status = "ABSTAIN_NO_SAFE_OUTPUT"
        selected_id: Optional[str] = None
        selected_kind = "none"
        pair = None
    elif not feasible:
        status = "FROZEN_BASE_FALLBACK"
        selected_id = base_id
        selected_kind = "frozen_base"
        pair = None
    else:
        # The action margin orders only already-feasible candidates; the
        # policy deviation is the preservation tie-break, never compensation.
        feasible.sort(
            key=lambda row: (
                -row["action_margin_over_registered_floor"],
                row["frozen_base_policy_deviation"],
                row["candidate_id"],
            )
        )
        winner = feasible[0]
        status = "CANDIDATE_CHOSEN"
        selected_id = winner["candidate_id"]
        selected_kind = "candidate"
        pair = {
            "chosen_id": selected_id,
            "rejected_id": base_id,
            "reference_model": "matched_frozen_base",
            "offline_preference_pair_only": True,
        }

    result: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "method": METHOD,
        "case_id": case_id,
        "seed": seed,
        "status": status,
        "selected_id": selected_id,
        "selected_kind": selected_kind,
        "frozen_base_record_id": base_id,
        "frozen_base_safe": base_safe,
        "frozen_base_safety_reasons": base_reasons,
        "frozen_base_action_outcome": base.get("action_outcome"),
        "candidate_evaluations": candidate_rows,
        "preference_pair": pair,
        "coverage": {
            "candidate_count": len(candidate_rows),
            "eligible_count": len(feasible),
            "fallback_or_abstain": status != "CANDIDATE_CHOSEN",
        },
        "selection_semantics": {
            "scalar_cross_gate_compensation": False,
            "uncertain_required_gate_is_success": False,
            "unregistered_missing_edge_is_not_applicable": False,
            "action_margin_used_only_after_hard_gates": True,
            "frozen_base_is_mandatory": True,
        },
        "target_inputs_consumed": False,
        "real_target_decoded": False,
        "training_authorized": False,
        "renderer_authorized": False,
        "generator_injection_authorized": False,
        "scientific_claim_authorized": False,
    }
    result["receipt_sha256"] = _digest(result)
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OCEGConstrainedSelectionError(f"cannot read input: {error}") from error
    return _mapping(value, "input")


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        raise OCEGConstrainedSelectionError("refusing to overwrite output") from error


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _write_exclusive(args.output, select_oceg_candidate(_read_json(args.input)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

