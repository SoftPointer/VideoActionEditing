#!/usr/bin/env python3
"""Fail-closed offline admission for canonical OCEG-R1 observations.

This program does not decode video, run SAM2, CoTracker, DINO, a renderer, or
an optimizer.  It consumes already-published canonical graph observations and
checks only the registered Z0 controls:

* target forward versus reverse, shuffle, and source-noop;
* self-anchor action versus noop, reverse, and phase-0-static per appearance;
* cross-appearance critical-edge consensus;
* identity, contact, terminal-hold, and unresolved-evidence hard gates.

Frozen Base is mandatory provenance, but it is deliberately not a graph
positive.  Its input and receipt fields force ``graph_success`` to JSON null
and forbid supplying a B0 graph observation.  Any valid ``uncertain`` status
rejects the complete bundle.  A mechanically admitted synthetic fixture is
still not scientific or renderer evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, NoReturn, Sequence


INPUT_SCHEMA = "bernini-oceg-z0-canonical-observations-v1"
RECEIPT_SCHEMA = "bernini-oceg-z0-admission-receipt-v1"
INPUT_KIND = "canonical_graph_observations_not_raw_media"

AXES = (
    "role_binding",
    "relative_kinematics",
    "contact_state",
    "event_order",
    "terminal_hold",
    "secondary_effect",
)
MANDATORY_AXES = (
    "role_binding",
    "relative_kinematics",
    "event_order",
    "terminal_hold",
)
TARGET_KEYS = ("forward", "reverse", "shuffle", "source_noop")
TARGET_ARMS = {
    "forward": "target_forward",
    "reverse": "target_reverse",
    "shuffle": "target_shuffle",
    "source_noop": "source_noop",
}
ANCHOR_KEYS = ("action", "noop", "reverse", "static")
ANCHOR_ARMS = {
    "action": "anchor_action",
    "noop": "anchor_noop",
    "reverse": "anchor_reverse",
    "static": "anchor_phase0_static",
}
CONTRAST_AXIS = {
    "reverse": "event_order",
    "shuffle": "event_order",
    "source_noop": "relative_kinematics",
    "noop": "relative_kinematics",
    "static": "relative_kinematics",
}
OBSERVATION_STATUS = ("observed", "uncertain")
GATE_STATUS = ("pass", "fail", "uncertain")
CONTACT_GATE_STATUS = GATE_STATUS + ("not_applicable",)
NODE_ROLES = (
    "actor",
    "effector_hand",
    "tool",
    "moving_object",
    "support",
    "patient",
)
NODE_OWNERSHIP = ("source_owned", "instruction_introduced")
NODE_LIFECYCLE_STATUS = GATE_STATUS
MORPH_STATUS = ("not_detected", "detected", "uncertain")
EFFECTOR_EVIDENCE_KEYS = (
    "support_edge_off_on",
    "relative_height_reversal",
    "terminal_supported_hold",
)
TERMINAL_MODES = (
    "visible_at_terminal",
    "out_of_frame_after_confirmed_support_release",
    "unresolved_disappearance",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

# Profiles are code-owned.  A caller cannot lower thresholds in its JSON.
PROFILES: Mapping[str, Mapping[str, Any]] = {
    "synthetic_smoke": {
        "exact_case_count": 1,
        "minimum_control_win_cases": 1,
        "minimum_consensus_pass_cases": 1,
        "minimum_hard_gate_pass_cases": 1,
        "anchor_appearance_count": 3,
        "comparison_margin": 0.05,
        "identity_idf1_min": 0.85,
        "contact_macro_f1_min": 0.85,
        "terminal_macro_f1_min": 0.90,
        "terminal_hold_rgb_frames_min": 8,
        "critical_edge_cosine_min": 0.95,
        "critical_edge_distance_max": 0.15,
    },
    "development_draft": {
        "exact_case_count": 4,
        "minimum_control_win_cases": 3,
        "minimum_consensus_pass_cases": 4,
        "minimum_hard_gate_pass_cases": 4,
        "anchor_appearance_count": 3,
        "comparison_margin": 0.05,
        "identity_idf1_min": 0.85,
        "contact_macro_f1_min": 0.85,
        "terminal_macro_f1_min": 0.90,
        "terminal_hold_rgb_frames_min": 8,
        "critical_edge_cosine_min": 0.95,
        "critical_edge_distance_max": 0.15,
    },
    "formal_confirmation_draft": {
        "exact_case_count": 12,
        "minimum_control_win_cases": 9,
        "minimum_consensus_pass_cases": 12,
        "minimum_hard_gate_pass_cases": 12,
        "anchor_appearance_count": 3,
        "comparison_margin": 0.05,
        "identity_idf1_min": 0.85,
        "contact_macro_f1_min": 0.85,
        "terminal_macro_f1_min": 0.90,
        "terminal_hold_rgb_frames_min": 8,
        "critical_edge_cosine_min": 0.95,
        "critical_edge_distance_max": 0.15,
    },
}


class OCEGZ0AdmissionError(ValueError):
    """Malformed or ambiguous canonical-observation input."""


def fail(message: str) -> NoReturn:
    raise OCEGZ0AdmissionError(message)


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
        raise OCEGZ0AdmissionError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    fail(f"non-finite JSON constant is forbidden: {value}")


def load_json_strict(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise OCEGZ0AdmissionError(f"cannot read input: {error}") from error
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_constant,
        )
    except OCEGZ0AdmissionError:
        raise
    except Exception as error:
        raise OCEGZ0AdmissionError(f"invalid JSON input: {error}") from error
    if type(value) is not dict:
        fail("input root must be one JSON object")
    return value


def _exact_keys(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != set(keys) or len(value) != len(keys):
        fail(f"{label} keys must be exactly {tuple(keys)!r}")
    return value


def _list(value: Any, label: str, *, nonempty: bool = False) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        fail(f"{label} must be {'a nonempty ' if nonempty else 'a '}list")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        fail(f"{label} is not a canonical identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} is not lowercase SHA-256")
    return value


def _finite_unit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        fail(f"{label} must be finite in [0,1]")
    return number


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} must be an exact non-negative integer")
    return value


def _validate_measure(
    value: Any, label: str, *, status_domain: Sequence[str] = OBSERVATION_STATUS
) -> Mapping[str, Any]:
    row = _exact_keys(value, ("status", "value"), label)
    status = row["status"]
    if status not in status_domain:
        fail(f"{label}.status is outside {tuple(status_domain)!r}")
    if status == "uncertain":
        if row["value"] is not None:
            fail(f"{label}.value must be null when uncertain")
    else:
        _finite_unit(row["value"], f"{label}.value")
    return row


def _validate_score_map(
    value: Any, required_axes: Sequence[str], label: str
) -> Mapping[str, Any]:
    row = _exact_keys(value, required_axes, label)
    for axis in required_axes:
        _validate_measure(row[axis], f"{label}.{axis}")
    return row


def _validate_optional_gate_number(
    status: str, value: Any, label: str, *, integer: bool = False
) -> None:
    if status in ("uncertain", "not_applicable"):
        if value is not None:
            fail(f"{label} must be null for status {status}")
    elif integer:
        _nonnegative_int(value, label)
    else:
        _finite_unit(value, label)


def _validate_hard_gates(
    value: Any, *, contact_required: bool, label: str
) -> Mapping[str, Any]:
    gates = _exact_keys(value, ("identity", "contact", "terminal", "uncertainty"), label)

    identity = _exact_keys(
        gates["identity"], ("status", "idf1", "id_switch_count"), f"{label}.identity"
    )
    if identity["status"] not in GATE_STATUS:
        fail(f"{label}.identity.status differs")
    _validate_optional_gate_number(identity["status"], identity["idf1"], f"{label}.identity.idf1")
    _validate_optional_gate_number(
        identity["status"], identity["id_switch_count"],
        f"{label}.identity.id_switch_count", integer=True,
    )

    contact = _exact_keys(
        gates["contact"], ("status", "macro_f1"), f"{label}.contact"
    )
    if contact["status"] not in CONTACT_GATE_STATUS:
        fail(f"{label}.contact.status differs")
    if contact_required and contact["status"] == "not_applicable":
        fail(f"{label}.contact cannot be not_applicable for a contact case")
    if not contact_required and contact["status"] != "not_applicable":
        fail(f"{label}.contact must be not_applicable for a non-contact case")
    _validate_optional_gate_number(
        contact["status"], contact["macro_f1"], f"{label}.contact.macro_f1"
    )

    terminal = _exact_keys(
        gates["terminal"], ("status", "macro_f1", "hold_rgb_frames"),
        f"{label}.terminal",
    )
    if terminal["status"] not in GATE_STATUS:
        fail(f"{label}.terminal.status differs")
    _validate_optional_gate_number(
        terminal["status"], terminal["macro_f1"], f"{label}.terminal.macro_f1"
    )
    _validate_optional_gate_number(
        terminal["status"], terminal["hold_rgb_frames"],
        f"{label}.terminal.hold_rgb_frames", integer=True,
    )

    uncertainty = _exact_keys(
        gates["uncertainty"],
        ("status", "unresolved_required_nodes", "unresolved_required_edges"),
        f"{label}.uncertainty",
    )
    if uncertainty["status"] not in GATE_STATUS:
        fail(f"{label}.uncertainty.status differs")
    for key in ("unresolved_required_nodes", "unresolved_required_edges"):
        _validate_optional_gate_number(
            uncertainty["status"], uncertainty[key],
            f"{label}.uncertainty.{key}", integer=True,
        )
    return gates


def _validate_observation(
    value: Any,
    *,
    expected_arm: str,
    required_axes: Sequence[str],
    contact_required: bool,
    label: str,
) -> Mapping[str, Any]:
    row = _exact_keys(
        value,
        ("observation_id", "arm", "evidence_sha256", "scores", "hard_gates"),
        label,
    )
    _identifier(row["observation_id"], f"{label}.observation_id")
    if row["arm"] != expected_arm:
        fail(f"{label}.arm must be {expected_arm!r}")
    _sha256(row["evidence_sha256"], f"{label}.evidence_sha256")
    _validate_score_map(row["scores"], required_axes, f"{label}.scores")
    _validate_hard_gates(
        row["hard_gates"], contact_required=contact_required, label=f"{label}.hard_gates"
    )
    return row


def _validate_frozen_base(value: Any, label: str) -> Mapping[str, Any]:
    row = _exact_keys(
        value,
        (
            "record_id",
            "seed",
            "media_sha256",
            "graph_observation_supplied",
            "graph_success",
            "used_as_graph_positive",
            "comparison_role",
        ),
        label,
    )
    _identifier(row["record_id"], f"{label}.record_id")
    _nonnegative_int(row["seed"], f"{label}.seed")
    _sha256(row["media_sha256"], f"{label}.media_sha256")
    if (
        row["graph_observation_supplied"] is not False
        or row["graph_success"] is not None
        or row["used_as_graph_positive"] is not False
        or row["comparison_role"] != "matched_quality_reference_only"
    ):
        fail(f"{label} must record B0 without a graph-success claim")
    return row


def _status(value: Any, domain: Sequence[str], label: str) -> str:
    if not isinstance(value, str) or value not in domain:
        fail(f"{label} must be one of {tuple(domain)!r}")
    return value


def _validate_graph_registry(value: Any, label: str) -> Mapping[str, Any]:
    """Validate semantic boundaries that scalar graph scores cannot establish.

    In particular, this prevents an instruction-introduced object from being
    retroactively bound to a source object, prevents a latent/off-screen
    effector from acquiring a fabricated mask/contact label, and requires
    explicit pre-exit evidence before an invisible terminal object may inherit
    a bounded support state.
    """

    registry = _exact_keys(
        value,
        (
            "product_graph_authority",
            "target_usage",
            "nodes",
            "effector",
            "terminal_visibility",
        ),
        label,
    )
    if registry["product_graph_authority"] != "instruction_and_self_anchor_generic_node_only":
        fail(f"{label}.product_graph_authority differs")
    if registry["target_usage"] != "teacher_evaluation_only":
        fail(f"{label}.target_usage differs")

    nodes = _list(registry["nodes"], f"{label}.nodes", nonempty=True)
    node_ids: set[str] = set()
    node_roles: dict[str, str] = {}
    moving_object_ids: set[str] = set()
    source_owned_count = 0
    for index, node_value in enumerate(nodes):
        node_label = f"{label}.nodes[{index}]"
        node = _exact_keys(
            node_value,
            (
                "node_id",
                "role",
                "ownership",
                "introduction_authority",
                "source_node_id",
                "first_reliable_phase",
                "preappearance_state",
                "source_identity_match_required",
                "postappearance_persistence",
                "morph_or_split_from_source",
                "source_noninterference",
            ),
            node_label,
        )
        node_id = _identifier(node["node_id"], f"{node_label}.node_id")
        if node_id in node_ids:
            fail(f"{node_label} duplicates node_id")
        node_ids.add(node_id)
        role = _status(node["role"], NODE_ROLES, f"{node_label}.role")
        node_roles[node_id] = role
        if role == "moving_object":
            moving_object_ids.add(node_id)
        ownership = _status(node["ownership"], NODE_OWNERSHIP, f"{node_label}.ownership")
        _status(
            node["postappearance_persistence"],
            NODE_LIFECYCLE_STATUS,
            f"{node_label}.postappearance_persistence",
        )
        _status(
            node["morph_or_split_from_source"],
            MORPH_STATUS,
            f"{node_label}.morph_or_split_from_source",
        )
        if ownership == "source_owned":
            source_owned_count += 1
            _identifier(node["source_node_id"], f"{node_label}.source_node_id")
            if (
                node["introduction_authority"] != "source_registry"
                or node["first_reliable_phase"] != 0
                or node["preappearance_state"] != "not_applicable"
                or node["source_identity_match_required"] is not True
            ):
                fail(f"{node_label} source-owned lifecycle contract differs")
            _status(
                node["source_noninterference"],
                GATE_STATUS,
                f"{node_label}.source_noninterference",
            )
        else:
            if (
                node["introduction_authority"]
                != "instruction_self_anchor_generic_node"
                or node["source_node_id"] is not None
                or isinstance(node["first_reliable_phase"], bool)
                or not isinstance(node["first_reliable_phase"], int)
                or node["first_reliable_phase"] <= 0
                or node["preappearance_state"]
                != "unresolved_until_first_reliable_appearance"
                or node["source_identity_match_required"] is not False
                or node["source_noninterference"] != "not_applicable"
            ):
                fail(f"{node_label} instruction-introduced lifecycle contract differs")
    if source_owned_count == 0 or not moving_object_ids:
        fail(f"{label} requires a source-owned node and a moving-object node")

    effector = _exact_keys(
        registry["effector"],
        (
            "mode",
            "observed_node_id",
            "mask_identity_claimed",
            "contact_truth_observed",
            "latent_action_evidence",
        ),
        f"{label}.effector",
    )
    evidence = _exact_keys(
        effector["latent_action_evidence"],
        EFFECTOR_EVIDENCE_KEYS,
        f"{label}.effector.latent_action_evidence",
    )
    if effector["mode"] == "observed_node":
        node_id = _identifier(
            effector["observed_node_id"], f"{label}.effector.observed_node_id"
        )
        if node_id not in node_ids or node_roles[node_id] not in (
            "actor",
            "effector_hand",
            "tool",
        ):
            fail(f"{label}.effector observed_node_id is not an observed effector role")
        if type(effector["mask_identity_claimed"]) is not bool:
            fail(f"{label}.effector.mask_identity_claimed must be boolean")
        if type(effector["contact_truth_observed"]) is not bool:
            fail(f"{label}.effector.contact_truth_observed must be boolean")
        if any(evidence[key] != "not_applicable" for key in EFFECTOR_EVIDENCE_KEYS):
            fail(f"{label}.effector latent evidence must be not_applicable when observed")
    elif effector["mode"] == "exogenous_or_offscreen_effector":
        if (
            effector["observed_node_id"] is not None
            or effector["mask_identity_claimed"] is not False
            or effector["contact_truth_observed"] is not False
        ):
            fail(f"{label}.effector latent role cannot claim mask, identity, or contact truth")
        for key in EFFECTOR_EVIDENCE_KEYS:
            _status(evidence[key], GATE_STATUS, f"{label}.effector.latent_action_evidence.{key}")
    else:
        fail(f"{label}.effector.mode differs")

    terminal_rows = _list(
        registry["terminal_visibility"], f"{label}.terminal_visibility", nonempty=True
    )
    terminal_ids: set[str] = set()
    for index, terminal_value in enumerate(terminal_rows):
        terminal_label = f"{label}.terminal_visibility[{index}]"
        row = _exact_keys(
            terminal_value,
            (
                "node_id",
                "mode",
                "preexit_approach_contact_phase_count",
                "hand_release",
                "trajectory_to_known_support_or_frame_boundary",
                "bounded_support_state",
            ),
            terminal_label,
        )
        node_id = _identifier(row["node_id"], f"{terminal_label}.node_id")
        if node_id not in moving_object_ids or node_id in terminal_ids:
            fail(f"{terminal_label} must uniquely reference a moving-object node")
        terminal_ids.add(node_id)
        mode = _status(row["mode"], TERMINAL_MODES, f"{terminal_label}.mode")
        evidence_fields = (
            "hand_release",
            "trajectory_to_known_support_or_frame_boundary",
            "bounded_support_state",
        )
        if mode == "visible_at_terminal":
            if row["preexit_approach_contact_phase_count"] is not None or any(
                row[key] != "not_applicable" for key in evidence_fields
            ):
                fail(f"{terminal_label} visible terminal fields must be not_applicable")
        elif mode == "out_of_frame_after_confirmed_support_release":
            _nonnegative_int(
                row["preexit_approach_contact_phase_count"],
                f"{terminal_label}.preexit_approach_contact_phase_count",
            )
            for key in evidence_fields:
                _status(row[key], GATE_STATUS, f"{terminal_label}.{key}")
        else:
            if row["preexit_approach_contact_phase_count"] is not None or any(
                row[key] != "uncertain" for key in evidence_fields
            ):
                fail(f"{terminal_label} unresolved disappearance must stay uncertain")
    if terminal_ids != moving_object_ids:
        fail(f"{label}.terminal_visibility must cover every moving-object node exactly once")
    return registry


def _validate_consensus(
    value: Any, appearance_ids: Sequence[str], label: str
) -> Sequence[Mapping[str, Any]]:
    rows = _list(value, label, nonempty=True)
    wanted_pairs = {
        tuple(sorted((appearance_ids[left], appearance_ids[right])))
        for left in range(len(appearance_ids))
        for right in range(left + 1, len(appearance_ids))
    }
    observed_pairs: set[tuple[str, str]] = set()
    edge_registry: tuple[str, ...] | None = None
    for index, item in enumerate(rows):
        row_label = f"{label}[{index}]"
        row = _exact_keys(item, ("left", "right", "critical_edges"), row_label)
        left = _identifier(row["left"], f"{row_label}.left")
        right = _identifier(row["right"], f"{row_label}.right")
        if left == right or left not in appearance_ids or right not in appearance_ids:
            fail(f"{row_label} references an invalid appearance pair")
        pair = tuple(sorted((left, right)))
        if pair in observed_pairs:
            fail(f"{row_label} duplicates appearance pair {pair!r}")
        observed_pairs.add(pair)
        edges = _list(row["critical_edges"], f"{row_label}.critical_edges", nonempty=True)
        edge_ids: list[str] = []
        for edge_index, edge_value in enumerate(edges):
            edge_label = f"{row_label}.critical_edges[{edge_index}]"
            edge = _exact_keys(
                edge_value,
                ("edge_id", "status", "aligned_cosine", "normalized_distance"),
                edge_label,
            )
            edge_id = _identifier(edge["edge_id"], f"{edge_label}.edge_id")
            if edge_id in edge_ids:
                fail(f"{edge_label} duplicates a critical edge")
            edge_ids.append(edge_id)
            if edge["status"] not in OBSERVATION_STATUS:
                fail(f"{edge_label}.status differs")
            if edge["status"] == "uncertain":
                if edge["aligned_cosine"] is not None or edge["normalized_distance"] is not None:
                    fail(f"{edge_label} uncertain metrics must be null")
            else:
                _finite_unit(edge["aligned_cosine"], f"{edge_label}.aligned_cosine")
                _finite_unit(edge["normalized_distance"], f"{edge_label}.normalized_distance")
        current_registry = tuple(edge_ids)
        if edge_registry is None:
            edge_registry = current_registry
        elif current_registry != edge_registry:
            fail(f"{row_label} critical-edge registry/order differs")
    if observed_pairs != wanted_pairs:
        fail(f"{label} must contain every appearance pair exactly once")
    return rows


def validate_bundle(value: Any) -> Mapping[str, Any]:
    root = _exact_keys(
        value,
        (
            "schema_version",
            "profile",
            "bundle_id",
            "input_kind",
            "observer_execution",
            "cases",
        ),
        "input",
    )
    if root["schema_version"] != INPUT_SCHEMA:
        fail("input schema_version differs")
    profile = root["profile"]
    if profile not in PROFILES:
        fail(f"profile must be one of {tuple(PROFILES)!r}")
    policy = PROFILES[profile]
    _identifier(root["bundle_id"], "bundle_id")
    if root["input_kind"] != INPUT_KIND:
        fail("input_kind differs")
    execution = _exact_keys(
        root["observer_execution"], ("origin", "validator_runs_extractor"),
        "observer_execution",
    )
    wanted_origin = "synthetic_smoke" if profile == "synthetic_smoke" else "external_observer"
    if execution["origin"] != wanted_origin or execution["validator_runs_extractor"] is not False:
        fail("observer_execution origin or extractor boundary differs")

    cases = _list(root["cases"], "cases", nonempty=True)
    if len(cases) != int(policy["exact_case_count"]):
        fail(f"profile {profile!r} requires exactly {policy['exact_case_count']} cases")
    case_ids: set[str] = set()
    for case_index, case_value in enumerate(cases):
        label = f"cases[{case_index}]"
        case = _exact_keys(
            case_value,
            (
                "case_id",
                "source_media_sha256",
                "target_media_sha256",
                "observer_receipt_sha256",
                "required_axes",
                "graph_registry",
                "target_observations",
                "anchor_appearances",
                "multiappearance_consensus",
                "frozen_base_records",
            ),
            label,
        )
        case_id = _identifier(case["case_id"], f"{label}.case_id")
        if case_id in case_ids:
            fail(f"duplicate case_id {case_id!r}")
        case_ids.add(case_id)
        for key in ("source_media_sha256", "target_media_sha256", "observer_receipt_sha256"):
            _sha256(case[key], f"{label}.{key}")

        required_axes = _list(case["required_axes"], f"{label}.required_axes", nonempty=True)
        if (
            len(set(required_axes)) != len(required_axes)
            or any(axis not in AXES for axis in required_axes)
            or any(axis not in required_axes for axis in MANDATORY_AXES)
            or required_axes != [axis for axis in AXES if axis in required_axes]
        ):
            fail(f"{label}.required_axes differs from canonical ordered axis subset")
        contact_required = "contact_state" in required_axes

        _validate_graph_registry(case["graph_registry"], f"{label}.graph_registry")

        target = _exact_keys(case["target_observations"], TARGET_KEYS, f"{label}.target_observations")
        observation_ids: set[str] = set()
        for key in TARGET_KEYS:
            observation = _validate_observation(
                target[key],
                expected_arm=TARGET_ARMS[key],
                required_axes=required_axes,
                contact_required=contact_required,
                label=f"{label}.target_observations.{key}",
            )
            if observation["observation_id"] in observation_ids:
                fail(f"{label} duplicates observation_id")
            observation_ids.add(observation["observation_id"])

        appearances = _list(case["anchor_appearances"], f"{label}.anchor_appearances", nonempty=True)
        if len(appearances) != int(policy["anchor_appearance_count"]):
            fail(f"{label} anchor appearance count differs from profile")
        appearance_ids: list[str] = []
        anchor_assets: set[str] = set()
        middle_receipts: set[str] = set()
        for appearance_index, appearance_value in enumerate(appearances):
            appearance_label = f"{label}.anchor_appearances[{appearance_index}]"
            appearance = _exact_keys(
                appearance_value,
                (
                    "appearance_id",
                    "anchor_asset_sha256",
                    "middle_feature_receipt_sha256",
                    "observations",
                ),
                appearance_label,
            )
            appearance_id = _identifier(appearance["appearance_id"], f"{appearance_label}.appearance_id")
            asset_sha = _sha256(appearance["anchor_asset_sha256"], f"{appearance_label}.anchor_asset_sha256")
            receipt_sha = _sha256(
                appearance["middle_feature_receipt_sha256"],
                f"{appearance_label}.middle_feature_receipt_sha256",
            )
            if appearance_id in appearance_ids or asset_sha in anchor_assets or receipt_sha in middle_receipts:
                fail(f"{appearance_label} appearance/asset/receipt must be unique")
            appearance_ids.append(appearance_id)
            anchor_assets.add(asset_sha)
            middle_receipts.add(receipt_sha)
            observations = _exact_keys(
                appearance["observations"], ANCHOR_KEYS, f"{appearance_label}.observations"
            )
            for key in ANCHOR_KEYS:
                observation = _validate_observation(
                    observations[key],
                    expected_arm=ANCHOR_ARMS[key],
                    required_axes=required_axes,
                    contact_required=contact_required,
                    label=f"{appearance_label}.observations.{key}",
                )
                if observation["observation_id"] in observation_ids:
                    fail(f"{label} duplicates observation_id")
                observation_ids.add(observation["observation_id"])

        _validate_consensus(
            case["multiappearance_consensus"], appearance_ids,
            f"{label}.multiappearance_consensus",
        )
        frozen = _list(case["frozen_base_records"], f"{label}.frozen_base_records", nonempty=True)
        frozen_ids: set[str] = set()
        frozen_seeds: set[int] = set()
        for frozen_index, frozen_value in enumerate(frozen):
            record = _validate_frozen_base(
                frozen_value, f"{label}.frozen_base_records[{frozen_index}]"
            )
            if record["record_id"] in frozen_ids or record["seed"] in frozen_seeds:
                fail(f"{label} duplicates Frozen Base record or seed")
            frozen_ids.add(record["record_id"])
            frozen_seeds.add(record["seed"])
    return root


def _score(observation: Mapping[str, Any], required_axes: Sequence[str]) -> float | None:
    values = []
    for axis in required_axes:
        measure = observation["scores"][axis]
        if measure["status"] == "uncertain":
            return None
        values.append(float(measure["value"]))
    return min(values)


def _axis_value(observation: Mapping[str, Any], axis: str) -> float | None:
    measure = observation["scores"][axis]
    return None if measure["status"] == "uncertain" else float(measure["value"])


def _contrast(
    positive: Mapping[str, Any],
    negative: Mapping[str, Any],
    *,
    required_axes: Sequence[str],
    discriminative_axis: str,
    margin: float,
) -> Mapping[str, Any]:
    positive_score = _score(positive, required_axes)
    negative_score = _score(negative, required_axes)
    positive_axis = _axis_value(positive, discriminative_axis)
    negative_axis = _axis_value(negative, discriminative_axis)
    uncertain = any(
        value is None for value in (positive_score, negative_score, positive_axis, negative_axis)
    )
    if uncertain:
        return {
            "discriminative_axis": discriminative_axis,
            "minimum_margin": margin,
            "positive_graph_score": positive_score,
            "negative_graph_score": negative_score,
            "overall_margin": None,
            "axis_margin": None,
            "uncertain": True,
            "passed": False,
        }
    overall_margin = float(positive_score - negative_score)
    axis_margin = float(positive_axis - negative_axis)
    return {
        "discriminative_axis": discriminative_axis,
        "minimum_margin": margin,
        "positive_graph_score": positive_score,
        "negative_graph_score": negative_score,
        "overall_margin": overall_margin,
        "axis_margin": axis_margin,
        "uncertain": False,
        "passed": overall_margin >= margin and axis_margin >= margin,
    }


def _observation_has_uncertainty(observation: Mapping[str, Any]) -> bool:
    if any(item["status"] == "uncertain" for item in observation["scores"].values()):
        return True
    return any(item["status"] == "uncertain" for item in observation["hard_gates"].values())


def _hard_gate_result(
    observations: Sequence[Mapping[str, Any]],
    *,
    contact_required: bool,
    policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    identity_rows = [row["hard_gates"]["identity"] for row in observations]
    contact_rows = [row["hard_gates"]["contact"] for row in observations]
    terminal_rows = [row["hard_gates"]["terminal"] for row in observations]
    uncertainty_rows = [row["hard_gates"]["uncertainty"] for row in observations]

    identity = all(
        row["status"] == "pass"
        and float(row["idf1"]) >= float(policy["identity_idf1_min"])
        and int(row["id_switch_count"]) == 0
        for row in identity_rows
    )
    if contact_required:
        contact = all(
            row["status"] == "pass"
            and float(row["macro_f1"]) >= float(policy["contact_macro_f1_min"])
            for row in contact_rows
        )
    else:
        contact = all(row["status"] == "not_applicable" for row in contact_rows)
    terminal = all(
        row["status"] == "pass"
        and float(row["macro_f1"]) >= float(policy["terminal_macro_f1_min"])
        and int(row["hold_rgb_frames"]) >= int(policy["terminal_hold_rgb_frames_min"])
        for row in terminal_rows
    )
    uncertainty = all(
        row["status"] == "pass"
        and int(row["unresolved_required_nodes"]) == 0
        and int(row["unresolved_required_edges"]) == 0
        for row in uncertainty_rows
    )
    return {
        "identity": identity,
        "contact": contact,
        "terminal": terminal,
        "uncertainty": uncertainty,
        "passed": identity and contact and terminal and uncertainty,
    }


def _consensus_result(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> Mapping[str, Any]:
    pair_results = []
    any_uncertain = False
    for row in rows:
        edges = []
        for edge in row["critical_edges"]:
            uncertain = edge["status"] == "uncertain"
            any_uncertain = any_uncertain or uncertain
            passed = (
                not uncertain
                and float(edge["aligned_cosine"]) >= float(policy["critical_edge_cosine_min"])
                and float(edge["normalized_distance"]) <= float(policy["critical_edge_distance_max"])
            )
            edges.append(
                {
                    "edge_id": edge["edge_id"],
                    "aligned_cosine": edge["aligned_cosine"],
                    "normalized_distance": edge["normalized_distance"],
                    "uncertain": uncertain,
                    "passed": passed,
                }
            )
        pair_results.append(
            {
                "left": row["left"],
                "right": row["right"],
                "critical_edges": edges,
                "passed": all(item["passed"] for item in edges),
            }
        )
    return {
        "minimum_aligned_cosine": policy["critical_edge_cosine_min"],
        "maximum_normalized_distance": policy["critical_edge_distance_max"],
        "any_uncertain": any_uncertain,
        "pairs": pair_results,
        "passed": not any_uncertain and all(row["passed"] for row in pair_results),
    }


def _graph_registry_result(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    node_rows = registry["nodes"]
    node_results = []
    any_uncertain = False
    for node in node_rows:
        persistence = node["postappearance_persistence"]
        morph = node["morph_or_split_from_source"]
        noninterference = node["source_noninterference"]
        uncertain = (
            persistence == "uncertain"
            or morph == "uncertain"
            or noninterference == "uncertain"
        )
        any_uncertain = any_uncertain or uncertain
        persistence_passed = persistence == "pass"
        no_morph_or_split = morph == "not_detected"
        noninterference_passed = (
            noninterference == "pass"
            if node["ownership"] == "source_owned"
            else noninterference == "not_applicable"
        )
        passed = persistence_passed and no_morph_or_split and noninterference_passed
        node_results.append(
            {
                "node_id": node["node_id"],
                "role": node["role"],
                "ownership": node["ownership"],
                "source_identity_match_required": node["source_identity_match_required"],
                "first_reliable_phase": node["first_reliable_phase"],
                "persistence_passed": persistence_passed,
                "no_morph_or_split": no_morph_or_split,
                "source_noninterference_passed": noninterference_passed,
                "uncertain": uncertain,
                "passed": passed,
            }
        )

    effector = registry["effector"]
    if effector["mode"] == "exogenous_or_offscreen_effector":
        evidence = effector["latent_action_evidence"]
        effector_uncertain = any(value == "uncertain" for value in evidence.values())
        effector_passed = all(value == "pass" for value in evidence.values())
        any_uncertain = any_uncertain or effector_uncertain
    else:
        effector_uncertain = False
        effector_passed = True
    effector_result = {
        "mode": effector["mode"],
        "mask_identity_claimed": effector["mask_identity_claimed"],
        "contact_truth_observed": effector["contact_truth_observed"],
        "latent_action_evidence": effector["latent_action_evidence"],
        "uncertain": effector_uncertain,
        "passed": effector_passed,
    }

    terminal_results = []
    for row in registry["terminal_visibility"]:
        if row["mode"] == "visible_at_terminal":
            uncertain = False
            passed = True
        elif row["mode"] == "out_of_frame_after_confirmed_support_release":
            evidence = (
                row["hand_release"],
                row["trajectory_to_known_support_or_frame_boundary"],
                row["bounded_support_state"],
            )
            uncertain = any(value == "uncertain" for value in evidence)
            passed = (
                row["preexit_approach_contact_phase_count"] >= 2
                and all(value == "pass" for value in evidence)
            )
        else:
            uncertain = True
            passed = False
        any_uncertain = any_uncertain or uncertain
        terminal_results.append(
            {
                **row,
                "uncertain": uncertain,
                "passed": passed,
            }
        )

    identity_passed = all(row["passed"] for row in node_results)
    terminal_passed = all(row["passed"] for row in terminal_results)
    return {
        "product_graph_authority": registry["product_graph_authority"],
        "target_usage": registry["target_usage"],
        "nodes": node_results,
        "effector": effector_result,
        "terminal_visibility": terminal_results,
        "identity": identity_passed,
        "contact_or_latent_action_grounding": effector_passed,
        "terminal": terminal_passed,
        "uncertainty": not any_uncertain,
        "any_uncertain": any_uncertain,
        "passed": identity_passed and effector_passed and terminal_passed and not any_uncertain,
    }


def _evaluate_case(case: Mapping[str, Any], policy: Mapping[str, Any]) -> Mapping[str, Any]:
    required_axes = tuple(case["required_axes"])
    contact_required = "contact_state" in required_axes
    margin = float(policy["comparison_margin"])
    target = case["target_observations"]
    target_controls = {
        key: _contrast(
            target["forward"], target[key], required_axes=required_axes,
            discriminative_axis=CONTRAST_AXIS[key], margin=margin,
        )
        for key in ("reverse", "shuffle", "source_noop")
    }

    anchor_results = []
    action_observations: list[Mapping[str, Any]] = [target["forward"]]
    all_observations: list[Mapping[str, Any]] = list(target.values())
    for appearance in case["anchor_appearances"]:
        observations = appearance["observations"]
        action_observations.append(observations["action"])
        all_observations.extend(observations.values())
        controls = {
            key: _contrast(
                observations["action"], observations[key], required_axes=required_axes,
                discriminative_axis=CONTRAST_AXIS[key], margin=margin,
            )
            for key in ("noop", "reverse", "static")
        }
        anchor_results.append(
            {
                "appearance_id": appearance["appearance_id"],
                "controls": controls,
                "passed": all(row["passed"] for row in controls.values()),
            }
        )

    observation_hard = _hard_gate_result(
        action_observations, contact_required=contact_required, policy=policy
    )
    registry = _graph_registry_result(case["graph_registry"])
    hard = {
        "identity": observation_hard["identity"] and registry["identity"],
        "contact": (
            observation_hard["contact"]
            and registry["contact_or_latent_action_grounding"]
        ),
        "terminal": observation_hard["terminal"] and registry["terminal"],
        "uncertainty": observation_hard["uncertainty"] and registry["uncertainty"],
    }
    hard["passed"] = all(hard.values())
    consensus = _consensus_result(case["multiappearance_consensus"], policy)
    observation_uncertain = any(_observation_has_uncertainty(row) for row in all_observations)
    any_uncertain = (
        observation_uncertain or consensus["any_uncertain"] or registry["any_uncertain"]
    )
    target_pass = all(row["passed"] for row in target_controls.values())
    anchor_pass = all(row["passed"] for row in anchor_results)
    admitted = target_pass and anchor_pass and consensus["passed"] and hard["passed"] and not any_uncertain
    failure_reasons = []
    if any_uncertain:
        failure_reasons.append("uncertain_evidence_present")
    if not target_pass:
        failure_reasons.append("target_forward_did_not_beat_all_controls")
    if not anchor_pass:
        failure_reasons.append("anchor_action_did_not_beat_all_controls")
    if not consensus["passed"]:
        failure_reasons.append("multiappearance_consensus_failed")
    if not hard["passed"]:
        failure_reasons.append("identity_contact_terminal_or_uncertainty_hard_gate_failed")
    return {
        "case_id": case["case_id"],
        "required_axes": list(required_axes),
        "target_controls": target_controls,
        "target_all_controls_passed": target_pass,
        "anchor_appearances": anchor_results,
        "anchor_all_controls_passed": anchor_pass,
        "multiappearance_consensus": consensus,
        "graph_registry": registry,
        "hard_gates": hard,
        "any_uncertain": any_uncertain,
        "admitted": admitted,
        "failure_reasons": failure_reasons,
    }


def evaluate_bundle(
    value: Mapping[str, Any], *, input_file_sha256: str | None = None
) -> Mapping[str, Any]:
    root = validate_bundle(value)
    profile = root["profile"]
    policy = dict(PROFILES[profile])
    case_results = [_evaluate_case(case, policy) for case in root["cases"]]
    case_count = len(case_results)

    target_counts = {
        control: sum(row["target_controls"][control]["passed"] for row in case_results)
        for control in ("reverse", "shuffle", "source_noop")
    }
    anchor_counts = {
        control: sum(
            all(
                appearance["controls"][control]["passed"]
                for appearance in row["anchor_appearances"]
            )
            for row in case_results
        )
        for control in ("noop", "reverse", "static")
    }
    consensus_count = sum(row["multiappearance_consensus"]["passed"] for row in case_results)
    hard_count = sum(row["hard_gates"]["passed"] for row in case_results)
    hard_axis_counts = {
        axis: sum(row["hard_gates"][axis] for row in case_results)
        for axis in ("identity", "contact", "terminal", "uncertainty")
    }
    uncertainty_count = sum(row["any_uncertain"] for row in case_results)
    minimum_control = int(policy["minimum_control_win_cases"])
    minimum_hard = int(policy["minimum_hard_gate_pass_cases"])
    global_gates = {
        "target_forward_over_reverse": target_counts["reverse"] >= minimum_control,
        "target_forward_over_shuffle": target_counts["shuffle"] >= minimum_control,
        "target_forward_over_source_noop": target_counts["source_noop"] >= minimum_control,
        "anchor_action_over_noop": anchor_counts["noop"] >= minimum_control,
        "anchor_action_over_reverse": anchor_counts["reverse"] >= minimum_control,
        "anchor_action_over_static": anchor_counts["static"] >= minimum_control,
        "multiappearance_consensus": consensus_count >= int(policy["minimum_consensus_pass_cases"]),
        "identity_hard_gate": hard_axis_counts["identity"] >= minimum_hard,
        "contact_hard_gate": hard_axis_counts["contact"] >= minimum_hard,
        "terminal_hard_gate": hard_axis_counts["terminal"] >= minimum_hard,
        "uncertainty_hard_gate": hard_axis_counts["uncertainty"] >= minimum_hard,
        "all_hard_gates": hard_count >= minimum_hard,
        "no_uncertainty_anywhere": uncertainty_count == 0,
    }
    admitted = all(global_gates.values())
    summary = {
        "case_count": case_count,
        "target_control_win_case_counts": target_counts,
        "anchor_control_win_case_counts": anchor_counts,
        "consensus_pass_case_count": consensus_count,
        "hard_gate_axis_pass_case_counts": hard_axis_counts,
        "hard_gate_pass_case_count": hard_count,
        "uncertain_case_count": uncertainty_count,
        "fully_admitted_case_count": sum(row["admitted"] for row in case_results),
        "global_gates": global_gates,
        "mechanical_admission_passed": admitted,
    }
    frozen_registry = [
        {
            "case_id": case["case_id"],
            "records": case["frozen_base_records"],
            "graph_success_claimed": False,
            "used_as_graph_positive": False,
        }
        for case in root["cases"]
    ]
    input_digest = input_file_sha256 or object_sha256(root)
    _sha256(input_digest, "input file/object digest")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "MECHANICALLY_ADMITTED" if admitted else "REJECTED",
        "input": {
            "schema_version": root["schema_version"],
            "bundle_id": root["bundle_id"],
            "profile": profile,
            "sha256": input_digest,
        },
        "policy": policy,
        "case_results": case_results,
        "summary": summary,
        "frozen_base_registry": frozen_registry,
        "claim_limits": {
            "validator_ran_sam2": False,
            "validator_ran_cotracker": False,
            "validator_ran_dino_or_vjepa": False,
            "validator_decoded_video": False,
            "validator_ran_renderer": False,
            "validator_verified_media_or_observer_authenticity": False,
            "frozen_base_graph_success_claimed": False,
            "renderer_effectiveness_claimed": False,
            "stable_transferable_action_representation_claimed": False,
            "scientific_claim_authorized": False,
        },
    }
    receipt["receipt_sha256"] = object_sha256(receipt)
    return receipt


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise OCEGZ0AdmissionError(f"refusing to overwrite output: {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    input_path = args.input.resolve(strict=True)
    output_path = args.output.expanduser().absolute()
    if input_path == output_path:
        fail("input and output paths must differ")
    value = load_json_strict(input_path)
    receipt = evaluate_bundle(value, input_file_sha256=file_sha256(input_path))
    write_json_exclusive(output_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output_path),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
