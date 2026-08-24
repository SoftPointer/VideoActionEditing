#!/usr/bin/env python3
"""Detached seven-axis audit packets for Bernini SAIC Stage-B.

This module intentionally provides *diagnostic* pairing only.  Its builder
calls a live, qualified :class:`SAICEventRewardBoundary`, but the serialized
result is caller-re-signable and carries no provenance.  The whole-frame
preservation measurements likewise arrive as caller-supplied detached scalars;
no qualified evaluator runtime or trust root is present here.  Consequently
no receipt produced by this module can authorize an optimizer step.

The hard audit keeps seven non-compensating axes.  ``quality`` is the minimum
of appearance, technical quality, and temporal consistency, and those three
components are gated independently as well.  ``source_bind`` uses both wrong
source and dropped-source counterfactual errors.  ``inverse`` is a monotone
transform of inverse reconstruction error.  Pure-T2V visuals, targets, donor
media, masks, pose, flow, tracks, and trajectories have no scoring route.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

try:  # Package import.
    from . import saic_event_reward_v1 as event_reward
    from . import saic_rollout_preference_set_v1 as rollout_contract
except ImportError:  # Direct import used by the repository tests/scripts.
    import saic_event_reward_v1 as event_reward
    import saic_rollout_preference_set_v1 as rollout_contract

try:
    import torch
except ImportError:  # Validation reports a closed failure at call time.
    torch = None


SCHEMA_VERSION = "bernini-saic-seven-axis-candidate-audit-v1"
MEASUREMENT_BUNDLE_SCHEMA_VERSION = "bernini-saic-whole-frame-measurement-bundle-v1"
EVENT_MEDIA_BINDING_SCHEMA_VERSION = "bernini-saic-event-media-binding-v1"
PAIR_SCHEMA_VERSION = "bernini-saic-seven-axis-diagnostic-pair-v1"
PREFERENCE_SET_SCHEMA_VERSION = "bernini-saic-seven-axis-diagnostic-set-v1"

BOOTSTRAP_MODE = "bootstrap"
STRICT_MODE = "strict"
MODES = (BOOTSTRAP_MODE, STRICT_MODE)
ARMS = ("dog", "human")
FRAME_COUNT = 81
PURE_T2V_ROLE = "qualified_event_critic_calibration_only"

PRESERVATION_AXES = (
    "identity",
    "camera",
    "background",
    "non_target",
    "quality",
    "source_bind",
    "inverse",
)
EXPLICIT_QUALITY_COMPONENTS = (
    "appearance_preservation",
    "technical_quality",
    "temporal_consistency",
)
UNIT_COMPONENTS = (
    "identity_preservation",
    "camera_preservation",
    "background_preservation",
    "non_target_motion_preservation",
    *EXPLICIT_QUALITY_COMPONENTS,
)
ERROR_COMPONENTS = (
    "correct_source_reconstruction_error",
    "wrong_source_reconstruction_error",
    "dropped_source_reconstruction_error",
    "inverse_reconstruction_error",
)
MEASUREMENT_COMPONENTS = (*UNIT_COMPONENTS, *ERROR_COMPONENTS)


def _expected_arms() -> tuple[str, str]:
    """Reconstruct the fixed dual-arm contract independently of globals."""

    return ("dog", "human")


def _expected_modes() -> tuple[str, str]:
    return ("bootstrap", "strict")


def _expected_phase_order() -> tuple[str, str, str, str]:
    return ("onset", "transition", "completion", "hold")


def _expected_frame_count() -> int:
    return 81


def _expected_pure_t2v_role() -> str:
    return "qualified_event_critic_calibration_only"


def _expected_preservation_axes() -> tuple[str, ...]:
    return (
        "identity",
        "camera",
        "background",
        "non_target",
        "quality",
        "source_bind",
        "inverse",
    )


def _expected_quality_components() -> tuple[str, str, str]:
    return (
        "appearance_preservation",
        "technical_quality",
        "temporal_consistency",
    )


def _expected_unit_components() -> tuple[str, ...]:
    return (
        "identity_preservation",
        "camera_preservation",
        "background_preservation",
        "non_target_motion_preservation",
        *_expected_quality_components(),
    )


def _expected_error_components() -> tuple[str, ...]:
    return (
        "correct_source_reconstruction_error",
        "wrong_source_reconstruction_error",
        "dropped_source_reconstruction_error",
        "inverse_reconstruction_error",
    )


def _expected_measurement_components() -> tuple[str, ...]:
    return (*_expected_unit_components(), *_expected_error_components())


def _expected_thresholds() -> dict[str, Any]:
    """Return the only preregistered diagnostic threshold policy."""

    axes = _expected_preservation_axes()
    quality_components = _expected_quality_components()
    return {
        "minimum_event_delta": 0.20,
        "axis_absolute_floors": {
            "identity": 0.75,
            "camera": 0.75,
            "background": 0.75,
            "non_target": 0.75,
            "quality": 0.75,
            "source_bind": 0.10,
            "inverse": 0.75,
        },
        "axis_baseline_slacks": {axis: 0.10 for axis in axes},
        "axis_pair_tolerances": {axis: 0.02 for axis in axes},
        "component_absolute_floors": {
            component: 0.75 for component in quality_components
        },
        "component_baseline_slacks": {
            component: 0.10 for component in quality_components
        },
        "component_pair_tolerances": {
            component: 0.02 for component in quality_components
        },
    }


def _expected_input_closure() -> dict[str, bool]:
    # Literals are deliberately reconstructed here.  Validation never trusts
    # the exported module object, which callers can rebind in Python.
    return {
        "candidate_output_media_read": True,
        "registered_source_media_read": True,
        "unregistered_media_read": False,
        "pure_t2v_visual_condition_target_noise_or_donor_used": False,
        "paired_target_proposal_or_donor_read": False,
        "mask_pose_flow_track_trajectory_read": False,
        "whole_frame_scoring": True,
        "selected_actor_localization_used": False,
    }


def _expected_authority_contract() -> dict[str, Any]:
    return {
        "serialized_builder_event_boundary_call_authenticated": False,
        "serialized_packet_provenance": "none_caller_re_signable_diagnostic_only",
        "measurement_runtime_qualified": False,
        "measurement_receipts_caller_supplied": True,
        "media_evaluator_executed_by_this_module": False,
        "optimizer_authority": False,
        "event_absolute_and_pair_delta_are_distinct": True,
        "seven_axes_are_noncompensating": True,
        "appearance_technical_temporal_independently_gated": True,
        "pure_t2v_endpoint_condition_target_noise_or_donor_used": False,
        "mask_pose_flow_track_trajectory_used": False,
        "semantic_localization_claimed": False,
        "multi_actor_localization_claimed": False,
    }


# Read-only convenience snapshots.  The implementation reconstructs literals
# above, so rebinding either public-ish module global cannot weaken validation.
_INPUT_CLOSURE = MappingProxyType(_expected_input_closure())
_AUTHORITY_CONTRACT = MappingProxyType(_expected_authority_contract())

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")

_BINDING_FIELDS = frozenset(
    {
        "candidate_id",
        "arm",
        "source_id",
        "instruction_id",
        "policy_sha256",
        "source_media_sha256",
        "output_media_sha256",
        "endpoint_latent_sha256",
        "legacy_candidate_digest",
        "rollout_receipt_digest",
        "codec_receipt_digest",
    }
)
_MEASUREMENT_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "legacy_candidate_digest",
        "rollout_receipt_digest",
        "codec_receipt_digest",
        "source_media_sha256",
        "output_media_sha256",
        "evaluator_set_sha256",
        "upstream_receipt_digest_by_component",
        "score_value_by_component",
        "whole_frame_exact81",
        "frame_count",
        "score_tensor_detached_fp32",
        "input_closure",
        "bundle_digest",
    }
)
_EVENT_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "legacy_candidate_digest",
        "rollout_receipt_digest",
        "source_media_sha256",
        "output_media_sha256",
        "evaluated_media_sha256",
        "event_rollout_id",
        "event_score_packet_digest",
        "critic_checkpoint_sha256",
        "critic_qualification_receipt_digest",
        "frozen_critic",
        "decoded_exact81",
        "frame_count",
        "score_only_output",
        "pure_t2v_role",
        "pure_t2v_media_read_during_candidate_scoring",
        "pure_t2v_latent_read_during_candidate_scoring",
        "pure_t2v_noise_read_during_candidate_scoring",
        "pure_t2v_condition_used",
        "pure_t2v_target_used",
        "pure_t2v_donor_used",
        "paired_target_or_donor_read",
        "mask_pose_flow_track_trajectory_read",
        "binding_digest",
    }
)
_EVENT_EVIDENCE_FIELDS = frozenset(
    {
        "mode",
        "event_rollout_id",
        "action_family",
        "critic_checkpoint_sha256",
        "critic_qualification_receipt_digest",
        "event_score_packet_digest",
        "stage_margins",
        "weakest_phase",
        "weakest_margin",
        "relative_action_margin_pass",
        "relative_pairing_eligible",
        "absolute_action_pass_by_phase",
        "absolute_margin_pass_by_phase",
        "absolute_four_stage_pass_by_phase",
        "absolute_four_stage_pass",
        "event_decision_digest",
    }
)
_PACKET_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_binding",
        "event_evidence",
        "event_media_binding_receipt",
        "component_scores",
        "axis_scores",
        "measurement_bundle",
        "authority_contract",
        "packet_digest",
    }
)


class SAICSevenAxisScorePacketError(ValueError):
    """The detached audit contract was violated."""


def _fail(message: str) -> None:
    raise SAICSevenAxisScorePacketError(message)


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
        raise SAICSevenAxisScorePacketError(
            "value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    if any(type(key) is not str for key in value):
        _fail(f"{label} keys must be strings")
    actual = set(value)
    if actual != fields:
        _fail(
            f"{label} schema differs; missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        _fail(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        _fail(f"{label} must be a canonical identifier")
    return value


def _bool(value: Any, *, label: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be boolean")
    if expected is not None and value is not expected:
        _fail(f"{label} must be {expected}")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite numeric")
    return result


def _unit(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result <= 1.0:
        _fail(f"{label} must lie in [0,1]")
    return result


def _nonnegative(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result < 0.0:
        _fail(f"{label} must be nonnegative")
    return result


def _sealed(row: Mapping[str, Any], field: str, *, label: str) -> None:
    digest = _sha(row[field], label=f"{label}.{field}")
    body = {key: value for key, value in row.items() if key != field}
    if digest != object_sha256(body):
        _fail(f"{label} digest differs")


def _exact_named_map(
    value: Any,
    names: Sequence[str],
    *,
    label: str,
    parser,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        _fail(f"{label} must contain exactly {list(names)}")
    return {name: parser(value[name], label=f"{label}.{name}") for name in names}


def _candidate_binding(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "arm": candidate["arm"],
        "source_id": candidate["source_id"],
        "instruction_id": candidate["instruction_id"],
        "policy_sha256": candidate["policy_sha256"],
        "source_media_sha256": candidate["source_media_sha256"],
        "output_media_sha256": candidate["output_media_sha256"],
        "endpoint_latent_sha256": candidate["endpoint_latent_sha256"],
        "legacy_candidate_digest": candidate["candidate_digest"],
        "rollout_receipt_digest": candidate["rollout_receipt"]["receipt_digest"],
        "codec_receipt_digest": candidate["codec_reencode_receipt"]["receipt_digest"],
    }


def _validate_binding(value: Any) -> dict[str, Any]:
    row = _closed(value, _BINDING_FIELDS, label="candidate binding")
    _safe_id(row["candidate_id"], label="candidate binding.candidate_id")
    if row["arm"] not in _expected_arms():
        _fail("candidate binding.arm must be dog or human")
    _safe_id(row["source_id"], label="candidate binding.source_id")
    _safe_id(row["instruction_id"], label="candidate binding.instruction_id")
    for key in (
        "policy_sha256",
        "source_media_sha256",
        "output_media_sha256",
        "endpoint_latent_sha256",
        "legacy_candidate_digest",
        "rollout_receipt_digest",
        "codec_receipt_digest",
    ):
        _sha(row[key], label=f"candidate binding.{key}")
    return json.loads(canonical_json_bytes(row).decode("ascii"))


def _validate_input_closure(value: Any, *, label: str) -> dict[str, bool]:
    expected_closure = _expected_input_closure()
    row = _closed(value, frozenset(expected_closure), label=label)
    for key, expected in expected_closure.items():
        _bool(row[key], label=f"{label}.{key}", expected=expected)
    return expected_closure


def _validate_measurement_bundle(
    value: Any, binding: Mapping[str, Any]
) -> dict[str, Any]:
    measurement_components = _expected_measurement_components()
    unit_components = _expected_unit_components()
    error_components = _expected_error_components()
    row = _closed(value, _MEASUREMENT_FIELDS, label="measurement bundle")
    if row["schema_version"] != MEASUREMENT_BUNDLE_SCHEMA_VERSION:
        _fail("measurement bundle schema_version differs")
    for key in (
        "candidate_id",
        "legacy_candidate_digest",
        "rollout_receipt_digest",
        "codec_receipt_digest",
        "source_media_sha256",
        "output_media_sha256",
    ):
        if row[key] != binding[key]:
            _fail(f"measurement bundle {key} differs from candidate binding")
    _sha(row["evaluator_set_sha256"], label="measurement bundle.evaluator_set_sha256")
    upstream = _exact_named_map(
        row["upstream_receipt_digest_by_component"],
        measurement_components,
        label="measurement bundle.upstream_receipt_digest_by_component",
        parser=_sha,
    )
    scores: dict[str, float] = {}
    score_map = row["score_value_by_component"]
    if not isinstance(score_map, Mapping) or set(score_map) != set(
        measurement_components
    ):
        _fail("measurement bundle.score_value_by_component keys differ")
    for component in unit_components:
        scores[component] = _unit(score_map[component], label=f"measurement.{component}")
    for component in error_components:
        scores[component] = _nonnegative(
            score_map[component], label=f"measurement.{component}"
        )
    _bool(row["whole_frame_exact81"], label="whole_frame_exact81", expected=True)
    if (
        type(row["frame_count"]) is not int
        or row["frame_count"] != _expected_frame_count()
    ):
        _fail("measurement bundle.frame_count must be exact integer 81")
    _bool(
        row["score_tensor_detached_fp32"],
        label="score_tensor_detached_fp32",
        expected=True,
    )
    closure = _validate_input_closure(
        row["input_closure"], label="measurement bundle.input_closure"
    )
    _sealed(row, "bundle_digest", label="measurement bundle")
    normalized = dict(row)
    normalized["upstream_receipt_digest_by_component"] = upstream
    normalized["score_value_by_component"] = scores
    normalized["input_closure"] = closure
    return json.loads(canonical_json_bytes(normalized).decode("ascii"))


def _validate_event_binding(
    value: Any,
    binding: Mapping[str, Any],
    event_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    row = _closed(value, _EVENT_BINDING_FIELDS, label="event media binding")
    if row["schema_version"] != EVENT_MEDIA_BINDING_SCHEMA_VERSION:
        _fail("event media binding schema_version differs")
    for key in (
        "candidate_id",
        "legacy_candidate_digest",
        "rollout_receipt_digest",
        "source_media_sha256",
        "output_media_sha256",
    ):
        if row[key] != binding[key]:
            _fail(f"event media binding {key} differs")
    if row["evaluated_media_sha256"] != binding["output_media_sha256"]:
        _fail("event evaluator did not score the registered candidate output")
    links = {
        "event_rollout_id": "event_rollout_id",
        "event_score_packet_digest": "event_score_packet_digest",
        "critic_checkpoint_sha256": "critic_checkpoint_sha256",
        "critic_qualification_receipt_digest": "critic_qualification_receipt_digest",
    }
    for receipt_key, evidence_key in links.items():
        if row[receipt_key] != event_evidence[evidence_key]:
            _fail(f"event media binding {receipt_key} differs")
    for key in (
        "legacy_candidate_digest",
        "rollout_receipt_digest",
        "source_media_sha256",
        "output_media_sha256",
        "evaluated_media_sha256",
        "event_score_packet_digest",
        "critic_checkpoint_sha256",
        "critic_qualification_receipt_digest",
    ):
        _sha(row[key], label=f"event media binding.{key}")
    if row["pure_t2v_role"] != _expected_pure_t2v_role():
        _fail("pure-T2V role differs")
    for key in ("frozen_critic", "decoded_exact81", "score_only_output"):
        _bool(row[key], label=f"event media binding.{key}", expected=True)
    if (
        type(row["frame_count"]) is not int
        or row["frame_count"] != _expected_frame_count()
    ):
        _fail("event media binding.frame_count must be exact integer 81")
    for key in (
        "pure_t2v_media_read_during_candidate_scoring",
        "pure_t2v_latent_read_during_candidate_scoring",
        "pure_t2v_noise_read_during_candidate_scoring",
        "pure_t2v_condition_used",
        "pure_t2v_target_used",
        "pure_t2v_donor_used",
        "paired_target_or_donor_read",
        "mask_pose_flow_track_trajectory_read",
    ):
        _bool(row[key], label=f"event media binding.{key}", expected=False)
    _sealed(row, "binding_digest", label="event media binding")
    return json.loads(canonical_json_bytes(row).decode("ascii"))


def _event_evidence(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": decision["mode"],
        "event_rollout_id": decision["rollout_id"],
        "action_family": decision["action_family"],
        "critic_checkpoint_sha256": decision["critic_checkpoint_sha256"],
        "critic_qualification_receipt_digest": decision[
            "qualification_receipt_digest"
        ],
        "event_score_packet_digest": decision["candidate_score_packet_digest"],
        "stage_margins": decision["stage_margins"],
        "weakest_phase": decision["weakest_phase"],
        "weakest_margin": decision["weakest_margin"],
        "relative_action_margin_pass": decision["relative_action_margin_pass"],
        "relative_pairing_eligible": decision["relative_pairing_eligible"],
        "absolute_action_pass_by_phase": decision["absolute_action_pass_by_phase"],
        "absolute_margin_pass_by_phase": decision["absolute_margin_pass_by_phase"],
        "absolute_four_stage_pass_by_phase": decision[
            "absolute_four_stage_pass_by_phase"
        ],
        "absolute_four_stage_pass": decision["absolute_four_stage_pass"],
        "event_decision_digest": decision["decision_digest"],
    }


def _validate_phase_bool_map(value: Any, *, label: str) -> dict[str, bool]:
    phases = _expected_phase_order()
    if not isinstance(value, Mapping) or set(value) != set(phases):
        _fail(f"{label} phase keys differ")
    return {
        phase: _bool(value[phase], label=f"{label}.{phase}")
        for phase in phases
    }


def _validate_event_evidence(value: Any) -> dict[str, Any]:
    row = _closed(value, _EVENT_EVIDENCE_FIELDS, label="event evidence")
    phases = _expected_phase_order()
    if row["mode"] not in _expected_modes():
        _fail("event evidence mode differs")
    _safe_id(row["event_rollout_id"], label="event evidence.event_rollout_id")
    _safe_id(row["action_family"], label="event evidence.action_family")
    for key in (
        "critic_checkpoint_sha256",
        "critic_qualification_receipt_digest",
        "event_score_packet_digest",
        "event_decision_digest",
    ):
        _sha(row[key], label=f"event evidence.{key}")
    margins = _exact_named_map(
        row["stage_margins"],
        phases,
        label="event evidence.stage_margins",
        parser=_finite,
    )
    weakest_phase = min(
        phases,
        key=lambda phase: (
            margins[phase],
            phases.index(phase),
        ),
    )
    if row["weakest_phase"] != weakest_phase:
        _fail("event evidence weakest phase derivation differs")
    weakest_margin = _finite(row["weakest_margin"], label="event weakest margin")
    if weakest_margin != margins[weakest_phase]:
        _fail("event evidence weakest margin derivation differs")
    relative_pass = _bool(
        row["relative_action_margin_pass"], label="relative_action_margin_pass"
    )
    relative_eligible = _bool(
        row["relative_pairing_eligible"], label="relative_pairing_eligible"
    )
    if relative_eligible is not (row["mode"] == "bootstrap" and relative_pass):
        _fail("event relative eligibility derivation differs")
    action_pass = _validate_phase_bool_map(
        row["absolute_action_pass_by_phase"], label="absolute action pass"
    )
    margin_pass = _validate_phase_bool_map(
        row["absolute_margin_pass_by_phase"], label="absolute margin pass"
    )
    combined = _validate_phase_bool_map(
        row["absolute_four_stage_pass_by_phase"], label="absolute combined pass"
    )
    expected_combined = {
        phase: action_pass[phase] and margin_pass[phase]
        for phase in phases
    }
    if combined != expected_combined:
        _fail("event absolute action/margin conjunction differs")
    absolute = _bool(row["absolute_four_stage_pass"], label="absolute event pass")
    if absolute is not all(combined.values()):
        _fail("event absolute four-stage conjunction differs")
    normalized = dict(row)
    normalized["stage_margins"] = margins
    normalized["absolute_action_pass_by_phase"] = action_pass
    normalized["absolute_margin_pass_by_phase"] = margin_pass
    normalized["absolute_four_stage_pass_by_phase"] = combined
    return json.loads(canonical_json_bytes(normalized).decode("ascii"))


def _derive_axes(component_scores: Mapping[str, float]) -> dict[str, float]:
    return {
        "identity": component_scores["identity_preservation"],
        "camera": component_scores["camera_preservation"],
        "background": component_scores["background_preservation"],
        "non_target": component_scores["non_target_motion_preservation"],
        "quality": min(
            component_scores[component]
            for component in _expected_quality_components()
        ),
        "source_bind": min(
            component_scores["wrong_source_reconstruction_error"],
            component_scores["dropped_source_reconstruction_error"],
        )
        - component_scores["correct_source_reconstruction_error"],
        "inverse": 1.0
        / (1.0 + component_scores["inverse_reconstruction_error"]),
    }


def _validate_packet_mapping(value: Any) -> dict[str, Any]:
    row = _closed(value, _PACKET_FIELDS, label="candidate score packet")
    if row["schema_version"] != SCHEMA_VERSION:
        _fail("candidate score packet schema_version differs")
    binding = _validate_binding(row["candidate_binding"])
    event = _validate_event_evidence(row["event_evidence"])
    if event["action_family"] != binding["instruction_id"]:
        _fail("event evidence action family differs from candidate instruction")
    bundle = _validate_measurement_bundle(row["measurement_bundle"], binding)
    components: dict[str, float] = {}
    component_map = row["component_scores"]
    if not isinstance(component_map, Mapping) or set(component_map) != set(
        _expected_measurement_components()
    ):
        _fail("component_scores keys differ")
    for component in _expected_unit_components():
        components[component] = _unit(
            component_map[component], label=f"component_scores.{component}"
        )
    for component in _expected_error_components():
        components[component] = _nonnegative(
            component_map[component], label=f"component_scores.{component}"
        )
    if components != bundle["score_value_by_component"]:
        _fail("component scores differ from measurement bundle")
    derived = _derive_axes(components)
    axis_map = row["axis_scores"]
    axes_contract = _expected_preservation_axes()
    if not isinstance(axis_map, Mapping) or set(axis_map) != set(axes_contract):
        _fail("axis_scores must contain exactly seven axes")
    axes = {
        axis: _finite(axis_map[axis], label=f"axis_scores.{axis}")
        for axis in axes_contract
    }
    if axes != derived:
        _fail("seven-axis derivation differs")
    event_binding = _validate_event_binding(
        row["event_media_binding_receipt"], binding, event
    )
    expected_authority = _expected_authority_contract()
    if row["authority_contract"] != expected_authority:
        _fail("candidate authority contract differs")
    _sealed(row, "packet_digest", label="candidate score packet")
    normalized = dict(row)
    normalized["candidate_binding"] = binding
    normalized["event_evidence"] = event
    normalized["event_media_binding_receipt"] = event_binding
    normalized["component_scores"] = components
    normalized["axis_scores"] = axes
    normalized["measurement_bundle"] = bundle
    normalized["authority_contract"] = expected_authority
    if canonical_json_bytes(normalized) != canonical_json_bytes(row):
        _fail("candidate score packet is not canonically normalized")
    return normalized


def validate_candidate_score_packet(value: Any) -> bytes:
    """Return immutable canonical diagnostic bytes with no provenance claim."""

    if type(value) is bytes:
        try:
            source: Any = json.loads(value.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SAICSevenAxisScorePacketError(
                "candidate packet bytes are not canonical JSON"
            ) from error
        if canonical_json_bytes(source) != value:
            _fail("candidate packet bytes are not exact canonical bytes")
    elif isinstance(value, Mapping):
        source = value
    else:
        _fail("candidate packet must be a mapping or canonical bytes")
    return canonical_json_bytes(_validate_packet_mapping(source))


def _detached_measurements(value: Any) -> dict[str, float]:
    if torch is None:
        _fail("torch is required to validate detached FP32 measurements")
    components = _expected_measurement_components()
    unit_components = _expected_unit_components()
    if not isinstance(value, Mapping) or set(value) != set(components):
        _fail("detached measurements must contain exactly the registered components")
    result: dict[str, float] = {}
    for component in components:
        tensor = value[component]
        if (
            type(tensor) is not torch.Tensor
            or tensor.dtype != torch.float32
            or tensor.ndim != 0
            or tensor.numel() != 1
            or tensor.requires_grad
            or tensor.grad_fn is not None
        ):
            _fail(f"{component} must be an actual detached scalar FP32 tensor")
        scalar = float(tensor.item())
        if component in unit_components:
            result[component] = _unit(scalar, label=component)
        else:
            result[component] = _nonnegative(scalar, label=component)
    return result


def build_candidate_score_packet(
    candidate: Mapping[str, Any],
    *,
    event_boundary: Any,
    event_candidate_scores: Mapping[str, Any],
    event_media_binding_receipt: Mapping[str, Any],
    event_mode: str,
    detached_measurements: Mapping[str, Any],
    measurement_bundle: Mapping[str, Any],
) -> bytes:
    """Evaluate once and return re-signable, diagnostic-only canonical bytes."""

    try:
        validated_candidate = rollout_contract.validate_candidate(candidate)
    except Exception as error:
        raise SAICSevenAxisScorePacketError("legacy candidate validation failed") from error
    if type(event_boundary) is not event_reward.SAICEventRewardBoundary:
        _fail("event_boundary must be an exact qualified SAICEventRewardBoundary")
    if event_mode not in _expected_modes():
        _fail("event_mode must be bootstrap or strict")
    try:
        decision = event_boundary.evaluate(event_candidate_scores, mode=event_mode)
    except Exception as error:
        raise SAICSevenAxisScorePacketError("qualified event evaluation failed") from error
    binding = _candidate_binding(validated_candidate)
    if decision["candidate_id"] != binding["candidate_id"]:
        _fail("event decision candidate differs")
    if decision["action_family"] != binding["instruction_id"]:
        _fail("event decision action family differs")
    if event_candidate_scores.get("policy_checkpoint_sha256") != binding["policy_sha256"]:
        _fail("event scores policy differs from candidate policy")
    event = _validate_event_evidence(_event_evidence(decision))
    event_binding = _validate_event_binding(
        event_media_binding_receipt, binding, event
    )
    components = _detached_measurements(detached_measurements)
    bundle = _validate_measurement_bundle(measurement_bundle, binding)
    if components != bundle["score_value_by_component"]:
        _fail("detached measurements differ from measurement bundle score values")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "candidate_binding": binding,
        "event_evidence": event,
        "event_media_binding_receipt": event_binding,
        "component_scores": components,
        "axis_scores": _derive_axes(components),
        "measurement_bundle": bundle,
        "authority_contract": _expected_authority_contract(),
    }
    packet = {**unsigned, "packet_digest": object_sha256(unsigned)}
    return canonical_json_bytes(_validate_packet_mapping(packet))


def _threshold_map(
    value: Any,
    names: Sequence[str],
    *,
    label: str,
    nonnegative: bool,
) -> dict[str, float]:
    parser = _nonnegative if nonnegative else _finite
    return _exact_named_map(value, names, label=label, parser=parser)


def _globally_unique(rows: Sequence[Mapping[str, Any]], key_path: tuple[str, ...]) -> None:
    values = []
    for row in rows:
        value: Any = row
        for key in key_path:
            value = value[key]
        values.append(value)
    if len(set(values)) != len(values):
        _fail(f"all candidate/base {'.'.join(key_path)} values must be globally unique")


def _cell(row: Mapping[str, Any]) -> tuple[str, str, str]:
    binding = row["candidate_binding"]
    return binding["arm"], binding["source_id"], binding["instruction_id"]


def _pair_is_admissible(
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
    base: Mapping[str, Any],
    *,
    mode: str,
    minimum_event_delta: float,
    axis_absolute_floors: Mapping[str, float],
    axis_baseline_slacks: Mapping[str, float],
    axis_pair_tolerances: Mapping[str, float],
    component_absolute_floors: Mapping[str, float],
    component_baseline_slacks: Mapping[str, float],
    component_pair_tolerances: Mapping[str, float],
) -> dict[str, Any] | None:
    axes_contract = _expected_preservation_axes()
    quality_components = _expected_quality_components()
    cb = chosen["candidate_binding"]
    rb = rejected["candidate_binding"]
    bb = base["candidate_binding"]
    if _cell(chosen) != _cell(rejected) or _cell(chosen) != _cell(base):
        return None
    if cb["policy_sha256"] != rb["policy_sha256"]:
        return None
    if cb["source_media_sha256"] != rb["source_media_sha256"] or cb[
        "source_media_sha256"
    ] != bb["source_media_sha256"]:
        return None
    ce, re = chosen["event_evidence"], rejected["event_evidence"]
    if (
        ce["critic_checkpoint_sha256"] != re["critic_checkpoint_sha256"]
        or ce["critic_qualification_receipt_digest"]
        != re["critic_qualification_receipt_digest"]
        or ce["action_family"] != re["action_family"]
    ):
        return None
    if not (
        chosen["measurement_bundle"]["evaluator_set_sha256"]
        == rejected["measurement_bundle"]["evaluator_set_sha256"]
        == base["measurement_bundle"]["evaluator_set_sha256"]
    ):
        return None
    event_delta = ce["weakest_margin"] - re["weakest_margin"]
    if event_delta < minimum_event_delta:
        return None
    if mode == "bootstrap" and not ce["relative_pairing_eligible"]:
        return None
    if mode == "strict" and not ce["absolute_four_stage_pass"]:
        return None
    effective_axis_floors = {
        axis: max(
            axis_absolute_floors[axis],
            base["axis_scores"][axis] - axis_baseline_slacks[axis],
        )
        for axis in axes_contract
    }
    if any(
        chosen["axis_scores"][axis] < effective_axis_floors[axis]
        or rejected["axis_scores"][axis] < effective_axis_floors[axis]
        or chosen["axis_scores"][axis] + axis_pair_tolerances[axis]
        < rejected["axis_scores"][axis]
        for axis in axes_contract
    ):
        return None
    effective_component_floors = {
        component: max(
            component_absolute_floors[component],
            base["component_scores"][component]
            - component_baseline_slacks[component],
        )
        for component in quality_components
    }
    if any(
        chosen["component_scores"][component]
        < effective_component_floors[component]
        or rejected["component_scores"][component]
        < effective_component_floors[component]
        or chosen["component_scores"][component]
        + component_pair_tolerances[component]
        < rejected["component_scores"][component]
        for component in quality_components
    ):
        return None
    body = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "arm": cb["arm"],
        "source_id": cb["source_id"],
        "instruction_id": cb["instruction_id"],
        "current_policy_sha256": cb["policy_sha256"],
        "frozen_base_policy_sha256": bb["policy_sha256"],
        "chosen_candidate_id": cb["candidate_id"],
        "chosen_packet_digest": chosen["packet_digest"],
        "rejected_candidate_id": rb["candidate_id"],
        "rejected_packet_digest": rejected["packet_digest"],
        "frozen_base_packet_digest": base["packet_digest"],
        "chosen_event_absolute_score": ce["weakest_margin"],
        "rejected_event_absolute_score": re["weakest_margin"],
        "event_pair_delta": event_delta,
        "minimum_event_delta": minimum_event_delta,
        "chosen_relative_pairing_eligible": ce["relative_pairing_eligible"],
        "chosen_absolute_four_stage_event_pass": ce["absolute_four_stage_pass"],
        "axis_effective_floors": effective_axis_floors,
        "axis_chosen_minus_rejected": {
            axis: chosen["axis_scores"][axis] - rejected["axis_scores"][axis]
            for axis in axes_contract
        },
        "component_effective_floors": effective_component_floors,
        "component_chosen_minus_rejected": {
            component: chosen["component_scores"][component]
            - rejected["component_scores"][component]
            for component in quality_components
        },
        "seven_axis_hard_conjunction_pass": True,
        "explicit_quality_component_conjunction_pass": True,
        "optimizer_authority": False,
    }
    return {**body, "pair_digest": object_sha256(body)}


_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "current_policy_sha256",
        "frozen_base_policy_sha256",
        "required_arms",
        "candidate_packet_canonical_json",
        "frozen_base_packet_canonical_json",
        "thresholds",
        "global_evaluator_set_sha256",
        "global_event_critic_checkpoint_sha256",
        "global_event_qualification_receipt_digest",
        "diagnostic_admissible_pair_count_by_arm",
        "diagnostic_pairs",
        "diagnostic_pairing_eligible",
        "serialized_input_provenance",
        "builder_authentication_claimed",
        "whole_frame_measurement_runtime_qualified",
        "optimizer_authorized_pair_digests",
        "optimizer_step_allowed",
        "zero_update_reason",
        "scalar_reward_or_weighted_compensation_used",
        "pure_t2v_endpoint_condition_target_noise_or_donor_used",
        "receipt_digest",
    }
)
def _normalized_thresholds(
    *,
    minimum_event_delta: Any,
    axis_absolute_floors: Any,
    axis_baseline_slacks: Any,
    axis_pair_tolerances: Any,
    component_absolute_floors: Any,
    component_baseline_slacks: Any,
    component_pair_tolerances: Any,
) -> dict[str, Any]:
    axes_contract = _expected_preservation_axes()
    quality_components = _expected_quality_components()
    return {
        "minimum_event_delta": _nonnegative(
            minimum_event_delta, label="minimum_event_delta"
        ),
        "axis_absolute_floors": _threshold_map(
            axis_absolute_floors,
            axes_contract,
            label="axis_absolute_floors",
            nonnegative=False,
        ),
        "axis_baseline_slacks": _threshold_map(
            axis_baseline_slacks,
            axes_contract,
            label="axis_baseline_slacks",
            nonnegative=True,
        ),
        "axis_pair_tolerances": _threshold_map(
            axis_pair_tolerances,
            axes_contract,
            label="axis_pair_tolerances",
            nonnegative=True,
        ),
        "component_absolute_floors": _threshold_map(
            component_absolute_floors,
            quality_components,
            label="component_absolute_floors",
            nonnegative=False,
        ),
        "component_baseline_slacks": _threshold_map(
            component_baseline_slacks,
            quality_components,
            label="component_baseline_slacks",
            nonnegative=True,
        ),
        "component_pair_tolerances": _threshold_map(
            component_pair_tolerances,
            quality_components,
            label="component_pair_tolerances",
            nonnegative=True,
        ),
    }


def _validate_threshold_bundle(value: Any) -> dict[str, Any]:
    expected = _expected_thresholds()
    row = _closed(value, frozenset(expected), label="diagnostic thresholds")
    normalized = _normalized_thresholds(**row)
    if normalized != expected:
        _fail("diagnostic thresholds differ from preregistered policy")
    return expected


def _packet_from_canonical_bytes(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not bytes:
        _fail(f"{label} must contain canonical diagnostic packet bytes")
    return json.loads(validate_candidate_score_packet(value).decode("ascii"))


def _embedded_packets(value: Any, *, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        _fail(f"{label} must be a list of canonical ASCII JSON strings")
    packets: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        try:
            packet_bytes = item.encode("ascii")
        except UnicodeEncodeError as error:
            raise SAICSevenAxisScorePacketError(
                f"{label}[{index}] must be ASCII"
            ) from error
        packets.append(
            _packet_from_canonical_bytes(
                packet_bytes, label=f"{label}[{index}]"
            )
        )
    return tuple(packets)


def _one_global_digest(
    rows: Sequence[Mapping[str, Any]],
    path: tuple[str, ...],
    *,
    label: str,
) -> str:
    values: set[str] = set()
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.add(_sha(value, label=label))
    if len(values) != 1:
        _fail(f"all dog/human candidate/base rows must bind one global {label}")
    return next(iter(values))


def _compute_diagnostic_selection(
    candidates: Sequence[Mapping[str, Any]],
    bases: Sequence[Mapping[str, Any]],
    *,
    current_policy: str,
    base_policy: str,
    mode: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if current_policy == base_policy:
        _fail("current policy must differ from frozen base policy")
    if any(
        row["candidate_binding"]["policy_sha256"] != current_policy
        for row in candidates
    ):
        _fail("all candidates must bind one explicit current policy")
    if any(
        row["candidate_binding"]["policy_sha256"] != base_policy
        for row in bases
    ):
        _fail("all frozen bases must bind one explicit frozen base policy")
    all_rows = (*candidates, *bases)
    if not all_rows:
        _fail("the diagnostic pilot requires at least one candidate/base row")
    if any(row["event_evidence"]["mode"] != mode for row in all_rows):
        _fail("candidate/base event modes must equal the requested mode")
    for path in (
        ("candidate_binding", "candidate_id"),
        ("packet_digest",),
        ("candidate_binding", "legacy_candidate_digest"),
        ("candidate_binding", "output_media_sha256"),
        ("candidate_binding", "endpoint_latent_sha256"),
        ("candidate_binding", "rollout_receipt_digest"),
        ("event_evidence", "event_rollout_id"),
    ):
        _globally_unique(all_rows, path)
    evaluator_sha = _one_global_digest(
        all_rows,
        ("measurement_bundle", "evaluator_set_sha256"),
        label="whole-frame evaluator set",
    )
    critic_sha = _one_global_digest(
        all_rows,
        ("event_evidence", "critic_checkpoint_sha256"),
        label="event critic checkpoint",
    )
    qualification_sha = _one_global_digest(
        all_rows,
        ("event_evidence", "critic_qualification_receipt_digest"),
        label="event critic qualification receipt",
    )

    candidate_cells = {_cell(row) for row in candidates}
    base_by_cell: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for base in bases:
        cell = _cell(base)
        if cell in base_by_cell:
            _fail("frozen base coverage must be exactly one packet per cell")
        base_by_cell[cell] = base
    if set(base_by_cell) != candidate_cells:
        _fail("frozen base coverage must exactly match candidate cells")

    arms_contract = _expected_arms()
    admissible: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in arms_contract
    }
    for chosen in candidates:
        base = base_by_cell[_cell(chosen)]
        for rejected in candidates:
            if rejected is chosen:
                continue
            pair = _pair_is_admissible(
                chosen,
                rejected,
                base,
                mode=mode,
                minimum_event_delta=thresholds["minimum_event_delta"],
                axis_absolute_floors=thresholds["axis_absolute_floors"],
                axis_baseline_slacks=thresholds["axis_baseline_slacks"],
                axis_pair_tolerances=thresholds["axis_pair_tolerances"],
                component_absolute_floors=thresholds[
                    "component_absolute_floors"
                ],
                component_baseline_slacks=thresholds[
                    "component_baseline_slacks"
                ],
                component_pair_tolerances=thresholds[
                    "component_pair_tolerances"
                ],
            )
            if pair is not None:
                admissible[pair["arm"]].append(pair)
    for arm in arms_contract:
        admissible[arm].sort(
            key=lambda pair: (-pair["event_pair_delta"], pair["pair_digest"])
        )
    counts = {arm: len(admissible[arm]) for arm in arms_contract}
    complete = all(counts[arm] > 0 for arm in arms_contract)
    return {
        "counts": counts,
        "pairs": [admissible[arm][0] for arm in arms_contract] if complete else [],
        "eligible": complete,
        "global_evaluator_set_sha256": evaluator_sha,
        "global_event_critic_checkpoint_sha256": critic_sha,
        "global_event_qualification_receipt_digest": qualification_sha,
    }


def _zero_update_reason(complete: bool) -> str:
    if complete:
        return "whole-frame measurement runtime and evaluator trust root are unqualified"
    return (
        "dog/human hard diagnostic conjunction is incomplete; measurement "
        "runtime is also unqualified"
    )


def validate_source_hard_preference_set(value: Any) -> bytes:
    """Recompute and validate a canonical, explicitly untrusted diagnostic."""

    if type(value) is not bytes:
        _fail("preference-set receipt must be canonical bytes")
    try:
        decoded = json.loads(value.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SAICSevenAxisScorePacketError("receipt is not canonical JSON") from error
    if canonical_json_bytes(decoded) != value:
        _fail("preference-set receipt bytes are not exact canonical bytes")
    row = _closed(decoded, _RESULT_FIELDS, label="preference-set receipt")
    if (
        row["schema_version"] != PREFERENCE_SET_SCHEMA_VERSION
        or row["mode"] not in _expected_modes()
    ):
        _fail("preference-set schema or mode differs")
    current = _sha(row["current_policy_sha256"], label="current policy")
    base = _sha(row["frozen_base_policy_sha256"], label="frozen base policy")
    if current == base:
        _fail("current policy and frozen base policy must differ")
    if row["required_arms"] != list(_expected_arms()):
        _fail("required arms differ")
    candidates = _embedded_packets(
        row["candidate_packet_canonical_json"],
        label="embedded candidate packets",
    )
    bases = _embedded_packets(
        row["frozen_base_packet_canonical_json"],
        label="embedded frozen-base packets",
    )
    thresholds = _validate_threshold_bundle(row["thresholds"])
    recomputed = _compute_diagnostic_selection(
        candidates,
        bases,
        current_policy=current,
        base_policy=base,
        mode=row["mode"],
        thresholds=thresholds,
    )
    if canonical_json_bytes(row["thresholds"]) != canonical_json_bytes(thresholds):
        _fail("root threshold maps are not canonically normalized")
    for key in (
        "global_evaluator_set_sha256",
        "global_event_critic_checkpoint_sha256",
        "global_event_qualification_receipt_digest",
    ):
        _sha(row[key], label=key)
        if row[key] != recomputed[key]:
            _fail(f"root {key} differs from embedded packets")
    counts = row["diagnostic_admissible_pair_count_by_arm"]
    if canonical_json_bytes(counts) != canonical_json_bytes(recomputed["counts"]):
        _fail("diagnostic pair counts differ from embedded-packet recomputation")
    if not isinstance(row["diagnostic_pairs"], list):
        _fail("diagnostic_pairs must be a list")
    if canonical_json_bytes(row["diagnostic_pairs"]) != canonical_json_bytes(
        recomputed["pairs"]
    ):
        _fail("diagnostic pairs differ from embedded-packet recomputation")
    eligible = _bool(
        row["diagnostic_pairing_eligible"], label="diagnostic eligibility"
    )
    if eligible is not recomputed["eligible"]:
        _fail("diagnostic eligibility differs from embedded-packet recomputation")
    if row["serialized_input_provenance"] != (
        "caller_supplied_digest_sealed_re_signable_diagnostic_only"
    ):
        _fail("serialized input provenance must remain explicitly untrusted")
    _bool(
        row["builder_authentication_claimed"],
        label="builder authentication claimed",
        expected=False,
    )
    _bool(row["whole_frame_measurement_runtime_qualified"], label="measurement runtime qualified", expected=False)
    if row["optimizer_authorized_pair_digests"] != []:
        _fail("diagnostic receipt cannot contain optimizer-authorized pairs")
    _bool(row["optimizer_step_allowed"], label="optimizer step", expected=False)
    if row["zero_update_reason"] != _zero_update_reason(recomputed["eligible"]):
        _fail("zero_update_reason differs from recomputed diagnostic state")
    _bool(row["scalar_reward_or_weighted_compensation_used"], label="scalar compensation", expected=False)
    _bool(row["pure_t2v_endpoint_condition_target_noise_or_donor_used"], label="pure-T2V endpoint route", expected=False)
    _sealed(row, "receipt_digest", label="preference-set receipt")
    return value


def build_source_hard_preference_set(
    candidate_packets: Sequence[bytes],
    *,
    frozen_base_packets: Sequence[bytes],
    current_policy_sha256: str,
    frozen_base_policy_sha256: str,
    mode: str,
) -> bytes:
    """Build the preregistered two-arm diagnostic with no provenance authority."""

    if isinstance(candidate_packets, (str, bytes)) or not isinstance(candidate_packets, Sequence):
        _fail("candidate_packets must be a sequence")
    if isinstance(frozen_base_packets, (str, bytes)) or not isinstance(frozen_base_packets, Sequence):
        _fail("frozen_base_packets must be a sequence")
    current_policy = _sha(current_policy_sha256, label="current_policy_sha256")
    base_policy = _sha(frozen_base_policy_sha256, label="frozen_base_policy_sha256")
    if current_policy == base_policy:
        _fail("current policy must differ from frozen base policy")
    if mode not in _expected_modes():
        _fail("mode must be bootstrap or strict")
    thresholds = _expected_thresholds()
    candidate_bytes = tuple(candidate_packets)
    base_bytes = tuple(frozen_base_packets)
    candidates = tuple(
        _packet_from_canonical_bytes(value, label=f"candidate_packets[{index}]")
        for index, value in enumerate(candidate_bytes)
    )
    bases = tuple(
        _packet_from_canonical_bytes(value, label=f"frozen_base_packets[{index}]")
        for index, value in enumerate(base_bytes)
    )
    computed = _compute_diagnostic_selection(
        candidates,
        bases,
        current_policy=current_policy,
        base_policy=base_policy,
        mode=mode,
        thresholds=thresholds,
    )
    unsigned = {
        "schema_version": PREFERENCE_SET_SCHEMA_VERSION,
        "mode": mode,
        "current_policy_sha256": current_policy,
        "frozen_base_policy_sha256": base_policy,
        "required_arms": list(_expected_arms()),
        "candidate_packet_canonical_json": [
            value.decode("ascii") for value in candidate_bytes
        ],
        "frozen_base_packet_canonical_json": [
            value.decode("ascii") for value in base_bytes
        ],
        "thresholds": thresholds,
        "global_evaluator_set_sha256": computed[
            "global_evaluator_set_sha256"
        ],
        "global_event_critic_checkpoint_sha256": computed[
            "global_event_critic_checkpoint_sha256"
        ],
        "global_event_qualification_receipt_digest": computed[
            "global_event_qualification_receipt_digest"
        ],
        "diagnostic_admissible_pair_count_by_arm": computed["counts"],
        "diagnostic_pairs": computed["pairs"],
        "diagnostic_pairing_eligible": computed["eligible"],
        "serialized_input_provenance": (
            "caller_supplied_digest_sealed_re_signable_diagnostic_only"
        ),
        "builder_authentication_claimed": False,
        "whole_frame_measurement_runtime_qualified": False,
        "optimizer_authorized_pair_digests": [],
        "optimizer_step_allowed": False,
        "zero_update_reason": _zero_update_reason(computed["eligible"]),
        "scalar_reward_or_weighted_compensation_used": False,
        "pure_t2v_endpoint_condition_target_noise_or_donor_used": False,
    }
    receipt = canonical_json_bytes(
        {**unsigned, "receipt_digest": object_sha256(unsigned)}
    )
    return validate_source_hard_preference_set(receipt)


__all__ = [
    "ARMS",
    "BOOTSTRAP_MODE",
    "ERROR_COMPONENTS",
    "EVENT_MEDIA_BINDING_SCHEMA_VERSION",
    "EXPLICIT_QUALITY_COMPONENTS",
    "FRAME_COUNT",
    "MEASUREMENT_BUNDLE_SCHEMA_VERSION",
    "MEASUREMENT_COMPONENTS",
    "PAIR_SCHEMA_VERSION",
    "PREFERENCE_SET_SCHEMA_VERSION",
    "PRESERVATION_AXES",
    "PURE_T2V_ROLE",
    "SAICSevenAxisScorePacketError",
    "SCHEMA_VERSION",
    "STRICT_MODE",
    "UNIT_COMPONENTS",
    "build_candidate_score_packet",
    "build_source_hard_preference_set",
    "canonical_json_bytes",
    "object_sha256",
    "validate_candidate_score_packet",
    "validate_source_hard_preference_set",
]
