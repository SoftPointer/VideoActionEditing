#!/usr/bin/env python3
"""Fail-closed CAPER Stage-1 same-source preference admission.

This module is the trust boundary between frozen rollout/evaluation jobs and a
later preference trainer.  It deliberately does not import torch, construct a
loss, or call an optimizer.  Its only mutable operation is writing a sealed
admission materialization when invoked through the CLI.

The input is a *candidate* manifest.  Contract violations (for example a T2V
video used as ``y+``, different source IDs, a policy/checkpoint mismatch,
split leakage, target-video access, or scalar compensation) raise
``CAPERAdmissionError`` and produce no materialization.  Scientifically valid
rejections (a failed preservation gate or a non-strict action ordering) are
recorded as rejected pairs.  If no fit pair survives, the materialization
contains an explicit zero-update certificate and
``optimizer_step_allowed=false``.

Pure-T2V rollouts are represented only in the separate calibration-owner
registry.  Their IDs and media/receipt/audit commitments are forbidden from
aliasing either endpoint of a preference pair.  Held-out pairs are admitted
for evaluation evidence only and can never become optimizer targets.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any


SCHEMA_VERSION = "bernini-caper-stage1-preference-candidates-v1"
MATERIALIZATION_SCHEMA_VERSION = (
    "bernini-caper-stage1-preference-admission-v1"
)
ZERO_UPDATE_SCHEMA_VERSION = "bernini-caper-stage1-zero-update-certificate-v1"
ROLLOUT_RECEIPT_SCHEMA_VERSION = "bernini-caper-rollout-receipt-v1"
REWARD_AUDIT_SCHEMA_VERSION = "bernini-caper-reward-audit-v1"
EXPOSURE_LEDGER_SCHEMA_VERSION = "bernini-caper-exposure-ledger-v1"
POPULATION_LEDGER_SCHEMA_VERSION = "bernini-caper-rollout-population-ledger-v1"
FAILED_ROLLOUT_RECEIPT_SCHEMA_VERSION = "bernini-caper-failed-rollout-receipt-v1"
DECODED_MEDIA_CONTRACT = "pyav-rgb24-exact81-fps25-v1"
MIN_POPULATION_SIZE_K = 2
MAX_POPULATION_SIZE_K = 64
PAIR_SELECTION_RULE = (
    "all_preregistered_attempts_recorded_choose_max_action_lcb_vs_min_action_ucb_tie_seed_order"
)
SEED_BALANCE_RULE = "each_preregistered_seed_exactly_once_per_source_population"
REWARD_EVALUATOR_TYPE = "decoded_rgb24_action_preservation_evaluator"
REWARD_EVALUATOR_INPUTS = (
    "decoded_candidate_rgb24",
    "action_instruction",
    "action_rubric",
)
REWARD_EVALUATOR_OPTIONAL_INPUTS = ("decoded_source_rgb24",)
REWARD_EVALUATOR_FORBIDDEN_INPUTS = (
    "mask",
    "track",
    "pose",
    "flow",
    "pure_t2v_origin_metadata",
)
MILESTONES = ("start", "transition", "terminal", "hold")

SPLITS = ("fit", "heldout")
PRESERVATION_GATES = (
    "identity",
    "object_correspondence",
    "background",
    "camera",
    "non_target",
    "quality",
)
SPLIT_ISOLATION_AXES = (
    "source_id",
    "source_media_sha256",
    "identity_id",
    "scene_id",
    "action_family",
)
ACTION_GRADES = (
    "absent_or_wrong",
    "started_only",
    "transition_incomplete",
    "terminal_reached",
    "terminal_reached_and_held",
)
ACTION_GRADE_ORDER = {grade: index for index, grade in enumerate(ACTION_GRADES)}

PREFERENCE_TARGET_MODE = (
    "same_source_on_policy_source_conditioned_video_editing_only"
)
PURE_T2V_ROLE = "action_reward_calibration_owner_only"
PRESERVATION_SEMANTICS = "conjunctive_hard_gates_no_compensation"
NO_PAIR_BEHAVIOR = "emit_zero_update_certificate"
ACTION_SELECTION_BASIS = (
    "action_event_order_only_after_both_endpoints_pass_all_preservation_gates"
)
SOURCE_CONDITIONING_INPUTS = (
    "source_video",
    "edit_instruction",
    "sealed_gaussian",
)
PURE_T2V_INPUTS = ("text_prompt", "sealed_gaussian")
SOURCE_CANDIDATE_ROLE = "source_conditioned_candidate"
PURE_T2V_LEDGER_ROLE = "pure_t2v_calibration_owner"
SOURCE_AUDIT_ROLE = "source_candidate_admission"
PURE_T2V_AUDIT_ROLE = "pure_t2v_action_calibration"
FAILED_ROLLOUT_ROLE = "source_conditioned_failed_attempt"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_UTC_RE = re.compile(
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "created_utc",
        "purpose",
        "admission_contract",
        "deployable_policy",
        "reward_evaluator",
        "rollout_population_ledger",
        "exposure_ledger",
        "pure_t2v_calibration_owners",
        "splits",
    }
)
_CONTRACT_FIELDS = frozenset(
    {
        "stage",
        "preference_target_mode",
        "pure_t2v_role",
        "pure_t2v_preference_target_allowed",
        "target_video_dependency_allowed",
        "scalar_reward_admission_allowed",
        "scalar_compensation_allowed",
        "preservation_gate_semantics",
        "preservation_gate_names",
        "split_isolation_axes",
        "no_valid_fit_pair_behavior",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "policy_id",
        "model_family",
        "checkpoint_tree_sha256",
        "inference_contract_sha256",
        "deployable",
        "source_conditioned",
        "weights_frozen_during_rollout",
    }
)
_POLICY_BINDING_FIELDS = frozenset(
    {
        "policy_id",
        "checkpoint_tree_sha256",
        "inference_contract_sha256",
    }
)
_ARTIFACT_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_REWARD_EVALUATOR_FIELDS = frozenset(
    {
        "evaluator_id",
        "evaluator_type",
        "evaluator_version",
        "weights_artifact",
        "runtime_artifact",
        "input_contract_sha256",
        "required_inputs",
        "optional_inputs",
        "forbidden_inputs",
    }
)
_EVALUATOR_BINDING_FIELDS = frozenset(
    {
        "evaluator_id",
        "evaluator_type",
        "evaluator_version",
        "weights_artifact_sha256",
        "runtime_artifact_sha256",
        "input_contract_sha256",
    }
)
_SPLITS_FIELDS = frozenset(SPLITS)
_PAIR_FIELDS = frozenset(
    {
        "pair_id",
        "population_id",
        "split",
        "source_id",
        "identity_id",
        "scene_id",
        "action_family",
        "edit_instruction",
        "source_media",
        "source_intake_receipt",
        "target_video_dependency",
        "scalar_reward_used_for_admission",
        "scalar_compensation_used",
        "winner",
        "loser",
        "action_ordering",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "population_id",
        "seed_id",
        "declared_role",
        "generation_mode",
        "eligible_as_preference_target",
        "source_id",
        "identity_id",
        "scene_id",
        "action_family",
        "edit_instruction",
        "policy_binding",
        "conditioning_attestation",
        "output_media",
        "rollout_receipt",
        "reward_audit",
        "preservation_hard_gates",
    }
)
_CONDITIONING_FIELDS = frozenset(
    {
        "input_kinds",
        "source_media_sha256",
        "target_video_read",
        "target_video_latent_read",
        "paired_target_read",
        "pure_t2v_media_read",
        "pure_t2v_latent_read",
    }
)
_ACTION_ORDERING_FIELDS = frozenset(
    {
        "rubric_id",
        "winner_action_grade",
        "loser_action_grade",
        "strict_preference_claimed",
        "selection_basis",
        "pair_selection_rule",
        "winner_action_score",
        "winner_action_uncertainty",
        "loser_action_score",
        "loser_action_uncertainty",
        "pairwise_confidence_margin",
        "minimum_required_pairwise_margin",
    }
)
_OWNER_FIELDS = frozenset(
    {
        "owner_id",
        "split",
        "role",
        "generation_mode",
        "action_family",
        "prompt",
        "generator_binding",
        "output_media",
        "rollout_receipt",
        "reward_audit",
        "eligible_as_preference_target",
        "eligible_as_training_target",
        "target_video_dependency",
    }
)
_GENERATOR_BINDING_FIELDS = frozenset(
    {
        "generator_id",
        "checkpoint_tree_sha256",
        "inference_contract_sha256",
    }
)
_ROLLOUT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "rollout_id",
        "lineage_root_sha256",
        "rollout_role",
        "split",
        "policy_id",
        "checkpoint_tree_sha256",
        "inference_contract_sha256",
        "source_id",
        "source_media_sha256",
        "action_family",
        "instruction_text_sha256",
        "accepted_inputs",
        "sealed_gaussian_sha256",
        "population_id",
        "seed_id",
        "attempt_status",
        "failure_code",
        "target_video_read",
        "target_video_latent_read",
        "paired_target_read",
        "pure_t2v_media_read",
        "pure_t2v_latent_read",
        "output_media_sha256",
        "output_media_size_bytes",
        "frame_count",
        "fps",
        "decoded_media_contract",
        "decoded_frame_count",
        "decoded_fps_numerator",
        "decoded_fps_denominator",
        "decoded_height",
        "decoded_width",
        "decoded_rgb24_sha256",
        "source_role_digest",
        "receipt_sha256",
    }
)
_REWARD_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "audit_id",
        "rollout_id",
        "lineage_root_sha256",
        "source_role_digest",
        "output_media_sha256",
        "rollout_receipt_artifact_sha256",
        "audit_role",
        "checkpoint_tree_sha256",
        "action_family",
        "rubric_id",
        "audit_protocol_sha256",
        "evaluator_binding",
        "evaluator_inputs",
        "mask_read",
        "track_read",
        "pose_read",
        "flow_read",
        "pure_t2v_origin_metadata_read",
        "action_grade",
        "action_score",
        "action_uncertainty",
        "action_margin_to_grade_threshold",
        "milestone_frame_evidence",
        "preservation_hard_gates",
        "scalar_compensation_used",
        "target_video_dependency",
        "audit_sha256",
    }
)
_EXPOSURE_LEDGER_FIELDS = frozenset(
    {
        "schema_version",
        "ledger_id",
        "checkpoint_tree_sha256",
        "inference_contract_sha256",
        "source_revision_sha256",
        "action_taxonomy_sha256",
        "reward_audit_protocol_sha256",
        "entries",
        "ledger_sha256",
    }
)
_EXPOSURE_ENTRY_FIELDS = frozenset(
    {
        "rollout_id",
        "lineage_root_sha256",
        "rollout_role",
        "split",
        "population_id",
        "seed_id",
        "policy_id",
        "checkpoint_tree_sha256",
        "inference_contract_sha256",
        "source_id",
        "source_media_sha256",
        "action_family",
        "instruction_text_sha256",
        "output_media_sha256",
        "rollout_receipt_artifact_sha256",
        "reward_audit_artifact_sha256",
        "source_role_digest",
    }
)
_POPULATION_LEDGER_FIELDS = frozenset(
    {
        "schema_version",
        "ledger_id",
        "policy_id",
        "checkpoint_tree_sha256",
        "inference_contract_sha256",
        "source_revision_sha256",
        "action_taxonomy_sha256",
        "population_size_k",
        "preregistered_seed_ids",
        "pair_selection_rule",
        "seed_balance_rule",
        "populations",
        "ledger_sha256",
    }
)
_POPULATION_FIELDS = frozenset(
    {
        "population_id",
        "split",
        "source_id",
        "identity_id",
        "scene_id",
        "action_family",
        "edit_instruction",
        "source_media",
        "source_intake_receipt",
        "attempts",
    }
)
_ATTEMPT_FIELDS = frozenset(
    {
        "seed_id",
        "candidate_id",
        "attempt_status",
        "failure_code",
        "output_media",
        "rollout_receipt",
        "reward_audit",
        "source_role_digest",
    }
)
_MILESTONE_FIELDS = frozenset(
    {"milestone", "frame_index", "score", "threshold", "margin", "uncertainty"}
)
_DECODED_MEDIA_FIELDS = frozenset(
    {
        "decoded_media_contract",
        "decoded_frame_count",
        "decoded_fps_numerator",
        "decoded_fps_denominator",
        "decoded_height",
        "decoded_width",
        "decoded_rgb24_sha256",
    }
)


class CAPERAdmissionError(ValueError):
    """The candidate manifest violates the closed CAPER trust boundary."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes, rejecting non-JSON/non-finite data."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CAPERAdmissionError("value is not finite canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CAPERAdmissionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise CAPERAdmissionError(f"non-finite JSON number: {value}")


def load_manifest(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load JSON while rejecting duplicate keys and non-finite constants."""

    manifest_path = Path(path)
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except CAPERAdmissionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CAPERAdmissionError(f"cannot load manifest: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise CAPERAdmissionError("manifest root must be an object")
    return payload


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CAPERAdmissionError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise CAPERAdmissionError(f"{label} keys must be strings")
    return value


def _closed(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    row = _mapping(value, label)
    actual = set(row)
    expected = set(fields)
    if actual != expected:
        raise CAPERAdmissionError(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return row


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CAPERAdmissionError(f"{label} must be an array")
    return value


def _exact_bool(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise CAPERAdmissionError(f"{label} must be exactly {str(expected).lower()}")


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CAPERAdmissionError(
            f"{label} must be a non-empty, surrounding-whitespace-free string"
        )
    if "\x00" in value:
        raise CAPERAdmissionError(f"{label} contains NUL")
    return value


def _identifier(value: Any, label: str) -> str:
    value = _nonempty_text(value, label)
    if _ID_RE.fullmatch(value) is None:
        raise CAPERAdmissionError(f"{label} is not a closed identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CAPERAdmissionError(f"{label} must be a lowercase SHA-256")
    return value


def _artifact_path(value: Any, label: str) -> str:
    value = _nonempty_text(value, label)
    if "\\" in value:
        raise CAPERAdmissionError(f"{label} must use POSIX separators")
    components = value.split("/")
    if value.startswith("/"):
        components = components[1:]
    if not components or any(part in ("", ".", "..") for part in components):
        raise CAPERAdmissionError(f"{label} must not contain dot/traversal components")
    if value.endswith("/"):
        raise CAPERAdmissionError(f"{label} must name a file")
    return value


def _artifact(value: Any, label: str) -> dict[str, Any]:
    row = _closed(value, _ARTIFACT_FIELDS, label)
    path = _artifact_path(row["path"], f"{label}.path")
    digest = _sha256(row["sha256"], f"{label}.sha256")
    size = row["size_bytes"]
    if type(size) is not int or size <= 0:
        raise CAPERAdmissionError(f"{label}.size_bytes must be a positive integer")
    return {"path": path, "sha256": digest, "size_bytes": size}


def _resolve_artifact_path(path: str, base_dir: Path | None) -> Path:
    artifact = Path(path)
    if artifact.is_absolute():
        return artifact
    if base_dir is None:
        raise CAPERAdmissionError(
            f"relative artifact path requires base_dir during verification: {path}"
        )
    base = base_dir.resolve(strict=False)
    # Resolve the parent only.  Resolving ``candidate`` itself would dereference
    # a final-component symlink before :func:`_verify_artifact` can reject it.
    candidate = base / artifact
    resolved_parent = candidate.parent.resolve(strict=False)
    try:
        resolved_parent.relative_to(base)
    except ValueError as error:
        raise CAPERAdmissionError(
            f"artifact parent escapes base_dir (possibly through a symlink): {path}"
        ) from error
    return candidate


def _verify_artifact(artifact: Mapping[str, Any], *, base_dir: Path | None) -> None:
    path = _resolve_artifact_path(str(artifact["path"]), base_dir)
    if path.is_symlink() or not path.is_file():
        raise CAPERAdmissionError(f"artifact is missing, non-file, or symlinked: {path}")
    stat = path.stat()
    if stat.st_size != artifact["size_bytes"]:
        raise CAPERAdmissionError(f"artifact size differs: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != artifact["sha256"]:
        raise CAPERAdmissionError(f"artifact SHA-256 differs: {path}")


def _decode_exact81_media_artifact(
    artifact: Mapping[str, Any], *, base_dir: Path | None, label: str
) -> dict[str, Any]:
    """Decode actual bytes with PyAV and hash logical RGB24 frames."""

    _verify_artifact(artifact, base_dir=base_dir)
    path = _resolve_artifact_path(str(artifact["path"]), base_dir)
    try:
        import av
    except Exception as error:
        raise CAPERAdmissionError(
            "PyAV is required for Stage-1 decoded-media admission"
        ) from error
    digest = hashlib.sha256()
    frame_count = 0
    geometry: set[tuple[int, int]] = set()
    try:
        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise CAPERAdmissionError(f"{label} must have one video stream")
            stream = streams[0]
            rate = stream.average_rate
            if rate is None or int(rate.numerator) != 25 or int(rate.denominator) != 1:
                raise CAPERAdmissionError(f"{label} must decode at exact fps 25/1")
            for frame in container.decode(stream):
                rgb = frame.to_ndarray(format="rgb24")
                if rgb.ndim != 3 or int(rgb.shape[2]) != 3:
                    raise CAPERAdmissionError(f"{label} decoded frame is not RGB24")
                height, width = int(rgb.shape[0]), int(rgb.shape[1])
                geometry.add((height, width))
                digest.update(struct.pack(">III", frame_count, height, width))
                digest.update(rgb.tobytes(order="C"))
                frame_count += 1
    except CAPERAdmissionError:
        raise
    except Exception as error:
        raise CAPERAdmissionError(f"{label} media decode failed") from error
    if frame_count != 81 or len(geometry) != 1:
        raise CAPERAdmissionError(
            f"{label} must decode to exact81 fixed-geometry frames"
        )
    height, width = next(iter(geometry))
    if height <= 0 or width <= 0:
        raise CAPERAdmissionError(f"{label} decoded geometry is empty")
    # Detect a file replacement during decode.
    _verify_artifact(artifact, base_dir=base_dir)
    return {
        "decoded_media_contract": DECODED_MEDIA_CONTRACT,
        "decoded_frame_count": 81,
        "decoded_fps_numerator": 25,
        "decoded_fps_denominator": 1,
        "decoded_height": height,
        "decoded_width": width,
        "decoded_rgb24_sha256": digest.hexdigest(),
    }


def _finite_unit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CAPERAdmissionError(f"{label} must be a real scalar")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise CAPERAdmissionError(f"{label} must be finite in [0,1]")
    return numeric


def _decoded_media_commitment(value: Any, label: str) -> dict[str, Any]:
    row = _closed(value, _DECODED_MEDIA_FIELDS, label)
    if (
        row["decoded_media_contract"] != DECODED_MEDIA_CONTRACT
        or row["decoded_frame_count"] != 81
        or row["decoded_fps_numerator"] != 25
        or row["decoded_fps_denominator"] != 1
        or type(row["decoded_height"]) is not int
        or row["decoded_height"] <= 0
        or type(row["decoded_width"]) is not int
        or row["decoded_width"] <= 0
    ):
        raise CAPERAdmissionError(f"{label} decoded-media contract differs")
    _sha256(row["decoded_rgb24_sha256"], f"{label}.decoded_rgb24_sha256")
    return dict(row)


def _reward_evaluator(value: Any) -> dict[str, Any]:
    row = _closed(value, _REWARD_EVALUATOR_FIELDS, "reward_evaluator")
    evaluator = {
        "evaluator_id": _identifier(row["evaluator_id"], "reward_evaluator.evaluator_id"),
        "evaluator_type": row["evaluator_type"],
        "evaluator_version": _identifier(
            row["evaluator_version"], "reward_evaluator.evaluator_version"
        ),
        "weights_artifact": _artifact(
            row["weights_artifact"], "reward_evaluator.weights_artifact"
        ),
        "runtime_artifact": _artifact(
            row["runtime_artifact"], "reward_evaluator.runtime_artifact"
        ),
        "input_contract_sha256": _sha256(
            row["input_contract_sha256"], "reward_evaluator.input_contract_sha256"
        ),
        "required_inputs": list(
            _list(row["required_inputs"], "reward_evaluator.required_inputs")
        ),
        "optional_inputs": list(
            _list(row["optional_inputs"], "reward_evaluator.optional_inputs")
        ),
        "forbidden_inputs": list(
            _list(row["forbidden_inputs"], "reward_evaluator.forbidden_inputs")
        ),
    }
    if evaluator["evaluator_type"] != REWARD_EVALUATOR_TYPE:
        raise CAPERAdmissionError("reward_evaluator.evaluator_type differs")
    if tuple(evaluator["required_inputs"]) != REWARD_EVALUATOR_INPUTS:
        raise CAPERAdmissionError("reward evaluator required input closure differs")
    if tuple(evaluator["optional_inputs"]) != REWARD_EVALUATOR_OPTIONAL_INPUTS:
        raise CAPERAdmissionError("reward evaluator optional input closure differs")
    if tuple(evaluator["forbidden_inputs"]) != REWARD_EVALUATOR_FORBIDDEN_INPUTS:
        raise CAPERAdmissionError("reward evaluator forbidden input closure differs")
    return evaluator


def text_sha256(value: Any, label: str = "text") -> str:
    value = _nonempty_text(value, label)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_role_digest(value: Mapping[str, Any]) -> str:
    """Commit immutable rollout lineage and source role, never media path.

    ``output_media_sha256`` is deliberately absent: a rename or transcode may
    change the selected media bytes, but it cannot change a pure-T2V lineage
    into a source-conditioned lineage.  The exhaustive exposure ledger binds
    the current output bytes separately.
    """

    fields = (
        "lineage_root_sha256",
        "rollout_role",
        "split",
        "policy_id",
        "checkpoint_tree_sha256",
        "inference_contract_sha256",
        "source_id",
        "source_media_sha256",
        "action_family",
        "instruction_text_sha256",
        "accepted_inputs",
        "sealed_gaussian_sha256",
        "population_id",
        "seed_id",
        "attempt_status",
    )
    if set(value) != set(fields):
        raise CAPERAdmissionError("source-role digest payload keys differ")
    return object_sha256({key: value[key] for key in fields})


def _sealed_payload(value: Mapping[str, Any], seal_field: str, label: str) -> None:
    digest = _sha256(value.get(seal_field), f"{label}.{seal_field}")
    unsigned = dict(value)
    del unsigned[seal_field]
    if object_sha256(unsigned) != digest:
        raise CAPERAdmissionError(f"{label} self-seal differs")


def _load_canonical_json_artifact(
    artifact: Mapping[str, Any],
    *,
    base_dir: Path | None,
    label: str,
) -> Mapping[str, Any]:
    path = _resolve_artifact_path(str(artifact["path"]), base_dir)
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except CAPERAdmissionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CAPERAdmissionError(f"{label} is not canonical JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise CAPERAdmissionError(f"{label} root must be an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise CAPERAdmissionError(f"{label} bytes are not canonical JSON plus newline")
    return value


def _validate_rollout_receipt(
    value: Mapping[str, Any],
    *,
    expected_rollout_id: str,
    expected_role: str,
    expected_split: str,
    expected_policy_binding: Mapping[str, str],
    expected_population_id: str | None,
    expected_seed_id: str | None,
    expected_source_id: str | None,
    expected_source_media_sha256: str | None,
    expected_action_family: str,
    expected_instruction_text: str,
    expected_output_media: Mapping[str, Any],
    expected_decoded_media: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    row = _closed(value, _ROLLOUT_RECEIPT_FIELDS, label)
    if row["schema_version"] != ROLLOUT_RECEIPT_SCHEMA_VERSION:
        raise CAPERAdmissionError(f"{label}.schema_version differs")
    _sealed_payload(row, "receipt_sha256", label)
    if _identifier(row["rollout_id"], f"{label}.rollout_id") != expected_rollout_id:
        raise CAPERAdmissionError(f"{label}.rollout_id differs")
    lineage = _sha256(row["lineage_root_sha256"], f"{label}.lineage_root_sha256")
    if row["rollout_role"] != expected_role:
        raise CAPERAdmissionError(f"{label}.rollout_role differs")
    if row["split"] != expected_split:
        raise CAPERAdmissionError(f"{label}.split differs")
    for key in _POLICY_BINDING_FIELDS:
        observed = (
            _identifier(row[key], f"{label}.{key}")
            if key == "policy_id"
            else _sha256(row[key], f"{label}.{key}")
        )
        if observed != expected_policy_binding[key]:
            raise CAPERAdmissionError(f"{label}.{key} differs")
    action_family = _identifier(row["action_family"], f"{label}.action_family")
    if action_family != expected_action_family:
        raise CAPERAdmissionError(f"{label}.action_family differs")
    instruction_digest = _sha256(
        row["instruction_text_sha256"], f"{label}.instruction_text_sha256"
    )
    if instruction_digest != text_sha256(expected_instruction_text, label):
        raise CAPERAdmissionError(f"{label}.instruction_text_sha256 differs")
    gaussian = _sha256(
        row["sealed_gaussian_sha256"], f"{label}.sealed_gaussian_sha256"
    )
    if row["population_id"] != expected_population_id:
        raise CAPERAdmissionError(f"{label}.population_id differs")
    if row["seed_id"] != expected_seed_id:
        raise CAPERAdmissionError(f"{label}.seed_id differs")
    if row["attempt_status"] != "success" or row["failure_code"] is not None:
        raise CAPERAdmissionError(f"{label} successful rollout status differs")
    if expected_role == SOURCE_CANDIDATE_ROLE:
        if row["source_id"] != expected_source_id:
            raise CAPERAdmissionError(f"{label}.source_id differs")
        if row["source_media_sha256"] != expected_source_media_sha256:
            raise CAPERAdmissionError(f"{label}.source_media_sha256 differs")
        if tuple(row["accepted_inputs"]) != SOURCE_CONDITIONING_INPUTS:
            raise CAPERAdmissionError(f"{label}.accepted_inputs differs")
    elif expected_role == PURE_T2V_LEDGER_ROLE:
        if row["source_id"] is not None or row["source_media_sha256"] is not None:
            raise CAPERAdmissionError(f"{label} pure-T2V source fields must be null")
        if tuple(row["accepted_inputs"]) != PURE_T2V_INPUTS:
            raise CAPERAdmissionError(f"{label}.accepted_inputs differs")
    else:  # pragma: no cover - all callers pass a closed role.
        raise CAPERAdmissionError(f"{label} rollout role is unsupported")
    for key in (
        "target_video_read",
        "target_video_latent_read",
        "paired_target_read",
        "pure_t2v_media_read",
        "pure_t2v_latent_read",
    ):
        _exact_bool(row[key], False, f"{label}.{key}")
    output_sha = _sha256(row["output_media_sha256"], f"{label}.output_media_sha256")
    if output_sha != expected_output_media["sha256"]:
        raise CAPERAdmissionError(f"{label}.output_media_sha256 differs")
    if (
        type(row["output_media_size_bytes"]) is not int
        or row["output_media_size_bytes"] != expected_output_media["size_bytes"]
    ):
        raise CAPERAdmissionError(f"{label}.output_media_size_bytes differs")
    if (
        type(row["frame_count"]) is not int
        or row["frame_count"] != 81
        or type(row["fps"]) is not int
        or row["fps"] != 25
    ):
        raise CAPERAdmissionError(f"{label} must bind exact81 at 25 fps")
    for key in (
        "decoded_media_contract",
        "decoded_frame_count",
        "decoded_fps_numerator",
        "decoded_fps_denominator",
        "decoded_height",
        "decoded_width",
        "decoded_rgb24_sha256",
    ):
        if row[key] != expected_decoded_media[key]:
            raise CAPERAdmissionError(f"{label}.{key} differs from actual decoded bytes")
    digest_payload = {
        "lineage_root_sha256": lineage,
        "rollout_role": expected_role,
        "split": expected_split,
        "policy_id": expected_policy_binding["policy_id"],
        "checkpoint_tree_sha256": expected_policy_binding["checkpoint_tree_sha256"],
        "inference_contract_sha256": expected_policy_binding[
            "inference_contract_sha256"
        ],
        "source_id": expected_source_id,
        "source_media_sha256": expected_source_media_sha256,
        "action_family": expected_action_family,
        "instruction_text_sha256": instruction_digest,
        "accepted_inputs": list(
            SOURCE_CONDITIONING_INPUTS
            if expected_role == SOURCE_CANDIDATE_ROLE
            else PURE_T2V_INPUTS
        ),
        "sealed_gaussian_sha256": gaussian,
        "population_id": expected_population_id,
        "seed_id": expected_seed_id,
        "attempt_status": "success",
    }
    role_digest = _sha256(row["source_role_digest"], f"{label}.source_role_digest")
    if role_digest != source_role_digest(digest_payload):
        raise CAPERAdmissionError(f"{label}.source_role_digest differs")
    return dict(row)


def _validate_milestone_evidence(
    value: Any, *, action_grade: str, label: str
) -> list[dict[str, Any]]:
    rows = _list(value, label)
    if len(rows) != len(MILESTONES):
        raise CAPERAdmissionError(f"{label} must contain four milestones")
    result: list[dict[str, Any]] = []
    previous_frame = -1
    for index, raw in enumerate(rows):
        row_label = f"{label}[{index}]"
        row = _closed(raw, _MILESTONE_FIELDS, row_label)
        if row["milestone"] != MILESTONES[index]:
            raise CAPERAdmissionError(f"{row_label}.milestone order differs")
        frame_index = row["frame_index"]
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not 0 <= frame_index < 81
            or frame_index <= previous_frame
        ):
            raise CAPERAdmissionError(f"{row_label}.frame_index differs")
        previous_frame = frame_index
        score = _finite_unit(row["score"], f"{row_label}.score")
        threshold = _finite_unit(row["threshold"], f"{row_label}.threshold")
        uncertainty = _finite_unit(row["uncertainty"], f"{row_label}.uncertainty")
        margin = row["margin"]
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or abs(float(margin) - (score - threshold)) > 1.0e-9
        ):
            raise CAPERAdmissionError(f"{row_label}.margin differs")
        result.append(
            {
                "milestone": row["milestone"],
                "frame_index": frame_index,
                "score": score,
                "threshold": threshold,
                "margin": float(margin),
                "uncertainty": uncertainty,
            }
        )
    passes = {
        row["milestone"]: row["score"] - row["uncertainty"] >= row["threshold"]
        for row in result
    }
    grade_valid = {
        "terminal_reached_and_held": (
            passes["transition"] and passes["terminal"] and passes["hold"]
        ),
        "terminal_reached": (
            passes["transition"] and passes["terminal"] and not passes["hold"]
        ),
        "transition_incomplete": (
            passes["transition"] and not passes["terminal"]
        ),
        "started_only": not passes["transition"],
        "absent_or_wrong": not passes["transition"],
    }[action_grade]
    if not grade_valid:
        raise CAPERAdmissionError(f"{label} contradicts categorical action grade")
    return result


def _validate_reward_audit(
    value: Mapping[str, Any],
    *,
    expected_rollout_id: str,
    expected_lineage_root_sha256: str,
    expected_source_role_digest: str,
    expected_output_media_sha256: str,
    expected_rollout_receipt_artifact_sha256: str,
    expected_audit_role: str,
    expected_checkpoint_tree_sha256: str,
    expected_action_family: str,
    expected_audit_protocol_sha256: str,
    expected_evaluator: Mapping[str, Any],
    expected_rubric_id: str | None,
    expected_action_grade: str | None,
    expected_hard_gates: Mapping[str, bool] | None,
    expected_action_score: float | None,
    expected_action_uncertainty: float | None,
    label: str,
) -> dict[str, Any]:
    row = _closed(value, _REWARD_AUDIT_FIELDS, label)
    if row["schema_version"] != REWARD_AUDIT_SCHEMA_VERSION:
        raise CAPERAdmissionError(f"{label}.schema_version differs")
    _sealed_payload(row, "audit_sha256", label)
    exact = {
        "rollout_id": expected_rollout_id,
        "lineage_root_sha256": expected_lineage_root_sha256,
        "source_role_digest": expected_source_role_digest,
        "output_media_sha256": expected_output_media_sha256,
        "rollout_receipt_artifact_sha256": expected_rollout_receipt_artifact_sha256,
        "audit_role": expected_audit_role,
        "checkpoint_tree_sha256": expected_checkpoint_tree_sha256,
        "action_family": expected_action_family,
        "audit_protocol_sha256": expected_audit_protocol_sha256,
    }
    for key, expected in exact.items():
        if row[key] != expected:
            raise CAPERAdmissionError(f"{label}.{key} differs")
    evaluator_binding = _closed(
        row["evaluator_binding"], _EVALUATOR_BINDING_FIELDS, f"{label}.evaluator_binding"
    )
    expected_binding = {
        "evaluator_id": expected_evaluator["evaluator_id"],
        "evaluator_type": expected_evaluator["evaluator_type"],
        "evaluator_version": expected_evaluator["evaluator_version"],
        "weights_artifact_sha256": expected_evaluator["weights_artifact"]["sha256"],
        "runtime_artifact_sha256": expected_evaluator["runtime_artifact"]["sha256"],
        "input_contract_sha256": expected_evaluator["input_contract_sha256"],
    }
    if dict(evaluator_binding) != expected_binding:
        raise CAPERAdmissionError(f"{label}.evaluator_binding differs")
    _identifier(row["audit_id"], f"{label}.audit_id")
    rubric_id = _identifier(row["rubric_id"], f"{label}.rubric_id")
    if expected_rubric_id is not None and rubric_id != expected_rubric_id:
        raise CAPERAdmissionError(f"{label}.rubric_id differs")
    if row["action_grade"] not in ACTION_GRADES:
        raise CAPERAdmissionError(f"{label}.action_grade differs")
    if expected_action_grade is not None and row["action_grade"] != expected_action_grade:
        raise CAPERAdmissionError(f"{label}.action_grade differs from pair ordering")
    action_score = _finite_unit(row["action_score"], f"{label}.action_score")
    uncertainty = _finite_unit(
        row["action_uncertainty"], f"{label}.action_uncertainty"
    )
    if (
        expected_action_score is not None
        and abs(action_score - expected_action_score) > 1.0e-9
    ) or (
        expected_action_uncertainty is not None
        and abs(uncertainty - expected_action_uncertainty) > 1.0e-9
    ):
        raise CAPERAdmissionError(f"{label} score/uncertainty differs from pair rule")
    margin = row["action_margin_to_grade_threshold"]
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or abs(float(margin) - (action_score - 0.5)) > 1.0e-9
    ):
        raise CAPERAdmissionError(f"{label}.action_margin_to_grade_threshold differs")
    _validate_milestone_evidence(
        row["milestone_frame_evidence"],
        action_grade=row["action_grade"],
        label=f"{label}.milestone_frame_evidence",
    )
    _exact_bool(
        row["scalar_compensation_used"], False, f"{label}.scalar_compensation_used"
    )
    _exact_bool(
        row["target_video_dependency"], False, f"{label}.target_video_dependency"
    )
    for key in (
        "mask_read",
        "track_read",
        "pose_read",
        "flow_read",
        "pure_t2v_origin_metadata_read",
    ):
        _exact_bool(row[key], False, f"{label}.{key}")
    if expected_audit_role == SOURCE_AUDIT_ROLE:
        if tuple(row["evaluator_inputs"]) != (
            *REWARD_EVALUATOR_INPUTS,
            *REWARD_EVALUATOR_OPTIONAL_INPUTS,
        ):
            raise CAPERAdmissionError(f"{label}.evaluator_inputs differs")
        gates = _hard_gates(row["preservation_hard_gates"], f"{label}.preservation_hard_gates")
        if expected_hard_gates is not None and gates != dict(expected_hard_gates):
            raise CAPERAdmissionError(f"{label} hard gates differ from manifest")
    elif expected_audit_role == PURE_T2V_AUDIT_ROLE:
        if tuple(row["evaluator_inputs"]) != REWARD_EVALUATOR_INPUTS:
            raise CAPERAdmissionError(f"{label}.evaluator_inputs differs")
        if row["preservation_hard_gates"] is not None:
            raise CAPERAdmissionError(f"{label} pure-T2V preservation gates must be null")
    else:  # pragma: no cover
        raise CAPERAdmissionError(f"{label}.audit_role is unsupported")
    return dict(row)


def _validate_exposure_ledger(
    value: Mapping[str, Any], *, policy: Mapping[str, Any], label: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the exhaustive immutable rollout-role registry.

    The registry deliberately binds lineage and role independently of the
    current media path/bytes.  Consequently renaming or transcoding a pure-T2V
    output cannot turn its lineage into a source-conditioned endpoint.
    """

    row = _closed(value, _EXPOSURE_LEDGER_FIELDS, label)
    if row["schema_version"] != EXPOSURE_LEDGER_SCHEMA_VERSION:
        raise CAPERAdmissionError(f"{label}.schema_version differs")
    _sealed_payload(row, "ledger_sha256", label)
    _identifier(row["ledger_id"], f"{label}.ledger_id")
    if (
        _sha256(row["checkpoint_tree_sha256"], f"{label}.checkpoint_tree_sha256")
        != policy["checkpoint_tree_sha256"]
        or _sha256(
            row["inference_contract_sha256"],
            f"{label}.inference_contract_sha256",
        )
        != policy["inference_contract_sha256"]
    ):
        raise CAPERAdmissionError(f"{label} deployable-policy binding differs")
    for key in (
        "source_revision_sha256",
        "action_taxonomy_sha256",
        "reward_audit_protocol_sha256",
    ):
        _sha256(row[key], f"{label}.{key}")

    entries_raw = _list(row["entries"], f"{label}.entries")
    if not entries_raw:
        raise CAPERAdmissionError(f"{label}.entries must be non-empty")
    if entries_raw != sorted(entries_raw, key=lambda item: item.get("rollout_id", "")):
        raise CAPERAdmissionError(f"{label}.entries must be sorted by rollout_id")
    entries: dict[str, dict[str, Any]] = {}
    lineages: set[str] = set()
    for index, raw in enumerate(entries_raw):
        entry_label = f"{label}.entries[{index}]"
        entry = dict(_closed(raw, _EXPOSURE_ENTRY_FIELDS, entry_label))
        rollout_id = _identifier(entry["rollout_id"], f"{entry_label}.rollout_id")
        if rollout_id in entries:
            raise CAPERAdmissionError(f"duplicate exposure-ledger rollout_id: {rollout_id}")
        lineage = _sha256(
            entry["lineage_root_sha256"], f"{entry_label}.lineage_root_sha256"
        )
        if lineage in lineages:
            raise CAPERAdmissionError("exposure-ledger lineage roots must be unique")
        lineages.add(lineage)
        if entry["rollout_role"] not in (
            SOURCE_CANDIDATE_ROLE,
            PURE_T2V_LEDGER_ROLE,
            FAILED_ROLLOUT_ROLE,
        ):
            raise CAPERAdmissionError(f"{entry_label}.rollout_role differs")
        if entry["split"] not in SPLITS:
            raise CAPERAdmissionError(f"{entry_label}.split differs")
        _identifier(entry["policy_id"], f"{entry_label}.policy_id")
        for key in (
            "checkpoint_tree_sha256",
            "inference_contract_sha256",
            "instruction_text_sha256",
            "rollout_receipt_artifact_sha256",
            "source_role_digest",
        ):
            _sha256(entry[key], f"{entry_label}.{key}")
        _identifier(entry["action_family"], f"{entry_label}.action_family")
        if entry["rollout_role"] in (
            SOURCE_CANDIDATE_ROLE,
            FAILED_ROLLOUT_ROLE,
        ):
            _identifier(entry["source_id"], f"{entry_label}.source_id")
            _identifier(entry["population_id"], f"{entry_label}.population_id")
            _identifier(entry["seed_id"], f"{entry_label}.seed_id")
            _sha256(
                entry["source_media_sha256"],
                f"{entry_label}.source_media_sha256",
            )
            if entry["rollout_role"] == SOURCE_CANDIDATE_ROLE:
                _sha256(entry["output_media_sha256"], f"{entry_label}.output_media_sha256")
                _sha256(
                    entry["reward_audit_artifact_sha256"],
                    f"{entry_label}.reward_audit_artifact_sha256",
                )
            elif (
                entry["output_media_sha256"] is not None
                or entry["reward_audit_artifact_sha256"] is not None
            ):
                raise CAPERAdmissionError(
                    f"{entry_label} failed rollout cannot claim output/reward"
                )
        elif any(
            entry[key] is not None
            for key in ("source_id", "source_media_sha256", "population_id", "seed_id")
        ):
            raise CAPERAdmissionError(
                f"{entry_label} pure-T2V source fields must be null"
            )
        else:
            _sha256(entry["output_media_sha256"], f"{entry_label}.output_media_sha256")
            _sha256(
                entry["reward_audit_artifact_sha256"],
                f"{entry_label}.reward_audit_artifact_sha256",
            )
        entries[rollout_id] = entry
    return dict(row), entries


def _validate_population_ledger(
    value: Mapping[str, Any], *, policy: Mapping[str, Any], label: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    row = _closed(value, _POPULATION_LEDGER_FIELDS, label)
    if row["schema_version"] != POPULATION_LEDGER_SCHEMA_VERSION:
        raise CAPERAdmissionError(f"{label}.schema_version differs")
    _sealed_payload(row, "ledger_sha256", label)
    _identifier(row["ledger_id"], f"{label}.ledger_id")
    exact = {
        "policy_id": policy["policy_id"],
        "checkpoint_tree_sha256": policy["checkpoint_tree_sha256"],
        "inference_contract_sha256": policy["inference_contract_sha256"],
    }
    for key, expected in exact.items():
        if row[key] != expected:
            raise CAPERAdmissionError(f"{label}.{key} differs")
    for key in ("source_revision_sha256", "action_taxonomy_sha256"):
        _sha256(row[key], f"{label}.{key}")
    population_size_k = row["population_size_k"]
    if (
        type(population_size_k) is not int
        or not MIN_POPULATION_SIZE_K
        <= population_size_k
        <= MAX_POPULATION_SIZE_K
    ):
        raise CAPERAdmissionError(f"{label}.population_size_k differs")
    preregistered_seed_ids = _list(
        row["preregistered_seed_ids"], f"{label}.preregistered_seed_ids"
    )
    expected_seed_ids = [f"seed-{index}" for index in range(population_size_k)]
    if (
        preregistered_seed_ids != expected_seed_ids
        or row["pair_selection_rule"] != PAIR_SELECTION_RULE
        or row["seed_balance_rule"] != SEED_BALANCE_RULE
    ):
        raise CAPERAdmissionError(f"{label} preregistered K-seed contract differs")
    raw_populations = _list(row["populations"], f"{label}.populations")
    if not raw_populations:
        raise CAPERAdmissionError(f"{label}.populations must be non-empty")
    if raw_populations != sorted(
        raw_populations, key=lambda item: item.get("population_id", "")
    ):
        raise CAPERAdmissionError(f"{label}.populations must be sorted")
    populations: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    nested_artifacts: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for index, raw in enumerate(raw_populations):
        population_label = f"{label}.populations[{index}]"
        item = _closed(raw, _POPULATION_FIELDS, population_label)
        population_id = _identifier(
            item["population_id"], f"{population_label}.population_id"
        )
        source_id = _identifier(item["source_id"], f"{population_label}.source_id")
        if population_id in populations or source_id in source_ids:
            raise CAPERAdmissionError("population/source IDs must be unique")
        source_ids.add(source_id)
        if item["split"] not in SPLITS:
            raise CAPERAdmissionError(f"{population_label}.split differs")
        population: dict[str, Any] = {
            "population_id": population_id,
            "split": item["split"],
            "source_id": source_id,
            "identity_id": _identifier(
                item["identity_id"], f"{population_label}.identity_id"
            ),
            "scene_id": _identifier(item["scene_id"], f"{population_label}.scene_id"),
            "action_family": _identifier(
                item["action_family"], f"{population_label}.action_family"
            ),
            "edit_instruction": _nonempty_text(
                item["edit_instruction"], f"{population_label}.edit_instruction"
            ),
            "source_media": _artifact(
                item["source_media"], f"{population_label}.source_media"
            ),
            "source_intake_receipt": _artifact(
                item["source_intake_receipt"],
                f"{population_label}.source_intake_receipt",
            ),
        }
        nested_artifacts.extend(
            (population["source_media"], population["source_intake_receipt"])
        )
        attempts_raw = _list(item["attempts"], f"{population_label}.attempts")
        if len(attempts_raw) != population_size_k:
            raise CAPERAdmissionError(
                f"{population_label} must declare every preregistered attempt"
            )
        attempts: list[dict[str, Any]] = []
        for attempt_index, attempt_raw in enumerate(attempts_raw):
            attempt_label = f"{population_label}.attempts[{attempt_index}]"
            attempt_row = _closed(attempt_raw, _ATTEMPT_FIELDS, attempt_label)
            if attempt_row["seed_id"] != preregistered_seed_ids[attempt_index]:
                raise CAPERAdmissionError(
                    f"{population_label} seed balance/order differs"
                )
            candidate_id = _identifier(
                attempt_row["candidate_id"], f"{attempt_label}.candidate_id"
            )
            if candidate_id in candidate_ids:
                raise CAPERAdmissionError("population candidate IDs must be unique")
            candidate_ids.add(candidate_id)
            attempt = {
                "seed_id": attempt_row["seed_id"],
                "candidate_id": candidate_id,
                "attempt_status": attempt_row["attempt_status"],
                "failure_code": attempt_row["failure_code"],
                "output_media": (
                    None
                    if attempt_row["output_media"] is None
                    else _artifact(
                        attempt_row["output_media"], f"{attempt_label}.output_media"
                    )
                ),
                "rollout_receipt": _artifact(
                    attempt_row["rollout_receipt"], f"{attempt_label}.rollout_receipt"
                ),
                "reward_audit": (
                    None
                    if attempt_row["reward_audit"] is None
                    else _artifact(
                        attempt_row["reward_audit"], f"{attempt_label}.reward_audit"
                    )
                ),
                "source_role_digest": _sha256(
                    attempt_row["source_role_digest"],
                    f"{attempt_label}.source_role_digest",
                ),
            }
            nested_artifacts.append(attempt["rollout_receipt"])
            if attempt["attempt_status"] == "success":
                if (
                    attempt["failure_code"] is not None
                    or attempt["output_media"] is None
                    or attempt["reward_audit"] is None
                ):
                    raise CAPERAdmissionError(f"{attempt_label} success has failure_code")
                nested_artifacts.extend(
                    (attempt["output_media"], attempt["reward_audit"])
                )
            elif attempt["attempt_status"] == "failure":
                _identifier(attempt["failure_code"], f"{attempt_label}.failure_code")
                if (
                    attempt["output_media"] is not None
                    or attempt["reward_audit"] is not None
                ):
                    raise CAPERAdmissionError(
                        f"{attempt_label} failure must not claim output/reward"
                    )
            else:
                raise CAPERAdmissionError(f"{attempt_label}.attempt_status differs")
            attempts.append(attempt)
        population["attempts"] = attempts
        populations[population_id] = population
    return dict(row), populations, nested_artifacts


def _validate_failed_rollout_receipt(
    value: Mapping[str, Any],
    *,
    population: Mapping[str, Any],
    attempt: Mapping[str, Any],
    policy: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    row = _closed(value, _ROLLOUT_RECEIPT_FIELDS, label)
    if row["schema_version"] != FAILED_ROLLOUT_RECEIPT_SCHEMA_VERSION:
        raise CAPERAdmissionError(f"{label}.schema_version differs")
    _sealed_payload(row, "receipt_sha256", label)
    exact = {
        "rollout_id": attempt["candidate_id"],
        "rollout_role": FAILED_ROLLOUT_ROLE,
        "split": population["split"],
        "policy_id": policy["policy_id"],
        "checkpoint_tree_sha256": policy["checkpoint_tree_sha256"],
        "inference_contract_sha256": policy["inference_contract_sha256"],
        "source_id": population["source_id"],
        "source_media_sha256": population["source_media"]["sha256"],
        "action_family": population["action_family"],
        "instruction_text_sha256": text_sha256(population["edit_instruction"]),
        "population_id": population["population_id"],
        "seed_id": attempt["seed_id"],
        "attempt_status": "failure",
        "failure_code": attempt["failure_code"],
    }
    for key, expected in exact.items():
        if row[key] != expected:
            raise CAPERAdmissionError(f"{label}.{key} differs")
    lineage = _sha256(row["lineage_root_sha256"], f"{label}.lineage_root_sha256")
    gaussian = _sha256(row["sealed_gaussian_sha256"], f"{label}.sealed_gaussian_sha256")
    if tuple(row["accepted_inputs"]) != SOURCE_CONDITIONING_INPUTS:
        raise CAPERAdmissionError(f"{label}.accepted_inputs differs")
    for key in (
        "target_video_read",
        "target_video_latent_read",
        "paired_target_read",
        "pure_t2v_media_read",
        "pure_t2v_latent_read",
    ):
        _exact_bool(row[key], False, f"{label}.{key}")
    for key in (
        "output_media_sha256",
        "output_media_size_bytes",
        "frame_count",
        "fps",
        "decoded_media_contract",
        "decoded_frame_count",
        "decoded_fps_numerator",
        "decoded_fps_denominator",
        "decoded_height",
        "decoded_width",
        "decoded_rgb24_sha256",
    ):
        if row[key] is not None:
            raise CAPERAdmissionError(f"{label}.{key} must be null after failure")
    digest_payload = {
        "lineage_root_sha256": lineage,
        "rollout_role": FAILED_ROLLOUT_ROLE,
        "split": population["split"],
        "policy_id": policy["policy_id"],
        "checkpoint_tree_sha256": policy["checkpoint_tree_sha256"],
        "inference_contract_sha256": policy["inference_contract_sha256"],
        "source_id": population["source_id"],
        "source_media_sha256": population["source_media"]["sha256"],
        "action_family": population["action_family"],
        "instruction_text_sha256": text_sha256(population["edit_instruction"]),
        "accepted_inputs": list(SOURCE_CONDITIONING_INPUTS),
        "sealed_gaussian_sha256": gaussian,
        "population_id": population["population_id"],
        "seed_id": attempt["seed_id"],
        "attempt_status": "failure",
    }
    if row["source_role_digest"] != source_role_digest(digest_payload):
        raise CAPERAdmissionError(f"{label}.source_role_digest differs")
    return dict(row)


def _entry_must_equal(
    entry: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key, value in expected.items():
        if entry[key] != value:
            raise CAPERAdmissionError(f"{label}.{key} differs")


def _validate_semantic_evidence(
    *,
    ledger_artifact: Mapping[str, Any],
    population_ledger_artifact: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    policy: Mapping[str, Any],
    pairs: Mapping[str, list[dict[str, Any]]],
    owners: Sequence[Mapping[str, Any]],
    base_dir: Path | None,
) -> dict[str, Any]:
    """Parse receipts/audits and bind them to one exhaustive role ledger."""

    if not owners:
        raise CAPERAdmissionError(
            "pure_t2v_calibration_owners must be non-empty for reward calibration"
        )
    ledger_payload = _load_canonical_json_artifact(
        ledger_artifact, base_dir=base_dir, label="exposure_ledger"
    )
    ledger, entries = _validate_exposure_ledger(
        ledger_payload, policy=policy, label="exposure_ledger"
    )
    population_payload = _load_canonical_json_artifact(
        population_ledger_artifact,
        base_dir=base_dir,
        label="rollout_population_ledger",
    )
    population_ledger, populations, population_artifacts = _validate_population_ledger(
        population_payload, policy=policy, label="rollout_population_ledger"
    )
    if (
        population_ledger["source_revision_sha256"]
        != ledger["source_revision_sha256"]
        or population_ledger["action_taxonomy_sha256"]
        != ledger["action_taxonomy_sha256"]
    ):
        raise CAPERAdmissionError(
            "population/exposure source revision or action taxonomy differs"
        )
    for artifact in population_artifacts:
        _verify_artifact(artifact, base_dir=base_dir)
    pair_by_population = {
        pair["population_id"]: pair
        for split in SPLITS
        for pair in pairs[split]
    }
    if len(pair_by_population) != sum(len(pairs[split]) for split in SPLITS):
        raise CAPERAdmissionError("one population may produce at most one pair")
    all_attempts = {
        attempt["candidate_id"]: (population, attempt)
        for population in populations.values()
        for attempt in population["attempts"]
    }
    failure_attempts = {
        attempt["candidate_id"]: (population, attempt)
        for population in populations.values()
        for attempt in population["attempts"]
        if attempt["attempt_status"] == "failure"
    }
    expected_ids = set(all_attempts) | {owner["owner_id"] for owner in owners}
    if set(entries) != expected_ids:
        raise CAPERAdmissionError(
            "exposure ledger must exhaustively equal all K-seed attempts and T2V owners"
        )
    ledger_owner_ids = {
        rollout_id
        for rollout_id, entry in entries.items()
        if entry["rollout_role"] == PURE_T2V_LEDGER_ROLE
    }
    declared_owner_ids = {owner["owner_id"] for owner in owners}
    if ledger_owner_ids != declared_owner_ids:
        raise CAPERAdmissionError(
            "pure-T2V owners differ from immutable exposure-ledger roles"
        )

    fit_action_families = {
        population["action_family"]
        for population in populations.values()
        if population["split"] == "fit"
    } | {owner["action_family"] for owner in owners if owner["split"] == "fit"}
    heldout_action_families = {
        population["action_family"]
        for population in populations.values()
        if population["split"] == "heldout"
    } | {owner["action_family"] for owner in owners if owner["split"] == "heldout"}
    if fit_action_families & heldout_action_families:
        raise CAPERAdmissionError("fit/heldout action_family leakage through owner ledger")

    # The population ledger is the preregistered sampling frame.  Split
    # isolation therefore has to be checked over *every* population, including
    # failed attempts that never yielded a pair.  Checking only selected pairs
    # would let a caller hide leakage by declaring the leaking seed a failure.
    fit_populations = [
        population
        for population in populations.values()
        if population["split"] == "fit"
    ]
    heldout_populations = [
        population
        for population in populations.values()
        if population["split"] == "heldout"
    ]
    for axis in SPLIT_ISOLATION_AXES:
        key = "sha256" if axis == "source_media_sha256" else axis
        fit_values = {
            population["source_media"][key]
            if axis == "source_media_sha256"
            else population[key]
            for population in fit_populations
        }
        heldout_values = {
            population["source_media"][key]
            if axis == "source_media_sha256"
            else population[key]
            for population in heldout_populations
        }
        if fit_values & heldout_values:
            raise CAPERAdmissionError(f"population {axis} leakage across splits")

    decoded_media_by_rollout: dict[str, dict[str, Any]] = {}
    for population_id, population in populations.items():
        successful = [
            attempt
            for attempt in population["attempts"]
            if attempt["attempt_status"] == "success"
        ]
        pair = pair_by_population.get(population_id)
        if len(successful) >= 2:
            if pair is None:
                raise CAPERAdmissionError(
                    "population with at least two successes is missing its fixed-rule pair"
                )
            source_exact = {
                "split": pair["split"],
                "source_id": pair["source_id"],
                "identity_id": pair["identity_id"],
                "scene_id": pair["scene_id"],
                "action_family": pair["action_family"],
                "edit_instruction": pair["edit_instruction"],
                "source_media": pair["source_media"],
                "source_intake_receipt": pair["source_intake_receipt"],
            }
            for key, observed in source_exact.items():
                if population[key] != observed:
                    raise CAPERAdmissionError(
                        f"population {population_id} {key} differs from pair"
                    )
            endpoint_by_id = {
                pair[role]["candidate_id"]: pair[role]
                for role in ("winner", "loser")
            }
            if not set(endpoint_by_id) <= {
                attempt["candidate_id"] for attempt in successful
            }:
                raise CAPERAdmissionError("pair endpoint is not a successful attempt")
            for attempt in successful:
                endpoint = endpoint_by_id.get(attempt["candidate_id"])
                if endpoint is None:
                    continue
                if (
                    endpoint["seed_id"] != attempt["seed_id"]
                    or endpoint["rollout_receipt"] != attempt["rollout_receipt"]
                    or endpoint["output_media"] != attempt["output_media"]
                    or endpoint["reward_audit"] != attempt["reward_audit"]
                ):
                    raise CAPERAdmissionError(
                        "population attempt does not exactly bind endpoint evidence"
                    )
        elif pair is not None:
            raise CAPERAdmissionError(
                "a population with fewer than two successes cannot yield a pair"
            )
        _decode_exact81_media_artifact(
            population["source_media"],
            base_dir=base_dir,
            label=f"population {population_id} source",
        )

    if set(pair_by_population) - set(populations):
        raise CAPERAdmissionError("pair references an unregistered population")

    for rollout_id, (population, attempt) in failure_attempts.items():
        receipt = _validate_failed_rollout_receipt(
            _load_canonical_json_artifact(
                attempt["rollout_receipt"],
                base_dir=base_dir,
                label=f"failed attempt {rollout_id} receipt",
            ),
            population=population,
            attempt=attempt,
            policy=policy,
            label=f"failed attempt {rollout_id} receipt",
        )
        if attempt["source_role_digest"] != receipt["source_role_digest"]:
            raise CAPERAdmissionError("failed population role digest differs")
        _entry_must_equal(
            entries[rollout_id],
            {
                "rollout_id": rollout_id,
                "lineage_root_sha256": receipt["lineage_root_sha256"],
                "rollout_role": FAILED_ROLLOUT_ROLE,
                "split": population["split"],
                "population_id": population["population_id"],
                "seed_id": attempt["seed_id"],
                "policy_id": policy["policy_id"],
                "checkpoint_tree_sha256": policy["checkpoint_tree_sha256"],
                "inference_contract_sha256": policy["inference_contract_sha256"],
                "source_id": population["source_id"],
                "source_media_sha256": population["source_media"]["sha256"],
                "action_family": population["action_family"],
                "instruction_text_sha256": text_sha256(
                    population["edit_instruction"]
                ),
                "output_media_sha256": None,
                "rollout_receipt_artifact_sha256": attempt["rollout_receipt"][
                    "sha256"
                ],
                "reward_audit_artifact_sha256": None,
                "source_role_digest": receipt["source_role_digest"],
            },
            label=f"ledger entry {rollout_id}",
        )

    audit_protocol = ledger["reward_audit_protocol_sha256"]
    successful_evidence: dict[str, dict[str, Any]] = {}
    for rollout_id, (population, attempt) in all_attempts.items():
        if attempt["attempt_status"] != "success":
            continue
        output_media = attempt["output_media"]
        reward_audit = attempt["reward_audit"]
        assert output_media is not None and reward_audit is not None
        decoded_media = _decode_exact81_media_artifact(
            output_media,
            base_dir=base_dir,
            label=f"population attempt {rollout_id} output",
        )
        decoded_media_by_rollout[rollout_id] = decoded_media
        receipt = _validate_rollout_receipt(
            _load_canonical_json_artifact(
                attempt["rollout_receipt"],
                base_dir=base_dir,
                label=f"population attempt {rollout_id} receipt",
            ),
            expected_rollout_id=rollout_id,
            expected_role=SOURCE_CANDIDATE_ROLE,
            expected_split=population["split"],
            expected_policy_binding=policy,
            expected_population_id=population["population_id"],
            expected_seed_id=attempt["seed_id"],
            expected_source_id=population["source_id"],
            expected_source_media_sha256=population["source_media"]["sha256"],
            expected_action_family=population["action_family"],
            expected_instruction_text=population["edit_instruction"],
            expected_output_media=output_media,
            expected_decoded_media=decoded_media,
            label=f"population attempt {rollout_id} receipt",
        )
        if attempt["source_role_digest"] != receipt["source_role_digest"]:
            raise CAPERAdmissionError("population success role digest differs")
        audit = _validate_reward_audit(
            _load_canonical_json_artifact(
                reward_audit,
                base_dir=base_dir,
                label=f"population attempt {rollout_id} reward audit",
            ),
            expected_rollout_id=rollout_id,
            expected_lineage_root_sha256=receipt["lineage_root_sha256"],
            expected_source_role_digest=receipt["source_role_digest"],
            expected_output_media_sha256=output_media["sha256"],
            expected_rollout_receipt_artifact_sha256=attempt["rollout_receipt"][
                "sha256"
            ],
            expected_audit_role=SOURCE_AUDIT_ROLE,
            expected_checkpoint_tree_sha256=policy["checkpoint_tree_sha256"],
            expected_action_family=population["action_family"],
            expected_audit_protocol_sha256=audit_protocol,
            expected_evaluator=evaluator,
            expected_rubric_id=None,
            expected_action_grade=None,
            expected_hard_gates=None,
            expected_action_score=None,
            expected_action_uncertainty=None,
            label=f"population attempt {rollout_id} reward audit",
        )
        _entry_must_equal(
            entries[rollout_id],
            {
                "rollout_id": rollout_id,
                "lineage_root_sha256": receipt["lineage_root_sha256"],
                "rollout_role": SOURCE_CANDIDATE_ROLE,
                "split": population["split"],
                "population_id": population["population_id"],
                "seed_id": attempt["seed_id"],
                "policy_id": policy["policy_id"],
                "checkpoint_tree_sha256": policy["checkpoint_tree_sha256"],
                "inference_contract_sha256": policy["inference_contract_sha256"],
                "source_id": population["source_id"],
                "source_media_sha256": population["source_media"]["sha256"],
                "action_family": population["action_family"],
                "instruction_text_sha256": text_sha256(
                    population["edit_instruction"]
                ),
                "output_media_sha256": output_media["sha256"],
                "rollout_receipt_artifact_sha256": attempt["rollout_receipt"][
                    "sha256"
                ],
                "reward_audit_artifact_sha256": reward_audit["sha256"],
                "source_role_digest": receipt["source_role_digest"],
            },
            label=f"ledger entry {rollout_id}",
        )
        successful_evidence[rollout_id] = {
            "population": population,
            "attempt": attempt,
            "receipt": receipt,
            "audit": audit,
        }

    # Selection is recomputed over the complete preregistered population.  A
    # manifest cannot register only its favourite seed: every success is
    # decoded/audited above, then this deterministic rule fixes the endpoints.
    for population_id, population in populations.items():
        successes = [
            successful_evidence[attempt["candidate_id"]]
            for attempt in population["attempts"]
            if attempt["attempt_status"] == "success"
        ]
        if len(successes) < 2:
            continue
        pair = pair_by_population[population_id]
        seed_rank = {
            attempt["seed_id"]: index
            for index, attempt in enumerate(population["attempts"])
        }
        winner = max(
            successes,
            key=lambda evidence: (
                evidence["audit"]["action_score"]
                - evidence["audit"]["action_uncertainty"],
                -seed_rank[evidence["attempt"]["seed_id"]],
            ),
        )
        loser = min(
            (evidence for evidence in successes if evidence is not winner),
            key=lambda evidence: (
                evidence["audit"]["action_score"]
                + evidence["audit"]["action_uncertainty"],
                seed_rank[evidence["attempt"]["seed_id"]],
            ),
        )
        selected = {"winner": winner, "loser": loser}
        for role, evidence in selected.items():
            endpoint = pair[role]
            audit = evidence["audit"]
            attempt = evidence["attempt"]
            if endpoint["candidate_id"] != attempt["candidate_id"]:
                raise CAPERAdmissionError(
                    "pair endpoints violate fixed complete-population selection rule"
                )
            ordering = pair["action_ordering"]
            if (
                ordering["rubric_id"] != audit["rubric_id"]
                or ordering[f"{role}_action_grade"] != audit["action_grade"]
                or abs(ordering[f"{role}_action_score"] - audit["action_score"])
                > 1.0e-9
                or abs(
                    ordering[f"{role}_action_uncertainty"]
                    - audit["action_uncertainty"]
                )
                > 1.0e-9
            ):
                raise CAPERAdmissionError(
                    "fixed-rule endpoint audit differs from pair action ordering"
                )

    for split in SPLITS:
        for pair in pairs[split]:
            for role in ("winner", "loser"):
                candidate = pair[role]
                rollout_id = candidate["candidate_id"]
                entry = entries[rollout_id]
                decoded_media = _decode_exact81_media_artifact(
                    candidate["output_media"],
                    base_dir=base_dir,
                    label=f"candidate {rollout_id} output",
                )
                decoded_media_by_rollout[rollout_id] = decoded_media
                receipt = _validate_rollout_receipt(
                    _load_canonical_json_artifact(
                        candidate["rollout_receipt"],
                        base_dir=base_dir,
                        label=f"candidate {rollout_id} rollout receipt",
                    ),
                    expected_rollout_id=rollout_id,
                    expected_role=SOURCE_CANDIDATE_ROLE,
                    expected_split=split,
                    expected_policy_binding=candidate["policy_binding"],
                    expected_population_id=pair["population_id"],
                    expected_seed_id=candidate["seed_id"],
                    expected_source_id=pair["source_id"],
                    expected_source_media_sha256=pair["source_media"]["sha256"],
                    expected_action_family=pair["action_family"],
                    expected_instruction_text=pair["edit_instruction"],
                    expected_output_media=candidate["output_media"],
                    expected_decoded_media=decoded_media,
                    label=f"candidate {rollout_id} rollout receipt",
                )
                population_attempt = next(
                    attempt
                    for attempt in populations[pair["population_id"]]["attempts"]
                    if attempt["candidate_id"] == rollout_id
                )
                if population_attempt["source_role_digest"] != receipt["source_role_digest"]:
                    raise CAPERAdmissionError(
                        "population attempt source-role digest differs from receipt"
                    )
                expected_entry = {
                    "rollout_id": rollout_id,
                    "lineage_root_sha256": receipt["lineage_root_sha256"],
                    "rollout_role": SOURCE_CANDIDATE_ROLE,
                    "split": split,
                    "population_id": pair["population_id"],
                    "seed_id": candidate["seed_id"],
                    "policy_id": candidate["policy_binding"]["policy_id"],
                    "checkpoint_tree_sha256": candidate["policy_binding"][
                        "checkpoint_tree_sha256"
                    ],
                    "inference_contract_sha256": candidate["policy_binding"][
                        "inference_contract_sha256"
                    ],
                    "source_id": pair["source_id"],
                    "source_media_sha256": pair["source_media"]["sha256"],
                    "action_family": pair["action_family"],
                    "instruction_text_sha256": text_sha256(pair["edit_instruction"]),
                    "output_media_sha256": candidate["output_media"]["sha256"],
                    "rollout_receipt_artifact_sha256": candidate["rollout_receipt"][
                        "sha256"
                    ],
                    "reward_audit_artifact_sha256": candidate["reward_audit"]["sha256"],
                    "source_role_digest": receipt["source_role_digest"],
                }
                _entry_must_equal(entry, expected_entry, label=f"ledger entry {rollout_id}")
                _validate_reward_audit(
                    _load_canonical_json_artifact(
                        candidate["reward_audit"],
                        base_dir=base_dir,
                        label=f"candidate {rollout_id} reward audit",
                    ),
                    expected_rollout_id=rollout_id,
                    expected_lineage_root_sha256=receipt["lineage_root_sha256"],
                    expected_source_role_digest=receipt["source_role_digest"],
                    expected_output_media_sha256=candidate["output_media"]["sha256"],
                    expected_rollout_receipt_artifact_sha256=candidate[
                        "rollout_receipt"
                    ]["sha256"],
                    expected_audit_role=SOURCE_AUDIT_ROLE,
                    expected_checkpoint_tree_sha256=policy["checkpoint_tree_sha256"],
                    expected_action_family=pair["action_family"],
                    expected_audit_protocol_sha256=audit_protocol,
                    expected_evaluator=evaluator,
                    expected_rubric_id=pair["action_ordering"]["rubric_id"],
                    expected_action_grade=pair["action_ordering"][f"{role}_action_grade"],
                    expected_hard_gates=candidate["preservation_hard_gates"],
                    expected_action_score=pair["action_ordering"][
                        f"{role}_action_score"
                    ],
                    expected_action_uncertainty=pair["action_ordering"][
                        f"{role}_action_uncertainty"
                    ],
                    label=f"candidate {rollout_id} reward audit",
                )

    for owner in owners:
        rollout_id = owner["owner_id"]
        entry = entries[rollout_id]
        generator = {
            "policy_id": owner["generator_binding"]["generator_id"],
            "checkpoint_tree_sha256": owner["generator_binding"][
                "checkpoint_tree_sha256"
            ],
            "inference_contract_sha256": owner["generator_binding"][
                "inference_contract_sha256"
            ],
        }
        if generator["checkpoint_tree_sha256"] != policy["checkpoint_tree_sha256"]:
            raise CAPERAdmissionError("pure-T2V owner checkpoint differs from policy")
        decoded_media = _decode_exact81_media_artifact(
            owner["output_media"],
            base_dir=base_dir,
            label=f"owner {rollout_id} output",
        )
        decoded_media_by_rollout[rollout_id] = decoded_media
        receipt = _validate_rollout_receipt(
            _load_canonical_json_artifact(
                owner["rollout_receipt"],
                base_dir=base_dir,
                label=f"owner {rollout_id} rollout receipt",
            ),
            expected_rollout_id=rollout_id,
            expected_role=PURE_T2V_LEDGER_ROLE,
            expected_split=owner["split"],
            expected_policy_binding=generator,
            expected_population_id=None,
            expected_seed_id=None,
            expected_source_id=None,
            expected_source_media_sha256=None,
            expected_action_family=owner["action_family"],
            expected_instruction_text=owner["prompt"],
            expected_output_media=owner["output_media"],
            expected_decoded_media=decoded_media,
            label=f"owner {rollout_id} rollout receipt",
        )
        _entry_must_equal(
            entry,
            {
                "rollout_id": rollout_id,
                "lineage_root_sha256": receipt["lineage_root_sha256"],
                "rollout_role": PURE_T2V_LEDGER_ROLE,
                "split": owner["split"],
                "population_id": None,
                "seed_id": None,
                "policy_id": generator["policy_id"],
                "checkpoint_tree_sha256": generator["checkpoint_tree_sha256"],
                "inference_contract_sha256": generator["inference_contract_sha256"],
                "source_id": None,
                "source_media_sha256": None,
                "action_family": owner["action_family"],
                "instruction_text_sha256": text_sha256(owner["prompt"]),
                "output_media_sha256": owner["output_media"]["sha256"],
                "rollout_receipt_artifact_sha256": owner["rollout_receipt"]["sha256"],
                "reward_audit_artifact_sha256": owner["reward_audit"]["sha256"],
                "source_role_digest": receipt["source_role_digest"],
            },
            label=f"ledger entry {rollout_id}",
        )
        _validate_reward_audit(
            _load_canonical_json_artifact(
                owner["reward_audit"],
                base_dir=base_dir,
                label=f"owner {rollout_id} reward audit",
            ),
            expected_rollout_id=rollout_id,
            expected_lineage_root_sha256=receipt["lineage_root_sha256"],
            expected_source_role_digest=receipt["source_role_digest"],
            expected_output_media_sha256=owner["output_media"]["sha256"],
            expected_rollout_receipt_artifact_sha256=owner["rollout_receipt"]["sha256"],
            expected_audit_role=PURE_T2V_AUDIT_ROLE,
            expected_checkpoint_tree_sha256=policy["checkpoint_tree_sha256"],
            expected_action_family=owner["action_family"],
            expected_audit_protocol_sha256=audit_protocol,
            expected_evaluator=evaluator,
            expected_rubric_id=None,
            expected_action_grade=None,
            expected_hard_gates=None,
            expected_action_score=None,
            expected_action_uncertainty=None,
            label=f"owner {rollout_id} reward audit",
        )

    return {
        "policy_id": policy["policy_id"],
        "checkpoint_tree_sha256": policy["checkpoint_tree_sha256"],
        "inference_contract_sha256": policy["inference_contract_sha256"],
        "source_revision_sha256": ledger["source_revision_sha256"],
        "action_taxonomy_sha256": ledger["action_taxonomy_sha256"],
        "reward_audit_protocol_sha256": audit_protocol,
        "exposure_ledger_artifact_sha256": ledger_artifact["sha256"],
        "exposure_ledger_payload_sha256": ledger["ledger_sha256"],
        "rollout_population_ledger_artifact_sha256": population_ledger_artifact[
            "sha256"
        ],
        "rollout_population_ledger_payload_sha256": population_ledger[
            "ledger_sha256"
        ],
        "reward_evaluator_weights_sha256": evaluator["weights_artifact"]["sha256"],
        "reward_evaluator_runtime_sha256": evaluator["runtime_artifact"]["sha256"],
        "reward_evaluator_input_contract_sha256": evaluator[
            "input_contract_sha256"
        ],
        "verified_decoded_media_by_rollout": decoded_media_by_rollout,
    }


def _policy(value: Any) -> dict[str, Any]:
    row = _closed(value, _POLICY_FIELDS, "deployable_policy")
    policy = {
        "policy_id": _identifier(row["policy_id"], "deployable_policy.policy_id"),
        "model_family": _nonempty_text(
            row["model_family"], "deployable_policy.model_family"
        ),
        "checkpoint_tree_sha256": _sha256(
            row["checkpoint_tree_sha256"],
            "deployable_policy.checkpoint_tree_sha256",
        ),
        "inference_contract_sha256": _sha256(
            row["inference_contract_sha256"],
            "deployable_policy.inference_contract_sha256",
        ),
        "deployable": row["deployable"],
        "source_conditioned": row["source_conditioned"],
        "weights_frozen_during_rollout": row["weights_frozen_during_rollout"],
    }
    _exact_bool(policy["deployable"], True, "deployable_policy.deployable")
    _exact_bool(
        policy["source_conditioned"], True, "deployable_policy.source_conditioned"
    )
    _exact_bool(
        policy["weights_frozen_during_rollout"],
        True,
        "deployable_policy.weights_frozen_during_rollout",
    )
    return policy


def _policy_binding(
    value: Any, *, policy: Mapping[str, Any], label: str
) -> dict[str, str]:
    row = _closed(value, _POLICY_BINDING_FIELDS, label)
    binding = {
        "policy_id": _identifier(row["policy_id"], f"{label}.policy_id"),
        "checkpoint_tree_sha256": _sha256(
            row["checkpoint_tree_sha256"], f"{label}.checkpoint_tree_sha256"
        ),
        "inference_contract_sha256": _sha256(
            row["inference_contract_sha256"],
            f"{label}.inference_contract_sha256",
        ),
    }
    expected = {key: policy[key] for key in _POLICY_BINDING_FIELDS}
    if binding != expected:
        raise CAPERAdmissionError(
            f"{label} is not the sealed deployable policy/checkpoint"
        )
    return binding


def _hard_gates(value: Any, label: str) -> dict[str, bool]:
    row = _closed(value, frozenset(PRESERVATION_GATES), label)
    result: dict[str, bool] = {}
    for gate in PRESERVATION_GATES:
        if type(row[gate]) is not bool:
            raise CAPERAdmissionError(f"{label}.{gate} must be boolean")
        result[gate] = row[gate]
    return result


def _conditioning(
    value: Any, *, source_media_sha256: str, label: str
) -> dict[str, Any]:
    row = _closed(value, _CONDITIONING_FIELDS, label)
    input_kinds = tuple(_list(row["input_kinds"], f"{label}.input_kinds"))
    if input_kinds != SOURCE_CONDITIONING_INPUTS:
        raise CAPERAdmissionError(
            f"{label}.input_kinds must be the deployment input tuple"
        )
    digest = _sha256(row["source_media_sha256"], f"{label}.source_media_sha256")
    if digest != source_media_sha256:
        raise CAPERAdmissionError(f"{label} is bound to a different source media")
    for key in (
        "target_video_read",
        "target_video_latent_read",
        "paired_target_read",
        "pure_t2v_media_read",
        "pure_t2v_latent_read",
    ):
        _exact_bool(row[key], False, f"{label}.{key}")
    return {
        "input_kinds": list(input_kinds),
        "source_media_sha256": digest,
        "target_video_read": False,
        "target_video_latent_read": False,
        "paired_target_read": False,
        "pure_t2v_media_read": False,
        "pure_t2v_latent_read": False,
    }


def _candidate(
    value: Any,
    *,
    role: str,
    pair_context: Mapping[str, str],
    source_media_sha256: str,
    policy: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    row = _closed(value, _CANDIDATE_FIELDS, label)
    if row["declared_role"] != role:
        raise CAPERAdmissionError(f"{label}.declared_role must be {role!r}")
    if row["generation_mode"] != "source_conditioned_video_editing":
        raise CAPERAdmissionError(
            f"{label} must be a source-conditioned deployable-policy rollout; "
            "pure T2V can only be a calibration owner"
        )
    _exact_bool(
        row["eligible_as_preference_target"],
        True,
        f"{label}.eligible_as_preference_target",
    )

    candidate: dict[str, Any] = {
        "candidate_id": _identifier(row["candidate_id"], f"{label}.candidate_id"),
        "population_id": _identifier(
            row["population_id"], f"{label}.population_id"
        ),
        "seed_id": _identifier(row["seed_id"], f"{label}.seed_id"),
        "declared_role": role,
        "generation_mode": "source_conditioned_video_editing",
        "eligible_as_preference_target": True,
    }
    for key in (
        "source_id",
        "identity_id",
        "scene_id",
        "action_family",
        "edit_instruction",
    ):
        observed = (
            _nonempty_text(row[key], f"{label}.{key}")
            if key == "edit_instruction"
            else _identifier(row[key], f"{label}.{key}")
        )
        if observed != pair_context[key]:
            raise CAPERAdmissionError(f"{label}.{key} differs from its pair")
        candidate[key] = observed
    if candidate["population_id"] != pair_context["population_id"]:
        raise CAPERAdmissionError(f"{label}.population_id differs from its pair")

    candidate["policy_binding"] = _policy_binding(
        row["policy_binding"], policy=policy, label=f"{label}.policy_binding"
    )
    candidate["conditioning_attestation"] = _conditioning(
        row["conditioning_attestation"],
        source_media_sha256=source_media_sha256,
        label=f"{label}.conditioning_attestation",
    )
    for key in ("output_media", "rollout_receipt", "reward_audit"):
        candidate[key] = _artifact(row[key], f"{label}.{key}")
    candidate["preservation_hard_gates"] = _hard_gates(
        row["preservation_hard_gates"],
        f"{label}.preservation_hard_gates",
    )
    return candidate


def _action_ordering(value: Any, label: str) -> dict[str, Any]:
    row = _closed(value, _ACTION_ORDERING_FIELDS, label)
    result = {
        "rubric_id": _identifier(row["rubric_id"], f"{label}.rubric_id"),
        "winner_action_grade": row["winner_action_grade"],
        "loser_action_grade": row["loser_action_grade"],
        "strict_preference_claimed": row["strict_preference_claimed"],
        "selection_basis": row["selection_basis"],
        "pair_selection_rule": row["pair_selection_rule"],
        "winner_action_score": _finite_unit(
            row["winner_action_score"], f"{label}.winner_action_score"
        ),
        "winner_action_uncertainty": _finite_unit(
            row["winner_action_uncertainty"],
            f"{label}.winner_action_uncertainty",
        ),
        "loser_action_score": _finite_unit(
            row["loser_action_score"], f"{label}.loser_action_score"
        ),
        "loser_action_uncertainty": _finite_unit(
            row["loser_action_uncertainty"], f"{label}.loser_action_uncertainty"
        ),
        "pairwise_confidence_margin": row["pairwise_confidence_margin"],
        "minimum_required_pairwise_margin": _finite_unit(
            row["minimum_required_pairwise_margin"],
            f"{label}.minimum_required_pairwise_margin",
        ),
    }
    for key in ("winner_action_grade", "loser_action_grade"):
        if result[key] not in ACTION_GRADES:
            raise CAPERAdmissionError(f"{label}.{key} is not a closed action grade")
    _exact_bool(
        result["strict_preference_claimed"],
        True,
        f"{label}.strict_preference_claimed",
    )
    if result["selection_basis"] != ACTION_SELECTION_BASIS:
        raise CAPERAdmissionError(f"{label}.selection_basis differs")
    if result["pair_selection_rule"] != PAIR_SELECTION_RULE:
        raise CAPERAdmissionError(f"{label}.pair_selection_rule differs")
    margin = (
        result["winner_action_score"]
        - result["winner_action_uncertainty"]
        - result["loser_action_score"]
        - result["loser_action_uncertainty"]
    )
    observed_margin = row["pairwise_confidence_margin"]
    if (
        isinstance(observed_margin, bool)
        or not isinstance(observed_margin, (int, float))
        or abs(float(observed_margin) - margin) > 1.0e-9
        or margin < result["minimum_required_pairwise_margin"]
    ):
        raise CAPERAdmissionError(f"{label} pairwise margin/uncertainty differs")
    result["pairwise_confidence_margin"] = float(observed_margin)
    return result


def _owner(value: Any, *, label: str) -> dict[str, Any]:
    row = _closed(value, _OWNER_FIELDS, label)
    if row["split"] not in SPLITS:
        raise CAPERAdmissionError(f"{label}.split differs")
    if row["role"] != PURE_T2V_ROLE:
        raise CAPERAdmissionError(f"{label}.role differs")
    if row["generation_mode"] != "pure_t2v":
        raise CAPERAdmissionError(f"{label}.generation_mode must be pure_t2v")
    _exact_bool(
        row["eligible_as_preference_target"],
        False,
        f"{label}.eligible_as_preference_target",
    )
    _exact_bool(
        row["eligible_as_training_target"],
        False,
        f"{label}.eligible_as_training_target",
    )
    _exact_bool(
        row["target_video_dependency"],
        False,
        f"{label}.target_video_dependency",
    )
    generator = _closed(
        row["generator_binding"],
        _GENERATOR_BINDING_FIELDS,
        f"{label}.generator_binding",
    )
    owner: dict[str, Any] = {
        "owner_id": _identifier(row["owner_id"], f"{label}.owner_id"),
        "split": row["split"],
        "role": PURE_T2V_ROLE,
        "generation_mode": "pure_t2v",
        "action_family": _identifier(
            row["action_family"], f"{label}.action_family"
        ),
        "prompt": _nonempty_text(row["prompt"], f"{label}.prompt"),
        "generator_binding": {
            "generator_id": _identifier(
                generator["generator_id"], f"{label}.generator_binding.generator_id"
            ),
            "checkpoint_tree_sha256": _sha256(
                generator["checkpoint_tree_sha256"],
                f"{label}.generator_binding.checkpoint_tree_sha256",
            ),
            "inference_contract_sha256": _sha256(
                generator["inference_contract_sha256"],
                f"{label}.generator_binding.inference_contract_sha256",
            ),
        },
    }
    for key in ("output_media", "rollout_receipt", "reward_audit"):
        owner[key] = _artifact(row[key], f"{label}.{key}")
    owner.update(
        {
            "eligible_as_preference_target": False,
            "eligible_as_training_target": False,
            "target_video_dependency": False,
        }
    )
    return owner


def _contract(value: Any) -> dict[str, Any]:
    row = _closed(value, _CONTRACT_FIELDS, "admission_contract")
    exact_values = {
        "stage": "CAPER-stage1",
        "preference_target_mode": PREFERENCE_TARGET_MODE,
        "pure_t2v_role": PURE_T2V_ROLE,
        "preservation_gate_semantics": PRESERVATION_SEMANTICS,
        "no_valid_fit_pair_behavior": NO_PAIR_BEHAVIOR,
    }
    for key, expected in exact_values.items():
        if row[key] != expected:
            raise CAPERAdmissionError(f"admission_contract.{key} differs")
    for key in (
        "pure_t2v_preference_target_allowed",
        "target_video_dependency_allowed",
        "scalar_reward_admission_allowed",
        "scalar_compensation_allowed",
    ):
        _exact_bool(row[key], False, f"admission_contract.{key}")
    gate_names = tuple(
        _list(row["preservation_gate_names"], "preservation_gate_names")
    )
    if gate_names != PRESERVATION_GATES:
        raise CAPERAdmissionError("admission_contract preservation gate order differs")
    if tuple(_list(row["split_isolation_axes"], "split_isolation_axes")) != SPLIT_ISOLATION_AXES:
        raise CAPERAdmissionError("admission_contract split isolation axes differ")
    return {
        **exact_values,
        "pure_t2v_preference_target_allowed": False,
        "target_video_dependency_allowed": False,
        "scalar_reward_admission_allowed": False,
        "scalar_compensation_allowed": False,
        "preservation_gate_names": list(PRESERVATION_GATES),
        "split_isolation_axes": list(SPLIT_ISOLATION_AXES),
    }


def _pair(
    value: Any,
    *,
    split: str,
    policy: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    row = _closed(value, _PAIR_FIELDS, label)
    if row["split"] != split:
        raise CAPERAdmissionError(f"{label}.split must match its split container")
    _exact_bool(
        row["target_video_dependency"], False, f"{label}.target_video_dependency"
    )
    _exact_bool(
        row["scalar_reward_used_for_admission"],
        False,
        f"{label}.scalar_reward_used_for_admission",
    )
    _exact_bool(
        row["scalar_compensation_used"],
        False,
        f"{label}.scalar_compensation_used",
    )
    pair: dict[str, Any] = {
        "pair_id": _identifier(row["pair_id"], f"{label}.pair_id"),
        "population_id": _identifier(
            row["population_id"], f"{label}.population_id"
        ),
        "split": split,
        "source_id": _identifier(row["source_id"], f"{label}.source_id"),
        "identity_id": _identifier(row["identity_id"], f"{label}.identity_id"),
        "scene_id": _identifier(row["scene_id"], f"{label}.scene_id"),
        "action_family": _identifier(
            row["action_family"], f"{label}.action_family"
        ),
        "edit_instruction": _nonempty_text(
            row["edit_instruction"], f"{label}.edit_instruction"
        ),
        "target_video_dependency": False,
        "scalar_reward_used_for_admission": False,
        "scalar_compensation_used": False,
    }
    pair["source_media"] = _artifact(row["source_media"], f"{label}.source_media")
    pair["source_intake_receipt"] = _artifact(
        row["source_intake_receipt"], f"{label}.source_intake_receipt"
    )
    context = {
        key: pair[key]
        for key in (
            "source_id",
            "identity_id",
            "scene_id",
            "action_family",
            "edit_instruction",
            "population_id",
        )
    }
    pair["winner"] = _candidate(
        row["winner"],
        role="winner",
        pair_context=context,
        source_media_sha256=pair["source_media"]["sha256"],
        policy=policy,
        label=f"{label}.winner",
    )
    pair["loser"] = _candidate(
        row["loser"],
        role="loser",
        pair_context=context,
        source_media_sha256=pair["source_media"]["sha256"],
        policy=policy,
        label=f"{label}.loser",
    )
    pair["action_ordering"] = _action_ordering(
        row["action_ordering"], f"{label}.action_ordering"
    )
    return pair


def _ensure_unique(values: Sequence[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise CAPERAdmissionError(f"duplicate {label}: {sorted(duplicates)}")


def _check_pair_endpoint_separation(pair: Mapping[str, Any]) -> None:
    winner = pair["winner"]
    loser = pair["loser"]
    if winner["candidate_id"] == loser["candidate_id"]:
        raise CAPERAdmissionError(f"{pair['pair_id']} endpoints share candidate_id")
    for artifact_name in ("output_media", "rollout_receipt", "reward_audit"):
        left = winner[artifact_name]
        right = loser[artifact_name]
        if left["path"] == right["path"] or left["sha256"] == right["sha256"]:
            raise CAPERAdmissionError(
                f"{pair['pair_id']} endpoints alias {artifact_name}"
            )
    source_sha = pair["source_media"]["sha256"]
    if source_sha in (
        winner["output_media"]["sha256"],
        loser["output_media"]["sha256"],
    ):
        raise CAPERAdmissionError(
            f"{pair['pair_id']} candidate output aliases source media"
        )


def _check_split_isolation(pairs: Mapping[str, list[dict[str, Any]]]) -> None:
    for axis in SPLIT_ISOLATION_AXES:
        if axis == "source_media_sha256":
            fit = {pair["source_media"]["sha256"] for pair in pairs["fit"]}
            heldout = {
                pair["source_media"]["sha256"] for pair in pairs["heldout"]
            }
        else:
            fit = {pair[axis] for pair in pairs["fit"]}
            heldout = {pair[axis] for pair in pairs["heldout"]}
        overlap = fit & heldout
        if overlap:
            raise CAPERAdmissionError(
                f"fit/heldout {axis} leakage: {sorted(overlap)}"
            )


def _check_source_registry(pairs: Mapping[str, list[dict[str, Any]]]) -> None:
    registry: dict[str, tuple[str, str, str, str, str]] = {}
    media_owner: dict[str, str] = {}
    for split in SPLITS:
        for pair in pairs[split]:
            signature = (
                split,
                pair["identity_id"],
                pair["scene_id"],
                pair["source_media"]["sha256"],
                pair["source_intake_receipt"]["sha256"],
            )
            previous = registry.setdefault(pair["source_id"], signature)
            if previous != signature:
                raise CAPERAdmissionError(
                    f"source_id {pair['source_id']!r} has inconsistent provenance"
                )
            digest = pair["source_media"]["sha256"]
            previous_source = media_owner.setdefault(digest, pair["source_id"])
            if previous_source != pair["source_id"]:
                raise CAPERAdmissionError(
                    "one source-media digest is assigned to multiple source IDs"
                )


def _iter_bound_artifacts(
    pairs: Mapping[str, list[dict[str, Any]]],
    owners: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    source_seen: set[tuple[str, str, int]] = set()
    for split in SPLITS:
        for pair in pairs[split]:
            for key in ("source_media", "source_intake_receipt"):
                artifact = pair[key]
                signature = (
                    artifact["path"],
                    artifact["sha256"],
                    artifact["size_bytes"],
                )
                if signature not in source_seen:
                    artifacts.append(artifact)
                    source_seen.add(signature)
            for role in ("winner", "loser"):
                for key in ("output_media", "rollout_receipt", "reward_audit"):
                    artifacts.append(pair[role][key])
    for owner in owners:
        for key in ("output_media", "rollout_receipt", "reward_audit"):
            artifacts.append(owner[key])
    return artifacts


def _check_artifact_path_consistency(artifacts: Sequence[Mapping[str, Any]]) -> None:
    by_path: dict[str, tuple[str, int]] = {}
    for artifact in artifacts:
        binding = (artifact["sha256"], artifact["size_bytes"])
        previous = by_path.setdefault(artifact["path"], binding)
        if previous != binding:
            raise CAPERAdmissionError(
                f"artifact path has conflicting commitments: {artifact['path']}"
            )


def _check_artifact_role_separation(
    pairs: Mapping[str, list[dict[str, Any]]],
    owners: Sequence[Mapping[str, Any]],
) -> None:
    """Forbid media/receipt/audit laundering through a second artifact role."""

    source_paths: dict[str, str] = {}
    source_digests: dict[str, str] = {}
    endpoint_paths: dict[str, str] = {}
    endpoint_digests: dict[str, str] = {}

    def register(
        artifact: Mapping[str, Any],
        *,
        label: str,
        paths: dict[str, str],
        digests: dict[str, str],
        allow_same_label: bool,
    ) -> None:
        previous_path = paths.setdefault(artifact["path"], label)
        previous_digest = digests.setdefault(artifact["sha256"], label)
        if (not allow_same_label and previous_path != label) or (
            not allow_same_label and previous_digest != label
        ):
            raise CAPERAdmissionError(
                f"artifact aliases two security roles: {previous_path!r}/"
                f"{previous_digest!r} and {label!r}"
            )

    # Repeated pairs for the same source may intentionally share the sealed
    # source bytes and source-intake receipt, hence the stable source-specific
    # labels.  Media and receipt still cannot alias each other.
    for split in SPLITS:
        for pair in pairs[split]:
            for key in ("source_media", "source_intake_receipt"):
                register(
                    pair[key],
                    label=f"source:{pair['source_id']}:{key}",
                    paths=source_paths,
                    digests=source_digests,
                    allow_same_label=False,
                )

    # Every generated media, rollout receipt, and reward audit is
    # candidate/owner-specific.  Byte-identical copies under new paths are
    # rejected just like path aliases.
    for split in SPLITS:
        for pair in pairs[split]:
            for role in ("winner", "loser"):
                candidate = pair[role]
                for key in ("output_media", "rollout_receipt", "reward_audit"):
                    register(
                        candidate[key],
                        label=f"candidate:{candidate['candidate_id']}:{key}",
                        paths=endpoint_paths,
                        digests=endpoint_digests,
                        allow_same_label=False,
                    )
    for owner in owners:
        for key in ("output_media", "rollout_receipt", "reward_audit"):
            register(
                owner[key],
                label=f"t2v-owner:{owner['owner_id']}:{key}",
                paths=endpoint_paths,
                digests=endpoint_digests,
                allow_same_label=False,
            )

    path_overlap = set(source_paths) & set(endpoint_paths)
    digest_overlap = set(source_digests) & set(endpoint_digests)
    if path_overlap or digest_overlap:
        raise CAPERAdmissionError(
            "source media/receipt aliases generated media, rollout receipt, "
            "or reward audit"
        )


def _check_t2v_separation(
    owners: Sequence[Mapping[str, Any]],
    pairs: Mapping[str, list[dict[str, Any]]],
) -> None:
    owner_ids = {owner["owner_id"] for owner in owners}
    candidate_ids: set[str] = set()
    owner_commitments: set[tuple[str, str]] = set()
    for owner in owners:
        for key in ("output_media", "rollout_receipt", "reward_audit"):
            owner_commitments.add((owner[key]["path"], owner[key]["sha256"]))

    for split in SPLITS:
        for pair in pairs[split]:
            for role in ("winner", "loser"):
                candidate = pair[role]
                candidate_ids.add(candidate["candidate_id"])
                for key in ("output_media", "rollout_receipt", "reward_audit"):
                    artifact = candidate[key]
                    if any(
                        path == artifact["path"] or digest == artifact["sha256"]
                        for path, digest in owner_commitments
                    ):
                        raise CAPERAdmissionError(
                            "pure-T2V calibration artifact aliases a preference endpoint"
                        )
    overlap = owner_ids & candidate_ids
    if overlap:
        raise CAPERAdmissionError(
            f"pure-T2V owner IDs alias preference candidate IDs: {sorted(overlap)}"
        )


def validate_manifest(
    payload: Any,
    *,
    verify_files: bool = False,
    base_dir: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Validate and normalize a closed candidate manifest.

    A returned manifest is structurally trustworthy but can still contain
    pairs that fail a preservation gate or lack a strict action ordering.
    Those are scientific rejections handled by :func:`materialize_admission`.
    """

    root = _closed(payload, _ROOT_FIELDS, "manifest")
    if root["schema_version"] != SCHEMA_VERSION:
        raise CAPERAdmissionError("manifest.schema_version differs")
    manifest_id = _identifier(root["manifest_id"], "manifest.manifest_id")
    created_utc = root["created_utc"]
    if not isinstance(created_utc, str) or _UTC_RE.fullmatch(created_utc) is None:
        raise CAPERAdmissionError("manifest.created_utc must be UTC second precision")
    purpose = _nonempty_text(root["purpose"], "manifest.purpose")
    contract = _contract(root["admission_contract"])
    policy = _policy(root["deployable_policy"])
    evaluator = _reward_evaluator(root["reward_evaluator"])
    population_ledger = _artifact(
        root["rollout_population_ledger"], "rollout_population_ledger"
    )
    exposure_ledger = _artifact(root["exposure_ledger"], "exposure_ledger")

    owners = [
        _owner(value, label=f"pure_t2v_calibration_owners[{index}]")
        for index, value in enumerate(
            _list(root["pure_t2v_calibration_owners"], "pure_t2v_calibration_owners")
        )
    ]
    _ensure_unique([owner["owner_id"] for owner in owners], "owner_id")

    split_rows = _closed(root["splits"], _SPLITS_FIELDS, "splits")
    pairs: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        pairs[split] = [
            _pair(
                value,
                split=split,
                policy=policy,
                label=f"splits.{split}[{index}]",
            )
            for index, value in enumerate(_list(split_rows[split], f"splits.{split}"))
        ]

    all_pairs = [pair for split in SPLITS for pair in pairs[split]]
    _ensure_unique([pair["pair_id"] for pair in all_pairs], "pair_id")
    _ensure_unique(
        [pair[role]["candidate_id"] for pair in all_pairs for role in ("winner", "loser")],
        "candidate_id",
    )
    for pair in all_pairs:
        _check_pair_endpoint_separation(pair)
    _check_source_registry(pairs)
    _check_split_isolation(pairs)
    _check_t2v_separation(owners, pairs)

    evidence_roots = [
        exposure_ledger,
        population_ledger,
        evaluator["weights_artifact"],
        evaluator["runtime_artifact"],
    ]
    artifacts = [*evidence_roots, *_iter_bound_artifacts(pairs, owners)]
    _check_artifact_path_consistency(artifacts)
    _check_artifact_role_separation(pairs, owners)
    for index, root_artifact in enumerate(evidence_roots):
        for artifact in artifacts[index + 1 :]:
            if (
                artifact["path"] == root_artifact["path"]
                or artifact["sha256"] == root_artifact["sha256"]
            ):
                raise CAPERAdmissionError(
                    "ledger/evaluator artifact aliases another security role"
                )
    semantic_bindings = None
    decoded_media_by_rollout = None
    if verify_files:
        verified_base = None if base_dir is None else Path(base_dir)
        for artifact in artifacts:
            _verify_artifact(artifact, base_dir=verified_base)
        semantic_evidence = _validate_semantic_evidence(
            ledger_artifact=exposure_ledger,
            population_ledger_artifact=population_ledger,
            evaluator=evaluator,
            policy=policy,
            pairs=pairs,
            owners=owners,
            base_dir=verified_base,
        )
        decoded_media_by_rollout = semantic_evidence.pop(
            "verified_decoded_media_by_rollout"
        )
        semantic_bindings = semantic_evidence

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "created_utc": created_utc,
        "purpose": purpose,
        "admission_contract": contract,
        "deployable_policy": policy,
        "reward_evaluator": evaluator,
        "rollout_population_ledger": population_ledger,
        "exposure_ledger": exposure_ledger,
        "semantic_evidence_verified": verify_files,
        "semantic_bindings": semantic_bindings,
        "verified_decoded_media_by_rollout": decoded_media_by_rollout,
        "pure_t2v_calibration_owners": owners,
        "splits": pairs,
    }


def _pair_rejection_reasons(pair: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for role in ("winner", "loser"):
        failed = [
            gate
            for gate in PRESERVATION_GATES
            if pair[role]["preservation_hard_gates"][gate] is not True
        ]
        reasons.extend(f"{role}_preservation_gate_failed:{gate}" for gate in failed)
    ordering = pair["action_ordering"]
    if ACTION_GRADE_ORDER[ordering["winner_action_grade"]] <= ACTION_GRADE_ORDER[
        ordering["loser_action_grade"]
    ]:
        reasons.append("winner_action_not_strictly_better")
    return reasons


def _materialized_pair(
    pair: Mapping[str, Any],
    *,
    decoded_media_by_rollout: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if pair["split"] != "fit":
        raise CAPERAdmissionError("only fit pairs may be materialized as optimizer pairs")
    # Keep the complete normalized fit row.  This allows the later verifier to
    # rerun the closed same-source/on-policy contract without access to the
    # held-out manifest or any unmaterialized endpoint.
    result = {key: pair[key] for key in _PAIR_FIELDS}
    result["optimizer_target_allowed"] = True
    result["decoded_endpoint_media"] = {
        role: dict(decoded_media_by_rollout[pair[role]["candidate_id"]])
        for role in ("winner", "loser")
    }
    result["pair_sha256"] = object_sha256(
        {key: result[key] for key in _PAIR_FIELDS}
    )
    result["optimizer_pair_sha256"] = object_sha256(
        {
            **{key: result[key] for key in _PAIR_FIELDS},
            "optimizer_target_allowed": True,
            "decoded_endpoint_media": result["decoded_endpoint_media"],
        }
    )
    return result


def _heldout_audit_record(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Return held-out evidence metadata with no readable artifact path."""

    if pair["split"] != "heldout":
        raise CAPERAdmissionError("heldout audit record received a non-heldout pair")
    return {
        "pair_id": pair["pair_id"],
        "split": "heldout",
        "source_id": pair["source_id"],
        "identity_id": pair["identity_id"],
        "scene_id": pair["scene_id"],
        "action_family": pair["action_family"],
        "candidate_ids": [pair["winner"]["candidate_id"], pair["loser"]["candidate_id"]],
        "source_media_sha256": pair["source_media"]["sha256"],
        "winner_output_media_sha256": pair["winner"]["output_media"]["sha256"],
        "loser_output_media_sha256": pair["loser"]["output_media"]["sha256"],
        "winner_action_grade": pair["action_ordering"]["winner_action_grade"],
        "loser_action_grade": pair["action_ordering"]["loser_action_grade"],
        "optimizer_target_allowed": False,
        "pair_sha256": object_sha256(pair),
    }


def _zero_update_certificate(
    *,
    manifest_id: str,
    manifest_sha256: str,
    rejected_fit_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": ZERO_UPDATE_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "manifest_sha256": manifest_sha256,
        "reason": "no_valid_fit_same_source_on_policy_preference_pair",
        "rejected_fit_pair_ids": [row["pair_id"] for row in rejected_fit_pairs],
        "authorized_optimizer_pair_ids": [],
        "authorized_optimizer_steps": 0,
        "gradient_application_allowed": False,
        "optimizer_step_allowed": False,
        "parameter_update_required": False,
    }
    body["certificate_sha256"] = object_sha256(body)
    return body


def materialize_admission(
    payload: Any,
    *,
    verify_files: bool = True,
    base_dir: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Materialize admitted/rejected pairs and an optimizer authorization bit.

    ``verify_files`` defaults to true because a training index must bind real
    bytes, not merely well-formed digest strings.  Tests and dependency-light
    structural audits may set it false explicitly.
    """

    if verify_files is not True:
        raise CAPERAdmissionError(
            "training materialization requires verified artifact bytes and receipts"
        )
    manifest = validate_manifest(
        payload, verify_files=verify_files, base_dir=base_dir
    )
    manifest_sha256 = object_sha256(manifest)
    decoded_media_by_rollout = manifest["verified_decoded_media_by_rollout"]
    if not isinstance(decoded_media_by_rollout, Mapping):  # pragma: no cover
        raise CAPERAdmissionError("verified decoded-media commitments are missing")
    optimizer_pairs: list[dict[str, Any]] = []
    heldout_audit_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for split in SPLITS:
        for pair in manifest["splits"][split]:
            reasons = _pair_rejection_reasons(pair)
            if reasons:
                rejected.append(
                    {
                        "pair_id": pair["pair_id"],
                        "split": split,
                        "reasons": reasons,
                        "pair_sha256": object_sha256(pair),
                    }
                )
            elif split == "fit":
                optimizer_pairs.append(
                    _materialized_pair(
                        pair,
                        decoded_media_by_rollout=decoded_media_by_rollout,
                    )
                )
            else:
                heldout_audit_records.append(_heldout_audit_record(pair))

    optimizer_pair_ids = [pair["pair_id"] for pair in optimizer_pairs]
    optimizer_step_allowed = bool(optimizer_pair_ids)
    rejected_fit = [row for row in rejected if row["split"] == "fit"]

    calibration_owners = [
        {
            "owner_id": owner["owner_id"],
            "split": owner["split"],
            "role": PURE_T2V_ROLE,
            "generation_mode": "pure_t2v",
            "action_family": owner["action_family"],
            "generator_id": owner["generator_binding"]["generator_id"],
            "checkpoint_tree_sha256": owner["generator_binding"][
                "checkpoint_tree_sha256"
            ],
            "inference_contract_sha256": owner["generator_binding"][
                "inference_contract_sha256"
            ],
            "output_media_sha256": owner["output_media"]["sha256"],
            "rollout_receipt_artifact_sha256": owner["rollout_receipt"]["sha256"],
            "reward_audit_artifact_sha256": owner["reward_audit"]["sha256"],
            "eligible_as_preference_target": False,
            "eligible_as_training_target": False,
            "target_video_dependency": False,
        }
        for owner in manifest["pure_t2v_calibration_owners"]
    ]
    result: dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha256,
        "artifact_bytes_verified": verify_files,
        "semantic_evidence_verified": manifest["semantic_evidence_verified"],
        "bindings": {
            **manifest["semantic_bindings"],
            "authorized_fit_action_families": sorted(
                {pair["action_family"] for pair in optimizer_pairs}
            ),
            "heldout_action_families": sorted(
                {row["action_family"] for row in heldout_audit_records}
            ),
        },
        "admission_semantics": {
            "preference_target_mode": PREFERENCE_TARGET_MODE,
            "preservation_gate_semantics": PRESERVATION_SEMANTICS,
            "pure_t2v_role": PURE_T2V_ROLE,
            "scalar_compensation_allowed": False,
            "target_video_dependency_allowed": False,
            "heldout_optimizer_target_allowed": False,
        },
        "pure_t2v_calibration_owners": calibration_owners,
        "optimizer_pairs": optimizer_pairs,
        "heldout_audit_records": heldout_audit_records,
        "rejected_pairs": rejected,
        "authorized_optimizer_pair_ids": optimizer_pair_ids,
        "optimizer_step_allowed": optimizer_step_allowed,
        "zero_update_certificate": None,
    }
    if not optimizer_step_allowed:
        result["zero_update_certificate"] = _zero_update_certificate(
            manifest_id=manifest["manifest_id"],
            manifest_sha256=manifest_sha256,
            rejected_fit_pairs=rejected_fit,
        )
    result["materialization_sha256"] = object_sha256(result)
    return result


def verify_materialization_seal(value: Any) -> dict[str, Any]:
    """Verify seal, physical held-out exclusion, and optimizer target closure."""

    row = _closed(
        value,
        frozenset(
            {
                "schema_version",
                "manifest_id",
                "manifest_sha256",
                "artifact_bytes_verified",
                "semantic_evidence_verified",
                "bindings",
                "admission_semantics",
                "pure_t2v_calibration_owners",
                "optimizer_pairs",
                "heldout_audit_records",
                "rejected_pairs",
                "authorized_optimizer_pair_ids",
                "optimizer_step_allowed",
                "zero_update_certificate",
                "materialization_sha256",
            }
        ),
        "materialization",
    )
    if row.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        raise CAPERAdmissionError("materialization schema differs")
    _identifier(row["manifest_id"], "materialization.manifest_id")
    _sha256(row["manifest_sha256"], "materialization.manifest_sha256")
    _exact_bool(row["artifact_bytes_verified"], True, "artifact_bytes_verified")
    _exact_bool(
        row["semantic_evidence_verified"], True, "semantic_evidence_verified"
    )
    digest = _sha256(row.get("materialization_sha256"), "materialization_sha256")
    unsigned = dict(row)
    del unsigned["materialization_sha256"]
    if object_sha256(unsigned) != digest:
        raise CAPERAdmissionError("materialization SHA-256 differs")

    semantics = _closed(
        row["admission_semantics"],
        frozenset(
            {
                "preference_target_mode",
                "preservation_gate_semantics",
                "pure_t2v_role",
                "scalar_compensation_allowed",
                "target_video_dependency_allowed",
                "heldout_optimizer_target_allowed",
            }
        ),
        "materialization.admission_semantics",
    )
    if (
        semantics["preference_target_mode"] != PREFERENCE_TARGET_MODE
        or semantics["preservation_gate_semantics"] != PRESERVATION_SEMANTICS
        or semantics["pure_t2v_role"] != PURE_T2V_ROLE
    ):
        raise CAPERAdmissionError("materialization admission semantics differ")
    for key in (
        "scalar_compensation_allowed",
        "target_video_dependency_allowed",
        "heldout_optimizer_target_allowed",
    ):
        _exact_bool(semantics[key], False, f"materialization.admission_semantics.{key}")

    binding_fields = frozenset(
        {
            "policy_id",
            "checkpoint_tree_sha256",
            "inference_contract_sha256",
            "source_revision_sha256",
            "action_taxonomy_sha256",
            "reward_audit_protocol_sha256",
            "exposure_ledger_artifact_sha256",
            "exposure_ledger_payload_sha256",
            "rollout_population_ledger_artifact_sha256",
            "rollout_population_ledger_payload_sha256",
            "reward_evaluator_weights_sha256",
            "reward_evaluator_runtime_sha256",
            "reward_evaluator_input_contract_sha256",
            "authorized_fit_action_families",
            "heldout_action_families",
        }
    )
    bindings = _closed(row["bindings"], binding_fields, "materialization.bindings")
    _identifier(bindings["policy_id"], "materialization.bindings.policy_id")
    for key in binding_fields - {
        "policy_id",
        "authorized_fit_action_families",
        "heldout_action_families",
    }:
        _sha256(bindings[key], f"materialization.bindings.{key}")
    fit_action_families = _list(
        bindings["authorized_fit_action_families"],
        "materialization.bindings.authorized_fit_action_families",
    )
    heldout_action_families = _list(
        bindings["heldout_action_families"],
        "materialization.bindings.heldout_action_families",
    )
    for label, actions in (
        ("fit", fit_action_families),
        ("heldout", heldout_action_families),
    ):
        if actions != sorted(actions) or len(actions) != len(set(actions)):
            raise CAPERAdmissionError(f"{label} action families are not canonical")
        for action in actions:
            _identifier(action, f"materialization.bindings.{label}_action_family")
    if set(fit_action_families) & set(heldout_action_families):
        raise CAPERAdmissionError("materialized fit/heldout action families overlap")

    optimizer_pairs = _list(row["optimizer_pairs"], "optimizer_pairs")
    authorized_ids = _list(
        row["authorized_optimizer_pair_ids"], "authorized_optimizer_pair_ids"
    )
    observed_ids: list[str] = []
    for index, pair in enumerate(optimizer_pairs):
        pair_row = _closed(
            pair,
            frozenset(
                {
                    *_PAIR_FIELDS,
                    "optimizer_target_allowed",
                    "decoded_endpoint_media",
                    "pair_sha256",
                    "optimizer_pair_sha256",
                }
            ),
            f"optimizer_pairs[{index}]",
        )
        if pair_row.get("split") != "fit" or pair_row.get("optimizer_target_allowed") is not True:
            raise CAPERAdmissionError("optimizer_pairs may contain only admitted fit pairs")
        pair_payload = {key: pair_row[key] for key in _PAIR_FIELDS}
        if object_sha256(pair_payload) != _sha256(
            pair_row["pair_sha256"], f"optimizer_pairs[{index}].pair_sha256"
        ):
            raise CAPERAdmissionError("optimizer pair inner seal differs")
        decoded_rows = _closed(
            pair_row["decoded_endpoint_media"],
            frozenset({"winner", "loser"}),
            f"optimizer_pairs[{index}].decoded_endpoint_media",
        )
        normalized_decoded = {
            role: _decoded_media_commitment(
                decoded_rows[role],
                f"optimizer_pairs[{index}].decoded_endpoint_media.{role}",
            )
            for role in ("winner", "loser")
        }
        optimizer_pair_payload = {
            **pair_payload,
            "optimizer_target_allowed": True,
            "decoded_endpoint_media": normalized_decoded,
        }
        if object_sha256(optimizer_pair_payload) != _sha256(
            pair_row["optimizer_pair_sha256"],
            f"optimizer_pairs[{index}].optimizer_pair_sha256",
        ):
            raise CAPERAdmissionError("optimizer pair decoded-evidence seal differs")
        normalized_pair = _pair(
            pair_payload,
            split="fit",
            policy={
                "policy_id": bindings["policy_id"],
                "checkpoint_tree_sha256": bindings["checkpoint_tree_sha256"],
                "inference_contract_sha256": bindings[
                    "inference_contract_sha256"
                ],
            },
            label=f"optimizer_pairs[{index}]",
        )
        if normalized_pair["action_family"] not in fit_action_families:
            raise CAPERAdmissionError("optimizer pair action lacks common-ledger authority")
        if _pair_rejection_reasons(normalized_pair):
            raise CAPERAdmissionError("optimizer pair fails preservation/action admission")
        observed_ids.append(_identifier(pair_row.get("pair_id"), "optimizer pair_id"))
    if authorized_ids != observed_ids or len(set(observed_ids)) != len(observed_ids):
        raise CAPERAdmissionError("authorized optimizer IDs differ from optimizer pairs")

    def contains_path_key(candidate: Any) -> bool:
        if isinstance(candidate, Mapping):
            return "path" in candidate or any(
                contains_path_key(item) for item in candidate.values()
            )
        if isinstance(candidate, list):
            return any(contains_path_key(item) for item in candidate)
        return False

    heldout = _list(row["heldout_audit_records"], "heldout_audit_records")
    heldout_fields = frozenset(
        {
            "pair_id",
            "split",
            "source_id",
            "identity_id",
            "scene_id",
            "action_family",
            "candidate_ids",
            "source_media_sha256",
            "winner_output_media_sha256",
            "loser_output_media_sha256",
            "winner_action_grade",
            "loser_action_grade",
            "optimizer_target_allowed",
            "pair_sha256",
        }
    )
    for index, record in enumerate(heldout):
        heldout_row = _closed(
            record, heldout_fields, f"heldout_audit_records[{index}]"
        )
        if (
            heldout_row.get("split") != "heldout"
            or heldout_row.get("optimizer_target_allowed") is not False
            or contains_path_key(heldout_row)
        ):
            raise CAPERAdmissionError(
                "heldout evidence must be metadata-only and physically path-free"
            )
        if heldout_row["action_family"] not in heldout_action_families:
            raise CAPERAdmissionError("heldout action lacks audit-ledger authority")
    owners = _list(
        row["pure_t2v_calibration_owners"], "pure_t2v_calibration_owners"
    )
    if not owners or contains_path_key(owners):
        raise CAPERAdmissionError(
            "T2V calibration owners must be non-empty metadata-only records"
        )
    owner_fields = frozenset(
        {
            "owner_id",
            "split",
            "role",
            "generation_mode",
            "action_family",
            "generator_id",
            "checkpoint_tree_sha256",
            "inference_contract_sha256",
            "output_media_sha256",
            "rollout_receipt_artifact_sha256",
            "reward_audit_artifact_sha256",
            "eligible_as_preference_target",
            "eligible_as_training_target",
            "target_video_dependency",
        }
    )
    for index, owner in enumerate(owners):
        owner_row = _closed(owner, owner_fields, f"T2V owner[{index}]")
        for key in ("owner_id", "generator_id", "action_family"):
            _identifier(owner_row[key], f"T2V owner[{index}].{key}")
        if (
            owner_row["split"] not in SPLITS
            or owner_row["role"] != PURE_T2V_ROLE
            or owner_row["generation_mode"] != "pure_t2v"
            or owner_row["eligible_as_preference_target"] is not False
            or owner_row["eligible_as_training_target"] is not False
            or owner_row["target_video_dependency"] is not False
        ):
            raise CAPERAdmissionError("T2V owner role/eligibility differs")
        for key in owner_fields - {
            "owner_id",
            "split",
            "role",
            "generation_mode",
            "action_family",
            "generator_id",
            "eligible_as_preference_target",
            "eligible_as_training_target",
            "target_video_dependency",
        }:
            _sha256(owner_row[key], f"T2V owner[{index}].{key}")
        if owner_row["checkpoint_tree_sha256"] != bindings["checkpoint_tree_sha256"]:
            raise CAPERAdmissionError("T2V owner checkpoint differs from common ledger")

    rejected = _list(row["rejected_pairs"], "rejected_pairs")
    rejected_ids: set[str] = set()
    for index, rejected_pair in enumerate(rejected):
        rejected_row = _closed(
            rejected_pair,
            frozenset({"pair_id", "split", "reasons", "pair_sha256"}),
            f"rejected_pairs[{index}]",
        )
        pair_id = _identifier(rejected_row["pair_id"], "rejected pair_id")
        if pair_id in rejected_ids or pair_id in observed_ids:
            raise CAPERAdmissionError("rejected/optimizer pair IDs alias")
        rejected_ids.add(pair_id)
        if rejected_row["split"] not in SPLITS or contains_path_key(rejected_row):
            raise CAPERAdmissionError("rejected pair must be path-free metadata")
        reasons = _list(rejected_row["reasons"], "rejected pair reasons")
        if not reasons:
            raise CAPERAdmissionError("rejected pair must have at least one reason")
        for reason in reasons:
            _nonempty_text(reason, "rejected pair reason")
        _sha256(rejected_row["pair_sha256"], "rejected pair SHA256")

    step_allowed = row["optimizer_step_allowed"]
    if type(step_allowed) is not bool or step_allowed is not bool(observed_ids):
        raise CAPERAdmissionError("optimizer_step_allowed differs from fit pair closure")
    certificate = row.get("zero_update_certificate")
    if observed_ids and certificate is not None:
        raise CAPERAdmissionError("nonzero admission must not carry zero certificate")
    if not observed_ids and certificate is None:
        raise CAPERAdmissionError("zero admission requires zero-update certificate")
    if certificate is not None:
        cert = _mapping(certificate, "zero_update_certificate")
        cert_digest = _sha256(
            cert.get("certificate_sha256"),
            "zero_update_certificate.certificate_sha256",
        )
        unsigned_cert = dict(cert)
        del unsigned_cert["certificate_sha256"]
        if object_sha256(unsigned_cert) != cert_digest:
            raise CAPERAdmissionError("zero-update certificate SHA-256 differs")
        if (
            cert.get("schema_version") != ZERO_UPDATE_SCHEMA_VERSION
            or cert.get("manifest_id") != row["manifest_id"]
            or cert.get("manifest_sha256") != row["manifest_sha256"]
            or cert.get("reason")
            != "no_valid_fit_same_source_on_policy_preference_pair"
            or cert.get("rejected_fit_pair_ids")
            != [
                rejected_pair["pair_id"]
                for rejected_pair in rejected
                if rejected_pair["split"] == "fit"
            ]
        ):
            raise CAPERAdmissionError("zero-update certificate binding differs")
        _exact_bool(
            row.get("optimizer_step_allowed"), False, "optimizer_step_allowed"
        )
        _exact_bool(
            cert.get("optimizer_step_allowed"),
            False,
            "zero_update_certificate.optimizer_step_allowed",
        )
        if cert.get("authorized_optimizer_steps") != 0:
            raise CAPERAdmissionError(
                "zero-update certificate must authorize exactly zero steps"
            )
        if cert.get("authorized_optimizer_pair_ids") != []:
            raise CAPERAdmissionError("zero-update certificate authorizes pair IDs")
        for key in (
            "gradient_application_allowed",
            "parameter_update_required",
        ):
            _exact_bool(cert.get(key), False, f"zero-update certificate.{key}")
    return dict(row)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    data = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as error:
        raise CAPERAdmissionError(f"refusing to overwrite output: {path}") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--verify-files", action="store_true")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = load_manifest(args.manifest)
    base_dir = args.manifest.parent
    if args.command == "validate":
        normalized = validate_manifest(
            payload, verify_files=args.verify_files, base_dir=base_dir
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "manifest_id": normalized["manifest_id"],
                    "manifest_sha256": object_sha256(normalized),
                    "artifact_bytes_verified": args.verify_files,
                },
                sort_keys=True,
            )
        )
        return 0
    materialization = materialize_admission(
        payload, verify_files=True, base_dir=base_dir
    )
    _write_new_json(args.output, materialization)
    print(
        json.dumps(
            {
                "status": "materialized",
                "output": str(args.output),
                "optimizer_step_allowed": materialization[
                    "optimizer_step_allowed"
                ],
                "authorized_optimizer_pair_ids": materialization[
                    "authorized_optimizer_pair_ids"
                ],
                "materialization_sha256": materialization[
                    "materialization_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI exercised via functions.
    raise SystemExit(main())
