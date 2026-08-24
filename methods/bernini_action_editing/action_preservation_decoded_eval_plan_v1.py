#!/usr/bin/env python3
"""Build a sealed local decoded-evaluation plan for preservation-v2.

The plan is a complete, unfiltered Cartesian matrix.  It does not inspect
training loss, choose a checkpoint, launch a process, upload artifacts, or
authorize a scientific winner.  Its only mutation is a create-only local
publication rooted at a caller-supplied fresh directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import action_preservation_gate_v1 as gate


INPUT_SCHEMA = "bernini-action-preservation-decoded-eval-input-v1"
MANIFEST_SCHEMA = "bernini-action-preservation-decoded-eval-manifest-v1"
SHARD_SCHEMA = "bernini-action-preservation-decoded-eval-shard-v1"
REVIEW_CONTRACT_SCHEMA = "bernini-action-preservation-review-packet-contract-v1"
ACTION_REVIEW_CONTRACT_SCHEMA = (
    "bernini-action-preservation-action-review-contract-v1"
)
PUBLICATION_SCHEMA = "bernini-action-preservation-decoded-eval-publication-v3"
DIRECTORY_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-decoded-eval-directory-authority-v1"
)
HOLDER_DIRECTORY_COMPLETION_SCHEMA = (
    "bernini-action-preservation-decoded-eval-holder-directory-completion-v2"
)
FINAL_DIRECTORY_MERGE_SCHEMA = (
    "bernini-action-preservation-decoded-eval-final-directory-merge-v2"
)

ARMS = (
    "v2_onset_all",
    "v2_noop020_all",
    "v2_func010_all",
    "v2_func025_all",
    "v2_func050_all",
    "v2_onset_cross_qo",
    "v2_func010_cross_qo",
    "v2_func025_cross_qo",
)
CHECKPOINT_STEPS = (0, 5, 10, 20)
FITTED_IIDS = (
    "7b88a1ca1f804f41",
    "841b5e0080a1441d",
    "a35b590961d24694",
    "a66e6818e4144928",
)
POLICIES = ("none", "hard1_every_step")
PRIVATE_REVIEW_TOKENS = (
    *ARMS,
    "hard1_every_step",
    "checkpoint-",
    "checkpoint_step",
    "onset_policy",
    "adapter_sha256",
    "lora_safe_merge",
)

HOLDER_ROWS = (
    {
        "job_id": "136719",
        "node": "auh7-1b-gpu-306",
        "arms": ("v2_onset_all", "v2_noop020_all"),
        "base_control_iid": FITTED_IIDS[0],
    },
    {
        "job_id": "136141",
        "node": "auh7-1b-gpu-299",
        "arms": ("v2_func010_all", "v2_func025_all"),
        "base_control_iid": FITTED_IIDS[1],
    },
    {
        "job_id": "136309",
        "node": "auh7-1b-gpu-280",
        "arms": ("v2_func050_all", "v2_onset_cross_qo"),
        "base_control_iid": FITTED_IIDS[2],
    },
    {
        "job_id": "136140",
        "node": "auh7-1b-gpu-215",
        "arms": ("v2_func010_cross_qo", "v2_func025_cross_qo"),
        "base_control_iid": FITTED_IIDS[3],
    },
)

CANDIDATE_COUNT = len(ARMS) * len(CHECKPOINT_STEPS) * len(FITTED_IIDS) * len(POLICIES)
BASE_CONTROL_COUNT = len(FITTED_IIDS) * len(POLICIES)
TOTAL_DECODE_COUNT = CANDIDATE_COUNT + BASE_CONTROL_COUNT

REVIEW_AXES = gate.AXES
REVIEW_STATES = gate.STATES

INPUT_FILENAME = "evaluation_input.json"
MANIFEST_FILENAME = "evaluation_manifest.json"
REVIEW_CONTRACT_FILENAME = "review_packet_contract.json"
PUBLICATION_FILENAME = "publication_receipt.json"
DIRECTORY_AUTHORITY_FILENAME = "directory_authority.json"
SHARD_DIRECTORY = "shards"
OUTPUT_CANDIDATE_DIRECTORY = "candidates"
OUTPUT_BASE_DIRECTORY = "frozen_base_controls"
EXECUTION_SHARD_DIRECTORY = "execution_shards"
CONSUMPTION_AUTHORITY_DIRECTORY = "consumption_authority"
HOLDER_DIRECTORY_COMPLETION_SUFFIX = ".holder-directory-completion.json"
HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE = 0o600
HOLDER_DIRECTORY_COMPLETION_SEALED_MODE = 0o444

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

PIN_FIELDS = frozenset(
    {
        "source_manifest_sha256",
        "adapter_release_manifest_sha256",
        "model_release_manifest_sha256",
        "inference_source_sha256",
        "inference_release_manifest_sha256",
        "inference_config_sha256",
        "source_preprocessing_sha256",
        "calibration_digest",
    }
)
SOURCE_FIELDS = frozenset(
    {
        "iid",
        "source_video_sha256",
        "source_receipt_sha256",
        "instruction",
        "instruction_sha256",
        "action_review_contract",
        "seed",
    }
)
CHECKPOINT_FIELDS = frozenset(
    {
        "arm",
        "checkpoint_step",
        "checkpoint_receipt_sha256",
        "adapter_sha256",
    }
)


class DecodedEvaluationPlanError(RuntimeError):
    pass


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
        raise DecodedEvaluationPlanError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _public_review_text(value: str, *, label: str) -> str:
    lowered = value.lower()
    if any(token.lower() in lowered for token in PRIVATE_REVIEW_TOKENS):
        raise DecodedEvaluationPlanError(
            f"{label} leaks method/arm/checkpoint/policy authority"
        )
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise DecodedEvaluationPlanError(f"{label} field closure differs")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DecodedEvaluationPlanError(f"{label} is not a lowercase SHA-256")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise DecodedEvaluationPlanError(f"{label} is invalid")
    return value


def _absolute_nonroot_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DecodedEvaluationPlanError(f"{label} is not a path string")
    if not Path(value).is_absolute() or value == os.path.sep:
        raise DecodedEvaluationPlanError(f"{label} must be absolute and non-root")
    if os.path.normpath(value) != value:
        raise DecodedEvaluationPlanError(f"{label} must be lexically normalized")
    return value


def _verify_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    digest = _sha(value.get(field), label=f"{label} digest")
    payload = dict(value)
    payload.pop(field)
    if object_sha256(payload) != digest:
        raise DecodedEvaluationPlanError(f"{label} digest differs")
    return digest


def _media_contract() -> dict[str, Any]:
    return {
        "container": "mp4",
        "video_stream_count": 1,
        "frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
        "full_video_required": True,
        "create_only_output": True,
    }


def _policy_contract(name: str) -> dict[str, Any]:
    if name == "none":
        return {
            "name": name,
            "source_onset_policy": "none",
            "applied_every_solver_step": False,
            "phase0_hard_source_noise_clamp": False,
            "later_phase_ramp_reapplied": False,
        }
    if name == "hard1_every_step":
        return {
            "name": name,
            "source_onset_policy": "hard1_every_step",
            "applied_every_solver_step": True,
            "phase0_hard_source_noise_clamp": True,
            "later_phase_ramp_reapplied": False,
        }
    raise DecodedEvaluationPlanError(f"unknown onset policy: {name}")


def validate_action_review_contract(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "action_order_description",
        "action_order_description_sha256",
        "expected_onset_frame_min",
        "expected_onset_frame_max",
        "terminal_hold_start_frame_min",
        "terminal_hold_end_frame",
        "full_video_frame_count",
        "fps_num",
        "fps_den",
        "contract_digest",
    }
    row = dict(_closed(value, fields, label="action review contract"))
    if row["schema_version"] != ACTION_REVIEW_CONTRACT_SCHEMA:
        raise DecodedEvaluationPlanError("action review contract schema differs")
    description = row["action_order_description"]
    if (
        not isinstance(description, str)
        or not description.strip()
        or description != description.strip()
        or "\x00" in description
        or text_sha256(description) != row["action_order_description_sha256"]
    ):
        raise DecodedEvaluationPlanError("action order authority differs")
    _public_review_text(description, label="action order authority")
    integer_fields = (
        "expected_onset_frame_min",
        "expected_onset_frame_max",
        "terminal_hold_start_frame_min",
        "terminal_hold_end_frame",
        "full_video_frame_count",
        "fps_num",
        "fps_den",
    )
    if any(type(row[key]) is not int for key in integer_fields):
        raise DecodedEvaluationPlanError("action timing authority is not integral")
    if not (
        0 <= row["expected_onset_frame_min"]
        <= row["expected_onset_frame_max"]
        < row["terminal_hold_start_frame_min"]
        <= row["terminal_hold_end_frame"] == 80
        and row["full_video_frame_count"] == 81
        and row["fps_num"] == 25
        and row["fps_den"] == 1
    ):
        raise DecodedEvaluationPlanError("action timing authority differs")
    _verify_digest(row, field="contract_digest", label="action review contract")
    return row


def _review_submission_schema() -> dict[str, Any]:
    return {
        "schema_version": gate.BLIND_REVIEW_SCHEMA,
        "required_top_level_fields": [
            "schema_version",
            "evaluation_aggregate_digest",
            "evaluation_manifest_digest",
            "public_packet_digest",
            "private_mapping_digest",
            "blind_candidate_id",
            "blind_row_digest",
            "private_row_digest",
            "source_video_sha256",
            "source_receipt_sha256",
            "candidate_video_sha256",
            "matched_base_video_sha256",
            "candidate_output_digest",
            "matched_base_output_digest",
            "full_video_receipt_sha256",
            "matched_base_full_video_receipt_sha256",
            "instruction_sha256",
            "action_review_contract_digest",
            "decision_digest",
            "method_hidden",
            "reviewers",
            "axis_resolution",
            "ballot_closure_digest",
            "review_digest",
        ],
        "reviewer_entry_fields": [
            "schema_version",
            "public_packet_digest",
            "blind_candidate_id",
            "blind_row_digest",
            "source_video_sha256",
            "candidate_video_sha256",
            "matched_base_video_sha256",
            "candidate_output_digest",
            "matched_base_output_digest",
            "full_video_receipt_sha256",
            "matched_base_full_video_receipt_sha256",
            "instruction_sha256",
            "action_review_contract_digest",
            "reviewer_id",
            "independent_review",
            "full_video_reviewed",
            "labels",
            "ballot_digest",
        ],
        "reviewer_entry_schema_version": gate.BLIND_BALLOT_SCHEMA,
        "label_field_closure": list(REVIEW_AXES),
        "allowed_label_states": list(REVIEW_STATES),
        "minimum_reviewer_count": 2,
        "reviewer_ids_must_be_unique": True,
        "every_independent_review_must_be_true": True,
        "every_full_video_reviewed_must_be_true": True,
        "axis_resolution_must_equal_ballot_consensus": True,
        "aggregate_public_private_exact_row_binding_required": True,
        "instruction_and_action_timing_bound_per_row_and_ballot": True,
        "matched_base_output_and_receipt_bound_per_row_and_ballot": True,
        "method_arm_checkpoint_policy_fields_forbidden": True,
        "weighted_score_field_forbidden": True,
        "unresolved_or_disagreeing_axis_becomes": "undetermined",
    }


def build_review_packet_contract() -> dict[str, Any]:
    value = {
        "schema_version": REVIEW_CONTRACT_SCHEMA,
        "axes": [
            {
                "name": axis,
                "states": list(REVIEW_STATES),
                "required_for_every_full_video": True,
            }
            for axis in REVIEW_AXES
        ],
        "blinding": {
            "method_hidden": True,
            "arm_hidden": True,
            "checkpoint_hidden": True,
            "onset_policy_hidden": True,
            "candidate_order_randomized_from_separate_private_key": True,
            "private_key_must_not_be_in_public_packet": True,
        },
        "review": {
            "full_81_frame_video_required": True,
            "minimum_independent_reviewer_count": 2,
            "source_video_visible_as_identity_background_camera_authority": True,
            "action_anchor_may_only_define_action_order_and_onset": True,
            "instruction_visible": True,
            "opaque_action_timing_authority_visible": True,
            "matched_base_visible": True,
            "public_packet_and_blind_row_digest_required_per_ballot": True,
            "private_mapping_digest_required_at_aggregation": True,
            "all_axes_use_explicit_abstain": True,
            "reviewer_ids_must_be_unique": True,
            "per_reviewer_six_axis_ballot_required": True,
            "aggregate_must_be_derived_from_ballots": True,
            "ballot_disagreement_becomes_undetermined": True,
        },
        "machine_gate": {
            "measurement_schema": gate.MEASUREMENT_SCHEMA,
            "decision_schema": gate.DECISION_SCHEMA,
            "calibrated_evidence_required": True,
            "separate_machine_axes": list(gate.AXES),
            "machine_abstain_blocks_promotion": True,
            "all_supported_machine_axes_must_pass": True,
            "whole_frame_dino_identity_shortcut_forbidden": True,
            "fixed_source_mask_background_shortcut_forbidden": True,
        },
        "promotion": {
            "blind_full_video_review_required": True,
            "all_human_axes_must_pass": True,
            "machine_gate_must_pass": True,
            "loss_may_not_enter_promotion": True,
            "automatic_model_update": False,
            "outcome_if_any_fail": "STOP",
            "outcome_if_any_abstain": "STOP",
            "outcome_if_machine_pass_but_review_missing": "WAIT_FOR_BLIND_REVIEW",
            "maximum_positive_outcome": "ELIGIBLE_FOR_NEXT_20",
        },
        "submission_schema": _review_submission_schema(),
        "weighted_score": None,
        "weighted_compensation_forbidden": True,
    }
    value["contract_digest"] = object_sha256(value)
    return validate_review_packet_contract(value)


def validate_review_packet_contract(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "axes",
        "blinding",
        "review",
        "machine_gate",
        "promotion",
        "submission_schema",
        "weighted_score",
        "weighted_compensation_forbidden",
        "contract_digest",
    }
    row = dict(_closed(value, fields, label="review packet contract"))
    if row["schema_version"] != REVIEW_CONTRACT_SCHEMA:
        raise DecodedEvaluationPlanError("review contract schema differs")
    expected_axes = [
        {
            "name": axis,
            "states": list(REVIEW_STATES),
            "required_for_every_full_video": True,
        }
        for axis in REVIEW_AXES
    ]
    if row["axes"] != expected_axes:
        raise DecodedEvaluationPlanError("review axes or ABSTAIN states differ")
    # The remaining fields are deliberately compared to literal closed
    # contracts rather than treated as descriptive metadata.
    if row["blinding"] != {
        "method_hidden": True,
        "arm_hidden": True,
        "checkpoint_hidden": True,
        "onset_policy_hidden": True,
        "candidate_order_randomized_from_separate_private_key": True,
        "private_key_must_not_be_in_public_packet": True,
    }:
        raise DecodedEvaluationPlanError("review blinding contract differs")
    if row["review"] != {
        "full_81_frame_video_required": True,
        "minimum_independent_reviewer_count": 2,
        "source_video_visible_as_identity_background_camera_authority": True,
        "action_anchor_may_only_define_action_order_and_onset": True,
        "instruction_visible": True,
        "opaque_action_timing_authority_visible": True,
        "matched_base_visible": True,
        "public_packet_and_blind_row_digest_required_per_ballot": True,
        "private_mapping_digest_required_at_aggregation": True,
        "all_axes_use_explicit_abstain": True,
        "reviewer_ids_must_be_unique": True,
        "per_reviewer_six_axis_ballot_required": True,
        "aggregate_must_be_derived_from_ballots": True,
        "ballot_disagreement_becomes_undetermined": True,
    }:
        raise DecodedEvaluationPlanError("review authority contract differs")
    if row["machine_gate"] != {
        "measurement_schema": gate.MEASUREMENT_SCHEMA,
        "decision_schema": gate.DECISION_SCHEMA,
        "calibrated_evidence_required": True,
        "separate_machine_axes": list(gate.AXES),
        "machine_abstain_blocks_promotion": True,
        "all_supported_machine_axes_must_pass": True,
        "whole_frame_dino_identity_shortcut_forbidden": True,
        "fixed_source_mask_background_shortcut_forbidden": True,
    }:
        raise DecodedEvaluationPlanError("machine gate contract differs")
    if row["promotion"] != {
        "blind_full_video_review_required": True,
        "all_human_axes_must_pass": True,
        "machine_gate_must_pass": True,
        "loss_may_not_enter_promotion": True,
        "automatic_model_update": False,
        "outcome_if_any_fail": "STOP",
        "outcome_if_any_abstain": "STOP",
        "outcome_if_machine_pass_but_review_missing": "WAIT_FOR_BLIND_REVIEW",
        "maximum_positive_outcome": "ELIGIBLE_FOR_NEXT_20",
    }:
        raise DecodedEvaluationPlanError("promotion contract differs")
    if row["submission_schema"] != _review_submission_schema():
        raise DecodedEvaluationPlanError("review submission schema differs")
    if row["weighted_score"] is not None or row["weighted_compensation_forbidden"] is not True:
        raise DecodedEvaluationPlanError("weighted compensation is not forbidden")
    _verify_digest(row, field="contract_digest", label="review packet contract")
    return row


def validate_input_spec(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "evaluation_id",
        "evaluation_root",
        "pins",
        "sources",
        "checkpoints",
        "input_digest",
    }
    row = dict(_closed(value, fields, label="evaluation input"))
    if row["schema_version"] != INPUT_SCHEMA:
        raise DecodedEvaluationPlanError("evaluation input schema differs")
    _identifier(row["evaluation_id"], label="evaluation id")
    row["evaluation_root"] = _absolute_nonroot_path(
        row["evaluation_root"], label="evaluation root"
    )
    pins = dict(_closed(row["pins"], PIN_FIELDS, label="evaluation pins"))
    for key in sorted(PIN_FIELDS - {"calibration_digest"}):
        pins[key] = _sha(pins[key], label=key)
    if pins["calibration_digest"] is not None:
        pins["calibration_digest"] = _sha(
            pins["calibration_digest"], label="calibration_digest"
        )

    sources = row["sources"]
    if not isinstance(sources, list) or len(sources) != len(FITTED_IIDS):
        raise DecodedEvaluationPlanError("source row count must be exactly four")
    normalized_sources: list[dict[str, Any]] = []
    for expected_iid, source_value in zip(FITTED_IIDS, sources):
        source = dict(_closed(source_value, SOURCE_FIELDS, label="source row"))
        if source["iid"] != expected_iid:
            raise DecodedEvaluationPlanError("fitted IID order or identity differs")
        _sha(source["source_video_sha256"], label="source video")
        _sha(source["source_receipt_sha256"], label="source receipt")
        instruction = source["instruction"]
        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or instruction != instruction.strip()
            or "\x00" in instruction
        ):
            raise DecodedEvaluationPlanError("instruction is invalid")
        _sha(source["instruction_sha256"], label="instruction")
        if text_sha256(instruction) != source["instruction_sha256"]:
            raise DecodedEvaluationPlanError("instruction hash differs")
        _public_review_text(instruction, label="instruction")
        source["action_review_contract"] = validate_action_review_contract(
            source["action_review_contract"]
        )
        if (
            type(source["seed"]) is not int
            or source["seed"] < 0
            or source["seed"] >= 2**63
        ):
            raise DecodedEvaluationPlanError("evaluation seed is invalid")
        normalized_sources.append(source)

    checkpoints = row["checkpoints"]
    expected_keys = [
        (arm, step) for arm in ARMS for step in CHECKPOINT_STEPS
    ]
    if not isinstance(checkpoints, list) or len(checkpoints) != len(expected_keys):
        raise DecodedEvaluationPlanError("checkpoint input count must be exactly 32")
    normalized_checkpoints: list[dict[str, Any]] = []
    for expected_key, checkpoint_value in zip(expected_keys, checkpoints):
        checkpoint = dict(
            _closed(checkpoint_value, CHECKPOINT_FIELDS, label="checkpoint input")
        )
        if (checkpoint["arm"], checkpoint["checkpoint_step"]) != expected_key:
            raise DecodedEvaluationPlanError("checkpoint arm/step matrix differs")
        _sha(checkpoint["checkpoint_receipt_sha256"], label="checkpoint receipt")
        _sha(checkpoint["adapter_sha256"], label="adapter")
        normalized_checkpoints.append(checkpoint)

    _verify_digest(row, field="input_digest", label="evaluation input")
    row.update(pins=pins, sources=normalized_sources, checkpoints=normalized_checkpoints)
    return row


def build_input_spec(
    *,
    evaluation_id: str,
    evaluation_root: str | Path,
    pins: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = {
        "schema_version": INPUT_SCHEMA,
        "evaluation_id": evaluation_id,
        "evaluation_root": str(evaluation_root),
        "pins": dict(pins),
        "sources": [dict(item) for item in sources],
        "checkpoints": [dict(item) for item in checkpoints],
    }
    value["input_digest"] = object_sha256(value)
    return validate_input_spec(value)


def _record_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("record_digest", None)
    return object_sha256(payload)


def _candidate_id(*, arm: str, step: int, iid: str, seed: int, policy: str) -> str:
    return f"v2eval-{arm}-u{step}-{iid}-s{seed}-{policy}"


def _base_control_id(*, iid: str, seed: int, policy: str) -> str:
    return f"v2eval-frozen-base-{iid}-s{seed}-{policy}"


def _arm_holder(arm: str) -> Mapping[str, Any]:
    matches = [row for row in HOLDER_ROWS if arm in row["arms"]]
    if len(matches) != 1:
        raise DecodedEvaluationPlanError(f"arm holder mapping differs: {arm}")
    return matches[0]


def _base_holder(iid: str) -> Mapping[str, Any]:
    matches = [row for row in HOLDER_ROWS if iid == row["base_control_iid"]]
    if len(matches) != 1:
        raise DecodedEvaluationPlanError(f"base-control holder mapping differs: {iid}")
    return matches[0]


def _candidate_record(
    *,
    source: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    policy: str,
    pins: Mapping[str, Any],
) -> dict[str, Any]:
    arm = checkpoint["arm"]
    step = checkpoint["checkpoint_step"]
    iid = source["iid"]
    seed = source["seed"]
    holder = _arm_holder(arm)
    value = {
        "candidate_id": _candidate_id(
            arm=arm, step=step, iid=iid, seed=seed, policy=policy
        ),
        "kind": "adapter_candidate",
        "arm": arm,
        "checkpoint_step": step,
        "iid": iid,
        "seed": seed,
        "onset_policy": _policy_contract(policy),
        "source_video_sha256": source["source_video_sha256"],
        "source_receipt_sha256": source["source_receipt_sha256"],
        "instruction": source["instruction"],
        "instruction_sha256": source["instruction_sha256"],
        "action_review_contract": source["action_review_contract"],
        "checkpoint_receipt_sha256": checkpoint["checkpoint_receipt_sha256"],
        "adapter_sha256": checkpoint["adapter_sha256"],
        "matched_frozen_base_control_id": _base_control_id(
            iid=iid, seed=seed, policy=policy
        ),
        "adapter_release_manifest_sha256": pins["adapter_release_manifest_sha256"],
        "model_release_manifest_sha256": pins["model_release_manifest_sha256"],
        "inference_source_sha256": pins["inference_source_sha256"],
        "inference_release_manifest_sha256": pins[
            "inference_release_manifest_sha256"
        ],
        "inference_config_sha256": pins["inference_config_sha256"],
        "source_preprocessing_sha256": pins["source_preprocessing_sha256"],
        "calibration_digest": pins["calibration_digest"],
        "holder": {"job_id": holder["job_id"], "node": holder["node"]},
        "output_relpath": (
            f"candidates/{arm}/u{step}/{iid}/s{seed}/{policy}.mp4"
        ),
        "media_contract": _media_contract(),
        "target_video_available_to_inference": False,
        "training_loss_read_or_used_for_selection": False,
    }
    value["record_digest"] = _record_digest(value)
    return value


def _base_control_record(
    *, source: Mapping[str, Any], policy: str, pins: Mapping[str, Any]
) -> dict[str, Any]:
    iid = source["iid"]
    seed = source["seed"]
    holder = _base_holder(iid)
    value = {
        "control_id": _base_control_id(iid=iid, seed=seed, policy=policy),
        "kind": "frozen_base_control",
        "iid": iid,
        "seed": seed,
        "onset_policy": _policy_contract(policy),
        "source_video_sha256": source["source_video_sha256"],
        "source_receipt_sha256": source["source_receipt_sha256"],
        "instruction": source["instruction"],
        "instruction_sha256": source["instruction_sha256"],
        "action_review_contract": source["action_review_contract"],
        "adapter_sha256": None,
        "adapter_checkpoint_receipt_sha256": None,
        "model_release_manifest_sha256": pins["model_release_manifest_sha256"],
        "inference_source_sha256": pins["inference_source_sha256"],
        "inference_release_manifest_sha256": pins[
            "inference_release_manifest_sha256"
        ],
        "inference_config_sha256": pins["inference_config_sha256"],
        "source_preprocessing_sha256": pins["source_preprocessing_sha256"],
        "calibration_digest": pins["calibration_digest"],
        "holder": {"job_id": holder["job_id"], "node": holder["node"]},
        "deduplication_key": object_sha256(
            {"iid": iid, "seed": seed, "onset_policy": policy}
        ),
        "output_relpath": f"frozen_base_controls/{iid}/s{seed}/{policy}.mp4",
        "media_contract": _media_contract(),
        "target_video_available_to_inference": False,
        "training_loss_read_or_used_for_selection": False,
    }
    value["record_digest"] = _record_digest(value)
    return value


def build_manifest(input_spec: Mapping[str, Any]) -> dict[str, Any]:
    authority = validate_input_spec(input_spec)
    source_by_iid = {row["iid"]: row for row in authority["sources"]}
    checkpoint_by_key = {
        (row["arm"], row["checkpoint_step"]): row
        for row in authority["checkpoints"]
    }
    candidates = [
        _candidate_record(
            source=source_by_iid[iid],
            checkpoint=checkpoint_by_key[(arm, step)],
            policy=policy,
            pins=authority["pins"],
        )
        for arm in ARMS
        for step in CHECKPOINT_STEPS
        for iid in FITTED_IIDS
        for policy in POLICIES
    ]
    controls = [
        _base_control_record(
            source=source_by_iid[iid], policy=policy, pins=authority["pins"]
        )
        for iid in FITTED_IIDS
        for policy in POLICIES
    ]
    review = build_review_packet_contract()
    value = {
        "schema_version": MANIFEST_SCHEMA,
        "evaluation_id": authority["evaluation_id"],
        "evaluation_root": authority["evaluation_root"],
        "input_digest": authority["input_digest"],
        "pins": authority["pins"],
        "matrix": {
            "arms": list(ARMS),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "fitted_iids": list(FITTED_IIDS),
            "onset_policies": list(POLICIES),
            "eval_seed_count_per_iid": 1,
            "candidate_video_count": CANDIDATE_COUNT,
            "frozen_base_control_count": BASE_CONTROL_COUNT,
            "total_decode_count": TOTAL_DECODE_COUNT,
            "full_cartesian_product_required": True,
        },
        "selection_policy": {
            "candidate_subset_selection": "none_full_cartesian",
            "training_loss_read": False,
            "training_loss_filtering": False,
            "checkpoint_loss_ranking": False,
            "missing_candidate_is_failure": True,
        },
        "pairing_contract": {
            "same_source_instruction_seed_preprocessing_sampler_guidance": True,
            "same_initial_noise_for_none_vs_hard1_every_step": True,
            "same_seed_across_all_arms_and_checkpoints_for_each_iid": True,
            "only_adapter_checkpoint_and_onset_policy_may_vary": True,
            "frozen_base_controls_share_iid_seed_policy": True,
        },
        "execution_policy": {
            "local_plan_generation_only": True,
            "remote_launch_performed": False,
            "upload_performed": False,
            "fresh_create_only_root_required": True,
            "candidate_outputs_create_only": True,
        },
        "holder_assignment": [
            {
                "job_id": row["job_id"],
                "node": row["node"],
                "arms": list(row["arms"]),
                "base_control_iid": row["base_control_iid"],
            }
            for row in HOLDER_ROWS
        ],
        "review_packet_contract_digest": review["contract_digest"],
        "candidates": candidates,
        "frozen_base_controls": controls,
    }
    value["manifest_digest"] = object_sha256(value)
    return validate_manifest(value, input_spec=authority)


def _expected_candidates(input_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_by_iid = {row["iid"]: row for row in input_spec["sources"]}
    checkpoint_by_key = {
        (row["arm"], row["checkpoint_step"]): row
        for row in input_spec["checkpoints"]
    }
    return [
        _candidate_record(
            source=source_by_iid[iid],
            checkpoint=checkpoint_by_key[(arm, step)],
            policy=policy,
            pins=input_spec["pins"],
        )
        for arm in ARMS
        for step in CHECKPOINT_STEPS
        for iid in FITTED_IIDS
        for policy in POLICIES
    ]


def _expected_controls(input_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_by_iid = {row["iid"]: row for row in input_spec["sources"]}
    return [
        _base_control_record(
            source=source_by_iid[iid], policy=policy, pins=input_spec["pins"]
        )
        for iid in FITTED_IIDS
        for policy in POLICIES
    ]


def validate_manifest(
    value: Any, *, input_spec: Mapping[str, Any]
) -> dict[str, Any]:
    authority = validate_input_spec(input_spec)
    fields = {
        "schema_version",
        "evaluation_id",
        "evaluation_root",
        "input_digest",
        "pins",
        "matrix",
        "selection_policy",
        "pairing_contract",
        "execution_policy",
        "holder_assignment",
        "review_packet_contract_digest",
        "candidates",
        "frozen_base_controls",
        "manifest_digest",
    }
    row = dict(_closed(value, fields, label="evaluation manifest"))
    if row["schema_version"] != MANIFEST_SCHEMA:
        raise DecodedEvaluationPlanError("evaluation manifest schema differs")
    for key in ("evaluation_id", "evaluation_root", "input_digest", "pins"):
        if row[key] != authority[key]:
            raise DecodedEvaluationPlanError(f"manifest input binding differs: {key}")
    if row["matrix"] != {
        "arms": list(ARMS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "fitted_iids": list(FITTED_IIDS),
        "onset_policies": list(POLICIES),
        "eval_seed_count_per_iid": 1,
        "candidate_video_count": CANDIDATE_COUNT,
        "frozen_base_control_count": BASE_CONTROL_COUNT,
        "total_decode_count": TOTAL_DECODE_COUNT,
        "full_cartesian_product_required": True,
    }:
        raise DecodedEvaluationPlanError("evaluation matrix contract differs")
    if row["selection_policy"] != {
        "candidate_subset_selection": "none_full_cartesian",
        "training_loss_read": False,
        "training_loss_filtering": False,
        "checkpoint_loss_ranking": False,
        "missing_candidate_is_failure": True,
    }:
        raise DecodedEvaluationPlanError("loss-free selection policy differs")
    if row["pairing_contract"] != {
        "same_source_instruction_seed_preprocessing_sampler_guidance": True,
        "same_initial_noise_for_none_vs_hard1_every_step": True,
        "same_seed_across_all_arms_and_checkpoints_for_each_iid": True,
        "only_adapter_checkpoint_and_onset_policy_may_vary": True,
        "frozen_base_controls_share_iid_seed_policy": True,
    }:
        raise DecodedEvaluationPlanError("matched seed/policy pairing contract differs")
    if row["execution_policy"] != {
        "local_plan_generation_only": True,
        "remote_launch_performed": False,
        "upload_performed": False,
        "fresh_create_only_root_required": True,
        "candidate_outputs_create_only": True,
    }:
        raise DecodedEvaluationPlanError("local-only execution policy differs")
    expected_holders = [
        {
            "job_id": holder["job_id"],
            "node": holder["node"],
            "arms": list(holder["arms"]),
            "base_control_iid": holder["base_control_iid"],
        }
        for holder in HOLDER_ROWS
    ]
    if row["holder_assignment"] != expected_holders:
        raise DecodedEvaluationPlanError("holder assignment differs")
    review_digest = build_review_packet_contract()["contract_digest"]
    if row["review_packet_contract_digest"] != review_digest:
        raise DecodedEvaluationPlanError("review packet contract binding differs")

    expected_candidates = _expected_candidates(authority)
    if row["candidates"] != expected_candidates:
        raise DecodedEvaluationPlanError("candidate matrix or provenance binding differs")
    expected_controls = _expected_controls(authority)
    if row["frozen_base_controls"] != expected_controls:
        raise DecodedEvaluationPlanError("frozen-base control closure differs")
    if len({item["candidate_id"] for item in expected_candidates}) != CANDIDATE_COUNT:
        raise DecodedEvaluationPlanError("candidate identifiers are not unique")
    if len({item["output_relpath"] for item in expected_candidates}) != CANDIDATE_COUNT:
        raise DecodedEvaluationPlanError("candidate output paths are not unique")
    if len({item["deduplication_key"] for item in expected_controls}) != BASE_CONTROL_COUNT:
        raise DecodedEvaluationPlanError("frozen-base controls are not deduplicated")
    all_paths = [item["output_relpath"] for item in expected_candidates] + [
        item["output_relpath"] for item in expected_controls
    ]
    if len(set(all_paths)) != TOTAL_DECODE_COUNT:
        raise DecodedEvaluationPlanError("decode output paths alias")
    _verify_digest(row, field="manifest_digest", label="evaluation manifest")
    return row


def _shard_tasks(manifest: Mapping[str, Any], holder: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = [
        {"task_kind": "adapter_candidate", "record": item}
        for item in manifest["candidates"]
        if item["holder"]["job_id"] == holder["job_id"]
    ]
    tasks.extend(
        {"task_kind": "frozen_base_control", "record": item}
        for item in manifest["frozen_base_controls"]
        if item["holder"]["job_id"] == holder["job_id"]
    )
    return tasks


def _build_shard_value(
    manifest: Mapping[str, Any], holder: Mapping[str, Any]
) -> dict[str, Any]:
    tasks = _shard_tasks(manifest, holder)
    value = {
        "schema_version": SHARD_SCHEMA,
        "evaluation_id": manifest["evaluation_id"],
        "evaluation_manifest_digest": manifest["manifest_digest"],
        "holder": {"job_id": holder["job_id"], "node": holder["node"]},
        "assigned_arms": list(holder["arms"]),
        "assigned_base_control_iid": holder["base_control_iid"],
        "candidate_task_count": sum(
            item["task_kind"] == "adapter_candidate" for item in tasks
        ),
        "base_control_task_count": sum(
            item["task_kind"] == "frozen_base_control" for item in tasks
        ),
        "total_task_count": len(tasks),
        "tasks": tasks,
        "task_order": "manifest_order_candidates_then_deduplicated_controls",
        "training_loss_read_or_used_for_sharding": False,
        "remote_launch_performed": False,
        "upload_performed": False,
        "outputs_create_only": True,
    }
    value["shard_digest"] = object_sha256(value)
    return value


def build_shards(manifest: Mapping[str, Any], *, input_spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    authority = validate_manifest(manifest, input_spec=input_spec)
    return {
        holder["job_id"]: validate_shard(
            _build_shard_value(authority, holder),
            manifest=authority,
            input_spec=input_spec,
        )
        for holder in HOLDER_ROWS
    }


def validate_shard(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    input_spec: Mapping[str, Any],
) -> dict[str, Any]:
    authority = validate_manifest(manifest, input_spec=input_spec)
    fields = {
        "schema_version",
        "evaluation_id",
        "evaluation_manifest_digest",
        "holder",
        "assigned_arms",
        "assigned_base_control_iid",
        "candidate_task_count",
        "base_control_task_count",
        "total_task_count",
        "tasks",
        "task_order",
        "training_loss_read_or_used_for_sharding",
        "remote_launch_performed",
        "upload_performed",
        "outputs_create_only",
        "shard_digest",
    }
    row = dict(_closed(value, fields, label="evaluation shard"))
    if row["schema_version"] != SHARD_SCHEMA:
        raise DecodedEvaluationPlanError("evaluation shard schema differs")
    holder_matches = [
        holder
        for holder in HOLDER_ROWS
        if row.get("holder")
        == {"job_id": holder["job_id"], "node": holder["node"]}
    ]
    if len(holder_matches) != 1:
        raise DecodedEvaluationPlanError("shard holder differs")
    expected = _build_shard_value(authority, holder_matches[0])
    if row != expected:
        raise DecodedEvaluationPlanError("shard task/provenance closure differs")
    if row["candidate_task_count"] != CANDIDATE_COUNT // len(HOLDER_ROWS):
        raise DecodedEvaluationPlanError("per-holder candidate task count differs")
    if row["base_control_task_count"] != BASE_CONTROL_COUNT // len(HOLDER_ROWS):
        raise DecodedEvaluationPlanError("per-holder base-control count differs")
    _verify_digest(row, field="shard_digest", label="evaluation shard")
    return row


def build_bundle(input_spec: Mapping[str, Any]) -> dict[str, Any]:
    authority = validate_input_spec(input_spec)
    review = build_review_packet_contract()
    manifest = build_manifest(authority)
    shards = build_shards(manifest, input_spec=authority)
    return {
        "input_spec": authority,
        "review_contract": review,
        "manifest": manifest,
        "shards": shards,
    }


def holder_completion_reservation_relative(holder_job_id: str) -> str:
    if holder_job_id not in {row["job_id"] for row in HOLDER_ROWS}:
        raise DecodedEvaluationPlanError(
            "holder completion reservation holder differs"
        )
    return (
        f"{EXECUTION_SHARD_DIRECTORY}/{holder_job_id}"
        f"{HOLDER_DIRECTORY_COMPLETION_SUFFIX}"
    )


def build_directory_topology(
    manifest: Mapping[str, Any], *, input_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the exact pre-created directory ownership/closure contract."""
    value = validate_manifest(manifest, input_spec=input_spec)
    directories: dict[str, dict[str, Any]] = {}

    def add(relative: str, owner: str | None) -> None:
        path = Path(relative)
        if (
            path.is_absolute() or relative in ("", "/")
            or ".." in path.parts
        ):
            raise DecodedEvaluationPlanError("directory topology path differs")
        normalized = "." if relative == "." else path.as_posix()
        existing = directories.get(normalized)
        if existing is not None and existing["owner_holder_job_id"] != owner:
            raise DecodedEvaluationPlanError(
                f"directory topology owner differs: {normalized}"
            )
        directories[normalized] = {
            "relative_path": normalized,
            "owner_holder_job_id": owner,
            "expected_mode": 0o700,
            "expected_entries": [],
        }

    add(".", None)
    add(SHARD_DIRECTORY, None)
    add(OUTPUT_CANDIDATE_DIRECTORY, None)
    add(OUTPUT_BASE_DIRECTORY, None)
    add(EXECUTION_SHARD_DIRECTORY, None)
    for holder in HOLDER_ROWS:
        holder_job_id = holder["job_id"]
        holder_root = f"{EXECUTION_SHARD_DIRECTORY}/{holder_job_id}"
        task_parent = f"{holder_root}/tasks"
        add(holder_root, holder_job_id)
        add(
            f"{holder_root}/{CONSUMPTION_AUTHORITY_DIRECTORY}",
            holder_job_id,
        )
        add(task_parent, holder_job_id)
        for task in _shard_tasks(value, holder):
            record = task["record"]
            if task["task_kind"] == "adapter_candidate":
                task_id = record["candidate_id"]
            elif task["task_kind"] == "frozen_base_control":
                task_id = record["control_id"]
            else:
                raise DecodedEvaluationPlanError(
                    "directory topology task kind differs"
                )
            if (
                not isinstance(task_id, str)
                or _SAFE_ID.fullmatch(task_id) is None
            ):
                raise DecodedEvaluationPlanError(
                    "directory topology task id differs"
                )
            add(f"{task_parent}/{task_id}", holder_job_id)
    for record in value["candidates"]:
        owner = record["holder"]["job_id"]
        parent = Path(record["output_relpath"]).parent
        current = Path()
        for component in parent.parts:
            current /= component
            relative = current.as_posix()
            add(
                relative,
                owner if relative != OUTPUT_CANDIDATE_DIRECTORY else None,
            )
    for record in value["frozen_base_controls"]:
        owner = record["holder"]["job_id"]
        parent = Path(record["output_relpath"]).parent
        current = Path()
        for component in parent.parts:
            current /= component
            relative = current.as_posix()
            add(relative, owner if relative != OUTPUT_BASE_DIRECTORY else None)

    entry_sets: dict[str, set[str]] = {key: set() for key in directories}
    for relative in directories:
        if relative == ".":
            continue
        path = Path(relative)
        parent = "." if path.parent == Path(".") else path.parent.as_posix()
        if parent not in entry_sets:
            raise DecodedEvaluationPlanError(
                f"directory topology parent is absent: {relative}"
            )
        entry_sets[parent].add(path.name)
    entry_sets["."].update(
        {
            INPUT_FILENAME, MANIFEST_FILENAME, REVIEW_CONTRACT_FILENAME,
            PUBLICATION_FILENAME, DIRECTORY_AUTHORITY_FILENAME,
        }
    )
    entry_sets[SHARD_DIRECTORY].update(
        f"{holder['job_id']}.json" for holder in HOLDER_ROWS
    )
    entry_sets[EXECUTION_SHARD_DIRECTORY].update(
        Path(holder_completion_reservation_relative(holder["job_id"])).name
        for holder in HOLDER_ROWS
    )
    rows = []
    for relative in sorted(directories):
        row = dict(directories[relative])
        row["expected_entries"] = sorted(entry_sets[relative])
        rows.append(row)
    if len(rows) != len({row["relative_path"] for row in rows}):
        raise DecodedEvaluationPlanError("directory topology is not unique")
    owner_by_arm = {
        arm: _arm_holder(arm)["job_id"] for arm in ARMS
    }
    for row in rows:
        parts = Path(row["relative_path"]).parts
        if len(parts) >= 2 and parts[0] == OUTPUT_CANDIDATE_DIRECTORY:
            if row["owner_holder_job_id"] != owner_by_arm[parts[1]]:
                raise DecodedEvaluationPlanError(
                    "candidate directory ownership differs"
                )
    return rows


def _identity_row(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev, "inode": value.st_ino,
        "uid": value.st_uid, "gid": value.st_gid,
        "mode": value.st_mode, "nlink": value.st_nlink,
        "rdev": value.st_rdev, "size": value.st_size,
        "blocks": getattr(value, "st_blocks", 0),
        "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns,
    }


_IDENTITY_FIELDS = frozenset(
    {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
)


def build_holder_completion_reservations(
    *, evaluation_root: str | Path,
    identities: Mapping[str, tuple[os.stat_result, os.stat_result]] | None = None,
) -> list[dict[str, Any]]:
    root = _absolute_nonroot_path(str(evaluation_root), label="evaluation root")
    rows: list[dict[str, Any]] = []
    for holder in HOLDER_ROWS:
        holder_job_id = holder["job_id"]
        relative = holder_completion_reservation_relative(holder_job_id)
        pair = None if identities is None else identities.get(holder_job_id)
        if identities is not None and pair is None:
            raise DecodedEvaluationPlanError(
                f"holder completion reservation identity is absent: {holder_job_id}"
            )
        rows.append(
            {
                "holder_job_id": holder_job_id,
                "relative_path": relative,
                "path": str(Path(root) / relative),
                "sha256": _EMPTY_SHA256,
                "size": 0,
                "mode": HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE,
                "identity": None if pair is None else _identity_row(pair[0]),
                "parent_identity": (
                    None if pair is None else _identity_row(pair[1])
                ),
            }
        )
    return validate_holder_completion_reservations(
        rows,
        evaluation_root=root,
        materialized_required=identities is not None,
    )


def validate_holder_completion_reservations(
    value: Any, *, evaluation_root: str | Path,
    materialized_required: bool,
    directory_authority: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    root = Path(
        _absolute_nonroot_path(str(evaluation_root), label="evaluation root")
    )
    if not isinstance(value, list) or len(value) != len(HOLDER_ROWS):
        raise DecodedEvaluationPlanError(
            "holder completion reservation closure differs"
        )
    fields = {
        "holder_job_id", "relative_path", "path", "sha256", "size",
        "mode", "identity", "parent_identity",
    }
    parent_authority: Mapping[str, Any] | None = None
    if directory_authority is not None:
        candidates = [
            row for row in directory_authority.get("rows", [])
            if isinstance(row, Mapping)
            and row.get("relative_path") == EXECUTION_SHARD_DIRECTORY
        ]
        if len(candidates) != 1:
            raise DecodedEvaluationPlanError(
                "holder completion reservation parent authority differs"
            )
        parent_authority = candidates[0]
    result: list[dict[str, Any]] = []
    for raw, holder in zip(value, HOLDER_ROWS):
        row = dict(
            _closed(raw, fields, label="holder completion reservation")
        )
        holder_job_id = holder["job_id"]
        relative = holder_completion_reservation_relative(holder_job_id)
        identity = row["identity"]
        parent_identity = row["parent_identity"]
        if (
            row["holder_job_id"] != holder_job_id
            or row["relative_path"] != relative
            or row["path"] != str(root / relative)
            or row["sha256"] != _EMPTY_SHA256
            or row["size"] != 0
            or row["mode"] != HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE
        ):
            raise DecodedEvaluationPlanError(
                f"holder completion reservation binding differs: {holder_job_id}"
            )
        if materialized_required:
            for label, observed in (
                ("identity", identity), ("parent identity", parent_identity)
            ):
                if (
                    not isinstance(observed, Mapping)
                    or set(observed) != _IDENTITY_FIELDS
                    or any(
                        type(observed[key]) is not int or observed[key] < 0
                        for key in _IDENTITY_FIELDS
                    )
                ):
                    raise DecodedEvaluationPlanError(
                        f"holder completion reservation {label} differs"
                    )
            if (
                not stat.S_ISREG(identity["mode"])
                or stat.S_IMODE(identity["mode"])
                != HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE
                or identity["nlink"] != 1
                or identity["size"] != 0
                or not stat.S_ISDIR(parent_identity["mode"])
                or (
                    parent_authority is not None
                    and parent_identity != parent_authority.get("identity")
                )
            ):
                raise DecodedEvaluationPlanError(
                    "holder completion reservation physical authority differs"
                )
        elif identity is not None or parent_identity is not None:
            raise DecodedEvaluationPlanError(
                "unmaterialized holder completion reservation carries identity"
            )
        result.append(row)
    return result


def build_directory_authority(
    *, evaluation_root: str | Path, topology: Sequence[Mapping[str, Any]],
    identities: Mapping[str, tuple[os.stat_result, os.stat_result]] | None = None,
) -> dict[str, Any]:
    root = _absolute_nonroot_path(str(evaluation_root), label="evaluation root")
    rows: list[dict[str, Any]] = []
    for item in topology:
        relative = str(item["relative_path"])
        path = Path(root) if relative == "." else Path(root) / relative
        pair = None if identities is None else identities.get(relative)
        if identities is not None and pair is None:
            raise DecodedEvaluationPlanError(
                f"directory authority identity is absent: {relative}"
            )
        rows.append(
            {
                "relative_path": relative,
                "path": str(path),
                "owner_holder_job_id": item["owner_holder_job_id"],
                "expected_mode": item["expected_mode"],
                "expected_entries": list(item["expected_entries"]),
                "identity": None if pair is None else _identity_row(pair[0]),
                "parent_identity": (
                    None if pair is None else _identity_row(pair[1])
                ),
            }
        )
    value: dict[str, Any] = {
        "schema_version": DIRECTORY_AUTHORITY_SCHEMA,
        "evaluation_root": root,
        "materialized": identities is not None,
        "topology_digest": object_sha256(list(topology)),
        "rows": rows,
        "row_count": len(rows),
    }
    value["authority_digest"] = object_sha256(value)
    return validate_directory_authority(
        value, topology=topology, materialized_required=identities is not None
    )


def validate_directory_authority(
    value: Any, *, topology: Sequence[Mapping[str, Any]],
    materialized_required: bool,
) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_root", "materialized",
        "topology_digest", "rows", "row_count", "authority_digest",
    }
    row = dict(_closed(value, fields, label="directory authority"))
    _absolute_nonroot_path(row["evaluation_root"], label="directory authority root")
    if (
        row["schema_version"] != DIRECTORY_AUTHORITY_SCHEMA
        or type(row["materialized"]) is not bool
        or (materialized_required and row["materialized"] is not True)
        or row["topology_digest"] != object_sha256(list(topology))
        or row["row_count"] != len(topology)
    ):
        raise DecodedEvaluationPlanError("directory authority header differs")
    rows = row["rows"]
    if not isinstance(rows, list) or len(rows) != len(topology):
        raise DecodedEvaluationPlanError("directory authority row closure differs")
    expected_fields = {
        "relative_path", "path", "owner_holder_job_id", "expected_mode",
        "expected_entries", "identity", "parent_identity",
    }
    root = Path(row["evaluation_root"])
    for actual, expected in zip(rows, topology):
        item = dict(_closed(actual, expected_fields, label="directory authority row"))
        relative = expected["relative_path"]
        expected_path = root if relative == "." else root / relative
        if any(item[key] != expected[key] for key in (
            "relative_path", "owner_holder_job_id", "expected_mode",
            "expected_entries",
        )) or item["path"] != str(expected_path):
            raise DecodedEvaluationPlanError(
                f"directory authority topology differs: {relative}"
            )
        for identity_field in ("identity", "parent_identity"):
            identity = item[identity_field]
            if row["materialized"]:
                if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
                    raise DecodedEvaluationPlanError(
                        f"directory authority {identity_field} differs"
                    )
                if any(type(identity[key]) is not int or identity[key] < 0
                       for key in _IDENTITY_FIELDS):
                    raise DecodedEvaluationPlanError(
                        f"directory authority {identity_field} value differs"
                    )
                if (
                    identity_field == "identity"
                    and stat.S_IMODE(identity["mode"])
                    != item["expected_mode"]
                ):
                    raise DecodedEvaluationPlanError(
                        "directory authority mode projection differs"
                    )
            elif identity is not None:
                raise DecodedEvaluationPlanError(
                    "unmaterialized directory authority carries identity"
                )
    _verify_digest(row, field="authority_digest", label="directory authority")
    return row


def _holder_mutable_topology_rows(
    topology: Sequence[Mapping[str, Any]], holder_job_id: str,
) -> list[Mapping[str, Any]]:
    rows = [
        item for item in topology
        if item["owner_holder_job_id"] == holder_job_id
        and (
            item["expected_entries"] == []
            or item["relative_path"]
            == f"{EXECUTION_SHARD_DIRECTORY}/{holder_job_id}"
        )
    ]
    return sorted(rows, key=lambda item: item["relative_path"])


def validate_holder_directory_completion(
    value: Any, *, topology: Sequence[Mapping[str, Any]],
    base_directory_authority: Mapping[str, Any],
) -> dict[str, Any]:
    base = validate_directory_authority(
        base_directory_authority,
        topology=topology,
        materialized_required=True,
    )
    fields = {
        "schema_version", "evaluation_root", "base_authority_digest",
        "base_topology_digest", "holder_job_id", "holder_summary_digest",
        "rows", "row_count", "completion_digest",
    }
    row = dict(_closed(value, fields, label="holder directory completion"))
    holder_job_id = row["holder_job_id"]
    if holder_job_id not in {item["job_id"] for item in HOLDER_ROWS}:
        raise DecodedEvaluationPlanError(
            "holder directory completion holder differs"
        )
    expected_topology = _holder_mutable_topology_rows(
        topology, holder_job_id
    )
    expected_relatives = [item["relative_path"] for item in expected_topology]
    completion_rows = row["rows"]
    row_fields = {
        "relative_path", "path", "owner_holder_job_id", "expected_mode",
        "expected_entries", "identity", "parent_identity",
    }
    if (
        row["schema_version"] != HOLDER_DIRECTORY_COMPLETION_SCHEMA
        or row["evaluation_root"] != base["evaluation_root"]
        or row["base_authority_digest"] != base["authority_digest"]
        or row["base_topology_digest"] != base["topology_digest"]
        or not isinstance(completion_rows, list)
        or row["row_count"] != len(expected_relatives)
        or len(completion_rows) != len(expected_relatives)
        or [item.get("relative_path") for item in completion_rows]
        != expected_relatives
        or not isinstance(row["holder_summary_digest"], str)
        or _SHA256.fullmatch(row["holder_summary_digest"]) is None
    ):
        raise DecodedEvaluationPlanError(
            "holder directory completion header/row closure differs"
        )
    base_by_relative = {
        item["relative_path"]: item for item in base["rows"]
    }
    expected_by_relative = {
        item["relative_path"]: item for item in expected_topology
    }
    completion_by_relative = {
        item.get("relative_path"): item
        for item in completion_rows
        if isinstance(item, Mapping)
    }
    for raw in completion_rows:
        item = dict(
            _closed(
                raw, row_fields, label="holder directory completion row"
            )
        )
        relative = item["relative_path"]
        expected = expected_by_relative[relative]
        base_item = base_by_relative[relative]
        entries = item["expected_entries"]
        expected_path = Path(base["evaluation_root"]) / relative
        if (
            item["path"] != str(expected_path)
            or item["owner_holder_job_id"] != holder_job_id
            or item["expected_mode"] != expected["expected_mode"]
            or not isinstance(entries, list)
            or entries != sorted(set(entries))
            or not entries
            or any(
                type(name) is not str or name in ("", ".", "..")
                or os.path.sep in name
                for name in entries
            )
        ):
            raise DecodedEvaluationPlanError(
                f"holder directory completion row differs: {relative}"
            )
        if relative.startswith(
            (OUTPUT_CANDIDATE_DIRECTORY + "/", OUTPUT_BASE_DIRECTORY + "/")
        ) and entries != sorted(f"{policy}.mp4" for policy in POLICIES):
            raise DecodedEvaluationPlanError(
                f"holder output leaf closure differs: {relative}"
            )
        for field in ("identity", "parent_identity"):
            identity = item[field]
            if (
                not isinstance(identity, Mapping)
                or set(identity) != _IDENTITY_FIELDS
                or any(
                    type(identity[key]) is not int or identity[key] < 0
                    for key in _IDENTITY_FIELDS
                )
            ):
                raise DecodedEvaluationPlanError(
                    f"holder directory completion {field} differs"
                )
        immutable_fields = {"device", "inode", "uid", "gid", "mode", "rdev"}
        parent_path = Path(relative).parent
        parent_relative = (
            "." if parent_path == Path(".") else parent_path.as_posix()
        )
        mutable_parent = completion_by_relative.get(parent_relative)
        expected_parent_identity = (
            mutable_parent.get("identity")
            if isinstance(mutable_parent, Mapping)
            else base_item["parent_identity"]
        )
        if (
            {key: item["identity"][key] for key in immutable_fields}
            != {key: base_item["identity"][key] for key in immutable_fields}
            or item["parent_identity"] != expected_parent_identity
            or stat.S_IMODE(item["identity"]["mode"])
            != item["expected_mode"]
        ):
            raise DecodedEvaluationPlanError(
                f"holder directory completion identity differs: {relative}"
            )
    _verify_digest(
        row, field="completion_digest", label="holder directory completion"
    )
    return row


def merge_holder_directory_completions(
    *, topology: Sequence[Mapping[str, Any]],
    base_directory_authority: Mapping[str, Any],
    completions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    base = validate_directory_authority(
        base_directory_authority,
        topology=topology,
        materialized_required=True,
    )
    holder_ids = [item["job_id"] for item in HOLDER_ROWS]
    if not isinstance(completions, Mapping) or set(completions) != set(holder_ids):
        raise DecodedEvaluationPlanError(
            "holder directory completion set differs"
        )
    validated = {
        holder: validate_holder_directory_completion(
            completions[holder],
            topology=topology,
            base_directory_authority=base,
        )
        for holder in holder_ids
    }
    if any(validated[holder]["holder_job_id"] != holder for holder in holder_ids):
        raise DecodedEvaluationPlanError(
            "holder directory completion key differs"
        )
    updates = {
        item["relative_path"]: item
        for holder in holder_ids
        for item in validated[holder]["rows"]
    }
    expected_mutable = {
        item["relative_path"]
        for holder in holder_ids
        for item in _holder_mutable_topology_rows(topology, holder)
    }
    if set(updates) != expected_mutable or len(updates) != sum(
        validated[holder]["row_count"] for holder in holder_ids
    ):
        raise DecodedEvaluationPlanError(
            "holder directory completion union differs"
        )
    final_topology = []
    for item in topology:
        final_item = {
            "relative_path": item["relative_path"],
            "owner_holder_job_id": item["owner_holder_job_id"],
            "expected_mode": item["expected_mode"],
            "expected_entries": list(item["expected_entries"]),
        }
        update = updates.get(item["relative_path"])
        if update is not None:
            final_item["expected_entries"] = list(update["expected_entries"])
        final_topology.append(final_item)
    final_rows = []
    for base_item in base["rows"]:
        item = dict(base_item)
        update = updates.get(item["relative_path"])
        if update is not None:
            item["expected_entries"] = list(update["expected_entries"])
            item["identity"] = dict(update["identity"])
            item["parent_identity"] = dict(update["parent_identity"])
        final_rows.append(item)
    final_by_relative = {
        item["relative_path"]: item for item in final_rows
    }
    for item in final_rows:
        relative = item["relative_path"]
        if relative == ".":
            continue
        parent_path = Path(relative).parent
        parent_relative = (
            "." if parent_path == Path(".") else parent_path.as_posix()
        )
        parent = final_by_relative.get(parent_relative)
        if parent is None:
            raise DecodedEvaluationPlanError(
                f"final directory authority parent differs: {relative}"
            )
        item["parent_identity"] = dict(parent["identity"])
    final_authority: dict[str, Any] = {
        "schema_version": DIRECTORY_AUTHORITY_SCHEMA,
        "evaluation_root": base["evaluation_root"],
        "materialized": True,
        "topology_digest": object_sha256(final_topology),
        "rows": final_rows,
        "row_count": len(final_rows),
    }
    final_authority["authority_digest"] = object_sha256(final_authority)
    final_authority = validate_directory_authority(
        final_authority,
        topology=final_topology,
        materialized_required=True,
    )
    value: dict[str, Any] = {
        "schema_version": FINAL_DIRECTORY_MERGE_SCHEMA,
        "evaluation_root": base["evaluation_root"],
        "base_authority_digest": base["authority_digest"],
        "base_topology_digest": base["topology_digest"],
        "holder_completion_digests": {
            holder: validated[holder]["completion_digest"]
            for holder in holder_ids
        },
        "mutable_row_count": len(updates),
        "topology": final_topology,
        "topology_digest": object_sha256(final_topology),
        "directory_authority": final_authority,
    }
    value["merge_digest"] = object_sha256(value)
    return value


def _json_file_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest()


def build_publication_receipt(
    bundle: Mapping[str, Any],
    *, directory_authority: Mapping[str, Any] | None = None,
    holder_completion_reservations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    required = {"input_spec", "review_contract", "manifest", "shards"}
    if not isinstance(bundle, Mapping) or set(bundle) != required:
        raise DecodedEvaluationPlanError("evaluation bundle field closure differs")
    input_spec = validate_input_spec(bundle["input_spec"])
    review = validate_review_packet_contract(bundle["review_contract"])
    manifest = validate_manifest(bundle["manifest"], input_spec=input_spec)
    if manifest["review_packet_contract_digest"] != review["contract_digest"]:
        raise DecodedEvaluationPlanError("bundle review contract binding differs")
    shards_value = bundle["shards"]
    if not isinstance(shards_value, Mapping) or set(shards_value) != {
        holder["job_id"] for holder in HOLDER_ROWS
    }:
        raise DecodedEvaluationPlanError("bundle shard closure differs")
    shards = {
        holder["job_id"]: validate_shard(
            shards_value[holder["job_id"]],
            manifest=manifest,
            input_spec=input_spec,
        )
        for holder in HOLDER_ROWS
    }
    topology = build_directory_topology(manifest, input_spec=input_spec)
    authority = (
        build_directory_authority(
            evaluation_root=manifest["evaluation_root"],
            topology=topology,
        )
        if directory_authority is None
        else validate_directory_authority(
            directory_authority,
            topology=topology,
            materialized_required=True,
        )
    )
    if authority["evaluation_root"] != manifest["evaluation_root"]:
        raise DecodedEvaluationPlanError(
            "directory authority evaluation root differs"
        )
    if holder_completion_reservations is None:
        if authority["materialized"]:
            raise DecodedEvaluationPlanError(
                "materialized holder completion reservations are required"
            )
        completion_reservations = build_holder_completion_reservations(
            evaluation_root=manifest["evaluation_root"]
        )
    else:
        completion_reservations = validate_holder_completion_reservations(
            list(holder_completion_reservations),
            evaluation_root=manifest["evaluation_root"],
            materialized_required=authority["materialized"],
            directory_authority=authority,
        )
    authority_file_sha256 = hashlib.sha256(
        canonical_json_bytes(authority) + b"\n"
    ).hexdigest()
    files = [
        {"relpath": INPUT_FILENAME, "sha256": _json_file_sha(input_spec)},
        {"relpath": MANIFEST_FILENAME, "sha256": _json_file_sha(manifest)},
        {
            "relpath": REVIEW_CONTRACT_FILENAME,
            "sha256": _json_file_sha(review),
        },
    ]
    files.extend(
        {
            "relpath": f"{SHARD_DIRECTORY}/{holder['job_id']}.json",
            "sha256": _json_file_sha(shards[holder["job_id"]]),
        }
        for holder in HOLDER_ROWS
    )
    files.append(
        {
            "relpath": DIRECTORY_AUTHORITY_FILENAME,
            "sha256": authority_file_sha256,
        }
    )
    value = {
        "schema_version": PUBLICATION_SCHEMA,
        "evaluation_id": manifest["evaluation_id"],
        "evaluation_root": manifest["evaluation_root"],
        "input_digest": input_spec["input_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "review_packet_contract_digest": review["contract_digest"],
        "directory_topology": topology,
        "directory_topology_digest": object_sha256(topology),
        "directory_authority_file_sha256": authority_file_sha256,
        "directory_authority_digest": authority["authority_digest"],
        "directory_authority_materialized": authority["materialized"],
        "holder_completion_reservations": completion_reservations,
        "files": files,
        "payload_file_count": len(files),
        "candidate_video_plan_count": CANDIDATE_COUNT,
        "frozen_base_control_plan_count": BASE_CONTROL_COUNT,
        "local_plan_generation_only": True,
        "remote_launch_performed": False,
        "upload_performed": False,
        "create_only_publication": True,
    }
    value["publication_digest"] = object_sha256(value)
    return validate_publication_receipt(
        value,
        bundle=bundle,
        directory_authority=authority,
        verify_directory_authority=authority["materialized"],
    )


def validate_publication_receipt(
    value: Any, *, bundle: Mapping[str, Any],
    directory_authority: Mapping[str, Any] | None = None,
    verify_directory_authority: bool = False,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "evaluation_id",
        "evaluation_root",
        "input_digest",
        "manifest_digest",
        "review_packet_contract_digest",
        "directory_topology",
        "directory_topology_digest",
        "directory_authority_file_sha256",
        "directory_authority_digest",
        "directory_authority_materialized",
        "holder_completion_reservations",
        "files",
        "payload_file_count",
        "candidate_video_plan_count",
        "frozen_base_control_plan_count",
        "local_plan_generation_only",
        "remote_launch_performed",
        "upload_performed",
        "create_only_publication",
        "publication_digest",
    }
    row = dict(_closed(value, fields, label="publication receipt"))
    if row["schema_version"] != PUBLICATION_SCHEMA:
        raise DecodedEvaluationPlanError("publication schema differs")
    input_spec = validate_input_spec(bundle["input_spec"])
    review = validate_review_packet_contract(bundle["review_contract"])
    manifest = validate_manifest(bundle["manifest"], input_spec=input_spec)
    topology = build_directory_topology(manifest, input_spec=input_spec)
    if row["directory_topology"] != topology:
        raise DecodedEvaluationPlanError("publication directory topology differs")
    topology_digest = _sha(
        row["directory_topology_digest"],
        label="publication directory topology digest",
    )
    if topology_digest != object_sha256(topology):
        raise DecodedEvaluationPlanError(
            "publication directory topology digest differs"
        )
    authority_file_sha256 = _sha(
        row["directory_authority_file_sha256"],
        label="publication directory authority file SHA",
    )
    authority_digest = _sha(
        row["directory_authority_digest"],
        label="publication directory authority digest",
    )
    if type(row["directory_authority_materialized"]) is not bool:
        raise DecodedEvaluationPlanError(
            "publication directory authority materialization differs"
        )
    if row["directory_authority_materialized"] and (
        directory_authority is None or not verify_directory_authority
    ):
        raise DecodedEvaluationPlanError(
            "materialized directory authority verification is required"
        )
    if verify_directory_authority and directory_authority is None:
        raise DecodedEvaluationPlanError(
            "materialized directory authority is required"
        )
    if directory_authority is not None:
        authority = validate_directory_authority(
            directory_authority,
            topology=topology,
            materialized_required=verify_directory_authority,
        )
        observed_file_sha256 = hashlib.sha256(
            canonical_json_bytes(authority) + b"\n"
        ).hexdigest()
        if (
            authority["evaluation_root"] != manifest["evaluation_root"]
            or authority["authority_digest"] != authority_digest
            or authority["materialized"]
            is not row["directory_authority_materialized"]
            or observed_file_sha256 != authority_file_sha256
        ):
            raise DecodedEvaluationPlanError(
                "publication directory authority binding differs"
            )
    elif verify_directory_authority:
        raise DecodedEvaluationPlanError(
            "publication directory authority is absent"
        )
    completion_reservations = validate_holder_completion_reservations(
        row["holder_completion_reservations"],
        evaluation_root=manifest["evaluation_root"],
        materialized_required=row["directory_authority_materialized"],
        directory_authority=directory_authority,
    )
    expected_files = [
        {"relpath": INPUT_FILENAME, "sha256": _json_file_sha(input_spec)},
        {"relpath": MANIFEST_FILENAME, "sha256": _json_file_sha(manifest)},
        {"relpath": REVIEW_CONTRACT_FILENAME, "sha256": _json_file_sha(review)},
    ]
    for holder in HOLDER_ROWS:
        shard = validate_shard(
            bundle["shards"][holder["job_id"]],
            manifest=manifest,
            input_spec=input_spec,
        )
        expected_files.append(
            {
                "relpath": f"{SHARD_DIRECTORY}/{holder['job_id']}.json",
                "sha256": _json_file_sha(shard),
            }
        )
    expected_files.append(
        {
            "relpath": DIRECTORY_AUTHORITY_FILENAME,
            "sha256": authority_file_sha256,
        }
    )
    expected_scalar = {
        "evaluation_id": manifest["evaluation_id"],
        "evaluation_root": manifest["evaluation_root"],
        "input_digest": input_spec["input_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "review_packet_contract_digest": review["contract_digest"],
        "directory_topology": topology,
        "directory_topology_digest": topology_digest,
        "directory_authority_file_sha256": authority_file_sha256,
        "directory_authority_digest": authority_digest,
        "directory_authority_materialized": row[
            "directory_authority_materialized"
        ],
        "holder_completion_reservations": completion_reservations,
        "files": expected_files,
        "payload_file_count": 8,
        "candidate_video_plan_count": CANDIDATE_COUNT,
        "frozen_base_control_plan_count": BASE_CONTROL_COUNT,
        "local_plan_generation_only": True,
        "remote_launch_performed": False,
        "upload_performed": False,
        "create_only_publication": True,
    }
    for key, expected in expected_scalar.items():
        if row[key] != expected:
            raise DecodedEvaluationPlanError(f"publication binding differs: {key}")
    _verify_digest(row, field="publication_digest", label="publication receipt")
    return row


def _plain_directory(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise DecodedEvaluationPlanError(f"{label} does not exist") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise DecodedEvaluationPlanError(f"{label} is not a plain directory")


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid,
        value.st_mode, value.st_nlink, value.st_rdev, value.st_size,
        getattr(value, "st_blocks", 0), value.st_mtime_ns, value.st_ctime_ns,
    )


def _immutable_directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid,
        stat.S_IFMT(value.st_mode), value.st_rdev,
    )


_IMMUTABLE_DIRECTORY_FIELDS = frozenset(
    {"device", "inode", "uid", "gid", "mode", "rdev"}
)


def _immutable_directory_row(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev, "inode": value.st_ino,
        "uid": value.st_uid, "gid": value.st_gid,
        "mode": value.st_mode, "rdev": value.st_rdev,
    }


def _validate_immutable_directory_row(
    value: Any, *, label: str,
) -> dict[str, int]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _IMMUTABLE_DIRECTORY_FIELDS
        or any(
            type(value[field]) is not int or value[field] < 0
            for field in _IMMUTABLE_DIRECTORY_FIELDS
        )
        or not stat.S_ISDIR(value["mode"])
    ):
        raise DecodedEvaluationPlanError(f"{label} differs")
    return dict(value)


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


class RetainedPublicationRoot:
    """One fresh root whose entire publication stays below retained dirfds."""

    def __init__(
        self, *, path: Path, label: str, error_type: type[RuntimeError],
        barrier: Any, parent_fd: int, root_fd: int,
        parent_anchor: os.stat_result, root_creation: os.stat_result,
        parent_parent_fd: int | None = None,
        parent_immutable_identity: Mapping[str, int] | None = None,
        parent_parent_immutable_identity: Mapping[str, int] | None = None,
    ) -> None:
        self.path = path
        self.label = label
        self.error_type = error_type
        self.barrier = barrier
        self.parent_fd = parent_fd
        self.parent_anchor = parent_anchor
        self.parent_parent_fd = parent_parent_fd
        self.parent_immutable_identity = (
            _immutable_directory_row(parent_anchor)
            if parent_immutable_identity is None
            else dict(parent_immutable_identity)
        )
        self.parent_parent_immutable_identity = (
            None
            if parent_parent_immutable_identity is None
            else dict(parent_parent_immutable_identity)
        )
        self.states: dict[str, dict[str, Any]] = {
            ".": {
                "fd": root_fd, "parent": None, "name": path.name,
                "creation": root_creation, "anchor": root_creation,
                "entries": set(), "mode": 0o700,
                "foreign_mutable": False,
            }
        }
        self.reservations: list[dict[str, Any]] = []
        self.captures: list[dict[str, Any]] = []
        self.owner_holder_job_id: str | None = None
        self.materialized_topology: dict[str, Mapping[str, Any]] = {}
        self.closed = False

    def _fail(self, message: str) -> None:
        raise self.error_type(f"{self.label}: {message}")

    @classmethod
    def create(
        cls, path: str | Path, *, label: str = "publication root",
        error_type: type[RuntimeError] = DecodedEvaluationPlanError,
        barrier: Any = None,
        retained_parent_fd: int | None = None,
        retained_parent_parent_fd: int | None = None,
        expected_parent_immutable_identity: Mapping[str, int] | None = None,
        expected_parent_parent_immutable_identity: Mapping[str, int] | None = None,
    ) -> "RetainedPublicationRoot":
        root = Path(path)
        if (
            not root.is_absolute() or str(root) == os.path.sep
            or os.path.normpath(str(root)) != str(root)
            or root.name in ("", ".", "..")
            or not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")
        ):
            raise error_type(f"{label}: canonical root path differs")
        parent = root.parent
        parent_fd: int | None = None
        parent_parent_fd: int | None = None
        if retained_parent_fd is None:
            if any(
                value is not None
                for value in (
                    retained_parent_parent_fd,
                    expected_parent_immutable_identity,
                    expected_parent_parent_immutable_identity,
                )
            ):
                raise error_type(f"{label}: retained parent authority differs")
            try:
                if parent.resolve(strict=True) != parent:
                    raise error_type(f"{label}: canonical parent differs")
                parent_fd = os.open(
                    parent,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                )
            except (FileNotFoundError, OSError) as error:
                raise error_type(
                    f"{label}: cannot capture canonical parent"
                ) from error
        else:
            if (
                type(retained_parent_fd) is not int
                or retained_parent_fd < 3
                or type(retained_parent_parent_fd) is not int
                or retained_parent_parent_fd < 3
                or retained_parent_fd == retained_parent_parent_fd
            ):
                raise error_type(f"{label}: retained parent FD differs")
            try:
                parent_identity = _validate_immutable_directory_row(
                    expected_parent_immutable_identity,
                    label=f"{label} expected parent immutable identity",
                )
                parent_parent_identity = _validate_immutable_directory_row(
                    expected_parent_parent_immutable_identity,
                    label=f"{label} expected parent-parent immutable identity",
                )
                parent_fd = os.dup(retained_parent_fd)
                parent_parent_fd = os.dup(retained_parent_parent_fd)
                os.set_inheritable(parent_fd, False)
                os.set_inheritable(parent_parent_fd, False)
            except (DecodedEvaluationPlanError, OSError) as error:
                if parent_fd is not None:
                    os.close(parent_fd)
                if parent_parent_fd is not None:
                    os.close(parent_parent_fd)
                raise error_type(
                    f"{label}: cannot duplicate retained parent FD"
                ) from error
        assert parent_fd is not None
        root_fd: int | None = None
        try:
            before_parent = os.fstat(parent_fd)
            if parent_parent_fd is None:
                named_parent = parent.lstat()
                parent_replay_differs = (
                    stat.S_ISLNK(named_parent.st_mode)
                    or _stat_identity(before_parent) != _stat_identity(named_parent)
                )
                parent_identity = _immutable_directory_row(before_parent)
                parent_parent_identity = None
            else:
                named_parent = os.stat(
                    parent.name,
                    dir_fd=parent_parent_fd,
                    follow_symlinks=False,
                )
                parent_replay_differs = (
                    _stat_identity(before_parent) != _stat_identity(named_parent)
                    or _immutable_directory_row(before_parent) != parent_identity
                    or _immutable_directory_row(os.fstat(parent_parent_fd))
                    != parent_parent_identity
                )
            if not stat.S_ISDIR(before_parent.st_mode) or parent_replay_differs:
                raise error_type(f"{label}: parent identity differs")
            try:
                os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise error_type(f"{label} is not fresh: {root}")
            try:
                os.mkdir(root.name, 0o700, dir_fd=parent_fd)
            except FileExistsError as error:
                raise error_type(f"{label} is not fresh: {root}") from error
            root_fd = os.open(
                root.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            os.set_inheritable(parent_fd, False)
            os.set_inheritable(root_fd, False)
            os.fchmod(root_fd, 0o700)
            os.fsync(root_fd)
            os.fsync(parent_fd)
            parent_anchor = os.fstat(parent_fd)
            root_creation = os.fstat(root_fd)
            named_root = os.stat(
                root.name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                (
                    parent_parent_fd is None
                    and _stat_identity(parent_anchor)
                    != _stat_identity(parent.lstat())
                )
                or (
                    parent_parent_fd is not None
                    and _stat_identity(parent_anchor)
                    != _stat_identity(
                        os.stat(
                            parent.name,
                            dir_fd=parent_parent_fd,
                            follow_symlinks=False,
                        )
                    )
                )
                or _stat_identity(root_creation) != _stat_identity(named_root)
                or not stat.S_ISDIR(root_creation.st_mode)
                or stat.S_IMODE(root_creation.st_mode) != 0o700
                or os.listdir(root_fd) != []
            ):
                raise error_type(f"{label}: fresh root replay differs")
            authority = cls(
                path=root, label=label, error_type=error_type,
                barrier=barrier, parent_fd=parent_fd, root_fd=root_fd,
                parent_anchor=parent_anchor, root_creation=root_creation,
                parent_parent_fd=parent_parent_fd,
                parent_immutable_identity=parent_identity,
                parent_parent_immutable_identity=parent_parent_identity,
            )
            authority._call_barrier("after-root-create", ".")
            authority._check_all()
            return authority
        except BaseException:
            if root_fd is not None:
                os.close(root_fd)
            os.close(parent_fd)
            if parent_parent_fd is not None:
                os.close(parent_parent_fd)
            raise

    @classmethod
    def open_materialized(
        cls, path: str | Path, *, directory_authority: Mapping[str, Any],
        topology: Sequence[Mapping[str, Any]], holder_job_id: str | None,
        label: str = "materialized publication root",
        error_type: type[RuntimeError] = DecodedEvaluationPlanError,
        barrier: Any = None,
        retained_parent_fd: int | None = None,
        retained_parent_parent_fd: int | None = None,
        expected_parent_immutable_identity: Mapping[str, int] | None = None,
        expected_parent_parent_immutable_identity: Mapping[str, int] | None = None,
        expected_root_authority: Mapping[str, Any] | None = None,
    ) -> "RetainedPublicationRoot":
        root = Path(path)
        if holder_job_id is not None and holder_job_id not in {
            row["job_id"] for row in HOLDER_ROWS
        }:
            raise error_type(f"{label}: holder job ID differs")
        validated = validate_directory_authority(
            directory_authority,
            topology=topology,
            materialized_required=True,
        )
        if validated["evaluation_root"] != str(root):
            raise error_type(f"{label}: authority root differs")
        if (
            not root.is_absolute() or str(root) == os.path.sep
            or os.path.normpath(str(root)) != str(root)
            or root.name in ("", ".", "..")
            or not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")
        ):
            raise error_type(f"{label}: canonical path differs")
        flags = (
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            | os.O_NOFOLLOW
        )
        parent_fd: int | None = None
        parent_parent_fd: int | None = None
        if retained_parent_fd is None:
            if any(
                value is not None
                for value in (
                    retained_parent_parent_fd,
                    expected_parent_immutable_identity,
                    expected_parent_parent_immutable_identity,
                )
            ):
                raise error_type(f"{label}: retained parent authority differs")
            if root.parent.resolve(strict=True) != root.parent:
                raise error_type(f"{label}: canonical parent differs")
            parent_fd = os.open(root.parent, flags)
        else:
            if (
                type(retained_parent_fd) is not int
                or retained_parent_fd < 3
                or type(retained_parent_parent_fd) is not int
                or retained_parent_parent_fd < 3
                or retained_parent_fd == retained_parent_parent_fd
            ):
                raise error_type(f"{label}: retained parent FD differs")
            try:
                parent_identity = _validate_immutable_directory_row(
                    expected_parent_immutable_identity,
                    label=f"{label} expected parent immutable identity",
                )
                parent_parent_identity = _validate_immutable_directory_row(
                    expected_parent_parent_immutable_identity,
                    label=f"{label} expected parent-parent immutable identity",
                )
                parent_fd = os.dup(retained_parent_fd)
                parent_parent_fd = os.dup(retained_parent_parent_fd)
                os.set_inheritable(parent_fd, False)
                os.set_inheritable(parent_parent_fd, False)
            except (DecodedEvaluationPlanError, OSError) as error:
                if parent_fd is not None:
                    os.close(parent_fd)
                if parent_parent_fd is not None:
                    os.close(parent_parent_fd)
                raise error_type(
                    f"{label}: cannot duplicate retained parent authority"
                ) from error
        assert parent_fd is not None
        root_fd: int | None = None
        opened: list[int] = []
        try:
            os.set_inheritable(parent_fd, False)
            parent_now = os.fstat(parent_fd)
            root_row = next(
                item for item in validated["rows"]
                if item["relative_path"] == "."
            )
            expected_parent = root_row["parent_identity"]
            if parent_parent_fd is None:
                named_parent = root.parent.lstat()
                parent_replay_differs = (
                    _stat_identity(parent_now) != _stat_identity(named_parent)
                )
                parent_identity = _immutable_directory_row(parent_now)
                parent_parent_identity = None
            else:
                named_parent = os.stat(
                    root.parent.name,
                    dir_fd=parent_parent_fd,
                    follow_symlinks=False,
                )
                parent_replay_differs = (
                    _stat_identity(parent_now) != _stat_identity(named_parent)
                    or _immutable_directory_row(parent_now) != parent_identity
                    or _immutable_directory_row(os.fstat(parent_parent_fd))
                    != parent_parent_identity
                )
            if (
                parent_replay_differs
                or any(
                    _identity_row(parent_now)[key] != expected_parent[key]
                    for key in ("device", "inode", "uid", "gid", "mode", "rdev")
                )
            ):
                raise error_type(f"{label}: immutable parent authority differs")
            root_fd = os.open(root.name, flags, dir_fd=parent_fd)
            os.set_inheritable(root_fd, False)
            root_now = os.fstat(root_fd)
            if _identity_row(root_now) != root_row["identity"]:
                raise error_type(f"{label}: root identity differs")
            if expected_root_authority is not None:
                root_authority_fields = {
                    "schema_version", "path", "identity", "parent_identity",
                    "entries", "retained_parent_fd", "retained_root_fd",
                }
                if (
                    not isinstance(expected_root_authority, Mapping)
                    or set(expected_root_authority) != root_authority_fields
                    or expected_root_authority["schema_version"]
                    != "bernini-retained-directory-authority-v1"
                    or expected_root_authority["path"] != str(root)
                    or expected_root_authority["identity"] != root_row["identity"]
                    or expected_root_authority["parent_identity"]
                    != root_row["parent_identity"]
                    or expected_root_authority["entries"]
                    != root_row["expected_entries"]
                    or expected_root_authority["retained_parent_fd"] is not True
                    or expected_root_authority["retained_root_fd"] is not True
                ):
                    raise error_type(f"{label}: expected root authority differs")
            authority = cls(
                path=root, label=label, error_type=error_type,
                barrier=barrier, parent_fd=parent_fd, root_fd=root_fd,
                parent_anchor=parent_now, root_creation=root_now,
                parent_parent_fd=parent_parent_fd,
                parent_immutable_identity=parent_identity,
                parent_parent_immutable_identity=parent_parent_identity,
            )
            authority.owner_holder_job_id = holder_job_id
            authority.materialized_topology = {
                item["relative_path"]: item for item in topology
            }
            authority.states["."]["entries"] = set(
                root_row["expected_entries"]
            )
            authority.states["."]["mode"] = root_row["expected_mode"]
            row_by_relative = {
                item["relative_path"]: item for item in validated["rows"]
            }
            for relative in sorted(
                (key for key in row_by_relative if key != "."),
                key=lambda item: (item.count("/"), item),
            ):
                row = row_by_relative[relative]
                relative_path = Path(relative)
                parent_key = (
                    "." if relative_path.parent == Path(".")
                    else relative_path.parent.as_posix()
                )
                parent_state = authority.states.get(parent_key)
                if parent_state is None:
                    raise error_type(f"{label}: retained parent is absent")
                descriptor = os.open(
                    relative_path.name, flags, dir_fd=parent_state["fd"]
                )
                opened.append(descriptor)
                os.set_inheritable(descriptor, False)
                observed = os.fstat(descriptor)
                first_entries = os.listdir(descriptor)
                middle = os.fstat(descriptor)
                second_entries = os.listdir(descriptor)
                after = os.fstat(descriptor)
                if (
                    _identity_row(observed) != row["identity"]
                    or _stat_identity(observed) != _stat_identity(middle)
                    or _stat_identity(observed) != _stat_identity(after)
                    or _identity_row(os.fstat(parent_state["fd"]))
                    != row["parent_identity"]
                    or sorted(first_entries) != sorted(second_entries)
                    or sorted(first_entries)
                    != sorted(row["expected_entries"])
                    or len(first_entries) != len(row["expected_entries"])
                ):
                    raise error_type(
                        f"{label}: directory identity differs: {relative}"
                    )
                authority.states[relative] = {
                    "fd": descriptor, "parent": parent_key,
                    "name": relative_path.name, "creation": observed,
                    "anchor": observed,
                    "entries": set(row["expected_entries"]),
                    "mode": row["expected_mode"],
                    "foreign_mutable": (
                        holder_job_id is not None
                        and row["owner_holder_job_id"] is not None
                        and row["owner_holder_job_id"] != holder_job_id
                    ),
                }
            authority._call_barrier("after-materialized-open", ".")
            authority._check_all()
            opened.clear()
            root_fd = None
            return authority
        except BaseException:
            for descriptor in reversed(opened):
                os.close(descriptor)
            if root_fd is not None:
                os.close(root_fd)
            os.close(parent_fd)
            if parent_parent_fd is not None:
                os.close(parent_parent_fd)
            raise

    def _call_barrier(self, event: str, relative: str) -> None:
        if self.barrier is not None:
            self.barrier(event, self.path, relative)

    def _require_mutable(self, relative: str) -> None:
        if not self.materialized_topology:
            return
        if self.owner_holder_job_id is None:
            self._fail("materialized audit authority is read-only")
        topology = self.materialized_topology.get(relative)
        owner = None if topology is None else topology["owner_holder_job_id"]
        if owner != self.owner_holder_job_id:
            self._fail(f"holder does not own directory: {relative}")

    @staticmethod
    def _relative(value: str | Path, *, allow_root: bool = False) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise DecodedEvaluationPlanError("publication relative path escapes root")
        normalized = path.as_posix()
        if normalized in ("", "."):
            if allow_root:
                return "."
            raise DecodedEvaluationPlanError("publication leaf path differs")
        if any(part in ("", ".", "..") for part in path.parts):
            raise DecodedEvaluationPlanError("publication relative path differs")
        return normalized

    def _named_stat(self, relative: str) -> os.stat_result:
        state = self.states[relative]
        if relative == ".":
            return os.stat(
                self.path.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        parent = self.states[state["parent"]]
        return os.stat(
            state["name"], dir_fd=parent["fd"], follow_symlinks=False
        )

    def _check_external_parent(self) -> os.stat_result:
        observed = os.fstat(self.parent_fd)
        if self.parent_parent_fd is None:
            named = self.path.parent.lstat()
            parent_parent_differs = False
        else:
            named = os.stat(
                self.path.parent.name,
                dir_fd=self.parent_parent_fd,
                follow_symlinks=False,
            )
            parent_parent_differs = (
                self.parent_parent_immutable_identity is None
                or _immutable_directory_row(os.fstat(self.parent_parent_fd))
                != self.parent_parent_immutable_identity
            )
        if (
            _stat_identity(observed) != _stat_identity(self.parent_anchor)
            or _stat_identity(observed) != _stat_identity(named)
            or _immutable_directory_row(observed)
            != self.parent_immutable_identity
            or parent_parent_differs
        ):
            self._fail("retained parent identity differs")
        return observed

    def _observe_state(self, relative: str) -> os.stat_result:
        state = self.states[relative]
        before = os.fstat(state["fd"])
        first = os.listdir(state["fd"])
        middle = os.fstat(state["fd"])
        second = os.listdir(state["fd"])
        after = os.fstat(state["fd"])
        named = self._named_stat(relative)
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_IMODE(before.st_mode) != state["mode"]
            or _stat_identity(before) != _stat_identity(middle)
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(named)
            or _immutable_directory_identity(before)
            != _immutable_directory_identity(state["creation"])
            or sorted(first) != sorted(second)
            or (
                not state["foreign_mutable"]
                and (
                    sorted(first) != sorted(state["entries"])
                    or len(first) != len(state["entries"])
                )
            )
        ):
            self._fail(f"retained directory identity/closure differs: {relative}")
        return before

    def _check_all(self) -> None:
        if self.closed:
            self._fail("retained root is closed")
        self._check_external_parent()
        for relative in sorted(self.states, key=lambda item: (item.count("/"), item)):
            observed = self._observe_state(relative)
            if (
                not self.states[relative]["foreign_mutable"]
                and _stat_identity(observed) != _stat_identity(
                    self.states[relative]["anchor"]
                )
            ):
                self._fail(f"retained directory drifted: {relative}")
        for file_handle in (*self.reservations, *self.captures):
            self._observe_reservation(file_handle)

    def _observe_reservation(
        self, reservation: Mapping[str, Any],
    ) -> os.stat_result:
        descriptor = reservation["fd"]
        parent = self.states[reservation["parent"]]
        before = os.fstat(descriptor)
        first = _read_descriptor(descriptor)
        middle = os.fstat(descriptor)
        second = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(
            reservation["name"],
            dir_fd=parent["fd"],
            follow_symlinks=False,
        )
        expected_size = reservation["size"]
        expected_sha256 = reservation["sha256"]
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != reservation["mode"]
            or _stat_identity(before) != _stat_identity(middle)
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(named)
            or _stat_identity(before)
            != _stat_identity(reservation["anchor"])
            or first != second
            or len(first) != expected_size
            or hashlib.sha256(first).hexdigest() != expected_sha256
        ):
            self._fail(
                f"retained file identity/closure differs: "
                f"{reservation['relative']}"
            )
        return before

    def _refresh(self, relative: str) -> None:
        self._check_external_parent()
        observed = self._observe_state(relative)
        self.states[relative]["anchor"] = observed
        for other in self.states:
            if other == relative:
                continue
            current = self._observe_state(other)
            if (
                not self.states[other]["foreign_mutable"]
                and _stat_identity(current) != _stat_identity(
                    self.states[other]["anchor"]
                )
            ):
                self._fail(f"unowned directory mutation differs: {other}")
        for file_handle in (*self.reservations, *self.captures):
            self._observe_reservation(file_handle)

    def create_directory(self, relative: str | Path) -> None:
        leaf = self._relative(relative)
        if leaf in self.states:
            self._fail(f"directory already owned: {leaf}")
        path = Path(leaf)
        parent_key = "." if path.parent == Path(".") else path.parent.as_posix()
        if parent_key not in self.states:
            self._fail(f"directory parent is not retained: {leaf}")
        self._require_mutable(parent_key)
        self._check_all()
        parent = self.states[parent_key]
        descriptor: int | None = None
        try:
            os.mkdir(path.name, 0o700, dir_fd=parent["fd"])
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent["fd"],
            )
            os.set_inheritable(descriptor, False)
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
            os.fsync(parent["fd"])
            creation = os.fstat(descriptor)
            parent["entries"].add(path.name)
            self.states[leaf] = {
                "fd": descriptor, "parent": parent_key, "name": path.name,
                "creation": creation, "anchor": creation, "entries": set(),
                "mode": 0o700, "foreign_mutable": False,
            }
            if self.owner_holder_job_id is not None:
                self.materialized_topology[leaf] = {
                    "relative_path": leaf,
                    "owner_holder_job_id": self.owner_holder_job_id,
                    "expected_mode": 0o700,
                    "expected_entries": [],
                }
            descriptor = None
            self._call_barrier("after-directory-create", leaf)
            self._refresh(parent_key)
        except FileExistsError as error:
            self._fail(f"directory collision: {leaf}")
            raise AssertionError from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def reserve_file(
        self, relative: str | Path, *, mode: int = 0o400,
    ) -> dict[str, Any]:
        leaf = self._relative(relative)
        path = Path(leaf)
        parent_key = "." if path.parent == Path(".") else path.parent.as_posix()
        if parent_key not in self.states:
            self._fail(f"file parent is not retained: {leaf}")
        self._require_mutable(parent_key)
        self._check_all()
        parent = self.states[parent_key]
        flags = (
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor: int | None = None
        reservation: dict[str, Any] | None = None
        registered = False
        try:
            try:
                descriptor = os.open(
                    path.name, flags, mode, dir_fd=parent["fd"]
                )
            except FileExistsError as error:
                raise self.error_type(
                    f"{self.label}: refusing file collision: {leaf}"
                ) from error
            os.set_inheritable(descriptor, False)
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            os.fsync(parent["fd"])
            parent["entries"].add(path.name)
            reservation = {
                "relative": leaf, "parent": parent_key, "name": path.name,
                "fd": descriptor, "mode": mode, "filled": False,
                "anchor": os.fstat(descriptor), "size": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
            self.reservations.append(reservation)
            registered = True
            self._call_barrier("after-file-reserve", leaf)
            self._refresh(parent_key)
            descriptor = None
            return reservation
        except BaseException:
            if registered:
                assert reservation is not None
                self.reservations.remove(reservation)
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def holder_completion_reservation_rows(
        self, reservations: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        holder_ids = [holder["job_id"] for holder in HOLDER_ROWS]
        if not isinstance(reservations, Mapping) or set(reservations) != set(
            holder_ids
        ):
            self._fail("holder completion reservation set differs")
        self._check_all()
        identities: dict[str, tuple[os.stat_result, os.stat_result]] = {}
        for holder_job_id in holder_ids:
            reservation = reservations[holder_job_id]
            expected_relative = holder_completion_reservation_relative(
                holder_job_id
            )
            if (
                reservation not in self.reservations
                or reservation["relative"] != expected_relative
                or reservation["filled"]
                or reservation["mode"]
                != HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE
            ):
                self._fail(
                    f"holder completion reservation differs: {holder_job_id}"
                )
            observed = self._observe_reservation(reservation)
            parent = os.fstat(self.states[reservation["parent"]]["fd"])
            identities[holder_job_id] = (observed, parent)
        return build_holder_completion_reservations(
            evaluation_root=self.path, identities=identities
        )

    def fill_reserved(
        self, reservation: Mapping[str, Any], payload: bytes,
        *, final_mode: int | None = None,
    ) -> dict[str, Any]:
        if reservation not in self.reservations or reservation["filled"]:
            self._fail("file reservation differs")
        sealed_mode = reservation["mode"] if final_mode is None else final_mode
        if type(sealed_mode) is not int or sealed_mode not in {0o400, 0o444}:
            self._fail("reserved file final mode differs")
        self._check_all()
        descriptor = reservation["fd"]
        before_empty = os.fstat(descriptor)
        if before_empty.st_nlink != 1 or before_empty.st_size != 0:
            self._fail(f"reserved file is not empty: {reservation['relative']}")
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                self._fail("create-only file write made no progress")
            offset += count
        os.fchmod(descriptor, sealed_mode)
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        first = _read_descriptor(descriptor)
        middle = os.fstat(descriptor)
        second = _read_descriptor(descriptor)
        self._call_barrier("after-file-write", reservation["relative"])
        after = os.fstat(descriptor)
        parent = self.states[reservation["parent"]]
        named = os.stat(
            reservation["name"], dir_fd=parent["fd"], follow_symlinks=False
        )
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != sealed_mode
            or _stat_identity(before) != _stat_identity(middle)
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(named)
            or first != payload or second != payload
        ):
            self._fail(f"same-FD file replay differs: {reservation['relative']}")
        reservation["filled"] = True
        reservation["mode"] = sealed_mode
        reservation["anchor"] = before
        reservation["size"] = len(payload)
        reservation["sha256"] = hashlib.sha256(payload).hexdigest()
        self._check_all()
        identity = _identity_row(before)
        identity["mode"] = stat.S_IMODE(before.st_mode)
        return {
            "path": str(self.path / reservation["relative"]),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload), **identity,
        }

    def capture_holder_completion_reservation(
        self, publication_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Capture this holder's pre-created empty completion inode.

        The caller must pass the already bundle-validated publication receipt.
        This method additionally replays its digest and exact reservation rows,
        then retains the writable descriptor until this authority is closed.
        """
        holder = self.owner_holder_job_id
        if holder is None or not self.materialized_topology:
            self._fail("holder completion capture requires materialized owner")
        if (
            not isinstance(publication_receipt, Mapping)
            or publication_receipt.get("schema_version") != PUBLICATION_SCHEMA
            or publication_receipt.get("evaluation_root") != str(self.path)
            or publication_receipt.get("directory_authority_materialized")
            is not True
        ):
            self._fail("holder completion publication receipt differs")
        try:
            _verify_digest(
                publication_receipt,
                field="publication_digest",
                label="publication receipt",
            )
            rows = validate_holder_completion_reservations(
                publication_receipt.get("holder_completion_reservations"),
                evaluation_root=self.path,
                materialized_required=True,
            )
        except DecodedEvaluationPlanError as error:
            self._fail(str(error))
            raise AssertionError from error
        expected = next(
            row for row in rows if row["holder_job_id"] == holder
        )
        leaf = expected["relative_path"]
        if any(
            item["relative"] == leaf
            for item in (*self.reservations, *self.captures)
        ):
            self._fail("holder completion reservation is already captured")
        path = Path(leaf)
        parent_key = (
            "." if path.parent == Path(".") else path.parent.as_posix()
        )
        if parent_key != EXECUTION_SHARD_DIRECTORY or parent_key not in self.states:
            self._fail("holder completion reservation parent differs")
        self._check_all()
        parent = self.states[parent_key]
        descriptor: int | None = None
        reservation: dict[str, Any] | None = None
        try:
            descriptor = os.open(
                path.name,
                os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent["fd"],
            )
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first = _read_descriptor(descriptor)
            middle = os.fstat(descriptor)
            second = _read_descriptor(descriptor)
            named = os.stat(
                path.name, dir_fd=parent["fd"], follow_symlinks=False
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode)
                != HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE
                or _stat_identity(before) != _stat_identity(middle)
                or _stat_identity(before) != _stat_identity(named)
                or _identity_row(before) != expected["identity"]
                or _identity_row(os.fstat(parent["fd"]))
                != expected["parent_identity"]
                or first != b""
                or second != b""
            ):
                self._fail("holder completion reservation replay differs")
            reservation = {
                "relative": leaf,
                "parent": parent_key,
                "name": path.name,
                "fd": descriptor,
                "mode": HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE,
                "filled": False,
                "anchor": before,
                "size": 0,
                "sha256": _EMPTY_SHA256,
                "holder_job_id": holder,
            }
            self.reservations.append(reservation)
            descriptor = None
            self._call_barrier("after-holder-completion-capture", leaf)
            self._check_all()
            return reservation
        except OSError as error:
            if reservation is not None and reservation in self.reservations:
                self.reservations.remove(reservation)
                os.close(reservation["fd"])
            raise self.error_type(
                f"{self.label}: cannot capture holder completion reservation"
            ) from error
        except BaseException:
            if reservation is not None and reservation in self.reservations:
                self.reservations.remove(reservation)
                os.close(reservation["fd"])
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def fill_holder_completion_reservation(
        self, reservation: Mapping[str, Any], completion: Mapping[str, Any],
        *, topology: Sequence[Mapping[str, Any]],
        base_directory_authority: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            reservation not in self.reservations
            or reservation.get("holder_job_id") != self.owner_holder_job_id
        ):
            self._fail("holder completion reservation ownership differs")
        validated = validate_holder_directory_completion(
            completion,
            topology=topology,
            base_directory_authority=base_directory_authority,
        )
        if validated["holder_job_id"] != self.owner_holder_job_id:
            self._fail("holder completion object ownership differs")
        return self.fill_reserved(
            reservation,
            canonical_json_bytes(validated) + b"\n",
            final_mode=HOLDER_DIRECTORY_COMPLETION_SEALED_MODE,
        )

    def capture_filled_holder_completions(
        self, publication_receipt: Mapping[str, Any],
        *, topology: Sequence[Mapping[str, Any]],
        base_directory_authority: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Retain and validate the four fixed, filled completion inodes."""
        if self.owner_holder_job_id is not None or not self.materialized_topology:
            self._fail("filled completion capture requires read-only materialization")
        if (
            not isinstance(publication_receipt, Mapping)
            or publication_receipt.get("schema_version") != PUBLICATION_SCHEMA
            or publication_receipt.get("evaluation_root") != str(self.path)
            or publication_receipt.get("directory_authority_materialized")
            is not True
        ):
            self._fail("filled completion publication receipt differs")
        _verify_digest(
            publication_receipt,
            field="publication_digest",
            label="publication receipt",
        )
        reservations = validate_holder_completion_reservations(
            publication_receipt.get("holder_completion_reservations"),
            evaluation_root=self.path,
            materialized_required=True,
        )
        result: dict[str, dict[str, Any]] = {}
        immutable_fields = ("device", "inode", "uid", "gid", "rdev")
        for reservation in reservations:
            holder = reservation["holder_job_id"]
            raw, binding = self.read_bytes(
                reservation["relative_path"],
                expected_mode=HOLDER_DIRECTORY_COMPLETION_SEALED_MODE,
            )
            if (
                binding["path"] != reservation["path"]
                or any(
                    binding[field] != reservation["identity"][field]
                    for field in immutable_fields
                )
                or not stat.S_ISREG(reservation["identity"]["mode"])
            ):
                self._fail(
                    f"filled completion initial inode differs: {holder}"
                )
            try:
                decoded = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise self.error_type(
                    f"{self.label}: holder completion JSON differs"
                ) from error
            if (
                not isinstance(decoded, Mapping)
                or canonical_json_bytes(decoded) + b"\n" != raw
            ):
                self._fail("holder completion serialization differs")
            completion = validate_holder_directory_completion(
                decoded,
                topology=topology,
                base_directory_authority=base_directory_authority,
            )
            if completion["holder_job_id"] != holder:
                self._fail("holder completion reservation ownership differs")
            result[holder] = {
                "completion": completion,
                "file": binding,
            }
        if set(result) != {holder["job_id"] for holder in HOLDER_ROWS}:
            self._fail("filled holder completion closure differs")
        self._check_all()
        return result

    def write_bytes(
        self, relative: str | Path, payload: bytes, *, mode: int = 0o400,
    ) -> dict[str, Any]:
        reservation = self.reserve_file(relative, mode=mode)
        return self.fill_reserved(reservation, payload)

    def write_json(
        self, relative: str | Path, value: Mapping[str, Any],
        *, mode: int = 0o400,
    ) -> dict[str, Any]:
        return self.write_bytes(
            relative, canonical_json_bytes(value) + b"\n", mode=mode
        )

    def read_bytes(
        self, relative: str | Path, *, expected_sha256: str | None = None,
        expected_mode: int = 0o400,
    ) -> tuple[bytes, dict[str, Any]]:
        leaf = self._relative(relative)
        path = Path(leaf)
        parent_key = "." if path.parent == Path(".") else path.parent.as_posix()
        if parent_key not in self.states:
            self._fail(f"file parent is not retained: {leaf}")
        if expected_sha256 is not None and (
            type(expected_sha256) is not str
            or _SHA256.fullmatch(expected_sha256) is None
        ):
            self._fail(f"expected file SHA differs: {leaf}")
        if type(expected_mode) is not int or expected_mode < 0:
            self._fail(f"expected file mode differs: {leaf}")
        self._check_all()
        parent = self.states[parent_key]
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path.name, flags, dir_fd=parent["fd"])
        except OSError as error:
            raise self.error_type(
                f"{self.label}: cannot capture retained file: {leaf}"
            ) from error
        capture: dict[str, Any] | None = None
        try:
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first = _read_descriptor(descriptor)
            middle = os.fstat(descriptor)
            second = _read_descriptor(descriptor)
            self._call_barrier("after-file-read", leaf)
            after = os.fstat(descriptor)
            named = os.stat(
                path.name, dir_fd=parent["fd"], follow_symlinks=False
            )
            digest = hashlib.sha256(first).hexdigest()
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != expected_mode
                or _stat_identity(before) != _stat_identity(middle)
                or _stat_identity(before) != _stat_identity(after)
                or _stat_identity(before) != _stat_identity(named)
                or first != second
                or (
                    expected_sha256 is not None
                    and digest != expected_sha256
                )
            ):
                self._fail(f"same-FD file capture differs: {leaf}")
            identity = _identity_row(before)
            identity["mode"] = stat.S_IMODE(before.st_mode)
            binding = {
                "path": str(self.path / leaf), "sha256": digest,
                "size": len(first), **identity,
            }
            capture = {
                "relative": leaf, "parent": parent_key,
                "name": path.name, "fd": descriptor,
                "mode": expected_mode, "filled": True,
                "anchor": before, "size": len(first), "sha256": digest,
            }
            self.captures.append(capture)
            self._check_all()
            descriptor = -1
            return first, binding
        except BaseException:
            if capture is not None and capture in self.captures:
                self.captures.remove(capture)
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def seal(self, *, topology: Sequence[Mapping[str, Any]] | None = None) -> None:
        self._call_barrier("before-seal", ".")
        self._check_all()
        if topology is not None:
            expected = {
                item["relative_path"]: set(item["expected_entries"])
                for item in topology
            }
            expected_modes = {
                item["relative_path"]: item["expected_mode"]
                for item in topology
            }
            if set(expected) != set(self.states) or any(
                expected[key] != self.states[key]["entries"] for key in expected
            ) or any(
                expected_modes[key] != self.states[key]["mode"]
                for key in expected_modes
            ):
                self._fail("final directory topology differs")
        for relative in sorted(
            self.states, key=lambda item: (item.count("/"), item), reverse=True
        ):
            os.fsync(self.states[relative]["fd"])
        os.fsync(self.parent_fd)
        self._check_all()

    def set_directory_mode(
        self, relative: str | Path, mode: int,
    ) -> None:
        key = self._relative(relative, allow_root=True)
        if key not in self.states or mode not in {0o555, 0o700}:
            self._fail("directory seal mode differs")
        self._require_mutable(key)
        self._check_all()
        state = self.states[key]
        os.fchmod(state["fd"], mode)
        os.fsync(state["fd"])
        state["mode"] = mode
        state["anchor"] = self._observe_state(key)
        self._check_all()

    def authority_identities(
        self, *, topology: Sequence[Mapping[str, Any]],
    ) -> dict[str, tuple[os.stat_result, os.stat_result]]:
        self.seal(topology=topology)
        result: dict[str, tuple[os.stat_result, os.stat_result]] = {}
        for item in topology:
            relative = item["relative_path"]
            state = self.states[relative]
            identity = os.fstat(state["fd"])
            parent = (
                os.fstat(self.parent_fd)
                if relative == "."
                else os.fstat(self.states[state["parent"]]["fd"])
            )
            result[relative] = (identity, parent)
        return result

    def holder_directory_completion(
        self, *, topology: Sequence[Mapping[str, Any]],
        base_directory_authority: Mapping[str, Any],
        holder_summary_digest: str,
    ) -> dict[str, Any]:
        if self.owner_holder_job_id is None or not self.materialized_topology:
            self._fail("holder directory completion requires materialized owner")
        base = validate_directory_authority(
            base_directory_authority,
            topology=topology,
            materialized_required=True,
        )
        if base["evaluation_root"] != str(self.path):
            self._fail("holder directory completion root differs")
        if (
            type(holder_summary_digest) is not str
            or _SHA256.fullmatch(holder_summary_digest) is None
        ):
            self._fail("holder summary digest differs")
        self._check_all()
        holder = self.owner_holder_job_id
        mutable_rows = _holder_mutable_topology_rows(topology, holder)
        rows = []
        for item in mutable_rows:
            relative = item["relative_path"]
            state = self.states.get(relative)
            if state is None or state["foreign_mutable"]:
                self._fail(
                    f"holder mutable directory is not retained: {relative}"
                )
            identity = self._observe_state(relative)
            if _stat_identity(identity) != _stat_identity(state["anchor"]):
                self._fail(f"holder mutable directory drifted: {relative}")
            parent_identity = os.fstat(self.states[state["parent"]]["fd"])
            rows.append(
                {
                    "relative_path": relative,
                    "path": str(self.path / relative),
                    "owner_holder_job_id": holder,
                    "expected_mode": state["mode"],
                    "expected_entries": sorted(state["entries"]),
                    "identity": _identity_row(identity),
                    "parent_identity": _identity_row(parent_identity),
                }
            )
        value: dict[str, Any] = {
            "schema_version": HOLDER_DIRECTORY_COMPLETION_SCHEMA,
            "evaluation_root": str(self.path),
            "base_authority_digest": base["authority_digest"],
            "base_topology_digest": base["topology_digest"],
            "holder_job_id": holder,
            "holder_summary_digest": holder_summary_digest,
            "rows": rows,
            "row_count": len(rows),
        }
        value["completion_digest"] = object_sha256(value)
        return validate_holder_directory_completion(
            value,
            topology=topology,
            base_directory_authority=base,
        )

    def authority_row(self) -> dict[str, Any]:
        self._check_all()
        root = os.fstat(self.states["."]["fd"])
        parent = os.fstat(self.parent_fd)
        return {
            "schema_version": "bernini-retained-directory-authority-v1",
            "path": str(self.path), "identity": _identity_row(root),
            "parent_identity": _identity_row(parent),
            "entries": sorted(self.states["."]["entries"]),
            "retained_parent_fd": True, "retained_root_fd": True,
        }

    def directory_fd(self, relative: str | Path) -> int:
        key = self._relative(relative, allow_root=True)
        if key not in self.states:
            self._fail(f"retained directory is absent: {key}")
        self._check_all()
        return int(self.states[key]["fd"])

    def refresh_owned_directory(
        self, relative: str | Path, *, expected_entries: set[str],
    ) -> None:
        key = self._relative(relative, allow_root=True)
        if key not in self.states:
            self._fail(f"retained directory is absent: {key}")
        self._require_mutable(key)
        if any(
            type(item) is not str or item in ("", ".", "..")
            or os.path.sep in item
            for item in expected_entries
        ):
            self._fail("owned directory entry closure differs")
        self.states[key]["entries"] = set(expected_entries)
        self._refresh(key)

    def close(self) -> None:
        if self.closed:
            return
        for reservation in self.reservations:
            os.close(reservation["fd"])
        for capture in self.captures:
            os.close(capture["fd"])
        for relative in sorted(
            self.states, key=lambda item: (item.count("/"), item), reverse=True
        ):
            os.close(self.states[relative]["fd"])
        os.close(self.parent_fd)
        if self.parent_parent_fd is not None:
            os.close(self.parent_parent_fd)
        self.closed = True


def publish_bundle_authorized(
    bundle: Mapping[str, Any], *, publication_barrier: Any = None,
    retained_parent_fd: int | None = None,
    retained_parent_parent_fd: int | None = None,
    expected_parent_immutable_identity: Mapping[str, int] | None = None,
    expected_parent_parent_immutable_identity: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    input_spec = validate_input_spec(bundle["input_spec"])
    manifest = validate_manifest(bundle["manifest"], input_spec=input_spec)
    topology = build_directory_topology(manifest, input_spec=input_spec)
    root = Path(manifest["evaluation_root"])
    authority = RetainedPublicationRoot.create(
        root, label="evaluation publication root",
        error_type=DecodedEvaluationPlanError, barrier=publication_barrier,
        retained_parent_fd=retained_parent_fd,
        retained_parent_parent_fd=retained_parent_parent_fd,
        expected_parent_immutable_identity=expected_parent_immutable_identity,
        expected_parent_parent_immutable_identity=(
            expected_parent_parent_immutable_identity
        ),
    )
    try:
        for row in topology:
            if row["relative_path"] != ".":
                authority.create_directory(row["relative_path"])
        authority.write_json(INPUT_FILENAME, input_spec)
        authority.write_json(MANIFEST_FILENAME, manifest)
        authority.write_json(
            REVIEW_CONTRACT_FILENAME,
            validate_review_packet_contract(bundle["review_contract"]),
        )
        for holder in HOLDER_ROWS:
            shard = validate_shard(
                bundle["shards"][holder["job_id"]],
                manifest=manifest, input_spec=input_spec,
            )
            authority.write_json(
                f"{SHARD_DIRECTORY}/{holder['job_id']}.json", shard
            )
        completion_reservation_handles = {
            holder["job_id"]: authority.reserve_file(
                holder_completion_reservation_relative(holder["job_id"]),
                mode=HOLDER_DIRECTORY_COMPLETION_INITIAL_MODE,
            )
            for holder in HOLDER_ROWS
        }
        authority_reservation = authority.reserve_file(
            DIRECTORY_AUTHORITY_FILENAME
        )
        receipt_reservation = authority.reserve_file(PUBLICATION_FILENAME)
        completion_reservations = authority.holder_completion_reservation_rows(
            completion_reservation_handles
        )
        identities = authority.authority_identities(topology=topology)
        directory_authority = build_directory_authority(
            evaluation_root=root, topology=topology, identities=identities,
        )
        directory_binding = authority.fill_reserved(
            authority_reservation,
            canonical_json_bytes(directory_authority) + b"\n",
        )
        receipt = build_publication_receipt(
            bundle,
            directory_authority=directory_authority,
            holder_completion_reservations=completion_reservations,
        )
        if (
            directory_binding["sha256"]
            != receipt["directory_authority_file_sha256"]
        ):
            raise DecodedEvaluationPlanError(
                "published directory authority hash differs"
            )
        receipt_binding = authority.fill_reserved(
            receipt_reservation, canonical_json_bytes(receipt) + b"\n"
        )
        authority.seal(topology=topology)
        return {
            "output": str(root / PUBLICATION_FILENAME),
            "publication_receipt": receipt,
            "publication_receipt_file": receipt_binding,
            "directory_authority": directory_authority,
            "directory_authority_file": directory_binding,
            "holder_completion_reservations": completion_reservations,
            "root_authority": authority.authority_row(),
        }
    finally:
        authority.close()


def publish_bundle(
    bundle: Mapping[str, Any], *, publication_barrier: Any = None,
) -> Path:
    result = publish_bundle_authorized(
        bundle, publication_barrier=publication_barrier
    )
    return Path(result["output"])


def _load(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DecodedEvaluationPlanError(f"cannot load {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise DecodedEvaluationPlanError(f"{label} root is not an object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-spec", required=True)
    args = parser.parse_args(argv)
    bundle = build_bundle(_load(args.input_spec, label="evaluation input spec"))
    output = publish_bundle(bundle)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
