#!/usr/bin/env python3
"""Create-only controller for the action-preservation v2 canary loop.

This module is deliberately a control-plane component.  It can publish local
plans and receipts, validate already-produced training/decode evidence, and
emit one fail-closed next action.  It never starts a process, submits a job, or
updates model parameters.

The scientific transition is conjunctive:

* one fresh 20-update stage with checkpoints at relative steps 0/5/10/20;
* one decoded full-video candidate bound to the terminal checkpoint;
* one calibrated, independently recomputed ``action_preservation_gate_v1``
  decision with every axis passing; and
* one blinded full-video review by at least two reviewers with every axis
  passing.

Loss diagnostics are retained for engineering diagnosis only.  They are never
an input to the transition and cannot compensate for identity, background,
camera, quality, or action/order failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

import action_preservation_gate_v1 as gate


STAGE_PLAN_SCHEMA = "bernini-action-preservation-loop-stage-plan-v1"
STAGE_RECEIPT_SCHEMA = "bernini-action-preservation-loop-stage-receipt-v1"
TRANSITION_SCHEMA = "bernini-action-preservation-loop-transition-v1"

OBJECTIVE_FAMILY = "preservation_v2"
STAGE_UPDATES = 20
CHECKPOINT_STEPS = (0, 5, 10, 20)

STOP = "STOP"
WAIT_FOR_MACHINE_EVIDENCE = "WAIT_FOR_MACHINE_EVIDENCE"
WAIT_FOR_BLIND_REVIEW = "WAIT_FOR_BLIND_REVIEW"
ELIGIBLE_FOR_NEXT_20 = "ELIGIBLE_FOR_NEXT_20"
NEXT_ACTIONS = (
    STOP, WAIT_FOR_MACHINE_EVIDENCE, WAIT_FOR_BLIND_REVIEW,
    ELIGIBLE_FOR_NEXT_20,
)

PLAN_FILENAME = "stage_plan.json"
STAGE_RECEIPT_FILENAME = "stage_receipt.json"
WAIT_TRANSITION_FILENAME = "transition_wait_for_blind_review.json"
MACHINE_WAIT_TRANSITION_FILENAME = "transition_wait_for_machine_evidence.json"
TRANSITION_FILENAME = "transition_final.json"
STAGE_ROOT_AUTHORITY_FILENAME = "stage_root_authority.json"
STAGE_ROOT_AUTHORITY_SCHEMA = (
    "bernini-action-preservation-loop-stage-root-authority-v1"
)
PUBLICATION_RESULT_SCHEMA = (
    "bernini-action-preservation-loop-publication-result-v1"
)
WORK_ROOT_BINDING_ENV = "APV2_EVAL_WORK_ROOT_AUTHORITY"
TASK_FD_BINDING_ENV = "APV2_EVAL_INHERITED_AUTHORITY_FDS"
LOOP_RUNTIME_TARGET = "action_preservation_loop_controller_v1.py"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

PROVENANCE_FIELDS = frozenset(
    {
        "base_checkpoint_sha256",
        "teacher_cache_sha256",
        "training_manifest_sha256",
        "release_manifest_sha256",
        "trainer_source_sha256",
        "objective_source_sha256",
        "gate_source_sha256",
        "measurement_source_sha256",
        "calibration_digest",
    }
)


class ActionPreservationLoopError(RuntimeError):
    """A loop plan, receipt, or create-only publication is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return gate.canonical_json_bytes(value)
    except gate.ActionPreservationGateError as error:
        raise ActionPreservationLoopError(str(error)) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ActionPreservationLoopError(f"{label} field closure differs")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ActionPreservationLoopError(f"{label} is not a lowercase SHA-256")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ActionPreservationLoopError(f"{label} is invalid")
    return value


def _absolute_nonroot_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ActionPreservationLoopError(f"{label} is not a path string")
    path = Path(value)
    if not path.is_absolute() or value == os.path.sep:
        raise ActionPreservationLoopError(f"{label} must be absolute and non-root")
    if os.path.normpath(value) != value:
        raise ActionPreservationLoopError(f"{label} must be lexically normalized")
    return value


def _verify_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    digest = _sha(value.get(field), label=f"{label} digest")
    payload = dict(value)
    payload.pop(field)
    if object_sha256(payload) != digest:
        raise ActionPreservationLoopError(f"{label} digest differs")
    return digest


def _validate_provenance(value: Any) -> dict[str, str]:
    row = dict(_closed(value, PROVENANCE_FIELDS, label="input provenance"))
    for key in sorted(PROVENANCE_FIELDS):
        row[key] = _sha(row[key], label=key)
    return row


def _evaluation_contract() -> dict[str, Any]:
    return {
        "decoded_full_video_required": True,
        "measurement_schema": gate.MEASUREMENT_SCHEMA,
        "decision_schema": gate.DECISION_SCHEMA,
        "calibration_required": True,
        "separate_axes": list(gate.AXES),
        "weighted_compensation_forbidden": True,
        "blind_full_video_human_review_required": True,
        "minimum_reviewer_count": 2,
        "loss_may_authorize_transition": False,
    }


def _execution_policy() -> dict[str, Any]:
    return {
        "fresh_create_only_root_required": True,
        "previous_roots_may_be_reused": False,
        "controller_performs_remote_launch": False,
        "remote_launch_authorized": False,
        "automatic_model_update": False,
    }


def build_stage_plan(
    *,
    stage_id: str,
    stage_index: int,
    stage_root: str | Path,
    input_provenance: Mapping[str, Any],
    prior_stage_roots: Sequence[str | Path] = (),
    parent_stage_id: str | None = None,
    parent_plan_digest: str | None = None,
    parent_transition_digest: str | None = None,
) -> dict[str, Any]:
    """Build and validate one signed, launch-free 20-update stage plan."""

    root = str(stage_root)
    plan = {
        "schema_version": STAGE_PLAN_SCHEMA,
        "stage_id": stage_id,
        "stage_index": stage_index,
        "stage_root": root,
        "objective_family": OBJECTIVE_FAMILY,
        "training_contract": {
            "relative_update_count": STAGE_UPDATES,
            "relative_checkpoint_steps": list(CHECKPOINT_STEPS),
            "checkpoint_files_create_only": True,
            "terminal_checkpoint_required_for_decode": True,
        },
        "input_provenance": dict(input_provenance),
        "lineage": {
            "prior_stage_roots": [str(path) for path in prior_stage_roots],
            "parent_stage_id": parent_stage_id,
            "parent_plan_digest": parent_plan_digest,
            "parent_transition_digest": parent_transition_digest,
        },
        "evaluation_contract": _evaluation_contract(),
        "execution_policy": _execution_policy(),
    }
    plan["plan_digest"] = object_sha256(plan)
    return validate_stage_plan(plan)


def validate_stage_plan(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "stage_id",
        "stage_index",
        "stage_root",
        "objective_family",
        "training_contract",
        "input_provenance",
        "lineage",
        "evaluation_contract",
        "execution_policy",
        "plan_digest",
    }
    row = dict(_closed(value, fields, label="stage plan"))
    if row["schema_version"] != STAGE_PLAN_SCHEMA:
        raise ActionPreservationLoopError("stage plan schema differs")
    _identifier(row["stage_id"], label="stage id")
    if type(row["stage_index"]) is not int or row["stage_index"] < 0:
        raise ActionPreservationLoopError("stage index must be a non-negative integer")
    row["stage_root"] = _absolute_nonroot_path(row["stage_root"], label="stage root")
    if row["objective_family"] != OBJECTIVE_FAMILY:
        raise ActionPreservationLoopError("objective family differs")

    training = dict(
        _closed(
            row["training_contract"],
            {
                "relative_update_count",
                "relative_checkpoint_steps",
                "checkpoint_files_create_only",
                "terminal_checkpoint_required_for_decode",
            },
            label="training contract",
        )
    )
    if training["relative_update_count"] != STAGE_UPDATES:
        raise ActionPreservationLoopError("stage must contain exactly 20 updates")
    if training["relative_checkpoint_steps"] != list(CHECKPOINT_STEPS):
        raise ActionPreservationLoopError("checkpoint schedule must be exactly 0/5/10/20")
    if training["checkpoint_files_create_only"] is not True:
        raise ActionPreservationLoopError("checkpoint publication is not create-only")
    if training["terminal_checkpoint_required_for_decode"] is not True:
        raise ActionPreservationLoopError("terminal checkpoint is not decode-bound")

    provenance = _validate_provenance(row["input_provenance"])
    lineage = dict(
        _closed(
            row["lineage"],
            {
                "prior_stage_roots",
                "parent_stage_id",
                "parent_plan_digest",
                "parent_transition_digest",
            },
            label="lineage",
        )
    )
    roots = lineage["prior_stage_roots"]
    if not isinstance(roots, list) or len(roots) != row["stage_index"]:
        raise ActionPreservationLoopError("prior stage root count differs from stage index")
    normalized_roots = [
        _absolute_nonroot_path(item, label="prior stage root") for item in roots
    ]
    if len(set(normalized_roots)) != len(normalized_roots):
        raise ActionPreservationLoopError("prior stage roots are not unique")
    if row["stage_root"] in normalized_roots:
        raise ActionPreservationLoopError("stage root reuses a previous root")
    if row["stage_index"] == 0:
        if any(
            lineage[key] is not None
            for key in ("parent_stage_id", "parent_plan_digest", "parent_transition_digest")
        ):
            raise ActionPreservationLoopError("initial stage unexpectedly has a parent")
    else:
        _identifier(lineage["parent_stage_id"], label="parent stage id")
        _sha(lineage["parent_plan_digest"], label="parent plan digest")
        _sha(lineage["parent_transition_digest"], label="parent transition digest")

    evaluation = dict(
        _closed(
            row["evaluation_contract"],
            set(_evaluation_contract()),
            label="evaluation contract",
        )
    )
    if evaluation != _evaluation_contract():
        raise ActionPreservationLoopError("evaluation contract differs")
    policy = dict(
        _closed(
            row["execution_policy"], set(_execution_policy()), label="execution policy"
        )
    )
    if policy != _execution_policy():
        raise ActionPreservationLoopError("execution policy differs")
    _verify_digest(row, field="plan_digest", label="stage plan")
    row.update(
        training_contract=training,
        input_provenance=provenance,
        lineage={**lineage, "prior_stage_roots": normalized_roots},
        evaluation_contract=evaluation,
        execution_policy=policy,
    )
    return row


def build_stage_receipt(
    plan: Mapping[str, Any],
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    training_completion_receipt_sha256: str,
    candidate_id: str,
    candidate_video_sha256: str,
    source_video_sha256: str,
    decode_receipt_sha256: str,
    loss_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a terminal training/decode receipt bound to one stage plan."""

    authority = validate_stage_plan(plan)
    checkpoint_rows = [dict(item) for item in checkpoints]
    terminal_checkpoint_sha256 = (
        checkpoint_rows[-1].get("checkpoint_sha256") if checkpoint_rows else None
    )
    value = {
        "schema_version": STAGE_RECEIPT_SCHEMA,
        "stage_id": authority["stage_id"],
        "stage_index": authority["stage_index"],
        "stage_root": authority["stage_root"],
        "stage_plan_digest": authority["plan_digest"],
        "training": {
            "completed_relative_updates": STAGE_UPDATES,
            "training_completion_receipt_sha256": training_completion_receipt_sha256,
            "checkpoints": checkpoint_rows,
        },
        "decoded_candidate": {
            "candidate_id": candidate_id,
            "candidate_video_sha256": candidate_video_sha256,
            "source_video_sha256": source_video_sha256,
            "decoded_from_relative_step": CHECKPOINT_STEPS[-1],
            "decoded_from_checkpoint_sha256": terminal_checkpoint_sha256,
            "decode_receipt_sha256": decode_receipt_sha256,
            "full_video_emitted": True,
        },
        "loss_diagnostics": {
            "available": loss_receipt_sha256 is not None,
            "receipt_sha256": loss_receipt_sha256,
            "used_for_transition": False,
        },
        "controller_performed_remote_launch": False,
    }
    value["stage_receipt_digest"] = object_sha256(value)
    return validate_stage_receipt(value, plan=authority)


def validate_stage_receipt(value: Any, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    authority = validate_stage_plan(plan)
    fields = {
        "schema_version",
        "stage_id",
        "stage_index",
        "stage_root",
        "stage_plan_digest",
        "training",
        "decoded_candidate",
        "loss_diagnostics",
        "controller_performed_remote_launch",
        "stage_receipt_digest",
    }
    row = dict(_closed(value, fields, label="stage receipt"))
    if row["schema_version"] != STAGE_RECEIPT_SCHEMA:
        raise ActionPreservationLoopError("stage receipt schema differs")
    for key, plan_key in (
        ("stage_id", "stage_id"),
        ("stage_index", "stage_index"),
        ("stage_root", "stage_root"),
        ("stage_plan_digest", "plan_digest"),
    ):
        if row[key] != authority[plan_key]:
            raise ActionPreservationLoopError(f"stage receipt {key} binding differs")
    training = dict(
        _closed(
            row["training"],
            {
                "completed_relative_updates",
                "training_completion_receipt_sha256",
                "checkpoints",
            },
            label="training receipt",
        )
    )
    if training["completed_relative_updates"] != STAGE_UPDATES:
        raise ActionPreservationLoopError("terminal training receipt is not exactly 20 updates")
    _sha(
        training["training_completion_receipt_sha256"],
        label="training completion receipt",
    )
    checkpoints = training["checkpoints"]
    if not isinstance(checkpoints, list) or len(checkpoints) != len(CHECKPOINT_STEPS):
        raise ActionPreservationLoopError("checkpoint receipt count differs")
    normalized_checkpoints: list[dict[str, Any]] = []
    for expected_step, value_at_step in zip(CHECKPOINT_STEPS, checkpoints):
        item = dict(
            _closed(
                value_at_step,
                {"relative_step", "checkpoint_sha256", "checkpoint_receipt_sha256"},
                label="checkpoint receipt",
            )
        )
        if item["relative_step"] != expected_step:
            raise ActionPreservationLoopError("checkpoint schedule receipt differs")
        _sha(item["checkpoint_sha256"], label="checkpoint")
        _sha(item["checkpoint_receipt_sha256"], label="checkpoint receipt")
        normalized_checkpoints.append(item)

    candidate = dict(
        _closed(
            row["decoded_candidate"],
            {
                "candidate_id",
                "candidate_video_sha256",
                "source_video_sha256",
                "decoded_from_relative_step",
                "decoded_from_checkpoint_sha256",
                "decode_receipt_sha256",
                "full_video_emitted",
            },
            label="decoded candidate",
        )
    )
    _identifier(candidate["candidate_id"], label="candidate id")
    _sha(candidate["candidate_video_sha256"], label="candidate video")
    _sha(candidate["source_video_sha256"], label="source video")
    _sha(candidate["decoded_from_checkpoint_sha256"], label="decoded checkpoint")
    _sha(candidate["decode_receipt_sha256"], label="decode receipt")
    if candidate["decoded_from_relative_step"] != CHECKPOINT_STEPS[-1]:
        raise ActionPreservationLoopError("candidate was not decoded from terminal checkpoint")
    if (
        candidate["decoded_from_checkpoint_sha256"]
        != normalized_checkpoints[-1]["checkpoint_sha256"]
    ):
        raise ActionPreservationLoopError("decoded checkpoint hash differs from terminal checkpoint")
    if candidate["full_video_emitted"] is not True:
        raise ActionPreservationLoopError("decoded candidate is not a full video")

    loss = dict(
        _closed(
            row["loss_diagnostics"],
            {"available", "receipt_sha256", "used_for_transition"},
            label="loss diagnostics",
        )
    )
    if type(loss["available"]) is not bool:
        raise ActionPreservationLoopError("loss availability is not boolean")
    if loss["available"]:
        _sha(loss["receipt_sha256"], label="loss diagnostics receipt")
    elif loss["receipt_sha256"] is not None:
        raise ActionPreservationLoopError("unavailable loss diagnostics has a receipt")
    if loss["used_for_transition"] is not False:
        raise ActionPreservationLoopError("training loss may not authorize a transition")
    if row["controller_performed_remote_launch"] is not False:
        raise ActionPreservationLoopError("controller must not perform a remote launch")
    _verify_digest(row, field="stage_receipt_digest", label="stage receipt")
    row.update(
        training={**training, "checkpoints": normalized_checkpoints},
        decoded_candidate=candidate,
        loss_diagnostics=loss,
    )
    return row


def _transition_result(
    plan: Mapping[str, Any],
    *,
    receipt_digest: str | None,
    terminal_checkpoint_sha256: str | None,
    measurement_digest: str | None,
    calibration_digest: str | None,
    decision_digest: str | None,
    review_digest: str | None,
    promotion_digest: str | None,
    machine_status: str,
    human_status: str,
    next_action: str,
    reasons: Sequence[str],
) -> dict[str, Any]:
    if next_action not in NEXT_ACTIONS:
        raise ActionPreservationLoopError("next action differs")
    value = {
        "schema_version": TRANSITION_SCHEMA,
        "stage_id": plan["stage_id"],
        "stage_index": plan["stage_index"],
        "stage_root": plan["stage_root"],
        "stage_plan_digest": plan["plan_digest"],
        "stage_receipt_digest": receipt_digest,
        "terminal_checkpoint_sha256": terminal_checkpoint_sha256,
        "evidence": {
            "measurement_digest": measurement_digest,
            "calibration_digest": calibration_digest,
            "decision_digest": decision_digest,
            "review_digest": review_digest,
            "promotion_digest": promotion_digest,
        },
        "machine_status": machine_status,
        "human_status": human_status,
        "next_action": next_action,
        "reasons": list(reasons),
        "weighted_score": None,
        "loss_used_for_transition": False,
        "automatic_model_update": False,
        "controller_performed_remote_launch": False,
        "next_stage_requires_fresh_create_only_root": next_action
        == ELIGIBLE_FOR_NEXT_20,
    }
    value["transition_digest"] = object_sha256(value)
    return validate_transition_receipt(value, plan=plan)


def decide_next_action(
    plan: Mapping[str, Any],
    *,
    stage_receipt: Mapping[str, Any] | None,
    measurement: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
    decision: Mapping[str, Any] | None,
    blind_review: Mapping[str, Any] | None = None,
    evaluation_aggregate: Mapping[str, Any] | None = None,
    public_packet: Mapping[str, Any] | None = None,
    private_mapping: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a fail-closed transition; no training loss is inspected or scored."""

    authority = validate_stage_plan(plan)
    if stage_receipt is None:
        return _transition_result(
            authority,
            receipt_digest=None,
            terminal_checkpoint_sha256=None,
            measurement_digest=None,
            calibration_digest=None,
            decision_digest=None,
            review_digest=None,
            promotion_digest=None,
            machine_status="not_evaluated",
            human_status="not_evaluated",
            next_action=STOP,
            reasons=["stage_receipt_missing"],
        )
    try:
        receipt = validate_stage_receipt(stage_receipt, plan=authority)
    except ActionPreservationLoopError:
        return _transition_result(
            authority,
            receipt_digest=None,
            terminal_checkpoint_sha256=None,
            measurement_digest=None,
            calibration_digest=None,
            decision_digest=None,
            review_digest=None,
            promotion_digest=None,
            machine_status="invalid_evidence",
            human_status="not_evaluated",
            next_action=STOP,
            reasons=["stage_receipt_invalid"],
        )
    terminal_checkpoint = receipt["training"]["checkpoints"][-1]["checkpoint_sha256"]
    base_args = {
        "receipt_digest": receipt["stage_receipt_digest"],
        "terminal_checkpoint_sha256": terminal_checkpoint,
    }
    missing = [
        label
        for label, value in (
            ("decoded_measurement_missing", measurement),
            ("calibration_missing", calibration),
            ("strict_gate_decision_missing", decision),
        )
        if value is None
    ]
    if missing:
        return _transition_result(
            authority,
            **base_args,
            measurement_digest=None,
            calibration_digest=None,
            decision_digest=None,
            review_digest=None,
            promotion_digest=None,
            machine_status="abstain",
            human_status="review_may_proceed_in_parallel",
            next_action=WAIT_FOR_MACHINE_EVIDENCE,
            reasons=missing,
        )
    try:
        measured = gate.validate_measurement(measurement)
        calibrated = gate.validate_calibration(calibration)
    except gate.ActionPreservationGateError:
        return _transition_result(
            authority,
            **base_args,
            measurement_digest=None,
            calibration_digest=None,
            decision_digest=None,
            review_digest=None,
            promotion_digest=None,
            machine_status="invalid_evidence",
            human_status="not_evaluated",
            next_action=STOP,
            reasons=["measurement_or_calibration_invalid"],
        )
    candidate = receipt["decoded_candidate"]
    if (
        calibrated["calibration_digest"]
        != authority["input_provenance"]["calibration_digest"]
    ):
        return _transition_result(
            authority,
            **base_args,
            measurement_digest=measured["measurement_digest"],
            calibration_digest=calibrated["calibration_digest"],
            decision_digest=None,
            review_digest=None,
            promotion_digest=None,
            machine_status="invalid_evidence",
            human_status="not_evaluated",
            next_action=STOP,
            reasons=["calibration_plan_binding_differs"],
        )
    if any(
        measured[key] != candidate[key]
        for key in ("candidate_id", "candidate_video_sha256", "source_video_sha256")
    ):
        return _transition_result(
            authority,
            **base_args,
            measurement_digest=measured["measurement_digest"],
            calibration_digest=calibrated["calibration_digest"],
            decision_digest=None,
            review_digest=None,
            promotion_digest=None,
            machine_status="invalid_evidence",
            human_status="not_evaluated",
            next_action=STOP,
            reasons=["decoded_candidate_measurement_binding_differs"],
        )
    # Recompute from the original signed objects.  The gate validators return
    # normalized numeric values, which are useful below but must not replace
    # the canonical preimage whose digest was supplied by the evaluator.
    recomputed = gate.decide(measurement, calibration)
    try:
        decision_matches = canonical_json_bytes(decision) == canonical_json_bytes(recomputed)
    except ActionPreservationLoopError:
        decision_matches = False
    if not decision_matches:
        return _transition_result(
            authority,
            **base_args,
            measurement_digest=measured["measurement_digest"],
            calibration_digest=calibrated["calibration_digest"],
            decision_digest=None,
            review_digest=None,
            promotion_digest=None,
            machine_status="invalid_evidence",
            human_status="not_evaluated",
            next_action=STOP,
            reasons=["strict_gate_decision_differs_from_recomputation"],
        )
    decision_digest = recomputed["decision_digest"]
    evidence_args = {
        **base_args,
        "measurement_digest": measured["measurement_digest"],
        "calibration_digest": calibrated["calibration_digest"],
        "decision_digest": decision_digest,
    }
    if recomputed["status"] != "eligible_for_motion_ranking":
        return _transition_result(
            authority,
            **evidence_args,
            review_digest=None,
            promotion_digest=None,
            machine_status=recomputed["status"],
            human_status="not_evaluated",
            next_action=STOP,
            reasons=[f"machine_gate_{recomputed['status']}"],
        )
    if blind_review is None:
        return _transition_result(
            authority,
            **evidence_args,
            review_digest=None,
            promotion_digest=None,
            machine_status=recomputed["status"],
            human_status="pending",
            next_action=WAIT_FOR_BLIND_REVIEW,
            reasons=["blinded_full_video_human_review_required"],
        )
    if any(
        item is None
        for item in (evaluation_aggregate, public_packet, private_mapping)
    ):
        return _transition_result(
            authority,
            **evidence_args,
            review_digest=None,
            promotion_digest=None,
            machine_status=recomputed["status"],
            human_status="pending",
            next_action=WAIT_FOR_BLIND_REVIEW,
            reasons=["blind_review_packet_authority_required"],
        )
    try:
        promotion = gate.promotion_decision(
            recomputed,
            blind_review,
            evaluation_aggregate=evaluation_aggregate,
            public_packet=public_packet,
            private_mapping=private_mapping,
        )
    except gate.ActionPreservationGateError:
        return _transition_result(
            authority,
            **evidence_args,
            review_digest=None,
            promotion_digest=None,
            machine_status=recomputed["status"],
            human_status="invalid_evidence",
            next_action=STOP,
            reasons=["blind_review_invalid"],
        )
    if promotion["status"] == "eligible_for_next_20_update_stage":
        next_action = ELIGIBLE_FOR_NEXT_20
        human_status = "pass"
        reasons: list[str] = []
    else:
        next_action = STOP
        human_status = "fail_or_undetermined"
        reasons = [f"human_gate_{promotion['status']}"]
    return _transition_result(
        authority,
        **evidence_args,
        review_digest=promotion["review_digest"],
        promotion_digest=promotion["promotion_digest"],
        machine_status=recomputed["status"],
        human_status=human_status,
        next_action=next_action,
        reasons=reasons,
    )


def validate_transition_receipt(value: Any, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    authority = validate_stage_plan(plan)
    fields = {
        "schema_version",
        "stage_id",
        "stage_index",
        "stage_root",
        "stage_plan_digest",
        "stage_receipt_digest",
        "terminal_checkpoint_sha256",
        "evidence",
        "machine_status",
        "human_status",
        "next_action",
        "reasons",
        "weighted_score",
        "loss_used_for_transition",
        "automatic_model_update",
        "controller_performed_remote_launch",
        "next_stage_requires_fresh_create_only_root",
        "transition_digest",
    }
    row = dict(_closed(value, fields, label="transition receipt"))
    if row["schema_version"] != TRANSITION_SCHEMA:
        raise ActionPreservationLoopError("transition schema differs")
    for key, plan_key in (
        ("stage_id", "stage_id"),
        ("stage_index", "stage_index"),
        ("stage_root", "stage_root"),
        ("stage_plan_digest", "plan_digest"),
    ):
        if row[key] != authority[plan_key]:
            raise ActionPreservationLoopError(f"transition {key} binding differs")
    for key in ("stage_receipt_digest", "terminal_checkpoint_sha256"):
        if row[key] is not None:
            _sha(row[key], label=key)
    evidence = dict(
        _closed(
            row["evidence"],
            {
                "measurement_digest",
                "calibration_digest",
                "decision_digest",
                "review_digest",
                "promotion_digest",
            },
            label="transition evidence",
        )
    )
    for key, item in evidence.items():
        if item is not None:
            _sha(item, label=key)
    if row["next_action"] not in NEXT_ACTIONS:
        raise ActionPreservationLoopError("next action differs")
    if not isinstance(row["reasons"], list) or any(
        not isinstance(item, str) or not item for item in row["reasons"]
    ):
        raise ActionPreservationLoopError("transition reasons differ")
    if row["weighted_score"] is not None:
        raise ActionPreservationLoopError("weighted compensation is forbidden")
    for key in (
        "loss_used_for_transition",
        "automatic_model_update",
        "controller_performed_remote_launch",
    ):
        if row[key] is not False:
            raise ActionPreservationLoopError(f"{key} must be false")
    if row["next_stage_requires_fresh_create_only_root"] is not (
        row["next_action"] == ELIGIBLE_FOR_NEXT_20
    ):
        raise ActionPreservationLoopError("next-stage freshness flag differs")
    if row["next_action"] in (
        WAIT_FOR_MACHINE_EVIDENCE, WAIT_FOR_BLIND_REVIEW, ELIGIBLE_FOR_NEXT_20
    ):
        if row["stage_receipt_digest"] is None or row["terminal_checkpoint_sha256"] is None:
            raise ActionPreservationLoopError("continuing transition lacks stage receipt binding")
    if row["next_action"] in (WAIT_FOR_BLIND_REVIEW, ELIGIBLE_FOR_NEXT_20):
        if not all(
            evidence[key]
            for key in ("measurement_digest", "calibration_digest", "decision_digest")
        ):
            raise ActionPreservationLoopError("continuing transition lacks machine evidence")
    if row["next_action"] == WAIT_FOR_MACHINE_EVIDENCE:
        if any(evidence.values()):
            raise ActionPreservationLoopError(
                "machine-wait transition unexpectedly claims completed evidence"
            )
        if row["machine_status"] != "abstain":
            raise ActionPreservationLoopError("machine-wait status differs")
        if row["human_status"] != "review_may_proceed_in_parallel":
            raise ActionPreservationLoopError("parallel review status differs")
    if row["next_action"] == WAIT_FOR_BLIND_REVIEW:
        if evidence["review_digest"] is not None or evidence["promotion_digest"] is not None:
            raise ActionPreservationLoopError("waiting transition unexpectedly has human evidence")
        if row["machine_status"] != "eligible_for_motion_ranking":
            raise ActionPreservationLoopError("waiting transition lacks machine pass")
        if row["human_status"] != "pending":
            raise ActionPreservationLoopError("waiting transition human status differs")
    if row["next_action"] == ELIGIBLE_FOR_NEXT_20:
        if not all(evidence.values()):
            raise ActionPreservationLoopError("eligible transition lacks complete evidence")
        if row["machine_status"] != "eligible_for_motion_ranking":
            raise ActionPreservationLoopError("eligible transition lacks machine pass")
        if row["human_status"] != "pass":
            raise ActionPreservationLoopError("eligible transition lacks human pass")
    _verify_digest(row, field="transition_digest", label="transition receipt")
    row["evidence"] = evidence
    return row


def build_next_stage_plan(
    previous_plan: Mapping[str, Any],
    previous_transition: Mapping[str, Any],
    *,
    stage_id: str,
    stage_root: str | Path,
    input_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a fresh next-stage plan only from an eligible transition."""

    parent = validate_stage_plan(previous_plan)
    transition = validate_transition_receipt(previous_transition, plan=parent)
    if transition["next_action"] != ELIGIBLE_FOR_NEXT_20:
        raise ActionPreservationLoopError("previous stage is not eligible for another 20 updates")
    provenance = _validate_provenance(input_provenance)
    if provenance["base_checkpoint_sha256"] != transition["terminal_checkpoint_sha256"]:
        raise ActionPreservationLoopError("next base checkpoint does not bind parent terminal checkpoint")
    return build_stage_plan(
        stage_id=stage_id,
        stage_index=parent["stage_index"] + 1,
        stage_root=stage_root,
        input_provenance=provenance,
        prior_stage_roots=[
            *parent["lineage"]["prior_stage_roots"],
            parent["stage_root"],
        ],
        parent_stage_id=parent["stage_id"],
        parent_plan_digest=parent["plan_digest"],
        parent_transition_digest=transition["transition_digest"],
    )


def _plain_directory(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ActionPreservationLoopError(f"{label} does not exist") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ActionPreservationLoopError(f"{label} is not a plain directory")


def _identity_row(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev), "inode": int(value.st_ino),
        "uid": int(value.st_uid), "gid": int(value.st_gid),
        "mode": int(value.st_mode), "nlink": int(value.st_nlink),
        "rdev": int(value.st_rdev), "size": int(value.st_size),
        "blocks": int(getattr(value, "st_blocks", 0)),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _immutable_directory_identity(value: os.stat_result) -> dict[str, int]:
    row = _identity_row(value)
    return {
        key: row[key]
        for key in ("device", "inode", "uid", "gid", "mode", "rdev")
    }


def _validate_immutable_directory_identity(
    value: Any, *, label: str,
) -> dict[str, int]:
    fields = {"device", "inode", "uid", "gid", "mode", "rdev"}
    if (
        type(value) is not dict
        or set(value) != fields
        or any(type(value[field]) is not int or value[field] < 0
               for field in fields)
        or not stat.S_ISDIR(value["mode"])
    ):
        raise ActionPreservationLoopError(f"{label} identity differs")
    return dict(value)


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ActionPreservationLoopError(
                    f"{label} contains a duplicate key"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ActionPreservationLoopError(
            f"cannot decode {label}"
        ) from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ActionPreservationLoopError(
            f"{label} is not canonical newline JSON"
        )
    return value


def _load_loop_work_root_authority() -> dict[str, Any]:
    if os.environ.get(WORK_ROOT_BINDING_ENV) is None:
        raise ActionPreservationLoopError(
            "inherited WORK_ROOT A authority is absent"
        )
    if os.environ.get(TASK_FD_BINDING_ENV) is not None:
        raise ActionPreservationLoopError(
            "mixed WORK_ROOT A and task-FD B authorities are forbidden"
        )
    import action_preservation_decoded_eval_verified_release_v1 as runtime

    try:
        row = runtime.load_inherited_work_root_environment(
            verify_open_fds=True,
            expected_inheritable=False,
            verify_entries=False,
            allow_root_metadata_change=True,
        )
    except runtime.DecodedEvalVerifiedReleaseError as error:
        raise ActionPreservationLoopError(str(error)) from error
    if row["target"] != LOOP_RUNTIME_TARGET:
        raise ActionPreservationLoopError(
            "inherited WORK_ROOT A target is not the loop controller"
        )
    return row


def _replay_loop_work_root_authority(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    import action_preservation_decoded_eval_verified_release_v1 as runtime

    try:
        row = runtime.validate_inherited_work_root_binding(
            value,
            verify_open_fds=True,
            expected_inheritable=False,
            verify_entries=False,
            allow_root_metadata_change=True,
        )
    except runtime.DecodedEvalVerifiedReleaseError as error:
        raise ActionPreservationLoopError(str(error)) from error
    if row["target"] != LOOP_RUNTIME_TARGET:
        raise ActionPreservationLoopError(
            "inherited WORK_ROOT A target is not the loop controller"
        )
    return row


def _load_expected_json(
    path: str, expected_sha256: str, *, label: str,
) -> dict[str, Any]:
    try:
        return gate._load_expected(path, expected_sha256, label=label)
    except gate.ActionPreservationGateError as error:
        raise ActionPreservationLoopError(str(error)) from error


def _validate_stage_root_authority(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "work_root_authority_digest", "stage_root",
        "stage_root_name", "stage_root_immutable_identity", "plan_path",
        "plan_sha256", "plan_digest",
        "created_relative_to_inherited_work_root_fd", "authority_digest",
    }
    row = dict(_closed(value, fields, label="stage root authority"))
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    root = Path(row["stage_root"])
    plan = Path(row["plan_path"])
    identity = _validate_immutable_directory_identity(
        row["stage_root_immutable_identity"], label="stage root"
    )
    if (
        row["schema_version"] != STAGE_ROOT_AUTHORITY_SCHEMA
        or not root.is_absolute()
        or os.path.normpath(str(root)) != str(root)
        or root.name != row["stage_root_name"]
        or root.name in ("", ".", "..")
        or plan != root / PLAN_FILENAME
        or stat.S_IMODE(identity["mode"]) != 0o700
        or row["created_relative_to_inherited_work_root_fd"] is not True
    ):
        raise ActionPreservationLoopError(
            "stage root authority binding differs"
        )
    _sha(row["work_root_authority_digest"], label="work root authority")
    _sha(row["plan_sha256"], label="stage plan file")
    _sha(row["plan_digest"], label="stage plan")
    if (
        not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or object_sha256(unsigned) != claimed
    ):
        raise ActionPreservationLoopError(
            "stage root authority digest differs"
        )
    row["stage_root_immutable_identity"] = identity
    return row


class _HeldStageRoot:
    def __init__(
        self, *, work_root: Mapping[str, Any], path: Path,
        descriptor: int, immutable_identity: Mapping[str, Any],
    ) -> None:
        self.work_root = dict(work_root)
        self.path = path
        self.descriptor = descriptor
        self.immutable_identity = dict(immutable_identity)

    @classmethod
    def create(
        cls, work_root: Mapping[str, Any], path: Path,
    ) -> "_HeldStageRoot":
        row = _replay_loop_work_root_authority(work_root)
        if (
            not path.is_absolute()
            or os.path.normpath(str(path)) != str(path)
            or path.parent != Path(row["path"])
            or path.name in ("", ".", "..")
        ):
            raise ActionPreservationLoopError(
                "stage root must be a canonical direct WORK_ROOT child"
            )
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise ActionPreservationLoopError(
                "safe stage root creation is unavailable"
            )
        descriptor: int | None = None
        try:
            os.mkdir(path.name, 0o700, dir_fd=row["root_fd"])
            os.fsync(row["root_fd"])
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=row["root_fd"],
            )
            os.set_inheritable(descriptor, False)
            observed = os.fstat(descriptor)
            named = os.stat(
                path.name, dir_fd=row["root_fd"], follow_symlinks=False
            )
        except FileExistsError as error:
            if descriptor is not None:
                os.close(descriptor)
            raise ActionPreservationLoopError(
                f"stage root is not fresh: {path}"
            ) from error
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            raise ActionPreservationLoopError(
                "cannot create held stage root"
            ) from error
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o700
            or _identity_row(observed) != _identity_row(named)
            or os.get_inheritable(descriptor)
        ):
            os.close(descriptor)
            raise ActionPreservationLoopError(
                "held stage root creation identity differs"
            )
        result = cls(
            work_root=row, path=path, descriptor=descriptor,
            immutable_identity=_immutable_directory_identity(observed),
        )
        try:
            result.replay()
            return result
        except BaseException:
            result.close()
            raise

    @classmethod
    def open(
        cls, work_root: Mapping[str, Any], *, path: Path,
        authority_path: Path, authority_sha256: str,
    ) -> tuple["_HeldStageRoot", dict[str, Any]]:
        row = _replay_loop_work_root_authority(work_root)
        if (
            not path.is_absolute()
            or path.parent != Path(row["path"])
            or authority_path != path / STAGE_ROOT_AUTHORITY_FILENAME
        ):
            raise ActionPreservationLoopError(
                "stage root authority path differs"
            )
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=row["root_fd"],
            )
            os.set_inheritable(descriptor, False)
        except OSError as error:
            raise ActionPreservationLoopError(
                "cannot open held stage root"
            ) from error
        try:
            provisional = cls(
                work_root=row, path=path, descriptor=descriptor,
                immutable_identity=_immutable_directory_identity(
                    os.fstat(descriptor)
                ),
            )
        except BaseException:
            os.close(descriptor)
            raise
        try:
            raw, _ = provisional.read(
                STAGE_ROOT_AUTHORITY_FILENAME,
                expected_sha256=authority_sha256,
                label="stage root authority",
            )
            authority = _validate_stage_root_authority(
                _strict_json_bytes(raw, label="stage root authority")
            )
            if (
                authority["stage_root"] != str(path)
                or authority["stage_root_name"] != path.name
                or authority["work_root_authority_digest"]
                != row["work_root_authority_digest"]
                or authority["stage_root_immutable_identity"]
                != provisional.immutable_identity
            ):
                raise ActionPreservationLoopError(
                    "stage root authority continuity differs"
                )
            provisional.replay()
            return provisional, authority
        except BaseException:
            provisional.close()
            raise

    def replay(self) -> None:
        row = _replay_loop_work_root_authority(self.work_root)
        try:
            observed = os.fstat(self.descriptor)
            named = os.stat(
                self.path.name,
                dir_fd=row["root_fd"],
                follow_symlinks=False,
            )
        except OSError as error:
            raise ActionPreservationLoopError(
                "cannot replay held stage root"
            ) from error
        if (
            _immutable_directory_identity(observed)
            != self.immutable_identity
            or _immutable_directory_identity(named)
            != self.immutable_identity
            or os.get_inheritable(self.descriptor)
        ):
            raise ActionPreservationLoopError(
                "held stage root immutable identity differs"
            )

    def read(
        self, name: str, *, expected_sha256: str, label: str,
    ) -> tuple[bytes, dict[str, Any]]:
        expected = _sha(expected_sha256, label=f"{label} SHA")
        if name in ("", ".", "..") or "/" in name or "\x00" in name:
            raise ActionPreservationLoopError(f"{label} basename differs")
        self.replay()
        descriptor: int | None = None
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self.descriptor,
            )
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            first = _read_fd(descriptor)
            middle = os.fstat(descriptor)
            second = _read_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.descriptor, follow_symlinks=False
            )
        except OSError as error:
            raise ActionPreservationLoopError(
                f"cannot read held {label}"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        identity = _identity_row(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or identity != _identity_row(middle)
            or identity != _identity_row(after)
            or identity != _identity_row(named)
            or first != second
            or len(first) != before.st_size
            or hashlib.sha256(first).hexdigest() != expected
        ):
            raise ActionPreservationLoopError(
                f"held {label} physical binding differs"
            )
        self.replay()
        return first, identity

    def write_json(
        self, name: str, value: Mapping[str, Any], *, mode: int = 0o444,
    ) -> dict[str, Any]:
        if name in ("", ".", "..") or "/" in name or "\x00" in name:
            raise ActionPreservationLoopError(
                "stage output basename differs"
            )
        self.replay()
        payload = canonical_json_bytes(value) + b"\n"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                mode,
                dir_fd=self.descriptor,
            )
            os.set_inheritable(descriptor, False)
            offset = 0
            while offset < len(payload):
                count = os.write(descriptor, payload[offset:])
                if count <= 0:
                    raise ActionPreservationLoopError(
                        "held stage output write made no progress"
                    )
                offset += count
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            os.fsync(self.descriptor)
            before = os.fstat(descriptor)
            first = _read_fd(descriptor)
            middle = os.fstat(descriptor)
            second = _read_fd(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.descriptor, follow_symlinks=False
            )
        except FileExistsError as error:
            raise ActionPreservationLoopError(
                f"refusing to overwrite create-only artifact: {self.path / name}"
            ) from error
        except OSError as error:
            raise ActionPreservationLoopError(
                "cannot publish held stage output"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        identity = _identity_row(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != mode
            or identity != _identity_row(middle)
            or identity != _identity_row(after)
            or identity != _identity_row(named)
            or first != payload
            or second != payload
        ):
            raise ActionPreservationLoopError(
                "held stage output publication differs"
            )
        self.replay()
        return {
            "path": str(self.path / name),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload), "mode": mode, "identity": identity,
        }

    def exists(self, name: str) -> bool:
        try:
            os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError:
            pass


def _publish_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    _plain_directory(path.parent, label="publication parent")
    payload = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError as error:
        raise ActionPreservationLoopError(f"refusing to overwrite create-only artifact: {path}") from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ActionPreservationLoopError("create-only publication made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.read_bytes() != payload:
        raise ActionPreservationLoopError("create-only publication reread differs")


def publish_stage_plan(plan: Mapping[str, Any]) -> Path:
    """Create the fresh stage root and its immutable local plan."""

    authority = validate_stage_plan(plan)
    root = Path(authority["stage_root"])
    if os.path.lexists(root):
        raise ActionPreservationLoopError(f"stage root is not fresh: {root}")
    _plain_directory(root.parent, label="stage root parent")
    try:
        os.mkdir(root, 0o700)
    except FileExistsError as error:
        raise ActionPreservationLoopError(f"stage root is not fresh: {root}") from error
    output = root / PLAN_FILENAME
    _publish_create_only_json(output, authority)
    return output


def _verify_published_plan(plan: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    authority = validate_stage_plan(plan)
    root = Path(authority["stage_root"])
    _plain_directory(root, label="stage root")
    plan_path = root / PLAN_FILENAME
    expected = canonical_json_bytes(authority) + b"\n"
    try:
        plan_info = plan_path.lstat()
        observed = plan_path.read_bytes()
    except OSError as error:
        raise ActionPreservationLoopError("published stage plan is unavailable") from error
    if not stat.S_ISREG(plan_info.st_mode) or stat.S_ISLNK(plan_info.st_mode):
        raise ActionPreservationLoopError("published stage plan is not a plain file")
    if observed != expected:
        raise ActionPreservationLoopError("published stage plan differs")
    return authority, root


def publish_stage_receipt(plan: Mapping[str, Any], receipt: Mapping[str, Any]) -> Path:
    authority, root = _verify_published_plan(plan)
    validated = validate_stage_receipt(receipt, plan=authority)
    output = root / STAGE_RECEIPT_FILENAME
    _publish_create_only_json(output, validated)
    return output


def publish_transition_receipt(
    plan: Mapping[str, Any], transition: Mapping[str, Any]
) -> Path:
    authority, root = _verify_published_plan(plan)
    validated = validate_transition_receipt(transition, plan=authority)
    if validated["stage_receipt_digest"] is not None:
        stage_receipt_path = root / STAGE_RECEIPT_FILENAME
        try:
            receipt_info = stage_receipt_path.lstat()
            receipt_value = _load(stage_receipt_path, label="published stage receipt")
        except OSError as error:
            raise ActionPreservationLoopError("published stage receipt is unavailable") from error
        if not stat.S_ISREG(receipt_info.st_mode) or stat.S_ISLNK(receipt_info.st_mode):
            raise ActionPreservationLoopError("published stage receipt is not a plain file")
        receipt = validate_stage_receipt(receipt_value, plan=authority)
        if validated["stage_receipt_digest"] != receipt["stage_receipt_digest"]:
            raise ActionPreservationLoopError("transition does not bind published stage receipt")
    final_path = root / TRANSITION_FILENAME
    machine_wait_path = root / MACHINE_WAIT_TRANSITION_FILENAME
    blind_wait_path = root / WAIT_TRANSITION_FILENAME
    if validated["next_action"] == WAIT_FOR_MACHINE_EVIDENCE:
        if os.path.lexists(final_path) or os.path.lexists(blind_wait_path):
            raise ActionPreservationLoopError("a later transition already exists")
        output = machine_wait_path
    elif validated["next_action"] == WAIT_FOR_BLIND_REVIEW:
        if os.path.lexists(final_path):
            raise ActionPreservationLoopError("final transition already exists")
        if os.path.lexists(machine_wait_path):
            machine_waiting = validate_transition_receipt(
                _load(machine_wait_path, label="published machine-wait transition"),
                plan=authority,
            )
            if machine_waiting["next_action"] != WAIT_FOR_MACHINE_EVIDENCE:
                raise ActionPreservationLoopError("published machine-wait state differs")
            if any(
                machine_waiting[key] != validated[key]
                for key in ("stage_receipt_digest", "terminal_checkpoint_sha256")
            ):
                raise ActionPreservationLoopError(
                    "blind-review wait changes machine-wait stage binding"
                )
        output = blind_wait_path
    else:
        wait_path = blind_wait_path
        if os.path.lexists(wait_path):
            waiting = validate_transition_receipt(
                _load(wait_path, label="published waiting transition"), plan=authority
            )
            if waiting["next_action"] != WAIT_FOR_BLIND_REVIEW:
                raise ActionPreservationLoopError("published waiting transition state differs")
            locked_fields = (
                "stage_receipt_digest",
                "terminal_checkpoint_sha256",
            )
            if any(waiting[key] != validated[key] for key in locked_fields):
                raise ActionPreservationLoopError("final transition changes waiting-stage binding")
            machine_evidence = (
                "measurement_digest",
                "calibration_digest",
                "decision_digest",
            )
            if any(
                waiting["evidence"][key] != validated["evidence"][key]
                for key in machine_evidence
            ):
                raise ActionPreservationLoopError("final transition changes waiting machine evidence")
        elif os.path.lexists(machine_wait_path):
            machine_waiting = validate_transition_receipt(
                _load(machine_wait_path, label="published machine-wait transition"),
                plan=authority,
            )
            if any(
                machine_waiting[key] != validated[key]
                for key in ("stage_receipt_digest", "terminal_checkpoint_sha256")
            ):
                raise ActionPreservationLoopError(
                    "final transition changes machine-wait stage binding"
                )
        output = final_path
    _publish_create_only_json(output, validated)
    return output


def _load(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ActionPreservationLoopError(f"cannot load {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise ActionPreservationLoopError(f"{label} root is not an object")
    return dict(value)


def _publication_result(
    *, command: str, authority: Mapping[str, Any],
    authority_file_sha256: str, output: Mapping[str, Any],
    output_object_digest: str,
) -> dict[str, Any]:
    value = {
        "schema_version": PUBLICATION_RESULT_SCHEMA,
        "command": command,
        "stage_root": authority["stage_root"],
        "stage_root_authority_path": str(
            Path(authority["stage_root"]) / STAGE_ROOT_AUTHORITY_FILENAME
        ),
        "stage_root_authority_sha256": _sha(
            authority_file_sha256, label="stage root authority file"
        ),
        "stage_root_authority_digest": authority["authority_digest"],
        "output_path": output["path"],
        "output_sha256": output["sha256"],
        "output_object_digest": _sha(
            output_object_digest, label="loop output object"
        ),
    }
    value["result_digest"] = object_sha256(value)
    return value


def _create_and_publish_stage_plan(
    *, work_root: Mapping[str, Any], plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authority_plan = validate_stage_plan(plan)
    held = _HeldStageRoot.create(
        work_root, Path(authority_plan["stage_root"])
    )
    try:
        plan_file = held.write_json(PLAN_FILENAME, authority_plan)
        authority: dict[str, Any] = {
            "schema_version": STAGE_ROOT_AUTHORITY_SCHEMA,
            "work_root_authority_digest": work_root[
                "work_root_authority_digest"
            ],
            "stage_root": authority_plan["stage_root"],
            "stage_root_name": Path(authority_plan["stage_root"]).name,
            "stage_root_immutable_identity": dict(
                held.immutable_identity
            ),
            "plan_path": plan_file["path"],
            "plan_sha256": plan_file["sha256"],
            "plan_digest": authority_plan["plan_digest"],
            "created_relative_to_inherited_work_root_fd": True,
        }
        authority["authority_digest"] = object_sha256(authority)
        authority = _validate_stage_root_authority(authority)
        authority_file = held.write_json(
            STAGE_ROOT_AUTHORITY_FILENAME, authority
        )
        held.replay()
        return authority, authority_file, plan_file
    finally:
        held.close()


def _open_stage_context(
    *, work_root: Mapping[str, Any], plan_path: str,
    plan_sha256: str, authority_path: str, authority_sha256: str,
) -> tuple[_HeldStageRoot, dict[str, Any], dict[str, Any]]:
    plan_file = Path(plan_path)
    authority_file = Path(authority_path)
    if (
        not plan_file.is_absolute()
        or plan_file.name != PLAN_FILENAME
        or authority_file != (
            plan_file.parent / STAGE_ROOT_AUTHORITY_FILENAME
        )
    ):
        raise ActionPreservationLoopError(
            "stage plan/authority paths differ"
        )
    held, authority = _HeldStageRoot.open(
        work_root,
        path=plan_file.parent,
        authority_path=authority_file,
        authority_sha256=authority_sha256,
    )
    try:
        raw, _ = held.read(
            PLAN_FILENAME, expected_sha256=plan_sha256,
            label="published stage plan",
        )
        plan = validate_stage_plan(
            _strict_json_bytes(raw, label="published stage plan")
        )
        if (
            authority["plan_path"] != str(plan_file)
            or authority["plan_sha256"] != plan_sha256
            or authority["plan_digest"] != plan["plan_digest"]
            or authority["stage_root"] != plan["stage_root"]
        ):
            raise ActionPreservationLoopError(
                "stage plan differs from stage root authority"
            )
        held.replay()
        return held, authority, plan
    except BaseException:
        held.close()
        raise


def _load_held_transition(
    held: _HeldStageRoot, *, name: str, expected_sha256: str,
    plan: Mapping[str, Any], label: str,
) -> dict[str, Any]:
    raw, _ = held.read(name, expected_sha256=expected_sha256, label=label)
    return validate_transition_receipt(
        _strict_json_bytes(raw, label=label), plan=plan
    )


def _publish_transition_receipt_held(
    *, held: _HeldStageRoot, plan: Mapping[str, Any],
    transition: Mapping[str, Any], machine_wait_sha256: str | None,
    blind_wait_sha256: str | None,
) -> tuple[str, dict[str, Any]]:
    validated = validate_transition_receipt(transition, plan=plan)
    final_exists = held.exists(TRANSITION_FILENAME)
    machine_exists = held.exists(MACHINE_WAIT_TRANSITION_FILENAME)
    blind_exists = held.exists(WAIT_TRANSITION_FILENAME)
    if final_exists:
        raise ActionPreservationLoopError("final transition already exists")
    if machine_exists is not (machine_wait_sha256 is not None):
        raise ActionPreservationLoopError(
            "machine-wait predecessor SHA closure differs"
        )
    if blind_exists is not (blind_wait_sha256 is not None):
        raise ActionPreservationLoopError(
            "blind-wait predecessor SHA closure differs"
        )
    machine_waiting = (
        _load_held_transition(
            held, name=MACHINE_WAIT_TRANSITION_FILENAME,
            expected_sha256=machine_wait_sha256, plan=plan,
            label="published machine-wait transition",
        )
        if machine_wait_sha256 is not None else None
    )
    blind_waiting = (
        _load_held_transition(
            held, name=WAIT_TRANSITION_FILENAME,
            expected_sha256=blind_wait_sha256, plan=plan,
            label="published blind-wait transition",
        )
        if blind_wait_sha256 is not None else None
    )
    if validated["next_action"] == WAIT_FOR_MACHINE_EVIDENCE:
        if machine_waiting is not None or blind_waiting is not None:
            raise ActionPreservationLoopError(
                "machine-wait transition is not the first transition"
            )
        output_name = MACHINE_WAIT_TRANSITION_FILENAME
    elif validated["next_action"] == WAIT_FOR_BLIND_REVIEW:
        if blind_waiting is not None:
            raise ActionPreservationLoopError(
                "blind-review wait transition already exists"
            )
        if machine_waiting is not None:
            if machine_waiting["next_action"] != WAIT_FOR_MACHINE_EVIDENCE:
                raise ActionPreservationLoopError(
                    "published machine-wait state differs"
                )
            if any(
                machine_waiting[key] != validated[key]
                for key in (
                    "stage_receipt_digest",
                    "terminal_checkpoint_sha256",
                )
            ):
                raise ActionPreservationLoopError(
                    "blind-review wait changes machine-wait stage binding"
                )
        output_name = WAIT_TRANSITION_FILENAME
    else:
        if blind_waiting is not None:
            if blind_waiting["next_action"] != WAIT_FOR_BLIND_REVIEW:
                raise ActionPreservationLoopError(
                    "published blind-wait state differs"
                )
            for key in (
                "stage_receipt_digest", "terminal_checkpoint_sha256"
            ):
                if blind_waiting[key] != validated[key]:
                    raise ActionPreservationLoopError(
                        "final transition changes waiting-stage binding"
                    )
            for key in (
                "measurement_digest", "calibration_digest",
                "decision_digest",
            ):
                if (
                    blind_waiting["evidence"][key]
                    != validated["evidence"][key]
                ):
                    raise ActionPreservationLoopError(
                        "final transition changes waiting machine evidence"
                    )
        elif machine_waiting is not None and any(
            machine_waiting[key] != validated[key]
            for key in (
                "stage_receipt_digest", "terminal_checkpoint_sha256"
            )
        ):
            raise ActionPreservationLoopError(
                "final transition changes machine-wait stage binding"
            )
        output_name = TRANSITION_FILENAME
    output = held.write_json(output_name, validated)
    return output_name, output


_HELD_STAGE_LIFETIMES: list[_HeldStageRoot] = []


def _close_registered_stage_roots(
    function: Callable[..., int]
) -> Callable[..., int]:
    def wrapped(*args: Any, **kwargs: Any) -> int:
        if _HELD_STAGE_LIFETIMES:
            raise ActionPreservationLoopError(
                "held stage lifetime registry is not empty"
            )
        try:
            return function(*args, **kwargs)
        finally:
            while _HELD_STAGE_LIFETIMES:
                _HELD_STAGE_LIFETIMES.pop().close()
    return wrapped


@_close_registered_stage_roots
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="publish one fresh local stage plan")
    init.add_argument("--stage-id", required=True)
    init.add_argument("--stage-root", required=True)
    init.add_argument("--provenance", required=True)
    init.add_argument("--provenance-sha256", required=True)
    init.add_argument("--parent-plan")
    init.add_argument("--parent-plan-sha256")
    init.add_argument("--parent-transition")
    init.add_argument("--parent-transition-sha256")

    seal = subparsers.add_parser("seal-stage", help="validate and publish a stage receipt")
    seal.add_argument("--plan", required=True)
    seal.add_argument("--plan-sha256", required=True)
    seal.add_argument("--stage-authority", required=True)
    seal.add_argument("--stage-authority-sha256", required=True)
    seal.add_argument("--receipt", required=True)
    seal.add_argument("--receipt-sha256", required=True)

    advance = subparsers.add_parser("advance", help="emit a fail-closed transition receipt")
    advance.add_argument("--plan", required=True)
    advance.add_argument("--plan-sha256", required=True)
    advance.add_argument("--stage-authority", required=True)
    advance.add_argument("--stage-authority-sha256", required=True)
    advance.add_argument("--stage-receipt")
    advance.add_argument("--stage-receipt-sha256")
    advance.add_argument("--measurement")
    advance.add_argument("--measurement-sha256")
    advance.add_argument("--calibration")
    advance.add_argument("--calibration-sha256")
    advance.add_argument("--decision")
    advance.add_argument("--decision-sha256")
    advance.add_argument("--blind-review")
    advance.add_argument("--blind-review-sha256")
    advance.add_argument("--evaluation-complete")
    advance.add_argument("--evaluation-complete-sha256")
    advance.add_argument("--public-packet")
    advance.add_argument("--public-packet-sha256")
    advance.add_argument("--private-mapping")
    advance.add_argument("--private-mapping-sha256")
    advance.add_argument("--physical-bindings")
    advance.add_argument("--physical-bindings-sha256")
    advance.add_argument("--aggregate-completion-anchor")
    advance.add_argument("--machine-wait-transition-sha256")
    advance.add_argument("--blind-wait-transition-sha256")

    args = parser.parse_args(argv)
    work_root = _load_loop_work_root_authority()
    if args.command == "init":
        provenance = _load_expected_json(
            args.provenance, args.provenance_sha256, label="provenance"
        )
        parent_values = (
            args.parent_plan, args.parent_plan_sha256,
            args.parent_transition, args.parent_transition_sha256,
        )
        if any(value is None for value in parent_values) and any(
            value is not None for value in parent_values
        ):
            raise ActionPreservationLoopError(
                "parent plan/transition paths and SHAs must be complete"
            )
        if args.parent_plan is None:
            plan = build_stage_plan(
                stage_id=args.stage_id,
                stage_index=0,
                stage_root=args.stage_root,
                input_provenance=provenance,
            )
        else:
            plan = build_next_stage_plan(
                _load_expected_json(
                    args.parent_plan, args.parent_plan_sha256,
                    label="parent plan",
                ),
                _load_expected_json(
                    args.parent_transition, args.parent_transition_sha256,
                    label="parent transition",
                ),
                stage_id=args.stage_id,
                stage_root=args.stage_root,
                input_provenance=provenance,
            )
        authority, authority_file, output_file = (
            _create_and_publish_stage_plan(work_root=work_root, plan=plan)
        )
        result = _publication_result(
            command="init", authority=authority,
            authority_file_sha256=authority_file["sha256"],
            output=output_file, output_object_digest=plan["plan_digest"],
        )
    elif args.command == "seal-stage":
        held, authority, plan = _open_stage_context(
            work_root=work_root, plan_path=args.plan,
            plan_sha256=args.plan_sha256,
            authority_path=args.stage_authority,
            authority_sha256=args.stage_authority_sha256,
        )
        _HELD_STAGE_LIFETIMES.append(held)
        try:
            receipt = validate_stage_receipt(
                _load_expected_json(
                    args.receipt, args.receipt_sha256,
                    label="stage receipt",
                ),
                plan=plan,
            )
            output_file = held.write_json(STAGE_RECEIPT_FILENAME, receipt)
            held.replay()
        finally:
            held.close()
            _HELD_STAGE_LIFETIMES.remove(held)
        result = _publication_result(
            command="seal-stage", authority=authority,
            authority_file_sha256=args.stage_authority_sha256,
            output=output_file,
            output_object_digest=receipt["stage_receipt_digest"],
        )
    else:
        held, authority, plan = _open_stage_context(
            work_root=work_root, plan_path=args.plan,
            plan_sha256=args.plan_sha256,
            authority_path=args.stage_authority,
            authority_sha256=args.stage_authority_sha256,
        )
        _HELD_STAGE_LIFETIMES.append(held)
        for label, path, digest in (
            ("stage receipt", args.stage_receipt, args.stage_receipt_sha256),
            ("measurement", args.measurement, args.measurement_sha256),
            ("calibration", args.calibration, args.calibration_sha256),
            ("decision", args.decision, args.decision_sha256),
        ):
            if (path is None) is not (digest is None):
                raise ActionPreservationLoopError(
                    f"{label} path and literal SHA must be supplied together"
                )
        if args.stage_receipt is None:
            if held.exists(STAGE_RECEIPT_FILENAME):
                raise ActionPreservationLoopError(
                    "published stage receipt requires its literal path/SHA"
                )
            stage_receipt_value = None
        else:
            expected_stage_receipt = held.path / STAGE_RECEIPT_FILENAME
            if Path(args.stage_receipt) != expected_stage_receipt:
                raise ActionPreservationLoopError(
                    "stage receipt is outside the held stage root"
                )
            stage_receipt_raw, _ = held.read(
                STAGE_RECEIPT_FILENAME,
                expected_sha256=args.stage_receipt_sha256,
                label="published stage receipt",
            )
            stage_receipt_value = validate_stage_receipt(
                _strict_json_bytes(
                    stage_receipt_raw, label="published stage receipt"
                ),
                plan=plan,
            )
        review_specific_args = (
            args.blind_review,
            args.blind_review_sha256,
            args.evaluation_complete,
            args.evaluation_complete_sha256,
            args.public_packet,
            args.public_packet_sha256,
            args.private_mapping,
            args.private_mapping_sha256,
            args.physical_bindings,
            args.physical_bindings_sha256,
            args.aggregate_completion_anchor,
        )
        review_authority_args = (
            args.decision, args.decision_sha256, *review_specific_args
        )
        if any(item is not None for item in review_specific_args) and any(
            item is None for item in review_authority_args
        ):
            raise ActionPreservationLoopError(
                "review advance requires the complete expected-SHA packet authority"
            )
        if args.blind_review:
            decision_value = gate._load_expected(
                args.decision, args.decision_sha256, label="machine decision"
            )
            blind_review_value = gate._load_expected(
                args.blind_review, args.blind_review_sha256,
                label="blind review",
            )
            evaluation_aggregate_value = gate._load_expected(
                args.evaluation_complete, args.evaluation_complete_sha256,
                label="evaluation complete",
            )
            public_packet_value = gate._load_expected(
                args.public_packet, args.public_packet_sha256,
                label="public packet",
            )
            private_mapping_value = gate._load_expected(
                args.private_mapping, args.private_mapping_sha256,
                label="private mapping",
            )
            eval_bindings = gate._verify_cli_eval_release(
                physical_bindings_path=args.physical_bindings,
                physical_bindings_sha256=args.physical_bindings_sha256,
                evaluation_aggregate=evaluation_aggregate_value,
                public_packet=public_packet_value,
                private_mapping=private_mapping_value,
                evaluation_aggregate_path=args.evaluation_complete,
                evaluation_aggregate_sha256=(
                    args.evaluation_complete_sha256
                ),
                public_packet_path=args.public_packet,
                public_packet_sha256=args.public_packet_sha256,
                private_mapping_path=args.private_mapping,
                private_mapping_sha256=args.private_mapping_sha256,
                aggregate_completion_anchor_raw=args.aggregate_completion_anchor,
                work_root=work_root,
                expected_work_root_target=LOOP_RUNTIME_TARGET,
            )
            try:
                import action_preservation_decoded_eval_bridge_v1 as eval_bridge

                eval_bridge.validate_running_verified_capture(
                    eval_bindings,
                    target="action_preservation_loop_controller_v1.py",
                    expected_arguments=list(
                        sys.argv[1:] if argv is None else argv
                    ),
                    verify_file=True,
                )
            except eval_bridge.DecodedEvaluationBridgeError as error:
                raise ActionPreservationLoopError(str(error)) from error
        else:
            decision_value = (
                gate._load_expected(
                    args.decision, args.decision_sha256, label="machine decision"
                )
                if args.decision and args.decision_sha256 else None
            )
            blind_review_value = None
            evaluation_aggregate_value = None
            public_packet_value = None
            private_mapping_value = None
        transition = decide_next_action(
            plan,
            stage_receipt=stage_receipt_value,
            measurement=_load_expected_json(
                args.measurement, args.measurement_sha256,
                label="measurement",
            )
            if args.measurement
            else None,
            calibration=_load_expected_json(
                args.calibration, args.calibration_sha256,
                label="calibration",
            )
            if args.calibration
            else None,
            decision=decision_value,
            blind_review=blind_review_value,
            evaluation_aggregate=evaluation_aggregate_value,
            public_packet=public_packet_value,
            private_mapping=private_mapping_value,
        )
        try:
            _, output_file = _publish_transition_receipt_held(
                held=held, plan=plan, transition=transition,
                machine_wait_sha256=args.machine_wait_transition_sha256,
                blind_wait_sha256=args.blind_wait_transition_sha256,
            )
            held.replay()
        finally:
            held.close()
            _HELD_STAGE_LIFETIMES.remove(held)
        result = _publication_result(
            command="advance", authority=authority,
            authority_file_sha256=args.stage_authority_sha256,
            output=output_file,
            output_object_digest=transition["transition_digest"],
        )
    _replay_loop_work_root_authority(work_root)
    print(canonical_json_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
