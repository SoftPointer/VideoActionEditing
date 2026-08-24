#!/usr/bin/env python3
"""Fail-closed Stage-B event-reward consumption for Bernini SAIC.

This module is deliberately *not* a media evaluator.  Its complete input
closure is:

1. one opaque, frozen critic checkpoint (read only to bind its bytes),
2. one digest-sealed qualification receipt, and
3. one digest-sealed packet of scalar scores from a fresh on-policy rollout.

Video paths/bytes, frames, latents, noise, targets, proposals, event-bank
artifacts, masks, flow, pose, and trajectories are outside this boundary.  A
caller must run the frozen critic elsewhere and cross the boundary with only
the four-stage scalar scores defined below.

The event reward is the weakest (not mean) phase margin between the action
score and the strongest hard negative at that phase.  A single candidate can
only become eligible for later same-round Y+/Y- relative pairing; this module
never authorizes an optimizer step.  A strict absolute pass at all four phases
is necessary to enter the inverse cycle or satisfy the event side of
checkpoint publication.  Source/identity constraints remain an independent,
non-compensating publication prerequisite.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


QUALIFICATION_SCHEMA_VERSION = "bernini-saic-event-critic-qualification-v1"
CANDIDATE_SCORE_SCHEMA_VERSION = "bernini-saic-on-policy-event-scores-v1"
RESULT_SCHEMA_VERSION = "bernini-saic-event-reward-decision-v1"

PHASE_ORDER = ("onset", "transition", "completion", "hold")
NEGATIVE_ORDER = ("reverse", "incomplete", "camera_only", "appearance_only")
HOLDOUT_ORDER = ("identity", "scene", "seed", "action_family")
SCORE_ARM_ORDER = ("action", *NEGATIVE_ORDER)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,191}$")
_MAX_RECEIPT_BYTES = 2 * 1024 * 1024
_MAX_ABS_SCORE = 1.0e6


class SAICEventRewardError(RuntimeError):
    """Raised before an unqualified or out-of-closure input gains authority."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic finite JSON bytes or fail closed."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise SAICEventRewardError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fail(message: str) -> None:
    raise SAICEventRewardError(message)


def _exact_keys(value: Any, expected: Sequence[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        _fail(f"{label} keys differ: missing={missing} extra={extra}")
    return value


def _exact_order(value: Any, expected: Sequence[str], *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _fail(f"{label} must be a string list")
    actual = tuple(value)
    wanted = tuple(expected)
    if actual != wanted:
        _fail(f"{label} differs: expected={wanted} actual={actual}")
    return actual


def _strict_bool(value: Any, *, label: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be a boolean")
    if expected is not None and value is not expected:
        _fail(f"{label} must be {expected}")
    return value


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or abs(result) > _MAX_ABS_SCORE:
        _fail(f"{label} must be finite with absolute value <= {_MAX_ABS_SCORE:g}")
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        _fail(f"{label} must be a safe opaque identifier")
    return value


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)


def _regular_file_identity(path_like: str | os.PathLike[str], *, label: str) -> tuple[Path, os.stat_result]:
    path = Path(path_like)
    try:
        before = path.lstat()
    except OSError as error:
        raise SAICEventRewardError(f"cannot stat {label}") from error
    if stat.S_ISLNK(before.st_mode):
        _fail(f"{label} must not be a symbolic link")
    if not stat.S_ISREG(before.st_mode):
        _fail(f"{label} must be a regular file")
    return path, before


def _hash_regular_file(path_like: str | os.PathLike[str], *, label: str) -> tuple[str, int]:
    """Hash a regular non-symlink file while detecting a concurrent rewrite."""

    path, before = _regular_file_identity(path_like, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SAICEventRewardError(f"cannot open {label}") from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            _fail(f"{label} changed away from a regular file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            _fail(f"{label} changed while being opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields_before = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    stable_fields_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if stable_fields_before != stable_fields_after:
        _fail(f"{label} changed while it was read")
    return digest.hexdigest(), int(after.st_size)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _fail(f"JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def _read_small_regular_file(
    path_like: str | os.PathLike[str], *, label: str
) -> bytes:
    """Read a small non-symlink file without a path-swap read window."""

    path, before = _regular_file_identity(path_like, label=label)
    if before.st_size > _MAX_RECEIPT_BYTES:
        _fail(f"{label} exceeds {_MAX_RECEIPT_BYTES} bytes")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SAICEventRewardError(f"cannot open {label}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            _fail(f"{label} changed away from a regular file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            _fail(f"{label} changed while being opened")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, _MAX_RECEIPT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_RECEIPT_BYTES:
                _fail(f"{label} exceeds {_MAX_RECEIPT_BYTES} bytes")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        _fail(f"{label} changed while it was read")
    return b"".join(chunks)


def _load_json_regular_file(
    path_like: str | os.PathLike[str], *, label: str
) -> tuple[dict[str, Any], str]:
    raw = _read_small_regular_file(path_like, label=label)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SAICEventRewardError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        _fail(f"{label} root must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_phase_scores(value: Any, *, label: str) -> dict[str, float]:
    row = _exact_keys(value, PHASE_ORDER, label=label)
    return {phase: _finite(row[phase], label=f"{label}.{phase}") for phase in PHASE_ORDER}


def _validate_thresholds(value: Any) -> dict[str, Any]:
    row = _exact_keys(
        value,
        (
            "qualification_margin_floor",
            "bootstrap_relative_margin_floor",
            "absolute_action_score_floors",
            "absolute_margin_floors",
        ),
        label="qualification.thresholds",
    )
    qualification_floor = _finite(
        row["qualification_margin_floor"], label="qualification margin floor"
    )
    bootstrap_floor = _finite(
        row["bootstrap_relative_margin_floor"], label="bootstrap relative margin floor"
    )
    if qualification_floor < 0.0 or bootstrap_floor <= 0.0:
        _fail("qualification margin floor must be >=0 and bootstrap floor must be >0")
    absolute_scores = _validate_phase_scores(
        row["absolute_action_score_floors"], label="absolute action score floors"
    )
    absolute_margins = _validate_phase_scores(
        row["absolute_margin_floors"], label="absolute margin floors"
    )
    if any(value < 0.0 for value in absolute_margins.values()):
        _fail("absolute margin floors must be non-negative")
    return {
        "qualification_margin_floor": qualification_floor,
        "bootstrap_relative_margin_floor": bootstrap_floor,
        "absolute_action_score_floors": absolute_scores,
        "absolute_margin_floors": absolute_margins,
    }


def validate_qualification_receipt(
    value: Any,
    *,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
) -> dict[str, Any]:
    """Validate the complete qualification cross-product and checkpoint binding."""

    root = _exact_keys(
        value,
        (
            "schema_version",
            "critic_checkpoint",
            "phase_order",
            "negative_order",
            "holdout_order",
            "holdout_summary",
            "coverage_cells",
            "thresholds",
            "authority_contract",
            "receipt_digest",
        ),
        label="qualification receipt",
    )
    if root["schema_version"] != QUALIFICATION_SCHEMA_VERSION:
        _fail("qualification schema version differs")
    _exact_order(root["phase_order"], PHASE_ORDER, label="qualification phase order")
    _exact_order(root["negative_order"], NEGATIVE_ORDER, label="qualification negative order")
    _exact_order(root["holdout_order"], HOLDOUT_ORDER, label="qualification holdout order")

    declared_digest = _sha256(root["receipt_digest"], label="qualification receipt digest")
    unsigned = dict(root)
    del unsigned["receipt_digest"]
    if object_sha256(unsigned) != declared_digest:
        _fail("qualification receipt digest differs")

    checkpoint = _exact_keys(
        root["critic_checkpoint"],
        (
            "content_sha256",
            "byte_size",
            "state_dict_kind",
            "eval_mode",
            "requires_grad_parameter_count",
            "optimizer_state_present",
        ),
        label="qualification critic checkpoint",
    )
    if _sha256(checkpoint["content_sha256"], label="qualified checkpoint digest") != checkpoint_sha256:
        _fail("qualification is bound to different critic checkpoint bytes")
    if _strict_int(checkpoint["byte_size"], label="qualified checkpoint byte size") != checkpoint_bytes:
        _fail("qualification checkpoint byte size differs")
    if checkpoint["state_dict_kind"] != "critic_only_no_optimizer":
        _fail("qualification checkpoint must be critic-only")
    _strict_bool(checkpoint["eval_mode"], label="critic eval mode", expected=True)
    if _strict_int(
        checkpoint["requires_grad_parameter_count"],
        label="critic requires-grad parameter count",
    ) != 0:
        _fail("qualified critic is not frozen")
    _strict_bool(
        checkpoint["optimizer_state_present"],
        label="critic optimizer-state presence",
        expected=False,
    )

    summary = _exact_keys(root["holdout_summary"], HOLDOUT_ORDER, label="holdout summary")
    normalized_summary: dict[str, Any] = {}
    for dimension in HOLDOUT_ORDER:
        item = _exact_keys(
            summary[dimension],
            ("held_out_unit_count", "fit_overlap_count", "passed"),
            label=f"{dimension} holdout summary",
        )
        held_out_count = _strict_int(
            item["held_out_unit_count"], label=f"{dimension} held-out unit count", minimum=1
        )
        if _strict_int(item["fit_overlap_count"], label=f"{dimension} fit overlap count") != 0:
            _fail(f"{dimension} holdout overlaps critic fitting")
        _strict_bool(item["passed"], label=f"{dimension} holdout pass", expected=True)
        normalized_summary[dimension] = {
            "held_out_unit_count": held_out_count,
            "fit_overlap_count": 0,
            "passed": True,
        }

    thresholds = _validate_thresholds(root["thresholds"])
    cells = root["coverage_cells"]
    if not isinstance(cells, list):
        _fail("qualification coverage cells must be a list")
    expected_pairs = [(dimension, negative) for dimension in HOLDOUT_ORDER for negative in NEGATIVE_ORDER]
    if len(cells) != len(expected_pairs):
        _fail("qualification must contain the complete holdout-by-negative cross-product")
    normalized_cells: list[dict[str, Any]] = []
    seen: list[tuple[str, str]] = []
    for index, item in enumerate(cells):
        cell = _exact_keys(
            item,
            ("holdout_dimension", "negative_kind", "sample_count", "stage_margins", "weakest_margin", "passed"),
            label=f"qualification coverage cell {index}",
        )
        pair = (cell["holdout_dimension"], cell["negative_kind"])
        seen.append(pair)
        sample_count = _strict_int(cell["sample_count"], label=f"coverage cell {index} sample count", minimum=1)
        stage_margins = _validate_phase_scores(
            cell["stage_margins"], label=f"coverage cell {index} stage margins"
        )
        weakest = _finite(cell["weakest_margin"], label=f"coverage cell {index} weakest margin")
        actual_weakest = min(stage_margins.values())
        if not _same_float(weakest, actual_weakest):
            _fail(f"coverage cell {index} weakest margin is not the four-stage minimum")
        passed = all(
            margin >= thresholds["qualification_margin_floor"] for margin in stage_margins.values()
        )
        if _strict_bool(cell["passed"], label=f"coverage cell {index} pass") is not passed or not passed:
            _fail(f"coverage cell {index} failed qualification")
        normalized_cells.append(
            {
                "holdout_dimension": pair[0],
                "negative_kind": pair[1],
                "sample_count": sample_count,
                "stage_margins": stage_margins,
                "weakest_margin": weakest,
                "passed": True,
            }
        )
    if seen != expected_pairs:
        _fail("qualification coverage cells are missing, duplicated, or out of canonical order")

    authority = _exact_keys(
        root["authority_contract"],
        (
            "score_only_runtime_boundary",
            "receipt_alone_authorizes_optimizer",
            "receipt_alone_authorizes_inverse",
            "receipt_alone_authorizes_publication",
            "bootstrap_scope",
            "absolute_four_stage_pass_required_for_inverse",
            "absolute_four_stage_pass_required_for_publication",
            "external_source_constraints_still_required",
        ),
        label="qualification authority contract",
    )
    _strict_bool(
        authority["score_only_runtime_boundary"],
        label="score-only runtime boundary",
        expected=True,
    )
    for field in (
        "receipt_alone_authorizes_optimizer",
        "receipt_alone_authorizes_inverse",
        "receipt_alone_authorizes_publication",
    ):
        _strict_bool(authority[field], label=field.replace("_", " "), expected=False)
    if authority["bootstrap_scope"] != "same_round_relative_pairing_only":
        _fail("bootstrap authority scope differs")
    for field in (
        "absolute_four_stage_pass_required_for_inverse",
        "absolute_four_stage_pass_required_for_publication",
        "external_source_constraints_still_required",
    ):
        _strict_bool(authority[field], label=field.replace("_", " "), expected=True)

    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "critic_checkpoint": dict(checkpoint),
        "phase_order": list(PHASE_ORDER),
        "negative_order": list(NEGATIVE_ORDER),
        "holdout_order": list(HOLDOUT_ORDER),
        "holdout_summary": normalized_summary,
        "coverage_cells": normalized_cells,
        "thresholds": thresholds,
        "authority_contract": dict(authority),
        "receipt_digest": declared_digest,
    }


def validate_candidate_scores(
    value: Any,
    *,
    checkpoint_sha256: str,
    qualification_receipt_digest: str,
) -> dict[str, Any]:
    """Accept only scalar scores from a fresh current-policy exact81 rollout."""

    if not isinstance(value, Mapping):
        _fail("candidate scores must be an in-memory object, never a path or media artifact")
    root = _exact_keys(
        value,
        (
            "schema_version",
            "candidate_id",
            "rollout_id",
            "action_family",
            "policy_checkpoint_sha256",
            "critic_checkpoint_sha256",
            "qualification_receipt_digest",
            "rollout_contract",
            "phase_order",
            "negative_order",
            "scores",
            "score_packet_digest",
        ),
        label="candidate score packet",
    )
    if root["schema_version"] != CANDIDATE_SCORE_SCHEMA_VERSION:
        _fail("candidate score schema version differs")
    candidate_id = _safe_id(root["candidate_id"], label="candidate id")
    rollout_id = _safe_id(root["rollout_id"], label="rollout id")
    action_family = _safe_id(root["action_family"], label="action family")
    policy_digest = _sha256(root["policy_checkpoint_sha256"], label="policy checkpoint digest")
    critic_digest = _sha256(root["critic_checkpoint_sha256"], label="candidate critic checkpoint digest")
    if critic_digest != checkpoint_sha256:
        _fail("candidate scores were produced by a different critic checkpoint")
    receipt_digest = _sha256(
        root["qualification_receipt_digest"], label="candidate qualification receipt digest"
    )
    if receipt_digest != qualification_receipt_digest:
        _fail("candidate scores are bound to a different qualification receipt")
    _exact_order(root["phase_order"], PHASE_ORDER, label="candidate phase order")
    _exact_order(root["negative_order"], NEGATIVE_ORDER, label="candidate negative order")

    declared_digest = _sha256(root["score_packet_digest"], label="candidate score packet digest")
    unsigned = dict(root)
    del unsigned["score_packet_digest"]
    if object_sha256(unsigned) != declared_digest:
        _fail("candidate score packet digest differs")

    contract = _exact_keys(
        root["rollout_contract"],
        (
            "on_policy",
            "fresh_after_latest_update",
            "source_coordinate",
            "decoded_exact81",
            "frame_count",
            "scores_computed_by_frozen_critic",
            "event_bank_candidate",
            "payload_kind",
            "media_or_path_attached",
            "latent_attached",
            "noise_attached",
            "target_attached",
            "proposal_attached",
        ),
        label="candidate rollout contract",
    )
    _strict_bool(contract["on_policy"], label="candidate on-policy flag", expected=True)
    _strict_bool(
        contract["fresh_after_latest_update"], label="candidate freshness flag", expected=True
    )
    if contract["source_coordinate"] != "current_policy_rv2v":
        _fail("candidate is not in the current-policy RV2V source coordinate")
    _strict_bool(contract["decoded_exact81"], label="decoded exact81 flag", expected=True)
    if _strict_int(contract["frame_count"], label="candidate frame count", minimum=1) != 81:
        _fail("candidate must be exact81")
    _strict_bool(
        contract["scores_computed_by_frozen_critic"],
        label="frozen critic scoring flag",
        expected=True,
    )
    _strict_bool(contract["event_bank_candidate"], label="event-bank candidate flag", expected=False)
    if contract["payload_kind"] != "scalar_stage_scores_only":
        _fail("candidate payload is not scalar-stage-score-only")
    for field in (
        "media_or_path_attached",
        "latent_attached",
        "noise_attached",
        "target_attached",
        "proposal_attached",
    ):
        _strict_bool(contract[field], label=field.replace("_", " "), expected=False)

    scores_root = _exact_keys(root["scores"], SCORE_ARM_ORDER, label="candidate scores")
    scores = {
        arm: _validate_phase_scores(scores_root[arm], label=f"candidate {arm} scores")
        for arm in SCORE_ARM_ORDER
    }
    return {
        "schema_version": CANDIDATE_SCORE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "rollout_id": rollout_id,
        "action_family": action_family,
        "policy_checkpoint_sha256": policy_digest,
        "critic_checkpoint_sha256": critic_digest,
        "qualification_receipt_digest": receipt_digest,
        "rollout_contract": dict(contract),
        "phase_order": list(PHASE_ORDER),
        "negative_order": list(NEGATIVE_ORDER),
        "scores": scores,
        "score_packet_digest": declared_digest,
    }


@dataclass(frozen=True)
class SAICEventRewardBoundary:
    """A hash-bound critic qualification with no media-loading capability."""

    critic_checkpoint_path: Path
    qualification_receipt_path: Path
    critic_checkpoint_sha256: str
    critic_checkpoint_bytes: int
    qualification_file_sha256: str
    qualification_receipt_digest: str
    qualification: Mapping[str, Any]

    def _revalidate_files(self) -> dict[str, Any]:
        checkpoint_sha, checkpoint_bytes = _hash_regular_file(
            self.critic_checkpoint_path, label="critic checkpoint"
        )
        if (
            checkpoint_sha != self.critic_checkpoint_sha256
            or checkpoint_bytes != self.critic_checkpoint_bytes
        ):
            _fail("critic checkpoint changed after the reward boundary was loaded")
        receipt, file_sha = _load_json_regular_file(
            self.qualification_receipt_path, label="qualification receipt"
        )
        if file_sha != self.qualification_file_sha256:
            _fail("qualification receipt changed after the reward boundary was loaded")
        validated = validate_qualification_receipt(
            receipt,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_bytes=checkpoint_bytes,
        )
        if validated["receipt_digest"] != self.qualification_receipt_digest:
            _fail("qualification receipt identity changed")
        return validated

    def evaluate(self, candidate_scores: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
        """Evaluate one current on-policy scalar-score packet.

        ``mode='bootstrap'`` can only make a candidate eligible for a separate
        same-round relative pair builder.  This consumer never authorizes an
        optimizer step.  ``mode='strict'`` requires an absolute four-stage
        pass before inverse entry or event-side publication is authorized.
        """

        if mode not in ("bootstrap", "strict"):
            _fail("event reward mode must be 'bootstrap' or 'strict'")
        qualification = self._revalidate_files()
        candidate = validate_candidate_scores(
            candidate_scores,
            checkpoint_sha256=self.critic_checkpoint_sha256,
            qualification_receipt_digest=self.qualification_receipt_digest,
        )
        scores = candidate["scores"]
        # Always consume the freshly revalidated receipt, not the public
        # dataclass snapshot, so nested caller mutation cannot lower a gate.
        thresholds = qualification["thresholds"]

        strongest_negative: dict[str, float] = {}
        strongest_negative_kind: dict[str, str] = {}
        stage_margins: dict[str, float] = {}
        for phase in PHASE_ORDER:
            # Tuple comparison keeps tie handling deterministic in NEGATIVE_ORDER.
            best_index, best_kind, best_score = max(
                (
                    (index, negative, scores[negative][phase])
                    for index, negative in enumerate(NEGATIVE_ORDER)
                ),
                key=lambda row: (row[2], -row[0]),
            )
            del best_index
            strongest_negative[phase] = best_score
            strongest_negative_kind[phase] = best_kind
            stage_margins[phase] = scores["action"][phase] - best_score

        weakest_phase = min(PHASE_ORDER, key=lambda phase: (stage_margins[phase], PHASE_ORDER.index(phase)))
        weakest_margin = stage_margins[weakest_phase]
        relative_pass_by_phase = {
            phase: stage_margins[phase] >= thresholds["bootstrap_relative_margin_floor"]
            for phase in PHASE_ORDER
        }
        relative_pass = all(relative_pass_by_phase.values())
        absolute_action_pass_by_phase = {
            phase: scores["action"][phase] >= thresholds["absolute_action_score_floors"][phase]
            for phase in PHASE_ORDER
        }
        absolute_margin_pass_by_phase = {
            phase: stage_margins[phase] >= thresholds["absolute_margin_floors"][phase]
            for phase in PHASE_ORDER
        }
        absolute_pass_by_phase = {
            phase: absolute_action_pass_by_phase[phase] and absolute_margin_pass_by_phase[phase]
            for phase in PHASE_ORDER
        }
        absolute_four_stage_pass = all(absolute_pass_by_phase.values())

        bootstrap = mode == "bootstrap"
        relative_pairing_eligible = bootstrap and relative_pass
        inverse_authorized = (not bootstrap) and absolute_four_stage_pass
        event_publication_authorized = (not bootstrap) and absolute_four_stage_pass
        unsigned = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "candidate_id": candidate["candidate_id"],
            "rollout_id": candidate["rollout_id"],
            "action_family": candidate["action_family"],
            "mode": mode,
            "critic_checkpoint_sha256": self.critic_checkpoint_sha256,
            "qualification_receipt_digest": self.qualification_receipt_digest,
            "candidate_score_packet_digest": candidate["score_packet_digest"],
            "stage_margins": stage_margins,
            "strongest_negative_scores": strongest_negative,
            "strongest_negative_kinds": strongest_negative_kind,
            "weakest_phase": weakest_phase,
            "weakest_margin": weakest_margin,
            "event_reward_is_weakest_four_stage_margin": True,
            "relative_margin_pass_by_phase": relative_pass_by_phase,
            "relative_action_margin_pass": relative_pass,
            "relative_pairing_eligible": relative_pairing_eligible,
            "absolute_action_pass_by_phase": absolute_action_pass_by_phase,
            "absolute_margin_pass_by_phase": absolute_margin_pass_by_phase,
            "absolute_four_stage_pass_by_phase": absolute_pass_by_phase,
            "absolute_four_stage_pass": absolute_four_stage_pass,
            "authority": {
                "optimizer_update_authorized": False,
                "optimizer_scope": "none_single_candidate_consumer",
                "same_round_y_plus_y_minus_pair_builder_required": True,
                "seven_noncompensating_axes_required_for_optimizer": True,
                "inverse_cycle_entry_authorized": inverse_authorized,
                "event_side_checkpoint_publication_authorized": event_publication_authorized,
                "global_checkpoint_publication_authorized": False,
                "external_source_constraints_still_required": True,
                "candidate_media_or_training_target_authorized": False,
            },
        }
        return {**unsigned, "decision_digest": object_sha256(unsigned)}


def load_event_reward_boundary(
    critic_checkpoint_path: str | os.PathLike[str],
    qualification_receipt_path: str | os.PathLike[str],
) -> SAICEventRewardBoundary:
    """Load and bind the only two filesystem artifacts accepted by Stage-B."""

    checkpoint_path = Path(critic_checkpoint_path)
    receipt_path = Path(qualification_receipt_path)
    checkpoint_sha, checkpoint_bytes = _hash_regular_file(
        checkpoint_path, label="critic checkpoint"
    )
    receipt, receipt_file_sha = _load_json_regular_file(
        receipt_path, label="qualification receipt"
    )
    qualification = validate_qualification_receipt(
        receipt,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_bytes=checkpoint_bytes,
    )
    return SAICEventRewardBoundary(
        critic_checkpoint_path=checkpoint_path.resolve(),
        qualification_receipt_path=receipt_path.resolve(),
        critic_checkpoint_sha256=checkpoint_sha,
        critic_checkpoint_bytes=checkpoint_bytes,
        qualification_file_sha256=receipt_file_sha,
        qualification_receipt_digest=qualification["receipt_digest"],
        qualification=qualification,
    )


def consume_event_reward(
    *,
    critic_checkpoint_path: str | os.PathLike[str],
    qualification_receipt_path: str | os.PathLike[str],
    candidate_scores: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """One-shot convenience API with the same closed input surface."""

    return load_event_reward_boundary(
        critic_checkpoint_path,
        qualification_receipt_path,
    ).evaluate(candidate_scores, mode=mode)


__all__ = [
    "CANDIDATE_SCORE_SCHEMA_VERSION",
    "HOLDOUT_ORDER",
    "NEGATIVE_ORDER",
    "PHASE_ORDER",
    "QUALIFICATION_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SAICEventRewardBoundary",
    "SAICEventRewardError",
    "canonical_json_bytes",
    "consume_event_reward",
    "load_event_reward_boundary",
    "object_sha256",
    "validate_candidate_scores",
    "validate_qualification_receipt",
]
