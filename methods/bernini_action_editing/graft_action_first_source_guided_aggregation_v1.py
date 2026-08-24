#!/usr/bin/env python3
"""Action-first source-guided aggregation (ASGA) for GRAFT-Edit.

This module is a fail-closed *research observation* primitive.  It aggregates
detached action-only phase programs; it never consumes generated proposal RGB,
proposal latents, raw velocities, a target video, masks, pose, flow, tracks, or
an evaluator-selected index.

The selector is lexicographic:

1. seven fixed action/completion/preservation margins are computed internally
   from a fixed raw-score schema;
2. every margin must be strictly positive;
3. source compatibility is a normalized cosine observation used only inside
   the feasible set; and
4. an empty feasible set abstains without a best-of-K fallback.

Canonical receipts and SHA-256 digests bind observed bytes and declared input
boundaries.  They are not signatures and are not a same-process security
boundary.  This module therefore never authorizes an optimizer update or a
semantic action-editing claim; a separately sealed, calibrated outer authority
is required before training may consume an accepted observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch


METHOD = "bernini-graft-action-first-source-guided-aggregation-v2"
SCHEMA_VERSION = "bernini-graft-asga-research-observation-v2"
RETELLING_SCHEMA_VERSION = "bernini-graft-source-retelling-binding-v2"
CAPTIONER_RECEIPT_SCHEMA_VERSION = "bernini-graft-captioner-observation-v1"
COMPILER_RECEIPT_SCHEMA_VERSION = "bernini-graft-target-compiler-observation-v1"
PROPOSAL_BANK_SCHEMA_VERSION = "bernini-graft-action-program-bank-v2"
EXTRACTOR_RECEIPT_SCHEMA_VERSION = "bernini-graft-program-extractor-observation-v1"
EVIDENCE_SCHEMA_VERSION = "bernini-graft-asga-evidence-v2"
SCORER_RECEIPT_SCHEMA_VERSION = "bernini-graft-asga-scorer-observation-v1"
SELECTION_SCHEMA_VERSION = "bernini-graft-asga-selection-v2"

# K is the number of independently seeded candidates.  P is the number of
# ordered phases inside each candidate.  They are deliberately distinct.
CANDIDATE_COUNT = 5
PHASE_COUNT = 4
PHASE_ORDER = ("onset", "transition", "terminal", "hold")
PROGRAM_WIDTH = 32
BRANCH_COUNT = 4
COUNTERFACTUAL_BRANCH_ORDER = ("action", "noop", "reverse", "incomplete")
PROGRAM_COORDINATE = "detached_action_only_phase_program_k5_p4_d32_v1"

TEMPERATURE = 0.01
SOURCE_COMPATIBILITY_KIND = "normalized_source_cosine_observation_v1"
NORMALIZED_SCORE_MIN = -1.0
NORMALIZED_SCORE_MAX = 1.0
MAX_PORTABLE_SEED = (1 << 63) - 1

RAW_SCORE_NAMES = (
    "action_event_score",
    "noop_target_event_score",
    "reverse_target_event_score",
    "incomplete_target_event_score",
    "terminal_hold_score",
    "actor_preservation_delta",
    "camera_preservation_delta",
    "background_preservation_delta",
)
MARGIN_NAMES = (
    "action_minus_noop",
    "action_minus_reverse",
    "action_minus_incomplete",
    "terminal_hold",
    "actor_preservation",
    "camera_preservation",
    "background_preservation",
)
MARGIN_THRESHOLDS = (0.0,) * len(MARGIN_NAMES)

SOURCE_CAPTION_ORIGINS = (
    "sealed_dataset_source_caption_v1",
    "source_video_frozen_captioner_v1",
)
ONLINE_SOURCE_CAPTION_ORIGIN = "source_video_frozen_captioner_v1"
OFFLINE_SOURCE_CAPTION_ORIGIN = "sealed_dataset_source_caption_v1"

PROGRAM_EXTRACTOR_ALLOWED_INPUTS = (
    "source_retelling_binding",
    "edit_instruction_binding",
    "detached_frozen_teacher_hidden_counterfactuals",
    "counterfactual_execution_receipt_observations",
)
SCORER_ALLOWED_INPUTS = (
    "source_video_read_only",
    "source_retelling_binding",
    "target_retelling_binding",
    "edit_instruction_binding",
    "detached_action_phase_programs",
    "detached_frozen_teacher_hidden_counterfactuals",
    "fixed_scorer_configuration",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BANK_TOKEN = object()
_EVIDENCE_TOKEN = object()
_SELECTION_TOKEN = object()


class GraftASGAError(RuntimeError):
    """An ASGA shape, provenance, gate, or authority contract was violated."""


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
        raise GraftASGAError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GraftASGAError(f"{label} must be lowercase SHA-256")
    return value


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise GraftASGAError(f"{label} must be non-empty text without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise GraftASGAError(f"{label} is not valid UTF-8 text") from error
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, expected: Sequence[str], label: str
) -> None:
    actual = set(value.keys())
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise GraftASGAError(
            f"{label} keys differ; missing={missing!r}, extra={extra!r}"
        )


def _owned_canonical_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise GraftASGAError(f"{label} must be a plain canonical mapping")
    owned = json.loads(canonical_json_bytes(value).decode("ascii"))
    if type(owned) is not dict:
        raise GraftASGAError(f"{label} must be a JSON object")
    return owned


def _validate_self_digest(receipt: Mapping[str, Any], *, label: str) -> None:
    payload = dict(receipt)
    digest = payload.pop("receipt_digest", None)
    _require_sha256(digest, label=f"{label} receipt digest")
    if digest != object_sha256(payload):
        raise GraftASGAError(f"{label} receipt digest differs")


def _owned_fp32(value: Any, *, label: str, ndim: int) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or value.ndim != ndim
        or value.numel() <= 0
        or value.device.type == "meta"
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise GraftASGAError(
            f"{label} must be detached contiguous finite FP32 rank-{ndim}"
        )
    return value.detach().to(device="cpu").contiguous().clone()


def tensor_sha256(value: torch.Tensor, *, label: str) -> str:
    owned = _owned_fp32(value, label=label, ndim=value.ndim)
    raw = bytes(owned.view(torch.uint8).reshape(-1).tolist())
    header = canonical_json_bytes(
        {"dtype": str(owned.dtype), "shape": list(map(int, owned.shape))}
    )
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_receipt_json(receipt: Mapping[str, Any]) -> str:
    return canonical_json_bytes(receipt).decode("ascii")


def _receipt_from_json(value: str, *, label: str) -> Mapping[str, Any]:
    _require_text(value, label=label)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as error:
        raise GraftASGAError(f"{label} is not canonical JSON") from error
    owned = _owned_canonical_mapping(parsed, label=label)
    if _canonical_receipt_json(owned) != value:
        raise GraftASGAError(f"{label} bytes are not canonical")
    return owned


_CAPTIONER_RECEIPT_KEYS = (
    "schema_version",
    "source_caption_origin",
    "source_video_sha256",
    "source_caption_sha256",
    "captioner_model_digest",
    "captioner_code_digest",
    "source_video_read",
    "edit_instruction_read",
    "target_video_read",
    "proposal_media_read",
    "mask_pose_flow_track_read",
    "offline_only",
    "online_inference_available",
    "semantic_correctness_authority",
    "same_process_security_boundary",
    "receipt_digest",
)


def _validate_captioner_receipt(
    receipt_value: Any,
    *,
    source_video_sha256: str,
    source_caption_sha256: str,
    source_caption_origin: str,
) -> Tuple[str, str]:
    receipt = _owned_canonical_mapping(receipt_value, label="captioner receipt")
    _require_exact_keys(
        receipt, expected=_CAPTIONER_RECEIPT_KEYS, label="captioner receipt"
    )
    _validate_self_digest(receipt, label="captioner")
    offline_only = source_caption_origin == OFFLINE_SOURCE_CAPTION_ORIGIN
    expected = {
        "schema_version": CAPTIONER_RECEIPT_SCHEMA_VERSION,
        "source_caption_origin": source_caption_origin,
        "source_video_sha256": source_video_sha256,
        "source_caption_sha256": source_caption_sha256,
        "source_video_read": True,
        "edit_instruction_read": False,
        "target_video_read": False,
        "proposal_media_read": False,
        "mask_pose_flow_track_read": False,
        "offline_only": offline_only,
        "online_inference_available": not offline_only,
        "semantic_correctness_authority": False,
        "same_process_security_boundary": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value or type(receipt.get(key)) is not type(value):
            raise GraftASGAError(f"captioner receipt field {key!r} differs")
    _require_sha256(receipt["captioner_model_digest"], label="captioner model digest")
    _require_sha256(receipt["captioner_code_digest"], label="captioner code digest")
    receipt_json = _canonical_receipt_json(receipt)
    return receipt_json, hashlib.sha256(receipt_json.encode("ascii")).hexdigest()


_COMPILER_RECEIPT_KEYS = (
    "schema_version",
    "source_caption_sha256",
    "instruction_sha256",
    "target_caption_sha256",
    "compiler_model_digest",
    "compiler_code_digest",
    "compiler_inputs",
    "raw_instruction_retained",
    "source_video_read",
    "target_video_read",
    "proposal_media_read",
    "mask_pose_flow_track_read",
    "online_inference_available",
    "non_action_field_preservation_semantically_verified",
    "semantic_correctness_authority",
    "same_process_security_boundary",
    "receipt_digest",
)


def _validate_compiler_receipt(
    receipt_value: Any,
    *,
    source_caption_sha256: str,
    instruction_sha256: str,
    target_caption_sha256: str,
) -> Tuple[str, str]:
    receipt = _owned_canonical_mapping(receipt_value, label="compiler receipt")
    _require_exact_keys(
        receipt, expected=_COMPILER_RECEIPT_KEYS, label="compiler receipt"
    )
    _validate_self_digest(receipt, label="compiler")
    expected = {
        "schema_version": COMPILER_RECEIPT_SCHEMA_VERSION,
        "source_caption_sha256": source_caption_sha256,
        "instruction_sha256": instruction_sha256,
        "target_caption_sha256": target_caption_sha256,
        "compiler_inputs": "source_retelling_and_instruction_only",
        "raw_instruction_retained": True,
        "source_video_read": False,
        "target_video_read": False,
        "proposal_media_read": False,
        "mask_pose_flow_track_read": False,
        "online_inference_available": True,
        "non_action_field_preservation_semantically_verified": False,
        "semantic_correctness_authority": False,
        "same_process_security_boundary": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value or type(receipt.get(key)) is not type(value):
            raise GraftASGAError(f"compiler receipt field {key!r} differs")
    _require_sha256(receipt["compiler_model_digest"], label="compiler model digest")
    _require_sha256(receipt["compiler_code_digest"], label="compiler code digest")
    receipt_json = _canonical_receipt_json(receipt)
    return receipt_json, hashlib.sha256(receipt_json.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SourceRetellingBinding:
    schema_version: str
    source_video_sha256: str
    source_caption: str
    edit_instruction: str
    target_caption: str
    source_caption_origin: str
    offline_only: bool
    online_inference_closed: bool
    source_caption_sha256: str
    instruction_sha256: str
    target_caption_sha256: str
    captioner_receipt_json: str
    captioner_receipt_sha256: str
    target_compiler_receipt_json: str
    target_compiler_receipt_sha256: str
    target_compiler_inputs: str
    target_video_read: bool
    proposal_media_read: bool
    mask_pose_flow_track_read: bool
    non_action_field_preservation_semantically_verified: bool
    semantic_correctness_authority: bool
    same_process_security_boundary: bool
    digest: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_video_sha256": self.source_video_sha256,
            "source_caption": self.source_caption,
            "edit_instruction": self.edit_instruction,
            "target_caption": self.target_caption,
            "source_caption_origin": self.source_caption_origin,
            "offline_only": self.offline_only,
            "online_inference_closed": self.online_inference_closed,
            "source_caption_sha256": self.source_caption_sha256,
            "instruction_sha256": self.instruction_sha256,
            "target_caption_sha256": self.target_caption_sha256,
            "captioner_receipt_json": self.captioner_receipt_json,
            "captioner_receipt_sha256": self.captioner_receipt_sha256,
            "target_compiler_receipt_json": self.target_compiler_receipt_json,
            "target_compiler_receipt_sha256": self.target_compiler_receipt_sha256,
            "target_compiler_inputs": self.target_compiler_inputs,
            "target_video_read": self.target_video_read,
            "proposal_media_read": self.proposal_media_read,
            "mask_pose_flow_track_read": self.mask_pose_flow_track_read,
            "non_action_field_preservation_semantically_verified": (
                self.non_action_field_preservation_semantically_verified
            ),
            "semantic_correctness_authority": self.semantic_correctness_authority,
            "same_process_security_boundary": self.same_process_security_boundary,
        }

    def validate(self) -> None:
        if self.schema_version != RETELLING_SCHEMA_VERSION:
            raise GraftASGAError("source-retelling schema differs")
        _require_sha256(self.source_video_sha256, label="source video SHA")
        for text, digest, label in (
            (self.source_caption, self.source_caption_sha256, "source caption"),
            (self.edit_instruction, self.instruction_sha256, "edit instruction"),
            (self.target_caption, self.target_caption_sha256, "target caption"),
        ):
            _require_text(text, label=label)
            _require_sha256(digest, label=f"{label} SHA")
            if _text_sha256(text) != digest:
                raise GraftASGAError(f"{label} SHA differs")
        if self.source_caption_origin not in SOURCE_CAPTION_ORIGINS:
            raise GraftASGAError("source caption origin differs")
        offline_expected = self.source_caption_origin == OFFLINE_SOURCE_CAPTION_ORIGIN
        if type(self.offline_only) is not bool or self.offline_only is not offline_expected:
            raise GraftASGAError("source-retelling offline-only state differs")
        if (
            type(self.online_inference_closed) is not bool
            or self.online_inference_closed is not (not offline_expected)
        ):
            raise GraftASGAError("source-retelling inference closure differs")
        captioner_receipt = _receipt_from_json(
            self.captioner_receipt_json, label="captioner receipt JSON"
        )
        captioner_json, captioner_sha = _validate_captioner_receipt(
            captioner_receipt,
            source_video_sha256=self.source_video_sha256,
            source_caption_sha256=self.source_caption_sha256,
            source_caption_origin=self.source_caption_origin,
        )
        if (
            captioner_json != self.captioner_receipt_json
            or captioner_sha != self.captioner_receipt_sha256
        ):
            raise GraftASGAError("captioner receipt binding differs")
        compiler_receipt = _receipt_from_json(
            self.target_compiler_receipt_json,
            label="target compiler receipt JSON",
        )
        compiler_json, compiler_sha = _validate_compiler_receipt(
            compiler_receipt,
            source_caption_sha256=self.source_caption_sha256,
            instruction_sha256=self.instruction_sha256,
            target_caption_sha256=self.target_caption_sha256,
        )
        if (
            compiler_json != self.target_compiler_receipt_json
            or compiler_sha != self.target_compiler_receipt_sha256
        ):
            raise GraftASGAError("target compiler receipt binding differs")
        if self.target_compiler_inputs != "source_retelling_and_instruction_only":
            raise GraftASGAError("target compiler inputs are not source-only")
        false_fields = (
            self.target_video_read,
            self.proposal_media_read,
            self.mask_pose_flow_track_read,
            self.non_action_field_preservation_semantically_verified,
            self.semantic_correctness_authority,
            self.same_process_security_boundary,
        )
        if any(value is not False for value in false_fields):
            raise GraftASGAError("source-retelling authority is overstated")
        if self.digest != object_sha256(self.payload()):
            raise GraftASGAError("source-retelling digest differs")


def bind_source_retelling(
    *,
    source_video_sha256: str,
    source_caption: str,
    edit_instruction: str,
    target_caption: str,
    source_caption_origin: str,
    captioner_receipt: Mapping[str, Any],
    target_compiler_receipt: Mapping[str, Any],
) -> SourceRetellingBinding:
    source_video_sha = _require_sha256(source_video_sha256, label="source video SHA")
    source_text = _require_text(source_caption, label="source caption")
    instruction_text = _require_text(edit_instruction, label="edit instruction")
    target_text = _require_text(target_caption, label="target caption")
    if source_caption_origin not in SOURCE_CAPTION_ORIGINS:
        raise GraftASGAError("source caption origin differs")
    source_sha = _text_sha256(source_text)
    instruction_sha = _text_sha256(instruction_text)
    target_sha = _text_sha256(target_text)
    captioner_json, captioner_sha = _validate_captioner_receipt(
        captioner_receipt,
        source_video_sha256=source_video_sha,
        source_caption_sha256=source_sha,
        source_caption_origin=source_caption_origin,
    )
    compiler_json, compiler_sha = _validate_compiler_receipt(
        target_compiler_receipt,
        source_caption_sha256=source_sha,
        instruction_sha256=instruction_sha,
        target_caption_sha256=target_sha,
    )
    offline_only = source_caption_origin == OFFLINE_SOURCE_CAPTION_ORIGIN
    payload = {
        "schema_version": RETELLING_SCHEMA_VERSION,
        "source_video_sha256": source_video_sha,
        "source_caption": source_text,
        "edit_instruction": instruction_text,
        "target_caption": target_text,
        "source_caption_origin": source_caption_origin,
        "offline_only": offline_only,
        "online_inference_closed": not offline_only,
        "source_caption_sha256": source_sha,
        "instruction_sha256": instruction_sha,
        "target_caption_sha256": target_sha,
        "captioner_receipt_json": captioner_json,
        "captioner_receipt_sha256": captioner_sha,
        "target_compiler_receipt_json": compiler_json,
        "target_compiler_receipt_sha256": compiler_sha,
        "target_compiler_inputs": "source_retelling_and_instruction_only",
        "target_video_read": False,
        "proposal_media_read": False,
        "mask_pose_flow_track_read": False,
        "non_action_field_preservation_semantically_verified": False,
        "semantic_correctness_authority": False,
        "same_process_security_boundary": False,
    }
    binding = SourceRetellingBinding(**payload, digest=object_sha256(payload))
    binding.validate()
    return binding


def _seed_matrix(value: Any) -> Tuple[Tuple[int, ...], ...]:
    try:
        rows = tuple(tuple(row) for row in value)
    except TypeError as error:
        raise GraftASGAError("branch seed matrix must be nested sequences") from error
    if len(rows) != CANDIDATE_COUNT or any(len(row) != BRANCH_COUNT for row in rows):
        raise GraftASGAError("branch seed matrix must have fixed [5,4] shape")
    for row in rows:
        for item in row:
            if type(item) is not int or not 0 <= item <= MAX_PORTABLE_SEED:
                raise GraftASGAError("branch seeds must be portable nonnegative ints")
        if len(set(row)) != 1:
            raise GraftASGAError("counterfactual branches do not share one seed")
    candidate_seeds = tuple(row[0] for row in rows)
    if len(set(candidate_seeds)) != CANDIDATE_COUNT:
        raise GraftASGAError("candidate group seeds must be unique")
    return rows


def _sha_matrix(value: Any, *, label: str) -> Tuple[Tuple[str, ...], ...]:
    try:
        rows = tuple(tuple(row) for row in value)
    except TypeError as error:
        raise GraftASGAError(f"{label} must be nested sequences") from error
    if len(rows) != CANDIDATE_COUNT or any(len(row) != BRANCH_COUNT for row in rows):
        raise GraftASGAError(f"{label} must have fixed [5,4] shape")
    for row in rows:
        for item in row:
            _require_sha256(item, label=label)
    return rows


def _candidate_slice_sha256s(programs: torch.Tensor) -> Tuple[str, ...]:
    return tuple(
        tensor_sha256(programs[index].contiguous(), label=f"candidate {index} program")
        for index in range(CANDIDATE_COUNT)
    )


_EXTRACTOR_RECEIPT_KEYS = (
    "schema_version",
    "program_coordinate",
    "phase_order",
    "counterfactual_branch_order",
    "output_shape",
    "output_tensor_sha256",
    "output_candidate_slice_sha256s",
    "input_branch_execution_receipt_sha256s",
    "input_retelling_digest",
    "extractor_model_digest",
    "extractor_code_digest",
    "allowed_inputs",
    "action_program_only_observation",
    "proposal_rgb_read",
    "proposal_latent_read",
    "raw_velocity_read",
    "target_video_read",
    "mask_read",
    "pose_read",
    "flow_read",
    "track_read",
    "semantic_correctness_authority",
    "same_process_security_boundary",
    "receipt_digest",
)


def _validate_extractor_receipt(
    receipt_value: Any,
    *,
    programs: torch.Tensor,
    branch_execution_receipt_sha256s: Tuple[Tuple[str, ...], ...],
    retelling_digest: str,
) -> Tuple[str, str]:
    receipt = _owned_canonical_mapping(
        receipt_value, label="action-program extractor receipt"
    )
    _require_exact_keys(
        receipt,
        expected=_EXTRACTOR_RECEIPT_KEYS,
        label="action-program extractor receipt",
    )
    _validate_self_digest(receipt, label="action-program extractor")
    expected = {
        "schema_version": EXTRACTOR_RECEIPT_SCHEMA_VERSION,
        "program_coordinate": PROGRAM_COORDINATE,
        "phase_order": list(PHASE_ORDER),
        "counterfactual_branch_order": list(COUNTERFACTUAL_BRANCH_ORDER),
        "output_shape": [CANDIDATE_COUNT, PHASE_COUNT, PROGRAM_WIDTH],
        "output_tensor_sha256": tensor_sha256(programs, label="proposal programs"),
        "output_candidate_slice_sha256s": list(
            _candidate_slice_sha256s(programs)
        ),
        "input_branch_execution_receipt_sha256s": [
            list(row) for row in branch_execution_receipt_sha256s
        ],
        "input_retelling_digest": retelling_digest,
        "allowed_inputs": list(PROGRAM_EXTRACTOR_ALLOWED_INPUTS),
        "action_program_only_observation": True,
        "proposal_rgb_read": False,
        "proposal_latent_read": False,
        "raw_velocity_read": False,
        "target_video_read": False,
        "mask_read": False,
        "pose_read": False,
        "flow_read": False,
        "track_read": False,
        "semantic_correctness_authority": False,
        "same_process_security_boundary": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value or type(receipt.get(key)) is not type(value):
            raise GraftASGAError(f"extractor receipt field {key!r} differs")
    _require_sha256(receipt["extractor_model_digest"], label="extractor model digest")
    _require_sha256(receipt["extractor_code_digest"], label="extractor code digest")
    receipt_json = _canonical_receipt_json(receipt)
    return receipt_json, hashlib.sha256(receipt_json.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ProposalBankProvenance:
    schema_version: str
    shape: Tuple[int, int, int]
    program_coordinate: str
    phase_order: Tuple[str, ...]
    counterfactual_branch_order: Tuple[str, ...]
    tensor_sha256: str
    candidate_slice_sha256s: Tuple[str, ...]
    branch_seed_matrix: Tuple[Tuple[int, ...], ...]
    branch_gaussian_raw_sha256s: Tuple[Tuple[str, ...], ...]
    branch_schedule_digests: Tuple[Tuple[str, ...], ...]
    branch_prompt_digests: Tuple[Tuple[str, ...], ...]
    branch_execution_receipt_sha256s: Tuple[Tuple[str, ...], ...]
    shared_non_action_prompt_digest: str
    retelling_digest: str
    frozen_teacher_receipt_sha256: str
    checkpoint_digest: str
    extractor_receipt_json: str
    extractor_receipt_sha256: str
    same_seed_within_each_counterfactual_group: bool
    same_gaussian_within_each_counterfactual_group: bool
    schedule_locked_across_bank: bool
    branch_prompts_locked_across_candidates: bool
    candidate_seeds_unique: bool
    candidate_gaussians_unique: bool
    action_program_only_observation: bool
    proposal_rgb_or_latent_included: bool
    raw_velocity_included: bool
    target_video_used: bool
    semantic_correctness_authority: bool
    same_process_security_boundary: bool
    digest: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shape": list(self.shape),
            "program_coordinate": self.program_coordinate,
            "phase_order": list(self.phase_order),
            "counterfactual_branch_order": list(self.counterfactual_branch_order),
            "tensor_sha256": self.tensor_sha256,
            "candidate_slice_sha256s": list(self.candidate_slice_sha256s),
            "branch_seed_matrix": [list(row) for row in self.branch_seed_matrix],
            "branch_gaussian_raw_sha256s": [
                list(row) for row in self.branch_gaussian_raw_sha256s
            ],
            "branch_schedule_digests": [
                list(row) for row in self.branch_schedule_digests
            ],
            "branch_prompt_digests": [
                list(row) for row in self.branch_prompt_digests
            ],
            "branch_execution_receipt_sha256s": [
                list(row) for row in self.branch_execution_receipt_sha256s
            ],
            "shared_non_action_prompt_digest": self.shared_non_action_prompt_digest,
            "retelling_digest": self.retelling_digest,
            "frozen_teacher_receipt_sha256": self.frozen_teacher_receipt_sha256,
            "checkpoint_digest": self.checkpoint_digest,
            "extractor_receipt_json": self.extractor_receipt_json,
            "extractor_receipt_sha256": self.extractor_receipt_sha256,
            "same_seed_within_each_counterfactual_group": (
                self.same_seed_within_each_counterfactual_group
            ),
            "same_gaussian_within_each_counterfactual_group": (
                self.same_gaussian_within_each_counterfactual_group
            ),
            "schedule_locked_across_bank": self.schedule_locked_across_bank,
            "branch_prompts_locked_across_candidates": (
                self.branch_prompts_locked_across_candidates
            ),
            "candidate_seeds_unique": self.candidate_seeds_unique,
            "candidate_gaussians_unique": self.candidate_gaussians_unique,
            "action_program_only_observation": self.action_program_only_observation,
            "proposal_rgb_or_latent_included": self.proposal_rgb_or_latent_included,
            "raw_velocity_included": self.raw_velocity_included,
            "target_video_used": self.target_video_used,
            "semantic_correctness_authority": self.semantic_correctness_authority,
            "same_process_security_boundary": self.same_process_security_boundary,
        }

    def validate(self) -> None:
        if self.schema_version != PROPOSAL_BANK_SCHEMA_VERSION:
            raise GraftASGAError("proposal-bank schema differs")
        if self.shape != (CANDIDATE_COUNT, PHASE_COUNT, PROGRAM_WIDTH):
            raise GraftASGAError("proposal-bank shape differs")
        if self.program_coordinate != PROGRAM_COORDINATE:
            raise GraftASGAError("proposal-bank coordinate differs")
        if self.phase_order != PHASE_ORDER:
            raise GraftASGAError("proposal-bank phase order differs")
        if self.counterfactual_branch_order != COUNTERFACTUAL_BRANCH_ORDER:
            raise GraftASGAError("proposal-bank counterfactual order differs")
        _require_sha256(self.tensor_sha256, label="proposal tensor SHA")
        if len(self.candidate_slice_sha256s) != CANDIDATE_COUNT:
            raise GraftASGAError("candidate slice SHA count differs")
        for value in self.candidate_slice_sha256s:
            _require_sha256(value, label="candidate slice SHA")
        seeds = _seed_matrix(self.branch_seed_matrix)
        gaussians = _sha_matrix(
            self.branch_gaussian_raw_sha256s,
            label="branch Gaussian raw SHA",
        )
        schedules = _sha_matrix(
            self.branch_schedule_digests, label="branch schedule digest"
        )
        prompts = _sha_matrix(
            self.branch_prompt_digests, label="branch prompt digest"
        )
        receipts = _sha_matrix(
            self.branch_execution_receipt_sha256s,
            label="branch execution receipt SHA",
        )
        if any(len(set(row)) != 1 for row in gaussians):
            raise GraftASGAError("counterfactual branches do not share Gaussian bytes")
        if len({row[0] for row in gaussians}) != CANDIDATE_COUNT:
            raise GraftASGAError("candidate Gaussian bytes are not unique")
        if len({item for row in schedules for item in row}) != 1:
            raise GraftASGAError("scheduler is not locked across proposal bank")
        if any(row != prompts[0] for row in prompts[1:]):
            raise GraftASGAError("branch prompts change across candidate seeds")
        if len(set(prompts[0])) != BRANCH_COUNT:
            raise GraftASGAError("counterfactual branch prompts are not distinct")
        if len({item for row in receipts for item in row}) != (
            CANDIDATE_COUNT * BRANCH_COUNT
        ):
            raise GraftASGAError("branch execution receipt SHAs are not unique")
        for value, label in (
            (self.shared_non_action_prompt_digest, "shared non-action prompt digest"),
            (self.retelling_digest, "retelling digest"),
            (self.frozen_teacher_receipt_sha256, "teacher receipt SHA"),
            (self.checkpoint_digest, "checkpoint digest"),
            (self.extractor_receipt_sha256, "extractor receipt SHA"),
        ):
            _require_sha256(value, label=label)
        extractor = _receipt_from_json(
            self.extractor_receipt_json, label="extractor receipt JSON"
        )
        extractor_json, extractor_sha = _validate_extractor_receipt_payload_only(
            extractor,
            tensor_sha=self.tensor_sha256,
            slice_shas=self.candidate_slice_sha256s,
            shape=self.shape,
            branch_receipts=receipts,
            retelling_digest=self.retelling_digest,
        )
        if (
            extractor_json != self.extractor_receipt_json
            or extractor_sha != self.extractor_receipt_sha256
        ):
            raise GraftASGAError("extractor receipt binding differs")
        true_fields = (
            self.same_seed_within_each_counterfactual_group,
            self.same_gaussian_within_each_counterfactual_group,
            self.schedule_locked_across_bank,
            self.branch_prompts_locked_across_candidates,
            self.candidate_seeds_unique,
            self.candidate_gaussians_unique,
            self.action_program_only_observation,
        )
        false_fields = (
            self.proposal_rgb_or_latent_included,
            self.raw_velocity_included,
            self.target_video_used,
            self.semantic_correctness_authority,
            self.same_process_security_boundary,
        )
        if any(value is not True for value in true_fields) or any(
            value is not False for value in false_fields
        ):
            raise GraftASGAError("proposal-bank observed authority differs")
        if seeds != self.branch_seed_matrix:
            raise GraftASGAError("proposal-bank seed representation differs")
        if self.digest != object_sha256(self.payload()):
            raise GraftASGAError("proposal-bank provenance digest differs")


def _validate_extractor_receipt_payload_only(
    receipt_value: Any,
    *,
    tensor_sha: str,
    slice_shas: Tuple[str, ...],
    shape: Tuple[int, int, int],
    branch_receipts: Tuple[Tuple[str, ...], ...],
    retelling_digest: str,
) -> Tuple[str, str]:
    receipt = _owned_canonical_mapping(
        receipt_value, label="action-program extractor receipt"
    )
    _require_exact_keys(
        receipt,
        expected=_EXTRACTOR_RECEIPT_KEYS,
        label="action-program extractor receipt",
    )
    _validate_self_digest(receipt, label="action-program extractor")
    expected = {
        "schema_version": EXTRACTOR_RECEIPT_SCHEMA_VERSION,
        "program_coordinate": PROGRAM_COORDINATE,
        "phase_order": list(PHASE_ORDER),
        "counterfactual_branch_order": list(COUNTERFACTUAL_BRANCH_ORDER),
        "output_shape": list(shape),
        "output_tensor_sha256": tensor_sha,
        "output_candidate_slice_sha256s": list(slice_shas),
        "input_branch_execution_receipt_sha256s": [
            list(row) for row in branch_receipts
        ],
        "input_retelling_digest": retelling_digest,
        "allowed_inputs": list(PROGRAM_EXTRACTOR_ALLOWED_INPUTS),
        "action_program_only_observation": True,
        "proposal_rgb_read": False,
        "proposal_latent_read": False,
        "raw_velocity_read": False,
        "target_video_read": False,
        "mask_read": False,
        "pose_read": False,
        "flow_read": False,
        "track_read": False,
        "semantic_correctness_authority": False,
        "same_process_security_boundary": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value or type(receipt.get(key)) is not type(value):
            raise GraftASGAError(f"extractor receipt field {key!r} differs")
    _require_sha256(receipt["extractor_model_digest"], label="extractor model digest")
    _require_sha256(receipt["extractor_code_digest"], label="extractor code digest")
    receipt_json = _canonical_receipt_json(receipt)
    return receipt_json, hashlib.sha256(receipt_json.encode("ascii")).hexdigest()


class AuthenticatedProposalBank:
    __slots__ = ("tensor", "provenance", "retelling", "_token")

    def __init__(
        self,
        tensor: torch.Tensor,
        provenance: ProposalBankProvenance,
        retelling: SourceRetellingBinding,
        *,
        _token: object,
    ) -> None:
        if _token is not _BANK_TOKEN:
            raise GraftASGAError("proposal bank must be factory-created")
        self.tensor = tensor
        self.provenance = provenance
        self.retelling = retelling
        self._token = _token
        self.validate()

    def validate(self) -> None:
        if self._token is not _BANK_TOKEN:
            raise GraftASGAError("proposal-bank token differs")
        self.retelling.validate()
        self.provenance.validate()
        if self.provenance.retelling_digest != self.retelling.digest:
            raise GraftASGAError("proposal bank is bound to another retelling")
        owned = _owned_fp32(self.tensor, label="proposal programs", ndim=3)
        if tuple(owned.shape) != self.provenance.shape:
            raise GraftASGAError("live proposal shape differs")
        if tensor_sha256(owned, label="proposal programs") != self.provenance.tensor_sha256:
            raise GraftASGAError("live proposal tensor bytes differ")
        if _candidate_slice_sha256s(owned) != self.provenance.candidate_slice_sha256s:
            raise GraftASGAError("live candidate slice bytes differ")


def authenticate_proposal_bank(
    programs: torch.Tensor,
    *,
    branch_seed_matrix: Sequence[Sequence[int]],
    branch_gaussian_raw_sha256s: Sequence[Sequence[str]],
    branch_schedule_digests: Sequence[Sequence[str]],
    branch_prompt_digests: Sequence[Sequence[str]],
    branch_execution_receipt_sha256s: Sequence[Sequence[str]],
    shared_non_action_prompt_digest: str,
    retelling: SourceRetellingBinding,
    frozen_teacher_receipt_sha256: str,
    checkpoint_digest: str,
    extractor_receipt: Mapping[str, Any],
) -> AuthenticatedProposalBank:
    retelling.validate()
    owned = _owned_fp32(programs, label="proposal programs", ndim=3)
    if tuple(owned.shape) != (CANDIDATE_COUNT, PHASE_COUNT, PROGRAM_WIDTH):
        raise GraftASGAError("proposal programs must have fixed [K=5,P=4,D=32] shape")
    seeds = _seed_matrix(branch_seed_matrix)
    gaussians = _sha_matrix(
        branch_gaussian_raw_sha256s, label="branch Gaussian raw SHA"
    )
    schedules = _sha_matrix(
        branch_schedule_digests, label="branch schedule digest"
    )
    prompts = _sha_matrix(branch_prompt_digests, label="branch prompt digest")
    execution_receipts = _sha_matrix(
        branch_execution_receipt_sha256s,
        label="branch execution receipt SHA",
    )
    if any(len(set(row)) != 1 for row in gaussians):
        raise GraftASGAError("counterfactual branches must share exact Gaussian bytes")
    if len({row[0] for row in gaussians}) != CANDIDATE_COUNT:
        raise GraftASGAError("different candidates must use different Gaussian bytes")
    if len({item for row in schedules for item in row}) != 1:
        raise GraftASGAError("all branches must share one scheduler contract")
    if any(row != prompts[0] for row in prompts[1:]):
        raise GraftASGAError("branch prompts must remain fixed across candidate seeds")
    if len(set(prompts[0])) != BRANCH_COUNT:
        raise GraftASGAError("four counterfactual branch prompts must be distinct")
    if len({item for row in execution_receipts for item in row}) != (
        CANDIDATE_COUNT * BRANCH_COUNT
    ):
        raise GraftASGAError("branch execution receipt SHAs must be unique")
    shared_prompt_sha = _require_sha256(
        shared_non_action_prompt_digest,
        label="shared non-action prompt digest",
    )
    teacher_sha = _require_sha256(
        frozen_teacher_receipt_sha256, label="teacher receipt SHA"
    )
    checkpoint_sha = _require_sha256(checkpoint_digest, label="checkpoint digest")
    tensor_sha = tensor_sha256(owned, label="proposal programs")
    slice_shas = _candidate_slice_sha256s(owned)
    extractor_json, extractor_sha = _validate_extractor_receipt(
        extractor_receipt,
        programs=owned,
        branch_execution_receipt_sha256s=execution_receipts,
        retelling_digest=retelling.digest,
    )
    payload = {
        "schema_version": PROPOSAL_BANK_SCHEMA_VERSION,
        "shape": [CANDIDATE_COUNT, PHASE_COUNT, PROGRAM_WIDTH],
        "program_coordinate": PROGRAM_COORDINATE,
        "phase_order": list(PHASE_ORDER),
        "counterfactual_branch_order": list(COUNTERFACTUAL_BRANCH_ORDER),
        "tensor_sha256": tensor_sha,
        "candidate_slice_sha256s": list(slice_shas),
        "branch_seed_matrix": [list(row) for row in seeds],
        "branch_gaussian_raw_sha256s": [list(row) for row in gaussians],
        "branch_schedule_digests": [list(row) for row in schedules],
        "branch_prompt_digests": [list(row) for row in prompts],
        "branch_execution_receipt_sha256s": [
            list(row) for row in execution_receipts
        ],
        "shared_non_action_prompt_digest": shared_prompt_sha,
        "retelling_digest": retelling.digest,
        "frozen_teacher_receipt_sha256": teacher_sha,
        "checkpoint_digest": checkpoint_sha,
        "extractor_receipt_json": extractor_json,
        "extractor_receipt_sha256": extractor_sha,
        "same_seed_within_each_counterfactual_group": True,
        "same_gaussian_within_each_counterfactual_group": True,
        "schedule_locked_across_bank": True,
        "branch_prompts_locked_across_candidates": True,
        "candidate_seeds_unique": True,
        "candidate_gaussians_unique": True,
        "action_program_only_observation": True,
        "proposal_rgb_or_latent_included": False,
        "raw_velocity_included": False,
        "target_video_used": False,
        "semantic_correctness_authority": False,
        "same_process_security_boundary": False,
    }
    provenance = ProposalBankProvenance(
        schema_version=payload["schema_version"],
        shape=tuple(payload["shape"]),
        program_coordinate=payload["program_coordinate"],
        phase_order=tuple(payload["phase_order"]),
        counterfactual_branch_order=tuple(payload["counterfactual_branch_order"]),
        tensor_sha256=payload["tensor_sha256"],
        candidate_slice_sha256s=tuple(payload["candidate_slice_sha256s"]),
        branch_seed_matrix=tuple(tuple(row) for row in payload["branch_seed_matrix"]),
        branch_gaussian_raw_sha256s=tuple(
            tuple(row) for row in payload["branch_gaussian_raw_sha256s"]
        ),
        branch_schedule_digests=tuple(
            tuple(row) for row in payload["branch_schedule_digests"]
        ),
        branch_prompt_digests=tuple(
            tuple(row) for row in payload["branch_prompt_digests"]
        ),
        branch_execution_receipt_sha256s=tuple(
            tuple(row) for row in payload["branch_execution_receipt_sha256s"]
        ),
        shared_non_action_prompt_digest=payload["shared_non_action_prompt_digest"],
        retelling_digest=payload["retelling_digest"],
        frozen_teacher_receipt_sha256=payload["frozen_teacher_receipt_sha256"],
        checkpoint_digest=payload["checkpoint_digest"],
        extractor_receipt_json=payload["extractor_receipt_json"],
        extractor_receipt_sha256=payload["extractor_receipt_sha256"],
        same_seed_within_each_counterfactual_group=True,
        same_gaussian_within_each_counterfactual_group=True,
        schedule_locked_across_bank=True,
        branch_prompts_locked_across_candidates=True,
        candidate_seeds_unique=True,
        candidate_gaussians_unique=True,
        action_program_only_observation=True,
        proposal_rgb_or_latent_included=False,
        raw_velocity_included=False,
        target_video_used=False,
        semantic_correctness_authority=False,
        same_process_security_boundary=False,
        digest=object_sha256(payload),
    )
    return AuthenticatedProposalBank(
        owned, provenance, retelling, _token=_BANK_TOKEN
    )


def _compute_margins(raw_scores: torch.Tensor) -> torch.Tensor:
    owned = _owned_fp32(raw_scores, label="ASGA raw scores", ndim=2)
    if tuple(owned.shape) != (CANDIDATE_COUNT, len(RAW_SCORE_NAMES)):
        raise GraftASGAError("ASGA raw scores must have fixed [5,8] shape")
    if bool(
        torch.any(
            (owned < NORMALIZED_SCORE_MIN) | (owned > NORMALIZED_SCORE_MAX)
        ).item()
    ):
        raise GraftASGAError("ASGA raw scores must be normalized to [-1,1]")
    index = {name: offset for offset, name in enumerate(RAW_SCORE_NAMES)}
    action = owned[:, index["action_event_score"]]
    columns = (
        action - owned[:, index["noop_target_event_score"]],
        action - owned[:, index["reverse_target_event_score"]],
        action - owned[:, index["incomplete_target_event_score"]],
        owned[:, index["terminal_hold_score"]],
        owned[:, index["actor_preservation_delta"]],
        owned[:, index["camera_preservation_delta"]],
        owned[:, index["background_preservation_delta"]],
    )
    margins = torch.stack(columns, dim=1).contiguous()
    thresholds = torch.tensor(MARGIN_THRESHOLDS, dtype=torch.float32)
    return (margins - thresholds.unsqueeze(0)).contiguous()


_SCORER_RECEIPT_KEYS = (
    "schema_version",
    "proposal_bank_digest",
    "retelling_digest",
    "raw_score_names",
    "margin_names",
    "margin_thresholds_float_hex",
    "source_compatibility_kind",
    "normalized_score_range_float_hex",
    "raw_scores_tensor_sha256",
    "source_compatibility_tensor_sha256",
    "scorer_model_digest",
    "scorer_code_digest",
    "scorer_config_digest",
    "allowed_inputs",
    "target_video_read",
    "proposal_rgb_read",
    "proposal_latent_read",
    "raw_velocity_read",
    "mask_read",
    "pose_read",
    "flow_read",
    "track_read",
    "evaluator_selected_index_read",
    "semantic_correctness_authority",
    "same_process_security_boundary",
    "receipt_digest",
)


def _validate_scorer_receipt(
    receipt_value: Any,
    *,
    bank: AuthenticatedProposalBank,
    raw_scores: torch.Tensor,
    source_compatibility: torch.Tensor,
) -> Tuple[str, str]:
    receipt = _owned_canonical_mapping(receipt_value, label="ASGA scorer receipt")
    _require_exact_keys(
        receipt, expected=_SCORER_RECEIPT_KEYS, label="ASGA scorer receipt"
    )
    _validate_self_digest(receipt, label="ASGA scorer")
    expected = {
        "schema_version": SCORER_RECEIPT_SCHEMA_VERSION,
        "proposal_bank_digest": bank.provenance.digest,
        "retelling_digest": bank.retelling.digest,
        "raw_score_names": list(RAW_SCORE_NAMES),
        "margin_names": list(MARGIN_NAMES),
        "margin_thresholds_float_hex": [
            float(value).hex() for value in MARGIN_THRESHOLDS
        ],
        "source_compatibility_kind": SOURCE_COMPATIBILITY_KIND,
        "normalized_score_range_float_hex": [
            float(NORMALIZED_SCORE_MIN).hex(),
            float(NORMALIZED_SCORE_MAX).hex(),
        ],
        "raw_scores_tensor_sha256": tensor_sha256(
            raw_scores, label="ASGA raw scores"
        ),
        "source_compatibility_tensor_sha256": tensor_sha256(
            source_compatibility, label="source compatibility"
        ),
        "allowed_inputs": list(SCORER_ALLOWED_INPUTS),
        "target_video_read": False,
        "proposal_rgb_read": False,
        "proposal_latent_read": False,
        "raw_velocity_read": False,
        "mask_read": False,
        "pose_read": False,
        "flow_read": False,
        "track_read": False,
        "evaluator_selected_index_read": False,
        "semantic_correctness_authority": False,
        "same_process_security_boundary": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value or type(receipt.get(key)) is not type(value):
            raise GraftASGAError(f"ASGA scorer receipt field {key!r} differs")
    for key, label in (
        ("scorer_model_digest", "scorer model digest"),
        ("scorer_code_digest", "scorer code digest"),
        ("scorer_config_digest", "scorer config digest"),
    ):
        _require_sha256(receipt[key], label=label)
    receipt_json = _canonical_receipt_json(receipt)
    return receipt_json, hashlib.sha256(receipt_json.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ASGAEvidenceProvenance:
    schema_version: str
    raw_score_names: Tuple[str, ...]
    margin_names: Tuple[str, ...]
    margin_thresholds: Tuple[float, ...]
    source_compatibility_kind: str
    proposal_bank_digest: str
    raw_scores_tensor_sha256: str
    margins_tensor_sha256: str
    source_compatibility_tensor_sha256: str
    scorer_receipt_json: str
    scorer_receipt_sha256: str
    allowed_inputs: Tuple[str, ...]
    target_video_read: bool
    proposal_rgb_read: bool
    proposal_latent_read: bool
    raw_velocity_read: bool
    mask_read: bool
    pose_read: bool
    flow_read: bool
    track_read: bool
    evaluator_selected_index_read: bool
    semantic_correctness_authority: bool
    same_process_security_boundary: bool
    digest: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "raw_score_names": list(self.raw_score_names),
            "margin_names": list(self.margin_names),
            "margin_thresholds_float_hex": [
                float(value).hex() for value in self.margin_thresholds
            ],
            "source_compatibility_kind": self.source_compatibility_kind,
            "proposal_bank_digest": self.proposal_bank_digest,
            "raw_scores_tensor_sha256": self.raw_scores_tensor_sha256,
            "margins_tensor_sha256": self.margins_tensor_sha256,
            "source_compatibility_tensor_sha256": (
                self.source_compatibility_tensor_sha256
            ),
            "scorer_receipt_json": self.scorer_receipt_json,
            "scorer_receipt_sha256": self.scorer_receipt_sha256,
            "allowed_inputs": list(self.allowed_inputs),
            "target_video_read": self.target_video_read,
            "proposal_rgb_read": self.proposal_rgb_read,
            "proposal_latent_read": self.proposal_latent_read,
            "raw_velocity_read": self.raw_velocity_read,
            "mask_read": self.mask_read,
            "pose_read": self.pose_read,
            "flow_read": self.flow_read,
            "track_read": self.track_read,
            "evaluator_selected_index_read": self.evaluator_selected_index_read,
            "semantic_correctness_authority": self.semantic_correctness_authority,
            "same_process_security_boundary": self.same_process_security_boundary,
        }

    def validate(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise GraftASGAError("ASGA evidence schema differs")
        if self.raw_score_names != RAW_SCORE_NAMES or self.margin_names != MARGIN_NAMES:
            raise GraftASGAError("ASGA evidence score schema differs")
        if self.margin_thresholds != MARGIN_THRESHOLDS:
            raise GraftASGAError("ASGA margin thresholds differ")
        if self.source_compatibility_kind != SOURCE_COMPATIBILITY_KIND:
            raise GraftASGAError("source compatibility kind differs")
        for value, label in (
            (self.proposal_bank_digest, "proposal-bank digest"),
            (self.raw_scores_tensor_sha256, "raw scores SHA"),
            (self.margins_tensor_sha256, "margins SHA"),
            (self.source_compatibility_tensor_sha256, "source compatibility SHA"),
            (self.scorer_receipt_sha256, "scorer receipt SHA"),
        ):
            _require_sha256(value, label=label)
        if self.allowed_inputs != SCORER_ALLOWED_INPUTS:
            raise GraftASGAError("ASGA scorer allowed inputs differ")
        false_fields = (
            self.target_video_read,
            self.proposal_rgb_read,
            self.proposal_latent_read,
            self.raw_velocity_read,
            self.mask_read,
            self.pose_read,
            self.flow_read,
            self.track_read,
            self.evaluator_selected_index_read,
            self.semantic_correctness_authority,
            self.same_process_security_boundary,
        )
        if any(value is not False for value in false_fields):
            raise GraftASGAError("ASGA evidence authority is overstated")
        if self.digest != object_sha256(self.payload()):
            raise GraftASGAError("ASGA evidence provenance digest differs")


class AuthenticatedASGAEvidence:
    __slots__ = (
        "raw_scores",
        "margins",
        "source_compatibility",
        "provenance",
        "_token",
    )

    def __init__(
        self,
        *,
        raw_scores: torch.Tensor,
        margins: torch.Tensor,
        source_compatibility: torch.Tensor,
        provenance: ASGAEvidenceProvenance,
        _token: object,
    ) -> None:
        if _token is not _EVIDENCE_TOKEN:
            raise GraftASGAError("ASGA evidence must be factory-created")
        self.raw_scores = raw_scores
        self.margins = margins
        self.source_compatibility = source_compatibility
        self.provenance = provenance
        self._token = _token
        self.validate()

    @property
    def proposal_bank_digest(self) -> str:
        return self.provenance.proposal_bank_digest

    @property
    def digest(self) -> str:
        return self.provenance.digest

    def validate(self) -> None:
        if self._token is not _EVIDENCE_TOKEN:
            raise GraftASGAError("ASGA evidence token differs")
        raw = _owned_fp32(self.raw_scores, label="ASGA raw scores", ndim=2)
        margins = _owned_fp32(self.margins, label="ASGA margins", ndim=2)
        compatibility = _owned_fp32(
            self.source_compatibility, label="source compatibility", ndim=1
        )
        if tuple(raw.shape) != (CANDIDATE_COUNT, len(RAW_SCORE_NAMES)):
            raise GraftASGAError("ASGA raw score shape differs")
        if tuple(margins.shape) != (CANDIDATE_COUNT, len(MARGIN_NAMES)):
            raise GraftASGAError("ASGA margin shape differs")
        if tuple(compatibility.shape) != (CANDIDATE_COUNT,):
            raise GraftASGAError("source compatibility shape differs")
        if bool(
            torch.any(
                (compatibility < NORMALIZED_SCORE_MIN)
                | (compatibility > NORMALIZED_SCORE_MAX)
            ).item()
        ):
            raise GraftASGAError("source compatibility must be normalized to [-1,1]")
        recomputed_margins = _compute_margins(raw)
        if not torch.equal(margins, recomputed_margins):
            raise GraftASGAError("ASGA margins were not derived from fixed raw scores")
        self.provenance.validate()
        if tensor_sha256(raw, label="ASGA raw scores") != (
            self.provenance.raw_scores_tensor_sha256
        ):
            raise GraftASGAError("live ASGA raw score bytes differ")
        if tensor_sha256(margins, label="ASGA margins") != (
            self.provenance.margins_tensor_sha256
        ):
            raise GraftASGAError("live ASGA margin bytes differ")
        if tensor_sha256(compatibility, label="source compatibility") != (
            self.provenance.source_compatibility_tensor_sha256
        ):
            raise GraftASGAError("live source compatibility bytes differ")
        scorer_receipt = _receipt_from_json(
            self.provenance.scorer_receipt_json, label="ASGA scorer receipt JSON"
        )
        receipt_json = _canonical_receipt_json(scorer_receipt)
        if receipt_json != self.provenance.scorer_receipt_json:
            raise GraftASGAError("ASGA scorer receipt JSON differs")
        if hashlib.sha256(receipt_json.encode("ascii")).hexdigest() != (
            self.provenance.scorer_receipt_sha256
        ):
            raise GraftASGAError("ASGA scorer receipt bytes differ")


def authenticate_asga_evidence(
    bank: AuthenticatedProposalBank,
    *,
    raw_scores: torch.Tensor,
    source_compatibility: torch.Tensor,
    scorer_receipt: Mapping[str, Any],
) -> AuthenticatedASGAEvidence:
    bank.validate()
    owned_raw = _owned_fp32(raw_scores, label="ASGA raw scores", ndim=2)
    margins = _compute_margins(owned_raw)
    owned_compatibility = _owned_fp32(
        source_compatibility, label="source compatibility", ndim=1
    )
    if tuple(owned_compatibility.shape) != (CANDIDATE_COUNT,):
        raise GraftASGAError("source compatibility must have fixed [5] shape")
    if bool(
        torch.any(
            (owned_compatibility < NORMALIZED_SCORE_MIN)
            | (owned_compatibility > NORMALIZED_SCORE_MAX)
        ).item()
    ):
        raise GraftASGAError("source compatibility must be normalized to [-1,1]")
    scorer_json, scorer_sha = _validate_scorer_receipt(
        scorer_receipt,
        bank=bank,
        raw_scores=owned_raw,
        source_compatibility=owned_compatibility,
    )
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "raw_score_names": list(RAW_SCORE_NAMES),
        "margin_names": list(MARGIN_NAMES),
        "margin_thresholds_float_hex": [
            float(value).hex() for value in MARGIN_THRESHOLDS
        ],
        "source_compatibility_kind": SOURCE_COMPATIBILITY_KIND,
        "proposal_bank_digest": bank.provenance.digest,
        "raw_scores_tensor_sha256": tensor_sha256(
            owned_raw, label="ASGA raw scores"
        ),
        "margins_tensor_sha256": tensor_sha256(margins, label="ASGA margins"),
        "source_compatibility_tensor_sha256": tensor_sha256(
            owned_compatibility, label="source compatibility"
        ),
        "scorer_receipt_json": scorer_json,
        "scorer_receipt_sha256": scorer_sha,
        "allowed_inputs": list(SCORER_ALLOWED_INPUTS),
        "target_video_read": False,
        "proposal_rgb_read": False,
        "proposal_latent_read": False,
        "raw_velocity_read": False,
        "mask_read": False,
        "pose_read": False,
        "flow_read": False,
        "track_read": False,
        "evaluator_selected_index_read": False,
        "semantic_correctness_authority": False,
        "same_process_security_boundary": False,
    }
    provenance = ASGAEvidenceProvenance(
        schema_version=payload["schema_version"],
        raw_score_names=tuple(payload["raw_score_names"]),
        margin_names=tuple(payload["margin_names"]),
        margin_thresholds=MARGIN_THRESHOLDS,
        source_compatibility_kind=payload["source_compatibility_kind"],
        proposal_bank_digest=payload["proposal_bank_digest"],
        raw_scores_tensor_sha256=payload["raw_scores_tensor_sha256"],
        margins_tensor_sha256=payload["margins_tensor_sha256"],
        source_compatibility_tensor_sha256=payload[
            "source_compatibility_tensor_sha256"
        ],
        scorer_receipt_json=payload["scorer_receipt_json"],
        scorer_receipt_sha256=payload["scorer_receipt_sha256"],
        allowed_inputs=tuple(payload["allowed_inputs"]),
        target_video_read=False,
        proposal_rgb_read=False,
        proposal_latent_read=False,
        raw_velocity_read=False,
        mask_read=False,
        pose_read=False,
        flow_read=False,
        track_read=False,
        evaluator_selected_index_read=False,
        semantic_correctness_authority=False,
        same_process_security_boundary=False,
        digest=object_sha256(payload),
    )
    return AuthenticatedASGAEvidence(
        raw_scores=owned_raw,
        margins=margins,
        source_compatibility=owned_compatibility,
        provenance=provenance,
        _token=_EVIDENCE_TOKEN,
    )


def _snapshot_bank(bank: AuthenticatedProposalBank) -> AuthenticatedProposalBank:
    bank.validate()
    return AuthenticatedProposalBank(
        bank.tensor.clone(), bank.provenance, bank.retelling, _token=_BANK_TOKEN
    )


def _snapshot_evidence(
    evidence: AuthenticatedASGAEvidence,
) -> AuthenticatedASGAEvidence:
    evidence.validate()
    return AuthenticatedASGAEvidence(
        raw_scores=evidence.raw_scores.clone(),
        margins=evidence.margins.clone(),
        source_compatibility=evidence.source_compatibility.clone(),
        provenance=evidence.provenance,
        _token=_EVIDENCE_TOKEN,
    )


def _derive_selection(
    bank: AuthenticatedProposalBank,
    evidence: AuthenticatedASGAEvidence,
) -> Tuple[Tuple[int, ...], torch.Tensor, Optional[torch.Tensor], bool]:
    bank.validate()
    evidence.validate()
    if evidence.proposal_bank_digest != bank.provenance.digest:
        raise GraftASGAError("ASGA evidence is bound to another proposal bank")
    scorer_receipt = _receipt_from_json(
        evidence.provenance.scorer_receipt_json,
        label="ASGA scorer receipt JSON",
    )
    scorer_json, scorer_sha = _validate_scorer_receipt(
        scorer_receipt,
        bank=bank,
        raw_scores=evidence.raw_scores,
        source_compatibility=evidence.source_compatibility,
    )
    if (
        scorer_json != evidence.provenance.scorer_receipt_json
        or scorer_sha != evidence.provenance.scorer_receipt_sha256
    ):
        raise GraftASGAError("ASGA scorer receipt is not bound to live evidence")
    feasible_mask = torch.all(evidence.margins > 0.0, dim=1)
    feasible_indices = tuple(
        int(index)
        for index in torch.nonzero(feasible_mask, as_tuple=False).reshape(-1).tolist()
    )
    weights = torch.zeros(CANDIDATE_COUNT, dtype=torch.float32)
    if not feasible_indices:
        return feasible_indices, weights, None, True
    logits = evidence.source_compatibility[list(feasible_indices)].double()
    logits = logits / TEMPERATURE
    selected_weights = torch.softmax(logits - logits.max(), dim=0).float()
    weights[list(feasible_indices)] = selected_weights
    program = torch.einsum("k,kpd->pd", weights, bank.tensor).contiguous()
    return feasible_indices, weights, program, False


def _selection_receipt_payload(
    bank: AuthenticatedProposalBank,
    evidence: AuthenticatedASGAEvidence,
    *,
    feasible_indices: Tuple[int, ...],
    weights: torch.Tensor,
    program: Optional[torch.Tensor],
    abstained: bool,
) -> Mapping[str, Any]:
    provenance = bank.provenance
    retelling = bank.retelling
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "method": METHOD,
        "proposal_bank_digest": provenance.digest,
        "evidence_digest": evidence.digest,
        "retelling_digest": retelling.digest,
        "source_video_sha256": retelling.source_video_sha256,
        "source_caption": retelling.source_caption,
        "source_caption_sha256": retelling.source_caption_sha256,
        "edit_instruction": retelling.edit_instruction,
        "instruction_sha256": retelling.instruction_sha256,
        "target_caption": retelling.target_caption,
        "target_caption_sha256": retelling.target_caption_sha256,
        "source_caption_origin": retelling.source_caption_origin,
        "retelling_offline_only": retelling.offline_only,
        "retelling_online_inference_closed": retelling.online_inference_closed,
        "captioner_receipt_sha256": retelling.captioner_receipt_sha256,
        "target_compiler_receipt_sha256": (
            retelling.target_compiler_receipt_sha256
        ),
        "candidate_count_k": CANDIDATE_COUNT,
        "phase_count_p": PHASE_COUNT,
        "phase_order": list(PHASE_ORDER),
        "program_coordinate": PROGRAM_COORDINATE,
        "candidate_slice_sha256s": list(provenance.candidate_slice_sha256s),
        "counterfactual_branch_order": list(COUNTERFACTUAL_BRANCH_ORDER),
        "branch_seed_matrix": [list(row) for row in provenance.branch_seed_matrix],
        "branch_gaussian_raw_sha256s": [
            list(row) for row in provenance.branch_gaussian_raw_sha256s
        ],
        "branch_schedule_digests": [
            list(row) for row in provenance.branch_schedule_digests
        ],
        "branch_prompt_digests": [
            list(row) for row in provenance.branch_prompt_digests
        ],
        "branch_execution_receipt_sha256s": [
            list(row) for row in provenance.branch_execution_receipt_sha256s
        ],
        "shared_non_action_prompt_digest": (
            provenance.shared_non_action_prompt_digest
        ),
        "extractor_receipt_sha256": provenance.extractor_receipt_sha256,
        "raw_score_names": list(RAW_SCORE_NAMES),
        "raw_score_values_float_hex": [
            [float(value).hex() for value in row]
            for row in evidence.raw_scores.tolist()
        ],
        "margin_names": list(MARGIN_NAMES),
        "margin_values_float_hex": [
            [float(value).hex() for value in row]
            for row in evidence.margins.tolist()
        ],
        "source_compatibility_kind": SOURCE_COMPATIBILITY_KIND,
        "source_compatibility_float_hex": [
            float(value).hex() for value in evidence.source_compatibility.tolist()
        ],
        "scorer_receipt_sha256": evidence.provenance.scorer_receipt_sha256,
        "strict_positive_each_axis_required": True,
        "source_compatibility_used_only_after_hard_gates": True,
        "temperature_float_hex": float(TEMPERATURE).hex(),
        "feasible_indices": list(feasible_indices),
        "weights_float_hex": [float(value).hex() for value in weights.tolist()],
        "abstained": abstained,
        "all_failed_candidates_retained": True,
        "best_of_k_fallback_used": False,
        "optimizer_update_authorized": False,
        "external_sealed_training_authority_required": True,
        "proposal_rgb_aggregated": False,
        "proposal_latent_aggregated": False,
        "raw_velocity_aggregated": False,
        "action_program_aggregated": not abstained,
        "target_video_used": False,
        "proposal_rgb_used": False,
        "proposal_latent_used": False,
        "raw_velocity_used": False,
        "mask_used": False,
        "pose_used": False,
        "flow_used": False,
        "track_used": False,
        "evaluator_selected_index_used": False,
        "semantic_action_success_authority": False,
        "identity_preservation_authority": False,
        "quality_or_production_authority": False,
        "upstream_receipt_semantic_authority": False,
        "same_process_security_boundary": False,
        "dynaedit_official_reproduction_claimed": False,
    }
    if program is not None:
        payload["aggregated_program_sha256"] = tensor_sha256(
            program, label="aggregated program"
        )
    return payload


class ASGASelection:
    __slots__ = (
        "weights",
        "aggregated_program",
        "feasible_indices",
        "abstained",
        "receipt",
        "bank_snapshot",
        "evidence_snapshot",
        "_token",
    )

    def __init__(
        self,
        *,
        weights: torch.Tensor,
        aggregated_program: Optional[torch.Tensor],
        feasible_indices: Tuple[int, ...],
        abstained: bool,
        receipt: Mapping[str, Any],
        bank_snapshot: AuthenticatedProposalBank,
        evidence_snapshot: AuthenticatedASGAEvidence,
        _token: object,
    ) -> None:
        if _token is not _SELECTION_TOKEN:
            raise GraftASGAError("ASGA selection must be factory-created")
        self.weights = weights
        self.aggregated_program = aggregated_program
        self.feasible_indices = feasible_indices
        self.abstained = abstained
        self.receipt = json.loads(canonical_json_bytes(receipt).decode("ascii"))
        self.bank_snapshot = bank_snapshot
        self.evidence_snapshot = evidence_snapshot
        self._token = _token
        self.validate()

    def validate(self) -> None:
        if self._token is not _SELECTION_TOKEN:
            raise GraftASGAError("ASGA selection token differs")
        if type(self.abstained) is not bool:
            raise GraftASGAError("ASGA abstained state must be bool")
        if type(self.feasible_indices) is not tuple:
            raise GraftASGAError("ASGA feasible indices must be tuple")
        expected_indices, expected_weights, expected_program, expected_abstained = (
            _derive_selection(self.bank_snapshot, self.evidence_snapshot)
        )
        weights = _owned_fp32(self.weights, label="ASGA weights", ndim=1)
        if tuple(weights.shape) != (CANDIDATE_COUNT,):
            raise GraftASGAError("ASGA weight shape differs")
        if bool(torch.any(weights < 0.0).item()):
            raise GraftASGAError("ASGA weights must be nonnegative")
        if self.feasible_indices != expected_indices:
            raise GraftASGAError("ASGA feasible indices differ from snapshots")
        if self.abstained is not expected_abstained:
            raise GraftASGAError("ASGA abstention differs from snapshots")
        if not torch.equal(weights, expected_weights):
            raise GraftASGAError("ASGA weights differ from snapshot recomputation")
        if expected_abstained:
            if self.aggregated_program is not None:
                raise GraftASGAError("abstaining selection retained a program")
        else:
            program = _owned_fp32(
                self.aggregated_program, label="aggregated program", ndim=2
            )
            if tuple(program.shape) != (PHASE_COUNT, PROGRAM_WIDTH):
                raise GraftASGAError("aggregated program shape differs")
            if not math.isclose(
                float(weights.sum().item()),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise GraftASGAError("ASGA weights do not sum to one")
            if not torch.equal(program, expected_program):
                raise GraftASGAError(
                    "aggregated program differs from snapshot recomputation"
                )
        expected_payload = _selection_receipt_payload(
            self.bank_snapshot,
            self.evidence_snapshot,
            feasible_indices=expected_indices,
            weights=expected_weights,
            program=expected_program,
            abstained=expected_abstained,
        )
        expected_receipt = dict(expected_payload)
        expected_receipt["receipt_digest"] = object_sha256(expected_payload)
        if canonical_json_bytes(self.receipt) != canonical_json_bytes(expected_receipt):
            raise GraftASGAError("ASGA selection receipt differs from snapshots")


def select_action_programs(
    bank: AuthenticatedProposalBank,
    evidence: AuthenticatedASGAEvidence,
) -> ASGASelection:
    """Apply fixed hard gates, then normalized-cosine aggregation."""

    bank_snapshot = _snapshot_bank(bank)
    evidence_snapshot = _snapshot_evidence(evidence)
    feasible_indices, weights, program, abstained = _derive_selection(
        bank_snapshot, evidence_snapshot
    )
    receipt_payload = _selection_receipt_payload(
        bank_snapshot,
        evidence_snapshot,
        feasible_indices=feasible_indices,
        weights=weights,
        program=program,
        abstained=abstained,
    )
    receipt = dict(receipt_payload)
    receipt["receipt_digest"] = object_sha256(receipt_payload)
    return ASGASelection(
        weights=weights.clone(),
        aggregated_program=None if program is None else program.clone(),
        feasible_indices=feasible_indices,
        abstained=abstained,
        receipt=receipt,
        bank_snapshot=bank_snapshot,
        evidence_snapshot=evidence_snapshot,
        _token=_SELECTION_TOKEN,
    )


__all__ = [
    "ASGAEvidenceProvenance",
    "ASGASelection",
    "AuthenticatedASGAEvidence",
    "AuthenticatedProposalBank",
    "BRANCH_COUNT",
    "CANDIDATE_COUNT",
    "CAPTIONER_RECEIPT_SCHEMA_VERSION",
    "COMPILER_RECEIPT_SCHEMA_VERSION",
    "COUNTERFACTUAL_BRANCH_ORDER",
    "EVIDENCE_SCHEMA_VERSION",
    "EXTRACTOR_RECEIPT_SCHEMA_VERSION",
    "GraftASGAError",
    "MARGIN_NAMES",
    "MARGIN_THRESHOLDS",
    "METHOD",
    "NORMALIZED_SCORE_MAX",
    "NORMALIZED_SCORE_MIN",
    "OFFLINE_SOURCE_CAPTION_ORIGIN",
    "ONLINE_SOURCE_CAPTION_ORIGIN",
    "PHASE_COUNT",
    "PHASE_ORDER",
    "PROGRAM_COORDINATE",
    "PROGRAM_EXTRACTOR_ALLOWED_INPUTS",
    "PROGRAM_WIDTH",
    "ProposalBankProvenance",
    "RAW_SCORE_NAMES",
    "SCHEMA_VERSION",
    "SCORER_ALLOWED_INPUTS",
    "SCORER_RECEIPT_SCHEMA_VERSION",
    "SOURCE_CAPTION_ORIGINS",
    "SOURCE_COMPATIBILITY_KIND",
    "SourceRetellingBinding",
    "TEMPERATURE",
    "authenticate_asga_evidence",
    "authenticate_proposal_bank",
    "bind_source_retelling",
    "canonical_json_bytes",
    "object_sha256",
    "select_action_programs",
    "tensor_sha256",
]
