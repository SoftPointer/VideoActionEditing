#!/usr/bin/env python3
"""Fail-closed, source-coordinate proposal selection for GRAFT-Edit.

This module deliberately separates two operations:

* canonical in-memory envelope inspection is *self-attested* and can never
  release a program tensor;
* release requires an envelope loaded through the pinned, plain-0444,
  ``O_NOFOLLOW`` file boundary implemented here.

The envelope contains the canonical bytes and byte hashes of the calibration,
intervention, every trial execution receipt, every per-axis evaluator receipt,
and the exact-81 post execution/evaluator receipts.  Observations are rebuilt
only from the ten evaluator receipts.  Callers cannot supply score tensors.

This remains an operational integrity boundary, not a same-process Python
security proof, a semantic-correctness authority, or optimizer authority.
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
import struct
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch

import graft_action_first_source_guided_aggregation_v1 as asga


METHOD = "bernini-graft-source-conditioned-proposal-selector-v1"
SCHEMA_VERSION = "bernini-graft-source-conditioned-selection-v2"
ENVELOPE_SCHEMA_VERSION = "bernini-graft-selector-execution-envelope-v2"
CALIBRATION_SCHEMA_VERSION = "bernini-graft-selector-calibration-v2"
TRIAL_SCHEMA_VERSION = "bernini-graft-source-conditioned-trial-execution-v2"
POST_COMMIT_SCHEMA_VERSION = "bernini-graft-selector-exact81-execution-v2"
EVALUATOR_SCHEMA_VERSION = "bernini-graft-selector-axis-evaluator-v2"
INTERVENTION_SCHEMA_VERSION = "bernini-graft-selector-intervention-v2"

CANDIDATE_COUNT = asga.CANDIDATE_COUNT
FRAME_COUNT = 81
MAX_ENVELOPE_BYTES = 8 * 1024 * 1024
RAW_AXIS_NAMES = (
    "action_event",
    "noop_event",
    "reverse_event",
    "incomplete_event",
    "terminal_hold",
    "identity_preservation",
    "camera_preservation",
    "background_preservation",
    "non_target_preservation",
    "perceptual_quality",
)
RAW_AXIS_DIRECTIONS = (
    "higher_is_better",
    "lower_is_better",
    "lower_is_better",
    "lower_is_better",
    "higher_is_better",
    "higher_is_better",
    "higher_is_better",
    "higher_is_better",
    "higher_is_better",
    "higher_is_better",
)
GATE_NAMES = (
    "action_event",
    "action_minus_noop",
    "action_minus_reverse",
    "action_minus_incomplete",
    "terminal_hold",
    "identity_preservation",
    "camera_preservation",
    "background_preservation",
    "non_target_preservation",
    "perceptual_quality",
)
INTERVENTION_KINDS = (
    "baseline_source_and_retelling",
    "wrong_retelling",
    "drop_visual",
    "drop_retelling",
)
ABSENT_RETELLING_DIGEST = hashlib.sha256(
    b"GRAFT_SELECTOR_INTENTIONAL_ABSENT_RETELLING_V1"
).hexdigest()

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FP32_BITS_RE = re.compile(r"^[0-9a-f]{8}$")
_SELECTION_TOKEN = object()
_LOADED_ENVELOPE_TOKEN = object()


class GraftSelectorError(RuntimeError):
    """An envelope, binding, gate, or release contract was violated."""


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
        raise GraftSelectorError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise GraftSelectorError(f"{label} must be lowercase SHA-256")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    if type(value) is not dict:
        raise GraftSelectorError(f"{label} must be a plain mapping")
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise GraftSelectorError(
            f"{label} keys differ; missing={sorted(required-actual)!r}, "
            f"extra={sorted(actual-required)!r}"
        )


def _parse_canonical_object(raw: bytes, *, label: str) -> Mapping[str, Any]:
    if type(raw) is not bytes or not raw:
        raise GraftSelectorError(f"{label} must be non-empty canonical bytes")
    try:
        text = raw.decode("ascii")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GraftSelectorError(f"{label} is not canonical ASCII JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise GraftSelectorError(f"{label} bytes are not canonical")
    return value


def _parse_embedded_receipt(
    canonical_text: Any,
    expected_sha256: Any,
    *,
    expected_keys: Sequence[str],
    label: str,
) -> Tuple[Mapping[str, Any], str]:
    if type(canonical_text) is not str:
        raise GraftSelectorError(f"{label} canonical bytes must be ASCII text")
    try:
        raw = canonical_text.encode("ascii")
    except UnicodeEncodeError as error:
        raise GraftSelectorError(f"{label} canonical bytes must be ASCII") from error
    receipt_sha = _require_sha(expected_sha256, label=f"{label} byte SHA")
    if hashlib.sha256(raw).hexdigest() != receipt_sha:
        raise GraftSelectorError(f"{label} canonical byte hash differs")
    receipt = _parse_canonical_object(raw, label=label)
    _exact_keys(receipt, expected_keys, label)
    payload = dict(receipt)
    self_digest = _require_sha(payload.pop("receipt_digest", None), label=f"{label} self digest")
    if object_sha256(payload) != self_digest:
        raise GraftSelectorError(f"{label} self digest differs")
    return receipt, receipt_sha


def _float32_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", value))[0]


def fp32_encoding(value: float) -> Mapping[str, str]:
    """Return the one accepted exact FP32 JSON encoding (useful to producers)."""

    bits = _float32_bits(float(value))
    exact = struct.unpack(">f", struct.pack(">I", bits))[0]
    return {"fp32_bits": f"{bits:08x}", "fp32_hex": float(exact).hex()}


def _decode_fp32(value: Any, *, label: str, threshold: bool = False) -> float:
    _exact_keys(value, ("fp32_bits", "fp32_hex"), label)
    bits_text = value["fp32_bits"]
    hex_text = value["fp32_hex"]
    if type(bits_text) is not str or _FP32_BITS_RE.fullmatch(bits_text) is None:
        raise GraftSelectorError(f"{label} has non-canonical FP32 bits")
    if type(hex_text) is not str:
        raise GraftSelectorError(f"{label} has non-canonical FP32 hex")
    bits = int(bits_text, 16)
    decoded = struct.unpack(">f", struct.pack(">I", bits))[0]
    if not math.isfinite(decoded) or float(decoded).hex() != hex_text:
        raise GraftSelectorError(f"{label} FP32 bits/hex do not round-trip exactly")
    if _float32_bits(decoded) != bits:
        raise GraftSelectorError(f"{label} FP32 encoding is not exact")
    # Reject negative zero for observations and all negative scores.
    if bits >> 31 or decoded < 0.0:
        raise GraftSelectorError(f"{label} must be non-negative FP32")
    if threshold:
        exponent = (bits >> 23) & 0xFF
        # Subnormal thresholds can underflow in downstream arithmetic and are
        # therefore not accepted even though Python can represent them.
        if exponent == 0 or not (0.0 < decoded < 1.0):
            raise GraftSelectorError(
                f"{label} threshold must be normal exact FP32 strictly in (0,1)"
            )
    elif decoded > 1.0:
        raise GraftSelectorError(f"{label} observation must lie in [0,1]")
    return decoded


def tensor_sha256(value: torch.Tensor) -> str:
    owned = value.detach().to("cpu").contiguous()
    raw = bytes(owned.view(torch.uint8).reshape(-1).tolist())
    header = canonical_json_bytes(
        {"dtype": str(owned.dtype), "shape": list(map(int, owned.shape))}
    )
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


class LoadedExecutionEnvelope:
    """Opaque result of the pinned plain-file loader.

    Python cannot provide an in-process cryptographic capability boundary;
    the private token only prevents accidental direct construction.  Release
    integrity comes from the external expected SHA plus file checks.
    """

    __slots__ = ("_canonical_bytes", "file_sha256", "source_path", "_token")

    def __init__(
        self, canonical_bytes: bytes, file_sha256: str, source_path: str, *, _token: object
    ) -> None:
        if _token is not _LOADED_ENVELOPE_TOKEN:
            raise GraftSelectorError("loaded envelope must be file-loader-created")
        self._canonical_bytes = bytes(canonical_bytes)
        self.file_sha256 = file_sha256
        self.source_path = source_path
        self._token = _token


def load_sealed_execution_envelope(
    path: Union[str, os.PathLike[str]], *, expected_file_sha256: str
) -> LoadedExecutionEnvelope:
    """Load one exact envelope through a create/freeze-style release boundary."""

    expected = _require_sha(expected_file_sha256, label="expected envelope file SHA")
    source = os.fspath(path)
    if not os.path.isabs(source):
        raise GraftSelectorError("sealed envelope path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
    except OSError as error:
        raise GraftSelectorError("sealed envelope open failed") from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise GraftSelectorError("sealed envelope must be a regular file")
        if stat.S_IMODE(before.st_mode) != 0o444:
            raise GraftSelectorError("sealed envelope must have exact plain 0444 mode")
        if before.st_nlink != 1:
            raise GraftSelectorError("sealed envelope must have exactly one hard link")
        if before.st_size <= 0 or before.st_size > MAX_ENVELOPE_BYTES:
            raise GraftSelectorError("sealed envelope size is outside the fixed bound")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise GraftSelectorError("sealed envelope was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise GraftSelectorError("sealed envelope grew during read")
        after = os.fstat(fd)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
            before.st_nlink,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
            after.st_nlink,
        )
        if not stable:
            raise GraftSelectorError("sealed envelope metadata changed during read")
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise GraftSelectorError("sealed envelope differs from external pinned file SHA")
    _parse_canonical_object(raw, label="sealed execution envelope")
    return LoadedExecutionEnvelope(raw, actual, str(Path(source)), _token=_LOADED_ENVELOPE_TOKEN)


_CALIBRATION_KEYS = (
    "schema_version",
    "selector_method",
    "raw_axis_names",
    "gate_names",
    "axis_registry",
    "calibration_dataset_receipt_sha256",
    "calibration_run_receipt_sha256",
    "frozen_before_trial_execution",
    "disjoint_from_trial_sources",
    "trial_source_read",
    "proposal_bank_read",
    "target_video_read",
    "proposal_media_read",
    "mask_pose_flow_track_read",
    "semantic_correctness_authority",
    "optimizer_authority",
    "same_process_security_boundary",
    "receipt_digest",
)


_AXIS_REGISTRY_KEYS = (
    "axis_index", "axis_name", "score_direction", "score_domain", "gate_name",
    "gate_direction", "threshold_exact_fp32",
    "producer_code_sha256", "model_artifact_sha256",
    "model_config_sha256", "evaluator_runtime_sha256", "preprocess_config_sha256",
    "temporal_exact81_protocol_sha256", "temporal_frame_count",
    "score_aggregation_sha256", "prompt_counterfactual_sha256",
)


def _validate_calibration(
    receipt: Mapping[str, Any],
) -> Tuple[Tuple[float, ...], Tuple[Mapping[str, Any], ...]]:
    expected = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "selector_method": METHOD,
        "raw_axis_names": list(RAW_AXIS_NAMES),
        "gate_names": list(GATE_NAMES),
        "frozen_before_trial_execution": True,
        "disjoint_from_trial_sources": True,
        "trial_source_read": False,
        "proposal_bank_read": False,
        "target_video_read": False,
        "proposal_media_read": False,
        "mask_pose_flow_track_read": False,
        "semantic_correctness_authority": False,
        "optimizer_authority": False,
        "same_process_security_boundary": False,
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value or type(receipt.get(key)) is not type(expected_value):
            raise GraftSelectorError(f"calibration field {key!r} differs")
    _require_sha(receipt["calibration_dataset_receipt_sha256"], label="calibration dataset SHA")
    _require_sha(receipt["calibration_run_receipt_sha256"], label="calibration run SHA")
    registry = receipt["axis_registry"]
    if type(registry) is not list or len(registry) != len(RAW_AXIS_NAMES):
        raise GraftSelectorError("calibration axis registry must contain exactly ten entries")
    thresholds = []
    owned_registry = []
    for index, entry in enumerate(registry):
        _exact_keys(entry, _AXIS_REGISTRY_KEYS, f"calibration axis registry {index}")
        exact = {
            "axis_index": index,
            "axis_name": RAW_AXIS_NAMES[index],
            "score_direction": RAW_AXIS_DIRECTIONS[index],
            "score_domain": "closed_unit_interval",
            "gate_name": GATE_NAMES[index],
            "gate_direction": "strictly_greater_than_threshold",
            "temporal_frame_count": FRAME_COUNT,
        }
        for key, expected_value in exact.items():
            if entry.get(key) != expected_value or type(entry.get(key)) is not type(expected_value):
                raise GraftSelectorError(f"calibration axis registry {index} field {key!r} differs")
        for name in (
            "producer_code_sha256", "model_artifact_sha256", "model_config_sha256",
            "evaluator_runtime_sha256", "preprocess_config_sha256",
            "temporal_exact81_protocol_sha256", "score_aggregation_sha256",
            "prompt_counterfactual_sha256",
        ):
            _require_sha(entry.get(name), label=f"calibration axis {index} {name}")
        thresholds.append(
            _decode_fp32(
                entry.get("threshold_exact_fp32"),
                label=f"calibration threshold {index}",
                threshold=True,
            )
        )
        owned_registry.append(
            json.loads(canonical_json_bytes(entry).decode("ascii"))
        )
    return tuple(thresholds), tuple(owned_registry)


_INTERVENTION_KEYS = (
    "schema_version",
    "kind",
    "proposal_bank_digest",
    "original_source_video_sha256",
    "original_retelling_digest",
    "effective_retelling_digest",
    "source_visual_condition_present",
    "source_retelling_condition_present",
    "wrong_retelling_intentional",
    "semantic_interpretation_authority",
    "optimizer_authority",
    "receipt_digest",
)


@dataclass(frozen=True)
class InterventionProvenance:
    kind: str
    effective_retelling_digest: str
    source_visual_condition_present: bool
    source_retelling_condition_present: bool
    receipt_sha256: str


def _validate_intervention(
    receipt: Mapping[str, Any], receipt_sha: str, bank: asga.AuthenticatedProposalBank
) -> InterventionProvenance:
    expected = {
        "schema_version": INTERVENTION_SCHEMA_VERSION,
        "proposal_bank_digest": bank.provenance.digest,
        "original_source_video_sha256": bank.retelling.source_video_sha256,
        "original_retelling_digest": bank.retelling.digest,
        "semantic_interpretation_authority": False,
        "optimizer_authority": False,
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value or type(receipt.get(key)) is not type(expected_value):
            raise GraftSelectorError(f"intervention field {key!r} differs")
    kind = receipt.get("kind")
    if kind not in INTERVENTION_KINDS:
        raise GraftSelectorError("intervention kind differs")
    effective = _require_sha(receipt.get("effective_retelling_digest"), label="effective retelling SHA")
    flags = (
        receipt.get("source_visual_condition_present"),
        receipt.get("source_retelling_condition_present"),
        receipt.get("wrong_retelling_intentional"),
    )
    if any(type(item) is not bool for item in flags):
        raise GraftSelectorError("intervention condition flags must be bool")
    expected_by_kind = {
        "baseline_source_and_retelling": (bank.retelling.digest, True, True, False),
        "wrong_retelling": (None, True, True, True),
        "drop_visual": (bank.retelling.digest, False, True, False),
        "drop_retelling": (ABSENT_RETELLING_DIGEST, True, False, False),
    }
    expected_digest, visual, retelling, wrong = expected_by_kind[kind]
    if expected_digest is not None and effective != expected_digest:
        raise GraftSelectorError("intervention effective retelling binding differs")
    if kind == "wrong_retelling" and effective in (bank.retelling.digest, ABSENT_RETELLING_DIGEST):
        raise GraftSelectorError("wrong-retelling intervention did not bind a distinct retelling")
    if flags != (visual, retelling, wrong):
        raise GraftSelectorError("intervention condition flags differ")
    return InterventionProvenance(kind, effective, visual, retelling, receipt_sha)


_TRIAL_KEYS = (
    "schema_version", "selector_method", "execution_kind", "execution_id", "candidate_index",
    "proposal_bank_digest", "source_video_sha256", "original_retelling_digest",
    "effective_retelling_digest", "intervention_receipt_sha256",
    "calibration_receipt_sha256", "instruction_sha256",
    "candidate_gaussian_raw_sha256", "schedule_digest", "program_slice_sha256",
    "counterfactual_execution_receipt_sha256s", "matched_runtime_config_digest",
    "frame_count", "output_artifact_sha256", "output_artifact_byte_size",
    "output_artifact_shape", "output_artifact_dtype", "output_artifact_layout",
    "frame81_digest", "source_visual_condition_present",
    "source_retelling_condition_present", "proposal_rgb_read_by_selector",
    "proposal_latent_read_by_selector", "raw_velocity_read_by_selector",
    "target_video_read_by_selector", "mask_pose_flow_track_read_by_selector",
    "semantic_correctness_authority", "optimizer_authority",
    "same_process_security_boundary", "receipt_digest",
)

_POST_KEYS = (
    "schema_version", "selector_method", "execution_kind", "execution_id", "proposal_bank_digest",
    "selected_candidate_index", "selected_program_slice_sha256",
    "selected_trial_execution_receipt_sha256", "source_video_sha256",
    "original_retelling_digest", "effective_retelling_digest",
    "intervention_receipt_sha256", "calibration_receipt_sha256",
    "instruction_sha256", "gaussian_raw_sha256", "schedule_digest",
    "matched_runtime_config_digest", "frame_count", "output_artifact_sha256",
    "output_artifact_byte_size", "output_artifact_shape", "output_artifact_dtype",
    "output_artifact_layout", "frame81_digest", "program_executed_without_mutation",
    "source_visual_condition_present", "source_retelling_condition_present",
    "proposal_rgb_read_by_selector", "proposal_latent_read_by_selector",
    "raw_velocity_read_by_selector", "target_video_read_by_selector",
    "mask_pose_flow_track_read_by_selector", "semantic_correctness_authority",
    "optimizer_authority", "same_process_security_boundary", "receipt_digest",
)

_EVALUATOR_KEYS = (
    "schema_version", "selector_method", "evaluation_stage", "candidate_index",
    "axis_index", "axis_name", "score_direction", "score_domain", "gate_name",
    "gate_direction", "threshold_exact_fp32",
    "execution_receipt_sha256", "execution_id", "output_artifact_sha256",
    "output_artifact_frame81_digest", "axis_registry_entry_sha256",
    "producer_code_sha256", "model_artifact_sha256", "model_config_sha256",
    "evaluator_runtime_sha256", "preprocess_config_sha256",
    "temporal_exact81_protocol_sha256", "temporal_frame_count",
    "score_aggregation_sha256", "prompt_counterfactual_sha256",
    "value_exact_fp32", "frame_count", "semantic_correctness_authority",
    "optimizer_authority", "receipt_digest",
)


@dataclass(frozen=True)
class OutputBinding:
    artifact_sha256: str
    byte_size: int
    shape: Tuple[int, int, int, int]
    frame81_digest: str


def _validate_output_binding(receipt: Mapping[str, Any], *, label: str) -> OutputBinding:
    artifact_sha = _require_sha(receipt.get("output_artifact_sha256"), label=f"{label} output artifact SHA")
    frame81_digest = _require_sha(receipt.get("frame81_digest"), label=f"{label} frame81 digest")
    byte_size = receipt.get("output_artifact_byte_size")
    shape = receipt.get("output_artifact_shape")
    if type(byte_size) is not int or byte_size <= 0:
        raise GraftSelectorError(f"{label} output artifact byte size differs")
    if (
        type(shape) is not list
        or len(shape) != 4
        or any(type(item) is not int or item <= 0 for item in shape)
        or shape[0] != FRAME_COUNT
        or shape[3] != 3
    ):
        raise GraftSelectorError(f"{label} output artifact shape must be exact81 THWC RGB")
    if receipt.get("output_artifact_dtype") != "uint8" or type(receipt.get("output_artifact_dtype")) is not str:
        raise GraftSelectorError(f"{label} output artifact dtype differs")
    if receipt.get("output_artifact_layout") != "THWC_RGB" or type(receipt.get("output_artifact_layout")) is not str:
        raise GraftSelectorError(f"{label} output artifact layout differs")
    expected_size = math.prod(shape)
    if byte_size != expected_size:
        raise GraftSelectorError(f"{label} output artifact size/shape binding differs")
    return OutputBinding(artifact_sha, byte_size, tuple(shape), frame81_digest)


def _validate_axis_receipts(
    entries: Any,
    *,
    stage: str,
    candidate_index: int,
    execution_sha: str,
    execution_id: str,
    output: OutputBinding,
    axis_registry: Tuple[Mapping[str, Any], ...],
) -> torch.Tensor:
    if type(entries) is not list or len(entries) != len(RAW_AXIS_NAMES):
        raise GraftSelectorError(f"{stage} candidate {candidate_index} needs exactly ten evaluator receipts")
    values = []
    byte_shas = []
    for axis_index, entry in enumerate(entries):
        _exact_keys(entry, ("canonical_json", "sha256"), f"{stage} evaluator envelope {axis_index}")
        receipt, receipt_sha = _parse_embedded_receipt(
            entry["canonical_json"], entry["sha256"],
            expected_keys=_EVALUATOR_KEYS,
            label=f"{stage} evaluator receipt {axis_index}",
        )
        expected = {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "selector_method": METHOD,
            "evaluation_stage": stage,
            "candidate_index": candidate_index,
            "axis_index": axis_index,
            "axis_name": RAW_AXIS_NAMES[axis_index],
            "score_direction": axis_registry[axis_index]["score_direction"],
            "score_domain": axis_registry[axis_index]["score_domain"],
            "gate_name": axis_registry[axis_index]["gate_name"],
            "gate_direction": axis_registry[axis_index]["gate_direction"],
            "threshold_exact_fp32": axis_registry[axis_index]["threshold_exact_fp32"],
            "execution_receipt_sha256": execution_sha,
            "execution_id": execution_id,
            "output_artifact_sha256": output.artifact_sha256,
            "output_artifact_frame81_digest": output.frame81_digest,
            "axis_registry_entry_sha256": object_sha256(axis_registry[axis_index]),
            "producer_code_sha256": axis_registry[axis_index]["producer_code_sha256"],
            "model_artifact_sha256": axis_registry[axis_index]["model_artifact_sha256"],
            "model_config_sha256": axis_registry[axis_index]["model_config_sha256"],
            "evaluator_runtime_sha256": axis_registry[axis_index]["evaluator_runtime_sha256"],
            "preprocess_config_sha256": axis_registry[axis_index]["preprocess_config_sha256"],
            "temporal_exact81_protocol_sha256": axis_registry[axis_index]["temporal_exact81_protocol_sha256"],
            "temporal_frame_count": axis_registry[axis_index]["temporal_frame_count"],
            "score_aggregation_sha256": axis_registry[axis_index]["score_aggregation_sha256"],
            "prompt_counterfactual_sha256": axis_registry[axis_index]["prompt_counterfactual_sha256"],
            "frame_count": FRAME_COUNT,
            "semantic_correctness_authority": False,
            "optimizer_authority": False,
        }
        for key, expected_value in expected.items():
            if receipt.get(key) != expected_value or type(receipt.get(key)) is not type(expected_value):
                raise GraftSelectorError(f"{stage} evaluator {axis_index} field {key!r} differs")
        values.append(
            _decode_fp32(receipt.get("value_exact_fp32"), label=f"{stage} evaluator {axis_index} value")
        )
        byte_shas.append(receipt_sha)
    if len(set(byte_shas)) != len(RAW_AXIS_NAMES):
        raise GraftSelectorError(f"{stage} evaluator receipt bytes must be axis-unique")
    return torch.tensor(values, dtype=torch.float32).contiguous()


def _validate_trial_execution(
    receipt: Mapping[str, Any], *, index: int, bank: asga.AuthenticatedProposalBank,
    intervention: InterventionProvenance, calibration_sha: str,
) -> Tuple[str, str, OutputBinding]:
    expected = {
        "schema_version": TRIAL_SCHEMA_VERSION,
        "selector_method": METHOD,
        "execution_kind": "trial_candidate",
        "candidate_index": index,
        "proposal_bank_digest": bank.provenance.digest,
        "source_video_sha256": bank.retelling.source_video_sha256,
        "original_retelling_digest": bank.retelling.digest,
        "effective_retelling_digest": intervention.effective_retelling_digest,
        "intervention_receipt_sha256": intervention.receipt_sha256,
        "calibration_receipt_sha256": calibration_sha,
        "instruction_sha256": bank.retelling.instruction_sha256,
        "candidate_gaussian_raw_sha256": bank.provenance.branch_gaussian_raw_sha256s[index][0],
        "schedule_digest": bank.provenance.branch_schedule_digests[index][0],
        "program_slice_sha256": bank.provenance.candidate_slice_sha256s[index],
        "counterfactual_execution_receipt_sha256s": list(
            bank.provenance.branch_execution_receipt_sha256s[index]
        ),
        "frame_count": FRAME_COUNT,
        "source_visual_condition_present": intervention.source_visual_condition_present,
        "source_retelling_condition_present": intervention.source_retelling_condition_present,
        "proposal_rgb_read_by_selector": False,
        "proposal_latent_read_by_selector": False,
        "raw_velocity_read_by_selector": False,
        "target_video_read_by_selector": False,
        "mask_pose_flow_track_read_by_selector": False,
        "semantic_correctness_authority": False,
        "optimizer_authority": False,
        "same_process_security_boundary": False,
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value or type(receipt.get(key)) is not type(expected_value):
            raise GraftSelectorError(f"trial {index} field {key!r} differs")
    execution_id = _require_sha(receipt.get("execution_id"), label=f"trial {index} execution id")
    runtime_digest = _require_sha(receipt.get("matched_runtime_config_digest"), label="trial runtime digest")
    output = _validate_output_binding(receipt, label=f"trial {index}")
    return runtime_digest, execution_id, output


def _validate_post_execution(
    receipt: Mapping[str, Any], *, selected_index: int, selected_trial_sha: str,
    bank: asga.AuthenticatedProposalBank, intervention: InterventionProvenance,
    calibration_sha: str, runtime_digest: str,
) -> Tuple[str, OutputBinding]:
    expected = {
        "schema_version": POST_COMMIT_SCHEMA_VERSION,
        "selector_method": METHOD,
        "execution_kind": "post_commit_selected_candidate",
        "proposal_bank_digest": bank.provenance.digest,
        "selected_candidate_index": selected_index,
        "selected_program_slice_sha256": bank.provenance.candidate_slice_sha256s[selected_index],
        "selected_trial_execution_receipt_sha256": selected_trial_sha,
        "source_video_sha256": bank.retelling.source_video_sha256,
        "original_retelling_digest": bank.retelling.digest,
        "effective_retelling_digest": intervention.effective_retelling_digest,
        "intervention_receipt_sha256": intervention.receipt_sha256,
        "calibration_receipt_sha256": calibration_sha,
        "instruction_sha256": bank.retelling.instruction_sha256,
        "gaussian_raw_sha256": bank.provenance.branch_gaussian_raw_sha256s[selected_index][0],
        "schedule_digest": bank.provenance.branch_schedule_digests[selected_index][0],
        "matched_runtime_config_digest": runtime_digest,
        "frame_count": FRAME_COUNT,
        "program_executed_without_mutation": True,
        "source_visual_condition_present": intervention.source_visual_condition_present,
        "source_retelling_condition_present": intervention.source_retelling_condition_present,
        "proposal_rgb_read_by_selector": False,
        "proposal_latent_read_by_selector": False,
        "raw_velocity_read_by_selector": False,
        "target_video_read_by_selector": False,
        "mask_pose_flow_track_read_by_selector": False,
        "semantic_correctness_authority": False,
        "optimizer_authority": False,
        "same_process_security_boundary": False,
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value or type(receipt.get(key)) is not type(expected_value):
            raise GraftSelectorError(f"post execution field {key!r} differs")
    execution_id = _require_sha(receipt.get("execution_id"), label="post execution id")
    output = _validate_output_binding(receipt, label="post execution")
    return execution_id, output


def _gate_values(observations: torch.Tensor) -> torch.Tensor:
    action = observations[:, 0]
    return torch.stack(
        (action, action - observations[:, 1], action - observations[:, 2],
         action - observations[:, 3], observations[:, 4], observations[:, 5],
         observations[:, 6], observations[:, 7], observations[:, 8], observations[:, 9]),
        dim=1,
    ).contiguous()


def _discrete_choice(slacks: torch.Tensor, feasible: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    frontier = []
    for candidate in feasible:
        row = slacks[candidate]
        dominated = any(
            other != candidate
            and bool(torch.all(slacks[other] >= row).item())
            and bool(torch.any(slacks[other] > row).item())
            for other in feasible
        )
        if not dominated:
            frontier.append(candidate)

    def key(candidate: int) -> Tuple[Any, ...]:
        row = slacks[candidate]
        medoid_cost = sum(
            float(torch.sum(torch.abs(row - slacks[other])).item()) for other in feasible
        )
        return (
            float(torch.min(row).item()),
            -medoid_cost,
            tuple(sorted(float(x) for x in row.tolist())),
            -candidate,
        )

    return tuple(frontier), max(frontier, key=key)


_ENVELOPE_KEYS = (
    "schema_version", "selector_method", "proposal_bank_digest",
    "calibration_receipt", "intervention_receipt", "trial_executions",
    "post_commit_execution", "proposal_rgb_present", "proposal_latent_present",
    "raw_velocity_present", "target_video_present", "mask_pose_flow_track_present",
    "semantic_correctness_authority", "optimizer_authority",
    "same_process_security_boundary", "receipt_digest",
)
_RECEIPT_ENTRY_KEYS = ("canonical_json", "sha256")
_EXECUTION_ENTRY_KEYS = ("execution_receipt", "evaluator_receipts")


@dataclass(frozen=True)
class SelectionProvenance:
    payload_json: str
    digest: str

    def payload(self) -> Mapping[str, Any]:
        value = _parse_canonical_object(self.payload_json.encode("ascii"), label="selection provenance")
        return value


class SourceConditionedSelection:
    __slots__ = ("selected_program", "provenance", "_token")

    def __init__(self, selected_program: Optional[torch.Tensor], provenance: SelectionProvenance, *, _token: object) -> None:
        if _token is not _SELECTION_TOKEN:
            raise GraftSelectorError("selection must be factory-created")
        self.selected_program = selected_program
        self.provenance = provenance
        self._token = _token
        self.validate()

    def validate(self) -> None:
        if self._token is not _SELECTION_TOKEN:
            raise GraftSelectorError("selection token differs")
        payload = self.provenance.payload()
        if object_sha256(payload) != self.provenance.digest:
            raise GraftSelectorError("selection provenance digest differs")
        accepted = payload.get("accepted_candidate_index")
        trusted = payload.get("release_envelope_loaded_from_pinned_plain0444")
        if type(trusted) is not bool:
            raise GraftSelectorError("release trust observation differs")
        if accepted is None:
            if self.selected_program is not None:
                raise GraftSelectorError("non-released selection contains a program")
        else:
            if not trusted or type(accepted) is not int or self.selected_program is None:
                raise GraftSelectorError("accepted selection lacks trusted exact program")
            owned = self.selected_program.detach().to("cpu").contiguous()
            if tuple(owned.shape) != (asga.PHASE_COUNT, asga.PROGRAM_WIDTH):
                raise GraftSelectorError("selected program shape differs")
            if tensor_sha256(owned) != payload.get("selected_program_slice_sha256"):
                raise GraftSelectorError("selected program was mutated or interpolated")
        if payload.get("optimizer_authority") is not False or payload.get("semantic_correctness_authority") is not False:
            raise GraftSelectorError("selector cannot grant semantic or optimizer authority")


@dataclass(frozen=True)
class SelfAttestedEnvelopeInspection:
    """Structure-only report; intentionally has no selected-program channel."""

    payload_json: str
    digest: str

    def payload(self) -> Mapping[str, Any]:
        payload = _parse_canonical_object(
            self.payload_json.encode("ascii"), label="self-attested inspection"
        )
        if object_sha256(payload) != self.digest:
            raise GraftSelectorError("self-attested inspection digest differs")
        return payload


def inspect_self_attested_execution_envelope(
    canonical_bytes: bytes,
) -> SelfAttestedEnvelopeInspection:
    """Inspect only canonical structure; never parse scores or choose a slice."""

    if type(canonical_bytes) is not bytes:
        raise GraftSelectorError("self-attested inspection requires canonical bytes")
    raw = bytes(canonical_bytes)
    envelope = _parse_canonical_object(raw, label="self-attested execution envelope")
    _exact_keys(envelope, _ENVELOPE_KEYS, "self-attested execution envelope")
    unsigned = dict(envelope)
    self_digest = _require_sha(unsigned.pop("receipt_digest", None), label="envelope self digest")
    if object_sha256(unsigned) != self_digest:
        raise GraftSelectorError("execution envelope self digest differs")
    _exact_keys(
        envelope.get("calibration_receipt"), _RECEIPT_ENTRY_KEYS,
        "self-attested calibration receipt entry",
    )
    _exact_keys(
        envelope.get("intervention_receipt"), _RECEIPT_ENTRY_KEYS,
        "self-attested intervention receipt entry",
    )
    trials = envelope.get("trial_executions")
    if type(trials) is not list:
        raise GraftSelectorError("self-attested trial container must be a list")
    trial_axis_counts = []
    embedded_receipt_count = 2  # calibration and intervention receipt entries
    for index, trial in enumerate(trials):
        _exact_keys(trial, _EXECUTION_ENTRY_KEYS, f"self-attested trial envelope {index}")
        _exact_keys(
            trial.get("execution_receipt"), _RECEIPT_ENTRY_KEYS,
            f"self-attested trial execution entry {index}",
        )
        axes = trial.get("evaluator_receipts")
        if type(axes) is not list:
            raise GraftSelectorError("self-attested trial evaluator container must be a list")
        for axis_index, entry in enumerate(axes):
            _exact_keys(entry, _RECEIPT_ENTRY_KEYS, f"self-attested trial axis entry {index}:{axis_index}")
        trial_axis_counts.append(len(axes))
        embedded_receipt_count += 1 + len(axes)
    post = envelope.get("post_commit_execution")
    post_axis_count = 0
    if post is not None:
        _exact_keys(post, _EXECUTION_ENTRY_KEYS, "self-attested post envelope")
        _exact_keys(post.get("execution_receipt"), _RECEIPT_ENTRY_KEYS, "self-attested post execution entry")
        post_axes = post.get("evaluator_receipts")
        if type(post_axes) is not list:
            raise GraftSelectorError("self-attested post evaluator container must be a list")
        for axis_index, entry in enumerate(post_axes):
            _exact_keys(entry, _RECEIPT_ENTRY_KEYS, f"self-attested post axis entry {axis_index}")
        post_axis_count = len(post_axes)
        embedded_receipt_count += 1 + post_axis_count
    payload = {
        "inspection_schema_version": "bernini-graft-self-attested-envelope-inspection-v1",
        "canonical_envelope_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_envelope_byte_size": len(raw),
        "top_level_key_count": len(envelope),
        "trial_execution_count": len(trials),
        "trial_axis_receipt_counts": trial_axis_counts,
        "post_execution_present": post is not None,
        "post_axis_receipt_count": post_axis_count,
        "embedded_receipt_entry_count": embedded_receipt_count,
        "canonical_structure_only": True,
        "embedded_receipt_payloads_parsed": False,
        "score_values_parsed": False,
        "score_derived_outcome_computed": False,
        "program_released": False,
        "semantic_correctness_authority": False,
        "optimizer_authority": False,
        "same_process_security_boundary": False,
    }
    return SelfAttestedEnvelopeInspection(
        canonical_json_bytes(payload).decode("ascii"), object_sha256(payload)
    )


def select_source_conditioned_proposal(
    bank: asga.AuthenticatedProposalBank,
    execution_envelope: LoadedExecutionEnvelope,
) -> SourceConditionedSelection:
    """Select only from a sealed-loader object; raw bytes belong to inspect API."""

    if type(bank) is not asga.AuthenticatedProposalBank:
        raise GraftSelectorError("bank must be an authenticated ASGA proposal bank")
    bank.validate()
    if type(execution_envelope) is not LoadedExecutionEnvelope:
        raise GraftSelectorError("selection requires a sealed-loader execution envelope")
    envelope_object = execution_envelope
    if envelope_object._token is not _LOADED_ENVELOPE_TOKEN:
        raise GraftSelectorError("loaded envelope token differs")
    raw = envelope_object._canonical_bytes
    envelope_sha = envelope_object.file_sha256
    if hashlib.sha256(raw).hexdigest() != envelope_sha:
        raise GraftSelectorError("loaded envelope bytes changed after pinned file read")
    source_path: Optional[str] = envelope_object.source_path
    loaded = True
    envelope = _parse_canonical_object(raw, label="execution envelope")
    _exact_keys(envelope, _ENVELOPE_KEYS, "execution envelope")
    payload_without_digest = dict(envelope)
    digest = _require_sha(payload_without_digest.pop("receipt_digest", None), label="envelope self digest")
    if object_sha256(payload_without_digest) != digest:
        raise GraftSelectorError("execution envelope self digest differs")
    fixed = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "selector_method": METHOD,
        "proposal_bank_digest": bank.provenance.digest,
        "proposal_rgb_present": False,
        "proposal_latent_present": False,
        "raw_velocity_present": False,
        "target_video_present": False,
        "mask_pose_flow_track_present": False,
        "semantic_correctness_authority": False,
        "optimizer_authority": False,
        "same_process_security_boundary": False,
    }
    for key, expected_value in fixed.items():
        if envelope.get(key) != expected_value or type(envelope.get(key)) is not type(expected_value):
            raise GraftSelectorError(f"execution envelope field {key!r} differs")

    calibration_entry = envelope["calibration_receipt"]
    _exact_keys(calibration_entry, _RECEIPT_ENTRY_KEYS, "calibration entry")
    calibration, calibration_sha = _parse_embedded_receipt(
        calibration_entry["canonical_json"], calibration_entry["sha256"],
        expected_keys=_CALIBRATION_KEYS, label="calibration receipt",
    )
    thresholds, axis_registry = _validate_calibration(calibration)

    intervention_entry = envelope["intervention_receipt"]
    _exact_keys(intervention_entry, _RECEIPT_ENTRY_KEYS, "intervention entry")
    intervention_receipt, intervention_sha = _parse_embedded_receipt(
        intervention_entry["canonical_json"], intervention_entry["sha256"],
        expected_keys=_INTERVENTION_KEYS, label="intervention receipt",
    )
    intervention = _validate_intervention(intervention_receipt, intervention_sha, bank)

    trial_entries = envelope["trial_executions"]
    if type(trial_entries) is not list or len(trial_entries) != CANDIDATE_COUNT:
        raise GraftSelectorError("execution envelope needs exactly five trials")
    observations = []
    trial_shas = []
    trial_execution_ids = []
    runtime_digest: Optional[str] = None
    for index, entry in enumerate(trial_entries):
        _exact_keys(entry, _EXECUTION_ENTRY_KEYS, f"trial envelope {index}")
        execution_entry = entry["execution_receipt"]
        _exact_keys(execution_entry, _RECEIPT_ENTRY_KEYS, f"trial execution entry {index}")
        trial, trial_sha = _parse_embedded_receipt(
            execution_entry["canonical_json"], execution_entry["sha256"],
            expected_keys=_TRIAL_KEYS, label=f"trial execution receipt {index}",
        )
        current_runtime, execution_id, output = _validate_trial_execution(
            trial, index=index, bank=bank, intervention=intervention, calibration_sha=calibration_sha
        )
        if runtime_digest is None:
            runtime_digest = current_runtime
        elif current_runtime != runtime_digest:
            raise GraftSelectorError("trial runtime configuration digests differ")
        observations.append(
            _validate_axis_receipts(
                entry["evaluator_receipts"], stage="trial", candidate_index=index,
                execution_sha=trial_sha, execution_id=execution_id, output=output,
                axis_registry=axis_registry,
            )
        )
        trial_shas.append(trial_sha)
        trial_execution_ids.append(execution_id)
    assert runtime_digest is not None
    if len(set(trial_shas)) != CANDIDATE_COUNT:
        raise GraftSelectorError("trial execution receipt bytes must be candidate-unique")
    if len(set(trial_execution_ids)) != CANDIDATE_COUNT:
        raise GraftSelectorError("trial execution ids must be candidate-unique")
    observation_tensor = torch.stack(observations).contiguous()
    gates = _gate_values(observation_tensor)
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32).reshape(1, -1)
    slacks = gates - threshold_tensor
    feasible = tuple(
        index for index in range(CANDIDATE_COUNT)
        if bool(torch.all(slacks[index] > 0.0).item())
    )
    frontier: Tuple[int, ...] = ()
    selected_index: Optional[int] = None
    post_sha: Optional[str] = None
    post_observation_sha: Optional[str] = None
    if feasible:
        frontier, selected_index = _discrete_choice(slacks, feasible)

    post_entry = envelope["post_commit_execution"]
    post_verified = False
    if selected_index is None:
        if post_entry is not None:
            raise GraftSelectorError("post execution cannot override empty feasible set")
    else:
        if type(post_entry) is not dict:
            raise GraftSelectorError("feasible selection requires exact81 post execution")
        _exact_keys(post_entry, _EXECUTION_ENTRY_KEYS, "post execution envelope")
        post_execution_entry = post_entry["execution_receipt"]
        _exact_keys(post_execution_entry, _RECEIPT_ENTRY_KEYS, "post execution receipt entry")
        post_receipt, post_sha = _parse_embedded_receipt(
            post_execution_entry["canonical_json"], post_execution_entry["sha256"],
            expected_keys=_POST_KEYS, label="exact81 post execution receipt",
        )
        post_execution_id, post_output = _validate_post_execution(
            post_receipt, selected_index=selected_index,
            selected_trial_sha=trial_shas[selected_index], bank=bank,
            intervention=intervention, calibration_sha=calibration_sha,
            runtime_digest=runtime_digest,
        )
        if post_execution_id in trial_execution_ids:
            raise GraftSelectorError("post execution id must be independent from every trial execution")
        post_observation = _validate_axis_receipts(
            post_entry["evaluator_receipts"], stage="post", candidate_index=selected_index,
            execution_sha=post_sha, execution_id=post_execution_id, output=post_output,
            axis_registry=axis_registry,
        )
        post_gates = _gate_values(post_observation.reshape(1, -1))[0]
        if not bool(torch.all(post_gates - threshold_tensor[0] > 0.0).item()):
            raise GraftSelectorError("exact81 post evaluator receipts failed a calibrated hard gate")
        post_observation_sha = tensor_sha256(post_observation)
        post_verified = True

    accepted_index = selected_index if loaded and post_verified else None
    selected_program = None
    if accepted_index is not None:
        selected_program = bank.tensor[accepted_index].detach().to("cpu").contiguous().clone()
    if selected_index is None:
        status = "empty_feasible_set"
    elif loaded:
        status = "pinned_plain0444_exact81_released"
    else:
        status = "self_attested_exact81_observation_no_release"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "proposal_bank_digest": bank.provenance.digest,
        "source_video_sha256": bank.retelling.source_video_sha256,
        "original_retelling_digest": bank.retelling.digest,
        "effective_retelling_digest": intervention.effective_retelling_digest,
        "instruction_sha256": bank.retelling.instruction_sha256,
        "execution_envelope_sha256": envelope_sha,
        "execution_envelope_source_path": source_path,
        "release_envelope_loaded_from_pinned_plain0444": loaded,
        "canonical_bytes_self_attested_only": not loaded,
        "calibration_receipt_sha256": calibration_sha,
        "intervention_receipt_sha256": intervention_sha,
        "trial_execution_receipt_sha256s": trial_shas,
        "post_execution_receipt_sha256": post_sha,
        "matched_runtime_config_digest": runtime_digest,
        "raw_axis_names": list(RAW_AXIS_NAMES),
        "gate_names": list(GATE_NAMES),
        "trial_observations_recomputed_from_axis_receipts": True,
        "post_observation_recomputed_from_axis_receipts": post_verified,
        "trial_observations_tensor_sha256": tensor_sha256(observation_tensor),
        "post_observation_tensor_sha256": post_observation_sha,
        "gate_thresholds_exact_fp32": [fp32_encoding(x) for x in thresholds],
        "feasible_candidate_indices": list(feasible),
        "pareto_frontier_candidate_indices": list(frontier),
        "observed_candidate_index": selected_index,
        "accepted_candidate_index": accepted_index,
        "selected_program_slice_sha256": None if accepted_index is None else bank.provenance.candidate_slice_sha256s[accepted_index],
        "frame_count_required": FRAME_COUNT,
        "post_commit_verified": post_verified,
        "selection_status": status,
        "selection_rule": "strict-positive-exact-fp32-gates_then_pareto_maxmin_medoid_lexicographic_fixed-index",
        "program_aggregation_used": False,
        "selected_program_is_exact_bank_slice": accepted_index is not None,
        "intervention_kind": intervention.kind,
        "source_visual_condition_present": intervention.source_visual_condition_present,
        "source_retelling_condition_present": intervention.source_retelling_condition_present,
        "proposal_rgb_in_public_api": False,
        "proposal_latent_in_public_api": False,
        "raw_velocity_in_public_api": False,
        "target_video_in_public_api": False,
        "mask_pose_flow_track_in_public_api": False,
        "optimizer_authority": False,
        "semantic_correctness_authority": False,
        "same_process_security_boundary": False,
    }
    provenance = SelectionProvenance(
        canonical_json_bytes(payload).decode("ascii"), object_sha256(payload)
    )
    return SourceConditionedSelection(selected_program, provenance, _token=_SELECTION_TOKEN)


def release_source_conditioned_proposal_from_file(
    bank: asga.AuthenticatedProposalBank,
    path: Union[str, os.PathLike[str]],
    *,
    expected_file_sha256: str,
) -> SourceConditionedSelection:
    """Only public convenience API that can release ``selected_program``."""

    loaded = load_sealed_execution_envelope(path, expected_file_sha256=expected_file_sha256)
    return select_source_conditioned_proposal(bank, loaded)


__all__ = (
    "ABSENT_RETELLING_DIGEST", "CALIBRATION_SCHEMA_VERSION", "ENVELOPE_SCHEMA_VERSION",
    "EVALUATOR_SCHEMA_VERSION", "FRAME_COUNT", "GATE_NAMES", "GraftSelectorError",
    "INTERVENTION_KINDS", "INTERVENTION_SCHEMA_VERSION", "LoadedExecutionEnvelope",
    "METHOD", "POST_COMMIT_SCHEMA_VERSION", "RAW_AXIS_DIRECTIONS", "RAW_AXIS_NAMES",
    "SCHEMA_VERSION",
    "SelfAttestedEnvelopeInspection", "SourceConditionedSelection", "TRIAL_SCHEMA_VERSION",
    "canonical_json_bytes", "fp32_encoding", "inspect_self_attested_execution_envelope",
    "load_sealed_execution_envelope", "object_sha256",
    "release_source_conditioned_proposal_from_file", "select_source_conditioned_proposal",
    "tensor_sha256",
)
