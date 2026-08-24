#!/usr/bin/env python3
"""Fail-closed authority validator for full30 action data and PsiOut teachers.

This module grants no authority by constructing or repairing evidence.  It
only validates a pre-existing, content-sealed manifest.  In particular it
never opens, hashes, or accepts a path for posterior index 1 / synthetic
target data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
from types import MappingProxyType
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence, Tuple


SCHEMA_VERSION = "bernini-full30-action-data-teacher-authority-v3"
VALIDATION_SCHEMA = "bernini-full30-action-data-teacher-validation-v3"
TEACHER_ORIGIN_SCHEMA = "bernini-full30-action-teacher-origin-v1"
SOURCE_SCHEMA = "bernini-full30-action-eligible-source-v1"
PAIR_SCHEMA = "bernini-full30-action-source-pair-v1"
REVIEW_SCHEMA = "bernini-full30-action-pair-full81-review-v1"
REPRESENTATION_SCHEMA = "bernini-psiout-representation-admission-v2"
REPRESENTATION_EVIDENCE_SCHEMA = "bernini-psiout-anchor-evidence-v2"
REPRESENTATION_REVIEW_SCHEMA = "bernini-psiout-anchor-full81-blind-review-v1"
TENSOR_CONTAINER_SCHEMA = "bernini-full30-action-tensor-container-v1"
MATERIALIZATION_RUN_BINDING_SCHEMA = (
    "bernini-full30-action-materialization-run-binding-v1"
)
MATERIALIZATION_RUN_RECEIPT_SCHEMA = (
    "bernini-full30-action-psiout-materialization-run-v1"
)
MATERIALIZATION_RECORD_RECEIPT_SCHEMA = (
    "bernini-full30-action-psiout-materialization-record-v1"
)
MATERIALIZATION_STATE_RECEIPT_SCHEMA = "bernini-full30-action-psiout-same-state-v1"
MATERIALIZATION_FORWARD_RECEIPT_SCHEMA = (
    "bernini-full30-action-psiout-frozen-forward-v1"
)
MATERIALIZATION_NOISE_RECEIPT_SCHEMA = (
    "bernini-full30-action-psiout-noise-replay-v1"
)
MATERIALIZATION_PROVIDER_ABI = "full30-psiout-official-provider-v1"
MATERIALIZATION_RUNTIME_SCHEMA = "bernini-full30-action-frozen-runtime-identity-v2"
MATERIALIZATION_COMPUTE_SCHEMA = "bernini-full30-action-frozen-compute-contract-v1"
MATERIALIZATION_OUTPUT_POLICY_SCHEMA = (
    "bernini-full30-action-psiout-output-policy-v1"
)
PINNED_PSIOUT_PROTOCOL_SHA256 = (
    "67275ae09e7cb7b1e7e8fc43ce2928031b3fe8aabe213e8626000f37abad4ead"
)
PINNED_SIGMA_TABLE_SHA256 = (
    "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
)
PINNED_MATERIALIZER_SOURCE_SHA256 = (
    "a7daf7f81956818669f2d23e806034ab902aa34bcbb8e76315f1d2ee89c53b45"
)

SPLITS = ("fit", "confirmation", "heldout")
BRANCHES = ("action", "incomplete")
SIGMA_INDICES = (4, 12, 20, 28, 35, 38)
EXPECTED_SOURCE_COUNTS = {"fit": 64, "confirmation": 16, "heldout": 8}
EXPECTED_PAIR_COUNTS = {"fit": 128, "confirmation": 32, "heldout": 16}
EXPECTED_TEACHER_COUNTS = {"fit": 8, "confirmation": 8}
EXPECTED_REPRESENTATION_COUNTS = {"fit": 16, "confirmation": 16}
ASSIGNMENT_CAPACITY = {"fit": 8, "confirmation": 2, "heldout": 1}

WRONG_CONTROL_TYPES = ("wrong_actor", "wrong_object", "generic_wrong_motion")
SAME_EVENT_MINIMUM_COSINE = 0.55
WRONG_CONTROL_MINIMUM_MARGIN = 0.20
DUPLICATE_MAX_NULL_NORM = 1.0e-7
PROJECTED_TEACHER_MIN_RAW_NORM = 1.0e-4
DUPLICATE_SNR_DENOMINATOR_FLOOR = 1.0e-8
DUPLICATE_MIN_SNR = 100.0
NUISANCE_MAX_ABS_COSINE = 1.0e-5

TENSOR_SHAPE = (21, 32)
TENSOR_ELEMENTS = math.prod(TENSOR_SHAPE)
TENSOR_DTYPE = "float32-le"
TENSOR_SLICE_BYTES = TENSOR_ELEMENTS * 4
TENSOR_CONTAINER_MODE = 0o600
TENSOR_CONTAINER_MAGIC = b"BERNINI-FULL30-TENSOR-V1\x00"
TENSOR_CONTAINER_MAX_HEADER_BYTES = 256 * 1024
TENSOR_CONTAINER_MAX_FILE_BYTES = 2 * 1024 * 1024
TENSOR_CONTAINER_LAYOUT = "payload-relative-contiguous-v1"
MATERIALIZATION_RECEIPT_MODE = 0o600
MATERIALIZATION_RECORD_MAX_BYTES = 16 * 1024 * 1024
MATERIALIZATION_RUN_MAX_BYTES = 128 * 1024 * 1024

_SIGMA_RUNTIME_BINDINGS = {
    4: ("3f7a70da", 978),
    12: ("3f6bd0e9", 921),
    20: ("3f556787", 833),
    28: ("3f2ebaf8", 682),
    35: ("3ed6539a", 418),
    38: ("3e58b351", 211),
}

_REQUIRED_HELPER_MODULES = (
    "clean_source_visual_context_training_v1",
    "dclr_runtime_contract",
    "full30_action_runtime_v1",
    "graft_phase_a_native_training_closure_v1",
    "infer_dclr_reward_runtime_smoke",
    "inference_sigma_strata",
    "packed_preservation_lora_v2",
    "packed_preservation_release_v2",
    "source_self_runtime",
    "temporal_counterfactual_action_scorer_v1",
    "train_lora",
    "train_packed_preservation_lora_v2",
)

_ORIGIN_PSIOUT_TENSOR_KINDS = (
    "projected_unit",
    "projected_raw",
    "duplicate_forward_first",
    "duplicate_forward_second",
    "noop_forward_first",
    "noop_forward_second",
    "wrong_actor_projected_unit",
    "wrong_object_projected_unit",
    "generic_wrong_motion_projected_unit",
)
_CROSS_PSIOUT_TENSOR_KINDS = ("projected_unit",)
_NUISANCE_TENSOR_KINDS = ("camera_unit", "appearance_unit")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IID = re.compile(r"^[0-9a-f]{16}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class Full30ActionAuthorityError(RuntimeError):
    """Raised before incomplete or ambiguous evidence can authorize training."""


def fail(message: str) -> NoReturn:
    raise Full30ActionAuthorityError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Full30ActionAuthorityError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_record(value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    """Return a copy with a canonical content digest; useful to author fixtures."""

    unsigned = dict(value)
    if digest_field in unsigned:
        del unsigned[digest_field]
    return {**unsigned, digest_field: object_sha256(unsigned)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def _closed(value: Any, fields: Iterable[str], label: str) -> Mapping[str, Any]:
    expected = set(fields)
    if type(value) is not dict or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        fail(
            f"{label} field closure differs: observed={observed!r}, "
            f"expected={sorted(expected)!r}"
        )
    return value


def _verify_seal(value: Mapping[str, Any], digest_field: str, label: str) -> None:
    declared = _sha(value.get(digest_field), f"{label}.{digest_field}")
    unsigned = dict(value)
    del unsigned[digest_field]
    _require(object_sha256(unsigned) == declared, f"{label} digest differs")


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _iid(value: Any, label: str) -> str:
    if type(value) is not str or _IID.fullmatch(value) is None:
        fail(f"{label} must be a lowercase 16-hex IID")
    return value


def _safe_id(value: Any, label: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        fail(f"{label} must be a non-empty safe identifier")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        fail(f"{label} must be non-empty trimmed text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        fail(f"{label} must be a finite number")
    return float(value)


def _plain_file(value: Any, label: str) -> Path:
    if type(value) is not str:
        fail(f"{label} path must be text")
    path = Path(value)
    if not path.is_absolute():
        fail(f"{label} path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Full30ActionAuthorityError(f"{label} is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a plain non-symlink file")
    return path.resolve(strict=True)


def _verify_file(path_value: Any, sha_value: Any, label: str) -> Path:
    path = _plain_file(path_value, label)
    expected = _sha(sha_value, f"{label} SHA-256")
    _require(file_sha256(path) == expected, f"{label} file SHA-256 differs")
    return path


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: str | Path, expected_sha256: str) -> Mapping[str, Any]:
    source = _plain_file(str(path), "authority manifest")
    expected = _sha(expected_sha256, "authority manifest expected SHA-256")
    raw = source.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected, "authority manifest file SHA-256 differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30ActionAuthorityError("cannot decode authority manifest") from error
    if type(value) is not dict:
        fail("authority manifest root must be an object")
    return value


def _read_stable_plain_bytes(
    path_value: Any,
    sha_value: Any,
    *,
    label: str,
    exact_mode: Optional[int],
    maximum_bytes: int,
) -> tuple[Path, bytes]:
    """Reopen one bounded artifact without following its final path component."""

    if type(path_value) is not str:
        fail(f"{label} path must be text")
    path = Path(path_value)
    if not path.is_absolute():
        fail(f"{label} path must be absolute")
    expected = _sha(sha_value, f"{label} SHA-256")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Full30ActionAuthorityError(f"{label} is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a plain non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Full30ActionAuthorityError(f"{label} cannot be opened safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        if exact_mode is not None:
            _require(
                stat.S_IMODE(before.st_mode) == exact_mode,
                f"{label} mode must be exactly {exact_mode:#o}",
            )
        _require(
            0 < before.st_size <= maximum_bytes,
            f"{label} size is outside the bounded range",
        )
        parts: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            _require(bool(block), f"{label} was truncated while reading")
            parts.append(block)
            remaining -= len(block)
        _require(os.read(descriptor, 1) == b"", f"{label} grew while reading")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    _require(
        all(getattr(before, field) == getattr(after, field) for field in stable),
        f"{label} metadata changed while reading",
    )
    raw = b"".join(parts)
    _require(len(raw) == before.st_size, f"{label} byte length changed while reading")
    _require(hashlib.sha256(raw).hexdigest() == expected, f"{label} file SHA-256 differs")
    return path.resolve(strict=True), raw


def _load_strict_json_receipt(
    path_value: Any,
    sha_value: Any,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[Path, Mapping[str, Any]]:
    path, raw = _read_stable_plain_bytes(
        path_value,
        sha_value,
        label=label,
        exact_mode=MATERIALIZATION_RECEIPT_MODE,
        maximum_bytes=maximum_bytes,
    )
    _require(raw.endswith(b"\n"), f"{label} lacks one canonical newline")
    payload = raw[:-1]
    _require(bool(payload) and not payload.endswith(b"\n"), f"{label} newline closure differs")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30ActionAuthorityError(f"cannot decode {label}") from error
    if type(value) is not dict:
        fail(f"{label} root must be an object")
    _require(canonical_json_bytes(value) == payload, f"{label} is not canonical JSON")
    return path, value


_MATERIALIZATION_RUN_BINDING_FIELDS = {
    "schema_version",
    "path",
    "file_sha256",
    "run_digest",
    "binding_digest",
}
_MATERIALIZATION_RUNTIME_FIELDS = {
    "schema_version",
    "bernini_revision",
    "veomni_revision",
    "official_checkpoint_tree_sha256",
    "transformer_config_sha256",
    "sigma_table_sha256",
    "psiout_protocol_sha256",
    "official_provider_source_sha256",
    "official_provider_abi",
    "compute_contract",
    "compute_contract_digest",
    "frame_count",
    "fps",
    "sampler_steps",
    "runtime_digest",
}
_MATERIALIZATION_COMPUTE_FIELDS = {
    "schema_version",
    "model_eval",
    "torch_inference_mode",
    "official_frozen_native_only",
    "calibrator_peft_adapter_present",
    "frozen_effective_adapter_enabled",
    "frozen_effective_typed_patch_role_enabled",
    "base_compute_dtype",
    "autocast_dtype",
    "observer_output_dtype",
    "observer_output_stage",
    "observer_output_detached",
    "observer_output_contiguous",
    "same_state_counterfactual",
    "branch_and_noop_share_input_state",
    "world_size",
    "dp_size",
    "sp_size",
    "sp_order_contract",
    "all_rank_consensus",
}
_MATERIALIZATION_HELPER_SOURCE_FIELDS = {"module", "path", "file_sha256"}
_MATERIALIZATION_OUTPUT_POLICY_FIELDS = {
    "schema_version",
    "create_only",
    "container_mode_octal",
    "generated_rgb_decoded",
    "generated_rgb_used_as_model_input",
    "generated_rgb_used_as_regression_target",
    "generated_latent_used_as_absolute_regression_target",
    "model_parameters_updated",
    "optimizer_created",
    "persisted_tensor_role",
}
_MATERIALIZATION_RUN_FIELDS = {
    "schema_version",
    "plan_id",
    "plan_digest",
    "plan_authority",
    "population_digest",
    "record_order_sha256",
    "runtime_identity",
    "runtime_plan_digest",
    "official_helper_sources",
    "provider_abi",
    "official_provider",
    "test_only",
    "world_size",
    "dp_size",
    "sp_size",
    "sigma_indices",
    "sigma_authority",
    "record_count",
    "computation_digest",
    "record_receipts",
    "representation_sigma_evidence_candidates",
    "amplitude_sigma_calibration_candidates",
    "output_policy",
    "generated_rgb_decoded",
    "generated_rgb_used_as_model_input",
    "generated_rgb_used_as_regression_target",
    "generated_latent_used_as_absolute_regression_target",
    "model_parameters_updated",
    "optimizer_created",
    "run_digest",
}
_MATERIALIZATION_RECORD_REF_FIELDS = {
    "record_id",
    "record_kind",
    "path",
    "file_sha256",
    "record_receipt_digest",
    "candidate_evidence_digest",
}
_MATERIALIZATION_RECORD_FIELDS = {
    "schema_version",
    "plan_id",
    "plan_digest",
    "runtime_digest",
    "provider_abi",
    "official_provider",
    "test_only",
    "record_ordinal",
    "record_id",
    "record_digest",
    "record_kind",
    "evidence_id",
    "evidence_role",
    "teacher_cell_id",
    "branch",
    "record_authority",
    "record_conditions",
    "review_digest",
    "reviewed_media_sha256",
    "target_clean_latent_raw_sha256",
    "target_clean_latent_authority_digest",
    "source_clean_latent_raw_sha256",
    "source_posterior_index0_sha256",
    "noise_seed",
    "noise_raw_sha256",
    "noise_replay_receipt",
    "sigma_authority_digest",
    "state_receipts",
    "forward_receipts",
    "container_bindings",
    "sigma_metrics",
    "candidate_authority_evidence",
    "generated_rgb_decoded",
    "generated_rgb_used_as_model_input",
    "generated_rgb_used_as_regression_target",
    "generated_latent_used_as_absolute_regression_target",
    "model_parameters_updated",
    "optimizer_created",
    "record_receipt_digest",
}
_MATERIALIZATION_SIGMA_ROW_FIELDS = {
    "sigma_index",
    "timestep",
    "sigma_float32_be_hex",
}
_MATERIALIZATION_SIGMA_AUTHORITY_FIELDS = {
    "schema_version",
    "schedule_schema_version",
    "schedule_sha256",
    "scheduler_class",
    "num_train_timesteps",
    "num_inference_steps",
    "flow_shift_float64_hex",
    "timesteps_int64",
    "positive_sigmas_float32_be_hex",
    "terminal_sigma_float32_be_hex",
    "materialized_sigma_indices",
    "materialized_rows",
    "sigma_authority_digest",
}
_MATERIALIZATION_SIGMA_AUTHORITY_SCHEMA = (
    "bernini-full30-action-psiout-sigma-authority-v1"
)
_MATERIALIZATION_SIGMA_AUTHORITY_DIGEST = (
    "dc452c9d7b1a0df867f5a60332ba109180ac39743b9d566aa4ed785d04e224a7"
)
_MATERIALIZATION_CONTAINER_BINDING_FIELDS = {
    "container_kind",
    "path",
    "file_sha256",
    "slice_sha256",
}
_MATERIALIZATION_REPRESENTATION_FRAGMENT_FIELDS = {
    "teacher_cell_id",
    "branch",
    "origin_record_id",
    "cross_anchor_record_id",
    "origin_evidence_digest",
    "cross_anchor_evidence_digest",
    "sigma_evidence",
}
_MATERIALIZATION_AMPLITUDE_FRAGMENT_FIELDS = {
    "teacher_cell_id",
    "branch",
    "calibrator_record_ids",
    "calibrator_evidence_candidates",
    "sigma_calibrations",
}
_MATERIALIZATION_PLAN_FIELDS = {
    "schema_version",
    "plan_id",
    "status",
    "runtime",
    "population",
    "records",
    "output_policy",
    "plan_digest",
}
_MATERIALIZATION_PLAN_SCHEMA = "bernini-full30-action-psiout-materialization-plan-v1"
_MATERIALIZATION_RUNTIME_PLAN_FIELDS = {
    "schema_version",
    "frozen_runtime_identity",
    "bernini_root",
    "veomni_root",
    "checkpoint_root",
    "checkpoint_content_manifest_path",
    "checkpoint_content_manifest_sha256",
    "psiout_protocol_path",
    "official_provider_source_path",
    "official_helper_sources",
    "runtime_plan_digest",
}
_MATERIALIZATION_RUNTIME_PLAN_SCHEMA = "bernini-full30-action-psiout-runtime-plan-v1"
_MATERIALIZATION_POPULATION_FIELDS = {
    "schema_version",
    "population_id",
    "record_count",
    "teacher_record_count",
    "amplitude_record_count",
    "teacher_cell_ids",
    "record_order_sha256",
    "finite_closed_population",
    "block_probe",
    "population_digest",
}
_MATERIALIZATION_POPULATION_SCHEMA = "bernini-full30-action-psiout-population-v1"
_MATERIALIZATION_PLAN_RECORD_FIELDS = {
    "schema_version",
    "record_id",
    "record_kind",
    "evidence_id",
    "evidence_role",
    "teacher_cell_id",
    "analysis_split",
    "branch",
    "event_id",
    "actor_kind",
    "q0_id",
    "actor_id",
    "scene_id",
    "anchor_id",
    "anchor_iid",
    "pair_id",
    "source_iid",
    "review",
    "reviewed_media",
    "target_clean_latent",
    "target_clean_latent_authority",
    "source_clean_latent",
    "source_posterior_index0_path",
    "source_posterior_index0_sha256",
    "source_posterior_tensor_key",
    "noise",
    "conditions",
    "record_digest",
}
_MATERIALIZATION_PLAN_RECORD_SCHEMA = "bernini-full30-action-psiout-record-v1"
_MATERIALIZATION_ARTIFACT_FIELDS = {
    "schema_version",
    "path",
    "file_sha256",
    "tensor_key",
    "shape",
    "dtype",
    "tensor_raw_sha256",
}
_MATERIALIZATION_ARTIFACT_SCHEMA = "bernini-full30-action-psiout-fp32-artifact-v1"
_MATERIALIZATION_LATENT_AUTHORITY_FIELDS = {
    "schema_version",
    "path",
    "file_sha256",
    "digest_field",
    "digest",
    "media_json_pointer",
    "latent_json_pointer",
    "checkpoint_tree_sha256_json_pointer",
}
_MATERIALIZATION_LATENT_AUTHORITY_SCHEMA = (
    "bernini-full30-action-psiout-anchor-latent-authority-v1"
)
_MATERIALIZATION_REVIEW_BINDING_FIELDS = {
    "schema_version",
    "path",
    "file_sha256",
    "review_digest",
}
_MATERIALIZATION_REVIEW_BINDING_SCHEMA = "bernini-full30-action-psiout-review-binding-v1"
_MATERIALIZATION_MEDIA_FIELDS = {"path", "file_sha256"}
_MATERIALIZATION_NOISE_AUTHORITY_FIELDS = {"artifact", "seed", "generator"}
_MATERIALIZATION_CONDITION_FIELDS = {
    "schema_version",
    "role",
    "instruction",
    "instruction_utf8_sha256",
    "authority_path",
    "authority_file_sha256",
    "authority_digest_field",
    "authority_digest",
    "json_pointer",
    "text_field",
    "sha256_field",
    "control_anchor_id",
}
_MATERIALIZATION_CONDITION_SCHEMA = "bernini-full30-action-psiout-condition-v1"
_MATERIALIZATION_PROVENANCE_FIELDS = {
    "materialization_record_receipt_path",
    "materialization_record_receipt_sha256",
    "materialization_record_receipt_digest",
    "materialization_run_digest",
}
_MATERIALIZATION_TEACHER_ORIGIN_METRIC_FIELDS = {
    "sigma_index",
    "state_digest",
    "projected_unit_sha256",
    "projected_raw_sha256",
    "duplicate_forward_first_sha256",
    "duplicate_forward_second_sha256",
    "duplicate_forward_bytes_identical",
    "noop_forward_first_sha256",
    "noop_forward_second_sha256",
    "same_state_noop_minus_noop_null_norm",
    "projected_teacher_raw_norm",
    "signal_to_null_snr",
    "camera_unit_sha256",
    "appearance_unit_sha256",
    "camera_residual_cosine",
    "appearance_residual_cosine",
    "wrong_controls",
}
_MATERIALIZATION_TEACHER_CROSS_METRIC_FIELDS = {
    "sigma_index",
    "state_digest",
    "projected_unit_sha256",
    "camera_unit_sha256",
    "appearance_unit_sha256",
}
_MATERIALIZATION_AMPLITUDE_METRIC_FIELDS = {
    "sigma_index",
    "state_digest",
    "projected_slice_sha256",
    "amplitude_norm",
    "teacher_nuisance_camera_sha256",
    "teacher_nuisance_appearance_sha256",
}
_MATERIALIZATION_NOISE_FIELDS = {
    "schema_version",
    "provider_abi",
    "official_provider",
    "record_id",
    "seed",
    "generator",
    "shape",
    "artifact_raw_sha256",
    "replayed_raw_sha256",
    "byte_exact_replay",
    "noise_digest",
}
_MATERIALIZATION_STATE_FIELDS = {
    "schema_version",
    "provider_abi",
    "official_provider",
    "runtime_digest",
    "record_id",
    "record_kind",
    "teacher_cell_id",
    "branch",
    "sigma_index",
    "sigma_float32_be_hex",
    "timestep",
    "clean_raw_sha256",
    "source_raw_sha256",
    "noise_raw_sha256",
    "x_sigma_raw_sha256",
    "input_hashes",
    "target_tokens",
    "spatial_shape",
    "same_x_sigma_object_for_all_counterfactuals",
    "all_rank_consensus",
    "model_parameters_updated",
    "optimizer_created",
    "state_digest",
}
_MATERIALIZATION_FORWARD_FIELDS = {
    "schema_version",
    "provider_abi",
    "official_provider",
    "record_id",
    "condition_role",
    "condition_utf8_sha256",
    "shared_state_digest",
    "runtime_digest",
    "sigma_index",
    "sigma_float32_be_hex",
    "timestep",
    "output_stage",
    "official_frozen_native_only",
    "model_eval",
    "torch_inference_mode",
    "calibrator_peft_adapter_present",
    "frozen_effective_adapter_enabled",
    "frozen_effective_typed_patch_role_enabled",
    "base_compute_dtype",
    "autocast_dtype",
    "observer_output_dtype",
    "observer_output_detached",
    "observer_output_contiguous",
    "same_state_input_objects_reused",
    "same_state_input_bytes_unchanged",
    "all_rank_consensus",
    "post_head_velocity_raw_sha256",
    "model_parameters_updated",
    "optimizer_created",
    "forward_digest",
}


@dataclass(frozen=True)
class _MaterializationRunV1:
    binding: Mapping[str, Any]
    receipt: Mapping[str, Any]
    record_receipts: Mapping[str, Mapping[str, Any]]
    record_refs: Mapping[str, Mapping[str, Any]]


def _validate_materialization_runtime_identity(value: Any) -> Mapping[str, Any]:
    row = _closed(value, _MATERIALIZATION_RUNTIME_FIELDS, "materialization runtime")
    _verify_seal(row, "runtime_digest", "materialization runtime")
    _require(row["schema_version"] == MATERIALIZATION_RUNTIME_SCHEMA, "materialization runtime schema differs")
    for field in ("bernini_revision", "veomni_revision"):
        _require(
            type(row[field]) is str and _REVISION.fullmatch(row[field]) is not None,
            f"materialization runtime {field} differs",
        )
    for field in (
        "official_checkpoint_tree_sha256",
        "transformer_config_sha256",
        "sigma_table_sha256",
        "psiout_protocol_sha256",
        "official_provider_source_sha256",
    ):
        _sha(row[field], f"materialization runtime {field}")
    _require(
        row["sigma_table_sha256"] == PINNED_SIGMA_TABLE_SHA256,
        "materialization runtime sigma table differs",
    )
    _require(
        row["psiout_protocol_sha256"] == PINNED_PSIOUT_PROTOCOL_SHA256,
        "materialization runtime PsiOut protocol differs",
    )
    provider_source = Path(__file__).resolve(strict=True).with_name(
        "full30_action_psiout_materializer_v1.py"
    )
    _require(provider_source.is_file() and not provider_source.is_symlink(), "official materializer source is unavailable")
    _require(
        row["official_provider_source_sha256"]
        == file_sha256(provider_source)
        == PINNED_MATERIALIZER_SOURCE_SHA256,
        "materialization runtime provider source differs from the physical materializer",
    )
    _require(
        row["official_provider_abi"] == MATERIALIZATION_PROVIDER_ABI,
        "materialization runtime provider ABI differs",
    )
    compute = _closed(
        row["compute_contract"],
        _MATERIALIZATION_COMPUTE_FIELDS,
        "materialization runtime compute contract",
    )
    expected_compute = {
        "schema_version": MATERIALIZATION_COMPUTE_SCHEMA,
        "model_eval": True,
        "torch_inference_mode": True,
        "official_frozen_native_only": True,
        "calibrator_peft_adapter_present": False,
        "frozen_effective_adapter_enabled": False,
        "frozen_effective_typed_patch_role_enabled": False,
        "base_compute_dtype": "torch.bfloat16",
        "autocast_dtype": "torch.bfloat16",
        "observer_output_dtype": "torch.float32",
        "observer_output_stage": "post-final-norm-proj-out-target-velocity",
        "observer_output_detached": True,
        "observer_output_contiguous": True,
        "same_state_counterfactual": True,
        "branch_and_noop_share_input_state": True,
        "world_size": 4,
        "dp_size": 1,
        "sp_size": 4,
        "sp_order_contract": "official-world4-sp4-rank-order-v1",
        "all_rank_consensus": True,
    }
    _require(compute == expected_compute, "materialization runtime compute contract differs")
    _require(
        row["compute_contract_digest"] == object_sha256(compute),
        "materialization runtime compute contract digest differs",
    )
    _require(row["frame_count"] == 81, "materialization runtime frame count differs")
    _require(_number(row["fps"], "materialization runtime fps") == 25.0, "materialization runtime fps differs")
    _require(row["sampler_steps"] == 40, "materialization runtime sampler steps differ")
    return row


def _validate_materialization_sigma_authority(value: Any) -> Mapping[str, Any]:
    row = _closed(
        value,
        _MATERIALIZATION_SIGMA_AUTHORITY_FIELDS,
        "materialization sigma authority",
    )
    _verify_seal(row, "sigma_authority_digest", "materialization sigma authority")
    _require(
        row["schema_version"] == _MATERIALIZATION_SIGMA_AUTHORITY_SCHEMA,
        "materialization sigma authority schema differs",
    )
    _require(
        row["sigma_authority_digest"] == _MATERIALIZATION_SIGMA_AUTHORITY_DIGEST,
        "materialization sigma authority is not the frozen schedule receipt",
    )
    _require(
        row["schedule_sha256"] == PINNED_SIGMA_TABLE_SHA256,
        "materialization sigma schedule differs",
    )
    _require(
        row["materialized_sigma_indices"] == list(SIGMA_INDICES),
        "materialization sigma-index closure differs",
    )
    rows = row["materialized_rows"]
    if type(rows) is not list or len(rows) != len(SIGMA_INDICES):
        fail("materialization sigma rows do not close six sigmas")
    for ordinal, sigma_index in enumerate(SIGMA_INDICES):
        item = _closed(
            rows[ordinal],
            _MATERIALIZATION_SIGMA_ROW_FIELDS,
            f"materialization sigma rows[{ordinal}]",
        )
        sigma_hex, timestep = _SIGMA_RUNTIME_BINDINGS[sigma_index]
        _require(
            item
            == {
                "sigma_index": sigma_index,
                "timestep": timestep,
                "sigma_float32_be_hex": sigma_hex,
            },
            f"materialization sigma rows[{ordinal}] differs",
        )
    return row


def _validate_materialization_helper_sources(value: Any) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or len(value) != len(_REQUIRED_HELPER_MODULES):
        fail("materialization official helper-source closure differs")
    rows: list[Mapping[str, Any]] = []
    seen_paths: set[Path] = set()
    for ordinal, module in enumerate(_REQUIRED_HELPER_MODULES):
        label = f"materialization official_helper_sources[{ordinal}]"
        row = _closed(value[ordinal], _MATERIALIZATION_HELPER_SOURCE_FIELDS, label)
        _require(row["module"] == module, f"{label}.module/order differs")
        path = _verify_file(row["path"], row["file_sha256"], label)
        _require(path.name == f"{module}.py", f"{label} physical filename differs")
        _require(path not in seen_paths, "materialization helper source path is reused")
        seen_paths.add(path)
        rows.append(row)
    return tuple(rows)


def _validate_materialization_output_policy(value: Any) -> Mapping[str, Any]:
    row = _closed(
        value,
        _MATERIALIZATION_OUTPUT_POLICY_FIELDS,
        "materialization output policy",
    )
    expected = {
        "schema_version": MATERIALIZATION_OUTPUT_POLICY_SCHEMA,
        "create_only": True,
        "container_mode_octal": "0600",
        "generated_rgb_decoded": False,
        "generated_rgb_used_as_model_input": False,
        "generated_rgb_used_as_regression_target": False,
        "generated_latent_used_as_absolute_regression_target": False,
        "model_parameters_updated": False,
        "optimizer_created": False,
        "persisted_tensor_role": (
            "detached-post-head-psiout-or-same-mode-amplitude-evidence-only"
        ),
    }
    _require(row == expected, "materialization output policy differs")
    return row


def _validate_materialization_plan_authority(
    value: Any,
    *,
    run: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    plan = _closed(value, _MATERIALIZATION_PLAN_FIELDS, "materialization plan authority")
    _verify_seal(plan, "plan_digest", "materialization plan authority")
    _require(plan["schema_version"] == _MATERIALIZATION_PLAN_SCHEMA, "materialization plan schema differs")
    _safe_id(plan["plan_id"], "materialization plan id")
    _require(plan["status"] == "SEALED_REVIEWED_PRE_OPTIMIZER", "materialization plan is not review-sealed")
    _require(plan["plan_id"] == run["plan_id"] and plan["plan_digest"] == run["plan_digest"], "materialization run/plan binding differs")

    runtime = _closed(
        plan["runtime"],
        _MATERIALIZATION_RUNTIME_PLAN_FIELDS,
        "materialization runtime plan",
    )
    _verify_seal(runtime, "runtime_plan_digest", "materialization runtime plan")
    _require(runtime["schema_version"] == _MATERIALIZATION_RUNTIME_PLAN_SCHEMA, "materialization runtime-plan schema differs")
    _require(runtime["frozen_runtime_identity"] == run["runtime_identity"], "materialization runtime identity snapshot differs")
    _require(runtime["runtime_plan_digest"] == run["runtime_plan_digest"], "materialization runtime-plan digest differs")
    _require(runtime["official_helper_sources"] == run["official_helper_sources"], "materialization helper-source snapshots differ")
    provider_source = _verify_file(
        runtime["official_provider_source_path"],
        run["runtime_identity"]["official_provider_source_sha256"],
        "materialization runtime provider source",
    )
    _require(
        provider_source
        == Path(__file__).resolve(strict=True).with_name(
            "full30_action_psiout_materializer_v1.py"
        ),
        "materialization runtime provider source path differs",
    )
    psiout_source = _verify_file(
        runtime["psiout_protocol_path"],
        run["runtime_identity"]["psiout_protocol_sha256"],
        "materialization runtime PsiOut protocol",
    )
    _require(psiout_source.name == "full30_action_learning_v1.py", "materialization PsiOut protocol path differs")
    _verify_file(
        runtime["checkpoint_content_manifest_path"],
        runtime["checkpoint_content_manifest_sha256"],
        "materialization checkpoint content manifest",
    )
    for directory_field in ("bernini_root", "veomni_root", "checkpoint_root"):
        path = Path(runtime[directory_field]) if type(runtime[directory_field]) is str else Path("")
        _require(
            type(runtime[directory_field]) is str
            and path.is_absolute()
            and path.is_dir()
            and not path.is_symlink(),
            f"materialization runtime {directory_field} is not a plain directory",
        )

    records_value = plan["records"]
    if type(records_value) is not list or not records_value:
        fail("materialization plan record closure differs")
    records: list[Mapping[str, Any]] = []
    record_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for ordinal, item in enumerate(records_value):
        label = f"materialization plan records[{ordinal}]"
        record = _closed(item, _MATERIALIZATION_PLAN_RECORD_FIELDS, label)
        _verify_seal(record, "record_digest", label)
        _require(record["schema_version"] == _MATERIALIZATION_PLAN_RECORD_SCHEMA, f"{label}.schema differs")
        record_id = _safe_id(record["record_id"], f"{label}.record_id")
        evidence_id = _safe_id(record["evidence_id"], f"{label}.evidence_id")
        _require(record_id not in record_ids, "materialization plan record ID is reused")
        _require(evidence_id not in evidence_ids, "materialization plan evidence ID is reused")
        record_ids.add(record_id)
        evidence_ids.add(evidence_id)
        _require(record["record_kind"] in ("teacher_anchor", "amplitude_calibrator"), f"{label}.record_kind differs")
        _require(record["branch"] in BRANCHES, f"{label}.branch differs")
        records.append(record)

    teachers_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    amplitudes_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record["teacher_cell_id"]), str(record["branch"]))
        target = (
            teachers_by_key
            if record["record_kind"] == "teacher_anchor"
            else amplitudes_by_key
        )
        target[key].append(record)
    for key, rows in teachers_by_key.items():
        _require(
            sorted(str(row["evidence_role"]) for row in rows)
            == ["same_event_cross_anchor", "teacher_origin"],
            f"materialization teacher population {key!r} role closure differs",
        )
    for key, rows in amplitudes_by_key.items():
        _require(key in teachers_by_key, f"materialization amplitude population {key!r} has no teacher nuisance authority")
        _require(len(rows) == 2, f"materialization amplitude population {key!r} calibrator count differs")
        _require(len({str(row["pair_id"]) for row in rows}) == 2, f"materialization amplitude population {key!r} pair is reused")
        _require(len({str(row["source_iid"]) for row in rows}) == 2, f"materialization amplitude population {key!r} source is reused")

    population = _closed(
        plan["population"],
        _MATERIALIZATION_POPULATION_FIELDS,
        "materialization population",
    )
    _verify_seal(population, "population_digest", "materialization population")
    _require(population["schema_version"] == _MATERIALIZATION_POPULATION_SCHEMA, "materialization population schema differs")
    teacher_count = sum(row["record_kind"] == "teacher_anchor" for row in records)
    expected_population = {
        "record_count": len(records),
        "teacher_record_count": teacher_count,
        "amplitude_record_count": len(records) - teacher_count,
        "teacher_cell_ids": sorted(
            {str(row["teacher_cell_id"]) for row in records},
            key=lambda item: item.encode("utf-8"),
        ),
        "record_order_sha256": object_sha256([str(row["record_id"]) for row in records]),
        "finite_closed_population": True,
        "block_probe": False,
    }
    for field, expected in expected_population.items():
        _require(population[field] == expected, f"materialization population {field} differs")
    _require(population["population_digest"] == run["population_digest"], "materialization population digest differs from run")
    _require(population["record_order_sha256"] == run["record_order_sha256"], "materialization record order differs from run")
    _require(plan["output_policy"] == run["output_policy"], "materialization plan/run output policy differs")
    _validate_materialization_output_policy(plan["output_policy"])
    return plan, tuple(records)


def _validate_materialization_record_basics(
    value: Any,
    *,
    reference: Mapping[str, Any],
    run: Mapping[str, Any],
    plan_record: Mapping[str, Any],
    ordinal: int,
) -> Mapping[str, Any]:
    label = f"materialization record receipt[{ordinal}]"
    row = _closed(value, _MATERIALIZATION_RECORD_FIELDS, label)
    _verify_seal(row, "record_receipt_digest", label)
    _require(row["schema_version"] == MATERIALIZATION_RECORD_RECEIPT_SCHEMA, f"{label}.schema differs")
    expected = {
        "plan_id": run["plan_id"],
        "plan_digest": run["plan_digest"],
        "runtime_digest": run["runtime_identity"]["runtime_digest"],
        "provider_abi": MATERIALIZATION_PROVIDER_ABI,
        "official_provider": True,
        "test_only": False,
        "record_ordinal": ordinal,
        "record_id": plan_record["record_id"],
        "record_digest": plan_record["record_digest"],
        "record_kind": plan_record["record_kind"],
        "evidence_id": plan_record["evidence_id"],
        "evidence_role": plan_record["evidence_role"],
        "teacher_cell_id": plan_record["teacher_cell_id"],
        "branch": plan_record["branch"],
        "record_authority": plan_record,
        "record_conditions": plan_record["conditions"],
        "review_digest": plan_record["review"]["review_digest"],
        "reviewed_media_sha256": plan_record["reviewed_media"]["file_sha256"],
        "target_clean_latent_raw_sha256": plan_record["target_clean_latent"]["tensor_raw_sha256"],
        "target_clean_latent_authority_digest": (
            None
            if plan_record["target_clean_latent_authority"] is None
            else plan_record["target_clean_latent_authority"]["digest"]
        ),
        "source_clean_latent_raw_sha256": (
            None
            if plan_record["source_clean_latent"] is None
            else plan_record["source_clean_latent"]["tensor_raw_sha256"]
        ),
        "source_posterior_index0_sha256": plan_record["source_posterior_index0_sha256"],
        "noise_seed": plan_record["noise"]["seed"],
        "noise_raw_sha256": plan_record["noise"]["artifact"]["tensor_raw_sha256"],
        "sigma_authority_digest": run["sigma_authority"]["sigma_authority_digest"],
        "generated_rgb_decoded": False,
        "generated_rgb_used_as_model_input": False,
        "generated_rgb_used_as_regression_target": False,
        "generated_latent_used_as_absolute_regression_target": False,
        "model_parameters_updated": False,
        "optimizer_created": False,
    }
    for field, expected_value in expected.items():
        _require(row[field] == expected_value, f"{label}.{field} differs")
    _require(reference["record_id"] == row["record_id"], f"{label} run reference record differs")
    _require(reference["record_kind"] == row["record_kind"], f"{label} run reference kind differs")
    _require(reference["record_receipt_digest"] == row["record_receipt_digest"], f"{label} run reference digest differs")
    candidate = row["candidate_authority_evidence"]
    if type(candidate) is not dict:
        fail(f"{label}.candidate_authority_evidence must be an object")
    _verify_seal(candidate, "evidence_digest", f"{label}.candidate_authority_evidence")
    _require(reference["candidate_evidence_digest"] == candidate["evidence_digest"], f"{label} candidate digest differs from run reference")
    return row


def _load_materialization_run_v1(value: Any) -> _MaterializationRunV1:
    binding = _closed(
        value,
        _MATERIALIZATION_RUN_BINDING_FIELDS,
        "materialization_run_receipt",
    )
    _verify_seal(binding, "binding_digest", "materialization_run_receipt")
    _require(
        binding["schema_version"] == MATERIALIZATION_RUN_BINDING_SCHEMA,
        "materialization run binding schema differs",
    )
    _sha(binding["run_digest"], "materialization run binding digest")
    _path, run = _load_strict_json_receipt(
        binding["path"],
        binding["file_sha256"],
        label="materialization run receipt",
        maximum_bytes=MATERIALIZATION_RUN_MAX_BYTES,
    )
    run = _closed(run, _MATERIALIZATION_RUN_FIELDS, "materialization run receipt")
    _verify_seal(run, "run_digest", "materialization run receipt")
    _require(run["schema_version"] == MATERIALIZATION_RUN_RECEIPT_SCHEMA, "materialization run schema differs")
    _require(run["run_digest"] == binding["run_digest"], "materialization run binding digest differs")
    _safe_id(run["plan_id"], "materialization run plan_id")
    for field in (
        "plan_digest",
        "population_digest",
        "record_order_sha256",
        "runtime_plan_digest",
        "computation_digest",
    ):
        _sha(run[field], f"materialization run {field}")
    expected_run = {
        "provider_abi": MATERIALIZATION_PROVIDER_ABI,
        "official_provider": True,
        "test_only": False,
        "world_size": 4,
        "dp_size": 1,
        "sp_size": 4,
        "sigma_indices": list(SIGMA_INDICES),
        "generated_rgb_decoded": False,
        "generated_rgb_used_as_model_input": False,
        "generated_rgb_used_as_regression_target": False,
        "generated_latent_used_as_absolute_regression_target": False,
        "model_parameters_updated": False,
        "optimizer_created": False,
    }
    for field, expected in expected_run.items():
        _require(run[field] == expected, f"materialization run {field} differs")
    _validate_materialization_runtime_identity(run["runtime_identity"])
    _validate_materialization_sigma_authority(run["sigma_authority"])
    helpers = _validate_materialization_helper_sources(run["official_helper_sources"])
    _validate_materialization_output_policy(run["output_policy"])
    plan, plan_records = _validate_materialization_plan_authority(
        run["plan_authority"], run=run
    )
    _require(plan["runtime"]["official_helper_sources"] == list(helpers), "materialization helper snapshots differ")

    references = run["record_receipts"]
    if type(references) is not list or len(references) != len(plan_records):
        fail("materialization run record-reference closure differs")
    _require(run["record_count"] == len(plan_records), "materialization run record_count differs")
    record_receipts: dict[str, Mapping[str, Any]] = {}
    record_refs: dict[str, Mapping[str, Any]] = {}
    seen_paths: set[Path] = set()
    seen_file_shas: set[str] = set()
    seen_receipt_digests: set[str] = set()
    seen_candidate_digests: set[str] = set()
    for ordinal, plan_record in enumerate(plan_records):
        label = f"materialization run record_receipts[{ordinal}]"
        reference = _closed(references[ordinal], _MATERIALIZATION_RECORD_REF_FIELDS, label)
        record_id = str(plan_record["record_id"])
        _require(reference["record_id"] == record_id, f"{label}.record_id/order differs")
        _require(reference["record_kind"] == plan_record["record_kind"], f"{label}.record_kind differs")
        for field in ("file_sha256", "record_receipt_digest", "candidate_evidence_digest"):
            _sha(reference[field], f"{label}.{field}")
        receipt_path, receipt = _load_strict_json_receipt(
            reference["path"],
            reference["file_sha256"],
            label=label,
            maximum_bytes=MATERIALIZATION_RECORD_MAX_BYTES,
        )
        _require(receipt_path not in seen_paths, "materialization record receipt path is reused")
        _require(reference["file_sha256"] not in seen_file_shas, "materialization record receipt file SHA is reused")
        _require(reference["record_receipt_digest"] not in seen_receipt_digests, "materialization record receipt digest is reused")
        _require(reference["candidate_evidence_digest"] not in seen_candidate_digests, "materialization candidate evidence digest is reused")
        seen_paths.add(receipt_path)
        seen_file_shas.add(str(reference["file_sha256"]))
        seen_receipt_digests.add(str(reference["record_receipt_digest"]))
        seen_candidate_digests.add(str(reference["candidate_evidence_digest"]))
        receipt = _validate_materialization_record_basics(
            receipt,
            reference=reference,
            run=run,
            plan_record=plan_record,
            ordinal=ordinal,
        )
        _require(record_id not in record_receipts, "materialization record ID is reused")
        record_receipts[record_id] = receipt
        record_refs[record_id] = reference
    _require(
        run["record_order_sha256"] == object_sha256([str(row["record_id"]) for row in plan_records]),
        "materialization run record order SHA differs",
    )
    return _MaterializationRunV1(
        binding=binding,
        receipt=run,
        record_receipts=record_receipts,
        record_refs=record_refs,
    )


def _load_bound_json_object(
    path_value: Any,
    sha_value: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    _path, raw = _read_stable_plain_bytes(
        path_value,
        sha_value,
        label=label,
        exact_mode=None,
        maximum_bytes=16 * 1024 * 1024,
    )
    _require(raw.endswith(b"\n"), f"{label} lacks one canonical newline")
    payload = raw[:-1]
    _require(bool(payload) and not payload.endswith(b"\n"), f"{label} newline closure differs")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30ActionAuthorityError(f"cannot decode {label}") from error
    if type(value) is not dict:
        fail(f"{label} root must be an object")
    _require(canonical_json_bytes(value) == payload, f"{label} is not canonical JSON")
    return value


def _resolve_json_pointer(root: Any, pointer: Any, label: str) -> Any:
    if type(pointer) is not str or (pointer and not pointer.startswith("/")):
        fail(f"{label} must be an empty or absolute JSON pointer")
    current = root
    if not pointer:
        return current
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            _require(token in current, f"{label} token is absent: {token!r}")
            current = current[token]
        elif isinstance(current, list):
            _require(token.isdigit(), f"{label} list token differs")
            index = int(token)
            _require(0 <= index < len(current), f"{label} list index differs")
            current = current[index]
        else:
            fail(f"{label} traverses a scalar")
    return current


def _validate_fp32_artifact(value: Any, *, label: str) -> Mapping[str, Any]:
    row = _closed(value, _MATERIALIZATION_ARTIFACT_FIELDS, label)
    _require(row["schema_version"] == _MATERIALIZATION_ARTIFACT_SCHEMA, f"{label}.schema differs")
    tensor_key = _safe_id(row["tensor_key"], f"{label}.tensor_key")
    _require(row["dtype"] == "float32-le", f"{label}.dtype differs")
    shape = row["shape"]
    if (
        type(shape) is not list
        or len(shape) != 5
        or any(type(item) is not int or item <= 0 for item in shape)
        or tuple(shape[:3]) != (1, 16, 21)
        or shape[3] % 2
        or shape[4] % 2
    ):
        fail(f"{label}.shape must be [1,16,21,evenH,evenW]")
    _path, raw = _read_stable_plain_bytes(
        row["path"],
        row["file_sha256"],
        label=label,
        exact_mode=None,
        maximum_bytes=2 * 1024 * 1024 * 1024,
    )
    _sha(row["tensor_raw_sha256"], f"{label}.tensor_raw_sha256")
    _require(len(raw) >= 8, f"{label} safetensors prefix is truncated")
    header_length = struct.unpack("<Q", raw[:8])[0]
    _require(0 < header_length <= 4 * 1024 * 1024, f"{label} safetensors header length differs")
    payload_start = 8 + header_length
    _require(payload_start <= len(raw), f"{label} safetensors header is truncated")
    try:
        header = json.loads(
            raw[8:payload_start],
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30ActionAuthorityError(f"{label} safetensors header cannot be decoded") from error
    _require(type(header) is dict, f"{label} safetensors header differs")
    tensor_keys = [key for key in header if key != "__metadata__"]
    _require(tensor_keys == [tensor_key], f"{label} tensor-key closure differs")
    if "__metadata__" in header:
        metadata = header["__metadata__"]
        _require(
            type(metadata) is dict
            and all(type(key) is str and type(item) is str for key, item in metadata.items()),
            f"{label} safetensors metadata differs",
        )
    entry = _closed(header[tensor_key], {"dtype", "shape", "data_offsets"}, f"{label}.tensor")
    expected_bytes = math.prod(shape) * 4
    _require(entry["dtype"] == "F32", f"{label} safetensors dtype differs")
    _require(entry["shape"] == shape, f"{label} safetensors shape differs")
    _require(entry["data_offsets"] == [0, expected_bytes], f"{label} safetensors offsets differ")
    payload = raw[payload_start:]
    _require(len(payload) == expected_bytes, f"{label} safetensors payload/extra bytes differ")
    _require(
        hashlib.sha256(payload).hexdigest() == row["tensor_raw_sha256"],
        f"{label} tensor raw SHA-256 differs",
    )
    values = struct.unpack(f"<{math.prod(shape)}f", payload)
    _require(all(math.isfinite(item) for item in values), f"{label} tensor contains non-finite values")
    return row


def _validate_record_review_and_artifacts(
    record: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    runtime: Mapping[str, Any],
    label: str,
) -> Mapping[str, Mapping[str, Any]]:
    review_binding = _closed(
        record["review"],
        _MATERIALIZATION_REVIEW_BINDING_FIELDS,
        f"{label}.review",
    )
    _require(review_binding["schema_version"] == _MATERIALIZATION_REVIEW_BINDING_SCHEMA, f"{label}.review schema differs")
    review = _load_bound_json_object(
        review_binding["path"],
        review_binding["file_sha256"],
        label=f"{label}.review",
    )
    _verify_seal(review, "review_digest", f"{label}.review")
    _require(review["review_digest"] == review_binding["review_digest"], f"{label}.review digest differs")
    _require(review == candidate["pre_admission_blind_review"] if record["record_kind"] == "teacher_anchor" else review == candidate["pre_admission_review"], f"{label}.review candidate binding differs")

    media = _closed(record["reviewed_media"], _MATERIALIZATION_MEDIA_FIELDS, f"{label}.reviewed_media")
    _verify_file(media["path"], media["file_sha256"], f"{label}.reviewed_media")
    if record["record_kind"] == "teacher_anchor":
        _require(
            media["path"] == candidate["anchor_video_path"]
            and media["file_sha256"] == candidate["anchor_video_sha256"],
            f"{label}.reviewed_media candidate binding differs",
        )
    else:
        _require(
            media["path"] == candidate["baseline_output_path"]
            and media["file_sha256"] == candidate["baseline_output_sha256"],
            f"{label}.reviewed_media candidate binding differs",
        )

    target = _validate_fp32_artifact(record["target_clean_latent"], label=f"{label}.target_clean_latent")
    noise_row = _closed(record["noise"], _MATERIALIZATION_NOISE_AUTHORITY_FIELDS, f"{label}.noise")
    noise = _validate_fp32_artifact(noise_row["artifact"], label=f"{label}.noise.artifact")
    _require(target["shape"] == noise["shape"], f"{label} target/noise shape differs")
    _require(noise_row["generator"] == "torch-cpu-generator-manual-seed-randn-fp32-v1", f"{label}.noise generator differs")
    source: Optional[Mapping[str, Any]] = None
    if record["record_kind"] == "teacher_anchor":
        _require(record["source_clean_latent"] is None, f"{label} teacher source latent is not null")
        _require(
            record["source_posterior_index0_path"] is None
            and record["source_posterior_index0_sha256"] is None
            and record["source_posterior_tensor_key"] is None,
            f"{label} teacher source-posterior fields are not null",
        )
        latent_binding = _closed(
            record["target_clean_latent_authority"],
            _MATERIALIZATION_LATENT_AUTHORITY_FIELDS,
            f"{label}.target_clean_latent_authority",
        )
        _require(latent_binding["schema_version"] == _MATERIALIZATION_LATENT_AUTHORITY_SCHEMA, f"{label}.latent authority schema differs")
        authority = _load_bound_json_object(
            latent_binding["path"],
            latent_binding["file_sha256"],
            label=f"{label}.target_clean_latent_authority",
        )
        digest_field = latent_binding["digest_field"]
        _require(type(digest_field) is str and digest_field in authority, f"{label}.latent authority digest field differs")
        _verify_seal(authority, digest_field, f"{label}.target_clean_latent_authority")
        _require(authority[digest_field] == latent_binding["digest"], f"{label}.latent authority digest differs")
        authority_media = _resolve_json_pointer(authority, latent_binding["media_json_pointer"], f"{label}.media_json_pointer")
        authority_latent = _resolve_json_pointer(authority, latent_binding["latent_json_pointer"], f"{label}.latent_json_pointer")
        authority_checkpoint = _resolve_json_pointer(authority, latent_binding["checkpoint_tree_sha256_json_pointer"], f"{label}.checkpoint pointer")
        _require(
            isinstance(authority_media, Mapping)
            and authority_media.get("path") == media["path"]
            and authority_media.get("sha256") == media["file_sha256"],
            f"{label}.latent authority media binding differs",
        )
        _require(
            isinstance(authority_latent, Mapping)
            and authority_latent.get("path") == target["path"]
            and authority_latent.get("sha256") == target["file_sha256"]
            and authority_latent.get("tensor_key") == target["tensor_key"]
            and authority_latent.get("raw_value_sha256") == target["tensor_raw_sha256"]
            and authority_latent.get("shape") == target["shape"]
            and authority_latent.get("stored_dtype") == "torch.float32"
            and authority_latent.get("coordinate") == "bernini_normalized_clean_vae_latent"
            and authority_latent.get("native_sampler_before_vae_decode") is True
            and authority_latent.get("mp4_decode_reencode_used") is False,
            f"{label}.latent authority tensor binding differs",
        )
        _require(authority_checkpoint == runtime["official_checkpoint_tree_sha256"], f"{label}.latent checkpoint binding differs")
    else:
        _require(record["target_clean_latent_authority"] is None, f"{label} amplitude target authority is not null")
        source = _validate_fp32_artifact(record["source_clean_latent"], label=f"{label}.source_clean_latent")
        _require(source["shape"] == target["shape"], f"{label} source/target shape differs")
        _require(source["tensor_raw_sha256"] == target["tensor_raw_sha256"], f"{label} same-mode source/target latent differs")
        posterior = _verify_file(
            record["source_posterior_index0_path"],
            record["source_posterior_index0_sha256"],
            f"{label}.source_posterior_index0",
        )
        _require(posterior.name == f"{record['source_iid']}.source-posterior-index0.pt", f"{label} source posterior filename differs")
        _safe_id(record["source_posterior_tensor_key"], f"{label}.source_posterior_tensor_key")
    return MappingProxyType(
        {"target": target, "noise": noise, **({} if source is None else {"source": source})}
    )


def _validate_record_conditions(
    value: Any,
    *,
    record: Mapping[str, Any],
    label: str,
) -> Mapping[str, Mapping[str, Any]]:
    expected_roles = (
        (
            "branch",
            "noop",
            "camera_only",
            "appearance_only",
            *WRONG_CONTROL_TYPES,
        )
        if record["record_kind"] == "teacher_anchor"
        else ("branch", "noop")
    )
    if type(value) is not list or len(value) != len(expected_roles):
        fail(f"{label}.record_conditions closure differs")
    result: dict[str, Mapping[str, Any]] = {}
    instruction_shas: set[str] = set()
    for ordinal, expected_role in enumerate(expected_roles):
        condition_label = f"{label}.record_conditions[{ordinal}]"
        row = _closed(value[ordinal], _MATERIALIZATION_CONDITION_FIELDS, condition_label)
        _require(row["schema_version"] == _MATERIALIZATION_CONDITION_SCHEMA, f"{condition_label}.schema differs")
        _require(row["role"] == expected_role, f"{condition_label}.role/order differs")
        instruction = _text(row["instruction"], f"{condition_label}.instruction")
        instruction_sha = _sha(row["instruction_utf8_sha256"], f"{condition_label}.instruction_utf8_sha256")
        _require(hashlib.sha256(instruction.encode("utf-8")).hexdigest() == instruction_sha, f"{condition_label} instruction SHA differs")
        _require(instruction_sha not in instruction_shas, f"{label} condition instructions alias")
        instruction_shas.add(instruction_sha)
        authority = _load_bound_json_object(
            row["authority_path"],
            row["authority_file_sha256"],
            label=f"{condition_label}.authority",
        )
        digest_field = row["authority_digest_field"]
        _require(type(digest_field) is str and digest_field in authority, f"{condition_label}.authority digest field differs")
        _verify_seal(authority, digest_field, f"{condition_label}.authority")
        _require(authority[digest_field] == row["authority_digest"], f"{condition_label}.authority digest differs")
        target = _resolve_json_pointer(authority, row["json_pointer"], f"{condition_label}.json_pointer")
        _require(isinstance(target, Mapping), f"{condition_label}.json_pointer target differs")
        text_field = row["text_field"]
        sha_field = row["sha256_field"]
        _require(
            type(text_field) is str
            and type(sha_field) is str
            and target.get(text_field) == instruction
            and target.get(sha_field) == instruction_sha,
            f"{condition_label}.authority instruction binding differs",
        )
        if expected_role in WRONG_CONTROL_TYPES:
            _safe_id(row["control_anchor_id"], f"{condition_label}.control_anchor_id")
        else:
            _require(row["control_anchor_id"] is None, f"{condition_label}.control_anchor_id is not null")
        result[expected_role] = row
    return MappingProxyType(result)


def _validate_noise_state_forward_receipts(
    receipt: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    conditions: Mapping[str, Mapping[str, Any]],
    runtime_digest: str,
    label: str,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    noise = _closed(
        receipt["noise_replay_receipt"],
        _MATERIALIZATION_NOISE_FIELDS,
        f"{label}.noise_replay_receipt",
    )
    _verify_seal(noise, "noise_digest", f"{label}.noise_replay_receipt")
    expected_noise = {
        "schema_version": MATERIALIZATION_NOISE_RECEIPT_SCHEMA,
        "provider_abi": MATERIALIZATION_PROVIDER_ABI,
        "official_provider": True,
        "record_id": record["record_id"],
        "seed": record["noise"]["seed"],
        "generator": record["noise"]["generator"],
        "shape": record["noise"]["artifact"]["shape"],
        "artifact_raw_sha256": record["noise"]["artifact"]["tensor_raw_sha256"],
        "replayed_raw_sha256": record["noise"]["artifact"]["tensor_raw_sha256"],
        "byte_exact_replay": True,
    }
    for field, expected in expected_noise.items():
        _require(noise[field] == expected, f"{label}.noise_replay_receipt.{field} differs")

    states_value = receipt["state_receipts"]
    if type(states_value) is not list or len(states_value) != len(SIGMA_INDICES):
        fail(f"{label}.state_receipts must contain exactly six rows")
    states: list[Mapping[str, Any]] = []
    x_sigma_shas: set[str] = set()
    source_sha = (
        None
        if record["record_kind"] == "teacher_anchor"
        else artifacts["source"]["tensor_raw_sha256"]
    )
    for ordinal, sigma_index in enumerate(SIGMA_INDICES):
        state_label = f"{label}.state_receipts[{ordinal}]"
        state = _closed(states_value[ordinal], _MATERIALIZATION_STATE_FIELDS, state_label)
        _verify_seal(state, "state_digest", state_label)
        sigma_hex, timestep = _SIGMA_RUNTIME_BINDINGS[sigma_index]
        expected_state = {
            "schema_version": MATERIALIZATION_STATE_RECEIPT_SCHEMA,
            "provider_abi": MATERIALIZATION_PROVIDER_ABI,
            "official_provider": True,
            "runtime_digest": runtime_digest,
            "record_id": record["record_id"],
            "record_kind": record["record_kind"],
            "teacher_cell_id": record["teacher_cell_id"],
            "branch": record["branch"],
            "sigma_index": sigma_index,
            "sigma_float32_be_hex": sigma_hex,
            "timestep": timestep,
            "clean_raw_sha256": artifacts["target"]["tensor_raw_sha256"],
            "source_raw_sha256": source_sha,
            "noise_raw_sha256": artifacts["noise"]["tensor_raw_sha256"],
            "spatial_shape": artifacts["target"]["shape"],
            "same_x_sigma_object_for_all_counterfactuals": True,
            "all_rank_consensus": True,
            "model_parameters_updated": False,
            "optimizer_created": False,
        }
        for field, expected in expected_state.items():
            _require(state[field] == expected, f"{state_label}.{field} differs")
        _sha(state["x_sigma_raw_sha256"], f"{state_label}.x_sigma_raw_sha256")
        _require(state["x_sigma_raw_sha256"] not in x_sigma_shas, f"{label} x_sigma SHA is reused across sigmas")
        x_sigma_shas.add(str(state["x_sigma_raw_sha256"]))
        hashes = _closed(
            state["input_hashes"],
            {"noisy_latents", "rotary_embs", "target_mask", "timestep"},
            f"{state_label}.input_hashes",
        )
        for field, digest in hashes.items():
            _sha(digest, f"{state_label}.input_hashes.{field}")
        _require(type(state["target_tokens"]) is int and state["target_tokens"] > 0, f"{state_label}.target_tokens differs")
        states.append(state)

    per_sigma_roles = (
        (
            "branch",
            "branch",
            "noop",
            "noop",
            "camera_only",
            "appearance_only",
            *WRONG_CONTROL_TYPES,
        )
        if record["record_kind"] == "teacher_anchor" and record["evidence_role"] == "teacher_origin"
        else (
            ("branch", "branch", "noop", "noop", "camera_only", "appearance_only")
            if record["record_kind"] == "teacher_anchor"
            else ("branch", "noop")
        )
    )
    forwards_value = receipt["forward_receipts"]
    expected_forward_count = len(SIGMA_INDICES) * len(per_sigma_roles)
    if type(forwards_value) is not list or len(forwards_value) != expected_forward_count:
        fail(f"{label}.forward_receipts count differs")
    forwards: list[Mapping[str, Any]] = []
    cursor = 0
    for sigma_ordinal, sigma_index in enumerate(SIGMA_INDICES):
        sigma_hex, timestep = _SIGMA_RUNTIME_BINDINGS[sigma_index]
        state = states[sigma_ordinal]
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for expected_role in per_sigma_roles:
            forward_label = f"{label}.forward_receipts[{cursor}]"
            forward = _closed(
                forwards_value[cursor],
                _MATERIALIZATION_FORWARD_FIELDS,
                forward_label,
            )
            _verify_seal(forward, "forward_digest", forward_label)
            expected_forward = {
                "schema_version": MATERIALIZATION_FORWARD_RECEIPT_SCHEMA,
                "provider_abi": MATERIALIZATION_PROVIDER_ABI,
                "official_provider": True,
                "record_id": record["record_id"],
                "condition_role": expected_role,
                "condition_utf8_sha256": conditions[expected_role]["instruction_utf8_sha256"],
                "shared_state_digest": state["state_digest"],
                "runtime_digest": runtime_digest,
                "sigma_index": sigma_index,
                "sigma_float32_be_hex": sigma_hex,
                "timestep": timestep,
                "output_stage": "post-final-norm-proj-out-target-velocity",
                "official_frozen_native_only": True,
                "model_eval": True,
                "torch_inference_mode": True,
                "calibrator_peft_adapter_present": False,
                "frozen_effective_adapter_enabled": False,
                "frozen_effective_typed_patch_role_enabled": False,
                "base_compute_dtype": "torch.bfloat16",
                "autocast_dtype": "torch.bfloat16",
                "observer_output_dtype": "torch.float32",
                "observer_output_detached": True,
                "observer_output_contiguous": True,
                "same_state_input_objects_reused": True,
                "same_state_input_bytes_unchanged": True,
                "all_rank_consensus": True,
                "model_parameters_updated": False,
                "optimizer_created": False,
            }
            for field, expected in expected_forward.items():
                _require(forward[field] == expected, f"{forward_label}.{field} differs")
            _sha(forward["post_head_velocity_raw_sha256"], f"{forward_label}.post_head_velocity_raw_sha256")
            grouped[expected_role].append(forward)
            forwards.append(forward)
            cursor += 1
        for duplicate_role in ("branch", "noop"):
            if len(grouped[duplicate_role]) == 2:
                _require(
                    grouped[duplicate_role][0]["post_head_velocity_raw_sha256"]
                    == grouped[duplicate_role][1]["post_head_velocity_raw_sha256"],
                    f"{label} duplicate {duplicate_role} forward bytes differ",
                )
    return tuple(states), tuple(forwards)


def _teacher_noise_seed(teacher_cell_id: str, branch: str) -> int:
    payload = (
        b"full30-teacher-noise-v1\x00"
        + teacher_cell_id.encode("utf-8")
        + b"\x00"
        + branch.encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _candidate_without_provenance(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    unsigned = dict(evidence)
    unsigned.pop("evidence_digest", None)
    for field in _MATERIALIZATION_PROVENANCE_FIELDS:
        _require(field in unsigned, f"materialization provenance field is absent: {field}")
        del unsigned[field]
    return MappingProxyType(seal_record(unsigned, "evidence_digest"))


def _validate_materialization_record_for_evidence(
    evidence: Mapping[str, Any],
    *,
    run_authority: _MaterializationRunV1,
    container_slices: Mapping[str, Mapping[str, str]],
    label: str,
) -> Mapping[str, Any]:
    _require(
        evidence["materialization_run_digest"]
        == run_authority.receipt["run_digest"],
        f"{label} materialization run digest differs",
    )
    receipt_path = _plain_file(
        evidence["materialization_record_receipt_path"],
        f"{label}.materialization_record_receipt",
    )
    matches = [
        (record_id, reference)
        for record_id, reference in run_authority.record_refs.items()
        if Path(str(reference["path"])).resolve(strict=True) == receipt_path
    ]
    _require(len(matches) == 1, f"{label} materialization record is absent or ambiguous in run")
    record_id, reference = matches[0]
    expected_provenance = {
        "materialization_record_receipt_path": reference["path"],
        "materialization_record_receipt_sha256": reference["file_sha256"],
        "materialization_record_receipt_digest": reference["record_receipt_digest"],
        "materialization_run_digest": run_authority.receipt["run_digest"],
    }
    for field, expected in expected_provenance.items():
        _require(evidence[field] == expected, f"{label}.{field} differs")
    receipt = run_authority.record_receipts[record_id]
    candidate = _candidate_without_provenance(evidence)
    _require(
        dict(candidate) == receipt["candidate_authority_evidence"],
        f"{label} differs from materializer base candidate",
    )
    _require(
        candidate["evidence_digest"] == reference["candidate_evidence_digest"],
        f"{label} candidate digest differs",
    )
    record = receipt["record_authority"]
    expected_identity = {
        "evidence_id": evidence["evidence_id"],
        "evidence_role": evidence["evidence_role"],
        "teacher_cell_id": evidence["teacher_cell_id"],
        "branch": evidence["branch"],
    }
    for field, expected in expected_identity.items():
        _require(record[field] == expected, f"{label} record authority {field} differs")

    artifacts = _validate_record_review_and_artifacts(
        record,
        candidate=candidate,
        runtime=run_authority.receipt["runtime_identity"],
        label=f"{label}.materialization_record",
    )
    conditions = _validate_record_conditions(
        receipt["record_conditions"],
        record=record,
        label=f"{label}.materialization_record",
    )
    _require(receipt["record_conditions"] == record["conditions"], f"{label} condition snapshot differs")
    if record["record_kind"] == "teacher_anchor":
        _require(receipt["source_clean_latent_raw_sha256"] is None, f"{label} teacher source latent receipt is not null")
        _require(receipt["source_posterior_index0_sha256"] is None, f"{label} teacher source posterior receipt is not null")
        _require(
            receipt["noise_seed"]
            == _teacher_noise_seed(str(record["teacher_cell_id"]), str(record["branch"])),
            f"{label} teacher noise seed differs",
        )
    else:
        _require(
            receipt["source_posterior_index0_sha256"]
            == candidate["source_posterior_index0_sha256"],
            f"{label} source posterior differs from candidate",
        )
        _require(
            receipt["noise_seed"] == candidate["calibrator_noise_seed"]
            and receipt["noise_raw_sha256"] == candidate["calibrator_noise_sha256"],
            f"{label} calibrator noise binding differs",
        )
        _require(
            conditions["branch"]["instruction_utf8_sha256"]
            == candidate["instruction_utf8_sha256"],
            f"{label} branch condition differs from candidate",
        )
    states, _forwards = _validate_noise_state_forward_receipts(
        receipt,
        record=record,
        artifacts=artifacts,
        conditions=conditions,
        runtime_digest=str(run_authority.receipt["runtime_identity"]["runtime_digest"]),
        label=f"{label}.materialization_record",
    )

    bindings_value = receipt["container_bindings"]
    expected_kinds = (
        ("psiout", "nuisance")
        if record["record_kind"] == "teacher_anchor"
        else ("amplitude",)
    )
    if type(bindings_value) is not list or len(bindings_value) != len(expected_kinds):
        fail(f"{label} materialization container closure differs")
    for ordinal, expected_kind in enumerate(expected_kinds):
        binding_label = f"{label}.materialization_record.container_bindings[{ordinal}]"
        binding = _closed(
            bindings_value[ordinal],
            _MATERIALIZATION_CONTAINER_BINDING_FIELDS,
            binding_label,
        )
        _require(binding["container_kind"] == expected_kind, f"{binding_label}.kind/order differs")
        if expected_kind == "psiout":
            expected_path = candidate["psiout_sidecar_path"]
            expected_sha = candidate["psiout_sidecar_sha256"]
        elif expected_kind == "nuisance":
            expected_path = candidate["nuisance_packet_path"]
            expected_sha = candidate["nuisance_packet_sha256"]
        else:
            expected_path = candidate["amplitude_container_path"]
            expected_sha = candidate["amplitude_container_sha256"]
        _require(
            binding["path"] == expected_path
            and binding["file_sha256"] == expected_sha,
            f"{binding_label} path/SHA differs from candidate",
        )
        slices = binding["slice_sha256"]
        _require(
            type(slices) is dict
            and slices == dict(container_slices[expected_kind]),
            f"{binding_label} slice closure differs from reopened container",
        )

    metrics_value = receipt["sigma_metrics"]
    if type(metrics_value) is not list or len(metrics_value) != len(SIGMA_INDICES):
        fail(f"{label} materialization sigma-metric closure differs")
    for ordinal, sigma_index in enumerate(SIGMA_INDICES):
        metric_label = f"{label}.materialization_record.sigma_metrics[{ordinal}]"
        if record["record_kind"] == "amplitude_calibrator":
            fields = _MATERIALIZATION_AMPLITUDE_METRIC_FIELDS
        elif record["evidence_role"] == "teacher_origin":
            fields = _MATERIALIZATION_TEACHER_ORIGIN_METRIC_FIELDS
        else:
            fields = _MATERIALIZATION_TEACHER_CROSS_METRIC_FIELDS
        metric = _closed(metrics_value[ordinal], fields, metric_label)
        _require(metric["sigma_index"] == sigma_index, f"{metric_label}.sigma_index differs")
        _require(metric["state_digest"] == states[ordinal]["state_digest"], f"{metric_label}.state_digest differs")
        prefix = f"sigma_{sigma_index:02d}:"
        if record["record_kind"] == "amplitude_calibrator":
            _require(metric["projected_slice_sha256"] == container_slices["amplitude"][prefix + "projected_raw"], f"{metric_label}.projected slice differs")
            _number(metric["amplitude_norm"], f"{metric_label}.amplitude_norm")
            _sha(metric["teacher_nuisance_camera_sha256"], f"{metric_label}.teacher nuisance camera")
            _sha(metric["teacher_nuisance_appearance_sha256"], f"{metric_label}.teacher nuisance appearance")
        else:
            _require(metric["projected_unit_sha256"] == container_slices["psiout"][prefix + "projected_unit"], f"{metric_label}.projected unit differs")
            _require(metric["camera_unit_sha256"] == container_slices["nuisance"][prefix + "camera_unit"], f"{metric_label}.camera unit differs")
            _require(metric["appearance_unit_sha256"] == container_slices["nuisance"][prefix + "appearance_unit"], f"{metric_label}.appearance unit differs")
            if record["evidence_role"] == "teacher_origin":
                for field, kind in (
                    ("projected_raw_sha256", "projected_raw"),
                    ("duplicate_forward_first_sha256", "duplicate_forward_first"),
                    ("duplicate_forward_second_sha256", "duplicate_forward_second"),
                    ("noop_forward_first_sha256", "noop_forward_first"),
                    ("noop_forward_second_sha256", "noop_forward_second"),
                ):
                    _require(metric[field] == container_slices["psiout"][prefix + kind], f"{metric_label}.{field} differs")
                _require(metric["duplicate_forward_bytes_identical"] is True, f"{metric_label} duplicate flag differs")
                controls = metric["wrong_controls"]
                if type(controls) is not list or len(controls) != len(WRONG_CONTROL_TYPES):
                    fail(f"{metric_label}.wrong_controls closure differs")
                for control_ordinal, control_type in enumerate(WRONG_CONTROL_TYPES):
                    control = _closed(controls[control_ordinal], _WRONG_CONTROL_FIELDS, f"{metric_label}.wrong_controls[{control_ordinal}]")
                    _require(control["control_type"] == control_type, f"{metric_label} wrong-control order differs")
                    _require(
                        control["wrong_projected_slice_sha256"]
                        == container_slices["psiout"][prefix + f"{control_type}_projected_unit"],
                        f"{metric_label} wrong-control slice differs",
                    )
    return receipt


_TensorSlice = Tuple[bytes, Tuple[float, ...], str]


def _tensor_name(sigma_index: int, kind: str) -> str:
    return f"sigma_{sigma_index:02d}:{kind}"


def _expected_tensor_names(container_kind: str, evidence_role: str) -> tuple[str, ...]:
    if container_kind == "psiout":
        kinds = (
            _ORIGIN_PSIOUT_TENSOR_KINDS
            if evidence_role == "teacher_origin"
            else _CROSS_PSIOUT_TENSOR_KINDS
        )
    elif container_kind == "nuisance":
        kinds = _NUISANCE_TENSOR_KINDS
    else:
        fail(f"unknown tensor container kind: {container_kind!r}")
    return tuple(_tensor_name(sigma, kind) for sigma in SIGMA_INDICES for kind in kinds)


def _read_strict_tensor_file(value: Any, expected_sha256: Any, label: str) -> bytes:
    """Read one immutable tensor container without following a final symlink."""

    if type(value) is not str:
        fail(f"{label} path must be text")
    path = Path(value)
    if not path.is_absolute():
        fail(f"{label} path must be absolute")
    expected = _sha(expected_sha256, f"{label} SHA-256")
    try:
        path_metadata = path.lstat()
    except OSError as error:
        raise Full30ActionAuthorityError(f"{label} is unavailable: {path}") from error
    if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISREG(path_metadata.st_mode):
        fail(f"{label} must be a plain non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Full30ActionAuthorityError(f"{label} cannot be opened safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        _require(
            stat.S_IMODE(before.st_mode) == TENSOR_CONTAINER_MODE,
            f"{label} mode must be exactly {TENSOR_CONTAINER_MODE:#o}",
        )
        _require(
            0 < before.st_size <= TENSOR_CONTAINER_MAX_FILE_BYTES,
            f"{label} size is outside the bounded tensor-container range",
        )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            _require(bool(block), f"{label} was truncated while reading")
            chunks.append(block)
            remaining -= len(block)
        _require(os.read(descriptor, 1) == b"", f"{label} grew while reading")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    _require(
        all(getattr(before, field) == getattr(after, field) for field in stable_fields),
        f"{label} metadata changed while reading",
    )
    raw = b"".join(chunks)
    _require(len(raw) == before.st_size, f"{label} byte length changed while reading")
    _require(hashlib.sha256(raw).hexdigest() == expected, f"{label} file SHA-256 differs")
    return raw


_TENSOR_CONTAINER_FIELDS = {
    "schema_version",
    "container_kind",
    "evidence_id",
    "evidence_role",
    "teacher_cell_id",
    "branch",
    "dtype",
    "shape",
    "sigma_indices",
    "layout",
    "tensor_count",
    "payload_bytes",
    "entries",
}
_TENSOR_ENTRY_FIELDS = {"name", "dtype", "shape", "offset", "length", "sha256"}


def _validate_tensor_container(
    path_value: Any,
    sha_value: Any,
    *,
    container_kind: str,
    evidence_id: str,
    evidence_role: str,
    teacher_cell_id: str,
    branch: str,
    label: str,
) -> Mapping[str, _TensorSlice]:
    raw = _read_strict_tensor_file(path_value, sha_value, label)
    prefix_bytes = len(TENSOR_CONTAINER_MAGIC) + 4
    _require(len(raw) >= prefix_bytes, f"{label} tensor-container prefix is truncated")
    _require(raw.startswith(TENSOR_CONTAINER_MAGIC), f"{label} tensor-container magic differs")
    header_length = struct.unpack(
        ">I", raw[len(TENSOR_CONTAINER_MAGIC) : prefix_bytes]
    )[0]
    _require(
        0 < header_length <= TENSOR_CONTAINER_MAX_HEADER_BYTES,
        f"{label} tensor-container header length differs",
    )
    payload_start = prefix_bytes + header_length
    _require(payload_start <= len(raw), f"{label} tensor-container header is truncated")
    header_bytes = raw[prefix_bytes:payload_start]
    try:
        header = json.loads(
            header_bytes,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30ActionAuthorityError(f"{label} tensor-container header cannot be decoded") from error
    header = _closed(header, _TENSOR_CONTAINER_FIELDS, f"{label}.header")
    _require(
        canonical_json_bytes(header) == header_bytes,
        f"{label} tensor-container header is not canonical JSON",
    )
    _require(header["schema_version"] == TENSOR_CONTAINER_SCHEMA, f"{label} schema differs")
    _require(header["container_kind"] == container_kind, f"{label} container kind differs")
    _require(header["evidence_id"] == evidence_id, f"{label} evidence binding differs")
    _require(header["evidence_role"] == evidence_role, f"{label} evidence role differs")
    _require(header["teacher_cell_id"] == teacher_cell_id, f"{label} teacher cell differs")
    _require(header["branch"] == branch, f"{label} branch differs")
    _require(header["dtype"] == TENSOR_DTYPE, f"{label} dtype differs")
    _require(header["shape"] == list(TENSOR_SHAPE), f"{label} tensor shape differs")
    _require(header["sigma_indices"] == list(SIGMA_INDICES), f"{label} sigma table differs")
    _require(header["layout"] == TENSOR_CONTAINER_LAYOUT, f"{label} layout differs")

    expected_names = _expected_tensor_names(container_kind, evidence_role)
    _require(header["tensor_count"] == len(expected_names), f"{label} tensor count differs")
    expected_payload_bytes = len(expected_names) * TENSOR_SLICE_BYTES
    _require(
        header["payload_bytes"] == expected_payload_bytes,
        f"{label} declared payload length differs",
    )
    payload = raw[payload_start:]
    _require(
        len(payload) == expected_payload_bytes,
        f"{label} payload length/extra bytes differ",
    )
    entries = header["entries"]
    if type(entries) is not list or len(entries) != len(expected_names):
        fail(f"{label} entries do not close the canonical tensor set")
    result: dict[str, _TensorSlice] = {}
    for ordinal, expected_name in enumerate(expected_names):
        entry_label = f"{label}.entries[{ordinal}]"
        entry = _closed(entries[ordinal], _TENSOR_ENTRY_FIELDS, entry_label)
        _require(entry["name"] == expected_name, f"{entry_label}.name/order differs")
        _require(entry["dtype"] == TENSOR_DTYPE, f"{entry_label}.dtype differs")
        _require(entry["shape"] == list(TENSOR_SHAPE), f"{entry_label}.shape differs")
        expected_offset = ordinal * TENSOR_SLICE_BYTES
        _require(
            type(entry["offset"]) is int and entry["offset"] == expected_offset,
            f"{entry_label}.offset is not canonical",
        )
        _require(
            type(entry["length"]) is int and entry["length"] == TENSOR_SLICE_BYTES,
            f"{entry_label}.length differs",
        )
        declared_sha = _sha(entry["sha256"], f"{entry_label}.sha256")
        tensor_bytes = payload[expected_offset : expected_offset + TENSOR_SLICE_BYTES]
        actual_sha = hashlib.sha256(tensor_bytes).hexdigest()
        _require(actual_sha == declared_sha, f"{entry_label} tensor byte SHA-256 differs")
        values = struct.unpack(f"<{TENSOR_ELEMENTS}f", tensor_bytes)
        _require(
            all(math.isfinite(value) for value in values),
            f"{entry_label} contains a non-finite FP32 value",
        )
        result[expected_name] = (tensor_bytes, values, actual_sha)
    return result


def _vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(float(value) * float(value) for value in values))


def _vector_cosine(left: Sequence[float], right: Sequence[float], label: str) -> float:
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    _require(left_norm > 0.0 and right_norm > 0.0, f"{label} has a zero-norm vector")
    return math.fsum(float(a) * float(b) for a, b in zip(left, right)) / (
        left_norm * right_norm
    )


def _require_unit(values: Sequence[float], label: str) -> None:
    _require(
        math.isclose(_vector_norm(values), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6),
        f"{label} is not a unit FP32 tensor",
    )


def _require_metric_matches(claimed: float, actual: float, label: str) -> None:
    _require(
        math.isclose(claimed, actual, rel_tol=1.0e-12, abs_tol=1.0e-12),
        f"{label} does not match tensor-container bytes",
    )


def _compatibility_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["event_id"]),
        str(row["actor_kind"]),
        str(row["q0_id"]),
    )


def _assignment_sort_key(split: str, source: Mapping[str, Any]) -> bytes:
    payload = (
        b"full30-action-authority-assignment-v1\x00"
        + split.encode("ascii")
        + b"\x00"
        + str(source["source_iid"]).encode("ascii")
        + b"\x00"
        + str(source["source_video_sha256"]).encode("ascii")
    )
    return hashlib.sha256(payload).digest()


def deterministic_teacher_assignment_v1(
    sources: Sequence[Mapping[str, Any]],
    teacher_origins: Sequence[Mapping[str, Any]],
) -> Mapping[str, str]:
    """Compute the only admitted source-to-teacher mapping."""

    source_by_split_key: dict[tuple[str, tuple[str, str, str]], list[Mapping[str, Any]]] = defaultdict(list)
    for source in sources:
        split = str(source["analysis_split"])
        source_by_split_key[(split, _compatibility_key(source))].append(source)

    origins_by_split_key: dict[tuple[str, tuple[str, str, str]], list[Mapping[str, Any]]] = defaultdict(list)
    for origin in teacher_origins:
        split = str(origin["analysis_split"])
        origins_by_split_key[(split, _compatibility_key(origin))].append(origin)

    result: dict[str, str] = {}
    for split in SPLITS:
        origin_split = "fit" if split == "heldout" else split
        keys = {
            key
            for observed_split, key in source_by_split_key
            if observed_split == split
        }
        origin_keys = {
            key
            for observed_split, key in origins_by_split_key
            if observed_split == origin_split
        }
        _require(keys == origin_keys, f"{split} actor/q0/event compatibility buckets differ")
        for key in sorted(keys):
            bucket_sources = sorted(
                source_by_split_key[(split, key)],
                key=lambda row: (_assignment_sort_key(split, row), str(row["source_iid"])),
            )
            bucket_origins = sorted(
                origins_by_split_key[(origin_split, key)],
                key=lambda row: str(row["teacher_cell_id"]).encode("utf-8"),
            )
            capacity = ASSIGNMENT_CAPACITY[split]
            _require(
                len(bucket_sources) == len(bucket_origins) * capacity,
                f"{split} compatibility bucket capacity differs: {key!r}",
            )
            slots = [
                origin
                for _ in range(capacity)
                for origin in bucket_origins
            ]
            for source, origin in zip(bucket_sources, slots):
                source_iid = str(source["source_iid"])
                _require(source_iid not in result, "source assigned more than once")
                result[source_iid] = str(origin["teacher_cell_id"])
    _require(len(result) == len(sources), "deterministic teacher assignment is incomplete")
    return result


_TEACHER_FIELDS = {
    "schema_version",
    "teacher_cell_id",
    "analysis_split",
    "origin_iid",
    "origin_source_path",
    "origin_source_sha256",
    "origin_group_id",
    "event_id",
    "actor_kind",
    "q0_id",
    "actor_id",
    "scene_id",
    "origin_digest",
}


def _validate_teacher_origins(value: Any) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or len(value) != 16:
        fail("teacher_origins must contain exactly 16 rows")
    rows: list[Mapping[str, Any]] = []
    cell_ids: set[str] = set()
    identity_bindings: dict[str, tuple[str, str, str, str, str, str, str]] = {}
    iid_splits: dict[str, str] = {}
    sha_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    for index, item in enumerate(value):
        label = f"teacher_origins[{index}]"
        row = _closed(item, _TEACHER_FIELDS, label)
        _verify_seal(row, "origin_digest", label)
        _require(row["schema_version"] == TEACHER_ORIGIN_SCHEMA, f"{label} schema differs")
        cell_id = _safe_id(row["teacher_cell_id"], f"{label}.teacher_cell_id")
        _require(cell_id not in cell_ids, "teacher cell id is duplicated")
        cell_ids.add(cell_id)
        split = row["analysis_split"]
        _require(split in EXPECTED_TEACHER_COUNTS, f"{label}.analysis_split differs")
        origin_iid = _iid(row["origin_iid"], f"{label}.origin_iid")
        origin_sha = _sha(row["origin_source_sha256"], f"{label}.origin_source_sha256")
        _verify_file(row["origin_source_path"], origin_sha, f"{label}.origin_source")
        values = (
            origin_sha,
            _safe_id(row["origin_group_id"], f"{label}.origin_group_id"),
            _safe_id(row["event_id"], f"{label}.event_id"),
            _safe_id(row["actor_kind"], f"{label}.actor_kind"),
            _safe_id(row["q0_id"], f"{label}.q0_id"),
            _safe_id(row["actor_id"], f"{label}.actor_id"),
            _safe_id(row["scene_id"], f"{label}.scene_id"),
        )
        origin_group = str(values[1])
        _require(
            iid_splits.setdefault(origin_iid, str(split)) == split,
            "teacher origin IID crosses fit/confirmation",
        )
        _require(
            sha_splits.setdefault(origin_sha, str(split)) == split,
            "teacher origin SHA-256 crosses fit/confirmation",
        )
        _require(
            group_splits.setdefault(origin_group, str(split)) == split,
            "teacher origin group crosses fit/confirmation",
        )
        if origin_iid in identity_bindings:
            _require(identity_bindings[origin_iid] == values, "teacher origin IID binding differs")
        else:
            identity_bindings[origin_iid] = values
        rows.append(row)
    _require(
        Counter(row["analysis_split"] for row in rows) == Counter(EXPECTED_TEACHER_COUNTS),
        "teacher origin split counts differ",
    )
    return tuple(rows)


_SOURCE_FIELDS = {
    "schema_version",
    "source_iid",
    "analysis_split",
    "source_group_id",
    "source_video_path",
    "source_video_sha256",
    "source_posterior_index0_path",
    "source_posterior_index0_sha256",
    "source_posterior_tensor_key",
    "posterior_index_decoded",
    "physical_index0_only",
    "synthetic_target_index1_bytes_read",
    "synthetic_target_index1_decoded",
    "synthetic_target_index1_hashed",
    "actor_id",
    "scene_id",
    "event_id",
    "actor_kind",
    "q0_id",
    "source_motion_label",
    "source_digest",
}


def _validate_sources(
    value: Any, teacher_origins: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    expected_total = sum(EXPECTED_SOURCE_COUNTS.values())
    if type(value) is not list or len(value) != expected_total:
        fail(f"sources must contain exactly {expected_total} rows")
    rows: list[Mapping[str, Any]] = []
    seen_iids: set[str] = set()
    seen_video_sha: set[str] = set()
    seen_index0_sha: set[str] = set()
    group_splits: dict[str, str] = {}
    for index, item in enumerate(value):
        label = f"sources[{index}]"
        row = _closed(item, _SOURCE_FIELDS, label)
        _verify_seal(row, "source_digest", label)
        _require(row["schema_version"] == SOURCE_SCHEMA, f"{label} schema differs")
        split = row["analysis_split"]
        _require(split in SPLITS, f"{label}.analysis_split differs")
        iid = _iid(row["source_iid"], f"{label}.source_iid")
        _require(iid not in seen_iids, "source IID is duplicated across the full domain")
        seen_iids.add(iid)
        video_sha = _sha(row["source_video_sha256"], f"{label}.source_video_sha256")
        _verify_file(row["source_video_path"], video_sha, f"{label}.source_video")
        _require(video_sha not in seen_video_sha, "source video SHA-256 is duplicated across the full domain")
        seen_video_sha.add(video_sha)
        index0_sha = _sha(
            row["source_posterior_index0_sha256"],
            f"{label}.source_posterior_index0_sha256",
        )
        index0_path = _verify_file(
            row["source_posterior_index0_path"],
            index0_sha,
            f"{label}.source_posterior_index0",
        )
        _require(
            index0_path.name == f"{iid}.source-posterior-index0.pt",
            f"{label} index0 filename differs",
        )
        _require(index0_sha not in seen_index0_sha, "source index0 SHA-256 is duplicated")
        seen_index0_sha.add(index0_sha)
        _text(row["source_posterior_tensor_key"], f"{label}.source_posterior_tensor_key")
        _require(row["posterior_index_decoded"] == 0, f"{label} posterior index is not zero")
        _require(row["physical_index0_only"] is True, f"{label} is not physical index0-only")
        for field in (
            "synthetic_target_index1_bytes_read",
            "synthetic_target_index1_decoded",
            "synthetic_target_index1_hashed",
        ):
            _require(row[field] is False, f"{label}.{field} is not false")
        group = _safe_id(row["source_group_id"], f"{label}.source_group_id")
        previous_split = group_splits.setdefault(group, str(split))
        _require(previous_split == split, "source group crosses analysis splits")
        for field in (
            "actor_id",
            "scene_id",
            "event_id",
            "actor_kind",
            "q0_id",
            "source_motion_label",
        ):
            _safe_id(row[field], f"{label}.{field}")
        rows.append(row)
    _require(
        Counter(row["analysis_split"] for row in rows) == Counter(EXPECTED_SOURCE_COUNTS),
        "source split counts differ from 64/16/8",
    )
    origin_iids = {str(row["origin_iid"]) for row in teacher_origins}
    origin_shas = {str(row["origin_source_sha256"]) for row in teacher_origins}
    origin_groups = {str(row["origin_group_id"]) for row in teacher_origins}
    _require(not (seen_iids & origin_iids), "real-source IID overlaps a teacher origin")
    _require(not (seen_video_sha & origin_shas), "real-source SHA-256 overlaps a teacher origin")
    _require(not (set(group_splits) & origin_groups), "real-source group overlaps a teacher origin")
    return tuple(rows)


_REVIEW_FIELDS = {
    "schema_version",
    "review_id",
    "pair_id",
    "source_iid",
    "source_video_sha256",
    "branch",
    "frame_count",
    "fps",
    "entire_full81_video_viewed",
    "independent_reviewer",
    "reviewer_blinded_to_teacher_cell",
    "sealed_before_pair_admission",
    "actor_kind_compatible",
    "q0_compatible",
    "owner_object_verified",
    "source_motion_verified",
    "target_event_incompatible_with_source_motion",
    "review_digest",
}


def _validate_review(
    value: Any,
    *,
    pair_id: str,
    source: Mapping[str, Any],
    branch: str,
    label: str,
) -> Mapping[str, Any]:
    row = _closed(value, _REVIEW_FIELDS, label)
    _verify_seal(row, "review_digest", label)
    _require(row["schema_version"] == REVIEW_SCHEMA, f"{label} schema differs")
    _safe_id(row["review_id"], f"{label}.review_id")
    _require(row["pair_id"] == pair_id, f"{label}.pair_id differs")
    _require(row["source_iid"] == source["source_iid"], f"{label}.source_iid differs")
    _require(
        row["source_video_sha256"] == source["source_video_sha256"],
        f"{label}.source_video_sha256 differs",
    )
    _require(row["branch"] == branch, f"{label}.branch differs")
    _require(row["frame_count"] == 81, f"{label} is not full81")
    _require(_number(row["fps"], f"{label}.fps") == 25.0, f"{label}.fps differs")
    for field in (
        "entire_full81_video_viewed",
        "independent_reviewer",
        "reviewer_blinded_to_teacher_cell",
        "sealed_before_pair_admission",
        "actor_kind_compatible",
        "q0_compatible",
        "owner_object_verified",
        "source_motion_verified",
        "target_event_incompatible_with_source_motion",
    ):
        _require(row[field] is True, f"{label}.{field} is not true")
    return row


_PAIR_FIELDS = {
    "schema_version",
    "pair_id",
    "analysis_split",
    "source_iid",
    "source_video_sha256",
    "branch",
    "teacher_cell_id",
    "event_id",
    "actor_kind",
    "q0_id",
    "source_motion_label",
    "instruction",
    "instruction_utf8_sha256",
    "target_event_incompatible_with_source_motion",
    "optimizer_admitted",
    "pre_admission_full81_review",
    "pair_digest",
}


def _validate_pairs(
    value: Any,
    sources: Sequence[Mapping[str, Any]],
    teacher_origins: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    expected_total = sum(EXPECTED_PAIR_COUNTS.values())
    if type(value) is not list or len(value) != expected_total:
        fail(f"pairs must contain exactly {expected_total} rows")
    source_by_iid = {str(row["source_iid"]): row for row in sources}
    origin_by_cell = {str(row["teacher_cell_id"]): row for row in teacher_origins}
    assignment = deterministic_teacher_assignment_v1(sources, teacher_origins)
    assignment_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for source in sources:
        split = str(source["analysis_split"])
        assignment_groups[(split, assignment[str(source["source_iid"])])].append(source)
    for (split, cell_id), assigned in assignment_groups.items():
        capacity = ASSIGNMENT_CAPACITY[split]
        _require(len(assigned) == capacity, f"{split}/{cell_id} assignment capacity differs")
        for field in ("actor_id", "scene_id", "source_group_id"):
            _require(
                len({str(row[field]) for row in assigned}) == capacity,
                f"{split}/{cell_id} assigned {field} is not diverse",
            )
    rows: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    seen_review_ids: set[str] = set()
    branches_by_source: dict[str, set[str]] = defaultdict(set)
    instruction_hashes_by_source: dict[str, set[str]] = defaultdict(set)
    for index, item in enumerate(value):
        label = f"pairs[{index}]"
        row = _closed(item, _PAIR_FIELDS, label)
        _verify_seal(row, "pair_digest", label)
        _require(row["schema_version"] == PAIR_SCHEMA, f"{label} schema differs")
        pair_id = _safe_id(row["pair_id"], f"{label}.pair_id")
        _require(pair_id not in seen_ids, "pair id is duplicated")
        seen_ids.add(pair_id)
        source_iid = _iid(row["source_iid"], f"{label}.source_iid")
        _require(source_iid in source_by_iid, f"{label} references an unknown source")
        source = source_by_iid[source_iid]
        split = row["analysis_split"]
        _require(split == source["analysis_split"], f"{label}.analysis_split differs")
        _require(
            row["source_video_sha256"] == source["source_video_sha256"],
            f"{label}.source_video_sha256 differs",
        )
        branch = row["branch"]
        _require(branch in BRANCHES, f"{label}.branch differs")
        _require(branch not in branches_by_source[source_iid], "source branch is duplicated")
        branches_by_source[source_iid].add(str(branch))
        cell_id = _safe_id(row["teacher_cell_id"], f"{label}.teacher_cell_id")
        _require(cell_id in origin_by_cell, f"{label} references an unknown teacher cell")
        _require(assignment[source_iid] == cell_id, f"{label} violates deterministic teacher assignment")
        origin = origin_by_cell[cell_id]
        expected_origin_split = "fit" if split in ("fit", "heldout") else "confirmation"
        _require(origin["analysis_split"] == expected_origin_split, f"{label} teacher split differs")
        for field in ("event_id", "actor_kind", "q0_id"):
            _require(row[field] == source[field] == origin[field], f"{label}.{field} compatibility differs")
        _require(
            row["source_motion_label"] == source["source_motion_label"],
            f"{label}.source_motion_label differs",
        )
        _require(
            row["target_event_incompatible_with_source_motion"] is True,
            f"{label} admits a compatible source motion",
        )
        _require(
            row["optimizer_admitted"] is (split == "fit"),
            f"{label}.optimizer_admitted leaks a non-fit row or blocks a fit row",
        )
        instruction = _text(row["instruction"], f"{label}.instruction")
        instruction_sha = _sha(
            row["instruction_utf8_sha256"], f"{label}.instruction_utf8_sha256"
        )
        _require(
            hashlib.sha256(instruction.encode("utf-8")).hexdigest() == instruction_sha,
            f"{label} instruction UTF-8 SHA-256 differs",
        )
        instruction_hashes_by_source[source_iid].add(instruction_sha)
        review = _validate_review(
            row["pre_admission_full81_review"],
            pair_id=pair_id,
            source=source,
            branch=str(branch),
            label=f"{label}.pre_admission_full81_review",
        )
        review_id = str(review["review_id"])
        _require(review_id not in seen_review_ids, "full81 review id is reused across pairs")
        seen_review_ids.add(review_id)
        rows.append(row)
    _require(set(branches_by_source) == set(source_by_iid), "not every source has pair rows")
    _require(
        all(branches == set(BRANCHES) for branches in branches_by_source.values()),
        "each source must have exactly action and incomplete rows",
    )
    _require(
        all(len(values) == len(BRANCHES) for values in instruction_hashes_by_source.values()),
        "action and incomplete instructions are not distinct per source",
    )
    _require(
        Counter(row["analysis_split"] for row in rows) == Counter(EXPECTED_PAIR_COUNTS),
        "pair split counts differ",
    )
    return tuple(rows)


_REPRESENTATION_REVIEW_FIELDS = {
    "schema_version",
    "review_id",
    "evidence_id",
    "anchor_id",
    "anchor_video_sha256",
    "anchor_split",
    "branch",
    "event_id",
    "actor_kind",
    "q0_id",
    "actor_id",
    "scene_id",
    "frame_count",
    "fps",
    "entire_full81_video_viewed",
    "independent_reviewer",
    "reviewer_blinded_to_teacher_cell",
    "reviewer_blinded_to_representation_metrics",
    "sealed_before_sidecar_extraction",
    "sealed_before_representation_admission",
    "target_event_verified",
    "actor_identity_verified",
    "scene_verified",
    "review_digest",
}


def _validate_representation_review(
    value: Any,
    *,
    evidence: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    row = _closed(value, _REPRESENTATION_REVIEW_FIELDS, label)
    _verify_seal(row, "review_digest", label)
    _require(row["schema_version"] == REPRESENTATION_REVIEW_SCHEMA, f"{label} schema differs")
    _safe_id(row["review_id"], f"{label}.review_id")
    for field in (
        "evidence_id",
        "anchor_id",
        "anchor_video_sha256",
        "anchor_split",
        "branch",
        "event_id",
        "actor_kind",
        "q0_id",
        "actor_id",
        "scene_id",
    ):
        _require(row[field] == evidence[field], f"{label}.{field} differs")
    _require(row["frame_count"] == 81, f"{label} is not full81")
    _require(_number(row["fps"], f"{label}.fps") == 25.0, f"{label}.fps differs")
    for field in (
        "entire_full81_video_viewed",
        "independent_reviewer",
        "reviewer_blinded_to_teacher_cell",
        "reviewer_blinded_to_representation_metrics",
        "sealed_before_sidecar_extraction",
        "sealed_before_representation_admission",
        "target_event_verified",
        "actor_identity_verified",
        "scene_verified",
    ):
        _require(row[field] is True, f"{label}.{field} is not true")
    return row


_ANCHOR_EVIDENCE_FIELDS = {
    "schema_version",
    "evidence_id",
    "evidence_role",
    "teacher_cell_id",
    "anchor_id",
    "anchor_iid",
    "anchor_split",
    "branch",
    "event_id",
    "actor_kind",
    "q0_id",
    "actor_id",
    "scene_id",
    "anchor_video_path",
    "anchor_video_sha256",
    "psiout_sidecar_path",
    "psiout_sidecar_sha256",
    "nuisance_packet_path",
    "nuisance_packet_sha256",
    "all_tensor_values_finite",
    "pre_admission_blind_review",
    "evidence_digest",
}
_BOUND_ANCHOR_EVIDENCE_FIELDS = _ANCHOR_EVIDENCE_FIELDS | _MATERIALIZATION_PROVENANCE_FIELDS


def _validate_anchor_evidence(
    value: Any,
    *,
    evidence_role: str,
    origin: Mapping[str, Any],
    branch: str,
    run_authority: _MaterializationRunV1,
    label: str,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, _TensorSlice],
    Mapping[str, _TensorSlice],
    Mapping[str, Any],
]:
    row = _closed(value, _BOUND_ANCHOR_EVIDENCE_FIELDS, label)
    _verify_seal(row, "evidence_digest", label)
    _require(row["schema_version"] == REPRESENTATION_EVIDENCE_SCHEMA, f"{label} schema differs")
    _require(row["evidence_role"] == evidence_role, f"{label}.evidence_role differs")
    evidence_id = _safe_id(row["evidence_id"], f"{label}.evidence_id")
    _safe_id(row["anchor_id"], f"{label}.anchor_id")
    anchor_iid = _iid(row["anchor_iid"], f"{label}.anchor_iid")
    _require(row["teacher_cell_id"] == origin["teacher_cell_id"], f"{label}.teacher_cell_id differs")
    _require(row["anchor_split"] in ("fit", "confirmation"), f"{label}.anchor_split differs")
    _require(row["branch"] == branch, f"{label}.branch differs")
    for field in ("event_id", "actor_kind", "q0_id", "actor_id", "scene_id"):
        _safe_id(row[field], f"{label}.{field}")
    _require(row["event_id"] == origin["event_id"], f"{label}.event_id differs")
    _require(row["actor_kind"] == origin["actor_kind"], f"{label}.actor_kind differs")
    _require(row["q0_id"] == origin["q0_id"], f"{label}.q0_id differs")
    if evidence_role == "teacher_origin":
        _require(anchor_iid == origin["origin_iid"], f"{label}.anchor_iid differs from origin")
        _require(row["anchor_split"] == origin["analysis_split"], f"{label}.anchor_split differs from origin")
        _require(row["actor_id"] == origin["actor_id"], f"{label}.actor_id differs from origin")
        _require(row["scene_id"] == origin["scene_id"], f"{label}.scene_id differs from origin")
    else:
        _require(anchor_iid != origin["origin_iid"], f"{label} reuses the teacher origin IID")
        _require(row["actor_id"] != origin["actor_id"], f"{label} actor is not different")
        _require(row["scene_id"] != origin["scene_id"], f"{label} scene is not different")
    _verify_file(
        row["anchor_video_path"], row["anchor_video_sha256"], f"{label}.anchor_video"
    )
    sidecar = _validate_tensor_container(
        row["psiout_sidecar_path"],
        row["psiout_sidecar_sha256"],
        container_kind="psiout",
        evidence_id=evidence_id,
        evidence_role=evidence_role,
        teacher_cell_id=str(origin["teacher_cell_id"]),
        branch=branch,
        label=f"{label}.psiout_sidecar",
    )
    nuisance = _validate_tensor_container(
        row["nuisance_packet_path"],
        row["nuisance_packet_sha256"],
        container_kind="nuisance",
        evidence_id=evidence_id,
        evidence_role=evidence_role,
        teacher_cell_id=str(origin["teacher_cell_id"]),
        branch=branch,
        label=f"{label}.nuisance_packet",
    )
    _require(row["all_tensor_values_finite"] is True, f"{label} contains non-finite tensors")
    for name, tensor in nuisance.items():
        _require_unit(tensor[1], f"{label}.nuisance_packet[{name}]")
    _validate_representation_review(
        row["pre_admission_blind_review"],
        evidence=row,
        label=f"{label}.pre_admission_blind_review",
    )
    materialization_receipt = _validate_materialization_record_for_evidence(
        row,
        run_authority=run_authority,
        container_slices={
            "psiout": {name: tensor[2] for name, tensor in sidecar.items()},
            "nuisance": {name: tensor[2] for name, tensor in nuisance.items()},
        },
        label=label,
    )
    _require(
        materialization_receipt["record_kind"] == "teacher_anchor",
        f"{label} materialization record is not a teacher anchor",
    )
    return row, sidecar, nuisance, materialization_receipt


_WRONG_CONTROL_FIELDS = {
    "control_type",
    "control_anchor_id",
    "wrong_projected_slice_sha256",
    "wrong_event_cosine",
}

_SIGMA_EVIDENCE_FIELDS = {
    "sigma_index",
    "origin_projected_slice_sha256",
    "cross_anchor_projected_slice_sha256",
    "same_event_cosine",
    "duplicate_forward_first_sha256",
    "duplicate_forward_second_sha256",
    "duplicate_forward_bytes_identical",
    "same_state_noop_minus_noop_null_norm",
    "projected_teacher_raw_norm",
    "signal_to_null_snr",
    "camera_residual_cosine",
    "appearance_residual_cosine",
    "wrong_controls",
}


def _validate_sigma_evidence(
    value: Any,
    *,
    origin_anchor_id: str,
    cross_anchor_id: str,
    origin_sidecar: Mapping[str, _TensorSlice],
    cross_sidecar: Mapping[str, _TensorSlice],
    origin_nuisance: Mapping[str, _TensorSlice],
    label: str,
) -> None:
    if type(value) is not list or len(value) != len(SIGMA_INDICES):
        fail(f"{label} must contain exactly {len(SIGMA_INDICES)} sigma rows")
    control_ids: dict[str, str] = {}
    for ordinal, expected_sigma in enumerate(SIGMA_INDICES):
        sigma_label = f"{label}[{ordinal}]"
        row = _closed(value[ordinal], _SIGMA_EVIDENCE_FIELDS, sigma_label)
        _require(row["sigma_index"] == expected_sigma, f"{sigma_label}.sigma_index differs")

        origin_projected = origin_sidecar[_tensor_name(expected_sigma, "projected_unit")]
        cross_projected = cross_sidecar[_tensor_name(expected_sigma, "projected_unit")]
        projected_raw = origin_sidecar[_tensor_name(expected_sigma, "projected_raw")]
        duplicate_first = origin_sidecar[
            _tensor_name(expected_sigma, "duplicate_forward_first")
        ]
        duplicate_second = origin_sidecar[
            _tensor_name(expected_sigma, "duplicate_forward_second")
        ]
        noop_first = origin_sidecar[_tensor_name(expected_sigma, "noop_forward_first")]
        noop_second = origin_sidecar[_tensor_name(expected_sigma, "noop_forward_second")]
        camera_unit = origin_nuisance[_tensor_name(expected_sigma, "camera_unit")]
        appearance_unit = origin_nuisance[_tensor_name(expected_sigma, "appearance_unit")]

        for field, tensor in (
            ("origin_projected_slice_sha256", origin_projected),
            ("cross_anchor_projected_slice_sha256", cross_projected),
            ("duplicate_forward_first_sha256", duplicate_first),
            ("duplicate_forward_second_sha256", duplicate_second),
        ):
            declared = _sha(row[field], f"{sigma_label}.{field}")
            _require(
                declared == tensor[2],
                f"{sigma_label}.{field} differs from tensor-container bytes",
            )
        _require(
            row["duplicate_forward_bytes_identical"] is True
            and duplicate_first[0] == duplicate_second[0]
            and duplicate_first[2] == duplicate_second[2],
            f"{sigma_label} duplicate forward is not byte deterministic",
        )

        _require_unit(origin_projected[1], f"{sigma_label}.origin_projected")
        _require_unit(cross_projected[1], f"{sigma_label}.cross_anchor_projected")
        actual_same_event = _vector_cosine(
            origin_projected[1], cross_projected[1], f"{sigma_label}.same_event_cosine"
        )
        same_event = _number(row["same_event_cosine"], f"{sigma_label}.same_event_cosine")
        _require(-1.0 <= same_event <= 1.0, f"{sigma_label}.same_event_cosine is outside [-1,1]")
        _require(
            same_event >= SAME_EVENT_MINIMUM_COSINE,
            f"{sigma_label}.same_event_cosine is below {SAME_EVENT_MINIMUM_COSINE}",
        )
        _require_metric_matches(
            same_event, actual_same_event, f"{sigma_label}.same_event_cosine"
        )
        _require(
            actual_same_event >= SAME_EVENT_MINIMUM_COSINE,
            f"{sigma_label}.same_event tensor cosine is below {SAME_EVENT_MINIMUM_COSINE}",
        )

        null_norm = _number(
            row["same_state_noop_minus_noop_null_norm"],
            f"{sigma_label}.same_state_noop_minus_noop_null_norm",
        )
        raw_norm = _number(
            row["projected_teacher_raw_norm"], f"{sigma_label}.projected_teacher_raw_norm"
        )
        claimed_snr = _number(row["signal_to_null_snr"], f"{sigma_label}.signal_to_null_snr")
        _require(null_norm >= 0.0, f"{sigma_label} null norm is negative")
        _require(
            null_norm <= DUPLICATE_MAX_NULL_NORM,
            f"{sigma_label} null norm exceeds {DUPLICATE_MAX_NULL_NORM}",
        )
        _require(
            raw_norm >= PROJECTED_TEACHER_MIN_RAW_NORM,
            f"{sigma_label} projected raw norm is below {PROJECTED_TEACHER_MIN_RAW_NORM}",
        )
        computed_snr = raw_norm / max(null_norm, DUPLICATE_SNR_DENOMINATOR_FLOOR)
        _require(
            math.isclose(claimed_snr, computed_snr, rel_tol=1.0e-9, abs_tol=1.0e-9),
            f"{sigma_label} SNR arithmetic differs",
        )
        _require(claimed_snr >= DUPLICATE_MIN_SNR, f"{sigma_label} SNR is below {DUPLICATE_MIN_SNR}")

        actual_null_norm = math.sqrt(
            math.fsum(
                (float(first) - float(second)) ** 2
                for first, second in zip(noop_first[1], noop_second[1])
            )
        )
        actual_raw_norm = _vector_norm(projected_raw[1])
        _require(actual_raw_norm > 0.0, f"{sigma_label}.projected_raw has zero norm")
        _require(
            _vector_cosine(
                origin_projected[1], projected_raw[1], f"{sigma_label}.projected_raw_direction"
            )
            >= 1.0 - 1.0e-6,
            f"{sigma_label}.projected raw/unit directions differ",
        )
        actual_snr = actual_raw_norm / max(
            actual_null_norm, DUPLICATE_SNR_DENOMINATOR_FLOOR
        )
        _require_metric_matches(
            null_norm,
            actual_null_norm,
            f"{sigma_label}.same_state_noop_minus_noop_null_norm",
        )
        _require_metric_matches(
            raw_norm, actual_raw_norm, f"{sigma_label}.projected_teacher_raw_norm"
        )
        _require_metric_matches(claimed_snr, actual_snr, f"{sigma_label}.signal_to_null_snr")
        _require(
            actual_null_norm <= DUPLICATE_MAX_NULL_NORM,
            f"{sigma_label} tensor null norm exceeds {DUPLICATE_MAX_NULL_NORM}",
        )
        _require(
            actual_raw_norm >= PROJECTED_TEACHER_MIN_RAW_NORM,
            f"{sigma_label} tensor projected raw norm is below {PROJECTED_TEACHER_MIN_RAW_NORM}",
        )
        _require(
            actual_snr >= DUPLICATE_MIN_SNR,
            f"{sigma_label} tensor SNR is below {DUPLICATE_MIN_SNR}",
        )

        actual_nuisance = {
            "camera_residual_cosine": _vector_cosine(
                origin_projected[1], camera_unit[1], f"{sigma_label}.camera_residual_cosine"
            ),
            "appearance_residual_cosine": _vector_cosine(
                origin_projected[1],
                appearance_unit[1],
                f"{sigma_label}.appearance_residual_cosine",
            ),
        }
        for field in ("camera_residual_cosine", "appearance_residual_cosine"):
            residual = _number(row[field], f"{sigma_label}.{field}")
            _require(-1.0 <= residual <= 1.0, f"{sigma_label}.{field} is outside [-1,1]")
            _require(
                abs(residual) <= NUISANCE_MAX_ABS_COSINE,
                f"{sigma_label}.{field} exceeds {NUISANCE_MAX_ABS_COSINE}",
            )
            _require_metric_matches(residual, actual_nuisance[field], f"{sigma_label}.{field}")
            _require(
                abs(actual_nuisance[field]) <= NUISANCE_MAX_ABS_COSINE,
                f"{sigma_label}.{field} tensor cosine exceeds {NUISANCE_MAX_ABS_COSINE}",
            )

        controls = row["wrong_controls"]
        if type(controls) is not list or len(controls) != len(WRONG_CONTROL_TYPES):
            fail(f"{sigma_label}.wrong_controls must contain exactly {len(WRONG_CONTROL_TYPES)} rows")
        for control_ordinal, expected_type in enumerate(WRONG_CONTROL_TYPES):
            control_label = f"{sigma_label}.wrong_controls[{control_ordinal}]"
            control = _closed(controls[control_ordinal], _WRONG_CONTROL_FIELDS, control_label)
            _require(control["control_type"] == expected_type, f"{control_label}.control_type differs")
            control_id = _safe_id(control["control_anchor_id"], f"{control_label}.control_anchor_id")
            _require(
                control_id not in (origin_anchor_id, cross_anchor_id),
                f"{control_label} reuses a correct-event anchor",
            )
            previous_id = control_ids.setdefault(expected_type, control_id)
            _require(previous_id == control_id, f"{label} {expected_type} control anchor changes by sigma")
            control_tensor = origin_sidecar[
                _tensor_name(expected_sigma, f"{expected_type}_projected_unit")
            ]
            _require_unit(control_tensor[1], f"{control_label}.projected")
            control_sha = _sha(
                control["wrong_projected_slice_sha256"],
                f"{control_label}.wrong_projected_slice_sha256",
            )
            _require(
                control_sha == control_tensor[2],
                f"{control_label}.wrong_projected_slice_sha256 differs from tensor-container bytes",
            )
            wrong = _number(control["wrong_event_cosine"], f"{control_label}.wrong_event_cosine")
            _require(-1.0 <= wrong <= 1.0, f"{control_label}.wrong_event_cosine is outside [-1,1]")
            margin = same_event - wrong
            _require(
                margin + 1.0e-12 >= WRONG_CONTROL_MINIMUM_MARGIN,
                f"{control_label} correct-minus-wrong margin is below {WRONG_CONTROL_MINIMUM_MARGIN}",
            )
            actual_wrong = _vector_cosine(
                origin_projected[1],
                control_tensor[1],
                f"{control_label}.wrong_event_cosine",
            )
            _require_metric_matches(wrong, actual_wrong, f"{control_label}.wrong_event_cosine")
            _require(
                actual_same_event - actual_wrong + 1.0e-12
                >= WRONG_CONTROL_MINIMUM_MARGIN,
                f"{control_label} tensor correct-minus-wrong margin is below "
                f"{WRONG_CONTROL_MINIMUM_MARGIN}",
            )
    _require(
        len(set(control_ids.values())) == len(WRONG_CONTROL_TYPES),
        f"{label} wrong-control anchor IDs are not distinct",
    )


_REPRESENTATION_FIELDS = {
    "schema_version",
    "admission_id",
    "teacher_cell_id",
    "analysis_split",
    "branch",
    "event_id",
    "origin_evidence",
    "cross_anchor_evidence",
    "sigma_evidence",
    "optimizer_admitted",
    "admission_digest",
}


def _validate_representation_admissions(
    value: Any,
    teacher_origins: Sequence[Mapping[str, Any]],
    run_authority: _MaterializationRunV1,
) -> tuple[Mapping[str, Any], ...]:
    expected_total = sum(EXPECTED_REPRESENTATION_COUNTS.values())
    if type(value) is not list or len(value) != expected_total:
        fail(f"representation_admissions must contain exactly {expected_total} rows")
    origin_by_cell = {str(row["teacher_cell_id"]): row for row in teacher_origins}
    rows: list[Mapping[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_admission_ids: set[str] = set()
    seen_evidence_ids: set[str] = set()
    seen_anchor_ids: set[str] = set()
    # One independently reviewed same-event video may serve as the fixed
    # cross-actor/scene comparator for more than one teacher seed of that
    # event.  Reuse is safe only when every intrinsic video identity field is
    # byte-for-byte identical.  This matches the preregistered scientific
    # requirement (at least one independent comparator per event) without
    # manufacturing extra media merely to satisfy per-row bookkeeping.
    anchor_video_bindings: dict[
        str, tuple[str, str, str, str, str, str, str]
    ] = {}
    seen_review_ids: set[str] = set()
    seen_review_digests: set[str] = set()
    seen_sidecars: set[str] = set()
    seen_nuisance_packets: set[str] = set()
    seen_materialization_record_ids: set[str] = set()
    cross_anchor_bindings: dict[str, tuple[str, str, str, str, str]] = {}
    fragment_values = run_authority.receipt[
        "representation_sigma_evidence_candidates"
    ]
    if type(fragment_values) is not list:
        fail("materialization representation fragment closure differs")
    fragments: dict[tuple[str, str], Mapping[str, Any]] = {}
    for fragment_ordinal, item in enumerate(fragment_values):
        fragment = _closed(
            item,
            _MATERIALIZATION_REPRESENTATION_FRAGMENT_FIELDS,
            f"materialization representation fragments[{fragment_ordinal}]",
        )
        fragment_key = (str(fragment["teacher_cell_id"]), str(fragment["branch"]))
        _require(fragment_key not in fragments, "materialization representation fragment key is reused")
        fragments[fragment_key] = fragment
    for index, item in enumerate(value):
        label = f"representation_admissions[{index}]"
        row = _closed(item, _REPRESENTATION_FIELDS, label)
        _verify_seal(row, "admission_digest", label)
        _require(row["schema_version"] == REPRESENTATION_SCHEMA, f"{label} schema differs")
        admission_id = _safe_id(row["admission_id"], f"{label}.admission_id")
        _require(admission_id not in seen_admission_ids, "representation admission id is duplicated")
        seen_admission_ids.add(admission_id)
        cell_id = _safe_id(row["teacher_cell_id"], f"{label}.teacher_cell_id")
        _require(cell_id in origin_by_cell, f"{label} references an unknown teacher cell")
        origin = origin_by_cell[cell_id]
        branch = row["branch"]
        _require(branch in BRANCHES, f"{label}.branch differs")
        key = (cell_id, str(branch))
        _require(key not in seen_keys, "representation cell/branch is duplicated")
        seen_keys.add(key)
        _require(row["analysis_split"] == origin["analysis_split"], f"{label}.analysis_split differs")
        _require(row["event_id"] == origin["event_id"], f"{label}.event_id differs")
        (
            origin_evidence,
            origin_sidecar,
            origin_nuisance,
            origin_materialization,
        ) = _validate_anchor_evidence(
            row["origin_evidence"],
            evidence_role="teacher_origin",
            origin=origin,
            branch=str(branch),
            run_authority=run_authority,
            label=f"{label}.origin_evidence",
        )
        (
            cross_evidence,
            cross_sidecar,
            _cross_nuisance,
            cross_materialization,
        ) = _validate_anchor_evidence(
            row["cross_anchor_evidence"],
            evidence_role="same_event_cross_anchor",
            origin=origin,
            branch=str(branch),
            run_authority=run_authority,
            label=f"{label}.cross_anchor_evidence",
        )
        for materialization in (origin_materialization, cross_materialization):
            record_id = str(materialization["record_id"])
            _require(record_id not in seen_materialization_record_ids, "teacher materialization record is reused")
            seen_materialization_record_ids.add(record_id)
        _require(
            origin_evidence["anchor_id"] != cross_evidence["anchor_id"],
            f"{label} cross anchor reuses the origin anchor ID",
        )
        for evidence_label, evidence in (
            ("origin_evidence", origin_evidence),
            ("cross_anchor_evidence", cross_evidence),
        ):
            evidence_id = str(evidence["evidence_id"])
            _require(evidence_id not in seen_evidence_ids, "representation evidence id is reused")
            seen_evidence_ids.add(evidence_id)
            anchor_id = str(evidence["anchor_id"])
            _require(anchor_id not in seen_anchor_ids, "representation anchor id is reused")
            seen_anchor_ids.add(anchor_id)
            anchor_video_sha = _sha(
                evidence["anchor_video_sha256"],
                f"{label}.{evidence_label}.anchor_video_sha256",
            )
            video_binding = (
                str(evidence["anchor_iid"]),
                str(evidence["anchor_split"]),
                str(evidence["branch"]),
                str(evidence["event_id"]),
                str(evidence["actor_kind"]),
                str(evidence["actor_id"]),
                str(evidence["scene_id"]),
            )
            _require(
                anchor_video_bindings.setdefault(
                    anchor_video_sha, video_binding
                )
                == video_binding,
                "reused representation anchor video identity differs",
            )
            review = evidence["pre_admission_blind_review"]
            review_id = str(review["review_id"])
            review_digest = str(review["review_digest"])
            _require(review_id not in seen_review_ids, "representation blind-review id is reused")
            _require(
                review_digest not in seen_review_digests,
                "representation blind-review digest is reused",
            )
            seen_review_ids.add(review_id)
            seen_review_digests.add(review_digest)
            sidecar_sha = _sha(
                evidence["psiout_sidecar_sha256"],
                f"{label}.{evidence_label}.psiout_sidecar_sha256",
            )
            nuisance_sha = _sha(
                evidence["nuisance_packet_sha256"],
                f"{label}.{evidence_label}.nuisance_packet_sha256",
            )
            _require(sidecar_sha not in seen_sidecars, "PsiOut sidecar SHA-256 is reused")
            _require(
                nuisance_sha not in seen_nuisance_packets,
                "nuisance packet SHA-256 is reused",
            )
            seen_sidecars.add(sidecar_sha)
            seen_nuisance_packets.add(nuisance_sha)
        cross_anchor_id = str(cross_evidence["anchor_id"])
        cross_anchor_iid = str(cross_evidence["anchor_iid"])
        cross_binding = (
            str(cross_evidence["anchor_split"]),
            str(cross_evidence["event_id"]),
            str(cross_evidence["actor_id"]),
            str(cross_evidence["scene_id"]),
            str(cross_evidence["actor_kind"]),
        )
        _require(
            cross_anchor_bindings.setdefault(cross_anchor_iid, cross_binding) == cross_binding,
            "same-event cross anchor identity binding differs between branches",
        )
        _validate_sigma_evidence(
            row["sigma_evidence"],
            origin_anchor_id=str(origin_evidence["anchor_id"]),
            cross_anchor_id=cross_anchor_id,
            origin_sidecar=origin_sidecar,
            cross_sidecar=cross_sidecar,
            origin_nuisance=origin_nuisance,
            label=f"{label}.sigma_evidence",
        )
        _require(key in fragments, f"{label} has no materialization representation fragment")
        fragment = fragments[key]
        expected_fragment = {
            "teacher_cell_id": cell_id,
            "branch": branch,
            "origin_record_id": origin_materialization["record_id"],
            "cross_anchor_record_id": cross_materialization["record_id"],
            "origin_evidence_digest": origin_materialization[
                "candidate_authority_evidence"
            ]["evidence_digest"],
            "cross_anchor_evidence_digest": cross_materialization[
                "candidate_authority_evidence"
            ]["evidence_digest"],
            "sigma_evidence": row["sigma_evidence"],
        }
        _require(fragment == expected_fragment, f"{label} materialization representation fragment differs")
        for sigma_ordinal, sigma_row in enumerate(row["sigma_evidence"]):
            origin_metric = origin_materialization["sigma_metrics"][sigma_ordinal]
            cross_metric = cross_materialization["sigma_metrics"][sigma_ordinal]
            metric_bindings = {
                "origin_projected_slice_sha256": origin_metric[
                    "projected_unit_sha256"
                ],
                "cross_anchor_projected_slice_sha256": cross_metric[
                    "projected_unit_sha256"
                ],
                "duplicate_forward_first_sha256": origin_metric[
                    "duplicate_forward_first_sha256"
                ],
                "duplicate_forward_second_sha256": origin_metric[
                    "duplicate_forward_second_sha256"
                ],
                "duplicate_forward_bytes_identical": origin_metric[
                    "duplicate_forward_bytes_identical"
                ],
                "same_state_noop_minus_noop_null_norm": origin_metric[
                    "same_state_noop_minus_noop_null_norm"
                ],
                "projected_teacher_raw_norm": origin_metric[
                    "projected_teacher_raw_norm"
                ],
                "signal_to_null_snr": origin_metric["signal_to_null_snr"],
                "camera_residual_cosine": origin_metric[
                    "camera_residual_cosine"
                ],
                "appearance_residual_cosine": origin_metric[
                    "appearance_residual_cosine"
                ],
                "wrong_controls": origin_metric["wrong_controls"],
            }
            for field, expected in metric_bindings.items():
                _require(
                    sigma_row[field] == expected,
                    f"{label}.sigma_evidence[{sigma_ordinal}].{field} differs from materialization receipt",
                )
        expected_optimizer_role = origin["analysis_split"] == "fit"
        _require(
            row["optimizer_admitted"] is expected_optimizer_role,
            f"{label}.optimizer_admitted leaks confirmation or blocks fit",
        )
        rows.append(row)
    expected_keys = {
        (str(origin["teacher_cell_id"]), branch)
        for origin in teacher_origins
        for branch in BRANCHES
    }
    _require(seen_keys == expected_keys, "representation cell/branch closure differs")
    _require(
        Counter(row["analysis_split"] for row in rows)
        == Counter(EXPECTED_REPRESENTATION_COUNTS),
        "representation split counts differ",
    )
    _require(set(fragments) == expected_keys, "materialization representation fragment key closure differs")
    run_teacher_records = {
        record_id
        for record_id, receipt in run_authority.record_receipts.items()
        if receipt["record_kind"] == "teacher_anchor"
    }
    _require(
        seen_materialization_record_ids == run_teacher_records,
        "materialization teacher record closure has extra/missing records",
    )
    return tuple(rows)


def _validate_io_policy(value: Any) -> None:
    row = _closed(
        value,
        {
            "physical_payload",
            "posterior_index_decoded",
            "synthetic_target_index1_path_present",
            "synthetic_target_index1_bytes_read",
            "synthetic_target_index1_decoded",
            "synthetic_target_index1_hashed",
        },
        "source_io_policy",
    )
    _require(row["physical_payload"] == "source_posterior_index_0_only", "source I/O payload differs")
    _require(row["posterior_index_decoded"] == 0, "source I/O posterior index differs")
    for field in (
        "synthetic_target_index1_path_present",
        "synthetic_target_index1_bytes_read",
        "synthetic_target_index1_decoded",
        "synthetic_target_index1_hashed",
    ):
        _require(row[field] is False, f"source_io_policy.{field} is not false")


def _expected_counts() -> Mapping[str, Any]:
    return {
        "sources": {**EXPECTED_SOURCE_COUNTS, "total": sum(EXPECTED_SOURCE_COUNTS.values())},
        "pairs": {**EXPECTED_PAIR_COUNTS, "total": sum(EXPECTED_PAIR_COUNTS.values())},
        "teacher_origins": {
            **EXPECTED_TEACHER_COUNTS,
            "total": sum(EXPECTED_TEACHER_COUNTS.values()),
        },
        "representation_bundles": {
            **EXPECTED_REPRESENTATION_COUNTS,
            "total": sum(EXPECTED_REPRESENTATION_COUNTS.values()),
        },
        "representation_anchor_evidence": sum(EXPECTED_REPRESENTATION_COUNTS.values()) * 2,
        "representation_blind_reviews": sum(EXPECTED_REPRESENTATION_COUNTS.values()) * 2,
        "representation_sigma_rows": sum(EXPECTED_REPRESENTATION_COUNTS.values())
        * len(SIGMA_INDICES),
        "representation_wrong_control_rows": sum(EXPECTED_REPRESENTATION_COUNTS.values())
        * len(SIGMA_INDICES)
        * len(WRONG_CONTROL_TYPES),
        "pre_admission_full81_reviews": sum(EXPECTED_PAIR_COUNTS.values()),
        "optimizer_pair_rows": EXPECTED_PAIR_COUNTS["fit"],
        "optimizer_teacher_bundles": EXPECTED_REPRESENTATION_COUNTS["fit"],
    }


def _validate_authority(value: Any) -> None:
    row = _closed(
        value,
        {
            "status",
            "data_authority_complete",
            "teacher_authority_complete",
            "current_optimizer_pair_rows",
            "current_optimizer_teacher_bundles",
            "current_authority_nonzero",
            "optimizer_authorized",
        },
        "authority",
    )
    _require(row["status"] == "optimizer_admitted", "authority status is not admitted")
    _require(row["data_authority_complete"] is True, "data authority is incomplete")
    _require(row["teacher_authority_complete"] is True, "teacher authority is incomplete")
    _require(
        row["current_optimizer_pair_rows"] == EXPECTED_PAIR_COUNTS["fit"],
        "current optimizer pair authority is zero/incomplete",
    )
    _require(
        row["current_optimizer_teacher_bundles"] == EXPECTED_REPRESENTATION_COUNTS["fit"],
        "current optimizer teacher authority is zero/incomplete",
    )
    _require(row["current_authority_nonzero"] is True, "current authority is zero")
    _require(row["optimizer_authorized"] is True, "optimizer is not authorized")


_TOP_FIELDS = {
    "schema_version",
    "materialization_run_receipt",
    "authority",
    "source_io_policy",
    "teacher_origins",
    "sources",
    "pairs",
    "representation_admissions",
    "authority_counts",
    "manifest_digest",
}


def validate_full30_action_authority_v1(value: Any) -> Mapping[str, Any]:
    manifest = _closed(value, _TOP_FIELDS, "authority manifest")
    _verify_seal(manifest, "manifest_digest", "authority manifest")
    _require(manifest["schema_version"] == SCHEMA_VERSION, "authority manifest schema differs")
    materialization_run = _load_materialization_run_v1(
        manifest["materialization_run_receipt"]
    )
    _validate_authority(manifest["authority"])
    _validate_io_policy(manifest["source_io_policy"])
    teachers = _validate_teacher_origins(manifest["teacher_origins"])
    sources = _validate_sources(manifest["sources"], teachers)
    pairs = _validate_pairs(manifest["pairs"], sources, teachers)
    representations = _validate_representation_admissions(
        manifest["representation_admissions"], teachers, materialization_run
    )
    expected_counts = _expected_counts()
    _require(manifest["authority_counts"] == expected_counts, "authority count closure differs")
    receipt_unsigned = {
        "schema_version": VALIDATION_SCHEMA,
        "manifest_digest": manifest["manifest_digest"],
        "materialization_run_digest": materialization_run.receipt["run_digest"],
        "materialization_run_receipt_sha256": materialization_run.binding[
            "file_sha256"
        ],
        "materialization_record_receipts": len(
            materialization_run.record_receipts
        ),
        "source_counts": dict(Counter(row["analysis_split"] for row in sources)),
        "pair_counts": dict(Counter(row["analysis_split"] for row in pairs)),
        "teacher_origin_counts": dict(Counter(row["analysis_split"] for row in teachers)),
        "representation_counts": dict(
            Counter(row["analysis_split"] for row in representations)
        ),
        "representation_anchor_evidence": len(representations) * 2,
        "representation_blind_reviews": len(representations) * 2,
        "representation_sigma_rows": len(representations) * len(SIGMA_INDICES),
        "representation_wrong_control_rows": len(representations)
        * len(SIGMA_INDICES)
        * len(WRONG_CONTROL_TYPES),
        "synthetic_target_index1_bytes_read": False,
        "all_pre_admission_reviews_full81": True,
        "deterministic_assignment_verified": True,
        "representation_admission_verified": True,
        "optimizer_authorized": True,
    }
    return {
        **receipt_unsigned,
        "validation_digest": object_sha256(receipt_unsigned),
    }


def validate_manifest_file(path: str | Path, expected_sha256: str) -> Mapping[str, Any]:
    return validate_full30_action_authority_v1(_load_json(path, expected_sha256))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        receipt = validate_manifest_file(args.manifest, args.expected_sha256)
    except (Full30ActionAuthorityError, OSError) as error:
        print(f"full30 action authority rejected: {error}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
