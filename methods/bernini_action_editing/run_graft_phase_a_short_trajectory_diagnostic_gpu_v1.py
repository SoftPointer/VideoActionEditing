#!/usr/bin/env python3
"""Fresh WORLD8 trajectory diagnostic for the fixed GRAFT Phase-A short run.

This module joins three already frozen boundaries without expanding their
authority:

* the job-132549 A-lite consumer owns four exact81 source byte strings;
* the v2 native closure owns one source/no-op FM cell and its local gradient;
* the unchanged short-training core owns DP2xSP4 reduction, AdamW, the original
  confirmation thresholds, rollback, and the two fixed updates at exact40
  indices 29 then 38.

Ranks 0..3 form the dog SP4 arm and ranks 4..7 form the human SP4 arm.  Each
arm trains only on its preregistered fit row.  Its disjoint confirmation row is
never sent to the optimizer.  This diagnostic measures the same exact six
detached FP32 fields at indices 29 and 38 at three preregistered stages:
initial zero adapter, after update 29, and after update 38.  The fields are the
analytical
source/no-op target; correct-, same-family-fit-wrong-, and dropped-atlas no-op
fields; and correct- and dropped-atlas action fields.  The wrong/drop
interventions change only IdentityRebinder memory.  The native confirmation
source V-pack, noisy state, epsilon, source zS, negative text condition, and
sigma/timestep stay fixed; action versus no-op changes only the positive text
condition.

The diagnostic records same-noise L_correct/L_wrong/L_drop, relative gains,
action cosine/norm ratio, owner/atlas/route digests, and exact SP4 consensus at
every matrix cell.  The final stage is then submitted to the unchanged short
core's original noncompensating hard gates.  A scientific NO-GO is rolled back
and still returned as a zero-authority diagnostic receipt; it is never treated
as training success and never writes a checkpoint.

Before the adapter is installed, the runner captures adapter-off BF16 raw
outputs at exact40 indices 0 and 25.  After the final trajectory measurement,
the parity postflight must reproduce every raw byte at both indices.

There is one preregistered optimizer arm only: AdamW lr=1e-3 and exactly two
updates 29 then 38.  There is no learning-rate/step sweep and no post-hoc arm
selection.  No target video, generated proposal, T2V branch, source retelling, selector,
mask, pose, track, flow, or motion donor is an input.  No checkpoint or other
artifact is written.  A successful return is an in-memory diagnostic receipt,
not action, identity, quality, training, checkpoint, production, or scientific
authority.
"""

from __future__ import annotations

import argparse
import ctypes
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import re
import stat
import struct
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

import graft_a_lite_source_release_consumer_v1 as source_consumer
import graft_phase_a_native_training_closure_v2 as native_v2
import identity_rebinder_v1 as rebinder
import infer_lora as legacy
import infer_native_identity_generation_canary as native_generation
import infer_source_kv_carrier_oracle as source_audit
import inference_sigma_strata as sigma_strata
import run_graft_phase_a_native_gpu_canary_v1 as native_runner_v1
import train_graft_phase_a_a_lite_short_v1 as short_trainer
import tri_branch_unipc as sampler_contract


SCHEMA_VERSION = "bernini-graft-phase-a-short-trajectory-diagnostic-gpu-v1"
FAILURE_SCHEMA_VERSION = (
    "bernini-graft-phase-a-short-trajectory-diagnostic-failure-v1"
)
TRAJECTORY_MATRIX_SCHEMA_VERSION = (
    "bernini-graft-phase-a-short-trajectory-matrix-v1"
)
TRAJECTORY_STAGES = ("initial", "after_update_29", "after_update_38")
CONFIRMATION_FIELDS_SCHEMA_VERSION = (
    "bernini-graft-phase-a-short-six-field-provenance-v1"
)
ADAPTER_OFF_PARITY_SCHEMA_VERSION = (
    "bernini-graft-phase-a-short-adapter-off-bf16-parity-v1"
)
SERVICES_SCHEMA_VERSION = "bernini-graft-phase-a-short-runner-services-v1"
TERMINAL_GATE_PACKET_SCHEMA_VERSION = (
    "bernini-graft-phase-a-short-world8-terminal-gate-packet-v1"
)
TERMINAL_COORDINATION_SCHEMA_VERSION = (
    "bernini-graft-phase-a-short-world8-terminal-coordination-v1"
)
TERMINAL_ROLLBACK_PACKET_SCHEMA_VERSION = (
    "bernini-graft-phase-a-short-world8-terminal-rollback-packet-v1"
)
TERMINAL_ROLLBACK_MANIFEST_SCHEMA_VERSION = (
    "bernini-graft-phase-a-short-world8-terminal-rollback-manifest-v1"
)

WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
FRAME_COUNT = 81
LATENT_PHASES = 21
UPDATE_INDICES = (29, 38)
CONFIRMATION_INDICES = (29, 38)
ADAPTER_OFF_PARITY_INDICES = (0, 25)
PARITY_BRANCH_ROLES = ("negative", "noop_positive", "action_positive")
MAX_FULL_LOCAL_RESULT_PACKET_BYTES = 16 * 1024 * 1024
MAX_CTYPES_DIGEST_CHUNK_BYTES = 64 * 1024 * 1024

PINNED_CONSUMER_SOURCE_SHA256 = (
    "13ecb082ab3cff6f809b056c35715123be302a5c8d82a6760a7367861920ee75"
)
PINNED_NATIVE_V2_SOURCE_SHA256 = (
    "bf6a1d438183de5aa0460e729a39382e4597b3e43a4b9f1b3cdff5457439f20f"
)
PINNED_SHORT_TRAINER_SOURCE_SHA256 = (
    "73e39048bb8836fef33516eb1aae4cbc3f9fa4ecefcfb5d2695925bcb150f7bb"
)
PINNED_SHORT_TRAINER_EXECUTION_RUNTIME_SHA256 = (
    "f35f621938e7a6b8bd3b9b5a6b0fb782f5ebf483939f585f561e3908f993af3c"
)
PINNED_NATIVE_RUNNER_V1_SOURCE_SHA256 = (
    "e0b69442be284e091bad8d36a205bffe8bd314082188bfa55da72f4c2640945a"
)
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)

FAMILY_BY_DP_ARM = ("dog", "human")
FIT_IID_BY_DP_ARM = ("7b88a1ca1f804f41", "a35b590961d24694")
CONFIRMATION_IID_BY_DP_ARM = (
    "841b5e0080a1441d",
    "a66e6818e4144928",
)
ACTION_INSTRUCTION_BY_DP_ARM = (
    (
        "Have the main dog bend its hind legs, lower its pelvis into a stable "
        "seated pose at the same place and facing direction, and hold the sit."
    ),
    (
        "Have the main person shift weight onto both feet, rise smoothly, "
        "straighten the legs and torso, and hold a stable upright stand."
    ),
)
NOISE_BASE_SEED = 2026081001

AUTHORITY_FIELDS = tuple(short_trainer.AUTHORITY_FIELDS)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_TEST_NAME = re.compile(r"cpu_fake:[A-Za-z0-9._:-]{1,140}\Z")
_FORBIDDEN_PUBLIC_INPUT_FRAGMENTS = (
    "proposal",
    "asga",
    "selector",
    "retelling",
    "caption",
    "target_video",
    "generated_video",
    "mask",
    "pose",
    "track",
    "flow",
    "donor",
)
_CONFIRMATION_GATE_KEYS = (
    "correct_vs_wrong_noop_relative_gain",
    "correct_vs_drop_noop_relative_gain",
    "action_delta_correct_drop_norm_ratio",
    "action_delta_correct_drop_cosine",
)
_GLOBAL_NO_GO_ROLLBACK_SENTINEL = (
    "authenticated WORLD8 scientific NO-GO requires all-rank rollback without finish"
)


class GraftPhaseAShortGPUError(RuntimeError):
    """Fail closed while retaining a path-free in-memory diagnostic receipt."""

    def __init__(
        self, message: str, *, diagnostic_receipt: Optional[Mapping[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.diagnostic_receipt = diagnostic_receipt


class _World8ScientificNoGoRollback(RuntimeError):
    """Authenticated sentinel used only after the WORLD8 gate gather."""


def _plain_json_value(value: Any) -> Any:
    """Own immutable JSON containers before canonical serialization."""

    if type(value) in (dict, MappingProxyType):
        return {key: _plain_json_value(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_plain_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain_json_value(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GraftPhaseAShortGPUError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    plain = dict(value)
    if "digest" in plain:
        raise GraftPhaseAShortGPUError("sealed mapping already contains digest")
    plain["digest"] = object_sha256(plain)
    return MappingProxyType(plain)


def validate_sealed_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GraftPhaseAShortGPUError(f"{label} must be a mapping")
    plain = dict(value)
    digest = plain.pop("digest", None)
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise GraftPhaseAShortGPUError(f"{label} digest is not canonical SHA256")
    if object_sha256(plain) != digest:
        raise GraftPhaseAShortGPUError(f"{label} digest differs")
    return {**plain, "digest": digest}


def _false_authority() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_FIELDS}


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GraftPhaseAShortGPUError(f"{label} must be lowercase SHA256")
    return value


def _require_hex40(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise GraftPhaseAShortGPUError(f"{label} must be lowercase hex-40")
    return value


def file_sha256(path: Path | str) -> str:
    candidate = Path(path).resolve(strict=True)
    descriptor = os.open(
        candidate,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GraftPhaseAShortGPUError(f"not a plain file: {candidate}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise GraftPhaseAShortGPUError(f"file changed while hashing: {candidate}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def tensor_identity(value: torch.Tensor) -> Mapping[str, Any]:
    return native_runner_v1.tensor_identity(value)


def short_chunked_tensor_identity(value: torch.Tensor) -> Mapping[str, Any]:
    """v1-exact identity using bounded ctypes reads instead of bytes(storage)."""

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise GraftPhaseAShortGPUError("chunked identity requires an exact tensor")
    detached = value.detach()
    if type(detached) is not torch.Tensor:
        raise GraftPhaseAShortGPUError("chunked identity rejects detach hooks")
    owned = detached.cpu().contiguous().clone()
    if type(owned) is not torch.Tensor:
        raise GraftPhaseAShortGPUError("chunked identity ownership differs")
    storage = owned.untyped_storage()
    expected = int(owned.numel()) * int(owned.element_size())
    if int(storage.nbytes()) != expected:
        raise GraftPhaseAShortGPUError("chunked identity logical storage differs")
    raw_digest = hashlib.sha256()
    header = canonical_json_bytes(
        {"shape": [int(item) for item in owned.shape], "dtype": str(owned.dtype)}
    )
    content_digest = hashlib.sha256(header + b"\0")
    if expected:
        pointer = int(owned.data_ptr())
        if pointer == 0:
            raise GraftPhaseAShortGPUError("nonempty tensor has a null data pointer")
        for offset in range(0, expected, MAX_CTYPES_DIGEST_CHUNK_BYTES):
            size = min(MAX_CTYPES_DIGEST_CHUNK_BYTES, expected - offset)
            block = _PINNED_CTYPES_STRING_AT(pointer + offset, size)
            if type(block) is not bytes or len(block) != size:
                raise GraftPhaseAShortGPUError("ctypes tensor read differs")
            raw_digest.update(block)
            content_digest.update(block)
    return {
        "shape": [int(item) for item in owned.shape],
        "dtype": str(owned.dtype),
        "device_type_at_observation": value.device.type,
        "finite": bool(torch.isfinite(owned).all().item()),
        "byte_count": expected,
        "raw_sha256": raw_digest.hexdigest(),
        "content_sha256": content_digest.hexdigest(),
    }


def short_chunked_parameter_registry_digest(
    rows: Sequence[tuple[str, torch.nn.Parameter]],
) -> str:
    digest = hashlib.sha256()
    for name, parameter in rows:
        if not isinstance(name, str) or not isinstance(parameter, torch.nn.Parameter):
            raise GraftPhaseAShortGPUError("chunked parameter registry row differs")
        payload = canonical_json_bytes(
            {"name": name, "tensor": short_chunked_tensor_identity(parameter)}
        )
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


_PINNED_CTYPES_STRING_AT = ctypes.string_at
_PINNED_SHORT_CHUNKED_TENSOR_IDENTITY = short_chunked_tensor_identity
_PINNED_SHORT_CHUNKED_REGISTRY_DIGEST = short_chunked_parameter_registry_digest


_PINNED_IMPORT_IDENTITIES = MappingProxyType(
    {
        "source_consumer": source_consumer,
        "native_v2": native_v2,
        "short_trainer": short_trainer,
        "native_runner_v1": native_runner_v1,
    }
)
_PINNED_CONSUME = source_consumer.consume_graft_a_lite_source_release
_PINNED_VALIDATE_FOR_TRAINING = source_consumer.validate_for_training
_PINNED_OPEN_TRAINER = short_trainer.open_authenticated_short_training
_PINNED_SESSION_FAIL = short_trainer.PhaseAShortTrainingSession._fail
_PINNED_NATIVE_V2_CLASS = native_v2.PhaseANativeTrainingClosure


def assert_pinned_dependencies() -> None:
    """Bind exact committed sources and the process-local execution symbols."""

    expected = (
        (source_consumer, PINNED_CONSUMER_SOURCE_SHA256, "consumer"),
        (native_v2, PINNED_NATIVE_V2_SOURCE_SHA256, "native v2"),
        (short_trainer, PINNED_SHORT_TRAINER_SOURCE_SHA256, "short trainer"),
        (
            native_runner_v1,
            PINNED_NATIVE_RUNNER_V1_SOURCE_SHA256,
            "native v1 GPU runner",
        ),
    )
    for module, digest, label in expected:
        path = Path(module.__file__).resolve(strict=True)
        if file_sha256(path) != digest:
            raise GraftPhaseAShortGPUError(f"pinned {label} source differs")
    if (
        source_consumer.consume_graft_a_lite_source_release is not _PINNED_CONSUME
        or source_consumer.validate_for_training is not _PINNED_VALIDATE_FOR_TRAINING
        or short_trainer.open_authenticated_short_training is not _PINNED_OPEN_TRAINER
        or short_trainer.PhaseAShortTrainingSession._fail is not _PINNED_SESSION_FAIL
        or native_v2.PhaseANativeTrainingClosure is not _PINNED_NATIVE_V2_CLASS
        or native_v2.PhaseANativeTrainingClosureV2 is not _PINNED_NATIVE_V2_CLASS
        or native_v2.PINNED_V1_SOURCE_SHA256
        != "36861e8670fc77d65b469f86aa472a10d453f9ba5fc227a62303329c4c38409a"
        or short_trainer.PINNED_CONSUMER_SOURCE_SHA256
        != PINNED_CONSUMER_SOURCE_SHA256
        or short_trainer.PINNED_NATIVE_V2_SOURCE_SHA256
        != PINNED_NATIVE_V2_SOURCE_SHA256
        or short_trainer.PINNED_TRAINER_EXECUTION_RUNTIME_SHA256
        != PINNED_SHORT_TRAINER_EXECUTION_RUNTIME_SHA256
        or tuple(short_trainer.UPDATE_SCHEDULE_INDICES) != UPDATE_INDICES
        or tuple(short_trainer.CONFIRMATION_SCHEDULE_INDICES)
        != CONFIRMATION_INDICES
        or tuple(short_trainer.RUNNER_ADAPTER_OFF_PARITY_INDICES)
        != ADAPTER_OFF_PARITY_INDICES
        or (short_trainer.WORLD_SIZE, short_trainer.DP_SIZE, short_trainer.SP_SIZE)
        != (WORLD_SIZE, DP_SIZE, SP_SIZE)
        or any(
            globals().get(name) is not value
            for name, value in _PINNED_IMPORT_IDENTITIES.items()
        )
        or globals().get("_PINNED_CONSUME")
        is not source_consumer.consume_graft_a_lite_source_release
        or globals().get("_PINNED_VALIDATE_FOR_TRAINING")
        is not source_consumer.validate_for_training
        or globals().get("_PINNED_OPEN_TRAINER")
        is not short_trainer.open_authenticated_short_training
        or globals().get("_PINNED_SESSION_FAIL")
        is not short_trainer.PhaseAShortTrainingSession._fail
        or globals().get("_PINNED_NATIVE_V2_CLASS")
        is not native_v2.PhaseANativeTrainingClosure
        or ctypes.string_at is not _PINNED_CTYPES_STRING_AT
        or globals().get("short_chunked_tensor_identity")
        is not _PINNED_SHORT_CHUNKED_TENSOR_IDENTITY
        or globals().get("short_chunked_parameter_registry_digest")
        is not _PINNED_SHORT_CHUNKED_REGISTRY_DIGEST
        or MAX_CTYPES_DIGEST_CHUNK_BYTES != 64 * 1024 * 1024
    ):
        raise GraftPhaseAShortGPUError("pinned dependency namespace differs")
    short_trainer._assert_pinned_dependencies()  # noqa: SLF001


@dataclass(frozen=True)
class LocalFamilyRouting:
    dp_arm: int
    family: str
    fit_row: Any = field(repr=False, compare=False)
    confirmation_row: Any = field(repr=False, compare=False)
    fit_iid: str
    confirmation_iid: str
    source_release_result_digest: str
    pinset_digest: str
    routing_digest: str
    test_only: bool


def route_local_family(routing: Any, *, dp_arm: int) -> LocalFamilyRouting:
    """Select only the preregistered fit/confirmation pair for one DP arm."""

    assert_pinned_dependencies()
    if type(dp_arm) is not int or not 0 <= dp_arm < DP_SIZE:
        raise GraftPhaseAShortGPUError("DP arm must be 0 or 1")
    production = type(routing) is source_consumer.TrainerRouting
    test_only = type(routing).__name__ == "_TestRouting" and (
        type(routing).__module__ == short_trainer.__name__
    )
    if not (production or test_only):
        raise GraftPhaseAShortGPUError(
            "runner requires an exact production or authenticated CPU-test routing"
        )
    update_rows = routing.update_rows
    confirmation_rows = routing.confirmation_rows
    source_release_result_digest = _require_sha256(
        routing.source_release_result_digest,
        label="source release result digest",
    )
    pinset_digest = _require_sha256(
        routing.pinset_digest, label="source routing pinset digest"
    )
    routing_digest = _require_sha256(
        routing.routing_digest, label="source routing digest"
    )
    if len(update_rows) != DP_SIZE or len(confirmation_rows) != DP_SIZE:
        raise GraftPhaseAShortGPUError("source routing is not exact 2+2")
    fit = update_rows[dp_arm]
    confirmation = confirmation_rows[dp_arm]
    authority_differs = (
        any(vars(routing.authority).values())
        if production
        else any(routing.authority.values())
    )
    if (
        fit.iid != FIT_IID_BY_DP_ARM[dp_arm]
        or confirmation.iid != CONFIRMATION_IID_BY_DP_ARM[dp_arm]
        or not fit.optimizer_update_allowed
        or fit.optimizer_confirmation_only
        or confirmation.optimizer_update_allowed
        or not confirmation.optimizer_confirmation_only
        or fit.source_sha256 == confirmation.source_sha256
        or type(fit.source_bytes) is not bytes
        or type(confirmation.source_bytes) is not bytes
        or hashlib.sha256(fit.source_bytes).hexdigest() != fit.source_sha256
        or hashlib.sha256(confirmation.source_bytes).hexdigest()
        != confirmation.source_sha256
        or fit.noop_instruction != source_consumer.NOOP_INSTRUCTION
        or confirmation.noop_instruction != source_consumer.NOOP_INSTRUCTION
        or authority_differs
    ):
        raise GraftPhaseAShortGPUError("local fit/confirmation source routing differs")
    return LocalFamilyRouting(
        dp_arm=dp_arm,
        family=FAMILY_BY_DP_ARM[dp_arm],
        fit_row=fit,
        confirmation_row=confirmation,
        fit_iid=fit.iid,
        confirmation_iid=confirmation.iid,
        source_release_result_digest=source_release_result_digest,
        pinset_digest=pinset_digest,
        routing_digest=routing_digest,
        test_only=test_only,
    )


@dataclass(frozen=True)
class ConfirmationFieldSet:
    source_noop_target_velocity: torch.Tensor = field(repr=False, compare=False)
    correct_atlas_noop_velocity: torch.Tensor = field(repr=False, compare=False)
    wrong_atlas_noop_velocity: torch.Tensor = field(repr=False, compare=False)
    dropped_atlas_noop_velocity: torch.Tensor = field(repr=False, compare=False)
    correct_atlas_action_velocity: torch.Tensor = field(repr=False, compare=False)
    dropped_atlas_action_velocity: torch.Tensor = field(repr=False, compare=False)
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class TrajectoryConfirmationPlan:
    """Read-only row binding for pre/post-update trajectory measurements."""

    dp_arm: int
    row: Any = field(repr=False, compare=False)
    wrong_owner_row: Any = field(repr=False, compare=False)
    row_iid: str
    wrong_owner_iid: str
    schedule_indices: tuple[int, ...] = CONFIRMATION_INDICES
    field_roles: tuple[str, ...] = tuple(short_trainer.CONFIRMATION_FIELD_ROLES)


@dataclass(frozen=True)
class ShortGPURunnerResult:
    receipt: Mapping[str, Any]
    checkpoint_payload: None = None
    publication_payload: None = None


def _confirmation_tensors(value: ConfirmationFieldSet) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        (name, getattr(value, name)) for name in short_trainer.CONFIRMATION_FIELD_ROLES
    )


_CONFIRMATION_TRUE_FLAGS = (
    "exactly_six_fields",
    "all_fields_detached_fp32_finite_contiguous",
    "all_field_storages_pairwise_disjoint",
    "same_confirmation_source_zs_bytes_all_fields",
    "same_native_full_source_v_pack_bytes_all_model_fields",
    "same_noisy_target_object_and_bytes_all_fields",
    "same_epsilon_bytes_all_fields",
    "same_sigma_timestep_coordinate_all_fields",
    "same_negative_condition_bytes_all_model_fields",
    "correct_atlas_from_confirmation_row",
    "wrong_atlas_from_same_family_fit_row",
    "wrong_intervention_changes_only_identity_atlas_memory",
    "drop_intervention_disables_only_identity_rebinder_residual_and_atlas_route",
    "drop_retains_native_full_source_v_pack",
    "noop_positive_condition_shared_across_correct_wrong_drop",
    "action_positive_condition_shared_across_correct_drop",
    "action_noop_pair_differs_only_in_positive_text_embedding",
    "source_noop_target_velocity_recomputed_from_same_x_sigma_and_source_zs",
    "native_v2v_apg_field_formula_used",
)
_CONFIRMATION_FALSE_FLAGS = (
    "confirmation_row_consumed_by_optimizer",
    "wrong_atlas_is_cross_family",
    "native_source_v_pack_dropped",
    "negative_condition_changed_by_intervention",
    "noise_or_coordinate_changed_by_intervention",
    "target_video_used",
    "generated_proposal_used",
    "t2v_branch_used",
    "source_retelling_used",
    "selector_used",
    "mask_pose_track_flow_or_motion_donor_used",
)
_PRODUCTION_SAME_STATE_IDENTITY_FIELDS = (
    "confirmation_source_zs",
    "epsilon",
    "noisy_target_x_sigma",
    "native_visual_pack",
    "native_rotary_pack",
    "sigma",
    "timestep",
    "negative_condition",
    "noop_positive_condition",
    "action_positive_condition",
)


def validate_confirmation_field_set(
    value: Any,
    *,
    plan: short_trainer.ConfirmationPlan,
    schedule_index: int,
) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    """Authenticate tensors and the identities that make them same-state fields."""

    if type(value) is not ConfirmationFieldSet:
        raise GraftPhaseAShortGPUError("confirmation factory returned a non-field-set")
    if schedule_index not in CONFIRMATION_INDICES:
        raise GraftPhaseAShortGPUError("confirmation index differs")
    rows = _confirmation_tensors(value)
    if tuple(name for name, _ in rows) != tuple(short_trainer.CONFIRMATION_FIELD_ROLES):
        raise GraftPhaseAShortGPUError("confirmation field order differs")
    first_shape: Optional[tuple[int, ...]] = None
    first_device: Optional[torch.device] = None
    pointers = []
    identities: dict[str, Mapping[str, Any]] = {}
    for name, tensor in rows:
        if (
            type(tensor) is not torch.Tensor
            or tensor.dtype != torch.float32
            or tensor.device.type == "meta"
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
            or tensor.numel() <= 0
        ):
            raise GraftPhaseAShortGPUError(
                f"confirmation field is not detached finite contiguous FP32: {name}"
            )
        shape = tuple(int(item) for item in tensor.shape)
        if first_shape is None:
            first_shape, first_device = shape, tensor.device
        elif shape != first_shape or tensor.device != first_device:
            raise GraftPhaseAShortGPUError("confirmation field geometry differs")
        pointers.append(
            (tensor.device.type, tensor.device.index, int(tensor.untyped_storage().data_ptr()))
        )
        identities[name] = tensor_identity(tensor)
    if len(set(pointers)) != len(rows):
        raise GraftPhaseAShortGPUError("confirmation field storages alias")
    provenance = validate_sealed_mapping(
        value.provenance, label="six-field provenance"
    )
    expected_exact = {
        "schema_version": CONFIRMATION_FIELDS_SCHEMA_VERSION,
        "schedule_index": schedule_index,
        "confirmation_iid": plan.row_iid,
        "confirmation_source_sha256": plan.row.source_sha256,
        "wrong_owner_iid": plan.wrong_owner_iid,
        "wrong_owner_source_sha256": plan.wrong_owner_row.source_sha256,
        "field_roles": list(short_trainer.CONFIRMATION_FIELD_ROLES),
        "field_tensor_identities": identities,
    }
    if any(provenance.get(key) != expected for key, expected in expected_exact.items()):
        raise GraftPhaseAShortGPUError("six-field provenance identity differs")
    if any(provenance.get(name) is not True for name in _CONFIRMATION_TRUE_FLAGS):
        raise GraftPhaseAShortGPUError("six-field same-state provenance is incomplete")
    if any(provenance.get(name) is not False for name in _CONFIRMATION_FALSE_FLAGS):
        raise GraftPhaseAShortGPUError("six-field provenance crossed a denied boundary")
    if any(provenance.get(name) is not False for name in AUTHORITY_FIELDS):
        raise GraftPhaseAShortGPUError("six-field provenance elevated authority")
    if type(plan.row) is source_consumer.TrainerOwnedSourceRow:
        before = provenance.get("same_state_identities_before_model_fields")
        after = provenance.get("same_state_identities_after_all_fields")
        if (
            not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or tuple(before) != _PRODUCTION_SAME_STATE_IDENTITY_FIELDS
            or dict(before) != dict(after)
            or provenance.get(
                "same_state_tensor_identities_recomputed_byte_equal"
            )
            is not True
            or provenance.get(
                "wrong_route_receipts_differ_only_in_atlas_memory"
            )
            is not True
            or provenance.get(
                "drop_route_receipts_retain_v_branch_disable_only_rebinder"
            )
            is not True
            or provenance.get(
                "action_noop_route_receipts_equal_with_negative_raw_reuse"
            )
            is not True
        ):
            raise GraftPhaseAShortGPUError(
                "production six-field runtime identity proof differs"
            )
    return {name: tensor for name, tensor in rows}, value.provenance


def build_confirmation_provenance(
    *,
    plan: short_trainer.ConfirmationPlan,
    schedule_index: int,
    fields: Mapping[str, torch.Tensor],
    runtime_evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Seal production/test evidence; validation still recomputes every field hash."""

    if set(fields) != set(short_trainer.CONFIRMATION_FIELD_ROLES):
        raise GraftPhaseAShortGPUError("six-field builder inventory differs")
    evidence = dict(runtime_evidence)
    forbidden_overlap = (
        set(evidence)
        & (
            set(_CONFIRMATION_TRUE_FLAGS)
            | set(_CONFIRMATION_FALSE_FLAGS)
            | set(AUTHORITY_FIELDS)
        )
    )
    if forbidden_overlap:
        raise GraftPhaseAShortGPUError(
            "runtime evidence may not override fixed confirmation flags"
        )
    return seal_mapping(
        {
            "schema_version": CONFIRMATION_FIELDS_SCHEMA_VERSION,
            "schedule_index": schedule_index,
            "confirmation_iid": plan.row_iid,
            "confirmation_source_sha256": plan.row.source_sha256,
            "wrong_owner_iid": plan.wrong_owner_iid,
            "wrong_owner_source_sha256": plan.wrong_owner_row.source_sha256,
            "field_roles": list(short_trainer.CONFIRMATION_FIELD_ROLES),
            "field_tensor_identities": {
                name: tensor_identity(fields[name])
                for name in short_trainer.CONFIRMATION_FIELD_ROLES
            },
            **evidence,
            **{name: True for name in _CONFIRMATION_TRUE_FLAGS},
            **{name: False for name in _CONFIRMATION_FALSE_FLAGS},
            **_false_authority(),
        }
    )


def validate_adapter_off_parity(value: Any) -> Mapping[str, Any]:
    receipt = validate_sealed_mapping(value, label="adapter-off parity")
    rows = receipt.get("rows")
    if (
        receipt.get("schema_version") != ADAPTER_OFF_PARITY_SCHEMA_VERSION
        or receipt.get("schedule_indices") != list(ADAPTER_OFF_PARITY_INDICES)
        or receipt.get("branch_roles") != list(PARITY_BRANCH_ROLES)
        or not isinstance(rows, list)
        or len(rows) != len(ADAPTER_OFF_PARITY_INDICES) * len(PARITY_BRANCH_ROLES)
        or receipt.get("baseline_captured_before_adapter_install") is not True
        or receipt.get("comparison_executed_after_two_updates_and_confirmation")
        is not True
        or receipt.get("all_installed_zero_gate_raw_bytes_equal_adapter_off") is not True
        or receipt.get("raw_dtype") != "torch.bfloat16"
        or receipt.get("checkpoint_written") is not False
        or any(receipt.get(name) is not False for name in AUTHORITY_FIELDS)
    ):
        raise GraftPhaseAShortGPUError("adapter-off parity receipt differs")
    expected = [
        (index, role)
        for index in ADAPTER_OFF_PARITY_INDICES
        for role in PARITY_BRANCH_ROLES
    ]
    observed = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise GraftPhaseAShortGPUError("adapter-off parity row differs")
        observed.append((row.get("schedule_index"), row.get("branch_role")))
        if (
            row.get("adapter_route_gate_float64_hex") != 0.0.hex()
            or row.get("adapter_off_raw_sha256")
            != row.get("installed_zero_gate_raw_sha256")
            or row.get("raw_storage_byte_exact") is not True
            or row.get("native_full_source_v_pack_bytes_unchanged") is not True
            or row.get("noisy_target_bytes_unchanged") is not True
            or row.get("epsilon_bytes_unchanged") is not True
            or row.get("sigma_timestep_unchanged") is not True
            or row.get("condition_bytes_unchanged") is not True
            or row.get("target_video_used") is not False
        ):
            raise GraftPhaseAShortGPUError("adapter-off BF16 raw row failed parity")
        _require_sha256(row.get("adapter_off_raw_sha256"), label="adapter-off raw")
    if observed != expected:
        raise GraftPhaseAShortGPUError("adapter-off parity row order differs")
    return value


_SERVICES_TOKEN = object()


class AuthenticatedRunnerServices:
    """Opaque callbacks for either the exact official runtime or explicit fakes."""

    __slots__ = (
        "_make_update_cell",
        "_after_update",
        "_make_confirmation_fields",
        "_adapter_off_parity",
        "_callbacks",
        "_test_only",
        "_receipt",
        "_token",
        "_locked",
    )

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise GraftPhaseAShortGPUError("runner services are exact-type only")

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise GraftPhaseAShortGPUError(
            "runner services must be minted by an authenticator"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("AuthenticatedRunnerServices is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def _mint(
        cls,
        *,
        token: object,
        make_update_cell: Callable[..., Any],
        after_update: Callable[..., Any],
        make_confirmation_fields: Callable[..., Any],
        adapter_off_parity: Callable[..., Any],
        test_only: bool,
        receipt: Mapping[str, Any],
    ) -> "AuthenticatedRunnerServices":
        assert_pinned_dependencies()
        if token is not _SERVICES_TOKEN or type(test_only) is not bool:
            raise GraftPhaseAShortGPUError("runner service mint differs")
        callbacks = (
            make_update_cell,
            after_update,
            make_confirmation_fields,
            adapter_off_parity,
        )
        if any(not callable(value) for value in callbacks):
            raise GraftPhaseAShortGPUError("runner service callback is not callable")
        expected_parameters = (
            ("plan",),
            ("plan", "update_receipt"),
            ("plan", "schedule_index"),
            ("schedule_indices",),
        )
        for callback, expected in zip(callbacks, expected_parameters):
            if tuple(inspect.signature(callback).parameters) != expected:
                raise GraftPhaseAShortGPUError(
                    "runner service callback signature differs"
                )
        owned_receipt = validate_sealed_mapping(receipt, label="runner services")
        if (
            owned_receipt.get("schema_version") != SERVICES_SCHEMA_VERSION
            or owned_receipt.get("test_only") is not test_only
            or any(owned_receipt.get(name) is not False for name in AUTHORITY_FIELDS)
        ):
            raise GraftPhaseAShortGPUError("runner services receipt differs")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_make_update_cell", make_update_cell)
        object.__setattr__(instance, "_after_update", after_update)
        object.__setattr__(
            instance, "_make_confirmation_fields", make_confirmation_fields
        )
        object.__setattr__(instance, "_adapter_off_parity", adapter_off_parity)
        object.__setattr__(
            instance,
            "_callbacks",
            tuple(
                (
                    getattr(callback, "__func__", callback),
                    getattr(callback, "__self__", None),
                )
                for callback in callbacks
            ),
        )
        object.__setattr__(instance, "_test_only", test_only)
        object.__setattr__(instance, "_receipt", MappingProxyType(owned_receipt))
        object.__setattr__(instance, "_token", _SERVICES_TOKEN)
        object.__setattr__(instance, "_locked", True)
        instance.assert_live()
        return instance

    @property
    def test_only(self) -> bool:
        return self._test_only

    def assert_live(self) -> None:
        assert_pinned_dependencies()
        callbacks = (
            self._make_update_cell,
            self._after_update,
            self._make_confirmation_fields,
            self._adapter_off_parity,
        )
        observed = tuple(
            (
                getattr(callback, "__func__", callback),
                getattr(callback, "__self__", None),
            )
            for callback in callbacks
        )
        receipt = validate_sealed_mapping(self._receipt, label="runner services")
        if (
            type(self) is not AuthenticatedRunnerServices
            or self._token is not _SERVICES_TOKEN
            or observed != self._callbacks
            or receipt.get("test_only") is not self._test_only
            or any(receipt.get(name) is not False for name in AUTHORITY_FIELDS)
        ):
            raise GraftPhaseAShortGPUError("runner services changed after mint")

    def make_update_cell(
        self, *, plan: short_trainer.UpdateCellPlan
    ) -> native_v2.PhaseANativeTrainingClosure:
        self.assert_live()
        return self._make_update_cell(plan=plan)

    def after_update(
        self,
        *,
        plan: short_trainer.UpdateCellPlan,
        update_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.assert_live()
        return self._after_update(plan=plan, update_receipt=update_receipt)

    def make_confirmation_fields(
        self,
        *,
        plan: short_trainer.ConfirmationPlan,
        schedule_index: int,
    ) -> ConfirmationFieldSet:
        self.assert_live()
        return self._make_confirmation_fields(
            plan=plan, schedule_index=schedule_index
        )

    def adapter_off_parity(
        self, *, schedule_indices: tuple[int, int]
    ) -> Mapping[str, Any]:
        self.assert_live()
        return self._adapter_off_parity(schedule_indices=schedule_indices)

    def receipt(self) -> Mapping[str, Any]:
        self.assert_live()
        return self._receipt


def authenticate_cpu_test_services(
    *,
    test_name: str,
    make_update_cell: Callable[..., Any],
    after_update: Callable[..., Any],
    make_confirmation_fields: Callable[..., Any],
    adapter_off_parity: Callable[..., Any],
) -> AuthenticatedRunnerServices:
    """Mint injectable services that permanently deny production authority."""

    if not isinstance(test_name, str) or _SAFE_TEST_NAME.fullmatch(test_name) is None:
        raise GraftPhaseAShortGPUError("CPU service name must use cpu_fake namespace")
    receipt = seal_mapping(
        {
            "schema_version": SERVICES_SCHEMA_VERSION,
            "binding_label": test_name,
            "test_only": True,
            "official_checkpoint_runtime_bound": False,
            "gpu_execution_authorized": False,
            "callbacks_injected_for_unit_test": True,
            **_false_authority(),
        }
    )
    return AuthenticatedRunnerServices._mint(
        token=_SERVICES_TOKEN,
        make_update_cell=make_update_cell,
        after_update=after_update,
        make_confirmation_fields=make_confirmation_fields,
        adapter_off_parity=adapter_off_parity,
        test_only=True,
        receipt=receipt,
    )


def _rollback_open_session(
    session: short_trainer.PhaseAShortTrainingSession, error: Exception
) -> None:
    assert_pinned_dependencies()
    if type(session) is not short_trainer.PhaseAShortTrainingSession:
        raise GraftPhaseAShortGPUError("rollback session type differs") from error
    if session.phase == "failed":
        return
    if session.phase == "closed":
        raise GraftPhaseAShortGPUError(
            "runner failure occurred after irreversible core finalization"
        ) from error
    _PINNED_SESSION_FAIL(session, error)
    if session.phase != "failed":
        raise GraftPhaseAShortGPUError("short core did not enter failed state") from error


def _runner_failure_receipt(
    *,
    error: Exception,
    trace: Sequence[Mapping[str, Any]],
    session: Optional[short_trainer.PhaseAShortTrainingSession],
    local: LocalFamilyRouting,
    services: AuthenticatedRunnerServices,
) -> Mapping[str, Any]:
    trainer_failure: Optional[Mapping[str, Any]] = None
    if session is not None and session.phase == "failed":
        trainer_failure = session.failure_receipt()
    return seal_mapping(
        {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "status": "failed_rolled_back_no_checkpoint",
            "error": f"{type(error).__name__}:{error}",
            "dp_arm": local.dp_arm,
            "family": local.family,
            "fit_iid": local.fit_iid,
            "confirmation_iid": local.confirmation_iid,
            "trace": [dict(row) for row in trace],
            "trainer_failure_receipt": (
                None if trainer_failure is None else dict(trainer_failure)
            ),
            "runner_services_digest": services.receipt()["digest"],
            "trainable_parameters_rolled_back": (
                trainer_failure is not None
                and trainer_failure.get(
                    "trainable_parameters_restored_to_initial_snapshot"
                )
                is True
            ),
            "checkpoint_written": False,
            "checkpoint_payload_returned": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


def execute_authenticated_short_run(
    *,
    routing: Any,
    bindings: native_v2.AuthenticatedNativeBindings,
    collectives: short_trainer.AuthenticatedDP2SP4Backend,
    services: AuthenticatedRunnerServices,
) -> ShortGPURunnerResult:
    """Execute the fixed two-update, six-field, parity-before-finish protocol."""

    assert_pinned_dependencies()
    services.assert_live()
    if (
        type(bindings) is not native_v2.AuthenticatedNativeBindings
        or type(collectives) is not short_trainer.AuthenticatedDP2SP4Backend
        or services.test_only is not bindings.test_only
        or services.test_only is not collectives.test_only
    ):
        raise GraftPhaseAShortGPUError("runner production/test evidence was mixed")
    local = route_local_family(routing, dp_arm=collectives.dp_arm)
    if local.test_only is not services.test_only:
        raise GraftPhaseAShortGPUError("routing and runner service authority differ")
    session: Optional[short_trainer.PhaseAShortTrainingSession] = None
    trace: list[Mapping[str, Any]] = []
    update_route_receipts: list[Mapping[str, Any]] = []
    confirmation_provenance: list[Mapping[str, Any]] = []
    confirmation_admissions: list[Mapping[str, Any]] = []
    try:
        trace.append({"ordinal": 0, "operation": "open_short_training"})
        session = _PINNED_OPEN_TRAINER(
            routing=routing,
            bindings=bindings,
            collectives=collectives,
        )
        for update_ordinal, expected_index in enumerate(UPDATE_INDICES, start=1):
            plan = session.next_update_plan()
            trace.append(
                {
                    "ordinal": len(trace),
                    "operation": "next_update_plan",
                    "update_number": update_ordinal,
                    "schedule_index": plan.schedule_index,
                    "row_iid": plan.row_iid,
                }
            )
            if (
                plan.update_number != update_ordinal
                or plan.schedule_index != expected_index
                or plan.row is not local.fit_row
                or plan.row_iid != local.fit_iid
            ):
                raise GraftPhaseAShortGPUError("short-core update plan routing differs")
            cell = services.make_update_cell(plan=plan)
            trace.append(
                {
                    "ordinal": len(trace),
                    "operation": "make_native_v2_cell",
                    "update_number": update_ordinal,
                    "schedule_index": expected_index,
                }
            )
            if (
                type(cell) is not _PINNED_NATIVE_V2_CLASS
                or cell.bindings is not bindings
                or cell.schedule_index != expected_index
            ):
                raise GraftPhaseAShortGPUError("service returned the wrong v2 cell")
            update_receipt = session.run_update(plan=plan, cell=cell)
            trace.append(
                {
                    "ordinal": len(trace),
                    "operation": "run_update",
                    "update_number": update_ordinal,
                    "schedule_index": expected_index,
                    "update_receipt_digest": update_receipt["digest"],
                }
            )
            route_receipt = validate_sealed_mapping(
                services.after_update(
                    plan=plan, update_receipt=update_receipt
                ),
                label=f"update {update_ordinal} runner route evidence",
            )
            if (
                route_receipt.get("update_number") != update_ordinal
                or route_receipt.get("schedule_index") != expected_index
                or route_receipt.get("row_iid") != local.fit_iid
                or route_receipt.get("exact_four_native_forwards") is not True
                or route_receipt.get("forward_order")
                != [
                    ["measurement", "negative"],
                    ["measurement", "positive"],
                    ["replay", "negative"],
                    ["replay", "positive"],
                ]
                or route_receipt.get("fit_row_only") is not True
                or route_receipt.get("checkpoint_written") is not False
                or any(route_receipt.get(name) is not False for name in AUTHORITY_FIELDS)
            ):
                raise GraftPhaseAShortGPUError("update route evidence differs")
            update_route_receipts.append(route_receipt)
            trace.append(
                {
                    "ordinal": len(trace),
                    "operation": "admit_update_route_evidence",
                    "update_number": update_ordinal,
                    "schedule_index": expected_index,
                    "route_receipt_digest": route_receipt["digest"],
                }
            )

        plan = session.confirmation_plan()
        trace.append(
            {
                "ordinal": len(trace),
                "operation": "confirmation_plan",
                "row_iid": plan.row_iid,
                "wrong_owner_iid": plan.wrong_owner_iid,
            }
        )
        if (
            plan.row is not local.confirmation_row
            or plan.wrong_owner_row is not local.fit_row
            or plan.row_iid != local.confirmation_iid
            or plan.wrong_owner_iid != local.fit_iid
        ):
            raise GraftPhaseAShortGPUError(
                "confirmation is not held-out row plus same-family fit atlas"
            )
        for schedule_index in CONFIRMATION_INDICES:
            with torch.no_grad():
                field_set = services.make_confirmation_fields(
                    plan=plan, schedule_index=schedule_index
                )
                trace.append(
                    {
                        "ordinal": len(trace),
                        "operation": "measure_six_confirmation_fields",
                        "schedule_index": schedule_index,
                        "row_iid": plan.row_iid,
                        "wrong_owner_iid": plan.wrong_owner_iid,
                    }
                )
                fields, provenance = validate_confirmation_field_set(
                    field_set, plan=plan, schedule_index=schedule_index
                )
                admission = session.record_confirmation_fields(
                    plan=plan,
                    schedule_index=schedule_index,
                    **fields,
                )
            confirmation_provenance.append(provenance)
            confirmation_admissions.append(admission)
            trace.append(
                {
                    "ordinal": len(trace),
                    "operation": "admit_confirmation_fields",
                    "schedule_index": schedule_index,
                    "provenance_digest": provenance["digest"],
                    "admission_digest": admission["digest"],
                }
            )

        parity = validate_adapter_off_parity(
            services.adapter_off_parity(
                schedule_indices=ADAPTER_OFF_PARITY_INDICES
            )
        )
        trace.append(
            {
                "ordinal": len(trace),
                "operation": "admit_adapter_off_bf16_raw_parity",
                "schedule_indices": list(ADAPTER_OFF_PARITY_INDICES),
                "parity_digest": parity["digest"],
            }
        )
        trainer_result = session.finish()
        trace.append(
            {
                "ordinal": len(trace),
                "operation": "finish_in_memory_short_core",
                "trainer_receipt_digest": trainer_result.receipt["digest"],
            }
        )
        if (
            trainer_result.checkpoint_payload is not None
            or trainer_result.publication_payload is not None
            or trainer_result.receipt.get("checkpoint_written") is not False
            or trainer_result.receipt.get("publication_performed") is not False
        ):
            raise GraftPhaseAShortGPUError("short core returned a publication payload")
        final = seal_mapping(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed_in_memory_diagnostic_no_checkpoint",
                "complete": True,
                "topology": {
                    "world_size": WORLD_SIZE,
                    "data_parallel_size": DP_SIZE,
                    "sequence_parallel_size": SP_SIZE,
                    "rank": collectives.rank,
                    "dp_arm": local.dp_arm,
                    "sp_rank": collectives.sp_rank,
                    "family": local.family,
                },
                "dependency_source_sha256": {
                    "a_lite_consumer": PINNED_CONSUMER_SOURCE_SHA256,
                    "native_v2_closure": PINNED_NATIVE_V2_SOURCE_SHA256,
                    "short_trainer": PINNED_SHORT_TRAINER_SOURCE_SHA256,
                    "native_v1_gpu_runner_reuse": (
                        PINNED_NATIVE_RUNNER_V1_SOURCE_SHA256
                    ),
                },
                "source_routing": {
                    "routing_digest": local.routing_digest,
                    "fit_iid": local.fit_iid,
                    "confirmation_iid": local.confirmation_iid,
                    "fit_row_consumed_by_optimizer": True,
                    "confirmation_row_consumed_by_optimizer": False,
                    "wrong_atlas_iid": local.fit_iid,
                    "wrong_atlas_is_same_family_fit_row": True,
                    "owned_source_bytes_only": True,
                    "source_path_reopened_by_runner": False,
                },
                "execution_trace": [dict(row) for row in trace],
                "update_route_receipts": [
                    dict(row) for row in update_route_receipts
                ],
                "confirmation": {
                    "schedule_indices": list(CONFIRMATION_INDICES),
                    "field_roles": list(short_trainer.CONFIRMATION_FIELD_ROLES),
                    "provenance": [dict(row) for row in confirmation_provenance],
                    "admissions": [dict(row) for row in confirmation_admissions],
                    "exact_six_fields_per_index": True,
                    "same_state_interventions_verified": True,
                    "wrong_atlas_same_family_fit_verified": True,
                    "drop_disables_only_identity_rebinder_memory_verified": True,
                },
                "adapter_off_parity": dict(parity),
                "short_trainer_receipt": dict(trainer_result.receipt),
                "training_updates_executed_for_diagnostic": 2,
                "full_sampler_used": False,
                "decoded_media_output_created": False,
                "checkpoint_written": False,
                "checkpoint_payload_returned": False,
                "publication_performed": False,
                "result_staged_in_memory_only": True,
                "target_video_used": False,
                "generated_proposal_used": False,
                "t2v_branch_used": False,
                "source_retelling_used": False,
                "selector_used": False,
                "mask_pose_track_flow_or_motion_donor_used": False,
                **_false_authority(),
            }
        )
        return ShortGPURunnerResult(receipt=final)
    except Exception as error:
        try:
            if session is not None and session.phase != "closed":
                _rollback_open_session(session, error)
            diagnostic = _runner_failure_receipt(
                error=error,
                trace=trace,
                session=session,
                local=local,
                services=services,
            )
        except Exception as cleanup_error:
            diagnostic = seal_mapping(
                {
                    "schema_version": FAILURE_SCHEMA_VERSION,
                    "status": "failed_cleanup_not_authenticated_no_checkpoint",
                    "error": f"{type(error).__name__}:{error}",
                    "cleanup_error": f"{type(cleanup_error).__name__}:{cleanup_error}",
                    "checkpoint_written": False,
                    "publication_performed": False,
                    **_false_authority(),
                }
            )
        raise GraftPhaseAShortGPUError(
            "Phase-A short GPU protocol failed",
            diagnostic_receipt=diagnostic,
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--producer-receipt-path", required=True)
    parser.add_argument("--execution-receipt-path", required=True)
    parser.add_argument("--submission-receipt-path", required=True)
    parser.add_argument("--terminal-admission-path", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--producer-receipt-sha256", required=True)
    parser.add_argument("--execution-receipt-sha256", required=True)
    parser.add_argument("--submission-receipt-sha256", required=True)
    parser.add_argument("--terminal-admission-sha256", required=True)
    parser.add_argument(
        "--terminal-materializer-implementation-sha256", required=True
    )
    parser.add_argument("--terminal-materializer-runtime-sha256", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=EXPECTED_CHECKPOINT_MANIFEST_SHA256,
    )
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-identity-rebinder-sha256", required=True)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument(
        "--ack-two-update-diagnostic-no-checkpoint-no-scientific-claim",
        action="store_true",
    )
    return parser


def validate_cli(args: argparse.Namespace) -> argparse.Namespace:
    """Validate only externally pinned evidence; there is no output path."""

    assert_pinned_dependencies()
    if (
        args.ack_two_update_diagnostic_no_checkpoint_no_scientific_claim
        is not True
    ):
        raise GraftPhaseAShortGPUError(
            "--ack-two-update-diagnostic-no-checkpoint-no-scientific-claim is mandatory"
        )
    for name in (
        "manifest_sha256",
        "producer_receipt_sha256",
        "execution_receipt_sha256",
        "submission_receipt_sha256",
        "terminal_admission_sha256",
        "terminal_materializer_implementation_sha256",
        "terminal_materializer_runtime_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "expected_runner_sha256",
        "expected_identity_rebinder_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        _require_sha256(getattr(args, name), label=name)
    _require_hex40(args.expected_bernini_commit, label="expected Bernini commit")
    _require_hex40(args.expected_veomni_commit, label="expected VeOmni commit")
    if (
        args.expected_bernini_commit != rebinder.PINNED_BERNINI_SOURCE_COMMIT
        or args.expected_veomni_commit != legacy.trainer.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256
        != legacy.trainer.CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != EXPECTED_CHECKPOINT_MANIFEST_SHA256
        or file_sha256(Path(__file__)) != args.expected_runner_sha256
        or file_sha256(Path(rebinder.__file__))
        != args.expected_identity_rebinder_sha256
    ):
        raise GraftPhaseAShortGPUError("CLI source/checkpoint pin differs")
    for name in (
        "manifest_path",
        "producer_receipt_path",
        "execution_receipt_path",
        "submission_receipt_path",
        "terminal_admission_path",
        "checkpoint_content_manifest",
    ):
        value = Path(getattr(args, name))
        if not value.is_absolute():
            raise GraftPhaseAShortGPUError(f"{name} must be an absolute path")
    public_fields = tuple(vars(args))
    lowered = tuple(name.lower() for name in public_fields)
    if any(
        fragment in name
        for fragment in _FORBIDDEN_PUBLIC_INPUT_FRAGMENTS
        for name in lowered
    ):
        raise GraftPhaseAShortGPUError("runner CLI acquired a forbidden input")
    return args


def _read_exact_0444_file(path_value: str | Path, *, expected_sha256: str) -> bytes:
    expected = _require_sha256(expected_sha256, label="sealed file SHA256")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        raise GraftPhaseAShortGPUError("sealed file path must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise GraftPhaseAShortGPUError("sealed file cannot be resolved") from error
    if resolved != candidate:
        raise GraftPhaseAShortGPUError("sealed file path is not its exact realpath")
    descriptor = os.open(
        candidate,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
        ):
            raise GraftPhaseAShortGPUError(
                "sealed file must be regular mode-0444 link-count-one"
            )
        blocks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        raw = b"".join(blocks)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_mode,
            value.st_nlink,
        )
        if identity(before) != identity(after):
            raise GraftPhaseAShortGPUError("sealed file changed while read")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise GraftPhaseAShortGPUError("sealed file SHA256 differs")
        return raw
    finally:
        os.close(descriptor)


def consume_authenticated_source_routing(args: argparse.Namespace) -> Any:
    """Run the sole production consumer and its sole trainer-routing mint."""

    validate_cli(args)
    pins = source_consumer.ReleaseArtifactPins(
        manifest_sha256=args.manifest_sha256,
        producer_receipt_sha256=args.producer_receipt_sha256,
        execution_receipt_sha256=args.execution_receipt_sha256,
        submission_receipt_sha256=args.submission_receipt_sha256,
        terminal_admission_sha256=args.terminal_admission_sha256,
        terminal_materializer_implementation_sha256=(
            args.terminal_materializer_implementation_sha256
        ),
        terminal_materializer_runtime_sha256=(
            args.terminal_materializer_runtime_sha256
        ),
    )
    terminal = _read_exact_0444_file(
        args.terminal_admission_path,
        expected_sha256=args.terminal_admission_sha256,
    )
    release = _PINNED_CONSUME(
        manifest_path=args.manifest_path,
        producer_receipt_path=args.producer_receipt_path,
        execution_receipt_path=args.execution_receipt_path,
        submission_receipt_path=args.submission_receipt_path,
        terminal_admission_bytes=terminal,
        pins=pins,
    )
    routing = _PINNED_VALIDATE_FOR_TRAINING(release)
    if type(routing) is not source_consumer.TrainerRouting:
        raise GraftPhaseAShortGPUError("consumer did not mint exact trainer routing")
    route_local_family(routing, dp_arm=0)
    route_local_family(routing, dp_arm=1)
    return routing


def _gather_equal(
    value: Any, *, group: Any, count: int, label: str
) -> list[bytes]:
    import torch.distributed as dist

    # ``seal_mapping`` deliberately returns a read-only ``mappingproxy``.
    # PyTorch object collectives pickle their payload, and mappingproxy is not
    # pickleable.  Canonicalize before crossing the collective boundary so the
    # transmitted object is both pickle-safe and the exact equality witness.
    payload = canonical_json_bytes(value)
    rows: list[Any] = [None] * count
    dist.all_gather_object(rows, payload, group=group)
    if any(type(row) is not bytes or row != payload for row in rows):
        raise GraftPhaseAShortGPUError(f"{label} differs across ranks")
    return rows  # type: ignore[return-value]


@dataclass(frozen=True)
class ExactCoordinate:
    schedule_index: int
    sigma: torch.Tensor = field(repr=False, compare=False)
    timestep: torch.Tensor = field(repr=False, compare=False)
    receipt: Mapping[str, Any]


class Exact40CoordinateRegistry:
    """One immutable audit of the real scheduler with device-local lookups."""

    def __init__(self, scheduler: Any, *, device: torch.device) -> None:
        audit = sigma_strata.audit_runtime_unipc_schedule(
            scheduler, initialize=True
        )
        timesteps = getattr(scheduler, "timesteps", None)
        sigmas = getattr(scheduler, "sigmas", None)
        if (
            type(timesteps) is not torch.Tensor
            or timesteps.dtype != torch.int64
            or timesteps.device.type != "cpu"
            or tuple(timesteps.shape) != (40,)
            or type(sigmas) is not torch.Tensor
            or sigmas.dtype != torch.float32
            or sigmas.device.type != "cpu"
            or tuple(sigmas.shape) != (41,)
            or getattr(scheduler, "step_index", None) is not None
        ):
            raise GraftPhaseAShortGPUError("audited exact40 scheduler storage differs")
        self._scheduler = scheduler
        self._device = device
        self._timesteps = timesteps.detach().clone()
        self._sigmas = sigmas.detach().clone()
        self._state = {
            "timesteps": tensor_identity(timesteps),
            "sigmas": tensor_identity(sigmas),
            "step_index": None,
        }
        self.receipt = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-short-exact40-registry-v1",
                "audit": audit,
                "schedule_sha256": sigma_strata.SCHEDULE_SHA256,
                "scheduler_step_called": False,
                "allowed_update_indices": list(UPDATE_INDICES),
                "allowed_confirmation_indices": list(CONFIRMATION_INDICES),
                "mandatory_adapter_off_parity_indices": list(
                    ADAPTER_OFF_PARITY_INDICES
                ),
            }
        )

    def coordinate(self, schedule_index: int) -> ExactCoordinate:
        if type(schedule_index) is not int or schedule_index not in range(40):
            raise GraftPhaseAShortGPUError("exact40 coordinate index differs")
        sigma = self._sigmas[schedule_index].detach().clone().reshape(())
        timestep = self._timesteps[
            schedule_index : schedule_index + 1
        ].to(device=self._device).contiguous()
        expected_hex = sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
            schedule_index
        ]
        if (
            struct.pack(">f", float(sigma.item())).hex() != expected_hex
            or int(timestep.item())
            != sigma_strata.PINNED_TIMESTEPS[schedule_index]
        ):
            raise GraftPhaseAShortGPUError("exact40 sigma/timestep binding differs")
        return ExactCoordinate(
            schedule_index=schedule_index,
            sigma=sigma,
            timestep=timestep,
            receipt=seal_mapping(
                {
                    "schema_version": "bernini-graft-phase-a-short-coordinate-v1",
                    "schedule_index": schedule_index,
                    "timestep": int(timestep.item()),
                    "sigma_float32_be_hex": expected_hex,
                    "schedule_sha256": sigma_strata.SCHEDULE_SHA256,
                    "scheduler_step_called": False,
                }
            ),
        )

    def assert_unchanged(self) -> None:
        current = {
            "timesteps": tensor_identity(getattr(self._scheduler, "timesteps", None)),
            "sigmas": tensor_identity(getattr(self._scheduler, "sigmas", None)),
            "step_index": getattr(self._scheduler, "step_index", None),
        }
        if current != self._state:
            raise GraftPhaseAShortGPUError("scheduler changed in a no-step runner")


@dataclass(frozen=True)
class KeyedNoise:
    epsilon: torch.Tensor = field(repr=False, compare=False)
    receipt: Mapping[str, Any]


def keyed_fresh_gaussian(
    *,
    shape: Sequence[int],
    device: torch.device,
    source_sha256: str,
    purpose: str,
    schedule_index: int,
    base_seed: int = NOISE_BASE_SEED,
) -> KeyedNoise:
    dimensions = tuple(int(item) for item in shape)
    _require_sha256(source_sha256, label="noise source SHA256")
    if (
        not dimensions
        or any(item <= 0 for item in dimensions)
        or not isinstance(purpose, str)
        or not purpose.isascii()
        or len(purpose) > 160
        or type(schedule_index) is not int
        or schedule_index not in range(40)
        or type(base_seed) is not int
        or not 0 <= base_seed < 2**63
    ):
        raise GraftPhaseAShortGPUError("keyed Gaussian fields differ")
    key = {
        "schema_version": "bernini-graft-phase-a-short-keyed-gaussian-v1",
        "source_sha256": source_sha256,
        "purpose": purpose,
        "schedule_index": schedule_index,
        "base_seed": base_seed,
        "shape": list(dimensions),
        "dtype": "torch.float32",
        "generator_device": "cpu",
    }
    key_digest = object_sha256(key)
    seed = int.from_bytes(bytes.fromhex(key_digest[:16]), "big") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epsilon = torch.randn(
        dimensions,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device).contiguous()
    if (
        epsilon.requires_grad
        or epsilon.grad_fn is not None
        or not bool(torch.isfinite(epsilon).all().item())
    ):
        raise GraftPhaseAShortGPUError("keyed Gaussian tensor differs")
    return KeyedNoise(
        epsilon=epsilon,
        receipt=seal_mapping(
            {
                **key,
                "key_digest": key_digest,
                "derived_seed": seed,
                "fresh_per_source_purpose_index": True,
                "source_or_target_derived": False,
                "tensor": tensor_identity(epsilon),
            }
        ),
    )


@dataclass(frozen=True)
class SourceState:
    row: Any = field(repr=False, compare=False)
    source_tensor: torch.Tensor = field(repr=False, compare=False)
    source_latent: torch.Tensor = field(repr=False, compare=False)
    atlas_frames: torch.Tensor = field(repr=False, compare=False)
    metadata: Mapping[str, Any]
    receipt: Mapping[str, Any]


def _decode_owned_source_bytes(row: Any) -> tuple[torch.Tensor, Mapping[str, Any]]:
    raw = row.source_bytes
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != row.source_sha256:
        raise GraftPhaseAShortGPUError("owned source bytes changed before decode")
    with tempfile.TemporaryDirectory(prefix="graft-a-lite-owned-decode-") as root:
        path = Path(root) / "owned-source.mp4"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise GraftPhaseAShortGPUError("private source write stalled")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        source_tensor, metadata, observed_sha = (
            source_audit.prepare_hashed_source_snapshot(path)
        )
    if observed_sha != row.source_sha256 or hashlib.sha256(raw).hexdigest() != observed_sha:
        raise GraftPhaseAShortGPUError("owned source decode observed different bytes")
    if (
        tuple(source_tensor.shape[:3]) != (1, 3, FRAME_COUNT)
        or source_tensor.dtype != torch.float32
        or source_tensor.requires_grad
        or not source_tensor.is_contiguous()
        or not bool(torch.isfinite(source_tensor).all().item())
        or metadata.get("frame_count") != FRAME_COUNT
        or float(metadata.get("reported_fps", -1.0)) != 25.0
    ):
        raise GraftPhaseAShortGPUError("owned source exact81 decode differs")
    if type(row) is source_consumer.TrainerOwnedSourceRow:
        input_hw = tuple(int(item) for item in metadata["source_input_hw"])
        if (
            input_hw != (row.media.height, row.media.width)
            or row.media.frame_count != FRAME_COUNT
            or (row.media.fps_numerator, row.media.fps_denominator) != (25, 1)
        ):
            raise GraftPhaseAShortGPUError("consumer media and decoded source differ")
    return source_tensor, MappingProxyType(dict(metadata))


def _encode_source_state(
    *,
    row: Any,
    vae: torch.nn.Module,
    vae_encode: Callable[..., torch.Tensor],
    device: torch.device,
    sp_group: Any,
) -> SourceState:
    source_tensor, metadata = _decode_owned_source_bytes(row)
    pixels = source_tensor.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        latent = vae_encode(vae, pixels).float().contiguous()
    bucket = tuple(int(item) for item in metadata["source_derived_bucket_hw"])
    expected_shape = (1, 16, LATENT_PHASES, bucket[0] // 8, bucket[1] // 8)
    if (
        tuple(latent.shape) != expected_shape
        or latent.requires_grad
        or latent.grad_fn is not None
        or not bool(torch.isfinite(latent).all().item())
    ):
        raise GraftPhaseAShortGPUError("frozen source zS geometry differs")
    import torch.distributed as dist

    sp_members: list[Any] = [None] * SP_SIZE
    dist.all_gather_object(sp_members, int(dist.get_rank()), group=sp_group)
    if (
        any(type(rank) is not int for rank in sp_members)
        or len(set(sp_members)) != SP_SIZE
    ):
        raise GraftPhaseAShortGPUError("source-state SP4 membership differs")
    dist.broadcast(latent, src=sp_members[0], group=sp_group)
    _gather_equal(
        tensor_identity(latent), group=sp_group, count=SP_SIZE, label=f"{row.iid} zS"
    )
    atlas_frames, atlas_receipt = native_runner_v1.prepare_atlas_source_frames(
        source_tensor, device=device
    )
    _gather_equal(
        tensor_identity(atlas_frames),
        group=sp_group,
        count=SP_SIZE,
        label=f"{row.iid} atlas frames",
    )
    receipt = seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-owned-source-state-v1",
            "iid": row.iid,
            "owned_source_sha256": row.source_sha256,
            "owned_source_bytes_size": len(row.source_bytes),
            "decoded_from_private_owned_bytes_snapshot": True,
            "original_source_path_reopened": False,
            "exact81_25fps": True,
            "source_metadata": dict(metadata),
            "source_tensor": tensor_identity(source_tensor),
            "source_latent_zs": tensor_identity(latent),
            "atlas_frames": tensor_identity(atlas_frames),
            "atlas_view_receipt_digest": atlas_receipt["digest"],
            "rank0_of_sp4_broadcast_then_byte_exact": True,
            "target_video_used": False,
        }
    )
    return SourceState(
        row=row,
        source_tensor=source_tensor,
        source_latent=latent,
        atlas_frames=atlas_frames,
        metadata=metadata,
        receipt=receipt,
    )


@dataclass(frozen=True)
class NativePack:
    visual: torch.Tensor = field(repr=False, compare=False)
    rotary: torch.Tensor = field(repr=False, compare=False)
    source_tokens: int
    target_tokens: int
    visual_identity: Mapping[str, Any]
    rotary_identity: Mapping[str, Any]


def _build_native_pack(
    *,
    transformer: torch.nn.Module,
    source_latent: torch.Tensor,
    noisy_target: torch.Tensor,
) -> NativePack:
    if source_latent.shape != noisy_target.shape or source_latent is noisy_target:
        raise GraftPhaseAShortGPUError("native pack source/target geometry differs")
    patched = []
    context: AbstractContextManager[Any] = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if source_latent.device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), context:
        for source_id, value in zip((1.0, 0.0), (source_latent, noisy_target)):
            result = transformer.patch_vae_latent(
                hidden_states=value.to(dtype=torch.bfloat16), source_id=source_id
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise GraftPhaseAShortGPUError("native patch result differs")
            patched.append((result[0].detach(), result[1].detach()))
    source_tokens, source_rotary = patched[0]
    target_tokens, target_rotary = patched[1]
    token_count = LATENT_PHASES * (int(noisy_target.shape[3]) // 2) * (
        int(noisy_target.shape[4]) // 2
    )
    if (
        tuple(source_tokens.shape) != (1, token_count, 1536)
        or source_tokens.shape != target_tokens.shape
        or source_tokens.dtype != torch.bfloat16
        or target_tokens.dtype != torch.bfloat16
        or tuple(source_rotary.shape[:3]) != (1, 1, token_count)
        or source_rotary.shape != target_rotary.shape
    ):
        raise GraftPhaseAShortGPUError("native V-pack patch geometry differs")
    visual = torch.cat((source_tokens, target_tokens), dim=1).detach().contiguous()
    rotary = torch.cat((source_rotary, target_rotary), dim=2).detach().contiguous()
    if visual.requires_grad or rotary.requires_grad:
        raise GraftPhaseAShortGPUError("native V-pack retained a graph")
    return NativePack(
        visual=visual,
        rotary=rotary,
        source_tokens=token_count,
        target_tokens=token_count,
        visual_identity=tensor_identity(visual),
        rotary_identity=tensor_identity(rotary),
    )


def _native_raw_forward(
    *,
    diffusion: torch.nn.Module,
    pack: NativePack,
    coordinate: ExactCoordinate,
    condition: torch.Tensor,
    route_context: AbstractContextManager[Any],
) -> torch.Tensor:
    autocast: AbstractContextManager[Any] = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if pack.visual.device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast, route_context:
        raw = diffusion.shared_step(
            model_id="transformer_1",
            noisy_latents=pack.visual,
            timesteps=coordinate.timestep,
            cond_embeds=condition,
            rotary_embs=pack.rotary,
            batch_vae_seqlen=[pack.source_tokens + pack.target_tokens],
            batch_text_seqlen=[int(condition.shape[1])],
        )
    if (
        type(raw) is not torch.Tensor
        or tuple(raw.shape)
        != (1, pack.source_tokens + pack.target_tokens, 64)
        or raw.dtype != torch.bfloat16
        or raw.requires_grad
        or raw.grad_fn is not None
        or not bool(torch.isfinite(raw).all().item())
    ):
        raise GraftPhaseAShortGPUError("native shared_step raw output differs")
    return raw.detach().clone().contiguous()


def _guided_velocity_from_raw(
    *,
    bindings: native_v2.AuthenticatedNativeBindings,
    noisy_target: torch.Tensor,
    coordinate: ExactCoordinate,
    pack: NativePack,
    negative_raw: torch.Tensor,
    positive_raw: torch.Tensor,
) -> torch.Tensor:
    spatial = []
    for raw in (negative_raw, positive_raw):
        tail = raw[:, -pack.target_tokens :, :]
        spatial_raw = native_v2.unpack_wan_target_velocity(
            tail, spatial_shape=noisy_target.shape
        ).detach().contiguous()
        clean = (noisy_target - coordinate.sigma * spatial_raw).detach().contiguous()
        if clean.dtype != torch.float32:
            raise GraftPhaseAShortGPUError("native raw-to-clean order changed dtype")
        spatial.append(clean)
    momentum = bindings.momentum_buffer_factory(native_v2.APG_MOMENTUM)
    if (
        float(getattr(momentum, "momentum", math.nan)) != 0.0
        or float(getattr(momentum, "running_average", math.nan)) != 0.0
    ):
        raise GraftPhaseAShortGPUError("vendor APG momentum is not fresh zero")
    with torch.no_grad():
        guided = bindings.vendor_normalized_guidance(
            pred_cond=spatial[1],
            pred_uncond=spatial[0],
            guidance_scale=native_v2.GUIDANCE_SCALE,
            momentum_buffer=momentum,
            eta=native_v2.APG_ETA,
            norm_threshold=native_v2.APG_NORM_THRESHOLD,
        )
        velocity = ((noisy_target - guided) / coordinate.sigma).float().contiguous()
    if (
        type(velocity) is not torch.Tensor
        or velocity.dtype != torch.float32
        or velocity.shape != noisy_target.shape
        or velocity.requires_grad
        or velocity.grad_fn is not None
        or not bool(torch.isfinite(velocity).all().item())
    ):
        raise GraftPhaseAShortGPUError("vendor APG velocity differs")
    return velocity.detach().clone().contiguous()


class ShortTrainingAtlasRouteFactory:
    """Fresh graph-bearing fit atlas for each of four native closure forwards."""

    _EXPECTED = (
        ("measurement", "negative", False),
        ("measurement", "positive", False),
        ("replay", "negative", True),
        ("replay", "positive", True),
    )

    def __init__(
        self,
        *,
        handle: rebinder.IdentityRebinderHandle,
        sp_rank: int,
        sp_group: Any,
    ) -> None:
        if (
            not isinstance(handle, rebinder.IdentityRebinderHandle)
            or type(sp_rank) is not int
            or not 0 <= sp_rank < SP_SIZE
        ):
            raise GraftPhaseAShortGPUError("training route factory inputs differ")
        self.handle = handle
        self.sp_rank = sp_rank
        self.sp_group = sp_group
        self._pending: Optional[dict[str, Any]] = None
        self._completed: list[Mapping[str, Any]] = []

    def begin(
        self,
        *,
        update_number: int,
        schedule_index: int,
        row_iid: str,
        row_source_sha256: str,
        source_frames: torch.Tensor,
    ) -> None:
        if (
            self._pending is not None
            or update_number not in (1, 2)
            or schedule_index != UPDATE_INDICES[update_number - 1]
        ):
            raise GraftPhaseAShortGPUError("training route transaction differs")
        _require_sha256(row_source_sha256, label="route source SHA256")
        if (
            type(source_frames) is not torch.Tensor
            or source_frames.dtype != torch.float32
            or tuple(source_frames.shape[:3]) != (1, FRAME_COUNT, 3)
            or source_frames.requires_grad
            or not source_frames.is_contiguous()
            or not bool(torch.isfinite(source_frames).all().item())
        ):
            raise GraftPhaseAShortGPUError("training route source frames differ")
        self._pending = {
            "update_number": update_number,
            "schedule_index": schedule_index,
            "row_iid": row_iid,
            "row_source_sha256": row_source_sha256,
            "source_frames": source_frames,
            "source_frames_identity": tensor_identity(source_frames),
            "rows": [],
            "atlas_objects": [],
        }

    def __call__(
        self, *, request: native_v2.NativeForwardContextRequest
    ) -> AbstractContextManager[Any]:
        pending = self._pending
        if pending is None or not isinstance(
            request, native_v2.NativeForwardContextRequest
        ):
            raise GraftPhaseAShortGPUError("training route lacks an open transaction")
        rows = pending["rows"]
        position = len(rows)
        if position >= len(self._EXPECTED):
            raise GraftPhaseAShortGPUError("training route received extra forward")
        phase, role, graph_expected = self._EXPECTED[position]
        if (
            (request.phase, request.role) != (phase, role)
            or request.schedule_index != pending["schedule_index"]
            or torch.is_grad_enabled() is not graph_expected
            or tensor_identity(pending["source_frames"])
            != pending["source_frames_identity"]
        ):
            raise GraftPhaseAShortGPUError("training route call order/state differs")
        atlas = self.handle.build_atlas(
            pending["source_frames"],
            source_video_sha256=pending["row_source_sha256"],
        )
        if (
            not isinstance(atlas, rebinder.IdentityAtlas)
            or atlas.tokens.requires_grad is not graph_expected
            or (atlas.tokens.grad_fn is not None) is not graph_expected
            or any(atlas.tokens is prior for prior in pending["atlas_objects"])
        ):
            raise GraftPhaseAShortGPUError("training route fresh atlas differs")
        pending["atlas_objects"].append(atlas.tokens)
        atlas_identity = tensor_identity(atlas.tokens)
        _gather_equal(
            atlas_identity,
            group=self.sp_group,
            count=SP_SIZE,
            label=f"update{pending['update_number']} {phase} {role} atlas",
        )
        route = rebinder.IdentityRebinderRoute(
            total_tokens=request.total_tokens,
            condition_tokens=request.condition_tokens,
            sequence_parallel_rank=self.sp_rank,
            sequence_parallel_size=SP_SIZE,
            branch_name="V",
            sigma=request.sigma,
            atlas=atlas,
            enabled=True,
        )
        selector = route.local_target_selector(device=atlas.tokens.device)
        local_target_rows = int(torch.count_nonzero(selector).item())
        observation = native_v2.build_native_forward_context_observation(
            request=request,
            sequence_parallel_rank=self.sp_rank,
            sequence_parallel_size=SP_SIZE,
            local_target_selector=selector,
            route_gate=route.gate,
            adapter_graph_bearing=(
                graph_expected and local_target_rows > 0 and route.gate > 0.0
            ),
        )
        rows.append(
            {
                "ordinal": position,
                "phase": phase,
                "role": role,
                "graph_expected": graph_expected,
                "schedule_index": request.schedule_index,
                "route_gate_float64_hex": float(route.gate).hex(),
                "local_target_rows": local_target_rows,
                "adapter_graph_bearing": observation.adapter_graph_bearing,
                "atlas_tokens": atlas_identity,
                "atlas_receipt_digest": atlas.receipt()["digest"],
                "fresh_atlas_object": True,
            }
        )

        @contextmanager
        def entered() -> Any:
            with self.handle.route(route):
                yield observation

        return entered()

    def finish(
        self,
        *,
        plan: short_trainer.UpdateCellPlan,
        update_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        pending = self._pending
        if pending is None:
            raise GraftPhaseAShortGPUError("training route has no transaction to finish")
        rows = pending["rows"]
        observed = [(row["phase"], row["role"], row["graph_expected"]) for row in rows]
        expected = list(self._EXPECTED)
        if (
            observed != expected
            or plan.update_number != pending["update_number"]
            or plan.schedule_index != pending["schedule_index"]
            or plan.row_iid != pending["row_iid"]
            or update_receipt.get("schedule_index") != pending["schedule_index"]
            or len({row["atlas_tokens"]["content_sha256"] for row in rows}) != 1
        ):
            raise GraftPhaseAShortGPUError("training route completion differs")
        receipt = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-short-update-route-v1",
                "update_number": pending["update_number"],
                "schedule_index": pending["schedule_index"],
                "row_iid": pending["row_iid"],
                "row_source_sha256": pending["row_source_sha256"],
                "fit_row_only": True,
                "exact_four_native_forwards": True,
                "forward_order": [[row[0], row[1]] for row in self._EXPECTED],
                "fresh_atlas_per_forward": True,
                "measurement_atlas_detached": True,
                "replay_atlas_graph_bearing_only_on_target_owner": True,
                "rows": [dict(row) for row in rows],
                "checkpoint_written": False,
                **_false_authority(),
            }
        )
        self._completed.append(receipt)
        self._pending = None
        return receipt


def _official_forward_route_receipt() -> Mapping[str, Any]:
    return seal_mapping(
        {
            "schema_version": native_v2.FORWARD_ROUTE_SCHEMA_VERSION,
            "route_kind": "identity_rebinder_v1",
            "phase_a_active_schedule_indices": list(
                native_v2.PHASE_A_ACTIVE_SCHEDULE_INDICES
            ),
            "inactive_schedule_policy": "exact_zero_update_not_trained",
            "target_queries_only": True,
            "condition_rows_written": False,
            "external_oracle_inputs": False,
            "factory": "fresh_fit_identity_atlas_per_native_forward",
            "update_schedule_indices": list(UPDATE_INDICES),
            "confirmation_and_parity_not_routed_through_training_closure": True,
        }
    )


def _route_for_pack(
    *,
    handle: rebinder.IdentityRebinderHandle,
    pack: NativePack,
    sp_rank: int,
    sigma: float,
    atlas: Optional[rebinder.IdentityAtlas],
    mode: str,
) -> tuple[rebinder.IdentityRebinderRoute, AbstractContextManager[Any]]:
    if mode == "atlas":
        branch_name, enabled = "V", True
        if atlas is None:
            raise GraftPhaseAShortGPUError("atlas route lacks memory")
    elif mode == "drop":
        # The native model call remains the same full-source V-pack.  Only the
        # IdentityRebinder residual/memory route is disabled.
        branch_name, enabled, atlas = "V", False, None
    else:
        raise GraftPhaseAShortGPUError("identity route mode differs")
    route = rebinder.IdentityRebinderRoute(
        total_tokens=pack.source_tokens + pack.target_tokens,
        condition_tokens=pack.source_tokens,
        sequence_parallel_rank=sp_rank,
        sequence_parallel_size=SP_SIZE,
        branch_name=branch_name,
        sigma=sigma,
        atlas=atlas,
        enabled=enabled,
    )
    return route, handle.route(route)


def _capture_adapter_off_baseline(
    *,
    diffusion: torch.nn.Module,
    transformer: torch.nn.Module,
    schedule: Exact40CoordinateRegistry,
    confirmation: SourceState,
    negative_condition: torch.Tensor,
    noop_condition: torch.Tensor,
    action_condition: torch.Tensor,
) -> Mapping[str, Any]:
    rows = []
    conditions = {
        "negative": negative_condition,
        "noop_positive": noop_condition,
        "action_positive": action_condition,
    }
    for index in ADAPTER_OFF_PARITY_INDICES:
        coordinate = schedule.coordinate(index)
        noise = keyed_fresh_gaussian(
            shape=confirmation.source_latent.shape,
            device=confirmation.source_latent.device,
            source_sha256=confirmation.row.source_sha256,
            purpose="adapter-off-parity",
            schedule_index=index,
        )
        noisy, noisy_receipt = native_runner_v1.build_noisy_target(
            confirmation.source_latent,
            noise.epsilon,
            sigma=coordinate.sigma,
        )
        pack = _build_native_pack(
            transformer=transformer,
            source_latent=confirmation.source_latent,
            noisy_target=noisy,
        )
        for role in PARITY_BRANCH_ROLES:
            raw = _native_raw_forward(
                diffusion=diffusion,
                pack=pack,
                coordinate=coordinate,
                condition=conditions[role],
                route_context=nullcontext(),
            )
            identity = tensor_identity(raw)
            rows.append(
                {
                    "schedule_index": index,
                    "branch_role": role,
                    "coordinate_digest": coordinate.receipt["digest"],
                    "epsilon": tensor_identity(noise.epsilon),
                    "epsilon_receipt_digest": noise.receipt["digest"],
                    "noisy_target": tensor_identity(noisy),
                    "noisy_target_receipt_digest": noisy_receipt["digest"],
                    "visual_pack": pack.visual_identity,
                    "rotary_pack": pack.rotary_identity,
                    "condition": tensor_identity(conditions[role]),
                    "adapter_off_raw": identity,
                    "adapter_off_raw_sha256": identity["content_sha256"],
                }
            )
    return seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-short-preinstall-baseline-v1",
            "confirmation_iid": confirmation.row.iid,
            "schedule_indices": list(ADAPTER_OFF_PARITY_INDICES),
            "branch_roles": list(PARITY_BRANCH_ROLES),
            "captured_before_adapter_install": True,
            "raw_dtype": "torch.bfloat16",
            "rows": rows,
            "target_video_used": False,
            **_false_authority(),
        }
    )


class OfficialShortRuntime:
    """Live model state used only through authenticated runner services."""

    def __init__(
        self,
        *,
        local: LocalFamilyRouting,
        bindings: native_v2.AuthenticatedNativeBindings,
        diffusion: torch.nn.Module,
        transformer: torch.nn.Module,
        handle: rebinder.IdentityRebinderHandle,
        route_factory: ShortTrainingAtlasRouteFactory,
        schedule: Exact40CoordinateRegistry,
        fit: SourceState,
        confirmation: SourceState,
        negative_condition: torch.Tensor,
        noop_condition: torch.Tensor,
        action_condition: torch.Tensor,
        sp_rank: int,
        sp_group: Any,
        adapter_off_baseline: Mapping[str, Any],
    ) -> None:
        if (
            local.test_only
            or bindings.test_only
            or fit.row is not local.fit_row
            or confirmation.row is not local.confirmation_row
            or sp_rank not in range(SP_SIZE)
        ):
            raise GraftPhaseAShortGPUError("official short runtime inputs differ")
        self.local = local
        self.bindings = bindings
        self.diffusion = diffusion
        self.transformer = transformer
        self.handle = handle
        self.route_factory = route_factory
        self.schedule = schedule
        self.fit = fit
        self.confirmation = confirmation
        self.negative_condition = negative_condition
        self.noop_condition = noop_condition
        self.action_condition = action_condition
        self.sp_rank = sp_rank
        self.sp_group = sp_group
        self.adapter_off_baseline = adapter_off_baseline
        self._update_inputs: dict[int, Mapping[str, Any]] = {}
        self._confirmation_seen: list[int] = []

    def make_update_cell(
        self, *, plan: short_trainer.UpdateCellPlan
    ) -> native_v2.PhaseANativeTrainingClosure:
        if (
            plan.row is not self.fit.row
            or plan.row_iid != self.local.fit_iid
            or plan.schedule_index not in UPDATE_INDICES
            or plan.schedule_index in self._update_inputs
        ):
            raise GraftPhaseAShortGPUError("official update plan differs")
        coordinate = self.schedule.coordinate(plan.schedule_index)
        noise = keyed_fresh_gaussian(
            shape=self.fit.source_latent.shape,
            device=self.fit.source_latent.device,
            source_sha256=self.fit.row.source_sha256,
            purpose=f"optimizer-update-{plan.update_number}",
            schedule_index=plan.schedule_index,
        )
        noisy, noisy_receipt = native_runner_v1.build_noisy_target(
            self.fit.source_latent,
            noise.epsilon,
            sigma=coordinate.sigma,
        )
        self.route_factory.begin(
            update_number=plan.update_number,
            schedule_index=plan.schedule_index,
            row_iid=plan.row_iid,
            row_source_sha256=plan.row_source_sha256,
            source_frames=self.fit.atlas_frames,
        )
        self._update_inputs[plan.schedule_index] = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-short-update-input-v1",
                "update_number": plan.update_number,
                "schedule_index": plan.schedule_index,
                "row_iid": plan.row_iid,
                "source_state_receipt_digest": self.fit.receipt["digest"],
                "coordinate_digest": coordinate.receipt["digest"],
                "epsilon_receipt_digest": noise.receipt["digest"],
                "noisy_target_receipt_digest": noisy_receipt["digest"],
                "positive_condition_role": "canonical_source_noop_r2v",
                "negative_condition_role": "pinned_renderer_negative",
                "target_video_used": False,
            }
        )
        return native_v2.PhaseANativeTrainingClosure(
            bindings=self.bindings,
            source_video=self.fit.source_latent,
            noisy_target=noisy,
            negative_condition=self.negative_condition,
            positive_condition=self.noop_condition,
            schedule_index=plan.schedule_index,
            sigma=coordinate.sigma,
            timestep=coordinate.timestep,
        )

    def after_update(
        self,
        *,
        plan: short_trainer.UpdateCellPlan,
        update_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        route = self.route_factory.finish(
            plan=plan, update_receipt=update_receipt
        )
        base = dict(route)
        base.pop("digest")
        base["update_input_digest"] = self._update_inputs[plan.schedule_index][
            "digest"
        ]
        return seal_mapping(base)

    def _atlas(self, state: SourceState) -> rebinder.IdentityAtlas:
        with torch.no_grad():
            atlas = self.handle.build_atlas(
                state.atlas_frames,
                source_video_sha256=state.row.source_sha256,
            )
        if atlas.tokens.requires_grad or atlas.tokens.grad_fn is not None:
            raise GraftPhaseAShortGPUError("confirmation atlas retained a graph")
        _gather_equal(
            tensor_identity(atlas.tokens),
            group=self.sp_group,
            count=SP_SIZE,
            label=f"confirmation atlas {state.row.iid}",
        )
        return atlas

    def _raw_for_mode(
        self,
        *,
        pack: NativePack,
        coordinate: ExactCoordinate,
        condition: torch.Tensor,
        atlas: Optional[rebinder.IdentityAtlas],
        mode: str,
    ) -> tuple[torch.Tensor, Mapping[str, Any]]:
        route, context = _route_for_pack(
            handle=self.handle,
            pack=pack,
            sp_rank=self.sp_rank,
            sigma=float(coordinate.sigma.item()),
            atlas=atlas,
            mode=mode,
        )
        raw = _native_raw_forward(
            diffusion=self.diffusion,
            pack=pack,
            coordinate=coordinate,
            condition=condition,
            route_context=context,
        )
        return raw, route.receipt()

    def make_confirmation_fields(
        self,
        *,
        plan: short_trainer.ConfirmationPlan,
        schedule_index: int,
    ) -> ConfirmationFieldSet:
        if (
            plan.row is not self.confirmation.row
            or plan.wrong_owner_row is not self.fit.row
            or schedule_index not in CONFIRMATION_INDICES
            or schedule_index in self._confirmation_seen
            or torch.is_grad_enabled()
        ):
            raise GraftPhaseAShortGPUError("official confirmation request differs")
        coordinate = self.schedule.coordinate(schedule_index)
        noise = keyed_fresh_gaussian(
            shape=self.confirmation.source_latent.shape,
            device=self.confirmation.source_latent.device,
            source_sha256=self.confirmation.row.source_sha256,
            purpose="optimizer-confirmation",
            schedule_index=schedule_index,
        )
        noisy, noisy_receipt = native_runner_v1.build_noisy_target(
            self.confirmation.source_latent,
            noise.epsilon,
            sigma=coordinate.sigma,
        )
        pack = _build_native_pack(
            transformer=self.transformer,
            source_latent=self.confirmation.source_latent,
            noisy_target=noisy,
        )
        correct_atlas = self._atlas(self.confirmation)
        wrong_atlas = self._atlas(self.fit)

        def same_state_identities() -> dict[str, Any]:
            return {
                "confirmation_source_zs": tensor_identity(
                    self.confirmation.source_latent
                ),
                "epsilon": tensor_identity(noise.epsilon),
                "noisy_target_x_sigma": tensor_identity(noisy),
                "native_visual_pack": tensor_identity(pack.visual),
                "native_rotary_pack": tensor_identity(pack.rotary),
                "sigma": tensor_identity(coordinate.sigma),
                "timestep": tensor_identity(coordinate.timestep),
                "negative_condition": tensor_identity(self.negative_condition),
                "noop_positive_condition": tensor_identity(self.noop_condition),
                "action_positive_condition": tensor_identity(
                    self.action_condition
                ),
            }

        same_state_before = same_state_identities()
        atlas_identities_before = {
            "correct_confirmation_atlas": tensor_identity(correct_atlas.tokens),
            "wrong_same_family_fit_atlas": tensor_identity(wrong_atlas.tokens),
        }

        correct_negative, correct_route = self._raw_for_mode(
            pack=pack,
            coordinate=coordinate,
            condition=self.negative_condition,
            atlas=correct_atlas,
            mode="atlas",
        )
        correct_noop, correct_noop_route = self._raw_for_mode(
            pack=pack,
            coordinate=coordinate,
            condition=self.noop_condition,
            atlas=correct_atlas,
            mode="atlas",
        )
        correct_action, correct_action_route = self._raw_for_mode(
            pack=pack,
            coordinate=coordinate,
            condition=self.action_condition,
            atlas=correct_atlas,
            mode="atlas",
        )
        wrong_negative, wrong_route = self._raw_for_mode(
            pack=pack,
            coordinate=coordinate,
            condition=self.negative_condition,
            atlas=wrong_atlas,
            mode="atlas",
        )
        wrong_noop, wrong_noop_route = self._raw_for_mode(
            pack=pack,
            coordinate=coordinate,
            condition=self.noop_condition,
            atlas=wrong_atlas,
            mode="atlas",
        )
        drop_negative, drop_route = self._raw_for_mode(
            pack=pack,
            coordinate=coordinate,
            condition=self.negative_condition,
            atlas=None,
            mode="drop",
        )
        drop_noop, drop_noop_route = self._raw_for_mode(
            pack=pack,
            coordinate=coordinate,
            condition=self.noop_condition,
            atlas=None,
            mode="drop",
        )
        drop_action, drop_action_route = self._raw_for_mode(
            pack=pack,
            coordinate=coordinate,
            condition=self.action_condition,
            atlas=None,
            mode="drop",
        )

        target = ((noisy - self.confirmation.source_latent) / coordinate.sigma).float()
        fields = {
            "source_noop_target_velocity": target.detach().clone().contiguous(),
            "correct_atlas_noop_velocity": _guided_velocity_from_raw(
                bindings=self.bindings,
                noisy_target=noisy,
                coordinate=coordinate,
                pack=pack,
                negative_raw=correct_negative,
                positive_raw=correct_noop,
            ),
            "wrong_atlas_noop_velocity": _guided_velocity_from_raw(
                bindings=self.bindings,
                noisy_target=noisy,
                coordinate=coordinate,
                pack=pack,
                negative_raw=wrong_negative,
                positive_raw=wrong_noop,
            ),
            "dropped_atlas_noop_velocity": _guided_velocity_from_raw(
                bindings=self.bindings,
                noisy_target=noisy,
                coordinate=coordinate,
                pack=pack,
                negative_raw=drop_negative,
                positive_raw=drop_noop,
            ),
            "correct_atlas_action_velocity": _guided_velocity_from_raw(
                bindings=self.bindings,
                noisy_target=noisy,
                coordinate=coordinate,
                pack=pack,
                negative_raw=correct_negative,
                positive_raw=correct_action,
            ),
            "dropped_atlas_action_velocity": _guided_velocity_from_raw(
                bindings=self.bindings,
                noisy_target=noisy,
                coordinate=coordinate,
                pack=pack,
                negative_raw=drop_negative,
                positive_raw=drop_action,
            ),
        }
        same_state_after = same_state_identities()
        atlas_identities_after = {
            "correct_confirmation_atlas": tensor_identity(correct_atlas.tokens),
            "wrong_same_family_fit_atlas": tensor_identity(wrong_atlas.tokens),
        }
        if torch.equal(self.noop_condition, self.action_condition):
            raise GraftPhaseAShortGPUError("noop/action positive conditions alias")
        correct_routes_equal = (
            correct_route == correct_noop_route == correct_action_route
        )
        wrong_routes_equal = wrong_route == wrong_noop_route
        drop_routes_equal = drop_route == drop_noop_route == drop_action_route

        def without_atlas(value: Mapping[str, Any]) -> dict[str, Any]:
            plain = validate_sealed_mapping(value, label="confirmation route")
            plain.pop("digest")
            plain.pop("atlas_receipt_digest")
            return plain

        wrong_only_atlas = (
            without_atlas(correct_route) == without_atlas(wrong_route)
            and correct_route["atlas_receipt_digest"]
            == correct_atlas.receipt()["digest"]
            and wrong_route["atlas_receipt_digest"]
            == wrong_atlas.receipt()["digest"]
            and correct_route["atlas_receipt_digest"]
            != wrong_route["atlas_receipt_digest"]
        )
        correct_without_adapter = without_atlas(correct_route)
        drop_without_adapter = without_atlas(drop_route)
        for key in ("enabled", "gate_hex"):
            correct_without_adapter.pop(key)
            drop_without_adapter.pop(key)
        drop_only_rebinder = (
            correct_without_adapter == drop_without_adapter
            and drop_route["branch_name"] == "V"
            and drop_route["enabled"] is False
            and drop_route["atlas_receipt_digest"] is None
            and drop_route["gate_hex"] == 0.0.hex()
        )
        if (
            tuple(same_state_before) != _PRODUCTION_SAME_STATE_IDENTITY_FIELDS
            or same_state_before != same_state_after
            or atlas_identities_before != atlas_identities_after
            or not correct_routes_equal
            or not wrong_routes_equal
            or not drop_routes_equal
            or not wrong_only_atlas
            or not drop_only_rebinder
        ):
            raise GraftPhaseAShortGPUError(
                "confirmation model-field same-state identity changed"
            )
        runtime_evidence = {
            "confirmation_source_state_receipt_digest": self.confirmation.receipt[
                "digest"
            ],
            "wrong_fit_source_state_receipt_digest": self.fit.receipt["digest"],
            "coordinate": dict(coordinate.receipt),
            "epsilon": tensor_identity(noise.epsilon),
            "epsilon_receipt_digest": noise.receipt["digest"],
            "noisy_target": tensor_identity(noisy),
            "noisy_target_receipt_digest": noisy_receipt["digest"],
            "confirmation_source_zs": tensor_identity(
                self.confirmation.source_latent
            ),
            "native_visual_pack": pack.visual_identity,
            "native_rotary_pack": pack.rotary_identity,
            "negative_condition": tensor_identity(self.negative_condition),
            "noop_positive_condition": tensor_identity(self.noop_condition),
            "action_positive_condition": tensor_identity(self.action_condition),
            "correct_atlas": tensor_identity(correct_atlas.tokens),
            "wrong_atlas": tensor_identity(wrong_atlas.tokens),
            "same_state_identities_before_model_fields": same_state_before,
            "same_state_identities_after_all_fields": same_state_after,
            "same_state_tensor_identities_recomputed_byte_equal": True,
            "atlas_identities_before_model_fields": atlas_identities_before,
            "atlas_identities_after_all_fields": atlas_identities_after,
            "wrong_route_receipts_differ_only_in_atlas_memory": True,
            "drop_route_receipts_retain_v_branch_disable_only_rebinder": True,
            "action_noop_route_receipts_equal_with_negative_raw_reuse": True,
            "native_raw_call_order": [
                "correct_negative",
                "correct_noop_positive",
                "correct_action_positive",
                "wrong_negative",
                "wrong_noop_positive",
                "drop_negative",
                "drop_noop_positive",
                "drop_action_positive",
            ],
            "negative_raw_reused_for_correct_noop_and_action": True,
            "negative_raw_reused_for_drop_noop_and_action": True,
            "route_receipts": {
                "correct_negative": correct_route,
                "correct_noop": correct_noop_route,
                "correct_action": correct_action_route,
                "wrong_negative": wrong_route,
                "wrong_noop": wrong_noop_route,
                "drop_negative": drop_route,
                "drop_noop": drop_noop_route,
                "drop_action": drop_action_route,
            },
            "raw_tensor_identities": {
                "correct_negative": tensor_identity(correct_negative),
                "correct_noop": tensor_identity(correct_noop),
                "correct_action": tensor_identity(correct_action),
                "wrong_negative": tensor_identity(wrong_negative),
                "wrong_noop": tensor_identity(wrong_noop),
                "drop_negative": tensor_identity(drop_negative),
                "drop_noop": tensor_identity(drop_noop),
                "drop_action": tensor_identity(drop_action),
            },
            "ambient_torch_no_grad": True,
        }
        provenance = build_confirmation_provenance(
            plan=plan,
            schedule_index=schedule_index,
            fields=fields,
            runtime_evidence=runtime_evidence,
        )
        self._confirmation_seen.append(schedule_index)
        return ConfirmationFieldSet(
            **fields,
            provenance=provenance,
        )

    def adapter_off_parity(
        self, *, schedule_indices: tuple[int, int]
    ) -> Mapping[str, Any]:
        if (
            schedule_indices != ADAPTER_OFF_PARITY_INDICES
            or self._confirmation_seen != list(CONFIRMATION_INDICES)
        ):
            raise GraftPhaseAShortGPUError("adapter-off parity execution order differs")
        baseline = validate_sealed_mapping(
            self.adapter_off_baseline, label="preinstall adapter-off baseline"
        )
        baseline_rows = {
            (row["schedule_index"], row["branch_role"]): row
            for row in baseline["rows"]
        }
        conditions = {
            "negative": self.negative_condition,
            "noop_positive": self.noop_condition,
            "action_positive": self.action_condition,
        }
        rows = []
        for index in schedule_indices:
            coordinate = self.schedule.coordinate(index)
            noise = keyed_fresh_gaussian(
                shape=self.confirmation.source_latent.shape,
                device=self.confirmation.source_latent.device,
                source_sha256=self.confirmation.row.source_sha256,
                purpose="adapter-off-parity",
                schedule_index=index,
            )
            noisy, _ = native_runner_v1.build_noisy_target(
                self.confirmation.source_latent,
                noise.epsilon,
                sigma=coordinate.sigma,
            )
            pack = _build_native_pack(
                transformer=self.transformer,
                source_latent=self.confirmation.source_latent,
                noisy_target=noisy,
            )
            atlas = self._atlas(self.confirmation)
            for role in PARITY_BRANCH_ROLES:
                route, context = _route_for_pack(
                    handle=self.handle,
                    pack=pack,
                    sp_rank=self.sp_rank,
                    sigma=float(coordinate.sigma.item()),
                    atlas=atlas,
                    mode="atlas",
                )
                raw = _native_raw_forward(
                    diffusion=self.diffusion,
                    pack=pack,
                    coordinate=coordinate,
                    condition=conditions[role],
                    route_context=context,
                )
                observed = tensor_identity(raw)
                before = baseline_rows[(index, role)]
                row = {
                    "schedule_index": index,
                    "branch_role": role,
                    "adapter_route_gate_float64_hex": float(route.gate).hex(),
                    "adapter_off_raw_sha256": before["adapter_off_raw_sha256"],
                    "installed_zero_gate_raw_sha256": observed[
                        "content_sha256"
                    ],
                    "raw_storage_byte_exact": (
                        before["adapter_off_raw"] == observed
                    ),
                    "native_full_source_v_pack_bytes_unchanged": (
                        before["visual_pack"] == pack.visual_identity
                        and before["rotary_pack"] == pack.rotary_identity
                    ),
                    "noisy_target_bytes_unchanged": (
                        before["noisy_target"] == tensor_identity(noisy)
                    ),
                    "epsilon_bytes_unchanged": (
                        before["epsilon"] == tensor_identity(noise.epsilon)
                    ),
                    "sigma_timestep_unchanged": (
                        before["coordinate_digest"] == coordinate.receipt["digest"]
                    ),
                    "condition_bytes_unchanged": (
                        before["condition"] == tensor_identity(conditions[role])
                    ),
                    "target_video_used": False,
                }
                if not all(
                    row[key] is True
                    for key in (
                        "raw_storage_byte_exact",
                        "native_full_source_v_pack_bytes_unchanged",
                        "noisy_target_bytes_unchanged",
                        "epsilon_bytes_unchanged",
                        "sigma_timestep_unchanged",
                        "condition_bytes_unchanged",
                    )
                ) or row["adapter_route_gate_float64_hex"] != 0.0.hex():
                    raise GraftPhaseAShortGPUError(
                        f"adapter-off parity failed at index {index} role {role}"
                    )
                rows.append(row)
        self.schedule.assert_unchanged()
        return seal_mapping(
            {
                "schema_version": ADAPTER_OFF_PARITY_SCHEMA_VERSION,
                "schedule_indices": list(schedule_indices),
                "branch_roles": list(PARITY_BRANCH_ROLES),
                "baseline_captured_before_adapter_install": True,
                "comparison_executed_after_two_updates_and_confirmation": True,
                "all_installed_zero_gate_raw_bytes_equal_adapter_off": True,
                "raw_dtype": "torch.bfloat16",
                "rows": rows,
                "scheduler_unchanged": True,
                "checkpoint_written": False,
                **_false_authority(),
            }
        )


def _trajectory_plan(local: LocalFamilyRouting) -> TrajectoryConfirmationPlan:
    return TrajectoryConfirmationPlan(
        dp_arm=local.dp_arm,
        row=local.confirmation_row,
        wrong_owner_row=local.fit_row,
        row_iid=local.confirmation_iid,
        wrong_owner_iid=local.fit_iid,
    )


def _trajectory_digests(
    *,
    plan: TrajectoryConfirmationPlan,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    admitted = validate_sealed_mapping(provenance, label="trajectory provenance")
    routes = admitted.get("route_receipts")
    if not isinstance(routes, Mapping) or tuple(routes) != (
        "correct_negative",
        "correct_noop",
        "correct_action",
        "wrong_negative",
        "wrong_noop",
        "drop_negative",
        "drop_noop",
        "drop_action",
    ):
        raise GraftPhaseAShortGPUError("trajectory route inventory differs")
    route_digests = {
        name: validate_sealed_mapping(value, label=f"trajectory route {name}")[
            "digest"
        ]
        for name, value in routes.items()
    }
    correct_atlas = admitted.get("correct_atlas")
    wrong_atlas = admitted.get("wrong_atlas")
    if not isinstance(correct_atlas, Mapping) or not isinstance(wrong_atlas, Mapping):
        raise GraftPhaseAShortGPUError("trajectory atlas identity differs")
    owner_record = {
        "dp_arm": plan.dp_arm,
        "confirmation_iid": plan.row_iid,
        "confirmation_source_sha256": plan.row.source_sha256,
        "wrong_owner_iid": plan.wrong_owner_iid,
        "wrong_owner_source_sha256": plan.wrong_owner_row.source_sha256,
        "wrong_owner_is_same_family_fit_row": True,
    }
    atlas_record = {
        "correct_confirmation_atlas": dict(correct_atlas),
        "wrong_same_family_fit_atlas": dict(wrong_atlas),
    }
    same_state_record = {
        "coordinate": admitted.get("coordinate"),
        "epsilon_receipt_digest": admitted.get("epsilon_receipt_digest"),
        "noisy_target_receipt_digest": admitted.get(
            "noisy_target_receipt_digest"
        ),
        "confirmation_source_state_receipt_digest": admitted.get(
            "confirmation_source_state_receipt_digest"
        ),
        "wrong_fit_source_state_receipt_digest": admitted.get(
            "wrong_fit_source_state_receipt_digest"
        ),
    }
    return seal_mapping(
        {
            "schema_version": (
                "bernini-graft-phase-a-short-trajectory-owner-atlas-route-v1"
            ),
            "owner_digest": object_sha256(owner_record),
            "atlas_digest": object_sha256(atlas_record),
            "route_digest": object_sha256(route_digests),
            "same_state_digest": object_sha256(same_state_record),
            "owner": owner_record,
            "atlas": atlas_record,
            "route_digests": route_digests,
            "same_state": same_state_record,
            "checkpoint_written": False,
            **_false_authority(),
        }
    )


def _measure_trajectory_stage(
    *,
    runtime: OfficialShortRuntime,
    plan: TrajectoryConfirmationPlan,
    collectives: short_trainer.AuthenticatedDP2SP4Backend,
    stage: str,
    parameter_digest: str,
) -> tuple[Mapping[str, Any], dict[int, dict[str, torch.Tensor]]]:
    """Measure both fixed coordinates without admitting or weakening a gate."""

    if stage not in TRAJECTORY_STAGES:
        raise GraftPhaseAShortGPUError("trajectory stage differs")
    prior = tuple(runtime._confirmation_seen)  # noqa: SLF001
    if prior not in ((), CONFIRMATION_INDICES):
        raise GraftPhaseAShortGPUError("trajectory stage boundary differs")
    runtime._confirmation_seen.clear()  # noqa: SLF001
    before = short_chunked_parameter_registry_digest(
        runtime.bindings.named_trainable_parameters
    )
    if before != parameter_digest:
        raise GraftPhaseAShortGPUError("trajectory parameter digest differs")
    collectives.consensus(
        parameter_digest,
        scope="world",
        label=f"trajectory {stage} parameter registry",
    )
    cells = []
    retained: dict[int, dict[str, torch.Tensor]] = {}
    for schedule_index in CONFIRMATION_INDICES:
        with torch.no_grad():
            field_set = runtime.make_confirmation_fields(
                plan=plan, schedule_index=schedule_index
            )
            fields, provenance = validate_confirmation_field_set(
                field_set, plan=plan, schedule_index=schedule_index
            )
            metrics = validate_sealed_mapping(
                short_trainer._confirmation_field_metrics(  # noqa: SLF001
                    schedule_index=schedule_index,
                    **fields,
                ),
                label=f"trajectory {stage} index {schedule_index} metrics",
            )
        digest_packet = _trajectory_digests(plan=plan, provenance=provenance)
        local_evidence = {
            "sp_rank": collectives.sp_rank,
            "metrics_digest": metrics["digest"],
            "provenance_digest": provenance["digest"],
            "owner_digest": digest_packet["owner_digest"],
            "atlas_digest": digest_packet["atlas_digest"],
            "route_digest": digest_packet["route_digest"],
            "same_state_digest": digest_packet["same_state_digest"],
        }
        import torch.distributed as dist

        sp_evidence: list[Any] = [None] * SP_SIZE
        dist.all_gather_object(
            sp_evidence,
            local_evidence,
            group=runtime.sp_group,
        )
        if (
            [row.get("sp_rank") for row in sp_evidence]
            != list(range(SP_SIZE))
            or len({row.get("metrics_digest") for row in sp_evidence}) != 1
            or len({row.get("owner_digest") for row in sp_evidence}) != 1
            or len({row.get("atlas_digest") for row in sp_evidence}) != 1
            or len({row.get("same_state_digest") for row in sp_evidence}) != 1
        ):
            raise GraftPhaseAShortGPUError(
                "trajectory SP4 common evidence differs"
            )
        sp_manifest = seal_mapping(
            {
                "schema_version": (
                    "bernini-graft-phase-a-short-trajectory-sp4-consensus-v1"
                ),
                "dp_arm": plan.dp_arm,
                "stage": stage,
                "schedule_index": schedule_index,
                "ordered_local_evidence": sp_evidence,
                "metrics_owner_atlas_same_state_equal_across_sp4": True,
                "rank_local_route_digests_preserved": True,
            }
        )
        record = {
            "stage": stage,
            "schedule_index": schedule_index,
            "dp_arm": plan.dp_arm,
            "confirmation_iid": plan.row_iid,
            "wrong_owner_iid": plan.wrong_owner_iid,
            "parameter_digest": parameter_digest,
            "metrics_digest": metrics["digest"],
            "sp4_evidence_manifest_digest": sp_manifest["digest"],
        }
        consensus_digest = object_sha256(record)
        collectives.consensus(
            consensus_digest,
            scope="sp",
            label=f"trajectory {stage} index {schedule_index}",
        )
        cells.append(
            seal_mapping(
                {
                    "schema_version": (
                        "bernini-graft-phase-a-short-trajectory-cell-v1"
                    ),
                    **record,
                    "metrics": metrics,
                    "provenance": dict(provenance),
                    "owner_atlas_route": dict(digest_packet),
                    "sp4_evidence_manifest": dict(sp_manifest),
                    "sp4_consensus_digest": consensus_digest,
                    "sp4_exact_consensus": True,
                    "same_noise_within_correct_wrong_drop": True,
                    "original_thresholds_used_without_change": True,
                    "optimizer_update_performed_by_measurement": False,
                    "checkpoint_written": False,
                    **_false_authority(),
                }
            )
        )
        if stage == "after_update_38":
            retained[schedule_index] = fields
    after = short_chunked_parameter_registry_digest(
        runtime.bindings.named_trainable_parameters
    )
    if after != before:
        raise GraftPhaseAShortGPUError("trajectory measurement mutated parameters")
    return (
        seal_mapping(
            {
                "schema_version": TRAJECTORY_MATRIX_SCHEMA_VERSION,
                "stage": stage,
                "parameter_digest_before": before,
                "parameter_digest_after": after,
                "parameter_bytes_unchanged_by_measurement": True,
                "schedule_indices": list(CONFIRMATION_INDICES),
                "cells": [dict(cell) for cell in cells],
                "all_cells_sp4_exact_consensus": True,
                "checkpoint_written": False,
                **_false_authority(),
            }
        ),
        retained,
    )


def _validated_final_gate_vector(value: Any, *, label: str) -> dict[str, bool]:
    if not isinstance(value, Mapping) or tuple(value) != tuple(
        str(index) for index in CONFIRMATION_INDICES
    ):
        raise GraftPhaseAShortGPUError(f"{label} coordinate differs")
    result = dict(value)
    if any(type(result[str(index)]) is not bool for index in CONFIRMATION_INDICES):
        raise GraftPhaseAShortGPUError(f"{label} must contain exact booleans")
    return result


def _validated_final_metrics_digests(
    value: Any, *, label: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or tuple(value) != tuple(
        str(index) for index in CONFIRMATION_INDICES
    ):
        raise GraftPhaseAShortGPUError(f"{label} coordinate differs")
    result = dict(value)
    for index in CONFIRMATION_INDICES:
        _require_sha256(result[str(index)], label=f"{label} index{index}")
    return result


def _admit_world8_terminal_gate_packets(
    packets: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Authenticate one pre-terminal gate packet from every WORLD8 rank."""

    if not isinstance(packets, (list, tuple)) or len(packets) != WORLD_SIZE:
        raise GraftPhaseAShortGPUError("terminal gate WORLD8 coverage differs")
    admitted = []
    for rank, raw in enumerate(packets):
        packet = validate_sealed_mapping(
            raw, label=f"terminal gate rank{rank} packet"
        )
        arm, sp_rank = divmod(rank, SP_SIZE)
        vector = _validated_final_gate_vector(
            packet.get("final_gate_vector"),
            label=f"terminal gate rank{rank} vector",
        )
        digests = _validated_final_metrics_digests(
            packet.get("final_metrics_digests"),
            label=f"terminal gate rank{rank} metric digests",
        )
        if (
            packet.get("schema_version")
            != TERMINAL_GATE_PACKET_SCHEMA_VERSION
            or packet.get("global_rank") != rank
            or packet.get("dp_arm") != arm
            or packet.get("sp_rank") != sp_rank
            or packet.get("family") != FAMILY_BY_DP_ARM[arm]
            or packet.get("local_gate_pass") is not all(vector.values())
            or packet.get("gate_observed_before_original_admission") is not True
            or packet.get("checkpoint_written") is not False
            or packet.get("publication_performed") is not False
        ):
            raise GraftPhaseAShortGPUError("terminal gate packet differs")
        _assert_no_elevated_authority_or_checkpoint(packet)
        admitted.append(packet)

    family_rows = []
    for arm in range(DP_SIZE):
        rows = admitted[arm * SP_SIZE : (arm + 1) * SP_SIZE]
        vectors = {
            canonical_json_bytes(row["final_gate_vector"]) for row in rows
        }
        metric_sets = {
            canonical_json_bytes(row["final_metrics_digests"]) for row in rows
        }
        local_passes = {row["local_gate_pass"] for row in rows}
        if len(vectors) != 1 or len(metric_sets) != 1 or len(local_passes) != 1:
            raise GraftPhaseAShortGPUError(
                "terminal gate SP4 evidence differs within family"
            )
        family_rows.append(
            {
                "dp_arm": arm,
                "family": FAMILY_BY_DP_ARM[arm],
                "global_ranks": list(range(arm * SP_SIZE, (arm + 1) * SP_SIZE)),
                "final_gate_vector": dict(rows[0]["final_gate_vector"]),
                "final_metrics_digests": dict(rows[0]["final_metrics_digests"]),
                "local_gate_pass": rows[0]["local_gate_pass"],
                "all_four_sp_ranks_equal": True,
            }
        )
    world_any_no_go = any(not row["local_gate_pass"] for row in family_rows)
    return seal_mapping(
        {
            "schema_version": TERMINAL_COORDINATION_SCHEMA_VERSION,
            "rank_order": list(range(WORLD_SIZE)),
            "dp2_family_order": list(FAMILY_BY_DP_ARM),
            "ordered_gate_packets": [dict(row) for row in admitted],
            "family_gate_vectors": family_rows,
            "world_any_no_go": world_any_no_go,
            "world_all_pass": not world_any_no_go,
            "terminal_collective_protocol": (
                "global_no_go_all_rank_rollback_no_finish"
                if world_any_no_go
                else "all_pass_original_finish_all_ranks"
            ),
            "gate_gather_scope": "world8",
            "gate_gather_completed_before_original_admissions": True,
            "all_eight_gate_packets_authenticated": True,
            "family_sp4_gate_vectors_equal": True,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


def _validate_world8_terminal_gate_manifest(
    value: Any, *, label: str
) -> Mapping[str, Any]:
    admitted = validate_sealed_mapping(value, label=label)
    rebuilt = _admit_world8_terminal_gate_packets(
        admitted.get("ordered_gate_packets")
    )
    if dict(rebuilt) != admitted:
        raise GraftPhaseAShortGPUError(f"{label} deep admission differs")
    return admitted


def _world8_all_gather_authenticated(
    *,
    collectives: short_trainer.AuthenticatedDP2SP4Backend,
    value: Mapping[str, Any],
    label: str,
) -> list[Mapping[str, Any]]:
    """Use the already authenticated backend's exact WORLD8 process group."""

    assert_pinned_dependencies()
    if (
        type(collectives) is not short_trainer.AuthenticatedDP2SP4Backend
        or collectives.test_only
    ):
        raise GraftPhaseAShortGPUError(f"{label} backend differs")
    collectives.assert_live()
    import torch.distributed as dist

    world_group = object.__getattribute__(collectives, "_world_group")
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or int(dist.get_rank(group=world_group)) != collectives.rank
        or int(dist.get_world_size(group=world_group)) != WORLD_SIZE
    ):
        raise GraftPhaseAShortGPUError(f"{label} WORLD8 geometry differs")
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, dict(value), group=world_group)
    if any(not isinstance(row, Mapping) for row in gathered):
        raise GraftPhaseAShortGPUError(f"{label} gathered a non-mapping")
    return gathered


def _coordinate_world8_terminal_gate(
    *,
    collectives: short_trainer.AuthenticatedDP2SP4Backend,
    local: LocalFamilyRouting,
    final_gate_vector: Mapping[str, bool],
    final_metrics_digests: Mapping[str, str],
) -> Mapping[str, Any]:
    vector = _validated_final_gate_vector(
        final_gate_vector, label="local terminal gate vector"
    )
    metric_digests = _validated_final_metrics_digests(
        final_metrics_digests, label="local terminal metric digests"
    )
    local_packet = seal_mapping(
        {
            "schema_version": TERMINAL_GATE_PACKET_SCHEMA_VERSION,
            "global_rank": collectives.rank,
            "dp_arm": local.dp_arm,
            "sp_rank": collectives.sp_rank,
            "family": local.family,
            "final_gate_vector": vector,
            "final_metrics_digests": metric_digests,
            "local_gate_pass": all(vector.values()),
            "gate_observed_before_original_admission": True,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )
    return _admit_world8_terminal_gate_packets(
        _world8_all_gather_authenticated(
            collectives=collectives,
            value=local_packet,
            label="pre-terminal gate coordination",
        )
    )


def _admit_world8_terminal_rollback_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    gate_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate all-rank rollback after any family reports NO-GO."""

    gate = _validate_world8_terminal_gate_manifest(
        gate_manifest, label="terminal rollback gate manifest"
    )
    if gate.get("world_any_no_go") is not True:
        raise GraftPhaseAShortGPUError("terminal rollback lacks global NO-GO")
    if not isinstance(packets, (list, tuple)) or len(packets) != WORLD_SIZE:
        raise GraftPhaseAShortGPUError("terminal rollback WORLD8 coverage differs")
    admitted = []
    for rank, raw in enumerate(packets):
        packet = validate_sealed_mapping(
            raw, label=f"terminal rollback rank{rank} packet"
        )
        arm, sp_rank = divmod(rank, SP_SIZE)
        expected_local_pass = gate["ordered_gate_packets"][rank][
            "local_gate_pass"
        ]
        expected_path = (
            "local_gate_pass_global_no_go_rolled_back"
            if expected_local_pass
            else "local_gate_fail_global_no_go_rolled_back"
        )
        if (
            packet.get("schema_version")
            != TERMINAL_ROLLBACK_PACKET_SCHEMA_VERSION
            or packet.get("global_rank") != rank
            or packet.get("dp_arm") != arm
            or packet.get("sp_rank") != sp_rank
            or packet.get("family") != FAMILY_BY_DP_ARM[arm]
            or packet.get("local_gate_pass") is not expected_local_pass
            or packet.get("world_any_no_go") is not True
            or packet.get("local_terminal_path") != expected_path
            or _require_sha256(
                packet.get("initial_parameter_digest"),
                label=f"rollback rank{rank} initial digest",
            )
            != packet.get("terminal_parameter_digest")
            or _require_sha256(
                packet.get("trainer_failure_receipt_digest"),
                label=f"rollback rank{rank} failure receipt digest",
            )
            != packet.get("trainer_failure_receipt_digest")
            or packet.get("session_phase_after_rollback") != "failed"
            or packet.get("finish_called") is not False
            or packet.get("checkpoint_written") is not False
            or packet.get("publication_performed") is not False
        ):
            raise GraftPhaseAShortGPUError("terminal rollback packet differs")
        _assert_no_elevated_authority_or_checkpoint(packet)
        admitted.append(packet)
    initial_digests = {row["initial_parameter_digest"] for row in admitted}
    terminal_digests = {row["terminal_parameter_digest"] for row in admitted}
    if len(initial_digests) != 1 or terminal_digests != initial_digests:
        raise GraftPhaseAShortGPUError(
            "terminal rollback bytes do not equal one WORLD8 initial state"
        )
    return seal_mapping(
        {
            "schema_version": TERMINAL_ROLLBACK_MANIFEST_SCHEMA_VERSION,
            "gate_manifest_digest": gate["digest"],
            "rank_order": list(range(WORLD_SIZE)),
            "ordered_rollback_packets": [dict(row) for row in admitted],
            "world_any_no_go": True,
            "all_eight_skipped_finish": True,
            "all_eight_sessions_failed_closed": True,
            "all_eight_terminal_equal_initial": True,
            "one_world_initial_parameter_digest": next(iter(initial_digests)),
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


def _validate_world8_terminal_rollback_manifest(
    value: Any,
    *,
    gate_manifest: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    admitted = validate_sealed_mapping(value, label=label)
    rebuilt = _admit_world8_terminal_rollback_packets(
        admitted.get("ordered_rollback_packets"),
        gate_manifest=gate_manifest,
    )
    if dict(rebuilt) != admitted:
        raise GraftPhaseAShortGPUError(f"{label} deep admission differs")
    return admitted


def _coordinate_world8_terminal_rollback(
    *,
    collectives: short_trainer.AuthenticatedDP2SP4Backend,
    local: LocalFamilyRouting,
    gate_manifest: Mapping[str, Any],
    local_gate_pass: bool,
    local_terminal_path: str,
    initial_parameter_digest: str,
    terminal_parameter_digest: str,
    trainer_failure_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    failure = validate_sealed_mapping(
        trainer_failure_receipt, label="local terminal failure receipt"
    )
    local_packet = seal_mapping(
        {
            "schema_version": TERMINAL_ROLLBACK_PACKET_SCHEMA_VERSION,
            "global_rank": collectives.rank,
            "dp_arm": local.dp_arm,
            "sp_rank": collectives.sp_rank,
            "family": local.family,
            "local_gate_pass": local_gate_pass,
            "world_any_no_go": True,
            "local_terminal_path": local_terminal_path,
            "initial_parameter_digest": initial_parameter_digest,
            "terminal_parameter_digest": terminal_parameter_digest,
            "trainer_failure_receipt_digest": failure["digest"],
            "session_phase_after_rollback": "failed",
            "finish_called": False,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )
    return _admit_world8_terminal_rollback_packets(
        _world8_all_gather_authenticated(
            collectives=collectives,
            value=local_packet,
            label="post-rollback terminal coordination",
        ),
        gate_manifest=gate_manifest,
    )


def execute_authenticated_trajectory_diagnostic(
    *,
    routing: Any,
    bindings: native_v2.AuthenticatedNativeBindings,
    collectives: short_trainer.AuthenticatedDP2SP4Backend,
    services: AuthenticatedRunnerServices,
    runtime: OfficialShortRuntime,
) -> ShortGPURunnerResult:
    """Run the one fixed optimizer arm and publish PASS or scientific NO-GO."""

    assert_pinned_dependencies()
    services.assert_live()
    if (
        type(runtime) is not OfficialShortRuntime
        or type(bindings) is not native_v2.AuthenticatedNativeBindings
        or type(collectives) is not short_trainer.AuthenticatedDP2SP4Backend
        or runtime.bindings is not bindings
        or services.test_only
        or bindings.test_only
        or collectives.test_only
    ):
        raise GraftPhaseAShortGPUError("trajectory production evidence differs")
    local = route_local_family(routing, dp_arm=collectives.dp_arm)
    plan = _trajectory_plan(local)
    session: Optional[short_trainer.PhaseAShortTrainingSession] = None
    trajectory: list[Mapping[str, Any]] = []
    update_routes: list[Mapping[str, Any]] = []
    final_fields: dict[int, dict[str, torch.Tensor]] = {}
    original_admissions: list[Mapping[str, Any]] = []
    trace: list[Mapping[str, Any]] = []
    try:
        session = _PINNED_OPEN_TRAINER(
            routing=routing,
            bindings=bindings,
            collectives=collectives,
        )
        initial_digest = short_chunked_parameter_registry_digest(
            bindings.named_trainable_parameters
        )
        initial_matrix, _ = _measure_trajectory_stage(
            runtime=runtime,
            plan=plan,
            collectives=collectives,
            stage="initial",
            parameter_digest=initial_digest,
        )
        trajectory.append(initial_matrix)
        trace.append({"operation": "measure", "stage": "initial"})

        for update_number, schedule_index in enumerate(UPDATE_INDICES, start=1):
            update_plan = session.next_update_plan()
            if (
                update_plan.update_number != update_number
                or update_plan.schedule_index != schedule_index
                or update_plan.row is not local.fit_row
                or update_plan.row_iid != local.fit_iid
            ):
                raise GraftPhaseAShortGPUError("trajectory update plan differs")
            cell = services.make_update_cell(plan=update_plan)
            update_receipt = session.run_update(plan=update_plan, cell=cell)
            route = validate_sealed_mapping(
                services.after_update(
                    plan=update_plan, update_receipt=update_receipt
                ),
                label=f"trajectory update {update_number} route",
            )
            if (
                route.get("update_number") != update_number
                or route.get("schedule_index") != schedule_index
                or route.get("row_iid") != local.fit_iid
                or route.get("fit_row_only") is not True
                or route.get("exact_four_native_forwards") is not True
                or route.get("checkpoint_written") is not False
            ):
                raise GraftPhaseAShortGPUError("trajectory update route differs")
            update_routes.append(route)
            parameter_digest = short_chunked_parameter_registry_digest(
                bindings.named_trainable_parameters
            )
            stage = f"after_update_{schedule_index}"
            matrix, retained = _measure_trajectory_stage(
                runtime=runtime,
                plan=plan,
                collectives=collectives,
                stage=stage,
                parameter_digest=parameter_digest,
            )
            trajectory.append(matrix)
            if stage == "after_update_38":
                final_fields = retained
            trace.extend(
                (
                    {
                        "operation": "optimizer_update",
                        "update_number": update_number,
                        "schedule_index": schedule_index,
                        "update_receipt_digest": update_receipt["digest"],
                        "route_receipt_digest": route["digest"],
                    },
                    {"operation": "measure", "stage": stage},
                )
            )

        parity = validate_adapter_off_parity(
            services.adapter_off_parity(
                schedule_indices=ADAPTER_OFF_PARITY_INDICES
            )
        )
        final_stage_cells = trajectory[-1].get("cells")
        if not isinstance(final_stage_cells, list) or len(final_stage_cells) != 2:
            raise GraftPhaseAShortGPUError("final trajectory matrix differs")
        final_gate_vector: dict[str, bool] = {}
        final_metrics_digests: dict[str, str] = {}
        final_metrics_by_index: dict[int, Mapping[str, Any]] = {}
        for ordinal, schedule_index in enumerate(CONFIRMATION_INDICES):
            cell = validate_sealed_mapping(
                final_stage_cells[ordinal],
                label=f"final trajectory index{schedule_index} cell",
            )
            metrics = validate_sealed_mapping(
                cell.get("metrics"),
                label=f"final trajectory index{schedule_index} metrics",
            )
            if (
                cell.get("schedule_index") != schedule_index
                or metrics.get("schedule_index") != schedule_index
                or type(metrics.get("noncompensating_all_pass")) is not bool
            ):
                raise GraftPhaseAShortGPUError("final trajectory gate cell differs")
            final_gate_vector[str(schedule_index)] = metrics[
                "noncompensating_all_pass"
            ]
            final_metrics_digests[str(schedule_index)] = metrics["digest"]
            final_metrics_by_index[schedule_index] = metrics
        terminal_gate_manifest = _coordinate_world8_terminal_gate(
            collectives=collectives,
            local=local,
            final_gate_vector=final_gate_vector,
            final_metrics_digests=final_metrics_digests,
        )
        local_gate_pass = all(final_gate_vector.values())
        world_any_no_go = terminal_gate_manifest["world_any_no_go"]
        if (
            terminal_gate_manifest["ordered_gate_packets"][collectives.rank][
                "local_gate_pass"
            ]
            is not local_gate_pass
        ):
            raise GraftPhaseAShortGPUError("local terminal gate binding differs")
        trace.append(
            {
                "operation": "world8_terminal_gate_coordination",
                "manifest_digest": terminal_gate_manifest["digest"],
                "world_any_no_go": world_any_no_go,
                "completed_before_original_admissions": True,
            }
        )
        official_plan = session.confirmation_plan()
        if (
            official_plan.row is not plan.row
            or official_plan.wrong_owner_row is not plan.wrong_owner_row
            or set(final_fields) != set(CONFIRMATION_INDICES)
        ):
            raise GraftPhaseAShortGPUError("final original-gate binding differs")

        trainer_failure: Optional[Mapping[str, Any]] = None
        trainer_result: Optional[short_trainer.ShortTrainingResult] = None
        terminal_rollback_manifest: Optional[Mapping[str, Any]] = None
        local_terminal_path: str
        if not world_any_no_go:
            if not local_gate_pass:
                raise GraftPhaseAShortGPUError(
                    "WORLD8 all-pass protocol contains a local NO-GO"
                )
            for schedule_index in CONFIRMATION_INDICES:
                with torch.no_grad():
                    original_admissions.append(
                        session.record_confirmation_fields(
                            plan=official_plan,
                            schedule_index=schedule_index,
                            **final_fields[schedule_index],
                        )
                    )
            trainer_result = session.finish()
            trainer_success = validate_sealed_mapping(
                trainer_result.receipt, label="all-pass trainer success receipt"
            )
            if (
                trainer_result.checkpoint_payload is not None
                or trainer_result.publication_payload is not None
                or trainer_success.get("checkpoint_written") is not False
                or trainer_success.get("checkpoint_payload_returned") is not False
                or trainer_success.get("publication_performed") is not False
            ):
                raise GraftPhaseAShortGPUError(
                    "all-pass trainer terminal payload escaped memory-only boundary"
                )
            local_terminal_path = "local_gate_pass_world_all_pass_finished"
        elif local_gate_pass:
            for schedule_index in CONFIRMATION_INDICES:
                with torch.no_grad():
                    original_admissions.append(
                        session.record_confirmation_fields(
                            plan=official_plan,
                            schedule_index=schedule_index,
                            **final_fields[schedule_index],
                        )
                    )
            if session.phase != "ready_to_finalize":
                raise GraftPhaseAShortGPUError(
                    "locally passing arm did not reach ready_to_finalize"
                )
            sentinel = _World8ScientificNoGoRollback(
                _GLOBAL_NO_GO_ROLLBACK_SENTINEL
            )
            _rollback_open_session(session, sentinel)
            candidate = validate_sealed_mapping(
                session.failure_receipt(),
                label="coordinated local-pass rollback receipt",
            )
            if (
                candidate.get("failed_confirmation_metrics") is not None
                or candidate.get("failure_reason")
                != (
                    "_World8ScientificNoGoRollback:"
                    + _GLOBAL_NO_GO_ROLLBACK_SENTINEL
                )
                or candidate.get(
                    "trainable_parameters_restored_to_initial_snapshot"
                )
                is not True
            ):
                raise GraftPhaseAShortGPUError(
                    "coordinated local-pass rollback differs"
                )
            trainer_failure = candidate
            local_terminal_path = "local_gate_pass_global_no_go_rolled_back"
        else:
            gate_error: Optional[
                short_trainer.GraftPhaseAShortTrainingError
            ] = None
            for schedule_index in CONFIRMATION_INDICES:
                try:
                    with torch.no_grad():
                        original_admissions.append(
                            session.record_confirmation_fields(
                                plan=official_plan,
                                schedule_index=schedule_index,
                                **final_fields[schedule_index],
                            )
                        )
                except short_trainer.GraftPhaseAShortTrainingError as error:
                    gate_error = error
                    break
            if gate_error is None or session.phase != "failed":
                raise GraftPhaseAShortGPUError(
                    "local NO-GO did not exercise the original gate rollback"
                )
            candidate = validate_sealed_mapping(
                session.failure_receipt(), label="local gate failure receipt"
            )
            failed_metrics = validate_sealed_mapping(
                candidate.get("failed_confirmation_metrics"),
                label="local failed confirmation metrics",
            )
            first_failed_index = next(
                index
                for index in CONFIRMATION_INDICES
                if final_gate_vector[str(index)] is False
            )
            if (
                failed_metrics.get("schedule_index") != first_failed_index
                or failed_metrics.get("digest")
                != final_metrics_by_index[first_failed_index]["digest"]
                or failed_metrics.get("noncompensating_all_pass") is not False
                or "failed a noncompensating gate" not in str(gate_error)
                or candidate.get(
                    "trainable_parameters_restored_to_initial_snapshot"
                )
                is not True
            ):
                raise GraftPhaseAShortGPUError(
                    "local original-gate NO-GO evidence differs"
                )
            trainer_failure = candidate
            local_terminal_path = "local_gate_fail_global_no_go_rolled_back"

        terminal_parameter_digest = short_chunked_parameter_registry_digest(
            bindings.named_trainable_parameters
        )
        if world_any_no_go:
            if trainer_failure is None or terminal_parameter_digest != initial_digest:
                raise GraftPhaseAShortGPUError(
                    "scientific WORLD8 NO-GO rollback bytes differ"
                )
            terminal_rollback_manifest = _coordinate_world8_terminal_rollback(
                collectives=collectives,
                local=local,
                gate_manifest=terminal_gate_manifest,
                local_gate_pass=local_gate_pass,
                local_terminal_path=local_terminal_path,
                initial_parameter_digest=initial_digest,
                terminal_parameter_digest=terminal_parameter_digest,
                trainer_failure_receipt=trainer_failure,
            )
            trace.append(
                {
                    "operation": "world8_terminal_rollback_coordination",
                    "manifest_digest": terminal_rollback_manifest["digest"],
                    "all_eight_terminal_equal_initial": True,
                }
            )
        elif trainer_result is None or trainer_failure is not None:
            raise GraftPhaseAShortGPUError("WORLD8 all-pass terminal result differs")
        final = seal_mapping(
            {
                "schema_version": SCHEMA_VERSION,
                "status": (
                    (
                        "completed_local_gate_pass_global_no_go_rolled_back"
                        if local_gate_pass
                        else "completed_local_gate_fail_global_no_go_rolled_back"
                    )
                    if world_any_no_go
                    else "completed_original_confirmation_pass_no_checkpoint"
                ),
                "complete": True,
                "scientific_outcome": (
                    "NO_GO" if world_any_no_go else "ORIGINAL_GATE_PASS"
                ),
                "local_scientific_outcome": (
                    "LOCAL_GATE_PASS" if local_gate_pass else "LOCAL_GATE_NO_GO"
                ),
                "topology": {
                    "world_size": WORLD_SIZE,
                    "data_parallel_size": DP_SIZE,
                    "sequence_parallel_size": SP_SIZE,
                    "rank": collectives.rank,
                    "dp_arm": local.dp_arm,
                    "sp_rank": collectives.sp_rank,
                    "family": local.family,
                },
                "single_preregistered_optimizer_arm": {
                    "learning_rate_float64_hex": (
                        short_trainer.OPTIMIZER_LEARNING_RATE.hex()
                    ),
                    "update_schedule_indices": list(UPDATE_INDICES),
                    "update_count": 2,
                    "initial_state_replayed_once": True,
                    "learning_rate_or_step_sweep_performed": False,
                    "post_hoc_arm_selection_performed": False,
                },
                "source_routing": {
                    "routing_digest": local.routing_digest,
                    "fit_iid": local.fit_iid,
                    "confirmation_iid": local.confirmation_iid,
                    "confirmation_row_consumed_by_optimizer": False,
                },
                "trajectory_stage_order": list(TRAJECTORY_STAGES),
                "trajectory_schedule_indices": list(CONFIRMATION_INDICES),
                "trajectory_matrices": [dict(row) for row in trajectory],
                "trajectory_cell_count": 6,
                "all_cells_same_noise_within_interventions": True,
                "all_cells_sp4_exact_consensus": True,
                "update_route_receipts": [dict(row) for row in update_routes],
                "world8_terminal_coordination": dict(terminal_gate_manifest),
                "world8_terminal_rollback_manifest": (
                    None
                    if terminal_rollback_manifest is None
                    else dict(terminal_rollback_manifest)
                ),
                "original_confirmation_gate": {
                    "thresholds_unchanged": True,
                    "final_stage_gate_vector": final_gate_vector,
                    "final_stage_metrics_digests": final_metrics_digests,
                    "all_pass": local_gate_pass,
                    "local_all_pass": local_gate_pass,
                    "world_all_pass": not world_any_no_go,
                    "global_no_go": world_any_no_go,
                    "local_terminal_path": local_terminal_path,
                    "finish_called": not world_any_no_go,
                    "admissions_before_terminal_outcome": [
                        dict(row) for row in original_admissions
                    ],
                    "trainer_success_receipt": (
                        None
                        if trainer_result is None
                        else dict(trainer_result.receipt)
                    ),
                    "trainer_failure_receipt": (
                        None if trainer_failure is None else dict(trainer_failure)
                    ),
                    "rollback_to_initial_trainables": world_any_no_go,
                },
                "adapter_off_parity": dict(parity),
                "execution_trace": trace,
                "initial_parameter_digest": initial_digest,
                "terminal_parameter_digest": terminal_parameter_digest,
                "terminal_equals_initial_after_no_go": (
                    world_any_no_go
                    and terminal_parameter_digest == initial_digest
                ),
                "training_updates_executed_for_diagnostic": 2,
                "diagnostic_receipt_published_even_for_scientific_no_go": True,
                "full_sampler_used": False,
                "decoded_media_output_created": False,
                "checkpoint_written": False,
                "checkpoint_payload_returned": False,
                "publication_performed": False,
                "target_video_used": False,
                "generated_proposal_used": False,
                "t2v_branch_used": False,
                "source_retelling_used": False,
                "selector_used": False,
                "mask_pose_track_flow_or_motion_donor_used": False,
                **_false_authority(),
            }
        )
        return ShortGPURunnerResult(receipt=final)
    except Exception as error:
        try:
            if session is not None and session.phase not in {"closed", "failed"}:
                _rollback_open_session(session, error)
            failure = seal_mapping(
                {
                    "schema_version": FAILURE_SCHEMA_VERSION,
                    "status": "operational_failure_rolled_back_no_checkpoint",
                    "error": f"{type(error).__name__}:{error}",
                    "topology": {
                        "rank": collectives.rank,
                        "dp_arm": local.dp_arm,
                        "sp_rank": collectives.sp_rank,
                        "family": local.family,
                    },
                    "trajectory_matrices_completed": [
                        dict(row) for row in trajectory
                    ],
                    "completed_stage_count": len(trajectory),
                    "checkpoint_written": False,
                    "publication_performed": False,
                    **_false_authority(),
                }
            )
        except Exception as cleanup_error:
            failure = seal_mapping(
                {
                    "schema_version": FAILURE_SCHEMA_VERSION,
                    "status": "operational_failure_cleanup_not_authenticated",
                    "error": f"{type(error).__name__}:{error}",
                    "cleanup_error": (
                        f"{type(cleanup_error).__name__}:{cleanup_error}"
                    ),
                    "checkpoint_written": False,
                    **_false_authority(),
                }
            )
        raise GraftPhaseAShortGPUError(
            "trajectory diagnostic operational failure",
            diagnostic_receipt=failure,
        ) from error


def authenticate_official_services(
    runtime: OfficialShortRuntime,
) -> AuthenticatedRunnerServices:
    if type(runtime) is not OfficialShortRuntime:
        raise GraftPhaseAShortGPUError("official runtime type differs")
    receipt = seal_mapping(
        {
            "schema_version": SERVICES_SCHEMA_VERSION,
            "binding_label": "official_hash_bound_bernini_world8_dp2sp4",
            "test_only": False,
            "official_checkpoint_runtime_bound": True,
            "official_gpu_runtime_authenticated": True,
            "gpu_execution_authorized": False,
            "callbacks_injected_for_unit_test": False,
            "runner_source_sha256_required_from_cli": True,
            "no_checkpoint_publication": True,
            **_false_authority(),
        }
    )
    return AuthenticatedRunnerServices._mint(
        token=_SERVICES_TOKEN,
        make_update_cell=runtime.make_update_cell,
        after_update=runtime.after_update,
        make_confirmation_fields=runtime.make_confirmation_fields,
        adapter_off_parity=runtime.adapter_off_parity,
        test_only=False,
        receipt=receipt,
    )


def _assert_no_elevated_authority_or_checkpoint(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, bool) and (
                key in AUTHORITY_FIELDS
                or key.endswith("_authorized")
                or "authority" in key
            ) and item:
                raise GraftPhaseAShortGPUError("WORLD8 local result elevated authority")
            if key in {
                "checkpoint_written",
                "checkpoint_payload_returned",
                "publication_performed",
            } and item is not False:
                raise GraftPhaseAShortGPUError("WORLD8 local result published state")
            _assert_no_elevated_authority_or_checkpoint(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_elevated_authority_or_checkpoint(item)


def assemble_world8_local_results(packets: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Deeply admit all eight local receipts and expose both arm representatives."""

    if not isinstance(packets, (list, tuple)) or len(packets) != WORLD_SIZE:
        raise GraftPhaseAShortGPUError("WORLD8 full-result coverage differs")
    admitted = []
    for expected_rank, packet in enumerate(packets):
        if not isinstance(packet, Mapping) or packet.get("global_rank") != expected_rank:
            raise GraftPhaseAShortGPUError("WORLD8 result rank mapping differs")
        if len(canonical_json_bytes(packet)) >= MAX_FULL_LOCAL_RESULT_PACKET_BYTES:
            raise GraftPhaseAShortGPUError("WORLD8 full-result packet is oversized")
        result = validate_sealed_mapping(
            packet.get("local_result"), label=f"rank{expected_rank} local result"
        )
        if packet.get("result_digest") != result["digest"]:
            raise GraftPhaseAShortGPUError("WORLD8 wrapper/result digest differs")
        topology = result.get("topology")
        source = result.get("source_routing")
        arm = expected_rank // SP_SIZE
        sp_rank = expected_rank % SP_SIZE
        if (
            result.get("schema_version") != SCHEMA_VERSION
            or result.get("status") != "completed_in_memory_diagnostic_no_checkpoint"
            or result.get("complete") is not True
            or not isinstance(topology, Mapping)
            or topology.get("rank") != expected_rank
            or topology.get("dp_arm") != arm
            or topology.get("sp_rank") != sp_rank
            or topology.get("family") != FAMILY_BY_DP_ARM[arm]
            or not isinstance(source, Mapping)
            or source.get("fit_iid") != FIT_IID_BY_DP_ARM[arm]
            or source.get("confirmation_iid") != CONFIRMATION_IID_BY_DP_ARM[arm]
            or result.get("checkpoint_written") is not False
            or result.get("publication_performed") is not False
            or result.get("training_updates_executed_for_diagnostic") != 2
            or any(
                result.get(key) is not False
                for key in (
                    "full_sampler_used", "decoded_media_output_created",
                    "target_video_used", "generated_proposal_used",
                    "t2v_branch_used", "source_retelling_used", "selector_used",
                )
            )
        ):
            raise GraftPhaseAShortGPUError("WORLD8 local result contract differs")
        short = validate_sealed_mapping(
            result.get("short_trainer_receipt"),
            label=f"rank{expected_rank} short trainer result",
        )
        confirmation = short.get("confirmation")
        short_topology = short.get("topology")
        short_source = short.get("source_routing")
        if (
            short.get("schema_version") != short_trainer.SCHEMA_VERSION
            or short.get("status") != "completed_in_memory_orchestration"
            or not isinstance(short_topology, Mapping)
            or short_topology.get("rank") != expected_rank
            or short_topology.get("dp_arm") != arm
            or short_topology.get("sp_rank") != sp_rank
            or not isinstance(short_source, Mapping)
            or short_source.get("local_update_iid") != FIT_IID_BY_DP_ARM[arm]
            or short_source.get("local_confirmation_iid")
            != CONFIRMATION_IID_BY_DP_ARM[arm]
            or not isinstance(confirmation, Mapping)
            or confirmation.get(
                "all_indices_noncompensating_hard_gate_passed"
            )
            is not True
            or short.get("checkpoint_written") is not False
            or short.get("publication_performed") is not False
        ):
            raise GraftPhaseAShortGPUError("WORLD8 arm confirmation hard gate differs")
        metrics = confirmation.get("per_index_metrics")
        consensus = confirmation.get("sp4_consensus_digest")
        runner_confirmation = result.get("confirmation")
        provenances = (
            runner_confirmation.get("provenance")
            if isinstance(runner_confirmation, Mapping) else None
        )
        admissions = (
            runner_confirmation.get("admissions")
            if isinstance(runner_confirmation, Mapping) else None
        )
        if (
            not isinstance(metrics, Mapping)
            or tuple(metrics) != tuple(str(index) for index in CONFIRMATION_INDICES)
            or not isinstance(consensus, Mapping)
            or tuple(consensus) != tuple(str(index) for index in CONFIRMATION_INDICES)
            or not isinstance(runner_confirmation, Mapping)
            or runner_confirmation.get("schedule_indices") != list(CONFIRMATION_INDICES)
            or runner_confirmation.get("field_roles")
            != list(short_trainer.CONFIRMATION_FIELD_ROLES)
            or any(
                runner_confirmation.get(key) is not True
                for key in (
                    "exact_six_fields_per_index",
                    "same_state_interventions_verified",
                    "wrong_atlas_same_family_fit_verified",
                    "drop_disables_only_identity_rebinder_memory_verified",
                )
            )
            or not isinstance(provenances, list) or len(provenances) != 2
            or not isinstance(admissions, list) or len(admissions) != 2
        ):
            raise GraftPhaseAShortGPUError("WORLD8 per-index metrics coverage differs")
        for index in CONFIRMATION_INDICES:
            metric = validate_sealed_mapping(
                metrics[str(index)], label=f"rank{expected_rank} index{index} metrics"
            )
            gates = metric.get("noncompensating_gates")
            provenance = validate_sealed_mapping(
                provenances[CONFIRMATION_INDICES.index(index)],
                label=f"rank{expected_rank} index{index} provenance",
            )
            admission = validate_sealed_mapping(
                admissions[CONFIRMATION_INDICES.index(index)],
                label=f"rank{expected_rank} index{index} admission",
            )
            admission_metrics = validate_sealed_mapping(
                admission.get("metrics"),
                label=f"rank{expected_rank} index{index} admission metrics",
            )
            record = {
                "row_iid": CONFIRMATION_IID_BY_DP_ARM[arm],
                "wrong_owner_iid": FIT_IID_BY_DP_ARM[arm],
                "schedule_index": index,
                "metrics_digest": metric["digest"],
                "parameter_digest": admission.get("parameter_digest"),
                "base_digest": admission.get("base_digest"),
                "optimizer_digest": admission.get("optimizer_digest"),
            }
            if (
                metric.get("schema_version")
                != "bernini-graft-phase-a-confirmation-metrics-v1"
                or metric.get("schedule_index") != index
                or not isinstance(gates, Mapping)
                or len(gates) != 4
                or any(value is not True for value in gates.values())
                or metric.get("noncompensating_all_pass") is not True
                or provenance.get("schedule_index") != index
                or provenance.get("confirmation_iid")
                != CONFIRMATION_IID_BY_DP_ARM[arm]
                or provenance.get("wrong_owner_iid") != FIT_IID_BY_DP_ARM[arm]
                or any(provenance.get(key) is not True for key in _CONFIRMATION_TRUE_FLAGS)
                or any(provenance.get(key) is not False for key in _CONFIRMATION_FALSE_FLAGS)
                or any(
                    provenance.get(key) is not True
                    for key in (
                        "same_state_tensor_identities_recomputed_byte_equal",
                        "wrong_route_receipts_differ_only_in_atlas_memory",
                        "drop_route_receipts_retain_v_branch_disable_only_rebinder",
                        "action_noop_route_receipts_equal_with_negative_raw_reuse",
                    )
                )
                or admission.get("schema_version")
                != "bernini-graft-phase-a-confirmation-field-admission-v1"
                or any(admission.get(key) != value for key, value in record.items())
                or admission_metrics != metric
                or admission.get("sp4_consensus_digest") != object_sha256(record)
                or consensus[str(index)] != admission.get("sp4_consensus_digest")
            ):
                raise GraftPhaseAShortGPUError("WORLD8 per-index hard gate differs")
            _assert_no_elevated_authority_or_checkpoint(metric)
        validate_adapter_off_parity(result.get("adapter_off_parity"))
        routes = result.get("update_route_receipts")
        if not isinstance(routes, list) or len(routes) != 2:
            raise GraftPhaseAShortGPUError("WORLD8 update route coverage differs")
        for ordinal, route in enumerate(routes):
            admitted_route = validate_sealed_mapping(route, label="WORLD8 update route")
            if (
                admitted_route.get("update_number") != ordinal + 1
                or admitted_route.get("schedule_index") != UPDATE_INDICES[ordinal]
                or admitted_route.get("row_iid") != FIT_IID_BY_DP_ARM[arm]
                or admitted_route.get("fit_row_only") is not True
                or admitted_route.get("exact_four_native_forwards") is not True
                or admitted_route.get("forward_order")
                != [
                    ["measurement", "negative"],
                    ["measurement", "positive"],
                    ["replay", "negative"],
                    ["replay", "positive"],
                ]
                or admitted_route.get("checkpoint_written") is not False
            ):
                raise GraftPhaseAShortGPUError("WORLD8 update route differs")
        _assert_no_elevated_authority_or_checkpoint(result)
        admitted.append(result)
    arms = []
    for arm, representative_rank in enumerate((0, 4)):
        rows = admitted[arm * SP_SIZE : (arm + 1) * SP_SIZE]
        if len(rows) != SP_SIZE or rows[0]["topology"]["rank"] != representative_rank:
            raise GraftPhaseAShortGPUError("WORLD8 arm representative differs")
        arms.append(
            {
                "dp_arm": arm,
                "family": FAMILY_BY_DP_ARM[arm],
                "global_ranks": list(range(arm * SP_SIZE, (arm + 1) * SP_SIZE)),
                "representative_global_rank": representative_rank,
                "representative_full_receipt": rows[0],
                "per_rank_result_digests": [row["digest"] for row in rows],
                "all_four_confirmation_hard_gates_passed": True,
            }
        )
    return seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-short-world8-full-results-v1",
            "rank_order": list(range(WORLD_SIZE)),
            "dp2_family_order": list(FAMILY_BY_DP_ARM),
            "all_eight_full_local_receipts": admitted,
            "arm_representatives": arms,
            "dog_and_human_exact_coverage": True,
            "all_eight_confirmation_hard_gates_passed": True,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


def _finite_float64_hex(value: Any, *, label: str) -> float:
    if not isinstance(value, str):
        raise GraftPhaseAShortGPUError(f"{label} is not float64 hex")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise GraftPhaseAShortGPUError(f"{label} is not float64 hex") from error
    if not math.isfinite(result) or result.hex() != value:
        raise GraftPhaseAShortGPUError(f"{label} is not canonical finite float64 hex")
    return result


def _deep_validate_trajectory_metrics(
    value: Any, *, schedule_index: int, label: str
) -> Mapping[str, Any]:
    metrics = validate_sealed_mapping(value, label=label)
    roles = tuple(short_trainer.CONFIRMATION_FIELD_ROLES)
    hashes = metrics.get("field_tensor_sha256")
    losses = metrics.get("noop_fm_loss_float64_hex")
    gains = metrics.get("relative_gain_float64_hex")
    norms = metrics.get("action_delta_norm_float64_hex")
    gates = metrics.get("noncompensating_gates")
    shape = metrics.get("field_shape")
    if (
        metrics.get("schema_version")
        != "bernini-graft-phase-a-confirmation-metrics-v1"
        or metrics.get("schedule_index") != schedule_index
        or metrics.get("field_roles") != list(roles)
        or not isinstance(shape, list)
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
        or metrics.get("field_dtype") != "torch.float32"
        or metrics.get("field_device_type") != "cuda"
        or not isinstance(hashes, Mapping)
        or tuple(hashes) != roles
        or not isinstance(losses, Mapping)
        or tuple(losses) != ("correct_atlas", "wrong_atlas", "dropped_atlas")
        or metrics.get("relative_gain_formula")
        != "(L_control-L_correct)/max(L_control,float64_tiny)"
        or not isinstance(gains, Mapping)
        or tuple(gains) != ("correct_vs_wrong", "correct_vs_drop")
        or metrics.get("minimum_relative_gain_float64_hex")
        != short_trainer.MIN_CONFIRMATION_RELATIVE_GAIN.hex()
        or metrics.get("action_delta_formula") != "v_action-v_noop"
        or not isinstance(norms, Mapping)
        or tuple(norms) != ("correct_atlas", "dropped_atlas")
        or metrics.get("action_delta_correct_drop_norm_ratio_formula")
        != "norm(delta_correct)/max(norm(delta_drop),float64_tiny)"
        or metrics.get(
            "minimum_action_delta_correct_drop_norm_ratio_float64_hex"
        )
        != short_trainer.MIN_ACTION_DELTA_CORRECT_DROP_NORM_RATIO.hex()
        or metrics.get("minimum_action_delta_cosine_float64_hex")
        != short_trainer.MIN_ACTION_DELTA_COSINE.hex()
        or metrics.get("float64_tiny_hex") != short_trainer.FLOAT64_TINY.hex()
        or not isinstance(gates, Mapping)
        or tuple(gates) != _CONFIRMATION_GATE_KEYS
        or any(type(gates[key]) is not bool for key in _CONFIRMATION_GATE_KEYS)
        or metrics.get("metrics_computed_from_six_detached_fields_by_this_core")
        is not True
        or metrics.get("field_origin_same_noise_state_coordinate_verified_by_this_core")
        is not False
    ):
        raise GraftPhaseAShortGPUError(f"{label} structure differs")
    for role in roles:
        _require_sha256(hashes[role], label=f"{label} {role} tensor")
    correct_loss = _finite_float64_hex(
        losses["correct_atlas"], label=f"{label} correct loss"
    )
    wrong_loss = _finite_float64_hex(
        losses["wrong_atlas"], label=f"{label} wrong loss"
    )
    dropped_loss = _finite_float64_hex(
        losses["dropped_atlas"], label=f"{label} dropped loss"
    )
    if min(correct_loss, wrong_loss, dropped_loss) < 0.0:
        raise GraftPhaseAShortGPUError(f"{label} loss is negative")
    wrong_gain = _finite_float64_hex(
        gains["correct_vs_wrong"], label=f"{label} wrong gain"
    )
    dropped_gain = _finite_float64_hex(
        gains["correct_vs_drop"], label=f"{label} dropped gain"
    )
    correct_norm = _finite_float64_hex(
        norms["correct_atlas"], label=f"{label} correct action norm"
    )
    dropped_norm = _finite_float64_hex(
        norms["dropped_atlas"], label=f"{label} dropped action norm"
    )
    ratio = _finite_float64_hex(
        metrics.get("action_delta_correct_drop_norm_ratio_float64_hex"),
        label=f"{label} action norm ratio",
    )
    cosine = _finite_float64_hex(
        metrics.get("action_delta_correct_drop_cosine_float64_hex"),
        label=f"{label} action cosine",
    )
    tiny = short_trainer.FLOAT64_TINY
    expected_wrong_gain = (wrong_loss - correct_loss) / max(wrong_loss, tiny)
    expected_dropped_gain = (dropped_loss - correct_loss) / max(
        dropped_loss, tiny
    )
    expected_ratio = correct_norm / max(dropped_norm, tiny)
    expected_gates = {
        "correct_vs_wrong_noop_relative_gain": (
            wrong_gain >= short_trainer.MIN_CONFIRMATION_RELATIVE_GAIN
        ),
        "correct_vs_drop_noop_relative_gain": (
            dropped_gain >= short_trainer.MIN_CONFIRMATION_RELATIVE_GAIN
        ),
        "action_delta_correct_drop_norm_ratio": (
            ratio >= short_trainer.MIN_ACTION_DELTA_CORRECT_DROP_NORM_RATIO
        ),
        "action_delta_correct_drop_cosine": (
            cosine >= short_trainer.MIN_ACTION_DELTA_COSINE
        ),
    }
    if (
        correct_norm <= 0.0
        or dropped_norm <= 0.0
        or not -1.000000000000001 <= cosine <= 1.000000000000001
        or wrong_gain.hex() != expected_wrong_gain.hex()
        or dropped_gain.hex() != expected_dropped_gain.hex()
        or ratio.hex() != expected_ratio.hex()
        or dict(gates) != expected_gates
        or metrics.get("noncompensating_all_pass") is not all(gates.values())
    ):
        raise GraftPhaseAShortGPUError(f"{label} derived gates differ")
    _assert_no_elevated_authority_or_checkpoint(metrics)
    return metrics


def _deep_validate_trajectory_provenance(
    value: Any,
    *,
    schedule_index: int,
    arm: int,
    label: str,
) -> Mapping[str, Any]:
    provenance = validate_sealed_mapping(value, label=label)
    roles = tuple(short_trainer.CONFIRMATION_FIELD_ROLES)
    identities = provenance.get("field_tensor_identities")
    before = provenance.get("same_state_identities_before_model_fields")
    after = provenance.get("same_state_identities_after_all_fields")
    if (
        provenance.get("schema_version") != CONFIRMATION_FIELDS_SCHEMA_VERSION
        or provenance.get("schedule_index") != schedule_index
        or provenance.get("confirmation_iid")
        != CONFIRMATION_IID_BY_DP_ARM[arm]
        or provenance.get("wrong_owner_iid") != FIT_IID_BY_DP_ARM[arm]
        or provenance.get("field_roles") != list(roles)
        or not isinstance(identities, Mapping)
        or tuple(identities) != roles
        or any(
            provenance.get(name) is not True for name in _CONFIRMATION_TRUE_FLAGS
        )
        or any(
            provenance.get(name) is not False for name in _CONFIRMATION_FALSE_FLAGS
        )
        or any(
            provenance.get(name) is not True
            for name in (
                "same_state_tensor_identities_recomputed_byte_equal",
                "wrong_route_receipts_differ_only_in_atlas_memory",
                "drop_route_receipts_retain_v_branch_disable_only_rebinder",
                "action_noop_route_receipts_equal_with_negative_raw_reuse",
            )
        )
        or not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or tuple(before) != _PRODUCTION_SAME_STATE_IDENTITY_FIELDS
        or dict(before) != dict(after)
    ):
        raise GraftPhaseAShortGPUError(f"{label} structure differs")
    _require_sha256(
        provenance.get("confirmation_source_sha256"),
        label=f"{label} confirmation source",
    )
    _require_sha256(
        provenance.get("wrong_owner_source_sha256"),
        label=f"{label} wrong-owner source",
    )
    for role in roles:
        if not isinstance(identities[role], Mapping):
            raise GraftPhaseAShortGPUError(f"{label} tensor identity differs")
    _assert_no_elevated_authority_or_checkpoint(provenance)
    return provenance


def assemble_trajectory_world8_results(
    packets: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Deeply admit the eight 3x2 matrices, including scientific NO-GO."""

    if not isinstance(packets, (list, tuple)) or len(packets) != WORLD_SIZE:
        raise GraftPhaseAShortGPUError("trajectory WORLD8 coverage differs")
    admitted = []
    for rank, packet in enumerate(packets):
        if not isinstance(packet, Mapping) or packet.get("global_rank") != rank:
            raise GraftPhaseAShortGPUError("trajectory WORLD8 rank order differs")
        result = validate_sealed_mapping(
            packet.get("local_result"), label=f"trajectory rank{rank} result"
        )
        topology = result.get("topology")
        source = result.get("source_routing")
        matrices = result.get("trajectory_matrices")
        routes = result.get("update_route_receipts")
        arm = rank // SP_SIZE
        sp_rank = rank % SP_SIZE
        allowed_status = {
            "completed_local_gate_pass_global_no_go_rolled_back",
            "completed_local_gate_fail_global_no_go_rolled_back",
            "completed_original_confirmation_pass_no_checkpoint",
        }
        if (
            packet.get("result_digest") != result["digest"]
            or result.get("schema_version") != SCHEMA_VERSION
            or result.get("status") not in allowed_status
            or result.get("complete") is not True
            or not isinstance(topology, Mapping)
            or topology.get("rank") != rank
            or topology.get("dp_arm") != arm
            or topology.get("sp_rank") != sp_rank
            or topology.get("family") != FAMILY_BY_DP_ARM[arm]
            or not isinstance(source, Mapping)
            or source.get("fit_iid") != FIT_IID_BY_DP_ARM[arm]
            or source.get("confirmation_iid")
            != CONFIRMATION_IID_BY_DP_ARM[arm]
            or result.get("trajectory_stage_order") != list(TRAJECTORY_STAGES)
            or result.get("trajectory_schedule_indices")
            != list(CONFIRMATION_INDICES)
            or result.get("trajectory_cell_count") != 6
            or not isinstance(matrices, list)
            or len(matrices) != 3
            or not isinstance(routes, list)
            or len(routes) != 2
            or result.get("training_updates_executed_for_diagnostic") != 2
            or result.get("checkpoint_written") is not False
            or result.get("publication_performed") is not False
        ):
            raise GraftPhaseAShortGPUError(
                "trajectory WORLD8 local contract differs"
            )
        final_gate_vector = {}
        final_metrics_digests = {}
        stage_parameter_digests = []
        for stage_ordinal, stage in enumerate(TRAJECTORY_STAGES):
            matrix = validate_sealed_mapping(
                matrices[stage_ordinal],
                label=f"rank{rank} trajectory {stage}",
            )
            cells = matrix.get("cells")
            if (
                matrix.get("schema_version") != TRAJECTORY_MATRIX_SCHEMA_VERSION
                or matrix.get("stage") != stage
                or matrix.get("schedule_indices") != list(CONFIRMATION_INDICES)
                or matrix.get("parameter_bytes_unchanged_by_measurement")
                is not True
                or matrix.get("parameter_digest_after")
                != matrix.get("parameter_digest_before")
                or matrix.get("all_cells_sp4_exact_consensus") is not True
                or not isinstance(cells, list)
                or len(cells) != 2
            ):
                raise GraftPhaseAShortGPUError("trajectory matrix differs")
            stage_parameter_digests.append(matrix["parameter_digest_before"])
            for index_ordinal, index in enumerate(CONFIRMATION_INDICES):
                cell = validate_sealed_mapping(
                    cells[index_ordinal],
                    label=f"rank{rank} {stage} index{index}",
                )
                metrics = _deep_validate_trajectory_metrics(
                    cell.get("metrics"),
                    schedule_index=index,
                    label=f"rank{rank} {stage} index{index} metrics",
                )
                provenance = _deep_validate_trajectory_provenance(
                    cell.get("provenance"),
                    schedule_index=index,
                    arm=arm,
                    label=f"rank{rank} {stage} index{index} provenance",
                )
                digests = validate_sealed_mapping(
                    cell.get("owner_atlas_route"),
                    label=f"rank{rank} {stage} index{index} digests",
                )
                manifest = validate_sealed_mapping(
                    cell.get("sp4_evidence_manifest"),
                    label=f"rank{rank} {stage} index{index} SP4 manifest",
                )
                sp_rows = manifest.get("ordered_local_evidence")
                consensus_record = {
                    "stage": stage,
                    "schedule_index": index,
                    "dp_arm": arm,
                    "confirmation_iid": CONFIRMATION_IID_BY_DP_ARM[arm],
                    "wrong_owner_iid": FIT_IID_BY_DP_ARM[arm],
                    "parameter_digest": matrix["parameter_digest_before"],
                    "metrics_digest": metrics["digest"],
                    "sp4_evidence_manifest_digest": manifest["digest"],
                }
                if (
                    cell.get("schema_version")
                    != "bernini-graft-phase-a-short-trajectory-cell-v1"
                    or cell.get("stage") != stage
                    or cell.get("schedule_index") != index
                    or cell.get("dp_arm") != arm
                    or cell.get("confirmation_iid")
                    != CONFIRMATION_IID_BY_DP_ARM[arm]
                    or cell.get("wrong_owner_iid") != FIT_IID_BY_DP_ARM[arm]
                    or cell.get("parameter_digest")
                    != matrix["parameter_digest_before"]
                    or cell.get("metrics_digest") != metrics["digest"]
                    or cell.get("sp4_evidence_manifest_digest")
                    != manifest["digest"]
                    or cell.get("sp4_consensus_digest")
                    != object_sha256(consensus_record)
                    or cell.get("sp4_exact_consensus") is not True
                    or cell.get("same_noise_within_correct_wrong_drop") is not True
                    or cell.get("original_thresholds_used_without_change")
                    is not True
                    or digests.get("schema_version")
                    != "bernini-graft-phase-a-short-trajectory-owner-atlas-route-v1"
                    or any(
                        not isinstance(digests.get(name), str)
                        or _SHA256.fullmatch(digests[name]) is None
                        for name in (
                            "owner_digest",
                            "atlas_digest",
                            "same_state_digest",
                            "route_digest",
                        )
                    )
                    or manifest.get("schema_version")
                    != "bernini-graft-phase-a-short-trajectory-sp4-consensus-v1"
                    or manifest.get("dp_arm") != arm
                    or manifest.get("stage") != stage
                    or manifest.get("schedule_index") != index
                    or manifest.get(
                        "metrics_owner_atlas_same_state_equal_across_sp4"
                    )
                    is not True
                    or manifest.get("rank_local_route_digests_preserved")
                    is not True
                    or not isinstance(sp_rows, list)
                    or len(sp_rows) != SP_SIZE
                    or any(not isinstance(row, Mapping) for row in sp_rows)
                    or [row.get("sp_rank") for row in sp_rows]
                    != list(range(SP_SIZE))
                    or len({row.get("metrics_digest") for row in sp_rows}) != 1
                    or len({row.get("owner_digest") for row in sp_rows}) != 1
                    or len({row.get("atlas_digest") for row in sp_rows}) != 1
                    or len({row.get("same_state_digest") for row in sp_rows}) != 1
                    or any(
                        row.get("metrics_digest") != metrics["digest"]
                        or row.get("owner_digest") != digests["owner_digest"]
                        or row.get("atlas_digest") != digests["atlas_digest"]
                        or row.get("same_state_digest")
                        != digests["same_state_digest"]
                        or not isinstance(row.get("provenance_digest"), str)
                        or _SHA256.fullmatch(row["provenance_digest"]) is None
                        or not isinstance(row.get("route_digest"), str)
                        or _SHA256.fullmatch(row["route_digest"]) is None
                        for row in sp_rows
                    )
                    or sp_rows[sp_rank].get("provenance_digest")
                    != provenance["digest"]
                    or sp_rows[sp_rank].get("route_digest")
                    != digests["route_digest"]
                ):
                    raise GraftPhaseAShortGPUError(
                        "trajectory cell deep admission differs"
                    )
                if stage == "after_update_38":
                    final_gate_vector[str(index)] = metrics[
                        "noncompensating_all_pass"
                    ]
                    final_metrics_digests[str(index)] = metrics["digest"]
        if (
            len(set(stage_parameter_digests)) != len(TRAJECTORY_STAGES)
            or stage_parameter_digests[0] != result.get("initial_parameter_digest")
        ):
            raise GraftPhaseAShortGPUError(
                "trajectory stage parameter transitions differ"
            )
        for ordinal, raw_route in enumerate(routes):
            route = validate_sealed_mapping(
                raw_route, label=f"rank{rank} update route {ordinal + 1}"
            )
            if (
                route.get("schema_version")
                != "bernini-graft-phase-a-short-update-route-v1"
                or route.get("update_number") != ordinal + 1
                or route.get("schedule_index") != UPDATE_INDICES[ordinal]
                or route.get("row_iid") != FIT_IID_BY_DP_ARM[arm]
                or route.get("fit_row_only") is not True
                or route.get("exact_four_native_forwards") is not True
                or route.get("forward_order")
                != [
                    ["measurement", "negative"],
                    ["measurement", "positive"],
                    ["replay", "negative"],
                    ["replay", "positive"],
                ]
                or route.get("fresh_atlas_per_forward") is not True
                or route.get("measurement_atlas_detached") is not True
                or route.get(
                    "replay_atlas_graph_bearing_only_on_target_owner"
                )
                is not True
                or route.get("checkpoint_written") is not False
            ):
                raise GraftPhaseAShortGPUError("trajectory update route differs")
            _assert_no_elevated_authority_or_checkpoint(route)
        gate = result.get("original_confirmation_gate")
        final_gate_vector = _validated_final_gate_vector(
            final_gate_vector, label=f"rank{rank} reconstructed final gate"
        )
        final_metrics_digests = _validated_final_metrics_digests(
            final_metrics_digests,
            label=f"rank{rank} reconstructed final metrics",
        )
        local_gate_pass = all(final_gate_vector.values())
        terminal_gate = _validate_world8_terminal_gate_manifest(
            result.get("world8_terminal_coordination"),
            label=f"rank{rank} terminal gate manifest",
        )
        gate_packet = terminal_gate["ordered_gate_packets"][rank]
        world_no_go = terminal_gate["world_any_no_go"]
        expected_path = (
            (
                "local_gate_pass_global_no_go_rolled_back"
                if local_gate_pass
                else "local_gate_fail_global_no_go_rolled_back"
            )
            if world_no_go
            else "local_gate_pass_world_all_pass_finished"
        )
        expected_status = (
            (
                "completed_local_gate_pass_global_no_go_rolled_back"
                if local_gate_pass
                else "completed_local_gate_fail_global_no_go_rolled_back"
            )
            if world_no_go
            else "completed_original_confirmation_pass_no_checkpoint"
        )
        expected_admission_indices = list(CONFIRMATION_INDICES)
        if world_no_go and not local_gate_pass:
            first_failed = next(
                index
                for index in CONFIRMATION_INDICES
                if final_gate_vector[str(index)] is False
            )
            expected_admission_indices = list(CONFIRMATION_INDICES)[
                : list(CONFIRMATION_INDICES).index(first_failed)
            ]
        admissions = gate.get("admissions_before_terminal_outcome") if isinstance(gate, Mapping) else None
        if (
            not isinstance(gate, Mapping)
            or gate.get("thresholds_unchanged") is not True
            or gate.get("final_stage_gate_vector") != final_gate_vector
            or gate.get("final_stage_metrics_digests") != final_metrics_digests
            or gate.get("all_pass") is not local_gate_pass
            or gate.get("local_all_pass") is not local_gate_pass
            or gate.get("world_all_pass") is world_no_go
            or gate.get("global_no_go") is not world_no_go
            or gate.get("local_terminal_path") != expected_path
            or gate.get("finish_called") is world_no_go
            or result.get("status") != expected_status
            or result.get("scientific_outcome")
            != ("NO_GO" if world_no_go else "ORIGINAL_GATE_PASS")
            or result.get("local_scientific_outcome")
            != ("LOCAL_GATE_PASS" if local_gate_pass else "LOCAL_GATE_NO_GO")
            or gate_packet.get("final_gate_vector") != final_gate_vector
            or gate_packet.get("final_metrics_digests")
            != final_metrics_digests
            or gate_packet.get("local_gate_pass") is not local_gate_pass
            or not isinstance(admissions, list)
            or len(admissions) != len(expected_admission_indices)
            or (gate.get("rollback_to_initial_trainables") is True)
            is not world_no_go
            or (
                result.get("terminal_parameter_digest")
                == result.get("initial_parameter_digest")
            )
            is not world_no_go
            or (
                not world_no_go
                and result.get("terminal_parameter_digest")
                != stage_parameter_digests[-1]
            )
            or result.get("terminal_equals_initial_after_no_go")
            is not world_no_go
        ):
            raise GraftPhaseAShortGPUError(
                "trajectory original-gate outcome differs"
            )
        _require_sha256(
            result.get("initial_parameter_digest"),
            label=f"rank{rank} initial parameter digest",
        )
        _require_sha256(
            result.get("terminal_parameter_digest"),
            label=f"rank{rank} terminal parameter digest",
        )
        for ordinal, schedule_index in enumerate(expected_admission_indices):
            admission = validate_sealed_mapping(
                admissions[ordinal],
                label=f"rank{rank} admission index{schedule_index}",
            )
            admission_metrics = _deep_validate_trajectory_metrics(
                admission.get("metrics"),
                schedule_index=schedule_index,
                label=f"rank{rank} admission index{schedule_index} metrics",
            )
            if (
                admission.get("schema_version")
                != "bernini-graft-phase-a-confirmation-field-admission-v1"
                or admission.get("schedule_index") != schedule_index
                or admission.get("row_iid")
                != CONFIRMATION_IID_BY_DP_ARM[arm]
                or admission.get("wrong_owner_iid") != FIT_IID_BY_DP_ARM[arm]
                or admission_metrics["digest"]
                != final_metrics_digests[str(schedule_index)]
                or admission.get("metrics_digest") != admission_metrics["digest"]
                or admission.get("checkpoint_written") is not False
            ):
                raise GraftPhaseAShortGPUError(
                    "trajectory original admission differs"
                )
            _assert_no_elevated_authority_or_checkpoint(admission)

        trainer_failure = gate.get("trainer_failure_receipt")
        trainer_success = gate.get("trainer_success_receipt")
        rollback_manifest = result.get("world8_terminal_rollback_manifest")
        if world_no_go:
            failure = validate_sealed_mapping(
                trainer_failure, label=f"rank{rank} trainer failure receipt"
            )
            rollback = _validate_world8_terminal_rollback_manifest(
                rollback_manifest,
                gate_manifest=terminal_gate,
                label=f"rank{rank} terminal rollback manifest",
            )
            rollback_packet = rollback["ordered_rollback_packets"][rank]
            failed_metrics = failure.get("failed_confirmation_metrics")
            if (
                trainer_success is not None
                or failure.get("status") != "failed_rolled_back_no_checkpoint"
                or failure.get("rank") != rank
                or failure.get("dp_arm") != arm
                or failure.get("sp_rank") != sp_rank
                or failure.get("completed_optimizer_steps_before_failure") != 2
                or failure.get(
                    "trainable_parameters_restored_to_initial_snapshot"
                )
                is not True
                or rollback.get("gate_manifest_digest") != terminal_gate["digest"]
                or rollback_packet.get("initial_parameter_digest")
                != result.get("initial_parameter_digest")
                or rollback_packet.get("terminal_parameter_digest")
                != result.get("terminal_parameter_digest")
                or rollback_packet.get("trainer_failure_receipt_digest")
                != failure["digest"]
                or rollback_packet.get("local_terminal_path") != expected_path
            ):
                raise GraftPhaseAShortGPUError(
                    "trajectory WORLD8 rollback outcome differs"
                )
            if local_gate_pass:
                if (
                    failed_metrics is not None
                    or failure.get("failure_reason")
                    != (
                        "_World8ScientificNoGoRollback:"
                        + _GLOBAL_NO_GO_ROLLBACK_SENTINEL
                    )
                ):
                    raise GraftPhaseAShortGPUError(
                        "trajectory coordinated local-pass rollback differs"
                    )
            else:
                first_failed = next(
                    index
                    for index in CONFIRMATION_INDICES
                    if final_gate_vector[str(index)] is False
                )
                failed = _deep_validate_trajectory_metrics(
                    failed_metrics,
                    schedule_index=first_failed,
                    label=f"rank{rank} failed gate metrics",
                )
                if (
                    failed["digest"] != final_metrics_digests[str(first_failed)]
                    or failed.get("noncompensating_all_pass") is not False
                    or "failed a noncompensating gate"
                    not in str(failure.get("failure_reason"))
                ):
                    raise GraftPhaseAShortGPUError(
                        "trajectory local original-gate failure differs"
                    )
            _assert_no_elevated_authority_or_checkpoint(failure)
        else:
            success = validate_sealed_mapping(
                trainer_success, label=f"rank{rank} trainer success receipt"
            )
            if (
                trainer_failure is not None
                or rollback_manifest is not None
                or success.get("status") != "completed_in_memory_orchestration"
                or success.get("topology", {}).get("rank") != rank
                or success.get("checkpoint_written") is not False
            ):
                raise GraftPhaseAShortGPUError(
                    "trajectory WORLD8 all-pass outcome differs"
                )
            _assert_no_elevated_authority_or_checkpoint(success)
        _assert_no_elevated_authority_or_checkpoint(result)
        admitted.append(result)

    terminal_gate_digests = {
        row["world8_terminal_coordination"]["digest"] for row in admitted
    }
    if len(terminal_gate_digests) != 1:
        raise GraftPhaseAShortGPUError(
            "trajectory ranks disagree on terminal gate coordination"
        )
    terminal_gate_manifest = admitted[0]["world8_terminal_coordination"]
    overall_no_go = terminal_gate_manifest["world_any_no_go"]
    if any(
        (row["scientific_outcome"] == "NO_GO") is not overall_no_go
        for row in admitted
    ):
        raise GraftPhaseAShortGPUError(
            "trajectory ranks disagree on global scientific outcome"
        )
    if overall_no_go:
        rollback_digests = {
            row["world8_terminal_rollback_manifest"]["digest"]
            for row in admitted
        }
        if len(rollback_digests) != 1:
            raise GraftPhaseAShortGPUError(
                "trajectory ranks disagree on terminal rollback coordination"
            )
        terminal_rollback_manifest = admitted[0][
            "world8_terminal_rollback_manifest"
        ]
    else:
        terminal_rollback_manifest = None

    family_matrices = []
    for arm in range(DP_SIZE):
        rows = admitted[arm * SP_SIZE : (arm + 1) * SP_SIZE]
        for stage_ordinal, stage in enumerate(TRAJECTORY_STAGES):
            for index_ordinal, index in enumerate(CONFIRMATION_INDICES):
                cells = [
                    row["trajectory_matrices"][stage_ordinal]["cells"][
                        index_ordinal
                    ]
                    for row in rows
                ]
                if len({cell["sp4_consensus_digest"] for cell in cells}) != 1:
                    raise GraftPhaseAShortGPUError(
                        "trajectory SP4 receipts disagree"
                    )
        if (
            len({row["local_scientific_outcome"] for row in rows}) != 1
            or len(
                {
                    canonical_json_bytes(
                        row["original_confirmation_gate"][
                            "final_stage_gate_vector"
                        ]
                    )
                    for row in rows
                }
            )
            != 1
            or len(
                {
                    row["original_confirmation_gate"]["local_terminal_path"]
                    for row in rows
                }
            )
            != 1
        ):
            raise GraftPhaseAShortGPUError(
                "trajectory family terminal outcomes disagree across SP4"
            )
        representative = rows[0]
        family_matrices.append(
            {
                "dp_arm": arm,
                "family": FAMILY_BY_DP_ARM[arm],
                "global_ranks": list(
                    range(arm * SP_SIZE, (arm + 1) * SP_SIZE)
                ),
                "scientific_outcome": representative["scientific_outcome"],
                "local_scientific_outcome": representative[
                    "local_scientific_outcome"
                ],
                "final_stage_gate_vector": representative[
                    "original_confirmation_gate"
                ]["final_stage_gate_vector"],
                "local_terminal_path": representative[
                    "original_confirmation_gate"
                ]["local_terminal_path"],
                "trajectory_matrices": representative["trajectory_matrices"],
                "per_rank_result_digests": [row["digest"] for row in rows],
                "all_six_cells_sp4_exact_consensus": True,
            }
        )
    return seal_mapping(
        {
            "schema_version": (
                "bernini-graft-phase-a-short-trajectory-world8-results-v1"
            ),
            "status": (
                "completed_observed_scientific_no_go"
                if overall_no_go
                else "completed_original_confirmation_pass"
            ),
            "scientific_outcome": "NO_GO" if overall_no_go else "ORIGINAL_GATE_PASS",
            "rank_order": list(range(WORLD_SIZE)),
            "dp2_family_order": list(FAMILY_BY_DP_ARM),
            "all_eight_local_receipts": admitted,
            "dog_human_trajectory_matrices": family_matrices,
            "world8_terminal_coordination": dict(terminal_gate_manifest),
            "world8_terminal_rollback_manifest": (
                None
                if terminal_rollback_manifest is None
                else dict(terminal_rollback_manifest)
            ),
            "all_eight_rank_coverage": True,
            "all_twelve_family_cells_sp4_exact_consensus": True,
            "terminal_collective_sequence_unified_across_world8": True,
            "all_eight_terminal_equal_initial_after_no_go": overall_no_go,
            "single_preregistered_optimizer_arm": True,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


@dataclass(frozen=True)
class DistributedTopology:
    global_rank: int
    local_rank: int
    sp_rank: int
    dp_arm: int
    world_group: Any = field(repr=False, compare=False)
    sp_group: Any = field(repr=False, compare=False)
    dp_group: Any = field(repr=False, compare=False)
    receipt: Mapping[str, Any]


def _initialize_world8_dp2sp4(init_parallel_state: Callable[..., Any]) -> DistributedTopology:
    import torch.distributed as dist
    from bernini.parallel import get_parallel_state

    try:
        launch = {
            name: int(os.environ.get(name, ""))
            for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        }
    except ValueError as error:
        raise GraftPhaseAShortGPUError("WORLD8 torchrun environment differs") from error
    if (
        launch["WORLD_SIZE"] != WORLD_SIZE
        or launch["LOCAL_WORLD_SIZE"] != WORLD_SIZE
        or launch["RANK"] != launch["LOCAL_RANK"]
        or not 0 <= launch["RANK"] < WORLD_SIZE
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != WORLD_SIZE
        or getattr(torch.version, "hip", None) is None
        or dist.is_initialized()
    ):
        raise GraftPhaseAShortGPUError("runner requires AUH ROCm WORLD8 DP2xSP4")
    global_rank = launch["RANK"]
    local_rank = launch["LOCAL_RANK"]
    dp_arm = global_rank // SP_SIZE
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=global_rank,
        world_size=WORLD_SIZE,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    state = get_parallel_state()
    sp_group = getattr(state, "ulysses_group", None)
    dp_group = getattr(state, "dp_group", None)
    sp_rank = getattr(state, "ulysses_rank", None)
    if (
        getattr(state, "world_size", None) != WORLD_SIZE
        or getattr(state, "ulysses_enabled", None) is not True
        or getattr(state, "ulysses_size", None) != SP_SIZE
        or getattr(state, "dp_size", None) != DP_SIZE
        or getattr(state, "rank", None) != global_rank
        or type(sp_rank) is not int
        or not 0 <= sp_rank < SP_SIZE
        or sp_rank != global_rank % SP_SIZE
        or getattr(state, "dp_rank", None) != dp_arm
        or dist.get_world_size(sp_group) != SP_SIZE
        or dist.get_rank(sp_group) != sp_rank
        or dist.get_world_size(dp_group) != DP_SIZE
        or dist.get_rank(dp_group) != dp_arm
        or str(dist.get_backend(sp_group)).lower() != "nccl"
        or str(dist.get_backend(dp_group)).lower() != "nccl"
    ):
        raise GraftPhaseAShortGPUError("live Bernini SP4 state differs")
    expected_sp_members = tuple(
        range(dp_arm * SP_SIZE, (dp_arm + 1) * SP_SIZE)
    )
    sp_members: list[Any] = [None] * SP_SIZE
    dist.all_gather_object(sp_members, global_rank, group=sp_group)
    if tuple(sp_members) != expected_sp_members:
        raise GraftPhaseAShortGPUError("SP4 membership differs")
    expected_dp_members = (sp_rank, sp_rank + SP_SIZE)
    dp_members: list[Any] = [None] * DP_SIZE
    dist.all_gather_object(dp_members, global_rank, group=dp_group)
    if tuple(dp_members) != expected_dp_members:
        raise GraftPhaseAShortGPUError("orthogonal DP2 membership differs")
    receipt = seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-live-world8-dp2sp4-v1",
            "world_size": WORLD_SIZE,
            "dp_size": DP_SIZE,
            "sp_size": SP_SIZE,
            "global_rank": global_rank,
            "local_rank": local_rank,
            "dp_arm": dp_arm,
            "sp_rank": sp_rank,
            "sp_members": list(expected_sp_members),
            "dp_members": list(expected_dp_members),
            "backend": "nccl",
        }
    )
    return DistributedTopology(
        global_rank=global_rank,
        local_rank=local_rank,
        sp_rank=sp_rank,
        dp_arm=dp_arm,
        world_group=dist.group.WORLD,
        sp_group=sp_group,
        dp_group=dp_group,
        receipt=receipt,
    )


def _broadcast_initial_trainables(
    rows: Sequence[tuple[str, torch.nn.Parameter]], *, world_group: Any
) -> Mapping[str, Any]:
    import torch.distributed as dist

    for _, parameter in rows:
        dist.broadcast(parameter.data, src=0, group=world_group)
    output = [
        parameter
        for name, parameter in rows
        if name.endswith(".identity_rebinder.output.weight")
    ]
    if not output or any(
        int(torch.count_nonzero(parameter.detach()).item()) != 0
        for parameter in output
    ):
        raise GraftPhaseAShortGPUError("initial output projections are not exact zero")
    digest = native_runner_v1.parameter_registry_digest(rows)
    _gather_equal(
        digest,
        group=world_group,
        count=WORLD_SIZE,
        label="initial trainable registry",
    )
    return seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-short-initial-registry-v1",
            "rank0_broadcast_before_any_adapter_forward": True,
            "parameter_count": len(rows),
            "parameter_sha256": digest,
            "zero_output_projection_count": len(output),
            "zero_output_projection_exact": True,
        }
    )


def _encode_conditions(
    *,
    tokenizer: Any,
    renderer: torch.nn.Module,
    prompt_cleaner: Callable[[str], str],
    device: torch.device,
    local: LocalFamilyRouting,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """Encode conditions from the authenticated A-lite row no-op, not v1 text."""

    assert_pinned_dependencies()
    if type(local) is not LocalFamilyRouting:
        raise GraftPhaseAShortGPUError(
            "condition encoder requires exact authenticated local routing"
        )
    fit = local.fit_row
    confirmation = local.confirmation_row
    production_rows = (
        type(fit) is source_consumer.TrainerOwnedSourceRow
        and type(confirmation) is source_consumer.TrainerOwnedSourceRow
        and local.test_only is False
    )
    test_rows = (
        type(fit).__name__ == "_TestSourceRow"
        and type(fit).__module__ == short_trainer.__name__
        and type(confirmation) is type(fit)
        and local.test_only is True
    )
    if (
        not (production_rows or test_rows)
        or local.dp_arm not in range(DP_SIZE)
        or local.family != FAMILY_BY_DP_ARM[local.dp_arm]
        or local.fit_iid != FIT_IID_BY_DP_ARM[local.dp_arm]
        or local.confirmation_iid != CONFIRMATION_IID_BY_DP_ARM[local.dp_arm]
        or fit.iid != local.fit_iid
        or confirmation.iid != local.confirmation_iid
        or fit.noop_instruction != confirmation.noop_instruction
        or fit.noop_instruction != source_consumer.NOOP_INSTRUCTION
        or type(fit.noop_instruction) is not str
        or type(confirmation.noop_instruction) is not str
        or not fit.optimizer_update_allowed
        or fit.optimizer_confirmation_only
        or confirmation.optimizer_update_allowed
        or not confirmation.optimizer_confirmation_only
        or hashlib.sha256(fit.source_bytes).hexdigest() != fit.source_sha256
        or hashlib.sha256(confirmation.source_bytes).hexdigest()
        != confirmation.source_sha256
        or fit.source_sha256 == confirmation.source_sha256
    ):
        raise GraftPhaseAShortGPUError(
            "authenticated fit/confirmation no-op routing differs"
        )

    def row_binding(row: Any, *, role: str) -> Mapping[str, Any]:
        return seal_mapping(
            {
                "schema_version": (
                    "bernini-graft-phase-a-short-condition-row-binding-v1"
                ),
                "role": role,
                "iid": row.iid,
                "split": row.split,
                "optimizer_update_allowed": row.optimizer_update_allowed,
                "optimizer_confirmation_only": row.optimizer_confirmation_only,
                "source_sha256": row.source_sha256,
                "source_size_bytes": len(row.source_bytes),
                "noop_instruction_utf8_sha256": hashlib.sha256(
                    row.noop_instruction.encode("utf-8")
                ).hexdigest(),
                "routing_digest": local.routing_digest,
                "source_release_result_digest": (
                    local.source_release_result_digest
                ),
                "pinset_digest": local.pinset_digest,
            }
        )

    fit_binding = row_binding(fit, role="optimizer_fit")
    confirmation_binding = row_binding(
        confirmation, role="optimizer_confirmation"
    )
    routed_noop = fit.noop_instruction
    noop_prompt = legacy.build_training_prompt(
        routed_noop, prompt_cleaner=prompt_cleaner
    )
    expected_noop_prompt = legacy.MV2V_SYSTEM_PROMPT + routed_noop
    negative_prompt = legacy.DEFAULT_NEGATIVE_PROMPT
    if (
        noop_prompt != expected_noop_prompt
        or not negative_prompt
        or noop_prompt == negative_prompt
    ):
        raise GraftPhaseAShortGPUError(
            "routed A-lite no-op was rewritten or negative prompt differs"
        )
    instruction = ACTION_INSTRUCTION_BY_DP_ARM[local.dp_arm]
    action_prompt = legacy.build_training_prompt(
        instruction, prompt_cleaner=prompt_cleaner
    )
    noop_ids, noop_mask = legacy._tokenize_training_prompt(tokenizer, noop_prompt)
    action_ids, action_mask = legacy._tokenize_training_prompt(
        tokenizer, action_prompt
    )
    negative_ids, negative_mask = legacy._tokenize_renderer_negative(
        tokenizer, negative_prompt
    )
    with torch.no_grad():
        noop = renderer.encode_prompt(
            noop_ids.to(device), noop_mask.to(device)
        ).detach().contiguous()
        action = renderer.encode_prompt(
            action_ids.to(device), action_mask.to(device)
        ).detach().contiguous()
        negative = renderer.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach().contiguous()
    if (
        tuple(negative.shape) != (1, 512, 4096)
        or noop.shape != negative.shape
        or action.shape != negative.shape
        or any(value.dtype != torch.bfloat16 for value in (negative, noop, action))
        or torch.equal(negative, noop)
        or torch.equal(negative, action)
        or torch.equal(noop, action)
    ):
        raise GraftPhaseAShortGPUError("negative/noop/action condition contract differs")
    receipt = seal_mapping(
        {
            "schema_version": (
                "bernini-graft-phase-a-short-routed-noop-rv2v-conditions-v1"
            ),
            "family": local.family,
            "dp_arm": local.dp_arm,
            "authenticated_routing_digest": local.routing_digest,
            "source_release_result_digest": local.source_release_result_digest,
            "source_routing_pinset_digest": local.pinset_digest,
            "fit_iid": local.fit_iid,
            "confirmation_iid": local.confirmation_iid,
            "fit_source_sha256": fit.source_sha256,
            "confirmation_source_sha256": confirmation.source_sha256,
            "fit_row_binding": dict(fit_binding),
            "confirmation_row_binding": dict(confirmation_binding),
            "fit_row_noop_instruction_utf8_sha256": hashlib.sha256(
                fit.noop_instruction.encode("utf-8")
            ).hexdigest(),
            "confirmation_row_noop_instruction_utf8_sha256": hashlib.sha256(
                confirmation.noop_instruction.encode("utf-8")
            ).hexdigest(),
            "built_noop_prompt_utf8_sha256": hashlib.sha256(
                noop_prompt.encode("utf-8")
            ).hexdigest(),
            "built_noop_prompt_is_exact_mv2v_plus_routed_row_text": True,
            "negative_prompt_utf8_sha256": hashlib.sha256(
                negative_prompt.encode("utf-8")
            ).hexdigest(),
            "negative_prompt_is_pinned_legacy_default": True,
            "legacy_v1_canonical_noop_helper_consumed": False,
            "action_instruction_utf8_sha256": hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest(),
            "built_action_prompt_utf8_sha256": hashlib.sha256(
                action_prompt.encode("utf-8")
            ).hexdigest(),
            "action_instruction_is_fixed_in_runner_source": True,
            "action_instruction_is_not_a_source_retelling": True,
            "guidance_mode": native_v2.GUIDANCE_MODE,
            "negative": tensor_identity(negative),
            "noop_positive": tensor_identity(noop),
            "action_positive": tensor_identity(action),
            "pairwise_distinct": True,
            "t2v_system_prompt_used": False,
            "target_video_used": False,
            "source_retelling_used": False,
        }
    )
    return negative, noop, action, receipt


def _run_official_gpu(
    args: argparse.Namespace, routing: source_consumer.TrainerRouting
) -> ShortGPURunnerResult:
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise GraftPhaseAShortGPUError(str(error)) from error
    if int(transformer_config.get("num_attention_heads", -1)) != 12:
        raise GraftPhaseAShortGPUError("checkpoint head count differs")
    manifest_path = Path(args.checkpoint_content_manifest).resolve(strict=True)
    if (
        file_sha256(manifest_path)
        != args.expected_checkpoint_content_manifest_sha256
    ):
        raise GraftPhaseAShortGPUError("checkpoint content manifest differs")
    inference_hashes = legacy.validate_inference_source_files(bernini_root)
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    import bernini.models.transformer_wan as transformer_wan
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if (
        SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT
        or DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT
    ):
        raise GraftPhaseAShortGPUError("official RV2V prompt constants differ")
    try:
        topology = _initialize_world8_dp2sp4(init_parallel_state)
    except Exception:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    local = route_local_family(routing, dp_arm=topology.dp_arm)
    device = torch.device("cuda", topology.local_rank)
    handle: Optional[rebinder.IdentityRebinderHandle] = None
    try:
        source_binding = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-short-source-binding-v1",
                "runner_sha256": args.expected_runner_sha256,
                "consumer_sha256": PINNED_CONSUMER_SOURCE_SHA256,
                "native_v2_sha256": PINNED_NATIVE_V2_SOURCE_SHA256,
                "short_trainer_sha256": PINNED_SHORT_TRAINER_SOURCE_SHA256,
                "native_runner_v1_sha256": PINNED_NATIVE_RUNNER_V1_SOURCE_SHA256,
                "identity_rebinder_sha256": args.expected_identity_rebinder_sha256,
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "bernini_inference_files": inference_hashes,
            }
        )
        _gather_equal(
            source_binding,
            group=topology.world_group,
            count=WORLD_SIZE,
            label="runner/vendor source binding",
        )

        checkpoint_rows: list[Any] = [None]
        if topology.global_rank == 0:
            try:
                checkpoint_rows[0] = {
                    "ok": True,
                    "identity": source_audit.validate_checkpoint_content(
                        checkpoint,
                        manifest_path,
                        expected_manifest_sha256=(
                            args.expected_checkpoint_content_manifest_sha256
                        ),
                    ),
                }
            except Exception as error:
                checkpoint_rows[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(checkpoint_rows, src=0)
        checkpoint_result = checkpoint_rows[0]
        if (
            not isinstance(checkpoint_result, Mapping)
            or checkpoint_result.get("ok") is not True
        ):
            raise GraftPhaseAShortGPUError(
                f"checkpoint content validation failed: {checkpoint_result}"
            )
        checkpoint_identity = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-short-checkpoint-v1",
                "identity": dict(checkpoint_result["identity"]),
            }
        )
        _gather_equal(
            checkpoint_identity,
            group=topology.world_group,
            count=WORLD_SIZE,
            label="checkpoint content",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint),
            subfolder="tokenizer",
            **legacy.tokenizer_load_kwargs(),
        )
        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(
            config.to_dict(), checkpoint
        )
        if float(config.shift) != native_generation.FLOW_SHIFT or config.use_unipc is not True:
            raise GraftPhaseAShortGPUError("renderer is not pinned UniPC shift5")
        renderer = BerniniRendererModel(config)
        renderer.eval().requires_grad_(False)
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval().requires_grad_(False).to(device)
        fit = _encode_source_state(
            row=local.fit_row,
            vae=vae,
            vae_encode=_vae_encode,
            device=device,
            sp_group=topology.sp_group,
        )
        confirmation = _encode_source_state(
            row=local.confirmation_row,
            vae=vae,
            vae_encode=_vae_encode,
            device=device,
            sp_group=topology.sp_group,
        )
        vae.to("cpu")
        del vae
        torch.cuda.empty_cache()

        renderer.to(device)
        diffusion = source_audit.resolve_diffusion_core(renderer)
        transformer = diffusion.transformer
        if transformer is None or getattr(diffusion, "transformer_2", None) is not None:
            raise GraftPhaseAShortGPUError("runner requires one pinned transformer_1")
        renderer.eval().requires_grad_(False)
        wan_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
        )
        schedule = Exact40CoordinateRegistry(diffusion.scheduler, device=device)
        _gather_equal(
            schedule.receipt,
            group=topology.world_group,
            count=WORLD_SIZE,
            label="exact40 schedule registry",
        )
        negative, noop, action, condition_receipt = _encode_conditions(
            tokenizer=tokenizer,
            renderer=renderer,
            prompt_cleaner=prompt_clean,
            device=device,
            local=local,
        )
        _gather_equal(
            condition_receipt,
            group=topology.sp_group,
            count=SP_SIZE,
            label="family RV2V conditions",
        )
        renderer.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()

        adapter_off_baseline = _capture_adapter_off_baseline(
            diffusion=diffusion,
            transformer=transformer,
            schedule=schedule,
            confirmation=confirmation,
            negative_condition=negative,
            noop_condition=noop,
            action_condition=action,
        )
        _gather_equal(
            adapter_off_baseline,
            group=topology.sp_group,
            count=SP_SIZE,
            label="preinstall adapter-off baseline",
        )
        base_rows = native_runner_v1._base_parameter_rows(transformer)  # noqa: SLF001
        base_before = short_chunked_parameter_registry_digest(base_rows)
        handle = rebinder.install_identity_rebinder_v1(
            transformer,
            runtime_source_commit=bernini_revision,
            model_revision=rebinder.PINNED_BERNINI_MODEL_REVISION,
            checkpoint_manifest_sha256=(
                args.expected_checkpoint_content_manifest_sha256
            ),
        )
        transformer.eval()
        handle.atlas_encoder.eval()
        trainable_rows = handle.trainable_named_parameters()
        initialization = _broadcast_initial_trainables(
            trainable_rows, world_group=topology.world_group
        )
        route_factory = ShortTrainingAtlasRouteFactory(
            handle=handle,
            sp_rank=topology.sp_rank,
            sp_group=topology.sp_group,
        )
        route_receipt = _official_forward_route_receipt()
        bindings = native_v2.authenticate_pinned_native_bindings(
            diffusion=diffusion,
            transformer=transformer,
            named_trainable_parameters=trainable_rows,
            external_trainable_owner_modules={
                "atlas_encoder": handle.atlas_encoder
            },
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
            transformer_wan_path=Path(transformer_wan.__file__).resolve(
                strict=True
            ),
            bernini_commit=bernini_revision,
            forward_context_factory=route_factory,
            forward_route_receipt=route_receipt,
        )
        backend = short_trainer.authenticate_torch_distributed_world8_dp2sp4(
            world_group=topology.world_group,
            sp_group=topology.sp_group,
            dp_group=topology.dp_group,
        )
        runtime = OfficialShortRuntime(
            local=local,
            bindings=bindings,
            diffusion=diffusion,
            transformer=transformer,
            handle=handle,
            route_factory=route_factory,
            schedule=schedule,
            fit=fit,
            confirmation=confirmation,
            negative_condition=negative,
            noop_condition=noop,
            action_condition=action,
            sp_rank=topology.sp_rank,
            sp_group=topology.sp_group,
            adapter_off_baseline=adapter_off_baseline,
        )
        services = authenticate_official_services(runtime)
        result = execute_authenticated_trajectory_diagnostic(
            routing=routing,
            bindings=bindings,
            collectives=backend,
            services=services,
            runtime=runtime,
        )
        schedule.assert_unchanged()
        if any(parameter.grad is not None for _, parameter in base_rows):
            raise GraftPhaseAShortGPUError("frozen base acquired a gradient")
        base_after = short_chunked_parameter_registry_digest(base_rows)
        if base_after != base_before:
            raise GraftPhaseAShortGPUError("frozen base bytes changed")
        local_packet = {
            "global_rank": topology.global_rank,
            "result_digest": result.receipt["digest"],
            "local_result": dict(result.receipt),
        }
        if (
            len(canonical_json_bytes(local_packet)) >= MAX_FULL_LOCAL_RESULT_PACKET_BYTES
            or pickle.loads(pickle.dumps(local_packet, protocol=5)) != local_packet
        ):
            raise GraftPhaseAShortGPUError("local full-result packet is not bounded pickle-safe")
        full_packets: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(
            full_packets,
            local_packet,
        )
        full_world8_results = assemble_trajectory_world8_results(full_packets)
        summaries = [
            {
                "global_rank": packet["global_rank"],
                "dp_arm": packet["local_result"]["topology"]["dp_arm"],
                "sp_rank": packet["local_result"]["topology"]["sp_rank"],
                "family": packet["local_result"]["topology"]["family"],
                "result_digest": packet["result_digest"],
                "scientific_outcome": packet["local_result"][
                    "scientific_outcome"
                ],
                "trajectory_matrix_digests": [
                    row["digest"]
                    for row in packet["local_result"]["trajectory_matrices"]
                ],
                "adapter_off_parity_digest": packet["local_result"][
                    "adapter_off_parity"
                ]["digest"],
            }
            for packet in full_packets
        ]
        if [row.get("global_rank") for row in summaries] != list(range(WORLD_SIZE)):
            raise GraftPhaseAShortGPUError("WORLD8 result coverage differs")
        assembled = seal_mapping(
            {
                "schema_version": (
                    "bernini-graft-phase-a-short-trajectory-world8-result-set-v1"
                ),
                "status": full_world8_results["status"],
                "scientific_outcome": full_world8_results[
                    "scientific_outcome"
                ],
                "world8_rows": summaries,
                "world8_full_results": dict(full_world8_results),
                "topology_receipt": dict(topology.receipt),
                "source_binding": dict(source_binding),
                "checkpoint_identity": dict(checkpoint_identity),
                "initialization": dict(initialization),
                "local_result": dict(result.receipt),
                "base_sha256_before": base_before,
                "base_sha256_after": base_after,
                "base_bytes_unchanged": True,
                "base_gradients_all_none": True,
                "diagnostic_completed_even_for_scientific_no_go": True,
                "single_preregistered_optimizer_arm": True,
                "wan_diffusion_sha256": wan_sha,
                "transformer_wan_sha256": file_sha256(
                    Path(transformer_wan.__file__).resolve(strict=True)
                ),
                "runtime_versions": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "diffusers": diffusers_version,
                    "transformers": transformers_version,
                },
                "checkpoint_written": False,
                "publication_performed": False,
                **_false_authority(),
            }
        )
        dist.barrier()
        if topology.global_rank == 0:
            print(canonical_json_bytes(assembled).decode("ascii"), flush=True)
        dist.barrier()
        return result
    finally:
        if (
            handle is not None
            and not handle.restored
            and rebinder.active_route() is None
        ):
            handle.restore()
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    routing = consume_authenticated_source_routing(args)
    try:
        _run_official_gpu(args, routing)
        return 0
    except GraftPhaseAShortGPUError as error:
        diagnostic = error.diagnostic_receipt
        if diagnostic is not None:
            print(canonical_json_bytes(diagnostic).decode("ascii"), flush=True)
        raise


__all__ = [
    "ADAPTER_OFF_PARITY_INDICES",
    "ADAPTER_OFF_PARITY_SCHEMA_VERSION",
    "AUTHORITY_FIELDS",
    "AuthenticatedRunnerServices",
    "CONFIRMATION_FIELDS_SCHEMA_VERSION",
    "CONFIRMATION_INDICES",
    "ConfirmationFieldSet",
    "DP_SIZE",
    "FAILURE_SCHEMA_VERSION",
    "GraftPhaseAShortGPUError",
    "LocalFamilyRouting",
    "PINNED_CONSUMER_SOURCE_SHA256",
    "PINNED_NATIVE_V2_SOURCE_SHA256",
    "PINNED_SHORT_TRAINER_SOURCE_SHA256",
    "PINNED_SHORT_TRAINER_EXECUTION_RUNTIME_SHA256",
    "PARITY_BRANCH_ROLES",
    "SCHEMA_VERSION",
    "SP_SIZE",
    "ShortGPURunnerResult",
    "TRAJECTORY_MATRIX_SCHEMA_VERSION",
    "TRAJECTORY_STAGES",
    "TrajectoryConfirmationPlan",
    "UPDATE_INDICES",
    "WORLD_SIZE",
    "assert_pinned_dependencies",
    "assemble_world8_local_results",
    "assemble_trajectory_world8_results",
    "authenticate_cpu_test_services",
    "build_confirmation_provenance",
    "build_parser",
    "canonical_json_bytes",
    "consume_authenticated_source_routing",
    "execute_authenticated_short_run",
    "execute_authenticated_trajectory_diagnostic",
    "keyed_fresh_gaussian",
    "main",
    "object_sha256",
    "route_local_family",
    "seal_mapping",
    "short_chunked_parameter_registry_digest",
    "short_chunked_tensor_identity",
    "tensor_identity",
    "validate_adapter_off_parity",
    "validate_cli",
    "validate_confirmation_field_set",
]


if __name__ == "__main__":
    raise SystemExit(main())
