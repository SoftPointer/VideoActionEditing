#!/usr/bin/env python3
"""Train the exact-81 Bernini RAMP C0 oracle-program adapter on WORLD8.

The single-node topology is frozen to DP2 x Ulysses-SP4.  Ranks 0--3 run
program arm A and ranks 4--7 run program arm B.  The two arms share source,
generic text, Gaussian epsilon and the exact noisy target state at sigma=1;
the only model-input intervention is a different 21-token temporal transport.

Raw donor VAE tensors are hash-audited while loading the materialized shards
and then discarded.  Bernini receives exactly ``source + 21 motion + target``
patches.  Only the learned role embedding, program projector and target-row
self-attention Q LoRA are trainable.  This is a synthetic temporal-program C0
engineering canary, not evidence of semantic action editing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import mdr_exact_motion_analogy as analogy  # noqa: E402
import ramp_same_state_route_objective as objective  # noqa: E402
import ramp_target_row_lora as route  # noqa: E402
import train_lora as legacy  # noqa: E402


METHOD_NAME = "bernini-ramp-c0-exact81-oracle-program"
PAIR_CONFIG_SCHEMA = "bernini-ramp-c0-paired-program-config-v1"
RUN_RECEIPT_SCHEMA = "bernini-ramp-c0-world8-training-receipt-v1"
CHECKPOINT_SCHEMA = "bernini-ramp-c0-adapter-state-v1"
HISTORY_SCHEMA = "bernini-ramp-c0-step-history-v1"
MATERIALIZED_ROW_FORMAT = "bernini-ramp-motion-analogy-vae-row-v1"
SAMPLE_RECEIPT_FORMAT = "bernini-ramp-motion-analogy-vae-receipt-v1"
ROLE_TO_BLOB_FIELD = {
    "source_A": "source_a_vae_posterior_blob",
    "donor_before_B": "donor_b_before_vae_posterior_blob",
    "donor_after_TB": "donor_b_after_vae_posterior_blob",
    "target_TA": "target_ta_vae_posterior_blob",
}

WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FRAME_COUNT = 81
LATENT_PHASES = 21
TIMESTEP = 1000
SIGMA = 1.0
LORA_RANK = 8
LORA_ALPHA = 8.0
CANARY_STEPS = 1
C0_STEPS = 16
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 20260808
ROUTE_WEIGHT = 0.5
DONOR_IDENTITY_WEIGHT = 0.1
ORDER_WEIGHT = 0.05

SP_GROUP_RANKS = ((0, 1, 2, 3), (4, 5, 6, 7))
DP_GROUP_RANKS = ((0, 4), (1, 5), (2, 6), (3, 7))

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


class RAMPTrainingError(RuntimeError):
    """Raised before an ambiguous optimizer step or artifact publication."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RAMPTrainingError(f"value is not canonical finite JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _materialized_row_digest(row: Mapping[str, Any]) -> str:
    """Recompute the materializer's sealed digest without importing its stack."""

    candidate = {
        key: value
        for key, value in row.items()
        if key not in set(ROLE_TO_BLOB_FIELD.values()) | {"materialized_row_digest"}
    }
    candidate["vae_posterior_blob_sha256"] = {
        role: hashlib.sha256(bytes(row[field])).hexdigest()
        for role, field in ROLE_TO_BLOB_FIELD.items()
    }
    return object_sha256(candidate)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            snapshot = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        named_after = path.stat()
    except OSError as error:
        raise RAMPTrainingError(f"cannot hash {path}: {error}") from error
    observed = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    named = (
        named_after.st_dev,
        named_after.st_ino,
        named_after.st_size,
        named_after.st_mtime_ns,
    )
    if snapshot != observed or snapshot != named:
        raise RAMPTrainingError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise RAMPTrainingError("tensor digest requires one materialized tensor")
    tensor = value.detach().contiguous().cpu()
    metadata = canonical_json_bytes(
        {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    )
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise RAMPTrainingError(f"{label} must be a lowercase SHA-256")
    return value


def _plain_absolute_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise RAMPTrainingError(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    mode = resolved.lstat().st_mode
    if resolved != requested or not stat.S_ISREG(mode) or resolved.is_symlink():
        raise RAMPTrainingError(f"{label} must be a canonical plain file")
    return resolved


def _strict_json_load_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    """Parse exactly the byte string whose SHA-256 was authenticated.

    Accepting a path here would permit a hash-then-reopen race: the bytes used
    for authorization could differ from the bytes used for configuration.
    """

    def reject_constant(value: str) -> None:
        raise RAMPTrainingError(f"{label} contains non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RAMPTrainingError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RAMPTrainingError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, dict):
        raise RAMPTrainingError(f"{label} root must be one object")
    return value


def _read_bound_bytes(
    path: Path, expected_sha256: str, *, label: str
) -> tuple[bytes, str]:
    """Read one stable snapshot and bind all downstream parsing to its bytes."""

    expected = _require_sha256(expected_sha256, label=f"{label} expected SHA")
    try:
        before = path.stat()
        snapshot = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        raw = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise RAMPTrainingError(f"cannot read {label}: {error}") from error
    if snapshot != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RAMPTrainingError(f"{label} changed while reading")
    if len(raw) != before.st_size:
        raise RAMPTrainingError(f"{label} byte count differs from stable stat")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise RAMPTrainingError(
            f"{label} SHA differs: expected={expected} actual={actual}"
        )
    return raw, actual


@dataclass(frozen=True)
class BoundArmFiles:
    arm: str
    parquet: Path
    parquet_sha256: str
    receipt: Path
    receipt_sha256: str


@dataclass(frozen=True)
class PairedProgramConfig:
    path: Path
    sha256: str
    arm_a: BoundArmFiles
    arm_b: BoundArmFiles


@dataclass(frozen=True)
class MaterializedArm:
    arm: str
    row_id: str
    source_video_sha256: str
    donor_video_sha256: str
    source_blob: bytes
    target_blob: bytes
    source_blob_sha256: str
    donor_before_blob_sha256: str
    donor_after_blob_sha256: str
    target_blob_sha256: str
    program_kind: str
    program_parameter_hex: str
    program_digest: str
    instruction: str
    bucket_hw: tuple[int, int]
    posterior_shape: tuple[int, ...]
    vae_identity_digest: str
    materialized_row_digest: str
    parquet_path: str
    parquet_sha256: str
    receipt_path: str
    receipt_sha256: str


def _parse_arm_files(name: str, value: Any) -> BoundArmFiles:
    if not isinstance(value, dict) or frozenset(value) != {
        "parquet_path",
        "parquet_sha256",
        "receipt_path",
        "receipt_sha256",
    }:
        raise RAMPTrainingError(f"pair config {name} has a non-closed field set")
    parquet = _plain_absolute_file(value["parquet_path"], label=f"{name} parquet")
    receipt = _plain_absolute_file(value["receipt_path"], label=f"{name} receipt")
    parquet_sha = _require_sha256(value["parquet_sha256"], label=f"{name} parquet SHA")
    receipt_sha = _require_sha256(value["receipt_sha256"], label=f"{name} receipt SHA")
    if file_sha256(parquet) != parquet_sha or file_sha256(receipt) != receipt_sha:
        raise RAMPTrainingError(f"pair config {name} artifact SHA-256 differs")
    return BoundArmFiles(name, parquet, parquet_sha, receipt, receipt_sha)


def load_pair_config(path: str | Path, expected_sha256: str) -> PairedProgramConfig:
    config_path = _plain_absolute_file(path, label="RAMP pair config")
    expected = _require_sha256(expected_sha256, label="expected pair config SHA")
    raw, actual = _read_bound_bytes(
        config_path, expected, label="RAMP pair config"
    )
    value = _strict_json_load_bytes(raw, label="RAMP pair config")
    if frozenset(value) != {"schema_version", "arm_a", "arm_b"}:
        raise RAMPTrainingError("pair config must contain only schema_version/arm_a/arm_b")
    if value.get("schema_version") != PAIR_CONFIG_SCHEMA:
        raise RAMPTrainingError("pair config schema differs")
    arm_a = _parse_arm_files("arm_a", value["arm_a"])
    arm_b = _parse_arm_files("arm_b", value["arm_b"])
    if arm_a.parquet == arm_b.parquet or arm_a.receipt == arm_b.receipt:
        raise RAMPTrainingError("program arms must bind distinct shard/receipt files")
    return PairedProgramConfig(config_path, actual, arm_a, arm_b)


def _sealed_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw, _ = _read_bound_bytes(
        path, expected_sha256, label="materialized sample receipt"
    )
    value = _strict_json_load_bytes(raw, label="materialized sample receipt")
    declared = _require_sha256(value.get("receipt_digest"), label="sample receipt digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        raise RAMPTrainingError("sample receipt embedded digest differs")
    if (
        value.get("schema_version") != SAMPLE_RECEIPT_FORMAT
        or value.get("complete") is not True
        or value.get("frame_count") != FRAME_COUNT
        or value.get("latent_frame_count") != LATENT_PHASES
        or value.get("construction")
        != "source=A,donor_packet=(B,T(B)),target=T(A)"
    ):
        raise RAMPTrainingError("materialized sample receipt contract differs")
    forbidden_false = (
        "external_target_accepted",
        "paired_action_dataset_used",
        "mask_flow_pose_track_box_trajectory_used",
        "direct_21_phase_permutation_authorized",
        "posterior_sample_materialized",
        "downstream_independent_posterior_sampling_authorized",
        "training_authorized",
        "action_training_authorized",
        "natural_semantic_action_learned",
        "scientific_claim_authorized",
    )
    if any(value.get(name) is not False for name in forbidden_false):
        raise RAMPTrainingError("materialized sample receipt authorization surface differs")
    return value


def _read_one_parquet(files: BoundArmFiles) -> MaterializedArm:
    receipt = _sealed_receipt(files.receipt, files.receipt_sha256)
    before = files.parquet.stat()
    snapshot = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(files.parquet).to_pylist()
    except Exception as error:
        raise RAMPTrainingError(f"cannot deserialize {files.arm} parquet: {error}") from error
    after = files.parquet.stat()
    if snapshot != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RAMPTrainingError(f"{files.arm} parquet changed while reading")
    if file_sha256(files.parquet) != files.parquet_sha256:
        raise RAMPTrainingError(f"{files.arm} parquet changed after deserialization")
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise RAMPTrainingError(f"{files.arm} parquet must contain exactly one row")
    row = rows[0]
    required = {
        "schema_version",
        "row_id",
        "source_video_sha256",
        "donor_video_sha256",
        "program_kind",
        "program_parameter_hex",
        "program_digest",
        "generic_instruction",
        "motion_analogy_receipt_json",
        "media_contract_json",
        "vae_identity_json",
        "vae_metadata_json",
        "posterior_role_order_json",
        "target_origin",
        "shared_i0_used",
        "external_target_accepted",
        "mask_flow_pose_track_box_trajectory_used",
        "direct_21_phase_permutation_authorized",
        "training_authorized",
        "training_use_forbidden",
        "action_training_authorized",
        "scientific_claim_authorized",
        "source_a_vae_posterior_blob",
        "donor_b_before_vae_posterior_blob",
        "donor_b_after_vae_posterior_blob",
        "target_ta_vae_posterior_blob",
        "materialized_row_digest",
    }
    if not required.issubset(row):
        raise RAMPTrainingError(f"{files.arm} parquet is missing materializer fields")
    if row.get("schema_version") != MATERIALIZED_ROW_FORMAT:
        raise RAMPTrainingError(f"{files.arm} materialized row schema differs")
    if _materialized_row_digest(row) != row.get("materialized_row_digest"):
        raise RAMPTrainingError(f"{files.arm} materialized row digest differs")
    for name in (
        "shared_i0_used",
        "external_target_accepted",
        "mask_flow_pose_track_box_trajectory_used",
        "direct_21_phase_permutation_authorized",
        "training_authorized",
        "action_training_authorized",
        "scientific_claim_authorized",
    ):
        if row.get(name) is not False:
            raise RAMPTrainingError(f"{files.arm} forbidden field {name} became true")
    if row.get("training_use_forbidden") is not True:
        raise RAMPTrainingError(f"{files.arm} upstream training-use warning disappeared")

    receipt_input = receipt.get("input")
    receipt_program = receipt.get("program")
    if (
        not isinstance(receipt_input, dict)
        or not isinstance(receipt_input.get("source_A"), dict)
        or not isinstance(receipt_input.get("donor_B"), dict)
        or receipt_input["source_A"].get("sha256") != row.get("source_video_sha256")
        or receipt_input["donor_B"].get("sha256") != row.get("donor_video_sha256")
        or receipt_input.get("source_and_donor_paths_distinct") is not True
        or receipt_input.get("source_and_donor_sha256_distinct") is not True
        or receipt_input.get("external_target") is not None
        or receipt.get("four_independent_VAE_encode_calls") is not True
    ):
        raise RAMPTrainingError(f"{files.arm} receipt input construction differs")
    if (
        not isinstance(receipt_program, dict)
        or receipt_program.get("kind") != row.get("program_kind")
        or receipt_program.get("parameter_hex") != row.get("program_parameter_hex")
        or receipt_program.get("digest") != row.get("program_digest")
        or receipt.get("materialized_row_digest") != row.get("materialized_row_digest")
    ):
        raise RAMPTrainingError(f"{files.arm} receipt program/row binding differs")

    try:
        motion_receipt = json.loads(row["motion_analogy_receipt_json"])
        media = json.loads(row["media_contract_json"])
        vae_identity = json.loads(row["vae_identity_json"])
        vae_metadata = json.loads(row["vae_metadata_json"])
        role_order = json.loads(row["posterior_role_order_json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise RAMPTrainingError(f"{files.arm} embedded JSON is invalid") from error
    if role_order != list(ROLE_TO_BLOB_FIELD):
        raise RAMPTrainingError(f"{files.arm} posterior role order differs")
    declared_motion_digest = motion_receipt.get("receipt_digest")
    unsigned_motion = dict(motion_receipt)
    unsigned_motion.pop("receipt_digest", None)
    if object_sha256(unsigned_motion) != declared_motion_digest:
        raise RAMPTrainingError(f"{files.arm} motion builder receipt differs")
    bucket = media.get("source_derived_bucket_hw")
    if (
        not isinstance(bucket, list)
        or len(bucket) != 2
        or any(type(item) is not int or item <= 0 or item % 16 for item in bucket)
    ):
        raise RAMPTrainingError(f"{files.arm} bucket geometry differs")
    if receipt.get("source_derived_bucket_hw") != bucket:
        raise RAMPTrainingError(f"{files.arm} receipt bucket binding differs")
    metadata = vae_metadata.get("source_A")
    if not isinstance(metadata, dict):
        raise RAMPTrainingError(f"{files.arm} source posterior metadata is absent")
    posterior_shape = metadata.get("posterior_parameters_shape")
    if (
        not isinstance(posterior_shape, list)
        or len(posterior_shape) != 5
        or posterior_shape[0] != 1
        or posterior_shape[1] != 32
        or posterior_shape[2] != LATENT_PHASES
        or posterior_shape[3] % 2
        or posterior_shape[4] % 2
    ):
        raise RAMPTrainingError(f"{files.arm} posterior geometry differs")

    blobs: dict[str, bytes] = {}
    receipt_hashes = receipt.get("vae_posterior_blob_sha256")
    if not isinstance(receipt_hashes, dict):
        raise RAMPTrainingError(f"{files.arm} receipt has no posterior hashes")
    for role, field in ROLE_TO_BLOB_FIELD.items():
        blob = row.get(field)
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise RAMPTrainingError(f"{files.arm} {role} posterior blob is invalid")
        value = bytes(blob)
        if hashlib.sha256(value).hexdigest() != receipt_hashes.get(role):
            raise RAMPTrainingError(f"{files.arm} {role} posterior blob SHA differs")
        blobs[role] = value
    if receipt.get("parquet_sha256") != files.parquet_sha256:
        raise RAMPTrainingError(f"{files.arm} receipt does not bind its parquet")
    if receipt.get("row_id") != row.get("row_id"):
        raise RAMPTrainingError(f"{files.arm} receipt row id differs")
    vae_digest = vae_identity.get("vae_identity_digest")
    unsigned_vae = dict(vae_identity)
    unsigned_vae.pop("vae_identity_digest", None)
    if object_sha256(unsigned_vae) != vae_digest:
        raise RAMPTrainingError(f"{files.arm} VAE identity digest differs")

    result = MaterializedArm(
        arm=files.arm,
        row_id=str(row["row_id"]),
        source_video_sha256=str(row["source_video_sha256"]),
        donor_video_sha256=str(row["donor_video_sha256"]),
        source_blob=blobs["source_A"],
        target_blob=blobs["target_TA"],
        source_blob_sha256=receipt_hashes["source_A"],
        donor_before_blob_sha256=receipt_hashes["donor_before_B"],
        donor_after_blob_sha256=receipt_hashes["donor_after_TB"],
        target_blob_sha256=receipt_hashes["target_TA"],
        program_kind=str(row["program_kind"]),
        program_parameter_hex=str(row["program_parameter_hex"]),
        program_digest=str(row["program_digest"]),
        instruction=str(row["generic_instruction"]),
        bucket_hw=(int(bucket[0]), int(bucket[1])),
        posterior_shape=tuple(int(item) for item in posterior_shape),
        vae_identity_digest=str(vae_digest),
        materialized_row_digest=str(row["materialized_row_digest"]),
        parquet_path=str(files.parquet),
        parquet_sha256=files.parquet_sha256,
        receipt_path=str(files.receipt),
        receipt_sha256=files.receipt_sha256,
    )
    # Donor bytes are deliberately absent from MaterializedArm.  The local
    # temporary row/blob mappings die here, before any model input is built.
    del row, rows, blobs
    return result


def validate_pair_semantics(arm_a: MaterializedArm, arm_b: MaterializedArm) -> dict[str, Any]:
    """Require a matched A,B pair whose only conditioning change is program."""

    held_equal = {
        "source_video_sha256": arm_a.source_video_sha256 == arm_b.source_video_sha256,
        "donor_video_sha256": arm_a.donor_video_sha256 == arm_b.donor_video_sha256,
        "source_blob_sha256": arm_a.source_blob_sha256 == arm_b.source_blob_sha256,
        "donor_before_blob_sha256": (
            arm_a.donor_before_blob_sha256 == arm_b.donor_before_blob_sha256
        ),
        "instruction": arm_a.instruction == arm_b.instruction,
        "bucket_hw": arm_a.bucket_hw == arm_b.bucket_hw,
        "posterior_shape": arm_a.posterior_shape == arm_b.posterior_shape,
        "vae_identity": arm_a.vae_identity_digest == arm_b.vae_identity_digest,
    }
    if not all(held_equal.values()):
        raise RAMPTrainingError(
            f"paired program arms differ outside the program intervention: {held_equal}"
        )
    if arm_a.row_id == arm_b.row_id or arm_a.program_digest == arm_b.program_digest:
        raise RAMPTrainingError("paired program arms must have distinct rows/programs")
    if arm_a.target_blob_sha256 == arm_b.target_blob_sha256:
        raise RAMPTrainingError("distinct programs produced byte-equal target posteriors")
    if arm_a.instruction != analogy.GENERIC_DONOR_INSTRUCTION:
        raise RAMPTrainingError("C0 requires the committed generic donor-follow instruction")
    return {
        "held_equal": held_equal,
        "changed_conditioning_only": "21_token_temporal_transport",
        "program_a": arm_a.program_digest,
        "program_b": arm_b.program_digest,
        "target_supervision_differs": True,
        "raw_donor_transformer_tokens": 0,
    }


@dataclass(frozen=True)
class DistributedContract:
    world_size: int
    rank: int
    local_rank: int
    local_world_size: int

    @property
    def arm_index(self) -> int:
        return self.rank // SP_SIZE

    @property
    def sp_rank(self) -> int:
        return self.rank % SP_SIZE


@dataclass(frozen=True)
class ParallelContext:
    contract: DistributedContract
    world_group: Any
    sp_group: Any
    dp_group: Any


def distributed_contract(environment: Mapping[str, str] = os.environ) -> DistributedContract:
    values: dict[str, int] = {}
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        raw = environment.get(name)
        if raw is None or not raw.isdecimal():
            raise RAMPTrainingError(f"{name} must be a decimal integer")
        values[name] = int(raw)
    if values["WORLD_SIZE"] != WORLD_SIZE or values["LOCAL_WORLD_SIZE"] != WORLD_SIZE:
        raise RAMPTrainingError("RAMP C0 requires one exact WORLD8 node")
    if values["RANK"] != values["LOCAL_RANK"] or not 0 <= values["RANK"] < WORLD_SIZE:
        raise RAMPTrainingError("single-node rank/local-rank mapping differs")
    return DistributedContract(
        values["WORLD_SIZE"], values["RANK"], values["LOCAL_RANK"], values["LOCAL_WORLD_SIZE"]
    )


def initialise_distributed(contract: DistributedContract) -> Any:
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise RAMPTrainingError("WORLD8 C0 requires ROCm-visible accelerators")
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RAMPTrainingError("visible accelerator count differs from WORLD8")
    torch.cuda.set_device(contract.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    if dist.get_world_size() != WORLD_SIZE or dist.get_rank() != contract.rank:
        raise RAMPTrainingError("initialized RCCL world differs from torchrun")
    return torch.device("cuda", contract.local_rank)


def _group_members(group: Any, expected: tuple[int, ...]) -> None:
    import torch.distributed as dist

    gathered: list[Any] = [None] * len(expected)
    dist.all_gather_object(gathered, dist.get_rank(), group=group)
    if tuple(gathered) != expected:
        raise RAMPTrainingError(f"process group members differ: {gathered} != {expected}")


def validate_parallel_state(contract: DistributedContract, state: Any) -> ParallelContext:
    import torch.distributed as dist

    if (
        getattr(state, "world_size", None) != WORLD_SIZE
        or getattr(state, "ulysses_size", None) != SP_SIZE
        or getattr(state, "dp_size", None) != DP_SIZE
        or getattr(state, "rank", None) != contract.rank
        or getattr(state, "ulysses_rank", None) != contract.sp_rank
        or getattr(state, "dp_rank", None) != contract.arm_index
    ):
        raise RAMPTrainingError("Bernini DP2 x SP4 state differs")
    _group_members(state.ulysses_group, SP_GROUP_RANKS[contract.arm_index])
    _group_members(state.dp_group, DP_GROUP_RANKS[contract.sp_rank])
    return ParallelContext(contract, dist.group.WORLD, state.ulysses_group, state.dp_group)


def _world_all_true(value: bool, *, group: Any) -> bool:
    import torch
    import torch.distributed as dist

    probe = torch.tensor(int(value), dtype=torch.int32, device="cuda")
    dist.all_reduce(probe, op=dist.ReduceOp.MIN, group=group)
    return bool(probe.item())


def _digest_consensus(value: str, *, group: Any, expected_count: int, label: str) -> str:
    import torch.distributed as dist

    gathered: list[Any] = [None] * expected_count
    dist.all_gather_object(gathered, value, group=group)
    if any(item != value for item in gathered):
        raise RAMPTrainingError(f"{label} differs across replicated ranks")
    return value


def _load_posterior_mode(blob: bytes, mean: Any, std: Any) -> Any:
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    try:
        parameters = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    except TypeError:
        parameters = torch.load(io.BytesIO(blob), map_location="cpu")
    if (
        not isinstance(parameters, torch.Tensor)
        or parameters.dtype != torch.float32
        or parameters.requires_grad
        or parameters.ndim != 5
        or tuple(parameters.shape[:3]) != (1, 32, LATENT_PHASES)
        or not parameters.is_contiguous()
        or not bool(torch.isfinite(parameters).all().item())
    ):
        raise RAMPTrainingError("posterior parameters must be detached contiguous FP32 [1,32,21,H,W]")
    mode = DiagonalGaussianDistribution(parameters).mode().squeeze(0).float()
    mode = ((mode - mean) / std).detach().contiguous()
    if tuple(mode.shape[:2]) != (16, LATENT_PHASES) or not bool(torch.isfinite(mode).all().item()):
        raise RAMPTrainingError("normalized posterior mode geometry differs")
    return mode


def pack_latent_patches(latent: Any) -> Any:
    """Pack [16,21,H,W] in Bernini's exact t,h,w,c,pt,ph,pw order."""

    import torch

    if (
        not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float32
        or latent.ndim != 4
        or tuple(latent.shape[:2]) != (16, LATENT_PHASES)
        or int(latent.shape[2]) % 2
        or int(latent.shape[3]) % 2
    ):
        raise RAMPTrainingError("latent must be FP32 [16,21,evenH,evenW]")
    channels, frames, height, width = (int(item) for item in latent.shape)
    patches = (
        latent.reshape(channels, frames, height // 2, 2, width // 2, 2)
        .permute(1, 2, 4, 0, 3, 5)
        .reshape(frames * (height // 2) * (width // 2), channels, 1, 2, 2)
    )
    return patches.contiguous()


def packed_output_field(patches: Any) -> Any:
    """Convert [N,C,pt,ph,pw] to Wan output order [1,N,pt*ph*pw*C]."""

    import torch

    if (
        not isinstance(patches, torch.Tensor)
        or patches.ndim != 5
        or tuple(patches.shape[1:]) != (16, 1, 2, 2)
    ):
        raise RAMPTrainingError("packed output requires [N,16,1,2,2]")
    return patches.permute(0, 2, 3, 4, 1).reshape(1, int(patches.shape[0]), 64).contiguous()


def _program_for_arm(arm: MaterializedArm) -> tuple[Any, Any, Any]:
    import torch

    try:
        parameter = float.fromhex(arm.program_parameter_hex)
    except ValueError as error:
        raise RAMPTrainingError("program_parameter_hex is invalid") from error
    program = analogy.TemporalProgram(arm.program_kind, parameter)
    if program.digest != arm.program_digest:
        raise RAMPTrainingError("materialized program digest differs from committed core")
    transport = route.latent_phase_transport(program.output_to_input)
    patches = route.oracle_program_patches(transport)
    if not torch.equal(route.recover_oracle_transport(patches), transport):
        raise RAMPTrainingError("oracle program serialization roundtrip differs")
    return program, transport, patches


def _tokenize_generic_instruction(tokenizer: Any, instruction: str, device: Any) -> dict[str, Any]:
    from bernini.training.data import encode_renderer_messages

    messages = [
        {"type": "video", "has_loss": 0},
        {"type": "text", "has_loss": 0, "text": instruction},
        {"type": "video_gen", "has_loss": 1},
    ]
    tokenized = encode_renderer_messages(messages, tokenizer, "mv2v", False, False, False)
    if (
        tokenized["vae_type_list"].tolist() != [1, 1]
        or tokenized["video_vit_mask"].tolist() != [False, True]
        or tokenized["video_drop_mask"].tolist() != [False, False]
    ):
        raise RAMPTrainingError("official tokenizer did not preserve source/target video roles")
    fields = {
        "input_ids": tokenized["input_ids"].unsqueeze(0).to(device),
        "attention_mask": tokenized["attention_mask"].unsqueeze(0).to(device),
        "t5_input_lens": tokenized["t5_input_lens"].unsqueeze(0).to(device),
    }
    return fields


@dataclass(frozen=True)
class PreparedModelInputs:
    """The complete and only value surface reachable by Bernini forward."""

    input_patches: Any
    rotary: Any
    layout: route.TokenRoleLayout


@dataclass(frozen=True)
class PreparedArm:
    model_inputs: PreparedModelInputs
    clean_target: Any
    shared_epsilon: Any
    transport: Any
    source_mode_sha256: str
    epsilon_sha256: str
    noisy_target_sha256: str


def prepare_arm_tensors(
    source_mode: Any,
    target_mode: Any,
    epsilon: Any,
    program_patches: Any,
    transport: Any,
    rope: Any,
    device: Any,
) -> PreparedArm:
    import torch

    if not torch.equal(epsilon, epsilon.detach()) or tuple(epsilon.shape) != tuple(target_mode.shape):
        raise RAMPTrainingError("shared epsilon geometry differs from target mode")
    source_patches = pack_latent_patches(source_mode)
    target_patches = pack_latent_patches(target_mode)
    epsilon_patches = pack_latent_patches(epsilon)
    source_tokens = int(source_patches.shape[0])
    target_tokens = int(target_patches.shape[0])
    if source_tokens != target_tokens:
        raise RAMPTrainingError("source and target token counts differ")
    layout = route.TokenRoleLayout.contiguous(
        source_tokens=source_tokens, target_tokens=target_tokens
    )
    inputs = torch.cat([source_patches, program_patches, epsilon_patches], dim=0)
    if int(inputs.shape[0]) != layout.total_tokens:
        raise RAMPTrainingError("source+21-motion+target pack length differs")

    source_cuda = source_mode.unsqueeze(0).to(device)
    # Target RoPE is a geometry-only function, but use the actual noisy state
    # epsilon rather than the clean supervision tensor even for that call.  No
    # clean target value is therefore reachable from the model-input builder.
    noisy_target_cuda = epsilon.unsqueeze(0).to(device)
    program_latent = (
        program_patches.permute(1, 0, 2, 3, 4)
        .reshape(1, 16, LATENT_PHASES, 2, 2)
        .to(device)
    )
    source_rope = rope(source_cuda, source_id=1)
    motion_rope = rope(program_latent, source_id=2)
    target_rope = rope(noisy_target_cuda, source_id=0)
    rotary = torch.cat([source_rope, motion_rope, target_rope], dim=2)
    rotary = rotary.squeeze(0).permute(1, 0, 2).contiguous()
    if int(rotary.shape[0]) != layout.total_tokens:
        raise RAMPTrainingError("source+motion+target RoPE length differs")
    clean_target = packed_output_field(target_patches).to(device)
    shared = packed_output_field(epsilon_patches).to(device)
    input_cuda = inputs.to(device)
    transport_cuda = transport.to(device).detach().contiguous()
    return PreparedArm(
        model_inputs=PreparedModelInputs(
            input_patches=input_cuda,
            rotary=rotary,
            layout=layout,
        ),
        clean_target=clean_target,
        shared_epsilon=shared,
        transport=transport_cuda,
        source_mode_sha256=tensor_sha256(source_mode),
        epsilon_sha256=tensor_sha256(epsilon),
        noisy_target_sha256=tensor_sha256(epsilon),
    )


def _velocity_prediction(
    renderer: Any,
    transformer: Any,
    model_inputs: PreparedModelInputs,
    text_batch: Mapping[str, Any],
) -> Any:
    text_lens, text_embs = renderer.get_t5_text_embeddings(
        text_batch["input_ids"], text_batch["attention_mask"], text_batch["t5_input_lens"]
    )
    embedded = transformer.patch_embedding(model_inputs.input_patches).flatten(1).unsqueeze(0)
    rotary = model_inputs.rotary.permute(1, 0, 2).unsqueeze(0)
    prediction = renderer.diff_dec.shared_step(
        model_id="transformer_1",
        noisy_latents=embedded,
        timesteps=embedded.new_tensor([TIMESTEP], dtype=__import__("torch").int64),
        cond_embeds=text_embs,
        rotary_embs=rotary,
        batch_vae_seqlen=[model_inputs.layout.total_tokens],
        batch_text_seqlen=text_lens,
    )
    start = model_inputs.layout.source_tokens + model_inputs.layout.motion_tokens
    target = prediction[:, start : start + model_inputs.layout.target_tokens, :]
    expected_shape = (1, model_inputs.layout.target_tokens, route.PATCH_VALUES)
    if tuple(target.shape) != expected_shape or not target.requires_grad:
        raise RAMPTrainingError("Bernini target velocity geometry/graph differs")
    return target


def _objective_compatibility_logits(transport: Any) -> Any:
    """Supply the objective API's unused map slot without a model shortcut.

    The objective's registered interface requires differentiable transport
    logits even though C0 must not optimize a teacher-forced reconstruction of
    its own program input.  A standalone zero leaf satisfies validation; its
    map term is excluded explicitly by :func:`paired_prediction_loss`.
    """

    import torch

    if (
        not isinstance(transport, torch.Tensor)
        or tuple(transport.shape) != (LATENT_PHASES, LATENT_PHASES)
        or transport.dtype != torch.float32
        or transport.requires_grad
        or transport.grad_fn is not None
        or not transport.is_contiguous()
        or not bool(torch.isfinite(transport).all().item())
    ):
        raise RAMPTrainingError("program transport target must be detached FP32 [21,21]")
    if bool((transport < 0.0).any().item()) or not torch.allclose(
        transport.sum(dim=-1),
        torch.ones(LATENT_PHASES, dtype=torch.float32, device=transport.device),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RAMPTrainingError("program transport rows must be probabilities")
    logits = torch.zeros(
        (1, LATENT_PHASES, LATENT_PHASES),
        dtype=torch.float32,
        device=transport.device,
        requires_grad=True,
    )
    if (
        tuple(logits.shape) != (1, LATENT_PHASES, LATENT_PHASES)
        or logits.dtype != torch.float32
        or not logits.requires_grad
        or not logits.is_leaf
        or logits.grad_fn is not None
        or not logits.is_contiguous()
        or not bool(torch.isfinite(logits.detach()).all().item())
    ):
        raise RAMPTrainingError("compatibility logits must be a finite FP32 leaf [1,21,21]")
    return logits


def _exchange_detached_tensor(local: Any, *, group: Any) -> tuple[Any, Any]:
    import torch
    import torch.distributed as dist

    gathered = [torch.empty_like(local) for _ in range(DP_SIZE)]
    dist.all_gather(gathered, local.detach().contiguous(), group=group)
    if any(item.shape != local.shape or not bool(torch.isfinite(item).all().item()) for item in gathered):
        raise RAMPTrainingError("DP2 tensor exchange returned an invalid arm")
    return gathered[0].contiguous(), gathered[1].contiguous()


def _remote_gradient_proxy(value: Any) -> Any:
    """Adapt a remote value to the objective API without pretending it has a model graph."""

    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
    ):
        raise RAMPTrainingError("remote prediction exchange must be a detached tensor")
    leaf = value.requires_grad_(True)
    if not leaf.is_leaf or leaf.grad_fn is not None:
        raise RAMPTrainingError("remote partial prediction must remain an autograd leaf")
    return leaf


def paired_prediction_loss(result: Any) -> Any:
    """Return only losses whose gradients must pass through Bernini outputs."""

    import torch

    loss = (
        result.flow_matching_loss
        + ROUTE_WEIGHT * result.route_loss
        + DONOR_IDENTITY_WEIGHT * result.donor_identity_invariance_loss
        + ORDER_WEIGHT * result.order_invariance_loss
    )
    if (
        not isinstance(loss, torch.Tensor)
        or not loss.requires_grad
        or not bool(torch.isfinite(loss.detach()).item())
    ):
        raise RAMPTrainingError("paired prediction loss is non-finite or detached")
    return loss


def local_backward_loss(pair_loss: Any) -> Any:
    """Double one local partial so DP SUM/2 recovers the full pair derivative."""

    return 2.0 * pair_loss


def _synchronize_gradients(named: Sequence[tuple[str, Any]], parallel: ParallelContext) -> float:
    import torch
    import torch.distributed as dist

    if not named:
        raise RAMPTrainingError("adapter trainable parameter set is empty")
    ready = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for _, parameter in named
    )
    if not _world_all_true(ready, group=parallel.world_group):
        raise RAMPTrainingError("at least one adapter gradient is missing/non-finite")
    squared = torch.zeros((), dtype=torch.float32, device="cuda")
    for _, parameter in named:
        assert parameter.grad is not None
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=parallel.sp_group)
        parameter.grad.div_(float(SP_SIZE))
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=parallel.dp_group)
        parameter.grad.div_(float(DP_SIZE))
        squared.add_(parameter.grad.float().square().sum())
    norm = float(squared.sqrt().item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise RAMPTrainingError("synchronized adapter gradient norm is zero/non-finite")
    return norm


def _synchronize_initial_parameters(named: Sequence[tuple[str, Any]], world_group: Any) -> str:
    import torch.distributed as dist

    for _, parameter in named:
        dist.broadcast(parameter.data, src=0, group=world_group)
    digest = legacy.trainable_parameters_digest(named)
    return _digest_consensus(digest, group=world_group, expected_count=WORLD_SIZE, label="initial adapter")


def _parameter_consensus(named: Sequence[tuple[str, Any]], world_group: Any, label: str) -> str:
    digest = legacy.trainable_parameters_digest(named)
    return _digest_consensus(digest, group=world_group, expected_count=WORLD_SIZE, label=label)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_file_replace(temporary: Path, destination: Path) -> None:
    """Persist bytes, atomically replace the name, then persist that name."""

    _fsync_file(temporary)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _durable_file_replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    import torch

    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(value), temporary)
        _durable_file_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_safetensors(path: Path, named: Sequence[tuple[str, Any]]) -> None:
    from safetensors.torch import save_file

    tensors = {name: parameter.detach().cpu().float().contiguous() for name, parameter in named}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_file(tensors, str(temporary))
        _durable_file_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _verify_staged_run_bundle(stage: Path, receipt: Mapping[str, Any]) -> None:
    expected_files = {
        "adapter.safetensors",
        "optimizer.pt",
        "history.json",
        "receipt.json",
    }
    entries = list(stage.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise RAMPTrainingError("staged output contains a non-plain artifact")
    actual_files = {path.name for path in entries}
    if actual_files != expected_files:
        raise RAMPTrainingError(f"staged output artifact set differs: {actual_files}")

    expected_receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    expected_receipt_sha = hashlib.sha256(expected_receipt_bytes).hexdigest()
    receipt_bytes, _ = _read_bound_bytes(
        stage / "receipt.json",
        expected_receipt_sha,
        label="staged run receipt",
    )
    if receipt_bytes != expected_receipt_bytes:
        raise RAMPTrainingError("staged receipt bytes differ from in-memory receipt")
    parsed = _strict_json_load_bytes(receipt_bytes, label="staged run receipt")
    if parsed != dict(receipt):
        raise RAMPTrainingError("staged receipt object differs from in-memory receipt")
    declared = _require_sha256(parsed.get("receipt_digest"), label="run receipt digest")
    unsigned = dict(parsed)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        raise RAMPTrainingError("staged run receipt embedded digest differs")

    artifacts = parsed.get("artifacts")
    expected_artifact_names = expected_files - {"receipt.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifact_names:
        raise RAMPTrainingError("staged receipt artifact manifest differs")
    for name in sorted(expected_artifact_names):
        expected = _require_sha256(artifacts[name], label=f"{name} artifact SHA")
        if file_sha256(stage / name) != expected:
            raise RAMPTrainingError(f"staged {name} differs from receipt")


def _prepare_output_transaction(
    path: str | Path, rank: int, world_group: Any
) -> tuple[Path, Path]:
    """Reserve a hidden same-filesystem stage; publish only by atomic rename."""

    import torch.distributed as dist

    requested = Path(path).expanduser()
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or requested.suffix
        or _SAFE_OUTPUT_NAME.fullmatch(requested.name) is None
    ):
        raise RAMPTrainingError("output must be an absolute safe suffix-free directory")
    parent = requested.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir() or requested != parent / requested.name:
        raise RAMPTrainingError("output parent/path is not canonical")
    staging = parent / f".{requested.name}.staging"
    fresh = not (
        requested.exists()
        or requested.is_symlink()
        or staging.exists()
        or staging.is_symlink()
    )
    if _world_all_true(fresh, group=world_group) is not True:
        raise RAMPTrainingError("refusing to reuse training output or hidden staging path")
    if rank == 0:
        staging.mkdir(mode=0o750)
        _fsync_directory(parent)
    dist.barrier(group=world_group)
    if not staging.is_dir() or staging.is_symlink() or requested.exists():
        raise RAMPTrainingError("rank zero did not create a private output stage")
    return requested, staging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pair-config", required=True)
    parser.add_argument("--expected-pair-config-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=("engineering-canary", "afterok-c0"), required=True
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--ack-upstream-training-use-forbidden", action="store_true")
    parser.add_argument("--num-frames", type=int, choices=(FRAME_COUNT,), default=FRAME_COUNT)
    return parser


def validate_cli(args: argparse.Namespace) -> int:
    if args.ack_upstream_training_use_forbidden is not True:
        raise RAMPTrainingError("--ack-upstream-training-use-forbidden is mandatory")
    if args.num_frames != FRAME_COUNT:
        raise RAMPTrainingError("RAMP C0 is frozen to exact 81 frames")
    steps = CANARY_STEPS if args.mode == "engineering-canary" else C0_STEPS
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise RAMPTrainingError("learning rate must be finite and positive")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.0:
        raise RAMPTrainingError("max grad norm must be finite and positive")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise RAMPTrainingError("seed must lie in [0,2^63)")
    for name in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            raise RAMPTrainingError(f"{name} must be a lowercase full SHA-1")
    for name in (
        "expected_pair_config_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise RAMPTrainingError(f"{name} must be a lowercase SHA-256")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise RAMPTrainingError("checkpoint tree differs from audited Bernini 1.3B")
    return steps


def _arm_receipt(arm: MaterializedArm) -> dict[str, Any]:
    return {
        "arm": arm.arm,
        "row_id": arm.row_id,
        "source_video_sha256": arm.source_video_sha256,
        "donor_video_sha256": arm.donor_video_sha256,
        "source_posterior_sha256": arm.source_blob_sha256,
        "donor_before_posterior_sha256": arm.donor_before_blob_sha256,
        "donor_after_posterior_sha256": arm.donor_after_blob_sha256,
        "target_posterior_sha256": arm.target_blob_sha256,
        "program_kind": arm.program_kind,
        "program_parameter_hex": arm.program_parameter_hex,
        "program_digest": arm.program_digest,
        "bucket_hw": list(arm.bucket_hw),
        "posterior_shape": list(arm.posterior_shape),
        "parquet_path": arm.parquet_path,
        "parquet_sha256": arm.parquet_sha256,
        "receipt_path": arm.receipt_path,
        "receipt_sha256": arm.receipt_sha256,
        "materialized_row_digest": arm.materialized_row_digest,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    steps = validate_cli(args)
    pair = load_pair_config(args.pair_config, args.expected_pair_config_sha256)
    arm_a = _read_one_parquet(pair.arm_a)
    arm_b = _read_one_parquet(pair.arm_b)
    pair_contract = validate_pair_semantics(arm_a, arm_b)
    arms = (arm_a, arm_b)

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise RAMPTrainingError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise RAMPTrainingError("pinned Bernini attention-head count differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, __version__ as transformers_version
    from diffusers import __version__ as diffusers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state

    contract = distributed_contract()
    device = initialise_distributed(contract)
    parallel = validate_parallel_state(contract, init_parallel_state(ulysses_size=SP_SIZE))
    output, output_stage = _prepare_output_transaction(
        args.output, contract.rank, parallel.world_group
    )
    local_arm = arms[contract.arm_index]

    legacy.seed_same_sample(args.seed)
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False)
    renderer.eval()
    renderer.t5_text_encoder.eval()
    renderer.to(device)
    transformer = renderer.diff_dec.transformer
    if transformer is None or renderer.diff_dec.transformer_2 is not None:
        raise RAMPTrainingError("RAMP C0 requires only Bernini transformer_1")
    renderer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if not bool(getattr(transformer, "gradient_checkpointing", False)):
        raise RAMPTrainingError("RAMP C0 requires gradient checkpointing")
    adapter = route.install_ramp_adapter(
        transformer, rank=LORA_RANK, alpha=LORA_ALPHA
    )
    trainable = adapter.trainable_named_parameters()
    if not adapter.base_parameters_frozen():
        raise RAMPTrainingError("Bernini base changed trainability after RAMP install")
    initial_digest = _synchronize_initial_parameters(trainable, parallel.world_group)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    text_batch = _tokenize_generic_instruction(tokenizer, local_arm.instruction, device)
    text_digest = object_sha256(
        {
            name: tensor_sha256(value)
            for name, value in sorted(text_batch.items())
        }
    )
    _digest_consensus(
        text_digest,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="generic text tokens",
    )
    vae_mean, vae_std, _ = legacy._vae_statistics(checkpoint)
    source_mode = _load_posterior_mode(local_arm.source_blob, vae_mean, vae_std)
    target_mode = _load_posterior_mode(local_arm.target_blob, vae_mean, vae_std)
    if tuple(source_mode.shape) != tuple(target_mode.shape):
        raise RAMPTrainingError("source/target normalized mode geometry differs")
    source_digest = tensor_sha256(source_mode)
    _digest_consensus(
        source_digest,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="shared source posterior mode",
    )
    _, transport, program_patches = _program_for_arm(local_arm)
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    history: list[dict[str, Any]] = []
    layout_receipt: Optional[dict[str, Any]] = None

    for step in range(steps):
        noise_seed = int.from_bytes(
            hashlib.sha256(f"{args.seed}\0ramp-c0\0{step}".encode("ascii")).digest()[:8],
            "big",
        ) % (2**31)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        epsilon = torch.randn(
            tuple(target_mode.shape), generator=generator, dtype=torch.float32, device="cpu"
        ).contiguous()
        prepared = prepare_arm_tensors(
            source_mode,
            target_mode,
            epsilon,
            program_patches,
            transport,
            rope,
            device,
        )
        if layout_receipt is None:
            layout_receipt = prepared.model_inputs.layout.as_dict()
        elif layout_receipt != prepared.model_inputs.layout.as_dict():
            raise RAMPTrainingError("token layout changed between optimizer steps")
        _digest_consensus(
            prepared.epsilon_sha256,
            group=parallel.world_group,
            expected_count=WORLD_SIZE,
            label=f"step {step} shared epsilon",
        )
        invocation = route.RouteInvocation(
            prepared.model_inputs.layout,
            sequence_parallel_rank=contract.sp_rank,
            sequence_parallel_size=SP_SIZE,
        )
        optimizer.zero_grad(set_to_none=True)
        # The route context deliberately encloses BOTH forward and backward;
        # non-reentrant checkpoint recomputation therefore sees the same roles.
        with adapter.route(invocation):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                local_prediction = _velocity_prediction(
                    renderer, transformer, prepared.model_inputs, text_batch
                )
            prediction_digest = tensor_sha256(local_prediction)
            _digest_consensus(
                prediction_digest,
                group=parallel.sp_group,
                expected_count=SP_SIZE,
                label=f"step {step} arm prediction",
            )
            exchanged_prediction = _exchange_detached_tensor(
                local_prediction, group=parallel.dp_group
            )
            exchanged_target = _exchange_detached_tensor(
                prepared.clean_target, group=parallel.dp_group
            )
            if contract.arm_index == 0:
                pred_a = local_prediction
                remote_gradient_proxy = _remote_gradient_proxy(exchanged_prediction[1])
                pred_b = remote_gradient_proxy
            else:
                remote_gradient_proxy = _remote_gradient_proxy(exchanged_prediction[0])
                pred_a = remote_gradient_proxy
                pred_b = local_prediction
            clean_a, clean_b = exchanged_target
            compatibility_logits = _objective_compatibility_logits(prepared.transport)
            identity = objective.SameStateInterventionIdentity(
                source_sha256=prepared.source_mode_sha256,
                text_sha256=text_digest,
                epsilon_sha256=prepared.epsilon_sha256,
                noisy_target_sha256=prepared.noisy_target_sha256,
                timestep_token="sigma=1",
                program_a_sha256=arm_a.program_digest,
                program_b_sha256=arm_b.program_digest,
            )
            result = objective.sigma_one_same_state_route_objective(
                pred_a,
                pred_b,
                clean_a,
                clean_b,
                prepared.shared_epsilon,
                identity=identity,
                # Donor identity/order are structurally absent in C0.  These
                # exact aliases make their invariance losses zero; they are not
                # empirical C1 controls and are reported as such in receipts.
                donor_identity_prediction_a=pred_a,
                donor_identity_prediction_b=pred_a,
                order_prediction_a=pred_b,
                order_prediction_b=pred_b,
                transport_logits=compatibility_logits,
                transport_target=prepared.transport,
            )
            pair_loss = paired_prediction_loss(result)
            backward = local_backward_loss(pair_loss)
            if not _world_all_true(
                bool(torch.isfinite(backward.detach()).item()), group=parallel.world_group
            ):
                raise RAMPTrainingError("non-finite paired objective blocked optimizer step")
            backward.backward()
            if remote_gradient_proxy.grad is None:
                raise RAMPTrainingError("remote partial derivative was not evaluated")
            remote_gradient_proxy.grad = None
            if compatibility_logits.grad is not None:
                raise RAMPTrainingError("excluded map shortcut unexpectedly received a gradient")

        preclip_norm = _synchronize_gradients(trainable, parallel)
        clipped = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], args.max_grad_norm
        )
        if not math.isfinite(float(clipped)):
            raise RAMPTrainingError("gradient clipping produced a non-finite norm")
        optimizer.step()
        parameter_digest = _parameter_consensus(
            trainable, parallel.world_group, f"step {step + 1} adapter"
        )
        record = {
            "schema_version": HISTORY_SCHEMA,
            "step": step + 1,
            "noise_seed": noise_seed,
            "epsilon_sha256": prepared.epsilon_sha256,
            "flow_matching_loss": float(result.flow_matching_loss.detach().item()),
            "route_loss": float(result.route_loss.detach().item()),
            "route_explained_fraction": float(
                result.route_explained_fraction.detach().mean().item()
            ),
            "own_target_ranking": bool(result.own_target_ranking.detach().all().item()),
            "teacher_forced_map_shortcut_optimized": False,
            "local_backward_loss": float(backward.detach().item()),
            "preclip_gradient_norm": preclip_norm,
            "parameter_sha256": parameter_digest,
        }
        record_digest = object_sha256(record)
        # Pair terms and final parameters are WORLD-identical; gradients differ
        # by local partial until the explicit DP reduction above.
        consensus_projection = {
            key: value
            for key, value in record.items()
            if key not in {"local_backward_loss"}
        }
        _digest_consensus(
            object_sha256(consensus_projection),
            group=parallel.world_group,
            expected_count=WORLD_SIZE,
            label=f"step {step + 1} metric projection",
        )
        if contract.rank == 0:
            history.append({**record, "record_digest": record_digest})
            print(json.dumps(record, sort_keys=True), flush=True)

    final_digest = _parameter_consensus(trainable, parallel.world_group, "final adapter")
    dist.barrier(group=parallel.world_group)
    if contract.rank == 0:
        adapter_path = output_stage / "adapter.safetensors"
        optimizer_path = output_stage / "optimizer.pt"
        history_path = output_stage / "history.json"
        _atomic_safetensors(adapter_path, trainable)
        _atomic_torch_save(
            optimizer_path,
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "mode": args.mode,
                "optimizer_steps": steps,
                "optimizer": optimizer.state_dict(),
                "adapter_parameter_sha256": final_digest,
            },
        )
        _atomic_json(
            history_path,
            {"schema_version": HISTORY_SCHEMA, "steps": history},
        )
        receipt: dict[str, Any] = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "complete": True,
            "mode": args.mode,
            "engineering_canary": args.mode == "engineering-canary",
            "optimizer_steps": steps,
            "frame_count": FRAME_COUNT,
            "latent_phase_count": LATENT_PHASES,
            "sigma": SIGMA,
            "timestep": TIMESTEP,
            "noisy_target_equation": "x_1=epsilon",
            "pair_config": {
                "path": str(pair.path),
                "sha256": pair.sha256,
                "schema_version": PAIR_CONFIG_SCHEMA,
            },
            "arms": [_arm_receipt(arm_a), _arm_receipt(arm_b)],
            "pair_contract": pair_contract,
            "visual_pack": {
                **(layout_receipt or {}),
                "order": ["source_identity", "21_motion_program", "noisy_target"],
                "source_id_rope": [1, 2, 0],
                "raw_donor_tokens": 0,
                "raw_donor_blobs_hash_audited_then_discarded_before_pack": True,
                "target_supervision_is_not_model_input": True,
            },
            "adapter": dict(adapter.receipt()),
            "trainable_scope": [name for name, _ in trainable],
            "trainable_scope_exact": "role_embedding+program_projector+target_row_q_lora",
            "initial_adapter_sha256": initial_digest,
            "final_adapter_sha256": final_digest,
            "distributed": {
                "world_size": WORLD_SIZE,
                "data_parallel_size": DP_SIZE,
                "ulysses_sequence_parallel_size": SP_SIZE,
                "sp_groups": [list(item) for item in SP_GROUP_RANKS],
                "dp_groups": [list(item) for item in DP_GROUP_RANKS],
                "arm_assignment": {"ranks_0_3": "arm_a", "ranks_4_7": "arm_b"},
                "gradient_sync": [
                    "SP4_all_reduce_sum_then_divide_by_4",
                    "DP2_all_reduce_sum_then_divide_by_2",
                ],
                "pair_component_local_multiplier": 2.0,
                "map_component_local_multiplier": 0.0,
                "remote_prediction": "detached_gradient_proxy_for_local_partial_only",
                "cross_process_autograd_used": False,
                "remote_leaf_gradient_discarded_before_optimizer": True,
                "context_covers_forward_and_backward": True,
                "checkpoint_recomputation_role_context_preserved": True,
            },
            "objective": {
                "name": objective.SCHEMA_VERSION,
                "donor_identity_and_order_losses": "structural_zero_aliases_not_C1_controls",
                "wrong_donor_degradation_margin_trained": False,
                "teacher_forced_program_reconstruction_optimized": False,
                "objective_api_map_slot": "constant_zero_leaf_excluded_from_backward",
                "program_projector_gradient_source": "bernini_velocity_losses_only",
                "source_rich_noise_used": False,
                "iid_gaussian_noise_train_test_matched": True,
            },
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                "base_frozen": True,
                "gradient_checkpointing": True,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
            },
            "runtime": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
            },
            "artifacts": {
                "adapter.safetensors": file_sha256(adapter_path),
                "optimizer.pt": file_sha256(optimizer_path),
                "history.json": file_sha256(history_path),
            },
            "upstream_training_use_forbidden_acknowledged": True,
            "pretext_training_only": True,
            "natural_semantic_action_learned": False,
            "action_editing_claim_authorized": False,
            "video_quality_claim_authorized": False,
            "scientific_claim_authorized": False,
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
        }
        receipt["receipt_digest"] = object_sha256(receipt)
        _atomic_json(output_stage / "receipt.json", receipt)
        _verify_staged_run_bundle(output_stage, receipt)
        _fsync_directory(output_stage)
        if output.exists() or output.is_symlink():
            raise RAMPTrainingError("final output appeared before atomic publication")
        os.replace(output_stage, output)
        _fsync_directory(output.parent)
    dist.barrier(group=parallel.world_group)
    if not output.is_dir() or output.is_symlink() or output_stage.exists():
        raise RAMPTrainingError("atomic output publication did not complete")
    adapter.restore()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
