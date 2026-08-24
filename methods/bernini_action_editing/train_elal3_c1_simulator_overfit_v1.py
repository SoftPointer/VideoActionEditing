#!/usr/bin/env python3
"""Real Bernini ELAL-3 one-row oracle-q optimizer diagnostic.

This is deliberately narrow.  It trains a fresh Bernini-R 1.3B renderer with
rank-256 LoRA on all 240 native q/k/v/out attention affines and the full-w64
ELAL-3 dense modules.  The only admitted row is the simulator C1 push row.
ELAL-3 q is target-derived and teacher-forced.  The diffusion target is the
real Bernini VAE flow-matching velocity; the simulator's two-dimensional
signed motion is evidence only and is never substituted for that velocity.

This program is NOT source+instruction inference, formal C1, exact160, a
real-video experiment, or scientific evidence.  It contains no frozen
teacher, frozen velocity reference, self-distillation, reward, or
ActionPredictor path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import stat
import sys
import time
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

METHOD = "bernini-elal3-c1-simulator-oracle-q-overfit-v1"
RECEIPT_SCHEMA = "bernini-elal3-c1-simulator-overfit-training-receipt-v1"
CHECKPOINT_SCHEMA = "bernini-elal3-c1-simulator-overfit-checkpoint-v1"
LATENT_BUNDLE_SCHEMA = "bernini-elal3-simulator-c1-latent-bundle-v1"
LATENT_RECEIPT_SCHEMA = "bernini-elal3-simulator-c1-latent-bundle-receipt-v1"
EXTERNAL_OPTIMIZER_AUTHORITY_SCHEMA = (
    "bernini-elal3-simulator-optimizer-derivative-authority-v1"
)
EXTERNAL_OPTIMIZER_AUTHORITY_SHA256 = (
    "298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b"
)
EXTERNAL_OPTIMIZER_AUTHORITY_DIGEST = (
    "c1706ee5b3f8a3fa4c037dfa6dbdbc7d0b088d3682128e50e712e311dae35043"
)
MODEL_AUTHORITY_SCHEMA = "bernini-elal3-c1-real-model-authority-v1"
MODEL_AUTHORITY_SHA256 = (
    "4c2f4d28af646ab39bdeb775e1b651d523d83b3fc0b8e5c1dd4bc78fbd4f25ed"
)
MODEL_AUTHORITY_DIGEST = (
    "25255902f4c5ce6de94ce6c3666bcf85eae4bf8e360a217f327c6febd049d21b"
)
PACKET_MANIFEST_SHA256 = (
    "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"
)
LATENT_BUNDLE_SHA256 = (
    "8fbd27abf7b6eea0593b236a0594dcfad38b3bedf46cf42e77391ec5648fdedf"
)
LATENT_BUNDLE_SIZE = 39_138_208
LATENT_BUNDLE_RECEIPT_SHA256 = (
    "a400d11d0d1337daa61d74a25e040aab27b83cc75e62038b81b83f56075e4fcb"
)
LATENT_BUNDLE_RECEIPT_SIZE = 11_850
LATENT_BUNDLE_RECEIPT_DIGEST = (
    "81f0ab734249651b00571e94a616de5a04fb13aa53fd711e45554b5a76251d61"
)
LATENT_SHAPE = (1, 16, 21, 52, 70)
PATCH_GRID = (21, 26, 35)
TOKENS_PER_ROLE = 19_110
PACKED_TOTAL_TOKENS = 38_220
LOCAL_SP_TOKENS = 9_555
ROW_ID = "c1-two-entity-push-to-goal"
TENSOR_ORDER = (
    "source",
    "target",
    "anchor",
    "wrong_agent",
    "wrong_object",
    "role_swap",
    "reverse",
    "phase_shuffle",
)
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
BLOCKS = 30
HIDDEN = 1536
LATENT_CHANNELS = 16
LATENT_PHASES = 21
PATCH_VALUES = 64
LORA_AFFINES = 240
LORA_RANK = 256
ELAL3_FULL_W64_PARAMETERS = 9_979_934
LORA_PARAMETERS = 188_743_680
EXPECTED_TRAINABLE_PARAMETERS = LORA_PARAMETERS + ELAL3_FULL_W64_PARAMETERS
DEFAULT_LR = 1.0e-4
DEFAULT_MAX_GRAD_NORM = 1.0
MEMORY_FRACTION_GATE = 0.5
ACTIVATION_CHECKPOINT_PROFILE = "selective-nonreentrant-stride4-exact8"
ACTIVATION_CHECKPOINT_BLOCKS = tuple(range(0, BLOCKS, 4))
ACTIVATION_UNCHECKPOINTED_BLOCKS = tuple(
    index for index in range(BLOCKS) if index not in ACTIVATION_CHECKPOINT_BLOCKS
)
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
MODEL_INDEX_KEYS = 825
MODEL_SHARDS = (
    "diffusion_pytorch_model-00001-of-00002.safetensors",
    "diffusion_pytorch_model-00002-of-00002.safetensors",
)
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ELAL3C1TrainingError(RuntimeError):
    """Raised before accepting an ambiguous update or artifact."""


def fail(message: str) -> NoReturn:
    raise ELAL3C1TrainingError(message)


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
        raise ELAL3C1TrainingError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    named = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _require_sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    require_canonical_newline: bool = True,
) -> Mapping[str, Any]:
    expected = _require_sha(expected_sha256, label=f"{label} expected SHA")
    requested = path.expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be one absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        info = resolved.lstat()
    except OSError as error:
        raise ELAL3C1TrainingError(f"{label} is unavailable") from error
    if resolved != requested or not stat.S_ISREG(info.st_mode):
        fail(f"{label} canonical file type differs")
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        fail(f"{label} SHA-256 differs")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: fail(f"non-finite JSON token: {token}"),
        )
    except ELAL3C1TrainingError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3C1TrainingError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        fail(f"{label} must be one JSON object")
    if require_canonical_newline and payload != canonical_json_bytes(value) + b"\n":
        # The separately-issued optimizer authority is human-readable but its
        # exact bytes are pinned.  Callers opt out only for that one artifact.
        fail(f"{label} is not canonical JSON+newline")
    return dict(value)


def validate_external_optimizer_authority(
    path: Path, *, expected_sha256: str
) -> Mapping[str, Any]:
    if expected_sha256 != EXTERNAL_OPTIMIZER_AUTHORITY_SHA256:
        fail("external optimizer authority literal SHA differs")
    value = read_bound_json(
        path,
        expected_sha256=expected_sha256,
        label="external optimizer authority",
        require_canonical_newline=False,
    )
    unsigned = dict(value)
    digest = unsigned.pop("authority_digest", None)
    exact_keys = {
        "allowed_nodes",
        "allowed_operations",
        "authorization_basis",
        "authorized_row_id",
        "disallowed_claims",
        "fresh_optimizer_run_required",
        "max_optimizer_updates_per_arm",
        "oracle_q_teacher_forced_required",
        "packet_manifest_sha256",
        "packet_status_preserved",
        "schema_version",
        "status",
        "supersedes_packet_training_use_forbidden_for_exact_scope_only",
        "training_objective_restrictions",
    }
    restrictions = value.get("training_objective_restrictions")
    if (
        set(unsigned) != exact_keys
        or value.get("schema_version") != EXTERNAL_OPTIMIZER_AUTHORITY_SCHEMA
        or digest != EXTERNAL_OPTIMIZER_AUTHORITY_DIGEST
        or object_sha256(unsigned) != digest
        or value.get("authorized_row_id") != ROW_ID
        or value.get("max_optimizer_updates_per_arm") != 20
        or value.get("oracle_q_teacher_forced_required") is not True
        or value.get("packet_manifest_sha256") != PACKET_MANIFEST_SHA256
        or value.get("fresh_optimizer_run_required") is not True
        or value.get("supersedes_packet_training_use_forbidden_for_exact_scope_only")
        is not True
        or not isinstance(restrictions, Mapping)
        or dict(restrictions)
        != {
            "frozen_base_velocity_reference_forbidden": True,
            "frozen_teacher_self_distillation_forbidden": True,
            "hand_tuned_reward_scalar_forbidden": True,
            "target_grounded_event_and_context_flow_only": True,
        }
    ):
        fail("external optimizer authority closed scope differs")
    return value


def validate_runtime_authority_placement_v1(value: Mapping[str, Any]) -> Mapping[str, str]:
    expected_nodes = (
        {"holder_job_id": "141620", "node": "auh7-1b-gpu-226"},
        {"holder_job_id": "141618", "node": "auh7-1b-gpu-249"},
        {"holder_job_id": "141619", "node": "auh7-1b-gpu-257"},
    )
    expected_operations = (
        "frozen_bernini_vae_encode",
        "real_bernini_no_update_integration_probe",
        "oracle_q_exact_one_row_optimizer_overfit",
        "strict_checkpoint_reload_and_oracle_q_decode",
        "source_target_anchor_intervention_html_review",
    )
    claims = value.get("disallowed_claims")
    basis = value.get("authorization_basis")
    if (
        tuple(value.get("allowed_nodes", ())) != expected_nodes
        or tuple(value.get("allowed_operations", ())) != expected_operations
        or value.get("status") != "AUTHORIZED_SIMULATOR_ORACLE_Q_DIAGNOSTIC_ONLY"
        or value.get("packet_status_preserved") != "ELAL3_SIM_DIAGNOSTIC"
        or not isinstance(claims, Mapping)
        or set(claims) != {
            "exact160", "formal_c1", "production_model", "real_video_generalization",
            "scientific_promotion", "source_instruction_inference"
        }
        or any(item is not True for item in claims.values())
        or not isinstance(basis, Mapping)
        or dict(basis) != {
            "date": "2026-08-17",
            "requester": "workspace_user",
            "requester_explicitly_directed_training_on_nodes_226_249_257": True,
            "requester_previously_accepted_elal3_design": True,
        }
    ):
        fail("external optimizer authority placement/operation/claim scope differs")
    node = os.uname().nodename.split(".", 1)[0]
    job = os.environ.get("SLURM_JOB_ID")
    if {"holder_job_id": job, "node": node} not in expected_nodes:
        fail(f"runtime holder placement is not separately authorized: job={job}, node={node}")
    return {"holder_job_id": str(job), "node": node}


def _plain_relative(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or pure.as_posix() != relative or ".." in pure.parts:
        fail(f"{label} relative path differs")
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
        info = candidate.lstat()
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ELAL3C1TrainingError(f"{label} escapes or is unavailable") from error
    if candidate.is_symlink() or not stat.S_ISREG(info.st_mode):
        fail(f"{label} must be a non-symlink regular file")
    return resolved


def validate_model_authority(
    path: Path,
    *,
    expected_sha256: str,
    bernini_root: Path,
    checkpoint_root: Path,
) -> Mapping[str, Any]:
    if expected_sha256 != MODEL_AUTHORITY_SHA256:
        fail("real model authority literal SHA differs")
    value = read_bound_json(
        path,
        expected_sha256=expected_sha256,
        label="real model authority",
        require_canonical_newline=False,
    )
    exact = {
        "authority_digest",
        "bernini_root",
        "checkpoint_root",
        "constraints",
        "file_count",
        "files",
        "model_family",
        "python_env_root",
        "row_id",
        "runtime_versions",
        "schema_version",
    }
    unsigned = dict(value)
    digest = unsigned.pop("authority_digest", None)
    constraints = value.get("constraints")
    rows = value.get("files")
    if (
        set(value) != exact
        or value.get("schema_version") != MODEL_AUTHORITY_SCHEMA
        or digest != MODEL_AUTHORITY_DIGEST
        or object_sha256(unsigned) != digest
        or value.get("row_id") != ROW_ID
        or value.get("file_count") != 9
        or not isinstance(rows, list)
        or len(rows) != 9
        or str(bernini_root) != value.get("bernini_root")
        or str(checkpoint_root) != value.get("checkpoint_root")
        or not isinstance(constraints, Mapping)
        or dict(constraints)
        != {
            "allowed_operation": "elal3_c1_simulator_oracle_q_optimizer_diagnostic",
            "exact160_authorized": False,
            "formal_c1_authorized": False,
            "max_optimizer_updates_per_arm": 20,
            "real_video_claim_authorized": False,
            "scientific_claim_authorized": False,
            "source_instruction_inference_claim_authorized": False,
        }
    ):
        fail("real model authority envelope differs")
    python_root = Path(str(value.get("python_env_root"))).resolve(strict=True)
    roots = {"bernini": bernini_root, "checkpoint": checkpoint_root, "python_env": python_root}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "mode", "relative_path", "root", "sha256", "size"
        }:
            fail("real model authority file row differs")
        root_name = row.get("root")
        relative = row.get("relative_path")
        if root_name not in roots or type(relative) is not str:
            fail("real model authority file root/path differs")
        key = (root_name, relative)
        if key in seen:
            fail("real model authority contains duplicate file rows")
        seen.add(key)
        actual = _plain_relative(roots[root_name], relative, label=f"model pin {key}")
        info = actual.stat()
        if (
            type(row.get("mode")) is not int
            or stat.S_IMODE(info.st_mode) != row["mode"]
            or info.st_size != row.get("size")
            or file_sha256(actual) != row.get("sha256")
        ):
            fail(f"real model pin differs: {key}")
    required = {
        ("bernini", "bernini/pipeline.py"),
        ("checkpoint", "transformer/config.json"),
        ("checkpoint", "transformer/diffusion_pytorch_model.safetensors.index.json"),
        *(("checkpoint", f"transformer/{name}") for name in MODEL_SHARDS),
        ("checkpoint", "vae/config.json"),
        ("checkpoint", "vae/diffusion_pytorch_model.safetensors"),
        ("python_env", "diffusers/__init__.py"),
        ("python_env", "diffusers/models/autoencoders/autoencoder_kl_wan.py"),
    }
    if seen != required:
        fail("real model authority exact9 member closure differs")
    index_path = checkpoint_root / "transformer/diffusion_pytorch_model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3C1TrainingError("transformer model index is unreadable") from error
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if (
        not isinstance(weight_map, Mapping)
        or len(weight_map) != MODEL_INDEX_KEYS
        or set(weight_map.values()) != set(MODEL_SHARDS)
    ):
        fail("transformer index exact825/exact2 shard closure differs")
    return value


def require_model_authority_replay_identity_v1(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *, stage: str
) -> Mapping[str, Any]:
    """Require a byte-semantic replay of the exact9 model authority object."""

    if stage not in {"post_deserialize", "final_pre_publish"}:
        fail("real-model authority replay stage differs")
    reference_bytes = canonical_json_bytes(reference)
    candidate_bytes = canonical_json_bytes(candidate)
    if candidate_bytes != reference_bytes:
        fail(f"real-model authority {stage} replay differs from pre-load closure")
    return {
        "stage": stage,
        "authority_sha256": MODEL_AUTHORITY_SHA256,
        "authority_digest": MODEL_AUTHORITY_DIGEST,
        "replayed_object_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "exact9_rehashed_by_rank_zero": True,
        "world8_barrier_before_replay": True,
        "world8_broadcast_identity_verified": True,
    }


def replay_model_authority_world8_v1(
    *,
    dist: Any,
    group: Any,
    rank: int,
    reference: Mapping[str, Any],
    authority_path: Path,
    expected_sha256: str,
    bernini_root: Path,
    checkpoint_root: Path,
    stage: str,
) -> Mapping[str, Any]:
    """Bracket pathname deserialization/publication without 8-way NFS hashing."""

    dist.barrier(group=group)
    box: list[Any] = [None]
    if rank == 0:
        try:
            candidate = validate_model_authority(
                authority_path,
                expected_sha256=expected_sha256,
                bernini_root=bernini_root,
                checkpoint_root=checkpoint_root,
            )
            box[0] = {
                "ok": True,
                "value": candidate,
                "receipt": require_model_authority_replay_identity_v1(
                    reference, candidate, stage=stage
                ),
            }
        except Exception as error:
            box[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(box, src=0, group=group)
    result = box[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        fail(f"rank-zero real-model authority {stage} replay failed: {result!r}")
    receipt = require_model_authority_replay_identity_v1(
        reference, result["value"], stage=stage
    )
    if receipt != result.get("receipt"):
        fail(f"real-model authority {stage} broadcast receipt differs")
    dist.barrier(group=group)
    return receipt


def validate_stable_binding(
    value: Any, *, expected_path: Optional[Path], label: str
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path", "sha256", "size", "mode", "device", "inode", "nlink"
    }:
        fail(f"{label} stable binding schema differs")
    raw_path = value.get("path")
    if type(raw_path) is not str:
        fail(f"{label} stable binding path differs")
    path = Path(raw_path).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} stable binding needs an absolute non-symlink path")
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as error:
        raise ELAL3C1TrainingError(f"{label} stable file is unavailable") from error
    if (
        resolved != path
        or not stat.S_ISREG(info.st_mode)
        or (expected_path is not None and resolved != expected_path.resolve(strict=True))
        or type(value.get("size")) is not int
        or value["size"] != info.st_size
        or type(value.get("mode")) is not int
        or value["mode"] != stat.S_IMODE(info.st_mode)
        # VAST reports node-local st_dev (and may report node-local inode
        # aliases).  Those two receipt fields are provenance telemetry, not
        # cross-node admission literals.  The consumer authenticates the
        # current same-FD pre/post identity plus named replay and content SHA.
        or type(value.get("device")) is not int
        or type(value.get("inode")) is not int
        or type(value.get("nlink")) is not int
        or value["nlink"] != info.st_nlink
        or file_sha256(resolved) != _require_sha(value.get("sha256"), label=f"{label} SHA")
    ):
        fail(f"{label} stable identity/hash differs")
    return resolved


def tensor_sha256_v1(value: Any) -> str:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or not value.is_contiguous()
    ):
        fail("latent tensor hash requires contiguous CPU strided tensor")
    tensor = value.detach()
    header = canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": [int(item) for item in tensor.shape]}
    )
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


@dataclass(frozen=True)
class LatentBundleV1:
    source: Any
    target: Any
    patch_grid: tuple[int, int, int]
    bucket_hw: tuple[int, int]
    receipt: Mapping[str, Any]
    receipt_sha256: str
    bundle_sha256: str


def load_latent_bundle_v1(
    *,
    bundle_path: Path,
    expected_bundle_sha256: str,
    receipt_path: Path,
    expected_receipt_sha256: str,
    packet_root: Path,
    external_optimizer_authority_path: Path,
    model_authority_path: Path,
    checkpoint_root: Path,
) -> LatentBundleV1:
    import torch
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ELAL3C1TrainingError("safetensors runtime is required") from error

    bundle_sha = _require_sha(expected_bundle_sha256, label="latent bundle expected SHA")
    receipt_sha = _require_sha(expected_receipt_sha256, label="latent receipt expected SHA")
    if bundle_sha != LATENT_BUNDLE_SHA256:
        fail("latent bundle CLI SHA differs from registered v2 literal")
    if receipt_sha != LATENT_BUNDLE_RECEIPT_SHA256:
        fail("latent receipt CLI SHA differs from registered v2 literal")
    if file_sha256(external_optimizer_authority_path) != EXTERNAL_OPTIMIZER_AUTHORITY_SHA256:
        fail("latent consumer optimizer-authority copy differs")
    if file_sha256(model_authority_path) != MODEL_AUTHORITY_SHA256:
        fail("latent consumer real-model-authority copy differs")
    if receipt_path.stat().st_size != LATENT_BUNDLE_RECEIPT_SIZE:
        fail("latent receipt registered byte size differs")
    receipt = read_bound_json(
        receipt_path,
        expected_sha256=receipt_sha,
        label="latent bundle receipt",
        require_canonical_newline=True,
    )
    expected_top = {
        "schema_version",
        "row_id",
        "bundle",
        "bundle_format",
        "tensor_order",
        "tensor_rows",
        "bucket_hw",
        "latent_shape",
        "packet_manifest",
        "media_bindings",
        "checkpoint",
        "bernini_commit",
        "veomni_commit",
        "runtime_dependencies",
        "vae_encode_count",
        "each_media_independently_full_video_vae_encoded",
        "simulator_optimizer_diagnostic_authorized",
        "oracle_q_required_for_training",
        "source_instruction_inference",
        "formal_c1_authorized",
        "exact160_authorized",
        "scientific_claim_authorized",
        "real_video_data",
        "derivative_optimizer_authority",
        "real_model_authority",
        "receipt_digest",
    }
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest", None)
    if (
        set(receipt) != expected_top
        or receipt.get("schema_version") != LATENT_RECEIPT_SCHEMA
        or receipt.get("row_id") != ROW_ID
        or receipt.get("bundle_format") != "safetensors-exact8-fp32-v1"
        or tuple(receipt.get("tensor_order", ())) != TENSOR_ORDER
        or receipt.get("bernini_commit") != BERNINI_COMMIT
        or receipt.get("veomni_commit") != VEOMNI_COMMIT
        or receipt.get("vae_encode_count") != len(TENSOR_ORDER)
        or receipt.get("each_media_independently_full_video_vae_encoded") is not True
        or receipt.get("simulator_optimizer_diagnostic_authorized") is not True
        or receipt.get("oracle_q_required_for_training") is not True
        or receipt.get("source_instruction_inference") is not False
        or any(receipt.get(key) is not False for key in (
            "formal_c1_authorized", "exact160_authorized", "scientific_claim_authorized", "real_video_data"
        ))
        or digest != LATENT_BUNDLE_RECEIPT_DIGEST
        or object_sha256(unsigned) != digest
    ):
        fail("latent bundle receipt closed authority/envelope differs")
    bound_bundle = validate_stable_binding(
        receipt.get("bundle"), expected_path=bundle_path, label="latent bundle"
    )
    if receipt["bundle"]["sha256"] != bundle_sha:
        fail("latent bundle CLI/receipt SHA differs")
    if receipt["bundle"]["size"] != LATENT_BUNDLE_SIZE:
        fail("latent bundle registered byte size differs")
    packet_manifest = packet_root.resolve(strict=True) / "manifest.json"
    validate_stable_binding(
        receipt.get("packet_manifest"),
        expected_path=packet_manifest,
        label="latent packet manifest",
    )
    if receipt["packet_manifest"]["sha256"] != PACKET_MANIFEST_SHA256:
        fail("latent receipt packet manifest SHA differs")
    derivative = receipt.get("derivative_optimizer_authority")
    if not isinstance(derivative, Mapping) or set(derivative) != {
        "file", "authority_digest", "schema_version"
    }:
        fail("latent derivative optimizer authority binding differs")
    validate_stable_binding(
        derivative.get("file"),
        expected_path=None,
        label="latent derivative optimizer authority",
    )
    if (
        derivative["file"]["sha256"] != EXTERNAL_OPTIMIZER_AUTHORITY_SHA256
        or derivative.get("authority_digest") != EXTERNAL_OPTIMIZER_AUTHORITY_DIGEST
        or derivative.get("schema_version") != EXTERNAL_OPTIMIZER_AUTHORITY_SCHEMA
    ):
        fail("latent derivative optimizer authority identity differs")
    real_model = receipt.get("real_model_authority")
    if not isinstance(real_model, Mapping) or set(real_model) != {
        "file", "authority_digest", "schema_version", "verified_file_bindings",
        "verified_before_and_after_encoding"
    }:
        fail("latent real-model authority binding differs")
    validate_stable_binding(
        real_model.get("file"), expected_path=None, label="latent real-model authority"
    )
    if (
        real_model["file"]["sha256"] != MODEL_AUTHORITY_SHA256
        or real_model.get("authority_digest") != MODEL_AUTHORITY_DIGEST
        or real_model.get("schema_version") != MODEL_AUTHORITY_SCHEMA
        or real_model.get("verified_before_and_after_encoding") is not True
        or not isinstance(real_model.get("verified_file_bindings"), list)
        or len(real_model["verified_file_bindings"]) != 9
    ):
        fail("latent real-model authority closure differs")
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or dict(checkpoint) != {
        "path": str(checkpoint_root.resolve(strict=True)),
        "tree_sha256": CHECKPOINT_TREE_SHA256,
        "transformer_layers": BLOCKS,
        "transformer_hidden_width": HIDDEN,
    }:
        fail("latent checkpoint binding differs")
    bucket = receipt.get("bucket_hw")
    shape = receipt.get("latent_shape")
    if (
        not isinstance(bucket, list)
        or len(bucket) != 2
        or any(type(item) is not int or item <= 0 or item % 16 for item in bucket)
        or shape != list(LATENT_SHAPE)
        or shape != [1, LATENT_CHANNELS, LATENT_PHASES, bucket[0] // 8, bucket[1] // 8]
        or shape[3] % 2
        or shape[4] % 2
    ):
        fail("latent bucket/shape/patch geometry differs")
    rows = receipt.get("tensor_rows")
    media = receipt.get("media_bindings")
    if (
        not isinstance(rows, list)
        or len(rows) != len(TENSOR_ORDER)
        or not isinstance(media, Mapping)
        or set(media) != set(TENSOR_ORDER)
    ):
        fail("latent exact8 tensor/media rows differ")
    for role, row in zip(TENSOR_ORDER, rows):
        if not isinstance(row, Mapping) or set(row) != {
            "role", "shape", "dtype", "sha256", "source_media_sha256"
        }:
            fail("latent tensor row schema differs")
        if (
            row.get("role") != role
            or row.get("shape") != shape
            or row.get("dtype") != "torch.float32"
        ):
            fail(f"latent tensor row differs: {role}")
        expected_media = packet_root.resolve(strict=True) / "media" / ROW_ID / f"{role}.mp4"
        validate_stable_binding(
            media[role], expected_path=expected_media, label=f"latent media {role}"
        )
        if row.get("source_media_sha256") != media[role]["sha256"]:
            fail(f"latent media/tensor provenance differs: {role}")
    dependencies = receipt.get("runtime_dependencies")
    if not isinstance(dependencies, Mapping) or not dependencies:
        fail("latent runtime dependency bindings are absent")
    for name, binding in dependencies.items():
        if type(name) is not str or not name:
            fail("latent runtime dependency name differs")
        validate_stable_binding(binding, expected_path=None, label=f"latent dependency {name}")

    with safe_open(str(bound_bundle), framework="pt", device="cpu") as handle:
        keys = tuple(sorted(handle.keys()))
        if keys != tuple(sorted(TENSOR_ORDER)):
            fail("latent safetensors exact8 key closure differs")
        metadata = handle.metadata()
        wanted_metadata = {
            "schema_version": LATENT_BUNDLE_SCHEMA,
            "row_id": ROW_ID,
            "packet_manifest_sha256": PACKET_MANIFEST_SHA256,
            "bucket_hw": f"{bucket[0]},{bucket[1]}",
        }
        if metadata != wanted_metadata:
            fail("latent safetensors metadata differs")
        retained: dict[str, Any] = {}
        for role, row in zip(TENSOR_ORDER, rows):
            tensor = handle.get_tensor(role)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.device.type != "cpu"
                or list(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or not bool(torch.isfinite(tensor).all().item())
                or tensor_sha256_v1(tensor) != row["sha256"]
            ):
                fail(f"latent safetensors tensor differs: {role}")
            if role in ("source", "target"):
                retained[role] = tensor.clone(memory_format=torch.contiguous_format)
    return LatentBundleV1(
        source=retained["source"],
        target=retained["target"],
        patch_grid=PATCH_GRID,
        bucket_hw=(bucket[0], bucket[1]),
        receipt=receipt,
        receipt_sha256=receipt_sha,
        bundle_sha256=bundle_sha,
    )


def validate_derivative_authority_v1(
    value: Any, *, label_receipt: Mapping[str, Any], max_steps: int
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "status", "row_id", "source_packet",
        "external_authority_binding", "label_binding", "scope", "authority",
        "authority_digest"
    }:
        fail("derived oracle-q authority envelope differs")
    unsigned = dict(value)
    digest = unsigned.pop("authority_digest", None)
    authority = value.get("authority")
    scope = value.get("scope")
    external = value.get("external_authority_binding")
    if (
        type(digest) is not str
        or object_sha256(unsigned) != digest
        or value.get("row_id") != ROW_ID
        or not isinstance(authority, Mapping)
        or set(authority) != {
            "status", "simulator_optimizer_diagnostic_authorized", "training_authorized",
            "external_optimizer_authority_verified", "training_authority_source",
            "training_authority_scope", "formal_training_authorized",
            "formal_c0_c1_c2_go_authorized", "exact160_eligible",
            "exact160_claim_authorized", "scientific_claim_authorized",
            "real_video_data", "source_instruction_inference_authorized",
            "model_output_claim_authorized", "oracle_q_teacher_forced",
            "action_encoder_qualified", "action_predictor_present",
            "upstream_training_use_forbidden_acknowledged", "upstream_packet_mutated"
        }
        or authority.get("simulator_optimizer_diagnostic_authorized") is not True
        or authority.get("training_authorized") is not True
        or authority.get("external_optimizer_authority_verified") is not True
        or authority.get("training_authority_source")
        != "separately-issued-pinned-local-authority"
        or authority.get("oracle_q_teacher_forced") is not True
        or authority.get("upstream_training_use_forbidden_acknowledged") is not True
        or any(authority.get(key) is not False for key in (
            "formal_training_authorized", "formal_c0_c1_c2_go_authorized",
            "exact160_eligible", "exact160_claim_authorized", "scientific_claim_authorized",
            "real_video_data", "source_instruction_inference_authorized",
            "model_output_claim_authorized", "action_encoder_qualified",
            "action_predictor_present", "upstream_packet_mutated"
        ))
        or not isinstance(scope, Mapping)
        or scope.get("row_count") != 1
        or scope.get("row_id") != ROW_ID
        or scope.get("allowed_optimizer_updates_min") != 1
        or scope.get("allowed_optimizer_updates_max") != 20
        or max_steps not in (1, 10, 20)
        or scope.get("allowed_representation_variant") != "full"
        or scope.get("allowed_attention_width") != 64
        or scope.get("decoded_review_required") is not True
        or not isinstance(external, Mapping)
        or set(external) != {
            "relative_path", "file_sha256", "schema_version", "object_digest",
            "status", "authorized_row_id", "max_optimizer_updates_per_arm",
            "training_objective_restrictions"
        }
        or external.get("file_sha256") != EXTERNAL_OPTIMIZER_AUTHORITY_SHA256
        or external.get("object_digest") != EXTERNAL_OPTIMIZER_AUTHORITY_DIGEST
        or external.get("schema_version") != EXTERNAL_OPTIMIZER_AUTHORITY_SCHEMA
        or external.get("authorized_row_id") != ROW_ID
        or external.get("max_optimizer_updates_per_arm") != 20
        or value.get("label_binding", {}).get("label_digest")
        != label_receipt.get("label_digest")
    ):
        fail("derived oracle-q authority closed scope differs")
    return dict(value)


def pack_vae_partition_mask_v1(mask: Any, *, target_velocity: Any) -> Any:
    """Pack target-derived VAE-grid masks in native Wan patch order."""

    import torch

    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 5
        or tuple(int(item) for item in mask.shape[:3]) != (1, 1, LATENT_PHASES)
        or not isinstance(target_velocity, torch.Tensor)
        or target_velocity.ndim != 3
        or tuple(int(item) for item in target_velocity.shape[:1]) != (1,)
        or int(target_velocity.shape[2]) != PATCH_VALUES
        or int(mask.shape[3]) % 2
        or int(mask.shape[4]) % 2
        or mask.device != target_velocity.device
        or mask.requires_grad
    ):
        fail("VAE partition mask/target velocity geometry differs")
    expanded = mask.expand(1, LATENT_CHANNELS, -1, -1, -1)[0].contiguous()
    channels, phases, height, width = map(int, expanded.shape)
    packed = (
        expanded.reshape(channels, phases, height // 2, 2, width // 2, 2)
        .permute(1, 2, 4, 0, 3, 5)
        .reshape(1, phases * (height // 2) * (width // 2), PATCH_VALUES)
        .contiguous()
    )
    if packed.shape != target_velocity.shape or not bool(packed.any().item()):
        fail("packed VAE partition mask is empty or misaligned")
    return packed


def partitioned_flow_matching_loss_v1(
    prediction: Any,
    target_velocity: Any,
    event_mask: Any,
    context_mask: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Fixed equal partition mean; no hand-tuned scalar or annotation flow target."""

    import torch

    tensors = (prediction, target_velocity, event_mask, context_mask)
    if (
        any(not isinstance(value, torch.Tensor) for value in tensors)
        or prediction.shape != target_velocity.shape
        or event_mask.shape != prediction.shape
        or context_mask.shape != prediction.shape
        or event_mask.dtype != torch.bool
        or context_mask.dtype != torch.bool
        or event_mask.device != prediction.device
        or context_mask.device != prediction.device
        or bool((event_mask & context_mask).any().item())
        or not bool((event_mask | context_mask).all().item())
        or not bool(event_mask.any().item())
        or not bool(context_mask.any().item())
    ):
        fail("event/context partition is not a disjoint exhaustive nonempty split")
    squared = (prediction.float() - target_velocity.float()).square()
    event = squared[event_mask].mean()
    context = squared[context_mask].mean()
    total = (event + context) * 0.5
    if not bool(torch.isfinite(torch.stack((event, context, total))).all().item()):
        fail("partitioned flow-matching loss is non-finite")
    receipt = {
        "objective": "bernini_fm_target_velocity_target_event_context_equal_partition_mean",
        "event_loss": float(event.detach().item()),
        "context_loss": float(context.detach().item()),
        "total_loss": float(total.detach().item()),
        "event_elements": int(event_mask.sum().item()),
        "context_elements": int(context_mask.sum().item()),
        "fixed_partition_coefficients": [0.5, 0.5],
        "tunable_loss_weights": False,
        "simulator_signed_motion_used_as_diffusion_velocity": False,
    }
    return total, receipt


def exact_trainable_named_parameters_v1(model: Any) -> tuple[tuple[str, Any], ...]:
    named = tuple((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not named or len({name for name, _ in named}) != len(named):
        fail("trainable parameter inventory is empty or duplicated")
    bad = [
        name for name, _ in named
        if ".lora_A." not in name
        and ".lora_B." not in name
        and ".elal3_c0_v1." not in name
    ]
    lora_a = [name for name, _ in named if ".lora_A." in name]
    lora_b = [name for name, _ in named if ".lora_B." in name]
    count = sum(int(parameter.numel()) for _, parameter in named)
    if (
        bad
        or len(lora_a) != LORA_AFFINES
        or len(lora_b) != LORA_AFFINES
        or count != EXPECTED_TRAINABLE_PARAMETERS
    ):
        fail(
            "optimizer must contain exact all240-r256 LoRA plus ELAL3 full-w64 only: "
            f"bad={bad[:4]}, A={len(lora_a)}, B={len(lora_b)}, count={count}"
        )
    return named


def trainable_inventory_v1(named: Sequence[tuple[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        {
            "name": name,
            "shape": [int(item) for item in parameter.shape],
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in named
    ]


def trainable_digest_v1(named: Sequence[tuple[str, Any]]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in named:
        tensor = parameter.detach().contiguous().cpu()
        header = canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(header)
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def synchronize_initial_parameters_v1(
    named: Sequence[tuple[str, Any]],
    group: Any,
    *,
    dist_module: Any = None,
    expected_count: int = WORLD_SIZE,
) -> str:
    """Broadcast exact named trainables, then prove a scalar-safe WORLD8 digest."""

    if dist_module is None:
        import torch.distributed as dist_module
    names = [name for name, _ in named]
    if (
        len(named) != 668
        or len(set(names)) != len(names)
        or expected_count != WORLD_SIZE
    ):
        fail("initial trainable synchronization inventory differs")
    for _, parameter in named:
        dist_module.broadcast(parameter.data, src=0, group=group)
    digest = trainable_digest_v1(named)
    gathered: list[Any] = [None] * expected_count
    dist_module.all_gather_object(gathered, digest, group=group)
    if gathered != [digest] * expected_count:
        fail("initial trainable parameter digest lacks WORLD8 consensus")
    return digest


def elal3_all_parameter_graph_zero_v1(handle: Any, reference: Any) -> Any:
    """Give source-only SP ranks explicit finite zero grads for every ELAL parameter."""

    import torch

    if not isinstance(reference, torch.Tensor) or reference.numel() != 1:
        fail("ELAL graph-zero reference must be one scalar")
    zero = reference.new_zeros(())
    parameters = tuple(handle.components.parameters())
    if len(parameters) != 188 or sum(int(item.numel()) for item in parameters) != ELAL3_FULL_W64_PARAMETERS:
        fail("ELAL full-w64 graph-zero parameter closure differs")
    for parameter in parameters:
        zero = zero + parameter.reshape(-1)[0].to(dtype=reference.dtype) * 0.0
    return zero


def all_trainable_graph_zero_v1(
    named: Sequence[tuple[str, Any]], *, reference: Any
) -> Any:
    """Make every manual-reduction participant own an explicit local grad."""

    zero = reference.new_zeros(())
    if len(named) != 668:
        fail("all-trainable graph-zero expects 480 LoRA plus 188 ELAL tensors")
    for _, parameter in named:
        zero = zero + parameter.reshape(-1)[0].to(dtype=reference.dtype) * 0.0
    return zero


def synchronize_gradients_v1(
    named: Sequence[tuple[str, Any]], parallel: Any, *, bucket_bytes: int = 64 << 20
) -> float:
    import torch
    import torch.distributed as dist

    if not named or bucket_bytes <= 0:
        fail("gradient synchronization scope differs")
    if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()) for _, parameter in named):
        fail("all local LoRA/ELAL gradients must be present and finite before SP reduction")
    squared = torch.zeros((), device=named[0][1].device, dtype=torch.float64)
    bucket: list[Any] = []
    used = 0

    def reduce(values: Sequence[Any]) -> None:
        nonlocal squared
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in values])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=parallel.sp_group)
        flat.div_(float(SP_SIZE))
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=parallel.dp_group)
        flat.div_(float(DP_SIZE))
        offset = 0
        for parameter in values:
            count = parameter.grad.numel()
            parameter.grad.copy_(flat[offset : offset + count].view_as(parameter.grad))
            squared.add_(parameter.grad.detach().to(torch.float64).square().sum())
            offset += count
        if offset != flat.numel():
            fail("gradient bucket scatter differs")

    for _, parameter in named:
        size = parameter.grad.numel() * parameter.grad.element_size()
        if bucket and used + size > bucket_bytes:
            reduce(bucket)
            bucket, used = [], 0
        bucket.append(parameter)
        used += size
    if bucket:
        reduce(bucket)
    norm = float(torch.sqrt(squared).item())
    if not math.isfinite(norm) or norm <= 0.0:
        fail("synchronized WORLD8 gradient norm is zero/non-finite")
    return norm


def gradient_audit_v1(
    named: Sequence[tuple[str, Any]], *, completed_step: int, sp_rank: int
) -> Mapping[str, Any]:
    import torch

    block_squared = {index: 0.0 for index in range(BLOCKS)}
    lora = {"A": [], "B": []}
    builder_squared = 0.0
    zero_count = 0
    for name, parameter in named:
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
            fail(f"missing/non-finite gradient after reduction: {name}")
        norm = float(torch.linalg.vector_norm(parameter.grad.detach().float()).item())
        if norm == 0.0:
            zero_count += 1
        if ".lora_A." in name:
            lora["A"].append(norm)
        elif ".lora_B." in name:
            lora["B"].append(norm)
        elif ".elal3_c0_v1.memory_builder." in name:
            builder_squared += norm * norm
        else:
            match = re.search(r"\.elal3_c0_v1\.injections\.(\d+)\.", name)
            if match is None:
                fail(f"unclassified trainable gradient: {name}")
            block_squared[int(match.group(1))] += norm * norm
    block_norms = {str(key): math.sqrt(value) for key, value in block_squared.items()}
    if (
        len(lora["A"]) != LORA_AFFINES
        or len(lora["B"]) != LORA_AFFINES
        or any(value <= 0 for value in lora["B"])
        or (completed_step >= 2 and any(value <= 0 for value in lora["A"]))
        or builder_squared <= 0
        or any(value <= 0 for value in block_norms.values())
    ):
        fail("all240 LoRA/all30 ELAL synchronized gradient coverage differs")
    return {
        "completed_step": completed_step,
        "sp_rank": sp_rank,
        "local_target_owner": sp_rank in (2, 3),
        "all_local_gradients_present_finite_before_and_after_reduction": True,
        "source_only_sp_graph_zero_installed": True,
        "post_sp_reduction_elal_memory_builder_norm": math.sqrt(builder_squared),
        "post_sp_reduction_elal_block_norms": block_norms,
        "post_sp_reduction_all30_nonzero": True,
        "lora_A_positive": sum(value > 0 for value in lora["A"]),
        "lora_B_positive": sum(value > 0 for value in lora["B"]),
        "per_parameter_zero_count": zero_count,
    }


def hook_audit_v1(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    counts = {index: 0 for index in range(BLOCKS)}
    for row in records:
        index = row.get("block_index")
        if type(index) is not int or index not in counts:
            fail("ELAL hook audit block index differs")
        counts[index] += 1
        if row.get("source_bit_exact") is not True or row.get("padding_bit_exact") is not True:
            fail("ELAL hook changed source/padding rows")
    if any(value <= 0 for value in counts.values()):
        fail("ELAL forward/backward did not use all30 blocks")
    return {"all30_used": True, "calls_by_block": {str(k): v for k, v in counts.items()}}


def deterministic_seed(base: int, *coordinates: Any) -> int:
    payload = "\0".join(str(value) for value in (base, METHOD, *coordinates)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def serialized_model_load_v1() -> Iterator[None]:
    job = os.environ.get("SLURM_JOB_ID", "no-slurm")
    step = os.environ.get("SLURM_STEP_ID", "no-step")
    path = Path(f"/tmp/elal3-c1-{job}-{step}.model-load.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def install_selective_activation_checkpointing_v1(
    model: Any,
    *,
    context_fn: Any,
    checkpoint_fn: Any = None,
) -> tuple[int, ...]:
    """Checkpoint exact stride-4 blocks while preserving ELAL route replay."""

    import torch
    if checkpoint_fn is None:
        from torch.utils.checkpoint import checkpoint as checkpoint_fn
    transformer = model.get_base_model().diff_dec.transformer
    blocks = getattr(transformer, "blocks", None)
    if (
        blocks is None
        or len(blocks) != BLOCKS
        or bool(getattr(transformer, "gradient_checkpointing", False))
        or not callable(context_fn)
        or not callable(checkpoint_fn)
    ):
        fail("selective checkpointing requires exact30 with blanket mode disabled")
    chosen = ACTIVATION_CHECKPOINT_BLOCKS
    for index in chosen:
        block = blocks[index]
        original = block.forward

        def checkpointed_forward(
            *args: Any, _original: Any = original, **kwargs: Any
        ) -> Any:
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)
            return checkpoint_fn(
                _original,
                *args,
                use_reentrant=False,
                context_fn=context_fn,
                **kwargs,
            )

        block.forward = checkpointed_forward
    return chosen


def _pack_latent_v1(value: Any) -> Any:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or value.ndim != 4
        or tuple(int(item) for item in value.shape[:2]) != (LATENT_CHANNELS, LATENT_PHASES)
        or int(value.shape[2]) % 2
        or int(value.shape[3]) % 2
        or not value.is_contiguous()
    ):
        fail("latent patch input differs")
    channels, phases, height, width = map(int, value.shape)
    return (
        value.reshape(channels, phases, height // 2, 2, width // 2, 2)
        .permute(1, 2, 4, 0, 3, 5)
        .reshape(phases * (height // 2) * (width // 2), channels, 1, 2, 2)
        .contiguous()
    )


def prepare_flow_v1(
    *, source: Any, target: Any, epsilon: Any, coordinate: Any, rope: Any, device: Any
) -> Mapping[str, Any]:
    import torch

    for value, label in ((source, "source"), (target, "target"), (epsilon, "epsilon")):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or tuple(int(item) for item in value.shape) != LATENT_SHAPE
            or not value.is_contiguous()
        ):
            fail(f"{label} flow latent differs")
    source_clean = source[0].contiguous()
    target_clean = target[0].contiguous()
    noise = epsilon[0].contiguous()
    sigma = float(coordinate.sigma)
    noisy_target = ((1.0 - sigma) * target_clean + sigma * noise).contiguous()
    target_velocity = (noise - target_clean).contiguous()
    source_patches = _pack_latent_v1(source_clean)
    target_patches = _pack_latent_v1(noisy_target)
    velocity_patches = _pack_latent_v1(target_velocity)
    source_tokens = int(source_patches.shape[0])
    target_tokens = int(target_patches.shape[0])
    if (
        source_tokens != TOKENS_PER_ROLE
        or target_tokens != TOKENS_PER_ROLE
        or source_tokens + target_tokens != PACKED_TOTAL_TOKENS
    ):
        fail("registered v2 latent packed-token geometry differs")
    input_patches = torch.cat((source_patches, target_patches), dim=0).to(device)
    source_rope = rope(source.to(device), source_id=1)
    target_rope = rope(noisy_target.unsqueeze(0).to(device), source_id=0)
    rotary = torch.cat((source_rope, target_rope), dim=2).squeeze(0).permute(1, 0, 2).contiguous()
    velocity = velocity_patches.permute(0, 2, 3, 4, 1).reshape(
        1, target_tokens, PATCH_VALUES
    ).to(device)
    return {
        "input_patches": input_patches,
        "rotary": rotary,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "total_tokens": source_tokens + target_tokens,
        "patch_grid": (
            LATENT_PHASES,
            int(source.shape[3]) // 2,
            int(source.shape[4]) // 2,
        ),
        "target_velocity": velocity,
    }


def registered_sp4_partition_v1(
    *, total_tokens: int, condition_tokens: int, sp_rank: int
) -> Mapping[str, Any]:
    if (
        total_tokens != PACKED_TOTAL_TOKENS
        or condition_tokens != TOKENS_PER_ROLE
        or sp_rank not in range(SP_SIZE)
        or total_tokens % SP_SIZE
    ):
        fail("registered v2 SP4 partition inputs differ")
    local_tokens = total_tokens // SP_SIZE
    start = sp_rank * local_tokens
    stop = start + local_tokens
    source_only = stop <= condition_tokens
    target_only = start >= condition_tokens
    if (
        local_tokens != LOCAL_SP_TOKENS
        or source_only != (sp_rank in (0, 1))
        or target_only != (sp_rank in (2, 3))
        or source_only == target_only
    ):
        fail("registered v2 SP4 source/target rank ownership differs")
    return {
        "sp_rank": sp_rank,
        "local_start": start,
        "local_stop": stop,
        "local_tokens": local_tokens,
        "source_only": source_only,
        "target_only": target_only,
    }


def predict_target_v1(
    *, renderer: Any, packed: Mapping[str, Any], coordinate: Any, text_lens: Any,
    text_embs: Any
) -> Any:
    import torch

    rotary = packed["rotary"].permute(1, 0, 2).unsqueeze(0)
    value = renderer.diff_dec.shared_step(
        model_id="transformer_1",
        noisy_latents=packed["embedded"],
        timesteps=packed["embedded"].new_tensor([coordinate.timestep], dtype=torch.int64),
        cond_embeds=text_embs,
        rotary_embs=rotary,
        batch_vae_seqlen=[packed["total_tokens"]],
        batch_text_seqlen=text_lens,
    )
    target = value[:, packed["source_tokens"] :, :]
    if tuple(target.shape) != (1, packed["target_tokens"], PATCH_VALUES):
        fail("official Bernini target prediction geometry differs")
    return target


def materialize_text_condition_v1(
    *, tokenizer: Any, renderer: Any, runtime: Any, instruction: str, device: Any
) -> tuple[Any, Any, Mapping[str, Any]]:
    import torch

    tokenized = runtime.tokenize_generic_instruction(tokenizer, instruction, device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        text_lens, text_embs = renderer.get_t5_text_embeddings(
            tokenized["input_ids"], tokenized["attention_mask"], tokenized["t5_input_lens"]
        )
    if isinstance(text_embs, (list, tuple)):
        if len(text_embs) != 1:
            fail("official UMT5 embedding list differs")
        text_embs = text_embs[0]
    text_embs = text_embs.detach().clone(memory_format=torch.contiguous_format)
    if (
        not isinstance(text_embs, torch.Tensor)
        or list(text_embs.shape) != [1, 512, 4096]
        or text_embs.dtype != torch.bfloat16
        or text_embs.requires_grad
        or torch.is_inference(text_embs)
        or not bool(torch.isfinite(text_embs).all().item())
    ):
        fail("frozen contextual UMT5 embedding differs")
    if isinstance(text_lens, torch.Tensor):
        text_lens = text_lens.detach().clone(memory_format=torch.contiguous_format)
    receipt = {
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "tokenized_length": int(tokenized["t5_input_lens"].item()),
        "renderer_context_length": 512,
        "target_or_target_caption_entered_text_encoder": False,
    }
    return text_lens, text_embs, receipt


def atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
    except Exception:
        raise


def create_only_torch_save(path: Path, value: Any) -> None:
    import torch

    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)


def save_checkpoint_v1(
    *, root: Path, step: int, named: Sequence[tuple[str, Any]], optimizer: Any,
    common: Mapping[str, Any], save_optimizer: bool
) -> Mapping[str, Any]:
    import torch

    final = root / f"checkpoint-{step:08d}"
    if final.exists() or final.is_symlink():
        fail(f"refusing to overwrite checkpoint: {final}")
    final.mkdir(mode=0o700)
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in named}
    lora = {name: value for name, value in state.items() if ".lora_" in name}
    elal = {name: value for name, value in state.items() if ".elal3_c0_v1." in name}
    if len(lora) != 480 or len(elal) != 188 or set(lora) | set(elal) != set(state):
        fail("checkpoint LoRA/ELAL state partition differs")
    adapter_payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "lora_state": lora,
        "elal3_full_w64_state": elal,
        "formal_c1_authorized": False,
        "exact160_authorized": False,
        "scientific_claim_authorized": False,
        "source_instruction_inference": False,
        "oracle_q_teacher_forced": True,
    }
    adapter_path = final / "adapter-and-elal3.pt"
    create_only_torch_save(adapter_path, adapter_payload)
    loaded = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(loaded, Mapping)
        or set(loaded.get("lora_state", ())) != set(lora)
        or set(loaded.get("elal3_full_w64_state", ())) != set(elal)
        or any(
            not torch.equal(
                loaded[family][name].reshape(-1).view(torch.uint8),
                reference.reshape(-1).view(torch.uint8),
            )
            for family, values in (("lora_state", lora), ("elal3_full_w64_state", elal))
            for name, reference in values.items()
        )
    ):
        fail("strict checkpoint adapter reload differs")
    optimizer_path: Optional[Path] = None
    if save_optimizer:
        optimizer_path = final / "optimizer.pt"
        create_only_torch_save(optimizer_path, optimizer.state_dict())
        loaded_optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded_optimizer, Mapping) or set(loaded_optimizer) != {"state", "param_groups"}:
            fail("strict optimizer checkpoint reload differs")
    metadata = {
        **dict(common),
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "adapter_file": adapter_path.name,
        "adapter_sha256": file_sha256(adapter_path),
        "optimizer_file": optimizer_path.name if optimizer_path is not None else None,
        "optimizer_sha256": file_sha256(optimizer_path) if optimizer_path is not None else None,
        "strict_weights_only_reload_verified": True,
        "trainable_parameter_sha256": trainable_digest_v1(named),
    }
    metadata_path = final / "CHECKPOINT_RECEIPT.json"
    atomic_create_json(metadata_path, {**metadata, "receipt_digest": object_sha256(metadata)})
    os.chmod(final, 0o500)
    return {
        "step": step,
        "path": str(final),
        "adapter_sha256": metadata["adapter_sha256"],
        "optimizer_sha256": metadata["optimizer_sha256"],
        "checkpoint_receipt_sha256": file_sha256(metadata_path),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--packet-root", required=True)
    value.add_argument("--latent-bundle", required=True)
    value.add_argument("--expected-latent-bundle-sha256", required=True)
    value.add_argument("--latent-bundle-receipt", required=True)
    value.add_argument("--expected-latent-bundle-receipt-sha256", required=True)
    value.add_argument("--external-optimizer-authority", required=True)
    value.add_argument("--expected-external-optimizer-authority-sha256", required=True)
    value.add_argument("--model-authority", required=True)
    value.add_argument("--expected-model-authority-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--max-steps", type=int, choices=(1, 10, 20), required=True)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    value.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    value.add_argument("--expected-runner-source-sha256", required=True)
    value.add_argument("--expected-train-lora-source-sha256", required=True)
    value.add_argument("--expected-elal3-core-source-sha256", required=True)
    value.add_argument("--expected-elal3-label-source-sha256", required=True)
    value.add_argument("--expected-packed-lora-source-sha256", required=True)
    value.add_argument("--expected-runtime-source-sha256", required=True)
    value.add_argument("--expected-sigma-source-sha256", required=True)
    value.add_argument("--preflight-only", action="store_true")
    value.add_argument("--ack-simulator-oracle-q-overfit-only", action="store_true")
    value.add_argument("--ack-not-source-instruction-inference", action="store_true")
    value.add_argument("--ack-not-formal-c1", action="store_true")
    value.add_argument("--ack-not-exact160", action="store_true")
    value.add_argument("--ack-no-scientific-claim", action="store_true")
    return value


def validate_args(args: argparse.Namespace) -> None:
    if not all((
        args.ack_simulator_oracle_q_overfit_only,
        args.ack_not_source_instruction_inference,
        args.ack_not_formal_c1,
        args.ack_not_exact160,
        args.ack_no_scientific_claim,
    )):
        fail("all five oracle-q diagnostic acknowledgements are mandatory")
    if args.max_steps not in (1, 10, 20):
        fail("optimizer updates must be exactly 1, 10, or 20")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        fail("seed must be a non-negative signed-63 integer")
    if args.learning_rate != DEFAULT_LR or args.max_grad_norm != DEFAULT_MAX_GRAD_NORM:
        fail("optimizer hyperparameters are fixed for this diagnostic")
    sha_names = (
        "expected_latent_bundle_sha256",
        "expected_latent_bundle_receipt_sha256",
        "expected_external_optimizer_authority_sha256",
        "expected_model_authority_sha256",
        "expected_runner_source_sha256",
        "expected_train_lora_source_sha256",
        "expected_elal3_core_source_sha256",
        "expected_elal3_label_source_sha256",
        "expected_packed_lora_source_sha256",
        "expected_runtime_source_sha256",
        "expected_sigma_source_sha256",
    )
    for name in sha_names:
        _require_sha(getattr(args, name), label=name)
    if args.expected_latent_bundle_sha256 != LATENT_BUNDLE_SHA256:
        fail("latent bundle CLI SHA differs from registered v2 literal")
    if args.expected_latent_bundle_receipt_sha256 != LATENT_BUNDLE_RECEIPT_SHA256:
        fail("latent receipt CLI SHA differs from registered v2 literal")
    if args.expected_external_optimizer_authority_sha256 != EXTERNAL_OPTIMIZER_AUTHORITY_SHA256:
        fail("external optimizer authority CLI SHA differs from registered literal")
    if args.expected_model_authority_sha256 != MODEL_AUTHORITY_SHA256:
        fail("model authority CLI SHA differs from registered literal")
    output = Path(args.output).expanduser()
    if (
        not output.is_absolute()
        or output.exists()
        or output.is_symlink()
        or "elal3_c1" not in output.name.lower()
    ):
        fail("output must be one fresh absolute ELAL3_C1 path")


def _validate_imported_local_source(
    module: Any, *, path: Path, expected_sha256: str, label: str
) -> Mapping[str, Any]:
    imported = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    canonical = path.resolve(strict=True)
    if imported != canonical or file_sha256(canonical) != expected_sha256:
        fail(f"imported {label} source identity differs")
    return {"path": str(canonical), "sha256": expected_sha256, "size": canonical.stat().st_size}


def _memory_receipt_v1(device: Any, *, world_rank: int) -> Mapping[str, Any]:
    import torch

    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = allocated / float(total)
    return {
        "world_rank": world_rank,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "device_total_bytes": total,
        "peak_allocated_fraction": fraction,
        "strictly_greater_than_half": fraction > MEMORY_FRACTION_GATE,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    validate_args(args)
    runner_path = Path(__file__).resolve()
    if file_sha256(runner_path) != args.expected_runner_source_sha256:
        fail("runner source SHA differs")

    import train_lora as legacy
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=BERNINI_COMMIT,
            expected_veomni_commit=VEOMNI_COMMIT,
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise ELAL3C1TrainingError(str(error)) from error
    if (
        transformer_config.get("num_layers") != BLOCKS
        or transformer_config.get("hidden_size") not in (None, HIDDEN)
        or transformer_config.get("num_attention_heads") != 12
        or transformer_config.get("attention_head_dim") != 128
    ):
        fail("Bernini-R 1.3B transformer geometry differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import gc
    import torch
    import torch.distributed as dist
    import torch.utils.checkpoint as torch_checkpoint
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    import bernini.pipeline as bernini_pipeline
    import diffusers
    import diffusers.models.autoencoders.autoencoder_kl_wan as diffusers_wan
    import elal3_c0_v1 as elal3
    import elal3_simulator_label_v1 as label_module
    import packed_preservation_lora_v2 as packed_lora
    import source_self_runtime as runtime
    import inference_sigma_strata as sigma_strata

    local_sources = {
        "train_lora": _validate_imported_local_source(
            legacy, path=METHOD_ROOT / "train_lora.py",
            expected_sha256=args.expected_train_lora_source_sha256, label="train_lora"
        ),
        "elal3_core": _validate_imported_local_source(
            elal3, path=METHOD_ROOT / "elal3_c0_v1.py",
            expected_sha256=args.expected_elal3_core_source_sha256, label="ELAL3 core"
        ),
        "elal3_label": _validate_imported_local_source(
            label_module, path=METHOD_ROOT / "elal3_simulator_label_v1.py",
            expected_sha256=args.expected_elal3_label_source_sha256, label="ELAL3 label"
        ),
        "packed_lora": _validate_imported_local_source(
            packed_lora, path=METHOD_ROOT / "packed_preservation_lora_v2.py",
            expected_sha256=args.expected_packed_lora_source_sha256, label="packed LoRA"
        ),
        "runtime": _validate_imported_local_source(
            runtime, path=METHOD_ROOT / "source_self_runtime.py",
            expected_sha256=args.expected_runtime_source_sha256, label="WORLD8 runtime"
        ),
        "sigma": _validate_imported_local_source(
            sigma_strata, path=METHOD_ROOT / "inference_sigma_strata.py",
            expected_sha256=args.expected_sigma_source_sha256, label="sigma strata"
        ),
    }
    external_path = Path(args.external_optimizer_authority).expanduser().resolve(strict=True)
    model_authority_path = Path(args.model_authority).expanduser().resolve(strict=True)
    external_authority = validate_external_optimizer_authority(
        external_path, expected_sha256=args.expected_external_optimizer_authority_sha256
    )
    contract = runtime.distributed_contract()
    if (
        contract.world_size != WORLD_SIZE
        or contract.local_world_size != WORLD_SIZE
        or contract.topology.sp_size != SP_SIZE
        or contract.topology.dp_size != DP_SIZE
    ):
        fail("trainer requires one exact WORLD8 DP2xSP4 node")
    device = runtime.initialise_distributed(contract)
    parallel = runtime.validate_parallel_state(contract, init_parallel_state(ulysses_size=SP_SIZE))
    placement = validate_runtime_authority_placement_v1(external_authority)
    seed_everything(args.seed)

    model_box: list[Any] = [None]
    if contract.rank == 0:
        try:
            model_box[0] = {
                "ok": True,
                "value": validate_model_authority(
                    model_authority_path,
                    expected_sha256=args.expected_model_authority_sha256,
                    bernini_root=bernini_root,
                    checkpoint_root=checkpoint,
                ),
            }
        except Exception as error:
            model_box[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(model_box, src=0, group=parallel.world_group)
    if not isinstance(model_box[0], Mapping) or model_box[0].get("ok") is not True:
        fail(f"rank-zero real-model authority validation failed: {model_box[0]!r}")
    model_authority = model_box[0]["value"]
    model_rows = {(row["root"], row["relative_path"]): row for row in model_authority["files"]}
    imported_model_paths = {
        ("bernini", "bernini/pipeline.py"): Path(bernini_pipeline.__file__).resolve(strict=True),
        ("python_env", "diffusers/__init__.py"): Path(diffusers.__file__).resolve(strict=True),
        ("python_env", "diffusers/models/autoencoders/autoencoder_kl_wan.py"):
            Path(diffusers_wan.__file__).resolve(strict=True),
    }
    for key, imported in imported_model_paths.items():
        row = model_rows[key]
        root = bernini_root if key[0] == "bernini" else Path(model_authority["python_env_root"])
        expected = root / key[1]
        if imported != expected.resolve(strict=True) or file_sha256(imported) != row["sha256"]:
            fail(f"imported real-model implementation differs: {key}")

    packet_root = Path(args.packet_root).expanduser().resolve(strict=True)
    bundle = load_latent_bundle_v1(
        bundle_path=Path(args.latent_bundle).expanduser().resolve(strict=True),
        expected_bundle_sha256=args.expected_latent_bundle_sha256,
        receipt_path=Path(args.latent_bundle_receipt).expanduser().resolve(strict=True),
        expected_receipt_sha256=args.expected_latent_bundle_receipt_sha256,
        packet_root=packet_root,
        external_optimizer_authority_path=external_path,
        model_authority_path=model_authority_path,
        checkpoint_root=checkpoint,
    )
    label = label_module.load_oracle_q_label_v1(
        packet_root,
        row_id=ROW_ID,
        patch_grid=bundle.patch_grid,
        external_authority_path=external_path,
        external_authority_sha256=EXTERNAL_OPTIMIZER_AUTHORITY_SHA256,
        device=device,
        dtype=torch.float32,
    )
    derived_authority = label_module.build_derivative_authority_v1(
        label,
        external_authority_path=external_path,
        external_authority_sha256=EXTERNAL_OPTIMIZER_AUTHORITY_SHA256,
    )
    validate_derivative_authority_v1(
        derived_authority, label_receipt=label.receipt, max_steps=args.max_steps
    )
    runtime.digest_consensus(
        derived_authority["authority_digest"], group=parallel.world_group,
        expected_count=WORLD_SIZE, label="derived oracle-q authority"
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    with serialized_model_load_v1():
        renderer = BerniniRendererModel(config)
        renderer.requires_grad_(False)
        specs = packed_lora.select_projection_specs(renderer, "all-attention")
        model = get_peft_model(
            renderer,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_RANK,
                lora_dropout=0.0,
                bias="none",
                target_modules=[item.name for item in specs],
            ),
        )
        transformer = model.get_base_model().diff_dec.transformer
        elal_handle = elal3.install_elal3_c0_v1(
            transformer, variant="full", attention_width=64, hidden_size=HIDDEN
        )
        activation_checkpoint_blocks = install_selective_activation_checkpointing_v1(
            model,
            context_fn=elal3.elal3_checkpoint_context_fn_v1,
        )
        if activation_checkpoint_blocks != ACTIVATION_CHECKPOINT_BLOCKS:
            fail("selective activation checkpoint exact8 schedule differs")
        model.to(device)
    post_deserialize_model_replay = replay_model_authority_world8_v1(
        dist=dist,
        group=parallel.world_group,
        rank=contract.rank,
        reference=model_authority,
        authority_path=model_authority_path,
        expected_sha256=args.expected_model_authority_sha256,
        bernini_root=bernini_root,
        checkpoint_root=checkpoint,
        stage="post_deserialize",
    )
    runtime.digest_consensus(
        post_deserialize_model_replay["replayed_object_sha256"],
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="post-deserialize exact9 model authority replay",
    )
    model.train()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel) or model.__class__.__name__ in {
        "FullyShardedDataParallel", "DistributedDataParallel"
    }:
        fail("manual SP4->DP2 reduction forbids DDP/FSDP double ownership")
    base_renderer = model.get_base_model()
    base_renderer.t5_text_encoder.eval()
    if any(parameter.dtype != torch.float32 for parameter in elal_handle.components.parameters()):
        fail("ELAL3 full-w64 dense modules must remain FP32 after accelerator placement")
    named = exact_trainable_named_parameters_v1(model)
    inventory = trainable_inventory_v1(named)
    inventory_digest = object_sha256(inventory)
    packed_lora.validate_lora_installation(model, specs)
    runtime.digest_consensus(
        inventory_digest, group=parallel.world_group, expected_count=WORLD_SIZE,
        label="all240-r256 plus ELAL3 full-w64 inventory"
    )
    initial_digest = synchronize_initial_parameters_v1(named, parallel.world_group)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    instruction = str(label.verified_row.row["instruction"])
    text_lens, text_embs, text_receipt = materialize_text_condition_v1(
        tokenizer=tokenizer, renderer=base_renderer, runtime=runtime,
        instruction=instruction, device=device
    )
    base_renderer.t5_text_encoder = None
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if base_renderer.t5_text_encoder is not None:
        fail("frozen T5 was not retired before optimizer construction")
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)

    output = Path(args.output)
    checkpoints = output / "checkpoints"
    if contract.rank == 0:
        output.mkdir(mode=0o700)
        checkpoints.mkdir(mode=0o700)
    dist.barrier(group=parallel.world_group)
    common = {
        "method": METHOD,
        "status": "SIMULATOR_ORACLE_Q_OVERFIT_DIAGNOSTIC_ONLY",
        "row_id": ROW_ID,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "fresh_official_base": True,
        "resume_consumed": False,
        "oracle_q_teacher_forced": True,
        "source_instruction_inference": False,
        "formal_c1_authorized": False,
        "exact160_authorized": False,
        "scientific_claim_authorized": False,
        "real_video_data": False,
        "action_encoder_qualified": False,
        "action_predictor_present": False,
        "frozen_teacher_used": False,
        "frozen_velocity_reference_used": False,
        "self_distillation_used": False,
        "reward_used": False,
        "lora_affines": LORA_AFFINES,
        "lora_rank": LORA_RANK,
        "elal3_variant": "full-w64",
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        "trainable_inventory_digest": inventory_digest,
        "initial_trainable_sha256": initial_digest,
        "latent_bundle_sha256": bundle.bundle_sha256,
        "latent_bundle_receipt_sha256": bundle.receipt_sha256,
        "label_digest": label.receipt["label_digest"],
        "derived_authority_digest": derived_authority["authority_digest"],
        "external_optimizer_authority_sha256": EXTERNAL_OPTIMIZER_AUTHORITY_SHA256,
        "model_authority_sha256": MODEL_AUTHORITY_SHA256,
        "model_authority_digest": MODEL_AUTHORITY_DIGEST,
        "model_authority_verified_before_and_after_deserialization": True,
        "post_deserialize_model_authority_replay": post_deserialize_model_replay,
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
        "runtime_placement": placement,
        "local_source_closure": local_sources,
        "text_condition": text_receipt,
        "manual_gradient_reduction": "single_owner_SP4_mean_then_DP2_mean_no_DDP_no_FSDP",
        "memory_gate": "each_rank_max_memory_allocated_over_total_strictly_gt_0.5",
        "activation_checkpoint_profile": ACTIVATION_CHECKPOINT_PROFILE,
        "activation_checkpointed_blocks": list(ACTIVATION_CHECKPOINT_BLOCKS),
        "activation_uncheckpointed_blocks": list(ACTIVATION_UNCHECKPOINTED_BLOCKS),
        "activation_checkpoint_nonreentrant": True,
        "activation_checkpoint_elal_route_context_replay": True,
        "memory_gate_true_training_tensors_only": True,
        "dummy_or_padding_memory_allocations": False,
    }
    if args.preflight_only:
        preflight = {
            **common,
            "schema_version": RECEIPT_SCHEMA,
            "status": "PRECHECK_COMPLETE_NO_OPTIMIZER_CONSTRUCTED_NO_UPDATE",
            "completed_optimizer_steps": 0,
            "preflight_only": True,
        }
        if contract.rank == 0:
            atomic_create_json(
                output / "PRECHECK_RECEIPT.json",
                {**preflight, "receipt_digest": object_sha256(preflight)},
            )
        dist.barrier(group=parallel.world_group)
        return 0

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=DEFAULT_LR,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    checkpoint_records: list[Mapping[str, Any]] = []
    if contract.rank == 0:
        checkpoint_records.append(
            save_checkpoint_v1(
                root=checkpoints, step=0, named=named, optimizer=optimizer,
                common=common, save_optimizer=False
            )
        )
    dist.barrier(group=parallel.world_group)

    history: list[Mapping[str, Any]] = []
    parameter_digests = {initial_digest}
    started = time.monotonic()
    for step_zero in range(args.max_steps):
        completed = step_zero + 1
        coordinate = sigma_strata.select_sigma_stratum(step_zero)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        noise_seed = deterministic_seed(args.seed, "flow", step_zero, contract.arm_index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        epsilon = torch.randn(bundle.target.shape, generator=generator, dtype=torch.float32).contiguous()
        packed = dict(
            prepare_flow_v1(
                source=bundle.source, target=bundle.target, epsilon=epsilon,
                coordinate=coordinate, rope=rope, device=device
            )
        )
        if packed["patch_grid"] != bundle.patch_grid:
            fail("runtime flow patch grid differs from oracle-q bundle")
        local_partition = registered_sp4_partition_v1(
            total_tokens=packed["total_tokens"],
            condition_tokens=packed["source_tokens"],
            sp_rank=contract.sp_rank,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embedded = transformer.patch_embedding(packed["input_patches"]).flatten(1).unsqueeze(0)
        if tuple(embedded.shape) != (1, packed["total_tokens"], HIDDEN):
            fail("pre-SP packed embedding geometry differs")
        packed["embedded"] = embedded
        # Keep oracle-q ingress and the ELAL memory builder explicitly FP32;
        # renderer autocast starts only after this typed boundary.
        with torch.autocast(device_type="cuda", enabled=False):
            memory = elal_handle.build_memory(label.latent)
        route = elal3.ELAL3RouteV1(
            total_tokens=packed["total_tokens"],
            condition_tokens=packed["source_tokens"],
            sequence_parallel_rank=contract.sp_rank,
            sequence_parallel_size=SP_SIZE,
            memory=memory,
            route_identity=f"{ROW_ID}:seed{args.seed}:step{completed}:rank{contract.rank}",
        )
        audit_start = len(elal_handle.audit_records)
        with elal_handle.route(route):
            with torch_checkpoint.set_checkpoint_early_stop(False):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = predict_target_v1(
                        renderer=base_renderer, packed=packed, coordinate=coordinate,
                        text_lens=text_lens, text_embs=text_embs
                    )
                    event_mask = pack_vae_partition_mask_v1(
                        label.event_mask_vae, target_velocity=packed["target_velocity"]
                    )
                    context_mask = pack_vae_partition_mask_v1(
                        label.context_mask_vae, target_velocity=packed["target_velocity"]
                    )
                    raw_loss, loss_receipt = partitioned_flow_matching_loss_v1(
                        prediction, packed["target_velocity"], event_mask, context_mask
                    )
                    loss = raw_loss + all_trainable_graph_zero_v1(named, reference=raw_loss)
                if not bool(torch.isfinite(loss.detach()).item()):
                    fail("optimizer loss is non-finite")
                loss.backward()
        hook_receipt = hook_audit_v1(elal_handle.audit_records[audit_start:])
        local_ready = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
            for _, parameter in named
        )
        if not runtime.world_all_true(local_ready, group=parallel.world_group):
            fail("some SP rank lacks explicit finite local gradients before manual reduction")
        local_elal_sq = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for name, parameter in named if ".elal3_c0_v1." in name
        )
        local_target_owner = contract.sp_rank in (2, 3)
        if (local_target_owner and local_elal_sq <= 0.0) or (
            not local_target_owner and local_elal_sq != 0.0
        ):
            fail("actual-shape SP4 target-owner/source-only ELAL local gradient split differs")
        synchronized_norm = synchronize_gradients_v1(named, parallel)
        gradient_receipt = gradient_audit_v1(
            named, completed_step=completed, sp_rank=contract.sp_rank
        )
        preclip = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named], DEFAULT_MAX_GRAD_NORM
            ).item()
        )
        if not math.isfinite(preclip) or preclip <= 0.0:
            fail("preclip gradient norm is zero/non-finite")
        optimizer.step()
        torch.cuda.synchronize(device)
        memory_local = _memory_receipt_v1(device, world_rank=contract.rank)
        memory_world: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(memory_world, memory_local, group=parallel.world_group)
        if (
            [row["world_rank"] for row in memory_world] != list(range(WORLD_SIZE))
            or any(row["strictly_greater_than_half"] is not True for row in memory_world)
        ):
            fail(f"per-rank >50% allocated-memory gate failed: {memory_world!r}")
        parameter_digest = trainable_digest_v1(named)
        runtime.digest_consensus(
            parameter_digest, group=parallel.world_group, expected_count=WORLD_SIZE,
            label=f"post-update parameters step {completed}"
        )
        if parameter_digest in parameter_digests:
            fail("optimizer update did not change exact trainable bytes")
        parameter_digests.add(parameter_digest)
        local_step = {
            "world_rank": contract.rank,
            "dp_arm": contract.arm_index,
            "sp_rank": contract.sp_rank,
            "local_target_owner": local_target_owner,
            "registered_sp4_partition": local_partition,
            "local_elal_gradient_norm_before_reduction": math.sqrt(local_elal_sq),
            "hook_audit": hook_receipt,
            "gradient_audit": gradient_receipt,
            "loss": loss_receipt,
        }
        gathered: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local_step, group=parallel.world_group)
        step_receipt = {
            "step": completed,
            "row_id": ROW_ID,
            "noise_seeds_by_dp_arm": [
                deterministic_seed(args.seed, "flow", step_zero, arm) for arm in range(DP_SIZE)
            ],
            "sigma_stratum": coordinate.as_dict(),
            "loss_dp_leaders": [gathered[0]["loss"], gathered[SP_SIZE]["loss"]],
            "synchronized_gradient_norm": synchronized_norm,
            "preclip_gradient_norm": preclip,
            "parameter_sha256": parameter_digest,
            "all8_sp_gradient_and_hook_receipts": gathered,
            "memory_world8": memory_world,
            "memory_gate_all8_pass": True,
            "optimizer_step_executed": True,
        }
        if contract.rank == 0:
            history.append(step_receipt)
            print(json.dumps(step_receipt, sort_keys=True), flush=True)
        del epsilon, packed, embedded, memory, route, prediction, event_mask, context_mask, raw_loss, loss

    final_pre_publish_model_replay = replay_model_authority_world8_v1(
        dist=dist,
        group=parallel.world_group,
        rank=contract.rank,
        reference=model_authority,
        authority_path=model_authority_path,
        expected_sha256=args.expected_model_authority_sha256,
        bernini_root=bernini_root,
        checkpoint_root=checkpoint,
        stage="final_pre_publish",
    )
    runtime.digest_consensus(
        final_pre_publish_model_replay["replayed_object_sha256"],
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="final pre-publish exact9 model authority replay",
    )
    common["model_authority_verified_at_final_pre_publish"] = True
    common["final_pre_publish_model_authority_replay"] = final_pre_publish_model_replay
    final_digest = trainable_digest_v1(named)
    if contract.rank == 0:
        checkpoint_records.append(
            save_checkpoint_v1(
                root=checkpoints, step=args.max_steps, named=named, optimizer=optimizer,
                common=common, save_optimizer=True
            )
        )
        unsigned_receipt = {
            **common,
            "schema_version": RECEIPT_SCHEMA,
            "status": "TRAINING_COMPLETE_SIMULATOR_ORACLE_Q_OVERFIT_DIAGNOSTIC_ONLY",
            "preflight_only": False,
            "completed_optimizer_steps": args.max_steps,
            "requested_optimizer_steps": args.max_steps,
            "fresh_initialization_verified": True,
            "initial_parameter_sha256": initial_digest,
            "final_parameter_sha256": final_digest,
            "parameters_changed": final_digest != initial_digest,
            "history": history,
            "checkpoint_records": checkpoint_records,
            "memory_gate_all_steps_all8_strictly_gt_half": True,
            "all_steps_finite_nonzero_synchronized_gradients": True,
            "all_steps_all30_elal_used_and_nonzero_after_sp_reduction": True,
            "elapsed_seconds": time.monotonic() - started,
            "decoded_review_pending": True,
        }
        atomic_create_json(
            output / "TRAINING_RECEIPT.json",
            {**unsigned_receipt, "receipt_digest": object_sha256(unsigned_receipt)},
        )
    dist.barrier(group=parallel.world_group)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ELAL3C1TrainingError as error:
        print(f"ELAL3_C1_TRAINING_ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
