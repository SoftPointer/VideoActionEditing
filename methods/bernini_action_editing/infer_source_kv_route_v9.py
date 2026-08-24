#!/usr/bin/env python3
"""Strict trained-adapter deployment for Bernini CSV-ART V9.

The only external model conditions are one exact 81-frame source video and
one action instruction.  At every one of the forty official UniPC solver
steps this runner executes the following source-carrier cell on the *same*
packed state, timestep, rotary embedding, negative control and APG settings::

    source-only semantic-noop capture (adapter disabled)
    frozen negative, frozen noop, frozen action (adapter disabled)
    adapted noop, adapted action (the exact V9 adapter enabled)

All thirty self-attention blocks capture/replay detached post-RoPE source K/V.
Frozen action/noop fields are diagnostics only.  The sole deployment clean
field is

    E_k = S + Q0(A_theta,k - N_theta,k)
    Q0(X) = X - X[:, :, :1]

and the sole scheduler model output is

    v_deploy,k = (x_k - E_k) / sigma_k .

There is no rho, radius, clipping, frozen-field mixture, target video, mask,
track, swept tube, pose, trajectory, optical flow, or first-frame anchor.  The
computed velocity is passed to the original pinned UniPC ``step`` exactly
once.  The implementation fails closed on an incomplete/non-V9 adapter,
source or adapter TOCTOU, a different call graph, any non-positive sigma, or
anything other than exact all-30 capture40/replay200 evidence per rank.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as legacy  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import source_kv_replay as replay  # noqa: E402
import source_kv_route_batches as route_batches  # noqa: E402
import source_kv_route_scope as route_scope  # noqa: E402
import train_source_kv_route_auh as trainer  # noqa: E402
import tri_branch_unipc as tri  # noqa: E402


METHOD_NAME = "bernini-frozen-source-kv-csv-art-v9-deployment"
RECEIPT_SCHEMA = "bernini-source-kv-route-inference-receipt-v9"
EXPECTED_FRAMES = 81
EXPECTED_PHASES = 21
EXPECTED_STEPS = 40
EXPECTED_SEED = 2027
EXPECTED_ULYSSES_SIZE = 4
EXPECTED_PEFT_VERSION = "0.19.1"
EXPECTED_BLOCKS = tuple(range(30))
EXPECTED_BRANCH_ORDER = (
    "frozen_noop_source_only_carrier",
    "frozen_negative_full_pair",
    "frozen_noop_full_pair",
    "frozen_action_full_pair",
    "adapted_noop_full_pair",
    "adapted_action_full_pair",
)
REPLAY_BRANCH_ORDER = (
    "frozen_negative",
    "frozen_noop",
    "frozen_action",
    "adapted_noop",
    "adapted_action",
)
DEPLOYMENT_CLEAN_FORMULA = "E_k=S+Q0(A_theta,k-N_theta,k)"
DEPLOYMENT_VELOCITY_FORMULA = "v_deploy,k=(x_k-E_k)/sigma_k"
GAUGE_FORMULA = "Q0(X)=X-X[:,:,0:1]"
ADAPTER_READY_STATUS = (
    "post_save_adapter_tensor_file_roundtrip_and_"
    "optimizer_load_state_dict_complete"
)
EXPECTED_SERIALIZED_TARGET_PATTERN_COUNT = 34
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SourceKVRouteInferenceError(RuntimeError):
    """Raised before an unaudited V9 field reaches the scheduler."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceKVRouteInferenceError(f"{label} must be a mapping")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SourceKVRouteInferenceError(f"{label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SourceKVRouteInferenceError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _plain_file(path: Path, *, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as error:
        raise SourceKVRouteInferenceError(f"cannot stat {label}: {path}") from error
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise SourceKVRouteInferenceError(f"{label} is not a plain file: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceKVRouteInferenceError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise SourceKVRouteInferenceError(f"{label} must contain one JSON object")
    return value


def expected_serialized_target_patterns() -> list[str]:
    """PEFT 0.19's unique suffix cover of the exact-92 V9 scope."""

    patterns = ["attn2.to_q", "attn2.to_out.0"]
    for block in range(7, 23):
        patterns.extend(
            (f"{block}.attn1.to_q", f"{block}.attn1.to_out.0")
        )
    result = sorted(patterns)
    if len(result) != EXPECTED_SERIALIZED_TARGET_PATTERN_COUNT:
        raise SourceKVRouteInferenceError("V9 serialized target pattern count changed")
    return result


def _expand_serialized_target_patterns(patterns: Sequence[str]) -> list[str]:
    universe = [
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(30)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    ]
    return sorted(
        module
        for module in universe
        if any(
            module == pattern or module.endswith(f".{pattern}")
            for pattern in patterns
        )
    )


def _validate_adapter_config(config: Mapping[str, Any]) -> list[str]:
    if config.get("peft_type") != "LORA":
        raise SourceKVRouteInferenceError("adapter is not LoRA")
    if (
        config.get("r") != route_scope.LORA_RANK
        or float(config.get("lora_alpha", math.nan)) != route_scope.LORA_ALPHA
        or float(config.get("lora_dropout", math.nan)) != 0.0
        or config.get("bias") != "none"
        or config.get("modules_to_save") not in (None, [])
        or config.get("use_dora") not in (None, False)
        or config.get("use_rslora") not in (None, False)
    ):
        raise SourceKVRouteInferenceError("V9 LoRA hyperparameters differ")
    serialized = config.get("target_modules")
    expected_patterns = expected_serialized_target_patterns()
    canonical = route_scope.canonical_target_modules()
    if (
        not isinstance(serialized, list)
        or not all(isinstance(item, str) and item for item in serialized)
        or len(serialized) != len(set(serialized))
        or set(serialized) != set(expected_patterns)
        or _expand_serialized_target_patterns(serialized) != canonical
    ):
        raise SourceKVRouteInferenceError(
            "serialized target_modules are not the audited 34-pattern exact92 expansion"
        )
    return sorted(serialized)


def verify_official_source_prefix(
    *,
    transformer: Any,
    source_clean: Any,
    paired_hidden_states: Any,
    paired_rotary_embs: Any,
    source_tokens: int,
) -> bool:
    """Prove the pair prefix is Bernini's embedded source-token sequence.

    Bernini does not concatenate the raw ``[B,N,64]`` packed VAE latent with
    transformer hidden states.  Its official video-to-video path first calls
    ``transformer.patch_vae_latent(source, source_id=1.0)`` and concatenates
    the resulting ``[B,N,1536]`` patch-embedded tokens.  Recompute that exact
    operation here and require bit equality with the prefix observed by the
    hooked transformer call.
    """

    import torch

    patch = getattr(transformer, "patch_vae_latent", None)
    dtype = getattr(transformer, "dtype", None)
    if not callable(patch) or dtype is None:
        raise SourceKVRouteInferenceError(
            "cannot verify official source patch embedding"
        )
    if not isinstance(source_tokens, int) or isinstance(source_tokens, bool) or source_tokens <= 0:
        raise SourceKVRouteInferenceError("source token count differs")
    paired_shape = _shape(
        paired_hidden_states, label="paired transformer hidden states"
    )
    rotary_shape = _shape(paired_rotary_embs, label="paired transformer rotary")
    if len(paired_shape) != 3 or paired_shape[0] != 1 or paired_shape[1] < source_tokens:
        raise SourceKVRouteInferenceError("paired transformer hidden-state geometry differs")
    if (
        len(rotary_shape) != 4
        or rotary_shape[:2] != (1, 1)
        or rotary_shape[2] < source_tokens
    ):
        raise SourceKVRouteInferenceError("paired transformer rotary geometry differs")
    with torch.no_grad():
        patched = patch(source_clean.to(dtype=dtype), source_id=1.0)
    if not isinstance(patched, tuple) or len(patched) != 2:
        raise SourceKVRouteInferenceError("official source patch result differs")
    reference, reference_rotary = patched
    observed = paired_hidden_states[:, :source_tokens, :]
    observed_rotary = paired_rotary_embs[:, :, :source_tokens, :]
    if (
        tuple(reference.shape) != tuple(observed.shape)
        or reference.dtype != observed.dtype
        or reference.device != observed.device
        or not torch.equal(observed, reference)
        or tuple(reference_rotary.shape) != tuple(observed_rotary.shape)
        or reference_rotary.dtype != observed_rotary.dtype
        or reference_rotary.device != observed_rotary.device
        or not torch.equal(observed_rotary, reference_rotary)
    ):
        raise SourceKVRouteInferenceError(
            "paired source token/rotary prefix is not official patch(source clean, id=1)"
        )
    return True


def _validate_scope_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = dict(value)
    digest = manifest.pop("manifest_digest", None)
    if digest != route_scope.object_sha256(manifest):
        raise SourceKVRouteInferenceError("V9 scope manifest digest differs")
    lora = _require_mapping(value.get("lora"), label="scope lora")
    initialization = _require_mapping(
        value.get("initialization"), label="scope initialization"
    )
    route_scope.validate_target_module_names(lora.get("target_modules"))
    route_scope.validate_lora_hyperparameters(
        rank=lora.get("rank"),
        alpha=lora.get("alpha"),
        hidden_size=lora.get("hidden_size"),
        dropout=lora.get("dropout"),
        bias=lora.get("bias"),
    )
    route_scope.validate_fresh_initialization(initialization)
    if (
        value.get("schema_version") != route_scope.RECEIPT_MANIFEST_SCHEMA
        or value.get("method") != route_scope.METHOD_NAME
        or value.get("scope") != route_scope.SCOPE_NAME
        or lora.get("target_module_count") != 92
        or lora.get("adapter_tensor_count") != 184
        or lora.get("trainable_parameter_count") != 2_260_992
        or lora.get("target_modules_sha256")
        != route_scope.EXPECTED_TARGET_MODULES_SHA256
    ):
        raise SourceKVRouteInferenceError("V9 scope manifest is not exact92/184")
    return value


def validate_training_checkpoint_contract(
    *,
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    adapter_model_sha256: str,
    adapter_config_sha256: str,
    optimizer_checkpoint_sha256: str,
    expected_checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    """Accept only the completed fresh all30 exact40 V9 artifact."""

    serialized = _validate_adapter_config(adapter_config)
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    _require_sha256(digest, label="training receipt digest")
    if route_scope.object_sha256(candidate) != digest:
        raise SourceKVRouteInferenceError("training receipt digest differs")
    if (
        receipt.get("schema_version") != trainer.RECEIPT_SCHEMA
        or receipt.get("method") != trainer.METHOD_NAME
        or receipt.get("global_step") != 40
        or receipt.get("formal_exact40_complete") is not True
    ):
        raise SourceKVRouteInferenceError(
            "adapter is not one completed V9 exact40 checkpoint"
        )
    artifact = _require_mapping(
        receipt.get("artifact_validation"), label="artifact validation"
    )
    if (
        artifact.get("schema_version") != trainer.ARTIFACT_VALIDATION_SCHEMA
        or artifact.get("verified") is not True
        or artifact.get("status") != ADAPTER_READY_STATUS
        or artifact.get("adapter_tensor_file_roundtrip_verified") is not True
        or artifact.get("adapter_tensor_file_runtime_equality") is not True
        or artifact.get("torch_deserialize_verified") is not True
        or artifact.get("fresh_optimizer_load_state_dict_verified") is not True
        or artifact.get("optimizer_state_logical_equality_verified") is not True
        or artifact.get("runtime_adapter_loader_verified") is not False
        or artifact.get("fresh_base_peft_from_pretrained_verified") is not False
        or artifact.get("deployment_loader_claim_forbidden") is not True
        or artifact.get("adapter_tensor_count") != 184
        or artifact.get("trainable_parameter_count") != 2_260_992
        or artifact.get("state_parameter_count") != 184
        or artifact.get("state_step_values") != [40]
        or artifact.get("adapter_model_sha256") != adapter_model_sha256
        or artifact.get("adapter_config_sha256") != adapter_config_sha256
        or artifact.get("optimizer_checkpoint_sha256")
        != optimizer_checkpoint_sha256
    ):
        raise SourceKVRouteInferenceError(
            "adapter artifact is pending, partial, altered, or not strict-reload complete"
        )
    immutable = _require_mapping(
        receipt.get("immutable_contract"), label="immutable contract"
    )
    immutable_value = _require_mapping(
        immutable.get("value"), label="immutable contract value"
    )
    if immutable.get("digest") != route_scope.object_sha256(immutable_value):
        raise SourceKVRouteInferenceError("immutable contract digest differs")
    carrier = _require_mapping(
        immutable_value.get("carrier"), label="immutable carrier"
    )
    if (
        immutable_value.get("method") != trainer.METHOD_NAME
        or immutable_value.get("schema_version") != trainer.RECEIPT_SCHEMA
        or immutable_value.get("run_role") != "v9_main"
        or immutable_value.get("frames") != 81
        or immutable_value.get("latent_phases") != 21
        or immutable_value.get("max_steps") != 40
        or immutable_value.get("checkpoint_tree_sha256")
        != expected_checkpoint_tree_sha256
        or immutable_value.get("forward_order") != list(EXPECTED_BRANCH_ORDER)
        or immutable_value.get("forwards_per_candidate") != 6
        or immutable_value.get("training_diffusion_query") != "source(beta=0)"
        or immutable_value.get("paired_target_used_as_model_condition") is not False
        or immutable_value.get("inference_conditions")
        != ["source_video", "action_instruction"]
        or immutable_value.get("first_frame_anchor") is not False
        or immutable_value.get("target_clipping") is not False
        or immutable_value.get("target_energy_retention") != 1.0
        or immutable_value.get("resume_integrated") is not False
        or carrier.get("selection") != "all"
        or carrier.get("selected_blocks") != list(EXPECTED_BLOCKS)
        or carrier.get("selected_block_count") != 30
        or carrier.get("source_only") is not True
        or carrier.get("post_rope") is not True
    ):
        raise SourceKVRouteInferenceError("immutable V9 main contract differs")
    scope_manifest = _validate_scope_manifest(
        _require_mapping(
            immutable_value.get("lora_scope_manifest"),
            label="immutable scope manifest",
        )
    )
    receipt_adapter = _require_mapping(receipt.get("adapter"), label="adapter")
    if (
        receipt_adapter.get("scope_manifest") != scope_manifest
        or receipt_adapter.get("target_module_count") != 92
        or receipt_adapter.get("adapter_tensor_count") != 184
        or receipt_adapter.get("trainable_parameter_count") != 2_260_992
    ):
        raise SourceKVRouteInferenceError("receipt adapter declaration differs")
    step_audit = receipt.get("step_audit")
    if (
        not isinstance(step_audit, list)
        or len(step_audit) != 40
        or receipt.get("step_audit_sha256")
        != route_scope.object_sha256(step_audit)
    ):
        raise SourceKVRouteInferenceError("V9 step audit is incomplete or altered")
    try:
        exact40 = trainer.validate_exact40_step_audit(
            step_audit, block_selection="all"
        )
    except Exception as error:
        raise SourceKVRouteInferenceError(
            f"V9 exact40 engineering audit fails: {error}"
        ) from error
    if receipt.get("exact40_audit") != exact40:
        raise SourceKVRouteInferenceError("stored exact40 audit differs")
    dataset = _require_mapping(receipt.get("dataset"), label="dataset")
    input_integrity = _require_mapping(
        dataset.get("input_integrity"), label="dataset input integrity"
    )
    integrity_value = dict(input_integrity)
    integrity_digest = integrity_value.pop("audit_sha256", None)
    accesses = input_integrity.get("accesses")
    final_shards = input_integrity.get("final_accessed_shards")
    if (
        integrity_digest != route_scope.object_sha256(integrity_value)
        or input_integrity.get("validated") is not True
        or input_integrity.get("policy")
        != "pinned_index_hash_before_and_after_each_optimizer_read"
        or input_integrity.get("access_count") != 40
        or not isinstance(accesses, list)
        or len(accesses) != 40
        or input_integrity.get("accesses_sha256")
        != route_scope.object_sha256(accesses)
        or not isinstance(final_shards, list)
        or input_integrity.get("unique_accessed_shard_count")
        != len(final_shards)
        or input_integrity.get("dataset_summary_final_sha256")
        != trainer.PINNED_DATASET_SUMMARY_FILE_SHA256
        or input_integrity.get("dataset_index_final_sha256")
        != trainer.PINNED_DATASET_INDEX_SHA256
        or input_integrity.get("routing_final_sha256")
        != trainer.PINNED_ROUTING_SHA256
    ):
        raise SourceKVRouteInferenceError("training input-integrity audit differs")
    for index, (access, step) in enumerate(zip(accesses, step_audit)):
        if (
            not isinstance(access, Mapping)
            or access.get("access_ordinal") != index
            or access.get("row_index") != step.get("row_index")
            or access.get("hash_closed_read") is not True
            or access.get("cache_invalidated_before_read") is not True
            or access.get("before_read_sha256") != access.get("expected_sha256")
            or access.get("after_read_sha256") != access.get("expected_sha256")
            or step.get("input_shard_integrity") != access
        ):
            raise SourceKVRouteInferenceError(
                f"training input-integrity access {index} differs"
            )
    if any(
        not isinstance(row, Mapping)
        or row.get("expected_sha256") != row.get("final_sha256")
        for row in final_shards
    ):
        raise SourceKVRouteInferenceError("final accessed-shard hashes differ")
    if (
        receipt.get("paired_target_model_forward_access") is not False
        or receipt.get("external_mask_track_flow_pose_trajectory") is not False
        or receipt.get("first_frame_anchor") is not False
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
    ):
        raise SourceKVRouteInferenceError("V9 training safety declaration differs")
    transformers_version = receipt.get("transformers_version")
    if not isinstance(transformers_version, str) or not transformers_version:
        raise SourceKVRouteInferenceError("training Transformers version is missing")
    return {
        "receipt_digest": digest,
        "global_step": 40,
        "adapter_model_sha256": adapter_model_sha256,
        "serialized_target_modules": serialized,
        "target_modules": route_scope.canonical_target_modules(),
        "target_modules_sha256": route_scope.EXPECTED_TARGET_MODULES_SHA256,
        "scope_manifest_digest": scope_manifest["manifest_digest"],
        "checkpoint_parameter_digest": receipt_adapter[
            "checkpoint_parameter_digest"
        ],
        "initialization_digest": receipt_adapter["initialization_digest"],
        "transformers_version": transformers_version,
        "training_method_source_revision": immutable_value[
            "method_source_revision"
        ],
        "training_method_source_archive_sha256": immutable_value[
            "method_source_archive_sha256"
        ],
        "query_state_policy": immutable_value["query_state_policy"],
        "training_input_integrity_digest": integrity_digest,
    }


@dataclass(frozen=True)
class StagedSource:
    requested_path: Path
    staged_path: Path
    sha256: str
    byte_count: int
    source_device: int
    source_inode: int
    source_mtime_ns: int
    directory: Path


@dataclass(frozen=True)
class PlainFileSnapshot:
    label: str
    requested_path: Path
    staged_path: Path
    sha256: str
    byte_count: int
    source_device: int
    source_inode: int
    source_mtime_ns: int


@dataclass(frozen=True)
class StagedAdapterBundle:
    requested_bundle: legacy.AdapterBundle
    bundle: legacy.AdapterBundle
    requested_optimizer_path: Path
    optimizer_path: Path
    files: tuple[PlainFileSnapshot, ...]
    hashes: Mapping[str, str]
    directory: Path


def _hash_plain_file_nofollow(
    path: Path, *, label: str
) -> tuple[str, tuple[int, int, int, int]]:
    """Hash one stable regular-file identity through an ``O_NOFOLLOW`` fd."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourceKVRouteInferenceError(f"cannot open {label} without follow") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SourceKVRouteInferenceError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
        after = os.fstat(descriptor)
        try:
            pathname = path.lstat()
        except OSError as error:
            raise SourceKVRouteInferenceError(f"cannot restat {label}") from error
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (
            identity
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or identity
            != (
                pathname.st_dev,
                pathname.st_ino,
                pathname.st_size,
                pathname.st_mtime_ns,
            )
            or stat.S_ISLNK(pathname.st_mode)
            or copied != before.st_size
        ):
            raise SourceKVRouteInferenceError(f"{label} changed while being hashed")
        return digest.hexdigest(), tuple(int(item) for item in identity)
    finally:
        os.close(descriptor)


def _copy_plain_file_nofollow(
    source: Path, destination: Path, *, label: str
) -> PlainFileSnapshot:
    """Copy one immutable shared-path identity into a private plain file."""

    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    try:
        source_fd = os.open(source, source_flags)
    except OSError as error:
        raise SourceKVRouteInferenceError(f"cannot snapshot {label}") from error
    destination_fd: Optional[int] = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise SourceKVRouteInferenceError(f"{label} is not a regular file")
        destination_fd = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise SourceKVRouteInferenceError(
                        f"short write while staging {label}"
                    )
                view = view[written:]
            digest.update(chunk)
            copied += len(chunk)
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        pathname = source.lstat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (
            identity
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or identity
            != (
                pathname.st_dev,
                pathname.st_ino,
                pathname.st_size,
                pathname.st_mtime_ns,
            )
            or stat.S_ISLNK(pathname.st_mode)
            or copied != before.st_size
        ):
            raise SourceKVRouteInferenceError(f"{label} changed while being staged")
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)
    staged_sha, _ = _hash_plain_file_nofollow(
        destination, label=f"staged {label}"
    )
    if staged_sha != digest.hexdigest():
        raise SourceKVRouteInferenceError(f"staged {label} hash differs")
    return PlainFileSnapshot(
        label=label,
        requested_path=source,
        staged_path=destination,
        sha256=staged_sha,
        byte_count=copied,
        source_device=int(before.st_dev),
        source_inode=int(before.st_ino),
        source_mtime_ns=int(before.st_mtime_ns),
    )


def stage_adapter_snapshot(value: str | Path) -> StagedAdapterBundle:
    """Privately snapshot all four V9 files and load only those snapshots."""

    try:
        requested = legacy.resolve_adapter_bundle(value)
    except legacy.InferenceContractError as error:
        raise SourceKVRouteInferenceError(str(error)) from error
    requested_optimizer = _plain_file(
        requested.checkpoint_root / "optimizer.pt", label="optimizer checkpoint"
    )
    directory = Path(tempfile.mkdtemp(prefix="bernini-v9-adapter-"))
    os.chmod(directory, 0o700)
    root = directory / "checkpoint"
    adapter_dir = root / "adapter"
    adapter_dir.mkdir(parents=True, mode=0o700)
    specs = (
        ("adapter_config", requested.adapter_config_path, adapter_dir / "adapter_config.json"),
        ("adapter_model", requested.adapter_model_path, adapter_dir / "adapter_model.safetensors"),
        ("training_receipt", requested.training_receipt_path, root / "receipt.json"),
        ("optimizer_checkpoint", requested_optimizer, root / "optimizer.pt"),
    )
    try:
        files = tuple(
            _copy_plain_file_nofollow(source, destination, label=label)
            for label, source, destination in specs
        )
        staged_bundle = legacy.AdapterBundle(
            checkpoint_root=root,
            adapter_dir=adapter_dir,
            adapter_config_path=adapter_dir / "adapter_config.json",
            adapter_model_path=adapter_dir / "adapter_model.safetensors",
            training_receipt_path=root / "receipt.json",
        )
        result = StagedAdapterBundle(
            requested_bundle=requested,
            bundle=staged_bundle,
            requested_optimizer_path=requested_optimizer,
            optimizer_path=root / "optimizer.pt",
            files=files,
            hashes={item.label: item.sha256 for item in files},
            directory=directory,
        )
        verify_adapter_snapshot(result)
        return result
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def verify_adapter_snapshot(staged: StagedAdapterBundle) -> None:
    """Reprove both shared and private identities without following symlinks."""

    for item in staged.files:
        shared_sha, shared_identity = _hash_plain_file_nofollow(
            item.requested_path, label=f"shared {item.label}"
        )
        private_sha, _ = _hash_plain_file_nofollow(
            item.staged_path, label=f"private {item.label}"
        )
        expected_identity = (
            item.source_device,
            item.source_inode,
            item.byte_count,
            item.source_mtime_ns,
        )
        if (
            shared_sha != item.sha256
            or private_sha != item.sha256
            or shared_identity != expected_identity
        ):
            raise SourceKVRouteInferenceError(
                f"shared/private {item.label} snapshot identity differs"
            )


def cleanup_staged_adapter(staged: StagedAdapterBundle) -> None:
    shutil.rmtree(staged.directory, ignore_errors=True)


def validate_runtime_checkpoint_manifest(
    checkpoint: Path,
    manifest_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Dynamically verify the complete non-cache base-checkpoint identity."""

    if not checkpoint.is_absolute():
        raise SourceKVRouteInferenceError("checkpoint content root must be absolute")
    try:
        root_stat = checkpoint.lstat()
    except OSError as error:
        raise SourceKVRouteInferenceError("cannot stat checkpoint content root") from error
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise SourceKVRouteInferenceError(
            "checkpoint content root must be a non-symlink directory"
        )
    manifest = manifest_path or (
        METHOD_ROOT / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"
    )
    manifest = _plain_file(manifest, label="checkpoint content manifest")
    manifest_sha, _ = _hash_plain_file_nofollow(
        manifest, label="checkpoint content manifest"
    )
    if manifest_sha != trainer.CHECKPOINT_CONTENT_MANIFEST_SHA256:
        raise SourceKVRouteInferenceError("checkpoint content manifest SHA differs")
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise SourceKVRouteInferenceError("cannot read checkpoint content manifest") from error
    if len(lines) != trainer.CHECKPOINT_CONTENT_FILE_COUNT:
        raise SourceKVRouteInferenceError("checkpoint manifest file count differs")
    expected: dict[str, str] = {}
    pattern = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            raise SourceKVRouteInferenceError("checkpoint manifest syntax differs")
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise SourceKVRouteInferenceError("checkpoint manifest path is unsafe")
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in expected:
            raise SourceKVRouteInferenceError("checkpoint manifest path inventory differs")
        expected[normalized] = digest

    actual: set[str] = set()
    for path in checkpoint.rglob("*"):
        relative = path.relative_to(checkpoint)
        if ".cache" in relative.parts:
            continue
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise SourceKVRouteInferenceError("cannot stat checkpoint content") from error
        if stat.S_ISLNK(mode):
            raise SourceKVRouteInferenceError("checkpoint contains a symlink")
        if stat.S_ISREG(mode):
            actual.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise SourceKVRouteInferenceError(
                "checkpoint contains a non-regular filesystem entry"
            )
    if actual != set(expected):
        raise SourceKVRouteInferenceError("checkpoint file set differs from manifest")
    entries = []
    for relative in sorted(expected):
        digest, _ = _hash_plain_file_nofollow(
            checkpoint / relative, label=f"checkpoint file {relative}"
        )
        if digest != expected[relative]:
            raise SourceKVRouteInferenceError(
                f"checkpoint content hash differs: {relative}"
            )
        entries.append({"path": relative, "sha256": digest})
    identity = {
        "manifest_sha256": manifest_sha,
        "verified_file_count": len(entries),
        "entries": entries,
    }
    return {
        "validated": True,
        "manifest_sha256": manifest_sha,
        "verified_file_count": len(entries),
        "every_non_cache_file_verified": True,
        "entries_digest": route_scope.object_sha256(entries),
        "identity_digest": route_scope.object_sha256(identity),
    }


def tensor_sha256(value: Any, *, label: str) -> str:
    """Hash exact tensor metadata and bytes, including BF16 without conversion."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise SourceKVRouteInferenceError(f"{label} must be a tensor")
    tensor = value.detach().contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise SourceKVRouteInferenceError(f"{label} is non-finite")
    metadata = {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
    }
    digest = hashlib.sha256(route_scope.canonical_json_bytes(metadata))
    digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def stage_source_snapshot(value: str | Path) -> StagedSource:
    """Copy through one no-follow fd and bind the immutable staged bytes."""

    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise SourceKVRouteInferenceError("source video must be absolute")
    requested = _plain_file(requested.resolve(strict=True), label="source video")
    directory = Path(tempfile.mkdtemp(prefix="bernini-v9-source-"))
    os.chmod(directory, 0o700)
    staged = directory / "source.mp4"
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_fd = os.open(requested, source_flags)
    destination_fd: Optional[int] = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise SourceKVRouteInferenceError("opened source is not a regular file")
        destination_fd = os.open(
            staged,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise SourceKVRouteInferenceError("short write while staging source")
                view = view[written:]
            digest.update(chunk)
            copied += len(chunk)
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        pathname = requested.lstat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or identity != (
            pathname.st_dev,
            pathname.st_ino,
            pathname.st_size,
            pathname.st_mtime_ns,
        ):
            raise SourceKVRouteInferenceError("source changed while being staged")
        if copied != before.st_size:
            raise SourceKVRouteInferenceError("staged source byte count differs")
    except Exception:
        if destination_fd is not None:
            os.close(destination_fd)
            destination_fd = None
        os.close(source_fd)
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        try:
            os.close(source_fd)
        except OSError:
            pass
    staged_hash = _file_sha256(staged)
    if staged_hash != digest.hexdigest():
        shutil.rmtree(directory, ignore_errors=True)
        raise SourceKVRouteInferenceError("staged source hash differs after close")
    return StagedSource(
        requested_path=requested,
        staged_path=staged,
        sha256=staged_hash,
        byte_count=copied,
        source_device=int(before.st_dev),
        source_inode=int(before.st_ino),
        source_mtime_ns=int(before.st_mtime_ns),
        directory=directory,
    )


def cleanup_staged_source(staged: StagedSource) -> None:
    shutil.rmtree(staged.directory, ignore_errors=True)


def _lora_layers(adapter_controller: Any) -> list[tuple[str, Any]]:
    result = [
        (name, module)
        for name, module in adapter_controller.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    if len(result) != route_scope.EXPECTED_TARGET_MODULE_COUNT:
        raise SourceKVRouteInferenceError(
            f"runtime active LoRA layer count is {len(result)}, expected 92"
        )
    return result


def _adapter_disabled_state(layers: Sequence[tuple[str, Any]]) -> bool:
    values = [bool(getattr(module, "disable_adapters", False)) for _, module in layers]
    if len(set(values)) != 1:
        raise SourceKVRouteInferenceError("LoRA layers disagree on enabled state")
    return values[0]


@contextmanager
def _adapter_disabled(adapter_controller: Any) -> Iterator[None]:
    layers = _lora_layers(adapter_controller)
    if _adapter_disabled_state(layers):
        raise SourceKVRouteInferenceError("adapter is unexpectedly disabled")
    context = getattr(adapter_controller, "disable_adapter", None)
    if not callable(context):
        raise SourceKVRouteInferenceError("PEFT controller lacks disable_adapter")
    with context():
        if not _adapter_disabled_state(layers):
            raise SourceKVRouteInferenceError("adapter did not disable for frozen branch")
        yield
    if _adapter_disabled_state(layers):
        raise SourceKVRouteInferenceError("adapter did not re-enable after frozen branch")


def _strict_load_v9_adapter(
    *, base_model: Any, bundle: legacy.AdapterBundle
) -> tuple[Any, Any, int, Mapping[str, Any]]:
    import torch
    from peft import LoraConfig, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file

    inventory = dict(base_model.named_modules())
    targets = route_scope.validate_runtime_target_modules(inventory)
    config = LoraConfig.from_pretrained(str(bundle.adapter_dir), local_files_only=True)
    config.target_modules = set(targets)
    model = PeftModel.from_pretrained(
        base_model,
        str(bundle.adapter_dir),
        is_trainable=False,
        config=config,
        local_files_only=True,
    )
    saved = load_file(str(bundle.adapter_model_path), device="cpu")
    loaded = get_peft_model_state_dict(model, adapter_name="default")
    saved_validation = route_scope.validate_adapter_state(saved)
    loaded_validation = route_scope.validate_adapter_state(loaded)
    if saved_validation != loaded_validation or any(
        not torch.equal(saved[key], loaded[key].detach().cpu()) for key in saved
    ):
        raise SourceKVRouteInferenceError("strict V9 tensor reload differs")
    model.requires_grad_(False)
    model.eval()
    layers = _lora_layers(model)
    if _adapter_disabled_state(layers):
        raise SourceKVRouteInferenceError("V9 adapter is not active after reload")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SourceKVRouteInferenceError("inference model retains trainable parameters")
    return model, model.get_base_model(), len(saved), saved_validation


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.shape)
    except Exception as error:
        raise SourceKVRouteInferenceError(f"{label} has no concrete shape") from error
    return result


def _metadata_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SourceKVRouteInferenceError(f"{label} must be a sequence")
    try:
        return tuple(int(item) for item in value)
    except Exception as error:
        raise SourceKVRouteInferenceError(f"{label} contains a non-integer") from error


def _canonical_timestep_token(value: Any, *, step_index: int) -> str:
    numeric = tri._coerce_scalar(value, label="timestep")
    return f"step-{step_index}:float64-{numeric.hex()}"


@dataclass(frozen=True)
class DeploymentStepRecord:
    step_index: int
    timestep: float
    sigma: float
    model_id: str
    source_tokens: int
    pair_tokens: int
    forward_order: tuple[str, ...]
    capture_forwards: int
    replay_forwards: int
    frozen_replay_forwards: int
    adapted_replay_forwards: int
    original_scheduler_calls: int
    official_adapted_action_exact_parity: bool
    official_adapted_action_parity_max_abs: float
    quotient_rms: float
    frozen_quotient_rms: float
    executed_clean_rms: float
    deployed_velocity_rms: float
    phase0_quotient_exact_zero: bool
    source_phase0_exact_preservation: bool
    target_energy_retention: float
    target_clipped_fraction: float
    sigma_strictly_positive: bool


@dataclass
class DeploymentTrace:
    records: list[DeploymentStepRecord] = field(default_factory=list)
    sample_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        steps = []
        for record in self.records:
            value = asdict(record)
            value["forward_order"] = list(record.forward_order)
            steps.append(value)
        return {
            "sample_calls": self.sample_calls,
            "step_count": len(self.records),
            "steps": steps,
        }


@dataclass
class _CapturedBranch:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    bound: dict[str, Any]
    full_prediction: Any
    target_prediction: Any


@dataclass
class _ActiveSample:
    action_prompt: Any
    negative_prompt: Any
    apg: tri.APGParameters
    momenta: Mapping[str, Any]
    completed_steps: int = 0
    pending_negative: Optional[_CapturedBranch] = None
    branch_targets: dict[str, Any] = field(default_factory=dict)
    current_source_tokens: Optional[int] = None
    current_pair_tokens: Optional[int] = None
    current_timestep_token: Optional[str] = None


class InstalledSourceKVRouteDeployment:
    """One reversible five-replay adapter/scheduler integration."""

    def __init__(
        self,
        renderer_or_diffusion: Any,
        *,
        adapter_controller: Any,
        cache_bank: replay.SourceKVCacheBank,
        noop_prompt_embeds: Any,
        source_clean: Any,
        latent_shape: Sequence[int],
        rank: int,
        ulysses_size: int,
        expected_source_tokens: int,
        bernini_commit: str,
        wan_diffusion_path: Path,
    ) -> None:
        tri.validate_runtime_source_identity(
            bernini_commit=bernini_commit,
            wan_diffusion_path=wan_diffusion_path,
        )
        self.diffusion = tri.resolve_diffusion_core(renderer_or_diffusion)
        self.scheduler = self.diffusion.scheduler
        self.adapter_controller = adapter_controller
        self.cache_bank = cache_bank
        self.noop_prompt_embeds = noop_prompt_embeds
        self.source_clean = source_clean
        self.layout = tri.PackedLatentLayout.from_spatial_shape(latent_shape)
        self.rank = int(rank)
        self.ulysses_size = int(ulysses_size)
        self.expected_source_tokens = int(expected_source_tokens)
        self.trace = DeploymentTrace()
        self.restored = False
        self.source_prefix_verified = False
        self._active: Optional[_ActiveSample] = None
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._original_sample = self.diffusion.sample
        self._original_shared_step = self.diffusion.shared_step
        self._original_scheduler_step = self.scheduler.step
        if (
            self.ulysses_size != EXPECTED_ULYSSES_SIZE
            or not 0 <= self.rank < self.ulysses_size
            or self.expected_source_tokens != self.layout.tokens
            or tuple(int(item) for item in source_clean.shape)
            != (
                self.layout.batch,
                self.layout.channels,
                self.layout.frames,
                self.layout.height,
                self.layout.width,
            )
        ):
            raise SourceKVRouteInferenceError("deployment rank/source geometry differs")
        if self.layout.frames != EXPECTED_PHASES or self.layout.batch != 1:
            raise SourceKVRouteInferenceError("deployment requires one 21-phase source")
        if self.cache_bank.selected_block_indices != EXPECTED_BLOCKS:
            raise SourceKVRouteInferenceError("deployment requires all30 carrier blocks")
        if _shape(noop_prompt_embeds, label="no-op embeddings")[:1] != (1,):
            raise SourceKVRouteInferenceError("no-op embeddings must have batch one")
        if getattr(self.diffusion, "transformer_2", None) is not None:
            raise SourceKVRouteInferenceError("V9 deployment supports only 1.3B expert")
        if getattr(self.diffusion, "use_unipc", None) is not True:
            raise SourceKVRouteInferenceError("V9 deployment requires official UniPC")
        tri._validate_scheduler_contract(
            self.scheduler, expected_flow_shift=legacy.FLOW_SHIFT
        )
        for owner, name in (
            (self.diffusion, "sample"),
            (self.diffusion, "shared_step"),
            (self.scheduler, "step"),
        ):
            if name in vars(owner):
                raise SourceKVRouteInferenceError(
                    f"refusing to stack on instance override {name}"
                )
        _lora_layers(adapter_controller)

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        instance = vars(owner)
        had_instance = name in instance
        previous = instance.get(name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous))

    def install(self) -> None:
        if self._patches:
            raise SourceKVRouteInferenceError("deployment hook already installed")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, scheduler_wrapper):
            setattr(wrapper, "_bernini_source_kv_route_v9", self)
        try:
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
            self._set_patch(self.diffusion, "sample", sample_wrapper)
        except Exception:
            self.restore()
            raise
        self.restored = False

    def restore(self) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had_instance, previous = self._patches.pop()
            try:
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except Exception as error:
                errors.append(error)
        self._active = None
        self.restored = not errors
        if errors:
            raise SourceKVRouteInferenceError(
                f"failed to restore {len(errors)} deployment hook(s)"
            ) from errors[0]

    def _clear_bank(self) -> None:
        if self.cache_bank.identity is not None:
            self.cache_bank.clear()

    def _validate_sample(self, values: Mapping[str, Any]) -> tri.APGParameters:
        if (
            values.get("guidance_mode") != legacy.GUIDANCE_MODE
            or int(values.get("num_frames")) != EXPECTED_FRAMES
            or int(values.get("num_inference_steps")) != EXPECTED_STEPS
            or int(values.get("seed")) != EXPECTED_SEED
            or not math.isclose(
                float(values.get("flow_shift")),
                legacy.FLOW_SHIFT,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or values.get("image_vae_latents") is not None
            or values.get("multi_image_vae_latents") is not None
        ):
            raise SourceKVRouteInferenceError("official sample contract differs")
        videos = values.get("multi_video_vae_latents")
        if not isinstance(videos, (list, tuple)) or len(videos) != 1:
            raise SourceKVRouteInferenceError("exactly one source video is required")
        if videos[0] is not self.source_clean:
            raise SourceKVRouteInferenceError("sample source is not the staged source latent")
        norm = values.get("norm_threshold")
        threshold = norm[0] if isinstance(norm, (list, tuple)) else norm
        parameters = tri.APGParameters(
            guidance_scale=tri._coerce_scalar(values.get("omega_txt"), label="omega_txt"),
            omega_scale=tri._coerce_scalar(values.get("omega_scale"), label="omega_scale"),
            scale_transformer_2=False,
            eta=tri._coerce_scalar(values.get("eta"), label="eta"),
            norm_threshold=tri._coerce_scalar(threshold, label="norm_threshold"),
            momentum=tri._coerce_scalar(values.get("momentum"), label="momentum"),
        )
        if (
            parameters.guidance_scale != legacy.OMEGA_TEXT
            or parameters.eta != legacy.ETA
            or parameters.norm_threshold != legacy.NORM_THRESHOLD[0]
            or parameters.momentum != 0.0
        ):
            raise SourceKVRouteInferenceError("V9 APG parameters differ")
        return parameters

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.trace.sample_calls:
            raise SourceKVRouteInferenceError("exactly one sample call is permitted")
        values = tri._bind_call(self._original_sample, args, kwargs)
        parameters = self._validate_sample(values)
        state = _ActiveSample(
            action_prompt=values["prompt_embeds"],
            negative_prompt=values["uncond_prompt_embeds"],
            apg=parameters,
            momenta={
                name: tri._MomentumBuffer(parameters.momentum, branch=name)
                for name in (
                    "frozen_noop",
                    "frozen_action",
                    "adapted_noop",
                    "adapted_action",
                )
            },
        )
        self._active = state
        try:
            result = self._original_sample(*args, **kwargs)
            if (
                state.completed_steps != EXPECTED_STEPS
                or state.pending_negative is not None
                or state.branch_targets
                or self.cache_bank.identity is not None
            ):
                raise SourceKVRouteInferenceError(
                    "sample returned with an incomplete deployment step"
                )
            if self.source_prefix_verified is not True:
                raise SourceKVRouteInferenceError(
                    "official source patch prefix was never verified"
                )
            if any(buffer.update_count != EXPECTED_STEPS for buffer in state.momenta.values()):
                raise SourceKVRouteInferenceError("APG branch momentum counts differ")
            self.trace.sample_calls = 1
            return result
        finally:
            self._active = None
            self._clear_bank()

    def _pair_geometry(self, values: Mapping[str, Any]) -> tuple[int, int]:
        import torch

        shape = _shape(values.get("noisy_latents"), label="paired noisy latents")
        if len(shape) != 3 or shape[0] != 1 or shape[1] % 2:
            raise SourceKVRouteInferenceError("paired state must be [1,2N,D]")
        pair_tokens = shape[1]
        source_tokens = pair_tokens // 2
        rotary = _shape(values.get("rotary_embs"), label="paired rotary")
        lengths = _metadata_tuple(
            values.get("batch_vae_seqlen"), label="batch_vae_seqlen"
        )
        if (
            source_tokens != self.expected_source_tokens
            or pair_tokens != 2 * self.expected_source_tokens
            or lengths != (pair_tokens,)
            or len(rotary) != 4
            or rotary[:3] != (1, 1, pair_tokens)
        ):
            raise SourceKVRouteInferenceError("paired source/query geometry differs")
        if not self.source_prefix_verified:
            transformer = getattr(self.diffusion, "transformer", None)
            self.source_prefix_verified = verify_official_source_prefix(
                transformer=transformer,
                source_clean=self.source_clean,
                paired_hidden_states=values["noisy_latents"],
                paired_rotary_embs=values["rotary_embs"],
                source_tokens=source_tokens,
            )
        return source_tokens, pair_tokens

    def _target_prediction(self, value: Any, *, branch: str) -> Any:
        import torch

        shape = _shape(value, label=f"{branch} prediction")
        if (
            len(shape) != 3
            or shape[0] != 1
            or shape[1] < self.layout.tokens
            or shape[2] != self.layout.packed_channels
        ):
            raise SourceKVRouteInferenceError(f"{branch} prediction shape differs")
        target = value[:, -self.layout.tokens :, :]
        if _shape(target, label=f"{branch} target") != self.layout.packed_shape:
            raise SourceKVRouteInferenceError(f"{branch} target selection differs")
        if (
            target.dtype != torch.bfloat16
            or bool(target.requires_grad)
            or not bool(torch.isfinite(target).all())
        ):
            raise SourceKVRouteInferenceError(
                f"{branch} target must be finite no-grad native BF16"
            )
        return target

    def _invoke(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        *,
        mode: str,
        branch_tag: str,
        step_index: int,
        timestep_token: str,
    ) -> Any:
        with replay.source_kv_replay_invocation(
            self.cache_bank,
            mode=mode,
            branch_tag=branch_tag,
            generation=0,
            step_index=step_index,
            timestep_token=timestep_token,
            rank=self.rank,
            ulysses_size=self.ulysses_size,
        ):
            return self._original_shared_step(*args, **dict(kwargs))

    def _replace_prompt(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        prompt: Any,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        call_args, call_kwargs = tri._replace_argument(
            self._original_shared_step,
            args,
            kwargs,
            name="cond_embeds",
            value=prompt,
        )
        length = _shape(prompt, label="branch prompt embeddings")[1]
        return tri._replace_argument(
            self._original_shared_step,
            call_args,
            call_kwargs,
            name="batch_text_seqlen",
            value=[length],
        )

    def _capture_carrier(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        *,
        source_tokens: int,
        step_index: int,
        timestep_token: str,
    ) -> Any:
        values = tri._bind_call(self._original_shared_step, args, kwargs)
        call_args, call_kwargs = tuple(args), dict(kwargs)
        replacements = {
            "noisy_latents": values["noisy_latents"][:, :source_tokens, :],
            "rotary_embs": values["rotary_embs"][:, :, :source_tokens, :],
            "batch_vae_seqlen": [source_tokens],
        }
        for name, value in replacements.items():
            call_args, call_kwargs = tri._replace_argument(
                self._original_shared_step,
                call_args,
                call_kwargs,
                name=name,
                value=value,
            )
        call_args, call_kwargs = self._replace_prompt(
            call_args, call_kwargs, self.noop_prompt_embeds
        )
        return self._invoke(
            call_args,
            call_kwargs,
            mode=replay.CAPTURE_MODE,
            branch_tag=replay.CAPTURE_BRANCH_TAG,
            step_index=step_index,
            timestep_token=timestep_token,
        )

    def _assert_state_identity(
        self, left: Mapping[str, Any], right: Mapping[str, Any]
    ) -> None:
        if str(left.get("model_id")) != str(right.get("model_id")):
            raise SourceKVRouteInferenceError("negative/action model id differs")
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            if left.get(name) is not right.get(name):
                raise SourceKVRouteInferenceError(
                    f"five replay branches do not share exact {name} object"
                )
        if _metadata_tuple(
            left.get("batch_vae_seqlen"), label="left sequence length"
        ) != _metadata_tuple(
            right.get("batch_vae_seqlen"), label="right sequence length"
        ):
            raise SourceKVRouteInferenceError("branch sequence lengths differ")

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        state = self._active
        if state is None:
            raise SourceKVRouteInferenceError("shared_step outside validated sample")
        values = tri._bind_call(self._original_shared_step, args, kwargs)
        if str(values.get("model_id")) != "transformer_1":
            raise SourceKVRouteInferenceError("non-1.3B transformer route observed")
        source_tokens, pair_tokens = self._pair_geometry(values)
        step_index = state.completed_steps
        timestep_token = _canonical_timestep_token(
            values.get("timesteps"), step_index=step_index
        )

        if state.pending_negative is None:
            if values.get("cond_embeds") is not state.negative_prompt:
                raise SourceKVRouteInferenceError("first official branch is not negative")
            with torch.no_grad(), _adapter_disabled(self.adapter_controller):
                carrier_output = self._capture_carrier(
                    args,
                    kwargs,
                    source_tokens=source_tokens,
                    step_index=step_index,
                    timestep_token=timestep_token,
                )
                frozen_negative = self._invoke(
                    args,
                    kwargs,
                    mode=replay.REPLAY_MODE,
                    branch_tag="frozen_negative",
                    step_index=step_index,
                    timestep_token=timestep_token,
                )
            if (
                _shape(carrier_output, label="carrier output")[1] != source_tokens
                or carrier_output.dtype != torch.bfloat16
                or bool(getattr(carrier_output, "requires_grad", False))
                or not bool(torch.isfinite(carrier_output).all())
            ):
                self._clear_bank()
                raise SourceKVRouteInferenceError("source-only carrier output differs")
            state.pending_negative = _CapturedBranch(
                args=tuple(args),
                kwargs=dict(kwargs),
                bound=dict(values),
                full_prediction=frozen_negative,
                target_prediction=self._target_prediction(
                    frozen_negative, branch="frozen_negative"
                ),
            )
            state.current_source_tokens = source_tokens
            state.current_pair_tokens = pair_tokens
            state.current_timestep_token = timestep_token
            state.branch_targets = {
                "frozen_negative": state.pending_negative.target_prediction
            }
            return frozen_negative

        negative = state.pending_negative
        if values.get("cond_embeds") is not state.action_prompt:
            self._clear_bank()
            raise SourceKVRouteInferenceError("second official branch is not action")
        self._assert_state_identity(negative.bound, values)
        if (
            source_tokens != state.current_source_tokens
            or pair_tokens != state.current_pair_tokens
            or timestep_token != state.current_timestep_token
        ):
            self._clear_bank()
            raise SourceKVRouteInferenceError("negative/action step identity differs")
        noop_args, noop_kwargs = self._replace_prompt(
            args, kwargs, self.noop_prompt_embeds
        )
        with torch.no_grad(), _adapter_disabled(self.adapter_controller):
            frozen_noop = self._invoke(
                noop_args,
                noop_kwargs,
                mode=replay.REPLAY_MODE,
                branch_tag="frozen_noop",
                step_index=step_index,
                timestep_token=timestep_token,
            )
            frozen_action = self._invoke(
                args,
                kwargs,
                mode=replay.REPLAY_MODE,
                branch_tag="frozen_action",
                step_index=step_index,
                timestep_token=timestep_token,
            )
        if _adapter_disabled_state(_lora_layers(self.adapter_controller)):
            self._clear_bank()
            raise SourceKVRouteInferenceError("adapter disabled before adapted branches")
        with torch.no_grad():
            adapted_noop = self._invoke(
                noop_args,
                noop_kwargs,
                mode=replay.REPLAY_MODE,
                branch_tag="adapted_noop",
                step_index=step_index,
                timestep_token=timestep_token,
            )
            adapted_action = self._invoke(
                args,
                kwargs,
                mode=replay.REPLAY_MODE,
                branch_tag="adapted_action",
                step_index=step_index,
                timestep_token=timestep_token,
            )
        state.branch_targets.update(
            {
                name: self._target_prediction(value, branch=name)
                for name, value in (
                    ("frozen_noop", frozen_noop),
                    ("frozen_action", frozen_action),
                    ("adapted_noop", adapted_noop),
                    ("adapted_action", adapted_action),
                )
            }
        )
        if tuple(state.branch_targets) != REPLAY_BRANCH_ORDER:
            self._clear_bank()
            raise SourceKVRouteInferenceError("five replay branch order differs")
        return adapted_action

    @staticmethod
    def _rms(value: Any) -> float:
        return tri._coerce_scalar(
            value.float().square().mean().sqrt(), label="tensor RMS"
        )

    def _guided_clean_fields(
        self,
        *,
        state: _ActiveSample,
        sample: Any,
        sigma: Any,
    ) -> tuple[dict[str, Any], Any]:
        negative_v = tri._packed_to_spatial(
            state.branch_targets["frozen_negative"], self.layout
        )
        negative_clean = tri.pinned_raw_condition_clean(sample, negative_v, sigma)
        guided: dict[str, Any] = {}
        for name in (
            "frozen_noop",
            "frozen_action",
            "adapted_noop",
            "adapted_action",
        ):
            velocity = tri._packed_to_spatial(state.branch_targets[name], self.layout)
            condition_clean = tri.pinned_raw_condition_clean(sample, velocity, sigma)
            guided[name] = tri._normalized_guidance(
                condition_clean,
                negative_clean,
                state.apg.guidance_scale_for("transformer_1"),
                state.momenta[name],
                state.apg.eta,
                state.apg.norm_threshold,
            )
        return guided, negative_clean

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        state = self._active
        if state is None or state.pending_negative is None:
            raise SourceKVRouteInferenceError("scheduler.step lacks a replay cell")
        if tuple(state.branch_targets) != REPLAY_BRANCH_ORDER:
            raise SourceKVRouteInferenceError("scheduler.step arrived before five replays")
        official = tri._extract_argument(
            args, kwargs, index=0, name="model_output"
        )
        timestep = tri._extract_argument(args, kwargs, index=1, name="timestep")
        sample_packed = tri._extract_argument(args, kwargs, index=2, name="sample")
        step_index, sigma, sigma_float = tri._resolve_sigma(self.scheduler, timestep)
        if sigma_float <= 0.0:
            raise SourceKVRouteInferenceError("deployment sigma must be strictly positive")
        if step_index != state.completed_steps:
            raise SourceKVRouteInferenceError("scheduler and replay step indices differ")
        if _shape(sample_packed, label="scheduler sample") != self.layout.packed_shape:
            raise SourceKVRouteInferenceError("scheduler sample geometry differs")
        sample = tri._packed_to_spatial(sample_packed, self.layout)
        guided, _ = self._guided_clean_fields(state=state, sample=sample, sigma=sigma)

        # Prove the branch returned to official Bernini really is adapted action.
        official_rebuilt = tri._spatial_to_packed(
            (sample - guided["adapted_action"]) / sigma,
            self.layout,
        ).to(device=official.device, dtype=official.dtype)
        parity_error = official_rebuilt.float() - official.float()
        parity_max = tri._coerce_scalar(
            parity_error.abs().max(), label="official adapted-action parity max"
        )
        if not torch.equal(official_rebuilt, official):
            raise SourceKVRouteInferenceError(
                "official APG output is not the exact adapted-action branch"
            )

        # Sole deployment operator.  Frozen fields remain diagnostics only.
        executed_clean, quotient = compute_deployment_clean_field(
            source_clean=self.source_clean,
            adapted_action=guided["adapted_action"],
            adapted_noop=guided["adapted_noop"],
        )
        frozen_delta = guided["frozen_action"].float() - guided[
            "frozen_noop"
        ].float()
        frozen_quotient = frozen_delta - frozen_delta[:, :, :1]
        source = self.source_clean.float()
        phase0_zero = bool(torch.equal(quotient[:, :, :1], torch.zeros_like(quotient[:, :, :1])))
        source_phase0 = bool(torch.equal(executed_clean[:, :, :1], source[:, :, :1]))
        if not phase0_zero or not source_phase0 or not bool(torch.isfinite(executed_clean).all()):
            raise SourceKVRouteInferenceError("Q0/source phase-zero invariant differs")
        deployed_velocity = tri._spatial_to_packed(
            (sample - executed_clean) / sigma, self.layout
        ).to(device=official.device, dtype=official.dtype)
        if not bool(torch.isfinite(deployed_velocity).all()):
            raise SourceKVRouteInferenceError("deployed velocity is non-finite")
        call_args, call_kwargs = tri._replace_argument(
            self._original_scheduler_step,
            args,
            kwargs,
            name="model_output",
            value=deployed_velocity,
        )
        try:
            result = self._original_scheduler_step(*call_args, **call_kwargs)
        except Exception:
            self._clear_bank()
            raise
        self.trace.records.append(
            DeploymentStepRecord(
                step_index=step_index,
                timestep=tri._coerce_scalar(timestep, label="timestep"),
                sigma=sigma_float,
                model_id="transformer_1",
                source_tokens=int(state.current_source_tokens or 0),
                pair_tokens=int(state.current_pair_tokens or 0),
                forward_order=EXPECTED_BRANCH_ORDER,
                capture_forwards=1,
                replay_forwards=5,
                frozen_replay_forwards=3,
                adapted_replay_forwards=2,
                original_scheduler_calls=1,
                official_adapted_action_exact_parity=True,
                official_adapted_action_parity_max_abs=parity_max,
                quotient_rms=self._rms(quotient),
                frozen_quotient_rms=self._rms(frozen_quotient),
                executed_clean_rms=self._rms(executed_clean),
                deployed_velocity_rms=self._rms(deployed_velocity),
                phase0_quotient_exact_zero=phase0_zero,
                source_phase0_exact_preservation=source_phase0,
                target_energy_retention=1.0,
                target_clipped_fraction=0.0,
                sigma_strictly_positive=True,
            )
        )
        self._clear_bank()
        state.pending_negative = None
        state.branch_targets = {}
        state.current_source_tokens = None
        state.current_pair_tokens = None
        state.current_timestep_token = None
        state.completed_steps += 1
        return result


@contextmanager
def source_kv_route_deployment_hook(
    renderer_or_diffusion: Any,
    *,
    adapter_controller: Any,
    cache_bank: replay.SourceKVCacheBank,
    noop_prompt_embeds: Any,
    source_clean: Any,
    latent_shape: Sequence[int],
    rank: int,
    ulysses_size: int,
    expected_source_tokens: int,
    bernini_commit: str,
    wan_diffusion_path: Path,
) -> Iterator[InstalledSourceKVRouteDeployment]:
    installed = InstalledSourceKVRouteDeployment(
        renderer_or_diffusion,
        adapter_controller=adapter_controller,
        cache_bank=cache_bank,
        noop_prompt_embeds=noop_prompt_embeds,
        source_clean=source_clean,
        latent_shape=latent_shape,
        rank=rank,
        ulysses_size=ulysses_size,
        expected_source_tokens=expected_source_tokens,
        bernini_commit=bernini_commit,
        wan_diffusion_path=wan_diffusion_path,
    )
    installed.install()
    try:
        yield installed
    finally:
        installed.restore()


def validate_four_rank_forward_identities(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail before sampling unless all four ranks bind the same full input."""

    if len(rows) != EXPECTED_ULYSSES_SIZE:
        raise SourceKVRouteInferenceError("forward identity rank inventory differs")
    ranks = sorted(int(row.get("rank", -1)) for row in rows)
    if ranks != list(range(EXPECTED_ULYSSES_SIZE)):
        raise SourceKVRouteInferenceError("forward identity ranks differ")
    identities = []
    for row in rows:
        identity = _require_mapping(row.get("identity"), label="forward identity")
        digest = _require_sha256(
            row.get("identity_digest"), label="forward identity digest"
        )
        if route_scope.object_sha256(identity) != digest:
            raise SourceKVRouteInferenceError("forward identity digest differs")
        identities.append(dict(identity))
    if any(value != identities[0] for value in identities[1:]):
        raise SourceKVRouteInferenceError(
            "Ulysses ranks disagree on source/model/method/prompt identity"
        )
    return {
        "validated_before_forward": True,
        "rank_count": EXPECTED_ULYSSES_SIZE,
        "identity": identities[0],
        "identity_digest": route_scope.object_sha256(identities[0]),
    }


def validate_rank_runtime_certificate(
    *,
    core_receipt: Mapping[str, Any],
    trace: Mapping[str, Any],
    rank: int,
    hooks_restored: bool,
    forward_identity_digest: str,
    generated_latent_digest: str,
) -> dict[str, Any]:
    runtime = _require_mapping(core_receipt.get("runtime"), label="core runtime")
    cache = _require_mapping(runtime.get("cache"), label="core cache")
    per_block = runtime.get("per_block")
    expected_capture = 30 * 40
    expected_replay = 30 * 40 * 5
    expected_branch = {name: 30 * 40 for name in REPLAY_BRANCH_ORDER}
    _require_sha256(forward_identity_digest, label="forward identity digest")
    _require_sha256(generated_latent_digest, label="generated latent digest")
    if (
        core_receipt.get("block_indices") != list(EXPECTED_BLOCKS)
        or core_receipt.get("runtime_digest")
        != route_scope.object_sha256(runtime)
        or runtime.get("restored") is not True
        or hooks_restored is not True
        or cache.get("identity") is not None
        or cache.get("captured_blocks") != []
        or cache.get("capture_calls") != expected_capture
        or cache.get("replay_lookups") != expected_replay
        or cache.get("replay_branch_counts") != expected_branch
        or cache.get("replay_phase_counts")
        != {
            replay.EAGER_EXECUTION: expected_replay,
            replay.CHECKPOINT_FORWARD: 0,
            replay.CHECKPOINT_RECOMPUTE: 0,
        }
        or cache.get("retired_identity_count") != 40
        or not isinstance(per_block, list)
        or len(per_block) != 30
        or trace.get("sample_calls") != 1
        or trace.get("step_count") != 40
        or trace.get("source_prefix_verified") is not True
    ):
        raise SourceKVRouteInferenceError("rank-local deployment count audit differs")
    for block, row in enumerate(per_block):
        if (
            row.get("block_index") != block
            or row.get("capture_calls") != 40
            or row.get("replay_calls") != 200
            or row.get("branch_counts")
            != {
                replay.CAPTURE_BRANCH_TAG: 40,
                **{name: 40 for name in REPLAY_BRANCH_ORDER},
            }
            or row.get("execution_phase_counts")
            != {
                replay.EAGER_EXECUTION: 240,
                replay.CHECKPOINT_FORWARD: 0,
                replay.CHECKPOINT_RECOMPUTE: 0,
            }
        ):
            raise SourceKVRouteInferenceError(
                f"block {block} lacks capture40/replay200 evidence"
            )
    steps = trace.get("steps")
    if not isinstance(steps, list) or len(steps) != 40:
        raise SourceKVRouteInferenceError("deployment trace steps differ")
    for index, record in enumerate(steps):
        if (
            record.get("step_index") != index
            or record.get("forward_order") != list(EXPECTED_BRANCH_ORDER)
            or record.get("capture_forwards") != 1
            or record.get("replay_forwards") != 5
            or record.get("original_scheduler_calls") != 1
            or record.get("official_adapted_action_exact_parity") is not True
            or record.get("phase0_quotient_exact_zero") is not True
            or record.get("source_phase0_exact_preservation") is not True
            or record.get("target_energy_retention") != 1.0
            or record.get("target_clipped_fraction") != 0.0
            or record.get("sigma_strictly_positive") is not True
        ):
            raise SourceKVRouteInferenceError(f"deployment step {index} differs")
    return {
        "validated": True,
        "rank": int(rank),
        "all30": True,
        "sample_calls": 1,
        "step_count": 40,
        "per_layer_capture_calls": 40,
        "per_layer_replay_calls": 200,
        "rank_local_capture_calls": expected_capture,
        "rank_local_replay_lookups": expected_replay,
        "branch_counts": expected_branch,
        "unique_retired_step_identities": 40,
        "source_prefix_verified": True,
        "processor_restore": True,
        "sampler_scheduler_restore": True,
        "forward_identity_digest": forward_identity_digest,
        "generated_latent_digest": generated_latent_digest,
        "trace_digest": route_scope.object_sha256(trace),
        "core_runtime_digest": core_receipt["runtime_digest"],
    }


def validate_four_rank_certificates(
    certificates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(certificates) != 4 or sorted(int(row.get("rank", -1)) for row in certificates) != list(range(4)):
        raise SourceKVRouteInferenceError("four-rank certificate inventory differs")
    invariant = {
        key: certificates[0].get(key)
        for key in (
            "all30",
            "sample_calls",
            "step_count",
            "per_layer_capture_calls",
            "per_layer_replay_calls",
            "rank_local_capture_calls",
            "rank_local_replay_lookups",
            "branch_counts",
            "unique_retired_step_identities",
            "source_prefix_verified",
            "processor_restore",
            "sampler_scheduler_restore",
            "forward_identity_digest",
            "generated_latent_digest",
            "trace_digest",
            "core_runtime_digest",
        )
    }
    if any(any(row.get(key) != value for key, value in invariant.items()) for row in certificates):
        raise SourceKVRouteInferenceError(
            "Ulysses ranks disagree on deployment identity/count/output evidence"
        )
    return {
        "validated": True,
        "all_four_ranks_exact": True,
        "all_four_ranks_input_model_prompt_exact": True,
        "all_four_ranks_trace_core_exact": True,
        "all_four_ranks_generated_latent_exact": True,
        **invariant,
        "cross_rank_capture_calls": 4 * 1200,
        "cross_rank_replay_lookups": 4 * 6000,
        "per_rank": [dict(row) for row in certificates],
        "canonical_summaries": [
            {
                "rank": int(row["rank"]),
                "forward_identity_digest": row["forward_identity_digest"],
                "trace_digest": row["trace_digest"],
                "core_runtime_digest": row["core_runtime_digest"],
                "generated_latent_digest": row["generated_latent_digest"],
            }
            for row in certificates
        ],
        "certificates_digest": route_scope.object_sha256(
            [dict(row) for row in certificates]
        ),
    }


def deployment_operator_contract() -> dict[str, Any]:
    value = {
        "clean_field": DEPLOYMENT_CLEAN_FORMULA,
        "gauge": GAUGE_FORMULA,
        "scheduler_velocity": DEPLOYMENT_VELOCITY_FORMULA,
        "source_clean": "staged_source_video_vae_latent_S",
        "action_field": "adapted_action_after_shared_frozen_negative_APG",
        "noop_field": "adapted_noop_after_shared_frozen_negative_APG",
        "frozen_action_noop_role": "diagnostics_only_not_mixed_into_operator",
        "official_scheduler": "original_UniPCMultistepScheduler.step_once_per_solver_step",
        "sigma_policy": "strictly_positive_each_of_40_steps",
        "rho": None,
        "radius": None,
        "clipping": False,
        "field_mix": False,
        "target_energy_retention": 1.0,
        "target_clipped_fraction": 0.0,
    }
    value["contract_digest"] = route_scope.object_sha256(value)
    return value


def compute_deployment_clean_field(
    *, source_clean: Any, adapted_action: Any, adapted_noop: Any
) -> tuple[Any, Any]:
    """Execute the unbounded V9 clean-field operator on a 21-phase latent.

    This helper is intentionally independent of Bernini so the precise
    deployment algebra can be tested with real Torch before loading a model.
    It returns ``(E_k, Q0(A_theta-N_theta))`` and performs no release, radius,
    clipping, rescaling, or frozen-field mixture.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise SourceKVRouteInferenceError("deployment operator requires Torch") from error
    tensors = (source_clean, adapted_action, adapted_noop)
    if (
        any(not isinstance(value, torch.Tensor) for value in tensors)
        or any(value.ndim != 5 for value in tensors)
        or any(tuple(value.shape) != tuple(source_clean.shape) for value in tensors)
        or int(source_clean.shape[0]) != 1
        or int(source_clean.shape[2]) != EXPECTED_PHASES
        or any(value.device != source_clean.device for value in tensors)
        or any(not bool(torch.isfinite(value).all()) for value in tensors)
    ):
        raise SourceKVRouteInferenceError(
            "deployment fields must be finite same-device [1,C,21,H,W] tensors"
        )
    source = source_clean.float()
    delta = adapted_action.float() - adapted_noop.float()
    quotient = delta - delta[:, :, :1]
    executed = source + quotient
    if (
        not torch.equal(
            quotient[:, :, :1], torch.zeros_like(quotient[:, :, :1])
        )
        or not torch.equal(executed[:, :, :1], source[:, :, :1])
        or not bool(torch.isfinite(executed).all())
    ):
        raise SourceKVRouteInferenceError("deployment Q0/source gauge differs")
    return executed, quotient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict trained Bernini CSV-ART V9 source-only inference"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=EXPECTED_STEPS)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if (
        not isinstance(args.instruction, str)
        or not args.instruction.strip()
        or "\x00" in args.instruction
    ):
        raise SourceKVRouteInferenceError("instruction must be non-empty without NUL")
    if args.num_inference_steps != EXPECTED_STEPS or args.seed != EXPECTED_SEED:
        raise SourceKVRouteInferenceError("V9 deployment fixes 40 steps and seed 2027")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = str(getattr(args, name)).lower()
        if _SHA1_RE.fullmatch(value) is None:
            raise SourceKVRouteInferenceError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        _require_sha256(getattr(args, name), label=name)
    if (
        args.expected_bernini_commit.lower()
        != legacy.trainer.BERNINI_OFFICIAL_COMMIT
        or args.expected_veomni_commit.lower() != legacy.trainer.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256
        != legacy.trainer.CHECKPOINT_TREE_SHA256
    ):
        raise SourceKVRouteInferenceError("pinned source/checkpoint identity differs")


def build_receipt(
    *,
    args: argparse.Namespace,
    staged: StagedSource,
    source_metadata: Mapping[str, Any],
    source_tokens: int,
    output_path: Path,
    output_sha256: str,
    adapter_bundle: legacy.AdapterBundle,
    adapter_hashes: Mapping[str, str],
    adapter_identity: Mapping[str, Any],
    adapter_state_validation: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    forward_identity: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
    four_rank_runtime: Mapping[str, Any],
    rank0_core_receipt: Mapping[str, Any],
    rank0_trace: Mapping[str, Any],
) -> dict[str, Any]:
    instruction_bytes = args.instruction.encode("utf-8")
    rank0_trace_digest = route_scope.object_sha256(rank0_trace)
    rank0_core_runtime = _require_mapping(
        rank0_core_receipt.get("runtime"), label="rank0 core runtime"
    )
    if (
        four_rank_runtime.get("trace_digest") != rank0_trace_digest
        or four_rank_runtime.get("core_runtime_digest")
        != rank0_core_receipt.get("runtime_digest")
        or rank0_core_receipt.get("runtime_digest")
        != route_scope.object_sha256(rank0_core_runtime)
        or forward_identity.get("identity_digest")
        != four_rank_runtime.get("forward_identity_digest")
    ):
        raise SourceKVRouteInferenceError(
            "rank0 trace/core/forward identity does not bind four-rank evidence"
        )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "method_source_revision": args.method_source_revision.lower(),
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "bernini_inference_files": dict(inference_file_hashes),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_runtime_content": {
            **dict(checkpoint_identity),
            "verified_before_and_after_sampling": True,
        },
        "adapter": {
            "checkpoint_root": str(adapter_bundle.checkpoint_root),
            "training_receipt_digest": adapter_identity["receipt_digest"],
            "training_global_step": 40,
            "adapter_config_sha256": adapter_hashes["adapter_config"],
            "adapter_model_sha256": adapter_hashes["adapter_model"],
            "training_receipt_file_sha256": adapter_hashes["training_receipt"],
            "optimizer_checkpoint_sha256": adapter_hashes[
                "optimizer_checkpoint"
            ],
            "strict_tensor_reload_equal": True,
            "fresh_base_peft_from_pretrained_verified_by_this_runner": True,
            "loaded": True,
            "merged": False,
            "active_for_adapted_branches": True,
            "disabled_for_capture_and_frozen_branches": True,
            "target_module_count": 92,
            "adapter_tensor_count": 184,
            "trainable_parameter_count": 2_260_992,
            "target_modules_sha256": route_scope.EXPECTED_TARGET_MODULES_SHA256,
            "scope_manifest_digest": adapter_identity["scope_manifest_digest"],
            "state_validation": dict(adapter_state_validation),
            "v8_or_partial_adapter_rejected": True,
            "post_sampling_hashes_unchanged": True,
            "four_plain_files_staged_with_no_follow_fd": True,
            "config_receipt_and_peft_load_read_private_snapshot_only": True,
            "shared_and_private_snapshot_hashes_reverified_after_sampling": True,
        },
        "input": {
            "requested_source_video_path": str(staged.requested_path),
            "source_video_sha256": staged.sha256,
            "source_video_bytes": staged.byte_count,
            "source_staged_before_decode": True,
            "source_staging_no_follow_fd": True,
            "source_staging_hash_verified_after_close": True,
            "runtime_reads_only_private_staged_snapshot": True,
            "instruction_utf8_sha256": hashlib.sha256(instruction_bytes).hexdigest(),
            "instruction_utf8_bytes": len(instruction_bytes),
            "accepted_external_conditions": ["source_video", "action_instruction"],
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "generator_branch": False,
            "mask_or_swept_tube": False,
            "track_pose_trajectory_or_optical_flow": False,
            "first_frame_anchor": False,
        },
        "preprocessing": {
            **dict(source_metadata),
            "source_tokens": source_tokens,
            "pair_tokens": 2 * source_tokens,
            "frames": EXPECTED_FRAMES,
            "latent_phases": EXPECTED_PHASES,
        },
        "prompt_contract": {
            "task": "mv2v",
            "semantic_noop_instruction_sha256": route_batches.EXACT_NOOP_INSTRUCTION_SHA256,
            "negative_prompt_utf8_sha256": hashlib.sha256(
                legacy.DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
            ).hexdigest(),
            "tokenizer_fix_mistral_regex": True,
            "prompt_enhancer": False,
        },
        "sampling": {
            **legacy.sampler_contract(steps=40, seed=2027),
            "ulysses_size": 4,
            "single_expert": "transformer_1",
            "rank0_decode_and_save_only": True,
            "source_only_carrier_forwards_per_step": 1,
            "replay_forwards_per_step": 5,
            "original_scheduler_calls_per_step": 1,
        },
        "deployment_operator": deployment_operator_contract(),
        "carrier_runtime": {
            "selection": "all",
            "blocks": list(EXPECTED_BLOCKS),
            "capture": "source_only_semantic_noop_adapter_off_post_rope_kv",
            "replay_branch_order": list(REPLAY_BRANCH_ORDER),
            "per_layer_capture_calls": 40,
            "per_layer_replay_calls": 200,
            "forward_identity_gather": dict(forward_identity),
            "four_rank_certificate": dict(four_rank_runtime),
            "rank0_core_receipt": dict(rank0_core_receipt),
            "rank0_trace": dict(rank0_trace),
            "rank0_trace_digest": rank0_trace_digest,
            "rank0_core_runtime_digest": rank0_core_receipt["runtime_digest"],
        },
        "training_deployment_alignment": {
            "runtime_method_source_matches_training_source": True,
            "training_forward_order": list(EXPECTED_BRANCH_ORDER),
            "deployment_forward_order": list(EXPECTED_BRANCH_ORDER),
            "same_source_only_carrier": True,
            "same_all30_replay": True,
            "same_shared_frozen_negative_apg": True,
            "paired_target_loaded_at_inference": False,
            "training_query_state_policy": adapter_identity["query_state_policy"],
            "autoregressive_query_state_exposure_gap_declared": True,
        },
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
            "frame_count": 81,
            "fps": legacy.FPS,
            "height": source_metadata["source_derived_bucket_hw"][0],
            "width": source_metadata["source_derived_bucket_hw"][1],
            "audio_preserved": False,
        },
        "runtime_versions": dict(runtime_versions),
        "experimental_inference": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = route_scope.object_sha256(receipt)
    return receipt


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise SourceKVRouteInferenceError("stale temporary receipt exists")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    staged = stage_source_snapshot(args.source_video)
    staged_adapter: Optional[StagedAdapterBundle] = None
    try:
        output_path, receipt_path = legacy._resolve_output(args.output)
        staged_adapter = stage_adapter_snapshot(args.adapter_checkpoint)
        bundle = staged_adapter.bundle
        requested_bundle = staged_adapter.requested_bundle
        adapter_hashes = dict(staged_adapter.hashes)
        adapter_config = _read_json(bundle.adapter_config_path, label="adapter config")
        training_receipt = _read_json(
            bundle.training_receipt_path, label="training receipt"
        )
        adapter_identity = validate_training_checkpoint_contract(
            adapter_config=adapter_config,
            receipt=training_receipt,
            adapter_model_sha256=adapter_hashes["adapter_model"],
            adapter_config_sha256=adapter_hashes["adapter_config"],
            optimizer_checkpoint_sha256=adapter_hashes["optimizer_checkpoint"],
            expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        )
        if (
            adapter_identity["training_method_source_revision"]
            != args.method_source_revision.lower()
            or adapter_identity["training_method_source_archive_sha256"]
            != args.method_source_archive_sha256
        ):
            raise SourceKVRouteInferenceError(
                "runtime method archive does not exactly match V9 training source"
            )
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
        except legacy.trainer.TrainingContractError as error:
            raise SourceKVRouteInferenceError(str(error)) from error
        if transformer_config["num_attention_heads"] % 4:
            raise SourceKVRouteInferenceError("attention heads do not divide Ulysses=4")
        inference_file_hashes = legacy.validate_inference_source_files(bernini_root)
        legacy.trainer.activate_source_trees(bernini_root, veomni_root)

        import torch
        import torch.distributed as dist
        import peft
        from diffusers import __version__ as diffusers_version
        from diffusers.models import AutoencoderKLWan
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from transformers import AutoTokenizer, __version__ as transformers_version

        from bernini.cli import DEFAULT_NEG_PROMPT
        from bernini.io_utils import save_output
        from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
        from bernini.parallel import init_parallel_state
        from bernini.pipeline import _vae_decode, _vae_encode
        from bernini.training.data import SYSTEM_PROMPTS

        if (
            SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT
            or DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT
            or transformers_version != adapter_identity["transformers_version"]
            or peft.__version__ != EXPECTED_PEFT_VERSION
        ):
            raise SourceKVRouteInferenceError(
                "runtime prompt/Transformers/PEFT version contract differs"
            )
        distributed = legacy.inference_distributed_contract()
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
            raise SourceKVRouteInferenceError("four AUH ROCm GPUs are required")
        torch.cuda.set_device(distributed.local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=60),
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        init_parallel_state(ulysses_size=4)
        device = torch.device("cuda", distributed.local_rank)

        checkpoint_identity_box: list[Any] = [None]
        if distributed.rank == 0:
            checkpoint_identity_box[0] = validate_runtime_checkpoint_manifest(
                checkpoint
            )
        dist.broadcast_object_list(checkpoint_identity_box, src=0)
        checkpoint_identity = _require_mapping(
            checkpoint_identity_box[0], label="broadcast checkpoint identity"
        )
        if checkpoint_identity.get("validated") is not True:
            raise SourceKVRouteInferenceError("checkpoint runtime identity differs")

        source_tensor, source_metadata = legacy.prepare_exact_source(staged.staged_path)
        full_prompt = legacy.build_training_prompt(
            args.instruction, prompt_cleaner=prompt_clean
        )
        noop_prompt = legacy.build_training_prompt(
            route_batches.EXACT_NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
        )
        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        base_model = BerniniRendererModel(config)
        base_model.requires_grad_(False)
        base_model.eval()
        model, renderer, adapter_tensor_count, adapter_state_validation = (
            _strict_load_v9_adapter(base_model=base_model, bundle=bundle)
        )
        if adapter_tensor_count != 184:
            raise SourceKVRouteInferenceError("loaded adapter is not exact184")

        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
        )
        action_ids, action_mask = legacy._tokenize_training_prompt(
            tokenizer, full_prompt
        )
        noop_ids, noop_mask = legacy._tokenize_training_prompt(tokenizer, noop_prompt)
        negative_ids, negative_mask = legacy._tokenize_renderer_negative(
            tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT
        )
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval()
        vae.requires_grad_(False)
        vae.to(device)
        with torch.no_grad():
            source_latent = _vae_encode(
                vae, source_tensor.to(device=device, dtype=torch.float32)
            )
        bucket = source_metadata["source_derived_bucket_hw"]
        expected_latent_shape = (
            1,
            int(vae.config.z_dim),
            21,
            int(bucket[0]) // 8,
            int(bucket[1]) // 8,
        )
        if tuple(int(item) for item in source_latent.shape) != expected_latent_shape:
            raise SourceKVRouteInferenceError("source latent is not exact81/21 geometry")
        layout = tri.PackedLatentLayout.from_spatial_shape(expected_latent_shape)
        source_tokens = layout.tokens
        vae.to("cpu")
        del source_tensor
        torch.cuda.empty_cache()

        renderer.t5_text_encoder.to(device)
        with torch.no_grad():
            noop_embeddings = renderer.encode_prompt(
                noop_ids.to(device), noop_mask.to(device)
            )
        renderer.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()
        wan_diffusion_path = bernini_root / "bernini/models/wan_diffusion.py"
        sampling = legacy.sampler_contract(steps=40, seed=2027)
        diffusion = tri.resolve_diffusion_core(renderer)
        pre_schedule = sigma_strata.audit_runtime_unipc_schedule(
            diffusion.scheduler, initialize=True
        )
        forward_identity_value = {
            "schema_version": "bernini-v9-four-rank-forward-identity-v1",
            "source_video_sha256": staged.sha256,
            "source_video_bytes": staged.byte_count,
            "source_latent_sha256": tensor_sha256(
                source_latent, label="source clean latent"
            ),
            "instruction_utf8_sha256": hashlib.sha256(
                args.instruction.encode("utf-8")
            ).hexdigest(),
            "full_prompt_utf8_sha256": hashlib.sha256(
                full_prompt.encode("utf-8")
            ).hexdigest(),
            "noop_prompt_utf8_sha256": hashlib.sha256(
                noop_prompt.encode("utf-8")
            ).hexdigest(),
            "negative_prompt_utf8_sha256": hashlib.sha256(
                legacy.DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
            ).hexdigest(),
            "action_input_ids_sha256": tensor_sha256(
                action_ids, label="action input ids"
            ),
            "action_attention_mask_sha256": tensor_sha256(
                action_mask, label="action attention mask"
            ),
            "noop_input_ids_sha256": tensor_sha256(
                noop_ids, label="noop input ids"
            ),
            "noop_attention_mask_sha256": tensor_sha256(
                noop_mask, label="noop attention mask"
            ),
            "noop_embeddings_sha256": tensor_sha256(
                noop_embeddings, label="noop prompt embeddings"
            ),
            "negative_input_ids_sha256": tensor_sha256(
                negative_ids, label="negative input ids"
            ),
            "negative_attention_mask_sha256": tensor_sha256(
                negative_mask, label="negative attention mask"
            ),
            "adapter_file_sha256": adapter_hashes,
            "loaded_adapter_state_validation_digest": route_scope.object_sha256(
                adapter_state_validation
            ),
            "training_receipt_digest": adapter_identity["receipt_digest"],
            "base_checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
            "base_checkpoint_runtime_identity_digest": checkpoint_identity[
                "identity_digest"
            ],
            "method_source_revision": args.method_source_revision.lower(),
            "method_source_archive_sha256": args.method_source_archive_sha256,
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "bernini_inference_files_digest": route_scope.object_sha256(
                inference_file_hashes
            ),
            "frames": EXPECTED_FRAMES,
            "latent_phases": EXPECTED_PHASES,
            "steps": EXPECTED_STEPS,
            "seed": EXPECTED_SEED,
        }
        local_forward_identity = {
            "rank": distributed.rank,
            "identity": forward_identity_value,
            "identity_digest": route_scope.object_sha256(forward_identity_value),
        }
        forward_identity_rows: list[Any] = [None] * EXPECTED_ULYSSES_SIZE
        dist.all_gather_object(forward_identity_rows, local_forward_identity)
        forward_identity = validate_four_rank_forward_identities(
            forward_identity_rows
        )
        patch: Optional[replay.SourceKVReplayPatchHandle] = None
        hook: Optional[InstalledSourceKVRouteDeployment] = None
        try:
            with replay.source_kv_replay(renderer, selection="all") as installed_patch:
                patch = installed_patch
                with source_kv_route_deployment_hook(
                    renderer,
                    adapter_controller=model,
                    cache_bank=installed_patch.cache_bank,
                    noop_prompt_embeds=noop_embeddings,
                    source_clean=source_latent,
                    latent_shape=expected_latent_shape,
                    rank=distributed.rank,
                    ulysses_size=4,
                    expected_source_tokens=source_tokens,
                    bernini_commit=bernini_revision,
                    wan_diffusion_path=wan_diffusion_path,
                ) as installed_hook:
                    hook = installed_hook
                    with torch.no_grad():
                        generated_latent = renderer.sample(
                            input_ids=action_ids.to(device),
                            attention_mask=action_mask.to(device),
                            uncond_input_ids=negative_ids.to(device),
                            uncond_attention_mask=negative_mask.to(device),
                            image_vae_latents=None,
                            multi_video_vae_latents=[source_latent],
                            multi_image_vae_latents=None,
                            width=int(bucket[1]),
                            height=int(bucket[0]),
                            device=device,
                            **sampling,
                        )
            if patch is None or hook is None:
                raise SourceKVRouteInferenceError("deployment contexts did not install")
            core_receipt = patch.receipt()
            trace_value = hook.trace.as_dict()
            trace_value["source_prefix_verified"] = hook.source_prefix_verified
            generated_latent_digest = tensor_sha256(
                generated_latent, label="full generated latent"
            )
            local_certificate = validate_rank_runtime_certificate(
                core_receipt=core_receipt,
                trace=trace_value,
                rank=distributed.rank,
                hooks_restored=hook.restored,
                forward_identity_digest=forward_identity["identity_digest"],
                generated_latent_digest=generated_latent_digest,
            )
        except (
            replay.SourceKVReplayContractError,
            tri.TriBranchHookError,
            sigma_strata.InferenceSigmaStrataError,
        ) as error:
            raise SourceKVRouteInferenceError(str(error)) from error
        post_schedule = sigma_strata.audit_runtime_unipc_schedule(
            diffusion.scheduler, initialize=False
        )
        if pre_schedule != post_schedule:
            raise SourceKVRouteInferenceError("official UniPC schedule changed")
        if tuple(int(item) for item in generated_latent.shape) != expected_latent_shape:
            raise SourceKVRouteInferenceError("generated latent geometry differs")
        rank_certificates: list[Any] = [None] * 4
        dist.all_gather_object(rank_certificates, local_certificate)
        four_rank = validate_four_rank_certificates(rank_certificates)

        # Reprove all shared/private adapter identities and the complete base
        # checkpoint after sampling.  PEFT/config/receipt reads used only the
        # private adapter bundle above.
        verify_adapter_snapshot(staged_adapter)
        checkpoint_after_box: list[Any] = [None]
        if distributed.rank == 0:
            checkpoint_after_box[0] = validate_runtime_checkpoint_manifest(checkpoint)
        dist.broadcast_object_list(checkpoint_after_box, src=0)
        if checkpoint_after_box[0] != checkpoint_identity:
            raise SourceKVRouteInferenceError(
                "base checkpoint runtime identity changed during inference"
            )
        if _file_sha256(staged.staged_path) != staged.sha256:
            raise SourceKVRouteInferenceError("source snapshot changed during inference")
        model.to("cpu")
        del noop_embeddings, source_latent
        torch.cuda.empty_cache()

        if distributed.rank == 0:
            vae.to(device)
            with torch.no_grad():
                output = _vae_decode(vae, generated_latent)
            vae.to("cpu")
            if tuple(int(item) for item in output.shape) != (
                81,
                int(bucket[0]),
                int(bucket[1]),
                3,
            ):
                raise SourceKVRouteInferenceError("decoded output geometry differs")
            temporary = output_path.with_name(
                f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
            )
            if temporary.exists() or temporary.is_symlink():
                raise SourceKVRouteInferenceError("stale temporary output exists")
            save_output(output, str(temporary), fps=int(legacy.FPS))
            os.replace(temporary, output_path)
            from tools import materialize_vae

            frames, fps, encoded_hw = materialize_vae._decode_exact_video(output_path)
            legacy.validate_exact_video_metadata(int(frames.shape[0]), fps)
            if tuple(encoded_hw) != tuple(bucket):
                raise SourceKVRouteInferenceError("encoded output geometry differs")
            output_sha256 = _file_sha256(output_path)
            receipt = build_receipt(
                args=args,
                staged=staged,
                source_metadata=source_metadata,
                source_tokens=source_tokens,
                output_path=output_path,
                output_sha256=output_sha256,
                adapter_bundle=requested_bundle,
                adapter_hashes=adapter_hashes,
                adapter_identity=adapter_identity,
                adapter_state_validation=adapter_state_validation,
                checkpoint_identity=checkpoint_identity,
                forward_identity=forward_identity,
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                inference_file_hashes=inference_file_hashes,
                runtime_versions={
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "transformers": transformers_version,
                    "diffusers": diffusers_version,
                    "peft": peft.__version__,
                },
                four_rank_runtime=four_rank,
                rank0_core_receipt=core_receipt,
                rank0_trace=trace_value,
            )
            _atomic_write_json(receipt_path, receipt)
            print(route_scope.canonical_json_bytes(receipt).decode("utf-8"), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return 0
    finally:
        if staged_adapter is not None:
            cleanup_staged_adapter(staged_adapter)
        cleanup_staged_source(staged)


__all__ = [
    "DEPLOYMENT_CLEAN_FORMULA",
    "DEPLOYMENT_VELOCITY_FORMULA",
    "EXPECTED_BRANCH_ORDER",
    "EXPECTED_SEED",
    "EXPECTED_STEPS",
    "EXPECTED_PEFT_VERSION",
    "InstalledSourceKVRouteDeployment",
    "METHOD_NAME",
    "RECEIPT_SCHEMA",
    "SourceKVRouteInferenceError",
    "build_parser",
    "cleanup_staged_adapter",
    "compute_deployment_clean_field",
    "deployment_operator_contract",
    "expected_serialized_target_patterns",
    "source_kv_route_deployment_hook",
    "stage_adapter_snapshot",
    "stage_source_snapshot",
    "tensor_sha256",
    "validate_cli",
    "validate_four_rank_forward_identities",
    "validate_four_rank_certificates",
    "validate_rank_runtime_certificate",
    "validate_training_checkpoint_contract",
    "validate_runtime_checkpoint_manifest",
    "verify_adapter_snapshot",
    "verify_official_source_prefix",
]


if __name__ == "__main__":
    raise SystemExit(main())
