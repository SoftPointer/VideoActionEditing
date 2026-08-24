#!/usr/bin/env python3
"""Disposable paired action-edit engineering smoke for the 0817 program.

This entry point is deliberately *not* a D0/D1/D2 trainer.  It consumes only
the legacy 644-row paired Parquet release under its original experimental
authorization, selects the mechanically strict subset, and executes exactly
two WORLD8 DP2xSP4 updates from a fresh Bernini-R 1.3B base.  Its only purpose
is to close the large-LoRA, ActionPlanPredictorV1, block-injection, optimizer,
checkpoint, and reload engineering path before a qualified 0817 dataset
exists.  Checkpoints emitted here are PRE_D0_ENGINEERING_ONLY and may not be
promoted or used for scientific/action-quality claims.

The trainable renderer surface is the packed-preservation-v2 skeleton:
rank-256 LoRA on q/k/v/out of both attention modules in all 30 blocks plus
typed source/target patch and role parameters.  One formal
ActionPlanPredictorV1 reads only complete clean-source patch tokens and the
row's complete contextual instruction tokens.  Its 30 exactly-zero-initialized
heads inject phase/global action residuals into the target suffix after each
corresponding transformer block.  Targets never enter the predictor.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import stat
import sys
import tempfile
import threading
import time
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

METHOD = "bernini-action-edit-large-lora-0817-v1"
AUTHORITY = "PRE_D0_ENGINEERING_ONLY"
RECEIPT_SCHEMA = "bernini-action-edit-large-lora-0817-pre-d0-receipt-v1"
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
GRADIENT_ACCUMULATION = 4
GLOBAL_BATCH = GRADIENT_ACCUMULATION * DP_SIZE
MAX_STEPS = 2
FRAME_COUNT = 81
LATENT_PHASES = 21
LATENT_CHANNELS = 16
PATCH_VALUES = 64
TRANSFORMER_BLOCKS = 30
HIDDEN_WIDTH = 1536
INSTRUCTION_WIDTH = 4096
EXPECTED_DATASET_ROWS = 644
EXPECTED_STRICT_ROWS = 359
EXPECTED_NON_STRICT_ROWS = 285
DEFAULT_SEED = 20260817
DEFAULT_LR = 1.0e-4
DEFAULT_MAX_GRAD_NORM = 1.0
LORA_SCOPE = "all-attention"
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
ACTION_PLAN_PREDICTOR_SOURCE_SHA256 = (
    "464cd500f0ba1edb6cbe6d4f07287bfff346ae0ba7968c0d7c7f3cc7cb667308"
)
ACTION_PLAN_CONDITIONER_STATE_ABI_SHA256 = (
    "04c2fc8ff48fb8b027e912cd6c9c58cf19d4b554c84127fb6623268a9d1e398b"
)
RELEASE_MANIFEST_SCHEMA = (
    "bernini-action-edit-large-lora-0817-pre-d0-release-manifest-v1"
)
RELEASE_MEMBER_ROOT = "methods/bernini_action_editing"
RELEASE_FILES_AND_MODES = {
    "action_plan_predictor_v1.py": 0o444,
    "clean_source_visual_context_stage_b_contract_v1.py": 0o444,
    "inference_sigma_strata.py": 0o444,
    "packed_preservation_lora_v2.py": 0o444,
    "packed_preservation_release_v2.py": 0o444,
    "source_self_runtime.py": 0o444,
    "train_action_edit_large_lora_0817_v1.py": 0o444,
    "train_lora.py": 0o444,
}
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PreD0EngineeringError(RuntimeError):
    """Raised before an ambiguous update or artifact can be published."""


def fail(message: str) -> NoReturn:
    raise PreD0EngineeringError(message)


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
        raise PreD0EngineeringError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    named = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def validate_release_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str,
    method_root: Path = METHOD_ROOT,
) -> Mapping[str, Any]:
    """Authenticate the exact regular-file Python closure before imports."""

    if _SHA256.fullmatch(expected_sha256) is None:
        fail("release manifest SHA differs")
    requested = manifest_path.expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail("release manifest must be one absolute non-symlink file")
    try:
        manifest = requested.resolve(strict=True)
        root = method_root.expanduser().resolve(strict=True)
    except OSError as error:
        raise PreD0EngineeringError(f"release closure is unavailable: {error}") from error
    if (
        manifest != requested
        or not stat.S_ISREG(manifest.lstat().st_mode)
        or method_root.is_symlink()
        or not stat.S_ISDIR(root.lstat().st_mode)
    ):
        fail("release manifest/root canonical file types differ")
    before = manifest.stat()
    payload = manifest.read_bytes()
    after = manifest.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )
    if (
        identity(before) != identity(after)
        or len(payload) > 1024 * 1024
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        fail("release manifest stable bytes/SHA differ")
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreD0EngineeringError("release manifest is not UTF-8 JSON") from error
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema_version", "member_root", "files"}
        or raw.get("schema_version") != RELEASE_MANIFEST_SCHEMA
        or raw.get("member_root") != RELEASE_MEMBER_ROOT
        or not isinstance(raw.get("files"), list)
    ):
        fail("release manifest envelope differs")
    expected_paths = tuple(sorted(RELEASE_FILES_AND_MODES))
    rows = raw["files"]
    if len(rows) != len(expected_paths):
        fail("release manifest has missing/extra members")
    normalized = []
    for expected_relative, row in zip(expected_paths, rows):
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "mode",
            "size",
            "sha256",
        }:
            fail("release manifest member schema differs")
        relative = row.get("path")
        pure = PurePosixPath(str(relative))
        if (
            relative != expected_relative
            or pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in ("", ".", "..") for part in pure.parts)
            or row.get("mode") != RELEASE_FILES_AND_MODES[expected_relative]
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or type(row.get("sha256")) is not str
            or _SHA256.fullmatch(row["sha256"]) is None
        ):
            fail("release manifest exact sorted member closure differs")
        member = root / expected_relative
        try:
            member_lstat_before = member.lstat()
            resolved_member = member.resolve(strict=True)
        except OSError as error:
            raise PreD0EngineeringError(
                f"release member is unavailable: {expected_relative}: {error}"
            ) from error
        if (
            member.is_symlink()
            or resolved_member != member
            or not stat.S_ISREG(member_lstat_before.st_mode)
            or stat.S_IMODE(member_lstat_before.st_mode) != row["mode"]
            or member_lstat_before.st_size != row["size"]
            or file_sha256(member) != row["sha256"]
        ):
            fail(f"executed release member identity differs: {expected_relative}")
        member_lstat_after = member.lstat()
        if identity(member_lstat_before) != identity(member_lstat_after):
            fail(f"release member changed during validation: {expected_relative}")
        normalized.append(dict(row))
    if tuple(row["path"] for row in normalized) != expected_paths:
        fail("release manifest exact member set differs")
    return {
        "path": str(manifest),
        "sha256": expected_sha256,
        "schema_version": RELEASE_MANIFEST_SCHEMA,
        "member_root": RELEASE_MEMBER_ROOT,
        "members": normalized,
        "member_count": len(normalized),
        "member_set_sha256": object_sha256(normalized),
        "regular_non_symlink_exact_modes_sizes_hashes_verified": True,
    }


def validate_imported_release_modules(
    release_closure: Mapping[str, Any],
    modules: Mapping[str, Any],
    *,
    method_root: Path = METHOD_ROOT,
) -> Mapping[str, Any]:
    """Prove bare imports executed the already-authenticated member bytes."""

    if set(modules) != set(RELEASE_FILES_AND_MODES):
        fail("imported release module set has missing/extra members")
    rows = release_closure.get("members")
    if not isinstance(rows, list):
        fail("release closure member rows are unavailable")
    expected = {row.get("path"): row for row in rows if isinstance(row, Mapping)}
    if set(expected) != set(RELEASE_FILES_AND_MODES):
        fail("release closure member map differs during import authentication")
    receipts = []
    for relative in sorted(RELEASE_FILES_AND_MODES):
        module = modules[relative]
        file_value = getattr(module, "__file__", None)
        if type(file_value) is not str or not file_value:
            fail(f"imported release module lacks __file__: {relative}")
        requested = Path(file_value)
        try:
            resolved = requested.resolve(strict=True)
        except OSError as error:
            raise PreD0EngineeringError(
                f"imported release module is unavailable: {relative}: {error}"
            ) from error
        wanted = method_root.expanduser().resolve(strict=True) / relative
        row = expected[relative]
        if (
            requested.is_symlink()
            or resolved != wanted
            or not stat.S_ISREG(resolved.lstat().st_mode)
            or stat.S_IMODE(resolved.lstat().st_mode) != row["mode"]
            or resolved.stat().st_size != row["size"]
            or file_sha256(resolved) != row["sha256"]
        ):
            fail(f"imported release module identity differs: {relative}")
        receipts.append(
            {
                "path": relative,
                "resolved_file": str(resolved),
                "sha256": row["sha256"],
                "size": row["size"],
                "mode": row["mode"],
            }
        )
    return {
        "exact_imported_member_count": len(receipts),
        "members": receipts,
        "members_sha256": object_sha256(receipts),
        "resolved_under_method_root_and_rehashed_after_import": True,
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite JSON artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_seed(base: int, *coordinates: Any) -> int:
    payload = "\0".join(str(value) for value in (base, METHOD, *coordinates)).encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def serialized_model_load() -> Iterator[None]:
    """Serialize eight model loads inside one allocation step."""

    job = os.environ.get("SLURM_JOB_ID", "no-slurm")
    step = os.environ.get("SLURM_STEP_ID", "no-step")
    path = Path(f"/tmp/bernini-action-edit-0817-{job}-{step}.model-load.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def validate_experimental_authority(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept only the old explicitly non-formal experimental authority state."""

    required = {
        "schema_version": "bernini-r-action-vae-dataset-summary-v2",
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "experimental_training_acknowledged": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "complete": True,
        "expected_sample_count": EXPECTED_DATASET_ROWS,
        "materialized_sample_count": EXPECTED_DATASET_ROWS,
        "missing_sample_count": 0,
        "raw_strict_selection_rows": EXPECTED_STRICT_ROWS,
        "raw_non_strict_selection_rows": EXPECTED_NON_STRICT_ROWS,
        "materialized_strict_selection_rows": EXPECTED_STRICT_ROWS,
        "materialized_non_strict_selection_rows": EXPECTED_NON_STRICT_ROWS,
        "frame_count": FRAME_COUNT,
        "latent_frame_count": LATENT_PHASES,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            fail(f"legacy experimental dataset authority differs at {key}")
    if float(summary.get("fps", -1.0)) != 25.0:
        fail("legacy experimental dataset FPS differs")
    return {
        "authority": AUTHORITY,
        "preview_only": True,
        "formal_training_authorized": False,
        "experimental_training_acknowledged": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "target_quality_status": "mechanically_strict_legacy_gate_only_not_0817_qualified",
    }


def validate_strict_catalog(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int = EXPECTED_DATASET_ROWS,
    expected_strict: int = EXPECTED_STRICT_ROWS,
) -> tuple[Mapping[str, Any], ...]:
    """Validate a lightweight one-row-per-shard catalog and return strict rows."""

    if len(rows) != expected_rows:
        fail(f"legacy catalog row count differs: {len(rows)} != {expected_rows}")
    seen_iids: set[str] = set()
    seen_indices: set[int] = set()
    normalized: list[Mapping[str, Any]] = []
    for raw in rows:
        iid = raw.get("iid")
        index = raw.get("row_index")
        strict = raw.get("strict_selection_gates_all_true")
        shard_name = raw.get("parquet_name")
        posterior_shape = raw.get("posterior_parameter_shape")
        tokens_per_role = raw.get("tokens_per_role")
        if (
            type(iid) is not str
            or not iid
            or iid in seen_iids
            or type(index) is not int
            or not 0 <= index < expected_rows
            or index in seen_indices
            or type(strict) is not bool
            or shard_name != f"{iid}.parquet"
            or not isinstance(posterior_shape, (list, tuple))
            or len(posterior_shape) != 5
            or tuple(posterior_shape[:3])
            != (1, 2 * LATENT_CHANNELS, LATENT_PHASES)
            or any(type(value) is not int or value <= 0 for value in posterior_shape)
            or posterior_shape[3] % 2
            or posterior_shape[4] % 2
            or type(tokens_per_role) is not int
            or tokens_per_role
            != LATENT_PHASES * (posterior_shape[3] // 2) * (posterior_shape[4] // 2)
        ):
            fail("legacy mechanically-strict catalog schema differs")
        seen_iids.add(iid)
        seen_indices.add(index)
        normalized.append(
            {
                "iid": iid,
                "row_index": index,
                "strict_selection_gates_all_true": strict,
                "parquet_name": shard_name,
                "posterior_parameter_shape": list(posterior_shape),
                "tokens_per_role": tokens_per_role,
            }
        )
    if seen_indices != set(range(expected_rows)):
        fail("legacy catalog index closure differs")
    strict_rows = tuple(
        sorted(
            (
                row
                for row in normalized
                if row["strict_selection_gates_all_true"] is True
            ),
            key=lambda row: (str(row["iid"]), int(row["row_index"])),
        )
    )
    if len(strict_rows) != expected_strict:
        fail(f"mechanically strict row count differs: {len(strict_rows)}")
    if len(normalized) - len(strict_rows) != expected_rows - expected_strict:
        fail("mechanically non-strict row count differs")
    return strict_rows


def strict_two_step_schedule(
    strict_rows: Sequence[Mapping[str, Any]], *, max_steps: int = MAX_STEPS
) -> tuple[Mapping[str, Any], ...]:
    """Return 16 records cycled over the strict maximum-token geometry tier."""

    if max_steps != MAX_STEPS:
        fail("PRE_D0 runner accepts exactly two optimizer updates")
    required = max_steps * GLOBAL_BATCH
    if not strict_rows:
        fail("mechanically strict subset is empty")
    if any(
        type(row.get("tokens_per_role")) is not int
        or row["tokens_per_role"] <= 0
        for row in strict_rows
    ):
        fail("mechanically strict schedule lacks authenticated token geometry")
    global_strict_max_tokens = max(row["tokens_per_role"] for row in strict_rows)
    maximum_geometry_rows = sorted(
        (
            row
            for row in strict_rows
            if row["tokens_per_role"] == global_strict_max_tokens
        ),
        key=lambda row: (str(row["iid"]), int(row["row_index"])),
    )
    maximum_geometry_identities = {
        (str(row["iid"]), int(row["row_index"]))
        for row in maximum_geometry_rows
    }
    if (
        len(maximum_geometry_rows) != len(maximum_geometry_identities)
        or len(maximum_geometry_identities) < DP_SIZE
    ):
        fail(
            "global strict maximum-token geometry has fewer than two distinct DP rows"
        )
    selected = tuple(
        maximum_geometry_rows[index % len(maximum_geometry_rows)]
        for index in range(required)
    )
    if any(row["tokens_per_role"] != global_strict_max_tokens for row in selected):
        fail("two-step schedule escaped global strict maximum-token geometry")
    return selected


def build_strict_catalog_from_parquet(
    dataset: Any, *, legacy: Any
) -> tuple[Mapping[str, Any], ...]:
    """Authenticate one shard's real posterior shapes at a time (bounded RAM)."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PreD0EngineeringError("pyarrow is required for strict catalog scan") from error
    import torch
    rows: list[Mapping[str, Any]] = []
    for index, path in enumerate(dataset.files):
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if not {
            "iid",
            "strict_selection_gates_all_true",
            "video_vae_latents",
        }.issubset(names):
            fail(f"legacy shard lacks mechanical strict metadata: {path.name}")
        if parquet.metadata.num_rows != 1:
            fail(f"legacy shard must contain exactly one row: {path.name}")
        values = parquet.read(
            columns=["iid", "strict_selection_gates_all_true", "video_vae_latents"]
        ).to_pylist()
        if len(values) != 1:
            fail(f"legacy shard lightweight read differs: {path.name}")
        blobs = legacy._as_list(
            values[0].pop("video_vae_latents"), label="video_vae_latents"
        )
        if len(blobs) != 2:
            fail(f"legacy shard lacks source/target posteriors: {path.name}")
        tensors = [legacy._load_tensor_blob(blob) for blob in blobs]
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            fail(f"legacy shard posterior tensor type differs: {path.name}")
        shapes = [tuple(int(value) for value in tensor.shape) for tensor in tensors]
        if (
            shapes[0] != shapes[1]
            or len(shapes[0]) != 5
            or shapes[0][:3]
            != (1, 2 * LATENT_CHANNELS, LATENT_PHASES)
            or shapes[0][3] <= 0
            or shapes[0][4] <= 0
            or shapes[0][3] % 2
            or shapes[0][4] % 2
            or any(
                tensor.device.type != "cpu"
                or tensor.dtype != torch.float32
                or not tensor.is_contiguous()
                or tensor.requires_grad
                for tensor in tensors
            )
        ):
            fail(f"legacy shard actual posterior geometry differs: {path.name}")
        tokens_per_role = (
            LATENT_PHASES * (shapes[0][3] // 2) * (shapes[0][4] // 2)
        )
        rows.append(
            {
                **values[0],
                "row_index": index,
                "parquet_name": path.name,
                "posterior_parameter_shape": list(shapes[0]),
                "tokens_per_role": tokens_per_role,
            }
        )
        del tensors, blobs, values
    return validate_strict_catalog(rows)


def schedule_row(
    selected: Sequence[Mapping[str, Any]],
    *,
    optimizer_step_zero_based: int,
    microbatch_index: int,
    dp_arm: int,
) -> Mapping[str, Any]:
    for value, upper, label in (
        (optimizer_step_zero_based, MAX_STEPS, "optimizer step"),
        (microbatch_index, GRADIENT_ACCUMULATION, "microbatch"),
        (dp_arm, DP_SIZE, "DP arm"),
    ):
        if type(value) is not int or not 0 <= value < upper:
            fail(f"{label} coordinate differs")
    logical = (
        optimizer_step_zero_based * GLOBAL_BATCH
        + microbatch_index * DP_SIZE
        + dp_arm
    )
    if len(selected) != MAX_STEPS * GLOBAL_BATCH:
        fail("two-step row schedule closure differs")
    return selected[logical]


def _instruction_from_sanitized(sample: Mapping[str, Any], legacy: Any) -> str:
    messages = legacy._parse_inputs(sample.get("inputs"))
    legacy._validate_message_contract(messages)
    instruction = messages[1].get("text")
    if type(instruction) is not str or not instruction.strip():
        fail("paired row instruction is empty")
    return instruction.strip()


def paired_posterior_modes(
    sample: Mapping[str, Any], *, mean: Any, std: Any, legacy: Any
) -> tuple[Any, Any]:
    """Decode deterministic source and target posterior modes and normalize."""

    import torch

    legacy.validate_81_frame_latents(
        sample, expected_parameter_channels=2 * LATENT_CHANNELS
    )
    blobs = legacy._as_list(sample.get("video_vae_latents"), label="video_vae_latents")
    outputs = []
    for role, blob in zip(("source", "target"), blobs):
        parameters = legacy._load_tensor_blob(blob)
        if (
            not isinstance(parameters, torch.Tensor)
            or parameters.dtype != torch.float32
            or parameters.device.type != "cpu"
            or parameters.requires_grad
            or parameters.ndim != 5
            or tuple(int(value) for value in parameters.shape[:3])
            != (1, 2 * LATENT_CHANNELS, LATENT_PHASES)
            or not parameters.is_contiguous()
            or not bool(torch.isfinite(parameters).all().item())
        ):
            fail(f"{role} posterior parameters differ")
        # DiagonalGaussianDistribution.mode() is exactly the first C channels.
        mode = parameters[:, :LATENT_CHANNELS]
        normalized = ((mode - mean.unsqueeze(0)) / std.unsqueeze(0)).detach()
        normalized = normalized.float().contiguous()
        if not bool(torch.isfinite(normalized).all().item()):
            fail(f"{role} normalized posterior mode is non-finite")
        outputs.append(normalized)
    if outputs[0].shape != outputs[1].shape:
        fail("paired source/target posterior-mode geometry differs")
    return outputs[0], outputs[1]


def _pack_latent_patches(latent: Any) -> Any:
    import torch

    if (
        not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float32
        or latent.device.type != "cpu"
        or latent.ndim != 4
        or tuple(int(value) for value in latent.shape[:2])
        != (LATENT_CHANNELS, LATENT_PHASES)
        or int(latent.shape[2]) % 2
        or int(latent.shape[3]) % 2
        or not latent.is_contiguous()
    ):
        fail("latent patch input must be CPU FP32 [16,21,evenH,evenW]")
    channels, phases, height, width = (int(value) for value in latent.shape)
    return (
        latent.reshape(channels, phases, height // 2, 2, width // 2, 2)
        .permute(1, 2, 4, 0, 3, 5)
        .reshape(phases * (height // 2) * (width // 2), channels, 1, 2, 2)
        .contiguous()
    )


def _packed_output_field(patches: Any) -> Any:
    return patches.permute(0, 2, 3, 4, 1).reshape(1, int(patches.shape[0]), PATCH_VALUES)


def prepare_paired_flow(
    *, source: Any, target: Any, epsilon: Any, coordinate: Any, rope: Any, device: Any
) -> Mapping[str, Any]:
    """Pack clean source condition and noisy genuine edited target."""

    import torch

    for value, label in ((source, "source"), (target, "target"), (epsilon, "epsilon")):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.shape != source.shape
            or value.ndim != 5
            or tuple(int(item) for item in value.shape[:3])
            != (1, LATENT_CHANNELS, LATENT_PHASES)
        ):
            fail(f"{label} paired flow latent differs")
    source_clean = source.squeeze(0).contiguous()
    target_clean = target.squeeze(0).contiguous()
    eps = epsilon.squeeze(0).contiguous()
    sigma = float(coordinate.sigma)
    noisy_target = ((1.0 - sigma) * target_clean + sigma * eps).contiguous()
    target_velocity = (eps - target_clean).contiguous()
    source_patches = _pack_latent_patches(source_clean)
    target_patches = _pack_latent_patches(noisy_target)
    velocity_patches = _pack_latent_patches(target_velocity)
    source_tokens = int(source_patches.shape[0])
    target_tokens = int(target_patches.shape[0])
    patch_grid = (
        LATENT_PHASES,
        int(source_clean.shape[2]) // 2,
        int(source_clean.shape[3]) // 2,
    )
    if (
        source_tokens != target_tokens
        or source_tokens <= 0
        or target_tokens % LATENT_PHASES
    ):
        fail("source/target packed phase geometry differs")
    input_patches = torch.cat((source_patches, target_patches), dim=0).to(device)
    source_rope = rope(source.to(device), source_id=1)
    target_rope = rope(noisy_target.unsqueeze(0).to(device), source_id=0)
    rotary = torch.cat((source_rope, target_rope), dim=2)
    rotary = rotary.squeeze(0).permute(1, 0, 2).contiguous()
    return {
        "input_patches": input_patches,
        "rotary": rotary,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "total_tokens": source_tokens + target_tokens,
        "spatial_tokens_per_phase": target_tokens // LATENT_PHASES,
        "patch_grid": patch_grid,
        "target_velocity": _packed_output_field(velocity_patches).to(device),
    }


def canonical_instruction_tokens(text_embs: Any, text_lens: Any) -> Any:
    """Expose all and only the contextual tokens of one row instruction."""

    import torch

    if not isinstance(text_embs, torch.Tensor) or not text_embs.is_floating_point():
        fail("T5 contextual instruction embedding must be floating tensor")
    if text_embs.ndim == 2:
        value = text_embs.unsqueeze(0)
    elif text_embs.ndim == 3 and int(text_embs.shape[0]) == 1:
        value = text_embs
    else:
        fail("T5 contextual instruction embedding must be [L,4096] or [1,L,4096]")
    if int(value.shape[-1]) != INSTRUCTION_WIDTH:
        fail("T5 contextual instruction width differs")
    if not isinstance(text_lens, torch.Tensor):
        fail("T5 instruction lengths must be a tensor")
    lengths = text_lens.reshape(-1)
    if int(lengths.numel()) != 1:
        fail("one row must expose one complete instruction length")
    length = int(lengths[0].item())
    if not 0 < length <= int(value.shape[1]):
        fail("complete contextual instruction length differs")
    result = value[:, :length].contiguous()
    if not bool(torch.isfinite(result).all().item()):
        fail("contextual instruction tokens are non-finite")
    return result


def materialize_training_text_embedding(text_embs: Any) -> Any:
    """Copy a frozen T5 result out of inference-tensor ownership."""

    import torch

    if not isinstance(text_embs, torch.Tensor) or not text_embs.is_floating_point():
        fail("frozen T5 result must be a floating tensor")
    result = text_embs.detach().clone(memory_format=torch.contiguous_format)
    if (
        result.requires_grad
        or torch.is_inference(result)
        or not result.is_contiguous()
        or result.data_ptr() == text_embs.data_ptr()
    ):
        fail("frozen T5 result did not leave inference-tensor ownership")
    return result


def prepare_action_injection_route(
    *, conditioner: Any, embedded: Any, packed: Mapping[str, Any], instruction_tokens: Any
) -> Any:
    """Bind complete pre-SP THW source tokens to a certified target suffix."""

    from action_plan_predictor_v1 import certify_closed_target_suffix_route

    phases, height, width = packed["patch_grid"]
    source = embedded[:, : packed["source_tokens"], :].reshape(
        1, phases, height, width, HIDDEN_WIDTH
    )
    target = embedded[:, packed["source_tokens"] :, :].reshape(
        1, phases, height, width, HIDDEN_WIDTH
    )
    if (
        source.numel() != packed["source_tokens"] * HIDDEN_WIDTH
        or target.numel() != packed["target_tokens"] * HIDDEN_WIDTH
    ):
        fail("complete pre-SP source/target THW token closure differs")
    if instruction_tokens.device != source.device:
        fail("predictor source/instruction device ownership differs")
    # The predictor deliberately restores its input dtype.  Bernini's frozen
    # T5 and patch embedder are independently executed operators, so close the
    # dtype contract explicitly at this one pre-SP route boundary.
    instruction_tokens = instruction_tokens.to(dtype=source.dtype).contiguous()
    ownership = certify_closed_target_suffix_route(
        target,
        source_prefix_tokens=packed["source_tokens"],
        packed_total_tokens=packed["total_tokens"],
        audit_finite=True,
    )
    route = conditioner.prepare_route(source, instruction_tokens, ownership)
    if route.ownership.digest != ownership.digest:
        fail("conditioner route ownership changed after certification")
    return route


@dataclass
class ActionInjectionRoute:
    """Authenticated global source-prefix/target-suffix layout for SP4 hooks."""

    source_tokens: int
    target_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    plan: Any
    row_identity: str
    block_calls: list[int] = field(default_factory=list, init=False)
    forward_block_calls: list[int] = field(default_factory=list, init=False)
    recompute_block_calls: list[int] = field(default_factory=list, init=False)
    checkpoint_capture_indices: list[int] = field(default_factory=list, init=False)
    checkpoint_forward_context_indices: list[int] = field(
        default_factory=list, init=False
    )
    checkpoint_recompute_context_indices: list[int] = field(
        default_factory=list, init=False
    )
    _lease_state: str = field(default="new", init=False, repr=False)
    _lease_serial: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.source_tokens) is not int
            or self.source_tokens <= 0
            or type(self.target_tokens) is not int
            or self.target_tokens != self.source_tokens
            or self.target_tokens % LATENT_PHASES
            or self.sequence_parallel_size not in (1, SP_SIZE)
            or type(self.sequence_parallel_rank) is not int
            or not 0 <= self.sequence_parallel_rank < self.sequence_parallel_size
            or type(self.row_identity) is not str
            or not self.row_identity
        ):
            fail("action injection route geometry differs")

    def _begin_lease(self, serial: int) -> None:
        if (
            self._lease_state != "new"
            or self._lease_serial is not None
            or self.block_calls
            or self.forward_block_calls
            or self.recompute_block_calls
            or self.checkpoint_capture_indices
            or self.checkpoint_forward_context_indices
            or self.checkpoint_recompute_context_indices
        ):
            fail("stale or reused action route is forbidden")
        self._lease_state = "active"
        self._lease_serial = serial

    def _require_active_lease(self, serial: int) -> None:
        if self._lease_state != "active" or self._lease_serial != serial:
            fail("stale action route checkpoint replay is forbidden")

    def _end_lease(self, serial: int) -> None:
        self._require_active_lease(serial)
        self._lease_state = "closed"

    def record_checkpoint_capture(self) -> int:
        if self._lease_state != "active":
            fail("checkpoint capture requires an active action route lease")
        index = len(self.checkpoint_capture_indices)
        if index >= TRANSFORMER_BLOCKS:
            fail("action route captured more than 30 checkpoint contexts")
        self.checkpoint_capture_indices.append(index)
        return index

    def record_checkpoint_context(self, *, phase: str, index: int) -> None:
        if self._lease_state != "active":
            fail("checkpoint context requires an active action route lease")
        if index not in self.checkpoint_capture_indices:
            fail("checkpoint context did not originate from this action route")
        if phase == "forward":
            values = self.checkpoint_forward_context_indices
            if index != len(values):
                fail("action checkpoint forward contexts are not exact 0..29")
        elif phase == "recompute":
            values = self.checkpoint_recompute_context_indices
            if index in values:
                fail("action checkpoint recompute context was reused")
        else:
            fail("action checkpoint phase differs")
        if len(values) >= TRANSFORMER_BLOCKS:
            fail(f"action checkpoint {phase} context count exceeds 30")
        values.append(index)

    def record_block_call(self, *, phase: str, checkpoint_index: int, block_index: int) -> None:
        if self._lease_state != "active" or checkpoint_index != block_index:
            fail("Bernini block/checkpoint action-route identity differs")
        if phase == "forward":
            values = self.forward_block_calls
            if block_index != len(values):
                fail("action injection did not traverse blocks 0..29 in forward order")
        elif phase == "recompute":
            values = self.recompute_block_calls
            if block_index in values:
                fail("action injection checkpoint recomputed one block more than once")
        else:
            fail("action injection checkpoint phase differs")
        if len(values) >= TRANSFORMER_BLOCKS:
            fail(f"action injection {phase} block count exceeds 30")
        values.append(block_index)
        self.block_calls.append(block_index)

    @property
    def total_tokens(self) -> int:
        return self.source_tokens + self.target_tokens

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    @property
    def spatial_tokens_per_phase(self) -> int:
        return self.target_tokens // LATENT_PHASES

    def local_phase_indices_tuple(self) -> tuple[int, ...]:
        start = self.sequence_parallel_rank * self.local_length
        result = []
        for global_index in range(start, start + self.local_length):
            target_index = global_index - self.source_tokens
            if 0 <= target_index < self.target_tokens:
                result.append(target_index // self.spatial_tokens_per_phase)
            else:
                result.append(-1)
        return tuple(result)

    def local_phase_indices(self, *, device: Any) -> Any:
        import torch

        return torch.tensor(
            self.local_phase_indices_tuple(), dtype=torch.int64, device=device
        )

    def validate_forward_traversal(self) -> None:
        expected = tuple(range(TRANSFORMER_BLOCKS))
        if (
            tuple(self.checkpoint_capture_indices) != expected
            or tuple(self.checkpoint_forward_context_indices) != expected
            or tuple(self.forward_block_calls) != expected
            or tuple(self.block_calls[:TRANSFORMER_BLOCKS]) != expected
        ):
            fail("action injection did not traverse blocks 0..29 in forward order")

    def validate_forward_and_recompute_traversal(self) -> None:
        """Bind non-reentrant checkpointing to one forward plus one recompute."""

        self.validate_forward_traversal()
        counts = [self.block_calls.count(index) for index in range(TRANSFORMER_BLOCKS)]
        if (
            counts != [2] * TRANSFORMER_BLOCKS
            or len(self.block_calls) != 2 * TRANSFORMER_BLOCKS
            or len(self.checkpoint_recompute_context_indices) != TRANSFORMER_BLOCKS
            or set(self.checkpoint_recompute_context_indices) != set(range(TRANSFORMER_BLOCKS))
            or len(self.recompute_block_calls) != TRANSFORMER_BLOCKS
            or set(self.recompute_block_calls) != set(range(TRANSFORMER_BLOCKS))
        ):
            fail(
                "action injection requires exactly one forward and one checkpoint "
                "recompute per block/context, got "
                f"blocks={counts}, captures={len(self.checkpoint_capture_indices)}, "
                f"forward_contexts={len(self.checkpoint_forward_context_indices)}, "
                f"recompute_contexts={len(self.checkpoint_recompute_context_indices)}"
            )

    def receipt(self) -> Mapping[str, Any]:
        if self._lease_state != "closed":
            fail("action route receipt requires completed finally cleanup")
        counts = {str(index): self.block_calls.count(index) for index in range(TRANSFORMER_BLOCKS)}
        self.validate_forward_and_recompute_traversal()
        if set(self.block_calls) != set(range(TRANSFORMER_BLOCKS)):
            fail("action injection block-call closure differs")
        return {
            "row_identity": self.row_identity,
            "source_tokens": self.source_tokens,
            "target_tokens": self.target_tokens,
            "spatial_tokens_per_phase": self.spatial_tokens_per_phase,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "local_phase_indices_sha256": object_sha256(
                list(self.local_phase_indices_tuple())
            ),
            "block_call_counts": counts,
            "checkpoint_context_captures": len(self.checkpoint_capture_indices),
            "checkpoint_forward_contexts": len(
                self.checkpoint_forward_context_indices
            ),
            "checkpoint_recompute_contexts": len(
                self.checkpoint_recompute_context_indices
            ),
            "checkpoint_recompute_calls_per_block": 1,
            "exact_block_set_0_through_29": True,
            "source_or_padding_written": False,
        }


_ACTIVE_ACTION_ROUTE: ContextVar[Optional[ActionInjectionRoute]] = ContextVar(
    "bernini_action_edit_large_lora_0817_route_v1", default=None
)
_ACTION_CHECKPOINT_FORWARD = "forward"
_ACTION_CHECKPOINT_RECOMPUTE = "recompute"


@dataclass(frozen=True)
class _ActionCheckpointBinding:
    route: ActionInjectionRoute
    lease_serial: int
    phase: str
    checkpoint_index: int


_ACTIVE_ACTION_CHECKPOINT_BINDING: ContextVar[
    Optional[_ActionCheckpointBinding]
] = ContextVar(
    "bernini_action_edit_large_lora_0817_checkpoint_binding_v1", default=None
)


class _ActionRouteLeaseOwner:
    """One process-local lease; hooks still authenticate only via ContextVars."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._route: Optional[ActionInjectionRoute] = None
        self._serial = 0

    def acquire(self, route: ActionInjectionRoute) -> int:
        with self._lock:
            if self._route is not None:
                fail("concurrent or nested action route leases are forbidden")
            self._serial += 1
            route._begin_lease(self._serial)
            self._route = route
            return self._serial

    def require(self, route: ActionInjectionRoute, serial: int) -> None:
        with self._lock:
            if self._route is not route:
                fail("stale action route checkpoint replay is forbidden")
            route._require_active_lease(serial)

    def release(self, route: ActionInjectionRoute, serial: int) -> None:
        with self._lock:
            if self._route is not route:
                fail("action route lease owner changed before finally cleanup")
            try:
                route._end_lease(serial)
            finally:
                self._route = None

    def is_empty(self) -> bool:
        with self._lock:
            return self._route is None


_ACTION_ROUTE_LEASE_OWNER = _ActionRouteLeaseOwner()


def active_action_route() -> Optional[ActionInjectionRoute]:
    route = _ACTIVE_ACTION_ROUTE.get()
    if route is not None:
        serial = route._lease_serial
        if serial is None:
            fail("active action route has no lease serial")
        _ACTION_ROUTE_LEASE_OWNER.require(route, serial)
    return route


@contextmanager
def activate_action_route(route: ActionInjectionRoute) -> Iterator[None]:
    if not isinstance(route, ActionInjectionRoute):
        fail("action route type differs")
    if active_action_route() is not None:
        fail("nested action injection routes are forbidden")
    serial = _ACTION_ROUTE_LEASE_OWNER.acquire(route)
    try:
        token: Token[Optional[ActionInjectionRoute]] = _ACTIVE_ACTION_ROUTE.set(route)
    except Exception:
        _ACTION_ROUTE_LEASE_OWNER.release(route, serial)
        raise
    try:
        yield
    finally:
        try:
            _ACTIVE_ACTION_ROUTE.reset(token)
        finally:
            _ACTION_ROUTE_LEASE_OWNER.release(route, serial)


@contextmanager
def _replay_action_checkpoint_route(
    route: ActionInjectionRoute,
    *,
    lease_serial: int,
    phase: str,
    checkpoint_index: int,
) -> Iterator[None]:
    """Rebind one exact live lease in an autograd recompute context."""

    _ACTION_ROUTE_LEASE_OWNER.require(route, lease_serial)
    current = _ACTIVE_ACTION_ROUTE.get()
    if current is not None and current is not route:
        fail("checkpoint recomputation entered a different action route")
    if _ACTIVE_ACTION_CHECKPOINT_BINDING.get() is not None:
        fail("nested action checkpoint contexts are forbidden")
    route.record_checkpoint_context(phase=phase, index=checkpoint_index)
    route_token: Optional[Token[Optional[ActionInjectionRoute]]] = None
    if current is None:
        route_token = _ACTIVE_ACTION_ROUTE.set(route)
    binding = _ActionCheckpointBinding(
        route=route,
        lease_serial=lease_serial,
        phase=phase,
        checkpoint_index=checkpoint_index,
    )
    try:
        binding_token = _ACTIVE_ACTION_CHECKPOINT_BINDING.set(binding)
    except Exception:
        if route_token is not None:
            _ACTIVE_ACTION_ROUTE.reset(route_token)
        raise
    try:
        yield
    finally:
        try:
            _ACTIVE_ACTION_CHECKPOINT_BINDING.reset(binding_token)
        finally:
            if route_token is not None:
                _ACTIVE_ACTION_ROUTE.reset(route_token)


def action_route_checkpoint_context_fn() -> tuple[Any, Any]:
    """Capture one exact route for non-reentrant forward and recomputation."""

    route = active_action_route()
    if route is None or route._lease_serial is None:
        fail("checkpoint was created without an active action route")
    checkpoint_index = route.record_checkpoint_capture()
    return (
        _replay_action_checkpoint_route(
            route,
            lease_serial=route._lease_serial,
            phase=_ACTION_CHECKPOINT_FORWARD,
            checkpoint_index=checkpoint_index,
        ),
        _replay_action_checkpoint_route(
            route,
            lease_serial=route._lease_serial,
            phase=_ACTION_CHECKPOINT_RECOMPUTE,
            checkpoint_index=checkpoint_index,
        ),
    )


def validate_action_route_checkpointing_installation(
    transformer: Any,
) -> Mapping[str, Any]:
    """Audit the exact callable that HF installed on the live transformer."""

    checkpointing_func = getattr(transformer, "_gradient_checkpointing_func", None)
    keywords = getattr(checkpointing_func, "keywords", None)
    if (
        not bool(getattr(transformer, "gradient_checkpointing", False))
        or not isinstance(keywords, Mapping)
        or set(keywords) != {"use_reentrant", "context_fn"}
        or keywords.get("use_reentrant") is not False
        or keywords.get("context_fn") is not action_route_checkpoint_context_fn
    ):
        fail("live Bernini gradient-checkpoint route installation differs")
    return {
        "enabled": True,
        "use_reentrant": False,
        "context_fn": "action_route_checkpoint_context_fn",
        "live_partial_identity_verified": True,
        "checkpoint_contexts_per_microbatch": TRANSFORMER_BLOCKS,
        "early_stop_disabled_during_forward_and_backward": True,
    }


def _output_tensor(output: Any) -> tuple[Any, Any]:
    import torch

    if isinstance(output, torch.Tensor):
        return output, lambda value: value
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], lambda value: (value, *output[1:])
    fail("Bernini block output must be Tensor or tensor-first tuple")


def _tensor_bits_equal(left: Any, right: Any) -> bool:
    """Compare floating tensors byte-for-byte, including signed zero bits."""

    import torch

    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    return bool(
        torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


@dataclass
class InstalledActionPlanHooks:
    transformer: Any
    conditioner: Any
    handles: tuple[Any, ...]
    block_ids: tuple[int, ...]
    restored: bool = False

    def restore(self) -> None:
        if (
            self.restored
            or _ACTIVE_ACTION_ROUTE.get() is not None
            or not _ACTION_ROUTE_LEASE_OWNER.is_empty()
        ):
            fail("cannot restore active action-plan hooks")
        for handle in self.handles:
            handle.remove()
        if getattr(self.transformer, "action_plan_conditioner_v1", None) is not self.conditioner:
            fail("action-plan conditioner owner changed")
        delattr(self.transformer, "action_plan_conditioner_v1")
        self.restored = True


def install_action_plan_hooks(transformer: Any, conditioner: Any) -> InstalledActionPlanHooks:
    """Register exact block-indexed target-only post-block injections."""

    import torch
    from action_plan_predictor_v1 import ActionPlanConditionerV1

    if not isinstance(conditioner, ActionPlanConditionerV1):
        fail("formal ActionPlanConditionerV1 is required")
    conditioner.config.require_formal_0817()
    conditioner.injection.validate_block_traversal(tuple(range(TRANSFORMER_BLOCKS)))
    blocks = tuple(getattr(transformer, "blocks", ()))
    if len(blocks) != TRANSFORMER_BLOCKS or hasattr(
        transformer, "action_plan_conditioner_v1"
    ):
        fail("Bernini exact30 action hook structure differs")
    if any(
        not callable(getattr(block, "register_forward_hook", None))
        or not isinstance(getattr(block, "_forward_hooks", None), Mapping)
        or bool(block._forward_hooks)
        for block in blocks
    ):
        fail("all 30 Bernini blocks require empty auditable hook registries")
    first_parameter = next(transformer.parameters(), None)
    if first_parameter is None:
        fail("Bernini transformer has no materialized parameter")
    conditioner.to(device=first_parameter.device, dtype=torch.float32)
    transformer.add_module("action_plan_conditioner_v1", conditioner)
    handles = []
    try:
        for block_index, block in enumerate(blocks):

            def callback(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                bound_index: int = block_index,
            ) -> Any:
                route = active_action_route()
                if route is None:
                    fail("Bernini block executed without authenticated action route")
                binding = _ACTIVE_ACTION_CHECKPOINT_BINDING.get()
                if (
                    binding is None
                    or binding.route is not route
                    or binding.lease_serial != route._lease_serial
                    or binding.phase
                    not in (_ACTION_CHECKPOINT_FORWARD, _ACTION_CHECKPOINT_RECOMPUTE)
                ):
                    fail("Bernini block executed without authenticated checkpoint route")
                native, rebuild = _output_tensor(output)
                if (
                    native.ndim != 3
                    or int(native.shape[0]) != 1
                    or int(native.shape[1]) != route.local_length
                    or int(native.shape[2]) != HIDDEN_WIDTH
                    or not native.is_floating_point()
                ):
                    fail("Bernini local block hidden geometry differs")
                residual = conditioner.injection.residual(
                    route.plan, block_index=bound_index
                )
                if tuple(residual.shape) != (1, LATENT_PHASES, HIDDEN_WIDTH):
                    fail("block-indexed phase residual geometry differs")
                phases = route.local_phase_indices(device=native.device)
                selector = phases >= 0
                safe_phases = phases.clamp_min(0)
                local_delta = residual.index_select(1, safe_phases)
                target_selector = selector.view(1, -1, 1)
                target_adapted = native + local_delta.to(dtype=native.dtype)
                # torch.where selects the original native bytes on source and
                # append-padding rows.  Adding a numeric zero is insufficient:
                # it can canonicalize a negative-zero bit pattern.
                adapted = torch.where(target_selector, target_adapted, native)
                if bool((~selector).any().item()) and not _tensor_bits_equal(
                    adapted[:, ~selector, :], native[:, ~selector, :]
                ):
                    fail("action injection changed a source or padding token")
                route.record_block_call(
                    phase=binding.phase,
                    checkpoint_index=binding.checkpoint_index,
                    block_index=bound_index,
                )
                return rebuild(adapted)

            handles.append(block.register_forward_hook(callback))
    except Exception:
        for handle in handles:
            handle.remove()
        delattr(transformer, "action_plan_conditioner_v1")
        raise
    if len(handles) != TRANSFORMER_BLOCKS:
        fail("action injection hook count differs")
    return InstalledActionPlanHooks(
        transformer=transformer,
        conditioner=conditioner,
        handles=tuple(handles),
        block_ids=tuple(id(block) for block in blocks),
    )


def predict_target(
    *,
    renderer: Any,
    packed: Mapping[str, Any],
    coordinate: Any,
    text_lens: Any,
    text_embs: Any,
) -> Any:
    route = active_action_route()
    if route is None:
        fail("target prediction requires active action route")
    rotary = packed["rotary"].permute(1, 0, 2).unsqueeze(0)
    value = renderer.diff_dec.shared_step(
        model_id="transformer_1",
        noisy_latents=packed["embedded"],
        timesteps=packed["embedded"].new_tensor(
            [coordinate.timestep], dtype=__import__("torch").int64
        ),
        cond_embeds=text_embs,
        rotary_embs=rotary,
        batch_vae_seqlen=[packed["total_tokens"]],
        batch_text_seqlen=text_lens,
    )
    route.validate_forward_traversal()
    target = value[:, packed["source_tokens"] :, :]
    if tuple(target.shape) != (1, packed["target_tokens"], PATCH_VALUES):
        fail("official shared_step target prediction geometry differs")
    return target


def exact_trainable_named_parameters(
    model: Any, conditioner: Any
) -> tuple[tuple[str, Any], ...]:
    """Return only rank-256/typed/action-plan trainables, with exact ownership."""

    conditioner_ids = {id(parameter) for parameter in conditioner.parameters()}
    if not conditioner_ids:
        fail("action-plan conditioner parameter scope is empty")
    result = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not result or len({name for name, _ in result}) != len(result):
        fail("trainable parameter inventory is empty or duplicated")
    observed_conditioner_ids = {
        id(parameter) for _, parameter in result if id(parameter) in conditioner_ids
    }
    if observed_conditioner_ids != conditioner_ids:
        fail("registered action-plan conditioner trainable ownership differs")
    leaked = []
    for name, parameter in result:
        allowed = (
            ".lora_A." in name
            or ".lora_B." in name
            or ".source_delta." in name
            or ".target_delta." in name
            or ".role_embedding" in name
            or id(parameter) in conditioner_ids
        )
        if not allowed:
            leaked.append(name)
    if leaked:
        fail(f"base/text parameter leaked into PRE_D0 optimizer: {leaked[:8]}")
    return result


def trainable_inventory(named: Sequence[tuple[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(
        {
            "name": name,
            "shape": [int(value) for value in parameter.shape],
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in named
    )
    if not rows or len({row["name"] for row in rows}) != len(rows):
        fail("exact trainable inventory differs")
    return rows


def tensor_digest(named: Sequence[tuple[str, Any]]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in named:
        tensor = parameter.detach().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def gradient_digest(named: Sequence[tuple[str, Any]]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in named:
        if parameter.grad is None:
            fail(f"gradient digest missing parameter: {name}")
        gradient = parameter.grad.detach().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(gradient.shape), "dtype": str(gradient.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(gradient.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def export_trainable_state(named: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    import torch

    state = {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in named
    }
    if set(state) != {name for name, _ in named} or any(
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or not bool(torch.isfinite(value.float()).all().item())
        for value in state.values()
    ):
        fail("exported full trainable state differs")
    return state


def load_trainable_state_strict(
    named: Sequence[tuple[str, Any]], state: Mapping[str, Any]
) -> None:
    import torch

    parameters = dict(named)
    if not isinstance(state, Mapping) or set(state) != set(parameters):
        fail("strict full-state parameter-name set differs")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(parameter.shape)
                or value.dtype != parameter.dtype
                or not bool(torch.isfinite(value.float()).all().item())
            ):
                fail(f"strict full-state tensor differs: {name}")
            parameter.copy_(value.to(device=parameter.device))


def export_conditioner_state(conditioner: Any) -> Mapping[str, Any]:
    import torch

    state = {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in conditioner.state_dict().items()
    }
    if not state or any(
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or not bool(torch.isfinite(value.float()).all().item())
        for value in state.values()
    ):
        fail("exported ActionPlanConditionerV1 state differs")
    return state


def load_conditioner_state_strict(conditioner: Any, state: Mapping[str, Any]) -> None:
    import torch

    expected = conditioner.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        fail("strict conditioner state key set differs")
    for name, reference in expected.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != tuple(reference.shape)
            or value.dtype != reference.dtype
            or not bool(torch.isfinite(value.float()).all().item())
        ):
            fail(f"strict conditioner state tensor differs: {name}")
    # The module's custom loaders independently authenticate its ABI buffers.
    result = conditioner.load_state_dict(dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        fail("strict conditioner state reload closure differs")


def _cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _state_tree_bits_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality for optimizer/runtime checkpoint state."""

    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        left_cpu = left.detach().to(device="cpu").contiguous()
        right_cpu = right.detach().to(device="cpu").contiguous()
        return (
            left_cpu.shape == right_cpu.shape
            and left_cpu.dtype == right_cpu.dtype
            and torch.equal(
                left_cpu.reshape(-1).view(torch.uint8),
                right_cpu.reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_state_tree_bits_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_state_tree_bits_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def gather_world_runtime_state(
    *, step: int, distributed: Any, device: Any, world_group: Any
) -> Mapping[str, Any]:
    """Seal all eight RNG streams plus the deterministic next-row cursor."""

    import torch
    import torch.distributed as dist

    if type(step) is not int or not 0 <= step <= MAX_STEPS:
        fail("runtime-state checkpoint step differs")
    local = {
        "world_rank": int(distributed.rank),
        "dp_arm": int(distributed.arm_index),
        "sp_rank": int(distributed.sp_rank),
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu().contiguous(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state(device).cpu().contiguous(),
    }
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local, group=world_group)
    if (
        [row.get("world_rank") for row in gathered] != list(range(WORLD_SIZE))
        or [row.get("dp_arm") for row in gathered]
        != [rank // SP_SIZE for rank in range(WORLD_SIZE)]
        or [row.get("sp_rank") for row in gathered]
        != [rank % SP_SIZE for rank in range(WORLD_SIZE)]
    ):
        fail("WORLD8 runtime-state rank closure differs")
    next_cursor = None
    if step < MAX_STEPS:
        next_cursor = {
            "optimizer_step_zero_based": step,
            "microbatch_index": 0,
            "next_logical_record": step * GLOBAL_BATCH,
            "dp_arms": list(range(DP_SIZE)),
        }
    return {
        "schema_version": "bernini-action-edit-runtime-state-v1",
        "completed_optimizer_steps": step,
        "next_sampler_cursor": next_cursor,
        "scheduler": {
            "object": None,
            "policy": "constant_lr_no_scheduler_object",
            "learning_rate": DEFAULT_LR,
            "completed_steps": step,
        },
        "stochasticity": {
            "training_noise": "counter_based_per_row_torch_Generator_cpu",
            "dropout": 0.0,
            "rng_snapshots_retained_for_full_replay_abi": True,
        },
        "per_rank": gathered,
    }


def validate_adamw_state_abi(
    state: Mapping[str, Any], named: Sequence[tuple[str, Any]], *, step: int
) -> None:
    """Close the replicated FP32 AdamW group and two-moment state ABI."""

    import torch

    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        fail("AdamW checkpoint state envelope differs")
    groups = state["param_groups"]
    moments = state["state"]
    if not isinstance(groups, list) or len(groups) != 1:
        fail("engineering AdamW requires one replicated zero-WD parameter group")
    group = groups[0]
    expected_indices = list(range(len(named)))
    if (
        group.get("params") != expected_indices
        or float(group.get("lr", -1.0)) != DEFAULT_LR
        or tuple(group.get("betas", ())) != (0.9, 0.95)
        or float(group.get("eps", -1.0)) != 1.0e-8
        or float(group.get("weight_decay", -1.0)) != 0.0
        or group.get("amsgrad") is not False
    ):
        fail("engineering AdamW param-group ABI differs")
    if step == 0:
        if moments:
            fail("fresh P0 AdamW state must not contain moments")
        return
    if set(moments) != set(expected_indices):
        fail("AdamW two-moment parameter ownership differs")
    for index, (_, parameter) in enumerate(named):
        entry = moments[index]
        if not isinstance(entry, Mapping) or set(entry) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            fail(f"AdamW two-moment state keys differ at parameter {index}")
        first = entry["exp_avg"]
        second = entry["exp_avg_sq"]
        counter = entry["step"]
        if (
            not isinstance(first, torch.Tensor)
            or not isinstance(second, torch.Tensor)
            or tuple(first.shape) != tuple(parameter.shape)
            or tuple(second.shape) != tuple(parameter.shape)
            or first.dtype != torch.float32
            or second.dtype != torch.float32
            or not bool(torch.isfinite(first).all().item())
            or not bool(torch.isfinite(second).all().item())
            or not isinstance(counter, torch.Tensor)
            or int(counter.numel()) != 1
            or float(counter.item()) != float(step)
        ):
            fail(f"AdamW FP32 two-moment tensor ABI differs at parameter {index}")


def save_checkpoint(
    *,
    root: Path,
    step: int,
    named: Sequence[tuple[str, Any]],
    conditioner: Any,
    optimizer: Any,
    metadata: Mapping[str, Any],
    world_runtime_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Create-only checkpoint; rank0 roundtrips it and retains all8 RNG bytes."""

    import torch

    if type(step) is not int or not 0 <= step <= MAX_STEPS:
        fail("checkpoint step differs")
    final = root / f"checkpoint-{step:08d}"
    if final.exists() or final.is_symlink():
        fail(f"refusing to overwrite checkpoint: {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=root))
    adapter_path = temporary / "full_trainable_state.pt"
    optimizer_path = temporary / "optimizer.pt"
    runtime_path = temporary / "runtime_state.pt"
    metadata_path = temporary / "metadata.json"
    torch.save(
        {
            "schema_version": "bernini-action-edit-full-state-v1",
            "trainable_parameters": dict(export_trainable_state(named)),
            "action_plan_conditioner": dict(export_conditioner_state(conditioner)),
        },
        adapter_path,
    )
    optimizer_state = _cpu_tree(optimizer.state_dict())
    validate_adamw_state_abi(optimizer_state, named, step=step)
    torch.save(optimizer_state, optimizer_path)
    torch.save(_cpu_tree(world_runtime_state), runtime_path)
    loaded_adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(loaded_adapter, Mapping)
        or set(loaded_adapter)
        != {"schema_version", "trainable_parameters", "action_plan_conditioner"}
        or loaded_adapter.get("schema_version")
        != "bernini-action-edit-full-state-v1"
    ):
        fail("strict full-state checkpoint envelope differs")
    load_conditioner_state_strict(
        conditioner, loaded_adapter["action_plan_conditioner"]
    )
    load_trainable_state_strict(named, loaded_adapter["trainable_parameters"])
    loaded_optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    current_optimizer = optimizer.state_dict()
    validate_adamw_state_abi(loaded_optimizer, named, step=step)
    if (
        not isinstance(loaded_optimizer, Mapping)
        or set(loaded_optimizer) != {"state", "param_groups"}
        or len(loaded_optimizer["param_groups"]) != len(current_optimizer["param_groups"])
        or len(loaded_optimizer["state"]) != len(current_optimizer["state"])
    ):
        fail("strict optimizer reload schema differs")
    if not _state_tree_bits_equal(loaded_optimizer, _cpu_tree(current_optimizer)):
        fail("strict optimizer raw state bytes differ before reload")
    optimizer.load_state_dict(loaded_optimizer)
    validate_adamw_state_abi(optimizer.state_dict(), named, step=step)
    if not _state_tree_bits_equal(
        loaded_optimizer, _cpu_tree(optimizer.state_dict())
    ):
        fail("strict optimizer state differs after reload")
    loaded_runtime = torch.load(runtime_path, map_location="cpu", weights_only=True)
    if not _state_tree_bits_equal(loaded_runtime, _cpu_tree(world_runtime_state)):
        fail("persisted all8 RNG/sampler/scheduler state bytes differ")
    if (
        loaded_runtime.get("schema_version")
        != "bernini-action-edit-runtime-state-v1"
        or loaded_runtime.get("completed_optimizer_steps") != step
        or len(loaded_runtime.get("per_rank", ())) != WORLD_SIZE
    ):
        fail("persisted all8 runtime-state envelope differs")
    rank_zero_runtime = loaded_runtime["per_rank"][0]
    random.setstate(rank_zero_runtime["python_random_state"])
    torch.set_rng_state(rank_zero_runtime["torch_cpu_rng_state"])
    torch.cuda.set_rng_state(
        rank_zero_runtime["torch_cuda_rng_state"], device=named[0][1].device
    )
    restored_rank_zero = {
        **rank_zero_runtime,
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu().contiguous(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state(
            named[0][1].device
        ).cpu().contiguous(),
    }
    if not _state_tree_bits_equal(restored_rank_zero, rank_zero_runtime):
        fail("rank0 RNG state differs after strict reload")
    roundtrip_digest = tensor_digest(named)
    if roundtrip_digest != metadata.get("parameter_sha256"):
        fail("strict full-state reload changed trainable bytes")
    payload = {
        **dict(metadata),
        "step": step,
        "authority": AUTHORITY,
        "promotable": False,
        "adapter_file": adapter_path.name,
        "adapter_sha256": file_sha256(adapter_path),
        "optimizer_file": optimizer_path.name,
        "optimizer_sha256": file_sha256(optimizer_path),
        "runtime_state_file": runtime_path.name,
        "runtime_state_sha256": file_sha256(runtime_path),
        "rank0_full_trainable_state_roundtrip_reload_verified": True,
        "rank0_optimizer_roundtrip_reload_verified": True,
        "all8_rng_sampler_scheduler_state_bytes_persisted_verified": True,
        "rank0_rng_state_roundtrip_reload_verified": True,
        "roundtrip_parameter_sha256": roundtrip_digest,
        "trainable_inventory": list(trainable_inventory(named)),
    }
    metadata_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with metadata_path.open("rb") as handle:
        os.fsync(handle.fileno())
    fsync_directory(temporary)
    os.rename(temporary, final)
    fsync_directory(root)
    return {
        "step": step,
        "path": str(final),
        "adapter_sha256": payload["adapter_sha256"],
        "optimizer_sha256": payload["optimizer_sha256"],
        "runtime_state_sha256": payload["runtime_state_sha256"],
        "metadata_sha256": file_sha256(final / "metadata.json"),
        "rank0_full_trainable_optimizer_roundtrip_reload_verified": True,
        "all8_runtime_state_bytes_persisted_verified": True,
        "rank0_runtime_state_roundtrip_reload_verified": True,
    }


def synchronize_gradients_bucketed(
    named: Sequence[tuple[str, Any]], parallel: Any, *, bucket_bytes: int = 64 * 1024 * 1024
) -> float:
    """Average replicated gradients over SP4 and then DP2 in bounded buckets."""

    import torch
    import torch.distributed as dist
    import source_self_runtime as runtime

    if not named or bucket_bytes <= 0:
        fail("bucketed gradient scope differs")
    ready = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for _, parameter in named
    )
    if not runtime.world_all_true(ready, group=parallel.world_group):
        fail("at least one WORLD8 trainable gradient is absent/non-finite")
    buckets: list[list[Any]] = []
    current: list[Any] = []
    current_bytes = 0
    for _, parameter in named:
        assert parameter.grad is not None
        item_bytes = parameter.grad.numel() * parameter.grad.element_size()
        if current and current_bytes + item_bytes > bucket_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(parameter)
        current_bytes += item_bytes
    if current:
        buckets.append(current)
    squared = torch.zeros((), dtype=torch.float64, device=named[0][1].device)
    for bucket in buckets:
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=parallel.sp_group)
        flat.div_(float(SP_SIZE))
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=parallel.dp_group)
        flat.div_(float(DP_SIZE))
        offset = 0
        for parameter in bucket:
            assert parameter.grad is not None
            count = parameter.grad.numel()
            parameter.grad.copy_(flat[offset : offset + count].view_as(parameter.grad))
            squared.add_(parameter.grad.detach().to(torch.float64).square().sum())
            offset += count
        if offset != flat.numel():
            fail("bucketed gradient scatter closure differs")
    norm = float(torch.sqrt(squared).item())
    if not math.isfinite(norm) or norm <= 0.0:
        fail("WORLD8 synchronized gradient norm is zero/non-finite")
    return norm


def lora_affine_gradient_audit(
    named: Sequence[tuple[str, Any]], specs: Sequence[Any], *, completed_step: int
) -> Mapping[str, Any]:
    """Require exact per-affine LoRA bootstrap/second-step coverage."""

    import torch

    parameters = dict(named)
    result: dict[str, Any] = {}
    for factor in ("A", "B"):
        rows = []
        for spec in specs:
            matches = [
                (name, parameter)
                for name, parameter in parameters.items()
                if f"{spec.name}.lora_{factor}." in name
            ]
            if len(matches) != 1 or matches[0][1].grad is None:
                fail(f"LoRA affine gradient owner differs: {spec.name}/{factor}")
            name, parameter = matches[0]
            norm = float(
                torch.sqrt(parameter.grad.detach().to(torch.float64).square().sum()).item()
            )
            if not math.isfinite(norm) or norm < 0.0:
                fail(f"LoRA affine gradient is invalid: {name}")
            rows.append({"name": name, "norm": norm, "positive": norm > 0.0})
        required = factor == "B" or completed_step >= 2
        positive = sum(bool(row["positive"]) for row in rows)
        if required and positive != len(specs):
            fail(
                f"step {completed_step} has silent LoRA-{factor} affines: "
                f"{positive}/{len(specs)}"
            )
        result[f"lora_{factor}"] = {
            "expected_affines": len(specs),
            "positive_affines": positive,
            "all_positive_required": required,
            "rows_sha256": object_sha256(rows),
            "min_norm": min(row["norm"] for row in rows),
            "max_norm": max(row["norm"] for row in rows),
        }
    return result


def staged_gradient_audit(
    named: Sequence[tuple[str, Any]], specs: Sequence[Any], *, completed_step: int
) -> Mapping[str, Any]:
    """Enforce zero-init bootstrap at step 1 and full staged reach at step 2."""

    import torch

    if completed_step not in (1, 2):
        fail("staged gradient audit only covers updates one and two")
    rows = []
    groups: dict[str, Any] = {}
    injection_heads = {index: 0.0 for index in range(TRANSFORMER_BLOCKS)}
    predictor_families = {
        "source": 0.0,
        "instruction": 0.0,
        "queries": 0.0,
        "blocks": 0.0,
        "outputs": 0.0,
    }
    for name, parameter in named:
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
            fail(f"missing/non-finite staged gradient: {name}")
        norm = float(
            torch.sqrt(parameter.grad.detach().to(torch.float64).square().sum()).item()
        )
        if not math.isfinite(norm) or norm < 0.0:
            fail(f"invalid staged gradient norm: {name}")
        if ".lora_A." in name:
            group = "lora_A"
        elif ".lora_B." in name:
            group = "lora_B"
        elif ".source_delta." in name:
            group = "typed_source_patch"
        elif ".target_delta." in name:
            group = "typed_target_patch"
        elif ".role_embedding" in name:
            group = "typed_role"
        elif ".action_plan_conditioner_v1.injection.projections." in name:
            group = "injection"
            match = re.search(r"\.projections\.(\d+)\.(?:weight|bias)\Z", name)
            if match is None:
                fail(f"block injection parameter name differs: {name}")
            block_index = int(match.group(1))
            if block_index not in injection_heads:
                fail("block injection index exceeds exact30")
            injection_heads[block_index] += norm * norm
        elif ".action_plan_conditioner_v1.predictor." in name:
            group = "predictor"
            suffix = name.split(".predictor.", 1)[1]
            if suffix.startswith(("source_projection", "source_norm")):
                family = "source"
            elif suffix.startswith(("instruction_projection", "instruction_norm")):
                family = "instruction"
            elif suffix.startswith(
                ("actor_object_queries", "phase_queries", "global_query")
            ):
                family = "queries"
            elif suffix.startswith("blocks."):
                family = "blocks"
            else:
                family = "outputs"
            predictor_families[family] += norm * norm
        else:
            fail(f"unclassified staged gradient: {name}")
        groups.setdefault(group, 0.0)
        groups[group] += norm * norm
        rows.append({"name": name, "norm": norm, "positive": norm > 0.0})
    group_norms = {key: math.sqrt(value) for key, value in groups.items()}
    required_groups = {
        "lora_B",
        "typed_source_patch",
        "typed_target_patch",
        "typed_role",
        "injection",
    }
    if completed_step == 2:
        required_groups.update(("lora_A", "predictor"))
    if any(group_norms.get(name, 0.0) <= 0.0 for name in required_groups):
        fail(f"step {completed_step} staged gradient group is silent: {group_norms}")
    if completed_step == 1 and group_norms.get("predictor") != 0.0:
        fail("step-one predictor core must remain exactly zero behind zero-init heads")
    injection_norms = {
        str(index): math.sqrt(value) for index, value in injection_heads.items()
    }
    if any(value <= 0.0 for value in injection_norms.values()):
        fail("all exact30 zero-init projection heads must receive bootstrap gradient")
    predictor_norms = {
        name: math.sqrt(value) for name, value in predictor_families.items()
    }
    if completed_step == 1 and any(value != 0.0 for value in predictor_norms.values()):
        fail("step-one predictor family gradient must be exact zero")
    if completed_step == 2 and any(value <= 0.0 for value in predictor_norms.values()):
        fail(f"step-two predictor family coverage differs: {predictor_norms}")
    return {
        "completed_step": completed_step,
        "staged_contract": (
            "bootstrap_projection_heads_and_lora_B"
            if completed_step == 1
            else "predictor_core_and_lora_A_B_after_bootstrap"
        ),
        "group_norms": group_norms,
        "injection_head_norms": injection_norms,
        "predictor_family_norms": predictor_norms,
        "per_parameter": rows,
        "per_parameter_sha256": object_sha256(rows),
        "lora_affines": lora_affine_gradient_audit(
            named, specs, completed_step=completed_step
        ),
    }


def per_rank_memory(device: Any, distributed: Any) -> Mapping[str, Any]:
    import torch

    gib = float(1024**3)
    return {
        "world_rank": distributed.rank,
        "dp_arm": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "device": torch.cuda.get_device_name(device),
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "step_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "step_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--checkpoint-content-manifest", required=True)
    value.add_argument("--preprocessed-parquet-dir", required=True)
    value.add_argument("--dataset-summary", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--max-steps", type=int, choices=(MAX_STEPS,), default=MAX_STEPS)
    value.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    value.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    value.add_argument("--seed", type=int, default=DEFAULT_SEED)
    value.add_argument("--expected-bernini-commit", default=BERNINI_COMMIT)
    value.add_argument("--expected-veomni-commit", default=VEOMNI_COMMIT)
    value.add_argument("--expected-checkpoint-tree-sha256", default=CHECKPOINT_TREE_SHA256)
    value.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    value.add_argument("--workspace-base-revision", required=True)
    value.add_argument("--expected-runner-source-sha256", required=True)
    value.add_argument("--release-manifest", required=True)
    value.add_argument("--expected-release-manifest-sha256", required=True)
    value.add_argument("--ack-pre-d0-engineering-only", action="store_true")
    value.add_argument("--ack-legacy-target-quality-unqualified", action="store_true")
    value.add_argument("--ack-no-d0-or-scientific-claim", action="store_true")
    value.add_argument("--ack-fresh-base-disposable", action="store_true")
    return value


def validate_args(args: argparse.Namespace) -> None:
    acknowledgements = (
        args.ack_pre_d0_engineering_only,
        args.ack_legacy_target_quality_unqualified,
        args.ack_no_d0_or_scientific_claim,
        args.ack_fresh_base_disposable,
    )
    if not all(value is True for value in acknowledgements):
        fail("all PRE_D0 experimental acknowledgements are mandatory")
    if args.max_steps != MAX_STEPS:
        fail("PRE_D0 engineering runner accepts only max_steps=2")
    if args.learning_rate != DEFAULT_LR or args.max_grad_norm != DEFAULT_MAX_GRAD_NORM:
        fail("two-step optimizer hyperparameters are fixed")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        fail("seed differs")
    if (
        args.expected_bernini_commit != BERNINI_COMMIT
        or args.expected_veomni_commit != VEOMNI_COMMIT
        or args.expected_checkpoint_tree_sha256 != CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        fail("pinned source/model identity differs")
    if _SHA1.fullmatch(args.workspace_base_revision) is None:
        fail("workspace base revision must be one lowercase full SHA-1")
    if (
        _SHA256.fullmatch(args.expected_runner_source_sha256) is None
        or _SHA256.fullmatch(args.expected_release_manifest_sha256) is None
    ):
        fail("runner/release manifest SHAs must be lowercase full SHA-256")
    output = Path(args.output).expanduser()
    if (
        not output.is_absolute()
        or output.exists()
        or output.is_symlink()
        or "pre_d0_engineering" not in output.name.lower()
    ):
        fail("output must be one fresh absolute PRE_D0_ENGINEERING path")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    validate_args(args)

    release_closure = validate_release_manifest(
        Path(args.release_manifest),
        expected_sha256=args.expected_release_manifest_sha256,
    )
    if file_sha256(Path(__file__).resolve()) != args.expected_runner_source_sha256:
        fail("frozen runner source identity differs")

    import packed_preservation_lora_v2 as core
    import packed_preservation_release_v2 as release_contract
    import source_self_runtime as runtime
    import train_lora as legacy

    predictor_source = METHOD_ROOT / "action_plan_predictor_v1.py"
    if (
        not predictor_source.is_file()
        or file_sha256(predictor_source) != ACTION_PLAN_PREDICTOR_SOURCE_SHA256
    ):
        fail("ActionPlanPredictorV1 source identity differs from the sealed ABI")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=BERNINI_COMMIT,
                expected_veomni_commit=VEOMNI_COMMIT,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise PreD0EngineeringError(str(error)) from error
    if (
        transformer_config.get("num_layers") != TRANSFORMER_BLOCKS
        or transformer_config.get("attention_head_dim") != 128
        or transformer_config.get("num_attention_heads") != 12
    ):
        fail("Bernini-R 1.3B transformer geometry differs")
    checkpoint_manifest = Path(args.checkpoint_content_manifest).resolve(strict=True)
    if file_sha256(checkpoint_manifest) != CHECKPOINT_CONTENT_MANIFEST_SHA256:
        fail("checkpoint content manifest SHA differs")
    checkpoint_content = release_contract.validate_checkpoint_content(
        checkpoint,
        checkpoint_manifest,
        expected_manifest_sha256=CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    import torch.utils.checkpoint as torch_checkpoint
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    import action_plan_predictor_v1 as action_plan_module
    from action_plan_predictor_v1 import (
        ActionPlanConditionerV1,
        ActionPlanPredictorConfig,
        exact_parameter_inventory as action_parameter_inventory,
        exact_state_dict_abi as action_state_dict_abi,
        expected_conditioner_parameter_count,
    )
    import clean_source_visual_context_stage_b_contract_v1 as schedule_contract
    import inference_sigma_strata as sigma_strata_module

    if getattr(schedule_contract, "exact40", None) is not sigma_strata_module:
        fail("schedule contract transitive sigma-strata module ownership differs")
    imported_release_modules = validate_imported_release_modules(
        release_closure,
        {
            "action_plan_predictor_v1.py": action_plan_module,
            "clean_source_visual_context_stage_b_contract_v1.py": schedule_contract,
            "inference_sigma_strata.py": sigma_strata_module,
            "packed_preservation_lora_v2.py": core,
            "packed_preservation_release_v2.py": release_contract,
            "source_self_runtime.py": runtime,
            "train_action_edit_large_lora_0817_v1.py": sys.modules[__name__],
            "train_lora.py": legacy,
        },
    )

    distributed = runtime.distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.local_world_size != WORLD_SIZE
        or distributed.topology.dp_size != DP_SIZE
        or distributed.topology.sp_size != SP_SIZE
    ):
        fail("PRE_D0 runner requires one-node WORLD8 DP2xSP4")
    device = runtime.initialise_distributed(distributed)
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=SP_SIZE)
    )
    seed_everything(args.seed)
    if not callable(getattr(torch_checkpoint, "set_checkpoint_early_stop", None)):
        fail("PyTorch lacks non-reentrant checkpoint early-stop control")

    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    runtime.digest_consensus(
        dataset.signature,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="legacy full644 Parquet signature",
    )
    validation_envelope: list[Any] = [None]
    if distributed.rank == 0:
        try:
            summary_receipt = legacy.validate_preprocessed_dataset_summary(
                args.dataset_summary,
                dataset,
                allow_incomplete=False,
                allow_reward_selected_synthetic_targets=False,
            )
            summary_raw = legacy._read_json(
                Path(args.dataset_summary).expanduser().resolve(strict=True)
            )
            experimental_authority = validate_experimental_authority(summary_raw)
            strict_rows = build_strict_catalog_from_parquet(dataset, legacy=legacy)
            selected = strict_two_step_schedule(strict_rows)
            selected_unique_rows = len({row["iid"] for row in selected})
            global_strict_max_tokens = max(
                row["tokens_per_role"] for row in strict_rows
            )
            global_strict_max_rows = tuple(
                row
                for row in strict_rows
                if row["tokens_per_role"] == global_strict_max_tokens
            )
            global_strict_max_shapes = sorted(
                {
                    tuple(int(value) for value in row["posterior_parameter_shape"])
                    for row in global_strict_max_rows
                }
            )
            validation_envelope[0] = {
                "ok": True,
                "summary_receipt": summary_receipt,
                "experimental_authority": experimental_authority,
                "strict_catalog_digest": object_sha256(list(strict_rows)),
                "strict_catalog_count": len(strict_rows),
                "selected": list(selected),
                "selected_digest": object_sha256(list(selected)),
                "selected_schedule_records": len(selected),
                "selected_unique_rows": selected_unique_rows,
                "selected_rows_repeated": selected_unique_rows < len(selected),
                "global_strict_max_tokens_per_role": global_strict_max_tokens,
                "global_strict_max_patches_per_phase": (
                    global_strict_max_tokens // LATENT_PHASES
                ),
                "global_strict_max_tier_rows": len(global_strict_max_rows),
                "global_strict_max_tier_posterior_parameter_shapes": [
                    list(shape) for shape in global_strict_max_shapes
                ],
            }
        except Exception as error:
            validation_envelope[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(validation_envelope, src=0, group=parallel.world_group)
    validation = validation_envelope[0]
    if not isinstance(validation, Mapping) or validation.get("ok") is not True:
        fail(f"legacy PRE_D0 dataset validation failed: {validation!r}")
    selected = tuple(validation["selected"])
    if object_sha256(list(selected)) != validation["selected_digest"]:
        fail("broadcast two-step strict schedule digest differs")
    broadcast_unique = tuple(
        sorted(
            {str(row["iid"]): row for row in selected}.values(),
            key=lambda row: (str(row["iid"]), int(row["row_index"])),
        )
    )
    expected_broadcast_schedule = tuple(
        broadcast_unique[index % len(broadcast_unique)]
        for index in range(MAX_STEPS * GLOBAL_BATCH)
    )
    if expected_broadcast_schedule != selected:
        fail("broadcast maximum-geometry cyclic schedule differs")
    broadcast_unique_count = len(broadcast_unique)
    if (
        len(selected) != MAX_STEPS * GLOBAL_BATCH
        or broadcast_unique_count != validation["selected_unique_rows"]
        or (broadcast_unique_count < len(selected))
        is not validation["selected_rows_repeated"]
        or broadcast_unique_count
        != min(
            MAX_STEPS * GLOBAL_BATCH,
            validation["global_strict_max_tier_rows"],
        )
        or broadcast_unique_count < DP_SIZE
        or any(
            row["tokens_per_role"]
            != validation["global_strict_max_tokens_per_role"]
            for row in selected
        )
    ):
        fail("broadcast maximum-geometry schedule cardinality differs")

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    with serialized_model_load():
        renderer = BerniniRendererModel(config)
        renderer.requires_grad_(False)
        renderer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "context_fn": action_route_checkpoint_context_fn,
            }
        )
        specs = core.select_projection_specs(renderer, LORA_SCOPE)
        model = get_peft_model(
            renderer,
            LoraConfig(
                r=core.LORA_RANK,
                lora_alpha=core.LORA_ALPHA,
                lora_dropout=0.0,
                bias="none",
                target_modules=[item.name for item in specs],
            ),
        )
        transformer = model.get_base_model().diff_dec.transformer
        checkpointing_installation = (
            validate_action_route_checkpointing_installation(transformer)
        )
        core.install_typed_patch_embedding(transformer)
        conditioner = ActionPlanConditionerV1(ActionPlanPredictorConfig())
        hook_handle = install_action_plan_hooks(transformer, conditioner)
        model.to(device)
    model.train()
    base_renderer = model.get_base_model()
    base_renderer.t5_text_encoder.eval()
    conditioner.config.require_formal_0817()
    if any(parameter.dtype != torch.float32 for parameter in conditioner.parameters()):
        fail("ActionPlanPredictorV1 and exact30 injection heads must remain FP32")
    named_trainable = exact_trainable_named_parameters(model, conditioner)
    trainable_count = sum(int(parameter.numel()) for _, parameter in named_trainable)
    expected_action_count = expected_conditioner_parameter_count(
        conditioner.config, renderer_hidden_width=HIDDEN_WIDTH
    )
    expected_trainable_count = (
        core.EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNTS[LORA_SCOPE]
        + expected_action_count
    )
    if trainable_count != expected_trainable_count:
        fail(
            "large-LoRA+typed+action trainable count differs: "
            f"{trainable_count} != {expected_trainable_count}"
        )
    if any(
        parameter.device != device or parameter.dtype != torch.float32
        for _, parameter in named_trainable
    ):
        fail("all PRE_D0 trainables must be FP32 on the local accelerator")
    lora_installation = core.validate_lora_installation(model, specs)
    conditioner_state_abi = action_state_dict_abi(conditioner)
    if (
        conditioner_state_abi.get("abi_sha256")
        != ACTION_PLAN_CONDITIONER_STATE_ABI_SHA256
    ):
        fail("formal ActionPlanConditionerV1 semantic state ABI differs")
    architecture = {
        "large_lora_typed": core.architecture_receipt(LORA_SCOPE, specs),
        "action_plan_parameter_inventory": action_parameter_inventory(conditioner),
        "action_plan_state_dict_abi": conditioner_state_abi,
        "action_plan_predictor_source_sha256": ACTION_PLAN_PREDICTOR_SOURCE_SHA256,
        "gradient_checkpointing": checkpointing_installation,
        "exact30_post_block_injection": True,
        "source_plan_input": "complete_pre_sp_clean_source_patch_grid_BTHWC",
        "instruction_plan_input": "complete_unpadded_contextual_T5_tokens",
        "target_plan_input": False,
    }
    inventory = list(trainable_inventory(named_trainable))
    inventory_sha = object_sha256(inventory)
    runtime.digest_consensus(
        inventory_sha,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="PRE_D0 exact trainable inventory",
    )
    runtime.synchronize_initial_parameters(named_trainable, parallel.world_group)

    # Each DP arm pre-encodes its eight genuine row instructions, then retires T5.
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    local_selected = selected[distributed.arm_index :: DP_SIZE]
    if len(local_selected) != MAX_STEPS * GRADIENT_ACCUMULATION:
        fail("DP arm strict-row schedule count differs")
    local_unique_selected = tuple(
        sorted(
            {int(row["row_index"]): row for row in local_selected}.values(),
            key=lambda row: (str(row["iid"]), int(row["row_index"])),
        )
    )
    row_cache: dict[int, Mapping[str, Any]] = {}
    text_conditions: dict[int, Mapping[str, Any]] = {}
    for selected_row in local_unique_selected:
        row_index = int(selected_row["row_index"])
        raw_row = dataset[row_index]
        if (
            raw_row.get("iid") != selected_row["iid"]
            or raw_row.get("strict_selection_gates_all_true") is not True
        ):
            fail("scheduled row is not the bound mechanically strict Parquet row")
        sample = legacy.sanitize_preprocessed_row(raw_row)
        legacy.validate_81_frame_latents(
            sample, expected_parameter_channels=2 * LATENT_CHANNELS
        )
        instruction = _instruction_from_sanitized(sample, legacy)
        tokenized = runtime.tokenize_generic_instruction(tokenizer, instruction, device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                text_lens, text_embs = base_renderer.get_t5_text_embeddings(
                    tokenized["input_ids"],
                    tokenized["attention_mask"],
                    tokenized["t5_input_lens"],
                )
        renderer_max_sequence_length = int(
            getattr(base_renderer, "max_sequence_length", 0)
        )
        if isinstance(text_embs, (list, tuple)):
            if len(text_embs) != 1:
                fail("official UMT5 operator view must contain one embedding")
            text_embs = text_embs[0]
        actual_length = int(tokenized["t5_input_lens"].item())
        if (
            renderer_max_sequence_length != 512
            or [int(item) for item in text_lens]
            != [renderer_max_sequence_length]
            or list(tokenized["t5_input_lens"].shape) != [1, 1]
            or tokenized["input_ids"].ndim != 2
            or list(tokenized["input_ids"].shape)
            != list(tokenized["attention_mask"].shape)
            or int(tokenized["input_ids"].shape[0]) != 1
            or actual_length != int(tokenized["input_ids"].shape[1])
            or not 0 < actual_length <= renderer_max_sequence_length
            or not isinstance(text_embs, torch.Tensor)
            or list(text_embs.shape)
            != [1, renderer_max_sequence_length, INSTRUCTION_WIDTH]
            or text_embs.dtype != torch.bfloat16
            or text_embs.requires_grad
            or not torch.is_inference(text_embs)
            or not bool(torch.isfinite(text_embs).all().item())
        ):
            fail("frozen UMT5 renderer/tokenizer length contract differs")
        # Clone outside inference_mode: trainable predictor Linear layers must
        # be allowed to save these otherwise-frozen values for backward.
        text_embs = materialize_training_text_embedding(text_embs)
        if isinstance(text_lens, torch.Tensor):
            text_lens = text_lens.detach().clone(memory_format=torch.contiguous_format)
            if torch.is_inference(text_lens) or text_lens.requires_grad:
                fail("renderer text lengths retained inference ownership")
        if actual_length < renderer_max_sequence_length and bool(
            torch.count_nonzero(text_embs[:, actual_length:, :]).item()
        ):
            fail("official UMT5 operator padding rows are not exact zero")
        # text_lens=[512] belongs only to Bernini's packed operator.  The
        # predictor consumes all and only the tokenizer-authenticated rows.
        instruction_tokens = canonical_instruction_tokens(
            text_embs, tokenized["t5_input_lens"]
        )
        if (
            torch.is_inference(instruction_tokens)
            or instruction_tokens.requires_grad
        ):
            fail("unpadded predictor instruction view retained inference ownership")
        row_cache[row_index] = sample
        text_conditions[row_index] = {
            "instruction": instruction,
            "instruction_sha256": hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest(),
            "text_lens": text_lens,
            "text_embs": text_embs,
            "instruction_tokens": instruction_tokens,
            "actual_context_tokens": int(instruction_tokens.shape[1]),
        }
    base_renderer.t5_text_encoder = None
    del tokenizer, tokenized
    gc.collect()
    torch.cuda.empty_cache()
    if base_renderer.t5_text_encoder is not None:
        fail("frozen T5 was not retired after per-row instruction encoding")

    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    if z_dim != LATENT_CHANNELS:
        fail("Wan VAE latent width differs")
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    output = Path(args.output)
    checkpoints = output / "checkpoints"
    if distributed.rank == 0:
        output.mkdir(mode=0o700)
        checkpoints.mkdir(mode=0o700)
    dist.barrier(group=parallel.world_group)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=DEFAULT_LR,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
    )

    common_checkpoint_metadata = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "promotable": False,
        "formal_d0_dataset": False,
        "scientific_claim_authorized": False,
        "target_quality_qualified_for_0817": False,
        "fresh_official_base": True,
        "resume_consumed": False,
        "optimizer": {
            "class": "torch.optim.AdamW",
            "topology": "engineering_equivalent_replicated_not_formal_sharded",
            "learning_rate": DEFAULT_LR,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "parameter_group_policy": (
                "LoRA_predictor_injection_typed_WD0; dense_group_absent"
            ),
            "scheduler": "constant_lr_no_scheduler_object",
        },
        "architecture": architecture,
        "lora_installation": lora_installation,
        "lora_rank": core.LORA_RANK,
        "lora_alpha": core.LORA_ALPHA,
        "lora_scope": LORA_SCOPE,
        "trainable_parameter_count": trainable_count,
        "trainable_inventory_sha256": inventory_sha,
        "official_bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
        "checkpoint_content": checkpoint_content,
        # The workspace revision is only a base identity: the new untracked
        # bytes are independently bound by the frozen release manifest and
        # direct runner/predictor hashes.
        "workspace_base_revision": args.workspace_base_revision,
        "release_closure": release_closure,
        "imported_release_modules": imported_release_modules,
        "method_source_file_sha256": args.expected_runner_source_sha256,
        "legacy_dataset_summary": validation["summary_receipt"],
        "legacy_experimental_authority": validation["experimental_authority"],
        "strict_catalog_digest": validation["strict_catalog_digest"],
        "selected_two_step_schedule_digest": validation["selected_digest"],
        "selected_schedule_records": validation["selected_schedule_records"],
        "selected_unique_rows": validation["selected_unique_rows"],
        "selected_rows_repeated": validation["selected_rows_repeated"],
        "global_strict_max_tokens_per_role": validation[
            "global_strict_max_tokens_per_role"
        ],
        "global_strict_max_patches_per_phase": validation[
            "global_strict_max_patches_per_phase"
        ],
        "global_strict_max_tier_rows": validation["global_strict_max_tier_rows"],
        "global_strict_max_tier_posterior_parameter_shapes": validation[
            "global_strict_max_tier_posterior_parameter_shapes"
        ],
    }
    checkpoint_records: list[Mapping[str, Any]] = []
    parameter_digests: dict[int, str] = {}
    initial_digest = tensor_digest(named_trainable)
    runtime.digest_consensus(
        initial_digest,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="PRE_D0 P0",
    )
    parameter_digests[0] = initial_digest
    world_runtime_state = gather_world_runtime_state(
        step=0,
        distributed=distributed,
        device=device,
        world_group=parallel.world_group,
    )
    if distributed.rank == 0:
        checkpoint_records.append(
            save_checkpoint(
                root=checkpoints,
                step=0,
                named=named_trainable,
                conditioner=conditioner,
                optimizer=optimizer,
                metadata={
                    **common_checkpoint_metadata,
                    "parameter_sha256": initial_digest,
                },
                world_runtime_state=world_runtime_state,
            )
        )
    dist.barrier(group=parallel.world_group)

    history: list[Mapping[str, Any]] = []
    lifetime_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    lifetime_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    started = time.monotonic()
    for step_zero in range(MAX_STEPS):
        completed_step = step_zero + 1
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        micro_records: list[Mapping[str, Any]] = []
        for coordinate in schedule_contract.coordinates_for_optimizer_step(step_zero):
            selected_row = schedule_row(
                selected,
                optimizer_step_zero_based=step_zero,
                microbatch_index=coordinate.microbatch_index,
                dp_arm=distributed.arm_index,
            )
            row_index = int(selected_row["row_index"])
            sample = row_cache[row_index]
            source, target = paired_posterior_modes(
                sample, mean=vae_mean, std=vae_std, legacy=legacy
            )
            expected_parameter_shape = tuple(
                int(value) for value in selected_row["posterior_parameter_shape"]
            )
            expected_mode_shape = (
                expected_parameter_shape[0],
                LATENT_CHANNELS,
                *expected_parameter_shape[2:],
            )
            if (
                tuple(source.shape) != expected_mode_shape
                or tuple(target.shape) != expected_mode_shape
                or LATENT_PHASES
                * (expected_mode_shape[3] // 2)
                * (expected_mode_shape[4] // 2)
                != selected_row["tokens_per_role"]
            ):
                fail("scheduled row differs from authenticated maximum geometry")
            logical_record = (
                step_zero * GLOBAL_BATCH
                + coordinate.microbatch_index * DP_SIZE
                + distributed.arm_index
            )
            noise_seed = deterministic_seed(
                args.seed,
                "paired-flow",
                step_zero,
                coordinate.microbatch_index,
                distributed.arm_index,
                selected_row["iid"],
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(noise_seed)
            epsilon = torch.randn(
                tuple(target.shape), generator=generator, dtype=torch.float32
            ).contiguous()
            packed = dict(
                prepare_paired_flow(
                    source=source,
                    target=target,
                    epsilon=epsilon,
                    coordinate=coordinate,
                    rope=rope,
                    device=device,
                )
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with core.packed_role_layout(
                    packed["source_tokens"], packed["target_tokens"]
                ):
                    embedded = (
                        transformer.patch_embedding(packed["input_patches"])
                        .flatten(1)
                        .unsqueeze(0)
                    )
            if tuple(embedded.shape) != (
                1,
                packed["total_tokens"],
                HIDDEN_WIDTH,
            ):
                fail("complete pre-SP packed embedding geometry differs")
            packed["embedded"] = embedded
            text = text_conditions[row_index]
            action_route = prepare_action_injection_route(
                conditioner=conditioner,
                embedded=embedded,
                packed=packed,
                instruction_tokens=text["instruction_tokens"],
            )
            local_route = ActionInjectionRoute(
                source_tokens=packed["source_tokens"],
                target_tokens=packed["target_tokens"],
                sequence_parallel_rank=distributed.sp_rank,
                sequence_parallel_size=SP_SIZE,
                plan=action_route,
                row_identity=str(selected_row["iid"]),
            )
            with activate_action_route(local_route):
                # Disable non-reentrant checkpoint early-stop so every
                # registered block hook is replayed completely and auditable.
                # Keep both this policy and the authenticated route live
                # through backward recomputation.
                with torch_checkpoint.set_checkpoint_early_stop(False):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        prediction = predict_target(
                            renderer=base_renderer,
                            packed=packed,
                            coordinate=coordinate,
                            text_lens=text["text_lens"],
                            text_embs=text["text_embs"],
                        )
                        raw_loss = torch.nn.functional.mse_loss(
                            prediction.float(), packed["target_velocity"].float()
                        )
                        scaled_loss = raw_loss / float(GRADIENT_ACCUMULATION)
                    if not runtime.world_all_true(
                        bool(torch.isfinite(scaled_loss.detach()).item()),
                        group=parallel.world_group,
                    ):
                        fail("non-finite PRE_D0 paired flow loss")
                    scaled_loss.backward()
                local_route.validate_forward_and_recompute_traversal()
            micro_records.append(
                {
                    "microbatch": coordinate.microbatch_index,
                    "logical_record": logical_record,
                    "row_index": row_index,
                    "iid": selected_row["iid"],
                    "mechanically_strict_selection": True,
                    "target_quality_qualified_for_0817": False,
                    "instruction": text["instruction"],
                    "instruction_sha256": text["instruction_sha256"],
                    "actual_context_tokens": text["actual_context_tokens"],
                    "noise_seed": noise_seed,
                    "schedule_index": coordinate.schedule_index,
                    "sigma": coordinate.sigma,
                    "loss": float(raw_loss.detach().item()),
                    "source_posterior_statistic": "mode",
                    "target_posterior_statistic": "mode",
                    "posterior_parameter_shape": list(expected_parameter_shape),
                    "tokens_per_role": selected_row["tokens_per_role"],
                    "global_strict_max_geometry_bootstrap": True,
                    "source_is_condition_only": True,
                    "target_enters_predictor": False,
                    "source_sha256": runtime.tensor_sha256(source),
                    "target_sha256": runtime.tensor_sha256(target),
                    "route": local_route.receipt(),
                    "conditioner_route_metadata_digest": action_route.metadata_digest,
                    "ownership_digest": action_route.ownership.digest,
                }
            )
            del (
                source,
                target,
                epsilon,
                packed,
                embedded,
                action_route,
                local_route,
                prediction,
                raw_loss,
                scaled_loss,
            )

        synchronized_norm = synchronize_gradients_bucketed(named_trainable, parallel)
        gradient_audit = staged_gradient_audit(
            named_trainable, specs, completed_step=completed_step
        )
        gradient_audit_sha = object_sha256(gradient_audit)
        runtime.digest_consensus(
            gradient_audit_sha,
            group=parallel.world_group,
            expected_count=WORLD_SIZE,
            label=f"PRE_D0 staged gradient audit step {completed_step}",
        )
        synchronized_gradient_sha = gradient_digest(named_trainable)
        runtime.digest_consensus(
            synchronized_gradient_sha,
            group=parallel.world_group,
            expected_count=WORLD_SIZE,
            label=f"PRE_D0 synchronized gradients step {completed_step}",
        )
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named_trainable], DEFAULT_MAX_GRAD_NORM
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        local_memory = per_rank_memory(device, distributed)
        memory_world: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(memory_world, local_memory, group=parallel.world_group)
        if [item["world_rank"] for item in memory_world] != list(range(WORLD_SIZE)):
            fail("per-rank step memory receipt closure differs")
        parameter_digest = tensor_digest(named_trainable)
        runtime.digest_consensus(
            parameter_digest,
            group=parallel.world_group,
            expected_count=WORLD_SIZE,
            label=f"PRE_D0 P{completed_step}",
        )
        if parameter_digest in parameter_digests.values():
            fail("optimizer update did not change the exact trainable state")
        parameter_digests[completed_step] = parameter_digest
        local_step = {
            "world_rank": distributed.rank,
            "dp_arm": distributed.arm_index,
            "step": completed_step,
            "microbatches": micro_records,
            "synchronized_gradient_norm": synchronized_norm,
            "gradient_audit_sha256": gradient_audit_sha,
            "synchronized_gradient_sha256": synchronized_gradient_sha,
            "parameter_sha256": parameter_digest,
            "memory_world8": memory_world,
        }
        gathered_steps: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_steps, local_step, group=parallel.world_group)
        for arm_start in (0, SP_SIZE):
            signatures = []
            for rank in range(arm_start, arm_start + SP_SIZE):
                signatures.append(
                    [
                        (
                            row["logical_record"],
                            row["iid"],
                            row["noise_seed"],
                            row["instruction_sha256"],
                            row["loss"],
                        )
                        for row in gathered_steps[rank]["microbatches"]
                    ]
                )
            if len({object_sha256(value) for value in signatures}) != 1:
                fail(f"step {completed_step} SP4 paired-row evidence differs")
        leaders = [gathered_steps[0], gathered_steps[SP_SIZE]]
        observed_logical = sorted(
            row["logical_record"]
            for leader in leaders
            for row in leader["microbatches"]
        )
        expected_logical = list(
            range(step_zero * GLOBAL_BATCH, completed_step * GLOBAL_BATCH)
        )
        if observed_logical != expected_logical:
            fail("two-step DP2 logical-record closure differs")
        step_record = {
            "authority": AUTHORITY,
            "step": completed_step,
            "optimizer_step_executed": True,
            "logical_records": [
                row for leader in leaders for row in leader["microbatches"]
            ],
            "synchronized_gradient_norm": synchronized_norm,
            "gradient_audit": gradient_audit,
            "gradient_audit_sha256": gradient_audit_sha,
            "synchronized_gradient_sha256": synchronized_gradient_sha,
            "parameter_sha256": parameter_digest,
            "memory_world8": memory_world,
        }
        if distributed.rank == 0:
            history.append(step_record)
            print(json.dumps(step_record, sort_keys=True), flush=True)
            with (output / "history.jsonl").open("ab") as handle:
                handle.write(canonical_json_bytes(step_record) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        world_runtime_state = gather_world_runtime_state(
            step=completed_step,
            distributed=distributed,
            device=device,
            world_group=parallel.world_group,
        )
        if distributed.rank == 0:
            checkpoint_records.append(
                save_checkpoint(
                    root=checkpoints,
                    step=completed_step,
                    named=named_trainable,
                    conditioner=conditioner,
                    optimizer=optimizer,
                    metadata={
                        **common_checkpoint_metadata,
                        "parameter_sha256": parameter_digest,
                        "gradient_audit_sha256": gradient_audit_sha,
                        "synchronized_gradient_sha256": synchronized_gradient_sha,
                    },
                    world_runtime_state=world_runtime_state,
                )
            )
        dist.barrier(group=parallel.world_group)
        lifetime_peak_allocated = max(
            lifetime_peak_allocated, int(torch.cuda.max_memory_allocated(device))
        )
        lifetime_peak_reserved = max(
            lifetime_peak_reserved, int(torch.cuda.max_memory_reserved(device))
        )

    if tuple(sorted(parameter_digests)) != (0, 1, 2) or len(
        set(parameter_digests.values())
    ) != 3:
        fail("two-step P0/P1/P2 state closure differs")
    gib = float(1024**3)
    local_lifetime_memory = {
        "world_rank": distributed.rank,
        "dp_arm": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "lifetime_peak_allocated_gib": lifetime_peak_allocated / gib,
        "lifetime_peak_reserved_gib": lifetime_peak_reserved / gib,
        "covers_serialized_model_load_T5_and_two_updates": True,
    }
    lifetime_memory_world: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        lifetime_memory_world, local_lifetime_memory, group=parallel.world_group
    )
    if [item["world_rank"] for item in lifetime_memory_world] != list(range(WORLD_SIZE)):
        fail("lifetime WORLD8 memory receipt closure differs")
    local_terminal_hooks = {
        "world_rank": distributed.rank,
        "block_ids_unchanged": tuple(hook_handle.block_ids)
        == tuple(id(block) for block in transformer.blocks),
        "conditioner_owner_unchanged": getattr(
            transformer, "action_plan_conditioner_v1", None
        )
        is conditioner,
        "handle_count_exact30": len(hook_handle.handles) == TRANSFORMER_BLOCKS,
        "one_registered_forward_hook_per_block": all(
            len(getattr(block, "_forward_hooks", {})) == 1
            for block in transformer.blocks
        ),
        "hooks_not_restored": hook_handle.restored is False,
        "no_active_route_at_terminal": (
            _ACTIVE_ACTION_ROUTE.get() is None
            and _ACTIVE_ACTION_CHECKPOINT_BINDING.get() is None
            and _ACTION_ROUTE_LEASE_OWNER.is_empty()
        ),
    }
    terminal_hooks_world: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        terminal_hooks_world, local_terminal_hooks, group=parallel.world_group
    )
    if (
        [row.get("world_rank") for row in terminal_hooks_world]
        != list(range(WORLD_SIZE))
        or any(
            value is not True
            for row in terminal_hooks_world
            for key, value in row.items()
            if key != "world_rank"
        )
    ):
        fail("terminal exact30 hook WORLD8 ownership consensus differs")
    terminal_hook_consensus_sha = object_sha256(terminal_hooks_world)
    # No complete receipt can become visible until every rank has closed its
    # terminal hook ownership and agreed on the exact WORLD8 evidence.
    dist.barrier(group=parallel.world_group)
    if distributed.rank == 0:
        unsigned = {
            "schema_version": RECEIPT_SCHEMA,
            "method": METHOD,
            "authority": AUTHORITY,
            "status": "complete_pre_d0_two_update_engineering_smoke",
            "complete": True,
            "promotable": False,
            "formal_training_started": False,
            "counts_as_d0": False,
            "counts_as_d1": False,
            "counts_as_d2": False,
            "scientific_claim_authorized": False,
            "action_quality_claim_authorized": False,
            "target_quality_status": (
                "legacy_mechanical_strict_gate_only_pending_0817_rebuild"
            ),
            "optimizer_steps": MAX_STEPS,
            "fresh_official_base": True,
            "resume_consumed": False,
            "optimizer": {
                "class": "torch.optim.AdamW",
                "topology": "engineering_equivalent_replicated_not_formal_sharded",
                "fresh_state": True,
                "learning_rate": DEFAULT_LR,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "parameter_group_policy": (
                    "LoRA_predictor_injection_typed_WD0; dense_group_absent"
                ),
                "max_gradient_norm": DEFAULT_MAX_GRAD_NORM,
                "scheduler": "constant_lr_no_scheduler_object",
            },
            "architecture": architecture,
            "lora_installation": lora_installation,
            "trainable_parameter_count": trainable_count,
            "trainable_inventory_sha256": inventory_sha,
            "parameter_digests": {
                str(key): value for key, value in parameter_digests.items()
            },
            "checkpoint_steps": [0, 1, 2],
            "checkpoints": checkpoint_records,
            "all_checkpoints_rank0_full_trainable_optimizer_roundtrip_reloaded": all(
                row["rank0_full_trainable_optimizer_roundtrip_reload_verified"]
                for row in checkpoint_records
            ),
            "all_checkpoints_all8_runtime_state_bytes_persisted": all(
                row["all8_runtime_state_bytes_persisted_verified"]
                for row in checkpoint_records
            ),
            "all_checkpoints_rank0_runtime_state_roundtrip_reloaded": all(
                row["rank0_runtime_state_roundtrip_reload_verified"]
                for row in checkpoint_records
            ),
            "provenance": {
                "workspace_base_revision_only": args.workspace_base_revision,
                "workspace_revision_contains_new_files_claimed": False,
                "release_closure": release_closure,
                "imported_release_modules": imported_release_modules,
                "runner_source_sha256": args.expected_runner_source_sha256,
                "predictor_source_sha256": ACTION_PLAN_PREDICTOR_SOURCE_SHA256,
            },
            "history_steps": len(history),
            "dataset": {
                "legacy_full644_parquet": True,
                "formal_0817_manifest_consumed": False,
                "summary": validation["summary_receipt"],
                "experimental_authority": validation["experimental_authority"],
                "mechanically_strict_rows": validation["strict_catalog_count"],
                "strict_catalog_digest": validation["strict_catalog_digest"],
                "selection_scope": "global_maximum_within_mechanically_strict_rows",
                "selected_schedule_records": validation[
                    "selected_schedule_records"
                ],
                "selected_unique_rows": validation["selected_unique_rows"],
                "selected_rows_repeated": validation["selected_rows_repeated"],
                "global_strict_max_tokens_per_role": validation[
                    "global_strict_max_tokens_per_role"
                ],
                "global_strict_max_patches_per_phase": validation[
                    "global_strict_max_patches_per_phase"
                ],
                "global_strict_max_tier_rows": validation[
                    "global_strict_max_tier_rows"
                ],
                "global_strict_max_tier_posterior_parameter_shapes": validation[
                    "global_strict_max_tier_posterior_parameter_shapes"
                ],
                "selected_schedule_digest": validation["selected_digest"],
                "effective_scientific_sample_size_claimed": False,
                "teacher_anchor_qualification_claimed": False,
            },
            "supervision": {
                "source": "normalized_source_VAE_posterior_mode_condition_only",
                "instruction": "row_specific_complete_unpadded_T5_context",
                "clean_flow_target": "normalized_target_VAE_posterior_mode",
                "predictor_inputs": ["complete_clean_source_patch_grid", "instruction"],
                "predictor_target_access": False,
                "injection": "target_suffix_only_after_each_exact30_block",
            },
            "distributed": {
                "world_size": WORLD_SIZE,
                "dp_size": DP_SIZE,
                "sp_size": SP_SIZE,
                "gradient_accumulation": GRADIENT_ACCUMULATION,
                "global_batch": GLOBAL_BATCH,
                "gradient_sync": "SP4_mean_then_DP2_mean",
                "pre_sp_complete_source_predictor": True,
                "sp4_global_index_to_phase_bridge": True,
                "source_and_padding_bit_exact_under_injection": True,
                "checkpoint_forward_and_recompute_calls_per_block": 2,
            },
            "lifetime_memory_world8": lifetime_memory_world,
            "terminal_exact30_hook_world8": terminal_hooks_world,
            "terminal_exact30_hook_world8_sha256": terminal_hook_consensus_sha,
            "terminal_world8_consensus_precedes_receipt_publication": True,
            "elapsed_seconds": time.monotonic() - started,
            "parent_allocation_released": False,
        }
        receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        atomic_create_json(output / "receipt.json", receipt)
        print(
            json.dumps(
                {
                    "complete": True,
                    "authority": AUTHORITY,
                    "output": str(output),
                    "optimizer_steps": MAX_STEPS,
                    "promotable": False,
                    "parent_allocation_released": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=parallel.world_group)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
