#!/usr/bin/env python3
"""Matched Stage-B source-carrier probes and raw-conditional exact40 rollout.

This runtime consumes the sealed materialized posterior row used by training;
it never re-encodes the source video.  Its primary evidence is four stateless
queries at exact40 indices 16/29/35/38.  Each query reconstructs the training
coordinate exactly: target and registered appearance-corrupted donor use one
shared epsilon and sigma, while the three independently encoded clean RGB
references retain their sealed per-IID order.  The predicted clean target is
``x_sigma - sigma * v`` and is decoded independently.

The optional full40 mode runs a fresh UniPC scheduler without re-anchoring the
evolved target.  Every target state, including the initial exact epsilon at
the first positive sigma, is explicitly not claimed to satisfy the stateless
training target equation.  Routing the adapter at all 40 cells is a stronger
extrapolation ablation.  Neither mode is inversion.

The frozen-base arm installs the identical role/Q/O wrappers and zeroes every
adapter tensor.  The trained arm strictly loads the Stage-B safetensors and
its sibling training receipt.  Both arms therefore use the same custom pack,
source IDs, prompt, seed, scheduler and route mask.  An optional anchor action
video is hash-bound for display only and is never a model condition.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as legacy_infer  # noqa: E402
import inference_sigma_strata as exact40  # noqa: E402
import source_noised_ladder_v1 as ladder  # noqa: E402
import source_self_runtime as runtime  # noqa: E402
import train_lora as legacy_train  # noqa: E402
import train_source_noised_carrier_strata_v1 as stage_b  # noqa: E402

# These modules import torch at module load.  Keep them lazy so receipt-pair
# verification remains a genuinely CPU/model-free operation.
role: Any = None
dataset_core: Any = None


RECEIPT_SCHEMA = "bernini-source-noised-carrier-stage-b-inference-receipt-v1"
PAIR_RECEIPT_SCHEMA = "bernini-source-noised-carrier-stage-b-pair-receipt-v1"
FPS = 25.0
SP_SIZE = 4
INFERENCE_SEED = 2026081401
EXPECTED_SEALED_OUTPUT_HW = (592, 400)
EXPECTED_SEALED_DECODE_LATENT_SHAPE = (1, 16, 21, 74, 50)
ARMS = ("frozen_base", "trained")
MODES = (
    "registered-probes",
    "full40-evolved-target-all40-route-extrapolation",
)
STYLE_IDS = (1, 2)
REGISTERED_IID_STYLE = {
    "0014a41e55e44670": 1,
    "00435ad621c44fac": 2,
}
REGISTERED_IID_REFERENCE_ORDER = {
    "0014a41e55e44670": [0, 40, 80],
    "00435ad621c44fac": [0, 80, 40],
}
REGISTERED_IID_SEALED_ROW_AUTHORITY = {
    "0014a41e55e44670": {
        "style_id": 1,
        "row_digest": "327f69ed7889a1653da489b97a0964d23e7163508bb117b8b7b7432e695742b4",
        "source_video_sha256": "b0255970cdbb42375cd783e8f2ab9b8099d5f02ec96f62085be77e24eb5f2437",
        "clean_posterior_blob_sha256": "d01b509cac730ee5cd5af6fbbcc36e1ae2b931fc9982163e642f6b04d641669c",
        "style_posterior_blob_sha256": "d29b739cbfe101defc107f78503cbf1120eafbebf270a889e6cbbfe2aed96135",
        "reference_order": [0, 40, 80],
        "reference_posterior_blob_sha256_in_order": [
            "1818d1351cd563a9f9ebf45e82100e706efb0ea56d0f7b6d38a6bb6c793d13fc",
            "98d93c65045733e59a203758de78b14e084a577417bd9095fe4574da548c1ef1",
            "78ba90e4dd6fd93185ae48acd4d8c395c3344fda25a850ab8401068d168b89dc",
        ],
    },
    "00435ad621c44fac": {
        "style_id": 2,
        "row_digest": "1a04a38e06060d9ad29790a1185705ac8ba7401a30e8c5f738b1dcc0cecf6b44",
        "source_video_sha256": "b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1",
        "clean_posterior_blob_sha256": "f9135728c18d32d5304bf2d7f8f5e9fbcda08421707d78331526ea62627edb39",
        "style_posterior_blob_sha256": "b5892364716808b690289338a64fe1d8182c3b74621fad6ba1921a35f602cfab",
        "reference_order": [0, 80, 40],
        "reference_posterior_blob_sha256_in_order": [
            "ece7e686348528604c86839d1bf1dad3eed9a024b6834808eacc229e494aa886",
            "56b45906611576eaef0bb3e416ff876d6917e3721437c3405c799e02d292afc3",
            "5eb4d7301aa61ff40a187bd29ba86d4a5bb86ee6c8b60e0bc9f0d79e0ee4cf4f",
        ],
    },
}
EXPECTED_DATASET_MATERIALIZATION_SPEC_SHA256 = (
    "62468b24d4a57ec03d42ce8c006a707cbcf56588ef62d10632089eb5ad457920"
)
EXPECTED_DATASET_PARQUET_SHA256 = (
    "77d89b3ec2e563f624bab62451b49b616ffa7f7890db6105c4458617aac0d106"
)
EXPECTED_DATASET_RECEIPT_SHA256 = (
    "6ed77cf7d98391c2074e5938ab50d0688d457bddfd688f9a5825d455447a20bb"
)
EXPECTED_DATASET_RECEIPT_DIGEST = (
    "12ede44ebab03215e19574967a9afec3c634f246f2cfd2634a48ce0e3dea8738"
)
EXPECTED_STAGE_B_ADAPTER_SHA256 = (
    "f85de3518ec88ac86a33fcb574c328ae3bca581ea3a1f86a648ef051b14ec16c"
)
EXPECTED_STAGE_B_TRAINING_RECEIPT_SHA256 = (
    "7c61d23d00e442a7fada318ade279db1131db57da88d068233325f004cd1dca9"
)
EXPECTED_STAGE_B_TRAINING_RECEIPT_DIGEST = (
    "3255d13425f91986216ee40a7777a58665fc266ea5d489909c64682a7f1fd026"
)
EXPECTED_STAGE_B_OPTIMIZER_SHA256 = (
    "cf9113c74dce8f3f0c1e4a8c2a93e021f46e4346926c3d6009260b29330c8161"
)
EXPECTED_STAGE_B_HISTORY_SHA256 = (
    "fdac2070d5ed64028c18eef7453528c5140d81e70305d432319d9c59fb41e92a"
)
EXPECTED_STAGE_B_FINAL_PARAMETER_SHA256 = (
    "710e1715abfa993e7fead4dbd3740de199f0418da5811391a93f5ed22c929f85"
)
OUTPUT_FILES_BY_MODE = {
    "registered-probes": (
        "probe_index16.mp4",
        "probe_index29.mp4",
        "probe_index35.mp4",
        "probe_index38.mp4",
    ),
    "full40-evolved-target-all40-route-extrapolation": ("full40.mp4",),
}
INFERENCE_RECEIPT_KEYS = {
    "schema_version",
    "complete",
    "arm",
    "mode_contract",
    "source",
    "dataset",
    "tensor_binding",
    "instruction",
    "visual_pack",
    "execution_counts",
    "runtime_schedule_audit",
    "compute_consensus",
    "preforward_input_consensus",
    "adapter",
    "adapter_runtime_binding",
    "records",
    "outputs",
    "artifacts",
    "anchor_action_display",
    "model",
    "runtime",
    "method_source_revision",
    "method_source_archive_sha256",
    "inversion_claimed",
    "method_success_claimed",
    "scientific_claim_authorized",
    "receipt_digest",
}
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CHECKPOINT_CONTENT_FILE_COUNT = 23
EXPECTED_SOURCE_SELF_RUNTIME_SHA256 = (
    "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f"
)


class StageBInferenceError(RuntimeError):
    """Raised before ambiguous Stage-B inference or publication."""


def fail(message: str) -> NoReturn:
    raise StageBInferenceError(message)


def validate_vae_decoded_clip(
    decoded: Any,
    *,
    expected_height: int,
    expected_width: int,
) -> dict[str, Any]:
    """Validate Bernini's normalized numpy decode without changing its value."""

    import numpy as np

    if (
        type(expected_height) is not int
        or type(expected_width) is not int
        or expected_height <= 0
        or expected_width <= 0
    ):
        fail("VAE decoded output geometry authority differs")
    expected_shape = (81, expected_height, expected_width, 3)
    if (
        not isinstance(decoded, np.ndarray)
        or decoded.ndim != 4
        or tuple(int(item) for item in decoded.shape) != expected_shape
    ):
        fail("VAE decode differs from numpy [81,H,W,3] contract")
    if decoded.dtype.kind != "f" or decoded.dtype.itemsize not in (2, 4, 8):
        fail("VAE decode dtype differs from normalized floating numpy contract")
    if not bool(np.isfinite(decoded).all()):
        fail("VAE decoded numpy clip is non-finite")
    value_min = float(decoded.min())
    value_max = float(decoded.max())
    if value_min < 0.0 or value_max > 1.0:
        fail("VAE decode differs from normalized [0,1] save_output contract")
    return {
        "array_type": "numpy.ndarray",
        "shape": list(expected_shape),
        "dtype": str(decoded.dtype),
        "finite": True,
        "normalized_zero_one": True,
        "value_min": value_min,
        "value_max": value_max,
    }


def save_validated_vae_decoded_clip(
    decoded: Any,
    *,
    output_path: Path,
    expected_height: int,
    expected_width: int,
    fps: int,
    save_output_fn: Any,
) -> dict[str, Any]:
    """Validate then pass the original normalized numpy clip to Bernini."""

    audit = validate_vae_decoded_clip(
        decoded,
        expected_height=expected_height,
        expected_width=expected_width,
    )
    save_output_fn(decoded, str(output_path), fps=fps)
    return audit


def validate_runtime_dependency() -> dict[str, str]:
    path = Path(runtime.__file__).resolve(strict=True)
    actual = file_sha256(path)
    if actual != EXPECTED_SOURCE_SELF_RUNTIME_SHA256:
        fail(
            "source_self_runtime dependency differs from the audited Stage-B "
            f"inference closure: {actual}"
        )
    return {"path": str(path), "sha256": actual}


def _load_heavy_runtime_modules() -> tuple[Any, Any]:
    global role, dataset_core
    if role is None:
        import source_self_role_repaint as loaded_role

        role = loaded_role
    if dataset_core is None:
        import train_source_self_role_repaint as loaded_dataset_core

        dataset_core = loaded_dataset_core
    return role, dataset_core


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
        raise StageBInferenceError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if type(value) is not str or pattern.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-{1 if length == 40 else 256}")
    return value


def _plain_absolute_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise StageBInferenceError(f"{label} is unavailable: {error}") from error
    if resolved != requested or not stat.S_ISREG(mode) or resolved.is_symlink():
        fail(f"{label} must be a canonical plain file")
    return resolved


def _plain_absolute_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise StageBInferenceError(f"{label} is unavailable: {error}") from error
    if resolved != requested or not stat.S_ISDIR(mode) or resolved.is_symlink():
        fail(f"{label} must be a canonical plain directory")
    return resolved


def _read_json_bound(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    expected = _require_sha(expected_sha256, length=64, label=f"{label} SHA")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after) or hashlib.sha256(raw).hexdigest() != expected:
        fail(f"{label} bytes changed or SHA differs")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StageBInferenceError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        fail(f"{label} root must be one object")
    return value


def _validate_embedded_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    _require_sha(declared, length=64, label=f"{label} receipt digest")
    if object_sha256(unsigned) != declared:
        fail(f"{label} embedded digest differs")
    return str(declared)


def _require_exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        fail(f"{label} key closure differs")
    return value


def _validate_canonical_digest(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} must be one object")
    unsigned = dict(value)
    declared = unsigned.pop("digest", None)
    if _require_sha(declared, length=64, label=f"{label} digest") != object_sha256(unsigned):
        fail(f"{label} digest differs")
    return value


def validate_checkpoint_content(
    checkpoint: Path,
    manifest_value: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify the exact non-cache checkpoint file closure and every SHA."""

    manifest = _plain_absolute_file(manifest_value, label="checkpoint content manifest")
    expected_manifest = _require_sha(
        expected_manifest_sha256, length=64, label="checkpoint content manifest SHA"
    )
    actual_manifest = file_sha256(manifest)
    if actual_manifest != expected_manifest:
        fail("checkpoint content manifest SHA differs")
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise StageBInferenceError("cannot read checkpoint content manifest") from error
    if len(lines) != CHECKPOINT_CONTENT_FILE_COUNT:
        fail("checkpoint content manifest file count differs")
    expected: dict[str, str] = {}
    pattern = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            fail("checkpoint manifest line is not canonical sha256sum syntax")
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            fail("checkpoint manifest contains an unsafe path")
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in expected:
            fail("checkpoint manifest contains an empty/duplicate path")
        expected[normalized] = digest
    actual_paths: set[str] = set()
    for path in checkpoint.rglob("*"):
        relative = path.relative_to(checkpoint)
        if ".cache" in relative.parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            fail("checkpoint contains a non-cache symlink")
        if stat.S_ISREG(mode):
            actual_paths.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            fail("checkpoint contains a non-regular filesystem entry")
    if actual_paths != set(expected):
        fail("checkpoint non-cache file set differs from pinned manifest")
    rows: list[dict[str, str]] = []
    for relative in sorted(expected):
        path = _plain_absolute_file(checkpoint / relative, label=f"checkpoint file {relative}")
        actual = file_sha256(path)
        if actual != expected[relative]:
            fail(f"checkpoint content hash differs: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": actual_manifest,
        "verified_file_count": len(rows),
        "every_non_cache_file_sha256_verified": True,
        "verified_entries_digest": object_sha256(rows),
    }


@dataclass(frozen=True)
class BoundStageBAdapter:
    root: Path
    adapter: Path
    receipt: Path
    adapter_sha256: str
    receipt_sha256: str
    receipt_digest: str
    final_parameter_sha256: str
    receipt_value: Mapping[str, Any]
    artifact_sha256: Mapping[str, str]
    training_noise_seeds: tuple[int, ...]
    training_epsilon_sha256: tuple[str, ...]


def resolve_stage_b_adapter(
    value: str | Path,
    *,
    expected_adapter_sha256: str,
    expected_receipt_sha256: str,
) -> BoundStageBAdapter:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail("adapter checkpoint must be an absolute non-symlink training directory")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise StageBInferenceError(f"adapter checkpoint is unavailable: {error}") from error
    if root != requested or not root.is_dir() or root.is_symlink():
        fail("adapter checkpoint must be a canonical plain directory")
    entries = {path.name: path for path in root.iterdir()}
    if set(entries) != {"adapter.safetensors", "optimizer.pt", "history.json", "receipt.json"}:
        fail("Stage-B training artifact closure differs")
    if any(not path.is_file() or path.is_symlink() for path in entries.values()):
        fail("Stage-B training directory contains a non-plain artifact")
    adapter_path = entries["adapter.safetensors"]
    receipt_path = entries["receipt.json"]
    adapter_sha = file_sha256(adapter_path)
    receipt_sha = file_sha256(receipt_path)
    if (
        _require_sha(expected_adapter_sha256, length=64, label="adapter SHA")
        != EXPECTED_STAGE_B_ADAPTER_SHA256
        or _require_sha(
            expected_receipt_sha256, length=64, label="training receipt SHA"
        ) != EXPECTED_STAGE_B_TRAINING_RECEIPT_SHA256
        or adapter_sha != EXPECTED_STAGE_B_ADAPTER_SHA256
        or receipt_sha != EXPECTED_STAGE_B_TRAINING_RECEIPT_SHA256
    ):
        fail("Stage-B adapter SHA differs")
    receipt = _read_json_bound(receipt_path, expected_receipt_sha256, label="Stage-B training receipt")
    digest = _validate_embedded_digest(receipt, label="Stage-B training")
    if digest != EXPECTED_STAGE_B_TRAINING_RECEIPT_DIGEST:
        fail("Stage-B training receipt digest differs from v1 authority")
    required = {
        "schema_version": stage_b.RUN_RECEIPT_SCHEMA,
        "method": stage_b.METHOD_NAME,
        "complete": True,
        "mode": stage_b.MODE,
        "optimizer_steps": stage_b.OPTIMIZER_STEPS,
        "positive_gradient_steps": stage_b.OPTIMIZER_STEPS,
        "exact40_schedule_sha256": stage_b.EXPECTED_EXACT40_SCHEDULE_SHA256,
        "registered_schedule_indices": list(stage_b.REGISTERED_SCHEDULE_INDICES),
        "matched_same-noise_carrier_inference_runtime_required": True,
        "existing_sigma1_conditional_base_loader_compatible": False,
        "base_frozen": True,
        "vae_frozen_and_absent_from_training_process": True,
        "pretext_training_only": True,
        "method_success_claimed": False,
    }
    if any(receipt.get(key) != expected for key, expected in required.items()):
        fail("Stage-B training receipt contract differs")
    artifacts = receipt.get("artifacts")
    forward = receipt.get("forward_noising")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != {"adapter.safetensors", "optimizer.pt", "history.json"}
        or artifacts.get("adapter.safetensors") != adapter_sha
        or not isinstance(forward, Mapping)
        or forward.get("same_epsilon_target_and_donor_required") is not True
        or forward.get("forward_noising_only") is not True
        or forward.get("inversion_claimed") is not False
    ):
        fail("Stage-B adapter/forward-noising provenance differs")
    artifact_sha256: dict[str, str] = {}
    for name in ("adapter.safetensors", "optimizer.pt", "history.json"):
        expected = _require_sha(artifacts.get(name), length=64, label=f"training {name} SHA")
        actual = file_sha256(entries[name])
        if actual != expected:
            fail(f"Stage-B training artifact hash differs: {name}")
        artifact_sha256[name] = actual
    if artifact_sha256 != {
        "adapter.safetensors": EXPECTED_STAGE_B_ADAPTER_SHA256,
        "optimizer.pt": EXPECTED_STAGE_B_OPTIMIZER_SHA256,
        "history.json": EXPECTED_STAGE_B_HISTORY_SHA256,
    }:
        fail("Stage-B training artifacts differ from v1 authority")
    history = _read_json_bound(
        entries["history.json"], artifact_sha256["history.json"], label="Stage-B history"
    )
    steps = history.get("steps")
    if (
        history.get("schema_version") != stage_b.HISTORY_SCHEMA
        or history.get("optimizer_steps") != 4
        or history.get("registered_schedule_indices") != list(stage_b.REGISTERED_SCHEDULE_INDICES)
        or not isinstance(steps, list)
        or len(steps) != 4
    ):
        fail("Stage-B training history contract differs")
    noise_seeds: list[int] = []
    epsilon_hashes: list[str] = []
    expected_arms = (
        {
            "logical_arm": 0,
            "row_index": 0,
            "iid": "0014a41e55e44670",
            "wrong_ref_iid": "00435ad621c44fac",
            "main_style": 1,
        },
        {
            "logical_arm": 1,
            "row_index": 1,
            "iid": "00435ad621c44fac",
            "wrong_ref_iid": "0014a41e55e44670",
            "main_style": 2,
        },
    )
    for step_index, step in enumerate(steps):
        logical = step.get("logical_records") if isinstance(step, Mapping) else None
        if (
            not isinstance(logical, list)
            or len(logical) != 2
            or step.get("schedule_index") != stage_b.REGISTERED_SCHEDULE_INDICES[step_index]
        ):
            fail("Stage-B history logical-arm closure differs")
        for arm_index, row in enumerate(logical):
            if not isinstance(row, Mapping) or any(
                row.get(key) != expected
                for key, expected in expected_arms[arm_index].items()
            ):
                fail("Stage-B history registered logical arm differs")
            seed = row.get("noise_seed") if isinstance(row, Mapping) else None
            if type(seed) is not int or not 0 <= seed < 2**63:
                fail("Stage-B history noise seed differs")
            noise_seeds.append(seed)
            epsilon_sha = _require_sha(
                row.get("epsilon_sha256"), length=64, label="training epsilon SHA"
            )
            identities = row.get("tensor_identities")
            binding = row.get("shared_noise_binding")
            if (
                not isinstance(identities, Mapping)
                or identities.get("epsilon") != epsilon_sha
                or not isinstance(binding, Mapping)
                or binding.get("epsilon_sha256") != epsilon_sha
                or binding.get("schedule_index") != stage_b.REGISTERED_SCHEDULE_INDICES[step_index]
                or binding.get("same_epsilon_object_reused_during_target_and_donor_construction") is not True
                or binding.get("target_formula_recomputed_and_equal") is not True
                or binding.get("donor_formula_recomputed_and_equal") is not True
                or binding.get("same_sigma_registered_coordinate_reused") is not True
                or binding.get("forward_noising_only") is not True
                or binding.get("inversion_claimed") is not False
            ):
                fail("Stage-B history same-noise binding differs")
            declared_binding_digest = binding.get("digest")
            unsigned_binding = dict(binding)
            unsigned_binding.pop("digest", None)
            if (
                _require_sha(
                    declared_binding_digest,
                    length=64,
                    label="training shared-noise binding digest",
                )
                != object_sha256(unsigned_binding)
            ):
                fail("Stage-B history shared-noise binding digest differs")
            epsilon_hashes.append(epsilon_sha)
    if len(set(noise_seeds)) != 8 or len(set(epsilon_hashes)) != 8:
        fail("Stage-B eight training noise realizations are not distinct")
    final_digest = _require_sha(
        receipt.get("final_adapter_sha256"), length=64, label="final adapter parameter SHA"
    )
    if final_digest != EXPECTED_STAGE_B_FINAL_PARAMETER_SHA256:
        fail("Stage-B final parameter digest differs from v1 authority")
    return BoundStageBAdapter(
        root=root,
        adapter=adapter_path,
        receipt=receipt_path,
        adapter_sha256=adapter_sha,
        receipt_sha256=receipt_sha,
        receipt_digest=digest,
        final_parameter_sha256=final_digest,
        receipt_value=receipt,
        artifact_sha256=artifact_sha256,
        training_noise_seeds=tuple(noise_seeds),
        training_epsilon_sha256=tuple(epsilon_hashes),
    )


def validate_adapter_runtime_binding(
    bundle: BoundStageBAdapter,
    *,
    dataset: BoundDatasetRow,
    bernini_revision: str,
    veomni_revision: str,
    checkpoint_tree_sha256: str,
    inference_seed: int,
    inference_epsilon_sha256: str,
) -> dict[str, Any]:
    training_dataset = bundle.receipt_value.get("dataset")
    training_model = bundle.receipt_value.get("model")
    if (
        not isinstance(training_dataset, Mapping)
        or training_dataset.get("parquet_sha256") != dataset.dataset.parquet_sha256
        or training_dataset.get("receipt_sha256") != dataset.dataset.receipt_sha256
        or training_dataset.get("receipt_digest") != dataset.dataset.receipt_digest
        or not isinstance(training_model, Mapping)
        or training_model.get("bernini_commit") != bernini_revision
        or training_model.get("veomni_commit") != veomni_revision
        or training_model.get("checkpoint_tree_sha256") != checkpoint_tree_sha256
    ):
        fail("Stage-B adapter training dataset/model differs from inference runtime")
    if inference_seed in bundle.training_noise_seeds:
        fail("inference seed replays one of the eight Stage-B training noise seeds")
    inference_epsilon = _require_sha(
        inference_epsilon_sha256, length=64, label="inference epsilon SHA"
    )
    if inference_epsilon in bundle.training_epsilon_sha256:
        fail("inference epsilon replays one of the eight Stage-B training realizations")
    value = {
        "training_dataset_exact_match": True,
        "training_model_exact_match": True,
        "training_artifacts_sha256": dict(bundle.artifact_sha256),
        "training_final_parameter_sha256": bundle.final_parameter_sha256,
        "training_noise_seeds": list(bundle.training_noise_seeds),
        "training_epsilon_sha256": list(bundle.training_epsilon_sha256),
        "inference_seed_absent_from_training_noise_seeds": True,
        "inference_epsilon_absent_from_training_realizations": True,
    }
    return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class BoundDatasetRow:
    dataset: Any
    row: Any
    clean: Any
    donor: Any
    references: tuple[Any, Any, Any]
    style_id: int
    clean_blob_sha256: str
    donor_blob_sha256: str
    reference_blob_sha256: tuple[str, str, str]


def load_bound_dataset_row(
    *,
    dataset_root: str | Path,
    expected_spec_sha256: str,
    expected_parquet_sha256: str,
    expected_receipt_sha256: str,
    expected_receipt_digest: str,
    iid: str,
    style_id: int,
    checkpoint: Path,
) -> BoundDatasetRow:
    _load_heavy_runtime_modules()
    if REGISTERED_IID_STYLE.get(iid) != style_id:
        fail("IID/style arm differs from the registered two-arm Stage-B mapping")
    try:
        dataset = dataset_core._load_dataset(dataset_root, expected_spec_sha256)
    except dataset_core.SourceSelfTrainingError as error:
        raise StageBInferenceError(str(error)) from error
    if (
        dataset.parquet_sha256 != _require_sha(expected_parquet_sha256, length=64, label="dataset parquet SHA")
        or dataset.receipt_sha256 != _require_sha(expected_receipt_sha256, length=64, label="dataset receipt SHA")
        or dataset.receipt_digest != _require_sha(expected_receipt_digest, length=64, label="dataset receipt digest")
    ):
        fail("sealed materialized dataset identities differ")
    matches = [item for item in dataset.rows if item.iid == iid]
    if len(matches) != 1:
        fail("requested IID is absent or duplicated in sealed dataset")
    row = matches[0]
    mean, std, _ = legacy_train._vae_statistics(checkpoint)
    clean = dataset_core._posterior_mode(
        row.clean_blob, mean, std, phases=stage_b.LATENT_PHASES, label=f"{iid} clean"
    )
    donor_blob = row.style1_blob if style_id == 1 else row.style2_blob
    donor = dataset_core._posterior_mode(
        donor_blob, mean, std, phases=stage_b.LATENT_PHASES, label=f"{iid} style{style_id}"
    )
    references = tuple(
        dataset_core._posterior_mode(
            row.refs[index], mean, std, phases=stage_b.REFERENCE_PHASES, label=f"{iid} ref{index}"
        )
        for index in row.reference_order
    )
    return BoundDatasetRow(
        dataset=dataset,
        row=row,
        clean=clean,
        donor=donor,
        references=references,
        style_id=style_id,
        clean_blob_sha256=hashlib.sha256(row.clean_blob).hexdigest(),
        donor_blob_sha256=hashlib.sha256(donor_blob).hexdigest(),
        reference_blob_sha256=tuple(
            hashlib.sha256(row.refs[index]).hexdigest() for index in row.reference_order
        ),
    )


def route_enabled(mode: str, schedule_index: int) -> bool:
    if mode not in MODES or type(schedule_index) is not int or not 0 <= schedule_index < 40:
        fail("route lookup mode/index differs")
    if mode == "registered-probes":
        return schedule_index in stage_b.REGISTERED_SCHEDULE_INDICES
    return True


def mode_contract(mode: str) -> dict[str, Any]:
    if mode not in MODES:
        fail("unsupported Stage-B evaluation mode")
    probes = mode == "registered-probes"
    all40 = mode == "full40-evolved-target-all40-route-extrapolation"
    value = {
        "mode": mode,
        "raw_conditional_single_forward": True,
        "unconditional_branch_present": False,
        "apg_or_cfg_present": False,
        "scheduler": exact40.SCHEDULER_CLASS,
        "exact40_schedule_sha256": exact40.SCHEDULE_SHA256,
        "registered_schedule_indices": list(stage_b.REGISTERED_SCHEDULE_INDICES),
        "target_reanchored_during_rollout": False,
        "training_equation_pack_schedule_coordinate_exact": probes,
        "training_realization_replayed": False,
        "one_shared_inference_epsilon_across_registered_probes": probes,
        "full40_evolved_target": not probes,
        "evolved_target_after_first_step_matches_training_formula": False if not probes else None,
        "registered_target_training_formula_match_claimed": probes,
        "rollout_training_coordinate_match": (
            None if probes else "initial_epsilon_and_same-epsilon_donor_ladder_only"
        ),
        "rollout_distribution_shift_from_stateless_training": not probes,
        "adapter_route_policy": (
            "four_registered_cells_only" if not all40 else "all40_explicit_route_extrapolation"
        ),
        "route_outside_training_sigma_support": all40,
        "text_instruction": stage_b.GENERIC_INSTRUCTION,
        "text_action_prompt_ood": False,
        "donor_construction_forward_noising_only": True,
        "target_solver_kind": (
            "none_stateless_registered_queries"
            if probes
            else "noise_to_clean_unipc_denoising"
        ),
        "target_unipc_denoising_solver_executed": not probes,
        "inversion_claimed": False,
        "inversion_reverse_ode_executed": False,
        "solver_state_replayed": False,
        "exact_roundtrip_claimed": False,
        "method_success_claimed": False,
    }
    return {**value, "digest": object_sha256(value)}


def _zero_adapter(adapter: role.SourceSelfAdapterHandle) -> str:
    _load_heavy_runtime_modules()
    import torch

    with torch.no_grad():
        for _, parameter in adapter.trainable_named_parameters():
            parameter.zero_()
    digest = runtime.trainable_parameters_digest(adapter.trainable_named_parameters())
    if any(bool(parameter.count_nonzero().item()) for _, parameter in adapter.trainable_named_parameters()):
        fail("frozen-base adapter is not exactly zero")
    return digest


def strict_load_stage_b_adapter(
    transformer: Any,
    bundle: BoundStageBAdapter,
) -> tuple[role.SourceSelfAdapterHandle, dict[str, Any]]:
    """Install and strictly load the distinct Stage-B adapter closure.

    This loader lives in the inference runtime so the frozen role/training
    implementation is not broadened with a deployment-only schema.
    """

    _load_heavy_runtime_modules()
    import torch
    from safetensors import safe_open

    path = _plain_absolute_file(bundle.adapter, label="Stage-B adapter safetensors")
    before = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if file_sha256(path) != bundle.adapter_sha256:
        fail("Stage-B adapter changed before strict loading")
    adapter = role.install_source_self_adapter(
        transformer,
        rank=stage_b.LORA_RANK,
        alpha=stage_b.LORA_ALPHA,
        block_indices=role.TRAINABLE_BLOCK_INDICES,
    )
    try:
        with safe_open(str(path), framework="pt", device="cpu") as opened:
            keys = tuple(sorted(opened.keys()))
            metadata = dict(opened.metadata() or {})
            tensors = {key: opened.get_tensor(key).contiguous() for key in keys}
        after = path.stat()
        if identity(before) != identity(after) or file_sha256(path) != bundle.adapter_sha256:
            fail("Stage-B adapter changed while strict loading")
        expected_metadata = {
            "schema_version": stage_b.ADAPTER_FILE_SCHEMA,
            "role_adapter_schema_version": role.SCHEMA_VERSION,
            "block_indices_json": canonical_json_bytes(
                list(role.TRAINABLE_BLOCK_INDICES)
            ).decode("ascii"),
            "projections_json": canonical_json_bytes(
                ["attn1.to_q", "attn1.to_out.0"]
            ).decode("ascii"),
            "target_row_only": "true",
            "role_embedding": "donor_reference_target",
            "lora_rank": str(stage_b.LORA_RANK),
            "lora_alpha_hex": stage_b.LORA_ALPHA.hex(),
            "exact40_schedule_sha256": stage_b.EXPECTED_EXACT40_SCHEDULE_SHA256,
            "registered_schedule_indices_json": canonical_json_bytes(
                list(stage_b.REGISTERED_SCHEDULE_INDICES)
            ).decode("ascii"),
            "target_and_donor_same_epsilon": "true",
            "forward_noising_only": "true",
            "inversion_claimed": "false",
            "matched_carrier_runtime_required": "true",
        }
        if metadata != expected_metadata:
            fail("Stage-B adapter safetensors metadata differs")
        named = adapter.trainable_named_parameters()
        expected_names = tuple(sorted(name for name, _ in named))
        if keys != expected_names:
            fail("Stage-B adapter tensor key closure differs")
        parameter_map = dict(named)
        for name in expected_names:
            tensor = tensors[name]
            parameter = parameter_map[name]
            if (
                tensor.dtype != torch.float32
                or tuple(tensor.shape) != tuple(parameter.shape)
                or tensor.requires_grad
                or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all().item())
            ):
                fail(f"Stage-B adapter tensor contract differs: {name}")
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(tensors[name].to(device=parameter.device))
        load_receipt = {
            "schema_version": "bernini-source-noised-carrier-strict-load-v1",
            "path": str(path),
            "file_sha256": bundle.adapter_sha256,
            "exact40_schedule_sha256": stage_b.EXPECTED_EXACT40_SCHEDULE_SHA256,
            "registered_schedule_indices": list(stage_b.REGISTERED_SCHEDULE_INDICES),
            "metadata": metadata,
            "tensor_count": len(tensors),
            "tensor_names_sha256": object_sha256(list(expected_names)),
            "strict_tensor_closure": True,
            "base_parameters_frozen": adapter.base_parameters_frozen(),
            "forward_noising_only": True,
            "inversion_claimed": False,
        }
        return adapter, {
            **load_receipt,
            "digest": object_sha256(load_receipt),
        }
    except Exception:
        if not adapter.restored:
            adapter.restore()
        raise


def install_arm_adapter(
    transformer: Any,
    *,
    arm: str,
    bundle: Optional[BoundStageBAdapter],
) -> tuple[role.SourceSelfAdapterHandle, dict[str, Any]]:
    _load_heavy_runtime_modules()
    if arm not in ARMS:
        fail("inference arm differs")
    if arm == "trained":
        if bundle is None:
            fail("trained arm requires a bound Stage-B adapter")
        adapter, load_receipt = strict_load_stage_b_adapter(transformer, bundle)
        digest = runtime.trainable_parameters_digest(adapter.trainable_named_parameters())
        if digest != bundle.final_parameter_sha256:
            fail("strictly loaded adapter parameter digest differs from training receipt")
        return adapter, {
            "arm": arm,
            "weights": "strict_stage_b_training_adapter",
            "same_role_q_o_wrapper_as_frozen_base": True,
            "route_wrapper_installed": True,
            "all_adapter_tensors_exact_zero": False,
            "parameter_sha256": digest,
            "file_sha256": bundle.adapter_sha256,
            "training_receipt_sha256": bundle.receipt_sha256,
            "training_receipt_digest": bundle.receipt_digest,
            "strict_load": dict(load_receipt),
        }
    if bundle is not None:
        fail("frozen-base arm must not load trained adapter bytes")
    adapter = role.install_source_self_adapter(
        transformer,
        rank=stage_b.LORA_RANK,
        alpha=stage_b.LORA_ALPHA,
        block_indices=role.TRAINABLE_BLOCK_INDICES,
    )
    digest = _zero_adapter(adapter)
    return adapter, {
        "arm": arm,
        "weights": "all_adapter_tensors_exact_zero",
        "same_role_q_o_wrapper_as_trained": True,
        "route_wrapper_installed": True,
        "all_adapter_tensors_exact_zero": True,
        "parameter_sha256": digest,
        "file_sha256": None,
        "training_receipt_sha256": None,
        "training_receipt_digest": None,
        "strict_load": None,
    }


def _prepare_condition(
    *,
    clean: Any,
    donor: Any,
    references: Sequence[Any],
    epsilon: Any,
    sigma: float,
    coordinate: stage_b.RegisteredCarrierCoordinate,
    rope: Any,
    device: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    _load_heavy_runtime_modules()
    noisy_target = ladder.shared_noise_source_state(clean, epsilon, sigma)
    noisy_donor = ladder.shared_noise_source_state(donor, epsilon, sigma)
    condition = stage_b._condition(
        base=dataset_core,
        role=role,
        donor=noisy_donor,
        references=references,
        noisy_target=noisy_target,
        coordinate=coordinate,
        rope=rope,
        device=device,
    )
    binding = {
        "epsilon_sha256": runtime.tensor_sha256(epsilon),
        "clean_target_sha256": runtime.tensor_sha256(clean),
        "clean_donor_sha256": runtime.tensor_sha256(donor),
        "noised_target_sha256": runtime.tensor_sha256(noisy_target),
        "noised_donor_sha256": runtime.tensor_sha256(noisy_donor),
        "same_epsilon_object_target_donor": True,
        "same_sigma_target_donor": True,
        "sigma_float32_be_hex": coordinate.sigma_float32_be_hex,
    }
    return condition, noisy_target, {**binding, "digest": object_sha256(binding)}


def _raw_prediction(
    *,
    renderer: Any,
    transformer: Any,
    adapter: role.SourceSelfAdapterHandle,
    condition: Any,
    text_lens: Any,
    text_embs: Any,
    sp_rank: int,
    route_on: bool,
    world_group: Any,
) -> Any:
    _load_heavy_runtime_modules()
    import torch

    invocation = role.RouteInvocation(
        condition.layout,
        sequence_parallel_rank=sp_rank,
        sequence_parallel_size=SP_SIZE,
        enabled=route_on,
    )
    def embed_local() -> tuple[Any, Any]:
        with adapter.route(invocation), torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            embedded = (
                transformer.patch_embedding(condition.input_patches)
                .flatten(1)
                .unsqueeze(0)
            )
        rotary = condition.rotary.permute(1, 0, 2).unsqueeze(0)
        return embedded, rotary

    embedded, rotary = collective_rank_call(
        embed_local,
        group=world_group,
        label=f"registered probe {condition.coordinate.schedule_index} patch embedding",
    )
    runtime.digest_consensus(
        object_sha256(
            {
                "embedded_sha256": runtime.tensor_sha256(embedded),
                "rotary_sha256": runtime.tensor_sha256(rotary),
            }
        ),
        group=world_group,
        expected_count=4,
        label=f"registered probe {condition.coordinate.schedule_index} embedded input",
    )
    with adapter.route(invocation), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        value = renderer.diff_dec.shared_step(
            model_id="transformer_1",
            noisy_latents=embedded,
            timesteps=embedded.new_tensor(
                [condition.coordinate.timestep], dtype=torch.int64
            ),
            cond_embeds=text_embs,
            rotary_embs=rotary,
            batch_vae_seqlen=[condition.layout.total_tokens],
            batch_text_seqlen=text_lens,
        )
        prediction = value[
            :,
            condition.layout.condition_tokens : condition.layout.total_tokens,
            :,
        ]
    if tuple(int(item) for item in prediction.shape) != (
        1,
        condition.layout.target_tokens,
        role.PATCH_VALUES,
    ):
        fail("registered raw prediction geometry differs")
    return prediction


def _unpack_field(packed: Any, shape: Sequence[int]) -> Any:
    _load_heavy_runtime_modules()
    import torch

    batch, channels, phases, height, width = (int(item) for item in shape)
    if batch != 1 or channels != role.LATENT_CHANNELS or height % 2 or width % 2:
        fail("latent geometry cannot be unpacked")
    expected = phases * (height // 2) * (width // 2)
    if tuple(int(item) for item in packed.shape) != (1, expected, role.PATCH_VALUES):
        fail("packed prediction geometry differs")
    patches = packed.reshape(
        batch, phases, height // 2, width // 2, 1, 2, 2, channels
    )
    return (
        patches.permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(batch, channels, phases, height, width)
        .contiguous()
    )


def _pack_field(spatial: Any, shape: Sequence[int]) -> Any:
    """Wan-exact inverse of :func:`_unpack_field` for one target row."""

    _load_heavy_runtime_modules()
    expected = tuple(int(item) for item in shape)
    if tuple(int(item) for item in spatial.shape) != expected:
        fail("spatial target geometry differs before exact repack")
    return runtime.packed_output_field(
        dataset_core.pack_latent_patches(
            spatial.squeeze(0).contiguous(), phases=expected[2]
        )
    )


def _registered_probe_latents(
    *,
    renderer: Any,
    transformer: Any,
    adapter: role.SourceSelfAdapterHandle,
    bound: BoundDatasetRow,
    epsilon: Any,
    rope: Any,
    device: Any,
    text_lens: Any,
    text_embs: Any,
    sp_rank: int,
    world_group: Any,
) -> tuple[list[Any], list[dict[str, Any]]]:
    _load_heavy_runtime_modules()
    outputs: list[Any] = []
    records: list[dict[str, Any]] = []
    for coordinate in stage_b.validate_registered_schedule():
        condition, noisy_target, binding = collective_rank_call(
            lambda: _prepare_condition(
                clean=bound.clean,
                donor=bound.donor,
                references=bound.references,
                epsilon=epsilon,
                sigma=coordinate.sigma,
                coordinate=coordinate,
                rope=rope,
                device=device,
            ),
            group=world_group,
            label=f"registered probe {coordinate.schedule_index} preparation",
        )
        prepared_digest = object_sha256(
            {
                "schedule_index": coordinate.schedule_index,
                "input_patches_sha256": runtime.tensor_sha256(condition.input_patches),
                "rotary_sha256": runtime.tensor_sha256(condition.rotary),
                "layout": condition.layout.receipt(),
                "binding_digest": binding["digest"],
            }
        )
        runtime.digest_consensus(
            prepared_digest,
            group=world_group,
            expected_count=4,
            label=f"registered probe {coordinate.schedule_index} prepared input",
        )
        prediction = _raw_prediction(
            renderer=renderer,
            transformer=transformer,
            adapter=adapter,
            condition=condition,
            text_lens=text_lens,
            text_embs=text_embs,
            sp_rank=sp_rank,
            route_on=True,
            world_group=world_group,
        )
        def postprocess_probe() -> tuple[Any, dict[str, Any]]:
            import torch

            if prediction.dtype != torch.bfloat16 or not bool(torch.isfinite(prediction).all().item()):
                fail("registered probe raw branch must be finite BF16")
            velocity = _unpack_field(prediction, (1, *bound.clean.shape))
            sigma_cpu = torch.tensor(coordinate.sigma, dtype=torch.float32, device="cpu")
            if exact40._float32_hex(sigma_cpu, label="probe sigma") != coordinate.sigma_float32_be_hex:
                fail("registered probe CPU FP32 sigma bits differ")
            noisy_fp32 = noisy_target.unsqueeze(0).to(device).float()
            predicted_clean = (noisy_fp32 - sigma_cpu * velocity).detach().contiguous()
            if not bool(torch.isfinite(predicted_clean).all().item()):
                fail("registered probe predicted-clean tensor is non-finite")
            record = {
                **coordinate.receipt(),
                "target_equation_exact": True,
                "donor_equation_exact": True,
                "training_realization_replayed": False,
                "predicted_clean_equation": "x0_hat=x_target_sigma-sigma*v_raw_conditional",
                "raw_conditional_single_forward": True,
                "raw_prediction_dtype": str(prediction.dtype),
                "noisy_target_dtype": str(noisy_fp32.dtype),
                "sigma_dtype_device": "torch.float32/cpu/0d",
                "numeric_program": "fp32_noisy-minus-cpu_fp32_sigma-times-bf16_raw_velocity",
                "route_enabled": True,
                "prepared_input_digest": prepared_digest,
                "binding": dict(binding),
                "predicted_clean_sha256": runtime.tensor_sha256(predicted_clean),
            }
            return predicted_clean, record

        predicted_clean, record = collective_rank_call(
            postprocess_probe,
            group=world_group,
            label=f"registered probe {coordinate.schedule_index} postprocess",
        )
        outputs.append(predicted_clean)
        records.append(record)
    return outputs, records


def _full40_rollout(
    *,
    renderer: Any,
    transformer: Any,
    adapter: role.SourceSelfAdapterHandle,
    bound: BoundDatasetRow,
    epsilon: Any,
    rope: Any,
    device: Any,
    text_lens: Any,
    text_embs: Any,
    sp_rank: int,
    mode: str,
    world_group: Any,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    _load_heavy_runtime_modules()
    import torch
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(renderer.config.wan22_base),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=exact40.FLOW_SHIFT,
    )
    audit = exact40.audit_runtime_unipc_schedule(scheduler)
    if audit.get("schedule_sha256") != exact40.SCHEDULE_SHA256:
        fail("fresh rollout scheduler differs")
    # Bernini's official UniPC loop evolves the packed target row [1,N,64].
    # Spatial latents are reconstructed only to build the five-role condition.
    target = runtime.packed_output_field(
        dataset_core.pack_latent_patches(epsilon, phases=stage_b.LATENT_PHASES)
    ).to(device=device, dtype=torch.float32).contiguous()
    records: list[dict[str, Any]] = []
    for index, (timestep, sigma) in enumerate(
        zip(exact40.PINNED_TIMESTEPS, exact40.PINNED_POSITIVE_SIGMAS)
    ):
        def prepare_step() -> tuple[Any, ...]:
            cursor_before = getattr(scheduler, "step_index", None)
            if (index == 0 and cursor_before is not None) or (
                index > 0 and cursor_before != index
            ):
                fail("fresh UniPC pre-step cursor differs")
            noised_donor = ladder.shared_noise_source_state(bound.donor, epsilon, sigma)
            donor_patches = dataset_core.pack_latent_patches(
                noised_donor, phases=stage_b.LATENT_PHASES
            )
            reference_patches = [
                dataset_core.pack_latent_patches(item, phases=stage_b.REFERENCE_PHASES)
                for item in bound.references
            ]
            target_spatial = _unpack_field(target, (1, *bound.clean.shape))
            target_value = target_spatial.squeeze(0).detach().cpu().contiguous()
            repacked_target = _pack_field(
                target_value.unsqueeze(0), (1, *bound.clean.shape)
            ).to(device)
            if not torch.equal(repacked_target, target):
                fail("full40 packed/spatial condition roundtrip is not bit exact")
            target_before_sha = runtime.tensor_sha256(target)
            donor_sha = runtime.tensor_sha256(noised_donor)
            target_patches = dataset_core.pack_latent_patches(
                target_value, phases=stage_b.LATENT_PHASES
            )
            layout = role.TokenRoleLayout.contiguous(
                donor_tokens=int(donor_patches.shape[0]),
                reference_tokens=[int(item.shape[0]) for item in reference_patches],
                target_tokens=int(target_patches.shape[0]),
            )
            patches = torch.cat(
                (donor_patches, *reference_patches, target_patches), dim=0
            ).to(device)
            rotations = [
                rope(noised_donor.unsqueeze(0).to(device), source_id=1),
                *(
                    rope(item.unsqueeze(0).to(device), source_id=slot + 2)
                    for slot, item in enumerate(bound.references)
                ),
                rope(target_value.unsqueeze(0).to(device), source_id=0),
            ]
            rotary = (
                torch.cat(rotations, dim=2)
                .squeeze(0)
                .permute(1, 0, 2)
                .contiguous()
            )
            prepared_digest = object_sha256(
                {
                    "schedule_index": index,
                    "patches_sha256": runtime.tensor_sha256(patches),
                    "rotary_sha256": runtime.tensor_sha256(rotary),
                    "layout": layout.receipt(),
                    "target_before_sha256": target_before_sha,
                    "donor_sha256": donor_sha,
                }
            )
            return (
                cursor_before,
                noised_donor,
                target_before_sha,
                donor_sha,
                layout,
                patches,
                rotary,
                prepared_digest,
            )

        (
            cursor_before,
            noised_donor,
            target_before_sha,
            donor_sha,
            layout,
            patches,
            rotary,
            prepared_digest,
        ) = collective_rank_call(
            prepare_step,
            group=world_group,
            label=f"full40 step {index} preparation",
        )
        runtime.digest_consensus(
            prepared_digest,
            group=world_group,
            expected_count=4,
            label=f"full40 step {index} prepared input",
        )
        # stage_b._prediction deliberately rejects nonregistered coordinates;
        # call the identical raw primitive directly for all40 OOD execution.
        invocation = role.RouteInvocation(layout, sp_rank, SP_SIZE, enabled=route_enabled(mode, index))
        def embed_step() -> Any:
            with adapter.route(invocation), torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                return transformer.patch_embedding(patches).flatten(1).unsqueeze(0)

        embedded = collective_rank_call(
            embed_step,
            group=world_group,
            label=f"full40 step {index} patch embedding",
        )
        runtime.digest_consensus(
            runtime.tensor_sha256(embedded),
            group=world_group,
            expected_count=4,
            label=f"full40 step {index} embedded input",
        )
        with adapter.route(invocation), torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            value = renderer.diff_dec.shared_step(
                model_id="transformer_1",
                noisy_latents=embedded,
                timesteps=embedded.new_tensor([timestep], dtype=torch.int64),
                cond_embeds=text_embs,
                rotary_embs=rotary.permute(1, 0, 2).unsqueeze(0),
                batch_vae_seqlen=[layout.total_tokens],
                batch_text_seqlen=text_lens,
            )
            prediction = value[:, layout.condition_tokens:, :]
        def postprocess_step() -> tuple[Any, dict[str, Any]]:
            expected_target_shape = tuple(int(item) for item in target.shape)
            if (
                prediction.dtype != torch.bfloat16
                or tuple(int(item) for item in prediction.shape) != expected_target_shape
                or target.dtype != torch.float32
                or not bool(torch.isfinite(prediction).all().item())
                or not bool(torch.isfinite(target).all().item())
            ):
                fail("full40 raw prediction/packed target dtype or geometry differs")
            stepped = scheduler.step(
                prediction,
                scheduler.timesteps[index].to(device),
                target,
                return_dict=False,
            )
            if type(stepped) is not tuple or len(stepped) != 1:
                fail("fresh UniPC return_dict=False closure differs")
            if getattr(scheduler, "step_index", None) != index + 1:
                fail("fresh UniPC post-step cursor differs")
            next_target = stepped[0].detach().float().contiguous()
            if (
                next_target.dtype != torch.float32
                or tuple(int(item) for item in next_target.shape) != expected_target_shape
                or not bool(torch.isfinite(next_target).all().item())
            ):
                fail("full40 evolved target is non-finite")
            record = {
                "schedule_index": index,
                "timestep_int64": timestep,
                "sigma_float32_be_hex": exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index],
                "registered_training_coordinate": index in stage_b.REGISTERED_SCHEDULE_INDICES,
                "route_enabled": route_enabled(mode, index),
                "target_reanchored": False,
                "target_matches_stateless_training_formula": False,
                "initial_target_is_seeded_epsilon_only": index == 0,
                "donor_same_initial_epsilon_forward_noised": True,
                "packed_condition_roundtrip_bit_exact": True,
                "raw_prediction_dtype": str(prediction.dtype),
                "packed_target_pre_post_dtype": "torch.float32",
                "packed_target_shape": list(expected_target_shape),
                "scheduler_step_index_before": cursor_before,
                "scheduler_step_index_after": index + 1,
                "prepared_input_digest": prepared_digest,
                "noised_style_donor_sha256": donor_sha,
                "packed_target_sha256_before_step": target_before_sha,
                "raw_prediction_sha256": runtime.tensor_sha256(prediction),
                "target_sha256_after_step": runtime.tensor_sha256(next_target),
            }
            return next_target, record

        target, record = collective_rank_call(
            postprocess_step,
            group=world_group,
            label=f"full40 step {index} postprocess",
        )
        records.append(record)
    if getattr(scheduler, "step_index", None) != 40:
        fail("fresh UniPC terminal cursor differs")
    final_unpacked = _unpack_field(target, (1, *bound.clean.shape))
    schedule_audit = {
        **audit,
        "fresh_scheduler_instance_for_this_arm": True,
        "scheduler_steps_executed": 40,
        "raw_bf16_prediction_passed_without_precast": True,
        "packed_fp32_target_state_evolved": True,
        "scheduler_cursor_pre_post_exact": True,
        "terminal_step_index": 40,
        "final_packed_target_sha256": runtime.tensor_sha256(target),
        "final_unpacked_decode_latent_sha256": runtime.tensor_sha256(final_unpacked),
    }
    schedule_audit["digest"] = object_sha256(schedule_audit)
    return final_unpacked, records, schedule_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--expected-materialization-spec-sha256", required=True)
    parser.add_argument("--expected-dataset-parquet-sha256", required=True)
    parser.add_argument("--expected-dataset-receipt-sha256", required=True)
    parser.add_argument("--expected-dataset-receipt-digest", required=True)
    parser.add_argument("--iid", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--instruction", required=True, choices=(stage_b.GENERIC_INSTRUCTION,))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--style-id", required=True, type=int, choices=STYLE_IDS)
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--expected-training-receipt-sha256")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor-action-video")
    parser.add_argument("--expected-anchor-action-sha256")
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.arm == "trained":
        if not all(
            isinstance(getattr(args, name), str) and getattr(args, name)
            for name in ("adapter_checkpoint", "expected_adapter_sha256", "expected_training_receipt_sha256")
        ):
            fail("trained arm requires adapter directory and both file SHA pins")
    elif any(getattr(args, name) is not None for name in ("adapter_checkpoint", "expected_adapter_sha256", "expected_training_receipt_sha256")):
        fail("frozen-base arm forbids trained adapter arguments")
    if (args.anchor_action_video is None) != (args.expected_anchor_action_sha256 is None):
        fail("anchor display path and SHA must be supplied together or both absent")
    if args.seed != INFERENCE_SEED:
        fail(f"Stage-B evidence seed must be the fresh fixed value {INFERENCE_SEED}")
    if args.instruction != stage_b.GENERIC_INSTRUCTION:
        fail("matched Stage-B inference requires the training generic instruction")
    if (
        args.expected_materialization_spec_sha256
        != EXPECTED_DATASET_MATERIALIZATION_SPEC_SHA256
        or args.expected_dataset_parquet_sha256 != EXPECTED_DATASET_PARQUET_SHA256
        or args.expected_dataset_receipt_sha256 != EXPECTED_DATASET_RECEIPT_SHA256
        or args.expected_dataset_receipt_digest != EXPECTED_DATASET_RECEIPT_DIGEST
    ):
        fail("Stage-B inference dataset pins differ from v1 authority")
    if args.arm == "trained" and (
        args.expected_adapter_sha256 != EXPECTED_STAGE_B_ADAPTER_SHA256
        or args.expected_training_receipt_sha256
        != EXPECTED_STAGE_B_TRAINING_RECEIPT_SHA256
    ):
        fail("Stage-B trained adapter pins differ from v1 authority")
    for name in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        _require_sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_checkpoint_tree_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "method_source_archive_sha256",
        "expected_materialization_spec_sha256",
        "expected_dataset_parquet_sha256",
        "expected_dataset_receipt_sha256",
        "expected_dataset_receipt_digest",
        "expected_source_sha256",
    ):
        _require_sha(getattr(args, name), length=64, label=name)


def verify_inference_bundle(
    directory: Path,
    receipt: Mapping[str, Any],
) -> None:
    """Verify the inference-specific create-only artifact closure."""

    expected_outputs = OUTPUT_FILES_BY_MODE.get(str(receipt.get("mode_contract", {}).get("mode")))
    if expected_outputs is None:
        fail("inference receipt mode has no artifact closure")
    expected_entries = set(expected_outputs) | {"receipt.json"}
    entries = list(directory.iterdir())
    if (
        {path.name for path in entries} != expected_entries
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        fail("inference artifact closure differs")
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    if (directory / "receipt.json").read_bytes() != receipt_bytes:
        fail("inference receipt bytes differ from memory")
    parsed = json.loads(receipt_bytes.decode("ascii"))
    _validate_embedded_digest(parsed, label="inference")
    artifacts = parsed.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_outputs):
        fail("inference receipt artifact manifest differs")
    for name in expected_outputs:
        expected = _require_sha(artifacts.get(name), length=64, label=f"{name} artifact SHA")
        if file_sha256(directory / name) != expected:
            fail(f"inference artifact hash differs: {name}")
    outputs = parsed.get("outputs")
    if (
        not isinstance(outputs, list)
        or [item.get("name") for item in outputs if isinstance(item, Mapping)] != list(expected_outputs)
        or any(item.get("sha256") != artifacts[item["name"]] for item in outputs)
    ):
        fail("inference output records differ from artifact manifest")
    compute = parsed.get("compute_consensus")
    tensor_binding = parsed.get("tensor_binding")
    instruction = parsed.get("instruction")
    schedule = parsed.get("runtime_schedule_audit")
    adapter = parsed.get("adapter")
    if not all(
        isinstance(item, Mapping)
        for item in (compute, tensor_binding, instruction, schedule, adapter)
    ):
        fail("inference internal binding namespace is missing")
    compute_unsigned = dict(compute)
    compute_digest = compute_unsigned.pop("digest", None)
    if (
        _require_sha(compute_digest, length=64, label="compute consensus digest")
        != object_sha256(compute_unsigned)
        or compute.get("records_sha256") != object_sha256(parsed.get("records"))
        or compute.get("latent_outputs_sha256")
        != [item.get("decode_input_latent_sha256") for item in outputs]
        or compute.get("epsilon_sha256") != tensor_binding.get("epsilon_sha256")
        or compute.get("text_embedding_sha256")
        != instruction.get("token_and_embedding_binding", {}).get("embedding_sha256")
        or compute.get("runtime_schedule_audit_digest") != schedule.get("digest")
        or compute.get("adapter_parameter_sha256") != adapter.get("parameter_sha256")
    ):
        fail("inference internal compute binding differs")
    schedule_unsigned = dict(schedule)
    schedule_digest = schedule_unsigned.pop("digest", None)
    if (
        _require_sha(schedule_digest, length=64, label="runtime schedule audit digest")
        != object_sha256(schedule_unsigned)
    ):
        fail("runtime schedule audit digest differs")
    instruction_binding = instruction.get("token_and_embedding_binding")
    if not isinstance(instruction_binding, Mapping):
        fail("instruction token binding is missing")
    instruction_unsigned = dict(instruction_binding)
    instruction_digest = instruction_unsigned.pop("digest", None)
    if (
        _require_sha(instruction_digest, length=64, label="instruction binding digest")
        != object_sha256(instruction_unsigned)
        or instruction.get("utf8_sha256")
        != hashlib.sha256(str(instruction.get("text")).encode("utf-8")).hexdigest()
    ):
        fail("instruction binding differs")
    _require_exact_keys(
        instruction,
        {
            "text", "utf8_sha256", "matches_stage_b_training_generic_instruction",
            "token_and_embedding_binding",
        },
        label="instruction",
    )
    _require_exact_keys(
        instruction_binding,
        {
            "input_ids_sha256", "attention_mask_sha256", "t5_input_lens_sha256",
            "text_lens", "embedding_sha256", "digest",
        },
        label="instruction token binding",
    )
    _require_exact_keys(
        compute,
        {
            "records_sha256", "latent_outputs_sha256", "epsilon_sha256",
            "adapter_parameter_sha256", "text_embedding_sha256",
            "runtime_schedule_audit_digest", "digest",
        },
        label="compute consensus",
    )
    output_keys = {
        "name", "sha256", "frames", "fps", "hw", "decode_input_latent_sha256",
        "decode_input_latent_shape", "vae_frozen_eval",
    }
    for index, item in enumerate(outputs):
        _require_exact_keys(item, output_keys, label=f"output[{index}]")
        if (
            item.get("frames") != 81
            or item.get("fps") != 25.0
            or item.get("vae_frozen_eval") is not True
            or item.get("hw") != list(EXPECTED_SEALED_OUTPUT_HW)
            or item.get("decode_input_latent_shape")
            != list(EXPECTED_SEALED_DECODE_LATENT_SHAPE)
            or item["hw"]
            != [
                item["decode_input_latent_shape"][3] * 8,
                item["decode_input_latent_shape"][4] * 8,
            ]
        ):
            fail(f"output[{index}] media/latent geometry differs")


def publish_inference_transaction(
    output: Path,
    stage: Path,
    receipt: Optional[Mapping[str, Any]],
    *,
    rank: int,
    world_group: Any,
    rank_zero_error: Optional[str],
) -> None:
    """Verify, create-only publish, reverify, and broadcast rank-zero status."""

    import torch.distributed as dist

    paths = (str(output), str(stage))
    gathered: list[Any] = [None] * dist.get_world_size(group=world_group)
    dist.all_gather_object(gathered, paths, group=world_group)
    if any(item != paths for item in gathered):
        fail("inference publication paths differ across ranks")
    publication: list[Any] = [None]
    if rank == 0:
        try:
            if rank_zero_error is not None:
                fail(rank_zero_error)
            if receipt is None:
                fail("rank-zero inference receipt is missing")
            stage_stat = os.lstat(stage)
            reserved = runtime._OUTPUT_STAGE_IDENTITIES.get(str(stage))
            identity = (
                int(stage_stat.st_dev),
                int(stage_stat.st_ino),
                stat.S_IMODE(stage_stat.st_mode),
            )
            if not stat.S_ISDIR(stage_stat.st_mode) or reserved != identity:
                fail("reserved inference stage identity differs")
            verify_inference_bundle(stage, receipt)
            runtime.fsync_directory(stage)
            runtime._rename_directory_noreplace(stage, output)
            runtime.fsync_directory(output.parent)
            verify_inference_bundle(output, receipt)
            runtime._OUTPUT_STAGE_IDENTITIES.pop(str(stage), None)
            publication[0] = {
                "ok": True,
                "output": str(output),
                "receipt_digest": receipt.get("receipt_digest"),
            }
        except Exception as error:
            publication[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(publication, src=0, group=world_group)
    result = publication[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        fail(f"cannot publish inference transaction: {result!r}")


def collective_rank_call(callback: Any, *, group: Any, label: str) -> Any:
    """Turn rank-local compute exceptions into one symmetric WORLD4 failure."""

    import torch.distributed as dist

    result: Any = None
    error_record: Optional[dict[str, str]] = None
    try:
        result = callback()
    except Exception as error:
        error_record = {
            "error_type": type(error).__name__,
            "error": str(error),
        }
    gathered: list[Any] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, error_record, group=group)
    if any(item is not None for item in gathered):
        fail(f"{label} failed across ranks: {gathered!r}")
    return result


def validate_two_node_two_rank_placement(contract: Any, group: Any) -> dict[str, Any]:
    import torch.distributed as dist

    if contract.world_size != 4 or contract.local_world_size != 2:
        fail("Stage-B inference requires two nodes with two ranks per node")
    host = socket.gethostname()
    if not host or "\x00" in host:
        fail("distributed hostname is invalid")
    local = {
        "rank": contract.rank,
        "local_rank": contract.local_rank,
        "hostname": host,
    }
    gathered: list[Any] = [None] * 4
    dist.all_gather_object(gathered, local, group=group)
    expected_ranks = list(range(4))
    if [item.get("rank") for item in gathered if isinstance(item, Mapping)] != expected_ranks:
        fail("distributed rank placement differs")
    hosts: dict[str, list[tuple[int, int]]] = {}
    for item in gathered:
        if not isinstance(item, Mapping):
            fail("distributed placement row differs")
        hosts.setdefault(str(item["hostname"]), []).append(
            (int(item["rank"]), int(item["local_rank"]))
        )
    ordered_groups = sorted(
        (sorted(values) for values in hosts.values()), key=lambda values: values[0][0]
    )
    if (
        len(hosts) != 2
        or [values[0][0] for values in ordered_groups] != [0, 2]
        or any([local_rank for _, local_rank in values] != [0, 1] for values in ordered_groups)
    ):
        fail("Stage-B inference physical 2x2 mapping differs")
    value = {
        "world_size": 4,
        "local_world_size": 2,
        "nodes": 2,
        "ranks_per_node": 2,
        "ulysses_sp_size": 4,
        "sp4_crosses_nodes": True,
        "rank_hostname_local_rank": gathered,
    }
    return {**value, "digest": object_sha256(value)}


def _load_inference_receipt_directory(
    value: str | Path,
    *,
    arm: str,
    mode: str,
) -> tuple[Path, dict[str, Any]]:
    directory = _plain_absolute_directory(value, label=f"{arm} inference output")
    receipt_path = _plain_absolute_file(
        directory / "receipt.json", label=f"{arm} inference receipt"
    )
    try:
        receipt = json.loads(receipt_path.read_bytes().decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StageBInferenceError(f"cannot parse {arm} inference receipt") from error
    if (
        not isinstance(receipt, dict)
        or set(receipt) != INFERENCE_RECEIPT_KEYS
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("arm") != arm
        or not isinstance(receipt.get("mode_contract"), Mapping)
        or receipt["mode_contract"].get("mode") != mode
    ):
        fail(f"{arm} inference receipt identity differs")
    _validate_embedded_digest(receipt, label=f"{arm} inference")
    if receipt.get("mode_contract") != mode_contract(mode):
        fail(f"{arm} mode contract differs from runtime authority")
    _require_exact_keys(
        receipt.get("source"),
        {"path", "sha256", "model_condition_from_runtime_reencode", "display_and_sealed_row_cross_binding_only"},
        label=f"{arm} source",
    )
    _require_exact_keys(
        receipt.get("tensor_binding"),
        {"clean_sha256", "style_donor_sha256", "reference_sha256_in_order", "epsilon_sha256", "seed"},
        label=f"{arm} tensor binding",
    )
    dataset = receipt.get("dataset")
    _require_exact_keys(
        dataset,
        {
            "root", "parquet_sha256", "materialization_spec_sha256",
            "receipt_sha256", "receipt_digest", "iid", "row_digest",
            "source_video_sha256", "clean_posterior_blob_sha256", "style_id",
            "style_posterior_blob_sha256", "reference_index_set",
            "reference_order", "reference_posterior_blob_sha256_in_order",
            "six_independent_training_materialization_encodes_reused",
        },
        label=f"{arm} dataset",
    )
    source = receipt["source"]
    tensor_binding = receipt["tensor_binding"]
    sealed_row_authority = REGISTERED_IID_SEALED_ROW_AUTHORITY.get(dataset.get("iid"))
    if (
        not isinstance(sealed_row_authority, Mapping)
        or
        source.get("sha256") != dataset.get("source_video_sha256")
        or source.get("model_condition_from_runtime_reencode") is not False
        or source.get("display_and_sealed_row_cross_binding_only") is not True
        or REGISTERED_IID_STYLE.get(dataset.get("iid")) != dataset.get("style_id")
        or dataset.get("materialization_spec_sha256")
        != EXPECTED_DATASET_MATERIALIZATION_SPEC_SHA256
        or dataset.get("parquet_sha256") != EXPECTED_DATASET_PARQUET_SHA256
        or dataset.get("receipt_sha256") != EXPECTED_DATASET_RECEIPT_SHA256
        or dataset.get("receipt_digest") != EXPECTED_DATASET_RECEIPT_DIGEST
        or dataset.get("reference_index_set") != [0, 40, 80]
        or dataset.get("reference_order")
        != REGISTERED_IID_REFERENCE_ORDER.get(dataset.get("iid"))
        or any(
            dataset.get(key) != expected
            for key, expected in sealed_row_authority.items()
        )
        or not isinstance(dataset.get("reference_posterior_blob_sha256_in_order"), list)
        or len(dataset["reference_posterior_blob_sha256_in_order"]) != 3
        or dataset.get("six_independent_training_materialization_encodes_reused") is not True
        or tensor_binding.get("seed") != INFERENCE_SEED
        or not isinstance(tensor_binding.get("reference_sha256_in_order"), list)
        or len(tensor_binding["reference_sha256_in_order"]) != 3
    ):
        fail(f"{arm} sealed dataset/tensor binding differs")
    _require_exact_keys(
        receipt.get("visual_pack"),
        {"role_order", "source_ids", "same_custom_pack_every_query", "online_rgb_corruption_or_vae_reencode", "clean_posterior_used_as_donor"},
        label=f"{arm} visual pack",
    )
    visual = receipt["visual_pack"]
    if visual != {
        "role_order": [
            "forward_noised_registered_style_donor",
            "independent_clean_ref_slot0",
            "independent_clean_ref_slot1",
            "independent_clean_ref_slot2",
            "target",
        ],
        "source_ids": [1, 2, 3, 4, 0],
        "same_custom_pack_every_query": True,
        "online_rgb_corruption_or_vae_reencode": False,
        "clean_posterior_used_as_donor": False,
    }:
        fail(f"{arm} visual pack differs")
    _require_exact_keys(
        receipt.get("execution_counts"),
        {"raw_conditional_forward_calls", "unconditional_forward_calls", "cfg_or_apg_combinations", "scheduler_steps", "decoded_videos"},
        label=f"{arm} execution counts",
    )
    _require_exact_keys(
        receipt.get("anchor_action_display"),
        {"present", "path", "sha256", "full_video_must_be_embedded_by_web_report", "used_as_model_condition", "opened_for_hash_binding_only", "decoded_by_model_runtime", "vae_encoded_by_model_runtime", "routed_to_transformer", "latent_or_rgb_transplanted"},
        label=f"{arm} anchor binding",
    )
    preforward = receipt.get("preforward_input_consensus")
    _require_exact_keys(
        preforward,
        {
            "dataset_iid", "style_id", "clean_sha256", "donor_sha256",
            "reference_sha256_in_order", "epsilon_sha256", "text_binding_digest",
            "adapter_parameter_sha256", "source_ids", "digest",
        },
        label=f"{arm} preforward consensus",
    )
    preforward_unsigned = dict(preforward)
    preforward_digest = preforward_unsigned.pop("digest", None)
    if (
        _require_sha(preforward_digest, length=64, label=f"{arm} preforward digest")
        != object_sha256(preforward_unsigned)
    ):
        fail(f"{arm} preforward consensus digest differs")
    if (
        preforward.get("dataset_iid") != dataset.get("iid")
        or preforward.get("style_id") != dataset.get("style_id")
        or preforward.get("clean_sha256") != tensor_binding.get("clean_sha256")
        or preforward.get("donor_sha256") != tensor_binding.get("style_donor_sha256")
        or preforward.get("reference_sha256_in_order")
        != tensor_binding.get("reference_sha256_in_order")
        or preforward.get("epsilon_sha256") != tensor_binding.get("epsilon_sha256")
        or preforward.get("source_ids") != [1, 2, 3, 4, 0]
    ):
        fail(f"{arm} preforward input links differ")
    model = receipt.get("model")
    _require_exact_keys(
        model,
        {
            "bernini_commit", "veomni_commit", "checkpoint_tree_sha256",
            "checkpoint_content", "pinned_inference_source_files", "single_expert",
        },
        label=f"{arm} model",
    )
    model_runtime = receipt.get("runtime")
    _require_exact_keys(
        model_runtime,
        {
            "physical_placement", "torch", "torch_hip", "transformers", "diffusers",
            "source_self_runtime_dependency",
        },
        label=f"{arm} runtime",
    )
    placement = model_runtime.get("physical_placement")
    _require_exact_keys(
        placement,
        {
            "world_size", "local_world_size", "nodes", "ranks_per_node",
            "ulysses_sp_size", "sp4_crosses_nodes", "rank_hostname_local_rank", "digest",
        },
        label=f"{arm} physical placement",
    )
    _validate_canonical_digest(placement, label=f"{arm} physical placement")
    if any(
        placement.get(key) != expected
        for key, expected in {
            "world_size": 4,
            "local_world_size": 2,
            "nodes": 2,
            "ranks_per_node": 2,
            "ulysses_sp_size": 4,
            "sp4_crosses_nodes": True,
        }.items()
    ):
        fail(f"{arm} physical 2x2 placement differs")
    verify_inference_bundle(directory, receipt)
    return directory, receipt


def _validate_probe_schedule_records(receipt: Mapping[str, Any], *, arm: str) -> None:
    schedule = receipt.get("runtime_schedule_audit")
    expected_schedule = {
        "schedule_sha256": exact40.SCHEDULE_SHA256,
        "fresh_scheduler_instance_for_this_arm": False,
        "scheduler_steps_executed": 0,
        "stateless_registered_schedule_coordinates_only": True,
    }
    if schedule != {**expected_schedule, "digest": object_sha256(expected_schedule)}:
        fail(f"{arm} registered-probe schedule receipt differs")
    records = receipt.get("records")
    outputs = receipt.get("outputs")
    tensor_binding = receipt.get("tensor_binding")
    probe_keys = {
        "optimizer_step_zero_based", "optimizer_step_one_based", "schedule_index",
        "timestep_int64", "sigma", "sigma_float32_be_hex",
        "target_equation_exact", "donor_equation_exact",
        "training_realization_replayed", "predicted_clean_equation",
        "raw_conditional_single_forward", "raw_prediction_dtype",
        "noisy_target_dtype", "sigma_dtype_device", "numeric_program",
        "route_enabled", "prepared_input_digest", "binding",
        "predicted_clean_sha256",
    }
    binding_keys = {
        "epsilon_sha256", "clean_target_sha256", "clean_donor_sha256",
        "noised_target_sha256", "noised_donor_sha256",
        "same_epsilon_object_target_donor", "same_sigma_target_donor",
        "sigma_float32_be_hex", "digest",
    }
    if (
        not isinstance(records, list)
        or len(records) != 4
        or not isinstance(outputs, list)
        or len(outputs) != 4
        or not isinstance(tensor_binding, Mapping)
    ):
        fail(f"{arm} registered-probe record closure differs")
    for item, coordinate, output in zip(
        records, stage_b.validate_registered_schedule(), outputs
    ):
        _require_exact_keys(item, probe_keys, label=f"{arm} probe record")
        coordinate_receipt = coordinate.receipt()
        if any(item.get(key) != expected for key, expected in coordinate_receipt.items()):
            fail(f"{arm} registered-probe coordinate differs")
        if (
            item.get("target_equation_exact") is not True
            or item.get("donor_equation_exact") is not True
            or item.get("training_realization_replayed") is not False
            or item.get("predicted_clean_equation")
            != "x0_hat=x_target_sigma-sigma*v_raw_conditional"
            or item.get("raw_conditional_single_forward") is not True
            or item.get("raw_prediction_dtype") != "torch.bfloat16"
            or item.get("noisy_target_dtype") != "torch.float32"
            or item.get("sigma_dtype_device") != "torch.float32/cpu/0d"
            or item.get("numeric_program")
            != "fp32_noisy-minus-cpu_fp32_sigma-times-bf16_raw_velocity"
            or item.get("route_enabled") is not True
        ):
            fail(f"{arm} registered-probe numeric contract differs")
        _require_sha(
            item.get("prepared_input_digest"), length=64, label=f"{arm} probe prepared SHA"
        )
        predicted = _require_sha(
            item.get("predicted_clean_sha256"), length=64, label=f"{arm} probe output SHA"
        )
        if predicted != output.get("decode_input_latent_sha256"):
            fail(f"{arm} registered-probe decode link differs")
        binding = item.get("binding")
        _require_exact_keys(binding, binding_keys, label=f"{arm} probe binding")
        _validate_canonical_digest(binding, label=f"{arm} probe binding")
        for key in (
            "epsilon_sha256", "clean_target_sha256", "clean_donor_sha256",
            "noised_target_sha256", "noised_donor_sha256",
        ):
            _require_sha(binding.get(key), length=64, label=f"{arm} probe {key}")
        if (
            binding.get("epsilon_sha256") != tensor_binding.get("epsilon_sha256")
            or binding.get("clean_target_sha256") != tensor_binding.get("clean_sha256")
            or binding.get("clean_donor_sha256") != tensor_binding.get("style_donor_sha256")
            or binding.get("same_epsilon_object_target_donor") is not True
            or binding.get("same_sigma_target_donor") is not True
            or binding.get("sigma_float32_be_hex") != coordinate.sigma_float32_be_hex
        ):
            fail(f"{arm} registered-probe same-noise binding differs")


def _validate_full40_schedule_records(receipt: Mapping[str, Any], *, arm: str) -> None:
    schedule = receipt.get("runtime_schedule_audit")
    schedule_keys = {
        "schedule_sha256", "timesteps", "positive_sigmas",
        "positive_sigmas_float32_be_hex", "terminal_sigma",
        "terminal_sigma_float32_be_hex", "fresh_scheduler_instance_for_this_arm",
        "scheduler_steps_executed", "raw_bf16_prediction_passed_without_precast",
        "packed_fp32_target_state_evolved", "scheduler_cursor_pre_post_exact",
        "terminal_step_index", "final_packed_target_sha256",
        "final_unpacked_decode_latent_sha256", "digest",
    }
    _require_exact_keys(schedule, schedule_keys, label=f"{arm} full40 schedule")
    _validate_canonical_digest(schedule, label=f"{arm} full40 schedule")
    if (
        schedule.get("schedule_sha256") != exact40.SCHEDULE_SHA256
        or schedule.get("timesteps") != list(exact40.PINNED_TIMESTEPS)
        or schedule.get("positive_sigmas") != list(exact40.PINNED_POSITIVE_SIGMAS)
        or schedule.get("positive_sigmas_float32_be_hex")
        != list(exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX)
        or schedule.get("terminal_sigma") != 0.0
        or schedule.get("terminal_sigma_float32_be_hex")
        != exact40.TERMINAL_SIGMA_FLOAT32_HEX
        or schedule.get("fresh_scheduler_instance_for_this_arm") is not True
        or schedule.get("scheduler_steps_executed") != 40
        or schedule.get("raw_bf16_prediction_passed_without_precast") is not True
        or schedule.get("packed_fp32_target_state_evolved") is not True
        or schedule.get("scheduler_cursor_pre_post_exact") is not True
        or schedule.get("terminal_step_index") != 40
    ):
        fail(f"{arm} full40 schedule authority differs")
    for key in ("final_packed_target_sha256", "final_unpacked_decode_latent_sha256"):
        _require_sha(schedule.get(key), length=64, label=f"{arm} full40 {key}")
    records = receipt.get("records")
    outputs = receipt.get("outputs")
    record_keys = {
        "schedule_index", "timestep_int64", "sigma_float32_be_hex",
        "registered_training_coordinate", "route_enabled", "target_reanchored",
        "target_matches_stateless_training_formula",
        "initial_target_is_seeded_epsilon_only",
        "donor_same_initial_epsilon_forward_noised",
        "packed_condition_roundtrip_bit_exact", "raw_prediction_dtype",
        "packed_target_pre_post_dtype", "packed_target_shape",
        "scheduler_step_index_before", "scheduler_step_index_after",
        "prepared_input_digest", "noised_style_donor_sha256",
        "packed_target_sha256_before_step", "raw_prediction_sha256",
        "target_sha256_after_step",
    }
    if (
        not isinstance(records, list)
        or len(records) != 40
        or not isinstance(outputs, list)
        or len(outputs) != 1
    ):
        fail(f"{arm} full40 record/output closure differs")
    previous_after: Optional[str] = None
    packed_shape: Optional[list[int]] = None
    for index, item in enumerate(records):
        _require_exact_keys(item, record_keys, label=f"{arm} full40 record {index}")
        if (
            item.get("schedule_index") != index
            or item.get("timestep_int64") != exact40.PINNED_TIMESTEPS[index]
            or item.get("sigma_float32_be_hex")
            != exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index]
            or item.get("registered_training_coordinate")
            is not (index in stage_b.REGISTERED_SCHEDULE_INDICES)
            or item.get("route_enabled") is not True
            or item.get("target_reanchored") is not False
            or item.get("target_matches_stateless_training_formula") is not False
            or item.get("initial_target_is_seeded_epsilon_only") is not (index == 0)
            or item.get("donor_same_initial_epsilon_forward_noised") is not True
            or item.get("packed_condition_roundtrip_bit_exact") is not True
            or item.get("raw_prediction_dtype") != "torch.bfloat16"
            or item.get("packed_target_pre_post_dtype") != "torch.float32"
            or item.get("scheduler_step_index_before") != (None if index == 0 else index)
            or item.get("scheduler_step_index_after") != index + 1
        ):
            fail(f"{arm} full40 coordinate/numeric contract differs at {index}")
        shape = item.get("packed_target_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or shape[0] != 1
            or shape[2] != 64
            or any(type(value) is not int or value <= 0 for value in shape)
            or (packed_shape is not None and shape != packed_shape)
        ):
            fail(f"{arm} full40 packed target shape differs at {index}")
        packed_shape = shape
        for key in (
            "prepared_input_digest", "noised_style_donor_sha256",
            "packed_target_sha256_before_step", "raw_prediction_sha256",
            "target_sha256_after_step",
        ):
            _require_sha(item.get(key), length=64, label=f"{arm} full40 {key} {index}")
        before = item["packed_target_sha256_before_step"]
        if previous_after is not None and before != previous_after:
            fail(f"{arm} full40 target chain differs at {index}")
        previous_after = item["target_sha256_after_step"]
    if (
        previous_after != schedule.get("final_packed_target_sha256")
        or schedule.get("final_unpacked_decode_latent_sha256")
        != outputs[0].get("decode_input_latent_sha256")
    ):
        fail(f"{arm} full40 terminal/decode link differs")


def _validate_arm_adapter(receipt: Mapping[str, Any], *, arm: str) -> str:
    adapter = receipt.get("adapter")
    binding = receipt.get("adapter_runtime_binding")
    common = {
        "arm", "weights", "route_wrapper_installed", "all_adapter_tensors_exact_zero",
        "parameter_sha256", "file_sha256", "training_receipt_sha256",
        "training_receipt_digest", "strict_load",
    }
    wrapper_key = (
        "same_role_q_o_wrapper_as_trained"
        if arm == "frozen_base"
        else "same_role_q_o_wrapper_as_frozen_base"
    )
    _require_exact_keys(adapter, common | {wrapper_key}, label=f"{arm} adapter")
    if not isinstance(binding, Mapping):
        fail(f"{arm} adapter runtime binding is missing")
    parameter_sha = _require_sha(
        adapter.get("parameter_sha256"), length=64, label=f"{arm} parameter SHA"
    )
    if adapter.get(wrapper_key) is not True or adapter.get("route_wrapper_installed") is not True:
        fail(f"{arm} adapter wrapper contract differs")
    if arm == "frozen_base":
        if (
            adapter.get("arm") != arm
            or adapter.get("weights") != "all_adapter_tensors_exact_zero"
            or adapter.get("all_adapter_tensors_exact_zero") is not True
            or any(adapter.get(key) is not None for key in (
                "file_sha256", "training_receipt_sha256", "training_receipt_digest", "strict_load"
            ))
        ):
            fail("frozen-base adapter contract differs")
        return parameter_sha
    if (
        adapter.get("arm") != arm
        or adapter.get("weights") != "strict_stage_b_training_adapter"
        or adapter.get("all_adapter_tensors_exact_zero") is not False
    ):
        fail("trained adapter contract differs")
    file_sha = _require_sha(adapter.get("file_sha256"), length=64, label="trained adapter file SHA")
    training_receipt_sha = _require_sha(
        adapter.get("training_receipt_sha256"), length=64, label="training receipt SHA"
    )
    training_receipt_digest = _require_sha(
        adapter.get("training_receipt_digest"), length=64, label="training receipt digest"
    )
    strict = adapter.get("strict_load")
    strict_keys = {
        "schema_version", "path", "file_sha256", "exact40_schedule_sha256",
        "registered_schedule_indices", "metadata", "tensor_count",
        "tensor_names_sha256", "strict_tensor_closure", "base_parameters_frozen",
        "forward_noising_only", "inversion_claimed", "digest",
    }
    _require_exact_keys(strict, strict_keys, label="trained strict load")
    _validate_canonical_digest(strict, label="trained strict load")
    metadata = strict.get("metadata")
    expected_metadata = {
        "schema_version": stage_b.ADAPTER_FILE_SCHEMA,
        "role_adapter_schema_version": "bernini-source-self-role-repaint-adapter-v1",
        "block_indices_json": canonical_json_bytes(list(range(23))).decode("ascii"),
        "projections_json": canonical_json_bytes(["attn1.to_q", "attn1.to_out.0"]).decode("ascii"),
        "target_row_only": "true",
        "role_embedding": "donor_reference_target",
        "lora_rank": str(stage_b.LORA_RANK),
        "lora_alpha_hex": stage_b.LORA_ALPHA.hex(),
        "exact40_schedule_sha256": stage_b.EXPECTED_EXACT40_SCHEDULE_SHA256,
        "registered_schedule_indices_json": canonical_json_bytes(
            list(stage_b.REGISTERED_SCHEDULE_INDICES)
        ).decode("ascii"),
        "target_and_donor_same_epsilon": "true",
        "forward_noising_only": "true",
        "inversion_claimed": "false",
        "matched_carrier_runtime_required": "true",
    }
    if (
        file_sha != EXPECTED_STAGE_B_ADAPTER_SHA256
        or training_receipt_sha != EXPECTED_STAGE_B_TRAINING_RECEIPT_SHA256
        or training_receipt_digest != EXPECTED_STAGE_B_TRAINING_RECEIPT_DIGEST
        or parameter_sha != EXPECTED_STAGE_B_FINAL_PARAMETER_SHA256
        or strict.get("schema_version") != "bernini-source-noised-carrier-strict-load-v1"
        or not isinstance(strict.get("path"), str)
        or not Path(strict["path"]).is_absolute()
        or Path(strict["path"]).name != "adapter.safetensors"
        or strict.get("file_sha256") != file_sha
        or strict.get("exact40_schedule_sha256") != stage_b.EXPECTED_EXACT40_SCHEDULE_SHA256
        or strict.get("registered_schedule_indices") != list(stage_b.REGISTERED_SCHEDULE_INDICES)
        or strict.get("metadata") != expected_metadata
        or type(strict.get("tensor_count")) is not int
        or strict.get("tensor_count") <= 0
        or _SHA256.fullmatch(str(strict.get("tensor_names_sha256"))) is None
        or strict.get("strict_tensor_closure") is not True
        or strict.get("base_parameters_frozen") is not True
        or strict.get("forward_noising_only") is not True
        or strict.get("inversion_claimed") is not False
        or binding.get("training_final_parameter_sha256") != parameter_sha
        or binding.get("training_artifacts_sha256") != {
            "adapter.safetensors": EXPECTED_STAGE_B_ADAPTER_SHA256,
            "optimizer.pt": EXPECTED_STAGE_B_OPTIMIZER_SHA256,
            "history.json": EXPECTED_STAGE_B_HISTORY_SHA256,
        }
    ):
        fail("trained strict-load/parameter provenance differs")
    return parameter_sha


def validate_pair_receipts(
    base_dir: str | Path,
    trained_dir: str | Path,
    mode: str,
) -> dict[str, Any]:
    """CPU/model-free strict validator for one frozen/trained evidence pair."""

    if mode not in MODES:
        fail("pair verification mode differs")
    base_path, base = _load_inference_receipt_directory(
        base_dir, arm="frozen_base", mode=mode
    )
    trained_path, trained = _load_inference_receipt_directory(
        trained_dir, arm="trained", mode=mode
    )
    adapter_parameter_sha256: dict[str, str] = {}
    for arm_name, receipt in (("frozen_base", base), ("trained", trained)):
        if mode == "registered-probes":
            _validate_probe_schedule_records(receipt, arm=arm_name)
        else:
            _validate_full40_schedule_records(receipt, arm=arm_name)
        adapter_parameter_sha256[arm_name] = _validate_arm_adapter(
            receipt, arm=arm_name
        )
        if receipt.get("mode_contract") != mode_contract(mode):
            fail(f"{arm_name} mode contract differs from runtime authority")
        counts = receipt.get("execution_counts")
        expected_calls = 4 if mode == "registered-probes" else 40
        expected_steps = 0 if mode == "registered-probes" else 40
        expected_decodes = 4 if mode == "registered-probes" else 1
        if (
            not isinstance(counts, Mapping)
            or counts.get("raw_conditional_forward_calls") != expected_calls
            or counts.get("unconditional_forward_calls") != 0
            or counts.get("cfg_or_apg_combinations") != 0
            or counts.get("scheduler_steps") != expected_steps
            or counts.get("decoded_videos") != expected_decodes
        ):
            fail(f"{arm_name} execution count contract differs")
        anchor = receipt.get("anchor_action_display")
        if anchor != {
            "present": False,
            "path": None,
            "sha256": None,
            "full_video_must_be_embedded_by_web_report": False,
            "used_as_model_condition": False,
            "opened_for_hash_binding_only": False,
            "decoded_by_model_runtime": False,
            "vae_encoded_by_model_runtime": False,
            "routed_to_transformer": False,
            "latent_or_rgb_transplanted": False,
        }:
            fail(f"{arm_name} anchor must be absent from model runtime")
        if any(
            receipt.get(key) is not False
            for key in ("inversion_claimed", "method_success_claimed", "scientific_claim_authorized")
        ):
            fail(f"{arm_name} global scientific claim flags differ")
        runtime_record = receipt.get("runtime")
        dependency = (
            runtime_record.get("source_self_runtime_dependency")
            if isinstance(runtime_record, Mapping)
            else None
        )
        if (
            not isinstance(dependency, Mapping)
            or dependency.get("sha256") != EXPECTED_SOURCE_SELF_RUNTIME_SHA256
            or Path(str(dependency.get("path"))).name != "source_self_runtime.py"
        ):
            fail(f"{arm_name} audited source_self_runtime dependency differs")
        adapter_value = receipt.get("adapter")
        adapter_binding = receipt.get("adapter_runtime_binding")
        if arm_name == "frozen_base":
            expected_binding = {
                "training_dataset_exact_match": None,
                "training_model_exact_match": None,
                "training_artifacts_sha256": None,
                "training_final_parameter_sha256": None,
                "training_noise_seeds": None,
                "training_epsilon_sha256": None,
                "inference_seed_absent_from_training_noise_seeds": None,
                "inference_epsilon_absent_from_training_realizations": None,
                "reason": "frozen_base_does_not_open_training_artifacts",
            }
            if adapter_binding != expected_binding:
                fail("frozen-base training-artifact binding must be exact null closure")
        else:
            _require_exact_keys(
                adapter_binding,
                {
                    "training_dataset_exact_match", "training_model_exact_match",
                    "training_artifacts_sha256", "training_final_parameter_sha256",
                    "training_noise_seeds",
                    "training_epsilon_sha256",
                    "inference_seed_absent_from_training_noise_seeds",
                    "inference_epsilon_absent_from_training_realizations", "digest",
                },
                label="trained adapter runtime binding",
            )
            _validate_canonical_digest(adapter_binding, label="trained adapter runtime binding")
            seeds = adapter_binding.get("training_noise_seeds")
            epsilons = adapter_binding.get("training_epsilon_sha256")
            if (
                not isinstance(seeds, list)
                or len(seeds) != 8
                or len(set(seeds)) != 8
                or not isinstance(epsilons, list)
                or len(epsilons) != 8
                or len(set(epsilons)) != 8
                or any(_SHA256.fullmatch(str(item)) is None for item in epsilons)
                or adapter_binding.get("training_final_parameter_sha256")
                != adapter_parameter_sha256[arm_name]
            ):
                fail("trained adapter eight training realizations differ")
    if adapter_parameter_sha256["frozen_base"] == adapter_parameter_sha256["trained"]:
        fail("base/trained adapter parameter SHA must differ")
    identical_fields = (
        "mode_contract",
        "source",
        "dataset",
        "tensor_binding",
        "instruction",
        "visual_pack",
        "execution_counts",
        "anchor_action_display",
        "model",
        "runtime",
        "method_source_revision",
        "method_source_archive_sha256",
        "inversion_claimed",
        "method_success_claimed",
        "scientific_claim_authorized",
    )
    unequal = [key for key in identical_fields if base.get(key) != trained.get(key)]
    if unequal:
        fail(f"base/trained matched inputs or runtime differ: {unequal}")
    base_schedule = base.get("runtime_schedule_audit")
    trained_schedule = trained.get("runtime_schedule_audit")
    if not isinstance(base_schedule, Mapping) or not isinstance(trained_schedule, Mapping):
        fail("pair runtime schedule audit is missing")
    base_schedule_fixed = dict(base_schedule)
    trained_schedule_fixed = dict(trained_schedule)
    for value in (base_schedule_fixed, trained_schedule_fixed):
        value.pop("digest", None)
        if mode != "registered-probes":
            value.pop("final_packed_target_sha256", None)
            value.pop("final_unpacked_decode_latent_sha256", None)
    if base_schedule_fixed != trained_schedule_fixed:
        fail("base/trained fixed runtime schedule differs")
    base_preforward = base.get("preforward_input_consensus")
    trained_preforward = trained.get("preforward_input_consensus")
    if not isinstance(base_preforward, Mapping) or not isinstance(trained_preforward, Mapping):
        fail("pair preforward consensus is missing")
    for label, value in (("base", base_preforward), ("trained", trained_preforward)):
        unsigned = dict(value)
        declared = unsigned.pop("digest", None)
        if _require_sha(declared, length=64, label=f"{label} preforward digest") != object_sha256(unsigned):
            fail(f"{label} preforward digest differs")
    base_prefixed = dict(base_preforward)
    trained_prefixed = dict(trained_preforward)
    for value in (base_prefixed, trained_prefixed):
        value.pop("digest", None)
        value.pop("adapter_parameter_sha256", None)
    if base_prefixed != trained_prefixed:
        fail("base/trained fixed preforward inputs differ")
    base_adapter = base.get("adapter")
    trained_adapter = trained.get("adapter")
    trained_binding = trained.get("adapter_runtime_binding")
    if (
        not isinstance(base_adapter, Mapping)
        or base_adapter.get("arm") != "frozen_base"
        or base_adapter.get("weights") != "all_adapter_tensors_exact_zero"
        or base_adapter.get("all_adapter_tensors_exact_zero") is not True
        or base_adapter.get("route_wrapper_installed") is not True
        or not isinstance(trained_adapter, Mapping)
        or trained_adapter.get("arm") != "trained"
        or trained_adapter.get("weights") != "strict_stage_b_training_adapter"
        or trained_adapter.get("all_adapter_tensors_exact_zero") is not False
        or trained_adapter.get("route_wrapper_installed") is not True
        or not isinstance(trained_binding, Mapping)
        or trained_binding.get("training_dataset_exact_match") is not True
        or trained_binding.get("training_model_exact_match") is not True
        or trained_binding.get("inference_seed_absent_from_training_noise_seeds") is not True
        or trained_binding.get("inference_epsilon_absent_from_training_realizations") is not True
    ):
        fail("base/trained adapter arm contract differs")
    if (
        base_preforward.get("adapter_parameter_sha256")
        != base_adapter.get("parameter_sha256")
        or trained_preforward.get("adapter_parameter_sha256")
        != trained_adapter.get("parameter_sha256")
    ):
        fail("pair preforward adapter parameter link differs")
    base_compute = base.get("compute_consensus")
    trained_compute = trained.get("compute_consensus")
    if not isinstance(base_compute, Mapping) or not isinstance(trained_compute, Mapping):
        fail("pair compute consensus is missing")
    for key in ("epsilon_sha256", "text_embedding_sha256"):
        if base_compute.get(key) != trained_compute.get(key):
            fail(f"base/trained compute input differs: {key}")
    base_records = base.get("records")
    trained_records = trained.get("records")
    if not isinstance(base_records, list) or not isinstance(trained_records, list):
        fail("pair records are missing")
    if mode == "registered-probes":
        if len(base_records) != 4 or len(trained_records) != 4:
            fail("registered pair must contain four records per arm")
        indices = []
        for left, right in zip(base_records, trained_records):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                fail("registered pair record is not an object")
            indices.append(left.get("schedule_index"))
            left_common = dict(left)
            right_common = dict(right)
            left_common.pop("predicted_clean_sha256", None)
            right_common.pop("predicted_clean_sha256", None)
            if left_common != right_common:
                fail("registered base/trained coordinate inputs differ")
        if indices != list(stage_b.REGISTERED_SCHEDULE_INDICES):
            fail("registered pair schedule order differs")
        for arm_name, receipt, rows in (
            ("frozen_base", base, base_records),
            ("trained", trained, trained_records),
        ):
            if [row.get("predicted_clean_sha256") for row in rows] != [
                item.get("decode_input_latent_sha256") for item in receipt.get("outputs", [])
            ]:
                fail(f"{arm_name} probe record/decode latent links differ")
    else:
        if len(base_records) != 40 or len(trained_records) != 40:
            fail("full40 pair must contain forty records per arm")
        mutable = {
            "packed_target_sha256_before_step",
            "raw_prediction_sha256",
            "target_sha256_after_step",
            "prepared_input_digest",
        }
        for arm_name, rows in (("frozen_base", base_records), ("trained", trained_records)):
            previous_after: Optional[str] = None
            for index, row in enumerate(rows):
                if (
                    not isinstance(row, Mapping)
                    or row.get("schedule_index") != index
                    or row.get("route_enabled") is not True
                    or row.get("target_matches_stateless_training_formula") is not False
                    or row.get("target_reanchored") is not False
                    or row.get("initial_target_is_seeded_epsilon_only") is not (index == 0)
                    or row.get("packed_condition_roundtrip_bit_exact") is not True
                ):
                    fail(f"{arm_name} full40 record contract differs at {index}")
                before = _require_sha(
                    row.get("packed_target_sha256_before_step"),
                    length=64,
                    label=f"{arm_name} target before {index}",
                )
                _require_sha(
                    row.get("raw_prediction_sha256"),
                    length=64,
                    label=f"{arm_name} prediction {index}",
                )
                after = _require_sha(
                    row.get("target_sha256_after_step"),
                    length=64,
                    label=f"{arm_name} target after {index}",
                )
                if previous_after is not None and before != previous_after:
                    fail(f"{arm_name} full40 target chain breaks at {index}")
                previous_after = after
        for index, (left, right) in enumerate(zip(base_records, trained_records)):
            if left.get("noised_style_donor_sha256") != right.get("noised_style_donor_sha256"):
                fail(f"base/trained donor ladder differs at {index}")
            left_common = {key: value for key, value in left.items() if key not in mutable}
            right_common = {key: value for key, value in right.items() if key not in mutable}
            if left_common != right_common:
                fail(f"base/trained full40 fixed coordinate differs at {index}")
            if index == 0 and left.get("prepared_input_digest") != right.get("prepared_input_digest"):
                fail("base/trained full40 initial prepared input differs")
        if (
            base_records[0].get("packed_target_sha256_before_step")
            != trained_records[0].get("packed_target_sha256_before_step")
        ):
            fail("base/trained full40 initial epsilon state differs")
        for arm_name, receipt in (("frozen_base", base), ("trained", trained)):
            schedule = receipt.get("runtime_schedule_audit")
            outputs = receipt.get("outputs")
            if (
                not isinstance(schedule, Mapping)
                or not isinstance(outputs, list)
                or len(outputs) != 1
                or schedule.get("final_unpacked_decode_latent_sha256")
                != outputs[0].get("decode_input_latent_sha256")
            ):
                fail(f"{arm_name} full40 final decode latent link differs")
    base_outputs = base.get("outputs")
    trained_outputs = trained.get("outputs")
    if not isinstance(base_outputs, list) or not isinstance(trained_outputs, list):
        fail("pair output records are missing")
    output_projection = lambda rows: [
        {
            key: item.get(key)
            for key in ("name", "frames", "fps", "hw", "decode_input_latent_shape", "vae_frozen_eval")
        }
        for item in rows
        if isinstance(item, Mapping)
    ]
    if output_projection(base_outputs) != output_projection(trained_outputs):
        fail("base/trained decoded output geometry differs")
    pair = {
        "schema_version": PAIR_RECEIPT_SCHEMA,
        "complete": True,
        "mode": mode,
        "base": {
            "directory": str(base_path),
            "receipt_sha256": file_sha256(base_path / "receipt.json"),
            "receipt_digest": base["receipt_digest"],
        },
        "trained": {
            "directory": str(trained_path),
            "receipt_sha256": file_sha256(trained_path / "receipt.json"),
            "receipt_digest": trained["receipt_digest"],
        },
        "same_sealed_source_dataset_prompt_seed_epsilon_scheduler_route": True,
        "only_adapter_parameter_values_intentionally_differ": True,
        "anchor_used_as_condition": False,
        "inversion_claimed": False,
        "method_success_claimed": False,
        "scientific_claim_authorized": False,
    }
    return {**pair, "receipt_digest": object_sha256(pair)}


def verify_pair_main(argv: Sequence[str]) -> int:
    validate_runtime_dependency()
    parser = argparse.ArgumentParser(
        prog=f"{Path(__file__).name} verify-pair",
        description="Model-free strict validation of a Stage-B base/trained pair.",
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--trained-dir", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    pair = validate_pair_receipts(args.base_dir, args.trained_dir, args.mode)
    output = Path(args.output).expanduser()
    if (
        not output.is_absolute()
        or output.suffix != ".json"
        or output.is_symlink()
        or output.parent.resolve(strict=True) != output.parent
    ):
        fail("pair receipt output must be a fresh canonical absolute .json file")
    raw = canonical_json_bytes(pair) + b"\n"
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except OSError as error:
        raise StageBInferenceError(f"cannot create pair receipt: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        runtime.fsync_directory(output.parent)
        if output.is_symlink() or output.read_bytes() != raw:
            fail("published pair receipt bytes differ")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(canonical_json_bytes(pair).decode("ascii"), flush=True)
    return 0


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "verify-pair":
        return verify_pair_main(arguments[1:])
    args = build_parser().parse_args(arguments)
    validate_cli(args)
    runtime_dependency = validate_runtime_dependency()
    source = _plain_absolute_file(args.source_video, label="source display video")
    source_sha = file_sha256(source)
    if source_sha != args.expected_source_sha256:
        fail("source display video SHA differs")
    anchor: Optional[Path] = None
    anchor_sha: Optional[str] = None
    if args.anchor_action_video is not None:
        anchor = _plain_absolute_file(args.anchor_action_video, label="anchor action display video")
        anchor_sha = file_sha256(anchor)
        if anchor_sha != _require_sha(args.expected_anchor_action_sha256, length=64, label="anchor action SHA"):
            fail("anchor action display video SHA differs")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy_train.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = legacy_train.validate_checkpoint(args.checkpoint)
        checkpoint_identity = validate_checkpoint_content(
            checkpoint,
            args.checkpoint_content_manifest,
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        )
    except legacy_train.TrainingContractError as error:
        raise StageBInferenceError(str(error)) from error
    if args.expected_checkpoint_tree_sha256 != legacy_train.CHECKPOINT_TREE_SHA256:
        fail("checkpoint tree pin differs")
    try:
        inference_source_files = legacy_infer.validate_inference_source_files(bernini_root)
    except legacy_infer.InferenceContractError as error:
        raise StageBInferenceError(str(error)) from error
    legacy_train.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode

    topology = runtime.WORLD4_DP1_SP4
    contract = runtime.distributed_contract(topology=topology)
    device = runtime.initialise_distributed(contract)
    parallel = runtime.validate_parallel_state(contract, init_parallel_state(ulysses_size=SP_SIZE))
    physical_placement = validate_two_node_two_rank_placement(
        contract, parallel.world_group
    )
    output, stage = runtime.prepare_output_transaction(
        args.output, contract.rank, parallel.world_group
    )
    if transformer_config.get("num_attention_heads") != 12:
        fail("Bernini 1.3B attention geometry differs")
    bound = load_bound_dataset_row(
        dataset_root=args.dataset_root,
        expected_spec_sha256=args.expected_materialization_spec_sha256,
        expected_parquet_sha256=args.expected_dataset_parquet_sha256,
        expected_receipt_sha256=args.expected_dataset_receipt_sha256,
        expected_receipt_digest=args.expected_dataset_receipt_digest,
        iid=args.iid,
        style_id=args.style_id,
        checkpoint=checkpoint,
    )
    if bound.row.source_video_sha256 != source_sha:
        fail("display source SHA differs from sealed materialized row")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    epsilon = torch.randn(
        tuple(bound.clean.shape), generator=generator, dtype=torch.float32
    ).contiguous()
    epsilon_sha = runtime.tensor_sha256(epsilon)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy_train.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy_train.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).to(device)
    renderer.requires_grad_(False)
    renderer.eval()
    transformer = renderer.diff_dec.transformer
    if transformer is None or renderer.diff_dec.transformer_2 is not None:
        fail("Stage-B inference requires only transformer_1")
    bundle = (
        resolve_stage_b_adapter(
            args.adapter_checkpoint,
            expected_adapter_sha256=args.expected_adapter_sha256,
            expected_receipt_sha256=args.expected_training_receipt_sha256,
        )
        if args.arm == "trained"
        else None
    )
    adapter_runtime_binding = (
        validate_adapter_runtime_binding(
            bundle,
            dataset=bound,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
            inference_seed=args.seed,
            inference_epsilon_sha256=epsilon_sha,
        )
        if bundle is not None
        else {
            "training_dataset_exact_match": None,
            "training_model_exact_match": None,
            "training_artifacts_sha256": None,
            "training_final_parameter_sha256": None,
            "training_noise_seeds": None,
            "training_epsilon_sha256": None,
            "inference_seed_absent_from_training_noise_seeds": None,
            "inference_epsilon_absent_from_training_realizations": None,
            "reason": "frozen_base_does_not_open_training_artifacts",
        }
    )
    adapter, adapter_receipt = install_arm_adapter(transformer, arm=args.arm, bundle=bundle)
    runtime.digest_consensus(
        adapter_receipt["parameter_sha256"],
        group=parallel.world_group,
        expected_count=4,
        label="inference adapter",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy_infer.tokenizer_load_kwargs()
    )
    text = runtime.tokenize_generic_instruction(tokenizer, args.instruction, device)
    with torch.inference_mode():
        text_lens, text_embs = renderer.get_t5_text_embeddings(
            text["input_ids"], text["attention_mask"], text["t5_input_lens"]
        )
    text_binding = {
        "input_ids_sha256": runtime.tensor_sha256(text["input_ids"]),
        "attention_mask_sha256": runtime.tensor_sha256(text["attention_mask"]),
        "t5_input_lens_sha256": runtime.tensor_sha256(text["t5_input_lens"]),
        "text_lens": list(text_lens),
        "embedding_sha256": runtime.tensor_sha256(text_embs),
    }
    text_binding["digest"] = object_sha256(text_binding)
    preforward_binding = {
        "dataset_iid": bound.row.iid,
        "style_id": bound.style_id,
        "clean_sha256": runtime.tensor_sha256(bound.clean),
        "donor_sha256": runtime.tensor_sha256(bound.donor),
        "reference_sha256_in_order": [
            runtime.tensor_sha256(item) for item in bound.references
        ],
        "epsilon_sha256": epsilon_sha,
        "text_binding_digest": text_binding["digest"],
        "adapter_parameter_sha256": adapter_receipt["parameter_sha256"],
        "source_ids": [1, 2, 3, 4, 0],
    }
    preforward_binding["digest"] = object_sha256(preforward_binding)
    runtime.digest_consensus(
        preforward_binding["digest"],
        group=parallel.world_group,
        expected_count=4,
        label="inference preforward input bundle",
    )
    renderer.t5_text_encoder = None
    del tokenizer, text
    torch.cuda.empty_cache()
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    def rank_compute() -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
        if args.mode == "registered-probes":
            values, rows = _registered_probe_latents(
                renderer=renderer,
                transformer=transformer,
                adapter=adapter,
                bound=bound,
                epsilon=epsilon,
                rope=rope,
                device=device,
                text_lens=text_lens,
                text_embs=text_embs,
                sp_rank=contract.sp_rank,
                world_group=parallel.world_group,
            )
            schedule = {
                "schedule_sha256": exact40.SCHEDULE_SHA256,
                "fresh_scheduler_instance_for_this_arm": False,
                "scheduler_steps_executed": 0,
                "stateless_registered_schedule_coordinates_only": True,
            }
            schedule["digest"] = object_sha256(schedule)
        else:
            latent, rows, schedule = _full40_rollout(
                renderer=renderer,
                transformer=transformer,
                adapter=adapter,
                bound=bound,
                epsilon=epsilon,
                rope=rope,
                device=device,
                text_lens=text_lens,
                text_embs=text_embs,
                sp_rank=contract.sp_rank,
                mode=args.mode,
                world_group=parallel.world_group,
            )
            values = [latent]
        renderer.to("cpu")
        torch.cuda.empty_cache()
        return values, rows, schedule

    latent_outputs, records, schedule_audit = collective_rank_call(
        rank_compute,
        group=parallel.world_group,
        label="Stage-B inference compute",
    )
    compute_binding = {
        "records_sha256": object_sha256(records),
        "latent_outputs_sha256": [runtime.tensor_sha256(item) for item in latent_outputs],
        "epsilon_sha256": epsilon_sha,
        "adapter_parameter_sha256": adapter_receipt["parameter_sha256"],
        "text_embedding_sha256": text_binding["embedding_sha256"],
        "runtime_schedule_audit_digest": schedule_audit["digest"],
    }
    compute_binding["digest"] = object_sha256(compute_binding)
    runtime.digest_consensus(
        compute_binding["digest"],
        group=parallel.world_group,
        expected_count=4,
        label="inference compute result",
    )
    rank_zero_error: Optional[str] = None
    if contract.rank == 0:
        try:
            vae = AutoencoderKLWan.from_pretrained(
                str(checkpoint), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
            ).to(device)
            vae.requires_grad_(False)
            vae.eval()
            output_names = OUTPUT_FILES_BY_MODE[args.mode]
            output_records = []
            for name, latent_value in zip(output_names, latent_outputs):
                path = stage / name
                latent_shape = tuple(int(item) for item in latent_value.shape)
                if (
                    len(latent_shape) != 5
                    or latent_shape[:3] != (1, 16, 21)
                    or latent_value.dtype != torch.float32
                    or latent_value.device != device
                    or not latent_value.is_contiguous()
                    or not bool(torch.isfinite(latent_value).all().item())
                ):
                    fail("VAE decode input latent dtype/device/geometry differs")
                decode_input_sha = runtime.tensor_sha256(latent_value)
                expected_hw = [latent_shape[3] * 8, latent_shape[4] * 8]
                with torch.no_grad():
                    video = _vae_decode(vae, latent_value)
                save_validated_vae_decoded_clip(
                    video,
                    output_path=path,
                    expected_height=expected_hw[0],
                    expected_width=expected_hw[1],
                    fps=int(FPS),
                    save_output_fn=save_output,
                )
                frames, reported_fps, hw = __import__(
                    "tools.materialize_vae", fromlist=["_decode_exact_video"]
                )._decode_exact_video(path)
                legacy_infer.validate_exact_video_metadata(int(frames.shape[0]), reported_fps)
                if list(hw) != expected_hw:
                    fail("decoded MP4 H/W differs from latent x8 geometry")
                output_records.append(
                    {
                        "name": name,
                        "sha256": file_sha256(path),
                        "frames": 81,
                        "fps": 25.0,
                        "hw": list(hw),
                        "decode_input_latent_sha256": decode_input_sha,
                        "decode_input_latent_shape": list(latent_shape),
                        "vae_frozen_eval": True,
                    }
                )
            artifact_hashes = {item["name"]: item["sha256"] for item in output_records}
            receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "complete": True,
            "arm": args.arm,
            "mode_contract": mode_contract(args.mode),
            "source": {
                "path": str(source),
                "sha256": source_sha,
                "model_condition_from_runtime_reencode": False,
                "display_and_sealed_row_cross_binding_only": True,
            },
            "dataset": {
                "root": str(bound.dataset.root),
                "parquet_sha256": bound.dataset.parquet_sha256,
                "materialization_spec_sha256": args.expected_materialization_spec_sha256,
                "receipt_sha256": bound.dataset.receipt_sha256,
                "receipt_digest": bound.dataset.receipt_digest,
                "iid": bound.row.iid,
                "row_digest": bound.row.row_digest,
                "source_video_sha256": bound.row.source_video_sha256,
                "clean_posterior_blob_sha256": bound.clean_blob_sha256,
                "style_id": bound.style_id,
                "style_posterior_blob_sha256": bound.donor_blob_sha256,
                "reference_index_set": list(role.REFERENCE_RGB_INDICES),
                "reference_order": list(bound.row.reference_order),
                "reference_posterior_blob_sha256_in_order": list(bound.reference_blob_sha256),
                "six_independent_training_materialization_encodes_reused": True,
            },
            "tensor_binding": {
                "clean_sha256": runtime.tensor_sha256(bound.clean),
                "style_donor_sha256": runtime.tensor_sha256(bound.donor),
                "reference_sha256_in_order": [runtime.tensor_sha256(item) for item in bound.references],
                "epsilon_sha256": epsilon_sha,
                "seed": args.seed,
            },
            "instruction": {
                "text": args.instruction,
                "utf8_sha256": hashlib.sha256(args.instruction.encode("utf-8")).hexdigest(),
                "matches_stage_b_training_generic_instruction": True,
                "token_and_embedding_binding": text_binding,
            },
            "visual_pack": {
                "role_order": [
                    "forward_noised_registered_style_donor",
                    "independent_clean_ref_slot0",
                    "independent_clean_ref_slot1",
                    "independent_clean_ref_slot2",
                    "target",
                ],
                "source_ids": [1, 2, 3, 4, 0],
                "same_custom_pack_every_query": True,
                "online_rgb_corruption_or_vae_reencode": False,
                "clean_posterior_used_as_donor": False,
            },
            "execution_counts": {
                "raw_conditional_forward_calls": 4 if args.mode == "registered-probes" else 40,
                "unconditional_forward_calls": 0,
                "cfg_or_apg_combinations": 0,
                "scheduler_steps": 0 if args.mode == "registered-probes" else 40,
                "decoded_videos": len(output_names),
            },
            "runtime_schedule_audit": schedule_audit,
            "compute_consensus": compute_binding,
            "preforward_input_consensus": preforward_binding,
            "adapter": adapter_receipt,
            "adapter_runtime_binding": adapter_runtime_binding,
            "records": records,
            "outputs": output_records,
            "artifacts": artifact_hashes,
            "anchor_action_display": {
                "present": anchor is not None,
                "path": str(anchor) if anchor is not None else None,
                "sha256": anchor_sha,
                "full_video_must_be_embedded_by_web_report": anchor is not None,
                "used_as_model_condition": False,
                "opened_for_hash_binding_only": anchor is not None,
                "decoded_by_model_runtime": False,
                "vae_encoded_by_model_runtime": False,
                "routed_to_transformer": False,
                "latent_or_rgb_transplanted": False,
            },
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                "checkpoint_content": checkpoint_identity,
                "pinned_inference_source_files": inference_source_files,
                "single_expert": "transformer_1",
            },
            "runtime": {
                "physical_placement": physical_placement,
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
                "source_self_runtime_dependency": runtime_dependency,
            },
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
            "inversion_claimed": False,
            "method_success_claimed": False,
            "scientific_claim_authorized": False,
            }
            receipt["receipt_digest"] = object_sha256(receipt)
            _atomic_json(stage / "receipt.json", receipt)
            verify_inference_bundle(stage, receipt)
            print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
        except Exception as error:
            rank_zero_error = f"{type(error).__name__}: {error}"
    publish_inference_transaction(
        output,
        stage,
        receipt if contract.rank == 0 and rank_zero_error is None else None,
        rank=contract.rank,
        world_group=parallel.world_group,
        rank_zero_error=rank_zero_error,
    )
    adapter.restore()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
