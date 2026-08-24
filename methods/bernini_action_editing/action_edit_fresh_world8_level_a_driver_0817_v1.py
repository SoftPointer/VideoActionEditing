#!/usr/bin/env python3
"""Sealed one-shot driver for the 0817 PRE_D0 Level-A checkpoint consumer.

This bridge has exactly two operations:

* ``run`` authenticates the eleven-file deployment release, executes the
  checkpoint consumer from the authenticated source bytes, consumes P2 in one
  fresh WORLD8 DP2xSP4 process group, and atomically publishes the canonical
  ``bundle.consumer_receipt`` written by rank zero;
* ``compare`` authenticates two independently published WORLD8 receipts,
  invokes the frozen consumer's A/B comparator, and atomically publishes a
  non-promotable parity receipt.

It does not run the Bernini renderer, denoise forty steps, decode or emit MP4,
perform product inference, train, select a checkpoint, or authorize promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys
import time
import types
from typing import Any, Mapping, NoReturn, Sequence


METHOD = "bernini-action-edit-fresh-world8-level-a-driver-0817-v1"
AUTHORITY = "PRE_D0_ENGINEERING_ONLY"
RELEASE_SCHEMA = "bernini-action-edit-fresh-world8-level-a-release-manifest-v1"
RELEASE_MEMBER_ROOT = "methods/bernini_action_editing"
PARITY_RECEIPT_SCHEMA = (
    "bernini-action-edit-fresh-world8-level-a-sealed-parity-receipt-v2"
)
LAUNCH_AUTHORITY_SCHEMA = (
    "bernini-action-edit-fresh-world8-level-a-launch-authority-core-v2"
)
ATTEMPT_INTENT_SCHEMA = (
    "bernini-action-edit-fresh-world8-level-a-attempt-intent-v2"
)
LAUNCH_BINDING_SCHEMA = (
    "bernini-action-edit-fresh-world8-level-a-launch-binding-v2"
)
TERMINAL_AUTHORITY_SCHEMA = (
    "bernini-action-edit-fresh-world8-level-a-terminal-authority-v2"
)
TERMINAL_SUCCESS_SCHEMA = (
    "bernini-action-edit-fresh-world8-level-a-terminal-success-v2"
)
VALIDATION_SCHEMA = (
    "bernini-action-edit-fresh-world8-level-a-receipt-validation-v2"
)
DRIVER_FILENAME = "action_edit_fresh_world8_level_a_driver_0817_v1.py"
CONSUMER_FILENAME = "action_edit_checkpoint_consumer_0817_v1.py"
PRODUCT_FILENAME = "infer_action_edit_product_abi_0817_v1.py"
CONSUMER_MODULE = "action_edit_checkpoint_consumer_0817_v1"
RELEASE_MANIFEST_FILENAME = "RELEASE_MANIFEST.json"
OUTPUT_RECEIPT_FILENAME = "bundle.consumer_receipt"
P2_STEP = 2
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
PARENT_JOB_ID = 140846
PINNED_NODE = "auh7-1b-gpu-279"
TAG = "fresh-world8-level-a-r2-p2-launchbound-v2"
EXPERIMENT_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_action_editing_0817"
)
PINNED_ROOTS: Mapping[str, str] = {
    "release": str(EXPERIMENT_ROOT / "releases" / TAG),
    "launch": str(EXPERIMENT_ROOT / "launchers" / TAG),
    "attempt": str(EXPERIMENT_ROOT / "attempts" / TAG),
    "run": str(EXPERIMENT_ROOT / "runs" / TAG),
}
ATTEMPT_JOB_NAMES: Mapping[str, str] = {
    "A": "bernini0817-level-a-launchbound-v2-A",
    "B": "bernini0817-level-a-launchbound-v2-B",
}
PINNED_CONSUMER_SHA256 = (
    "8bf0a9e48e0b2443a8e2f8e0744d08591226a167ac6ace45ee513481f5a97b3a"
)
PINNED_PRODUCT_SHA256 = (
    "b16d8aef25b35df13e8294ef387e4d334170af65c2f43ece9894142d7cadac14"
)
PINNED_R2_RELEASE_MANIFEST_SHA256 = (
    "671179995a64f20ee773273e84b5eb3f1f0bbd018fbfa3c0c6dc41d56c5555f5"
)
PINNED_R2_CAMPAIGN_RECEIPT_SHA256 = (
    "8014b7b71413318d80162fba12b73d83d6b9d9de5ea57ad295643a238b0f8c0e"
)
PINNED_P2_PARAMETER_SHA256 = (
    "5f9c31e84ab9ec4330b07d86cb1a2fc79c7aa365f4bf88a9cdffc0c244dcaa3e"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

# The eight frozen r2 training members have immutable hashes.  The final
# driver hash is supplied by the sealed launcher and cross-checked against the
# authenticated deployment manifest, avoiding a self-hash recursion.
PINNED_MEMBER_SHA256: Mapping[str, str] = {
    "action_plan_predictor_v1.py": (
        "464cd500f0ba1edb6cbe6d4f07287bfff346ae0ba7968c0d7c7f3cc7cb667308"
    ),
    CONSUMER_FILENAME: PINNED_CONSUMER_SHA256,
    "clean_source_visual_context_stage_b_contract_v1.py": (
        "f782876fd2b90b7b1d517fc49db03b800f1d9924156575275a472e3ea79ff571"
    ),
    "inference_sigma_strata.py": (
        "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3"
    ),
    PRODUCT_FILENAME: PINNED_PRODUCT_SHA256,
    "packed_preservation_lora_v2.py": (
        "61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6"
    ),
    "packed_preservation_release_v2.py": (
        "581e7314f9ca403bc8f0aa3d7e82adb57a9a202f8318f99256a002ecd255b99c"
    ),
    "source_self_runtime.py": (
        "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f"
    ),
    "train_action_edit_large_lora_0817_v1.py": (
        "edf3d1d2a77cb2f713968f537ce85a7d92f0b7347a0474419fe5562fbd319bd9"
    ),
    "train_lora.py": (
        "eae8eaac25197112637f466e611ba7eae574266d4cd1b83e625195fb22b0476e"
    ),
}
EXPECTED_RELEASE_PATHS = tuple(sorted((*PINNED_MEMBER_SHA256, DRIVER_FILENAME)))
FORBIDDEN_RELEASE_NAMES = {"__pycache__"}
FORBIDDEN_RELEASE_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib"}

# Exact root schema emitted by the frozen checkpoint consumer.  The driver
# adds only ``launch_binding`` and ``receipt_digest`` before publication.
RAW_RECEIPT_KEYS = frozenset(
    {
        "schema_version", "method", "authority", "complete", "promotable",
        "formal_training_started", "counts_as_d0", "scientific_claim_authorized",
        "action_quality_claim_authorized", "checkpoint_step",
        "checkpoint_parameter_sha256", "loaded_parameter_sha256",
        "campaign_receipt_sha256", "checkpoint_metadata_sha256",
        "release_manifest_sha256", "runner_source_sha256",
        "predictor_source_sha256", "conditioner_state_abi_sha256",
        "all_trainables_fp32_at_load", "optimizer_state_validated_but_not_loaded",
        "training_runtime_state_validated_but_not_restored", "runtime",
        "fresh_process_session_id", "single_checkpoint_load_per_process",
        "training_attached_reference_present", "training_attached_reference_absent",
        "training_attached_reference_binding",
        "training_attached_conditioner_cell_reference_present",
        "training_attached_conditioner_cell_reference_absent",
        "training_attached_full_renderer_reference_present",
        "training_attached_full_renderer_reference_absent",
        "training_attached_full_renderer_reference_binding",
        "training_to_fresh_forward_parity_verified",
        "conditioner_cell_training_to_fresh_forward_parity_verified",
        "full_bernini_renderer_training_to_fresh_forward_parity_verified",
        "fresh_a_b_parity_verified", "promotion_authorized",
        "offline_product_inference_completed", "full40_denoise_executed",
        "mp4_emitted", "consumer_source_sha256", "product_bridge_source_sha256",
        "official_bernini_commit", "veomni_commit", "base_checkpoint_tree_sha256",
        "base_checkpoint_content", "imported_training_release", "lora_installation",
        "model_mode", "training_gradient_checkpoint_hooks_installed",
        "offline_single_forward_exact30_hooks_installed",
        "fresh_loaded_fixed_forward_executed",
        "fresh_loaded_fixed_forward_fingerprint",
        "fixed_forward_process_rng_unchanged", "fixed_forward_trainable_bytes_unchanged",
        "world8_consensus", "world8_consumer_complete",
        "fresh_world8_process_forward_exact_consensus_verified",
        "fresh_world8_process_forward_scope", "full_bernini_renderer_forward_executed",
        "checkpoint_bytes_conditioner_exact30_fresh_consumer_go",
    }
)
PUBLISHED_RECEIPT_KEYS = RAW_RECEIPT_KEYS | {"launch_binding", "receipt_digest"}
WORLD8_CONSENSUS_KEYS = frozenset(
    {
        "world_size", "rank_order", "consumer_receipt_sha256",
        "all8_exact_consensus", "rank_local_fresh_process_sessions",
        "eight_distinct_fresh_process_sessions",
    }
)
POST_CONSENSUS_ONLY_KEYS = frozenset(
    {
        "world8_consensus", "world8_consumer_complete",
        "fresh_world8_process_forward_exact_consensus_verified",
        "fresh_world8_process_forward_scope", "full_bernini_renderer_forward_executed",
        "checkpoint_bytes_conditioner_exact30_fresh_consumer_go",
    }
)


class LevelADriverError(RuntimeError):
    """Raised before untrusted bytes or ambiguous outputs can be consumed."""


def fail(message: str) -> NoReturn:
    raise LevelADriverError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be one lowercase full SHA-256")
    return value


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _absolute_plain_directory(
    value: str | Path, *, label: str, required_mode: int | None = None
) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be one absolute non-symlink directory")
    try:
        before = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise LevelADriverError(f"{label} is unavailable: {error}") from error
    if (
        resolved != requested
        or not stat.S_ISDIR(before.st_mode)
        or requested.is_symlink()
        or (required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode)
    ):
        fail(f"{label} canonical directory identity differs")
    return requested


def _absolute_plain_file(
    value: str | Path,
    *,
    label: str,
    required_mode: int | None = None,
    required_nlink: int | None = None,
) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be one absolute non-symlink file")
    try:
        before = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise LevelADriverError(f"{label} is unavailable: {error}") from error
    if (
        resolved != requested
        or not stat.S_ISREG(before.st_mode)
        or requested.is_symlink()
        or (required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode)
        or (required_nlink is not None and before.st_nlink != required_nlink)
    ):
        fail(f"{label} canonical file identity differs")
    return requested


def stable_read(
    path: Path,
    *,
    maximum: int,
    label: str,
    required_mode: int | None = None,
    required_nlink: int | None = None,
) -> bytes:
    file_path = _absolute_plain_file(
        path,
        label=label,
        required_mode=required_mode,
        required_nlink=required_nlink,
    )
    before = file_path.lstat()
    if before.st_size <= 0 or before.st_size > maximum:
        fail(f"{label} size is outside the frozen bound")
    try:
        with file_path.open("rb") as handle:
            payload = handle.read(maximum + 1)
            opened = os.fstat(handle.fileno())
    except OSError as error:
        raise LevelADriverError(f"{label} could not be read: {error}") from error
    after = file_path.lstat()
    if (
        len(payload) != before.st_size
        or len(payload) > maximum
        or _identity(before) != _identity(opened)
        or _identity(before) != _identity(after)
    ):
        fail(f"{label} changed during authentication")
    return payload


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def strict_json_bytes(
    payload: bytes, *, label: str, allow_terminal_lf: bool = False
) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: fail(f"{label} contains {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LevelADriverError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        fail(f"{label} root must be one object")
    canonical = canonical_json_bytes(value)
    accepted = (canonical, canonical + b"\n") if allow_terminal_lf else (canonical,)
    if payload not in accepted:
        fail(f"{label} bytes are not the canonical JSON encoding")
    return value


@dataclass(frozen=True)
class DeploymentRelease:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    driver_sha256: str
    members: tuple[Mapping[str, Any], ...]
    source_bytes: Mapping[str, bytes]


@dataclass(frozen=True)
class LaunchAuthority:
    path: Path
    sha256: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class AttemptIntent:
    path: Path
    sha256: str
    mtime_ns: int
    raw: Mapping[str, Any]


def _exact_keys(value: Any, expected: frozenset[str] | set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        fail(f"{label} exact schema differs")
    return value


def validate_launch_authority_core(
    value: str | Path,
    *,
    expected_sha256: str,
    release: DeploymentRelease,
) -> LaunchAuthority:
    """Authenticate the pre-frozen, non-recursive launch authority core."""

    expected_sha = require_sha(expected_sha256, label="launch authority core SHA")
    path = _absolute_plain_file(
        value,
        label="launch authority core",
        required_mode=0o444,
        required_nlink=1,
    )
    if path != Path(PINNED_ROOTS["launch"]) / "LAUNCH_AUTHORITY_CORE.json":
        fail("launch authority core path differs")
    payload = stable_read(
        path,
        maximum=1024 * 1024,
        label="launch authority core",
        required_mode=0o444,
        required_nlink=1,
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        fail("launch authority core SHA differs")
    raw = strict_json_bytes(
        payload, label="launch authority core", allow_terminal_lf=True
    )
    _exact_keys(
        raw,
        {
            "schema_version", "method", "authority", "status",
            "parent_allocation", "checkpoint", "topology", "roots", "release",
            "launcher_hash_chain", "attempts", "claims",
        },
        label="launch authority core",
    )
    parent = _exact_keys(
        raw.get("parent_allocation"),
        {"job_id", "node", "control_authorized"},
        label="launch authority parent allocation",
    )
    checkpoint = _exact_keys(
        raw.get("checkpoint"), {"step", "parameter_sha256"},
        label="launch authority checkpoint",
    )
    topology = _exact_keys(
        raw.get("topology"),
        {"world_size", "dp_size", "sp_size", "max_restarts", "attempt_order"},
        label="launch authority topology",
    )
    roots = _exact_keys(
        raw.get("roots"), {"release", "launch", "attempt", "run"},
        label="launch authority roots",
    )
    release_row = _exact_keys(
        raw.get("release"),
        {
            "manifest_sha256", "driver_sha256", "consumer_sha256",
            "product_bridge_sha256",
        },
        label="launch authority release",
    )
    chain = _exact_keys(
        raw.get("launcher_hash_chain"),
        {"step_payload_sha256", "rank_exec_sha256"},
        label="launch authority launcher hash chain",
    )
    attempts = _exact_keys(
        raw.get("attempts"), {"A", "B"}, label="launch authority attempts"
    )
    claims = _exact_keys(
        raw.get("claims"),
        {
            "conditioner_predictor_plus_exact30_cell_only",
            "full_bernini_renderer_forward_executed",
            "offline_product_inference_completed", "full40_denoise_executed",
            "mp4_emitted", "formal_training_started", "counts_as_d0",
            "scientific_claim_authorized", "promotion_authorized",
            "automatic_relaunch_authorized",
        },
        label="launch authority claims",
    )
    if (
        raw.get("schema_version") != LAUNCH_AUTHORITY_SCHEMA
        or raw.get("method") != METHOD
        or raw.get("authority") != AUTHORITY
        or raw.get("status") != "FROZEN_ONE_SHOT_LAUNCH_AUTHORITY"
        or dict(parent) != {
            "job_id": PARENT_JOB_ID,
            "node": PINNED_NODE,
            "control_authorized": False,
        }
        or dict(checkpoint) != {
            "step": P2_STEP,
            "parameter_sha256": PINNED_P2_PARAMETER_SHA256,
        }
        or dict(topology) != {
            "world_size": WORLD_SIZE,
            "dp_size": DP_SIZE,
            "sp_size": SP_SIZE,
            "max_restarts": 0,
            "attempt_order": ["A", "B"],
        }
        or dict(roots) != dict(PINNED_ROOTS)
        or release_row.get("manifest_sha256") != release.manifest_sha256
        or release_row.get("driver_sha256") != release.driver_sha256
        or release_row.get("consumer_sha256") != PINNED_CONSUMER_SHA256
        or release_row.get("product_bridge_sha256") != PINNED_PRODUCT_SHA256
        or any(
            require_sha(chain.get(key), label=f"launch authority {key}") == ""
            for key in ("step_payload_sha256", "rank_exec_sha256")
        )
        or dict(claims) != {
            "conditioner_predictor_plus_exact30_cell_only": True,
            "full_bernini_renderer_forward_executed": False,
            "offline_product_inference_completed": False,
            "full40_denoise_executed": False,
            "mp4_emitted": False,
            "formal_training_started": False,
            "counts_as_d0": False,
            "scientific_claim_authorized": False,
            "promotion_authorized": False,
            "automatic_relaunch_authorized": False,
        }
    ):
        fail("launch authority core values differ")
    for label in ("A", "B"):
        row = _exact_keys(
            attempts.get(label),
            {"intent_sha256", "job_name", "attempt_root", "output_root"},
            label=f"launch authority attempt {label}",
        )
        if (
            require_sha(row.get("intent_sha256"), label=f"attempt {label} intent SHA")
            != row.get("intent_sha256")
            or row.get("job_name") != ATTEMPT_JOB_NAMES[label]
            or row.get("attempt_root") != f'{PINNED_ROOTS["attempt"]}/{label}'
            or row.get("output_root") != f'{PINNED_ROOTS["run"]}/{label}'
        ):
            fail(f"launch authority attempt {label} differs")
    return LaunchAuthority(path=path, sha256=expected_sha, raw=dict(raw))


def validate_attempt_intent(
    value: str | Path,
    *,
    expected_sha256: str,
    attempt_label: str,
    authority: LaunchAuthority,
) -> AttemptIntent:
    if attempt_label not in ("A", "B"):
        fail("attempt intent label differs")
    expected_sha = require_sha(expected_sha256, label="attempt intent SHA")
    attempt_row = authority.raw["attempts"][attempt_label]
    if expected_sha != attempt_row["intent_sha256"]:
        fail("attempt intent SHA is not bound by launch authority")
    path = _absolute_plain_file(
        value,
        label=f"attempt {attempt_label} intent",
        required_mode=0o444,
        required_nlink=1,
    )
    expected_path = Path(attempt_row["attempt_root"]) / "STARTED" / "intent.json"
    if path != expected_path:
        fail("attempt intent canonical path differs")
    payload = stable_read(
        path,
        maximum=1024 * 1024,
        label=f"attempt {attempt_label} intent",
        required_mode=0o444,
        required_nlink=1,
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        fail("attempt intent bytes differ")
    raw = strict_json_bytes(payload, label=f"attempt {attempt_label} intent")
    _exact_keys(
        raw,
        {
            "schema_version", "method", "authority", "attempt", "parent_job_id",
            "node", "job_name", "release_root", "launch_root", "attempt_root",
            "output_root", "checkpoint_step", "world_size", "dp_size", "sp_size",
            "release_manifest_sha256", "driver_sha256", "consumer_sha256",
            "product_bridge_sha256", "step_payload_sha256", "rank_exec_sha256",
            "automatic_relaunch_authorized", "parent_control_authorized",
        },
        label=f"attempt {attempt_label} intent",
    )
    expected = {
        "schema_version": ATTEMPT_INTENT_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "attempt": attempt_label,
        "parent_job_id": PARENT_JOB_ID,
        "node": PINNED_NODE,
        "job_name": attempt_row["job_name"],
        "release_root": PINNED_ROOTS["release"],
        "launch_root": PINNED_ROOTS["launch"],
        "attempt_root": attempt_row["attempt_root"],
        "output_root": attempt_row["output_root"],
        "checkpoint_step": P2_STEP,
        "world_size": WORLD_SIZE,
        "dp_size": DP_SIZE,
        "sp_size": SP_SIZE,
        "release_manifest_sha256": authority.raw["release"]["manifest_sha256"],
        "driver_sha256": authority.raw["release"]["driver_sha256"],
        "consumer_sha256": PINNED_CONSUMER_SHA256,
        "product_bridge_sha256": PINNED_PRODUCT_SHA256,
        "step_payload_sha256": authority.raw["launcher_hash_chain"]["step_payload_sha256"],
        "rank_exec_sha256": authority.raw["launcher_hash_chain"]["rank_exec_sha256"],
        "automatic_relaunch_authorized": False,
        "parent_control_authorized": False,
    }
    if dict(raw) != expected:
        fail(f"attempt {attempt_label} intent values differ")
    info = path.lstat()
    return AttemptIntent(
        path=path, sha256=expected_sha, mtime_ns=info.st_mtime_ns, raw=dict(raw)
    )


def validate_deployment_release(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_driver_sha256: str,
) -> DeploymentRelease:
    """Authenticate an exact root-manifest + eleven-source-file closure."""

    manifest_sha = require_sha(
        expected_manifest_sha256, label="deployment release manifest SHA"
    )
    driver_sha = require_sha(expected_driver_sha256, label="driver source SHA")
    path = _absolute_plain_file(
        manifest_path,
        label="deployment release manifest",
        required_mode=0o444,
        required_nlink=1,
    )
    root = _absolute_plain_directory(
        path.parent, label="deployment release root", required_mode=0o555
    )
    if path != root / RELEASE_MANIFEST_FILENAME:
        fail("deployment release manifest basename differs")
    manifest_payload = stable_read(
        path,
        maximum=1024 * 1024,
        label="deployment release manifest",
        required_mode=0o444,
        required_nlink=1,
    )
    if hashlib.sha256(manifest_payload).hexdigest() != manifest_sha:
        fail("deployment release manifest SHA differs")
    raw = strict_json_bytes(
        manifest_payload,
        label="deployment release manifest",
        allow_terminal_lf=True,
    )
    if (
        set(raw) != {"schema_version", "member_root", "files"}
        or raw.get("schema_version") != RELEASE_SCHEMA
        or raw.get("member_root") != RELEASE_MEMBER_ROOT
        or not isinstance(raw.get("files"), list)
    ):
        fail("deployment release manifest envelope differs")

    try:
        root_entries = tuple(sorted(item.name for item in os.scandir(root)))
    except OSError as error:
        raise LevelADriverError(f"deployment release root scan failed: {error}") from error
    expected_entries = tuple(sorted((*EXPECTED_RELEASE_PATHS, RELEASE_MANIFEST_FILENAME)))
    if root_entries != expected_entries:
        fail("deployment release exact root closure differs")
    if any(
        entry in FORBIDDEN_RELEASE_NAMES
        or PurePosixPath(entry).suffix in FORBIDDEN_RELEASE_SUFFIXES
        for entry in root_entries
    ):
        fail("deployment release contains a forbidden cache/binary member")

    rows = raw["files"]
    if len(rows) != len(EXPECTED_RELEASE_PATHS):
        fail("deployment release member count differs")
    normalized: list[Mapping[str, Any]] = []
    sources: dict[str, bytes] = {}
    for expected_path, row in zip(EXPECTED_RELEASE_PATHS, rows):
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "mode",
            "size",
            "sha256",
        }:
            fail("deployment release member schema differs")
        relative = row.get("path")
        pure = PurePosixPath(str(relative))
        if (
            relative != expected_path
            or pure.is_absolute()
            or pure.as_posix() != relative
            or len(pure.parts) != 1
            or any(part in ("", ".", "..") for part in pure.parts)
            or row.get("mode") != 0o444
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or row["size"] > 2 * 1024 * 1024
        ):
            fail("deployment release exact sorted member closure differs")
        wanted_sha = require_sha(
            row.get("sha256"), label=f"deployment member {relative} SHA"
        )
        fixed_sha = driver_sha if relative == DRIVER_FILENAME else PINNED_MEMBER_SHA256.get(relative)
        if fixed_sha is None or wanted_sha != fixed_sha:
            fail(f"deployment member trust-root SHA differs: {relative}")
        member = root / relative
        payload = stable_read(
            member,
            maximum=2 * 1024 * 1024,
            label=f"deployment member {relative}",
            required_mode=0o444,
            required_nlink=1,
        )
        if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != wanted_sha:
            fail(f"deployment member bytes differ: {relative}")
        normalized.append(dict(row))
        sources[str(relative)] = payload
    return DeploymentRelease(
        root=root,
        manifest_path=path,
        manifest_sha256=manifest_sha,
        driver_sha256=driver_sha,
        members=tuple(normalized),
        source_bytes=dict(sources),
    )


def validate_executed_driver(release: DeploymentRelease) -> None:
    executed = Path(__file__)
    expected = release.root / DRIVER_FILENAME
    if (
        not executed.is_absolute()
        or executed.is_symlink()
        or executed.resolve(strict=True) != executed
        or executed != expected
        or stat.S_IMODE(executed.lstat().st_mode) != 0o444
        or executed.lstat().st_nlink != 1
        or hashlib.sha256(release.source_bytes[DRIVER_FILENAME]).hexdigest()
        != release.driver_sha256
    ):
        fail("executed driver is not the authenticated deployment member")


def load_consumer_from_authenticated_bytes(release: DeploymentRelease) -> Any:
    """Compile exactly the preflight consumer bytes without import machinery."""

    if CONSUMER_MODULE in sys.modules:
        fail("checkpoint consumer module was imported before release authentication")
    path = release.root / CONSUMER_FILENAME
    payload = release.source_bytes.get(CONSUMER_FILENAME)
    if (
        not isinstance(payload, bytes)
        or hashlib.sha256(payload).hexdigest() != PINNED_CONSUMER_SHA256
    ):
        fail("authenticated checkpoint consumer bytes differ")
    try:
        code = compile(
            payload,
            str(path),
            "exec",
            flags=0,
            dont_inherit=True,
            optimize=0,
        )
    except (SyntaxError, ValueError, TypeError) as error:
        raise LevelADriverError("authenticated checkpoint consumer did not compile") from error
    module = types.ModuleType(CONSUMER_MODULE)
    module.__file__ = str(path)
    module.__cached__ = None
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[CONSUMER_MODULE] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        if sys.modules.get(CONSUMER_MODULE) is module:
            del sys.modules[CONSUMER_MODULE]
        raise
    if (
        sys.modules.get(CONSUMER_MODULE) is not module
        or module.__file__ != str(path)
        or getattr(module, "PINNED_R2_RELEASE_MANIFEST_SHA256", None)
        != PINNED_R2_RELEASE_MANIFEST_SHA256
        or getattr(module, "PINNED_R2_CAMPAIGN_RECEIPT_SHA256", None)
        != PINNED_R2_CAMPAIGN_RECEIPT_SHA256
        or getattr(module, "PINNED_R2_P_STATE_SHA256", {}).get(P2_STEP)
        != PINNED_P2_PARAMETER_SHA256
    ):
        fail("authenticated checkpoint consumer authority differs")
    return module


def _validate_raw_level_a_receipt(receipt: Mapping[str, Any]) -> str:
    _exact_keys(receipt, RAW_RECEIPT_KEYS, label="raw consumer receipt")
    consensus = receipt.get("world8_consensus")
    _exact_keys(consensus, WORLD8_CONSENSUS_KEYS, label="WORLD8 consensus")
    sessions = consensus.get("rank_local_fresh_process_sessions")
    if (
        receipt.get("schema_version")
        != "bernini-action-edit-fresh-consumer-receipt-v1"
        or receipt.get("method") != "bernini-action-edit-checkpoint-consumer-0817-v1"
        or receipt.get("authority") != AUTHORITY
        or receipt.get("complete") is not True
        or receipt.get("formal_training_started") is not False
        or receipt.get("counts_as_d0") is not False
        or receipt.get("scientific_claim_authorized") is not False
        or receipt.get("action_quality_claim_authorized") is not False
        or receipt.get("checkpoint_step") != P2_STEP
        or receipt.get("checkpoint_parameter_sha256") != PINNED_P2_PARAMETER_SHA256
        or receipt.get("loaded_parameter_sha256") != PINNED_P2_PARAMETER_SHA256
        or receipt.get("campaign_receipt_sha256") != PINNED_R2_CAMPAIGN_RECEIPT_SHA256
        or receipt.get("release_manifest_sha256") != PINNED_R2_RELEASE_MANIFEST_SHA256
        or receipt.get("consumer_source_sha256") != PINNED_CONSUMER_SHA256
        or receipt.get("product_bridge_source_sha256") != PINNED_PRODUCT_SHA256
        or receipt.get("world8_consumer_complete") is not True
        or receipt.get("fresh_loaded_fixed_forward_executed") is not True
        or receipt.get("fresh_world8_process_forward_exact_consensus_verified")
        is not True
        or receipt.get("fresh_world8_process_forward_scope")
        != "conditioner_predictor_plus_exact30_cell_only_not_bernini_renderer"
        or receipt.get("full_bernini_renderer_forward_executed") is not False
        or receipt.get("checkpoint_bytes_conditioner_exact30_fresh_consumer_go")
        is not True
        or receipt.get("offline_product_inference_completed") is not False
        or receipt.get("full40_denoise_executed") is not False
        or receipt.get("mp4_emitted") is not False
        or receipt.get("promotion_authorized") is not False
        or receipt.get("promotable") is not False
        or receipt.get("training_attached_reference_absent") is not True
        or receipt.get("training_attached_full_renderer_reference_absent") is not True
        or receipt.get("training_to_fresh_forward_parity_verified") is not False
        or receipt.get("conditioner_cell_training_to_fresh_forward_parity_verified")
        is not False
        or receipt.get("full_bernini_renderer_training_to_fresh_forward_parity_verified")
        is not False
        or receipt.get("fresh_a_b_parity_verified") is not False
        or receipt.get("fixed_forward_process_rng_unchanged") is not True
        or receipt.get("fixed_forward_trainable_bytes_unchanged") is not True
        or receipt.get("training_gradient_checkpoint_hooks_installed") is not False
        or receipt.get("offline_single_forward_exact30_hooks_installed") is not True
        or consensus.get("world_size") != WORLD_SIZE
        or consensus.get("rank_order") != list(range(WORLD_SIZE))
        or consensus.get("all8_exact_consensus") is not True
        or consensus.get("eight_distinct_fresh_process_sessions") is not True
        or not isinstance(sessions, list)
        or len(sessions) != WORLD_SIZE
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in sessions
        )
        or len(set(sessions)) != WORLD_SIZE
        or receipt.get("fresh_process_session_id") not in sessions
    ):
        fail("consumer receipt exceeds or differs from frozen Level-A authority")
    consensus_sha = require_sha(
        consensus.get("consumer_receipt_sha256"),
        label="WORLD8 consumer consensus SHA",
    )
    common = dict(receipt)
    for key in POST_CONSENSUS_ONLY_KEYS:
        common.pop(key)
    common.pop("fresh_process_session_id")
    if object_sha256(common) != consensus_sha:
        fail("WORLD8 consumer consensus digest does not recompute")
    return consensus_sha


def _validate_launch_binding(
    binding: Any,
    *,
    authority: LaunchAuthority | None = None,
    intent: AttemptIntent | None = None,
) -> Mapping[str, Any]:
    row = _exact_keys(
        binding,
        {
            "schema_version", "attempt", "attempt_intent_sha256",
            "attempt_claim_mtime_ns", "launch_authority_core_sha256",
            "parent_job_id", "slurm_numeric_step", "node", "job_name",
            "release_manifest_sha256", "driver_source_sha256",
            "launcher_hash_chain",
        },
        label="receipt launch binding",
    )
    chain = _exact_keys(
        row.get("launcher_hash_chain"),
        {"step_payload_sha256", "rank_exec_sha256"},
        label="receipt launcher hash chain",
    )
    label = row.get("attempt")
    numeric_step = row.get("slurm_numeric_step")
    if (
        row.get("schema_version") != LAUNCH_BINDING_SCHEMA
        or label not in ("A", "B")
        or type(row.get("attempt_claim_mtime_ns")) is not int
        or row["attempt_claim_mtime_ns"] <= 0
        or row.get("parent_job_id") != PARENT_JOB_ID
        or not isinstance(numeric_step, str)
        or re.fullmatch(rf"{PARENT_JOB_ID}\.[0-9]+", numeric_step) is None
        or row.get("node") != PINNED_NODE
        or row.get("job_name") != ATTEMPT_JOB_NAMES.get(str(label))
        or row.get("release_manifest_sha256") != (
            authority.raw["release"]["manifest_sha256"]
            if authority is not None else row.get("release_manifest_sha256")
        )
        or any(
            require_sha(row.get(key), label=f"launch binding {key}") == ""
            for key in (
                "attempt_intent_sha256", "launch_authority_core_sha256",
                "release_manifest_sha256", "driver_source_sha256",
            )
        )
        or any(
            require_sha(chain.get(key), label=f"launch binding {key}") == ""
            for key in ("step_payload_sha256", "rank_exec_sha256")
        )
    ):
        fail("receipt launch binding values differ")
    if authority is not None:
        expected_attempt = authority.raw["attempts"][str(label)]
        if (
            row.get("launch_authority_core_sha256") != authority.sha256
            or row.get("attempt_intent_sha256") != expected_attempt["intent_sha256"]
            or row.get("release_manifest_sha256")
            != authority.raw["release"]["manifest_sha256"]
            or row.get("driver_source_sha256")
            != authority.raw["release"]["driver_sha256"]
            or dict(chain) != dict(authority.raw["launcher_hash_chain"])
        ):
            fail("receipt launch binding is outside launch authority core")
    if intent is not None:
        if (
            label != intent.raw["attempt"]
            or row.get("attempt_intent_sha256") != intent.sha256
            or row.get("attempt_claim_mtime_ns") != intent.mtime_ns
            or row.get("job_name") != intent.raw["job_name"]
        ):
            fail("receipt launch binding is outside attempt intent")
    return row


def _validate_level_a_receipt(
    receipt: Mapping[str, Any],
    *,
    authority: LaunchAuthority | None = None,
    intent: AttemptIntent | None = None,
) -> tuple[str, str]:
    _exact_keys(receipt, PUBLISHED_RECEIPT_KEYS, label="published consumer receipt")
    unsigned = dict(receipt)
    receipt_digest = require_sha(
        unsigned.pop("receipt_digest", None), label="published receipt digest"
    )
    if object_sha256(unsigned) != receipt_digest:
        fail("published consumer receipt self-digest differs")
    binding = unsigned.pop("launch_binding", None)
    _validate_launch_binding(binding, authority=authority, intent=intent)
    consensus_sha = _validate_raw_level_a_receipt(unsigned)
    return receipt_digest, consensus_sha


def build_launch_binding(
    *,
    attempt_label: str,
    authority: LaunchAuthority,
    intent: AttemptIntent,
    release: DeploymentRelease,
) -> Mapping[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    step_id = os.environ.get("SLURM_STEP_ID")
    node = socket.gethostname().split(".", 1)[0]
    if (
        job_id != str(PARENT_JOB_ID)
        or not isinstance(step_id, str)
        or re.fullmatch(r"[0-9]+", step_id) is None
        or node != PINNED_NODE
    ):
        fail("live Slurm launch identity differs")
    binding = {
        "schema_version": LAUNCH_BINDING_SCHEMA,
        "attempt": attempt_label,
        "attempt_intent_sha256": intent.sha256,
        "attempt_claim_mtime_ns": intent.mtime_ns,
        "launch_authority_core_sha256": authority.sha256,
        "parent_job_id": PARENT_JOB_ID,
        "slurm_numeric_step": f"{PARENT_JOB_ID}.{step_id}",
        "node": node,
        "job_name": intent.raw["job_name"],
        "release_manifest_sha256": release.manifest_sha256,
        "driver_source_sha256": release.driver_sha256,
        "launcher_hash_chain": dict(authority.raw["launcher_hash_chain"]),
    }
    _validate_launch_binding(binding, authority=authority, intent=intent)
    return binding


def _output_directory(value: str | Path) -> Path:
    root = _absolute_plain_directory(
        value, label="attempt output root", required_mode=0o700
    )
    with os.scandir(root) as entries:
        if next(entries, None) is not None:
            fail("attempt output root is not fresh and empty")
    return root


def _atomic_write_canonical(
    destination: Path,
    value: Mapping[str, Any],
    *,
    expected_parent_mode: int,
) -> str:
    parent = _absolute_plain_directory(
        destination.parent,
        label="receipt output parent",
        required_mode=expected_parent_mode,
    )
    if destination.parent != parent or destination.name in ("", ".", ".."):
        fail("receipt output destination differs")
    payload = canonical_json_bytes(value)
    temporary = f".{destination.name}.{os.getpid()}.tmp"
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary_created = False
    try:
        if destination.exists() or destination.is_symlink():
            fail("receipt output already exists")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        file_fd = os.open(temporary, flags, 0o400, dir_fd=directory_fd)
        temporary_created = True
        try:
            offset = 0
            while offset < len(payload):
                written_now = os.write(file_fd, payload[offset:])
                if written_now <= 0:
                    fail("provisional receipt write made no progress")
                offset += written_now
            os.fsync(file_fd)
            os.fchmod(file_fd, 0o444)
            written = os.fstat(file_fd)
            if (
                not stat.S_ISREG(written.st_mode)
                or stat.S_IMODE(written.st_mode) != 0o444
                or written.st_nlink != 1
                or written.st_size != len(payload)
            ):
                fail("provisional receipt topology differs")
        finally:
            os.close(file_fd)
        os.replace(
            temporary,
            destination.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_created = False
        os.fsync(directory_fd)
        final = os.stat(destination.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or stat.S_IMODE(final.st_mode) != 0o444
            or final.st_nlink != 1
            or final.st_size != len(payload)
        ):
            fail("terminal receipt topology differs")
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)
    observed = stable_read(
        destination,
        maximum=64 * 1024 * 1024,
        label="terminal receipt",
        required_mode=0o444,
        required_nlink=1,
    )
    if observed != payload:
        fail("terminal receipt bytes differ")
    return hashlib.sha256(payload).hexdigest()


def _create_only_publish_canonical(
    destination: Path,
    value: Mapping[str, Any],
    *,
    expected_parent_mode: int,
) -> str:
    """Publish one canonical file without an overwrite-capable final rename."""

    parent = _absolute_plain_directory(
        destination.parent,
        label="create-only output parent",
        required_mode=expected_parent_mode,
    )
    if destination.parent != parent or destination.name in ("", ".", ".."):
        fail("create-only output destination differs")
    payload = canonical_json_bytes(value)
    temporary = f".{destination.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary_created = False
    published = False
    try:
        if destination.exists() or destination.is_symlink():
            fail("create-only output already exists")
        file_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=directory_fd,
        )
        temporary_created = True
        try:
            offset = 0
            while offset < len(payload):
                written_now = os.write(file_fd, payload[offset:])
                if written_now <= 0:
                    fail("create-only provisional write made no progress")
                offset += written_now
            os.fsync(file_fd)
            os.fchmod(file_fd, 0o444)
            written = os.fstat(file_fd)
            if (
                not stat.S_ISREG(written.st_mode)
                or stat.S_IMODE(written.st_mode) != 0o444
                or written.st_nlink != 1
                or written.st_size != len(payload)
            ):
                fail("create-only provisional topology differs")
        finally:
            os.close(file_fd)
        # link(2) is an atomic no-replace publication: EEXIST cannot overwrite
        # a competing destination.  Removing the private name restores nlink=1.
        os.link(
            temporary,
            destination.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary, dir_fd=directory_fd)
        temporary_created = False
        os.fsync(directory_fd)
        final = os.stat(destination.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or stat.S_IMODE(final.st_mode) != 0o444
            or final.st_nlink != 1
            or final.st_size != len(payload)
        ):
            fail("create-only terminal topology differs")
    except FileExistsError as error:
        raise LevelADriverError("create-only output already exists") from error
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)
    if not published:
        fail("create-only output was not published")
    observed = stable_read(
        destination,
        maximum=64 * 1024 * 1024,
        label="create-only terminal output",
        required_mode=0o444,
        required_nlink=1,
    )
    if observed != payload:
        fail("create-only terminal bytes differ")
    return hashlib.sha256(payload).hexdigest()


def run_world8(args: argparse.Namespace) -> int:
    if args.attempt_label not in ("A", "B"):
        fail("attempt label must be exactly A or B")
    if require_sha(args.expected_consumer_source_sha256, label="consumer source SHA") != PINNED_CONSUMER_SHA256:
        fail("launcher consumer source SHA pin differs")
    if require_sha(args.expected_product_source_sha256, label="product source SHA") != PINNED_PRODUCT_SHA256:
        fail("launcher product source SHA pin differs")
    release = validate_deployment_release(
        args.release_manifest,
        expected_manifest_sha256=args.expected_release_manifest_sha256,
        expected_driver_sha256=args.expected_driver_source_sha256,
    )
    validate_executed_driver(release)
    authority = validate_launch_authority_core(
        args.launch_authority_core,
        expected_sha256=args.expected_launch_authority_core_sha256,
        release=release,
    )
    intent = validate_attempt_intent(
        args.attempt_intent,
        expected_sha256=args.expected_attempt_intent_sha256,
        attempt_label=args.attempt_label,
        authority=authority,
    )
    output_root = _output_directory(args.output_root)
    if output_root != Path(intent.raw["output_root"]):
        fail("attempt output root is outside frozen intent")
    consumer = load_consumer_from_authenticated_bytes(release)
    bundle = consumer.consume_frozen_r2_world8_checkpoint(
        release_manifest_path=args.r2_release_manifest,
        campaign_receipt_path=args.campaign_receipt,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_step=P2_STEP,
        bernini_root=args.bernini_root,
        veomni_root=args.veomni_root,
        base_checkpoint=args.base_checkpoint,
        checkpoint_content_manifest=args.checkpoint_content_manifest,
        expected_consumer_source_sha256=PINNED_CONSUMER_SHA256,
        expected_product_source_sha256=PINNED_PRODUCT_SHA256,
    )
    receipt = bundle.consumer_receipt
    if not isinstance(receipt, Mapping):
        fail("consumer did not return one receipt object")
    _validate_raw_level_a_receipt(receipt)
    launch_binding = build_launch_binding(
        attempt_label=args.attempt_label,
        authority=authority,
        intent=intent,
        release=release,
    )
    unsigned_receipt = {**dict(receipt), "launch_binding": dict(launch_binding)}
    published_receipt = {
        **unsigned_receipt,
        "receipt_digest": object_sha256(unsigned_receipt),
    }
    _validate_level_a_receipt(
        published_receipt, authority=authority, intent=intent
    )

    import torch

    dist = torch.distributed
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() != WORLD_SIZE
    ):
        fail("terminal receipt publication requires initialized WORLD8")
    rank = int(dist.get_rank())
    dist.barrier()
    receipt_path = output_root / OUTPUT_RECEIPT_FILENAME
    local_sha = ""
    if rank == 0:
        local_sha = _atomic_write_canonical(
            receipt_path, published_receipt, expected_parent_mode=0o700
        )
    dist.barrier()
    observed = stable_read(
        receipt_path,
        maximum=64 * 1024 * 1024,
        label="published consumer receipt",
        required_mode=0o444,
        required_nlink=1,
    )
    observed_value = strict_json_bytes(observed, label="published consumer receipt")
    _validate_level_a_receipt(observed_value, authority=authority, intent=intent)
    observed_sha = hashlib.sha256(observed).hexdigest()
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, {"rank": rank, "receipt_file_sha256": observed_sha})
    if (
        [row.get("rank") for row in rows if isinstance(row, Mapping)]
        != list(range(WORLD_SIZE))
        or len({row.get("receipt_file_sha256") for row in rows if isinstance(row, Mapping)})
        != 1
        or (rank == 0 and local_sha != observed_sha)
    ):
        fail("WORLD8 terminal receipt-file consensus differs")
    dist.barrier()
    print(
        f"PASS Level-A attempt={args.attempt_label} rank={rank} "
        f"receipt_sha256={observed_sha} renderer=false full40=false mp4=false",
        flush=True,
    )
    return 0


def _read_canonical_receipt(
    value: str | Path,
    *,
    label: str,
    authority: LaunchAuthority | None = None,
    intent: AttemptIntent | None = None,
) -> tuple[Path, Mapping[str, Any], str, tuple[int, int]]:
    path = _absolute_plain_file(
        value, label=label, required_mode=0o444, required_nlink=1
    )
    if intent is not None and path != Path(intent.raw["output_root"]) / OUTPUT_RECEIPT_FILENAME:
        fail(f"{label} canonical path differs")
    payload = stable_read(
        path,
        maximum=64 * 1024 * 1024,
        label=label,
        required_mode=0o444,
        required_nlink=1,
    )
    receipt = strict_json_bytes(payload, label=label)
    _validate_level_a_receipt(receipt, authority=authority, intent=intent)
    info = path.lstat()
    return path, receipt, hashlib.sha256(payload).hexdigest(), (info.st_dev, info.st_ino)


def _validation_record(
    *,
    receipt: Mapping[str, Any],
    receipt_file_sha256: str,
    authority: LaunchAuthority,
    intent: AttemptIntent,
) -> Mapping[str, Any]:
    receipt_digest, consensus_sha = _validate_level_a_receipt(
        receipt, authority=authority, intent=intent
    )
    binding = receipt["launch_binding"]
    unsigned = {
        "schema_version": VALIDATION_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "attempt": intent.raw["attempt"],
        "attempt_intent_sha256": intent.sha256,
        "launch_authority_core_sha256": authority.sha256,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_digest": receipt_digest,
        "world8_consensus_sha256": consensus_sha,
        "slurm_numeric_step": binding["slurm_numeric_step"],
        "node": binding["node"],
        "job_name": binding["job_name"],
        "receipt_validated": True,
        "full_bernini_renderer_forward_executed": False,
        "offline_product_inference_completed": False,
        "full40_denoise_executed": False,
        "mp4_emitted": False,
        "promotion_authorized": False,
    }
    return {**unsigned, "validation_digest": object_sha256(unsigned)}


def _read_validation_record(
    value: str | Path, *, label: str
) -> tuple[Mapping[str, Any], str]:
    path = _absolute_plain_file(
        value, label=label, required_mode=0o444, required_nlink=1
    )
    payload = stable_read(
        path, maximum=1024 * 1024, label=label, required_mode=0o444,
        required_nlink=1,
    )
    row = strict_json_bytes(payload, label=label)
    _exact_keys(
        row,
        {
            "schema_version", "method", "authority", "attempt",
            "attempt_intent_sha256", "launch_authority_core_sha256",
            "receipt_file_sha256", "receipt_digest", "world8_consensus_sha256",
            "slurm_numeric_step", "node", "job_name", "receipt_validated",
            "full_bernini_renderer_forward_executed",
            "offline_product_inference_completed", "full40_denoise_executed",
            "mp4_emitted", "promotion_authorized", "validation_digest",
        },
        label=label,
    )
    unsigned = dict(row)
    digest = require_sha(unsigned.pop("validation_digest", None), label=f"{label} digest")
    if (
        object_sha256(unsigned) != digest
        or row.get("schema_version") != VALIDATION_SCHEMA
        or row.get("method") != METHOD
        or row.get("authority") != AUTHORITY
        or row.get("attempt") not in ("A", "B")
        or row.get("receipt_validated") is not True
        or any(
            row.get(key) is not False
            for key in (
                "full_bernini_renderer_forward_executed",
                "offline_product_inference_completed", "full40_denoise_executed",
                "mp4_emitted", "promotion_authorized",
            )
        )
    ):
        fail(f"{label} values differ")
    for key in (
        "attempt_intent_sha256", "launch_authority_core_sha256",
        "receipt_file_sha256", "receipt_digest", "world8_consensus_sha256",
    ):
        require_sha(row.get(key), label=f"{label} {key}")
    return row, hashlib.sha256(payload).hexdigest()


def validate_receipt_command(args: argparse.Namespace) -> int:
    release = validate_deployment_release(
        args.release_manifest,
        expected_manifest_sha256=args.expected_release_manifest_sha256,
        expected_driver_sha256=args.expected_driver_source_sha256,
    )
    validate_executed_driver(release)
    authority = validate_launch_authority_core(
        args.launch_authority_core,
        expected_sha256=args.expected_launch_authority_core_sha256,
        release=release,
    )
    intent = validate_attempt_intent(
        args.attempt_intent,
        expected_sha256=args.expected_attempt_intent_sha256,
        attempt_label=args.attempt_label,
        authority=authority,
    )
    _, receipt, receipt_sha, _ = _read_canonical_receipt(
        args.receipt,
        label=f"attempt {args.attempt_label} published receipt",
        authority=authority,
        intent=intent,
    )
    record = _validation_record(
        receipt=receipt,
        receipt_file_sha256=receipt_sha,
        authority=authority,
        intent=intent,
    )
    if args.output_validation:
        wanted_names = {
            "receipt-validation-1.json", "receipt-validation-2.json"
        }
        output_path = Path(args.output_validation).expanduser()
        if (
            output_path.parent != Path(intent.raw["attempt_root"]) / "STARTED"
            or output_path.name not in wanted_names
        ):
            fail("receipt validation output canonical path differs")
        _atomic_write_canonical(
            output_path, record, expected_parent_mode=0o700
        )
    print(canonical_json_bytes(record).decode("utf-8"), flush=True)
    return 0


def publish_validation_pair(args: argparse.Namespace) -> int:
    """Create the fixed validation pair only after two identical dry probes."""

    release = validate_deployment_release(
        args.release_manifest,
        expected_manifest_sha256=args.expected_release_manifest_sha256,
        expected_driver_sha256=args.expected_driver_source_sha256,
    )
    validate_executed_driver(release)
    authority = validate_launch_authority_core(
        args.launch_authority_core,
        expected_sha256=args.expected_launch_authority_core_sha256,
        release=release,
    )
    intent = validate_attempt_intent(
        args.attempt_intent,
        expected_sha256=args.expected_attempt_intent_sha256,
        attempt_label=args.attempt_label,
        authority=authority,
    )
    _, receipt, receipt_sha, _ = _read_canonical_receipt(
        args.receipt,
        label=f"attempt {args.attempt_label} pair-publication receipt",
        authority=authority,
        intent=intent,
    )
    expected = _validation_record(
        receipt=receipt,
        receipt_file_sha256=receipt_sha,
        authority=authority,
        intent=intent,
    )
    probe_payloads = []
    probe_values = []
    for label, text in (
        ("first dry validation probe", args.validation_json_a),
        ("second dry validation probe", args.validation_json_b),
    ):
        try:
            payload = text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise LevelADriverError(f"{label} is not UTF-8") from error
        value = strict_json_bytes(payload, label=label)
        probe_payloads.append(payload)
        probe_values.append(value)
    if (
        probe_payloads[0] != probe_payloads[1]
        or probe_values[0] != probe_values[1]
        or probe_values[0] != expected
        or hashlib.sha256(probe_payloads[0]).hexdigest()
        != hashlib.sha256(probe_payloads[1]).hexdigest()
    ):
        fail("two dry full validation probes do not bind identical receipt bytes")
    first = Path(args.output_validation_a).expanduser()
    second = Path(args.output_validation_b).expanduser()
    expected_parent = Path(intent.raw["attempt_root"]) / "STARTED"
    if (
        first != expected_parent / "receipt-validation-1.json"
        or second != expected_parent / "receipt-validation-2.json"
    ):
        fail("fixed validation pair output path differs")
    first_sha = _create_only_publish_canonical(
        first, expected, expected_parent_mode=0o700
    )
    second_sha = _create_only_publish_canonical(
        second, expected, expected_parent_mode=0o700
    )
    if first_sha != second_sha:
        fail("fixed validation pair SHA differs after create-only publication")
    print(
        canonical_json_bytes(
            {
                "receipt_file_sha256": receipt_sha,
                "validation_pair_sha256": first_sha,
            }
        ).decode(),
        flush=True,
    )
    return 0


CONTROLLER_STATUS_SCHEMA = (
    "bernini-action-edit-fresh-world8-level-a-controller-status-v2"
)


def write_controller_status(args: argparse.Namespace) -> int:
    release = validate_deployment_release(
        args.release_manifest,
        expected_manifest_sha256=args.expected_release_manifest_sha256,
        expected_driver_sha256=args.expected_driver_source_sha256,
    )
    validate_executed_driver(release)
    authority = validate_launch_authority_core(
        args.launch_authority_core,
        expected_sha256=args.expected_launch_authority_core_sha256,
        release=release,
    )
    intent = validate_attempt_intent(
        args.attempt_intent,
        expected_sha256=args.expected_attempt_intent_sha256,
        attempt_label=args.attempt_label,
        authority=authority,
    )
    try:
        child_exit = int(args.child_exit)
    except (TypeError, ValueError) as error:
        raise LevelADriverError("child exit is not an integer") from error
    if child_exit < 0 or child_exit > 255:
        fail("child exit is outside 0..255")
    unsigned = {
        "schema_version": CONTROLLER_STATUS_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "attempt": args.attempt_label,
        "attempt_intent_sha256": intent.sha256,
        "launch_authority_core_sha256": authority.sha256,
        "parent_job_id": PARENT_JOB_ID,
        "node": PINNED_NODE,
        "child_exit": child_exit,
        "attempt_claimed": True,
        "parent_state_before": args.parent_state_before,
        "parent_state_after": args.parent_state_after,
        "receipt_validated": False,
        "parent_cancelled": False,
        "parent_released": False,
        "parent_requeued": False,
        "parent_signalled": False,
        "automatic_relaunch_authorized": False,
    }
    final = {**unsigned, "status_digest": object_sha256(unsigned)}
    output = Path(args.output_controller_status).expanduser()
    if output != Path(intent.raw["attempt_root"]) / "controller.status.json":
        fail("controller status canonical path differs")
    status_sha = _atomic_write_canonical(output, final, expected_parent_mode=0o700)
    print(canonical_json_bytes({"controller_status_sha256": status_sha}).decode(), flush=True)
    return 0


def _read_controller_status(
    value: str | Path,
    *,
    authority: LaunchAuthority,
    intent: AttemptIntent,
) -> tuple[Mapping[str, Any], str]:
    path = _absolute_plain_file(
        value, label="controller status", required_mode=0o444, required_nlink=1
    )
    if path != Path(intent.raw["attempt_root"]) / "controller.status.json":
        fail("controller status canonical path differs")
    payload = stable_read(
        path, maximum=1024 * 1024, label="controller status", required_mode=0o444,
        required_nlink=1,
    )
    row = strict_json_bytes(payload, label="controller status")
    _exact_keys(
        row,
        {
            "schema_version", "method", "authority", "attempt",
            "attempt_intent_sha256", "launch_authority_core_sha256",
            "parent_job_id", "node", "child_exit", "attempt_claimed",
            "parent_state_before", "parent_state_after", "receipt_validated",
            "parent_cancelled", "parent_released", "parent_requeued",
            "parent_signalled", "automatic_relaunch_authorized", "status_digest",
        },
        label="controller status",
    )
    unsigned = dict(row)
    claimed = require_sha(unsigned.pop("status_digest", None), label="controller status digest")
    if (
        object_sha256(unsigned) != claimed
        or row.get("schema_version") != CONTROLLER_STATUS_SCHEMA
        or row.get("method") != METHOD
        or row.get("authority") != AUTHORITY
        or row.get("attempt") != intent.raw["attempt"]
        or row.get("attempt_intent_sha256") != intent.sha256
        or row.get("launch_authority_core_sha256") != authority.sha256
        or row.get("parent_job_id") != PARENT_JOB_ID
        or row.get("node") != PINNED_NODE
        or row.get("child_exit") != 0
        or row.get("attempt_claimed") is not True
        or row.get("receipt_validated") is not False
        or any(
            row.get(key) is not False
            for key in (
                "parent_cancelled", "parent_released", "parent_requeued",
                "parent_signalled", "automatic_relaunch_authorized",
            )
        )
    ):
        fail("controller status is not an exact pre-validation success boundary")
    return row, hashlib.sha256(payload).hexdigest()


def seal_attempt_terminal(args: argparse.Namespace) -> int:
    release = validate_deployment_release(
        args.release_manifest,
        expected_manifest_sha256=args.expected_release_manifest_sha256,
        expected_driver_sha256=args.expected_driver_source_sha256,
    )
    validate_executed_driver(release)
    authority = validate_launch_authority_core(
        args.launch_authority_core,
        expected_sha256=args.expected_launch_authority_core_sha256,
        release=release,
    )
    intent = validate_attempt_intent(
        args.attempt_intent,
        expected_sha256=args.expected_attempt_intent_sha256,
        attempt_label=args.attempt_label,
        authority=authority,
    )
    attempt_root = Path(intent.raw["attempt_root"])
    if (
        Path(args.validation_a).expanduser() != attempt_root / "STARTED" / "receipt-validation-1.json"
        or Path(args.validation_b).expanduser() != attempt_root / "STARTED" / "receipt-validation-2.json"
        or Path(args.controller_status).expanduser() != attempt_root / "controller.status.json"
        or Path(args.output_terminal_authority).expanduser()
        != attempt_root / "terminal.authority.json"
        or Path(args.output_success).expanduser() != attempt_root / "SUCCESS"
    ):
        fail("attempt terminal canonical path closure differs")
    _, receipt, receipt_sha, _ = _read_canonical_receipt(
        args.receipt, label="terminal-bound consumer receipt",
        authority=authority, intent=intent,
    )
    receipt_digest, consensus_sha = _validate_level_a_receipt(
        receipt, authority=authority, intent=intent
    )
    validation_a, validation_a_sha = _read_validation_record(
        args.validation_a, label="receipt validation A"
    )
    validation_b, validation_b_sha = _read_validation_record(
        args.validation_b, label="receipt validation B"
    )
    expected_validation = _validation_record(
        receipt=receipt, receipt_file_sha256=receipt_sha,
        authority=authority, intent=intent,
    )
    if (
        validation_a != validation_b
        or validation_a != expected_validation
        or validation_a_sha != validation_b_sha
    ):
        fail("two consecutive full receipt validations differ")
    _, controller_status_sha = _read_controller_status(
        args.controller_status, authority=authority, intent=intent
    )
    binding = receipt["launch_binding"]
    terminal_unsigned = {
        "schema_version": TERMINAL_AUTHORITY_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "status": "SUCCESS",
        "attempt": args.attempt_label,
        "attempt_intent_sha256": intent.sha256,
        "launch_authority_core_sha256": authority.sha256,
        "controller_status_sha256": controller_status_sha,
        "receipt_validation_sha256": validation_a_sha,
        "consecutive_full_validations": 2,
        "receipt_file_sha256": receipt_sha,
        "receipt_digest": receipt_digest,
        "world8_consensus_sha256": consensus_sha,
        "slurm_numeric_step": binding["slurm_numeric_step"],
        "node": binding["node"],
        "job_name": binding["job_name"],
        "receipt_validated": True,
        "parent_untouched": True,
        "automatic_relaunch_authorized": False,
        "promotion_authorized": False,
    }
    terminal = {
        **terminal_unsigned,
        "terminal_digest": object_sha256(terminal_unsigned),
    }
    terminal_path = Path(args.output_terminal_authority).expanduser()
    terminal_sha = _atomic_write_canonical(
        terminal_path, terminal, expected_parent_mode=0o700
    )
    success_unsigned = {
        "schema_version": TERMINAL_SUCCESS_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "status": "SUCCESS",
        "attempt": args.attempt_label,
        "attempt_intent_sha256": intent.sha256,
        "launch_authority_core_sha256": authority.sha256,
        "terminal_authority_sha256": terminal_sha,
        "receipt_file_sha256": receipt_sha,
        "receipt_digest": receipt_digest,
        "world8_consensus_sha256": consensus_sha,
        "slurm_numeric_step": binding["slurm_numeric_step"],
        "node": binding["node"],
        "job_name": binding["job_name"],
        "parent_untouched": True,
        "automatic_relaunch_authorized": False,
        "promotion_authorized": False,
    }
    success = {**success_unsigned, "success_digest": object_sha256(success_unsigned)}
    success_sha = _atomic_write_canonical(
        Path(args.output_success).expanduser(), success, expected_parent_mode=0o700
    )
    print(
        canonical_json_bytes(
            {
                "receipt_file_sha256": receipt_sha,
                "terminal_authority_sha256": terminal_sha,
                "success_sha256": success_sha,
            }
        ).decode(),
        flush=True,
    )
    return 0


def _read_terminal_pair(
    *,
    terminal_value: str | Path,
    success_value: str | Path,
    receipt: Mapping[str, Any],
    receipt_file_sha256: str,
    authority: LaunchAuthority,
    intent: AttemptIntent,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    terminal_path = _absolute_plain_file(
        terminal_value, label="terminal authority", required_mode=0o444,
        required_nlink=1,
    )
    if terminal_path != Path(intent.raw["attempt_root"]) / "terminal.authority.json":
        fail("terminal authority canonical path differs")
    terminal_payload = stable_read(
        terminal_path, maximum=1024 * 1024, label="terminal authority",
        required_mode=0o444, required_nlink=1,
    )
    terminal = strict_json_bytes(terminal_payload, label="terminal authority")
    _exact_keys(
        terminal,
        {
            "schema_version", "method", "authority", "status", "attempt",
            "attempt_intent_sha256", "launch_authority_core_sha256",
            "controller_status_sha256", "receipt_validation_sha256",
            "consecutive_full_validations", "receipt_file_sha256", "receipt_digest",
            "world8_consensus_sha256", "slurm_numeric_step", "node", "job_name",
            "receipt_validated", "parent_untouched",
            "automatic_relaunch_authorized", "promotion_authorized", "terminal_digest",
        },
        label="terminal authority",
    )
    terminal_unsigned = dict(terminal)
    terminal_digest = require_sha(
        terminal_unsigned.pop("terminal_digest", None), label="terminal authority digest"
    )
    receipt_digest, consensus_sha = _validate_level_a_receipt(
        receipt, authority=authority, intent=intent
    )
    binding = receipt["launch_binding"]
    terminal_sha = hashlib.sha256(terminal_payload).hexdigest()
    if (
        object_sha256(terminal_unsigned) != terminal_digest
        or terminal.get("schema_version") != TERMINAL_AUTHORITY_SCHEMA
        or terminal.get("method") != METHOD
        or terminal.get("authority") != AUTHORITY
        or terminal.get("status") != "SUCCESS"
        or terminal.get("attempt") != intent.raw["attempt"]
        or terminal.get("attempt_intent_sha256") != intent.sha256
        or terminal.get("launch_authority_core_sha256") != authority.sha256
        or terminal.get("consecutive_full_validations") != 2
        or terminal.get("receipt_file_sha256") != receipt_file_sha256
        or terminal.get("receipt_digest") != receipt_digest
        or terminal.get("world8_consensus_sha256") != consensus_sha
        or terminal.get("slurm_numeric_step") != binding["slurm_numeric_step"]
        or terminal.get("node") != binding["node"]
        or terminal.get("job_name") != binding["job_name"]
        or terminal.get("receipt_validated") is not True
        or terminal.get("parent_untouched") is not True
        or terminal.get("automatic_relaunch_authorized") is not False
        or terminal.get("promotion_authorized") is not False
    ):
        fail("terminal authority values differ")
    for key in ("controller_status_sha256", "receipt_validation_sha256"):
        require_sha(terminal.get(key), label=f"terminal authority {key}")

    success_path = _absolute_plain_file(
        success_value, label="terminal SUCCESS", required_mode=0o444,
        required_nlink=1,
    )
    if success_path != Path(intent.raw["attempt_root"]) / "SUCCESS":
        fail("terminal SUCCESS canonical path differs")
    success_payload = stable_read(
        success_path, maximum=1024 * 1024, label="terminal SUCCESS",
        required_mode=0o444, required_nlink=1,
    )
    success = strict_json_bytes(success_payload, label="terminal SUCCESS")
    _exact_keys(
        success,
        {
            "schema_version", "method", "authority", "status", "attempt",
            "attempt_intent_sha256", "launch_authority_core_sha256",
            "terminal_authority_sha256", "receipt_file_sha256", "receipt_digest",
            "world8_consensus_sha256", "slurm_numeric_step", "node", "job_name",
            "parent_untouched", "automatic_relaunch_authorized",
            "promotion_authorized", "success_digest",
        },
        label="terminal SUCCESS",
    )
    success_unsigned = dict(success)
    success_digest = require_sha(
        success_unsigned.pop("success_digest", None), label="terminal SUCCESS digest"
    )
    if (
        object_sha256(success_unsigned) != success_digest
        or success.get("schema_version") != TERMINAL_SUCCESS_SCHEMA
        or success.get("method") != METHOD
        or success.get("authority") != AUTHORITY
        or success.get("status") != "SUCCESS"
        or success.get("attempt") != intent.raw["attempt"]
        or success.get("attempt_intent_sha256") != intent.sha256
        or success.get("launch_authority_core_sha256") != authority.sha256
        or success.get("terminal_authority_sha256") != terminal_sha
        or success.get("receipt_file_sha256") != receipt_file_sha256
        or success.get("receipt_digest") != receipt_digest
        or success.get("world8_consensus_sha256") != consensus_sha
        or success.get("slurm_numeric_step") != binding["slurm_numeric_step"]
        or success.get("node") != binding["node"]
        or success.get("job_name") != binding["job_name"]
        or success.get("parent_untouched") is not True
        or success.get("automatic_relaunch_authorized") is not False
        or success.get("promotion_authorized") is not False
    ):
        fail("terminal SUCCESS values differ")
    return terminal, success


SACCT_FIELDS = (
    "JobIDRaw,JobName%128,State,ExitCode,NodeList%128,NNodes,NTasks,"
    "AllocTRES%512,Start,End"
)


def _parse_tres(value: Any) -> Mapping[str, str]:
    if not isinstance(value, str) or not value:
        fail("terminal sacct AllocTRES is absent")
    result: dict[str, str] = {}
    for token in value.split(","):
        if token.count("=") != 1:
            fail("terminal sacct AllocTRES encoding differs")
        key, item = token.split("=", 1)
        if not key or not item or key in result:
            fail("terminal sacct AllocTRES closure differs")
        result[key] = item
    return result


def validate_terminal_sacct_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    receipt_a: Mapping[str, Any],
    receipt_b: Mapping[str, Any],
    intent_a: AttemptIntent,
    intent_b: AttemptIntent,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate two exact completed Slurm steps and reject historical replay."""

    if any(not isinstance(row, Mapping) for row in rows):
        fail("terminal sacct row schema differs")
    expected_keys = {
        "JobIDRaw", "JobName", "State", "ExitCode", "NodeList", "NNodes",
        "NTasks", "AllocTRES", "Start", "End",
    }
    wanted = {
        receipt_a["launch_binding"]["slurm_numeric_step"]: (receipt_a, intent_a),
        receipt_b["launch_binding"]["slurm_numeric_step"]: (receipt_b, intent_b),
    }
    if len(wanted) != 2:
        fail("A/B launches reused one numeric Slurm step")
    matched: dict[str, Mapping[str, Any]] = {}
    expected_tres = {
        "cpu": "32", "gres/gpu:mi210": "8", "gres/gpu": "8",
        "mem": "60G", "node": "1",
    }
    for row in rows:
        if set(row) != expected_keys:
            fail("terminal sacct exact row schema differs")
        job_id = row.get("JobIDRaw")
        if job_id not in wanted:
            continue
        if job_id in matched:
            fail("terminal sacct contains a duplicate exact numeric step")
        receipt, intent = wanted[str(job_id)]
        binding = receipt["launch_binding"]
        try:
            start = datetime.fromisoformat(str(row.get("Start")))
            end = datetime.fromisoformat(str(row.get("End")))
            start_epoch = start.timestamp()
            end_epoch = end.timestamp()
        except (TypeError, ValueError, OSError) as error:
            raise LevelADriverError("terminal sacct time encoding differs") from error
        if (
            row.get("JobName") != binding["job_name"]
            or row.get("State") != "COMPLETED"
            or row.get("ExitCode") != "0:0"
            or row.get("NodeList") != PINNED_NODE
            or row.get("NNodes") != "1"
            or row.get("NTasks") != "1"
            or dict(_parse_tres(row.get("AllocTRES"))) != expected_tres
            or start_epoch <= intent.mtime_ns / 1_000_000_000
            or end_epoch < start_epoch
            or binding["attempt_intent_sha256"] != intent.sha256
        ):
            fail("terminal sacct authority differs or predates attempt claim")
        matched[str(job_id)] = dict(row)
    if set(matched) != set(wanted):
        fail("terminal sacct has not exposed both exact completed steps")
    return matched[receipt_a["launch_binding"]["slurm_numeric_step"]], matched[
        receipt_b["launch_binding"]["slurm_numeric_step"]
    ]


def query_terminal_sacct(
    *,
    receipt_a: Mapping[str, Any],
    receipt_b: Mapping[str, Any],
    intent_a: AttemptIntent,
    intent_b: AttemptIntent,
    polls: int = 60,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    step_ids = ",".join(
        (
            receipt_a["launch_binding"]["slurm_numeric_step"],
            receipt_b["launch_binding"]["slurm_numeric_step"],
        )
    )
    last_error = "terminal sacct did not return a usable row"
    for poll in range(polls):
        completed = subprocess.run(
            [
                "/usr/bin/sacct", "-n", "-P", "-j", step_ids, "-o", SACCT_FIELDS,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        if completed.returncode == 0:
            parsed: list[Mapping[str, Any]] = []
            for line in completed.stdout.decode("utf-8", errors="strict").splitlines():
                if not line:
                    continue
                columns = line.split("|")
                if len(columns) == 11 and columns[-1] == "":
                    columns.pop()
                if len(columns) != 10:
                    parsed = []
                    last_error = "terminal sacct column closure differs"
                    break
                parsed.append(
                    dict(
                        zip(
                            (
                                "JobIDRaw", "JobName", "State", "ExitCode",
                                "NodeList", "NNodes", "NTasks", "AllocTRES",
                                "Start", "End",
                            ),
                            columns,
                        )
                    )
                )
            try:
                return validate_terminal_sacct_rows(
                    parsed, receipt_a=receipt_a, receipt_b=receipt_b,
                    intent_a=intent_a, intent_b=intent_b,
                )
            except LevelADriverError as error:
                last_error = str(error)
        else:
            last_error = completed.stderr.decode("utf-8", errors="replace")[:512]
        if poll + 1 < polls:
            time.sleep(2)
    fail(f"terminal sacct validation timed out: {last_error}")


def compare_world8(args: argparse.Namespace) -> int:
    release = validate_deployment_release(
        args.release_manifest,
        expected_manifest_sha256=args.expected_release_manifest_sha256,
        expected_driver_sha256=args.expected_driver_source_sha256,
    )
    validate_executed_driver(release)
    authority = validate_launch_authority_core(
        args.launch_authority_core,
        expected_sha256=args.expected_launch_authority_core_sha256,
        release=release,
    )
    intent_a = validate_attempt_intent(
        args.intent_a,
        expected_sha256=args.expected_intent_a_sha256,
        attempt_label="A",
        authority=authority,
    )
    intent_b = validate_attempt_intent(
        args.intent_b,
        expected_sha256=args.expected_intent_b_sha256,
        attempt_label="B",
        authority=authority,
    )
    consumer = load_consumer_from_authenticated_bytes(release)
    path_a, receipt_a, sha_a, inode_a = _read_canonical_receipt(
        args.receipt_a, label="WORLD8 receipt A", authority=authority,
        intent=intent_a,
    )
    path_b, receipt_b, sha_b, inode_b = _read_canonical_receipt(
        args.receipt_b, label="WORLD8 receipt B", authority=authority,
        intent=intent_b,
    )
    if path_a == path_b or inode_a == inode_b or path_a.parent == path_b.parent:
        fail("WORLD8 A/B receipts do not have distinct attempt roots and inodes")
    terminal_a, success_a = _read_terminal_pair(
        terminal_value=args.terminal_a,
        success_value=args.success_a,
        receipt=receipt_a,
        receipt_file_sha256=sha_a,
        authority=authority,
        intent=intent_a,
    )
    terminal_b, success_b = _read_terminal_pair(
        terminal_value=args.terminal_b,
        success_value=args.success_b,
        receipt=receipt_b,
        receipt_file_sha256=sha_b,
        authority=authority,
        intent=intent_b,
    )
    if (
        success_a["slurm_numeric_step"] == success_b["slurm_numeric_step"]
        or success_a["job_name"] == success_b["job_name"]
    ):
        fail("A/B terminal SUCCESS replayed one launch identity")
    sacct_a, sacct_b = query_terminal_sacct(
        receipt_a=receipt_a, receipt_b=receipt_b,
        intent_a=intent_a, intent_b=intent_b,
    )
    parity = consumer.compare_fresh_world8_consumer_receipts(receipt_a, receipt_b)
    if (
        not isinstance(parity, Mapping)
        or parity.get("authority") != AUTHORITY
        or parity.get("exact_parity") is not True
        or parity.get("world8_launches") != 2
        or parity.get("distinct_fresh_process_sessions") != 16
        or parity.get("os_process_independence_proven") is not True
        or parity.get("training_attached_reference") is not False
        or parity.get("full_bernini_renderer_forward_parity_claimed") is not False
        or parity.get("promotion_authorized") is not False
        or parity.get("promotable") is not False
    ):
        fail("frozen A/B comparator result differs")
    unsigned = {
        "schema_version": PARITY_RECEIPT_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "deployment_release_manifest_sha256": release.manifest_sha256,
        "driver_source_sha256": release.driver_sha256,
        "launch_authority_core_sha256": authority.sha256,
        "consumer_source_sha256": PINNED_CONSUMER_SHA256,
        "product_bridge_source_sha256": PINNED_PRODUCT_SHA256,
        "checkpoint_step": P2_STEP,
        "checkpoint_parameter_sha256": PINNED_P2_PARAMETER_SHA256,
        "receipt_a_file_sha256": sha_a,
        "receipt_b_file_sha256": sha_b,
        "intent_a_sha256": intent_a.sha256,
        "intent_b_sha256": intent_b.sha256,
        "terminal_a_digest": terminal_a["terminal_digest"],
        "terminal_b_digest": terminal_b["terminal_digest"],
        "success_a_digest": success_a["success_digest"],
        "success_b_digest": success_b["success_digest"],
        "slurm_step_a": success_a["slurm_numeric_step"],
        "slurm_step_b": success_b["slurm_numeric_step"],
        "terminal_sacct": {"A": dict(sacct_a), "B": dict(sacct_b)},
        "two_distinct_exact_completed_slurm_steps_verified": True,
        "steps_started_after_attempt_claims_verified": True,
        "fresh_world8_parity": dict(parity),
        "conditioner_predictor_plus_exact30_cell_only": True,
        "full_bernini_renderer_forward_executed": False,
        "offline_product_inference_completed": False,
        "full40_denoise_executed": False,
        "mp4_emitted": False,
        "training_started": False,
        "formal_training_started": False,
        "counts_as_d0": False,
        "scientific_claim_authorized": False,
        "promotion_authorized": False,
        "promotable": False,
    }
    final = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    parity_path = Path(args.output_parity_receipt).expanduser()
    if not parity_path.is_absolute() or parity_path.is_symlink():
        fail("parity output must be one absolute absent non-symlink path")
    parity_sha = _atomic_write_canonical(
        parity_path, final, expected_parent_mode=0o700
    )
    print(
        f"PASS Level-A independent WORLD8 A/B parity receipt_sha256={parity_sha} "
        "renderer=false full40=false mp4=false promotion=false",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--release-manifest", required=True)
    common.add_argument("--expected-release-manifest-sha256", required=True)
    common.add_argument("--expected-driver-source-sha256", required=True)

    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--attempt-label", required=True, choices=("A", "B"))
    run.add_argument("--launch-authority-core", required=True)
    run.add_argument("--expected-launch-authority-core-sha256", required=True)
    run.add_argument("--attempt-intent", required=True)
    run.add_argument("--expected-attempt-intent-sha256", required=True)
    run.add_argument("--r2-release-manifest", required=True)
    run.add_argument("--campaign-receipt", required=True)
    run.add_argument("--checkpoint-dir", required=True)
    run.add_argument("--bernini-root", required=True)
    run.add_argument("--veomni-root", required=True)
    run.add_argument("--base-checkpoint", required=True)
    run.add_argument("--checkpoint-content-manifest", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--expected-consumer-source-sha256", required=True)
    run.add_argument("--expected-product-source-sha256", required=True)

    validate = subparsers.add_parser("validate-receipt", parents=[common])
    validate.add_argument("--attempt-label", required=True, choices=("A", "B"))
    validate.add_argument("--launch-authority-core", required=True)
    validate.add_argument("--expected-launch-authority-core-sha256", required=True)
    validate.add_argument("--attempt-intent", required=True)
    validate.add_argument("--expected-attempt-intent-sha256", required=True)
    validate.add_argument("--receipt", required=True)
    validate.add_argument("--output-validation")

    publish = subparsers.add_parser("publish-validation-pair", parents=[common])
    publish.add_argument("--attempt-label", required=True, choices=("A", "B"))
    publish.add_argument("--launch-authority-core", required=True)
    publish.add_argument("--expected-launch-authority-core-sha256", required=True)
    publish.add_argument("--attempt-intent", required=True)
    publish.add_argument("--expected-attempt-intent-sha256", required=True)
    publish.add_argument("--receipt", required=True)
    publish.add_argument("--validation-json-a", required=True)
    publish.add_argument("--validation-json-b", required=True)
    publish.add_argument("--output-validation-a", required=True)
    publish.add_argument("--output-validation-b", required=True)

    status = subparsers.add_parser("write-controller-status", parents=[common])
    status.add_argument("--attempt-label", required=True, choices=("A", "B"))
    status.add_argument("--launch-authority-core", required=True)
    status.add_argument("--expected-launch-authority-core-sha256", required=True)
    status.add_argument("--attempt-intent", required=True)
    status.add_argument("--expected-attempt-intent-sha256", required=True)
    status.add_argument("--child-exit", required=True)
    status.add_argument("--parent-state-before", required=True)
    status.add_argument("--parent-state-after", required=True)
    status.add_argument("--output-controller-status", required=True)

    terminal = subparsers.add_parser("seal-attempt-terminal", parents=[common])
    terminal.add_argument("--attempt-label", required=True, choices=("A", "B"))
    terminal.add_argument("--launch-authority-core", required=True)
    terminal.add_argument("--expected-launch-authority-core-sha256", required=True)
    terminal.add_argument("--attempt-intent", required=True)
    terminal.add_argument("--expected-attempt-intent-sha256", required=True)
    terminal.add_argument("--receipt", required=True)
    terminal.add_argument("--validation-a", required=True)
    terminal.add_argument("--validation-b", required=True)
    terminal.add_argument("--controller-status", required=True)
    terminal.add_argument("--output-terminal-authority", required=True)
    terminal.add_argument("--output-success", required=True)

    compare = subparsers.add_parser("compare", parents=[common])
    compare.add_argument("--launch-authority-core", required=True)
    compare.add_argument("--expected-launch-authority-core-sha256", required=True)
    compare.add_argument("--intent-a", required=True)
    compare.add_argument("--expected-intent-a-sha256", required=True)
    compare.add_argument("--intent-b", required=True)
    compare.add_argument("--expected-intent-b-sha256", required=True)
    compare.add_argument("--receipt-a", required=True)
    compare.add_argument("--receipt-b", required=True)
    compare.add_argument("--terminal-a", required=True)
    compare.add_argument("--terminal-b", required=True)
    compare.add_argument("--success-a", required=True)
    compare.add_argument("--success-b", required=True)
    compare.add_argument("--output-parity-receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_world8(args)
    if args.command == "validate-receipt":
        return validate_receipt_command(args)
    if args.command == "publish-validation-pair":
        return publish_validation_pair(args)
    if args.command == "write-controller-status":
        return write_controller_status(args)
    if args.command == "seal-attempt-terminal":
        return seal_attempt_terminal(args)
    if args.command == "compare":
        return compare_world8(args)
    fail("unsupported command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LevelADriverError as error:
        print(f"Level-A driver refused: {error}", file=sys.stderr, flush=True)
        raise SystemExit(94) from error


__all__ = [
    "ATTEMPT_INTENT_SCHEMA",
    "ATTEMPT_JOB_NAMES",
    "AUTHORITY",
    "CONSUMER_FILENAME",
    "DRIVER_FILENAME",
    "DeploymentRelease",
    "EXPECTED_RELEASE_PATHS",
    "LevelADriverError",
    "LaunchAuthority",
    "AttemptIntent",
    "LAUNCH_AUTHORITY_SCHEMA",
    "LAUNCH_BINDING_SCHEMA",
    "METHOD",
    "OUTPUT_RECEIPT_FILENAME",
    "PARITY_RECEIPT_SCHEMA",
    "PUBLISHED_RECEIPT_KEYS",
    "RAW_RECEIPT_KEYS",
    "PINNED_CONSUMER_SHA256",
    "PINNED_PRODUCT_SHA256",
    "PRODUCT_FILENAME",
    "RELEASE_SCHEMA",
    "TERMINAL_AUTHORITY_SCHEMA",
    "TERMINAL_SUCCESS_SCHEMA",
    "canonical_json_bytes",
    "compare_world8",
    "load_consumer_from_authenticated_bytes",
    "object_sha256",
    "run_world8",
    "strict_json_bytes",
    "validate_attempt_intent",
    "validate_deployment_release",
    "validate_launch_authority_core",
    "validate_terminal_sacct_rows",
]
