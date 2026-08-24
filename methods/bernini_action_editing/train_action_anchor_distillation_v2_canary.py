#!/usr/bin/env python3
"""Fail-closed two-update action-anchor canary integration for Bernini.

This module is intentionally a small integration layer, not a second copy of
the Bernini loader.  It reuses the real paired-flow preparation, exact-30
post-block hooks, checkpoint route, and renderer ``shared_step`` from
``train_action_edit_large_lora_0817_v1``.  It replaces only the objective:

* ``q_pred`` is produced in FP32 from clean source + instruction only;
* externally frozen, independently qualified ``q_y`` is the sole point
  teacher;
* externally classified ``q_anchor`` values enter only routed InfoNCE;
* the genuine Bernini flow MSE is the preservation term; and
* the combined loss is back-propagated through the exact-30 target route.

No sidecar is inferred, repaired, generated, or authorized here.  Every JSON
and raw FP32 tensor is a regular non-symlink file below one frozen sidecar
root.  The launch supplies independent expected hashes; hashes declared only
inside the candidate manifest are never trust roots.  Full teacher validation
must finish before a caller constructs an optimizer.

The public runtime functions are suitable for the already-initialized WORLD8
objects created by the 0817 runner.  The command-line surface is a CPU-only
preflight/schema tool.  This separation lets a launcher authenticate missing
or malformed sidecars before importing Bernini or reserving model memory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import threading
from typing import Any, Callable, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

METHOD = "bernini-action-anchor-distillation-v2-canary"
AUTHORITY = "EXPLORATORY_TWO_UPDATE_ONLY"
SIDECAR_SCHEMA = "bernini-action-anchor-v2-frozen-sidecar-manifest-v1"
PREFLIGHT_SCHEMA = "bernini-action-anchor-v2-sidecar-preflight-v1"
OBJECTIVE_SCHEMA = "bernini-action-anchor-v2-objective-v1"
STEP1_AUDIT_SCHEMA = "bernini-action-anchor-v2-step1-gradient-audit-v1"
MAX_UPDATES = 2
WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
GRADIENT_ACCUMULATION = 4
GLOBAL_RECORDS = MAX_UPDATES * DP_SIZE * GRADIENT_ACCUMULATION
PHASE_COUNT = 21
ACTION_WIDTH = 256
HIDDEN_WIDTH = 1536
TRANSFORMER_BLOCKS = 30
RAW_PHASE_BYTES = PHASE_COUNT * ACTION_WIDTH * 4
RAW_GLOBAL_BYTES = ACTION_WIDTH * 4
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_JSON_LEAF_BYTES = 8 * 1024 * 1024
MAX_ANCHORS_PER_RECORD = 32
PREDICTOR_SOURCE_NAME = "action_plan_predictor_v1.py"
DISTILLATION_SOURCE_NAME = "action_anchor_distillation_v1.py"
RENDERER_RUNNER_SOURCE_NAME = "train_action_edit_large_lora_0817_v1.py"
SCHEDULE_SOURCE_NAME = "clean_source_visual_context_stage_b_contract_v1.py"
PACKED_CORE_SOURCE_NAME = "packed_preservation_lora_v2.py"
RUNTIME_SOURCE_NAME = "source_self_runtime.py"
LEGACY_LOADER_SOURCE_NAME = "train_lora.py"
WORLD8_ADAPTER_SOURCE_NAME = "train_action_anchor_distillation_v2_world8.py"
INFERENCE_SIGMA_SOURCE_NAME = "inference_sigma_strata.py"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = {
    "schema_version",
    "method",
    "authority",
    "complete",
    "exploratory_only",
    "formal_training_authorized",
    "scientific_claim_authorized",
    "optimizer_updates",
    "world_size",
    "dp_size",
    "sp_size",
    "gradient_accumulation",
    "teacher_authority_file",
    "teacher_authority_file_sha256",
    "teacher_authority_sha256",
    "classification_authority_sha256",
    "predictor_source_sha256",
    "distillation_source_sha256",
    "renderer_runner_source_sha256",
    "v2_runner_source_sha256",
    "schedule_source_sha256",
    "packed_core_source_sha256",
    "runtime_source_sha256",
    "legacy_loader_source_sha256",
    "world8_adapter_source_sha256",
    "inference_sigma_source_sha256",
    "renderer_release_manifest_sha256",
    "records",
    "manifest_digest",
}
_RECORD_FIELDS = {
    "logical_record",
    "dataset_iid",
    "dataset_row_index",
    "row_id",
    "source_media_file",
    "target_media_file",
    "source_sha256",
    "target_sha256",
    "instruction_file",
    "instruction_sha256",
    "source_mode_fp32le_file",
    "source_mode_fp32le_sha256",
    "source_mode_shape",
    "target_mode_fp32le_file",
    "target_mode_fp32le_sha256",
    "target_mode_shape",
    "source_mode_tensor_sha256",
    "target_mode_tensor_sha256",
    "renderer_text_tensor_sha256",
    "instruction_tokens_tensor_sha256",
    "q_y",
    "anchors",
}
_Q_LEAF_FIELDS = {
    "phase_fp32le_file",
    "phase_fp32le_sha256",
    "global_fp32le_file",
    "global_fp32le_sha256",
    "q_receipt_file",
    "q_receipt_file_sha256",
    "qualification_receipt_digest",
}
_ANCHOR_LEAF_FIELDS = _Q_LEAF_FIELDS | {
    "compatibility_receipt_file",
    "compatibility_receipt_file_sha256",
    "compatibility_decision_receipt_digest",
}


class ActionAnchorV2CanaryError(RuntimeError):
    """Raised before an unqualified teacher or ambiguous update can run."""


def fail(message: str) -> NoReturn:
    raise ActionAnchorV2CanaryError(message)


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
        raise ActionAnchorV2CanaryError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def runtime_tensor_sha256_v2(value: Any) -> str:
    """Hash shape, dtype, and exact tensor bytes without NumPy."""

    import torch

    if (
        type(value) is not torch.Tensor
        or value.device.type == "meta"
        or value.layout != torch.strided
    ):
        fail("runtime tensor hash requires one materialized strided tensor")
    tensor = value.detach().contiguous().cpu()
    metadata = canonical_json_bytes(
        {"shape": list(map(int, tensor.shape)), "dtype": str(tensor.dtype)}
    )
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = tensor.view(torch.uint8).reshape(-1)
    for start in range(0, int(raw.numel()), 1024 * 1024):
        digest.update(bytes(raw[start : start + 1024 * 1024].tolist()))
    return digest.hexdigest()


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
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _closed_dict(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        actual = set(value) if type(value) is dict else set()
        fail(
            f"{label} field closure differs: "
            f"missing={sorted(fields - actual)} extra={sorted(actual - fields)}"
        )
    return value


def _sha256(value: Any, *, label: str, authority: bool = False) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be one lowercase SHA-256")
    if authority and value == "0" * 64:
        fail(f"{label} cannot be an all-zero placeholder")
    return value


def _plain_absolute_file(path: Path, *, label: str, max_bytes: int) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        metadata = requested.lstat()
    except OSError as error:
        raise ActionAnchorV2CanaryError(f"{label} is unavailable: {error}") from error
    if (
        resolved != requested
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > max_bytes
    ):
        fail(f"{label} canonical regular-file/size contract differs")
    return resolved


def _read_pinned_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    max_bytes: int = MAX_JSON_LEAF_BYTES,
) -> dict[str, Any]:
    expected = _sha256(expected_sha256, label=f"expected {label} SHA-256", authority=True)
    resolved = _plain_absolute_file(path, label=label, max_bytes=max_bytes)
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or hashlib.sha256(payload).hexdigest() != expected:
        fail(f"{label} stable bytes/SHA-256 differ")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActionAnchorV2CanaryError(f"{label} is not UTF-8 JSON") from error
    if type(value) is not dict:
        fail(f"{label} must contain one exact JSON object")
    return value


def _relative_member(
    root: Path,
    value: Any,
    *,
    label: str,
    max_bytes: int = MAX_JSON_LEAF_BYTES,
) -> Path:
    if type(value) is not str:
        fail(f"{label} must be one canonical relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        fail(f"{label} must be one canonical relative path")
    candidate = root.joinpath(*pure.parts)
    resolved = _plain_absolute_file(candidate, label=label, max_bytes=max_bytes)
    try:
        resolved.relative_to(root)
    except ValueError:
        fail(f"{label} escaped the frozen sidecar root")
    return resolved


def _validate_leaf_file(
    root: Path,
    row: Mapping[str, Any],
    *,
    path_field: str,
    sha_field: str,
    label: str,
    exact_bytes: Optional[int] = None,
    max_bytes: int = MAX_JSON_LEAF_BYTES,
) -> Path:
    expected = _sha256(row[sha_field], label=f"{label} SHA-256", authority=True)
    path = _relative_member(root, row[path_field], label=label, max_bytes=max_bytes)
    if exact_bytes is not None and path.stat().st_size != exact_bytes:
        fail(f"{label} raw FP32 byte length differs")
    if file_sha256(path) != expected:
        fail(f"{label} file SHA-256 differs")
    return path


def sidecar_schema_template_v1() -> dict[str, Any]:
    """Return the exact required manifest shape with non-authoritative markers."""

    digest = "<externally-pinned-lowercase-sha256>"
    q_leaf = {
        "phase_fp32le_file": "q/row-000/q_y.phase.fp32le",
        "phase_fp32le_sha256": digest,
        "global_fp32le_file": "q/row-000/q_y.global.fp32le",
        "global_fp32le_sha256": digest,
        "q_receipt_file": "receipts/row-000.q_y.json",
        "q_receipt_file_sha256": digest,
        "qualification_receipt_digest": digest,
    }
    anchor = {
        **q_leaf,
        "phase_fp32le_file": "q/row-000/anchor-reverse.phase.fp32le",
        "global_fp32le_file": "q/row-000/anchor-reverse.global.fp32le",
        "q_receipt_file": "receipts/row-000.anchor-reverse.json",
        "compatibility_receipt_file": (
            "receipts/row-000.anchor-reverse.compatibility.json"
        ),
        "compatibility_receipt_file_sha256": digest,
        "compatibility_decision_receipt_digest": digest,
    }
    record = {
        "logical_record": 0,
        "dataset_iid": "<exact-dataset-iid>",
        "dataset_row_index": 0,
        "row_id": digest,
        "source_media_file": "media/row-000.source.mp4",
        "target_media_file": "media/row-000.target.mp4",
        "source_sha256": digest,
        "target_sha256": digest,
        "instruction_file": "instructions/row-000.utf8",
        "instruction_sha256": digest,
        "source_mode_fp32le_file": "latents/row-000.source-mode.fp32le",
        "source_mode_fp32le_sha256": digest,
        "source_mode_shape": [1, 16, 21, 30, 44],
        "target_mode_fp32le_file": "latents/row-000.target-mode.fp32le",
        "target_mode_fp32le_sha256": digest,
        "target_mode_shape": [1, 16, 21, 30, 44],
        "source_mode_tensor_sha256": digest,
        "target_mode_tensor_sha256": digest,
        "renderer_text_tensor_sha256": digest,
        "instruction_tokens_tensor_sha256": digest,
        "q_y": q_leaf,
        "anchors": [anchor],
    }
    return {
        "schema_version": SIDECAR_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "complete": True,
        "exploratory_only": True,
        "formal_training_authorized": False,
        "scientific_claim_authorized": False,
        "optimizer_updates": MAX_UPDATES,
        "world_size": WORLD_SIZE,
        "dp_size": DP_SIZE,
        "sp_size": SP_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "teacher_authority_file": "authority/teacher-authority.json",
        "teacher_authority_file_sha256": digest,
        "teacher_authority_sha256": digest,
        "classification_authority_sha256": digest,
        "predictor_source_sha256": digest,
        "distillation_source_sha256": digest,
        "renderer_runner_source_sha256": digest,
        "v2_runner_source_sha256": digest,
        "schedule_source_sha256": digest,
        "packed_core_source_sha256": digest,
        "runtime_source_sha256": digest,
        "legacy_loader_source_sha256": digest,
        "world8_adapter_source_sha256": digest,
        "inference_sigma_source_sha256": digest,
        "renderer_release_manifest_sha256": digest,
        "records": [
            {**record, "logical_record": index, "dataset_row_index": index}
            for index in range(GLOBAL_RECORDS)
        ],
        "manifest_digest": "<sha256-of-canonical-object-without-manifest_digest>",
    }


@dataclass(frozen=True)
class FrozenQLeafV2:
    phase_path: Path
    phase_sha256: str
    global_path: Path
    global_sha256: str
    q_receipt: Mapping[str, Any]
    q_receipt_file_sha256: str
    qualification_receipt_digest: str
    compatibility_receipt: Optional[Mapping[str, Any]] = None
    compatibility_receipt_file_sha256: Optional[str] = None
    compatibility_decision_receipt_digest: Optional[str] = None


@dataclass(frozen=True)
class FrozenRecordV2:
    logical_record: int
    dataset_iid: str
    dataset_row_index: int
    row_id: str
    source_media_path: Path
    target_media_path: Path
    source_sha256: str
    target_sha256: str
    instruction_path: Path
    instruction_sha256: str
    source_mode_path: Path
    source_mode_fp32le_sha256: str
    source_mode_shape: tuple[int, ...]
    target_mode_path: Path
    target_mode_fp32le_sha256: str
    target_mode_shape: tuple[int, ...]
    source_mode_tensor_sha256: str
    target_mode_tensor_sha256: str
    renderer_text_tensor_sha256: str
    instruction_tokens_tensor_sha256: str
    q_y: FrozenQLeafV2
    anchors: tuple[FrozenQLeafV2, ...]


_PREFLIGHT_REGISTRY_LOCK = threading.Lock()
_PREFLIGHT_REGISTRY: dict[Any, tuple[object, str]] = {}


@dataclass(frozen=True, eq=False)
class FrozenSidecarPreflightV2:
    schema_version: str
    manifest_path: Path
    manifest_file_sha256: str
    manifest_digest: str
    root: Path
    teacher_authority: Mapping[str, Any]
    teacher_authority_file_sha256: str
    teacher_authority_sha256: str
    classification_authority_sha256: str
    predictor_source_sha256: str
    distillation_source_sha256: str
    renderer_runner_source_sha256: str
    v2_runner_source_sha256: str
    schedule_source_sha256: str
    packed_core_source_sha256: str
    runtime_source_sha256: str
    legacy_loader_source_sha256: str
    world8_adapter_source_sha256: str
    inference_sigma_source_sha256: str
    renderer_release_manifest_path: Path
    renderer_release_manifest_sha256: str
    renderer_release_closure: Mapping[str, Any]
    records: tuple[FrozenRecordV2, ...]
    all_leaf_files_verified_before_optimizer: bool
    _lease: Any = field(repr=False, compare=False)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": PREFLIGHT_SCHEMA,
            "manifest_path": str(self.manifest_path),
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_digest": self.manifest_digest,
            "teacher_authority_file_sha256": self.teacher_authority_file_sha256,
            "teacher_authority_sha256": self.teacher_authority_sha256,
            "classification_authority_sha256": self.classification_authority_sha256,
            "predictor_source_sha256": self.predictor_source_sha256,
            "distillation_source_sha256": self.distillation_source_sha256,
            "renderer_runner_source_sha256": self.renderer_runner_source_sha256,
            "v2_runner_source_sha256": self.v2_runner_source_sha256,
            "schedule_source_sha256": self.schedule_source_sha256,
            "packed_core_source_sha256": self.packed_core_source_sha256,
            "runtime_source_sha256": self.runtime_source_sha256,
            "legacy_loader_source_sha256": self.legacy_loader_source_sha256,
            "world8_adapter_source_sha256": self.world8_adapter_source_sha256,
            "inference_sigma_source_sha256": self.inference_sigma_source_sha256,
            "renderer_release_manifest_path": str(
                self.renderer_release_manifest_path
            ),
            "renderer_release_manifest_sha256": (
                self.renderer_release_manifest_sha256
            ),
            "records": len(self.records),
            "anchors": sum(len(row.anchors) for row in self.records),
            "optimizer_updates": MAX_UPDATES,
            "all_leaf_files_verified_before_optimizer": (
                self.all_leaf_files_verified_before_optimizer
            ),
            "candidate_teacher_self_authorization_used": False,
            "formal_training_authorized": False,
            "scientific_claim_authorized": False,
        }


def _preflight_signature_v2(preflight: FrozenSidecarPreflightV2) -> str:
    records = []
    for row in preflight.records:
        leaves = (row.q_y, *row.anchors)
        records.append(
            {
                "logical_record": row.logical_record,
                "dataset_iid": row.dataset_iid,
                "dataset_row_index": row.dataset_row_index,
                "row_id": row.row_id,
                "media": [
                    str(row.source_media_path),
                    row.source_sha256,
                    str(row.target_media_path),
                    row.target_sha256,
                ],
                "instruction": [str(row.instruction_path), row.instruction_sha256],
                "modes": [
                    str(row.source_mode_path),
                    row.source_mode_fp32le_sha256,
                    list(row.source_mode_shape),
                    row.source_mode_tensor_sha256,
                    str(row.target_mode_path),
                    row.target_mode_fp32le_sha256,
                    list(row.target_mode_shape),
                    row.target_mode_tensor_sha256,
                ],
                "text": [
                    row.renderer_text_tensor_sha256,
                    row.instruction_tokens_tensor_sha256,
                ],
                "q": [
                    {
                        "phase": [str(leaf.phase_path), leaf.phase_sha256],
                        "global": [str(leaf.global_path), leaf.global_sha256],
                        "q_receipt": object_sha256(dict(leaf.q_receipt)),
                        "q_receipt_file_sha256": leaf.q_receipt_file_sha256,
                        "qualification": leaf.qualification_receipt_digest,
                        "compatibility": (
                            object_sha256(dict(leaf.compatibility_receipt))
                            if leaf.compatibility_receipt is not None
                            else None
                        ),
                        "compatibility_file_sha256": (
                            leaf.compatibility_receipt_file_sha256
                        ),
                        "decision": leaf.compatibility_decision_receipt_digest,
                    }
                    for leaf in leaves
                ],
            }
        )
    return object_sha256(
        {
            "receipt": preflight.receipt(),
            "root": str(preflight.root),
            "teacher_authority": object_sha256(dict(preflight.teacher_authority)),
            "renderer_release_closure": object_sha256(
                dict(preflight.renderer_release_closure)
            ),
            "records": records,
        }
    )


def _validate_q_leaf(
    root: Path,
    value: Any,
    *,
    label: str,
    anchor: bool,
) -> FrozenQLeafV2:
    fields = _ANCHOR_LEAF_FIELDS if anchor else _Q_LEAF_FIELDS
    row = _closed_dict(value, fields, label=label)
    phase_path = _validate_leaf_file(
        root,
        row,
        path_field="phase_fp32le_file",
        sha_field="phase_fp32le_sha256",
        label=f"{label} phase tensor",
        exact_bytes=RAW_PHASE_BYTES,
    )
    global_path = _validate_leaf_file(
        root,
        row,
        path_field="global_fp32le_file",
        sha_field="global_fp32le_sha256",
        label=f"{label} global tensor",
        exact_bytes=RAW_GLOBAL_BYTES,
    )
    receipt_path = _validate_leaf_file(
        root,
        row,
        path_field="q_receipt_file",
        sha_field="q_receipt_file_sha256",
        label=f"{label} q receipt",
    )
    q_receipt = _read_pinned_json(
        receipt_path,
        expected_sha256=row["q_receipt_file_sha256"],
        label=f"{label} q receipt",
    )
    expected_kind = "q_anchor" if anchor else "q_y"
    if q_receipt.get("q_kind") != expected_kind:
        fail(f"{label} receipt kind is not {expected_kind}")
    qualification = _sha256(
        row["qualification_receipt_digest"],
        label=f"{label} qualification receipt digest",
        authority=True,
    )
    compatibility: Optional[Mapping[str, Any]] = None
    compatibility_file_sha: Optional[str] = None
    decision_digest: Optional[str] = None
    if anchor:
        compatibility_path = _validate_leaf_file(
            root,
            row,
            path_field="compatibility_receipt_file",
            sha_field="compatibility_receipt_file_sha256",
            label=f"{label} compatibility receipt",
        )
        compatibility_file_sha = row["compatibility_receipt_file_sha256"]
        compatibility = _read_pinned_json(
            compatibility_path,
            expected_sha256=compatibility_file_sha,
            label=f"{label} compatibility receipt",
        )
        decision_digest = _sha256(
            row["compatibility_decision_receipt_digest"],
            label=f"{label} compatibility decision digest",
            authority=True,
        )
    return FrozenQLeafV2(
        phase_path=phase_path,
        phase_sha256=row["phase_fp32le_sha256"],
        global_path=global_path,
        global_sha256=row["global_fp32le_sha256"],
        q_receipt=q_receipt,
        q_receipt_file_sha256=row["q_receipt_file_sha256"],
        qualification_receipt_digest=qualification,
        compatibility_receipt=compatibility,
        compatibility_receipt_file_sha256=compatibility_file_sha,
        compatibility_decision_receipt_digest=decision_digest,
    )


def preflight_frozen_sidecars_v2(
    manifest_path: Path,
    *,
    renderer_release_manifest_path: Path,
    expected_manifest_file_sha256: str,
    expected_renderer_release_manifest_sha256: str,
    expected_teacher_authority_file_sha256: str,
    expected_teacher_authority_sha256: str,
    expected_classification_authority_sha256: str,
    expected_predictor_source_sha256: str,
    expected_distillation_source_sha256: str,
    expected_renderer_runner_source_sha256: str,
    expected_v2_runner_source_sha256: str,
    expected_schedule_source_sha256: str,
    expected_packed_core_source_sha256: str,
    expected_runtime_source_sha256: str,
    expected_legacy_loader_source_sha256: str,
    expected_world8_adapter_source_sha256: str,
    expected_inference_sigma_source_sha256: str,
) -> FrozenSidecarPreflightV2:
    """Authenticate the entire 16-record closure before model/optimizer setup."""

    expected_manifest_file_sha256 = _sha256(
        expected_manifest_file_sha256,
        label="externally expected sidecar manifest file SHA-256",
        authority=True,
    )
    expected_renderer_release_manifest_sha256 = _sha256(
        expected_renderer_release_manifest_sha256,
        label="externally expected 0817 release manifest SHA-256",
        authority=True,
    )
    expected_teacher_authority_file_sha256 = _sha256(
        expected_teacher_authority_file_sha256,
        label="externally expected teacher authority file SHA-256",
        authority=True,
    )
    expected_teacher_authority_sha256 = _sha256(
        expected_teacher_authority_sha256,
        label="externally expected teacher authority SHA-256",
        authority=True,
    )
    expected_classification_authority_sha256 = _sha256(
        expected_classification_authority_sha256,
        label="externally expected classification authority SHA-256",
        authority=True,
    )
    expected_predictor_source_sha256 = _sha256(
        expected_predictor_source_sha256,
        label="externally expected predictor source SHA-256",
        authority=True,
    )
    expected_distillation_source_sha256 = _sha256(
        expected_distillation_source_sha256,
        label="externally expected distillation source SHA-256",
        authority=True,
    )
    expected_renderer_runner_source_sha256 = _sha256(
        expected_renderer_runner_source_sha256,
        label="externally expected 0817 renderer runner source SHA-256",
        authority=True,
    )
    expected_v2_runner_source_sha256 = _sha256(
        expected_v2_runner_source_sha256,
        label="externally expected V2 runner source SHA-256",
        authority=True,
    )
    expected_schedule_source_sha256 = _sha256(
        expected_schedule_source_sha256,
        label="externally expected schedule source SHA-256",
        authority=True,
    )
    expected_packed_core_source_sha256 = _sha256(
        expected_packed_core_source_sha256,
        label="externally expected packed-core source SHA-256",
        authority=True,
    )
    expected_runtime_source_sha256 = _sha256(
        expected_runtime_source_sha256,
        label="externally expected distributed-runtime source SHA-256",
        authority=True,
    )
    expected_legacy_loader_source_sha256 = _sha256(
        expected_legacy_loader_source_sha256,
        label="externally expected legacy-loader source SHA-256",
        authority=True,
    )
    expected_world8_adapter_source_sha256 = _sha256(
        expected_world8_adapter_source_sha256,
        label="externally expected WORLD8 adapter source SHA-256",
        authority=True,
    )
    expected_inference_sigma_source_sha256 = _sha256(
        expected_inference_sigma_source_sha256,
        label="externally expected inference-sigma source SHA-256",
        authority=True,
    )
    manifest_path = _plain_absolute_file(
        manifest_path, label="sidecar manifest", max_bytes=MAX_MANIFEST_BYTES
    )
    manifest = _read_pinned_json(
        manifest_path,
        expected_sha256=expected_manifest_file_sha256,
        label="sidecar manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    manifest = _closed_dict(manifest, _MANIFEST_FIELDS, label="sidecar manifest")
    if (
        manifest["schema_version"] != SIDECAR_SCHEMA
        or manifest["method"] != METHOD
        or manifest["authority"] != AUTHORITY
        or manifest["complete"] is not True
        or manifest["exploratory_only"] is not True
        or manifest["formal_training_authorized"] is not False
        or manifest["scientific_claim_authorized"] is not False
        or manifest["optimizer_updates"] != MAX_UPDATES
        or manifest["world_size"] != WORLD_SIZE
        or manifest["dp_size"] != DP_SIZE
        or manifest["sp_size"] != SP_SIZE
        or manifest["gradient_accumulation"] != GRADIENT_ACCUMULATION
    ):
        fail("sidecar manifest safety/topology envelope differs")
    declared_manifest_digest = _sha256(
        manifest["manifest_digest"], label="sidecar manifest digest", authority=True
    )
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_digest")
    if object_sha256(unsigned_manifest) != declared_manifest_digest:
        fail("sidecar manifest canonical self-digest differs")
    for field, expected, label in (
        (
            "teacher_authority_file_sha256",
            expected_teacher_authority_file_sha256,
            "teacher authority file",
        ),
        (
            "teacher_authority_sha256",
            expected_teacher_authority_sha256,
            "teacher authority",
        ),
        (
            "classification_authority_sha256",
            expected_classification_authority_sha256,
            "classification authority",
        ),
        (
            "predictor_source_sha256",
            expected_predictor_source_sha256,
            "predictor source",
        ),
        (
            "distillation_source_sha256",
            expected_distillation_source_sha256,
            "distillation source",
        ),
        (
            "renderer_runner_source_sha256",
            expected_renderer_runner_source_sha256,
            "0817 renderer runner source",
        ),
        (
            "v2_runner_source_sha256",
            expected_v2_runner_source_sha256,
            "V2 runner source",
        ),
        (
            "schedule_source_sha256",
            expected_schedule_source_sha256,
            "schedule source",
        ),
        (
            "packed_core_source_sha256",
            expected_packed_core_source_sha256,
            "packed core source",
        ),
        (
            "runtime_source_sha256",
            expected_runtime_source_sha256,
            "distributed runtime source",
        ),
        (
            "legacy_loader_source_sha256",
            expected_legacy_loader_source_sha256,
            "legacy loader source",
        ),
        (
            "world8_adapter_source_sha256",
            expected_world8_adapter_source_sha256,
            "WORLD8 adapter source",
        ),
        (
            "inference_sigma_source_sha256",
            expected_inference_sigma_source_sha256,
            "inference sigma source",
        ),
        (
            "renderer_release_manifest_sha256",
            expected_renderer_release_manifest_sha256,
            "0817 release manifest",
        ),
    ):
        if manifest[field] != expected:
            fail(f"manifest {label} SHA-256 differs from external pin")

    root = manifest_path.parent.resolve(strict=True)
    authority_path = _relative_member(
        root, manifest["teacher_authority_file"], label="teacher authority file"
    )
    if file_sha256(authority_path) != expected_teacher_authority_file_sha256:
        fail("teacher authority file SHA-256 differs")
    teacher_authority = _read_pinned_json(
        authority_path,
        expected_sha256=expected_teacher_authority_file_sha256,
        label="teacher authority file",
    )
    # The distillation validator performs the closed authority schema audit.
    # This early equality prevents the candidate manifest from substituting a
    # different internally self-consistent authority before that validator.
    if teacher_authority.get("authority_digest") != expected_teacher_authority_sha256:
        fail("teacher authority object differs from the external authority pin")
    renderer_release_manifest_path = _plain_absolute_file(
        renderer_release_manifest_path,
        label="0817 renderer release manifest",
        max_bytes=1024 * 1024,
    )
    import train_action_edit_large_lora_0817_v1 as renderer_v1

    try:
        renderer_release_closure = renderer_v1.validate_release_manifest(
            renderer_release_manifest_path,
            expected_sha256=expected_renderer_release_manifest_sha256,
        )
    except Exception as error:
        raise ActionAnchorV2CanaryError(
            f"0817 renderer release closure differs: {error}"
        ) from error

    raw_records = manifest["records"]
    if type(raw_records) is not list or len(raw_records) != GLOBAL_RECORDS:
        fail(f"sidecar manifest requires exactly {GLOBAL_RECORDS} logical records")
    records: list[FrozenRecordV2] = []
    seen_row_ids: set[str] = set()
    for logical, raw_record in enumerate(raw_records):
        record = _closed_dict(
            raw_record, _RECORD_FIELDS, label=f"sidecar record[{logical}]"
        )
        if record["logical_record"] != logical:
            fail("sidecar logical records must be canonical 0..15 order")
        if (
            type(record["dataset_iid"]) is not str
            or not record["dataset_iid"]
            or record["dataset_iid"] != record["dataset_iid"].strip()
            or type(record["dataset_row_index"]) is not int
            or record["dataset_row_index"] < 0
        ):
            fail(f"sidecar record[{logical}] dataset binding differs")
        row_id = _sha256(
            record["row_id"], label=f"sidecar record[{logical}] row ID", authority=True
        )
        if row_id in seen_row_ids:
            fail("sidecar logical records require distinct externally bound row IDs")
        seen_row_ids.add(row_id)
        for field in (
            "source_sha256",
            "target_sha256",
            "instruction_sha256",
            "source_mode_fp32le_sha256",
            "target_mode_fp32le_sha256",
            "source_mode_tensor_sha256",
            "target_mode_tensor_sha256",
            "renderer_text_tensor_sha256",
            "instruction_tokens_tensor_sha256",
        ):
            _sha256(
                record[field],
                label=f"sidecar record[{logical}] {field}",
                authority=True,
            )
        shapes: list[tuple[int, ...]] = []
        for field in ("source_mode_shape", "target_mode_shape"):
            raw_shape = record[field]
            if (
                type(raw_shape) is not list
                or len(raw_shape) != 5
                or any(type(item) is not int or item <= 0 for item in raw_shape)
            ):
                fail(f"sidecar record[{logical}] {field} differs")
            shape = tuple(raw_shape)
            if (
                shape[:3] != (1, 16, PHASE_COUNT)
                or shape[3] % 2
                or shape[4] % 2
            ):
                fail(f"sidecar record[{logical}] {field} geometry differs")
            shapes.append(shape)
        if shapes[0] != shapes[1]:
            fail(f"sidecar record[{logical}] source/target mode shapes differ")
        mode_bytes = math.prod(shapes[0]) * 4
        if mode_bytes > 2 * 1024**3:
            fail(f"sidecar record[{logical}] normalized modes exceed size limit")
        source_media_path = _validate_leaf_file(
            root,
            record,
            path_field="source_media_file",
            sha_field="source_sha256",
            label=f"sidecar record[{logical}] original source media",
            max_bytes=16 * 1024**3,
        )
        target_media_path = _validate_leaf_file(
            root,
            record,
            path_field="target_media_file",
            sha_field="target_sha256",
            label=f"sidecar record[{logical}] original target media",
            max_bytes=16 * 1024**3,
        )
        instruction_path = _validate_leaf_file(
            root,
            record,
            path_field="instruction_file",
            sha_field="instruction_sha256",
            label=f"sidecar record[{logical}] instruction UTF-8 bytes",
            max_bytes=1024 * 1024,
        )
        try:
            instruction = instruction_path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise ActionAnchorV2CanaryError(
                f"sidecar record[{logical}] instruction is not UTF-8"
            ) from error
        if not instruction or instruction != instruction.strip() or "\x00" in instruction:
            fail(f"sidecar record[{logical}] instruction text differs")
        source_mode_path = _validate_leaf_file(
            root,
            record,
            path_field="source_mode_fp32le_file",
            sha_field="source_mode_fp32le_sha256",
            label=f"sidecar record[{logical}] normalized source mode",
            exact_bytes=mode_bytes,
            max_bytes=2 * 1024**3,
        )
        target_mode_path = _validate_leaf_file(
            root,
            record,
            path_field="target_mode_fp32le_file",
            sha_field="target_mode_fp32le_sha256",
            label=f"sidecar record[{logical}] normalized target mode",
            exact_bytes=mode_bytes,
            max_bytes=2 * 1024**3,
        )
        q_y = _validate_q_leaf(
            root, record["q_y"], label=f"sidecar record[{logical}] q_y", anchor=False
        )
        raw_anchors = record["anchors"]
        if (
            type(raw_anchors) is not list
            or not raw_anchors
            or len(raw_anchors) > MAX_ANCHORS_PER_RECORD
        ):
            fail(
                f"sidecar record[{logical}] needs 1..{MAX_ANCHORS_PER_RECORD} anchors"
            )
        anchors = tuple(
            _validate_q_leaf(
                root,
                raw_anchor,
                label=f"sidecar record[{logical}] anchor[{index}]",
                anchor=True,
            )
            for index, raw_anchor in enumerate(raw_anchors)
        )
        q_y_item = q_y.q_receipt.get("items")
        if type(q_y_item) is not list or len(q_y_item) != 1:
            fail(f"sidecar record[{logical}] q_y must be a batch-one receipt")
        bound = q_y_item[0]
        if type(bound) is not dict or any(
            bound.get(name) != record[name]
            for name in ("row_id", "source_sha256", "instruction_sha256")
        ) or bound.get("endpoint_sha256") != record["target_sha256"]:
            fail(f"sidecar record[{logical}] q_y dataset binding differs")
        for anchor_index, anchor_leaf in enumerate(anchors):
            anchor_items = anchor_leaf.q_receipt.get("items")
            compatibility = anchor_leaf.compatibility_receipt
            if type(anchor_items) is not list or len(anchor_items) != 1:
                fail(
                    f"sidecar record[{logical}] anchor[{anchor_index}] must be batch one"
                )
            anchor_item = anchor_items[0]
            if type(anchor_item) is not dict or any(
                anchor_item.get(name) != record[name]
                for name in ("row_id", "source_sha256", "instruction_sha256")
            ):
                fail(f"sidecar record[{logical}] anchor dataset binding differs")
            if (
                type(compatibility) is not dict
                or compatibility.get("q_y_receipt_digest")
                != q_y.q_receipt.get("receipt_digest")
                or compatibility.get("q_anchor_receipt_digest")
                != anchor_leaf.q_receipt.get("receipt_digest")
                or compatibility.get("classification_authority_sha256")
                != expected_classification_authority_sha256
            ):
                fail(f"sidecar record[{logical}] anchor compatibility binding differs")
        records.append(
            FrozenRecordV2(
                logical_record=logical,
                dataset_iid=record["dataset_iid"],
                dataset_row_index=record["dataset_row_index"],
                row_id=row_id,
                source_media_path=source_media_path,
                target_media_path=target_media_path,
                source_sha256=record["source_sha256"],
                target_sha256=record["target_sha256"],
                instruction_path=instruction_path,
                instruction_sha256=record["instruction_sha256"],
                source_mode_path=source_mode_path,
                source_mode_fp32le_sha256=record["source_mode_fp32le_sha256"],
                source_mode_shape=shapes[0],
                target_mode_path=target_mode_path,
                target_mode_fp32le_sha256=record["target_mode_fp32le_sha256"],
                target_mode_shape=shapes[1],
                source_mode_tensor_sha256=record["source_mode_tensor_sha256"],
                target_mode_tensor_sha256=record["target_mode_tensor_sha256"],
                renderer_text_tensor_sha256=record[
                    "renderer_text_tensor_sha256"
                ],
                instruction_tokens_tensor_sha256=record[
                    "instruction_tokens_tensor_sha256"
                ],
                q_y=q_y,
                anchors=anchors,
            )
        )
    for source_name, expected, label in (
        (
            Path(__file__).name,
            expected_v2_runner_source_sha256,
            "V2 action-anchor runner",
        ),
        (
            PREDICTOR_SOURCE_NAME,
            expected_predictor_source_sha256,
            "ActionPlanPredictorV1",
        ),
        (
            DISTILLATION_SOURCE_NAME,
            expected_distillation_source_sha256,
            "action-anchor distillation",
        ),
        (
            RENDERER_RUNNER_SOURCE_NAME,
            expected_renderer_runner_source_sha256,
            "0817 renderer runner",
        ),
        (SCHEDULE_SOURCE_NAME, expected_schedule_source_sha256, "two-update schedule"),
        (PACKED_CORE_SOURCE_NAME, expected_packed_core_source_sha256, "packed core"),
        (RUNTIME_SOURCE_NAME, expected_runtime_source_sha256, "distributed runtime"),
        (LEGACY_LOADER_SOURCE_NAME, expected_legacy_loader_source_sha256, "legacy loader"),
        (
            WORLD8_ADAPTER_SOURCE_NAME,
            expected_world8_adapter_source_sha256,
            "WORLD8 adapter",
        ),
        (
            INFERENCE_SIGMA_SOURCE_NAME,
            expected_inference_sigma_source_sha256,
            "inference sigma schedule",
        ),
    ):
        source = METHOD_ROOT / source_name
        if (
            not source.is_file()
            or source.is_symlink()
            or file_sha256(source) != expected
        ):
            fail(f"executed {label} source differs from external pin")
    lease = object()
    preflight = FrozenSidecarPreflightV2(
        schema_version=PREFLIGHT_SCHEMA,
        manifest_path=manifest_path,
        manifest_file_sha256=expected_manifest_file_sha256,
        manifest_digest=declared_manifest_digest,
        root=root,
        teacher_authority=teacher_authority,
        teacher_authority_file_sha256=expected_teacher_authority_file_sha256,
        teacher_authority_sha256=expected_teacher_authority_sha256,
        classification_authority_sha256=expected_classification_authority_sha256,
        predictor_source_sha256=expected_predictor_source_sha256,
        distillation_source_sha256=expected_distillation_source_sha256,
        renderer_runner_source_sha256=expected_renderer_runner_source_sha256,
        v2_runner_source_sha256=expected_v2_runner_source_sha256,
        schedule_source_sha256=expected_schedule_source_sha256,
        packed_core_source_sha256=expected_packed_core_source_sha256,
        runtime_source_sha256=expected_runtime_source_sha256,
        legacy_loader_source_sha256=expected_legacy_loader_source_sha256,
        world8_adapter_source_sha256=expected_world8_adapter_source_sha256,
        inference_sigma_source_sha256=expected_inference_sigma_source_sha256,
        renderer_release_manifest_path=renderer_release_manifest_path,
        renderer_release_manifest_sha256=(
            expected_renderer_release_manifest_sha256
        ),
        renderer_release_closure=renderer_release_closure,
        records=tuple(records),
        all_leaf_files_verified_before_optimizer=True,
        _lease=lease,
    )
    with _PREFLIGHT_REGISTRY_LOCK:
        _PREFLIGHT_REGISTRY[preflight] = (lease, _preflight_signature_v2(preflight))
    return preflight


def _require_preflight_capability_v2(preflight: Any) -> FrozenSidecarPreflightV2:
    if type(preflight) is not FrozenSidecarPreflightV2:
        fail("frozen sidecar preflight capability is absent or forged")
    with _PREFLIGHT_REGISTRY_LOCK:
        registered = _PREFLIGHT_REGISTRY.get(preflight)
        if (
            type(registered) is not tuple
            or len(registered) != 2
            or registered[0] is not preflight._lease
            or registered[1] != _preflight_signature_v2(preflight)
        ):
            fail("frozen sidecar preflight capability is absent or forged")
    return preflight


def _load_fp32le(
    path: Path, *, count: int, expected_sha256: str, device: Any
) -> Any:
    import torch

    expected = _sha256(
        expected_sha256, label="raw FP32 sidecar expected SHA-256", authority=True
    )
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    named = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )
    if (
        identity(before) != identity(after)
        or identity(before) != identity(named)
        or len(payload) != count * 4
        or hashlib.sha256(payload).hexdigest() != expected
    ):
        fail(f"raw FP32 sidecar byte count changed: {path}")
    values = struct.unpack(f"<{count}f", payload)
    tensor = torch.tensor(values, dtype=torch.float32, device=device).contiguous()
    if not bool(torch.isfinite(tensor).all().item()):
        fail(f"raw FP32 sidecar contains non-finite values: {path}")
    return tensor


@dataclass(frozen=True)
class FrozenRuntimePayloadV2:
    """Sidecar-owned media/modes/text before runtime T5 materialization."""

    record: FrozenRecordV2
    source_mode: Any
    target_mode: Any
    instruction: str


def load_frozen_runtime_payload_v2(record: FrozenRecordV2) -> FrozenRuntimePayloadV2:
    """Reopen and authenticate one complete source/target runtime payload."""

    if type(record) is not FrozenRecordV2:
        fail("frozen runtime payload requires one exact manifest record")
    if (
        file_sha256(record.source_media_path) != record.source_sha256
        or file_sha256(record.target_media_path) != record.target_sha256
    ):
        fail("frozen runtime original media changed after preflight")
    try:
        raw_instruction = record.instruction_path.read_bytes()
        instruction = raw_instruction.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ActionAnchorV2CanaryError(
            "frozen runtime instruction changed after preflight"
        ) from error
    if (
        hashlib.sha256(raw_instruction).hexdigest() != record.instruction_sha256
        or not instruction
        or instruction != instruction.strip()
        or "\x00" in instruction
    ):
        fail("frozen runtime instruction changed after preflight")
    source = _load_fp32le(
        record.source_mode_path,
        count=math.prod(record.source_mode_shape),
        expected_sha256=record.source_mode_fp32le_sha256,
        device="cpu",
    ).reshape(record.source_mode_shape)
    target = _load_fp32le(
        record.target_mode_path,
        count=math.prod(record.target_mode_shape),
        expected_sha256=record.target_mode_fp32le_sha256,
        device="cpu",
    ).reshape(record.target_mode_shape)
    if (
        runtime_tensor_sha256_v2(source) != record.source_mode_tensor_sha256
        or runtime_tensor_sha256_v2(target) != record.target_mode_tensor_sha256
    ):
        fail("frozen runtime normalized mode tensor digest differs")
    return FrozenRuntimePayloadV2(
        record=record,
        source_mode=source,
        target_mode=target,
        instruction=instruction,
    )


@dataclass(frozen=True)
class PreparedTeacherRecordV2:
    record: FrozenRecordV2
    q_y: Any
    q_y_receipt: Mapping[str, Any]
    q_y_qualification_digest: str
    anchors: tuple[Any, ...]
    anchor_qualification_digests: tuple[tuple[str, ...], ...]
    compatibility_decision_digests: tuple[str, ...]
    predictor_source_sha256: str
    distillation_source_sha256: str
    renderer_runner_source_sha256: str
    v2_runner_source_sha256: str
    teacher_authority_sha256: str
    classification_authority_sha256: str
    contrastive_positive_pair_count: int
    contrastive_negative_pair_count: int
    excluded_pair_count: int


@dataclass(frozen=True)
class BoundRuntimeRecordV2:
    """Actual media/latent/text payload bound to one frozen teacher record."""

    teacher: PreparedTeacherRecordV2
    source_media_path: Path
    target_media_path: Path
    source_mode: Any
    target_mode: Any
    instruction: str
    text_lens: Any
    text_embs: Any
    instruction_tokens: Any


def bind_runtime_record_v2(
    teacher: PreparedTeacherRecordV2,
    *,
    logical_record: int,
    dataset_iid: str,
    dataset_row_index: int,
    source_media_path: Path,
    target_media_path: Path,
    source_mode: Any,
    target_mode: Any,
    instruction: str,
    text_lens: Any,
    text_embs: Any,
    instruction_tokens: Any,
) -> BoundRuntimeRecordV2:
    """Hash actual bytes/tensors/text; declared dataset hashes are insufficient."""

    import torch

    if type(teacher) is not PreparedTeacherRecordV2:
        fail("runtime binding requires one fully validated teacher capability")
    record = teacher.record
    if (
        logical_record != record.logical_record
        or dataset_iid != record.dataset_iid
        or dataset_row_index != record.dataset_row_index
    ):
        fail("runtime dataset coordinate differs from frozen teacher record")
    source_path = _plain_absolute_file(
        source_media_path, label="runtime source media", max_bytes=16 * 1024**3
    )
    target_path = _plain_absolute_file(
        target_media_path, label="runtime target media", max_bytes=16 * 1024**3
    )
    if (
        file_sha256(source_path) != record.source_sha256
        or file_sha256(target_path) != record.target_sha256
    ):
        fail("runtime source/target original bytes differ from frozen sidecar")
    for value, expected, label in (
        (source_mode, record.source_mode_tensor_sha256, "source"),
        (target_mode, record.target_mode_tensor_sha256, "target"),
    ):
        if (
            type(value) is not torch.Tensor
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or not value.is_contiguous()
            or value.requires_grad
            or value.ndim != 5
            or tuple(map(int, value.shape[:3])) != (1, 16, PHASE_COUNT)
            or not bool(torch.isfinite(value).all().item())
            or runtime_tensor_sha256_v2(value) != expected
        ):
            fail(f"runtime normalized {label} mode differs from frozen tensor pin")
    if (
        type(instruction) is not str
        or not instruction
        or instruction != instruction.strip()
        or hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        != record.instruction_sha256
    ):
        fail("runtime instruction bytes differ from frozen sidecar")
    if type(text_lens) is torch.Tensor:
        renderer_lengths = [int(item) for item in text_lens.reshape(-1)]
    elif type(text_lens) in (list, tuple):
        renderer_lengths = [int(item) for item in text_lens]
    else:
        renderer_lengths = []
    if (
        type(text_embs) is not torch.Tensor
        or type(instruction_tokens) is not torch.Tensor
        or text_embs.ndim != 3
        or tuple(map(int, text_embs.shape)) != (1, 512, 4096)
        or instruction_tokens.ndim != 3
        or int(instruction_tokens.shape[0]) != 1
        or int(instruction_tokens.shape[2]) != 4096
        or not 0 < int(instruction_tokens.shape[1]) <= 512
        or text_embs.dtype != instruction_tokens.dtype
        or text_embs.device != instruction_tokens.device
        or text_embs.requires_grad
        or instruction_tokens.requires_grad
        or not text_embs.is_contiguous()
        or not instruction_tokens.is_contiguous()
        or not torch.equal(
            instruction_tokens,
            text_embs[:, : int(instruction_tokens.shape[1]), :],
        )
        or not bool(torch.isfinite(text_embs).all().item())
        or renderer_lengths != [512]
        or runtime_tensor_sha256_v2(text_embs)
        != record.renderer_text_tensor_sha256
        or runtime_tensor_sha256_v2(instruction_tokens)
        != record.instruction_tokens_tensor_sha256
    ):
        fail("runtime frozen T5/text view closure differs")
    return BoundRuntimeRecordV2(
        teacher=teacher,
        source_media_path=source_path,
        target_media_path=target_path,
        source_mode=source_mode,
        target_mode=target_mode,
        instruction=instruction,
        text_lens=text_lens,
        text_embs=text_embs,
        instruction_tokens=instruction_tokens,
    )


def _verify_executed_sources_v2(preflight: FrozenSidecarPreflightV2) -> None:
    for source_name, expected, label in (
        (
            Path(__file__).name,
            preflight.v2_runner_source_sha256,
            "V2 action-anchor runner",
        ),
        (
            PREDICTOR_SOURCE_NAME,
            preflight.predictor_source_sha256,
            "ActionPlanPredictorV1",
        ),
        (
            DISTILLATION_SOURCE_NAME,
            preflight.distillation_source_sha256,
            "action-anchor distillation",
        ),
        (
            RENDERER_RUNNER_SOURCE_NAME,
            preflight.renderer_runner_source_sha256,
            "0817 renderer runner",
        ),
        (SCHEDULE_SOURCE_NAME, preflight.schedule_source_sha256, "two-update schedule"),
        (PACKED_CORE_SOURCE_NAME, preflight.packed_core_source_sha256, "packed core"),
        (RUNTIME_SOURCE_NAME, preflight.runtime_source_sha256, "distributed runtime"),
        (LEGACY_LOADER_SOURCE_NAME, preflight.legacy_loader_source_sha256, "legacy loader"),
        (
            WORLD8_ADAPTER_SOURCE_NAME,
            preflight.world8_adapter_source_sha256,
            "WORLD8 adapter",
        ),
        (
            INFERENCE_SIGMA_SOURCE_NAME,
            preflight.inference_sigma_source_sha256,
            "inference sigma schedule",
        ),
    ):
        if file_sha256(METHOD_ROOT / source_name) != expected:
            fail(f"executed {label} source changed after preflight")
    import train_action_edit_large_lora_0817_v1 as renderer_v1

    try:
        closure = renderer_v1.validate_release_manifest(
            preflight.renderer_release_manifest_path,
            expected_sha256=preflight.renderer_release_manifest_sha256,
        )
    except Exception as error:
        raise ActionAnchorV2CanaryError(
            f"0817 renderer release closure changed after preflight: {error}"
        ) from error
    if closure != preflight.renderer_release_closure:
        fail("0817 renderer release closure changed after preflight")


def materialize_and_validate_teachers_v2(
    preflight: FrozenSidecarPreflightV2, *, device: Any
) -> tuple[PreparedTeacherRecordV2, ...]:
    """Fully validate all teacher tensors/receipts before optimizer creation."""

    preflight = _require_preflight_capability_v2(preflight)
    _verify_executed_sources_v2(preflight)
    import action_anchor_distillation_v1 as distillation_v1
    import action_plan_predictor_v1 as predictor_v1

    _require_exact_import_v2(
        predictor_v1, PREDICTOR_SOURCE_NAME, preflight.predictor_source_sha256
    )
    _require_exact_import_v2(
        distillation_v1,
        DISTILLATION_SOURCE_NAME,
        preflight.distillation_source_sha256,
    )
    ActionPlanOutput = predictor_v1.ActionPlanOutput
    RoutedAnchorV1 = distillation_v1.RoutedAnchorV1
    validate_compatibility_receipt_v1 = (
        distillation_v1.validate_compatibility_receipt_v1
    )
    validate_q_receipt_v1 = distillation_v1.validate_q_receipt_v1

    prepared: list[PreparedTeacherRecordV2] = []
    for row in preflight.records:
        q_y = ActionPlanOutput(
            phase_tokens=_load_fp32le(
                row.q_y.phase_path,
                count=PHASE_COUNT * ACTION_WIDTH,
                expected_sha256=row.q_y.phase_sha256,
                device=device,
            ).reshape(1, PHASE_COUNT, ACTION_WIDTH),
            global_token=_load_fp32le(
                row.q_y.global_path,
                count=ACTION_WIDTH,
                expected_sha256=row.q_y.global_sha256,
                device=device,
            ).reshape(1, ACTION_WIDTH),
        )
        validate_q_receipt_v1(
            row.q_y.q_receipt,
            plan=q_y,
            expected_teacher_authority_sha256=preflight.teacher_authority_sha256,
            expected_qualification_receipt_digests=[
                row.q_y.qualification_receipt_digest
            ],
        )
        routed: list[Any] = []
        qualification_pins: list[tuple[str, ...]] = []
        decision_pins: list[str] = []
        positive_pairs = 0
        negative_pairs = 0
        excluded_pairs = 0
        for anchor in row.anchors:
            anchor_plan = ActionPlanOutput(
                phase_tokens=_load_fp32le(
                    anchor.phase_path,
                    count=PHASE_COUNT * ACTION_WIDTH,
                    expected_sha256=anchor.phase_sha256,
                    device=device,
                ).reshape(1, PHASE_COUNT, ACTION_WIDTH),
                global_token=_load_fp32le(
                    anchor.global_path,
                    count=ACTION_WIDTH,
                    expected_sha256=anchor.global_sha256,
                    device=device,
                ).reshape(1, ACTION_WIDTH),
            )
            validate_q_receipt_v1(
                anchor.q_receipt,
                plan=anchor_plan,
                expected_teacher_authority_sha256=preflight.teacher_authority_sha256,
                expected_qualification_receipt_digests=[
                    anchor.qualification_receipt_digest
                ],
            )
            if anchor.compatibility_receipt is None:
                fail("validated anchor lost its compatibility receipt")
            checked_compatibility = validate_compatibility_receipt_v1(
                anchor.compatibility_receipt,
                q_y_receipt=row.q_y.q_receipt,
                q_anchor_receipt=anchor.q_receipt,
                expected_teacher_authority_sha256=preflight.teacher_authority_sha256,
                expected_classification_authority_sha256=(
                    preflight.classification_authority_sha256
                ),
                expected_q_y_qualification_receipt_digests=[
                    row.q_y.qualification_receipt_digest
                ],
                expected_q_anchor_qualification_receipt_digests=[
                    anchor.qualification_receipt_digest
                ],
                expected_decision_receipt_digest=(
                    anchor.compatibility_decision_receipt_digest
                ),
            )
            for item in checked_compatibility["items"]:
                role = (item["training_use"], item["contrastive_role"])
                if role == ("contrastive-only", "positive"):
                    positive_pairs += 1
                elif role == ("contrastive-only", "negative"):
                    negative_pairs += 1
                elif role == ("excluded", "none"):
                    excluded_pairs += 1
                else:  # pragma: no cover - stable validator owns this closure.
                    fail("validated anchor returned an unknown training route")
            routed.append(
                RoutedAnchorV1(
                    plan=anchor_plan,
                    q_receipt=anchor.q_receipt,
                    compatibility_receipt=anchor.compatibility_receipt,
                )
            )
            qualification_pins.append((anchor.qualification_receipt_digest,))
            if anchor.compatibility_decision_receipt_digest is None:
                fail("validated anchor lost its compatibility decision pin")
            decision_pins.append(anchor.compatibility_decision_receipt_digest)
        _require_active_contrastive_pairs_v2(
            logical_record=row.logical_record,
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs,
            excluded_pairs=excluded_pairs,
        )
        prepared.append(
            PreparedTeacherRecordV2(
                record=row,
                q_y=q_y,
                q_y_receipt=row.q_y.q_receipt,
                q_y_qualification_digest=row.q_y.qualification_receipt_digest,
                anchors=tuple(routed),
                anchor_qualification_digests=tuple(qualification_pins),
                compatibility_decision_digests=tuple(decision_pins),
                predictor_source_sha256=preflight.predictor_source_sha256,
                distillation_source_sha256=preflight.distillation_source_sha256,
                renderer_runner_source_sha256=(
                    preflight.renderer_runner_source_sha256
                ),
                v2_runner_source_sha256=preflight.v2_runner_source_sha256,
                teacher_authority_sha256=preflight.teacher_authority_sha256,
                classification_authority_sha256=(
                    preflight.classification_authority_sha256
                ),
                contrastive_positive_pair_count=positive_pairs,
                contrastive_negative_pair_count=negative_pairs,
                excluded_pair_count=excluded_pairs,
            )
        )
    return tuple(prepared)


def _require_active_contrastive_pairs_v2(
    *,
    logical_record: int,
    positive_pairs: int,
    negative_pairs: int,
    excluded_pairs: int,
) -> None:
    if (
        type(logical_record) is not int
        or not 0 <= logical_record < GLOBAL_RECORDS
        or any(
            type(value) is not int or value < 0
            for value in (positive_pairs, negative_pairs, excluded_pairs)
        )
        or negative_pairs < 1
    ):
        fail(
            f"qualified row {logical_record} has no active q_anchor negative; "
            "InfoNCE would be identically zero"
        )


@dataclass(frozen=True)
class FP32ActionInjectionRouteV2:
    """One FP32 student plan plus its differentiable renderer-dtype view."""

    q_pred_fp32: Any
    renderer_route: Any
    fp32_to_renderer_cast: bool


def prepare_fp32_action_injection_route_v2(
    *,
    conditioner: Any,
    embedded: Any,
    packed: Mapping[str, Any],
    instruction_tokens: Any,
) -> Any:
    """Predict FP32 q from complete pre-SP source, then bind target ownership."""

    import torch
    from action_plan_predictor_v1 import (
        ActionPlanOutput,
        certify_closed_target_suffix_route,
    )

    phases, height, width = packed["patch_grid"]
    source_native = embedded[:, : packed["source_tokens"], :].reshape(
        1, phases, height, width, HIDDEN_WIDTH
    )
    target_native = embedded[:, packed["source_tokens"] :, :].reshape(
        1, phases, height, width, HIDDEN_WIDTH
    )
    if (
        phases != PHASE_COUNT
        or source_native.numel() != packed["source_tokens"] * HIDDEN_WIDTH
        or target_native.numel() != packed["target_tokens"] * HIDDEN_WIDTH
        or instruction_tokens.device != source_native.device
    ):
        fail("V2 complete pre-SP source/target/instruction closure differs")
    ownership = certify_closed_target_suffix_route(
        target_native,
        source_prefix_tokens=packed["source_tokens"],
        packed_total_tokens=packed["total_tokens"],
        audit_finite=True,
    )
    # Predictor trainables and q ABI are FP32.  One differentiable plan view is
    # cast at this audited boundary to satisfy the renderer route's native
    # dtype certificate; the FP32 plan remains the distillation student.
    source_fp32 = source_native.to(dtype=torch.float32).contiguous()
    instruction_fp32 = instruction_tokens.to(dtype=torch.float32).contiguous()
    plan = conditioner.predictor(source_fp32, instruction_fp32)
    if (
        plan.phase_tokens.dtype != torch.float32
        or plan.global_token.dtype != torch.float32
        or not plan.phase_tokens.is_contiguous()
        or not plan.global_token.is_contiguous()
    ):
        fail("V2 q_pred is not a closed FP32 action plan")
    # The stable injection ABI authenticates plan dtype against target hidden
    # dtype.  Keep the FP32 student as the unique distillation value and make
    # one differentiable cast view solely for Bernini's exact30 hook.
    renderer_plan = ActionPlanOutput(
        phase_tokens=plan.phase_tokens.to(dtype=target_native.dtype).contiguous(),
        global_token=plan.global_token.to(dtype=target_native.dtype).contiguous(),
    )
    renderer_route = conditioner.injection.bind_route(
        renderer_plan, ownership, audit_finite=True
    )
    if (
        renderer_route.ownership.digest != ownership.digest
        or renderer_route.plan.phase_tokens.dtype != target_native.dtype
        or renderer_route.plan.global_token.dtype != target_native.dtype
    ):
        fail("V2 renderer-dtype plan route differs after explicit cast")
    return FP32ActionInjectionRouteV2(
        q_pred_fp32=plan,
        renderer_route=renderer_route,
        fp32_to_renderer_cast=True,
    )


def _q_pred_bindings(teacher: PreparedTeacherRecordV2) -> list[dict[str, Any]]:
    items = teacher.q_y_receipt.get("items")
    if type(items) is not list or len(items) != 1 or type(items[0]) is not dict:
        fail("qualified q_y receipt lost its batch-one binding")
    target = items[0]
    return [
        {
            "row_id": target["row_id"],
            "source_sha256": target["source_sha256"],
            "instruction_sha256": target["instruction_sha256"],
            "endpoint_sha256": None,
            "semantics": target["semantics"],
            "teacher_evidence": None,
        }
    ]


@dataclass(frozen=True)
class ActionAnchorObjectiveV2:
    schema_version: str
    total: Any
    action_only: Any
    flow_preservation: Any
    smooth_l1: Any
    cosine: Any
    infonce: Any
    q_pred_receipt_digest: str
    q_y_receipt_digest: str
    point_pair_count: int
    contrastive_positive_pair_count: int
    contrastive_negative_pair_count: int
    excluded_pair_count: int


def action_anchor_objective_v2(
    *,
    action_route: Any,
    teacher: PreparedTeacherRecordV2,
    flow_preservation_loss: Any,
    predictor_source_sha256: str,
    teacher_authority_sha256: str,
    classification_authority_sha256: str,
    smooth_l1_weight: float = 1.0,
    cosine_weight: float = 1.0,
    infonce_weight: float = 1.0,
    flow_weight: float = 1.0,
    temperature: float = 0.07,
) -> ActionAnchorObjectiveV2:
    """Bind q_pred and compute point(q_y)+InfoNCE(q_anchor)+real flow."""

    if (
        predictor_source_sha256 != teacher.predictor_source_sha256
        or teacher_authority_sha256 != teacher.teacher_authority_sha256
        or classification_authority_sha256
        != teacher.classification_authority_sha256
        or file_sha256(METHOD_ROOT / PREDICTOR_SOURCE_NAME)
        != teacher.predictor_source_sha256
        or file_sha256(METHOD_ROOT / DISTILLATION_SOURCE_NAME)
        != teacher.distillation_source_sha256
    ):
        fail("V2 objective source identity differs from frozen preflight pins")
    for name, value in (
        ("smooth_l1_weight", smooth_l1_weight),
        ("cosine_weight", cosine_weight),
        ("infonce_weight", infonce_weight),
        ("flow_weight", flow_weight),
        ("temperature", temperature),
    ):
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            fail(f"V2 {name} must be one finite number")
    if float(flow_weight) <= 0:
        fail("V2 real renderer flow preservation weight must be positive")
    import torch
    import action_anchor_distillation_v1 as distillation_v1
    import action_plan_predictor_v1 as predictor_v1

    _require_exact_import_v2(
        predictor_v1, PREDICTOR_SOURCE_NAME, teacher.predictor_source_sha256
    )
    _require_exact_import_v2(
        distillation_v1,
        DISTILLATION_SOURCE_NAME,
        teacher.distillation_source_sha256,
    )
    DistillationLossConfigV1 = distillation_v1.DistillationLossConfigV1
    action_anchor_distillation_loss_v1 = (
        distillation_v1.action_anchor_distillation_loss_v1
    )
    build_q_receipt_v1 = distillation_v1.build_q_receipt_v1

    if type(action_route) is not FP32ActionInjectionRouteV2:
        fail("V2 objective requires the exact FP32/renderer dual route")
    plan = action_route.q_pred_fp32
    if (
        plan.phase_tokens.dtype != torch.float32
        or plan.global_token.dtype != torch.float32
        or not plan.phase_tokens.requires_grad
        or not plan.global_token.requires_grad
    ):
        fail("V2 objective requires autograd-connected FP32 q_pred")
    if (
        type(flow_preservation_loss) is not torch.Tensor
        or flow_preservation_loss.numel() != 1
        or not flow_preservation_loss.requires_grad
        or not bool(torch.isfinite(flow_preservation_loss.detach()).item())
    ):
        fail("V2 objective requires one finite real renderer flow loss")
    q_pred_receipt = build_q_receipt_v1(
        q_kind="q_pred",
        plan=plan,
        bindings=_q_pred_bindings(teacher),
        producer_artifact_sha256=predictor_source_sha256,
    )
    config = DistillationLossConfigV1(
        smooth_l1_weight=float(smooth_l1_weight),
        cosine_weight=float(cosine_weight),
        infonce_weight=float(infonce_weight),
        preservation_weight=float(flow_weight),
        temperature=float(temperature),
    )
    loss = action_anchor_distillation_loss_v1(
        q_pred=plan,
        q_y=teacher.q_y,
        q_pred_receipt=q_pred_receipt,
        q_y_receipt=teacher.q_y_receipt,
        expected_teacher_authority_sha256=teacher_authority_sha256,
        expected_classification_authority_sha256=classification_authority_sha256,
        expected_q_y_qualification_receipt_digests=[
            teacher.q_y_qualification_digest
        ],
        expected_anchor_qualification_receipt_digests=[
            list(pins) for pins in teacher.anchor_qualification_digests
        ],
        expected_compatibility_decision_receipt_digests=list(
            teacher.compatibility_decision_digests
        ),
        anchors=teacher.anchors,
        preservation_loss=flow_preservation_loss,
        config=config,
    )
    if (
        loss.point_pair_count != 1
        or loss.contrastive_positive_pair_count
        != teacher.contrastive_positive_pair_count
        or loss.contrastive_negative_pair_count
        != teacher.contrastive_negative_pair_count
        or loss.excluded_pair_count != teacher.excluded_pair_count
        or loss.contrastive_negative_pair_count < 1
    ):
        fail("V2 active q_anchor InfoNCE pair-count closure differs")
    action_only = (
        float(smooth_l1_weight) * loss.smooth_l1
        + float(cosine_weight) * loss.cosine
        + float(infonce_weight) * loss.infonce
    )
    # Recompose exactly: the distillation module remains the authority for the
    # total; this equality catches accidental anchor point-loss additions.
    recomposed = action_only + float(flow_weight) * loss.preservation
    if not bool(torch.allclose(loss.total.detach(), recomposed.detach(), rtol=0, atol=0)):
        fail("V2 objective decomposition differs from the strict loss contract")
    return ActionAnchorObjectiveV2(
        schema_version=OBJECTIVE_SCHEMA,
        total=loss.total,
        action_only=action_only,
        flow_preservation=loss.preservation,
        smooth_l1=loss.smooth_l1,
        cosine=loss.cosine,
        infonce=loss.infonce,
        q_pred_receipt_digest=q_pred_receipt["receipt_digest"],
        q_y_receipt_digest=teacher.q_y_receipt["receipt_digest"],
        point_pair_count=loss.point_pair_count,
        contrastive_positive_pair_count=loss.contrastive_positive_pair_count,
        contrastive_negative_pair_count=loss.contrastive_negative_pair_count,
        excluded_pair_count=loss.excluded_pair_count,
    )


def _gradient_l2(parameters: Sequence[Any]) -> float:
    total = 0.0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None:
            value = gradient.detach().float()
            total += float(torch_sum_square(value))
    return math.sqrt(total)


def torch_sum_square(value: Any) -> float:
    return float((value * value).sum().item())


def certify_zero_init_exact30_v2(conditioner: Any) -> dict[str, Any]:
    """Certify the step-1 Jacobian split before any optimizer update."""

    import torch

    projections = tuple(conditioner.injection.projections)
    if len(projections) != TRANSFORMER_BLOCKS:
        fail("V2 step1 audit requires exactly 30 injection projections")
    per_block = []
    for index, projection in enumerate(projections):
        parameters = tuple(projection.parameters())
        if not parameters or any(
            parameter.dtype != torch.float32
            or bool(torch.count_nonzero(parameter.detach()).item())
            for parameter in parameters
        ):
            fail(f"V2 injection block {index} is not exact FP32 zero at P0")
        per_block.append(index)
    return {
        "schema_version": STEP1_AUDIT_SCHEMA,
        "p0_exact_zero_injection_blocks": per_block,
        "flow_to_predictor_jacobian": "exact-zero-by-zero-output-projection",
        "flow_to_injection_jacobian": "active",
        "action_loss_to_predictor_jacobian": "active",
        "action_loss_to_injection_jacobian": "disconnected",
        "q_y_unique_point_teacher": True,
        "q_anchor_point_gradient": False,
    }


def audit_step1_gradients_v2(conditioner: Any) -> dict[str, Any]:
    """Check observed combined gradients after the first backward at P0."""

    predictor_parameters = tuple(conditioner.predictor.parameters())
    projections = tuple(conditioner.injection.projections)
    predictor_norm = _gradient_l2(predictor_parameters)
    injection_norms = [_gradient_l2(tuple(block.parameters())) for block in projections]
    if not math.isfinite(predictor_norm) or predictor_norm <= 0:
        fail("step1 action loss did not train ActionPlanPredictorV1")
    if len(injection_norms) != TRANSFORMER_BLOCKS or any(
        not math.isfinite(value) or value <= 0 for value in injection_norms
    ):
        fail("step1 real renderer flow did not train every exact30 injection head")
    return {
        "schema_version": STEP1_AUDIT_SCHEMA,
        "predictor_gradient_l2": predictor_norm,
        "injection_gradient_l2_by_block": injection_norms,
        "predictor_gradient_source_at_p0": "action-distillation-only",
        "injection_gradient_source_at_p0": "renderer-flow-only",
        "verified": True,
    }


@dataclass(frozen=True)
class RenderedMicrobatchV2:
    objective: ActionAnchorObjectiveV2
    local_route_receipt: Mapping[str, Any]
    prediction: Any
    flow_loss: Any


def render_action_anchor_microbatch_v2(
    *,
    base_renderer: Any,
    transformer: Any,
    conditioner: Any,
    runtime_record: BoundRuntimeRecordV2,
    coordinate: Any,
    epsilon: Any,
    rope: Any,
    device: Any,
    sequence_parallel_rank: int,
    torch_checkpoint: Any,
    packed_role_layout: Callable[..., Any],
    loss_kwargs: Optional[Mapping[str, float]] = None,
) -> RenderedMicrobatchV2:
    """Run the genuine Bernini exact30 route and construct the V2 loss graph."""

    if type(runtime_record) is not BoundRuntimeRecordV2:
        fail("V2 renderer requires an exact runtime/teacher binding")
    teacher = runtime_record.teacher
    if (
        file_sha256(METHOD_ROOT / RENDERER_RUNNER_SOURCE_NAME)
        != teacher.renderer_runner_source_sha256
    ):
        fail("0817 renderer runner source changed after teacher preflight")
    import torch
    import train_action_edit_large_lora_0817_v1 as renderer_v1

    if (
        renderer_v1.TRANSFORMER_BLOCKS != TRANSFORMER_BLOCKS
        or renderer_v1.HIDDEN_WIDTH != HIDDEN_WIDTH
        or len(tuple(transformer.blocks)) != TRANSFORMER_BLOCKS
    ):
        fail("executed 0817 Bernini exact30 renderer identity differs")
    if (
        file_sha256(runtime_record.source_media_path)
        != teacher.record.source_sha256
        or file_sha256(runtime_record.target_media_path)
        != teacher.record.target_sha256
        or runtime_tensor_sha256_v2(runtime_record.source_mode)
        != teacher.record.source_mode_tensor_sha256
        or runtime_tensor_sha256_v2(runtime_record.target_mode)
        != teacher.record.target_mode_tensor_sha256
        or runtime_tensor_sha256_v2(runtime_record.text_embs)
        != teacher.record.renderer_text_tensor_sha256
        or runtime_tensor_sha256_v2(runtime_record.instruction_tokens)
        != teacher.record.instruction_tokens_tensor_sha256
        or hashlib.sha256(runtime_record.instruction.encode("utf-8")).hexdigest()
        != teacher.record.instruction_sha256
    ):
        fail("runtime record changed after its frozen teacher binding")
    packed = dict(
        renderer_v1.prepare_paired_flow(
            source=runtime_record.source_mode,
            target=runtime_record.target_mode,
            epsilon=epsilon,
            coordinate=coordinate,
            rope=rope,
            device=device,
        )
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with packed_role_layout(packed["source_tokens"], packed["target_tokens"]):
            embedded = transformer.patch_embedding(packed["input_patches"]).flatten(1).unsqueeze(0)
    if tuple(embedded.shape) != (1, packed["total_tokens"], HIDDEN_WIDTH):
        fail("V2 real renderer packed embedding geometry differs")
    packed["embedded"] = embedded
    action_route = prepare_fp32_action_injection_route_v2(
        conditioner=conditioner,
        embedded=embedded,
        packed=packed,
        instruction_tokens=runtime_record.instruction_tokens,
    )
    local_route = renderer_v1.ActionInjectionRoute(
        source_tokens=packed["source_tokens"],
        target_tokens=packed["target_tokens"],
        sequence_parallel_rank=sequence_parallel_rank,
        sequence_parallel_size=SP_SIZE,
        plan=action_route.renderer_route,
        row_identity=teacher.record.row_id,
    )
    with renderer_v1.activate_action_route(local_route):
        with torch_checkpoint.set_checkpoint_early_stop(False):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = renderer_v1.predict_target(
                    renderer=base_renderer,
                    packed=packed,
                    coordinate=coordinate,
                    text_lens=runtime_record.text_lens,
                    text_embs=runtime_record.text_embs,
                )
                flow_loss = torch.nn.functional.mse_loss(
                    prediction.float(), packed["target_velocity"].float()
                )
            objective = action_anchor_objective_v2(
                action_route=action_route,
                teacher=teacher,
                flow_preservation_loss=flow_loss,
                predictor_source_sha256=teacher.predictor_source_sha256,
                teacher_authority_sha256=teacher.teacher_authority_sha256,
                classification_authority_sha256=(
                    teacher.classification_authority_sha256
                ),
                **dict(loss_kwargs or {}),
            )
            (objective.total / float(GRADIENT_ACCUMULATION)).backward()
        local_route.validate_forward_and_recompute_traversal()
    return RenderedMicrobatchV2(
        objective=objective,
        local_route_receipt=local_route.receipt(),
        prediction=prediction,
        flow_loss=flow_loss,
    )


@dataclass(frozen=True)
class RuntimeRecordInputV2:
    logical_record: int
    dataset_iid: str
    dataset_row_index: int
    source_media_path: Path
    target_media_path: Path
    source_mode: Any
    target_mode: Any
    instruction: str
    text_lens: Any
    text_embs: Any
    instruction_tokens: Any


_EXECUTION_REGISTRY_LOCK = threading.Lock()
_EXECUTION_REGISTRY: dict[Any, Any] = {}


@dataclass(frozen=True)
class _ExecutionSealV2:
    lease: Any
    runtime_object_ids: tuple[int, ...]
    local_input_ids: tuple[int, ...]
    named_parameter_identity: tuple[tuple[str, int], ...]
    inventory_sha256: str
    initial_parameter_sha256: str
    global_contract_sha256: str
    local_runtime_sha256: str
    scalar_contract: tuple[Any, ...]


@dataclass(frozen=True, eq=False)
class PreparedWorld8CanaryV2:
    """Immutable one-shot capability issued after full teacher/data audit."""

    preflight: FrozenSidecarPreflightV2
    model: Any
    base_renderer: Any
    transformer: Any
    conditioner: Any
    hook_handle: Any
    parallel: Any
    distributed: Any
    rope: Any
    device: Any
    local_inputs: tuple[RuntimeRecordInputV2, ...]
    learning_rate: float
    max_grad_norm: float
    seed: int
    loss_items: tuple[tuple[str, float], ...]
    _lease: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class CanaryRunResultV2:
    """Auditable terminal evidence; the canary is deliberately non-resumable."""

    history: tuple[Mapping[str, Any], ...]
    parameter_sha256_p0_p1_p2: tuple[str, str, str]
    optimizer_updates: int
    terminal_trainable_state_available_on_live_model: bool


def _validate_loss_config_v2(
    loss_kwargs: Optional[Mapping[str, float]],
) -> tuple[tuple[str, float], ...]:
    resolved = {
        "smooth_l1_weight": 1.0,
        "cosine_weight": 1.0,
        "infonce_weight": 1.0,
        "flow_weight": 1.0,
        "temperature": 0.07,
        **dict(loss_kwargs or {}),
    }
    expected = {
        "smooth_l1_weight",
        "cosine_weight",
        "infonce_weight",
        "flow_weight",
        "temperature",
    }
    if set(resolved) != expected or any(
        type(resolved[name]) not in (int, float)
        or not math.isfinite(float(resolved[name]))
        for name in expected
    ):
        fail("V2 closed loss configuration differs")
    if (
        float(resolved["smooth_l1_weight"]) < 0
        or float(resolved["cosine_weight"]) < 0
        or float(resolved["infonce_weight"]) <= 0
        or float(resolved["flow_weight"]) <= 0
        or float(resolved["temperature"]) <= 0
        or float(resolved["smooth_l1_weight"])
        + float(resolved["cosine_weight"])
        <= 0
    ):
        fail("V2 point/contrastive/flow loss weights must all remain active")
    return tuple(sorted((name, float(value)) for name, value in resolved.items()))


def _require_exact_import_v2(module: Any, source_name: str, expected: str) -> None:
    module_file = getattr(module, "__file__", None)
    source = METHOD_ROOT / source_name
    if (
        type(module_file) is not str
        or Path(module_file).resolve(strict=True) != source.resolve(strict=True)
        or source.is_symlink()
        or file_sha256(source) != expected
    ):
        fail(f"executed module ownership differs: {source_name}")


def _validate_initialized_runtime_v2(
    *,
    preflight: FrozenSidecarPreflightV2,
    model: Any,
    base_renderer: Any,
    transformer: Any,
    conditioner: Any,
    hook_handle: Any,
    parallel: Any,
    distributed: Any,
    device: Any,
) -> None:
    import torch
    import torch.distributed as dist
    import torch.utils.checkpoint as torch_checkpoint
    import action_anchor_distillation_v1 as distillation_module
    import action_plan_predictor_v1 as action_plan_module
    import clean_source_visual_context_stage_b_contract_v1 as schedule_contract
    import inference_sigma_strata as sigma_strata
    import packed_preservation_lora_v2 as packed_core
    import packed_preservation_release_v2 as release_contract
    import source_self_runtime as runtime
    import train_action_edit_large_lora_0817_v1 as renderer_v1
    import train_lora as legacy
    from action_plan_predictor_v1 import ActionPlanConditionerV1, exact_state_dict_abi

    _require_exact_import_v2(
        renderer_v1, RENDERER_RUNNER_SOURCE_NAME, preflight.renderer_runner_source_sha256
    )
    _require_exact_import_v2(
        schedule_contract, SCHEDULE_SOURCE_NAME, preflight.schedule_source_sha256
    )
    _require_exact_import_v2(
        packed_core, PACKED_CORE_SOURCE_NAME, preflight.packed_core_source_sha256
    )
    _require_exact_import_v2(
        runtime, RUNTIME_SOURCE_NAME, preflight.runtime_source_sha256
    )
    _require_exact_import_v2(
        distillation_module,
        DISTILLATION_SOURCE_NAME,
        preflight.distillation_source_sha256,
    )
    _require_exact_import_v2(
        sigma_strata,
        INFERENCE_SIGMA_SOURCE_NAME,
        preflight.inference_sigma_source_sha256,
    )
    if getattr(schedule_contract, "exact40", None) is not sigma_strata:
        fail("V2 schedule transitive inference-sigma ownership differs")
    try:
        renderer_v1.validate_imported_release_modules(
            preflight.renderer_release_closure,
            {
                "action_plan_predictor_v1.py": action_plan_module,
                "clean_source_visual_context_stage_b_contract_v1.py": schedule_contract,
                "inference_sigma_strata.py": sigma_strata,
                "packed_preservation_lora_v2.py": packed_core,
                "packed_preservation_release_v2.py": release_contract,
                "source_self_runtime.py": runtime,
                "train_action_edit_large_lora_0817_v1.py": renderer_v1,
                "train_lora.py": legacy,
            },
        )
    except Exception as error:
        raise ActionAnchorV2CanaryError(
            f"V2 imported 0817 release closure differs: {error}"
        ) from error
    if (
        type(distributed) is not runtime.DistributedContract
        or type(parallel) is not runtime.ParallelContext
        or parallel.contract is not distributed
        or distributed.topology != runtime.WORLD8_DP2_SP4
        or distributed.world_size != WORLD_SIZE
        or distributed.local_world_size != WORLD_SIZE
        or distributed.rank != distributed.local_rank
        or not 0 <= distributed.arm_index < DP_SIZE
        or not 0 <= distributed.sp_rank < SP_SIZE
        or device != torch.device("cuda", distributed.local_rank)
        or not dist.is_initialized()
        or dist.get_world_size() != WORLD_SIZE
        or dist.get_rank() != distributed.rank
        or parallel.world_group is not dist.group.WORLD
        or dist.get_world_size(group=parallel.sp_group) != SP_SIZE
        or dist.get_world_size(group=parallel.dp_group) != DP_SIZE
        or dist.get_rank(group=parallel.sp_group) != distributed.sp_rank
        or dist.get_rank(group=parallel.dp_group) != distributed.arm_index
    ):
        fail("V2 formal WORLD8 runtime/parallel ownership differs")
    if not callable(getattr(torch_checkpoint, "set_checkpoint_early_stop", None)):
        fail("V2 runtime lacks checkpoint early-stop control")
    if (
        type(conditioner) is not ActionPlanConditionerV1
        or type(hook_handle) is not renderer_v1.InstalledActionPlanHooks
        or renderer_v1.TRANSFORMER_BLOCKS != TRANSFORMER_BLOCKS
        or renderer_v1.HIDDEN_WIDTH != HIDDEN_WIDTH
        or model.get_base_model() is not base_renderer
        or base_renderer.diff_dec.transformer is not transformer
        or hook_handle.transformer is not transformer
        or hook_handle.conditioner is not conditioner
        or getattr(transformer, "action_plan_conditioner_v1", None) is not conditioner
        or tuple(hook_handle.block_ids)
        != tuple(id(block) for block in transformer.blocks)
        or len(tuple(hook_handle.handles)) != TRANSFORMER_BLOCKS
        or len(tuple(transformer.blocks)) != TRANSFORMER_BLOCKS
        or hook_handle.restored is not False
        or any(len(getattr(block, "_forward_hooks", {})) != 1 for block in transformer.blocks)
    ):
        fail("V2 initialized Bernini/conditioner/exact30 hook ownership differs")
    renderer_v1.validate_action_route_checkpointing_installation(transformer)
    conditioner.config.require_formal_0817()
    if (
        exact_state_dict_abi(conditioner).get("abi_sha256")
        != renderer_v1.ACTION_PLAN_CONDITIONER_STATE_ABI_SHA256
        or any(
            parameter.dtype != torch.float32 or parameter.device != device
            for parameter in conditioner.parameters()
        )
    ):
        fail("V2 formal conditioner state/dtype/device ABI differs")


def _bind_local_records_v2(
    preflight: FrozenSidecarPreflightV2,
    *,
    distributed: Any,
    device: Any,
    local_inputs: Sequence[RuntimeRecordInputV2],
) -> tuple[BoundRuntimeRecordV2, ...]:
    all_teachers = materialize_and_validate_teachers_v2(preflight, device=device)
    expected_local = all_teachers[distributed.arm_index :: DP_SIZE]
    if (
        type(local_inputs) not in (list, tuple)
        or len(local_inputs) != MAX_UPDATES * GRADIENT_ACCUMULATION
        or len(expected_local) != len(local_inputs)
    ):
        fail("V2 local DP-arm runtime input coverage differs")
    bound: list[BoundRuntimeRecordV2] = []
    for teacher, payload in zip(expected_local, local_inputs):
        if type(payload) is not RuntimeRecordInputV2:
            fail("V2 runtime input must be an exact RuntimeRecordInputV2")
        item = bind_runtime_record_v2(
            teacher,
            logical_record=payload.logical_record,
            dataset_iid=payload.dataset_iid,
            dataset_row_index=payload.dataset_row_index,
            source_media_path=payload.source_media_path,
            target_media_path=payload.target_media_path,
            source_mode=payload.source_mode,
            target_mode=payload.target_mode,
            instruction=payload.instruction,
            text_lens=payload.text_lens,
            text_embs=payload.text_embs,
            instruction_tokens=payload.instruction_tokens,
        )
        if (
            item.text_embs.device != device
            or item.instruction_tokens.device != device
            or teacher.q_y.phase_tokens.device != device
            or any(anchor.plan.phase_tokens.device != device for anchor in teacher.anchors)
        ):
            fail("V2 runtime text/teacher tensors are not on the local accelerator")
        bound.append(item)
    expected_logical = tuple(range(distributed.arm_index, GLOBAL_RECORDS, DP_SIZE))
    if tuple(item.teacher.record.logical_record for item in bound) != expected_logical:
        fail("V2 local DP schedule/parity differs")
    return tuple(bound)


def _local_runtime_digest_v2(records: Sequence[BoundRuntimeRecordV2]) -> str:
    return object_sha256(
        [
            {
                "logical_record": item.teacher.record.logical_record,
                "row_id": item.teacher.record.row_id,
                "source_sha256": file_sha256(item.source_media_path),
                "target_sha256": file_sha256(item.target_media_path),
                "source_mode": runtime_tensor_sha256_v2(item.source_mode),
                "target_mode": runtime_tensor_sha256_v2(item.target_mode),
                "instruction_sha256": hashlib.sha256(
                    item.instruction.encode("utf-8")
                ).hexdigest(),
                "renderer_text": runtime_tensor_sha256_v2(item.text_embs),
                "instruction_tokens": runtime_tensor_sha256_v2(
                    item.instruction_tokens
                ),
            }
            for item in records
        ]
    )


def _world8_preoptimizer_consensus_v2(
    *,
    preflight: FrozenSidecarPreflightV2,
    parallel: Any,
    distributed: Any,
    records: Sequence[BoundRuntimeRecordV2],
    named: Sequence[tuple[str, Any]],
    learning_rate: float,
    max_grad_norm: float,
    seed: int,
    loss_items: tuple[tuple[str, float], ...],
) -> tuple[str, str, str, str]:
    import torch.distributed as dist
    import source_self_runtime as runtime
    import train_action_edit_large_lora_0817_v1 as renderer_v1

    global_contract = object_sha256(
        {
            "manifest_file_sha256": preflight.manifest_file_sha256,
            "manifest_digest": preflight.manifest_digest,
            "teacher_authority_sha256": preflight.teacher_authority_sha256,
            "classification_authority_sha256": preflight.classification_authority_sha256,
            "executed_sources": preflight.receipt(),
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "loss_items": list(loss_items),
            "topology": "WORLD8_DP2xSP4_GA4_exact2",
        }
    )
    runtime.digest_consensus(
        global_contract,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="V2 global manifest/config contract",
    )
    local_runtime = _local_runtime_digest_v2(records)
    runtime.digest_consensus(
        local_runtime,
        group=parallel.sp_group,
        expected_count=SP_SIZE,
        label="V2 SP4 local runtime records",
    )
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered,
        {
            "rank": distributed.rank,
            "arm": distributed.arm_index,
            "logical": [item.teacher.record.logical_record for item in records],
            "digest": local_runtime,
        },
        group=parallel.world_group,
    )
    if [item.get("rank") for item in gathered] != list(range(WORLD_SIZE)):
        fail("V2 WORLD8 runtime record gather differs")
    for rank, item in enumerate(gathered):
        arm = rank // SP_SIZE
        if (
            item.get("arm") != arm
            or item.get("logical") != list(range(arm, GLOBAL_RECORDS, DP_SIZE))
            or item.get("digest") != gathered[arm * SP_SIZE].get("digest")
        ):
            fail("V2 WORLD8 DP/SP logical-record consensus differs")
    if gathered[0].get("digest") == gathered[SP_SIZE].get("digest"):
        fail("V2 DP arms unexpectedly duplicate the runtime record payload")
    inventory = object_sha256(list(renderer_v1.trainable_inventory(named)))
    runtime.digest_consensus(
        inventory,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="V2 exact trainable inventory",
    )
    parameters = renderer_v1.tensor_digest(named)
    runtime.digest_consensus(
        parameters,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="V2 initial trainable state",
    )
    return global_contract, local_runtime, inventory, parameters


def prepare_world8_canary_v2(
    preflight: FrozenSidecarPreflightV2,
    *,
    model: Any,
    base_renderer: Any,
    transformer: Any,
    conditioner: Any,
    hook_handle: Any,
    parallel: Any,
    distributed: Any,
    rope: Any,
    device: Any,
    local_inputs: Sequence[RuntimeRecordInputV2],
    learning_rate: float = 1.0e-4,
    max_grad_norm: float = 1.0,
    seed: int = 20260818,
    loss_kwargs: Optional[Mapping[str, float]] = None,
) -> PreparedWorld8CanaryV2:
    """Issue the only optimizer capability after all 16 teachers and 8 rows close."""

    preflight = _require_preflight_capability_v2(preflight)
    _verify_executed_sources_v2(preflight)
    import torch
    import train_action_edit_large_lora_0817_v1 as renderer_v1

    if (
        type(learning_rate) not in (int, float)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
        or type(max_grad_norm) not in (int, float)
        or not math.isfinite(float(max_grad_norm))
        or max_grad_norm <= 0
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        fail("V2 optimizer hyperparameters/seed differ")
    resolved_loss = _validate_loss_config_v2(loss_kwargs)
    _validate_initialized_runtime_v2(
        preflight=preflight,
        model=model,
        base_renderer=base_renderer,
        transformer=transformer,
        conditioner=conditioner,
        hook_handle=hook_handle,
        parallel=parallel,
        distributed=distributed,
        device=device,
    )
    bound = _bind_local_records_v2(
        preflight,
        distributed=distributed,
        device=device,
        local_inputs=local_inputs,
    )
    named = tuple(renderer_v1.exact_trainable_named_parameters(model, conditioner))
    if not named or any(
        parameter.dtype != torch.float32 or parameter.device != device
        for _, parameter in named
    ):
        fail("V2 exact trainable inventory differs before optimizer creation")
    certify_zero_init_exact30_v2(conditioner)
    global_contract, local_runtime, inventory, initial_parameters = (
        _world8_preoptimizer_consensus_v2(
            preflight=preflight,
            parallel=parallel,
            distributed=distributed,
            records=bound,
            named=named,
            learning_rate=float(learning_rate),
            max_grad_norm=float(max_grad_norm),
            seed=seed,
            loss_items=resolved_loss,
        )
    )
    lease = object()
    execution = PreparedWorld8CanaryV2(
        preflight=preflight,
        model=model,
        base_renderer=base_renderer,
        transformer=transformer,
        conditioner=conditioner,
        hook_handle=hook_handle,
        parallel=parallel,
        distributed=distributed,
        rope=rope,
        device=device,
        local_inputs=tuple(local_inputs),
        learning_rate=float(learning_rate),
        max_grad_norm=float(max_grad_norm),
        seed=seed,
        loss_items=resolved_loss,
        _lease=lease,
    )
    seal = _ExecutionSealV2(
        lease=lease,
        runtime_object_ids=tuple(
            id(value)
            for value in (
                preflight,
                model,
                base_renderer,
                transformer,
                conditioner,
                hook_handle,
                parallel,
                distributed,
                rope,
                device,
            )
        ),
        local_input_ids=tuple(id(value) for value in execution.local_inputs),
        named_parameter_identity=tuple((name, id(parameter)) for name, parameter in named),
        inventory_sha256=inventory,
        initial_parameter_sha256=initial_parameters,
        global_contract_sha256=global_contract,
        local_runtime_sha256=local_runtime,
        scalar_contract=(
            execution.learning_rate,
            execution.max_grad_norm,
            execution.seed,
            execution.loss_items,
        ),
    )
    with _EXECUTION_REGISTRY_LOCK:
        _EXECUTION_REGISTRY[execution] = seal
    return execution


def run_exact_two_updates_v2(
    execution: PreparedWorld8CanaryV2,
) -> CanaryRunResultV2:
    """Run genuine flow+action loss; no callback or external optimizer exists."""

    if type(execution) is not PreparedWorld8CanaryV2:
        fail("V2 execution capability is absent, forged, or already consumed")
    with _EXECUTION_REGISTRY_LOCK:
        seal = _EXECUTION_REGISTRY.pop(execution, None)
        if type(seal) is not _ExecutionSealV2 or seal.lease is not execution._lease:
            fail("V2 execution capability is absent, forged, or already consumed")
    _verify_executed_sources_v2(execution.preflight)
    import torch
    import torch.utils.checkpoint as torch_checkpoint
    import clean_source_visual_context_stage_b_contract_v1 as schedule_contract
    import packed_preservation_lora_v2 as packed_core
    import source_self_runtime as runtime
    import train_action_edit_large_lora_0817_v1 as renderer_v1

    runtime_ids = tuple(
        id(value)
        for value in (
            execution.preflight,
            execution.model,
            execution.base_renderer,
            execution.transformer,
            execution.conditioner,
            execution.hook_handle,
            execution.parallel,
            execution.distributed,
            execution.rope,
            execution.device,
        )
    )
    if (
        runtime_ids != seal.runtime_object_ids
        or tuple(id(value) for value in execution.local_inputs) != seal.local_input_ids
        or (
            execution.learning_rate,
            execution.max_grad_norm,
            execution.seed,
            execution.loss_items,
        )
        != seal.scalar_contract
    ):
        fail("V2 sealed execution payload changed after issuance")
    _validate_initialized_runtime_v2(
        preflight=execution.preflight,
        model=execution.model,
        base_renderer=execution.base_renderer,
        transformer=execution.transformer,
        conditioner=execution.conditioner,
        hook_handle=execution.hook_handle,
        parallel=execution.parallel,
        distributed=execution.distributed,
        device=execution.device,
    )
    named = tuple(
        renderer_v1.exact_trainable_named_parameters(
            execution.model, execution.conditioner
        )
    )
    if tuple((name, id(parameter)) for name, parameter in named) != (
        seal.named_parameter_identity
    ):
        fail("V2 exact trainable identity changed after capability issuance")
    zero_receipt = certify_zero_init_exact30_v2(execution.conditioner)
    # These reopen every q/media/mode/text byte and issue new bound records.
    # No optimizer object exists until this second complete audit succeeds.
    local_records = _bind_local_records_v2(
        execution.preflight,
        distributed=execution.distributed,
        device=execution.device,
        local_inputs=execution.local_inputs,
    )
    global_contract, local_runtime, inventory, p0 = _world8_preoptimizer_consensus_v2(
        preflight=execution.preflight,
        parallel=execution.parallel,
        distributed=execution.distributed,
        records=local_records,
        named=named,
        learning_rate=execution.learning_rate,
        max_grad_norm=execution.max_grad_norm,
        seed=execution.seed,
        loss_items=execution.loss_items,
    )
    if (
        global_contract != seal.global_contract_sha256
        or local_runtime != seal.local_runtime_sha256
        or inventory != seal.inventory_sha256
        or p0 != seal.initial_parameter_sha256
    ):
        fail("V2 signed pre-optimizer state changed after capability issuance")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=execution.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    history: list[Mapping[str, Any]] = []
    parameter_digests = [p0]
    loss_kwargs = dict(execution.loss_items)
    for step_zero in range(MAX_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        coordinates = tuple(
            schedule_contract.coordinates_for_optimizer_step(step_zero)
        )
        if len(coordinates) != GRADIENT_ACCUMULATION:
            fail("V2 exact schedule did not provide four microbatches")
        outputs: list[RenderedMicrobatchV2] = []
        for microbatch, coordinate in enumerate(coordinates):
            record = local_records[
                step_zero * GRADIENT_ACCUMULATION + microbatch
            ]
            noise_seed = renderer_v1.deterministic_seed(
                execution.seed,
                "action-anchor-v2-flow",
                step_zero,
                microbatch,
                execution.distributed.arm_index,
                record.teacher.record.row_id,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(noise_seed)
            epsilon = torch.randn(
                tuple(record.target_mode.shape),
                generator=generator,
                dtype=torch.float32,
            ).contiguous()
            outputs.append(
                render_action_anchor_microbatch_v2(
                    base_renderer=execution.base_renderer,
                    transformer=execution.transformer,
                    conditioner=execution.conditioner,
                    runtime_record=record,
                    coordinate=coordinate,
                    epsilon=epsilon,
                    rope=execution.rope,
                    device=execution.device,
                    sequence_parallel_rank=execution.distributed.sp_rank,
                    torch_checkpoint=torch_checkpoint,
                    packed_role_layout=packed_core.packed_role_layout,
                    loss_kwargs=loss_kwargs,
                )
            )
        if any(
            output.objective.point_pair_count != 1
            or output.objective.flow_preservation.untyped_storage().data_ptr()
            != output.flow_loss.untyped_storage().data_ptr()
            for output in outputs
        ):
            fail("V2 pre-optimizer objective evidence closure differs")
        synchronized = renderer_v1.synchronize_gradients_bucketed(
            named, execution.parallel
        )
        step1 = (
            audit_step1_gradients_v2(execution.conditioner)
            if step_zero == 0
            else None
        )
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named],
            execution.max_grad_norm,
        )
        optimizer.step()
        parameter_digest = renderer_v1.tensor_digest(named)
        runtime.digest_consensus(
            parameter_digest,
            group=execution.parallel.world_group,
            expected_count=WORLD_SIZE,
            label=f"V2 trainable state P{step_zero + 1}",
        )
        if parameter_digest in parameter_digests:
            fail("V2 optimizer update did not change the exact trainable state")
        parameter_digests.append(parameter_digest)
        microbatch_losses = []
        for record, output in zip(
            local_records[
                step_zero * GRADIENT_ACCUMULATION :
                (step_zero + 1) * GRADIENT_ACCUMULATION
            ],
            outputs,
        ):
            scalars = {
                "total": float(output.objective.total.detach().item()),
                "action_only": float(output.objective.action_only.detach().item()),
                "smooth_l1": float(output.objective.smooth_l1.detach().item()),
                "cosine": float(output.objective.cosine.detach().item()),
                "infonce": float(output.objective.infonce.detach().item()),
                "flow": float(output.flow_loss.detach().item()),
            }
            if any(not math.isfinite(value) for value in scalars.values()):
                fail("V2 non-finite objective evidence after optimizer step")
            microbatch_losses.append(
                {
                    "logical_record": record.teacher.record.logical_record,
                    "row_id": record.teacher.record.row_id,
                    "losses": scalars,
                    "q_pred_receipt_digest": output.objective.q_pred_receipt_digest,
                    "q_y_receipt_digest": output.objective.q_y_receipt_digest,
                    "point_pair_count": output.objective.point_pair_count,
                    "contrastive_positive_pair_count": (
                        output.objective.contrastive_positive_pair_count
                    ),
                    "contrastive_negative_pair_count": (
                        output.objective.contrastive_negative_pair_count
                    ),
                    "excluded_pair_count": output.objective.excluded_pair_count,
                    "active_q_anchor_infonce": (
                        output.objective.contrastive_negative_pair_count > 0
                    ),
                    "route": output.local_route_receipt,
                }
            )
        history.append(
            {
                "step": step_zero + 1,
                "optimizer_step_executed": True,
                "microbatches": GRADIENT_ACCUMULATION,
                "logical_records": [
                    record.teacher.record.logical_record
                    for record in local_records[
                        step_zero * GRADIENT_ACCUMULATION :
                        (step_zero + 1) * GRADIENT_ACCUMULATION
                    ]
                ],
                "synchronized_gradient_norm": synchronized,
                "microbatch_objectives": microbatch_losses,
                "parameter_sha256_before": parameter_digests[-2],
                "parameter_sha256_after": parameter_digests[-1],
                "p0_zero_init_certificate": zero_receipt if step_zero == 0 else None,
                "step1_gradient_decomposition": step1,
                "q_y_unique_point_teacher": True,
                "q_anchor_point_teacher": False,
                "real_renderer_flow_preservation": True,
                "q_pred_fp32_then_renderer_cast": True,
            }
        )
    if len(history) != MAX_UPDATES:
        fail("V2 terminal two-update closure differs")
    if len(parameter_digests) != 3 or len(set(parameter_digests)) != 3:
        fail("V2 terminal P0/P1/P2 parameter closure differs")
    return CanaryRunResultV2(
        history=tuple(history),
        parameter_sha256_p0_p1_p2=tuple(parameter_digests),
        optimizer_updates=MAX_UPDATES,
        terminal_trainable_state_available_on_live_model=True,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--print-required-sidecar-schema", action="store_true")
    value.add_argument("--sidecar-manifest")
    value.add_argument("--renderer-release-manifest")
    value.add_argument("--expected-sidecar-manifest-file-sha256")
    value.add_argument("--expected-teacher-authority-file-sha256")
    value.add_argument("--expected-teacher-authority-sha256")
    value.add_argument("--expected-classification-authority-sha256")
    value.add_argument("--expected-predictor-source-sha256")
    value.add_argument("--expected-distillation-source-sha256")
    value.add_argument("--expected-renderer-runner-source-sha256")
    value.add_argument("--expected-v2-runner-source-sha256")
    value.add_argument("--expected-schedule-source-sha256")
    value.add_argument("--expected-packed-core-source-sha256")
    value.add_argument("--expected-runtime-source-sha256")
    value.add_argument("--expected-legacy-loader-source-sha256")
    value.add_argument("--expected-world8-adapter-source-sha256")
    value.add_argument("--expected-inference-sigma-source-sha256")
    value.add_argument("--expected-renderer-release-manifest-sha256")
    value.add_argument("--preflight-only", action="store_true")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.print_required_sidecar_schema:
        print(json.dumps(sidecar_schema_template_v1(), indent=2, sort_keys=True))
        return 0
    required = {
        "--sidecar-manifest": args.sidecar_manifest,
        "--renderer-release-manifest": args.renderer_release_manifest,
        "--expected-sidecar-manifest-file-sha256": (
            args.expected_sidecar_manifest_file_sha256
        ),
        "--expected-teacher-authority-file-sha256": (
            args.expected_teacher_authority_file_sha256
        ),
        "--expected-teacher-authority-sha256": args.expected_teacher_authority_sha256,
        "--expected-classification-authority-sha256": (
            args.expected_classification_authority_sha256
        ),
        "--expected-predictor-source-sha256": args.expected_predictor_source_sha256,
        "--expected-distillation-source-sha256": (
            args.expected_distillation_source_sha256
        ),
        "--expected-renderer-runner-source-sha256": (
            args.expected_renderer_runner_source_sha256
        ),
        "--expected-v2-runner-source-sha256": args.expected_v2_runner_source_sha256,
        "--expected-schedule-source-sha256": args.expected_schedule_source_sha256,
        "--expected-packed-core-source-sha256": args.expected_packed_core_source_sha256,
        "--expected-runtime-source-sha256": args.expected_runtime_source_sha256,
        "--expected-legacy-loader-source-sha256": (
            args.expected_legacy_loader_source_sha256
        ),
        "--expected-world8-adapter-source-sha256": (
            args.expected_world8_adapter_source_sha256
        ),
        "--expected-inference-sigma-source-sha256": (
            args.expected_inference_sigma_source_sha256
        ),
        "--expected-renderer-release-manifest-sha256": (
            args.expected_renderer_release_manifest_sha256
        ),
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        fail(
            "frozen action sidecars are missing; no optimizer is authorized; "
            f"missing={missing}; run --print-required-sidecar-schema"
        )
    preflight = preflight_frozen_sidecars_v2(
        Path(args.sidecar_manifest),
        renderer_release_manifest_path=Path(args.renderer_release_manifest),
        expected_manifest_file_sha256=args.expected_sidecar_manifest_file_sha256,
        expected_renderer_release_manifest_sha256=(
            args.expected_renderer_release_manifest_sha256
        ),
        expected_teacher_authority_file_sha256=(
            args.expected_teacher_authority_file_sha256
        ),
        expected_teacher_authority_sha256=args.expected_teacher_authority_sha256,
        expected_classification_authority_sha256=(
            args.expected_classification_authority_sha256
        ),
        expected_predictor_source_sha256=args.expected_predictor_source_sha256,
        expected_distillation_source_sha256=args.expected_distillation_source_sha256,
        expected_renderer_runner_source_sha256=(
            args.expected_renderer_runner_source_sha256
        ),
        expected_v2_runner_source_sha256=args.expected_v2_runner_source_sha256,
        expected_schedule_source_sha256=args.expected_schedule_source_sha256,
        expected_packed_core_source_sha256=args.expected_packed_core_source_sha256,
        expected_runtime_source_sha256=args.expected_runtime_source_sha256,
        expected_legacy_loader_source_sha256=(
            args.expected_legacy_loader_source_sha256
        ),
        expected_world8_adapter_source_sha256=(
            args.expected_world8_adapter_source_sha256
        ),
        expected_inference_sigma_source_sha256=(
            args.expected_inference_sigma_source_sha256
        ),
    )
    # CPU tensor/receipt validation is intentionally part of preflight-only;
    # a file-hash scan alone does not qualify q_y or route q_anchor.
    deep = materialize_and_validate_teachers_v2(preflight, device="cpu")
    print(
        json.dumps(
            {
                **preflight.receipt(),
                "all_teacher_tensors_and_receipts_deep_validated": len(deep),
            },
            sort_keys=True,
        )
    )
    if not args.preflight_only:
        fail(
            "CPU preflight completed; model launch must call run_exact_two_updates_v2 "
            "from the WORLD8 0817 Bernini runtime adapter"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
