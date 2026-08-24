#!/usr/bin/env python3
"""Official full-30 Bernini action-LoRA trainer.

This is the executable consumer of the full30 contracts.  It is deliberately
not a block/carrier probe: the only production architecture is rank-256 LoRA
on q/k/v/out in attn1+attn2 of all 30 blocks, plus the typed source/target
patch and role parameters (188,946,432 FP32 trainables).

The import prelude contains only Python's standard library.  Model, authority,
objective, optimizer, and checkpoint code cannot be imported until the exact
executed release has been content-validated.  The small orchestration API is
also dependency-injected so hostile CPU tests can exercise the real phase
ordering without loading Bernini or starting a distributed process.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import random
import re
import stat
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
METHOD = "bernini-full30-action-lora-v1"
RECEIPT_SCHEMA_VERSION = "bernini-full30-action-training-receipt-v1"
UPDATE_RECEIPT_SCHEMA_VERSION = "bernini-full30-action-update-receipt-v1"
RELEASE_SCHEMA_VERSION = "bernini-full30-action-training-release-v1"
SEGMENT_GATE_SCHEMA_VERSION = "bernini-full30-action-segment-review-gate-v2"
REDUCED_CANARY_TO_FRESH_FORMAL_GATE_SCHEMA_VERSION = (
    "bernini-full30-action-reduced-canary-to-fresh-formal-gate-v1"
)
REDUCED_CANARY_TO_FRESH_FORMAL_EVIDENCE_SCHEMA_VERSION = (
    "bernini-full30-action-reduced-canary-to-fresh-formal-evidence-v1"
)
# Compatibility aliases for callers that only need to author the gate.  The
# accepted payload is the closed reduced->fresh transition above; the former
# equal-authority gate is intentionally no longer accepted.
FRESH_FORMAL_CANARY_GATE_SCHEMA_VERSION = (
    REDUCED_CANARY_TO_FRESH_FORMAL_GATE_SCHEMA_VERSION
)
FRESH_FORMAL_CANARY_EVIDENCE_SCHEMA_VERSION = (
    REDUCED_CANARY_TO_FRESH_FORMAL_EVIDENCE_SCHEMA_VERSION
)
BOUNDARY_SCHEMA_VERSION = "bernini-full30-action-boundary-plan-v1"
AUTHORITY_PROJECTION_SCHEMA_VERSION = "bernini-full30-action-authority-projections-v1"
TEACHER_COMPOSITE_SCHEMA_VERSION = "bernini-full30-action-teacher-amplitude-composite-v1"
AMPLITUDE_RUNTIME_BINDING_SCHEMA_VERSION = "bernini-full30-action-amplitude-runtime-binding-v1"
AMPLITUDE_RUNTIME_IDENTITY_SCHEMA_VERSION = (
    "bernini-full30-action-frozen-runtime-identity-v2"
)
AMPLITUDE_COMPUTE_CONTRACT_SCHEMA_VERSION = (
    "bernini-full30-action-frozen-compute-contract-v1"
)
OFFICIAL_PROVIDER_ABI = "full30-psiout-official-provider-v1"

WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
LOCAL_RECORDS = 4
GLOBAL_BATCH = 8
MAX_UPDATES = 160
FRAME_COUNT = 81
LATENT_PHASES = 21
REGISTERED_FORMAL_ENDPOINTS = (20, 40, 60, 80, 100, 120, 140, 160)
FIRST_SEGMENT_CHECKPOINTS = (0, 5, 10, 20)
CANARY_CHECKPOINTS = (0, 1, 2)
PROFILES = ("disposable-canary-2", "review-gated-segment")
ARMS = ("action+retain", "action-only")
NOOP_INSTRUCTION = (
    "Keep the source video exactly unchanged, including every subject, "
    "appearance, action, camera motion, background, timing, and composition."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,511}$")
_RELEASE_VALIDATED = False
_VALIDATED_RELEASE_RECEIPT: Optional[Mapping[str, Any]] = None
_BOUND_RELEASE_MODULES: dict[str, Any] = {}
_BUSINESS_MODULES: Optional["BusinessModulesV1"] = None

REQUIRED_RELEASE_FILES = frozenset(
    {
        "train_full30_action_lora_v1.py",
        "full30_action_learning_v1.py",
        "full30_action_runtime_v1.py",
        "full30_action_optimizer_v1.py",
        "full30_action_checkpoint_v1.py",
        "full30_action_training_step_v1.py",
        "full30_action_data_teacher_authority_v1.py",
        "full30_action_amplitude_authority_v1.py",
        "full30_action_mechanism_canary_authority_v1.py",
        "packed_preservation_lora_v2.py",
        "source_self_runtime.py",
        "train_packed_preservation_lora_v2.py",
        "packed_preservation_release_v2.py",
        "clean_source_visual_context_training_v1.py",
        "inference_sigma_strata.py",
        "infer_dclr_reward_runtime_smoke.py",
        "graft_phase_a_native_training_closure_v1.py",
        "train_lora.py",
        "dual_conditional_ratio_core.py",
        "dclr_runtime_contract.py",
        "infer_source_kv_carrier_oracle.py",
        "motion_residual.py",
        "infer_lora.py",
        "source_kv_replay.py",
        "source_kv_route_batches.py",
        "full30_action_psiout_materializer_v1.py",
    }
)
FROZEN_RELEASE_MEMBER_SHA256 = MappingProxyType(
    {
        "full30_action_runtime_v1.py": (
            "9179394fddfd17a2a02773b3f94c77024dabdc5076f92980c3e637c1d0dd7da1"
        ),
        "full30_action_training_step_v1.py": (
            "c3cf9b5f51a0247de20ad687b51109e8088e297d20c17fec722362da4c4c7ee2"
        ),
        "full30_action_data_teacher_authority_v1.py": (
            "d210628791e6861b3cfb141bd9bca930966b9dfc9050d54460e69c18d0883e2a"
        ),
        "full30_action_amplitude_authority_v1.py": (
            "103f9f676b8126615d6fa7916b9c9e4dd37003fbacda0055f046d6a8de8f0f93"
        ),
        "full30_action_psiout_materializer_v1.py": (
            "a7daf7f81956818669f2d23e806034ab902aa34bcbb8e76315f1d2ee89c53b45"
        ),
        "full30_action_mechanism_canary_authority_v1.py": (
            "2aacb0d88e47db29676c72636aaba9cbc7508b4d85db215e62c7c715809fb17d"
        ),
    }
)

AMPLITUDE_COMPUTE_CONTRACT = MappingProxyType(
    {
        "schema_version": AMPLITUDE_COMPUTE_CONTRACT_SCHEMA_VERSION,
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
)
EXECUTED_FROZEN_ROUTE_CONTRACT = MappingProxyType(
    {
        "model_eval": True,
        "torch_inference_mode": True,
        "peft_disable_adapter": True,
        "official_frozen_native_only": True,
        "prior_model_mode_restored": True,
        "detached": True,
    }
)


class Full30ActionTrainingError(RuntimeError):
    """Raised before an ambiguous update can be counted or published."""


def fail(message: str) -> NoReturn:
    raise Full30ActionTrainingError(message)


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain_json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Full30ActionTrainingError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _seal(value: Mapping[str, Any]) -> Mapping[str, Any]:
    unsigned = dict(value)
    if "receipt_digest" in unsigned:
        fail("unsigned receipt already contains receipt_digest")
    return MappingProxyType({**unsigned, "receipt_digest": object_sha256(unsigned)})


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def _plain_absolute_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        metadata = requested.lstat()
    except OSError as error:
        raise Full30ActionTrainingError(f"{label} is unavailable: {error}") from error
    if resolved != requested or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be one canonical plain file")
    return resolved


def _plain_absolute_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = requested.resolve(strict=True)
        metadata = requested.lstat()
    except OSError as error:
        raise Full30ActionTrainingError(f"{label} is unavailable: {error}") from error
    if resolved != requested or not stat.S_ISDIR(metadata.st_mode):
        fail(f"{label} must be one canonical plain directory")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_nlink,
            )
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        named = path.lstat()
    except OSError as error:
        raise Full30ActionTrainingError(f"cannot hash {path}: {error}") from error
    observed = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )
    if identity != observed(after) or identity != observed(named):
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _strict_json_file(path: Path, *, expected_sha256: str, label: str) -> Mapping[str, Any]:
    expected = _sha256(expected_sha256, label=f"{label} expected SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 32 * 1024 * 1024:
            fail(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                fail(f"{label} was truncated while reading")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1) != b"":
            fail(f"{label} grew while reading")
        after = os.fstat(descriptor)
    except OSError as error:
        raise Full30ActionTrainingError(f"cannot read {label}: {error}") from error
    finally:
        try:
            os.close(descriptor)
        except UnboundLocalError:
            pass
    named = path.lstat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        fail(f"{label} changed while reading")
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected:
        fail(f"{label} SHA-256 differs")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                fail(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30ActionTrainingError(f"cannot decode {label}") from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        fail(f"{label} must be one canonical JSON object plus newline")
    return value


def validate_executed_release_v1(
    *,
    method_root: str | Path,
    manifest: str | Path,
    expected_manifest_sha256: str,
    expected_release_sha256: str,
    test_only_required_files: Optional[Iterable[str]] = None,
    test_only_require_current_entrypoint: bool = True,
) -> Mapping[str, Any]:
    """Validate the exact executed source closure before business imports.

    The release manifest is intentionally simple and archive-format agnostic;
    a launcher may extract tar/OCI/other immutable storage, but the bytes that
    Python executes must exactly equal the manifest and its release digest.
    """

    global _RELEASE_VALIDATED, _VALIDATED_RELEASE_RECEIPT
    if _RELEASE_VALIDATED:
        fail("executed release validation may occur only once per process")
    root = _plain_absolute_directory(method_root, label="method root")
    manifest_path = _plain_absolute_file(manifest, label="release manifest")
    expected_release = _sha256(
        expected_release_sha256, label="expected release SHA-256"
    )
    value = _strict_json_file(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        label="release manifest",
    )
    fields = {
        "schema_version",
        "exact_member_closure",
        "files",
        "release_sha256",
        "manifest_digest",
    }
    unsigned = dict(value)
    declared_manifest_digest = unsigned.pop("manifest_digest", None)
    rows = value.get("files")
    if (
        set(value) != fields
        or value.get("schema_version") != RELEASE_SCHEMA_VERSION
        or value.get("exact_member_closure") is not True
        or type(rows) is not list
        or value.get("release_sha256") != expected_release
        or declared_manifest_digest != object_sha256(unsigned)
    ):
        fail("release manifest schema/digest differs")
    required = (
        frozenset(test_only_required_files)
        if test_only_required_files is not None
        else REQUIRED_RELEASE_FILES
    )
    observed_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(rows):
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            fail(f"release file row {index} field closure differs")
        relative = item["path"]
        if (
            type(relative) is not str
            or _SAFE_FILE.fullmatch(relative) is None
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or relative in seen
        ):
            fail(f"release file row {index} path differs")
        seen.add(relative)
        expected_file = _sha256(item["sha256"], label=f"release file {relative}")
        path = root / relative
        try:
            resolved = path.resolve(strict=True)
            metadata = path.lstat()
        except OSError as error:
            raise Full30ActionTrainingError(
                f"release member is unavailable: {relative}"
            ) from error
        if (
            resolved != path
            or path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or file_sha256(path) != expected_file
        ):
            fail(f"executed release member differs: {relative}")
        observed_rows.append({"path": relative, "sha256": expected_file})
    if not required.issubset(seen):
        fail(f"release omits required business files: {sorted(required - seen)}")
    if test_only_required_files is None:
        observed_by_path = {
            row["path"]: row["sha256"] for row in observed_rows
        }
        if any(
            observed_by_path.get(path) != expected_sha
            for path, expected_sha in FROZEN_RELEASE_MEMBER_SHA256.items()
        ):
            fail("executed release differs from frozen runtime/step/authority APIs")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"executed release contains symlink: {relative}")
        if stat.S_ISREG(metadata.st_mode):
            actual_files.add(relative)
        elif not stat.S_ISDIR(metadata.st_mode):
            fail(f"executed release contains special entry: {relative}")
    if actual_files != seen:
        fail("executed method root is not the exact release member closure")
    release_payload = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "files": observed_rows,
    }
    if object_sha256(release_payload) != expected_release:
        fail("release SHA-256 differs from recomputed executed closure")
    if test_only_require_current_entrypoint:
        expected_entrypoint = root / "train_full30_action_lora_v1.py"
        if Path(__file__).resolve(strict=True) != expected_entrypoint:
            fail("validated release root is not the currently executed trainer")
    receipt = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "method_root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "release_sha256": expected_release,
        "file_count": len(observed_rows),
        "exact_member_closure_verified": True,
        "business_imports_occurred_before_validation": False,
        "files": observed_rows,
    }
    sealed = _seal(receipt)
    _RELEASE_VALIDATED = True
    _VALIDATED_RELEASE_RECEIPT = sealed
    return sealed


def _release_member_index_v1(
    release_receipt: Mapping[str, Any],
) -> Mapping[str, str]:
    if (
        not _RELEASE_VALIDATED
        or release_receipt is not _VALIDATED_RELEASE_RECEIPT
    ):
        fail("business import receipt is not the executed validated release")
    unsigned = dict(release_receipt)
    declared = unsigned.pop("receipt_digest", None)
    rows = release_receipt.get("files")
    if (
        declared != object_sha256(unsigned)
        or type(rows) is not list
        or release_receipt.get("exact_member_closure_verified") is not True
    ):
        fail("executed release receipt seal differs")
    result: dict[str, str] = {}
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != {"path", "sha256"}
            or row["path"] in result
        ):
            fail("executed release receipt file index differs")
        result[row["path"]] = _sha256(
            row["sha256"], label=f"release member {row['path']}"
        )
    return MappingProxyType(result)


def _verified_module_origin_v1(
    *, module: Any, expected_path: Path, expected_sha256: str, label: str
) -> None:
    module_file = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    try:
        actual_file = Path(module_file).resolve(strict=True)
        actual_origin = Path(spec_origin).resolve(strict=True)
    except (OSError, TypeError) as error:
        raise Full30ActionTrainingError(
            f"{label} has no canonical source origin"
        ) from error
    if (
        actual_file != expected_path
        or actual_origin != expected_path
        or file_sha256(expected_path) != expected_sha256
    ):
        fail(f"{label} source path/SHA differs from executed release")


def _secure_import_release_module_v1(
    *,
    module_name: str,
    relative_path: str,
    release_receipt: Mapping[str, Any],
) -> Any:
    """Load one exact release member and reject all ambient module reuse."""

    index = _release_member_index_v1(release_receipt)
    expected_sha = index.get(relative_path)
    if expected_sha is None:
        fail(f"business import is absent from executed release: {relative_path}")
    root = Path(release_receipt["method_root"])
    expected_path = root / relative_path
    if _plain_absolute_file(expected_path, label=module_name) != expected_path:
        fail(f"business import path differs: {module_name}")

    existing = sys.modules.get(module_name)
    if existing is not None:
        if _BOUND_RELEASE_MODULES.get(module_name) is not existing:
            fail(f"business module was preloaded outside release binder: {module_name}")
        _verified_module_origin_v1(
            module=existing,
            expected_path=expected_path,
            expected_sha256=expected_sha,
            label=module_name,
        )
        return existing

    spec = importlib.util.spec_from_file_location(module_name, expected_path)
    if (
        spec is None
        or spec.loader is None
        or getattr(spec, "origin", None) != str(expected_path)
    ):
        fail(f"cannot construct exact release loader for {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        _verified_module_origin_v1(
            module=module,
            expected_path=expected_path,
            expected_sha256=expected_sha,
            label=module_name,
        )
    except Exception:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    _BOUND_RELEASE_MODULES[module_name] = module
    return module


@dataclass(frozen=True)
class BusinessModulesV1:
    torch: Any
    learning: Any
    branch_runtime: Any
    optimizer: Any
    checkpoint: Any
    training_step: Any
    authority: Any
    amplitude_authority: Any
    canary_authority: Any
    packed_core: Any
    distributed_runtime: Any
    packed_trainer: Any
    packed_release: Any
    source_data: Any
    sigma: Any
    inference_helper: Any
    graft_helper: Any
    psiout_materializer: Any
    legacy: Any


def load_business_modules_v1(
    release_receipt: Optional[Mapping[str, Any]] = None,
) -> BusinessModulesV1:
    """Import every experiment/model dependency only after release validation."""

    global _BUSINESS_MODULES
    if not _RELEASE_VALIDATED:
        fail("business modules are forbidden before executed release validation")
    if release_receipt is None:
        fail("business imports require the exact executed release receipt")
    _release_member_index_v1(release_receipt)
    if _BUSINESS_MODULES is None:
        release_root = str(release_receipt["method_root"])
        sys.path[:] = [item for item in sys.path if item != release_root]
        sys.path.insert(0, release_root)

        def local(name: str) -> Any:
            return _secure_import_release_module_v1(
                module_name=name,
                relative_path=f"{name}.py",
                release_receipt=release_receipt,
            )

        local_names = (
            "full30_action_learning_v1",
            "full30_action_optimizer_v1",
            "full30_action_checkpoint_v1",
            "full30_action_data_teacher_authority_v1",
            "full30_action_amplitude_authority_v1",
            "full30_action_mechanism_canary_authority_v1",
            "packed_preservation_lora_v2",
            "source_self_runtime",
            "train_packed_preservation_lora_v2",
            "packed_preservation_release_v2",
            "clean_source_visual_context_training_v1",
            "inference_sigma_strata",
            "train_lora",
            "dual_conditional_ratio_core",
            "dclr_runtime_contract",
            "motion_residual",
            "source_kv_replay",
            "source_kv_route_batches",
            "infer_lora",
            "infer_source_kv_carrier_oracle",
            "infer_dclr_reward_runtime_smoke",
            "graft_phase_a_native_training_closure_v1",
            "full30_action_runtime_v1",
            "full30_action_training_step_v1",
            "full30_action_psiout_materializer_v1",
        )
        hostile = [
            name
            for name in local_names
            if name in sys.modules and name not in _BOUND_RELEASE_MODULES
        ]
        if hostile:
            fail(f"release business modules were preloaded: {hostile}")
        bound = {name: local(name) for name in local_names}
        _BUSINESS_MODULES = BusinessModulesV1(
            torch=importlib.import_module("torch"),
            learning=bound["full30_action_learning_v1"],
            branch_runtime=bound["full30_action_runtime_v1"],
            optimizer=bound["full30_action_optimizer_v1"],
            checkpoint=bound["full30_action_checkpoint_v1"],
            training_step=bound["full30_action_training_step_v1"],
            authority=bound["full30_action_data_teacher_authority_v1"],
            amplitude_authority=bound["full30_action_amplitude_authority_v1"],
            canary_authority=bound[
                "full30_action_mechanism_canary_authority_v1"
            ],
            packed_core=bound["packed_preservation_lora_v2"],
            distributed_runtime=bound["source_self_runtime"],
            packed_trainer=bound["train_packed_preservation_lora_v2"],
            packed_release=bound["packed_preservation_release_v2"],
            source_data=bound["clean_source_visual_context_training_v1"],
            sigma=bound["inference_sigma_strata"],
            inference_helper=bound["infer_dclr_reward_runtime_smoke"],
            graft_helper=bound["graft_phase_a_native_training_closure_v1"],
            psiout_materializer=bound["full30_action_psiout_materializer_v1"],
            legacy=bound["train_lora"],
        )
    return _BUSINESS_MODULES


@dataclass(frozen=True)
class BoundaryPlanV1:
    profile: str
    start_update: int
    stop_update: int
    checkpoint_updates: tuple[int, ...]
    requires_review_gate: bool
    requires_resume: bool

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": BOUNDARY_SCHEMA_VERSION,
            "profile": self.profile,
            "start_update": self.start_update,
            "stop_update": self.stop_update,
            "checkpoint_updates": list(self.checkpoint_updates),
            "requires_review_gate": self.requires_review_gate,
            "requires_resume": self.requires_resume,
            "maximum_updates": (
                2 if self.profile == "disposable-canary-2" else MAX_UPDATES
            ),
        }
        return MappingProxyType({**value, "boundary_plan_sha256": object_sha256(value)})


def build_boundary_plan_v1(
    profile: str, *, start_update: int, stop_update: int
) -> BoundaryPlanV1:
    if profile not in PROFILES:
        fail("execution profile differs")
    if type(start_update) is not int or type(stop_update) is not int:
        fail("update boundaries must be integers")
    if profile == "disposable-canary-2":
        if (start_update, stop_update) != (0, 2):
            fail("disposable canary is exactly fresh u0/u1/u2")
        return BoundaryPlanV1(profile, 0, 2, CANARY_CHECKPOINTS, False, False)
    if start_update == 0:
        if stop_update != 20:
            fail("fresh formal trajectory is exactly u0->u20")
        checkpoints = FIRST_SEGMENT_CHECKPOINTS
        resume = False
    else:
        if (
            start_update not in REGISTERED_FORMAL_ENDPOINTS[:-1]
            or stop_update != start_update + 20
            or stop_update not in REGISTERED_FORMAL_ENDPOINTS
        ):
            fail("formal resume must advance one registered 20-update segment")
        checkpoints = (stop_update,)
        resume = True
    return BoundaryPlanV1(
        profile, start_update, stop_update, checkpoints, True, resume
    )


@dataclass(frozen=True)
class FreshFormalCanaryInputsV1:
    gate_path: Path
    gate_sha256: str
    receipt_path: Path
    receipt_sha256: str
    u2_reference_path: Path
    u2_reference_sha256: str


def validate_segment_gate_v1(
    *,
    path: str | Path,
    expected_sha256: str,
    plan: BoundaryPlanV1,
    arm: str,
    authority_manifest_sha256: str,
    amplitude_authority_manifest_sha256: str,
    model_sha256: str,
    release_sha256: str,
    resume_reference_sha256: Optional[str],
    fresh_formal_canary_gate_sha256: Optional[str],
) -> Mapping[str, Any]:
    gate_path = _plain_absolute_file(path, label="segment review gate")
    value = _strict_json_file(
        gate_path,
        expected_sha256=expected_sha256,
        label="segment review gate",
    )
    fields = {
        "schema_version",
        "status",
        "arm",
        "profile",
        "start_update",
        "stop_update",
        "boundary_plan_sha256",
        "authority_manifest_sha256",
        "amplitude_authority_manifest_sha256",
        "model_sha256",
        "release_sha256",
        "resume_reference_sha256",
        "fresh_formal_canary_gate_sha256",
        "full81_review_required_before_next_segment",
        "gate_digest",
    }
    unsigned = dict(value)
    gate_digest = unsigned.pop("gate_digest", None)
    plan_receipt = plan.receipt()
    expected = {
        "schema_version": SEGMENT_GATE_SCHEMA_VERSION,
        "status": "approved",
        "arm": arm,
        "profile": plan.profile,
        "start_update": plan.start_update,
        "stop_update": plan.stop_update,
        "boundary_plan_sha256": plan_receipt["boundary_plan_sha256"],
        "authority_manifest_sha256": authority_manifest_sha256,
        "amplitude_authority_manifest_sha256": (
            amplitude_authority_manifest_sha256
        ),
        "model_sha256": model_sha256,
        "release_sha256": release_sha256,
        "resume_reference_sha256": resume_reference_sha256,
        "fresh_formal_canary_gate_sha256": (
            fresh_formal_canary_gate_sha256
        ),
        "full81_review_required_before_next_segment": True,
    }
    if set(value) != fields or unsigned != expected or gate_digest != object_sha256(unsigned):
        fail("segment review gate binding/digest differs")
    return MappingProxyType(value)


_CHECKPOINT_REFERENCE_FIELDS = frozenset(
    {
        "checkpoint_sequence",
        "completed_updates",
        "manifest_sha256",
        "manifest_digest",
        "history_sha256",
        "rng_sha256",
        "schedule_prefix_sha256",
        "trainable_state_sha256",
        "optimizer_v_state_sha256",
    }
)
_CANARY_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "complete",
        "arm",
        "profile",
        "start_update",
        "stop_update",
        "optimizer_updates_executed",
        "optimizer_update_count",
        "boundary_plan",
        "segment_gate",
        "fresh_formal_canary_evidence",
        "bindings",
        "release",
        "authority_validation",
        "amplitude_authority_validation",
        "amplitude_runtime_binding",
        "data_teacher_authority_manifest_sha256",
        "amplitude_authority_manifest_sha256",
        "authority_projections",
        "schedule_full_sha256",
        "schedule_prefix_sha256",
        "noise_authority_sha256",
        "architecture",
        "lora_installation",
        "trainable_parameter_count",
        "authoritative_inventory_sha256",
        "final_optimizer_identity",
        "checkpoints",
        "official_model",
        "objective",
        "distributed",
        "synthetic_target_bytes_read",
        "synthetic_target_index1_bytes_read",
        "parent_allocation_released",
        "receipt_digest",
    }
)


def _closed_dict_v1(value: Any, fields: Iterable[str], *, label: str) -> dict[str, Any]:
    expected = frozenset(fields)
    if type(value) is not dict or set(value) != expected:
        fail(f"{label} field closure differs")
    return dict(value)


def _validate_canary_checkpoint_record_v1(
    *,
    row: Any,
    expected_update: int,
    expected_sequence: int,
    expected_bindings: Mapping[str, Any],
    expected_schedule_full_sha256: str,
    expected_schedule_prefix_sha256: str,
    expected_inventory_sha256: str,
    previous_reference: Optional[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Path, Path]:
    record = _closed_dict_v1(
        row,
        {"completed_updates", "path", "reference_path", "reference"},
        label=f"canary u{expected_update} checkpoint record",
    )
    reference = _closed_dict_v1(
        record["reference"],
        _CHECKPOINT_REFERENCE_FIELDS,
        label=f"canary u{expected_update} reference",
    )
    for field in _CHECKPOINT_REFERENCE_FIELDS - {
        "checkpoint_sequence",
        "completed_updates",
    }:
        _sha256(reference[field], label=f"canary u{expected_update} {field}")
    if (
        record["completed_updates"] != expected_update
        or reference["completed_updates"] != expected_update
        or reference["checkpoint_sequence"] != expected_sequence
        or reference["schedule_prefix_sha256"]
        != expected_schedule_prefix_sha256
    ):
        fail(f"canary u{expected_update} reference boundary differs")
    reference_path = _plain_absolute_file(
        record["reference_path"],
        label=f"canary u{expected_update} reference file",
    )
    reference_file = _strict_json_file(
        reference_path,
        expected_sha256=file_sha256(reference_path),
        label=f"canary u{expected_update} reference file",
    )
    if reference_file != reference:
        fail(f"canary u{expected_update} reference file differs from receipt")
    checkpoint_path = _plain_absolute_directory(
        record["path"], label=f"canary u{expected_update} checkpoint"
    )
    manifest_path = _plain_absolute_file(
        checkpoint_path / "manifest.json",
        label=f"canary u{expected_update} checkpoint manifest",
    )
    manifest = _strict_json_file(
        manifest_path,
        expected_sha256=reference["manifest_sha256"],
        label=f"canary u{expected_update} checkpoint manifest",
    )
    manifest_unsigned = dict(manifest)
    manifest_digest = manifest_unsigned.pop("manifest_digest", None)
    try:
        actual = {
            "completed_updates": manifest["progress"]["completed_updates"],
            "checkpoint_sequence": manifest["checkpoint_sequence"],
            "schedule_full_sha256": manifest["schedule"]["full_sha256"],
            "schedule_prefix_sha256": manifest["schedule"]["prefix_sha256"],
            "inventory_sha256": manifest["capacity"][
                "authoritative_inventory_sha256"
            ],
            "history_sha256": manifest["history"]["sha256"],
            "rng_sha256": manifest["rng"]["sha256"],
            "trainable_state_sha256": manifest["trainables"]["state_sha256"],
            "optimizer_v_state_sha256": manifest["optimizer"]["state_sha256"],
            "optimizer_update_count": manifest["optimizer"]["update_count"],
            "production_capacity_authorized": manifest["capacity"][
                "production_capacity_authorized"
            ],
        }
    except (KeyError, TypeError) as error:
        raise Full30ActionTrainingError(
            f"canary u{expected_update} checkpoint manifest is incomplete"
        ) from error
    expected_actual = {
        "completed_updates": expected_update,
        "checkpoint_sequence": expected_sequence,
        "schedule_full_sha256": expected_schedule_full_sha256,
        "schedule_prefix_sha256": expected_schedule_prefix_sha256,
        "inventory_sha256": expected_inventory_sha256,
        "history_sha256": reference["history_sha256"],
        "rng_sha256": reference["rng_sha256"],
        "trainable_state_sha256": reference["trainable_state_sha256"],
        "optimizer_v_state_sha256": reference["optimizer_v_state_sha256"],
        "optimizer_update_count": expected_update,
        "production_capacity_authorized": True,
    }
    if (
        manifest_digest != object_sha256(manifest_unsigned)
        or manifest_digest != reference["manifest_digest"]
        or actual != expected_actual
        or manifest.get("bindings") != dict(expected_bindings)
        or manifest.get("previous_checkpoint")
        != (None if previous_reference is None else dict(previous_reference))
    ):
        fail(f"canary u{expected_update} checkpoint manifest binding differs")
    return MappingProxyType(reference), checkpoint_path, reference_path


def _validation_digest_v1(value: Any, *, label: str) -> str:
    if not isinstance(value, Mapping):
        fail(f"{label} is not a mapping")
    unsigned = dict(value)
    declared = unsigned.pop("validation_digest", None)
    if declared != object_sha256(unsigned):
        fail(f"{label} digest differs")
    return _sha256(declared, label=f"{label} validation digest")


def _embedded_release_member_sha256_v1(
    release: Any, relative_path: str
) -> str:
    if not isinstance(release, Mapping) or type(release.get("files")) is not list:
        fail("completed canary release inventory is incomplete")
    matches = [
        row.get("sha256")
        for row in release["files"]
        if isinstance(row, Mapping) and row.get("path") == relative_path
    ]
    if len(matches) != 1:
        fail(f"completed canary release member is absent/ambiguous: {relative_path}")
    return _sha256(matches[0], label=f"completed canary {relative_path}")


def _validate_formal_population_receipts_v1(
    *,
    authority_validation: Mapping[str, Any],
    amplitude_validation: Mapping[str, Any],
    authority_manifest_sha256: str,
    amplitude_authority_manifest_sha256: str,
) -> tuple[str, str]:
    data_digest = _validation_digest_v1(
        authority_validation, label="fresh formal data authority validation"
    )
    amplitude_digest = _validation_digest_v1(
        amplitude_validation,
        label="fresh formal amplitude authority validation",
    )
    if (
        authority_validation.get("schema_version")
        != "bernini-full30-action-data-teacher-validation-v3"
        or authority_validation.get("source_counts")
        != {"fit": 64, "confirmation": 16, "heldout": 8}
        or authority_validation.get("pair_counts")
        != {"fit": 128, "confirmation": 32, "heldout": 16}
        or authority_validation.get("teacher_origin_counts")
        != {"fit": 8, "confirmation": 8}
        or authority_validation.get("representation_counts")
        != {"fit": 16, "confirmation": 16}
        or authority_validation.get("synthetic_target_index1_bytes_read") is not False
        or authority_validation.get("optimizer_authorized") is not True
    ):
        fail("fresh formal data authority is not the closed 64/16/8 population")
    if (
        amplitude_validation.get("schema_version")
        != "bernini-full30-action-amplitude-validation-v2"
        or amplitude_validation.get("manifest_file_sha256")
        != amplitude_authority_manifest_sha256
        or amplitude_validation.get("parent_manifest_file_sha256")
        != authority_manifest_sha256
        or amplitude_validation.get("optimizer_bundles") != 16
        or amplitude_validation.get("calibrator_evidence") != 32
        or amplitude_validation.get("frozen_fail_evidence") != 32
        or amplitude_validation.get("sigma_floor_rows") != 96
        or amplitude_validation.get("optimizer_authorized") is not True
    ):
        fail("fresh formal amplitude authority is not the closed formal population")
    return data_digest, amplitude_digest


def validate_reduced_canary_to_fresh_formal_v1(
    *,
    inputs: FreshFormalCanaryInputsV1,
    arm: str,
    formal_plan: BoundaryPlanV1,
    release_sha256: str,
    model_sha256: str,
    authority_manifest_sha256: str,
    amplitude_authority_manifest_sha256: str,
    authority_projections: Mapping[str, str],
    noise_authority_sha256: str,
    schedule_full_sha256: str,
    schedule_prefix_u0_sha256: str,
    architecture: Mapping[str, Any],
    authoritative_inventory_sha256: str,
    trainable_parameter_count: int,
    bindings: Mapping[str, Any],
    formal_authority_validation: Mapping[str, Any],
    formal_amplitude_validation: Mapping[str, Any],
    optimizer_code_sha256: str,
    training_step_code_sha256: str,
) -> Mapping[str, Any]:
    """Validate a reduced-canary review as a transition to independent formal u0.

    Canary checkpoints are reopened as evidence only.  Their parameter,
    optimizer, RNG, and schedule authorities remain canary-local and are never
    accepted as formal resume inputs.
    """

    if (
        arm not in ARMS
        or formal_plan.profile != "review-gated-segment"
        or (formal_plan.start_update, formal_plan.stop_update) != (0, 20)
        or formal_plan.requires_resume is not False
        or type(trainable_parameter_count) is not int
        or trainable_parameter_count != 188_946_432
    ):
        fail("reduced canary transition was requested outside fresh formal u0->u20")
    for label, value in (
        ("release", release_sha256),
        ("model", model_sha256),
        ("formal authority", authority_manifest_sha256),
        ("formal amplitude authority", amplitude_authority_manifest_sha256),
        ("formal noise authority", noise_authority_sha256),
        ("formal schedule", schedule_full_sha256),
        ("formal u0 schedule prefix", schedule_prefix_u0_sha256),
        ("optimizer code", optimizer_code_sha256),
        ("training-step code", training_step_code_sha256),
        ("authoritative inventory", authoritative_inventory_sha256),
    ):
        _sha256(value, label=f"{label} SHA-256")
    formal_projection = _closed_dict_v1(
        dict(authority_projections),
        {
            "data_sha256",
            "parent_teacher_sha256",
            "amplitude_manifest_sha256",
            "amplitude_validation_digest",
            "teacher_sha256",
            "nuisance_sha256",
        },
        label="fresh formal authority projection",
    )
    for field, value in formal_projection.items():
        _sha256(value, label=f"fresh formal {field}")
    if formal_projection["amplitude_manifest_sha256"] != amplitude_authority_manifest_sha256:
        fail("fresh formal amplitude projection binding differs")
    formal_data_validation_digest, formal_amplitude_validation_digest = (
        _validate_formal_population_receipts_v1(
            authority_validation=formal_authority_validation,
            amplitude_validation=formal_amplitude_validation,
            authority_manifest_sha256=authority_manifest_sha256,
            amplitude_authority_manifest_sha256=(
                amplitude_authority_manifest_sha256
            ),
        )
    )
    if (
        formal_projection["amplitude_validation_digest"]
        != formal_amplitude_validation_digest
    ):
        fail("fresh formal amplitude validation projection differs")

    canary_plan = build_boundary_plan_v1(
        "disposable-canary-2", start_update=0, stop_update=2
    )
    receipt = _strict_json_file(
        inputs.receipt_path,
        expected_sha256=inputs.receipt_sha256,
        label="completed same-arm reduced canary receipt",
    )
    _closed_dict_v1(receipt, _CANARY_RECEIPT_FIELDS, label="reduced canary receipt")
    _validate_sealed_receipt_v1(receipt, label="reduced canary receipt")
    canary_bindings = receipt.get("bindings")
    canary_projection_value = receipt.get("authority_projections")
    receipt_release = receipt.get("release")
    final_optimizer = receipt.get("final_optimizer_identity")
    official_model = receipt.get("official_model")
    canary_architecture = receipt.get("architecture")
    canary_data_validation = receipt.get("authority_validation")
    canary_amplitude_validation = receipt.get("amplitude_authority_validation")
    if not all(
        isinstance(value, Mapping)
        for value in (
            canary_bindings,
            canary_projection_value,
            receipt_release,
            final_optimizer,
            official_model,
            canary_architecture,
            canary_data_validation,
            canary_amplitude_validation,
        )
    ):
        fail("completed reduced canary receipt is incomplete")
    canary_projection = _closed_dict_v1(
        dict(canary_projection_value),
        {
            "data_sha256",
            "parent_teacher_sha256",
            "amplitude_manifest_sha256",
            "amplitude_validation_digest",
            "teacher_sha256",
            "nuisance_sha256",
        },
        label="reduced canary authority projection",
    )
    for field, value in canary_projection.items():
        _sha256(value, label=f"reduced canary {field}")
    canary_data_validation_digest = _validation_digest_v1(
        canary_data_validation, label="reduced canary data authority validation"
    )
    canary_amplitude_validation_digest = _validation_digest_v1(
        canary_amplitude_validation,
        label="reduced canary amplitude authority validation",
    )
    if (
        canary_data_validation.get("schema_version")
        != "bernini-full30-action-mechanism-canary-validation-v1"
        or canary_data_validation.get("population_profile")
        != "same_origin_two_seed_mechanism_only_v1"
        or canary_data_validation.get("formal_authority") is not False
        or canary_data_validation.get("mechanism_only") is not True
        or canary_data_validation.get("same_origin_profile_verified") is not True
        or canary_data_validation.get("shared_origin_identities") != 1
        or canary_data_validation.get("generalization") is not False
        or canary_data_validation.get("identity_generalization") is not False
        or canary_data_validation.get("event_family_generalization") is not False
        or canary_data_validation.get("optimizer_authorized") is not True
        or canary_data_validation.get("synthetic_target_bytes_read") is not False
        or canary_amplitude_validation.get("schema_version")
        != "bernini-full30-action-mechanism-canary-amplitude-validation-v1"
        or canary_amplitude_validation.get("population_profile")
        != "same_origin_two_seed_mechanism_only_v1"
        or canary_amplitude_validation.get("formal_authority") is not False
        or canary_amplitude_validation.get("mechanism_only") is not True
        or canary_amplitude_validation.get("generalization") is not False
        or canary_amplitude_validation.get("identity_generalization") is not False
        or canary_amplitude_validation.get("event_family_generalization") is not False
        or canary_amplitude_validation.get("optimizer_authorized") is not True
        or canary_data_validation.get("manifest_file_sha256")
        != receipt.get("data_teacher_authority_manifest_sha256")
        or canary_amplitude_validation.get("manifest_file_sha256")
        != receipt.get("amplitude_authority_manifest_sha256")
        or canary_amplitude_validation.get("parent_manifest_file_sha256")
        != receipt.get("data_teacher_authority_manifest_sha256")
        or canary_projection["amplitude_manifest_sha256"]
        != receipt.get("amplitude_authority_manifest_sha256")
        or canary_projection["amplitude_validation_digest"]
        != canary_amplitude_validation_digest
    ):
        fail("completed reduced canary authority is not mechanism-only")
    canary_optimizer_code_sha256 = _embedded_release_member_sha256_v1(
        receipt_release, "full30_action_optimizer_v1.py"
    )
    canary_training_step_code_sha256 = _embedded_release_member_sha256_v1(
        receipt_release, "full30_action_training_step_v1.py"
    )
    shared_binding_fields = (
        "arm",
        "release_sha256",
        "model_sha256",
        "runtime_sha256",
        "objective_sha256",
    )
    try:
        shared_bindings_match = all(
            canary_bindings[field] == bindings[field]
            for field in shared_binding_fields
        )
        canary_authority_bindings_match = (
            canary_bindings["data_sha256"] == canary_projection["data_sha256"]
            and canary_bindings["teacher_sha256"]
            == canary_projection["teacher_sha256"]
            and canary_bindings["nuisance_sha256"]
            == canary_projection["nuisance_sha256"]
            and canary_bindings["noise_sha256"]
            == receipt["noise_authority_sha256"]
        )
    except (KeyError, TypeError) as error:
        raise Full30ActionTrainingError(
            "completed reduced canary checkpoint bindings are incomplete"
        ) from error
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("method") != METHOD
        or receipt.get("complete") is not True
        or receipt.get("arm") != arm
        or receipt.get("profile") != "disposable-canary-2"
        or (receipt.get("start_update"), receipt.get("stop_update")) != (0, 2)
        or receipt.get("optimizer_updates_executed") != 2
        or receipt.get("optimizer_update_count") != 2
        or receipt.get("boundary_plan") != dict(canary_plan.receipt())
        or receipt.get("segment_gate") is not None
        or receipt.get("fresh_formal_canary_evidence") is not None
        or receipt_release.get("release_sha256") != release_sha256
        or not shared_bindings_match
        or not canary_authority_bindings_match
        or canary_optimizer_code_sha256 != optimizer_code_sha256
        or canary_training_step_code_sha256 != training_step_code_sha256
        or official_model.get("model_sha256") != model_sha256
        or canary_architecture.get("digest") != architecture.get("digest")
        or receipt.get("trainable_parameter_count") != trainable_parameter_count
        or receipt.get("authoritative_inventory_sha256")
        != authoritative_inventory_sha256
        or final_optimizer.get("update_count") != 2
        or final_optimizer.get("inventory_sha256")
        != authoritative_inventory_sha256
        or receipt.get("synthetic_target_index1_bytes_read") is not False
        or receipt.get("synthetic_target_bytes_read") is not False
    ):
        fail("completed reduced canary does not match shared execution identity")

    canary_schedule_full_sha256 = _sha256(
        receipt.get("schedule_full_sha256"),
        label="reduced canary schedule full SHA-256",
    )
    canary_inventory_sha256 = _sha256(
        receipt.get("authoritative_inventory_sha256"),
        label="reduced canary inventory SHA-256",
    )
    checkpoint_rows = receipt.get("checkpoints")
    if type(checkpoint_rows) is not list or len(checkpoint_rows) != 3:
        fail("completed reduced canary does not close u0/u1/u2 checkpoints")
    references: list[Mapping[str, Any]] = []
    checkpoint_paths: list[Path] = []
    reference_paths: list[Path] = []
    canary_prefixes: dict[str, str] = {}
    for update, row in enumerate(checkpoint_rows):
        try:
            declared_prefix = row["reference"]["schedule_prefix_sha256"]
        except (KeyError, TypeError) as error:
            raise Full30ActionTrainingError(
                f"reduced canary u{update} schedule prefix is absent"
            ) from error
        declared_prefix = _sha256(
            declared_prefix, label=f"reduced canary u{update} schedule prefix"
        )
        reference, checkpoint_path, reference_path = (
            _validate_canary_checkpoint_record_v1(
                row=row,
                expected_update=update,
                expected_sequence=update,
                expected_bindings=canary_bindings,
                expected_schedule_full_sha256=canary_schedule_full_sha256,
                expected_schedule_prefix_sha256=declared_prefix,
                expected_inventory_sha256=canary_inventory_sha256,
                previous_reference=(None if update == 0 else references[-1]),
            )
        )
        references.append(reference)
        checkpoint_paths.append(checkpoint_path)
        reference_paths.append(reference_path)
        canary_prefixes[str(update)] = declared_prefix
    if receipt.get("schedule_prefix_sha256") != canary_prefixes["2"]:
        fail("completed reduced canary terminal schedule prefix differs")
    u2_reference = references[2]
    if (
        reference_paths[2] != inputs.u2_reference_path
        or file_sha256(inputs.u2_reference_path) != inputs.u2_reference_sha256
    ):
        fail("completed reduced canary u2 reference path/SHA differs")

    canary_evidence = {
        "profile": canary_plan.profile,
        "population_profile": "same_origin_two_seed_mechanism_only_v1",
        "boundary_plan_sha256": canary_plan.receipt()["boundary_plan_sha256"],
        "authority_manifest_sha256": receipt[
            "data_teacher_authority_manifest_sha256"
        ],
        "amplitude_authority_manifest_sha256": receipt[
            "amplitude_authority_manifest_sha256"
        ],
        "authority_projections": dict(canary_projection),
        "data_validation_digest": canary_data_validation_digest,
        "amplitude_validation_digest": canary_amplitude_validation_digest,
        "noise_authority_sha256": receipt["noise_authority_sha256"],
        "schedule_full_sha256": canary_schedule_full_sha256,
        "schedule_prefix_sha256_by_update": canary_prefixes,
        "receipt_path": str(inputs.receipt_path),
        "receipt_sha256": inputs.receipt_sha256,
        "receipt_digest": receipt["receipt_digest"],
        "u2_checkpoint_path": str(checkpoint_paths[2]),
        "u2_reference_path": str(inputs.u2_reference_path),
        "u2_reference_sha256": inputs.u2_reference_sha256,
        "u2_reference_digest": object_sha256(u2_reference),
        "completed_updates": 2,
        "review_outcome": "GO",
    }
    formal_evidence = {
        "profile": formal_plan.profile,
        "start_update": 0,
        "stop_update": 20,
        "boundary_plan_sha256": formal_plan.receipt()["boundary_plan_sha256"],
        "authority_manifest_sha256": authority_manifest_sha256,
        "amplitude_authority_manifest_sha256": amplitude_authority_manifest_sha256,
        "authority_projections": formal_projection,
        "data_validation_digest": formal_data_validation_digest,
        "amplitude_validation_digest": formal_amplitude_validation_digest,
        "noise_authority_sha256": noise_authority_sha256,
        "schedule_full_sha256": schedule_full_sha256,
        "schedule_prefix_u0_sha256": schedule_prefix_u0_sha256,
        "source_population": {"fit": 64, "confirmation": 16, "heldout": 8},
        "optimizer_pair_population": 128,
        "optimizer_teacher_population": 16,
        "optimizer_amplitude_population": 16,
        "formal_authority": True,
    }
    shared_execution_identity = {
        "arm": arm,
        "release_sha256": release_sha256,
        "model_sha256": model_sha256,
        "runtime_code_sha256": bindings["runtime_sha256"],
        "objective_code_sha256": bindings["objective_sha256"],
        "optimizer_code_sha256": optimizer_code_sha256,
        "training_step_code_sha256": training_step_code_sha256,
        "architecture_digest": architecture["digest"],
        "authoritative_inventory_sha256": authoritative_inventory_sha256,
        "trainable_parameter_count": trainable_parameter_count,
    }
    fresh_start = {
        "formal_start_update": 0,
        "formal_requires_resume": False,
        "canary_checkpoint_manifest_only_reopened": True,
        "canary_checkpoint_loaded": False,
        "canary_trainable_state_loaded": False,
        "canary_optimizer_state_loaded": False,
        "canary_rng_state_loaded": False,
        "canary_schedule_used_for_formal": False,
    }
    expected_gate = {
        "schema_version": REDUCED_CANARY_TO_FRESH_FORMAL_GATE_SCHEMA_VERSION,
        "status": "GO",
        "transition": "reduced_canary_to_fresh_formal_v1",
        "arm": arm,
        "canary": canary_evidence,
        "formal": formal_evidence,
        "shared_execution_identity": shared_execution_identity,
        "fresh_start": fresh_start,
    }
    gate = _strict_json_file(
        inputs.gate_path,
        expected_sha256=inputs.gate_sha256,
        label="reduced canary to fresh formal gate",
    )
    gate_unsigned = _closed_dict_v1(
        gate,
        {*expected_gate, "gate_digest"},
        label="reduced canary to fresh formal gate",
    )
    declared_gate_digest = gate_unsigned.pop("gate_digest")
    if (
        gate_unsigned != expected_gate
        or declared_gate_digest != object_sha256(gate_unsigned)
    ):
        fail("reduced canary to fresh formal gate binding/digest differs")
    evidence = {
        "schema_version": REDUCED_CANARY_TO_FRESH_FORMAL_EVIDENCE_SCHEMA_VERSION,
        "transition": "reduced_canary_to_fresh_formal_v1",
        "arm": arm,
        "canary": canary_evidence,
        "formal": formal_evidence,
        "shared_execution_identity": shared_execution_identity,
        "fresh_start": fresh_start,
        "gate_path": str(inputs.gate_path),
        "gate_sha256": inputs.gate_sha256,
        "gate_digest": declared_gate_digest,
        "canary_and_formal_authorities_independently_validated": True,
        "canary_state_is_not_formal_initial_state": True,
    }
    return _seal(evidence)


# Retained as an API alias; semantics are the reduced->fresh transition above.
validate_fresh_formal_canary_evidence_v1 = (
    validate_reduced_canary_to_fresh_formal_v1
)


@dataclass(frozen=True)
class TeacherPacketV1:
    teacher_unit: Any = field(repr=False, compare=False)
    minimum_amplitude: Any = field(repr=False, compare=False)
    minimum_amplitude_float32_le_sha256: str
    minimum_amplitude_bundle_digest: str
    minimum_amplitude_calibration_id: str
    nuisance_packet: Any = field(repr=False, compare=False)
    authority_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedRecordV1:
    schedule_row: Any
    runtime_record: Any = field(repr=False, compare=False)
    teacher: TeacherPacketV1 = field(repr=False, compare=False)
    step_record: Any = field(repr=False, compare=False)
    authority_receipt: Mapping[str, Any] = field(default_factory=dict)


def _validate_prepared_step_records_v1(
    records: Sequence[PreparedRecordV1], *, training_step_module: Any
) -> tuple[PreparedRecordV1, ...]:
    values = tuple(records)
    if len(values) != LOCAL_RECORDS or any(
        not isinstance(item, PreparedRecordV1) for item in values
    ):
        fail("one local DP rank requires exactly four prepared records")
    if any(
        not isinstance(
            item.step_record, training_step_module.Full30LocalMicroRecordV1
        )
        for item in values
    ):
        fail("prepared training-step record type differs")
    return values


def _validate_sealed_receipt_v1(
    receipt: Mapping[str, Any], *, label: str
) -> None:
    if not isinstance(receipt, Mapping):
        fail(f"{label} is not a mapping")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if declared != object_sha256(unsigned):
        fail(f"{label} digest differs")


def execute_one_update_v1(
    *,
    arm: str,
    records: Sequence[PreparedRecordV1],
    runtime: Any,
    optimizer: Any,
    full_schedule: Sequence[Any],
    rank: int,
    training_step_module: Any,
    gradient_mean: Callable[[Any], Mapping[str, Any]],
    world_consensus: Callable[[Any], Mapping[str, Any]],
    optimizer_all_reduce_sum: Callable[[Any], Optional[Any]],
    autocast_context: Callable[[], Any] = nullcontext,
    test_only_allow_small_capacity: bool = False,
) -> Any:
    """Thinly invoke the sole registered scientific update implementation."""

    if arm not in ARMS:
        fail("formal arm differs")
    prepared = _validate_prepared_step_records_v1(
        records, training_step_module=training_step_module
    )
    before = getattr(optimizer, "update_count", None)
    try:
        result = training_step_module.execute_full30_action_training_step_v1(
            runtime=runtime,
            optimizer=optimizer,
            arm=arm,
            rank=rank,
            update_index=before,
            full_schedule=full_schedule,
            local_records=tuple(item.step_record for item in prepared),
            gradient_mean=gradient_mean,
            world_consensus=world_consensus,
            optimizer_all_reduce_sum=optimizer_all_reduce_sum,
            autocast_context=autocast_context,
            test_only_allow_small_capacity=test_only_allow_small_capacity,
        )
    except Exception:
        if getattr(optimizer, "update_count", None) != before:
            fail("failed training step changed the optimizer update count")
        raise
    if not isinstance(
        result, training_step_module.Full30ActionTrainingStepResultV1
    ):
        fail("training-step result type differs")
    try:
        canonical_step = training_step_module.canonical_receipt_bytes(
            result.receipt
        )
    except Exception as error:
        raise Full30ActionTrainingError(
            "training-step receipt is not canonically sealed"
        ) from error
    if canonical_step != canonical_json_bytes(dict(result.receipt)):
        fail("training-step canonical receipt bytes differ")
    _validate_sealed_receipt_v1(
        result.optimizer_receipt, label="optimizer receipt"
    )
    embedded = result.receipt.get("optimizer")
    if (
        getattr(optimizer, "update_count", None) != before + 1
        or result.receipt.get("status") != "committed"
        or result.receipt.get("arm") != arm
        or result.receipt.get("update_count_before") != before
        or result.receipt.get("update_count_after") != before + 1
        or not isinstance(embedded, Mapping)
        or embedded.get("step_call_count") != 1
        or embedded.get("receipt_digest")
        != result.optimizer_receipt.get("receipt_digest")
        or result.optimizer_receipt.get("arm") != arm
        or result.optimizer_receipt.get("update_count_before") != before
        or result.optimizer_receipt.get("update_count_after") != before + 1
    ):
        fail("training-step and optimizer receipts are not exactly bound")
    return result


def gather_global_update_receipt_v1(
    *,
    step_result: Any,
    prepared_records: Sequence[PreparedRecordV1],
    parallel: Any,
    modules: BusinessModulesV1,
) -> Mapping[str, Any]:
    """Aggregate per-rank sealed step evidence without reimplementing it."""

    if not isinstance(
        step_result, modules.training_step.Full30ActionTrainingStepResultV1
    ):
        fail("global receipt input is not a training-step result")
    prepared = _validate_prepared_step_records_v1(
        prepared_records, training_step_module=modules.training_step
    )
    local = {
        "rank": parallel.contract.rank,
        "step_receipt": dict(step_result.receipt),
        "optimizer_receipt": dict(step_result.optimizer_receipt),
        "record_authority": [
            dict(item.authority_receipt) for item in prepared
        ],
    }
    local["envelope_digest"] = object_sha256(local)
    torch_dist = importlib.import_module("torch.distributed")
    gathered: list[Any] = [None] * WORLD_SIZE
    torch_dist.all_gather_object(
        gathered, local, group=parallel.world_group
    )

    for rank, envelope in enumerate(gathered):
        if not isinstance(envelope, Mapping):
            fail("WORLD8 training-step envelope is not a mapping")
        unsigned = dict(envelope)
        digest = unsigned.pop("envelope_digest", None)
        if (
            set(envelope)
            != {
                "rank",
                "step_receipt",
                "optimizer_receipt",
                "record_authority",
                "envelope_digest",
            }
            or envelope.get("rank") != rank
            or digest != object_sha256(unsigned)
        ):
            fail("WORLD8 training-step envelope digest/rank differs")
        step_receipt = envelope["step_receipt"]
        try:
            modules.training_step.canonical_receipt_bytes(step_receipt)
        except Exception as error:
            raise Full30ActionTrainingError(
                "gathered training-step receipt is invalid"
            ) from error
        _validate_sealed_receipt_v1(
            envelope["optimizer_receipt"],
            label=f"rank {rank} optimizer receipt",
        )
        if (
            step_receipt.get("rank") != rank
            or step_receipt.get("optimizer", {}).get("receipt_digest")
            != envelope["optimizer_receipt"].get("receipt_digest")
            or type(envelope["record_authority"]) is not list
            or len(envelope["record_authority"]) != LOCAL_RECORDS
        ):
            fail("WORLD8 step/optimizer/record envelope binding differs")

    optimizer_bytes = [
        canonical_json_bytes(envelope["optimizer_receipt"])
        for envelope in gathered
    ]
    if len(set(optimizer_bytes)) != 1:
        fail("WORLD8 optimizer receipts are not exactly equal")
    optimizer_receipt = dict(gathered[0]["optimizer_receipt"])

    for first in (0, SP_SIZE):
        leader = gathered[first]
        leader_step = leader["step_receipt"]
        leader_authority = leader["record_authority"]
        for rank in range(first + 1, first + SP_SIZE):
            replica = gathered[rank]
            replica_step = replica["step_receipt"]
            common_fields = (
                "arm",
                "update_count_before",
                "update_count_after",
                "schedule",
                "inventory",
                "records",
                "noop_replay_records",
                "runtime",
                "world_consensus",
                "optimizer",
                "objective_contract",
            )
            if any(
                replica_step.get(field) != leader_step.get(field)
                for field in common_fields
            ) or replica["record_authority"] != leader_authority:
                fail("SP4 replica step evidence differs")
            for field in ("action_sha256", "noop_sha256", "coverage_gate"):
                if replica_step.get("gradients", {}).get(field) != leader_step.get(
                    "gradients", {}
                ).get(field):
                    fail("SP4 synchronized gradient evidence differs")

    leaders = (gathered[0], gathered[SP_SIZE])
    first_step, second_step = (item["step_receipt"] for item in leaders)
    common_global_fields = (
        "arm",
        "world_size",
        "dp_size",
        "sp_size",
        "update_count_before",
        "update_count_after",
        "global_batch",
        "inventory",
        "runtime",
        "world_consensus",
        "optimizer",
        "objective_contract",
    )
    if any(
        first_step.get(field) != second_step.get(field)
        for field in common_global_fields
    ):
        fail("DP2 leader global step evidence differs")
    for field in ("action_sha256", "noop_sha256", "coverage_gate"):
        if first_step.get("gradients", {}).get(field) != second_step.get(
            "gradients", {}
        ).get(field):
            fail("DP2 synchronized gradient evidence differs")

    global_records: list[Mapping[str, Any]] = []
    global_noop_records: list[Mapping[str, Any]] = []
    for leader in leaders:
        step_receipt = leader["step_receipt"]
        authority_rows = leader["record_authority"]
        step_rows = step_receipt.get("records")
        if type(step_rows) is not list or len(step_rows) != LOCAL_RECORDS:
            fail("DP leader action record closure differs")
        for step_row, authority_row in zip(step_rows, authority_rows):
            if (
                step_row.get("global_index")
                != authority_row.get("global_index")
                or step_row.get("row_id") != authority_row.get("row_id")
                or step_row.get("objective_authority_digest")
                != authority_row.get("objective_authority_digest")
            ):
                fail("prepared and executed record authority differs")
            global_records.append(
                {
                    "step": dict(step_row),
                    "preparation_authority": dict(authority_row),
                }
            )
        noop_rows = step_receipt.get("noop_replay_records")
        expected_noop = LOCAL_RECORDS if step_receipt["arm"] == "action+retain" else 0
        if type(noop_rows) is not list or len(noop_rows) != expected_noop:
            fail("DP leader noop replay record closure differs")
        global_noop_records.extend(dict(item) for item in noop_rows)
    global_records.sort(key=lambda item: item["step"]["global_index"])
    global_noop_records.sort(key=lambda item: item["global_index"])
    global_indices = [item["step"]["global_index"] for item in global_records]
    start = first_step["update_count_before"] * GLOBAL_BATCH
    if global_indices != list(range(start, start + GLOBAL_BATCH)):
        fail("global update record indices do not close one exact batch")
    if len({item["step"]["row_id"] for item in global_records}) != GLOBAL_BATCH:
        fail("global update contains duplicate record rows")

    value = {
        "schema_version": UPDATE_RECEIPT_SCHEMA_VERSION,
        "training_step_schema_version": modules.training_step.SCHEMA_VERSION,
        "status": "committed",
        "arm": first_step["arm"],
        "schedule_update": first_step["update_count_before"],
        "completed_update": first_step["update_count_after"],
        "global_record_count": GLOBAL_BATCH,
        "global_physical_evaluations": first_step["runtime"][
            "formal_physical_evaluation_count"
        ],
        "records": global_records,
        "noop_replay_records": global_noop_records,
        "dp_leader_step_receipts": [
            dict(first_step),
            dict(second_step),
        ],
        "rank_step_receipt_digests": [
            envelope["step_receipt"]["receipt_digest"]
            for envelope in gathered
        ],
        "rank_envelope_digests": [
            envelope["envelope_digest"] for envelope in gathered
        ],
        "action_gradient_sha256": first_step["gradients"]["action_sha256"],
        "noop_gradient_sha256": first_step["gradients"]["noop_sha256"],
        "action_gradient_audit": first_step["gradients"]["coverage_gate"],
        "optimizer_receipt": optimizer_receipt,
        "SP4_replica_evidence_equal": True,
        "DP2_global_batch_closed": True,
        "WORLD8_optimizer_receipts_exactly_equal": True,
    }
    return _seal(value)


def _authority_projection_digests(
    manifest: Mapping[str, Any],
    *,
    amplitude_manifest_sha256: str,
    amplitude_validation_digest: str,
) -> Mapping[str, str]:
    """Derive closed data/teacher/nuisance identities from one admitted manifest."""

    data_payload = {
        "schema_version": AUTHORITY_PROJECTION_SCHEMA_VERSION,
        "kind": "eligible-source-and-pair-authority",
        "sources": [
            {
                "source_iid": row["source_iid"],
                "analysis_split": row["analysis_split"],
                "source_video_sha256": row["source_video_sha256"],
                "source_posterior_index0_sha256": row[
                    "source_posterior_index0_sha256"
                ],
                "source_digest": row["source_digest"],
            }
            for row in manifest["sources"]
        ],
        "pairs": [
            {
                "pair_id": row["pair_id"],
                "analysis_split": row["analysis_split"],
                "source_iid": row["source_iid"],
                "branch": row["branch"],
                "teacher_cell_id": row["teacher_cell_id"],
                "instruction_utf8_sha256": row["instruction_utf8_sha256"],
                "pair_digest": row["pair_digest"],
            }
            for row in manifest["pairs"]
        ],
    }
    teacher_payload = {
        "schema_version": AUTHORITY_PROJECTION_SCHEMA_VERSION,
        "kind": "reviewed-psiout-teacher-authority",
        "teacher_origins": [
            {
                "teacher_cell_id": row["teacher_cell_id"],
                "analysis_split": row["analysis_split"],
                "origin_digest": row["origin_digest"],
            }
            for row in manifest["teacher_origins"]
        ],
        "representation_admissions": [
            {
                "admission_id": row["admission_id"],
                "teacher_cell_id": row["teacher_cell_id"],
                "analysis_split": row["analysis_split"],
                "branch": row["branch"],
                "origin_psiout_sidecar_sha256": row["origin_evidence"][
                    "psiout_sidecar_sha256"
                ],
                "admission_digest": row["admission_digest"],
            }
            for row in manifest["representation_admissions"]
        ],
    }
    if "teacher_seed_bindings" in manifest:
        teacher_payload["population_profile"] = manifest.get(
            "population_profile"
        )
        teacher_payload["teacher_seed_bindings"] = [
            {
                "teacher_cell_id": row["teacher_cell_id"],
                "origin_iid": row["origin_iid"],
                "generation_seed": row["generation_seed"],
                "candidate_bindings": [
                    {
                        "branch": candidate["branch"],
                        "latent_authority_receipt_file_sha256": candidate[
                            "latent_authority_receipt_file_sha256"
                        ],
                        "latent_authority_receipt_digest": candidate[
                            "latent_authority_receipt_digest"
                        ],
                        "candidate_envelope_file_sha256": candidate[
                            "candidate_envelope_file_sha256"
                        ],
                        "native_receipt_file_sha256": candidate[
                            "native_receipt_file_sha256"
                        ],
                        "native_receipt_digest": candidate[
                            "native_receipt_digest"
                        ],
                        "materialization_record_receipt_file_sha256": candidate[
                            "materialization_record_receipt_file_sha256"
                        ],
                        "materialization_record_receipt_digest": candidate[
                            "materialization_record_receipt_digest"
                        ],
                        "candidate_binding_digest": candidate[
                            "candidate_binding_digest"
                        ],
                    }
                    for candidate in row["candidate_bindings"]
                ],
                "binding_digest": row["binding_digest"],
            }
            for row in manifest["teacher_seed_bindings"]
        ]
    nuisance_payload = {
        "schema_version": AUTHORITY_PROJECTION_SCHEMA_VERSION,
        "kind": "reviewed-camera-appearance-nuisance-authority",
        "packets": [
            {
                "admission_id": row["admission_id"],
                "teacher_cell_id": row["teacher_cell_id"],
                "branch": row["branch"],
                "origin_nuisance_packet_sha256": row["origin_evidence"][
                    "nuisance_packet_sha256"
                ],
                "cross_nuisance_packet_sha256": row["cross_anchor_evidence"][
                    "nuisance_packet_sha256"
                ],
            }
            for row in manifest["representation_admissions"]
        ],
    }
    parent_teacher_sha = object_sha256(teacher_payload)
    amplitude_sha = _sha256(
        amplitude_manifest_sha256,
        label="amplitude authority manifest SHA-256",
    )
    amplitude_validation_sha = _sha256(
        amplitude_validation_digest,
        label="amplitude authority validation digest",
    )
    composite_payload = {
        "schema_version": TEACHER_COMPOSITE_SCHEMA_VERSION,
        "parent_teacher_projection_sha256": parent_teacher_sha,
        "amplitude_manifest_file_sha256": amplitude_sha,
        "amplitude_validation_digest": amplitude_validation_sha,
    }
    return MappingProxyType(
        {
            "data_sha256": object_sha256(data_payload),
            "parent_teacher_sha256": parent_teacher_sha,
            "amplitude_manifest_sha256": amplitude_sha,
            "amplitude_validation_digest": amplitude_validation_sha,
            "teacher_sha256": object_sha256(composite_payload),
            "nuisance_sha256": object_sha256(nuisance_payload),
        }
    )


class Full30AuthorityRuntimeIndexV1:
    """Runtime view of the already admitted full30 authority manifest."""

    def __init__(
        self,
        *,
        path: str | Path,
        expected_sha256: str,
        amplitude_path: str | Path,
        expected_amplitude_sha256: str,
        modules: BusinessModulesV1,
        expected_data_sha256: str,
        expected_teacher_sha256: str,
        expected_nuisance_sha256: str,
        profile: str = "review-gated-segment",
    ) -> None:
        if profile not in PROFILES:
            fail("authority runtime profile differs")
        self.profile = profile
        self.is_disposable_canary = profile == "disposable-canary-2"
        self.path = _plain_absolute_file(path, label="full30 authority manifest")
        self.manifest_sha256 = _sha256(
            expected_sha256, label="authority manifest expected SHA-256"
        )
        self.amplitude_path = _plain_absolute_file(
            amplitude_path, label="full30 amplitude authority manifest"
        )
        self.amplitude_manifest_sha256 = _sha256(
            expected_amplitude_sha256,
            label="amplitude authority manifest expected SHA-256",
        )
        if self.is_disposable_canary:
            admitted = modules.canary_authority.load_mechanism_canary_authority_v1(
                manifest_path=self.path,
                expected_manifest_sha256=self.manifest_sha256,
                amplitude_manifest_path=self.amplitude_path,
                expected_amplitude_manifest_sha256=(
                    self.amplitude_manifest_sha256
                ),
                materializer_module=modules.psiout_materializer,
                checkpoint_module=modules.checkpoint,
                learning_module=modules.learning,
            )
            self.validation_receipt = admitted.data.validation_receipt
            manifest = dict(admitted.data.manifest)
            self.amplitude = admitted.amplitude
        else:
            self.validation_receipt = modules.authority.validate_manifest_file(
                self.path, self.manifest_sha256
            )
            manifest = _strict_json_file(
                self.path,
                expected_sha256=self.manifest_sha256,
                label="validated full30 authority manifest",
            )
            self.amplitude = modules.amplitude_authority.load_amplitude_authority_v1(
                manifest_path=self.amplitude_path,
                expected_manifest_sha256=self.amplitude_manifest_sha256,
                parent_manifest_path=self.path,
                expected_parent_manifest_sha256=self.manifest_sha256,
            )
        self.manifest = MappingProxyType(manifest)
        if (
            self.amplitude.parent_manifest_file_sha256
            != self.manifest_sha256
        ):
            fail("amplitude authority parent manifest binding differs")
        amplitude_manifest = _strict_json_file(
            self.amplitude_path,
            expected_sha256=self.amplitude_manifest_sha256,
            label="validated full30 amplitude authority manifest",
        )
        amplitude_runtime = amplitude_manifest.get("frozen_runtime_identity")
        if (
            type(amplitude_runtime) is not dict
            or amplitude_runtime.get("runtime_digest")
            != self.amplitude.frozen_runtime_digest
        ):
            fail("amplitude authority frozen runtime identity differs")
        self.amplitude_runtime_identity = MappingProxyType(
            dict(amplitude_runtime)
        )
        amplitude_validation_digest = self.amplitude.validation_receipt.get(
            "validation_digest"
        )
        projections = _authority_projection_digests(
            manifest,
            amplitude_manifest_sha256=self.amplitude.manifest_file_sha256,
            amplitude_validation_digest=amplitude_validation_digest,
        )
        expected = {
            "data_sha256": _sha256(
                expected_data_sha256, label="expected data authority SHA-256"
            ),
            "teacher_sha256": _sha256(
                expected_teacher_sha256, label="expected teacher authority SHA-256"
            ),
            "nuisance_sha256": _sha256(
                expected_nuisance_sha256,
                label="expected nuisance authority SHA-256",
            ),
        }
        if {
            key: projections[key] for key in expected
        } != expected:
            fail("authority data/teacher/nuisance projection SHA differs")
        self.projection_digests = projections
        self.sources = {
            row["source_iid"]: row
            for row in manifest["sources"]
            if row["analysis_split"] == "fit"
        }
        self.pairs = {
            row["pair_id"]: row
            for row in manifest["pairs"]
            if row["analysis_split"] == "fit" and row["optimizer_admitted"] is True
        }
        self.representations = {
            (row["teacher_cell_id"], row["branch"]): row
            for row in manifest["representation_admissions"]
            if row["analysis_split"] == "fit" and row["optimizer_admitted"] is True
        }
        expected_counts = (
            (8, 16, 4)
            if self.is_disposable_canary
            else (64, 128, 16)
        )
        if (
            len(self.sources),
            len(self.pairs),
            len(self.representations),
        ) != expected_counts:
            fail(
                "optimizer authority runtime index does not close the selected profile"
            )
        self._modules = modules
        self._latent_cache: dict[str, Any] = {}
        self._teacher_cache: dict[tuple[str, str, int, str], TeacherPacketV1] = {}
        self.schedule_authority_receipt: Optional[Mapping[str, Any]] = None
        self._executable_teacher_coordinates: Optional[
            frozenset[tuple[str, str, int]]
        ] = None

    def schedule_rows(self) -> tuple[Any, ...]:
        learning = self._modules.learning
        rows = tuple(
            learning.ActionPairRow(
                row_id=pair["pair_id"],
                source_id=pair["source_iid"],
                branch=pair["branch"],
                teacher_cell_id=pair["teacher_cell_id"],
            )
            for pair in self.pairs.values()
        )
        return rows

    def build_schedule(self, *, run_seed: int) -> tuple[Any, ...]:
        if not self.is_disposable_canary:
            return self._modules.learning.build_formal_schedule_v1(
                self.schedule_rows(), run_seed=run_seed
            )
        if run_seed != self.amplitude.validation_receipt.get(
            "schedule_run_seed"
        ):
            fail("canary run seed differs from official materializer admission")
        schedule = self._modules.canary_authority.build_checkpoint_scaffold_schedule_v1(
            self.schedule_rows(),
            run_seed=run_seed,
            learning_module=self._modules.learning,
            checkpoint_module=self._modules.checkpoint,
        )
        self.schedule_authority_receipt = (
            self._modules.canary_authority.schedule_authority_receipt_v1(
                schedule, checkpoint_module=self._modules.checkpoint
            )
        )
        canonical = self._modules.checkpoint.canonical_schedule_v2(schedule)
        self._executable_teacher_coordinates = frozenset(
            (
                str(row["row"]["teacher_cell_id"]),
                str(row["row"]["branch"]),
                int(row["sigma_index"]),
            )
            for row in canonical[:16]
        )
        return schedule

    def authorize_scheduled_row(self, scheduled: Any) -> None:
        if not self.is_disposable_canary:
            return
        self._modules.canary_authority.authorize_scheduled_row_v1(
            scheduled, admitted_pairs=self.pairs
        )
        coordinate = (
            str(scheduled.row.teacher_cell_id),
            str(scheduled.row.branch),
            int(scheduled.sigma_index),
        )
        if (
            self._executable_teacher_coordinates is None
            or coordinate not in self._executable_teacher_coordinates
        ):
            fail("canary scheduled teacher coordinate is non-executable evidence")

    def load_normalized_source(
        self, source_iid: str, *, vae_mean: Any, vae_std: Any
    ) -> Any:
        torch = self._modules.torch
        if source_iid not in self.sources:
            fail("scheduled source IID is not optimizer-admitted")
        if source_iid in self._latent_cache:
            return self._latent_cache[source_iid]
        row = self.sources[source_iid]
        path = _plain_absolute_file(
            row["source_posterior_index0_path"],
            label=f"{source_iid} physical posterior index0",
        )
        expected = row["source_posterior_index0_sha256"]
        if file_sha256(path) != expected:
            fail(f"{source_iid} physical posterior index0 SHA differs")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            fail(f"{source_iid} physical posterior changed during read")
        parameters = self._modules.source_data._decode_source_posterior_parameters(
            raw, iid=source_iid
        )
        source_mode = parameters[:, :16].float().contiguous()
        value = ((source_mode - vae_mean) / vae_std).detach().float().contiguous()
        if (
            value.device.type != "cpu"
            or tuple(int(item) for item in value.shape[:3]) != (1, 16, 21)
            or value.requires_grad
            or not bool(torch.isfinite(value).all().item())
        ):
            fail("normalized authority source latent differs")
        self._latent_cache[source_iid] = value
        return value

    def teacher_packet(
        self,
        *,
        teacher_cell_id: str,
        branch: str,
        sigma_index: int,
        device: Any,
    ) -> TeacherPacketV1:
        if self.is_disposable_canary and (
            self._executable_teacher_coordinates is None
            or (teacher_cell_id, branch, sigma_index)
            not in self._executable_teacher_coordinates
        ):
            fail("canary teacher coordinate is non-executable evidence")
        torch = self._modules.torch
        key = (teacher_cell_id, branch, sigma_index, str(device))
        if key in self._teacher_cache:
            return self._teacher_cache[key]
        admission = self.representations.get((teacher_cell_id, branch))
        if admission is None or sigma_index not in self._modules.learning.SIGMA_INDICES:
            fail("scheduled teacher cell/branch/sigma is not admitted")
        evidence = admission["origin_evidence"]
        authority = self._modules.authority
        psiout = authority._validate_tensor_container(
            evidence["psiout_sidecar_path"],
            evidence["psiout_sidecar_sha256"],
            container_kind="psiout",
            evidence_id=evidence["evidence_id"],
            evidence_role="teacher_origin",
            teacher_cell_id=teacher_cell_id,
            branch=branch,
            label=f"runtime teacher {teacher_cell_id}/{branch}",
        )
        nuisance = authority._validate_tensor_container(
            evidence["nuisance_packet_path"],
            evidence["nuisance_packet_sha256"],
            container_kind="nuisance",
            evidence_id=evidence["evidence_id"],
            evidence_role="teacher_origin",
            teacher_cell_id=teacher_cell_id,
            branch=branch,
            label=f"runtime nuisance {teacher_cell_id}/{branch}",
        )
        tensor_name = authority._tensor_name
        teacher_slice = psiout[tensor_name(sigma_index, "projected_unit")]
        camera_slice = nuisance[tensor_name(sigma_index, "camera_unit")]
        appearance_slice = nuisance[tensor_name(sigma_index, "appearance_unit")]
        teacher_tensor = torch.tensor(
            teacher_slice[1], dtype=torch.float32, device=device
        ).reshape(1, 21, 32).contiguous()
        camera_tensor = torch.tensor(
            camera_slice[1], dtype=torch.float32, device=device
        ).reshape(1, 21, 32).contiguous()
        appearance_tensor = torch.tensor(
            appearance_slice[1], dtype=torch.float32, device=device
        ).reshape(1, 21, 32).contiguous()
        packet = self._modules.learning.build_nuisance_packet_v1(
            camera_tensor, appearance_tensor
        )
        teacher_unit = self._modules.learning.teacher_unit_v1(
            teacher_tensor
        ).detach().contiguous()
        amplitude = self.amplitude.resolve(
            teacher_cell_id, branch, sigma_index
        )
        minimum_amplitude = torch.tensor(
            [amplitude.value_float32], dtype=torch.float32, device=device
        ).detach().contiguous()
        if (
            tuple(int(item) for item in minimum_amplitude.shape) != (1,)
            or not bool((minimum_amplitude > 0.0).all().item())
        ):
            fail("resolved sealed minimum amplitude differs")
        receipt = {
            "teacher_cell_id": teacher_cell_id,
            "branch": branch,
            "sigma_index": sigma_index,
            "admission_digest": admission["admission_digest"],
            "evidence_digest": evidence["evidence_digest"],
            "psiout_sidecar_sha256": evidence["psiout_sidecar_sha256"],
            "nuisance_packet_sha256": evidence["nuisance_packet_sha256"],
            "teacher_slice_sha256": teacher_slice[2],
            "camera_slice_sha256": camera_slice[2],
            "appearance_slice_sha256": appearance_slice[2],
            "amplitude_manifest_sha256": (
                self.amplitude.manifest_file_sha256
            ),
            "amplitude_validation_digest": self.amplitude.validation_receipt[
                "validation_digest"
            ],
            "amplitude_calibration_id": amplitude.calibration_id,
            "amplitude_bundle_digest": amplitude.bundle_digest,
            "minimum_amplitude_float32_be_hex": amplitude.float32_be_hex,
            "minimum_amplitude_float32_le_sha256": (
                amplitude.float32_le_sha256
            ),
        }
        result = TeacherPacketV1(
            teacher_unit=teacher_unit,
            minimum_amplitude=minimum_amplitude,
            minimum_amplitude_float32_le_sha256=(
                amplitude.float32_le_sha256
            ),
            minimum_amplitude_bundle_digest=amplitude.bundle_digest,
            minimum_amplitude_calibration_id=amplitude.calibration_id,
            nuisance_packet=packet,
            authority_receipt=MappingProxyType(receipt),
        )
        self._teacher_cache[key] = result
        return result


def validate_amplitude_runtime_binding_v1(
    *,
    authority_index: Full30AuthorityRuntimeIndexV1,
    bernini_revision: str,
    veomni_revision: str,
    model_sha256: str,
    transformer_config_sha256: str,
    sigma_table_sha256: str,
    psiout_protocol_sha256: str,
    official_provider_source_sha256: str,
    executed_runtime_source_sha256: str,
) -> Mapping[str, Any]:
    """Bind presealed Frozen calibrators to the exact runtime being trained."""

    runtime = authority_index.amplitude_runtime_identity
    expected = {
        "schema_version": AMPLITUDE_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "official_checkpoint_tree_sha256": _sha256(
            model_sha256, label="amplitude runtime model SHA-256"
        ),
        "transformer_config_sha256": _sha256(
            transformer_config_sha256,
            label="amplitude runtime transformer config SHA-256",
        ),
        "sigma_table_sha256": _sha256(
            sigma_table_sha256,
            label="amplitude runtime sigma table SHA-256",
        ),
        "psiout_protocol_sha256": _sha256(
            psiout_protocol_sha256,
            label="amplitude runtime PsiOut protocol SHA-256",
        ),
        "official_provider_source_sha256": _sha256(
            official_provider_source_sha256,
            label="amplitude official provider source SHA-256",
        ),
        "official_provider_abi": OFFICIAL_PROVIDER_ABI,
        "compute_contract": dict(AMPLITUDE_COMPUTE_CONTRACT),
        "compute_contract_digest": object_sha256(
            AMPLITUDE_COMPUTE_CONTRACT
        ),
        "frame_count": FRAME_COUNT,
        "fps": 25.0,
        "sampler_steps": 40,
    }
    if (
        type(bernini_revision) is not str
        or re.fullmatch(r"[0-9a-f]{40}", bernini_revision) is None
        or type(veomni_revision) is not str
        or re.fullmatch(r"[0-9a-f]{40}", veomni_revision) is None
    ):
        fail("amplitude runtime source revisions differ")
    runtime_unsigned = dict(runtime)
    runtime_digest = runtime_unsigned.pop("runtime_digest", None)
    if any(runtime.get(field) != value for field, value in expected.items()):
        fail("amplitude authority was calibrated under a different runtime")
    if (
        set(runtime) != {*expected, "runtime_digest"}
        or runtime_digest != object_sha256(runtime_unsigned)
        or runtime_digest != authority_index.amplitude.frozen_runtime_digest
        or runtime_unsigned != expected
    ):
        fail("amplitude authority runtime field closure/digest differs")
    executed_runtime_sha = _sha256(
        executed_runtime_source_sha256,
        label="executed full30 runtime source SHA-256",
    )
    if executed_runtime_sha != FROZEN_RELEASE_MEMBER_SHA256[
        "full30_action_runtime_v1.py"
    ]:
        fail("executed Frozen route source differs from frozen runtime API")
    value = {
        "schema_version": AMPLITUDE_RUNTIME_BINDING_SCHEMA_VERSION,
        "amplitude_manifest_sha256": (
            authority_index.amplitude.manifest_file_sha256
        ),
        "amplitude_validation_digest": authority_index.amplitude.validation_receipt[
            "validation_digest"
        ],
        "frozen_runtime_digest": authority_index.amplitude.frozen_runtime_digest,
        "actual_runtime": expected,
        "executed_training_runtime": {
            "source_sha256": executed_runtime_sha,
            "frozen_route_contract": dict(EXECUTED_FROZEN_ROUTE_CONTRACT),
            "base_compute_dtype": "torch.bfloat16",
            "autocast_dtype": "torch.bfloat16",
            "trainable_dtype": "torch.float32",
            "optimizer_v_dtype": "torch.float32",
            "model_training_mode_outside_frozen_slots": True,
            "world_size": WORLD_SIZE,
            "dp_size": DP_SIZE,
            "sp_size": SP_SIZE,
            "gradient_reduction": "SP4-mean-then-DP2-mean",
        },
        "all_runtime_identities_exactly_equal": True,
    }
    return _seal(value)


def schedule_noise_sha256_v1(schedule: Sequence[Any], *, run_seed: int) -> str:
    rows = [
        {
            "global_index": row.global_index,
            "update": row.update,
            "microbatch": row.microbatch,
            "dp_rank": row.dp_rank,
            "row_id": row.row.row_id,
            "source_id": row.row.source_id,
            "branch": row.row.branch,
            "teacher_cell_id": row.row.teacher_cell_id,
            "sigma_index": row.sigma_index,
            "noise_seed": row.noise_seed,
        }
        for row in schedule
    ]
    return object_sha256(
        {
            "schema_version": "bernini-full30-action-schedule-noise-authority-v1",
            "run_seed": run_seed,
            "rows": rows,
        }
    )


def prepare_runtime_record_v1(
    *,
    scheduled: Any,
    authority_index: Full30AuthorityRuntimeIndexV1,
    conditions: Mapping[str, Any],
    noop_condition: Any,
    vae_mean: Any,
    vae_std: Any,
    rope: Any,
    device: Any,
) -> PreparedRecordV1:
    modules = authority_index._modules
    torch = modules.torch
    # The disposable checkpoint tail is rejected here, before any source,
    # condition, teacher, nuisance, or amplitude payload can be opened.
    authority_index.authorize_scheduled_row(scheduled)
    pair = authority_index.pairs.get(scheduled.row.row_id)
    if (
        pair is None
        or pair["source_iid"] != scheduled.row.source_id
        or pair["branch"] != scheduled.row.branch
        or pair["teacher_cell_id"] != scheduled.row.teacher_cell_id
    ):
        fail("scheduled action row differs from admitted pair authority")
    source = authority_index.load_normalized_source(
        scheduled.row.source_id, vae_mean=vae_mean, vae_std=vae_std
    )
    sigma_index = int(scheduled.sigma_index)
    sigma_value = modules.sigma.PINNED_POSITIVE_SIGMAS[sigma_index]
    timestep_value = modules.sigma.PINNED_TIMESTEPS[sigma_index]
    coordinate = SimpleNamespace(sigma=sigma_value, timestep=timestep_value)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(scheduled.noise_seed))
    epsilon = torch.randn(
        tuple(source.shape), generator=generator, dtype=torch.float32
    ).contiguous()
    packed = modules.packed_trainer.prepare_restoration_pair(
        clean=source,
        corrupted_source=source,
        epsilon=epsilon,
        coordinate=coordinate,
        rope=rope,
        device=device,
    )
    tokens = int(packed["source_tokens"])
    source_patches = packed["input_patches"][:tokens].detach().clone().contiguous()
    target_patches = packed["input_patches"][tokens:].detach().clone().contiguous()
    rotary = (
        packed["rotary"].permute(1, 0, 2).unsqueeze(0).detach().contiguous()
    )
    timestep = torch.tensor(
        [float(timestep_value)], dtype=torch.float32, device=device
    )
    branch_condition = conditions.get(pair["instruction_utf8_sha256"])
    if branch_condition is None:
        fail("pair instruction condition is absent")
    runtime_record = modules.branch_runtime.Full30ActionRecordV1(
        row_id=pair["pair_id"],
        source_iid=pair["source_iid"],
        branch=pair["branch"],
        source_patches=source_patches,
        noisy_target_patches=target_patches,
        rotary_embs=rotary,
        timestep=timestep,
        spatial_shape=tuple(int(item) for item in source.shape),
        branch_condition=modules.branch_runtime.ConditionBindingV1(
            role="branch",
            authority_sha256=pair["instruction_utf8_sha256"],
            condition=branch_condition,
        ),
        noop_condition=modules.branch_runtime.ConditionBindingV1(
            role="noop",
            authority_sha256=hashlib.sha256(
                NOOP_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            condition=noop_condition,
        ),
    )
    teacher = authority_index.teacher_packet(
        teacher_cell_id=pair["teacher_cell_id"],
        branch=pair["branch"],
        sigma_index=sigma_index,
        device=device,
    )
    noop_target = (
        (epsilon - source)
        .to(device=device, dtype=torch.float32)
        .detach()
        .contiguous()
    )
    objective = modules.training_step.seal_record_objective_authority_v1(
        row_id=pair["pair_id"],
        source_id=pair["source_iid"],
        branch=pair["branch"],
        teacher_cell_id=pair["teacher_cell_id"],
        sigma_index=sigma_index,
        noise_seed=int(scheduled.noise_seed),
        teacher_unit=teacher.teacher_unit,
        minimum_amplitude=teacher.minimum_amplitude,
        minimum_amplitude_float32_le_sha256=(
            teacher.minimum_amplitude_float32_le_sha256
        ),
        minimum_amplitude_bundle_digest=(
            teacher.minimum_amplitude_bundle_digest
        ),
        minimum_amplitude_calibration_id=(
            teacher.minimum_amplitude_calibration_id
        ),
        nuisance_packet=teacher.nuisance_packet,
        noop_target_velocity=noop_target,
        data_teacher_authority_manifest_sha256=(
            authority_index.manifest_sha256
        ),
        amplitude_authority_manifest_sha256=(
            authority_index.amplitude.manifest_file_sha256
        ),
    )
    step_record = modules.training_step.Full30LocalMicroRecordV1(
        scheduled=scheduled,
        runtime_record=runtime_record,
        objective=objective,
    )
    record_authority = {
        "global_index": scheduled.global_index,
        "update": scheduled.update,
        "microbatch": scheduled.microbatch,
        "dp_rank": scheduled.dp_rank,
        "row_id": pair["pair_id"],
        "pair_digest": pair["pair_digest"],
        "source_iid": pair["source_iid"],
        "source_posterior_index0_sha256": authority_index.sources[
            pair["source_iid"]
        ]["source_posterior_index0_sha256"],
        "instruction_utf8_sha256": pair["instruction_utf8_sha256"],
        "teacher_cell_id": pair["teacher_cell_id"],
        "sigma_index": sigma_index,
        "sigma_float32_be_hex": modules.sigma.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
            sigma_index
        ],
        "timestep": timestep_value,
        "noise_seed": scheduled.noise_seed,
        "epsilon_sha256": modules.distributed_runtime.tensor_sha256(epsilon),
        "epsilon_exact_sha256": modules.training_step.tensor_sha256_v1(
            epsilon, label="scheduled epsilon"
        ),
        "normalized_source_exact_sha256": (
            modules.training_step.tensor_sha256_v1(
                source, label="normalized real source"
            )
        ),
        "epsilon_minus_source_target_sha256": objective.noop_target_sha256,
        "teacher_unit_sha256": objective.teacher_unit_sha256,
        "minimum_amplitude_sha256": objective.minimum_amplitude_sha256,
        "minimum_amplitude_float32_le_sha256": (
            objective.minimum_amplitude_float32_le_sha256
        ),
        "minimum_amplitude_bundle_digest": (
            objective.minimum_amplitude_bundle_digest
        ),
        "minimum_amplitude_calibration_id": (
            objective.minimum_amplitude_calibration_id
        ),
        "nuisance_packet_sha256": objective.nuisance_packet_sha256,
        "objective_authority_digest": objective.authority_digest,
        "data_teacher_authority_manifest_sha256": (
            objective.data_teacher_authority_manifest_sha256
        ),
        "amplitude_authority_manifest_sha256": (
            objective.amplitude_authority_manifest_sha256
        ),
        "teacher_amplitude_authority": dict(teacher.authority_receipt),
        "synthetic_target_bytes_read": False,
        "synthetic_target_index1_bytes_read": False,
    }
    return PreparedRecordV1(
        schedule_row=scheduled,
        runtime_record=runtime_record,
        teacher=teacher,
        step_record=step_record,
        authority_receipt=MappingProxyType(record_authority),
    )


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _torch_state_bytes(value: Any) -> bytes:
    return value.detach().contiguous().cpu().numpy().tobytes(order="C")


def capture_world8_rng_state_v1(*, modules: BusinessModulesV1, parallel: Any) -> Mapping[str, Any]:
    torch = modules.torch
    torch_dist = importlib.import_module("torch.distributed")
    local = {
        "python": _encode_bytes(pickle.dumps(random.getstate(), protocol=4)),
        "cpu": _encode_bytes(_torch_state_bytes(torch.get_rng_state())),
        "cuda": _encode_bytes(
            _torch_state_bytes(torch.cuda.get_rng_state(parallel.contract.local_rank))
        ),
    }
    gathered: list[Any] = [None] * WORLD_SIZE
    torch_dist.all_gather_object(gathered, local, group=parallel.world_group)
    if any(type(row) is not dict or set(row) != {"python", "cpu", "cuda"} for row in gathered):
        fail("WORLD8 RNG gather differs")
    return MappingProxyType(
        {
            "schema_version": modules.checkpoint.RNG_SCHEMA_VERSION,
            "python_rank_state_b64": [row["python"] for row in gathered],
            "torch_cpu_rank_state_b64": [row["cpu"] for row in gathered],
            "torch_cuda_rank_state_b64": [row["cuda"] for row in gathered],
        }
    )


def rng_artifact_sha256_v1(value: Mapping[str, Any], *, checkpoint_module: Any) -> str:
    return hashlib.sha256(checkpoint_module.canonical_json_bytes(value) + b"\n").hexdigest()


class _RngRestoreTransactionV1:
    def __init__(
        self,
        state: Mapping[str, Any],
        *,
        rank: int,
        local_rank: int,
        modules: BusinessModulesV1,
    ) -> None:
        self.state = dict(state)
        self.rank = rank
        self.local_rank = local_rank
        self.modules = modules
        torch = modules.torch
        self.before_python = random.getstate()
        self.before_cpu = torch.get_rng_state().clone()
        self.before_cuda = torch.cuda.get_rng_state(local_rank).clone()
        self.committed = False

    @staticmethod
    def _bytes(value: str) -> bytes:
        try:
            return base64.b64decode(value.encode("ascii"), validate=True)
        except (ValueError, UnicodeError) as error:
            raise Full30ActionTrainingError("checkpoint RNG base64 differs") from error

    def commit(self) -> str:
        torch = self.modules.torch
        if self.committed:
            fail("RNG restore transaction committed twice")
        try:
            python_state = pickle.loads(
                self._bytes(self.state["python_rank_state_b64"][self.rank])
            )
            cpu_raw = bytearray(
                self._bytes(self.state["torch_cpu_rank_state_b64"][self.rank])
            )
            cuda_raw = bytearray(
                self._bytes(self.state["torch_cuda_rank_state_b64"][self.rank])
            )
            cpu = torch.frombuffer(cpu_raw, dtype=torch.uint8).clone()
            cuda = torch.frombuffer(cuda_raw, dtype=torch.uint8).clone()
            random.setstate(python_state)
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda, self.local_rank)
        except Exception as error:
            self.rollback()
            raise Full30ActionTrainingError("cannot commit checkpoint RNG state") from error
        self.committed = True
        return rng_artifact_sha256_v1(
            self.state, checkpoint_module=self.modules.checkpoint
        )

    def rollback(self) -> None:
        torch = self.modules.torch
        random.setstate(self.before_python)
        torch.set_rng_state(self.before_cpu)
        torch.cuda.set_rng_state(self.before_cuda, self.local_rank)
        self.committed = False


def _checkpoint_status_consensus(local: Mapping[str, Any], *, parallel: Any) -> Mapping[str, Any]:
    torch_dist = importlib.import_module("torch.distributed")
    gathered: list[Any] = [None] * WORLD_SIZE
    torch_dist.all_gather_object(gathered, dict(local), group=parallel.world_group)
    if all(row.get("ok") is True for row in gathered):
        digests = {row.get("digest") for row in gathered}
        if len(digests) == 1:
            return {"ok": True, "digest": next(iter(digests)), "participant_count": WORLD_SIZE}
    return {
        "ok": False,
        "error": "one-or-more-WORLD8 checkpoint preflights differed",
        "participant_count": WORLD_SIZE,
    }


def _checkpoint_result_broadcast(local: Optional[Mapping[str, Any]], *, parallel: Any) -> Mapping[str, Any]:
    torch_dist = importlib.import_module("torch.distributed")
    payload = [None if local is None else dict(local)]
    torch_dist.broadcast_object_list(payload, src=0, group=parallel.world_group)
    if not isinstance(payload[0], Mapping):
        fail("checkpoint rank-zero result broadcast differs")
    return payload[0]


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite receipt: {path}")
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
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def bind_global_update_authorities_v1(
    *,
    receipt: Mapping[str, Any],
    scheduled_global_rows: Sequence[Any],
    schedule_full_sha256: str,
    schedule_prefix_sha256: str,
    noise_authority_sha256: str,
    teacher_authority_sha256: str,
    data_teacher_authority_manifest_sha256: str,
    amplitude_authority_manifest_sha256: str,
) -> Mapping[str, Any]:
    if type(receipt) not in (dict, MappingProxyType) and not isinstance(receipt, Mapping):
        fail("global update receipt differs")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if declared != object_sha256(unsigned):
        fail("global update receipt digest differs before authority binding")
    rows = tuple(scheduled_global_rows)
    if len(rows) != GLOBAL_BATCH:
        fail("update authority requires exactly eight scheduled rows")
    schedule_rows = [
        {
            "global_index": row.global_index,
            "epoch": row.epoch,
            "update": row.update,
            "microbatch": row.microbatch,
            "dp_rank": row.dp_rank,
            "sigma_index": row.sigma_index,
            "noise_seed": row.noise_seed,
            "row": {
                "row_id": row.row.row_id,
                "source_id": row.row.source_id,
                "branch": row.row.branch,
                "teacher_cell_id": row.row.teacher_cell_id,
            },
        }
        for row in rows
    ]
    bound = {
        **unsigned,
        "pre_authority_binding_receipt_digest": declared,
        "authority_binding": {
            "schedule_full_sha256": _sha256(
                schedule_full_sha256, label="schedule full SHA-256"
            ),
            "schedule_prefix_sha256": _sha256(
                schedule_prefix_sha256, label="schedule prefix SHA-256"
            ),
            "schedule_update_rows_sha256": object_sha256(schedule_rows),
            "noise_authority_sha256": _sha256(
                noise_authority_sha256, label="noise authority SHA-256"
            ),
            "teacher_authority_sha256": _sha256(
                teacher_authority_sha256, label="teacher authority SHA-256"
            ),
            "data_teacher_authority_manifest_sha256": _sha256(
                data_teacher_authority_manifest_sha256,
                label="data/teacher authority manifest SHA-256",
            ),
            "amplitude_authority_manifest_sha256": _sha256(
                amplitude_authority_manifest_sha256,
                label="amplitude authority manifest SHA-256",
            ),
            "scheduled_rows": schedule_rows,
        },
    }
    return _seal(bound)


def _load_checkpoint_reference_file(
    *, path: str | Path, expected_sha256: str, checkpoint_module: Any, label: str
) -> Any:
    source = _plain_absolute_file(path, label=label)
    value = _strict_json_file(source, expected_sha256=expected_sha256, label=label)
    return checkpoint_module.CheckpointReference.from_mapping(value)


def _validate_fresh_output(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute() or os.path.lexists(path):
        fail("output must be one fresh absolute path")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise Full30ActionTrainingError("output parent is unavailable") from error
    if parent != path.parent or not parent.is_dir():
        fail("output parent must be one canonical directory")
    return path


def _condition_from_instruction(
    *, instruction: str, instruction_sha256: str, tokenizer: Any, renderer: Any,
    device: Any, modules: BusinessModulesV1, task_name: str
) -> Any:
    expected = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    if expected != instruction_sha256:
        fail("instruction bytes differ from admitted instruction SHA-256")
    tokenized = modules.distributed_runtime.tokenize_generic_instruction(
        tokenizer, instruction, device
    )
    text_lens, text_embs = renderer.get_t5_text_embeddings(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        tokenized["t5_input_lens"],
    )
    prompt = {
        "task_name": task_name,
        "instruction_sha256": instruction_sha256,
        "input_ids": tokenized["input_ids"].detach().cpu().tolist(),
        "attention_mask": tokenized["attention_mask"].detach().cpu().tolist(),
    }
    return modules.inference_helper.TextCondition(
        text_lens=text_lens,
        text_embs=text_embs,
        prompt_sha256=object_sha256(prompt),
        instruction_sha256=instruction_sha256,
        task_name=task_name,
    )


def _reference_file_payload(reference: Any) -> Mapping[str, Any]:
    value = reference.as_dict()
    return MappingProxyType(dict(value))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--method-root", required=True)
    value.add_argument("--release-manifest", required=True)
    value.add_argument("--expected-release-manifest-sha256", required=True)
    value.add_argument("--expected-release-sha256", required=True)

    value.add_argument("--authority-manifest", required=True)
    value.add_argument("--expected-authority-manifest-sha256", required=True)
    value.add_argument("--amplitude-authority-manifest", required=True)
    value.add_argument(
        "--expected-amplitude-authority-manifest-sha256", required=True
    )
    value.add_argument("--expected-data-authority-sha256", required=True)
    value.add_argument("--expected-teacher-authority-sha256", required=True)
    value.add_argument("--expected-nuisance-authority-sha256", required=True)
    value.add_argument("--expected-noise-authority-sha256", required=True)

    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--checkpoint-content-manifest", required=True)
    value.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    value.add_argument("--expected-model-sha256", required=True)
    value.add_argument("--expected-bernini-commit", required=True)
    value.add_argument("--expected-veomni-commit", required=True)

    value.add_argument("--output", required=True)
    value.add_argument("--arm", choices=ARMS, required=True)
    value.add_argument("--profile", choices=PROFILES, required=True)
    value.add_argument("--start-update", type=int, required=True)
    value.add_argument("--stop-update", type=int, required=True)
    value.add_argument("--expected-boundary-plan-sha256", required=True)
    value.add_argument("--run-seed", type=int, required=True)

    value.add_argument("--segment-review-gate")
    value.add_argument("--expected-segment-review-gate-sha256")
    value.add_argument("--canary-gate")
    value.add_argument("--expected-canary-gate-sha256")
    value.add_argument("--canary-receipt")
    value.add_argument("--expected-canary-receipt-sha256")
    value.add_argument("--canary-u2-reference")
    value.add_argument("--expected-canary-u2-reference-sha256")
    value.add_argument("--resume-checkpoint")
    value.add_argument("--resume-reference")
    value.add_argument("--expected-resume-reference-sha256")
    value.add_argument("--resume-previous-reference")
    value.add_argument("--expected-resume-previous-reference-sha256")
    return value


def _validate_cli_bindings(
    args: argparse.Namespace, *, release_receipt: Mapping[str, Any]
) -> tuple[
    BoundaryPlanV1,
    Path,
    Optional[Mapping[str, Any]],
    Optional[FreshFormalCanaryInputsV1],
]:
    if args.arm not in ARMS or args.profile not in PROFILES:
        fail("arm/profile differs")
    if type(args.run_seed) is not int or not 0 <= args.run_seed < 2**64:
        fail("run seed must be an unsigned 64-bit integer")
    plan = build_boundary_plan_v1(
        args.profile,
        start_update=args.start_update,
        stop_update=args.stop_update,
    )
    plan_receipt = plan.receipt()
    if plan_receipt["boundary_plan_sha256"] != _sha256(
        args.expected_boundary_plan_sha256,
        label="expected boundary plan SHA-256",
    ):
        fail("boundary plan SHA-256 differs")
    output = _validate_fresh_output(args.output)
    resume_values = (
        args.resume_checkpoint,
        args.resume_reference,
        args.expected_resume_reference_sha256,
        args.resume_previous_reference,
        args.expected_resume_previous_reference_sha256,
    )
    if plan.requires_resume:
        if any(value is None for value in resume_values):
            fail("strict formal resume requires current and predecessor references")
    elif any(value is not None for value in resume_values):
        fail("fresh execution may not consume a resume checkpoint/reference")
    canary_values = (
        args.canary_gate,
        args.expected_canary_gate_sha256,
        args.canary_receipt,
        args.expected_canary_receipt_sha256,
        args.canary_u2_reference,
        args.expected_canary_u2_reference_sha256,
    )
    fresh_formal = (
        plan.profile == "review-gated-segment"
        and plan.start_update == 0
        and plan.stop_update == 20
    )
    canary_inputs: Optional[FreshFormalCanaryInputsV1] = None
    if fresh_formal:
        if any(value is None for value in canary_values):
            fail("fresh formal u0->u20 requires sealed same-arm u2 canary evidence")
        canary_inputs = FreshFormalCanaryInputsV1(
            gate_path=_plain_absolute_file(
                args.canary_gate, label="fresh formal canary gate"
            ),
            gate_sha256=_sha256(
                args.expected_canary_gate_sha256,
                label="fresh formal canary gate SHA-256",
            ),
            receipt_path=_plain_absolute_file(
                args.canary_receipt, label="completed canary receipt"
            ),
            receipt_sha256=_sha256(
                args.expected_canary_receipt_sha256,
                label="completed canary receipt SHA-256",
            ),
            u2_reference_path=_plain_absolute_file(
                args.canary_u2_reference, label="completed canary u2 reference"
            ),
            u2_reference_sha256=_sha256(
                args.expected_canary_u2_reference_sha256,
                label="completed canary u2 reference SHA-256",
            ),
        )
    elif any(value is not None for value in canary_values):
        fail("canary evidence is accepted only by a fresh formal u0->u20 run")
    gate_values = (
        args.segment_review_gate,
        args.expected_segment_review_gate_sha256,
    )
    gate: Optional[Mapping[str, Any]] = None
    if plan.requires_review_gate:
        if any(value is None for value in gate_values):
            fail("formal segment requires an exact review gate")
        gate = validate_segment_gate_v1(
            path=args.segment_review_gate,
            expected_sha256=args.expected_segment_review_gate_sha256,
            plan=plan,
            arm=args.arm,
            authority_manifest_sha256=args.expected_authority_manifest_sha256,
            amplitude_authority_manifest_sha256=(
                args.expected_amplitude_authority_manifest_sha256
            ),
            model_sha256=args.expected_model_sha256,
            release_sha256=release_receipt["release_sha256"],
            resume_reference_sha256=(
                args.expected_resume_reference_sha256
                if plan.requires_resume
                else None
            ),
            fresh_formal_canary_gate_sha256=(
                canary_inputs.gate_sha256
                if canary_inputs is not None
                else None
            ),
        )
    elif any(value is not None for value in gate_values):
        fail("disposable canary may not masquerade as a reviewed formal segment")
    return plan, output, gate, canary_inputs


def _save_checkpoint_world8(
    *,
    target: Path,
    optimizer: Any,
    completed_updates: int,
    schedule: Sequence[Any],
    history: Sequence[Mapping[str, Any]],
    rng_state: Mapping[str, Any],
    bindings: Any,
    inventory_sha256: str,
    previous: Any,
    parallel: Any,
    modules: BusinessModulesV1,
) -> Any:
    return modules.checkpoint.save_checkpoint(
        target,
        optimizer=optimizer,
        completed_updates=completed_updates,
        full_schedule=schedule,
        history=history,
        rng_state=rng_state,
        bindings=bindings,
        authoritative_inventory_sha256=inventory_sha256,
        previous_checkpoint=previous,
        rank=parallel.contract.rank,
        world_size=WORLD_SIZE,
        status_consensus=lambda local: _checkpoint_status_consensus(
            local, parallel=parallel
        ),
        result_broadcast=lambda local: _checkpoint_result_broadcast(
            local, parallel=parallel
        ),
    )


def build_gradient_mean_callback_v1(
    *, modules: BusinessModulesV1, parallel: Any, bucket_bytes: int = 64 * 1024 * 1024
) -> Callable[[Any], Mapping[str, Any]]:
    """Build the transport-only SP4/DP2 mean required by the step core."""

    if type(bucket_bytes) is not int or bucket_bytes <= 0:
        fail("gradient collective bucket size differs")
    torch = modules.torch
    torch_dist = importlib.import_module("torch.distributed")
    topology = parallel.contract.topology

    def callback(request: Any) -> Mapping[str, Any]:
        if not isinstance(
            request, modules.training_step.GradientCollectiveRequestV1
        ):
            fail("gradient collective request type differs")
        rank = parallel.contract.rank
        if request.rank != rank:
            fail("gradient collective request rank differs")
        if request.scope == "SP4":
            expected_ranks = topology.sp_group_ranks[
                parallel.contract.arm_index
            ]
            group = parallel.sp_group
        elif request.scope == "DP2":
            expected_ranks = topology.dp_group_ranks[
                parallel.contract.sp_rank
            ]
            group = parallel.dp_group
        else:
            fail("gradient collective scope differs")
        if tuple(request.group_ranks) != tuple(expected_ranks):
            fail("gradient collective participant ranks differ")
        gradients = request.gradients
        if not isinstance(gradients, dict) or not gradients:
            fail("gradient collective mapping is empty")
        names = tuple(sorted(gradients, key=lambda item: item.encode("utf-8")))
        ready = all(
            isinstance(gradients[name], torch.Tensor)
            and gradients[name].dtype == torch.float32
            and gradients[name].is_contiguous()
            and bool(torch.isfinite(gradients[name]).all().item())
            for name in names
        )
        if not modules.distributed_runtime.world_all_true(
            ready, group=parallel.world_group
        ):
            fail("at least one WORLD8 gradient collective input differs")

        buckets: list[list[str]] = []
        current: list[str] = []
        current_bytes = 0
        for name in names:
            value = gradients[name]
            item_bytes = int(value.numel()) * int(value.element_size())
            if current and current_bytes + item_bytes > bucket_bytes:
                buckets.append(current)
                current = []
                current_bytes = 0
            current.append(name)
            current_bytes += item_bytes
        if current:
            buckets.append(current)
        participant_count = len(expected_ranks)
        with torch.no_grad():
            for bucket in buckets:
                flat = torch.cat([gradients[name].reshape(-1) for name in bucket])
                torch_dist.all_reduce(
                    flat, op=torch_dist.ReduceOp.SUM, group=group
                )
                flat.div_(float(participant_count))
                offset = 0
                for name in bucket:
                    value = gradients[name]
                    count = int(value.numel())
                    value.copy_(flat[offset : offset + count].view_as(value))
                    offset += count
                if offset != int(flat.numel()):
                    fail("gradient collective bucket scatter differs")
        return modules.training_step.expected_gradient_collective_receipt_v1(
            request
        )

    return callback


def build_world_consensus_callback_v1(
    *, modules: BusinessModulesV1, parallel: Any
) -> Callable[[Any], Mapping[str, Any]]:
    """Build the transport-only WORLD8 equality gate used before optimizer.step."""

    torch_dist = importlib.import_module("torch.distributed")

    def callback(request: Any) -> Mapping[str, Any]:
        if not isinstance(request, modules.training_step.WorldConsensusRequestV1):
            fail("WORLD consensus request type differs")
        if request.rank != parallel.contract.rank:
            fail("WORLD consensus request rank differs")
        gathered: list[Any] = [None] * WORLD_SIZE
        local = {"rank": request.rank, "digest": request.digest}
        torch_dist.all_gather_object(
            gathered, local, group=parallel.world_group
        )
        expected = [
            {"rank": rank, "digest": request.digest}
            for rank in range(WORLD_SIZE)
        ]
        if gathered != expected:
            fail("pre-optimizer WORLD8 payload digests differ")
        return modules.training_step.expected_world_consensus_receipt_v1(
            request
        )

    return callback


def _release_file_sha(release_receipt: Mapping[str, Any], relative: str) -> str:
    matches = [row["sha256"] for row in release_receipt["files"] if row["path"] == relative]
    if len(matches) != 1:
        fail(f"release source identity is absent/ambiguous: {relative}")
    return matches[0]


def _official_main(
    args: argparse.Namespace,
    *,
    release_receipt: Mapping[str, Any],
    modules: BusinessModulesV1,
) -> int:
    plan, output, segment_gate, canary_inputs = _validate_cli_bindings(
        args, release_receipt=release_receipt
    )
    authority_index = Full30AuthorityRuntimeIndexV1(
        path=args.authority_manifest,
        expected_sha256=args.expected_authority_manifest_sha256,
        amplitude_path=args.amplitude_authority_manifest,
        expected_amplitude_sha256=(
            args.expected_amplitude_authority_manifest_sha256
        ),
        modules=modules,
        expected_data_sha256=args.expected_data_authority_sha256,
        expected_teacher_sha256=args.expected_teacher_authority_sha256,
        expected_nuisance_sha256=args.expected_nuisance_authority_sha256,
        profile=plan.profile,
    )
    schedule = authority_index.build_schedule(run_seed=args.run_seed)
    canonical_schedule = modules.checkpoint.canonical_schedule_v2(schedule)
    schedule_full_sha, schedule_prefix_u0_sha = (
        modules.checkpoint.schedule_digests_v2(schedule, 0)
    )
    noise_sha = schedule_noise_sha256_v1(schedule, run_seed=args.run_seed)
    if noise_sha != _sha256(
        args.expected_noise_authority_sha256,
        label="expected noise authority SHA-256",
    ):
        fail("schedule/noise authority SHA-256 differs")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            modules.legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint_root, transformer_config = modules.legacy.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise Full30ActionTrainingError(
            f"official model/source validation failed: {error}"
        ) from error
    model_sha = _sha256(args.expected_model_sha256, label="expected model SHA-256")
    if (
        transformer_config.get("num_layers") != 30
        or transformer_config.get("attention_head_dim") != 128
        or model_sha != modules.packed_release.CHECKPOINT_TREE_SHA256
    ):
        fail("official Bernini-R 1.3B model identity/geometry differs")
    checkpoint_content = modules.packed_release.validate_checkpoint_content(
        checkpoint_root,
        _plain_absolute_file(
            args.checkpoint_content_manifest,
            label="checkpoint content manifest",
        ),
        expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
    )
    if checkpoint_content["tree_sha256"] != model_sha:
        fail("validated checkpoint tree differs from expected model SHA-256")
    transformer_config_path = _plain_absolute_file(
        checkpoint_root / "transformer" / "config.json",
        label="official transformer config",
    )
    amplitude_runtime_binding = validate_amplitude_runtime_binding_v1(
        authority_index=authority_index,
        bernini_revision=bernini_revision,
        veomni_revision=veomni_revision,
        model_sha256=model_sha,
        transformer_config_sha256=file_sha256(transformer_config_path),
        sigma_table_sha256=modules.sigma.SCHEDULE_SHA256,
        psiout_protocol_sha256=_release_file_sha(
            release_receipt, "full30_action_learning_v1.py"
        ),
        official_provider_source_sha256=_release_file_sha(
            release_receipt, "full30_action_psiout_materializer_v1.py"
        ),
        executed_runtime_source_sha256=_release_file_sha(
            release_receipt, "full30_action_runtime_v1.py"
        ),
    )
    modules.legacy.activate_source_trees(bernini_root, veomni_root)

    torch = modules.torch
    torch_dist = importlib.import_module("torch.distributed")
    peft = importlib.import_module("peft")
    transformers = importlib.import_module("transformers")
    renderer_module = importlib.import_module("bernini.models.renderer")
    wan_module = importlib.import_module("bernini.models.transformer_wan")
    parallel_module = importlib.import_module("bernini.parallel")

    distributed = modules.distributed_runtime.distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.local_world_size != WORLD_SIZE
        or distributed.topology.profile != "world8-dp2-sp4"
    ):
        fail("official full30 trainer requires one exact WORLD8 DP2xSP4 node")
    device = modules.distributed_runtime.initialise_distributed(distributed)
    parallel = modules.distributed_runtime.validate_parallel_state(
        distributed, parallel_module.init_parallel_state(ulysses_size=SP_SIZE)
    )
    random.seed(args.run_seed)
    torch.manual_seed(args.run_seed)
    torch.cuda.manual_seed_all(args.run_seed)

    config = renderer_module.BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **modules.legacy.renderer_config_overrides(checkpoint_root),
    )
    config.dtype = torch.bfloat16
    modules.legacy.validate_renderer_config_mapping(
        config.to_dict(), checkpoint_root
    )
    with modules.packed_trainer.serialized_model_load():
        renderer = renderer_module.BerniniRendererModel(config)
        renderer.requires_grad_(False)
        renderer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        specs = modules.packed_core.select_projection_specs(
            renderer, "all-attention"
        )
        model = peft.get_peft_model(
            renderer,
            peft.LoraConfig(
                r=modules.packed_core.LORA_RANK,
                lora_alpha=modules.packed_core.LORA_ALPHA,
                lora_dropout=0.0,
                bias="none",
                target_modules=[item.name for item in specs],
            ),
        )
        transformer = model.get_base_model().diff_dec.transformer
        modules.packed_core.install_typed_patch_embedding(transformer)
        model.to(device)
    model.train()
    base_renderer = model.get_base_model()
    base_renderer.t5_text_encoder.eval()
    named_trainable = modules.packed_core.trainable_named_parameters(model)
    trainable_count = modules.packed_core.verify_trainable_parameter_count(
        model, "all-attention"
    )
    lora_installation = modules.packed_core.validate_lora_installation(model, specs)
    if (
        len(specs) != 240
        or trainable_count != 188_946_432
        or any(
            parameter.dtype != torch.float32 or parameter.device != device
            for _, parameter in named_trainable
        )
    ):
        fail("full30 FP32 trainable capacity/device closure differs")
    architecture = modules.packed_core.architecture_receipt(
        "all-attention", specs
    )
    modules.distributed_runtime.synchronize_initial_parameters(
        named_trainable, parallel.world_group
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(checkpoint_root),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=modules.legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    conditions: dict[str, Any] = {}
    with torch.inference_mode():
        for pair in authority_index.pairs.values():
            instruction_sha = pair["instruction_utf8_sha256"]
            if instruction_sha not in conditions:
                conditions[instruction_sha] = _condition_from_instruction(
                    instruction=pair["instruction"],
                    instruction_sha256=instruction_sha,
                    tokenizer=tokenizer,
                    renderer=base_renderer,
                    device=device,
                    modules=modules,
                    task_name=f"full30-{pair['branch']}",
                )
        noop_sha = hashlib.sha256(NOOP_INSTRUCTION.encode("utf-8")).hexdigest()
        noop_condition = _condition_from_instruction(
            instruction=NOOP_INSTRUCTION,
            instruction_sha256=noop_sha,
            tokenizer=tokenizer,
            renderer=base_renderer,
            device=device,
            modules=modules,
            task_name="full30-noop",
        )
    base_renderer.t5_text_encoder = None
    del tokenizer
    torch.cuda.empty_cache()
    if base_renderer.t5_text_encoder is not None:
        fail("frozen T5 was not retired after authority-bound conditions")

    vae_mean, vae_std, z_dim = modules.legacy._vae_statistics(checkpoint_root)
    if z_dim != 16:
        fail("official Wan VAE latent width differs")
    vae_mean = vae_mean.reshape(1, 16, 1, 1, 1).float().contiguous()
    vae_std = vae_std.reshape(1, 16, 1, 1, 1).float().contiguous()
    if not bool((vae_std > 0).all().item()):
        fail("official VAE standard deviation differs")
    rope = wan_module.WanRotaryPosEmbed(
        128, (1, 2, 2), 1024, use_src_id_rotary_emb=True
    )
    branch_runtime = modules.branch_runtime.Full30ActionRuntimeV1(
        renderer=base_renderer,
        transformer=transformer,
        adapter_controller=model,
    )
    optimizer = modules.optimizer.Full30ActionFirstOptimizerV1(named_trainable)
    inventory = modules.checkpoint.inventory_identity_v2(optimizer)
    inventory_sha = inventory["inventory_sha256"]
    modules.distributed_runtime.digest_consensus(
        inventory_sha,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="full30 authoritative trainable inventory",
    )

    bindings = modules.checkpoint.CheckpointBindings(
        arm=args.arm,
        release_sha256=release_receipt["release_sha256"],
        model_sha256=model_sha,
        data_sha256=authority_index.projection_digests["data_sha256"],
        teacher_sha256=authority_index.projection_digests["teacher_sha256"],
        nuisance_sha256=authority_index.projection_digests["nuisance_sha256"],
        noise_sha256=noise_sha,
        runtime_sha256=_release_file_sha(
            release_receipt, "full30_action_runtime_v1.py"
        ),
        objective_sha256=_release_file_sha(
            release_receipt, "full30_action_learning_v1.py"
        ),
    )

    fresh_formal_canary_evidence: Optional[Mapping[str, Any]] = None
    if canary_inputs is not None:
        fresh_formal_canary_evidence = (
            validate_reduced_canary_to_fresh_formal_v1(
                inputs=canary_inputs,
                arm=args.arm,
                formal_plan=plan,
                release_sha256=release_receipt["release_sha256"],
                model_sha256=model_sha,
                authority_manifest_sha256=authority_index.manifest_sha256,
                amplitude_authority_manifest_sha256=(
                    authority_index.amplitude.manifest_file_sha256
                ),
                authority_projections=authority_index.projection_digests,
                noise_authority_sha256=noise_sha,
                schedule_full_sha256=schedule_full_sha,
                schedule_prefix_u0_sha256=schedule_prefix_u0_sha,
                architecture=architecture,
                authoritative_inventory_sha256=inventory_sha,
                trainable_parameter_count=trainable_count,
                bindings=bindings.as_dict(),
                formal_authority_validation=(
                    authority_index.validation_receipt
                ),
                formal_amplitude_validation=(
                    authority_index.amplitude.validation_receipt
                ),
                optimizer_code_sha256=_release_file_sha(
                    release_receipt, "full30_action_optimizer_v1.py"
                ),
                training_step_code_sha256=_release_file_sha(
                    release_receipt, "full30_action_training_step_v1.py"
                ),
            )
        )

    if distributed.rank == 0:
        output.mkdir(mode=0o700)
        (output / "checkpoints").mkdir(mode=0o700)
        (output / "updates").mkdir(mode=0o700)
        (output / "checkpoint-references").mkdir(mode=0o700)
    torch_dist.barrier(group=parallel.world_group)

    history: list[Mapping[str, Any]] = []
    checkpoint_records: list[Mapping[str, Any]] = []
    previous_checkpoint: Any = None
    if plan.requires_resume:
        current_reference = _load_checkpoint_reference_file(
            path=args.resume_reference,
            expected_sha256=args.expected_resume_reference_sha256,
            checkpoint_module=modules.checkpoint,
            label="resume current checkpoint reference",
        )
        predecessor_reference = _load_checkpoint_reference_file(
            path=args.resume_previous_reference,
            expected_sha256=args.expected_resume_previous_reference_sha256,
            checkpoint_module=modules.checkpoint,
            label="resume predecessor checkpoint reference",
        )
        loaded = modules.checkpoint.load_checkpoint(
            args.resume_checkpoint,
            optimizer=optimizer,
            expected_bindings=bindings,
            expected_full_schedule=schedule,
            expected_completed_updates=plan.start_update,
            expected_previous_checkpoint=predecessor_reference,
            expected_reference=current_reference,
            authoritative_inventory_sha256=inventory_sha,
            rank=distributed.rank,
            world_size=WORLD_SIZE,
            status_consensus=lambda local: _checkpoint_status_consensus(
                local, parallel=parallel
            ),
        )
        modules.checkpoint.restore_checkpoint_state(
            loaded,
            optimizer=optimizer,
            rng_transaction_factory=lambda state: _RngRestoreTransactionV1(
                state,
                rank=distributed.rank,
                local_rank=distributed.local_rank,
                modules=modules,
            ),
        )
        history = [dict(row) for row in loaded.history]
        previous_checkpoint = loaded
    else:
        initial_rng = capture_world8_rng_state_v1(
            modules=modules, parallel=parallel
        )
        initial_target = output / "checkpoints" / "u00000000"
        initial_reference = _save_checkpoint_world8(
            target=initial_target,
            optimizer=optimizer,
            completed_updates=0,
            schedule=schedule,
            history=history,
            rng_state=initial_rng,
            bindings=bindings,
            inventory_sha256=inventory_sha,
            previous=None,
            parallel=parallel,
            modules=modules,
        )
        previous_checkpoint = initial_reference
        if distributed.rank == 0:
            reference_path = output / "checkpoint-references" / "u00000000.json"
            _atomic_create_json(reference_path, _reference_file_payload(initial_reference))
            checkpoint_records.append(
                {
                    "completed_updates": 0,
                    "path": str(initial_target),
                    "reference_path": str(reference_path),
                    "reference": initial_reference.as_dict(),
                }
            )
    torch_dist.barrier(group=parallel.world_group)
    if optimizer.update_count != plan.start_update:
        fail("optimizer update count differs at segment entry")

    gradient_mean = build_gradient_mean_callback_v1(
        modules=modules, parallel=parallel
    )
    world_consensus = build_world_consensus_callback_v1(
        modules=modules, parallel=parallel
    )

    def optimizer_all_reduce(values: Any) -> None:
        torch_dist.all_reduce(
            values, op=torch_dist.ReduceOp.SUM, group=parallel.world_group
        )
        return None

    for update in range(plan.start_update, plan.stop_update):
        before_identity = modules.checkpoint.optimizer_state_identity_v2(optimizer)
        rng_before = capture_world8_rng_state_v1(modules=modules, parallel=parallel)
        global_rows = schedule[update * GLOBAL_BATCH : (update + 1) * GLOBAL_BATCH]
        local_rows = tuple(
            row for row in global_rows if row.dp_rank == distributed.arm_index
        )
        prepared = tuple(
            prepare_runtime_record_v1(
                scheduled=row,
                authority_index=authority_index,
                conditions=conditions,
                noop_condition=noop_condition,
                vae_mean=vae_mean,
                vae_std=vae_std,
                rope=rope,
                device=device,
            )
            for row in local_rows
        )
        step_result = execute_one_update_v1(
            arm=args.arm,
            records=prepared,
            runtime=branch_runtime,
            optimizer=optimizer,
            full_schedule=schedule,
            rank=distributed.rank,
            training_step_module=modules.training_step,
            gradient_mean=gradient_mean,
            world_consensus=world_consensus,
            optimizer_all_reduce_sum=optimizer_all_reduce,
            autocast_context=lambda: torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ),
        )
        global_receipt = gather_global_update_receipt_v1(
            step_result=step_result,
            prepared_records=prepared,
            parallel=parallel,
            modules=modules,
        )
        after_identity = modules.checkpoint.optimizer_state_identity_v2(optimizer)
        rng_after = capture_world8_rng_state_v1(modules=modules, parallel=parallel)
        completed = update + 1
        _schedule_full, schedule_prefix = modules.checkpoint.schedule_digests_v2(
            canonical_schedule, completed
        )
        global_receipt = bind_global_update_authorities_v1(
            receipt=global_receipt,
            scheduled_global_rows=global_rows,
            schedule_full_sha256=schedule_full_sha,
            schedule_prefix_sha256=schedule_prefix,
            noise_authority_sha256=noise_sha,
            teacher_authority_sha256=authority_index.projection_digests[
                "teacher_sha256"
            ],
            data_teacher_authority_manifest_sha256=(
                authority_index.manifest_sha256
            ),
            amplitude_authority_manifest_sha256=(
                authority_index.amplitude.manifest_file_sha256
            ),
        )
        rng_before_sha = rng_artifact_sha256_v1(
            rng_before, checkpoint_module=modules.checkpoint
        )
        rng_after_sha = rng_artifact_sha256_v1(
            rng_after, checkpoint_module=modules.checkpoint
        )
        history_row = modules.checkpoint.build_history_row_v2(
            update_count=completed,
            optimizer_receipt_digest=global_receipt["optimizer_receipt"][
                "receipt_digest"
            ],
            parameters_before_sha256=before_identity["trainable_state_sha256"],
            parameters_after_sha256=after_identity["trainable_state_sha256"],
            optimizer_v_before_sha256=before_identity[
                "optimizer_v_state_sha256"
            ],
            optimizer_v_after_sha256=after_identity[
                "optimizer_v_state_sha256"
            ],
            rng_before_sha256=rng_before_sha,
            rng_after_sha256=rng_after_sha,
            schedule_prefix_sha256=schedule_prefix,
        )
        history.append(history_row)
        if distributed.rank == 0:
            _atomic_create_json(
                output / "updates" / f"u{completed:08d}.json",
                global_receipt,
            )
        if completed in plan.checkpoint_updates:
            target = output / "checkpoints" / f"u{completed:08d}"
            reference = _save_checkpoint_world8(
                target=target,
                optimizer=optimizer,
                completed_updates=completed,
                schedule=schedule,
                history=history,
                rng_state=rng_after,
                bindings=bindings,
                inventory_sha256=inventory_sha,
                previous=previous_checkpoint,
                parallel=parallel,
                modules=modules,
            )
            previous_checkpoint = reference
            if distributed.rank == 0:
                reference_path = (
                    output
                    / "checkpoint-references"
                    / f"u{completed:08d}.json"
                )
                _atomic_create_json(reference_path, _reference_file_payload(reference))
                checkpoint_records.append(
                    {
                        "completed_updates": completed,
                        "path": str(target),
                        "reference_path": str(reference_path),
                        "reference": reference.as_dict(),
                    }
                )
        torch_dist.barrier(group=parallel.world_group)
        del prepared, step_result, global_receipt

    if optimizer.update_count != plan.stop_update or len(history) != plan.stop_update:
        fail("segment terminal optimizer/history count differs")
    final_identity = modules.checkpoint.optimizer_state_identity_v2(optimizer)
    final_prefix = modules.checkpoint.schedule_digests_v2(
        canonical_schedule, plan.stop_update
    )[1]
    if distributed.rank == 0:
        unsigned = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "method": METHOD,
            "complete": True,
            "arm": args.arm,
            "profile": plan.profile,
            "start_update": plan.start_update,
            "stop_update": plan.stop_update,
            "optimizer_updates_executed": plan.stop_update - plan.start_update,
            "optimizer_update_count": optimizer.update_count,
            "boundary_plan": dict(plan.receipt()),
            "segment_gate": None if segment_gate is None else dict(segment_gate),
            "fresh_formal_canary_evidence": (
                None
                if fresh_formal_canary_evidence is None
                else dict(fresh_formal_canary_evidence)
            ),
            "bindings": bindings.as_dict(),
            "release": dict(release_receipt),
            "authority_validation": dict(authority_index.validation_receipt),
            "amplitude_authority_validation": dict(
                authority_index.amplitude.validation_receipt
            ),
            "amplitude_runtime_binding": dict(amplitude_runtime_binding),
            "data_teacher_authority_manifest_sha256": (
                authority_index.manifest_sha256
            ),
            "amplitude_authority_manifest_sha256": (
                authority_index.amplitude.manifest_file_sha256
            ),
            "authority_projections": dict(authority_index.projection_digests),
            "schedule_full_sha256": schedule_full_sha,
            "schedule_prefix_sha256": final_prefix,
            "noise_authority_sha256": noise_sha,
            "architecture": architecture,
            "lora_installation": lora_installation,
            "trainable_parameter_count": trainable_count,
            "authoritative_inventory_sha256": inventory_sha,
            "final_optimizer_identity": dict(final_identity),
            "checkpoints": checkpoint_records,
            "official_model": {
                "model_sha256": model_sha,
                "checkpoint_content": checkpoint_content,
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "base_compute_dtype": "torch.bfloat16/autocast",
                "trainable_dtype": "torch.float32",
                "optimizer_v_dtype": "torch.float32",
            },
            "objective": {
                "post_head_stage": modules.branch_runtime.POST_HEAD_STAGE,
                "psiout_shape": [21, 32],
                "action_direction_plus_same_mode_and_sealed_amplitude_floor": True,
                "amplitude_floor_is_max_same_mode_and_sealed_minimum": True,
                "noop_target_is_epsilon_minus_real_source": True,
                "action_row_regresses_full_source_trajectory": False,
                "main_noop_restoration_replay_per_local_rank": (
                    4 if args.arm == "action+retain" else 0
                ),
                "execution_authority": {
                    "population_profile": (
                        authority_index.validation_receipt.get(
                            "population_profile"
                        )
                        if authority_index.is_disposable_canary
                        else None
                    ),
                    "formal_authority": not authority_index.is_disposable_canary,
                    "mechanism_only": authority_index.is_disposable_canary,
                    "generalization": False,
                    "identity_generalization": False,
                    "event_family_generalization": False,
                    "non_executable_evidence_trainer_read_authorized": False,
                    "real_source_identity_or_generalization_claimed": False,
                    "tail_serialization_only": (
                        authority_index.is_disposable_canary
                    ),
                    "maximum_updates": (
                        2
                        if authority_index.is_disposable_canary
                        else MAX_UPDATES
                    ),
                    "u3_authorized": not authority_index.is_disposable_canary,
                    "synthetic_target_index1_bytes_read": False,
                    "synthetic_target_bytes_read": False,
                    "schedule_authority": (
                        None
                        if authority_index.schedule_authority_receipt is None
                        else dict(
                            authority_index.schedule_authority_receipt
                        )
                    ),
                },
            },
            "distributed": {
                "world_size": WORLD_SIZE,
                "dp_size": DP_SIZE,
                "sp_size": SP_SIZE,
                "gradient_reduction": "SP4-mean-then-DP2-mean",
                "global_batch": GLOBAL_BATCH,
                "physical_evaluations_per_update": (
                    32 if args.arm == "action+retain" else 24
                ),
            },
            "synthetic_target_index1_bytes_read": False,
            "synthetic_target_bytes_read": False,
            "parent_allocation_released": False,
        }
        receipt = _seal(unsigned)
        _atomic_create_json(output / "receipt.json", receipt)
        print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
    torch_dist.barrier(group=parallel.world_group)
    torch_dist.destroy_process_group()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    # No authority/model/objective/optimizer import is reachable before this
    # exact executed-byte closure passes.
    release_receipt = validate_executed_release_v1(
        method_root=args.method_root,
        manifest=args.release_manifest,
        expected_manifest_sha256=args.expected_release_manifest_sha256,
        expected_release_sha256=args.expected_release_sha256,
    )
    modules = load_business_modules_v1(release_receipt)
    return _official_main(
        args, release_receipt=release_receipt, modules=modules
    )


__all__ = [
    "ARMS",
    "BoundaryPlanV1",
    "BusinessModulesV1",
    "CANARY_CHECKPOINTS",
    "FreshFormalCanaryInputsV1",
    "Full30ActionTrainingError",
    "PreparedRecordV1",
    "REDUCED_CANARY_TO_FRESH_FORMAL_EVIDENCE_SCHEMA_VERSION",
    "REDUCED_CANARY_TO_FRESH_FORMAL_GATE_SCHEMA_VERSION",
    "TeacherPacketV1",
    "bind_global_update_authorities_v1",
    "build_gradient_mean_callback_v1",
    "build_boundary_plan_v1",
    "build_world_consensus_callback_v1",
    "canonical_json_bytes",
    "execute_one_update_v1",
    "gather_global_update_receipt_v1",
    "load_business_modules_v1",
    "main",
    "object_sha256",
    "parser",
    "schedule_noise_sha256_v1",
    "validate_amplitude_runtime_binding_v1",
    "validate_executed_release_v1",
    "validate_fresh_formal_canary_evidence_v1",
    "validate_reduced_canary_to_fresh_formal_v1",
    "validate_segment_gate_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
