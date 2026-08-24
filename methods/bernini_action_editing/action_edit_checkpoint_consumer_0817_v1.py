#!/usr/bin/env python3
"""Fail-closed fresh-process consumer for the 0817 PRE_D0 checkpoints.

This module does not train, decode video, select a checkpoint, or authorize a
scientific claim.  It authenticates the frozen two-update engineering release,
the complete P0/P1/P2 run receipt, and one selected checkpoint before copying
FP32 trainable state into a freshly constructed Bernini-R model.

There are deliberately two different parity authorities:

* ``posthoc_fresh_a_b`` proves two independently constructed consumers agree;
* ``training_attached_pre_save`` is written inside a future training process
  before checkpoint publication and is the only authority that can prove the
  requested training-state -> fresh-process forward parity.

The r2 runner did not persist the latter artifact.  Consumers must report that
absence; they must never relabel post-hoc parity as training-attached parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import stat
import sys
import time
import types
from typing import Any, Mapping, NoReturn, Optional, Sequence
import uuid


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

METHOD = "bernini-action-edit-checkpoint-consumer-0817-v1"
AUTHORITY = "PRE_D0_ENGINEERING_ONLY"
CONSUMER_SCHEMA = "bernini-action-edit-fresh-consumer-receipt-v1"
TRAINING_REFERENCE_SCHEMA = (
    "bernini-action-edit-training-attached-fixed-forward-reference-v1"
)
TRAINING_REFERENCE_BINDING_SCHEMA = (
    "bernini-action-edit-training-reference-checkpoint-binding-v1"
)
FRESH_PARITY_SCHEMA = "bernini-action-edit-fresh-a-b-fixed-forward-parity-v1"
FRESH_WORLD8_PARITY_SCHEMA = (
    "bernini-action-edit-fresh-world8-a-b-fixed-forward-parity-v1"
)
TRAIN_RUNNER_METHOD = "bernini-action-edit-large-lora-0817-v1"
TRAIN_RECEIPT_SCHEMA = "bernini-action-edit-large-lora-0817-pre-d0-receipt-v1"
FULL_STATE_SCHEMA = "bernini-action-edit-full-state-v1"
RUNTIME_STATE_SCHEMA = "bernini-action-edit-runtime-state-v1"
RELEASE_MANIFEST_SCHEMA = (
    "bernini-action-edit-large-lora-0817-pre-d0-release-manifest-v1"
)
RELEASE_MEMBER_ROOT = "methods/bernini_action_editing"
PINNED_TRAIN_RUNNER_SHA256 = (
    "edf3d1d2a77cb2f713968f537ce85a7d92f0b7347a0474419fe5562fbd319bd9"
)
PINNED_PREDICTOR_SOURCE_SHA256 = (
    "464cd500f0ba1edb6cbe6d4f07287bfff346ae0ba7968c0d7c7f3cc7cb667308"
)
PINNED_CONDITIONER_ABI_SHA256 = (
    "04c2fc8ff48fb8b027e912cd6c9c58cf19d4b554c84127fb6623268a9d1e398b"
)
PINNED_R2_CAMPAIGN_RECEIPT_SHA256 = (
    "8014b7b71413318d80162fba12b73d83d6b9d9de5ea57ad295643a238b0f8c0e"
)
PINNED_R2_RELEASE_MANIFEST_SHA256 = (
    "671179995a64f20ee773273e84b5eb3f1f0bbd018fbfa3c0c6dc41d56c5555f5"
)
PINNED_R2_RELEASE_MEMBER_SET_SHA256 = (
    "b2556330c45cc8db8b8b6497e821fd773fe724113c8bb1860a5b343301776306"
)
PINNED_R2_P_STATE_SHA256 = {
    0: "e26c5fd00a581e7710b60eef29a691763b03915ee73c25ffec82cb0bc8bba891",
    1: "d40391c7a2c9fa72e02b9dedc44f835b9eb3ce0b8f626cf0e36e576efb961970",
    2: "5f9c31e84ab9ec4330b07d86cb1a2fc79c7aa365f4bf88a9cdffc0c244dcaa3e",
}
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
CHECKPOINT_STEPS = (0, 1, 2)
TRANSFORMER_BLOCKS = 30
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
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROCESS_SESSION_ID = hashlib.sha256(
    (
        f"{os.getpid()}:{time.time_ns()}:{uuid.uuid4().hex}:"
        f"{Path(sys.executable).resolve()}"
    ).encode("utf-8")
).hexdigest()
_PROCESS_LOAD_CONSUMED = False


class CheckpointConsumerError(RuntimeError):
    """Raised before any unverified checkpoint can influence inference."""


def fail(message: str) -> NoReturn:
    raise CheckpointConsumerError(message)


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be one lowercase full SHA-256")
    return value


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be one absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        before = requested.lstat()
    except OSError as error:
        raise CheckpointConsumerError(f"{label} is unavailable: {error}") from error
    if (
        resolved != requested
        or not stat.S_ISREG(before.st_mode)
        or requested.is_symlink()
    ):
        fail(f"{label} canonical file type differs")
    return requested


def _plain_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be one absolute non-symlink directory")
    try:
        resolved = requested.resolve(strict=True)
        mode = requested.lstat().st_mode
    except OSError as error:
        raise CheckpointConsumerError(f"{label} is unavailable: {error}") from error
    if resolved != requested or not stat.S_ISDIR(mode) or requested.is_symlink():
        fail(f"{label} canonical directory type differs")
    return requested


def _stable_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if _identity(before) != _identity(after) or len(payload) > maximum:
        fail(f"{label} changed during read or exceeds its size bound")
    return payload


def _read_json_snapshot(
    value: str | Path,
    *,
    expected_sha256: str,
    label: str,
    maximum: int = 32 * 1024 * 1024,
) -> tuple[Path, Mapping[str, Any]]:
    expected = _require_sha(expected_sha256, label=f"expected {label} SHA")
    path = _plain_file(value, label=label)
    payload = _stable_bytes(path, maximum=maximum, label=label)
    if hashlib.sha256(payload).hexdigest() != expected:
        fail(f"{label} SHA differs")
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointConsumerError(f"{label} is not UTF-8 JSON") from error
    if not isinstance(raw, Mapping):
        fail(f"{label} root must be an object")
    return path, raw


@dataclass(frozen=True)
class ReleasePreflight:
    manifest_path: Path
    manifest_sha256: str
    member_root: Path
    members: tuple[Mapping[str, Any], ...]
    source_bytes: Mapping[str, bytes]


def authenticate_release_before_import(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
    method_root: str | Path = METHOD_ROOT,
) -> ReleasePreflight:
    """Authenticate every executable member before importing training code."""

    if expected_sha256 != PINNED_R2_RELEASE_MANIFEST_SHA256:
        fail("frozen r2 release manifest trust-root SHA differs")
    root = _plain_directory(method_root, label="release member root")
    path, raw = _read_json_snapshot(
        manifest_path,
        expected_sha256=expected_sha256,
        label="release manifest",
        maximum=1024 * 1024,
    )
    if (
        set(raw) != {"schema_version", "member_root", "files"}
        or raw.get("schema_version") != RELEASE_MANIFEST_SCHEMA
        or raw.get("member_root") != RELEASE_MEMBER_ROOT
        or not isinstance(raw.get("files"), list)
    ):
        fail("release manifest envelope differs")
    expected_paths = tuple(sorted(RELEASE_FILES_AND_MODES))
    rows = raw["files"]
    if len(rows) != len(expected_paths):
        fail("release manifest member count differs")
    normalized = []
    authenticated_sources: dict[str, bytes] = {}
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
        ):
            fail("release manifest exact sorted member closure differs")
        expected_member_sha = _require_sha(
            row.get("sha256"), label=f"release member {relative} SHA"
        )
        member = _plain_file(root / expected_relative, label=f"release member {relative}")
        before = member.lstat()
        source_bytes = _stable_bytes(
            member,
            maximum=2 * 1024 * 1024,
            label=f"release member {relative}",
        )
        if (
            stat.S_IMODE(before.st_mode) != row["mode"]
            or before.st_size != row["size"]
            or hashlib.sha256(source_bytes).hexdigest() != expected_member_sha
            or _identity(before) != _identity(member.lstat())
        ):
            fail(f"release member identity differs: {relative}")
        if (
            relative == "train_action_edit_large_lora_0817_v1.py"
            and expected_member_sha != PINNED_TRAIN_RUNNER_SHA256
        ):
            fail("frozen training runner SHA differs")
        if (
            relative == "action_plan_predictor_v1.py"
            and expected_member_sha != PINNED_PREDICTOR_SOURCE_SHA256
        ):
            fail("frozen ActionPlanPredictorV1 source SHA differs")
        normalized.append(dict(row))
        authenticated_sources[expected_relative] = source_bytes
    return ReleasePreflight(
        manifest_path=path,
        manifest_sha256=expected_sha256,
        member_root=root,
        members=tuple(normalized),
        source_bytes=dict(authenticated_sources),
    )


def _load_authenticated_source_module(
    *,
    module_name: str,
    source_path: Path,
    source_bytes: bytes,
    expected_sha256: str,
) -> Any:
    """Execute exactly the authenticated source snapshot without import caches.

    In particular, this does not consult ``sys.meta_path``, ``__pycache__``, a
    sourceless ``.pyc``, or an extension-module shadow.  The source bytes that
    passed preflight are the bytes compiled and executed here.
    """

    expected = _require_sha(expected_sha256, label=f"{module_name} source SHA")
    if (
        not isinstance(source_bytes, bytes)
        or hashlib.sha256(source_bytes).hexdigest() != expected
        or module_name in sys.modules
    ):
        fail(f"authenticated source-module precondition differs: {module_name}")
    path = _plain_file(source_path, label=f"{module_name} authenticated source")
    try:
        code = compile(
            source_bytes,
            str(path),
            "exec",
            flags=0,
            dont_inherit=True,
            optimize=0,
        )
    except (SyntaxError, ValueError, TypeError) as error:
        raise CheckpointConsumerError(
            f"authenticated source did not compile: {module_name}"
        ) from error
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__cached__ = None
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    if sys.modules.get(module_name) is not module or module.__file__ != str(path):
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        fail(f"authenticated source module replaced its identity: {module_name}")
    return module


def _load_authenticated_release_module(
    release: ReleasePreflight, module_name: str
) -> Any:
    filename = f"{module_name}.py"
    rows = {
        str(row.get("path")): row
        for row in release.members
        if isinstance(row, Mapping)
    }
    row = rows.get(filename)
    source = release.source_bytes.get(filename)
    if (
        set(rows) != set(RELEASE_FILES_AND_MODES)
        or set(release.source_bytes) != set(RELEASE_FILES_AND_MODES)
        or not isinstance(row, Mapping)
        or not isinstance(source, bytes)
        or row.get("size") != len(source)
    ):
        fail("authenticated release source-byte closure differs")
    return _load_authenticated_source_module(
        module_name=module_name,
        source_path=release.member_root / filename,
        source_bytes=source,
        expected_sha256=str(row.get("sha256")),
    )


def import_authenticated_training_modules(
    release: ReleasePreflight,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Load exact preflight bytes and reject any already-imported shadow copy."""

    all_release_module_names = tuple(
        Path(filename).stem for filename in sorted(RELEASE_FILES_AND_MODES)
    )
    if any(name in sys.modules for name in all_release_module_names):
        fail("fresh consumer requires release modules absent before source-byte load")
    if release.member_root != METHOD_ROOT.resolve(strict=True):
        fail("authenticated release root is not the executed consumer method root")
    predictor = _load_authenticated_release_module(
        release, "action_plan_predictor_v1"
    )
    runner = _load_authenticated_release_module(
        release, "train_action_edit_large_lora_0817_v1"
    )
    for module, filename, expected in (
        (runner, "train_action_edit_large_lora_0817_v1.py", PINNED_TRAIN_RUNNER_SHA256),
        (predictor, "action_plan_predictor_v1.py", PINNED_PREDICTOR_SOURCE_SHA256),
    ):
        module_path = _plain_file(getattr(module, "__file__", ""), label=filename)
        if module_path != release.member_root / filename or file_sha256(module_path) != expected:
            fail(f"authenticated imported module identity differs: {filename}")
    closure = runner.validate_release_manifest(
        release.manifest_path,
        expected_sha256=release.manifest_sha256,
        method_root=release.member_root,
    )
    if closure.get("member_count") != len(RELEASE_FILES_AND_MODES):
        fail("runner release closure differs after import")
    return runner, predictor, closure


@dataclass(frozen=True)
class CampaignReceipt:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    parameter_digests: Mapping[int, str]
    checkpoint_records: Mapping[int, Mapping[str, Any]]


def validate_campaign_receipt(
    receipt_path: str | Path, *, expected_sha256: str
) -> CampaignReceipt:
    """Validate the complete nonpromotable P0/P1/P2 authority envelope."""

    if expected_sha256 != PINNED_R2_CAMPAIGN_RECEIPT_SHA256:
        fail("frozen r2 campaign receipt trust-root SHA differs")
    path, raw = _read_json_snapshot(
        receipt_path,
        expected_sha256=expected_sha256,
        label="campaign receipt",
    )
    unsigned = dict(raw)
    observed_digest = unsigned.pop("receipt_digest", None)
    if observed_digest != object_sha256(unsigned):
        fail("campaign receipt self-digest differs")
    required = {
        "schema_version": TRAIN_RECEIPT_SCHEMA,
        "method": TRAIN_RUNNER_METHOD,
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
        "optimizer_steps": 2,
        "fresh_official_base": True,
        "resume_consumed": False,
        "checkpoint_steps": [0, 1, 2],
        "all_checkpoints_rank0_full_trainable_optimizer_roundtrip_reloaded": True,
        "all_checkpoints_all8_runtime_state_bytes_persisted": True,
        "all_checkpoints_rank0_runtime_state_roundtrip_reloaded": True,
        "terminal_world8_consensus_precedes_receipt_publication": True,
        "parent_allocation_released": False,
    }
    if any(raw.get(key) != expected for key, expected in required.items()):
        fail("campaign receipt authority/completion fields differ")
    digests_raw = raw.get("parameter_digests")
    if not isinstance(digests_raw, Mapping) or set(digests_raw) != {"0", "1", "2"}:
        fail("campaign P0/P1/P2 digest set differs")
    parameter_digests = {
        step: _require_sha(digests_raw[str(step)], label=f"P{step} parameter digest")
        for step in CHECKPOINT_STEPS
    }
    if (
        len(set(parameter_digests.values())) != 3
        or parameter_digests != PINNED_R2_P_STATE_SHA256
    ):
        fail("campaign P0/P1/P2 states are not three distinct states")
    checkpoint_rows = raw.get("checkpoints")
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != 3:
        fail("campaign checkpoint record count differs")
    checkpoint_records: dict[int, Mapping[str, Any]] = {}
    for step, row in zip(CHECKPOINT_STEPS, checkpoint_rows):
        if not isinstance(row, Mapping) or row.get("step") != step:
            fail("campaign checkpoint record order differs")
        for key in (
            "adapter_sha256",
            "optimizer_sha256",
            "runtime_state_sha256",
            "metadata_sha256",
        ):
            _require_sha(row.get(key), label=f"checkpoint {step} {key}")
        if (
            row.get("rank0_full_trainable_optimizer_roundtrip_reload_verified")
            is not True
            or row.get("all8_runtime_state_bytes_persisted_verified") is not True
            or row.get("rank0_runtime_state_roundtrip_reload_verified") is not True
        ):
            fail("campaign checkpoint roundtrip evidence differs")
        checkpoint_records[step] = dict(row)
    architecture = raw.get("architecture")
    state_abi = architecture.get("action_plan_state_dict_abi") if isinstance(
        architecture, Mapping
    ) else None
    if (
        not isinstance(architecture, Mapping)
        or architecture.get("action_plan_predictor_source_sha256")
        != PINNED_PREDICTOR_SOURCE_SHA256
        or architecture.get("exact30_post_block_injection") is not True
        or architecture.get("target_plan_input") is not False
        or not isinstance(state_abi, Mapping)
        or state_abi.get("abi_sha256") != PINNED_CONDITIONER_ABI_SHA256
    ):
        fail("campaign predictor/injection architecture ABI differs")
    provenance = raw.get("provenance")
    release_closure = provenance.get("release_closure") if isinstance(
        provenance, Mapping
    ) else None
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("runner_source_sha256") != PINNED_TRAIN_RUNNER_SHA256
        or provenance.get("predictor_source_sha256")
        != PINNED_PREDICTOR_SOURCE_SHA256
        or not isinstance(release_closure, Mapping)
        or release_closure.get("schema_version") != RELEASE_MANIFEST_SCHEMA
        or release_closure.get("sha256")
        != PINNED_R2_RELEASE_MANIFEST_SHA256
        or release_closure.get("member_set_sha256")
        != PINNED_R2_RELEASE_MEMBER_SET_SHA256
        or release_closure.get("member_root") != RELEASE_MEMBER_ROOT
        or release_closure.get("member_count") != len(RELEASE_FILES_AND_MODES)
        or release_closure.get(
            "regular_non_symlink_exact_modes_sizes_hashes_verified"
        )
        is not True
    ):
        fail("campaign frozen release provenance differs")
    optimizer = raw.get("optimizer")
    if (
        not isinstance(optimizer, Mapping)
        or optimizer.get("class") != "torch.optim.AdamW"
        or optimizer.get("fresh_state") is not True
        or optimizer.get("scheduler") != "constant_lr_no_scheduler_object"
        or optimizer.get("topology")
        != "engineering_equivalent_replicated_not_formal_sharded"
    ):
        fail("campaign optimizer/scheduler declaration differs")
    distributed = raw.get("distributed")
    if (
        not isinstance(distributed, Mapping)
        or distributed.get("world_size") != WORLD_SIZE
        or distributed.get("dp_size") != DP_SIZE
        or distributed.get("sp_size") != SP_SIZE
        or distributed.get("pre_sp_complete_source_predictor") is not True
        or distributed.get("source_and_padding_bit_exact_under_injection") is not True
        or distributed.get("checkpoint_forward_and_recompute_calls_per_block") != 2
    ):
        fail("campaign WORLD8/action route declaration differs")
    dataset = raw.get("dataset")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("formal_0817_manifest_consumed") is not False
        or dataset.get("effective_scientific_sample_size_claimed") is not False
        or dataset.get("teacher_anchor_qualification_claimed") is not False
    ):
        fail("campaign nonformal data authority differs")
    return CampaignReceipt(
        path=path,
        sha256=expected_sha256,
        raw=dict(raw),
        parameter_digests=parameter_digests,
        checkpoint_records=checkpoint_records,
    )


@dataclass(frozen=True)
class CheckpointPreflight:
    directory: Path
    step: int
    metadata: Mapping[str, Any]
    metadata_sha256: str
    adapter_path: Path
    optimizer_path: Path
    runtime_path: Path
    parameter_sha256: str


def validate_checkpoint_preflight(
    checkpoint_dir: str | Path,
    *,
    step: int,
    campaign: CampaignReceipt,
    expected_release_manifest_sha256: str,
) -> CheckpointPreflight:
    if type(step) is not int or step not in CHECKPOINT_STEPS:
        fail("selected checkpoint step must be exactly P0, P1, or P2")
    directory = _plain_directory(checkpoint_dir, label="selected checkpoint")
    record = campaign.checkpoint_records[step]
    try:
        recorded_path = Path(str(record.get("path"))).expanduser().resolve(strict=True)
    except OSError as error:
        raise CheckpointConsumerError("recorded checkpoint path is unavailable") from error
    if recorded_path != directory:
        fail("selected checkpoint path differs from the authenticated campaign receipt")
    children = tuple(directory.iterdir())
    if set(item.name for item in children) != {
        "full_trainable_state.pt",
        "optimizer.pt",
        "runtime_state.pt",
        "metadata.json",
    } or any(item.is_symlink() or not item.is_file() for item in children):
        fail("selected checkpoint exact four-file closure differs")
    metadata_path, metadata = _read_json_snapshot(
        directory / "metadata.json",
        expected_sha256=record["metadata_sha256"],
        label="checkpoint metadata",
    )
    required = {
        "schema_version": TRAIN_RECEIPT_SCHEMA,
        "method": TRAIN_RUNNER_METHOD,
        "authority": AUTHORITY,
        "promotable": False,
        "formal_d0_dataset": False,
        "scientific_claim_authorized": False,
        "target_quality_qualified_for_0817": False,
        "fresh_official_base": True,
        "resume_consumed": False,
        "step": step,
        "rank0_full_trainable_state_roundtrip_reload_verified": True,
        "rank0_optimizer_roundtrip_reload_verified": True,
        "all8_rng_sampler_scheduler_state_bytes_persisted_verified": True,
        "rank0_rng_state_roundtrip_reload_verified": True,
    }
    if any(metadata.get(key) != expected for key, expected in required.items()):
        fail("checkpoint metadata authority/roundtrip fields differ")
    parameter_sha = _require_sha(
        metadata.get("parameter_sha256"), label="checkpoint parameter SHA"
    )
    if (
        parameter_sha != campaign.parameter_digests[step]
        or metadata.get("roundtrip_parameter_sha256") != parameter_sha
        or metadata.get("trainable_inventory_sha256")
        != campaign.raw.get("trainable_inventory_sha256")
        or metadata.get("architecture") != campaign.raw.get("architecture")
        or metadata.get("method_source_file_sha256")
        != PINNED_TRAIN_RUNNER_SHA256
    ):
        fail("checkpoint metadata P-state/architecture closure differs")
    release = metadata.get("release_closure")
    provenance_release = campaign.raw.get("provenance", {}).get("release_closure")
    if (
        not isinstance(release, Mapping)
        or release != provenance_release
        or release.get("sha256")
        != _require_sha(
            expected_release_manifest_sha256,
            label="expected release manifest SHA",
        )
    ):
        fail("checkpoint release manifest authority differs")
    paths = {
        "adapter": _plain_file(
            directory / str(metadata.get("adapter_file")), label="full trainable state"
        ),
        "optimizer": _plain_file(
            directory / str(metadata.get("optimizer_file")), label="optimizer state"
        ),
        "runtime": _plain_file(
            directory / str(metadata.get("runtime_state_file")), label="runtime state"
        ),
    }
    hashes = {
        "adapter": metadata.get("adapter_sha256"),
        "optimizer": metadata.get("optimizer_sha256"),
        "runtime": metadata.get("runtime_state_sha256"),
    }
    record_hash_keys = {
        "adapter": "adapter_sha256",
        "optimizer": "optimizer_sha256",
        "runtime": "runtime_state_sha256",
    }
    for label, path in paths.items():
        expected = _require_sha(hashes[label], label=f"metadata {label} SHA")
        if expected != record[record_hash_keys[label]] or file_sha256(path) != expected:
            fail(f"checkpoint {label} bytes differ")
    if metadata_path != directory / "metadata.json":
        fail("checkpoint metadata path differs")
    return CheckpointPreflight(
        directory=directory,
        step=step,
        metadata=dict(metadata),
        metadata_sha256=record["metadata_sha256"],
        adapter_path=paths["adapter"],
        optimizer_path=paths["optimizer"],
        runtime_path=paths["runtime"],
        parameter_sha256=parameter_sha,
    )


def _state_tree_bits_equal(left: Any, right: Any, *, torch: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and bool(
                torch.equal(
                    left.detach().contiguous().reshape(-1).view(torch.uint8).cpu(),
                    right.detach().contiguous().reshape(-1).view(torch.uint8).cpu(),
                )
            )
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(
                _state_tree_bits_equal(left[key], right[key], torch=torch)
                for key in left
            )
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _state_tree_bits_equal(a, b, torch=torch)
                for a, b in zip(left, right)
            )
        )
    return type(left) is type(right) and left == right


def _runtime_state_receipt(runtime: Mapping[str, Any], *, step: int, torch: Any) -> Mapping[str, Any]:
    expected_cursor = None
    if step < 2:
        expected_cursor = {
            "optimizer_step_zero_based": step,
            "microbatch_index": 0,
            "next_logical_record": step * 8,
            "dp_arms": [0, 1],
        }
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != {
            "schema_version",
            "completed_optimizer_steps",
            "next_sampler_cursor",
            "scheduler",
            "stochasticity",
            "per_rank",
        }
        or runtime.get("schema_version") != RUNTIME_STATE_SCHEMA
        or runtime.get("completed_optimizer_steps") != step
        or runtime.get("next_sampler_cursor") != expected_cursor
        or runtime.get("scheduler")
        != {
            "object": None,
            "policy": "constant_lr_no_scheduler_object",
            "learning_rate": 1.0e-4,
            "completed_steps": step,
        }
        or runtime.get("stochasticity")
        != {
            "training_noise": "counter_based_per_row_torch_Generator_cpu",
            "dropout": 0.0,
            "rng_snapshots_retained_for_full_replay_abi": True,
        }
    ):
        fail("checkpoint sampler/scheduler/stochasticity ABI differs")
    rows = runtime.get("per_rank")
    if not isinstance(rows, list) or len(rows) != WORLD_SIZE:
        fail("checkpoint all8 RNG row count differs")
    rng_hashes = []
    for rank, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "world_rank",
                "dp_arm",
                "sp_rank",
                "python_random_state",
                "torch_cpu_rng_state",
                "torch_cuda_rng_state",
            }
            or row.get("world_rank") != rank
            or row.get("dp_arm") != rank // SP_SIZE
            or row.get("sp_rank") != rank % SP_SIZE
            or not isinstance(row.get("python_random_state"), tuple)
        ):
            fail("checkpoint all8 RNG rank topology differs")
        for key in ("torch_cpu_rng_state", "torch_cuda_rng_state"):
            value = row.get(key)
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.uint8
                or not value.is_contiguous()
                or int(value.numel()) <= 0
            ):
                fail(f"checkpoint rank {rank} {key} bytes differ")
        body = {
            "world_rank": rank,
            "dp_arm": row["dp_arm"],
            "sp_rank": row["sp_rank"],
            "python_random_state_repr": repr(row["python_random_state"]),
            "torch_cpu_rng_sha256": hashlib.sha256(
                bytes(row["torch_cpu_rng_state"].tolist())
            ).hexdigest(),
            "torch_cuda_rng_sha256": hashlib.sha256(
                bytes(row["torch_cuda_rng_state"].tolist())
            ).hexdigest(),
        }
        rng_hashes.append(object_sha256(body))
    return {
        "training_runtime_schema": RUNTIME_STATE_SCHEMA,
        "completed_optimizer_steps": step,
        "sampler_cursor_exact": True,
        "scheduler_declaration_exact": True,
        "all8_rng_snapshots_present": True,
        "all8_rng_snapshot_digests": rng_hashes,
        "training_rng_restored_into_inference_process": False,
    }


def _rng_snapshot(torch: Any) -> Mapping[str, Any]:
    cuda_rows = []
    if torch.cuda.is_available():
        cuda_rows = [value.detach().cpu().clone() for value in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": cuda_rows,
    }


def _torch_load_authenticated(
    path: Path, *, expected_sha256: str, torch: Any, label: str
) -> Any:
    """Hash and deserialize one stable open file description, not two path reads."""

    expected = _require_sha(expected_sha256, label=f"expected {label} SHA")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != expected:
            fail(f"{label} stable open-file SHA differs before deserialize")
        handle.seek(0)
        try:
            value = torch.load(handle, map_location="cpu", weights_only=True)
        except TypeError as error:
            raise CheckpointConsumerError(
                f"PyTorch weights_only {label} loading is mandatory"
            ) from error
        after = os.fstat(handle.fileno())
        if _identity(before) != _identity(after):
            fail(f"{label} changed during deserialize")
    return value


def _assert_fp32_finite_named(named: Sequence[tuple[str, Any]], *, torch: Any) -> None:
    if not named or len({name for name, _ in named}) != len(named):
        fail("fresh trainable parameter inventory is empty or duplicated")
    for name, parameter in named:
        if (
            not isinstance(parameter, torch.Tensor)
            or parameter.dtype != torch.float32
            or not parameter.requires_grad
            or not bool(torch.isfinite(parameter.detach()).all().item())
        ):
            fail(f"fresh FP32 trainable invariant differs: {name}")


@dataclass(frozen=True)
class LoadedCheckpoint:
    receipt: Mapping[str, Any]
    campaign: CampaignReceipt
    checkpoint: CheckpointPreflight
    runner: Any
    predictor_module: Any


def validate_training_reference_checkpoint_binding(
    metadata: Mapping[str, Any], *, step: int, parameter_sha256: str
) -> Optional[Mapping[str, Any]]:
    """Return None for r2; authenticate a future pre-save reference binding."""

    binding = metadata.get("training_attached_fixed_forward_reference")
    if binding is None:
        return None
    if not isinstance(binding, Mapping) or set(binding) != {
        "schema_version",
        "origin",
        "checkpoint_step",
        "checkpoint_parameter_sha256",
        "metadata_file",
        "metadata_file_sha256",
        "tensor_file",
        "tensor_file_sha256",
    }:
        fail("training-attached reference checkpoint binding field set differs")
    if (
        binding.get("schema_version") != TRAINING_REFERENCE_BINDING_SCHEMA
        or binding.get("origin") != "training_process_pre_checkpoint_export"
        or binding.get("checkpoint_step") != step
        or binding.get("checkpoint_parameter_sha256") != parameter_sha256
        or not isinstance(binding.get("metadata_file"), str)
        or Path(binding["metadata_file"]).name != binding["metadata_file"]
        or not isinstance(binding.get("tensor_file"), str)
        or Path(binding["tensor_file"]).name != binding["tensor_file"]
    ):
        fail("training-attached reference checkpoint binding semantics differ")
    _require_sha(binding.get("metadata_file_sha256"), label="training reference metadata SHA")
    _require_sha(binding.get("tensor_file_sha256"), label="training reference tensor SHA")
    return dict(binding)


@dataclass(frozen=True)
class FreshWorld8ModelBundle:
    """Fully authenticated model/runtime objects retained by offline inference."""

    model: Any
    renderer: Any
    transformer: Any
    conditioner: Any
    offline_hooks: Any
    lora_specs: tuple[Any, ...]
    sigma_contract_module: Any
    distributed: Any
    parallel: Any
    device: Any
    checkpoint: LoadedCheckpoint
    consumer_receipt: Mapping[str, Any]


def load_fresh_checkpoint_strict(
    *,
    model: Any,
    conditioner: Any,
    checkpoint: CheckpointPreflight,
    campaign: CampaignReceipt,
    runner: Any,
    predictor_module: Any,
    release_closure: Mapping[str, Any],
    torch_module: Any,
    require_formal_profile: bool = True,
) -> LoadedCheckpoint:
    """Load once per process without restoring training RNG or scheduler state."""

    global _PROCESS_LOAD_CONSUMED
    if _PROCESS_LOAD_CONSUMED:
        fail("fresh checkpoint consumer is single-use per operating-system process")
    _PROCESS_LOAD_CONSUMED = True
    torch = torch_module
    record = campaign.checkpoint_records.get(checkpoint.step)
    if (
        campaign.sha256 != PINNED_R2_CAMPAIGN_RECEIPT_SHA256
        or campaign.parameter_digests != PINNED_R2_P_STATE_SHA256
        or not isinstance(record, Mapping)
        or record.get("metadata_sha256") != checkpoint.metadata_sha256
        or record.get("adapter_sha256")
        != checkpoint.metadata.get("adapter_sha256")
        or record.get("optimizer_sha256")
        != checkpoint.metadata.get("optimizer_sha256")
        or record.get("runtime_state_sha256")
        != checkpoint.metadata.get("runtime_state_sha256")
        or checkpoint.parameter_sha256
        != campaign.parameter_digests.get(checkpoint.step)
    ):
        fail("selected checkpoint/campaign trust-root cross-binding differs")
    if require_formal_profile:
        conditioner.config.require_formal_0817()
    named = runner.exact_trainable_named_parameters(model, conditioner)
    _assert_fp32_finite_named(named, torch=torch)
    observed_inventory_sha = runner.object_sha256(list(runner.trainable_inventory(named)))
    if observed_inventory_sha != checkpoint.metadata.get("trainable_inventory_sha256"):
        fail("fresh model exact trainable inventory SHA differs")
    abi = predictor_module.exact_state_dict_abi(conditioner)
    expected_abi = (
        PINNED_CONDITIONER_ABI_SHA256
        if require_formal_profile
        else checkpoint.metadata.get("architecture", {})
        .get("action_plan_state_dict_abi", {})
        .get("abi_sha256")
    )
    if abi.get("abi_sha256") != expected_abi:
        fail("fresh conditioner semantic state ABI differs before load")
    rng_before = _rng_snapshot(torch)
    state = _torch_load_authenticated(
        checkpoint.adapter_path,
        expected_sha256=checkpoint.metadata["adapter_sha256"],
        torch=torch,
        label="full trainable state",
    )
    optimizer_state = _torch_load_authenticated(
        checkpoint.optimizer_path,
        expected_sha256=checkpoint.metadata["optimizer_sha256"],
        torch=torch,
        label="optimizer state",
    )
    runtime_state = _torch_load_authenticated(
        checkpoint.runtime_path,
        expected_sha256=checkpoint.metadata["runtime_state_sha256"],
        torch=torch,
        label="runtime state",
    )
    if (
        not isinstance(state, Mapping)
        or set(state) != {
            "schema_version",
            "trainable_parameters",
            "action_plan_conditioner",
        }
        or state.get("schema_version") != FULL_STATE_SCHEMA
        or not isinstance(state.get("trainable_parameters"), Mapping)
        or not isinstance(state.get("action_plan_conditioner"), Mapping)
    ):
        fail("full trainable checkpoint envelope differs")
    checkpoint_named = state["trainable_parameters"]
    if set(checkpoint_named) != {name for name, _ in named}:
        fail("full trainable checkpoint parameter-name set differs")
    for name, reference in named:
        value = checkpoint_named[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or tuple(value.shape) != tuple(reference.shape)
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
        ):
            fail(f"full trainable checkpoint FP32 tensor differs: {name}")
    conditioner_names = {id(value): name for name, value in conditioner.named_parameters()}
    conditioner_state = state["action_plan_conditioner"]
    duplicate_count = 0
    for full_name, parameter in named:
        local_name = conditioner_names.get(id(parameter))
        if local_name is not None:
            duplicate_count += 1
            if local_name not in conditioner_state or not _state_tree_bits_equal(
                checkpoint_named[full_name], conditioner_state[local_name], torch=torch
            ):
                fail("duplicated conditioner/full-trainable checkpoint bytes differ")
    if duplicate_count != len(conditioner_names):
        fail("conditioner ownership is not complete in the full trainable state")
    runner.load_conditioner_state_strict(conditioner, conditioner_state)
    runner.load_trainable_state_strict(named, checkpoint_named)
    loaded_digest = runner.tensor_digest(named)
    if loaded_digest != checkpoint.parameter_sha256:
        fail("fresh strict load changed the authenticated P-state bytes")
    if predictor_module.exact_state_dict_abi(conditioner).get("abi_sha256") != expected_abi:
        fail("conditioner semantic state ABI changed during strict load")
    runner.validate_adamw_state_abi(optimizer_state, named, step=checkpoint.step)
    runtime_receipt = _runtime_state_receipt(
        runtime_state, step=checkpoint.step, torch=torch
    )
    rng_after = _rng_snapshot(torch)
    if not _state_tree_bits_equal(rng_before, rng_after, torch=torch):
        fail("checkpoint consumer mutated process RNG or restored training RNG")
    release_in_metadata = checkpoint.metadata.get("release_closure")
    if (
        release_in_metadata.get("sha256") != release_closure.get("sha256")
        or release_in_metadata.get("member_set_sha256")
        != release_closure.get("member_set_sha256")
    ):
        fail("executed release differs from checkpoint release closure")
    training_reference_binding = validate_training_reference_checkpoint_binding(
        checkpoint.metadata,
        step=checkpoint.step,
        parameter_sha256=checkpoint.parameter_sha256,
    )
    receipt = {
        "schema_version": CONSUMER_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "complete": True,
        "promotable": False,
        "formal_training_started": False,
        "counts_as_d0": False,
        "scientific_claim_authorized": False,
        "action_quality_claim_authorized": False,
        "checkpoint_step": checkpoint.step,
        "checkpoint_parameter_sha256": checkpoint.parameter_sha256,
        "loaded_parameter_sha256": loaded_digest,
        "campaign_receipt_sha256": campaign.sha256,
        "checkpoint_metadata_sha256": checkpoint.metadata_sha256,
        "release_manifest_sha256": release_closure.get("sha256"),
        "runner_source_sha256": PINNED_TRAIN_RUNNER_SHA256,
        "predictor_source_sha256": PINNED_PREDICTOR_SOURCE_SHA256,
        "conditioner_state_abi_sha256": expected_abi,
        "all_trainables_fp32_at_load": True,
        "optimizer_state_validated_but_not_loaded": True,
        "training_runtime_state_validated_but_not_restored": True,
        "runtime": runtime_receipt,
        "fresh_process_session_id": _PROCESS_SESSION_ID,
        "single_checkpoint_load_per_process": True,
        "training_attached_reference_present": training_reference_binding is not None,
        "training_attached_reference_absent": training_reference_binding is None,
        "training_attached_reference_binding": training_reference_binding,
        "training_attached_conditioner_cell_reference_present": (
            training_reference_binding is not None
        ),
        "training_attached_conditioner_cell_reference_absent": (
            training_reference_binding is None
        ),
        "training_attached_full_renderer_reference_present": False,
        "training_attached_full_renderer_reference_absent": True,
        "training_attached_full_renderer_reference_binding": None,
        "training_to_fresh_forward_parity_verified": False,
        "conditioner_cell_training_to_fresh_forward_parity_verified": False,
        "full_bernini_renderer_training_to_fresh_forward_parity_verified": False,
        "fresh_a_b_parity_verified": False,
        "promotion_authorized": False,
        "offline_product_inference_completed": False,
        "full40_denoise_executed": False,
        "mp4_emitted": False,
    }
    return LoadedCheckpoint(
        receipt=receipt,
        campaign=campaign,
        checkpoint=checkpoint,
        runner=runner,
        predictor_module=predictor_module,
    )


def build_and_load_fresh_world8_model(
    *,
    bernini_root: str | Path,
    veomni_root: str | Path,
    base_checkpoint: str | Path,
    checkpoint_content_manifest: str | Path,
    selected_checkpoint: CheckpointPreflight,
    campaign: CampaignReceipt,
    release_preflight: ReleasePreflight,
    runner: Any,
    predictor_module: Any,
    release_closure: Mapping[str, Any],
    expected_consumer_source_sha256: str,
    expected_product_source_sha256: str,
) -> FreshWorld8ModelBundle:
    """Construct and load the exact training architecture in a fresh WORLD8.

    The old frozen release authenticates every training-owned component.  The
    consumer and product bridge are separately hash-pinned because they did not
    exist in the r2 training release.  No gradient checkpointing/training hook
    is installed: inference uses the product bridge's single-forward hooks.
    """

    if (
        release_preflight.manifest_sha256
        != PINNED_R2_RELEASE_MANIFEST_SHA256
        or release_closure.get("sha256")
        != PINNED_R2_RELEASE_MANIFEST_SHA256
        or release_closure.get("member_set_sha256")
        != PINNED_R2_RELEASE_MEMBER_SET_SHA256
        or campaign.sha256 != PINNED_R2_CAMPAIGN_RECEIPT_SHA256
        or selected_checkpoint.parameter_sha256
        != PINNED_R2_P_STATE_SHA256.get(selected_checkpoint.step)
    ):
        fail("fresh WORLD8 r2 release/campaign/checkpoint trust roots differ")
    consumer_sha = _require_sha(
        expected_consumer_source_sha256, label="consumer source SHA"
    )
    product_sha = _require_sha(
        expected_product_source_sha256, label="product bridge source SHA"
    )
    executed_consumer = Path(__file__)
    if (
        not executed_consumer.is_absolute()
        or executed_consumer.is_symlink()
        or executed_consumer.resolve(strict=True) != executed_consumer
        or stat.S_IMODE(executed_consumer.lstat().st_mode) != 0o444
        or file_sha256(executed_consumer) != consumer_sha
    ):
        fail("executed checkpoint consumer source SHA differs")
    product_module_name = "infer_action_edit_product_abi_0817_v1"
    product_path = METHOD_ROOT / f"{product_module_name}.py"
    if (
        not product_path.is_file()
        or product_path.is_symlink()
        or stat.S_IMODE(product_path.lstat().st_mode) != 0o444
        or file_sha256(product_path) != product_sha
    ):
        fail("executed product bridge source SHA differs")
    late_release_modules = (
        "packed_preservation_lora_v2",
        "packed_preservation_release_v2",
        "source_self_runtime",
        "train_lora",
        "clean_source_visual_context_stage_b_contract_v1",
        "inference_sigma_strata",
    )
    if product_module_name in sys.modules or any(
        name in sys.modules for name in late_release_modules
    ):
        fail("fresh WORLD8 consumer requires product/dependencies absent before import")
    if any(
        name == "bernini"
        or name.startswith("bernini.")
        or name == "veomni"
        or name.startswith("veomni.")
        or name == "peft"
        or name.startswith("peft.")
        for name in sys.modules
    ):
        fail("fresh WORLD8 consumer found a pre-imported model/runtime package")
    product_source = _stable_bytes(
        product_path,
        maximum=2 * 1024 * 1024,
        label="product bridge source",
    )
    if hashlib.sha256(product_source).hexdigest() != product_sha:
        fail("product bridge stable source bytes differ")
    product_module = _load_authenticated_source_module(
        module_name=product_module_name,
        source_path=product_path,
        source_bytes=product_source,
        expected_sha256=product_sha,
    )

    # All eight training dependencies are imported only after the release was
    # authenticated.  Execute the preflight snapshots directly; never consult
    # import caches or a same-name extension/sourceless shadow.
    sigma_strata = _load_authenticated_release_module(
        release_preflight, "inference_sigma_strata"
    )
    schedule_contract = _load_authenticated_release_module(
        release_preflight, "clean_source_visual_context_stage_b_contract_v1"
    )
    core = _load_authenticated_release_module(
        release_preflight, "packed_preservation_lora_v2"
    )
    release_contract = _load_authenticated_release_module(
        release_preflight, "packed_preservation_release_v2"
    )
    runtime = _load_authenticated_release_module(
        release_preflight, "source_self_runtime"
    )
    legacy = _load_authenticated_release_module(
        release_preflight, "train_lora"
    )
    imported_release = runner.validate_imported_release_modules(
        release_closure,
        {
            "action_plan_predictor_v1.py": predictor_module,
            "clean_source_visual_context_stage_b_contract_v1.py": schedule_contract,
            "inference_sigma_strata.py": sigma_strata,
            "packed_preservation_lora_v2.py": core,
            "packed_preservation_release_v2.py": release_contract,
            "source_self_runtime.py": runtime,
            "train_action_edit_large_lora_0817_v1.py": runner,
            "train_lora.py": legacy,
        },
        method_root=release_preflight.member_root,
    )
    if getattr(schedule_contract, "exact40", None) is not sigma_strata:
        fail("authenticated schedule transitive module ownership differs")

    try:
        source_root, omni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                bernini_root,
                veomni_root,
                expected_bernini_commit=runner.BERNINI_COMMIT,
                expected_veomni_commit=runner.VEOMNI_COMMIT,
            )
        )
        checkpoint_root, transformer_config = legacy.validate_checkpoint(
            base_checkpoint
        )
    except legacy.TrainingContractError as error:
        raise CheckpointConsumerError(str(error)) from error
    if (
        transformer_config.get("num_layers") != runner.TRANSFORMER_BLOCKS
        or transformer_config.get("attention_head_dim") != 128
        or transformer_config.get("num_attention_heads") != 12
    ):
        fail("fresh Bernini-R base transformer geometry differs")
    content_manifest = _plain_file(
        checkpoint_content_manifest, label="base checkpoint content manifest"
    )
    if file_sha256(content_manifest) != runner.CHECKPOINT_CONTENT_MANIFEST_SHA256:
        fail("base checkpoint content manifest SHA differs")
    checkpoint_content = release_contract.validate_checkpoint_content(
        checkpoint_root,
        content_manifest,
        expected_manifest_sha256=runner.CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    legacy.activate_source_trees(source_root, omni_root)

    import torch
    from peft import LoraConfig, get_peft_model
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = runtime.distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.local_world_size != WORLD_SIZE
        or distributed.topology.dp_size != DP_SIZE
        or distributed.topology.sp_size != SP_SIZE
    ):
        fail("fresh checkpoint consumer requires one-node WORLD8 DP2xSP4")
    device = runtime.initialise_distributed(distributed)
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=SP_SIZE)
    )

    config = BerniniRendererConfig.from_pretrained(
        str(source_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint_root),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint_root)
    with runner.serialized_model_load():
        renderer = BerniniRendererModel(config)
        renderer.requires_grad_(False)
        specs = tuple(core.select_projection_specs(renderer, runner.LORA_SCOPE))
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
        core.install_typed_patch_embedding(transformer)
        conditioner = predictor_module.ActionPlanConditionerV1(
            predictor_module.ActionPlanPredictorConfig()
        )
        conditioner.config.require_formal_0817()
        offline_hooks = product_module.install_offline_action_plan_hooks(
            transformer=transformer,
            conditioner=conditioner,
            torch_module=torch,
        )
        model.to(device)
    model.eval()
    conditioner.config.require_formal_0817()
    lora_installation = core.validate_lora_installation(model, specs)
    named = runner.exact_trainable_named_parameters(model, conditioner)
    if any(
        parameter.device != device or parameter.dtype != torch.float32
        for _, parameter in named
    ):
        fail("fresh WORLD8 model trainables are not accelerator-local FP32")
    with runner.serialized_model_load():
        loaded = load_fresh_checkpoint_strict(
            model=model,
            conditioner=conditioner,
            checkpoint=selected_checkpoint,
            campaign=campaign,
            runner=runner,
            predictor_module=predictor_module,
            release_closure=release_closure,
            torch_module=torch,
            require_formal_profile=True,
        )
    local_receipt = {
        **dict(loaded.receipt),
        "consumer_source_sha256": consumer_sha,
        "product_bridge_source_sha256": product_sha,
        "official_bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "base_checkpoint_tree_sha256": runner.CHECKPOINT_TREE_SHA256,
        "base_checkpoint_content": checkpoint_content,
        "imported_training_release": imported_release,
        "lora_installation": lora_installation,
        "model_mode": "eval_with_fp32_persisted_trainables",
        "training_gradient_checkpoint_hooks_installed": False,
        "offline_single_forward_exact30_hooks_installed": True,
    }
    rng_before_fixed_forward = _rng_snapshot(torch)
    fixed_forward = product_module.fixed_forward_tensors(
        conditioner=conditioner,
        predictor_module=predictor_module,
        torch_module=torch,
    )
    fixed_forward_fingerprint = product_module.fixed_forward_fingerprint(
        fixed_forward, torch_module=torch
    )
    del fixed_forward
    rng_after_fixed_forward = _rng_snapshot(torch)
    if not _state_tree_bits_equal(
        rng_before_fixed_forward, rng_after_fixed_forward, torch=torch
    ):
        fail("fresh loaded fixed forward mutated process RNG")
    if runner.tensor_digest(named) != loaded.receipt["loaded_parameter_sha256"]:
        fail("fresh loaded fixed forward mutated persisted trainable bytes")
    local_receipt.update(
        {
            "fresh_loaded_fixed_forward_executed": True,
            "fresh_loaded_fixed_forward_fingerprint": fixed_forward_fingerprint,
            "fixed_forward_process_rng_unchanged": True,
            "fixed_forward_trainable_bytes_unchanged": True,
        }
    )
    consensus = world8_consensus_receipt(
        local_receipt,
        distributed_module=torch.distributed,
        group=parallel.world_group,
        rank_local_fields=("fresh_process_session_id",),
    )
    consumer_receipt = {
        **local_receipt,
        "world8_consensus": consensus,
        "world8_consumer_complete": True,
        "fresh_world8_process_forward_exact_consensus_verified": True,
        "fresh_world8_process_forward_scope": (
            "conditioner_predictor_plus_exact30_cell_only_not_bernini_renderer"
        ),
        "full_bernini_renderer_forward_executed": False,
        "checkpoint_bytes_conditioner_exact30_fresh_consumer_go": True,
        "offline_product_inference_completed": False,
        "full40_denoise_executed": False,
        "mp4_emitted": False,
        "promotion_authorized": False,
    }
    return FreshWorld8ModelBundle(
        model=model,
        renderer=model.get_base_model(),
        transformer=transformer,
        conditioner=conditioner,
        offline_hooks=offline_hooks,
        lora_specs=specs,
        sigma_contract_module=sigma_strata,
        distributed=distributed,
        parallel=parallel,
        device=device,
        checkpoint=loaded,
        consumer_receipt=consumer_receipt,
    )


def consume_frozen_r2_world8_checkpoint(
    *,
    release_manifest_path: str | Path,
    campaign_receipt_path: str | Path,
    checkpoint_dir: str | Path,
    checkpoint_step: int,
    bernini_root: str | Path,
    veomni_root: str | Path,
    base_checkpoint: str | Path,
    checkpoint_content_manifest: str | Path,
    expected_consumer_source_sha256: str,
    expected_product_source_sha256: str,
) -> FreshWorld8ModelBundle:
    """One ordered public entry point for a new torchrun WORLD8 process."""

    release = authenticate_release_before_import(
        release_manifest_path,
        expected_sha256=PINNED_R2_RELEASE_MANIFEST_SHA256,
        method_root=METHOD_ROOT,
    )
    runner, predictor, release_closure = import_authenticated_training_modules(
        release
    )
    campaign = validate_campaign_receipt(
        campaign_receipt_path,
        expected_sha256=PINNED_R2_CAMPAIGN_RECEIPT_SHA256,
    )
    checkpoint = validate_checkpoint_preflight(
        checkpoint_dir,
        step=checkpoint_step,
        campaign=campaign,
        expected_release_manifest_sha256=PINNED_R2_RELEASE_MANIFEST_SHA256,
    )
    return build_and_load_fresh_world8_model(
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        base_checkpoint=base_checkpoint,
        checkpoint_content_manifest=checkpoint_content_manifest,
        selected_checkpoint=checkpoint,
        campaign=campaign,
        release_preflight=release,
        runner=runner,
        predictor_module=predictor,
        release_closure=release_closure,
        expected_consumer_source_sha256=expected_consumer_source_sha256,
        expected_product_source_sha256=expected_product_source_sha256,
    )


def world8_consensus_receipt(
    receipt: Mapping[str, Any],
    *,
    distributed_module: Any,
    group: Any,
    rank_local_fields: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Require eight ranks to agree on the complete consumer receipt digest."""

    dist = distributed_module
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size(group=group) != WORLD_SIZE
    ):
        fail("fresh checkpoint consumer requires initialized WORLD8 consensus")
    if tuple(rank_local_fields) not in ((), ("fresh_process_session_id",)):
        fail("WORLD8 consensus rank-local field policy differs")
    common = dict(receipt)
    rank_local = {}
    for key in rank_local_fields:
        value = common.pop(key, None)
        if key == "fresh_process_session_id":
            rank_local[key] = _require_sha(value, label="fresh process session ID")
    rank = int(dist.get_rank(group=group))
    local = {
        "world_rank": rank,
        "receipt_sha256": object_sha256(common),
        "rank_local": rank_local,
    }
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, local, group=group)
    if (
        any(not isinstance(row, Mapping) for row in rows)
        or [row.get("world_rank") for row in rows] != list(range(WORLD_SIZE))
        or len({row.get("receipt_sha256") for row in rows}) != 1
    ):
        fail("fresh checkpoint WORLD8 receipt consensus differs")
    if rank_local_fields:
        sessions = [
            row.get("rank_local", {}).get("fresh_process_session_id") for row in rows
        ]
        if (
            any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in sessions
            )
            or len(set(sessions)) != WORLD_SIZE
            or sessions[rank] != rank_local["fresh_process_session_id"]
        ):
            fail("fresh checkpoint WORLD8 process-session identities differ")
    else:
        sessions = []
    return {
        "world_size": WORLD_SIZE,
        "rank_order": list(range(WORLD_SIZE)),
        "consumer_receipt_sha256": rows[0]["receipt_sha256"],
        "all8_exact_consensus": True,
        "rank_local_fresh_process_sessions": sessions,
        "eight_distinct_fresh_process_sessions": bool(rank_local_fields),
    }


def compare_fresh_world8_consumer_receipts(
    receipt_a: Mapping[str, Any], receipt_b: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Compare two independent WORLD8 launches without calling either training-attached."""

    common_fields = (
        "authority",
        "checkpoint_step",
        "checkpoint_parameter_sha256",
        "loaded_parameter_sha256",
        "campaign_receipt_sha256",
        "checkpoint_metadata_sha256",
        "release_manifest_sha256",
        "runner_source_sha256",
        "predictor_source_sha256",
        "conditioner_state_abi_sha256",
        "consumer_source_sha256",
        "product_bridge_source_sha256",
        "fresh_loaded_fixed_forward_fingerprint",
    )
    session_sets = []
    for label, receipt in (("A", receipt_a), ("B", receipt_b)):
        consensus = receipt.get("world8_consensus")
        sessions = (
            consensus.get("rank_local_fresh_process_sessions")
            if isinstance(consensus, Mapping)
            else None
        )
        if (
            receipt.get("schema_version") != CONSUMER_SCHEMA
            or receipt.get("authority") != AUTHORITY
            or receipt.get("promotable") is not False
            or receipt.get("promotion_authorized") is not False
            or receipt.get("world8_consumer_complete") is not True
            or receipt.get("fresh_loaded_fixed_forward_executed") is not True
            or receipt.get("fresh_world8_process_forward_exact_consensus_verified")
            is not True
            or receipt.get("fresh_world8_process_forward_scope")
            != "conditioner_predictor_plus_exact30_cell_only_not_bernini_renderer"
            or receipt.get("full_bernini_renderer_forward_executed") is not False
            or receipt.get("training_to_fresh_forward_parity_verified") is not False
            or not isinstance(receipt.get("fresh_loaded_fixed_forward_fingerprint"), Mapping)
            or not isinstance(consensus, Mapping)
            or consensus.get("world_size") != WORLD_SIZE
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
            fail(f"fresh WORLD8 consumer receipt {label} differs")
        session_sets.append(set(sessions))
    if session_sets[0] & session_sets[1]:
        fail("fresh WORLD8 A/B launches reused an operating-system process session")
    differences = [
        key for key in common_fields if receipt_a.get(key) != receipt_b.get(key)
    ]
    if differences:
        fail(f"fresh WORLD8 A/B fixed checkpoint/forward fields differ: {differences}")
    return {
        "schema_version": FRESH_WORLD8_PARITY_SCHEMA,
        "method": METHOD,
        "authority": AUTHORITY,
        "promotable": False,
        "origin": "posthoc_two_independent_world8_fresh_consumer_launches",
        "checkpoint_parameter_sha256": receipt_a["checkpoint_parameter_sha256"],
        "exact_parity": True,
        "exact_or_bounded_parity_pass": True,
        "world8_launches": 2,
        "distinct_fresh_process_sessions": 2 * WORLD_SIZE,
        "os_process_independence_proven": True,
        "disjoint_object_and_parameter_storage_verified": True,
        "training_attached_reference": False,
        "training_to_fresh_forward_parity_claimed": False,
        "full_bernini_renderer_forward_parity_claimed": False,
        "fresh_loaded_fixed_forward_fingerprint": receipt_a[
            "fresh_loaded_fixed_forward_fingerprint"
        ],
        "promotion_authorized": False,
    }


def bind_parity_result(
    load_receipt: Mapping[str, Any],
    *,
    fresh_a_b: Optional[Mapping[str, Any]],
    training_attached: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Keep post-hoc and training-attached parity in non-interchangeable fields."""

    result = dict(load_receipt)
    if fresh_a_b is not None:
        if (
            fresh_a_b.get("schema_version") != FRESH_WORLD8_PARITY_SCHEMA
            or fresh_a_b.get("authority") != AUTHORITY
            or fresh_a_b.get("promotable") is not False
            or fresh_a_b.get("checkpoint_parameter_sha256")
            != load_receipt.get("checkpoint_parameter_sha256")
            or fresh_a_b.get("exact_or_bounded_parity_pass") is not True
            or fresh_a_b.get("origin")
            != "posthoc_two_independent_world8_fresh_consumer_launches"
            or fresh_a_b.get("os_process_independence_proven") is not True
            or fresh_a_b.get("disjoint_object_and_parameter_storage_verified")
            is not True
        ):
            fail("fresh-A/B parity receipt differs")
        result["fresh_a_b_parity_verified"] = True
        result["fresh_a_b_parity"] = dict(fresh_a_b)
    if training_attached is not None:
        binding = load_receipt.get("training_attached_reference_binding")
        if (
            not isinstance(binding, Mapping)
            or training_attached.get("schema_version") != TRAINING_REFERENCE_SCHEMA
            or training_attached.get("authority") != AUTHORITY
            or training_attached.get("promotable") is not False
            or training_attached.get("checkpoint_parameter_sha256")
            != load_receipt.get("checkpoint_parameter_sha256")
            or training_attached.get("origin")
            != "training_process_pre_checkpoint_export"
            or training_attached.get("fresh_consumer_parity_pass") is not True
            or training_attached.get(
                "training_to_fresh_forward_parity_verified"
            )
            is not True
            or training_attached.get(
                "conditioner_cell_training_to_fresh_forward_parity_verified"
            )
            is not True
            or training_attached.get(
                "full_bernini_renderer_training_to_fresh_forward_parity_verified"
            )
            is not False
            or training_attached.get("metadata_file_sha256")
            != binding.get("metadata_file_sha256")
            or training_attached.get("tensor_file_sha256")
            != binding.get("tensor_file_sha256")
        ):
            fail("training-attached parity receipt differs")
        result["training_attached_reference_present"] = True
        result["training_attached_reference_absent"] = False
        result["training_to_fresh_forward_parity_verified"] = True
        result["conditioner_cell_training_to_fresh_forward_parity_verified"] = True
        result["full_bernini_renderer_training_to_fresh_forward_parity_verified"] = False
        result["training_attached_parity"] = dict(training_attached)
    # PRE_D0 engineering artifacts remain nonpromotable even when both parity
    # checks pass.  Formal promotion has separate data/model/evaluation gates.
    result["promotion_authorized"] = False
    result["promotable"] = False
    return result


__all__ = [
    "AUTHORITY",
    "CONSUMER_SCHEMA",
    "CampaignReceipt",
    "CheckpointConsumerError",
    "CheckpointPreflight",
    "FRESH_PARITY_SCHEMA",
    "FRESH_WORLD8_PARITY_SCHEMA",
    "FreshWorld8ModelBundle",
    "LoadedCheckpoint",
    "METHOD",
    "PINNED_CONDITIONER_ABI_SHA256",
    "PINNED_PREDICTOR_SOURCE_SHA256",
    "PINNED_R2_CAMPAIGN_RECEIPT_SHA256",
    "PINNED_R2_P_STATE_SHA256",
    "PINNED_R2_RELEASE_MANIFEST_SHA256",
    "PINNED_R2_RELEASE_MEMBER_SET_SHA256",
    "PINNED_TRAIN_RUNNER_SHA256",
    "ReleasePreflight",
    "TRAINING_REFERENCE_SCHEMA",
    "TRAINING_REFERENCE_BINDING_SCHEMA",
    "authenticate_release_before_import",
    "bind_parity_result",
    "build_and_load_fresh_world8_model",
    "canonical_json_bytes",
    "compare_fresh_world8_consumer_receipts",
    "consume_frozen_r2_world8_checkpoint",
    "file_sha256",
    "import_authenticated_training_modules",
    "load_fresh_checkpoint_strict",
    "object_sha256",
    "validate_campaign_receipt",
    "validate_checkpoint_preflight",
    "validate_training_reference_checkpoint_binding",
    "world8_consensus_receipt",
]
