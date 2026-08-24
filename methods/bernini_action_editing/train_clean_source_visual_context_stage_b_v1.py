#!/usr/bin/env python3
"""WORLD8 DP2xSP4 exact80 clean-source visual-context Stage-B pilot.

The frozen Bernini RV2V base receives a clean real source video latent and a
forward-noised copy of that same real source as the target.  Only target rows
enter the standard flow-matching MSE.  A separate visual-context adapter lets
those target queries attend to source-only memory.  The main arm builds memory
from the detached clean source; the registered variant uses the exact same
forward-noised source state (same sigma and epsilon as the target).  Native
self-attention, text cross-attention, base weights, VAE data and T5 weights are
never optimized.

No synthetic target posterior, reward, scalar evaluator, action anchor or
pixel target is read.  A SHA-pinned decoded Stage-A admission is validated
before the optimizer is constructed.  The continuous trajectory writes
create-only checkpoints at 0/20/40/60/80.  A separate paired feasibility scope
runs one real four-microbatch backward window and synchronized gradient audit
without constructing an optimizer, changing parameters, or writing checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile
import time
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_adapter_v1 as visual  # noqa: E402
import clean_source_visual_context_pair_controller_v1 as pair_contract  # noqa: E402
import clean_source_visual_context_stage_b_contract_v1 as contract  # noqa: E402
import clean_source_visual_context_training_v1 as data_contract  # noqa: E402


METHOD = "bernini-clean-source-visual-context-stage-b-v1"
MODE = "clean-source-visual-context-stage-b-v1"
RECEIPT_SCHEMA = "bernini-clean-source-visual-context-stage-b-training-v1"
PREFLIGHT_RECEIPT_SCHEMA = (
    "bernini-clean-source-visual-context-stage-b-structural-preflight-v1"
)
BACKWARD_PREFLIGHT_RECEIPT_SCHEMA = (
    "bernini-clean-source-visual-context-stage-b-backward-feasibility-v1"
)
HISTORY_SCHEMA = "bernini-clean-source-visual-context-stage-b-history-v1"
ADAPTER_FILE_SCHEMA = "bernini-clean-source-visual-context-adapter-file-v1"
TOPOLOGY = "world8-dp2-sp4"
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FRAME_COUNT = 81
LATENT_PHASES = 21
PATCH_VALUES = 64
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 20260814
EXPECTED_VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
EXPECTED_CHECKPOINT_CONTENT_FILE_COUNT = 23
METHOD_RELEASE_SCHEMA = "bernini-clean-source-visual-context-stage-b-release-v1"
METHOD_RELEASE_GENERATION = "r1"
METHOD_RELEASE_ARCHIVE_FORMAT = "ustar-owner0-mtime0-mode0444-v1"
METHOD_RELEASE_MEMBER_ROOT = "methods/bernini_action_editing"
METHOD_RELEASE_FILES = (
    "clean_source_visual_context_adapter_v1.py",
    "clean_source_visual_context_training_v1.py",
    "clean_source_visual_context_stage_b_contract_v1.py",
    "train_clean_source_visual_context_stage_b_v1.py",
    "clean_source_visual_context_pair_controller_v1.py",
    "source_self_runtime.py",
    "train_lora.py",
    "inference_sigma_strata.py",
    "scripts/auh_preservation_rank_cache_exec_v1.sh",
    "scripts/auh_train_clean_source_visual_context_stage_b_holder_v1.sh",
    "scripts/auh_train_clean_source_visual_context_main_holder_v1.sh",
    "scripts/auh_train_clean_source_visual_context_noised_holder_v1.sh",
    "scripts/auh_preflight_clean_source_visual_context_main_holder_v1.sh",
    "scripts/auh_preflight_clean_source_visual_context_noised_holder_v1.sh",
    "scripts/auh_materialize_clean_source_visual_context_source_only_v3_holder_v1.sh",
)
EXPECTED_TRANSFORMER_WAN_SHA256 = (
    "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223"
)
EXPECTED_VEOMNI_SP_UTILS_SHA256 = (
    "17abb6e969097bc6bae35be6498d1b7edd9d4f8d78f836f35fff137606c84361"
)
EXPECTED_VEOMNI_SP_DATA_SHA256 = (
    "b635d5272a6dadfa9f6b2501345179db8b5f76a15e575c7a1bf46ffe01550a60"
)
EXPECTED_VEOMNI_SP_ULYSSES_SHA256 = (
    "98230d5219d5f327fafc436ec5d2df99c97a341676c2e964f7b8ec5470fd48e2"
)
EXPECTED_BERNINI_PARALLEL_OPS_SHA256 = (
    "c264f28b7b011ce01204ec5b0f11acd08adb6568a9855108b866fb9ce1a2ce30"
)
EXACT_NOOP_INSTRUCTION = (
    "Keep the source video exactly unchanged, including every subject, "
    "appearance, action, camera motion, background, timing, and composition."
)
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CleanSourceVisualStageBTrainingError(RuntimeError):
    """Raised before an ambiguous update or artifact can be published."""


def fail(message: str) -> NoReturn:
    raise CleanSourceVisualStageBTrainingError(message)


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
        raise CleanSourceVisualStageBTrainingError(
            "value is not canonical finite ASCII JSON"
        ) from error


def stage_b_trainable_parameters_digest(
    named: Sequence[tuple[str, Any]],
) -> str:
    """Hash Stage-B parameters, including scalar residual gates, in legacy order.

    Metadata and parameter byte order intentionally match the shared runtime
    for every non-scalar tensor.  Flattening only for dtype reinterpretation
    makes a zero-dimensional ``residual_gain`` safe without changing its
    recorded shape or the named-parameter order.
    """

    import torch

    digest = hashlib.sha256()
    for name, parameter in named:
        tensor = parameter.detach().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(
            tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes(order="C")
        )
    return digest.hexdigest()


def stage_b_parameter_consensus(
    named: Sequence[tuple[str, Any]],
    world_group: Any,
    label: str,
    *,
    expected_count: int = WORLD_SIZE,
) -> str:
    """Require the scalar-safe Stage-B parameter digest on every WORLD8 rank."""

    import torch.distributed as dist

    value = stage_b_trainable_parameters_digest(named)
    gathered: list[Any] = [None] * expected_count
    dist.all_gather_object(gathered, value, group=world_group)
    if any(item != value for item in gathered):
        fail(f"{label} differs across replicated ranks")
    return value


def stage_b_gradients_digest(named: Sequence[tuple[str, Any]]) -> str:
    """Hash every synchronized gradient without touching parameter storage."""

    import torch

    digest = hashlib.sha256()
    for name, parameter in named:
        if parameter.grad is None:
            fail(f"visual-context gradient is absent: {name}")
        gradient = parameter.grad.detach().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(gradient.shape), "dtype": str(gradient.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(
            gradient.reshape(-1).view(torch.uint8).cpu().numpy().tobytes(order="C")
        )
    return digest.hexdigest()


def stage_b_synchronize_initial_parameters(
    named: Sequence[tuple[str, Any]],
    world_group: Any,
    *,
    expected_count: int = WORLD_SIZE,
) -> str:
    """Broadcast then attest the scalar-safe Stage-B initialization."""

    import torch.distributed as dist

    if not named:
        fail("adapter trainable parameter set is empty")
    for _, parameter in named:
        dist.broadcast(parameter.data, src=0, group=world_group)
    return stage_b_parameter_consensus(
        named,
        world_group,
        "initial adapter",
        expected_count=expected_count,
    )


_STEP0_PARITY_SHARED_FIELDS = (
    "iid",
    "row_position",
    "manifest_index",
    "noise_seed",
    "optimizer_step",
    "checkpoint_interval",
    "step_in_checkpoint_interval",
    "microbatch_index",
    "interval_micro_ordinal",
    "interval_schedule_cycle",
    "schedule_index",
    "timestep_int64",
    "sigma",
    "sigma_float32_be_hex",
    "memory_input_kind",
    "input_patch_shape",
    "prediction_shape",
    "zero_output_projection_names",
    "zero_output_projections_exact",
    "optimizer_constructed",
    "checkpoint_written",
)
_STEP0_PARITY_SP_LOCAL_FIELDS = (
    "world_rank",
    "sp_rank",
    "disabled_route_receipt",
    "enabled_route_receipt",
    "local_target_selector_count",
    "disabled_route_prediction_sha256",
    "enabled_zero_init_route_prediction_sha256",
    "bit_exact_equal",
)
_STEP0_ROUTE_FIELDS = {
    "total_tokens",
    "condition_tokens",
    "target_tokens",
    "sequence_parallel_rank",
    "sequence_parallel_size",
    "enabled",
    "memory_digest",
    "query_rows",
    "key_value_rows",
    "digest",
}


def _validated_step0_route_receipt(
    value: Any, *, sp_rank: int, enabled: bool
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _STEP0_ROUTE_FIELDS:
        fail("step-0 parity SP-local route receipt fields differ")
    unsigned = dict(value)
    declared_digest = unsigned.pop("digest", None)
    total = value.get("total_tokens")
    condition = value.get("condition_tokens")
    target = value.get("target_tokens")
    memory_digest = value.get("memory_digest")
    if (
        type(total) is not int
        or type(condition) is not int
        or type(target) is not int
        or total <= 0
        or not 0 <= condition < total
        or target != total - condition
        or value.get("sequence_parallel_rank") != sp_rank
        or value.get("sequence_parallel_size") != SP_SIZE
        or value.get("enabled") is not enabled
        or value.get("query_rows") != "local_target_suffix_only"
        or value.get("key_value_rows")
        != "independent_registered_source_visual_memory_only"
        or declared_digest != object_sha256(unsigned)
    ):
        fail("step-0 parity SP-local route receipt differs")
    if enabled:
        _sha(memory_digest, length=64, label="step-0 enabled-route memory digest")
    elif memory_digest is not None:
        fail("step-0 disabled route unexpectedly carries visual memory")
    return dict(value)


def validate_step0_exact_base_parity_world8(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Validate per-rank parity without equating legitimate SP-local outputs."""

    if isinstance(records, (str, bytes)) or len(records) != WORLD_SIZE:
        fail("step-0 parity requires exactly eight gathered rank records")
    required_fields = {
        "dp_arm",
        *_STEP0_PARITY_SHARED_FIELDS,
        *_STEP0_PARITY_SP_LOCAL_FIELDS,
    }
    normalized: list[Mapping[str, Any]] = []
    seen_world_ranks: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != required_fields:
            fail("step-0 parity gathered rank record fields differ")
        world_rank = record.get("world_rank")
        dp_arm = record.get("dp_arm")
        sp_rank = record.get("sp_rank")
        if (
            type(world_rank) is not int
            or type(dp_arm) is not int
            or type(sp_rank) is not int
            or dp_arm not in range(DP_SIZE)
            or sp_rank not in range(SP_SIZE)
            or world_rank != dp_arm * SP_SIZE + sp_rank
            or world_rank in seen_world_ranks
        ):
            fail("step-0 parity WORLD8 rank/DP/SP binding differs")
        seen_world_ranks.add(world_rank)
        disabled_sha = _sha(
            record.get("disabled_route_prediction_sha256"),
            length=64,
            label="step-0 disabled-route prediction",
        )
        enabled_sha = _sha(
            record.get("enabled_zero_init_route_prediction_sha256"),
            length=64,
            label="step-0 enabled-route prediction",
        )
        if (
            record.get("bit_exact_equal") is not True
            or disabled_sha != enabled_sha
            or record.get("zero_output_projections_exact") is not True
            or record.get("optimizer_constructed") is not False
            or record.get("checkpoint_written") is not False
        ):
            fail("step-0 per-rank base/zero-init parity differs")
        contract.validate_memory_input_kind(record.get("memory_input_kind"))
        disabled_route = _validated_step0_route_receipt(
            record.get("disabled_route_receipt"),
            sp_rank=sp_rank,
            enabled=False,
        )
        enabled_route = _validated_step0_route_receipt(
            record.get("enabled_route_receipt"),
            sp_rank=sp_rank,
            enabled=True,
        )
        shared_route_fields = (
            "total_tokens",
            "condition_tokens",
            "target_tokens",
            "sequence_parallel_rank",
            "sequence_parallel_size",
            "query_rows",
            "key_value_rows",
        )
        if any(disabled_route[key] != enabled_route[key] for key in shared_route_fields):
            fail("step-0 disabled/enabled SP-local route geometry differs")
        local_length = math.ceil(disabled_route["total_tokens"] / SP_SIZE)
        local_start = sp_rank * local_length
        local_stop = local_start + local_length
        expected_selector_count = max(
            0,
            min(local_stop, disabled_route["total_tokens"])
            - max(local_start, disabled_route["condition_tokens"]),
        )
        if record.get("local_target_selector_count") != expected_selector_count:
            fail("step-0 SP-local target selector count differs")
        normalized.append(dict(record))
    if seen_world_ranks != set(range(WORLD_SIZE)):
        fail("step-0 parity WORLD8 rank closure differs")

    arm_records = []
    shared_route_field_names = (
        "total_tokens",
        "condition_tokens",
        "target_tokens",
        "sequence_parallel_size",
        "query_rows",
        "key_value_rows",
    )
    for dp_arm in range(DP_SIZE):
        local = sorted(
            (record for record in normalized if record["dp_arm"] == dp_arm),
            key=lambda record: record["sp_rank"],
        )
        if [record["sp_rank"] for record in local] != list(range(SP_SIZE)):
            fail("step-0 parity DP arm lacks exact SP ranks 0..3")
        shared = {
            key: local[0][key] for key in _STEP0_PARITY_SHARED_FIELDS
        }
        if any(
            {key: record[key] for key in _STEP0_PARITY_SHARED_FIELDS} != shared
            for record in local[1:]
        ):
            fail("step-0 parity DP-arm shared semantic fields differ")
        shared_route_semantics = {
            key: local[0]["disabled_route_receipt"][key]
            for key in shared_route_field_names
        }
        shared_route_semantics["enabled_memory_digest"] = local[0][
            "enabled_route_receipt"
        ]["memory_digest"]
        if any(
            {
                **{
                    key: record["disabled_route_receipt"][key]
                    for key in shared_route_field_names
                },
                "enabled_memory_digest": record["enabled_route_receipt"][
                    "memory_digest"
                ],
            }
            != shared_route_semantics
            for record in local[1:]
        ):
            fail("step-0 parity DP-arm shared route semantic fields differ")
        arm_records.append(
            {
                "dp_arm": dp_arm,
                "shared_semantics": shared,
                "shared_semantics_digest": object_sha256(shared),
                "shared_route_semantics": shared_route_semantics,
                "shared_route_semantics_digest": object_sha256(
                    shared_route_semantics
                ),
                "sp_ranks": list(range(SP_SIZE)),
                "sp_local_records": [
                    {
                        key: record[key]
                        for key in _STEP0_PARITY_SP_LOCAL_FIELDS
                    }
                    for record in local
                ],
                "per_rank_base_vs_zero_init_sha_equal": True,
                "cross_sp_prediction_sha_equality_required": False,
            }
        )
    unsigned = {
        "world_size": WORLD_SIZE,
        "physical_dp_size": DP_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "world_ranks": list(range(WORLD_SIZE)),
        "shared_fields_required_equal_within_dp_arm": list(
            _STEP0_PARITY_SHARED_FIELDS
        ),
        "shared_route_fields_required_equal_within_dp_arm": [
            *shared_route_field_names,
            "enabled_memory_digest",
        ],
        "sp_local_fields_allowed_to_differ": list(_STEP0_PARITY_SP_LOCAL_FIELDS),
        "dp_arm_records": arm_records,
        "bit_exact_base_vs_zero_init_on_every_rank": True,
        "cross_sp_prediction_sha_equality_required": False,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, length: int, label: str) -> str:
    expression = _SHA1 if length == 40 else _SHA256
    if type(value) is not str or expression.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA digest")
    return value


def _validated_source_artifact(
    path_value: str | Path, *, expected_sha256: str, label: str
) -> Path:
    requested = Path(path_value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise CleanSourceVisualStageBTrainingError(
            f"{label} is unavailable: {error}"
        ) from error
    if resolved != requested or not resolved.is_file() or resolved.is_symlink():
        fail(f"{label} must be one canonical plain file")
    if contract.file_sha256(resolved) != _sha(
        expected_sha256, length=64, label=f"{label} expected SHA"
    ):
        fail(f"{label} SHA-256 differs")
    return resolved


def validate_executed_method_release(
    *,
    method_root: Path,
    archive_path: str | Path,
    archive_sha256: str,
    manifest_path: str | Path,
    manifest_sha256: str,
    method_revision: str,
) -> Mapping[str, Any]:
    """Bind executed bytes to one canonical exact-member source archive."""

    archive = _validated_source_artifact(
        archive_path,
        expected_sha256=archive_sha256,
        label="method source archive",
    )
    manifest_file = _validated_source_artifact(
        manifest_path,
        expected_sha256=manifest_sha256,
        label="method source manifest",
    )
    try:
        manifest_raw = manifest_file.read_bytes()
        manifest = json.loads(manifest_raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CleanSourceVisualStageBTrainingError(
            f"cannot read method source manifest: {error}"
        ) from error
    if not isinstance(manifest, Mapping):
        fail("method source manifest root differs")
    expected_fields = {
        "schema_version",
        "release_generation",
        "archive_format",
        "member_root",
        "file_count",
        "files",
        "revision_kind",
        "content_closure_sha1",
        "git_commit_claimed",
        "exact_member_closure",
        "manifest_digest",
    }
    if (
        set(manifest) != expected_fields
        or manifest_raw != canonical_json_bytes(manifest) + b"\n"
        or manifest.get("schema_version") != METHOD_RELEASE_SCHEMA
        or manifest.get("release_generation") != METHOD_RELEASE_GENERATION
        or manifest.get("archive_format") != METHOD_RELEASE_ARCHIVE_FORMAT
        or manifest.get("member_root") != METHOD_RELEASE_MEMBER_ROOT
        or manifest.get("revision_kind") != "content-closure-sha1"
        or manifest.get("git_commit_claimed") is not False
        or manifest.get("exact_member_closure") is not True
    ):
        fail("method source manifest schema/canonical closure differs")
    unsigned = dict(manifest)
    declared_manifest_digest = unsigned.pop("manifest_digest", None)
    if declared_manifest_digest != object_sha256(unsigned):
        fail("method source embedded manifest digest differs")
    rows = manifest.get("files")
    if not isinstance(rows, list) or manifest.get("file_count") != len(rows):
        fail("method source manifest file count differs")
    normalized_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "sha256",
            "size",
            "mode",
        }:
            fail("method source manifest member fields differ")
        relative = row.get("path")
        pure = PurePosixPath(str(relative))
        if (
            type(relative) is not str
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != relative
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or row.get("mode") != "0444"
        ):
            fail("method source manifest member path/metadata differs")
        _sha(row.get("sha256"), length=64, label=f"method member {relative}")
        normalized_rows.append(dict(row))
    relative_names = tuple(str(row["path"]) for row in normalized_rows)
    if relative_names != METHOD_RELEASE_FILES or len(set(relative_names)) != len(
        relative_names
    ):
        fail("method source exact required member closure differs")
    closure_payload = {
        "member_root": METHOD_RELEASE_MEMBER_ROOT,
        "files": normalized_rows,
    }
    closure_sha1 = hashlib.sha1(canonical_json_bytes(closure_payload)).hexdigest()
    if (
        manifest.get("content_closure_sha1") != closure_sha1
        or method_revision != closure_sha1
    ):
        fail("method source content-closure revision differs")

    expected_archive_names = [
        f"{METHOD_RELEASE_MEMBER_ROOT}/{relative}" for relative in relative_names
    ]
    try:
        with tarfile.open(archive, mode="r:") as tar:
            members = tar.getmembers()
            if [member.name for member in members] != expected_archive_names:
                fail("method source archive member closure differs")
            for member, row in zip(members, normalized_rows):
                payload = tar.extractfile(member)
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.uid != 0
                    or member.gid != 0
                    or member.mtime != 0
                    or stat.S_IMODE(member.mode) != 0o444
                    or member.size != row["size"]
                    or payload is None
                    or hashlib.sha256(payload.read()).hexdigest()
                    != row["sha256"]
                ):
                    fail(f"method source archive member differs: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise CleanSourceVisualStageBTrainingError(
            f"cannot validate method source archive: {error}"
        ) from error

    root = method_root.resolve(strict=True)
    if (
        root != method_root
        or root.is_symlink()
        or not root.is_dir()
        or tuple(root.parts[-2:]) != ("methods", "bernini_action_editing")
    ):
        fail("executed method root is not the canonical extracted member root")
    observed_files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            fail("executed method root contains a symlink/special entry")
        if stat.S_ISREG(mode):
            observed_files.add(relative)
    if observed_files != set(relative_names):
        fail("executed method root file set differs from sealed archive")
    executed_entries: list[Mapping[str, Any]] = []
    for row in normalized_rows:
        path = root / str(row["path"])
        mode = path.lstat().st_mode
        if (
            not stat.S_ISREG(mode)
            or stat.S_IMODE(mode) != 0o444
            or path.stat().st_size != row["size"]
            or data_contract.file_sha256(path) != row["sha256"]
        ):
            fail(f"executed method member differs: {row['path']}")
        executed_entries.append(
            {"path": row["path"], "sha256": row["sha256"]}
        )
    value = {
        "method_root": str(root),
        "schema_version": METHOD_RELEASE_SCHEMA,
        "archive": str(archive),
        "archive_sha256": archive_sha256,
        "manifest": str(manifest_file),
        "manifest_sha256": manifest_sha256,
        "manifest_digest": declared_manifest_digest,
        "revision": method_revision,
        "content_closure_sha1": closure_sha1,
        "exact_member_count": len(executed_entries),
        "archive_members_verified": True,
        "executed_tree_exact_member_closure": True,
        "executed_entries_digest": object_sha256(executed_entries),
    }
    return {**value, "digest": object_sha256(value)}


def validate_checkpoint_content(
    checkpoint: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Rehash every non-cache base-checkpoint file against the pinned list."""

    root = checkpoint.resolve(strict=True)
    if root != checkpoint or root.is_symlink() or not root.is_dir():
        fail("base checkpoint content root differs")
    manifest = _validated_source_artifact(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        label="checkpoint content manifest",
    )
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CleanSourceVisualStageBTrainingError(
            f"cannot read checkpoint content manifest: {error}"
        ) from error
    if len(lines) != EXPECTED_CHECKPOINT_CONTENT_FILE_COUNT:
        fail("checkpoint content manifest file count differs")
    expected: dict[str, str] = {}
    pattern = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            fail("checkpoint manifest line is not canonical sha256sum syntax")
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not normalized
            or normalized in expected
        ):
            fail("checkpoint manifest contains an unsafe/duplicate path")
        expected[normalized] = digest
    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            fail("checkpoint contains a non-cache symlink")
        if stat.S_ISREG(mode):
            actual.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            fail("checkpoint contains a non-regular filesystem entry")
    if actual != set(expected):
        fail("checkpoint non-cache file set differs from pinned manifest")
    entries = []
    for relative in sorted(expected):
        path = root / relative
        if path.resolve(strict=True) != path or not path.is_file() or path.is_symlink():
            fail(f"checkpoint file path differs: {relative}")
        digest = data_contract.file_sha256(path)
        if digest != expected[relative]:
            fail(f"checkpoint content hash differs: {relative}")
        entries.append({"path": relative, "sha256": digest})
    value = {
        "checkpoint_root": str(root),
        "tree_sha256": contract.EXPECTED_CHECKPOINT_TREE_SHA256,
        "manifest_path": str(manifest),
        "manifest_sha256": expected_manifest_sha256,
        "verified_file_count": len(entries),
        "every_non_cache_file_sha256_verified": True,
        "verified_entries_digest": object_sha256(entries),
    }
    return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class PackedNoOpCondition:
    input_patches: Any
    rotary: Any
    condition_tokens: int
    target_tokens: int
    total_tokens: int
    target_velocity: Any
    memory_input: Any
    tensor_identities: Mapping[str, str]


@dataclass(frozen=True)
class PreparedArm:
    dp_arm: int
    row_position: int
    manifest_index: int
    iid: str
    action_family: str
    source_video_sha256: str
    noise_seed: int
    condition: PackedNoOpCondition
    memory: visual.CleanSourceVisualMemory
    route: visual.VisualContextRoute
    source_order_gate: Optional[Mapping[str, Any]]


def _noise_seed(
    base_seed: int,
    optimizer_step_zero_based: int,
    microbatch_index: int,
    dp_arm: int,
) -> int:
    raw = (
        f"{base_seed}\0clean-source-visual-context-stage-b-v1\0"
        f"{optimizer_step_zero_based}\0{microbatch_index}\0{dp_arm}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31)


def _pack_latent_patches(latent: Any) -> Any:
    import torch

    if (
        not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float32
        or latent.device.type != "cpu"
        or latent.ndim != 4
        or tuple(int(item) for item in latent.shape[:2]) != (16, LATENT_PHASES)
        or int(latent.shape[2]) <= 0
        or int(latent.shape[3]) <= 0
        or int(latent.shape[2]) % 2
        or int(latent.shape[3]) % 2
        or not latent.is_contiguous()
        or not bool(torch.isfinite(latent).all().item())
    ):
        fail("latent must be detached CPU FP32 [16,21,evenH,evenW]")
    channels, phases, height, width = (int(item) for item in latent.shape)
    return (
        latent.reshape(channels, phases, height // 2, 2, width // 2, 2)
        .permute(1, 2, 4, 0, 3, 5)
        .reshape(phases * (height // 2) * (width // 2), channels, 1, 2, 2)
        .contiguous()
    )


def _forward_noised(clean: Any, epsilon: Any, sigma: float) -> Any:
    import torch

    if (
        not isinstance(clean, torch.Tensor)
        or not isinstance(epsilon, torch.Tensor)
        or clean is epsilon
        or tuple(clean.shape) != tuple(epsilon.shape)
        or clean.dtype != torch.float32
        or epsilon.dtype != torch.float32
        or clean.device.type != "cpu"
        or epsilon.device.type != "cpu"
        or not 0.0 < sigma <= 1.0
    ):
        fail("forward-noising inputs differ")
    return ((1.0 - sigma) * clean + sigma * epsilon).detach().contiguous()


def prepare_noop_condition(
    *,
    clean_source: Any,
    epsilon: Any,
    coordinate: contract.Exact80Coordinate,
    memory_input_kind: str,
    rope: Any,
    device: Any,
    runtime: Any,
) -> PackedNoOpCondition:
    """Pack native clean-source + noisy-target and a separate memory input."""

    import torch

    kind = contract.validate_memory_input_kind(memory_input_kind)
    if (
        not isinstance(clean_source, torch.Tensor)
        or clean_source.dtype != torch.float32
        or clean_source.device.type != "cpu"
        or clean_source.ndim != 5
        or tuple(int(item) for item in clean_source.shape[:3])
        != (1, 16, LATENT_PHASES)
        or clean_source.requires_grad
        or not clean_source.is_contiguous()
    ):
        fail("clean source geometry differs")
    clean = clean_source.squeeze(0).contiguous()
    epsilon4 = epsilon.squeeze(0).contiguous()
    noisy_target = _forward_noised(clean, epsilon4, coordinate.sigma)
    target_velocity = (epsilon4 - clean).detach().contiguous()
    memory_input = clean if kind == "clean_source" else noisy_target
    donor_patches = _pack_latent_patches(clean)
    target_patches = _pack_latent_patches(noisy_target)
    velocity_patches = _pack_latent_patches(target_velocity)
    condition_tokens = int(donor_patches.shape[0])
    target_tokens = int(target_patches.shape[0])
    if condition_tokens != target_tokens:
        fail("native no-op source/target token geometry differs")
    input_patches = torch.cat((donor_patches, target_patches), dim=0).to(device)
    donor_rope = rope(clean_source.to(device), source_id=1)
    target_rope = rope(noisy_target.unsqueeze(0).to(device), source_id=0)
    rotary = torch.cat((donor_rope, target_rope), dim=2)
    rotary = rotary.squeeze(0).permute(1, 0, 2).contiguous()
    if int(rotary.shape[0]) != condition_tokens + target_tokens:
        fail("native no-op rotary geometry differs")
    target_field = runtime.packed_output_field(velocity_patches).to(device)
    identities = {
        "clean_source": runtime.tensor_sha256(clean_source),
        "epsilon": runtime.tensor_sha256(epsilon),
        "noisy_target": runtime.tensor_sha256(noisy_target),
        "target_velocity": runtime.tensor_sha256(target_velocity),
        "native_clean_source_condition": runtime.tensor_sha256(clean),
        "visual_context_input": runtime.tensor_sha256(memory_input),
    }
    if (
        kind == "clean_source"
        and identities["visual_context_input"]
        != identities["native_clean_source_condition"]
    ):
        fail("clean visual-context input identity differs")
    if (
        kind == "same_noise_forward_noised_source"
        and identities["visual_context_input"] != identities["noisy_target"]
    ):
        fail("same-noise visual-context input identity differs")
    return PackedNoOpCondition(
        input_patches=input_patches,
        rotary=rotary,
        condition_tokens=condition_tokens,
        target_tokens=target_tokens,
        total_tokens=condition_tokens + target_tokens,
        target_velocity=target_field,
        memory_input=memory_input.unsqueeze(0).contiguous(),
        tensor_identities=identities,
    )


def source_order_structural_gate(
    *,
    handle: visual.CleanSourceVisualContextHandle,
    clean_source: Any,
    source_video_sha256: str,
    runtime: Any,
) -> Mapping[str, Any]:
    """Prove reverse-phase memory is not just a K/V token permutation."""

    import torch

    clean = clean_source.to(
        device=next(handle.encoder.parameters()).device,
        dtype=torch.float32,
    ).detach().contiguous()
    reverse = clean.flip(2).contiguous()
    with torch.no_grad():
        forward_memory = handle.build_memory(
            clean,
            source_video_sha256=source_video_sha256,
            memory_input_latent_sha256=runtime.tensor_sha256(clean_source),
            input_kind="clean_source",
        )
        reverse_memory = handle.build_memory(
            reverse,
            source_video_sha256=source_video_sha256,
            memory_input_latent_sha256=runtime.tensor_sha256(
                clean_source.flip(2).contiguous()
            ),
            input_kind="clean_source",
        )
    phases, height, width = forward_memory.pooled_grid
    if phases != LATENT_PHASES or reverse_memory.pooled_grid != forward_memory.pooled_grid:
        fail("source-order gate lost exact21 phase geometry")
    forward_phase_reversal = (
        forward_memory.tokens.reshape(1, phases, height * width, -1)
        .flip(1)
        .reshape_as(forward_memory.tokens)
        .contiguous()
    )
    if torch.equal(reverse_memory.tokens, forward_phase_reversal):
        fail("reverse-phase memory remains a bare token permutation")
    value = {
        "phase_count": phases,
        "spatial_tokens_per_phase": height * width,
        "position_representation": "fixed_absolute_3d_fourier_phase_y_x_v1",
        "forward_memory_sha256": runtime.tensor_sha256(forward_memory.tokens),
        "reverse_memory_sha256": runtime.tensor_sha256(reverse_memory.tokens),
        "forward_phase_reversal_sha256": runtime.tensor_sha256(
            forward_phase_reversal
        ),
        "reverse_is_bare_forward_token_permutation": False,
        "feature_evaluator_used": False,
        "optimizer_supervision": False,
    }
    return {**value, "digest": object_sha256(value)}


def _prediction(
    *,
    renderer: Any,
    transformer: Any,
    condition: PackedNoOpCondition,
    coordinate: contract.Exact80Coordinate,
    text_lens: Any,
    text_embs: Any,
) -> Any:
    import torch

    embedded = transformer.patch_embedding(condition.input_patches).flatten(1).unsqueeze(0)
    rotary = condition.rotary.permute(1, 0, 2).unsqueeze(0)
    value = renderer.diff_dec.shared_step(
        model_id="transformer_1",
        noisy_latents=embedded,
        timesteps=embedded.new_tensor([coordinate.timestep], dtype=torch.int64),
        cond_embeds=text_embs,
        rotary_embs=rotary,
        batch_vae_seqlen=[condition.total_tokens],
        batch_text_seqlen=text_lens,
    )
    target = value[
        :,
        condition.condition_tokens : condition.condition_tokens
        + condition.target_tokens,
        :,
    ]
    if tuple(target.shape) != (1, condition.target_tokens, PATCH_VALUES):
        fail("target-only flow prediction geometry differs")
    return target


def _zero_dependency(prediction: Any, trainable: Sequence[tuple[str, Any]]) -> Any:
    """Materialize exact-zero grads on SP ranks that own no target suffix."""

    anchor = prediction.new_zeros(())
    for _, parameter in trainable:
        anchor = anchor + parameter.reshape(-1)[0].to(prediction.dtype) * 0.0
    return prediction + anchor


def grouped_gradient_norms(
    trainable: Sequence[tuple[str, Any]],
) -> Mapping[str, float]:
    """Return exhaustive FP64 norms for every representation-learning path."""

    import torch

    if not trainable:
        fail("visual-context grouped gradient scope is empty")
    device = trainable[0][1].device
    squared = {
        name: torch.zeros((), dtype=torch.float64, device=device)
        for name in ("encoder", "query", "key", "value", "output", "gate")
    }
    counts = {name: 0 for name in squared}
    for name, parameter in trainable:
        if name.startswith("encoder."):
            group = "encoder"
        elif ".query." in name or ".query_norm." in name:
            group = "query"
        elif ".key." in name:
            group = "key"
        elif ".value." in name:
            group = "value"
        elif ".output." in name:
            group = "output"
        elif name.endswith(".residual_gain"):
            group = "gate"
        else:
            fail(f"unclassified visual-context gradient parameter: {name}")
        if parameter.grad is None:
            fail(f"visual-context gradient is absent: {name}")
        gradient = parameter.grad.detach().to(dtype=torch.float64)
        squared[group] = squared[group] + torch.sum(gradient * gradient)
        counts[group] += 1
    if any(count <= 0 for count in counts.values()):
        fail("visual-context grouped gradient scope is incomplete")
    result = {
        name: float(torch.sqrt(value).item()) for name, value in squared.items()
    }
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        fail("visual-context grouped gradient norm is non-finite")
    return result


def linux_host_memory_receipt() -> Mapping[str, float]:
    """Read current/peak process RSS from Linux procfs without sampling threads."""

    values: dict[str, float] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                values["current_rss_gib"] = float(line.split()[1]) / float(1024**2)
            elif line.startswith("VmHWM:"):
                values["peak_rss_gib"] = float(line.split()[1]) / float(1024**2)
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise CleanSourceVisualStageBTrainingError(
            f"cannot read Linux process RSS: {error}"
        ) from error
    if set(values) != {"current_rss_gib", "peak_rss_gib"}:
        fail("Linux process RSS fields differ")
    return values


def run_backward_feasibility_microbatches(
    *,
    args: argparse.Namespace,
    runtime: Any,
    renderer: Any,
    transformer: Any,
    handle: visual.CleanSourceVisualContextHandle,
    trainable: Sequence[tuple[str, Any]],
    parallel: Any,
    distributed: Any,
    device: Any,
    rope: Any,
    text_lens: Any,
    text_embs: Any,
    train_rows: Sequence[Any],
    manifest_index_by_iid: Mapping[str, int],
    store: Any,
    initial_parameter_digest: str,
) -> Mapping[str, Any]:
    """Run one exact accumulation window without constructing an optimizer."""

    import torch
    import torch.distributed as dist

    if any(parameter.grad is not None for _, parameter in trainable):
        fail("backward feasibility requires an empty initial gradient state")
    before = stage_b_parameter_consensus(
        trainable, parallel.world_group, "backward-feasibility parameters before"
    )
    if before != initial_parameter_digest:
        fail("backward-feasibility initialization digest differs")

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    coordinates = contract.coordinates_for_optimizer_step(0)
    if len(coordinates) != contract.GRADIENT_ACCUMULATION_STEPS:
        fail("backward feasibility requires exactly four microbatches")
    microbatch_receipts: list[Mapping[str, Any]] = []

    for coordinate in coordinates:
        row_position = contract.train_row_position(
            optimizer_step_zero_based=0,
            microbatch_index=coordinate.microbatch_index,
            dp_arm=distributed.arm_index,
        )
        selected_row = train_rows[row_position]
        manifest_index = manifest_index_by_iid[selected_row.iid]
        sample = store.load(manifest_index)
        if sample.split != "train" or sample.iid != selected_row.iid:
            fail("backward feasibility accessed a non-train source row")
        runtime.digest_consensus(
            str(sample.receipt()["digest"]),
            group=parallel.sp_group,
            expected_count=SP_SIZE,
            label=f"backward microbatch {coordinate.microbatch_index} DP source row",
        )
        seed = _noise_seed(
            args.seed, 0, coordinate.microbatch_index, distributed.arm_index
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        epsilon = torch.randn(
            tuple(sample.clean_noop_target.shape),
            generator=generator,
            dtype=torch.float32,
        ).contiguous()
        packed = prepare_noop_condition(
            clean_source=sample.clean_noop_target,
            epsilon=epsilon,
            coordinate=coordinate,
            memory_input_kind=args.memory_input_kind,
            rope=rope,
            device=device,
            runtime=runtime,
        )
        memory_input_device = packed.memory_input.to(
            device=device, dtype=torch.float32
        ).detach().contiguous()
        memory = handle.build_memory(
            memory_input_device,
            source_video_sha256=sample.source_video_sha256,
            memory_input_latent_sha256=packed.tensor_identities[
                "visual_context_input"
            ],
            input_kind=args.memory_input_kind,
        )
        route = visual.VisualContextRoute(
            packed.total_tokens,
            packed.condition_tokens,
            distributed.sp_rank,
            SP_SIZE,
            memory,
        )
        with handle.route(route):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = _prediction(
                    renderer=renderer,
                    transformer=transformer,
                    condition=packed,
                    coordinate=coordinate,
                    text_lens=text_lens,
                    text_embs=text_embs,
                )
                prediction = _zero_dependency(prediction, trainable)
                raw_loss = visual.no_op_flow_matching_loss(
                    prediction=prediction,
                    target_velocity=packed.target_velocity,
                )
                scaled_loss = raw_loss / float(
                    contract.GRADIENT_ACCUMULATION_STEPS
                )
            if not runtime.world_all_true(
                bool(torch.isfinite(scaled_loss.detach()).item()),
                group=parallel.world_group,
            ):
                fail("non-finite no-op loss blocked backward feasibility")
            raw_loss_value = float(raw_loss.detach().item())
            scaled_loss_value = float(scaled_loss.detach().item())
            scaled_loss.backward()

        torch.cuda.synchronize(device)
        local = {
            "world_rank": distributed.rank,
            "dp_arm": distributed.arm_index,
            "sp_rank": distributed.sp_rank,
            **coordinate.receipt(),
            "row_position": row_position,
            "manifest_index": manifest_index,
            "iid": sample.iid,
            "action_family": selected_row.action_family,
            "source_video_sha256": sample.source_video_sha256,
            "noise_seed": seed,
            "memory_input_kind": args.memory_input_kind,
            "raw_loss": raw_loss_value,
            "scaled_loss_backward": scaled_loss_value,
            "loss_divisor_before_backward": contract.GRADIENT_ACCUMULATION_STEPS,
            "target_rows_only": True,
            "backward_executed": True,
            "optimizer_constructed": False,
            "optimizer_step_executed": False,
            "synthetic_target_accessed": False,
            "reward_used": False,
        }
        sp_projection = {
            key: value
            for key, value in local.items()
            if key not in {"world_rank", "sp_rank"}
        }
        runtime.digest_consensus(
            object_sha256(sp_projection),
            group=parallel.sp_group,
            expected_count=SP_SIZE,
            label=f"backward microbatch {coordinate.microbatch_index} SP4 record",
        )
        gathered: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local, group=parallel.world_group)
        leaders = [gathered[0], gathered[4]]
        if (
            [item.get("dp_arm") for item in leaders] != [0, 1]
            or len({item.get("iid") for item in leaders}) != 2
            or any(
                item.get("microbatch_index") != coordinate.microbatch_index
                or item.get("schedule_index") != coordinate.schedule_index
                for item in leaders
            )
        ):
            fail("backward-feasibility WORLD8 microbatch closure differs")
        microbatch_receipts.append(
            {
                **coordinate.receipt(),
                "logical_records": [
                    {key: value for key, value in item.items() if key != "sp_rank"}
                    for item in leaders
                ],
                "world8_backward_complete": True,
            }
        )
        del (
            sample,
            epsilon,
            packed,
            memory_input_device,
            memory,
            route,
            prediction,
            raw_loss,
            scaled_loss,
            local,
            sp_projection,
            gathered,
            leaders,
        )

    if any(parameter.grad is None for _, parameter in trainable):
        fail("backward feasibility did not materialize every adapter gradient")
    synchronized_norm = runtime.synchronize_gradients(trainable, parallel)
    component_norms = grouped_gradient_norms(trainable)
    if component_norms["output"] <= 0.0:
        fail("backward feasibility output projection received no gradient")
    gradient_digest = stage_b_gradients_digest(trainable)
    gradient_digests: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gradient_digests, gradient_digest, group=parallel.world_group
    )
    if len(set(gradient_digests)) != 1:
        fail("DP2xSP4 synchronized gradients differ across WORLD8")
    norm_record = {
        "total": synchronized_norm,
        "components": dict(component_norms),
        "gradient_sha256": gradient_digest,
    }
    norm_records: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(norm_records, norm_record, group=parallel.world_group)
    if any(record != norm_record for record in norm_records):
        fail("DP2xSP4 synchronized gradient norm closure differs")

    after = stage_b_parameter_consensus(
        trainable, parallel.world_group, "backward-feasibility parameters after"
    )
    if after != before:
        fail("backward feasibility changed adapter parameters")
    torch.cuda.synchronize(device)
    gib = float(1024**3)
    local_resources = {
        "world_rank": distributed.rank,
        "dp_arm": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "device_name": torch.cuda.get_device_name(device),
        "allocated_gib_after_backward": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib_after_backward": torch.cuda.memory_reserved(device) / gib,
        "peak_allocated_gib_four_microbatch_backward": (
            torch.cuda.max_memory_allocated(device) / gib
        ),
        "peak_reserved_gib_four_microbatch_backward": (
            torch.cuda.max_memory_reserved(device) / gib
        ),
        "host_process_memory": linux_host_memory_receipt(),
    }
    resource_world: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(resource_world, local_resources, group=parallel.world_group)
    if [item.get("world_rank") for item in resource_world] != list(range(WORLD_SIZE)):
        fail("backward-feasibility resource closure differs")

    for _, parameter in trainable:
        parameter.grad = None
    gc.collect()
    elapsed = time.monotonic() - started
    value = {
        "microbatches_per_dp_arm": len(microbatch_receipts),
        "logical_records": len(microbatch_receipts) * DP_SIZE,
        "gradient_accumulation_steps": contract.GRADIENT_ACCUMULATION_STEPS,
        "microbatches": microbatch_receipts,
        "gradient_sync": {
            "order": "SP4_mean_then_DP2_mean",
            "finite_all_parameters_world8": True,
            "identical_full_gradient_digest_world8": True,
            "zero_init_output_projection_required_positive": True,
            "zero_init_upstream_components_may_be_exact_zero": [
                "encoder",
                "query",
                "key",
                "value",
                "gate",
            ],
            "total_norm": synchronized_norm,
            "component_norms": dict(component_norms),
            "gradient_sha256": gradient_digest,
        },
        "parameters": {
            "sha256_before": before,
            "sha256_after": after,
            "unchanged": True,
            "gradients_cleared_after_measurement": True,
        },
        "resources_world8": resource_world,
        "elapsed_seconds": elapsed,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "checkpoint_written": False,
    }
    return {**value, "digest": object_sha256(value)}


def _atomic_adapter_safetensors(
    path: Path,
    handle: visual.CleanSourceVisualContextHandle,
    *,
    memory_input_kind: str,
    manifest_digest: str,
    admission_digest: str,
    runtime: Any,
) -> None:
    from safetensors.torch import save_file

    tensors = handle.state_dict_for_save()
    metadata = {
        "schema_version": ADAPTER_FILE_SCHEMA,
        "adapter_schema_version": visual.SCHEMA_VERSION,
        "memory_input_kind": memory_input_kind,
        "block_indices_json": canonical_json_bytes(
            list(handle.block_indices)
        ).decode("ascii"),
        "source_only_manifest_digest": manifest_digest,
        "stage_a_admission_digest": admission_digest,
        "optimizer_steps": str(contract.OPTIMIZER_STEPS),
        "checkpoint_steps_json": canonical_json_bytes(
            list(contract.CHECKPOINT_STEPS)
        ).decode("ascii"),
        "objective": "standard_target_only_noop_flow_matching",
        "synthetic_target_accessed": "false",
        "reward_used": "false",
    }
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".safetensors",
            delete=False,
        ) as file_handle:
            temporary = Path(file_handle.name)
        save_file(dict(tensors), str(temporary), metadata=metadata)
        runtime.durable_file_replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resource_budget_receipt(memory_input_kind: str) -> Mapping[str, Any]:
    kind = contract.validate_memory_input_kind(memory_input_kind)
    if kind == "clean_source":
        elapsed = (280, 600)
        gpu = (28, 48)
        host = (43, 59)
    else:
        elapsed = (285, 615)
        gpu = (29, 49)
        host = (43, 59)
    value = {
        "arm": kind,
        "optimizer_steps": 80,
        "gradient_accumulation_steps": 4,
        "training_model_forwards_per_dp_arm": 320,
        "pre_optimizer_exact_parity_forwards_per_dp_arm": 2,
        "total_model_forwards_per_dp_arm": 322,
        "logical_training_records": 640,
        "world_size": 8,
        "gpu_type": "MI210-64GiB",
        "estimated_elapsed_minutes_low": elapsed[0],
        "estimated_elapsed_minutes_high": elapsed[1],
        "estimated_peak_gpu_memory_gib_low": gpu[0],
        "estimated_peak_gpu_memory_gib_high": gpu[1],
        "required_gpu_memory_gib": 64,
        "estimated_peak_host_memory_gib_low": host[0],
        "estimated_peak_host_memory_gib_high": host[1],
        "required_holder_host_memory_gib": 64,
        "estimate_only_until_gpu_smoke": True,
        "main_and_variant_run_independently": True,
    }
    return {**value, "digest": object_sha256(value)}


def pair_invariants_receipt(
    *,
    args: argparse.Namespace,
    manifest: data_contract.SourceOnlySplitManifest,
    method_release_identity: Mapping[str, Any],
    checkpoint_content_identity: Mapping[str, Any],
    initial_parameter_digest: str,
    stage_a_admission_digest: Optional[str],
) -> Mapping[str, Any]:
    """Fields that must be byte-identical across clean/noised pair arms."""

    _sha(initial_parameter_digest, length=64, label="initial parameter digest")
    if stage_a_admission_digest is not None:
        _sha(stage_a_admission_digest, length=64, label="Stage-A admission digest")
    value = {
        "source_only_manifest_file_sha256": (
            args.expected_source_only_manifest_sha256
        ),
        "source_only_manifest_digest": manifest.manifest_digest,
        "stage_a_admission_digest": stage_a_admission_digest,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "method_source_manifest_sha256": args.method_source_manifest_sha256,
        "method_source_manifest_digest": method_release_identity.get(
            "manifest_digest"
        ),
        "executed_method_entries_digest": method_release_identity.get(
            "executed_entries_digest"
        ),
        "bernini_commit": args.expected_bernini_commit,
        "veomni_commit": args.expected_veomni_commit,
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_content_manifest_sha256": (
            args.expected_checkpoint_content_manifest_sha256
        ),
        "checkpoint_verified_entries_digest": checkpoint_content_identity.get(
            "verified_entries_digest"
        ),
        "initial_parameter_digest": initial_parameter_digest,
        "seed": args.seed,
        "optimizer_steps": args.optimizer_steps,
        "gradient_accumulation_steps": contract.GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": contract.GLOBAL_BATCH,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "num_frames": args.num_frames,
        "exact40_schedule_sha256": contract.EXPECTED_SCHEDULE_SHA256,
        "objective": "standard_target_only_noop_flow_matching",
        "topology": TOPOLOGY,
        "synthetic_target_accessed": False,
        "reward_used": False,
    }
    return {**value, "digest": object_sha256(value)}


_BACKWARD_UPSTREAM_INVARIANT_FIELDS = (
    "source_only_manifest_file_sha256",
    "source_only_manifest_digest",
    "stage_a_admission_digest",
    "bernini_commit",
    "veomni_commit",
    "checkpoint_tree_sha256",
    "checkpoint_content_manifest_sha256",
    "checkpoint_verified_entries_digest",
    "initial_parameter_digest",
    "seed",
    "optimizer_steps",
    "gradient_accumulation_steps",
    "effective_global_batch",
    "learning_rate",
    "max_grad_norm",
    "num_frames",
    "exact40_schedule_sha256",
    "objective",
    "topology",
    "synthetic_target_accessed",
    "reward_used",
)


def validate_backward_upstream_pair_invariants(
    *, current: Mapping[str, Any], upstream: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Bind the smoke to the admitted pair while allowing a new code release.

    The method-release identity necessarily changes when this no-update scope is
    added.  Every scientific/runtime invariant from the prior paired WORLD8
    preflight remains exact, including initialization and source-only bytes.
    """

    if current.get("stage_a_admission_digest") is not None:
        fail("backward feasibility must remain pre-Stage-A and pre-optimizer")
    current_projection = {
        name: current.get(name) for name in _BACKWARD_UPSTREAM_INVARIANT_FIELDS
    }
    upstream_projection = {
        name: upstream.get(name) for name in _BACKWARD_UPSTREAM_INVARIANT_FIELDS
    }
    if current_projection != upstream_projection:
        fail("backward feasibility differs from admitted upstream pair invariants")
    value = {
        "comparison": "exact_except_new_method_release_identity",
        "fields": list(_BACKWARD_UPSTREAM_INVARIANT_FIELDS),
        "projection": current_projection,
        "method_release_fields_intentionally_rebound": [
            "method_source_revision",
            "method_source_archive_sha256",
            "method_source_manifest_sha256",
            "method_source_manifest_digest",
            "executed_method_entries_digest",
        ],
    }
    return {**value, "digest": object_sha256(value)}


def audit_packed_sp_sources(
    bernini_root: Path, veomni_root: Path
) -> Mapping[str, Any]:
    """Bind the packed append-pad/SP4 implementation used by this route.

    The adapter's local target selector assumes the exact Bernini sequence
    path audited here: concatenate the global source/target sequence, append
    padding to the right, then give each Ulysses rank one contiguous slice.
    A different runtime must fail before the optimizer is constructed.
    """

    transformer_source = (
        bernini_root / "bernini/models/transformer_wan.py"
    ).resolve(strict=True)
    sp_utils_source = (
        veomni_root / "veomni/distributed/sequence_parallel/utils.py"
    ).resolve(strict=True)
    sp_data_source = (
        veomni_root / "veomni/distributed/sequence_parallel/data.py"
    ).resolve(strict=True)
    sp_ulysses_source = (
        veomni_root / "veomni/distributed/sequence_parallel/ulysses.py"
    ).resolve(strict=True)
    bernini_parallel_source = (
        bernini_root / "bernini/parallel/ops.py"
    ).resolve(strict=True)
    transformer_sha = data_contract.file_sha256(transformer_source)
    sp_utils_sha = data_contract.file_sha256(sp_utils_source)
    sp_data_sha = data_contract.file_sha256(sp_data_source)
    sp_ulysses_sha = data_contract.file_sha256(sp_ulysses_source)
    bernini_parallel_sha = data_contract.file_sha256(bernini_parallel_source)
    if (
        transformer_sha != EXPECTED_TRANSFORMER_WAN_SHA256
        or sp_utils_sha != EXPECTED_VEOMNI_SP_UTILS_SHA256
        or sp_data_sha != EXPECTED_VEOMNI_SP_DATA_SHA256
        or sp_ulysses_sha != EXPECTED_VEOMNI_SP_ULYSSES_SHA256
        or bernini_parallel_sha != EXPECTED_BERNINI_PARALLEL_OPS_SHA256
    ):
        fail("Bernini packed/SP source bytes differ from the audited route")
    value = {
        "bernini_transformer_source": str(transformer_source),
        "bernini_transformer_source_sha256": transformer_sha,
        "veomni_sequence_parallel_utils_source": str(sp_utils_source),
        "veomni_sequence_parallel_utils_sha256": sp_utils_sha,
        "veomni_sequence_parallel_data_source": str(sp_data_source),
        "veomni_sequence_parallel_data_sha256": sp_data_sha,
        "veomni_sequence_parallel_ulysses_source": str(sp_ulysses_source),
        "veomni_sequence_parallel_ulysses_sha256": sp_ulysses_sha,
        "bernini_parallel_ops_source": str(bernini_parallel_source),
        "bernini_parallel_ops_sha256": bernini_parallel_sha,
        "global_layout": "clean_source_condition_prefix_then_noisy_target_suffix",
        "padding": "append_right_to_sp4_multiple",
        "sp_partition": "contiguous_equal_length_slice_after_append_padding",
        "adapter_selector": "same_global_suffix_mask_then_same_contiguous_sp4_slice",
        "padding_rows_written_by_adapter": False,
        "native_packed_attention_changed": False,
    }
    return {**value, "digest": object_sha256(value)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-only-manifest", required=True)
    parser.add_argument("--expected-source-only-manifest-sha256", required=True)
    parser.add_argument("--stage-a-admission")
    parser.add_argument("--expected-stage-a-admission-sha256")
    parser.add_argument("--formal-pair-admission")
    parser.add_argument("--expected-formal-pair-admission-sha256")
    parser.add_argument("--expected-initial-parameter-digest")
    parser.add_argument("--preflight-pair-receipt")
    parser.add_argument("--expected-preflight-pair-receipt-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-output-root", required=True)
    parser.add_argument("--mode", choices=(MODE,), required=True)
    parser.add_argument(
        "--execution-scope",
        choices=(
            "formal-exact80",
            "structural-parity-preflight",
            "backward-feasibility-preflight",
        ),
        required=True,
    )
    parser.add_argument("--parallel-topology", choices=(TOPOLOGY,), required=True)
    parser.add_argument(
        "--memory-input-kind", choices=contract.MEMORY_INPUT_KINDS, required=True
    )
    parser.add_argument("--optimizer-steps", type=int, choices=(80,), default=80)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256", required=True
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--method-source-manifest", required=True)
    parser.add_argument("--method-source-manifest-sha256", required=True)
    parser.add_argument("--ack-upstream-training-use-forbidden", action="store_true")
    parser.add_argument("--ack-user-authorized-exploratory-training", action="store_true")
    parser.add_argument("--num-frames", type=int, choices=(81,), default=81)
    return parser


def validate_cli_and_admission(
    args: argparse.Namespace,
) -> tuple[Optional[contract.StageAAdmission], Mapping[str, Any]]:
    """Model-free validation called before any optimizer can exist."""

    if (
        args.mode != MODE
        or args.parallel_topology != TOPOLOGY
        or args.optimizer_steps != contract.OPTIMIZER_STEPS
        or args.num_frames != FRAME_COUNT
    ):
        fail("Stage-B requires exact81 WORLD8 DP2xSP4 continuous exact80")
    if (
        args.ack_upstream_training_use_forbidden is not True
        or args.ack_user_authorized_exploratory_training is not True
    ):
        fail("both exploratory full644 acknowledgements are mandatory")
    if args.learning_rate != DEFAULT_LEARNING_RATE:
        fail("Stage-B v1 learning rate is fixed at 1e-4")
    if args.max_grad_norm != DEFAULT_MAX_GRAD_NORM:
        fail("Stage-B v1 max grad norm is fixed at 1.0")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        fail("seed must lie in [0,2^63)")
    contract.validate_memory_input_kind(args.memory_input_kind)
    contract.exact80_coordinates()
    contract.sample_coverage_receipt()
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        _sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_source_only_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "method_source_archive_sha256",
        "method_source_manifest_sha256",
    ):
        _sha(getattr(args, name), length=64, label=name)
    if (
        args.expected_bernini_commit != contract.EXPECTED_BERNINI_COMMIT
        or args.expected_veomni_commit != EXPECTED_VEOMNI_COMMIT
        or args.expected_checkpoint_tree_sha256
        != contract.EXPECTED_CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != EXPECTED_CHECKPOINT_MANIFEST_SHA256
    ):
        fail("pinned Bernini/VeOmni/checkpoint identity differs")
    output = Path(args.output).expanduser()
    checkpoints = Path(args.checkpoint_output_root).expanduser()
    if (
        not output.is_absolute()
        or not checkpoints.is_absolute()
        or checkpoints.name != "checkpoints"
        or checkpoints.parent != output.parent
        or output.suffix
    ):
        fail("checkpoint root must be the output sibling named checkpoints")
    if checkpoints.exists() or checkpoints.is_symlink():
        fail("checkpoint root must be fresh before any optimizer is constructed")
    method_release_identity = validate_executed_method_release(
        method_root=METHOD_ROOT,
        archive_path=args.method_source_archive,
        archive_sha256=args.method_source_archive_sha256,
        manifest_path=args.method_source_manifest,
        manifest_sha256=args.method_source_manifest_sha256,
        method_revision=args.method_source_revision,
    )
    if args.execution_scope == "structural-parity-preflight":
        if (
            args.stage_a_admission is not None
            or args.expected_stage_a_admission_sha256 is not None
            or args.formal_pair_admission is not None
            or args.expected_formal_pair_admission_sha256 is not None
            or args.expected_initial_parameter_digest is not None
            or args.preflight_pair_receipt is not None
            or args.expected_preflight_pair_receipt_sha256 is not None
        ):
            fail("structural preflight must not consume Stage-A/pair admission")
        return None, method_release_identity
    if args.execution_scope == "backward-feasibility-preflight":
        if (
            args.stage_a_admission is not None
            or args.expected_stage_a_admission_sha256 is not None
            or args.formal_pair_admission is not None
            or args.expected_formal_pair_admission_sha256 is not None
            or args.expected_initial_parameter_digest is not None
            or type(args.preflight_pair_receipt) is not str
            or type(args.expected_preflight_pair_receipt_sha256) is not str
        ):
            fail(
                "backward feasibility requires only the prior structural pair receipt"
            )
        _sha(
            args.expected_preflight_pair_receipt_sha256,
            length=64,
            label="expected_preflight_pair_receipt_sha256",
        )
        _validated_source_artifact(
            args.preflight_pair_receipt,
            expected_sha256=args.expected_preflight_pair_receipt_sha256,
            label="structural pair receipt",
        )
        return None, method_release_identity
    if (
        args.preflight_pair_receipt is not None
        or args.expected_preflight_pair_receipt_sha256 is not None
    ):
        fail("formal exact80 must consume its formal admission, not smoke inputs")
    if (
        type(args.stage_a_admission) is not str
        or type(args.expected_stage_a_admission_sha256) is not str
    ):
        fail("formal exact80 requires a Stage-A admission path and SHA")
    _sha(
        args.expected_stage_a_admission_sha256,
        length=64,
        label="expected_stage_a_admission_sha256",
    )
    if (
        type(args.formal_pair_admission) is not str
        or type(args.expected_formal_pair_admission_sha256) is not str
    ):
        fail("formal exact80 requires a shared formal pair admission and SHA")
    _sha(
        args.expected_formal_pair_admission_sha256,
        length=64,
        label="expected_formal_pair_admission_sha256",
    )
    if type(args.expected_initial_parameter_digest) is not str:
        fail("formal exact80 requires the jointly preflighted initial digest")
    _sha(
        args.expected_initial_parameter_digest,
        length=64,
        label="expected_initial_parameter_digest",
    )
    _validated_source_artifact(
        args.formal_pair_admission,
        expected_sha256=args.expected_formal_pair_admission_sha256,
        label="formal pair admission",
    )
    # This call is intentionally before torch.optim.AdamW in main().  Only the
    # structural-parity preflight above may proceed without admission, and it
    # returns before optimizer construction/backward/checkpoint publication.
    try:
        return (
            contract.load_stage_a_admission(
                args.stage_a_admission,
                expected_sha256=args.expected_stage_a_admission_sha256,
            ),
            method_release_identity,
        )
    except contract.CleanSourceVisualStageBContractError as error:
        raise CleanSourceVisualStageBTrainingError(str(error)) from error


def _publish_output(
    *,
    runtime: Any,
    output: Path,
    stage: Path,
    receipt: Optional[Mapping[str, Any]],
    rank: int,
    world_group: Any,
    rank_zero_error: Optional[str],
) -> None:
    import torch.distributed as dist

    result: list[Any] = [None]
    if rank == 0:
        try:
            if rank_zero_error is not None:
                fail(rank_zero_error)
            if receipt is None:
                fail("rank-zero receipt is absent")
            runtime.verify_staged_run_bundle(stage, receipt)
            runtime.fsync_directory(stage)
            os.rename(stage, output)
            runtime.fsync_directory(output.parent)
            runtime.verify_staged_run_bundle(output, receipt)
            result[0] = {
                "ok": True,
                "output": str(output),
                "receipt_digest": receipt.get("receipt_digest"),
            }
        except Exception as error:
            result[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(result, src=0, group=world_group)
    if not isinstance(result[0], Mapping) or result[0].get("ok") is not True:
        fail(f"cannot publish Stage-B output: {result[0]!r}")


def _publish_preflight_output(
    *,
    runtime: Any,
    output: Path,
    stage: Path,
    receipt: Optional[Mapping[str, Any]],
    rank: int,
    world_group: Any,
) -> None:
    """Atomically publish the receipt-only, no-optimizer preflight bundle."""

    import torch.distributed as dist

    result: list[Any] = [None]
    if rank == 0:
        try:
            if receipt is None:
                fail("rank-zero structural preflight receipt is absent")
            runtime.atomic_json(stage / "receipt.json", receipt)
            entries = list(stage.iterdir())
            if (
                len(entries) != 1
                or entries[0].name != "receipt.json"
                or not entries[0].is_file()
                or entries[0].is_symlink()
            ):
                fail("structural preflight output must contain receipt.json only")
            raw = entries[0].read_bytes()
            if raw != canonical_json_bytes(receipt) + b"\n":
                fail("structural preflight receipt bytes differ")
            unsigned = dict(receipt)
            declared = unsigned.pop("receipt_digest", None)
            if declared != object_sha256(unsigned):
                fail("structural preflight embedded digest differs")
            runtime.fsync_directory(stage)
            os.rename(stage, output)
            runtime.fsync_directory(output.parent)
            if (
                list(path.name for path in output.iterdir()) != ["receipt.json"]
                or (output / "receipt.json").read_bytes() != raw
            ):
                fail("published structural preflight bundle differs")
            result[0] = {
                "ok": True,
                "output": str(output),
                "receipt_digest": declared,
            }
        except Exception as error:
            result[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(result, src=0, group=world_group)
    if not isinstance(result[0], Mapping) or result[0].get("ok") is not True:
        fail(f"cannot publish structural preflight output: {result[0]!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import source_self_runtime as runtime
    import train_lora as legacy

    args = build_parser().parse_args(argv)
    admission, method_release_identity = validate_cli_and_admission(args)
    manifest_path = Path(args.source_only_manifest).expanduser()
    if data_contract.file_sha256(manifest_path) != args.expected_source_only_manifest_sha256:
        fail("source-only split manifest file SHA-256 differs")
    manifest = data_contract.load_source_only_split_manifest(
        manifest_path, verify_files=True
    )
    dataset_authorization = data_contract.authorize_exploratory_training(
        manifest,
        ack_upstream_training_use_forbidden=(
            args.ack_upstream_training_use_forbidden
        ),
        ack_user_authorized_exploratory_training=(
            args.ack_user_authorized_exploratory_training
        ),
    )
    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    if (
        data_contract.file_sha256(checkpoint_manifest)
        != args.expected_checkpoint_content_manifest_sha256
    ):
        fail("checkpoint content manifest SHA-256 differs")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise CleanSourceVisualStageBTrainingError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        fail("pinned Bernini attention-head count differs")
    packed_sp_audit = audit_packed_sp_sources(bernini_root, veomni_root)
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import UniPCMultistepScheduler, __version__ as diffusers_version
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state

    topology = runtime.parallel_topology(TOPOLOGY)
    distributed = runtime.distributed_contract(topology=topology)
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.local_world_size != WORLD_SIZE
        or distributed.rank != distributed.local_rank
    ):
        fail("Stage-B requires one-node WORLD8 DP2xSP4")
    device = runtime.initialise_distributed(distributed)
    checkpoint_content_box: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_content_box[0] = {
                "ok": True,
                "identity": validate_checkpoint_content(
                    checkpoint,
                    checkpoint_manifest,
                    expected_manifest_sha256=(
                        args.expected_checkpoint_content_manifest_sha256
                    ),
                ),
            }
        except Exception as error:
            checkpoint_content_box[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_content_box, src=0)
    checkpoint_content_result = checkpoint_content_box[0]
    if (
        not isinstance(checkpoint_content_result, Mapping)
        or checkpoint_content_result.get("ok") is not True
        or not isinstance(checkpoint_content_result.get("identity"), Mapping)
    ):
        fail(
            "base checkpoint content verification failed: "
            f"{checkpoint_content_result!r}"
        )
    checkpoint_content_identity = dict(checkpoint_content_result["identity"])
    # Measure the complete model-load plus actual packed-route parity path.
    # The structural scope reports this per WORLD8 rank and exits before an
    # optimizer, backward call, or checkpoint root can exist.
    torch.cuda.reset_peak_memory_stats(device)
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=SP_SIZE)
    )
    legacy.seed_same_sample(args.seed)
    renderer_config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    renderer_config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(renderer_config.to_dict(), checkpoint)
    renderer = None
    for loading_rank in range(WORLD_SIZE):
        if distributed.rank == loading_rank:
            renderer = BerniniRendererModel(renderer_config)
            renderer.requires_grad_(False)
            renderer.eval()
            renderer.t5_text_encoder.eval()
            renderer.to(device)
        dist.barrier(group=parallel.world_group)
    if renderer is None:
        fail("rank-serialized renderer load did not materialize every rank")
    transformer = renderer.diff_dec.transformer
    if transformer is None or renderer.diff_dec.transformer_2 is not None:
        fail("Stage-B requires only frozen Bernini transformer_1")
    renderer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": visual.checkpoint_route_context_fn,
        }
    )
    if not bool(getattr(transformer, "gradient_checkpointing", False)):
        fail("Stage-B requires non-reentrant gradient checkpointing")
    handle = visual.install_clean_source_visual_context_adapter_v1(
        transformer,
        runtime_source_commit=bernini_revision,
        model_revision=visual.PINNED_BERNINI_MODEL_REVISION,
        checkpoint_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        block_indices=(
            admission.installed_sparse_block_indices
            if admission is not None
            else visual.DEFAULT_BLOCK_INDICES
        ),
    )
    trainable = handle.trainable_named_parameters()
    if not handle.base_parameters_frozen() or not handle.native_structure_untouched():
        fail("adapter install changed frozen native Bernini structure")
    initial_parameter_digest = stage_b_synchronize_initial_parameters(
        trainable, parallel.world_group
    )
    pair_invariants = pair_invariants_receipt(
        args=args,
        manifest=manifest,
        method_release_identity=method_release_identity,
        checkpoint_content_identity=checkpoint_content_identity,
        initial_parameter_digest=initial_parameter_digest,
        stage_a_admission_digest=(
            admission.receipt_digest if admission is not None else None
        ),
    )
    backward_upstream_pair: Optional[Mapping[str, Any]] = None
    backward_upstream_gate: Optional[Mapping[str, Any]] = None
    if args.execution_scope == "backward-feasibility-preflight":
        backward_upstream_pair = pair_contract.load_canonical_receipt(
            args.preflight_pair_receipt,
            expected_file_sha256=args.expected_preflight_pair_receipt_sha256,
            expected_schema=pair_contract.PREFLIGHT_PAIR_SCHEMA,
        )
        upstream_shared = pair_contract.validate_preflight_pair_receipt(
            backward_upstream_pair
        )
        backward_upstream_gate = validate_backward_upstream_pair_invariants(
            current=pair_invariants,
            upstream=upstream_shared,
        )
        if upstream_shared.get("initial_parameter_digest") != initial_parameter_digest:
            fail("backward feasibility initialization differs from paired preflight")
    formal_pair_admission: Optional[Mapping[str, Any]] = None
    if admission is not None:
        formal_pair_admission = pair_contract.load_formal_pair_admission(
            args.formal_pair_admission,
            expected_file_sha256=args.expected_formal_pair_admission_sha256,
            memory_input_kind=args.memory_input_kind,
            expected_shared_invariants_without_initial=pair_invariants,
        )
        if (
            formal_pair_admission.get("expected_initial_parameter_digest")
            != initial_parameter_digest
            or args.expected_initial_parameter_digest != initial_parameter_digest
            or formal_pair_admission.get("shared_pair_invariants")
            != pair_invariants
            or formal_pair_admission.get("stage_a_admission")
            != admission.receipt()
        ):
            fail(
                "actual adapter initialization differs from the two-arm "
                "preflight/formal pair admission"
            )

    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(checkpoint),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=5.0,
    )
    schedule_audit = __import__("inference_sigma_strata").audit_runtime_unipc_schedule(
        scheduler
    )
    if schedule_audit.get("schedule_sha256") != contract.EXPECTED_SCHEDULE_SHA256:
        fail("runtime UniPC schedule differs")
    del scheduler

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    tokenized = runtime.tokenize_generic_instruction(
        tokenizer, EXACT_NOOP_INSTRUCTION, device
    )
    with torch.inference_mode():
        text_lens, text_embs = renderer.get_t5_text_embeddings(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            tokenized["t5_input_lens"],
        )
    if getattr(text_embs, "requires_grad", False):
        fail("frozen T5 embeddings require gradients")
    renderer.t5_text_encoder = None
    del tokenizer, tokenized
    gc.collect()
    torch.cuda.empty_cache()
    if renderer.t5_text_encoder is not None:
        fail("frozen T5 encoder was not released")

    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    if z_dim != 16:
        fail("Wan VAE z dimension differs")
    store = data_contract.PinnedPhysicalSourceOnlyPosteriorStore(
        manifest,
        vae_latents_mean=vae_mean.unsqueeze(0).float().contiguous(),
        vae_latents_std=vae_std.unsqueeze(0).float().contiguous(),
        verify_files_on_first_access=True,
    )
    train_rows = manifest.rows_for_split("train")
    if len(train_rows) != 64:
        fail("source-only train split differs from 64")
    manifest_index_by_iid = {
        row.iid: index for index, row in enumerate(manifest.rows)
    }
    train_manifest_indices = tuple(
        manifest_index_by_iid[row.iid] for row in train_rows
    )
    source_only_preload = store.preload(train_manifest_indices)
    if (
        source_only_preload.get("preloaded_rows") != 64
        or source_only_preload.get("legacy_parquet_opened") is not False
        or source_only_preload.get("synthetic_target_index1_bytes_read") is not False
    ):
        fail("physical source-only preload closure differs")
    runtime.digest_consensus(
        str(source_only_preload["digest"]),
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="physical source-only preload",
    )

    # Before an optimizer or immutable checkpoint exists, exercise the actual
    # Bernini packed/SP path twice on one registered microbatch per DP arm:
    # once with the adapter route disabled and once with source memory enabled.
    # The zero output projections require bit-exact prediction parity.
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    parity_coordinate = contract.coordinates_for_optimizer_step(0)[0]
    parity_row_position = contract.train_row_position(
        optimizer_step_zero_based=0,
        microbatch_index=parity_coordinate.microbatch_index,
        dp_arm=distributed.arm_index,
    )
    parity_row = train_rows[parity_row_position]
    parity_manifest_index = manifest_index_by_iid[parity_row.iid]
    parity_sample = store.load(parity_manifest_index)
    parity_seed = _noise_seed(
        args.seed,
        0,
        parity_coordinate.microbatch_index,
        distributed.arm_index,
    )
    parity_generator = torch.Generator(device="cpu")
    parity_generator.manual_seed(parity_seed)
    parity_epsilon = torch.randn(
        tuple(parity_sample.clean_noop_target.shape),
        generator=parity_generator,
        dtype=torch.float32,
    ).contiguous()
    parity_packed = prepare_noop_condition(
        clean_source=parity_sample.clean_noop_target,
        epsilon=parity_epsilon,
        coordinate=parity_coordinate,
        memory_input_kind=args.memory_input_kind,
        rope=rope,
        device=device,
        runtime=runtime,
    )
    zero_output_names = [
        name for name, _ in trainable if name.endswith(".output.weight")
    ]
    if (
        len(zero_output_names)
        != len(
            admission.installed_sparse_block_indices
            if admission is not None
            else visual.DEFAULT_BLOCK_INDICES
        )
        or any(
            bool(torch.count_nonzero(parameter.detach()).item())
            for name, parameter in trainable
            if name.endswith(".output.weight")
        )
    ):
        fail("step-0 adapter output projections are not exactly zero")
    with torch.no_grad():
        parity_memory = handle.build_memory(
            parity_packed.memory_input.to(
                device=device, dtype=torch.float32
            ).detach().contiguous(),
            source_video_sha256=parity_sample.source_video_sha256,
            memory_input_latent_sha256=parity_packed.tensor_identities[
                "visual_context_input"
            ],
            input_kind=args.memory_input_kind,
        )
        parity_disabled_route = visual.VisualContextRoute(
            parity_packed.total_tokens,
            parity_packed.condition_tokens,
            distributed.sp_rank,
            SP_SIZE,
            None,
            enabled=False,
        )
        parity_enabled_route = visual.VisualContextRoute(
            parity_packed.total_tokens,
            parity_packed.condition_tokens,
            distributed.sp_rank,
            SP_SIZE,
            parity_memory,
        )
        with handle.route(parity_disabled_route):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                parity_base_prediction = _prediction(
                    renderer=renderer,
                    transformer=transformer,
                    condition=parity_packed,
                    coordinate=parity_coordinate,
                    text_lens=text_lens,
                    text_embs=text_embs,
                )
        with handle.route(parity_enabled_route):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                parity_adapter_prediction = _prediction(
                    renderer=renderer,
                    transformer=transformer,
                    condition=parity_packed,
                    coordinate=parity_coordinate,
                    text_lens=text_lens,
                    text_embs=text_embs,
                )
    local_parity_equal = torch.equal(
        parity_base_prediction, parity_adapter_prediction
    )
    if not runtime.world_all_true(
        local_parity_equal, group=parallel.world_group
    ):
        fail("step-0 zero-init adapter lost bit-exact frozen-base parity")
    local_parity_record = {
        "world_rank": distributed.rank,
        "dp_arm": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "iid": parity_sample.iid,
        "row_position": parity_row_position,
        "manifest_index": parity_manifest_index,
        "noise_seed": parity_seed,
        **parity_coordinate.receipt(),
        "memory_input_kind": args.memory_input_kind,
        "input_patch_shape": list(parity_packed.input_patches.shape),
        "prediction_shape": list(parity_base_prediction.shape),
        "disabled_route_receipt": dict(parity_disabled_route.receipt()),
        "enabled_route_receipt": dict(parity_enabled_route.receipt()),
        "local_target_selector_count": int(
            parity_enabled_route.local_target_selector(device=device).sum().item()
        ),
        "disabled_route_prediction_sha256": runtime.tensor_sha256(
            parity_base_prediction.detach().cpu().contiguous()
        ),
        "enabled_zero_init_route_prediction_sha256": runtime.tensor_sha256(
            parity_adapter_prediction.detach().cpu().contiguous()
        ),
        "bit_exact_equal": True,
        "zero_output_projection_names": zero_output_names,
        "zero_output_projections_exact": True,
        "optimizer_constructed": False,
        "checkpoint_written": False,
    }
    parity_gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        parity_gathered, local_parity_record, group=parallel.world_group
    )
    parity_world8_gate = validate_step0_exact_base_parity_world8(parity_gathered)
    step0_exact_base_parity_unsigned = {
        "route_comparison": "disabled_base_vs_enabled_zero_init_adapter",
        "actual_bernini_packed_sp_path_executed": True,
        "bit_exact_all_world8_ranks": True,
        "all_registered_output_projections_exactly_zero": True,
        "optimizer_constructed_during_check": False,
        "checkpoint_written_during_check": False,
        "world8_rank_structural_gate": parity_world8_gate,
        "dp_arm_records": parity_world8_gate["dp_arm_records"],
        "per_rank_base_vs_zero_init_sha_equality_required": True,
        "cross_sp_prediction_sha_equality_required": False,
    }
    step0_exact_base_parity = {
        **step0_exact_base_parity_unsigned,
        "digest": object_sha256(step0_exact_base_parity_unsigned),
    }
    del (
        parity_sample,
        parity_epsilon,
        parity_packed,
        parity_memory,
        parity_disabled_route,
        parity_enabled_route,
        parity_base_prediction,
        parity_adapter_prediction,
        local_parity_record,
        parity_gathered,
        parity_world8_gate,
    )
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    gib = float(1024**3)
    local_cuda_memory = {
        "rank": distributed.rank,
        "dp_arm": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "device_name": torch.cuda.get_device_name(device),
        "allocated_gib_after_parity": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib_after_parity": torch.cuda.memory_reserved(device) / gib,
        "peak_allocated_gib_model_load_through_parity": (
            torch.cuda.max_memory_allocated(device) / gib
        ),
        "peak_reserved_gib_model_load_through_parity": (
            torch.cuda.max_memory_reserved(device) / gib
        ),
        "host_process_memory": linux_host_memory_receipt(),
    }
    cuda_memory_world: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        cuda_memory_world, local_cuda_memory, group=parallel.world_group
    )
    if [record.get("rank") for record in cuda_memory_world] != list(
        range(WORLD_SIZE)
    ):
        fail("WORLD8 structural preflight CUDA-memory closure differs")

    if args.execution_scope == "structural-parity-preflight":
        if admission is not None:
            fail("structural preflight unexpectedly consumed Stage-A admission")
        checkpoint_root = Path(args.checkpoint_output_root).expanduser()
        if checkpoint_root.exists() or checkpoint_root.is_symlink():
            fail("structural preflight must not create the checkpoint root")
        output, stage = runtime.prepare_output_transaction(
            args.output, distributed.rank, parallel.world_group
        )
        receipt: Optional[Mapping[str, Any]] = None
        if distributed.rank == 0:
            unsigned = {
                "schema_version": PREFLIGHT_RECEIPT_SCHEMA,
                "method": METHOD,
                "complete": True,
                "execution_scope": "structural-parity-preflight",
                "mode": MODE,
                "memory_input_kind": args.memory_input_kind,
                "pair_invariants": pair_invariants,
                "stage_a": {
                    "admission_required_for_this_scope": False,
                    "admission_consumed": False,
                    "optimizer_authorization_granted": False,
                    "formal_exact80_still_requires_verified_admission": True,
                    "installed_blocks_are_unadmitted_structural_defaults": list(
                        visual.DEFAULT_BLOCK_INDICES
                    ),
                    "causal_localization_claimed": False,
                },
                "authority": {
                    "gpu_runtime_executed": True,
                    "world8_model_loaded": True,
                    "optimizer_constructed": False,
                    "backward_executed": False,
                    "optimizer_step_count": 0,
                    "training_executed": False,
                    "checkpoint_written": False,
                    "checkpoint_root_created": False,
                    "decoded_inference_executed": False,
                    "html_review_generated": False,
                    "quality_claimed": False,
                    "scientific_success_claimed": False,
                },
                "step0_exact_base_parity": step0_exact_base_parity,
                "cuda_memory_world8": cuda_memory_world,
                "dataset": {
                    **manifest.receipt(),
                    "manifest_path": str(manifest_path),
                    "manifest_file_sha256": (
                        args.expected_source_only_manifest_sha256
                    ),
                    "physical_index0_train_rows_preloaded": 64,
                    "legacy_parquet_opened_by_stage_b": False,
                    "synthetic_target_index1_bytes_read_by_stage_b": False,
                },
                "objective": {
                    "optimizer_objective_executed": False,
                    "reward_used": False,
                    "scalar_evaluator_used": False,
                    "synthetic_target_accessed": False,
                },
                "adapter": {
                    "architecture": dict(handle.receipt()),
                    "runtime_memory_input_kind": args.memory_input_kind,
                    "native_structure_untouched": True,
                    "base_parameters_frozen": True,
                    "initial_parameter_digest": initial_parameter_digest,
                },
                "distributed": {
                    "profile": TOPOLOGY,
                    "world_size": WORLD_SIZE,
                    "physical_dp_size": DP_SIZE,
                    "ulysses_sp_size": SP_SIZE,
                    "actual_bernini_packed_sp_path_executed": True,
                    "packed_sp_source_audit": packed_sp_audit,
                },
                "schedule_loaded_but_not_trained": schedule_audit,
                "source_only_preload": dict(source_only_preload),
                "resource_estimate_for_formal_arm": resource_budget_receipt(
                    args.memory_input_kind
                ),
                "model": {
                    "bernini_commit": bernini_revision,
                    "veomni_commit": veomni_revision,
                    "model_revision": visual.PINNED_BERNINI_MODEL_REVISION,
                    "checkpoint_tree_sha256": (
                        args.expected_checkpoint_tree_sha256
                    ),
                    "checkpoint_content_manifest_sha256": (
                        args.expected_checkpoint_content_manifest_sha256
                    ),
                    "checkpoint_content_identity": checkpoint_content_identity,
                },
                "method_source": dict(method_release_identity),
                "runtime": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "transformers": transformers_version,
                    "diffusers": diffusers_version,
                },
            }
            receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        _publish_preflight_output(
            runtime=runtime,
            output=output,
            stage=stage,
            receipt=receipt,
            rank=distributed.rank,
            world_group=parallel.world_group,
        )
        dist.barrier(group=parallel.world_group)
        if checkpoint_root.exists() or checkpoint_root.is_symlink():
            fail("structural preflight created a forbidden checkpoint root")
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "arm": args.memory_input_kind,
                        "execution_scope": "structural-parity-preflight",
                        "optimizer_constructed": False,
                        "backward_executed": False,
                        "checkpoint_written": False,
                        "parent_allocation_released": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.destroy_process_group()
        return 0

    if args.execution_scope == "backward-feasibility-preflight":
        if (
            admission is not None
            or formal_pair_admission is not None
            or backward_upstream_pair is None
            or backward_upstream_gate is None
        ):
            fail("backward feasibility admission boundary differs")
        checkpoint_root = Path(args.checkpoint_output_root).expanduser()
        if checkpoint_root.exists() or checkpoint_root.is_symlink():
            fail("backward feasibility must not create the checkpoint root")
        backward_receipt = run_backward_feasibility_microbatches(
            args=args,
            runtime=runtime,
            renderer=renderer,
            transformer=transformer,
            handle=handle,
            trainable=trainable,
            parallel=parallel,
            distributed=distributed,
            device=device,
            rope=rope,
            text_lens=text_lens,
            text_embs=text_embs,
            train_rows=train_rows,
            manifest_index_by_iid=manifest_index_by_iid,
            store=store,
            initial_parameter_digest=initial_parameter_digest,
        )
        output, stage = runtime.prepare_output_transaction(
            args.output, distributed.rank, parallel.world_group
        )
        receipt: Optional[Mapping[str, Any]] = None
        if distributed.rank == 0:
            unsigned = {
                "schema_version": BACKWARD_PREFLIGHT_RECEIPT_SCHEMA,
                "method": METHOD,
                "complete": True,
                "execution_scope": "backward-feasibility-preflight",
                "mode": MODE,
                "memory_input_kind": args.memory_input_kind,
                "pair_invariants": pair_invariants,
                "upstream_structural_pair": {
                    "receipt_path": str(
                        Path(args.preflight_pair_receipt).resolve(strict=True)
                    ),
                    "receipt_file_sha256": (
                        args.expected_preflight_pair_receipt_sha256
                    ),
                    "receipt_digest": backward_upstream_pair["receipt_digest"],
                    "invariant_gate": backward_upstream_gate,
                },
                "authority": {
                    "gpu_runtime_executed": True,
                    "world8_model_loaded": True,
                    "four_microbatch_forward_executed": True,
                    "four_microbatch_loss_executed": True,
                    "four_microbatch_backward_executed": True,
                    "dp2_sp4_gradient_sync_executed": True,
                    "optimizer_constructed": False,
                    "optimizer_step_count": 0,
                    "parameters_changed": False,
                    "checkpoint_written": False,
                    "checkpoint_root_created": False,
                    "training_executed": False,
                    "formal_optimizer_authorized": False,
                    "scientific_success_claimed": False,
                },
                "backward_feasibility": backward_receipt,
                "step0_exact_base_parity": step0_exact_base_parity,
                "model_load_through_parity_resources_world8": cuda_memory_world,
                "dataset": {
                    **manifest.receipt(),
                    "manifest_path": str(manifest_path),
                    "manifest_file_sha256": (
                        args.expected_source_only_manifest_sha256
                    ),
                    "physical_index0_train_rows_preloaded": 64,
                    "legacy_parquet_opened_by_stage_b": False,
                    "synthetic_target_index1_bytes_read_by_stage_b": False,
                },
                "objective": {
                    "objective_executed_for_backward_only": True,
                    "name": "standard_target_only_noop_flow_matching",
                    "reward_used": False,
                    "scalar_evaluator_used": False,
                    "synthetic_target_accessed": False,
                },
                "adapter": {
                    "architecture": dict(handle.receipt()),
                    "runtime_memory_input_kind": args.memory_input_kind,
                    "base_parameters_frozen": True,
                    "initial_parameter_digest": initial_parameter_digest,
                },
                "distributed": {
                    "profile": TOPOLOGY,
                    "world_size": WORLD_SIZE,
                    "physical_dp_size": DP_SIZE,
                    "ulysses_sp_size": SP_SIZE,
                    "gradient_sync": "SP4_mean_then_DP2_mean",
                    "actual_bernini_packed_sp_path_executed": True,
                    "packed_sp_source_audit": packed_sp_audit,
                },
                "method_source": dict(method_release_identity),
                "parent_allocation_released": False,
            }
            receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        _publish_preflight_output(
            runtime=runtime,
            output=output,
            stage=stage,
            receipt=receipt,
            rank=distributed.rank,
            world_group=parallel.world_group,
        )
        dist.barrier(group=parallel.world_group)
        if checkpoint_root.exists() or checkpoint_root.is_symlink():
            fail("backward feasibility created a forbidden checkpoint root")
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "execution_scope": "backward-feasibility-preflight",
                        "microbatches_per_dp_arm": 4,
                        "optimizer_constructed": False,
                        "parameters_changed": False,
                        "checkpoint_written": False,
                        "parent_allocation_released": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.destroy_process_group()
        return 0

    if admission is None:
        fail("formal exact80 reached optimizer boundary without Stage-A admission")
    if formal_pair_admission is None:
        fail("formal exact80 reached optimizer boundary without pair admission")

    # Every model, schedule, text and physical index-0 source preflight above
    # must pass before the output transaction, optimizer or step-0 checkpoint
    # exists.  A failed preflight therefore cannot poison immutable cadence.
    output, stage = runtime.prepare_output_transaction(
        args.output, distributed.rank, parallel.world_group
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    checkpoint_authorization = {
        "dataset_authorization": dict(dataset_authorization),
        "stage_a_admission_digest": admission.receipt_digest,
        "formal_pair_admission_digest": formal_pair_admission[
            "receipt_digest"
        ],
        "stage_a_runtime_receipt_digests": list(
            admission.runtime_receipt_digests
        ),
        "stage_a_passed_block_bands": list(admission.passed_block_bands),
        "installed_preregistered_sparse_block_indices": list(
            admission.installed_sparse_block_indices
        ),
        "per_block_causal_localization_claimed": False,
        "memory_input_kind": args.memory_input_kind,
        "optimizer_authorized": True,
        "gradient_accumulation_steps": contract.GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": contract.GLOBAL_BATCH,
        "logical_training_records": contract.LOGICAL_RECORDS,
        "scalar_reward_used": False,
        "synthetic_target_accessed": False,
    }
    coordinator = (
        data_contract.VisualContextCheckpointCoordinator(
            output_directory=args.checkpoint_output_root,
            handle=handle,
            optimizer=optimizer,
            manifest=manifest,
            authorization_receipt=checkpoint_authorization,
        )
        if distributed.rank == 0
        else None
    )
    checkpoint_records: list[Mapping[str, Any]] = []

    def coordinate_checkpoint(completed_step: int) -> None:
        box: list[Any] = [None]
        if distributed.rank == 0:
            assert coordinator is not None
            box[0] = (
                coordinator.start_before_first_optimizer_step()
                if completed_step == 0
                else coordinator.after_optimizer_step(completed_step)
            )
        dist.broadcast_object_list(box, src=0, group=parallel.world_group)
        record = box[0]
        if completed_step in contract.CHECKPOINT_STEPS:
            if (
                not isinstance(record, Mapping)
                or record.get("step") != completed_step
                or record.get("logical_records_seen")
                != completed_step * contract.GLOBAL_BATCH
            ):
                fail(f"checkpoint {completed_step} was not published in-loop")
            checkpoint_records.append(dict(record))
        elif record is not None:
            fail("checkpoint appeared outside exact cadence")
        dist.barrier(group=parallel.world_group)

    coordinate_checkpoint(0)

    previous_parameter_digest = initial_parameter_digest
    history_steps: list[Mapping[str, Any]] = []
    positive_gradient_steps = 0
    order_gate_receipt: Optional[Mapping[str, Any]] = None
    training_started_monotonic = time.monotonic()

    for step_zero_based in range(contract.OPTIMIZER_STEPS):
        optimizer.zero_grad(set_to_none=True)
        coordinates = contract.coordinates_for_optimizer_step(step_zero_based)
        gathered_microbatches: list[Mapping[str, Any]] = []

        # Four independent microbatches are accumulated on every DP arm.  The
        # loss is divided before backward, and SP/DP gradient synchronization
        # is deliberately deferred until all four graphs have contributed.
        for coordinate in coordinates:
            row_position = contract.train_row_position(
                optimizer_step_zero_based=step_zero_based,
                microbatch_index=coordinate.microbatch_index,
                dp_arm=distributed.arm_index,
            )
            selected_row = train_rows[row_position]
            manifest_index = manifest_index_by_iid[selected_row.iid]
            sample = store.load(manifest_index)
            if sample.split != "train" or sample.iid != selected_row.iid:
                fail("optimizer accessed a non-train source row")
            runtime.digest_consensus(
                str(sample.receipt()["digest"]),
                group=parallel.sp_group,
                expected_count=SP_SIZE,
                label=(
                    f"step {coordinate.optimizer_step} microbatch "
                    f"{coordinate.microbatch_index} DP source row"
                ),
            )
            seed = _noise_seed(
                args.seed,
                step_zero_based,
                coordinate.microbatch_index,
                distributed.arm_index,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            epsilon = torch.randn(
                tuple(sample.clean_noop_target.shape),
                generator=generator,
                dtype=torch.float32,
            ).contiguous()
            packed = prepare_noop_condition(
                clean_source=sample.clean_noop_target,
                epsilon=epsilon,
                coordinate=coordinate,
                memory_input_kind=args.memory_input_kind,
                rope=rope,
                device=device,
                runtime=runtime,
            )
            memory_input_device = packed.memory_input.to(
                device=device, dtype=torch.float32
            ).detach().contiguous()
            memory = handle.build_memory(
                memory_input_device,
                source_video_sha256=sample.source_video_sha256,
                memory_input_latent_sha256=packed.tensor_identities[
                    "visual_context_input"
                ],
                input_kind=args.memory_input_kind,
            )
            route = visual.VisualContextRoute(
                packed.total_tokens,
                packed.condition_tokens,
                distributed.sp_rank,
                SP_SIZE,
                memory,
            )
            local_selector_has_target = bool(
                route.local_target_selector(device=device).any().item()
            )
            local_order_gate = None
            if step_zero_based == 0 and coordinate.microbatch_index == 0:
                local_order_gate = source_order_structural_gate(
                    handle=handle,
                    clean_source=sample.clean_noop_target,
                    source_video_sha256=sample.source_video_sha256,
                    runtime=runtime,
                )
                runtime.digest_consensus(
                    str(local_order_gate["digest"]),
                    group=parallel.sp_group,
                    expected_count=SP_SIZE,
                    label="source-order structural gate",
                )

            with handle.route(route):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = _prediction(
                        renderer=renderer,
                        transformer=transformer,
                        condition=packed,
                        coordinate=coordinate,
                        text_lens=text_lens,
                        text_embs=text_embs,
                    )
                    prediction = _zero_dependency(prediction, trainable)
                    raw_loss = visual.no_op_flow_matching_loss(
                        prediction=prediction,
                        target_velocity=packed.target_velocity,
                    )
                    scaled_loss = raw_loss / float(
                        contract.GRADIENT_ACCUMULATION_STEPS
                    )
                if not runtime.world_all_true(
                    bool(torch.isfinite(scaled_loss.detach()).item()),
                    group=parallel.world_group,
                ):
                    fail("non-finite accumulated no-op flow loss blocked update")
                raw_loss_value = float(raw_loss.detach().item())
                scaled_loss_value = float(scaled_loss.detach().item())
                scaled_loss.backward()

            local_record = {
                "schema_version": HISTORY_SCHEMA,
                **coordinate.receipt(),
                "dp_arm": distributed.arm_index,
                "sp_rank": distributed.sp_rank,
                "row_position": row_position,
                "manifest_index": manifest_index,
                "iid": sample.iid,
                "action_family": selected_row.action_family,
                "source_video_sha256": sample.source_video_sha256,
                "noise_seed": seed,
                "memory_input_kind": args.memory_input_kind,
                "tensor_identities": dict(packed.tensor_identities),
                "memory_receipt": dict(memory.receipt()),
                "route_receipt": dict(route.receipt()),
                "local_selector_has_target": local_selector_has_target,
                "replicated_zero_dependency_added": True,
                "raw_loss": raw_loss_value,
                "loss_divisor_before_backward": (
                    contract.GRADIENT_ACCUMULATION_STEPS
                ),
                "scaled_loss_backward": scaled_loss_value,
                "objective": "mse(target_velocity_prediction,epsilon-clean_source)",
                "target_rows_only": True,
                "source_order_structural_gate": local_order_gate,
                "synthetic_target_posterior_accessed": False,
                "reward_used": False,
            }
            projection = {
                key: value for key, value in local_record.items() if key != "sp_rank"
            }
            runtime.digest_consensus(
                object_sha256(projection),
                group=parallel.sp_group,
                expected_count=SP_SIZE,
                label=(
                    f"history step {coordinate.optimizer_step} microbatch "
                    f"{coordinate.microbatch_index} DP arm"
                ),
            )
            gathered: list[Any] = [None] * WORLD_SIZE
            dist.all_gather_object(gathered, local_record, group=parallel.world_group)
            leaders = [gathered[0], gathered[4]]
            if (
                [item.get("dp_arm") for item in leaders] != [0, 1]
                or len({item.get("iid") for item in leaders}) != 2
                or any(
                    item.get("schedule_index") != coordinate.schedule_index
                    or item.get("microbatch_index") != coordinate.microbatch_index
                    for item in leaders
                )
            ):
                fail("WORLD8 DP2 microbatch history closure differs")
            gathered_microbatches.append(
                {
                    "schema_version": HISTORY_SCHEMA,
                    **coordinate.receipt(),
                    "logical_records": [
                        {
                            key: value
                            for key, value in item.items()
                            if key != "sp_rank"
                        }
                        for item in leaders
                    ],
                    "backward_executed": True,
                    "optimizer_step_executed": False,
                }
            )
            if local_order_gate is not None:
                order_gate_receipt = gathered_microbatches[-1][
                    "logical_records"
                ][0]["source_order_structural_gate"]
            del (
                sample,
                epsilon,
                packed,
                memory_input_device,
                memory,
                route,
                prediction,
                raw_loss,
                scaled_loss,
                local_record,
                projection,
                gathered,
                leaders,
            )

        if len(gathered_microbatches) != contract.GRADIENT_ACCUMULATION_STEPS:
            fail("optimizer update did not receive exactly four microbatches")
        if any(parameter.grad is None for _, parameter in trainable):
            fail("zero dependency did not materialize every replicated gradient")
        preclip_norm = runtime.synchronize_gradients(trainable, parallel)
        component_grad_norms = grouped_gradient_norms(trainable)
        if component_grad_norms["output"] <= 0.0:
            fail("visual-context output projection received no learning signal")
        if step_zero_based >= 1 and any(
            value <= 0.0 for value in component_grad_norms.values()
        ):
            fail(
                "visual-context encoder/Q/K/V/output/gate path is inactive "
                "after the zero-init first update"
            )
        clipped = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], args.max_grad_norm
        )
        if not math.isfinite(float(clipped)) or preclip_norm <= 0.0:
            fail("Stage-B synchronized gradient norm differs")
        optimizer.step()
        positive_gradient_steps += 1
        parameter_digest = stage_b_parameter_consensus(
            trainable,
            parallel.world_group,
            f"clean-source visual-context step {step_zero_based + 1}",
        )
        if parameter_digest == previous_parameter_digest:
            fail("optimizer step did not change adapter parameters")
        update_record = {
            "schema_version": HISTORY_SCHEMA,
            "optimizer_step": step_zero_based + 1,
            "gradient_accumulation_steps": (
                contract.GRADIENT_ACCUMULATION_STEPS
            ),
            "effective_global_batch": contract.GLOBAL_BATCH,
            "logical_records_in_update": contract.GLOBAL_BATCH,
            "microbatches": gathered_microbatches,
            "optimizer_step_executed": True,
            "mean_raw_logical_record_loss": sum(
                float(record["raw_loss"])
                for microbatch in gathered_microbatches
                for record in microbatch["logical_records"]
            )
            / float(contract.GLOBAL_BATCH),
            "preclip_gradient_norm_dp2_sp4_mean": preclip_norm,
            "preclip_component_gradient_norms": component_grad_norms,
            "parameter_sha256_before": previous_parameter_digest,
            "parameter_sha256_after": parameter_digest,
        }
        runtime.digest_consensus(
            object_sha256(update_record),
            group=parallel.world_group,
            expected_count=WORLD_SIZE,
            label=f"history optimizer update {step_zero_based + 1}",
        )
        history_steps.append(update_record)
        previous_parameter_digest = parameter_digest
        if distributed.rank == 0:
            progress = {
                "schema_version": (
                    "bernini-clean-source-visual-context-stage-b-progress-v1"
                ),
                "global_step": step_zero_based + 1,
                "logical_records_seen": (
                    (step_zero_based + 1) * contract.GLOBAL_BATCH
                ),
                "microbatch_mean_raw_losses": [
                    sum(
                        float(record["raw_loss"])
                        for record in microbatch["logical_records"]
                    )
                    / float(DP_SIZE)
                    for microbatch in gathered_microbatches
                ],
                "mean_raw_logical_record_loss": update_record[
                    "mean_raw_logical_record_loss"
                ],
                "preclip_gradient_norm_dp2_sp4_mean": preclip_norm,
                "preclip_component_gradient_norms": component_grad_norms,
                "parameter_sha256_after": parameter_digest,
                "cuda_memory_allocated_gib": (
                    torch.cuda.memory_allocated(device) / float(1024**3)
                ),
                "cuda_memory_reserved_gib": (
                    torch.cuda.memory_reserved(device) / float(1024**3)
                ),
                "cuda_peak_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / float(1024**3)
                ),
                "timestamp_unix_seconds": time.time(),
                "elapsed_seconds": time.monotonic() - training_started_monotonic,
            }
            print(json.dumps(progress, sort_keys=True), flush=True)
        coordinate_checkpoint(step_zero_based + 1)
        del gathered_microbatches, update_record
        gc.collect()
        torch.cuda.empty_cache()

    if (
        len(history_steps) != 80
        or positive_gradient_steps != 80
        or sum(
            len(microbatch["logical_records"])
            for update in history_steps
            for microbatch in update["microbatches"]
        )
        != contract.LOGICAL_RECORDS
        or [item.get("step") for item in checkpoint_records]
        != list(contract.CHECKPOINT_STEPS)
        or order_gate_receipt is None
    ):
        fail("continuous exact80 history/checkpoint closure differs")
    coordinator_receipt: list[Any] = [None]
    if distributed.rank == 0:
        assert coordinator is not None
        coordinator_receipt[0] = coordinator.finalize()
    dist.broadcast_object_list(
        coordinator_receipt, src=0, group=parallel.world_group
    )
    if not isinstance(coordinator_receipt[0], Mapping):
        fail("checkpoint coordinator did not finalize")
    decode_chain = contract.checkpoint_decode_chain(
        checkpoint_records,
        manifest_digest=manifest.manifest_digest,
        admission_digest=admission.receipt_digest,
        memory_input_kind=args.memory_input_kind,
    )
    # Reopen and strict-load every immutable cadence checkpoint in decode
    # order.  This is a real loader exercise, not just a metadata chain.  The
    # final step-80 load must restore the exact live final adapter state.
    final_live_parameter_digest = previous_parameter_digest
    strict_checkpoint_reloads: list[Mapping[str, Any]] = []
    for expected_step, record in zip(contract.CHECKPOINT_STEPS, checkpoint_records):
        metadata = contract.load_visual_context_checkpoint(
            record["path"],
            expected_file_sha256=record["file_sha256"],
            expected_step=expected_step,
            expected_manifest_digest=manifest.manifest_digest,
            expected_admission_digest=admission.receipt_digest,
            expected_memory_input_kind=args.memory_input_kind,
            handle=handle,
        )
        loaded_parameter_digest = stage_b_parameter_consensus(
            trainable,
            parallel.world_group,
            f"strict checkpoint reload step {expected_step}",
        )
        if metadata.get("adapter_parameter_digest") != record.get(
            "adapter_parameter_digest"
        ):
            fail(f"strict checkpoint reload metadata differs at step {expected_step}")
        strict_checkpoint_reloads.append(
            {
                "step": expected_step,
                "logical_records_seen": expected_step * contract.GLOBAL_BATCH,
                "path": record["path"],
                "file_sha256": record["file_sha256"],
                "checkpoint_state_digest": record["adapter_parameter_digest"],
                "runtime_parameter_digest_after_strict_load": (
                    loaded_parameter_digest
                ),
                "strict_load_succeeded": True,
            }
        )
    previous_parameter_digest = stage_b_parameter_consensus(
        trainable,
        parallel.world_group,
        "strict checkpoint reload final step 80",
    )
    if previous_parameter_digest != final_live_parameter_digest:
        fail("strict step-80 checkpoint did not restore the final live adapter")

    dist.barrier(group=parallel.world_group)
    receipt: Optional[Mapping[str, Any]] = None
    rank_zero_error: Optional[str] = None
    if distributed.rank == 0:
        try:
            adapter_path = stage / "adapter.safetensors"
            optimizer_path = stage / "optimizer.pt"
            history_path = stage / "history.json"
            _atomic_adapter_safetensors(
                adapter_path,
                handle,
                memory_input_kind=args.memory_input_kind,
                manifest_digest=manifest.manifest_digest,
                admission_digest=admission.receipt_digest,
                runtime=runtime,
            )
            runtime.atomic_torch_save(
                optimizer_path,
                {
                    "schema_version": RECEIPT_SCHEMA,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": 80,
                    "adapter_parameter_digest": previous_parameter_digest,
                    "memory_input_kind": args.memory_input_kind,
                },
            )
            history_value = {
                "schema_version": HISTORY_SCHEMA,
                "optimizer_steps": 80,
                "continuous_trajectory": True,
                "effective_batch_and_coverage": contract.sample_coverage_receipt(),
                "coordinates": [
                    item.receipt() for item in contract.exact80_coordinates()
                ],
                "steps": history_steps,
            }
            runtime.atomic_json(history_path, history_value)
            unsigned = {
                "schema_version": RECEIPT_SCHEMA,
                "method": METHOD,
                "complete": True,
                "mode": MODE,
                "memory_input_kind": args.memory_input_kind,
                "pair_invariants": pair_invariants,
                "optimizer_steps": 80,
                "positive_gradient_steps": positive_gradient_steps,
                "continuous_trajectory": True,
                "checkpoint_steps": list(contract.CHECKPOINT_STEPS),
                "checkpoint_records": [dict(item) for item in checkpoint_records],
                "checkpoint_coordinator": dict(coordinator_receipt[0]),
                "checkpoint_decode_chain": decode_chain,
                "strict_checkpoint_reloads": strict_checkpoint_reloads,
                "checkpoint_publication": {
                    "create_only_non_overwriting": True,
                    "fresh_root_required_before_optimizer": True,
                    "root": str(Path(args.checkpoint_output_root).expanduser()),
                    "atomic_with_training_bundle": False,
                    "orphaned_checkpoints_possible_if_final_bundle_publish_fails": True,
                },
                "post_training_review_integration": {
                    "all_checkpoints_strictly_loadable": True,
                    "fixed_sentinel_inference_launcher_implemented": False,
                    "checkpoint_videos_decoded": False,
                    "html_review_generated": False,
                    "review_complete": False,
                    "hard_remaining_gap": (
                        "decode the same fixed source/instruction/seed sentinel "
                        "manifest at steps 0/20/40/60/80 for both arms and build HTML"
                    ),
                },
                "stage_a_admission": admission.receipt(),
                "formal_pair_admission": {
                    key: value
                    for key, value in formal_pair_admission.items()
                    if key != "expected_initial_parameter_digest"
                },
                "dataset": {
                    **manifest.receipt(),
                    "manifest_path": str(manifest_path),
                    "manifest_file_sha256": args.expected_source_only_manifest_sha256,
                    "optimizer_split": "train",
                    "optimizer_rows": 64,
                    "confirmation_rows": 16,
                    "heldout_action_canary_rows": 8,
                    "posterior_index_0_accessed": True,
                    "posterior_index_1_synthetic_target_accessed": False,
                    "physical_index0_train_rows_preloaded": 64,
                    "legacy_parquet_opened_by_stage_b": False,
                    "synthetic_target_index1_bytes_read_by_stage_b": False,
                },
                "effective_batch_and_coverage": contract.sample_coverage_receipt(),
                "schedule": {
                    "exact40_schedule_sha256": contract.EXPECTED_SCHEDULE_SHA256,
                    "checkpoint_intervals": 4,
                    "microbatches_per_dp_arm_per_interval": 80,
                    "schedule_cycles_per_dp_arm_per_interval": 2,
                    "microbatch_uses_per_coordinate_per_dp_arm_per_interval": 2,
                    "runtime_audit": schedule_audit,
                },
                "objective": {
                    "name": "standard_target_only_noop_flow_matching",
                    "equations": {
                        "x_target_sigma": "(1-sigma)*z_source+sigma*epsilon",
                        "target_velocity": "epsilon-z_source",
                        "loss": "mean_squared_error(predicted_target_velocity,target_velocity)",
                    },
                    "native_source_condition": "clean_real_source_posterior_index_0",
                    "visual_context_input": args.memory_input_kind,
                    "same_epsilon_variant_binding": (
                        args.memory_input_kind
                        == "same_noise_forward_noised_source"
                    ),
                    "synthetic_target": False,
                    "pixel_regression": False,
                    "frozen_feature_reward": False,
                    "vlm_reward": False,
                    "rl": False,
                },
                "adapter": {
                    "architecture": dict(handle.receipt()),
                    "runtime_memory_input_binding": {
                        "input_kind": args.memory_input_kind,
                        "memory_encoder_reads_forward_noise": (
                            args.memory_input_kind
                            == "same_noise_forward_noised_source"
                        ),
                        "same_epsilon_as_native_target": (
                            args.memory_input_kind
                            == "same_noise_forward_noised_source"
                        ),
                        "synthetic_target_accessed": False,
                    },
                    "runtime_block_admission_binding": {
                        "status": (
                            "middle_bands_passed_with_preregistered_sparse_"
                            "representatives_installed"
                        ),
                        "stage_a_admission_digest": admission.receipt_digest,
                        "passed_block_bands": list(
                            admission.passed_block_bands
                        ),
                        "installed_sparse_representatives": list(
                            admission.installed_sparse_block_indices
                        ),
                        "per_block_causal_localization_claimed": False,
                        "adapter_self_authorized": False,
                    },
                },
                "step0_exact_base_parity": step0_exact_base_parity,
                "source_order_structural_gate": order_gate_receipt,
                "frozen_scope": {
                    "bernini_base": True,
                    "native_self_attention": True,
                    "native_text_cross_attention": True,
                    "vae": True,
                    "vae_loaded_in_training_process": False,
                    "precomputed_vae_posterior_index_0_only": True,
                    "t5": True,
                    "t5_released_after_frozen_embedding": True,
                    "only_visual_context_components_trainable": True,
                },
                "distributed": {
                    "profile": TOPOLOGY,
                    "world_size": 8,
                    "physical_dp_size": 2,
                    "ulysses_sp_size": 4,
                    "sp_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
                    "dp_groups": [[0, 4], [1, 5], [2, 6], [3, 7]],
                    "rank_serialized_renderer_load": True,
                    "gradient_checkpointing_non_reentrant": True,
                    "checkpoint_route_context_replayed": True,
                    "replicated_zero_dependency_for_condition_only_sp_ranks": True,
                    "dp2_sp4_gradient_mean": True,
                    "parameter_consensus_after_every_update": True,
                    "packed_sp_source_audit": packed_sp_audit,
                    "pre_optimizer_step0_parity_cuda_memory_world8": (
                        cuda_memory_world
                    ),
                },
                "resource_budget": resource_budget_receipt(args.memory_input_kind),
                "model": {
                    "bernini_commit": bernini_revision,
                    "veomni_commit": veomni_revision,
                    "model_revision": visual.PINNED_BERNINI_MODEL_REVISION,
                    "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                    "checkpoint_content_manifest_sha256": args.expected_checkpoint_content_manifest_sha256,
                    "checkpoint_content_identity": checkpoint_content_identity,
                },
                "method_source": dict(method_release_identity),
                "runtime": {
                    "torch": torch.__version__,
                    "torch_hip": str(torch.version.hip),
                    "transformers": transformers_version,
                    "diffusers": diffusers_version,
                },
                "authority": {
                    "gpu_runtime_executed": True,
                    "decoded_checkpoint_inference_executed": False,
                    "decoded_quality_claimed": False,
                    "action_editing_success_claimed": False,
                    "scientific_success_claimed": False,
                    "long_training_automatically_authorized": False,
                },
                "artifacts": {
                    "adapter.safetensors": runtime.file_sha256(adapter_path),
                    "optimizer.pt": runtime.file_sha256(optimizer_path),
                    "history.json": runtime.file_sha256(history_path),
                },
            }
            receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
            runtime.atomic_json(stage / "receipt.json", receipt)
        except Exception as error:
            rank_zero_error = f"{type(error).__name__}: {error}"
    _publish_output(
        runtime=runtime,
        output=output,
        stage=stage,
        receipt=receipt,
        rank=distributed.rank,
        world_group=parallel.world_group,
        rank_zero_error=rank_zero_error,
    )
    dist.barrier(group=parallel.world_group)
    if distributed.rank == 0:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "arm": args.memory_input_kind,
                    "optimizer_steps": 80,
                    "checkpoint_steps": list(contract.CHECKPOINT_STEPS),
                    "parent_allocation_released": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CleanSourceVisualStageBTrainingError",
    "PackedNoOpCondition",
    "PreparedArm",
    "audit_packed_sp_sources",
    "build_parser",
    "main",
    "prepare_noop_condition",
    "resource_budget_receipt",
    "stage_b_parameter_consensus",
    "stage_b_synchronize_initial_parameters",
    "stage_b_trainable_parameters_digest",
    "source_order_structural_gate",
    "validate_step0_exact_base_parity_world8",
    "validate_cli_and_admission",
]
