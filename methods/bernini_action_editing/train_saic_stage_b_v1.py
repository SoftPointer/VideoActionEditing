#!/usr/bin/env python3
"""Fail-closed SAIC-v2 Stage-B integration preflight and orchestrator.

This is intentionally not a placeholder trainer.  The executable validates
every artifact that would be allowed to influence Stage-B, audits the fixed
WORLD8=DP2xSP4/exact81 execution plan, and then refuses to create an optimizer
or output directory while the repository lacks a byte-audited native runtime
for any required operation.

The current blockers are substantive: Bernini's official ``sample`` method
does not expose the per-step raw noisy target before its source-conditioned
transformer forwards; the qualified event critic has a score-only consumer
but no media-to-four-stage-score executor; and the codec/seven-axis/fresh
accept-or-rollback evaluators are not registered as one provenance-closed
runtime.  Guessing those seams would silently turn an online method into a
train-only representation or admit unbound scalar scores.  Consequently the
non-preflight path raises *before* model loading, optimizer construction, or
filesystem publication.  Once those runtime bridges exist, this file must be
revised and requalified rather than bypassed with an acknowledgement flag.

No target edit, pure-T2V media/latent/noise, action ID, mask, pose, flow,
track, trajectory, or motion donor is accepted by this command line.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import infer_source_kv_carrier_oracle as checkpoint_audit  # noqa: E402
import saic_event_reward_v1 as event_reward  # noqa: E402
import saic_inverse_recoverability_v1 as inverse_recoverability  # noqa: E402
import saic_online_motion_field_v1 as online_motion  # noqa: E402
import saic_rf_preference_objective_v1 as rf_preference  # noqa: E402
import saic_rollout_preference_set_v1 as rollout_preference  # noqa: E402
import saic_source_anchor_adapter_v1 as source_anchor  # noqa: E402
import saic_temporal_action_operator_v2 as temporal_operator  # noqa: E402
import source_self_runtime as distributed_runtime  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_saic_source_anchor_v1 as source_anchor_trainer  # noqa: E402


METHOD_NAME = "bernini-saic-stage-b-v1"
PREFLIGHT_SCHEMA_VERSION = "bernini-saic-stage-b-preflight-v1"
PUBLISHED_CHECKPOINT_SCHEMA_VERSION = "bernini-saic-stage-b-checkpoint-v1"
WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
FRAME_COUNT = 81
LATENT_PHASES = 21
EXACT40_STEPS = 40
ROLLOUT_K = 4
OUTER_CYCLES = 2
LOCAL_UPDATES_PER_CYCLE = 4
SIGMA_CELLS_PER_UPDATE = 2
LEARNING_RATE = 5.0e-7
MAX_GRAD_NORM = 0.5
WEIGHT_DECAY = 0.0
STEP_SCALES = (1.0, 0.5, 0.25, 0.125)
UPDATE_INDICES = (4, 12, 20, 28, 33, 34, 35, 37)
FORBIDDEN_UPDATE_INDICES = (38, 39)
REQUIRED_ARMS = ("dog", "human")
PRESERVATION_AXES = (
    "identity",
    "camera",
    "background",
    "non_target",
    "quality",
    "source_bind",
    "inverse",
)
FORBIDDEN_PUBLIC_ARGUMENTS = frozenset(
    {
        "action_id",
        "target_video",
        "paired_target",
        "pure_t2v_video",
        "pure_t2v_media",
        "pure_t2v_latent",
        "pure_t2v_noise",
        "proposal_video",
        "proposal_latent",
        "donor_video",
        "donor_latent",
        "mask",
        "pose",
        "flow",
        "track",
        "trajectory",
        "swept_tube",
    }
)

_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MAX_JSON_BYTES = 8 * 1024 * 1024

_PREFLIGHT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "artifact_identities",
        "topology",
        "execution_plan",
        "primitive_contracts",
        "runtime_capabilities",
        "runtime_blockers",
        "artifacts_qualified",
        "runtime_complete",
        "optimizer_created",
        "optimizer_updates",
        "output_created",
        "training_started",
        "semantic_action_editing_success_claimed",
        "preflight_digest",
    }
)

_PUBLISHED_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "complete",
        "publication_authorized",
        "world_size",
        "data_parallel_size",
        "sequence_parallel_size",
        "frame_count",
        "latent_phases",
        "exact40_steps",
        "rollout_k",
        "outer_cycles_completed",
        "optimizer_update_count",
        "source_anchor_adapter_sha256",
        "critic_checkpoint_sha256",
        "critic_qualification_receipt_digest",
        "motion_adapter_sha256",
        "motion_adapter_state_tensor_sha256",
        "action_operator_contract_digest",
        "online_motion_contract_digest",
        "confirmation_gate",
        "inference_contract",
        "method_source_revision",
        "method_source_archive_sha256",
        "receipt_digest",
    }
)


class SAICStageBTrainingError(RuntimeError):
    """Raised before an ambiguous artifact or incomplete runtime can train."""


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
        raise SAICStageBTrainingError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, bits: int, label: str) -> str:
    pattern = _SHA1 if bits == 160 else _SHA256
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise SAICStageBTrainingError(f"{label} must be lowercase SHA-{bits}")
    return value


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise SAICStageBTrainingError(
            f"{label} schema differs; missing={sorted(set(fields)-actual)} "
            f"extra={sorted(actual-set(fields))}"
        )
    return value


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        before = path.lstat()
    except OSError as error:
        raise SAICStageBTrainingError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SAICStageBTrainingError(f"{label} must be a plain regular file")
    return before


def _hash_stable_file(path: Path, *, label: str) -> tuple[str, int]:
    before = _regular_file(path, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SAICStageBTrainingError(f"cannot open {label}: {path}") from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise SAICStageBTrainingError(f"{label} changed while opening")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise SAICStageBTrainingError(f"cannot read {label}: {path}") from error
    finally:
        os.close(descriptor)
    named = _regular_file(path, label=label)
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if (
        identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or identity(after) != identity(named)
    ):
        raise SAICStageBTrainingError(f"{label} changed while hashing")
    return digest.hexdigest(), int(after.st_size)


def _canonical_plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path("/"):
        raise SAICStageBTrainingError(f"{label} must be absolute and non-root")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SAICStageBTrainingError(f"cannot resolve {label}: {path}") from error
    if resolved != path or path.is_symlink() or not path.is_file():
        raise SAICStageBTrainingError(f"{label} must be a canonical plain file")
    return path


def _canonical_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path("/"):
        raise SAICStageBTrainingError(f"{label} must be absolute and non-root")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SAICStageBTrainingError(f"cannot resolve {label}: {path}") from error
    if resolved != path or path.is_symlink() or not path.is_dir():
        raise SAICStageBTrainingError(f"{label} must be a canonical plain directory")
    return path


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    sha256: str
    byte_size: int
    device: int
    inode: int
    mtime_ns: int

    @classmethod
    def capture(
        cls, value: str | Path, expected_sha256: str, *, label: str
    ) -> "FileSnapshot":
        path = _canonical_plain_file(value, label=label)
        expected = _sha(expected_sha256, bits=256, label=f"{label} expected digest")
        observed, byte_size = _hash_stable_file(path, label=label)
        if observed != expected:
            raise SAICStageBTrainingError(f"{label} SHA-256 differs")
        identity = path.stat()
        return cls(
            path=path,
            sha256=observed,
            byte_size=byte_size,
            device=int(identity.st_dev),
            inode=int(identity.st_ino),
            mtime_ns=int(identity.st_mtime_ns),
        )

    def assert_unchanged(self) -> None:
        observed, size = _hash_stable_file(self.path, label="bound artifact")
        identity = self.path.stat()
        if (
            observed,
            size,
            int(identity.st_dev),
            int(identity.st_ino),
            int(identity.st_mtime_ns),
        ) != (
            self.sha256,
            self.byte_size,
            self.device,
            self.inode,
            self.mtime_ns,
        ):
            raise SAICStageBTrainingError(f"bound artifact changed: {self.path}")

    def receipt(self) -> Mapping[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }


def _read_json_snapshot(snapshot: FileSnapshot, *, label: str) -> Mapping[str, Any]:
    snapshot.assert_unchanged()
    if snapshot.byte_size > _MAX_JSON_BYTES:
        raise SAICStageBTrainingError(f"{label} exceeds {_MAX_JSON_BYTES} bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise SAICStageBTrainingError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise SAICStageBTrainingError(f"{label} contains {value}")

    try:
        value = json.loads(
            snapshot.path.read_text(encoding="ascii"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SAICStageBTrainingError(f"cannot decode {label}") from error
    if not isinstance(value, Mapping):
        raise SAICStageBTrainingError(f"{label} root must be an object")
    snapshot.assert_unchanged()
    return value


def _sealed_receipt(
    value: Mapping[str, Any], *, digest_field: str, label: str
) -> Mapping[str, Any]:
    declared = _sha(value.get(digest_field), bits=256, label=f"{label} digest")
    body = {key: item for key, item in value.items() if key != digest_field}
    if object_sha256(body) != declared:
        raise SAICStageBTrainingError(f"{label} embedded digest differs")
    return value


def validate_world8_environment(
    environment: Mapping[str, str] = os.environ,
) -> Mapping[str, int]:
    try:
        world = int(environment["WORLD_SIZE"])
        rank = int(environment["RANK"])
        local_rank = int(environment["LOCAL_RANK"])
    except (KeyError, ValueError) as error:
        raise SAICStageBTrainingError(
            "Stage-B runtime requires torchrun WORLD_SIZE/RANK/LOCAL_RANK"
        ) from error
    if (
        world != WORLD_SIZE
        or not 0 <= rank < world
        or not 0 <= local_rank < world
        or local_rank != rank
    ):
        raise SAICStageBTrainingError("Stage-B requires exactly WORLD8 on one node")
    local_world = environment.get("LOCAL_WORLD_SIZE")
    if local_world is not None:
        try:
            if int(local_world) != WORLD_SIZE:
                raise SAICStageBTrainingError(
                    "Stage-B requires all WORLD8 ranks on one node"
                )
        except ValueError as error:
            raise SAICStageBTrainingError("LOCAL_WORLD_SIZE is invalid") from error
    arm_index = rank // SP_SIZE
    sp_rank = rank % SP_SIZE
    if arm_index not in (0, 1) or sp_rank not in range(SP_SIZE):
        raise SAICStageBTrainingError("WORLD8 does not map to DP2xSP4")
    if (arm_index == 0) != (rank < 4):
        raise SAICStageBTrainingError("GPUs/ranks 0..3 must own the dog arm")
    return {
        "world_size": world,
        "rank": rank,
        "local_rank": local_rank,
        "data_parallel_size": DP_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "arm_index": arm_index,
        "sequence_parallel_rank": sp_rank,
    }


def validate_world8_accelerators(topology: Mapping[str, int]) -> Mapping[str, Any]:
    """Bind runtime preflight to eight visible AUH ROCm accelerators."""

    try:
        import torch
    except Exception as error:  # pragma: no cover - AUH runtime only
        raise SAICStageBTrainingError("PyTorch is unavailable for GPU audit") from error
    if (
        not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
        or torch.cuda.device_count() != WORLD_SIZE
    ):
        raise SAICStageBTrainingError(
            "Stage-B runtime requires exactly eight visible AUH ROCm GPUs"
        )
    local_rank = topology.get("local_rank")
    if type(local_rank) is not int or not 0 <= local_rank < WORLD_SIZE:
        raise SAICStageBTrainingError("GPU audit local rank differs")
    torch.cuda.set_device(local_rank)
    return {
        **dict(topology),
        "visible_accelerator_count": int(torch.cuda.device_count()),
        "torch_hip": str(torch.version.hip),
        "device_type": "cuda_rocm",
    }


def validate_world8_dp2sp4_collectives(
    args: argparse.Namespace, topology: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Initialize and audit Bernini's real WORLD8 DP2xUlysses-SP4 groups."""

    try:
        import torch.distributed as dist
        bernini_root, veomni_root, _bernini_revision, _veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        legacy.activate_source_trees(bernini_root, veomni_root)
        from bernini.parallel import init_parallel_state
    except Exception as error:  # pragma: no cover - AUH runtime only
        raise SAICStageBTrainingError(
            f"cannot prepare Bernini distributed preflight: {error}"
        ) from error
    if dist.is_initialized():
        raise SAICStageBTrainingError(
            "distributed process group must be uninitialized at Stage-B entry"
        )
    parallel = None
    try:
        contract = distributed_runtime.distributed_contract()
        distributed_runtime.initialise_distributed(contract)
        parallel = distributed_runtime.validate_parallel_state(
            contract, init_parallel_state(ulysses_size=SP_SIZE)
        )
        if not distributed_runtime.world_all_true(
            True, group=parallel.world_group
        ):
            raise SAICStageBTrainingError("WORLD8 RCCL truth collective differed")
        distributed_runtime.digest_consensus(
            "saic-stage-b-world8-dp2sp4-v1",
            group=parallel.world_group,
            expected_count=WORLD_SIZE,
            label="Stage-B distributed preflight",
        )
        dist.barrier(group=parallel.world_group)
        result = {
            **dict(topology),
            "bernini_parallel_world_size": int(parallel.contract.world_size),
            "bernini_parallel_dp_size": DP_SIZE,
            "bernini_parallel_ulysses_size": SP_SIZE,
            "rccl_world_collective_passed": True,
            "sp_group_membership_collective_passed": True,
            "dp_group_membership_collective_passed": True,
        }
    except Exception as error:  # pragma: no cover - AUH runtime only
        if isinstance(error, SAICStageBTrainingError):
            raise
        raise SAICStageBTrainingError(
            f"WORLD8 DP2xSP4 collective preflight failed: {error}"
        ) from error
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return result


def resolve_create_only_output(value: str | Path) -> Path:
    output = Path(value).expanduser()
    if not output.is_absolute() or output == Path("/"):
        raise SAICStageBTrainingError("output must be absolute and non-root")
    if _SAFE_BASENAME.fullmatch(output.name) is None:
        raise SAICStageBTrainingError("output basename is unsafe")
    if output.exists() or output.is_symlink():
        raise SAICStageBTrainingError("output is create-only")
    parent = _canonical_directory(output.parent, label="output parent")
    canonical = parent / output.name
    if canonical != output:
        raise SAICStageBTrainingError("output must already be canonical")
    return output


def validate_stage_a_bundle(
    *, adapter: FileSnapshot, receipt: FileSnapshot
) -> Mapping[str, Any]:
    value = _read_json_snapshot(receipt, label="Stage-A receipt")
    _sealed_receipt(value, digest_field="receipt_digest", label="Stage-A receipt")
    required_top_level = {
        "schema_version",
        "method",
        "complete",
        "run_contract",
        "manifest",
        "native_runtime",
        "objective",
        "adapter",
        "heldout_gate",
        "scientific_limitations",
        "artifacts",
        "model",
        "runtime",
        "method_source_revision",
        "method_source_archive_sha256",
        "source_anchor_pretext_only",
        "action_training",
        "semantic_action_editing_success",
        "decoded_rgb_appearance_preservation_success",
        "source_anchor_checkpoint_publication_authorized",
        "action_stage_authorized",
        "smoke_incomplete_row_coverage",
        "receipt_digest",
    }
    if set(value) != required_top_level:
        raise SAICStageBTrainingError("Stage-A receipt root schema differs")
    run_contract = value["run_contract"]
    adapter_row = value["adapter"]
    gate = value["heldout_gate"]
    artifacts = value["artifacts"]
    limitations = value["scientific_limitations"]
    if not all(
        isinstance(item, Mapping)
        for item in (run_contract, adapter_row, gate, artifacts, limitations)
    ):
        raise SAICStageBTrainingError("Stage-A nested receipt schema differs")
    if (
        value["schema_version"] != source_anchor_trainer.RUN_RECEIPT_SCHEMA
        or value["method"] != source_anchor_trainer.METHOD_NAME
        or value["complete"] is not True
        or run_contract.get("mode") != "formal"
        or run_contract.get("world_size") != WORLD_SIZE
        or run_contract.get("data_parallel_size") != DP_SIZE
        or run_contract.get("sequence_parallel_size") != SP_SIZE
        or run_contract.get("frame_count") != FRAME_COUNT
        or run_contract.get("optimizer_updates")
        != source_anchor_trainer.FORMAL_UPDATES
        or run_contract.get("all_train_rows_used_once_as_clean_endpoint") is not True
        or adapter_row.get("checkpoint_published") is not True
        or gate.get("noncompensating_all_pass") is not True
        or value["source_anchor_checkpoint_publication_authorized"] is not True
        or value["source_anchor_pretext_only"] is not True
        or value["action_training"] is not False
        or value["semantic_action_editing_success"] is not False
        or value["decoded_rgb_appearance_preservation_success"] is not False
        or value["action_stage_authorized"] is not False
        or value["smoke_incomplete_row_coverage"] is not False
        or limitations.get("future_action_stage_requires_fresh_rollout_nonregression")
        is not True
        or limitations.get(
            "future_action_stage_must_test_action_and_identity_camera_background_separately"
        )
        is not True
    ):
        raise SAICStageBTrainingError("Stage-A publication/gate contract differs")
    declared_adapter_sha = artifacts.get("adapter.safetensors")
    if declared_adapter_sha != adapter.sha256:
        raise SAICStageBTrainingError("Stage-A receipt is bound to different adapter bytes")
    roundtrip = adapter_row.get("safetensors_roundtrip")
    if (
        not isinstance(roundtrip, Mapping)
        or roundtrip.get("schema_version") != source_anchor_trainer.SAFETENSORS_SCHEMA
        or roundtrip.get("file_sha256") != adapter.sha256
        or roundtrip.get("roundtrip_byte_exact_tensors") is not True
        or roundtrip.get("metadata_closed") is not True
    ):
        raise SAICStageBTrainingError("Stage-A safetensors roundtrip differs")
    return {
        "adapter_sha256": adapter.sha256,
        "receipt_sha256": receipt.sha256,
        "receipt_digest": value["receipt_digest"],
        "adapter_state_tensor_sha256": roundtrip.get("state_tensor_sha256"),
        "heldout_gate_digest": gate.get("digest"),
        "source_anchor_pretext_only": True,
        "fresh_stage_b_nonregression_still_required": True,
    }


def validate_source_manifest(
    snapshot: FileSnapshot, *, verify_bound_files: bool
) -> Mapping[str, Any]:
    value = source_set.load_manifest(snapshot.path)
    try:
        summary = source_set.validate_manifest(
            value, verify_bound_files=verify_bound_files
        )
    except Exception as error:
        raise SAICStageBTrainingError(f"SAIC source-set validation failed: {error}") from error
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise SAICStageBTrainingError("Stage-B requires the immutable eight-source set")
    for row in rows:
        if (
            row.get("analysis_split") == "fit"
            and len(row.get("rollout_seeds", ())) != 2
        ) or (
            row.get("analysis_split") == "confirmation"
            and len(row.get("rollout_seeds", ())) != 3
        ):
            raise SAICStageBTrainingError("source-set seed census differs")
    snapshot.assert_unchanged()
    return {
        "file_sha256": snapshot.sha256,
        "row_count": len(rows),
        "fit_count": sum(row["analysis_split"] == "fit" for row in rows),
        "confirmation_count": sum(
            row["analysis_split"] == "confirmation" for row in rows
        ),
        "validator_summary_sha256": object_sha256(summary),
        "manifest_is_source_only_and_does_not_authorize_optimizer": True,
    }


def validate_checkpoint_content(
    *,
    checkpoint: Path,
    manifest: FileSnapshot,
) -> Mapping[str, Any]:
    try:
        identity = checkpoint_audit.validate_checkpoint_content(
            checkpoint,
            manifest.path,
            expected_manifest_sha256=manifest.sha256,
        )
    except Exception as error:
        raise SAICStageBTrainingError(
            f"Bernini checkpoint content validation failed: {error}"
        ) from error
    manifest.assert_unchanged()
    return dict(identity)


def validate_critic_bundle(
    *, checkpoint: FileSnapshot, qualification: FileSnapshot
) -> event_reward.SAICEventRewardBoundary:
    try:
        boundary = event_reward.load_event_reward_boundary(
            checkpoint.path, qualification.path
        )
    except Exception as error:
        raise SAICStageBTrainingError(
            f"critic qualification validation failed: {error}"
        ) from error
    if (
        boundary.critic_checkpoint_sha256 != checkpoint.sha256
        or boundary.critic_checkpoint_bytes != checkpoint.byte_size
        or boundary.qualification_file_sha256 != qualification.sha256
    ):
        raise SAICStageBTrainingError("critic bundle snapshot identity differs")
    return boundary


def execution_plan() -> Mapping[str, Any]:
    """Return the closed Stage-B algorithm; this is not an execution claim."""

    return {
        "frame_count": FRAME_COUNT,
        "latent_phases": LATENT_PHASES,
        "exact40_steps": EXACT40_STEPS,
        "rollout_k_per_source_per_round": ROLLOUT_K,
        "outer_cycles": OUTER_CYCLES,
        "local_updates_per_cycle": LOCAL_UPDATES_PER_CYCLE,
        "sigma_cells_per_update": SIGMA_CELLS_PER_UPDATE,
        "registered_update_indices": list(UPDATE_INDICES),
        "forbidden_exact_base_indices": list(FORBIDDEN_UPDATE_INDICES),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "fresh_accept_step_scales": list(STEP_SCALES),
        "required_arms": list(REQUIRED_ARMS),
        "arm_assignment": {"ranks_0_3": "dog", "ranks_4_7": "human"},
        "rollout_path": (
            "current_source_conditioned_editor_with_frozen_stage_a_and_"
            "online_action_minus_noop_t2v_field"
        ),
        "endpoint_path": "decode_rgb8_codec_roundtrip_vae_reencode_detach",
        "candidate_origin": "fresh_current_policy_only",
        "pair_gate": {
            "event_gain_required": True,
            "preservation_axes": list(PRESERVATION_AXES),
            "all_axis_floors_required": True,
            "all_axis_noninferiority_required": True,
            "weighted_compensation": False,
            "dog_and_human_pair_required": True,
        },
        "objective": "reference_relative_rectified_flow_serial_output_leaf_vjp",
        "inverse": (
            "codec_midpoint_as_unique_visual_source_only_after_absolute_"
            "four_stage_event_and_recoverability_authorization"
        ),
        "accept_or_rollback": (
            "fresh_exact81_same_seed_evaluation_after_each_scaled_candidate_step_"
            "and_byte_identical_rollback_on_any_axis_regression"
        ),
        "pure_t2v_visual_role": "critic_calibration_only_never_endpoint_or_condition",
        "inference_inputs": ["source_video", "natural_language_instruction"],
        "forbidden_visual_or_privileged_inputs": sorted(FORBIDDEN_PUBLIC_ARGUMENTS),
    }


def primitive_contracts() -> Mapping[str, Any]:
    """Bind the implemented mathematical primitives without granting runtime."""

    signatures = {
        "online_motion": set(
            inspect.signature(online_motion.build_online_motion_field).parameters
        ),
        "event_reward": set(
            inspect.signature(event_reward.consume_event_reward).parameters
        ),
        "pair_gate": set(
            inspect.signature(rollout_preference.build_preference_set).parameters
        ),
        "rf_preference": set(
            inspect.signature(rf_preference.reference_relative_rf_preference).parameters
        ),
        "inverse_authorization": set(
            inspect.signature(
                inverse_recoverability.authorize_inverse_flow_matching
            ).parameters
        ),
        "inverse_flow_matching": set(
            inspect.signature(
                inverse_recoverability.authorized_inverse_flow_matching
            ).parameters
        ),
    }
    exposed = {
        name: sorted(parameters & FORBIDDEN_PUBLIC_ARGUMENTS)
        for name, parameters in signatures.items()
    }
    if any(exposed.values()):
        raise SAICStageBTrainingError(
            f"Stage-B primitive exposes forbidden visual input: {exposed}"
        )
    operator_contract = {
        "schema_version": temporal_operator.SCHEMA_VERSION,
        "blocks": list(temporal_operator.ACTION_BLOCK_INDICES),
        "rank": temporal_operator.ACTION_OPERATOR_RANK,
        "active_sigma_indices": list(temporal_operator.ACTIVE_SIGMA_INDICES),
        "low_sigma_exact_base_indices": list(
            temporal_operator.LOW_SIGMA_EXACT_BASE_INDICES
        ),
        "motion_field_schema": online_motion.SCHEMA_VERSION,
    }
    online_contract = {
        "schema_version": online_motion.SCHEMA_VERSION,
        "builder": "build_online_motion_field",
        "builder_parameters": sorted(signatures["online_motion"]),
        "latent_channels": online_motion.LATENT_CHANNELS,
        "latent_phases_exact81": online_motion.LATENT_PHASES_EXACT81,
        "phase_code_dimension": online_motion.PHASE_CODE_DIM,
        "training_and_inference_same_builder": True,
        "t2v_media_or_proposal_consumed": False,
    }
    return {
        "source_anchor_schema": source_anchor.SCHEMA_VERSION,
        "online_motion_schema": online_motion.SCHEMA_VERSION,
        "temporal_operator": operator_contract,
        "temporal_operator_contract_digest": object_sha256(operator_contract),
        "online_motion_contract": online_contract,
        "online_motion_contract_digest": object_sha256(online_contract),
        "event_reward_schema": event_reward.RESULT_SCHEMA_VERSION,
        "preference_set_schema": rollout_preference.PREFERENCE_SET_SCHEMA_VERSION,
        "rf_objective": dict(rf_preference.contract_receipt()),
        "inverse_schema": inverse_recoverability.SCHEMA_VERSION,
        "forbidden_signature_inputs_exposed": exposed,
    }


def runtime_capability_audit() -> Mapping[str, bool]:
    """Audit executable bridges, distinct from the mathematical primitives."""

    # These booleans are deliberately literal.  A future implementation must
    # replace them only together with byte-identity tests for the new bridge.
    return {
        "stage_a_strict_safetensors_loader": callable(
            getattr(source_anchor.SAICSourceAnchorHandle, "load_trainable_state_dict", None)
        ),
        "online_motion_field_primitive": callable(
            getattr(online_motion, "build_online_motion_field", None)
        ),
        "temporal_operator_primitive": callable(
            getattr(temporal_operator, "install_saic_temporal_action_operator", None)
        ),
        "hard_seven_axis_pair_gate_primitive": callable(
            getattr(rollout_preference, "build_preference_set", None)
        ),
        "reference_relative_rf_primitive": callable(
            getattr(rf_preference, "reference_relative_rf_preference", None)
        ),
        "inverse_authorization_primitive": callable(
            getattr(inverse_recoverability, "authorize_inverse_flow_matching", None)
        ),
        "native_sampler_pre_forward_raw_state_hook": False,
        "online_t2v_action_noop_executor_inside_same_native_step": False,
        "qualified_critic_checkpoint_media_executor": False,
        "seven_axis_whole_frame_score_packet_builder": False,
        "fixed_rgb8_codec_vae_reencode_receipt_bridge": False,
        "serial_rf_native_vjp_executor_for_online_operator": False,
        "fresh_scaled_step_accept_rollback_executor": False,
        "temporal_operator_closed_checkpoint_loader_and_publisher": False,
    }


def runtime_blockers(capabilities: Mapping[str, bool]) -> tuple[str, ...]:
    names = {
        "native_sampler_pre_forward_raw_state_hook": (
            "official UniPC sampling lacks a byte-audited hook exposing the current "
            "raw [1,16,21,H,W] noisy target before native transformer forwards"
        ),
        "online_t2v_action_noop_executor_inside_same_native_step": (
            "no bridge queries frozen T2V action/no-op on that exact state and then "
            "routes the source-conditioned student forward at the same timestep"
        ),
        "qualified_critic_checkpoint_media_executor": (
            "the event boundary consumes sealed scalar packets but no qualified "
            "critic checkpoint loader executes media-to-four-stage scoring"
        ),
        "seven_axis_whole_frame_score_packet_builder": (
            "no provenance-closed evaluator builds all seven source-relative axes"
        ),
        "fixed_rgb8_codec_vae_reencode_receipt_bridge": (
            "no fixed RGB8 codec roundtrip and VAE re-encode bridge atomically binds "
            "the detached endpoint latent to its receipt"
        ),
        "serial_rf_native_vjp_executor_for_online_operator": (
            "no native replay executor serializes chosen/rejected RF output cotangents "
            "through the online action route"
        ),
        "fresh_scaled_step_accept_rollback_executor": (
            "no fresh exact81 same-seed post-step evaluator can accept or byte-identically rollback"
        ),
        "temporal_operator_closed_checkpoint_loader_and_publisher": (
            "the temporal operator exposes state_dict_for_save but no qualified closed loader/publication path"
        ),
    }
    return tuple(message for key, message in names.items() if capabilities.get(key) is not True)


@dataclass(frozen=True)
class StageBArtifacts:
    source_manifest: FileSnapshot
    stage_a_adapter: FileSnapshot
    stage_a_receipt: FileSnapshot
    critic_checkpoint: FileSnapshot
    critic_qualification: FileSnapshot
    checkpoint_content_manifest: FileSnapshot
    source_summary: Mapping[str, Any]
    stage_a_summary: Mapping[str, Any]
    critic_boundary: event_reward.SAICEventRewardBoundary
    checkpoint_identity: Mapping[str, Any]

    def assert_unchanged(self) -> None:
        for snapshot in (
            self.source_manifest,
            self.stage_a_adapter,
            self.stage_a_receipt,
            self.critic_checkpoint,
            self.critic_qualification,
            self.checkpoint_content_manifest,
        ):
            snapshot.assert_unchanged()
        # Re-open/re-hash both critic files through the reward boundary too.
        self.critic_boundary._revalidate_files()  # noqa: SLF001 - trust boundary audit


def validate_cli(args: argparse.Namespace) -> None:
    for name in ("method_source_revision", "expected_bernini_commit", "expected_veomni_commit"):
        _sha(getattr(args, name), bits=160, label=name)
    for name in (
        "method_source_archive_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "expected_source_manifest_sha256",
        "expected_stage_a_adapter_sha256",
        "expected_stage_a_receipt_sha256",
        "expected_critic_checkpoint_sha256",
        "expected_critic_qualification_sha256",
    ):
        _sha(getattr(args, name), bits=256, label=name)
    if args.expected_bernini_commit != legacy.BERNINI_OFFICIAL_COMMIT:
        raise SAICStageBTrainingError("Bernini source revision differs")
    if args.expected_veomni_commit != legacy.VEOMNI_TESTED_COMMIT:
        raise SAICStageBTrainingError("VeOmni source revision differs")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise SAICStageBTrainingError("Bernini checkpoint tree identity differs")
    for name, expected in (
        ("num_frames", FRAME_COUNT),
        ("rollout_k", ROLLOUT_K),
        ("outer_cycles", OUTER_CYCLES),
        ("local_updates_per_cycle", LOCAL_UPDATES_PER_CYCLE),
    ):
        if type(getattr(args, name)) is not int or getattr(args, name) != expected:
            raise SAICStageBTrainingError(f"{name} must be exactly {expected}")
    for name, expected in (
        ("learning_rate", LEARNING_RATE),
        ("max_grad_norm", MAX_GRAD_NORM),
    ):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected
        ):
            raise SAICStageBTrainingError(f"{name} must be exactly {expected}")
    if args.ack_incomplete_runtime_preflight is not True:
        raise SAICStageBTrainingError(
            "explicit incomplete-runtime preflight acknowledgement is required"
        )


def preflight_artifacts(
    args: argparse.Namespace,
    *,
    verify_bound_sources: bool = True,
    verify_model_environment: bool = True,
) -> StageBArtifacts:
    source_manifest = FileSnapshot.capture(
        args.source_manifest,
        args.expected_source_manifest_sha256,
        label="source manifest",
    )
    stage_a_adapter = FileSnapshot.capture(
        args.stage_a_adapter,
        args.expected_stage_a_adapter_sha256,
        label="Stage-A adapter",
    )
    stage_a_receipt = FileSnapshot.capture(
        args.stage_a_receipt,
        args.expected_stage_a_receipt_sha256,
        label="Stage-A receipt",
    )
    critic_checkpoint = FileSnapshot.capture(
        args.critic_checkpoint,
        args.expected_critic_checkpoint_sha256,
        label="critic checkpoint",
    )
    critic_qualification = FileSnapshot.capture(
        args.critic_qualification,
        args.expected_critic_qualification_sha256,
        label="critic qualification",
    )
    checkpoint_content_manifest = FileSnapshot.capture(
        args.checkpoint_content_manifest,
        args.expected_checkpoint_content_manifest_sha256,
        label="checkpoint content manifest",
    )
    source_summary = validate_source_manifest(
        source_manifest, verify_bound_files=verify_bound_sources
    )
    stage_a_summary = validate_stage_a_bundle(
        adapter=stage_a_adapter, receipt=stage_a_receipt
    )
    critic_boundary = validate_critic_bundle(
        checkpoint=critic_checkpoint, qualification=critic_qualification
    )

    checkpoint = _canonical_directory(args.checkpoint, label="Bernini checkpoint")
    checkpoint_identity: Mapping[str, Any] = {
        "checkpoint_validation_deferred_for_unit_contract": True
    }
    if verify_model_environment:
        bernini_root = _canonical_directory(args.bernini_root, label="Bernini root")
        veomni_root = _canonical_directory(args.veomni_root, label="VeOmni root")
        try:
            legacy.validate_source_trees(
                bernini_root,
                veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
            _checkpoint_path, transformer_config = legacy.validate_checkpoint(checkpoint)
        except Exception as error:
            raise SAICStageBTrainingError(
                f"official source/checkpoint validation failed: {error}"
            ) from error
        if (
            transformer_config.get("num_attention_heads") != 12
            or int(transformer_config["num_attention_heads"]) % SP_SIZE
        ):
            raise SAICStageBTrainingError(
                "Bernini-R 1.3B must expose 12 heads divisible by SP4"
            )
        checkpoint_identity = validate_checkpoint_content(
            checkpoint=checkpoint, manifest=checkpoint_content_manifest
        )
    artifacts = StageBArtifacts(
        source_manifest=source_manifest,
        stage_a_adapter=stage_a_adapter,
        stage_a_receipt=stage_a_receipt,
        critic_checkpoint=critic_checkpoint,
        critic_qualification=critic_qualification,
        checkpoint_content_manifest=checkpoint_content_manifest,
        source_summary=source_summary,
        stage_a_summary=stage_a_summary,
        critic_boundary=critic_boundary,
        checkpoint_identity=checkpoint_identity,
    )
    artifacts.assert_unchanged()
    return artifacts


def build_preflight_receipt(
    *, artifacts: StageBArtifacts, topology: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    capabilities = runtime_capability_audit()
    blockers = runtime_blockers(capabilities)
    body = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "artifact_identities": {
            "source_manifest": dict(artifacts.source_manifest.receipt()),
            "source_summary": dict(artifacts.source_summary),
            "stage_a_adapter": dict(artifacts.stage_a_adapter.receipt()),
            "stage_a_receipt": dict(artifacts.stage_a_receipt.receipt()),
            "stage_a_summary": dict(artifacts.stage_a_summary),
            "critic_checkpoint": dict(artifacts.critic_checkpoint.receipt()),
            "critic_qualification": dict(artifacts.critic_qualification.receipt()),
            "critic_qualification_receipt_digest": (
                artifacts.critic_boundary.qualification_receipt_digest
            ),
            "checkpoint_content_manifest": dict(
                artifacts.checkpoint_content_manifest.receipt()
            ),
            "checkpoint_content_identity": dict(artifacts.checkpoint_identity),
        },
        "topology": None if topology is None else dict(topology),
        "execution_plan": dict(execution_plan()),
        "primitive_contracts": dict(primitive_contracts()),
        "runtime_capabilities": dict(capabilities),
        "runtime_blockers": list(blockers),
        "artifacts_qualified": True,
        "runtime_complete": not blockers,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "output_created": False,
        "training_started": False,
        "semantic_action_editing_success_claimed": False,
    }
    value = {**body, "preflight_digest": object_sha256(body)}
    _closed(value, _PREFLIGHT_FIELDS, label="Stage-B preflight receipt")
    return value


def validate_published_checkpoint_receipt(
    value: Mapping[str, Any],
    *,
    motion_adapter_sha256: str,
    stage_a_adapter_sha256: str,
) -> Mapping[str, Any]:
    """Validate a future Stage-B checkpoint receipt for inference.

    No such receipt exists today.  Keeping the schema here prevents an
    unqualified tensor file from later being treated as a trained editor.
    """

    row = _closed(value, _PUBLISHED_CHECKPOINT_FIELDS, label="Stage-B checkpoint receipt")
    _sealed_receipt(row, digest_field="receipt_digest", label="Stage-B checkpoint receipt")
    confirmation = row["confirmation_gate"]
    inference = row["inference_contract"]
    if not isinstance(confirmation, Mapping) or not isinstance(inference, Mapping):
        raise SAICStageBTrainingError("Stage-B checkpoint nested schema differs")
    expected_confirmation = {
        "dog_each_source_two_of_three_four_stage_event",
        "human_each_source_two_of_three_four_stage_event",
        "all_seven_axes_noninferior_to_frozen_base",
        "noop_exact",
        "correct_source_beats_wrong_and_drop",
        "camera_or_appearance_shortcut_rejected",
        "a1_inverse_ranking_beats_a0_on_multiple_unseen_sources",
        "all_seeds_reported",
    }
    expected_inference = {
        "online_motion_field_recomputed_each_step",
        "source_video_and_natural_language_only",
        "action_id_used",
        "mask_pose_flow_track_trajectory_used",
        "training_and_inference_route_identical",
    }
    if set(confirmation) != expected_confirmation or set(inference) != expected_inference:
        raise SAICStageBTrainingError("Stage-B publication gate schema differs")
    contracts = primitive_contracts()
    if (
        row["schema_version"] != PUBLISHED_CHECKPOINT_SCHEMA_VERSION
        or row["method"] != METHOD_NAME
        or row["complete"] is not True
        or row["publication_authorized"] is not True
        or row["world_size"] != WORLD_SIZE
        or row["data_parallel_size"] != DP_SIZE
        or row["sequence_parallel_size"] != SP_SIZE
        or row["frame_count"] != FRAME_COUNT
        or row["latent_phases"] != LATENT_PHASES
        or row["exact40_steps"] != EXACT40_STEPS
        or row["rollout_k"] != ROLLOUT_K
        or row["outer_cycles_completed"] != OUTER_CYCLES
        or type(row["optimizer_update_count"]) is not int
        or row["optimizer_update_count"] <= 0
        or row["motion_adapter_sha256"]
        != _sha(motion_adapter_sha256, bits=256, label="motion adapter digest")
        or row["source_anchor_adapter_sha256"]
        != _sha(stage_a_adapter_sha256, bits=256, label="Stage-A adapter digest")
        or row["action_operator_contract_digest"]
        != contracts["temporal_operator_contract_digest"]
        or row["online_motion_contract_digest"]
        != contracts["online_motion_contract_digest"]
        or any(value is not True for value in confirmation.values())
        or inference.get("online_motion_field_recomputed_each_step") is not True
        or inference.get("source_video_and_natural_language_only") is not True
        or inference.get("action_id_used") is not False
        or inference.get("mask_pose_flow_track_trajectory_used") is not False
        or inference.get("training_and_inference_route_identical") is not True
    ):
        raise SAICStageBTrainingError("Stage-B checkpoint is not publishable")
    for field in (
        "critic_checkpoint_sha256",
        "critic_qualification_receipt_digest",
        "motion_adapter_state_tensor_sha256",
        "action_operator_contract_digest",
        "online_motion_contract_digest",
        "method_source_archive_sha256",
    ):
        _sha(row[field], bits=256, label=field)
    _sha(row["method_source_revision"], bits=160, label="method_source_revision")
    return json.loads(canonical_json_bytes(row).decode("ascii"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256", required=True
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--stage-a-adapter", required=True)
    parser.add_argument("--expected-stage-a-adapter-sha256", required=True)
    parser.add_argument("--stage-a-receipt", required=True)
    parser.add_argument("--expected-stage-a-receipt-sha256", required=True)
    parser.add_argument("--critic-checkpoint", required=True)
    parser.add_argument("--expected-critic-checkpoint-sha256", required=True)
    parser.add_argument("--critic-qualification", required=True)
    parser.add_argument("--expected-critic-qualification-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=FRAME_COUNT)
    parser.add_argument("--rollout-k", type=int, default=ROLLOUT_K)
    parser.add_argument("--outer-cycles", type=int, default=OUTER_CYCLES)
    parser.add_argument(
        "--local-updates-per-cycle", type=int, default=LOCAL_UPDATES_PER_CYCLE
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--artifact-preflight-only", action="store_true")
    parser.add_argument("--runtime-preflight-only", action="store_true")
    parser.add_argument(
        "--ack-incomplete-runtime-preflight",
        action="store_true",
        help="Acknowledge that this revision cannot create an optimizer or train.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    if args.artifact_preflight_only and args.runtime_preflight_only:
        raise SAICStageBTrainingError("choose at most one preflight scope")
    # Resolve output now, but never create it in this incomplete revision.
    resolve_create_only_output(args.output)
    artifacts = preflight_artifacts(args)
    topology = None
    if not args.artifact_preflight_only:
        topology = validate_world8_accelerators(validate_world8_environment())
        topology = validate_world8_dp2sp4_collectives(args, topology)
    receipt = build_preflight_receipt(artifacts=artifacts, topology=topology)
    artifacts.assert_unchanged()
    if args.artifact_preflight_only or args.runtime_preflight_only:
        if topology is None or topology.get("rank") == 0:
            print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
        return 0
    blockers = tuple(receipt["runtime_blockers"])
    if not blockers:
        raise SAICStageBTrainingError(
            "capability audit unexpectedly reports complete, but no qualified execution body exists"
        )
    raise SAICStageBTrainingError(
        "Stage-B training is fail-closed before model/optimizer/output creation; "
        + " | ".join(blockers)
    )


__all__ = [
    "DP_SIZE",
    "EXACT40_STEPS",
    "FORBIDDEN_PUBLIC_ARGUMENTS",
    "FRAME_COUNT",
    "LATENT_PHASES",
    "METHOD_NAME",
    "OUTER_CYCLES",
    "PREFLIGHT_SCHEMA_VERSION",
    "PRESERVATION_AXES",
    "PUBLISHED_CHECKPOINT_SCHEMA_VERSION",
    "ROLLOUT_K",
    "SAICStageBTrainingError",
    "SP_SIZE",
    "StageBArtifacts",
    "UPDATE_INDICES",
    "WORLD_SIZE",
    "build_parser",
    "build_preflight_receipt",
    "canonical_json_bytes",
    "execution_plan",
    "main",
    "object_sha256",
    "primitive_contracts",
    "resolve_create_only_output",
    "runtime_blockers",
    "runtime_capability_audit",
    "validate_cli",
    "validate_published_checkpoint_receipt",
    "validate_stage_a_bundle",
    "validate_world8_environment",
    "validate_world8_accelerators",
    "validate_world8_dp2sp4_collectives",
]


if __name__ == "__main__":
    raise SystemExit(main())
