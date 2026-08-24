#!/usr/bin/env python3
"""Replay Phase-A in one WORLD8 process, then decode two exact81 diagnostics.

The upstream active14 job is an ``afterok`` queue gate only.  This runner calls
``replay_active14_for_downstream`` so short(29,38), field14(0..39 no-grad) and
active14(26..39 optimizer transaction) are reconstructed from the sealed base
source in this process.  The callback receives the still-live trained adapter
and performs one preregistered action rollout for each DP2 arm.  No upstream
checkpoint is consumed and no checkpoint is written.

Only fresh Gaussian initialization is accepted.  Each step uses the complete
confirmation source V-pack, the same source atlas, the pinned negative text and
the preregistered action text.  The exact40 UniPC state chain and the final
normalized latent are hashed on all four SP ranks.  One leader per arm decodes
exactly 81 frames at 25fps.  A staging tree is renamed to the final output only
after both arm artifacts, all eight packets and the parent receipt close.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import stat
import struct
from typing import Any, Mapping, Optional, Sequence

import torch

import graft_phase_a_full_exact81_decoded_v1 as full81
import infer_native_identity_generation_canary as native_generation
import infer_source_value_residual_oracle as value_audit
import run_graft_phase_a_a_lite_short_gpu_v1 as short_runner
import run_graft_phase_a_active14_transaction_gpu_v1 as active14_runner
import train_graft_phase_a_active14_transaction_v1 as active14_core


SCHEMA_VERSION = "bernini-graft-phase-a-full-exact81-decoded-gpu-runner-v1"
PLAN_SCHEMA_VERSION = "bernini-graft-phase-a-full-exact81-decoded-world8-plan-v1"
PARENT_SCHEMA_VERSION = "bernini-graft-phase-a-full-exact81-decoded-parent-v1"
ACTIVE14_PARENT_SCHEMA_VERSION = "bernini-graft-phase-a-active14-world8-parent-v1"
MAX_PACKET_BYTES = 64 * 1024 * 1024
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SOURCE_INPUT_HW_BY_DP_ARM = ((704, 736), (896, 704))
DERIVED_BUCKET_HW_BY_DP_ARM = ((480, 496), (544, 432))


class FullExact81GPUError(RuntimeError):
    """Fail closed without publishing a final output or a checkpoint."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_exact_plain_file(path_value: str | Path, *, expected_sha256: str) -> bytes:
    full81.require_sha256(expected_sha256, label="expected file SHA")
    path = Path(path_value)
    if not path.is_absolute():
        raise FullExact81GPUError("authenticated input path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise FullExact81GPUError(f"cannot open authenticated input {path}") from error
    try:
        before = os.fstat(fd)
        linked = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or (before.st_dev, before.st_ino, before.st_size)
            != (linked.st_dev, linked.st_ino, linked.st_size)
        ):
            raise FullExact81GPUError("authenticated input identity/mode differs")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise FullExact81GPUError("authenticated input changed or SHA differs")
        return raw
    finally:
        os.close(fd)


def build_parser() -> argparse.ArgumentParser:
    parser = active14_runner.build_parser()
    parser.description = __doc__
    parser.add_argument("--full-exact81-plan-path", required=True)
    parser.add_argument("--expected-full-exact81-plan-sha256", required=True)
    parser.add_argument("--expected-full-exact81-core-sha256", required=True)
    parser.add_argument("--expected-full-exact81-runner-sha256", required=True)
    parser.add_argument("--full-expected-active14-core-sha256", required=True)
    parser.add_argument("--full-expected-active14-runner-sha256", required=True)
    parser.add_argument("--full-expected-active14-source-commit", required=True)
    parser.add_argument("--full-expected-active14-plan-sha256", required=True)
    parser.add_argument("--full-expected-active14-launcher-sha256", required=True)
    parser.add_argument("--upstream-active14-receipt-path", required=True)
    parser.add_argument("--expected-upstream-active14-receipt-sha256", required=True)
    parser.add_argument("--expected-active14-job-id", required=True)
    parser.add_argument("--full-exact81-output-dir", required=True)
    parser.add_argument(
        "--ack-same-process-replay-no-checkpoint-operational-only",
        action="store_true",
    )
    return parser


def _load_plan(args: argparse.Namespace) -> Mapping[str, Any]:
    raw = _read_exact_plain_file(
        args.full_exact81_plan_path,
        expected_sha256=args.expected_full_exact81_plan_sha256,
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullExact81GPUError("full exact81 plan is not ASCII JSON") from error
    if raw != full81.canonical_json_bytes(value) + b"\n":
        raise FullExact81GPUError("full exact81 plan is not canonical newline JSON")
    expected_keys = {
        "conditions",
        "decoder",
        "schema_version",
        "dependency",
        "state_continuity",
        "replay_order",
        "sampling",
        "families",
        "source_release",
        "runtime",
        "topology",
        "resources",
        "fresh_output",
        "checkpoint_policy",
        "rollback_policy",
        "claim_scope",
        "authority",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise FullExact81GPUError("full exact81 plan root schema differs")
    dependency = value.get("dependency")
    continuity = value.get("state_continuity")
    sampling = value.get("sampling")
    conditions = value.get("conditions")
    decoder = value.get("decoder")
    runtime = value.get("runtime")
    resources = value.get("resources")
    if (
        value.get("schema_version") != PLAN_SCHEMA_VERSION
        or dependency
        != {
            "job_id": args.expected_active14_job_id,
            "kind": "afterok",
            "purpose": "queue-gate-only-no-weight-lineage",
            "receipt_path": args.upstream_active14_receipt_path,
            "receipt_sha256_policy": "derive-from-stable-sealed-file-after-afterok",
        }
        or not str(args.expected_active14_job_id).isdigit()
        or continuity
        != {
            "checkpoint_available_from_dependency": False,
            "dependency_transports_weights": False,
            "same_process_from_base_required": True,
            "short_29_38_replayed": True,
            "field14_0_39_replayed": True,
            "active14_26_39_replayed": True,
            "decode_continuation_before_restore": True,
        }
        or value.get("replay_order")
        != [
            "sealed-source-and-base-checkpoint-admission",
            "short-updates-29-38-confirmation-parity",
            "field14-no-grad-indices-0-39",
            "active14-transaction-indices-26-39",
            "full-action-exact40-rollout",
            "vae-exact81-decode",
            "world8-receipt-and-atomic-publish",
        ]
        or sampling
        != {
            "frame_count": full81.FRAME_COUNT,
            "latent_phases": full81.LATENT_PHASES,
            "fps_fraction": "25/1",
            "num_inference_steps": full81.NUM_INFERENCE_STEPS,
            "scheduler": "pinned-UniPC-flow-shift-5",
            "schedule_sha256": short_runner.sigma_strata.SCHEDULE_SHA256,
            "initial_state": "fresh-source-keyed-standard-gaussian",
            "source_condition": "full-confirmation-source-v-pack",
            "negative_condition": "pinned-renderer-negative",
            "positive_condition": "preregistered-family-action-text",
            "target_video_used": False,
            "clean_source_initial_latent_used": False,
            "best_of_n": False,
        }
        or conditions
        != {
            "negative": "pinned-renderer-negative-embedding-from-active14",
            "positive_by_family_utf8_sha256": {
                family: hashlib.sha256(
                    short_runner.ACTION_INSTRUCTION_BY_DP_ARM[arm].encode("utf-8")
                ).hexdigest()
                for arm, family in enumerate(full81.FAMILY_ORDER)
            },
            "positive_embedding_from_same_active14_condition_encoder": True,
            "source_atlas": "confirmation-row-full-frame-orderless-atlas",
            "source_v_pack": "confirmation-row-full-exact81-native-v2v",
            "condition_selection_or_retelling_used": False,
        }
        or decoder
        != {
            "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
            "checkpoint_content_manifest_sha256": (
                args.expected_checkpoint_content_manifest_sha256
            ),
            "class": "diffusers.models.AutoencoderKLWan",
            "subfolder": "vae",
            "load_dtype": "torch.float32",
            "decode_helper": "bernini.pipeline._vae_decode",
            "output_tensor_layout": "T,H,W,C",
            "output_frame_count": 81,
            "output_fps_fraction": "25/1",
            "latent_artifact_required": True,
            "mp4_artifact_required": True,
            "semantic_evaluator_present": False,
        }
        or value.get("topology")
        != {
            "allocation": "single-node-8xMI210",
            "world_size": 8,
            "dp_size": 2,
            "sp_size": 4,
        }
        or resources
        != {
            "nodes": 1,
            "ntasks": 1,
            "gpus": 8,
            "cpus_per_task": 64,
            "memory_gib": 256,
            "time_limit_hours": 72,
        }
        or value.get("checkpoint_policy")
        != {
            "dependency_checkpoint_consumed": False,
            "checkpoint_written": False,
            "checkpoint_publishable": False,
            "in_memory_weights_exist_only_until_process_exit": True,
        }
        or value.get("rollback_policy")
        != {
            "active14_owner_restores_adapter_and_process_group": True,
            "failed_staging_never_becomes_final": True,
            "no_partial_success_receipt": True,
            "failed_staging_is_non_authoritative_diagnostic_only": True,
        }
        or value.get("claim_scope")
        != "decoded-media-operational-integrity-only-no-action-identity-quality-or-scientific-authority"
        or value.get("authority") != full81.false_authority()
    ):
        raise FullExact81GPUError("full exact81 plan contract differs")
    expected_families = [
        {
            "dp_arm": arm,
            "family": family,
            "confirmation_iid": short_runner.CONFIRMATION_IID_BY_DP_ARM[arm],
            "action_instruction": short_runner.ACTION_INSTRUCTION_BY_DP_ARM[arm],
            "seed": (2308110001 + arm),
            "source_input_hw": list(SOURCE_INPUT_HW_BY_DP_ARM[arm]),
            "derived_bucket_hw": list(DERIVED_BUCKET_HW_BY_DP_ARM[arm]),
        }
        for arm, family in enumerate(full81.FAMILY_ORDER)
    ]
    if value.get("families") != expected_families:
        raise FullExact81GPUError("full exact81 family registration differs")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "full_exact81_core_sha256",
            "full_exact81_runner_sha256",
            "active14_core_sha256",
            "active14_runner_sha256",
            "active14_source_commit",
            "active14_plan_sha256",
            "active14_launcher_sha256",
            "bernini_commit",
            "veomni_commit",
        }
        or runtime.get("full_exact81_core_sha256")
        != args.expected_full_exact81_core_sha256
        or runtime.get("full_exact81_runner_sha256")
        != args.expected_full_exact81_runner_sha256
        or runtime.get("active14_core_sha256")
        != args.full_expected_active14_core_sha256
        or runtime.get("active14_runner_sha256")
        != args.full_expected_active14_runner_sha256
        or runtime.get("active14_source_commit")
        != args.full_expected_active14_source_commit
        or runtime.get("active14_plan_sha256")
        != args.full_expected_active14_plan_sha256
        or runtime.get("active14_launcher_sha256")
        != args.full_expected_active14_launcher_sha256
        or runtime.get("bernini_commit") != args.expected_bernini_commit
        or runtime.get("veomni_commit") != args.expected_veomni_commit
    ):
        raise FullExact81GPUError("full exact81 runtime pins differ")
    release = value.get("source_release")
    release_stem = (
        "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
        "VideoEdit_experiments/bernini_graft_v1_20260810/a_lite_source_release/"
        "10d9798_0c0aff6_c4_r1/outputs/canary4/graft_a_lite_source_canary4"
    )
    expected_release_artifacts = {
        "manifest": {
            "path": release_stem + ".manifest.jsonl",
            "sha256": "0bfe9447fb3649625a4dface5c93ba0e55530285f9d63e7b15552b0bb4f9b1af",
        },
        "producer": {
            "path": release_stem + ".receipt.json",
            "sha256": "01a87422753c5946c84df7c26198be992793de536159b1e427d4a00ab91ffea6",
        },
        "execution": {
            "path": release_stem + ".execution.receipt.json",
            "sha256": "101a9cb25a4c898be0ed1b0406a3aec866a153d12ff79c6b890fb78c1146efd6",
        },
        "submission": {
            "path": release_stem + ".submission.receipt.json",
            "sha256": "35ddc72443361ee588e5bd5c852df144293036a6128ae7fc9fab20e600e95b82",
        },
    }
    if (
        not isinstance(release, Mapping)
        or set(release)
        != {
            "artifacts",
            "cross_clip_identity_authority",
            "fps_fraction",
            "frame_count",
            "job_id",
            "same_clip_a_lite_only",
            "terminal_admission",
        }
        or release.get("job_id") != "132549"
        or release.get("frame_count") != 81
        or release.get("fps_fraction") != "25/1"
        or release.get("same_clip_a_lite_only") is not True
        or release.get("cross_clip_identity_authority") is not False
        or release.get("artifacts") != expected_release_artifacts
        or release.get("terminal_admission")
        != {
            "path": release_stem + ".terminal.admission.receipt.json",
            "slurm_state": "COMPLETED",
            "slurm_exit_code": "0:0",
        }
    ):
        raise FullExact81GPUError("full exact81 source release differs")
    fresh = value.get("fresh_output")
    if (
        fresh
        != {
            "create_only": True,
            "staging_then_atomic_rename": True,
            "artifacts_mode": "0444",
            "directories_mode": "0555",
            "exact81_media_probe_required": True,
            "latent_and_video_sha256_required": True,
            "reuse_prior_output_forbidden": True,
        }
    ):
        raise FullExact81GPUError("full exact81 output policy differs")
    return value


def _load_upstream_active14_parent(
    args: argparse.Namespace, plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    raw = _read_exact_plain_file(
        args.upstream_active14_receipt_path,
        expected_sha256=args.expected_upstream_active14_receipt_sha256,
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullExact81GPUError("active14 parent receipt is not ASCII JSON") from error
    if not isinstance(value, Mapping) or raw != full81.canonical_json_bytes(value) + b"\n":
        raise FullExact81GPUError("active14 parent receipt is not canonical newline JSON")
    expected_keys = {
        "schema_version",
        "status",
        "complete",
        "pass",
        "job_id",
        "field14_dependency_job_id",
        "field14_receipt_file_sha256",
        "runner_result_digest",
        "runtime",
        "validated",
        "runner_result",
        "checkpoint_written",
        "publication_performed",
        "authority",
        "receipt_digest",
    }
    if set(value) != expected_keys:
        raise FullExact81GPUError("active14 parent receipt schema differs")
    receipt_digest = full81.require_sha256(
        value.get("receipt_digest"), label="active14 parent receipt digest"
    )
    core = {key: item for key, item in value.items() if key != "receipt_digest"}
    if full81.object_sha256(core) != receipt_digest:
        raise FullExact81GPUError("active14 parent receipt digest differs")
    dependency = plan.get("dependency")
    if (
        value.get("schema_version") != ACTIVE14_PARENT_SCHEMA_VERSION
        or value.get("status") != "completed_operational_no_checkpoint"
        or value.get("complete") is not True
        or value.get("pass") is not True
        or value.get("job_id") != args.expected_active14_job_id
        or value.get("field14_dependency_job_id")
        != args.expected_upstream_field14_job_id
        or value.get("field14_receipt_file_sha256")
        != args.expected_upstream_field14_receipt_sha256
        or value.get("checkpoint_written") is not False
        or value.get("publication_performed") is not False
        or not isinstance(value.get("authority"), Mapping)
        or any(value["authority"].values())
        or dependency.get("receipt_path") != args.upstream_active14_receipt_path
        or dependency.get("receipt_sha256_policy")
        != "derive-from-stable-sealed-file-after-afterok"
    ):
        raise FullExact81GPUError("active14 parent receipt admission differs")
    runtime = value.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "source_git_commit",
            "active14_core_sha256",
            "active14_runner_sha256",
            "active14_plan_sha256",
            "source_archive_sha256",
            "runtime_closure_manifest_sha256",
            "launcher_sha256",
        }
        or runtime.get("source_git_commit")
        != args.full_expected_active14_source_commit
        or runtime.get("active14_core_sha256")
        != args.full_expected_active14_core_sha256
        or runtime.get("active14_runner_sha256")
        != args.full_expected_active14_runner_sha256
        or runtime.get("active14_plan_sha256")
        != args.full_expected_active14_plan_sha256
        or runtime.get("launcher_sha256")
        != args.full_expected_active14_launcher_sha256
    ):
        raise FullExact81GPUError("active14 parent runtime binding differs")
    full81.require_sha256(
        runtime.get("source_archive_sha256"), label="active14 source archive SHA"
    )
    full81.require_sha256(
        runtime.get("runtime_closure_manifest_sha256"),
        label="active14 runtime closure SHA",
    )
    result = active14_core.validate_sealed_mapping(
        value.get("runner_result"), label="active14 parent runner result"
    )
    active14_core.assert_no_authority(result)
    source_binding = active14_core.validate_sealed_mapping(
        result.get("source_binding"), label="active14 parent source binding"
    )
    qualification = active14_core.validate_sealed_mapping(
        result.get("upstream_qualification"),
        label="active14 parent upstream qualification",
    )
    if (
        value.get("runner_result_digest") != result.get("digest")
        or result.get("status")
        != "completed_same_process_short_field14_active14_no_checkpoint"
        or result.get("dependency_afterok_is_queue_gate_only") is not True
        or result.get("weights_inherited_from_dependency_job") is not False
        or result.get("checkpoint_written") is not False
        or result.get("publication_performed") is not False
        or source_binding.get("active14_core_sha256")
        != args.full_expected_active14_core_sha256
        or source_binding.get("active14_runner_sha256")
        != args.full_expected_active14_runner_sha256
        or source_binding.get("plan_sha256")
        != args.full_expected_active14_plan_sha256
        or qualification.get("field14_job_id")
        != args.expected_upstream_field14_job_id
        or qualification.get("weights_inherited_from_dependency_job") is not False
    ):
        raise FullExact81GPUError("active14 parent runner evidence differs")
    return {
        "path": args.upstream_active14_receipt_path,
        "file_sha256": args.expected_upstream_active14_receipt_sha256,
        "receipt_digest": receipt_digest,
    }


def validate_cli(args: argparse.Namespace) -> argparse.Namespace:
    active14_runner.validate_cli(args)
    for name in (
        "expected_full_exact81_plan_sha256",
        "expected_full_exact81_core_sha256",
        "expected_full_exact81_runner_sha256",
        "full_expected_active14_core_sha256",
        "full_expected_active14_runner_sha256",
        "full_expected_active14_plan_sha256",
        "full_expected_active14_launcher_sha256",
        "expected_upstream_active14_receipt_sha256",
    ):
        full81.require_sha256(getattr(args, name), label=name)
    if (
        file_sha256(full81.__file__) != args.expected_full_exact81_core_sha256
        or file_sha256(__file__) != args.expected_full_exact81_runner_sha256
        or file_sha256(active14_runner.__file__)
        != args.full_expected_active14_runner_sha256
        or not args.ack_same_process_replay_no_checkpoint_operational_only
    ):
        raise FullExact81GPUError("full exact81 source pin/acknowledgement differs")
    if re.fullmatch(r"[0-9a-f]{40}", args.full_expected_active14_source_commit) is None:
        raise FullExact81GPUError("active14 source commit pin differs")
    output = Path(args.full_exact81_output_dir)
    if (
        not output.is_absolute()
        or output == Path("/")
        or _SAFE_NAME.fullmatch(output.name) is None
        or output.exists()
    ):
        raise FullExact81GPUError("full exact81 final output must be fresh absolute safe path")
    _load_plan(args)
    return args


def _safe_stage_paths(output: Path) -> tuple[Path, Path]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise FullExact81GPUError("full exact81 requires a numeric Slurm job id")
    stage = _stage_path_for_output(output, job_id=job_id)
    if stage.exists() or output.exists() or stage.parent != output.parent:
        raise FullExact81GPUError("full exact81 staging/final path is not fresh")
    return stage, output


def _stage_path_for_output(output: Path, *, job_id: str) -> Path:
    if not job_id.isdigit():
        raise FullExact81GPUError("full exact81 stage job id must be numeric")
    return output.with_name(f".{output.name}.staging-job{job_id}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_create_only(path: Path, raw: bytes, *, final_mode: int = 0o444) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path, flags, 0o400)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise FullExact81GPUError("create-only write made no progress")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, final_mode)
        os.fsync(fd)
        if os.fstat(fd).st_nlink != 1:
            raise FullExact81GPUError("create-only artifact link count differs")
    finally:
        os.close(fd)


def _seal_artifact(
    path: Path,
    *,
    role: str,
    staging_root: Path,
    final_root: Path,
    content_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    if path.parent.parent != staging_root or path.is_symlink():
        raise FullExact81GPUError("artifact lies outside one family staging directory")
    os.chmod(path, 0o444)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        linked = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or (before.st_dev, before.st_ino, before.st_size)
            != (linked.st_dev, linked.st_ino, linked.st_size)
        ):
            raise FullExact81GPUError("sealed artifact identity differs")
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(fd)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise FullExact81GPUError("sealed artifact changed while hashing")
        os.fsync(fd)
    finally:
        os.close(fd)
    relative = path.relative_to(staging_root)
    return {
        "schema_version": full81.ARTIFACT_SCHEMA_VERSION,
        "role": role,
        "path": str(final_root / relative),
        "relative_path": relative.as_posix(),
        "size_bytes": before.st_size,
        "mode": "0444",
        "regular_file": True,
        "link_count_one": True,
        "sha256": digest.hexdigest(),
        "opened_nofollow_and_revalidated": True,
        "content_binding": dict(content_binding),
    }


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    if (
        not source.is_absolute()
        or not target.is_absolute()
        or source.parent != target.parent
        or source == target
    ):
        raise FullExact81GPUError("atomic publish paths differ")
    before = source.lstat()
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise FullExact81GPUError("atomic publish source is not a real directory")
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FullExact81GPUError("atomic publish target already exists")
    parent = source.parent
    parent_before = parent.lstat()
    if not stat.S_ISDIR(parent_before.st_mode) or stat.S_ISLNK(parent_before.st_mode):
        raise FullExact81GPUError("atomic publish parent is not a real directory")
    parent_fd = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        parent_opened = os.fstat(parent_fd)
        parent_identity = (parent_opened.st_dev, parent_opened.st_ino)
        if parent_identity != (parent_before.st_dev, parent_before.st_ino):
            raise FullExact81GPUError("atomic publish parent changed while opening")
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise FullExact81GPUError("renameat2 no-replace is unavailable")
        else:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            result = renameat2(
                parent_fd,
                os.fsencode(source.name),
                parent_fd,
                os.fsencode(target.name),
                1,
            )
            error_number = 0 if result == 0 else ctypes.get_errno()
        unsupported = {
            errno.EINVAL,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        }
        if error_number in unsupported:
            reservation_fd: Optional[int] = None
            reservation_identity: Optional[tuple[int, ...]] = None
            try:
                os.mkdir(target.name, 0o700, dir_fd=parent_fd)
                reservation_fd = os.open(
                    target.name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                reservation = os.fstat(reservation_fd)
                if (
                    not stat.S_ISDIR(reservation.st_mode)
                    or stat.S_ISLNK(reservation.st_mode)
                    or stat.S_IMODE(reservation.st_mode) != 0o700
                    or reservation.st_nlink < 2
                    or os.listdir(reservation_fd)
                ):
                    raise FullExact81GPUError(
                        "atomic publish reservation admission differs"
                    )
                reservation_identity = (
                    reservation.st_dev,
                    reservation.st_ino,
                    reservation.st_mode,
                    reservation.st_nlink,
                )
                source_now = os.stat(
                    source.name, dir_fd=parent_fd, follow_symlinks=False
                )
                target_now = os.stat(
                    target.name, dir_fd=parent_fd, follow_symlinks=False
                )
                parent_now = os.fstat(parent_fd)
                if (
                    (source_now.st_dev, source_now.st_ino, source_now.st_mode, source_now.st_nlink)
                    != (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
                    or (
                        target_now.st_dev,
                        target_now.st_ino,
                        target_now.st_mode,
                        target_now.st_nlink,
                    )
                    != reservation_identity
                    or (parent_now.st_dev, parent_now.st_ino) != parent_identity
                    or os.listdir(reservation_fd)
                ):
                    raise FullExact81GPUError(
                        "atomic publish reservation changed before rename"
                    )
                # The exact final-path directory is the publication lock:
                # mkdir is create-only, and its admitted inode is retained open.
                # Make it inert only after the final empty-directory check;
                # Lustre rejects listdir(2) through an FD after mode 000.
                os.fchmod(reservation_fd, 0o000)
                reservation = os.fstat(reservation_fd)
                reservation_identity = (
                    reservation.st_dev,
                    reservation.st_ino,
                    reservation.st_mode,
                    reservation.st_nlink,
                )
                source_now = os.stat(
                    source.name, dir_fd=parent_fd, follow_symlinks=False
                )
                target_now = os.stat(
                    target.name, dir_fd=parent_fd, follow_symlinks=False
                )
                parent_now = os.fstat(parent_fd)
                if (
                    stat.S_IMODE(reservation.st_mode) != 0o000
                    or (
                        source_now.st_dev,
                        source_now.st_ino,
                        source_now.st_mode,
                        source_now.st_nlink,
                    )
                    != (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
                    or (
                        target_now.st_dev,
                        target_now.st_ino,
                        target_now.st_mode,
                        target_now.st_nlink,
                    )
                    != reservation_identity
                    or (parent_now.st_dev, parent_now.st_ino) != parent_identity
                ):
                    raise FullExact81GPUError(
                        "atomic publish inert reservation changed before rename"
                    )
                os.fsync(parent_fd)
                os.rename(
                    source.name,
                    target.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                error_number = 0
            finally:
                # On any fallback failure, retain the reservation exactly as
                # failure evidence.  The caller quarantines the target and
                # stage under job-unique names; this function never performs
                # a path-based cleanup that could remove a replaced inode.
                if reservation_fd is not None:
                    os.close(reservation_fd)
        if error_number != 0:
            raise FullExact81GPUError(
                f"atomic no-replace publish failed errno={error_number}"
            )
        os.fsync(parent_fd)
        parent_after = parent.lstat()
        if (parent_after.st_dev, parent_after.st_ino) != parent_identity:
            raise FullExact81GPUError("atomic publish parent identity changed")
    finally:
        os.close(parent_fd)
    after = target.lstat()
    if source.exists() or source.is_symlink() or (
        before.st_dev,
        before.st_ino,
    ) != (after.st_dev, after.st_ino):
        raise FullExact81GPUError("atomic publish identity differs after rename")


def _route_trace_projection(
    route: Any, *, context: Any, coordinate: Any
) -> Mapping[str, Any]:
    try:
        raw = full81.validate_sealed_mapping(
            route.receipt(), label="rank-local exact40 route receipt"
        )
    except (AttributeError, full81.FullExact81ContractError) as error:
        raise FullExact81GPUError("rank-local route receipt is not sealed") from error
    expected_keys = {
        "branch_name",
        "total_tokens",
        "condition_tokens",
        "target_tokens",
        "sequence_parallel_rank",
        "sequence_parallel_size",
        "sigma_hex",
        "gate_hex",
        "atlas_receipt_digest",
        "source_memory_owned_by_V_VI_only",
        "enabled",
        "digest",
    }
    total_tokens = raw.get("total_tokens")
    condition_tokens = raw.get("condition_tokens")
    schedule_index = coordinate.schedule_index
    sigma = float(coordinate.sigma.item())
    expected_gate = short_runner.rebinder.mid_low_sigma_gate(sigma)
    if (
        set(raw) != expected_keys
        or type(schedule_index) is not int
        or schedule_index not in range(full81.NUM_INFERENCE_STEPS)
        or sigma
        != short_runner.sigma_strata.PINNED_POSITIVE_SIGMAS[schedule_index]
        or raw.get("branch_name") != "V"
        or type(total_tokens) is not int
        or total_tokens <= 0
        or type(condition_tokens) is not int
        or not 0 <= condition_tokens < total_tokens
        or raw.get("target_tokens") != total_tokens - condition_tokens
        or raw.get("sequence_parallel_rank") != context.topology.sp_rank
        or raw.get("sequence_parallel_size") != full81.SP_SIZE
        or raw.get("sigma_hex") != sigma.hex()
        or raw.get("gate_hex") != expected_gate.hex()
        or not isinstance(raw.get("atlas_receipt_digest"), str)
        or len(raw["atlas_receipt_digest"]) != 64
        or any(character not in "0123456789abcdef" for character in raw["atlas_receipt_digest"])
        or raw.get("source_memory_owned_by_V_VI_only") is not True
        or raw.get("enabled") is not True
        or (schedule_index < 26 and expected_gate != 0.0)
        or (
            schedule_index >= 26
            and (not math.isfinite(expected_gate) or expected_gate <= 0.0)
        )
    ):
        raise FullExact81GPUError("rank-local route receipt contract differs")
    projection = full81.seal_mapping(
        {
            "schema_version": (
                "bernini-graft-phase-a-full-exact81-route-projection-v1"
            ),
            "schedule_index": schedule_index,
            "branch_name": raw["branch_name"],
            "total_tokens": total_tokens,
            "condition_tokens": condition_tokens,
            "target_tokens": raw["target_tokens"],
            "sequence_parallel_size": raw["sequence_parallel_size"],
            "sigma_hex": raw["sigma_hex"],
            "gate_hex": raw["gate_hex"],
            "atlas_receipt_digest": raw["atlas_receipt_digest"],
            "source_memory_owned_by_V_VI_only": True,
            "enabled": True,
            "all_sp4_ranks_apply_same_global_route": True,
            "local_rank_validated_before_projection": True,
            "rank_specific_receipt_digest_not_cross_rank_comparable": True,
        }
    )
    try:
        full81._validate_route_projection(  # noqa: SLF001
            projection,
            schedule_index=schedule_index,
            sigma_float32_be_hex=struct.pack(">f", sigma).hex(),
        )
    except full81.FullExact81ContractError as error:
        raise FullExact81GPUError("rank-invariant route projection differs") from error
    return projection


def _step_state(
    *,
    context: Any,
    state: torch.Tensor,
    coordinate: Any,
    atlas: Any,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    pack = short_runner._build_native_pack(  # noqa: SLF001
        transformer=context.transformer,
        source_latent=context.confirmation.source_latent,
        noisy_target=state,
    )
    negative_route, negative_context = short_runner._route_for_pack(  # noqa: SLF001
        handle=context.handle,
        pack=pack,
        sp_rank=context.topology.sp_rank,
        sigma=float(coordinate.sigma.item()),
        atlas=atlas,
        mode="atlas",
    )
    negative_raw = short_runner._native_raw_forward(  # noqa: SLF001
        diffusion=context.diffusion,
        pack=pack,
        coordinate=coordinate,
        condition=context.negative_condition,
        route_context=negative_context,
    )
    action_route, action_context = short_runner._route_for_pack(  # noqa: SLF001
        handle=context.handle,
        pack=pack,
        sp_rank=context.topology.sp_rank,
        sigma=float(coordinate.sigma.item()),
        atlas=atlas,
        mode="atlas",
    )
    action_raw = short_runner._native_raw_forward(  # noqa: SLF001
        diffusion=context.diffusion,
        pack=pack,
        coordinate=coordinate,
        condition=context.action_condition,
        route_context=action_context,
    )
    negative_route_projection = _route_trace_projection(
        negative_route, context=context, coordinate=coordinate
    )
    action_route_projection = _route_trace_projection(
        action_route, context=context, coordinate=coordinate
    )
    if negative_route_projection != action_route_projection:
        raise FullExact81GPUError("negative/action route projections differ")
    velocity = short_runner._guided_velocity_from_raw(  # noqa: SLF001
        bindings=context.bindings,
        noisy_target=state,
        coordinate=coordinate,
        pack=pack,
        negative_raw=negative_raw,
        positive_raw=action_raw,
    )
    before = short_runner.tensor_identity(state)
    try:
        stepped = context.diffusion.scheduler.step(
            velocity,
            coordinate.timestep,
            state,
            return_dict=False,
        )
    except Exception as error:
        raise FullExact81GPUError("pinned UniPC scheduler step failed") from error
    if not isinstance(stepped, tuple) or len(stepped) < 1:
        raise FullExact81GPUError("pinned UniPC scheduler return differs")
    next_state = stepped[0]
    if (
        type(next_state) is not torch.Tensor
        or next_state.shape != state.shape
        or next_state.device != state.device
        or next_state.requires_grad
        or next_state.grad_fn is not None
        or not bool(torch.isfinite(next_state).all().item())
    ):
        raise FullExact81GPUError("exact40 next state differs")
    next_state = next_state.float().contiguous()
    row = {
        "schedule_index": coordinate.schedule_index,
        "timestep": int(coordinate.timestep.item()),
        "sigma_float32_be_hex": struct.pack(
            ">f", float(coordinate.sigma.item())
        ).hex(),
        "state_before": before,
        "native_visual_pack": pack.visual_identity,
        "native_rotary_pack": pack.rotary_identity,
        "negative_raw": short_runner.tensor_identity(negative_raw),
        "action_raw": short_runner.tensor_identity(action_raw),
        "guided_velocity": short_runner.tensor_identity(velocity),
        "state_after": short_runner.tensor_identity(next_state),
        "route_receipts": {
            "negative": dict(negative_route_projection),
            "action": dict(action_route_projection),
        },
        "scheduler_step_call_count": 1,
        "source_conditioned": True,
        "action_positive_condition": True,
        "target_video_used": False,
        "clean_source_initial_latent_used": False,
    }
    return next_state, row


def _exact40_action_rollout(context: Any, *, seed: int) -> tuple[Any, ...]:
    # Reset the official scheduler exactly once after all no-step field work.
    schedule = short_runner.Exact40CoordinateRegistry(
        context.diffusion.scheduler, device=context.device
    )
    noise = short_runner.keyed_fresh_gaussian(
        shape=context.confirmation.source_latent.shape,
        device=context.device,
        source_sha256=context.confirmation.row.source_sha256,
        purpose=f"full-exact81-action-seed-{seed}",
        schedule_index=0,
    )
    # The keyed generator includes the preregistered seed in its purpose; the
    # source never participates in x_1 and is only used as native conditioning.
    state = noise.epsilon.float().detach().clone().contiguous()
    rows = []
    with torch.no_grad():
        atlas = context.handle.build_atlas(
            context.confirmation.atlas_frames,
            source_video_sha256=context.confirmation.row.source_sha256,
        )
        if (
            type(atlas.tokens) is not torch.Tensor
            or atlas.tokens.requires_grad
            or atlas.tokens.grad_fn is not None
            or not bool(torch.isfinite(atlas.tokens).all().item())
        ):
            raise FullExact81GPUError("full rollout atlas retained a graph")
        for index in range(full81.NUM_INFERENCE_STEPS):
            coordinate = schedule.coordinate(index)
            state, row = _step_state(
                context=context,
                state=state,
                coordinate=coordinate,
                atlas=atlas,
            )
            rows.append(row)
    if getattr(context.diffusion.scheduler, "step_index", None) != 40:
        raise FullExact81GPUError("exact40 scheduler did not advance exactly 40 times")
    trace = full81.seal_mapping(
        {
            "schema_version": full81.TRACE_SCHEMA_VERSION,
            "rows": rows,
            "official_unipc_step_count": 40,
            "initial_state_role": "fresh-source-keyed-standard-gaussian",
            "source_condition_role": "full-confirmation-source-v-pack",
            "positive_condition_role": "preregistered-action-text",
            "same_gaussian_seed_across_sp4": True,
            "cross_index_selection_used": False,
            "checkpoint_loaded_from_dependency": False,
            "checkpoint_written": False,
        }
    )
    full81.validate_exact40_trace(trace)
    return state, noise, trace


def _decode_and_seal_arm(
    *,
    context: Any,
    endpoint: torch.Tensor,
    staging_root: Path,
    final_root: Path,
    expected_height: int,
    expected_width: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    from diffusers.models import AutoencoderKLWan
    from bernini.io_utils import save_output
    from bernini.pipeline import _vae_decode
    from safetensors import safe_open
    from tools import materialize_vae

    family = full81.FAMILY_ORDER[context.topology.dp_arm]
    family_stage = staging_root / family
    os.mkdir(family_stage, 0o700)
    latent_path = family_stage / "endpoint.normalized-clean-latent.safetensors"
    video_path = family_stage / "decoded-exact81.mp4"
    if tuple(int(item) for item in endpoint.shape) != (
        1,
        16,
        full81.LATENT_PHASES,
        expected_height // 8,
        expected_width // 8,
    ):
        raise FullExact81GPUError("exact81 endpoint bucket geometry differs")
    endpoint_before = short_runner.tensor_identity(endpoint)
    latent_roundtrip = native_generation._save_normalized_clean_latent_atomically(
        latent_path,
        endpoint,
        artifact_role="native_sampler_proposal",
    )
    with safe_open(str(latent_path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != ["normalized_clean_latent"]:
            raise FullExact81GPUError("staged latent safetensors key differs")
        restored = opened.get_tensor("normalized_clean_latent").contiguous()
        latent_metadata = dict(opened.metadata() or {})
    restored_identity = short_runner.tensor_identity(restored)
    if (
        latent_roundtrip.get("path") != str(latent_path)
        or latent_roundtrip.get("tensor_key") != "normalized_clean_latent"
        or latent_roundtrip.get("shape") != endpoint_before["shape"]
        or latent_roundtrip.get("stored_dtype") != endpoint_before["dtype"]
        or latent_roundtrip.get("artifact_role") != "native_sampler_proposal"
        or latent_roundtrip.get("roundtrip_byte_exact_fp32") is not True
        or latent_metadata
        != {
            "coordinate": "bernini_normalized_clean_vae_latent",
            "frame_contract": "exact81_latent21",
            "artifact_role": "native_sampler_proposal",
            "source": "native_sampler_before_vae_decode",
        }
        or restored_identity["shape"] != endpoint_before["shape"]
        or restored_identity["dtype"] != endpoint_before["dtype"]
        or restored_identity["raw_sha256"] != endpoint_before["raw_sha256"]
        or restored_identity["content_sha256"]
        != endpoint_before["content_sha256"]
    ):
        raise FullExact81GPUError("staged latent does not bind exact endpoint")
    vae = AutoencoderKLWan.from_pretrained(
        str(context.checkpoint_root),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False).to(context.device)
    try:
        with torch.no_grad():
            decoded = _vae_decode(vae, endpoint)
        endpoint_after_decode = short_runner.tensor_identity(endpoint)
        if endpoint_after_decode != endpoint_before:
            raise FullExact81GPUError("VAE decode mutated exact40 endpoint")
        if tuple(int(item) for item in decoded.shape) != (
            full81.FRAME_COUNT,
            expected_height,
            expected_width,
            3,
        ):
            raise FullExact81GPUError("VAE decoded exact81 shape differs")
        value_audit.save_video_atomically(
            decoded,
            video_path,
            fps=full81.FPS_NUMERATOR,
            save_output_fn=save_output,
        )
        encoded, _encoded_fps, encoded_hw = materialize_vae._decode_exact_video(
            video_path
        )
        if (
            int(encoded.shape[0]) != full81.FRAME_COUNT
            or tuple(int(item) for item in encoded_hw)
            != (expected_height, expected_width)
        ):
            raise FullExact81GPUError("encoded exact81 media probe differs")
        opened_video = None
        opened_ffprobe = None
        try:
            opened_video = short_runner.source_consumer._open_source(  # noqa: SLF001
                str(video_path), label="full exact81 staged video"
            )
            opened_ffprobe = short_runner.source_consumer._open_frozen_ffprobe()  # noqa: SLF001
            probe = short_runner.source_consumer._probe_with_frozen_ffprobe(  # noqa: SLF001
                opened_video, opened_ffprobe
            )
            exact_media = short_runner.source_consumer._normalize_probe(  # noqa: SLF001
                probe, label="full exact81 staged video"
            )
        finally:
            if opened_video is not None:
                os.close(opened_video.fd)
                os.close(opened_video.parent_fd)
            if opened_ffprobe is not None:
                os.close(opened_ffprobe.fd)
        if (
            exact_media.frame_count != full81.FRAME_COUNT
            or exact_media.fps_numerator != full81.FPS_NUMERATOR
            or exact_media.fps_denominator != full81.FPS_DENOMINATOR
            or exact_media.height != expected_height
            or exact_media.width != expected_width
        ):
            raise FullExact81GPUError("portable ffprobe exact81 media differs")
        media = {
            "schema_version": full81.MEDIA_SCHEMA_VERSION,
            "frame_count": full81.FRAME_COUNT,
            "fps_numerator": full81.FPS_NUMERATOR,
            "fps_denominator": full81.FPS_DENOMINATOR,
            "reported_fps_numerator": full81.FPS_NUMERATOR,
            "reported_fps_denominator": full81.FPS_DENOMINATOR,
            "height": expected_height,
            "width": expected_width,
            "decoded_tensor_shape": [
                full81.FRAME_COUNT,
                expected_height,
                expected_width,
                3,
            ],
            "codec_content_interpreted_for_semantics": False,
        }
    finally:
        vae.to("cpu")
        del vae
        torch.cuda.empty_cache()
    latent = _seal_artifact(
        latent_path,
        role="normalized-clean-latent",
        staging_root=staging_root,
        final_root=final_root,
        content_binding={
            "kind": "safetensors-exact-endpoint-tensor",
            "tensor_key": "normalized_clean_latent",
            "tensor_shape": restored_identity["shape"],
            "tensor_dtype": restored_identity["dtype"],
            "tensor_raw_sha256": restored_identity["raw_sha256"],
            "tensor_content_sha256": restored_identity["content_sha256"],
            "endpoint_raw_sha256": endpoint_before["raw_sha256"],
            "endpoint_content_sha256": endpoint_before["content_sha256"],
            "safetensors_roundtrip_verified": True,
        },
    )
    if latent["sha256"] != latent_roundtrip.get("sha256"):
        raise FullExact81GPUError("sealed latent file SHA differs from roundtrip")
    video = _seal_artifact(
        video_path,
        role="decoded-exact81-video",
        staging_root=staging_root,
        final_root=final_root,
        content_binding={
            "kind": "same-call-vae-decode-from-sealed-endpoint",
            "decoded_from_endpoint_raw_sha256": endpoint_before["raw_sha256"],
            "decoded_from_endpoint_content_sha256": endpoint_before[
                "content_sha256"
            ],
            "endpoint_unchanged_after_decode": True,
            "semantic_content_interpreted": False,
        },
    )
    os.chmod(family_stage, 0o555)
    _fsync_directory(family_stage)
    return media, latent, video


def _validate_continuation_context(context: Any) -> Any:
    required = (
        "topology",
        "backend",
        "diffusion",
        "transformer",
        "renderer",
        "handle",
        "bindings",
        "schedule",
        "fit",
        "confirmation",
        "negative_condition",
        "noop_condition",
        "action_condition",
        "short_receipt",
        "field14_receipt",
        "device",
        "local_rank",
        "bernini_revision",
        "checkpoint_root",
        "checkpoint_content_identity",
        "checkpoint_manifest_sha256",
        "trainable_final_digest",
        "frozen_base_final_digest",
    )
    if any(not hasattr(context, name) for name in required):
        raise FullExact81GPUError("active14 continuation context is incomplete")
    topology = context.topology
    if (
        topology.global_rank not in range(8)
        or topology.dp_arm not in range(2)
        or topology.sp_rank not in range(4)
        or context.confirmation.row.iid
        != short_runner.CONFIRMATION_IID_BY_DP_ARM[topology.dp_arm]
    ):
        raise FullExact81GPUError("active14 continuation topology differs")
    return topology


def _validate_continuation_geometry(
    context: Any, family_plan: Mapping[str, Any]
) -> tuple[int, int]:
    dp_arm = context.topology.dp_arm
    expected_input = SOURCE_INPUT_HW_BY_DP_ARM[dp_arm]
    expected_bucket = DERIVED_BUCKET_HW_BY_DP_ARM[dp_arm]
    metadata = context.confirmation.metadata
    try:
        observed_input = tuple(int(item) for item in metadata["source_input_hw"])
        observed_bucket = tuple(
            int(item) for item in metadata["source_derived_bucket_hw"]
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise FullExact81GPUError("confirmation geometry metadata differs") from error
    if (
        not isinstance(metadata, Mapping)
        or observed_input != expected_input
        or observed_bucket != expected_bucket
        or family_plan.get("source_input_hw") != list(expected_input)
        or family_plan.get("derived_bucket_hw") != list(expected_bucket)
        or (
            context.confirmation.row.media.height,
            context.confirmation.row.media.width,
        )
        != expected_input
        or tuple(int(item) for item in context.confirmation.source_latent.shape)
        != (1, 16, full81.LATENT_PHASES, expected_bucket[0] // 8, expected_bucket[1] // 8)
    ):
        raise FullExact81GPUError("confirmation source/bucket geometry differs")
    return expected_bucket


def _callbacks_factory(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    upstream_active14_parent: Mapping[str, Any],
):
    output = Path(args.full_exact81_output_dir)
    prepared: dict[str, Any] = {}

    def prepare(
        context: Any, active14_precommit_receipt: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        import torch.distributed as dist

        topology = _validate_continuation_context(context)
        precommit = active14_core.validate_sealed_mapping(
            active14_precommit_receipt, label="active14 precommit receipt"
        )
        active14_core.assert_no_authority(precommit)
        if (
            precommit.get("status")
            != "active14_updates_complete_downstream_prepare_pending"
            or precommit.get("checkpoint_written") is not False
            or precommit.get("publication_performed") is not False
        ):
            raise FullExact81GPUError("active14 precommit contract differs")
        family_plan = plan["families"][topology.dp_arm]
        expected_height, expected_width = _validate_continuation_geometry(
            context, family_plan
        )
        seed = family_plan["seed"]
        job_id = os.environ.get("SLURM_JOB_ID", "")
        if not job_id.isdigit():
            raise FullExact81GPUError("full exact81 requires a numeric Slurm job id")
        stage = _stage_path_for_output(output, job_id=job_id)
        final = output
        stage_box: list[Any] = [None]
        if topology.global_rank == 0:
            try:
                if (
                    stage.exists()
                    or stage.is_symlink()
                    or final.exists()
                    or final.is_symlink()
                    or stage.parent != final.parent
                ):
                    raise FullExact81GPUError(
                        "full exact81 staging/final path is not fresh"
                    )
                os.mkdir(stage, 0o700)
                _fsync_directory(stage.parent)
                stage_box[0] = {
                    "ok": True,
                    "stage": str(stage),
                    "final": str(final),
                }
            except BaseException as error:
                stage_box[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}:{error}",
                }
        dist.broadcast_object_list(
            stage_box, src=0, group=topology.world_group
        )
        if stage_box[0] != {
            "ok": True,
            "stage": str(stage),
            "final": str(final),
        }:
            raise FullExact81GPUError("shared staging creation failed")
        if not stage.is_dir() or final.exists():
            raise FullExact81GPUError("shared staging creation differs")

        trainable_before = context.trainable_final_digest
        base_before = context.frozen_base_final_digest
        endpoint, noise, trace = _exact40_action_rollout(context, seed=seed)
        trainable_after_rollout = active14_core._registry_digest(  # noqa: SLF001
            active14_core._trainable_registry(context.bindings)  # noqa: SLF001
        )
        base_after_rollout = active14_core._registry_digest(  # noqa: SLF001
            active14_core._frozen_registry(context.bindings)  # noqa: SLF001
        )
        if (
            trainable_before != trainable_after_rollout
            or base_before != base_after_rollout
        ):
            raise FullExact81GPUError("exact81 rollout changed parameter bytes")
        endpoint_identity = short_runner.tensor_identity(endpoint)
        noise_identity = short_runner.tensor_identity(noise.epsilon)
        short_runner._gather_equal(  # noqa: SLF001
            endpoint_identity,
            group=topology.sp_group,
            count=4,
            label=f"{full81.FAMILY_ORDER[topology.dp_arm]} exact81 endpoint",
        )
        short_runner._gather_equal(  # noqa: SLF001
            dict(trace),
            group=topology.sp_group,
            count=4,
            label=f"{full81.FAMILY_ORDER[topology.dp_arm]} exact40 trace",
        )
        artifact_box: list[Any] = [None]
        if topology.sp_rank == 0:
            try:
                media, latent, video = _decode_and_seal_arm(
                    context=context,
                    endpoint=endpoint,
                    staging_root=stage,
                    final_root=final,
                    expected_height=expected_height,
                    expected_width=expected_width,
                )
                artifact_box[0] = {
                    "ok": True,
                    "artifacts": {
                        "media": media,
                        "latent": latent,
                        "video": video,
                    },
                }
            except BaseException as error:
                artifact_box[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}:{error}",
                }
        dist.broadcast_object_list(
            artifact_box,
            src=topology.dp_arm * full81.SP_SIZE,
            group=topology.sp_group,
        )
        arm_envelope = artifact_box[0]
        if (
            not isinstance(arm_envelope, Mapping)
            or type(arm_envelope.get("ok")) is not bool
        ):
            raise FullExact81GPUError("arm leader artifact broadcast differs")
        decode_status = {
            "global_rank": topology.global_rank,
            "dp_arm": topology.dp_arm,
            "sp_rank": topology.sp_rank,
            "ok": arm_envelope["ok"],
            "error": arm_envelope.get("error"),
        }
        decode_statuses: list[Any] = [None] * full81.WORLD_SIZE
        dist.all_gather_object(
            decode_statuses, decode_status, group=topology.world_group
        )
        if any(
            not isinstance(row, Mapping)
            or row.get("global_rank") != rank
            or row.get("dp_arm") != rank // full81.SP_SIZE
            or row.get("sp_rank") != rank % full81.SP_SIZE
            or row.get("ok") is not True
            for rank, row in enumerate(decode_statuses)
        ):
            raise FullExact81GPUError("one or more exact81 arm decodes failed")
        artifacts = arm_envelope.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise FullExact81GPUError("successful arm artifacts differ")
        trainable_after_decode = active14_core._registry_digest(  # noqa: SLF001
            active14_core._trainable_registry(context.bindings)  # noqa: SLF001
        )
        base_after_decode = active14_core._registry_digest(  # noqa: SLF001
            active14_core._frozen_registry(context.bindings)  # noqa: SLF001
        )
        if (
            trainable_after_decode != trainable_before
            or base_after_decode != base_before
        ):
            raise FullExact81GPUError("exact81 decode changed parameter bytes")
        local_result = full81.build_local_result(
            global_rank=topology.global_rank,
            dp_arm=topology.dp_arm,
            family=full81.FAMILY_ORDER[topology.dp_arm],
            confirmation_iid=context.confirmation.row.iid,
            source_sha256=context.confirmation.row.source_sha256,
            action_prompt_sha256=hashlib.sha256(
                short_runner.ACTION_INSTRUCTION_BY_DP_ARM[topology.dp_arm].encode(
                    "utf-8"
                )
            ).hexdigest(),
            seed=seed,
            short_receipt_digest=context.short_receipt["digest"],
            field14_receipt_digest=context.field14_receipt["digest"],
            active14_precommit_digest=precommit["digest"],
            initial_gaussian_identity=noise_identity,
            endpoint_identity=endpoint_identity,
            exact40_trace=trace,
            media=artifacts["media"],
            latent_artifact=artifacts["latent"],
            video_artifact=artifacts["video"],
            output_root=str(final),
            expected_height=expected_height,
            expected_width=expected_width,
            trainable_sha256_before_decode=trainable_before,
            trainable_sha256_after_decode=trainable_after_decode,
            base_sha256_before_decode=base_before,
            base_sha256_after_decode=base_after_decode,
        )
        packet = json.loads(full81.canonical_json_bytes(local_result))
        if (
            len(full81.canonical_json_bytes(packet)) >= MAX_PACKET_BYTES
            or pickle.loads(pickle.dumps(packet, protocol=5)) != packet
        ):
            raise FullExact81GPUError("local result is not bounded pickle-safe")
        packets: list[Any] = [None] * full81.WORLD_SIZE
        dist.all_gather_object(packets, packet, group=topology.world_group)
        world8 = full81.assemble_world8_result(packets)
        preparation = full81.seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-full-exact81-preparation-v1",
                "status": "decoded_artifacts_staged_active14_commit_pending",
                "preparation_completed": True,
                "plan_sha256": args.expected_full_exact81_plan_sha256,
                "active14_precommit_digest": precommit["digest"],
                "world8": dict(world8),
                "staging_root": str(stage),
                "final_output_root": str(final),
                "published": False,
                "checkpoint_written": False,
                "publication_performed": False,
                **full81.false_authority(),
            }
        )
        full81.assert_no_elevated_authority(preparation)
        prepared.update(
            {
                "context_id": id(context),
                "global_rank": topology.global_rank,
                "stage": stage,
                "final": final,
                "world8": world8,
                "preparation": preparation,
                "precommit": precommit,
            }
        )
        return preparation

    def finalize(
        context: Any,
        active14_commit_receipt: Mapping[str, Any],
        preparation_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        import torch.distributed as dist

        topology = _validate_continuation_context(context)
        commit = active14_core.validate_sealed_mapping(
            active14_commit_receipt, label="active14 commit receipt"
        )
        active14_core.assert_no_authority(commit)
        preparation = full81.validate_sealed_mapping(
            preparation_receipt, label="full exact81 preparation receipt"
        )
        if (
            prepared.get("context_id") != id(context)
            or prepared.get("preparation", {}).get("digest")
            != preparation.get("digest")
            or preparation.get("published") is not False
            or preparation.get("active14_precommit_digest")
            != prepared.get("precommit", {}).get("digest")
            or commit.get("checkpoint_written") is not False
            or commit.get("publication_performed") is not False
            or commit.get("transaction_committed_in_memory") is not True
        ):
            raise FullExact81GPUError("active14 commit/preparation binding differs")
        stage = prepared["stage"]
        final = prepared["final"]
        parent = full81.seal_mapping(
            {
                "schema_version": PARENT_SCHEMA_VERSION,
                "status": "same_process_phase_a_replay_then_exact40_exact81_decoded",
                "plan": dict(plan),
                "plan_sha256": args.expected_full_exact81_plan_sha256,
                "world8": dict(prepared["world8"]),
                "active14_precommit_digest": preparation[
                    "active14_precommit_digest"
                ],
                "active14_commit_receipt": commit,
                "active14_commit_receipt_digest": commit["digest"],
                "preparation_receipt_digest": preparation["digest"],
                "source_binding": {
                    "full_exact81_core_sha256": args.expected_full_exact81_core_sha256,
                    "full_exact81_runner_sha256": args.expected_full_exact81_runner_sha256,
                    "active14_core_sha256": args.full_expected_active14_core_sha256,
                    "active14_runner_sha256": args.full_expected_active14_runner_sha256,
                    "active14_dependency_job_id": args.expected_active14_job_id,
                    "upstream_active14_parent_receipt_path": upstream_active14_parent[
                        "path"
                    ],
                    "upstream_active14_parent_receipt_sha256": upstream_active14_parent[
                        "file_sha256"
                    ],
                    "upstream_active14_parent_receipt_digest": upstream_active14_parent[
                        "receipt_digest"
                    ],
                    "dependency_afterok_is_queue_gate_only": True,
                    "weights_inherited_from_dependency": False,
                    "checkpoint_loaded_from_dependency": False,
                },
                "checkpoint_content_identity": dict(
                    context.checkpoint_content_identity
                ),
                "checkpoint_manifest_sha256": context.checkpoint_manifest_sha256,
                "bernini_revision": context.bernini_revision,
                "output_root": str(final),
                "atomic_publish_protocol": (
                    "sealed-stage-single-rename-after-active14-outer-close"
                ),
                "publish_deferred_until_active14_outer_close": True,
                "checkpoint_written": False,
                "publication_performed": False,
                "visual_semantics_evaluated": False,
                **full81.false_authority(),
            }
        )
        full81.assert_no_elevated_authority(parent)
        seal_box: list[Any] = [None]
        if topology.global_rank == 0:
            try:
                _write_create_only(
                    stage / "receipt.json",
                    full81.canonical_json_bytes(parent) + b"\n",
                )
                os.chmod(stage, 0o555)
                _fsync_directory(stage)
                seal_box[0] = {"ok": True, "parent_digest": parent["digest"]}
            except BaseException as error:
                seal_box[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}:{error}",
                }
        dist.broadcast_object_list(
            seal_box, src=0, group=topology.world_group
        )
        if (
            not isinstance(seal_box[0], Mapping)
            or seal_box[0].get("ok") is not True
        ):
            raise FullExact81GPUError("full exact81 staging seal failed")
        result = full81.seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-full-exact81-finalize-v1",
                "finalize_completed": True,
                "active14_commit_receipt_digest": commit["digest"],
                "preparation_receipt_digest": preparation["digest"],
                "parent_receipt_digest": parent["digest"],
                "final_output_root": str(final),
                "publication_ready": True,
                "published": False,
                "checkpoint_written": False,
                "publication_performed": False,
                **full81.false_authority(),
            }
        )
        prepared["parent"] = parent
        prepared["finalize"] = result
        return result

    def publish_after_outer_close() -> None:
        if prepared.get("global_rank") != 0:
            return
        stage = prepared.get("stage")
        final = prepared.get("final")
        if (
            not isinstance(stage, Path)
            or not isinstance(final, Path)
            or final.exists()
            or final.is_symlink()
            or not stage.is_dir()
            or stage.is_symlink()
            or stat.S_IMODE(stage.lstat().st_mode) != 0o555
            or prepared.get("finalize", {}).get("publication_ready") is not True
            or prepared.get("finalize", {}).get("published") is not False
        ):
            raise FullExact81GPUError("deferred full exact81 publish admission differs")
        _rename_directory_noreplace(stage, final)
        _fsync_directory(final.parent)
        prepared["published_after_active14_outer_close"] = True
        print(
            full81.canonical_json_bytes(prepared["parent"]).decode("ascii"),
            flush=True,
        )

    return prepare, finalize, publish_after_outer_close


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    plan = _load_plan(args)
    upstream_active14_parent = _load_upstream_active14_parent(args, plan)
    routing = short_runner.consume_authenticated_source_routing(args)
    prepare, finalize, publish_after_outer_close = _callbacks_factory(
        args, plan, upstream_active14_parent
    )
    try:
        result = active14_runner.replay_active14_for_downstream(
            args, routing, prepare=prepare, finalize=finalize
        )
        if not isinstance(result, Mapping):
            raise FullExact81GPUError(
                "active14 downstream finalization result differs"
            )
        full81.assert_no_elevated_authority(result)
        publish_after_outer_close()
    except BaseException:
        output = Path(args.full_exact81_output_dir)
        if os.environ.get("RANK", "") in ("", "0"):
            job_id = os.environ.get("SLURM_JOB_ID", "")
            if not job_id.isdigit():
                raise FullExact81GPUError(
                    "failed full exact81 cleanup lacks numeric job id"
                )
            stage = _stage_path_for_output(output, job_id=job_id)
            candidates = (
                (output, f".{output.name}.failed-postcommit-job{job_id}"),
                (stage, f".{output.name}.failed-staging-job{job_id}"),
            )
            for candidate, quarantine_name in candidates:
                try:
                    linked = candidate.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISDIR(linked.st_mode):
                    raise FullExact81GPUError(
                        "failed full exact81 path is not a real directory"
                    )
                quarantine = output.with_name(quarantine_name)
                if quarantine.exists():
                    raise FullExact81GPUError(
                        "failed full exact81 quarantine path exists"
                    )
                os.rename(candidate, quarantine)
                _fsync_directory(output.parent)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FullExact81GPUError",
    "PARENT_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "_load_plan",
    "_rename_directory_noreplace",
    "_safe_stage_paths",
    "_seal_artifact",
    "_stage_path_for_output",
    "_step_state",
    "build_parser",
    "file_sha256",
    "main",
    "validate_cli",
]
