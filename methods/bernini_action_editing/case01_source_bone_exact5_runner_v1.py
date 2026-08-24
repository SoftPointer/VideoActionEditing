#!/usr/bin/env python3
"""Exact-five case01 R64 runner derived around the frozen successful r5 stack.

The frozen runner remains byte-for-byte unchanged.  This wrapper pins and
source-loads it, reuses its retained model/adapter authority, publication
handoff, four-rank Torchrun, and post-use replay implementation, and replaces
only the historical exact16/canary2 plan and final-report closures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Mapping, Sequence


FROZEN_RUNNER_SHA256 = (
    "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223"
)
EXACT5_EVAL_SHA256 = (
    "891551d46b0ca11362fe8d446f202036b9957baa943de0aec6da1f3ad45d7d58"
)
RUNNER_SCHEMA = "case01-source-bone-exact5-runner-attestation-v1"
FAILURE_SCHEMA = "case01-source-bone-exact5-runner-failure-v1"
PHYSICAL_BINDINGS_SCHEMA = "case01-source-bone-exact5-physical-bindings-v1"
_FROZEN_RUNNER_BASENAME = "full644_exploratory_matched_runner_auh_r5.py"
_EXACT5_EVAL_BASENAME = "case01_source_bone_exact5_eval_v1.py"
_FROZEN_MODULE_NAME = "_case01_exact5_frozen_r5_runner"
_EXACT5_EVAL_MODULE_NAME = "_case01_source_bone_exact5_eval_v1"


class Exact5RunnerBootstrapError(RuntimeError):
    """The frozen runner or exact-five wrapper bootstrap differs."""


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_pinned_source(path: Path, expected_sha256: str, *, label: str) -> str:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise Exact5RunnerBootstrapError(f"{label} path differs")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named)
        or digest.hexdigest() != expected_sha256
    ):
        raise Exact5RunnerBootstrapError(f"{label} source identity differs")
    try:
        return b"".join(chunks).decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise Exact5RunnerBootstrapError(f"{label} source is not UTF-8") from error


def _early_main_entry_gate() -> None:
    """Reject a direct/non-isolated wrapper before loading any frozen code."""

    if __name__ != "__main__":
        return
    raw = os.environ.get("FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY")
    if (
        sys.platform != "linux"
        or not Path("/proc/self/fd").is_dir()
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or not sys.dont_write_bytecode
        or raw is None
    ):
        raise Exact5RunnerBootstrapError(
            "exact5 runner requires captured-source -I -S -B entry"
        )
    try:
        entry = json.loads(raw)
        runner_path = Path(entry["runner_path"])
        runner_fd = entry["runner_fd"]
        before = os.fstat(runner_fd)
        named = runner_path.lstat()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise Exact5RunnerBootstrapError("early captured runner entry differs") from error
    if (
        entry.get("schema_version")
        != "full644-exploratory-matched-captured-runner-entry-authority-v1"
        or runner_path.resolve(strict=True) != Path(__file__).resolve(strict=True)
        or type(runner_fd) is not int
        or runner_fd < 3
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or _identity(before) != _identity(named)
    ):
        raise Exact5RunnerBootstrapError("early captured runner binding differs")


_early_main_entry_gate()
_METHOD_ROOT = Path(__file__).resolve(strict=True).parent
_FROZEN_RUNNER_PATH = _METHOD_ROOT / _FROZEN_RUNNER_BASENAME
_EXACT5_EVAL_PATH = _METHOD_ROOT / _EXACT5_EVAL_BASENAME


def _load_source_module(
    name: str,
    path: Path,
    expected_sha256: str,
    *,
    file_override: Path | None = None,
) -> types.ModuleType:
    if name in sys.modules:
        raise Exact5RunnerBootstrapError(f"{name} was imported before bootstrap")
    source = _read_pinned_source(path, expected_sha256, label=name)
    module = types.ModuleType(name)
    module.__file__ = str(path if file_override is None else file_override)
    module.__package__ = None
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, origin=str(path)
    )
    module.__builtins__ = __builtins__
    sys.modules[name] = module
    try:
        exec(
            compile(source, str(path), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


# ``file_override`` is intentional: the frozen implementation's self-identity
# checks must bind the captured wrapper inode, while its compiled-code origin
# remains visibly the frozen source path for audit.
frozen = _load_source_module(
    _FROZEN_MODULE_NAME,
    _FROZEN_RUNNER_PATH,
    FROZEN_RUNNER_SHA256,
    file_override=Path(__file__).resolve(strict=True),
)
exact5 = _load_source_module(
    _EXACT5_EVAL_MODULE_NAME,
    _EXACT5_EVAL_PATH,
    EXACT5_EVAL_SHA256,
)


def _bind_exact5_globals() -> None:
    expected_old = tuple(
        f"shared8-{index:02d}-{arm}"
        for index in range(8)
        for arm in ("base", "full644")
    )
    if (
        frozen.TASK_IDS != expected_old
        or frozen.CANARY_TASK_IDS != expected_old[:2]
        or frozen.FULL16_CAMPAIGN != "full16-production"
        or frozen.CASE00_CANARY_CAMPAIGN != "case00-pair-canary"
    ):
        raise Exact5RunnerBootstrapError("frozen exact16/canary2 globals differ")
    # These are the complete globals read by the reused task path.  The full16
    # mode is made unreachable; exact5 occupies the canary branch so the
    # frozen RunnerExecution correctly reports a non-formal campaign.
    frozen.TASK_IDS = exact5.TASK_IDS
    frozen.CANARY_TASK_IDS = exact5.TASK_IDS
    frozen.FULL16_CAMPAIGN = "disabled-frozen-full16"
    frozen.CASE00_CANARY_CAMPAIGN = exact5.CAMPAIGN
    frozen.SCHEMA = RUNNER_SCHEMA
    frozen.FAILURE_SCHEMA = FAILURE_SCHEMA


_bind_exact5_globals()


def validate_task_order(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        value = exact5.validate_plan(plan, reopen_sources=True)
    except exact5.Exact5EvalError as error:
        raise frozen.MatchedRunnerV2Error(str(error)) from error
    tasks = value["tasks"]
    root = Path(tasks[0]["output"]["video_path"]).parent
    if (
        not root.is_absolute()
        or os.path.normpath(str(root)) != str(root)
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
    ):
        raise frozen.MatchedRunnerV2Error("exact5 publication root differs")
    publication_paths = {
        Path(task["output"][leaf])
        for task in tasks
        for leaf in ("video_path", "receipt_path")
    }
    if len(publication_paths) != 10:
        raise frozen.MatchedRunnerV2Error(
            "exact5 publication paths are not distinct"
        )
    internal_paths: set[Path] = set()
    for index, task in enumerate(tasks):
        prefix = f".matched-v2-{index:02d}-{task['task_id']}"
        suffixes = [
            *frozen._BASE_ARTIFACT_SUFFIXES,
            *frozen._ADAPTER_ARTIFACT_SUFFIXES,
            ".log",
            "-runner-task.json",
        ]
        for suffix in suffixes:
            path = root / (prefix + suffix)
            if path.exists() or path.is_symlink() or path in internal_paths:
                raise frozen.MatchedRunnerV2Error(
                    "exact5 internal artifact is not fresh"
                )
            internal_paths.add(path)
    if publication_paths & internal_paths:
        raise frozen.MatchedRunnerV2Error(
            "exact5 publication paths overlap internal artifacts"
        )
    return [dict(task) for task in tasks]


def select_campaign_tasks(
    plan: Mapping[str, Any], campaign_mode: str
) -> tuple[dict[str, Any], ...]:
    tasks = tuple(validate_task_order(plan))
    if campaign_mode != exact5.CAMPAIGN:
        raise frozen.MatchedRunnerV2Error("only the exact5 campaign is enabled")
    if tuple(task["task_id"] for task in tasks) != exact5.TASK_IDS:
        raise frozen.MatchedRunnerV2Error("exact5 task selection differs")
    return tasks


def replay_task_authority_artifacts_exact5(
    output_root: Path,
    output_root_fd: int,
    task_result: Mapping[str, Any],
    verified_result: Mapping[str, Any],
    publication_authority: Mapping[str, Any],
    publication_handoff: Mapping[str, Any],
) -> dict[str, Any]:
    index = task_result.get("task_index")
    if (
        type(index) is not int
        or index not in range(5)
        or task_result.get("task_id") != exact5.TASK_IDS[index]
    ):
        raise frozen.MatchedRunnerV2Error("exact5 task result index differs")
    return frozen.replay_task_authority_artifacts(
        output_root,
        output_root_fd,
        task_result,
        verified_result,
        publication_authority,
        publication_handoff,
    )


def _complete_execution(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    execution: Any,
    final_parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    entry_authority = frozen.validate_captured_runner_entry(
        args.entry_authority, args=args
    )
    if (
        args.campaign_mode != exact5.CAMPAIGN
        or tuple(task.get("task_id") for task in tasks) != exact5.TASK_IDS
    ):
        raise frozen.MatchedRunnerV2Error("exact5 completion task closure differs")
    task_results, model_final = execution.run()
    if (
        len(task_results) != 5
        or tuple(row.get("task_id") for row in task_results) != exact5.TASK_IDS
        or tuple(row.get("task_index") for row in task_results) != tuple(range(5))
        or set(execution.publication_authorities) != set(exact5.TASK_IDS)
        or set(execution.publication_handoffs) != set(exact5.TASK_IDS)
    ):
        raise frozen.MatchedRunnerV2Error(
            "exact5 task/publication authority closure differs"
        )
    if execution.output_root_fd is None or execution.output_root_identity is None:
        raise frozen.MatchedRunnerV2Error("retained output root disappeared")
    frozen._validate_held_directory(
        execution.output_root_fd,
        execution.output_root,
        execution.output_root_identity,
    )
    frozen._validate_embedded_digest(
        model_final, "model_final_digest", label="model final"
    )
    if (
        model_final.get("task_count") != 5
        or model_final.get("model_capture_digest")
        != task_results[0]["model_capture_digest"]
        or model_final.get("task_consumption_digests")
        != [row["consumption_digest"] for row in task_results]
        or any(
            row.get("ffmpeg_exec_authority_digest")
            != execution.ffmpeg_exec_authority_digest
            for row in task_results
        )
    ):
        raise frozen.MatchedRunnerV2Error("exact5 model/task final closure differs")
    try:
        report = exact5.verify_results(
            plan,
            frozen_v2=frozen.v2,
            publication_root_fd=execution.output_root_fd,
            ffprobe_authority=execution.ffprobe_authority,
            publication_authorities=execution.publication_authorities,
        )
    except exact5.Exact5EvalError as error:
        raise frozen.MatchedRunnerV2Error(str(error)) from error
    verified_rows = report.get("results")
    if (
        not isinstance(verified_rows, list)
        or len(verified_rows) != 5
        or tuple(row.get("task_id") for row in verified_rows) != exact5.TASK_IDS
        or report.get("formal_full16_report") is not False
        or report.get("manual_blind_review_required") is not True
        or report.get("retained_publication_root_fd_replayed") is not True
        or report.get("retained_ffprobe_executable_fd_replayed") is not True
        or report.get("retained_publication_leaf_fds_replayed") is not True
    ):
        raise frozen.MatchedRunnerV2Error("exact5 verified report closure differs")
    verified_by_task = {row["task_id"]: row for row in verified_rows}
    artifact_replays = [
        replay_task_authority_artifacts_exact5(
            execution.output_root,
            execution.output_root_fd,
            row,
            verified_by_task[row["task_id"]],
            execution.publication_authorities[row["task_id"]],
            execution.publication_handoffs[row["task_id"]],
        )
        for row in task_results
    ]
    report_parent = final_parents["output_report"]
    attestation_parent = final_parents["runner_attestation"]
    frozen._validate_final_parent(report_parent)
    frozen._validate_final_parent(attestation_parent)
    report_path = report_parent["path"]
    _, report_sha = frozen._write_json_at(
        report_parent["parent_fd"], report_path.name, report, mode=0o444
    )
    attestation: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA,
        "status": "EXACT5_COMPLETE_AWAITING_BLIND_REVIEW",
        "campaign_mode": exact5.CAMPAIGN,
        "formal_full16_report": False,
        "manual_blind_review_required": True,
        "plan": {
            "path": str(Path(args.plan).resolve(strict=True)),
            "sha256": args.plan_sha256,
            "plan_digest": plan["plan_digest"],
        },
        "physical_bindings": dict(bindings),
        "captured_runner_entry": {
            "authority_digest": entry_authority["authority_digest"],
            "release_digest": entry_authority["release_digest"],
            "bootstrap_sha256": entry_authority["bootstrap_sha256"],
            "captured_source_entry": True,
            "held_through_attestation_publication": True,
        },
        "retained_publication_root": {
            "path": str(execution.output_root),
            "fd": execution.output_root_fd,
            "immutable_identity": execution.output_root_identity,
            "held_through_attestation_publication": True,
        },
        "retained_ffprobe_executable": {
            "authority_digest": execution.ffprobe_authority["authority_digest"],
            "fd": execution.ffprobe_authority["fd"],
            "source_path": execution.ffprobe_authority["source_path"],
            "sha256": execution.ffprobe_authority["sha256"],
            "held_through_result_verification": True,
        },
        "retained_task_publications": {
            task_id: {
                "authority_digest": row["authority_digest"],
                "receipt_fd": row["receipt_fd"],
                "output_fd": row["output_fd"],
                "held_through_result_verification": True,
            }
            for task_id, row in sorted(execution.publication_authorities.items())
        },
        "retained_child_publication_handoffs": {
            task_id: {
                "authority_digest": row["authority_digest"],
                "fd": row["fd"],
                "payload_digest": frozen.read_sealed_publication_handoff(
                    row,
                    next(task for task in tasks if task["task_id"] == task_id),
                )["payload_digest"],
                "held_sealed_through_attestation": True,
            }
            for task_id, row in sorted(execution.publication_handoffs.items())
        },
        "retained_final_parents": {
            label: {
                "path": str(row["path"].parent),
                "fd": row["parent_fd"],
                "immutable_identity": row["parent_identity"],
            }
            for label, row in final_parents.items()
        },
        "task_count": 5,
        "task_ids": list(exact5.TASK_IDS),
        "unselected_task_ids": [],
        "unselected_task_count": 0,
        "all_exact5_tasks_attempted_exactly_once": True,
        "all_exact5_tasks_succeeded": True,
        "retry_count": 0,
        "task_result_digests": [row["task_result_digest"] for row in task_results],
        "task_environment_digests": [
            row["environment_digest"] for row in task_results
        ],
        "ffmpeg_exec_authority_digest": execution.ffmpeg_exec_authority_digest,
        "all_rank0_encoders_used_retained_ffmpeg_executable": True,
        "task_results": task_results,
        "task_artifact_replays": artifact_replays,
        "runner_task_json_replayed_for_all_tasks": True,
        "native_publication_before_parent_post_use_replay": True,
        "all_model_adapter_post_use_replays_complete": True,
        "native_receipts_replayed_0400_single_link": all(
            row["native_receipt_mode"] == 0o400
            and row["native_receipt_nlink"] == 1
            for row in artifact_replays
        ),
        "model_capture_digest": task_results[0]["model_capture_digest"],
        "same_model_capture_all_exact5_tasks": len(
            {row["model_capture_digest"] for row in task_results}
        )
        == 1,
        "model_final": model_final,
        "verified_report": {
            "path": str(report_path),
            "sha256": report_sha,
            "report_digest": report["report_digest"],
            "verified_task_count": report["task_count"],
        },
        "reused_frozen_execution_contract": {
            "frozen_runner_sha256": FROZEN_RUNNER_SHA256,
            "retained_model_adapter_fd_closure": True,
            "sealed_publication_handoff": True,
            "four_rank_torchrun": True,
            "post_use_replay": True,
        },
        "exploratory_only": True,
        "scientific_claim_authorized": False,
        "formal_claim_authorized": False,
    }
    if (
        attestation["same_model_capture_all_exact5_tasks"] is not True
        or attestation["native_receipts_replayed_0400_single_link"] is not True
        or set(execution.publication_authorities) != set(exact5.TASK_IDS)
        or set(execution.publication_handoffs) != set(exact5.TASK_IDS)
    ):
        raise frozen.MatchedRunnerV2Error("exact5 final attestation closure differs")
    attestation["attestation_digest"] = exact5.object_sha256(attestation)
    frozen._validate_held_directory(
        execution.output_root_fd,
        execution.output_root,
        execution.output_root_identity,
    )
    frozen._validate_final_parent(report_parent)
    frozen._validate_final_parent(attestation_parent)
    frozen.validate_captured_runner_entry(entry_authority, args=args)
    attestation_path = attestation_parent["path"]
    frozen._write_json_at(
        attestation_parent["parent_fd"],
        attestation_path.name,
        attestation,
        mode=0o444,
    )
    return attestation


def execute(args: argparse.Namespace) -> dict[str, Any]:
    entry_authority = frozen.validate_captured_runner_entry(
        args.entry_authority, args=args
    )
    if args.campaign_mode != exact5.CAMPAIGN:
        raise frozen.MatchedRunnerV2Error("non-exact5 campaign is disabled")
    allocation = frozen._allocation_authority(
        args.holder_job_id,
        args.expected_node,
        args.expected_allocation_gpu_count,
    )
    try:
        plan = exact5.load_plan(args.plan, args.plan_sha256)
    except exact5.Exact5EvalError as error:
        raise frozen.MatchedRunnerV2Error(str(error)) from error
    if frozen.v2.validate_terminal_checkpoint_manifest(
        plan["checkpoint_manifest"]["path"],
        plan["checkpoint_manifest"]["sha256"],
    ) != plan["checkpoint_manifest"]:
        raise frozen.MatchedRunnerV2Error(
            "terminal checkpoint changed before exact5 execution"
        )
    tasks = select_campaign_tasks(plan, args.campaign_mode)
    final_artifacts = frozen._preflight_final_artifacts(
        args, validate_task_order(plan)
    )
    identities = {
        "runner": frozen._identity(__file__, args.runner_sha256),
        "frozen_runner": frozen._identity(
            _FROZEN_RUNNER_PATH, FROZEN_RUNNER_SHA256
        ),
        "exact5_eval": frozen._identity(_EXACT5_EVAL_PATH, EXACT5_EVAL_SHA256),
        "bridge": frozen._identity(
            args.bridge_script, args.bridge_script_sha256
        ),
        "adapter": frozen._identity(
            args.adapter_script, args.adapter_script_sha256
        ),
        "eval_v1": frozen._identity(
            args.eval_v1_source, args.eval_v1_source_sha256
        ),
        "eval_v2": frozen._identity(
            args.eval_v2_source, args.eval_v2_source_sha256
        ),
        "model_authority": frozen._identity(
            args.model_authority_source,
            args.model_authority_source_sha256,
        ),
        "python": frozen._identity(args.python, args.python_sha256),
        "torchrun_source": frozen._identity(
            args.torchrun_source, args.torchrun_source_sha256
        ),
        "torchrun_handler_source": frozen._identity(
            args.torchrun_handler_source,
            args.torchrun_handler_source_sha256,
        ),
        "torch_local_agent_source": frozen._identity(
            args.torch_local_agent_source,
            args.torch_local_agent_source_sha256,
        ),
        "torch_dynamic_rendezvous_source": frozen._identity(
            args.torch_dynamic_rendezvous_source,
            args.torch_dynamic_rendezvous_source_sha256,
        ),
        "torch_multiprocessing_api_source": frozen._identity(
            args.torch_multiprocessing_api_source,
            args.torch_multiprocessing_api_source_sha256,
        ),
        "model_manifest": frozen._identity(
            args.model_manifest, args.model_manifest_sha256
        ),
        "ffmpeg": frozen._identity(
            args.ffmpeg_executable, args.ffmpeg_executable_sha256
        ),
        "ffprobe": frozen._identity(
            plan["producer"]["ffprobe_path"],
            plan["producer"]["ffprobe_sha256"],
        ),
    }
    if (
        args.model_manifest_sha256 != frozen.EXPECTED_MODEL_MANIFEST_SHA256
        or args.eval_v1_source_sha256 != frozen.EXPECTED_EVAL_V1_SHA256
        or args.eval_v2_source_sha256 != frozen.EXPECTED_EVAL_V2_SHA256
        or args.model_authority_source_sha256
        != frozen.EXPECTED_MODEL_AUTHORITY_SHA256
        or args.torchrun_source_sha256 != frozen.TORCHRUN_SOURCE_SHA256
        or args.torchrun_handler_source_sha256
        != frozen.TORCHRUN_HANDLER_SHA256
        or args.torch_local_agent_source_sha256
        != frozen.TORCH_LOCAL_ELASTIC_AGENT_SHA256
        or args.torch_dynamic_rendezvous_source_sha256
        != frozen.TORCH_DYNAMIC_RENDEZVOUS_SHA256
        or args.torch_multiprocessing_api_source_sha256
        != frozen.TORCH_MULTIPROCESSING_API_SHA256
    ):
        raise frozen.MatchedRunnerV2Error("model/Torch exact source pin differs")
    infer_identity = frozen._identity(
        plan["producer"]["infer_lora_path"],
        plan["producer"]["infer_lora_sha256"],
    )
    if (
        Path(args.adapter_script).resolve(strict=True).parent
        != Path(plan["producer"]["infer_lora_path"]).resolve(strict=True).parent
        or Path(args.bridge_script).resolve(strict=True).parent
        != Path(args.adapter_script).resolve(strict=True).parent
        or Path(__file__).resolve(strict=True).parent
        != Path(args.adapter_script).resolve(strict=True).parent
        or Path(frozen.v1.__file__).resolve(strict=True)
        != Path(args.eval_v1_source).resolve(strict=True)
        or Path(frozen.v2.__file__).resolve(strict=True)
        != Path(args.eval_v2_source).resolve(strict=True)
        or Path(frozen.model_authority.__file__).resolve(strict=True)
        != Path(args.model_authority_source).resolve(strict=True)
        or any(
            getattr(module, "__cached__", None) is not None
            for module in (frozen.v1, frozen.v2, frozen.model_authority, exact5)
        )
    ):
        raise frozen.MatchedRunnerV2Error("exact5 frozen release differs")
    ffprobe_authority = frozen.capture_ffprobe_authority(
        identities["ffprobe"], plan["producer"]
    )
    try:
        exec_authority = frozen.capture_exec_authority(identities)
    except BaseException:
        frozen.close_ffprobe_authority(ffprobe_authority)
        raise
    try:
        args.exec_authority = exec_authority
        args.ffprobe_authority = ffprobe_authority
        bindings: dict[str, Any] = {
            "schema_version": PHYSICAL_BINDINGS_SCHEMA,
            "plan_path": str(Path(args.plan).resolve(strict=True)),
            "plan_sha256": args.plan_sha256,
            "plan_digest": plan["plan_digest"],
            "asset_authority_digest": plan["asset_authority"][
                "authority_digest"
            ],
            "allocation": allocation,
            "identities": {**identities, "infer_lora": infer_identity},
            "captured_runner_entry": entry_authority,
            "captured_runner_entry_required": True,
            "exec_authority": exec_authority,
            "exec_authority_retained_source_and_python_fds": True,
            "ffprobe_authority": ffprobe_authority,
            "ffprobe_retained_executable_fd": True,
            "isolated_child_interpreters": "-I -S -B",
            "child_environment_exact_allowlist": True,
            "model_root": str(Path(args.model_root).resolve(strict=True)),
            "bernini_root": str(Path(args.bernini_root).resolve(strict=True)),
            "veomni_root": str(Path(args.veomni_root).resolve(strict=True)),
            "campaign_mode": exact5.CAMPAIGN,
            "formal_full16_report": False,
            "task_count": 5,
            "task_ids": list(exact5.TASK_IDS),
            "retry_allowed": False,
            "final_artifacts": final_artifacts,
        }
        bindings["physical_bindings_digest"] = exact5.object_sha256(bindings)
        args.physical_bindings_digest = bindings["physical_bindings_digest"]
    except BaseException:
        frozen.close_exec_authority(exec_authority)
        frozen.close_ffprobe_authority(ffprobe_authority)
        raise
    final_parents: dict[str, dict[str, Any]] = {}
    execution: Any | None = None
    try:
        final_parents = frozen._hold_final_artifact_parents(final_artifacts)
        execution = frozen.RunnerExecution(args, plan, tasks)
        return _complete_execution(
            args, plan, tasks, bindings, execution, final_parents
        )
    finally:
        if execution is not None:
            execution.close_descriptors()
        else:
            frozen.close_exec_authority(exec_authority)
            frozen.close_ffprobe_authority(ffprobe_authority)
        frozen._close_final_parents(final_parents)


def build_parser() -> argparse.ArgumentParser:
    parser = frozen.build_parser()
    parser.description = __doc__
    campaign_actions = [
        action for action in parser._actions if action.dest == "campaign_mode"
    ]
    if len(campaign_actions) != 1:
        raise Exact5RunnerBootstrapError("frozen campaign parser action differs")
    campaign_actions[0].choices = (exact5.CAMPAIGN,)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # These are deliberately called here, not delegated to a frozen submodule
    # main, so the captured wrapper remains the validated entry authority.
    frozen._require_isolated_runner_startup()
    entry_authority = frozen.validate_captured_runner_entry()
    args: argparse.Namespace | None = None
    try:
        try:
            args = build_parser().parse_args(argv)
            if args.campaign_mode != exact5.CAMPAIGN:
                raise frozen.MatchedRunnerV2Error(
                    "only the exact5 campaign is enabled"
                )
            args.entry_authority = frozen.validate_captured_runner_entry(
                entry_authority, args=args
            )
            result = execute(args)
        except Exception as error:
            if args is None:
                raise
            failure = {
                "schema_version": FAILURE_SCHEMA,
                "status": "FAILED_NO_RETRY",
                "error_type": type(error).__name__,
                "error": str(error),
                "plan_path": str(Path(args.plan)),
                "plan_sha256": args.plan_sha256,
                "runner_path": str(Path(__file__).resolve(strict=True)),
                "retry_allowed": False,
                "partial_outputs_are_not_results": True,
                "scientific_claim_authorized": False,
            }
            failure["failure_digest"] = exact5.object_sha256(failure)
            try:
                frozen.v1.write_create_only(args.runner_attestation, failure)
            except Exception:
                pass
            raise
        try:
            print(exact5.canonical_json_bytes(result).decode("utf-8"))
        except (BrokenPipeError, OSError):
            pass
        return 0
    finally:
        frozen.close_captured_runner_entry(entry_authority)


if __name__ == "__main__":
    raise SystemExit(main())
