#!/usr/bin/env python3
"""Fail-closed controller for the two independent WORLD4 action-editing runs.

The controller never owns either retained allocation.  It starts numbered
``srun`` children through the holder launcher and, on failure, signals only the
local wrapper process groups that it created.  There is deliberately no Slurm
cancel/release/requeue operation in this module.

The formal main run is staged: ``stage-r64`` is only a retention checkpoint and
is never accepted as an action result.  ``resume-po40`` must bind that exact
checkpoint and its receipt.  The action-only control always starts from the
same frozen base and shared P/O initialization, never from the R checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, NoReturn, Optional, Sequence


sys.dont_write_bytecode = True

PAIR_PLAN_SCHEMA = "bernini-generic-source-anchored-action-pair-plan-v1"
PAIR_RECEIPT_SCHEMA = "bernini-generic-source-anchored-action-pair-receipt-v1"
TRAINING_RECEIPT_SCHEMA = (
    "bernini-generic-source-anchored-action-training-receipt-v1"
)
SOURCE_MANIFEST_SHA256 = (
    "128064fd335c4e48c567217c6e7bae43555a904875625c9d1e21178e6f7fcc3d"
)
GPU_MEMORY_LIMIT_GIB = 52.0
HOST_MEMORY_LIMIT_GIB = 60.0
WORLD_SIZE = 4
PARALLEL_TOPOLOGY = "world4-dp1-sp4"
LAUNCH_CONFIRMATION = "launch-approved-generic-pair-136309-136141"

COMMON_LAUNCHER = (
    "scripts/auh_train_generic_source_anchored_action_world4_holder_v1.sh"
)
TRAINER = "train_generic_source_anchored_action_v1.py"
CORE = "generic_source_anchored_action_v1.py"
MANIFEST_VALIDATOR = "tools/generic_action_manifest_v1.py"

ARM_BINDINGS: Mapping[str, Mapping[str, Any]] = {
    "joint_stage_r64": {
        "holder_job": 136309,
        "holder_node": "auh7-1b-gpu-280",
        "execution_profile": "stage-r64",
        "optimizer_steps": 64,
        "carrier_policy": "installed_trainable",
        "requires_action_manifests": False,
        "requires_resume": False,
        "complete_action_result": False,
    },
    "joint_resume_po40": {
        "holder_job": 136309,
        "holder_node": "auh7-1b-gpu-280",
        "execution_profile": "resume-po40",
        "optimizer_steps": 40,
        "carrier_policy": "resume_frozen_stage_r64",
        "requires_action_manifests": True,
        "requires_resume": True,
        "complete_action_result": True,
    },
    "action_only_no_carrier": {
        "holder_job": 136141,
        "holder_node": "auh7-1b-gpu-299",
        "execution_profile": "action-only40",
        "optimizer_steps": 40,
        "carrier_policy": "not_installed_or_exact_zero_frozen",
        "requires_action_manifests": True,
        "requires_resume": False,
        "complete_action_result": True,
    },
    "smoke_r": {
        "holder_job": 136309,
        "holder_node": "auh7-1b-gpu-280",
        "execution_profile": "smoke-r",
        "optimizer_steps": 1,
        "carrier_policy": "installed_trainable_disposable",
        "requires_action_manifests": False,
        "requires_resume": False,
        "complete_action_result": False,
    },
    "smoke_p": {
        "holder_job": 136309,
        "holder_node": "auh7-1b-gpu-280",
        "execution_profile": "smoke-p",
        "optimizer_steps": 1,
        "carrier_policy": "inactive_exact_zero_disposable",
        "requires_action_manifests": True,
        "requires_resume": False,
        "complete_action_result": False,
    },
    "smoke_o": {
        "holder_job": 136309,
        "holder_node": "auh7-1b-gpu-280",
        "execution_profile": "smoke-o",
        "optimizer_steps": 1,
        "carrier_policy": "inactive_exact_zero_disposable",
        "requires_action_manifests": True,
        "requires_resume": False,
        "complete_action_result": False,
    },
}


class GenericActionPairError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise GenericActionPairError(message)


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
        raise GenericActionPairError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        fail(f"input changed while hashing: {path}")
    return digest.hexdigest()


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        fail(f"{label} must be one lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not stat.S_ISREG(resolved.lstat().st_mode):
        fail(f"{label} must be one canonical plain file")
    return resolved


def _plain_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not stat.S_ISDIR(resolved.lstat().st_mode):
        fail(f"{label} must be one canonical directory")
    return resolved


def _fresh_path(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or requested.exists()
        or requested.is_symlink()
        or requested.parent.resolve(strict=True) != requested.parent
    ):
        fail(f"{label} must be one fresh canonical path")
    return requested


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GenericActionPairError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, Mapping):
        fail(f"{label} must be a JSON object")
    return value


def _verify_file(path: Path, expected: str, *, label: str) -> str:
    expected = _digest(expected, label=f"expected {label} SHA-256")
    observed = file_sha256(path)
    if observed != expected:
        fail(f"{label} SHA-256 differs")
    return observed


def _validate_port(value: int, *, label: str) -> int:
    if type(value) is not int or not 1024 <= value <= 65535:
        fail(f"{label} must be one unprivileged TCP port")
    return value


def _action_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    representation = _plain_file(
        args.representation_manifest, label="representation manifest"
    )
    pairs = _plain_file(args.source_pair_manifest, label="source-pair manifest")
    _verify_file(
        representation,
        args.expected_representation_manifest_sha256,
        label="representation manifest",
    )
    _verify_file(
        pairs,
        args.expected_source_pair_manifest_sha256,
        label="source-pair manifest",
    )
    return representation, pairs


def validate_inputs(
    args: argparse.Namespace, *, require_action_manifests: bool
) -> Mapping[str, Any]:
    """Close every source/data pin before any run directory or child exists."""

    method_root = _plain_directory(args.method_root, label="method root")
    python_bin = _plain_file(args.python_bin, label="Python executable")
    if not os.access(python_bin, os.X_OK):
        fail("Python executable is not executable")
    trainer = _plain_file(method_root / TRAINER, label="trainer source")
    core = _plain_file(method_root / CORE, label="core source")
    launcher = _plain_file(
        method_root / COMMON_LAUNCHER, label="common holder launcher"
    )
    trainer_sha = _verify_file(
        trainer, args.expected_trainer_sha256, label="trainer source"
    )
    core_sha = _verify_file(core, args.expected_core_sha256, label="core source")
    launcher_sha = _verify_file(
        launcher, args.expected_launcher_sha256, label="common holder launcher"
    )
    validator: Optional[Path] = None
    validator_sha: Optional[str] = None
    if require_action_manifests:
        validator = _plain_file(
            method_root / MANIFEST_VALIDATOR, label="action-manifest validator"
        )
        validator_sha = _verify_file(
            validator,
            args.expected_manifest_validator_sha256,
            label="action-manifest validator",
        )
    elif args.expected_manifest_validator_sha256 is not None:
        fail("R-only profile must not consume an action-manifest validator pin")

    source = _plain_file(args.source_manifest, label="source manifest")
    if args.expected_source_manifest_sha256 != SOURCE_MANIFEST_SHA256:
        fail("source manifest pin is not the preregistered 64/16/8 authority")
    _verify_file(source, SOURCE_MANIFEST_SHA256, label="source manifest")

    archive = _plain_file(args.method_archive, label="method archive")
    release_manifest = _plain_file(args.method_manifest, label="method manifest")
    archive_sha = _verify_file(
        archive, args.expected_method_archive_sha256, label="method archive"
    )
    release_manifest_sha = _verify_file(
        release_manifest,
        args.expected_method_manifest_sha256,
        label="method manifest",
    )

    representation: Optional[Path] = None
    pairs: Optional[Path] = None
    representation_sha: Optional[str] = None
    pairs_sha: Optional[str] = None
    if require_action_manifests:
        representation, pairs = _action_inputs(args)
        representation_sha = file_sha256(representation)
        pairs_sha = file_sha256(pairs)
        if validator is None:
            fail("action profile lacks its validator")
        command = [
            str(python_bin),
            "-B",
            str(validator),
            "validate",
            "--representation",
            str(representation),
            "--pairs",
            str(pairs),
        ]
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            fail(f"action-manifest validator rejected authority: {result.stdout[-2000:]}")

    return {
        "method_root": str(method_root),
        "python_bin": str(python_bin),
        "trainer_sha256": trainer_sha,
        "core_sha256": core_sha,
        "launcher_sha256": launcher_sha,
        "manifest_validator_sha256": validator_sha,
        "method_archive": str(archive),
        "method_archive_sha256": archive_sha,
        "method_manifest": str(release_manifest),
        "method_manifest_sha256": release_manifest_sha,
        "source_manifest": str(source),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "representation_manifest": None if representation is None else str(representation),
        "representation_manifest_sha256": representation_sha,
        "source_pair_manifest": None if pairs is None else str(pairs),
        "source_pair_manifest_sha256": pairs_sha,
    }


def build_arm_plan(
    args: argparse.Namespace,
    *,
    arm_id: str,
    run_root: Path,
    master_port: int,
) -> Mapping[str, Any]:
    if arm_id not in ARM_BINDINGS:
        fail(f"unregistered arm: {arm_id}")
    binding = dict(ARM_BINDINGS[arm_id])
    closure = validate_inputs(
        args, require_action_manifests=bool(binding["requires_action_manifests"])
    )
    resume: Optional[Mapping[str, Any]] = None
    if binding["requires_resume"]:
        checkpoint = _plain_file(args.resume_checkpoint, label="R64 resume checkpoint")
        receipt = _plain_file(args.resume_receipt, label="R64 resume receipt")
        checkpoint_sha = _verify_file(
            checkpoint,
            args.expected_resume_checkpoint_sha256,
            label="R64 resume checkpoint",
        )
        receipt_sha = _verify_file(
            receipt,
            args.expected_resume_receipt_sha256,
            label="R64 resume receipt",
        )
        retained = validate_training_receipt(
            receipt,
            expected_profile="stage-r64",
            expected_steps=64,
            expected_complete_action_result=False,
        )
        resume = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "receipt": str(receipt),
            "receipt_sha256": receipt_sha,
            "receipt_digest": retained["receipt_digest"],
        }
    elif any(
        getattr(args, name, None)
        for name in (
            "resume_checkpoint",
            "expected_resume_checkpoint_sha256",
            "resume_receipt",
            "expected_resume_receipt_sha256",
        )
    ):
        fail("non-resume arm must not consume an R64 checkpoint")
    _validate_port(master_port, label=f"{arm_id} master port")
    unsigned = {
        "schema_version": PAIR_PLAN_SCHEMA,
        "arm_id": arm_id,
        "binding": binding,
        "run_root": str(run_root),
        "master_port": master_port,
        "parallel_topology": PARALLEL_TOPOLOGY,
        "world_size": WORLD_SIZE,
        "gpu_memory_limit_gib_strictly_less_than": GPU_MEMORY_LIMIT_GIB,
        "host_memory_limit_gib_strictly_less_than": HOST_MEMORY_LIMIT_GIB,
        "single_model_per_node": True,
        "rank_partition_by_action_or_actor": False,
        "input_closure": closure,
        "resume": resume,
    }
    return {**unsigned, "plan_digest": object_sha256(unsigned)}


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail(f"output must be a fresh absolute path: {path}")
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _peak_values(value: Any, *, label: str) -> list[float]:
    if isinstance(value, Mapping):
        raw = list(value.values())
    elif isinstance(value, list):
        raw = value
    else:
        fail(f"{label} must be a rank mapping or list")
    if len(raw) != WORLD_SIZE:
        fail(f"{label} must cover exact WORLD4")
    peaks: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            fail(f"{label} contains a non-numeric peak")
        number = float(item)
        if not number >= 0.0 or number == float("inf"):
            fail(f"{label} contains a non-finite peak")
        peaks.append(number)
    return peaks


def validate_training_receipt(
    path: Path,
    *,
    expected_profile: str,
    expected_steps: int,
    expected_complete_action_result: bool,
) -> Mapping[str, Any]:
    value = dict(_load_json(path, label="training receipt"))
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    resources = value.get("resources")
    distributed = value.get("distributed")
    if not isinstance(resources, Mapping) or not isinstance(distributed, Mapping):
        fail("training receipt lacks nested resource/distributed closure")
    gpu = _peak_values(
        resources.get("gpu_peak_reserved_gib_by_rank"),
        label="GPU peak-reserved receipt",
    )
    host = _peak_values(
        resources.get("host_peak_rss_gib_by_rank"), label="host peak-RSS receipt"
    )
    host_cgroup = _peak_values(
        resources.get("host_cgroup_peak_gib_by_rank"),
        label="host cgroup peak receipt",
    )
    pair_invariants = value.get("pair_invariants")
    if expected_profile in {"smoke-r", "stage-r64"}:
        observed_steps = value.get("stage_r_updates")
    elif expected_profile == "smoke-p":
        observed_steps = value.get("planner_updates")
    elif expected_profile == "smoke-o":
        observed_steps = value.get("operator_updates")
    else:
        planner_updates = value.get("planner_updates")
        operator_updates = value.get("operator_updates")
        observed_steps = (
            planner_updates + operator_updates
            if type(planner_updates) is int and type(operator_updates) is int
            else None
        )
    if (
        value.get("schema_version") != TRAINING_RECEIPT_SCHEMA
        or declared != object_sha256(unsigned)
        or value.get("complete") is not True
        or value.get("execution_profile") != expected_profile
        or observed_steps != expected_steps
        or distributed.get("topology") != PARALLEL_TOPOLOGY
        or distributed.get("world_size") != WORLD_SIZE
        or distributed.get("one_shared_model") is not True
        or distributed.get("same_logical_row_on_all_ranks") is not True
        or distributed.get("rank_action_family_partition") is not False
        or value.get("complete_action_result") is not expected_complete_action_result
        or not isinstance(pair_invariants, Mapping)
        or max(gpu) >= GPU_MEMORY_LIMIT_GIB
        or max(host) >= HOST_MEMORY_LIMIT_GIB
        or max(host_cgroup) >= HOST_MEMORY_LIMIT_GIB
    ):
        fail("training receipt profile/topology/authority/memory gate differs")
    return value


def _terminate_process_groups(
    processes: Sequence[subprocess.Popen[Any]], *, timeout_seconds: float = 30.0
) -> None:
    """Signal only local wrapper sessions; retained job IDs are never targets."""

    groups = [process.pid for process in processes]
    for group, process in zip(groups, processes):
        if process.poll() is not None:
            continue
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes):
            break
        time.sleep(0.2)
    for group, process in zip(groups, processes):
        if process.poll() is None:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise GenericActionPairError(
                f"wrapper process group survived SIGKILL: {process.pid}"
            ) from error


def _arm_environment(
    args: argparse.Namespace,
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
) -> Mapping[str, str]:
    binding = plan["binding"]
    closure = plan["input_closure"]
    resume = plan["resume"]
    environment = dict(os.environ)
    environment.update(
        {
            "GSA_CONFIRM_CHILD": LAUNCH_CONFIRMATION,
            "GSA_ARM_ID": str(plan["arm_id"]),
            "GSA_HOLDER_JOB": str(binding["holder_job"]),
            "GSA_HOLDER_NODE": str(binding["holder_node"]),
            "GSA_EXECUTION_PROFILE": str(binding["execution_profile"]),
            "GSA_CARRIER_POLICY": str(binding["carrier_policy"]),
            "GSA_RUN_ROOT": str(plan["run_root"]),
            "GSA_MASTER_PORT": str(plan["master_port"]),
            "GSA_AUTHORITY_PLAN": str(plan_path),
            "GSA_AUTHORITY_PLAN_SHA256": file_sha256(plan_path),
            "GSA_METHOD_ROOT": str(closure["method_root"]),
            "GSA_TRAINER_SHA256": str(closure["trainer_sha256"]),
            "GSA_CORE_SHA256": str(closure["core_sha256"]),
            "GSA_LAUNCHER_SHA256": str(closure["launcher_sha256"]),
            "GSA_METHOD_ARCHIVE": str(closure["method_archive"]),
            "GSA_METHOD_ARCHIVE_SHA256": str(closure["method_archive_sha256"]),
            "GSA_METHOD_MANIFEST": str(closure["method_manifest"]),
            "GSA_METHOD_MANIFEST_SHA256": str(closure["method_manifest_sha256"]),
            "GSA_SOURCE_MANIFEST": str(closure["source_manifest"]),
            "GSA_SOURCE_MANIFEST_SHA256": str(closure["source_manifest_sha256"]),
            "GSA_PYTHON_BIN": str(args.python_bin),
        }
    )
    if closure["manifest_validator_sha256"] is not None:
        environment["GSA_MANIFEST_VALIDATOR_SHA256"] = str(
            closure["manifest_validator_sha256"]
        )
    else:
        environment.pop("GSA_MANIFEST_VALIDATOR_SHA256", None)
    for field, variable in (
        ("representation_manifest", "GSA_REPRESENTATION_MANIFEST"),
        ("representation_manifest_sha256", "GSA_REPRESENTATION_MANIFEST_SHA256"),
        ("source_pair_manifest", "GSA_SOURCE_PAIR_MANIFEST"),
        ("source_pair_manifest_sha256", "GSA_SOURCE_PAIR_MANIFEST_SHA256"),
    ):
        if closure[field] is not None:
            environment[variable] = str(closure[field])
        else:
            environment.pop(variable, None)
    for field, variable in (
        ("checkpoint", "GSA_RESUME_CHECKPOINT"),
        ("checkpoint_sha256", "GSA_RESUME_CHECKPOINT_SHA256"),
        ("receipt", "GSA_RESUME_RECEIPT"),
        ("receipt_sha256", "GSA_RESUME_RECEIPT_SHA256"),
    ):
        if resume is not None:
            environment[variable] = str(resume[field])
        else:
            environment.pop(variable, None)
    return environment


def _launch_plans(
    args: argparse.Namespace,
    *,
    plans: Sequence[Mapping[str, Any]],
    pair_root: Path,
    concurrent: bool,
) -> list[Mapping[str, Any]]:
    if args.confirm_launch != LAUNCH_CONFIRMATION:
        fail("explicit launch confirmation differs")
    pair_root.mkdir(mode=0o700)
    (pair_root / "logs").mkdir(mode=0o700)
    processes: list[subprocess.Popen[Any]] = []
    logs: list[Any] = []
    plan_paths: list[Path] = []
    try:
        for plan in plans:
            plan_path = pair_root / f"{plan['arm_id']}.plan.json"
            _write_create_only(plan_path, plan)
            plan_paths.append(plan_path)
            run_root = Path(str(plan["run_root"]))
            log = (pair_root / "logs" / f"{plan['arm_id']}.controller.log").open(
                "xb", buffering=0
            )
            logs.append(log)
            process = subprocess.Popen(
                [
                    "bash",
                    str(Path(str(plan["input_closure"]["method_root"])) / COMMON_LAUNCHER),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_arm_environment(args, plan=plan, plan_path=plan_path),
                start_new_session=True,
            )
            processes.append(process)
            if not concurrent:
                status = process.wait()
                if status != 0:
                    fail(f"serial arm failed: {plan['arm_id']} exit={status}")
        if concurrent:
            while True:
                statuses = [process.poll() for process in processes]
                if any(status not in (None, 0) for status in statuses):
                    fail(f"paired child failed: {statuses}")
                if all(status == 0 for status in statuses):
                    break
                time.sleep(5)
    except BaseException:
        _terminate_process_groups(processes)
        raise
    finally:
        for log in logs:
            log.close()

    receipts: list[Mapping[str, Any]] = []
    for plan in plans:
        binding = plan["binding"]
        receipt_path = Path(str(plan["run_root"])) / "training" / "run_receipt.json"
        receipts.append(
            validate_training_receipt(
                receipt_path,
                expected_profile=str(binding["execution_profile"]),
                expected_steps=int(binding["optimizer_steps"]),
                expected_complete_action_result=bool(
                    binding["complete_action_result"]
                ),
            )
        )
    return receipts


def seal_formal_pair(
    main_receipt_path: Path,
    control_receipt_path: Path,
) -> Mapping[str, Any]:
    main = validate_training_receipt(
        main_receipt_path,
        expected_profile="resume-po40",
        expected_steps=40,
        expected_complete_action_result=True,
    )
    control = validate_training_receipt(
        control_receipt_path,
        expected_profile="action-only40",
        expected_steps=40,
        expected_complete_action_result=True,
    )
    if main["pair_invariants"] != control["pair_invariants"]:
        fail("main/control shared base, manifests, P/O order, seed, or initialization differ")
    unsigned = {
        "schema_version": PAIR_RECEIPT_SCHEMA,
        "complete": True,
        "decision": "paired_training_complete_review_still_required",
        "main": {
            "path": str(main_receipt_path),
            "file_sha256": file_sha256(main_receipt_path),
            "receipt_digest": main["receipt_digest"],
        },
        "control": {
            "path": str(control_receipt_path),
            "file_sha256": file_sha256(control_receipt_path),
            "receipt_digest": control["receipt_digest"],
        },
        "shared_pair_invariants": main["pair_invariants"],
        "world4_children_are_independent_single_models": True,
        "rank_partition_by_action_or_actor": False,
        "parent_allocations_released": False,
        "decoded_review_complete": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--method-archive", required=True)
    parser.add_argument("--expected-method-archive-sha256", required=True)
    parser.add_argument("--method-manifest", required=True)
    parser.add_argument("--expected-method-manifest-sha256", required=True)
    parser.add_argument("--expected-trainer-sha256", required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("--expected-manifest-validator-sha256")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument(
        "--expected-source-manifest-sha256", default=SOURCE_MANIFEST_SHA256
    )
    parser.add_argument("--representation-manifest")
    parser.add_argument("--expected-representation-manifest-sha256")
    parser.add_argument("--source-pair-manifest")
    parser.add_argument("--expected-source-pair-manifest-sha256")
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--expected-resume-checkpoint-sha256")
    parser.add_argument("--resume-receipt")
    parser.add_argument("--expected-resume-receipt-sha256")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan-arm")
    _add_common(plan)
    plan.add_argument("--arm-id", choices=tuple(ARM_BINDINGS), required=True)
    plan.add_argument("--run-root", required=True)
    plan.add_argument("--master-port", type=int, required=True)

    launch = commands.add_parser("launch-arm")
    _add_common(launch)
    launch.add_argument("--arm-id", choices=tuple(ARM_BINDINGS), required=True)
    launch.add_argument("--pair-root", required=True)
    launch.add_argument("--master-port", type=int, required=True)
    launch.add_argument("--confirm-launch", required=True)

    smoke = commands.add_parser("launch-smokes")
    _add_common(smoke)
    smoke.add_argument("--pair-root", required=True)
    smoke.add_argument("--base-master-port", type=int, required=True)
    smoke.add_argument("--confirm-launch", required=True)

    seal = commands.add_parser("seal-formal-pair")
    seal.add_argument("--main-receipt", required=True)
    seal.add_argument("--control-receipt", required=True)
    seal.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "seal-formal-pair":
        main_receipt = _plain_file(args.main_receipt, label="main receipt")
        control_receipt = _plain_file(args.control_receipt, label="control receipt")
        output = _fresh_path(args.output, label="formal pair receipt")
        value = seal_formal_pair(main_receipt, control_receipt)
        _write_create_only(output, value)
        print(canonical_json_bytes(value).decode("ascii"), flush=True)
        return 0

    if args.command == "plan-arm":
        run_root = _fresh_path(args.run_root, label="arm run root")
        plan = build_arm_plan(
            args,
            arm_id=args.arm_id,
            run_root=run_root,
            master_port=args.master_port,
        )
        print(canonical_json_bytes(plan).decode("ascii"), flush=True)
        return 0

    pair_root = _fresh_path(args.pair_root, label="controller output root")
    if args.command == "launch-arm":
        run_root = pair_root / args.arm_id
        plan = build_arm_plan(
            args,
            arm_id=args.arm_id,
            run_root=run_root,
            master_port=args.master_port,
        )
        receipts = _launch_plans(
            args, plans=[plan], pair_root=pair_root, concurrent=False
        )
        print(canonical_json_bytes(receipts[0]).decode("ascii"), flush=True)
        return 0

    if args.command == "launch-smokes":
        _validate_port(args.base_master_port, label="smoke base master port")
        if args.base_master_port > 65533:
            fail("smoke base port leaves no room for three distinct ports")
        plans = [
            build_arm_plan(
                args,
                arm_id=arm_id,
                run_root=pair_root / arm_id,
                master_port=args.base_master_port + index,
            )
            for index, arm_id in enumerate(("smoke_r", "smoke_p", "smoke_o"))
        ]
        receipts = _launch_plans(
            args, plans=plans, pair_root=pair_root, concurrent=False
        )
        summary = {
            "complete": True,
            "profiles": [row["execution_profile"] for row in receipts],
            "disposable": True,
            "formal_checkpoint_consumed": False,
        }
        print(canonical_json_bytes(summary).decode("ascii"), flush=True)
        return 0
    fail("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
