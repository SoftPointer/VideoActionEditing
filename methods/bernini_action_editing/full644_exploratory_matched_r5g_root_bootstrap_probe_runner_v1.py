#!/usr/bin/env python3
"""CPU-only runner used to attest the r5g full16 captured-source bootstrap.

This program is substituted for the production runner only while the r5d
launcher materializes its CPU bootstrap diagnostic.  It validates the exact
Shared8 Base/R64 full16 argv and captured-entry authority, records the AUH Slurm
source fields, and publishes one create-only receipt.  It deliberately does
not import Torch or open the model, checkpoint, source videos, authority root,
or rank cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "full644-exploratory-matched-r5g-full16-root-bootstrap-cpu-probe-v1"
ENTRY_SCHEMA = "full644-exploratory-matched-captured-runner-entry-authority-v1"
ENTRY_ENV = "FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY"
CAMPAIGN = "full16-production"
TASK_IDS = tuple(
    f"shared8-{index:02d}-{arm}"
    for index in range(8)
    for arm in ("base", "full644")
)
SLURM_FIELDS = (
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_STEP_GPUS",
    "SLURM_NNODES",
    "SLURM_STEP_NUM_NODES",
    "SLURM_JOB_NODELIST",
    "SLURM_STEP_NODELIST",
)
SLURM_ABSENT_FIELDS = ("SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class R5DRootBootstrapProbeError(RuntimeError):
    """The isolated CPU bootstrap contract differs."""


def validate_campaign_contract(campaign: Any, selected: Sequence[Any]) -> None:
    if campaign != CAMPAIGN or tuple(selected) != TASK_IDS:
        raise R5DRootBootstrapProbeError("full16 selection differs")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise R5DRootBootstrapProbeError("duplicate JSON key")
        result[key] = value
    return result


def strict_json_text(raw: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise R5DRootBootstrapProbeError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value).decode("utf-8") != raw:
        raise R5DRootBootstrapProbeError(f"{label} is not canonical JSON")
    return value


def _identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": info.st_mode,
        "nlink": info.st_nlink,
        "rdev": info.st_rdev,
        "size": info.st_size,
        "blocks": getattr(info, "st_blocks", 0),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _pread_exact(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size <= 0:
        raise R5DRootBootstrapProbeError("captured entry size differs")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size:
        raise R5DRootBootstrapProbeError("captured entry read is incomplete")
    return raw


def validate_entry(value: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    fields = {
        "schema_version",
        "runner_fd",
        "runner_path",
        "runner_sha256",
        "runner_identity",
        "python_fd",
        "python_path",
        "python_sha256",
        "python_identity",
        "release_digest",
        "bootstrap_sha256",
        "entry_method",
        "slurm_export_none_required",
        "bash_privileged_startup_required",
        "captured_source_entry",
        "authority_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise R5DRootBootstrapProbeError("captured entry field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    runner_identity = row.get("runner_identity")
    python_identity = row.get("python_identity")
    if (
        row.get("schema_version") != ENTRY_SCHEMA
        or claimed != object_sha256(unsigned)
        or row.get("entry_method")
        != "slurm-spooled-or-trusted-stdin-held-python-fd-v1"
        or row.get("slurm_export_none_required") is not True
        or row.get("bash_privileged_startup_required") is not True
        or row.get("captured_source_entry") is not True
        or type(row.get("runner_fd")) is not int
        or type(row.get("python_fd")) is not int
        or row["runner_fd"] < 3
        or row["python_fd"] < 3
        or row["runner_fd"] == row["python_fd"]
        or not isinstance(runner_identity, dict)
        or not isinstance(python_identity, dict)
        or set(runner_identity) != set(_identity(os.fstat(row["runner_fd"])))
        or set(python_identity) != set(_identity(os.fstat(row["python_fd"])))
        or any(type(item) is not int for item in (*runner_identity.values(), *python_identity.values()))
        or any(
            type(row.get(field)) is not str or SHA256_RE.fullmatch(row[field]) is None
            for field in ("runner_sha256", "python_sha256", "release_digest", "bootstrap_sha256")
        )
    ):
        raise R5DRootBootstrapProbeError("captured entry value differs")
    runner_path = Path(row["runner_path"])
    python_path = Path(row["python_path"])
    if (
        not runner_path.is_absolute()
        or not python_path.is_absolute()
        or os.path.normpath(str(runner_path)) != str(runner_path)
        or os.path.normpath(str(python_path)) != str(python_path)
        or runner_path != Path(__file__).resolve(strict=True)
        or args.runner_sha256 != row["runner_sha256"]
        or args.python_sha256 != row["python_sha256"]
        or Path(args.python) != python_path
    ):
        raise R5DRootBootstrapProbeError("captured entry path binding differs")
    for descriptor, path, expected_identity, expected_sha, executable in (
        (row["runner_fd"], runner_path, runner_identity, row["runner_sha256"], False),
        (row["python_fd"], python_path, python_identity, row["python_sha256"], True),
    ):
        before = os.fstat(descriptor)
        raw = _pread_exact(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (not executable and stat.S_IMODE(before.st_mode) != 0o444)
            or (executable and not before.st_mode & 0o111)
            or _identity(before) != expected_identity
            or _identity(after) != expected_identity
            or _identity(named) != expected_identity
            or hashlib.sha256(raw).hexdigest() != expected_sha
            or os.get_inheritable(descriptor)
        ):
            raise R5DRootBootstrapProbeError("captured entry FD replay differs")
    if _identity(os.stat("/proc/self/exe")) != python_identity:
        raise R5DRootBootstrapProbeError("running Python identity differs")
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-mode", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--runner-attestation", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--bridge-script", required=True)
    parser.add_argument("--bridge-script-sha256", required=True)
    parser.add_argument("--adapter-script", required=True)
    parser.add_argument("--adapter-script-sha256", required=True)
    parser.add_argument("--eval-v1-source", required=True)
    parser.add_argument("--eval-v1-source-sha256", required=True)
    parser.add_argument("--eval-v2-source", required=True)
    parser.add_argument("--eval-v2-source-sha256", required=True)
    parser.add_argument("--model-authority-source", required=True)
    parser.add_argument("--model-authority-source-sha256", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--ffmpeg-executable", required=True)
    parser.add_argument("--ffmpeg-executable-sha256", required=True)
    parser.add_argument("--torchrun-source", required=True)
    parser.add_argument("--torchrun-source-sha256", required=True)
    parser.add_argument("--torchrun-handler-source", required=True)
    parser.add_argument("--torchrun-handler-source-sha256", required=True)
    parser.add_argument("--torch-local-agent-source", required=True)
    parser.add_argument("--torch-local-agent-source-sha256", required=True)
    parser.add_argument("--torch-dynamic-rendezvous-source", required=True)
    parser.add_argument("--torch-dynamic-rendezvous-source-sha256", required=True)
    parser.add_argument("--torch-multiprocessing-api-source", required=True)
    parser.add_argument("--torch-multiprocessing-api-source-sha256", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--rank-cache-root", required=True)
    parser.add_argument("--holder-job-id", required=True)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--expected-allocation-gpu-count", required=True, type=int)
    return parser


def _validate_paths(args: argparse.Namespace) -> None:
    existing_files = (
        args.plan,
        args.bridge_script,
        args.adapter_script,
        args.eval_v1_source,
        args.eval_v2_source,
        args.model_authority_source,
        args.python,
        args.ffmpeg_executable,
        args.torchrun_source,
        args.torchrun_handler_source,
        args.torch_local_agent_source,
        args.torch_dynamic_rendezvous_source,
        args.torch_multiprocessing_api_source,
        args.model_manifest,
    )
    directories = (args.model_root, args.bernini_root, args.veomni_root)
    fresh = (
        args.output_report,
        args.runner_attestation,
        args.authority_root,
        args.rank_cache_root,
    )
    for raw in existing_files:
        path = Path(raw)
        if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
            raise R5DRootBootstrapProbeError("bootstrap file path differs")
    for raw in directories:
        path = Path(raw)
        if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
            raise R5DRootBootstrapProbeError("bootstrap directory path differs")
    for raw in fresh:
        path = Path(raw)
        if (
            not path.is_absolute()
            or os.path.normpath(str(path)) != str(path)
            or path.exists()
            or path.is_symlink()
            or not path.parent.is_dir()
            or path.parent.is_symlink()
        ):
            raise R5DRootBootstrapProbeError("bootstrap fresh path differs")


def _publish_receipt(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor = os.open(
        path.name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0,
        dir_fd=parent_fd,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise R5DRootBootstrapProbeError("receipt write made no progress")
            offset += count
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0
            or before.st_nlink != 1
            or _identity(before) != _identity(named)
            or _pread_exact(descriptor, len(raw)) != raw
        ):
            raise R5DRootBootstrapProbeError("receipt staging replay differs")
        sentinel = f"R5G_FULL16_ROOT_BOOTSTRAP_CPU_PROBE_PASS {value['receipt_digest']}\n".encode("ascii")
        if os.write(1, sentinel) != len(sentinel):
            raise R5DRootBootstrapProbeError("receipt sentinel write differs")
        os.fchmod(descriptor, 0o400)
        os._exit(0)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    if (
        sys.platform != "linux"
        or not Path("/proc/self/fd").is_dir()
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or not sys.dont_write_bytecode
        or "torch" in sys.modules
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise R5DRootBootstrapProbeError("isolated CPU bootstrap differs")
    args = build_parser().parse_args(argv)
    validate_campaign_contract(args.campaign_mode, TASK_IDS)
    if (
        args.expected_allocation_gpu_count != 8
        or type(args.holder_job_id) is not str
        or not args.holder_job_id.isascii()
        or not args.holder_job_id.isdecimal()
        or str(int(args.holder_job_id)) != args.holder_job_id
        or not args.expected_node.startswith("auh7-1b-gpu-")
        or any(
            SHA256_RE.fullmatch(getattr(args, field)) is None
            for field in vars(args)
            if field.endswith("sha256")
        )
    ):
        raise R5DRootBootstrapProbeError("bootstrap argv semantics differ")
    _validate_paths(args)
    raw_entry = os.environ.get(ENTRY_ENV)
    if raw_entry is None:
        raise R5DRootBootstrapProbeError("captured entry is absent")
    entry = validate_entry(strict_json_text(raw_entry, label="captured entry"), args)
    if any(field in os.environ for field in SLURM_ABSENT_FIELDS):
        raise R5DRootBootstrapProbeError("unsupported Slurm field is present")
    slurm = {field: os.environ.get(field) for field in SLURM_FIELDS}
    step_id = slurm.get("SLURM_STEP_ID")
    if (
        slurm
        != {
            "SLURM_JOB_ID": args.holder_job_id,
            "SLURM_STEP_ID": step_id,
            "SLURM_GPUS_ON_NODE": "8",
            "SLURM_GPUS_PER_NODE": "8",
            "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7",
            "SLURM_NNODES": "1",
            "SLURM_STEP_NUM_NODES": "1",
            "SLURM_JOB_NODELIST": args.expected_node,
            "SLURM_STEP_NODELIST": args.expected_node,
        }
        or type(step_id) is not str
        or not step_id.isascii()
        or not step_id.isdecimal()
        or int(step_id) <= 0
        or str(int(step_id)) != step_id
    ):
        raise R5DRootBootstrapProbeError("AUH Slurm source contract differs")
    fd_targets: list[str] = []
    for name in os.listdir("/proc/self/fd"):
        if not name.isdecimal():
            continue
        try:
            target = os.readlink("/proc/self/fd/" + name)
        except OSError:
            continue
        if (
            target.startswith(args.model_root)
            or "checkpoint-00000644" in target
            or target == "/dev/kfd"
            or target.startswith("/dev/dri/")
        ):
            fd_targets.append(target)
    if fd_targets:
        raise R5DRootBootstrapProbeError("GPU/model FD target was opened")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "campaign_mode": CAMPAIGN,
        "selected_task_ids": list(TASK_IDS),
        "unselected_task_count": 0,
        "plan_path": args.plan,
        "plan_sha256": args.plan_sha256,
        "slurm_step_id": step_id,
        "slurm_environment_source_names": sorted(SLURM_FIELDS),
        "slurm_fields_observed_absent": sorted(SLURM_ABSENT_FIELDS),
        "captured_source_entry": True,
        "entry_authority_digest": entry["authority_digest"],
        "release_digest": entry["release_digest"],
        "isolated_python": True,
        "torch_imported": False,
        "gpu_or_model_payload_fd_targets_observed_at_probe_end": [],
        "probe_runner_opened_model_or_checkpoint_payload": False,
        "probe_runner_imported_or_executed_torch": False,
        "gpu_device_fd_observed_at_probe_end": False,
        "formal_report_generated": False,
        "html_generated": False,
        "formal_full16_report": True,
        "canary_stops_after_pair_for_manual_visual_review": False,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    _publish_receipt(Path(args.output_report), receipt)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
