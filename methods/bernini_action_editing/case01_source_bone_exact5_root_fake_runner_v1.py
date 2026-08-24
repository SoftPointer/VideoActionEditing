#!/usr/bin/env python3
"""No-torch fake runner used to exercise the exact5 captured root bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types
from typing import Any, Mapping, Sequence


CAMPAIGN = "case01-source-bone-exact5-r64-canary"
ENTRY_SCHEMA = "full644-exploratory-matched-captured-runner-entry-authority-v1"
TASK_IDS = (
    "case01-exact_original-full644", "case01-codec_only_present-full644",
    "case01-bone_removed-full644", "case01-bone_translated_up150-full644",
    "case01-sham_control_up150-full644",
)
SHA_RE = re.compile(r"[0-9a-f]{64}")
EXACT5_EVAL_SHA256 = (
    "891551d46b0ca11362fe8d446f202036b9957baa943de0aec6da1f3ad45d7d58"
)


class FakeRunnerError(RuntimeError):
    """The captured-entry fake-runner contract differs."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def ident(info: os.stat_result) -> dict[str, int]:
    return {"device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid,
            "gid": info.st_gid, "mode": info.st_mode, "nlink": info.st_nlink,
            "rdev": info.st_rdev, "size": info.st_size,
            "blocks": getattr(info, "st_blocks", 0), "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns}


def stable(path: Path, expected: str, mode: int | None = None) -> bytes:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path) or path.is_symlink() or path.resolve(strict=True) != path:
        raise FakeRunnerError(f"noncanonical named authority: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd); chunks = []; offset = 0
        while offset < before.st_size:
            block = os.pread(fd, min(1_048_576, before.st_size - offset), offset)
            if not block: break
            chunks.append(block); offset += len(block)
        after = os.fstat(fd); named = path.lstat()
    finally: os.close(fd)
    raw = b"".join(chunks)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not raw
            or len(raw) != before.st_size or ident(before) != ident(after)
            or ident(before) != ident(named) or hashlib.sha256(raw).hexdigest() != expected
            or (mode is not None and stat.S_IMODE(before.st_mode) != mode)):
        raise FakeRunnerError(f"named authority differs: {path}")
    return raw


def load_module(name: str, path: Path, expected: str) -> types.ModuleType:
    raw = stable(path, expected, 0o444)
    module = types.ModuleType(name); module.__file__ = str(path)
    module.__package__ = None; module.__loader__ = None; module.__spec__ = None
    module.__cached__ = None; module.__builtins__ = __builtins__; sys.modules[name] = module
    exec(compile(raw.decode("utf-8", "strict"), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def create(path: Path, value: Mapping[str, Any]) -> None:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or path.name in {"", ".", ".."}
    ):
        raise FakeRunnerError("receipt target path differs")
    raw = canonical(value) + b"\n"
    parent = path.parent
    parent_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | os.O_CLOEXEC,
    )
    try:
        parent_info = os.fstat(parent_fd); parent_named = os.lstat(parent)
        if (
            os.path.realpath(parent) != str(parent)
            or not stat.S_ISDIR(parent_info.st_mode)
            or ident(parent_info) != ident(parent_named)
            or parent_info.st_uid != 2012 or parent_info.st_gid != 2000
            or stat.S_IMODE(parent_info.st_mode) != 0o755
        ):
            raise FakeRunnerError("receipt parent authority differs")
        fd = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0, dir_fd=parent_fd,
        )
        try:
            offset = 0
            while offset < len(raw):
                count = os.write(fd, raw[offset:])
                if count <= 0:
                    raise FakeRunnerError("receipt write made no progress")
                offset += count
            os.fsync(fd); before = os.fstat(fd)
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                stat.S_IMODE(before.st_mode) != 0 or before.st_nlink != 1
                or before.st_uid != 2012 or before.st_gid != 2000
                or ident(before) != ident(named)
                or os.pread(fd, len(raw), 0) != raw
            ):
                raise FakeRunnerError("receipt staging replay differs")
            os.fchmod(fd, 0o400); os.fsync(fd)
            after = os.fstat(fd)
            named_after = os.stat(
                path.name, dir_fd=parent_fd, follow_symlinks=False,
            )
            if (
                stat.S_IMODE(after.st_mode) != 0o400
                or ident(after) != ident(named_after)
                or os.pread(fd, len(raw), 0) != raw
            ):
                raise FakeRunnerError("receipt commit replay differs")
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "campaign-mode", "plan", "plan-sha256", "output-report",
        "runner-attestation", "runner-sha256", "bridge-script",
        "bridge-script-sha256", "adapter-script", "adapter-script-sha256",
        "eval-v1-source",
        "eval-v1-source-sha256", "eval-v2-source", "eval-v2-source-sha256",
        "model-authority-source", "model-authority-source-sha256",
        "python", "python-sha256",
        "ffmpeg-executable", "ffmpeg-executable-sha256", "torchrun-source",
        "torchrun-source-sha256", "torchrun-handler-source",
        "torchrun-handler-source-sha256", "torch-local-agent-source",
        "torch-local-agent-source-sha256", "torch-dynamic-rendezvous-source",
        "torch-dynamic-rendezvous-source-sha256",
        "torch-multiprocessing-api-source", "torch-multiprocessing-api-source-sha256",
        "model-root", "model-manifest", "model-manifest-sha256", "bernini-root",
        "veomni-root", "authority-root", "rank-cache-root", "holder-job-id",
        "expected-node", "expected-allocation-gpu-count",
    ):
        parser.add_argument("--" + name, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.campaign_mode != CAMPAIGN or args.holder_job_id != "143808" or args.expected_node != "auh7-1b-gpu-292" or args.expected_allocation_gpu_count != "8":
        raise FakeRunnerError("fake runner campaign/allocation argument differs")
    expected_environment = {
        "SLURM_JOB_ID": "143808", "SLURM_GPUS_ON_NODE": "8", "SLURM_GPUS_PER_NODE": "8",
        "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7", "SLURM_NNODES": "1",
        "SLURM_STEP_NUM_NODES": "1", "SLURM_JOB_NODELIST": "auh7-1b-gpu-292",
        "SLURM_STEP_NODELIST": "auh7-1b-gpu-292",
    }
    allowed = set(expected_environment) | {"SLURM_STEP_ID", "FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY"}
    if set(os.environ) != allowed or any(os.environ.get(key) != value for key, value in expected_environment.items()):
        raise FakeRunnerError("captured root environment closure differs")
    step = os.environ.get("SLURM_STEP_ID", "")
    if not step.isascii() or not step.isdecimal() or int(step) <= 394 or str(int(step)) != step:
        raise FakeRunnerError("captured root numeric step differs")
    try:
        entry = json.loads(os.environ["FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY"])
    except (KeyError, json.JSONDecodeError) as error:
        raise FakeRunnerError("captured entry authority JSON differs") from error
    unsigned = dict(entry) if isinstance(entry, dict) else {}
    claimed = unsigned.pop("authority_digest", None)
    source_path = Path(__file__)
    source_raw = stable(source_path, args.runner_sha256, 0o444)
    if (entry.get("schema_version") != ENTRY_SCHEMA or claimed != digest(unsigned)
            or entry.get("runner_path") != str(source_path)
            or entry.get("runner_sha256") != args.runner_sha256
            or entry.get("runner_identity") != ident(source_path.lstat())
            or entry.get("captured_source_entry") is not True
            or entry.get("slurm_export_none_required") is not True
            or entry.get("bash_privileged_startup_required") is not True
            or entry.get("entry_method") != "slurm-spooled-or-trusted-stdin-held-python-fd-v1"
            or SHA_RE.fullmatch(entry.get("release_digest", "")) is None
            or hashlib.sha256(source_raw).hexdigest() != args.runner_sha256):
        raise FakeRunnerError("captured runner entry authority differs")
    exact_eval_path = source_path.parent / "case01_source_bone_exact5_eval_v1.py"
    exact_eval = load_module(
        "_exact5_fake_eval", exact_eval_path, EXACT5_EVAL_SHA256,
    )
    plan = exact_eval.load_plan(args.plan, args.plan_sha256)
    if plan.get("task_count") != 5 or [row.get("task_id") for row in plan.get("tasks", [])] != list(TASK_IDS):
        raise FakeRunnerError("fake runner exact5 plan differs")
    hash_values = [value for key, value in vars(args).items() if key.endswith("sha256")]
    if any(SHA_RE.fullmatch(value) is None for value in hash_values):
        raise FakeRunnerError("fake runner SHA argument differs")
    for path_value in (args.authority_root, args.rank_cache_root, args.runner_attestation):
        path = Path(path_value)
        if path.exists() or path.is_symlink():
            raise FakeRunnerError("diagnostic unused target is not fresh")
    if "torch" in sys.modules:
        raise FakeRunnerError("captured fake runner imported torch")
    receipt: dict[str, Any] = {
        "schema_version": "case01-source-bone-exact5-root-fake-runner-probe-v1",
        "status": "PASS", "campaign_mode": CAMPAIGN,
        "holder_job_id": args.holder_job_id, "expected_node": args.expected_node,
        "slurm_step_id": step, "task_count": 5,
        "selected_task_ids": list(TASK_IDS), "plan_sha256": args.plan_sha256,
        "plan_digest": plan["plan_digest"], "runner_sha256": args.runner_sha256,
        "entry_authority_digest": entry["authority_digest"],
        "release_digest": entry["release_digest"],
        "captured_source_entry": True, "held_python_fd_entry": True,
        "all_exact18_named_identities_replayed_by_root_bootstrap": True,
        "slurm_environment_from_step": True, "torch_imported": False,
        "renderer_executed": False,
    }
    receipt["receipt_digest"] = digest(receipt)
    create(Path(args.output_report), receipt)
    print("CASE01_EXACT5_ROOT_FAKE_PASS " + receipt["receipt_digest"])
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform != "linux" or not Path("/proc/self/fd").is_dir() or sys.flags.isolated != 1 or sys.flags.no_site != 1 or sys.flags.ignore_environment != 1 or not sys.dont_write_bytecode:
        raise FakeRunnerError("isolated fake-runner startup differs")
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
