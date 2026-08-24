#!/usr/bin/env python3
"""Capture the exact PyTorch 2.7.1 torchrun worker environment at _popen.

This diagnostic deliberately runs only four ``/usr/bin/true`` children.  It
does not import any Bernini/VACE model module, initialize CUDA, or form a
worker process group.  The task-local patch observes the final environment
constructed by PyTorch's real ``SubprocessHandler`` and then delegates to the
original ``_popen`` implementation.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "full644-exploratory-matched-rank-env-capture-probe-v1"
TORCH_VERSION = "2.7.1+rocm6.3"
TRUE_SHA256 = "89c77cc9a7d6432f3efba4bf2699764a0f2084892cbb1914bcb1c741450c8779"
TRUE_PATH = Path("/usr/bin/true")
PRODUCERS = {
    "torchrun": (
        "torch/distributed/run.py",
        "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c",
    ),
    "subprocess_handler": (
        "torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py",
        "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87",
    ),
    "local_elastic_agent": (
        "torch/distributed/elastic/agent/server/local_elastic_agent.py",
        "71f390071316417643aa91514ebb170b3adb7eca5c1fe8286d03fe2eef21e497",
    ),
    "dynamic_rendezvous": (
        "torch/distributed/elastic/rendezvous/dynamic_rendezvous.py",
        "adc34f683614cdc6de5f5cc64e34ee7201b0671609a7ee574b9731f4266e5cec",
    ),
    "multiprocessing_api": (
        "torch/distributed/elastic/multiprocessing/api.py",
        "f815c915fd857bbff12b4d00530c7c1ffb0badfcd48c41e7f378c65828192ef7",
    ),
}
WORKER_KEYS = {
    "LOCAL_RANK",
    "RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "GROUP_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_ERROR_FILE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "OMP_NUM_THREADS",
}
UUID4_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


class ProbeError(RuntimeError):
    """The diagnostic did not execute the exact expected CPU-only path."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def stat_identity(info: os.stat_result) -> dict[str, int]:
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


def hash_fd(descriptor: int, size: int) -> str:
    if type(size) is not int or size <= 0:
        raise ProbeError("producer size differs")
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise ProbeError("producer short read")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        raise ProbeError("producer grew during read")
    return digest.hexdigest()


def open_pinned_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise ProbeError(f"producer path differs: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
        ):
            raise ProbeError(f"producer identity differs: {path}")
        observed_sha256 = hash_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        if stat_identity(before) != stat_identity(after):
            raise ProbeError(f"producer changed during capture: {path}")
        if observed_sha256 != expected_sha256:
            raise ProbeError(f"producer digest differs: {path}")
        os.set_inheritable(descriptor, False)
        return {
            "path": str(path),
            "sha256": observed_sha256,
            "identity": stat_identity(after),
            "fd": descriptor,
        }
    except BaseException:
        os.close(descriptor)
        raise


def replay_pinned_file(row: Mapping[str, Any]) -> None:
    descriptor = row["fd"]
    current = os.fstat(descriptor)
    if (
        stat_identity(current) != row["identity"]
        or hash_fd(descriptor, current.st_size) != row["sha256"]
        or os.get_inheritable(descriptor)
    ):
        raise ProbeError(f"producer replay differs: {row['path']}")
    named = open_pinned_file(Path(row["path"]), row["sha256"])
    try:
        if named["identity"] != row["identity"]:
            raise ProbeError(f"producer named replay differs: {row['path']}")
    finally:
        os.close(named["fd"])


def parse_canonical_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if str(value) == raw else None


def expected_for_rank(rank: int, *, agent_store: str) -> dict[str, Any]:
    return {
        "LOCAL_RANK": str(rank),
        "RANK": str(rank),
        "GROUP_RANK": "0",
        "ROLE_RANK": str(rank),
        "ROLE_NAME": "default",
        "LOCAL_WORLD_SIZE": "4",
        "WORLD_SIZE": "4",
        "GROUP_WORLD_SIZE": "1",
        "ROLE_WORLD_SIZE": "4",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": {"predicate": "canonical-decimal-in-1..65535"},
        "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_MAX_RESTARTS": "0",
        "TORCHELASTIC_RUN_ID": {"predicate": "lowercase-rfc4122-uuid4"},
        "TORCHELASTIC_USE_AGENT_STORE": agent_store,
        "TORCHELASTIC_ERROR_FILE": {
            "predicate": f"absolute-normalized-.../{rank}/error.json"
        },
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "OMP_NUM_THREADS": "4",
    }


def field_matches(key: str, expected: Any, observed: str | None, rank: int) -> bool:
    if type(expected) is str:
        return observed == expected
    if key == "MASTER_PORT":
        value = parse_canonical_int(observed)
        return value is not None and value in range(1, 65536)
    if key == "TORCHELASTIC_RUN_ID":
        return observed is not None and UUID4_RE.fullmatch(observed) is not None
    if key == "TORCHELASTIC_ERROR_FILE":
        if observed is None:
            return False
        path = Path(observed)
        return (
            path.is_absolute()
            and os.path.normpath(observed) == observed
            and path.name == "error.json"
            and path.parent.name == str(rank)
        )
    raise ProbeError(f"unknown expected predicate: {key}")


def diff_environment(
    observed: Mapping[str, str], expected: Mapping[str, Any], rank: int
) -> dict[str, Any]:
    return {
        key: {"expected": expected[key], "observed": observed.get(key)}
        for key in sorted(expected)
        if not field_matches(key, expected[key], observed.get(key), rank)
    }


def gpu_device_descriptors() -> list[str]:
    paths = []
    fd_root = Path("/proc/self/fd")
    if not fd_root.is_dir():
        return paths
    for entry in fd_root.iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target == "/dev/kfd" or target.startswith("/dev/dri/"):
            paths.append(target)
    return sorted(paths)


def validate_slurm_environment(
    environment: Mapping[str, str], *, expected_job_id: str, expected_node: str,
    hostname: str,
) -> dict[str, str | None]:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_STEP_NODELIST",
        "SLURM_NNODES",
        "SLURM_STEP_NUM_NODES",
        "SLURM_GPUS_ON_NODE",
        "SLURM_GPUS_PER_NODE",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
        "SLURM_JOB_NUM_NODES",
    )
    observed = {key: environment.get(key) for key in keys}
    step_id = parse_canonical_int(observed["SLURM_STEP_ID"])
    if (
        observed["SLURM_JOB_ID"] == expected_job_id
        and step_id is not None
        and step_id > 0
        and observed["SLURM_JOB_NODELIST"] == expected_node
        and observed["SLURM_STEP_NODELIST"] == expected_node
        and observed["SLURM_NNODES"] == "1"
        and observed["SLURM_STEP_NUM_NODES"] == "1"
        and observed["SLURM_GPUS_ON_NODE"] == "8"
        and observed["SLURM_GPUS_PER_NODE"] == "8"
        and observed["SLURM_STEP_GPUS"] == "0,1,2,3,4,5,6,7"
        and observed["SLURM_JOB_GPUS"] is None
        and observed["SLURM_JOB_NUM_NODES"] is None
        and hostname == expected_node
    ):
        return observed
    raise ProbeError("Slurm CPU probe binding differs")


def forbidden_model_modules(module_names: Sequence[str]) -> list[str]:
    prefixes = ("bernini", "diffusers", "peft", "transformers", "vace", "veomni")
    exact = {
        "infer_lora",
        "full644_exploratory_matched_infer_adapter_v2",
        "full644_exploratory_matched_infer_adapter_gpu47_v3",
    }
    return sorted(
        name
        for name in module_names
        if name in exact
        or any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )


def retained_executable_popen_factory(
    *, original_popen: Any, true_path: Path, true_fd: int
) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def authority_process_popen(*popen_args: Any, **popen_kwargs: Any) -> Any:
        if (
            popen_args
            or set(popen_kwargs)
            != {"args", "env", "stdout", "stderr", "start_new_session"}
            or tuple(popen_kwargs["args"]) != (str(true_path),)
            or type(popen_kwargs["env"]) is not dict
            or popen_kwargs["start_new_session"] is not True
        ):
            raise ProbeError("SubprocessHandler Popen call differs")
        executable = f"/proc/self/fd/{true_fd}"
        calls.append(
            {
                "observed_argv": list(popen_kwargs["args"]),
                "executed_argv": [executable],
                "executable": executable,
                "pass_fds": [true_fd],
            }
        )
        return original_popen(
            args=(executable,),
            env=dict(popen_kwargs["env"]),
            stdout=popen_kwargs["stdout"],
            stderr=popen_kwargs["stderr"],
            start_new_session=True,
            close_fds=True,
            pass_fds=(true_fd,),
            executable=executable,
        )

    return authority_process_popen, calls


def write_receipt(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    digest = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    final = dict(unsigned)
    final["receipt_digest"] = digest
    raw = canonical_bytes(final) + b"\n"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise ProbeError("receipt short write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o400
            or info.st_nlink != 1
            or info.st_size != len(raw)
            or os.pread(descriptor, len(raw), 0) != raw
        ):
            raise ProbeError("receipt replay differs")
        return {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "receipt_digest": digest,
            "identity": stat_identity(info),
        }
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--probe-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if not (
        sys.flags.isolated == 1
        and sys.flags.no_site == 1
        and sys.flags.dont_write_bytecode == 1
        and sys.flags.optimize == 0
    ):
        raise ProbeError("probe interpreter flags differ")
    args = build_parser().parse_args(argv)
    site_root = Path(args.site_packages)
    work_root = Path(args.work_root)
    receipt_path = Path(args.receipt)
    true_path = TRUE_PATH
    if (
        not site_root.is_absolute()
        or site_root != site_root.resolve(strict=True)
        or not work_root.is_absolute()
        or work_root != work_root.resolve(strict=True)
        or not work_root.is_dir()
        or receipt_path.parent != work_root
        or receipt_path.exists()
    ):
        raise ProbeError("probe path contract differs")
    expected_pycache = work_root / "pycache"
    if (
        sys.pycache_prefix is not None
        or not expected_pycache.is_dir()
        or any(expected_pycache.iterdir())
    ):
        raise ProbeError("fresh bootstrap pycache differs")
    sys.pycache_prefix = str(expected_pycache)
    if sys.pycache_prefix != str(expected_pycache):
        raise ProbeError("probe pycache activation differs")
    slurm_environment = validate_slurm_environment(
        os.environ,
        expected_job_id=args.expected_job_id,
        expected_node=args.expected_node,
        hostname=socket.gethostname(),
    )
    step_id = parse_canonical_int(slurm_environment["SLURM_STEP_ID"])
    if step_id is None:
        raise ProbeError("validated Slurm step disappeared")
    tmp_root = work_root / "tmp"
    tmp_root.mkdir(mode=0o700)
    controlled_environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "4",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TMPDIR": str(tmp_root),
        "TMP": str(tmp_root),
        "TEMP": str(tmp_root),
        "HOME": str(work_root),
        "SLURM_JOB_ID": args.expected_job_id,
        "SLURM_STEP_ID": str(step_id),
    }
    os.environ.clear()
    os.environ.update(controlled_environment)
    if "TORCH_DISABLE_SHARE_RDZV_TCP_STORE" in os.environ:
        raise ProbeError("dynamic rendezvous opt-out was injected")

    cleanup = ExitStack()
    try:
        probe_row = open_pinned_file(
            Path(__file__).resolve(strict=True), args.probe_sha256
        )
        cleanup.callback(os.close, probe_row["fd"])
        producer_rows = {}
        for role, (relative, expected_sha256) in PRODUCERS.items():
            row = open_pinned_file(site_root / relative, expected_sha256)
            cleanup.callback(os.close, row["fd"])
            producer_rows[role] = row
        true_row = open_pinned_file(true_path, TRUE_SHA256)
        cleanup.callback(os.close, true_row["fd"])
        true_identity = true_row["identity"]
        if (
            stat.S_IMODE(true_identity["mode"]) != 0o755
            or true_identity["uid"] != 0
            or true_identity["gid"] != 0
            or true_identity["nlink"] != 1
        ):
            raise ProbeError("root-owned /usr/bin/true identity differs")
        before_torch_gpu_fds = gpu_device_descriptors()
        if before_torch_gpu_fds:
            raise ProbeError("GPU device was open before Torch import")
    except BaseException:
        cleanup.close()
        raise
    try:
        sys.path.append(str(site_root))
        import torch
        import torch.distributed.run as run_module
        from torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler import (
            SubprocessHandler,
        )

        imported_modules = {
            "torchrun": run_module,
            "subprocess_handler": sys.modules[
                "torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler"
            ],
            "local_elastic_agent": sys.modules[
                "torch.distributed.elastic.agent.server.local_elastic_agent"
            ],
            "dynamic_rendezvous": sys.modules[
                "torch.distributed.elastic.rendezvous.dynamic_rendezvous"
            ],
            "multiprocessing_api": sys.modules[
                "torch.distributed.elastic.multiprocessing.api"
            ],
        }
        if torch.__version__ != TORCH_VERSION:
            raise ProbeError("Torch version differs")
        for role, module in imported_modules.items():
            if Path(module.__file__).resolve(strict=True) != Path(
                producer_rows[role]["path"]
            ):
                raise ProbeError(f"producer origin differs: {role}")
        base_environment = dict(os.environ)
        original_popen = SubprocessHandler._popen
        handler_module = imported_modules["subprocess_handler"]
        original_process_popen = handler_module.subprocess.Popen
        if original_process_popen is not subprocess.Popen:
            raise ProbeError("stdlib Popen origin differs before probe patch")
        captures: list[dict[str, Any]] = []
        authority_process_popen, authority_exec_calls = (
            retained_executable_popen_factory(
                original_popen=original_process_popen,
                true_path=true_path,
                true_fd=true_row["fd"],
            )
        )

        def capture_popen(
            handler: Any, child_argv: tuple[Any, ...], child_env: dict[str, str]
        ) -> Any:
            if (
                type(child_argv) is not tuple
                or tuple(child_argv) != (str(true_path),)
                or type(child_env) is not dict
                or any(
                    type(key) is not str or type(value) is not str
                    for key, value in child_env.items()
                )
            ):
                raise ProbeError("rank spawn argv/environment shape differs")
            captures.append(
                {
                    "handler_local_rank": handler.local_rank_id,
                    "argv": list(child_argv),
                    "environment": dict(child_env),
                }
            )
            return original_popen(handler, child_argv, child_env)

        handler_module.subprocess.Popen = authority_process_popen
        SubprocessHandler._popen = capture_popen
        try:
            result = run_module.main(
                [
                    "--standalone",
                    "--nnodes=1",
                    "--nproc_per_node=4",
                    "--max_restarts=0",
                    "--local-addr=localhost",
                    "--no-python",
                    str(true_path),
                ]
            )
        finally:
            restored_handler_popen = SubprocessHandler._popen
            restored_process_popen = handler_module.subprocess.Popen
            SubprocessHandler._popen = original_popen
            handler_module.subprocess.Popen = original_process_popen
        if (
            restored_handler_popen is not capture_popen
            or restored_process_popen is not authority_process_popen
            or SubprocessHandler._popen is not original_popen
            or handler_module.subprocess.Popen is not original_process_popen
            or subprocess.Popen is not original_process_popen
        ):
            raise ProbeError("task-local SubprocessHandler patch restore differs")
        if result is not None:
            raise ProbeError("torchrun return value differs")
        if len(captures) != 4:
            raise ProbeError("torchrun did not spawn exactly four ranks")
        if len(authority_exec_calls) != 4 or os.get_inheritable(true_row["fd"]):
            raise ProbeError("held /usr/bin/true execution count differs")
        captures.sort(key=lambda row: row["handler_local_rank"])
        if [row["handler_local_rank"] for row in captures] != list(range(4)):
            raise ProbeError("captured rank set differs")

        rank_rows = []
        for rank, capture in enumerate(captures):
            environment = capture["environment"]
            observed_worker = {
                key: environment[key]
                for key in sorted(WORKER_KEYS)
                if key in environment
            }
            missing = sorted(WORKER_KEYS - set(environment))
            unexpected = sorted(set(environment) - set(base_environment) - WORKER_KEYS)
            changed_base = {
                key: {"expected": value, "observed": environment.get(key)}
                for key, value in sorted(base_environment.items())
                if key not in WORKER_KEYS and environment.get(key) != value
            }
            frozen_expected = expected_for_rank(rank, agent_store="False")
            corrected_expected = expected_for_rank(rank, agent_store="True")
            rank_rows.append(
                {
                    "rank": rank,
                    "argv": capture["argv"],
                    "observed_worker_environment": observed_worker,
                    "observed_full_environment": {
                        key: environment[key] for key in sorted(environment)
                    },
                    "expected_frozen_bridge_aaf375": frozen_expected,
                    "diff_frozen_bridge_aaf375": diff_environment(
                        observed_worker, frozen_expected, rank
                    ),
                    "expected_corrected_contract": corrected_expected,
                    "diff_corrected_contract": diff_environment(
                        observed_worker, corrected_expected, rank
                    ),
                    "missing_worker_keys": missing,
                    "unexpected_full_environment_keys": unexpected,
                    "changed_inherited_environment": changed_base,
                }
            )
        if any(
            row["diff_corrected_contract"]
            or row["missing_worker_keys"]
            or row["unexpected_full_environment_keys"]
            or row["changed_inherited_environment"]
            for row in rank_rows
        ):
            raise ProbeError("corrected rank environment contract differs")
        expected_old_diff = {
            "TORCHELASTIC_USE_AGENT_STORE": {
                "expected": "False",
                "observed": "True",
            }
        }
        if any(row["diff_frozen_bridge_aaf375"] != expected_old_diff for row in rank_rows):
            raise ProbeError("frozen bridge mismatch is not unique")
        model_modules = forbidden_model_modules(tuple(sys.modules))
        after_torch_gpu_fds = gpu_device_descriptors()
        if (
            torch.cuda.is_initialized()
            or torch.distributed.is_initialized()
            or before_torch_gpu_fds
            or after_torch_gpu_fds
            or model_modules
        ):
            raise ProbeError(
                "CUDA/HIP, process group, or model code was initialized"
            )
        if sys.pycache_prefix != str(expected_pycache) or any(
            expected_pycache.iterdir()
        ):
            raise ProbeError("probe bytecode cache changed")
        replay_pinned_file(probe_row)
        for row in producer_rows.values():
            replay_pinned_file(row)
        replay_pinned_file(true_row)
        receipt_payload = {
            "schema_version": SCHEMA,
            "status": "PASS",
            "job_binding": {
                "observed_slurm_environment": slurm_environment,
                "hostname": socket.gethostname(),
                "expected_job_id": args.expected_job_id,
                "expected_node": args.expected_node,
            },
            "torch_version": torch.__version__,
            "probe_source": {
                key: value for key, value in probe_row.items() if key != "fd"
            },
            "torchrun_argv": [
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=4",
                "--max_restarts=0",
                "--local-addr=localhost",
                "--no-python",
                str(true_path),
            ],
            "base_environment": {
                key: base_environment[key] for key in sorted(base_environment)
            },
            "producer_sources": {
                role: {
                    key: value for key, value in row.items() if key != "fd"
                }
                for role, row in sorted(producer_rows.items())
            },
            "diagnostic_trust_boundaries": {
                "probe_and_torch_sources": (
                    "named-path pre/post identity+sha under trusted same-UID namespace; "
                    "operational observation, not cryptographic executed-byte authority"
                ),
                "rank_executable": (
                    "root-owned /usr/bin/true mode0755 nlink1 captured then executed "
                    "as /proc/self/fd/N with exact singleton pass_fds"
                ),
            },
            "rank_executable": {
                **{
                    key: value for key, value in true_row.items() if key != "fd"
                },
                "execution_authority": (
                    "root-owned-mode0755-nlink1-retained-fd-executable"
                ),
            },
            "rank_captures": rank_rows,
            "held_rank_executable_calls": authority_exec_calls,
            "summary": {
                "rank_count": 4,
                "all_children_exit_zero": True,
                "frozen_bridge_unique_diff": "TORCHELASTIC_USE_AGENT_STORE",
                "frozen_bridge_expected": "False",
                "observed": "True",
                "corrected_contract_diff_count": 0,
                "cuda_initialized": False,
                "torch_process_group_initialized": False,
                "gpu_device_descriptors": [],
                "model_modules": [],
                "model_or_adapter_imported": False,
                "slurm_present_fields": [
                    "SLURM_GPUS_ON_NODE",
                    "SLURM_GPUS_PER_NODE",
                    "SLURM_JOB_ID",
                    "SLURM_JOB_NODELIST",
                    "SLURM_NNODES",
                    "SLURM_STEP_GPUS",
                    "SLURM_STEP_ID",
                    "SLURM_STEP_NODELIST",
                    "SLURM_STEP_NUM_NODES",
                ],
                "slurm_observed_absent_fields": [
                    "SLURM_JOB_GPUS",
                    "SLURM_JOB_NUM_NODES",
                ],
            },
        }
        receipt = write_receipt(receipt_path, receipt_payload)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    finally:
        cleanup.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
