#!/bin/bash -p

# Submit the GRAFT A-lite canary through a closed, receipt-bearing boundary.
# This wrapper accepts no argv and exactly the seven GRAFT variables exported
# to the job.  It replaces the environment before starting its single Python
# supervisor; that supervisor fd-binds both the launcher and /usr/bin/sbatch.

case "$-" in
  *p*) ;;
  *) echo "[submit-graft-a-lite] ERROR: Bash privileged mode is required" >&2; exit 2 ;;
esac

set -Eeuo pipefail
umask 077

fail() { echo "[submit-graft-a-lite] ERROR: $*" >&2; exit 2; }
[[ "$#" -eq 0 ]] || fail "arbitrary arguments are forbidden"

readonly required_sbatch_path=/usr/bin/sbatch
readonly required_fd_root=/proc/self/fd
readonly required_fd_stat_identity=true
readonly required_execute_sbatch_from_fd=true
readonly export_names_csv=GRAFT_A_LITE_SOURCE_ARCHIVE,GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256,GRAFT_A_LITE_PYTHON_BIN,GRAFT_A_LITE_PYTHON_SHA256,GRAFT_A_LITE_LAUNCHER_SOURCE,GRAFT_A_LITE_LAUNCHER_SHA256,GRAFT_A_LITE_OUTPUT_STEM

readonly -a required_graft_names=(
  GRAFT_A_LITE_SOURCE_ARCHIVE
  GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256
  GRAFT_A_LITE_PYTHON_BIN
  GRAFT_A_LITE_PYTHON_SHA256
  GRAFT_A_LITE_LAUNCHER_SOURCE
  GRAFT_A_LITE_LAUNCHER_SHA256
  GRAFT_A_LITE_OUTPUT_STEM
)

observed_graft_count=0
for variable_name in ${!GRAFT_A_LITE_*}; do
  case "${variable_name}" in
    GRAFT_A_LITE_SOURCE_ARCHIVE|GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256|GRAFT_A_LITE_PYTHON_BIN|GRAFT_A_LITE_PYTHON_SHA256|GRAFT_A_LITE_LAUNCHER_SOURCE|GRAFT_A_LITE_LAUNCHER_SHA256|GRAFT_A_LITE_OUTPUT_STEM) ;;
    *) fail "unexpected GRAFT interface variable: ${variable_name}" ;;
  esac
  ((observed_graft_count+=1))
done
[[ "${observed_graft_count}" -eq 7 ]] || fail "exactly seven GRAFT interface variables are required"
for variable_name in "${required_graft_names[@]}"; do
  [[ -n "${!variable_name}" ]] || fail "${variable_name} must be nonempty"
done

# PATH, SBATCH_*, LD_*, BASH_ENV and imported functions are not forwarded.
# The absolute Python path is itself one of the seven hash-bound inputs.
exec /usr/bin/env -i \
  PATH=/usr/bin:/bin LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 \
  "${GRAFT_A_LITE_PYTHON_BIN}" -I -S -B - \
  "$0" "${required_sbatch_path}" "${required_fd_root}" \
  "${required_fd_stat_identity}" \
  "${required_execute_sbatch_from_fd}" \
  "${export_names_csv}" \
  "${GRAFT_A_LITE_SOURCE_ARCHIVE}" \
  "${GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256}" \
  "${GRAFT_A_LITE_PYTHON_BIN}" \
  "${GRAFT_A_LITE_PYTHON_SHA256}" \
  "${GRAFT_A_LITE_LAUNCHER_SOURCE}" \
  "${GRAFT_A_LITE_LAUNCHER_SHA256}" \
  "${GRAFT_A_LITE_OUTPUT_STEM}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

(wrapper_path, sbatch_path, fd_root, require_fd_stat_identity,
 execute_sbatch_from_fd, export_names_csv,
 source_archive, source_archive_sha, python_path, python_sha,
 launcher_path, launcher_sha, output_stem) = sys.argv[1:]

EXPECTED_EXPORT_NAMES = (
    "GRAFT_A_LITE_SOURCE_ARCHIVE",
    "GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256",
    "GRAFT_A_LITE_PYTHON_BIN",
    "GRAFT_A_LITE_PYTHON_SHA256",
    "GRAFT_A_LITE_LAUNCHER_SOURCE",
    "GRAFT_A_LITE_LAUNCHER_SHA256",
    "GRAFT_A_LITE_OUTPUT_STEM",
)
EXPECTED_SUPERVISOR_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
    "LANG": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
}
SCHEDULER_ARGUMENTS = (
    "--parsable",
    f"--export={','.join(EXPECTED_EXPORT_NAMES)}",
    "--partition=faculty",
    "--qos=bgqos",
    "--nodes=1",
    "--ntasks=1",
    "--cpus-per-task=8",
    "--mem=32G",
    "--gres=gpu:mi210:1",
    "--time=00:20:00",
    "--job-name=graft-a-lite-c4",
)
OUTPUT_SUFFIXES = (
    "",
    ".manifest.jsonl",
    ".receipt.json",
    ".execution.receipt.json",
    ".submission.receipt.json",
)
JOB_ID_RE = re.compile(r"([1-9][0-9]*)(?:;([A-Za-z0-9][A-Za-z0-9._-]*))?\Z")


def die(message: str) -> None:
    raise SystemExit(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def object_digest(value: object) -> str:
    return digest(canonical(value))


def full_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def inode_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        die(f"{label} is not lowercase SHA-256")


def open_retained_plain(
    path_text: str,
    label: str,
    *,
    mode: int | None = None,
    executable: bool = False,
) -> tuple[Path, int, bytes, tuple[int, int, int, int, int], dict[str, object]]:
    path = Path(path_text)
    if not path.is_absolute():
        die(f"{label} is not absolute")
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        die(f"cannot resolve {label}: {error}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        die(f"{label} is not a plain file")
    if resolved != path:
        die(f"{label} is not its exact realpath")
    if mode is not None and stat.S_IMODE(before.st_mode) != mode:
        die(f"{label} mode differs")
    if executable and not before.st_mode & 0o111:
        die(f"{label} is not executable")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if full_identity(opened) != full_identity(before):
            die(f"{label} changed while opening")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
        leaf = path.lstat()
        if full_identity(opened) != full_identity(after) or full_identity(after) != full_identity(leaf):
            die(f"{label} changed while reading")
        raw = b"".join(chunks)
        os.lseek(fd, 0, os.SEEK_SET)
        observation = {
            "path": str(path),
            "sha256": digest(raw),
            "size_bytes": len(raw),
            "mode": format(stat.S_IMODE(after.st_mode), "04o"),
            "device": after.st_dev,
            "inode": after.st_ino,
        }
        return path, fd, raw, full_identity(after), observation
    except BaseException:
        os.close(fd)
        raise


def validate_retained_plain(
    path: Path,
    fd: int,
    raw: bytes,
    identity: tuple[int, int, int, int, int],
    label: str,
) -> None:
    try:
        leaf = path.lstat()
    except OSError as error:
        die(f"cannot revalidate {label}: {error}")
    if full_identity(os.fstat(fd)) != identity or full_identity(leaf) != identity:
        die(f"{label} path or fd identity changed")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    os.lseek(fd, 0, os.SEEK_SET)
    if b"".join(chunks) != raw:
        die(f"{label} retained bytes changed")


def read_running_executable() -> tuple[bytes, dict[str, object]]:
    if sys.platform.startswith("linux") and Path("/proc/self/exe").exists():
        source = "/proc/self/exe"
        transport = "linux_proc_self_exe"
    else:
        source = str(Path(sys.executable).resolve(strict=True))
        transport = "sys_executable_path_fallback"
    fd = os.open(source, os.O_RDONLY)
    try:
        metadata = os.fstat(fd)
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if not stat.S_ISREG(after.st_mode) or full_identity(metadata) != full_identity(after):
        die("running Python executable identity differs")
    raw = b"".join(chunks)
    return raw, {
        "source": source,
        "transport": transport,
        "sha256": digest(raw),
        "size_bytes": len(raw),
        "device": after.st_dev,
        "inode": after.st_ino,
    }


def read_closed_plain(
    path_text: str,
    label: str,
    *,
    mode: int | None = None,
    executable: bool = False,
) -> tuple[bytes, dict[str, object]]:
    path, fd, raw, _, observation = open_retained_plain(
        path_text, label, mode=mode, executable=executable
    )
    del path
    try:
        return raw, observation
    finally:
        os.close(fd)


def parent_path_matches(parent: Path, identity: tuple[int, int]) -> bool:
    try:
        metadata = parent.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and inode_identity(metadata) == identity
    )


def open_output_parent(stem_text: str) -> tuple[Path, str, int, tuple[int, int]]:
    stem = Path(stem_text)
    if not stem.is_absolute() or not stem.name or stem.name in {".", ".."}:
        die("output stem is not an absolute leaf")
    parent = stem.parent
    try:
        before = parent.lstat()
        resolved = parent.resolve(strict=True)
    except OSError as error:
        die(f"output parent must already exist: {error}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode) or resolved != parent:
        die("output parent is not an exact plain directory")
    fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(fd)
    identity = inode_identity(opened)
    if identity != inode_identity(before) or not parent_path_matches(parent, identity):
        os.close(fd)
        die("output parent changed while pinning")
    for suffix in OUTPUT_SUFFIXES:
        name = f"{stem.name}{suffix}"
        try:
            os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            os.close(fd)
            die(f"cannot preflight output {name}: {error}")
        os.close(fd)
        die(f"create-only output already exists: {name}")
    return parent, stem.name, fd, identity


def create_receipt_at(parent_fd: int, name: str, raw: bytes) -> None:
    fd = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_fd,
    )
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                die("zero-byte submission receipt write")
            offset += written
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        reopened: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            reopened.append(block)
        if b"".join(reopened) != raw:
            die("submission receipt same-fd reopen differs")
    finally:
        os.close(fd)
    os.fsync(parent_fd)


if tuple(export_names_csv.split(",")) != EXPECTED_EXPORT_NAMES:
    die("export-name interface differs")
if require_fd_stat_identity not in {"true", "false"}:
    die("fd transport identity policy differs")
if execute_sbatch_from_fd not in {"true", "false"}:
    die("sbatch execution transport policy differs")
require_sha256(source_archive_sha, "source archive SHA-256")
require_sha256(python_sha, "Python SHA-256")
require_sha256(launcher_sha, "launcher SHA-256")
if dict(os.environ) != EXPECTED_SUPERVISOR_ENVIRONMENT:
    die("submission supervisor environment differs from fixed allowlist")
if (
    sys.flags.isolated != 1
    or sys.flags.no_site != 1
    or sys.flags.dont_write_bytecode != 1
    or sys.flags.ignore_environment != 1
    or sys.flags.no_user_site != 1
):
    die("Python isolated invocation flags differ")

parent, stem_name, parent_fd, parent_identity = open_output_parent(output_stem)
launcher_fd = -1
sbatch_fd = -1
try:
    wrapper_raw, wrapper_file = read_closed_plain(wrapper_path, "submission wrapper", mode=0o444)
    archive_raw, archive_file = read_closed_plain(source_archive, "source archive", mode=0o444)
    python_raw, python_file = read_closed_plain(
        python_path, "configured Python", executable=True
    )
    running_raw, running_file = read_running_executable()
    if digest(archive_raw) != source_archive_sha:
        die("source archive SHA-256 differs")
    if digest(python_raw) != python_sha or digest(running_raw) != python_sha:
        die("configured or running Python SHA-256 differs")
    if Path(sys.executable).resolve(strict=True) != Path(python_path):
        die("running Python differs from configured exact realpath")

    (launcher_file_path, launcher_fd, launcher_raw, launcher_identity,
     launcher_file) = open_retained_plain(launcher_path, "launcher", mode=0o444)
    if digest(launcher_raw) != launcher_sha:
        die("launcher SHA-256 differs")
    (sbatch_file_path, sbatch_fd, sbatch_raw, sbatch_identity,
     sbatch_file) = open_retained_plain(sbatch_path, "sbatch", executable=True)

    fd_root_path = Path(fd_root)
    if not fd_root_path.is_absolute() or not fd_root_path.is_dir():
        die("production retained-fd transport is unavailable")
    launcher_transport = f"{fd_root.rstrip('/')}/{launcher_fd}"
    sbatch_transport = f"{fd_root.rstrip('/')}/{sbatch_fd}"
    if require_fd_stat_identity == "true":
        if inode_identity(os.stat(launcher_transport)) != inode_identity(os.fstat(launcher_fd)):
            die("launcher fd transport differs")
        if inode_identity(os.stat(sbatch_transport)) != inode_identity(os.fstat(sbatch_fd)):
            die("sbatch fd transport differs")

    graft_values = {
        "GRAFT_A_LITE_SOURCE_ARCHIVE": source_archive,
        "GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256": source_archive_sha,
        "GRAFT_A_LITE_PYTHON_BIN": python_path,
        "GRAFT_A_LITE_PYTHON_SHA256": python_sha,
        "GRAFT_A_LITE_LAUNCHER_SOURCE": launcher_path,
        "GRAFT_A_LITE_LAUNCHER_SHA256": launcher_sha,
        "GRAFT_A_LITE_OUTPUT_STEM": output_stem,
    }
    child_environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        **graft_values,
    }
    sbatch_execution_path = (
        sbatch_transport if execute_sbatch_from_fd == "true" else sbatch_path
    )
    actual_argv = [sbatch_execution_path, *SCHEDULER_ARGUMENTS, launcher_transport]
    completed = subprocess.run(
        actual_argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
        pass_fds=(launcher_fd, sbatch_fd),
        env=child_environment,
    )
    if completed.returncode != 0:
        die(
            f"sbatch failed with exit {completed.returncode}; "
            f"stderr_sha256={digest(completed.stderr)}"
        )
    try:
        job_token = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        die("sbatch job ID output is not ASCII")
    if not job_token.endswith("\n") or job_token.count("\n") != 1:
        die("sbatch job ID output framing differs")
    matched = JOB_ID_RE.fullmatch(job_token[:-1])
    if matched is None:
        die("sbatch parsable job ID differs")
    job_id, scheduler_cluster = matched.groups()

    validate_retained_plain(
        launcher_file_path, launcher_fd, launcher_raw, launcher_identity, "launcher"
    )
    validate_retained_plain(
        sbatch_file_path, sbatch_fd, sbatch_raw, sbatch_identity, "sbatch"
    )
    for path_text, wanted, label, mode in (
        (wrapper_path, digest(wrapper_raw), "submission wrapper", 0o444),
        (source_archive, source_archive_sha, "source archive", 0o444),
        (python_path, python_sha, "configured Python", None),
    ):
        current, _ = read_closed_plain(
            path_text,
            label,
            mode=mode,
            executable=(label == "configured Python"),
        )
        if digest(current) != wanted:
            die(f"{label} changed after sbatch return")
    if dict(os.environ) != EXPECTED_SUPERVISOR_ENVIRONMENT:
        die("submission supervisor environment changed")
    if not parent_path_matches(parent, parent_identity):
        die("output parent path identity changed after sbatch return")

    exported_value_observations = [
        {
            "name": name,
            "value_sha256": digest(graft_values[name].encode("utf-8")),
            "value_size_bytes": len(graft_values[name].encode("utf-8")),
        }
        for name in EXPECTED_EXPORT_NAMES
    ]
    recorded_argv = [str(Path(sbatch_path)), *SCHEDULER_ARGUMENTS, "<retained_launcher_fd>"]
    core = {
        "schema_version": "bernini-graft-a-lite-source-submission-receipt-v1",
        "status": "submitted",
        "submission_success": True,
        "job_success": None,
        "job_terminal_state_observed": False,
        "effective_submission_request_verified": False,
        "submitted_job": {
            "job_id": job_id,
            "scheduler_cluster": scheduler_cluster,
            "parsable_stdout_sha256": digest(completed.stdout),
            "parsable_stdout_bytes": len(completed.stdout),
            "stderr_sha256": digest(completed.stderr),
            "stderr_bytes": len(completed.stderr),
        },
        "request": {
            "job_name": "graft-a-lite-c4",
            "partition": "faculty",
            "qos": "bgqos",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 8,
            "memory": "32G",
            "gpu_resource_requested": "gpu:mi210:1",
            "walltime": "00:20:00",
            "parsable_requested": True,
            "effective_scheduler_request_or_allocation_observed": False,
        },
        "submission_boundary": {
            "wrapper": wrapper_file,
            "python_configured_file": python_file,
            "python_running_executable": running_file,
            "python_runtime_closure_verified": False,
            "formal_runtime_authority": False,
            "initial_bash_and_dynamic_loader_closure_verified": False,
            "sbatch": {
                **sbatch_file,
                "configured_absolute_path": sbatch_path,
                "file_sha256_expected": None,
                "file_sha256_observed": digest(sbatch_raw),
                "file_hash_observation_only": True,
                "executed_from_retained_fd": execute_sbatch_from_fd == "true",
                "fd_transport_inode_identity_verified": (
                    require_fd_stat_identity == "true"
                ),
                "path_lookup_used": False,
            },
            "launcher": {
                **launcher_file,
                "sha256_expected": launcher_sha,
                "sha256_matched": True,
                "fd_retained_across_sbatch_return": True,
                "submitted_through_retained_fd": True,
                "fd_transport_inode_identity_verified": (
                    require_fd_stat_identity == "true"
                ),
            },
            "source_archive": {
                **archive_file,
                "sha256_expected": source_archive_sha,
                "sha256_matched": True,
            },
            "environment_replaced_before_python_supervisor": True,
            "inherited_path_sbatch_and_ld_variables_forwarded": False,
        },
        "export_contract": {
            "names": list(EXPECTED_EXPORT_NAMES),
            "exact_names_csv": ",".join(EXPECTED_EXPORT_NAMES),
            "contains_all": False,
            "exported_value_observations": exported_value_observations,
            "exported_value_observations_digest": object_digest(
                exported_value_observations
            ),
            "sbatch_child_environment_names": sorted(child_environment),
            "sbatch_child_environment_digest": object_digest(
                [
                    {
                        "name": name,
                        "value_sha256": digest(child_environment[name].encode("utf-8")),
                    }
                    for name in sorted(child_environment)
                ]
            ),
        },
        "argv_contract": {
            "recorded_argv": recorded_argv,
            "recorded_argv_digest": object_digest(recorded_argv),
            "actual_fd_argv_digest": object_digest(actual_argv),
            "launcher_argument_is_retained_fd_transport": True,
            "sbatch_executable_is_retained_fd_transport": (
                execute_sbatch_from_fd == "true"
            ),
            "arbitrary_user_arguments_accepted": False,
        },
        "outputs": {
            "logical_output_stem": str(Path(output_stem)),
            "submission_receipt_path": str(
                parent / f"{stem_name}.submission.receipt.json"
            ),
            "submission_receipt_create_only": True,
            "submission_receipt_mode": "0444",
            "output_parent_fd_retained": True,
            "output_parent_path_identity_revalidated": True,
        },
        "authority": {
            "action_authority": False,
            "identity_authority": False,
            "quality_authority": False,
            "training_authority": False,
            "production_authority": False,
            "data_governance_authority": False,
            "data_license_authority": False,
            "scientific_success_claimed": False,
        },
        "failure_semantics": {
            "submission_success_is_not_job_success": True,
            "job_success_requires_terminal_scheduler_and_execution_receipts": True,
            "receipt_alone_does_not_prove_wrapper_exit_zero": True,
            "consumer_must_require_wrapper_exit_zero": True,
            "successful_submission_may_exist_if_receipt_creation_fails": True,
            "automatic_job_cancellation_on_receipt_failure": False,
        },
    }
    receipt = {**core, "receipt_digest": object_digest(core)}
    receipt_raw = canonical(receipt) + b"\n"
    receipt_name = f"{stem_name}.submission.receipt.json"
    create_receipt_at(parent_fd, receipt_name, receipt_raw)
finally:
    for descriptor in (launcher_fd, sbatch_fd, parent_fd):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
PY
