#!/usr/bin/env python3
"""Exactly-once submit the fresh zero-science SAIC r6 rendezvous canary v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
EXPECTED_GUARD_SHA256 = (
    "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
)
EXPECTED_ARCHIVE_SHA256 = (
    "d7dad8b3af1bd06a6bb0bb5ddfa66607302bc98c4acf95d8cf065efba86ae7c6"
)
EXPECTED_LAUNCHER_SHA256 = (
    "6314a4d7f9fd99ab9713f6d956f6fd0f6571511aff43e2d0dd9b80f7cc439cea"
)
EXPECTED_RUNTIME_SHA256 = (
    "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36"
)
ARCHIVE_RUNTIME = (
    "methods/bernini_action_editing/generate_saic_pure_t2v_event_bank_topup_v2.py"
)


def die(message: str) -> None:
    raise SystemExit(f"submit-saic-r6-rendezvous-canary-v2: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def exact_file(value: str, expected_sha: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or SHA256.fullmatch(expected_sha) is None:
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o444
        or digest_file(path) != expected_sha
    ):
        die(f"{label} bytes/mode differ")
    return path


def exact_executable(value: str, expected_sha: str, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or SHA256.fullmatch(expected_sha) is None
        or path.resolve(strict=True) != path
    ):
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or not os.access(path, os.X_OK)
        or digest_file(path) != expected_sha
    ):
        die(f"{label} bytes differ")
    return path


def exact_directory(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        die(f"{label} is not an exact private directory")
    return path


def directory_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        wrote = os.write(descriptor, payload[offset:])
        if wrote <= 0:
            die("submission receipt write stalled")
        offset += wrote


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_archive(archive: Path) -> None:
    runtime_payload = None
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                die("source archive member escaped")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                die("source archive contains non-plain entry")
            if member_path.as_posix() == ARCHIVE_RUNTIME:
                if runtime_payload is not None or not member.isfile():
                    die("source archive runtime member differs")
                extracted = handle.extractfile(member)
                if extracted is None:
                    die("source archive runtime is unreadable")
                runtime_payload = extracted.read()
    if runtime_payload is None or digest_bytes(runtime_payload) != EXPECTED_RUNTIME_SHA256:
        die("source archive scientific runtime bytes differ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--guard", required=True)
    parser.add_argument("--guard-sha256", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.guard_sha256 != EXPECTED_GUARD_SHA256:
        die("guard v2 SHA pin differs")
    if args.archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        die("sealed source archive SHA pin differs")
    if args.launcher_sha256 != EXPECTED_LAUNCHER_SHA256:
        die("canary v2 launcher SHA pin differs")
    launcher = exact_file(args.launcher, args.launcher_sha256, "canary v2 launcher")
    archive = exact_file(args.archive, args.archive_sha256, "source archive")
    guard = exact_file(args.guard, args.guard_sha256, "rendezvous guard v2")
    python_bin = exact_executable(args.python, args.python_sha256, "Python")
    validate_archive(archive)

    if not Path("/proc/self/fd").is_dir():
        die("Linux retained-fd launcher transport is unavailable")
    launcher_descriptor = os.open(launcher, os.O_RDONLY | os.O_NOFOLLOW)
    launcher_descriptor_info = os.fstat(launcher_descriptor)
    if (
        not stat.S_ISREG(launcher_descriptor_info.st_mode)
        or launcher_descriptor_info.st_nlink != 1
        or stat.S_IMODE(launcher_descriptor_info.st_mode) != 0o444
        or digest_descriptor(launcher_descriptor) != args.launcher_sha256
    ):
        os.close(launcher_descriptor)
        die("retained canary v2 launcher bytes differ")

    output_parent = exact_directory(args.output_parent, "fresh canary output parent")
    if SAFE_NAME.fullmatch(output_parent.name) is None or any(output_parent.iterdir()):
        die("fresh canary output parent differs")
    receipt = Path(args.receipt)
    if receipt != output_parent / "submission-receipt.json":
        die("submission receipt path differs")
    if receipt.exists() or receipt.is_symlink():
        die("submission receipt is not fresh")
    output_parent_identity = directory_identity(output_parent)

    exports = {
        "SAIC_R6_CANARY_V2_SOURCE_ARCHIVE": str(archive),
        "SAIC_R6_CANARY_V2_SOURCE_ARCHIVE_SHA256": args.archive_sha256,
        "SAIC_R6_CANARY_V2_GUARD": str(guard),
        "SAIC_R6_CANARY_V2_GUARD_SHA256": args.guard_sha256,
        "SAIC_R6_CANARY_V2_PYTHON": str(python_bin),
        "SAIC_R6_CANARY_V2_PYTHON_SHA256": args.python_sha256,
        "SAIC_R6_CANARY_V2_OUTPUT_PARENT": str(output_parent),
        "SAIC_R6_CANARY_V2_LAUNCHER": str(launcher),
        "SAIC_R6_CANARY_V2_LAUNCHER_SHA256": args.launcher_sha256,
        "SAIC_R6_CANARY_V2_SUBMISSION_RECEIPT": str(receipt),
    }
    if any(
        "," in name or "," in value or "\n" in name or "\n" in value
        for name, value in exports.items()
    ):
        die("sbatch export value differs")

    descriptor = os.open(
        receipt, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    reservation = os.fstat(descriptor)
    reservation_identity = (reservation.st_dev, reservation.st_ino)
    if (
        not stat.S_ISREG(reservation.st_mode)
        or reservation.st_nlink != 1
        or stat.S_IMODE(reservation.st_mode) != 0o600
    ):
        os.close(descriptor)
        die("submission reservation identity differs")
    provisional = {
        "schema_version": "saic-r6-rendezvous-canary-submission-v2",
        "status": "reserved_before_sbatch_not_submission_success",
        "submission_success": False,
        "job_success": None,
        "launcher_sha256": args.launcher_sha256,
        "guard_sha256": args.guard_sha256,
        "source_archive_sha256": args.archive_sha256,
    }
    write_all(descriptor, canonical(provisional) + b"\n")
    os.fsync(descriptor)
    fsync_directory(output_parent)
    public_before_sbatch = receipt.lstat()
    if (
        list(output_parent.iterdir()) != [receipt]
        or receipt.resolve(strict=True) != receipt
        or not stat.S_ISREG(public_before_sbatch.st_mode)
        or stat.S_ISLNK(public_before_sbatch.st_mode)
        or public_before_sbatch.st_nlink != 1
        or stat.S_IMODE(public_before_sbatch.st_mode) != 0o600
        or (public_before_sbatch.st_dev, public_before_sbatch.st_ino)
        != reservation_identity
    ):
        os.close(descriptor)
        die("pre-sbatch submission reservation closure differs")

    command = [
        "/usr/bin/sbatch",
        "--parsable",
        f"--output={output_parent}/slurm-%j.out",
        f"--error={output_parent}/slurm-%j.err",
        "--export=NONE," + ",".join(
            f"{name}={value}" for name, value in exports.items()
        ),
        f"/proc/self/fd/{launcher_descriptor}",
    ]
    retained_before_sbatch = os.fstat(launcher_descriptor)
    if (
        not stat.S_ISREG(retained_before_sbatch.st_mode)
        or retained_before_sbatch.st_nlink != 1
        or stat.S_IMODE(retained_before_sbatch.st_mode) != 0o444
        or digest_descriptor(launcher_descriptor) != args.launcher_sha256
    ):
        os.close(launcher_descriptor)
        os.close(descriptor)
        die("retained launcher changed before sbatch")
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            pass_fds=(launcher_descriptor,),
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    finally:
        try:
            os.close(launcher_descriptor)
        except OSError:
            pass
    try:
        stdout_text = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        os.close(descriptor)
        die("sbatch stdout is not ASCII")
    match = re.fullmatch(r"([0-9]+)(?:;([^\n;]+))?\n?", stdout_text)
    if completed.returncode != 0 or match is None:
        os.close(descriptor)
        die(
            "sbatch failed; 0600 reservation retained: "
            f"exit={completed.returncode} stderr_sha256={digest_bytes(completed.stderr)}"
        )
    job_id = match.group(1)

    submitted_provisional = {
        **provisional,
        "status": "sbatch_returned_job_id_receipt_not_terminal",
        "submitted_job_id": job_id,
        "sbatch_stdout_sha256": digest_bytes(completed.stdout),
        "sbatch_stderr_sha256": digest_bytes(completed.stderr),
    }
    staged_payload = canonical(submitted_provisional) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    write_all(descriptor, staged_payload)
    os.fsync(descriptor)

    public = receipt.lstat()
    if (
        directory_identity(output_parent) != output_parent_identity
        or receipt.resolve(strict=True) != receipt
        or not stat.S_ISREG(public.st_mode)
        or stat.S_ISLNK(public.st_mode)
        or public.st_nlink != 1
        or stat.S_IMODE(public.st_mode) != 0o600
        or (public.st_dev, public.st_ino) != reservation_identity
    ):
        os.close(descriptor)
        die("post-sbatch submission reservation pathname differs")

    core = {
        "schema_version": "saic-r6-rendezvous-canary-submission-v2",
        "status": "submitted",
        "submission_success": True,
        "job_success": None,
        "submitted_job": {
            "job_id": job_id,
            "cluster": match.group(2),
            "stdout_sha256": digest_bytes(completed.stdout),
            "stderr_sha256": digest_bytes(completed.stderr),
        },
        "request": {
            "job_name": "saic-r6-rdzv-cny2",
            "partition": "faculty",
            "qos": "bgqos",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 32,
            "memory": "64G",
            "walltime": "00:30:00",
            "gpu_resource_requested": "gpu:mi210:8",
            "world_topology": "two_concurrent_world4",
            "candidate_count": 60,
            "hold": False,
            "dependency": None,
            "scientific_generation": False,
        },
        "submission_boundary": {
            "environment_replaced": True,
            "exact_job_export_names": list(exports),
            "export_all": False,
            "reservation_created_before_sbatch": True,
            "same_inode_retained": True,
            "launcher_submitted_from_retained_fd": True,
            "reservation_device": reservation.st_dev,
            "reservation_inode": reservation.st_ino,
            "success_mode": "0444",
        },
        "inputs": {
            "launcher": str(launcher),
            "launcher_sha256": args.launcher_sha256,
            "guard": str(guard),
            "guard_sha256": args.guard_sha256,
            "source_archive": str(archive),
            "source_archive_sha256": args.archive_sha256,
            "runtime_sha256": EXPECTED_RUNTIME_SHA256,
            "python": str(python_bin),
            "python_sha256": args.python_sha256,
        },
        "outputs": {
            "output_parent": str(output_parent),
            "job_output_root": f"{output_parent}/job-{job_id}",
            "submission_receipt": str(receipt),
            "fresh_before_submission": True,
        },
        "authority": {
            "scientific": False,
            "generation": False,
            "training": False,
            "publication": False,
            "formal_job_authorized": False,
        },
    }
    value = {**core, "receipt_digest": digest_bytes(canonical(core))}
    payload = canonical(value) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    write_all(descriptor, payload)
    os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.read(descriptor, len(payload) + 1) != payload:
        os.close(descriptor)
        die("submission receipt reread differs before terminal publication")
    public = receipt.lstat()
    if (
        receipt.resolve(strict=True) != receipt
        or not stat.S_ISREG(public.st_mode)
        or stat.S_ISLNK(public.st_mode)
        or public.st_nlink != 1
        or stat.S_IMODE(public.st_mode) != 0o600
        or public.st_size != len(payload)
        or (public.st_dev, public.st_ino) != reservation_identity
    ):
        os.close(descriptor)
        die("submission receipt pathname differs before terminal publication")
    fsync_directory(output_parent)
    os.close(descriptor)

    publisher = os.open(receipt, os.O_RDWR | os.O_NOFOLLOW)
    try:
        observed = os.fstat(publisher)
        public = receipt.lstat()
        if (
            directory_identity(output_parent) != output_parent_identity
            or output_parent.resolve(strict=True) != output_parent
            or receipt.resolve(strict=True) != receipt
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o600
            or observed.st_size != len(payload)
            or (observed.st_dev, observed.st_ino) != reservation_identity
            or not stat.S_ISREG(public.st_mode)
            or stat.S_ISLNK(public.st_mode)
            or public.st_nlink != 1
            or (public.st_dev, public.st_ino) != reservation_identity
        ):
            die("public submission reservation differs")
        os.lseek(publisher, 0, os.SEEK_SET)
        if os.read(publisher, len(payload) + 1) != payload:
            die("public submission payload differs")
    except BaseException:
        os.close(publisher)
        raise
    os.fchmod(publisher, 0o444)
    try:
        os.close(publisher)
    except OSError:
        pass
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
