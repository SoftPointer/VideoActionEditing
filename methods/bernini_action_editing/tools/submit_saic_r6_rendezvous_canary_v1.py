#!/usr/bin/env python3
"""Exactly-once submit the non-scientific SAIC r6 rendezvous canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any


SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def die(message: str) -> None:
    raise SystemExit(f"submit-saic-r6-rendezvous-canary: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--guard", required=True)
    parser.add_argument("--guard-sha256", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--receipt", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    launcher = exact_file(args.launcher, args.launcher_sha256, "launcher")
    archive = exact_file(args.archive, args.archive_sha256, "source archive")
    guard = exact_file(args.guard, args.guard_sha256, "rendezvous guard")
    python_bin = Path(args.python)
    if (
        not python_bin.is_absolute()
        or python_bin.resolve(strict=True) != python_bin
        or not python_bin.is_file()
        or python_bin.is_symlink()
        or not os.access(python_bin, os.X_OK)
    ):
        die("Python executable differs")
    output_parent = Path(args.output_parent)
    if (
        not output_parent.is_absolute()
        or output_parent.resolve(strict=True) != output_parent
        or not output_parent.is_dir()
        or output_parent.is_symlink()
        or any(output_parent.glob("job-*"))
    ):
        die("fresh canary output parent differs")
    receipt = Path(args.receipt)
    if not receipt.is_absolute() or receipt.parent != output_parent:
        die("submission receipt path differs")
    descriptor = os.open(
        receipt,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    provisional = {
        "schema_version": "saic-r6-rendezvous-canary-submission-v1",
        "status": "reserved_before_sbatch_not_submission_success",
        "submission_success": False,
        "job_success": None,
        "launcher_sha256": args.launcher_sha256,
        "guard_sha256": args.guard_sha256,
        "source_archive_sha256": args.archive_sha256,
    }
    provisional_payload = canonical(provisional) + b"\n"
    os.write(descriptor, provisional_payload)
    os.fsync(descriptor)
    exports = {
        "SAIC_R6_CANARY_SOURCE_ARCHIVE": str(archive),
        "SAIC_R6_CANARY_SOURCE_ARCHIVE_SHA256": args.archive_sha256,
        "SAIC_R6_CANARY_GUARD": str(guard),
        "SAIC_R6_CANARY_GUARD_SHA256": args.guard_sha256,
        "SAIC_R6_CANARY_PYTHON": str(python_bin),
        "SAIC_R6_CANARY_OUTPUT_PARENT": str(output_parent),
    }
    if any("," in name or "," in value for name, value in exports.items()):
        die("sbatch export contains comma")
    command = [
        "sbatch",
        "--parsable",
        f"--output={output_parent}/slurm-%j.out",
        f"--error={output_parent}/slurm-%j.err",
        "--export=NONE," + ",".join(f"{name}={value}" for name, value in exports.items()),
        str(launcher),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    match = re.fullmatch(r"([0-9]+)(?:;([^\n;]+))?\n?", completed.stdout)
    if completed.returncode != 0 or match is None:
        os.close(descriptor)
        die(f"sbatch failed; 0600 reservation retained: {completed.stderr.strip()}")
    job_id = match.group(1)
    core = {
        "schema_version": "saic-r6-rendezvous-canary-submission-v1",
        "status": "submitted",
        "submission_success": True,
        "job_success": None,
        "submitted_job": {
            "job_id": job_id,
            "cluster": match.group(2),
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        },
        "request": {
            "job_name": "saic-r6-rdzv-cny",
            "gpu_resource_requested": "gpu:mi210:8",
            "world_topology": "two_concurrent_world4",
            "candidate_count": 60,
            "hold": False,
            "dependency": None,
            "scientific_generation": False,
        },
        "inputs": {
            "launcher": str(launcher),
            "launcher_sha256": args.launcher_sha256,
            "guard": str(guard),
            "guard_sha256": args.guard_sha256,
            "source_archive": str(archive),
            "source_archive_sha256": args.archive_sha256,
        },
        "outputs": {
            "output_parent": str(output_parent),
            "job_output_root": f"{output_parent}/job-{job_id}",
            "submission_receipt": str(receipt),
            "reservation_created_before_sbatch": True,
            "same_inode_retained": True,
        },
        "authority": {
            "scientific": False,
            "generation": False,
            "training": False,
            "publication": False,
        },
    }
    value = {**core, "receipt_digest": hashlib.sha256(canonical(core)).hexdigest()}
    payload = canonical(value) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    offset = 0
    while offset < len(payload):
        wrote = os.write(descriptor, payload[offset:])
        if wrote <= 0:
            os.close(descriptor)
            die("submission receipt write stalled")
        offset += wrote
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o444)
    os.fsync(descriptor)
    final = os.fstat(descriptor)
    os.close(descriptor)
    if stat.S_IMODE(final.st_mode) != 0o444 or final.st_nlink != 1:
        die("terminal submission receipt differs")
    print(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
