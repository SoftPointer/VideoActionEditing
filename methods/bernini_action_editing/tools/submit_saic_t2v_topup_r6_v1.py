#!/usr/bin/env python3
"""Exactly-once submit the fresh SAIC pure-T2V r6 full60 job."""

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
REVISION = re.compile(r"[0-9a-f]{40}\Z")
EXPECTED_RUNTIME_SHA256 = (
    "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36"
)
ARCHIVE_RUNTIME = (
    "methods/bernini_action_editing/generate_saic_pure_t2v_event_bank_topup_v2.py"
)
ARCHIVE_GUARD = "methods/bernini_action_editing/saic_t2v_rendezvous_guard_v1.py"
ARCHIVE_LAUNCHER = (
    "methods/bernini_action_editing/scripts/"
    "auh_generate_saic_pure_t2v_event_bank_topup_all8_v3.sbatch"
)
ARCHIVE_SOURCE_MANIFEST = (
    "methods/bernini_action_editing/assets/saic_reversible_source_set_v1.json"
)
ARCHIVE_EVENT_SPEC = (
    "methods/bernini_action_editing/assets/saic_pure_t2v_event_bank_topup_v2.json"
)


def die(message: str) -> None:
    raise SystemExit(f"submit-saic-t2v-topup-r6: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        die(f"{label} is not a plain directory")
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


def archive_payloads(archive: Path, revision: str) -> dict[str, bytes]:
    if REVISION.fullmatch(revision) is None:
        die("source revision differs")
    required = {
        ARCHIVE_RUNTIME,
        ARCHIVE_GUARD,
        ARCHIVE_LAUNCHER,
        ARCHIVE_SOURCE_MANIFEST,
        ARCHIVE_EVENT_SPEC,
    }
    values: dict[str, bytes] = {}
    with tarfile.open(archive, "r:*") as handle:
        if handle.pax_headers.get("comment") != revision:
            die("source archive revision differs")
        for member in handle.getmembers():
            original = PurePosixPath(member.name)
            if original.is_absolute() or ".." in original.parts:
                die("source archive member escaped")
            normalized = original.as_posix()
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                die("source archive contains non-plain entry")
            if normalized in required:
                if not member.isfile() or normalized in values:
                    die("source archive required member differs")
                extracted = handle.extractfile(member)
                if extracted is None:
                    die("source archive member is unreadable")
                values[normalized] = extracted.read()
    if set(values) != required:
        die("source archive closure differs")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--event-spec", required=True)
    parser.add_argument("--event-spec-sha256", required=True)
    parser.add_argument("--rendezvous-guard-sha256", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--static-ffmpeg", required=True)
    parser.add_argument("--static-ffmpeg-sha256", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--slurm-log-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    launcher = exact_file(args.launcher, args.launcher_sha256, "launcher")
    archive = exact_file(
        args.source_archive, args.source_archive_sha256, "source archive"
    )
    source_manifest = exact_file(
        args.source_manifest, args.source_manifest_sha256, "source manifest"
    )
    event_spec = exact_file(args.event_spec, args.event_spec_sha256, "event spec")
    checkpoint_manifest = exact_file(
        args.checkpoint_manifest,
        args.checkpoint_manifest_sha256,
        "checkpoint manifest",
    )
    python_bin = exact_executable(args.python, args.python_sha256, "Python")
    ffmpeg_bin = exact_executable(
        args.static_ffmpeg, args.static_ffmpeg_sha256, "static ffmpeg"
    )
    bernini_root = exact_directory(args.bernini_root, "Bernini root")
    veomni_root = exact_directory(args.veomni_root, "VeOmni root")
    checkpoint = exact_directory(args.checkpoint, "checkpoint")

    payloads = archive_payloads(archive, args.source_revision)
    expected_archive_digests = {
        ARCHIVE_RUNTIME: EXPECTED_RUNTIME_SHA256,
        ARCHIVE_GUARD: args.rendezvous_guard_sha256,
        ARCHIVE_LAUNCHER: args.launcher_sha256,
        ARCHIVE_SOURCE_MANIFEST: args.source_manifest_sha256,
        ARCHIVE_EVENT_SPEC: args.event_spec_sha256,
    }
    if any(SHA256.fullmatch(value) is None for value in expected_archive_digests.values()):
        die("archive member SHA pin differs")
    for name, expected in expected_archive_digests.items():
        if digest_bytes(payloads[name]) != expected:
            die(f"source archive member SHA differs: {name}")

    output_root = Path(args.output_root)
    if not output_root.is_absolute() or output_root == Path("/"):
        die("output root identity differs")
    output_parent = exact_directory(str(output_root.parent), "output parent")
    if output_root != output_parent / output_root.name or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", output_root.name
    ):
        die("output root is not canonical")
    if output_root.exists() or output_root.is_symlink():
        die("output root is not fresh")
    receipt = Path(args.receipt)
    if receipt != Path(str(output_root) + ".submission.receipt.json"):
        die("submission receipt path differs")
    if receipt.exists() or receipt.is_symlink():
        die("submission receipt is not fresh")
    slurm_log_dir = exact_directory(args.slurm_log_dir, "Slurm log directory")
    output_parent_identity = directory_identity(output_parent)
    log_dir_identity = directory_identity(slurm_log_dir)

    exports = {
        "SAIC_T2V_V3_SOURCE_ARCHIVE": str(archive),
        "SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256": args.source_archive_sha256,
        "SAIC_T2V_V3_SOURCE_REVISION": args.source_revision,
        "SAIC_T2V_V3_SOURCE_MANIFEST": str(source_manifest),
        "SAIC_T2V_V3_SOURCE_MANIFEST_SHA256": args.source_manifest_sha256,
        "SAIC_T2V_V3_EVENT_SPEC": str(event_spec),
        "SAIC_T2V_V3_EVENT_SPEC_SHA256": args.event_spec_sha256,
        "SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256": args.rendezvous_guard_sha256,
        "BERNINI_OFFICIAL_ROOT": str(bernini_root),
        "BERNINI_VEOMNI_ROOT": str(veomni_root),
        "BERNINI_ACTION_CHECKPOINT": str(checkpoint),
        "BERNINI_CHECKPOINT_CONTENT_MANIFEST": str(checkpoint_manifest),
        "SAIC_T2V_V3_OUTPUT_ROOT": str(output_root),
        "SAIC_T2V_V3_PYTHON_BIN": str(python_bin),
        "SAIC_T2V_V3_STATIC_FFMPEG": str(ffmpeg_bin),
    }
    if any("," in name or "," in value or "\n" in value for name, value in exports.items()):
        die("sbatch export value differs")

    descriptor = os.open(
        receipt, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    reservation = os.fstat(descriptor)
    reservation_identity = (reservation.st_dev, reservation.st_ino)
    if (
        not stat.S_ISREG(reservation.st_mode)
        or stat.S_IMODE(reservation.st_mode) != 0o600
        or reservation.st_nlink != 1
    ):
        os.close(descriptor)
        die("submission reservation identity differs")
    provisional = {
        "schema_version": "saic-t2v-topup-r6-submission-v1",
        "status": "reserved_before_sbatch_not_submission_success",
        "submission_success": False,
        "job_success": None,
        "launcher_sha256": args.launcher_sha256,
        "source_archive_sha256": args.source_archive_sha256,
    }
    write_all(descriptor, canonical(provisional) + b"\n")
    os.fsync(descriptor)

    command = [
        "/usr/bin/sbatch",
        "--parsable",
        f"--output={slurm_log_dir}/saic-t2v-topup-r6-%j.out",
        f"--error={slurm_log_dir}/saic-t2v-topup-r6-%j.err",
        "--export=NONE," + ",".join(f"{name}={value}" for name, value in exports.items()),
        str(launcher),
    ]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
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
    if (
        directory_identity(output_parent) != output_parent_identity
        or directory_identity(slurm_log_dir) != log_dir_identity
    ):
        os.close(descriptor)
        die("post-sbatch directory identity differs; 0600 reservation retained")
    public_reservation = receipt.lstat()
    if (
        receipt.resolve(strict=True) != receipt
        or not stat.S_ISREG(public_reservation.st_mode)
        or stat.S_ISLNK(public_reservation.st_mode)
        or stat.S_IMODE(public_reservation.st_mode) != 0o600
        or public_reservation.st_nlink != 1
        or (public_reservation.st_dev, public_reservation.st_ino) != reservation_identity
    ):
        os.close(descriptor)
        die("post-sbatch submission reservation pathname differs")

    job_id = match.group(1)
    core = {
        "schema_version": "saic-t2v-topup-r6-submission-v1",
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
            "job_name": "saic-t2v-topup-r6",
            "partition": "faculty",
            "qos": "bgqos",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 32,
            "memory": "256G",
            "walltime": "24:00:00",
            "gpu_resource_requested": "gpu:mi210:8",
            "world_topology": "two_concurrent_world4_sp4",
            "candidate_count": 60,
            "hold": False,
            "dependency": None,
        },
        "submission_boundary": {
            "environment_replaced": True,
            "exact_job_export_names": list(exports),
            "export_all": False,
            "reservation_created_before_sbatch": True,
            "same_inode_retained": True,
            "reservation_device": reservation.st_dev,
            "reservation_inode": reservation.st_ino,
            "success_mode": "0444",
        },
        "inputs": {
            "launcher": str(launcher),
            "launcher_sha256": args.launcher_sha256,
            "source_archive": str(archive),
            "source_archive_sha256": args.source_archive_sha256,
            "source_revision": args.source_revision,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": args.source_manifest_sha256,
            "event_spec": str(event_spec),
            "event_spec_sha256": args.event_spec_sha256,
            "rendezvous_guard_sha256": args.rendezvous_guard_sha256,
            "checkpoint_manifest": str(checkpoint_manifest),
            "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
            "python": str(python_bin),
            "python_sha256": args.python_sha256,
            "static_ffmpeg": str(ffmpeg_bin),
            "static_ffmpeg_sha256": args.static_ffmpeg_sha256,
        },
        "outputs": {
            "output_root": str(output_root),
            "submission_receipt": str(receipt),
            "slurm_log_dir": str(slurm_log_dir),
            "fresh_before_submission": True,
        },
        "authority": {
            "diagnostic_event_bank": True,
            "training": False,
            "checkpoint": False,
            "scientific_success_claimed": False,
            "action_edit_success_claimed": False,
            "job_success_claimed": False,
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
        die("submission receipt reread differs before success transition")
    public_before_transition = receipt.lstat()
    if (
        receipt.resolve(strict=True) != receipt
        or not stat.S_ISREG(public_before_transition.st_mode)
        or stat.S_ISLNK(public_before_transition.st_mode)
        or stat.S_IMODE(public_before_transition.st_mode) != 0o600
        or public_before_transition.st_nlink != 1
        or public_before_transition.st_size != len(payload)
        or (public_before_transition.st_dev, public_before_transition.st_ino)
        != reservation_identity
    ):
        os.close(descriptor)
        die("submission receipt pathname differs before success transition")
    os.fchmod(descriptor, 0o444)
    os.fsync(descriptor)
    final = os.fstat(descriptor)
    try:
        public_final = receipt.lstat()
        final_matches = (
            receipt.resolve(strict=True) == receipt
            and stat.S_ISREG(final.st_mode)
            and not stat.S_ISLNK(public_final.st_mode)
            and stat.S_ISREG(public_final.st_mode)
            and stat.S_IMODE(final.st_mode) == 0o444
            and stat.S_IMODE(public_final.st_mode) == 0o444
            and final.st_nlink == 1
            and public_final.st_nlink == 1
            and final.st_size == len(payload)
            and public_final.st_size == len(payload)
            and (final.st_dev, final.st_ino) == reservation_identity
            and (public_final.st_dev, public_final.st_ino) == reservation_identity
        )
    except (OSError, RuntimeError):
        final_matches = False
    if not final_matches:
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        die("terminal submission receipt differs")
    os.close(descriptor)
    print(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
