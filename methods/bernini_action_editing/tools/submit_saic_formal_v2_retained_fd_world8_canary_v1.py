#!/usr/bin/env python3
"""Exactly-once submit the zero-science formal-v2 retained-FD WORLD8 canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import types
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_WRAPPER_SHA256 = (
    "fb3f1ac4e8f87f4833d45ff6be184ae863df0becbb8998dced22ed28ba240bd2"
)
EXPECTED_PAYLOAD_SHA256 = (
    "96335bf5f5a0896fdbf7e88dbe774bd5f4c03e0e44ecab96757834b9413d6750"
)
EXPECTED_GUARD_SHA256 = (
    "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
)
EXPECTED_RUNTIME_SHA256 = (
    "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
)
EXPECTED_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
EXPECTED_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
EXPECTED_POSTFLIGHT_SHA256 = (
    "99343d9a8effa4a5a53e8007dcfa615af4cbaec1cfa9fc57088e38ad6a9a5987"
)
EXPECTED_PROBE_VALIDATOR_SHA256 = (
    "3b5e78a0815fbfdee0404293ad800c640133deacc6b1bfafec12a02ca40ef85b"
)
PROBE_ADMISSION = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/canaries/"
    "compute-bash-retained-fd-probe-8283e73d-r1/probe-admission.json"
)
PROBE_ADMISSION_SHA256 = (
    "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8"
)
PROBE_ADMISSION_DIGEST = (
    "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7"
)
ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
RELEASE = ROOT / (
    "releases/saic-formal-v2-retained-fd-world8-canary-"
    "96335bf5-fb3f1ac4-r10"
)
INPUTS = RELEASE / "inputs"
POSTFLIGHT_ROOT = RELEASE / "postflight"
EXPECTED_WRAPPER = INPUTS / "auh_canary_saic_formal_v2_retained_fd_world8_v1.sbatch"
EXPECTED_PAYLOAD = (
    INPUTS / "auh_canary_saic_formal_v2_retained_fd_world8_payload_v1.sh"
)
EXPECTED_GUARD = INPUTS / "saic_t2v_rendezvous_guard_v2.py"
EXPECTED_RUNTIME = INPUTS / "generate_saic_pure_t2v_event_bank_topup_v2.py"
EXPECTED_SOURCE_ARCHIVE = INPUTS / "videoedit-saic-20c2193-methods.tar"
EXPECTED_PROBE_VALIDATOR = INPUTS / "probe_admission_binding_v1.py"
EXPECTED_POSTFLIGHT = (
    POSTFLIGHT_ROOT / "postflight_saic_formal_v2_retained_fd_world8_canary_v1.py"
)
EXPECTED_RELEASE_MANIFEST = RELEASE / "release-manifest.json"
EXPECTED_OUTPUT_PARENT = ROOT / (
    "canaries/saic-formal-v2-retained-fd-world8-canary-"
    "96335bf5-fb3f1ac4-r10"
)
EXPECTED_RECEIPT = EXPECTED_OUTPUT_PARENT / "submission-receipt.json"
EXPECTED_LOG_DIR = ROOT / (
    "slurm/saic-formal-v2-retained-fd-world8-canary-"
    "96335bf5-fb3f1ac4-r10"
)
EXPORT_NAMES = [
    "SAIC_FV2_FD_CANARY_PAYLOAD",
    "SAIC_FV2_FD_CANARY_PAYLOAD_SHA256",
    "SAIC_FV2_FD_CANARY_GUARD",
    "SAIC_FV2_FD_CANARY_GUARD_SHA256",
    "SAIC_FV2_FD_CANARY_RUNTIME",
    "SAIC_FV2_FD_CANARY_RUNTIME_SHA256",
    "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE",
    "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_SHA256",
    "SAIC_FV2_FD_CANARY_PYTHON",
    "SAIC_FV2_FD_CANARY_PYTHON_SHA256",
    "SAIC_FV2_FD_CANARY_OUTPUT_PARENT",
    "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_DEVICE",
    "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_INODE",
    "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT",
    "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_DEVICE",
    "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_INODE",
    "SAIC_FV2_FD_CANARY_WRAPPER",
    "SAIC_FV2_FD_CANARY_WRAPPER_SHA256",
    "SAIC_FV2_FD_CANARY_POSTFLIGHT",
    "SAIC_FV2_FD_CANARY_POSTFLIGHT_SHA256",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_SHA256",
    "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_DIGEST",
    "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR",
    "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR_SHA256",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_SHA256",
    "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_DIGEST",
]
RELEASE_MANIFEST_FIELDS = {
    "schema_version", "status", "stem", "release_root", "output_parent",
    "inputs", "postflight", "executables", "probe_admission", "authority",
    "receipt_digest",
}
AUTHORITY = {
    "scientific": False, "generation": False, "training": False,
    "publication": False, "formal_job_authorized": False,
}


def die(message: str) -> None:
    raise SystemExit(f"submit-saic-formal-v2-fd-world8-canary-v1: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def exact_file(value: str, expected: Path, expected_sha: str, label: str) -> Path:
    path = Path(value)
    if (
        path != expected or not path.is_absolute()
        or path.resolve(strict=True) != path
        or SHA256.fullmatch(expected_sha) is None
    ):
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444
        or sha_file(path) != expected_sha
    ):
        die(f"{label} bytes/mode differ")
    return path


def retained_exact_file_bytes(
    value: str, expected: Path, label: str,
) -> tuple[Path, bytes]:
    path = Path(value)
    if path != expected or not path.is_absolute():
        die(f"{label} identity differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        leaf_before = path.lstat()
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or not stat.S_ISREG(leaf_before.st_mode)
            or stat.S_ISLNK(leaf_before.st_mode) or leaf_before.st_nlink != 1
            or (leaf_before.st_dev, leaf_before.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            die(f"{label} retained identity differs")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        leaf_after = path.lstat()
        identity = lambda info: (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
        )
        if (
            identity(before) != identity(after)
            or (leaf_after.st_dev, leaf_after.st_ino)
            != (after.st_dev, after.st_ino)
            or not stat.S_ISREG(leaf_after.st_mode)
            or stat.S_ISLNK(leaf_after.st_mode) or leaf_after.st_nlink != 1
            or stat.S_IMODE(leaf_after.st_mode) != 0o444
        ):
            die(f"{label} changed during retained read")
        return path, b"".join(chunks)
    finally:
        os.close(descriptor)


def validate_release_manifest(
    raw: bytes, *, probe_binding: dict[str, Any],
) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        die(f"release manifest encoding differs: {error}")
    if (
        not isinstance(value, dict)
        or set(value) != RELEASE_MANIFEST_FIELDS
        or value.get("schema_version")
        != "saic-formal-v2-retained-fd-world8-release-manifest-v1"
    ):
        die("release manifest schema differs")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(claimed, str) or SHA256.fullmatch(claimed) is None
        or claimed != sha_bytes(canonical(unsigned))
        or raw != canonical(value) + b"\n"
    ):
        die("release manifest seal differs")
    expected_inputs = {
        "wrapper": str(EXPECTED_WRAPPER),
        "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "payload": str(EXPECTED_PAYLOAD),
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "guard": str(EXPECTED_GUARD),
        "guard_sha256": EXPECTED_GUARD_SHA256,
        "runtime": str(EXPECTED_RUNTIME),
        "runtime_sha256": EXPECTED_RUNTIME_SHA256,
        "source_archive": str(EXPECTED_SOURCE_ARCHIVE),
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "probe_validator": str(EXPECTED_PROBE_VALIDATOR),
        "probe_validator_sha256": EXPECTED_PROBE_VALIDATOR_SHA256,
    }
    if (
        value.get("status") != "sealed_before_canary_submission"
        or value.get("stem") != RELEASE.name
        or value.get("release_root") != str(RELEASE)
        or value.get("output_parent") != str(EXPECTED_OUTPUT_PARENT)
        or value.get("inputs") != expected_inputs
        or value.get("postflight") != {
            "path": str(EXPECTED_POSTFLIGHT),
            "sha256": EXPECTED_POSTFLIGHT_SHA256,
            "sha256_pinned_outside_postflight_source": True,
        }
        or value.get("executables") != {
            "python": str(EXPECTED_PYTHON),
            "python_sha256": EXPECTED_PYTHON_SHA256,
            "sacct": "/usr/bin/sacct",
            "sacct_sha256": (
                "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
            ),
        }
        or value.get("probe_admission") != probe_binding
        or value.get("authority") != AUTHORITY
    ):
        die("release manifest content differs")
    return value


def exact_directory(value: str, expected: Path, label: str, mode: int) -> Path:
    path = Path(value)
    if path != expected or not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode
    ):
        die(f"{label} mode/owner differs")
    return path


def exact_executable(value: str, expected_sha: str) -> Path:
    path = Path(value)
    if path != EXPECTED_PYTHON or path.resolve(strict=True) != path:
        die("Python identity differs")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1 or not os.access(path, os.X_OK)
        or sha_file(path) != expected_sha
    ):
        die("Python bytes differ")
    return path


def directory_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        wrote = os.write(descriptor, payload[offset:])
        if wrote <= 0:
            die("submission receipt write stalled")
        offset += wrote


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--wrapper-sha256", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--payload-sha256", required=True)
    parser.add_argument("--guard", required=True)
    parser.add_argument("--guard-sha256", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--postflight", required=True)
    parser.add_argument("--postflight-sha256", required=True)
    parser.add_argument("--probe-validator", required=True)
    parser.add_argument("--probe-validator-sha256", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--slurm-log-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_hashes = {
        "wrapper": (args.wrapper_sha256, EXPECTED_WRAPPER_SHA256),
        "payload": (args.payload_sha256, EXPECTED_PAYLOAD_SHA256),
        "guard": (args.guard_sha256, EXPECTED_GUARD_SHA256),
        "runtime": (args.runtime_sha256, EXPECTED_RUNTIME_SHA256),
        "source archive": (
            args.source_archive_sha256, EXPECTED_SOURCE_ARCHIVE_SHA256,
        ),
        "postflight": (args.postflight_sha256, EXPECTED_POSTFLIGHT_SHA256),
        "probe validator": (
            args.probe_validator_sha256, EXPECTED_PROBE_VALIDATOR_SHA256,
        ),
        "Python": (args.python_sha256, EXPECTED_PYTHON_SHA256),
    }
    for label, (actual, expected) in expected_hashes.items():
        if actual != expected or SHA256.fullmatch(expected) is None:
            die(f"{label} SHA pin differs")
    wrapper = exact_file(
        args.wrapper, EXPECTED_WRAPPER, EXPECTED_WRAPPER_SHA256, "wrapper"
    )
    payload = exact_file(
        args.payload, EXPECTED_PAYLOAD, EXPECTED_PAYLOAD_SHA256, "payload"
    )
    guard = exact_file(args.guard, EXPECTED_GUARD, EXPECTED_GUARD_SHA256, "guard")
    runtime = exact_file(
        args.runtime, EXPECTED_RUNTIME, EXPECTED_RUNTIME_SHA256, "runtime"
    )
    source_archive = exact_file(
        args.source_archive, EXPECTED_SOURCE_ARCHIVE,
        EXPECTED_SOURCE_ARCHIVE_SHA256, "source archive",
    )
    postflight = exact_file(
        args.postflight, EXPECTED_POSTFLIGHT,
        EXPECTED_POSTFLIGHT_SHA256, "postflight",
    )
    probe_validator, validator_raw = retained_exact_file_bytes(
        args.probe_validator, EXPECTED_PROBE_VALIDATOR, "probe validator",
    )
    if sha_bytes(validator_raw) != EXPECTED_PROBE_VALIDATOR_SHA256:
        die("probe validator retained bytes differ")
    validator = types.ModuleType("sealed_probe_admission_binding_v1")
    exec(compile(validator_raw, str(probe_validator), "exec"), validator.__dict__)
    probe_binding = validator.validate_probe_admission(
        PROBE_ADMISSION,
        expected_sha=PROBE_ADMISSION_SHA256,
        expected_digest=PROBE_ADMISSION_DIGEST,
    )
    release_manifest, release_manifest_raw = retained_exact_file_bytes(
        args.release_manifest, EXPECTED_RELEASE_MANIFEST, "release manifest"
    )
    release_manifest_value = validate_release_manifest(
        release_manifest_raw, probe_binding=probe_binding,
    )
    release_manifest_sha = sha_bytes(release_manifest_raw)
    release_manifest_digest = release_manifest_value["receipt_digest"]
    python_bin = exact_executable(args.python, EXPECTED_PYTHON_SHA256)
    release = exact_directory(str(RELEASE), RELEASE, "release root", 0o555)
    inputs = exact_directory(str(INPUTS), INPUTS, "release input directory", 0o555)
    postflight_root = exact_directory(
        str(POSTFLIGHT_ROOT), POSTFLIGHT_ROOT, "release postflight directory", 0o555
    )
    if set(release.iterdir()) != {INPUTS, POSTFLIGHT_ROOT, release_manifest}:
        die("release root closure differs")
    if set(inputs.iterdir()) != {
        wrapper, payload, guard, runtime, source_archive, probe_validator,
    }:
        die("release input closure differs")
    if set(postflight_root.iterdir()) != {postflight}:
        die("release postflight closure differs")
    output_parent = exact_directory(
        args.output_parent, EXPECTED_OUTPUT_PARENT, "fresh output parent", 0o700
    )
    log_dir = exact_directory(
        args.slurm_log_dir, EXPECTED_LOG_DIR, "Slurm log directory", 0o700
    )
    if any(output_parent.iterdir()):
        die("fresh output parent is not empty")
    receipt = Path(args.receipt)
    if receipt != EXPECTED_RECEIPT or receipt.exists() or receipt.is_symlink():
        die("submission receipt is not fresh")
    output_identity = directory_identity(output_parent)
    log_identity = directory_identity(log_dir)

    if not Path("/proc/self/fd").is_dir():
        die("retained-fd wrapper transport is unavailable")
    wrapper_descriptor = os.open(wrapper, os.O_RDONLY | os.O_NOFOLLOW)
    wrapper_info = os.fstat(wrapper_descriptor)
    wrapper_identity = (wrapper_info.st_dev, wrapper_info.st_ino)
    if (
        not stat.S_ISREG(wrapper_info.st_mode) or wrapper_info.st_nlink != 1
        or stat.S_IMODE(wrapper_info.st_mode) != 0o444
        or sha_descriptor(wrapper_descriptor) != EXPECTED_WRAPPER_SHA256
    ):
        os.close(wrapper_descriptor)
        die("retained wrapper bytes differ")

    descriptor = os.open(
        receipt, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    reserved = os.fstat(descriptor)
    reserved_identity = (reserved.st_dev, reserved.st_ino)
    if (
        not stat.S_ISREG(reserved.st_mode) or reserved.st_nlink != 1
        or stat.S_IMODE(reserved.st_mode) != 0o600
    ):
        os.close(wrapper_descriptor); os.close(descriptor)
        die("submission reservation identity differs")
    provisional = {
        "schema_version": "saic-formal-v2-retained-fd-world8-submission-v2",
        "status": "reserved_before_sbatch_not_submission_success",
        "submission_success": False, "job_success": None,
        "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "guard_sha256": EXPECTED_GUARD_SHA256,
        "runtime_sha256": EXPECTED_RUNTIME_SHA256,
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "postflight_sha256": EXPECTED_POSTFLIGHT_SHA256,
        "release_manifest_file_sha256": release_manifest_sha,
        "release_manifest_digest": release_manifest_digest,
        "probe_admission_sha256": PROBE_ADMISSION_SHA256,
        "probe_admission_digest": PROBE_ADMISSION_DIGEST,
        "probe_validator_sha256": EXPECTED_PROBE_VALIDATOR_SHA256,
    }
    write_all(descriptor, canonical(provisional) + b"\n")
    os.fsync(descriptor)
    fsync_directory(output_parent)
    public = receipt.lstat()
    retained = os.fstat(wrapper_descriptor)
    if (
        set(output_parent.iterdir()) != {receipt}
        or directory_identity(output_parent) != output_identity
        or directory_identity(log_dir) != log_identity
        or receipt.resolve(strict=True) != receipt
        or not stat.S_ISREG(public.st_mode) or stat.S_ISLNK(public.st_mode)
        or public.st_nlink != 1 or stat.S_IMODE(public.st_mode) != 0o600
        or (public.st_dev, public.st_ino) != reserved_identity
        or (retained.st_dev, retained.st_ino) != wrapper_identity
        or retained.st_nlink != 1 or sha_descriptor(wrapper_descriptor)
        != EXPECTED_WRAPPER_SHA256
    ):
        os.close(wrapper_descriptor); os.close(descriptor)
        die("pre-sbatch boundary differs")

    exports = {
        "SAIC_FV2_FD_CANARY_PAYLOAD": str(payload),
        "SAIC_FV2_FD_CANARY_PAYLOAD_SHA256": EXPECTED_PAYLOAD_SHA256,
        "SAIC_FV2_FD_CANARY_GUARD": str(guard),
        "SAIC_FV2_FD_CANARY_GUARD_SHA256": EXPECTED_GUARD_SHA256,
        "SAIC_FV2_FD_CANARY_RUNTIME": str(runtime),
        "SAIC_FV2_FD_CANARY_RUNTIME_SHA256": EXPECTED_RUNTIME_SHA256,
        "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE": str(source_archive),
        "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_SHA256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "SAIC_FV2_FD_CANARY_PYTHON": str(python_bin),
        "SAIC_FV2_FD_CANARY_PYTHON_SHA256": EXPECTED_PYTHON_SHA256,
        "SAIC_FV2_FD_CANARY_OUTPUT_PARENT": str(output_parent),
        "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_DEVICE": str(output_identity[0]),
        "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_INODE": str(output_identity[1]),
        "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT": str(receipt),
        "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_DEVICE": str(reserved_identity[0]),
        "SAIC_FV2_FD_CANARY_SUBMISSION_RECEIPT_INODE": str(reserved_identity[1]),
        "SAIC_FV2_FD_CANARY_WRAPPER": str(wrapper),
        "SAIC_FV2_FD_CANARY_WRAPPER_SHA256": EXPECTED_WRAPPER_SHA256,
        "SAIC_FV2_FD_CANARY_POSTFLIGHT": str(postflight),
        "SAIC_FV2_FD_CANARY_POSTFLIGHT_SHA256": EXPECTED_POSTFLIGHT_SHA256,
        "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST": str(release_manifest),
        "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_SHA256": release_manifest_sha,
        "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_DIGEST": release_manifest_digest,
        "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR": str(probe_validator),
        "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR_SHA256": (
            EXPECTED_PROBE_VALIDATOR_SHA256
        ),
        "SAIC_FV2_FD_CANARY_PROBE_ADMISSION": str(PROBE_ADMISSION),
        "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_SHA256": PROBE_ADMISSION_SHA256,
        "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_DIGEST": PROBE_ADMISSION_DIGEST,
    }
    if list(exports) != EXPORT_NAMES or any(
        "," in name or "," in value or "\n" in name or "\n" in value
        for name, value in exports.items()
    ):
        os.close(wrapper_descriptor); os.close(descriptor)
        die("exact export closure differs")
    command = [
        "/usr/bin/sbatch", "--parsable",
        f"--output={log_dir}/saic-fv2-fd-w8-cny1-%j.out",
        f"--error={log_dir}/saic-fv2-fd-w8-cny1-%j.err",
        "--export=NONE," + ",".join(f"{key}={value}" for key, value in exports.items()),
        f"/proc/self/fd/{wrapper_descriptor}",
    ]
    try:
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, timeout=120,
            pass_fds=(wrapper_descriptor,),
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    finally:
        try:
            os.close(wrapper_descriptor)
        except OSError:
            pass
    try:
        stdout = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        os.close(descriptor); die("sbatch stdout is not ASCII")
    match = re.fullmatch(r"([0-9]+)(?:;([^\n;]+))?\n?", stdout)
    if completed.returncode != 0 or match is None:
        os.close(descriptor)
        die(
            "sbatch failed; 0600 reservation retained: "
            f"exit={completed.returncode} stderr_sha256={sha_bytes(completed.stderr)}"
        )
    job_id = match.group(1)

    core = {
        "schema_version": "saic-formal-v2-retained-fd-world8-submission-v2",
        "status": "submitted", "submission_success": True, "job_success": None,
        "submitted_job": {
            "job_id": job_id, "cluster": match.group(2),
            "stdout_sha256": sha_bytes(completed.stdout),
            "stderr_sha256": sha_bytes(completed.stderr),
        },
        "request": {
            "job_name": "saic-fv2-fd-w8-cny1", "partition": "faculty",
            "qos": "bgqos", "nodes": 1, "ntasks": 1, "cpus_per_task": 16,
            "memory": "32G", "walltime": "00:15:00",
            "gpu_resource_requested": "gpu:mi210:8",
            "world_topology": "two_concurrent_world4",
            "hold": False, "dependency": None, "scientific_generation": False,
        },
        "submission_boundary": {
            "environment_replaced": True,
            "exact_job_export_names": EXPORT_NAMES,
            "export_all": False,
            "reservation_created_before_sbatch": True,
            "same_inode_retained": True,
            "launcher_submitted_from_retained_fd": True,
            "reservation_device": reserved.st_dev,
            "reservation_inode": reserved.st_ino,
            "output_parent_device": output_identity[0],
            "output_parent_inode": output_identity[1],
            "success_mode": "0444",
        },
        "inputs": {
            "wrapper": str(wrapper), "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
            "payload": str(payload), "payload_sha256": EXPECTED_PAYLOAD_SHA256,
            "guard": str(guard), "guard_sha256": EXPECTED_GUARD_SHA256,
            "runtime": str(runtime), "runtime_sha256": EXPECTED_RUNTIME_SHA256,
            "source_archive": str(source_archive),
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
            "python": str(python_bin), "python_sha256": EXPECTED_PYTHON_SHA256,
            "postflight": str(postflight),
            "postflight_sha256": EXPECTED_POSTFLIGHT_SHA256,
            "release_manifest": str(release_manifest),
            "release_manifest_file_sha256": release_manifest_sha,
            "release_manifest_digest": release_manifest_digest,
            "probe_validator": str(probe_validator),
            "probe_validator_sha256": EXPECTED_PROBE_VALIDATOR_SHA256,
            "probe_admission": str(PROBE_ADMISSION),
            "probe_admission_sha256": PROBE_ADMISSION_SHA256,
            "probe_admission_digest": PROBE_ADMISSION_DIGEST,
            "probe_admission_binding": probe_binding,
        },
        "outputs": {
            "output_parent": str(output_parent),
            "job_output_root": str(output_parent / f"job-{job_id}"),
            "submission_receipt": str(receipt), "fresh_before_submission": True,
        },
        "authority": {
            "scientific": False, "generation": False, "training": False,
            "publication": False, "formal_job_authorized": False,
        },
    }
    value = {**core, "receipt_digest": sha_bytes(canonical(core))}
    payload_bytes = canonical(value) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    write_all(descriptor, payload_bytes)
    os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.read(descriptor, len(payload_bytes) + 1) != payload_bytes:
        os.close(descriptor); die("submission receipt reread differs")
    public = receipt.lstat()
    if (
        directory_identity(output_parent) != output_identity
        or directory_identity(log_dir) != log_identity
        or receipt.resolve(strict=True) != receipt
        or set(output_parent.iterdir()) - {receipt}
        or not stat.S_ISREG(public.st_mode) or stat.S_ISLNK(public.st_mode)
        or public.st_nlink != 1 or stat.S_IMODE(public.st_mode) != 0o600
        or public.st_size != len(payload_bytes)
        or (public.st_dev, public.st_ino) != reserved_identity
        or (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
        != reserved_identity
    ):
        os.close(descriptor); die("terminal submission reservation differs")
    fsync_directory(output_parent)
    public = receipt.lstat()
    if (
        directory_identity(output_parent) != output_identity
        or receipt.resolve(strict=True) != receipt
        or set(output_parent.iterdir()) - {receipt}
        or (public.st_dev, public.st_ino) != reserved_identity
        or stat.S_IMODE(public.st_mode) != 0o600 or public.st_nlink != 1
    ):
        os.close(descriptor); die("pre-publication pathname differs")
    os.fchmod(descriptor, 0o444)
    try:
        os.close(descriptor)
    except OSError:
        pass
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
