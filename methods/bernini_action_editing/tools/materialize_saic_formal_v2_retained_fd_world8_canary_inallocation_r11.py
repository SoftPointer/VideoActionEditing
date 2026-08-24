#!/usr/bin/env python3
"""Materialize the fresh r11 in-allocation WORLD8 release on AUH.

The guard is accepted only from the immutable r10 release path and exact
1a38… hash; the locally drifted guard is never an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
STEM = "saic-formal-v2-retained-fd-world8-canary-inallocation-r11"
RELEASE = ROOT / "releases" / STEM
INPUTS = RELEASE / "inputs"
POSTFLIGHT_ROOT = RELEASE / "postflight"
OUTPUT_PARENT = ROOT / "canaries" / STEM
LOG_DIR = ROOT / "slurm" / STEM
EXTERNAL_R10_RELEASE = ROOT / (
    "releases/saic-formal-v2-retained-fd-world8-canary-96335bf5-fb3f1ac4-r10"
)
EXTERNAL_GUARD = EXTERNAL_R10_RELEASE / "inputs/saic_t2v_rendezvous_guard_v2.py"
EXTERNAL_GUARD_SHA256 = "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
EXTERNAL_R10_MANIFEST_SHA256 = "a358a18e0b5ea497f1c6d99cbcfdf1b4b51229c5f596309c02a0b39bb51055ba"
EXPECTED = {
    "launcher": (
        "launch_saic_formal_v2_retained_fd_world8_canary_inallocation_r11.py",
        "63aba833c48aff608a3bf9d19d4cc5c04d2681e9c54de66b5078908164e5bc1c",
    ),
    "wrapper": (
        "auh_canary_saic_formal_v2_retained_fd_world8_inallocation_r11.sh",
        "d53bc14fc89f34a5062132e381fe0c19d120d890d623e27b193e0d55838e2825",
    ),
    "payload": (
        "auh_canary_saic_formal_v2_retained_fd_world8_payload_inallocation_r11.sh",
        "8d7e4fc18f2e6ee7eb977972267dc03156c8ba8c35f5e14d1509c95bdb04a5fa",
    ),
    "postflight": (
        "postflight_saic_formal_v2_retained_fd_world8_canary_inallocation_r11.py",
        "0e60c2724743759fd439dd44ba79ae9518d13331bd9e762e7353716e824f842b",
    ),
}
R10_INPUTS = {
    "runtime": (
        "generate_saic_pure_t2v_event_bank_topup_v2.py",
        "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36",
    ),
    "source_archive": (
        "videoedit-saic-20c2193-methods.tar",
        "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b",
    ),
    "probe_validator": (
        "probe_admission_binding_v1.py",
        "3b5e78a0815fbfdee0404293ad800c640133deacc6b1bfafec12a02ca40ef85b",
    ),
}
PYTHON = Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12")
PYTHON_SHA256 = "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
SACCT = Path("/usr/bin/sacct")
SACCT_SHA256 = "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
PROBE_ADMISSION = ROOT / (
    "canaries/compute-bash-retained-fd-probe-8283e73d-r1/probe-admission.json"
)
PROBE_ADMISSION_SHA256 = "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8"
PROBE_ADMISSION_DIGEST = "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7"
AUTHORITY = {
    "scientific": False,
    "generation": False,
    "training": False,
    "publication": False,
    "formal_job_authorized": False,
}


def die(message: str) -> None:
    raise SystemExit(f"materialize-saic-fv2-fd-world8-inallocation-r11: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def retained_source(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor); leaf = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or not stat.S_ISREG(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
            or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino)
        ):
            die(f"{label} retained identity differs")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks); after = os.fstat(descriptor)
        leaf_after = path.lstat()
        key = lambda info: (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            stat.S_IMODE(info.st_mode), info.st_nlink, info.st_uid,
        )
        if (
            key(before) != key(after) or after.st_size != len(raw)
            or (after.st_dev, after.st_ino)
            != (leaf_after.st_dev, leaf_after.st_ino)
        ):
            die(f"{label} changed during retained read")
        return raw, after
    finally:
        os.close(descriptor)


def sha_file(path: Path) -> str:
    raw, _ = retained_source(path, str(path))
    return hashlib.sha256(raw).hexdigest()


def exact_source(path: Path, expected_sha: str, label: str) -> Path:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    raw, info = retained_source(path, label)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1 or SHA256.fullmatch(expected_sha) is None
        or hashlib.sha256(raw).hexdigest() != expected_sha
    ):
        die(f"{label} bytes differ")
    return path


def copy_create_only(source: Path, destination: Path) -> None:
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    destination_fd: int | None = None
    try:
        source_before = os.fstat(source_fd); source_leaf = source.lstat()
        if (
            not stat.S_ISREG(source_before.st_mode) or source_before.st_nlink != 1
            or not stat.S_ISREG(source_leaf.st_mode) or stat.S_ISLNK(source_leaf.st_mode)
            or (source_before.st_dev, source_before.st_ino)
            != (source_leaf.st_dev, source_leaf.st_ino)
        ):
            die("copy source retained identity differs")
        destination_fd = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        source_digest = hashlib.sha256(); source_size = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            source_digest.update(chunk); source_size += len(chunk)
            view = memoryview(chunk)
            while view:
                wrote = os.write(destination_fd, view)
                if wrote <= 0:
                    die("copy stalled")
                view = view[wrote:]
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o444)
        source_after = os.fstat(source_fd); source_leaf_after = source.lstat()
        if (
            (source_before.st_dev, source_before.st_ino, source_before.st_size,
             source_before.st_mtime_ns)
            != (source_after.st_dev, source_after.st_ino, source_after.st_size,
                source_after.st_mtime_ns)
            or source_size != source_after.st_size
            or (source_after.st_dev, source_after.st_ino)
            != (source_leaf_after.st_dev, source_leaf_after.st_ino)
            or source_digest.hexdigest() != sha_file(destination)
        ):
            die("copy retained bytes differ")
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for label in EXPECTED:
        value.add_argument("--" + label.replace("_", "-"), required=True)
    value.add_argument("--external-guard", required=True)
    value.add_argument("--external-guard-sha256", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source_paths = {
        label: exact_source(
            Path(getattr(args, label)), expected_sha, label.replace("_", " ")
        )
        for label, (_, expected_sha) in EXPECTED.items()
    }
    guard = Path(args.external_guard)
    if guard != EXTERNAL_GUARD or args.external_guard_sha256 != EXTERNAL_GUARD_SHA256:
        die("external immutable guard path/hash differs")
    exact_source(guard, EXTERNAL_GUARD_SHA256, "external immutable r10 guard")
    _, guard_info = retained_source(guard, "external immutable r10 guard metadata")
    if (
        guard_info.st_uid != os.getuid()
        or stat.S_IMODE(guard_info.st_mode) != 0o444
        or guard_info.st_nlink != 1 or guard_info.st_size != 70977
    ):
        die("external immutable r10 guard metadata differs")
    r10_manifest = EXTERNAL_R10_RELEASE / "release-manifest.json"
    exact_source(r10_manifest, EXTERNAL_R10_MANIFEST_SHA256, "external r10 manifest")
    for path, expected_sha, label in (
        (PYTHON, PYTHON_SHA256, "Python"), (SACCT, SACCT_SHA256, "sacct")
    ):
        exact_source(path, expected_sha, label)
        if not os.access(path, os.X_OK):
            die(f"{label} is not executable")
    if not os.access(Path("/usr/bin/python3"), os.X_OK):
        die("host OS compute bootstrap Python is not executable")
    inherited_sources: dict[str, Path] = {}
    for label, (basename, expected_sha) in R10_INPUTS.items():
        inherited_sources[label] = exact_source(
            EXTERNAL_R10_RELEASE / "inputs" / basename, expected_sha,
            f"external r10 {label}",
        )
    for fresh in (RELEASE, OUTPUT_PARENT, LOG_DIR):
        if fresh.exists() or fresh.is_symlink():
            die(f"fresh target exists: {fresh}")
    RELEASE.mkdir(mode=0o700)
    INPUTS.mkdir(mode=0o700)
    POSTFLIGHT_ROOT.mkdir(mode=0o700)
    OUTPUT_PARENT.mkdir(mode=0o700)
    LOG_DIR.mkdir(mode=0o700)
    installed: dict[str, Path] = {}
    for label, (basename, expected_sha) in EXPECTED.items():
        destination = (POSTFLIGHT_ROOT if label == "postflight" else INPUTS) / basename
        copy_create_only(source_paths[label], destination)
        if sha_file(destination) != expected_sha:
            die(f"installed {label} differs")
        installed[label] = destination
    for label, source in (("guard", guard), *inherited_sources.items()):
        destination = INPUTS / source.name
        copy_create_only(source, destination)
        installed[label] = destination
    inputs = {
        label: {"path": str(path), "sha256": sha_file(path)}
        for label, path in sorted(installed.items())
        if label != "postflight"
    }
    unsigned = {
        "schema_version": (
            "saic-formal-v2-retained-fd-world8-inallocation-release-manifest-v1"
        ),
        "status": "sealed_before_inallocation_step",
        "stem": STEM,
        "release_root": str(RELEASE),
        "output_parent": str(OUTPUT_PARENT),
        "log_directory": str(LOG_DIR),
        "parent_allocation_job_id": "134936",
        "expected_node": "auh7-1b-gpu-185",
        "inputs": inputs,
        "postflight": {
            "path": str(installed["postflight"]),
            "sha256": EXPECTED["postflight"][1],
        },
        "immutable_ancestor": {
            "release_root": str(EXTERNAL_R10_RELEASE),
            "release_manifest_file_sha256": EXTERNAL_R10_MANIFEST_SHA256,
            "guard_source_path": str(EXTERNAL_GUARD),
            "guard_sha256": EXTERNAL_GUARD_SHA256,
            "guard_copied_from_external_immutable_release": True,
            "local_guard_source_forbidden": True,
        },
        "executables": {
            "python": str(PYTHON), "python_sha256": PYTHON_SHA256,
            "compute_bootstrap_python": "/usr/bin/python3",
            "compute_bootstrap_python_trust": "host_os_absolute_path",
            "sacct": str(SACCT), "sacct_sha256": SACCT_SHA256,
        },
        "probe_admission": {
            "path": str(PROBE_ADMISSION), "file_sha256": PROBE_ADMISSION_SHA256,
            "receipt_digest": PROBE_ADMISSION_DIGEST,
        },
        "authority": AUTHORITY,
    }
    value = dict(unsigned)
    value["receipt_digest"] = hashlib.sha256(canonical(unsigned)).hexdigest()
    manifest = RELEASE / "release-manifest.json"
    descriptor = os.open(
        manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        raw = canonical(value) + b"\n"
        view = memoryview(raw)
        while view:
            wrote = os.write(descriptor, view)
            if wrote <= 0:
                die("manifest write stalled")
            view = view[wrote:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    os.chmod(INPUTS, 0o555)
    os.chmod(POSTFLIGHT_ROOT, 0o555)
    os.chmod(RELEASE, 0o555)
    for path in (INPUTS, POSTFLIGHT_ROOT, RELEASE, OUTPUT_PARENT, LOG_DIR):
        fsync_directory(path)
    print(json.dumps({
        "release": str(RELEASE), "output_parent": str(OUTPUT_PARENT),
        "log_directory": str(LOG_DIR), "release_manifest": str(manifest),
        "release_manifest_file_sha256": sha_file(manifest),
        "release_manifest_digest": value["receipt_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
