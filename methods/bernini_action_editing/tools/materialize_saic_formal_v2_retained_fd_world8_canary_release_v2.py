#!/usr/bin/env python3
"""Materialize the fresh r10 retained-FD WORLD8 canary release on AUH.

The r4 canary proved the retained descriptor handoff but failed while importing
the formal runtime because only the top-level runtime file was packaged.  R5
adds the already sealed formal source archive and makes the zero-science canary
import the runtime from that complete dependency closure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
STEM = "saic-formal-v2-retained-fd-world8-canary-96335bf5-fb3f1ac4-r10"
RELEASE = ROOT / "releases" / STEM
INPUTS = RELEASE / "inputs"
POSTFLIGHT_ROOT = RELEASE / "postflight"
OUTPUT_PARENT = ROOT / "canaries" / STEM
LOG_DIR = ROOT / "slurm" / STEM
PROBE_ADMISSION = (
    ROOT
    / "canaries/compute-bash-retained-fd-probe-8283e73d-r1/"
    "probe-admission.json"
)
PROBE_ADMISSION_SHA256 = (
    "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8"
)
PROBE_ADMISSION_DIGEST = (
    "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7"
)
PYTHON = Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12")
PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
SACCT = Path("/usr/bin/sacct")
SACCT_SHA256 = (
    "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
)
EXPECTED = {
    "wrapper": (
        "auh_canary_saic_formal_v2_retained_fd_world8_v1.sbatch",
        "fb3f1ac4e8f87f4833d45ff6be184ae863df0becbb8998dced22ed28ba240bd2",
    ),
    "payload": (
        "auh_canary_saic_formal_v2_retained_fd_world8_payload_v1.sh",
        "96335bf5f5a0896fdbf7e88dbe774bd5f4c03e0e44ecab96757834b9413d6750",
    ),
    "guard": (
        "saic_t2v_rendezvous_guard_v2.py",
        "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965",
    ),
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
    "postflight": (
        "postflight_saic_formal_v2_retained_fd_world8_canary_v1.py",
        "99343d9a8effa4a5a53e8007dcfa615af4cbaec1cfa9fc57088e38ad6a9a5987",
    ),
}
AUTHORITY = {
    "scientific": False,
    "generation": False,
    "training": False,
    "publication": False,
    "formal_job_authorized": False,
}


def die(message: str) -> None:
    raise SystemExit(f"materialize-saic-fv2-fd-world8-r10: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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


def exact_source(path: Path, expected_sha: str, label: str) -> Path:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} source identity differs")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or SHA256.fullmatch(expected_sha) is None
        or sha_file(path) != expected_sha
    ):
        die(f"{label} source bytes differ")
    return path


def exact_private_parent(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        die(f"{label} is not private")
    return path


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        wrote = os.write(descriptor, payload[offset:])
        if wrote <= 0:
            die("write stalled")
        offset += wrote


def copy_create_only(source: Path, destination: Path) -> None:
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    destination_fd = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            write_all(destination_fd, chunk)
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o444)
    finally:
        os.close(source_fd)
        os.close(destination_fd)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_validator(path: Path):
    spec = importlib.util.spec_from_file_location("sealed_probe_validator", path)
    if spec is None or spec.loader is None:
        die("probe validator import differs")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in EXPECTED:
        parser.add_argument("--" + label.replace("_", "-"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_paths = {
        label: exact_source(
            Path(getattr(args, label)), expected_sha, label.replace("_", " ")
        )
        for label, (_, expected_sha) in EXPECTED.items()
    }
    for executable, expected_sha, label in (
        (PYTHON, PYTHON_SHA256, "Python"),
        (SACCT, SACCT_SHA256, "sacct"),
    ):
        exact_source(executable, expected_sha, label)
        if not os.access(executable, os.X_OK):
            die(f"{label} is not executable")

    for parent, label in (
        (ROOT / "releases", "release parent"),
        (ROOT / "canaries", "canary parent"),
        (ROOT / "slurm", "Slurm parent"),
    ):
        exact_private_parent(parent, label)
    for fresh in (RELEASE, OUTPUT_PARENT, LOG_DIR):
        if fresh.exists() or fresh.is_symlink():
            die(f"fresh target already exists: {fresh}")

    RELEASE.mkdir(mode=0o700)
    INPUTS.mkdir(mode=0o700)
    POSTFLIGHT_ROOT.mkdir(mode=0o700)
    OUTPUT_PARENT.mkdir(mode=0o700)
    LOG_DIR.mkdir(mode=0o700)
    try:
        installed: dict[str, Path] = {}
        for label, (basename, expected_sha) in EXPECTED.items():
            parent = POSTFLIGHT_ROOT if label == "postflight" else INPUTS
            destination = parent / basename
            copy_create_only(source_paths[label], destination)
            if sha_file(destination) != expected_sha:
                die(f"installed {label} bytes differ")
            installed[label] = destination

        validator = load_validator(installed["probe_validator"])
        probe_binding = validator.validate_probe_admission(
            PROBE_ADMISSION,
            expected_sha=PROBE_ADMISSION_SHA256,
            expected_digest=PROBE_ADMISSION_DIGEST,
        )
        inputs = {
            label: value
            for label, value in (
                ("wrapper", str(installed["wrapper"])),
                ("wrapper_sha256", EXPECTED["wrapper"][1]),
                ("payload", str(installed["payload"])),
                ("payload_sha256", EXPECTED["payload"][1]),
                ("guard", str(installed["guard"])),
                ("guard_sha256", EXPECTED["guard"][1]),
                ("runtime", str(installed["runtime"])),
                ("runtime_sha256", EXPECTED["runtime"][1]),
                ("source_archive", str(installed["source_archive"])),
                ("source_archive_sha256", EXPECTED["source_archive"][1]),
                ("probe_validator", str(installed["probe_validator"])),
                ("probe_validator_sha256", EXPECTED["probe_validator"][1]),
            )
        }
        unsigned = {
            "schema_version": (
                "saic-formal-v2-retained-fd-world8-release-manifest-v1"
            ),
            "status": "sealed_before_canary_submission",
            "stem": STEM,
            "release_root": str(RELEASE),
            "output_parent": str(OUTPUT_PARENT),
            "inputs": inputs,
            "postflight": {
                "path": str(installed["postflight"]),
                "sha256": EXPECTED["postflight"][1],
                "sha256_pinned_outside_postflight_source": True,
            },
            "executables": {
                "python": str(PYTHON),
                "python_sha256": PYTHON_SHA256,
                "sacct": str(SACCT),
                "sacct_sha256": SACCT_SHA256,
            },
            "probe_admission": probe_binding,
            "authority": AUTHORITY,
        }
        manifest = dict(unsigned)
        manifest["receipt_digest"] = sha_bytes(canonical(unsigned))
        manifest_path = RELEASE / "release-manifest.json"
        descriptor = os.open(
            manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            write_all(descriptor, canonical(manifest) + b"\n")
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)

        os.chmod(INPUTS, 0o555)
        os.chmod(POSTFLIGHT_ROOT, 0o555)
        os.chmod(RELEASE, 0o555)
        for path in (INPUTS, POSTFLIGHT_ROOT, RELEASE, OUTPUT_PARENT, LOG_DIR):
            fsync_directory(path)
    except BaseException:
        # A partial release is intentionally left in place for audit.  Fresh r10
        # must never reuse or overwrite a failed namespace.
        raise

    print(
        json.dumps(
            {
                "release": str(RELEASE),
                "output_parent": str(OUTPUT_PARENT),
                "slurm_log_dir": str(LOG_DIR),
                "release_manifest": str(RELEASE / "release-manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
