#!/usr/bin/env python3
"""Create-only sealed release manifest for one review-v2 source freeze."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


FORMAL_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
RELEASE_ROOT = (
    FORMAL_ROOT
    / "releases/saic-t2v-topup-r6-formal-v2-r2-retfd-20260812-r1-review-v2-r1"
)
INPUTS = RELEASE_ROOT / "inputs"
OUTPUT = RELEASE_ROOT / "release-manifest.json"
FORMAL_OUTPUT_ROOT = (
    FORMAL_ROOT / "runs/t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1"
)
FORMAL_FILES = {
    "master_receipt":
        FORMAL_OUTPUT_ROOT / "saic-pure-t2v-event-bank-topup-receipt.json",
    "submission_receipt":
        Path(str(FORMAL_OUTPUT_ROOT) + ".submission.receipt.json"),
}
FILES = {
    "manifest_materializer":
        INPUTS / "materialize_saic_t2v_topup_review_v2_release_manifest_v1.py",
    "adapter": INPUTS / "build_saic_t2v_event_bank_topup_detached_review_v2.py",
    "launcher":
        INPUTS / "auh_build_saic_t2v_event_bank_topup_detached_review_v2_cpu.sbatch",
    "submitter":
        INPUTS / "submit_saic_t2v_event_bank_topup_detached_review_v2.py",
    "postflight":
        INPUTS / "postflight_saic_t2v_event_bank_topup_detached_review_v2.py",
    "hostile": INPUTS / "test_saic_t2v_topup_detached_review_v2_release_auh.py",
    "source_archive": INPUTS / "videoedit-saic-20c2193-methods.tar",
}
EXECUTABLES = {
    "python": Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"),
    "ffmpeg": Path(
        "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
        "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
    ),
    "compute_bash": Path("/usr/bin/bash"),
    "sacct": Path("/usr/bin/sacct"),
}
ZERO_AUTHORITY = {
    "scientific": False, "human_review": False, "event_verified": False,
    "identity_preservation_verified": False, "candidate_selection": False,
    "seed_selection": False, "training_target": False, "training": False,
    "optimizer_step": False, "parameter_update": False,
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_file(path: Path, *, executable: bool = False) -> dict[str, str]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        public = path.lstat()
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (not executable and before.st_uid != os.getuid())
            or stat.S_IMODE(before.st_mode) & 0o022
            or (not executable and stat.S_IMODE(before.st_mode) != 0o444)
            or (executable and not os.access(path, os.X_OK))
            or not stat.S_ISREG(public.st_mode)
            or stat.S_ISLNK(public.st_mode)
            or (before.st_dev, before.st_ino) != (public.st_dev, public.st_ino)
        ):
            raise SystemExit(f"release input identity differs: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        public_after = path.lstat()
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
            stat.S_IMODE(item.st_mode), item.st_nlink, item.st_uid,
        )
        if identity(before) != identity(after) or identity(after) != identity(public_after):
            raise SystemExit(f"release input changed while retained: {path}")
        return {"path": str(path), "sha256": digest.hexdigest()}
    finally:
        os.close(descriptor)


def main() -> int:
    if (
        Path(__file__) != FILES["manifest_materializer"]
        or Path(__file__).resolve(strict=True) != FILES["manifest_materializer"]
    ):
        raise SystemExit("review release materializer execution path differs")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise SystemExit("review release manifest target is not fresh")
    release_info = RELEASE_ROOT.lstat()
    inputs_info = INPUTS.lstat()
    if (
        RELEASE_ROOT.resolve(strict=True) != RELEASE_ROOT
        or INPUTS.resolve(strict=True) != INPUTS
        or not stat.S_ISDIR(release_info.st_mode)
        or not stat.S_ISDIR(inputs_info.st_mode)
        or stat.S_ISLNK(release_info.st_mode)
        or stat.S_ISLNK(inputs_info.st_mode)
        or release_info.st_uid != os.getuid()
        or inputs_info.st_uid != os.getuid()
        or stat.S_IMODE(release_info.st_mode) & 0o022
        or stat.S_IMODE(inputs_info.st_mode) & 0o022
    ):
        raise SystemExit("review release directory identity differs")
    if (
        set(INPUTS.iterdir()) != set(FILES.values())
        or set(RELEASE_ROOT.iterdir()) != {INPUTS}
    ):
        raise SystemExit("review pre-manifest release namespace differs")
    body = {
        "schema_version": "saic-t2v-topup-review-v2-release-manifest-v1",
        "status": "sealed_before_review_submission",
        "release_root": str(RELEASE_ROOT),
        "inputs": {name: exact_file(path) for name, path in FILES.items()},
        "formal_inputs": {
            name: exact_file(path) for name, path in FORMAL_FILES.items()
        },
        "executables": {
            name: exact_file(path, executable=True)
            for name, path in EXECUTABLES.items()
        },
        "authority": ZERO_AUTHORITY,
    }
    value = {**body, "receipt_digest": hashlib.sha256(canonical(body)).hexdigest()}
    payload = canonical(value) + b"\n"
    parent_descriptor = os.open(
        RELEASE_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    descriptor = -1
    try:
        opened_parent = os.fstat(parent_descriptor)
        if (
            (opened_parent.st_dev, opened_parent.st_ino)
            != (release_info.st_dev, release_info.st_ino)
        ):
            raise SystemExit("retained release parent identity differs")
        descriptor = os.open(
            OUTPUT.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        offset = 0
        while offset < len(payload):
            wrote = os.write(descriptor, payload[offset:])
            if wrote <= 0:
                raise SystemExit("release manifest write stalled")
            offset += wrote
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(payload) + 1) != payload:
            raise SystemExit("release manifest same-FD reread differs")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        public = os.stat(
            OUTPUT.name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
        public_path = OUTPUT.lstat()
        retained = os.fstat(descriptor)
        if (
            os.read(descriptor, len(payload) + 1) != payload
            or not stat.S_ISREG(retained.st_mode)
            or retained.st_nlink != 1
            or retained.st_uid != os.getuid()
            or stat.S_IMODE(retained.st_mode) != 0o444
            or retained.st_size != len(payload)
            or not stat.S_ISREG(public.st_mode)
            or not stat.S_ISREG(public_path.st_mode)
            or stat.S_ISLNK(public_path.st_mode)
            or public.st_nlink != 1
            or public_path.st_nlink != 1
            or public.st_uid != os.getuid()
            or public_path.st_uid != os.getuid()
            or stat.S_IMODE(public.st_mode) != 0o444
            or stat.S_IMODE(public_path.st_mode) != 0o444
            or public.st_size != len(payload)
            or public_path.st_size != len(payload)
            or (retained.st_dev, retained.st_ino)
            != (public.st_dev, public.st_ino)
            or (retained.st_dev, retained.st_ino)
            != (public_path.st_dev, public_path.st_ino)
            or set(RELEASE_ROOT.iterdir()) != {INPUTS, OUTPUT}
        ):
            raise SystemExit("sealed release manifest identity differs")
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    print(
        canonical({
            "path": str(OUTPUT), "sha256": hashlib.sha256(payload).hexdigest(),
            "receipt_digest": value["receipt_digest"],
        }).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
