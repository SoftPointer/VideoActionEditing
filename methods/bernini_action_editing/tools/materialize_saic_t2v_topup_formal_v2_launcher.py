#!/usr/bin/env python3
"""Materialize the audited guard-v2 overlay of the frozen SAIC r6 launcher."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
from typing import Sequence


EXPECTED_BASE_SHA256 = (
    "12c1b2baaecfd479f65f9b5dbf0dbae17cd87196767e93c254fe2cffc895f29d"
)
EXPECTED_GUARD_V2_SHA256 = (
    "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
)

REPLACEMENTS = (
    (
        b'rendezvous_guard_sha256="${SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256:?set rendezvous guard SHA-256}"\n',
        b'rendezvous_guard_sha256="${SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256:?set rendezvous guard SHA-256}"\n'
        b'external_rendezvous_guard="${SAIC_T2V_FV2_RENDEZVOUS_GUARD_FD_PATH:?set retained guard-v2 fd path}"\n',
    ),
    (
        b'for name in source_archive source_manifest event_spec bernini_root veomni_root checkpoint checkpoint_manifest output_root python_bin ffmpeg_bin; do\n',
        b'for name in source_archive source_manifest event_spec external_rendezvous_guard bernini_root veomni_root checkpoint checkpoint_manifest output_root python_bin ffmpeg_bin; do\n',
    ),
    (
        b'event_spec="$(realpath -e -- "${event_spec}")"\n',
        b'event_spec="$(realpath -e -- "${event_spec}")"\n'
        b'[[ "${external_rendezvous_guard}" =~ ^/proc/[0-9]+/fd/[0-9]+$ ]] || fail "retained guard-v2 fd path differs"\n',
    ),
    (
        b'[[ -f "${event_spec}" && ! -L "${event_spec}" ]] || fail "event spec is not plain"\n',
        b'[[ -f "${event_spec}" && ! -L "${event_spec}" ]] || fail "event spec is not plain"\n'
        b'[[ -r "${external_rendezvous_guard}" && -f "${external_rendezvous_guard}" && "$(stat -Lc \'%h\' -- "${external_rendezvous_guard}")" == 1 && "$(sha256sum "${external_rendezvous_guard}" | awk \'{print $1}\')" == "${rendezvous_guard_sha256}" ]] || fail "retained external guard-v2 differs"\n',
    ),
    (
        b'rendezvous_guard="${method_root}/saic_t2v_rendezvous_guard_v1.py"\n',
        b'rendezvous_guard="${external_rendezvous_guard}"\n',
    ),
    (
        b'[[ -f "${rendezvous_guard}" && ! -L "${rendezvous_guard}" && "$(sha256sum "${rendezvous_guard}" | awk \'{print $1}\')" == "${rendezvous_guard_sha256}" ]] || fail "rendezvous guard bytes differ"\n',
        b'[[ "${rendezvous_guard}" =~ ^/proc/[0-9]+/fd/[0-9]+$ && -r "${rendezvous_guard}" && -f "${rendezvous_guard}" && "$(stat -Lc \'%h\' -- "${rendezvous_guard}")" == 1 && "$(sha256sum "${rendezvous_guard}" | awk \'{print $1}\')" == "${rendezvous_guard_sha256}" ]] || fail "retained rendezvous guard bytes differ"\n',
    ),
    (
        b'"schema_version": "saic-t2v-topup-rendezvous-dynamic-plan-v1",\n',
        b'"schema_version": "saic-t2v-topup-rendezvous-dynamic-plan-v2",\n',
    ),
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_file(path: Path, expected_sha256: str, label: str) -> bytes:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise SystemExit(f"{label} identity differs")
    info = path.lstat()
    payload = path.read_bytes()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o444
        or sha256(payload) != expected_sha256
    ):
        raise SystemExit(f"{label} bytes differ")
    return payload


def directory_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SystemExit("effective launcher parent is not a plain directory")
    return info.st_dev, info.st_ino


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def transform(base: bytes) -> bytes:
    result = base
    for old, new in REPLACEMENTS:
        if result.count(old) != 1:
            raise SystemExit("frozen launcher overlay anchor count differs")
        result = result.replace(old, new, 1)
    if result == base or result.count(b"saic_t2v_rendezvous_guard_v1.py") != 3:
        # Three remaining references are the unchanged archive member and its
        # archived unit-test/closure assertions.  None is executed as the
        # operational rendezvous guard after the exact assignment replacement.
        raise SystemExit("guard-v2 overlay closure differs")
    if result.count(b'"saic-t2v-topup-rendezvous-dynamic-plan-v2"') != 1:
        raise SystemExit("guard-v2 plan schema overlay differs")
    if (
        result.count(b'! -L "${rendezvous_guard}"') != 0
        or result.count(b"retained rendezvous guard bytes differ") != 1
    ):
        raise SystemExit("retained guard runtime admission overlay differs")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-launcher", required=True)
    parser.add_argument("--guard-v2", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_path = Path(args.base_launcher)
    guard_path = Path(args.guard_v2)
    output = Path(args.output)
    base = exact_file(base_path, EXPECTED_BASE_SHA256, "frozen base launcher")
    exact_file(guard_path, EXPECTED_GUARD_V2_SHA256, "external guard v2")
    if (
        not output.is_absolute()
        or output.parent.resolve(strict=True) != output.parent
        or output.exists()
        or output.is_symlink()
    ):
        raise SystemExit("effective launcher output is not fresh/canonical")
    output_parent_identity = directory_identity(output.parent)
    payload = transform(base)
    descriptor = os.open(
        output, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        offset = 0
        while offset < len(payload):
            wrote = os.write(descriptor, payload[offset:])
            if wrote <= 0:
                raise RuntimeError("effective launcher write stalled")
            offset += wrote
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(payload) + 1) != payload:
            raise RuntimeError("effective launcher reread differs")
        observed = os.fstat(descriptor)
        leaf = output.lstat()
        if (
            observed.st_nlink != 1
            or leaf.st_nlink != 1
            or not stat.S_ISREG(observed.st_mode)
            or not stat.S_ISREG(leaf.st_mode)
            or stat.S_ISLNK(leaf.st_mode)
            or (observed.st_dev, observed.st_ino) != (leaf.st_dev, leaf.st_ino)
            or observed.st_size != len(payload)
            or directory_identity(output.parent) != output_parent_identity
        ):
            raise RuntimeError("effective launcher identity differs")
        fsync_directory(output.parent)
        if directory_identity(output.parent) != output_parent_identity:
            raise RuntimeError("effective launcher parent identity differs")
    except BaseException:
        os.close(descriptor)
        raise
    os.fchmod(descriptor, 0o444)
    try:
        os.close(descriptor)
    except OSError:
        pass
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
