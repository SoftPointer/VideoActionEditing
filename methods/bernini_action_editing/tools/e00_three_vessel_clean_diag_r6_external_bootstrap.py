#!/usr/bin/env python3
"""Stdlib-only R6 root verifier and opened-FD consumer executor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


ROOT_SCHEMA = "bernini-e00-clean-diagnostic-r6-external-bootstrap-root-v6"
REVISION_TAG = "E00_DFIX2_CLEAN_DIAG_R6_EXTERNAL_BOOTSTRAP_20260821"
HEX = set("0123456789abcdef")


class BootstrapError(RuntimeError):
    pass


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        raise BootstrapError(f"{label} is not a lowercase SHA-256")
    return value


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise BootstrapError(f"{label} is unsafe")
    return value


def _open_plain(path: Path, label: str) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise BootstrapError(f"cannot open {label}: {path}") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise BootstrapError(f"{label} is not a regular file: {path}")
    return descriptor


def _fd_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _fd_read_all(descriptor: int) -> bytes:
    chunks = []
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def _reject_cache_bytecode(package_root: Path) -> None:
    if not package_root.is_dir() or package_root.is_symlink():
        raise BootstrapError("package root is not a plain directory")
    for current, directories, files in os.walk(str(package_root), topdown=True, followlinks=False):
        if "__pycache__" in directories:
            raise BootstrapError(f"package contains __pycache__: {current}")
        for name in files:
            if name.endswith(".pyc"):
                raise BootstrapError(f"package contains bytecode: {Path(current) / name}")


def validate_root(
    *, package_root: Path, root_path: Path, expected_root_sha256: str,
    consumer_relative: Optional[str],
) -> tuple[Mapping[str, Any], Optional[int]]:
    _hex(expected_root_sha256, "expected root SHA-256")
    root_descriptor = _open_plain(root_path, "R6 root")
    try:
        root_bytes = _fd_read_all(root_descriptor)
        if hashlib.sha256(root_bytes).hexdigest() != expected_root_sha256:
            raise BootstrapError("R6 root bytes differ")
        try:
            value = json.loads(root_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise BootstrapError("R6 root is unreadable") from error
    finally:
        os.close(root_descriptor)
    for field, expected in (
        ("schema_version", ROOT_SCHEMA), ("revision_tag", REVISION_TAG),
        ("complete", True), ("immutable", True),
        ("one_way_root_pins", True), ("consumers_pin_root", False),
        ("runtime_diagnostic_only", True), ("property_preservation_fix_claimed", False),
    ):
        if value.get(field) != expected:
            raise BootstrapError(f"R6 root {field} differs")
    pins = value.get("pins")
    consumers = value.get("runtime_consumers")
    if not isinstance(pins, dict) or not pins:
        raise BootstrapError("R6 root pins are absent")
    if value.get("pinned_file_count") != len(pins):
        raise BootstrapError("R6 root pin count differs")
    if not isinstance(consumers, list) or not consumers:
        raise BootstrapError("R6 runtime consumers are absent")
    if any(relative not in pins for relative in consumers):
        raise BootstrapError("R6 root does not pin every runtime consumer")
    _reject_cache_bytecode(package_root)
    kept_descriptor: Optional[int] = None
    selected = _safe_relative(consumer_relative, "consumer") if consumer_relative is not None else None
    if selected is not None and selected not in consumers:
        raise BootstrapError("requested consumer is not registered")
    for relative, expected_sha in pins.items():
        relative = _safe_relative(relative, "root pin")
        _hex(expected_sha, f"root pin {relative}")
        descriptor = _open_plain(package_root / relative, f"root pin {relative}")
        try:
            if _fd_sha256(descriptor) != expected_sha:
                raise BootstrapError(f"R6 pinned current bytes differ: {relative}")
            if relative == selected:
                kept_descriptor = descriptor
                descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    _reject_cache_bytecode(package_root)
    return value, kept_descriptor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-root-sha256", required=True)
    parser.add_argument("--consumer-relative")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("consumer_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    package_root = Path(args.package_root).resolve()
    root_path = Path(args.root).resolve()
    expected_root_path = package_root / "methods/bernini_action_editing/assets/e00_three_vessel_clean_diag_r6_EXTERNAL_ROOT.json"
    if root_path != expected_root_path.resolve():
        raise BootstrapError("R6 root path differs")
    if args.verify_only and args.consumer_relative is not None:
        raise BootstrapError("verify-only does not accept a consumer")
    if not args.verify_only and args.consumer_relative is None:
        raise BootstrapError("consumer is required for execution")
    root, descriptor = validate_root(
        package_root=package_root, root_path=root_path,
        expected_root_sha256=args.expected_root_sha256,
        consumer_relative=args.consumer_relative,
    )
    if args.verify_only:
        print(json.dumps({
            "schema_version": "bernini-e00-clean-diagnostic-r6-bootstrap-verification-v6",
            "revision_tag": REVISION_TAG,
            "complete": True,
            "root_sha256": args.expected_root_sha256,
            "pinned_file_count": root["pinned_file_count"],
            "cache_bytecode_absent": True,
            "local_modules_imported": False,
        }, sort_keys=True), flush=True)
        return 0
    if descriptor is None:
        raise BootstrapError("verified consumer descriptor is absent")
    os.set_inheritable(descriptor, True)
    fd_root = "/proc/self/fd" if Path("/proc/self/fd").is_dir() else "/dev/fd"
    consumer_fd_path = f"{fd_root}/{descriptor}"
    consumer_args = list(args.consumer_args)
    if consumer_args and consumer_args[0] == "--":
        consumer_args = consumer_args[1:]
    environment = dict(os.environ)
    environment.update({
        "E00_R6_BOOTSTRAP_VERIFIED": "1",
        "E00_R6_PACKAGE_ROOT_VERIFIED": str(package_root),
        "E00_R6_ROOT_PATH_VERIFIED": str(root_path),
        "E00_R6_EXPECTED_ROOT_SHA256": args.expected_root_sha256,
        "E00_R6_BOOTSTRAP_VERIFIER": str(Path(__file__).resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    os.execve("/bin/bash", ["bash", consumer_fd_path, *consumer_args], environment)
    raise BootstrapError("consumer exec unexpectedly returned")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as error:
        print(f"R6_BOOTSTRAP_REJECTED: {error}", file=sys.stderr)
        raise SystemExit(7)
