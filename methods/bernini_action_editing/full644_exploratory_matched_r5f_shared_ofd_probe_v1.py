#!/usr/bin/env python3
"""Linux CPU-only proof that shared-OFD consumers must use ``pread``.

The parent retains one regular-file descriptor at a nonzero offset.  Four
forked children cross a barrier and each perform two positional reads of the
whole authority.  A separate open file description runs a deliberately
staged ``lseek``/``read`` control, where one reader succeeds and three observe
the shared-offset race.  No Torch, GPU, SSH, Slurm, or model code is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "full644-exploratory-matched-r5f-shared-ofd-probe-v1"
WORKER_SCHEMA = "full644-exploratory-matched-r5f-shared-ofd-worker-v1"
WORKER_COUNT = 4
READS_PER_WORKER = 2
MAX_AUTHORITY_SIZE = 8 * 1024 * 1024
_HEX256 = re.compile(r"[0-9a-f]{64}")


class R5FSharedOFDProbeError(RuntimeError):
    """A closed probe contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise R5FSharedOFDProbeError("noncanonical-json", "value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stat_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "mode": int(info.st_mode),
        "nlink": int(info.st_nlink),
        "rdev": int(info.st_rdev),
        "size": int(info.st_size),
        "blocks": int(getattr(info, "st_blocks", 0)),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _fstat_identity(descriptor: int) -> dict[str, int]:
    try:
        info = os.fstat(descriptor)
    except OSError as error:
        raise R5FSharedOFDProbeError("fd-unavailable", "retained FD is unavailable") from error
    if not stat.S_ISREG(info.st_mode):
        raise R5FSharedOFDProbeError("fd-not-regular", "retained FD is not regular")
    return stat_identity(info)


def _validate_expected(size: int, digest: str) -> None:
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or size > MAX_AUTHORITY_SIZE
    ):
        raise R5FSharedOFDProbeError("invalid-size", "authority size is outside the closed range")
    if not isinstance(digest, str) or _HEX256.fullmatch(digest) is None:
        raise R5FSharedOFDProbeError("invalid-digest", "expected SHA-256 is malformed")


def verify_double_pread(
    descriptor: int,
    expected_identity: Mapping[str, int],
    expected_size: int,
    expected_sha256: str,
    *,
    pread: Callable[[int, int, int], bytes] = os.pread,
) -> dict[str, Any]:
    """Perform exactly two whole-file positional reads without moving the OFD offset."""
    _validate_expected(expected_size, expected_sha256)
    digests: list[str] = []
    for read_index in range(READS_PER_WORKER):
        if _fstat_identity(descriptor) != dict(expected_identity):
            raise R5FSharedOFDProbeError("identity-mismatch", "retained FD identity changed")
        try:
            raw = pread(descriptor, expected_size + 1, 0)
        except OSError as error:
            raise R5FSharedOFDProbeError("fd-unavailable", "positional read failed") from error
        if not isinstance(raw, bytes):
            raise R5FSharedOFDProbeError("invalid-read", "positional read did not return bytes")
        if len(raw) < expected_size:
            raise R5FSharedOFDProbeError("short-read", "positional read was short")
        if len(raw) > expected_size:
            raise R5FSharedOFDProbeError("size-mismatch", "authority grew beyond its sealed size")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha256:
            raise R5FSharedOFDProbeError("digest-mismatch", "positional read digest differed")
        if _fstat_identity(descriptor) != dict(expected_identity):
            raise R5FSharedOFDProbeError("identity-mismatch", "retained FD identity changed")
        digests.append(digest)
    return {"read_count": len(digests), "sha256": digests}


def gpu_device_descriptors() -> list[str]:
    rows: list[str] = []
    root = Path("/proc/self/fd")
    if not root.is_dir():
        return rows
    for entry in root.iterdir():
        try:
            target = os.readlink(str(entry))
        except OSError:
            continue
        if target == "/dev/kfd" or target.startswith("/dev/dri/"):
            rows.append(target)
    return sorted(rows)


def _validate_cpu_only(*, require_isolation: bool) -> None:
    if require_isolation and (
        sys.platform != "linux"
        or not Path("/proc/self/fd").is_dir()
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or not sys.dont_write_bytecode
        or sys.flags.optimize not in (0, 1)
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise R5FSharedOFDProbeError(
            "runtime-not-isolated", "probe requires Linux python -I -S -B"
        )
    if "torch" in sys.modules:
        raise R5FSharedOFDProbeError("torch-loaded", "Torch must remain unloaded")
    if gpu_device_descriptors():
        raise R5FSharedOFDProbeError("gpu-fd-open", "GPU device descriptor is open")


def _close(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        offset += os.write(descriptor, raw[offset:])


def _read_all(descriptor: int, limit: int = 128 * 1024) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        block = os.read(descriptor, 4096)
        if not block:
            break
        size += len(block)
        if size > limit:
            raise R5FSharedOFDProbeError("worker-report-large", "worker report is too large")
        chunks.append(block)
    return b"".join(chunks)


def _worker_result(
    mode: str,
    worker_id: int,
    descriptor: int,
    identity: Mapping[str, int],
    size: int,
    digest: str,
) -> dict[str, Any]:
    if mode == "pread":
        proof = verify_double_pread(descriptor, identity, size, digest)
        return {
            "schema_version": WORKER_SCHEMA,
            "worker_id": worker_id,
            "status": "PASS",
            "method": "pread",
            **proof,
            "torch_loaded": "torch" in sys.modules,
            "gpu_device_fds": gpu_device_descriptors(),
        }
    raw = os.read(descriptor, size + 1)
    observed = hashlib.sha256(raw).hexdigest() if len(raw) == size else None
    if len(raw) == size and observed != digest:
        raise R5FSharedOFDProbeError("digest-mismatch", "legacy full read digest differed")
    return {
        "schema_version": WORKER_SCHEMA,
        "worker_id": worker_id,
        "status": "FULL" if len(raw) == size else "CONTENDED",
        "method": "lseek-read",
        "bytes_read": len(raw),
        "sha256": observed,
        "rejection_code": None if len(raw) == size else "short-read",
    }


def _run_barrier_workers(
    descriptor: int,
    identity: Mapping[str, int],
    size: int,
    digest: str,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    if not hasattr(os, "fork") or mode not in {"pread", "lseek-read"}:
        raise R5FSharedOFDProbeError("fork-unavailable", "fork worker mode is unavailable")
    channels = [(*os.pipe(), *os.pipe(), *os.pipe()) for _ in range(WORKER_COUNT)]
    all_fds = [descriptor] + [fd for row in channels for fd in row]
    pids: list[int] = []
    try:
        for worker_id, row in enumerate(channels):
            ready_r, ready_w, go_r, go_w, result_r, result_w = row
            pid = os.fork()
            if pid == 0:
                keep = {descriptor, ready_w, go_r, result_w}
                for child_fd in all_fds:
                    if child_fd not in keep:
                        _close(child_fd)
                result: dict[str, Any]
                exit_code = 0
                try:
                    _validate_cpu_only(require_isolation=False)
                    if _fstat_identity(descriptor) != dict(identity):
                        raise R5FSharedOFDProbeError("identity-mismatch", "worker FD identity differed")
                    if mode == "lseek-read":
                        os.lseek(descriptor, 0, os.SEEK_SET)
                    _write_all(ready_w, b"R")
                    if os.read(go_r, 1) != b"G":
                        raise R5FSharedOFDProbeError("barrier-failed", "worker barrier failed")
                    result = _worker_result(mode, worker_id, descriptor, identity, size, digest)
                except R5FSharedOFDProbeError as error:
                    exit_code = 1
                    result = {
                        "schema_version": WORKER_SCHEMA,
                        "worker_id": worker_id,
                        "status": "REJECT",
                        "rejection_code": error.code,
                    }
                except BaseException:
                    exit_code = 1
                    result = {
                        "schema_version": WORKER_SCHEMA,
                        "worker_id": worker_id,
                        "status": "REJECT",
                        "rejection_code": "unexpected-worker-error",
                    }
                try:
                    _write_all(result_w, canonical_bytes(result) + b"\n")
                finally:
                    os._exit(exit_code)
            pids.append(pid)

        for ready_r, ready_w, go_r, go_w, result_r, result_w in channels:
            _close(ready_w)
            _close(go_r)
            _close(result_w)
        ready = [os.read(row[0], 1) for row in channels]
        for row in channels:
            _write_all(row[3], b"G")
            _close(row[3])
        reports: list[dict[str, Any]] = []
        for row in channels:
            raw = _read_all(row[4])
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise R5FSharedOFDProbeError("worker-report-invalid", "worker report is invalid") from error
            if raw != canonical_bytes(value) + b"\n" or not isinstance(value, dict):
                raise R5FSharedOFDProbeError("worker-report-invalid", "worker report is noncanonical")
            reports.append(value)
        statuses = [os.waitpid(pid, 0)[1] for pid in pids]
        if ready != [b"R"] * WORKER_COUNT:
            raise R5FSharedOFDProbeError("barrier-failed", "not all workers reached the barrier")
        if mode == "pread" and any(status != 0 for status in statuses):
            raise R5FSharedOFDProbeError("pread-worker-failed", "a pread worker rejected")
        return sorted(reports, key=lambda row: row["worker_id"])
    finally:
        for row in channels:
            for channel_fd in row:
                _close(channel_fd)
        for pid in pids:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass


def run_pread_concurrency(
    descriptor: int, identity: Mapping[str, int], size: int, digest: str, sentinel: int
) -> dict[str, Any]:
    os.lseek(descriptor, sentinel, os.SEEK_SET)
    before = os.lseek(descriptor, 0, os.SEEK_CUR)
    workers = _run_barrier_workers(descriptor, identity, size, digest, mode="pread")
    after = os.lseek(descriptor, 0, os.SEEK_CUR)
    if before != sentinel or after != sentinel:
        raise R5FSharedOFDProbeError("offset-moved", "pread workers moved the parent OFD offset")
    if any(
        row.get("status") != "PASS"
        or row.get("read_count") != READS_PER_WORKER
        or row.get("sha256") != [digest, digest]
        or row.get("torch_loaded") is not False
        or row.get("gpu_device_fds") != []
        for row in workers
    ):
        raise R5FSharedOFDProbeError("pread-proof-invalid", "pread worker proof differed")
    return {"parent_offset_before": before, "parent_offset_after": after, "workers": workers}


def run_legacy_contention_control(
    descriptor: int, identity: Mapping[str, int], size: int, digest: str
) -> dict[str, Any]:
    workers = _run_barrier_workers(descriptor, identity, size, digest, mode="lseek-read")
    full = [row for row in workers if row.get("status") == "FULL"]
    contended = [row for row in workers if row.get("status") == "CONTENDED"]
    if (
        len(full) != 1
        or len(contended) != WORKER_COUNT - 1
        or any(row.get("rejection_code") != "short-read" for row in contended)
    ):
        raise R5FSharedOFDProbeError("legacy-control-invalid", "legacy race was not detected exactly")
    return {
        "full_read_count": len(full),
        "short_read_count": len(contended),
        "contention_detected": True,
        "workers": workers,
    }


def _expect_rejection(label: str, expected_code: str, action: Callable[[], Any]) -> dict[str, str]:
    try:
        action()
    except R5FSharedOFDProbeError as error:
        if error.code != expected_code:
            raise R5FSharedOFDProbeError("hostile-wrong-rejection", f"{label} rejected as {error.code}")
        return {"case": label, "rejection_code": error.code, "status": "REJECTED"}
    raise R5FSharedOFDProbeError("hostile-accepted", f"{label} was accepted")


def exercise_hostile_rejections(
    descriptor: int, identity: Mapping[str, int], size: int, digest: str
) -> list[dict[str, str]]:
    closed = os.dup(descriptor)
    os.close(closed)
    rows = [
        _expect_rejection(
            "closed-fd",
            "fd-unavailable",
            lambda: verify_double_pread(closed, identity, size, digest),
        )
    ]
    reused = os.dup(descriptor)
    os.close(reused)
    replacement = os.open(os.devnull, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    if replacement != reused:
        os.dup2(replacement, reused, inheritable=False)
    try:
        rows.append(
            _expect_rejection(
                "reused-fd",
                "fd-not-regular",
                lambda: verify_double_pread(reused, identity, size, digest),
            )
        )
    finally:
        _close(reused)
        if replacement != reused:
            _close(replacement)

    def short_pread(fd: int, count: int, offset: int) -> bytes:
        return os.pread(fd, max(0, count - 2), offset)

    rows.append(
        _expect_rejection(
            "short-read",
            "short-read",
            lambda: verify_double_pread(descriptor, identity, size, digest, pread=short_pread),
        )
    )
    wrong = ("0" if digest[0] != "0" else "1") + digest[1:]
    rows.append(
        _expect_rejection(
            "wrong-digest",
            "digest-mismatch",
            lambda: verify_double_pread(descriptor, identity, size, wrong),
        )
    )
    return rows


def write_receipt(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(path)
    receipt = dict(payload)
    if "receipt_digest" in receipt:
        raise R5FSharedOFDProbeError("receipt-invalid", "receipt digest is reserved")
    receipt["receipt_digest"] = object_sha256(receipt)
    raw = canonical_bytes(receipt) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(output), flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode) or os.fstat(descriptor).st_nlink != 1:
            raise R5FSharedOFDProbeError("receipt-invalid", "receipt target is not a private regular file")
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if stat.S_IMODE(final.st_mode) != 0o400 or final.st_nlink != 1 or final.st_size != len(raw):
            raise R5FSharedOFDProbeError("receipt-invalid", "receipt commit validation failed")
    finally:
        os.close(descriptor)
    return {
        "path": str(output.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "mode": 0o400,
        "receipt_digest": receipt["receipt_digest"],
    }


def run_probe(source: Path, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    _validate_expected(expected_size, expected_sha256)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(source), flags)
    try:
        identity = _fstat_identity(descriptor)
        try:
            named = stat_identity(source.lstat())
        except OSError as error:
            raise R5FSharedOFDProbeError("named-source-unavailable", "named source is unavailable") from error
        if identity != named or identity["size"] != expected_size:
            raise R5FSharedOFDProbeError("identity-mismatch", "named and retained identities differ")
        verify_double_pread(descriptor, identity, expected_size, expected_sha256)
        sentinel = min(17, expected_size)
        pread_proof = run_pread_concurrency(
            descriptor, identity, expected_size, expected_sha256, sentinel
        )
        control = os.open(
            f"/proc/self/fd/{descriptor}",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            if _fstat_identity(control) != identity:
                raise R5FSharedOFDProbeError("identity-mismatch", "control OFD identity differed")
            legacy = run_legacy_contention_control(control, identity, expected_size, expected_sha256)
        finally:
            os.close(control)
        hostiles = exercise_hostile_rejections(
            descriptor, identity, expected_size, expected_sha256
        )
        if os.lseek(descriptor, 0, os.SEEK_CUR) != sentinel:
            raise R5FSharedOFDProbeError("offset-moved", "parent retained offset changed after proof")
        return {
            "schema_version": SCHEMA,
            "status": "PASS",
            "runtime": {
                "platform": sys.platform,
                "isolated": sys.flags.isolated,
                "no_site": sys.flags.no_site,
                "ignore_environment": sys.flags.ignore_environment,
                "dont_write_bytecode": bool(sys.dont_write_bytecode),
                "optimize": sys.flags.optimize,
                "torch_loaded": "torch" in sys.modules,
                "gpu_device_fds": gpu_device_descriptors(),
            },
            "authority": {
                "path": str(source.resolve()),
                "size": expected_size,
                "sha256": expected_sha256,
                "identity": identity,
            },
            "pread_concurrency": pread_proof,
            "legacy_lseek_read_control": legacy,
            "hostile_rejections": hostiles,
            "summary": {
                "worker_count": WORKER_COUNT,
                "pread_count": WORKER_COUNT * READS_PER_WORKER,
                "hostile_rejection_count": len(hostiles),
                "legacy_contention_detected": True,
                "parent_offset_preserved": True,
            },
        }
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cpu_only(require_isolation=True)
    source = Path(args.source)
    receipt_path = Path(args.receipt)
    if not source.is_absolute() or not receipt_path.is_absolute() or receipt_path.parent.resolve() != receipt_path.parent:
        raise R5FSharedOFDProbeError("path-invalid", "source and receipt must be canonical absolute paths")
    payload = run_probe(source, args.expected_size, args.expected_sha256)
    reference = write_receipt(receipt_path, payload)
    sys.stdout.buffer.write(canonical_bytes(reference) + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
