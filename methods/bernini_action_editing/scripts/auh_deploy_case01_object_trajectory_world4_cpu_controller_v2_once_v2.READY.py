#!/usr/bin/env python3
"""Create-only AUH deployer for the frozen case01 CPU READY controller.

The checked-in source is inert.  A separately reviewed state-only READY copy
may capture the exact controller, the separate frozen remote bootstrap, and
the actual SSH executable/key/known-hosts through retained descriptors.  On
Darwin, all three are consumed by their canonical names: the kernel refuses
to execute system SSH through ``/dev/fd`` and OpenSSH closes inherited file
descriptors before opening credentials.  Their pinned bytes and exact inodes
remain held and are replayed around the sole spawn.  System SSH additionally
requires a restricted root-owned inode on the read-only sealed system volume
and a bounded Mach-O code-signature structure check.  Exact SHA-256 is the
content authority; structural parsing is not signature authentication.

Credential parents are canonical owner-controlled, non-writable directories.
A same-UID/root/kernel/mount attacker and the residual named lookup windows
are explicitly outside this controller's threat model.  One isolated remote
Python reserves the sibling admission receipt with O_EXCL, builds a sealed
shadow, and performs one ordinary same-parent POSIX rename.  The whole tree is
atomically visible; the held receipt inode becomes admission only at immutable
0400.  This is cooperative exclusion, not a kernel no-overwrite claim.  This
deployer never starts Slurm and never executes the deployed controller.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


SCHEMA = "case01-object-trajectory-world4-cpu-controller-deploy-v2"
MANIFEST_SCHEMA = SCHEMA + "-manifest"
PAYLOAD_SCHEMA = SCHEMA + "-payload"
RECEIPT_SCHEMA = SCHEMA + "-receipt"
RECEIPT_RESERVATION_SCHEMA = SCHEMA + "-receipt-reservation"
TERMINAL_SCHEMA = SCHEMA + "-terminal"
LOCAL_TERMINAL_SCHEMA = SCHEMA + "-local-commit-terminal-authority"
AUDIT_SCHEMA = SCHEMA + "-local-audit"
STAGE_OPERATION = "stage"
RECOVER_RECEIPT_OPERATION = "recover-receipt"
COMMIT_RECOVERY_REQUIRED_RC = 75
CONTROLLER_STATE = "READY_EXPLICIT_CREATE_ONLY_CONTROLLER_DEPLOY"
READY_STATE = "READY_EXPLICIT_CREATE_ONLY_CONTROLLER_DEPLOY"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

LOCAL_ROOT = Path("/Users/siriuschu/ML/VideoEditing/VideoEdit")
LOCAL_COMMIT_TERMINAL_PATH = LOCAL_ROOT / (
    "artifacts/case01_object_trajectory_world4_cpu_controller_deploy_v2."
    "commit_terminal.json"
)
LOCAL_CONTROLLER = LOCAL_ROOT / (
    "methods/bernini_action_editing/"
    "case01_object_trajectory_exact5_world4_cpu_auh_controller_v2.READY.py"
)
LOCAL_CONTROLLER_SHA256 = (
    "9d5aebcdf4b7938848e0763b839010fbd58df196f8a0155515b05a032cc99cbd"
)
LOCAL_CONTROLLER_SIZE = 86_998
LOCAL_BOOTSTRAP = LOCAL_ROOT / (
    "methods/bernini_action_editing/tools/"
    "case01_object_trajectory_world4_cpu_controller_deploy_bootstrap_v1.py"
)
LOCAL_BOOTSTRAP_SHA256 = (
    "2dca27942a3faae8fb0db019784682e288029f6794f573ae848c6292d0a15216"
)
LOCAL_BOOTSTRAP_SIZE = 118_408
LOCAL_UID = 501
LOCAL_GID = 20

REMOTE_PARENT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments"
)
REMOTE_TARGET_ROOT = (
    REMOTE_PARENT
    / "bernini_case01_object_trajectory_exact5_world4_cpu_controller_v2"
)
CONTROLLER_BASENAME = (
    "case01_object_trajectory_exact5_world4_cpu_auh_controller_v2.READY.py"
)
REMOTE_CONTROLLER_PATH = REMOTE_TARGET_ROOT / CONTROLLER_BASENAME
REMOTE_RECEIPT_PATH = REMOTE_PARENT / (
    "bernini_case01_object_trajectory_exact5_world4_cpu_controller_v2."
    "deployment_receipt_v2.json"
)
REMOTE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
REMOTE_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
REMOTE_PYTHON_SIZE = 31_490_256
REMOTE_UID = 2012
REMOTE_GID = 2000
FILE_MODE = 0o444
RECEIPT_MODE = 0o400
RECEIPT_RESERVATION_MODE = 0o600
DIRECTORY_MODE = 0o555

SSH_PATH = Path("/usr/bin/ssh")
SSH_SHA256 = "75ae4b414b57e0c52ad1cb24a9d7dae2496071fdf153c7fc8e94db3c9c4b0faa"
SSH_SIZE = 1_474_128
DARWIN_SF_RESTRICTED = 0x0008_0000
SSH_FAT_ARCHITECTURES = (
    (0x0100_0007, 0x0000_0003, 16_384, 720_544, 14),
    (0x0100_000C, 0x8000_0002, 737_280, 736_848, 14),
)
SSH_CODE_DIRECTORIES = (
    (170, 696_288, 5_766),
    (174, 712_464, 5_894),
)
SSH_IDENTITY = Path("/Users/siriuschu/.ssh/ciai2/id_ed25519")
SSH_IDENTITY_SHA256 = (
    "b41c88847bd284896de55df0231c5c6fced1d1b32a4a3aca6e8682b5eaaf8651"
)
SSH_IDENTITY_SIZE = 419
SSH_KNOWN_HOSTS = Path("/Users/siriuschu/.ssh/known_hosts")
SSH_KNOWN_HOSTS_SHA256 = (
    "3337d55aea085faada7937b20aa4cd12a908c13f1c4142704832bba46145bbaa"
)
SSH_KNOWN_HOSTS_SIZE = 18_620
SSH_DESTINATION = "guangyi.chen@172.27.112.248"

TRANSPORT_TIMEOUT_SECONDS = 180
PROCESS_TERM_GRACE_SECONDS = 1.0
PROCESS_KILL_GRACE_SECONDS = 3.0
PROCESS_POLL_SECONDS = 0.02
MAX_BOOTSTRAP_SIZE = 131_072
MAX_PAYLOAD_SIZE = 262_144
TRANSPORT_DIAGNOSTIC_SCHEMA = SCHEMA + "-transport-terminal-diagnostic-v1"
TRANSPORT_DIAGNOSTIC_PREFIX_LIMIT = 4_096
MAX_RECEIPT_SIZE = 1_048_576


class ControllerDeployerError(RuntimeError):
    """The create-only controller deployment contract differs."""


class ControllerCommitRecoveryRequired(ControllerDeployerError):
    """The target committed but its sibling receipt remains non-admissible."""

    def __init__(self, terminal: Mapping[str, Any]) -> None:
        super().__init__("controller target committed; recover-receipt is required")
        self.terminal = dict(terminal)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ControllerDeployerError("value is not canonical JSON") from error


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_uid),
        int(info.st_gid), int(info.st_mode), int(info.st_nlink),
        int(info.st_rdev), int(info.st_size),
        int(getattr(info, "st_blocks", 0)), int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _inode_anchor(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(stat.S_IFMT(info.st_mode)),
    )


def _inode_anchor_from_identity(identity: Sequence[int]) -> tuple[int, ...]:
    if len(identity) != 11:
        raise ControllerDeployerError("full inode identity shape differs")
    return (
        int(identity[0]), int(identity[1]), int(identity[2]), int(identity[3]),
        int(stat.S_IFMT(identity[4])),
    )


def _read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise ControllerDeployerError("held read size differs")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1_048_576, size - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size or os.pread(descriptor, 1, size) != b"":
        raise ControllerDeployerError("held read is incomplete")
    return raw


class HeldAuthority:
    def __init__(
        self, path: Path, descriptor: int, identity: tuple[int, ...],
        raw: bytes, sha256: str,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.identity = identity
        self.raw = raw
        self.sha256 = sha256

    def replay(self) -> None:
        opened = os.fstat(self.descriptor)
        named = os.lstat(self.path)
        replay = _read_fd(self.descriptor, opened.st_size)
        if (
            _identity(opened) != self.identity
            or _identity(named) != self.identity
            or replay != self.raw
            or hashlib.sha256(replay).hexdigest() != self.sha256
        ):
            raise ControllerDeployerError(
                f"held authority changed: {self.path}"
            )

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def _open_authority(
    path: Path, *, sha256: str, size: int, uid: int, gid: int,
    mode: int, nlink: int = 1,
) -> HeldAuthority:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or path.resolve(strict=True) != path
        or type(sha256) is not str or SHA_RE.fullmatch(sha256) is None
        or type(size) is not int or size <= 0
    ):
        raise ControllerDeployerError(f"authority path or pin differs: {path}")
    named = os.lstat(path)
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != nlink
        or named.st_size != size or named.st_uid != uid or named.st_gid != gid
        or stat.S_IMODE(named.st_mode) != mode
    ):
        raise ControllerDeployerError(f"authority identity differs: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor, before.st_size)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named_after = os.lstat(path)
        if (
            _identity(before) != _identity(named)
            or _identity(before) != _identity(middle)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named_after)
            or first != second
            or hashlib.sha256(first).hexdigest() != sha256
        ):
            raise ControllerDeployerError(f"authority replay differs: {path}")
        return HeldAuthority(path, descriptor, _identity(before), first, sha256)
    except BaseException:
        os.close(descriptor)
        raise


def _open_local_terminal_authority(path: Path) -> HeldAuthority:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.resolve(strict=True) != path
    ):
        raise ControllerDeployerError(
            f"local terminal authority path differs: {path}"
        )
    named = os.lstat(path)
    if (
        not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or stat.S_IMODE(named.st_mode) != RECEIPT_MODE
        or named.st_uid != LOCAL_UID
        or named.st_gid != LOCAL_GID
    ):
        raise ControllerDeployerError(
            f"local terminal authority identity differs: {path}"
        )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor, before.st_size)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named_after = os.lstat(path)
        if (
            _identity(before) != _identity(named)
            or _identity(before) != _identity(middle)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named_after)
            or first != second
        ):
            raise ControllerDeployerError(
                f"local terminal authority replay differs: {path}"
            )
        return HeldAuthority(
            path, descriptor, _identity(before), first,
            hashlib.sha256(first).hexdigest(),
        )
    except BaseException:
        os.close(descriptor)
        raise


def _local_terminal_value(
    remote_terminal: Mapping[str, Any], path: Path,
    inode_anchor: tuple[int, ...],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": LOCAL_TERMINAL_SCHEMA,
        "status": "TARGET_RENAMED_LOCAL_TERMINAL_PERSISTED",
        "terminal_path": str(path),
        "local_terminal_inode_anchor": list(inode_anchor),
        "remote_commit_terminal": dict(remote_terminal),
        "launch_allowed": False,
    }
    value["authority_digest"] = object_digest(value)
    return value


def _valid_local_commit_terminal_path(path: Path) -> bool:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.parent != LOCAL_COMMIT_TERMINAL_PATH.parent
    ):
        return False
    if path == LOCAL_COMMIT_TERMINAL_PATH:
        return True
    prefix = LOCAL_COMMIT_TERMINAL_PATH.name + "."
    return (
        path.name.startswith(prefix)
        and SHA_RE.fullmatch(path.name[len(prefix):]) is not None
    )


def _terminal_output_path(
    operation: str, remote_terminal: Mapping[str, Any],
) -> Path:
    if operation == STAGE_OPERATION:
        return LOCAL_COMMIT_TERMINAL_PATH
    digest = remote_terminal.get("terminal_digest")
    if (
        operation != RECOVER_RECEIPT_OPERATION
        or SHA_RE.fullmatch(str(digest)) is None
    ):
        raise ControllerDeployerError(
            "refreshed remote terminal digest differs"
        )
    return LOCAL_COMMIT_TERMINAL_PATH.parent / (
        LOCAL_COMMIT_TERMINAL_PATH.name + "." + str(digest)
    )


def _validate_local_terminal_value(
    value: Mapping[str, Any], *, path: Path, identity: tuple[int, ...],
) -> dict[str, Any]:
    remote_terminal = value.get("remote_commit_terminal")
    if type(remote_terminal) is not dict:
        raise ControllerDeployerError(
            "local commit terminal remote value differs"
        )
    if path != LOCAL_COMMIT_TERMINAL_PATH:
        digest = remote_terminal.get("terminal_digest")
        expected = LOCAL_COMMIT_TERMINAL_PATH.parent / (
            LOCAL_COMMIT_TERMINAL_PATH.name + "." + str(digest)
        )
        if SHA_RE.fullmatch(str(digest)) is None or path != expected:
            raise ControllerDeployerError(
                "refreshed local commit terminal path differs"
            )
    expected_value = _local_terminal_value(
        remote_terminal, path, _inode_anchor_from_identity(identity),
    )
    if value != expected_value:
        raise ControllerDeployerError(
            "local commit terminal authority differs"
        )
    return dict(value)


class HeldCommitTerminal:
    def __init__(
        self, authority: HeldAuthority, value: Mapping[str, Any],
    ) -> None:
        self.authority = authority
        self.value = dict(value)

    def replay(self) -> dict[str, Any]:
        self.authority.replay()
        opened = os.fstat(self.authority.descriptor)
        if (
            stat.S_IMODE(opened.st_mode) != RECEIPT_MODE
            or opened.st_uid != LOCAL_UID
            or opened.st_gid != LOCAL_GID
        ):
            raise ControllerDeployerError(
                "held local commit terminal mode differs"
            )
        value = _strict_remote_json(
            self.authority.raw, label="local commit terminal authority",
        )
        validated = _validate_local_terminal_value(
            value, path=self.authority.path, identity=self.authority.identity,
        )
        if validated != self.value:
            raise ControllerDeployerError(
                "held local commit terminal value changed"
            )
        return validated

    def close(self) -> None:
        self.authority.close()


def _write_local_commit_terminal(
    path: Path, remote_terminal: Mapping[str, Any],
) -> HeldCommitTerminal:
    if (
        not _valid_local_commit_terminal_path(path)
        or path.parent.resolve(strict=True) != path.parent
    ):
        raise ControllerDeployerError("local commit terminal path differs")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor = -1
    committed = False
    try:
        parent_info = os.fstat(parent_fd)
        named_parent = os.lstat(path.parent)
        if (
            _identity(parent_info) != _identity(named_parent)
            or not stat.S_ISDIR(parent_info.st_mode)
            or parent_info.st_uid != LOCAL_UID
            or parent_info.st_gid != LOCAL_GID
        ):
            raise ControllerDeployerError(
                "local terminal held parent differs"
            )
        descriptor = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0,
            dir_fd=parent_fd,
        )
        created = os.fstat(descriptor)
        value = _local_terminal_value(
            remote_terminal, path, _inode_anchor(created),
        )
        raw = canonical(value) + b"\n"
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise ControllerDeployerError(
                    "local terminal write made no progress"
                )
            offset += count
        os.fsync(descriptor)
        if _read_fd(descriptor, len(raw)) != raw:
            raise ControllerDeployerError(
                "local terminal preseal replay differs"
            )
        os.fchmod(descriptor, RECEIPT_MODE)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        named = os.stat(
            path.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            _identity(opened) != _identity(named)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != RECEIPT_MODE
            or opened.st_uid != LOCAL_UID
            or opened.st_gid != LOCAL_GID
        ):
            raise ControllerDeployerError("local terminal seal differs")
        os.fsync(parent_fd)
        authority = HeldAuthority(
            path, descriptor, _identity(opened), raw,
            hashlib.sha256(raw).hexdigest(),
        )
        held = HeldCommitTerminal(authority, value)
        held.replay()
        committed = True
        return held
    finally:
        try:
            if descriptor >= 0 and not committed:
                opened = os.fstat(descriptor)
                try:
                    named = os.stat(
                        path.name, dir_fd=parent_fd, follow_symlinks=False,
                    )
                except FileNotFoundError:
                    named = None
                if (
                    named is not None
                    and _inode_anchor(opened) == _inode_anchor(named)
                ):
                    os.unlink(path.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
        finally:
            if descriptor >= 0 and not committed:
                os.close(descriptor)
            os.close(parent_fd)


def _open_local_commit_terminal(path: Path) -> HeldCommitTerminal:
    if not _valid_local_commit_terminal_path(path):
        raise ControllerDeployerError(
            "local commit terminal coordinate differs"
        )
    authority = _open_local_terminal_authority(path)
    try:
        value = _strict_remote_json(
            authority.raw, label="local commit terminal authority",
        )
        validated = _validate_local_terminal_value(
            value, path=path, identity=authority.identity,
        )
        held = HeldCommitTerminal(authority, validated)
        held.replay()
        return held
    except BaseException:
        authority.close()
        raise


def manifest_value() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "target_root": str(REMOTE_TARGET_ROOT),
        "receipt_path": str(REMOTE_RECEIPT_PATH),
        "remote_python": {
            "path": str(REMOTE_PYTHON),
            "sha256": REMOTE_PYTHON_SHA256,
            "size": REMOTE_PYTHON_SIZE,
        },
        "remote_uid": REMOTE_UID,
        "remote_gid": REMOTE_GID,
        "file_count": 1,
        "file_mode": FILE_MODE,
        "directory_mode": DIRECTORY_MODE,
        "receipt_mode": RECEIPT_MODE,
        "directories": ["."],
        "files": [{
            "relative": CONTROLLER_BASENAME,
            "sha256": LOCAL_CONTROLLER_SHA256,
            "size": LOCAL_CONTROLLER_SIZE,
        }],
        "publication_protocol": (
            "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation"
        ),
        "whole_tree_atomically_visible": True,
        "rename_noreplace": False,
        "cooperative_writer_exclusion": True,
        "target_absent_rechecked": True,
        "receipt_is_consumption_gate": True,
        "uncooperative_same_uid_race_out_of_scope": True,
        "launch_allowed": False,
    }
    value["manifest_digest"] = object_digest(value)
    return value


def payload_value(
    controller_raw: bytes, bootstrap_sha256: str, *,
    operation: str = STAGE_OPERATION,
    commit_terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        len(controller_raw) != LOCAL_CONTROLLER_SIZE
        or hashlib.sha256(controller_raw).hexdigest()
        != LOCAL_CONTROLLER_SHA256
        or bootstrap_sha256 != LOCAL_BOOTSTRAP_SHA256
        or operation not in (STAGE_OPERATION, RECOVER_RECEIPT_OPERATION)
        or (operation == STAGE_OPERATION and commit_terminal is not None)
        or (
            operation == RECOVER_RECEIPT_OPERATION
            and type(commit_terminal) is not dict
        )
    ):
        raise ControllerDeployerError("payload source authority differs")
    value: dict[str, Any] = {
        "schema_version": PAYLOAD_SCHEMA,
        "operation": operation,
        "commit_terminal": (
            None if commit_terminal is None else dict(commit_terminal)
        ),
        "manifest": manifest_value(),
        "bootstrap_source_sha256": bootstrap_sha256,
        "files": [{
            "relative": CONTROLLER_BASENAME,
            "sha256": LOCAL_CONTROLLER_SHA256,
            "size": LOCAL_CONTROLLER_SIZE,
            "content_b64": base64.b64encode(controller_raw).decode("ascii"),
        }],
    }
    value["authority_digest"] = object_digest(value)
    return value


def _payload_raw_for_operation(
    payload: Mapping[str, Any], operation: str,
) -> bytes:
    if operation not in (STAGE_OPERATION, RECOVER_RECEIPT_OPERATION):
        raise ControllerDeployerError("payload operation differs")
    value = dict(payload)
    value.pop("authority_digest", None)
    value["operation"] = operation
    if operation == STAGE_OPERATION:
        value["commit_terminal"] = None
    elif type(value.get("commit_terminal")) is not dict:
        raise ControllerDeployerError(
            "receipt-recovery commit terminal is absent"
        )
    value["authority_digest"] = object_digest(value)
    return canonical(value) + b"\n"


REMOTE_LOADER_SOURCE = (
    "import hashlib,sys\n"
    "def exact(n):\n"
    " out=[]\n"
    " while n:\n"
    "  block=sys.stdin.buffer.read(n)\n"
    "  if not block: raise RuntimeError('held frame incomplete')\n"
    "  out.append(block);n-=len(block)\n"
    " return b''.join(out)\n"
    "line=sys.stdin.buffer.readline(32)\n"
    "if not line.endswith(b'\\n') or not line[:-1].isdigit() or line.startswith(b'0'): raise RuntimeError('held length')\n"
    "size=int(line[:-1])\n"
    f"if size>{MAX_BOOTSTRAP_SIZE}: raise RuntimeError('bootstrap bound')\n"
    "source=exact(size)\n"
    f"payload=sys.stdin.buffer.read({MAX_PAYLOAD_SIZE + 1})\n"
    f"if not payload or len(payload)>{MAX_PAYLOAD_SIZE}: raise RuntimeError('payload bound')\n"
    "if hashlib.sha256(source).hexdigest()!=sys.argv[1]: raise RuntimeError('bootstrap SHA')\n"
    "if hashlib.sha256(payload).hexdigest()!=sys.argv[2]: raise RuntimeError('payload SHA')\n"
    "scope={'__name__':'_held_controller_deploy_bootstrap','__file__':'/held/case01/controller-deploy-bootstrap-v2.py'}\n"
    "exec(compile(source,'<held-controller-deploy-bootstrap>','exec'),scope)\n"
    "result=scope['_remote_dispatch'](payload,sys.argv[2],sys.argv[1])\n"
    "sys.stdout.buffer.write(scope['canonical'](result)+b'\\n')\n"
)


def _frame(bootstrap_raw: bytes, payload_raw: bytes) -> bytes:
    if (
        not bootstrap_raw or len(bootstrap_raw) > MAX_BOOTSTRAP_SIZE
        or not payload_raw or len(payload_raw) > MAX_PAYLOAD_SIZE
        or not payload_raw.endswith(b"\n")
    ):
        raise ControllerDeployerError("held frame input differs")
    return (
        str(len(bootstrap_raw)).encode("ascii") + b"\n"
        + bootstrap_raw + payload_raw
    )


def _transport_profile() -> dict[str, Any]:
    return {
        "ssh": {
            "path": str(SSH_PATH), "sha256": SSH_SHA256, "size": SSH_SIZE,
            "uid": 0, "gid": 0, "mode": 0o755, "nlink": 1,
        },
        "identity": {
            "path": str(SSH_IDENTITY), "sha256": SSH_IDENTITY_SHA256,
            "size": SSH_IDENTITY_SIZE, "uid": LOCAL_UID, "gid": LOCAL_GID,
            "mode": 0o600, "nlink": 1,
        },
        "known_hosts": {
            "path": str(SSH_KNOWN_HOSTS), "sha256": SSH_KNOWN_HOSTS_SHA256,
            "size": SSH_KNOWN_HOSTS_SIZE, "uid": LOCAL_UID,
            "gid": LOCAL_GID, "mode": 0o600, "nlink": 1,
        },
        "destination": SSH_DESTINATION,
        "configuration": [
            "-F=/dev/null", "BatchMode=yes", "IdentitiesOnly=yes",
            f"IdentityFile={SSH_IDENTITY}",
            f"UserKnownHostsFile={SSH_KNOWN_HOSTS}",
            "GlobalKnownHostsFile=/dev/null", "StrictHostKeyChecking=yes",
            "UpdateHostKeys=no", "ClearAllForwardings=yes", "RequestTTY=no",
            "KbdInteractiveAuthentication=no", "PasswordAuthentication=no",
            "PreferredAuthentications=publickey", "PubkeyAuthentication=yes",
            "GSSAPIAuthentication=no", "HostbasedAuthentication=no",
            "CanonicalizeHostname=no", "CheckHostIP=no",
            "VerifyHostKeyDNS=no", "ControlMaster=no", "ControlPath=none",
            "ControlPersist=no", "ForwardAgent=no", "ForwardX11=no",
            "ExitOnForwardFailure=yes", "ConnectTimeout=30",
            "ConnectionAttempts=1", "ServerAliveInterval=15",
            "ServerAliveCountMax=2", "LogLevel=ERROR", "port=22",
        ],
        "named_path_authority": {
            "ssh": "restricted_root_inode_on_read_only_system_volume",
            "credentials": "held_exact_inode_with_owner_controlled_parents",
            "held_replays": [
                "after_open", "pre_spawn", "immediate_post_spawn",
                "post_reap",
            ],
            "same_uid_root_kernel_mount_attacker_out_of_scope": True,
            "residual_named_lookup_window_absence_claimed": False,
        },
        "close_fds": True,
        "pass_fds": [],
    }


def authorization_token() -> str:
    authority = {
        "schema_version": SCHEMA + "-authorization-v1",
        "manifest": manifest_value(),
        "controller_sha256": LOCAL_CONTROLLER_SHA256,
        "controller_size": LOCAL_CONTROLLER_SIZE,
        "bootstrap_sha256": LOCAL_BOOTSTRAP_SHA256,
        "bootstrap_size": LOCAL_BOOTSTRAP_SIZE,
        "remote_loader_sha256": hashlib.sha256(
            REMOTE_LOADER_SOURCE.encode("utf-8")
        ).hexdigest(),
        "local_commit_terminal": {
            "schema_version": LOCAL_TERMINAL_SCHEMA,
            "initial_path": str(LOCAL_COMMIT_TERMINAL_PATH),
            "mode": RECEIPT_MODE,
            "create_only": True,
            "refreshed_paths_bind_remote_terminal_digest": True,
        },
        "explicit_recover_receipt_cli": True,
        "automatic_remote_retry": False,
        "transport_profile": _transport_profile(),
        "remote_entry": [str(REMOTE_PYTHON), "-I", "-S", "-B", "-c"],
        "single_attempt": True,
        "retry_allowed": False,
        "launch_allowed": False,
        "slurm_allowed": False,
    }
    return object_digest(authority)


def _open_sources() -> tuple[HeldAuthority, HeldAuthority]:
    controller = _open_authority(
        LOCAL_CONTROLLER,
        sha256=LOCAL_CONTROLLER_SHA256, size=LOCAL_CONTROLLER_SIZE,
        uid=LOCAL_UID, gid=LOCAL_GID, mode=0o644,
    )
    try:
        bootstrap = _open_authority(
            LOCAL_BOOTSTRAP,
            sha256=LOCAL_BOOTSTRAP_SHA256, size=LOCAL_BOOTSTRAP_SIZE,
            uid=LOCAL_UID, gid=LOCAL_GID, mode=0o644,
        )
    except BaseException:
        controller.close()
        raise
    return controller, bootstrap


def _open_transport_authorities() -> list[HeldAuthority]:
    authorities: list[HeldAuthority] = []
    specs = (
        (SSH_PATH, SSH_SHA256, SSH_SIZE, 0, 0, 0o755),
        (
            SSH_IDENTITY, SSH_IDENTITY_SHA256, SSH_IDENTITY_SIZE,
            LOCAL_UID, LOCAL_GID, 0o600,
        ),
        (
            SSH_KNOWN_HOSTS, SSH_KNOWN_HOSTS_SHA256, SSH_KNOWN_HOSTS_SIZE,
            LOCAL_UID, LOCAL_GID, 0o600,
        ),
    )
    try:
        for path, digest, size, uid, gid, mode in specs:
            authorities.append(_open_authority(
                path, sha256=digest, size=size, uid=uid, gid=gid,
                mode=mode,
            ))
        return authorities
    except BaseException as primary_error:
        try:
            _close_transport_authorities(authorities)
        except BaseException as cleanup_error:
            raise ControllerDeployerError(
                "partial SSH transport authority close differs",
            ) from cleanup_error
        raise primary_error


def _close_transport_authorities(
    authorities: Sequence[HeldAuthority],
) -> None:
    errors: list[BaseException] = []
    for authority in authorities:
        try:
            authority.close()
        except BaseException as error:
            errors.append(error)
    if errors:
        raise ControllerDeployerError(
            "SSH transport authority close differs",
        ) from errors[0]


# Compatibility name retained for existing local-only tests and reviewers.
def _open_transport() -> list[HeldAuthority]:
    return _open_transport_authorities()


def _ssh_arguments(
    remote_command: str, transport: Sequence[HeldAuthority],
) -> list[str]:
    if (
        len(transport) != 3
        or any(authority.descriptor < 3 for authority in transport)
        or tuple(authority.path for authority in transport)
        != (SSH_PATH, SSH_IDENTITY, SSH_KNOWN_HOSTS)
        or SSH_PATH != Path("/usr/bin/ssh")
    ):
        raise ControllerDeployerError("held SSH transport closure differs")
    return [
        str(SSH_PATH), "-F", "/dev/null", "-T",
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", f"IdentityFile={SSH_IDENTITY}",
        "-o", f"UserKnownHostsFile={SSH_KNOWN_HOSTS}",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no",
        "-o", "ClearAllForwardings=yes", "-o", "RequestTTY=no",
        "-o", "KbdInteractiveAuthentication=no",
        "-o", "PasswordAuthentication=no",
        "-o", "PreferredAuthentications=publickey",
        "-o", "PubkeyAuthentication=yes", "-o", "GSSAPIAuthentication=no",
        "-o", "HostbasedAuthentication=no",
        "-o", "CanonicalizeHostname=no", "-o", "CheckHostIP=no",
        "-o", "VerifyHostKeyDNS=no", "-o", "ControlMaster=no",
        "-o", "ControlPath=none", "-o", "ControlPersist=no",
        "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
        "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=30",
        "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=2", "-o", "LogLevel=ERROR",
        "-p", "22", SSH_DESTINATION, remote_command,
    ]


def _validate_code_directory(
    raw: bytes,
    absolute: int,
    child_size: int,
    expected: tuple[int, int, int],
) -> None:
    """Validate the bounded SHA-256 CodeDirectory layout, not its trust chain."""
    if absolute < 0 or child_size < 88 or absolute + child_size > len(raw):
        raise ControllerDeployerError("held SSH CodeDirectory range differs")
    (
        magic, length, version, flags, hash_offset, identifier_offset,
        special_slots, code_slots, code_limit,
    ) = struct.unpack_from(">IIIIIIIII", raw, absolute)
    hash_size, hash_type, platform, page_exponent = struct.unpack_from(
        ">BBBB", raw, absolute + 36,
    )
    spare2, scatter_offset, team_offset, spare3 = struct.unpack_from(
        ">IIII", raw, absolute + 40,
    )
    code_limit64, exec_base, exec_limit, exec_flags = struct.unpack_from(
        ">QQQQ", raw, absolute + 56,
    )
    expected_code_slots, expected_code_limit, expected_length = expected
    identifier = b"com.apple.ssh\x00"
    identifier_end = identifier_offset + len(identifier)
    special_start = hash_offset - special_slots * hash_size
    code_hash_end = hash_offset + code_slots * hash_size
    if (
        magic != 0xFADE0C02
        or length != child_size
        or length != expected_length
        or version != 0x0002_0400
        or flags != 0
        or identifier_offset != 88
        or special_slots != 7
        or code_slots != expected_code_slots
        or code_limit != expected_code_limit
        or hash_size != 32
        or hash_type != 2
        or platform != 15
        or page_exponent != 12
        or any((spare2, scatter_offset, team_offset, spare3, code_limit64))
        or exec_base != 0
        or exec_limit <= 0
        or exec_limit > code_limit
        or exec_flags != 1
        or identifier_end != special_start
        or code_hash_end != length
        or code_slots != (code_limit + (1 << page_exponent) - 1)
        // (1 << page_exponent)
        or raw[
            absolute + identifier_offset:absolute + identifier_end
        ] != identifier
    ):
        raise ControllerDeployerError("held SSH CodeDirectory closure differs")


def _validate_embedded_signature(
    raw: bytes,
    absolute: int,
    data_size: int,
    expected_code_directory: tuple[int, int, int],
) -> None:
    """Validate the bounded embedded SuperBlob and its contiguous children."""
    if absolute < 0 or data_size < 12 or absolute + data_size > len(raw):
        raise ControllerDeployerError("held SSH SuperBlob range differs")
    magic, super_length, child_count = struct.unpack_from(">III", raw, absolute)
    index_end = 12 + 8 * child_count
    if (
        magic != 0xFADE0CC0
        or child_count != 5
        or index_end > super_length
        or super_length > data_size
    ):
        raise ControllerDeployerError("held SSH SuperBlob header differs")
    indexed: list[tuple[int, int]] = []
    for index in range(child_count):
        slot, relative = struct.unpack_from(
            ">II", raw, absolute + 12 + 8 * index,
        )
        indexed.append((slot, relative))
    if (
        tuple(slot for slot, _relative in indexed) != (0, 2, 5, 7, 0x10000)
        or len({slot for slot, _relative in indexed}) != child_count
        or len({relative for _slot, relative in indexed}) != child_count
    ):
        raise ControllerDeployerError("held SSH SuperBlob index differs")
    expected_child_magic = {
        0: 0xFADE0C02,
        2: 0xFADE0C01,
        5: 0xFADE7171,
        7: 0xFADE7172,
        0x10000: 0xFADE0B01,
    }
    cursor = index_end
    for slot, relative in sorted(indexed, key=lambda row: row[1]):
        if relative != cursor or relative + 8 > super_length:
            raise ControllerDeployerError(
                "held SSH SuperBlob child offset differs",
            )
        child_magic, child_size = struct.unpack_from(
            ">II", raw, absolute + relative,
        )
        if (
            child_magic != expected_child_magic[slot]
            or child_size < 8
            or relative + child_size > super_length
        ):
            raise ControllerDeployerError("held SSH SuperBlob child differs")
        if slot == 0:
            _validate_code_directory(
                raw, absolute + relative, child_size,
                expected_code_directory,
            )
        cursor = relative + child_size
    if (
        cursor != super_length
        or any(raw[absolute + super_length:absolute + data_size])
    ):
        raise ControllerDeployerError("held SSH SuperBlob padding differs")


def _macho_code_signature_ranges(raw: bytes) -> tuple[tuple[int, int], ...]:
    """Validate the exact fat/slice/signature structure of pinned system SSH."""
    if len(raw) != SSH_SIZE or raw[:4] != b"\xca\xfe\xba\xbe":
        raise ControllerDeployerError(
            "held SSH is not the pinned fat Mach-O format",
        )
    architecture_count = struct.unpack_from(">I", raw, 4)[0]
    if architecture_count != len(SSH_FAT_ARCHITECTURES):
        raise ControllerDeployerError(
            "held SSH Mach-O architecture closure differs",
        )
    architectures = tuple(
        struct.unpack_from(">IIIII", raw, 8 + 20 * index)
        for index in range(architecture_count)
    )
    if architectures != SSH_FAT_ARCHITECTURES:
        raise ControllerDeployerError(
            "held SSH Mach-O architecture rows differ",
        )
    ordered_slices = sorted(
        (offset, size) for _cpu, _subtype, offset, size, _align in architectures
    )
    for index, (offset, size) in enumerate(ordered_slices):
        if (
            offset <= 0
            or size <= 32
            or offset + size > len(raw)
            or (
                index
                and offset
                < ordered_slices[index - 1][0] + ordered_slices[index - 1][1]
            )
        ):
            raise ControllerDeployerError("held SSH Mach-O slice range differs")
    signatures: list[tuple[int, int]] = []
    for architecture_index, architecture in enumerate(architectures):
        cpu, subtype, slice_offset, slice_size, alignment = architecture
        if (
            slice_offset % (1 << alignment) != 0
            or raw[slice_offset:slice_offset + 4] != b"\xcf\xfa\xed\xfe"
        ):
            raise ControllerDeployerError(
                "held SSH Mach-O slice alignment differs",
            )
        (
            magic, header_cpu, header_subtype, file_type, command_count,
            command_bytes, _flags, reserved,
        ) = struct.unpack_from("<IIIIIIII", raw, slice_offset)
        command_start = slice_offset + 32
        command_end = command_start + command_bytes
        if (
            magic != 0xFEEDFACF
            or header_cpu != cpu
            or header_subtype != subtype
            or file_type != 2
            or command_count <= 0
            or reserved != 0
            or command_end > slice_offset + slice_size
        ):
            raise ControllerDeployerError("held SSH Mach-O header differs")
        cursor = command_start
        slice_signatures: list[tuple[int, int, int]] = []
        for _index in range(command_count):
            if cursor + 8 > command_end:
                raise ControllerDeployerError(
                    "held SSH Mach-O command header differs",
                )
            command, command_size = struct.unpack_from("<II", raw, cursor)
            if (
                command_size < 8
                or command_size % 8 != 0
                or cursor + command_size > command_end
            ):
                raise ControllerDeployerError(
                    "held SSH Mach-O command size differs",
                )
            if command == 0x1D:
                if command_size != 16:
                    raise ControllerDeployerError(
                        "held SSH code-sign command differs",
                    )
                data_offset, data_size = struct.unpack_from(
                    "<II", raw, cursor + 8,
                )
                absolute = slice_offset + data_offset
                if (
                    data_size <= 0
                    or absolute < command_end
                    or data_offset + data_size != slice_size
                ):
                    raise ControllerDeployerError(
                        "held SSH code-sign range differs",
                    )
                slice_signatures.append((absolute, data_size, data_offset))
            cursor += command_size
        if cursor != command_end or len(slice_signatures) != 1:
            raise ControllerDeployerError(
                "held SSH embedded code-sign closure differs",
            )
        absolute, data_size, data_offset = slice_signatures[0]
        expected_directory = SSH_CODE_DIRECTORIES[architecture_index]
        if data_offset != expected_directory[1]:
            raise ControllerDeployerError("held SSH signed code-limit differs")
        _validate_embedded_signature(
            raw, absolute, data_size, expected_directory,
        )
        signatures.append((absolute, data_size))
    return tuple(signatures)


def _validate_read_only_filesystem(
    descriptor: int, path: Path, expected_device: int,
) -> None:
    opened = os.fstat(descriptor)
    held_fs = os.fstatvfs(descriptor)
    named_fs = os.statvfs(path)
    read_only = getattr(os, "ST_RDONLY", 1)
    if (
        opened.st_dev != expected_device
        or getattr(held_fs, "f_fsid", None) is None
        or held_fs.f_fsid != named_fs.f_fsid
        or not (held_fs.f_flag & read_only)
        or not (named_fs.f_flag & read_only)
    ):
        raise ControllerDeployerError(
            "named system SSH filesystem authority differs",
        )


def _validate_system_parent_chain(expected_device: int) -> None:
    for path in (Path("/"), Path("/usr"), Path("/usr/bin")):
        before = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        try:
            opened = os.fstat(descriptor)
            named = os.lstat(path)
            if (
                _identity(before) != _identity(opened)
                or _identity(opened) != _identity(named)
                or not stat.S_ISDIR(opened.st_mode)
                or opened.st_uid != 0
                or opened.st_gid != 0
                or stat.S_IMODE(opened.st_mode) != 0o755
                or opened.st_mode & 0o022
            ):
                raise ControllerDeployerError(
                    "named system SSH parent authority differs",
                )
            _validate_read_only_filesystem(descriptor, path, expected_device)
        finally:
            os.close(descriptor)


def _validate_system_ssh_authority(authority: HeldAuthority) -> None:
    authority.replay()
    opened = os.fstat(authority.descriptor)
    named = os.lstat(SSH_PATH)
    opened_flags = int(getattr(opened, "st_flags", 0))
    named_flags = int(getattr(named, "st_flags", 0))
    signatures = _macho_code_signature_ranges(authority.raw)
    if (
        sys.platform != "darwin"
        or authority.path != SSH_PATH
        or SSH_PATH != Path("/usr/bin/ssh")
        or authority.sha256 != SSH_SHA256
        or len(authority.raw) != SSH_SIZE
        or hashlib.sha256(authority.raw).hexdigest() != SSH_SHA256
        or _identity(opened) != authority.identity
        or _identity(named) != authority.identity
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) != 0o755
        or opened.st_uid != 0
        or opened.st_gid != 0
        or opened_flags != named_flags
        or not (opened_flags & DARWIN_SF_RESTRICTED)
        or len(signatures) != len(SSH_FAT_ARCHITECTURES)
    ):
        raise ControllerDeployerError("named system SSH authority differs")
    _validate_read_only_filesystem(
        authority.descriptor, SSH_PATH, opened.st_dev,
    )
    _validate_system_parent_chain(opened.st_dev)


def _validate_credential_parent(path: Path, expected_device: int) -> None:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.resolve(strict=True) != path
    ):
        raise ControllerDeployerError(
            "named SSH credential parent path differs",
        )
    before = os.lstat(path)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(path)
        if (
            _identity(before) != _identity(opened)
            or _identity(opened) != _identity(named)
            or not stat.S_ISDIR(opened.st_mode)
            or opened.st_dev != expected_device
            or opened.st_uid != LOCAL_UID
            or opened.st_gid != LOCAL_GID
            or stat.S_IMODE(opened.st_mode) != 0o755
            or opened.st_mode & 0o022
        ):
            raise ControllerDeployerError(
                "named SSH credential parent differs",
            )
    finally:
        os.close(descriptor)


def _validate_named_credential_authority(
    authority: HeldAuthority,
    expected_path: Path,
    expected_sha256: str,
    expected_size: int,
) -> int:
    authority.replay()
    opened = os.fstat(authority.descriptor)
    named = os.lstat(expected_path)
    if (
        not expected_path.is_absolute()
        or os.path.normpath(str(expected_path)) != str(expected_path)
        or authority.path != expected_path
        or expected_path.resolve(strict=True) != expected_path
        or authority.sha256 != expected_sha256
        or len(authority.raw) != expected_size
        or hashlib.sha256(authority.raw).hexdigest() != expected_sha256
        or _identity(opened) != authority.identity
        or _identity(named) != authority.identity
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_uid != LOCAL_UID
        or opened.st_gid != LOCAL_GID
    ):
        raise ControllerDeployerError(
            "named SSH credential authority differs",
        )
    return int(opened.st_dev)


def _validate_named_transport_authorities(
    transport: Sequence[HeldAuthority],
) -> None:
    if (
        len(transport) != 3
        or tuple(authority.path for authority in transport)
        != (SSH_PATH, SSH_IDENTITY, SSH_KNOWN_HOSTS)
    ):
        raise ControllerDeployerError(
            "named SSH transport authority set differs",
        )
    _validate_system_ssh_authority(transport[0])
    identity_device = _validate_named_credential_authority(
        transport[1], SSH_IDENTITY, SSH_IDENTITY_SHA256, SSH_IDENTITY_SIZE,
    )
    hosts_device = _validate_named_credential_authority(
        transport[2], SSH_KNOWN_HOSTS,
        SSH_KNOWN_HOSTS_SHA256, SSH_KNOWN_HOSTS_SIZE,
    )
    if identity_device != hosts_device:
        raise ControllerDeployerError(
            "named SSH credential device closure differs",
        )
    parents = (SSH_KNOWN_HOSTS.parent, SSH_IDENTITY.parent)
    if parents != (
        Path("/Users/siriuschu/.ssh"),
        Path("/Users/siriuschu/.ssh/ciai2"),
    ):
        raise ControllerDeployerError(
            "named SSH credential parent set differs",
        )
    for parent in parents:
        _validate_credential_parent(parent, identity_device)


def _close_process_pipes(
    process: subprocess.Popen[bytes],
) -> None:
    errors: list[BaseException] = []
    for pipe in (process.stdout, process.stderr):
        if pipe is not None and not pipe.closed:
            try:
                pipe.close()
            except BaseException as error:
                errors.append(error)
    if errors:
        raise ControllerDeployerError(
            "SSH terminal pipe close differs",
        ) from errors[0]


def _process_group_present(group: int) -> bool:
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        if error.errno == errno.EPERM:
            return True
        raise ControllerDeployerError("SSH process-group probe differs") from error


def _signal_group(group: int, number: int) -> None:
    try:
        os.killpg(group, number)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    except OSError as error:
        if error.errno in (errno.ESRCH, errno.EPERM):
            return
        raise ControllerDeployerError("SSH process-group signal differs") from error


def _poll_group_absent(
    process: subprocess.Popen[bytes], group: int, deadline: float,
) -> bool:
    while True:
        process.poll()
        if not _process_group_present(group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(PROCESS_POLL_SECONDS)


def _seal_process_group(
    process: subprocess.Popen[bytes], group: int,
) -> None:
    pipe_error: BaseException | None = None
    try:
        _close_process_pipes(process)
    except BaseException as error:
        pipe_error = error
    if _process_group_present(group):
        _signal_group(group, signal.SIGTERM)
        _poll_group_absent(
            process, group, time.monotonic() + PROCESS_TERM_GRACE_SECONDS,
        )
    if _process_group_present(group):
        _signal_group(group, signal.SIGKILL)
    try:
        process.wait(timeout=PROCESS_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired as error:
        try:
            process.kill()
        finally:
            try:
                process.wait(timeout=PROCESS_KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired as final_error:
                raise ControllerDeployerError(
                    "SSH direct child was not reaped",
                ) from final_error
        if process.poll() is None:
            raise ControllerDeployerError(
                "SSH direct child reap differs",
            ) from error
    if not _poll_group_absent(
        process, group, time.monotonic() + PROCESS_KILL_GRACE_SECONDS,
    ):
        raise ControllerDeployerError("SSH process group did not reach ESRCH")
    if process.poll() is None:
        raise ControllerDeployerError("SSH direct child remains unreaped")
    if pipe_error is not None:
        raise ControllerDeployerError(
            "SSH terminal pipe seal differs",
        ) from pipe_error


def _seal_unverified_process(process: subprocess.Popen[bytes]) -> None:
    try:
        _close_process_pipes(process)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=PROCESS_TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=PROCESS_KILL_GRACE_SECONDS)
    if process.poll() is None:
        raise ControllerDeployerError(
            "unverified SSH direct child was not reaped",
        )


# Compatibility name retained for focused process-group tests.
def _seal_process(
    process: subprocess.Popen[bytes], group: int | None,
) -> None:
    if group is None:
        _seal_unverified_process(process)
    else:
        _seal_process_group(process, group)


def _bounded_stream_diagnostic(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise ControllerDeployerError("SSH diagnostic stream type differs")
    prefix = raw[:TRANSPORT_DIAGNOSTIC_PREFIX_LIMIT]
    return {
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "newline_count": raw.count(b"\n"),
        "prefix_size": len(prefix),
        "prefix_b64": base64.b64encode(prefix).decode("ascii"),
        "truncated": len(raw) > len(prefix),
    }


def _transport_terminal_diagnostic(
    *, reason: str, returncode: int | None,
    stdout: bytes, stderr: bytes, streams_complete: bool,
) -> dict[str, Any]:
    if (
        reason not in ("timeout_partial", "terminal_contract")
        or (returncode is not None and type(returncode) is not int)
        or type(streams_complete) is not bool
    ):
        raise ControllerDeployerError(
            "SSH terminal diagnostic authority differs",
        )
    value: dict[str, Any] = {
        "schema_version": TRANSPORT_DIAGNOSTIC_SCHEMA,
        "reason": reason,
        "returncode": returncode,
        "streams_complete": streams_complete,
        "prefix_limit": TRANSPORT_DIAGNOSTIC_PREFIX_LIMIT,
        "stdout": _bounded_stream_diagnostic(stdout),
        "stderr": _bounded_stream_diagnostic(stderr),
        "remote_output_is_untrusted": True,
        "prefix_may_contain_remote_echo_of_input": True,
        "diagnostic_is_exact_bounded_remote_output_prefix": True,
    }
    value["diagnostic_digest"] = object_digest(value)
    return value


def _execute_remote(
    held_input: Any, *, bootstrap_sha256: str, payload_sha256: str,
) -> bytes:
    transport: list[HeldAuthority] = []
    environment = {
        "PATH": "/usr/bin:/bin", "HOME": "/var/empty",
        "LANG": "C", "LC_ALL": "C",
    }
    process: subprocess.Popen[bytes] | None = None
    group: int | None = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    stdout = b""
    stderr = b""
    try:
        transport = _open_transport_authorities()
        _validate_named_transport_authorities(transport)
        remote_command = shlex.join([
            "/usr/bin/env", "-i", str(REMOTE_PYTHON),
            "-I", "-S", "-B", "-c", REMOTE_LOADER_SOURCE,
            bootstrap_sha256, payload_sha256,
        ])
        command = _ssh_arguments(remote_command, transport)
        _validate_named_transport_authorities(transport)
        held_input.seek(0)
        process = subprocess.Popen(
            command, stdin=held_input, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=environment,
            start_new_session=True, close_fds=True,
            pass_fds=(),
        )
        # start_new_session=True makes the child's pid its process-group id
        # before exec.  Retain that deterministic PGID even if a very fast
        # leader exits before the parent can observe getpgid(); descendants
        # are still sealed by TERM -> KILL -> ESRCH on this exact group.
        group = process.pid
        _validate_named_transport_authorities(transport)
        try:
            observed = os.getpgid(process.pid)
        except (OSError, ProcessLookupError) as error:
            raise ControllerDeployerError(
                "SSH process-group creation was not observable",
            ) from error
        if observed != process.pid:
            raise ControllerDeployerError(
                "SSH start_new_session process group differs",
            )
        try:
            stdout, stderr = process.communicate(
                timeout=TRANSPORT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            partial_stdout = error.output if type(error.output) is bytes else b""
            partial_stderr = error.stderr if type(error.stderr) is bytes else b""
            diagnostic = _transport_terminal_diagnostic(
                reason="timeout_partial",
                returncode=process.returncode,
                stdout=partial_stdout,
                stderr=partial_stderr,
                streams_complete=False,
            )
            raise ControllerDeployerError(
                "single SSH controller deployment attempt timed out: "
                + canonical(diagnostic).decode("utf-8"),
            ) from error
        if process.returncode != 0 or stderr or stdout.count(b"\n") != 1:
            diagnostic = _transport_terminal_diagnostic(
                reason="terminal_contract",
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
                streams_complete=True,
            )
            raise ControllerDeployerError(
                "single SSH controller deployment terminal streams differ: "
                + canonical(diagnostic).decode("utf-8"),
            )
    except BaseException as error:
        primary_error = error
    finally:
        if process is not None:
            try:
                if group is None:
                    _seal_unverified_process(process)
                else:
                    _seal_process_group(process, group)
            except BaseException as error:
                cleanup_error = error
        try:
            if transport:
                _validate_named_transport_authorities(transport)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        try:
            _close_transport_authorities(transport)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
    if cleanup_error is not None:
        raise ControllerDeployerError(
            "SSH process/transport zero gate differs",
        ) from (primary_error if primary_error is not None else cleanup_error)
    if primary_error is not None:
        raise primary_error
    return stdout


def _strict_remote_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ControllerDeployerError(f"duplicate key in {label}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise ControllerDeployerError(f"remote {label} is not JSON") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise ControllerDeployerError(f"remote {label} is not canonical")
    return value


def _validated_target_identity(value: Any) -> tuple[int, ...]:
    if (
        type(value) is not list
        or len(value) != 11
        or any(type(item) is not int for item in value)
    ):
        raise ControllerDeployerError("remote target identity shape differs")
    identity = tuple(value)
    if (
        identity[0] <= 0
        or identity[1] <= 0
        or identity[2] != REMOTE_UID
        or identity[3] != REMOTE_GID
        or not stat.S_ISDIR(identity[4])
        or stat.S_IMODE(identity[4]) != DIRECTORY_MODE
        or identity[5] < 2
        or identity[6] != 0
        or any(item < 0 for item in identity[7:])
    ):
        raise ControllerDeployerError("remote target identity value differs")
    return identity


def _validated_receipt_anchor(value: Any) -> tuple[int, ...]:
    if (
        type(value) is not list
        or len(value) != 5
        or any(type(item) is not int for item in value)
    ):
        raise ControllerDeployerError("remote receipt inode anchor differs")
    anchor = tuple(value)
    if (
        anchor[0] <= 0
        or anchor[1] <= 0
        or anchor[2] != REMOTE_UID
        or anchor[3] != REMOTE_GID
        or anchor[4] != stat.S_IFREG
    ):
        raise ControllerDeployerError("remote receipt anchor value differs")
    return anchor


def _validated_receipt_full_identity(value: Any) -> tuple[int, ...]:
    if (
        type(value) is not list
        or len(value) != 11
        or any(type(item) is not int for item in value)
    ):
        raise ControllerDeployerError(
            "remote receipt reservation identity shape differs"
        )
    identity = tuple(value)
    if (
        identity[0] <= 0
        or identity[1] <= 0
        or identity[2] != REMOTE_UID
        or identity[3] != REMOTE_GID
        or not stat.S_ISREG(identity[4])
        or stat.S_IMODE(identity[4])
        not in (RECEIPT_RESERVATION_MODE, RECEIPT_MODE)
        or identity[5] != 1
        or identity[6] != 0
        or identity[7] < 0
        or any(item < 0 for item in identity[8:])
    ):
        raise ControllerDeployerError(
            "remote receipt reservation identity value differs"
        )
    return identity


def _validated_receipt_reservation_state(value: Any) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("available")) is not bool:
        raise ControllerDeployerError(
            "remote receipt reservation state differs"
        )
    if value["available"] is False:
        if set(value) != {"available"}:
            raise ControllerDeployerError(
                "unavailable remote receipt state closure differs"
            )
        return {"available": False}
    if set(value) != {
        "available", "inode_anchor", "identity", "mode", "size", "sha256",
    }:
        raise ControllerDeployerError(
            "remote receipt reservation state closure differs"
        )
    identity = _validated_receipt_full_identity(value["identity"])
    anchor = value["inode_anchor"]
    if (
        type(anchor) is not list
        or len(anchor) != 5
        or any(type(item) is not int for item in anchor)
        or tuple(anchor) != _inode_anchor_from_identity(identity)
        or value["mode"] != stat.S_IMODE(identity[4])
        or value["mode"]
        not in (RECEIPT_RESERVATION_MODE, RECEIPT_MODE)
        or type(value["size"]) is not int
        or value["size"] != identity[7]
        or not 0 <= value["size"] <= MAX_RECEIPT_SIZE
        or type(value["sha256"]) is not str
        or SHA_RE.fullmatch(value["sha256"]) is None
    ):
        raise ControllerDeployerError(
            "remote receipt reservation state value differs"
        )
    return dict(value)


def _expected_commit_terminal(
    *, stage_payload_sha256: str, bootstrap_sha256: str,
    target_identity: tuple[int, ...],
    receipt_reservation_state: Mapping[str, Any],
    rename_result: str, rename_classification: str,
    receipt_phase: str, receipt_authoritative: bool,
    named_target_same_held_inode: bool, recovery_admissible: bool,
) -> dict[str, Any]:
    manifest = manifest_value()
    value: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "status": "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
        "operation": STAGE_OPERATION,
        "target_root": str(REMOTE_TARGET_ROOT),
        "receipt_path": str(REMOTE_RECEIPT_PATH),
        "manifest_digest": manifest["manifest_digest"],
        "request_payload_sha256": stage_payload_sha256,
        "stage_payload_sha256": stage_payload_sha256,
        "bootstrap_source_sha256": bootstrap_sha256,
        "target_root_identity": list(target_identity),
        "target_rename_commit_point_crossed": True,
        "rename_attempted": True,
        "rename_attempt_count": 1,
        "rename_result": rename_result,
        "rename_classification": rename_classification,
        "whole_tree_atomically_visible": True,
        "rename_noreplace": False,
        "target_absent_rechecked": True,
        "receipt_is_consumption_gate": True,
        "receipt_reservation_state": dict(receipt_reservation_state),
        "receipt_phase": receipt_phase,
        "receipt_authoritative": receipt_authoritative,
        "recovery_admissible": recovery_admissible,
        "named_target_same_held_inode": named_target_same_held_inode,
        "zero_publication_claim": False,
        "uncooperative_same_uid_race_out_of_scope": True,
        "launch_allowed": False,
    }
    value["terminal_digest"] = object_digest(value)
    return value


def _validate_remote_receipt(
    value: Mapping[str, Any], *, request_payload_sha256: str,
    stage_payload_sha256: str, bootstrap_sha256: str,
    commit_terminal_digest: str | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ControllerDeployerError("remote receipt type differs")
    expected_keys = {
        "schema_version", "status", "operation", "target_root",
        "receipt_path", "manifest_digest", "request_payload_sha256",
        "stage_payload_sha256", "bootstrap_source_sha256", "file_count",
        "files", "directories", "file_mode", "directory_mode",
        "receipt_mode", "held_parent_identity_replayed",
        "ancestor_chain_nofollow", "publication_protocol",
        "rename_noreplace", "cooperative_writer_exclusion",
        "receipt_is_consumption_gate", "receipt_is_admission",
        "uncooperative_same_uid_race_out_of_scope", "target_observation",
        "commit_terminal_digest", "receipt_inode_anchor", "launch_allowed",
        "receipt_digest",
    }
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    operation = value.get("operation")
    if operation == STAGE_OPERATION:
        expected_status = "STAGED_RECEIPT_GATED"
        expected_kind = "live_posix_rename_under_held_receipt_reservation"
        renamed = True
        absent_rechecked = True
    elif operation == RECOVER_RECEIPT_OPERATION:
        expected_status = "RECOVERED_RECEIPT_ONLY"
        expected_kind = "recovered_existing_exact_controller_current_inode"
        renamed = False
        absent_rechecked = False
    else:
        raise ControllerDeployerError("remote receipt operation differs")
    observation = value.get("target_observation")
    if type(observation) is not dict:
        raise ControllerDeployerError("remote target observation differs")
    target_identity = _validated_target_identity(
        observation.get("root_identity"),
    )
    expected_observation = {
        "kind": expected_kind,
        "root_identity": list(target_identity),
        "held_inode_continuity": True,
        "ordinary_posix_rename_performed_this_operation": renamed,
        "rename_noreplace_performed_this_operation": False,
        "target_absent_rechecked_before_rename": absent_rechecked,
        "whole_tree_atomically_visible": True,
        "historical_replacement_claim": "not_made",
    }
    _validated_receipt_anchor(value.get("receipt_inode_anchor"))
    expected_files = [{
        "relative": CONTROLLER_BASENAME,
        "sha256": LOCAL_CONTROLLER_SHA256,
        "size": LOCAL_CONTROLLER_SIZE,
        "mode": FILE_MODE,
        "nlink": 1,
    }]
    if (
        set(value) != expected_keys
        or claimed != object_digest(unsigned)
        or value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("status") != expected_status
        or value.get("target_root") != str(REMOTE_TARGET_ROOT)
        or value.get("receipt_path") != str(REMOTE_RECEIPT_PATH)
        or value.get("manifest_digest") != manifest_value()["manifest_digest"]
        or value.get("request_payload_sha256") != request_payload_sha256
        or value.get("stage_payload_sha256") != stage_payload_sha256
        or value.get("bootstrap_source_sha256") != bootstrap_sha256
        or value.get("file_count") != 1
        or value.get("files") != expected_files
        or value.get("directories") != ["."]
        or value.get("file_mode") != FILE_MODE
        or value.get("directory_mode") != DIRECTORY_MODE
        or value.get("receipt_mode") != RECEIPT_MODE
        or value.get("held_parent_identity_replayed") is not True
        or value.get("ancestor_chain_nofollow") is not True
        or value.get("publication_protocol")
        != "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation"
        or value.get("rename_noreplace") is not False
        or value.get("cooperative_writer_exclusion") is not True
        or value.get("receipt_is_consumption_gate") is not True
        or value.get("receipt_is_admission") is not True
        or value.get("uncooperative_same_uid_race_out_of_scope") is not True
        or observation != expected_observation
        or value.get("commit_terminal_digest") != commit_terminal_digest
        or value.get("launch_allowed") is not False
    ):
        raise ControllerDeployerError("remote final receipt authority differs")
    return dict(value)


def _validate_remote_result(
    raw: bytes, *, operation: str, request_payload_sha256: str,
    stage_payload_sha256: str, bootstrap_sha256: str,
    commit_terminal: Mapping[str, Any] | None,
) -> dict[str, Any]:
    value = _strict_remote_json(raw, label="deployment result")
    if value.get("schema_version") == RECEIPT_SCHEMA:
        receipt_operation = value.get("operation")
        if operation == STAGE_OPERATION:
            allowed = (STAGE_OPERATION,)
        elif operation == RECOVER_RECEIPT_OPERATION:
            allowed = (STAGE_OPERATION, RECOVER_RECEIPT_OPERATION)
        else:
            raise ControllerDeployerError(
                "remote result request operation differs"
            )
        if receipt_operation not in allowed:
            raise ControllerDeployerError(
                "remote receipt operation/request differs"
            )
        if receipt_operation == STAGE_OPERATION:
            expected_request = stage_payload_sha256
            expected_terminal_digest = None
        else:
            expected_request = request_payload_sha256
            if type(commit_terminal) is not dict:
                raise ControllerDeployerError(
                    "recovered receipt lacks commit terminal authority"
                )
            expected_terminal_digest = commit_terminal.get("terminal_digest")
        return _validate_remote_receipt(
            value,
            request_payload_sha256=expected_request,
            stage_payload_sha256=stage_payload_sha256,
            bootstrap_sha256=bootstrap_sha256,
            commit_terminal_digest=expected_terminal_digest,
        )
    if value.get("schema_version") == TERMINAL_SCHEMA:
        target_identity = _validated_target_identity(
            value.get("target_root_identity")
        )
        reservation_state = _validated_receipt_reservation_state(
            value.get("receipt_reservation_state")
        )
        rename_result = value.get("rename_result")
        rename_classification = value.get("rename_classification")
        receipt_phase = value.get("receipt_phase")
        receipt_authoritative = value.get("receipt_authoritative")
        named_target_same = value.get("named_target_same_held_inode")
        recovery_admissible = value.get("recovery_admissible")
        if (
            (rename_result, rename_classification) not in (
                ("returned_success", "target_is_held_shadow"),
                ("raised", "applied_then_error_target_is_held_shadow"),
            )
            or receipt_phase not in (
                "reserved_0600", "partial_0600", "sealed_0400_exact",
                "sealed_0400_unverified", "unavailable",
            )
            or type(receipt_authoritative) is not bool
            or type(named_target_same) is not bool
            or type(recovery_admissible) is not bool
            or recovery_admissible is not True
            or named_target_same is not True
            or reservation_state["available"] is not True
            or receipt_authoritative != (receipt_phase == "sealed_0400_exact")
            or (
                receipt_phase in ("reserved_0600", "partial_0600")
                and reservation_state["mode"] != RECEIPT_RESERVATION_MODE
            )
            or (
                receipt_phase == "sealed_0400_exact"
                and reservation_state["mode"] != RECEIPT_MODE
            )
        ):
            raise ControllerDeployerError(
                "remote commit recovery admission differs"
            )
        expected = _expected_commit_terminal(
            stage_payload_sha256=stage_payload_sha256,
            bootstrap_sha256=bootstrap_sha256,
            target_identity=target_identity,
            receipt_reservation_state=reservation_state,
            rename_result=rename_result,
            rename_classification=rename_classification,
            receipt_phase=receipt_phase,
            receipt_authoritative=receipt_authoritative,
            named_target_same_held_inode=named_target_same,
            recovery_admissible=recovery_admissible,
        )
        if value != expected:
            raise ControllerDeployerError(
                "remote commit terminal closure differs"
            )
        if operation == RECOVER_RECEIPT_OPERATION:
            if type(commit_terminal) is not dict:
                raise ControllerDeployerError(
                    "refreshed terminal lacks prior authority"
                )
            prior_state = _validated_receipt_reservation_state(
                commit_terminal.get("receipt_reservation_state")
            )
            if (
                value.get("target_root_identity")
                != commit_terminal.get("target_root_identity")
                or (rename_result, rename_classification) != (
                    commit_terminal.get("rename_result"),
                    commit_terminal.get("rename_classification"),
                )
                or prior_state["available"] is not True
                or _inode_anchor_from_identity(prior_state["identity"])
                != _inode_anchor_from_identity(reservation_state["identity"])
            ):
                raise ControllerDeployerError(
                    "refreshed recovery terminal inode chain differs"
                )
        elif operation != STAGE_OPERATION:
            raise ControllerDeployerError(
                "remote commit terminal operation differs"
            )
        return dict(value)
    raise ControllerDeployerError("remote deployment result schema differs")


def audit_value() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "state": CONTROLLER_STATE,
        "authorization_token": authorization_token(),
        "controller": {
            "path": str(LOCAL_CONTROLLER),
            "sha256": LOCAL_CONTROLLER_SHA256,
            "size": LOCAL_CONTROLLER_SIZE,
        },
        "bootstrap": {
            "path": str(LOCAL_BOOTSTRAP),
            "sha256": LOCAL_BOOTSTRAP_SHA256,
            "size": LOCAL_BOOTSTRAP_SIZE,
        },
        "manifest": manifest_value(),
        "remote_loader_sha256": hashlib.sha256(
            REMOTE_LOADER_SOURCE.encode("utf-8")
        ).hexdigest(),
        "transport_profile_digest": object_digest(_transport_profile()),
        "named_transport": True,
        "held_fd_transport": False,
        "credential_descriptor_transport": False,
        "system_ssh_descriptor_exec": False,
        "transport_authority_replay_points": [
            "after_open", "pre_spawn", "immediate_post_spawn", "post_reap",
        ],
        "same_uid_root_kernel_mount_attacker_out_of_scope": True,
        "residual_named_lookup_window_absence_claimed": False,
        "remote_isolated_flags": ["-I", "-S", "-B"],
        "single_attempt": True,
        "retry_allowed": False,
        "launch_allowed": False,
        "slurm_allowed": False,
    }
    value["audit_digest"] = object_digest(value)
    return value


def controller(
    *, execute: bool, operation: str = STAGE_OPERATION,
    terminal_path: Path | None = None,
) -> dict[str, Any]:
    source_authorities: tuple[HeldAuthority, HeldAuthority] | None = None
    transport: list[HeldAuthority] = []
    held_input: Any = None
    held_terminals: list[HeldCommitTerminal] = []
    terminal_persistence_failed = False
    try:
        source_authorities = _open_sources()
        controller_source, bootstrap = source_authorities
        for authority in source_authorities:
            authority.replay()
        audit = audit_value()
        if not execute:
            if operation != STAGE_OPERATION or terminal_path is not None:
                raise ControllerDeployerError(
                    "local audit operation differs"
                )
            transport = _open_transport_authorities()
            _validate_named_transport_authorities(transport)
            return audit
        remote_commit_terminal: dict[str, Any] | None = None
        if operation == STAGE_OPERATION:
            if terminal_path is not None:
                raise ControllerDeployerError(
                    "stage terminal path must be absent"
                )
        elif operation == RECOVER_RECEIPT_OPERATION:
            if terminal_path is None:
                raise ControllerDeployerError(
                    "receipt recovery terminal path is absent"
                )
            held_terminal = _open_local_commit_terminal(terminal_path)
            held_terminals.append(held_terminal)
            terminal_value = held_terminal.replay()
            remote_commit_terminal = dict(
                terminal_value["remote_commit_terminal"]
            )
            if remote_commit_terminal.get("recovery_admissible") is not True:
                raise ControllerDeployerError(
                    "local commit terminal does not admit recovery"
                )
        else:
            raise ControllerDeployerError("controller operation differs")
        payload = payload_value(
            controller_source.raw, bootstrap.sha256,
            operation=operation, commit_terminal=remote_commit_terminal,
        )
        payload_raw = canonical(payload) + b"\n"
        payload_sha256 = hashlib.sha256(payload_raw).hexdigest()
        stage_payload_sha256 = hashlib.sha256(
            _payload_raw_for_operation(payload, STAGE_OPERATION)
        ).hexdigest()
        if remote_commit_terminal is not None:
            _validate_remote_result(
                canonical(remote_commit_terminal) + b"\n",
                operation=RECOVER_RECEIPT_OPERATION,
                request_payload_sha256=payload_sha256,
                stage_payload_sha256=stage_payload_sha256,
                bootstrap_sha256=bootstrap.sha256,
                commit_terminal=remote_commit_terminal,
            )
        framed = _frame(bootstrap.raw, payload_raw)
        held_input = tempfile.TemporaryFile()
        if held_input.write(framed) != len(framed):
            raise ControllerDeployerError("held stdin write differs")
        held_input.flush()
        os.fsync(held_input.fileno())
        held_input.seek(0)
        stdin_identity = _identity(os.fstat(held_input.fileno()))
        if held_input.read() != framed:
            raise ControllerDeployerError("held stdin replay differs")
        stdout = _execute_remote(
            held_input,
            bootstrap_sha256=bootstrap.sha256,
            payload_sha256=payload_sha256,
        )
        held_input.seek(0)
        if (
            _identity(os.fstat(held_input.fileno())) != stdin_identity
            or held_input.read() != framed
        ):
            raise ControllerDeployerError("held stdin changed")
        for authority in source_authorities:
            authority.replay()
        result = _validate_remote_result(
            stdout, operation=operation,
            request_payload_sha256=payload_sha256,
            stage_payload_sha256=stage_payload_sha256,
            bootstrap_sha256=bootstrap.sha256,
            commit_terminal=remote_commit_terminal,
        )
        if result.get("schema_version") == TERMINAL_SCHEMA:
            try:
                output_terminal = _write_local_commit_terminal(
                    _terminal_output_path(operation, result), result,
                )
                held_terminals.append(output_terminal)
                persisted = output_terminal.replay()
            except BaseException as persistence_error:
                terminal_persistence_failed = True
                raise ControllerDeployerError(
                    "TARGET_RENAMED_LOCAL_TERMINAL_PERSISTENCE_FAILED_"
                    "PERMANENT_HOLD"
                ) from persistence_error
            raise ControllerCommitRecoveryRequired(persisted)
        return result
    finally:
        close_errors: list[BaseException] = []
        if held_input is not None:
            try:
                held_input.close()
            except BaseException as error:
                close_errors.append(error)
        try:
            _close_transport_authorities(transport)
        except BaseException as error:
            close_errors.append(error)
        if source_authorities is not None:
            for authority in source_authorities:
                try:
                    authority.replay()
                except BaseException as error:
                    close_errors.append(error)
                try:
                    authority.close()
                except BaseException as error:
                    close_errors.append(error)
        if not terminal_persistence_failed:
            for terminal in held_terminals:
                try:
                    terminal.replay()
                except BaseException as error:
                    close_errors.append(error)
        for terminal in held_terminals:
            try:
                terminal.close()
            except BaseException as error:
                close_errors.append(error)
        if close_errors:
            raise ControllerDeployerError(
                "local held authority close differs",
            ) from close_errors[0]


def main(argv: Sequence[str] | None = None) -> int:
    # This state gate must remain before argv parsing and before every explicit
    # open, tempfile, subprocess, directory, file, network, or Slurm action.
    if CONTROLLER_STATE != READY_STATE:
        print(
            "HOLD: controller deployer awaits review and a state-only copy",
            file=sys.stderr,
        )
        return 88
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        if values == ["--audit-local"]:
            result = controller(execute=False)
        elif (
            len(values) == 2 and values[0] == "--execute"
            and values[1] == authorization_token()
        ):
            result = controller(execute=True)
        elif (
            len(values) == 3
            and values[0] == RECOVER_RECEIPT_OPERATION
            and values[2] == authorization_token()
        ):
            result = controller(
                execute=True,
                operation=RECOVER_RECEIPT_OPERATION,
                terminal_path=Path(values[1]),
            )
        else:
            raise ControllerDeployerError("argv or authorization token differs")
        print(canonical(result).decode("utf-8"))
        return 0
    except ControllerCommitRecoveryRequired as error:
        print(canonical(error.terminal).decode("utf-8"), file=sys.stderr)
        return COMMIT_RECOVERY_REQUIRED_RC
    except (OSError, ValueError, KeyError, ControllerDeployerError) as error:
        print(f"controller deployer refused: {error}", file=sys.stderr)
        return 96


if __name__ == "__main__":
    raise SystemExit(main())
