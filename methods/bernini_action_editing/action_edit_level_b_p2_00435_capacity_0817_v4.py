#!/usr/bin/env python3
"""Fail-closed, no-tensor node279 capacity admission for Level-B v4.

The producer executes one combined ROCm SMI report.  ROCm SMI may itself use
``/bin/sh`` and ``/usr/bin/ps`` for every PID reported; "one invocation" here
means one top-level ROCm SMI producer, not one process in the whole tree.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, NoReturn, Sequence


METHOD = "bernini-action-edit-level-b-p2-00435-capacity-0817-v4"
SCHEMA = "bernini-action-edit-level-b-p2-capacity-receipt-v4"
AUTHORITY = "PRE_D0_ENGINEERING_ONLY"
TAG = "fresh-world8-level-b-p2-00435-v4"
NODE = "auh7-1b-gpu-279"
REMOTE_USER_HOST = "guangyi.chen@auh7-1b-gpu-279"
EXPERIMENT_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_action_editing_0817"
)
LAUNCH_ROOT = EXPERIMENT_ROOT / "launchers" / TAG
CAPACITY_SELF = LAUNCH_ROOT / "action_edit_level_b_p2_00435_capacity_0817_v4.py"
ATTEMPT_ROOT = Path(
    EXPERIMENT_ROOT / "attempts" / TAG
)
ATTEMPT_STARTED = ATTEMPT_ROOT / "STARTED"
PHASES = ("foreground", "controller", "step")
MAX_RECEIPT_AGE_NS = 120 * 1_000_000_000
MAX_FUTURE_SKEW_NS = 5 * 1_000_000_000
EXPECTED_TOTAL_BYTES = 68_702_699_520
MIN_FREE_BASIS_POINTS = 9_500
MAX_GPU_USE_PERCENT = 0
MAX_ROCM_STDOUT_BYTES = 64 * 1024
PY_CACHE_PREFIX = Path(
    "/nonexistent/bernini-level-b-p2-00435-v4/pycache"
)

PYTHON = Path("/usr/bin/python3.10")
PYTHON_PIN = (0o755, 1, 5_937_800,
              "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96")
SOURCE_DIR = Path("/opt/rocm-7.0.0/libexec/rocm_smi")
ENTRY = SOURCE_DIR / "rocm_smi.py"
BINDINGS = SOURCE_DIR / "rsmiBindings.py"
BINDINGS_INIT = SOURCE_DIR / "rsmiBindingsInit.py"
SOURCE_PINS = {
    ENTRY: (0o755, 1, 209_073,
            "9ab2f8213922acc6f8d2d4f03c77a975958aaf010b0cad4f1e2c3ff7bbf9f655"),
    BINDINGS: (0o755, 1, 24_285,
               "339080b99d95aed1657a7c0a8c90ad0e51b506361d8591f5d9ec8e1399c1c40f"),
    BINDINGS_INIT: (0o755, 1, 2_113,
                    "473474723f806145825a52c739b4e110b38cef897dd64394d3bf552f4ade3b71"),
}
ROCM_LIBRARY_LINK = Path("/opt/rocm-7.0.0/lib/librocm_smi64.so.7")
ROCM_LIBRARY_LINK_TEXT = "librocm_smi64.so.7.8.70000"
ROCM_LIBRARY_TARGET = Path(
    "/opt/rocm-7.0.0/lib/librocm_smi64.so.7.8.70000"
)
ROCM_LIBRARY_TARGET_PIN = (
    0o644, 1, 1_383_840,
    "c08a51ffa7051a67264e9e7bf899abb0c5adee0366d200b452152a40c74b45f0",
)
SHELL_LINK = Path("/bin/sh")
SHELL_LINK_TEXT = "dash"
SHELL_TARGET = Path("/usr/bin/dash")
SHELL_TARGET_PIN = (
    0o755, 1, 125_688,
    "4f291296e89b784cd35479fca606f228126e3641f5bcaee68dee36583d7c9483",
)
PS = Path("/usr/bin/ps")
PS_PIN = (
    0o755, 1, 141_776,
    "207df9d438f75185ab3af2ab1173d104831a6631c28ef40d38b2ab43de27b40f",
)

SSH = Path("/usr/bin/ssh")
SSH_PIN = (
    0o755, 1, 846_888,
    "3a9c5d143150f0b2816ab1a5a7c58a9f970280b061f617abee54d2834a498b53",
)
KNOWN_HOSTS = LAUNCH_ROOT / "node279_known_hosts"
KNOWN_HOSTS_PIN = (
    0o444, 1, 142,
    "376ed12f9662eba4fe41396853713c9e2ad30bc3069698016f295853ce3e4454",
    2012, 2000,
)
REMOTE_HOME = Path("/vast/users/guangyi.chen")
BIN_LINK = Path("/bin")
BIN_LINK_TEXT = "usr/bin"
USR_BIN = Path("/usr/bin")
BASH = Path("/usr/bin/bash")
BASH_PIN = (
    0o755, 1, 1_396_520,
    "59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4",
)
BASHRC = REMOTE_HOME / ".bashrc"
BASHRC_PIN = (
    0o644, 1, 1_167,
    "04288c473a1034e3dc8d28174a2469aa2eb614b55b02f639e3f7273174fe1aed",
    2012, 2000,
)
REMOTE_STARTUP_ABSENT = (
    REMOTE_HOME / ".ssh" / "rc",
    Path("/etc/ssh/sshrc"),
    REMOTE_HOME / ".ssh" / "environment",
)
SSH_ENV = {
    "HOME": str(REMOTE_HOME),
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}
ENV_TOOL = Path("/usr/bin/env")
ENV_TOOL_PIN = (
    0o755, 1, 43_976,
    "85036540673319c6c2f54233fd2b9e45a8a71246b51cc96c4e6ab8ee6c419eb0",
)

ROCM_ARGS = (
    "--showproductname", "--showuniqueid", "--showserial", "--showbus",
    "--showuse", "--showmeminfo", "vram", "--showpids", "--json",
)
INITIAL_SYS_PATH = (
    "/usr/lib/python310.zip",
    "/usr/lib/python3.10",
    "/usr/lib/python3.10/lib-dynload",
)
WRAPPER = (
    "import builtins,sys;"
    f"expected={INITIAL_SYS_PATH!r};"
    "assert tuple(sys.path)==expected,sys.path;"
    "assert (sys.flags.dont_write_bytecode,sys.flags.no_user_site,"
    "sys.flags.no_site,sys.flags.ignore_environment,sys.flags.isolated,"
    "sys.flags.utf8_mode)==(1,1,1,1,1,1),sys.flags;"
    "sys.path[:]=list(expected);"
    f"source={str(SOURCE_DIR)!r};entry={str(ENTRY)!r};"
    f"args={[str(ENTRY), *ROCM_ARGS]!r};"
    "assert sys.argv==['-c']+args,sys.argv;"
    "sys.path.insert(0,source);builtins.exit=sys.exit;sys.argv=args;"
    "raw=builtins.open(entry,'rb').read();"
    "code=builtins.compile(raw,entry,'exec',dont_inherit=True,optimize=0);"
    "scope={'__name__':'__main__','__file__':entry,'__package__':None,"
    "'__cached__':None,'__spec__':None,'__builtins__':builtins.__dict__};"
    "builtins.exec(code,scope,scope)"
)
WRAPPER_SHA256 = hashlib.sha256(WRAPPER.encode("utf-8")).hexdigest()
PRODUCER_ARGV = (
    str(PYTHON), "-I", "-S", "-B", "-X",
    f"pycache_prefix={PY_CACHE_PREFIX}", "-c", WRAPPER,
    str(ENTRY), *ROCM_ARGS,
)
PRODUCER_ENV = {
    "HOME": "/nonexistent/bernini-level-b-p2-00435-v4",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "ROCM_SMI_LIB_PATH": str(ROCM_LIBRARY_LINK),
}

CARD_KEYS = {
    "Unique ID", "GPU use (%)", "GFX Activity", "Serial Number", "PCI Bus",
    "VRAM Total Memory (B)", "VRAM Total Used Memory (B)", "Card Series",
    "Card Model", "Card Vendor", "Card SKU", "Subsystem ID", "Device Rev",
    "Node ID", "GUID", "GFX Version",
}
CARD_IDENTITIES = (
    ("0x689a2657f9e1b9f9", "692436006534", "0000:05:00.0", "4", "36198"),
    ("0x58888af031a47d0", "692436011134", "0000:08:00.0", "5", "6306"),
    ("0xc3dbb1e0ab16e71c", "692436021333", "0000:47:00.0", "2", "7382"),
    ("0xf2d7837b93230bf0", "692436011280", "0000:4A:00.0", "3", "35090"),
    ("0x7d8ea22ddb71f5c3", "692436011256", "0000:85:00.0", "8", "11527"),
    ("0x4ac9b614aadc1888", "692436021294", "0000:88:00.0", "9", "47299"),
    ("0x74ec1d6644d11b53", "692436021202", "0000:C5:00.0", "6", "56630"),
    ("0x263640c503016da8", "692436006349", "0000:C8:00.0", "7", "18674"),
)
FIXED_CARD_FIELDS = {
    "Card Series": "AMD Instinct MI210",
    "Card Model": "0x740f",
    "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
    "Card SKU": "D67301V",
    "Subsystem ID": "0x0c34",
    "Device Rev": "0x02",
    "GFX Version": "gfx90a",
    "VRAM Total Memory (B)": str(EXPECTED_TOTAL_BYTES),
}


class CapacityError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise CapacityError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def exact_value(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected):
        fail(f"{label} JSON type differs")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            fail(f"{label} key closure differs")
        for key in expected:
            exact_value(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            fail(f"{label} list length differs")
        for index, (observed, wanted) in enumerate(zip(actual, expected)):
            exact_value(observed, wanted, f"{label}[{index}]")
        return
    if actual != expected:
        fail(f"{label} value differs")


def strict_json(raw: bytes) -> Any:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if not isinstance(key, str) or key in out:
                fail("duplicate or non-string JSON key")
            out[key] = value
        return out

    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique,
            parse_constant=lambda value: fail(f"non-finite JSON value: {value}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON: {exc}")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def decode_canonical_base64(value: Any, label: str) -> bytes:
    if type(value) is not str or not value or "\n" in value or "\r" in value:
        fail(f"{label} base64 type or framing differs")
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        fail(f"{label} base64 differs: {exc}")
    if base64.b64encode(raw).decode("ascii") != value:
        fail(f"{label} base64 is not canonical")
    return raw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_uint(value: Any, label: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        fail(f"{label} is not one canonical unsigned decimal")
    return int(value)


def owned_plain_snapshot(
    path: Path, pin: Sequence[Any], expected_uid: int, expected_gid: int
) -> Mapping[str, Any]:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        fail(f"non-canonical executable authority: {path}")
    info = path.lstat()
    mode, nlink, size, expected_sha = pin
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != mode
        or info.st_nlink != nlink
        or info.st_size != size
        or info.st_uid != expected_uid
        or info.st_gid != expected_gid
    ):
        fail(f"plain-file authority differs: {path}")
    observed_sha = sha256_file(path)
    after = path.lstat()
    if (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    ) or observed_sha != expected_sha:
        fail(f"unstable or unpinned authority: {path}")
    return {
        "path": str(path), "mode": mode, "nlink": nlink, "size": size,
        "sha256": observed_sha, "uid": info.st_uid, "gid": info.st_gid,
    }


def plain_snapshot(path: Path, pin: Sequence[Any]) -> Mapping[str, Any]:
    return owned_plain_snapshot(path, pin, 0, 0)


def symlink_snapshot(path: Path, link_text: str, target: Path) -> Mapping[str, Any]:
    if not path.is_absolute():
        fail(f"non-absolute symlink authority: {path}")
    info = path.lstat()
    if (
        not stat.S_ISLNK(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o777
        or info.st_nlink != 1
        or info.st_size != len(link_text)
        or os.readlink(path) != link_text
        or path.resolve(strict=True) != target
        or info.st_uid != 0
        or info.st_gid != 0
    ):
        fail(f"symlink authority differs: {path}")
    return {
        "path": str(path), "mode": 0o777, "nlink": 1,
        "size": info.st_size, "link_text": link_text,
        "resolved_target": str(target), "uid": info.st_uid, "gid": info.st_gid,
    }


def lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def directory_snapshot(
    path: Path, expected_mode: int, expected_uid: int, expected_gid: int
) -> Mapping[str, Any]:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        fail(f"non-canonical directory authority: {path}")
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != expected_mode
        or info.st_uid != expected_uid
        or info.st_gid != expected_gid
    ):
        fail(f"directory authority differs: {path}")
    return {
        "path": str(path), "mode": expected_mode,
        "uid": expected_uid, "gid": expected_gid,
    }


def user_plain_snapshot(path: Path, pin: Sequence[Any]) -> Mapping[str, Any]:
    mode, nlink, size, digest, uid, gid = pin
    return owned_plain_snapshot(path, (mode, nlink, size, digest), uid, gid)


def sealed_capacity_member_snapshot() -> Mapping[str, Any]:
    path = CAPACITY_SELF
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        fail("capacity member path is not canonical")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
        or before.st_size <= 0 or before.st_size > 128 * 1024
        or before.st_uid != 2012 or before.st_gid != 2000
    ):
        fail("capacity member topology differs")
    digest = sha256_file(path)
    after = path.lstat()
    if (
        before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
        before.st_size, before.st_mtime_ns, before.st_ctime_ns,
    ) != (
        after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
        after.st_size, after.st_mtime_ns, after.st_ctime_ns,
    ):
        fail("capacity member changed during read")
    return {
        "path": str(path), "mode": 0o444, "nlink": 1,
        "size": before.st_size, "sha256": digest, "uid": 2012, "gid": 2000,
    }


def source_directory_snapshot() -> Mapping[str, Any]:
    if SOURCE_DIR.is_symlink() or SOURCE_DIR.resolve(strict=True) != SOURCE_DIR:
        fail("ROCm source directory is not canonical")
    info = SOURCE_DIR.lstat()
    if (
        not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o755
        or info.st_uid != 0 or info.st_gid != 0
    ):
        fail("ROCm source directory authority differs")
    entries = sorted(item.name for item in os.scandir(SOURCE_DIR))
    expected = ["__pycache__", "rocm_smi.py", "rsmiBindings.py", "rsmiBindingsInit.py"]
    if entries != expected:
        fail("ROCm source directory closure differs")
    return {"path": str(SOURCE_DIR), "mode": 0o755, "uid": 0, "gid": 0,
            "entries": entries}


def executable_closure_snapshot() -> Mapping[str, Any]:
    source_dir = source_directory_snapshot()
    sources = [plain_snapshot(path, SOURCE_PINS[path]) for path in SOURCE_PINS]
    library_link = symlink_snapshot(
        ROCM_LIBRARY_LINK, ROCM_LIBRARY_LINK_TEXT, ROCM_LIBRARY_TARGET
    )
    library_target = plain_snapshot(ROCM_LIBRARY_TARGET, ROCM_LIBRARY_TARGET_PIN)
    shell_link = symlink_snapshot(SHELL_LINK, SHELL_LINK_TEXT, SHELL_TARGET)
    shell_target = plain_snapshot(SHELL_TARGET, SHELL_TARGET_PIN)
    return {
        "python": plain_snapshot(PYTHON, PYTHON_PIN),
        "source_directory": source_dir,
        "sources": sources,
        "library_link": library_link,
        "library_target": library_target,
        "shell_link": shell_link,
        "shell_target": shell_target,
        "ps": plain_snapshot(PS, PS_PIN),
    }


def remote_shell_boundary_snapshot() -> Mapping[str, Any]:
    if os.getuid() != 2012 or os.getgid() != 2000:
        fail("direct-node account identity differs")
    if Path.cwd() != Path("/"):
        fail("direct-node command cwd differs")
    expected_environment = {
        "HOME": "/nonexistent/bernini-level-b-p2-00435-v4-remote-target",
        "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin",
    }
    if dict(os.environ) != expected_environment:
        fail("direct-node target environment differs")
    account = pwd.getpwuid(2012)
    account_mapping = {
        "name": account.pw_name, "uid": account.pw_uid, "gid": account.pw_gid,
        "home": account.pw_dir, "shell": account.pw_shell,
    }
    expected_account_mapping = {
        "name": "guangyi.chen", "uid": 2012, "gid": 2000,
        "home": str(REMOTE_HOME), "shell": "/bin/bash",
    }
    if account_mapping != expected_account_mapping:
        fail("direct-node NSS account mapping differs")
    absent = []
    for path in REMOTE_STARTUP_ABSENT:
        if lexists(path):
            fail(f"direct-node startup surface unexpectedly exists: {path}")
        absent.append(str(path))
    return {
        "account_mapping": account_mapping,
        "bin_link": symlink_snapshot(BIN_LINK, BIN_LINK_TEXT, USR_BIN),
        "usr_bin_directory": directory_snapshot(USR_BIN, 0o755, 0, 0),
        "bash": plain_snapshot(BASH, BASH_PIN),
        "env_tool": plain_snapshot(ENV_TOOL, ENV_TOOL_PIN),
        "bashrc": user_plain_snapshot(BASHRC, BASHRC_PIN),
        "absent_startup_paths": absent,
        "environment_after_absolute_env_i": expected_environment,
        "same_user_login_shell_startup_and_transitive_conda_are_trusted_boundary": True,
        "remote_shell_was_entered_before_absolute_env_i": True,
        "nss_and_passwd_resolution_are_system_trust_boundary": True,
    }


def local_transport_claim() -> Mapping[str, Any]:
    return {
        "mode": "local-node279",
        "direct_node_ssh_used": False,
        "sample_was_taken_in_current_step_process_tree": True,
    }


def remote_target_environment() -> Mapping[str, str]:
    return {
        "HOME": "/nonexistent/bernini-level-b-p2-00435-v4-remote-target",
        "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin",
    }


def remote_target_command(phase: str, challenge: str) -> str:
    validate_phase_challenge(phase, challenge)
    if phase not in {"foreground", "controller"}:
        fail("direct-node SSH phase differs")
    environment = remote_target_environment()
    assignments = " ".join(f"{key}={environment[key]}" for key in sorted(environment))
    return (
        "cd / && exec /usr/bin/env -i " + assignments + " "
        + f"{PYTHON} -I -S -B {CAPACITY_SELF} "
        + f"direct-node-target-base64 {phase} {challenge}"
    )


def ssh_argv(phase: str, challenge: str) -> tuple[str, ...]:
    command = remote_target_command(phase, challenge)
    options = (
        "BatchMode=yes", "ConnectTimeout=10", "ConnectionAttempts=1",
        "StrictHostKeyChecking=yes", f"UserKnownHostsFile={KNOWN_HOSTS}",
        "GlobalKnownHostsFile=/dev/null", "HostKeyAlgorithms=ssh-ed25519",
        "UpdateHostKeys=no", "IdentitiesOnly=yes", "GSSAPIAuthentication=no",
        "KbdInteractiveAuthentication=no", "PasswordAuthentication=no",
        "ForwardAgent=no", "ForwardX11=no", "ClearAllForwardings=yes",
        "PermitLocalCommand=no", "ControlMaster=no", "ControlPath=none",
        "CanonicalizeHostname=no", "Hostname=auh7-1b-gpu-279",
        "KnownHostsCommand=none", "VerifyHostKeyDNS=no",
        "ProxyCommand=none", "ProxyJump=none",
    )
    argv = [str(SSH), "-F", "/dev/null", "-n", "-T"]
    for option in options:
        argv.extend(("-o", option))
    argv.extend((REMOTE_USER_HOST, command))
    return tuple(argv)


def ssh_authority_snapshot() -> Mapping[str, Any]:
    return {
        "ssh_client": plain_snapshot(SSH, SSH_PIN),
        "sealed_known_hosts": user_plain_snapshot(KNOWN_HOSTS, KNOWN_HOSTS_PIN),
        "capacity_member": sealed_capacity_member_snapshot(),
    }


def _validate_plain_receipt_row(row: Any, path: Path, pin: Sequence[Any]) -> None:
    required = {"path", "mode", "nlink", "size", "sha256", "uid", "gid"}
    if not isinstance(row, dict) or set(row) != required:
        fail(f"receipt plain-file schema differs: {path}")
    mode, nlink, size, expected_sha = pin
    exact_value(row, {
        "path": str(path), "mode": mode, "nlink": nlink, "size": size,
        "sha256": expected_sha, "uid": 0, "gid": 0,
    }, f"receipt plain-file pin: {path}")


def _validate_user_plain_receipt_row(row: Any, path: Path, pin: Sequence[Any]) -> None:
    mode, nlink, size, digest, uid, gid = pin
    required = {"path", "mode", "nlink", "size", "sha256", "uid", "gid"}
    if not isinstance(row, dict) or set(row) != required:
        fail(f"receipt user plain-file schema differs: {path}")
    exact_value(row, {
        "path": str(path), "mode": mode, "nlink": nlink, "size": size,
        "sha256": digest, "uid": uid, "gid": gid,
    }, f"receipt user plain-file pin: {path}")


def _validate_capacity_member_receipt_row(row: Any) -> None:
    required = {"path", "mode", "nlink", "size", "sha256", "uid", "gid"}
    if not isinstance(row, dict) or set(row) != required:
        fail("capacity member receipt schema differs")
    if (
        row["path"] != str(CAPACITY_SELF) or type(row["mode"]) is not int
        or row["mode"] != 0o444 or type(row["nlink"]) is not int
        or row["nlink"] != 1 or type(row["size"]) is not int
        or row["size"] <= 0 or row["size"] > 128 * 1024
        or type(row["sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
        or type(row["uid"]) is not int or row["uid"] != 2012
        or type(row["gid"]) is not int or row["gid"] != 2000
    ):
        fail("capacity member receipt authority differs")


def _validate_symlink_receipt_row(
    row: Any, path: Path, link_text: str, target: Path
) -> None:
    required = {"path", "mode", "nlink", "size", "link_text", "resolved_target",
                "uid", "gid"}
    if not isinstance(row, dict) or set(row) != required:
        fail(f"receipt symlink schema differs: {path}")
    exact_value(row, {
        "path": str(path), "mode": 0o777, "nlink": 1, "size": len(link_text),
        "link_text": link_text, "resolved_target": str(target), "uid": 0, "gid": 0,
    }, f"receipt symlink pin: {path}")


def validate_executable_closure_receipt(value: Any) -> None:
    required = {"python", "source_directory", "sources", "library_link",
                "library_target", "shell_link", "shell_target", "ps"}
    if not isinstance(value, dict) or set(value) != required:
        fail("receipt executable closure schema differs")
    _validate_plain_receipt_row(value["python"], PYTHON, PYTHON_PIN)
    source_dir = value["source_directory"]
    if not isinstance(source_dir, dict) or set(source_dir) != {
        "path", "mode", "uid", "gid", "entries"
    }:
        fail("receipt source-directory schema differs")
    exact_value(source_dir, {
        "path": str(SOURCE_DIR), "mode": 0o755, "uid": 0, "gid": 0,
        "entries": ["__pycache__", "rocm_smi.py", "rsmiBindings.py",
                    "rsmiBindingsInit.py"],
    }, "receipt source-directory pin")
    sources = value["sources"]
    if not isinstance(sources, list) or len(sources) != len(SOURCE_PINS):
        fail("receipt source-file closure differs")
    for row, path in zip(sources, SOURCE_PINS):
        _validate_plain_receipt_row(row, path, SOURCE_PINS[path])
    _validate_symlink_receipt_row(
        value["library_link"], ROCM_LIBRARY_LINK,
        ROCM_LIBRARY_LINK_TEXT, ROCM_LIBRARY_TARGET,
    )
    _validate_plain_receipt_row(
        value["library_target"], ROCM_LIBRARY_TARGET, ROCM_LIBRARY_TARGET_PIN
    )
    _validate_symlink_receipt_row(
        value["shell_link"], SHELL_LINK, SHELL_LINK_TEXT, SHELL_TARGET
    )
    _validate_plain_receipt_row(value["shell_target"], SHELL_TARGET, SHELL_TARGET_PIN)
    _validate_plain_receipt_row(value["ps"], PS, PS_PIN)


def validate_remote_shell_boundary_receipt(value: Any) -> None:
    required = {
        "account_mapping", "bin_link", "usr_bin_directory", "bash", "env_tool",
        "bashrc", "absent_startup_paths",
        "environment_after_absolute_env_i",
        "same_user_login_shell_startup_and_transitive_conda_are_trusted_boundary",
        "remote_shell_was_entered_before_absolute_env_i",
        "nss_and_passwd_resolution_are_system_trust_boundary",
    }
    if not isinstance(value, dict) or set(value) != required:
        fail("remote shell boundary receipt schema differs")
    fixed = {
        "account_mapping": {
            "name": "guangyi.chen", "uid": 2012, "gid": 2000,
            "home": str(REMOTE_HOME), "shell": "/bin/bash",
        },
        "absent_startup_paths": [str(path) for path in REMOTE_STARTUP_ABSENT],
        "environment_after_absolute_env_i": {
            "HOME": "/nonexistent/bernini-level-b-p2-00435-v4-remote-target",
            "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin",
        },
        "same_user_login_shell_startup_and_transitive_conda_are_trusted_boundary": True,
        "remote_shell_was_entered_before_absolute_env_i": True,
        "nss_and_passwd_resolution_are_system_trust_boundary": True,
    }
    for key, expected in fixed.items():
        exact_value(value[key], expected, f"remote shell boundary.{key}")
    _validate_symlink_receipt_row(value["bin_link"], BIN_LINK, BIN_LINK_TEXT, USR_BIN)
    exact_value(value["usr_bin_directory"], {
        "path": str(USR_BIN), "mode": 0o755, "uid": 0, "gid": 0,
    }, "remote shell /usr/bin directory")
    _validate_plain_receipt_row(value["bash"], BASH, BASH_PIN)
    _validate_plain_receipt_row(value["env_tool"], ENV_TOOL, ENV_TOOL_PIN)
    _validate_user_plain_receipt_row(value["bashrc"], BASHRC, BASHRC_PIN)


def parse_report(raw: bytes) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not raw or len(raw) > MAX_ROCM_STDOUT_BYTES or b"\x00" in raw:
        fail("ROCm stdout size or NUL boundary differs")
    if (
        raw.count(b"\n") != 1 or raw[:1] != b"{" or raw[-2:] != b"}\n"
        or b"\r" in raw or b"\t" in raw
    ):
        fail("ROCm stdout must contain exactly one trailing LF")
    report = strict_json(raw[:-1])
    expected_top = {f"card{index}" for index in range(8)} | {"system"}
    if not isinstance(report, dict) or set(report) != expected_top:
        fail("ROCm top-level card closure differs")
    cards = []
    for index, identity in enumerate(CARD_IDENTITIES):
        card = report[f"card{index}"]
        if not isinstance(card, dict) or set(card) != CARD_KEYS:
            fail(f"card{index} field closure differs")
        if any(not isinstance(value, str) for value in card.values()):
            fail(f"card{index} contains a non-string field")
        unique_id, serial, bus, node_id, guid = identity
        expected_identity = {
            **FIXED_CARD_FIELDS, "Unique ID": unique_id, "Serial Number": serial,
            "PCI Bus": bus, "Node ID": node_id, "GUID": guid,
        }
        for key, expected in expected_identity.items():
            if card[key] != expected:
                fail(f"card{index} identity differs: {key}")
        use = canonical_uint(card["GPU use (%)"], f"card{index} GPU use")
        gfx = canonical_uint(card["GFX Activity"], f"card{index} GFX activity")
        total = canonical_uint(card["VRAM Total Memory (B)"], f"card{index} total")
        used = canonical_uint(card["VRAM Total Used Memory (B)"], f"card{index} used")
        if use != MAX_GPU_USE_PERCENT or total != EXPECTED_TOTAL_BYTES or used > total:
            fail(f"card{index} use/total/used admission differs")
        free = total - used
        if free * 10_000 < total * MIN_FREE_BASIS_POINTS:
            fail(f"card{index} free VRAM is below 95 percent")
        cards.append({
            "index": index, "unique_id": unique_id, "serial_number": serial,
            "pci_bus": bus, "node_id": int(node_id), "guid": int(guid),
            "series": card["Card Series"], "model": card["Card Model"],
            "vendor": card["Card Vendor"], "sku": card["Card SKU"],
            "subsystem_id": card["Subsystem ID"], "device_rev": card["Device Rev"],
            "gfx_version": card["GFX Version"], "gpu_use_percent": use,
            "gfx_activity": gfx, "vram_total_bytes": total,
            "vram_used_bytes": used, "vram_free_bytes": free,
            "free_basis_points_floor": free * 10_000 // total,
        })
    system = report["system"]
    if not isinstance(system, dict) or len(system) != 1:
        fail("system process closure must be the sole gpuagent row")
    key, value = next(iter(system.items()))
    match = re.fullmatch(r"PID([1-9][0-9]*)", key)
    if match is None or value != "gpuagent, 0, 0, 0, 0":
        fail("system process row differs")
    processes = [{"pid": int(match.group(1)), "name": "gpuagent",
                  "gpu_count": 0, "vram_bytes": 0,
                  "cpu_time": 0, "command_count": 0}]
    return cards, processes


def validate_phase_challenge(phase: Any, challenge: Any) -> None:
    if type(phase) is not str or phase not in PHASES:
        fail("sample phase differs")
    if type(challenge) is not str or re.fullmatch(r"[0-9a-f]{64}", challenge) is None:
        fail("sample challenge differs")


def build_receipt(
    raw_report: bytes, phase: str, challenge: str, observed_time_ns: int,
    closure: Mapping[str, Any], transport: Mapping[str, Any] | None = None,
) -> bytes:
    validate_phase_challenge(phase, challenge)
    cards, processes = parse_report(raw_report)
    out = {
        "schema_version": SCHEMA, "method": METHOD, "authority": AUTHORITY,
        "tag": TAG, "node": NODE, "sample_phase": phase,
        "sample_challenge": challenge, "observed_unix_time_ns": observed_time_ns,
        "transport": dict(local_transport_claim() if transport is None else transport),
        "producer": {
            "top_level_invocation_count": 1,
            "coherent_single_report_not_hardware_atomic_snapshot": True,
            "argv": list(PRODUCER_ARGV), "argv_sha256": sha256_bytes(
                b"\0".join(value.encode("utf-8") for value in PRODUCER_ARGV)
            ),
            "wrapper_sha256": WRAPPER_SHA256, "cwd": "/",
            "environment": dict(PRODUCER_ENV),
            "environment_inherited_from_caller": False,
            "rocm_smi_exit_code": 0, "rocm_smi_stderr_size": 0,
            "rocm_smi_stderr_sha256": sha256_bytes(b""),
            "rocm_smi_stdout_size": len(raw_report),
            "rocm_smi_stdout_sha256": sha256_bytes(raw_report),
            "rocm_smi_stdout_base64": base64.b64encode(raw_report).decode("ascii"),
            "internal_pid_helpers": "/bin/sh -c /usr/bin/ps per reported PID",
            "pidof_call_path_reached": False,
            "pycache_prefix": str(PY_CACHE_PREFIX),
            "pycache_prefix_absent_before_and_after": True,
            "torch_or_hip_python_imported_by_member": False,
            "dummy_tensor_or_gpu_memory_allocated_by_member": False,
            "selected_producer_closure": closure,
            "selected_producer_closure_stable_pre_and_post": True,
            "system_trust_boundary": [
                "root-owned Python standard library",
                "root-owned dynamic loader and native dependencies",
                "kernel, AMDGPU driver, and ROCm management ABI",
            ],
        },
        "thresholds": {
            "gpu_count": 8, "gpu_use_percent_required": MAX_GPU_USE_PERCENT,
            "vram_total_bytes_required": EXPECTED_TOTAL_BYTES,
            "minimum_free_basis_points": MIN_FREE_BASIS_POINTS,
            "per_card_not_average": True,
        },
        "cards": cards, "system_processes": processes, "passed": True,
    }
    unsigned = canonical_json_bytes(out)
    out["receipt_digest"] = sha256_bytes(unsigned)
    return canonical_json_bytes(out)


def validate_receipt_bytes(
    raw: bytes, expected_sha: str, expected_phase: str,
    expected_challenge: str, now_ns: int | None = None,
    _expected_transport_mode: str | None = None,
    _enforce_max_age: bool = True,
) -> Mapping[str, Any]:
    validate_phase_challenge(expected_phase, expected_challenge)
    if not _enforce_max_age and expected_phase != "step":
        fail("archival validation is restricted to step receipts")
    if type(expected_sha) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        fail("expected receipt SHA differs")
    if sha256_bytes(raw) != expected_sha or b"\n" in raw or b"\x00" in raw:
        fail("receipt byte authority differs")
    value = strict_json(raw)
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        fail("receipt is not canonical JSON")
    required = {
        "schema_version", "method", "authority", "tag", "node", "sample_phase",
        "sample_challenge", "observed_unix_time_ns", "producer", "thresholds",
        "cards", "system_processes", "passed", "receipt_digest", "transport",
    }
    if set(value) != required:
        fail("receipt top-level closure differs")
    validate_phase_challenge(value["sample_phase"], value["sample_challenge"])
    if (
        value["schema_version"] != SCHEMA or value["method"] != METHOD
        or value["authority"] != AUTHORITY or value["tag"] != TAG
        or value["node"] != NODE or value["sample_phase"] != expected_phase
        or value["sample_challenge"] != expected_challenge
        or value["passed"] is not True
    ):
        fail("receipt identity or phase differs")
    issued = value["observed_unix_time_ns"]
    if isinstance(issued, bool) or not isinstance(issued, int) or issued <= 0:
        fail("receipt timestamp differs")
    current = time.time_ns() if now_ns is None else now_ns
    if issued - current > MAX_FUTURE_SKEW_NS:
        fail("receipt is too far in the future")
    if _enforce_max_age and current - issued > MAX_RECEIPT_AGE_NS:
        fail("receipt is stale")
    digest = value["receipt_digest"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        fail("receipt digest format differs")
    unsigned = dict(value)
    del unsigned["receipt_digest"]
    if digest != sha256_bytes(canonical_json_bytes(unsigned)):
        fail("receipt digest differs")
    producer = value["producer"]
    required_producer = {
        "top_level_invocation_count", "coherent_single_report_not_hardware_atomic_snapshot",
        "argv", "argv_sha256", "wrapper_sha256", "cwd", "environment",
        "environment_inherited_from_caller", "rocm_smi_exit_code",
        "rocm_smi_stderr_size", "rocm_smi_stderr_sha256", "rocm_smi_stdout_size",
        "rocm_smi_stdout_sha256", "rocm_smi_stdout_base64", "internal_pid_helpers",
        "pidof_call_path_reached", "pycache_prefix",
        "pycache_prefix_absent_before_and_after", "torch_or_hip_python_imported_by_member",
        "dummy_tensor_or_gpu_memory_allocated_by_member", "selected_producer_closure",
        "selected_producer_closure_stable_pre_and_post", "system_trust_boundary",
    }
    if not isinstance(producer, dict) or set(producer) != required_producer:
        fail("receipt producer closure differs")
    fixed_producer = {
        "top_level_invocation_count": 1,
        "coherent_single_report_not_hardware_atomic_snapshot": True,
        "argv": list(PRODUCER_ARGV),
        "argv_sha256": sha256_bytes(b"\0".join(v.encode() for v in PRODUCER_ARGV)),
        "wrapper_sha256": WRAPPER_SHA256, "cwd": "/",
        "environment": dict(PRODUCER_ENV), "environment_inherited_from_caller": False,
        "rocm_smi_exit_code": 0, "rocm_smi_stderr_size": 0,
        "rocm_smi_stderr_sha256": sha256_bytes(b""),
        "internal_pid_helpers": "/bin/sh -c /usr/bin/ps per reported PID",
        "pidof_call_path_reached": False, "pycache_prefix": str(PY_CACHE_PREFIX),
        "pycache_prefix_absent_before_and_after": True,
        "torch_or_hip_python_imported_by_member": False,
        "dummy_tensor_or_gpu_memory_allocated_by_member": False,
        "selected_producer_closure_stable_pre_and_post": True,
        "system_trust_boundary": [
            "root-owned Python standard library",
            "root-owned dynamic loader and native dependencies",
            "kernel, AMDGPU driver, and ROCm management ABI",
        ],
    }
    for key, expected in fixed_producer.items():
        exact_value(producer[key], expected, f"receipt producer claim.{key}")
    validate_executable_closure_receipt(producer["selected_producer_closure"])
    report_raw = decode_canonical_base64(
        producer["rocm_smi_stdout_base64"], "receipt report"
    )
    exact_value(
        producer["rocm_smi_stdout_size"], len(report_raw),
        "receipt producer report size",
    )
    exact_value(
        producer["rocm_smi_stdout_sha256"], sha256_bytes(report_raw),
        "receipt producer report SHA",
    )
    cards, processes = parse_report(report_raw)
    exact_value(value["cards"], cards, "receipt derived cards")
    exact_value(value["system_processes"], processes, "receipt derived processes")
    exact_value(value["thresholds"], {
        "gpu_count": 8, "gpu_use_percent_required": 0,
        "vram_total_bytes_required": EXPECTED_TOTAL_BYTES,
        "minimum_free_basis_points": MIN_FREE_BASIS_POINTS,
        "per_card_not_average": True,
    }, "receipt threshold contract")
    expected_transport_mode = _expected_transport_mode
    if expected_transport_mode is None:
        expected_transport_mode = (
            "local-node279" if expected_phase == "step" else "direct-node-ssh"
        )
    validate_transport_receipt(
        value["transport"], expected_transport_mode, expected_phase,
        expected_challenge, value, now_ns=current,
        enforce_max_age=_enforce_max_age,
    )
    return value


def validate_transport_receipt(
    transport: Any, expected_mode: str, phase: str, challenge: str,
    receipt_value: Mapping[str, Any], now_ns: int, enforce_max_age: bool,
) -> None:
    if not isinstance(transport, dict) or transport.get("mode") != expected_mode:
        fail("capacity receipt transport mode differs")
    if expected_mode == "local-node279":
        exact_value(transport, local_transport_claim(), "local capacity transport")
        if phase != "step":
            fail("local capacity transport phase differs")
        return
    if expected_mode == "direct-node-ssh-target":
        required = {
            "mode", "remote_shell_boundary",
            "remote_shell_boundary_stable_pre_and_post",
            "capacity_member", "capacity_member_stable_pre_and_post",
            "capacity_member_sha256_must_be_bound_by_launch_authority",
        }
        if set(transport) != required:
            fail("direct-node target transport schema differs")
        exact_value(
            transport["remote_shell_boundary_stable_pre_and_post"], True,
            "direct-node target shell stability",
        )
        validate_remote_shell_boundary_receipt(transport["remote_shell_boundary"])
        _validate_capacity_member_receipt_row(transport["capacity_member"])
        exact_value(
            transport["capacity_member_stable_pre_and_post"], True,
            "direct-node target capacity-member stability",
        )
        exact_value(
            transport["capacity_member_sha256_must_be_bound_by_launch_authority"],
            True, "direct-node target external member-SHA binding",
        )
        if phase not in {"foreground", "controller"}:
            fail("direct-node target phase differs")
        return
    if expected_mode != "direct-node-ssh":
        fail("unknown capacity receipt transport mode")
    required = {
        "mode", "ssh_top_level_invocation_count", "ssh_client",
        "sealed_known_hosts", "capacity_member", "ssh_argv", "ssh_argv_sha256",
        "ssh_environment",
        "ssh_environment_inherited_from_caller", "ssh_exit_code",
        "ssh_stderr_size", "ssh_stderr_sha256", "ssh_stdout_size",
        "ssh_stdout_sha256", "remote_target_receipt_base64",
        "remote_target_receipt_size", "remote_target_receipt_sha256",
        "ssh_client_and_known_hosts_stable_pre_and_post",
        "remote_target_transport_mode", "fresh_challenge_bound_to_ssh_argv",
        "same_user_ssh_authentication_and_login_startup_are_trusted_boundary",
        "ssh_system_trust_boundary",
        "capacity_member_sha256_must_be_bound_by_launch_authority",
    }
    if set(transport) != required:
        fail("direct-node SSH transport schema differs")
    expected_argv = ssh_argv(phase, challenge)
    fixed = {
        "mode": "direct-node-ssh", "ssh_top_level_invocation_count": 1,
        "ssh_argv": list(expected_argv),
        "ssh_argv_sha256": sha256_bytes(
            b"\0".join(item.encode("utf-8") for item in expected_argv)
        ),
        "ssh_environment": dict(SSH_ENV),
        "ssh_environment_inherited_from_caller": False,
        "ssh_exit_code": 0, "ssh_stderr_size": 0,
        "ssh_stderr_sha256": sha256_bytes(b""),
        "ssh_client_and_known_hosts_stable_pre_and_post": True,
        "remote_target_transport_mode": "direct-node-ssh-target",
        "fresh_challenge_bound_to_ssh_argv": True,
        "same_user_ssh_authentication_and_login_startup_are_trusted_boundary": True,
        "capacity_member_sha256_must_be_bound_by_launch_authority": True,
        "ssh_system_trust_boundary": [
            "root-owned SSH client dynamic loader and shared libraries",
            "root-owned resolver and NSS stack",
            "kernel and network stack",
            "remote sshd, server configuration, and host private key",
        ],
    }
    for key, expected in fixed.items():
        exact_value(transport[key], expected, f"direct-node SSH transport.{key}")
    _validate_plain_receipt_row(transport["ssh_client"], SSH, SSH_PIN)
    _validate_user_plain_receipt_row(
        transport["sealed_known_hosts"], KNOWN_HOSTS, KNOWN_HOSTS_PIN
    )
    _validate_capacity_member_receipt_row(transport["capacity_member"])
    target_raw = decode_canonical_base64(
        transport["remote_target_receipt_base64"], "remote target receipt"
    )
    exact_value(
        transport["remote_target_receipt_size"], len(target_raw),
        "remote target receipt size",
    )
    exact_value(
        transport["remote_target_receipt_sha256"], sha256_bytes(target_raw),
        "remote target receipt SHA",
    )
    target_stdout = transport["remote_target_receipt_base64"].encode("ascii")
    exact_value(
        transport["ssh_stdout_size"], len(target_stdout), "SSH stdout size"
    )
    exact_value(
        transport["ssh_stdout_sha256"], sha256_bytes(target_stdout),
        "SSH stdout SHA",
    )
    target = validate_receipt_bytes(
        target_raw, sha256_bytes(target_raw), phase, challenge,
        now_ns=now_ns, _expected_transport_mode="direct-node-ssh-target",
        _enforce_max_age=enforce_max_age,
    )
    exact_value(
        transport["capacity_member"], target["transport"]["capacity_member"],
        "caller/target capacity-member observation",
    )
    target_body = dict(target)
    final_body = dict(receipt_value)
    for body in (target_body, final_body):
        del body["transport"]
        del body["receipt_digest"]
    exact_value(final_body, target_body, "direct-node target/final receipt body")


def probe(
    phase: str, challenge: str, transport: Mapping[str, Any] | None = None
) -> bytes:
    validate_phase_challenge(phase, challenge)
    if socket.gethostname() != NODE:
        fail("physical hostname differs")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        fail("torch was imported before capacity admission")
    if lexists(PY_CACHE_PREFIX):
        fail("isolated pycache prefix already exists")
    before = executable_closure_snapshot()
    completed = subprocess.run(
        PRODUCER_ARGV, cwd="/", env=PRODUCER_ENV, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
    )
    observed_time_ns = time.time_ns()
    after = executable_closure_snapshot()
    if before != after or lexists(PY_CACHE_PREFIX):
        fail("producer closure or pycache prefix changed")
    if completed.returncode != 0 or completed.stderr != b"":
        fail(f"ROCm producer failed rc={completed.returncode} stderr_size={len(completed.stderr)}")
    return build_receipt(
        completed.stdout, phase, challenge, observed_time_ns, before, transport
    )


def direct_node_target_probe(phase: str, challenge: str) -> bytes:
    validate_phase_challenge(phase, challenge)
    if phase not in {"foreground", "controller"}:
        fail("direct-node target phase differs")
    before = remote_shell_boundary_snapshot()
    member_before = sealed_capacity_member_snapshot()
    raw = probe(phase, challenge, {
        "mode": "direct-node-ssh-target",
        "remote_shell_boundary": before,
        "remote_shell_boundary_stable_pre_and_post": True,
        "capacity_member": member_before,
        "capacity_member_stable_pre_and_post": True,
        "capacity_member_sha256_must_be_bound_by_launch_authority": True,
    })
    after = remote_shell_boundary_snapshot()
    member_after = sealed_capacity_member_snapshot()
    if before != after or member_before != member_after:
        fail("direct-node shell or capacity-member boundary changed during sample")
    return raw


def finalize_direct_node_receipt(
    target_raw: bytes, encoded_target: str, phase: str, challenge: str,
    ssh_snapshot: Mapping[str, Any], now_ns: int | None = None,
) -> bytes:
    current = time.time_ns() if now_ns is None else now_ns
    target = validate_receipt_bytes(
        target_raw, sha256_bytes(target_raw), phase, challenge, now_ns=current,
        _expected_transport_mode="direct-node-ssh-target",
    )
    argv = ssh_argv(phase, challenge)
    completed_stdout = encoded_target.encode("ascii")
    final = dict(target)
    final["transport"] = {
        "mode": "direct-node-ssh", "ssh_top_level_invocation_count": 1,
        "ssh_client": ssh_snapshot["ssh_client"],
        "sealed_known_hosts": ssh_snapshot["sealed_known_hosts"],
        "capacity_member": ssh_snapshot["capacity_member"],
        "ssh_argv": list(argv),
        "ssh_argv_sha256": sha256_bytes(
            b"\0".join(item.encode("utf-8") for item in argv)
        ),
        "ssh_environment": dict(SSH_ENV),
        "ssh_environment_inherited_from_caller": False,
        "ssh_exit_code": 0, "ssh_stderr_size": 0,
        "ssh_stderr_sha256": sha256_bytes(b""),
        "ssh_stdout_size": len(completed_stdout),
        "ssh_stdout_sha256": sha256_bytes(completed_stdout),
        "remote_target_receipt_base64": encoded_target,
        "remote_target_receipt_size": len(target_raw),
        "remote_target_receipt_sha256": sha256_bytes(target_raw),
        "ssh_client_and_known_hosts_stable_pre_and_post": True,
        "remote_target_transport_mode": "direct-node-ssh-target",
        "fresh_challenge_bound_to_ssh_argv": True,
        "same_user_ssh_authentication_and_login_startup_are_trusted_boundary": True,
        "capacity_member_sha256_must_be_bound_by_launch_authority": True,
        "ssh_system_trust_boundary": [
            "root-owned SSH client dynamic loader and shared libraries",
            "root-owned resolver and NSS stack",
            "kernel and network stack",
            "remote sshd, server configuration, and host private key",
        ],
    }
    del final["receipt_digest"]
    final["receipt_digest"] = sha256_bytes(canonical_json_bytes(final))
    raw = canonical_json_bytes(final)
    validate_receipt_bytes(raw, sha256_bytes(raw), phase, challenge, now_ns=current)
    return raw


def direct_node_probe(phase: str, challenge: str) -> bytes:
    validate_phase_challenge(phase, challenge)
    if phase not in {"foreground", "controller"}:
        fail("direct-node SSH phase differs")
    before = ssh_authority_snapshot()
    argv = ssh_argv(phase, challenge)
    completed = subprocess.run(
        argv, cwd="/", env=SSH_ENV, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
    )
    after = ssh_authority_snapshot()
    if before != after:
        fail("SSH client or sealed known-host authority changed")
    if completed.returncode != 0 or completed.stderr != b"":
        fail(
            f"direct-node SSH failed rc={completed.returncode} "
            f"stderr_size={len(completed.stderr)}"
        )
    if (
        not completed.stdout or len(completed.stdout) > 256 * 1024
        or b"\n" in completed.stdout or b"\r" in completed.stdout
        or b"\x00" in completed.stdout
    ):
        fail("direct-node SSH stdout framing differs")
    try:
        encoded_target = completed.stdout.decode("ascii")
    except UnicodeDecodeError as exc:
        fail(f"direct-node SSH stdout encoding differs: {exc}")
    target_raw = decode_canonical_base64(encoded_target, "direct-node target receipt")
    return finalize_direct_node_receipt(
        target_raw, encoded_target, phase, challenge, before
    )


def stable_receipt_file(path: Path, expected_sha: str) -> bytes:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        fail("receipt path is not one canonical plain file")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1 or before.st_size <= 0 or before.st_size > 128 * 1024
    ):
        fail("receipt topology differs")
    raw = path.read_bytes()
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
        before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
        after.st_size, after.st_mtime_ns, after.st_ctime_ns,
    ) or sha256_bytes(raw) != expected_sha:
        fail("receipt changed during read")
    return raw


def publish_receipt(raw: bytes, expected_sha: str, phase: str, challenge: str,
                    final_path: Path, now_ns: int | None = None) -> None:
    validate_receipt_bytes(
        raw, expected_sha, phase, challenge,
        now_ns=time.time_ns() if now_ns is None else now_ns,
    )
    expected_name = {
        "foreground": "foreground-capacity-receipt.json",
        "controller": "controller-capacity-receipt.json",
        "step": "step-capacity-receipt.json",
    }.get(phase)
    expected_parent = ATTEMPT_ROOT if phase == "foreground" else ATTEMPT_STARTED
    if expected_name is None or final_path != expected_parent / expected_name:
        fail("capacity receipt publication path differs")
    parent = final_path.parent
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        fail("capacity receipt parent is not canonical")
    parent_info = parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_IMODE(parent_info.st_mode) != 0o700:
        fail("capacity receipt parent topology differs")
    if lexists(final_path):
        fail("capacity receipt final path already exists")
    temporary = parent / f".{expected_name}.{challenge}.tmp"
    if lexists(temporary):
        fail("capacity receipt temporary path already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(temporary, flags, 0o444)
        os.fchmod(descriptor, 0o444)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                fail("capacity receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        temp_info = temporary.lstat()
        if (
            not stat.S_ISREG(temp_info.st_mode)
            or stat.S_IMODE(temp_info.st_mode) != 0o444
            or temp_info.st_nlink != 1
            or sha256_file(temporary) != expected_sha
        ):
            fail("capacity receipt temporary topology differs")
        os.link(temporary, final_path, follow_symlinks=False)
        linked = True
        os.unlink(temporary)
        linked = False
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        final_info = final_path.lstat()
        if (
            not stat.S_ISREG(final_info.st_mode)
            or stat.S_IMODE(final_info.st_mode) != 0o444
            or final_info.st_nlink != 1
            or sha256_file(final_path) != expected_sha
        ):
            fail("capacity receipt published topology differs")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not linked and lexists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args == ["challenge"]:
            sys.stdout.write(os.getrandom(32).hex())
            return 0
        if len(args) == 3 and args[0] == "probe-base64":
            if args[1] != "step":
                fail("local probe-base64 phase differs")
            sys.stdout.buffer.write(base64.b64encode(probe(args[1], args[2])))
            return 0
        if len(args) == 3 and args[0] == "probe":
            if args[1] != "step":
                fail("local probe phase differs")
            sys.stdout.buffer.write(probe(args[1], args[2]))
            return 0
        if len(args) == 3 and args[0] == "direct-node-target-base64":
            sys.stdout.buffer.write(
                base64.b64encode(direct_node_target_probe(args[1], args[2]))
            )
            return 0
        if len(args) == 3 and args[0] == "remote-probe-base64":
            sys.stdout.buffer.write(
                base64.b64encode(direct_node_probe(args[1], args[2]))
            )
            return 0
        if len(args) == 5 and args[0] == "validate-base64":
            raw = decode_canonical_base64(args[1], "receipt argument")
            validate_receipt_bytes(raw, args[2], args[3], args[4])
            sys.stdout.buffer.write(raw)
            return 0
        if len(args) == 5 and args[0] == "validate-file":
            raw = stable_receipt_file(Path(args[1]), args[2])
            validate_receipt_bytes(raw, args[2], args[3], args[4])
            sys.stdout.buffer.write(raw)
            return 0
        if len(args) == 5 and args[0] == "validate-file-archival":
            if args[3] != "step":
                fail("archival validation is restricted to step receipts")
            raw = stable_receipt_file(Path(args[1]), args[2])
            validate_receipt_bytes(
                raw, args[2], args[3], args[4], _enforce_max_age=False
            )
            sys.stdout.buffer.write(raw)
            return 0
        if len(args) == 6 and args[0] == "publish-base64":
            raw = decode_canonical_base64(args[1], "publication argument")
            publish_receipt(raw, args[2], args[3], args[4], Path(args[5]))
            sys.stdout.write(args[2])
            return 0
        fail("exact argv differs")
    except CapacityError as exc:
        print(f"Level-B v4 capacity refused: {exc}", file=sys.stderr)
        return 98
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Level-B v4 capacity refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 98


if __name__ == "__main__":
    raise SystemExit(main())
