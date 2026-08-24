#!/usr/bin/env python3
"""Linux CPU-only integration probe for the AUH r5d consumption fix.

The probe builds tiny, exact-shape model and adapter authorities, crosses a
real ``exec`` boundary with the production inherited-FD allowlist, excludes
both coordinator-private parent descriptors, deliberately reuses their
numbers in the child, and calls the frozen authority's
``load_consumption_input`` while the r5d rank patch is active.  The frozen
modules keep Torch lazy on this path; no Torch/model runtime, real model
weights, GPU, Slurm, SSH, runner, bridge, launcher, or adapter source is
changed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
import types
from typing import Any, Mapping, Sequence


SCHEMA = "full644-exploratory-matched-r5d-cpu-consumption-probe-v1"
CHILD_SCHEMA = "full644-exploratory-matched-r5d-cpu-consumption-child-v1"
INHERITED_ENV = "APV2_EVAL_INHERITED_AUTHORITY_FDS"
CHILD_TOKEN = "--_r5d-cpu-consumption-child"
SOURCE_CLOSURE_DIGEST = (
    "5bc55b2732ed1ae1c100c14434337967f7e4e343e82555508132fc3d4be71b9e"
)
SOURCE_SPECS = {
    "r5d_adapter": (
        "full644_exploratory_matched_infer_adapter_auh_r5d.py",
        "5794e1f0e5ecb84ffdb37f618fe63696ee4f87176952ac083c8c91792a9d192a",
        25_854,
    ),
    "base_adapter": (
        "full644_exploratory_matched_infer_adapter_v2.py",
        "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca",
        123_645,
    ),
    "model_authority": (
        "action_preservation_decoded_eval_model_authority_v2.py",
        "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
        115_209,
    ),
    "infer_lora": (
        "infer_lora.py",
        "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
        177_300,
    ),
    "train_lora": (
        "train_lora.py",
        "ead547b8309e1b5ae5c831444e9f5d1d8e1785fed5fe39cf7b97f13f82a9ce85",
        157_494,
    ),
    "self_generated": (
        "self_generated_action_preservation_v2.py",
        "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c",
        11_334,
    ),
    "build_renderer_dataset": (
        "tools/build_renderer_dataset.py",
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
        31_012,
    ),
    "materialize_vae": (
        "tools/materialize_vae.py",
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
        32_195,
    ),
}
R5D_MODULE_NAME = "_full644_r5d_cpu_consumption_probe_adapter"
_PROBE_CODE_AUTHORITY: dict[str, Any] | None = None
_PROBE_CAPTURED_SOURCE: str | None = None
_COMGR_STDERR_LINE = re.compile(
    r"Unbundle Objects Error: '/[^'\n]+/comgr-[0-9A-Za-z_-]+/output/"
    r"hipfatbin-hipv4-amdgcn-amd-amdhsa--gfx[0-9A-Za-z:+_.-]+\.o': "
    r"Invalid argument"
)


class R5DCPUConsumptionProbeError(RuntimeError):
    """The isolated CPU consumption proof differed from its closed contract."""


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
        raise R5DCPUConsumptionProbeError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def bytes_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def _validate_isolated_linux() -> None:
    if (
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
        raise R5DCPUConsumptionProbeError(
            "probe requires Linux python -I -S -B with no customization"
        )


def gpu_device_descriptors() -> list[str]:
    rows: list[str] = []
    root = Path("/proc/self/fd")
    if not root.is_dir():
        return rows
    for item in root.iterdir():
        try:
            target = os.readlink(item)
        except OSError:
            continue
        if target == "/dev/kfd" or target.startswith("/dev/dri/"):
            rows.append(target)
    return sorted(rows)


def unexpected_weight_descriptors(binding: Mapping[str, Any]) -> list[str]:
    allowed = {
        item["fd"]
        for item in binding["fd_rows"]
        if item["scope"] in {"model", "adapter"}
        and item["role"] == "file"
    }
    rows: list[str] = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            descriptor = int(entry.name)
            target = os.readlink(entry)
        except (OSError, ValueError):
            continue
        if target.endswith((".safetensors", ".bin", ".pt", ".pth")) and descriptor not in allowed:
            rows.append(f"{descriptor}:{target}")
    return sorted(rows)


def classify_child_stderr(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {
            "kind": "empty",
            "line_count": 0,
            "sha256": bytes_sha256(raw),
            "traceback_runtime_oom_count": 0,
        }
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise R5DCPUConsumptionProbeError("child stderr is not UTF-8") from error
    lines = text.splitlines()
    forbidden = ("Traceback", "RuntimeError", "OutOfMemory", "out of memory", "Killed", "Segmentation fault")
    if (
        not text.endswith("\n")
        or "\r" in text
        or not lines
        or len(lines) > 4096
        or any(token in text for token in forbidden)
        or any(_COMGR_STDERR_LINE.fullmatch(line) is None for line in lines)
    ):
        raise R5DCPUConsumptionProbeError("child stderr is not exact ROCm COMGR noise")
    return {
        "kind": "rocm-comgr-unbundle-invalid-argument",
        "line_count": len(lines),
        "sha256": bytes_sha256(raw),
        "traceback_runtime_oom_count": 0,
    }


def _stable_source(path: Path, expected_sha256: str, expected_size: int) -> tuple[str, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
        or before.st_size != expected_size
        or stat_identity(before) != stat_identity(after)
        or stat_identity(before) != stat_identity(named)
        or bytes_sha256(raw) != expected_sha256
    ):
        raise R5DCPUConsumptionProbeError(f"sealed source differs: {path}")
    try:
        source = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise R5DCPUConsumptionProbeError(f"source is not UTF-8: {path}") from error
    return source, {
        "path": str(path),
        "sha256": expected_sha256,
        "size": expected_size,
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": before.st_nlink,
    }


def _pread_sha256(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            raise R5DCPUConsumptionProbeError("retained probe source short read")
        digest.update(block)
        offset += len(block)
    if os.pread(descriptor, 1, size):
        raise R5DCPUConsumptionProbeError("retained probe source grew")
    return digest.hexdigest()


def open_probe_code_authority(path: Path, expected_sha256: str) -> tuple[str, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset
            )
            if not block:
                raise R5DCPUConsumptionProbeError(
                    "retained probe source short capture"
                )
            chunks.append(block)
            offset += len(block)
        if os.pread(descriptor, 1, before.st_size):
            raise R5DCPUConsumptionProbeError("retained probe source grew")
        raw = b"".join(chunks)
        digest = bytes_sha256(raw)
        after = os.fstat(descriptor)
        named = path.lstat()
        identity = stat_identity(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
            or identity != stat_identity(after)
            or identity != stat_identity(named)
            or digest != expected_sha256
            or os.get_inheritable(descriptor)
        ):
            raise R5DCPUConsumptionProbeError("retained probe source differs")
        try:
            source = raw.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise R5DCPUConsumptionProbeError(
                "retained probe source is not UTF-8"
            ) from error
        return source, {
            "fd": descriptor,
            "named_path": str(path),
            "sha256": digest,
            "size": before.st_size,
            "identity": identity,
            "mode": stat.S_IMODE(before.st_mode),
            "nlink": before.st_nlink,
        }
    except BaseException:
        os.close(descriptor)
        raise


def _replay_child_probe_code(binding_fds: Sequence[int], private_fds: Sequence[int]) -> dict[str, Any]:
    try:
        row = json.loads(os.environ["R5D_PROBE_CODE_AUTHORITY"])
    except (KeyError, TypeError, ValueError) as error:
        raise R5DCPUConsumptionProbeError("child probe-code authority differs") from error
    if not isinstance(row, dict) or set(row) != {
        "fd", "named_path", "sha256", "size", "identity", "mode", "nlink"
    }:
        raise R5DCPUConsumptionProbeError("child probe-code fields differ")
    descriptor = row["fd"]
    try:
        named = Path(row["named_path"]).lstat()
    except (OSError, TypeError, ValueError) as error:
        raise R5DCPUConsumptionProbeError(
            "child retained probe-code named replay differs"
        ) from error
    if (
        type(descriptor) is not int
        or descriptor < 3
        or descriptor in binding_fds
        or descriptor in private_fds
        or stat_identity(os.fstat(descriptor)) != row["identity"]
        or stat_identity(named) != row["identity"]
        or stat.S_IMODE(named.st_mode) != row["mode"]
        or named.st_nlink != row["nlink"]
        or _pread_sha256(descriptor, row["size"]) != row["sha256"]
    ):
        raise R5DCPUConsumptionProbeError("child retained probe-code replay differs")
    os.set_inheritable(descriptor, False)
    return row


def _replay_parent_probe_code(row: Mapping[str, Any]) -> None:
    descriptor = row["fd"]
    try:
        named = Path(row["named_path"]).lstat()
    except (OSError, TypeError, ValueError) as error:
        raise R5DCPUConsumptionProbeError(
            "parent retained probe-code named replay differs"
        ) from error
    if (
        stat_identity(os.fstat(descriptor)) != row["identity"]
        or stat_identity(named) != row["identity"]
        or stat.S_IMODE(named.st_mode) != row["mode"]
        or named.st_nlink != row["nlink"]
        or _pread_sha256(descriptor, row["size"]) != row["sha256"]
        or os.get_inheritable(descriptor)
    ):
        raise R5DCPUConsumptionProbeError("parent retained probe-code replay differs")


def build_child_command(
    captured_source: str, expected_sha256: str, optimize: int
) -> list[str]:
    try:
        raw = captured_source.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise R5DCPUConsumptionProbeError(
            "captured probe source is not strict UTF-8"
        ) from error
    if (
        optimize not in (0, 1)
        or not raw
        or len(raw) > 120 * 1024
        or bytes_sha256(raw) != expected_sha256
    ):
        raise R5DCPUConsumptionProbeError("captured child source authority differs")
    return [
        "/proc/self/exe",
        "-I",
        "-S",
        "-B",
        *(["-O"] if optimize else []),
        "-c",
        captured_source,
        CHILD_TOKEN,
    ]


def proc_self_executable_authority() -> dict[str, Any]:
    try:
        info = os.stat("/proc/self/exe", follow_symlinks=True)
    except OSError as error:
        raise R5DCPUConsumptionProbeError("/proc/self/exe is unavailable") from error
    if not stat.S_ISREG(info.st_mode):
        raise R5DCPUConsumptionProbeError("/proc/self/exe type differs")
    return {"proc_path": "/proc/self/exe", "identity": stat_identity(info)}


def _replay_child_executable() -> dict[str, Any]:
    try:
        expected = json.loads(os.environ["R5D_PROBE_PYTHON_EXEC_AUTHORITY"])
    except (KeyError, TypeError, ValueError) as error:
        raise R5DCPUConsumptionProbeError("child Python authority differs") from error
    observed = proc_self_executable_authority()
    if expected != observed:
        raise R5DCPUConsumptionProbeError("child /proc/self/exe identity differs")
    return observed


def validate_source_closure(methods_root: str | Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    root = Path(methods_root)
    if (
        not root.is_absolute()
        or os.path.normpath(str(root)) != str(root)
        or root.resolve(strict=True) != root
        or root.is_symlink()
        or not root.is_dir()
        or stat.S_IMODE(root.stat().st_mode) != 0o555
    ):
        raise R5DCPUConsumptionProbeError("production methods root differs")
    rows: dict[str, dict[str, Any]] = {}
    closure_rows: list[dict[str, Any]] = []
    for role, (relative, digest, size) in sorted(SOURCE_SPECS.items()):
        _, row = _stable_source(root / relative, digest, size)
        rows[role] = {"relative_path": relative, **row}
        closure_rows.append(
            {
                "role": role,
                "relative_path": relative,
                "sha256": digest,
                "size": size,
                "mode": 0o444,
                "nlink": 1,
            }
        )
    if object_sha256(closure_rows) != SOURCE_CLOSURE_DIGEST:
        raise R5DCPUConsumptionProbeError("production source closure digest differs")
    return root, rows


def load_production_sources(
    methods_root: str | Path, site_packages_root: str | Path
) -> tuple[Any, Any, dict[str, Any]]:
    root, rows = validate_source_closure(methods_root)
    site_root = Path(site_packages_root)
    if (
        not site_root.is_absolute()
        or os.path.normpath(str(site_root)) != str(site_root)
        or site_root.resolve(strict=True) != site_root
        or site_root.is_symlink()
        or not site_root.is_dir()
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
        or any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    ):
        raise R5DCPUConsumptionProbeError("site-packages authority differs")
    forbidden = {
        R5D_MODULE_NAME,
        "_full644_exploratory_matched_infer_adapter_r5c_base",
        "action_preservation_decoded_eval_model_authority_v2",
        "train_lora",
        "infer_lora",
        "self_generated_action_preservation_v2",
        "tools",
        "tools.build_renderer_dataset",
        "tools.materialize_vae",
    }
    if forbidden.intersection(sys.modules) or any(
        value and Path(value).resolve(strict=False) == root for value in sys.path
    ):
        raise R5DCPUConsumptionProbeError("production source was importable before source-only gate")
    sys.path.append(str(site_root))
    if sys.path[-1] != str(site_root):
        raise R5DCPUConsumptionProbeError("site-packages activation differs")
    r5d_path = root / SOURCE_SPECS["r5d_adapter"][0]
    source, _ = _stable_source(r5d_path, SOURCE_SPECS["r5d_adapter"][1], SOURCE_SPECS["r5d_adapter"][2])
    module = types.ModuleType(R5D_MODULE_NAME)
    module.__file__ = str(r5d_path)
    module.__package__ = ""
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(R5D_MODULE_NAME, loader=None, origin=str(r5d_path))
    sys.modules[R5D_MODULE_NAME] = module
    path_snapshot = list(sys.path)
    try:
        exec(compile(source, str(r5d_path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(R5D_MODULE_NAME, None)
        raise
    finally:
        sys.path[:] = path_snapshot
    authority = module.model_authority
    expected_origins = {
        R5D_MODULE_NAME: root / SOURCE_SPECS["r5d_adapter"][0],
        "_full644_exploratory_matched_infer_adapter_r5c_base": root / SOURCE_SPECS["base_adapter"][0],
        "action_preservation_decoded_eval_model_authority_v2": root / SOURCE_SPECS["model_authority"][0],
        "infer_lora": root / SOURCE_SPECS["infer_lora"][0],
        "train_lora": root / SOURCE_SPECS["train_lora"][0],
        "self_generated_action_preservation_v2": root / SOURCE_SPECS["self_generated"][0],
        "tools.build_renderer_dataset": root / SOURCE_SPECS["build_renderer_dataset"][0],
        "tools.materialize_vae": root / SOURCE_SPECS["materialize_vae"][0],
    }
    for name, expected in expected_origins.items():
        loaded = sys.modules.get(name)
        if (
            loaded is None
            or getattr(loaded, "__cached__", None) is not None
            or getattr(loaded, "__file__", None) is None
            or Path(loaded.__file__).resolve(strict=True) != expected
        ):
            raise R5DCPUConsumptionProbeError(f"source-only origin differs: {name}")
    if (
        module.base.model_authority is not authority
        or module.base.infer_lora is not sys.modules["infer_lora"]
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
        or any(value and Path(value).resolve(strict=False) == root for value in sys.path)
    ):
        raise R5DCPUConsumptionProbeError("source origin/sitecustomize gate differs")
    return module, authority, {
        "methods_root": str(root),
        "site_packages_root": str(site_root),
        "source_closure_digest": SOURCE_CLOSURE_DIGEST,
        "sources": rows,
        "source_only_origins_verified": True,
        "sitecustomize_absent": True,
    }


def write_create_only_canonical(path: str | Path, value: Mapping[str, Any], *, mode: int = 0o400) -> dict[str, Any]:
    target = Path(path)
    if not target.is_absolute() or target.name in ("", ".", "..") or target.exists():
        raise R5DCPUConsumptionProbeError("create-only output path differs")
    raw = canonical_bytes(dict(value)) + b"\n"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise R5DCPUConsumptionProbeError("create-only output short write")
            offset += written
        os.fsync(descriptor)
        current = os.fstat(descriptor)
        named = target.lstat()
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_IMODE(current.st_mode) != 0o600
            or current.st_nlink != 1
            or stat_identity(current) != stat_identity(named)
            or os.pread(descriptor, len(raw), 0) != raw
        ):
            raise R5DCPUConsumptionProbeError("create-only output replay differs")
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    return {"path": str(target), "sha256": bytes_sha256(raw), "size": len(raw), "mode": mode}


def write_receipt(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt, reference = prepare_receipt(path, payload)
    row = write_create_only_canonical(path, receipt, mode=0o400)
    if row != {key: reference[key] for key in ("path", "sha256", "size", "mode")}:
        raise R5DCPUConsumptionProbeError("receipt commit reference differs")
    return reference


def prepare_receipt(
    path: str | Path, payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    unsigned = dict(payload)
    if "receipt_digest" in unsigned:
        raise R5DCPUConsumptionProbeError("receipt digest is caller-supplied")
    receipt = dict(unsigned)
    receipt["receipt_digest"] = object_sha256(unsigned)
    raw = canonical_bytes(receipt) + b"\n"
    reference = {
        "path": str(Path(path)),
        "sha256": bytes_sha256(raw),
        "size": len(raw),
        "mode": 0o400,
        "receipt_digest": receipt["receipt_digest"],
    }
    return receipt, reference


def commit_receipt_and_exit(
    path: str | Path,
    receipt: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    """Make fchmod(0400) the only success commit, then exit without cleanup."""

    target = Path(path)
    raw = canonical_bytes(dict(receipt)) + b"\n"
    expected_reference = {
        "path": str(target),
        "sha256": bytes_sha256(raw),
        "size": len(raw),
        "mode": 0o400,
        "receipt_digest": receipt["receipt_digest"],
    }
    if (
        dict(reference) != expected_reference
        or not target.is_absolute()
        or target.name in ("", ".", "..")
        or target.exists()
    ):
        raise R5DCPUConsumptionProbeError("final receipt reference differs")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(target, flags, 0o000)
    committed = False
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise R5DCPUConsumptionProbeError("final receipt short write")
            offset += written
        os.fsync(descriptor)
        current = os.fstat(descriptor)
        named = target.lstat()
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_IMODE(current.st_mode) != 0o000
            or current.st_nlink != 1
            or stat_identity(current) != stat_identity(named)
            or current.st_size != len(raw)
            or os.pread(descriptor, len(raw), 0) != raw
            or os.pread(descriptor, 1, len(raw))
        ):
            raise R5DCPUConsumptionProbeError("final receipt precommit replay differs")
        parent_fd = _open_directory(target.parent)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        sys.stdout.buffer.write(canonical_bytes(expected_reference) + b"\n")
        sys.stdout.buffer.flush()
        exit_now = os._exit
        committed = True
        os.fchmod(descriptor, 0o400)
        exit_now(0)
    finally:
        if not committed:
            os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods-root", required=True)
    parser.add_argument("--site-packages-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--probe-sha256", required=True)
    return parser


def _write_fixture_file(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise R5DCPUConsumptionProbeError("fixture short write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _open_directory(path: Path) -> int:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    os.set_inheritable(descriptor, False)
    return descriptor


def _write_member(
    authority: Any, root_fd: int, name: str, value: Mapping[str, Any]
) -> tuple[str, str]:
    raw = authority.canonical_json_bytes(dict(value)) + b"\n"
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
        dir_fd=root_fd,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise R5DCPUConsumptionProbeError("task receipt short write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)
    return f"/proc/self/fd/{root_fd}/{name}", bytes_sha256(raw)


def _build_model(authority: Any, state: Path, view_parent: Path, parent_fd: int) -> Any:
    root = state / "fake-model"
    rows: list[str] = []
    for index, relative in enumerate(authority.MODEL_RELATIVE_FILES):
        raw = f"r5d-cpu-model:{index}:{relative}\n".encode("utf-8")
        _write_fixture_file(root / relative, raw)
        rows.append(f"{bytes_sha256(raw)}  ./{relative}")
    for relative in authority.MODEL_RELATIVE_DIRECTORIES:
        (root if relative == "." else root / relative).chmod(0o700)
    manifest = state / "fake-model.sha256"
    _write_fixture_file(manifest, ("\n".join(rows) + "\n").encode("utf-8"))
    return authority.ModelAuthority.capture(
        model_root=root,
        manifest_path=manifest,
        private_parent=view_parent,
        private_parent_fd=parent_fd,
        view_name="model-view",
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
        expected_device=None,
        expected_manifest_sha256=bytes_sha256(manifest.read_bytes()),
        expected_file_mode=0o600,
    )


def _build_adapter(
    authority: Any, state: Path, view_parent: Path, parent_fd: int, task_id: str
) -> Any:
    root = state / "fake-adapter"
    payloads = {
        "receipt.json": b'{"fixture":"r5d-cpu"}\n',
        "adapter/README.md": b"r5d cpu fixture\n",
        "adapter/adapter_config.json": b'{"peft_type":"LORA"}\n',
        "adapter/adapter_model.safetensors": b"r5d-cpu-adapter",
        "optimizer.pt": b"not-consumed",
    }
    expected: dict[str, str] = {}
    for relative, raw in payloads.items():
        _write_fixture_file(root / relative, raw)
        if relative != "optimizer.pt":
            expected[relative] = bytes_sha256(raw)
    root.chmod(0o700)
    (root / "adapter").chmod(0o700)
    return authority.AdapterAuthority.capture(
        task_id=task_id,
        checkpoint_root=root,
        expected_sha256=expected,
        private_parent=view_parent,
        private_parent_fd=parent_fd,
        view_name="adapter-view",
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
        expected_file_mode=0o600,
    )


def _resign_binding(authority: Any, value: Mapping[str, Any]) -> dict[str, Any]:
    row = json.loads(canonical_bytes(value).decode("utf-8"))
    row["fd_rows"] = sorted(row["fd_rows"], key=lambda item: item["fd"])
    row["fd_count"] = len(row["fd_rows"])
    row["fd_rows_digest"] = authority.object_sha256(row["fd_rows"])
    row.pop("fd_binding_digest", None)
    row["fd_binding_digest"] = authority.object_sha256(row)
    return row


def _resign_input(authority: Any, value: Mapping[str, Any]) -> dict[str, Any]:
    row = json.loads(canonical_bytes(value).decode("utf-8"))
    row.pop("consumption_input_digest", None)
    row["consumption_input_digest"] = authority.object_sha256(row)
    return row


def _replace_binding_row(
    authority: Any,
    binding: Mapping[str, Any],
    *,
    scope: str,
    role: str,
    relative_path: str,
    descriptor: int,
    source_path: Path,
) -> dict[str, Any]:
    value = json.loads(canonical_bytes(binding).decode("utf-8"))
    matches = [
        item for item in value["fd_rows"]
        if item["scope"] == scope
        and item["role"] == role
        and item["relative_path"] == relative_path
    ]
    if len(matches) != 1:
        raise R5DCPUConsumptionProbeError("hostile binding row differs")
    matches[0].update(
        {
            "fd": descriptor,
            "source_path": str(source_path),
            "identity": stat_identity(os.fstat(descriptor)),
        }
    )
    return _resign_binding(authority, value)


def _reuse_private_numbers(numbers: Sequence[int]) -> list[dict[str, Any]]:
    targets = sorted(set(numbers))
    source = os.open("/dev/null", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        keeper = fcntl.fcntl(source, fcntl.F_DUPFD_CLOEXEC, max(targets) + 1)
    finally:
        os.close(source)
    try:
        for target in targets:
            os.dup2(keeper, target, inheritable=False)
        rows = [
            {
                "fd": target,
                "target": os.readlink(f"/proc/self/fd/{target}"),
                "identity": stat_identity(os.fstat(target)),
            }
            for target in targets
        ]
        if any(
            row["target"] != "/dev/null"
            or not stat.S_ISCHR(row["identity"]["mode"])
            or os.get_inheritable(row["fd"])
            for row in rows
        ):
            raise R5DCPUConsumptionProbeError(
                "private-parent FD number reuse differs"
            )
        return rows
    finally:
        os.close(keeper)


def _validate_private_parent_roles(
    binding: Any, roles: Any, arm: Any
) -> dict[str, int]:
    arm_roles = {
        "base": {"model"},
        "full": {"model", "adapter"},
        "hostile-digest": {"model", "adapter"},
        "hostile-task": {"model", "adapter"},
        "hostile-adapter-namespace": {"model", "adapter"},
        "hostile-adapter-leaf": {"model", "adapter"},
    }
    if (
        not isinstance(binding, dict)
        or not isinstance(binding.get("fd_rows"), list)
        or arm not in arm_roles
        or not isinstance(roles, dict)
    ):
        raise R5DCPUConsumptionProbeError("private-parent role binding differs")
    adapter_bound = binding.get("adapter_capture_digest") is not None
    adapter_scoped = any(
        isinstance(item, dict) and item.get("scope") == "adapter"
        for item in binding["fd_rows"]
    )
    expected = {"model"} | ({"adapter"} if adapter_bound else set())
    if (
        adapter_bound is not adapter_scoped
        or expected != arm_roles[arm]
        or set(roles) != expected
        or any(type(value) is not int or value < 3 for value in roles.values())
    ):
        raise R5DCPUConsumptionProbeError("private-parent role binding differs")
    return dict(roles)


def _child_main() -> int:
    _validate_isolated_linux()
    executable_authority = _replay_child_executable()
    raw_binding = os.environ[INHERITED_ENV]
    try:
        binding = json.loads(raw_binding)
        private_roles = json.loads(os.environ["R5D_PROBE_PRIVATE_PARENT_ROLES"])
    except (ValueError, TypeError, KeyError) as error:
        raise R5DCPUConsumptionProbeError("child environment differs") from error
    private_roles = _validate_private_parent_roles(
        binding, private_roles, os.environ.get("R5D_PROBE_ARM")
    )
    private_fds = sorted(set(private_roles.values()))
    if canonical_bytes(binding).decode("utf-8") != raw_binding:
        raise R5DCPUConsumptionProbeError("child inherited binding is not canonical")
    inherited = tuple(item["fd"] for item in binding["fd_rows"])
    if tuple(sorted(inherited)) != inherited or set(inherited).intersection(private_fds):
        raise R5DCPUConsumptionProbeError("private parent entered child allowlist")
    code_authority = _replay_child_probe_code(inherited, private_fds)
    for descriptor in inherited:
        os.fstat(descriptor)
        os.set_inheritable(descriptor, False)
    reused = _reuse_private_numbers(private_fds)
    before_gpu = gpu_device_descriptors()
    r5d, authority, source = load_production_sources(
        os.environ["R5D_PROBE_METHODS_ROOT"],
        os.environ["R5D_PROBE_SITE_PACKAGES_ROOT"],
    )
    outcome = "SUCCESS"
    error_text: str | None = None
    model_digest: str | None = None
    adapter_digest: str | None = None
    try:
        with r5d.patched_rank_validation():
            _, model, adapter = authority.load_consumption_input(
                os.environ["R5D_PROBE_INPUT_PATH"],
                expected_sha256=os.environ["R5D_PROBE_INPUT_SHA256"],
                expected_digest=os.environ["R5D_PROBE_INPUT_DIGEST"],
                verify_views=True,
            )
        model_digest = model["capture_digest"]
        adapter_digest = None if adapter is None else adapter["capture_digest"]
    except authority.ModelConsumptionAuthorityError as error:
        outcome = "REJECTED"
        error_text = str(error)
    after_gpu = gpu_device_descriptors()
    unexpected_weights = unexpected_weight_descriptors(binding)
    torch = sys.modules.get("torch")
    forbidden_runtime_modules = sorted(
        name
        for name in sys.modules
        if name in {"peft", "diffusers", "transformers", "bernini", "vace", "veomni"}
        or any(
            name.startswith(prefix + ".")
            for prefix in ("peft", "diffusers", "transformers", "bernini", "vace", "veomni")
        )
    )
    if (
        before_gpu
        or after_gpu
        or torch is not None
        or forbidden_runtime_modules
        or unexpected_weights
    ):
        raise R5DCPUConsumptionProbeError("GPU/model runtime entered CPU probe")
    result = {
        "schema_version": CHILD_SCHEMA,
        "arm": os.environ["R5D_PROBE_ARM"],
        "outcome": outcome,
        "error": error_text,
        "model_capture_digest": model_digest,
        "adapter_capture_digest": adapter_digest,
        "inherited_fds": list(inherited),
        "private_parent_fds": private_fds,
        "private_parent_role_fds": private_roles,
        "private_parent_fd_reuse": reused,
        "private_parent_fds_excluded": True,
        "source_closure_digest": source["source_closure_digest"],
        "source_origin_gate": True,
        "sitecustomize_absent": True,
        "imported_as_module_not_production_main": True,
        "executed_from_parent_verified_captured_source": True,
        "held_probe_code_fd_is_publication_replay_only": True,
        "probe_source_sha256": code_authority["sha256"],
        "executed_via_proc_self_exe": True,
        "python_executable_authority": executable_authority,
        "gpu_device_descriptors_before": before_gpu,
        "gpu_device_descriptors_after": after_gpu,
        "torch_imported": False,
        "torch_cuda_initialized": False,
        "torch_distributed_initialized": False,
        "model_or_peft_runtime_modules": forbidden_runtime_modules,
        "unexpected_model_weight_descriptors": unexpected_weights,
        "production_model_weights_loaded": False,
    }
    sys.stdout.buffer.write(canonical_bytes(result) + b"\n")
    return 0


def _spawn_child(
    *,
    binding: Mapping[str, Any],
    private_parent_roles: Mapping[str, int],
    methods_root: Path,
    site_root: Path,
    input_path: str,
    input_sha256: str,
    input_digest: str,
    arm: str,
) -> dict[str, Any]:
    if _PROBE_CODE_AUTHORITY is None or _PROBE_CAPTURED_SOURCE is None:
        raise R5DCPUConsumptionProbeError("parent probe-code authority is absent")
    code_fd = _PROBE_CODE_AUTHORITY["fd"]
    _replay_parent_probe_code(_PROBE_CODE_AUTHORITY)
    python_authority = proc_self_executable_authority()
    private_fds = sorted(set(private_parent_roles.values()))
    inherited = tuple(item["fd"] for item in binding["fd_rows"])
    if (
        tuple(sorted(inherited)) != inherited
        or set(inherited).intersection(private_fds)
        or code_fd in inherited
        or code_fd in private_fds
    ):
        raise R5DCPUConsumptionProbeError("child pass_fds closure differs")
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        INHERITED_ENV: canonical_bytes(binding).decode("utf-8"),
        "R5D_PROBE_PRIVATE_PARENT_ROLES": canonical_bytes(dict(private_parent_roles)).decode("utf-8"),
        "R5D_PROBE_METHODS_ROOT": str(methods_root),
        "R5D_PROBE_SITE_PACKAGES_ROOT": str(site_root),
        "R5D_PROBE_INPUT_PATH": input_path,
        "R5D_PROBE_INPUT_SHA256": input_sha256,
        "R5D_PROBE_INPUT_DIGEST": input_digest,
        "R5D_PROBE_ARM": arm,
        "R5D_PROBE_CODE_AUTHORITY": canonical_bytes(_PROBE_CODE_AUTHORITY).decode("utf-8"),
        "R5D_PROBE_PYTHON_EXEC_AUTHORITY": canonical_bytes(python_authority).decode("utf-8"),
    }
    command = build_child_command(
        _PROBE_CAPTURED_SOURCE,
        _PROBE_CODE_AUTHORITY["sha256"],
        sys.flags.optimize,
    )
    completed = subprocess.run(
        command,
        check=False,
        executable="/proc/self/exe",
        close_fds=True,
        pass_fds=tuple(sorted((*inherited, code_fd))),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc_self_executable_authority() != python_authority:
        raise R5DCPUConsumptionProbeError("parent /proc/self/exe identity changed")
    _replay_parent_probe_code(_PROBE_CODE_AUTHORITY)
    if completed.returncode != 0:
        raise R5DCPUConsumptionProbeError(
            "child failed: " + completed.stderr.decode("utf-8", "replace")
        )
    try:
        result = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise R5DCPUConsumptionProbeError("child stdout is not JSON") from error
    if completed.stdout != canonical_bytes(result) + b"\n" or result.get("schema_version") != CHILD_SCHEMA:
        raise R5DCPUConsumptionProbeError("child stdout contract differs")
    stderr_attestation = classify_child_stderr(completed.stderr)
    if stderr_attestation["kind"] != "empty":
        raise R5DCPUConsumptionProbeError(
            "child emitted stderr on the Torch-lazy consumption path"
        )
    result["stderr_attestation"] = stderr_attestation
    return result


def _build_consumption(
    *,
    authority: Any,
    model: Any,
    adapter: Any | None,
    task_id: str,
    task_fd: int,
    task_root: Path,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    model_pre = model.begin_task(task_id)
    adapter_pre = None if adapter is None else adapter.begin_use()
    model_path, model_sha = _write_member(
        authority, task_fd, f"{prefix}-model-capture.json", model.capture_receipt
    )
    adapter_path: str | None = None
    adapter_sha: str | None = None
    if adapter is not None:
        adapter_path, adapter_sha = _write_member(
            authority,
            task_fd,
            f"{prefix}-adapter-capture.json",
            adapter.capture_receipt,
        )
    task_binding = authority.task_publication_root_binding(
        descriptor=task_fd, path=task_root
    )
    binding = authority.build_inherited_fd_binding(
        task_id=task_id,
        model_capture=model.capture_receipt,
        adapter_capture=None if adapter is None else adapter.capture_receipt,
        task_publication_root=task_binding,
    )
    consumption = authority.build_consumption_input(
        task_id=task_id,
        physical_bindings_digest=object_sha256({"fixture": task_id}),
        model_capture=model.capture_receipt,
        model_pre_use=model_pre,
        model_capture_receipt_path=model_path,
        model_capture_receipt_sha256=model_sha,
        adapter_capture=None if adapter is None else adapter.capture_receipt,
        adapter_pre_use=adapter_pre,
        adapter_capture_receipt_path=adapter_path,
        adapter_capture_receipt_sha256=adapter_sha,
        inherited_fd_binding=binding,
        task_publication_root=task_binding,
        production_mode=True,
    )
    input_path, input_sha = _write_member(
        authority, task_fd, f"{prefix}-consumption-input.json", consumption
    )
    return binding, consumption, input_path, input_sha


def _private_parent_roles(model: Any, adapter: Any | None) -> dict[str, int]:
    values = {
        "model": model.capture_receipt["private_parent"]["authority_fd"]
    }
    if adapter is not None:
        values["adapter"] = adapter.capture_receipt["private_parent"]["authority_fd"]
    return values


def _require_child(
    row: Mapping[str, Any], *, outcome: str, error_contains: str | None
) -> dict[str, Any]:
    result = dict(row)
    if result.get("outcome") != outcome:
        raise R5DCPUConsumptionProbeError(
            f"child outcome differs: {result.get('outcome')} {result.get('error')}"
        )
    error = result.get("error")
    if error_contains is None:
        if error is not None:
            raise R5DCPUConsumptionProbeError("successful child reported an error")
    elif not isinstance(error, str) or error_contains not in error:
        raise R5DCPUConsumptionProbeError(
            f"child rejection differs: {error!r}"
        )
    return result


def run_probe(
    *, methods_root: Path, site_root: Path, state: Path
) -> dict[str, Any]:
    if gpu_device_descriptors():
        raise R5DCPUConsumptionProbeError("GPU descriptor was open before probe")
    r5d, authority, source_authority = load_production_sources(
        methods_root, site_root
    )
    if r5d.__name__ == "__main__":
        raise R5DCPUConsumptionProbeError("r5d unexpectedly used production main")
    view_parent = state / "views"
    task_root = state / "tasks"
    view_parent.mkdir(mode=0o700)
    task_root.mkdir(mode=0o700)
    view_parent_fd = _open_directory(view_parent)
    task_fd = _open_directory(task_root)
    model = None
    adapter = None
    model_closed = False
    adapter_closed = False
    hostile_fds: list[int] = []
    try:
        model = _build_model(authority, state, view_parent, view_parent_fd)
        base_binding, base_input, base_path, base_sha = _build_consumption(
            authority=authority,
            model=model,
            adapter=None,
            task_id="r5d-cpu-base",
            task_fd=task_fd,
            task_root=task_root,
            prefix="base",
        )
        base_child = _require_child(
            _spawn_child(
                binding=base_binding,
                private_parent_roles=_private_parent_roles(model, None),
                methods_root=methods_root,
                site_root=site_root,
                input_path=base_path,
                input_sha256=base_sha,
                input_digest=base_input["consumption_input_digest"],
                arm="base",
            ),
            outcome="SUCCESS",
            error_contains=None,
        )
        model.end_task("r5d-cpu-base")
        model.record_task_consumption(base_input["consumption_input_digest"])

        adapter = _build_adapter(
            authority, state, view_parent, view_parent_fd, "r5d-cpu-full"
        )
        full_binding, full_input, full_path, full_sha = _build_consumption(
            authority=authority,
            model=model,
            adapter=adapter,
            task_id="r5d-cpu-full",
            task_fd=task_fd,
            task_root=task_root,
            prefix="full",
        )
        private_parent_roles = _private_parent_roles(model, adapter)
        full_child = _require_child(
            _spawn_child(
                binding=full_binding,
                private_parent_roles=private_parent_roles,
                methods_root=methods_root,
                site_root=site_root,
                input_path=full_path,
                input_sha256=full_sha,
                input_digest=full_input["consumption_input_digest"],
                arm="full",
            ),
            outcome="SUCCESS",
            error_contains=None,
        )

        digest_hostile = _require_child(
            _spawn_child(
                binding=full_binding,
                private_parent_roles=private_parent_roles,
                methods_root=methods_root,
                site_root=site_root,
                input_path=full_path,
                input_sha256=full_sha,
                input_digest="0" * 64,
                arm="hostile-digest",
            ),
            outcome="REJECTED",
            error_contains="literal digest differs",
        )

        task_binding = json.loads(canonical_bytes(full_binding).decode("utf-8"))
        task_binding["task_id"] = "r5d-cpu-hostile-task"
        task_binding = _resign_binding(authority, task_binding)
        task_input = json.loads(canonical_bytes(full_input).decode("utf-8"))
        task_input["inherited_fds"] = task_binding
        task_input = _resign_input(authority, task_input)
        task_path, task_sha = _write_member(
            authority, task_fd, "hostile-task-input.json", task_input
        )
        task_hostile = _require_child(
            _spawn_child(
                binding=task_binding,
                private_parent_roles=private_parent_roles,
                methods_root=methods_root,
                site_root=site_root,
                input_path=task_path,
                input_sha256=task_sha,
                input_digest=task_input["consumption_input_digest"],
                arm="hostile-task",
            ),
            outcome="REJECTED",
            error_contains="capture digest/task binding differs",
        )

        hostile_namespace = state / "hostile-adapter-namespace"
        hostile_namespace.mkdir(mode=0o700)
        namespace_fd = _open_directory(hostile_namespace)
        hostile_fds.append(namespace_fd)
        namespace_binding = _replace_binding_row(
            authority,
            full_binding,
            scope="adapter",
            role="namespace_root",
            relative_path=".",
            descriptor=namespace_fd,
            source_path=hostile_namespace,
        )
        namespace_input = json.loads(canonical_bytes(full_input).decode("utf-8"))
        namespace_input["inherited_fds"] = namespace_binding
        namespace_input["adapter"]["view_root"] = f"/proc/self/fd/{namespace_fd}"
        namespace_input = _resign_input(authority, namespace_input)
        namespace_path, namespace_sha = _write_member(
            authority, task_fd, "hostile-adapter-namespace-input.json", namespace_input
        )
        namespace_hostile = _require_child(
            _spawn_child(
                binding=namespace_binding,
                private_parent_roles=private_parent_roles,
                methods_root=methods_root,
                site_root=site_root,
                input_path=namespace_path,
                input_sha256=namespace_sha,
                input_digest=namespace_input["consumption_input_digest"],
                arm="hostile-adapter-namespace",
            ),
            outcome="REJECTED",
            error_contains="adapter capture/input binding differs",
        )

        hostile_leaf = state / "hostile-adapter-leaf.bin"
        _write_fixture_file(hostile_leaf, b"hostile-adapter-leaf")
        leaf_fd = os.open(
            hostile_leaf,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        os.set_inheritable(leaf_fd, False)
        hostile_fds.append(leaf_fd)
        leaf_binding = _replace_binding_row(
            authority,
            full_binding,
            scope="adapter",
            role="file",
            relative_path="adapter/adapter_model.safetensors",
            descriptor=leaf_fd,
            source_path=hostile_leaf,
        )
        leaf_input = json.loads(canonical_bytes(full_input).decode("utf-8"))
        leaf_input["inherited_fds"] = leaf_binding
        leaf_input = _resign_input(authority, leaf_input)
        leaf_path, leaf_sha = _write_member(
            authority, task_fd, "hostile-adapter-leaf-input.json", leaf_input
        )
        leaf_hostile = _require_child(
            _spawn_child(
                binding=leaf_binding,
                private_parent_roles=private_parent_roles,
                methods_root=methods_root,
                site_root=site_root,
                input_path=leaf_path,
                input_sha256=leaf_sha,
                input_digest=leaf_input["consumption_input_digest"],
                arm="hostile-adapter-leaf",
            ),
            outcome="REJECTED",
            error_contains="inherited FD/capture binding differs",
        )

        adapter.end_use()
        adapter.finalize_and_close()
        adapter_closed = True
        model.end_task("r5d-cpu-full")
        model.record_task_consumption(full_input["consumption_input_digest"])
        model.finalize(expected_task_count=2)
        model.close()
        model_closed = True
        torch = sys.modules.get("torch")
        if (
            gpu_device_descriptors()
            or torch is not None
        ):
            raise R5DCPUConsumptionProbeError("GPU runtime entered parent probe")
        return {
            "source_authority": source_authority,
            "execution_contract": {
                "linux_cpu_only": True,
                "isolated_no_site_no_bytecode": True,
                "sitecustomize_absent": True,
                "torch_imported": False,
                "torch_cuda_initialized": False,
                "torch_distributed_initialized": False,
                "model_or_peft_runtime_modules": [],
                "gpu_device_descriptors": [],
                "r5d_imported_as_module_not_production_main": True,
                "real_exec_pass_fds": True,
                "children_executed_from_parent_verified_captured_source": True,
                "held_probe_code_fd_is_publication_replay_only": True,
                "probe_code_fd_is_outside_authority_binding_rows": True,
                "private_parent_numbers_reused_as_dev_null": True,
            },
            "arms": {
                "base": {
                    "task_id": "r5d-cpu-base",
                    "consumption_input_sha256": base_sha,
                    "consumption_input_digest": base_input["consumption_input_digest"],
                    "child": base_child,
                },
                "full": {
                    "task_id": "r5d-cpu-full",
                    "consumption_input_sha256": full_sha,
                    "consumption_input_digest": full_input["consumption_input_digest"],
                    "child": full_child,
                },
            },
            "hostile_gates": {
                "digest": digest_hostile,
                "task": task_hostile,
                "adapter_namespace": namespace_hostile,
                "adapter_leaf": leaf_hostile,
            },
        }
    finally:
        for descriptor in reversed(hostile_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if adapter is not None and not adapter_closed:
            try:
                adapter.abort(reason="r5d cpu probe cleanup")
            except Exception:
                pass
        if model is not None and not model_closed:
            try:
                model.abort(reason="r5d cpu probe cleanup")
            except Exception:
                pass
        os.close(task_fd)
        os.close(view_parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    global _PROBE_CAPTURED_SOURCE, _PROBE_CODE_AUTHORITY
    if len(sys.argv) == 2 and sys.argv[1] == CHILD_TOKEN and argv is None:
        return _child_main()
    _validate_isolated_linux()
    args = build_parser().parse_args(argv)
    work_root = Path(args.work_root)
    receipt_path = Path(args.receipt)
    methods_root = Path(args.methods_root)
    site_root = Path(args.site_packages_root)
    probe_path = Path(__file__).resolve(strict=True)
    if (
        not work_root.is_absolute()
        or work_root.resolve(strict=True) != work_root
        or work_root.is_symlink()
        or not work_root.is_dir()
        or stat.S_IMODE(work_root.stat().st_mode) != 0o700
        or any(work_root.iterdir())
        or receipt_path.parent != work_root
        or receipt_path.exists()
    ):
        raise R5DCPUConsumptionProbeError("fresh work-root/receipt contract differs")
    probe_source, probe_authority = open_probe_code_authority(
        probe_path, args.probe_sha256
    )
    if not probe_source.startswith("#!/usr/bin/env python3\n"):
        os.close(probe_authority["fd"])
        raise R5DCPUConsumptionProbeError("probe source framing differs")
    _PROBE_CODE_AUTHORITY = probe_authority
    _PROBE_CAPTURED_SOURCE = probe_source
    try:
        with tempfile.TemporaryDirectory(prefix=".r5d-cpu-state-", dir=work_root) as temporary:
            result = run_probe(
                methods_root=methods_root,
                site_root=site_root,
                state=Path(temporary).resolve(strict=True),
            )
    finally:
        _PROBE_CAPTURED_SOURCE = None
        _PROBE_CODE_AUTHORITY = None
        os.close(probe_authority["fd"])
    if any(work_root.iterdir()):
        raise R5DCPUConsumptionProbeError("temporary probe state was not removed")
    payload = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "probe_source": {
            key: value for key, value in probe_authority.items() if key != "fd"
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "optimize": sys.flags.optimize,
            "proc_self_exe_authority": proc_self_executable_authority(),
        },
        **result,
        "summary": {
            "successful_arms": ["base", "full"],
            "rejected_hostiles": [
                "digest", "task", "adapter_namespace", "adapter_leaf"
            ],
            "success_count": 2,
            "hostile_rejection_count": 4,
        },
    }
    receipt_value, receipt_reference = prepare_receipt(receipt_path, payload)
    commit_receipt_and_exit(receipt_path, receipt_value, receipt_reference)
    raise R5DCPUConsumptionProbeError("final receipt commit returned")


if __name__ == "__main__":
    raise SystemExit(main())
