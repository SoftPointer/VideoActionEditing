#!/usr/bin/env python3
"""Generate/audit the two BOX-EXP-013 arms incomplete-only clips.

The renderer/runtime is the byte-frozen BOX-EXP-011 fit-repair resource stack.
Only the two new incomplete candidates are executable.  Each new receipt is
paired with one external blind-passing action receipt from BOX-EXP-011, and the
official initial Gaussian identity is compared across the two separate runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full30_action_arms_incomplete_repair_exact2_plan_v1 as plan_contract  # noqa: E402
import full30_action_fit_repair_exact8_generator_v1 as frozen_generator  # noqa: E402


HOLDER_JOB = "136140"
HOLDER_NODE = "auh7-1b-gpu-215"
RESOURCE_PREIMAGE_SHA256 = (
    "2fa752f284cbe96f869eb65595f1c2ca1a5c64185789282e0c5b3c429bc0e446"
)
RESOURCE_SPECIALIZED_BASENAME = (
    "reserve4_fixed_generation_sp4_136140_specialized_v1.py"
)
RESOURCE_SPECIALIZED_SHA256 = (
    "c0749bdb7694128fcf5deffb7503c46e3dbe2967adabec4aca4b5c800a5ed01b"
)
RESOURCE_SPECIALIZED_SIZE = 213_427
RESOURCE_SPECIALIZED_MODULE_NAME = (
    "_bernini_full30_fit_repair_resource_r6_c0749bdb"
)
STRICT_CONTROLLER_SHA256 = (
    "c5826cce6a6841c8bafa4deb90f000d1fd23bdec00d935fc657faacbe2f7a69c"
)
REVOKED_R5_RESOURCE_SPECIALIZED_SHA256 = (
    "aa2f5c01c9d231ad5340cbb572c1523546fa2e148143ee1b5bf04f53f005f017"
)
SHARD_SCHEMA = "bernini-full30-action-arms-incomplete-repair-exact2-shard-v3"
AUDIT_SCHEMA = "bernini-full30-action-arms-incomplete-repair-exact2-audit-v3"
GAP_SCHEMA = "bernini-full30-action-arms-incomplete-repair-exact2-gap-v1"
COMPUTE_PREFLIGHT_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-compute-preflight-v2"
)
ALLOWED_NODE_LOCAL_FILESYSTEM_TYPES = frozenset({"ext2/ext3"})
CHILD_SCRATCH_AUTHORITY = "launcher_created_compute_child_tmp"
CHILD_TASK_SCRATCH_BIND_SCHEMA = (
    "bernini-full30-action-arms-incomplete-repair-exact2-child-task-scratch-bind-v1"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STRICT_CONTROLLER_CACHE: Optional[tuple[ModuleType, tuple[object, ...]]] = None
_RESOURCE_MODULE_CACHE: Optional[ModuleType] = None
_RESOURCE_MODULE_PRIMITIVES: Optional[tuple[object, ...]] = None


class ArmsIncompleteExact2GenerationError(RuntimeError):
    """Raised before partial, widened, or physically unbound output can pass."""


def fail(message: str) -> NoReturn:
    raise ArmsIncompleteExact2GenerationError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ArmsIncompleteExact2GenerationError(
            "value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_verified_file_bytes(
    value: str | Path, expected_sha256: str, label: str
) -> tuple[Path, bytes]:
    """Read one canonical single-link file twice from the same no-follow fd."""

    path = Path(value)
    require(
        path.is_absolute() and SHA256_RE.fullmatch(expected_sha256) is not None,
        f"{label} path/SHA declaration differs",
    )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArmsIncompleteExact2GenerationError(
            f"{label} cannot be opened without following links"
        ) from error

    def stable_fields(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_blocks,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    try:
        opened = os.fstat(descriptor)
        first_chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            first_chunks.append(chunk)
        middle = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second_chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            second_chunks.append(chunk)
        closed = os.fstat(descriptor)
    except OSError as error:
        raise ArmsIncompleteExact2GenerationError(
            f"{label} stable read failed"
        ) from error
    finally:
        os.close(descriptor)
    try:
        named = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2GenerationError(
            f"{label} named identity is unavailable"
        ) from error
    first = b"".join(first_chunks)
    second = b"".join(second_chunks)
    require(
        resolved == path
        and stat.S_ISREG(opened.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and opened.st_nlink == 1
        and stable_fields(opened)
        == stable_fields(middle)
        == stable_fields(closed)
        == stable_fields(named)
        and first == second
        and len(first) == opened.st_size
        and len(first) > 0
        and hashlib.sha256(first).hexdigest() == expected_sha256,
        f"{label} stable source identity/SHA differs",
    )
    return path, first


def plain_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata, resolved = path.lstat(), path.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2GenerationError(f"{label} is unavailable") from error
    require(
        resolved == path
        and stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain file",
    )
    return resolved


def plain_dir(value: str | Path, label: str) -> Path:
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata, resolved = path.lstat(), path.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2GenerationError(f"{label} is unavailable") from error
    require(
        resolved == path
        and stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain directory",
    )
    return resolved


def load_signed_json(
    value: str | Path, label: str, expected_sha256: str
) -> tuple[Mapping[str, Any], Path, str]:
    require(
        SHA256_RE.fullmatch(expected_sha256) is not None,
        f"{label} expected SHA-256 differs",
    )
    path, raw = _stable_verified_file_bytes(value, expected_sha256, label)
    observed = hashlib.sha256(raw).hexdigest()
    try:
        loaded = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ArmsIncompleteExact2GenerationError(
            f"{label} is not canonical JSON"
        ) from error
    require(
        isinstance(loaded, dict)
        and raw == canonical_json_bytes(loaded) + b"\n",
        f"{label} bytes are not canonical JSON",
    )
    unsigned = dict(loaded)
    declared = unsigned.pop("receipt_digest", None)
    require(
        SHA256_RE.fullmatch(str(declared)) is not None
        and object_sha256(unsigned) == declared,
        f"{label} receipt digest differs",
    )
    return loaded, path, observed


def load_compute_preflight(
    value: str | Path, expected_sha256: str
) -> tuple[Mapping[str, Any], Path, str]:
    receipt, path, observed = load_signed_json(
        value, "compute-child preflight", expected_sha256
    )
    runtime = receipt.get("runtime")
    scratch = receipt.get("scratch_parent")
    prepare_ref = receipt.get("scratch_prepare")
    probes = receipt.get("external_action_media_probes")
    require(
        receipt.get("schema_version") == COMPUTE_PREFLIGHT_SCHEMA
        and isinstance(runtime, Mapping)
        and runtime.get("slurm_job_id") == HOLDER_JOB
        and runtime.get("hostname") == HOLDER_NODE
        and runtime.get("sole_numbered_compute_child_required") is True
        and isinstance(prepare_ref, Mapping)
        and set(prepare_ref) == {"path", "file_sha256", "receipt_digest"}
        and Path(str(prepare_ref.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(prepare_ref.get("file_sha256"))) is not None
        and SHA256_RE.fullmatch(str(prepare_ref.get("receipt_digest"))) is not None
        and isinstance(scratch, Mapping)
        and Path(str(scratch.get("path"))).is_absolute()
        and scratch.get("filesystem_type")
        in ALLOWED_NODE_LOCAL_FILESYSTEM_TYPES
        and scratch.get("authority") == CHILD_SCRATCH_AUTHORITY
        and scratch.get("mount_filesystem_type") == "ext4"
        and scratch.get("mount_source") == "/dev/mapper/vgroot-lvroot"
        and scratch.get("mount_major_minor") == "253:0"
        and scratch.get("slurm_tmpdir_present_before_prepare") is False
        and isinstance(probes, list)
        and len(probes) == 2
        and [row.get("seed") for row in probes]
        == [2026080821, 2026080921]
        and all(
            row.get("frame_count") == 81
            and row.get("fps") == 25
            and row.get("ffprobe_count_frames") is True
            for row in probes
        )
        and receipt.get("completed_before_monitor_smoke_model_or_generation")
        is True
        and receipt.get("formal_candidate_count_at_gate") == 0
        and receipt.get("diagnostic_task_count") == 0
        and receipt.get("optimizer_authorized") is False,
        "compute-child preflight authority differs",
    )
    return receipt, path, observed


def load_task_scratch_bind(
    value: str | Path, expected_sha256: str
) -> tuple[Mapping[str, Any], Path, str]:
    receipt, path, observed = load_signed_json(
        value, "child task scratch bind", expected_sha256
    )
    inner = receipt.get("scratch_inner")
    outer = receipt.get("scratch_outer")
    reference_fields = {"path", "file_sha256", "receipt_digest"}
    runtime_fields = {
        "slurm_job_id",
        "slurm_step_id",
        "hostname",
        "sole_numbered_compute_child_required",
    }
    outer_fields = {
        "path",
        "basename",
        "canonical_non_symlink",
        "device",
        "device_major_minor",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
    }
    inner_fields = {
        "path",
        "basename",
        "device",
        "device_major_minor",
        "inode",
        "uid",
        "gid",
        "mode_octal",
        "link_count",
    }
    lock_fields = inner_fields | {"size_bytes", "empty_file_sha256"}
    probe_fields = inner_fields | {"size_bytes", "file_sha256"}
    runtime = receipt.get("runtime")
    step_id = str(runtime.get("slurm_step_id", "")) if type(runtime) is dict else ""
    load_lock = receipt.get("renderer_load_lock")
    retained_probe = receipt.get("retained_probe_file")
    require(
        type(receipt) is dict
        and set(receipt)
        == {
            "schema_version",
            "authority",
            "runtime",
            "scratch_prepare",
            "compute_preflight",
            "scratch_outer",
            "scratch_inner",
            "renderer_load_lock",
            "retained_probe_file",
            "creation",
            "formal_candidate_count_at_gate",
            "diagnostic_task_count",
            "optimizer_authorized",
            "receipt_digest",
        }
        and receipt.get("schema_version") == CHILD_TASK_SCRATCH_BIND_SCHEMA
        and receipt.get("authority") == CHILD_SCRATCH_AUTHORITY
        and type(runtime) is dict
        and set(runtime) == runtime_fields
        and runtime.get("slurm_job_id") == HOLDER_JOB
        and step_id.isdecimal()
        and runtime.get("hostname") == HOLDER_NODE
        and runtime.get("sole_numbered_compute_child_required") is True
        and type(receipt.get("scratch_prepare")) is dict
        and set(receipt["scratch_prepare"]) == reference_fields
        and type(receipt.get("compute_preflight")) is dict
        and set(receipt["compute_preflight"]) == reference_fields
        and all(
            Path(str(receipt[name].get("path"))).is_absolute()
            and SHA256_RE.fullmatch(str(receipt[name].get("file_sha256")))
            is not None
            and SHA256_RE.fullmatch(str(receipt[name].get("receipt_digest")))
            is not None
            for name in ("scratch_prepare", "compute_preflight")
        )
        and type(outer) is dict
        and set(outer) == outer_fields
        and outer.get("path") == f"/tmp/BOX-EXP-013-r6-{HOLDER_JOB}-{step_id}"
        and outer.get("basename") == f"BOX-EXP-013-r6-{HOLDER_JOB}-{step_id}"
        and outer.get("canonical_non_symlink") is True
        and outer.get("device") == os.makedev(253, 0)
        and outer.get("device_major_minor") == "253:0"
        and outer.get("uid") == 2012
        and outer.get("gid") == 2000
        and outer.get("mode_octal") == "0700"
        and outer.get("link_count") == 2
        and type(outer.get("inode")) is int
        and outer["inode"] > 0
        and type(inner) is dict
        and set(inner) == inner_fields
        and Path(str(inner.get("path"))).parent == Path(str(outer.get("path")))
        and inner.get("basename") == Path(str(inner.get("path"))).name
        and re.fullmatch(
            rf"arms-incomplete-exact2-{HOLDER_JOB}-{step_id}\.[0-9a-f]{{8}}",
            str(inner.get("basename")),
        )
        is not None
        and inner.get("device") == outer.get("device")
        and inner.get("device_major_minor") == "253:0"
        and inner.get("uid") == 2012
        and inner.get("gid") == 2000
        and inner.get("mode_octal") == "0700"
        and inner.get("link_count") == 2
        and type(inner.get("inode")) is int
        and inner["inode"] > 0
        and type(load_lock) is dict
        and set(load_lock) == lock_fields
        and load_lock.get("path")
        == str(Path(str(inner.get("path"))) / "renderer-load.lock")
        and load_lock.get("basename") == "renderer-load.lock"
        and load_lock.get("device") == inner.get("device")
        and load_lock.get("device_major_minor") == "253:0"
        and type(load_lock.get("inode")) is int
        and load_lock["inode"] > 0
        and load_lock.get("uid") == 2012
        and load_lock.get("gid") == 2000
        and load_lock.get("mode_octal") == "0400"
        and load_lock.get("link_count") == 1
        and load_lock.get("size_bytes") == 0
        and load_lock.get("empty_file_sha256") == hashlib.sha256(b"").hexdigest()
        and type(retained_probe) is dict
        and set(retained_probe) == probe_fields
        and retained_probe.get("path")
        == str(Path(str(outer.get("path"))) / ".box-exp-013-r6-oexcl-fsync-probe")
        and retained_probe.get("basename")
        == ".box-exp-013-r6-oexcl-fsync-probe"
        and retained_probe.get("device") == outer.get("device")
        and retained_probe.get("device_major_minor") == "253:0"
        and type(retained_probe.get("inode")) is int
        and retained_probe["inode"] > 0
        and retained_probe.get("uid") == 2012
        and retained_probe.get("gid") == 2000
        and retained_probe.get("mode_octal") == "0600"
        and retained_probe.get("link_count") == 1
        and retained_probe.get("size_bytes")
        == len(b"BOX-EXP-013-r6-node-local-probe-v1\n")
        and retained_probe.get("file_sha256")
        == hashlib.sha256(b"BOX-EXP-013-r6-node-local-probe-v1\n").hexdigest()
        and type(receipt.get("creation")) is dict
        and set(receipt["creation"])
        == {
            "nonce_hex",
            "controller_generated_nonce",
            "caller_path_or_nonce_allowed",
            "outer_inventory_exact_retained_probe_before_mkdirat",
            "mkdirat_create_only",
            "outer_directory_fsync_after_mkdir",
            "outer_inventory_exact_retained_probe_and_inner_after_mkdir",
            "renderer_lock_openat_o_excl_no_follow",
            "renderer_lock_fchmod_0400",
            "renderer_lock_file_fsync",
            "inner_directory_fsync_after_renderer_lock",
        }
        and receipt["creation"].get("nonce_hex")
        == str(inner.get("basename", "")).rsplit(".", 1)[-1]
        and re.fullmatch(
            r"[0-9a-f]{8}", str(receipt["creation"].get("nonce_hex"))
        )
        is not None
        and {
            key: item
            for key, item in receipt["creation"].items()
            if key != "nonce_hex"
        }
        == {
            "controller_generated_nonce": True,
            "caller_path_or_nonce_allowed": False,
            "outer_inventory_exact_retained_probe_before_mkdirat": True,
            "mkdirat_create_only": True,
            "outer_directory_fsync_after_mkdir": True,
            "outer_inventory_exact_retained_probe_and_inner_after_mkdir": True,
            "renderer_lock_openat_o_excl_no_follow": True,
            "renderer_lock_fchmod_0400": True,
            "renderer_lock_file_fsync": True,
            "inner_directory_fsync_after_renderer_lock": True,
        }
        and receipt.get("formal_candidate_count_at_gate") == 0
        and receipt.get("diagnostic_task_count") == 0
        and receipt.get("optimizer_authorized") is False,
        "child task scratch bind authority differs",
    )
    return receipt, path, observed


def _strict_public_entry_scratch_chain(
    *,
    controller_plan: str | Path,
    expected_controller_plan_sha256: str,
    scratch_prepare: str | Path,
    expected_scratch_prepare_sha256: str,
    compute_preflight: str | Path,
    expected_compute_preflight_sha256: str,
    task_scratch_bind: str | Path,
    expected_task_scratch_bind_sha256: str,
    exact2_plan: Mapping[str, Any],
    exact2_plan_path: Path,
    exact2_plan_sha256: str,
) -> tuple[
    Mapping[str, Any], Path, str, Mapping[str, Any], Path, str
]:
    """Strict physical gate shared by every public model/resource entry."""

    global _STRICT_CONTROLLER_CACHE
    module_name = "full30_action_arms_incomplete_repair_exact2_controller_v1"
    expected_module_path, controller_source = _stable_verified_file_bytes(
        METHOD_ROOT / f"{module_name}.py",
        STRICT_CONTROLLER_SHA256,
        "strict public-entry controller",
    )
    if _STRICT_CONTROLLER_CACHE is None:
        require(
            sys.modules.get(module_name) is None,
            "preloaded strict public-entry controller is forbidden",
        )
        controller = ModuleType(module_name)
        specification = importlib.machinery.ModuleSpec(
            module_name, loader=None, origin=str(expected_module_path)
        )
        controller.__file__ = str(expected_module_path)
        controller.__package__ = ""
        controller.__loader__ = None
        controller.__spec__ = specification
        sys.modules[module_name] = controller
        try:
            code = compile(
                controller_source,
                str(expected_module_path),
                "exec",
                dont_inherit=True,
            )
            exec(code, controller.__dict__)
        except Exception as error:
            if sys.modules.get(module_name) is controller:
                sys.modules.pop(module_name, None)
            raise ArmsIncompleteExact2GenerationError(
                "strict public-entry controller import failed"
            ) from error
        primitives = (
            controller.load_controller_plan,
            controller.load_json,
            controller._replay_child_scratch_prepare_physical,
            controller.validate_compute_preflight,
            controller._replay_child_task_scratch_bind_physical,
        )
        _STRICT_CONTROLLER_CACHE = (controller, primitives)
    else:
        controller, primitives = _STRICT_CONTROLLER_CACHE
        require(
            sys.modules.get(module_name) is controller
            and primitives
            == (
                controller.load_controller_plan,
                controller.load_json,
                controller._replay_child_scratch_prepare_physical,
                controller.validate_compute_preflight,
                controller._replay_child_task_scratch_bind_physical,
            ),
            "strict public-entry controller cache/primitives drifted",
        )
    specification = getattr(controller, "__spec__", None)
    try:
        imported_path = Path(str(controller.__file__)).resolve(strict=True)
    except (AttributeError, OSError) as error:
        raise ArmsIncompleteExact2GenerationError(
            "strict public-entry controller origin is unavailable"
        ) from error
    require(
        type(controller) is ModuleType
        and controller.__name__ == module_name
        and imported_path == expected_module_path
        and specification is not None
        and specification.name == module_name
        and specification.origin == str(expected_module_path)
        and sys.modules.get(module_name) is controller,
        "strict public-entry controller origin/module identity differs",
    )
    # Reopen the named source after module execution/cache validation.  The
    # executed code came from the already verified bytes above; this replay
    # also fails a post-exec replacement before any resource/model entry.
    _stable_verified_file_bytes(
        expected_module_path,
        STRICT_CONTROLLER_SHA256,
        "strict public-entry controller post-exec replay",
    )
    try:
        sealed_plan, sealed_plan_path, sealed_plan_sha, sealed_exact2 = (
            controller.load_controller_plan(
                controller_plan, expected_controller_plan_sha256
            )
        )
        prepare, prepare_path, prepare_sha = controller.load_json(
            scratch_prepare,
            "public-entry scratch prepare",
            expected_scratch_prepare_sha256,
        )
        controller._replay_child_scratch_prepare_physical(
            prepare, require_initial_link_count=False
        )
        preflight, preflight_path, preflight_sha = controller.load_json(
            compute_preflight,
            "public-entry compute preflight",
            expected_compute_preflight_sha256,
        )
        controller.validate_compute_preflight(preflight)
        task_bind, task_bind_path, task_bind_sha = controller.load_json(
            task_scratch_bind,
            "public-entry task scratch bind",
            expected_task_scratch_bind_sha256,
        )
        controller._replay_child_task_scratch_bind_physical(task_bind, prepare)
    except Exception as error:
        raise ArmsIncompleteExact2GenerationError(str(error)) from error
    controller_ref = {
        "path": str(sealed_plan_path),
        "file_sha256": sealed_plan_sha,
        "plan_digest": sealed_plan["plan_digest"],
    }
    prepare_ref = {
        "path": str(prepare_path),
        "file_sha256": prepare_sha,
        "receipt_digest": prepare["receipt_digest"],
    }
    preflight_ref = {
        "path": str(preflight_path),
        "file_sha256": preflight_sha,
        "receipt_digest": preflight["receipt_digest"],
    }
    require(
        sealed_exact2 == exact2_plan
        and sealed_plan["exact2_plan"]
        == {
            **sealed_plan["exact2_plan"],
            "path": str(exact2_plan_path),
            "file_sha256": exact2_plan_sha256,
            "plan_digest": exact2_plan["plan_digest"],
        }
        and prepare["controller_plan"] == controller_ref
        and preflight["controller_plan"] == controller_ref
        and preflight["scratch_prepare"] == prepare_ref
        and task_bind["scratch_prepare"] == prepare_ref
        and task_bind["compute_preflight"] == preflight_ref
        and task_bind["runtime"] == preflight["runtime"]
        and task_bind["scratch_outer"]["path"]
        == preflight["scratch_parent"]["path"]
        and task_bind["scratch_outer"]["device"]
        == preflight["scratch_parent"]["device"]
        and task_bind["scratch_outer"]["inode"]
        == preflight["scratch_parent"]["inode"],
        "public-entry controller/plan/prepare/compute/task chain differs",
    )
    _stable_verified_file_bytes(
        expected_module_path,
        STRICT_CONTROLLER_SHA256,
        "strict public-entry controller post-validation replay",
    )
    return (
        preflight,
        preflight_path,
        preflight_sha,
        task_bind,
        task_bind_path,
        task_bind_sha,
    )


def replay_task_scratch_bind_physical(value: Mapping[str, Any]) -> Path:
    inner = plain_dir(value["scratch_inner"]["path"], "bound inner task scratch")
    outer = plain_dir(value["scratch_outer"]["path"], "bound outer scratch")
    inner_metadata = inner.lstat()
    outer_metadata = outer.lstat()
    lock = plain_file(
        value["renderer_load_lock"]["path"], "bound renderer load lock"
    )
    lock_metadata = lock.lstat()
    probe = plain_file(
        value["retained_probe_file"]["path"], "retained outer O_EXCL probe"
    )
    probe_metadata = probe.lstat()
    require(
        inner.parent == outer
        and sorted(outer.iterdir(), key=lambda path: path.name)
        == sorted([probe, inner], key=lambda path: path.name)
        and not outer.is_symlink()
        and not inner.is_symlink()
        and stat.S_ISDIR(outer_metadata.st_mode)
        and outer_metadata.st_dev == value["scratch_outer"]["device"]
        and outer_metadata.st_ino == value["scratch_outer"]["inode"]
        and outer_metadata.st_uid == value["scratch_outer"]["uid"] == 2012
        and outer_metadata.st_gid == value["scratch_outer"]["gid"] == 2000
        and stat.S_IMODE(outer_metadata.st_mode) == 0o700
        and outer_metadata.st_nlink >= 2
        and stat.S_ISDIR(inner_metadata.st_mode)
        and inner_metadata.st_dev == value["scratch_inner"]["device"]
        and inner_metadata.st_ino == value["scratch_inner"]["inode"]
        and inner_metadata.st_uid == value["scratch_inner"]["uid"] == 2012
        and inner_metadata.st_gid == value["scratch_inner"]["gid"] == 2000
        and stat.S_IMODE(inner_metadata.st_mode) == 0o700
        and inner_metadata.st_nlink >= 2,
        "bound outer/inner task scratch was renamed/recreated or drifted",
    )
    require(
        lock.parent == inner
        and lock.name == "renderer-load.lock"
        and not lock.is_symlink()
        and stat.S_ISREG(lock_metadata.st_mode)
        and lock_metadata.st_dev == value["renderer_load_lock"]["device"]
        and lock_metadata.st_ino == value["renderer_load_lock"]["inode"]
        and lock_metadata.st_uid == value["renderer_load_lock"]["uid"] == 2012
        and lock_metadata.st_gid == value["renderer_load_lock"]["gid"] == 2000
        and stat.S_IMODE(lock_metadata.st_mode) == 0o400
        and lock_metadata.st_nlink == 1
        and lock_metadata.st_size == 0
        and file_sha256(lock) == hashlib.sha256(b"").hexdigest(),
        "bound renderer load lock was renamed/recreated or drifted",
    )
    require(
        probe.parent == outer
        and probe.name == ".box-exp-013-r6-oexcl-fsync-probe"
        and not probe.is_symlink()
        and stat.S_ISREG(probe_metadata.st_mode)
        and probe_metadata.st_dev == value["retained_probe_file"]["device"]
        and probe_metadata.st_ino == value["retained_probe_file"]["inode"]
        and probe_metadata.st_uid == value["retained_probe_file"]["uid"] == 2012
        and probe_metadata.st_gid == value["retained_probe_file"]["gid"] == 2000
        and stat.S_IMODE(probe_metadata.st_mode) == 0o600
        and probe_metadata.st_nlink == 1
        and probe_metadata.st_size == value["retained_probe_file"]["size_bytes"]
        and file_sha256(probe) == value["retained_probe_file"]["file_sha256"],
        "retained outer O_EXCL probe was renamed/recreated or drifted",
    )
    return inner


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    parent = plain_dir(path.parent, "receipt output parent")
    require(
        path.is_absolute()
        and parent == path.parent
        and not path.exists()
        and not path.is_symlink(),
        "receipt output must be a fresh canonical absolute path",
    )
    raw = canonical_json_bytes(value) + b"\n"
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        os.fchmod(descriptor, 0o400)
        opened = os.fstat(descriptor)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "receipt write made no progress")
            offset += written
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        require(
            stat.S_ISREG(sealed.st_mode)
            and (sealed.st_dev, sealed.st_ino) == (opened.st_dev, opened.st_ino)
            and sealed.st_uid == os.geteuid()
            and sealed.st_gid == os.getegid()
            and stat.S_IMODE(sealed.st_mode) == 0o400
            and sealed.st_nlink == 1
            and sealed.st_size == len(raw),
            "receipt opened-file identity differs",
        )
        os.close(descriptor)
        descriptor = None
        _fsync_directory(parent)
        replay = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            replay_stat = os.fstat(replay)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(replay, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(replay)
        observed = hashlib.sha256(raw).hexdigest()
        require(
            (replay_stat.st_dev, replay_stat.st_ino)
            == (opened.st_dev, opened.st_ino)
            and replay_stat.st_nlink == 1
            and stat.S_IMODE(replay_stat.st_mode) == 0o400
            and b"".join(chunks) == raw
            and path.resolve(strict=True) == path,
            "receipt durable replay differs",
        )
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise
    return observed


def load_resource_contract(value: str | Path) -> ModuleType:
    """Load/reuse only the r6-owned specialized resource postimage."""

    global _RESOURCE_MODULE_CACHE, _RESOURCE_MODULE_PRIMITIVES
    path, raw = _stable_verified_file_bytes(
        value,
        RESOURCE_SPECIALIZED_SHA256,
        "r6 136140 resource contract",
    )
    fixed, fixed_raw = _stable_verified_file_bytes(
        METHOD_ROOT / "tools" / RESOURCE_SPECIALIZED_BASENAME,
        RESOURCE_SPECIALIZED_SHA256,
        "fixed r6 136140 resource contract",
    )
    require(
        path == fixed
        and raw == fixed_raw
        and path.name == RESOURCE_SPECIALIZED_BASENAME
        and len(raw) == RESOURCE_SPECIALIZED_SIZE
        and hashlib.sha256(raw).hexdigest() == RESOURCE_SPECIALIZED_SHA256
        and hashlib.sha256(raw).hexdigest()
        != REVOKED_R5_RESOURCE_SPECIALIZED_SHA256
        and hashlib.sha256(raw).hexdigest() != RESOURCE_PREIMAGE_SHA256
        and raw.count(b"136141") == 0
        and raw.count(b"136140") == 7,
        "r6 136140 resource specialization identity differs",
    )
    present = sys.modules.get(RESOURCE_SPECIALIZED_MODULE_NAME)
    if _RESOURCE_MODULE_CACHE is None:
        require(present is None, "untrusted r6 resource specialization is preloaded")
        module = ModuleType(RESOURCE_SPECIALIZED_MODULE_NAME)
        specification = importlib.machinery.ModuleSpec(
            RESOURCE_SPECIALIZED_MODULE_NAME,
            loader=None,
            origin=str(path),
        )
        module.__file__ = str(path)
        module.__package__ = ""
        module.__loader__ = None
        module.__spec__ = specification
        sys.modules[RESOURCE_SPECIALIZED_MODULE_NAME] = module
        try:
            code = compile(raw, str(path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
        except Exception as error:
            sys.modules.pop(RESOURCE_SPECIALIZED_MODULE_NAME, None)
            raise ArmsIncompleteExact2GenerationError(
                "verified r6 resource specialization execution failed"
            ) from error
        _stable_verified_file_bytes(
            path,
            RESOURCE_SPECIALIZED_SHA256,
            "r6 resource specialization post-exec replay",
        )
        _RESOURCE_MODULE_CACHE = module
        _RESOURCE_MODULE_PRIMITIVES = tuple(
            getattr(module, name)
            for name in (
                "load_compile_smoke_receipt",
                "_load_compile_smoke_receipt_postretention_attested",
                "replay_retained_compile_smoke_root",
                "load_host_cgroup_memory_monitor_start",
                "_journal_prefix",
                "_sample_row",
                "_process_identity_is_live",
            )
        )
    module = _RESOURCE_MODULE_CACHE
    require(module is not None, "r6 resource module cache is absent")
    specification = getattr(module, "__spec__", None)
    primitives = tuple(
        getattr(module, name, None)
        for name in (
            "load_compile_smoke_receipt",
            "_load_compile_smoke_receipt_postretention_attested",
            "replay_retained_compile_smoke_root",
            "load_host_cgroup_memory_monitor_start",
            "_journal_prefix",
            "_sample_row",
            "_process_identity_is_live",
        )
    )
    require(
        present in {None, module}
        and sys.modules.get(RESOURCE_SPECIALIZED_MODULE_NAME) is module
        and type(module) is ModuleType
        and module.__name__ == RESOURCE_SPECIALIZED_MODULE_NAME
        and getattr(module, "__file__", None) == str(path)
        and specification is not None
        and specification.name == RESOURCE_SPECIALIZED_MODULE_NAME
        and specification.origin == str(path)
        and getattr(module, "COMPILE_SMOKE_SCHEMA", None)
        == "bernini-generic-action-fit40-compile-smoke-v9"
        and getattr(module, "HOST_MEMORY_LIMIT_GIB", None) == 60
        and getattr(module, "HOST_MEMORY_SAFE_CEILING_GIB", None) == 56
        and getattr(module, "HOST_MEMORY_SAMPLE_INTERVAL_NS", None) == 10_000_000
        and getattr(module, "T2V_GPU_MEMORY_LIMIT_GIB", None) == 52
        and _RESOURCE_MODULE_PRIMITIVES is not None
        and all(
            observed is expected
            for observed, expected in zip(primitives, _RESOURCE_MODULE_PRIMITIVES)
        )
        and all(callable(item) for item in primitives),
        "cached r6 resource module origin/primitives drifted",
    )
    _stable_verified_file_bytes(
        path,
        RESOURCE_SPECIALIZED_SHA256,
        "cached r6 resource specialization replay",
    )
    return module


def _validate_candidate_receipt(
    resource: ModuleType, task: Mapping[str, Any], receipt_path: Path
) -> Mapping[str, Any]:
    try:
        receipt = frozen_generator._validate_candidate_receipt(
            resource, task, receipt_path
        )
    except Exception as error:
        raise ArmsIncompleteExact2GenerationError(str(error)) from error
    candidate = receipt.get("candidate", {})
    mp4 = receipt.get("artifacts", {}).get("mp4", {})
    require(
        candidate.get("candidate_id") == task["candidate_id"]
        and candidate.get("semantic_branch") == "incomplete"
        and candidate.get("seed") == task["seed"]
        and candidate.get("full_t2v_caption_utf8_sha256")
        == plan_contract.INCOMPLETE_PROMPT_SHA256
        and mp4.get("frame_count") == 81,
        f"candidate is not the exact81 frozen-prompt incomplete clip: {task['candidate_id']}",
    )
    mp4_path = plain_file(mp4.get("path", ""), f"generated MP4 {task['candidate_id']}")
    require(
        str(mp4_path) == mp4.get("path")
        and mp4_path.name == "t2v.mp4"
        and mp4_path.parent == receipt_path.parent
        and SHA256_RE.fullmatch(str(mp4.get("sha256"))) is not None
        and file_sha256(mp4_path) == mp4.get("sha256")
        and mp4.get("fps") == 25,
        f"generated MP4 physical binding differs: {task['candidate_id']}",
    )
    return receipt


def _candidate_receipt_row(
    *, task: Mapping[str, Any], receipt: Mapping[str, Any], receipt_path: Path
) -> dict[str, Any]:
    """Bind one audit row to its receipt bytes and generated MP4 bytes."""

    mp4 = receipt["artifacts"]["mp4"]
    return {
        "candidate_id": task["candidate_id"],
        "calibration_group_id": task["calibration_group_id"],
        "semantic_branch": "incomplete",
        "candidate_receipt_path": str(receipt_path),
        "candidate_receipt_file_sha256": file_sha256(receipt_path),
        "candidate_receipt_digest": receipt["receipt_digest"],
        "generated_mp4_path": mp4["path"],
        "generated_mp4_file_sha256": mp4["sha256"],
        "generated_mp4_frame_count": 81,
        "generated_mp4_fps": 25,
    }


def _validate_signed_json_file(
    reference: Mapping[str, Any], label: str
) -> tuple[Mapping[str, Any], Path]:
    try:
        value, path, _ = plan_contract.load_json(
            reference["runtime_path"], label, reference["file_sha256"]
        )
    except plan_contract.ArmsIncompleteExact2PlanError as error:
        raise ArmsIncompleteExact2GenerationError(str(error)) from error
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    require(
        declared == reference["receipt_digest"]
        and object_sha256(unsigned) == declared,
        f"{label} receipt digest differs",
    )
    return value, path


def _validate_external_action_artifacts(
    action: Mapping[str, Any], resource: Optional[ModuleType] = None
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    seed = action["seed"]
    mp4_path = plain_file(action["mp4"]["runtime_path"], f"external action MP4 seed {seed}")
    gaussian_path = plain_file(
        action["official_initial_gaussian"]["runtime_path"],
        f"external action Gaussian seed {seed}",
    )
    native, _ = _validate_signed_json_file(
        action["native_receipt"], f"external native receipt seed {seed}"
    )
    calibration, _ = _validate_signed_json_file(
        action["calibration_receipt"], f"external calibration receipt seed {seed}"
    )
    candidate = calibration.get("candidate", {})
    mp4 = calibration.get("artifacts", {}).get("mp4", {})
    gaussian = calibration.get("artifacts", {}).get("official_initial_gaussian", {})
    native_output = native.get("outputs", {}).get("t2v", {})
    native_gaussian = native.get("initial_noise_artifacts", {}).get("t2v", {})
    identity = plan_contract._gaussian_identity(gaussian)
    require(
        file_sha256(mp4_path) == action["mp4"]["file_sha256"]
        and file_sha256(gaussian_path)
        == action["official_initial_gaussian"]["file_sha256"]
        and candidate.get("candidate_id") == action["candidate_id"]
        and candidate.get("semantic_branch") == "action"
        and candidate.get("seed") == seed
        and candidate.get("calibration_group_id") == action["calibration_group_id"]
        and mp4.get("path") == action["mp4"]["runtime_path"]
        and mp4.get("sha256") == action["mp4"]["file_sha256"]
        and mp4.get("frame_count") == 81
        and gaussian.get("path") == action["official_initial_gaussian"]["runtime_path"]
        and gaussian.get("sha256")
        == action["official_initial_gaussian"]["file_sha256"]
        and identity == action["official_initial_gaussian"]["identity"]
        and native_output.get("path") == action["mp4"]["runtime_path"]
        and native_output.get("sha256") == action["mp4"]["file_sha256"]
        and native_output.get("frame_count") == 81
        and plan_contract._gaussian_identity(native_gaussian) == identity,
        f"external action physical artifact binding differs for seed {seed}",
    )
    if resource is not None:
        _physical_gaussian_identity(
            resource, gaussian, f"external action Gaussian seed {seed}"
        )
    return calibration, native


def _physical_gaussian_identity(
    resource: ModuleType, artifact: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    """Reopen one safetensors container and recompute its tensor identity."""

    try:
        evidence = resource._physical_safetensor_tensor_evidence(
            artifact,
            artifact_name=label,
            expected_key="official_initial_gaussian",
            expected_metadata=resource.GAUSSIAN_SAFETENSORS_METADATA,
            expected_label="official_initial_gaussian_t2v",
        )
    except Exception as error:
        raise ArmsIncompleteExact2GenerationError(str(error)) from error
    tensor = evidence.get("tensor_identity", {})
    declared = plan_contract._gaussian_identity(artifact)
    recomputed = {
        "raw_value_sha256": tensor.get("raw_storage_sha256"),
        "content_sha256": tensor.get("content_sha256"),
        "shape": tensor.get("shape"),
        "dtype": tensor.get("dtype"),
        "stored_dtype": artifact.get("stored_dtype"),
        "generator_initial_seed": artifact.get("generator_initial_seed"),
    }
    require(
        evidence.get("exact_single_tensor_key") is True
        and evidence.get("container_file_sha256") == artifact.get("sha256")
        and recomputed == declared,
        f"{label} physical tensor identity differs from its receipt",
    )
    return recomputed


def _validate_new_physical_artifacts(
    task: Mapping[str, Any], receipt: Mapping[str, Any],
    resource: Optional[ModuleType] = None,
) -> Mapping[str, Any]:
    mp4 = receipt.get("artifacts", {}).get("mp4", {})
    gaussian = receipt.get("artifacts", {}).get("official_initial_gaussian", {})
    mp4_path = plain_file(mp4.get("path", ""), f"new incomplete MP4 {task['candidate_id']}")
    gaussian_path = plain_file(
        gaussian.get("path", ""), f"new incomplete Gaussian {task['candidate_id']}"
    )
    require(
        file_sha256(mp4_path) == mp4.get("sha256")
        and file_sha256(gaussian_path) == gaussian.get("sha256")
        and mp4.get("frame_count") == 81,
        f"new incomplete physical artifact binding differs: {task['candidate_id']}",
    )
    declared = plan_contract._gaussian_identity(gaussian)
    if resource is not None:
        require(
            _physical_gaussian_identity(
                resource, gaussian, f"new incomplete Gaussian {task['candidate_id']}"
            )
            == declared,
            f"new incomplete Gaussian safe_open identity differs: {task['candidate_id']}",
        )
    return declared


def cross_run_same_gaussian_proof(
    *, task: Mapping[str, Any], action: Mapping[str, Any],
    incomplete_receipt: Mapping[str, Any], action_receipt: Mapping[str, Any],
    resource: Optional[ModuleType] = None,
) -> Mapping[str, Any]:
    incomplete_identity = _validate_new_physical_artifacts(
        task, incomplete_receipt, resource
    )
    action_artifact = action_receipt["artifacts"]["official_initial_gaussian"]
    action_identity = plan_contract._gaussian_identity(action_artifact)
    if resource is not None:
        require(
            _physical_gaussian_identity(
                resource, action_artifact, f"external action Gaussian seed {task['seed']}"
            )
            == action_identity,
            f"external action Gaussian safe_open identity differs for seed {task['seed']}",
        )
    require(
        task["seed"] == action["seed"]
        and task["calibration_group_id"] == action["calibration_group_id"]
        and action_identity == action["official_initial_gaussian"]["identity"]
        and incomplete_identity == action_identity,
        f"cross-run action/incomplete Gaussian differs for seed {task['seed']}",
    )
    incomplete_mp4 = incomplete_receipt["artifacts"]["mp4"]
    incomplete_gaussian = incomplete_receipt["artifacts"]["official_initial_gaussian"]
    return {
        "seed": task["seed"],
        "calibration_group_id": task["calibration_group_id"],
        "branch_order": ["action", "incomplete"],
        "candidate_ids": [action["candidate_id"], task["candidate_id"]],
        "external_action": {
            "mp4_path": action["mp4"]["runtime_path"],
            "mp4_file_sha256": action["mp4"]["file_sha256"],
            "native_receipt_path": action["native_receipt"]["runtime_path"],
            "native_receipt_file_sha256": action["native_receipt"]["file_sha256"],
            "calibration_receipt_path": action["calibration_receipt"]["runtime_path"],
            "calibration_receipt_file_sha256": action["calibration_receipt"]["file_sha256"],
            "gaussian_path": action["official_initial_gaussian"]["runtime_path"],
            "gaussian_file_sha256": action["official_initial_gaussian"]["file_sha256"],
        },
        "new_incomplete": {
            "mp4_path": incomplete_mp4["path"],
            "mp4_file_sha256": incomplete_mp4["sha256"],
            "gaussian_path": incomplete_gaussian["path"],
            "gaussian_file_sha256": incomplete_gaussian["sha256"],
        },
        "official_gaussian_identity": action_identity,
        "same_seed": True,
        "cross_run": True,
        "action_incomplete_official_gaussian_tensor_values_byte_equal": True,
        "physical_artifacts_reopened": True,
        "physical_safetensors_safe_open_recomputed": resource is not None,
    }


def run_shard(args: argparse.Namespace) -> int:
    plan, plan_path, plan_sha = plan_contract.load_plan(
        args.plan, args.expected_plan_sha256
    )
    matching = list(plan["admission_tasks"])
    require(
        args.group_id == "sp4-a"
        and len(matching) == 2
        and all(task["group_id"] == "sp4-a" for task in matching)
        and all(task["semantic_branch"] == "incomplete" for task in matching),
        "sealed exact2 shard scope differs",
    )
    visible = "0,1,2,3"
    require(
        all(task["visible_gpus"] == [0, 1, 2, 3] for task in matching)
        and os.environ.get("ROCR_VISIBLE_DEVICES") == visible,
        "sealed exact2 shard GPU mapping differs",
    )
    (
        preflight,
        preflight_path,
        preflight_sha,
        task_bind,
        task_bind_path,
        task_bind_sha,
    ) = _strict_public_entry_scratch_chain(
        controller_plan=args.controller_plan,
        expected_controller_plan_sha256=args.expected_controller_plan_sha256,
        scratch_prepare=args.scratch_prepare,
        expected_scratch_prepare_sha256=args.expected_scratch_prepare_sha256,
        compute_preflight=args.compute_preflight,
        expected_compute_preflight_sha256=args.expected_compute_preflight_sha256,
        task_scratch_bind=args.task_scratch_bind,
        expected_task_scratch_bind_sha256=args.expected_task_scratch_bind_sha256,
        exact2_plan=plan,
        exact2_plan_path=plan_path,
        exact2_plan_sha256=plan_sha,
    )
    bound_inner = replay_task_scratch_bind_physical(task_bind)
    require(
        task_bind["compute_preflight"]
        == {
            "path": str(preflight_path),
            "file_sha256": preflight_sha,
            "receipt_digest": preflight["receipt_digest"],
        }
        and task_bind["scratch_prepare"] == preflight["scratch_prepare"]
        and task_bind["runtime"] == preflight["runtime"]
        and task_bind["scratch_outer"]["path"]
        == preflight["scratch_parent"]["path"]
        and task_bind["scratch_outer"]["device"]
        == preflight["scratch_parent"]["device"]
        and task_bind["scratch_outer"]["inode"]
        == preflight["scratch_parent"]["inode"],
        "task scratch bind full prepare/compute/outer chain differs before resource use",
    )
    resource = load_resource_contract(args.resource_contract)
    try:
        binding = resource._runtime_binding(args)
        require(
            isinstance(binding, tuple) and len(binding) == 7,
            "r6 retained runtime binding tuple differs",
        )
        (
            runtime,
            python,
            worker,
            rank_exec,
            rank_exec_source,
            torchrun_source,
            scratch,
        ) = binding
        runtime_lock = runtime.get("serialized_host_checkpoint_load", {})
        bound_lock = task_bind["renderer_load_lock"]
        require(
            runtime_lock.get("path") == bound_lock["path"]
            and runtime_lock.get("sha256") == bound_lock["empty_file_sha256"]
            and runtime_lock.get("mode") == bound_lock["mode_octal"]
            and all(
                runtime_lock.get(field) == bound_lock[field]
                for field in (
                    "device",
                    "inode",
                    "uid",
                    "gid",
                    "link_count",
                    "size_bytes",
                )
            ),
            "resource renderer lock is not the controller-bound original inode",
        )
        smoke = frozen_generator.validate_resource_smoke(args, resource, runtime)
    except Exception as error:
        raise ArmsIncompleteExact2GenerationError(str(error)) from error
    preflight_scratch = preflight["scratch_parent"]
    rank_scratch = runtime.get("node_local_scratch")
    require(
        isinstance(rank_scratch, Mapping)
        and rank_scratch
        == smoke.get("runtime", {}).get("node_local_scratch")
        and rank_scratch.get("path") == str(scratch)
        and rank_scratch.get("path") == str(bound_inner)
        and Path(str(rank_scratch.get("path"))).parent
        == Path(preflight_scratch["path"])
        and rank_scratch.get("filesystem_type")
        == preflight_scratch["filesystem_type"],
        "compute stat-f preflight is not bound to rank resource scratch receipt",
    )
    require(
        task_bind["scratch_outer"]["path"] == preflight_scratch["path"],
        "task scratch bind is not chained to compute preflight",
    )
    output = Path(args.output_dir)
    require(
        output.is_absolute()
        and output != Path("/")
        and not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.parent.is_symlink(),
        "shard output must be a fresh absolute directory",
    )
    output.mkdir(mode=0o700)
    receipt_rows: list[dict[str, Any]] = []
    proofs: list[Mapping[str, Any]] = []
    action_by_seed = {row["seed"]: row for row in plan["external_action_cells"]}
    for task in matching:
        candidate_output = output / task["candidate_id"]
        try:
            resource.assert_live_host_cgroup_memory_monitor()
            command = resource._candidate_command(
                args,
                task=task,
                candidate_output=candidate_output,
                python=python,
                worker=worker,
                rank_exec=rank_exec,
                rank_exec_source=rank_exec_source,
                torchrun_source=torchrun_source,
                runtime=runtime,
            )
            environment = resource._candidate_environment(
                expected_visible=visible,
                python=python,
                scratch=scratch,
                cache_token=(
                    f"arms-incomplete-exact2-{task['calibration_group_id']}"
                ),
                runtime=runtime,
            )
            candidate_census = resource._run_candidate_under_live_monitor(
                command, environment
            )
            resource.assert_live_host_cgroup_memory_monitor()
        except subprocess.CalledProcessError as error:
            raise ArmsIncompleteExact2GenerationError(
                f"generation failed for {task['candidate_id']}"
            ) from error
        except Exception as error:
            if isinstance(error, ArmsIncompleteExact2GenerationError):
                raise
            raise ArmsIncompleteExact2GenerationError(str(error)) from error
        receipt_path = candidate_output / "pair-v5-t2v-calibration-receipt.json"
        receipt = _validate_candidate_receipt(resource, task, receipt_path)
        action = action_by_seed[task["seed"]]
        action_receipt, _ = _validate_external_action_artifacts(action, resource)
        proof = cross_run_same_gaussian_proof(
            task=task,
            action=action,
            incomplete_receipt=receipt,
            action_receipt=action_receipt,
            resource=resource,
        )
        proofs.append(proof)
        receipt_rows.append(
            {
                **_candidate_receipt_row(
                    task=task, receipt=receipt, receipt_path=receipt_path
                ),
                "full81_pass_pending_independent_review": True,
                "post_candidate_cgroup_census": candidate_census,
            }
        )
    require(len(proofs) == 2, "exact2 cross-run proof count differs")
    replay_task_scratch_bind_physical(task_bind)
    unsigned = {
        "schema_version": SHARD_SCHEMA,
        "plan": {
            "path": str(plan_path),
            "file_sha256": plan_sha,
            "plan_digest": plan["plan_digest"],
        },
        "group_id": "sp4-a",
        "visible_gpus": [0, 1, 2, 3],
        "candidate_count": 2,
        "new_branch_order": ["incomplete", "incomplete"],
        "candidate_receipts": receipt_rows,
        "cross_run_same_gaussian_pair_proofs": proofs,
        "resource_compile_smoke": {
            "path": str(
                plain_file(
                    args.resource_compile_smoke_receipt,
                    "resource compile-smoke receipt",
                )
            ),
            "file_sha256": args.expected_resource_compile_smoke_receipt_sha256,
            "receipt_digest": smoke["receipt_digest"],
            "formal_candidate_count_at_gate": 0,
        },
        "compute_preflight": {
            "path": str(preflight_path),
            "file_sha256": preflight_sha,
            "receipt_digest": preflight["receipt_digest"],
        },
        "task_scratch_bind": {
            "path": str(task_bind_path),
            "file_sha256": task_bind_sha,
            "receipt_digest": task_bind["receipt_digest"],
        },
        "rank_resource_scratch_binding": {
            "preflight_scratch_parent_path": preflight_scratch["path"],
            "rank_task_scratch_path": rank_scratch["path"],
            "filesystem_type": rank_scratch["filesystem_type"],
            "source_environment_variable": "GADP_NODE_LOCAL_SCRATCH_FSTYPE",
            "preflight_stat_f_matches_rank_resource_receipt": True,
            "compile_smoke_runtime_matches_rank_resource_receipt": True,
        },
        "diagnostic_task_count": 0,
        "diagnostic_generation_authorized": False,
        "action_generation_authorized": False,
        "independent_full81_review_performed": False,
        "q_input_authorized": False,
        "a_min_input_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
    }
    value = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    write_create_only(output / "arms-incomplete-repair-exact2-shard-receipt-v2.json", value)
    return 0


def validate_shard_scratch_binding(
    *,
    root: Path,
    plan: Mapping[str, Any],
    resource: ModuleType,
    compute_preflight: str | Path,
    expected_compute_preflight_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Replay the stat-f -> environment -> rank-resource-receipt chain."""

    return _validate_shard_scratch_binding_core(
        root=root,
        plan=plan,
        resource=resource,
        compute_preflight=compute_preflight,
        expected_compute_preflight_sha256=expected_compute_preflight_sha256,
        postretention_attested=False,
    )


def _validate_shard_scratch_binding_postretention_attested(
    *,
    root: Path,
    plan: Mapping[str, Any],
    resource: ModuleType,
    compute_preflight: str | Path,
    expected_compute_preflight_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Parent receipt/shared-media replay after the child retention seals."""

    return _validate_shard_scratch_binding_core(
        root=root,
        plan=plan,
        resource=resource,
        compute_preflight=compute_preflight,
        expected_compute_preflight_sha256=expected_compute_preflight_sha256,
        postretention_attested=True,
    )


def _validate_shard_scratch_binding_core(
    *,
    root: Path,
    plan: Mapping[str, Any],
    resource: ModuleType,
    compute_preflight: str | Path,
    expected_compute_preflight_sha256: str,
    postretention_attested: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:

    preflight, preflight_path, preflight_sha = load_compute_preflight(
        compute_preflight, expected_compute_preflight_sha256
    )
    shard_candidate = plain_file(
        root / "arms-incomplete-repair-exact2-shard-receipt-v2.json",
        "exact2 shard receipt",
    )
    shard, shard_path, shard_sha = load_signed_json(
        shard_candidate,
        "exact2 shard receipt",
        file_sha256(shard_candidate),
    )
    expected_shard_fields = {
        "schema_version",
        "plan",
        "group_id",
        "visible_gpus",
        "candidate_count",
        "new_branch_order",
        "candidate_receipts",
        "cross_run_same_gaussian_pair_proofs",
        "resource_compile_smoke",
        "compute_preflight",
        "task_scratch_bind",
        "rank_resource_scratch_binding",
        "diagnostic_task_count",
        "diagnostic_generation_authorized",
        "action_generation_authorized",
        "independent_full81_review_performed",
        "q_input_authorized",
        "a_min_input_authorized",
        "training_performed",
        "optimizer_created",
        "optimizer_authorized",
        "receipt_digest",
    }
    preflight_ref = {
        "path": str(preflight_path),
        "file_sha256": preflight_sha,
        "receipt_digest": preflight["receipt_digest"],
    }
    rank_binding = shard.get("rank_resource_scratch_binding")
    smoke_ref = shard.get("resource_compile_smoke")
    task_bind_ref = shard.get("task_scratch_bind")
    task_bind = None
    task_bind_path = None
    task_bind_sha = None
    if isinstance(task_bind_ref, Mapping) and set(task_bind_ref) == {
        "path", "file_sha256", "receipt_digest"
    }:
        task_bind, task_bind_path, task_bind_sha = load_task_scratch_bind(
            task_bind_ref["path"], task_bind_ref["file_sha256"]
        )
        if not postretention_attested:
            replay_task_scratch_bind_physical(task_bind)
    require(
        set(shard) == expected_shard_fields
        and shard.get("schema_version") == SHARD_SCHEMA
        and shard.get("plan")
        == {
            "path": plan.get("_path", shard.get("plan", {}).get("path")),
            "file_sha256": plan.get(
                "_file_sha256", shard.get("plan", {}).get("file_sha256")
            ),
            "plan_digest": plan["plan_digest"],
        }
        and shard.get("group_id") == "sp4-a"
        and shard.get("visible_gpus") == [0, 1, 2, 3]
        and shard.get("candidate_count") == 2
        and shard.get("new_branch_order") == ["incomplete", "incomplete"]
        and isinstance(shard.get("candidate_receipts"), list)
        and len(shard["candidate_receipts"]) == 2
        and shard.get("compute_preflight") == preflight_ref
        and isinstance(task_bind_ref, Mapping)
        and task_bind is not None
        and str(task_bind_path) == task_bind_ref.get("path")
        and task_bind_sha == task_bind_ref.get("file_sha256")
        and task_bind.get("receipt_digest") == task_bind_ref.get("receipt_digest")
        and task_bind.get("compute_preflight") == preflight_ref
        and task_bind.get("scratch_prepare") == preflight.get("scratch_prepare")
        and task_bind.get("runtime") == preflight.get("runtime")
        and task_bind.get("scratch_outer", {}).get("path")
        == preflight.get("scratch_parent", {}).get("path")
        and task_bind.get("scratch_outer", {}).get("device")
        == preflight.get("scratch_parent", {}).get("device")
        and task_bind.get("scratch_outer", {}).get("inode")
        == preflight.get("scratch_parent", {}).get("inode")
        and task_bind.get("scratch_inner", {}).get("path")
        == rank_binding.get("rank_task_scratch_path")
        and isinstance(smoke_ref, Mapping)
        and set(smoke_ref)
        == {
            "path",
            "file_sha256",
            "receipt_digest",
            "formal_candidate_count_at_gate",
        }
        and smoke_ref.get("formal_candidate_count_at_gate") == 0
        and isinstance(rank_binding, Mapping)
        and set(rank_binding)
        == {
            "preflight_scratch_parent_path",
            "rank_task_scratch_path",
            "filesystem_type",
            "source_environment_variable",
            "preflight_stat_f_matches_rank_resource_receipt",
            "compile_smoke_runtime_matches_rank_resource_receipt",
        }
        and rank_binding.get("preflight_scratch_parent_path")
        == preflight["scratch_parent"]["path"]
        and Path(str(rank_binding.get("rank_task_scratch_path"))).parent
        == Path(preflight["scratch_parent"]["path"])
        and rank_binding.get("filesystem_type")
        == preflight["scratch_parent"]["filesystem_type"]
        and rank_binding.get("source_environment_variable")
        == "GADP_NODE_LOCAL_SCRATCH_FSTYPE"
        and rank_binding.get("preflight_stat_f_matches_rank_resource_receipt")
        is True
        and rank_binding.get("compile_smoke_runtime_matches_rank_resource_receipt")
        is True
        and shard.get("diagnostic_task_count") == 0
        and shard.get("diagnostic_generation_authorized") is False
        and shard.get("action_generation_authorized") is False
        and shard.get("optimizer_authorized") is False,
        "shard stat-f/rank resource binding differs",
    )
    try:
        smoke_loader = (
            resource._load_compile_smoke_receipt_postretention_attested
            if postretention_attested
            else resource.load_compile_smoke_receipt
        )
        smoke, smoke_path, smoke_sha = smoke_loader(
            smoke_ref["path"], smoke_ref["file_sha256"]
        )
    except Exception as error:
        raise ArmsIncompleteExact2GenerationError(str(error)) from error
    if not postretention_attested:
        try:
            resource.replay_retained_compile_smoke_root(smoke)
        except Exception as error:
            raise ArmsIncompleteExact2GenerationError(str(error)) from error
    runtime_lock = smoke.get("runtime", {}).get(
        "serialized_host_checkpoint_load", {}
    )
    monitor_runtime = smoke.get("runtime", {}).get(
        "host_cgroup_sampled_memory_monitor", {}
    )
    candidate_row_fields = {
        "candidate_id",
        "calibration_group_id",
        "semantic_branch",
        "candidate_receipt_path",
        "candidate_receipt_file_sha256",
        "candidate_receipt_digest",
        "generated_mp4_path",
        "generated_mp4_file_sha256",
        "generated_mp4_frame_count",
        "generated_mp4_fps",
        "full81_pass_pending_independent_review",
        "post_candidate_cgroup_census",
    }
    try:
        for row in shard["candidate_receipts"]:
            require(
                isinstance(row, Mapping)
                and set(row) == candidate_row_fields
                and row.get("semantic_branch") == "incomplete"
                and row.get("generated_mp4_frame_count") == 81
                and row.get("generated_mp4_fps") == 25
                and row.get("full81_pass_pending_independent_review") is True,
                "exact2 candidate receipt/census row shape differs",
            )
            resource._validate_post_candidate_cgroup_census_shape(
                row.get("post_candidate_cgroup_census"), monitor_runtime
            )
    except Exception as error:
        if isinstance(error, ArmsIncompleteExact2GenerationError):
            raise
        raise ArmsIncompleteExact2GenerationError(str(error)) from error
    bound_lock = task_bind["renderer_load_lock"]
    require(
        str(smoke_path) == smoke_ref["path"]
        and smoke_sha == smoke_ref["file_sha256"]
        and smoke.get("receipt_digest") == smoke_ref["receipt_digest"]
        and smoke.get("runtime", {}).get("node_local_scratch")
        == {
            "path": rank_binding["rank_task_scratch_path"],
            "filesystem_type": rank_binding["filesystem_type"],
        },
        "compile-smoke rank resource scratch receipt replay differs",
    )
    require(
        runtime_lock.get("path") == bound_lock["path"]
        and runtime_lock.get("sha256") == bound_lock["empty_file_sha256"]
        and runtime_lock.get("mode") == bound_lock["mode_octal"]
        and all(
            runtime_lock.get(field) == bound_lock[field]
            for field in (
                "device",
                "inode",
                "uid",
                "gid",
                "link_count",
                "size_bytes",
            )
        ),
        "compile-smoke renderer lock differs from controller-bound inode",
    )
    shard_ref = {
        "path": str(shard_path),
        "file_sha256": shard_sha,
        "receipt_digest": shard["receipt_digest"],
    }
    return preflight_ref, rank_binding, shard_ref


def audit_exact2(
    *, plan_path: str | Path, expected_plan_sha256: str,
    controller_plan: str | Path, expected_controller_plan_sha256: str,
    scratch_prepare: str | Path, expected_scratch_prepare_sha256: str,
    generation_root: str | Path, output: str | Path, gap_output: str | Path,
    compute_preflight: str | Path, expected_compute_preflight_sha256: str,
    task_scratch_bind: str | Path, expected_task_scratch_bind_sha256: str,
) -> Mapping[str, Any]:
    plan, resolved, plan_sha = plan_contract.load_plan(
        plan_path, expected_plan_sha256
    )
    expected_tasks = list(plan["admission_tasks"])
    expected_ids = [task["candidate_id"] for task in expected_tasks]
    root = plain_dir(generation_root, "formal exact2 generation root")
    (
        bound_preflight,
        bound_preflight_path,
        bound_preflight_sha,
        bound_task,
        bound_task_path,
        bound_task_sha,
    ) = _strict_public_entry_scratch_chain(
        controller_plan=controller_plan,
        expected_controller_plan_sha256=expected_controller_plan_sha256,
        scratch_prepare=scratch_prepare,
        expected_scratch_prepare_sha256=expected_scratch_prepare_sha256,
        compute_preflight=compute_preflight,
        expected_compute_preflight_sha256=expected_compute_preflight_sha256,
        task_scratch_bind=task_scratch_bind,
        expected_task_scratch_bind_sha256=expected_task_scratch_bind_sha256,
        exact2_plan=plan,
        exact2_plan_path=resolved,
        exact2_plan_sha256=plan_sha,
    )
    replay_task_scratch_bind_physical(bound_task)
    require(
        bound_task["compute_preflight"]
        == {
            "path": str(bound_preflight_path),
            "file_sha256": bound_preflight_sha,
            "receipt_digest": bound_preflight["receipt_digest"],
        }
        and bound_task["scratch_prepare"]
        == bound_preflight["scratch_prepare"]
        and bound_task["runtime"] == bound_preflight["runtime"]
        and bound_task["scratch_outer"]["path"]
        == bound_preflight["scratch_parent"]["path"]
        and bound_task["scratch_outer"]["device"]
        == bound_preflight["scratch_parent"]["device"]
        and bound_task["scratch_outer"]["inode"]
        == bound_preflight["scratch_parent"]["inode"],
        "audit task scratch bind full prepare/compute/outer chain differs",
    )
    resource = load_resource_contract(
        METHOD_ROOT / "tools" / RESOURCE_SPECIALIZED_BASENAME
    )
    preflight_ref, rank_binding, shard_ref = validate_shard_scratch_binding(
        root=root,
        plan={**plan, "_path": str(resolved), "_file_sha256": plan_sha},
        resource=resource,
        compute_preflight=compute_preflight,
        expected_compute_preflight_sha256=expected_compute_preflight_sha256,
    )
    observed_paths: dict[str, Path] = {}
    for path in root.rglob("pair-v5-t2v-calibration-receipt.json"):
        candidate_id = path.parent.name
        require(candidate_id not in observed_paths, "duplicate formal candidate receipt")
        observed_paths[candidate_id] = path.resolve(strict=True)
    missing = [candidate_id for candidate_id in expected_ids if candidate_id not in observed_paths]
    unexpected = sorted(set(observed_paths) - set(expected_ids))
    gap_unsigned = {
        "schema_version": GAP_SCHEMA,
        "plan_path": str(resolved),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "expected_candidate_count": 2,
        "observed_candidate_count": len(observed_paths),
        "missing_candidate_ids": missing,
        "unexpected_candidate_ids": unexpected,
        "diagnostic_task_count": 0,
        "action_generation_authorized": False,
        "optimizer_authorized": False,
    }
    gap = {**gap_unsigned, "receipt_digest": object_sha256(gap_unsigned)}
    write_create_only(Path(gap_output), gap)
    require(not missing and not unexpected, "formal exact2 closure differs; gap receipt written")
    action_by_seed = {row["seed"]: row for row in plan["external_action_cells"]}
    rows: list[dict[str, Any]] = []
    proofs: list[Mapping[str, Any]] = []
    for task in expected_tasks:
        receipt_path = observed_paths[task["candidate_id"]]
        receipt = _validate_candidate_receipt(resource, task, receipt_path)
        action = action_by_seed[task["seed"]]
        action_receipt, _ = _validate_external_action_artifacts(action, resource)
        proofs.append(
            cross_run_same_gaussian_proof(
                task=task,
                action=action,
                incomplete_receipt=receipt,
                action_receipt=action_receipt,
                resource=resource,
            )
        )
        rows.append(
            _candidate_receipt_row(
                task=task, receipt=receipt, receipt_path=receipt_path
            )
        )
    require(len(proofs) == 2, "formal exact2 proof closure differs")
    replay_task_scratch_bind_physical(bound_task)
    require(
        rank_binding["rank_task_scratch_path"]
        == bound_task["scratch_inner"]["path"],
        "formal audit rank scratch differs from bound original inner inode",
    )
    unsigned = {
        "schema_version": AUDIT_SCHEMA,
        "plan_path": str(resolved),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "dataset": plan_contract.DATASET,
        "candidate_count": 2,
        "comparator_cell_count": 2,
        "new_branch_order": ["incomplete", "incomplete"],
        "compute_preflight": preflight_ref,
        "task_scratch_bind": {
            "path": str(bound_task_path),
            "file_sha256": bound_task_sha,
            "receipt_digest": bound_task["receipt_digest"],
        },
        "rank_resource_scratch_binding": rank_binding,
        "shard_receipt": shard_ref,
        "candidate_receipts": rows,
        "cross_run_same_gaussian_pair_proofs": proofs,
        "all_candidates_exact81": True,
        "independent_full81_review_performed": False,
        "review_admission_authorized": False,
        "q_input_authorized": False,
        "a_min_input_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "diagnostic_task_count": 0,
        "diagnostic_generation_observed_or_required": False,
        "action_generation_observed_or_required": False,
    }
    audit = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    write_create_only(Path(output), audit)
    return audit


def add_runtime_args(command: argparse.ArgumentParser) -> None:
    command.add_argument("--plan", required=True)
    command.add_argument("--expected-plan-sha256", required=True)
    command.add_argument("--controller-plan", required=True)
    command.add_argument("--expected-controller-plan-sha256", required=True)
    command.add_argument("--scratch-prepare", required=True)
    command.add_argument("--expected-scratch-prepare-sha256", required=True)
    command.add_argument("--resource-contract", required=True)
    command.add_argument("--resource-compile-smoke-receipt", required=True)
    command.add_argument(
        "--expected-resource-compile-smoke-receipt-sha256", required=True
    )
    command.add_argument("--python", required=True)
    command.add_argument("--bernini-root", required=True)
    command.add_argument("--veomni-root", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--checkpoint-content-manifest", required=True)
    command.add_argument("--method-source-revision", required=True)
    command.add_argument("--method-source-archive-sha256", required=True)
    command.add_argument("--master-port", type=int, required=True)
    command.add_argument("--compute-preflight", required=True)
    command.add_argument("--expected-compute-preflight-sha256", required=True)
    command.add_argument("--task-scratch-bind", required=True)
    command.add_argument("--expected-task-scratch-bind-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-sp4")
    add_runtime_args(run)
    run.add_argument("--group-id", choices=("sp4-a",), required=True)
    run.add_argument("--output-dir", required=True)
    audit = commands.add_parser("audit-exact2")
    audit.add_argument("--plan", required=True)
    audit.add_argument("--expected-plan-sha256", required=True)
    audit.add_argument("--controller-plan", required=True)
    audit.add_argument("--expected-controller-plan-sha256", required=True)
    audit.add_argument("--scratch-prepare", required=True)
    audit.add_argument("--expected-scratch-prepare-sha256", required=True)
    audit.add_argument("--generation-root", required=True)
    audit.add_argument("--compute-preflight", required=True)
    audit.add_argument("--expected-compute-preflight-sha256", required=True)
    audit.add_argument("--task-scratch-bind", required=True)
    audit.add_argument("--expected-task-scratch-bind-sha256", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--gap-output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-sp4":
        return run_shard(args)
    value = audit_exact2(
        plan_path=args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        controller_plan=args.controller_plan,
        expected_controller_plan_sha256=args.expected_controller_plan_sha256,
        scratch_prepare=args.scratch_prepare,
        expected_scratch_prepare_sha256=args.expected_scratch_prepare_sha256,
        generation_root=args.generation_root,
        compute_preflight=args.compute_preflight,
        expected_compute_preflight_sha256=args.expected_compute_preflight_sha256,
        task_scratch_bind=args.task_scratch_bind,
        expected_task_scratch_bind_sha256=args.expected_task_scratch_bind_sha256,
        output=args.output,
        gap_output=args.gap_output,
    )
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA",
    "ArmsIncompleteExact2GenerationError",
    "COMPUTE_PREFLIGHT_SCHEMA",
    "GAP_SCHEMA",
    "RESOURCE_SPECIALIZED_BASENAME",
    "SHARD_SCHEMA",
    "_validate_candidate_receipt",
    "_validate_external_action_artifacts",
    "audit_exact2",
    "cross_run_same_gaussian_proof",
    "file_sha256",
    "load_resource_contract",
    "load_compute_preflight",
    "object_sha256",
    "validate_shard_scratch_binding",
]
