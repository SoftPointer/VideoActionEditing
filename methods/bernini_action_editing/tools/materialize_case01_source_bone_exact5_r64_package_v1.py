#!/usr/bin/env python3
"""Materialize the fresh case01 exact-five R64 canary package without Slurm.

The input is a sealed source snapshot.  In particular, the historical
``infer_lora.py`` is copied from the sealed r5f snapshot lineage by the
snapshot builder; this program never reads that file from a working tree.
The resulting package contains a nineteen-file immutable production release,
the independent asset-audit receipt, an exact-five plan, and a captured-entry
launch payload.  It does not submit or execute a Slurm step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Mapping, Sequence


EXPERIMENTS = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments"
)
TARGETS = {
    ("143808", "auh7-1b-gpu-292"): EXPERIMENTS
    / "bernini_object_grounded_case01_0821_exact5_r64_canary_v1",
}
INPUT_ROOT = (
    EXPERIMENTS
    / "bernini_object_grounded_case01_0821_exact5_inputs_0a62b740_r1"
)
RANK_CACHE_ROOT = Path(
    "/tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache"
)
VACE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
VACE_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
FFMPEG = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
FFPROBE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/"
    "runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
)
MODEL_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
BERNINI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/"
    "phase_a_native_gpu_canary_dual4_all8_v1/releases/"
    "source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591"
)
VEOMNI_ROOT = BERNINI_ROOT.parent / "VeOmni-f90b3dc6"
CHECKPOINT_MANIFEST = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_full644_exploratory_r64_job141620_v5/"
    "runs/full644-r64-reference-dpo-preservation-one-pass-v5/"
    "checkpoint-00000644/checkpoint_manifest.json"
)
TORCH_ROOT = VACE_PYTHON.parent.parent / "lib/python3.12/site-packages/torch"
AUDIT_RELATIVE = (
    "md/action_editing/20260821_man/evidence/"
    "case01_exact5_intervention_asset_independent_audit_v1.json"
)
SNAPSHOT_MANIFEST = "case01_exact5_source_snapshot_manifest_v1.json"
SOURCE_SNAPSHOT_ROOT = (
    EXPERIMENTS
    / "bernini_object_grounded_case01_0821_exact5_source_snapshot_24_20260821_r1"
)
OLD_R5F_SNAPSHOT_ROOT = (
    EXPERIMENTS
    / "bernini_full644_exploratory_matched_r5f_source_snapshot_21_20260820_r1"
)
SOURCE_STAGING_ROOT = (
    EXPERIMENTS
    / "bernini_object_grounded_case01_0821_exact5_source_staging_v1"
)
CAMPAIGN = "case01-source-bone-exact5-r64-canary"
TASK_IDS = (
    "case01-exact_original-full644",
    "case01-codec_only_present-full644",
    "case01-bone_removed-full644",
    "case01-bone_translated_up150-full644",
    "case01-sham_control_up150-full644",
)

# Nineteen physical release files: the old r5f seventeen with its root launcher
# replaced by the exact5 launcher, plus the wrapper runner and exact5 evaluator.
# Every historical byte, including infer_lora.py, is supplied by the sealed
# snapshot; the source-snapshot builder enforces that provenance separately.
RELEASE_FILES = {
    "methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json":
        "953933f1161b6d62826d388ba5ed42e42792fbf5f2bdeea199c1eb13cd251b4a",
    "methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl":
        "c05c4e5b5bf85de882bde32c71a984d736247733e586ed91d40026b12aaaf701",
    "methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py":
        "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
    "methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256":
        "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
    "methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py":
        "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d",
    "methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py":
        "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py":
        "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5f.py":
        "52dcde8797484504ab28a5c59c532c0877a145ea24762edaf3105a21b0719e19",
    "methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py":
        "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223",
    "methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5.py":
        "cb201398940d59393fa58471dc2c3f9fdf001c7e881ec891ce892bb460cf01ba",
    "methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py":
        "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136",
    "methods/bernini_action_editing/infer_lora.py":
        "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
    "methods/bernini_action_editing/self_generated_action_preservation_v2.py":
        "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c",
    "methods/bernini_action_editing/tools/build_renderer_dataset.py":
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    "methods/bernini_action_editing/tools/materialize_vae.py":
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    "methods/bernini_action_editing/train_lora.py":
        "ead547b8309e1b5ae5c831444e9f5d1d8e1785fed5fe39cf7b97f13f82a9ce85",
    "methods/bernini_action_editing/case01_source_bone_exact5_runner_v1.py":
        "4a46e870b61ff42345c523cca97b8170c065033bc2af9a0353be9df4373ec3ea",
    "methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py":
        "891551d46b0ca11362fe8d446f202036b9957baa943de0aec6da1f3ad45d7d58",
    "methods/bernini_action_editing/case01_source_bone_exact5_spooled_launcher_auh_v1.py":
        "00b68ca8221dd343cb9ca8393c9205cccf6a61d474c56e56c9b081570418d390",
}
AUDIT_SHA256 = (
    "040c53a3647ae957212a1d2d6da3ffa75b4207ace07e1c7ba6ce128033dce969"
)
AUDIT_SIZE = 8_285
DIAGNOSTIC_SOURCE_FILES = {
    "methods/bernini_action_editing/case01_source_bone_exact5_static_probe_v1.py":
        "3eabfcd6fedc264018c18aec6c518a77aa4e093dcb9d2b65f371244e6ac57f02",
    "methods/bernini_action_editing/case01_source_bone_exact5_root_fake_runner_v1.py":
        "414be72dc7b428b6b34ad038c4315cb0b336f28db74f4ee2273a8d14cd8218a1",
}
CHECKPOINT = {
    "path": str(CHECKPOINT_MANIFEST),
    "sha256": "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2",
    "manifest_digest": "7bae23da51a3c5a67adb41ee85dd026c374d2581bd3409e868e18b2f6f4dffc4",
    "global_step": 644,
    "receipt_digest": "aaf348a7daa6c5ca2fe721771857287125ee02eb2c9a499f45b11a2e113d15d7",
    "file_count": 5,
    "adapter_config_sha256": "94bfaf73d714d7e77095ff68ce57e24932e0c05bde324263f5fe321660b95f62",
    "adapter_model_sha256": "44efdc5a0501238250b1d32ae2859abe248ffc37b152cd8db86ff84b378d6b22",
    "training_receipt_sha256": "3402c8c93c092bfc4490bf86790ab6429b4cbaad38358956cb0beeb5df7d4c4c",
    "optimizer_sha256": "77b7b22db4da92f28f23b4ae91c7271f55ab6a92353bfc8b0bbeb30529a7af63",
}
PRODUCER_STATIC = {
    "inference_receipt_schema": "bernini-r-1p3b-action-lora-inference-receipt-v5",
    "infer_lora_sha256": "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
    "method_source_revision": "ce4cffc1e8a144448c92252d9fb63087f03bbd8c",
    "method_source_archive_sha256": "12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828",
    "ffprobe_sha256": "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5",
}


class Exact5PackageError(RuntimeError):
    """The exact5 package materialization contract differs."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid,
        "gid": info.st_gid, "mode": info.st_mode, "nlink": info.st_nlink,
        "rdev": info.st_rdev, "size": info.st_size,
        "blocks": getattr(info, "st_blocks", 0), "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def stable_file(
    path: Path, expected_sha256: str | None = None,
    expected_mode: int | None = None,
) -> bytes:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or path.is_symlink() or path.resolve(strict=True) != path
    ):
        raise Exact5PackageError(f"noncanonical stable path: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(descriptor, min(1_048_576, before.st_size - offset), offset)
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not raw
        or len(raw) != before.st_size or identity(before) != identity(after)
        or identity(before) != identity(named)
        or (expected_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != expected_sha256)
        or (expected_mode is not None
            and stat.S_IMODE(before.st_mode) != expected_mode)
    ):
        raise Exact5PackageError(f"stable file differs: {path}")
    return raw


def create_file(path: Path, raw: bytes, mode: int) -> None:
    if not raw or path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise Exact5PackageError(f"fresh file target differs: {path}")
    descriptor = os.open(
        path, os.O_RDWR | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise Exact5PackageError("create-only write made no progress")
            offset += count
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        named = path.lstat()
        replay = os.pread(descriptor, len(raw), 0)
        if (
            not stat.S_ISREG(staged.st_mode) or stat.S_IMODE(staged.st_mode) != 0
            or staged.st_nlink != 1 or identity(staged) != identity(named)
            or replay != raw
        ):
            raise Exact5PackageError("create-only staging replay differs")
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def open_exact_tree(
    root: Path,
    expected_file_modes: Mapping[str, int],
    expected_directory_modes: Mapping[str, int],
) -> dict[str, int]:
    """Open an exact regular-file/directory tree without following links."""
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    opened: dict[str, int] = {}
    files: set[str] = set()
    pending: list[tuple[str, int]] = []
    try:
        root_fd = os.open(root, flags)
        opened["."] = root_fd
        pending.append((".", root_fd))
        while pending:
            relative, directory_fd = pending.pop()
            before = os.fstat(directory_fd)
            path = root if relative == "." else root / relative
            named = os.lstat(path)
            if (
                not stat.S_ISDIR(before.st_mode)
                or identity(before) != identity(named)
                or before.st_uid != 2012 or before.st_gid != 2000
                or stat.S_IMODE(before.st_mode)
                != expected_directory_modes.get(relative)
            ):
                raise Exact5PackageError(
                    f"package directory authority differs: {relative}"
                )
            for name in os.listdir(directory_fd):
                if not name or name in {".", ".."} or "/" in name:
                    raise Exact5PackageError("package entry name differs")
                child_relative = name if relative == "." else f"{relative}/{name}"
                child = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISREG(child.st_mode):
                    if (
                        child.st_uid != 2012 or child.st_gid != 2000
                        or child.st_nlink != 1
                        or stat.S_IMODE(child.st_mode)
                        != expected_file_modes.get(child_relative)
                    ):
                        raise Exact5PackageError(
                            f"package file authority differs: {child_relative}"
                        )
                    files.add(child_relative)
                elif stat.S_ISDIR(child.st_mode):
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                    held = os.fstat(child_fd)
                    if identity(held) != identity(child):
                        os.close(child_fd)
                        raise Exact5PackageError(
                            f"package directory identity differs: {child_relative}"
                        )
                    opened[child_relative] = child_fd
                    pending.append((child_relative, child_fd))
                else:
                    raise Exact5PackageError(
                        f"package contains a link or special entry: {child_relative}"
                    )
        if files != set(expected_file_modes) or set(opened) != set(expected_directory_modes):
            raise Exact5PackageError("package physical tree closure differs")
        return opened
    except BaseException:
        for descriptor in opened.values():
            os.close(descriptor)
        raise


def close_directories(opened: Mapping[str, int]) -> None:
    for descriptor in opened.values():
        os.close(descriptor)


def load_module(name: str, path: Path, digest: str) -> types.ModuleType:
    raw = stable_file(path, digest, 0o444)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = None
    module.__loader__ = None
    module.__spec__ = None
    module.__cached__ = None
    module.__builtins__ = __builtins__
    sys.modules[name] = module
    try:
        exec(
            compile(raw.decode("utf-8", "strict"), str(path), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def mkdir_fresh(path: Path, mode: int = 0o700) -> None:
    os.mkdir(path, mode)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        held = os.fstat(descriptor)
        named = os.lstat(path)
        if (
            not stat.S_ISDIR(held.st_mode) or identity(held) != identity(named)
            or held.st_uid != 2012 or held.st_gid != 2000
            or stat.S_IMODE(held.st_mode) != mode
        ):
            raise Exact5PackageError(f"fresh directory authority differs: {path}")
    finally:
        os.close(descriptor)


def _expected_snapshot_files() -> set[str]:
    return set(RELEASE_FILES) | set(DIAGNOSTIC_SOURCE_FILES) | {
        "methods/bernini_action_editing/tools/"
        "materialize_case01_source_bone_exact5_r64_package_v1.py",
        AUDIT_RELATIVE,
        SNAPSHOT_MANIFEST,
    }


def _preflight_snapshot(
    source_root: Path, expected_materializer_sha256: str,
) -> dict[str, bytes]:
    if (
        source_root != SOURCE_SNAPSHOT_ROOT or not source_root.is_absolute()
        or os.path.normpath(str(source_root)) != str(source_root)
        or source_root.is_symlink() or source_root.resolve(strict=True) != source_root
    ):
        raise Exact5PackageError("source snapshot root differs")
    manifest_raw = stable_file(source_root / SNAPSHOT_MANIFEST, expected_mode=0o444)
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Exact5PackageError("source snapshot manifest is not JSON") from error
    unsigned = dict(manifest) if isinstance(manifest, dict) else {}
    claimed = unsigned.pop("snapshot_digest", None)
    rows = manifest.get("files") if isinstance(manifest, Mapping) else None
    expected_nonmanifest = _expected_snapshot_files() - {SNAPSHOT_MANIFEST}
    manifest_fields = {
        "schema_version", "status", "target_root", "old_r5f_snapshot_root",
        "new_staging_root", "file_count", "old_reused_file_count",
        "new_staged_file_count", "physical_file_count_including_manifest",
        "release_file_count", "sealed_r5f_infer_lora_reused",
        "working_tree_infer_lora_read", "slurm_step_launched", "files",
        "snapshot_digest",
    }
    if (
        not isinstance(manifest, dict) or set(manifest) != manifest_fields
        or manifest_raw != canonical_json_bytes(manifest) + b"\n"
        or manifest.get("schema_version")
        != "case01-source-bone-exact5-source-snapshot-v1"
        or manifest.get("status") != "SEALED_NOT_EXECUTED"
        or claimed != object_sha256(unsigned)
        or manifest.get("target_root") != str(source_root)
        or manifest.get("old_r5f_snapshot_root") != str(OLD_R5F_SNAPSHOT_ROOT)
        or manifest.get("new_staging_root") != str(SOURCE_STAGING_ROOT)
        or manifest.get("file_count") != 23
        or manifest.get("old_reused_file_count") != 16
        or manifest.get("new_staged_file_count") != 7
        or manifest.get("physical_file_count_including_manifest") != 24
        or manifest.get("release_file_count") != 19
        or manifest.get("sealed_r5f_infer_lora_reused") is not True
        or manifest.get("working_tree_infer_lora_read") is not False
        or manifest.get("slurm_step_launched") is not False
        or not isinstance(rows, list)
        or len(rows) != 23
        or len({row.get("path") for row in rows if isinstance(row, Mapping)}) != 23
        or {row.get("path") for row in rows if isinstance(row, Mapping)}
        != expected_nonmanifest
        or [row.get("path") for row in rows if isinstance(row, Mapping)]
        != sorted(expected_nonmanifest)
    ):
        raise Exact5PackageError("source snapshot manifest closure differs")
    expected_hashes = {
        **RELEASE_FILES,
        **DIAGNOSTIC_SOURCE_FILES,
        AUDIT_RELATIVE: AUDIT_SHA256,
        "methods/bernini_action_editing/tools/"
        "materialize_case01_source_bone_exact5_r64_package_v1.py":
            expected_materializer_sha256,
    }
    raw_by_relative: dict[str, bytes] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping) or set(row) != {
                "path", "sha256", "size", "mode", "provenance"
            }
        ):
            raise Exact5PackageError("source snapshot row differs")
        relative = row["path"]
        expected_provenance = (
            "sealed_r5f_snapshot" if relative in RELEASE_FILES
            and relative not in {
                "methods/bernini_action_editing/case01_source_bone_exact5_runner_v1.py",
                "methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py",
                "methods/bernini_action_editing/case01_source_bone_exact5_spooled_launcher_auh_v1.py",
            }
            else "new_exact5_staging"
        )
        if (
            row.get("sha256") != expected_hashes.get(relative)
            or row.get("mode") != 0o444
            or row.get("provenance") != expected_provenance
        ):
            raise Exact5PackageError(f"source snapshot row pin differs: {relative}")
        raw = stable_file(source_root / relative, row["sha256"], 0o444)
        info = (source_root / relative).lstat()
        if (
            len(raw) != row.get("size")
            or info.st_uid != 2012 or info.st_gid != 2000
        ):
            raise Exact5PackageError(f"source snapshot row size differs: {relative}")
        raw_by_relative[relative] = raw
    expected_snapshot_files = _expected_snapshot_files()
    expected_directories = {"."}
    for relative in expected_snapshot_files:
        parent = Path(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent
    snapshot_file_modes = {
        relative: 0o444 for relative in expected_snapshot_files
    }
    snapshot_directory_modes = {
        relative: 0o555 for relative in expected_directories
    }
    opened = open_exact_tree(
        source_root, snapshot_file_modes, snapshot_directory_modes,
    )
    close_directories(opened)
    return raw_by_relative


def launch_input(root: Path, job_id: str, node: str, plan: Path) -> dict[str, Any]:
    method = root / "release/methods/bernini_action_editing"
    return {
        "schema_version": "case01-source-bone-exact5-root-launch-input-auh-v1",
        "entry_mode": "trusted_stdin",
        "runner": str(method / "case01_source_bone_exact5_runner_v1.py"),
        "frozen_runner": str(method / "full644_exploratory_matched_runner_auh_r5.py"),
        "exact5_eval": str(method / "case01_source_bone_exact5_eval_v1.py"),
        "bridge": str(method / "full644_exploratory_matched_torchrun_fd_bridge_v2.py"),
        "adapter": str(method / "full644_exploratory_matched_infer_adapter_auh_r5f.py"),
        "base_adapter": str(method / "full644_exploratory_matched_infer_adapter_v2.py"),
        "eval_v1": str(method / "full644_exploratory_matched_eval_v1.py"),
        "eval_v2": str(method / "full644_exploratory_matched_eval_v2.py"),
        "model_authority": str(method / "action_preservation_decoded_eval_model_authority_v2.py"),
        "torchrun_source": str(TORCH_ROOT / "distributed/run.py"),
        "torchrun_handler_source": str(
            TORCH_ROOT / "distributed/elastic/multiprocessing/"
            "subprocess_handler/subprocess_handler.py"
        ),
        "torch_local_agent_source": str(
            TORCH_ROOT / "distributed/elastic/agent/server/local_elastic_agent.py"
        ),
        "torch_dynamic_rendezvous_source": str(
            TORCH_ROOT / "distributed/elastic/rendezvous/dynamic_rendezvous.py"
        ),
        "torch_multiprocessing_api_source": str(
            TORCH_ROOT / "distributed/elastic/multiprocessing/api.py"
        ),
        "model_manifest": str(method / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"),
        "python": str(VACE_PYTHON), "ffmpeg": str(FFMPEG), "plan": str(plan),
        "output_report": str(root / "final/case01_source_bone_exact5_r64_report_v1.json"),
        "runner_attestation": str(root / "final/case01_source_bone_exact5_runner_attestation_v1.json"),
        "model_root": str(MODEL_ROOT), "bernini_root": str(BERNINI_ROOT),
        "veomni_root": str(VEOMNI_ROOT),
        "authority_root": str(root / "runtime/model-authority"),
        "rank_cache_root": str(RANK_CACHE_ROOT),
        "holder_job_id": job_id, "expected_node": node,
        "campaign_mode": CAMPAIGN,
    }


def _make_directories(root: Path) -> None:
    relative_directories = {
        "release", "authority", "plan", "launch", "evidence", "outputs",
        "outputs/media", "final", "logs", "runtime", "diagnostics",
    }
    for relative in RELEASE_FILES:
        parent = Path("release") / Path(relative).parent
        while str(parent) != ".":
            relative_directories.add(str(parent))
            parent = parent.parent
    for relative in sorted(relative_directories, key=lambda item: (len(Path(item).parts), item)):
        # The package root is fresh and every directory appears exactly once
        # in this set.  An unexpected concurrent entry must fail os.mkdir.
        mkdir_fresh(root / relative)


def shell_quote(value: str) -> str:
    if "\x00" in value:
        raise Exact5PackageError("shell literal contains NUL")
    return "'" + value.replace("'", "'\"'\"'") + "'"


CAPTURED_STATIC_BOOTSTRAP = r'''import hashlib,os,stat,sys,types
if len(sys.argv)<17: raise RuntimeError("static bootstrap argv differs")
pyfd_raw,srcfd_raw,python_path,python_sha,source_path,source_sha=sys.argv[1:7]
slurm_values=sys.argv[7:16]; probe_argv=sys.argv[16:]
try: pyfd=int(pyfd_raw); srcfd=int(srcfd_raw)
except ValueError as error: raise RuntimeError("static bootstrap FD differs") from error
if pyfd<3 or srcfd<3 or pyfd==srcfd or not os.get_inheritable(pyfd) or not os.get_inheritable(srcfd): raise RuntimeError("static bootstrap inherited FD differs")
def ident(value): return (value.st_dev,value.st_ino,value.st_uid,value.st_gid,value.st_mode,value.st_nlink,value.st_rdev,value.st_size,getattr(value,"st_blocks",0),value.st_mtime_ns,value.st_ctime_ns)
def read(fd,size):
 out=[]; offset=0
 while offset<size:
  block=os.pread(fd,min(1048576,size-offset),offset)
  if not block: break
  out.append(block); offset+=len(block)
 raw=b"".join(out)
 if len(raw)!=size: raise RuntimeError("static bootstrap short read")
 return raw
source_raw=None
for fd,path,pin,executable in ((pyfd,python_path,python_sha,True),(srcfd,source_path,source_sha,False)):
 before=os.fstat(fd); first=read(fd,before.st_size); middle=os.fstat(fd); second=read(fd,before.st_size); after=os.fstat(fd); named=os.lstat(path)
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=2012 or before.st_gid!=2000 or ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named) or first!=second or hashlib.sha256(first).hexdigest()!=pin or (not executable and stat.S_IMODE(before.st_mode)!=0o444) or (executable and not before.st_mode&0o111): raise RuntimeError("static bootstrap authority differs: "+path)
 if executable and ident(before)!=ident(os.stat("/proc/self/exe")): raise RuntimeError("static bootstrap Python process differs")
 if not executable: source_raw=first
 os.set_inheritable(fd,False)
if type(source_raw) is not bytes: raise RuntimeError("static source bytes differ")
names=("SLURM_JOB_ID","SLURM_STEP_ID","SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_NODE","SLURM_STEP_GPUS","SLURM_NNODES","SLURM_STEP_NUM_NODES","SLURM_JOB_NODELIST","SLURM_STEP_NODELIST")
if set(os.environ) not in (set(),{"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"]!="C.UTF-8"): raise RuntimeError("static bootstrap environment differs")
os.environ.clear(); os.environ.update(dict(zip(names,slurm_values,strict=True)))
module=types.ModuleType("__main__"); module.__file__=source_path; module.__package__=None; module.__loader__=None; module.__spec__=None; module.__cached__=None; module.__builtins__=__builtins__; sys.modules["__main__"]=module; sys.argv=[source_path,*probe_argv]
exec(compile(source_raw.decode("utf-8","strict"),source_path,"exec",dont_inherit=True),module.__dict__)'''


def build_static_payload(
    *, root: Path, source: Path, source_sha256: str, plan_sha256: str,
    launch_receipt_sha256: str,
) -> bytes:
    receipt = root / "evidence/exact5_static_probe_receipt_v1.json"
    probe_args = [
        "--root", str(root), "--plan-sha256", plan_sha256,
        "--launch-receipt-sha256", launch_receipt_sha256,
        "--receipt", str(receipt),
    ]
    lines = [
        "#!/bin/bash -p", "set -euo pipefail", "umask 077",
        '[[ "$-" == *p* ]] || exit 91',
        '[[ "$0" == "bash" || "$0" == "/bin/bash" || "$0" == "-bash" ]] || exit 92',
        '[[ -z "${BASH_ENV-}" && -z "${ENV-}" ]] || exit 93',
        '[[ "${SLURM_JOB_ID-}" == 143808 && "${SLURM_STEP_ID-}" =~ ^[1-9][0-9]*$ ]] || exit 94',
        '(( 10#$SLURM_STEP_ID > 394 )) || exit 95',
        '[[ "${SLURM_GPUS_ON_NODE-}" == 8 && "${SLURM_GPUS_PER_NODE-}" == 8 && "${SLURM_STEP_GPUS-}" == 0,1,2,3,4,5,6,7 ]] || exit 96',
        '[[ "${SLURM_NNODES-}" == 1 && "${SLURM_STEP_NUM_NODES-}" == 1 && "${SLURM_JOB_NODELIST-}" == auh7-1b-gpu-292 && "${SLURM_STEP_NODELIST-}" == auh7-1b-gpu-292 ]] || exit 97',
        '[[ -z "${SLURM_JOB_GPUS+x}" && -z "${SLURM_JOB_NUM_NODES+x}" ]] || exit 98',
        "if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi",
        f"readonly EXACT5_PYTHON={shell_quote(str(VACE_PYTHON))}",
        f"readonly EXACT5_STATIC={shell_quote(str(source))}",
        'exec {EXACT5_PYTHON_FD}<"$EXACT5_PYTHON"',
        'exec {EXACT5_STATIC_FD}<"$EXACT5_STATIC"',
        '[[ "$EXACT5_PYTHON_FD" =~ ^[0-9]+$ && "$EXACT5_STATIC_FD" =~ ^[0-9]+$ ]] || exit 99',
        "exec -c \"/proc/self/fd/$EXACT5_PYTHON_FD\" -I -S -B -c "
        + shell_quote(CAPTURED_STATIC_BOOTSTRAP)
        + ' "$EXACT5_PYTHON_FD" "$EXACT5_STATIC_FD" '
        + shell_quote(str(VACE_PYTHON)) + " " + shell_quote(VACE_PYTHON_SHA256)
        + " " + shell_quote(str(source)) + " " + shell_quote(source_sha256)
        + ' "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURM_GPUS_ON_NODE" "$SLURM_GPUS_PER_NODE" "$SLURM_STEP_GPUS" "$SLURM_NNODES" "$SLURM_STEP_NUM_NODES" "$SLURM_JOB_NODELIST" "$SLURM_STEP_NODELIST" '
        + " ".join(shell_quote(value) for value in probe_args),
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    key = (args.job_id, args.node)
    if key not in TARGETS:
        raise Exact5PackageError("unsupported exact5 package binding")
    if os.geteuid() != 2012 or os.getegid() != 2000:
        raise Exact5PackageError("materializer owner authority differs")
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in RELEASE_FILES.values()
    ):
        raise Exact5PackageError("release pin remains unresolved")
    root = TARGETS[key]
    if root.exists() or root.is_symlink():
        raise Exact5PackageError(f"fresh exact5 package root exists: {root}")
    if RANK_CACHE_ROOT.exists() or RANK_CACHE_ROOT.is_symlink():
        raise Exact5PackageError("fresh exact5 rank cache already exists")
    source_root = Path(args.source_root)
    if (
        len(args.materializer_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.materializer_sha256)
    ):
        raise Exact5PackageError("captured materializer SHA argument differs")
    source_bytes = _preflight_snapshot(source_root, args.materializer_sha256)

    mkdir_fresh(root)
    _make_directories(root)
    artifacts: dict[str, dict[str, Any]] = {}
    for relative, digest in RELEASE_FILES.items():
        raw = source_bytes[relative]
        target = root / "release" / relative
        create_file(target, raw, 0o444)
        artifacts[f"release/{relative}"] = {
            "sha256": digest, "size": len(raw), "mode": 0o444,
        }
    audit_raw = source_bytes[AUDIT_RELATIVE]
    if len(audit_raw) != AUDIT_SIZE:
        raise Exact5PackageError("independent audit source size differs")
    audit_path = root / "authority/case01_exact5_intervention_asset_independent_audit_v1.json"
    create_file(audit_path, audit_raw, 0o444)
    artifacts[str(audit_path.relative_to(root))] = {
        "sha256": AUDIT_SHA256, "size": len(audit_raw), "mode": 0o444,
    }

    diagnostic_sources: dict[str, Path] = {}
    for relative, digest_value in DIAGNOSTIC_SOURCE_FILES.items():
        raw = source_bytes[relative]
        target = root / "diagnostics" / Path(relative).name
        create_file(target, raw, 0o444)
        diagnostic_sources[relative] = target
        artifacts[str(target.relative_to(root))] = {
            "sha256": digest_value, "size": len(raw), "mode": 0o444,
        }

    method = root / "release/methods/bernini_action_editing"
    exact5_eval = load_module(
        "_case01_exact5_eval_materializer",
        method / "case01_source_bone_exact5_eval_v1.py",
        RELEASE_FILES["methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py"],
    )
    authority = exact5_eval.build_asset_authority(
        INPUT_ROOT / "manifest.json", INPUT_ROOT, audit_path,
    )
    producer = {
        **PRODUCER_STATIC,
        "infer_lora_path": str(method / "infer_lora.py"),
        "ffprobe_path": str(FFPROBE),
    }
    plan = exact5_eval.build_plan(
        asset_authority=authority, checkpoint_manifest=CHECKPOINT,
        producer=producer, output_root=root / "outputs/media",
    )
    plan_raw = canonical_json_bytes(plan) + b"\n"
    plan_path = root / "plan/case01_source_bone_exact5_r64_plan_v1.json"
    create_file(plan_path, plan_raw, 0o444)
    plan_sha256 = hashlib.sha256(plan_raw).hexdigest()
    if exact5_eval.load_plan(str(plan_path), plan_sha256) != plan:
        raise Exact5PackageError("exact5 plan replay differs")
    artifacts[str(plan_path.relative_to(root))] = {
        "sha256": plan_sha256, "size": len(plan_raw), "mode": 0o444,
    }

    input_value = launch_input(root, args.job_id, args.node, plan_path)
    input_raw = canonical_json_bytes(input_value) + b"\n"
    input_path = root / "launch/root_launch_input_exact5_v1.json"
    create_file(input_path, input_raw, 0o444)
    launcher = load_module(
        "_case01_exact5_launcher_materializer",
        method / "case01_source_bone_exact5_spooled_launcher_auh_v1.py",
        RELEASE_FILES[
            "methods/bernini_action_editing/"
            "case01_source_bone_exact5_spooled_launcher_auh_v1.py"
        ],
    )
    payload_path = root / "launch/root_launch_payload_exact5_v1.sh"
    receipt_path = root / "launch/root_launch_receipt_exact5_v1.json"
    receipt = launcher.materialize(str(input_path), str(payload_path), str(receipt_path))
    receipt_raw = stable_file(receipt_path, expected_mode=0o400)
    payload_raw = stable_file(
        payload_path, receipt["payload_sha256"], expected_mode=0o444,
    )
    if (
        json.loads(receipt_raw) != receipt
        or receipt_raw != canonical_json_bytes(receipt) + b"\n"
        or receipt.get("status") != "MATERIALIZED_NOT_SUBMITTED"
        or receipt.get("release", {}).get("selected_task_ids") != list(TASK_IDS)
        or len(receipt.get("release", {}).get("identities", {})) != 18
        or receipt.get("payload_size") != len(payload_raw)
    ):
        raise Exact5PackageError("captured exact5 launch receipt replay differs")
    for path, raw, mode in (
        (input_path, input_raw, 0o444), (payload_path, payload_raw, 0o444),
        (receipt_path, receipt_raw, 0o400),
    ):
        artifacts[str(path.relative_to(root))] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
            "mode": mode,
        }

    static_source_relative = (
        "methods/bernini_action_editing/case01_source_bone_exact5_static_probe_v1.py"
    )
    static_source = diagnostic_sources[static_source_relative]
    static_payload_raw = build_static_payload(
        root=root, source=static_source,
        source_sha256=DIAGNOSTIC_SOURCE_FILES[static_source_relative],
        plan_sha256=plan_sha256,
        launch_receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
    )
    static_payload_path = root / "diagnostics/exact5_static_probe_payload_v1.sh"
    create_file(static_payload_path, static_payload_raw, 0o444)
    artifacts[str(static_payload_path.relative_to(root))] = {
        "sha256": hashlib.sha256(static_payload_raw).hexdigest(),
        "size": len(static_payload_raw), "mode": 0o444,
    }

    # Build an adjacent method shadow so the real exact5 root bootstrap can be
    # exercised with a no-torch fake runner while all other exact18 identities
    # retain their production bytes.  This tree is diagnostic, not release.
    shadow = root / "diagnostics/root_fake_release_shadow"
    mkdir_fresh(shadow)
    fake_relative = (
        "methods/bernini_action_editing/"
        "case01_source_bone_exact5_root_fake_runner_v1.py"
    )
    fake_raw = source_bytes[fake_relative]
    fake_sha256 = DIAGNOSTIC_SOURCE_FILES[fake_relative]
    shadow_sources = {
        "case01_source_bone_exact5_runner_v1.py": (fake_raw, fake_sha256),
        "full644_exploratory_matched_runner_auh_r5.py": (
            source_bytes["methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py"],
            RELEASE_FILES["methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py"],
        ),
        "case01_source_bone_exact5_eval_v1.py": (
            source_bytes["methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py"],
            RELEASE_FILES["methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py"],
        ),
        "full644_exploratory_matched_torchrun_fd_bridge_v2.py": (
            source_bytes["methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py"],
            RELEASE_FILES["methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py"],
        ),
        "full644_exploratory_matched_infer_adapter_auh_r5f.py": (
            source_bytes["methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5f.py"],
            RELEASE_FILES["methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5f.py"],
        ),
        "full644_exploratory_matched_infer_adapter_v2.py": (
            source_bytes["methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py"],
            RELEASE_FILES["methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py"],
        ),
        "full644_exploratory_matched_eval_v1.py": (
            source_bytes["methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py"],
            RELEASE_FILES["methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py"],
        ),
        "full644_exploratory_matched_eval_v2.py": (
            source_bytes["methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py"],
            RELEASE_FILES["methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py"],
        ),
        "action_preservation_decoded_eval_model_authority_v2.py": (
            source_bytes["methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py"],
            RELEASE_FILES["methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py"],
        ),
    }
    for name, (raw, digest_value) in shadow_sources.items():
        target = shadow / name
        create_file(target, raw, 0o444)
        artifacts[str(target.relative_to(root))] = {
            "sha256": digest_value, "size": len(raw), "mode": 0o444,
        }
    fake_input = dict(input_value)
    fake_input.update({
        "runner": str(shadow / "case01_source_bone_exact5_runner_v1.py"),
        "frozen_runner": str(shadow / "full644_exploratory_matched_runner_auh_r5.py"),
        "exact5_eval": str(shadow / "case01_source_bone_exact5_eval_v1.py"),
        "bridge": str(shadow / "full644_exploratory_matched_torchrun_fd_bridge_v2.py"),
        "adapter": str(shadow / "full644_exploratory_matched_infer_adapter_auh_r5f.py"),
        "base_adapter": str(shadow / "full644_exploratory_matched_infer_adapter_v2.py"),
        "eval_v1": str(shadow / "full644_exploratory_matched_eval_v1.py"),
        "eval_v2": str(shadow / "full644_exploratory_matched_eval_v2.py"),
        "model_authority": str(shadow / "action_preservation_decoded_eval_model_authority_v2.py"),
        "output_report": str(root / "evidence/exact5_root_fake_runner_probe_receipt_v1.json"),
        "runner_attestation": str(root / "diagnostics/unused-root-fake-attestation.json"),
        "authority_root": str(root / "diagnostics/unused-root-fake-authority"),
        "rank_cache_root": str(root / "diagnostics/unused-root-fake-rank-cache"),
    })
    fake_input_raw = canonical_json_bytes(fake_input) + b"\n"
    fake_input_path = root / "diagnostics/root_fake_launch_input_v1.json"
    create_file(fake_input_path, fake_input_raw, 0o444)
    original_runner_pin = launcher.EXPECTED_STATIC_SHA256["runner"]
    launcher.EXPECTED_STATIC_SHA256["runner"] = fake_sha256
    try:
        fake_payload_path = root / "diagnostics/root_fake_launch_payload_v1.sh"
        fake_receipt_path = root / "diagnostics/root_fake_launch_materialization_receipt_v1.json"
        fake_receipt = launcher.materialize(
            str(fake_input_path), str(fake_payload_path), str(fake_receipt_path),
        )
    finally:
        launcher.EXPECTED_STATIC_SHA256["runner"] = original_runner_pin
    fake_payload_raw = stable_file(
        fake_payload_path, fake_receipt["payload_sha256"], 0o444,
    )
    fake_receipt_raw = stable_file(fake_receipt_path, expected_mode=0o400)
    if (
        json.loads(fake_receipt_raw) != fake_receipt
        or fake_receipt.get("release", {}).get("identities", {}).get("runner", {}).get("sha256")
        != fake_sha256
        or len(fake_receipt.get("release", {}).get("identities", {})) != 18
    ):
        raise Exact5PackageError("root fake launch materialization replay differs")
    for path, raw, mode in (
        (fake_input_path, fake_input_raw, 0o444),
        (fake_payload_path, fake_payload_raw, 0o444),
        (fake_receipt_path, fake_receipt_raw, 0o400),
    ):
        artifacts[str(path.relative_to(root))] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
            "mode": mode,
        }
    report: dict[str, Any] = {
        "schema_version": "case01-source-bone-exact5-r64-materialization-v1",
        "status": "MATERIALIZED_NOT_SUBMITTED",
        "root": str(root), "holder_job_id": args.job_id,
        "expected_node": args.node, "campaign_mode": CAMPAIGN,
        "selected_task_ids": list(TASK_IDS), "task_count": 5,
        "physical_release_file_count": len(RELEASE_FILES),
        "production_identity_count": 18,
        "production_identity_decomposition": {
            "r5f_roles_with_exact5_wrapper_runner": 16,
            "additional_frozen_runner": 1, "additional_exact5_eval": 1,
        },
        "sealed_r5f_infer_lora_reused": True,
        "working_tree_infer_lora_read": False,
        "captured_materializer_sha256": args.materializer_sha256,
        "input_root": str(INPUT_ROOT),
        "independent_audit": {
            "path": str(audit_path), "sha256": AUDIT_SHA256,
            "size": AUDIT_SIZE, "audit_digest": authority["independent_audit_receipt_digest"],
        },
        "plan": {"path": str(plan_path), "sha256": plan_sha256,
                 "size": len(plan_raw), "plan_digest": plan["plan_digest"]},
        "launch": {
            "input": str(input_path), "input_sha256": hashlib.sha256(input_raw).hexdigest(),
            "payload": str(payload_path), "payload_sha256": receipt["payload_sha256"],
            "payload_size": receipt["payload_size"], "receipt": str(receipt_path),
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_digest": receipt["receipt_digest"],
            "release_digest": receipt["release_digest"],
        },
        "cpu_admission": {
            "required_before_gpu_attempt": True,
            "static_probe": {
                "source": str(static_source),
                "source_sha256": DIAGNOSTIC_SOURCE_FILES[static_source_relative],
                "payload": str(static_payload_path),
                "payload_sha256": hashlib.sha256(static_payload_raw).hexdigest(),
                "receipt": str(root / "evidence/exact5_static_probe_receipt_v1.json"),
                "executed": False,
            },
            "captured_root_fake_runner_probe": {
                "runner_sha256": fake_sha256,
                "payload": str(fake_payload_path),
                "payload_sha256": fake_receipt["payload_sha256"],
                "materialization_receipt": str(fake_receipt_path),
                "materialization_receipt_sha256": hashlib.sha256(fake_receipt_raw).hexdigest(),
                "execution_receipt": str(root / "evidence/exact5_root_fake_runner_probe_receipt_v1.json"),
                "executed": False,
            },
        },
        "rank_cache_root": str(RANK_CACHE_ROOT),
        "fresh_outputs": True, "fresh_final": True, "fresh_runtime": True,
        "publication_final_internal_paths_pairwise_disjoint": True,
        "slurm_step_launched": False, "gpu_attempt_claimed": False,
        "retry_allowed_after_gpu_attempt": False,
        "artifacts_before_materialization_receipt": artifacts,
    }
    report["receipt_digest"] = object_sha256(report)
    report_raw = canonical_json_bytes(report) + b"\n"
    report_path = root / "authority/package_materialization_receipt_v1.json"
    create_file(report_path, report_raw, 0o400)

    expected_files = set(artifacts) | {str(report_path.relative_to(root))}
    expected_directories = {
        ".", "release", "authority", "plan", "launch", "diagnostics",
        "diagnostics/root_fake_release_shadow", "evidence", "outputs",
        "outputs/media", "final", "logs", "runtime",
    }
    for relative in RELEASE_FILES:
        parent = Path("release") / Path(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent
    file_modes = {
        relative: (
            0o400 if relative == str(report_path.relative_to(root))
            else artifacts[relative]["mode"]
        )
        for relative in expected_files
    }
    staged_directory_modes = {relative: 0o700 for relative in expected_directories}
    opened = open_exact_tree(root, file_modes, staged_directory_modes)
    final_directory_modes = {
        relative: (
            0o555
            if relative == "release" or relative.startswith("release/")
            or relative in {"authority", "plan", "launch", "diagnostics/root_fake_release_shadow"}
            else 0o755
        )
        for relative in expected_directories
    }
    try:
        # Freeze only held directory objects obtained by the exact no-follow
        # traversal.  Never chmod a directory by a mutable named path.
        for relative in sorted(
            opened, key=lambda item: (len(Path(item).parts), item), reverse=True,
        ):
            descriptor = opened[relative]
            path = root if relative == "." else root / relative
            before = os.fstat(descriptor)
            named = os.lstat(path)
            if identity(before) != identity(named) or not stat.S_ISDIR(before.st_mode):
                raise Exact5PackageError("package directory changed before freeze")
            os.fchmod(descriptor, final_directory_modes[relative])
            after = os.fstat(descriptor)
            named_after = os.lstat(path)
            if (
                identity(after) != identity(named_after)
                or stat.S_IMODE(after.st_mode) != final_directory_modes[relative]
            ):
                raise Exact5PackageError("package directory freeze replay differs")
    finally:
        close_directories(opened)
    replayed = open_exact_tree(root, file_modes, final_directory_modes)
    close_directories(replayed)
    for relative, metadata in artifacts.items():
        stable_file(root / relative, metadata["sha256"], metadata["mode"])
    stable_file(
        report_path, hashlib.sha256(report_raw).hexdigest(), expected_mode=0o400,
    )
    for relative in ("evidence", "outputs/media", "final", "runtime"):
        if list((root / relative).iterdir()):
            raise Exact5PackageError(f"fresh result directory is not empty: {relative}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--materializer-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if (
        sys.platform != "linux" or not Path("/proc/self/fd").is_dir()
        or sys.flags.isolated != 1 or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1 or not sys.dont_write_bytecode
        or "torch" in sys.modules or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise Exact5PackageError("isolated materializer startup differs")
    report = _materialize(build_parser().parse_args(argv))
    print(canonical_json_bytes(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
