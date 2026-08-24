#!/usr/bin/env python3
"""Materialize the sealed case01 object-trajectory exact-five HOLD package.

No Slurm command is imported or invoked.  Before any package entry is created,
the final AUHv2 staging receipt, live exact15, exact35 snapshot manifest, and
snapshot publication receipt are replayed.  The package contains a release25,
six copied condition/admission authorities, an immutable non-launchable plan,
and create-only admission inputs.  The final object runner/evaluator bytes are
fixed to the independently reviewed terminal pins.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence


EXPERIMENTS = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments"
)
SOURCE_SNAPSHOT_ROOT = (
    EXPERIMENTS
    / "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1"
)
INPUT_ROOT = (
    EXPERIMENTS / "bernini_object_grounded_case01_0821_exact5_inputs_0a62b740_r1"
)
TARGET_ROOT = (
    EXPERIMENTS / "bernini_case01_object_trajectory_exact5_r64_canary_v1"
)
PACKAGE_PUBLICATION_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_r64_canary_v1."
    "publication_receipt_v2.json"
)
RANK_CACHE_ROOT = Path(
    "/tmp/bernini-case01-object-trajectory-exact5-r64-job143808-node292-r1-rank-cache"
)
VACE_PYTHON = Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12")
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
CAMPAIGN = "case01-object-trajectory-exact5-r64-engineering-oracle"
SNAPSHOT_MANIFEST_NAME = (
    "case01_object_trajectory_exact5_source_snapshot_manifest_v2.json"
)
MATERIALIZER_RELATIVE = (
    "methods/bernini_action_editing/tools/"
    "materialize_case01_object_trajectory_exact5_r64_hold_package_v1.py"
)
SNAPSHOT_BUILDER_RELATIVE = (
    "methods/bernini_action_editing/tools/"
    "build_case01_object_trajectory_exact5_source_snapshot_v1.py"
)
OLD_EXACT5_SNAPSHOT = (
    EXPERIMENTS
    / "bernini_object_grounded_case01_0821_exact5_source_snapshot_24_20260821_r1"
)
SNAPSHOT_STAGING_ROOT = (
    EXPERIMENTS / "bernini_case01_object_trajectory_exact5_source_staging_v1"
)
SNAPSHOT_STAGING_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_staging_v1.receipt_v1.json"
)
SNAPSHOT_PUBLICATION_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1."
    "receipt_v2.json"
)
STAGING_RECEIPT_COPY_RELATIVE = "authority/source_staging_receipt_auh_v2.json"
STAGING_SCHEMA = "case01-object-trajectory-exact5-source-stager-auh-v2"
STAGING_MANIFEST_SCHEMA = STAGING_SCHEMA + "-manifest"
STAGING_RECEIPT_SCHEMA = STAGING_SCHEMA + "-receipt"
STAGING_PUBLICATION_PROTOCOL = (
    "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation"
)
STAGING_REMOTE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
STAGING_REMOTE_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
STAGING_REMOTE_PYTHON_SIZE = 31_490_256
STAGING_BOOTSTRAP_SHA256 = (
    "33c63bb114d6008bd32c67819cd86fb4acce7b796696c7ed34f41a431836e08a"
)
STAGING_REMOTE_UID = 2012
STAGING_REMOTE_GID = 2000
STAGING_FILE_MODE = 0o444
STAGING_DIRECTORY_MODE = 0o555
STAGING_RECEIPT_MODE = 0o400
PACKAGE_PUBLICATION_SCHEMA = (
    "case01-object-trajectory-exact5-r64-package-publication-v2"
)
PUBLICATION_RESERVATION_MODE = 0o600
PUBLICATION_RECEIPT_MODE = 0o400

FINAL_RUNNER_SHA256 = "e47b81643c1d17e5099a9b33f16ca75521001ad52d2df2305b46b7e8c4d5ac4c"
FINAL_EVAL_SHA256 = "47cc871b82b8cf7762db9183997440eeabd287b1c702d9cd7421fd43e0a555e0"
FINAL_WRAPPER_SHA256 = "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9"

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
    "methods/bernini_action_editing/case01_object_trajectory_exact5_runner_v1.py":
        FINAL_RUNNER_SHA256,
    "methods/bernini_action_editing/case01_object_trajectory_exact5_eval_v1.py":
        FINAL_EVAL_SHA256,
    "methods/bernini_action_editing/case01_object_trajectory_exact5_spooled_launcher_auh_v1.py":
        "a81e812627125a24d72ec956b384e30479df379b66b5a94da871021c3e14267f",
    "methods/bernini_action_editing/infer_case01_object_trajectory_oracle_v1.py":
        FINAL_WRAPPER_SHA256,
    "methods/bernini_action_editing/object_trajectory_projection_v1.py":
        "a851afd220d770e6c8082ff8e7f0a0c2b5a5be894bcbf9feeaf8bca4cc6c9e7e",
    "methods/bernini_action_editing/case01_oracle_object_trajectory_v1.py":
        "a7d4e008e78d373370b832c0754e5d7420b504fa8b2022eef9a9bb899added8a",
    "methods/bernini_action_editing/infer_lora_full644_r5_frozen_acc46.py":
        "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
}
if len(RELEASE_FILES) != 25:
    raise RuntimeError("release25 closure differs")

DIAGNOSTIC_FILES = {
    "methods/bernini_action_editing/case01_object_trajectory_exact5_static_probe_v1.py":
        "071256da47635fc3481f51b48e7e5eddddc963a5345b1dda405473744d2c01a9",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_root_fake_runner_v1.py":
        "0d73fdaa4a4f1817f572eea471661850098ffe5aa54f54a9927c37a7e3f2a872",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_world4_probe_v1.py":
        "71c52ea3d7b36f07fdf5f9af3c9ecadf2020123795d9a4e10888a67eb0c7536b",
}

SNAPSHOT_AUTHORITY_FILES = {
    "artifacts/object_grounded_case01_0821_sam2_masklets_r2/receipt.json":
        "c9dc8567d4f147f943171d2d7abb55e52aadc685fbfc2f75fff4e837e2ab0b50",
    "methods/bernini_action_editing/assets/case01_288545b9c031491a_g0_sparse_annotations_v1.json":
        "e5185a1edd72fa8a1f2ece15e98c67d66e3fa65a2a9eb724bf06031c4d0e2020",
    "artifacts/case01_oracle_object_trajectory_v1/scaffold.json":
        "7b1bec6e9764a1297bb0029f8fea01ebe4b2deab0acc2c7f07fdee96bc0a098a",
    "md/action_editing/20260821_man/evidence/case01_object_trajectory_scaffold_independent_audit_v1.json":
        "acbe4a6e635e3429605a8aac4d655816fd6187ea7aec77d5a8b1e08a56a47e0e",
}
LEGACY_ALIAS_RELATIVE = (
    "methods/bernini_action_editing/infer_lora_full644_r5_frozen_acc46.py"
)
LEGACY_REUSED_PATHS = frozenset(tuple(RELEASE_FILES)[:18])
FORMAL_REVIEW_TEST = {
    "path": "methods/bernini_action_editing/tests/"
            "test_case01_object_trajectory_exact5_core_v1.py",
    "sha256": "a22895e47766c506fcb8265035ab9d7a91cae9a940c6e3ab5371a21673a2a8b4",
    "size": 50_880,
    "sealed_bytes_in_snapshot": False,
    "role": "formal_stop_review_evidence_not_runtime_or_release_authority",
}
if (
    len(LEGACY_REUSED_PATHS) != 18
    or len(set(RELEASE_FILES) | set(DIAGNOSTIC_FILES)
           | set(SNAPSHOT_AUTHORITY_FILES) | {MATERIALIZER_RELATIVE}) != 33
):
    raise RuntimeError("snapshot content33 decomposition differs")

SOURCE_VIDEO = ("videos/exact_original.mp4", "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18", 10_887_043)
AUX_VIDEO = ("videos/bone_removed.mp4", "8c525385832586fa7b7fd7ae6e5701c599694d26ee27b502dbf0bb582e55e1c9", 5_424_975)
CHECKPOINT = {
    "path": str(CHECKPOINT_MANIFEST), "pin_complete": True,
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
SHA_RE = re.compile(r"[0-9a-f]{64}")


class HoldPackageError(RuntimeError):
    pass


class PublicationCommittedError(HoldPackageError):
    """The package inode is committed but publication was not fully clean."""

    def __init__(self, identity: tuple[int, ...], observation: Mapping[str, Any]):
        super().__init__("package target committed without clean publication terminal")
        self.identity = identity
        self.observation = dict(observation)


class PublicationReceiptTerminalError(HoldPackageError):
    """A 0400 receipt commit is immutable but its terminal audit is unclear."""

    def __init__(self, observation: Mapping[str, Any]):
        super().__init__("immutable package receipt commit requires manual HOLD audit")
        self.observation = dict(observation)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def blocked_sources() -> tuple[str, ...]:
    return tuple(relative for relative, pin in RELEASE_FILES.items()
                 if SHA_RE.fullmatch(pin) is None)


def require_final_pins() -> None:
    blocked = blocked_sources()
    if blocked:
        raise HoldPackageError("HOLD: final source pins blocked: " + ",".join(blocked))


def require_fresh_package_paths(root: Path, rank_cache_root: Path) -> None:
    # lexists/lstat is intentional: a dangling symlink is occupied authority,
    # not a fresh cache path.
    if os.path.lexists(root) or os.path.lexists(rank_cache_root):
        raise HoldPackageError("fresh package/cache contract differs")


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_rdev, info.st_size,
        getattr(info, "st_blocks", 0), info.st_mtime_ns, info.st_ctime_ns,
    )


def _inode_anchor(info: os.stat_result) -> list[int]:
    return [
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(stat.S_IFMT(info.st_mode)),
    ]


def _read_fd(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1_048_576, size - offset), offset)
        if not block:
            break
        chunks.append(block); offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size or os.pread(descriptor, 1, size) != b"":
        raise HoldPackageError("held read is incomplete")
    return raw


def stable(
    path: Path, expected: str | None, size: int | None = None,
    *, expected_mode: int | None = None,
) -> bytes:
    """Read one authority without ever opening a named special file."""
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise HoldPackageError(f"noncanonical authority refused: {path}")
    try:
        named_before = os.lstat(path)
    except OSError as error:
        raise HoldPackageError(f"missing authority: {path}") from error
    if not stat.S_ISREG(named_before.st_mode) or named_before.st_nlink != 1:
        raise HoldPackageError(f"authority is not one regular single-link file: {path}")
    if path.resolve(strict=True) != path:
        raise HoldPackageError(f"authority resolves elsewhere: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or _identity(before) != _identity(named_before)
            or (size is not None and before.st_size != size)
            or (expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode)
        ):
            raise HoldPackageError(f"opened authority identity differs: {path}")
        pieces: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            piece = os.pread(
                descriptor, min(1_048_576, before.st_size - offset), offset,
            )
            if not piece:
                break
            pieces.append(piece); offset += len(piece)
        raw = b"".join(pieces)
        middle = os.fstat(descriptor)
        replay = b"".join(
            os.pread(descriptor, min(1_048_576, before.st_size - at), at)
            for at in range(0, before.st_size, 1_048_576)
        )
        eof = os.pread(descriptor, 1, before.st_size)
        after = os.fstat(descriptor)
        named_after = os.lstat(path)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(raw).hexdigest()
    if (
        len(raw) != before.st_size or replay != raw or eof != b""
        or _identity(before) != _identity(middle)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(named_after)
        or (expected is not None and digest != expected)
        or (size is not None and len(raw) != size)
        or (expected_mode is not None
            and stat.S_IMODE(before.st_mode) != expected_mode)
    ):
        raise HoldPackageError(f"stable authority differs: {path}")
    return raw


def create(path: Path, raw: bytes, mode: int) -> None:
    if os.path.lexists(path):
        raise HoldPackageError(f"target is not create-only: {path}")
    try:
        parent_info = os.lstat(path.parent)
    except OSError as error:
        raise HoldPackageError(f"target parent is missing: {path.parent}") from error
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or path.parent.resolve(strict=True) != path.parent
    ):
        raise HoldPackageError(f"target parent is not one real directory: {path.parent}")
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                raise HoldPackageError("write made no progress")
            offset += written
        os.fsync(fd); os.fchmod(fd, mode); os.fsync(fd)
    finally:
        os.close(fd)


def fsync_shadow_directories(root: Path) -> None:
    """Durably close every completed shadow directory, children first."""
    walked = [Path(directory) for directory, _subdirs, _files in os.walk(
        root, topdown=False,
    )]
    if not walked or walked[-1] != root:
        raise HoldPackageError("package shadow directory walk differs")
    for directory in walked:
        named_before = os.lstat(directory)
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            opened_before = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened_before.st_mode)
                or _identity(opened_before) != _identity(named_before)
            ):
                raise HoldPackageError(
                    f"package shadow directory identity differs: {directory}"
                )
            os.fsync(descriptor)
            opened_after = os.fstat(descriptor)
            named_after = os.lstat(directory)
            if (
                _identity(opened_after) != _identity(opened_before)
                or _identity(named_after) != _identity(opened_before)
            ):
                raise HoldPackageError(
                    f"package shadow directory changed while syncing: {directory}"
                )
        finally:
            os.close(descriptor)


def load_module_bytes(path: Path, raw: bytes, name: str) -> Any:
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    if spec is None:
        raise HoldPackageError("module spec creation failed")
    module = importlib.util.module_from_spec(spec); module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(raw.decode("utf-8", "strict"), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None); raise
    return module


def load_module(path: Path, pin: str, name: str) -> Any:
    return load_module_bytes(path, stable(path, pin), name)


def identity_row(path: Path, expected: str | None = None) -> dict[str, Any]:
    raw = stable(path, expected)
    observed = hashlib.sha256(raw).hexdigest()
    return {"path": str(path), "sha256": observed, "size": len(raw)}


def strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise HoldPackageError(f"duplicate JSON key in {label}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise HoldPackageError(f"invalid JSON authority: {label}") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise HoldPackageError(f"noncanonical JSON authority: {label}")
    return value


def expected_directories(files: set[str]) -> set[str]:
    result = {"."}
    for relative in files:
        parent = Path(relative).parent
        while str(parent) != ".":
            result.add(str(parent)); parent = parent.parent
    return result


def require_real_directory_chain(path: Path, *, label: str) -> None:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise HoldPackageError(f"{label} parent is not canonical")
    chain: list[Path] = []
    current = path
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    for component in reversed(chain):
        try:
            info = os.lstat(component)
        except OSError as error:
            raise HoldPackageError(f"missing {label} ancestor: {component}") from error
        if not stat.S_ISDIR(info.st_mode) or component.resolve(strict=True) != component:
            raise HoldPackageError(f"linked/non-directory {label} ancestor: {component}")


def open_held_parent(target: Path) -> int:
    require_real_directory_chain(target.parent, label="package target")
    descriptor = os.open(
        target.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    opened = os.fstat(descriptor); named = os.lstat(target.parent)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or _identity(opened) != _identity(named)
        or target.parent.resolve(strict=True) != target.parent
    ):
        os.close(descriptor)
        raise HoldPackageError("held package parent identity differs")
    return descriptor


class HeldPublicationReservation:
    def __init__(self, name: str, descriptor: int, anchor: list[int], raw: bytes):
        self.name = name
        self.descriptor = descriptor
        self.anchor = list(anchor)
        self.raw = raw

    def replay(self, parent_fd: int, *, expected_mode: int) -> bytes:
        opened = os.fstat(self.descriptor)
        named = os.stat(self.name, dir_fd=parent_fd, follow_symlinks=False)
        raw = _read_fd(self.descriptor, opened.st_size)
        if (
            _inode_anchor(opened) != self.anchor
            or _inode_anchor(named) != self.anchor
            or _identity(opened) != _identity(named)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != expected_mode
            or raw != self.raw
        ):
            raise HoldPackageError("held package publication reservation changed")
        return raw

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def create_publication_reservation(
    parent_fd: int, *, receipt_path: Path, target_root: Path,
) -> HeldPublicationReservation:
    if receipt_path.parent != target_root.parent or receipt_path.name in ("", ".", ".."):
        raise HoldPackageError("package publication receipt path differs")
    descriptor = os.open(
        receipt_path.name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0), 0, dir_fd=parent_fd,
    )
    try:
        anchor = _inode_anchor(os.fstat(descriptor))
        value: dict[str, Any] = {
            "schema_version": PACKAGE_PUBLICATION_SCHEMA + "-reservation",
            "status": "RESERVED_NOT_CONSUMPTION_AUTHORITY",
            "target_root": str(target_root),
            "receipt_path": str(receipt_path),
            "publication_protocol": STAGING_PUBLICATION_PROTOCOL,
            "rename_noreplace": False,
            "cooperative_writer_exclusion": True,
            "uncooperative_same_uid_race_out_of_scope": True,
            "retry_allowed": False,
            "receipt_inode_anchor": anchor,
            "receipt_is_consumption_gate": False,
            "receipt_is_admission": False,
        }
        value["reservation_digest"] = object_sha(value)
        raw = canonical(value) + b"\n"
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise HoldPackageError("package reservation write made no progress")
            offset += count
        os.fsync(descriptor); os.fchmod(descriptor, PUBLICATION_RESERVATION_MODE)
        os.fsync(descriptor); os.fsync(parent_fd)
        held = HeldPublicationReservation(receipt_path.name, descriptor, anchor, raw)
        held.replay(parent_fd, expected_mode=PUBLICATION_RESERVATION_MODE)
        return held
    except BaseException:
        os.close(descriptor)
        raise


def publish_under_reservation(
    parent_fd: int, shadow_name: str, target_name: str,
    reservation: HeldPublicationReservation,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """One NFS-compatible rename under a cooperative held O_EXCL gate."""
    reservation.replay(parent_fd, expected_mode=PUBLICATION_RESERVATION_MODE)
    try:
        os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise HoldPackageError("package target appeared before ordinary rename")
    shadow_fd = os.open(
        shadow_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        anchor = _inode_anchor(os.fstat(shadow_fd))
        rename_error: OSError | None = None
        try:
            os.rename(
                shadow_name, target_name,
                src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
            )
        except OSError as error:
            rename_error = error

        def observed_namespace() -> tuple[str, os.stat_result | None]:
            try:
                source = os.stat(
                    shadow_name, dir_fd=parent_fd, follow_symlinks=False,
                )
            except FileNotFoundError:
                source = None
            try:
                target = os.stat(
                    target_name, dir_fd=parent_fd, follow_symlinks=False,
                )
            except FileNotFoundError:
                target = None
            if source is None and target is not None and _inode_anchor(target) == anchor:
                return "target_same_inode_source_absent", target
            if source is not None and _inode_anchor(source) == anchor and target is None:
                return "source_same_inode_target_absent", None
            return "ambiguous_namespace", target

        namespace_state, _target = observed_namespace()
        if namespace_state != "target_same_inode_source_absent":
            if namespace_state == "source_same_inode_target_absent":
                raise HoldPackageError(
                    "ordinary package rename was not applied"
                    + ("" if rename_error is None else f": errno={rename_error.errno}")
                )
            raise HoldPackageError("ordinary package rename outcome is ambiguous")
        fsync_error: OSError | None = None
        try:
            os.fsync(parent_fd)
        except OSError as error:
            fsync_error = error
        namespace_state, named = observed_namespace()
        if namespace_state != "target_same_inode_source_absent" or named is None:
            raise HoldPackageError("package namespace changed after committed rename")
        opened = os.fstat(shadow_fd)
        if (
            _inode_anchor(opened) != anchor or _inode_anchor(named) != anchor
            or _identity(opened) != _identity(named)
        ):
            raise HoldPackageError("package publication inode continuity differs")
        reservation.replay(parent_fd, expected_mode=PUBLICATION_RESERVATION_MODE)
        observation = {
            "namespace_state": namespace_state,
            "rename_returned_zero": rename_error is None,
            "rename_error_errno": None if rename_error is None else rename_error.errno,
            "parent_fsync_returned_zero": fsync_error is None,
            "parent_fsync_error_errno": None if fsync_error is None else fsync_error.errno,
        }
        identity = _identity(opened)
        if rename_error is not None or fsync_error is not None:
            raise PublicationCommittedError(identity, observation)
        return identity, observation
    finally:
        os.close(shadow_fd)


def _audit_sealed_publication_receipt(
    parent_fd: int, reservation: HeldPublicationReservation,
    raw: bytes, materialized: Mapping[str, Any],
) -> dict[str, Any]:
    opened = os.fstat(reservation.descriptor)
    named = os.stat(
        reservation.name, dir_fd=parent_fd, follow_symlinks=False,
    )
    target_root_value = materialized.get("target_root")
    target_identity_value = materialized.get("target_root_identity")
    receipt_path_value = materialized.get("receipt_path")
    if (
        type(target_root_value) is not str
        or type(receipt_path_value) is not str
        or type(target_identity_value) is not list
        or len(target_identity_value) != 11
    ):
        raise HoldPackageError("package receipt target authority is incomplete")
    target_root = Path(target_root_value)
    receipt_path = Path(receipt_path_value)
    if (
        not target_root.is_absolute()
        or os.path.normpath(str(target_root)) != str(target_root)
        or not receipt_path.is_absolute()
        or os.path.normpath(str(receipt_path)) != str(receipt_path)
        or receipt_path.name != reservation.name
        or receipt_path.parent != target_root.parent
    ):
        raise HoldPackageError("package receipt publication paths differ")
    target_named = os.stat(
        target_root.name, dir_fd=parent_fd, follow_symlinks=False,
    )
    target_path_named = os.lstat(target_root)
    receipt_path_named = os.lstat(receipt_path)
    parsed = strict_json(raw, label="package publication receipt")
    if (
        _inode_anchor(opened) != reservation.anchor
        or _inode_anchor(named) != reservation.anchor
        or _inode_anchor(receipt_path_named) != reservation.anchor
        or _identity(opened) != _identity(named)
        or _identity(opened) != _identity(receipt_path_named)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) != PUBLICATION_RECEIPT_MODE
        or opened.st_size != len(raw)
        or _read_fd(reservation.descriptor, len(raw)) != raw
        or parsed != materialized
        or materialized.get("receipt_inode_anchor") != list(reservation.anchor)
        or _identity(target_named) != tuple(target_identity_value)
        or _identity(target_path_named) != tuple(target_identity_value)
        or not stat.S_ISDIR(target_named.st_mode)
    ):
        raise HoldPackageError("package sealed receipt/target audit differs")
    return parsed


def seal_publication_receipt(
    parent_fd: int, reservation: HeldPublicationReservation,
    value: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    reservation.replay(parent_fd, expected_mode=PUBLICATION_RESERVATION_MODE)
    materialized = dict(value)
    materialized["receipt_inode_anchor"] = list(reservation.anchor)
    materialized.pop("receipt_digest", None)
    materialized["receipt_digest"] = object_sha(materialized)
    raw = canonical(materialized) + b"\n"
    os.fchmod(reservation.descriptor, PUBLICATION_RESERVATION_MODE)
    os.ftruncate(reservation.descriptor, 0)
    offset = 0
    while offset < len(raw):
        count = os.pwrite(reservation.descriptor, raw[offset:], offset)
        if count <= 0:
            raise HoldPackageError("package receipt write made no progress")
        offset += count
    os.fsync(reservation.descriptor)
    fchmod_error: OSError | None = None
    try:
        os.fchmod(reservation.descriptor, PUBLICATION_RECEIPT_MODE)
    except OSError as error:
        fchmod_error = error
    try:
        parsed = _audit_sealed_publication_receipt(
            parent_fd, reservation, raw, materialized,
        )
    except BaseException as audit_error:
        opened = os.fstat(reservation.descriptor)
        named = os.stat(
            reservation.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        if (
            fchmod_error is not None
            and stat.S_IMODE(opened.st_mode) == PUBLICATION_RESERVATION_MODE
            and stat.S_IMODE(named.st_mode) == PUBLICATION_RESERVATION_MODE
            and _inode_anchor(opened) == reservation.anchor
            and _inode_anchor(named) == reservation.anchor
            and _identity(opened) == _identity(named)
            and _read_fd(reservation.descriptor, opened.st_size) == raw
        ):
            reservation.raw = raw
            raise HoldPackageError(
                "package receipt fchmod was not applied; 0600 reservation remains"
            ) from fchmod_error
        raise PublicationReceiptTerminalError({
            "classification": "immutable_0400_receipt_commit_ambiguous_manual_hold",
            "fchmod_returned_zero": fchmod_error is None,
            "fchmod_error_errno": (
                None if fchmod_error is None else fchmod_error.errno
            ),
            "observed_mode": stat.S_IMODE(opened.st_mode),
            "audit_error": type(audit_error).__name__,
        }) from audit_error
    reservation.raw = raw
    receipt_fsync_error: OSError | None = None
    parent_fsync_error: OSError | None = None
    try:
        os.fsync(reservation.descriptor)
    except OSError as error:
        receipt_fsync_error = error
    try:
        os.fsync(parent_fd)
    except OSError as error:
        parent_fsync_error = error
    try:
        parsed = _audit_sealed_publication_receipt(
            parent_fd, reservation, raw, materialized,
        )
    except BaseException as audit_error:
        raise PublicationReceiptTerminalError({
            "classification": "immutable_0400_receipt_postseal_ambiguous_manual_hold",
            "fchmod_returned_zero": fchmod_error is None,
            "fchmod_error_errno": (
                None if fchmod_error is None else fchmod_error.errno
            ),
            "receipt_fsync_returned_zero": receipt_fsync_error is None,
            "receipt_fsync_error_errno": (
                None if receipt_fsync_error is None
                else receipt_fsync_error.errno
            ),
            "parent_fsync_returned_zero": parent_fsync_error is None,
            "parent_fsync_error_errno": (
                None if parent_fsync_error is None else parent_fsync_error.errno
            ),
            "audit_error": type(audit_error).__name__,
        }) from audit_error
    if fchmod_error is not None or receipt_fsync_error is not None or parent_fsync_error is not None:
        try:
            reservation.replay(parent_fd, expected_mode=PUBLICATION_RECEIPT_MODE)
        except BaseException as audit_error:
            raise PublicationReceiptTerminalError({
                "classification": "immutable_0400_receipt_replay_ambiguous_manual_hold",
                "audit_error": type(audit_error).__name__,
            }) from audit_error
    reservation.replay(parent_fd, expected_mode=PUBLICATION_RECEIPT_MODE)
    return raw, parsed


def validate_shadow_package(
    root: Path, expected: Mapping[str, tuple[str, int, int]],
    *, extra_directories: set[str],
) -> None:
    actual: set[str] = set()
    actual_directories = {"."}
    pending = [(root, ".")]
    while pending:
        directory, prefix = pending.pop()
        info = os.lstat(directory)
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o022:
            raise HoldPackageError(f"shadow package directory differs: {prefix}")
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = entry.name if prefix == "." else f"{prefix}/{entry.name}"
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    actual_directories.add(relative)
                    pending.append((Path(entry.path), relative))
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    actual.add(relative)
                else:
                    raise HoldPackageError(
                        f"shadow package special/link entry refused: {relative}"
                    )
    if (
        actual != set(expected)
        or actual_directories
        != expected_directories(set(expected)) | extra_directories
    ):
        raise HoldPackageError("shadow package exact file closure differs")
    for relative, (digest, size, mode) in expected.items():
        stable(root / relative, digest, size, expected_mode=mode)


def validate_exact_tree(
    root: Path, *, expected_files: set[str], file_mode: int,
    directory_mode: int, file_modes: Mapping[str, int] | None = None,
    expected_uid: int | None = None, expected_gid: int | None = None,
) -> None:
    if not root.is_absolute() or os.path.normpath(str(root)) != str(root):
        raise HoldPackageError("snapshot root is not canonical")
    try:
        root_info = os.lstat(root)
    except OSError as error:
        raise HoldPackageError("snapshot root is missing") from error
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or stat.S_IMODE(root_info.st_mode) != directory_mode
        or root.resolve(strict=True) != root
    ):
        raise HoldPackageError("snapshot root is not one sealed real directory")
    actual_files: set[str] = set()
    actual_directories = {"."}
    pending = [(root, ".")]
    while pending:
        directory, prefix = pending.pop()
        info = os.lstat(directory)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != directory_mode
            or (expected_uid is not None and info.st_uid != expected_uid)
            or (expected_gid is not None and info.st_gid != expected_gid)
        ):
            raise HoldPackageError(f"snapshot directory differs: {prefix}")
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = entry.name if prefix == "." else f"{prefix}/{entry.name}"
                child = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(child.st_mode):
                    expected_mode = (
                        file_mode if file_modes is None
                        else file_modes.get(relative, file_mode)
                    )
                    if (
                        child.st_nlink != 1
                        or stat.S_IMODE(child.st_mode) != expected_mode
                        or (expected_uid is not None and child.st_uid != expected_uid)
                        or (expected_gid is not None and child.st_gid != expected_gid)
                    ):
                        raise HoldPackageError(f"snapshot file differs: {relative}")
                    actual_files.add(relative)
                elif stat.S_ISDIR(child.st_mode):
                    actual_directories.add(relative)
                    pending.append((Path(entry.path), relative))
                else:
                    raise HoldPackageError(
                        f"snapshot special/link entry refused: {relative}"
                    )
    if (
        actual_files != expected_files
        or actual_directories != expected_directories(expected_files)
    ):
        raise HoldPackageError("snapshot physical exact-tree closure differs")


def snapshot_expected_files(materializer_sha256: str) -> dict[str, str]:
    if SHA_RE.fullmatch(materializer_sha256) is None:
        raise HoldPackageError("materializer SHA pin is incomplete")
    result = {
        **RELEASE_FILES, **DIAGNOSTIC_FILES, **SNAPSHOT_AUTHORITY_FILES,
        MATERIALIZER_RELATIVE: materializer_sha256,
    }
    if len(result) != 33:
        raise HoldPackageError("snapshot content33 key closure differs")
    return result


def ordered_directories(files: Sequence[str] | set[str]) -> list[str]:
    return sorted(expected_directories(set(files)), key=lambda value: (value.count("/"), value))


def expected_staging_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": STAGING_MANIFEST_SCHEMA,
        "target_root": str(SNAPSHOT_STAGING_ROOT),
        "receipt_path": str(SNAPSHOT_STAGING_RECEIPT_PATH),
        "remote_python": {
            "path": str(STAGING_REMOTE_PYTHON),
            "sha256": STAGING_REMOTE_PYTHON_SHA256,
            "size": STAGING_REMOTE_PYTHON_SIZE,
        },
        "remote_uid": STAGING_REMOTE_UID,
        "remote_gid": STAGING_REMOTE_GID,
        "file_count": 15,
        "file_mode": STAGING_FILE_MODE,
        "directory_mode": STAGING_DIRECTORY_MODE,
        "receipt_mode": STAGING_RECEIPT_MODE,
        "directories": ordered_directories([row["relative"] for row in rows]),
        "files": [dict(row) for row in rows],
        "publication_protocol": STAGING_PUBLICATION_PROTOCOL,
        "whole_tree_atomically_visible": True,
        "rename_noreplace": False,
        "cooperative_writer_exclusion": True,
        "target_absent_rechecked": True,
        "receipt_is_consumption_gate": True,
        "uncooperative_same_uid_race_out_of_scope": True,
        "launch_allowed": False,
    }
    value["manifest_digest"] = object_sha(value)
    return value


def validate_staging_receipt_value(
    value: Mapping[str, Any], *, receipt_info: os.stat_result,
    root_identity: tuple[int, ...], rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = {
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
    unsigned = dict(value); claimed = unsigned.pop("receipt_digest", None)
    manifest = expected_staging_manifest(rows)
    tree_rows = [
        {"relative": row["relative"], "sha256": row["sha256"],
         "size": row["size"], "mode": STAGING_FILE_MODE, "nlink": 1}
        for row in rows
    ]
    operation = value.get("operation")
    if operation == "stage":
        status = "STAGED_RECEIPT_GATED"
        observation = {
            "kind": "live_posix_rename_under_held_receipt_reservation",
            "root_identity": list(root_identity),
            "held_inode_continuity": True,
            "ordinary_posix_rename_performed_this_operation": True,
            "rename_noreplace_performed_this_operation": False,
            "target_absent_rechecked_before_rename": True,
            "whole_tree_atomically_visible": True,
            "historical_replacement_claim": "not_made",
        }
        payload_relation = value.get("request_payload_sha256") == value.get(
            "stage_payload_sha256"
        )
        terminal_relation = value.get("commit_terminal_digest") is None
    elif operation == "recover-receipt":
        status = "RECOVERED_RECEIPT_ONLY"
        observation = {
            "kind": "recovered_existing_exact15_current_inode",
            "root_identity": list(root_identity),
            "held_inode_continuity": True,
            "ordinary_posix_rename_performed_this_operation": False,
            "rename_noreplace_performed_this_operation": False,
            "target_absent_rechecked_before_rename": False,
            "whole_tree_atomically_visible": True,
            "historical_replacement_claim": "not_made",
        }
        payload_relation = True
        terminal_relation = (
            type(value.get("commit_terminal_digest")) is str
            and SHA_RE.fullmatch(value["commit_terminal_digest"]) is not None
        )
    else:
        raise HoldPackageError("staging receipt operation differs")
    if (
        set(value) != fields or claimed != object_sha(unsigned)
        or value.get("schema_version") != STAGING_RECEIPT_SCHEMA
        or value.get("status") != status
        or value.get("target_root") != str(SNAPSHOT_STAGING_ROOT)
        or value.get("receipt_path") != str(SNAPSHOT_STAGING_RECEIPT_PATH)
        or value.get("manifest_digest") != manifest["manifest_digest"]
        or any(
            type(value.get(key)) is not str or SHA_RE.fullmatch(value[key]) is None
            for key in ("request_payload_sha256", "stage_payload_sha256",
                        "bootstrap_source_sha256")
        )
        or value.get("bootstrap_source_sha256") != STAGING_BOOTSTRAP_SHA256
        or not payload_relation or not terminal_relation
        or value.get("file_count") != 15 or value.get("files") != tree_rows
        or value.get("directories") != manifest["directories"]
        or value.get("file_mode") != STAGING_FILE_MODE
        or value.get("directory_mode") != STAGING_DIRECTORY_MODE
        or value.get("receipt_mode") != STAGING_RECEIPT_MODE
        or value.get("held_parent_identity_replayed") is not True
        or value.get("ancestor_chain_nofollow") is not True
        or value.get("publication_protocol") != STAGING_PUBLICATION_PROTOCOL
        or value.get("rename_noreplace") is not False
        or value.get("cooperative_writer_exclusion") is not True
        or value.get("receipt_is_consumption_gate") is not True
        or value.get("receipt_is_admission") is not True
        or value.get("uncooperative_same_uid_race_out_of_scope") is not True
        or value.get("target_observation") != observation
        or value.get("receipt_inode_anchor") != _inode_anchor(receipt_info)
        or receipt_info.st_uid != STAGING_REMOTE_UID
        or receipt_info.st_gid != STAGING_REMOTE_GID
        or value.get("launch_allowed") is not False
    ):
        raise HoldPackageError("final AUHv2 staging receipt closure differs")


def replay_staging_receipt_authority(
    copied_raw: bytes, rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the copied receipt to its same-byte live receipt and exact15 tree."""
    receipt_raw = stable(
        SNAPSHOT_STAGING_RECEIPT_PATH, hashlib.sha256(copied_raw).hexdigest(),
        len(copied_raw), expected_mode=STAGING_RECEIPT_MODE,
    )
    if receipt_raw != copied_raw:
        raise HoldPackageError("copied/live staging receipt bytes differ")
    value = strict_json(receipt_raw, label="final AUHv2 staging receipt")
    root_fd = os.open(
        SNAPSHOT_STAGING_ROOT,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(root_fd); named = os.lstat(SNAPSHOT_STAGING_ROOT)
        root_identity = _identity(before)
        if (
            _identity(named) != root_identity
            or not stat.S_ISDIR(before.st_mode)
            or stat.S_IMODE(before.st_mode) != STAGING_DIRECTORY_MODE
            or before.st_uid != STAGING_REMOTE_UID
            or before.st_gid != STAGING_REMOTE_GID
            or SNAPSHOT_STAGING_ROOT.resolve(strict=True) != SNAPSHOT_STAGING_ROOT
        ):
            raise HoldPackageError("live staging root identity differs")
        expected = {row["relative"]: row for row in rows}
        validate_exact_tree(
            SNAPSHOT_STAGING_ROOT, expected_files=set(expected),
            file_mode=STAGING_FILE_MODE, directory_mode=STAGING_DIRECTORY_MODE,
            expected_uid=STAGING_REMOTE_UID, expected_gid=STAGING_REMOTE_GID,
        )
        for relative, row in expected.items():
            stable_raw = stable(
                SNAPSHOT_STAGING_ROOT / relative, row["sha256"], row["size"],
                expected_mode=STAGING_FILE_MODE,
            )
            source_info = os.lstat(SNAPSHOT_STAGING_ROOT / relative)
            if (
                source_info.st_uid != STAGING_REMOTE_UID
                or source_info.st_gid != STAGING_REMOTE_GID
                or len(stable_raw) != row["size"]
            ):
                raise HoldPackageError(f"live staging leaf owner differs: {relative}")
        after = os.fstat(root_fd); named_after = os.lstat(SNAPSHOT_STAGING_ROOT)
        if _identity(after) != root_identity or _identity(named_after) != root_identity:
            raise HoldPackageError("live staging root changed during replay")
        receipt_info = os.lstat(SNAPSHOT_STAGING_RECEIPT_PATH)
        validate_staging_receipt_value(
            value, receipt_info=receipt_info,
            root_identity=root_identity, rows=rows,
        )
    finally:
        os.close(root_fd)
    return {
        "source_path": str(SNAPSHOT_STAGING_RECEIPT_PATH),
        "snapshot_relative": STAGING_RECEIPT_COPY_RELATIVE,
        "sha256": hashlib.sha256(copied_raw).hexdigest(),
        "size": len(copied_raw), "mode": STAGING_RECEIPT_MODE,
        "schema_version": STAGING_RECEIPT_SCHEMA,
        "receipt_digest": value["receipt_digest"],
        "staging_manifest_digest": value["manifest_digest"],
        "staging_target_root_identity": value["target_observation"]["root_identity"],
        "staging_file_count": 15, "copied_as_snapshot_leaf": True,
        "replayed_before_and_after_snapshot_build": True,
    }


def preflight_snapshot(
    source_root: Path, *, manifest_sha256: str, materializer_sha256: str,
    require_configured_root: bool = True,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Validate/replay the receipt-gated exact35 snapshot before publication."""
    if require_configured_root and source_root != SOURCE_SNAPSHOT_ROOT:
        raise HoldPackageError("source root is not the configured exact35 snapshot")
    if SHA_RE.fullmatch(manifest_sha256) is None:
        raise HoldPackageError("snapshot manifest SHA pin is incomplete")
    publication_receipt_path = (
        SNAPSHOT_PUBLICATION_RECEIPT_PATH
        if require_configured_root
        else source_root.with_name(source_root.name + ".receipt_v2.json")
    )
    # The sibling 0400 publication receipt is consumed before the snapshot
    # root.  A reservation/0600/partial/tampered receipt cannot authorize use.
    publication_receipt_raw = stable(
        publication_receipt_path, None, expected_mode=PUBLICATION_RECEIPT_MODE,
    )
    publication_receipt = strict_json(
        publication_receipt_raw, label="snapshot publication receipt",
    )
    manifest_raw = stable(
        source_root / SNAPSHOT_MANIFEST_NAME, manifest_sha256,
        expected_mode=0o444,
    )
    manifest = strict_json(manifest_raw, label="exact35 snapshot manifest")
    unsigned = dict(manifest); claimed = unsigned.pop("manifest_digest", None)
    rows = manifest.get("files")
    fields = {
        "schema_version", "status", "launch_allowed", "old_snapshot_root",
        "staging_root", "staging_receipt_path",
        "snapshot_publication_receipt_path", "target_root", "content_leaf_count",
        "physical_file_count_including_manifest", "release_file_count",
        "legacy_alias_is_distinct_regular_inode", "formal_review_test",
        "builder_authority", "staging_receipt_authority",
        "publication_protocol", "rename_noreplace",
        "cooperative_writer_exclusion", "target_absent_rechecked",
        "whole_tree_atomically_visible",
        "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
        "files", "manifest_digest",
    }
    builder_authority = manifest.get("builder_authority")
    staging_authority = manifest.get("staging_receipt_authority")
    expected_base = snapshot_expected_files(materializer_sha256)
    if (
        set(manifest) != fields
        or manifest.get("schema_version")
        != "case01-object-trajectory-exact5-source-snapshot-v2"
        or manifest.get("status") != "SEALED_SOURCE_ONLY_NOT_LAUNCHABLE"
        or manifest.get("launch_allowed") is not False
        or manifest.get("old_snapshot_root") != str(OLD_EXACT5_SNAPSHOT)
        or manifest.get("staging_root") != str(SNAPSHOT_STAGING_ROOT)
        or manifest.get("staging_receipt_path")
        != str(SNAPSHOT_STAGING_RECEIPT_PATH)
        or manifest.get("snapshot_publication_receipt_path")
        != str(publication_receipt_path)
        or manifest.get("target_root") != str(source_root)
        or manifest.get("content_leaf_count") != 34
        or manifest.get("physical_file_count_including_manifest") != 35
        or manifest.get("release_file_count") != 25
        or manifest.get("legacy_alias_is_distinct_regular_inode") is not True
        or manifest.get("formal_review_test") != FORMAL_REVIEW_TEST
        or manifest.get("publication_protocol") != STAGING_PUBLICATION_PROTOCOL
        or manifest.get("rename_noreplace") is not False
        or manifest.get("cooperative_writer_exclusion") is not True
        or manifest.get("target_absent_rechecked") is not True
        or manifest.get("whole_tree_atomically_visible") is not True
        or manifest.get("uncooperative_same_uid_race_out_of_scope") is not True
        or manifest.get("retry_allowed") is not False
        or type(builder_authority) is not dict
        or set(builder_authority) != {
            "path", "sha256", "size", "sealed_bytes_in_snapshot"
        }
        or builder_authority.get("path")
        != str(SNAPSHOT_STAGING_ROOT / SNAPSHOT_BUILDER_RELATIVE)
        or SHA_RE.fullmatch(builder_authority.get("sha256", "")) is None
        or type(builder_authority.get("size")) is not int
        or builder_authority["size"] <= 0
        or builder_authority.get("sealed_bytes_in_snapshot") is not False
        or type(staging_authority) is not dict
        or set(staging_authority) != {
            "source_path", "snapshot_relative", "sha256", "size", "mode",
            "schema_version", "receipt_digest", "staging_manifest_digest",
            "staging_target_root_identity", "staging_file_count",
            "copied_as_snapshot_leaf",
            "replayed_before_and_after_snapshot_build",
        }
        or staging_authority.get("source_path")
        != str(SNAPSHOT_STAGING_RECEIPT_PATH)
        or staging_authority.get("snapshot_relative")
        != STAGING_RECEIPT_COPY_RELATIVE
        or SHA_RE.fullmatch(staging_authority.get("sha256", "")) is None
        or type(staging_authority.get("size")) is not int
        or staging_authority["size"] <= 0
        or staging_authority.get("mode") != STAGING_RECEIPT_MODE
        or staging_authority.get("schema_version") != STAGING_RECEIPT_SCHEMA
        or SHA_RE.fullmatch(staging_authority.get("receipt_digest", "")) is None
        or SHA_RE.fullmatch(
            staging_authority.get("staging_manifest_digest", "")
        ) is None
        or type(staging_authority.get("staging_target_root_identity")) is not list
        or len(staging_authority["staging_target_root_identity"]) != 11
        or staging_authority.get("staging_file_count") != 15
        or staging_authority.get("copied_as_snapshot_leaf") is not True
        or staging_authority.get(
            "replayed_before_and_after_snapshot_build"
        ) is not True
        or claimed != hashlib.sha256(canonical(unsigned)).hexdigest()
        or type(rows) is not list or len(rows) != 34
    ):
        raise HoldPackageError("exact35 snapshot manifest closure differs")
    expected = {
        **expected_base,
        STAGING_RECEIPT_COPY_RELATIVE: staging_authority["sha256"],
    }
    if (
        [row.get("path") if type(row) is dict else None for row in rows]
        != sorted(expected)
    ):
        raise HoldPackageError("exact35 snapshot row ordering differs")
    file_modes = {relative: 0o444 for relative in expected}
    file_modes[STAGING_RECEIPT_COPY_RELATIVE] = STAGING_RECEIPT_MODE
    file_modes[SNAPSHOT_MANIFEST_NAME] = 0o444
    validate_exact_tree(
        source_root,
        expected_files=set(expected) | {SNAPSHOT_MANIFEST_NAME},
        file_mode=0o444, directory_mode=0o555, file_modes=file_modes,
    )
    raw_by_relative: dict[str, bytes] = {}
    for row in rows:
        relative = row["path"]
        provenance = (
            "copied_exact_auh_v2_staging_receipt_authority"
            if relative == STAGING_RECEIPT_COPY_RELATIVE
            else "independent_inode_copy_of_sealed_legacy_infer"
            if relative == LEGACY_ALIAS_RELATIVE
            else "sealed_legacy_exact5_snapshot"
            if relative in LEGACY_REUSED_PATHS
            else "fresh_pinned_staging"
        )
        if (
            set(row) != {"path", "sha256", "size", "mode", "provenance"}
            or row.get("sha256") != expected[relative]
            or type(row.get("size")) is not int or row["size"] <= 0
            or row.get("mode") != file_modes[relative]
            or row.get("provenance") != provenance
        ):
            raise HoldPackageError(f"exact35 manifest row differs: {relative}")
        raw = stable(
            source_root / relative, expected[relative], row["size"],
            expected_mode=file_modes[relative],
        )
        raw_by_relative[relative] = raw
    executing_raw = stable(Path(__file__).resolve(), materializer_sha256)
    if executing_raw != raw_by_relative[MATERIALIZER_RELATIVE]:
        raise HoldPackageError("executing materializer differs from snapshot leaf")
    fresh_rows = [
        {"relative": row["path"], "sha256": row["sha256"], "size": row["size"]}
        for row in rows if row["provenance"] == "fresh_pinned_staging"
    ]
    fresh_rows.append({
        "relative": SNAPSHOT_BUILDER_RELATIVE,
        "sha256": builder_authority["sha256"],
        "size": builder_authority["size"],
    })
    fresh_rows.sort(key=lambda row: row["relative"])
    if len(fresh_rows) != 15:
        raise HoldPackageError("staging exact15 rows cannot be reconstructed")
    replayed_staging = replay_staging_receipt_authority(
        raw_by_relative[STAGING_RECEIPT_COPY_RELATIVE], fresh_rows,
    )
    if replayed_staging != staging_authority:
        raise HoldPackageError("snapshot staging receipt authority differs")
    publication_fields = {
        "schema_version", "status", "target_root", "receipt_path",
        "manifest_path", "manifest_sha256", "manifest_digest",
        "staging_receipt_sha256", "staging_receipt_digest",
        "content_leaf_count", "physical_file_count_including_manifest",
        "publication_protocol", "rename_noreplace",
        "cooperative_writer_exclusion",
        "target_absent_rechecked_before_rename",
        "ordinary_posix_rename_performed", "publication_observation",
        "whole_tree_atomically_visible",
        "uncooperative_same_uid_race_out_of_scope", "retry_allowed",
        "target_root_identity", "receipt_mode", "receipt_is_consumption_gate",
        "receipt_is_admission",
        "launch_allowed", "receipt_inode_anchor", "receipt_digest",
    }
    publication_unsigned = dict(publication_receipt)
    publication_claimed = publication_unsigned.pop("receipt_digest", None)
    source_identity = _identity(os.lstat(source_root))
    publication_info = os.lstat(publication_receipt_path)
    if (
        set(publication_receipt) != publication_fields
        or publication_claimed != object_sha(publication_unsigned)
        or publication_receipt.get("schema_version")
        != "case01-object-trajectory-exact5-source-snapshot-publication-v2-receipt"
        or publication_receipt.get("status") != "PUBLISHED_RECEIPT_GATED"
        or publication_receipt.get("target_root") != str(source_root)
        or publication_receipt.get("receipt_path") != str(publication_receipt_path)
        or publication_receipt.get("manifest_path")
        != str(source_root / SNAPSHOT_MANIFEST_NAME)
        or publication_receipt.get("manifest_sha256") != manifest_sha256
        or publication_receipt.get("manifest_digest") != claimed
        or publication_receipt.get("staging_receipt_sha256")
        != staging_authority["sha256"]
        or publication_receipt.get("staging_receipt_digest")
        != staging_authority["receipt_digest"]
        or publication_receipt.get("content_leaf_count") != 34
        or publication_receipt.get("physical_file_count_including_manifest") != 35
        or publication_receipt.get("publication_protocol")
        != STAGING_PUBLICATION_PROTOCOL
        or publication_receipt.get("rename_noreplace") is not False
        or publication_receipt.get("cooperative_writer_exclusion") is not True
        or publication_receipt.get(
            "target_absent_rechecked_before_rename"
        ) is not True
        or publication_receipt.get("ordinary_posix_rename_performed") is not True
        or publication_receipt.get("publication_observation") != {
            "namespace_state": "target_same_inode_source_absent",
            "rename_returned_zero": True,
            "rename_error_errno": None,
            "parent_fsync_returned_zero": True,
            "parent_fsync_error_errno": None,
        }
        or publication_receipt.get("whole_tree_atomically_visible") is not True
        or publication_receipt.get(
            "uncooperative_same_uid_race_out_of_scope"
        ) is not True
        or publication_receipt.get("retry_allowed") is not False
        or publication_receipt.get("target_root_identity") != list(source_identity)
        or publication_receipt.get("receipt_mode") != PUBLICATION_RECEIPT_MODE
        or publication_receipt.get("receipt_is_consumption_gate") is not True
        or publication_receipt.get("receipt_is_admission") is not True
        or publication_receipt.get("launch_allowed") is not False
        or publication_receipt.get("receipt_inode_anchor")
        != _inode_anchor(publication_info)
    ):
        raise HoldPackageError("snapshot publication receipt closure differs")
    # Close the preflight with a second receipt/live-tree replay.
    if replay_staging_receipt_authority(
        raw_by_relative[STAGING_RECEIPT_COPY_RELATIVE], fresh_rows,
    ) != staging_authority:
        raise HoldPackageError("snapshot staging receipt post-replay differs")
    return raw_by_relative, {
        "path": str(source_root / SNAPSHOT_MANIFEST_NAME),
        "sha256": manifest_sha256, "size": len(manifest_raw),
        "manifest_digest": claimed, "content_leaf_count": 34,
        "physical_file_count": 35,
        "staging_receipt_authority": staging_authority,
        "snapshot_publication_receipt": {
            "path": str(publication_receipt_path),
            "sha256": hashlib.sha256(publication_receipt_raw).hexdigest(),
            "size": len(publication_receipt_raw),
            "receipt_digest": publication_receipt["receipt_digest"],
        },
        "staging_rows": fresh_rows,
    }


def real_directory(path: Path, *, label: str) -> Path:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise HoldPackageError(f"missing {label} directory: {path}") from error
    if not stat.S_ISDIR(info.st_mode):
        raise HoldPackageError(f"noncanonical {label} directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise HoldPackageError(f"noncanonical {label} directory: {path}")
    return resolved


def _runtime_identities_from_preflight(
    root: Path, plan_path: Path, plan_raw: bytes, launcher: Any,
    release_bytes: Mapping[str, bytes],
    runtime_preflight: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    method_prefix = "methods/bernini_action_editing/"
    internal = {
        "runner": method_prefix + "case01_object_trajectory_exact5_runner_v1.py",
        "legacy_exact5_runner": method_prefix + "case01_source_bone_exact5_runner_v1.py",
        "object_eval": method_prefix + "case01_object_trajectory_exact5_eval_v1.py",
        "legacy_exact5_eval": method_prefix + "case01_source_bone_exact5_eval_v1.py",
        "frozen_runner": method_prefix + "full644_exploratory_matched_runner_auh_r5.py",
        "bridge": method_prefix + "full644_exploratory_matched_torchrun_fd_bridge_v2.py",
        "adapter": method_prefix + "infer_case01_object_trajectory_oracle_v1.py",
        "legacy_infer_alias": method_prefix + "infer_lora_full644_r5_frozen_acc46.py",
        "trajectory_projection": method_prefix + "object_trajectory_projection_v1.py",
        "trajectory_scaffold_module": method_prefix + "case01_oracle_object_trajectory_v1.py",
        "base_adapter": method_prefix + "full644_exploratory_matched_infer_adapter_v2.py",
        "eval_v1": method_prefix + "full644_exploratory_matched_eval_v1.py",
        "eval_v2": method_prefix + "full644_exploratory_matched_eval_v2.py",
        "model_authority": method_prefix + "action_preservation_decoded_eval_model_authority_v2.py",
        "base_model_manifest": method_prefix + "audits/bernini_r13_ff4c5d4_checkpoint.sha256",
    }
    result: dict[str, Any] = {}
    for role in launcher.IDENTITY_ROLES:
        if role in internal:
            relative = internal[role]; raw = release_bytes[relative]
            row = {
                "path": str(root / "release" / relative),
                "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
            }
        elif role == "plan":
            row = {
                "path": str(plan_path),
                "sha256": hashlib.sha256(plan_raw).hexdigest(),
                "size": len(plan_raw),
            }
        else:
            source = runtime_preflight.get(role)
            if not isinstance(source, Mapping):
                raise HoldPackageError(f"missing runtime preflight row: {role}")
            row = dict(source)
        expected = launcher.EXPECTED_STATIC_SHA256.get(role)
        if expected is not None and row.get("sha256") != expected:
            raise HoldPackageError(f"preflight identity SHA differs: {role}")
        result[role] = row
    if tuple(result) != launcher.IDENTITY_ROLES or len(result) != 25:
        raise HoldPackageError("exact25 runtime ordering differs")
    return result


def _file_authority_from_bytes(
    eval_module: Any, path: Path, raw: bytes, *, role: str,
    payload_digest: str | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256(raw).hexdigest()
    row: dict[str, Any] = {
        "schema_version": eval_module.FILE_AUTHORITY_SCHEMA,
        "role": role, "complete": True, "path": str(path),
        "sha256": digest, "size": len(raw),
        "payload_digest": digest if payload_digest is None else payload_digest,
    }
    row["authority_digest"] = eval_module.object_sha256(row)
    return eval_module.validate_file_authority(
        row, expected_role=role, reopen=False,
    )


def materialize(
    root: Path, source_root: Path, job_id: str, node: str, *,
    snapshot_manifest_sha256: str, materializer_sha256: str,
) -> dict[str, Any]:
    require_final_pins()
    if root != TARGET_ROOT or source_root != SOURCE_SNAPSHOT_ROOT:
        raise HoldPackageError("package/source root is not the sealed configured path")
    require_fresh_package_paths(root, RANK_CACHE_ROOT)
    if os.path.lexists(PACKAGE_PUBLICATION_RECEIPT_PATH):
        raise HoldPackageError("package publication receipt is not fresh")
    require_real_directory_chain(root.parent, label="package target")
    # The manifest, all 33 declared leaves (including this materializer), and
    # the physical exact35 tree are consumed before the first package mkdir.
    snapshot_bytes, snapshot_evidence = preflight_snapshot(
        source_root, manifest_sha256=snapshot_manifest_sha256,
        materializer_sha256=materializer_sha256,
    )
    if replay_staging_receipt_authority(
        snapshot_bytes[STAGING_RECEIPT_COPY_RELATIVE],
        snapshot_evidence["staging_rows"],
    ) != snapshot_evidence["staging_receipt_authority"]:
        raise HoldPackageError("pre-package staging receipt replay differs")
    release_bytes = {
        relative: snapshot_bytes[relative]
        for relative, pin in RELEASE_FILES.items()
    }
    diagnostic_bytes = {
        relative: snapshot_bytes[relative]
        for relative, pin in DIAGNOSTIC_FILES.items()
    }
    authority_bytes = {
        relative: snapshot_bytes[relative]
        for relative, pin in SNAPSHOT_AUTHORITY_FILES.items()
    }
    source_rel, source_pin, source_size = SOURCE_VIDEO
    aux_rel, aux_pin, aux_size = AUX_VIDEO
    condition_bytes = {
        "exact_original_source": stable(INPUT_ROOT / source_rel, source_pin, source_size),
        "aux_bone_removed_source": stable(INPUT_ROOT / aux_rel, aux_pin, aux_size),
    }
    launcher_relative = (
        "methods/bernini_action_editing/"
        "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py"
    )
    launcher_source = load_module_bytes(
        source_root / launcher_relative, release_bytes[launcher_relative],
        "_case01_hold_launcher_preflight",
    )
    eval_relative = (
        "methods/bernini_action_editing/"
        "case01_object_trajectory_exact5_eval_v1.py"
    )
    eval_module = load_module_bytes(
        source_root / eval_relative, release_bytes[eval_relative],
        "_case01_hold_eval_preflight",
    )
    runtime_preflight = {
        "python": identity_row(VACE_PYTHON),
        "ffmpeg": identity_row(FFMPEG),
        "ffprobe": identity_row(FFPROBE),
        "torchrun_source": identity_row(
            TORCH_ROOT / "distributed/run.py",
            launcher_source.EXPECTED_STATIC_SHA256["torchrun_source"],
        ),
        "torchrun_handler_source": identity_row(
            TORCH_ROOT
            / "distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py",
            launcher_source.EXPECTED_STATIC_SHA256["torchrun_handler_source"],
        ),
        "torch_local_agent_source": identity_row(
            TORCH_ROOT / "distributed/elastic/agent/server/local_elastic_agent.py",
            launcher_source.EXPECTED_STATIC_SHA256["torch_local_agent_source"],
        ),
        "torch_dynamic_rendezvous_source": identity_row(
            TORCH_ROOT / "distributed/elastic/rendezvous/dynamic_rendezvous.py",
            launcher_source.EXPECTED_STATIC_SHA256["torch_dynamic_rendezvous_source"],
        ),
        "torch_multiprocessing_api_source": identity_row(
            TORCH_ROOT / "distributed/elastic/multiprocessing/api.py",
            launcher_source.EXPECTED_STATIC_SHA256["torch_multiprocessing_api_source"],
        ),
        "r64_checkpoint_manifest": identity_row(
            CHECKPOINT_MANIFEST,
            launcher_source.EXPECTED_STATIC_SHA256["r64_checkpoint_manifest"],
        ),
    }
    for directory, label in (
        (MODEL_ROOT, "model root"), (BERNINI_ROOT, "Bernini root"),
        (VEOMNI_ROOT, "VeOmni root"),
    ):
        real_directory(directory, label=label)
    publish_root = root
    parent_fd = open_held_parent(publish_root)
    for path, label in (
        (publish_root, "package target"),
        (PACKAGE_PUBLICATION_RECEIPT_PATH, "package publication receipt"),
    ):
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            os.close(parent_fd)
            raise HoldPackageError(f"{label} appeared before reservation")
    reservation = create_publication_reservation(
        parent_fd, receipt_path=PACKAGE_PUBLICATION_RECEIPT_PATH,
        target_root=publish_root,
    )
    shadow_name = (
        f".{publish_root.name}.shadow.{os.getpid()}.{secrets.token_hex(12)}"
    )
    os.mkdir(shadow_name, mode=0o700, dir_fd=parent_fd)
    root = publish_root.parent / shadow_name
    for relative in ("release", "authority/conditions", "plan", "launch",
                     "diagnostics", "evidence", "outputs/media", "final",
                     "logs", "runtime"):
        components = Path(relative).parts
        current = root
        for component in components:
            current = current / component
            if not os.path.lexists(current):
                current.mkdir(mode=0o700)
            elif not stat.S_ISDIR(os.lstat(current).st_mode):
                raise HoldPackageError(f"package directory differs: {current}")
    for relative in sorted(expected_directories(set(RELEASE_FILES)) - {"."}):
        directory = root / "release" / relative
        if not os.path.lexists(directory):
            directory.mkdir(mode=0o700)
    artifacts: dict[str, Any] = {}
    expected_shadow: dict[str, tuple[str, int, int]] = {}

    def record(relative: str, raw: bytes, mode: int) -> Path:
        target = root / relative
        create(target, raw, mode)
        expected_shadow[relative] = (
            hashlib.sha256(raw).hexdigest(), len(raw), mode,
        )
        return target

    for relative, pin in RELEASE_FILES.items():
        raw = release_bytes[relative]
        target = record("release/" + relative, raw, 0o444)
        artifacts[str(target.relative_to(root))] = {"sha256": pin, "size": len(raw)}
    for relative, pin in DIAGNOSTIC_FILES.items():
        raw = diagnostic_bytes[relative]
        target = record("diagnostics/" + Path(relative).name, raw, 0o444)
        artifacts[str(target.relative_to(root))] = {"sha256": pin, "size": len(raw)}
    authority_targets: dict[str, Path] = {}
    authority_raw: dict[str, bytes] = {}
    for role, source, pin, size in (
        ("exact_original_source", INPUT_ROOT / source_rel, source_pin, source_size),
        ("aux_bone_removed_source", INPUT_ROOT / aux_rel, aux_pin, aux_size),
    ):
        raw = condition_bytes[role]
        name = Path(source_rel if role.startswith("exact") else aux_rel).name
        record("authority/conditions/" + name, raw, 0o444)
        authority_targets[role] = publish_root / "authority/conditions" / name
        authority_raw[role] = raw
    staged_authority_names = {
        "stage0_masks": "stage0_receipt.json",
        "g0_mouth_track": "g0_sparse_annotations.json",
        "trajectory_scaffold": "trajectory_scaffold.json",
        "scaffold_independent_audit": "scaffold_independent_audit.json",
    }
    staged_roles = tuple(staged_authority_names)
    if len(staged_roles) != len(SNAPSHOT_AUTHORITY_FILES):
        raise HoldPackageError("staged authority role closure differs")
    for (relative, pin), role in zip(SNAPSHOT_AUTHORITY_FILES.items(), staged_roles):
        raw = authority_bytes[relative]
        name = staged_authority_names[role]
        record("authority/conditions/" + name, raw, 0o444)
        authority_targets[role] = publish_root / "authority/conditions" / name
        authority_raw[role] = raw
    method = publish_root / "release/methods/bernini_action_editing"
    source_authority = _file_authority_from_bytes(
        eval_module, authority_targets["exact_original_source"],
        authority_raw["exact_original_source"], role="exact_original_source",
    )
    conditions = {
        "stage0_masks": _file_authority_from_bytes(
            eval_module, authority_targets["stage0_masks"],
            authority_raw["stage0_masks"], role="stage0_masks",
            payload_digest=eval_module.EXPECTED_STAGE0_RECEIPT_DIGEST),
        "g0_mouth_track": _file_authority_from_bytes(
            eval_module, authority_targets["g0_mouth_track"],
            authority_raw["g0_mouth_track"], role="g0_mouth_track"),
        "trajectory_scaffold": _file_authority_from_bytes(
            eval_module, authority_targets["trajectory_scaffold"],
            authority_raw["trajectory_scaffold"], role="trajectory_scaffold",
            payload_digest=eval_module.EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST),
        "aux_bone_removed_source": _file_authority_from_bytes(
            eval_module, authority_targets["aux_bone_removed_source"],
            authority_raw["aux_bone_removed_source"], role="aux_bone_removed_source"),
    }
    admissions = {"scaffold_independent_audit": _file_authority_from_bytes(
        eval_module, authority_targets["scaffold_independent_audit"],
        authority_raw["scaffold_independent_audit"],
        role="scaffold_independent_audit",
        payload_digest=eval_module.EXPECTED_SCAFFOLD_AUDIT_DIGEST)}
    producer = {
        "inference_receipt_schemas": {
            "off": eval_module.LEGACY_INFERENCE_RECEIPT_SCHEMA,
            "route_or_active": eval_module.INFERENCE_RECEIPT_SCHEMA,
        },
        "infer_lora_path": str(method / "infer_lora_full644_r5_frozen_acc46.py"),
        "infer_lora_sha256": "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
        "infer_lora_size": 177_300,
        "infer_lora_role": "frozen_legacy_exact5_infer_lora_not_workspace_head",
        "inference_wrapper_path": str(method / "infer_case01_object_trajectory_oracle_v1.py"),
        "inference_wrapper_sha256": FINAL_WRAPPER_SHA256,
        "inference_wrapper_size": 74_281,
        "trajectory_projection_module_path": str(method / "object_trajectory_projection_v1.py"),
        "trajectory_projection_module_sha256": "a851afd220d770e6c8082ff8e7f0a0c2b5a5be894bcbf9feeaf8bca4cc6c9e7e",
        "trajectory_projection_module_size": 47_588,
        "trajectory_scaffold_module_path": str(method / "case01_oracle_object_trajectory_v1.py"),
        "trajectory_scaffold_module_sha256": "a7d4e008e78d373370b832c0754e5d7420b504fa8b2022eef9a9bb899added8a",
        "trajectory_scaffold_module_size": 35_803,
        "ffprobe_path": str(FFPROBE),
        "ffprobe_sha256": runtime_preflight["ffprobe"]["sha256"],
        "ffprobe_size": runtime_preflight["ffprobe"]["size"],
        "method_source_revision": "ce4cffc1e8a144448c92252d9fb63087f03bbd8c",
        "method_source_archive_sha256": "12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828",
        "pins_complete": True,
    }
    plan = eval_module.build_plan(
        source_authority=source_authority, condition_authorities=conditions,
        admission_authorities=admissions, checkpoint_manifest=CHECKPOINT,
        producer=producer, output_root=root / "outputs/media", launch_allowed=False)
    # build_plan has already performed its full schema validation against the
    # fresh shadow.  Replace only the five publication paths with their final
    # atomic-publish names, then recompute and explicitly close that transform.
    plan.pop("plan_digest")
    for task, task_id in zip(plan["tasks"], eval_module.TASK_IDS):
        video = publish_root / "outputs/media" / f"{task_id}.mp4"
        task["output"] = {
            "video_path": str(video),
            "receipt_path": str(video.with_name(video.name + ".receipt.json")),
            "create_only": True,
        }
    plan["plan_digest"] = eval_module.object_sha256(plan)
    if (
        [task.get("task_id") for task in plan["tasks"]]
        != list(eval_module.TASK_IDS)
        or any(
            not task["output"]["video_path"].startswith(
                str(publish_root / "outputs/media") + os.sep
            )
            for task in plan["tasks"]
        )
    ):
        raise HoldPackageError("logical publication rewrite differs")
    plan_raw = canonical(plan) + b"\n"
    plan_relative = "plan/case01_object_trajectory_exact5_r64_HOLD_plan_v1.json"
    record(plan_relative, plan_raw, 0o444)
    plan_path = publish_root / plan_relative
    launcher = launcher_source
    launch_input = {
        "schema_version": launcher.INPUT_SCHEMA, "entry_mode": "trusted_stdin",
        "campaign_mode": CAMPAIGN, "holder_job_id": job_id,
        "expected_node": node, "expected_allocation_gpu_count": 8,
        "identities": _runtime_identities_from_preflight(
            publish_root, plan_path, plan_raw, launcher, release_bytes,
            runtime_preflight,
        ),
        "output_report": str(publish_root / "final/object_trajectory_exact5_report_v1.json"),
        "runner_attestation": str(publish_root / "final/object_trajectory_exact5_runner_attestation_v1.json"),
        "model_root": str(MODEL_ROOT), "bernini_root": str(BERNINI_ROOT),
        "veomni_root": str(VEOMNI_ROOT),
        "authority_root": str(publish_root / "runtime/model-authority"),
        "rank_cache_root": str(RANK_CACHE_ROOT),
    }
    launcher.validate_input(
        launch_input, reopen=False, plan_override=plan,
    )
    input_relative = "launch/root_launch_input_HOLD_v1.json"
    input_raw = canonical(launch_input) + b"\n"
    input_path = record(input_relative, input_raw, 0o444)
    payload_relative = "launch/root_launch_payload_HOLD_v1.sh"
    receipt_relative = "launch/root_launch_receipt_HOLD_v1.json"
    launch_receipt = launcher.materialize(
        input_path, root / payload_relative, root / receipt_relative,
        reopen_identities=False, plan_override=plan,
        logical_input_path=publish_root / input_relative,
        logical_payload_path=publish_root / payload_relative,
    )
    payload_raw = launcher._hold_payload(launch_receipt["release"])
    receipt_raw = launcher.canonical_json_bytes(launch_receipt) + b"\n"
    expected_shadow[payload_relative] = (
        hashlib.sha256(payload_raw).hexdigest(), len(payload_raw), 0o444,
    )
    expected_shadow[receipt_relative] = (
        hashlib.sha256(receipt_raw).hexdigest(), len(receipt_raw), 0o400,
    )
    report: dict[str, Any] = {
        "schema_version": "case01-object-trajectory-exact5-r64-hold-materialization-v1",
        "status": "MATERIALIZED_HOLD_NOT_SUBMITTED", "launch_allowed": False,
        "root": str(publish_root), "source_snapshot_root": str(source_root),
        "source_snapshot": snapshot_evidence,
        "source_staging_receipt_authority": snapshot_evidence[
            "staging_receipt_authority"
        ],
        "package_publication_receipt_path": str(
            PACKAGE_PUBLICATION_RECEIPT_PATH
        ),
        "publication_protocol": STAGING_PUBLICATION_PROTOCOL,
        "rename_noreplace": False,
        "cooperative_writer_exclusion": True,
        "uncooperative_same_uid_race_out_of_scope": True,
        "retry_allowed": False,
        "release_file_count": 25, "production_identity_count": 25,
        "condition_and_admission_authority_count": 6,
        "plan": {"path": str(plan_path), "sha256": hashlib.sha256(plan_raw).hexdigest(),
                 "plan_digest": plan["plan_digest"]},
        "launch": launch_receipt,
        "admission": {"static_executed": False, "root_fake_executed": False,
                      "world4_executed": False},
        "slurm_step_launched": False, "gpu_attempt_claimed": False,
        "artifacts": artifacts,
    }
    report["receipt_digest"] = object_sha(report)
    report_relative = "authority/package_materialization_receipt_v1.json"
    report_raw = canonical(report) + b"\n"
    record(report_relative, report_raw, 0o400)
    for relative in ("release", "authority/conditions", "plan", "launch"):
        os.chmod(root / relative, 0o555)
    fsync_shadow_directories(root)
    validate_shadow_package(
        root, expected_shadow,
        extra_directories={
            "evidence", "outputs", "outputs/media", "final", "logs",
            "runtime",
        },
    )
    # Recheck all upstream consumption authority immediately before the one
    # ordinary same-parent POSIX rename.  The O_EXCL sibling receipt excludes
    # conforming writers; an uncooperative same-UID writer is out of scope.
    if replay_staging_receipt_authority(
        snapshot_bytes[STAGING_RECEIPT_COPY_RELATIVE],
        snapshot_evidence["staging_rows"],
    ) != snapshot_evidence["staging_receipt_authority"]:
        raise HoldPackageError("pre-publish staging receipt replay differs")
    if _identity(os.fstat(parent_fd)) != _identity(os.lstat(publish_root.parent)):
        raise HoldPackageError("held package parent changed before publish")
    commit_error: PublicationCommittedError | None = None
    try:
        target_identity, publication_observation = publish_under_reservation(
            parent_fd, shadow_name, publish_root.name, reservation,
        )
    except PublicationCommittedError as error:
        commit_error = error
        target_identity = error.identity
        publication_observation = error.observation
    publication_value = {
        "schema_version": PACKAGE_PUBLICATION_SCHEMA + "-receipt",
        "status": (
            "PUBLISHED_RECEIPT_GATED" if commit_error is None
            else "PUBLISHED_COMMIT_ERROR_NOT_ADMISSION"
        ),
        "target_root": str(publish_root),
        "receipt_path": str(PACKAGE_PUBLICATION_RECEIPT_PATH),
        "materialization_receipt_path": str(
            publish_root / report_relative
        ),
        "materialization_receipt_sha256": hashlib.sha256(report_raw).hexdigest(),
        "materialization_receipt_digest": report["receipt_digest"],
        "source_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "source_snapshot_manifest_digest": snapshot_evidence["manifest_digest"],
        "source_staging_receipt_sha256": snapshot_evidence[
            "staging_receipt_authority"
        ]["sha256"],
        "source_staging_receipt_digest": snapshot_evidence[
            "staging_receipt_authority"
        ]["receipt_digest"],
        "publication_protocol": STAGING_PUBLICATION_PROTOCOL,
        "rename_noreplace": False,
        "cooperative_writer_exclusion": True,
        "target_absent_rechecked_before_rename": True,
        "ordinary_posix_rename_performed": True,
        "publication_observation": publication_observation,
        "whole_tree_atomically_visible": True,
        "uncooperative_same_uid_race_out_of_scope": True,
        "retry_allowed": False,
        "target_root_identity": list(target_identity),
        "receipt_mode": PUBLICATION_RECEIPT_MODE,
        "receipt_is_consumption_gate": commit_error is None,
        "receipt_is_admission": commit_error is None,
        "launch_allowed": False,
    }
    seal_publication_receipt(parent_fd, reservation, publication_value)
    reservation.close()
    os.close(parent_fd)
    if commit_error is not None:
        raise HoldPackageError(
            "package committed but publication syscall/fsync terminal failed"
        ) from commit_error
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(TARGET_ROOT))
    parser.add_argument("--source-root", default=str(SOURCE_SNAPSHOT_ROOT))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--materializer-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = materialize(
            Path(args.root), Path(args.source_root), args.job_id, args.node,
            snapshot_manifest_sha256=args.snapshot_manifest_sha256,
            materializer_sha256=args.materializer_sha256,
        )
    except (OSError, ValueError, HoldPackageError) as error:
        print(str(error), file=sys.stderr)
        return 88 if str(error).startswith("HOLD:") else 96
    print(json.dumps(report, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
