#!/usr/bin/env python3
"""Build the sealed source snapshot for the case01 exact-five R64 canary.

Historical production bytes are read exclusively from the already sealed r5f
snapshot.  Only the new exact5 evaluator, wrapper, launcher, package
materializer, independent audit receipt, and two CPU-admission diagnostic
sources may come from the fresh staging root.  The destination is create-only
and is never used to launch Slurm.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


EXPERIMENTS = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments"
)
OLD_R5F_SNAPSHOT = (
    EXPERIMENTS
    / "bernini_full644_exploratory_matched_r5f_source_snapshot_21_20260820_r1"
)
STAGING_ROOT = (
    EXPERIMENTS
    / "bernini_object_grounded_case01_0821_exact5_source_staging_v1"
)
TARGET_ROOT = (
    EXPERIMENTS
    / "bernini_object_grounded_case01_0821_exact5_source_snapshot_24_20260821_r1"
)
SNAPSHOT_MANIFEST = "case01_exact5_source_snapshot_manifest_v1.json"
BUILDER_RELATIVE = (
    "methods/bernini_action_editing/tools/"
    "build_case01_source_bone_exact5_source_snapshot_v1.py"
)
AUDIT_RELATIVE = (
    "md/action_editing/20260821_man/evidence/"
    "case01_exact5_intervention_asset_independent_audit_v1.json"
)

# Deliberately excludes the old r5f root launcher.  Every row is copied from
# OLD_R5F_SNAPSHOT and cannot fall back to a working tree or staging root.
OLD_REUSED_FILES = {
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
}
NEW_STAGED_FILES = {
    "methods/bernini_action_editing/case01_source_bone_exact5_runner_v1.py":
        "4a46e870b61ff42345c523cca97b8170c065033bc2af9a0353be9df4373ec3ea",
    "methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py":
        "891551d46b0ca11362fe8d446f202036b9957baa943de0aec6da1f3ad45d7d58",
    "methods/bernini_action_editing/case01_source_bone_exact5_spooled_launcher_auh_v1.py":
        "00b68ca8221dd343cb9ca8393c9205cccf6a61d474c56e56c9b081570418d390",
    "methods/bernini_action_editing/tools/"
    "materialize_case01_source_bone_exact5_r64_package_v1.py":
        "937023f00ab5aa86ce0a5c6274c0901be35e4bd852152b55928abbf26c6102b8",
    AUDIT_RELATIVE:
        "040c53a3647ae957212a1d2d6da3ffa75b4207ace07e1c7ba6ce128033dce969",
    "methods/bernini_action_editing/case01_source_bone_exact5_static_probe_v1.py":
        "3eabfcd6fedc264018c18aec6c518a77aa4e093dcb9d2b65f371244e6ac57f02",
    "methods/bernini_action_editing/case01_source_bone_exact5_root_fake_runner_v1.py":
        "414be72dc7b428b6b34ad038c4315cb0b336f28db74f4ee2273a8d14cd8218a1",
}


class Exact5SnapshotError(RuntimeError):
    """The exact5 source snapshot contract differs."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def ident(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_rdev, info.st_size,
        getattr(info, "st_blocks", 0), info.st_mtime_ns, info.st_ctime_ns,
    )


def stable_file(path: Path, digest: str | None, *, sealed: bool) -> tuple[bytes, int]:
    if (
        not path.is_absolute() or os.path.normpath(str(path)) != str(path)
        or path.is_symlink() or path.resolve(strict=True) != path
    ):
        raise Exact5SnapshotError(f"source path differs: {path}")
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
        middle = os.fstat(descriptor)
        raw_again = b"".join(
            os.pread(descriptor, min(1_048_576, before.st_size - offset), offset)
            for offset in range(0, before.st_size, 1_048_576)
        )
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    mode = stat.S_IMODE(before.st_mode)
    if (
        not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not raw
        or len(raw) != before.st_size or raw_again != raw
        or ident(before) != ident(middle) or ident(before) != ident(after)
        or ident(before) != ident(named) or before.st_uid != 2012
        or before.st_gid != 2000
        or (sealed and mode != 0o444)
        or (not sealed and mode & 0o022 != 0)
        or (digest is not None and hashlib.sha256(raw).hexdigest() != digest)
    ):
        raise Exact5SnapshotError(f"source authority differs: {path}")
    return raw, mode


def create_file(path: Path, raw: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(
        path, os.O_RDWR | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise Exact5SnapshotError("snapshot write made no progress")
            offset += count
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(staged.st_mode) or stat.S_IMODE(staged.st_mode) != 0
            or staged.st_nlink != 1 or ident(staged) != ident(named)
            or os.pread(descriptor, len(raw), 0) != raw
        ):
            raise Exact5SnapshotError("snapshot staging replay differs")
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def open_exact_tree(
    root: Path,
    expected_file_modes: Mapping[str, int],
    expected_directory_modes: Mapping[str, int],
) -> dict[str, int]:
    """Open and validate an exact regular-file/directory tree without following links."""
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
                not stat.S_ISDIR(before.st_mode) or ident(before) != ident(named)
                or before.st_uid != 2012 or before.st_gid != 2000
                or stat.S_IMODE(before.st_mode)
                != expected_directory_modes.get(relative)
            ):
                raise Exact5SnapshotError(
                    f"snapshot directory authority differs: {relative}"
                )
            for name in os.listdir(directory_fd):
                if not name or name in {".", ".."} or "/" in name:
                    raise Exact5SnapshotError("snapshot entry name differs")
                child_relative = name if relative == "." else f"{relative}/{name}"
                child = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISREG(child.st_mode):
                    if (
                        child.st_uid != 2012 or child.st_gid != 2000
                        or child.st_nlink != 1
                        or stat.S_IMODE(child.st_mode)
                        != expected_file_modes.get(child_relative)
                    ):
                        raise Exact5SnapshotError(
                            f"snapshot file authority differs: {child_relative}"
                        )
                    files.add(child_relative)
                elif stat.S_ISDIR(child.st_mode):
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                    held = os.fstat(child_fd)
                    if ident(held) != ident(child):
                        os.close(child_fd)
                        raise Exact5SnapshotError(
                            f"snapshot directory identity differs: {child_relative}"
                        )
                    opened[child_relative] = child_fd
                    pending.append((child_relative, child_fd))
                else:
                    raise Exact5SnapshotError(
                        f"snapshot contains a link or special entry: {child_relative}"
                    )
        if files != set(expected_file_modes) or set(opened) != set(expected_directory_modes):
            raise Exact5SnapshotError("snapshot physical tree closure differs")
        return opened
    except BaseException:
        for descriptor in opened.values():
            os.close(descriptor)
        raise


def close_directories(opened: Mapping[str, int]) -> None:
    for descriptor in opened.values():
        os.close(descriptor)


def _exact_staging_closure() -> None:
    expected = set(NEW_STAGED_FILES) | {BUILDER_RELATIVE}
    actual: set[str] = set()
    actual_directories = {"."}
    for path in STAGING_ROOT.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise Exact5SnapshotError("staging contains a link or special entry")
        if path.is_file():
            actual.add(str(path.relative_to(STAGING_ROOT)))
        elif path.is_dir():
            actual_directories.add(str(path.relative_to(STAGING_ROOT)))
    if actual != expected:
        raise Exact5SnapshotError("staging physical file closure differs")
    expected_directories = {"."}
    for relative in expected:
        parent = Path(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent
    if actual_directories != expected_directories:
        raise Exact5SnapshotError("staging directory closure differs")
    for relative in expected_directories:
        path = STAGING_ROOT if relative == "." else STAGING_ROOT / relative
        info = path.lstat()
        if (
            not stat.S_ISDIR(info.st_mode) or path.is_symlink()
            or info.st_uid != 2012 or info.st_gid != 2000
            or stat.S_IMODE(info.st_mode) & 0o002
        ):
            raise Exact5SnapshotError("staging directory authority differs")


def build_snapshot() -> dict[str, Any]:
    if os.geteuid() != 2012 or os.getegid() != 2000:
        raise Exact5SnapshotError("snapshot builder owner authority differs")
    for path, label in (
        (OLD_R5F_SNAPSHOT, "old sealed snapshot"),
        (STAGING_ROOT, "new staging root"),
    ):
        if (
            not path.is_absolute() or path.is_symlink() or not path.is_dir()
            or path.resolve(strict=True) != path
        ):
            raise Exact5SnapshotError(f"{label} differs")
    if TARGET_ROOT.exists() or TARGET_ROOT.is_symlink():
        raise Exact5SnapshotError("fresh exact5 snapshot target exists")
    if set(OLD_REUSED_FILES) & set(NEW_STAGED_FILES):
        raise Exact5SnapshotError("old/new source roles overlap")
    if (
        len(OLD_REUSED_FILES) != 16 or len(NEW_STAGED_FILES) != 7
        or "methods/bernini_action_editing/infer_lora.py" not in OLD_REUSED_FILES
        or "methods/bernini_action_editing/infer_lora.py" in NEW_STAGED_FILES
        or any("spooled_launcher_auh_r5f.py" in path for path in OLD_REUSED_FILES)
    ):
        raise Exact5SnapshotError("source provenance partition differs")
    _exact_staging_closure()
    # Bind the named builder too; the trusted controller separately binds it to
    # the captured bytes used to enter this program.
    stable_file(STAGING_ROOT / BUILDER_RELATIVE, None, sealed=False)

    sources: dict[str, tuple[bytes, str]] = {}
    for relative, digest in OLD_REUSED_FILES.items():
        raw, _ = stable_file(OLD_R5F_SNAPSHOT / relative, digest, sealed=True)
        sources[relative] = (raw, "sealed_r5f_snapshot")
    for relative, digest in NEW_STAGED_FILES.items():
        raw, _ = stable_file(STAGING_ROOT / relative, digest, sealed=False)
        sources[relative] = (raw, "new_exact5_staging")

    os.mkdir(TARGET_ROOT, 0o700)
    target_directories: set[str] = set()
    for relative in sources:
        parent = Path(relative).parent
        while str(parent) != ".":
            target_directories.add(str(parent))
            parent = parent.parent
    for relative in sorted(
        target_directories,
        key=lambda item: (len(Path(item).parts), item),
    ):
        # Every directory is expected to be absent.  os.mkdir fails closed if
        # a concurrent path (including a symlink) appears.
        os.mkdir(TARGET_ROOT / relative, 0o700)
    rows: list[dict[str, Any]] = []
    for relative in sorted(sources):
        raw, provenance = sources[relative]
        target = TARGET_ROOT / relative
        create_file(target, raw)
        rows.append({
            "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw), "mode": 0o444, "provenance": provenance,
        })
    manifest: dict[str, Any] = {
        "schema_version": "case01-source-bone-exact5-source-snapshot-v1",
        "status": "SEALED_NOT_EXECUTED",
        "target_root": str(TARGET_ROOT),
        "old_r5f_snapshot_root": str(OLD_R5F_SNAPSHOT),
        "new_staging_root": str(STAGING_ROOT),
        "file_count": len(rows), "old_reused_file_count": 16,
        "new_staged_file_count": 7,
        "physical_file_count_including_manifest": len(rows) + 1,
        "release_file_count": 19,
        "sealed_r5f_infer_lora_reused": True,
        "working_tree_infer_lora_read": False,
        "slurm_step_launched": False,
        "files": rows,
    }
    manifest["snapshot_digest"] = object_sha256(manifest)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    create_file(TARGET_ROOT / SNAPSHOT_MANIFEST, manifest_raw)

    expected_files = set(sources) | {SNAPSHOT_MANIFEST}
    expected_directories = {"."}
    for relative in expected_files:
        parent = Path(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent
    file_modes = {relative: 0o444 for relative in expected_files}
    staged_directory_modes = {relative: 0o700 for relative in expected_directories}
    opened = open_exact_tree(TARGET_ROOT, file_modes, staged_directory_modes)
    try:
        # Seal only the already-opened real directory objects.  A concurrent
        # named-path swap therefore cannot redirect chmod through a symlink.
        for relative in sorted(
            opened, key=lambda item: (len(Path(item).parts), item), reverse=True,
        ):
            descriptor = opened[relative]
            path = TARGET_ROOT if relative == "." else TARGET_ROOT / relative
            before = os.fstat(descriptor)
            named = os.lstat(path)
            if ident(before) != ident(named) or not stat.S_ISDIR(before.st_mode):
                raise Exact5SnapshotError("snapshot directory changed before seal")
            os.fchmod(descriptor, 0o555)
            after = os.fstat(descriptor)
            named_after = os.lstat(path)
            if (
                ident(after) != ident(named_after)
                or stat.S_IMODE(after.st_mode) != 0o555
            ):
                raise Exact5SnapshotError("snapshot directory seal replay differs")
    finally:
        close_directories(opened)
    sealed_directory_modes = {relative: 0o555 for relative in expected_directories}
    replayed = open_exact_tree(TARGET_ROOT, file_modes, sealed_directory_modes)
    close_directories(replayed)
    for relative, (raw, _) in sources.items():
        replay, _ = stable_file(
            TARGET_ROOT / relative, hashlib.sha256(raw).hexdigest(), sealed=True,
        )
        if replay != raw:
            raise Exact5SnapshotError("sealed snapshot replay differs")
    stable_file(
        TARGET_ROOT / SNAPSHOT_MANIFEST,
        hashlib.sha256(manifest_raw).hexdigest(), sealed=True,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    if argv not in (None, (), []):
        raise Exact5SnapshotError("snapshot builder accepts no arguments")
    if (
        sys.platform != "linux" or not Path("/proc/self/fd").is_dir()
        or sys.flags.isolated != 1 or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1 or not sys.dont_write_bytecode
        or "sitecustomize" in sys.modules or "usercustomize" in sys.modules
    ):
        raise Exact5SnapshotError("isolated snapshot-builder startup differs")
    manifest = build_snapshot()
    print(canonical_json_bytes(manifest).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
