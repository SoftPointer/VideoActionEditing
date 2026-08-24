#!/usr/bin/env python3
"""Build the receipt-gated exact35 sealed source snapshot for the trajectory exact-five.

Historical bytes are accepted only from the sealed legacy exact5 snapshot.
Fresh sources and four small authorities are accepted only from the staging
root after its final AUHv2 receipt is consumed.  That 0400 receipt is copied
as a snapshot leaf.  The legacy infer alias is a separate inode, never a link.
"""

from __future__ import annotations

import argparse
import hashlib
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
OLD_EXACT5_SNAPSHOT = (
    EXPERIMENTS / "bernini_object_grounded_case01_0821_exact5_source_snapshot_24_20260821_r1"
)
STAGING_ROOT = EXPERIMENTS / "bernini_case01_object_trajectory_exact5_source_staging_v1"
STAGING_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_staging_v1.receipt_v1.json"
)
TARGET_ROOT = EXPERIMENTS / "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1"
SNAPSHOT_PUBLICATION_RECEIPT_PATH = EXPERIMENTS / (
    "bernini_case01_object_trajectory_exact5_source_snapshot_35_20260822_r1."
    "receipt_v2.json"
)
MANIFEST_NAME = "case01_object_trajectory_exact5_source_snapshot_manifest_v2.json"
OLD_MANIFEST_NAME = "case01_exact5_source_snapshot_manifest_v1.json"
BUILDER_RELATIVE = (
    "methods/bernini_action_editing/tools/"
    "build_case01_object_trajectory_exact5_source_snapshot_v1.py"
)
OLD_R5F_SNAPSHOT = (
    EXPERIMENTS / "bernini_full644_exploratory_matched_r5f_source_snapshot_21_20260820_r1"
)
OLD_EXACT5_STAGING = (
    EXPERIMENTS / "bernini_object_grounded_case01_0821_exact5_source_staging_v1"
)
FORMAL_REVIEW_TEST = {
    "path": "methods/bernini_action_editing/tests/test_case01_object_trajectory_exact5_core_v1.py",
    "sha256": "a22895e47766c506fcb8265035ab9d7a91cae9a940c6e3ab5371a21673a2a8b4",
    "size": 50_880,
    "sealed_bytes_in_snapshot": False,
    "role": "formal_stop_review_evidence_not_runtime_or_release_authority",
}
FINAL_RUNNER_SHA256 = "e47b81643c1d17e5099a9b33f16ca75521001ad52d2df2305b46b7e8c4d5ac4c"
FINAL_EVAL_SHA256 = "47cc871b82b8cf7762db9183997440eeabd287b1c702d9cd7421fd43e0a555e0"
FINAL_WRAPPER_SHA256 = "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9"

REUSED_FILES = {
    "methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json": "953933f1161b6d62826d388ba5ed42e42792fbf5f2bdeea199c1eb13cd251b4a",
    "methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl": "c05c4e5b5bf85de882bde32c71a984d736247733e586ed91d40026b12aaaf701",
    "methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py": "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
    "methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256": "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
    "methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py": "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d",
    "methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py": "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py": "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca",
    "methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5f.py": "52dcde8797484504ab28a5c59c532c0877a145ea24762edaf3105a21b0719e19",
    "methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py": "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223",
    "methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5.py": "cb201398940d59393fa58471dc2c3f9fdf001c7e881ec891ce892bb460cf01ba",
    "methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py": "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136",
    "methods/bernini_action_editing/infer_lora.py": "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
    "methods/bernini_action_editing/self_generated_action_preservation_v2.py": "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c",
    "methods/bernini_action_editing/tools/build_renderer_dataset.py": "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    "methods/bernini_action_editing/tools/materialize_vae.py": "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    "methods/bernini_action_editing/train_lora.py": "ead547b8309e1b5ae5c831444e9f5d1d8e1785fed5fe39cf7b97f13f82a9ce85",
    "methods/bernini_action_editing/case01_source_bone_exact5_runner_v1.py": "4a46e870b61ff42345c523cca97b8170c065033bc2af9a0353be9df4373ec3ea",
    "methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py": "891551d46b0ca11362fe8d446f202036b9957baa943de0aec6da1f3ad45d7d58",
}

STAGED_FILES = {
    "methods/bernini_action_editing/case01_object_trajectory_exact5_runner_v1.py": FINAL_RUNNER_SHA256,
    "methods/bernini_action_editing/case01_object_trajectory_exact5_eval_v1.py": FINAL_EVAL_SHA256,
    "methods/bernini_action_editing/case01_object_trajectory_exact5_spooled_launcher_auh_v1.py": "a81e812627125a24d72ec956b384e30479df379b66b5a94da871021c3e14267f",
    "methods/bernini_action_editing/infer_case01_object_trajectory_oracle_v1.py": FINAL_WRAPPER_SHA256,
    "methods/bernini_action_editing/object_trajectory_projection_v1.py": "a851afd220d770e6c8082ff8e7f0a0c2b5a5be894bcbf9feeaf8bca4cc6c9e7e",
    "methods/bernini_action_editing/case01_oracle_object_trajectory_v1.py": "a7d4e008e78d373370b832c0754e5d7420b504fa8b2022eef9a9bb899added8a",
    "methods/bernini_action_editing/tools/materialize_case01_object_trajectory_exact5_r64_hold_package_v1.py": "31c0184c8187fe0224c92bcb425dd0ec27731e7197898bd552aef82f83fa49f9",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_static_probe_v1.py": "071256da47635fc3481f51b48e7e5eddddc963a5345b1dda405473744d2c01a9",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_root_fake_runner_v1.py": "0d73fdaa4a4f1817f572eea471661850098ffe5aa54f54a9927c37a7e3f2a872",
    "methods/bernini_action_editing/case01_object_trajectory_exact5_world4_probe_v1.py": "71c52ea3d7b36f07fdf5f9af3c9ecadf2020123795d9a4e10888a67eb0c7536b",
    "artifacts/object_grounded_case01_0821_sam2_masklets_r2/receipt.json": "c9dc8567d4f147f943171d2d7abb55e52aadc685fbfc2f75fff4e837e2ab0b50",
    "methods/bernini_action_editing/assets/case01_288545b9c031491a_g0_sparse_annotations_v1.json": "e5185a1edd72fa8a1f2ece15e98c67d66e3fa65a2a9eb724bf06031c4d0e2020",
    "artifacts/case01_oracle_object_trajectory_v1/scaffold.json": "7b1bec6e9764a1297bb0029f8fea01ebe4b2deab0acc2c7f07fdee96bc0a098a",
    "md/action_editing/20260821_man/evidence/case01_object_trajectory_scaffold_independent_audit_v1.json": "acbe4a6e635e3429605a8aac4d655816fd6187ea7aec77d5a8b1e08a56a47e0e",
}
OLD_EXACT5_FILES = {
    **REUSED_FILES,
    "methods/bernini_action_editing/case01_source_bone_exact5_spooled_launcher_auh_v1.py":
        "00b68ca8221dd343cb9ca8393c9205cccf6a61d474c56e56c9b081570418d390",
    "methods/bernini_action_editing/tools/materialize_case01_source_bone_exact5_r64_package_v1.py":
        "937023f00ab5aa86ce0a5c6274c0901be35e4bd852152b55928abbf26c6102b8",
    "md/action_editing/20260821_man/evidence/case01_exact5_intervention_asset_independent_audit_v1.json":
        "040c53a3647ae957212a1d2d6da3ffa75b4207ace07e1c7ba6ce128033dce969",
    "methods/bernini_action_editing/case01_source_bone_exact5_static_probe_v1.py":
        "3eabfcd6fedc264018c18aec6c518a77aa4e093dcb9d2b65f371244e6ac57f02",
    "methods/bernini_action_editing/case01_source_bone_exact5_root_fake_runner_v1.py":
        "414be72dc7b428b6b34ad038c4315cb0b336f28db74f4ee2273a8d14cd8218a1",
}
OLD_EXACT5_FRESH = set(OLD_EXACT5_FILES) - set(REUSED_FILES) | {
    "methods/bernini_action_editing/case01_source_bone_exact5_runner_v1.py",
    "methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py",
}
ALIAS_RELATIVE = "methods/bernini_action_editing/infer_lora_full644_r5_frozen_acc46.py"
ALIAS_SOURCE = "methods/bernini_action_editing/infer_lora.py"
STAGING_RECEIPT_COPY_RELATIVE = (
    "authority/source_staging_receipt_auh_v2.json"
)
STAGING_SCHEMA = "case01-object-trajectory-exact5-source-stager-auh-v2"
STAGING_MANIFEST_SCHEMA = STAGING_SCHEMA + "-manifest"
STAGING_RECEIPT_SCHEMA = STAGING_SCHEMA + "-receipt"
STAGING_PUBLICATION_PROTOCOL = (
    "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation"
)
SNAPSHOT_PUBLICATION_SCHEMA = (
    "case01-object-trajectory-exact5-source-snapshot-publication-v2"
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
PUBLICATION_RESERVATION_MODE = 0o600
PUBLICATION_RECEIPT_MODE = 0o400
SHA_RE = re.compile(r"[0-9a-f]{64}")
if (
    len(REUSED_FILES) != 18 or len(STAGED_FILES) != 14
    or len(OLD_EXACT5_FILES) != 23 or len(OLD_EXACT5_FRESH) != 7
):
    raise RuntimeError("exact34 source decomposition differs")


class SnapshotError(RuntimeError):
    pass


class PublicationCommittedError(SnapshotError):
    """The target inode is committed but publication was not fully clean."""

    def __init__(self, identity: tuple[int, ...], observation: Mapping[str, Any]):
        super().__init__("snapshot target committed without clean publication terminal")
        self.identity = identity
        self.observation = dict(observation)


class PublicationReceiptTerminalError(SnapshotError):
    """A 0400 receipt commit is immutable but its terminal audit is unclear."""

    def __init__(self, observation: Mapping[str, Any]):
        super().__init__("immutable snapshot receipt commit requires manual HOLD audit")
        self.observation = dict(observation)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def blocked_sources() -> tuple[str, ...]:
    return tuple(path for path, pin in STAGED_FILES.items() if SHA_RE.fullmatch(pin) is None)


def ident(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_rdev, info.st_size,
        getattr(info, "st_blocks", 0), info.st_mtime_ns, info.st_ctime_ns,
    )


def strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise SnapshotError(f"duplicate JSON key in {label}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise SnapshotError(f"invalid JSON authority: {label}") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise SnapshotError(f"noncanonical JSON authority: {label}")
    return value


def read_fd(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise SnapshotError("held read size differs")
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
        raise SnapshotError("held read is incomplete")
    return raw


class HeldRegular:
    def __init__(
        self, path: Path, descriptor: int, identity: tuple[int, ...], raw: bytes,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.identity = identity
        self.raw = raw

    def replay(self, *, expected_mode: int) -> None:
        opened = os.fstat(self.descriptor)
        named = os.lstat(self.path)
        raw = read_fd(self.descriptor, opened.st_size)
        if (
            ident(opened) != self.identity
            or ident(named) != self.identity
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != expected_mode
            or raw != self.raw
        ):
            raise SnapshotError(f"held authority changed: {self.path}")

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def open_held_regular(
    path: Path, *, expected_sha256: str | None, expected_size: int | None,
    expected_mode: int,
) -> HeldRegular:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise SnapshotError(f"noncanonical held authority: {path}")
    try:
        named = os.lstat(path)
    except OSError as error:
        raise SnapshotError(f"missing held authority: {path}") from error
    if (
        not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
        or stat.S_IMODE(named.st_mode) != expected_mode
        or path.resolve(strict=True) != path
        or (expected_size is not None and named.st_size != expected_size)
    ):
        raise SnapshotError(f"named held authority differs: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        first = read_fd(descriptor, before.st_size)
        middle = os.fstat(descriptor)
        second = read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named_after = os.lstat(path)
        if (
            ident(before) != ident(named) or ident(before) != ident(middle)
            or ident(before) != ident(after)
            or ident(before) != ident(named_after) or first != second
            or (expected_sha256 is not None
                and hashlib.sha256(first).hexdigest() != expected_sha256)
            or (expected_size is not None and len(first) != expected_size)
        ):
            raise SnapshotError(f"held authority replay differs: {path}")
        return HeldRegular(path, descriptor, ident(before), first)
    except BaseException:
        os.close(descriptor)
        raise


def read_stable(
    path: Path, pin: str | None, *, expected_mode: int | None = None,
) -> bytes:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise SnapshotError(f"noncanonical source: {path}")
    try:
        named_before = os.lstat(path)
    except OSError as error:
        raise SnapshotError(f"missing source authority: {path}") from error
    # This gate deliberately precedes resolve/open/read.  O_NONBLOCK protects
    # the remaining lstat/open race from a swapped FIFO.
    if not stat.S_ISREG(named_before.st_mode) or named_before.st_nlink != 1:
        raise SnapshotError(f"source is not one regular single-link file: {path}")
    if path.resolve(strict=True) != path:
        raise SnapshotError(f"source path resolves elsewhere: {path}")
    fd = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or ident(before) != ident(named_before)
        ):
            raise SnapshotError(f"opened source identity differs: {path}")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(fd, min(1_048_576, before.st_size - offset), offset)
            if not block:
                break
            chunks.append(block); offset += len(block)
        raw = b"".join(chunks)
        middle = os.fstat(fd)
        replay = b"".join(
            os.pread(fd, min(1_048_576, before.st_size - at), at)
            for at in range(0, before.st_size, 1_048_576)
        )
        eof = os.pread(fd, 1, before.st_size)
        after = os.fstat(fd)
        named_after = os.lstat(path)
    finally:
        os.close(fd)
    if (
        len(raw) != before.st_size or replay != raw or eof != b""
        or ident(before) != ident(middle) or ident(before) != ident(after)
        or ident(before) != ident(named_after)
        or (pin is not None and hashlib.sha256(raw).hexdigest() != pin)
        or (expected_mode is not None
            and stat.S_IMODE(before.st_mode) != expected_mode)
    ):
        raise SnapshotError(f"source authority differs: {path}")
    return raw


def expected_directories(files: Sequence[str] | set[str]) -> set[str]:
    result = {"."}
    for relative in files:
        parent = Path(relative).parent
        while str(parent) != ".":
            result.add(str(parent)); parent = parent.parent
    return result


def ordered_directories(files: Sequence[str] | set[str]) -> list[str]:
    return sorted(expected_directories(files), key=lambda value: (value.count("/"), value))


def staging_source_rows(builder_sha256: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative, sha256 in {**STAGED_FILES, BUILDER_RELATIVE: builder_sha256}.items():
        path = STAGING_ROOT / relative
        try:
            size = os.lstat(path).st_size
        except OSError as error:
            raise SnapshotError(f"missing physical15 source: {path}") from error
        rows.append({"relative": relative, "sha256": sha256, "size": size})
    rows.sort(key=lambda row: row["relative"])
    if len(rows) != 15 or len({row["relative"] for row in rows}) != 15:
        raise SnapshotError("physical15 source row closure differs")
    return rows


def expected_staging_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": STAGING_MANIFEST_SCHEMA,
        "target_root": str(STAGING_ROOT),
        "receipt_path": str(STAGING_RECEIPT_PATH),
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
    value["manifest_digest"] = object_digest(value)
    return value


def _receipt_inode_anchor(info: os.stat_result) -> list[int]:
    return [
        int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_gid),
        int(stat.S_IFMT(info.st_mode)),
    ]


def _validate_staging_receipt(
    receipt: Mapping[str, Any], *, receipt_info: os.stat_result,
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
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_digest", None)
    manifest = expected_staging_manifest(rows)
    tree_rows = [
        {
            "relative": row["relative"], "sha256": row["sha256"],
            "size": row["size"], "mode": STAGING_FILE_MODE, "nlink": 1,
        }
        for row in rows
    ]
    operation = receipt.get("operation")
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
        payload_relation = (
            receipt.get("request_payload_sha256")
            == receipt.get("stage_payload_sha256")
        )
        terminal_relation = receipt.get("commit_terminal_digest") is None
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
            type(receipt.get("commit_terminal_digest")) is str
            and SHA_RE.fullmatch(receipt["commit_terminal_digest"]) is not None
        )
    else:
        raise SnapshotError("staging receipt operation differs")
    if (
        set(receipt) != fields
        or claimed != object_digest(unsigned)
        or receipt.get("schema_version") != STAGING_RECEIPT_SCHEMA
        or receipt.get("status") != status
        or receipt.get("target_root") != str(STAGING_ROOT)
        or receipt.get("receipt_path") != str(STAGING_RECEIPT_PATH)
        or receipt.get("manifest_digest") != manifest["manifest_digest"]
        or any(
            type(receipt.get(key)) is not str
            or SHA_RE.fullmatch(receipt[key]) is None
            for key in (
                "request_payload_sha256", "stage_payload_sha256",
                "bootstrap_source_sha256",
            )
        )
        or receipt.get("bootstrap_source_sha256") != STAGING_BOOTSTRAP_SHA256
        or not payload_relation or not terminal_relation
        or receipt.get("file_count") != 15
        or receipt.get("files") != tree_rows
        or receipt.get("directories") != manifest["directories"]
        or receipt.get("file_mode") != STAGING_FILE_MODE
        or receipt.get("directory_mode") != STAGING_DIRECTORY_MODE
        or receipt.get("receipt_mode") != STAGING_RECEIPT_MODE
        or receipt.get("held_parent_identity_replayed") is not True
        or receipt.get("ancestor_chain_nofollow") is not True
        or receipt.get("publication_protocol") != STAGING_PUBLICATION_PROTOCOL
        or receipt.get("rename_noreplace") is not False
        or receipt.get("cooperative_writer_exclusion") is not True
        or receipt.get("receipt_is_consumption_gate") is not True
        or receipt.get("receipt_is_admission") is not True
        or receipt.get("uncooperative_same_uid_race_out_of_scope") is not True
        or receipt.get("target_observation") != observation
        or receipt.get("receipt_inode_anchor") != _receipt_inode_anchor(receipt_info)
        or receipt_info.st_uid != STAGING_REMOTE_UID
        or receipt_info.st_gid != STAGING_REMOTE_GID
        or receipt.get("launch_allowed") is not False
    ):
        raise SnapshotError("final AUHv2 staging receipt closure differs")


class HeldStagingGate:
    def __init__(
        self, *, receipt: HeldRegular, receipt_value: dict[str, Any],
        root_descriptor: int, root_identity: tuple[int, ...],
        leaves: Mapping[str, HeldRegular], rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self.receipt = receipt
        self.receipt_value = dict(receipt_value)
        self.root_descriptor = root_descriptor
        self.root_identity = root_identity
        self.leaves = dict(leaves)
        self.rows = [dict(row) for row in rows]

    def replay(self) -> None:
        self.receipt.replay(expected_mode=STAGING_RECEIPT_MODE)
        opened = os.fstat(self.root_descriptor)
        named = os.lstat(STAGING_ROOT)
        if (
            ident(opened) != self.root_identity
            or ident(named) != self.root_identity
            or not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != STAGING_DIRECTORY_MODE
            or opened.st_uid != STAGING_REMOTE_UID
            or opened.st_gid != STAGING_REMOTE_GID
        ):
            raise SnapshotError("held staging target identity changed")
        validate_exact_tree(
            STAGING_ROOT,
            expected_files=set(self.leaves), file_mode=STAGING_FILE_MODE,
            directory_mode=STAGING_DIRECTORY_MODE,
            expected_uid=STAGING_REMOTE_UID, expected_gid=STAGING_REMOTE_GID,
        )
        for authority in self.leaves.values():
            authority.replay(expected_mode=STAGING_FILE_MODE)
        value = strict_json(self.receipt.raw, label="final AUHv2 staging receipt")
        _validate_staging_receipt(
            value, receipt_info=os.fstat(self.receipt.descriptor),
            root_identity=self.root_identity, rows=self.rows,
        )
        if value != self.receipt_value:
            raise SnapshotError("held staging receipt value changed")

    def close(self) -> None:
        for authority in self.leaves.values():
            authority.close()
        self.leaves.clear()
        if self.root_descriptor >= 0:
            os.close(self.root_descriptor)
            self.root_descriptor = -1
        self.receipt.close()


def open_staging_gate(builder_sha256: str) -> HeldStagingGate:
    """Open the final 0400 receipt first, then bind its live exact15 tree."""
    receipt = open_held_regular(
        STAGING_RECEIPT_PATH, expected_sha256=None, expected_size=None,
        expected_mode=STAGING_RECEIPT_MODE,
    )
    root_descriptor = -1
    leaves: dict[str, HeldRegular] = {}
    try:
        receipt_value = strict_json(
            receipt.raw, label="final AUHv2 staging receipt",
        )
        rows = staging_source_rows(builder_sha256)
        root_descriptor = os.open(
            STAGING_ROOT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        root_info = os.fstat(root_descriptor)
        named_root = os.lstat(STAGING_ROOT)
        root_identity = ident(root_info)
        if (
            ident(named_root) != root_identity
            or not stat.S_ISDIR(root_info.st_mode)
            or stat.S_IMODE(root_info.st_mode) != STAGING_DIRECTORY_MODE
            or root_info.st_uid != STAGING_REMOTE_UID
            or root_info.st_gid != STAGING_REMOTE_GID
            or STAGING_ROOT.resolve(strict=True) != STAGING_ROOT
        ):
            raise SnapshotError("staging target root authority differs")
        validate_exact_tree(
            STAGING_ROOT,
            expected_files={row["relative"] for row in rows},
            file_mode=STAGING_FILE_MODE,
            directory_mode=STAGING_DIRECTORY_MODE,
            expected_uid=STAGING_REMOTE_UID, expected_gid=STAGING_REMOTE_GID,
        )
        for row in rows:
            leaves[row["relative"]] = open_held_regular(
                STAGING_ROOT / row["relative"],
                expected_sha256=row["sha256"], expected_size=row["size"],
                expected_mode=STAGING_FILE_MODE,
            )
        _validate_staging_receipt(
            receipt_value, receipt_info=os.fstat(receipt.descriptor),
            root_identity=root_identity, rows=rows,
        )
        held = HeldStagingGate(
            receipt=receipt, receipt_value=receipt_value,
            root_descriptor=root_descriptor, root_identity=root_identity,
            leaves=leaves, rows=rows,
        )
        held.replay()
        return held
    except BaseException:
        for authority in leaves.values():
            authority.close()
        if root_descriptor >= 0:
            os.close(root_descriptor)
        receipt.close()
        raise


def require_real_directory_chain(path: Path, *, label: str) -> None:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise SnapshotError(f"{label} parent is not canonical")
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
            raise SnapshotError(f"missing {label} ancestor: {component}") from error
        if not stat.S_ISDIR(info.st_mode) or component.resolve(strict=True) != component:
            raise SnapshotError(f"linked/non-directory {label} ancestor: {component}")


def open_held_parent(target: Path) -> int:
    require_real_directory_chain(target.parent, label="snapshot target")
    descriptor = os.open(
        target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    opened = os.fstat(descriptor); named = os.lstat(target.parent)
    if ident(opened) != ident(named) or target.parent.resolve(strict=True) != target.parent:
        os.close(descriptor)
        raise SnapshotError("held snapshot parent identity differs")
    return descriptor


class HeldPublicationReservation:
    def __init__(
        self, *, name: str, descriptor: int, anchor: list[int], raw: bytes,
    ) -> None:
        self.name = name
        self.descriptor = descriptor
        self.anchor = list(anchor)
        self.raw = raw

    def replay(self, parent_fd: int, *, expected_mode: int) -> bytes:
        opened = os.fstat(self.descriptor)
        named = os.stat(self.name, dir_fd=parent_fd, follow_symlinks=False)
        raw = read_fd(self.descriptor, opened.st_size)
        if (
            _receipt_inode_anchor(opened) != self.anchor
            or _receipt_inode_anchor(named) != self.anchor
            or ident(opened) != ident(named)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != expected_mode
            or raw != self.raw
        ):
            raise SnapshotError("held snapshot publication reservation changed")
        return raw

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def create_publication_reservation(
    parent_fd: int, *, receipt_path: Path, target_root: Path,
) -> HeldPublicationReservation:
    if receipt_path.parent != target_root.parent or receipt_path.name in ("", ".", ".."):
        raise SnapshotError("snapshot publication receipt path differs")
    descriptor = os.open(
        receipt_path.name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0, dir_fd=parent_fd,
    )
    try:
        created = os.fstat(descriptor)
        anchor = _receipt_inode_anchor(created)
        value: dict[str, Any] = {
            "schema_version": SNAPSHOT_PUBLICATION_SCHEMA + "-reservation",
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
        value["reservation_digest"] = object_digest(value)
        raw = canonical(value) + b"\n"
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise SnapshotError("snapshot reservation write made no progress")
            offset += count
        os.fsync(descriptor); os.fchmod(descriptor, PUBLICATION_RESERVATION_MODE)
        os.fsync(descriptor); os.fsync(parent_fd)
        held = HeldPublicationReservation(
            name=receipt_path.name, descriptor=descriptor,
            anchor=anchor, raw=raw,
        )
        held.replay(parent_fd, expected_mode=PUBLICATION_RESERVATION_MODE)
        return held
    except BaseException:
        os.close(descriptor)
        raise


def publish_under_reservation(
    parent_fd: int, shadow_name: str, target_name: str,
    reservation: HeldPublicationReservation,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """One ordinary same-parent rename under a cooperative O_EXCL gate.

    This is truthful on NFS and deliberately makes no kernel no-replace claim.
    Another conforming writer is excluded by the held receipt reservation; a
    same-UID/root/kernel/mount attacker is outside the declared threat model.
    """
    reservation.replay(parent_fd, expected_mode=PUBLICATION_RESERVATION_MODE)
    try:
        os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise SnapshotError("snapshot target appeared before ordinary rename")
    shadow_fd = os.open(
        shadow_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        shadow_before = os.fstat(shadow_fd)
        shadow_anchor = _receipt_inode_anchor(shadow_before)
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
            if source is None and target is not None and _receipt_inode_anchor(target) == shadow_anchor:
                return "target_same_inode_source_absent", target
            if source is not None and _receipt_inode_anchor(source) == shadow_anchor and target is None:
                return "source_same_inode_target_absent", None
            return "ambiguous_namespace", target

        namespace_state, target_after_rename = observed_namespace()
        if namespace_state != "target_same_inode_source_absent":
            if namespace_state == "source_same_inode_target_absent":
                raise SnapshotError(
                    "ordinary snapshot rename was not applied"
                    + ("" if rename_error is None else f": errno={rename_error.errno}")
                )
            raise SnapshotError("ordinary snapshot rename outcome is ambiguous")
        fsync_error: OSError | None = None
        try:
            os.fsync(parent_fd)
        except OSError as error:
            fsync_error = error
        namespace_state, named = observed_namespace()
        if namespace_state != "target_same_inode_source_absent" or named is None:
            raise SnapshotError("snapshot namespace changed after committed rename")
        named = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(shadow_fd)
        if (
            _receipt_inode_anchor(named) != shadow_anchor
            or _receipt_inode_anchor(opened) != shadow_anchor
            or ident(named) != ident(opened)
            or stat.S_IMODE(opened.st_mode) != 0o555
        ):
            raise SnapshotError("snapshot publication inode continuity differs")
        reservation.replay(parent_fd, expected_mode=PUBLICATION_RESERVATION_MODE)
        observation = {
            "namespace_state": namespace_state,
            "rename_returned_zero": rename_error is None,
            "rename_error_errno": None if rename_error is None else rename_error.errno,
            "parent_fsync_returned_zero": fsync_error is None,
            "parent_fsync_error_errno": None if fsync_error is None else fsync_error.errno,
        }
        identity = ident(opened)
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
        raise SnapshotError("snapshot receipt target authority is incomplete")
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
        raise SnapshotError("snapshot receipt publication paths differ")
    target_named = os.stat(
        target_root.name, dir_fd=parent_fd, follow_symlinks=False,
    )
    target_path_named = os.lstat(target_root)
    receipt_path_named = os.lstat(receipt_path)
    parsed = strict_json(raw, label="snapshot publication receipt")
    if (
        _receipt_inode_anchor(opened) != reservation.anchor
        or _receipt_inode_anchor(named) != reservation.anchor
        or _receipt_inode_anchor(receipt_path_named) != reservation.anchor
        or ident(opened) != ident(named)
        or ident(opened) != ident(receipt_path_named)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) != PUBLICATION_RECEIPT_MODE
        or opened.st_size != len(raw)
        or read_fd(reservation.descriptor, len(raw)) != raw
        or parsed != materialized
        or materialized.get("receipt_inode_anchor") != list(reservation.anchor)
        or ident(target_named) != tuple(target_identity_value)
        or ident(target_path_named) != tuple(target_identity_value)
        or not stat.S_ISDIR(target_named.st_mode)
    ):
        raise SnapshotError("snapshot sealed receipt/target audit differs")
    return parsed


def seal_publication_receipt(
    parent_fd: int, reservation: HeldPublicationReservation,
    value: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    reservation.replay(parent_fd, expected_mode=PUBLICATION_RESERVATION_MODE)
    materialized = dict(value)
    materialized["receipt_inode_anchor"] = list(reservation.anchor)
    materialized.pop("receipt_digest", None)
    materialized["receipt_digest"] = object_digest(materialized)
    raw = canonical(materialized) + b"\n"
    os.fchmod(reservation.descriptor, PUBLICATION_RESERVATION_MODE)
    os.ftruncate(reservation.descriptor, 0)
    offset = 0
    while offset < len(raw):
        count = os.pwrite(reservation.descriptor, raw[offset:], offset)
        if count <= 0:
            raise SnapshotError("snapshot receipt write made no progress")
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
        # Only a definitely uncommitted fchmod may remain a mutable 0600
        # reservation.  Once either fchmod returned or exact 0400 was
        # observed, this inode is an immutable receipt terminal.
        if (
            fchmod_error is not None
            and stat.S_IMODE(opened.st_mode) == PUBLICATION_RESERVATION_MODE
            and stat.S_IMODE(named.st_mode) == PUBLICATION_RESERVATION_MODE
            and _receipt_inode_anchor(opened) == reservation.anchor
            and _receipt_inode_anchor(named) == reservation.anchor
            and ident(opened) == ident(named)
            and read_fd(reservation.descriptor, opened.st_size) == raw
        ):
            reservation.raw = raw
            raise SnapshotError(
                "snapshot receipt fchmod was not applied; 0600 reservation remains"
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
        # No chmod/truncate/unlink is legal after the 0400 commit point.
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
    # The reservation entry was already parent-fsynced at creation.  Exact
    # held/named 0400 receipt bytes plus exact target identity are sufficient
    # to accept an applied-then-error fchmod or post-commit fsync error.
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


def validate_exact_tree(
    root: Path, *, expected_files: set[str], file_mode: int,
    directory_mode: int | None,
    file_modes: Mapping[str, int] | None = None,
    expected_uid: int | None = None, expected_gid: int | None = None,
) -> None:
    if not root.is_absolute() or os.path.normpath(str(root)) != str(root):
        raise SnapshotError("tree root is not canonical")
    root_info = os.lstat(root)
    if not stat.S_ISDIR(root_info.st_mode) or root.resolve(strict=True) != root:
        raise SnapshotError("tree root is not one real directory")
    actual_files: set[str] = set()
    actual_directories = {"."}
    pending = [(root, ".")]
    while pending:
        directory, prefix = pending.pop()
        info = os.lstat(directory)
        if (
            not stat.S_ISDIR(info.st_mode)
            or (directory_mode is not None
                and stat.S_IMODE(info.st_mode) != directory_mode)
            or (directory_mode is None and stat.S_IMODE(info.st_mode) & 0o022)
            or (expected_uid is not None and info.st_uid != expected_uid)
            or (expected_gid is not None and info.st_gid != expected_gid)
        ):
            raise SnapshotError(f"tree directory authority differs: {prefix}")
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
                        raise SnapshotError(f"tree file authority differs: {relative}")
                    actual_files.add(relative)
                elif stat.S_ISDIR(child.st_mode):
                    actual_directories.add(relative)
                    pending.append((Path(entry.path), relative))
                else:
                    raise SnapshotError(f"tree contains special/link entry: {relative}")
    if (
        actual_files != expected_files
        or actual_directories != expected_directories(expected_files)
    ):
        raise SnapshotError("physical tree closure differs")


def validate_old_snapshot(old_root: Path) -> dict[str, bytes]:
    expected_tree = set(OLD_EXACT5_FILES) | {OLD_MANIFEST_NAME}
    validate_exact_tree(
        old_root, expected_files=expected_tree, file_mode=0o444,
        directory_mode=0o555,
    )
    manifest_raw = read_stable(
        old_root / OLD_MANIFEST_NAME, None, expected_mode=0o444,
    )
    manifest = strict_json(manifest_raw, label="old exact24 manifest")
    unsigned = dict(manifest); claimed = unsigned.pop("snapshot_digest", None)
    rows = manifest.get("files")
    fields = {
        "schema_version", "status", "target_root", "old_r5f_snapshot_root",
        "new_staging_root", "file_count", "old_reused_file_count",
        "new_staged_file_count", "physical_file_count_including_manifest",
        "release_file_count", "sealed_r5f_infer_lora_reused",
        "working_tree_infer_lora_read", "slurm_step_launched", "files",
        "snapshot_digest",
    }
    if (
        set(manifest) != fields
        or manifest.get("schema_version")
        != "case01-source-bone-exact5-source-snapshot-v1"
        or manifest.get("status") != "SEALED_NOT_EXECUTED"
        or claimed != hashlib.sha256(canonical(unsigned)).hexdigest()
        or manifest.get("target_root") != str(old_root)
        or manifest.get("old_r5f_snapshot_root") != str(OLD_R5F_SNAPSHOT)
        or manifest.get("new_staging_root") != str(OLD_EXACT5_STAGING)
        or manifest.get("file_count") != 23
        or manifest.get("old_reused_file_count") != 16
        or manifest.get("new_staged_file_count") != 7
        or manifest.get("physical_file_count_including_manifest") != 24
        or manifest.get("release_file_count") != 19
        or manifest.get("sealed_r5f_infer_lora_reused") is not True
        or manifest.get("working_tree_infer_lora_read") is not False
        or manifest.get("slurm_step_launched") is not False
        or type(rows) is not list or len(rows) != 23
        or [row.get("path") if type(row) is dict else None for row in rows]
        != sorted(OLD_EXACT5_FILES)
    ):
        raise SnapshotError("old exact24 manifest closure differs")
    raw_by_relative: dict[str, bytes] = {}
    for row in rows:
        relative = row["path"]
        expected_provenance = (
            "new_exact5_staging" if relative in OLD_EXACT5_FRESH
            else "sealed_r5f_snapshot"
        )
        if (
            set(row) != {"path", "sha256", "size", "mode", "provenance"}
            or row.get("sha256") != OLD_EXACT5_FILES[relative]
            or row.get("mode") != 0o444
            or row.get("provenance") != expected_provenance
            or type(row.get("size")) is not int or row["size"] <= 0
        ):
            raise SnapshotError(f"old exact24 manifest row differs: {relative}")
        raw = read_stable(
            old_root / relative, OLD_EXACT5_FILES[relative], expected_mode=0o444,
        )
        if len(raw) != row["size"]:
            raise SnapshotError(f"old exact24 row size differs: {relative}")
        raw_by_relative[relative] = raw
    return raw_by_relative


def validate_staging(
    staging_root: Path, *, builder_sha256: str,
    held_gate: HeldStagingGate | None = None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Return exact14 leaves plus the pinned, executing builder authority."""
    if SHA_RE.fullmatch(builder_sha256) is None:
        raise SnapshotError("builder SHA pin is incomplete")
    if held_gate is None:
        validate_exact_tree(
            staging_root,
            expected_files=set(STAGED_FILES) | {BUILDER_RELATIVE}, file_mode=0o444,
            directory_mode=0o555,
        )
        raw_by_relative = {
            relative: read_stable(staging_root / relative, pin, expected_mode=0o444)
            for relative, pin in STAGED_FILES.items()
        }
        builder_raw = read_stable(
            staging_root / BUILDER_RELATIVE, builder_sha256, expected_mode=0o444,
        )
    else:
        if staging_root != STAGING_ROOT:
            raise SnapshotError("held staging gate root differs")
        held_gate.replay()
        raw_by_relative = {
            relative: held_gate.leaves[relative].raw for relative in STAGED_FILES
        }
        builder_raw = held_gate.leaves[BUILDER_RELATIVE].raw
    executing_raw = read_stable(Path(__file__).resolve(), builder_sha256)
    if executing_raw != builder_raw:
        raise SnapshotError("executing builder differs from staged builder authority")
    authority = {
        "path": str(staging_root / BUILDER_RELATIVE),
        "sha256": builder_sha256, "size": len(builder_raw),
        "sealed_bytes_in_snapshot": False,
    }
    return raw_by_relative, authority


def create(path: Path, raw: bytes, mode: int = 0o444) -> None:
    if mode not in (0o400, 0o444):
        raise SnapshotError(f"snapshot file mode differs: {path}")
    if os.path.lexists(path) or not path.parent.is_dir():
        raise SnapshotError(f"snapshot target is not create-only: {path}")
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                raise SnapshotError("snapshot write made no progress")
            offset += written
        os.fsync(fd); os.fchmod(fd, mode); os.fsync(fd)
    finally:
        os.close(fd)
    observed = read_stable(
        path, hashlib.sha256(raw).hexdigest(), expected_mode=mode,
    )
    if observed != raw:
        raise SnapshotError(f"created snapshot replay differs: {path}")


def fsync_shadow_directories(root: Path) -> None:
    """Durably close every directory in a completed shadow, children first."""
    walked = [Path(directory) for directory, _subdirs, _files in os.walk(
        root, topdown=False,
    )]
    if not walked or walked[-1] != root:
        raise SnapshotError("snapshot shadow directory walk differs")
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
                or ident(opened_before) != ident(named_before)
            ):
                raise SnapshotError(
                    f"snapshot shadow directory identity differs: {directory}"
                )
            os.fsync(descriptor)
            opened_after = os.fstat(descriptor)
            named_after = os.lstat(directory)
            if (
                ident(opened_after) != ident(opened_before)
                or ident(named_after) != ident(opened_before)
            ):
                raise SnapshotError(
                    f"snapshot shadow directory changed while syncing: {directory}"
                )
        finally:
            os.close(descriptor)


def build(
    old_root: Path, staging_root: Path, target_root: Path, *,
    builder_sha256: str,
) -> dict[str, Any]:
    blocked = blocked_sources()
    if blocked:
        raise SnapshotError("HOLD: final source pins blocked: " + ",".join(blocked))
    for supplied, expected, label in (
        (old_root, OLD_EXACT5_SNAPSHOT, "old root"),
        (staging_root, STAGING_ROOT, "staging root"),
        (target_root, TARGET_ROOT, "target root"),
    ):
        if supplied != expected:
            raise SnapshotError(f"CLI {label} is not the sealed configured path")
    if (
        os.path.lexists(target_root)
        or os.path.lexists(SNAPSHOT_PUBLICATION_RECEIPT_PATH)
    ):
        raise SnapshotError("snapshot target/publication receipt is not fresh")
    require_real_directory_chain(target_root.parent, label="snapshot target")
    held_gate: HeldStagingGate | None = None
    parent_fd = -1
    reservation: HeldPublicationReservation | None = None
    try:
        # Receipt-first is literal: the final canonical 0400 AUHv2 receipt is
        # opened before the staging root or any physical15 leaf.  Everything
        # below remains read-only until both source trees have been replayed.
        held_gate = open_staging_gate(builder_sha256)
        old_bytes = validate_old_snapshot(old_root)
        staged_bytes, builder_authority = validate_staging(
            staging_root, builder_sha256=builder_sha256,
            held_gate=held_gate,
        )
        held_gate.replay()
        staging_receipt_raw = held_gate.receipt.raw
        staging_receipt_sha256 = hashlib.sha256(staging_receipt_raw).hexdigest()
        staging_receipt_value = held_gate.receipt_value
        staging_receipt_authority = {
            "source_path": str(STAGING_RECEIPT_PATH),
            "snapshot_relative": STAGING_RECEIPT_COPY_RELATIVE,
            "sha256": staging_receipt_sha256,
            "size": len(staging_receipt_raw),
            "mode": STAGING_RECEIPT_MODE,
            "schema_version": STAGING_RECEIPT_SCHEMA,
            "receipt_digest": staging_receipt_value["receipt_digest"],
            "staging_manifest_digest": staging_receipt_value["manifest_digest"],
            "staging_target_root_identity": staging_receipt_value[
                "target_observation"
            ]["root_identity"],
            "staging_file_count": 15,
            "copied_as_snapshot_leaf": True,
            "replayed_before_and_after_snapshot_build": True,
        }
        rows: list[dict[str, Any]] = []
        raw_by_relative: dict[str, bytes] = {}
        for relative, pin in REUSED_FILES.items():
            raw = old_bytes[relative]; raw_by_relative[relative] = raw
            rows.append({"path": relative, "sha256": pin, "size": len(raw),
                         "mode": 0o444,
                         "provenance": "sealed_legacy_exact5_snapshot"})
        for relative, pin in STAGED_FILES.items():
            raw = staged_bytes[relative]; raw_by_relative[relative] = raw
            rows.append({"path": relative, "sha256": pin, "size": len(raw),
                         "mode": 0o444,
                         "provenance": "fresh_pinned_staging"})
        alias_raw = raw_by_relative[ALIAS_SOURCE]
        raw_by_relative[ALIAS_RELATIVE] = alias_raw
        rows.append({"path": ALIAS_RELATIVE,
                     "sha256": hashlib.sha256(alias_raw).hexdigest(),
                     "size": len(alias_raw), "mode": 0o444,
                     "provenance": "independent_inode_copy_of_sealed_legacy_infer"})
        raw_by_relative[STAGING_RECEIPT_COPY_RELATIVE] = staging_receipt_raw
        rows.append({
            "path": STAGING_RECEIPT_COPY_RELATIVE,
            "sha256": staging_receipt_sha256,
            "size": len(staging_receipt_raw), "mode": STAGING_RECEIPT_MODE,
            "provenance": "copied_exact_auh_v2_staging_receipt_authority",
        })
        if len(raw_by_relative) != 34 or len(set(raw_by_relative)) != 34:
            raise SnapshotError("source leaf count differs")

        publish_root = target_root
        parent_fd = open_held_parent(publish_root)
        for path, label in (
            (publish_root, "snapshot target"),
            (SNAPSHOT_PUBLICATION_RECEIPT_PATH, "snapshot receipt"),
        ):
            try:
                os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise SnapshotError(f"{label} appeared before reservation")
        reservation = create_publication_reservation(
            parent_fd, receipt_path=SNAPSHOT_PUBLICATION_RECEIPT_PATH,
            target_root=publish_root,
        )
        held_gate.replay()
        shadow_name = (
            f".{publish_root.name}.shadow.{os.getpid()}.{secrets.token_hex(12)}"
        )
        os.mkdir(shadow_name, mode=0o700, dir_fd=parent_fd)
        shadow_root = publish_root.parent / shadow_name
        for relative in sorted(expected_directories(set(raw_by_relative)) - {"."}):
            (shadow_root / relative).mkdir(mode=0o700)
        for relative, raw in raw_by_relative.items():
            create(
                shadow_root / relative, raw,
                STAGING_RECEIPT_MODE
                if relative == STAGING_RECEIPT_COPY_RELATIVE else 0o444,
            )
        manifest: dict[str, Any] = {
            "schema_version": "case01-object-trajectory-exact5-source-snapshot-v2",
            "status": "SEALED_SOURCE_ONLY_NOT_LAUNCHABLE",
            "launch_allowed": False,
            "old_snapshot_root": str(old_root), "staging_root": str(staging_root),
            "staging_receipt_path": str(STAGING_RECEIPT_PATH),
            "snapshot_publication_receipt_path": str(
                SNAPSHOT_PUBLICATION_RECEIPT_PATH
            ),
            "target_root": str(publish_root), "content_leaf_count": 34,
            "physical_file_count_including_manifest": 35,
            "release_file_count": 25,
            "legacy_alias_is_distinct_regular_inode": True,
            "builder_authority": builder_authority,
            "staging_receipt_authority": staging_receipt_authority,
            "publication_protocol": STAGING_PUBLICATION_PROTOCOL,
            "rename_noreplace": False,
            "cooperative_writer_exclusion": True,
            "target_absent_rechecked": True,
            "whole_tree_atomically_visible": True,
            "uncooperative_same_uid_race_out_of_scope": True,
            "retry_allowed": False,
            "formal_review_test": FORMAL_REVIEW_TEST,
            "files": sorted(rows, key=lambda row: row["path"]),
        }
        manifest["manifest_digest"] = object_digest(manifest)
        manifest_raw = canonical(manifest) + b"\n"
        create(shadow_root / MANIFEST_NAME, manifest_raw)
        for directory, subdirs, _files in os.walk(shadow_root, topdown=False):
            for name in subdirs:
                os.chmod(Path(directory) / name, 0o555)
        os.chmod(shadow_root, 0o555)
        fsync_shadow_directories(shadow_root)
        expected_files = set(raw_by_relative) | {MANIFEST_NAME}
        file_modes = {relative: 0o444 for relative in expected_files}
        file_modes[STAGING_RECEIPT_COPY_RELATIVE] = STAGING_RECEIPT_MODE
        validate_exact_tree(
            shadow_root, expected_files=expected_files, file_mode=0o444,
            directory_mode=0o555, file_modes=file_modes,
        )
        alias_info = os.lstat(shadow_root / ALIAS_RELATIVE)
        source_info = os.lstat(shadow_root / ALIAS_SOURCE)
        if alias_info.st_ino == source_info.st_ino or alias_info.st_nlink != 1:
            raise SnapshotError("legacy alias is not a distinct regular inode")
        held_gate.replay()
        if ident(os.fstat(parent_fd)) != ident(os.lstat(publish_root.parent)):
            raise SnapshotError("held snapshot parent changed before publish")
        commit_error: PublicationCommittedError | None = None
        try:
            target_identity, publication_observation = publish_under_reservation(
                parent_fd, shadow_name, publish_root.name, reservation,
            )
        except PublicationCommittedError as error:
            commit_error = error
            target_identity = error.identity
            publication_observation = error.observation
        held_gate.replay()
        validate_exact_tree(
            publish_root, expected_files=expected_files, file_mode=0o444,
            directory_mode=0o555, file_modes=file_modes,
        )
        publication_value = {
            "schema_version": SNAPSHOT_PUBLICATION_SCHEMA + "-receipt",
            "status": (
                "PUBLISHED_RECEIPT_GATED" if commit_error is None
                else "PUBLISHED_COMMIT_ERROR_NOT_ADMISSION"
            ),
            "target_root": str(publish_root),
            "receipt_path": str(SNAPSHOT_PUBLICATION_RECEIPT_PATH),
            "manifest_path": str(publish_root / MANIFEST_NAME),
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "manifest_digest": manifest["manifest_digest"],
            "staging_receipt_sha256": staging_receipt_sha256,
            "staging_receipt_digest": staging_receipt_value["receipt_digest"],
            "content_leaf_count": 34,
            "physical_file_count_including_manifest": 35,
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
        if commit_error is not None:
            raise SnapshotError(
                "snapshot committed but publication syscall/fsync terminal failed"
            ) from commit_error
        return manifest
    finally:
        if reservation is not None:
            reservation.close()
        if parent_fd >= 0:
            os.close(parent_fd)
        if held_gate is not None:
            held_gate.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", default=str(OLD_EXACT5_SNAPSHOT))
    parser.add_argument("--staging-root", default=str(STAGING_ROOT))
    parser.add_argument("--target-root", default=str(TARGET_ROOT))
    parser.add_argument("--builder-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = build(
            Path(args.old_root), Path(args.staging_root), Path(args.target_root),
            builder_sha256=args.builder_sha256,
        )
    except (OSError, ValueError, SnapshotError) as error:
        print(str(error), file=sys.stderr)
        return 88 if str(error).startswith("HOLD:") else 96
    print(json.dumps(value, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
