#!/usr/bin/env python3
"""Create r7 exact15-r3 deployment inputs and print later-stage interfaces.

The commands in this helper are intentionally split at independent literal
SHA review boundaries.  ``phase-a`` only publishes the deployment request.
``phase-b`` only publishes the source/runtime spec after replaying a Phase-A
deployment receipt.  Interface commands print argv and never execute them.
No command launches Slurm, torchrun, inference, training, or a GPU process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA = "bernini-action-preservation-decoded-eval-deployment-request-v3"
SOURCE_RUNTIME_SCHEMA = "bernini-action-preservation-source-runtime-spec-v2"
SOURCE_PREPROCESSING_SCHEMA = (
    "bernini-action-preservation-decoded-eval-source-preprocessing-authority-v1"
)
RELEASE_GENERATION = "preservation-v2-decoded-eval-exact15-r3"
EVALUATION_ID = "apv2-r7-exact264-exact15-r3-2752c4ae"

REMOTE_BASE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments"
)
BUNDLE_ROOT = REMOTE_BASE / (
    "action_preservation_v2_decoded_eval_exact15_r3_r7_"
    "2752c4ae_1c3e40ec_20260816"
)
ARTIFACT_ROOT = BUNDLE_ROOT / "exact15-r3-release"
CONTROLLER_PATH = BUNDLE_ROOT / (
    "action_preservation_decoded_eval_deployment_controller_v1.py"
)
RUNTIME_SOURCE_PATH = BUNDLE_ROOT / (
    "action_preservation_decoded_eval_verified_release_v1.py"
)
SOURCE_PREPROCESSING_PATH = BUNDLE_ROOT / (
    "action_preservation_decoded_eval_r7_source_preprocessing_authority_v1.json"
)
INPUT_AUTHORITY_PATH = BUNDLE_ROOT / (
    "action_preservation_decoded_eval_r7_exact15_r3_input_authority.json"
)
PREPARE_PATH = BUNDLE_ROOT / (
    "prepare_action_preservation_decoded_eval_r7_exact15_r3_v1.py"
)
WORK_ROOT = REMOTE_BASE / (
    "action_preservation_v2_seed20260818_four_holder_r7_"
    "decoded_eval_exact15_r3_2752c4ae"
)
MATERIALIZED_RELEASE_ROOT = WORK_ROOT / "materialized-release"
DEPLOYMENT_REQUEST_PATH = WORK_ROOT / "deployment-request.json"
CONTROLLER_AUTHORITY_PATH = WORK_ROOT / "controller-authority.json"
DEPLOYMENT_RECEIPT_PATH = WORK_ROOT / "deployment-receipt.json"
SOURCE_SPEC_PATH = WORK_ROOT / "source-runtime-spec.json"
SOURCE_SPEC_AUTHORITY_PATH = WORK_ROOT / "source-spec-authority.json"
EVALUATION_ROOT = WORK_ROOT / "decoded-eval"
BRIDGE_ROOT = WORK_ROOT / "bridge"
LAUNCH_ROOT = WORK_ROOT / "launch"
AGGREGATE_ROOT = WORK_ROOT / "aggregate"
BLINDING_KEY_PATH = WORK_ROOT / "blind-key.bin"

EXPERIMENT_ROOT = REMOTE_BASE / (
    "action_preservation_v2_seed20260818_four_holder_r7"
)
TRAINING_COMPLETE_PATH = EXPERIMENT_ROOT / "TRAINING_COMPLETE.json"
TRAINING_AUDIT_PATH = EXPERIMENT_ROOT / "logs/training-audit.json"
TRAINING_COMPLETE_SHA256 = (
    "2752c4aee78c833c55f7c66bf5bebf84d42f6babe00692dd6cc94b918842409b"
)
TRAINING_AUDIT_SHA256 = (
    "70b743eb566ba80406473b3dbabfcacffacd028c811aef34069cbbd3aa5c59c5"
)
SOURCE_REVISION = "54a2bafa2a09ddcd26add20c211ea9f055d339c3"
SOURCE_ARCHIVE_SHA256 = (
    "71357c8a4212fd985ffc4f73e8422ae412502756e63c014bc1c260c10c53273f"
)

SOURCE_MANIFEST_PATH = REMOTE_BASE / (
    "action_quotient_job140846_v4/source_only/manifest.json"
)
ADAPTER_RELEASE_MANIFEST_PATH = REMOTE_BASE / (
    "action_preservation_v2_seed20260818_four_holder_release_"
    "54a2bafa_nfssafe1/source.manifest.json"
)
MODEL_MANIFEST_PATH = REMOTE_BASE / (
    "bernini_counterfactual_identity_orbit_v5_20260808_c099c6f/runtime/"
    "source_ea900d5/methods/bernini_action_editing/audits/"
    "bernini_r13_ff4c5d4_checkpoint.sha256"
)
BERNINI_ROOT = REMOTE_BASE / "motive_action_repr_auto/vendor/Bernini-2d2b4591"
VEOMNI_ROOT = REMOTE_BASE / (
    "bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
)
MODEL_ROOT = REMOTE_BASE.parent / (
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
INFERENCE_CONFIG_PATH = (
    BERNINI_ROOT / "configs/bernini_renderer_wan21_1p3b/config.json"
)

ROOT_PYTHON_PATH = Path("/usr/bin/python3.10")
FROZEN_PYTHON_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
SITE_PACKAGES_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"
)
TORCHRUN_PATH = SITE_PACKAGES_PATH / "torch/distributed/run.py"
FFPROBE_PATH = Path("/usr/bin/ffprobe")

CONTROLLER_SHA256 = (
    "4b463fdc92ee5a4070aaeeb352404c62523e309b07c0575cf6814700a70f9857"
)
RUNTIME_SOURCE_SHA256 = (
    "21765305dde685fc648c1e37e3fd360261048f80e92e16db324cb529194dbc3f"
)
ARCHIVE_SHA256 = (
    "ac810a3f2fbf30673015184e19d75b68b5c7bed7f879aec45d79ac80dd6a7bc1"
)
MANIFEST_SHA256 = (
    "5c0659502653cced2987b9152bace0fd979c91839d44fdcf1f88a25a50fa12da"
)
MANIFEST_DIGEST = (
    "f650a6fab618dc95b38c01163c870352bf86cd6c62c168ef699e412ee20126a7"
)
CONTENT_REVISION = "1c3e40ec6b63449c3427144ad8368028fad3bbc3"
ENVELOPE_SHA256 = (
    "95d8b2fa79fcf25af7831fa76551ae585c55df90acaff8f25390f6b248f9f383"
)
ENVELOPE_DIGEST = (
    "6ad3b84ea4014a3d598678faf195f00a1b377badc1d8ffa9cfca35c0b6019f68"
)
SOURCE_PREPROCESSING_SHA256 = (
    "f0ee7196c00fb0dd0b4345707ec8a069ee2ba20a6f304b1982ef8d7945be15dd"
)
INPUT_AUTHORITY_SHA256 = (
    "e90a8652bb54e3ebde8dc435ffe27bc7523d91e47d436bcf1b80ab2bfceb628d"
)
ROOT_PYTHON_SHA256 = (
    "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
)
FROZEN_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
TORCHRUN_SHA256 = (
    "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c"
)
FFPROBE_SHA256 = (
    "d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd"
)
SOURCE_MANIFEST_SHA256 = (
    "62fee73b3d84015f2e72edcd4da14b51f7695980a4ba892420ca137aa50e9ad8"
)
ADAPTER_RELEASE_MANIFEST_SHA256 = (
    "ce97493465dc0d5b3733be25966f6d2ca909ac24931c4840daa4c73dc4c62198"
)
MODEL_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
INFERENCE_CONFIG_SHA256 = (
    "4659e97bbb09f6c9baa3528dcdbb23064998e2f92aace8e8fd4b02776c529496"
)
INFER_SHA256 = (
    "dde5e3293e4fc833618c970eb51ba61fef4c66ef38dd1e67ab0e12b142f05e48"
)
DECODER_SHA256 = (
    "0b30ff6d2e4d17b20844abbeea5c26e51d376740cab092f905854279ad713fd1"
)
EXECUTOR_SHA256 = (
    "8915693b5816d7309e9f66f5a2b08975e579286c6df9e8ea410791e0ad3cce29"
)
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class R7DeploymentPreparationError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise R7DeploymentPreparationError("value is not finite canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R7DeploymentPreparationError(message)


def sha256(value: Any, *, label: str) -> str:
    require(type(value) is str and _SHA256_RE.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA-256")
    return value


def strict_json(raw: bytes, *, label: str, canonical: bool = True) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            require(key not in result, f"{label} contains a duplicate key")
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise R7DeploymentPreparationError(f"cannot decode {label}") from error
    require(type(value) is dict, f"{label} root differs")
    if canonical:
        require(raw == canonical_json_bytes(value) + b"\n",
                f"{label} serialization differs")
    return value


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid,
        value.st_mode, value.st_nlink, value.st_rdev, value.st_size,
        getattr(value, "st_blocks", 0), value.st_mtime_ns, value.st_ctime_ns,
    )


def _identity_row(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "mode": value.st_mode,
        "nlink": value.st_nlink,
        "rdev": value.st_rdev,
        "size": value.st_size,
        "blocks": getattr(value, "st_blocks", 0),
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _immutable_directory_identity(value: os.stat_result) -> dict[str, int]:
    return {
        key: _identity_row(value)[key]
        for key in ("device", "inode", "uid", "gid", "mode", "rdev")
    }


def _validate_work_root_authority(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version", "path", "parent_path", "creation_identity",
        "immutable_identity", "parent_immutable_identity", "initial_entries",
        "retained_parent_fd_through_request_publication",
        "retained_root_fd_through_request_publication", "authority_digest",
    }
    require(type(value) is dict and set(value) == fields,
            "work root authority field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    identity_fields = {
        "device", "inode", "uid", "gid", "mode", "nlink", "rdev",
        "size", "blocks", "mtime_ns", "ctime_ns",
    }
    immutable_fields = {"device", "inode", "uid", "gid", "mode", "rdev"}
    creation = row["creation_identity"]
    immutable = row["immutable_identity"]
    parent_immutable = row["parent_immutable_identity"]
    require(
        row["schema_version"]
        == "bernini-action-preservation-decoded-eval-work-root-authority-v1"
        and Path(row["path"]).is_absolute()
        and Path(row["parent_path"]).is_absolute()
        and Path(row["path"]).parent == Path(row["parent_path"])
        and type(creation) is dict
        and set(creation) == identity_fields
        and all(type(creation[key]) is int for key in identity_fields)
        and stat.S_ISDIR(creation["mode"])
        and stat.S_IMODE(creation["mode"]) == 0o700
        and type(immutable) is dict
        and set(immutable) == immutable_fields
        and all(type(immutable[key]) is int for key in immutable_fields)
        and immutable == {key: creation[key] for key in immutable_fields}
        and type(parent_immutable) is dict
        and set(parent_immutable) == immutable_fields
        and all(type(parent_immutable[key]) is int for key in immutable_fields)
        and stat.S_ISDIR(parent_immutable["mode"])
        and row["initial_entries"] == []
        and row["retained_parent_fd_through_request_publication"] is True
        and row["retained_root_fd_through_request_publication"] is True
        and type(claimed) is str
        and _SHA256_RE.fullmatch(claimed) is not None
        and claimed == object_sha256(unsigned),
        "work root authority differs",
    )
    return row


def _directory_entries(
    path: Path,
    *,
    expected: set[str],
    expected_mode: int = 0o700,
    label: str = "work root",
) -> dict[str, Any]:
    require(path.is_absolute(), f"{label} path differs")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    require(
        hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
        "safe directory replay is unavailable",
    )
    descriptor = os.open(path, flags | os.O_NOFOLLOW | os.O_DIRECTORY)
    try:
        before = os.fstat(descriptor)
        first = os.listdir(descriptor)
        middle = os.fstat(descriptor)
        second = os.listdir(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    require(
        stat.S_ISDIR(before.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and stat.S_IMODE(before.st_mode) == expected_mode
        and _identity(before) == _identity(middle) == _identity(after)
        and _identity(before) == _identity(named)
        and sorted(first) == sorted(second) == sorted(expected)
        and len(first) == len(expected),
        f"{label} exact entry closure differs",
    )
    return {
        "path": str(path),
        "mode": expected_mode,
        "entries": sorted(first),
        "device": before.st_dev,
        "inode": before.st_ino,
    }


class _PhaseAWorkAuthority:
    def __init__(
        self,
        *,
        parent_fd: int,
        root_fd: int,
        parent_anchor: os.stat_result,
        creation_identity: os.stat_result,
        immutable_identity: Mapping[str, int] | None = None,
        parent_immutable_identity: Mapping[str, int] | None = None,
        authority_value: Mapping[str, Any] | None = None,
    ) -> None:
        self.parent_fd = parent_fd
        self.root_fd = root_fd
        self.parent_anchor = parent_anchor
        self.creation_identity = creation_identity
        self.root_anchor = creation_identity
        self.immutable_identity = dict(
            immutable_identity
            if immutable_identity is not None
            else _immutable_directory_identity(creation_identity)
        )
        self.parent_immutable_identity = dict(
            parent_immutable_identity
            if parent_immutable_identity is not None
            else _immutable_directory_identity(parent_anchor)
        )
        self.authority_value = (
            None if authority_value is None else dict(authority_value)
        )
        self.closed = False

    @classmethod
    def create(cls) -> "_PhaseAWorkAuthority":
        require(WORK_ROOT.is_absolute(), "work root path differs")
        parent = WORK_ROOT.parent
        require(
            parent.resolve(strict=True) == parent
            and not os.path.lexists(WORK_ROOT),
            "work root is not fresh",
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        require(
            hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
            "safe directory creation is unavailable",
        )
        parent_fd = os.open(parent, flags | os.O_NOFOLLOW | os.O_DIRECTORY)
        root_fd: int | None = None
        try:
            require(
                stat.S_ISDIR(os.fstat(parent_fd).st_mode),
                "work parent differs",
            )
            os.mkdir(WORK_ROOT.name, mode=0o700, dir_fd=parent_fd)
            root_fd = os.open(
                WORK_ROOT.name,
                flags | os.O_NOFOLLOW | os.O_DIRECTORY,
                dir_fd=parent_fd,
            )
            os.set_inheritable(parent_fd, False)
            os.set_inheritable(root_fd, False)
            os.fchmod(root_fd, 0o700)
            os.fsync(root_fd)
            os.fsync(parent_fd)
            creation = os.fstat(root_fd)
            named = os.stat(
                WORK_ROOT.name, dir_fd=parent_fd, follow_symlinks=False
            )
            parent_anchor = os.fstat(parent_fd)
            named_parent = parent.lstat()
            require(
                stat.S_ISDIR(creation.st_mode)
                and stat.S_IMODE(creation.st_mode) == 0o700
                and _identity(creation) == _identity(named)
                and _identity(parent_anchor) == _identity(named_parent)
                and os.get_inheritable(parent_fd) is False
                and os.get_inheritable(root_fd) is False,
                "fresh work root creation replay differs",
            )
            authority = cls(
                parent_fd=parent_fd,
                root_fd=root_fd,
                parent_anchor=parent_anchor,
                creation_identity=creation,
            )
            authority.entries(expected=set())
            return authority
        except Exception:
            if root_fd is not None:
                os.close(root_fd)
            os.close(parent_fd)
            raise

    @classmethod
    def reopen(cls, value: Mapping[str, Any]) -> "_PhaseAWorkAuthority":
        authority = _validate_work_root_authority(value)
        root_path = Path(authority["path"])
        parent_path = Path(authority["parent_path"])
        require(
            root_path == WORK_ROOT and parent_path == WORK_ROOT.parent,
            "deployment work root differs from helper authority",
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        require(
            hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
            "safe directory replay is unavailable",
        )
        parent_fd = os.open(
            parent_path, flags | os.O_NOFOLLOW | os.O_DIRECTORY
        )
        root_fd: int | None = None
        try:
            root_fd = os.open(
                root_path.name,
                flags | os.O_NOFOLLOW | os.O_DIRECTORY,
                dir_fd=parent_fd,
            )
            os.set_inheritable(parent_fd, False)
            os.set_inheritable(root_fd, False)
            parent_anchor = os.fstat(parent_fd)
            root_anchor = os.fstat(root_fd)
            require(
                _immutable_directory_identity(root_anchor)
                == authority["immutable_identity"]
                and _immutable_directory_identity(parent_anchor)
                == authority["parent_immutable_identity"],
                "reopened work root physical identity differs",
            )
            observed = cls(
                parent_fd=parent_fd,
                root_fd=root_fd,
                parent_anchor=parent_anchor,
                creation_identity=root_anchor,
                immutable_identity=authority["immutable_identity"],
                parent_immutable_identity=authority[
                    "parent_immutable_identity"
                ],
                authority_value=authority,
            )
            observed._named_replay()
            return observed
        except Exception:
            if root_fd is not None:
                os.close(root_fd)
            os.close(parent_fd)
            raise

    def _require_open(self) -> None:
        require(not self.closed, "phase-A work authority is closed")

    def _named_replay(self) -> tuple[os.stat_result, os.stat_result]:
        self._require_open()
        try:
            root = os.fstat(self.root_fd)
            named_root = os.stat(
                WORK_ROOT.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
            parent = os.fstat(self.parent_fd)
            named_parent = WORK_ROOT.parent.lstat()
        except OSError as error:
            raise R7DeploymentPreparationError(
                "phase-A held work root named identity is unavailable"
            ) from error
        require(
            _identity(root) == _identity(named_root)
            and _identity(parent) == _identity(self.parent_anchor)
            and _identity(parent) == _identity(named_parent)
            and _immutable_directory_identity(root)
            == self.immutable_identity
            and _immutable_directory_identity(parent)
            == self.parent_immutable_identity,
            "phase-A held work root named identity differs",
        )
        return root, parent

    def entries(self, *, expected: set[str]) -> dict[str, Any]:
        before, _parent = self._named_replay()
        require(
            _identity(before) == _identity(self.root_anchor),
            "phase-A held work root identity drifted",
        )
        first = os.listdir(self.root_fd)
        middle = os.fstat(self.root_fd)
        second = os.listdir(self.root_fd)
        after, parent = self._named_replay()
        require(
            _identity(before) == _identity(middle) == _identity(after)
            and sorted(first) == sorted(second) == sorted(expected)
            and len(first) == len(expected),
            "phase-A held work root exact entry closure differs",
        )
        return {
            "path": str(WORK_ROOT),
            "mode": 0o700,
            "entries": sorted(first),
            "identity": _identity_row(after),
            "parent_identity": _identity_row(parent),
            "retained_parent_fd": True,
            "retained_root_fd": True,
        }

    def refresh_after_owned_entry_mutation(
        self, *, expected: set[str]
    ) -> dict[str, Any]:
        root, _parent = self._named_replay()
        self.root_anchor = root
        return self.entries(expected=expected)

    def request_authority(self) -> dict[str, Any]:
        if self.authority_value is not None:
            return dict(self.authority_value)
        value: dict[str, Any] = {
            "schema_version": (
                "bernini-action-preservation-decoded-eval-work-root-authority-v1"
            ),
            "path": str(WORK_ROOT),
            "parent_path": str(WORK_ROOT.parent),
            "creation_identity": _identity_row(self.creation_identity),
            "immutable_identity": _immutable_directory_identity(
                self.creation_identity
            ),
            "parent_immutable_identity": _immutable_directory_identity(
                self.parent_anchor
            ),
            "initial_entries": [],
            "retained_parent_fd_through_request_publication": True,
            "retained_root_fd_through_request_publication": True,
        }
        value["authority_digest"] = object_sha256(value)
        return value

    def close(self) -> None:
        if not self.closed:
            os.close(self.root_fd)
            os.close(self.parent_fd)
            self.closed = True


def stable_file(
    path: Path, *, label: str, expected_sha256: str,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    require(path.is_absolute() and path.resolve(strict=True) == path,
            f"{label} path differs")
    require(hasattr(os, "O_NOFOLLOW"), f"{label} no-follow unavailable")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(first).hexdigest()
    require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(named.st_mode)
        and before.st_nlink == 1
        and _identity(before) == _identity(middle) == _identity(after)
        and _identity(before) == _identity(named)
        and first == second and digest == expected_sha256
        and (expected_mode is None or stat.S_IMODE(before.st_mode) == expected_mode),
        f"{label} stable physical identity or bytes differ",
    )
    return first, {
        "path": str(path), "sha256": digest, "size": len(first),
        "mode": stat.S_IMODE(before.st_mode), "device": before.st_dev,
        "inode": before.st_ino, "uid": before.st_uid, "gid": before.st_gid,
        "nlink": before.st_nlink, "rdev": before.st_rdev,
        "blocks": getattr(before, "st_blocks", 0),
        "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
    }


def pair(value: Mapping[str, Any]) -> dict[str, str]:
    return {"path": str(value["path"]), "sha256": str(value["sha256"])}


def write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(path.is_absolute() and path.parent.is_dir(), "output path differs")
    require(not os.path.lexists(path), "output path is not fresh")
    raw = canonical_json_bytes(value) + b"\n"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    require(hasattr(os, "O_NOFOLLOW"), "no-follow creation unavailable")
    descriptor = os.open(path, flags | os.O_NOFOLLOW, 0o444)
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            require(count > 0, "write made no progress")
            offset += count
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
        require(
            before.st_nlink == 1 and stat.S_IMODE(before.st_mode) == 0o444
            and _identity(before) == _identity(middle) == _identity(after)
            and _identity(before) == _identity(named)
            and first == raw == second,
            "create-only same-FD replay differs",
        )
    finally:
        os.close(descriptor)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "mode": 0o444,
    }


def _write_create_only_at(
    authority: _PhaseAWorkAuthority,
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_entries_after: set[str],
) -> dict[str, Any]:
    require(
        path.is_absolute()
        and path.parent == WORK_ROOT
        and path.name not in ("", ".", "..")
        and os.path.sep not in path.name,
        "held-FD output path differs",
    )
    raw = canonical_json_bytes(value) + b"\n"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    require(hasattr(os, "O_NOFOLLOW"), "no-follow creation unavailable")
    descriptor = os.open(
        path.name,
        flags | os.O_NOFOLLOW,
        0o444,
        dir_fd=authority.root_fd,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            require(count > 0, "held-FD write made no progress")
            offset += count
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.fsync(authority.root_fd)
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(
            path.name,
            dir_fd=authority.root_fd,
            follow_symlinks=False,
        )
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) == 0o444
            and _identity(before) == _identity(middle) == _identity(after)
            and _identity(before) == _identity(named)
            and first == raw == second,
            "held-FD create-only same-FD replay differs",
        )
    finally:
        os.close(descriptor)
    authority.refresh_after_owned_entry_mutation(
        expected=expected_entries_after
    )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "mode": 0o444,
    }


def _controller_namespace() -> dict[str, Any]:
    raw, _binding = stable_file(
        CONTROLLER_PATH, label="detached controller",
        expected_sha256=CONTROLLER_SHA256, expected_mode=0o444,
    )
    namespace: dict[str, Any] = {
        "__name__": "_apv2_r7_detached_controller",
        "__file__": str(CONTROLLER_PATH), "__package__": None,
        "__spec__": None, "__builtins__": __builtins__,
    }
    exec(compile(raw, str(CONTROLLER_PATH), "exec", dont_inherit=True), namespace)
    required = {
        "ROOT_CONTROLLER_BOOTSTRAP_SOURCE", "validate_request",
        "load_deployment_receipt", "controller_bootstrap_argv",
    }
    require(required.issubset(namespace), "detached controller API differs")
    return namespace


def controller_prefix(namespace: Mapping[str, Any]) -> list[str]:
    bootstrap = namespace["ROOT_CONTROLLER_BOOTSTRAP_SOURCE"]
    require(type(bootstrap) is str and bootstrap, "controller bootstrap differs")
    return [
        "/usr/bin/env", "-i", "HOME=/vast/users/guangyi.chen",
        "USER=guangyi.chen", "LOGNAME=guangyi.chen", "PATH=/usr/bin:/bin",
        str(ROOT_PYTHON_PATH), "-I", "-S", "-B", "-c", bootstrap,
        str(CONTROLLER_PATH), CONTROLLER_SHA256,
    ]


def _validate_bundle_and_static_inputs() -> dict[str, Any]:
    _directory_entries(
        BUNDLE_ROOT,
        expected={
            ARTIFACT_ROOT.name,
            CONTROLLER_PATH.name,
            RUNTIME_SOURCE_PATH.name,
            SOURCE_PREPROCESSING_PATH.name,
            INPUT_AUTHORITY_PATH.name,
            PREPARE_PATH.name,
        },
        expected_mode=0o555,
        label="deployment bundle",
    )
    _directory_entries(
        ARTIFACT_ROOT,
        expected={
            "source.tar",
            "source.manifest.json",
            "deployment-envelope.json",
        },
        expected_mode=0o555,
        label="release artifact directory",
    )
    archive = stable_file(
        ARTIFACT_ROOT / "source.tar", label="r3 source archive",
        expected_sha256=ARCHIVE_SHA256, expected_mode=0o444,
    )[1]
    manifest = stable_file(
        ARTIFACT_ROOT / "source.manifest.json", label="r3 source manifest",
        expected_sha256=MANIFEST_SHA256, expected_mode=0o444,
    )[1]
    envelope = stable_file(
        ARTIFACT_ROOT / "deployment-envelope.json", label="r3 envelope",
        expected_sha256=ENVELOPE_SHA256, expected_mode=0o444,
    )[1]
    runtime_source = stable_file(
        RUNTIME_SOURCE_PATH, label="detached r3 runtime",
        expected_sha256=RUNTIME_SOURCE_SHA256, expected_mode=0o444,
    )[1]
    controller = stable_file(
        CONTROLLER_PATH, label="detached controller",
        expected_sha256=CONTROLLER_SHA256, expected_mode=0o444,
    )[1]
    stable_file(
        SOURCE_PREPROCESSING_PATH, label="source preprocessing authority",
        expected_sha256=SOURCE_PREPROCESSING_SHA256, expected_mode=0o444,
    )
    stable_file(
        INPUT_AUTHORITY_PATH, label="r7 input authority",
        expected_sha256=INPUT_AUTHORITY_SHA256, expected_mode=0o444,
    )
    root_python = stable_file(
        ROOT_PYTHON_PATH, label="root Python",
        expected_sha256=ROOT_PYTHON_SHA256, expected_mode=0o755,
    )[1]
    frozen_python = stable_file(
        FROZEN_PYTHON_PATH, label="frozen Python",
        expected_sha256=FROZEN_PYTHON_SHA256, expected_mode=0o755,
    )[1]
    torchrun = stable_file(
        TORCHRUN_PATH, label="torchrun source",
        expected_sha256=TORCHRUN_SHA256, expected_mode=0o644,
    )[1]
    return {
        "archive": archive, "manifest": manifest, "envelope": envelope,
        "runtime_source": runtime_source, "controller": controller,
        "root_python": root_python, "frozen_python": frozen_python,
        "torchrun": torchrun,
    }


def build_phase_a_request(
    *, work_authority: _PhaseAWorkAuthority,
) -> tuple[dict[str, Any], dict[str, Any]]:
    work_authority.entries(expected=set())
    work_root_authority = work_authority.request_authority()
    bindings = _validate_bundle_and_static_inputs()
    value: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA,
        "release_generation": RELEASE_GENERATION,
        "work_root_authority": work_root_authority,
        "controller": pair(bindings["controller"]),
        "root_python": pair(bindings["root_python"]),
        "frozen_python": pair(bindings["frozen_python"]),
        "site_packages_path": str(SITE_PACKAGES_PATH),
        "torchrun": pair(bindings["torchrun"]),
        "release_root": str(MATERIALIZED_RELEASE_ROOT),
        "archive": pair(bindings["archive"]),
        "manifest": pair(bindings["manifest"]),
        "manifest_digest": MANIFEST_DIGEST,
        "content_revision": CONTENT_REVISION,
        "envelope": pair(bindings["envelope"]),
        "envelope_digest": ENVELOPE_DIGEST,
        "verified_runtime_source": pair(bindings["runtime_source"]),
        "source_runtime_spec_path": str(SOURCE_SPEC_PATH),
        "source_spec_authority_receipt_path": str(SOURCE_SPEC_AUTHORITY_PATH),
        "controller_authority_receipt_path": str(CONTROLLER_AUTHORITY_PATH),
        "deployment_receipt_path": str(DEPLOYMENT_RECEIPT_PATH),
        "automatic_retry": False, "network_allowed": False,
        "scientific_promotion_authorized": False,
    }
    value["request_digest"] = object_sha256(value)
    namespace = _controller_namespace()
    try:
        validated = namespace["validate_request"](value)
    except Exception as error:
        raise R7DeploymentPreparationError(str(error)) from error
    require(validated == value, "controller request projection differs")
    return value, namespace


def publish_phase_a_request() -> dict[str, Any]:
    authority = _PhaseAWorkAuthority.create()
    try:
        initial = authority.entries(expected=set())
        value, namespace = build_phase_a_request(work_authority=authority)
        binding = _write_create_only_at(
            authority,
            DEPLOYMENT_REQUEST_PATH,
            value,
            expected_entries_after={DEPLOYMENT_REQUEST_PATH.name},
        )
        after_request = authority.entries(
            expected={DEPLOYMENT_REQUEST_PATH.name}
        )
        argv = controller_prefix(namespace) + [
            "capture-authority", "--deployment-request",
            str(DEPLOYMENT_REQUEST_PATH), "--deployment-request-sha256",
            binding["sha256"],
        ]
        return {
            "status": "R7_EXACT15_R3_PHASE_A_REQUEST_PREPARED_NOT_EXECUTED",
            "deployment_request": binding,
            "request_digest": value["request_digest"],
            "work_root_authority": value["work_root_authority"],
            "work_root_initial": initial,
            "work_root_after_request": after_request,
            "phase_a_expected_final_entries": sorted(
                {
                    DEPLOYMENT_REQUEST_PATH.name,
                    MATERIALIZED_RELEASE_ROOT.name,
                    CONTROLLER_AUTHORITY_PATH.name,
                    DEPLOYMENT_RECEIPT_PATH.name,
                }
            ),
            "controller_argv": argv,
            "controller_bootstrap_source_sha256": hashlib.sha256(
                namespace["ROOT_CONTROLLER_BOOTSTRAP_SOURCE"].encode("utf-8")
            ).hexdigest(),
            "remote_process_executed": False,
            "gpu_used": False,
        }
    finally:
        authority.close()


def validate_phase_a_completion(*, deployment_receipt_sha256: str) -> dict[str, Any]:
    expected_entries = {
        DEPLOYMENT_REQUEST_PATH.name,
        MATERIALIZED_RELEASE_ROOT.name,
        CONTROLLER_AUTHORITY_PATH.name,
        DEPLOYMENT_RECEIPT_PATH.name,
    }
    work_root = _directory_entries(WORK_ROOT, expected=expected_entries)
    namespace = _controller_namespace()
    try:
        deployment, _runtime = namespace["load_deployment_receipt"](
            DEPLOYMENT_RECEIPT_PATH,
            expected_sha256=sha256(
                deployment_receipt_sha256,
                label="deployment receipt literal SHA",
            ),
        )
    except Exception as error:
        raise R7DeploymentPreparationError(str(error)) from error
    request_raw, request_file = stable_file(
        DEPLOYMENT_REQUEST_PATH,
        label="phase-A deployment request",
        expected_sha256=deployment["deployment_request"]["sha256"],
        expected_mode=0o444,
    )
    request = strict_json(request_raw, label="phase-A deployment request")
    require(
        deployment["release_generation"] == RELEASE_GENERATION
        and deployment["work_root_authority"]
        == request["work_root_authority"]
        and deployment["work_root_expected_phase_a_entries"]
        == sorted(expected_entries)
        and deployment[
            "work_root_held_fd_through_controller_publication"
        ] is True
        and deployment["deployment_request"] == request_file
        and deployment["deployment_request_digest"] == request["request_digest"]
        and deployment["release"]["release_root"]["path"]
        == str(MATERIALIZED_RELEASE_ROOT)
        and deployment["source_runtime_spec_path"] == str(SOURCE_SPEC_PATH)
        and deployment["source_spec_authority_receipt_path"]
        == str(SOURCE_SPEC_AUTHORITY_PATH)
        and deployment["controller_authority"]["receipt"]["path"]
        == str(CONTROLLER_AUTHORITY_PATH),
        "phase-A completion continuity differs",
    )
    return {
        "status": "R7_EXACT15_R3_PHASE_A_COMPLETION_VERIFIED",
        "work_root": work_root,
        "deployment_request": request_file,
        "deployment_receipt": {
            "path": str(DEPLOYMENT_RECEIPT_PATH),
            "sha256": deployment_receipt_sha256,
        },
        "controller_authority": deployment["controller_authority"],
        "remote_process_executed_by_this_command": False,
        "gpu_used": False,
    }


def _load_source_preprocessing() -> tuple[dict[str, Any], dict[str, Any]]:
    raw, binding = stable_file(
        SOURCE_PREPROCESSING_PATH, label="source preprocessing authority",
        expected_sha256=SOURCE_PREPROCESSING_SHA256, expected_mode=0o444,
    )
    value = strict_json(raw, label="source preprocessing authority")
    unsigned = dict(value)
    claimed = unsigned.pop("authority_digest", None)
    require(
        value.get("schema_version") == SOURCE_PREPROCESSING_SCHEMA
        and claimed == object_sha256(unsigned)
        and value.get("source_video_bytes_consumed_directly") is True
        and value.get("precomputed_transformed_source_artifact_used") is False
        and value.get("training_loss_read_or_used_for_selection") is False
        and value.get("source_order")
        == ["7b88a1ca1f804f41", "841b5e0080a1441d",
            "a35b590961d24694", "a66e6818e4144928"],
        "source preprocessing authority differs",
    )
    for source in value.get("sources", []):
        stable_file(
            Path(source["source_video_path"]), label=f"source video {source['iid']}",
            expected_sha256=sha256(source["source_video_sha256"], label="video SHA"),
        )
        stable_file(
            Path(source["source_receipt_path"]), label=f"source receipt {source['iid']}",
            expected_sha256=sha256(source["source_receipt_sha256"], label="receipt SHA"),
        )
    return value, binding


def _phase_a_expected_entries() -> set[str]:
    return {
        DEPLOYMENT_REQUEST_PATH.name,
        MATERIALIZED_RELEASE_ROOT.name,
        CONTROLLER_AUTHORITY_PATH.name,
        DEPLOYMENT_RECEIPT_PATH.name,
    }


def _open_phase_b_context(
    *, deployment_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], _PhaseAWorkAuthority]:
    namespace = _controller_namespace()
    try:
        deployment, _runtime = namespace["load_deployment_receipt"](
            DEPLOYMENT_RECEIPT_PATH,
            expected_sha256=sha256(
                deployment_receipt_sha256, label="deployment receipt literal SHA"
            ),
        )
    except Exception as error:
        raise R7DeploymentPreparationError(str(error)) from error
    require(
        deployment["release_generation"] == RELEASE_GENERATION
        and deployment["work_root_expected_phase_a_entries"]
        == sorted(_phase_a_expected_entries())
        and deployment["source_runtime_spec_path"] == str(SOURCE_SPEC_PATH)
        and deployment["source_spec_authority_receipt_path"]
        == str(SOURCE_SPEC_AUTHORITY_PATH),
        "deployment receipt r7 path continuity differs",
    )
    authority = _PhaseAWorkAuthority.reopen(
        deployment["work_root_authority"]
    )
    try:
        authority.entries(expected=_phase_a_expected_entries())
    except Exception:
        authority.close()
        raise
    return deployment, namespace, authority


def _build_phase_b_spec_from_context(
    *,
    deployment: Mapping[str, Any],
    namespace: dict[str, Any],
    work_authority: _PhaseAWorkAuthority,
) -> tuple[dict[str, Any], dict[str, Any]]:
    work_authority.entries(expected=_phase_a_expected_entries())
    preprocessing, preprocessing_file = _load_source_preprocessing()
    source_manifest = stable_file(
        SOURCE_MANIFEST_PATH, label="source manifest",
        expected_sha256=SOURCE_MANIFEST_SHA256, expected_mode=0o444,
    )[1]
    adapter_manifest = stable_file(
        ADAPTER_RELEASE_MANIFEST_PATH, label="r7 adapter release manifest",
        expected_sha256=ADAPTER_RELEASE_MANIFEST_SHA256, expected_mode=0o444,
    )[1]
    model_manifest = stable_file(
        MODEL_MANIFEST_PATH, label="model release manifest",
        expected_sha256=MODEL_MANIFEST_SHA256, expected_mode=0o444,
    )[1]
    inference_config = stable_file(
        INFERENCE_CONFIG_PATH, label="inference config",
        expected_sha256=INFERENCE_CONFIG_SHA256, expected_mode=0o444,
    )[1]
    ffprobe = stable_file(
        FFPROBE_PATH, label="ffprobe",
        expected_sha256=FFPROBE_SHA256, expected_mode=0o755,
    )[1]
    release = deployment["release"]
    authority = deployment["controller_authority"]
    method_root = MATERIALIZED_RELEASE_ROOT / "methods/bernini_action_editing"
    infer = stable_file(
        method_root / "infer_lora.py", label="materialized infer_lora",
        expected_sha256=INFER_SHA256, expected_mode=0o444,
    )[1]
    decoder = stable_file(
        method_root / "action_preservation_decoded_eval_decoder_adapter_v1.py",
        label="materialized decoder", expected_sha256=DECODER_SHA256,
        expected_mode=0o555,
    )[1]
    pins = {
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "adapter_release_manifest_sha256": ADAPTER_RELEASE_MANIFEST_SHA256,
        "model_release_manifest_sha256": MODEL_MANIFEST_SHA256,
        "inference_source_sha256": INFER_SHA256,
        "inference_release_manifest_sha256": MANIFEST_SHA256,
        "inference_config_sha256": INFERENCE_CONFIG_SHA256,
        "source_preprocessing_sha256": SOURCE_PREPROCESSING_SHA256,
        "calibration_digest": None,
    }
    value: dict[str, Any] = {
        "schema_version": SOURCE_RUNTIME_SCHEMA,
        "pins": pins,
        "pin_files": {
            "source_manifest": pair(source_manifest),
            "adapter_release_manifest": pair(adapter_manifest),
            "model_release_manifest": pair(model_manifest),
            "inference_release_manifest": pair(release["manifest"]),
            "inference_config": pair(inference_config),
            "source_preprocessing": pair(preprocessing_file),
            "calibration": None,
        },
        "sources": preprocessing["sources"],
        "runtime": {
            "root_python": pair(deployment["root_python"]),
            "python": pair(deployment["frozen_python"]),
            "site_packages": deployment["site_packages"]["path"],
            "torchrun": pair(deployment["torchrun"]["source"]),
            "deployment_controller": pair(deployment["controller"]),
            "controller_authority": {
                "receipt": pair(authority["receipt"]),
                "authority_digest": authority["authority_digest"],
            },
            "infer_lora": pair(infer), "decoder_adapter": pair(decoder),
            "ffprobe": pair(ffprobe),
            "eval_release_root": release["release_root"]["path"],
            "eval_release_archive": pair(release["archive"]),
            "eval_release_envelope": pair(release["envelope"]),
            "eval_release_manifest_digest": release["manifest_digest"],
            "eval_release_content_revision": release["content_revision"],
            "eval_release_envelope_digest": release["envelope_digest"],
            "bernini_root": str(BERNINI_ROOT), "veomni_root": str(VEOMNI_ROOT),
            "model_checkpoint_root": str(MODEL_ROOT),
            "expected_bernini_commit": BERNINI_COMMIT,
            "expected_veomni_commit": VEOMNI_COMMIT,
            "expected_checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
            "method_source_revision": SOURCE_REVISION,
            "method_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "num_inference_steps": 40,
        },
    }
    value["spec_digest"] = object_sha256(value)
    work_authority.entries(expected=_phase_a_expected_entries())
    return value, namespace


def build_phase_b_spec(
    *, deployment_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deployment, namespace, authority = _open_phase_b_context(
        deployment_receipt_sha256=deployment_receipt_sha256
    )
    try:
        return _build_phase_b_spec_from_context(
            deployment=deployment,
            namespace=namespace,
            work_authority=authority,
        )
    finally:
        authority.close()


def publish_phase_b_spec(*, deployment_receipt_sha256: str) -> dict[str, Any]:
    deployment, namespace, authority = _open_phase_b_context(
        deployment_receipt_sha256=deployment_receipt_sha256
    )
    try:
        value, _ = _build_phase_b_spec_from_context(
            deployment=deployment,
            namespace=namespace,
            work_authority=authority,
        )
        expected_after = _phase_a_expected_entries() | {SOURCE_SPEC_PATH.name}
        binding = _write_create_only_at(
            authority,
            SOURCE_SPEC_PATH,
            value,
            expected_entries_after=expected_after,
        )
        after_spec = authority.entries(expected=expected_after)
        argv = controller_prefix(namespace) + [
            "capture-source-spec-authority", "--deployment-receipt",
            str(DEPLOYMENT_RECEIPT_PATH), "--deployment-receipt-sha256",
            deployment_receipt_sha256, "--source-runtime-spec",
            str(SOURCE_SPEC_PATH), "--source-runtime-spec-sha256",
            binding["sha256"],
        ]
        return {
            "status": "R7_EXACT15_R3_PHASE_B_SPEC_PREPARED_NOT_EXECUTED",
            "source_runtime_spec": binding,
            "spec_digest": value["spec_digest"],
            "source_preprocessing_sha256": SOURCE_PREPROCESSING_SHA256,
            "work_root_authority": deployment["work_root_authority"],
            "work_root_after_source_spec": after_spec,
            "phase_b_expected_final_entries": sorted(
                expected_after | {SOURCE_SPEC_AUTHORITY_PATH.name}
            ),
            "controller_argv": argv,
            "remote_process_executed": False,
            "gpu_used": False,
        }
    finally:
        authority.close()


def _run_target_prefix(
    *, deployment_receipt_sha256: str, target: str, capture_receipt: Path,
    source_spec_authority_sha256: str,
) -> list[str]:
    namespace = _controller_namespace()
    value = controller_prefix(namespace) + [
        "run-target", "--deployment-receipt", str(DEPLOYMENT_RECEIPT_PATH),
        "--deployment-receipt-sha256",
        sha256(deployment_receipt_sha256, label="deployment receipt literal SHA"),
        "--target", target, "--capture-receipt", str(capture_receipt),
    ]
    value += [
        "--source-spec-authority", str(SOURCE_SPEC_AUTHORITY_PATH),
        "--source-spec-authority-sha256",
        sha256(source_spec_authority_sha256,
               label="source spec authority literal SHA"),
    ]
    return value + ["--"]


def bridge_interface(
    *, deployment_receipt_sha256: str, source_spec_authority_sha256: str,
    source_runtime_spec_sha256: str,
) -> dict[str, Any]:
    capture = WORK_ROOT / "bridge.runtime-capture.json"
    argv = _run_target_prefix(
        deployment_receipt_sha256=deployment_receipt_sha256,
        target="action_preservation_decoded_eval_bridge_v1.py",
        capture_receipt=capture,
        source_spec_authority_sha256=source_spec_authority_sha256,
    ) + [
        "--experiment-root", str(EXPERIMENT_ROOT),
        "--training-complete-sha256", TRAINING_COMPLETE_SHA256,
        "--source-runtime-spec", str(SOURCE_SPEC_PATH),
        "--source-runtime-spec-sha256",
        sha256(source_runtime_spec_sha256, label="source spec literal SHA"),
        "--evaluation-id", EVALUATION_ID,
        "--evaluation-root", str(EVALUATION_ROOT),
        "--bridge-root", str(BRIDGE_ROOT),
    ]
    return {"stage": "bridge", "argv": argv, "capture_receipt": str(capture)}


def launcher_interface(
    *, deployment_receipt_sha256: str, source_spec_authority_sha256: str,
    physical_bindings_sha256: str,
) -> dict[str, Any]:
    method_root = MATERIALIZED_RELEASE_ROOT / "methods/bernini_action_editing"
    physical = BRIDGE_ROOT / "physical_bindings.json"
    capture = WORK_ROOT / "launcher.runtime-capture.json"
    argv = _run_target_prefix(
        deployment_receipt_sha256=deployment_receipt_sha256,
        target="action_preservation_decoded_eval_launcher_v1.py",
        capture_receipt=capture,
        source_spec_authority_sha256=source_spec_authority_sha256,
    ) + [
        "--evaluation-root", str(EVALUATION_ROOT),
        "--launch-root", str(LAUNCH_ROOT),
        "--python", str(FROZEN_PYTHON_PATH),
        "--python-sha256", FROZEN_PYTHON_SHA256,
        "--executor", str(method_root / "action_preservation_decoded_eval_executor_v2.py"),
        "--executor-sha256", EXECUTOR_SHA256,
        "--decoder-adapter",
        str(method_root / "action_preservation_decoded_eval_decoder_adapter_v1.py"),
        "--decoder-adapter-sha256", DECODER_SHA256,
        "--ffprobe", str(FFPROBE_PATH), "--ffprobe-sha256", FFPROBE_SHA256,
        "--physical-bindings", str(physical),
        "--physical-bindings-sha256",
        sha256(physical_bindings_sha256, label="physical bindings literal SHA"),
    ]
    return {"stage": "launcher", "argv": argv, "capture_receipt": str(capture)}


def aggregate_interface(
    *, deployment_receipt_sha256: str, source_spec_authority_sha256: str,
    physical_bindings_sha256: str,
) -> dict[str, Any]:
    physical = BRIDGE_ROOT / "physical_bindings.json"
    capture = AGGREGATE_ROOT.with_name(AGGREGATE_ROOT.name + ".runtime-capture.json")
    argv = _run_target_prefix(
        deployment_receipt_sha256=deployment_receipt_sha256,
        target="action_preservation_decoded_eval_aggregate_v2.py",
        capture_receipt=capture,
        source_spec_authority_sha256=source_spec_authority_sha256,
    ) + [
        "--evaluation-root", str(EVALUATION_ROOT),
        "--physical-bindings", str(physical),
        "--physical-bindings-sha256",
        sha256(physical_bindings_sha256, label="physical bindings literal SHA"),
        "--blinding-key-file", str(BLINDING_KEY_PATH),
        "--aggregate-root", str(AGGREGATE_ROOT),
    ]
    return {"stage": "aggregate", "argv": argv, "capture_receipt": str(capture)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("phase-a")
    phase_a_verify = commands.add_parser("phase-a-verify")
    phase_a_verify.add_argument("--deployment-receipt-sha256", required=True)
    phase_b = commands.add_parser("phase-b")
    phase_b.add_argument("--deployment-receipt-sha256", required=True)
    bridge = commands.add_parser("bridge-interface")
    bridge.add_argument("--deployment-receipt-sha256", required=True)
    bridge.add_argument("--source-spec-authority-sha256", required=True)
    bridge.add_argument("--source-runtime-spec-sha256", required=True)
    launcher = commands.add_parser("launcher-interface")
    launcher.add_argument("--deployment-receipt-sha256", required=True)
    launcher.add_argument("--source-spec-authority-sha256", required=True)
    launcher.add_argument("--physical-bindings-sha256", required=True)
    aggregate = commands.add_parser("aggregate-interface")
    aggregate.add_argument("--deployment-receipt-sha256", required=True)
    aggregate.add_argument("--source-spec-authority-sha256", required=True)
    aggregate.add_argument("--physical-bindings-sha256", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "phase-a":
        result = publish_phase_a_request()
    elif args.command == "phase-a-verify":
        result = validate_phase_a_completion(
            deployment_receipt_sha256=args.deployment_receipt_sha256
        )
    elif args.command == "phase-b":
        result = publish_phase_b_spec(
            deployment_receipt_sha256=args.deployment_receipt_sha256
        )
    elif args.command == "bridge-interface":
        result = bridge_interface(
            deployment_receipt_sha256=args.deployment_receipt_sha256,
            source_spec_authority_sha256=args.source_spec_authority_sha256,
            source_runtime_spec_sha256=args.source_runtime_spec_sha256,
        )
    elif args.command == "launcher-interface":
        result = launcher_interface(
            deployment_receipt_sha256=args.deployment_receipt_sha256,
            source_spec_authority_sha256=args.source_spec_authority_sha256,
            physical_bindings_sha256=args.physical_bindings_sha256,
        )
    else:
        result = aggregate_interface(
            deployment_receipt_sha256=args.deployment_receipt_sha256,
            source_spec_authority_sha256=args.source_spec_authority_sha256,
            physical_bindings_sha256=args.physical_bindings_sha256,
        )
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_ROOT", "CONTROLLER_AUTHORITY_PATH", "CONTROLLER_PATH",
    "DEPLOYMENT_RECEIPT_PATH", "DEPLOYMENT_REQUEST_PATH", "EVALUATION_ROOT",
    "MATERIALIZED_RELEASE_ROOT", "R7DeploymentPreparationError",
    "SOURCE_PREPROCESSING_PATH", "SOURCE_SPEC_AUTHORITY_PATH",
    "SOURCE_SPEC_PATH", "WORK_ROOT", "aggregate_interface",
    "bridge_interface", "build_phase_a_request", "build_phase_b_spec",
    "canonical_json_bytes",
    "launcher_interface", "object_sha256", "publish_phase_a_request",
    "publish_phase_b_spec", "validate_phase_a_completion",
    "write_create_only",
]
