#!/usr/bin/env python3
"""Independent evidence replay for the local-only v15c-r7 release.

This verifier is intentionally not an observer launcher.  It authenticates an
exact read-only snapshot, loads every validator from that snapshot, and then
recomputes the track, assignment, postflight, file-registry and rendered-media
claims from published bytes.  A successful call remains a mechanical local
audit only: it cannot authorize an observer run, localization, routing, decode,
or training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence


RELEASE_SCHEMA = "bernini-source-object-proposal-role-v15c-r7-local-release"
REPLAY_SCHEMA = "bernini-source-object-proposal-role-v15c-r7-evidence-replay"
RELEASE_TAG = "v15c-r7-local"
RELEASE_RELATIVE_PATH = (
    "methods/bernini_action_editing/assets/"
    "e00_source_sam2_proposal_role_probe_v15c_r7_release.json"
)
EXPECTED_CORE_MEMBER_COUNT = 8
EXPECTED_SNAPSHOT_FILE_COUNT = 9
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SNAPSHOT_POLICY = {
    "construction_phase": {
        "directory_mode": "0700",
        "member_mode": "0400",
        "receipt_semantics": "historical_observation_before_sealing",
    },
    "sealed_phase": {
        "directory_mode": "0500",
        "member_mode": "0400",
        "receipt_semantics": "current_state_reverified_at_runtime",
    },
    "core_member_count": EXPECTED_CORE_MEMBER_COUNT,
    "snapshot_file_count": EXPECTED_SNAPSHOT_FILE_COUNT,
    "exact_tree_required": True,
    "reject_extra_symlink_pyc": True,
    "single_link_regular_files_required": True,
}
RELEASE_KEYS = {
    "schema_version",
    "tag",
    "core_member_count",
    "snapshot_file_count",
    "members",
    "manifest_relative_path",
    "snapshot_policy",
    "observer_only",
    "observer_execution_authorized",
    "human_audit_action",
    "remote_gpu_status",
    "local_evidence_status",
    "route_status",
    "scientific_claim_authorized",
    "route_authorized",
    "decode_authorized",
    "training_authorized",
    "release_sha256",
}
MODULE_RELATIVES = {
    "materializer": (
        "methods/bernini_action_editing/"
        "materialize_source_sam2_proposal_tracks_v15c_r6.py"
    ),
    "core": (
        "methods/bernini_action_editing/"
        "source_object_proposal_role_probe_v15c.py"
    ),
    "runner": (
        "methods/bernini_action_editing/"
        "run_source_object_proposal_role_probe_v15c_r6.py"
    ),
    "postflight": (
        "methods/bernini_action_editing/"
        "postflight_source_sam2_proposal_role_probe_v15c_r6.py"
    ),
    "builder": (
        "methods/bernini_action_editing/tools/"
        "build_source_object_proposal_role_v15c_r6_review.py"
    ),
}
CONVENTIONAL_MODULE_NAMES = {
    "materializer": "materialize_source_sam2_proposal_tracks_v15c_r6",
    "core": "source_object_proposal_role_probe_v15c",
    "runner": "run_source_object_proposal_role_probe_v15c_r6",
    "postflight": "postflight_source_sam2_proposal_role_probe_v15c_r6",
    "builder": "v15c_r7_sealed_review_builder",
}
VIDEO_KEYS = ("source", "all", "old_actor", "new_actor", "recipient")
DISPLAY_FRAMES = [0, 20, 40, 60, 80]
EXPECTED_MEDIA_CONTRACT = {
    "source": {
        "video": "media/source.mp4",
        "contact_sheet": "media/source_f00_20_40_60_80.jpg",
    },
    "all": {
        "video": "media/all_proposals.mp4",
        "contact_sheet": "media/all_proposals_f00_20_40_60_80.jpg",
    },
    "old_actor": {
        "video": "media/old_actor.mp4",
        "contact_sheet": "media/old_actor_f00_20_40_60_80.jpg",
    },
    "new_actor": {
        "video": "media/new_actor.mp4",
        "contact_sheet": "media/new_actor_f00_20_40_60_80.jpg",
    },
    "recipient": {
        "video": "media/recipient.mp4",
        "contact_sheet": "media/recipient_f00_20_40_60_80.jpg",
    },
}


class FinalizeV15CR7Error(RuntimeError):
    """A release, tensor, replay, registry, or media-content gate differs."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise FinalizeV15CR7Error(f"{label} is not lowercase SHA256")
    return value


def require_exact_keys(value: Any, keys: Sequence[str], label: str) -> None:
    if type(value) is not dict or set(value) != set(keys):
        raise FinalizeV15CR7Error(f"{label} exact keys differ")


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise FinalizeV15CR7Error(f"{label} is absent") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FinalizeV15CR7Error(f"{label} is not one regular non-symlink file")
    return info


def file_sha256(path: Path) -> str:
    _regular(path, str(path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    current = _regular(path, str(path))
    if (
        identity_before != identity_after
        or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
    ):
        raise FinalizeV15CR7Error("file changed while hashing")
    return digest.hexdigest()


def read_stable_bytes(path: Path, label: str) -> bytes:
    _regular(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = _regular(path, label)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
        raise FinalizeV15CR7Error(f"{label} changed while reading")
    return b"".join(chunks)


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(read_stable_bytes(path, "JSON input").decode("utf-8"))
    except Exception as error:
        raise FinalizeV15CR7Error("JSON input differs") from error
    if type(value) is not dict:
        raise FinalizeV15CR7Error("JSON input is not one object")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    payload = dict(value)
    claimed = require_sha(payload.pop(field, None), field)
    if claimed != object_sha256(payload):
        raise FinalizeV15CR7Error(f"{field} self-hash differs")


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise FinalizeV15CR7Error(f"{label} path differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or value.endswith(".pyc")
        or "__pycache__" in path.parts
    ):
        raise FinalizeV15CR7Error(f"{label} path differs")
    return value


def _verify_manifest(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    require_exact_keys(value, RELEASE_KEYS, "release")
    verify_self_hash(value, "release_sha256")
    members = value.get("members")
    if (
        value.get("schema_version") != RELEASE_SCHEMA
        or value.get("tag") != RELEASE_TAG
        or value.get("core_member_count") != EXPECTED_CORE_MEMBER_COUNT
        or value.get("snapshot_file_count") != EXPECTED_SNAPSHOT_FILE_COUNT
        or type(members) is not list
        or len(members) != EXPECTED_CORE_MEMBER_COUNT
        or value.get("manifest_relative_path") != RELEASE_RELATIVE_PATH
        or value.get("snapshot_policy") != SNAPSHOT_POLICY
        or value.get("observer_only") is not True
        or value.get("observer_execution_authorized") is not False
        or value.get("human_audit_action") != "reject_only"
        or value.get("remote_gpu_status") != "REMOTE_GPU_UNAUDITED"
        or value.get("local_evidence_status") != "LOCAL_RELEASE_UNAUDITED"
        or value.get("route_status") != "ROUTE_NO_GO"
        or value.get("scientific_claim_authorized") is not False
        or value.get("route_authorized") is not False
        or value.get("decode_authorized") is not False
        or value.get("training_authorized") is not False
    ):
        raise FinalizeV15CR7Error("release authority differs")
    normalized = []
    paths = []
    for row in members:
        require_exact_keys(row, {"path", "sha256", "size"}, "release member")
        relative = _relative(row.get("path"), "release member")
        digest = require_sha(row.get("sha256"), "release member hash")
        size = row.get("size")
        if type(size) is not int or size <= 0:
            raise FinalizeV15CR7Error("release member size differs")
        normalized.append({"path": relative, "sha256": digest, "size": size})
        paths.append(relative)
    if paths != sorted(paths) or len(set(paths)) != EXPECTED_CORE_MEMBER_COUNT:
        raise FinalizeV15CR7Error("release member registry differs")
    required_modules = set(MODULE_RELATIVES.values())
    required_modules.add(
        "methods/bernini_action_editing/"
        "finalize_source_sam2_proposal_role_probe_v15c_r7.py"
    )
    required_modules.add(
        "methods/bernini_action_editing/tools/v15c_r7_external_bootstrap.py"
    )
    required_modules.add(
        "methods/bernini_action_editing/assets/"
        "e00_source_sam2_proposal_role_probe_v15c_r6.json"
    )
    if set(paths) != required_modules:
        raise FinalizeV15CR7Error("release does not pin the exact r7 closure")
    return tuple(normalized)


def verify_release(
    root: Path, release_path: Path, expected_release_sha256: str
) -> Mapping[str, Any]:
    expected = require_sha(expected_release_sha256, "external release hash")
    root = root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise FinalizeV15CR7Error("release root differs")
    canonical = root / RELEASE_RELATIVE_PATH
    if release_path.absolute() != canonical.absolute():
        raise FinalizeV15CR7Error("release placement differs")
    if file_sha256(canonical) != expected:
        raise FinalizeV15CR7Error("release file hash differs")
    manifest = read_json(canonical)
    members = _verify_manifest(manifest)
    for row in members:
        path = root / row["path"]
        info = _regular(path, row["path"])
        if info.st_size != row["size"] or file_sha256(path) != row["sha256"]:
            raise FinalizeV15CR7Error("release member bytes differ")
    return manifest


def _scan_tree(root: Path) -> tuple[set[str], set[str]]:
    files = set()
    directories = {"."}
    for current_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        for name in directory_names:
            path = current / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise FinalizeV15CR7Error("snapshot contains a symlink/non-directory")
            if name == "__pycache__":
                raise FinalizeV15CR7Error("snapshot contains __pycache__")
            directories.add(relative)
        for name in file_names:
            path = current / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or name.endswith(".pyc")
                or info.st_nlink != 1
            ):
                raise FinalizeV15CR7Error(
                    "snapshot contains symlink/non-file/pyc/hardlink"
                )
            files.add(relative)
    return files, directories


def verify_snapshot(
    snapshot: Path, manifest: Mapping[str, Any], expected_release_sha256: str
) -> Mapping[str, Mapping[str, Any]]:
    snapshot = snapshot.absolute()
    expected_files = {
        RELEASE_RELATIVE_PATH,
        *(row["path"] for row in manifest["members"]),
    }
    expected_directories = {"."}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    files, directories = _scan_tree(snapshot)
    if files != expected_files or directories != expected_directories:
        raise FinalizeV15CR7Error("snapshot exact tree differs")
    rows = {
        row["path"]: {"sha256": row["sha256"], "size": row["size"]}
        for row in manifest["members"]
    }
    release_file = snapshot / RELEASE_RELATIVE_PATH
    rows[RELEASE_RELATIVE_PATH] = {
        "sha256": require_sha(expected_release_sha256, "snapshot release hash"),
        "size": _regular(release_file, "snapshot release").st_size,
    }
    observed = {}
    for relative in sorted(expected_files):
        path = snapshot / relative
        info = _regular(path, relative)
        digest = file_sha256(path)
        if (
            digest != rows[relative]["sha256"]
            or info.st_size != rows[relative]["size"]
            or stat.S_IMODE(info.st_mode) != 0o400
            or info.st_nlink != 1
        ):
            raise FinalizeV15CR7Error("snapshot member bytes/mode/link-count differ")
        observed[relative] = {"sha256": digest, "size": info.st_size}
    for relative in expected_directories:
        path = snapshot if relative == "." else snapshot / relative
        if stat.S_IMODE(path.lstat().st_mode) != 0o500:
            raise FinalizeV15CR7Error("snapshot directory mode differs")
    return observed


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FinalizeV15CR7Error("sealed module spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    try:
        loaded = Path(module.__file__).resolve(strict=True)
    except Exception as error:
        raise FinalizeV15CR7Error("sealed module path is unavailable") from error
    if loaded != path.resolve(strict=True):
        raise FinalizeV15CR7Error("sealed module resolved outside the snapshot")
    return module


def load_sealed_modules(
    snapshot: Path, manifest: Mapping[str, Any]
) -> SimpleNamespace:
    """Load validators only after the caller authenticated the sealed tree."""

    member_rows = {row["path"]: row for row in manifest["members"]}
    for relative in MODULE_RELATIVES.values():
        if relative not in member_rows:
            raise FinalizeV15CR7Error("sealed validator is absent from the release")
    method_root = snapshot / "methods/bernini_action_editing"
    inserted = str(method_root)
    sys.path.insert(0, inserted)
    loaded = {}
    try:
        for key in ("materializer", "core", "runner", "postflight", "builder"):
            name = CONVENTIONAL_MODULE_NAMES[key]
            previous = sys.modules.get(name)
            if previous is not None:
                previous_file = getattr(previous, "__file__", None)
                if previous_file is None or Path(previous_file).resolve() != (
                    snapshot / MODULE_RELATIVES[key]
                ).resolve():
                    del sys.modules[name]
            loaded[key] = _load_module(name, snapshot / MODULE_RELATIVES[key])
    finally:
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
        elif inserted in sys.path:
            sys.path.remove(inserted)
    return SimpleNamespace(**loaded)


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise FinalizeV15CR7Error("safetensors header has a duplicate key")
        result[key] = value
    return result


def strict_safetensors(
    path: Path,
    expected: Mapping[str, tuple[str, tuple[int, ...]]],
    *,
    expected_file_sha256: str,
    expected_array_sha256: Optional[Mapping[str, str]] = None,
    expected_metadata: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    """Parse and hash actual tensor payloads without trusting a receipt."""

    import numpy as np

    info = _regular(path, "safetensors input")
    if info.st_size <= 8 or file_sha256(path) != require_sha(
        expected_file_sha256, "safetensors file hash"
    ):
        raise FinalizeV15CR7Error("safetensors file bytes differ")
    raw = read_stable_bytes(path, "safetensors input")
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise FinalizeV15CR7Error("safetensors bytes changed after authentication")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if header_length <= 1 or header_length > 16 * 1024 * 1024:
        raise FinalizeV15CR7Error("safetensors header length differs")
    data_start = 8 + header_length
    if data_start >= len(raw):
        raise FinalizeV15CR7Error("safetensors payload is empty")
    try:
        header = json.loads(
            raw[8:data_start].decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
        )
    except FinalizeV15CR7Error:
        raise
    except Exception as error:
        raise FinalizeV15CR7Error("safetensors header differs") from error
    if type(header) is not dict:
        raise FinalizeV15CR7Error("safetensors header is not one object")
    metadata = header.pop("__metadata__", None)
    if expected_metadata is not None and metadata != dict(expected_metadata):
        raise FinalizeV15CR7Error("safetensors metadata differs")
    if set(header) != set(expected):
        raise FinalizeV15CR7Error("safetensors key registry differs")
    dtype_map = {
        "F32": np.dtype("<f4"),
        "F64": np.dtype("<f8"),
        "I64": np.dtype("<i8"),
        "I32": np.dtype("<i4"),
        "U8": np.dtype("u1"),
        "I8": np.dtype("i1"),
        "BOOL": np.dtype("bool"),
    }
    rows = {}
    intervals = []
    expected_hashes = dict(expected_array_sha256 or {})
    if expected_array_sha256 is not None and set(expected_hashes) != set(expected):
        raise FinalizeV15CR7Error("tensor array-hash registry differs")
    for key in sorted(header):
        row = header[key]
        require_exact_keys(row, {"dtype", "shape", "data_offsets"}, f"tensor {key}")
        dtype_name = row.get("dtype")
        shape = row.get("shape")
        offsets = row.get("data_offsets")
        expected_dtype, expected_shape = expected[key]
        if (
            dtype_name != expected_dtype
            or dtype_name not in dtype_map
            or type(shape) is not list
            or tuple(shape) != tuple(expected_shape)
            or any(type(item) is not int or item <= 0 for item in shape)
            or type(offsets) is not list
            or len(offsets) != 2
            or any(type(item) is not int or item < 0 for item in offsets)
            or offsets[0] >= offsets[1]
        ):
            raise FinalizeV15CR7Error(f"tensor {key} descriptor differs")
        element_count = math.prod(shape)
        byte_count = element_count * dtype_map[dtype_name].itemsize
        if offsets[1] - offsets[0] != byte_count:
            raise FinalizeV15CR7Error(f"tensor {key} byte extent differs")
        begin = data_start + offsets[0]
        end = data_start + offsets[1]
        if end > len(raw):
            raise FinalizeV15CR7Error(f"tensor {key} exceeds the file")
        array = np.frombuffer(raw, dtype=dtype_map[dtype_name], count=element_count, offset=begin)
        array = array.reshape(tuple(shape))
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            raise FinalizeV15CR7Error(f"tensor {key} is non-finite")
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"|")
        digest.update(",".join(str(int(item)) for item in array.shape).encode("ascii"))
        digest.update(b"|")
        digest.update(array.tobytes(order="C"))
        array_digest = digest.hexdigest()
        if expected_array_sha256 is not None and array_digest != require_sha(
            expected_hashes[key], f"tensor {key} array hash"
        ):
            raise FinalizeV15CR7Error(f"tensor {key} content hash differs")
        intervals.append((offsets[0], offsets[1], key))
        rows[key] = {
            "dtype": dtype_name,
            "shape": list(shape),
            "array_sha256": array_digest,
            "finite": True,
        }
    intervals.sort()
    cursor = 0
    for begin, end, _ in intervals:
        if begin != cursor:
            raise FinalizeV15CR7Error("safetensors data extents have a gap/overlap")
        cursor = end
    if data_start + cursor != len(raw):
        raise FinalizeV15CR7Error("safetensors has trailing/unregistered bytes")
    return {
        "file_sha256": file_sha256(path),
        "size": len(raw),
        "header_length": header_length,
        "metadata": metadata,
        "tensors": rows,
    }


def _r6_tensor_contract(core: Any) -> Mapping[str, tuple[str, tuple[int, ...]]]:
    contract = {
        "aggregate_affinity": ("F32", (5, 21, 37, 25)),
        "aggregate_legacy_null_affinity": ("F32", (21, 37, 25)),
        "aggregate_null_span_affinity": ("F32", (64, 21, 37, 25)),
        "aggregate_shuffled_affinity": ("F32", (5, 21, 37, 25)),
    }
    for block in core.BLOCK_INDICES:
        prefix = f"block_{block:02d}"
        contract[f"{prefix}_affinity"] = ("F32", (5, 21, 37, 25))
        contract[f"{prefix}_legacy_null_affinity"] = ("F32", (21, 37, 25))
        contract[f"{prefix}_null_span_affinity"] = ("F32", (64, 21, 37, 25))
        contract[f"{prefix}_shuffled_affinity"] = ("F32", (5, 21, 37, 25))
    contract.update(
        {
            "calibration_standardized_role_maps": (
                "F32",
                (5, 5, 21, 37, 25),
            ),
            "calibration_exploratory_track_masks_u8": (
                "U8",
                (5, 5, 21, 37, 25),
            ),
            "calibration_strict_aggregate_masks_u8": (
                "U8",
                (5, 21, 37, 25),
            ),
            "calibration_strict_block_masks_u8": (
                "U8",
                (5, 5, 21, 37, 25),
            ),
        }
    )
    return contract


def _expected_result(
    *,
    modules: SimpleNamespace,
    spec: Mapping[str, Any],
    r6_receipt: Mapping[str, Any],
    track_receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[Mapping[str, Any], str]:
    tracks, reopened = modules.core.load_tracks_for_v15c(
        paths["track_receipt"], paths["track_tensors"]
    )
    if reopened != track_receipt:
        raise FinalizeV15CR7Error("track loader receipt differs")
    affinity = modules.core.load_r6_affinity_for_v15c(paths["r6_tensors"])
    replay = dict(
        modules.core.run_source_object_proposal_role_probe_v15c(
            tracks=tracks,
            affinity=affinity,
            thresholds=modules.runner.thresholds_from_spec(spec),
        )
    )
    core_receipt = require_sha(
        replay.pop("receipt_sha256", None), "assignment core receipt"
    )
    replay["provenance"] = {
        "spec_raw_sha256": file_sha256(paths["spec"]),
        "spec_canonical_sha256": modules.core.object_sha256(spec),
        "source_video_sha256": spec["source"]["sha256"],
        "source_text_provenance_sha256": spec["r6"][
            "source_text_provenance_sha256"
        ],
        "r6_receipt_file_sha256": file_sha256(paths["r6_receipt"]),
        "r6_affinity_file_sha256": file_sha256(paths["r6_tensors"]),
        "r6_internal_receipt_sha256": r6_receipt["receipt_sha256"],
        "track_receipt_file_sha256": file_sha256(paths["track_receipt"]),
        "track_tensor_file_sha256": file_sha256(paths["track_tensors"]),
        "track_internal_receipt_sha256": track_receipt["receipt_sha256"],
        "track_output_manifest_file_sha256": file_sha256(
            paths["track_receipt"].parent / "output_manifest.json"
        ),
        "assignment_core_receipt_sha256": core_receipt,
    }
    replay["route_authorized"] = False
    replay["training_authorized"] = False
    replay["decode_authorized"] = False
    replay["receipt_sha256"] = modules.core.object_sha256(replay)
    return replay, core_receipt


def require_non_dummy_track_receipt(
    receipt: Mapping[str, Any], modules: SimpleNamespace
) -> None:
    """Reject self-hashed placeholders before any downstream replay."""

    require_exact_keys(receipt, modules.materializer.TRACK_RECEIPT_KEYS, "track receipt")
    verify_self_hash(receipt, "receipt_sha256")
    proposals = receipt.get("proposals")
    if (
        receipt.get("schema_version") != modules.core.TRACK_SCHEMA_VERSION
        or type(proposals) is not list
        or not 1 <= len(proposals) <= modules.core.MAXIMUM_PROPOSAL_COUNT
        or receipt.get("proposal_count") != len(proposals)
        or not receipt.get("repeat_transcripts")
        or not receipt.get("repeat")
    ):
        raise FinalizeV15CR7Error("track receipt is empty/dummy")
    for field in (
        "phase_coverage_tensor_sha256",
        "phase_coverage_array_sha256",
        "artifact_manifest_file_sha256",
        "artifact_manifest_internal_sha256",
    ):
        require_sha(receipt.get(field), f"track receipt {field}")


def verify_track_output_manifest_nonempty(
    root: Path, modules: SimpleNamespace
) -> Mapping[str, Any]:
    """Reopen the exact non-empty materializer output file registry."""

    manifest_path = root / "output_manifest.json"
    manifest = read_json(manifest_path)
    require_exact_keys(
        manifest,
        {
            "schema_version",
            "files",
            "route_authorized",
            "training_authorized",
            "manifest_sha256",
        },
        "track output manifest",
    )
    verify_self_hash(manifest, "manifest_sha256")
    files = manifest.get("files")
    required = {
        "artifact_manifest.json",
        "phase_coverage.safetensors",
        "track_receipt.json",
    }
    observed, _ = _scan_tree(root)
    observed.discard("output_manifest.json")
    if (
        manifest.get("schema_version") != modules.materializer.OUTPUT_MANIFEST_SCHEMA
        or type(files) is not dict
        or not files
        or list(files) != sorted(files)
        or not required.issubset(files)
        or set(files) != observed
        or manifest.get("route_authorized") is not False
        or manifest.get("training_authorized") is not False
    ):
        raise FinalizeV15CR7Error("track output manifest is empty/dummy/inexact")
    for relative, row in files.items():
        require_exact_keys(row, {"sha256", "size"}, "track output member")
        path = root / _relative(relative, "track output member")
        info = _regular(path, relative)
        if (
            info.st_size <= 0
            or row.get("size") != info.st_size
            or require_sha(row.get("sha256"), "track output member hash")
            != file_sha256(path)
        ):
            raise FinalizeV15CR7Error("track output manifest member differs")
    return manifest


def _expected_postflight(
    *,
    modules: SimpleNamespace,
    result: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    gates = {key: True for key in modules.postflight.GATE_KEYS}
    modules.runner.require_exact_keys(
        gates, modules.postflight.GATE_KEYS, "replayed postflight gates"
    )
    receipt = {
        "schema_version": modules.postflight.SCHEMA,
        "status": "POSTFLIGHT_PASS_REJECT_ONLY_OVERLAY_PENDING",
        "gates": gates,
        "file_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "mechanical_candidate_qualified": result.get(
            "mechanical_candidate_qualified"
        ),
        "assignments_for_reject_only_audit": result.get("assignments"),
        "human_audit_action": "reject_only",
        "human_audit_may_authorize_route": False,
        "localization_semantically_certified": False,
        "action_success_certified": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
        "optimizer_updates": 0,
        "renderer_forward_calls": 0,
    }
    receipt["receipt_sha256"] = modules.core.object_sha256(receipt)
    modules.runner.require_exact_keys(
        receipt, modules.postflight.POSTFLIGHT_KEYS, "replayed postflight"
    )
    return receipt


def replay_observer_evidence(
    *,
    modules: SimpleNamespace,
    spec_path: Path,
    source: Path,
    r6_receipt_path: Path,
    r6_tensors: Path,
    run_root: Path,
) -> Mapping[str, Any]:
    """Recompute every non-media observer claim from its actual bytes."""

    run_root = run_root.absolute()
    if run_root.is_symlink() or not run_root.is_dir():
        raise FinalizeV15CR7Error("observer run root differs")
    run_root = run_root.resolve(strict=True)
    paths = {
        "spec": spec_path.resolve(strict=True),
        "source": source.resolve(strict=True),
        "r6_receipt": r6_receipt_path.resolve(strict=True),
        "r6_tensors": r6_tensors.resolve(strict=True),
        "track_receipt": (run_root / "tracks/track_receipt.json").resolve(
            strict=True
        ),
        "track_tensors": (run_root / "tracks/phase_coverage.safetensors").resolve(
            strict=True
        ),
        "result": (run_root / "result.json").resolve(strict=True),
    }
    spec = modules.materializer.read_spec(paths["spec"])
    if file_sha256(paths["source"]) != spec["source"]["sha256"]:
        raise FinalizeV15CR7Error("source bytes differ from the sealed spec")
    r6_receipt = read_json(paths["r6_receipt"])
    verify_self_hash(r6_receipt, "receipt_sha256")
    if (
        file_sha256(paths["r6_receipt"])
        != spec["r6"]["probe_receipt_file_sha256"]
        or r6_receipt.get("receipt_sha256")
        != spec["r6"]["probe_receipt_internal_sha256"]
    ):
        raise FinalizeV15CR7Error("r6 receipt file/internal binding differs")
    track_receipt = read_json(paths["track_receipt"])
    require_non_dummy_track_receipt(track_receipt, modules)
    output_manifest = verify_track_output_manifest_nonempty(
        paths["track_receipt"].parent, modules
    )
    try:
        modules.runner.validate_r6_receipt(r6_receipt, spec, paths["r6_tensors"])
        modules.runner.validate_track_bundle(
            track_receipt,
            spec,
            paths["track_receipt"],
            paths["track_tensors"],
        )
    except Exception as error:
        raise FinalizeV15CR7Error("sealed track-bundle replay failed") from error

    proposals = track_receipt["proposals"]
    track_expected = {
        "phase_coverage": (
            "F32",
            (
                len(proposals),
                len(modules.core.PHASE_FRAMES),
                modules.core.GRID_HEIGHT,
                modules.core.GRID_WIDTH,
            ),
        )
    }
    track_tensor_receipt = strict_safetensors(
        paths["track_tensors"],
        track_expected,
        expected_file_sha256=track_receipt["phase_coverage_tensor_sha256"],
        expected_array_sha256={
            "phase_coverage": track_receipt["phase_coverage_array_sha256"]
        },
    )
    diagnostics = r6_receipt.get("diagnostics")
    if type(diagnostics) is not dict:
        raise FinalizeV15CR7Error("r6 diagnostics receipt differs")
    r6_tensor_receipt = strict_safetensors(
        paths["r6_tensors"],
        _r6_tensor_contract(modules.core),
        expected_file_sha256=spec["r6"]["affinity_tensor_file_sha256"],
        expected_array_sha256=diagnostics.get("tensor_sha256"),
        expected_metadata={
            "metadata_sha256": diagnostics.get("metadata_sha256"),
            "schema_version": "bernini-source-owned-instance-role-null64-affinity-v15b-r6",
        },
    )
    expected_result, core_receipt = _expected_result(
        modules=modules,
        spec=spec,
        r6_receipt=r6_receipt,
        track_receipt=track_receipt,
        paths=paths,
    )
    published_result = read_json(paths["result"])
    verify_self_hash(published_result, "receipt_sha256")
    if published_result != expected_result:
        raise FinalizeV15CR7Error("runner result differs from core assignment replay")
    postflight_path = (run_root / "postflight.json").resolve(strict=True)
    postflight_paths = dict(paths)
    postflight_paths["result"] = paths["result"]
    expected_postflight = _expected_postflight(
        modules=modules,
        result=published_result,
        paths=postflight_paths,
    )
    published_postflight = read_json(postflight_path)
    verify_self_hash(published_postflight, "receipt_sha256")
    if published_postflight != expected_postflight:
        raise FinalizeV15CR7Error(
            "postflight gates/file registry differ from independent replay"
        )
    return {
        "spec": spec,
        "track_receipt": track_receipt,
        "result": published_result,
        "postflight": published_postflight,
        "track_safetensors": track_tensor_receipt,
        "r6_safetensors": r6_tensor_receipt,
        "assignment_core_receipt_sha256": core_receipt,
        "output_manifest_file_sha256": file_sha256(
            run_root / "tracks/output_manifest.json"
        ),
        "output_manifest_internal_sha256": output_manifest["manifest_sha256"],
    }


def _decode_video(path: Path) -> tuple[list[Any], float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FinalizeV15CR7Error("video reopen failed")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if getattr(frame, "shape", None) is None or tuple(frame.shape) != (
                1056,
                704,
                3,
            ):
                raise FinalizeV15CR7Error("decoded video frame differs")
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) != 81 or abs(fps - 25.0) > 1.0e-6:
        raise FinalizeV15CR7Error("decoded video geometry differs")
    return frames, fps


def _frames_sha256(frames: Sequence[Any]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for index, frame in enumerate(frames):
        value = np.ascontiguousarray(frame)
        digest.update(struct.pack("<I", index))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"|")
        digest.update(",".join(str(int(item)) for item in value.shape).encode("ascii"))
        digest.update(b"|")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _compare_video_content(left: Path, right: Path, label: str) -> str:
    import numpy as np

    left_frames, left_fps = _decode_video(left)
    right_frames, right_fps = _decode_video(right)
    if left_fps != right_fps or len(left_frames) != len(right_frames):
        raise FinalizeV15CR7Error(f"{label} video timing differs")
    for index, (left_frame, right_frame) in enumerate(zip(left_frames, right_frames)):
        if not np.array_equal(left_frame, right_frame):
            raise FinalizeV15CR7Error(
                f"{label} decoded content differs at frame {index}"
            )
    return _frames_sha256(left_frames)


def _compare_sheet_content(left: Path, right: Path, label: str) -> str:
    import cv2
    import numpy as np

    left_image = cv2.imread(str(left), cv2.IMREAD_COLOR)
    right_image = cv2.imread(str(right), cv2.IMREAD_COLOR)
    if (
        not isinstance(left_image, np.ndarray)
        or not isinstance(right_image, np.ndarray)
        or left_image.shape != (528, 1760, 3)
        or right_image.shape != left_image.shape
        or not np.array_equal(left_image, right_image)
    ):
        raise FinalizeV15CR7Error(f"{label} contact-sheet content differs")
    return hashlib.sha256(np.ascontiguousarray(left_image).tobytes(order="C")).hexdigest()


def _validate_review_receipts(
    *,
    modules: SimpleNamespace,
    run_root: Path,
    source: Path,
    track_receipt_path: Path,
    result_path: Path,
    postflight_path: Path,
) -> Mapping[str, Any]:
    review = run_root / "review"
    overlay_path = review / "overlay_receipt.json"
    media_path = review / "media_validation.json"
    overlay = read_json(overlay_path)
    verify_self_hash(overlay, "receipt_sha256")
    modules.runner.require_exact_keys(
        overlay, modules.builder.OVERLAY_RECEIPT_KEYS, "overlay receipt"
    )
    if modules.builder.MEDIA_CONTRACT != EXPECTED_MEDIA_CONTRACT:
        raise FinalizeV15CR7Error("sealed builder media contract differs")
    expected_files = {
        "index.html",
        "media_validation.json",
        *(
            relative
            for row in EXPECTED_MEDIA_CONTRACT.values()
            for relative in row.values()
        ),
    }
    files = overlay.get("files")
    if (
        overlay.get("schema_version") != modules.builder.SCHEMA
        or overlay.get("status") != "SYNCHRONIZED_REJECT_ONLY_OVERLAY_COMPLETE"
        or type(files) is not dict
        or set(files) != expected_files
        or overlay.get("media_contract") != EXPECTED_MEDIA_CONTRACT
        or overlay.get("display_frames") != DISPLAY_FRAMES
        or overlay.get("all_role_contact_sheets_present") is not True
        or overlay.get("all_unassigned_rows_include_full_failure_evidence") is not True
        or overlay.get("synchronized_playback") is not True
        or overlay.get("human_audit_action") != "reject_only"
        or overlay.get("approve_action_available") is not False
        or overlay.get("threshold_mutation_available") is not False
        or overlay.get("localization_semantically_certified") is not False
        or overlay.get("route_authorized") is not False
        or overlay.get("decode_authorized") is not False
        or overlay.get("training_authorized") is not False
    ):
        raise FinalizeV15CR7Error("overlay authority/registry differs")
    observed_files, _ = _scan_tree(review)
    if observed_files != expected_files | {"overlay_receipt.json"}:
        raise FinalizeV15CR7Error("review exact tree differs")
    for relative in sorted(expected_files):
        row = files[relative]
        require_exact_keys(row, {"sha256", "size"}, "overlay file")
        path = review / _relative(relative, "overlay file")
        info = _regular(path, relative)
        if (
            require_sha(row.get("sha256"), "overlay file hash") != file_sha256(path)
            or row.get("size") != info.st_size
            or info.st_size <= 0
        ):
            raise FinalizeV15CR7Error("overlay file bytes differ")
    expected_inputs = {
        "source_sha256": file_sha256(source),
        "track_receipt_sha256": file_sha256(track_receipt_path),
        "result_sha256": file_sha256(result_path),
        "postflight_sha256": file_sha256(postflight_path),
    }
    if overlay.get("inputs") != expected_inputs:
        raise FinalizeV15CR7Error("overlay input binding differs")
    media = read_json(media_path)
    verify_self_hash(media, "receipt_sha256")
    require_exact_keys(
        media,
        {
            "schema_version",
            "media_contract",
            "required_contract",
            "display_frames",
            "videos",
            "all_media_gates_pass",
            "receipt_sha256",
        },
        "media validation",
    )
    videos = media.get("videos")
    if (
        media.get("schema_version") != modules.builder.MEDIA_SCHEMA
        or media.get("media_contract") != EXPECTED_MEDIA_CONTRACT
        or media.get("required_contract")
        != {"frame_count": 81, "fps": 25.0, "width": 704, "height": 1056}
        or media.get("display_frames") != DISPLAY_FRAMES
        or type(videos) is not dict
        or set(videos) != set(VIDEO_KEYS)
        or media.get("all_media_gates_pass") is not True
        or file_sha256(media_path) != overlay.get("media_validation_receipt_sha256")
    ):
        raise FinalizeV15CR7Error("media validation receipt differs")
    for key in VIDEO_KEYS:
        row = videos[key]
        require_exact_keys(
            row,
            {
                "relative_path",
                "sha256",
                "frame_count",
                "fps",
                "width",
                "height",
                "gates",
            },
            "media video row",
        )
        relative = EXPECTED_MEDIA_CONTRACT[key]["video"]
        if (
            row.get("relative_path") != relative
            or row.get("sha256") != file_sha256(review / relative)
            or row.get("frame_count") != 81
            or float(row.get("fps", -1.0)) != 25.0
            or row.get("width") != 704
            or row.get("height") != 1056
            or row.get("gates")
            != {
                "frame_count_81": True,
                "fps_25": True,
                "width_704": True,
                "height_1056": True,
            }
        ):
            raise FinalizeV15CR7Error("media video receipt differs")
    return {"overlay": overlay, "media": media}


def replay_review_content(
    *,
    modules: SimpleNamespace,
    run_root: Path,
    source: Path,
    track_receipt: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Rebuild all videos/sheets/index and compare actual decoded content."""

    track_receipt_path = run_root / "tracks/track_receipt.json"
    result_path = run_root / "result.json"
    postflight_path = run_root / "postflight.json"
    receipt_rows = _validate_review_receipts(
        modules=modules,
        run_root=run_root,
        source=source,
        track_receipt_path=track_receipt_path,
        result_path=result_path,
        postflight_path=postflight_path,
    )
    source_frames, _ = _decode_video(source)
    proposal_ids = [row["proposal_id"] for row in track_receipt["proposals"]]
    if result.get("proposal_ids") != proposal_ids:
        raise FinalizeV15CR7Error("review proposal registry differs")
    assignments = result.get("assignments")
    if type(assignments) is not dict or set(assignments) != set(modules.core.ROLE_NAMES):
        raise FinalizeV15CR7Error("review role assignment registry differs")
    review = run_root / "review"
    replay_root = Path(tempfile.mkdtemp(prefix=".v15c-r7-media-replay-", dir=str(run_root)))
    os.chmod(replay_root, 0o700)
    try:
        media_root = replay_root / "media"
        media_root.mkdir(mode=0o700)
        modules.builder._render_contact_sheet(
            frames=source_frames,
            root=track_receipt_path.parent,
            proposal_ids=[],
            label="SOURCE AUTHORITY",
            output=replay_root
            / EXPECTED_MEDIA_CONTRACT["source"]["contact_sheet"],
        )
        modules.builder._render_overlay(
            frames=source_frames,
            root=track_receipt_path.parent,
            proposal_ids=proposal_ids,
            label=f"ALL PROPOSALS ({len(proposal_ids)}); NOT SEMANTIC GT",
            output=replay_root / EXPECTED_MEDIA_CONTRACT["all"]["video"],
        )
        modules.builder._render_contact_sheet(
            frames=source_frames,
            root=track_receipt_path.parent,
            proposal_ids=proposal_ids,
            label=f"ALL {len(proposal_ids)} SOURCE PROPOSALS",
            output=replay_root / EXPECTED_MEDIA_CONTRACT["all"]["contact_sheet"],
        )
        for role in modules.core.ROLE_NAMES:
            candidate = assignments[role]
            selected = [] if candidate is None else [candidate]
            label = f"{role}: {candidate or 'UNASSIGNED'}; REJECT ONLY"
            modules.builder._render_overlay(
                frames=source_frames,
                root=track_receipt_path.parent,
                proposal_ids=selected,
                label=label,
                output=replay_root / EXPECTED_MEDIA_CONTRACT[role]["video"],
            )
            modules.builder._render_contact_sheet(
                frames=source_frames,
                root=track_receipt_path.parent,
                proposal_ids=selected,
                label=label,
                output=replay_root / EXPECTED_MEDIA_CONTRACT[role]["contact_sheet"],
            )
        expected_index = modules.builder._html(result, EXPECTED_MEDIA_CONTRACT).encode(
            "utf-8"
        )
        if (review / "index.html").read_bytes() != expected_index:
            raise FinalizeV15CR7Error("review index content differs")
        video_digests = {
            "source": _frames_sha256(source_frames),
        }
        if file_sha256(review / EXPECTED_MEDIA_CONTRACT["source"]["video"]) != file_sha256(source):
            raise FinalizeV15CR7Error("review source video is not the source authority")
        for key in ("all", *modules.core.ROLE_NAMES):
            video_digests[key] = _compare_video_content(
                review / EXPECTED_MEDIA_CONTRACT[key]["video"],
                replay_root / EXPECTED_MEDIA_CONTRACT[key]["video"],
                key,
            )
        sheet_digests = {}
        for key in VIDEO_KEYS:
            sheet_digests[key] = _compare_sheet_content(
                review / EXPECTED_MEDIA_CONTRACT[key]["contact_sheet"],
                replay_root / EXPECTED_MEDIA_CONTRACT[key]["contact_sheet"],
                key,
            )
        return {
            "overlay_file_sha256": file_sha256(review / "overlay_receipt.json"),
            "media_validation_file_sha256": file_sha256(
                review / "media_validation.json"
            ),
            "canonical_decoded_video_sha256": video_digests,
            "canonical_contact_sheet_pixels_sha256": sheet_digests,
            "index_sha256": file_sha256(review / "index.html"),
            "content_rebuilt_from_source_masks_assignments": True,
            "all_five_videos_verified": True,
            "all_five_contact_sheets_verified": True,
            "receipt_rows": receipt_rows,
        }
    finally:
        shutil.rmtree(replay_root)


def verify_evidence_bundle(
    *,
    root: Path,
    snapshot: Path,
    release_path: Path,
    release_sha256: str,
    run_root: Path,
    source: Path,
    r6_receipt: Path,
    r6_tensors: Path,
) -> Mapping[str, Any]:
    """Public fail-closed verifier; module injection is deliberately impossible."""

    manifest = verify_release(root, release_path, release_sha256)
    snapshot_manifest = verify_release(
        snapshot,
        snapshot / RELEASE_RELATIVE_PATH,
        release_sha256,
    )
    if snapshot_manifest != manifest:
        raise FinalizeV15CR7Error("source and snapshot manifests differ")
    snapshot_rows = verify_snapshot(snapshot, manifest, release_sha256)
    modules = load_sealed_modules(snapshot, manifest)
    if verify_snapshot(snapshot, manifest, release_sha256) != snapshot_rows:
        raise FinalizeV15CR7Error("snapshot changed while loading validators")
    spec_path = snapshot / (
        "methods/bernini_action_editing/assets/"
        "e00_source_sam2_proposal_role_probe_v15c_r6.json"
    )
    spec = modules.materializer.read_spec(spec_path)
    expected_external = {
        source: Path(spec["source"]["path"]),
        r6_receipt: Path(spec["r6"]["probe_receipt_path"]),
        r6_tensors: Path(spec["r6"]["affinity_tensor_path"]),
    }
    for actual, expected in expected_external.items():
        try:
            same = os.path.samefile(actual.resolve(strict=True), expected.resolve(strict=True))
        except OSError as error:
            raise FinalizeV15CR7Error("canonical external input is unavailable") from error
        if not same:
            raise FinalizeV15CR7Error("external input path authority differs")
    observer = replay_observer_evidence(
        modules=modules,
        spec_path=spec_path,
        source=source,
        r6_receipt_path=r6_receipt,
        r6_tensors=r6_tensors,
        run_root=run_root,
    )
    review = replay_review_content(
        modules=modules,
        run_root=run_root,
        source=source,
        track_receipt=observer["track_receipt"],
        result=observer["result"],
    )
    if verify_snapshot(snapshot, manifest, release_sha256) != snapshot_rows:
        raise FinalizeV15CR7Error("snapshot changed during evidence replay")
    result = {
        "schema_version": REPLAY_SCHEMA,
        "status": "LOCAL_EVIDENCE_REPLAY_PASS_REMOTE_STILL_UNAUDITED",
        "release_file_sha256": release_sha256,
        "release_internal_sha256": manifest["release_sha256"],
        "snapshot_files": snapshot_rows,
        "observer_replay": {
            key: value
            for key, value in observer.items()
            if key not in {"spec", "track_receipt", "result", "postflight"}
        },
        "review_replay": review,
        "observer_execution_authorized": False,
        "remote_gpu_status": "REMOTE_GPU_UNAUDITED",
        "localization_semantically_certified": False,
        "scientific_claim_authorized": False,
        "route_status": "ROUTE_NO_GO",
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    result["receipt_sha256"] = object_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    release = commands.add_parser("verify-release")
    release.add_argument("--root", required=True, type=Path)
    release.add_argument("--release-manifest", required=True, type=Path)
    release.add_argument("--expected-sha256", required=True)
    snapshot = commands.add_parser("verify-snapshot")
    snapshot.add_argument("--snapshot", required=True, type=Path)
    snapshot.add_argument("--release-sha256", required=True)
    evidence = commands.add_parser("verify-evidence")
    evidence.add_argument("--root", required=True, type=Path)
    evidence.add_argument("--snapshot", required=True, type=Path)
    evidence.add_argument("--release-manifest", required=True, type=Path)
    evidence.add_argument("--release-sha256", required=True)
    evidence.add_argument("--run-root", required=True, type=Path)
    evidence.add_argument("--source", required=True, type=Path)
    evidence.add_argument("--r6-receipt", required=True, type=Path)
    evidence.add_argument("--r6-tensors", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify-release":
        verify_release(args.root, args.release_manifest, args.expected_sha256)
    elif args.command == "verify-snapshot":
        manifest = verify_release(
            args.snapshot,
            args.snapshot / RELEASE_RELATIVE_PATH,
            args.release_sha256,
        )
        verify_snapshot(args.snapshot, manifest, args.release_sha256)
    elif args.command == "verify-evidence":
        receipt = verify_evidence_bundle(
            root=args.root,
            snapshot=args.snapshot,
            release_path=args.release_manifest,
            release_sha256=args.release_sha256,
            run_root=args.run_root,
            source=args.source,
            r6_receipt=args.r6_receipt,
            r6_tensors=args.r6_tensors,
        )
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    else:  # pragma: no cover
        raise FinalizeV15CR7Error("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
