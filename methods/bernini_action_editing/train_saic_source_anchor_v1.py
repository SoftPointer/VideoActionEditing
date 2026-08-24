#!/usr/bin/env python3
"""Train the target-free SAIC Stage-A source-appearance anchor pilot.

This executable is deliberately *not* action training.  It uses 64 real v17
exact81 sources as clean flow endpoints and keeps 16 identities outside the
optimizer.  The native condition video is a deterministic 21-phase scramble;
four RGB references stay source-derived.  A correct-source native RV2V field
must reconstruct the real source endpoint and must beat a same-bucket wrong
source.  No target edit, action proposal, donor, mask, pose, flow, track, or
trajectory is accepted by the manifest or model-call interface.

The only trainable parameters are the late blocks 23..29 self-attention Q/O
rank-8 residuals from ``saic_source_anchor_adapter_v1``.  They are active only
at native exact40 coordinates 35..39 and only on the target suffix of real V
and VI packs.  WORLD8 is fixed to DP2 x Ulysses-SP4.  Four native RV2V/APG
fields are evaluated for every prediction.  To bound memory, prediction leaves
are differentiated first and the three source-conditioned fields are replayed
serially for their exact linear VJP; the source-free field has identically zero
adapter VJP.  Gradient checkpointing is forbidden.

The formal v1 pass performs exactly 32 updates per DP arm, so every one of the
64 optimizer sources is used exactly once as a clean endpoint.  A separate
one- or two-update smoke mode is explicitly incomplete and cannot publish a
checkpoint.  A fresh held-out no-op reconstruction and wrong-source dependence audit runs before and after optimization.  A failed
gate publishes a receipt/history bundle with no weights.  Only a passed gate
may publish the closed safetensors adapter.  Neither outcome claims action
editing, appearance preservation in decoded RGB, or generalization.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora  # noqa: E402
import infer_native_identity_generation_canary as native_infer  # noqa: E402
import infer_source_kv_carrier_oracle as checkpoint_audit  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import pair_v5_native_bridge as native_bridge  # noqa: E402
import saic_source_anchor_adapter_v1 as anchor_adapter  # noqa: E402
import saic_source_anchor_objective_v1 as anchor_objective  # noqa: E402
import source_self_native_ref_contrastive_v3 as native  # noqa: E402
import source_self_native_rv2v_guidance as guidance  # noqa: E402
import source_self_runtime as runtime  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_pair_v5_action_preference as native_runtime  # noqa: E402
from tools import build_saic_source_anchor_manifest_v1 as manifest_builder  # noqa: E402


METHOD_NAME = "bernini-saic-source-anchor-stage-a-v1"
RUN_RECEIPT_SCHEMA = "bernini-saic-source-anchor-run-receipt-v2"
HISTORY_SCHEMA = "bernini-saic-source-anchor-history-v2"
SAFETENSORS_SCHEMA = "bernini-saic-source-anchor-safetensors-v1"
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FRAME_COUNT = 81
FPS = 25.0
LATENT_PHASES = 21
REFERENCE_INDICES = (0, 27, 53, 80)
TRAIN_PER_ARM = 32
HOLDOUT_PER_ARM = 8
FORMAL_UPDATES = 32
SMOKE_MAX_UPDATES = 2
ACTIVE_SIGMA_INDICES = (35, 36, 37, 38, 39)
DEFAULT_LEARNING_RATE = 1.0e-5
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 20260809
FORMAL_GRADIENT_ACCUMULATION_STEPS = 1
NOOP_INSTRUCTION = (
    "Preserve the source video exactly. Keep every subject, appearance, action, "
    "camera, background, timing, and motion unchanged."
)
VJP_RTOL = 2.0e-5
VJP_ATOL = 2.0e-5
NOOP_RELATIVE_TOLERANCE = 0.02
NOOP_ABSOLUTE_TOLERANCE = 1.0e-6
STRICT_FLOW_IMPROVEMENT_EPSILON = 1.0e-5
STRICT_ADVANTAGE_IMPROVEMENT_EPSILON = 1.0e-5
STRICT_POSITIVE_FRACTION_STEP = 1.0 / 16.0
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "optimizer_authorized",
        "source_root",
        "selection_seed",
        "frame_count",
        "fps",
        "train_count",
        "holdout_count",
        "selected_bucket_counts",
        "eligible_bucket_counts",
        "strict_action_iids_excluded",
        "wrong_source_policy",
        "holdout_used_by_optimizer",
        "train_rows",
        "holdout_rows",
        "input_closure",
        "manifest_digest",
    }
)
_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "split",
        "row_index",
        "dp_arm",
        "iid",
        "source_video_path",
        "source_video_sha256",
        "wrong_iid",
        "wrong_source_video_path",
        "wrong_source_video_sha256",
        "frame_count",
        "fps",
        "reported_fps",
        "bucket_hw",
        "scramble_seed",
        "row_digest",
    }
)


class SAICSourceAnchorTrainingError(RuntimeError):
    """Raised before an ambiguous update or checkpoint can be accepted."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SAICSourceAnchorTrainingError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, expected: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise SAICSourceAnchorTrainingError(
            f"{label} keys differ: missing={sorted(set(expected)-actual)} "
            f"extra={sorted(actual-set(expected))}"
        )
    return value


def _require_sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise SAICSourceAnchorTrainingError(
            f"{label} must be lowercase SHA-{'1' if length == 40 else '256'}"
        )
    return value


def _stable_bytes(path: Path, *, expected_sha256: str, label: str) -> bytes:
    expected = _require_sha(expected_sha256, length=64, label=f"{label} SHA-256")
    if path.is_symlink() or not path.is_file():
        raise SAICSourceAnchorTrainingError(f"{label} must be a plain file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda value: (  # noqa: E731
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise SAICSourceAnchorTrainingError(f"{label} changed while reading")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SAICSourceAnchorTrainingError(f"{label} SHA-256 differs")
    return raw


def _strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise SAICSourceAnchorTrainingError(f"{label} contains {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SAICSourceAnchorTrainingError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SAICSourceAnchorTrainingError(f"cannot decode {label}") from error
    if not isinstance(value, Mapping):
        raise SAICSourceAnchorTrainingError(f"{label} root must be an object")
    return value


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str

    @classmethod
    def capture(cls, path: Path, expected_sha256: str, *, label: str) -> "FileSnapshot":
        _stable_bytes(path, expected_sha256=expected_sha256, label=label)
        stat = path.stat()
        return cls(
            path,
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            expected_sha256,
        )

    def assert_unchanged(self) -> None:
        stat = self.path.stat()
        if (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        ) != (self.device, self.inode, self.size, self.mtime_ns):
            raise SAICSourceAnchorTrainingError(f"input changed: {self.path}")
        _stable_bytes(self.path, expected_sha256=self.sha256, label="bound input")


@dataclass(frozen=True)
class RetainedSourceBinding:
    """One source inode held open by the job's long-lived Bash supervisor."""

    declared_path: Path
    sha256: str
    supervisor_pid: int
    descriptor: int
    device: int
    inode: int
    size: int
    mtime_ns: int
    uid: int
    mode: int

    @property
    def proc_path(self) -> Path:
        return Path(f"/proc/{self.supervisor_pid}/fd/{self.descriptor}")

    def assert_parent_binding(self, *, label: str) -> None:
        """Require the descriptor to remain installed in the retained parent."""

        try:
            info = self.proc_path.lstat()
        except (FileNotFoundError, OSError) as error:
            raise SAICSourceAnchorTrainingError(
                f"{label} retained parent descriptor is absent"
            ) from error
        if not stat.S_ISLNK(info.st_mode):
            raise SAICSourceAnchorTrainingError(
                f"{label} retained parent descriptor path differs"
            )

    def stable_bytes(self, *, label: str) -> bytes:
        """Read the retained inode, never the mutable manifest pathname."""

        self.assert_parent_binding(label=label)
        descriptor = os.open(self.proc_path, os.O_RDONLY)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_dev != self.device
                or before.st_ino != self.inode
                or before.st_size != self.size
                or before.st_mtime_ns != self.mtime_ns
                or before.st_uid != self.uid
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != self.mode
            ):
                raise SAICSourceAnchorTrainingError(
                    f"{label} retained source identity differs"
                )
            result = hashlib.sha256()
            os.lseek(descriptor, 0, os.SEEK_SET)
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                result.update(chunk)
            os.lseek(descriptor, 0, os.SEEK_SET)
            after = os.fstat(descriptor)
            if (
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_uid, after.st_nlink, stat.S_IMODE(after.st_mode))
                != (self.device, self.inode, self.size, self.mtime_ns,
                    self.uid, 1, self.mode)
                or result.hexdigest() != self.sha256
            ):
                raise SAICSourceAnchorTrainingError(
                    f"{label} retained source changed while reading"
                )
            return b""
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class SourceRow:
    split: str
    row_index: int
    dp_arm: int
    iid: str
    source_path: Path
    source_runtime_path: Path
    source_sha256: str
    wrong_iid: str
    wrong_path: Path
    wrong_runtime_path: Path
    wrong_sha256: str
    bucket_hw: tuple[int, int]
    scramble_seed: int
    row_digest: str


@dataclass(frozen=True)
class SourceManifest:
    snapshot: FileSnapshot
    manifest_digest: str
    train_rows: tuple[SourceRow, ...]
    holdout_rows: tuple[SourceRow, ...]
    retained_sources: Mapping[Path, RetainedSourceBinding]
    retained_source_map_sha256: Optional[str]

    def rows_for_arm(self, *, split: str, arm: int) -> tuple[SourceRow, ...]:
        source = self.train_rows if split == "train" else self.holdout_rows
        return tuple(row for row in source if row.dp_arm == arm)

    def assert_unchanged(self) -> None:
        self.snapshot.assert_unchanged()
        observed: dict[Path, str] = {}
        for row in self.train_rows + self.holdout_rows:
            observed[row.source_path] = row.source_sha256
            observed[row.wrong_path] = row.wrong_sha256
        if len(observed) != len(self.train_rows) + len(self.holdout_rows):
            raise SAICSourceAnchorTrainingError(
                "manifest source path closure is not one file per identity"
            )
        for path, digest in sorted(observed.items(), key=lambda item: str(item[0])):
            binding = self.retained_sources.get(path)
            if binding is None:
                _stable_bytes(path, expected_sha256=digest, label="bound source video")
            else:
                if binding.sha256 != digest:
                    raise SAICSourceAnchorTrainingError(
                        "retained source path/SHA binding differs"
                    )
                binding.stable_bytes(label="bound retained source video")


def load_retained_source_fd_map(
    raw_value: str,
    *,
    expected_sha256: str,
    expected_supervisor_pid: int,
) -> Mapping[Path, RetainedSourceBinding]:
    """Validate the stage0-generated exact80 retained source descriptor map."""

    if type(raw_value) is not str or not raw_value or "\x00" in raw_value:
        raise SAICSourceAnchorTrainingError("retained source FD map is absent")
    try:
        raw = raw_value.encode("ascii")
    except UnicodeEncodeError as error:
        raise SAICSourceAnchorTrainingError(
            "retained source FD map is not ASCII"
        ) from error
    expected = _require_sha(
        expected_sha256, length=64, label="retained source FD map SHA-256"
    )
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SAICSourceAnchorTrainingError("retained source FD map SHA-256 differs")
    value = _strict_json(raw, label="retained source FD map")
    if raw != canonical_json_bytes(value):
        raise SAICSourceAnchorTrainingError(
            "retained source FD map bytes are not canonical JSON"
        )
    if set(value) != {
        "schema_version", "supervisor_pid", "source_count", "rows",
        "source_descriptors_held_by_parent_supervisor",
        "workers_open_only_parent_proc_fd_paths", "map_digest",
    }:
        raise SAICSourceAnchorTrainingError("retained source FD map fields differ")
    unsigned = dict(value)
    claimed = _require_sha(
        unsigned.pop("map_digest"), length=64,
        label="retained source FD map digest",
    )
    rows = value.get("rows")
    if (
        value.get("schema_version")
        != "saic-source-anchor-retained-source-fd-map-v1"
        or value.get("supervisor_pid") != expected_supervisor_pid
        or value.get("source_count") != 80
        or value.get("source_descriptors_held_by_parent_supervisor") is not True
        or value.get("workers_open_only_parent_proc_fd_paths") is not True
        or object_sha256(unsigned) != claimed
        or not isinstance(rows, list)
        or len(rows) != 80
    ):
        raise SAICSourceAnchorTrainingError("retained source FD map contract differs")
    result: dict[Path, RetainedSourceBinding] = {}
    descriptors: set[int] = set()
    identities: set[tuple[int, int]] = set()
    for item in rows:
        if not isinstance(item, Mapping) or set(item) != {
            "declared_path", "sha256", "descriptor", "device", "inode",
            "size", "mtime_ns", "uid", "mode",
        }:
            raise SAICSourceAnchorTrainingError("retained source FD row fields differ")
        declared = _canonical_plain_path(
            item["declared_path"], label="retained source declared path",
            verify_file=False,
        )
        digest = _require_sha(
            item["sha256"], length=64, label="retained source SHA-256"
        )
        integer_fields = {
            name: item[name]
            for name in ("descriptor", "device", "inode", "size", "mtime_ns", "uid", "mode")
        }
        if (
            any(type(number) is not int for number in integer_fields.values())
            or integer_fields["descriptor"] < 3
            or integer_fields["device"] < 0
            or integer_fields["inode"] <= 0
            or integer_fields["size"] <= 0
            or integer_fields["mtime_ns"] <= 0
            or integer_fields["uid"] != os.getuid()
            or integer_fields["mode"] & 0o022
            or not integer_fields["mode"] & 0o400
            or declared in result
            or integer_fields["descriptor"] in descriptors
            or (integer_fields["device"], integer_fields["inode"]) in identities
        ):
            raise SAICSourceAnchorTrainingError("retained source FD row differs")
        binding = RetainedSourceBinding(
            declared, digest, expected_supervisor_pid,
            integer_fields["descriptor"], integer_fields["device"],
            integer_fields["inode"], integer_fields["size"],
            integer_fields["mtime_ns"], integer_fields["uid"],
            integer_fields["mode"],
        )
        binding.stable_bytes(label=f"retained source {declared.name}")
        result[declared] = binding
        descriptors.add(binding.descriptor)
        identities.add((binding.device, binding.inode))
    if list(result) != sorted(result, key=str):
        raise SAICSourceAnchorTrainingError("retained source FD rows are not sorted")
    return result


def _canonical_plain_path(value: Any, *, label: str, verify_file: bool) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise SAICSourceAnchorTrainingError(f"{label} must be a path string")
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path("/"):
        raise SAICSourceAnchorTrainingError(f"{label} must be absolute non-root")
    if verify_file:
        resolved = path.resolve(strict=True)
        if resolved != path or path.is_symlink() or not path.is_file():
            raise SAICSourceAnchorTrainingError(f"{label} must be a canonical plain file")
    return path


def load_manifest(
    path_value: str | Path,
    *,
    expected_sha256: str,
    verify_files: bool = True,
    retained_sources: Optional[Mapping[Path, RetainedSourceBinding]] = None,
    retained_source_map_sha256: Optional[str] = None,
) -> SourceManifest:
    path = _canonical_plain_path(str(path_value), label="manifest", verify_file=True)
    snapshot = FileSnapshot.capture(path, expected_sha256, label="manifest")
    root = _closed(_strict_json(_stable_bytes(path, expected_sha256=expected_sha256, label="manifest"), label="manifest"), _ROOT_FIELDS, label="manifest")
    unsigned_root = dict(root)
    declared_digest = _require_sha(
        unsigned_root.pop("manifest_digest"), length=64, label="manifest digest"
    )
    if (
        root["schema_version"] != manifest_builder.SCHEMA_VERSION
        or root["optimizer_authorized"] is not False
        or root["frame_count"] != FRAME_COUNT
        or float(root["fps"]) != FPS
        or root["train_count"] != manifest_builder.TRAIN_COUNT
        or root["holdout_count"] != manifest_builder.HOLDOUT_COUNT
        or root["strict_action_iids_excluded"]
        != sorted(manifest_builder.STRICT_ACTION_IIDS)
        or root["wrong_source_policy"]
        != "same_split_same_bucket_same_dp_arm_fixed_point_free"
        or root["holdout_used_by_optimizer"] is not False
        or object_sha256(unsigned_root) != declared_digest
    ):
        raise SAICSourceAnchorTrainingError("manifest root contract differs")
    selected_bucket_counts = root["selected_bucket_counts"]
    if (
        not isinstance(selected_bucket_counts, Mapping)
        or not selected_bucket_counts
        or any(
            type(key) is not str
            or re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", key) is None
            or type(count) is not int
            or count <= 0
            or count % 2
            for key, count in selected_bucket_counts.items()
        )
        or sum(selected_bucket_counts.values()) != 80
    ):
        raise SAICSourceAnchorTrainingError("selected bucket-count closure differs")

    def parse_rows(value: Any, *, split: str, count: int) -> tuple[SourceRow, ...]:
        if not isinstance(value, list) or len(value) != count:
            raise SAICSourceAnchorTrainingError(f"{split} row count differs")
        result: list[SourceRow] = []
        for expected_index, item in enumerate(value):
            row = _closed(item, _ROW_FIELDS, label=f"{split} row")
            unsigned = dict(row)
            row_digest = _require_sha(
                unsigned.pop("row_digest"), length=64, label="row digest"
            )
            source_path = _canonical_plain_path(
                row["source_video_path"], label="source video",
                verify_file=verify_files and retained_sources is None,
            )
            wrong_path = _canonical_plain_path(
                row["wrong_source_video_path"],
                label="wrong source video",
                verify_file=verify_files and retained_sources is None,
            )
            source_sha = _require_sha(
                row["source_video_sha256"], length=64, label="source SHA"
            )
            wrong_sha = _require_sha(
                row["wrong_source_video_sha256"], length=64, label="wrong SHA"
            )
            source_runtime_path = source_path
            wrong_runtime_path = wrong_path
            if retained_sources is not None:
                source_binding = retained_sources.get(source_path)
                wrong_binding = retained_sources.get(wrong_path)
                if (
                    source_binding is None
                    or wrong_binding is None
                    or source_binding.sha256 != source_sha
                    or wrong_binding.sha256 != wrong_sha
                ):
                    raise SAICSourceAnchorTrainingError(
                        "manifest row is not bound to exact retained source FDs"
                    )
                source_binding.stable_bytes(label="source video")
                wrong_binding.stable_bytes(label="wrong source")
                source_runtime_path = source_binding.proc_path
                wrong_runtime_path = wrong_binding.proc_path
            elif verify_files:
                _stable_bytes(source_path, expected_sha256=source_sha, label="source video")
                _stable_bytes(wrong_path, expected_sha256=wrong_sha, label="wrong source")
            if (
                row["schema_version"] != manifest_builder.SCHEMA_VERSION
                or row["split"] != split
                or row["row_index"] != expected_index
                or type(row["dp_arm"]) is not int
                or row["dp_arm"] not in (0, 1)
                or row["iid"] in manifest_builder.STRICT_ACTION_IIDS
                or row["wrong_iid"] in manifest_builder.STRICT_ACTION_IIDS
                or row["iid"] == row["wrong_iid"]
                or row["frame_count"] != FRAME_COUNT
                or float(row["fps"]) != FPS
                or abs(float(row["reported_fps"]) - FPS) > 1.0e-3
                or not isinstance(row["bucket_hw"], list)
                or len(row["bucket_hw"]) != 2
                or any(
                    type(item) is not int or item <= 0 or item % 16
                    for item in row["bucket_hw"]
                )
                or type(row["scramble_seed"]) is not int
                or not 0 <= row["scramble_seed"] < 2**63
                or object_sha256(unsigned) != row_digest
            ):
                raise SAICSourceAnchorTrainingError(f"{split} row contract differs")
            result.append(
                SourceRow(
                    split,
                    expected_index,
                    int(row["dp_arm"]),
                    str(row["iid"]),
                    source_path,
                    source_runtime_path,
                    source_sha,
                    str(row["wrong_iid"]),
                    wrong_path,
                    wrong_runtime_path,
                    wrong_sha,
                    (int(row["bucket_hw"][0]), int(row["bucket_hw"][1])),
                    int(row["scramble_seed"]),
                    row_digest,
                )
            )
        iids = {row.iid for row in result}
        if len(iids) != count or any(row.wrong_iid not in iids for row in result):
            raise SAICSourceAnchorTrainingError(f"{split} derangement closure differs")
        by_iid = {row.iid: row for row in result}
        if any(
            row.wrong_path != by_iid[row.wrong_iid].source_path
            or row.wrong_sha256 != by_iid[row.wrong_iid].source_sha256
            or row.bucket_hw != by_iid[row.wrong_iid].bucket_hw
            for row in result
        ):
            raise SAICSourceAnchorTrainingError(
                f"{split} wrong-source path/SHA binding differs"
            )
        for arm in range(DP_SIZE):
            arm_rows = [row for row in result if row.dp_arm == arm]
            expected = count // DP_SIZE
            if len(arm_rows) != expected or any(
                next(candidate for candidate in result if candidate.iid == row.wrong_iid).dp_arm
                != arm
                for row in arm_rows
            ):
                raise SAICSourceAnchorTrainingError(f"{split} DP partition differs")
        return tuple(result)

    train = parse_rows(root["train_rows"], split="train", count=64)
    heldout = parse_rows(root["holdout_rows"], split="holdout", count=16)
    if {row.iid for row in train} & {row.iid for row in heldout}:
        raise SAICSourceAnchorTrainingError("held-out identities leaked into optimizer")
    observed_bucket_counts: dict[str, int] = {}
    for row in train + heldout:
        key = f"{row.bucket_hw[0]}x{row.bucket_hw[1]}"
        observed_bucket_counts[key] = observed_bucket_counts.get(key, 0) + 1
    if dict(sorted(observed_bucket_counts.items())) != dict(selected_bucket_counts):
        raise SAICSourceAnchorTrainingError("selected bucket counts differ from rows")
    expected_paths = {row.source_path for row in train + heldout}
    if retained_sources is not None and set(retained_sources) != expected_paths:
        raise SAICSourceAnchorTrainingError(
            "retained source FD map is not the manifest exact80 closure"
        )
    return SourceManifest(
        snapshot, declared_digest, train, heldout,
        dict(retained_sources or {}), retained_source_map_sha256,
    )


def schedule_index_for_update(update_index: int) -> int:
    if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index < 0:
        raise SAICSourceAnchorTrainingError("update index must be nonnegative")
    return ACTIVE_SIGMA_INDICES[update_index % len(ACTIVE_SIGMA_INDICES)]


def noise_seed(*, seed: int, update_index: int, dp_arm: int, phase: str) -> int:
    if phase not in {"train", "heldout-before", "heldout-after"}:
        raise SAICSourceAnchorTrainingError("unknown noise phase")
    material = f"saic-anchor-noise-v1\0{seed}\0{phase}\0{update_index}\0{dp_arm}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def validate_cli(args: argparse.Namespace) -> Mapping[str, Any]:
    if (
        (WORLD_SIZE, SP_SIZE, DP_SIZE)
        != (runtime.WORLD_SIZE, runtime.SP_SIZE, runtime.DP_SIZE)
        or args.num_frames != FRAME_COUNT
    ):
        raise SAICSourceAnchorTrainingError("WORLD8 DP2xSP4 exact81 contract differs")
    if args.mode == "formal":
        if args.max_updates != FORMAL_UPDATES:
            raise SAICSourceAnchorTrainingError(
                "formal v1 requires exactly 32 updates, one clean endpoint per arm row"
            )
        if args.ack_incomplete_row_coverage_smoke:
            raise SAICSourceAnchorTrainingError("formal mode forbids smoke acknowledgement")
        exact_formal = {
            "learning_rate": DEFAULT_LEARNING_RATE,
            "max_grad_norm": DEFAULT_MAX_GRAD_NORM,
            "wrong_source_margin": anchor_objective.DEFAULT_WRONG_SOURCE_MARGIN,
            "ranking_weight": anchor_objective.DEFAULT_RANKING_WEIGHT,
            "seed": DEFAULT_SEED,
            "gradient_accumulation_steps": FORMAL_GRADIENT_ACCUMULATION_STEPS,
        }
        observed_formal = {
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
            "wrong_source_margin": args.wrong_source_margin,
            "ranking_weight": args.ranking_weight,
            "seed": args.seed,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
        if observed_formal != exact_formal:
            raise SAICSourceAnchorTrainingError(
                "formal optimizer/hyperparameter contract differs"
            )
    elif args.mode == "smoke":
        if (
            type(args.max_updates) is not int
            or not 1 <= args.max_updates <= SMOKE_MAX_UPDATES
            or args.ack_incomplete_row_coverage_smoke is not True
        ):
            raise SAICSourceAnchorTrainingError(
                "smoke mode requires 1..2 updates and incomplete-row-coverage acknowledgement"
            )
    else:
        raise SAICSourceAnchorTrainingError("unknown source-anchor run mode")
    for name in (
        "learning_rate",
        "max_grad_norm",
        "wrong_source_margin",
        "ranking_weight",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise SAICSourceAnchorTrainingError(f"{name} must be finite and positive")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise SAICSourceAnchorTrainingError("seed must lie in [0,2^63)")
    if args.gradient_accumulation_steps != FORMAL_GRADIENT_ACCUMULATION_STEPS:
        raise SAICSourceAnchorTrainingError("gradient accumulation must be exactly one")
    if re.fullmatch(r"[1-9][0-9]*", args.slurm_job_id) is None:
        raise SAICSourceAnchorTrainingError("Slurm job ID differs")
    if args.ack_source_anchor_only_no_action_claim is not True:
        raise SAICSourceAnchorTrainingError("source-anchor-only acknowledgement is required")
    for name in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        _require_sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "method_source_archive_sha256",
        "trainer_source_sha256",
        "release_manifest_sha256",
        "release_manifest_digest",
        "submission_receipt_sha256",
        "submission_receipt_digest",
        "python_executable_sha256",
        "python_version_stdout_sha256",
        "formal_full60_admission_sha256",
        "formal_full60_admission_digest",
        "source_fd_map_sha256",
        "archive_member_manifest_sha256",
        "extracted_tree_manifest_sha256",
        "archive_binding_receipt_digest",
        "runtime_origin_manifest_sha256",
        "runtime_origin_receipt_digest",
    ):
        _require_sha(getattr(args, name), length=64, label=name)
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise SAICSourceAnchorTrainingError("checkpoint identity differs")
    return {
        "world_size": WORLD_SIZE,
        "data_parallel_size": DP_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "frame_count": FRAME_COUNT,
        "optimizer_updates": args.max_updates,
        "mode": args.mode,
        "all_train_rows_used_once_as_clean_endpoint": args.mode == "formal",
        "active_sigma_indices": list(ACTIVE_SIGMA_INDICES),
        "train_rows_per_dp_arm": TRAIN_PER_ARM,
        "heldout_rows_per_dp_arm": HOLDOUT_PER_ARM,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "wrong_source_margin": args.wrong_source_margin,
        "ranking_weight": args.ranking_weight,
        "seed": args.seed,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "active_adapter_blocks": list(range(23, 30)),
        "adapter_rank": 8,
        "adapter_projections": ["attn1.to_q", "attn1.to_out.0"],
        "source_video_count": 80,
        "source_fd_map_sha256": args.source_fd_map_sha256,
        "source_supervisor_pid": args.source_supervisor_pid,
        "source_video_pathname_fallback_allowed": False,
        "source_video_retained_inode_rehashed_before_and_after_decode": True,
        "archive_member_manifest_sha256": args.archive_member_manifest_sha256,
        "extracted_tree_manifest_sha256": args.extracted_tree_manifest_sha256,
        "runtime_origin_manifest_sha256": args.runtime_origin_manifest_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256", required=True
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--max-updates", type=int, default=FORMAL_UPDATES)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument(
        "--wrong-source-margin",
        type=float,
        default=anchor_objective.DEFAULT_WRONG_SOURCE_MARGIN,
    )
    parser.add_argument(
        "--ranking-weight", type=float, default=anchor_objective.DEFAULT_RANKING_WEIGHT
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=FORMAL_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--num-frames", type=int, choices=(FRAME_COUNT,), default=FRAME_COUNT)
    parser.add_argument("--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--trainer-source-sha256", required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--release-manifest-digest", required=True)
    parser.add_argument("--submission-receipt-sha256", required=True)
    parser.add_argument("--submission-receipt-digest", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--python-executable-sha256", required=True)
    parser.add_argument("--python-version-stdout-sha256", required=True)
    parser.add_argument("--formal-full60-admission-sha256", required=True)
    parser.add_argument("--formal-full60-admission-digest", required=True)
    parser.add_argument("--source-fd-map-sha256", required=True)
    parser.add_argument("--source-supervisor-pid", type=int, required=True)
    parser.add_argument("--archive-member-manifest-sha256", required=True)
    parser.add_argument("--extracted-tree-manifest-sha256", required=True)
    parser.add_argument("--archive-binding-receipt-digest", required=True)
    parser.add_argument("--runtime-origin-manifest-sha256", required=True)
    parser.add_argument("--runtime-origin-receipt-digest", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--ack-source-anchor-only-no-action-claim", action="store_true")
    parser.add_argument("--ack-incomplete-row-coverage-smoke", action="store_true")
    return parser


@dataclass(frozen=True)
class EncodedSource:
    iid: str
    clean: Any
    references: tuple[Any, ...]
    tensor_digest: str


def _broadcast_sp_tensor(value: Any, *, parallel: runtime.ParallelContext) -> None:
    import torch.distributed as dist

    source_rank = runtime.SP_GROUP_RANKS[parallel.contract.arm_index][0]
    dist.broadcast(value, src=source_rank, group=parallel.sp_group)


def _encode_assigned_sources(
    manifest: SourceManifest,
    *,
    vae: Any,
    device: Any,
    parallel: runtime.ParallelContext,
) -> Mapping[str, EncodedSource]:
    """Encode each arm's 32 train + 8 held-out sources and four refs once."""

    import torch
    import torch.distributed as dist
    from bernini.pipeline import _vae_encode

    assigned_rows = manifest.rows_for_arm(split="train", arm=parallel.contract.arm_index) + manifest.rows_for_arm(split="holdout", arm=parallel.contract.arm_index)
    if len(assigned_rows) != TRAIN_PER_ARM + HOLDOUT_PER_ARM:
        raise SAICSourceAnchorTrainingError("assigned source count differs")
    by_iid = {row.iid: row for row in assigned_rows}
    if len(by_iid) != len(assigned_rows) or any(row.wrong_iid not in by_iid for row in assigned_rows):
        raise SAICSourceAnchorTrainingError("assigned wrong-source closure differs")
    result: dict[str, EncodedSource] = {}
    leader = parallel.contract.sp_rank == 0
    source_rank = runtime.SP_GROUP_RANKS[parallel.contract.arm_index][0]
    vae.to(device)
    for iid in sorted(by_iid):
        row = by_iid[iid]
        metadata_box: list[Any] = [None]
        tensors: list[Any] = []
        if leader:
            binding = manifest.retained_sources.get(row.source_path)
            if binding is None:
                _stable_bytes(
                    row.source_runtime_path,
                    expected_sha256=row.source_sha256,
                    label=f"source {iid}",
                )
            else:
                binding.stable_bytes(label=f"retained source {iid}")
            pixels, metadata = infer_lora.prepare_exact_source(
                row.source_runtime_path
            )
            if binding is not None:
                binding.stable_bytes(
                    label=f"retained source {iid} after complete decode"
                )
            if (
                tuple(metadata.get("source_derived_bucket_hw", ())) != row.bucket_hw
                or metadata.get("frame_count") != FRAME_COUNT
                or float(metadata.get("fps", -1)) != FPS
            ):
                raise SAICSourceAnchorTrainingError(f"source {iid} preprocessing changed")
            pixels = pixels.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                clean = _vae_encode(vae, pixels).float().detach().contiguous()
                references = tuple(
                    _vae_encode(vae, pixels[:, :, index : index + 1].contiguous())
                    .float()
                    .detach()
                    .contiguous()
                    for index in REFERENCE_INDICES
                )
            tensors = [clean, *references]
            metadata_box[0] = {
                "shapes": [list(map(int, tensor.shape)) for tensor in tensors],
                "digests": [runtime.tensor_sha256(tensor) for tensor in tensors],
            }
            del pixels
        dist.broadcast_object_list(metadata_box, src=source_rank, group=parallel.sp_group)
        metadata = metadata_box[0]
        if (
            not isinstance(metadata, Mapping)
            or not isinstance(metadata.get("shapes"), list)
            or len(metadata["shapes"]) != 5
            or not isinstance(metadata.get("digests"), list)
            or len(metadata["digests"]) != 5
        ):
            raise SAICSourceAnchorTrainingError("broadcast source metadata differs")
        if not leader:
            tensors = [
                torch.empty(tuple(shape), dtype=torch.float32, device=device)
                for shape in metadata["shapes"]
            ]
        for tensor in tensors:
            _broadcast_sp_tensor(tensor, parallel=parallel)
        if (
            tuple(tensors[0].shape[:3]) != (1, 16, LATENT_PHASES)
            or any(
                tuple(reference.shape)
                != (1, 16, 1, int(tensors[0].shape[3]), int(tensors[0].shape[4]))
                for reference in tensors[1:]
            )
            or [runtime.tensor_sha256(tensor) for tensor in tensors]
            != metadata["digests"]
        ):
            raise SAICSourceAnchorTrainingError(f"encoded source {iid} differs")
        tensor_digest = object_sha256(metadata["digests"])
        runtime.digest_consensus(
            tensor_digest,
            group=parallel.sp_group,
            expected_count=SP_SIZE,
            label=f"encoded source {iid}",
        )
        result[iid] = EncodedSource(
            iid,
            tensors[0].cpu().contiguous(),
            tuple(tensor.cpu().contiguous() for tensor in tensors[1:]),
            tensor_digest,
        )
        del tensors
        torch.cuda.empty_cache()
    vae.to("cpu")
    return result


def _to_device(source: EncodedSource, *, device: Any) -> EncodedSource:
    return EncodedSource(
        source.iid,
        source.clean.to(device=device).contiguous(),
        tuple(value.to(device=device).contiguous() for value in source.references),
        source.tensor_digest,
    )


def _scrambled_condition(clean: Any, *, seed: int) -> tuple[Any, tuple[int, ...]]:
    return anchor_objective.scramble_source_condition(clean, seed=seed)


def _assert_only_source_condition_differs(
    *,
    correct: EncodedSource,
    wrong: EncodedSource,
    correct_condition: Any,
    wrong_condition: Any,
    state: Any,
    target: Any,
    epsilon: Any,
    sigma: Any,
    timestep: Any,
    cond_embeds: Any,
    uncond_embeds: Any,
) -> Mapping[str, Any]:
    """Audit the correct/wrong intervention before either native forward.

    There is exactly one clean endpoint, one epsilon, one sigma/timestep and
    one prompt pair.  The two calls may differ only in the complete native
    source condition (scrambled video plus its four source-derived refs).
    """

    import torch

    sigma_value = float(sigma.detach().item()) if type(sigma) is torch.Tensor and sigma.numel() == 1 else math.nan
    expected_state = (
        (1.0 - sigma_value) * correct.clean.float() + sigma_value * epsilon.float()
        if math.isfinite(sigma_value) and type(epsilon) is torch.Tensor
        else None
    )
    expected_target = (
        epsilon.float() - correct.clean.float()
        if type(epsilon) is torch.Tensor
        else None
    )
    common_ok = (
        type(state) is torch.Tensor
        and type(target) is torch.Tensor
        and type(epsilon) is torch.Tensor
        and type(sigma) is torch.Tensor
        and type(timestep) is torch.Tensor
        and type(cond_embeds) is torch.Tensor
        and type(uncond_embeds) is torch.Tensor
        and tuple(state.shape) == tuple(correct.clean.shape)
        and tuple(target.shape) == tuple(correct.clean.shape)
        and tuple(epsilon.shape) == tuple(correct.clean.shape)
        and sigma.numel() == 1
        and timestep.numel() == 1
        and torch.equal(state.float(), expected_state)
        and torch.equal(target.float(), expected_target)
        and not any(
            tensor.requires_grad or tensor.grad_fn is not None
            for tensor in (
                state,
                target,
                epsilon,
                sigma,
                timestep,
                cond_embeds,
                uncond_embeds,
            )
        )
    )
    condition_differs = (
        correct.iid != wrong.iid
        and tuple(correct_condition.shape) == tuple(wrong_condition.shape)
        and not torch.equal(correct_condition, wrong_condition)
        and len(correct.references) == len(wrong.references) == 4
        and all(
            tuple(left.shape) == tuple(right.shape)
            for left, right in zip(correct.references, wrong.references)
        )
        and any(
            not torch.equal(left, right)
            for left, right in zip(correct.references, wrong.references)
        )
    )
    if not common_ok or not condition_differs:
        raise SAICSourceAnchorTrainingError(
            "correct/wrong must share endpoint/noise/sigma/prompts and differ only in full source+4refs"
        )
    value = {
        "same_clean_source_endpoint": True,
        "same_noisy_target_state": True,
        "same_flow_target": True,
        "same_official_gaussian": True,
        "same_actual_sigma_and_timestep": True,
        "same_conditional_and_unconditional_prompt_embeddings": True,
        "only_complete_scrambled_source_plus_four_refs_differs": True,
        "correct_source_iid": correct.iid,
        "wrong_source_iid": wrong.iid,
    }
    return {**value, "digest": object_sha256(value)}


def _native_rows(pack: native.NativeRV2VPack, *, cond: Any, uncond: Any) -> tuple[tuple[str, Any, Any, float], ...]:
    rows = (
        ("none_uncond", pack.none, uncond, -0.25),
        ("V_uncond", pack.video, uncond, -3.25),
        ("VI_uncond", pack.video_image, uncond, 0.5),
        ("VI_cond", pack.video_image, cond, 4.0),
    )
    if (
        tuple(item[0] for item in rows) != tuple(guidance.guidance_receipt()["forward_order"])
        or any(
            native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS[name] != coefficient
            for name, _, _, coefficient in rows
        )
        or sum(item[3] for item in rows) != 1.0
    ):
        raise SAICSourceAnchorTrainingError("native APG four-field registry differs")
    return rows


def _build_pack(transformer: Any, condition: Any, references: Sequence[Any], state: Any) -> native.NativeRV2VPack:
    import torch

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return native.build_native_rv2v_pack(
            transformer,
            donor_video=condition,
            image_references=references,
            noisy_target=state,
        )


def _forward_branch(
    diffusion: Any,
    branch: native.NativeRV2VBranch,
    *,
    timestep: Any,
    text: Any,
    handle: anchor_adapter.SAICSourceAnchorHandle,
) -> Any:
    route = (
        handle.route(branch=branch, scheduler=diffusion.scheduler, timestep=timestep)
        if branch.name in anchor_adapter.FULL_SOURCE_BRANCHES
        else nullcontext()
    )
    with route:
        return native.forward_native_target_branch(
            diffusion, branch, timestep=timestep, cond_embeds=text
        )


@dataclass(frozen=True)
class DetachedPrediction:
    guided: Any
    components: Mapping[str, Any]


def _guided_prediction_no_grad(
    diffusion: Any,
    transformer: Any,
    *,
    condition: Any,
    references: Sequence[Any],
    state: Any,
    timestep: Any,
    cond_embeds: Any,
    uncond_embeds: Any,
    handle: anchor_adapter.SAICSourceAnchorHandle,
) -> DetachedPrediction:
    import torch

    pack = _build_pack(transformer, condition, references, state)
    components: dict[str, Any] = {}
    with torch.no_grad():
        for name, branch, text, _ in _native_rows(pack, cond=cond_embeds, uncond=uncond_embeds):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                packed = _forward_branch(
                    diffusion, branch, timestep=timestep, text=text, handle=handle
                )
                components[name] = native_bridge._unpack_spatial_velocity(
                    packed, video_shape=state.shape
                ).float().detach()
    guided = sum(
        components[name] * float(coefficient)
        for name, _, _, coefficient in _native_rows(
            pack, cond=cond_embeds, uncond=uncond_embeds
        )
    ).float().detach()
    if tuple(guided.shape) != tuple(state.shape) or not bool(torch.isfinite(guided).all().item()):
        raise SAICSourceAnchorTrainingError("native APG prediction differs")
    return DetachedPrediction(guided, components)


def _serial_prediction_vjp(
    diffusion: Any,
    transformer: Any,
    *,
    condition: Any,
    references: Sequence[Any],
    state: Any,
    timestep: Any,
    cond_embeds: Any,
    uncond_embeds: Any,
    handle: anchor_adapter.SAICSourceAnchorHandle,
    output_cotangent: Any,
    expected: DetachedPrediction,
) -> float:
    import torch

    if (
        tuple(output_cotangent.shape) != tuple(state.shape)
        or output_cotangent.requires_grad
        or not bool(torch.isfinite(output_cotangent).all().item())
    ):
        raise SAICSourceAnchorTrainingError("guided output cotangent differs")
    pack = _build_pack(transformer, condition, references, state)
    maximum = 0.0
    # none_uncond is still part of every no-grad APG prediction, but it has no
    # V/VI route and therefore an identically-zero derivative w.r.t. this adapter.
    for name, branch, text, coefficient in _native_rows(
        pack, cond=cond_embeds, uncond=uncond_embeds
    ):
        if branch.name == "none":
            continue
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed = _forward_branch(
                diffusion, branch, timestep=timestep, text=text, handle=handle
            )
            spatial = native_bridge._unpack_spatial_velocity(
                packed, video_shape=state.shape
            )
        difference = (spatial.detach().float() - expected.components[name]).abs()
        local_max = float(difference.max().item())
        scale = float(expected.components[name].abs().max().item())
        if local_max > VJP_ATOL + VJP_RTOL * scale:
            raise SAICSourceAnchorTrainingError(
                f"serial VJP replay changed {name}: max={local_max} scale={scale}"
            )
        maximum = max(maximum, local_max)
        torch.autograd.backward(
            spatial,
            grad_tensors=output_cotangent.to(spatial.dtype) * float(coefficient),
        )
    return maximum


def _native_coordinate(index: int, *, device: Any) -> tuple[Any, Any]:
    import torch

    anchor_objective.validate_active_sigma_index(index)
    sigma = torch.tensor(
        [native.NATIVE_UNIPC40_SIGMAS[index]], dtype=torch.float32, device=device
    ).detach()
    timestep = torch.tensor(
        [native.NATIVE_UNIPC40_TIMESTEPS[index]], dtype=torch.float32, device=device
    ).detach()
    return sigma, timestep


def _fresh_epsilon(shape: Sequence[int], *, seed: int, device: Any) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device=device).contiguous().detach()


def _attach_text(renderer: Any, tokenizer: Any, *, device: Any, parallel: runtime.ParallelContext) -> tuple[Any, Any, str]:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    negative_ids, negative_mask = native_runtime._tokenize_negative(
        tokenizer, infer_lora.DEFAULT_NEGATIVE_PROMPT
    )
    complete = native_infer.build_task_prompt(
        "rv2v", NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )
    positive_ids, positive_mask = native_runtime._tokenize_positive(tokenizer, complete)
    with __import__("torch").inference_mode():
        cond = renderer.encode_prompt(
            positive_ids.to(device), positive_mask.to(device)
        ).detach()
        uncond = renderer.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
    _broadcast_sp_tensor(cond, parallel=parallel)
    _broadcast_sp_tensor(uncond, parallel=parallel)
    if tuple(cond.shape) != (1, 512, 4096) or tuple(uncond.shape) != (1, 512, 4096) or __import__("torch").equal(cond, uncond):
        raise SAICSourceAnchorTrainingError("frozen no-op text embeddings differ")
    return cond, uncond, hashlib.sha256(complete.encode("utf-8")).hexdigest()


def _sp_world_records(local: Mapping[str, Any], *, parallel: runtime.ParallelContext, label: str) -> list[Mapping[str, Any]]:
    import torch.distributed as dist

    projection = {key: value for key, value in local.items() if key != "sp_rank"}
    runtime.digest_consensus(
        object_sha256(projection),
        group=parallel.sp_group,
        expected_count=SP_SIZE,
        label=label,
    )
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, dict(local), group=parallel.world_group)
    return [gathered[0], gathered[4]]


def _evaluate_heldout(
    rows: Sequence[SourceRow],
    encoded: Mapping[str, EncodedSource],
    *,
    phase: str,
    diffusion: Any,
    transformer: Any,
    handle: anchor_adapter.SAICSourceAnchorHandle,
    cond_embeds: Any,
    uncond_embeds: Any,
    device: Any,
    parallel: runtime.ParallelContext,
    seed: int,
) -> Mapping[str, Any]:
    import torch
    import torch.distributed as dist

    if phase not in {"heldout-before", "heldout-after"} or len(rows) != HOLDOUT_PER_ARM:
        raise SAICSourceAnchorTrainingError("held-out audit contract differs")
    local_rows: list[Mapping[str, Any]] = []
    for local_index, row in enumerate(rows):
        correct = _to_device(encoded[row.iid], device=device)
        wrong = _to_device(encoded[row.wrong_iid], device=device)
        correct_condition, order = _scrambled_condition(
            correct.clean, seed=row.scramble_seed
        )
        wrong_condition, wrong_order = _scrambled_condition(
            wrong.clean, seed=row.scramble_seed
        )
        if order != wrong_order:
            raise SAICSourceAnchorTrainingError("correct/wrong scramble coordinate differs")
        schedule_index = ACTIVE_SIGMA_INDICES[local_index % len(ACTIVE_SIGMA_INDICES)]
        sigma, timestep = _native_coordinate(schedule_index, device=device)
        seed_value = noise_seed(
            seed=seed,
            update_index=row.row_index,
            dp_arm=parallel.contract.arm_index,
            # The same held-out noise is mandatory before and after.
            phase="heldout-before",
        )
        epsilon = _fresh_epsilon(correct.clean.shape, seed=seed_value, device=device)
        _broadcast_sp_tensor(epsilon, parallel=parallel)
        state, target = anchor_objective.build_source_flow_state(
            correct.clean, epsilon, sigma=float(sigma.item())
        )
        intervention_audit = _assert_only_source_condition_differs(
            correct=correct,
            wrong=wrong,
            correct_condition=correct_condition,
            wrong_condition=wrong_condition,
            state=state,
            target=target,
            epsilon=epsilon,
            sigma=sigma,
            timestep=timestep,
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
        )
        correct_prediction = _guided_prediction_no_grad(
            diffusion,
            transformer,
            condition=correct_condition,
            references=correct.references,
            state=state,
            timestep=timestep,
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            handle=handle,
        )
        wrong_prediction = _guided_prediction_no_grad(
            diffusion,
            transformer,
            condition=wrong_condition,
            references=wrong.references,
            state=state,
            timestep=timestep,
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            handle=handle,
        )
        correct_error = float((correct_prediction.guided - target).square().mean().item())
        wrong_error = float((wrong_prediction.guided - target).square().mean().item())
        local_rows.append(
            {
                "iid": row.iid,
                "wrong_iid": row.wrong_iid,
                "dp_arm": parallel.contract.arm_index,
                "schedule_index": schedule_index,
                "noise_seed": seed_value,
                "correct_flow_error": correct_error,
                "wrong_source_flow_error": wrong_error,
                "wrong_source_advantage": wrong_error - correct_error,
                "condition_intervention_audit_digest": intervention_audit["digest"],
            }
        )
        del correct, wrong, correct_condition, wrong_condition, epsilon, state, target
        torch.cuda.empty_cache()
    local_digest = object_sha256(local_rows)
    runtime.digest_consensus(
        local_digest,
        group=parallel.sp_group,
        expected_count=SP_SIZE,
        label=f"{phase} rows",
    )
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local_rows, group=parallel.world_group)
    rows16 = list(gathered[0]) + list(gathered[4])
    if len(rows16) != 16 or len({row["iid"] for row in rows16}) != 16:
        raise SAICSourceAnchorTrainingError("held-out world aggregation differs")
    correct_mean = sum(row["correct_flow_error"] for row in rows16) / len(rows16)
    advantage_mean = sum(row["wrong_source_advantage"] for row in rows16) / len(rows16)
    positive_fraction = sum(row["wrong_source_advantage"] > 0.0 for row in rows16) / len(rows16)
    value = {
        "phase": phase,
        "row_count": len(rows16),
        "one_registered_coordinate_per_row": True,
        "correct_flow_error_mean": correct_mean,
        "wrong_source_advantage_mean": advantage_mean,
        "wrong_source_positive_fraction": positive_fraction,
        "rows": rows16,
    }
    return {**value, "digest": object_sha256(value)}


def heldout_gate(before: Mapping[str, Any], after: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the noncompensating no-op and source-dependence publication gate."""

    for label, value in (("before", before), ("after", after)):
        if (
            not isinstance(value, Mapping)
            or value.get("row_count") != 16
            or not all(
                isinstance(value.get(name), (int, float))
                and math.isfinite(float(value[name]))
                for name in (
                    "correct_flow_error_mean",
                    "wrong_source_advantage_mean",
                    "wrong_source_positive_fraction",
                )
            )
        ):
            raise SAICSourceAnchorTrainingError(f"{label} held-out summary differs")
    no_op_limit = (
        float(before["correct_flow_error_mean"]) * (1.0 + NOOP_RELATIVE_TOLERANCE)
        + NOOP_ABSOLUTE_TOLERANCE
    )
    no_op_pass = float(after["correct_flow_error_mean"]) <= no_op_limit
    dependence_floor = max(0.0, float(before["wrong_source_advantage_mean"]))
    wrong_source_pass = (
        float(after["wrong_source_advantage_mean"]) >= dependence_floor
        and float(after["wrong_source_positive_fraction"])
        >= float(before["wrong_source_positive_fraction"])
        and float(after["wrong_source_positive_fraction"]) >= 0.75
    )
    strict_improvements = {
        "correct_flow_error_decreased": (
            float(after["correct_flow_error_mean"])
            <= float(before["correct_flow_error_mean"])
            - STRICT_FLOW_IMPROVEMENT_EPSILON
        ),
        "wrong_source_advantage_increased": (
            float(after["wrong_source_advantage_mean"])
            >= float(before["wrong_source_advantage_mean"])
            + STRICT_ADVANTAGE_IMPROVEMENT_EPSILON
        ),
        "wrong_source_positive_fraction_increased": (
            float(after["wrong_source_positive_fraction"])
            >= float(before["wrong_source_positive_fraction"])
            + STRICT_POSITIVE_FRACTION_STEP
        ),
    }
    at_least_one_strict = any(strict_improvements.values())
    scientific_pass = no_op_pass and wrong_source_pass and at_least_one_strict
    value = {
        "heldout_before": dict(before),
        "heldout_after": dict(after),
        "no_op_reconstruction_noninferior": no_op_pass,
        "no_op_error_limit": no_op_limit,
        "wrong_source_dependence_noninferior_and_positive": wrong_source_pass,
        "wrong_source_advantage_floor": dependence_floor,
        "minimum_positive_fraction": 0.75,
        "strict_improvement_thresholds": {
            "flow_error_absolute": STRICT_FLOW_IMPROVEMENT_EPSILON,
            "wrong_source_advantage_absolute": STRICT_ADVANTAGE_IMPROVEMENT_EPSILON,
            "positive_fraction_one_heldout_row": STRICT_POSITIVE_FRACTION_STEP,
        },
        "strict_improvements": strict_improvements,
        "at_least_one_strict_improvement": at_least_one_strict,
        "noncompensating_all_pass": scientific_pass,
        # This is only candidate eligibility inside the trainer.  The external
        # terminal postflight remains the sole checkpoint publisher.
        "checkpoint_publication_allowed": scientific_pass,
        "decoded_rgb_preservation_claim": False,
        "action_editing_claim": False,
    }
    return {**value, "digest": object_sha256(value)}


def _atomic_safetensors(
    path: Path,
    *,
    handle: anchor_adapter.SAICSourceAnchorHandle,
    metadata: Mapping[str, str],
) -> Mapping[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    state = dict(handle.state_dict_for_save())
    expected_keys = {name for name, _ in handle.trainable_named_parameters()}
    if set(state) != expected_keys or not state:
        raise SAICSourceAnchorTrainingError("adapter state key closure differs")
    closed_metadata = {
        "schema_version",
        "adapter_schema_version",
        "adapter_contract_digest",
        "state_tensor_sha256",
        "state_key_sha256",
        "optimizer_updates",
        "heldout_gate_digest",
        "source_anchor_only",
        "semantic_action_success",
    }
    if set(metadata) != closed_metadata or any(type(value) is not str for value in metadata.values()):
        raise SAICSourceAnchorTrainingError("safetensors metadata closure differs")
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as file_handle:
            temporary = Path(file_handle.name)
        save_file(state, str(temporary), metadata=dict(metadata))
        descriptor = os.open(temporary, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        os.link(temporary, path, follow_symlinks=False)
        runtime.fsync_directory(path.parent)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    loaded: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        observed_metadata = opened.metadata()
        for name in opened.keys():
            loaded[name] = opened.get_tensor(name).float().contiguous()
    if observed_metadata != dict(metadata) or set(loaded) != set(state) or any(
        not torch.equal(state[name], loaded[name]) for name in state
    ):
        raise SAICSourceAnchorTrainingError("safetensors strict roundtrip differs")
    value = {
        "schema_version": SAFETENSORS_SCHEMA,
        "file_sha256": runtime.file_sha256(path),
        "state_key_count": len(state),
        "state_key_sha256": object_sha256(sorted(state)),
        "state_tensor_sha256": anchor_adapter.trainable_state_digest(state),
        "metadata_closed": True,
        "roundtrip_byte_exact_tensors": True,
    }
    return {**value, "digest": object_sha256(value)}


def _create_only_bytes(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise SAICSourceAnchorTrainingError(f"{label} destination is not fresh")
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SAICSourceAnchorTrainingError(f"{label} write stalled")
            view = view[written:]
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(payload) + 1) != payload:
            raise SAICSourceAnchorTrainingError(f"{label} same-FD reread differs")
        retained = os.fstat(descriptor)
        public = path.lstat()
        if (
            not stat.S_ISREG(retained.st_mode)
            or not stat.S_ISREG(public.st_mode)
            or stat.S_ISLNK(public.st_mode)
            or retained.st_nlink != 1
            or public.st_nlink != 1
            or stat.S_IMODE(retained.st_mode) != 0o600
            or (retained.st_dev, retained.st_ino) != (public.st_dev, public.st_ino)
            or retained.st_size != len(payload)
        ):
            raise SAICSourceAnchorTrainingError(f"{label} identity differs")
    finally:
        os.close(descriptor)
    runtime.fsync_directory(path.parent)


def _create_only_json(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    _create_only_bytes(
        path,
        canonical_json_bytes(value) + b"\n",
        label=label,
    )


def _publish(stage: Path, output: Path) -> None:
    if output.exists() or output.is_symlink() or not stage.is_dir() or stage.is_symlink():
        raise SAICSourceAnchorTrainingError("create-only publication precondition differs")
    parent_fd = os.open(
        output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    output_fd: Optional[int] = None
    try:
        os.mkdir(output.name, mode=0o700, dir_fd=parent_fd)
        output_fd = os.open(
            output.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        for source in sorted(stage.iterdir(), key=lambda value: value.name):
            info = source.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise SAICSourceAnchorTrainingError(
                    f"staged artifact identity differs: {source.name}"
                )
            os.link(
                source.name,
                source.name,
                src_dir_fd=stage_fd,
                dst_dir_fd=output_fd,
                follow_symlinks=False,
            )
        os.fsync(output_fd)
        os.fsync(parent_fd)
        for source in sorted(stage.iterdir(), key=lambda value: value.name):
            os.unlink(source.name, dir_fd=stage_fd)
        os.fsync(stage_fd)
        os.rmdir(stage.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if output_fd is not None:
            os.close(output_fd)
        os.close(stage_fd)
        os.close(parent_fd)
    if (
        not output.is_dir()
        or output.is_symlink()
        or stat.S_IMODE(output.lstat().st_mode) != 0o700
        or any(
            not item.is_file()
            or item.is_symlink()
            or item.lstat().st_nlink != 1
            or stat.S_IMODE(item.lstat().st_mode) != 0o600
            for item in output.iterdir()
        )
    ):
        raise SAICSourceAnchorTrainingError("create-only publication closure differs")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_contract = validate_cli(args)
    invoked_trainer = Path(__file__)
    if (
        not invoked_trainer.is_absolute()
        or invoked_trainer.is_symlink()
        or not invoked_trainer.is_file()
    ):
        raise SAICSourceAnchorTrainingError(
            "invoked trainer must be an absolute plain overlay file"
        )
    _stable_bytes(
        invoked_trainer,
        expected_sha256=args.trainer_source_sha256,
        label="retained trainer source",
    )
    python_executable = _canonical_plain_path(
        args.python_executable, label="Python executable", verify_file=True
    )
    if Path(sys.executable).resolve(strict=True) != python_executable:
        raise SAICSourceAnchorTrainingError("running Python path differs")
    _stable_bytes(
        python_executable,
        expected_sha256=args.python_executable_sha256,
        label="Python executable",
    )
    python_version = subprocess.run(
        [str(python_executable), "--version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    if (
        python_version.returncode != 0
        or hashlib.sha256(python_version.stdout).hexdigest()
        != args.python_version_stdout_sha256
    ):
        raise SAICSourceAnchorTrainingError("running Python version differs")
    if type(args.source_supervisor_pid) is not int or args.source_supervisor_pid <= 1:
        raise SAICSourceAnchorTrainingError("source supervisor PID differs")
    retained_sources = load_retained_source_fd_map(
        os.environ.get("SAIC_ANCHOR_SOURCE_FD_MAP", ""),
        expected_sha256=args.source_fd_map_sha256,
        expected_supervisor_pid=args.source_supervisor_pid,
    )
    for name in (
        "archive_member_manifest_sha256", "extracted_tree_manifest_sha256",
        "archive_binding_receipt_digest", "runtime_origin_manifest_sha256",
        "runtime_origin_receipt_digest",
    ):
        _require_sha(getattr(args, name), length=64, label=name)
    if (
        args.archive_member_manifest_sha256
        != "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
        or args.runtime_origin_manifest_sha256
        != "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
    ):
        raise SAICSourceAnchorTrainingError("archive/runtime origin closure differs")
    manifest = load_manifest(
        args.manifest,
        expected_sha256=args.expected_manifest_sha256,
        verify_files=True,
        retained_sources=retained_sources,
        retained_source_map_sha256=args.source_fd_map_sha256,
    )
    checkpoint_manifest_path = _canonical_plain_path(
        args.checkpoint_content_manifest,
        label="checkpoint content manifest",
        verify_file=True,
    )
    checkpoint_manifest_snapshot = FileSnapshot.capture(
        checkpoint_manifest_path,
        args.expected_checkpoint_content_manifest_sha256,
        label="checkpoint content manifest",
    )
    try:
        checkpoint_content_identity = checkpoint_audit.validate_checkpoint_content(
            Path(args.checkpoint),
            checkpoint_manifest_path,
            expected_manifest_sha256=(
                args.expected_checkpoint_content_manifest_sha256
            ),
        )
    except Exception as error:
        raise SAICSourceAnchorTrainingError(
            f"checkpoint content validation failed: {error}"
        ) from error
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "manifest_digest": manifest.manifest_digest,
                    "train_count": len(manifest.train_rows),
                    "holdout_count": len(manifest.holdout_rows),
                    "checkpoint_content_identity": checkpoint_content_identity,
                    "optimizer_created": False,
                    "source_anchor_only": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise SAICSourceAnchorTrainingError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise SAICSourceAnchorTrainingError("pinned Bernini head count differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = runtime.distributed_contract()
    device = runtime.initialise_distributed(distributed)
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=SP_SIZE)
    )
    output, stage = runtime.prepare_output_transaction(
        args.output, distributed.rank, parallel.world_group
    )
    if distributed.rank == 0:
        os.chmod(stage, 0o700)
        runtime.fsync_directory(stage.parent)
    dist.barrier(group=parallel.world_group)
    if stat.S_IMODE(stage.lstat().st_mode) != 0o700:
        raise SAICSourceAnchorTrainingError("private training stage mode differs")
    legacy.seed_same_sample(args.seed)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **infer_lora.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).eval().requires_grad_(False)
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise SAICSourceAnchorTrainingError("source anchor requires transformer_1 only")
    disable = getattr(renderer, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    if bool(getattr(transformer, "gradient_checkpointing", False)) or bool(
        getattr(transformer, "is_gradient_checkpointing", False)
    ):
        raise SAICSourceAnchorTrainingError("gradient checkpointing remains enabled")
    handle = anchor_adapter.install_saic_source_anchor_adapter(transformer)
    trainable = handle.trainable_named_parameters()
    if not handle.base_parameters_frozen() or not handle.scope_untouched():
        raise SAICSourceAnchorTrainingError("source-anchor trainability closure differs")

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False)
    encoded = _encode_assigned_sources(
        manifest, vae=vae, device=device, parallel=parallel
    )
    del vae
    torch.cuda.empty_cache()

    renderer.to(device)
    renderer.eval()
    schedule_receipt = sigma_strata.audit_runtime_unipc_schedule(
        diffusion.scheduler, initialize=True
    )
    if schedule_receipt["schedule_sha256"] != sigma_strata.SCHEDULE_SHA256:
        raise SAICSourceAnchorTrainingError("actual exact40 scheduler differs")
    initial_digest = runtime.synchronize_initial_parameters(
        trainable, parallel.world_group
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    cond_embeds, uncond_embeds, prompt_sha256 = _attach_text(
        renderer, tokenizer, device=device, parallel=parallel
    )
    renderer.t5_text_encoder.to("cpu")
    del tokenizer
    torch.cuda.empty_cache()

    heldout_rows = manifest.rows_for_arm(
        split="holdout", arm=distributed.arm_index
    )
    before = _evaluate_heldout(
        heldout_rows,
        encoded,
        phase="heldout-before",
        diffusion=diffusion,
        transformer=transformer,
        handle=handle,
        cond_embeds=cond_embeds,
        uncond_embeds=uncond_embeds,
        device=device,
        parallel=parallel,
        seed=args.seed,
    )

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    training_rows = manifest.rows_for_arm(split="train", arm=distributed.arm_index)
    history: list[Mapping[str, Any]] = []
    for update_index in range(args.max_updates):
        row = training_rows[update_index % len(training_rows)]
        correct = _to_device(encoded[row.iid], device=device)
        wrong = _to_device(encoded[row.wrong_iid], device=device)
        correct_condition, order = _scrambled_condition(
            correct.clean, seed=row.scramble_seed
        )
        wrong_condition, wrong_order = _scrambled_condition(
            wrong.clean, seed=row.scramble_seed
        )
        if order != wrong_order:
            raise SAICSourceAnchorTrainingError("training scramble coordinates differ")
        schedule_index = schedule_index_for_update(update_index)
        sigma, timestep = _native_coordinate(schedule_index, device=device)
        seed_value = noise_seed(
            seed=args.seed,
            update_index=update_index,
            dp_arm=distributed.arm_index,
            phase="train",
        )
        epsilon = _fresh_epsilon(correct.clean.shape, seed=seed_value, device=device)
        _broadcast_sp_tensor(epsilon, parallel=parallel)
        state, target = anchor_objective.build_source_flow_state(
            correct.clean, epsilon, sigma=float(sigma.item())
        )
        intervention_audit = _assert_only_source_condition_differs(
            correct=correct,
            wrong=wrong,
            correct_condition=correct_condition,
            wrong_condition=wrong_condition,
            state=state,
            target=target,
            epsilon=epsilon,
            sigma=sigma,
            timestep=timestep,
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
        )
        optimizer.zero_grad(set_to_none=True)
        correct_detached = _guided_prediction_no_grad(
            diffusion,
            transformer,
            condition=correct_condition,
            references=correct.references,
            state=state,
            timestep=timestep,
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            handle=handle,
        )
        wrong_detached = _guided_prediction_no_grad(
            diffusion,
            transformer,
            condition=wrong_condition,
            references=wrong.references,
            state=state,
            timestep=timestep,
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            handle=handle,
        )
        correct_leaf = correct_detached.guided.clone().requires_grad_(True)
        wrong_leaf = wrong_detached.guided.clone().requires_grad_(True)
        objective = anchor_objective.build_source_anchor_objective(
            correct_source_prediction=correct_leaf,
            wrong_source_prediction=wrong_leaf,
            source_flow_target=target,
            wrong_source_margin=args.wrong_source_margin,
            ranking_weight=args.ranking_weight,
        )
        finite = bool(torch.isfinite(objective.loss.detach()).item())
        if not runtime.world_all_true(finite, group=parallel.world_group):
            raise SAICSourceAnchorTrainingError("non-finite objective blocked update")
        objective.loss.backward()
        if correct_leaf.grad is None or wrong_leaf.grad is None:
            raise SAICSourceAnchorTrainingError("prediction leaves lack cotangents")
        replay_max = _serial_prediction_vjp(
            diffusion,
            transformer,
            condition=correct_condition,
            references=correct.references,
            state=state,
            timestep=timestep,
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            handle=handle,
            output_cotangent=correct_leaf.grad.detach(),
            expected=correct_detached,
        )
        replay_max = max(
            replay_max,
            _serial_prediction_vjp(
                diffusion,
                transformer,
                condition=wrong_condition,
                references=wrong.references,
                state=state,
                timestep=timestep,
                cond_embeds=cond_embeds,
                uncond_embeds=uncond_embeds,
                handle=handle,
                output_cotangent=wrong_leaf.grad.detach(),
                expected=wrong_detached,
            ),
        )
        grad_norm = runtime.synchronize_gradients(trainable, parallel)
        clipped = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], args.max_grad_norm
        )
        if not math.isfinite(float(clipped)):
            raise SAICSourceAnchorTrainingError("gradient clipping is non-finite")
        optimizer.step()
        parameter_digest = runtime.parameter_consensus(
            trainable,
            parallel.world_group,
            f"SAIC source-anchor update {update_index + 1}",
        )
        local = {
            "update_index": update_index,
            "dp_arm": distributed.arm_index,
            "sp_rank": distributed.sp_rank,
            "iid": row.iid,
            "wrong_iid": row.wrong_iid,
            "row_digest": row.row_digest,
            "schedule_index": schedule_index,
            "noise_seed": seed_value,
            "scramble_indices": list(order),
            "condition_intervention_audit_digest": intervention_audit["digest"],
            "loss": float(objective.loss.detach().item()),
            "correct_flow_loss": float(objective.correct_flow_loss.detach().item()),
            "wrong_source_flow_loss": float(objective.wrong_source_flow_loss.detach().item()),
            "wrong_source_advantage": float(objective.wrong_source_advantage.detach().item()),
            "ranking_hinge": float(objective.ranking_hinge.detach().item()),
            "preclip_gradient_norm_world_average": grad_norm,
            "vjp_replay_max_abs": replay_max,
            "four_native_apg_fields_evaluated_per_condition": True,
            "three_source_conditioned_fields_replayed_for_vjp": True,
            "none_field_adapter_vjp_exact_zero": True,
            "parameter_digest_after": parameter_digest,
        }
        dp_records = _sp_world_records(
            local, parallel=parallel, label=f"source-anchor update {update_index}"
        )
        history.append({"update_index": update_index, "dp_records": dp_records})
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "update": update_index + 1,
                        "schedule_index": schedule_index,
                        "loss_dp0": dp_records[0]["loss"],
                        "loss_dp1": dp_records[1]["loss"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del (
            correct,
            wrong,
            correct_condition,
            wrong_condition,
            epsilon,
            state,
            target,
            correct_detached,
            wrong_detached,
            correct_leaf,
            wrong_leaf,
            objective,
        )
        torch.cuda.empty_cache()

    final_digest = runtime.parameter_consensus(
        trainable, parallel.world_group, "SAIC source-anchor final"
    )
    if final_digest == initial_digest:
        raise SAICSourceAnchorTrainingError("optimizer left the adapter unchanged")
    after = _evaluate_heldout(
        heldout_rows,
        encoded,
        phase="heldout-after",
        diffusion=diffusion,
        transformer=transformer,
        handle=handle,
        cond_embeds=cond_embeds,
        uncond_embeds=uncond_embeds,
        device=device,
        parallel=parallel,
        seed=args.seed,
    )
    gate = heldout_gate(before, after)
    manifest.assert_unchanged()
    checkpoint_manifest_snapshot.assert_unchanged()
    try:
        final_checkpoint_content_identity = (
            checkpoint_audit.validate_checkpoint_content(
                Path(args.checkpoint),
                checkpoint_manifest_path,
                expected_manifest_sha256=(
                    args.expected_checkpoint_content_manifest_sha256
                ),
            )
        )
    except Exception as error:
        raise SAICSourceAnchorTrainingError(
            f"checkpoint content changed during training: {error}"
        ) from error
    if object_sha256(final_checkpoint_content_identity) != object_sha256(
        checkpoint_content_identity
    ):
        raise SAICSourceAnchorTrainingError(
            "checkpoint content identity changed during training"
        )
    dist.barrier(group=parallel.world_group)

    if distributed.rank == 0:
        history_unsigned = {
            "schema_version": HISTORY_SCHEMA,
            "complete": True,
            "optimizer_updates": args.max_updates,
            "update_indices": list(range(args.max_updates)),
            "rows": history,
        }
        history_object = {
            **history_unsigned,
            "history_digest": object_sha256(history_unsigned),
        }
        history_path = stage / "history.json"
        _create_only_json(history_path, history_object, label="training history")
        adapter_roundtrip: Optional[Mapping[str, Any]] = None
        if gate["noncompensating_all_pass"] is True and args.mode == "formal":
            state = handle.state_dict_for_save()
            metadata = {
                "schema_version": SAFETENSORS_SCHEMA,
                "adapter_schema_version": anchor_adapter.SCHEMA_VERSION,
                "adapter_contract_digest": str(handle.receipt()["digest"]),
                "state_tensor_sha256": anchor_adapter.trainable_state_digest(state),
                "state_key_sha256": object_sha256(sorted(state)),
                "optimizer_updates": str(args.max_updates),
                "heldout_gate_digest": str(gate["digest"]),
                "source_anchor_only": "true",
                "semantic_action_success": "false",
            }
            adapter_roundtrip = _atomic_safetensors(
                stage / "adapter.safetensors", handle=handle, metadata=metadata
            )
        scientific_status = (
            "FORMAL_GATE_PASS_CHECKPOINT_CANDIDATE"
            if adapter_roundtrip is not None
            else "OPERATIONAL_COMPLETED_SCIENTIFIC_NO_GO"
        )
        receipt: dict[str, Any] = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "complete": True,
            "status": scientific_status,
            "run_contract": dict(run_contract),
            "manifest": {
                "path": str(manifest.snapshot.path),
                "file_sha256": manifest.snapshot.sha256,
                "manifest_digest": manifest.manifest_digest,
                "train_count": len(manifest.train_rows),
                "holdout_count": len(manifest.holdout_rows),
                "holdout_used_by_optimizer": False,
                "manifest_and_all_80_source_files_post_training_mutation_audit_passed": True,
                "source_video_count": 80,
                "source_fd_map_sha256": args.source_fd_map_sha256,
                "source_supervisor_pid": args.source_supervisor_pid,
                "all_source_videos_read_only_via_parent_retained_proc_fds": True,
                "each_encoded_source_rehashed_before_and_after_complete_decode": True,
            },
            "source_archive_closure": {
                "archive_member_manifest_sha256": (
                    args.archive_member_manifest_sha256
                ),
                "extracted_tree_manifest_sha256": (
                    args.extracted_tree_manifest_sha256
                ),
                "archive_member_count": 864,
                "archive_regular_file_count": 853,
                "archive_directory_count": 11,
                "extracted_tree_manifest_source": "actual_lstat_after_extraction",
                "archive_binding_receipt_digest": (
                    args.archive_binding_receipt_digest
                ),
                "runtime_origin_project_module_count": 14,
                "runtime_origin_manifest_sha256": (
                    args.runtime_origin_manifest_sha256
                ),
                "runtime_origin_receipt_digest": (
                    args.runtime_origin_receipt_digest
                ),
                "runtime_import_origins_all_from_extracted_archive": True,
            },
            "native_runtime": {
                "mode": "RV2V-4",
                "four_apg_fields": list(guidance.guidance_receipt()["forward_order"]),
                "guidance_digest": guidance.guidance_receipt()["digest"],
                "schedule_sha256": schedule_receipt["schedule_sha256"],
                "active_sigma_indices": list(ACTIVE_SIGMA_INDICES),
                "condition_video": "deterministic_21_phase_scramble",
                "clean_endpoint": "real_unscrambled_source",
                "reference_rgb_indices": list(REFERENCE_INDICES),
                "gradient_checkpointing_enabled": False,
                "serial_output_leaf_vjp": True,
                "correct_wrong_common_random_coordinate_audit": {
                    "same_clean_source_endpoint": True,
                    "same_noisy_target_and_flow_target": True,
                    "same_official_gaussian": True,
                    "same_actual_sigma_timestep": True,
                    "same_conditional_unconditional_prompt_embeddings": True,
                    "only_complete_source_video_plus_four_refs_condition_differs": True,
                },
            },
            "objective": {
                "schema_version": anchor_objective.SCHEMA_VERSION,
                "correct_source_flow_matching": True,
                "wrong_source_ranking": True,
                "wrong_source_margin": args.wrong_source_margin,
                "ranking_weight": args.ranking_weight,
                "no_op_instruction_sha256": prompt_sha256,
            },
            "adapter": {
                **dict(handle.receipt()),
                "initial_parameter_digest": initial_digest,
                "final_parameter_digest": final_digest,
                "changed_by_optimizer": True,
                "checkpoint_candidate_materialized": adapter_roundtrip is not None,
                "checkpoint_published": False,
                "safetensors_roundtrip": adapter_roundtrip,
            },
            "heldout_gate": gate,
            "scientific_limitations": {
                "four_source_references_carry_original_pose": True,
                "pose_lock_shortcut_risk": True,
                "stage_a_gate_scope": (
                    "heldout_noop_flow_noninferiority_and_correct_vs_wrong_source_dependence_only"
                ),
                "stage_a_may_authorize_action_training": False,
                "future_action_stage_requires_fresh_rollout_nonregression": True,
                "future_action_stage_must_test_action_and_identity_camera_background_separately": True,
            },
            "artifacts": {
                "history.json": runtime.file_sha256(history_path),
                **(
                    {"adapter.safetensors": runtime.file_sha256(stage / "adapter.safetensors")}
                    if adapter_roundtrip is not None
                    else {}
                ),
            },
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                "checkpoint_content_manifest_path": str(
                    checkpoint_manifest_snapshot.path
                ),
                "checkpoint_content_manifest_file_sha256": (
                    checkpoint_manifest_snapshot.sha256
                ),
                "checkpoint_content_identity": checkpoint_content_identity,
                "checkpoint_content_post_training_revalidated": True,
                "single_expert": "transformer_1",
            },
            "runtime": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
                "python_executable": str(python_executable),
                "python_executable_sha256": args.python_executable_sha256,
                "python_version_stdout_sha256": args.python_version_stdout_sha256,
                "slurm_job_id": args.slurm_job_id,
                "submission_receipt_sha256": args.submission_receipt_sha256,
                "submission_receipt_digest": args.submission_receipt_digest,
                "release_manifest_sha256": args.release_manifest_sha256,
                "release_manifest_digest": args.release_manifest_digest,
            },
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
            "trainer_source_sha256": args.trainer_source_sha256,
            "formal_full60_admission_sha256": (
                args.formal_full60_admission_sha256
            ),
            "formal_full60_admission_digest": (
                args.formal_full60_admission_digest
            ),
            "source_anchor_pretext_only": True,
            "action_training": False,
            "semantic_action_editing_success": False,
            "decoded_rgb_appearance_preservation_success": False,
            "source_anchor_checkpoint_candidate_eligible": (
                gate["noncompensating_all_pass"] is True and args.mode == "formal"
            ),
            "source_anchor_checkpoint_publication_authorized": False,
            "action_stage_authorized": False,
            "semantic_action_authorized": False,
            "decoded_rgb_identity_authorized": False,
            "stage_a_checkpoint_release_requires_external_terminal_postflight": True,
            "smoke_incomplete_row_coverage": args.mode == "smoke",
        }
        receipt["receipt_digest"] = runtime.object_sha256(receipt)
        _create_only_json(
            stage / "receipt.json", receipt, label="training run receipt"
        )
        expected_files = {"history.json", "receipt.json"} | (
            {"adapter.safetensors"} if adapter_roundtrip is not None else set()
        )
        if {path.name for path in stage.iterdir()} != expected_files or any(
            not path.is_file() or path.is_symlink() for path in stage.iterdir()
        ):
            raise SAICSourceAnchorTrainingError("staged artifact closure differs")
        _publish(stage, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "optimizer_updates": args.max_updates,
                    "heldout_gate_passed": gate["noncompensating_all_pass"],
                    "checkpoint_candidate_materialized": adapter_roundtrip is not None,
                    "checkpoint_published": False,
                    "status": scientific_status,
                    "action_training": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=parallel.world_group)
    if not output.is_dir() or output.is_symlink() or stage.exists():
        raise SAICSourceAnchorTrainingError("atomic publication did not complete")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
