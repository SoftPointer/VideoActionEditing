#!/usr/bin/env python3
"""Train PAIR-v5 Action-LoRA with native exact81 reference-corrected flow DPO.

The optimizer boundary is deliberately narrow.  It accepts only a hash-pinned
safe-Pareto winner/loser manifest, the common exact81 source, four source RGB
references (frames 0/27/53/80), and one complete edit instruction.  T2V
proposals are calibration provenance only: proposal pixels/latents/noise,
paired targets, donors, masks, flow, pose, tracks, and trajectories have no
schema or model-call slot.

Indices 0..37 of Bernini's native exact40 schedule perform preference updates.
Indices 38 and 39 are explicit zero-update frozen-anchor audits: no flow-DPO
loss is constructed and ``optimizer.step`` is not called.  The only trainable
weights are the independent cross-attention Q/O Action-LoRA factors; Bernini,
the required CIO self-attention adapter, and the reference policy stay frozen.

This executable is an experiment mechanism, not evidence that action editing
works.  A complete receipt proves only that the sealed update ran as specified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native_infer  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import pair_v5_action_adapter as action_adapter  # noqa: E402
import pair_v5_action_energy_calibration as calibration  # noqa: E402
import pair_v5_flow_dpo as flow_dpo  # noqa: E402
import pair_v5_native_bridge as native_bridge  # noqa: E402
import pair_v5_native_rollout_spec as rollout_spec  # noqa: E402
import pair_v5_safe_pareto as safe_pareto  # noqa: E402
import source_self_native_ref_contrastive_v3 as native  # noqa: E402
import source_self_native_rv2v_guidance as guidance  # noqa: E402
import source_self_runtime as runtime  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_pair_v5_action_preference as native_runtime  # noqa: E402


METHOD_NAME = "bernini-pair-v5-native-reference-corrected-flow-dpo-v2"
MANIFEST_SCHEMA = "bernini-pair-v5-native-flow-dpo-manifest-v1"
PAIR_SCHEMA = "bernini-pair-v5-native-flow-dpo-pair-v1"
CANDIDATE_MEDIA_SCHEMA = "bernini-pair-v5-native-flow-dpo-media-v1"
RUN_RECEIPT_SCHEMA = "bernini-pair-v5-native-flow-dpo-run-receipt-v2"
HISTORY_SCHEMA = "bernini-pair-v5-native-flow-dpo-history-v2"
ADAPTER_FILE_SCHEMA = "bernini-pair-v5-action-lora-checkpoint-v1"

WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FRAME_COUNT = 81
FPS = 25.0
LATENT_PHASES = 21
LATENT_CHANNELS = 16
REFERENCE_INDICES = (0, 27, 53, 80)
MIN_PAIRS = 1
MAX_PAIRS = 4
EXACT40_STEPS = 40
TRAINABLE_SIGMA_INDICES = (
    action_adapter.HIGH_SIGMA_INDICES + action_adapter.MID_SIGMA_INDICES
)
AUDIT_SIGMA_INDICES = action_adapter.LOW_SIGMA_INDICES
DEFAULT_BETA = 1000.0
DEFAULT_LEARNING_RATE = 1.0e-6
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 20260808

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "optimizer_authorized",
        "action_calibration",
        "pair_count",
        "pairs",
        "input_closure",
        "manifest_digest",
    }
)
_CALIBRATION_FIELDS = frozenset(
    {
        "receipt_path",
        "receipt_sha256",
        "registered_receipt_digest",
        "optimizer_provenance",
    }
)
_PAIR_FIELDS = frozenset(
    {
        "schema_version",
        "pair_id",
        "source_video_path",
        "source_video_sha256",
        "source_frame_count",
        "source_fps",
        "source_reference_indices",
        "instruction",
        "instruction_sha256",
        "selector_policy",
        "selector_state_before",
        "selector_candidates",
        "selector_calibrator_provenance",
        "selector_receipt",
        "winner",
        "loser",
        "pair_digest",
    }
)
_MEDIA_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "candidate_digest",
        "artifact_kind",
        "artifact_path",
        "artifact_sha256",
        "tensor_key",
        "latent_shape",
        "native_rollout_receipt_path",
        "native_rollout_receipt_sha256",
        "native_rollout_receipt_digest",
        "media_digest",
    }
)
_INPUT_CLOSURE = {
    "accepted": [
        "sealed_safe_pareto_winner_clean",
        "sealed_safe_pareto_loser_clean",
        "source_video",
        "source_frames_0_27_53_80",
        "complete_edit_instruction",
    ],
    "t2v_proposal_media": False,
    "proposal_latent_or_noise": False,
    "paired_target": False,
    "motion_donor": False,
    "mask": False,
    "flow": False,
    "pose": False,
    "track": False,
    "trajectory": False,
    "custom_target_noise": False,
}


class PairV5NativeDPOTrainingError(RuntimeError):
    """A sealed input or native training operation violated PAIR-v5."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV5NativeDPOTrainingError(
            "value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV5NativeDPOTrainingError(f"{label} must be an object")
    actual = set(value)
    if not all(isinstance(key, str) for key in actual) or actual != set(fields):
        raise PairV5NativeDPOTrainingError(
            f"{label} keys differ: missing={sorted(set(fields) - actual)} "
            f"extra={sorted(actual - set(fields))}"
        )
    return value


def _sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1 if length == 40 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        kind = "1" if length == 40 else "256"
        raise PairV5NativeDPOTrainingError(f"{label} must be lowercase SHA-{kind}")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PairV5NativeDPOTrainingError(f"{label} must be a safe identifier")
    return value


def _plain_absolute_path(value: Any, *, label: str, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PairV5NativeDPOTrainingError(f"{label} must be a path string")
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path("/"):
        raise PairV5NativeDPOTrainingError(f"{label} must be absolute and non-root")
    if not must_exist:
        return path
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PairV5NativeDPOTrainingError(f"{label} is unavailable") from error
    if resolved != path or path.is_symlink() or not path.is_file():
        raise PairV5NativeDPOTrainingError(
            f"{label} must be a canonical plain file"
        )
    return path


def _read_stable(path: Path, *, expected_sha256: str, label: str) -> bytes:
    expected = _sha(expected_sha256, length=64, label=f"{label} SHA-256")
    if path.is_symlink() or not path.is_file():
        raise PairV5NativeDPOTrainingError(f"{label} must be a plain file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    before_id = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_id != after_id or len(raw) != before.st_size:
        raise PairV5NativeDPOTrainingError(f"{label} changed while reading")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise PairV5NativeDPOTrainingError(f"{label} SHA-256 differs")
    return raw


def _strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise PairV5NativeDPOTrainingError(f"{label} contains {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PairV5NativeDPOTrainingError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5NativeDPOTrainingError(f"cannot decode {label}") from error
    if not isinstance(value, Mapping):
        raise PairV5NativeDPOTrainingError(f"{label} root must be an object")
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
    def capture(
        cls, path: Path, *, expected_sha256: str, label: str
    ) -> "FileSnapshot":
        _read_stable(path, expected_sha256=expected_sha256, label=label)
        stat = path.stat()
        return cls(
            path,
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            expected_sha256,
        )

    @classmethod
    def unchecked(cls, path: Path, sha256: str) -> "FileSnapshot":
        return cls(path, 0, 0, 0, 0, sha256)

    def assert_unchanged(self) -> None:
        if self.device == 0 and self.inode == 0:
            raise PairV5NativeDPOTrainingError("unchecked snapshot cannot be audited")
        current = self.path.stat()
        identity = (
            int(current.st_dev),
            int(current.st_ino),
            int(current.st_size),
            int(current.st_mtime_ns),
        )
        if identity != (self.device, self.inode, self.size, self.mtime_ns):
            raise PairV5NativeDPOTrainingError(f"input changed: {self.path}")
        if runtime.file_sha256(self.path) != self.sha256:
            raise PairV5NativeDPOTrainingError(f"input bytes changed: {self.path}")

    def receipt(self) -> Mapping[str, Any]:
        return {
            "path": str(self.path),
            "size": self.size,
            "sha256": self.sha256,
            "pre_post_stat_and_hash_stable": True,
        }


@dataclass(frozen=True)
class CandidateMedia:
    candidate_id: str
    candidate_digest: str
    artifact_kind: str
    artifact: FileSnapshot
    tensor_key: Optional[str]
    latent_shape: Optional[tuple[int, int, int, int, int]]
    rollout_receipt: FileSnapshot
    rollout_receipt_digest: str
    media_digest: str


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    source: FileSnapshot
    instruction: str
    instruction_sha256: str
    winner: CandidateMedia
    loser: CandidateMedia
    selected_pair_digest: str
    selector_receipt_digest: str
    pair_digest: str


@dataclass(frozen=True)
class PairManifest:
    path: Path
    raw_sha256: str
    manifest_digest: str
    calibration_receipt: FileSnapshot
    calibration_receipt_digest: str
    calibration_provenance_digest: str
    pairs: tuple[PreferencePair, ...]
    snapshots: tuple[FileSnapshot, ...]

    def assert_unchanged(self) -> None:
        for snapshot in self.snapshots:
            snapshot.assert_unchanged()


def _validate_media(
    value: Any,
    *,
    endpoint: Mapping[str, str],
    source_path: Path,
    source_sha256: str,
    instruction: str,
    instruction_sha256: str,
    verify_files: bool,
    verify_tensor_headers: bool,
) -> CandidateMedia:
    row = _closed(value, _MEDIA_FIELDS, label="candidate media")
    if row["schema_version"] != CANDIDATE_MEDIA_SCHEMA:
        raise PairV5NativeDPOTrainingError("candidate media schema differs")
    candidate_id = _safe_id(row["candidate_id"], label="candidate_id")
    candidate_digest = _sha(
        row["candidate_digest"], length=64, label="candidate_digest"
    )
    if (
        candidate_id != endpoint["candidate_id"]
        or candidate_digest != endpoint["candidate_digest"]
    ):
        raise PairV5NativeDPOTrainingError(
            "media does not bind the selected endpoint"
        )
    kind = row["artifact_kind"]
    if kind not in {
        "normalized_clean_latent_safetensors",
        "exact81_mp4",
    }:
        raise PairV5NativeDPOTrainingError(
            "candidate artifact kind is not registered"
        )
    artifact_path = _plain_absolute_path(
        row["artifact_path"], label="candidate artifact", must_exist=verify_files
    )
    artifact_sha = _sha(row["artifact_sha256"], length=64, label="artifact SHA")
    receipt_path = _plain_absolute_path(
        row["native_rollout_receipt_path"],
        label="native rollout receipt",
        must_exist=verify_files,
    )
    receipt_sha = _sha(
        row["native_rollout_receipt_sha256"],
        length=64,
        label="native rollout receipt SHA",
    )
    receipt_digest = _sha(
        row["native_rollout_receipt_digest"],
        length=64,
        label="native rollout receipt digest",
    )
    if kind == "normalized_clean_latent_safetensors":
        if row["tensor_key"] != "normalized_clean_latent":
            raise PairV5NativeDPOTrainingError("clean latent tensor key differs")
        shape = row["latent_shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 5
            or any(type(item) is not int for item in shape)
            or tuple(shape[:3]) != (1, LATENT_CHANNELS, LATENT_PHASES)
            or shape[3] <= 0
            or shape[4] <= 0
            or shape[3] % 2
            or shape[4] % 2
        ):
            raise PairV5NativeDPOTrainingError(
                "clean latent geometry is not exact81"
            )
        tensor_key: Optional[str] = "normalized_clean_latent"
        latent_shape: Optional[tuple[int, int, int, int, int]] = tuple(shape)
    else:
        if row["tensor_key"] is not None or row["latent_shape"] is not None:
            raise PairV5NativeDPOTrainingError(
                "MP4 media cannot declare latent fields"
            )
        tensor_key = None
        latent_shape = None
    unsigned = dict(row)
    embedded = _sha(unsigned.pop("media_digest"), length=64, label="media digest")
    if embedded != object_sha256(unsigned):
        raise PairV5NativeDPOTrainingError(
            "candidate media embedded digest differs"
        )
    if not verify_files:
        return CandidateMedia(
            candidate_id,
            candidate_digest,
            kind,
            FileSnapshot.unchecked(artifact_path, artifact_sha),
            tensor_key,
            latent_shape,
            FileSnapshot.unchecked(receipt_path, receipt_sha),
            receipt_digest,
            embedded,
        )
    artifact_snapshot = FileSnapshot.capture(
        artifact_path, expected_sha256=artifact_sha, label="candidate artifact"
    )
    receipt_snapshot = FileSnapshot.capture(
        receipt_path,
        expected_sha256=receipt_sha,
        label="native rollout receipt",
    )
    receipt = _strict_json(
        _read_stable(
            receipt_path,
            expected_sha256=receipt_sha,
            label="native rollout receipt",
        ),
        label="native rollout receipt",
    )
    declared = _sha(
        receipt.get("receipt_digest"),
        length=64,
        label="native rollout receipt digest",
    )
    receipt_unsigned = dict(receipt)
    receipt_unsigned.pop("receipt_digest")
    if declared != receipt_digest or object_sha256(receipt_unsigned) != receipt_digest:
        raise PairV5NativeDPOTrainingError(
            "native rollout receipt embedded digest differs"
        )
    if receipt.get("schema_version") != rollout_spec.RECEIPT_SCHEMA_VERSION:
        raise PairV5NativeDPOTrainingError("native rollout receipt schema differs")
    if receipt.get("sampling_contract") != rollout_spec.SAMPLING_CONTRACT:
        raise PairV5NativeDPOTrainingError(
            "native rollout sampling contract differs"
        )
    if receipt.get("semantic_input_closure") != rollout_spec.SEMANTIC_INPUT_CLOSURE:
        raise PairV5NativeDPOTrainingError(
            "native rollout admits a forbidden input"
        )
    candidate_value = receipt.get("candidate")
    if not isinstance(candidate_value, Mapping):
        raise PairV5NativeDPOTrainingError(
            "native rollout candidate binding is absent"
        )
    try:
        rollout_candidate = rollout_spec.validate_candidate(candidate_value)
    except Exception as error:
        raise PairV5NativeDPOTrainingError(
            "native rollout candidate binding is invalid"
        ) from error
    if (
        rollout_candidate["candidate_id"] != candidate_id
        or rollout_candidate["source_video"] != str(source_path)
        or rollout_candidate["source_video_sha256"] != source_sha256
        or rollout_candidate["complete_caption"] != instruction
        or rollout_candidate["complete_caption_sha256"] != instruction_sha256
        or rollout_candidate["caption_contract"] != rollout_spec.CAPTION_CONTRACT
    ):
        raise PairV5NativeDPOTrainingError(
            "native rollout source/instruction/candidate binding differs"
        )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PairV5NativeDPOTrainingError(
            "native rollout artifact binding is absent"
        )
    artifact_key = (
        "predecode_clean_latent"
        if kind == "normalized_clean_latent_safetensors"
        else "mp4"
    )
    bound_artifact = artifacts.get(artifact_key)
    if (
        not isinstance(bound_artifact, Mapping)
        or bound_artifact.get("path") != str(artifact_path)
        or bound_artifact.get("sha256") != artifact_sha
    ):
        raise PairV5NativeDPOTrainingError(
            "candidate artifact is not bound by its native rollout receipt"
        )
    if kind == "normalized_clean_latent_safetensors" and (
        bound_artifact.get("tensor_key") != tensor_key
        or bound_artifact.get("shape") != list(latent_shape or ())
        or bound_artifact.get("stored_dtype") != "torch.float32"
        or bound_artifact.get("native_sampler_before_vae_decode") is not True
        or bound_artifact.get("mp4_decode_reencode_used") is not False
    ):
        raise PairV5NativeDPOTrainingError(
            "native predecode clean-latent provenance differs"
        )
    if verify_tensor_headers and kind == "normalized_clean_latent_safetensors":
        from safetensors import safe_open

        with safe_open(str(artifact_path), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [tensor_key]:
                raise PairV5NativeDPOTrainingError(
                    "clean latent file key closure differs"
                )
            tensor = opened.get_tensor(str(tensor_key))
        if (
            str(tensor.dtype) != "torch.float32"
            or tuple(int(item) for item in tensor.shape) != latent_shape
            or not tensor.is_contiguous()
        ):
            raise PairV5NativeDPOTrainingError(
                "clean latent file header differs"
            )
    return CandidateMedia(
        candidate_id,
        candidate_digest,
        kind,
        artifact_snapshot,
        tensor_key,
        latent_shape,
        receipt_snapshot,
        receipt_digest,
        embedded,
    )


def _ffprobe_exact81(path: Path) -> Mapping[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        streams = json.loads(result.stdout)["streams"]
    except (OSError, subprocess.SubprocessError, KeyError, json.JSONDecodeError) as error:
        raise PairV5NativeDPOTrainingError(
            f"cannot probe exact81 video {path}"
        ) from error
    if len(streams) != 1:
        raise PairV5NativeDPOTrainingError("video stream closure differs")
    stream = streams[0]
    if (
        int(stream.get("nb_read_frames", -1)) != FRAME_COUNT
        or stream.get("avg_frame_rate") != "25/1"
        or int(stream.get("width", 0)) <= 0
        or int(stream.get("height", 0)) <= 0
    ):
        raise PairV5NativeDPOTrainingError("video is not exact81/25fps")
    return dict(stream)


def load_pair_manifest(
    path_value: str | Path,
    *,
    expected_sha256: str,
    verify_files: bool = True,
    verify_media_metadata: bool = True,
    verify_tensor_headers: bool = True,
) -> PairManifest:
    """Validate authorization, calibration, selector replay, and media hashes."""

    manifest_path = _plain_absolute_path(
        str(path_value), label="PAIR manifest", must_exist=True
    )
    expected_manifest_sha = _sha(
        expected_sha256, length=64, label="manifest SHA"
    )
    raw = _read_stable(
        manifest_path,
        expected_sha256=expected_manifest_sha,
        label="PAIR manifest",
    )
    root = _closed(
        _strict_json(raw, label="PAIR manifest"),
        _ROOT_FIELDS,
        label="manifest",
    )
    if root["schema_version"] != MANIFEST_SCHEMA:
        raise PairV5NativeDPOTrainingError(
            "PAIR training manifest schema differs"
        )
    # Authorization is intentionally checked before opening calibration,
    # selector, media, renderer, or optimizer state.
    if root["optimizer_authorized"] is not True:
        raise PairV5NativeDPOTrainingError(
            "PAIR manifest does not authorize an optimizer"
        )
    if root["input_closure"] != _INPUT_CLOSURE:
        raise PairV5NativeDPOTrainingError(
            "PAIR manifest admits a forbidden input"
        )
    unsigned_root = dict(root)
    declared_manifest_digest = _sha(
        unsigned_root.pop("manifest_digest"),
        length=64,
        label="manifest digest",
    )
    if declared_manifest_digest != object_sha256(unsigned_root):
        raise PairV5NativeDPOTrainingError(
            "PAIR manifest embedded digest differs"
        )
    if not verify_files:
        raise PairV5NativeDPOTrainingError(
            "calibration receipt verification cannot be disabled"
        )

    calibration_row = _closed(
        root["action_calibration"],
        _CALIBRATION_FIELDS,
        label="action calibration",
    )
    calibration_path = _plain_absolute_path(
        calibration_row["receipt_path"],
        label="calibration receipt",
        must_exist=True,
    )
    calibration_sha = _sha(
        calibration_row["receipt_sha256"],
        length=64,
        label="calibration receipt SHA",
    )
    registered_calibration_digest = _sha(
        calibration_row["registered_receipt_digest"],
        length=64,
        label="registered calibration digest",
    )
    calibration_raw = _read_stable(
        calibration_path,
        expected_sha256=calibration_sha,
        label="calibration receipt",
    )
    try:
        checked_calibration = calibration.validate_calibration_receipt(
            _strict_json(calibration_raw, label="calibration receipt")
        )
        checked_optimizer_provenance = calibration.validate_calibrator_provenance(
            calibration_row["optimizer_provenance"]
        )
    except Exception as error:
        raise PairV5NativeDPOTrainingError(
            "action calibration validation failed"
        ) from error
    if (
        checked_calibration["receipt_digest"] != registered_calibration_digest
        or checked_calibration["optimizer_authorized"] is not True
    ):
        raise PairV5NativeDPOTrainingError(
            "action calibration did not authorize training"
        )
    if (
        checked_optimizer_provenance["calibration_receipt_digest"]
        != registered_calibration_digest
        or checked_optimizer_provenance["optimizer_authorized"] is not True
    ):
        raise PairV5NativeDPOTrainingError(
            "calibration optimizer provenance differs"
        )

    pair_rows = root["pairs"]
    if (
        not isinstance(pair_rows, list)
        or not MIN_PAIRS <= len(pair_rows) <= MAX_PAIRS
        or type(root["pair_count"]) is not int
        or root["pair_count"] != len(pair_rows)
    ):
        raise PairV5NativeDPOTrainingError(
            "PAIR manifest must contain one to four pairs"
        )
    pairs: list[PreferencePair] = []
    snapshots: list[FileSnapshot] = [
        FileSnapshot.capture(
            manifest_path,
            expected_sha256=expected_manifest_sha,
            label="PAIR manifest",
        ),
        FileSnapshot.capture(
            calibration_path,
            expected_sha256=calibration_sha,
            label="calibration receipt",
        ),
    ]
    seen_pair_ids: set[str] = set()
    seen_media: set[str] = set()
    for index, value in enumerate(pair_rows):
        row = _closed(value, _PAIR_FIELDS, label=f"pairs[{index}]")
        if row["schema_version"] != PAIR_SCHEMA:
            raise PairV5NativeDPOTrainingError(
                "preference pair schema differs"
            )
        pair_id = _safe_id(row["pair_id"], label="pair_id")
        if pair_id in seen_pair_ids:
            raise PairV5NativeDPOTrainingError(
                "preference pair ID is duplicated"
            )
        seen_pair_ids.add(pair_id)
        source_path = _plain_absolute_path(
            row["source_video_path"], label="source video", must_exist=True
        )
        source_sha = _sha(
            row["source_video_sha256"],
            length=64,
            label="source video SHA",
        )
        if (
            row["source_frame_count"] != FRAME_COUNT
            or type(row["source_fps"]) is not float
            or row["source_fps"] != FPS
            or row["source_reference_indices"] != list(REFERENCE_INDICES)
        ):
            raise PairV5NativeDPOTrainingError(
                "source exact81/RV2V-4 contract differs"
            )
        instruction = row["instruction"]
        instruction_sha = row["instruction_sha256"]
        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or "\x00" in instruction
            or hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            != instruction_sha
        ):
            raise PairV5NativeDPOTrainingError(
                "instruction text/hash differs"
            )
        try:
            policy = safe_pareto.validate_policy(row["selector_policy"])
            selector_provenance = safe_pareto.validate_calibrator_provenance(
                row["selector_calibrator_provenance"]
            )
        except Exception as error:
            raise PairV5NativeDPOTrainingError(
                "selector policy/provenance validation failed"
            ) from error
        if (
            selector_provenance["calibration_receipt_digest"]
            != registered_calibration_digest
            or selector_provenance["calibration_receipt_sha256"]
            != calibration_sha
        ):
            raise PairV5NativeDPOTrainingError(
                "selector uses a different calibration"
            )
        try:
            replayed = safe_pareto.replay_and_verify_receipt(
                row["selector_receipt"],
                state=row["selector_state_before"],
                candidates=row["selector_candidates"],
                policy=policy,
                calibrator_provenance=selector_provenance,
            )
        except Exception as error:
            raise PairV5NativeDPOTrainingError(
                "safe-Pareto selector receipt did not replay"
            ) from error
        selected = replayed["selected_pair"]
        if selected is None or replayed["decision"] == "no_eligible_pair":
            raise PairV5NativeDPOTrainingError(
                "selector receipt contains no safe pair"
            )
        winner_endpoint = {
            "candidate_id": selected["winner_candidate_id"],
            "candidate_digest": selected["winner_candidate_digest"],
        }
        loser_endpoint = {
            "candidate_id": selected["loser_candidate_id"],
            "candidate_digest": selected["loser_candidate_digest"],
        }
        winner = _validate_media(
            row["winner"],
            endpoint=winner_endpoint,
            source_path=source_path,
            source_sha256=source_sha,
            instruction=instruction,
            instruction_sha256=instruction_sha,
            verify_files=True,
            verify_tensor_headers=verify_tensor_headers,
        )
        loser = _validate_media(
            row["loser"],
            endpoint=loser_endpoint,
            source_path=source_path,
            source_sha256=source_sha,
            instruction=instruction,
            instruction_sha256=instruction_sha,
            verify_files=True,
            verify_tensor_headers=verify_tensor_headers,
        )
        if (
            winner.artifact.sha256 == loser.artifact.sha256
            or winner.candidate_id == loser.candidate_id
        ):
            raise PairV5NativeDPOTrainingError(
                "chosen/rejected media are identical"
            )
        for media in (winner, loser):
            if media.artifact.sha256 in seen_media:
                raise PairV5NativeDPOTrainingError(
                    "candidate media is reused across pairs"
                )
            seen_media.add(media.artifact.sha256)
        source_snapshot = FileSnapshot.capture(
            source_path, expected_sha256=source_sha, label="source video"
        )
        if verify_media_metadata:
            _ffprobe_exact81(source_path)
            for media in (winner, loser):
                if media.artifact_kind == "exact81_mp4":
                    _ffprobe_exact81(media.artifact.path)
        unsigned_pair = dict(row)
        embedded_pair_digest = _sha(
            unsigned_pair.pop("pair_digest"),
            length=64,
            label="pair digest",
        )
        if embedded_pair_digest != object_sha256(unsigned_pair):
            raise PairV5NativeDPOTrainingError(
                "preference pair embedded digest differs"
            )
        pairs.append(
            PreferencePair(
                pair_id,
                source_snapshot,
                instruction,
                instruction_sha,
                winner,
                loser,
                selected["pair_digest"],
                replayed["receipt_digest"],
                embedded_pair_digest,
            )
        )
        snapshots.extend(
            (
                source_snapshot,
                winner.artifact,
                winner.rollout_receipt,
                loser.artifact,
                loser.rollout_receipt,
            )
        )
    return PairManifest(
        manifest_path,
        expected_manifest_sha,
        declared_manifest_digest,
        snapshots[1],
        registered_calibration_digest,
        checked_optimizer_provenance["provenance_digest"],
        tuple(pairs),
        tuple(snapshots),
    )


def exact40_schedule_index(schedule_step: int) -> int:
    """Return the preregistered exact40 coordinate for one schedule step."""

    if type(schedule_step) is not int or schedule_step < 0:
        raise PairV5NativeDPOTrainingError(
            "schedule step must be nonnegative"
        )
    index = schedule_step % EXACT40_STEPS
    action_adapter.sigma_gate(index)
    return index


def is_frozen_anchor_audit(schedule_step: int) -> bool:
    return exact40_schedule_index(schedule_step) in AUDIT_SIGMA_INDICES


def expected_optimizer_updates(schedule_steps: int) -> int:
    if (
        type(schedule_steps) is not int
        or schedule_steps <= 0
        or schedule_steps % EXACT40_STEPS
    ):
        raise PairV5NativeDPOTrainingError(
            "schedule steps must be a positive multiple of exact40"
        )
    return schedule_steps // EXACT40_STEPS * len(TRAINABLE_SIGMA_INDICES)


def noise_seed(
    *, seed: int, schedule_step: int, accumulation_index: int, dp_rank: int
) -> int:
    material = (
        f"{seed}\x00pair-v5-fresh-epsilon\x00{schedule_step}\x00"
        f"{accumulation_index}\x00{dp_rank}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63


def build_shared_pair_states(
    chosen_clean: Any, rejected_clean: Any, epsilon: Any, sigma: Any
) -> tuple[Any, Any]:
    """Construct the exact states reconstructed by the flow-DPO core."""

    import torch

    values = (chosen_clean, rejected_clean, epsilon, sigma)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise PairV5NativeDPOTrainingError(
            "shared pair state inputs must be tensors"
        )
    if (
        chosen_clean.dtype != torch.float32
        or rejected_clean.dtype != torch.float32
        or epsilon.dtype != torch.float32
        or sigma.dtype != torch.float32
        or chosen_clean.shape != rejected_clean.shape
        or chosen_clean.shape != epsilon.shape
        or chosen_clean.device != rejected_clean.device
        or chosen_clean.device != epsilon.device
        or chosen_clean.device != sigma.device
        or sigma.numel() != 1
        or any(value.requires_grad or value.grad_fn is not None for value in values)
    ):
        raise PairV5NativeDPOTrainingError(
            "shared pair state inputs differ"
        )
    view = sigma.reshape(1, *([1] * (chosen_clean.ndim - 1)))
    return (
        ((1.0 - view) * chosen_clean + view * epsilon).detach(),
        ((1.0 - view) * rejected_clean + view * epsilon).detach(),
    )


def native_vjp_branch_registry(
    pack: Any, cond_embeds: Any, uncond_embeds: Any
) -> tuple[tuple[str, Any, Any, float], ...]:
    rows = (
        (
            "none_uncond",
            pack.none,
            uncond_embeds,
            native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS["none_uncond"],
        ),
        (
            "V_uncond",
            pack.video,
            uncond_embeds,
            native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS["V_uncond"],
        ),
        (
            "VI_uncond",
            pack.video_image,
            uncond_embeds,
            native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS["VI_uncond"],
        ),
        (
            "VI_cond",
            pack.video_image,
            cond_embeds,
            native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS["VI_cond"],
        ),
    )
    if (
        tuple(name for name, _, _, _ in rows)
        != tuple(guidance.guidance_receipt()["forward_order"])
        or not math.isclose(
            sum(coefficient for _, _, _, coefficient in rows),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise PairV5NativeDPOTrainingError(
            "native RV2V VJP registry differs"
        )
    return rows


def validate_cross_module_contract() -> Mapping[str, Any]:
    """Fail closed if adapter, bridge, loss, or exact40 contracts diverge."""

    bridge = dict(native_bridge.bridge_contract_receipt())
    loss = dict(flow_dpo.contract_receipt())
    schedule = dict(native.native_unipc40_schedule_receipt())
    if (
        (bridge["frame_count"], bridge["latent_channels"], bridge["latent_phases"])
        != (FRAME_COUNT, LATENT_CHANNELS, LATENT_PHASES)
        or (
            loss["frame_count"],
            loss["latent_channels"],
            loss["latent_phases"],
        )
        != (FRAME_COUNT, LATENT_CHANNELS, LATENT_PHASES)
        or bridge["rv2v_reference_frame_indices"] != list(REFERENCE_INDICES)
        or bridge["rv2v_reference_count"] != len(REFERENCE_INDICES)
    ):
        raise PairV5NativeDPOTrainingError(
            "cross-module exact81/RV2V-4 geometry differs"
        )
    if (
        tuple(TRAINABLE_SIGMA_INDICES) != tuple(range(38))
        or tuple(AUDIT_SIGMA_INDICES) != (38, 39)
        or tuple(TRAINABLE_SIGMA_INDICES + AUDIT_SIGMA_INDICES)
        != tuple(range(EXACT40_STEPS))
    ):
        raise PairV5NativeDPOTrainingError(
            "cross-module exact40 partition differs"
        )
    expected_coefficients = {
        "none_uncond": -0.25,
        "V_uncond": -3.25,
        "VI_uncond": 0.5,
        "VI_cond": 4.0,
    }
    if native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS != expected_coefficients:
        raise PairV5NativeDPOTrainingError(
            "cross-module native VJP coefficients differ"
        )
    forbidden_false = (
        bridge.get("proposal_visual_data_consumed") is False
        and bridge.get("paired_target_consumed") is False
        and bridge.get("mask_flow_pose_track_trajectory_consumed") is False
        and loss.get("proposal_visual_data_consumed") is False
        and loss.get("paired_target_consumed") is False
        and loss.get("mask_flow_pose_track_trajectory_consumed") is False
    )
    if not forbidden_false:
        raise PairV5NativeDPOTrainingError(
            "cross-module information closure differs"
        )
    value = {
        "bridge_contract_digest": bridge["digest"],
        "flow_dpo_contract_digest": loss["digest"],
        "exact40_schedule_digest": schedule["digest"],
        "exact40_schedule_sha256": action_adapter.sigma_strata.SCHEDULE_SHA256,
        "dynamic_update_indices": list(TRAINABLE_SIGMA_INDICES),
        "zero_update_audit_indices": list(AUDIT_SIGMA_INDICES),
        "native_vjp_coefficients": expected_coefficients,
        "forbidden_inputs_absent": True,
    }
    return {**value, "digest": object_sha256(value)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--expected-pair-manifest-sha256", required=True)
    parser.add_argument("--frozen-cio-adapter", required=True)
    parser.add_argument("--expected-frozen-cio-adapter-sha256", required=True)
    parser.add_argument("--frozen-cio-receipt", required=True)
    parser.add_argument("--expected-frozen-cio-receipt-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-schedule-steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--num-frames", type=int, choices=(FRAME_COUNT,), default=FRAME_COUNT
    )
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--ack-experimental-no-action-success-claim", action="store_true"
    )
    return parser


def validate_cli(args: argparse.Namespace) -> Mapping[str, Any]:
    if (
        (WORLD_SIZE, SP_SIZE, DP_SIZE)
        != (runtime.WORLD_SIZE, runtime.SP_SIZE, runtime.DP_SIZE)
        or args.num_frames != FRAME_COUNT
    ):
        raise PairV5NativeDPOTrainingError(
            "WORLD8 DP2xSP4 exact81 contract differs"
        )
    if args.ack_experimental_no_action_success_claim is not True:
        raise PairV5NativeDPOTrainingError(
            "experimental no-success-claim acknowledgement is required"
        )
    update_count = expected_optimizer_updates(args.max_schedule_steps)
    if (
        type(args.gradient_accumulation_steps) is not int
        or not 1 <= args.gradient_accumulation_steps <= 16
    ):
        raise PairV5NativeDPOTrainingError(
            "gradient accumulation must lie in [1,16]"
        )
    for name in ("learning_rate", "beta", "max_grad_norm"):
        value = getattr(args, name)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise PairV5NativeDPOTrainingError(
                f"{name} must be finite and positive"
            )
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise PairV5NativeDPOTrainingError("seed must lie in [0,2^63)")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        _sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_pair_manifest_sha256",
        "expected_frozen_cio_adapter_sha256",
        "expected_frozen_cio_receipt_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        _sha(getattr(args, name), length=64, label=name)
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise PairV5NativeDPOTrainingError("checkpoint identity differs")
    return {
        "world_size": WORLD_SIZE,
        "data_parallel_size": DP_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "frame_count": FRAME_COUNT,
        "schedule_steps": args.max_schedule_steps,
        "optimizer_updates": update_count,
        "zero_update_audits": args.max_schedule_steps - update_count,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }


def preflight_training_inputs(
    args: argparse.Namespace,
) -> tuple[PairManifest, FileSnapshot, FileSnapshot]:
    """Run all authorization/schema/hash gates before model/optimizer creation."""

    validate_cli(args)
    validate_cross_module_contract()
    manifest = load_pair_manifest(
        args.pair_manifest,
        expected_sha256=args.expected_pair_manifest_sha256,
        verify_files=True,
        verify_media_metadata=True,
        verify_tensor_headers=True,
    )
    cio_path = _plain_absolute_path(
        args.frozen_cio_adapter,
        label="frozen CIO adapter",
        must_exist=True,
    )
    cio_receipt_path = _plain_absolute_path(
        args.frozen_cio_receipt,
        label="frozen CIO receipt",
        must_exist=True,
    )
    cio_snapshot = FileSnapshot.capture(
        cio_path,
        expected_sha256=args.expected_frozen_cio_adapter_sha256,
        label="frozen CIO adapter",
    )
    cio_receipt_snapshot = FileSnapshot.capture(
        cio_receipt_path,
        expected_sha256=args.expected_frozen_cio_receipt_sha256,
        label="frozen CIO receipt",
    )
    cio_receipt = _strict_json(
        _read_stable(
            cio_receipt_path,
            expected_sha256=args.expected_frozen_cio_receipt_sha256,
            label="frozen CIO receipt",
        ),
        label="frozen CIO receipt",
    )
    if (
        cio_receipt.get("complete") is not True
        or cio_receipt.get("semantic_action_learned") is not False
        or cio_receipt.get("action_editing_claim_authorized") is not False
        or cio_receipt.get("artifacts", {}).get("adapter.safetensors")
        != args.expected_frozen_cio_adapter_sha256
    ):
        raise PairV5NativeDPOTrainingError(
            "frozen CIO receipt contract differs"
        )
    return manifest, cio_snapshot, cio_receipt_snapshot


@dataclass(frozen=True)
class VisualRuntimePair:
    contract: PreferencePair
    condition_video: Any
    image_references: tuple[Any, ...]
    chosen_clean: Any
    rejected_clean: Any
    tensor_digest: str


@dataclass(frozen=True)
class RuntimePair:
    contract: PreferencePair
    condition_video: Any
    image_references: tuple[Any, ...]
    chosen_clean: Any
    rejected_clean: Any
    cond_embeds: Any
    uncond_embeds: Any
    tensor_digest: str


def _broadcast_sp(value: Any, *, parallel: runtime.ParallelContext) -> None:
    import torch.distributed as dist

    source_rank = runtime.SP_GROUP_RANKS[parallel.contract.arm_index][0]
    dist.broadcast(value, src=source_rank, group=parallel.sp_group)


def _load_clean_latent(media: CandidateMedia, *, vae: Any, device: Any) -> Any:
    import torch

    media.artifact.assert_unchanged()
    if media.artifact_kind == "normalized_clean_latent_safetensors":
        from safetensors import safe_open

        with safe_open(
            str(media.artifact.path), framework="pt", device="cpu"
        ) as opened:
            value = opened.get_tensor(str(media.tensor_key)).float().contiguous()
        if (
            tuple(int(item) for item in value.shape) != media.latent_shape
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all().item())
        ):
            raise PairV5NativeDPOTrainingError(
                "clean latent changed after preflight"
            )
        return value.to(device=device).contiguous().detach()
    from bernini.pipeline import _vae_encode

    pixels, metadata, observed_sha = source_audit.prepare_hashed_source_snapshot(
        media.artifact.path
    )
    if (
        observed_sha != media.artifact.sha256
        or metadata["frame_count"] != FRAME_COUNT
        or float(metadata["fps"]) != FPS
    ):
        raise PairV5NativeDPOTrainingError(
            "candidate video changed after preflight"
        )
    with torch.no_grad():
        return (
            _vae_encode(vae, pixels.to(device=device, dtype=torch.float32))
            .float()
            .detach()
            .contiguous()
        )


def _prepare_visual_runtime_pairs(
    manifest: PairManifest,
    *,
    vae: Any,
    device: Any,
    parallel: runtime.ParallelContext,
) -> tuple[VisualRuntimePair, ...]:
    import torch
    from bernini.pipeline import _vae_encode

    result: list[VisualRuntimePair] = []
    vae.to(device)
    for pair in manifest.pairs:
        source_pixels, metadata, source_sha = (
            source_audit.prepare_hashed_source_snapshot(pair.source.path)
        )
        if (
            source_sha != pair.source.sha256
            or metadata["frame_count"] != FRAME_COUNT
            or float(metadata["fps"]) != FPS
        ):
            raise PairV5NativeDPOTrainingError(
                "source video changed after preflight"
            )
        pixels = source_pixels.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            source_latent = _vae_encode(vae, pixels).float().detach().contiguous()
            references = tuple(
                _vae_encode(
                    vae,
                    pixels[:, :, index : index + 1].contiguous(),
                )
                .float()
                .detach()
                .contiguous()
                for index in REFERENCE_INDICES
            )
        chosen = _load_clean_latent(pair.winner, vae=vae, device=device)
        rejected = _load_clean_latent(pair.loser, vae=vae, device=device)
        expected_shape = tuple(int(item) for item in source_latent.shape)
        if (
            expected_shape[:3] != (1, LATENT_CHANNELS, LATENT_PHASES)
            or tuple(chosen.shape) != expected_shape
            or tuple(rejected.shape) != expected_shape
            or torch.equal(chosen, rejected)
            or any(
                tuple(reference.shape)
                != (1, LATENT_CHANNELS, 1, *expected_shape[3:])
                for reference in references
            )
        ):
            raise PairV5NativeDPOTrainingError(
                "PAIR/source latent geometry differs"
            )
        tensors = (source_latent, *references, chosen, rejected)
        for tensor in tensors:
            _broadcast_sp(tensor, parallel=parallel)
        tensor_digest = object_sha256(
            [runtime.tensor_sha256(tensor) for tensor in tensors]
        )
        runtime.digest_consensus(
            tensor_digest,
            group=parallel.sp_group,
            expected_count=SP_SIZE,
            label=f"PAIR tensors {pair.pair_id}",
        )
        result.append(
            VisualRuntimePair(
                pair,
                source_latent,
                references,
                chosen,
                rejected,
                tensor_digest,
            )
        )
        del pixels, source_pixels
        torch.cuda.empty_cache()
    vae.to("cpu")
    return tuple(result)


def _attach_frozen_text_conditions(
    visual_pairs: Sequence[VisualRuntimePair],
    *,
    renderer: Any,
    tokenizer: Any,
    device: Any,
    parallel: runtime.ParallelContext,
) -> tuple[RuntimePair, ...]:
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    negative_ids, negative_mask = native_runtime._tokenize_negative(
        tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT
    )
    with torch.inference_mode():
        unconditional = renderer.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
    _broadcast_sp(unconditional, parallel=parallel)
    result: list[RuntimePair] = []
    for visual in visual_pairs:
        complete_prompt = native_infer.build_task_prompt(
            "rv2v",
            visual.contract.instruction,
            prompt_cleaner=prompt_clean,
        )
        positive_ids, positive_mask = native_runtime._tokenize_positive(
            tokenizer, complete_prompt
        )
        with torch.inference_mode():
            conditional = renderer.encode_prompt(
                positive_ids.to(device), positive_mask.to(device)
            ).detach()
        _broadcast_sp(conditional, parallel=parallel)
        if (
            tuple(conditional.shape) != (1, 512, 4096)
            or tuple(unconditional.shape) != (1, 512, 4096)
            or torch.equal(conditional, unconditional)
        ):
            raise PairV5NativeDPOTrainingError(
                "frozen text embedding closure differs"
            )
        result.append(
            RuntimePair(
                visual.contract,
                visual.condition_video,
                visual.image_references,
                visual.chosen_clean,
                visual.rejected_clean,
                conditional,
                unconditional,
                visual.tensor_digest,
            )
        )
    return tuple(result)


def _native_coordinate(index: int, *, device: Any) -> tuple[Any, Any]:
    import torch

    sigma = torch.tensor(
        [native.NATIVE_UNIPC40_SIGMAS[index]],
        dtype=torch.float32,
        device=device,
    ).detach()
    timestep = torch.tensor(
        [native.NATIVE_UNIPC40_TIMESTEPS[index]],
        dtype=torch.float32,
        device=device,
    ).detach()
    checked_sigma, checked_timestep, checked_index = (
        native_bridge._native_schedule_coordinate(sigma, timestep)
    )
    if checked_index != index:
        raise PairV5NativeDPOTrainingError(
            "native exact40 coordinate differs"
        )
    return checked_sigma, checked_timestep


def _fresh_shared_epsilon(shape: Sequence[int], *, seed: int, device: Any) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return value.to(device=device).contiguous().detach()


def _guided_prediction(
    diffusion: Any,
    transformer: Any,
    pair: RuntimePair,
    state: Any,
    *,
    timestep: Any,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Any,
    sp_rank: int,
    schedule_index: int,
    action_enabled: bool,
) -> Any:
    pack = native_runtime._build_pack(
        transformer,
        pair.condition_video,
        pair.image_references,
        state,
    )
    try:
        return native_runtime._guided_prediction_no_grad(
            diffusion,
            pack,
            timestep=timestep,
            cond_embeds=pair.cond_embeds,
            uncond_embeds=pair.uncond_embeds,
            action_handle=action_handle,
            cio_handle=cio_handle,
            sp_rank=sp_rank,
            sigma_index=schedule_index,
            action_enabled=action_enabled,
            video_shape=state.shape,
        )
    finally:
        del pack


def _serial_prediction_vjp(
    diffusion: Any,
    transformer: Any,
    pair: RuntimePair,
    state: Any,
    *,
    timestep: Any,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Any,
    sp_rank: int,
    schedule_index: int,
    output_cotangent: Any,
    expected_guided: Any,
) -> float:
    return native_runtime._replay_prediction_vjp(
        diffusion,
        transformer,
        source_video=pair.condition_video,
        references=pair.image_references,
        x_sigma=state,
        timestep=timestep,
        cond_embeds=pair.cond_embeds,
        uncond_embeds=pair.uncond_embeds,
        action_handle=action_handle,
        cio_handle=cio_handle,
        sp_rank=sp_rank,
        sigma_index=schedule_index,
        output_cotangent=output_cotangent,
        expected_guided=expected_guided,
    )


def _sp_and_world_step_record(
    local_record: Mapping[str, Any],
    *,
    parallel: runtime.ParallelContext,
) -> list[Mapping[str, Any]]:
    import torch.distributed as dist

    projection = {
        key: value for key, value in local_record.items() if key != "sp_rank"
    }
    runtime.digest_consensus(
        object_sha256(projection),
        group=parallel.sp_group,
        expected_count=SP_SIZE,
        label="PAIR-v5 SP schedule record",
    )
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, dict(local_record), group=parallel.world_group)
    return [gathered[0], gathered[4]]


def _audit_low_sigma_step(
    *,
    schedule_step: int,
    schedule_index: int,
    pair: RuntimePair,
    diffusion: Any,
    transformer: Any,
    optimizer: Any,
    trainable: Sequence[tuple[str, Any]],
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Any,
    distributed: runtime.DistributedContract,
    parallel: runtime.ParallelContext,
    device: Any,
    seed: int,
    optimizer_updates_before: int,
) -> Mapping[str, Any]:
    import torch

    if schedule_index not in AUDIT_SIGMA_INDICES:
        raise PairV5NativeDPOTrainingError(
            "frozen-anchor audit received a dynamic coordinate"
        )
    optimizer.zero_grad(set_to_none=True)
    before_digest = runtime.parameter_consensus(
        trainable,
        parallel.world_group,
        f"PAIR-v5 frozen-anchor before {schedule_step}",
    )
    sigma, timestep = _native_coordinate(schedule_index, device=device)
    seed_value = noise_seed(
        seed=seed,
        schedule_step=schedule_step,
        accumulation_index=0,
        dp_rank=distributed.arm_index,
    )
    epsilon = _fresh_shared_epsilon(
        pair.chosen_clean.shape, seed=seed_value, device=device
    )
    _broadcast_sp(epsilon, parallel=parallel)
    chosen_state, rejected_state = build_shared_pair_states(
        pair.chosen_clean, pair.rejected_clean, epsilon, sigma
    )
    equal_by_endpoint: dict[str, bool] = {}
    for endpoint, state in (
        ("chosen", chosen_state),
        ("rejected", rejected_state),
    ):
        action_output = _guided_prediction(
            diffusion,
            transformer,
            pair,
            state,
            timestep=timestep,
            action_handle=action_handle,
            cio_handle=cio_handle,
            sp_rank=distributed.sp_rank,
            schedule_index=schedule_index,
            action_enabled=True,
        )
        reference_output = _guided_prediction(
            diffusion,
            transformer,
            pair,
            state,
            timestep=timestep,
            action_handle=action_handle,
            cio_handle=cio_handle,
            sp_rank=distributed.sp_rank,
            schedule_index=schedule_index,
            action_enabled=False,
        )
        equal_by_endpoint[endpoint] = bool(
            torch.equal(action_output, reference_output)
        )
    exact_base = all(equal_by_endpoint.values())
    if not runtime.world_all_true(exact_base, group=parallel.world_group):
        raise PairV5NativeDPOTrainingError(
            "low-sigma action route is not byte-exact frozen base"
        )
    if any(parameter.grad is not None for _, parameter in trainable):
        raise PairV5NativeDPOTrainingError(
            "low-sigma audit unexpectedly constructed gradients"
        )
    after_digest = runtime.parameter_consensus(
        trainable,
        parallel.world_group,
        f"PAIR-v5 frozen-anchor after {schedule_step}",
    )
    if after_digest != before_digest:
        raise PairV5NativeDPOTrainingError(
            "low-sigma audit changed Action-LoRA parameters"
        )
    return {
        "schedule_step": schedule_step,
        "schedule_index": schedule_index,
        "phase": "low_sigma_frozen_anchor_audit",
        "dp_rank": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "pair_id": pair.contract.pair_id,
        "pair_digest": pair.contract.pair_digest,
        "noise_seed": seed_value,
        "fresh_epsilon_sha256": runtime.tensor_sha256(epsilon),
        "flow_dpo_constructed": False,
        "loss_constructed": False,
        "backward_called": False,
        "optimizer_step_called": False,
        "optimizer_updates_before": optimizer_updates_before,
        "optimizer_updates_after": optimizer_updates_before,
        "chosen_action_equals_reference_byte_exact": equal_by_endpoint["chosen"],
        "rejected_action_equals_reference_byte_exact": equal_by_endpoint["rejected"],
        "parameter_digest_before": before_digest,
        "parameter_digest_after": after_digest,
        "zero_update_audit_passed": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, cio_snapshot, cio_receipt_snapshot = preflight_training_inputs(args)
    run_contract = validate_cli(args)
    cross_contract = validate_cross_module_contract()
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "optimizer_authorized": True,
                    "pair_count": len(manifest.pairs),
                    "manifest_digest": manifest.manifest_digest,
                    "expected_optimizer_updates": run_contract[
                        "optimizer_updates"
                    ],
                    "preflight_only": True,
                    "semantic_action_editing_success": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise PairV5NativeDPOTrainingError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise PairV5NativeDPOTrainingError(
            "pinned Bernini attention-head count differs"
        )
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
    legacy.seed_same_sample(args.seed)

    # Keep the renderer and both adapters on CPU while the VAE prepares all
    # exact81 source/candidate latents.  This avoids VAE+renderer co-residency.
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False)
    renderer.eval()
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise PairV5NativeDPOTrainingError(
            "PAIR-v5 requires transformer_1 only"
        )
    cio_handle, cio_load_receipt = native_runtime._load_optional_frozen_cio(
        transformer,
        str(cio_snapshot.path),
        cio_snapshot.sha256,
    )
    if cio_handle is None or cio_load_receipt.get("loaded") is not True:
        raise PairV5NativeDPOTrainingError("required frozen CIO did not load")
    action_handle = action_adapter.install_pair_v5_action_adapter(transformer)
    trainable = action_handle.trainable_named_parameters()
    if (
        not action_handle.base_parameters_frozen()
        or not trainable
        or any(
            parameter.requires_grad
            for name, parameter in transformer.named_parameters()
            if "action_lora_" not in name
        )
        or any(
            "attn2" not in name
            or not (name.endswith("action_lora_a.weight") or name.endswith("action_lora_b.weight"))
            for name, _ in trainable
        )
    ):
        raise PairV5NativeDPOTrainingError(
            "Action-LoRA trainability closure differs"
        )

    vae = (
        AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
    )
    visual_pairs = _prepare_visual_runtime_pairs(
        manifest, vae=vae, device=device, parallel=parallel
    )
    del vae
    torch.cuda.empty_cache()

    renderer.to(device)
    renderer.eval()
    disable = getattr(renderer, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    if bool(getattr(transformer, "gradient_checkpointing", False)) or bool(
        getattr(transformer, "is_gradient_checkpointing", False)
    ):
        raise PairV5NativeDPOTrainingError(
            "gradient checkpointing remains enabled"
        )
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
    runtime_pairs = _attach_frozen_text_conditions(
        visual_pairs,
        renderer=renderer,
        tokenizer=tokenizer,
        device=device,
        parallel=parallel,
    )
    renderer.t5_text_encoder.to("cpu")
    del tokenizer, visual_pairs
    torch.cuda.empty_cache()

    # Every fail-closed authorization/hash/schema gate above precedes this
    # optimizer construction.
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    history: list[Mapping[str, Any]] = []
    optimizer_updates = 0
    for schedule_step in range(args.max_schedule_steps):
        schedule_index = exact40_schedule_index(schedule_step)
        if schedule_index in AUDIT_SIGMA_INDICES:
            row_index = (
                schedule_step * DP_SIZE + distributed.arm_index
            ) % len(runtime_pairs)
            local = _audit_low_sigma_step(
                schedule_step=schedule_step,
                schedule_index=schedule_index,
                pair=runtime_pairs[row_index],
                diffusion=diffusion,
                transformer=transformer,
                optimizer=optimizer,
                trainable=trainable,
                action_handle=action_handle,
                cio_handle=cio_handle,
                distributed=distributed,
                parallel=parallel,
                device=device,
                seed=args.seed,
                optimizer_updates_before=optimizer_updates,
            )
            dp_records = _sp_and_world_step_record(local, parallel=parallel)
            history.append(
                {
                    "schedule_step": schedule_step,
                    "schedule_index": schedule_index,
                    "phase": "low_sigma_frozen_anchor_audit",
                    "dp_records": dp_records,
                }
            )
            if distributed.rank == 0:
                print(
                    json.dumps(
                        {
                            "schedule_step": schedule_step,
                            "schedule_index": schedule_index,
                            "optimizer_step": False,
                            "frozen_anchor_audit": True,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            continue

        if schedule_index not in TRAINABLE_SIGMA_INDICES:
            raise PairV5NativeDPOTrainingError(
                "exact40 coordinate is neither update nor audit"
            )
        optimizer.zero_grad(set_to_none=True)
        micro_records: list[Mapping[str, Any]] = []
        loss_sum = 0.0
        replay_max = 0.0
        for accumulation_index in range(args.gradient_accumulation_steps):
            row_index = (
                schedule_step * DP_SIZE * args.gradient_accumulation_steps
                + distributed.arm_index * args.gradient_accumulation_steps
                + accumulation_index
            ) % len(runtime_pairs)
            pair = runtime_pairs[row_index]
            sigma, timestep = _native_coordinate(schedule_index, device=device)
            seed_value = noise_seed(
                seed=args.seed,
                schedule_step=schedule_step,
                accumulation_index=accumulation_index,
                dp_rank=distributed.arm_index,
            )
            epsilon = _fresh_shared_epsilon(
                pair.chosen_clean.shape, seed=seed_value, device=device
            )
            _broadcast_sp(epsilon, parallel=parallel)
            chosen_state, rejected_state = build_shared_pair_states(
                pair.chosen_clean,
                pair.rejected_clean,
                epsilon,
                sigma,
            )
            detached_student: dict[str, Any] = {}
            detached_reference: dict[str, Any] = {}
            for endpoint, state in (
                ("chosen", chosen_state),
                ("rejected", rejected_state),
            ):
                detached_student[endpoint] = _guided_prediction(
                    diffusion,
                    transformer,
                    pair,
                    state,
                    timestep=timestep,
                    action_handle=action_handle,
                    cio_handle=cio_handle,
                    sp_rank=distributed.sp_rank,
                    schedule_index=schedule_index,
                    action_enabled=True,
                )
                detached_reference[endpoint] = _guided_prediction(
                    diffusion,
                    transformer,
                    pair,
                    state,
                    timestep=timestep,
                    action_handle=action_handle,
                    cio_handle=cio_handle,
                    sp_rank=distributed.sp_rank,
                    schedule_index=schedule_index,
                    action_enabled=False,
                ).detach()
            chosen_leaf = detached_student["chosen"].clone().requires_grad_(True)
            rejected_leaf = detached_student["rejected"].clone().requires_grad_(True)
            student_chosen = chosen_leaf + torch.zeros(
                (), dtype=chosen_leaf.dtype, device=device
            )
            student_rejected = rejected_leaf + torch.zeros(
                (), dtype=rejected_leaf.dtype, device=device
            )
            result = flow_dpo.reference_corrected_flow_dpo(
                pair.chosen_clean,
                pair.rejected_clean,
                epsilon,
                sigma,
                student_chosen,
                student_rejected,
                detached_reference["chosen"],
                detached_reference["rejected"],
                beta=args.beta,
            )
            if not torch.equal(result.chosen_x_sigma, chosen_state) or not torch.equal(
                result.rejected_x_sigma, rejected_state
            ):
                raise PairV5NativeDPOTrainingError(
                    "flow-DPO did not reconstruct shared candidate states"
                )
            finite = bool(torch.isfinite(result.loss.detach()).item())
            if not runtime.world_all_true(finite, group=parallel.world_group):
                raise PairV5NativeDPOTrainingError(
                    "non-finite DPO loss blocked update"
                )
            (result.loss / float(args.gradient_accumulation_steps)).backward()
            if chosen_leaf.grad is None or rejected_leaf.grad is None:
                raise PairV5NativeDPOTrainingError(
                    "DPO leaves have no output cotangent"
                )
            for state, cotangent, expected in (
                (
                    chosen_state,
                    chosen_leaf.grad.detach(),
                    detached_student["chosen"],
                ),
                (
                    rejected_state,
                    rejected_leaf.grad.detach(),
                    detached_student["rejected"],
                ),
            ):
                replay_max = max(
                    replay_max,
                    _serial_prediction_vjp(
                        diffusion,
                        transformer,
                        pair,
                        state,
                        timestep=timestep,
                        action_handle=action_handle,
                        cio_handle=cio_handle,
                        sp_rank=distributed.sp_rank,
                        schedule_index=schedule_index,
                        output_cotangent=cotangent,
                        expected_guided=expected,
                    ),
                )
            loss_sum += float(result.loss.detach().item())
            micro_records.append(
                {
                    "accumulation_index": accumulation_index,
                    "pair_id": pair.contract.pair_id,
                    "pair_digest": pair.contract.pair_digest,
                    "winner_candidate_id": pair.contract.winner.candidate_id,
                    "loser_candidate_id": pair.contract.loser.candidate_id,
                    "noise_seed": seed_value,
                    "fresh_epsilon_sha256": runtime.tensor_sha256(epsilon),
                    "shared_epsilon_and_sigma": True,
                    "loss": float(result.loss.detach().item()),
                    "advantage": float(result.advantage.detach().item()),
                    "student_gap": float(result.student_gap.detach().item()),
                    "reference_gap": float(result.reference_gap.detach().item()),
                }
            )
            del (
                chosen_state,
                rejected_state,
                epsilon,
                detached_student,
                detached_reference,
                chosen_leaf,
                rejected_leaf,
                student_chosen,
                student_rejected,
                result,
            )
            torch.cuda.empty_cache()
        preclip_norm = runtime.synchronize_gradients(trainable, parallel)
        clipped = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], args.max_grad_norm
        )
        if not math.isfinite(float(clipped)):
            raise PairV5NativeDPOTrainingError(
                "gradient clipping is non-finite"
            )
        optimizer.step()
        optimizer_updates += 1
        parameter_digest = runtime.parameter_consensus(
            trainable,
            parallel.world_group,
            f"PAIR-v5 action adapter update {optimizer_updates}",
        )
        local_record = {
            "schedule_step": schedule_step,
            "schedule_index": schedule_index,
            "phase": action_adapter.sigma_gate(schedule_index)[0],
            "dp_rank": distributed.arm_index,
            "sp_rank": distributed.sp_rank,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "micro_records": micro_records,
            "loss_mean": loss_sum / float(args.gradient_accumulation_steps),
            "preclip_gradient_norm_world_average": preclip_norm,
            "vjp_replay_max_abs": replay_max,
            "flow_dpo_constructed": True,
            "optimizer_step_called": True,
            "optimizer_updates_after": optimizer_updates,
            "parameter_digest_after": parameter_digest,
        }
        dp_records = _sp_and_world_step_record(local_record, parallel=parallel)
        history.append(
            {
                "schedule_step": schedule_step,
                "schedule_index": schedule_index,
                "phase": local_record["phase"],
                "dp_records": dp_records,
            }
        )
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "schedule_step": schedule_step,
                        "schedule_index": schedule_index,
                        "optimizer_update": optimizer_updates,
                        "loss_dp0": dp_records[0]["loss_mean"],
                        "loss_dp1": dp_records[1]["loss_mean"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    expected_updates = expected_optimizer_updates(args.max_schedule_steps)
    observed_indices = [record["schedule_index"] for record in history]
    expected_indices = [
        step % EXACT40_STEPS for step in range(args.max_schedule_steps)
    ]
    if (
        optimizer_updates != expected_updates
        or len(history) != args.max_schedule_steps
        or observed_indices != expected_indices
    ):
        raise PairV5NativeDPOTrainingError(
            "exact40 history/update coverage differs"
        )
    for record in history:
        index = record["schedule_index"]
        if index in AUDIT_SIGMA_INDICES:
            if any(
                dp["optimizer_step_called"] is not False
                or dp["loss_constructed"] is not False
                or dp["zero_update_audit_passed"] is not True
                for dp in record["dp_records"]
            ):
                raise PairV5NativeDPOTrainingError(
                    "low-sigma receipt is not a zero-update audit"
                )
    final_digest = runtime.parameter_consensus(
        trainable, parallel.world_group, "PAIR-v5 final action adapter"
    )
    if final_digest == initial_digest:
        raise PairV5NativeDPOTrainingError(
            "optimizer did not change Action-LoRA"
        )
    manifest.assert_unchanged()
    cio_snapshot.assert_unchanged()
    cio_receipt_snapshot.assert_unchanged()
    dist.barrier(group=parallel.world_group)

    if distributed.rank == 0:
        adapter_path = stage / "adapter.safetensors"
        optimizer_path = stage / "optimizer.pt"
        history_path = stage / "history.json"
        adapter_roundtrip = native_runtime._save_action_adapter(
            adapter_path, action_handle
        )
        runtime.atomic_torch_save(
            optimizer_path,
            {
                "schema_version": RUN_RECEIPT_SCHEMA,
                "optimizer": optimizer.state_dict(),
                "schedule_steps": args.max_schedule_steps,
                "optimizer_updates": optimizer_updates,
                "adapter_parameter_digest": final_digest,
            },
        )
        history_object = {
            "schema_version": HISTORY_SCHEMA,
            "schedule_step_count": args.max_schedule_steps,
            "optimizer_update_count": optimizer_updates,
            "zero_update_audit_count": args.max_schedule_steps
            - optimizer_updates,
            "complete_exact40_dynamic_coverage": True,
            "records": history,
        }
        runtime.atomic_json(history_path, history_object)
        receipt: dict[str, Any] = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "complete": True,
            "run_contract": dict(run_contract),
            "cross_module_contract": dict(cross_contract),
            "schedule_steps": args.max_schedule_steps,
            "optimizer_updates": optimizer_updates,
            "expected_optimizer_updates": expected_updates,
            "exact40": {
                "cycles": args.max_schedule_steps // EXACT40_STEPS,
                "all_schedule_indices_in_order": observed_indices,
                "dynamic_update_indices_per_cycle": list(
                    TRAINABLE_SIGMA_INDICES
                ),
                "zero_update_audit_indices_per_cycle": list(
                    AUDIT_SIGMA_INDICES
                ),
                "updates_per_cycle": len(TRAINABLE_SIGMA_INDICES),
                "audits_per_cycle": len(AUDIT_SIGMA_INDICES),
                "indices_38_39_flow_dpo_constructed": False,
                "indices_38_39_optimizer_step_called": False,
                "complete_exact40_dynamic_coverage": True,
            },
            "manifest": {
                "file": dict(manifest.snapshots[0].receipt()),
                "manifest_digest": manifest.manifest_digest,
                "optimizer_authorized": True,
                "calibration_file": dict(
                    manifest.calibration_receipt.receipt()
                ),
                "calibration_receipt_digest": manifest.calibration_receipt_digest,
                "calibration_optimizer_provenance_digest": (
                    manifest.calibration_provenance_digest
                ),
                "pairs": [
                    {
                        "pair_id": pair.pair_id,
                        "pair_digest": pair.pair_digest,
                        "safe_pareto_selected_pair_digest": pair.selected_pair_digest,
                        "safe_pareto_selector_receipt_digest": pair.selector_receipt_digest,
                        "source": dict(pair.source.receipt()),
                        "source_reference_indices": list(REFERENCE_INDICES),
                        "instruction_sha256": pair.instruction_sha256,
                        "winner": {
                            "candidate_id": pair.winner.candidate_id,
                            "candidate_digest": pair.winner.candidate_digest,
                            "artifact": dict(pair.winner.artifact.receipt()),
                            "native_rollout_receipt": dict(
                                pair.winner.rollout_receipt.receipt()
                            ),
                            "native_rollout_receipt_digest": (
                                pair.winner.rollout_receipt_digest
                            ),
                        },
                        "loser": {
                            "candidate_id": pair.loser.candidate_id,
                            "candidate_digest": pair.loser.candidate_digest,
                            "artifact": dict(pair.loser.artifact.receipt()),
                            "native_rollout_receipt": dict(
                                pair.loser.rollout_receipt.receipt()
                            ),
                            "native_rollout_receipt_digest": (
                                pair.loser.rollout_receipt_digest
                            ),
                        },
                    }
                    for pair in manifest.pairs
                ],
                "post_training_input_mutation_audit_passed": True,
            },
            "model_input_closure": dict(_INPUT_CLOSURE),
            "native_rv2v4": {
                "source_video_and_four_source_rgb_references_only": True,
                "reference_indices": list(REFERENCE_INDICES),
                "same_fresh_epsilon_sigma_source_refs_text_for_pair": True,
                "reference_policy": (
                    "same_frozen_bernini_plus_frozen_cio_action_lora_disabled"
                ),
                "serial_exact_linear_vjp": True,
                "one_transformer_graph_resident_at_a_time": True,
                "gradient_checkpointing_enabled": False,
                "proposal_or_target_consumed": False,
            },
            "objective": {
                "flow_dpo_contract": dict(flow_dpo.contract_receipt()),
                "beta": args.beta,
                "gradient_accumulation_steps": (
                    args.gradient_accumulation_steps
                ),
            },
            "adapter": {
                **dict(action_handle.receipt()),
                "trainable_scope": "attn2_qo_action_lora_only",
                "initial_parameter_digest": initial_digest,
                "final_parameter_digest": final_digest,
                "changed_by_optimizer": True,
                "safetensors_roundtrip": dict(adapter_roundtrip),
                "frozen_cio": {
                    **dict(cio_load_receipt),
                    "adapter_file": dict(cio_snapshot.receipt()),
                    "training_receipt_file": dict(
                        cio_receipt_snapshot.receipt()
                    ),
                    "active_in_student_and_reference": True,
                    "optimized": False,
                },
            },
            "distributed": {
                "world_size": WORLD_SIZE,
                "data_parallel_size": DP_SIZE,
                "sequence_parallel_size": SP_SIZE,
                "all_eight_gpus_used": True,
                "sp_groups": [list(item) for item in runtime.SP_GROUP_RANKS],
                "dp_groups": [list(item) for item in runtime.DP_GROUP_RANKS],
            },
            "optimizer": {
                "type": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": 0.0,
                "max_gradient_norm": args.max_grad_norm,
            },
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                "single_expert": "transformer_1",
            },
            "runtime": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
            },
            "history_summary": history,
            "artifacts": {
                "adapter.safetensors": runtime.file_sha256(adapter_path),
                "optimizer.pt": runtime.file_sha256(optimizer_path),
                "history.json": runtime.file_sha256(history_path),
            },
            "engineering_experiment_only": True,
            "semantic_action_editing_success": False,
            "video_quality_claim_authorized": False,
            "scientific_generalization_claim_authorized": False,
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
        }
        receipt["receipt_digest"] = runtime.object_sha256(receipt)
        runtime.atomic_json(stage / "receipt.json", receipt)
        runtime.verify_staged_run_bundle(stage, receipt)
        runtime.fsync_directory(stage)
        native_runtime._publish_create_only(stage, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "schedule_steps": args.max_schedule_steps,
                    "optimizer_updates": optimizer_updates,
                    "adapter_parameter_digest": final_digest,
                    "semantic_action_editing_success": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=parallel.world_group)
    if not output.is_dir() or output.is_symlink() or stage.exists():
        raise PairV5NativeDPOTrainingError(
            "atomic output publication did not complete"
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SIGMA_INDICES",
    "CANDIDATE_MEDIA_SCHEMA",
    "MANIFEST_SCHEMA",
    "PAIR_SCHEMA",
    "PairManifest",
    "PairV5NativeDPOTrainingError",
    "TRAINABLE_SIGMA_INDICES",
    "build_parser",
    "build_shared_pair_states",
    "exact40_schedule_index",
    "expected_optimizer_updates",
    "is_frozen_anchor_audit",
    "load_pair_manifest",
    "main",
    "native_vjp_branch_registry",
    "noise_seed",
    "object_sha256",
    "preflight_training_inputs",
    "validate_cli",
    "validate_cross_module_contract",
]
