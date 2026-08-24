#!/usr/bin/env python3
"""Train the PAIR-v5 Bernini Action-LoRA from sealed safe preferences.

This is the executable optimization boundary for PAIR-v5.  A row contains a
source video, the four deploy-time source-frame references, a complete action
caption, and a winner/loser pair selected from full native RV2V-4 rollouts.
The rollout clean latents are DPO endpoints, never visual conditions.  Their
recorded sampler Gaussians are hash-audited provenance and are never loaded.

Each update draws one *fresh* FP32 Gaussian and one high/mid coordinate from
the released exact40 schedule.  That same Gaussian object and physical sigma
are used for the chosen and rejected states.  Student and frozen-reference
predictions both use Bernini's native source-video plus four independently
encoded RGB-reference pack and the same complete caption.  The only trainable
parameters are the block-0..22 cross-attention Q/O Action-LoRA factors.  The
low-sigma route remains the exact frozen policy.  An optional frozen CIO
self-attention Q/O adapter can be loaded before the independent Action-LoRA;
it is active in both student and reference queries and is never optimized.

The full four-field Bernini graph is too large to retain for both endpoints.
As in the validated CIO trainer, this program first evaluates guided outputs
without gradients, differentiates the small reference-corrected flow-DPO
leaf graph, then replays one native field at a time with the exact linear VJP
coefficients.  At most one transformer graph is resident at once.

The optimizer is fail-closed.  It cannot be constructed unless (1) the pinned
self-generated T2V action calibration receipt authorizes optimization and
(2) the safe-Pareto selection receipt replays exactly to a non-empty pair.
No T2V proposal/donor, paired target, mask, flow, pose, track, or trajectory
has a schema slot or model-call path here.

The first executable mode is deliberately one exact81 WORLD8/DP2xSP4 update.
It is an engineering canary, not evidence of successful action editing.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_action_adapter as action_adapter  # noqa: E402
import pair_v5_action_energy_calibration as calibration  # noqa: E402
import pair_v5_candidate_evaluator_packet as evaluator_packet  # noqa: E402
import pair_v5_flow_dpo as flow_dpo  # noqa: E402
import pair_v5_native_bridge as native_bridge  # noqa: E402
import pair_v5_native_rollout_spec as rollout_contract  # noqa: E402
import pair_v5_safe_pareto as safe_pareto  # noqa: E402
import source_self_native_ref_contrastive_v3 as native  # noqa: E402
import source_self_native_rv2v_guidance as guidance  # noqa: E402
import source_self_native_target_adapter as cio_adapter  # noqa: E402
import source_self_runtime as runtime  # noqa: E402
import infer_lora as inference_legacy  # noqa: E402
import train_lora as legacy  # noqa: E402


METHOD_NAME = "bernini-pair-v5-action-preference"
MANIFEST_SCHEMA = "bernini-pair-v5-action-preference-manifest-v2"
PAIR_ROW_SCHEMA = "bernini-pair-v5-action-preference-row-v1"
ROLLOUT_BINDING_SCHEMA = "bernini-pair-v5-rollout-binding-v1"
FILE_BINDING_SCHEMA = "bernini-pair-v5-file-binding-v1"
RUN_RECEIPT_SCHEMA = "bernini-pair-v5-action-preference-training-receipt-v1"
HISTORY_SCHEMA = "bernini-pair-v5-action-preference-history-v1"
ADAPTER_CHECKPOINT_SCHEMA = "bernini-pair-v5-action-lora-checkpoint-v1"
CIO_ADAPTER_CHECKPOINT_SCHEMA = "bernini-native-target-row-qo-lora-checkpoint-v2"

WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FRAME_COUNT = 81
FPS = 25.0
LATENT_CHANNELS = 16
LATENT_PHASES = 21
REFERENCE_INDICES = (0, 27, 53, 80)
ACTION_SIGMA_INDICES = (
    action_adapter.HIGH_SIGMA_INDICES + action_adapter.MID_SIGMA_INDICES
)
DEFAULT_SEED = 20260808
DEFAULT_LEARNING_RATE = 1.0e-6
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_BETA = 1000.0
VJP_REPLAY_RTOL = 2.0e-5
VJP_REPLAY_ATOL = 2.0e-5

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

_FILE_FIELDS = frozenset({"schema_version", "path", "sha256"})
_ROLLOUT_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "candidate_digest",
        "receipt",
        "expected_receipt_digest",
    }
)
_EVALUATOR_PACKET_BINDING_FIELDS = frozenset(
    {"candidate_id", "packet", "expected_packet_digest", "rollout"}
)
_PAIR_FIELDS = frozenset(
    {
        "schema_version",
        "pair_id",
        "source_video",
        "reference_frame_indices",
        "complete_caption",
        "complete_caption_sha256",
        "action_family",
        "chosen_rollout",
        "rejected_rollout",
        "sample_weight",
        "pair_digest",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "calibration_receipt",
        "expected_calibration_receipt_digest",
        "calibration_optimizer_provenance",
        "evaluator_registry",
        "candidate_evaluator_packets",
        "selector_policy",
        "selector_state_before",
        "selector_candidates",
        "selector_action_scores",
        "selector_calibrator_provenance",
        "selector_receipt",
        "expected_selector_receipt_digest",
        "pairs",
        "input_closure",
        "manifest_digest",
    }
)
_INPUT_CLOSURE = {
    "student_visual_conditions": [
        "source_video",
        "source_rgb_refs_0_27_53_80",
    ],
    "student_text_condition": "complete_source_content_caption_with_requested_new_action",
    "chosen_rejected_rollout_latents_are_dpo_endpoints_only": True,
    "recorded_rollout_noise_is_hash_audited_but_never_loaded": True,
    "fresh_shared_gaussian_and_sigma_constructed_inside_training": True,
    "t2v_proposal_media_consumed": False,
    "donor_consumed": False,
    "paired_target_consumed": False,
    "mask_flow_pose_track_trajectory_consumed": False,
}


class PairV5PreferenceTrainingError(RuntimeError):
    """A sealed input or native training operation violated PAIR-v5."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV5PreferenceTrainingError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, expected: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV5PreferenceTrainingError(f"{label} must be a mapping")
    keys = set(value)
    if not all(isinstance(key, str) for key in keys):
        raise PairV5PreferenceTrainingError(f"{label} keys must be strings")
    if keys != set(expected):
        raise PairV5PreferenceTrainingError(
            f"{label} closure differs; missing={sorted(expected - keys)}, "
            f"extra={sorted(keys - expected)}"
        )
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PairV5PreferenceTrainingError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise PairV5PreferenceTrainingError(f"{label} must be lowercase SHA-1")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PairV5PreferenceTrainingError(f"{label} is not a safe identifier")
    return value


def _embedded_digest(value: Mapping[str, Any], field: str, *, label: str) -> str:
    declared = _sha256(value.get(field), label=f"{label} {field}")
    unsigned = dict(value)
    unsigned.pop(field)
    if object_sha256(unsigned) != declared:
        raise PairV5PreferenceTrainingError(f"{label} embedded digest differs")
    return declared


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PairV5PreferenceTrainingError(f"{label} contains {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PairV5PreferenceTrainingError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5PreferenceTrainingError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, dict):
        raise PairV5PreferenceTrainingError(f"{label} root must be one object")
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
        cls, value: str | Path, *, expected_sha256: str, label: str
    ) -> "FileSnapshot":
        expected = _sha256(expected_sha256, label=f"{label} expected SHA")
        requested = Path(value).expanduser()
        if not requested.is_absolute() or requested == Path("/") or requested.is_symlink():
            raise PairV5PreferenceTrainingError(f"{label} must be an absolute plain file")
        try:
            path = requested.resolve(strict=True)
            before = path.stat()
        except OSError as error:
            raise PairV5PreferenceTrainingError(f"cannot stat {label}: {error}") from error
        if path != requested or not path.is_file() or path.is_symlink():
            raise PairV5PreferenceTrainingError(f"{label} path is not canonical/plain")
        digest = runtime.file_sha256(path)
        after = path.stat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise PairV5PreferenceTrainingError(f"{label} changed while hashing")
        if digest != expected:
            raise PairV5PreferenceTrainingError(f"{label} SHA-256 differs")
        return cls(path, *map(int, identity), digest)

    def assert_unchanged(self, *, label: str) -> None:
        observed = FileSnapshot.capture(
            self.path, expected_sha256=self.sha256, label=label
        )
        if observed != self:
            raise PairV5PreferenceTrainingError(f"{label} changed during training")

    def receipt(self) -> Mapping[str, Any]:
        return {
            "path": str(self.path),
            "size": self.size,
            "sha256": self.sha256,
            "pre_post_stat_and_hash_stable": True,
        }


def _file_binding(value: Any, *, label: str) -> FileSnapshot:
    row = _closed(value, _FILE_FIELDS, label=label)
    if row["schema_version"] != FILE_BINDING_SCHEMA:
        raise PairV5PreferenceTrainingError(f"{label} schema differs")
    return FileSnapshot.capture(
        row["path"], expected_sha256=row["sha256"], label=label
    )


def _load_bound_json(snapshot: FileSnapshot, *, label: str) -> dict[str, Any]:
    before = snapshot.path.stat()
    raw = snapshot.path.read_bytes()
    after = snapshot.path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or hashlib.sha256(raw).hexdigest() != snapshot.sha256
    ):
        raise PairV5PreferenceTrainingError(f"{label} changed while reading")
    return _strict_json(raw, label=label)


@dataclass(frozen=True)
class RolloutEndpoint:
    candidate_id: str
    candidate_digest: str
    receipt_snapshot: FileSnapshot
    receipt_digest: str
    clean_latent_snapshot: FileSnapshot
    recorded_noise_snapshot: FileSnapshot
    mp4_snapshot: FileSnapshot
    native_receipt_snapshot: FileSnapshot
    clean_tensor_key: str

    def assert_unchanged(self, *, label: str) -> None:
        self.receipt_snapshot.assert_unchanged(label=f"{label} PAIR receipt")
        self.clean_latent_snapshot.assert_unchanged(label=f"{label} clean latent")
        self.recorded_noise_snapshot.assert_unchanged(label=f"{label} recorded noise")
        self.mp4_snapshot.assert_unchanged(label=f"{label} MP4")
        self.native_receipt_snapshot.assert_unchanged(label=f"{label} native receipt")


def _artifact_snapshot(value: Any, *, label: str) -> FileSnapshot:
    if not isinstance(value, Mapping):
        raise PairV5PreferenceTrainingError(f"{label} artifact must be a mapping")
    return FileSnapshot.capture(
        value.get("path"), expected_sha256=value.get("sha256"), label=label
    )


def _validate_native_receipt(
    snapshot: FileSnapshot,
    *,
    expected_digest: str,
    source_sha256: str,
    caption_sha256: str,
) -> Mapping[str, Any]:
    value = _load_bound_json(snapshot, label="native rollout receipt")
    declared = _sha256(value.get("receipt_digest"), label="native receipt digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest")
    # Bernini's native receipt uses ensure_ascii=True canonical JSON.  ASCII is
    # a strict subset of the UTF-8 canonical encoder used here for these keys.
    if object_sha256(unsigned) != declared or declared != expected_digest:
        raise PairV5PreferenceTrainingError("native rollout receipt digest differs")
    native_input = value.get("input")
    sampling = value.get("sampling", {}).get("rv2v")
    conditioning = value.get("conditioning", {}).get("rv2v")
    if (
        not isinstance(native_input, Mapping)
        or native_input.get("source_video_sha256") != source_sha256
        or native_input.get("action_prompt_utf8_sha256") != caption_sha256
        or native_input.get("target_video") is not False
        or native_input.get("external_reference_image_or_video") is not False
        or native_input.get("external_mask_flow_pose_track_trajectory") is not False
        or not isinstance(sampling, Mapping)
        or sampling.get("num_frames") != FRAME_COUNT
        or sampling.get("num_inference_steps") != 40
        or sampling.get("target_initialization")
        != rollout_contract.TARGET_INITIALIZATION
        or not isinstance(conditioning, Mapping)
        or conditioning.get("full_source_video_count") != 1
        or conditioning.get("source_derived_reference_count") != 4
        or conditioning.get("source_frame_indices") != list(REFERENCE_INDICES)
        or conditioning.get("reference_from_temporal_video_latent_slice") is not False
    ):
        raise PairV5PreferenceTrainingError("native receipt is not deploy-matched RV2V-4")
    return value


def _validate_rollout_binding(
    value: Any,
    *,
    source_sha256: str,
    caption: str,
    caption_sha256: str,
    label: str,
) -> RolloutEndpoint:
    row = _closed(value, _ROLLOUT_FIELDS, label=label)
    if row["schema_version"] != ROLLOUT_BINDING_SCHEMA:
        raise PairV5PreferenceTrainingError(f"{label} schema differs")
    candidate_id = _safe_id(row["candidate_id"], label=f"{label} candidate ID")
    candidate_digest = _sha256(
        row["candidate_digest"], label=f"{label} candidate digest"
    )
    receipt_snapshot = _file_binding(row["receipt"], label=f"{label} receipt")
    receipt = _load_bound_json(receipt_snapshot, label=f"{label} receipt")
    expected_digest = _sha256(
        row["expected_receipt_digest"], label=f"{label} expected receipt digest"
    )
    declared = _sha256(receipt.get("receipt_digest"), label=f"{label} receipt digest")
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest")
    if (
        receipt.get("schema_version") != rollout_contract.RECEIPT_SCHEMA_VERSION
        or object_sha256(unsigned) != declared
        or declared != expected_digest
        or receipt.get("sampling_contract") != rollout_contract.SAMPLING_CONTRACT
        or receipt.get("semantic_input_closure")
        != rollout_contract.SEMANTIC_INPUT_CLOSURE
    ):
        raise PairV5PreferenceTrainingError(f"{label} PAIR rollout receipt differs")
    candidate = receipt.get("candidate")
    if not isinstance(candidate, Mapping):
        raise PairV5PreferenceTrainingError(f"{label} candidate binding is absent")
    normalized = rollout_contract.validate_candidate(candidate)
    if (
        normalized["candidate_id"] != candidate_id
        or normalized["source_video_sha256"] != source_sha256
        or normalized["complete_caption"] != caption
        or normalized["complete_caption_sha256"] != caption_sha256
    ):
        raise PairV5PreferenceTrainingError(
            f"{label} rollout source/caption differs from preference row"
        )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "mp4",
        "predecode_clean_latent",
        "official_initial_gaussian",
    }:
        raise PairV5PreferenceTrainingError(f"{label} artifact closure differs")
    clean_record = artifacts["predecode_clean_latent"]
    noise_record = artifacts["official_initial_gaussian"]
    if (
        not isinstance(clean_record, Mapping)
        or clean_record.get("tensor_key") != "normalized_clean_latent"
        or clean_record.get("shape", [])[:3] != [1, LATENT_CHANNELS, LATENT_PHASES]
        or clean_record.get("stored_dtype") != "torch.float32"
        or clean_record.get("native_sampler_before_vae_decode") is not True
        or not isinstance(noise_record, Mapping)
        or noise_record.get("tensor_key") != "official_initial_gaussian"
        or noise_record.get("shape", [])[:3] != [1, LATENT_CHANNELS, LATENT_PHASES]
    ):
        raise PairV5PreferenceTrainingError(f"{label} latent/noise contract differs")
    clean_snapshot = _artifact_snapshot(clean_record, label=f"{label} clean latent")
    noise_snapshot = _artifact_snapshot(noise_record, label=f"{label} recorded noise")
    mp4_snapshot = _artifact_snapshot(artifacts["mp4"], label=f"{label} MP4")
    native_snapshot = FileSnapshot.capture(
        receipt.get("native_receipt_path"),
        expected_sha256=receipt.get("native_receipt_sha256"),
        label=f"{label} native receipt",
    )
    native_receipt = _validate_native_receipt(
        native_snapshot,
        expected_digest=_sha256(
            receipt.get("native_receipt_digest"),
            label=f"{label} native receipt expected digest",
        ),
        source_sha256=source_sha256,
        caption_sha256=caption_sha256,
    )
    native_clean = native_receipt.get("outputs", {}).get("rv2v", {}).get(
        "normalized_clean_latent"
    )
    native_noise = native_receipt.get("initial_noise_artifacts", {}).get("rv2v")
    native_mp4 = native_receipt.get("outputs", {}).get("rv2v")
    if (
        not isinstance(native_clean, Mapping)
        or native_clean.get("path") != str(clean_snapshot.path)
        or native_clean.get("sha256") != clean_snapshot.sha256
        or not isinstance(native_noise, Mapping)
        or native_noise.get("path") != str(noise_snapshot.path)
        or native_noise.get("sha256") != noise_snapshot.sha256
        or not isinstance(native_mp4, Mapping)
        or native_mp4.get("path") != str(mp4_snapshot.path)
        or native_mp4.get("sha256") != mp4_snapshot.sha256
    ):
        raise PairV5PreferenceTrainingError(
            f"{label} PAIR/native artifact binding differs"
        )
    return RolloutEndpoint(
        candidate_id=candidate_id,
        candidate_digest=candidate_digest,
        receipt_snapshot=receipt_snapshot,
        receipt_digest=declared,
        clean_latent_snapshot=clean_snapshot,
        recorded_noise_snapshot=noise_snapshot,
        mp4_snapshot=mp4_snapshot,
        native_receipt_snapshot=native_snapshot,
        clean_tensor_key="normalized_clean_latent",
    )


@dataclass(frozen=True)
class PreferenceRow:
    pair_id: str
    source_video_snapshot: FileSnapshot
    complete_caption: str
    complete_caption_sha256: str
    chosen: RolloutEndpoint
    rejected: RolloutEndpoint
    sample_weight: float
    pair_digest: str

    def assert_unchanged(self) -> None:
        self.source_video_snapshot.assert_unchanged(
            label=f"{self.pair_id} source video"
        )
        self.chosen.assert_unchanged(label=f"{self.pair_id} chosen")
        self.rejected.assert_unchanged(label=f"{self.pair_id} rejected")


@dataclass(frozen=True)
class PreferenceManifest:
    snapshot: FileSnapshot
    manifest_id: str
    manifest_digest: str
    calibration_snapshot: FileSnapshot
    calibration_receipt: Mapping[str, Any]
    calibration_optimizer_provenance: Mapping[str, Any]
    evaluator_registry: Mapping[str, Any]
    evaluator_packets: Mapping[str, Mapping[str, Any]]
    evaluator_packet_snapshots: tuple[FileSnapshot, ...]
    selector_snapshot: FileSnapshot
    selector_receipt: Mapping[str, Any]
    selector_policy: Mapping[str, Any]
    rows: tuple[PreferenceRow, ...]

    def assert_unchanged(self) -> None:
        self.snapshot.assert_unchanged(label="preference manifest")
        self.calibration_snapshot.assert_unchanged(label="calibration receipt")
        for snapshot in self.evaluator_packet_snapshots:
            snapshot.assert_unchanged(label="candidate evaluator packet")
        self.selector_snapshot.assert_unchanged(label="selector receipt")
        for row in self.rows:
            row.assert_unchanged()


def load_preference_manifest(
    path_value: str | Path, *, expected_sha256: str
) -> PreferenceManifest:
    """Load, replay, and bind the complete optimizer authorization packet."""

    manifest_snapshot = FileSnapshot.capture(
        path_value, expected_sha256=expected_sha256, label="preference manifest"
    )
    value = _load_bound_json(manifest_snapshot, label="preference manifest")
    row = _closed(value, _MANIFEST_FIELDS, label="preference manifest")
    if row["schema_version"] != MANIFEST_SCHEMA:
        raise PairV5PreferenceTrainingError("preference manifest schema differs")
    manifest_id = _safe_id(row["manifest_id"], label="manifest_id")
    manifest_digest = _embedded_digest(row, "manifest_digest", label="manifest")
    if row["input_closure"] != _INPUT_CLOSURE:
        raise PairV5PreferenceTrainingError("manifest admits a forbidden model input")

    calibration_snapshot = _file_binding(
        row["calibration_receipt"], label="calibration receipt"
    )
    calibration_value = _load_bound_json(
        calibration_snapshot, label="calibration receipt"
    )
    try:
        checked_calibration = calibration.validate_calibration_receipt(calibration_value)
    except calibration.PairV5CalibrationError as error:
        raise PairV5PreferenceTrainingError(str(error)) from error
    expected_calibration_digest = _sha256(
        row["expected_calibration_receipt_digest"],
        label="expected calibration receipt digest",
    )
    if (
        checked_calibration["receipt_digest"] != expected_calibration_digest
        or checked_calibration["optimizer_authorized"] is not True
        or not checked_calibration["mapping_by_family"]
    ):
        raise PairV5PreferenceTrainingError(
            "self-generated action calibration did not authorize optimization"
        )
    try:
        calibration_optimizer_provenance = (
            calibration.validate_calibrator_provenance(
                row["calibration_optimizer_provenance"]
            )
        )
    except calibration.PairV5CalibrationError as error:
        raise PairV5PreferenceTrainingError(str(error)) from error
    if (
        calibration_optimizer_provenance["optimizer_authorized"] is not True
        or calibration_optimizer_provenance["calibration_receipt_digest"]
        != checked_calibration["receipt_digest"]
        or calibration_optimizer_provenance["calibrator_id"]
        != checked_calibration["calibrator_id"]
    ):
        raise PairV5PreferenceTrainingError(
            "calibration optimizer provenance differs from the pinned receipt"
        )

    try:
        checked_evaluator_registry = evaluator_packet.validate_registry(
            row["evaluator_registry"]
        )
    except evaluator_packet.PairV5EvaluatorPacketError as error:
        raise PairV5PreferenceTrainingError(str(error)) from error
    raw_packet_bindings = row["candidate_evaluator_packets"]
    if not isinstance(raw_packet_bindings, list) or not raw_packet_bindings:
        raise PairV5PreferenceTrainingError(
            "manifest requires candidate evaluator packet bindings"
        )
    evaluator_packets: dict[str, Mapping[str, Any]] = {}
    evaluator_rollout_bindings: dict[str, Mapping[str, Any]] = {}
    evaluator_packet_snapshots: list[FileSnapshot] = []
    for raw_binding in raw_packet_bindings:
        binding = _closed(
            raw_binding,
            _EVALUATOR_PACKET_BINDING_FIELDS,
            label="candidate evaluator packet binding",
        )
        candidate_id = _safe_id(
            binding["candidate_id"], label="evaluator packet candidate ID"
        )
        if candidate_id in evaluator_packets:
            raise PairV5PreferenceTrainingError(
                "candidate evaluator packet binding is duplicated"
            )
        snapshot = _file_binding(
            binding["packet"], label=f"{candidate_id} evaluator packet"
        )
        packet_value = _load_bound_json(
            snapshot, label=f"{candidate_id} evaluator packet"
        )
        try:
            checked_packet = evaluator_packet.validate_packet(packet_value)
        except evaluator_packet.PairV5EvaluatorPacketError as error:
            raise PairV5PreferenceTrainingError(str(error)) from error
        expected_packet_digest = _sha256(
            binding["expected_packet_digest"],
            label=f"{candidate_id} expected evaluator packet digest",
        )
        if (
            checked_packet["candidate_id"] != candidate_id
            or checked_packet["packet_digest"] != expected_packet_digest
            or checked_packet["evaluator_registry_digest"]
            != checked_evaluator_registry["registry_digest"]
        ):
            raise PairV5PreferenceTrainingError(
                f"{candidate_id} evaluator packet binding differs"
            )
        evaluator_packets[candidate_id] = checked_packet
        evaluator_rollout_bindings[candidate_id] = _closed(
            binding["rollout"],
            _ROLLOUT_FIELDS,
            label=f"{candidate_id} evaluator-bound rollout",
        )
        evaluator_packet_snapshots.append(snapshot)

    try:
        policy = safe_pareto.validate_policy(row["selector_policy"])
        state = safe_pareto.validate_state(row["selector_state_before"], policy)
        candidates = [
            safe_pareto.validate_candidate(item)
            for item in row["selector_candidates"]
        ]
        action_scores = [
            calibration.validate_rv2v_candidate_score(item)
            for item in row["selector_action_scores"]
        ]
        provenance = safe_pareto.validate_calibrator_provenance(
            row["selector_calibrator_provenance"]
        )
    except (TypeError, safe_pareto.PairV5ContractError) as error:
        raise PairV5PreferenceTrainingError(str(error)) from error
    except calibration.PairV5CalibrationError as error:
        raise PairV5PreferenceTrainingError(str(error)) from error
    candidates_by_id = {item["candidate_id"]: item for item in candidates}
    action_scores_by_id = {item["candidate_id"]: item for item in action_scores}
    try:
        replayed_action_scores = {
            candidate_id: calibration.apply_calibrator(
                score["raw_candidate_own_score"],
                score["action_family"],
                checked_calibration,
                registered_calibration_receipt_digest=checked_calibration[
                    "receipt_digest"
                ],
            )
            for candidate_id, score in action_scores_by_id.items()
        }
    except calibration.PairV5CalibrationError as error:
        raise PairV5PreferenceTrainingError(str(error)) from error
    if (
        len(candidates_by_id) != len(candidates)
        or len(action_scores_by_id) != len(action_scores)
        or set(action_scores_by_id) != set(candidates_by_id)
        or set(evaluator_packets) != set(candidates_by_id)
        or tuple(evaluator_packet.HARD_NEGATIVE_FLAGS)
        != tuple(safe_pareto.HARD_NEGATIVE_FLAGS)
        or any(
            score["calibration_receipt_digest"]
            != checked_calibration["receipt_digest"]
            or score["frozen_scorer_receipt_digest"]
            != checked_calibration["frozen_scorer_receipt_digest"]
            or replayed_action_scores[candidate_id]
            != score["calibrated_action_score"]
            or score["calibrated_action_score"]
            != candidates_by_id[candidate_id]["action_score"]
            for candidate_id, score in action_scores_by_id.items()
        )
    ):
        raise PairV5PreferenceTrainingError(
            "selector action scores do not close to calibrated candidate-own receipts"
        )
    if (
        provenance["calibration_receipt_digest"]
        != checked_calibration["receipt_digest"]
        or provenance["calibrator_id"] != checked_calibration["calibrator_id"]
        or provenance["calibration_receipt_sha256"] != calibration_snapshot.sha256
    ):
        raise PairV5PreferenceTrainingError(
            "safe selector provenance binds a different action calibrator"
        )

    candidate_rollout_endpoints: dict[str, RolloutEndpoint] = {}
    for candidate_id, candidate_record in candidates_by_id.items():
        rollout_binding = evaluator_rollout_bindings[candidate_id]
        receipt_binding = _closed(
            rollout_binding["receipt"], _FILE_FIELDS, label=f"{candidate_id} receipt"
        )
        receipt_snapshot = _file_binding(
            receipt_binding, label=f"{candidate_id} evaluator-bound receipt"
        )
        receipt_value = _load_bound_json(
            receipt_snapshot, label=f"{candidate_id} evaluator-bound receipt"
        )
        rollout_candidate = receipt_value.get("candidate")
        if not isinstance(rollout_candidate, Mapping):
            raise PairV5PreferenceTrainingError(
                f"{candidate_id} evaluator-bound rollout lacks candidate"
            )
        try:
            normalized_rollout_candidate = rollout_contract.validate_candidate(
                rollout_candidate
            )
        except rollout_contract.PairRolloutSpecError as error:
            raise PairV5PreferenceTrainingError(str(error)) from error
        packet = evaluator_packets[candidate_id]
        endpoint = _validate_rollout_binding(
            rollout_binding,
            source_sha256=packet["source_video_sha256"],
            caption=normalized_rollout_candidate["complete_caption"],
            caption_sha256=packet["complete_caption_sha256"],
            label=f"{candidate_id} evaluator-bound rollout",
        )
        if endpoint.candidate_digest != candidate_record["candidate_digest"]:
            raise PairV5PreferenceTrainingError(
                f"{candidate_id} safe candidate digest is not bound to rollout"
            )
        try:
            evaluator_packet.verify_packet(
                packet,
                registry=checked_evaluator_registry,
                safe_candidate=candidate_record,
                action_score_receipt=action_scores_by_id[candidate_id],
                rollout_receipt_digest=endpoint.receipt_digest,
                mp4_sha256=endpoint.mp4_snapshot.sha256,
                source_video_sha256=packet["source_video_sha256"],
                complete_caption_sha256=packet["complete_caption_sha256"],
                expected_action_evaluator_sha256=provenance[
                    "action_evaluator_sha256"
                ],
                expected_action_model_digest=checked_calibration[
                    "frozen_scorer_receipt_digest"
                ],
            )
        except evaluator_packet.PairV5EvaluatorPacketError as error:
            raise PairV5PreferenceTrainingError(str(error)) from error
        candidate_rollout_endpoints[candidate_id] = endpoint

    selector_snapshot = _file_binding(
        row["selector_receipt"], label="selector receipt"
    )
    selector_value = _load_bound_json(selector_snapshot, label="selector receipt")
    try:
        selector_receipt = safe_pareto.replay_and_verify_receipt(
            selector_value,
            state=state,
            candidates=candidates,
            policy=policy,
            calibrator_provenance=provenance,
        )
    except safe_pareto.PairV5ContractError as error:
        raise PairV5PreferenceTrainingError(str(error)) from error
    expected_selector_digest = _sha256(
        row["expected_selector_receipt_digest"],
        label="expected selector receipt digest",
    )
    selected = selector_receipt["selected_pair"]
    if (
        selector_receipt["receipt_digest"] != expected_selector_digest
        or selected is None
        or selector_receipt["decision"] == "no_eligible_pair"
    ):
        raise PairV5PreferenceTrainingError(
            "safe-Pareto selector emitted no optimizer-authorized preference pair"
        )

    raw_pairs = row["pairs"]
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise PairV5PreferenceTrainingError("manifest requires at least one pair row")
    rows: list[PreferenceRow] = []
    seen_pair_ids: set[str] = set()
    selected_key = (
        selected["winner_candidate_id"],
        selected["winner_candidate_digest"],
        selected["loser_candidate_id"],
        selected["loser_candidate_digest"],
    )
    observed_keys: set[tuple[str, str, str, str]] = set()
    for raw_pair in raw_pairs:
        pair = _closed(raw_pair, _PAIR_FIELDS, label="preference row")
        if pair["schema_version"] != PAIR_ROW_SCHEMA:
            raise PairV5PreferenceTrainingError("preference row schema differs")
        pair_id = _safe_id(pair["pair_id"], label="pair_id")
        if pair_id in seen_pair_ids:
            raise PairV5PreferenceTrainingError("pair_id is duplicated")
        seen_pair_ids.add(pair_id)
        pair_digest = _embedded_digest(pair, "pair_digest", label=f"pair {pair_id}")
        source_snapshot = _file_binding(
            pair["source_video"], label=f"{pair_id} source video"
        )
        if pair["reference_frame_indices"] != list(REFERENCE_INDICES):
            raise PairV5PreferenceTrainingError(
                f"{pair_id} references must be RGB frames 0,27,53,80"
            )
        caption = pair["complete_caption"]
        if not isinstance(caption, str) or not caption.strip() or "\x00" in caption:
            raise PairV5PreferenceTrainingError(f"{pair_id} caption is invalid")
        caption_digest = _sha256(
            pair["complete_caption_sha256"], label=f"{pair_id} caption SHA"
        )
        if hashlib.sha256(caption.encode("utf-8")).hexdigest() != caption_digest:
            raise PairV5PreferenceTrainingError(f"{pair_id} caption SHA differs")
        action_family = pair["action_family"]
        if (
            not isinstance(action_family, str)
            or action_family not in checked_calibration["action_family_order"]
        ):
            raise PairV5PreferenceTrainingError(
                f"{pair_id} action family is outside the calibrated registry"
            )
        weight = pair["sample_weight"]
        if type(weight) is not float or not math.isfinite(weight) or weight <= 0.0:
            raise PairV5PreferenceTrainingError(
                f"{pair_id} sample_weight must be a positive JSON float"
            )
        chosen_binding = _closed(
            pair["chosen_rollout"], _ROLLOUT_FIELDS, label=f"{pair_id} chosen"
        )
        rejected_binding = _closed(
            pair["rejected_rollout"], _ROLLOUT_FIELDS, label=f"{pair_id} rejected"
        )
        chosen_id = _safe_id(
            chosen_binding["candidate_id"], label=f"{pair_id} chosen candidate ID"
        )
        rejected_id = _safe_id(
            rejected_binding["candidate_id"], label=f"{pair_id} rejected candidate ID"
        )
        if (
            chosen_id not in candidate_rollout_endpoints
            or rejected_id not in candidate_rollout_endpoints
            or dict(chosen_binding) != dict(evaluator_rollout_bindings[chosen_id])
            or dict(rejected_binding) != dict(evaluator_rollout_bindings[rejected_id])
        ):
            raise PairV5PreferenceTrainingError(
                f"{pair_id} endpoint differs from evaluator-bound rollout"
            )
        chosen = candidate_rollout_endpoints[chosen_id]
        rejected = candidate_rollout_endpoints[rejected_id]
        if any(
            evaluator_packets[endpoint.candidate_id]["source_video_sha256"]
            != source_snapshot.sha256
            or evaluator_packets[endpoint.candidate_id]["complete_caption_sha256"]
            != caption_digest
            for endpoint in (chosen, rejected)
        ):
            raise PairV5PreferenceTrainingError(
                f"{pair_id} evaluator packet source/caption differs"
            )
        key = (
            chosen.candidate_id,
            chosen.candidate_digest,
            rejected.candidate_id,
            rejected.candidate_digest,
        )
        if key in observed_keys:
            raise PairV5PreferenceTrainingError("selected candidate pair is duplicated")
        observed_keys.add(key)
        if (
            action_scores_by_id[chosen.candidate_id]["action_family"]
            != action_family
            or action_scores_by_id[rejected.candidate_id]["action_family"]
            != action_family
        ):
            raise PairV5PreferenceTrainingError(
                f"{pair_id} endpoints use a different calibrated action family"
            )
        rows.append(
            PreferenceRow(
                pair_id,
                source_snapshot,
                caption,
                caption_digest,
                chosen,
                rejected,
                weight,
                pair_digest,
            )
        )
    if observed_keys != {selected_key}:
        # v1 intentionally binds one selection event to exactly one row.  A
        # future batched format must carry/replay one receipt per pair.
        raise PairV5PreferenceTrainingError(
            "v1 manifest pair rows do not close exactly to the selected safe pair"
        )
    manifest_snapshot.assert_unchanged(label="preference manifest")
    return PreferenceManifest(
        manifest_snapshot,
        manifest_id,
        manifest_digest,
        calibration_snapshot,
        checked_calibration,
        calibration_optimizer_provenance,
        checked_evaluator_registry,
        dict(evaluator_packets),
        tuple(evaluator_packet_snapshots),
        selector_snapshot,
        selector_receipt,
        policy,
        tuple(rows),
    )


def registered_action_sigma_index(
    *, seed: int, step: int, pair_digest: str, dp_rank: int
) -> int:
    """Choose only a pre-registered high/mid exact40 coordinate."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**63
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or isinstance(dp_rank, bool)
        or not isinstance(dp_rank, int)
        or not 0 <= dp_rank < DP_SIZE
    ):
        raise PairV5PreferenceTrainingError("sigma selector arguments differ")
    _sha256(pair_digest, label="pair digest")
    material = f"{seed}\0pair-v5-action-dpo\0{step}\0{pair_digest}\0{dp_rank}".encode(
        "ascii"
    )
    ordinal = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    index = ACTION_SIGMA_INDICES[ordinal % len(ACTION_SIGMA_INDICES)]
    if action_adapter.sigma_gate(index)[1] <= 0.0:
        raise PairV5PreferenceTrainingError("training selected a low-sigma base-only row")
    return index


def fresh_shared_epsilon(
    shape: Sequence[int], *, seed: int, device: Any
) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return value.to(device=device).contiguous().detach()


def _fresh_noise_seed(base_seed: int, step: int, pair_digest: str, dp_rank: int) -> int:
    material = f"{base_seed}\0pair-v5-fresh-epsilon\0{step}\0{pair_digest}\0{dp_rank}".encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


def _load_clean_latent(endpoint: RolloutEndpoint) -> Any:
    import torch
    from safetensors import safe_open

    endpoint.clean_latent_snapshot.assert_unchanged(label="clean latent before load")
    with safe_open(
        str(endpoint.clean_latent_snapshot.path), framework="pt", device="cpu"
    ) as opened:
        if list(opened.keys()) != [endpoint.clean_tensor_key]:
            raise PairV5PreferenceTrainingError("clean latent safetensors key differs")
        value = opened.get_tensor(endpoint.clean_tensor_key).contiguous()
        metadata = dict(opened.metadata() or {})
    if (
        value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3])
        != (1, LATENT_CHANNELS, LATENT_PHASES)
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or int(value.shape[3]) % 2
        or int(value.shape[4]) % 2
        or not bool(torch.isfinite(value).all().item())
        or metadata.get("coordinate") != "bernini_normalized_clean_vae_latent"
        or metadata.get("frame_contract") != "exact81_latent21"
        or metadata.get("artifact_role") != "native_sampler_proposal"
    ):
        raise PairV5PreferenceTrainingError("clean latent tensor/metadata differs")
    return value.detach()


def _load_optional_frozen_cio(
    transformer: Any,
    path_value: Optional[str],
    expected_sha256: Optional[str],
) -> tuple[
    Optional[cio_adapter.NativeTargetAdapterHandle],
    Mapping[str, Any],
    Optional[FileSnapshot],
]:
    import torch
    from safetensors import safe_open

    if path_value in (None, "") and expected_sha256 in (None, ""):
        return None, {"loaded": False}, None
    if not path_value or not expected_sha256:
        raise PairV5PreferenceTrainingError(
            "frozen CIO path and SHA-256 must be supplied together"
        )
    snapshot = FileSnapshot.capture(
        path_value, expected_sha256=expected_sha256, label="frozen CIO adapter"
    )
    handle = cio_adapter.install_native_target_adapter(
        transformer,
        rank=8,
        alpha=8.0,
        block_indices=cio_adapter.DEFAULT_BLOCK_INDICES,
    )
    named = dict(handle.trainable_named_parameters())
    with safe_open(str(snapshot.path), framework="pt", device="cpu") as opened:
        keys = list(opened.keys())
        state = {name: opened.get_tensor(name).contiguous() for name in keys}
        metadata = dict(opened.metadata() or {})
    if set(state) != set(named):
        raise PairV5PreferenceTrainingError("frozen CIO adapter key closure differs")
    with torch.no_grad():
        for name, parameter in named.items():
            value = state[name]
            if (
                value.dtype != torch.float32
                or tuple(value.shape) != tuple(parameter.shape)
                or not bool(torch.isfinite(value).all().item())
            ):
                raise PairV5PreferenceTrainingError(
                    f"frozen CIO tensor differs: {name}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    receipt_before_freeze = dict(handle.receipt())
    expected_metadata = {
        "schema_version": CIO_ADAPTER_CHECKPOINT_SCHEMA,
        "adapter_contract_digest": str(receipt_before_freeze["digest"]),
        "block_indices_json": canonical_json_bytes(
            list(cio_adapter.DEFAULT_BLOCK_INDICES)
        ).decode("ascii"),
        "rank": "8",
        "alpha_hex": float(8.0).hex(),
        "rho_hex": float(0.0).hex(),
        "native_guidance_digest": str(guidance.guidance_receipt()["digest"]),
        "native_schedule_digest": str(
            native.native_unipc40_schedule_receipt()["digest"]
        ),
        "native_rv2v4_reference_contract_digest": str(
            native.native_rv2v4_reference_contract()["digest"]
        ),
        "reference_rgb_indices_json": canonical_json_bytes(
            list(REFERENCE_INDICES)
        ).decode("ascii"),
        "gradient_checkpointing_enabled": "false",
        "adapter_activation_schedule": "all_40_native_unipc_forward_coordinates",
        "inference_requires_same_target_route_and_rho": "true",
    }
    if metadata != expected_metadata:
        raise PairV5PreferenceTrainingError(
            "frozen CIO checkpoint metadata/contract differs"
        )
    for parameter in named.values():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise PairV5PreferenceTrainingError("frozen CIO left trainable parameters")
    return handle, {
        "loaded": True,
        "file": dict(snapshot.receipt()),
        "adapter_contract_before_freeze": receipt_before_freeze,
        "metadata": metadata,
        "active_in_student_and_reference": True,
        "optimized": False,
        "metadata_validated_exactly": True,
    }, snapshot


def _native_rows(
    pack: native.NativeRV2VPack,
    *,
    cond_embeds: Any,
    uncond_embeds: Any,
) -> tuple[tuple[str, native.NativeRV2VBranch, Any, float], ...]:
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
    if tuple(item[0] for item in rows) != tuple(
        guidance.guidance_receipt()["forward_order"]
    ) or not math.isclose(
        sum(item[3] for item in rows), 1.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise PairV5PreferenceTrainingError("native RV2V VJP registry differs")
    return rows


def _route_stack(
    stack: ExitStack,
    *,
    branch: native.NativeRV2VBranch,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Optional[cio_adapter.NativeTargetAdapterHandle],
    sp_rank: int,
    sigma_index: int,
    action_enabled: bool,
) -> None:
    if cio_handle is not None:
        stack.enter_context(
            cio_handle.route(
                cio_adapter.NativeTargetRoute(
                    total_tokens=branch.total_tokens,
                    condition_tokens=branch.condition_tokens,
                    sequence_parallel_rank=sp_rank,
                    sequence_parallel_size=SP_SIZE,
                    branch_name=branch.name,
                    enabled=True,
                )
            )
        )
    stack.enter_context(
        action_handle.route(
            action_adapter.PairV5ActionRoute(
                total_tokens=branch.total_tokens,
                condition_tokens=branch.condition_tokens,
                sequence_parallel_rank=sp_rank,
                sequence_parallel_size=SP_SIZE,
                branch_name=branch.name,
                sigma_schedule_index=sigma_index,
                enabled=action_enabled,
            )
        )
    )


def _build_pack(
    transformer: Any,
    source_video: Any,
    references: Sequence[Any],
    x_sigma: Any,
) -> native.NativeRV2VPack:
    import torch

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return native.build_native_rv2v_pack(
            transformer,
            donor_video=source_video,
            image_references=references,
            noisy_target=x_sigma,
        )


def _forward_native_branch(
    diffusion: Any,
    branch: native.NativeRV2VBranch,
    *,
    timestep: Any,
    text: Any,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Optional[cio_adapter.NativeTargetAdapterHandle],
    sp_rank: int,
    sigma_index: int,
    action_enabled: bool,
) -> Any:
    with ExitStack() as stack:
        _route_stack(
            stack,
            branch=branch,
            action_handle=action_handle,
            cio_handle=cio_handle,
            sp_rank=sp_rank,
            sigma_index=sigma_index,
            action_enabled=action_enabled,
        )
        return native.forward_native_target_branch(
            diffusion, branch, timestep=timestep, cond_embeds=text
        )


def _guided_prediction_no_grad(
    diffusion: Any,
    pack: native.NativeRV2VPack,
    *,
    timestep: Any,
    cond_embeds: Any,
    uncond_embeds: Any,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Optional[cio_adapter.NativeTargetAdapterHandle],
    sp_rank: int,
    sigma_index: int,
    action_enabled: bool,
    video_shape: Sequence[int],
) -> Any:
    import torch

    components: dict[str, Any] = {}
    with torch.no_grad():
        for name, branch, text, _ in _native_rows(
            pack, cond_embeds=cond_embeds, uncond_embeds=uncond_embeds
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                components[name] = _forward_native_branch(
                    diffusion,
                    branch,
                    timestep=timestep,
                    text=text,
                    action_handle=action_handle,
                    cio_handle=cio_handle,
                    sp_rank=sp_rank,
                    sigma_index=sigma_index,
                    action_enabled=action_enabled,
                )
    none = components["none_uncond"]
    video = components["V_uncond"]
    vi_u = components["VI_uncond"]
    vi_c = components["VI_cond"]
    guided = (
        none
        + guidance.OMEGA_VIDEO * (video - none)
        + guidance.OMEGA_IMAGE * (vi_u - video)
        + guidance.OMEGA_TEXT * (vi_c - vi_u)
    )
    return native_bridge._unpack_spatial_velocity(
        guided.float(), video_shape=video_shape
    ).detach()


def _replay_prediction_vjp(
    diffusion: Any,
    transformer: Any,
    *,
    source_video: Any,
    references: Sequence[Any],
    x_sigma: Any,
    timestep: Any,
    cond_embeds: Any,
    uncond_embeds: Any,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Optional[cio_adapter.NativeTargetAdapterHandle],
    sp_rank: int,
    sigma_index: int,
    output_cotangent: Any,
    expected_guided: Any,
) -> float:
    import torch

    if (
        output_cotangent.shape != x_sigma.shape
        or output_cotangent.requires_grad
        or not bool(torch.isfinite(output_cotangent).all().item())
    ):
        raise PairV5PreferenceTrainingError("guided output cotangent differs")
    pack = _build_pack(transformer, source_video, references, x_sigma)
    replay: dict[str, Any] = {}
    for name, branch, text, coefficient in _native_rows(
        pack, cond_embeds=cond_embeds, uncond_embeds=uncond_embeds
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed = _forward_native_branch(
                diffusion,
                branch,
                timestep=timestep,
                text=text,
                action_handle=action_handle,
                cio_handle=cio_handle,
                sp_rank=sp_rank,
                sigma_index=sigma_index,
                action_enabled=True,
            )
            spatial = native_bridge._unpack_spatial_velocity(
                packed, video_shape=x_sigma.shape
            )
        replay[name] = spatial.detach()
        torch.autograd.backward(
            spatial,
            grad_tensors=output_cotangent.to(spatial.dtype) * float(coefficient),
        )
    none = replay["none_uncond"]
    video = replay["V_uncond"]
    vi_u = replay["VI_uncond"]
    vi_c = replay["VI_cond"]
    replay_guided = (
        none
        + guidance.OMEGA_VIDEO * (video - none)
        + guidance.OMEGA_IMAGE * (vi_u - video)
        + guidance.OMEGA_TEXT * (vi_c - vi_u)
    ).float()
    difference = (replay_guided - expected_guided.float()).abs()
    maximum = float(difference.max().item())
    scale = float(expected_guided.float().abs().max().item())
    if maximum > VJP_REPLAY_ATOL + VJP_REPLAY_RTOL * scale:
        raise PairV5PreferenceTrainingError(
            f"native VJP replay changed guided prediction: max_abs={maximum} scale={scale}"
        )
    return maximum


def _tokenize_positive(tokenizer: Any, text: str) -> tuple[Any, Any]:
    import torch

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids, mask = encoded.input_ids, encoded.attention_mask
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[0] != 1:
        raise PairV5PreferenceTrainingError("positive tokenization differs")
    if ids.shape[1] >= 512:
        return ids[:, :512], mask[:, :512]
    padding = 512 - ids.shape[1]
    return (
        torch.cat((ids, ids.new_zeros((1, padding))), dim=1),
        torch.cat((mask, mask.new_zeros((1, padding))), dim=1),
    )


def _tokenize_negative(tokenizer: Any, text: str) -> tuple[Any, Any]:
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    if tuple(encoded.input_ids.shape) != (1, 512):
        raise PairV5PreferenceTrainingError("negative tokenization differs")
    return encoded.input_ids, encoded.attention_mask


def _broadcast_within_sp(value: Any, *, parallel: runtime.ParallelContext) -> None:
    import torch.distributed as dist

    source_rank = runtime.SP_GROUP_RANKS[parallel.contract.arm_index][0]
    dist.broadcast(value, src=source_rank, group=parallel.sp_group)


def _frozen_text_embedding_cache(
    renderer: Any,
    tokenizer: Any,
    rows: Sequence[PreferenceRow],
    *,
    device: Any,
    parallel: runtime.ParallelContext,
    build_task_prompt: Any,
    prompt_cleaner: Any,
) -> tuple[Mapping[str, tuple[Any, Any]], Mapping[str, Any]]:
    import torch

    negative_ids, negative_mask = _tokenize_negative(
        tokenizer, inference_legacy.DEFAULT_NEGATIVE_PROMPT
    )
    cache: dict[str, tuple[Any, Any]] = {}
    receipts: dict[str, Any] = {}
    with torch.inference_mode():
        unconditional = renderer.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
        _broadcast_within_sp(unconditional, parallel=parallel)
        for row in rows:
            deployment_prompt = build_task_prompt(
                "rv2v", row.complete_caption, prompt_cleaner=prompt_cleaner
            )
            positive_ids, positive_mask = _tokenize_positive(
                tokenizer, deployment_prompt
            )
            conditional = renderer.encode_prompt(
                positive_ids.to(device), positive_mask.to(device)
            ).detach()
            _broadcast_within_sp(conditional, parallel=parallel)
            if (
                conditional.shape != unconditional.shape
                or tuple(conditional.shape) != (1, 512, 4096)
                or conditional.requires_grad
                or not bool(torch.isfinite(conditional).all().item())
            ):
                raise PairV5PreferenceTrainingError("frozen prompt embedding differs")
            cache[row.pair_id] = (
                conditional.cpu().contiguous(),
                unconditional.cpu().contiguous(),
            )
            receipts[row.pair_id] = {
                "complete_caption_sha256": row.complete_caption_sha256,
                "deployment_prompt_sha256": hashlib.sha256(
                    deployment_prompt.encode("utf-8")
                ).hexdigest(),
                "conditional_tensor_sha256": runtime.tensor_sha256(conditional),
                "unconditional_tensor_sha256": runtime.tensor_sha256(unconditional),
            }
    return cache, receipts


def _source_condition_cache(
    vae: Any,
    rows: Sequence[PreferenceRow],
    *,
    device: Any,
    parallel: runtime.ParallelContext,
    source_audit: Any,
    vae_encode: Any,
) -> tuple[Mapping[str, tuple[Any, tuple[Any, ...]]], Mapping[str, Any]]:
    import torch

    cache: dict[str, tuple[Any, tuple[Any, ...]]] = {}
    receipts: dict[str, Any] = {}
    vae.to(device)
    for row in rows:
        pixels, metadata, digest = source_audit.prepare_hashed_source_snapshot(
            row.source_video_snapshot.path
        )
        if digest != row.source_video_snapshot.sha256:
            raise PairV5PreferenceTrainingError("source decoder observed different bytes")
        if (
            metadata.get("frame_count") != FRAME_COUNT
            or float(metadata.get("fps", -1.0)) != FPS
        ):
            raise PairV5PreferenceTrainingError("source video is not exact81/25fps")
        pixels = pixels.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            source = vae_encode(vae, pixels).float().contiguous()
            refs = tuple(
                vae_encode(
                    vae, pixels[:, :, index : index + 1, :, :].contiguous()
                ).float().contiguous()
                for index in REFERENCE_INDICES
            )
        for value in (source, *refs):
            _broadcast_within_sp(value, parallel=parallel)
        if (
            tuple(source.shape[:3]) != (1, LATENT_CHANNELS, LATENT_PHASES)
            or any(tuple(ref.shape[:3]) != (1, LATENT_CHANNELS, 1) for ref in refs)
            or any(ref.shape[3:] != source.shape[3:] for ref in refs)
        ):
            raise PairV5PreferenceTrainingError("native source/reference geometry differs")
        cache[row.pair_id] = (
            source.cpu().contiguous(),
            tuple(ref.cpu().contiguous() for ref in refs),
        )
        receipts[row.pair_id] = {
            "source_video": dict(row.source_video_snapshot.receipt()),
            "source_metadata": dict(metadata),
            "source_latent_sha256": runtime.tensor_sha256(source),
            "reference_indices": list(REFERENCE_INDICES),
            "reference_latent_sha256": [runtime.tensor_sha256(ref) for ref in refs],
            "reference_encodes_independent_from_full_video": True,
            "authoritative_sp_group_rank": runtime.SP_GROUP_RANKS[
                parallel.contract.arm_index
            ][0],
        }
        del pixels, source, refs
        torch.cuda.empty_cache()
    vae.to("cpu")
    return cache, receipts


def _save_action_adapter(
    path: Path,
    handle: action_adapter.PairV5ActionAdapterHandle,
) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    state = dict(handle.state_dict_for_save())
    metadata = {
        "schema_version": ADAPTER_CHECKPOINT_SCHEMA,
        "action_adapter_contract_digest": str(handle.receipt()["digest"]),
        "flow_dpo_contract_digest": str(flow_dpo.contract_receipt()["digest"]),
        "native_bridge_contract_digest": str(
            native_bridge.bridge_contract_receipt()["digest"]
        ),
        "frame_count": str(FRAME_COUNT),
        "reference_indices": "0,27,53,80",
        "sigma_activation": "exact40_indices_0_through_37_high_mid_only",
        "low_sigma_indices_38_39": "direct_frozen_policy",
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".safetensors", delete=False
    ) as temporary_file:
        temporary = Path(temporary_file.name)
    try:
        save_file(state, str(temporary), metadata=metadata)
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            loaded = {name: opened.get_tensor(name).contiguous() for name in opened.keys()}
            loaded_metadata = dict(opened.metadata() or {})
        if loaded_metadata != metadata or set(loaded) != set(state) or any(
            loaded[name].dtype != torch.float32
            or not torch.equal(loaded[name], state[name])
            for name in state
        ):
            raise PairV5PreferenceTrainingError("Action-LoRA safetensors roundtrip differs")
        runtime.durable_file_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "file_sha256": runtime.file_sha256(path),
        "tensor_count": len(state),
        "metadata": metadata,
        "roundtrip_exact": True,
    }


def _publish_create_only(stage: Path, output: Path) -> None:
    expected = {"adapter.safetensors", "optimizer.pt", "history.json", "receipt.json"}
    staged = {path.name: path for path in stage.iterdir()}
    if set(staged) != expected:
        raise PairV5PreferenceTrainingError("staged output closure differs")
    output.mkdir(mode=0o750)
    try:
        for name in sorted(expected - {"receipt.json"}):
            os.link(stage / name, output / name)
        os.link(stage / "receipt.json", output / "receipt.json")
        runtime.fsync_directory(output)
        runtime.fsync_directory(output.parent)
    except Exception:
        # A visible final directory without receipt.json is the recovery marker.
        raise
    for name in expected:
        (stage / name).unlink()
    stage.rmdir()
    runtime.fsync_directory(output.parent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preference-manifest", required=True)
    parser.add_argument("--expected-preference-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, choices=(1,), default=1)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-frames", type=int, choices=(FRAME_COUNT,), default=FRAME_COUNT)
    parser.add_argument("--frozen-cio-adapter", default=None)
    parser.add_argument("--expected-frozen-cio-adapter-sha256", default=None)
    parser.add_argument("--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--ack-exploratory-no-action-editing-claim", action="store_true")
    return parser


def validate_cli(args: argparse.Namespace) -> Mapping[str, Any]:
    if (WORLD_SIZE, SP_SIZE, DP_SIZE) != (
        runtime.WORLD_SIZE,
        runtime.SP_SIZE,
        runtime.DP_SIZE,
    ) or args.num_frames != FRAME_COUNT:
        raise PairV5PreferenceTrainingError("WORLD8 DP2xSP4 exact81 contract differs")
    if args.max_steps != 1:
        raise PairV5PreferenceTrainingError("v1 permits exactly one optimizer-step canary")
    if args.ack_exploratory_no_action_editing_claim is not True:
        raise PairV5PreferenceTrainingError(
            "explicit no-action-editing-claim acknowledgement is mandatory"
        )
    for name in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        _sha1(getattr(args, name), label=name)
    for name in (
        "expected_preference_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        _sha256(getattr(args, name), label=name)
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise PairV5PreferenceTrainingError("checkpoint tree is outside audited Bernini 1.3B")
    if (args.frozen_cio_adapter in (None, "")) != (
        args.expected_frozen_cio_adapter_sha256 in (None, "")
    ):
        raise PairV5PreferenceTrainingError("CIO adapter path/SHA must be paired")
    if args.expected_frozen_cio_adapter_sha256 not in (None, ""):
        _sha256(args.expected_frozen_cio_adapter_sha256, label="frozen CIO SHA")
    for name in ("learning_rate", "max_grad_norm", "beta"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise PairV5PreferenceTrainingError(f"{name} must be finite and positive")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or not 0 <= args.seed < 2**63:
        raise PairV5PreferenceTrainingError("seed must lie in [0,2^63)")
    return {
        "world_size": WORLD_SIZE,
        "data_parallel_size": DP_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "max_steps": 1,
        "engineering_canary_only": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_contract = validate_cli(args)
    manifest = load_preference_manifest(
        args.preference_manifest,
        expected_sha256=args.expected_preference_manifest_sha256,
    )
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
        raise PairV5PreferenceTrainingError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise PairV5PreferenceTrainingError("pinned Bernini attention-head count differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    import infer_native_identity_generation_canary as native_canary

    distributed = runtime.distributed_contract()
    device = runtime.initialise_distributed(distributed)
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=SP_SIZE)
    )
    output, stage = runtime.prepare_output_transaction(
        args.output, distributed.rank, parallel.world_group
    )

    legacy.seed_same_sample(args.seed)
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

    # Encode the deploy-time source video and four independent RGB refs before
    # moving the renderer to the accelerator.  The two DP arms may use
    # different rows; each SP4 group establishes its own authoritative copy.
    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False)
    source_cache, source_receipts = _source_condition_cache(
        vae,
        manifest.rows,
        device=device,
        parallel=parallel,
        source_audit=native_canary.source_audit,
        vae_encode=_vae_encode,
    )
    del vae
    torch.cuda.empty_cache()

    renderer.to(device)
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise PairV5PreferenceTrainingError("PAIR-v5 requires transformer_1 only")
    disable = getattr(renderer, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    if bool(getattr(transformer, "gradient_checkpointing", False)) or bool(
        getattr(transformer, "is_gradient_checkpointing", False)
    ):
        raise PairV5PreferenceTrainingError("gradient checkpointing remains enabled")

    frozen_cio, cio_receipt, frozen_cio_snapshot = _load_optional_frozen_cio(
        transformer,
        args.frozen_cio_adapter,
        args.expected_frozen_cio_adapter_sha256,
    )
    action_handle = action_adapter.install_pair_v5_action_adapter(transformer)
    renderer.eval()
    trainable = action_handle.trainable_named_parameters()
    if not action_handle.base_parameters_frozen() or any(
        parameter.requires_grad
        for name, parameter in transformer.named_parameters()
        if "action_lora_" not in name
    ):
        raise PairV5PreferenceTrainingError("Action-LoRA trainability closure differs")
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
    text_cache, text_receipts = _frozen_text_embedding_cache(
        renderer,
        tokenizer,
        manifest.rows,
        device=device,
        parallel=parallel,
        build_task_prompt=native_canary.build_task_prompt,
        prompt_cleaner=prompt_clean,
    )
    renderer.t5_text_encoder.to("cpu")
    del tokenizer
    torch.cuda.empty_cache()

    endpoint_cache = {
        row.pair_id: (_load_clean_latent(row.chosen), _load_clean_latent(row.rejected))
        for row in manifest.rows
    }
    for row in manifest.rows:
        chosen, rejected = endpoint_cache[row.pair_id]
        source, refs = source_cache[row.pair_id]
        if (
            chosen.shape != rejected.shape
            or chosen.shape != source.shape
            or any(ref.shape[3:] != chosen.shape[3:] for ref in refs)
            or torch.equal(chosen, rejected)
        ):
            raise PairV5PreferenceTrainingError("preference endpoint geometry differs")

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    history: list[Mapping[str, Any]] = []
    for global_step in range(args.max_steps):
        row_index = (global_step * DP_SIZE + distributed.arm_index) % len(manifest.rows)
        row = manifest.rows[row_index]
        chosen_cpu, rejected_cpu = endpoint_cache[row.pair_id]
        source_cpu, refs_cpu = source_cache[row.pair_id]
        cond_cpu, uncond_cpu = text_cache[row.pair_id]
        chosen = chosen_cpu.to(device=device).contiguous().detach()
        rejected = rejected_cpu.to(device=device).contiguous().detach()
        source = source_cpu.to(device=device).contiguous().detach()
        refs = tuple(value.to(device=device).contiguous().detach() for value in refs_cpu)
        conditional = cond_cpu.to(device=device).contiguous().detach()
        unconditional = uncond_cpu.to(device=device).contiguous().detach()
        for tensor in (chosen, rejected, source, *refs, conditional, unconditional):
            _broadcast_within_sp(tensor, parallel=parallel)

        sigma_index = registered_action_sigma_index(
            seed=args.seed,
            step=global_step,
            pair_digest=row.pair_digest,
            dp_rank=distributed.arm_index,
        )
        sigma = torch.tensor(
            [native.NATIVE_UNIPC40_SIGMAS[sigma_index]],
            dtype=torch.float32,
            device=device,
        ).detach()
        timestep = torch.tensor(
            [native.NATIVE_UNIPC40_TIMESTEPS[sigma_index]],
            dtype=torch.float32,
            device=device,
        ).detach()
        noise_seed = _fresh_noise_seed(
            args.seed, global_step, row.pair_digest, distributed.arm_index
        )
        epsilon = fresh_shared_epsilon(chosen.shape, seed=noise_seed, device=device)
        _broadcast_within_sp(epsilon, parallel=parallel)
        sigma_view = sigma.reshape(1, 1, 1, 1, 1)
        chosen_x = ((1.0 - sigma_view) * chosen + sigma_view * epsilon).detach()
        rejected_x = ((1.0 - sigma_view) * rejected + sigma_view * epsilon).detach()

        detached_student: dict[str, Any] = {}
        detached_reference: dict[str, Any] = {}
        for name, state in (("chosen", chosen_x), ("rejected", rejected_x)):
            pack = _build_pack(transformer, source, refs, state)
            detached_student[name] = _guided_prediction_no_grad(
                diffusion,
                pack,
                timestep=timestep,
                cond_embeds=conditional,
                uncond_embeds=unconditional,
                action_handle=action_handle,
                cio_handle=frozen_cio,
                sp_rank=distributed.sp_rank,
                sigma_index=sigma_index,
                action_enabled=True,
                video_shape=state.shape,
            )
            detached_reference[name] = _guided_prediction_no_grad(
                diffusion,
                pack,
                timestep=timestep,
                cond_embeds=conditional,
                uncond_embeds=unconditional,
                action_handle=action_handle,
                cio_handle=frozen_cio,
                sp_rank=distributed.sp_rank,
                sigma_index=sigma_index,
                action_enabled=False,
                video_shape=state.shape,
            ).detach()
            del pack
        student_chosen = detached_student["chosen"].detach().requires_grad_(True)
        student_rejected = detached_student["rejected"].detach().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        result = flow_dpo.reference_corrected_flow_dpo(
            chosen,
            rejected,
            epsilon,
            sigma,
            student_chosen,
            student_rejected,
            detached_reference["chosen"],
            detached_reference["rejected"],
            beta=args.beta,
            sample_weight=torch.tensor(
                [row.sample_weight], dtype=torch.float32, device=device
            ).detach(),
        )
        if not torch.equal(result.chosen_x_sigma, chosen_x) or not torch.equal(
            result.rejected_x_sigma, rejected_x
        ):
            raise PairV5PreferenceTrainingError(
                "flow-DPO did not reconstruct the shared candidate states"
            )
        if not runtime.world_all_true(
            bool(torch.isfinite(result.loss.detach()).item()), group=parallel.world_group
        ):
            raise PairV5PreferenceTrainingError("non-finite DPO loss blocked update")
        result.loss.backward()
        if student_chosen.grad is None or student_rejected.grad is None:
            raise PairV5PreferenceTrainingError("DPO leaves have no output cotangent")
        replay_max = 0.0
        for state, cotangent, expected in (
            (chosen_x, student_chosen.grad.detach(), detached_student["chosen"]),
            (rejected_x, student_rejected.grad.detach(), detached_student["rejected"]),
        ):
            replay_max = max(
                replay_max,
                _replay_prediction_vjp(
                    diffusion,
                    transformer,
                    source_video=source,
                    references=refs,
                    x_sigma=state,
                    timestep=timestep,
                    cond_embeds=conditional,
                    uncond_embeds=unconditional,
                    action_handle=action_handle,
                    cio_handle=frozen_cio,
                    sp_rank=distributed.sp_rank,
                    sigma_index=sigma_index,
                    output_cotangent=cotangent,
                    expected_guided=expected,
                ),
            )
        preclip_norm = runtime.synchronize_gradients(trainable, parallel)
        clipped = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], args.max_grad_norm
        )
        if not math.isfinite(float(clipped)):
            raise PairV5PreferenceTrainingError("gradient clipping is non-finite")
        optimizer.step()
        parameter_digest = runtime.parameter_consensus(
            trainable,
            parallel.world_group,
            f"PAIR-v5 action adapter step {global_step + 1}",
        )
        local_record = {
            "step": global_step + 1,
            "dp_rank": distributed.arm_index,
            "sp_rank": distributed.sp_rank,
            "row_index": row_index,
            "pair_id": row.pair_id,
            "pair_digest": row.pair_digest,
            "chosen_candidate_id": row.chosen.candidate_id,
            "rejected_candidate_id": row.rejected.candidate_id,
            "sigma_schedule_index": sigma_index,
            "sigma_gate": action_adapter.sigma_gate(sigma_index)[0],
            "noise_seed": noise_seed,
            "fresh_epsilon_sha256": runtime.tensor_sha256(epsilon),
            "recorded_rollout_noise_loaded": False,
            "loss": float(result.loss.detach().item()),
            "advantage": float(result.advantage.detach().item()),
            "student_gap": float(result.student_gap.detach().item()),
            "reference_gap": float(result.reference_gap.detach().item()),
            "preclip_gradient_norm_world_average": preclip_norm,
            "vjp_replay_max_abs": replay_max,
            "parameter_digest": parameter_digest,
        }
        sp_projection = {key: value for key, value in local_record.items() if key != "sp_rank"}
        runtime.digest_consensus(
            object_sha256(sp_projection),
            group=parallel.sp_group,
            expected_count=SP_SIZE,
            label="PAIR-v5 SP step record",
        )
        gathered: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local_record, group=parallel.world_group)
        history.append({"step": global_step + 1, "dp_records": [gathered[0], gathered[4]]})
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "step": global_step + 1,
                        "loss_dp0": gathered[0]["loss"],
                        "loss_dp1": gathered[4]["loss"],
                        "preclip_gradient_norm": preclip_norm,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del chosen, rejected, source, refs, conditional, unconditional, epsilon
        torch.cuda.empty_cache()

    final_digest = runtime.parameter_consensus(
        trainable, parallel.world_group, "PAIR-v5 final action adapter"
    )
    if final_digest == initial_digest:
        raise PairV5PreferenceTrainingError("optimizer did not change Action-LoRA")
    if frozen_cio_snapshot is not None:
        frozen_cio_snapshot.assert_unchanged(label="frozen CIO adapter after training")
    cio_receipt = {
        **dict(cio_receipt),
        "post_training_file_mutation_audit_passed": True,
    }
    manifest.assert_unchanged()
    dist.barrier(group=parallel.world_group)
    if distributed.rank == 0:
        adapter_path = stage / "adapter.safetensors"
        optimizer_path = stage / "optimizer.pt"
        history_path = stage / "history.json"
        adapter_roundtrip = _save_action_adapter(adapter_path, action_handle)
        runtime.atomic_torch_save(
            optimizer_path,
            {
                "schema_version": RUN_RECEIPT_SCHEMA,
                "optimizer": optimizer.state_dict(),
                "global_step": args.max_steps,
                "adapter_parameter_digest": final_digest,
            },
        )
        runtime.atomic_json(
            history_path,
            {
                "schema_version": HISTORY_SCHEMA,
                "step_count": args.max_steps,
                "records": history,
            },
        )
        receipt: dict[str, Any] = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "complete": True,
            "optimizer_steps": args.max_steps,
            "run_contract": dict(run_contract),
            "manifest": {
                "id": manifest.manifest_id,
                "file": dict(manifest.snapshot.receipt()),
                "manifest_digest": manifest.manifest_digest,
                "calibration_file": dict(manifest.calibration_snapshot.receipt()),
                "calibration_receipt_digest": manifest.calibration_receipt[
                    "receipt_digest"
                ],
                "calibration_optimizer_authorized": True,
                "calibration_optimizer_provenance_digest": (
                    manifest.calibration_optimizer_provenance["provenance_digest"]
                ),
                "evaluator_registry": dict(manifest.evaluator_registry),
                "evaluator_packet_digest_by_candidate": {
                    candidate_id: packet["packet_digest"]
                    for candidate_id, packet in manifest.evaluator_packets.items()
                },
                "all_selector_candidates_bound_to_rollout_and_evaluator_packet": True,
                "selector_file": dict(manifest.selector_snapshot.receipt()),
                "selector_receipt_digest": manifest.selector_receipt[
                    "receipt_digest"
                ],
                "selector_decision": manifest.selector_receipt["decision"],
                "selector_replayed_exactly": True,
                "rows": [
                    {
                        "pair_id": row.pair_id,
                        "pair_digest": row.pair_digest,
                        "source_video": dict(row.source_video_snapshot.receipt()),
                        "chosen_candidate_id": row.chosen.candidate_id,
                        "rejected_candidate_id": row.rejected.candidate_id,
                        "chosen_clean_latent": dict(
                            row.chosen.clean_latent_snapshot.receipt()
                        ),
                        "rejected_clean_latent": dict(
                            row.rejected.clean_latent_snapshot.receipt()
                        ),
                        "chosen_recorded_noise": dict(
                            row.chosen.recorded_noise_snapshot.receipt()
                        ),
                        "rejected_recorded_noise": dict(
                            row.rejected.recorded_noise_snapshot.receipt()
                        ),
                    }
                    for row in manifest.rows
                ],
                "post_training_input_mutation_audit_passed": True,
            },
            "model_input_closure": dict(_INPUT_CLOSURE),
            "native_rv2v4": {
                "bridge_contract": dict(native_bridge.bridge_contract_receipt()),
                "source_condition_receipts_by_pair": dict(source_receipts),
                "text_condition_receipts_by_pair": dict(text_receipts),
                "reference_indices": list(REFERENCE_INDICES),
                "references_independently_encoded_from_rgb": True,
                "exact81": True,
                "exact40_schedule": dict(native.native_unipc40_schedule_receipt()),
                "training_sigma_indices": list(ACTION_SIGMA_INDICES),
                "low_sigma_indices_exact_base": list(action_adapter.LOW_SIGMA_INDICES),
                "same_fresh_epsilon_and_sigma_for_chosen_rejected": True,
                "recorded_rollout_noise_loaded": False,
                "serial_exact_linear_vjp": True,
                "one_transformer_graph_resident_at_a_time": True,
            },
            "objective": {
                "flow_dpo_contract": dict(flow_dpo.contract_receipt()),
                "beta": args.beta,
                "reference_policy": "same_frozen_bernini_plus_optional_cio_action_lora_disabled",
            },
            "adapter": {
                **dict(action_handle.receipt()),
                "initial_parameter_digest": initial_digest,
                "final_parameter_digest": final_digest,
                "changed_by_optimizer": True,
                "safetensors_roundtrip": dict(adapter_roundtrip),
                "frozen_cio": dict(cio_receipt),
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
            "history_summary": history,
            "runtime": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
            },
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                "single_expert": "transformer_1",
            },
            "artifacts": {
                "adapter.safetensors": runtime.file_sha256(adapter_path),
                "optimizer.pt": runtime.file_sha256(optimizer_path),
                "history.json": runtime.file_sha256(history_path),
            },
            "engineering_canary_only": True,
            "semantic_action_editing_success": False,
            "video_quality_claim_authorized": False,
            "scientific_generalization_claim_authorized": False,
            "long_training_automatically_submitted": False,
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
        }
        receipt["receipt_digest"] = runtime.object_sha256(receipt)
        runtime.atomic_json(stage / "receipt.json", receipt)
        runtime.verify_staged_run_bundle(stage, receipt)
        runtime.fsync_directory(stage)
        _publish_create_only(stage, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "optimizer_steps": args.max_steps,
                    "adapter_parameter_digest": final_digest,
                    "semantic_action_editing_success": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=parallel.world_group)
    if not output.is_dir() or output.is_symlink() or stage.exists():
        raise PairV5PreferenceTrainingError("atomic output publication did not complete")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_SIGMA_INDICES",
    "FILE_BINDING_SCHEMA",
    "MANIFEST_SCHEMA",
    "PAIR_ROW_SCHEMA",
    "ROLLOUT_BINDING_SCHEMA",
    "PairV5PreferenceTrainingError",
    "PreferenceManifest",
    "PreferenceRow",
    "RolloutEndpoint",
    "build_parser",
    "canonical_json_bytes",
    "fresh_shared_epsilon",
    "load_preference_manifest",
    "object_sha256",
    "registered_action_sigma_index",
    "validate_cli",
]
