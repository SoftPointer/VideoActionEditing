#!/usr/bin/env python3
"""Train PAIR-v5 Action-LoRA from same-state pure-T2V action guidance.

This executable is a fallback for the case where no safe native-RV2V
preference pair exists.  It consumes a sealed bank of independently
EVENT-QUALIFIED exact81 pure-T2V latents and each latent's own official sampler
Gaussian.  At every exact40 coordinate it queries frozen Bernini under the
action plus all nine hard negatives on one patched ``y_sigma`` state, then
serially replays the output cotangents through Action-LoRA.

The topology is one WORLD8 node arranged as DP2 x Ulysses-SP4.  The frozen
Bernini base is shared by teacher and student; only ``attn2`` Q/O Action-LoRA
blocks 0..22 are trainable.  Exact40 indices 38/39 perform no model callback,
backward, or optimizer step.  No RV2V/source/target/donor/mask/flow/pose/track
input exists in the manifest or model callback.

A completed run is only evidence that the sealed self-guidance objective ran;
it is not evidence that action editing succeeds on held-out source videos.
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
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dclr_runtime_contract as t2v_runtime  # noqa: E402
import infer_source_kv_carrier_oracle as checkpoint_audit  # noqa: E402
import pair_v5_action_adapter as action_adapter  # noqa: E402
import pair_v5_native_bridge as native_bridge  # noqa: E402
import pair_v5_t2v_guidance_distill as guidance  # noqa: E402
import score_pair_v5_t2v_energy_bank_v3 as energy_scorer  # noqa: E402
import source_self_runtime as distributed_runtime  # noqa: E402
import train_lora as legacy  # noqa: E402


METHOD_NAME = "bernini-pair-v5-same-state-t2v-action-guidance-distill"
MANIFEST_SCHEMA = "bernini-pair-v5-t2v-guidance-manifest-v1"
EVENT_SCHEMA = "bernini-pair-v5-t2v-guidance-event-v1"
RUN_RECEIPT_SCHEMA = "bernini-pair-v5-t2v-guidance-run-receipt-v1"
HISTORY_SCHEMA = "bernini-pair-v5-t2v-guidance-history-v1"
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
MIN_EVENTS = 2
MAX_EVENTS = 16
DEFAULT_LEARNING_RATE = 1.0e-6
DEFAULT_MAX_GRAD_NORM = 1.0
VJP_RTOL = 2.0e-5
VJP_ATOL = 2.0e-5
CAGD_AUTHORIZATION_SCHEMA = "bernini-pair-v5-cagd-recomputed-authorization-v3"

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

_INPUT_CLOSURE = {
    "accepted": [
        "event_qualified_pure_t2v_normalized_clean_latent",
        "same_event_official_native_sampler_gaussian",
        "closed_action_plus_nine_hard_negative_prompt_bank",
        "event_qualification_receipt",
        "passed_action_calibration_receipt",
    ],
    "pure_t2v_visual_role": "same_coordinate_frozen_field_query_only",
    "optimizer_analysis_split": "fit_only",
    "confirmation_event_consumed": False,
    "rv2v_source_video": False,
    "rv2v_reference_frames": False,
    "rv2v_target_or_pseudo_target": False,
    "proposal_rgb": False,
    "cross_video_latent_or_residual": False,
    "motion_donor": False,
    "mask": False,
    "flow": False,
    "pose": False,
    "track": False,
    "trajectory": False,
}

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "optimizer_authorized",
        "checkpoint_tree_sha256",
        "event_count",
        "events",
        "input_closure",
        "manifest_digest",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "action_family",
        "analysis_split",
        "prompt_by_branch",
        "prompt_bank_sha256",
        "clean_latent_path",
        "clean_latent_file_sha256",
        "clean_latent_tensor_key",
        "official_gaussian_path",
        "official_gaussian_file_sha256",
        "official_gaussian_tensor_key",
        "eligibility_receipt_path",
        "eligibility_receipt_file_sha256",
        "event_digest",
    }
)
_ELIGIBILITY_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "action_family",
        "analysis_split",
        "frame_count",
        "latent_shape",
        "event_qualified",
        "calibration_confirmation_passed",
        "calibration_optimizer_authorized",
        "clean_t2v_latent_tensor_sha256",
        "official_gaussian_tensor_sha256",
        "official_gaussian_artifact_sha256",
        "checkpoint_tree_sha256",
        "prompt_bank_sha256",
        "action_adapter_schema_sha256",
        "event_qualification_receipt_digest",
        "calibration_receipt_digest",
        "pure_t2v_positive_role",
        "rv2v_target_input_noise_donor",
        "optimizer_authorized",
        "receipt_digest",
    }
)
_SCORER_GROUP_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "frozen_checkpoint_receipt_digest",
        "checkpoint_content_binding",
        "schedule_coordinate",
        "candidate_count",
        "candidate_receipt_digests",
        "primary_score_field",
        "phase_conjunctive_role",
        "input_closure",
        "training_performed",
        "optimizer_authorized",
        "scientific_action_editing_claim",
        "method_source_revision",
        "method_source_archive_sha256",
        "bernini_revision",
        "veomni_revision",
        "receipt_digest",
    }
)


class PairV5T2VGuidanceTrainingError(RuntimeError):
    """Raised before an ambiguous training input or update can be used."""


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
        raise PairV5T2VGuidanceTrainingError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1_RE if length == 40 else _SHA256_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise PairV5T2VGuidanceTrainingError(
            f"{label} must be lowercase SHA-{'1' if length == 40 else '256'}"
        )
    return value


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise PairV5T2VGuidanceTrainingError(
            f"{label} keys differ: missing={sorted(set(fields)-actual)} extra={sorted(actual-set(fields))}"
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PairV5T2VGuidanceTrainingError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _plain_absolute_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or "\x00" in value:
        raise PairV5T2VGuidanceTrainingError(f"{label} must be path text")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV5T2VGuidanceTrainingError(f"{label} must be an absolute plain file")
    return path


def _read_bound(path: Path, expected_sha256: str, *, label: str) -> bytes:
    expected = _sha(expected_sha256, length=64, label=f"{label} SHA-256")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or hashlib.sha256(raw).hexdigest() != expected
    ):
        raise PairV5T2VGuidanceTrainingError(f"{label} changed or hash differs")
    return raw


def _strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise PairV5T2VGuidanceTrainingError(f"{label} contains {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PairV5T2VGuidanceTrainingError(f"{label} duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5T2VGuidanceTrainingError(f"{label} is not strict ASCII JSON") from error
    if not isinstance(value, Mapping):
        raise PairV5T2VGuidanceTrainingError(f"{label} root must be an object")
    return value


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str

    def assert_unchanged(self) -> None:
        if not self.path.is_file() or self.path.is_symlink() or _file_sha256(self.path) != self.sha256:
            raise PairV5T2VGuidanceTrainingError(f"bound file changed: {self.path}")


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    action_family: str
    analysis_split: str
    prompt_by_branch: Mapping[str, str]
    prompt_bank_sha256: str
    clean_latent: FileBinding
    clean_latent_tensor_key: str
    official_gaussian: FileBinding
    official_gaussian_tensor_key: str
    eligibility_file: FileBinding
    eligibility: guidance.GuidanceEligibility
    event_digest: str

    def assert_unchanged(self) -> None:
        self.clean_latent.assert_unchanged()
        self.official_gaussian.assert_unchanged()
        self.eligibility_file.assert_unchanged()


@dataclass(frozen=True)
class EventRuntime:
    spec: EventSpec
    event_latent_cpu: Any
    official_epsilon_cpu: Any


@dataclass(frozen=True)
class CAGDAuthorization:
    evidence_file: FileBinding
    evidence_digest: str
    authorization_digest: str
    calibration_receipt_digest: str
    checkpoint_content_receipt_digest: str
    scorer_group_files: tuple[FileBinding, ...]
    scorer_group_receipt_digests: tuple[str, ...]

    def assert_unchanged(self) -> None:
        self.evidence_file.assert_unchanged()
        for binding in self.scorer_group_files:
            binding.assert_unchanged()


@dataclass(frozen=True)
class GuidanceManifest:
    path: Path
    raw_sha256: str
    checkpoint_tree_sha256: str
    events: tuple[EventSpec, ...]
    manifest_digest: str

    def assert_unchanged(self) -> None:
        if _file_sha256(self.path) != self.raw_sha256:
            raise PairV5T2VGuidanceTrainingError("guidance manifest changed during training")
        for event in self.events:
            event.assert_unchanged()


def _load_eligibility(path: Path, expected_sha256: str) -> guidance.GuidanceEligibility:
    raw = _read_bound(path, expected_sha256, label="eligibility receipt")
    value = _closed(_strict_json(raw, label="eligibility receipt"), _ELIGIBILITY_FIELDS, label="eligibility receipt")
    unsigned = dict(value)
    declared = _sha(unsigned.pop("receipt_digest"), length=64, label="eligibility receipt digest")
    if object_sha256(unsigned) != declared:
        raise PairV5T2VGuidanceTrainingError("eligibility embedded digest differs")
    if (
        value["schema_version"] != guidance.ELIGIBILITY_SCHEMA
        or value["frame_count"] != guidance.FRAME_COUNT
        or value["pure_t2v_positive_role"] != "same_coordinate_frozen_field_query_only"
        or value["rv2v_target_input_noise_donor"] is not False
    ):
        raise PairV5T2VGuidanceTrainingError("eligibility information-flow contract differs")
    shape = value["latent_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 5
        or any(type(item) is not int for item in shape)
    ):
        raise PairV5T2VGuidanceTrainingError("eligibility latent shape differs")
    return guidance.GuidanceEligibility(
        sample_id=value["sample_id"],
        action_family=value["action_family"],
        analysis_split=value["analysis_split"],
        latent_shape=tuple(shape),
        event_qualified=value["event_qualified"],
        calibration_confirmation_passed=value["calibration_confirmation_passed"],
        calibration_optimizer_authorized=value["calibration_optimizer_authorized"],
        clean_t2v_latent_tensor_sha256=value["clean_t2v_latent_tensor_sha256"],
        official_gaussian_tensor_sha256=value["official_gaussian_tensor_sha256"],
        official_gaussian_artifact_sha256=value["official_gaussian_artifact_sha256"],
        checkpoint_tree_sha256=value["checkpoint_tree_sha256"],
        prompt_bank_sha256=value["prompt_bank_sha256"],
        action_adapter_schema_sha256=value["action_adapter_schema_sha256"],
        event_qualification_receipt_digest=value["event_qualification_receipt_digest"],
        calibration_receipt_digest=value["calibration_receipt_digest"],
        optimizer_authorized=value["optimizer_authorized"],
        receipt_digest=declared,
    )


def load_manifest(path_value: str, expected_sha256: str) -> GuidanceManifest:
    path = _plain_absolute_file(path_value, label="guidance manifest")
    raw = _read_bound(path, expected_sha256, label="guidance manifest")
    root = _closed(_strict_json(raw, label="guidance manifest"), _ROOT_FIELDS, label="guidance manifest")
    unsigned = dict(root)
    declared = _sha(unsigned.pop("manifest_digest"), length=64, label="manifest digest")
    if object_sha256(unsigned) != declared:
        raise PairV5T2VGuidanceTrainingError("manifest embedded digest differs")
    if root["schema_version"] != MANIFEST_SCHEMA or root["optimizer_authorized"] is not True:
        raise PairV5T2VGuidanceTrainingError("manifest does not authorize optimization")
    if root["input_closure"] != _INPUT_CLOSURE:
        raise PairV5T2VGuidanceTrainingError("manifest input closure differs")
    checkpoint_digest = _sha(root["checkpoint_tree_sha256"], length=64, label="checkpoint tree SHA-256")
    rows = root["events"]
    if (
        not isinstance(rows, list)
        or not MIN_EVENTS <= len(rows) <= MAX_EVENTS
        or root["event_count"] != len(rows)
    ):
        raise PairV5T2VGuidanceTrainingError(f"manifest requires {MIN_EVENTS}..{MAX_EVENTS} events")
    events: list[EventSpec] = []
    seen: set[str] = set()
    for ordinal, raw_event in enumerate(rows):
        event = _closed(raw_event, _EVENT_FIELDS, label=f"event[{ordinal}]")
        unsigned_event = dict(event)
        event_digest = _sha(unsigned_event.pop("event_digest"), length=64, label="event digest")
        if object_sha256(unsigned_event) != event_digest:
            raise PairV5T2VGuidanceTrainingError(f"event[{ordinal}] digest differs")
        event_id = event["event_id"]
        action_family = event["action_family"]
        analysis_split = event["analysis_split"]
        prompts = guidance.validate_prompt_bank(event["prompt_by_branch"])
        prompt_digest = _sha(
            event["prompt_bank_sha256"],
            length=64,
            label=f"event[{ordinal}] prompt bank SHA-256",
        )
        if guidance.prompt_bank_sha256(prompts) != prompt_digest:
            raise PairV5T2VGuidanceTrainingError(
                f"event[{ordinal}] prompt bank digest differs"
            )
        if (
            event["schema_version"] != EVENT_SCHEMA
            or not isinstance(event_id, str)
            or _SAFE_ID_RE.fullmatch(event_id) is None
            or not isinstance(action_family, str)
            or _SAFE_ID_RE.fullmatch(action_family) is None
            or analysis_split != "fit"
            or event_id in seen
        ):
            raise PairV5T2VGuidanceTrainingError(
                f"event[{ordinal}] identity/split differs; optimizer events must be fit"
            )
        seen.add(event_id)
        clean_path = _plain_absolute_file(event["clean_latent_path"], label="clean latent")
        noise_path = _plain_absolute_file(event["official_gaussian_path"], label="official Gaussian")
        eligibility_path = _plain_absolute_file(event["eligibility_receipt_path"], label="eligibility receipt")
        clean_sha = _sha(event["clean_latent_file_sha256"], length=64, label="clean latent file SHA-256")
        noise_sha = _sha(event["official_gaussian_file_sha256"], length=64, label="Gaussian file SHA-256")
        eligibility_sha = _sha(event["eligibility_receipt_file_sha256"], length=64, label="eligibility file SHA-256")
        for bound_path, bound_sha, label in (
            (clean_path, clean_sha, "clean latent"),
            (noise_path, noise_sha, "official Gaussian"),
            (eligibility_path, eligibility_sha, "eligibility receipt"),
        ):
            if _file_sha256(bound_path) != bound_sha:
                raise PairV5T2VGuidanceTrainingError(f"{label} file SHA-256 differs")
        eligibility = _load_eligibility(eligibility_path, eligibility_sha)
        if (
            eligibility.sample_id != event_id
            or eligibility.action_family != action_family
            or eligibility.analysis_split != analysis_split
            or eligibility.official_gaussian_artifact_sha256 != noise_sha
            or eligibility.checkpoint_tree_sha256 != checkpoint_digest
            or eligibility.prompt_bank_sha256 != prompt_digest
            or eligibility.optimizer_authorized is not True
        ):
            raise PairV5T2VGuidanceTrainingError(f"event[{ordinal}] eligibility binding differs")
        for key_name in ("clean_latent_tensor_key", "official_gaussian_tensor_key"):
            key = event[key_name]
            if not isinstance(key, str) or not key or "\x00" in key:
                raise PairV5T2VGuidanceTrainingError(f"event[{ordinal}] {key_name} differs")
        events.append(
            EventSpec(
                event_id=event_id,
                action_family=action_family,
                analysis_split=analysis_split,
                prompt_by_branch=prompts,
                prompt_bank_sha256=prompt_digest,
                clean_latent=FileBinding(clean_path, clean_sha),
                clean_latent_tensor_key=event["clean_latent_tensor_key"],
                official_gaussian=FileBinding(noise_path, noise_sha),
                official_gaussian_tensor_key=event["official_gaussian_tensor_key"],
                eligibility_file=FileBinding(eligibility_path, eligibility_sha),
                eligibility=eligibility,
                event_digest=event_digest,
            )
        )
    # The runtime advances two events per exact40 cycle.  Simulate a complete
    # modular orbit here so the invariant applies to *every* future DP0/DP1
    # pair, not merely cycle zero (the previous check silently failed for
    # manifests with more than two events).
    for cycle in range(len(events)):
        left = events[(cycle * DP_SIZE) % len(events)]
        right = events[(cycle * DP_SIZE + 1) % len(events)]
        if left.action_family == right.action_family:
            raise PairV5T2VGuidanceTrainingError(
                f"cycle{cycle} DP0/DP1 fit events must cover distinct action families"
            )
        if left.prompt_bank_sha256 == right.prompt_bank_sha256:
            raise PairV5T2VGuidanceTrainingError(
                f"cycle{cycle} distinct action families cannot alias one prompt bank"
            )
    return GuidanceManifest(
        path=path,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoint_tree_sha256=checkpoint_digest,
        events=tuple(events),
        manifest_digest=declared,
    )


def load_event_tensors(manifest: GuidanceManifest) -> tuple[EventRuntime, ...]:
    """Load and authenticate all tensors before model/optimizer construction."""

    import torch
    from safetensors import safe_open

    result: list[EventRuntime] = []
    for event in manifest.events:
        with safe_open(str(event.clean_latent.path), framework="pt", device="cpu") as opened:
            if event.clean_latent_tensor_key not in opened.keys():
                raise PairV5T2VGuidanceTrainingError("clean latent tensor key is absent")
            clean = opened.get_tensor(event.clean_latent_tensor_key).float().contiguous()
        with safe_open(str(event.official_gaussian.path), framework="pt", device="cpu") as opened:
            if event.official_gaussian_tensor_key not in opened.keys():
                raise PairV5T2VGuidanceTrainingError("official Gaussian tensor key is absent")
            epsilon = opened.get_tensor(event.official_gaussian_tensor_key).float().contiguous()
        guidance._exact81(clean, label="manifest pure-T2V event latent")
        guidance._exact81(epsilon, label="manifest official Gaussian")
        event.eligibility.validate(
            event_latent=clean,
            official_epsilon=epsilon,
            prompt_by_branch=event.prompt_by_branch,
            checkpoint_tree_sha256=manifest.checkpoint_tree_sha256,
        )
        result.append(EventRuntime(event, clean, epsilon))
    return tuple(result)


def _load_scorer_group_receipt(
    path_value: str,
    expected_sha256: str,
    *,
    ordinal: int,
) -> tuple[Mapping[str, Any], FileBinding]:
    path = _plain_absolute_file(path_value, label=f"scorer group receipt[{ordinal}]")
    raw = _read_bound(
        path,
        expected_sha256,
        label=f"scorer group receipt[{ordinal}]",
    )
    row = dict(
        _closed(
            _strict_json(raw, label=f"scorer group receipt[{ordinal}]"),
            _SCORER_GROUP_RECEIPT_FIELDS,
            label=f"scorer group receipt[{ordinal}]",
        )
    )
    declared = _sha(
        row["receipt_digest"],
        length=64,
        label=f"scorer group receipt[{ordinal}] digest",
    )
    unsigned = dict(row)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        raise PairV5T2VGuidanceTrainingError(
            f"scorer group receipt[{ordinal}] embedded digest differs"
        )
    return row, FileBinding(path, _sha(expected_sha256, length=64, label="group receipt SHA-256"))


def _load_cagd_authorization(
    args: argparse.Namespace,
    manifest: GuidanceManifest,
) -> CAGDAuthorization:
    """Recompute all v3 evidence and bind it to the actual runtime checkpoint.

    A legacy eligibility JSON never reaches this function as an authority.  It
    is accepted only after the independent validator has reconstructed it from
    the rendered-bank generation receipt, raw frozen-MACE score, detached event
    audit, fit calibration, and held-out confirmation calibration.
    """

    try:
        import validate_pair_v5_cagd_evidence_v3 as evidence_validator

        authorization = evidence_validator.validate_evidence(
            args.cagd_validator_evidence,
            expected_evidence_sha256=args.expected_cagd_validator_evidence_sha256,
            checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        )
    except Exception as error:
        raise PairV5T2VGuidanceTrainingError(
            f"CAGD v3 evidence recomputation failed: {error}"
        ) from error
    if (
        authorization.get("schema_version") != CAGD_AUTHORIZATION_SCHEMA
        or authorization.get("optimizer_authorized") is not True
        or authorization.get("legacy_eligibility_self_declaration_trusted") is not False
        or authorization.get("all_source_files_and_receipts_revalidated") is not True
        or authorization.get("calibration_recomputed_from_raw_global_scores") is not True
        or authorization.get("confirmation_event_count_for_optimizer") != 0
        or authorization.get("guidance_manifest_file_sha256") != manifest.raw_sha256
    ):
        raise PairV5T2VGuidanceTrainingError(
            "recomputed CAGD authorization does not authorize this exact fit manifest"
        )
    authorization_digest = _sha(
        authorization.get("authorization_digest"),
        length=64,
        label="CAGD authorization digest",
    )
    unsigned_authorization = dict(authorization)
    unsigned_authorization.pop("authorization_digest")
    if object_sha256(unsigned_authorization) != authorization_digest:
        raise PairV5T2VGuidanceTrainingError("CAGD authorization digest differs")

    evidence_path = _plain_absolute_file(
        args.cagd_validator_evidence, label="CAGD validator evidence"
    )
    evidence_sha = _sha(
        args.expected_cagd_validator_evidence_sha256,
        length=64,
        label="CAGD validator evidence SHA-256",
    )
    evidence_raw = _read_bound(
        evidence_path, evidence_sha, label="CAGD validator evidence"
    )
    evidence = _strict_json(evidence_raw, label="CAGD validator evidence")
    if evidence.get("evidence_digest") != authorization.get("evidence_digest"):
        raise PairV5T2VGuidanceTrainingError("CAGD evidence/authorization digest differs")

    score_bindings = evidence.get("score_receipt_files")
    if not isinstance(score_bindings, list) or not score_bindings:
        raise PairV5T2VGuidanceTrainingError("CAGD score receipt bindings are absent")
    score_receipts: dict[str, Mapping[str, Any]] = {}
    score_receipt_digests: set[str] = set()
    for ordinal, binding in enumerate(score_bindings):
        if not isinstance(binding, Mapping):
            raise PairV5T2VGuidanceTrainingError("CAGD score binding differs")
        candidate_id = binding.get("candidate_id")
        score_path = _plain_absolute_file(
            binding.get("path"), label=f"CAGD score receipt[{ordinal}]"
        )
        score_file_sha = _sha(
            binding.get("file_sha256"),
            length=64,
            label=f"CAGD score receipt[{ordinal}] SHA-256",
        )
        score_raw = _strict_json(
            _read_bound(
                score_path,
                score_file_sha,
                label=f"CAGD score receipt[{ordinal}]",
            ),
            label=f"CAGD score receipt[{ordinal}]",
        )
        try:
            score = energy_scorer.validate_score_receipt(score_raw)
        except energy_scorer.PairV5T2VEnergyScoringError as error:
            raise PairV5T2VGuidanceTrainingError(str(error)) from error
        if (
            candidate_id != score["candidate_id"]
            or binding.get("receipt_digest") != score["receipt_digest"]
            or binding.get("raw_global_action_energy_score")
            != score["raw_global_action_energy_score"]
            or candidate_id in score_receipts
        ):
            raise PairV5T2VGuidanceTrainingError("CAGD score binding identity differs")
        score_receipts[candidate_id] = score
        score_receipt_digests.add(score["receipt_digest"])

    paths = args.scorer_group_receipt
    hashes = args.expected_scorer_group_receipt_sha256
    if len(paths) != 2 or len(hashes) != 2:
        raise PairV5T2VGuidanceTrainingError("exactly two SP4 scorer group receipts are required")
    group_rows: list[Mapping[str, Any]] = []
    group_files: list[FileBinding] = []
    for ordinal, (path_value, digest) in enumerate(zip(paths, hashes)):
        row, binding = _load_scorer_group_receipt(path_value, digest, ordinal=ordinal)
        group_rows.append(row)
        group_files.append(binding)
    if {row["group_id"] for row in group_rows} != {"sp4-a", "sp4-b"}:
        raise PairV5T2VGuidanceTrainingError("scorer group receipt closure differs")

    source_spec_sha = authorization.get("source_bank_spec_sha256")
    source_bank_digest = authorization.get("source_bank_receipt_digest")
    checkpoint_receipt_digests: set[str] = set()
    covered_score_digests: list[str] = []
    for row in group_rows:
        candidates = row["candidate_receipt_digests"]
        if not isinstance(candidates, list) or row["candidate_count"] != len(candidates):
            raise PairV5T2VGuidanceTrainingError("scorer group candidate count differs")
        if (
            row["schema_version"] != energy_scorer.GROUP_RECEIPT_SCHEMA
            or row["root_spec_raw_sha256"] != source_spec_sha
            or row["bank_receipt_digest"] != source_bank_digest
            or row["schedule_coordinate"] != energy_scorer.schedule_coordinate_receipt()
            or row["primary_score_field"] != "raw_global_action_energy_score"
            or row["phase_conjunctive_role"]
            != "diagnostic_only_never_calibration_gate"
            or row["input_closure"] != energy_scorer.SCORE_INPUT_CLOSURE
            or row["training_performed"] is not False
            or row["optimizer_authorized"] is not False
            or row["scientific_action_editing_claim"] is not False
            or row["method_source_revision"] != args.method_source_revision
            or row["method_source_archive_sha256"]
            != args.method_source_archive_sha256
            or row["bernini_revision"] != args.expected_bernini_commit
            or row["veomni_revision"] != args.expected_veomni_commit
        ):
            raise PairV5T2VGuidanceTrainingError("scorer group provenance differs")
        for digest in candidates:
            covered_score_digests.append(
                _sha(digest, length=64, label="group candidate score digest")
            )
        checkpoint_receipt_digests.add(
            _sha(
                row["frozen_checkpoint_receipt_digest"],
                length=64,
                label="frozen checkpoint receipt digest",
            )
        )
    if (
        len(covered_score_digests) != len(set(covered_score_digests))
        or set(covered_score_digests) != score_receipt_digests
        or len(checkpoint_receipt_digests) != 1
    ):
        raise PairV5T2VGuidanceTrainingError(
            "two scorer groups do not exactly cover the validated score receipts"
        )

    try:
        checkpoint_identity = checkpoint_audit.validate_checkpoint_content(
            Path(args.checkpoint),
            Path(args.checkpoint_content_manifest),
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        )
    except Exception as error:
        raise PairV5T2VGuidanceTrainingError(
            f"runtime checkpoint content audit failed: {error}"
        ) from error
    checkpoint_receipt_digest = energy_scorer.object_sha256(checkpoint_identity)
    if checkpoint_receipt_digests != {checkpoint_receipt_digest}:
        raise PairV5T2VGuidanceTrainingError(
            "runtime checkpoint differs from the frozen scorer checkpoint"
        )
    frozen_certificate = {
        "base_frozen": True,
        "trainable_parameter_tensors": 0,
        "trainable_parameter_elements": 0,
        "lora_module_count": 0,
    }
    expected_content_binding = energy_scorer.checkpoint_content_binding(
        checkpoint_identity, frozen_certificate
    )
    if any(
        row["checkpoint_content_binding"] != expected_content_binding
        for row in group_rows
    ) or any(
        score["frozen_checkpoint_receipt_digest"] != checkpoint_receipt_digest
        or score["checkpoint_content_binding"] != expected_content_binding
        or score["root_spec_raw_sha256"] != source_spec_sha
        or score["bank_receipt_digest"] != source_bank_digest
        for score in score_receipts.values()
    ):
        raise PairV5T2VGuidanceTrainingError(
            "score receipts do not bind the audited runtime checkpoint/bank"
        )

    return CAGDAuthorization(
        evidence_file=FileBinding(evidence_path, evidence_sha),
        evidence_digest=_sha(
            authorization["evidence_digest"], length=64, label="evidence digest"
        ),
        authorization_digest=authorization_digest,
        calibration_receipt_digest=_sha(
            authorization["recomputed_calibration_receipt_digest"],
            length=64,
            label="recomputed calibration receipt digest",
        ),
        checkpoint_content_receipt_digest=checkpoint_receipt_digest,
        scorer_group_files=tuple(group_files),
        scorer_group_receipt_digests=tuple(
            _sha(row["receipt_digest"], length=64, label="group receipt digest")
            for row in group_rows
        ),
    )


def _disable_gradient_checkpointing(renderer: Any, transformer: Any) -> Mapping[str, Any]:
    disable = getattr(renderer, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    for owner in (renderer, transformer):
        if hasattr(owner, "gradient_checkpointing"):
            setattr(owner, "gradient_checkpointing", False)
    if bool(getattr(renderer, "is_gradient_checkpointing", False)) or bool(
        getattr(transformer, "gradient_checkpointing", False)
    ):
        raise PairV5T2VGuidanceTrainingError("gradient checkpointing remains enabled")
    return {
        "disabled": True,
        "reason": "branch-local Action-LoRA route context must survive exact serial VJP",
    }


class NativeT2VGuidanceCallback:
    """Bernini target-only callback caching one patched same-state packet."""

    def __init__(
        self,
        *,
        diffusion: Any,
        transformer: Any,
        action_handle: action_adapter.PairV5ActionAdapterHandle,
        condition_by_branch: Mapping[str, Any],
        prompt_by_branch: Mapping[str, str],
        sp_rank: int,
    ) -> None:
        import torch

        if set(condition_by_branch) != set(guidance.BRANCH_ORDER):
            raise PairV5T2VGuidanceTrainingError("text embedding branch closure differs")
        self.diffusion = diffusion
        self.transformer = transformer
        self.action_handle = action_handle
        self.condition_by_branch = dict(condition_by_branch)
        self.prompt_by_branch = guidance.validate_prompt_bank(prompt_by_branch)
        self.sp_rank = sp_rank
        self._query_id: Optional[int] = None
        self._branch: Any = None
        self._video_shape: Optional[tuple[int, ...]] = None
        self._torch = torch

    def _patch(self, query: guidance.SameStateQuery) -> None:
        torch = self._torch
        if self._query_id is not None and self._query_id != id(query):
            self._query_id = None
            self._branch = None
            self._video_shape = None
        if self._query_id is not None:
            return
        dtype = getattr(self.transformer, "dtype", None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise PairV5T2VGuidanceTrainingError("transformer dtype differs")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            patched = self.transformer.patch_vae_latent(
                query.x_sigma.to(dtype=dtype), source_id=0
            )
        if not isinstance(patched, (tuple, list)) or len(patched) != 2:
            raise PairV5T2VGuidanceTrainingError("native T2V patch result differs")
        self._branch = t2v_runtime.build_t2v_target_branch(
            patched[0], patched[1], target_source_id=0
        )
        self._video_shape = tuple(int(item) for item in query.x_sigma.shape)
        self._query_id = id(query)

    def __call__(self, request: guidance.DenoiseRequest) -> Any:
        torch = self._torch
        if not isinstance(request, guidance.DenoiseRequest):
            raise PairV5T2VGuidanceTrainingError("native callback request type differs")
        request.query.assert_unchanged()
        if (
            request.branch not in guidance.BRANCH_ORDER
            or self.prompt_by_branch[request.branch] != request.prompt
        ):
            raise PairV5T2VGuidanceTrainingError("native callback prompt binding differs")
        self._patch(request.query)
        branch = self._branch
        video_shape = self._video_shape
        if branch is None or video_shape is None:
            raise PairV5T2VGuidanceTrainingError("native same-state packet is absent")
        route = action_adapter.PairV5ActionRoute(
            total_tokens=branch.total_token_count,
            condition_tokens=0,
            sequence_parallel_rank=self.sp_rank,
            sequence_parallel_size=SP_SIZE,
            branch_name="none",
            sigma_schedule_index=request.query.schedule_index,
            enabled=request.adapter_enabled,
        )
        condition = self.condition_by_branch[request.branch]
        with self.action_handle.route(route), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            packed = self.diffusion.shared_step(
                model_id="transformer_1",
                noisy_latents=branch.noisy_latents,
                timesteps=request.query.timestep,
                cond_embeds=condition,
                rotary_embs=branch.rotary_embs,
                batch_vae_seqlen=list(branch.batch_vae_seqlen),
                batch_text_seqlen=[t2v_runtime.PINNED_TEXT_TOKENS],
            )
        if (
            not isinstance(packed, torch.Tensor)
            or tuple(int(item) for item in packed.shape)
            != (1, branch.total_token_count, t2v_runtime.PINNED_PATCH_DIM)
        ):
            raise PairV5T2VGuidanceTrainingError("native T2V prediction geometry differs")
        spatial = native_bridge._unpack_spatial_velocity(
            packed[:, -branch.target_token_count :, :], video_shape=video_shape
        )
        request.query.assert_unchanged()
        return spatial


def _broadcast_sp(value: Any, *, source_rank: int, sp_group: Any) -> Any:
    import torch.distributed as dist

    dist.broadcast(value, src=source_rank, group=sp_group)
    return value


def _encode_prompt_bank(
    *,
    renderer: Any,
    tokenizer: Any,
    prompt_by_branch: Mapping[str, str],
    device: Any,
    parallel: distributed_runtime.ParallelContext,
) -> Mapping[str, Any]:
    import torch

    source_rank = distributed_runtime.SP_GROUP_RANKS[parallel.contract.arm_index][0]
    result: dict[str, Any] = {}
    for branch in guidance.BRANCH_ORDER:
        ids, mask = legacy._tokenize_training_prompt(tokenizer, prompt_by_branch[branch])
        with torch.inference_mode():
            embedding = renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
        _broadcast_sp(embedding, source_rank=source_rank, sp_group=parallel.sp_group)
        if (
            tuple(embedding.shape)
            != (1, t2v_runtime.PINNED_TEXT_TOKENS, t2v_runtime.PINNED_TEXT_DIM)
            or embedding.requires_grad
            or not bool(torch.isfinite(embedding).all().item())
        ):
            raise PairV5T2VGuidanceTrainingError(f"prompt embedding {branch} differs")
        result[branch] = embedding
    if len({guidance.tensor_sha256(value.float()) for value in result.values()}) != len(result):
        raise PairV5T2VGuidanceTrainingError("two prompt embeddings alias exactly")
    return result


def _save_action_adapter(path: Path, handle: action_adapter.PairV5ActionAdapterHandle) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    state = dict(handle.state_dict_for_save())
    save_file(state, str(path), metadata={"schema_version": action_adapter.SCHEMA_VERSION})
    roundtrip: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if set(opened.keys()) != set(state):
            raise PairV5T2VGuidanceTrainingError("saved Action-LoRA key closure differs")
        for name in sorted(state):
            value = opened.get_tensor(name)
            if not torch.equal(value, state[name]):
                raise PairV5T2VGuidanceTrainingError(f"saved Action-LoRA tensor differs: {name}")
            roundtrip[name] = guidance.tensor_sha256(value)
    return {
        "path": str(path),
        "file_sha256": distributed_runtime.file_sha256(path),
        "tensor_count": len(state),
        "roundtrip_tensor_digest": object_sha256(roundtrip),
    }


def _publish(stage: Path, output: Path) -> None:
    os.replace(stage, output)
    distributed_runtime.fsync_directory(output.parent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=checkpoint_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument("--event-manifest", required=True)
    parser.add_argument("--expected-event-manifest-sha256", required=True)
    parser.add_argument("--cagd-validator-evidence", required=True)
    parser.add_argument("--expected-cagd-validator-evidence-sha256", required=True)
    parser.add_argument("--scorer-group-receipt", action="append", required=True)
    parser.add_argument(
        "--expected-scorer-group-receipt-sha256", action="append", required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-schedule-steps", required=True, type=int)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT)
    parser.add_argument("--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--ack-experimental-no-action-success-claim", action="store_true")
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.ack_experimental_no_action_success_claim is not True:
        raise PairV5T2VGuidanceTrainingError("experimental no-success-claim acknowledgement is required")
    if (
        type(args.max_schedule_steps) is not int
        or args.max_schedule_steps < 40
        or args.max_schedule_steps % 40
    ):
        raise PairV5T2VGuidanceTrainingError("max schedule steps must be a positive exact40 multiple")
    for name in ("learning_rate", "max_grad_norm"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise PairV5T2VGuidanceTrainingError(f"{name} must be finite positive")
    for name in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        _sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_event_manifest_sha256",
        "expected_cagd_validator_evidence_sha256",
        "expected_checkpoint_content_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        _sha(getattr(args, name), length=64, label=name)
    if (
        len(args.scorer_group_receipt) != 2
        or len(args.expected_scorer_group_receipt_sha256) != 2
    ):
        raise PairV5T2VGuidanceTrainingError(
            "exactly two scorer group receipt path/hash arguments are required"
        )
    for ordinal, digest in enumerate(args.expected_scorer_group_receipt_sha256):
        _sha(digest, length=64, label=f"scorer group receipt[{ordinal}] SHA-256")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise PairV5T2VGuidanceTrainingError("checkpoint identity differs from pinned Bernini-R 1.3B")
    if (
        args.expected_checkpoint_content_manifest_sha256
        != checkpoint_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        raise PairV5T2VGuidanceTrainingError(
            "checkpoint content manifest differs from the pinned Bernini-R 1.3B audit"
        )


def preflight(
    args: argparse.Namespace,
) -> tuple[GuidanceManifest, tuple[EventRuntime, ...], CAGDAuthorization]:
    """Complete every authorization/hash/tensor gate before model and AdamW."""

    validate_cli(args)
    manifest = load_manifest(args.event_manifest, args.expected_event_manifest_sha256)
    if manifest.checkpoint_tree_sha256 != args.expected_checkpoint_tree_sha256:
        raise PairV5T2VGuidanceTrainingError("manifest and CLI checkpoint identities differ")
    authorization = _load_cagd_authorization(args, manifest)
    events = load_event_tensors(manifest)
    return manifest, events, authorization


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, events, authorization = preflight(args)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_only": True,
                    "optimizer_authorized": True,
                    "event_count": len(events),
                    "manifest_digest": manifest.manifest_digest,
                    "cagd_authorization_digest": authorization.authorization_digest,
                    "calibration_receipt_digest": authorization.calibration_receipt_digest,
                    "legacy_eligibility_self_declaration_trusted": False,
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
        raise PairV5T2VGuidanceTrainingError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise PairV5T2VGuidanceTrainingError("pinned Bernini head count differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = distributed_runtime.distributed_contract()
    device = distributed_runtime.initialise_distributed(distributed)
    parallel = distributed_runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=SP_SIZE)
    )
    output, stage = distributed_runtime.prepare_output_transaction(
        args.output, distributed.rank, parallel.world_group
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False).eval().to(device)
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise PairV5T2VGuidanceTrainingError("guidance distillation requires one 1.3B expert")
    checkpointing = _disable_gradient_checkpointing(renderer, transformer)
    action_handle = action_adapter.install_pair_v5_action_adapter(transformer)
    trainable = action_handle.trainable_named_parameters()
    if not action_handle.base_parameters_frozen():
        raise PairV5T2VGuidanceTrainingError("Action-LoRA/base parameter closure differs")
    initial_digest = distributed_runtime.synchronize_initial_parameters(
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
    conditions_by_digest: dict[str, Mapping[str, Any]] = {}
    for event in manifest.events:
        if event.prompt_bank_sha256 not in conditions_by_digest:
            conditions_by_digest[event.prompt_bank_sha256] = _encode_prompt_bank(
                renderer=renderer,
                tokenizer=tokenizer,
                prompt_by_branch=event.prompt_by_branch,
                device=device,
                parallel=parallel,
            )
    del tokenizer

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    history: list[Mapping[str, Any]] = []
    optimizer_updates = 0
    schedule_indices: list[int] = []
    for schedule_step in range(args.max_schedule_steps):
        schedule_index = schedule_step % 40
        schedule_indices.append(schedule_index)
        event_index = (
            (schedule_step // 40) * DP_SIZE + distributed.arm_index
        ) % len(events)
        runtime_event = events[event_index]
        source_rank = distributed_runtime.SP_GROUP_RANKS[distributed.arm_index][0]
        event_latent = runtime_event.event_latent_cpu.to(device=device).contiguous()
        epsilon = runtime_event.official_epsilon_cpu.to(device=device).contiguous()
        _broadcast_sp(event_latent, source_rank=source_rank, sp_group=parallel.sp_group)
        _broadcast_sp(epsilon, source_rank=source_rank, sp_group=parallel.sp_group)
        event_tensor_digest = object_sha256(
            [guidance.tensor_sha256(event_latent), guidance.tensor_sha256(epsilon)]
        )
        distributed_runtime.digest_consensus(
            event_tensor_digest,
            group=parallel.sp_group,
            expected_count=SP_SIZE,
            label=f"guidance event tensors step {schedule_step}",
        )
        optimizer.zero_grad(set_to_none=True)
        parameter_before = distributed_runtime.trainable_parameters_digest(trainable)
        callback = NativeT2VGuidanceCallback(
            diffusion=diffusion,
            transformer=transformer,
            action_handle=action_handle,
            condition_by_branch=conditions_by_digest[
                runtime_event.spec.prompt_bank_sha256
            ],
            prompt_by_branch=runtime_event.spec.prompt_by_branch,
            sp_rank=distributed.sp_rank,
        )
        cell = guidance.run_same_state_cell(
            event_latent,
            epsilon,
            schedule_index=schedule_index,
            eligibility=runtime_event.spec.eligibility,
            prompt_by_branch=runtime_event.spec.prompt_by_branch,
            checkpoint_tree_sha256=manifest.checkpoint_tree_sha256,
            denoise_callback=callback,
            leaf_vjp_mode=True,
        )
        gate_name, gate_weight = action_adapter.sigma_gate(schedule_index)
        if cell.zero_update:
            if gate_name != "low_base_only" or schedule_index not in action_adapter.LOW_SIGMA_INDICES:
                raise PairV5T2VGuidanceTrainingError("zero-update appeared outside low sigma")
            if any(parameter.grad is not None for _, parameter in trainable):
                raise PairV5T2VGuidanceTrainingError("low-sigma anchor constructed gradients")
            parameter_after = distributed_runtime.parameter_consensus(
                trainable, parallel.world_group, f"guidance low anchor {schedule_step}"
            )
            if parameter_after != parameter_before:
                raise PairV5T2VGuidanceTrainingError("low-sigma anchor changed Action-LoRA")
            record = {
                "schedule_step": schedule_step + 1,
                "schedule_index": schedule_index,
                "event_id": runtime_event.spec.event_id,
                "action_family": runtime_event.spec.action_family,
                "analysis_split": runtime_event.spec.analysis_split,
                "prompt_bank_sha256": runtime_event.spec.prompt_bank_sha256,
                "event_digest": runtime_event.spec.event_digest,
                "update_kind": "frozen_base_anchor_zero_update",
                "optimizer_step_called": False,
                "loss": None,
                "action_match_loss": None,
                "negative_parity_loss": None,
                "preclip_gradient_norm": None,
                "vjp_replay_max_abs": None,
                "parameter_digest": parameter_after,
                "dp_rank": distributed.arm_index,
                "sp_rank": distributed.sp_rank,
                "cell_receipt_digest": cell.receipt["receipt_digest"],
            }
        else:
            if (
                gate_name == "low_base_only"
                or gate_weight <= 0.0
                or cell.objective is None
                or cell.packet is None
            ):
                raise PairV5T2VGuidanceTrainingError("trainable guidance cell closure differs")
            cell.objective.loss.backward()
            replay = guidance.replay_student_vjp(
                cell.packet,
                runtime_event.spec.prompt_by_branch,
                callback,
                rtol=VJP_RTOL,
                atol=VJP_ATOL,
            )
            for name, parameter in transformer.named_parameters():
                allowed = "action_lora_a.weight" in name or "action_lora_b.weight" in name
                if not allowed and parameter.grad is not None:
                    raise PairV5T2VGuidanceTrainingError(f"frozen base received gradient: {name}")
            preclip = distributed_runtime.synchronize_gradients(trainable, parallel)
            clipped = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], args.max_grad_norm
            )
            if not math.isfinite(float(clipped)):
                raise PairV5T2VGuidanceTrainingError("gradient clipping is non-finite")
            optimizer.step()
            optimizer_updates += 1
            parameter_after = distributed_runtime.parameter_consensus(
                trainable, parallel.world_group, f"guidance optimizer update {optimizer_updates}"
            )
            record = {
                "schedule_step": schedule_step + 1,
                "schedule_index": schedule_index,
                "event_id": runtime_event.spec.event_id,
                "action_family": runtime_event.spec.action_family,
                "analysis_split": runtime_event.spec.analysis_split,
                "prompt_bank_sha256": runtime_event.spec.prompt_bank_sha256,
                "event_digest": runtime_event.spec.event_digest,
                "update_kind": "same_state_counterfactual_action_guidance",
                "optimizer_step_called": True,
                "loss": float(cell.objective.loss.detach().item()),
                "action_match_loss": float(cell.objective.action_match_loss.detach().item()),
                "negative_parity_loss": float(cell.objective.negative_parity_loss.detach().item()),
                "preclip_gradient_norm": preclip,
                "vjp_replay_max_abs": max(replay.values()),
                "parameter_digest": parameter_after,
                "dp_rank": distributed.arm_index,
                "sp_rank": distributed.sp_rank,
                "cell_receipt_digest": cell.receipt["receipt_digest"],
            }
        projection = {key: value for key, value in record.items() if key != "sp_rank"}
        distributed_runtime.digest_consensus(
            object_sha256(projection),
            group=parallel.sp_group,
            expected_count=SP_SIZE,
            label=f"guidance step record {schedule_step}",
        )
        gathered: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, record, group=parallel.world_group)
        history.append(
            {
                "schedule_step": schedule_step + 1,
                "schedule_index": schedule_index,
                "dp_records": [gathered[0], gathered[4]],
            }
        )
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "schedule_step": schedule_step + 1,
                        "schedule_index": schedule_index,
                        "update_kind": record["update_kind"],
                        "loss_dp0": gathered[0]["loss"],
                        "loss_dp1": gathered[4]["loss"],
                        "optimizer_updates": optimizer_updates,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del event_latent, epsilon, callback, cell

    expected_indices = list(range(40)) * (args.max_schedule_steps // 40)
    if schedule_indices != expected_indices:
        raise PairV5T2VGuidanceTrainingError("trainer did not execute complete exact40 cycles")
    expected_updates = 38 * (args.max_schedule_steps // 40)
    if optimizer_updates != expected_updates:
        raise PairV5T2VGuidanceTrainingError("optimizer update count differs from exact40 gate")
    final_digest = distributed_runtime.parameter_consensus(
        trainable, parallel.world_group, "guidance final Action-LoRA"
    )
    if final_digest == initial_digest:
        raise PairV5T2VGuidanceTrainingError("guidance training did not change Action-LoRA")
    manifest.assert_unchanged()
    authorization.assert_unchanged()
    dist.barrier(group=parallel.world_group)
    if distributed.rank == 0:
        adapter_path = stage / "adapter.safetensors"
        optimizer_path = stage / "optimizer.pt"
        history_path = stage / "history.json"
        adapter_save = _save_action_adapter(adapter_path, action_handle)
        distributed_runtime.atomic_torch_save(
            optimizer_path,
            {
                "schema_version": RUN_RECEIPT_SCHEMA,
                "optimizer": optimizer.state_dict(),
                "schedule_steps": args.max_schedule_steps,
                "optimizer_updates": optimizer_updates,
                "adapter_parameter_digest": final_digest,
                "manifest_digest": manifest.manifest_digest,
            },
        )
        distributed_runtime.atomic_json(
            history_path,
            {
                "schema_version": HISTORY_SCHEMA,
                "schedule_steps": args.max_schedule_steps,
                "optimizer_updates": optimizer_updates,
                "records": history,
            },
        )
        receipt: dict[str, Any] = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "complete": True,
            "optimizer_authorized_by_event_manifest": True,
            "optimizer_authorized_by_recomputed_v3_evidence": True,
            "cagd_authorization": {
                "evidence_path": str(authorization.evidence_file.path),
                "evidence_file_sha256": authorization.evidence_file.sha256,
                "evidence_digest": authorization.evidence_digest,
                "authorization_digest": authorization.authorization_digest,
                "recomputed_calibration_receipt_digest": authorization.calibration_receipt_digest,
                "runtime_checkpoint_content_receipt_digest": authorization.checkpoint_content_receipt_digest,
                "scorer_group_receipt_digests": list(
                    authorization.scorer_group_receipt_digests
                ),
                "legacy_eligibility_self_declaration_trusted": False,
                "confirmation_rows_consumed_by_optimizer": False,
                "post_training_evidence_files_unchanged": True,
            },
            "event_manifest": {
                "path": str(manifest.path),
                "raw_sha256": manifest.raw_sha256,
                "manifest_digest": manifest.manifest_digest,
                "event_count": len(manifest.events),
                "event_ids": [event.event_id for event in manifest.events],
                "action_families": [event.action_family for event in manifest.events],
                "analysis_splits": [event.analysis_split for event in manifest.events],
                "prompt_bank_sha256_by_event": {
                    event.event_id: event.prompt_bank_sha256
                    for event in manifest.events
                },
                "cycle0_dp_assignment": [
                    {
                        "dp_rank": rank,
                        "event_id": manifest.events[rank].event_id,
                        "action_family": manifest.events[rank].action_family,
                        "analysis_split": manifest.events[rank].analysis_split,
                        "prompt_bank_sha256": manifest.events[rank].prompt_bank_sha256,
                    }
                    for rank in range(DP_SIZE)
                ],
                "confirmation_events_consumed_by_optimizer": False,
                "post_training_all_inputs_unchanged": True,
            },
            "exact81": True,
            "schedule_steps": args.max_schedule_steps,
            "optimizer_updates": optimizer_updates,
            "complete_exact40_cycles": args.max_schedule_steps // 40,
            "sigma_gate": {
                "trainable_high": list(action_adapter.HIGH_SIGMA_INDICES),
                "trainable_mid": list(action_adapter.MID_SIGMA_INDICES),
                "low_base_only_zero_update": list(action_adapter.LOW_SIGMA_INDICES),
                "low_model_callback_called": False,
                "low_backward_called": False,
                "low_optimizer_step_called": False,
                "schedule_sha256": action_adapter.sigma_strata.SCHEDULE_SHA256,
            },
            "objective": dict(guidance.contract_receipt()),
            "adapter": {
                **dict(action_handle.receipt()),
                "schema_sha256": guidance.ACTION_ADAPTER_SCHEMA_SHA256,
                "initial_parameter_digest": initial_digest,
                "final_parameter_digest": final_digest,
                "changed_by_training": True,
                "save_roundtrip": adapter_save,
                "transfer_target": "native_RV2V_same_attn2_QO_modules",
            },
            "teacher_student": {
                "frozen_base_checkpoint_shared": True,
                "one_y_sigma_object_per_cell": True,
                "action_plus_all_nine_hard_negatives": True,
                "serial_output_leaf_vjp": True,
                "one_transformer_graph_resident_at_a_time": True,
                "vjp_rtol": VJP_RTOL,
                "vjp_atol": VJP_ATOL,
            },
            "distributed": {
                "world_size": WORLD_SIZE,
                "data_parallel_size": DP_SIZE,
                "sequence_parallel_size": SP_SIZE,
                "all_eight_gpus_used": True,
            },
            "optimizer": {
                "type": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": 0.0,
                "max_grad_norm": args.max_grad_norm,
            },
            "gradient_checkpointing": dict(checkpointing),
            "input_closure": dict(_INPUT_CLOSURE),
            "pure_t2v_video_used_as_rv2v_target_input_noise_or_donor": False,
            "cross_video_residual_transport": False,
            "semantic_action_editing_success_claimed": False,
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": manifest.checkpoint_tree_sha256,
            },
            "source": {
                "method_source_revision": args.method_source_revision,
                "method_source_archive_sha256": args.method_source_archive_sha256,
            },
            "artifacts": {
                "adapter.safetensors": distributed_runtime.file_sha256(adapter_path),
                "optimizer.pt": distributed_runtime.file_sha256(optimizer_path),
                "history.json": distributed_runtime.file_sha256(history_path),
            },
        }
        receipt["receipt_digest"] = object_sha256(receipt)
        distributed_runtime.atomic_json(stage / "receipt.json", receipt)
        distributed_runtime.verify_staged_run_bundle(stage, receipt)
        _publish(stage, output)
    dist.barrier(group=parallel.world_group)
    if distributed.rank == 0:
        print(
            json.dumps(
                {
                    "complete": True,
                    "output": str(output),
                    "optimizer_updates": optimizer_updates,
                    "adapter_parameter_digest": final_digest,
                    "semantic_action_editing_success_claimed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
