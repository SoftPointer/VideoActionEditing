"""Fail-closed schemas for DCLR counterfactual preference data.

This module deliberately contains no model or tensor code.  It validates the
four immutable artifacts needed between DCLR reward diagnostics and an offline
LoRA preference update:

* source-video plus action-program records;
* pre-registered hard-action and wrong-source counterfactual banks;
* source-conditioned rollout receipts; and
* legal one-sided-near-miss preference pairs.

Every artifact is closed-schema and carries an embedded canonical-JSON digest.
File loaders additionally require a caller-pinned SHA-256.  Edited target
media and spatial privileged conditions are forbidden recursively.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping, Sequence


SOURCE_ACTION_SCHEMA = "bernini-dclr-source-action-v3"
COUNTERFACTUAL_BANK_SCHEMA = "bernini-dclr-counterfactual-bank-v3"
ROLLOUT_RECEIPT_SCHEMA = "bernini-dclr-rollout-receipt-v3"
PREFERENCE_PAIR_SCHEMA = "bernini-dclr-preference-pair-v3"
SPLIT_LEDGER_SCHEMA = "bernini-dclr-full-split-ledger-v1"
NATIVE_PROVENANCE_SCHEMA = "bernini-dclr-native-rollout-provenance-v1"
CONTENT_ARTIFACT_SCHEMA = "bernini-dclr-content-artifact-v1"
CHECKPOINT_CONTENT_SCHEMA = "bernini-dclr-checkpoint-content-v1"
RAW_REWARD_ARTIFACT_SCHEMA = "bernini-dclr-raw-reward-artifact-v1"
THRESHOLD_CALIBRATION_SCHEMA = "bernini-dclr-threshold-calibration-v1"
EVALUATOR_ARTIFACT_SCHEMA = "bernini-dclr-evaluator-artifact-v1"
SIGMA_BANK_ARTIFACT_SCHEMA = "bernini-dclr-sigma-bank-artifact-v1"
ALTERNATIVE_SEMANTIC_EVIDENCE_SCHEMA = (
    "bernini-dclr-alternative-semantic-evidence-v1"
)
REWARD_EVIDENCE_SCHEMA = "bernini-dclr-reward-evidence-v2"

EXPECTED_FRAME_COUNT = 81
EXPECTED_FPS = 25.0
MIN_WRONG_SOURCE_DECOYS = 2
MIN_REWARD_CAL_SAMPLES = 32

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")

_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "split",
        "identity_group_id",
        "scene_group_id",
        "composition_group_id",
        "source_video_sha256",
        "source_action",
        "source_action_sha256",
        "geometry",
        "matching_metadata",
        "static_predicates",
        "edit_instruction",
        "edit_instruction_sha256",
        "action_program",
        "record_digest",
    }
)
_GEOMETRY_FIELDS = frozenset(
    {
        "frame_count",
        "fps",
        "bucket_height",
        "bucket_width",
        "reference_count",
    }
)
_MATCHING_FIELDS = frozenset(
    {
        "actor_category",
        "actor_count",
        "patient_category",
        "patient_count",
        "camera_motion_bin",
        "crop_bin",
        "motion_energy_bin",
    }
)
_ACTION_PROGRAM_FIELDS = frozenset(
    {
        "actor_role",
        "patient_role",
        "preconditions",
        "ordered_milestones",
        "terminal_hold_required",
        "action_ontology_id",
    }
)
_BANK_FIELDS = frozenset(
    {
        "schema_version",
        "bank_id",
        "source_manifest_sha256",
        "registered_before_rollouts",
        "rows",
        "bank_digest",
    }
)
_BANK_ROW_FIELDS = frozenset(
    {
        "sample_id",
        "source_record_digest",
        "target_whitespace_token_count",
        "max_abs_length_delta_tokens",
        "hard_alternatives",
        "wrong_source_decoys",
    }
)
_ALTERNATIVE_FIELDS = frozenset(
    {
        "alternative_id",
        "mutation_axis",
        "full_caption",
        "full_caption_sha256",
        "whitespace_token_count",
        "length_delta_tokens",
        "static_predicates",
        "changed_action_predicates",
        "semantic_evidence",
        "pre_registered",
    }
)
_DECOY_FIELDS = frozenset(
    {
        "decoy_id",
        "sample_id",
        "source_record_digest",
        "source_video_sha256",
        "split",
        "identity_group_id",
        "geometry_digest",
        "matching_metadata_digest",
        "pre_registered",
    }
)
_ROLLOUT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_id",
        "sample_id",
        "source_record_digest",
        "counterfactual_bank_digest",
        "policy_id",
        "policy_sha256",
        "policy_revision",
        "arm",
        "collection_episode_id",
        "candidate_set_size",
        "candidate_slot",
        "candidate_seed",
        "output_video_sha256",
        "clean_latent_sha256",
        "native_provenance_digest",
        "reward_version",
        "condition_closure",
        "evaluated_alternative_ids",
        "evaluated_wrong_source_decoy_ids",
        "action_axis_pass",
        "preservation_axis_pass",
        "action_pass",
        "preservation_pass",
        "joint_pass",
        "reward_evidence",
        "receipt_digest",
    }
)

_SPLIT_LEDGER_FIELDS = frozenset(
    {
        "schema_version",
        "ledger_id",
        "source_manifest_sha256",
        "entries",
        "ledger_digest",
    }
)
_SPLIT_LEDGER_ENTRY_FIELDS = frozenset(
    {
        "sample_id",
        "split",
        "identity_group_id",
        "scene_group_id",
        "composition_group_id",
        "source_video_sha256",
        "source_record_digest",
    }
)
_CONTENT_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "content_sha256",
        "media_type",
        "artifact_digest",
    }
)
_LATENT_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "content_sha256",
        "coordinate",
        "tensor_key",
        "dtype",
        "shape",
        "native_sampler_before_vae_decode",
        "mp4_decode_reencode_used",
        "artifact_digest",
    }
)
_CHECKPOINT_CONTENT_FIELDS = frozenset(
    {
        "schema_version",
        "tree_sha256",
        "manifest_sha256",
        "verified_entries_digest",
        "verified_file_count",
        "every_file_sha256_verified",
        "artifact_digest",
    }
)
_NATIVE_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "provenance_id",
        "sample_id",
        "source_record_digest",
        "source_video_artifact",
        "edit_instruction",
        "edit_instruction_sha256",
        "policy_id",
        "policy_artifact",
        "policy_sha256",
        "policy_revision",
        "checkpoint_content",
        "arm",
        "collection_episode_id",
        "candidate_set_size",
        "candidate_slot",
        "candidate_seed",
        "output_video_artifact",
        "clean_latent_artifact",
        "external_inputs",
        "paired_target_accessed",
        "provenance_digest",
    }
)
_RAW_REWARD_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "sample_id",
        "source_record_digest",
        "split_ledger_digest",
        "native_provenance_digest",
        "output_video_sha256",
        "clean_latent_sha256",
        "reward_version",
        "evaluator_artifact_digest",
        "sigma_bank_artifact_digest",
        "evaluated_alternative_ids",
        "evaluated_wrong_source_decoy_ids",
        "alternative_semantic_evidence_digests",
        "wrong_source_evidence_sha256_by_decoy",
        "action_axis_raw_scores",
        "preservation_axis_raw_scores",
        "artifact_digest",
    }
)
_THRESHOLD_FIELDS = frozenset({"threshold", "higher_is_better"})
_THRESHOLD_CALIBRATION_FIELDS = frozenset(
    {
        "schema_version",
        "calibration_id",
        "source_manifest_sha256",
        "split_ledger_digest",
        "evaluator_artifact_digest",
        "sigma_bank_artifact_digest",
        "calibration_sample_ids",
        "action_axis_thresholds",
        "preservation_axis_thresholds",
        "artifact_digest",
    }
)
_EVALUATOR_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "evaluator_id",
        "implementation_artifact",
        "checkpoint_artifact",
        "frozen_before_rollouts",
        "independent_from_policy",
        "artifact_digest",
    }
)
_SIGMA_BANK_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "bank_id",
        "sigmas",
        "weights",
        "registered_before_rollouts",
        "artifact_digest",
    }
)
_CONDITION_CLOSURE_FIELDS = frozenset(
    {"external_inputs", "privileged_inputs_accessed"}
)
_PAIR_FIELDS = frozenset(
    {
        "schema_version",
        "pair_id",
        "winner_receipt_digest",
        "loser_receipt_digest",
        "pair_type",
        "collection_policy_revision",
        "training_policy_revision",
        "pair_digest",
    }
)

_ALTERNATIVE_SEMANTIC_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_sha256",
        "evaluator_sha256",
        "target_instruction_sha256",
        "source_action_sha256",
        "alternative_caption_sha256",
        "mutation_axis",
        "verdict",
        "evidence_digest",
    }
)
_REWARD_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "raw_reward_evidence_sha256",
        "threshold_calibration_sha256",
        "evaluator_sha256",
        "sigma_bank_sha256",
        "alternative_semantic_evidence_digests",
        "wrong_source_evidence_sha256_by_decoy",
        "action_axis_calibrated_margins",
        "preservation_axis_calibrated_margins",
        "evidence_digest",
    }
)

_ALLOWED_SPLITS = frozenset(
    {"train", "reward_cal", "policy_val", "sealed_test", "dev"}
)
_ALLOWED_MUTATION_AXES = frozenset(
    {
        "no_op",
        "source_action",
        "wrong_action",
        "wrong_direction",
        "wrong_limb",
        "reverse_order",
        "incomplete",
        "no_hold",
        "already_complete",
        "camera_only",
        "wrong_actor",
        "wrong_patient",
        "object_self_motion",
    }
)
_ALLOWED_PAIR_TYPES = frozenset(
    {"action_nearmiss", "preservation_nearmiss"}
)

# These are payload/conditioning keys, not harmless prose mentioning a target.
# Closed schemas catch unknown top-level keys; the recursive scan also catches
# privileged payloads smuggled into otherwise free-form nested dictionaries.
_FORBIDDEN_CONDITION_KEYS = frozenset(
    {
        "target",
        "paired_target",
        "target_video",
        "target_video_path",
        "target_video_sha256",
        "target_media",
        "target_frames",
        "mask",
        "mask_path",
        "mask_tensor",
        "swept_tube",
        "track",
        "track_path",
        "tracking_data",
        "pose",
        "pose_path",
        "trajectory",
        "trajectory_path",
        "flow",
        "optical_flow",
        "reference_video",
        "reference_media",
    }
)


class DCLRCounterfactualBankError(ValueError):
    """A DCLR artifact violates its closed, hash-bound contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise DCLRCounterfactualBankError(
            f"value is not canonical-JSON serializable: {error}"
        ) from error
    return text.encode("utf-8")


def canonical_object_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical UTF-8 JSON for ``value``."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def embedded_object_sha256(value: Mapping[str, Any], digest_field: str) -> str:
    """Hash a mapping after excluding one top-level embedded digest field."""

    if not isinstance(value, Mapping):
        raise DCLRCounterfactualBankError("embedded digest owner must be an object")
    payload = {key: item for key, item in value.items() if key != digest_field}
    return canonical_object_sha256(payload)


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Hash one existing non-symlink regular file."""

    file_path = Path(path)
    try:
        metadata = file_path.lstat()
    except OSError as error:
        raise DCLRCounterfactualBankError(
            f"cannot stat hash-bound file {file_path}: {error}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DCLRCounterfactualBankError(
            f"hash-bound path must be a non-symlink regular file: {file_path}"
        )
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise DCLRCounterfactualBankError(
            f"cannot read hash-bound file {file_path}: {error}"
        ) from error
    return digest.hexdigest()


def _require_exact_fields(
    label: str, value: Any, expected: frozenset[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DCLRCounterfactualBankError(f"{label} must be an object")
    observed = frozenset(value.keys())
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise DCLRCounterfactualBankError(
            f"{label} fields are not closed: missing={missing}, extra={extra}"
        )
    return value


def _require_string(label: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DCLRCounterfactualBankError(
            f"{label} must be a nonempty, whitespace-trimmed string"
        )
    return value


def _require_slug(label: str, value: Any) -> str:
    result = _require_string(label, value)
    if _SLUG_RE.fullmatch(result) is None:
        raise DCLRCounterfactualBankError(f"{label} is not a canonical slug")
    return result


def _require_sha256(label: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DCLRCounterfactualBankError(
            f"{label} must be one lowercase SHA-256 digest"
        )
    return value


def _require_bool(label: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise DCLRCounterfactualBankError(f"{label} must be boolean")
    return value


def _require_int(label: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DCLRCounterfactualBankError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _require_positive_number(label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DCLRCounterfactualBankError(f"{label} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise DCLRCounterfactualBankError(f"{label} must be a positive number")
    return result


def _require_finite_number(label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DCLRCounterfactualBankError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DCLRCounterfactualBankError(f"{label} must be a finite number")
    return result


def _require_string_list(
    label: str,
    value: Any,
    *,
    minimum: int = 1,
    preserve_order: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < minimum:
        raise DCLRCounterfactualBankError(
            f"{label} must be a list with at least {minimum} entries"
        )
    result = tuple(
        _require_string(f"{label}[{index}]", item)
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise DCLRCounterfactualBankError(f"{label} contains duplicate entries")
    if not preserve_order and result != tuple(sorted(result)):
        raise DCLRCounterfactualBankError(f"{label} must be sorted")
    return result


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _reject_privileged_condition_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalize_key(key)
            if normalized in _FORBIDDEN_CONDITION_KEYS:
                raise DCLRCounterfactualBankError(
                    "privileged paired-target/spatial field is forbidden at "
                    f"{path}.{key}"
                )
            _reject_privileged_condition_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_privileged_condition_keys(item, path=f"{path}[{index}]")


def _validate_embedded_digest(
    label: str, record: Mapping[str, Any], digest_field: str
) -> str:
    declared = _require_sha256(f"{label}.{digest_field}", record[digest_field])
    observed = embedded_object_sha256(record, digest_field)
    if declared != observed:
        raise DCLRCounterfactualBankError(
            f"{label} embedded digest mismatch: {declared} != {observed}"
        )
    return declared


def _whitespace_token_count(text: str) -> int:
    return len(text.split())


def _expected_composition_group_id(
    matching_metadata: Mapping[str, Any], action_program: Mapping[str, Any]
) -> str:
    """Return the ontology tuple used for actor×action×patient holdout.

    Free-form milestone text is deliberately absent.  A paraphrase must not
    create a new split group; the explicit action ontology ID is the only
    admitted action component.
    """

    actor = _require_slug(
        "matching_metadata.actor_category", matching_metadata["actor_category"]
    )
    action = _require_slug(
        "action_program.action_ontology_id", action_program["action_ontology_id"]
    )
    patient_value = matching_metadata["patient_category"]
    patient = (
        "none"
        if patient_value is None
        else _require_slug("matching_metadata.patient_category", patient_value)
    )
    composition = f"{actor}--{action}--{patient}"
    if _SLUG_RE.fullmatch(composition) is None:
        raise DCLRCounterfactualBankError(
            "derived ontology composition_group_id is not a canonical slug"
        )
    return composition


def _validate_geometry(value: Any, *, label: str) -> Mapping[str, Any]:
    geometry = _require_exact_fields(label, value, _GEOMETRY_FIELDS)
    frame_count = _require_int(
        f"{label}.frame_count", geometry["frame_count"], minimum=1
    )
    if frame_count != EXPECTED_FRAME_COUNT:
        raise DCLRCounterfactualBankError(
            f"{label}.frame_count must equal {EXPECTED_FRAME_COUNT}"
        )
    fps = _require_positive_number(f"{label}.fps", geometry["fps"])
    if fps != EXPECTED_FPS:
        raise DCLRCounterfactualBankError(
            f"{label}.fps must equal {EXPECTED_FPS:g}"
        )
    _require_int(f"{label}.bucket_height", geometry["bucket_height"], minimum=1)
    _require_int(f"{label}.bucket_width", geometry["bucket_width"], minimum=1)
    _require_int(f"{label}.reference_count", geometry["reference_count"], minimum=0)
    return geometry


def _validate_matching_metadata(value: Any, *, label: str) -> Mapping[str, Any]:
    metadata = _require_exact_fields(label, value, _MATCHING_FIELDS)
    _require_slug(f"{label}.actor_category", metadata["actor_category"])
    _require_int(f"{label}.actor_count", metadata["actor_count"], minimum=1)
    patient_count = _require_int(
        f"{label}.patient_count", metadata["patient_count"], minimum=0
    )
    patient_category = metadata["patient_category"]
    if patient_count == 0:
        if patient_category is not None:
            raise DCLRCounterfactualBankError(
                f"{label}.patient_category must be null when patient_count is zero"
            )
    else:
        _require_slug(f"{label}.patient_category", patient_category)
    for key in ("camera_motion_bin", "crop_bin", "motion_energy_bin"):
        _require_slug(f"{label}.{key}", metadata[key])
    return metadata


def validate_source_action_record(record: Any) -> Mapping[str, Any]:
    """Validate one source-only action record and return it unchanged."""

    _reject_privileged_condition_keys(record)
    source = _require_exact_fields("source action record", record, _SOURCE_FIELDS)
    if source["schema_version"] != SOURCE_ACTION_SCHEMA:
        raise DCLRCounterfactualBankError("unexpected source action schema")
    _require_slug("sample_id", source["sample_id"])
    split = _require_slug("split", source["split"])
    if split not in _ALLOWED_SPLITS:
        raise DCLRCounterfactualBankError(f"unsupported split: {split}")
    _require_slug("identity_group_id", source["identity_group_id"])
    _require_slug("scene_group_id", source["scene_group_id"])
    composition_group_id = _require_slug(
        "composition_group_id", source["composition_group_id"]
    )
    _require_sha256("source_video_sha256", source["source_video_sha256"])
    source_action = _require_string("source_action", source["source_action"])
    source_action_sha = _require_sha256(
        "source_action_sha256", source["source_action_sha256"]
    )
    if source_action_sha != hashlib.sha256(source_action.encode("utf-8")).hexdigest():
        raise DCLRCounterfactualBankError("source action SHA-256 mismatch")
    _validate_geometry(source["geometry"], label="geometry")
    matching_metadata = _validate_matching_metadata(
        source["matching_metadata"], label="matching_metadata"
    )
    static_predicates = _require_string_list(
        "static_predicates", source["static_predicates"]
    )
    instruction = _require_string("edit_instruction", source["edit_instruction"])
    instruction_sha = _require_sha256(
        "edit_instruction_sha256", source["edit_instruction_sha256"]
    )
    observed_instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    if instruction_sha != observed_instruction_sha:
        raise DCLRCounterfactualBankError("edit instruction SHA-256 mismatch")
    if source_action == instruction:
        raise DCLRCounterfactualBankError(
            "source action must differ from the requested target action"
        )

    program = _require_exact_fields(
        "action_program", source["action_program"], _ACTION_PROGRAM_FIELDS
    )
    _require_string("action_program.actor_role", program["actor_role"])
    _require_slug(
        "action_program.action_ontology_id", program["action_ontology_id"]
    )
    if program["patient_role"] is not None:
        _require_string("action_program.patient_role", program["patient_role"])
    _require_string_list("action_program.preconditions", program["preconditions"])
    milestones = _require_string_list(
        "action_program.ordered_milestones",
        program["ordered_milestones"],
        minimum=2,
    )
    _require_bool(
        "action_program.terminal_hold_required",
        program["terminal_hold_required"],
    )
    if set(static_predicates).intersection(milestones):
        raise DCLRCounterfactualBankError(
            "static predicates and ordered action milestones must be disjoint"
        )
    expected_composition = _expected_composition_group_id(
        matching_metadata, program
    )
    if composition_group_id != expected_composition:
        raise DCLRCounterfactualBankError(
            "composition_group_id must equal the explicit "
            "actor_category--action_ontology_id--patient_category tuple"
        )
    _validate_embedded_digest("source action record", source, "record_digest")
    return source


def validate_source_action_records(
    records: Iterable[Any],
) -> dict[str, Mapping[str, Any]]:
    """Validate and index unique source records by sample ID."""

    indexed: dict[str, Mapping[str, Any]] = {}
    digests: set[str] = set()
    split_by_identity: dict[str, str] = {}
    split_by_scene: dict[str, str] = {}
    split_by_source_video: dict[str, str] = {}
    split_by_action_composition: dict[str, str] = {}
    for index, value in enumerate(records):
        try:
            record = validate_source_action_record(value)
        except DCLRCounterfactualBankError as error:
            raise DCLRCounterfactualBankError(
                f"source record {index} is invalid: {error}"
            ) from error
        sample_id = str(record["sample_id"])
        digest = str(record["record_digest"])
        if sample_id in indexed:
            raise DCLRCounterfactualBankError(
                f"duplicate source sample_id: {sample_id}"
            )
        if digest in digests:
            raise DCLRCounterfactualBankError(
                f"duplicate source record digest: {digest}"
            )
        split = str(record["split"])
        group_keys = (
            (
                "identity_group_id",
                str(record["identity_group_id"]),
                split_by_identity,
            ),
            ("scene_group_id", str(record["scene_group_id"]), split_by_scene),
            (
                "source_video_sha256",
                str(record["source_video_sha256"]),
                split_by_source_video,
            ),
            (
                "action_composition",
                str(record["composition_group_id"]),
                split_by_action_composition,
            ),
        )
        for label, group_id, registry in group_keys:
            previous = registry.setdefault(group_id, split)
            if previous != split:
                raise DCLRCounterfactualBankError(
                    f"{label} crosses data splits: {previous} vs {split}"
                )
        indexed[sample_id] = record
        digests.add(digest)
    if not indexed:
        raise DCLRCounterfactualBankError("source manifest is empty")
    return indexed


def validate_full_split_ledger(
    document: Any,
    sources: Mapping[str, Mapping[str, Any]],
    *,
    expected_source_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Validate one full-manifest split ledger, not a training-only subset.

    Coverage must be exactly the caller-hash-bound source manifest.  Identity,
    scene, source bytes, and explicit ontology composition groups may occur in
    only one split across that complete ledger.
    """

    validated_sources = validate_source_action_records(sources.values())
    if set(validated_sources) != set(sources) or any(
        validated_sources[key] is not sources[key] for key in validated_sources
    ):
        raise DCLRCounterfactualBankError(
            "split-ledger source registry keys differ from embedded sample IDs"
        )
    ledger = _require_exact_fields(
        "full split ledger", document, _SPLIT_LEDGER_FIELDS
    )
    if ledger["schema_version"] != SPLIT_LEDGER_SCHEMA:
        raise DCLRCounterfactualBankError("unexpected full split ledger schema")
    _require_slug("full split ledger.ledger_id", ledger["ledger_id"])
    manifest_sha = _require_sha256(
        "full split ledger.source_manifest_sha256",
        ledger["source_manifest_sha256"],
    )
    if manifest_sha != _require_sha256(
        "expected_source_manifest_sha256", expected_source_manifest_sha256
    ):
        raise DCLRCounterfactualBankError(
            "full split ledger is bound to another source manifest"
        )
    entries = ledger["entries"]
    if not isinstance(entries, list) or not entries:
        raise DCLRCounterfactualBankError(
            "full split ledger entries must be a nonempty list"
        )
    indexed: dict[str, Mapping[str, Any]] = {}
    split_by_group: dict[str, dict[str, str]] = {
        "identity_group_id": {},
        "scene_group_id": {},
        "composition_group_id": {},
        "source_video_sha256": {},
    }
    ordered_sample_ids: list[str] = []
    for index, raw_entry in enumerate(entries):
        label = f"full split ledger entry {index}"
        entry = _require_exact_fields(
            label, raw_entry, _SPLIT_LEDGER_ENTRY_FIELDS
        )
        sample_id = _require_slug(f"{label}.sample_id", entry["sample_id"])
        if sample_id in indexed:
            raise DCLRCounterfactualBankError(
                f"duplicate full split ledger sample: {sample_id}"
            )
        if sample_id not in validated_sources:
            raise DCLRCounterfactualBankError(
                f"full split ledger references unknown sample: {sample_id}"
            )
        source = validated_sources[sample_id]
        for field in (
            "split",
            "identity_group_id",
            "scene_group_id",
            "composition_group_id",
            "source_video_sha256",
        ):
            if entry[field] != source[field]:
                raise DCLRCounterfactualBankError(
                    f"{label}.{field} differs from the source manifest"
                )
        if entry["source_record_digest"] != source["record_digest"]:
            raise DCLRCounterfactualBankError(
                f"{label}.source_record_digest differs from the source manifest"
            )
        split = _require_slug(f"{label}.split", entry["split"])
        if split not in _ALLOWED_SPLITS:
            raise DCLRCounterfactualBankError(
                f"{label}.split is unsupported: {split}"
            )
        _require_sha256(
            f"{label}.source_video_sha256", entry["source_video_sha256"]
        )
        _require_sha256(
            f"{label}.source_record_digest", entry["source_record_digest"]
        )
        for group_field, registry in split_by_group.items():
            group = str(entry[group_field])
            previous = registry.setdefault(group, split)
            if previous != split:
                raise DCLRCounterfactualBankError(
                    f"full ledger {group_field} crosses data splits: "
                    f"{previous} vs {split}"
                )
        indexed[sample_id] = entry
        ordered_sample_ids.append(sample_id)
    if ordered_sample_ids != sorted(ordered_sample_ids):
        raise DCLRCounterfactualBankError(
            "full split ledger entries must be sorted by sample_id"
        )
    if set(indexed) != set(validated_sources):
        missing = sorted(set(validated_sources) - set(indexed))
        extra = sorted(set(indexed) - set(validated_sources))
        raise DCLRCounterfactualBankError(
            "full split ledger does not cover the complete source manifest: "
            f"missing={missing}, extra={extra}"
        )
    _validate_embedded_digest("full split ledger", ledger, "ledger_digest")
    return ledger


def _validate_alternative_semantic_evidence(
    value: Any,
    *,
    source: Mapping[str, Any],
    alternative: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    evidence = _require_exact_fields(
        label, value, _ALTERNATIVE_SEMANTIC_EVIDENCE_FIELDS
    )
    if evidence["schema_version"] != ALTERNATIVE_SEMANTIC_EVIDENCE_SCHEMA:
        raise DCLRCounterfactualBankError(
            f"{label} has an unexpected semantic-evidence schema"
        )
    _require_sha256(f"{label}.evidence_sha256", evidence["evidence_sha256"])
    _require_sha256(f"{label}.evaluator_sha256", evidence["evaluator_sha256"])
    expected_fields = {
        "target_instruction_sha256": source["edit_instruction_sha256"],
        "source_action_sha256": source["source_action_sha256"],
        "alternative_caption_sha256": alternative["full_caption_sha256"],
        "mutation_axis": alternative["mutation_axis"],
    }
    for field, expected in expected_fields.items():
        if evidence[field] != expected:
            raise DCLRCounterfactualBankError(
                f"{label}.{field} is not bound to the source/alternative"
            )
    if evidence["verdict"] != "valid_hard_negative_for_target_action":
        raise DCLRCounterfactualBankError(
            f"{label}.verdict does not authorize a hard negative"
        )
    _validate_embedded_digest(label, evidence, "evidence_digest")
    return evidence


def _validate_alternative(
    value: Any,
    *,
    source: Mapping[str, Any],
    target_token_count: int,
    max_abs_delta: int,
    label: str,
) -> Mapping[str, Any]:
    alternative = _require_exact_fields(label, value, _ALTERNATIVE_FIELDS)
    _require_slug(f"{label}.alternative_id", alternative["alternative_id"])
    axis = _require_slug(f"{label}.mutation_axis", alternative["mutation_axis"])
    if axis not in _ALLOWED_MUTATION_AXES:
        raise DCLRCounterfactualBankError(
            f"{label}.mutation_axis is unsupported: {axis}"
        )
    caption = _require_string(f"{label}.full_caption", alternative["full_caption"])
    if caption == source["edit_instruction"]:
        raise DCLRCounterfactualBankError(
            f"{label}.full_caption must differ from the target instruction"
        )
    caption_sha = _require_sha256(
        f"{label}.full_caption_sha256", alternative["full_caption_sha256"]
    )
    if caption_sha != hashlib.sha256(caption.encode("utf-8")).hexdigest():
        raise DCLRCounterfactualBankError(f"{label} caption SHA-256 mismatch")
    token_count = _require_int(
        f"{label}.whitespace_token_count",
        alternative["whitespace_token_count"],
        minimum=1,
    )
    if token_count != _whitespace_token_count(caption):
        raise DCLRCounterfactualBankError(
            f"{label}.whitespace_token_count does not match full_caption"
        )
    delta = alternative["length_delta_tokens"]
    if isinstance(delta, bool) or not isinstance(delta, int):
        raise DCLRCounterfactualBankError(
            f"{label}.length_delta_tokens must be an integer"
        )
    if delta != token_count - target_token_count:
        raise DCLRCounterfactualBankError(
            f"{label}.length_delta_tokens is inconsistent"
        )
    if abs(delta) > max_abs_delta:
        raise DCLRCounterfactualBankError(
            f"{label} exceeds the pre-registered caption-length delta"
        )
    static_predicates = _require_string_list(
        f"{label}.static_predicates", alternative["static_predicates"]
    )
    if static_predicates != tuple(source["static_predicates"]):
        raise DCLRCounterfactualBankError(
            f"{label} changed or reordered static predicates"
        )
    changed = _require_string_list(
        f"{label}.changed_action_predicates",
        alternative["changed_action_predicates"],
    )
    if set(changed).intersection(static_predicates):
        raise DCLRCounterfactualBankError(
            f"{label} labels a static predicate as changed"
        )
    if axis == "source_action":
        if caption != source["source_action"]:
            raise DCLRCounterfactualBankError(
                f"{label} source_action caption differs from the source record"
            )
    elif caption == source["source_action"]:
        raise DCLRCounterfactualBankError(
            f"{label} reuses source_action under another mutation axis"
        )
    _validate_alternative_semantic_evidence(
        alternative["semantic_evidence"],
        source=source,
        alternative=alternative,
        label=f"{label}.semantic_evidence",
    )
    if alternative["pre_registered"] is not True:
        raise DCLRCounterfactualBankError(
            f"{label} must prove pre_registered=true"
        )
    return alternative


def _validate_decoy(
    value: Any,
    *,
    source: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> Mapping[str, Any]:
    decoy = _require_exact_fields(label, value, _DECOY_FIELDS)
    _require_slug(f"{label}.decoy_id", decoy["decoy_id"])
    decoy_sample_id = _require_slug(f"{label}.sample_id", decoy["sample_id"])
    if decoy_sample_id not in sources:
        raise DCLRCounterfactualBankError(
            f"{label} references an unknown source sample: {decoy_sample_id}"
        )
    decoy_source = sources[decoy_sample_id]
    if decoy_sample_id == source["sample_id"]:
        raise DCLRCounterfactualBankError(f"{label} reuses the correct source")
    declared_digest = _require_sha256(
        f"{label}.source_record_digest", decoy["source_record_digest"]
    )
    if declared_digest != decoy_source["record_digest"]:
        raise DCLRCounterfactualBankError(
            f"{label} source record digest mismatch"
        )
    if decoy["source_video_sha256"] != decoy_source["source_video_sha256"]:
        raise DCLRCounterfactualBankError(f"{label} source video hash mismatch")
    _require_sha256(f"{label}.source_video_sha256", decoy["source_video_sha256"])
    if decoy_source["source_video_sha256"] == source["source_video_sha256"]:
        raise DCLRCounterfactualBankError(
            f"{label} must use different source-video bytes"
        )
    if decoy["split"] != decoy_source["split"] or decoy["split"] != source["split"]:
        raise DCLRCounterfactualBankError(
            f"{label} must come from the same split as the correct source"
        )
    if decoy["identity_group_id"] != decoy_source["identity_group_id"]:
        raise DCLRCounterfactualBankError(
            f"{label} identity-group declaration mismatch"
        )
    if decoy_source["identity_group_id"] == source["identity_group_id"]:
        raise DCLRCounterfactualBankError(
            f"{label} must use a distinct identity group"
        )
    geometry_digest = _require_sha256(
        f"{label}.geometry_digest", decoy["geometry_digest"]
    )
    if geometry_digest != canonical_object_sha256(decoy_source["geometry"]):
        raise DCLRCounterfactualBankError(f"{label} geometry digest mismatch")
    if decoy_source["geometry"] != source["geometry"]:
        raise DCLRCounterfactualBankError(
            f"{label} geometry does not exactly match the correct source"
        )
    matching_digest = _require_sha256(
        f"{label}.matching_metadata_digest",
        decoy["matching_metadata_digest"],
    )
    if matching_digest != canonical_object_sha256(
        decoy_source["matching_metadata"]
    ):
        raise DCLRCounterfactualBankError(
            f"{label} matching-metadata digest mismatch"
        )
    if decoy_source["matching_metadata"] != source["matching_metadata"]:
        raise DCLRCounterfactualBankError(
            f"{label} actor/patient/camera/crop/motion metadata is not matched"
        )
    if decoy["pre_registered"] is not True:
        raise DCLRCounterfactualBankError(
            f"{label} must prove pre_registered=true"
        )
    return decoy


def validate_counterfactual_bank(
    document: Any,
    sources: Mapping[str, Mapping[str, Any]],
    *,
    expected_source_manifest_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Validate a pre-rollout alternative/decoy bank against source records."""

    _reject_privileged_condition_keys(document)
    bank = _require_exact_fields("counterfactual bank", document, _BANK_FIELDS)
    if bank["schema_version"] != COUNTERFACTUAL_BANK_SCHEMA:
        raise DCLRCounterfactualBankError("unexpected counterfactual bank schema")
    _require_slug("bank_id", bank["bank_id"])
    manifest_sha = _require_sha256(
        "source_manifest_sha256", bank["source_manifest_sha256"]
    )
    if (
        expected_source_manifest_sha256 is not None
        and manifest_sha
        != _require_sha256(
            "expected_source_manifest_sha256", expected_source_manifest_sha256
        )
    ):
        raise DCLRCounterfactualBankError(
            "counterfactual bank is bound to another source manifest"
        )
    if bank["registered_before_rollouts"] is not True:
        raise DCLRCounterfactualBankError(
            "counterfactual bank must prove registered_before_rollouts=true"
        )
    if not isinstance(bank["rows"], list) or not bank["rows"]:
        raise DCLRCounterfactualBankError("counterfactual bank rows must be nonempty")

    seen_samples: set[str] = set()
    for row_index, raw_row in enumerate(bank["rows"]):
        label = f"counterfactual bank row {row_index}"
        row = _require_exact_fields(label, raw_row, _BANK_ROW_FIELDS)
        sample_id = _require_slug(f"{label}.sample_id", row["sample_id"])
        if sample_id in seen_samples:
            raise DCLRCounterfactualBankError(
                f"duplicate counterfactual bank sample: {sample_id}"
            )
        if sample_id not in sources:
            raise DCLRCounterfactualBankError(
                f"counterfactual bank references unknown sample: {sample_id}"
            )
        seen_samples.add(sample_id)
        source = sources[sample_id]
        if row["source_record_digest"] != source["record_digest"]:
            raise DCLRCounterfactualBankError(
                f"{label} source record digest mismatch"
            )
        _require_sha256(
            f"{label}.source_record_digest", row["source_record_digest"]
        )
        target_count = _require_int(
            f"{label}.target_whitespace_token_count",
            row["target_whitespace_token_count"],
            minimum=1,
        )
        if target_count != _whitespace_token_count(source["edit_instruction"]):
            raise DCLRCounterfactualBankError(
                f"{label} target caption length metadata is inconsistent"
            )
        max_abs_delta = _require_int(
            f"{label}.max_abs_length_delta_tokens",
            row["max_abs_length_delta_tokens"],
            minimum=0,
        )
        alternatives = row["hard_alternatives"]
        if not isinstance(alternatives, list) or len(alternatives) < 2:
            raise DCLRCounterfactualBankError(
                f"{label} requires at least two hard alternatives"
            )
        alternative_ids: set[str] = set()
        captions: set[str] = set()
        axes: set[str] = set()
        for alternative_index, raw_alternative in enumerate(alternatives):
            alternative = _validate_alternative(
                raw_alternative,
                source=source,
                target_token_count=target_count,
                max_abs_delta=max_abs_delta,
                label=f"{label}.hard_alternatives[{alternative_index}]",
            )
            alternative_id = str(alternative["alternative_id"])
            caption = str(alternative["full_caption"])
            if alternative_id in alternative_ids:
                raise DCLRCounterfactualBankError(
                    f"{label} has duplicate alternative IDs"
                )
            if caption in captions:
                raise DCLRCounterfactualBankError(
                    f"{label} has duplicate alternative captions"
                )
            alternative_ids.add(alternative_id)
            captions.add(caption)
            axes.add(str(alternative["mutation_axis"]))
        if "no_op" not in axes:
            raise DCLRCounterfactualBankError(
                f"{label} must pre-register a no_op alternative"
            )
        if "source_action" not in axes:
            raise DCLRCounterfactualBankError(
                f"{label} must pre-register the observed source_action"
            )

        decoys = row["wrong_source_decoys"]
        if not isinstance(decoys, list) or len(decoys) < MIN_WRONG_SOURCE_DECOYS:
            raise DCLRCounterfactualBankError(
                f"{label} requires at least {MIN_WRONG_SOURCE_DECOYS} "
                "wrong-source decoys"
            )
        decoy_ids: set[str] = set()
        decoy_samples: set[str] = set()
        decoy_identity_groups: set[str] = set()
        for decoy_index, raw_decoy in enumerate(decoys):
            decoy = _validate_decoy(
                raw_decoy,
                source=source,
                sources=sources,
                label=f"{label}.wrong_source_decoys[{decoy_index}]",
            )
            decoy_id = str(decoy["decoy_id"])
            decoy_sample = str(decoy["sample_id"])
            identity_group = str(decoy["identity_group_id"])
            if decoy_id in decoy_ids or decoy_sample in decoy_samples:
                raise DCLRCounterfactualBankError(
                    f"{label} contains duplicate wrong-source decoys"
                )
            if identity_group in decoy_identity_groups:
                raise DCLRCounterfactualBankError(
                    f"{label} wrong-source ensemble repeats an identity group"
                )
            decoy_ids.add(decoy_id)
            decoy_samples.add(decoy_sample)
            decoy_identity_groups.add(identity_group)
    _validate_embedded_digest("counterfactual bank", bank, "bank_digest")
    return bank


def _bank_rows_by_sample(bank: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["sample_id"]): row for row in bank["rows"]}


def _validate_axis_map(label: str, value: Any) -> Mapping[str, bool]:
    if not isinstance(value, Mapping) or not value:
        raise DCLRCounterfactualBankError(f"{label} must be a nonempty object")
    result: dict[str, bool] = {}
    for raw_axis, passed in value.items():
        axis = _require_slug(f"{label} axis", raw_axis)
        if not isinstance(passed, bool):
            raise DCLRCounterfactualBankError(
                f"{label}.{axis} must be boolean"
            )
        result[axis] = passed
    return result


def _validate_digest_map(
    label: str,
    value: Any,
    *,
    expected_keys: set[str],
    expected_values: Mapping[str, str] | None = None,
    require_unique_values: bool = False,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise DCLRCounterfactualBankError(
            f"{label} keys must exactly match {sorted(expected_keys)}"
        )
    result: dict[str, str] = {}
    for raw_key, raw_digest in value.items():
        key = _require_slug(f"{label} key", raw_key)
        digest = _require_sha256(f"{label}.{key}", raw_digest)
        if expected_values is not None and digest != expected_values[key]:
            raise DCLRCounterfactualBankError(
                f"{label}.{key} differs from registered semantic evidence"
            )
        result[key] = digest
    if require_unique_values and len(set(result.values())) != len(result):
        raise DCLRCounterfactualBankError(
            f"{label} must bind distinct raw evidence for every control"
        )
    return result


def _validate_calibrated_margin_map(
    label: str,
    value: Any,
    *,
    axis_pass: Mapping[str, bool],
) -> dict[str, float]:
    expected_keys = set(axis_pass)
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise DCLRCounterfactualBankError(
            f"{label} keys must exactly match the declared hard-gate axes"
        )
    result: dict[str, float] = {}
    for raw_axis, raw_margin in value.items():
        axis = _require_slug(f"{label} axis", raw_axis)
        margin = _require_finite_number(f"{label}.{axis}", raw_margin)
        if margin == 0.0:
            raise DCLRCounterfactualBankError(
                f"{label}.{axis} lies exactly on an ambiguous threshold"
            )
        if axis_pass[axis] != (margin > 0.0):
            raise DCLRCounterfactualBankError(
                f"{label}.{axis} sign differs from its hard-gate boolean"
            )
        result[axis] = margin
    return result


def _validate_content_artifact(
    value: Any,
    *,
    label: str,
    expected_kind: str,
    expected_content_sha256: str | None = None,
) -> Mapping[str, Any]:
    artifact = _require_exact_fields(label, value, _CONTENT_ARTIFACT_FIELDS)
    if artifact["schema_version"] != CONTENT_ARTIFACT_SCHEMA:
        raise DCLRCounterfactualBankError(f"{label} schema differs")
    if artifact["artifact_kind"] != expected_kind:
        raise DCLRCounterfactualBankError(
            f"{label}.artifact_kind must equal {expected_kind}"
        )
    content_sha = _require_sha256(
        f"{label}.content_sha256", artifact["content_sha256"]
    )
    if expected_content_sha256 is not None and content_sha != _require_sha256(
        f"{label} expected content SHA-256", expected_content_sha256
    ):
        raise DCLRCounterfactualBankError(
            f"{label} content SHA-256 differs from its bound runtime field"
        )
    media_type = _require_string(f"{label}.media_type", artifact["media_type"])
    if "/" not in media_type or "\x00" in media_type:
        raise DCLRCounterfactualBankError(
            f"{label}.media_type must be a canonical MIME-like string"
        )
    _validate_embedded_digest(label, artifact, "artifact_digest")
    return artifact


def _validate_latent_artifact(
    value: Any,
    *,
    label: str,
    expected_content_sha256: str,
) -> Mapping[str, Any]:
    artifact = _require_exact_fields(label, value, _LATENT_ARTIFACT_FIELDS)
    if artifact["schema_version"] != CONTENT_ARTIFACT_SCHEMA:
        raise DCLRCounterfactualBankError(f"{label} schema differs")
    if artifact["artifact_kind"] != "native_sampler_clean_latent":
        raise DCLRCounterfactualBankError(
            f"{label} must be a native sampler clean latent"
        )
    if _require_sha256(
        f"{label}.content_sha256", artifact["content_sha256"]
    ) != _require_sha256(
        f"{label} expected content SHA-256", expected_content_sha256
    ):
        raise DCLRCounterfactualBankError(
            f"{label} content SHA-256 differs from the rollout"
        )
    if (
        artifact["coordinate"] != "bernini_normalized_clean_vae_latent"
        or artifact["tensor_key"] != "normalized_clean_latent"
        or artifact["dtype"] != "torch.float32"
        or artifact["native_sampler_before_vae_decode"] is not True
        or artifact["mp4_decode_reencode_used"] is not False
    ):
        raise DCLRCounterfactualBankError(
            f"{label} is not an exact pre-decode normalized FP32 latent"
        )
    shape = artifact["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 5
        or any(type(item) is not int or item <= 0 for item in shape)
        or shape[:3] != [1, 16, 21]
        or shape[3] % 2
        or shape[4] % 2
    ):
        raise DCLRCounterfactualBankError(
            f"{label}.shape must be exact81 [1,16,21,H,W] with even H/W"
        )
    _validate_embedded_digest(label, artifact, "artifact_digest")
    return artifact


def _validate_checkpoint_content(
    value: Any, *, label: str
) -> Mapping[str, Any]:
    content = _require_exact_fields(label, value, _CHECKPOINT_CONTENT_FIELDS)
    if content["schema_version"] != CHECKPOINT_CONTENT_SCHEMA:
        raise DCLRCounterfactualBankError(f"{label} schema differs")
    for field in ("tree_sha256", "manifest_sha256", "verified_entries_digest"):
        _require_sha256(f"{label}.{field}", content[field])
    _require_int(
        f"{label}.verified_file_count", content["verified_file_count"], minimum=1
    )
    if content["every_file_sha256_verified"] is not True:
        raise DCLRCounterfactualBankError(
            f"{label} must prove every checkpoint file hash was verified"
        )
    _validate_embedded_digest(label, content, "artifact_digest")
    return content


def _artifact_from_registry(
    digest: Any,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
    *,
    expected_schema: str,
    label: str,
) -> Mapping[str, Any]:
    key = _require_sha256(f"{label} digest", digest)
    if not isinstance(artifacts_by_digest, Mapping):
        raise DCLRCounterfactualBankError(
            "content-addressed artifact registry must be a mapping"
        )
    try:
        artifact = artifacts_by_digest[key]
    except KeyError as error:
        raise DCLRCounterfactualBankError(
            f"{label} content artifact is absent from the registry"
        ) from error
    if not isinstance(artifact, Mapping):
        raise DCLRCounterfactualBankError(f"{label} artifact must be an object")
    if artifact.get("schema_version") != expected_schema:
        raise DCLRCounterfactualBankError(f"{label} artifact schema differs")
    digest_field = (
        "provenance_digest"
        if expected_schema == NATIVE_PROVENANCE_SCHEMA
        else "artifact_digest"
    )
    if artifact.get(digest_field) != key:
        raise DCLRCounterfactualBankError(
            f"{label} registry key differs from embedded content digest"
        )
    if embedded_object_sha256(artifact, digest_field) != key:
        raise DCLRCounterfactualBankError(
            f"{label} artifact content digest cannot be recomputed"
        )
    return artifact


def _validate_native_provenance(
    value: Any,
    *,
    source: Mapping[str, Any],
    rollout: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    provenance = _require_exact_fields(label, value, _NATIVE_PROVENANCE_FIELDS)
    if provenance["schema_version"] != NATIVE_PROVENANCE_SCHEMA:
        raise DCLRCounterfactualBankError(f"{label} schema differs")
    _require_slug(f"{label}.provenance_id", provenance["provenance_id"])
    exact_fields = (
        "sample_id",
        "source_record_digest",
        "policy_id",
        "policy_sha256",
        "policy_revision",
        "arm",
        "collection_episode_id",
        "candidate_set_size",
        "candidate_slot",
        "candidate_seed",
    )
    for field in exact_fields:
        if provenance[field] != rollout[field]:
            raise DCLRCounterfactualBankError(
                f"{label}.{field} differs from the rollout receipt"
            )
    source_artifact = _validate_content_artifact(
        provenance["source_video_artifact"],
        label=f"{label}.source_video_artifact",
        expected_kind="source_video",
        expected_content_sha256=str(source["source_video_sha256"]),
    )
    del source_artifact
    instruction = _require_string(
        f"{label}.edit_instruction", provenance["edit_instruction"]
    )
    instruction_sha = _require_sha256(
        f"{label}.edit_instruction_sha256",
        provenance["edit_instruction_sha256"],
    )
    if (
        instruction != source["edit_instruction"]
        or instruction_sha != source["edit_instruction_sha256"]
        or hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        != instruction_sha
    ):
        raise DCLRCounterfactualBankError(
            f"{label} edit prompt is not byte-bound to the source record"
        )
    _validate_content_artifact(
        provenance["policy_artifact"],
        label=f"{label}.policy_artifact",
        expected_kind="policy_checkpoint",
        expected_content_sha256=str(rollout["policy_sha256"]),
    )
    checkpoint_content = _validate_checkpoint_content(
        provenance["checkpoint_content"],
        label=f"{label}.checkpoint_content",
    )
    if checkpoint_content["tree_sha256"] != rollout["policy_sha256"]:
        raise DCLRCounterfactualBankError(
            f"{label}.checkpoint_content tree is not the rollout policy bytes"
        )
    _validate_content_artifact(
        provenance["output_video_artifact"],
        label=f"{label}.output_video_artifact",
        expected_kind="rollout_output_video",
        expected_content_sha256=str(rollout["output_video_sha256"]),
    )
    _validate_latent_artifact(
        provenance["clean_latent_artifact"],
        label=f"{label}.clean_latent_artifact",
        expected_content_sha256=str(rollout["clean_latent_sha256"]),
    )
    external_inputs = _require_string_list(
        f"{label}.external_inputs", provenance["external_inputs"]
    )
    if set(external_inputs) != {"source_video", "edit_instruction"}:
        raise DCLRCounterfactualBankError(
            f"{label} external inputs must be source_video + edit_instruction"
        )
    if provenance["paired_target_accessed"] is not False:
        raise DCLRCounterfactualBankError(
            f"{label} accessed a paired target and cannot enter preference training"
        )
    _validate_embedded_digest(label, provenance, "provenance_digest")
    return provenance


def _validate_evaluator_artifact(value: Any, *, label: str) -> Mapping[str, Any]:
    artifact = _require_exact_fields(label, value, _EVALUATOR_ARTIFACT_FIELDS)
    if artifact["schema_version"] != EVALUATOR_ARTIFACT_SCHEMA:
        raise DCLRCounterfactualBankError(f"{label} schema differs")
    _require_slug(f"{label}.evaluator_id", artifact["evaluator_id"])
    _validate_content_artifact(
        artifact["implementation_artifact"],
        label=f"{label}.implementation_artifact",
        expected_kind="evaluator_implementation",
    )
    _validate_content_artifact(
        artifact["checkpoint_artifact"],
        label=f"{label}.checkpoint_artifact",
        expected_kind="evaluator_checkpoint",
    )
    if (
        artifact["frozen_before_rollouts"] is not True
        or artifact["independent_from_policy"] is not True
    ):
        raise DCLRCounterfactualBankError(
            f"{label} must be frozen before rollouts and independent from policy"
        )
    _validate_embedded_digest(label, artifact, "artifact_digest")
    return artifact


def _validate_sigma_bank_artifact(value: Any, *, label: str) -> Mapping[str, Any]:
    artifact = _require_exact_fields(label, value, _SIGMA_BANK_ARTIFACT_FIELDS)
    if artifact["schema_version"] != SIGMA_BANK_ARTIFACT_SCHEMA:
        raise DCLRCounterfactualBankError(f"{label} schema differs")
    _require_slug(f"{label}.bank_id", artifact["bank_id"])
    sigmas = artifact["sigmas"]
    weights = artifact["weights"]
    if (
        not isinstance(sigmas, list)
        or not isinstance(weights, list)
        or len(sigmas) < 2
        or len(sigmas) != len(weights)
    ):
        raise DCLRCounterfactualBankError(
            f"{label} requires equal sigma/weight lists of length >=2"
        )
    sigma_values = tuple(
        _require_finite_number(f"{label}.sigmas[{index}]", value)
        for index, value in enumerate(sigmas)
    )
    weight_values = tuple(
        _require_finite_number(f"{label}.weights[{index}]", value)
        for index, value in enumerate(weights)
    )
    if (
        any(not 0.0 < value < 1.0 for value in sigma_values)
        or len(set(sigma_values)) != len(sigma_values)
        or any(value < 0.0 for value in weight_values)
        or sum(weight_values) <= 0.0
    ):
        raise DCLRCounterfactualBankError(
            f"{label} sigma/weight values violate the frozen reward bank"
        )
    if artifact["registered_before_rollouts"] is not True:
        raise DCLRCounterfactualBankError(
            f"{label} must be registered before rollouts"
        )
    _validate_embedded_digest(label, artifact, "artifact_digest")
    return artifact


def _validate_threshold_map(label: str, value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise DCLRCounterfactualBankError(
            f"{label} must be a nonempty axis threshold mapping"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for raw_axis, raw_threshold in value.items():
        axis = _require_slug(f"{label} axis", raw_axis)
        threshold = _require_exact_fields(
            f"{label}.{axis}", raw_threshold, _THRESHOLD_FIELDS
        )
        _require_finite_number(
            f"{label}.{axis}.threshold", threshold["threshold"]
        )
        _require_bool(
            f"{label}.{axis}.higher_is_better",
            threshold["higher_is_better"],
        )
        result[axis] = threshold
    return result


def _validate_calibration_artifact(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    evaluator_digest: str,
    sigma_bank_digest: str,
    label: str,
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    artifact = _require_exact_fields(
        label, value, _THRESHOLD_CALIBRATION_FIELDS
    )
    if artifact["schema_version"] != THRESHOLD_CALIBRATION_SCHEMA:
        raise DCLRCounterfactualBankError(f"{label} schema differs")
    _require_slug(f"{label}.calibration_id", artifact["calibration_id"])
    if artifact["source_manifest_sha256"] != source_manifest_sha256:
        raise DCLRCounterfactualBankError(
            f"{label} is bound to another source manifest"
        )
    if artifact["split_ledger_digest"] != split_ledger["ledger_digest"]:
        raise DCLRCounterfactualBankError(
            f"{label} is bound to another full split ledger"
        )
    if (
        artifact["evaluator_artifact_digest"] != evaluator_digest
        or artifact["sigma_bank_artifact_digest"] != sigma_bank_digest
    ):
        raise DCLRCounterfactualBankError(
            f"{label} evaluator/sigma-bank binding differs"
        )
    sample_ids = _require_string_list(
        f"{label}.calibration_sample_ids",
        artifact["calibration_sample_ids"],
        minimum=MIN_REWARD_CAL_SAMPLES,
        preserve_order=False,
    )
    for sample_id in sample_ids:
        if sample_id not in sources or sources[sample_id]["split"] != "reward_cal":
            raise DCLRCounterfactualBankError(
                f"{label} sample {sample_id} is not in the reward_cal split"
            )
    action_thresholds = _validate_threshold_map(
        f"{label}.action_axis_thresholds", artifact["action_axis_thresholds"]
    )
    preservation_thresholds = _validate_threshold_map(
        f"{label}.preservation_axis_thresholds",
        artifact["preservation_axis_thresholds"],
    )
    _validate_embedded_digest(label, artifact, "artifact_digest")
    return artifact, action_thresholds, preservation_thresholds


def _validate_raw_score_map(
    label: str, value: Any, *, expected_axes: set[str]
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != expected_axes:
        raise DCLRCounterfactualBankError(
            f"{label} axes must exactly match {sorted(expected_axes)}"
        )
    return {
        _require_slug(f"{label} axis", axis): _require_finite_number(
            f"{label}.{axis}", score
        )
        for axis, score in value.items()
    }


def _expected_calibrated_margins(
    raw_scores: Mapping[str, float],
    thresholds: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for axis, score in raw_scores.items():
        threshold = float(thresholds[axis]["threshold"])
        orientation = 1.0 if thresholds[axis]["higher_is_better"] else -1.0
        result[axis] = orientation * (score - threshold)
    return result


def _validate_margin_recomputation(
    label: str,
    declared: Mapping[str, float],
    recomputed: Mapping[str, float],
) -> None:
    if set(declared) != set(recomputed):
        raise DCLRCounterfactualBankError(
            f"{label} axes differ from content-addressed raw/calibration artifacts"
        )
    for axis in declared:
        if not math.isclose(
            float(declared[axis]),
            float(recomputed[axis]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise DCLRCounterfactualBankError(
                f"{label}.{axis} cannot be recomputed from raw score and threshold"
            )


def _validate_raw_reward_artifact(
    value: Any,
    *,
    source: Mapping[str, Any],
    rollout: Mapping[str, Any],
    native_provenance: Mapping[str, Any],
    bank_row: Mapping[str, Any],
    split_ledger: Mapping[str, Any],
    evaluator_digest: str,
    sigma_bank_digest: str,
    action_thresholds: Mapping[str, Mapping[str, Any]],
    preservation_thresholds: Mapping[str, Mapping[str, Any]],
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
    label: str,
) -> tuple[Mapping[str, Any], dict[str, float], dict[str, float]]:
    """Validate the immutable raw scores from which all receipt gates derive."""

    _reject_privileged_condition_keys(value)
    artifact = _require_exact_fields(
        label, value, _RAW_REWARD_ARTIFACT_FIELDS
    )
    if artifact["schema_version"] != RAW_REWARD_ARTIFACT_SCHEMA:
        raise DCLRCounterfactualBankError(f"{label} schema differs")
    _require_slug(f"{label}.evidence_id", artifact["evidence_id"])
    expected_fields = {
        "sample_id": source["sample_id"],
        "source_record_digest": source["record_digest"],
        "split_ledger_digest": split_ledger["ledger_digest"],
        "native_provenance_digest": native_provenance["provenance_digest"],
        "output_video_sha256": rollout["output_video_sha256"],
        "clean_latent_sha256": rollout["clean_latent_sha256"],
        "reward_version": rollout["reward_version"],
        "evaluator_artifact_digest": evaluator_digest,
        "sigma_bank_artifact_digest": sigma_bank_digest,
    }
    for field, expected in expected_fields.items():
        if artifact[field] != expected:
            raise DCLRCounterfactualBankError(
                f"{label}.{field} differs from its bound rollout context"
            )
    for field in (
        "source_record_digest",
        "split_ledger_digest",
        "native_provenance_digest",
        "output_video_sha256",
        "clean_latent_sha256",
        "evaluator_artifact_digest",
        "sigma_bank_artifact_digest",
    ):
        _require_sha256(f"{label}.{field}", artifact[field])

    alternatives = _require_string_list(
        f"{label}.evaluated_alternative_ids",
        artifact["evaluated_alternative_ids"],
        preserve_order=False,
    )
    if alternatives != tuple(rollout["evaluated_alternative_ids"]):
        raise DCLRCounterfactualBankError(
            f"{label} hard-alternative set differs from the rollout receipt"
        )
    decoys = _require_string_list(
        f"{label}.evaluated_wrong_source_decoy_ids",
        artifact["evaluated_wrong_source_decoy_ids"],
        preserve_order=False,
    )
    if decoys != tuple(rollout["evaluated_wrong_source_decoy_ids"]):
        raise DCLRCounterfactualBankError(
            f"{label} wrong-source set differs from the rollout receipt"
        )

    alternative_expected = {
        str(item["alternative_id"]): str(
            item["semantic_evidence"]["evidence_digest"]
        )
        for item in bank_row["hard_alternatives"]
    }
    _validate_digest_map(
        f"{label}.alternative_semantic_evidence_digests",
        artifact["alternative_semantic_evidence_digests"],
        expected_keys=set(alternative_expected),
        expected_values=alternative_expected,
    )
    decoy_ids = {
        str(item["decoy_id"]) for item in bank_row["wrong_source_decoys"]
    }
    wrong_source_artifacts = _validate_digest_map(
        f"{label}.wrong_source_evidence_sha256_by_decoy",
        artifact["wrong_source_evidence_sha256_by_decoy"],
        expected_keys=decoy_ids,
        require_unique_values=True,
    )
    for decoy_id, artifact_digest in wrong_source_artifacts.items():
        wrong_source_artifact = _artifact_from_registry(
            artifact_digest,
            artifacts_by_digest,
            expected_schema=CONTENT_ARTIFACT_SCHEMA,
            label=f"{label}.wrong_source_artifact[{decoy_id}]",
        )
        _validate_content_artifact(
            wrong_source_artifact,
            label=f"{label}.wrong_source_artifact[{decoy_id}]",
            expected_kind=f"wrong_source_reward_evidence.{decoy_id}",
        )

    action_scores = _validate_raw_score_map(
        f"{label}.action_axis_raw_scores",
        artifact["action_axis_raw_scores"],
        expected_axes=set(action_thresholds),
    )
    preservation_scores = _validate_raw_score_map(
        f"{label}.preservation_axis_raw_scores",
        artifact["preservation_axis_raw_scores"],
        expected_axes=set(preservation_thresholds),
    )
    _validate_embedded_digest(label, artifact, "artifact_digest")
    return artifact, action_scores, preservation_scores


def _validate_reward_evidence(
    value: Any,
    *,
    source: Mapping[str, Any],
    rollout: Mapping[str, Any],
    native_provenance: Mapping[str, Any],
    bank_row: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
    action_axis_pass: Mapping[str, bool],
    preservation_axis_pass: Mapping[str, bool],
    label: str,
) -> Mapping[str, Any]:
    evidence = _require_exact_fields(label, value, _REWARD_EVIDENCE_FIELDS)
    if evidence["schema_version"] != REWARD_EVIDENCE_SCHEMA:
        raise DCLRCounterfactualBankError(
            f"{label} has an unexpected reward-evidence schema"
        )
    evaluator_digest = _require_sha256(
        f"{label}.evaluator_sha256", evidence["evaluator_sha256"]
    )
    evaluator = _artifact_from_registry(
        evaluator_digest,
        artifacts_by_digest,
        expected_schema=EVALUATOR_ARTIFACT_SCHEMA,
        label=f"{label}.evaluator",
    )
    _validate_evaluator_artifact(evaluator, label=f"{label}.evaluator")

    sigma_bank_digest = _require_sha256(
        f"{label}.sigma_bank_sha256", evidence["sigma_bank_sha256"]
    )
    sigma_bank = _artifact_from_registry(
        sigma_bank_digest,
        artifacts_by_digest,
        expected_schema=SIGMA_BANK_ARTIFACT_SCHEMA,
        label=f"{label}.sigma_bank",
    )
    _validate_sigma_bank_artifact(sigma_bank, label=f"{label}.sigma_bank")

    calibration_digest = _require_sha256(
        f"{label}.threshold_calibration_sha256",
        evidence["threshold_calibration_sha256"],
    )
    calibration = _artifact_from_registry(
        calibration_digest,
        artifacts_by_digest,
        expected_schema=THRESHOLD_CALIBRATION_SCHEMA,
        label=f"{label}.threshold_calibration",
    )
    _, action_thresholds, preservation_thresholds = (
        _validate_calibration_artifact(
            calibration,
            sources=sources,
            split_ledger=split_ledger,
            source_manifest_sha256=source_manifest_sha256,
            evaluator_digest=evaluator_digest,
            sigma_bank_digest=sigma_bank_digest,
            label=f"{label}.threshold_calibration",
        )
    )
    if set(action_thresholds) != set(action_axis_pass):
        raise DCLRCounterfactualBankError(
            f"{label} action threshold axes differ from rollout hard gates"
        )
    if set(preservation_thresholds) != set(preservation_axis_pass):
        raise DCLRCounterfactualBankError(
            f"{label} preservation threshold axes differ from rollout hard gates"
        )

    raw_digest = _require_sha256(
        f"{label}.raw_reward_evidence_sha256",
        evidence["raw_reward_evidence_sha256"],
    )
    raw_artifact = _artifact_from_registry(
        raw_digest,
        artifacts_by_digest,
        expected_schema=RAW_REWARD_ARTIFACT_SCHEMA,
        label=f"{label}.raw_reward",
    )
    raw_artifact, action_raw_scores, preservation_raw_scores = (
        _validate_raw_reward_artifact(
            raw_artifact,
            source=source,
            rollout=rollout,
            native_provenance=native_provenance,
            bank_row=bank_row,
            split_ledger=split_ledger,
            evaluator_digest=evaluator_digest,
            sigma_bank_digest=sigma_bank_digest,
            action_thresholds=action_thresholds,
            preservation_thresholds=preservation_thresholds,
            artifacts_by_digest=artifacts_by_digest,
            label=f"{label}.raw_reward",
        )
    )

    alternative_expected = {
        str(item["alternative_id"]): str(
            item["semantic_evidence"]["evidence_digest"]
        )
        for item in bank_row["hard_alternatives"]
    }
    alternative_digests = _validate_digest_map(
        f"{label}.alternative_semantic_evidence_digests",
        evidence["alternative_semantic_evidence_digests"],
        expected_keys=set(alternative_expected),
        expected_values=alternative_expected,
    )
    if alternative_digests != raw_artifact[
        "alternative_semantic_evidence_digests"
    ]:
        raise DCLRCounterfactualBankError(
            f"{label} semantic-evidence map differs from raw reward artifact"
        )
    decoy_ids = {
        str(item["decoy_id"]) for item in bank_row["wrong_source_decoys"]
    }
    wrong_source_digests = _validate_digest_map(
        f"{label}.wrong_source_evidence_sha256_by_decoy",
        evidence["wrong_source_evidence_sha256_by_decoy"],
        expected_keys=decoy_ids,
        require_unique_values=True,
    )
    if wrong_source_digests != raw_artifact[
        "wrong_source_evidence_sha256_by_decoy"
    ]:
        raise DCLRCounterfactualBankError(
            f"{label} wrong-source evidence differs from raw reward artifact"
        )
    action_margins = _validate_calibrated_margin_map(
        f"{label}.action_axis_calibrated_margins",
        evidence["action_axis_calibrated_margins"],
        axis_pass=action_axis_pass,
    )
    preservation_margins = _validate_calibrated_margin_map(
        f"{label}.preservation_axis_calibrated_margins",
        evidence["preservation_axis_calibrated_margins"],
        axis_pass=preservation_axis_pass,
    )
    _validate_margin_recomputation(
        f"{label}.action_axis_calibrated_margins",
        action_margins,
        _expected_calibrated_margins(action_raw_scores, action_thresholds),
    )
    _validate_margin_recomputation(
        f"{label}.preservation_axis_calibrated_margins",
        preservation_margins,
        _expected_calibrated_margins(
            preservation_raw_scores, preservation_thresholds
        ),
    )
    _validate_embedded_digest(label, evidence, "evidence_digest")
    return evidence


def validate_rollout_receipt(
    receipt: Any,
    sources: Mapping[str, Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Validate one source-only rollout and its hard-gate truth table."""

    validated_sources = validate_source_action_records(sources.values())
    if set(validated_sources) != set(sources) or any(
        sources[key] is not validated_sources[key] for key in validated_sources
    ):
        raise DCLRCounterfactualBankError(
            "rollout source registry keys differ from embedded sample IDs"
        )
    validated_ledger = validate_full_split_ledger(
        split_ledger,
        validated_sources,
        expected_source_manifest_sha256=source_manifest_sha256,
    )
    validated_bank = validate_counterfactual_bank(
        bank,
        validated_sources,
        expected_source_manifest_sha256=source_manifest_sha256,
    )
    sources = validated_sources
    split_ledger = validated_ledger
    bank = validated_bank

    _reject_privileged_condition_keys(receipt)
    rollout = _require_exact_fields("rollout receipt", receipt, _ROLLOUT_FIELDS)
    if rollout["schema_version"] != ROLLOUT_RECEIPT_SCHEMA:
        raise DCLRCounterfactualBankError("unexpected rollout receipt schema")
    _require_slug("receipt_id", rollout["receipt_id"])
    sample_id = _require_slug("sample_id", rollout["sample_id"])
    if sample_id not in sources:
        raise DCLRCounterfactualBankError(
            f"rollout references unknown source sample: {sample_id}"
        )
    source = sources[sample_id]
    if rollout["source_record_digest"] != source["record_digest"]:
        raise DCLRCounterfactualBankError("rollout source record digest mismatch")
    _require_sha256("source_record_digest", rollout["source_record_digest"])
    if rollout["counterfactual_bank_digest"] != bank["bank_digest"]:
        raise DCLRCounterfactualBankError("rollout counterfactual-bank digest mismatch")
    _require_sha256(
        "counterfactual_bank_digest", rollout["counterfactual_bank_digest"]
    )
    if sample_id not in _bank_rows_by_sample(bank):
        raise DCLRCounterfactualBankError(
            "rollout source has no pre-registered counterfactual-bank row"
        )
    bank_row = _bank_rows_by_sample(bank)[sample_id]
    _require_slug("policy_id", rollout["policy_id"])
    _require_sha256("policy_sha256", rollout["policy_sha256"])
    _require_int("policy_revision", rollout["policy_revision"], minimum=0)
    _require_slug("arm", rollout["arm"])
    _require_slug("collection_episode_id", rollout["collection_episode_id"])
    candidate_set_size = _require_int(
        "candidate_set_size", rollout["candidate_set_size"], minimum=1
    )
    candidate_slot = _require_int(
        "candidate_slot", rollout["candidate_slot"], minimum=0
    )
    if candidate_slot >= candidate_set_size:
        raise DCLRCounterfactualBankError(
            "candidate_slot must lie inside candidate_set_size"
        )
    _require_int("candidate_seed", rollout["candidate_seed"], minimum=0)
    _require_sha256("output_video_sha256", rollout["output_video_sha256"])
    _require_sha256("clean_latent_sha256", rollout["clean_latent_sha256"])
    native_provenance_digest = _require_sha256(
        "native_provenance_digest", rollout["native_provenance_digest"]
    )
    _require_slug("reward_version", rollout["reward_version"])

    closure = _require_exact_fields(
        "condition_closure",
        rollout["condition_closure"],
        _CONDITION_CLOSURE_FIELDS,
    )
    external_inputs = _require_string_list(
        "condition_closure.external_inputs", closure["external_inputs"]
    )
    if (
        set(external_inputs) != {"source_video", "edit_instruction"}
        or len(external_inputs) != 2
    ):
        raise DCLRCounterfactualBankError(
            "rollout external inputs must be exactly source_video + edit_instruction"
        )
    if closure["privileged_inputs_accessed"] != []:
        raise DCLRCounterfactualBankError(
            "rollout must prove that no privileged input was accessed"
        )

    native_provenance = _artifact_from_registry(
        native_provenance_digest,
        artifacts_by_digest,
        expected_schema=NATIVE_PROVENANCE_SCHEMA,
        label="native rollout provenance",
    )
    native_provenance = _validate_native_provenance(
        native_provenance,
        source=source,
        rollout=rollout,
        label="native rollout provenance",
    )
    if tuple(external_inputs) != tuple(native_provenance["external_inputs"]):
        raise DCLRCounterfactualBankError(
            "condition closure differs from native rollout provenance"
        )

    evaluated_alternatives = _require_string_list(
        "evaluated_alternative_ids",
        rollout["evaluated_alternative_ids"],
        preserve_order=False,
    )
    registered_alternatives = tuple(
        sorted(
            str(item["alternative_id"])
            for item in bank_row["hard_alternatives"]
        )
    )
    if evaluated_alternatives != registered_alternatives:
        raise DCLRCounterfactualBankError(
            "rollout must evaluate every pre-registered hard alternative"
        )
    evaluated_decoys = _require_string_list(
        "evaluated_wrong_source_decoy_ids",
        rollout["evaluated_wrong_source_decoy_ids"],
        preserve_order=False,
    )
    registered_decoys = tuple(
        sorted(
            str(item["decoy_id"])
            for item in bank_row["wrong_source_decoys"]
        )
    )
    if evaluated_decoys != registered_decoys:
        raise DCLRCounterfactualBankError(
            "rollout must evaluate every pre-registered wrong-source decoy"
        )

    action_axes = _validate_axis_map("action_axis_pass", rollout["action_axis_pass"])
    preservation_axes = _validate_axis_map(
        "preservation_axis_pass", rollout["preservation_axis_pass"]
    )
    action_pass = _require_bool("action_pass", rollout["action_pass"])
    preservation_pass = _require_bool(
        "preservation_pass", rollout["preservation_pass"]
    )
    joint_pass = _require_bool("joint_pass", rollout["joint_pass"])
    if action_pass != all(action_axes.values()):
        raise DCLRCounterfactualBankError(
            "rollout action_pass is inconsistent with per-axis hard gates"
        )
    if preservation_pass != all(preservation_axes.values()):
        raise DCLRCounterfactualBankError(
            "rollout preservation_pass is inconsistent with per-axis hard gates"
        )
    if joint_pass != (action_pass and preservation_pass):
        raise DCLRCounterfactualBankError(
            "rollout joint_pass must be action_pass AND preservation_pass"
        )
    _validate_reward_evidence(
        rollout["reward_evidence"],
        source=source,
        rollout=rollout,
        native_provenance=native_provenance,
        bank_row=bank_row,
        sources=sources,
        split_ledger=split_ledger,
        source_manifest_sha256=source_manifest_sha256,
        artifacts_by_digest=artifacts_by_digest,
        action_axis_pass=action_axes,
        preservation_axis_pass=preservation_axes,
        label="reward_evidence",
    )
    _validate_embedded_digest("rollout receipt", rollout, "receipt_digest")
    return rollout


def validate_rollout_receipts(
    receipts: Iterable[Any],
    sources: Mapping[str, Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Validate receipts and index them by embedded receipt digest."""

    indexed: dict[str, Mapping[str, Any]] = {}
    receipt_ids: set[str] = set()
    candidate_keys: set[tuple[str, str, int, str, int]] = set()
    episodes: dict[
        tuple[str, str, str, int, str, str],
        dict[str, Any],
    ] = {}
    for index, value in enumerate(receipts):
        try:
            receipt = validate_rollout_receipt(
                value,
                sources,
                bank,
                split_ledger=split_ledger,
                source_manifest_sha256=source_manifest_sha256,
                artifacts_by_digest=artifacts_by_digest,
            )
        except DCLRCounterfactualBankError as error:
            raise DCLRCounterfactualBankError(
                f"rollout receipt {index} is invalid: {error}"
            ) from error
        digest = str(receipt["receipt_digest"])
        receipt_id = str(receipt["receipt_id"])
        candidate_key = (
            str(receipt["sample_id"]),
            str(receipt["policy_id"]),
            int(receipt["policy_revision"]),
            str(receipt["arm"]),
            int(receipt["candidate_seed"]),
        )
        if digest in indexed or receipt_id in receipt_ids:
            raise DCLRCounterfactualBankError("duplicate rollout receipt")
        if candidate_key in candidate_keys:
            raise DCLRCounterfactualBankError(
                "duplicate source/policy-revision/candidate-seed rollout"
            )
        indexed[digest] = receipt
        receipt_ids.add(receipt_id)
        candidate_keys.add(candidate_key)
        episode_key = (
            str(receipt["sample_id"]),
            str(receipt["policy_id"]),
            str(receipt["policy_sha256"]),
            int(receipt["policy_revision"]),
            str(receipt["arm"]),
            str(receipt["collection_episode_id"]),
        )
        episode = episodes.setdefault(
            episode_key,
            {
                "candidate_set_size": int(receipt["candidate_set_size"]),
                "slots": set(),
                "seeds": set(),
            },
        )
        if episode["candidate_set_size"] != int(receipt["candidate_set_size"]):
            raise DCLRCounterfactualBankError(
                "collection episode changes candidate_set_size"
            )
        slot = int(receipt["candidate_slot"])
        seed = int(receipt["candidate_seed"])
        if slot in episode["slots"] or seed in episode["seeds"]:
            raise DCLRCounterfactualBankError(
                "collection episode repeats a candidate slot or seed"
            )
        episode["slots"].add(slot)
        episode["seeds"].add(seed)
    if not indexed:
        raise DCLRCounterfactualBankError("rollout receipt set is empty")
    for episode in episodes.values():
        size = int(episode["candidate_set_size"])
        if episode["slots"] != set(range(size)) or len(episode["seeds"]) != size:
            raise DCLRCounterfactualBankError(
                "collection episode is incomplete for its candidate_set_size"
            )
    return indexed


def _validate_preference_context(
    sources: Mapping[str, Mapping[str, Any]],
    bank: Mapping[str, Any],
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
) -> tuple[
    dict[str, Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]
]:
    validated_sources = validate_source_action_records(sources.values())
    if set(validated_sources) != set(sources) or any(
        sources[key] is not validated_sources[key] for key in validated_sources
    ):
        raise DCLRCounterfactualBankError(
            "preference source registry keys differ from embedded sample IDs"
        )
    validated_ledger = validate_full_split_ledger(
        split_ledger,
        validated_sources,
        expected_source_manifest_sha256=source_manifest_sha256,
    )
    validated_bank = validate_counterfactual_bank(
        bank,
        validated_sources,
        expected_source_manifest_sha256=source_manifest_sha256,
    )
    return validated_sources, validated_bank, validated_ledger


def _validate_preference_pair_against_validated_context(
    pair: Any,
    receipts_by_digest: Mapping[str, Mapping[str, Any]],
    validated_sources: Mapping[str, Mapping[str, Any]],
    validated_bank: Mapping[str, Any],
    validated_split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    _reject_privileged_condition_keys(pair)
    preference = _require_exact_fields("preference pair", pair, _PAIR_FIELDS)
    if preference["schema_version"] != PREFERENCE_PAIR_SCHEMA:
        raise DCLRCounterfactualBankError("unexpected preference pair schema")
    _require_slug("pair_id", preference["pair_id"])
    winner_digest = _require_sha256(
        "winner_receipt_digest", preference["winner_receipt_digest"]
    )
    loser_digest = _require_sha256(
        "loser_receipt_digest", preference["loser_receipt_digest"]
    )
    if winner_digest == loser_digest:
        raise DCLRCounterfactualBankError("winner and loser receipts must differ")
    try:
        raw_winner = receipts_by_digest[winner_digest]
        raw_loser = receipts_by_digest[loser_digest]
    except KeyError as error:
        raise DCLRCounterfactualBankError(
            "preference pair references an unknown rollout receipt"
        ) from error
    winner = validate_rollout_receipt(
        raw_winner,
        validated_sources,
        validated_bank,
        split_ledger=validated_split_ledger,
        source_manifest_sha256=source_manifest_sha256,
        artifacts_by_digest=artifacts_by_digest,
    )
    loser = validate_rollout_receipt(
        raw_loser,
        validated_sources,
        validated_bank,
        split_ledger=validated_split_ledger,
        source_manifest_sha256=source_manifest_sha256,
        artifacts_by_digest=artifacts_by_digest,
    )
    if (
        winner["receipt_digest"] != winner_digest
        or loser["receipt_digest"] != loser_digest
    ):
        raise DCLRCounterfactualBankError(
            "preference receipt registry key differs from embedded digest"
        )

    same_fields = (
        "sample_id",
        "source_record_digest",
        "counterfactual_bank_digest",
        "policy_id",
        "policy_sha256",
        "policy_revision",
        "arm",
        "reward_version",
    )
    for field in same_fields:
        if winner[field] != loser[field]:
            raise DCLRCounterfactualBankError(
                f"preference pair winner/loser differ in {field}"
            )
    source = validated_sources[str(winner["sample_id"])]
    if source["split"] != "train":
        raise DCLRCounterfactualBankError(
            "preference training accepts train split only; "
            f"got {source['split']}"
        )
    if winner["collection_episode_id"] != loser["collection_episode_id"]:
        raise DCLRCounterfactualBankError(
            "preference pair candidates must come from one collection episode"
        )
    if (
        winner["candidate_set_size"] != 2
        or loser["candidate_set_size"] != 2
        or {winner["candidate_slot"], loser["candidate_slot"]} != {0, 1}
    ):
        raise DCLRCounterfactualBankError(
            "preference training requires the complete exact-K2 candidate set"
        )
    if winner["candidate_seed"] == loser["candidate_seed"]:
        raise DCLRCounterfactualBankError(
            "preference pair candidates must use distinct seeds"
        )
    if (
        winner["output_video_sha256"] == loser["output_video_sha256"]
        or winner["clean_latent_sha256"] == loser["clean_latent_sha256"]
    ):
        raise DCLRCounterfactualBankError(
            "preference pair candidates must have distinct media and latents"
        )
    winner_evidence = winner["reward_evidence"]
    loser_evidence = loser["reward_evidence"]
    for field in (
        "threshold_calibration_sha256",
        "evaluator_sha256",
        "sigma_bank_sha256",
        "alternative_semantic_evidence_digests",
    ):
        if winner_evidence[field] != loser_evidence[field]:
            raise DCLRCounterfactualBankError(
                f"preference candidates differ in reward evidence {field}"
            )
    if (
        winner_evidence["raw_reward_evidence_sha256"]
        == loser_evidence["raw_reward_evidence_sha256"]
    ):
        raise DCLRCounterfactualBankError(
            "preference candidates must have distinct raw reward evidence"
        )
    if winner["joint_pass"] is not True:
        raise DCLRCounterfactualBankError(
            "preference winner must pass the strict joint gate"
        )

    pair_type = _require_slug("pair_type", preference["pair_type"])
    if pair_type not in _ALLOWED_PAIR_TYPES:
        raise DCLRCounterfactualBankError(f"unsupported pair_type: {pair_type}")
    loser_action_axes = loser["action_axis_pass"]
    loser_preservation_axes = loser["preservation_axis_pass"]
    if pair_type == "action_nearmiss":
        if loser["action_pass"] is not False or loser["preservation_pass"] is not True:
            raise DCLRCounterfactualBankError(
                "action_nearmiss loser must fail action only"
            )
        if sum(not passed for passed in loser_action_axes.values()) != 1:
            raise DCLRCounterfactualBankError(
                "action_nearmiss loser must fail exactly one action axis"
            )
    else:
        if loser["action_pass"] is not True or loser["preservation_pass"] is not False:
            raise DCLRCounterfactualBankError(
                "preservation_nearmiss loser must fail preservation only"
            )
        if sum(not passed for passed in loser_preservation_axes.values()) != 1:
            raise DCLRCounterfactualBankError(
                "preservation_nearmiss loser must fail exactly one preservation axis"
            )
    if loser["joint_pass"] is not False:
        raise DCLRCounterfactualBankError("preference loser cannot be joint-pass")

    collection_revision = _require_int(
        "collection_policy_revision",
        preference["collection_policy_revision"],
        minimum=0,
    )
    training_revision = _require_int(
        "training_policy_revision",
        preference["training_policy_revision"],
        minimum=0,
    )
    if collection_revision != winner["policy_revision"]:
        raise DCLRCounterfactualBankError(
            "pair collection revision differs from rollout policy revision"
        )
    policy_lag = training_revision - collection_revision
    if policy_lag < 0 or policy_lag > 1:
        raise DCLRCounterfactualBankError(
            "preference buffer policy lag must remain in {0, 1}"
        )
    _validate_embedded_digest("preference pair", preference, "pair_digest")
    return preference


def validate_preference_pair(
    pair: Any,
    receipts_by_digest: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Validate one train-only, exact-K2, closed-evidence preference pair.

    This public entry point deliberately revalidates source, bank, and complete
    rollout receipts.  It is safe to call without first trusting a separately
    constructed receipt registry.
    """

    validated_sources, validated_bank, validated_split_ledger = (
        _validate_preference_context(
            sources, bank, split_ledger, source_manifest_sha256
        )
    )
    return _validate_preference_pair_against_validated_context(
        pair,
        receipts_by_digest,
        validated_sources,
        validated_bank,
        validated_split_ledger,
        source_manifest_sha256,
        artifacts_by_digest,
    )


def validate_preference_pairs(
    pairs: Iterable[Any],
    receipts_by_digest: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Validate preference pairs and index them by pair digest."""

    indexed: dict[str, Mapping[str, Any]] = {}
    pair_ids: set[str] = set()
    used_receipt_pairs: set[tuple[str, str]] = set()
    used_receipts: set[str] = set()
    used_episodes: set[tuple[str, str, int, str, str]] = set()
    training_revisions: set[int] = set()
    validated_sources, validated_bank, validated_split_ledger = (
        _validate_preference_context(
            sources, bank, split_ledger, source_manifest_sha256
        )
    )
    for index, value in enumerate(pairs):
        try:
            pair = _validate_preference_pair_against_validated_context(
                value,
                receipts_by_digest,
                validated_sources,
                validated_bank,
                validated_split_ledger,
                source_manifest_sha256,
                artifacts_by_digest,
            )
        except DCLRCounterfactualBankError as error:
            raise DCLRCounterfactualBankError(
                f"preference pair {index} is invalid: {error}"
            ) from error
        digest = str(pair["pair_digest"])
        pair_id = str(pair["pair_id"])
        receipt_pair = (
            str(pair["winner_receipt_digest"]),
            str(pair["loser_receipt_digest"]),
        )
        winner = receipts_by_digest[receipt_pair[0]]
        episode_key = (
            str(winner["sample_id"]),
            str(winner["policy_id"]),
            int(winner["policy_revision"]),
            str(winner["arm"]),
            str(winner["collection_episode_id"]),
        )
        if digest in indexed or pair_id in pair_ids:
            raise DCLRCounterfactualBankError("duplicate preference pair")
        if receipt_pair in used_receipt_pairs:
            raise DCLRCounterfactualBankError(
                "duplicate winner/loser receipt pairing"
            )
        if set(receipt_pair).intersection(used_receipts):
            raise DCLRCounterfactualBankError(
                "one exact-K2 rollout receipt cannot be reused across training pairs"
            )
        if episode_key in used_episodes:
            raise DCLRCounterfactualBankError(
                "one exact-K2 collection episode can produce at most one pair"
            )
        indexed[digest] = pair
        pair_ids.add(pair_id)
        used_receipt_pairs.add(receipt_pair)
        used_receipts.update(receipt_pair)
        used_episodes.add(episode_key)
        training_revisions.add(int(pair["training_policy_revision"]))
    if not indexed:
        raise DCLRCounterfactualBankError("preference pair set is empty")
    if len(training_revisions) != 1:
        raise DCLRCounterfactualBankError(
            "one preference file must target exactly one training policy revision"
        )
    return indexed


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DCLRCounterfactualBankError(
                f"JSON object contains duplicate key: {key}"
            )
        result[key] = value
    return result


def _parse_json(text: str, *, label: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except DCLRCounterfactualBankError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise DCLRCounterfactualBankError(f"invalid {label}: {error}") from error


def _read_hash_bound_text(
    path: str | os.PathLike[str], expected_sha256: str
) -> str:
    expected = _require_sha256("expected file SHA-256", expected_sha256)
    observed = file_sha256(path)
    if observed != expected:
        raise DCLRCounterfactualBankError(
            f"hash-bound file digest mismatch: {observed} != {expected}"
        )
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise DCLRCounterfactualBankError(
            f"cannot read UTF-8 hash-bound file {path}: {error}"
        ) from error


def load_hash_bound_json(
    path: str | os.PathLike[str], expected_sha256: str
) -> Any:
    """Load one caller-hash-pinned UTF-8 JSON document."""

    return _parse_json(
        _read_hash_bound_text(path, expected_sha256), label="JSON document"
    )


def load_hash_bound_jsonl(
    path: str | os.PathLike[str], expected_sha256: str
) -> list[Any]:
    """Load nonempty caller-hash-pinned UTF-8 JSONL without blank rows."""

    text = _read_hash_bound_text(path, expected_sha256)
    if not text:
        raise DCLRCounterfactualBankError("JSONL file is empty")
    records: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise DCLRCounterfactualBankError(
                f"JSONL contains a blank row at line {line_number}"
            )
        records.append(_parse_json(line, label=f"JSONL line {line_number}"))
    if not records:
        raise DCLRCounterfactualBankError("JSONL file has no records")
    return records


def load_source_action_jsonl(
    path: str | os.PathLike[str], expected_sha256: str
) -> dict[str, Mapping[str, Any]]:
    return validate_source_action_records(
        load_hash_bound_jsonl(path, expected_sha256)
    )


def load_counterfactual_bank_json(
    path: str | os.PathLike[str],
    expected_sha256: str,
    sources: Mapping[str, Mapping[str, Any]],
    *,
    source_manifest_sha256: str,
) -> Mapping[str, Any]:
    return validate_counterfactual_bank(
        load_hash_bound_json(path, expected_sha256),
        sources,
        expected_source_manifest_sha256=source_manifest_sha256,
    )


def load_full_split_ledger_json(
    path: str | os.PathLike[str],
    expected_sha256: str,
    sources: Mapping[str, Mapping[str, Any]],
    *,
    source_manifest_sha256: str,
) -> Mapping[str, Any]:
    return validate_full_split_ledger(
        load_hash_bound_json(path, expected_sha256),
        sources,
        expected_source_manifest_sha256=source_manifest_sha256,
    )


def load_rollout_receipts_jsonl(
    path: str | os.PathLike[str],
    expected_sha256: str,
    sources: Mapping[str, Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return validate_rollout_receipts(
        load_hash_bound_jsonl(path, expected_sha256),
        sources,
        bank,
        split_ledger=split_ledger,
        source_manifest_sha256=source_manifest_sha256,
        artifacts_by_digest=artifacts_by_digest,
    )


def load_preference_pairs_jsonl(
    path: str | os.PathLike[str],
    expected_sha256: str,
    receipts_by_digest: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    split_ledger: Mapping[str, Any],
    source_manifest_sha256: str,
    artifacts_by_digest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return validate_preference_pairs(
        load_hash_bound_jsonl(path, expected_sha256),
        receipts_by_digest,
        sources,
        bank,
        split_ledger=split_ledger,
        source_manifest_sha256=source_manifest_sha256,
        artifacts_by_digest=artifacts_by_digest,
    )


__all__ = [
    "ALTERNATIVE_SEMANTIC_EVIDENCE_SCHEMA",
    "COUNTERFACTUAL_BANK_SCHEMA",
    "CHECKPOINT_CONTENT_SCHEMA",
    "CONTENT_ARTIFACT_SCHEMA",
    "DCLRCounterfactualBankError",
    "EVALUATOR_ARTIFACT_SCHEMA",
    "EXPECTED_FPS",
    "EXPECTED_FRAME_COUNT",
    "MIN_REWARD_CAL_SAMPLES",
    "MIN_WRONG_SOURCE_DECOYS",
    "PREFERENCE_PAIR_SCHEMA",
    "NATIVE_PROVENANCE_SCHEMA",
    "RAW_REWARD_ARTIFACT_SCHEMA",
    "ROLLOUT_RECEIPT_SCHEMA",
    "REWARD_EVIDENCE_SCHEMA",
    "SOURCE_ACTION_SCHEMA",
    "SIGMA_BANK_ARTIFACT_SCHEMA",
    "SPLIT_LEDGER_SCHEMA",
    "THRESHOLD_CALIBRATION_SCHEMA",
    "canonical_object_sha256",
    "embedded_object_sha256",
    "file_sha256",
    "load_counterfactual_bank_json",
    "load_full_split_ledger_json",
    "load_hash_bound_json",
    "load_hash_bound_jsonl",
    "load_preference_pairs_jsonl",
    "load_rollout_receipts_jsonl",
    "load_source_action_jsonl",
    "validate_counterfactual_bank",
    "validate_full_split_ledger",
    "validate_preference_pair",
    "validate_preference_pairs",
    "validate_rollout_receipt",
    "validate_rollout_receipts",
    "validate_source_action_record",
    "validate_source_action_records",
]
