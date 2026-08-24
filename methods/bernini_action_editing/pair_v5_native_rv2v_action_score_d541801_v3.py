#!/usr/bin/env python3
"""Source-bound native-RV2V action scoring for PAIR-v5.

This module is the authority boundary between a sealed population of native
Bernini RV2V-4 rollouts and a later identity/action safe-Pareto evaluator.  It
does **not** infer action success.  For every action-only rollout it:

* independently revalidates the d541801 frozen-T2V v3 scalar calibration graph
  and all forty formal score receipts in an isolated pinned source tree;
* matches the rollout to exactly one sealed calibration cell/action family;
* rebuilds that cell's action-plus-nine prompts with Bernini's official T2V
  task prompt builder;
* evaluates global MACE at native exact40 index 33 (physical sigma
  0.5161304473876953, discrete model timestep 516) on the rollout's own native
  predecode clean latent and its own captured official sampler Gaussian; and
* applies only the clipped-affine map sealed for that action family.

The generated T2V media, generated RV2V MP4, source condition latent, labels,
targets, masks, flow, pose, tracks, and trajectories are never tensor inputs to
the scorer.  Source and media paths are nevertheless hash-bound provenance.
The emitted safe-Pareto record is an action-metric input only; it explicitly
cannot select a candidate without separate source-identity, camera, and
temporal-consistency evidence.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native_generation  # noqa: E402
import mace_candidate_action_energy as mace  # noqa: E402
import pair_v5_native_bridge as native_bridge  # noqa: E402
import pair_v5_native_rollout_spec as rollout_contract  # noqa: E402
import pair_v5_phase_conjunctive_energy as phase_energy  # noqa: E402
import pair_v5_t2v_calibration_bank_spec as t2v_bank_contract  # noqa: E402
import pair_v5_t2v_energy_calibration_v3 as calibration_v3  # noqa: E402
import pair_v5_t2v_score_d541801_v3_compat as formal_v3_compat  # noqa: E402
import score_pair_v5_t2v_energy_bank_v3 as frozen_mace_runtime  # noqa: E402
import validate_pair_v5_t2v_calibration_d541801_v3 as mainline_calibration  # noqa: E402


SCORE_SCHEMA = "bernini-pair-v5-native-rv2v-global-action-score-d541801-v3"
GROUP_SCHEMA = "bernini-pair-v5-native-rv2v-global-action-score-group-d541801-v3"
SAFE_PARETO_ACTION_SCHEMA = "bernini-pair-v5-safe-pareto-action-input-d541801-v3"
SCORE_FILENAME = "pair-v5-native-rv2v-action-score-d541801-v3.json"
GROUP_FILENAME = "pair-v5-native-rv2v-action-score-{group_id}-d541801-v3.json"
SAFE_PARETO_ACTION_FILENAME = "pair-v5-safe-pareto-action-input-d541801-v3.json"

PILOT_SIGMA = frozen_mace_runtime.PILOT_SIGMA
PILOT_NATIVE_TIMESTEP = frozen_mace_runtime.PILOT_NATIVE_SCHEDULER_TIMESTEP
PILOT_SCHEDULE_INDEX = frozen_mace_runtime.PILOT_SCHEDULE_INDEX

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

NATIVE_SCORE_INPUT_CLOSURE = {
    "accepted_tensor_inputs": [
        "candidate_own_native_rv2v_predecode_clean_latent",
        "same_candidate_official_native_sampler_gaussian",
    ],
    "accepted_semantic_inputs": [
        "sealed_same_cell_action_plus_nine_t2v_prompt_bank"
    ],
    "source_video_tensor_consumed_by_scorer": False,
    "source_video_hash_bound_as_provenance": True,
    "rv2v_generated_mp4_consumed_by_scorer": False,
    "rv2v_generated_mp4_hash_bound_as_provenance": True,
    "rv2v_source_condition_latent_consumed_by_scorer": False,
    "t2v_calibration_media_consumed_by_scorer": False,
    "t2v_proposal_as_target_donor_input_or_noise": False,
    "formal_action_scalar_definition": formal_v3_compat.V3_SCALAR_DEFINITION,
    "formal_v3_scalar_recomputed_from_branch_energies": True,
    "active_repository_action_scalar_consumed": False,
    "decimal_or_log1p_action_scalar_consumed": False,
    "formal_v3_compatibility_scalar_consumed": True,
    "cross_device_numeric_tolerance_used_for_formal_gate": False,
    "paired_target_video_or_latent": False,
    "mask_flow_pose_track_trajectory": False,
    "event_audit_label_consumed_by_model": False,
    "training_performed": False,
    "optimizer_step_performed": False,
}

_PAIR_ROLLOUT_FIELDS = frozenset(
    {
        "schema_version",
        "root_spec_raw_sha256",
        "candidate_envelope_sha256",
        "group_id",
        "visible_gpus",
        "runtime_topology",
        "ordinal",
        "candidate",
        "sampling_contract",
        "semantic_input_closure",
        "native_receipt_path",
        "native_receipt_sha256",
        "native_receipt_digest",
        "artifacts",
        "receipt_digest",
    }
)

_CALIBRATION_MAP_FIELDS = frozenset(
    {
        "kind",
        "score_field",
        "lower_raw_anchor",
        "upper_raw_anchor",
        "clip_min",
        "clip_max",
        "fit_positive_count",
        "fit_negative_count",
        "anchor_source_split",
        "mapping_digest",
    }
)

_CALIBRATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "calibrator_id",
        "preregistration_digest",
        "source_bank_spec_sha256",
        "source_bank_receipt_digest",
        "score_field",
        "phase_conjunctive_score_used_for_calibration",
        "phase_conjunctive_role",
        "frame_count",
        "action_family_order",
        "branch_order",
        "fit_row_count",
        "confirmation_row_count",
        "fit_row_set_digest",
        "confirmation_row_set_digest",
        "event_audit_receipt_set_digest",
        "frozen_scorer_receipt_set_digest",
        "coverage_counts",
        "mapping_by_family",
        "confirmation_metrics",
        "raw_score_evidence_by_family",
        "decision_threshold",
        "confirmation_thresholds",
        "gates",
        "fit_event_qualified_action_candidate_ids",
        "confirmation_rows_consumed_by_optimizer",
        "t2v_media_consumed_by_calibrator",
        "t2v_media_as_rv2v_target_donor_input_or_noise",
        "optimizer_authorized",
        "failure_reasons",
        "scientific_action_editing_claim",
        "receipt_digest",
    }
)

_SCORE_FIELDS = frozenset(
    {
        "schema_version",
        "candidate",
        "population",
        "source",
        "rollout",
        "calibration",
        "prompts",
        "artifacts",
        "frozen_runtime",
        "score_coordinate",
        "mace",
        "input_closure",
        "optimizer_authorized",
        "scientific_action_editing_claim",
        "receipt_digest",
    }
)

_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "group_id",
        "visible_gpus",
        "ordinal",
        "seed",
        "guidance",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "calibration_action_candidate_id",
        "source_video_sha256",
        "complete_caption_utf8_sha256",
    }
)

_POPULATION_FIELDS = frozenset(
    {
        "spec_path",
        "spec_raw_sha256",
        "candidate_count",
        "global_candidate_order",
        "global_candidate_order_digest",
        "group_candidate_order",
        "group_candidate_order_digest",
        "population_family_order",
        "population_split_order",
        "all_candidates_match_calibration_action_branch",
        "negative_branch_candidates_present",
    }
)

_SOURCE_FIELDS = frozenset(
    {
        "source_video_path",
        "source_video_sha256_declared",
        "source_video_sha256_recomputed",
        "calibration_geometry_source_path",
        "calibration_geometry_source_sha256",
        "source_path_and_hash_match_calibration_action_cell",
        "source_tensor_consumed_by_scorer",
    }
)

_ROLLOUT_FIELDS = frozenset(
    {
        "rollout_root",
        "pair_receipt_path",
        "pair_receipt_file_sha256",
        "pair_receipt_digest",
        "native_receipt_path",
        "native_receipt_file_sha256",
        "native_receipt_digest",
        "candidate_envelope_sha256",
        "runtime_topology",
        "native_arm",
        "native_condition_mode",
        "checkpoint_tree_sha256",
        "source_condition_identity_digest",
        "source_condition_artifact_sha256",
        "generated_mp4_sha256",
        "generated_mp4_consumed_by_scorer",
    }
)

_CALIBRATION_FIELDS = frozenset(
    {
        "mainline_authorization_digest",
        "formal_score_provenance_set_digest",
        "formal_score_schema",
        "formal_score_scalar_definition",
        "formal_v3_source_revision",
        "formal_v3_source_archive_sha256",
        "formal_v3_source_binding_digest",
        "family_mapping_set_digest",
        "calibration_root",
        "t2v_score_root",
        "preregistration_path",
        "preregistration_file_sha256",
        "preregistration_digest",
        "calibration_receipt_path",
        "calibration_receipt_file_sha256",
        "calibration_receipt_digest",
        "source_t2v_bank_spec_path",
        "source_t2v_bank_spec_sha256",
        "source_t2v_bank_receipt_path",
        "source_t2v_bank_receipt_file_sha256",
        "source_t2v_bank_receipt_digest",
        "action_family_id",
        "family_mapping",
        "family_mapping_digest",
        "decision_threshold",
        "calibration_maps_authorized",
        "native_rv2v_optimizer_authorized",
        "t2v_media_consumed_by_native_scorer",
    }
)

_PROMPT_FIELDS = frozenset(
    {
        "branch_order",
        "full_t2v_caption_by_branch",
        "full_t2v_caption_utf8_sha256_by_branch",
        "prompt_by_branch",
        "prompt_utf8_sha256_by_branch",
        "prompt_builder_contract",
        "prompt_registry_digest",
        "calibration_group_id",
    }
)

_PROMPT_BUILDER_FIELDS = frozenset(
    {
        "builder",
        "arm",
        "training_task_name",
        "prompt_cleaner",
        "system_prompt_utf8_sha256",
        "task_binding_clause_utf8_sha256",
        "builder_source_utf8_sha256",
        "prompt_cleaner_source_utf8_sha256",
        "contract_digest",
    }
)

_FROZEN_PACKET_FIELDS = frozenset(
    {
        "packet_receipt_digest",
        "prompt_registry_digest",
        "frozen_model_receipt_digest",
        "candidate_shape",
        "sigma_float32_bits_hex",
        "timestep_float32_bits_hex",
        "native_schedule_digest",
        "native_schedule_index",
        "native_scheduler_timestep",
        "timestep_mapping",
        "physical_sigma_and_model_timestep_share_native_exact40_index",
        "legacy_1000_sigma_timestep_rejected",
        "binding_digest",
    }
)

_ARTIFACT_FIELDS = frozenset(
    {
        "clean_latent_path",
        "clean_latent_artifact_sha256",
        "clean_latent_tensor_sha256",
        "official_gaussian_path",
        "official_gaussian_artifact_sha256",
        "official_gaussian_raw_value_sha256",
        "official_gaussian_content_sha256",
        "official_gaussian_tensor_sha256",
        "official_gaussian_generator_seed",
        "sigma_tensor_sha256",
        "candidate_own_x_sigma_tensor_sha256",
        "tensor_shape",
        "x_sigma_construction",
        "clean_and_gaussian_are_same_candidate_artifacts",
    }
)

_FROZEN_RUNTIME_FIELDS = frozenset(
    {
        "checkpoint_content_identity",
        "frozen_checkpoint_receipt_digest",
        "checkpoint_content_binding",
        "frozen_scorer_receipt_digest",
        "frozen_t2v_packet_binding",
        "all_loaded_parameters_frozen_before_and_after",
    }
)

_MACE_FIELDS = frozenset(
    {
        "definition",
        "energy_epsilon",
        "global_action_energy",
        "global_hard_negative_energy_by_branch",
        "global_negative_log_energy_ratio_by_branch",
        "global_hardest_negative_branch",
        "formal_v3_energy_packet",
        "formal_v3_energy_packet_digest",
        "formal_v3_source_revision",
        "raw_global_action_energy_score",
        "calibrated_family_action_score",
        "decision_threshold",
        "passes_calibrated_action_metric",
        "phase_conjunctive_score_diagnostic",
        "phase_diagnostic_receipt_digest",
        "phase_diagnostic_used_for_action_metric",
    }
)

_SAFE_PARETO_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibrated_action_score",
        "action_decision_threshold",
        "passes_calibrated_action_metric",
        "native_score_receipt_digest",
        "population_spec_raw_sha256",
        "source_video_sha256",
        "rollout_receipt_digest",
        "calibration_receipt_digest",
        "prompt_registry_digest",
        "candidate_own_x_sigma_tensor_sha256",
        "discovery_selection_candidate",
        "confirmation_metric_only",
        "requires_source_identity_metric",
        "requires_camera_metric",
        "requires_temporal_consistency_metric",
        "standalone_candidate_selection_authorized",
        "optimizer_authorized",
        "action_editing_success_inferred",
        "record_digest",
    }
)


class PairV5NativeRV2VActionScoreError(RuntimeError):
    """A native rollout, calibration, frozen score, or binding failed closed."""


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
        raise PairV5NativeRV2VActionScoreError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PairV5NativeRV2VActionScoreError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise PairV5NativeRV2VActionScoreError(
            f"{label} fields differ: missing={sorted(set(fields)-actual)}, "
            f"extra={sorted(actual-set(fields))}"
        )
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5NativeRV2VActionScoreError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise PairV5NativeRV2VActionScoreError(f"{label} must be a canonical safe ID")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PairV5NativeRV2VActionScoreError(f"{label} must be a finite JSON float")
    return value


def _plain_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PairV5NativeRV2VActionScoreError(f"{label} path differs")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV5NativeRV2VActionScoreError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_directory(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PairV5NativeRV2VActionScoreError(f"{label} path differs")
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise PairV5NativeRV2VActionScoreError(
            f"{label} must be an absolute plain directory"
        )
    return path.resolve(strict=True)


def _reject_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PairV5NativeRV2VActionScoreError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_file(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise PairV5NativeRV2VActionScoreError(f"{label} contains {token}")

    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5NativeRV2VActionScoreError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise PairV5NativeRV2VActionScoreError(f"{label} root must be an object")
    return value


def _read_bound_json(
    value: Any, expected_sha256: Any, *, label: str
) -> tuple[dict[str, Any], Path]:
    path = _plain_file(value, label=label)
    if file_sha256(path) != _sha256(expected_sha256, label=f"{label} SHA-256"):
        raise PairV5NativeRV2VActionScoreError(f"{label} file SHA-256 differs")
    return _read_json_file(path, label=label), path


def _verify_embedded(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    unsigned = dict(value)
    digest = _sha256(unsigned.pop(field, None), label=f"{label} {field}")
    if object_sha256(unsigned) != digest:
        raise PairV5NativeRV2VActionScoreError(f"{label} embedded digest differs")
    return digest


def _verify_embedded_with_canonicalizer(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
    canonicalizer: Any,
) -> str:
    unsigned = dict(value)
    digest = _sha256(unsigned.pop(field, None), label=f"{label} {field}")
    if hashlib.sha256(canonicalizer(unsigned)).hexdigest() != digest:
        raise PairV5NativeRV2VActionScoreError(f"{label} embedded digest differs")
    return digest


def validate_family_mapping(value: Any) -> dict[str, Any]:
    """Validate the exact v3 fit-only clipped-affine family map."""

    row = dict(_closed(value, _CALIBRATION_MAP_FIELDS, label="family mapping"))
    unsigned = dict(row)
    digest = _sha256(unsigned.pop("mapping_digest"), label="family mapping digest")
    if calibration_v3.object_sha256(unsigned) != digest:
        raise PairV5NativeRV2VActionScoreError("family mapping digest differs")
    lower = _finite_float(row["lower_raw_anchor"], label="lower raw anchor")
    upper = _finite_float(row["upper_raw_anchor"], label="upper raw anchor")
    if (
        row["kind"] != "clipped_affine_fit_only"
        or row["score_field"] != "raw_global_action_energy_score"
        or not upper > lower
        or row["clip_min"] != 0.0
        or row["clip_max"] != 1.0
        or type(row["fit_positive_count"]) is not int
        or row["fit_positive_count"] < 1
        or type(row["fit_negative_count"]) is not int
        or row["fit_negative_count"] < 1
        or row["anchor_source_split"] != "fit"
    ):
        raise PairV5NativeRV2VActionScoreError(
            "family mapping is not the sealed v3 fit-only clipped affine map"
        )
    return row


def apply_family_mapping(raw_score: Any, mapping: Any) -> float:
    """Apply only the exact map used by v3 fit/confirmation calibration."""

    raw = _finite_float(raw_score, label="raw global action energy score")
    checked = validate_family_mapping(mapping)
    value = (raw - checked["lower_raw_anchor"]) / (
        checked["upper_raw_anchor"] - checked["lower_raw_anchor"]
    )
    return float(min(1.0, max(0.0, value)))


def validate_calibration_receipt(value: Any) -> dict[str, Any]:
    row = dict(
        _closed(value, _CALIBRATION_RECEIPT_FIELDS, label="v3 calibration receipt")
    )
    digest = _verify_embedded(row, field="receipt_digest", label="v3 calibration receipt")
    if (
        row["schema_version"] != calibration_v3.CALIBRATION_RECEIPT_SCHEMA
        or row["score_field"] != "raw_global_action_energy_score"
        or row["phase_conjunctive_score_used_for_calibration"] is not False
        or row["phase_conjunctive_role"]
        != "diagnostic_only_never_optimizer_gate"
        or row["frame_count"] != 81
        or row["branch_order"] != list(mace.BRANCH_ORDER)
        or row["confirmation_rows_consumed_by_optimizer"] is not False
        or row["t2v_media_consumed_by_calibrator"] is not False
        or row["t2v_media_as_rv2v_target_donor_input_or_noise"] is not False
        or row["optimizer_authorized"] is not True
        or row["failure_reasons"] != []
        or row["scientific_action_editing_claim"] is not False
    ):
        raise PairV5NativeRV2VActionScoreError(
            "v3 calibration did not authorize the frozen family maps"
        )
    families = row["action_family_order"]
    mappings = row["mapping_by_family"]
    if (
        not isinstance(families, list)
        or not families
        or len(families) != len(set(families))
        or not isinstance(mappings, Mapping)
        or list(mappings) != families
    ):
        raise PairV5NativeRV2VActionScoreError("calibration family order/closure differs")
    for family in families:
        _safe_id(family, label="calibration action family")
        validate_family_mapping(mappings[family])
    threshold = _finite_float(row["decision_threshold"], label="decision threshold")
    if not 0.0 <= threshold <= 1.0:
        raise PairV5NativeRV2VActionScoreError("decision threshold lies outside [0,1]")
    _sha256(row["source_bank_spec_sha256"], label="source bank spec SHA-256")
    _sha256(row["source_bank_receipt_digest"], label="source bank receipt digest")
    row["receipt_digest"] = digest
    return row


def load_mainline_calibration_bundle(
    *,
    t2v_bank_spec: str | Path,
    t2v_bank_spec_sha256: str,
    t2v_bank_receipt: str | Path,
    t2v_bank_receipt_sha256: str,
    t2v_score_root: str | Path,
    t2v_calibration_root: str | Path,
    t2v_calibration_receipt_sha256: str,
    t2v_preregistration_sha256: str,
    checkpoint_tree_sha256: str,
    formal_v3_method_root: str | Path,
    formal_v3_source_revision: str,
    formal_v3_source_archive_sha256: str,
    python_executable: str | Path = __import__("sys").executable,
) -> dict[str, Any]:
    """Recompute the formal d541801-v3 scalar graph and expose only its maps."""

    try:
        bundle = mainline_calibration.load_mainline_calibration_bundle(
            root_spec=t2v_bank_spec,
            root_spec_sha256=t2v_bank_spec_sha256,
            bank_receipt=t2v_bank_receipt,
            bank_receipt_sha256=t2v_bank_receipt_sha256,
            score_root=t2v_score_root,
            calibration_root=t2v_calibration_root,
            calibration_receipt_sha256=t2v_calibration_receipt_sha256,
            preregistration_sha256=t2v_preregistration_sha256,
            checkpoint_tree_sha256=checkpoint_tree_sha256,
            formal_v3_method_root=formal_v3_method_root,
            formal_v3_source_revision=formal_v3_source_revision,
            formal_v3_source_archive_sha256=formal_v3_source_archive_sha256,
            python_executable=python_executable,
        )
    except mainline_calibration.PairV5MainlineCalibrationError as error:
        raise PairV5NativeRV2VActionScoreError(
            f"formal T2V d541801-v3 scalar calibration failed revalidation: {error}"
        ) from error
    authorization = bundle.get("authorization")
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("calibration_maps_authorized") is not True
        or authorization.get("native_rv2v_optimizer_authorized") is not False
        or authorization.get(
            "t2v_media_latent_gaussian_or_proposal_exported_to_native_scorer"
        )
        is not False
        or authorization.get("formal_score_schema")
        != formal_v3_compat.FORMAL_SCORE_SCHEMA
        or authorization.get("formal_score_scalar_definition")
        != formal_v3_compat.V3_SCALAR_DEFINITION
        or authorization.get("formal_v3_source_revision")
        != formal_v3_compat.PINNED_SOURCE_REVISION
        or authorization.get("formal_v3_source_archive_sha256")
        != formal_v3_source_archive_sha256
        or not isinstance(authorization.get("formal_v3_source_binding_digest"), str)
        or authorization.get("active_repository_score_schema_consumed") is not False
        or authorization.get("active_repository_action_scalar_consumed") is not False
        or authorization.get("decimal_or_log1p_action_scalar_consumed") is not False
    ):
        raise PairV5NativeRV2VActionScoreError(
            "formal d541801-v3 T2V calibration exceeded map-only authority"
        )
    calibration = validate_calibration_receipt(bundle["calibration"])
    if (
        calibration["source_bank_spec_sha256"] != bundle["t2v_spec_sha256"]
        or calibration["source_bank_receipt_digest"]
        != bundle["t2v_bank_receipt_digest"]
        or calibration["preregistration_digest"]
        != bundle["preregistration"]["preregistration_digest"]
        or calibration["receipt_digest"]
        != authorization["calibration_receipt_digest"]
        or bundle["checkpoint_tree_sha256"] != checkpoint_tree_sha256
    ):
        raise PairV5NativeRV2VActionScoreError(
            "formal d541801-v3 T2V calibration binding differs"
        )
    return bundle


def calibration_cells(t2v_spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return closed calibration cells keyed by ``calibration_group_id``."""

    try:
        spec = t2v_bank_contract.validate_root_spec(t2v_spec)
    except t2v_bank_contract.PairT2VCalibrationSpecError as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    cells: dict[str, dict[str, Any]] = {}
    for group in spec["groups"]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for candidate in group["candidates"]:
            grouped.setdefault(candidate["calibration_group_id"], []).append(candidate)
        for cell_id, rows in grouped.items():
            if [row["semantic_branch"] for row in rows] != list(mace.BRANCH_ORDER):
                raise PairV5NativeRV2VActionScoreError("calibration prompt cell order differs")
            action = rows[0]
            cells[cell_id] = {
                "group_id": group["group_id"],
                "visible_gpus": list(group["visible_gpus"]),
                "analysis_split": action["analysis_split"],
                "action_family_id": action["action_family_id"],
                "calibration_group_id": cell_id,
                "action_candidate": action,
                "rows": rows,
                "caption_by_branch": {
                    row["semantic_branch"]: row["full_t2v_caption"] for row in rows
                },
                "caption_sha256_by_branch": {
                    row["semantic_branch"]: row["full_t2v_caption_utf8_sha256"]
                    for row in rows
                },
            }
    return cells


def bind_population_to_calibration(
    population_spec: Mapping[str, Any],
    t2v_spec: Mapping[str, Any],
    *,
    calibration_family_order: Sequence[str],
) -> dict[str, Any]:
    """Prove that a sealed native population contains calibration action rows only."""

    try:
        population = rollout_contract.validate_root_spec(population_spec)
    except rollout_contract.PairRolloutSpecError as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    cells = calibration_cells(t2v_spec)
    action_key_to_cell: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in cells.values():
        action = cell["action_candidate"]
        key = (
            action["geometry_source_video_sha256"],
            action["full_t2v_caption_utf8_sha256"],
            action["full_t2v_caption"],
        )
        if key in action_key_to_cell:
            raise PairV5NativeRV2VActionScoreError(
                "two calibration action cells share source/caption identity"
            )
        action_key_to_cell[key] = cell
    expected_families = list(calibration_family_order)
    if not expected_families or len(expected_families) != len(set(expected_families)):
        raise PairV5NativeRV2VActionScoreError("calibration family order differs")

    bound: list[dict[str, Any]] = []
    group_orders: dict[str, list[str]] = {}
    seen_cell_seeds: dict[str, set[int]] = {}
    seen_cells: set[str] = set()
    for group in population["groups"]:
        order: list[str] = []
        for ordinal, candidate in enumerate(group["candidates"]):
            key = (
                candidate["source_video_sha256"],
                candidate["complete_caption_sha256"],
                candidate["complete_caption"],
            )
            cell = action_key_to_cell.get(key)
            if cell is None:
                raise PairV5NativeRV2VActionScoreError(
                    f"native candidate {candidate['candidate_id']} is not a sealed action branch"
                )
            action = cell["action_candidate"]
            if (
                group["group_id"] != cell["group_id"]
                or group["visible_gpus"] != cell["visible_gpus"]
                or candidate["source_video"] != action["geometry_source_video"]
                or candidate["source_video_sha256"]
                != action["geometry_source_video_sha256"]
                or candidate["complete_caption"] != action["full_t2v_caption"]
                or candidate["complete_caption_sha256"]
                != action["full_t2v_caption_utf8_sha256"]
                or action["semantic_branch"] != "action"
            ):
                raise PairV5NativeRV2VActionScoreError(
                    "native candidate/calibration action-cell binding differs"
                )
            seeds = seen_cell_seeds.setdefault(cell["calibration_group_id"], set())
            if candidate["seed"] in seeds:
                raise PairV5NativeRV2VActionScoreError(
                    "native population repeats a seed within one action cell"
                )
            seeds.add(candidate["seed"])
            seen_cells.add(cell["calibration_group_id"])
            order.append(candidate["candidate_id"])
            bound.append(
                {
                    "group_id": group["group_id"],
                    "visible_gpus": list(group["visible_gpus"]),
                    "ordinal": ordinal,
                    "candidate": candidate,
                    "cell": cell,
                }
            )
        group_orders[group["group_id"]] = order
    if seen_cells != set(cells):
        raise PairV5NativeRV2VActionScoreError(
            "native action population does not cover every calibrated family/split cell"
        )
    replicate_counts = {len(value) for value in seen_cell_seeds.values()}
    if len(replicate_counts) != 1 or next(iter(replicate_counts)) < 2:
        raise PairV5NativeRV2VActionScoreError(
            "each calibration action cell requires the same at-least-two sealed seeds"
        )
    observed_families = []
    for family in expected_families:
        if any(row["cell"]["action_family_id"] == family for row in bound):
            observed_families.append(family)
    if observed_families != expected_families or {
        row["cell"]["action_family_id"] for row in bound
    } != set(expected_families):
        raise PairV5NativeRV2VActionScoreError("native population family closure differs")
    splits = [split for split in calibration_v3.ANALYSIS_SPLITS if any(
        row["cell"]["analysis_split"] == split for row in bound
    )]
    if splits != list(calibration_v3.ANALYSIS_SPLITS):
        raise PairV5NativeRV2VActionScoreError("native population split closure differs")
    global_order = [row["candidate"]["candidate_id"] for row in bound]
    return {
        "population": population,
        "bound_rows": bound,
        "global_candidate_order": global_order,
        "global_candidate_order_digest": object_sha256(global_order),
        "group_candidate_order": group_orders,
        "group_candidate_order_digest": {
            key: object_sha256(value) for key, value in group_orders.items()
        },
        "family_order": expected_families,
        "split_order": splits,
        "cell_ids": sorted(seen_cells),
    }


def _verify_artifact(
    value: Any, *, label: str, required_parent: Optional[Path] = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV5NativeRV2VActionScoreError(f"{label} must be an artifact object")
    path = _plain_file(value.get("path"), label=label)
    expected = _sha256(value.get("sha256"), label=f"{label} SHA-256")
    if required_parent is not None and path.parent != required_parent:
        raise PairV5NativeRV2VActionScoreError(f"{label} escaped its candidate directory")
    if file_sha256(path) != expected:
        raise PairV5NativeRV2VActionScoreError(f"{label} file SHA-256 differs")
    return dict(value)


def _expected_candidate_envelope_sha256(
    *, row: Mapping[str, Any], root_spec_sha256: str
) -> str:
    envelope = {
        "schema_version": rollout_contract.CANDIDATE_SCHEMA_VERSION,
        "root_spec_raw_sha256": root_spec_sha256,
        "group_id": row["group_id"],
        "visible_gpus": row["visible_gpus"],
        "ordinal": row["ordinal"],
        "sampling_contract": rollout_contract.SAMPLING_CONTRACT,
        "semantic_input_closure": rollout_contract.SEMANTIC_INPUT_CLOSURE,
        "candidate": row["candidate"],
    }
    return hashlib.sha256(
        rollout_contract.canonical_json_bytes(envelope) + b"\n"
    ).hexdigest()


def _verify_native_rv2v_receipt(
    native_receipt: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    checkpoint_tree_sha256: str,
    candidate_dir: Path,
) -> dict[str, Any]:
    native_digest = _verify_embedded_with_canonicalizer(
        native_receipt,
        field="receipt_digest",
        label="native RV2V receipt",
        canonicalizer=native_generation.legacy.canonical_json_bytes,
    )
    if (
        native_receipt.get("schema_version") != native_generation.SCHEMA_VERSION
        or native_receipt.get("method") != native_generation.METHOD
        or native_receipt.get("arms") != ["rv2v"]
    ):
        raise PairV5NativeRV2VActionScoreError(
            "native receipt did not execute frozen RV2V-only"
        )
    native_input = native_receipt.get("input")
    if not isinstance(native_input, Mapping) or (
        native_input.get("source_video_path") != candidate["source_video"]
        or native_input.get("source_video_sha256") != candidate["source_video_sha256"]
        or native_input.get("action_prompt_utf8_sha256")
        != candidate["complete_caption_sha256"]
        or native_input.get("accepted_external_conditions")
        != ["source_video", "action_prompt"]
        or native_input.get("target_video") is not False
        or native_input.get("external_reference_image_or_video") is not False
        or native_input.get("external_mask_flow_pose_track_trajectory") is not False
        or native_input.get("external_first_frame_anchor") is not False
    ):
        raise PairV5NativeRV2VActionScoreError("native RV2V input closure differs")
    checkpoint = native_receipt.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("tree_sha256") != checkpoint_tree_sha256:
        raise PairV5NativeRV2VActionScoreError("native RV2V checkpoint tree differs")
    preprocessing = native_receipt.get("preprocessing")
    bucket = preprocessing.get("source_derived_bucket_hw") if isinstance(preprocessing, Mapping) else None
    if (
        not isinstance(preprocessing, Mapping)
        or preprocessing.get("frame_count") != 81
        or preprocessing.get("fps") != 25
        or not isinstance(bucket, list)
        or len(bucket) != 2
        or any(type(item) is not int or item <= 0 or item % 16 for item in bucket)
    ):
        raise PairV5NativeRV2VActionScoreError("native RV2V preprocessing differs")
    conditioning_root = native_receipt.get("conditioning")
    conditioning = conditioning_root.get("rv2v") if isinstance(conditioning_root, Mapping) else None
    if not isinstance(conditioning, Mapping) or (
        conditioning.get("full_source_video_count") != 1
        or conditioning.get("source_derived_reference_count") != 4
        or conditioning.get("source_frame_indices") != [0, 27, 53, 80]
        or conditioning.get("reference_encoding")
        != "independent_rgb_frame_to_wan_vae_[1,C,1,H,W]"
        or conditioning.get("reference_from_temporal_video_latent_slice") is not False
        or conditioning.get("source_ids") != native_generation.source_id_contract("rv2v")
    ):
        raise PairV5NativeRV2VActionScoreError("native RV2V condition contract differs")
    sampling_root = native_receipt.get("sampling")
    sampling = sampling_root.get("rv2v") if isinstance(sampling_root, Mapping) else None
    expected_sampling = {
        **native_generation.native_sampling_contract(
            "rv2v", steps=40, seed=candidate["seed"]
        ),
        "target_initialization": rollout_contract.TARGET_INITIALIZATION,
        "target_mixed_with_source_latent": False,
        "custom_sampler_or_scheduler": False,
        "same_seed_and_target_shape_across_arms": True,
        "single_expert": "transformer_1",
        "ulysses_size": 4,
    }
    expected_sampling["norm_threshold"] = list(expected_sampling["norm_threshold"])
    # The wrapper seals candidate-specific guidance by changing only these
    # native module globals before invoking the unchanged sampler.
    expected_sampling.update(candidate["guidance"])
    if sampling != expected_sampling:
        raise PairV5NativeRV2VActionScoreError("native RV2V sampling differs")
    geometry = native_receipt.get("latent_geometry")
    expected_shape = [1, 16, 21, bucket[0] // 8, bucket[1] // 8]
    if not isinstance(geometry, Mapping) or geometry.get("video_latent_shape") != expected_shape:
        raise PairV5NativeRV2VActionScoreError("native RV2V latent geometry differs")
    identities = native_receipt.get("condition_identities")
    if not isinstance(identities, Mapping) or (
        not isinstance(identities.get("full_source_video"), Mapping)
        or not isinstance(identities.get("references"), Mapping)
        or list(identities["references"]) != ["0", "27", "53", "80"]
        or not isinstance(identities.get("rank_zero_broadcasts"), Mapping)
    ):
        raise PairV5NativeRV2VActionScoreError("native source-condition identities differ")
    source_condition = _verify_artifact(
        native_receipt.get("source_condition_artifact"),
        label="native source condition artifact",
        required_parent=candidate_dir,
    )
    output_root = native_receipt.get("outputs")
    output = output_root.get("rv2v") if isinstance(output_root, Mapping) else None
    noise_root = native_receipt.get("initial_noise_artifacts")
    gaussian = noise_root.get("rv2v") if isinstance(noise_root, Mapping) else None
    if not isinstance(output, Mapping) or not isinstance(gaussian, Mapping):
        raise PairV5NativeRV2VActionScoreError("native RV2V output/noise differs")
    clean = output.get("normalized_clean_latent")
    if not isinstance(clean, Mapping) or (
        output.get("frame_count") != 81
        or output.get("fps") != 25
        or output.get("height") != bucket[0]
        or output.get("width") != bucket[1]
        or clean.get("shape") != expected_shape
        or clean.get("native_sampler_before_vae_decode") is not True
        or clean.get("mp4_decode_reencode_used") is not False
    ):
        raise PairV5NativeRV2VActionScoreError(
            "native RV2V clean latent is not exact81 predecode state"
        )
    if (
        gaussian.get("shape") != expected_shape
        or gaussian.get("generator_initial_seed") != candidate["seed"]
        or gaussian.get("captured_from_native_sampler") is not True
        or gaussian.get("external_initial_noise_injection") is not False
        or gaussian.get("source_or_target_derived") is not False
        or gaussian.get("observer_changed_return_value") is not False
        or gaussian.get("official_randn_tensor_call_count") != 1
        or gaussian.get("original_return_tensor_forwarded_by_identity") is not True
    ):
        raise PairV5NativeRV2VActionScoreError(
            "candidate-own official Gaussian provenance differs"
        )
    for name, artifact in (
        ("native RV2V MP4", output),
        ("native RV2V clean latent", clean),
        ("native RV2V official Gaussian", gaussian),
    ):
        _verify_artifact(artifact, label=name, required_parent=candidate_dir)
    freeze = native_receipt.get("freeze_certificate")
    try:
        frozen_mace_runtime._validated_freeze_certificate(freeze)
    except Exception as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    interpretation = native_receipt.get("interpretation")
    if not isinstance(interpretation, Mapping) or interpretation.get("training_performed") is not False:
        raise PairV5NativeRV2VActionScoreError("native RV2V receipt performed training")
    return {
        "native_receipt_digest": native_digest,
        "mp4": dict(output),
        "predecode_clean_latent": dict(clean),
        "official_initial_gaussian": dict(gaussian),
        "source_condition_artifact": source_condition,
        "source_condition_identity_digest": object_sha256(identities),
        "latent_shape": expected_shape,
    }


def load_native_group_population(
    *,
    population_spec_path: str | Path,
    population_spec_sha256: str,
    rollout_root: str | Path,
    group_id: str,
    calibration_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate one current-family SP4 action population and all artifacts."""

    spec_path = _plain_file(population_spec_path, label="native population spec")
    spec_sha = _sha256(population_spec_sha256, label="native population spec SHA-256")
    try:
        population_spec, observed = rollout_contract.load_sealed_spec(spec_path, spec_sha)
    except rollout_contract.PairRolloutSpecError as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    if observed != spec_sha:
        raise PairV5NativeRV2VActionScoreError("native population spec digest differs")
    calibration = validate_calibration_receipt(calibration_bundle["calibration"])
    binding = bind_population_to_calibration(
        population_spec,
        calibration_bundle["t2v_spec"],
        calibration_family_order=calibration["action_family_order"],
    )
    if group_id not in ("sp4-a", "sp4-b"):
        raise PairV5NativeRV2VActionScoreError("native score group differs")
    root = _plain_directory(rollout_root, label="native rollout root")
    bound_rows = [row for row in binding["bound_rows"] if row["group_id"] == group_id]
    if not bound_rows:
        raise PairV5NativeRV2VActionScoreError("native score group is empty")
    checkpoint_tree = _sha256(
        calibration_bundle["checkpoint_tree_sha256"], label="checkpoint tree SHA-256"
    )
    loaded: list[dict[str, Any]] = []
    gaussian_values_by_cell: dict[str, set[str]] = {}
    for row in bound_rows:
        candidate = row["candidate"]
        candidate_dir = root / candidate["candidate_id"]
        if not candidate_dir.is_dir() or candidate_dir.is_symlink() or candidate_dir.parent != root:
            raise PairV5NativeRV2VActionScoreError("candidate rollout directory differs")
        pair_path = candidate_dir / "pair-v5-rollout-receipt.json"
        pair_receipt = _read_json_file(
            _plain_file(pair_path, label="PAIR native rollout receipt"),
            label="PAIR native rollout receipt",
        )
        _closed(pair_receipt, _PAIR_ROLLOUT_FIELDS, label="PAIR native rollout receipt")
        pair_digest = _verify_embedded_with_canonicalizer(
            pair_receipt,
            field="receipt_digest",
            label="PAIR native rollout receipt",
            canonicalizer=rollout_contract.canonical_json_bytes,
        )
        expected_envelope_sha = _expected_candidate_envelope_sha256(
            row=row, root_spec_sha256=spec_sha
        )
        expected_topology = {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": ",".join(str(item) for item in row["visible_gpus"]),
        }
        if (
            pair_receipt["schema_version"] != rollout_contract.RECEIPT_SCHEMA_VERSION
            or pair_receipt["root_spec_raw_sha256"] != spec_sha
            or pair_receipt["candidate_envelope_sha256"] != expected_envelope_sha
            or pair_receipt["group_id"] != group_id
            or pair_receipt["visible_gpus"] != row["visible_gpus"]
            or pair_receipt["runtime_topology"] != expected_topology
            or pair_receipt["ordinal"] != row["ordinal"]
            or pair_receipt["candidate"] != candidate
            or pair_receipt["sampling_contract"] != rollout_contract.SAMPLING_CONTRACT
            or pair_receipt["semantic_input_closure"]
            != rollout_contract.SEMANTIC_INPUT_CLOSURE
        ):
            raise PairV5NativeRV2VActionScoreError(
                "PAIR native rollout/population binding differs"
            )
        native_path = _plain_file(
            pair_receipt["native_receipt_path"], label="native RV2V receipt"
        )
        if native_path != candidate_dir / "receipt.json":
            raise PairV5NativeRV2VActionScoreError("native receipt escaped candidate directory")
        native_sha = file_sha256(native_path)
        if native_sha != pair_receipt["native_receipt_sha256"]:
            raise PairV5NativeRV2VActionScoreError("native RV2V receipt SHA-256 differs")
        native_receipt = _read_json_file(native_path, label="native RV2V receipt")
        native_artifacts = _verify_native_rv2v_receipt(
            native_receipt,
            candidate=candidate,
            checkpoint_tree_sha256=checkpoint_tree,
            candidate_dir=candidate_dir,
        )
        expected_artifacts = {
            "mp4": native_artifacts["mp4"],
            "predecode_clean_latent": native_artifacts["predecode_clean_latent"],
            "official_initial_gaussian": native_artifacts["official_initial_gaussian"],
        }
        if (
            pair_receipt["native_receipt_digest"]
            != native_artifacts["native_receipt_digest"]
            or pair_receipt["artifacts"] != expected_artifacts
        ):
            raise PairV5NativeRV2VActionScoreError(
                "PAIR/native RV2V receipt artifact binding differs"
            )
        source = _plain_file(candidate["source_video"], label="native source video")
        if file_sha256(source) != candidate["source_video_sha256"]:
            raise PairV5NativeRV2VActionScoreError("native source video SHA-256 differs")
        gaussian_raw = _sha256(
            native_artifacts["official_initial_gaussian"].get("raw_value_sha256"),
            label="candidate-own Gaussian raw-value SHA-256",
        )
        seen_gaussians = gaussian_values_by_cell.setdefault(
            row["cell"]["calibration_group_id"], set()
        )
        if gaussian_raw in seen_gaussians:
            raise PairV5NativeRV2VActionScoreError(
                "two seeds in one native action cell reused a Gaussian tensor value"
            )
        seen_gaussians.add(gaussian_raw)
        loaded.append(
            {
                **row,
                "population_spec_path": str(spec_path),
                "population_spec_sha256": spec_sha,
                "population_binding": binding,
                "rollout_root": str(root),
                "pair_receipt_path": str(pair_path),
                "pair_receipt_file_sha256": file_sha256(pair_path),
                "pair_receipt_digest": pair_digest,
                "pair_receipt": pair_receipt,
                "native_receipt_path": str(native_path),
                "native_receipt_file_sha256": native_sha,
                "native_receipt": native_receipt,
                "native_artifacts": native_artifacts,
                "source_video_path": str(source),
                "source_video_file_sha256": candidate["source_video_sha256"],
            }
        )
    expected_order = binding["group_candidate_order"][group_id]
    if [row["candidate"]["candidate_id"] for row in loaded] != expected_order:
        raise PairV5NativeRV2VActionScoreError("native group candidate order differs")
    return binding, loaded


def prompt_binding_from_cell(
    cell: Mapping[str, Any], *, prompt_cleaner: Optional[Any] = None
) -> dict[str, Any]:
    captions = {
        branch: cell["caption_by_branch"][branch] for branch in mace.BRANCH_ORDER
    }
    caption_hashes = {
        branch: _sha256(
            cell["caption_sha256_by_branch"][branch],
            label=f"{branch} caption SHA-256",
        )
        for branch in mace.BRANCH_ORDER
    }
    for branch in mace.BRANCH_ORDER:
        if hashlib.sha256(captions[branch].encode("utf-8")).hexdigest() != caption_hashes[branch]:
            raise PairV5NativeRV2VActionScoreError(
                f"sealed calibration caption hash differs for {branch}"
            )
    try:
        prompts = frozen_mace_runtime.official_prompt_bank_from_captions(
            captions, prompt_cleaner=prompt_cleaner
        )
        builder = frozen_mace_runtime.prompt_builder_contract()
    except frozen_mace_runtime.PairV5T2VEnergyScoringError as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    prompt_hashes = {
        branch: hashlib.sha256(prompts[branch].encode("utf-8")).hexdigest()
        for branch in mace.BRANCH_ORDER
    }
    return {
        "branch_order": list(mace.BRANCH_ORDER),
        "full_t2v_caption_by_branch": captions,
        "full_t2v_caption_utf8_sha256_by_branch": caption_hashes,
        "prompt_by_branch": prompts,
        "prompt_utf8_sha256_by_branch": prompt_hashes,
        "prompt_builder_contract": builder,
        "prompt_registry_digest": native_bridge.object_sha256(prompts),
        "calibration_group_id": cell["calibration_group_id"],
    }


def candidate_coordinate_binding(
    clean: Any,
    epsilon: Any,
    sigma: Any,
    *,
    clean_artifact: Mapping[str, Any],
    gaussian_artifact: Mapping[str, Any],
    candidate_seed: int,
) -> dict[str, Any]:
    """Hash the candidate-own clean/Gaussian coordinate used by MACE."""

    import torch

    if (
        not isinstance(clean, torch.Tensor)
        or not isinstance(epsilon, torch.Tensor)
        or not isinstance(sigma, torch.Tensor)
        or clean.dtype != torch.float32
        or epsilon.dtype != torch.float32
        or sigma.dtype != torch.float32
        or clean.requires_grad
        or epsilon.requires_grad
        or sigma.requires_grad
        or clean.shape != epsilon.shape
        or clean.ndim != 5
        or tuple(int(item) for item in clean.shape[:3]) != (1, 16, 21)
        or tuple(int(item) for item in sigma.shape) != (1,)
        or not bool(torch.isfinite(clean).all().item())
        or not bool(torch.isfinite(epsilon).all().item())
        or not bool(torch.isfinite(sigma).all().item())
    ):
        raise PairV5NativeRV2VActionScoreError(
            "candidate coordinate must be detached FP32 exact81 clean/Gaussian/sigma"
        )
    if struct.pack("!f", float(sigma.item())).hex() != struct.pack("!f", PILOT_SIGMA).hex():
        raise PairV5NativeRV2VActionScoreError("candidate coordinate sigma differs")
    if gaussian_artifact.get("generator_initial_seed") != candidate_seed:
        raise PairV5NativeRV2VActionScoreError(
            "official Gaussian does not belong to the scored candidate seed"
        )
    if clean_artifact.get("path") == gaussian_artifact.get("path"):
        raise PairV5NativeRV2VActionScoreError("clean and Gaussian artifacts alias")
    sigma_view = sigma.reshape(1, 1, 1, 1, 1)
    x_sigma = ((1.0 - sigma_view) * clean + sigma_view * epsilon).detach()
    return {
        "clean_latent_path": str(clean_artifact["path"]),
        "clean_latent_artifact_sha256": _sha256(
            clean_artifact.get("sha256"), label="clean latent artifact SHA-256"
        ),
        "clean_latent_tensor_sha256": frozen_mace_runtime.tensor_sha256(clean),
        "official_gaussian_path": str(gaussian_artifact["path"]),
        "official_gaussian_artifact_sha256": _sha256(
            gaussian_artifact.get("sha256"), label="Gaussian artifact SHA-256"
        ),
        "official_gaussian_raw_value_sha256": _sha256(
            gaussian_artifact.get("raw_value_sha256"),
            label="Gaussian raw-value SHA-256",
        ),
        "official_gaussian_content_sha256": _sha256(
            gaussian_artifact.get("content_sha256"),
            label="Gaussian content SHA-256",
        ),
        "official_gaussian_tensor_sha256": frozen_mace_runtime.tensor_sha256(epsilon),
        "official_gaussian_generator_seed": candidate_seed,
        "sigma_tensor_sha256": frozen_mace_runtime.tensor_sha256(sigma),
        "candidate_own_x_sigma_tensor_sha256": frozen_mace_runtime.tensor_sha256(x_sigma),
        "tensor_shape": [int(item) for item in clean.shape],
        "x_sigma_construction": "fp32_(1-sigma)*candidate_clean+sigma*candidate_own_official_gaussian",
        "clean_and_gaussian_are_same_candidate_artifacts": True,
    }


def _candidate_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    cell = row["cell"]
    return {
        "candidate_id": candidate["candidate_id"],
        "group_id": row["group_id"],
        "visible_gpus": list(row["visible_gpus"]),
        "ordinal": row["ordinal"],
        "seed": candidate["seed"],
        "guidance": dict(candidate["guidance"]),
        "analysis_split": cell["analysis_split"],
        "action_family_id": cell["action_family_id"],
        "calibration_group_id": cell["calibration_group_id"],
        "calibration_action_candidate_id": cell["action_candidate"]["candidate_id"],
        "source_video_sha256": candidate["source_video_sha256"],
        "complete_caption_utf8_sha256": candidate["complete_caption_sha256"],
    }


def _population_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    population = row["population_binding"]
    return {
        "spec_path": row["population_spec_path"],
        "spec_raw_sha256": row["population_spec_sha256"],
        "candidate_count": len(population["global_candidate_order"]),
        "global_candidate_order": list(population["global_candidate_order"]),
        "global_candidate_order_digest": population["global_candidate_order_digest"],
        "group_candidate_order": list(
            population["group_candidate_order"][row["group_id"]]
        ),
        "group_candidate_order_digest": population["group_candidate_order_digest"][
            row["group_id"]
        ],
        "population_family_order": list(population["family_order"]),
        "population_split_order": list(population["split_order"]),
        "all_candidates_match_calibration_action_branch": True,
        "negative_branch_candidates_present": False,
    }


def _source_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    action = row["cell"]["action_candidate"]
    return {
        "source_video_path": row["source_video_path"],
        "source_video_sha256_declared": row["candidate"]["source_video_sha256"],
        "source_video_sha256_recomputed": row["source_video_file_sha256"],
        "calibration_geometry_source_path": action["geometry_source_video"],
        "calibration_geometry_source_sha256": action[
            "geometry_source_video_sha256"
        ],
        "source_path_and_hash_match_calibration_action_cell": True,
        "source_tensor_consumed_by_scorer": False,
    }


def _rollout_binding(row: Mapping[str, Any], checkpoint_tree_sha256: str) -> dict[str, Any]:
    artifacts = row["native_artifacts"]
    pair = row["pair_receipt"]
    return {
        "rollout_root": row["rollout_root"],
        "pair_receipt_path": row["pair_receipt_path"],
        "pair_receipt_file_sha256": row["pair_receipt_file_sha256"],
        "pair_receipt_digest": row["pair_receipt_digest"],
        "native_receipt_path": row["native_receipt_path"],
        "native_receipt_file_sha256": row["native_receipt_file_sha256"],
        "native_receipt_digest": artifacts["native_receipt_digest"],
        "candidate_envelope_sha256": pair["candidate_envelope_sha256"],
        "runtime_topology": dict(pair["runtime_topology"]),
        "native_arm": "rv2v",
        "native_condition_mode": "rv2v4",
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "source_condition_identity_digest": artifacts[
            "source_condition_identity_digest"
        ],
        "source_condition_artifact_sha256": artifacts[
            "source_condition_artifact"
        ]["sha256"],
        "generated_mp4_sha256": artifacts["mp4"]["sha256"],
        "generated_mp4_consumed_by_scorer": False,
    }


def _calibration_binding(
    row: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    calibration = validate_calibration_receipt(bundle["calibration"])
    family = row["cell"]["action_family_id"]
    mapping = validate_family_mapping(calibration["mapping_by_family"][family])
    authorization = bundle["authorization"]
    return {
        "mainline_authorization_digest": authorization["authorization_digest"],
        "formal_score_provenance_set_digest": bundle[
            "formal_score_provenance_set_digest"
        ],
        "formal_score_schema": authorization["formal_score_schema"],
        "formal_score_scalar_definition": authorization[
            "formal_score_scalar_definition"
        ],
        "formal_v3_source_revision": authorization[
            "formal_v3_source_revision"
        ],
        "formal_v3_source_archive_sha256": authorization[
            "formal_v3_source_archive_sha256"
        ],
        "formal_v3_source_binding_digest": authorization[
            "formal_v3_source_binding_digest"
        ],
        "family_mapping_set_digest": bundle["family_mapping_set_digest"],
        "calibration_root": bundle["calibration_root"],
        "t2v_score_root": bundle["t2v_score_root"],
        "preregistration_path": bundle["preregistration_path"],
        "preregistration_file_sha256": bundle["preregistration_file_sha256"],
        "preregistration_digest": bundle["preregistration"][
            "preregistration_digest"
        ],
        "calibration_receipt_path": bundle["calibration_path"],
        "calibration_receipt_file_sha256": bundle["calibration_file_sha256"],
        "calibration_receipt_digest": calibration["receipt_digest"],
        "source_t2v_bank_spec_path": bundle["t2v_spec_path"],
        "source_t2v_bank_spec_sha256": bundle["t2v_spec_sha256"],
        "source_t2v_bank_receipt_path": bundle["t2v_bank_receipt_path"],
        "source_t2v_bank_receipt_file_sha256": bundle[
            "t2v_bank_receipt_file_sha256"
        ],
        "source_t2v_bank_receipt_digest": bundle["t2v_bank_receipt_digest"],
        "action_family_id": family,
        "family_mapping": mapping,
        "family_mapping_digest": mapping["mapping_digest"],
        "decision_threshold": calibration["decision_threshold"],
        "calibration_maps_authorized": True,
        "native_rv2v_optimizer_authorized": False,
        "t2v_media_consumed_by_native_scorer": False,
    }


def make_score_receipt(
    *,
    row: Mapping[str, Any],
    calibration_bundle: Mapping[str, Any],
    prompt_binding: Mapping[str, Any],
    clean: Any,
    epsilon: Any,
    sigma: Any,
    score: native_bridge.FrozenT2VActionScore,
    scorer_packet_receipt: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    freeze_certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one executed native score; this function never accepts media tensors."""

    import torch

    if not isinstance(score, native_bridge.FrozenT2VActionScore):
        raise PairV5NativeRV2VActionScoreError("score must be a frozen native MACE result")
    prompts = dict(_closed(prompt_binding, _PROMPT_FIELDS, label="prompt binding"))
    if prompts["calibration_group_id"] != row["cell"]["calibration_group_id"]:
        raise PairV5NativeRV2VActionScoreError("prompt/candidate calibration cell differs")
    coordinate = candidate_coordinate_binding(
        clean,
        epsilon,
        sigma,
        clean_artifact=row["native_artifacts"]["predecode_clean_latent"],
        gaussian_artifact=row["native_artifacts"]["official_initial_gaussian"],
        candidate_seed=row["candidate"]["seed"],
    )
    if not torch.equal(score.energy.x_sigma, (
        (1.0 - sigma.reshape(1, 1, 1, 1, 1)) * clean
        + sigma.reshape(1, 1, 1, 1, 1) * epsilon
    )):
        raise PairV5NativeRV2VActionScoreError(
            "executed MACE x_sigma is not the candidate-own coordinate"
        )
    try:
        v3_energy = formal_v3_compat.make_native_v3_energy_packet(score.energy)
    except formal_v3_compat.PairV5T2VScoreV3CompatibilityError as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    action_energy = v3_energy["global_action_energy"]
    negative_energies = v3_energy[
        "global_hard_negative_energy_by_branch"
    ]
    hardest_branch = v3_energy["global_hardest_negative_branch"]
    raw = v3_energy["raw_global_action_energy_score"]
    calibration_binding = _calibration_binding(row, calibration_bundle)
    calibrated = apply_family_mapping(raw, calibration_binding["family_mapping"])
    threshold = calibration_binding["decision_threshold"]
    checkpoint_binding = frozen_mace_runtime.checkpoint_content_binding(
        checkpoint_identity, freeze_certificate
    )
    checkpoint_identity_checked = frozen_mace_runtime._validated_checkpoint_identity(
        checkpoint_identity
    )
    packet_binding = frozen_mace_runtime.frozen_t2v_packet_binding(
        scorer_packet_receipt, score.receipt
    )
    unsigned = {
        "schema_version": SCORE_SCHEMA,
        "candidate": _candidate_binding(row),
        "population": _population_binding(row),
        "source": _source_binding(row),
        "rollout": _rollout_binding(
            row, calibration_bundle["checkpoint_tree_sha256"]
        ),
        "calibration": calibration_binding,
        "prompts": prompts,
        "artifacts": coordinate,
        "frozen_runtime": {
            "checkpoint_content_identity": checkpoint_identity_checked,
            "frozen_checkpoint_receipt_digest": object_sha256(
                checkpoint_identity_checked
            ),
            "checkpoint_content_binding": checkpoint_binding,
            "frozen_scorer_receipt_digest": _sha256(
                score.receipt["digest"], label="frozen scorer receipt digest"
            ),
            "frozen_t2v_packet_binding": packet_binding,
            "all_loaded_parameters_frozen_before_and_after": True,
        },
        "score_coordinate": frozen_mace_runtime.schedule_coordinate_receipt(),
        "mace": {
            "definition": formal_v3_compat.V3_SCALAR_DEFINITION,
            "energy_epsilon": v3_energy["energy_epsilon"],
            "global_action_energy": action_energy,
            "global_hard_negative_energy_by_branch": negative_energies,
            "global_negative_log_energy_ratio_by_branch": v3_energy[
                "global_negative_log_energy_ratio_by_branch"
            ],
            "global_hardest_negative_branch": hardest_branch,
            "formal_v3_energy_packet": v3_energy,
            "formal_v3_energy_packet_digest": v3_energy["packet_digest"],
            "formal_v3_source_revision": formal_v3_compat.PINNED_SOURCE_REVISION,
            "raw_global_action_energy_score": raw,
            "calibrated_family_action_score": calibrated,
            "decision_threshold": threshold,
            "passes_calibrated_action_metric": calibrated >= threshold,
            "phase_conjunctive_score_diagnostic": float(
                score.phase_energy.reward.item()
            ),
            "phase_diagnostic_receipt_digest": _sha256(
                score.phase_energy.receipt["receipt_digest"],
                label="phase diagnostic receipt digest",
            ),
            "phase_diagnostic_used_for_action_metric": False,
        },
        "input_closure": NATIVE_SCORE_INPUT_CLOSURE,
        "optimizer_authorized": False,
        "scientific_action_editing_claim": False,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    return validate_score_receipt(receipt)


def _validate_prompt_binding(value: Any) -> dict[str, Any]:
    row = dict(_closed(value, _PROMPT_FIELDS, label="prompt binding"))
    if row["branch_order"] != list(mace.BRANCH_ORDER):
        raise PairV5NativeRV2VActionScoreError("prompt branch order differs")
    captions = row["full_t2v_caption_by_branch"]
    caption_hashes = row["full_t2v_caption_utf8_sha256_by_branch"]
    prompts = row["prompt_by_branch"]
    prompt_hashes = row["prompt_utf8_sha256_by_branch"]
    for name, registry in (
        ("caption", captions),
        ("caption hash", caption_hashes),
        ("prompt", prompts),
        ("prompt hash", prompt_hashes),
    ):
        if not isinstance(registry, Mapping) or set(registry) != set(mace.BRANCH_ORDER):
            raise PairV5NativeRV2VActionScoreError(f"{name} registry order differs")
    try:
        mace.validate_prompt_closure(prompts)
    except mace.MACECandidateActionEnergyError as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    for branch in mace.BRANCH_ORDER:
        caption = captions[branch]
        prompt = prompts[branch]
        if (
            type(caption) is not str
            or type(prompt) is not str
            or hashlib.sha256(caption.encode("utf-8")).hexdigest()
            != _sha256(caption_hashes[branch], label=f"{branch} caption SHA-256")
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            != _sha256(prompt_hashes[branch], label=f"{branch} prompt SHA-256")
        ):
            raise PairV5NativeRV2VActionScoreError(
                f"prompt/caption text hash differs for {branch}"
            )
    if native_bridge.object_sha256(dict(prompts)) != _sha256(
        row["prompt_registry_digest"], label="prompt registry digest"
    ):
        raise PairV5NativeRV2VActionScoreError("prompt registry digest differs")
    builder = row["prompt_builder_contract"]
    if not isinstance(builder, Mapping) or set(builder) != set(
        _PROMPT_BUILDER_FIELDS
    ):
        raise PairV5NativeRV2VActionScoreError("prompt builder contract differs")
    unsigned_builder = dict(builder)
    builder_digest = _sha256(
        unsigned_builder.pop("contract_digest", None),
        label="prompt builder contract digest",
    )
    if frozen_mace_runtime.object_sha256(unsigned_builder) != builder_digest or (
        builder.get("builder")
        != "infer_native_identity_generation_canary.build_task_prompt"
        or builder.get("arm") != "t2v"
        or builder.get("training_task_name") != "t2v"
        or builder.get("prompt_cleaner")
        != "diffusers.pipelines.wan.pipeline_wan.prompt_clean"
    ):
        raise PairV5NativeRV2VActionScoreError("official T2V prompt builder differs")
    _safe_id(row["calibration_group_id"], label="prompt calibration group")
    return row


def _validate_frozen_packet(value: Any, *, prompt_digest: str, model_digest: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_FROZEN_PACKET_FIELDS):
        raise PairV5NativeRV2VActionScoreError("frozen T2V packet binding differs")
    unsigned = dict(value)
    declared = _sha256(
        unsigned.pop("binding_digest", None), label="packet binding digest"
    )
    coordinate = frozen_mace_runtime.schedule_coordinate_receipt()
    shape = value.get("candidate_shape")
    if (
        frozen_mace_runtime.object_sha256(unsigned) != declared
        or value.get("prompt_registry_digest") != prompt_digest
        or value.get("frozen_model_receipt_digest") != model_digest
        or not isinstance(shape, list)
        or len(shape) != 5
        or shape[:3] != [1, 16, 21]
        or any(type(item) is not int or item <= 0 for item in shape)
        or value.get("sigma_float32_bits_hex")
        != coordinate["physical_sigma_float32_be_hex"]
        or value.get("timestep_float32_bits_hex")
        != coordinate["frozen_t2v_scorer_timestep_float32_be_hex"]
        or value.get("native_schedule_digest") != coordinate["schedule_digest"]
        or value.get("native_schedule_index") != PILOT_SCHEDULE_INDEX
        or value.get("native_scheduler_timestep") != PILOT_NATIVE_TIMESTEP
        or value.get("timestep_mapping")
        != coordinate["frozen_t2v_scorer_timestep_mapping"]
        or value.get("physical_sigma_and_model_timestep_share_native_exact40_index")
        is not True
        or value.get("legacy_1000_sigma_timestep_rejected") is not True
    ):
        raise PairV5NativeRV2VActionScoreError(
            "frozen packet is not the native exact40 t=516 MACE packet"
        )


def validate_score_receipt(value: Any) -> dict[str, Any]:
    row = dict(_closed(value, _SCORE_FIELDS, label="native action score receipt"))
    digest = _verify_embedded(
        row, field="receipt_digest", label="native action score receipt"
    )
    if (
        row["schema_version"] != SCORE_SCHEMA
        or row["input_closure"] != NATIVE_SCORE_INPUT_CLOSURE
        or row["optimizer_authorized"] is not False
        or row["scientific_action_editing_claim"] is not False
    ):
        raise PairV5NativeRV2VActionScoreError("native score authority differs")

    candidate = dict(_closed(row["candidate"], _CANDIDATE_FIELDS, label="candidate"))
    for name in (
        "candidate_id",
        "group_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "calibration_action_candidate_id",
    ):
        _safe_id(candidate[name], label=name)
    if (
        candidate["group_id"] not in ("sp4-a", "sp4-b")
        or candidate["visible_gpus"]
        != ([0, 1, 2, 3] if candidate["group_id"] == "sp4-a" else [4, 5, 6, 7])
        or type(candidate["ordinal"]) is not int
        or candidate["ordinal"] < 0
        or type(candidate["seed"]) is not int
        or not 0 <= candidate["seed"] < 2**63
        or candidate["analysis_split"] not in calibration_v3.ANALYSIS_SPLITS
        or not isinstance(candidate["guidance"], Mapping)
        or set(candidate["guidance"]) != set(rollout_contract.DEFAULT_GUIDANCE)
        or any(
            type(candidate["guidance"][name]) is not float
            or not math.isfinite(candidate["guidance"][name])
            or candidate["guidance"][name] < 0.0
            for name in rollout_contract.DEFAULT_GUIDANCE
        )
    ):
        raise PairV5NativeRV2VActionScoreError("candidate identity contract differs")
    for name in ("source_video_sha256", "complete_caption_utf8_sha256"):
        _sha256(candidate[name], label=name)

    population = dict(
        _closed(row["population"], _POPULATION_FIELDS, label="population binding")
    )
    if (
        not isinstance(population["global_candidate_order"], list)
        or len(population["global_candidate_order"]) != population["candidate_count"]
        or len(set(population["global_candidate_order"]))
        != population["candidate_count"]
        or object_sha256(population["global_candidate_order"])
        != population["global_candidate_order_digest"]
        or not isinstance(population["group_candidate_order"], list)
        or object_sha256(population["group_candidate_order"])
        != population["group_candidate_order_digest"]
        or candidate["candidate_id"] not in population["group_candidate_order"]
        or candidate["ordinal"]
        != population["group_candidate_order"].index(candidate["candidate_id"])
        or candidate["action_family_id"] not in population["population_family_order"]
        or population["population_split_order"] != list(calibration_v3.ANALYSIS_SPLITS)
        or population["all_candidates_match_calibration_action_branch"] is not True
        or population["negative_branch_candidates_present"] is not False
    ):
        raise PairV5NativeRV2VActionScoreError("population order/action closure differs")
    _sha256(population["spec_raw_sha256"], label="population spec SHA-256")

    source = dict(_closed(row["source"], _SOURCE_FIELDS, label="source binding"))
    for name in (
        "source_video_sha256_declared",
        "source_video_sha256_recomputed",
        "calibration_geometry_source_sha256",
    ):
        _sha256(source[name], label=name)
    if (
        source["source_video_sha256_declared"]
        != source["source_video_sha256_recomputed"]
        or source["source_video_sha256_declared"]
        != source["calibration_geometry_source_sha256"]
        or source["source_video_sha256_declared"] != candidate["source_video_sha256"]
        or source["source_video_path"] != source["calibration_geometry_source_path"]
        or source["source_path_and_hash_match_calibration_action_cell"] is not True
        or source["source_tensor_consumed_by_scorer"] is not False
    ):
        raise PairV5NativeRV2VActionScoreError("source/calibration cell binding differs")

    rollout = dict(_closed(row["rollout"], _ROLLOUT_FIELDS, label="rollout binding"))
    for name in (
        "pair_receipt_file_sha256",
        "pair_receipt_digest",
        "native_receipt_file_sha256",
        "native_receipt_digest",
        "candidate_envelope_sha256",
        "checkpoint_tree_sha256",
        "source_condition_identity_digest",
        "source_condition_artifact_sha256",
        "generated_mp4_sha256",
    ):
        _sha256(rollout[name], label=name)
    if (
        rollout["runtime_topology"]
        != {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": ",".join(str(item) for item in candidate["visible_gpus"]),
        }
        or rollout["native_arm"] != "rv2v"
        or rollout["native_condition_mode"] != "rv2v4"
        or rollout["generated_mp4_consumed_by_scorer"] is not False
    ):
        raise PairV5NativeRV2VActionScoreError("native rollout topology/role differs")

    calibration = dict(
        _closed(row["calibration"], _CALIBRATION_FIELDS, label="calibration binding")
    )
    mapping = validate_family_mapping(calibration["family_mapping"])
    for name in (
        "mainline_authorization_digest",
        "formal_score_provenance_set_digest",
        "formal_v3_source_archive_sha256",
        "formal_v3_source_binding_digest",
        "family_mapping_set_digest",
        "preregistration_file_sha256",
        "preregistration_digest",
        "calibration_receipt_file_sha256",
        "calibration_receipt_digest",
        "source_t2v_bank_spec_sha256",
        "source_t2v_bank_receipt_file_sha256",
        "source_t2v_bank_receipt_digest",
        "family_mapping_digest",
    ):
        _sha256(calibration[name], label=name)
    threshold = _finite_float(calibration["decision_threshold"], label="decision threshold")
    if (
        calibration["action_family_id"] != candidate["action_family_id"]
        or calibration["formal_score_schema"]
        != formal_v3_compat.FORMAL_SCORE_SCHEMA
        or calibration["formal_score_scalar_definition"]
        != formal_v3_compat.V3_SCALAR_DEFINITION
        or calibration["formal_v3_source_revision"]
        != formal_v3_compat.PINNED_SOURCE_REVISION
        or calibration["family_mapping_digest"] != mapping["mapping_digest"]
        or not 0.0 <= threshold <= 1.0
        or calibration["calibration_maps_authorized"] is not True
        or calibration["native_rv2v_optimizer_authorized"] is not False
        or calibration["t2v_media_consumed_by_native_scorer"] is not False
    ):
        raise PairV5NativeRV2VActionScoreError("calibration family authority differs")

    prompts = _validate_prompt_binding(row["prompts"])
    if (
        prompts["calibration_group_id"] != candidate["calibration_group_id"]
        or prompts["full_t2v_caption_utf8_sha256_by_branch"]["action"]
        != candidate["complete_caption_utf8_sha256"]
    ):
        raise PairV5NativeRV2VActionScoreError("candidate/action prompt binding differs")

    artifacts = dict(_closed(row["artifacts"], _ARTIFACT_FIELDS, label="artifact binding"))
    for name in (
        "clean_latent_artifact_sha256",
        "clean_latent_tensor_sha256",
        "official_gaussian_artifact_sha256",
        "official_gaussian_raw_value_sha256",
        "official_gaussian_content_sha256",
        "official_gaussian_tensor_sha256",
        "sigma_tensor_sha256",
        "candidate_own_x_sigma_tensor_sha256",
    ):
        _sha256(artifacts[name], label=name)
    if (
        artifacts["official_gaussian_generator_seed"] != candidate["seed"]
        or not isinstance(artifacts["tensor_shape"], list)
        or len(artifacts["tensor_shape"]) != 5
        or artifacts["tensor_shape"][:3] != [1, 16, 21]
        or artifacts["x_sigma_construction"]
        != "fp32_(1-sigma)*candidate_clean+sigma*candidate_own_official_gaussian"
        or artifacts["clean_and_gaussian_are_same_candidate_artifacts"] is not True
        or artifacts["clean_latent_path"] == artifacts["official_gaussian_path"]
    ):
        raise PairV5NativeRV2VActionScoreError("candidate-own artifact coordinate differs")

    runtime = dict(
        _closed(row["frozen_runtime"], _FROZEN_RUNTIME_FIELDS, label="frozen runtime")
    )
    try:
        checkpoint_identity = frozen_mace_runtime._validated_checkpoint_identity(
            runtime["checkpoint_content_identity"]
        )
        rebuilt_checkpoint = frozen_mace_runtime.checkpoint_content_binding(
            checkpoint_identity,
            runtime["checkpoint_content_binding"]["freeze_certificate"],
        )
    except Exception as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    model_digest = _sha256(
        runtime["frozen_checkpoint_receipt_digest"],
        label="frozen checkpoint receipt digest",
    )
    _sha256(runtime["frozen_scorer_receipt_digest"], label="frozen scorer receipt digest")
    if (
        object_sha256(checkpoint_identity) != model_digest
        or rebuilt_checkpoint != runtime["checkpoint_content_binding"]
        or runtime["all_loaded_parameters_frozen_before_and_after"] is not True
    ):
        raise PairV5NativeRV2VActionScoreError("frozen checkpoint binding differs")
    _validate_frozen_packet(
        runtime["frozen_t2v_packet_binding"],
        prompt_digest=prompts["prompt_registry_digest"],
        model_digest=model_digest,
    )
    if row["score_coordinate"] != frozen_mace_runtime.schedule_coordinate_receipt():
        raise PairV5NativeRV2VActionScoreError("native score coordinate differs")

    score = dict(_closed(row["mace"], _MACE_FIELDS, label="MACE score"))
    raw = _finite_float(
        score["raw_global_action_energy_score"], label="raw action score"
    )
    calibrated = _finite_float(
        score["calibrated_family_action_score"], label="calibrated action score"
    )
    phase = _finite_float(
        score["phase_conjunctive_score_diagnostic"], label="phase diagnostic"
    )
    del phase
    try:
        v3_energy = formal_v3_compat.validate_native_v3_energy_packet(
            score["formal_v3_energy_packet"]
        )
    except formal_v3_compat.PairV5T2VScoreV3CompatibilityError as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    expected_calibrated = apply_family_mapping(raw, mapping)
    if (
        score["definition"]
        != formal_v3_compat.V3_SCALAR_DEFINITION
        or score["formal_v3_source_revision"]
        != formal_v3_compat.PINNED_SOURCE_REVISION
        or score["formal_v3_energy_packet_digest"] != v3_energy["packet_digest"]
        or score["energy_epsilon"] != v3_energy["energy_epsilon"]
        or score["global_action_energy"]
        != v3_energy["global_action_energy"]
        or score["global_hard_negative_energy_by_branch"]
        != v3_energy["global_hard_negative_energy_by_branch"]
        or score["global_negative_log_energy_ratio_by_branch"]
        != v3_energy["global_negative_log_energy_ratio_by_branch"]
        or score["global_hardest_negative_branch"]
        != v3_energy["global_hardest_negative_branch"]
        or raw != v3_energy["raw_global_action_energy_score"]
        or calibrated != expected_calibrated
        or score["decision_threshold"] != threshold
        or score["passes_calibrated_action_metric"] != (calibrated >= threshold)
        or type(score["passes_calibrated_action_metric"]) is not bool
        or score["phase_diagnostic_used_for_action_metric"] is not False
    ):
        raise PairV5NativeRV2VActionScoreError("MACE/calibration score closure differs")
    _sha256(score["phase_diagnostic_receipt_digest"], label="phase receipt digest")
    row["receipt_digest"] = digest
    return row


def safe_pareto_action_record(score_receipt: Any) -> dict[str, Any]:
    """Project a native receipt into a non-authoritative safe-Pareto input."""

    score = validate_score_receipt(score_receipt)
    candidate = score["candidate"]
    split = candidate["analysis_split"]
    unsigned = {
        "schema_version": SAFE_PARETO_ACTION_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "analysis_split": split,
        "action_family_id": candidate["action_family_id"],
        "calibrated_action_score": score["mace"]["calibrated_family_action_score"],
        "action_decision_threshold": score["mace"]["decision_threshold"],
        "passes_calibrated_action_metric": score["mace"][
            "passes_calibrated_action_metric"
        ],
        "native_score_receipt_digest": score["receipt_digest"],
        "population_spec_raw_sha256": score["population"]["spec_raw_sha256"],
        "source_video_sha256": score["candidate"]["source_video_sha256"],
        "rollout_receipt_digest": score["rollout"]["pair_receipt_digest"],
        "calibration_receipt_digest": score["calibration"][
            "calibration_receipt_digest"
        ],
        "prompt_registry_digest": score["prompts"]["prompt_registry_digest"],
        "candidate_own_x_sigma_tensor_sha256": score["artifacts"][
            "candidate_own_x_sigma_tensor_sha256"
        ],
        "discovery_selection_candidate": split == "fit",
        "confirmation_metric_only": split == "confirmation",
        "requires_source_identity_metric": True,
        "requires_camera_metric": True,
        "requires_temporal_consistency_metric": True,
        "standalone_candidate_selection_authorized": False,
        "optimizer_authorized": False,
        "action_editing_success_inferred": False,
    }
    value = {**unsigned, "record_digest": object_sha256(unsigned)}
    return validate_safe_pareto_action_record(value)


def validate_safe_pareto_action_record(value: Any) -> dict[str, Any]:
    row = dict(_closed(value, _SAFE_PARETO_FIELDS, label="safe-Pareto action record"))
    digest = _verify_embedded(
        row, field="record_digest", label="safe-Pareto action record"
    )
    if (
        row["schema_version"] != SAFE_PARETO_ACTION_SCHEMA
        or row["analysis_split"] not in calibration_v3.ANALYSIS_SPLITS
        or row["discovery_selection_candidate"]
        != (row["analysis_split"] == "fit")
        or row["confirmation_metric_only"]
        != (row["analysis_split"] == "confirmation")
        or row["requires_source_identity_metric"] is not True
        or row["requires_camera_metric"] is not True
        or row["requires_temporal_consistency_metric"] is not True
        or row["standalone_candidate_selection_authorized"] is not False
        or row["optimizer_authorized"] is not False
        or row["action_editing_success_inferred"] is not False
    ):
        raise PairV5NativeRV2VActionScoreError("safe-Pareto authority differs")
    _finite_float(row["calibrated_action_score"], label="safe-Pareto action score")
    threshold = _finite_float(
        row["action_decision_threshold"], label="safe-Pareto action threshold"
    )
    if (
        not 0.0 <= threshold <= 1.0
        or type(row["passes_calibrated_action_metric"]) is not bool
    ):
        raise PairV5NativeRV2VActionScoreError("safe-Pareto action scalar differs")
    for name in (
        "native_score_receipt_digest",
        "population_spec_raw_sha256",
        "source_video_sha256",
        "rollout_receipt_digest",
        "calibration_receipt_digest",
        "prompt_registry_digest",
        "candidate_own_x_sigma_tensor_sha256",
    ):
        _sha256(row[name], label=name)
    row["record_digest"] = digest
    return row


def verify_score_against_context(
    score_receipt: Any,
    *,
    row: Mapping[str, Any],
    calibration_bundle: Mapping[str, Any],
    clean: Any,
    epsilon: Any,
    sigma: Any,
) -> dict[str, Any]:
    """Replay all non-model bindings, including the candidate-own x_sigma hash."""

    score = validate_score_receipt(score_receipt)
    prompt = prompt_binding_from_cell(row["cell"])
    coordinate = candidate_coordinate_binding(
        clean,
        epsilon,
        sigma,
        clean_artifact=row["native_artifacts"]["predecode_clean_latent"],
        gaussian_artifact=row["native_artifacts"]["official_initial_gaussian"],
        candidate_seed=row["candidate"]["seed"],
    )
    expected = {
        "candidate": _candidate_binding(row),
        "population": _population_binding(row),
        "source": _source_binding(row),
        "rollout": _rollout_binding(
            row, calibration_bundle["checkpoint_tree_sha256"]
        ),
        "calibration": _calibration_binding(row, calibration_bundle),
        "prompts": prompt,
        "artifacts": coordinate,
    }
    for name, value in expected.items():
        if score[name] != value:
            raise PairV5NativeRV2VActionScoreError(
                f"score receipt {name} differs from sealed runtime context"
            )
    for path_field, sha_field, section in (
        ("source_video_path", "source_video_sha256_recomputed", score["source"]),
        ("pair_receipt_path", "pair_receipt_file_sha256", score["rollout"]),
        ("native_receipt_path", "native_receipt_file_sha256", score["rollout"]),
        ("clean_latent_path", "clean_latent_artifact_sha256", score["artifacts"]),
        ("official_gaussian_path", "official_gaussian_artifact_sha256", score["artifacts"]),
    ):
        path = _plain_file(section[path_field], label=path_field)
        if file_sha256(path) != section[sha_field]:
            raise PairV5NativeRV2VActionScoreError(f"{path_field} changed after scoring")
    return score


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise PairV5NativeRV2VActionScoreError(f"refusing to overwrite {path}")
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    os.chmod(path, 0o400)
    return file_sha256(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-spec", required=True)
    parser.add_argument("--expected-population-spec-sha256", required=True)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--t2v-bank-spec", required=True)
    parser.add_argument("--expected-t2v-bank-spec-sha256", required=True)
    parser.add_argument("--t2v-bank-receipt", required=True)
    parser.add_argument("--expected-t2v-bank-receipt-sha256", required=True)
    parser.add_argument("--t2v-score-root", required=True)
    parser.add_argument("--t2v-calibration-root", required=True)
    parser.add_argument(
        "--expected-t2v-calibration-receipt-sha256", required=True
    )
    parser.add_argument("--expected-t2v-preregistration-sha256", required=True)
    parser.add_argument("--formal-v3-method-root", required=True)
    parser.add_argument(
        "--formal-v3-source-revision",
        default=formal_v3_compat.PINNED_SOURCE_REVISION,
    )
    parser.add_argument("--formal-v3-source-archive-sha256", required=True)
    parser.add_argument("--checkpoint-tree-sha256", required=True)
    parser.add_argument("--group-id", choices=("sp4-a", "sp4-b"), required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-bernini-commit",
        default=native_generation.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=native_generation.legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--ack-action-metric-is-not-action-success", action="store_true")
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_population_spec_sha256",
        "expected_t2v_bank_spec_sha256",
        "expected_t2v_bank_receipt_sha256",
        "expected_t2v_calibration_receipt_sha256",
        "expected_t2v_preregistration_sha256",
        "formal_v3_source_archive_sha256",
        "checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        _sha256(getattr(args, name), label=name)
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
        "formal_v3_source_revision",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
            raise PairV5NativeRV2VActionScoreError(
                f"{name} must be lowercase SHA-1"
            )
    if args.ack_action_metric_is_not_action_success is not True:
        raise PairV5NativeRV2VActionScoreError(
            "action-metric/non-success acknowledgement is mandatory"
        )
    if args.formal_v3_source_revision != formal_v3_compat.PINNED_SOURCE_REVISION:
        raise PairV5NativeRV2VActionScoreError(
            "formal calibration source must be exact d541801 v3"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    output = Path(args.output_dir)
    if (
        not output.is_absolute()
        or output == Path("/")
        or output.exists()
        or output.is_symlink()
    ):
        raise PairV5NativeRV2VActionScoreError(
            "output must be a fresh absolute non-root directory"
        )

    legacy = native_generation.legacy
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise PairV5NativeRV2VActionScoreError(
            "pinned Bernini attention heads differ"
        )
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    # This expensive recomputation occurs before process-group/model creation.
    # A malformed/NO-GO calibration or rollout therefore fails before GPU
    # parameters are allocated.
    bundle = load_mainline_calibration_bundle(
        t2v_bank_spec=args.t2v_bank_spec,
        t2v_bank_spec_sha256=args.expected_t2v_bank_spec_sha256,
        t2v_bank_receipt=args.t2v_bank_receipt,
        t2v_bank_receipt_sha256=args.expected_t2v_bank_receipt_sha256,
        t2v_score_root=args.t2v_score_root,
        t2v_calibration_root=args.t2v_calibration_root,
        t2v_calibration_receipt_sha256=(
            args.expected_t2v_calibration_receipt_sha256
        ),
        t2v_preregistration_sha256=args.expected_t2v_preregistration_sha256,
        checkpoint_tree_sha256=args.checkpoint_tree_sha256,
        formal_v3_method_root=args.formal_v3_method_root,
        formal_v3_source_revision=args.formal_v3_source_revision,
        formal_v3_source_archive_sha256=args.formal_v3_source_archive_sha256,
        python_executable=__import__("sys").executable,
    )
    population_binding, bound_rows = load_native_group_population(
        population_spec_path=args.population_spec,
        population_spec_sha256=args.expected_population_spec_sha256,
        rollout_root=args.rollout_root,
        group_id=args.group_id,
        calibration_bundle=bundle,
    )

    import torch
    import torch.distributed as dist
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise PairV5NativeRV2VActionScoreError(
            "native frozen scorer requires four AUH ROCm GPUs"
        )
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=4)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            identity = native_generation.source_audit.validate_checkpoint_content(
                checkpoint, Path(args.checkpoint_content_manifest)
            )
            checkpoint_rows[0] = {"ok": True, "identity": identity}
        except Exception as error:
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_result = checkpoint_rows[0]
    if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get("ok") is not True:
        raise PairV5NativeRV2VActionScoreError(
            f"rank-zero checkpoint audit failed: {checkpoint_result}"
        )
    checkpoint_identity = dict(checkpoint_result["identity"])
    checkpoint_receipt_digest = object_sha256(checkpoint_identity)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
    try:
        freeze_certificate = native_generation.source_audit.model_freeze_certificate(
            renderer
        )
    except Exception as error:
        raise PairV5NativeRV2VActionScoreError(str(error)) from error
    frozen_mace_runtime.checkpoint_content_binding(
        checkpoint_identity, freeze_certificate
    )
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise PairV5NativeRV2VActionScoreError(
            "global MACE requires frozen transformer_1 only"
        )
    if any(parameter.requires_grad for parameter in renderer.parameters()):
        raise PairV5NativeRV2VActionScoreError(
            "frozen native scorer contains trainable parameters"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    sigma = torch.tensor([PILOT_SIGMA], dtype=torch.float32, device=device)
    phase_commitment = frozen_mace_runtime.diagnostic_phase_commitment()

    if distributed.rank == 0:
        output.mkdir(parents=False)
    dist.barrier()
    scored_receipts: list[dict[str, Any]] = []
    safe_records: list[dict[str, Any]] = []
    prompt_digest_by_cell: dict[str, str] = {}
    try:
        rows_by_cell: dict[str, list[dict[str, Any]]] = {}
        for row in bound_rows:
            rows_by_cell.setdefault(row["cell"]["calibration_group_id"], []).append(row)
        for cell_id, cell_rows in rows_by_cell.items():
            prompt_binding = prompt_binding_from_cell(
                cell_rows[0]["cell"], prompt_cleaner=prompt_clean
            )
            prompts = prompt_binding["prompt_by_branch"]
            prompt_digest_by_cell[cell_id] = prompt_binding["prompt_registry_digest"]
            conditions = frozen_mace_runtime._encode_prompt_bank(
                renderer, tokenizer, prompts, device=device
            )
            scorer = frozen_mace_runtime.NativeExact40FrozenBerniniT2VScorer(
                diffusion,
                transformer,
                prompts,
                conditions,
                frozen_model_receipt_digest=checkpoint_receipt_digest,
                model_id="transformer_1",
            )
            for row in cell_rows:
                candidate_id = row["candidate"]["candidate_id"]
                clean_cpu = frozen_mace_runtime._load_exact81_tensor(
                    row["native_artifacts"]["predecode_clean_latent"],
                    key="normalized_clean_latent",
                    label=f"{candidate_id} native RV2V clean latent",
                )
                epsilon_cpu = frozen_mace_runtime._load_exact81_tensor(
                    row["native_artifacts"]["official_initial_gaussian"],
                    key="official_initial_gaussian",
                    label=f"{candidate_id} candidate-own official Gaussian",
                )
                frozen_mace_runtime.verify_native_tensor_value_identity(
                    epsilon_cpu,
                    row["native_artifacts"]["official_initial_gaussian"],
                    label=f"{candidate_id} candidate-own official Gaussian",
                )
                clean = clean_cpu.to(device=device).contiguous()
                epsilon = epsilon_cpu.to(device=device).contiguous()
                if clean.shape != epsilon.shape:
                    raise PairV5NativeRV2VActionScoreError(
                        "candidate clean/Gaussian geometry differs"
                    )
                result = native_bridge.score_frozen_t2v_action_energy(
                    clean,
                    epsilon,
                    sigma,
                    prompts,
                    scorer,
                    phase_commitment,
                    registered_phase_weight_digest=phase_commitment[
                        "registration_digest"
                    ],
                )
                try:
                    freeze_after = native_generation.source_audit.model_freeze_certificate(
                        renderer
                    )
                except Exception as error:
                    raise PairV5NativeRV2VActionScoreError(str(error)) from error
                if freeze_after != freeze_certificate or any(
                    parameter.requires_grad for parameter in renderer.parameters()
                ):
                    raise PairV5NativeRV2VActionScoreError(
                        "frozen renderer changed during native candidate scoring"
                    )
                packet = scorer.last_packet_receipt
                if not isinstance(packet, Mapping):
                    raise PairV5NativeRV2VActionScoreError(
                        "frozen scorer emitted no exact40 packet receipt"
                    )
                receipt = make_score_receipt(
                    row=row,
                    calibration_bundle=bundle,
                    prompt_binding=prompt_binding,
                    clean=clean,
                    epsilon=epsilon,
                    sigma=sigma,
                    score=result,
                    scorer_packet_receipt=packet,
                    checkpoint_identity=checkpoint_identity,
                    freeze_certificate=freeze_certificate,
                )
                # Replay all source/population/artifact bindings before publish.
                verify_score_against_context(
                    receipt,
                    row=row,
                    calibration_bundle=bundle,
                    clean=clean,
                    epsilon=epsilon,
                    sigma=sigma,
                )
                gathered: list[Any] = [None] * distributed.world_size
                dist.all_gather_object(gathered, receipt["receipt_digest"])
                if len(set(gathered)) != 1:
                    raise PairV5NativeRV2VActionScoreError(
                        "SP4 native score receipt digests differ"
                    )
                safe_record = safe_pareto_action_record(receipt)
                if distributed.rank == 0:
                    candidate_output = output / candidate_id
                    candidate_output.mkdir()
                    _write_create_only(
                        candidate_output / SCORE_FILENAME,
                        receipt,
                    )
                    _write_create_only(
                        candidate_output / SAFE_PARETO_ACTION_FILENAME,
                        safe_record,
                    )
                    scored_receipts.append(receipt)
                    safe_records.append(safe_record)
                del clean, epsilon, clean_cpu, epsilon_cpu, result
            del scorer, conditions

        if distributed.rank == 0:
            expected_order = population_binding["group_candidate_order"][args.group_id]
            if [item["candidate"]["candidate_id"] for item in scored_receipts] != expected_order:
                raise PairV5NativeRV2VActionScoreError(
                    "published native score order differs from sealed population"
                )
            group_unsigned = {
                "schema_version": GROUP_SCHEMA,
                "group_id": args.group_id,
                "visible_gpus": [0, 1, 2, 3]
                if args.group_id == "sp4-a"
                else [4, 5, 6, 7],
                "population_spec_raw_sha256": args.expected_population_spec_sha256,
                "candidate_order": expected_order,
                "candidate_order_digest": object_sha256(expected_order),
                "score_receipt_digests": [
                    item["receipt_digest"] for item in scored_receipts
                ],
                "safe_pareto_record_digests": [
                    item["record_digest"] for item in safe_records
                ],
                "candidate_own_x_sigma_sha256_by_candidate": {
                    item["candidate"]["candidate_id"]: item["artifacts"][
                        "candidate_own_x_sigma_tensor_sha256"
                    ]
                    for item in scored_receipts
                },
                "source_video_sha256_by_candidate": {
                    item["candidate"]["candidate_id"]: item["candidate"][
                        "source_video_sha256"
                    ]
                    for item in scored_receipts
                },
                "mainline_calibration_authorization_digest": bundle[
                    "authorization"
                ]["authorization_digest"],
                "formal_t2v_score_provenance_set_digest": bundle[
                    "formal_score_provenance_set_digest"
                ],
                "formal_t2v_score_schema": bundle["authorization"][
                    "formal_score_schema"
                ],
                "formal_t2v_score_scalar_definition": bundle["authorization"][
                    "formal_score_scalar_definition"
                ],
                "formal_v3_source_revision": bundle["authorization"][
                    "formal_v3_source_revision"
                ],
                "formal_v3_source_archive_sha256": bundle["authorization"][
                    "formal_v3_source_archive_sha256"
                ],
                "formal_v3_source_binding_digest": bundle["authorization"][
                    "formal_v3_source_binding_digest"
                ],
                "active_repository_action_scalar_consumed": False,
                "decimal_or_log1p_action_scalar_consumed": False,
                "family_mapping_set_digest": bundle["family_mapping_set_digest"],
                "calibration_receipt_digest": bundle["calibration"]["receipt_digest"],
                "checkpoint_tree_sha256": args.checkpoint_tree_sha256,
                "frozen_checkpoint_receipt_digest": checkpoint_receipt_digest,
                "prompt_registry_digest_by_cell": prompt_digest_by_cell,
                "score_coordinate": frozen_mace_runtime.schedule_coordinate_receipt(),
                "input_closure": NATIVE_SCORE_INPUT_CLOSURE,
                "safe_pareto_requires_identity_camera_temporal_metrics": True,
                "optimizer_authorized": False,
                "scientific_action_editing_claim": False,
                "method_source_revision": args.method_source_revision,
                "method_source_archive_sha256": args.method_source_archive_sha256,
                "bernini_revision": bernini_revision,
                "veomni_revision": veomni_revision,
            }
            group_receipt = {
                **group_unsigned,
                "receipt_digest": object_sha256(group_unsigned),
            }
            _write_create_only(
                output / GROUP_FILENAME.format(group_id=args.group_id),
                group_receipt,
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GROUP_FILENAME",
    "GROUP_SCHEMA",
    "NATIVE_SCORE_INPUT_CLOSURE",
    "PILOT_NATIVE_TIMESTEP",
    "PILOT_SCHEDULE_INDEX",
    "PILOT_SIGMA",
    "PairV5NativeRV2VActionScoreError",
    "SAFE_PARETO_ACTION_SCHEMA",
    "SAFE_PARETO_ACTION_FILENAME",
    "SCORE_FILENAME",
    "SCORE_SCHEMA",
    "apply_family_mapping",
    "bind_population_to_calibration",
    "calibration_cells",
    "candidate_coordinate_binding",
    "load_mainline_calibration_bundle",
    "load_native_group_population",
    "make_score_receipt",
    "prompt_binding_from_cell",
    "safe_pareto_action_record",
    "validate_calibration_receipt",
    "validate_family_mapping",
    "validate_safe_pareto_action_record",
    "validate_score_receipt",
    "verify_score_against_context",
]
