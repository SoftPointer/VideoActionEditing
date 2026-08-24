#!/usr/bin/env python3
"""Fail-closed, CPU-only admission for V10-A2 P0 qualification evidence.

This program does not decode video, load a foundation model or generator,
create an optimizer, run backward, or update parameters.  It validates the
byte-pinned qualification contract and the existing provisional64 evidence.
The checked-in v1 contract deliberately has no pinned perceptual, ancestry,
access-ledger, observer, completion-seal, or registrar artifacts, so its only
legal current decision is QUALIFICATION_NO_ABSTAIN.

The semantic validators for future target-blind perceptual and frozen-observer
receipts are included here so missing evidence cannot later be replaced by a
single self-attested PASS bit.  Such receipts still require byte pins in a new
preregistered authority and independent completion seals.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, NoReturn, Sequence

try:
    from . import v10a2_hierarchical_object_event_graph_preflight_v1 as main_v10a2
    from . import v10a2_p0_source_only_64_provisional_preflight_v1 as provisional
except ImportError:  # direct script/test import
    import v10a2_hierarchical_object_event_graph_preflight_v1 as main_v10a2
    import v10a2_p0_source_only_64_provisional_preflight_v1 as provisional


PREREG_SCHEMA = (
    "bernini-v10a2-p0-target-blind-perceptual-frozen-observer-"
    "qualification-prereg-v1"
)
RECEIPT_SCHEMA = (
    "bernini-v10a2-p0-target-blind-perceptual-frozen-observer-"
    "qualification-receipt-v1"
)
RECEIPT_SCHEMA_AUTHORITY = (
    "bernini-v10a2-p0-target-blind-perceptual-frozen-observer-"
    "qualification-receipt-schema-v1"
)
PERCEPTUAL_SCHEMA = (
    "bernini-v10a2-p0-target-blind-perceptual-evidence-receipt-v1"
)
OBSERVER_SCHEMA = (
    "bernini-v10a2-p0-frozen-observer-observability-evidence-receipt-v1"
)
COMPLETION_SEAL_SCHEMA = "bernini-v10a2-p0-qualification-external-completion-seal-v1"
QUALIFICATION_ID = (
    "v10a2_p0_target_blind_perceptual_frozen_observer_qualification_v1"
)
ONLY_STATUS = "QUALIFICATION_NO_ABSTAIN"
ONLY_DECISION = "ABSTAIN"

METHOD_ROOT = Path(__file__).resolve().parent
DEFAULT_PREREG_PATH = (
    METHOD_ROOT
    / "assets"
    / "v10a2_p0_target_blind_perceptual_frozen_observer_qualification_prereg_v1.json"
)
DEFAULT_RECEIPT_SCHEMA_PATH = (
    METHOD_ROOT
    / "assets"
    / "v10a2_p0_target_blind_perceptual_frozen_observer_qualification_receipt_schema_v1.json"
)
DEFAULT_V10A2_PREREG_PATH = (
    METHOD_ROOT / "assets" / "v10a2_hierarchical_object_event_graph_prereg_v1.json"
)
DEFAULT_REGISTRY_PATH = provisional.DEFAULT_REGISTRY_PATH
DEFAULT_ACTUAL_MANIFEST_PATH = provisional.DEFAULT_ACTUAL_MANIFEST_PATH

EXPECTED_PREREG_FILE_SHA256 = (
    "7c486303600d0d8dce951f61b50914c0e696b4983fda2c8c49429a2495405e93"
)
EXPECTED_PREREG_SELF_SHA256 = (
    "43b071d50ccd17ddca1935c3289adcfcc831af17fadaaea99ea6945dbadad164"
)
EXPECTED_RECEIPT_SCHEMA_FILE_SHA256 = (
    "6bcb2dcfc365a538c304c3199456e06df40620b50869cfb0f586398c961b0cb9"
)
EXPECTED_RECEIPT_SCHEMA_SELF_SHA256 = (
    "f049d734f00d83a115dc61a96f5bb1466634c3e7105ec6d2edf069c4bf8c83ee"
)
EXPECTED_V10A2_PREREG_FILE_SHA256 = (
    "e8411902ab4c58199025bc580eff861f0fd71184fec6a2bce4643ab287297fc3"
)
EXPECTED_V10A2_PREREG_SELF_SHA256 = (
    "721f9a47f300c985fb3a9aa7fa98233fd2f037dba2c00bc31396457fb855938b"
)

STRATA = provisional.STRATA
FOUNDATION_MODELS = ("sam2", "dinov2", "cotracker", "vjepa2")
REQUIRED_ARTIFACT_IDS = (
    "V10A2_FOUNDATION_MODEL_BINDING_AND_REMOTE_WEIGHT_REVALIDATION_RECEIPT_MISSING",
    "SOURCE_ANCESTOR_PROVENANCE_RECEIPT_MISSING",
    "TARGET_BLIND_ACCESS_LEDGER_AND_INDEPENDENT_AUDITOR_SEAL_MISSING",
    "TARGET_BLIND_PERCEPTUAL_QUALIFICATION_RECEIPT_MISSING",
    "TARGET_BLIND_PERCEPTUAL_EXTERNAL_COMPLETION_SEAL_MISSING",
    "FROZEN_OBSERVER_PARENT_PART_INTERACTION_QUALIFICATION_RECEIPT_MISSING",
    "FROZEN_OBSERVER_EXTERNAL_COMPLETION_SEAL_MISSING",
    "QUALIFICATION_EXECUTOR_IMMUTABLE_SNAPSHOT_MISSING",
    "SOURCE_ONLY_HELD16_OR_PREREGISTERED_CROSSFIT_AUTHORITY_MISSING",
    "OFFICIAL_SOURCE_ONLY_REGISTRAR_RELEASE_MISSING",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class V10A2P0QualificationError(RuntimeError):
    """Raised when a qualification authority or receipt is not admissible."""


def fail(message: str) -> NoReturn:
    raise V10A2P0QualificationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise V10A2P0QualificationError(
            "value is not canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    fail(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"JSON evidence must be one regular non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except V10A2P0QualificationError:
        raise
    except Exception as error:
        raise V10A2P0QualificationError(f"cannot parse {path}: {error}") from error
    if type(value) is not dict:
        fail(f"JSON root must be an object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        fail(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        fail(f"{label} must be an array")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be a finite number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        fail(f"{label} must be a finite number")
    return result


def _verify_self_hash(value: Mapping[str, Any], key: str, label: str) -> str:
    expected = _sha256(value.get(key), f"{label}.{key}")
    payload = dict(value)
    payload.pop(key, None)
    actual = object_sha256(payload)
    if actual != expected:
        fail(f"{label} self hash differs")
    return actual


def _verify_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        fail(f"{label} schema differs")


def validate_receipt_schema_authority(
    value: Mapping[str, Any], *, observed_file_sha256: str
) -> Mapping[str, Any]:
    if observed_file_sha256 != EXPECTED_RECEIPT_SCHEMA_FILE_SHA256:
        fail("qualification receipt schema bytes differ")
    digest = _verify_self_hash(value, "receipt_schema_sha256", "receipt schema")
    if digest != EXPECTED_RECEIPT_SCHEMA_SELF_SHA256:
        fail("qualification receipt schema self hash differs from pin")
    if value.get("$id") != RECEIPT_SCHEMA_AUTHORITY:
        fail("qualification receipt schema identity differs")
    if value.get("type") != "object" or value.get("additionalProperties") is not False:
        fail("qualification receipt schema is not closed")
    required = _array(value.get("required"), "receipt schema.required")
    for key in (
        "status",
        "decision",
        "execution",
        "frozen_base_b0",
        "target_teacher_boundary",
        "permissions",
        "blockers",
        "receipt_sha256",
    ):
        if key not in required:
            fail(f"qualification receipt schema omits hard field: {key}")
    return value


def validate_qualification_prereg(
    value: Mapping[str, Any], *, observed_file_sha256: str
) -> Mapping[str, Any]:
    if observed_file_sha256 != EXPECTED_PREREG_FILE_SHA256:
        fail("qualification preregistration bytes differ")
    _verify_exact_keys(
        value,
        {
            "schema_version",
            "qualification_id",
            "status",
            "scope",
            "input_pins",
            "frozen_base_b0",
            "target_teacher_boundary",
            "target_blind_perceptual_qualification",
            "frozen_observer_observability_qualification",
            "evidence_admission",
            "current_authorization",
            "claim_boundary",
            "qualification_prereg_sha256",
        },
        "qualification preregistration",
    )
    if value.get("schema_version") != PREREG_SCHEMA:
        fail("qualification preregistration schema differs")
    if value.get("qualification_id") != QUALIFICATION_ID:
        fail("qualification identity differs")
    if value.get("status") != "DESIGN_ONLY_QUALIFICATION_NO_ABSTAIN":
        fail("qualification preregistration must remain design-only NO/ABSTAIN")
    digest = _verify_self_hash(
        value, "qualification_prereg_sha256", "qualification preregistration"
    )
    if digest != EXPECTED_PREREG_SELF_SHA256:
        fail("qualification preregistration self hash differs from pin")

    scope = _mapping(value.get("scope"), "scope")
    if scope != {
        "purpose": "qualify the fixed provisional64 source-only pool before any P0 slot pretraining",
        "read_only_qualification": True,
        "training": False,
        "generator_execution": False,
        "generator_update": False,
        "binder_update": False,
        "slot_update": False,
        "gpu_launch_authority": False,
        "synthetic_or_mock_evidence_can_pass": False,
    }:
        fail("qualification read-only/no-training scope differs")

    pins = _mapping(value.get("input_pins"), "input_pins")
    if pins.get("v10a2_prereg") != {
        "relative_path": "assets/v10a2_hierarchical_object_event_graph_prereg_v1.json",
        "file_sha256": EXPECTED_V10A2_PREREG_FILE_SHA256,
        "self_sha256": EXPECTED_V10A2_PREREG_SELF_SHA256,
    }:
        fail("V10-A2 main preregistration pin differs")
    if pins.get("provisional_registry") != {
        "relative_path": "assets/v10a2_p0_source_only_64_provisional_v1.json",
        "file_sha256": provisional.EXPECTED_REGISTRY_FILE_SHA256,
        "self_sha256": provisional.EXPECTED_REGISTRY_SELF_SHA256,
        "candidate_count": 64,
        "strata_count": 4,
        "rows_per_stratum": 16,
    }:
        fail("provisional64 pin/count differs")
    if pins.get("actual_exclusion_manifest") != {
        "relative_path": "assets/target_factorized_soft_ot_graph_teacher_manifest_v5_r1b.json",
        "file_sha256": provisional.EXPECTED_ACTUAL_MANIFEST_FILE_SHA256,
        "self_sha256": provisional.EXPECTED_ACTUAL_MANIFEST_SELF_SHA256,
        "pair_count": 16,
        "media_count": 32,
        "used_only_by_external_exclusion_auditor": True,
        "released_to_p0_training": False,
    }:
        fail("actual exclusion manifest pin/firewall differs")
    if pins.get("qualification_receipt_schema") != {
        "relative_path": "assets/v10a2_p0_target_blind_perceptual_frozen_observer_qualification_receipt_schema_v1.json",
        "file_sha256": EXPECTED_RECEIPT_SCHEMA_FILE_SHA256,
        "self_sha256": EXPECTED_RECEIPT_SCHEMA_SELF_SHA256,
    }:
        fail("qualification receipt schema pin differs")

    b0 = _mapping(value.get("frozen_base_b0"), "frozen_base_b0")
    if b0 != {
        "arm_id": "B0_FROZEN_BASE",
        "first_class_future_arm_preserved": True,
        "replaced_by_historical_p0_or_metadata_row": False,
        "executed_during_this_qualification": False,
        "generator_loaded_during_this_qualification": False,
        "capture_calls": 0,
        "generator_forward_calls": 0,
        "parameter_updates": 0,
        "future_mev_40_cell_action_passive_parity_still_required": True,
        "future_factorial_36_cell_b0_closure_still_required": True,
        "future_total_b0_action_cell_count": 76,
        "future_bit_exact_80_transformer_41_latent_and_decode_parity_still_required": True,
        "future_quality_reference_and_fallback_still_required": True,
    }:
        fail("Frozen Base B0 was weakened or replaced")

    boundary = _mapping(value.get("target_teacher_boundary"), "target_teacher_boundary")
    if boundary != {
        "selected_resolution": "A_SELF_GENERATED_ROUTE_REWARD_TARGET_EXTERNAL_EVALUATION_ONLY",
        "p0_source_only_target_teacher_read_count": 0,
        "p1_development_target_graph_role": "binder_only_structured_teacher_if_separately_authorized",
        "p1_target_rgb_latent_hidden_allowed": False,
        "p2_locked_actual_graph_role": "external_evaluator_only_after_threshold_and_checkpoint_seal",
        "p3_route_process_target_teacher_read_count": 0,
        "p3_route_process_actual_locked_graph_read_count": 0,
        "p3_reward_source": "frozen_binder_same_state_self_generated_action_minus_mean_noop_a_noop_b_middle_layer_only",
        "actual_target_graph_used_in_generator_loss": False,
        "actual_locked_graph_visible_to_external_evaluator_only": True,
        "any_counter_or_role_violation": ONLY_STATUS,
    }:
        fail("target-teacher/P3 route firewall differs")

    perceptual = _mapping(
        value.get("target_blind_perceptual_qualification"),
        "target_blind_perceptual_qualification",
    )
    if perceptual.get("receipt_schema_version") != PERCEPTUAL_SCHEMA:
        fail("perceptual evidence schema differs")
    roles = _mapping(perceptual.get("target_blind_roles"), "target_blind_roles")
    for key in (
        "auditor_must_be_disjoint_from_training_process",
        "byte_pinned_process_access_ledger_required",
        "independent_completion_seal_required",
    ):
        if roles.get(key) is not True:
            fail(f"target-blind role gate differs: {key}")
    for key in (
        "auditor_release_contains_target_features",
        "auditor_release_contains_target_neighbor_identities_or_scores",
        "candidate_cluster_commitments_released_to_p0_training",
    ):
        if roles.get(key) is not False:
            fail(f"target-blind release leaks protected evidence: {key}")
    sampling = _mapping(perceptual.get("media_and_sampling"), "media_and_sampling")
    if (
        sampling.get("candidate_media_count"),
        sampling.get("actual_exclusion_media_count"),
        sampling.get("fixed_frame_grid_count_per_media"),
        sampling.get("candidate_decoded_frame_count"),
        sampling.get("actual_decoded_frame_count"),
    ) != (64, 32, 32, 2048, 1024):
        fail("perceptual media/frame-grid count differs")
    near = _mapping(perceptual.get("near_duplicate_rule"), "near_duplicate_rule")
    if (
        near.get("phash_hamming_max"),
        near.get("phash_close_aligned_frame_min"),
        near.get("aligned_dinov2_cosine_median_min"),
        near.get("vjepa2_video_cosine_min"),
    ) != (6, 8, 0.95, 0.97):
        fail("perceptual near-duplicate thresholds differ")
    if near.get("threshold_equality_is_duplicate") is not True:
        fail("perceptual threshold equality must reject")
    comparison = _mapping(
        perceptual.get("comparison_and_cluster_gate"),
        "comparison_and_cluster_gate",
    )
    for key, expected in {
        "candidate_candidate_pair_count": 2016,
        "candidate_actual_pair_count": 2048,
        "total_pair_count": 4064,
        "candidate_candidate_near_duplicate_edge_count_max": 0,
        "candidate_actual_near_duplicate_edge_count_max": 0,
        "candidate_unique_perceptual_cluster_count": 64,
        "candidate_ancestor_unique_count": 64,
        "candidate_actual_ancestor_overlap_count": 0,
    }.items():
        if comparison.get(key) != expected:
            fail(f"perceptual comparison/cluster gate differs: {key}")

    observer = _mapping(
        value.get("frozen_observer_observability_qualification"),
        "frozen_observer_observability_qualification",
    )
    if observer.get("receipt_schema_version") != OBSERVER_SCHEMA:
        fail("observer evidence schema differs")
    if observer.get("foundation_models") != list(FOUNDATION_MODELS):
        fail("observer foundation model list differs")
    source_only = _mapping(observer.get("source_only_access"), "source_only_access")
    if source_only != {
        "candidate_source_media_count": 64,
        "target_media_read_count": 0,
        "prompt_caption_instruction_action_or_family_read_count": 0,
        "actual_manifest_read_by_observer_process": False,
        "real_observer_outputs_required": True,
        "synthetic_or_mock_outputs_allowed": False,
        "metadata_to_model_input_count": 0,
    }:
        fail("observer source-only access firewall differs")
    closure = _mapping(observer.get("foundation_closure"), "foundation_closure")
    for key in (
        "eval_mode_required",
        "weight_config_preprocessor_source_and_environment_sha_required",
        "parameter_buffer_version_pointer_and_content_closure_required",
    ):
        if closure.get(key) is not True:
            fail(f"observer closure gate differs: {key}")
    for key in (
        "autograd_enabled",
        "optimizer_created",
        "legacy_v3_mechanics_receipt_alone_satisfies_v10a2",
    ):
        if closure.get(key) is not False:
            fail(f"observer no-training/provenance gate differs: {key}")
    for key in (
        "requires_grad_true_count",
        "gradient_tensor_count",
        "backward_calls",
        "parameter_updates",
    ):
        if closure.get(key) != 0:
            fail(f"observer update counter must be zero: {key}")
    calls = _mapping(observer.get("expected_real_call_counts"), "expected_real_call_counts")
    if calls != {
        "decoded_candidate_videos": 64,
        "decoded_view_sequences": 128,
        "decoded_tau_per_video": 8,
        "sam2_keyframe_calls": 1024,
        "dinov2_keyframe_calls": 1024,
        "cotracker_video_calls": 128,
        "vjepa2_video_calls": 128,
    }:
        fail("observer real call counts differ")
    parent = _mapping(observer.get("parent_slot_observability_gate"), "parent gate")
    if (
        parent.get("cross_view_parent_track_idf1_min"),
        parent.get("cross_view_parent_mask_iou_median_min"),
        parent.get("mean_dustbin_mass_max"),
        parent.get("single_tau_dustbin_mass_max"),
    ) != (0.75, 0.55, 0.2, 0.35):
        fail("observer parent/dustbin hard thresholds differ")
    part = _mapping(observer.get("part_slot_observability_gate"), "part gate")
    if (
        part.get("part_parent_relative_mass_min"),
        part.get("part_mass_outside_parent_max"),
        part.get("cross_view_part_mask_iou_min"),
    ) != (0.02, 0.0, 0.35):
        fail("observer part hard thresholds differ")
    interaction = _mapping(observer.get("interaction_observability_gate"), "interaction gate")
    if (
        interaction.get("cross_view_dynamic_edge_f1_min"),
        interaction.get("event_phase_error_tau_max"),
        interaction.get("positive_dynamic_edge_distinct_tau_min"),
    ) != (0.8, 1, 3):
        fail("observer interaction hard thresholds differ")

    safety = _mapping(observer.get("generator_and_training_safety"), "generator safety")
    expected_safety = {
        "generator_import_count": 0,
        "generator_loaded": False,
        "generator_forward_calls": 0,
        "generator_capture_calls": 0,
        "route_or_lora_loaded": False,
        "binder_loaded": False,
        "slot_model_loaded": False,
        "optimizer_created": False,
        "backward_calls": 0,
        "parameter_updates": 0,
        "b0_executed": False,
        "b0_replaced": False,
    }
    if safety != expected_safety:
        fail("observer qualification loaded/trained generator, binder, or slot model")

    admission = _mapping(value.get("evidence_admission"), "evidence_admission")
    if admission.get("qualification_runner_mode") != "cpu_readonly_receipt_admission":
        fail("qualification runner mode differs")
    if admission.get("arbitrary_cli_receipt_can_override_preregistered_pin") is not False:
        fail("arbitrary CLI evidence cannot override preregistered pins")
    if admission.get("rehashing_can_create_authority") is not False:
        fail("rehashing cannot create qualification authority")
    artifacts = _array(admission.get("required_artifacts"), "required_artifacts")
    if [row.get("id") for row in artifacts if type(row) is dict] != list(REQUIRED_ARTIFACT_IDS):
        fail("qualification required-artifact registry differs")
    for row in artifacts:
        if row != {
            "id": row["id"],
            "present": False,
            "path": None,
            "file_sha256": None,
        }:
            fail(f"v1 qualification artifact must remain explicitly missing: {row.get('id')}")

    authorization = _mapping(value.get("current_authorization"), "current_authorization")
    if authorization != {
        "decision": ONLY_DECISION,
        "qualification_evidence_complete": False,
        "official_source_only_registry": False,
        "p0_slot_pretraining_authorized": False,
        "binder_training_authorized": False,
        "generator_training_authorized": False,
        "gpu_launch_authorized": False,
        "parameter_updates_authorized": False,
    }:
        fail("qualification v1 authorization must remain ABSTAIN / all false")
    return value


def _registry_rows(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        _mapping(row, f"registry.rows[{ordinal}]")
        for ordinal, row in enumerate(_array(registry.get("rows"), "registry.rows"))
    ]


def _validate_model_closure_rows(value: Any, label: str) -> None:
    rows = _array(value, label)
    if len(rows) != 4:
        fail(f"{label} must contain four real frozen models")
    names: list[str] = []
    for raw in rows:
        row = _mapping(raw, f"{label}[]")
        _verify_exact_keys(
            row,
            {
                "name",
                "weight_closure_sha256",
                "config_preprocess_source_closure_sha256",
                "environment_closure_sha256",
                "state_before_sha256",
                "state_after_sha256",
                "eval_mode",
                "requires_grad_true_count",
                "gradient_tensor_count",
                "real_model",
            },
            f"{label}[]",
        )
        name = row.get("name")
        if name not in FOUNDATION_MODELS:
            fail(f"{label} contains an unregistered model")
        names.append(name)
        for key in (
            "weight_closure_sha256",
            "config_preprocess_source_closure_sha256",
            "environment_closure_sha256",
            "state_before_sha256",
            "state_after_sha256",
        ):
            _sha256(row.get(key), f"{label}.{name}.{key}")
        if row.get("state_before_sha256") != row.get("state_after_sha256"):
            fail(f"{label}.{name} state changed")
        if row.get("eval_mode") is not True or row.get("real_model") is not True:
            fail(f"{label}.{name} is not a real eval-mode model")
        if row.get("requires_grad_true_count") != 0 or row.get("gradient_tensor_count") != 0:
            fail(f"{label}.{name} has trainable/gradient tensors")
    if tuple(names) != FOUNDATION_MODELS:
        fail(f"{label} model order/identity differs")


def validate_perceptual_evidence_receipt(
    value: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    observed_file_sha256: str,
    expected_file_sha256: str,
    expected_model_binding_sha256: str,
    expected_ancestor_receipt_sha256: str,
    expected_access_ledger_sha256: str,
    expected_completion_seal_sha256: str,
) -> Mapping[str, Any]:
    """Validate a future externally pinned perceptual receipt.

    This function cannot create an evidence pin or completion authority.  All
    expected hashes must come from a later immutable preregistration.
    """

    if observed_file_sha256 != _sha256(expected_file_sha256, "expected perceptual file"):
        fail("perceptual evidence receipt bytes differ from preregistered pin")
    _verify_exact_keys(
        value,
        {
            "schema_version",
            "qualification_id",
            "decision",
            "pins",
            "provenance",
            "target_blind",
            "sampling",
            "thresholds",
            "comparisons",
            "model_closure",
            "safety",
            "raw_release",
            "rows",
            "receipt_sha256",
        },
        "perceptual evidence receipt",
    )
    if value.get("schema_version") != PERCEPTUAL_SCHEMA:
        fail("perceptual evidence receipt schema differs")
    if value.get("qualification_id") != QUALIFICATION_ID or value.get("decision") != "PASS":
        fail("perceptual evidence receipt does not explicitly PASS this qualification")
    _verify_self_hash(value, "receipt_sha256", "perceptual evidence receipt")

    pins = _mapping(value.get("pins"), "perceptual.pins")
    if pins != {
        "qualification_prereg_self_sha256": EXPECTED_PREREG_SELF_SHA256,
        "provisional_registry_file_sha256": provisional.EXPECTED_REGISTRY_FILE_SHA256,
        "provisional_registry_self_sha256": provisional.EXPECTED_REGISTRY_SELF_SHA256,
        "actual_manifest_file_sha256": provisional.EXPECTED_ACTUAL_MANIFEST_FILE_SHA256,
        "actual_manifest_self_sha256": provisional.EXPECTED_ACTUAL_MANIFEST_SELF_SHA256,
    }:
        fail("perceptual evidence input pins differ")
    provenance = _mapping(value.get("provenance"), "perceptual.provenance")
    _verify_exact_keys(
        provenance,
        {
            "runner_source_sha256",
            "model_binding_receipt_sha256",
            "ancestor_provenance_receipt_sha256",
            "target_blind_access_ledger_sha256",
            "external_completion_seal_sha256",
            "real_models_executed",
            "synthetic_or_mock_features",
        },
        "perceptual.provenance",
    )
    _sha256(provenance.get("runner_source_sha256"), "perceptual runner source")
    expected_links = {
        "model_binding_receipt_sha256": expected_model_binding_sha256,
        "ancestor_provenance_receipt_sha256": expected_ancestor_receipt_sha256,
        "target_blind_access_ledger_sha256": expected_access_ledger_sha256,
        "external_completion_seal_sha256": expected_completion_seal_sha256,
    }
    for key, expected in expected_links.items():
        if provenance.get(key) != _sha256(expected, f"expected {key}"):
            fail(f"perceptual provenance link differs: {key}")
    if provenance.get("real_models_executed") is not True:
        fail("perceptual evidence did not execute real models")
    if provenance.get("synthetic_or_mock_features") is not False:
        fail("synthetic/mock perceptual evidence cannot pass")

    target_blind = _mapping(value.get("target_blind"), "perceptual.target_blind")
    if target_blind != {
        "auditor_process_disjoint_from_training": True,
        "training_process_target_media_reads": 0,
        "training_process_target_feature_reads": 0,
        "released_target_feature_bytes": 0,
        "released_target_neighbor_identities": 0,
        "released_target_neighbor_scores": 0,
        "released_candidate_cluster_commitments_to_training": 0,
        "protected_feature_files_released": False,
        "access_ledger_complete": True,
    }:
        fail("perceptual receipt violates target-blind access/release")
    sampling = _mapping(value.get("sampling"), "perceptual.sampling")
    if sampling != {
        "candidate_media_count": 64,
        "actual_media_count": 32,
        "frame_grid_count_per_media": 32,
        "candidate_decoded_frame_count": 2048,
        "actual_decoded_frame_count": 1024,
        "candidate_media_sha256_revalidated_count": 64,
        "actual_media_sha256_revalidated_count": 32,
        "decode_failure_count": 0,
        "missing_transform_count": 0,
        "nonfinite_measure_count": 0,
    }:
        fail("perceptual sampling/decode closure differs")
    thresholds = _mapping(value.get("thresholds"), "perceptual.thresholds")
    if thresholds != {
        "phash_hamming_max_inclusive": 6,
        "phash_close_aligned_frame_min": 8,
        "dinov2_aligned_median_cosine_min_inclusive": 0.95,
        "vjepa2_video_cosine_min_inclusive": 0.97,
        "threshold_equality_is_duplicate": True,
        "temporal_directions": ["forward", "reverse"],
        "nuisance_alignments": ["trim", "time_shift", "reencode", "resize", "crop", "horizontal_flip"],
    }:
        fail("perceptual receipt threshold/transform contract differs")
    comparisons = _mapping(value.get("comparisons"), "perceptual.comparisons")
    if comparisons != {
        "candidate_candidate_pair_count": 2016,
        "candidate_actual_pair_count": 2048,
        "total_pair_count": 4064,
        "candidate_candidate_near_duplicate_edge_count": 0,
        "candidate_actual_near_duplicate_edge_count": 0,
        "candidate_unique_perceptual_cluster_count": 64,
        "candidate_ancestor_unique_count": 64,
        "candidate_candidate_ancestor_collision_count": 0,
        "candidate_actual_ancestor_overlap_count": 0,
        "missing_pair_comparison_count": 0,
    }:
        fail("perceptual comparison/cluster/ancestor result does not pass")
    _validate_model_closure_rows(value.get("model_closure"), "perceptual.model_closure")
    safety = _mapping(value.get("safety"), "perceptual.safety")
    if safety != {
        "autograd_enabled": False,
        "optimizer_created": False,
        "backward_calls": 0,
        "parameter_updates": 0,
        "generator_loaded": False,
        "generator_forward_calls": 0,
        "b0_executed": False,
        "b0_replaced": False,
    }:
        fail("perceptual qualification performed training/generator/B0 work")
    raw = _mapping(value.get("raw_release"), "perceptual.raw_release")
    if raw != {
        "raw_actual_frames_released": False,
        "raw_actual_embeddings_released": False,
        "raw_actual_neighbor_scores_released": False,
        "raw_candidate_embeddings_released": False,
        "zeroization_verified": True,
    }:
        fail("perceptual receipt releases raw/protected evidence")

    expected_registry_rows = _registry_rows(registry)
    rows = _array(value.get("rows"), "perceptual.rows")
    if len(rows) != 64:
        fail("perceptual evidence must contain exactly 64 candidate rows")
    clusters: list[str] = []
    for ordinal, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"perceptual.rows[{ordinal}]")
        if row != {
            "ordinal": ordinal,
            "uuid": expected_registry_rows[ordinal]["uuid"],
            "source_media_sha256": expected_registry_rows[ordinal]["source_media_sha256"],
            "opaque_cluster_commitment": row.get("opaque_cluster_commitment"),
            "candidate_near_duplicate_count": 0,
            "actual_near_duplicate_count": 0,
            "ancestor_overlap": False,
            "status": "PASS",
        }:
            fail(f"perceptual candidate row identity/result differs: {ordinal}")
        clusters.append(
            _sha256(row.get("opaque_cluster_commitment"), f"perceptual row {ordinal} cluster")
        )
    if len(set(clusters)) != 64:
        fail("perceptual candidate clusters are not unique")
    return value


def validate_observer_evidence_receipt(
    value: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    observed_file_sha256: str,
    expected_file_sha256: str,
    expected_model_binding_sha256: str,
    expected_completion_seal_sha256: str,
) -> Mapping[str, Any]:
    """Validate a future externally pinned real frozen-observer receipt."""

    if observed_file_sha256 != _sha256(expected_file_sha256, "expected observer file"):
        fail("observer evidence receipt bytes differ from preregistered pin")
    _verify_exact_keys(
        value,
        {
            "schema_version",
            "qualification_id",
            "decision",
            "pins",
            "provenance",
            "source_only",
            "sampling",
            "call_counts",
            "model_closure",
            "safety",
            "aggregate",
            "rows",
            "receipt_sha256",
        },
        "observer evidence receipt",
    )
    if value.get("schema_version") != OBSERVER_SCHEMA:
        fail("observer evidence receipt schema differs")
    if value.get("qualification_id") != QUALIFICATION_ID or value.get("decision") != "PASS":
        fail("observer evidence receipt does not explicitly PASS this qualification")
    _verify_self_hash(value, "receipt_sha256", "observer evidence receipt")
    pins = _mapping(value.get("pins"), "observer.pins")
    if pins != {
        "qualification_prereg_self_sha256": EXPECTED_PREREG_SELF_SHA256,
        "provisional_registry_file_sha256": provisional.EXPECTED_REGISTRY_FILE_SHA256,
        "provisional_registry_self_sha256": provisional.EXPECTED_REGISTRY_SELF_SHA256,
    }:
        fail("observer evidence input pins differ")
    provenance = _mapping(value.get("provenance"), "observer.provenance")
    if set(provenance) != {
        "runner_source_sha256",
        "model_binding_receipt_sha256",
        "external_completion_seal_sha256",
        "real_models_executed",
        "synthetic_or_mock_outputs",
    }:
        fail("observer provenance schema differs")
    _sha256(provenance.get("runner_source_sha256"), "observer runner source")
    if provenance.get("model_binding_receipt_sha256") != _sha256(
        expected_model_binding_sha256, "expected observer model binding"
    ):
        fail("observer model-binding receipt differs")
    if provenance.get("external_completion_seal_sha256") != _sha256(
        expected_completion_seal_sha256, "expected observer completion seal"
    ):
        fail("observer completion-seal link differs")
    if provenance.get("real_models_executed") is not True:
        fail("observer receipt did not execute real frozen models")
    if provenance.get("synthetic_or_mock_outputs") is not False:
        fail("synthetic/mock observer outputs cannot pass")
    source_only = _mapping(value.get("source_only"), "observer.source_only")
    if source_only != {
        "candidate_source_media_count": 64,
        "target_media_read_count": 0,
        "actual_manifest_read_count": 0,
        "prompt_caption_instruction_action_or_family_read_count": 0,
        "metadata_to_model_input_count": 0,
    }:
        fail("observer qualification is not source-only")
    sampling = _mapping(value.get("sampling"), "observer.sampling")
    if sampling != {
        "view_ids": ["reference", "evaluation"],
        "tau_per_view": 8,
        "temporal_index_overlap_count": 0,
        "geometry_inverse_map_missing_count": 0,
        "decoded_candidate_videos": 64,
        "decoded_view_sequences": 128,
        "decode_failure_count": 0,
    }:
        fail("observer dual-view sampling closure differs")
    calls = _mapping(value.get("call_counts"), "observer.call_counts")
    if calls != {
        "sam2_keyframe_calls": 1024,
        "dinov2_keyframe_calls": 1024,
        "cotracker_video_calls": 128,
        "vjepa2_video_calls": 128,
    }:
        fail("observer real foundation call counts differ")
    _validate_model_closure_rows(value.get("model_closure"), "observer.model_closure")
    safety = _mapping(value.get("safety"), "observer.safety")
    if safety != {
        "autograd_enabled": False,
        "optimizer_created": False,
        "backward_calls": 0,
        "parameter_updates": 0,
        "generator_import_count": 0,
        "generator_loaded": False,
        "generator_forward_calls": 0,
        "generator_capture_calls": 0,
        "route_or_lora_loaded": False,
        "binder_loaded": False,
        "slot_model_loaded": False,
        "b0_executed": False,
        "b0_replaced": False,
    }:
        fail("observer qualification performed training/generator/B0 work")

    aggregate = _mapping(value.get("aggregate"), "observer.aggregate")
    if aggregate != {
        "row_count": 64,
        "parent_qualified_row_count": 64,
        "part_rich_row_count": aggregate.get("part_rich_row_count"),
        "dynamic_edge_lifecycle_row_count": aggregate.get("dynamic_edge_lifecycle_row_count"),
        "three_plus_member_group_positive_row_count": aggregate.get("three_plus_member_group_positive_row_count"),
        "part_rich_rows_by_stratum": aggregate.get("part_rich_rows_by_stratum"),
        "dynamic_edge_rows_by_stratum": aggregate.get("dynamic_edge_rows_by_stratum"),
        "group_positive_rows_by_stratum": aggregate.get("group_positive_rows_by_stratum"),
        "pair_inventory_complete_row_count": 64,
        "nonfinite_count": 0,
        "observer_abstention_count": 0,
        "forced_match_count": 0,
        "raw_masks_tracks_descriptors_or_hidden_released": False,
    }:
        fail("observer aggregate schema/hard zero counts differ")
    for key in (
        "part_rich_row_count",
        "dynamic_edge_lifecycle_row_count",
        "three_plus_member_group_positive_row_count",
    ):
        if type(aggregate.get(key)) is not int or aggregate[key] < 16:
            fail(f"observer pool lacks required coverage: {key}")
    for key in (
        "part_rich_rows_by_stratum",
        "dynamic_edge_rows_by_stratum",
        "group_positive_rows_by_stratum",
    ):
        counts = _mapping(aggregate.get(key), f"observer.aggregate.{key}")
        if set(counts) != set(STRATA) or any(
            type(counts[stratum]) is not int or counts[stratum] < 4
            for stratum in STRATA
        ):
            fail(f"observer pool lacks balanced stratum coverage: {key}")

    expected_registry_rows = _registry_rows(registry)
    rows = _array(value.get("rows"), "observer.rows")
    if len(rows) != 64:
        fail("observer evidence must contain exactly 64 rows")
    observed_part: list[str] = []
    observed_dynamic: list[str] = []
    observed_group: list[str] = []
    for ordinal, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"observer.rows[{ordinal}]")
        _verify_exact_keys(
            row,
            {
                "ordinal",
                "uuid",
                "source_media_sha256",
                "stratum",
                "parent_track_count",
                "min_parent_visible_tau",
                "min_parent_soft_mass_fraction",
                "min_parent_alive_posterior",
                "parent_track_idf1",
                "parent_mask_iou_median",
                "parent_slot_alias_count",
                "identity_switches_max",
                "mean_dustbin_mass",
                "single_tau_dustbin_mass_max",
                "part_rich",
                "qualifying_part_count",
                "min_part_visible_tau",
                "min_part_parent_relative_mass",
                "max_part_outside_parent_mass",
                "min_part_mask_iou",
                "pair_inventory_complete",
                "dynamic_edge_lifecycle",
                "dynamic_edge_distinct_tau",
                "dynamic_edge_same_endpoints",
                "phase0_only_edge",
                "later_persist_or_deactivate",
                "relative_velocity_all_finite",
                "dynamic_edge_f1",
                "event_phase_error_tau",
                "three_plus_member_group_positive",
                "group_member_count",
                "group_active_tau",
                "group_members_unique",
                "group_permutation_equivariant",
                "nonfinite_count",
                "forced_match_count",
                "status",
            },
            f"observer.rows[{ordinal}]",
        )
        registered = expected_registry_rows[ordinal]
        if (
            row.get("ordinal") != ordinal
            or row.get("uuid") != registered["uuid"]
            or row.get("source_media_sha256") != registered["source_media_sha256"]
            or row.get("stratum") != registered["stratum"]
        ):
            fail(f"observer row identity/replay binding differs: {ordinal}")
        parents = row.get("parent_track_count")
        if type(parents) is not int or not 2 <= parents <= 12:
            fail(f"observer parent count fails: {ordinal}")
        if type(row.get("min_parent_visible_tau")) is not int or row["min_parent_visible_tau"] < 6:
            fail(f"observer parent visibility fails: {ordinal}")
        if _finite_number(row.get("min_parent_soft_mass_fraction"), "parent mass") < 0.001:
            fail(f"observer parent soft mass fails: {ordinal}")
        if _finite_number(row.get("min_parent_alive_posterior"), "parent alive") < 0.5:
            fail(f"observer parent alive posterior fails: {ordinal}")
        if _finite_number(row.get("parent_track_idf1"), "parent IDF1") < 0.75:
            fail(f"observer parent IDF1 fails: {ordinal}")
        if _finite_number(row.get("parent_mask_iou_median"), "parent IoU") < 0.55:
            fail(f"observer parent mask IoU fails: {ordinal}")
        if type(row.get("identity_switches_max")) is not int or row["identity_switches_max"] > 1:
            fail(f"observer identity switch gate fails: {ordinal}")
        if row.get("parent_slot_alias_count") != 0:
            fail(f"observer parent slots alias: {ordinal}")
        if _finite_number(row.get("mean_dustbin_mass"), "mean dustbin") > 0.2:
            fail(f"observer mean dustbin gate fails: {ordinal}")
        if _finite_number(row.get("single_tau_dustbin_mass_max"), "tau dustbin") > 0.35:
            fail(f"observer single-tau dustbin gate fails: {ordinal}")
        if row.get("nonfinite_count") != 0 or row.get("forced_match_count") != 0:
            fail(f"observer row has nonfinite/forced match: {ordinal}")
        if row.get("pair_inventory_complete") is not True:
            fail(f"observer pair opportunity inventory is incomplete: {ordinal}")

        if row.get("part_rich") is True:
            if type(row.get("qualifying_part_count")) is not int or row["qualifying_part_count"] < 2:
                fail(f"observer part-rich row lacks two parts: {ordinal}")
            if type(row.get("min_part_visible_tau")) is not int or row["min_part_visible_tau"] < 4:
                fail(f"observer part visibility fails: {ordinal}")
            if _finite_number(row.get("min_part_parent_relative_mass"), "part mass") < 0.02:
                fail(f"observer part relative mass fails: {ordinal}")
            if _finite_number(row.get("max_part_outside_parent_mass"), "outside mass") != 0.0:
                fail(f"observer part leaves parent support: {ordinal}")
            if _finite_number(row.get("min_part_mask_iou"), "part IoU") < 0.35:
                fail(f"observer part IoU fails: {ordinal}")
            observed_part.append(row["stratum"])
        else:
            if not (
                row.get("qualifying_part_count") == 0
                and row.get("min_part_visible_tau") is None
                and row.get("min_part_parent_relative_mass") is None
                and row.get("max_part_outside_parent_mass") is None
                and row.get("min_part_mask_iou") is None
            ):
                fail(f"observer null part row carries fake part metrics: {ordinal}")

        if row.get("dynamic_edge_lifecycle") is True:
            if type(row.get("dynamic_edge_distinct_tau")) is not int or row["dynamic_edge_distinct_tau"] < 3:
                fail(f"observer dynamic edge has insufficient temporal support: {ordinal}")
            if row.get("dynamic_edge_same_endpoints") is not True:
                fail(f"observer dynamic edge changes endpoints: {ordinal}")
            if row.get("phase0_only_edge") is not False:
                fail(f"observer dynamic edge is phase0-only: {ordinal}")
            if row.get("later_persist_or_deactivate") is not True:
                fail(f"observer dynamic edge lacks lifecycle: {ordinal}")
            if row.get("relative_velocity_all_finite") is not True:
                fail(f"observer dynamic edge velocity is nonfinite: {ordinal}")
            if _finite_number(row.get("dynamic_edge_f1"), "dynamic edge F1") < 0.8:
                fail(f"observer dynamic edge F1 fails: {ordinal}")
            if _finite_number(row.get("event_phase_error_tau"), "event phase error") > 1:
                fail(f"observer event phase error fails: {ordinal}")
            observed_dynamic.append(row["stratum"])
        else:
            if not (
                row.get("dynamic_edge_distinct_tau") == 0
                and row.get("dynamic_edge_same_endpoints") is True
                and row.get("phase0_only_edge") is False
                and row.get("later_persist_or_deactivate") is False
                and row.get("relative_velocity_all_finite") is True
                and row.get("dynamic_edge_f1") is None
                and row.get("event_phase_error_tau") is None
            ):
                fail(f"observer qualified-null edge row carries fake edge metrics: {ordinal}")

        if row.get("three_plus_member_group_positive") is True:
            if type(row.get("group_member_count")) is not int or not 3 <= row["group_member_count"] <= 4:
                fail(f"observer group member arity fails: {ordinal}")
            if row.get("group_members_unique") is not True:
                fail(f"observer group repeats a member: {ordinal}")
            if row.get("group_permutation_equivariant") is not True:
                fail(f"observer group is not permutation equivariant: {ordinal}")
            if type(row.get("group_active_tau")) is not int or row["group_active_tau"] < 2:
                fail(f"observer group lacks temporal support: {ordinal}")
            observed_group.append(row["stratum"])
        else:
            if not (
                row.get("group_member_count") == 0
                and row.get("group_active_tau") == 0
                and row.get("group_members_unique") is True
                and row.get("group_permutation_equivariant") is True
            ):
                fail(f"observer qualified-null group row carries fake group evidence: {ordinal}")
        if row.get("status") != "PASS":
            fail(f"observer row did not pass complete opportunity audit: {ordinal}")

    observed_counts = {
        "part_rich_row_count": len(observed_part),
        "dynamic_edge_lifecycle_row_count": len(observed_dynamic),
        "three_plus_member_group_positive_row_count": len(observed_group),
        "part_rich_rows_by_stratum": dict(Counter(observed_part)),
        "dynamic_edge_rows_by_stratum": dict(Counter(observed_dynamic)),
        "group_positive_rows_by_stratum": dict(Counter(observed_group)),
    }
    for key, observed in observed_counts.items():
        if aggregate.get(key) != observed:
            fail(f"observer aggregate does not match rows: {key}")
    return value


def validate_external_completion_seal(
    value: Mapping[str, Any],
    *,
    stage: str,
    evidence_receipt_file_sha256: str,
    evidence_receipt_self_sha256: str,
) -> Mapping[str, Any]:
    _verify_exact_keys(
        value,
        {
            "schema_version",
            "qualification_id",
            "stage",
            "decision",
            "external_controller",
            "producer_process_disjoint",
            "candidate_file_presence_is_completion_authority",
            "evidence_receipt_file_sha256",
            "evidence_receipt_self_sha256",
            "seal_sha256",
        },
        "external completion seal",
    )
    if value.get("schema_version") != COMPLETION_SEAL_SCHEMA:
        fail("external completion seal schema differs")
    if value.get("qualification_id") != QUALIFICATION_ID or value.get("stage") != stage:
        fail("external completion seal identity/stage differs")
    if value.get("decision") != "PASS":
        fail("external completion seal did not PASS")
    if value.get("external_controller") is not True or value.get("producer_process_disjoint") is not True:
        fail("completion seal is not external/process-disjoint")
    if value.get("candidate_file_presence_is_completion_authority") is not False:
        fail("candidate evidence presence cannot complete qualification")
    if value.get("evidence_receipt_file_sha256") != _sha256(
        evidence_receipt_file_sha256, "sealed evidence file"
    ):
        fail("completion seal evidence file link differs")
    if value.get("evidence_receipt_self_sha256") != _sha256(
        evidence_receipt_self_sha256, "sealed evidence self hash"
    ):
        fail("completion seal evidence self link differs")
    _verify_self_hash(value, "seal_sha256", "external completion seal")
    return value


def _missing_stage_receipt() -> Mapping[str, Any]:
    return {
        "status": "MISSING",
        "receipt_present": False,
        "receipt_file_sha256": None,
        "receipt_self_sha256": None,
        "external_completion_seal_present": False,
        "external_completion_seal_file_sha256": None,
        "evidence_admitted": False,
    }


def build_qualification_receipt(
    *,
    qualification_prereg: Mapping[str, Any],
    qualification_prereg_file_sha256: str,
    receipt_schema: Mapping[str, Any],
    receipt_schema_file_sha256: str,
    v10a2_prereg: Mapping[str, Any],
    v10a2_prereg_file_sha256: str,
    registry: Mapping[str, Any],
    registry_file_sha256: str,
    actual_manifest: Mapping[str, Any],
    actual_manifest_file_sha256: str,
    unpinned_perceptual_receipt_supplied: bool = False,
    unpinned_observer_receipt_supplied: bool = False,
) -> Mapping[str, Any]:
    validate_qualification_prereg(
        qualification_prereg,
        observed_file_sha256=qualification_prereg_file_sha256,
    )
    validate_receipt_schema_authority(
        receipt_schema, observed_file_sha256=receipt_schema_file_sha256
    )
    if v10a2_prereg_file_sha256 != EXPECTED_V10A2_PREREG_FILE_SHA256:
        fail("V10-A2 main preregistration bytes differ from qualification pin")
    try:
        main_v10a2.validate_preregistration(v10a2_prereg)
    except main_v10a2.V10A2PreflightError as error:
        fail(f"V10-A2 main preregistration differs: {error}")
    if v10a2_prereg.get("preregistration_sha256") != EXPECTED_V10A2_PREREG_SELF_SHA256:
        fail("V10-A2 main preregistration self hash differs from qualification pin")
    try:
        provisional.validate_provisional_registry(
            registry,
            observed_registry_file_sha256=registry_file_sha256,
            actual_manifest=actual_manifest,
            actual_manifest_file_sha256=actual_manifest_file_sha256,
        )
    except provisional.V10A2P0ProvisionalPreflightError as error:
        fail(f"provisional64 input differs: {error}")

    blockers = list(REQUIRED_ARTIFACT_IDS)
    if unpinned_perceptual_receipt_supplied:
        blockers.append("UNPREREGISTERED_PERCEPTUAL_RECEIPT_IGNORED")
    if unpinned_observer_receipt_supplied:
        blockers.append("UNPREREGISTERED_OBSERVER_RECEIPT_IGNORED")
    value = {
        "schema_version": RECEIPT_SCHEMA,
        "qualification_id": QUALIFICATION_ID,
        "status": ONLY_STATUS,
        "decision": ONLY_DECISION,
        "contract": {
            "file_sha256": qualification_prereg_file_sha256,
            "self_sha256": EXPECTED_PREREG_SELF_SHA256,
            "verified": True,
        },
        "inputs": {
            "provisional_registry_file_sha256": registry_file_sha256,
            "provisional_registry_self_sha256": provisional.EXPECTED_REGISTRY_SELF_SHA256,
            "actual_manifest_file_sha256": actual_manifest_file_sha256,
            "actual_manifest_self_sha256": provisional.EXPECTED_ACTUAL_MANIFEST_SELF_SHA256,
            "candidate_count": 64,
            "actual_media_count": 32,
            "target_or_action_metadata_released_to_p0": False,
        },
        "execution": {
            "mode": "cpu_readonly_receipt_admission",
            "gpu_used_by_this_runner": False,
            "models_loaded_by_this_runner": False,
            "optimizer_created": False,
            "backward_calls": 0,
            "parameter_updates": 0,
            "generator_loaded": False,
            "generator_forward_calls": 0,
        },
        "perceptual_qualification": _missing_stage_receipt(),
        "observer_qualification": _missing_stage_receipt(),
        "frozen_base_b0": {
            "arm_id": "B0_FROZEN_BASE",
            "first_class_future_arm_preserved": True,
            "replaced_by_historical_or_metadata_row": False,
            "executed_by_qualification": False,
            "capture_calls": 0,
            "generator_forward_calls": 0,
            "parameter_updates": 0,
            "future_mev_40_cell_parity_still_required": True,
            "future_factorial_36_cell_b0_closure_still_required": True,
            "future_total_b0_action_cell_count": 76,
            "fallback_still_required": True,
        },
        "target_teacher_boundary": {
            "p0_observer_process_target_teacher_read_count": 0,
            "p3_route_process_target_teacher_read_count": 0,
            "p3_route_process_actual_locked_graph_read_count": 0,
            "p3_reward_source": "frozen_binder_same_state_self_generated_action_minus_mean_noop_a_noop_b_middle_layer_only",
            "actual_target_graph_used_in_generator_loss": False,
            "actual_locked_graph_external_evaluator_only": True,
        },
        "permissions": {
            "p0_slot_pretraining_authorized": False,
            "binder_training_authorized": False,
            "generator_training_authorized": False,
            "gpu_launch_authorized": False,
            "parameter_updates_authorized": False,
        },
        "blockers": blockers,
        "claim_boundary": {
            "perceptual_disjointness_supported": False,
            "observer_observability_supported": False,
            "official_registry_supported": False,
            "representation_supported": False,
            "transfer_supported": False,
            "training_run": False,
        },
    }
    return {**value, "receipt_sha256": object_sha256(value)}


def run_qualification(
    qualification_prereg_path: Path = DEFAULT_PREREG_PATH,
    receipt_schema_path: Path = DEFAULT_RECEIPT_SCHEMA_PATH,
    v10a2_prereg_path: Path = DEFAULT_V10A2_PREREG_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    actual_manifest_path: Path = DEFAULT_ACTUAL_MANIFEST_PATH,
    *,
    perceptual_receipt_path: Path | None = None,
    observer_receipt_path: Path | None = None,
) -> Mapping[str, Any]:
    # Deliberately do not open unpinned evidence.  Presence is not authority and
    # protected receipts must first be pinned by a later preregistration.
    return build_qualification_receipt(
        qualification_prereg=load_json(qualification_prereg_path),
        qualification_prereg_file_sha256=file_sha256(qualification_prereg_path),
        receipt_schema=load_json(receipt_schema_path),
        receipt_schema_file_sha256=file_sha256(receipt_schema_path),
        v10a2_prereg=load_json(v10a2_prereg_path),
        v10a2_prereg_file_sha256=file_sha256(v10a2_prereg_path),
        registry=load_json(registry_path),
        registry_file_sha256=file_sha256(registry_path),
        actual_manifest=load_json(actual_manifest_path),
        actual_manifest_file_sha256=file_sha256(actual_manifest_path),
        unpinned_perceptual_receipt_supplied=perceptual_receipt_path is not None,
        unpinned_observer_receipt_supplied=observer_receipt_path is not None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG_PATH)
    parser.add_argument("--receipt-schema", type=Path, default=DEFAULT_RECEIPT_SCHEMA_PATH)
    parser.add_argument("--v10a2-preregistration", type=Path, default=DEFAULT_V10A2_PREREG_PATH)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--actual-manifest", type=Path, default=DEFAULT_ACTUAL_MANIFEST_PATH)
    parser.add_argument("--perceptual-receipt", type=Path)
    parser.add_argument("--observer-receipt", type=Path)
    args = parser.parse_args(argv)
    receipt = run_qualification(
        qualification_prereg_path=args.preregistration,
        receipt_schema_path=args.receipt_schema,
        v10a2_prereg_path=args.v10a2_preregistration,
        registry_path=args.registry,
        actual_manifest_path=args.actual_manifest,
        perceptual_receipt_path=args.perceptual_receipt,
        observer_receipt_path=args.observer_receipt,
    )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ACTUAL_MANIFEST_PATH",
    "DEFAULT_PREREG_PATH",
    "DEFAULT_RECEIPT_SCHEMA_PATH",
    "DEFAULT_REGISTRY_PATH",
    "DEFAULT_V10A2_PREREG_PATH",
    "EXPECTED_PREREG_FILE_SHA256",
    "EXPECTED_PREREG_SELF_SHA256",
    "EXPECTED_RECEIPT_SCHEMA_FILE_SHA256",
    "EXPECTED_RECEIPT_SCHEMA_SELF_SHA256",
    "ONLY_DECISION",
    "ONLY_STATUS",
    "QUALIFICATION_ID",
    "REQUIRED_ARTIFACT_IDS",
    "V10A2P0QualificationError",
    "build_qualification_receipt",
    "canonical_json_bytes",
    "file_sha256",
    "load_json",
    "main",
    "object_sha256",
    "run_qualification",
    "validate_external_completion_seal",
    "validate_observer_evidence_receipt",
    "validate_perceptual_evidence_receipt",
    "validate_qualification_prereg",
    "validate_receipt_schema_authority",
]
