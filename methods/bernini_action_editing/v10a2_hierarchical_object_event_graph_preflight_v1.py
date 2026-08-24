#!/usr/bin/env python3
"""CPU-only, fail-closed preflight for the V10-A2 representation pilot.

This module is deliberately not an official runner.  It verifies the sealed
V10-A2 preregistration, the byte-pinned 16-row R1b source manifest, and the
sanitized 64-row provisional P0 evidence, then emits the only currently legal
result: ``PRE_RUN_NO``.  It has no API for supplying synthetic dependency
receipts, changing a blocker to present, authorizing a GPU, constructing an
optimizer, or launching training.

The 64 provisional rows are integrity evidence, not an official registry:
target-blind perceptual exclusion and frozen-observer parent/part/interaction
qualification are still missing.  The teacher permissions/releases, official
runner, immutable snapshot, and independent pre-run audit are also missing.
A later executable stage requires a new sealed revision; this validator cannot
be rehashed into readiness.
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
    from . import v10a2_p0_source_only_64_provisional_preflight_v1 as p0_provisional
except ImportError:  # Direct script execution has no package context.
    import v10a2_p0_source_only_64_provisional_preflight_v1 as p0_provisional


SCHEMA = "bernini-v10a2-hierarchical-object-event-graph-prereg-v1"
EXPERIMENT_ID = "v10a2_hierarchical_object_event_graph_pilot_v1"
RECEIPT_SCHEMA = "bernini-v10a2-hierarchical-object-event-graph-preflight-v1"
ONLY_STATUS = "PRE_RUN_NO"

METHOD_ROOT = Path(__file__).resolve().parent
DEFAULT_PREREGISTRATION_PATH = (
    METHOD_ROOT / "assets" / "v10a2_hierarchical_object_event_graph_prereg_v1.json"
)
DEFAULT_SOURCE_MANIFEST_PATH = (
    METHOD_ROOT / "assets" / "target_factorized_soft_ot_graph_teacher_manifest_v5_r1b.json"
)
DEFAULT_PROVISIONAL_REGISTRY_PATH = (
    METHOD_ROOT / "assets" / "v10a2_p0_source_only_64_provisional_v1.json"
)

EXPECTED_PREREGISTRATION_SHA256 = (
    "721f9a47f300c985fb3a9aa7fa98233fd2f037dba2c00bc31396457fb855938b"
)
EXPECTED_SOURCE_FILE_SHA256 = (
    "d43ff7f7c14b2c25bf949798fb71839f6f0e6325d8829784f2d9eef5a1516929"
)
EXPECTED_SOURCE_SELF_SHA256 = (
    "231da71f38bdd982a9276b02fb3351200b372563cfdded39fb7f7f6f7de93446"
)
EXPECTED_SOURCE_SCHEMA = "bernini-target-factorized-soft-ot-graph-manifest-v5-r1b"
EXPECTED_SOURCE_EXPERIMENT = "target_factorized_soft_ot_graph_teacher_pilot_v5_r1b"

TRAIN_ORDINALS = (0, 1, 3, 4, 8, 9, 10, 14)
CALIBRATION_ORDINALS = (5, 12, 13, 15)
LOCKED_ORDINALS = (2, 6, 7, 11)
FAMILIES = (
    "articulated_ordered_motion",
    "contact_transfer",
    "lifecycle_entry_exit",
    "multi_entity_interaction",
)
LOFO_FAMILIES = (
    "articulated_ordered_motion",
    "multi_entity_interaction",
    "lifecycle_entry_exit",
    "contact_transfer",
)
EXPECTED_ROWS = (
    (0, "3d808f3fcd6ec4f1fb8f519fa51d79f3c1f5b6658d1d07d54001baadb5955a54", "383ad770-b3ac-44a8-bf59-ad5f1bcb1bcc", "multi_entity_interaction", "development_report", "binder_train"),
    (1, "a1d559bc02d98fc7953336abe056e4e34705bfe3e3046b03132d45dc53eba2fe", "feb01281-0940-462b-a142-342558574d14", "lifecycle_entry_exit", "development_report", "binder_train"),
    (2, "fe4ec9413d3d37e5e14845229889a1ee0629d02838bae2fdb45f80ef0650e4dc", "4940bfd0-71b1-42e6-888d-e4292fc08864", "articulated_ordered_motion", "locked_validation", "locked_actual"),
    (3, "3c5085d6f90fe532d74d76c6fcb7072d37e0c01c5217f2b4bd77d10fe4addaee", "c23cfec4-4f62-4f59-b86a-1c3492d489a7", "contact_transfer", "development_report", "binder_train"),
    (4, "3b2494c276faacf8b632eb5a9d3a3e1aac4e5dd6a9ed9ecfc7f105a5800adb7c", "625fbb07-7e89-42bd-8d88-a0b5d154d136", "contact_transfer", "development_report", "binder_train"),
    (5, "7c790138dc115b62726617b739288c17e489c8b3299713d87a494de91278a5bb", "bb35e75d-5e68-447a-8f2a-082a417d0b92", "contact_transfer", "development_report", "calibration"),
    (6, "c4184f51d9e8a31d05e301f6ff559865b77b1559d6580fad1f6a8dd8b615f146", "fe93cb66-2a74-4167-97bc-adbe7ec85e5d", "multi_entity_interaction", "locked_validation", "locked_actual"),
    (7, "b6c120d4c3332dd63d94c6ddc29c22911ae5e1bf10732177213886915628c52c", "07fac509-c337-4e7f-8312-c889b643c82b", "lifecycle_entry_exit", "locked_validation", "locked_actual"),
    (8, "2cc12b36518091286b8d305ff727c4097db20adfa483cb96d639c391159a8f85", "09c19944-6ba5-4c79-8689-537633156158", "multi_entity_interaction", "development_report", "binder_train"),
    (9, "71be4d30afca7449b0eb80cb575c4c0abc3f0fb54ed6325d57ac47efaf6dc8c4", "ab68d4e1-1327-429d-bf94-216a8eedde1b", "lifecycle_entry_exit", "development_report", "binder_train"),
    (10, "89dc5aa6dedae68480cd6ca6a01931f9e928c62990fe7fb8c0835cec660f4b49", "587093a2-f535-44c6-bbd2-56cde3f062e6", "articulated_ordered_motion", "development_report", "binder_train"),
    (11, "981fb7f98968c62e2446cbdbe743e8ac0f0dc0980ce76710567d4dfe7bd93fd0", "cf874922-b3d8-42d6-a09d-96af7731ac7d", "contact_transfer", "locked_validation", "locked_actual"),
    (12, "459d35d03f7209e185fbd7cef38713d85b7a25043279b9ab2614e897bec9dcae", "21120e42-4f23-445c-8862-bdecc94dee9d", "lifecycle_entry_exit", "development_report", "calibration"),
    (13, "1ea210054afedba0005cf222f428357d0551c56d5139d577073eef914c898105", "3ddb123a-0ca4-4b82-9eb4-43f64931f6a3", "multi_entity_interaction", "development_report", "calibration"),
    (14, "2754e5b32a96c5dbf913d124142d36418827936a521b8628f1b5b89006f4460c", "f7af6d28-f57d-483c-a9aa-03c32e2b2a00", "articulated_ordered_motion", "development_report", "binder_train"),
    (15, "3916eede64dd1dfc5f54c85de9324b32dd44cb779c0d4dde973a4837880c7c38", "97940925-cd8c-47b1-a6e2-d6ae70945137", "articulated_ordered_motion", "development_report", "calibration"),
)

EXPECTED_COUNTS = {
    "train_cells": 8,
    "calibration_cells": 16,
    "locked_cells": 16,
    "mev_action_cells": 40,
    "source_only_factorial_action_cells": 36,
    "total_action_cells": 76,
    "mev_cell_specific_trajectory_count": 200,
    "mev_shared_null_trajectory_count": 24,
    "mev_trajectory_count": 224,
    "factorial_b0_action_trajectory_count": 72,
    "factorial_shared_noop_trajectory_count": 24,
    "factorial_trajectory_count": 96,
    "total_trajectories": 320,
    "decoded_video_count": 320,
    "standard_transformer_calls": 25600,
    "mev_arrest_extra_action_diagnostic_calls": 1600,
    "mev_action_same_state_noop_ab_calls": 240,
    "mev_lexical_same_state_noop_ab_calls": 240,
    "mev_arrest_selected_noop_b_calls": 120,
    "factorial_same_state_noop_ab_calls": 216,
    "total_transformer_calls": 28016,
    "mev_selected_conditional_capture_calls": 1272,
    "factorial_selected_conditional_capture_calls": 324,
    "selected_conditional_capture_calls": 1596,
    "mev_projected_block_rows": 5088,
    "factorial_projected_block_rows": 1296,
    "projected_block_rows": 6384,
}

BLOCKERS = (
    p0_provisional.V10A2_BLOCKER,
    "DEVELOPMENT_TEACHER_BINDER_ONLY_PERMISSION_12_MISSING",
    "DEVELOPMENT_TEACHER_RELEASES_12_MISSING",
    "LOCKED_BLIND_TEACHER_RELEASES_4_MISSING",
    "SOURCE_ONLY_FACTORIAL_REGISTRY_PROMPTS_NOOPS_MISSING",
    "DUAL_QUALITY_OBSERVER_AND_BLIND_RATER_PREREG_MISSING",
    "P3_ZERO_TARGET_MOUNT_FIREWALL_RECEIPT_MISSING",
    "OFFICIAL_RUNNER_MISSING",
    "IMMUTABLE_SNAPSHOT_MISSING",
    "INDEPENDENT_PRE_RUN_AUDIT_MISSING",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class V10A2PreflightError(RuntimeError):
    """Raised when a sealed preregistration or source authority differs."""


def fail(message: str) -> NoReturn:
    raise V10A2PreflightError(message)


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
        raise V10A2PreflightError("value is not canonical ASCII JSON") from error


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
        fail(f"JSON authority must be one regular non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except V10A2PreflightError:
        raise
    except Exception as error:
        raise V10A2PreflightError(f"cannot parse {path}: {error}") from error
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


def _verify_self_hash(value: Mapping[str, Any], key: str, label: str) -> str:
    expected = _sha256(value.get(key), f"{label}.{key}")
    payload = dict(value)
    payload.pop(key, None)
    actual = object_sha256(payload)
    if actual != expected:
        fail(f"{label} self hash differs")
    return actual


def _expected_partition(ordinal: int) -> str:
    if ordinal in TRAIN_ORDINALS:
        return "binder_train"
    if ordinal in CALIBRATION_ORDINALS:
        return "calibration"
    if ordinal in LOCKED_ORDINALS:
        return "locked_actual"
    fail(f"unexpected source ordinal: {ordinal}")


def _project_prereg_row(row: Any) -> tuple[Any, ...]:
    value = _mapping(row, "fixed_split.rows[]")
    if set(value) != {
        "ordinal", "pair_id", "uuid", "family", "report_split", "a2_partition"
    }:
        fail("fixed split row schema differs")
    return (
        value["ordinal"],
        value["pair_id"],
        value["uuid"],
        value["family"],
        value["report_split"],
        value["a2_partition"],
    )


def validate_preregistration(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if value.get("schema_version") != SCHEMA:
        fail("preregistration schema differs")
    if value.get("experiment_id") != EXPERIMENT_ID:
        fail("experiment id differs")
    if value.get("status") != ONLY_STATUS:
        fail("the only legal preregistration status is PRE_RUN_NO")
    digest = _verify_self_hash(value, "preregistration_sha256", "preregistration")
    if digest != EXPECTED_PREREGISTRATION_SHA256:
        fail("preregistration is not the independently pinned V10-A2 revision")

    pin = _mapping(value.get("source_manifest_pin"), "source_manifest_pin")
    if pin != {
        "relative_path": "assets/target_factorized_soft_ot_graph_teacher_manifest_v5_r1b.json",
        "file_sha256": EXPECTED_SOURCE_FILE_SHA256,
        "self_sha256": EXPECTED_SOURCE_SELF_SHA256,
        "schema_version": EXPECTED_SOURCE_SCHEMA,
        "experiment_id": EXPECTED_SOURCE_EXPERIMENT,
        "row_count": 16,
    }:
        fail("source manifest byte/self pin differs")

    split = _mapping(value.get("fixed_split"), "fixed_split")
    if split.get("binder_train_ordinals") != list(TRAIN_ORDINALS):
        fail("binder-train split differs")
    if split.get("calibration_ordinals") != list(CALIBRATION_ORDINALS):
        fail("calibration split differs")
    if split.get("locked_actual_ordinals") != list(LOCKED_ORDINALS):
        fail("locked actual split differs")
    if split.get("counts") != {
        "binder_train": 8, "calibration": 4, "locked_actual": 4, "all": 16
    } or split.get("family_count_per_all_rows") != 4:
        fail("fixed split counts differ")
    rows = tuple(_project_prereg_row(row) for row in _array(split.get("rows"), "fixed_split.rows"))
    if rows != EXPECTED_ROWS:
        fail("fixed 16-row identity/order/partition differs")
    if len({row[1] for row in rows}) != 16 or len({row[2] for row in rows}) != 16:
        fail("fixed rows contain duplicate pair or UUID")
    if Counter(row[3] for row in rows) != Counter({family: 4 for family in FAMILIES}):
        fail("fixed rows are not four per family")

    slot = _mapping(value.get("source_only_slot_pretraining"), "source_only_slot_pretraining")
    if slot != {
        "required_video_count": 64,
        "official_registry_present": False,
        "provisional_registry_present": True,
        "provisional_registry_relative_path": "assets/v10a2_p0_source_only_64_provisional_v1.json",
        "provisional_registry_file_sha256": p0_provisional.EXPECTED_REGISTRY_FILE_SHA256,
        "provisional_registry_self_sha256": p0_provisional.EXPECTED_REGISTRY_SELF_SHA256,
        "provisional_candidate_count": 64,
        "provisional_eligible_count": 283,
        "provisional_strata_selected_counts": {
            "camera0_occlusion0": 16,
            "camera0_occlusion1": 16,
            "camera1_occlusion0": 16,
            "camera1_occlusion1": 16,
        },
        "provisional_integrity_only": True,
        "perceptual_cluster_exclusion_complete": False,
        "frozen_observer_qualification_complete": False,
        "p0_slot_pretraining_authorized": False,
        "must_exclude_all_16_uuids_media_ancestors_and_perceptual_clusters": True,
        "target_video_allowed": False,
        "instruction_caption_or_family_allowed": False,
        "generator_forward_calls": 0,
        "trainable_scope": "slot_part_uncertainty_only",
    }:
        fail("64-source slot-pretraining fail-closed contract differs")

    b0 = _mapping(value.get("frozen_base"), "frozen_base")
    expected_b0 = {
        "arm_id": "B0_FROZEN_BASE",
        "first_class_per_action_cell": True,
        "required_mev_action_cells": 40,
        "required_source_only_factorial_action_cells": 36,
        "required_total_action_cells": 76,
        "full_40_step_trajectory": True,
        "route_adapter_binder_hook_and_capture_off": True,
        "capture_calls": 0,
        "passive_action_must_be_bit_exact": True,
        "parity_transformer_outputs": 80,
        "parity_sampler_latents": 41,
        "decoded_video_bytes_must_match": True,
        "is_historical_p0": False,
        "quality_reference": True,
        "absolute_source_referenced_quality_gate_required": True,
        "failed_b0_cell_may_be_removed_from_denominator": False,
        "fallback": True,
    }
    if b0 != expected_b0:
        fail("Frozen Base contract differs")

    matrix = _mapping(value.get("trajectory_matrix"), "trajectory_matrix")
    for key in (
        "train_cells", "calibration_cells", "locked_cells", "mev_action_cells",
        "source_only_factorial_action_cells", "total_action_cells",
        "mev_cell_specific_trajectory_count", "mev_shared_null_trajectory_count",
        "mev_trajectory_count", "factorial_b0_action_trajectory_count",
        "factorial_shared_noop_trajectory_count", "factorial_trajectory_count",
        "total_trajectories", "decoded_video_count", "standard_transformer_calls",
        "mev_arrest_extra_action_diagnostic_calls",
        "mev_action_same_state_noop_ab_calls",
        "mev_lexical_same_state_noop_ab_calls",
        "mev_arrest_selected_noop_b_calls",
        "factorial_same_state_noop_ab_calls", "total_transformer_calls",
    ):
        if matrix.get(key) != EXPECTED_COUNTS[key]:
            fail(f"trajectory count differs: {key}")
    if matrix.get("cell_specific_arms") != [
        "B0_FROZEN_BASE", "ACTION_PASSIVE", "ARREST_NATIVE",
        "LEXICAL_DENIAL_NATIVE", "NOOP_A_NOUN_MATCHED_NATIVE",
    ]:
        fail("cell-specific arm registry differs")
    if matrix.get("same_state_null_companions") != [
        "NOOP_A_NOUN_MATCHED_NATIVE", "NOOP_B_NOUN_MATCHED_NATIVE"
    ]:
        fail("dual noun-matched NOOP registry differs")
    if matrix.get("sampler_steps") != 40 or matrix.get("standard_cfg_forwards_per_step") != 2:
        fail("sampler geometry differs")
    if matrix["standard_transformer_calls"] != matrix["total_trajectories"] * 40 * 2:
        fail("standard transformer call arithmetic differs")
    if matrix["mev_trajectory_count"] != (
        matrix["mev_cell_specific_trajectory_count"]
        + matrix["mev_shared_null_trajectory_count"]
    ):
        fail("MEV trajectory arithmetic differs")
    if matrix["factorial_trajectory_count"] != (
        matrix["factorial_b0_action_trajectory_count"]
        + matrix["factorial_shared_noop_trajectory_count"]
    ):
        fail("factorial trajectory arithmetic differs")
    if matrix["total_trajectories"] != (
        matrix["mev_trajectory_count"] + matrix["factorial_trajectory_count"]
    ):
        fail("full trajectory arithmetic differs")
    if matrix["total_transformer_calls"] != (
        matrix["standard_transformer_calls"]
        + matrix["mev_arrest_extra_action_diagnostic_calls"]
        + matrix["mev_action_same_state_noop_ab_calls"]
        + matrix["mev_lexical_same_state_noop_ab_calls"]
        + matrix["mev_arrest_selected_noop_b_calls"]
        + matrix["factorial_same_state_noop_ab_calls"]
    ):
        fail("total transformer call arithmetic differs")

    capture = _mapping(value.get("capture_contract"), "capture_contract")
    for key in (
        "mev_selected_conditional_capture_calls",
        "factorial_selected_conditional_capture_calls",
        "selected_conditional_capture_calls",
        "mev_projected_block_rows",
        "factorial_projected_block_rows",
        "projected_block_rows",
    ):
        if capture.get(key) != EXPECTED_COUNTS[key]:
            fail(f"capture count differs: {key}")
    if capture.get("denoising_step_indices") != [4, 19, 34] or capture.get("block_indices") != [6, 12, 18, 24]:
        fail("capture step/block registry differs")
    if capture.get("denoising_cells") != ["high", "mid", "low"]:
        fail("capture denoising cell names differ")
    if capture.get("b0_capture_calls") != 0 or capture.get("text_prefix_removed_before_binder") is not True:
        fail("B0/text capture boundary differs")
    if capture.get("raw_middle_tensor_persisted") is not False:
        fail("raw middle tensors may not persist")
    if capture["projected_block_rows"] != capture["selected_conditional_capture_calls"] * 4:
        fail("capture-to-block-row arithmetic differs")
    if capture["selected_conditional_capture_calls"] != (
        capture["mev_selected_conditional_capture_calls"]
        + capture["factorial_selected_conditional_capture_calls"]
    ) or capture["projected_block_rows"] != (
        capture["mev_projected_block_rows"]
        + capture["factorial_projected_block_rows"]
    ):
        fail("MEV/factorial capture subtotal arithmetic differs")

    anchor = _mapping(value.get("same_state_middle_anchor"), "same_state_middle_anchor")
    if anchor != {
        "form": "ACTION_minus_mean_NOOP_A_NOOP_B_at_same_native_action_latent_plus_relative_realized_action_slot_trajectory",
        "same_closure_required": True,
        "independently_authored_noun_matched_noops_required": 2,
        "action_residual_formula": "S_ACTION_minus_0.5_times_S_NOOP_A_plus_S_NOOP_B",
        "null_residual_formula": "S_NOOP_A_minus_S_NOOP_B",
        "per_section_action_norm_must_exceed_presealed_null_q99": True,
        "action_conditional_drives_action_sampler": True,
        "noop_companion_is_diagnostic_only": True,
        "arrest_action_conditional_is_diagnostic_only": True,
        "arrest_noop_companion_drives_sampler": True,
        "video_tokens_only": True,
        "decoded_self_generated_video_is_anchor": False,
        "target_video_latent_hidden_is_anchor": False,
        "target_graph_selects_slots_edges_or_thresholds": False,
        "pure_relative_graph_signature": True,
        "forbidden_signature_fields": [
            "absolute_phase0_centroid",
            "absolute_frame_layout",
            "absolute_object_scale",
            "raw_slot_appearance",
            "role_specific_target_descriptor",
        ],
    }:
        fail("same-state middle-layer anchor contract differs")

    slots = _mapping(value.get("hierarchical_slots"), "hierarchical_slots")
    if slots.get("max_parent_slots") != 12 or slots.get("max_part_slots_per_parent") != 4:
        fail("parent/part slot caps differ")
    for key in (
        "variable_alive_cardinality", "explicit_dustbin",
        "source_only_queries_sealed_before_action_verbs",
    ):
        if slots.get(key) is not True:
            fail(f"hierarchical slot requirement differs: {key}")
    if slots.get("noun_binding_can_create_delete_split_or_reorder_slots") is not False or slots.get("family_or_score_can_select_slots") is not False:
        fail("noun/family may not author slots")
    if slots.get("states") != [
        "birth", "visible", "occluded", "reentry", "membership_loss", "death", "right_censored"
    ]:
        fail("slot lifecycle state registry differs")

    graph = _mapping(value.get("dynamic_graph"), "dynamic_graph")
    if graph.get("rebuilt_each_tau") is not True or graph.get("pair_edges_dynamic") is not True:
        fail("graph must be dynamically rebuilt")
    if graph.get("relative_to_first_reliable_phase_only") is not True:
        fail("graph worldlines must be relative to first reliable phase")
    if graph.get("absolute_centroid_layout_or_scale_allowed") is not False:
        fail("absolute centroid/layout/scale leakage is forbidden")
    if graph.get("edge_types") != [
        "contact", "support_or_containment", "relative_motion", "coordination",
        "parent_part_articulation", "visibility_lifecycle",
    ]:
        fail("dynamic edge type registry differs")
    if (graph.get("group_event_factor_count_per_tau"), graph.get("group_event_factor_min_arity"), graph.get("group_event_factor_max_arity")) != (4, 2, 4):
        fail("group-event factor contract differs")
    if graph.get("edge_and_factor_eval_threshold") != 0.5 or graph.get("drop_intervention_before_aggregation") is not True:
        fail("graph threshold/drop contract differs")
    if graph.get("caption_or_family_classifier_substitution_allowed") is not False:
        fail("caption/family classification cannot substitute for the graph")

    factorial = _mapping(
        value.get("source_only_factorial_contract"),
        "source_only_factorial_contract",
    )
    if factorial != {
        "target_blind": True,
        "appearance_carrier_count": 3,
        "genuinely_different_action_program_count": 3,
        "generation_seed_count": 2,
        "wording_count": 2,
        "action_cell_count": 36,
        "all_carriers_role_compatible_with_all_actions": True,
        "native_action_trajectory_required_per_cell": True,
        "local_replay_may_substitute_for_cross_appearance_transfer": False,
        "carrier_perceptual_clusters_must_be_distinct": True,
        "noop_a_and_noop_b_independently_authored": True,
        "noop_a_and_noop_b_semantically_equivalent_noun_matched": True,
        "prompt_bytes_token_ids_spans_role_map_and_media_sha_presealed": True,
        "matched_action_cosine_min_across_appearances_seeds_wordings": 0.8,
        "within_carrier_wrong_action_margin_min": 0.1,
        "top1_action_cells_must_pass": "36/36",
        "null_q99_discovered_without_held_seed_wording": True,
        "held_seed_wording_must_exceed_null_q99": True,
        "appearance_seed_wording_probe_95pct_upper_bound": "chance_plus_0.10",
        "must_pass_before_target_graph_open": True,
    }:
        fail("source-only appearance/action factorial contract differs")

    stages = _mapping(value.get("training_stages"), "training_stages")
    if stages.get("binder_and_generator_joint_updates_allowed") is not False:
        fail("binder/generator joint updates are forbidden")
    expected_stage_flags = {
        "P0": (True, False, False, False),
        "P1": (True, False, False, False),
        "P2": (False, False, False, False),
        "P3": (False, False, True, False),
    }
    for name, expected in expected_stage_flags.items():
        stage = _mapping(stages.get(name), f"training_stages.{name}")
        observed = (
            stage.get("slot_or_binder_updates_allowed"),
            stage.get("generator_base_updates_allowed"),
            stage.get("route_or_lora_updates_allowed"),
            stage.get("currently_authorized"),
        )
        if observed != expected:
            fail(f"stage freeze/current authorization differs: {name}")
    if stages["P1"].get("development_graph_metadata_only") is not True or stages["P1"].get("locked_target_access_allowed") is not False:
        fail("P1 teacher/locked boundary differs")
    if stages["P1"].get("development_target_role") != "binder_only_anonymous_relative_graph_teacher":
        fail("P1 target must remain binder-only anonymous relative graph metadata")
    if stages["P1"].get("raw_target_rgb_latent_hidden_token_mask_descriptor_allowed") is not False:
        fail("P1 raw target payload is forbidden")
    if stages["P2"].get("locked_target_external_evaluator_only_after_threshold_seal") is not True:
        fail("P2 blind locked boundary differs")
    if stages["P3"].get("requires_full_representation_and_lofo_pass") is not True:
        fail("P3 must remain conditional on representation and LOFO")
    if stages["P3"].get("target_dataset_mount_allowed") is not False:
        fail("P3 target dataset mount is forbidden")
    if stages["P3"].get("target_teacher_mount_allowed") is not False:
        fail("P3 target teacher mount is forbidden")
    if stages["P3"].get("target_teacher_read_count_required") != 0:
        fail("P3 target teacher read count must be zero")
    if stages["P3"].get("target_derived_graph_scalar_program_id_or_release_digest_allowed") is not False:
        fail("P3 target-derived signal is forbidden")

    programs = _array(value.get("locked_actual_programs"), "locked_actual_programs")
    if [(row.get("ordinal"), row.get("family"), row.get("program_id")) for row in programs if type(row) is dict] != [
        (2, "articulated_ordered_motion", "cockatoo_beak_then_head_turn"),
        (6, "multi_entity_interaction", "two_people_laptop_group_shock"),
        (7, "lifecycle_entry_exit", "backpack_tray_then_phone_pickup_then_exit"),
        (11, "contact_transfer", "woman_ring_touch_then_box_close"),
    ] or len(programs) != 4:
        fail("locked actual program registry differs")
    if any(not row.get("required_roles") or len(row.get("ordered_events", [])) < 2 for row in programs):
        fail("locked actual roles/events are incomplete")

    gates = _mapping(value.get("representation_gates"), "representation_gates")
    if gates.get("locked_action_cells_must_all_pass") != "16/16" or gates.get("own_vs_each_control_and_wrong_program_margin_min") != 0.1:
        fail("representation admission margin/count differs")
    if gates.get("source_only_factorial_action_cells_must_all_pass") != "36/36":
        fail("source-only factorial admission count differs")
    if gates.get("factorial_must_pass_before_target_graph_open") is not True:
        fail("target graph may not open before the factorial passes")
    if gates.get("primary_denoising_cells") != ["mid", "low"] or gates.get("section_compensation_allowed") is not False:
        fail("representation denoising/noncompensation boundary differs")
    for key in ("missing_required_role_is_failure",):
        if gates.get(key) is not True:
            fail(f"representation fail-closed gate differs: {key}")

    lofo = _mapping(value.get("lofo_contract"), "lofo_contract")
    if lofo.get("folds") != [f"hold_out_{family}" for family in LOFO_FAMILIES]:
        fail("LOFO fold registry differs")
    if (lofo.get("train_development_rows_per_fold"), lofo.get("held_family_development_rows_per_fold"), lofo.get("held_locked_rows_per_fold")) != (9, 3, 1):
        fail("LOFO split counts differ")
    if lofo.get("same_architecture_optimizer_steps_and_thresholds") is not True or lofo.get("all_four_folds_required_for_unseen_family_transfer_claim") is not True:
        fail("LOFO invariance/claim boundary differs")

    ablation = _mapping(value.get("ablation_contract"), "ablation_contract")
    if ablation.get("retrained") != [
        "NO_SAME_STATE_ANCHOR", "GLOBAL_POOL_NO_SLOTS", "STATIC_GRAPH", "RANDOM_UNPRETRAINED_SLOTS"
    ]:
        fail("retrained ablation registry differs")
    if ablation.get("deterministic") != [
        "TEXT_ONLY", "PHASE_REVERSE", "PHASE_SHUFFLE", "SLOT_PERMUTE",
        "FACTOR_PERMUTE", "ROLE_SCRAMBLE", "DROP_REQUIRED_EDGE",
        "DROP_REQUIRED_FACTOR", "NOOP_NULL_SWAP", "CARRIER_PROGRAM_SWAP",
    ]:
        fail("deterministic ablation registry differs")
    if ablation.get("factor_permutation_max_absolute_error") != 1e-6:
        fail("factor permutation gate differs")
    if ablation.get("noop_null_swap_action_max_absolute_error") != 1e-6:
        fail("NOOP null-swap invariance gate differs")
    if ablation.get("nuisance_probe_unit") != "pair_or_trajectory_not_denoising_cell":
        fail("nuisance probe unit differs")
    if ablation.get("nuisance_probe_label_universe_and_split_presealed") is not True:
        fail("nuisance probe split is not presealed")
    if ablation.get("nuisance_probe_95pct_upper_bound") != "chance_plus_0.10":
        fail("nuisance probe confidence bound differs")

    quality = _mapping(value.get("quality_gates"), "quality_gates")
    if quality.get("noncompensable") is not True or quality.get("any_failure_falls_back_to_b0") is not True:
        fail("quality must be noncompensable with B0 fallback")
    if quality.get("architecture_disjoint_frozen_observer_count") != 2:
        fail("two architecture-disjoint frozen quality observers are required")
    if quality.get("q_select_gradient_allowed") is not False:
        fail("quality selection observer may not provide gradients")
    if quality.get("q_audit_hidden_from_trainer_and_selector_until_freeze") is not True:
        fail("quality audit observer must remain hidden until freeze")
    if quality.get("observer_abstention_or_disagreement_is_failure") is not True:
        fail("quality observer abstention/disagreement must fail closed")
    expected_quality_scalars = {
        "sharpness_hf_median_ratio_min": 0.95,
        "sharpness_hf_median_ratio_max": 1.1,
        "sharpness_hf_single_tau_max": 1.25,
        "sharpness_hf_tau_below_0_90_max_count": 1,
        "identity_cosine_drop_max": 0.02,
        "background_flow_warped_lpips_max": 0.03,
        "flicker_ratio_min": 0.8,
        "flicker_ratio_max": 1.1,
        "repeated_frame_fraction_increase_max": 0.05,
        "nonedited_slot_centroid_departure_L_max": 0.1,
        "dustbin_mass_increase_max": 0.05,
    }
    if any(quality.get(key) != expected for key, expected in expected_quality_scalars.items()):
        fail("quality scalar gate differs")
    if quality.get("track_inventory_must_equal_reference") is not True or quality.get("video_prior_floor") != "reference_fold_B0_NOOP_fifth_percentile":
        fail("quality inventory/video-prior gate differs")
    if quality.get("candidate_normalized_video_prior_min_relative_to_matched_b0") != -0.02:
        fail("matched-B0 video-prior floor differs")
    if (
        quality.get("blind_human_rater_count"),
        quality.get("blind_human_not_worse_than_b0_min_votes"),
    ) != (3, 2):
        fail("blind human B0 comparison contract differs")
    if quality.get("blind_human_target_hidden") is not True or quality.get("blind_human_per_cell_required") is not True:
        fail("blind human target/cell boundary differs")

    reward = _mapping(value.get("reward_contract"), "reward_contract")
    if reward != {
        "quality_is_hard_feasible_set_not_additive_reward": True,
        "per_required_section_b0_relative_improvement_min": 0.05,
        "own_vs_control_and_wrong_program_margin_min": 0.1,
        "selection_form": "minimum_clipped_B0_relative_improvement_across_required_sections",
        "failed_candidate_reward": "negative_infinity",
        "failed_candidate_output": "B0_FROZEN_BASE",
        "binder_frozen_during_route_training": True,
        "target_rgb_latent_hidden_in_generator_loss": False,
        "target_teacher_graph_scalar_program_id_or_digest_in_generator_loss": False,
        "p3_target_teacher_read_count_required": 0,
    }:
        fail("reward/fallback contract differs")

    teacher = _mapping(value.get("teacher_gate"), "teacher_gate")
    if teacher.get("current_v3_role") != "mechanics_canary_only_not_representation_or_training_authority":
        fail("current V3 role was overstated")
    if (teacher.get("required_development_teacher_count"), teacher.get("required_locked_blind_release_count")) != (12, 4):
        fail("teacher release counts differ")
    if teacher.get("hierarchical_parts_required") != ["beak_head", "eyes_mouth"] or teacher.get("group_event_factors_required") is not True:
        fail("teacher hierarchical/group gate differs")
    for key in ("reference_eval_temporal_views_disjoint", "anonymous_graph_metadata_only", "teacher_abstention_keeps_pre_run_no"):
        if teacher.get(key) is not True:
            fail(f"teacher fail-closed gate differs: {key}")
    if teacher.get("raw_target_payload_export_allowed") is not False:
        fail("raw target payload export is forbidden")
    if teacher.get("p3_target_teacher_mount_allowed") is not False:
        fail("P3 target teacher mount is forbidden")
    if teacher.get("p3_target_teacher_read_count_required") != 0:
        fail("P3 target teacher read count must be zero")
    if teacher.get("locked_actual_external_evaluator_only_after_all_hashes_freeze") is not True:
        fail("locked target may open only in the external evaluator after freeze")

    dependencies = _array(value.get("dependency_requirements"), "dependency_requirements")
    if [row.get("id") for row in dependencies if type(row) is dict] != list(BLOCKERS) or len(dependencies) != len(BLOCKERS):
        fail("dependency blocker registry differs")
    for row in dependencies:
        if row != {
            "id": row["id"],
            "required": True,
            "present": False,
            "artifact_path": None,
            "artifact_sha256": None,
        }:
            fail(f"dependency must remain explicitly absent in v1: {row.get('id')}")

    authorization = _mapping(value.get("current_authorization"), "current_authorization")
    if authorization != {
        "status": ONLY_STATUS,
        "can_emit_ready_status": False,
        "gpu_launch_authorized": False,
        "slot_pretraining_authorized": False,
        "binder_training_authorized": False,
        "generator_training_authorized": False,
        "official_runner_authorized": False,
        "parameter_updates_authorized": False,
    }:
        fail("current authorization must remain hard PRE_RUN_NO")

    claims = _mapping(value.get("claim_boundary"), "claim_boundary")
    if claims != {
        "design_registered": True,
        "experiment_run": False,
        "representation_admitted": False,
        "stable_action_representation_supported": False,
        "transferable_action_representation_supported": False,
        "scientific_claim_authorized": False,
    }:
        fail("claim boundary differs")
    return value


def validate_source_manifest(
    value: Mapping[str, Any], *, observed_file_sha256: str
) -> Mapping[str, Any]:
    if observed_file_sha256 != EXPECTED_SOURCE_FILE_SHA256:
        fail("source manifest bytes differ")
    if value.get("schema_version") != EXPECTED_SOURCE_SCHEMA or value.get("experiment_id") != EXPECTED_SOURCE_EXPERIMENT:
        fail("source manifest identity differs")
    if _verify_self_hash(value, "manifest_sha256", "source manifest") != EXPECTED_SOURCE_SELF_SHA256:
        fail("source manifest self hash differs from the byte pin")
    authority = _mapping(value.get("authority"), "source manifest.authority")
    for key in (
        "formal_sft_authorized", "generator_training_authorized",
        "generator_connection_authorized", "development_parameter_fitting_authorized",
        "locked_validation_parameter_fitting_authorized",
    ):
        if authority.get(key) is not False:
            fail(f"source manifest unexpectedly grants authority: {key}")

    pairs = _array(value.get("pairs"), "source manifest.pairs")
    if len(pairs) != 16:
        fail("source manifest must contain exactly 16 rows")
    projected = []
    for row in pairs:
        item = _mapping(row, "source manifest.pairs[]")
        ordinal = item.get("ordinal")
        projected.append((
            ordinal,
            item.get("pair_id"),
            item.get("uuid"),
            item.get("interaction_family"),
            item.get("report_split"),
            _expected_partition(ordinal),
        ))
        if item.get("formal_sft_authorized") is not False:
            fail(f"source row {ordinal} unexpectedly authorizes fitting")
    if tuple(projected) != EXPECTED_ROWS:
        fail("source manifest 16-row identities/order/split differ")
    if Counter(row[3] for row in projected) != Counter({family: 4 for family in FAMILIES}):
        fail("source manifest family balance differs")
    return value


def build_preflight_receipt(
    *,
    preregistration: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_manifest_file_sha256: str,
    provisional_registry: Mapping[str, Any],
    provisional_registry_file_sha256: str,
) -> Mapping[str, Any]:
    validate_preregistration(preregistration)
    validate_source_manifest(
        source_manifest, observed_file_sha256=source_manifest_file_sha256
    )
    try:
        p0_provisional.validate_provisional_registry(
            provisional_registry,
            observed_registry_file_sha256=provisional_registry_file_sha256,
            actual_manifest=source_manifest,
            actual_manifest_file_sha256=source_manifest_file_sha256,
        )
    except p0_provisional.V10A2P0ProvisionalPreflightError as error:
        fail(f"provisional P0 evidence differs: {error}")
    value = {
        "schema_version": RECEIPT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": ONLY_STATUS,
        "launch_authorized": False,
        "gpu_launch_authorized": False,
        "slot_pretraining_authorized": False,
        "binder_training_authorized": False,
        "generator_training_authorized": False,
        "official_runner_authorized": False,
        "training_executed": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "generator_forward_calls": 0,
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "source_manifest": {
            "file_sha256": source_manifest_file_sha256,
            "self_sha256": EXPECTED_SOURCE_SELF_SHA256,
            "row_count": 16,
            "fixed_split_verified": True,
            "formal_sft_authorized": False,
        },
        "p0_provisional_evidence": {
            "integrity_verified": True,
            "candidate_count": 64,
            "eligible_count_recorded": 283,
            "selected_strata": {
                "camera0_occlusion0": 16,
                "camera0_occlusion1": 16,
                "camera1_occlusion0": 16,
                "camera1_occlusion1": 16,
            },
            "exact_uuid_path_media_overlap_with_actual": 0,
            "registry_file_sha256": provisional_registry_file_sha256,
            "registry_self_sha256": p0_provisional.EXPECTED_REGISTRY_SELF_SHA256,
            "official_source_only_registry": False,
            "perceptual_exclusion_complete": False,
            "frozen_observer_qualification_complete": False,
            "p0_slot_pretraining_authorized": False,
            "blocker": p0_provisional.V10A2_BLOCKER,
        },
        "fixed_split": {
            "binder_train_ordinals": list(TRAIN_ORDINALS),
            "calibration_ordinals": list(CALIBRATION_ORDINALS),
            "locked_actual_ordinals": list(LOCKED_ORDINALS),
        },
        "design_counts": dict(EXPECTED_COUNTS),
        "frozen_base_contract_verified_not_executed": True,
        "same_state_anchor_registered_not_executed": True,
        "dual_noun_matched_noop_mean_and_null_registered_not_executed": True,
        "source_only_3x3x2x2_factorial_registered_not_executed": True,
        "pure_relative_graph_signature_registered_not_executed": True,
        "hierarchical_pair_group_graph_registered_not_executed": True,
        "dual_observer_and_three_rater_quality_gate_registered_not_executed": True,
        "p3_target_teacher_zero_read_firewall_registered_not_executed": True,
        "lofo_quality_reward_gates_registered_not_executed": True,
        "current_v3_is_mechanics_only": True,
        "authorization_blockers": list(BLOCKERS),
        "missing_dependency_count": len(BLOCKERS),
        "can_emit_ready_status": False,
        "representation_admitted": False,
        "stable_action_representation_supported": False,
        "transferable_action_representation_supported": False,
        "scientific_claim_authorized": False,
    }
    return {**value, "receipt_sha256": object_sha256(value)}


def run_preflight(
    preregistration_path: Path = DEFAULT_PREREGISTRATION_PATH,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST_PATH,
    provisional_registry_path: Path = DEFAULT_PROVISIONAL_REGISTRY_PATH,
) -> Mapping[str, Any]:
    preregistration = load_json(preregistration_path)
    source_manifest = load_json(source_manifest_path)
    provisional_registry = load_json(provisional_registry_path)
    return build_preflight_receipt(
        preregistration=preregistration,
        source_manifest=source_manifest,
        source_manifest_file_sha256=file_sha256(source_manifest_path),
        provisional_registry=provisional_registry,
        provisional_registry_file_sha256=file_sha256(provisional_registry_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST_PATH)
    parser.add_argument(
        "--provisional-registry",
        type=Path,
        default=DEFAULT_PROVISIONAL_REGISTRY_PATH,
    )
    args = parser.parse_args(argv)
    receipt = run_preflight(
        args.preregistration,
        args.source_manifest,
        args.provisional_registry,
    )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKERS",
    "CALIBRATION_ORDINALS",
    "DEFAULT_PREREGISTRATION_PATH",
    "DEFAULT_PROVISIONAL_REGISTRY_PATH",
    "DEFAULT_SOURCE_MANIFEST_PATH",
    "EXPECTED_COUNTS",
    "EXPECTED_PREREGISTRATION_SHA256",
    "EXPECTED_ROWS",
    "LOCKED_ORDINALS",
    "ONLY_STATUS",
    "TRAIN_ORDINALS",
    "V10A2PreflightError",
    "build_preflight_receipt",
    "canonical_json_bytes",
    "file_sha256",
    "load_json",
    "main",
    "object_sha256",
    "run_preflight",
    "validate_preregistration",
    "validate_source_manifest",
]
