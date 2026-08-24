from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from motive import goku_full_motion_contract as contract
from motive import goku_full_motion_finalize as finalizer
from motive import goku_full_motion_instruction as instruction
from motive import goku_full_motion_postcheck as postcheck
from motive import goku_full_motion_qwen as qwen


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _evidence(description: str) -> dict:
    return {
        "schema_version": contract.MOTION_EVIDENCE_SCHEMA,
        "start_frame": 0,
        "end_frame": 80,
        "description": description,
    }


def _source(iid: str = "postcheck-sample-001") -> dict:
    return {
        "schema_version": contract.SOURCE_CENSUS_SCHEMA,
        "iid": iid,
        "clip": {
            "schema_version": contract.CLIP_SCHEMA,
            "frame_count": 81,
            "fps": "25/1",
            "timeline_span_seconds": 3.2,
            "single_continuous_shot": True,
        },
        "source_quality": "high",
        "scene_description": "A walking person and a seated dog are in a park",
        "i0_visible_entities": [
            "the walking person on the left",
            "the seated dog on the right",
        ],
        "i0_entity_registry": [
            {
                "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                "entity_id": "entity_01",
                "entity_type": "person",
                "stable_reference": "the walking person on the left",
                "i0_bbox_xyxy_1000": [50, 100, 450, 950],
                "viewer_region": "center_left",
                "region_ordinal": 1,
                "role": "dynamic_subject",
                "visible_at_i0": True,
                "reachable_at_i0": True,
                "confidence": "high",
            },
            {
                "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                "entity_id": "entity_02",
                "entity_type": "animal",
                "stable_reference": "the seated dog on the right",
                "i0_bbox_xyxy_1000": [550, 200, 900, 800],
                "viewer_region": "center_right",
                "region_ordinal": 1,
                "role": "static_salient",
                "visible_at_i0": True,
                "reachable_at_i0": False,
                "confidence": "high",
            },
        ],
        "motion_inventory_complete": True,
        "crowd_or_unresolved_motion": False,
        "diffuse_unresolved_motion": False,
        "dynamic_units": [
            {
                "schema_version": contract.SOURCE_DYNAMIC_UNIT_SCHEMA,
                "unit_id": "unit_01",
                "entity_id": "entity_01",
                "entity_type": "person",
                "stable_reference": "the walking person on the left",
                "visible_at_i0": True,
                "independent_motion": True,
                "i0_state": "The person stands with both arms down",
                "source_action_signature": "walk_forward",
                "source_motion": "walks steadily forward with both arms down",
                "source_motion_components": [
                    {
                        "schema_version": contract.SOURCE_MOTION_COMPONENT_SCHEMA,
                        "component_id": "component_01",
                        "component_type": "locomotion",
                        "motion_signature": "walk_forward",
                        "motion_description": "walks steadily forward",
                        "dependent_entity_ids": [],
                        "motion_evidence": [
                            _evidence(
                                "the person's legs and body advance across the park"
                            )
                        ],
                    }
                ],
                "motion_evidence": [
                    _evidence("the person's legs and body advance across the park")
                ],
                "confidence": "high",
            }
        ],
        "static_salient_people": [
            {
                "schema_version": contract.SOURCE_STATIC_ENTITY_SCHEMA,
                "unit_id": "static_person_01",
                "entity_id": "entity_02",
                "entity_type": "animal",
                "stable_reference": "the seated dog on the right",
                "visible_at_i0": True,
                "i0_state": "The dog sits facing the camera",
                "source_state": "remain_still",
                "motion_evidence": [
                    _evidence("the seated dog holds the same resting pose")
                ],
                "confidence": "high",
            }
        ],
        "camera": {
            "schema_version": contract.SOURCE_CAMERA_SCHEMA,
            "camera_id": "camera",
            "motion_class": "locked_off",
            "motion_signature": "locked_off",
            "motion_description": "locked off",
            "dynamic": False,
            "motion_evidence": [
                _evidence("trees and path remain aligned throughout the shot")
            ],
            "confidence": "high",
        },
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _i0_grounding(source: dict) -> dict:
    units_by_entity = {
        unit["entity_id"]: unit
        for unit in (
            *source["dynamic_units"],
            *source["static_salient_people"],
        )
    }
    subjects = []
    for entity in source["i0_entity_registry"]:
        if entity["entity_type"] not in {"person", "animal"}:
            continue
        unit = units_by_entity[entity["entity_id"]]
        subjects.append(
            {
                "schema_version": qwen.I0_GROUNDED_SUBJECT_SCHEMA,
                "subject_id": entity["entity_id"],
                "entity_type": entity["entity_type"],
                "stable_reference": entity["stable_reference"],
                "i0_bbox_xyxy_1000": entity["i0_bbox_xyxy_1000"],
                "i0_state": unit["i0_state"],
                "viewer_left_extremity_height": "below_waist",
                "viewer_left_extremity_state": (
                    "viewer-left extremity rests below the waist at I0"
                ),
                "viewer_right_extremity_height": "below_waist",
                "viewer_right_extremity_state": (
                    "viewer-right extremity rests below the waist at I0"
                ),
                "confidence": "high",
            }
        )
    grounding = {
        "schema_version": qwen.I0_GROUNDING_SCHEMA,
        "iid": source["iid"],
        "subjects": subjects,
        "all_visible_people_and_animals_enumerated": True,
        "uncertainty_codes": [],
        "confidence": "high",
    }
    validated = qwen.validate_i0_grounding(
        grounding, expected_iid=source["iid"]
    )
    qwen.validate_source_census_i0_binding(source, validated)
    return validated


def _coverage_artifacts(
    source: dict,
    secondary_source: dict,
    i0_grounding: dict,
    source_inventory_alignment: dict,
) -> tuple[dict, dict, dict]:
    proposal = {
        "schema_version": qwen.CHANGE_REGION_SCHEMA,
        "proposal_id": "proposal_01",
        "cell_row": 3,
        "cell_column": 2,
        "bbox_xyxy_1000": [250, 500, 500, 750],
        "changed_pixel_count": 64,
        "bbox_area_pixels": 100,
        "changed_fraction_ppm": 640_000,
        "delta_at_percentile_milli": 20_000,
    }
    proposals = qwen.validate_change_region_proposals(
        {
            "schema_version": qwen.CHANGE_REGION_PROPOSALS_SCHEMA,
            "iid": source["iid"],
            "frame_indices": list(qwen.AUTHORITY_FRAME_INDICES),
            "grid_rows": qwen.AUTHORITY_GRID_ROWS,
            "grid_columns": qwen.AUTHORITY_GRID_COLUMNS,
            "delta_threshold": qwen.CHANGE_REGION_DELTA_THRESHOLD,
            "minimum_changed_fraction_ppm": (
                qwen.CHANGE_CELL_MIN_CHANGED_FRACTION_PPM
            ),
            "delta_percentile_milli": qwen.CHANGE_CELL_DELTA_PERCENTILE_MILLI,
            "minimum_delta_at_percentile_milli": (
                qwen.CHANGE_CELL_MIN_DELTA_AT_PERCENTILE_MILLI
            ),
            "regions": [proposal],
            "active_cell_count": 1,
            "global_changed_fraction_ppm": 40_000,
            "all_active_cells_emitted": True,
        },
        expected_iid=source["iid"],
    )
    units = {
        item["entity_id"]: item
        for item in (*source["dynamic_units"], *source["static_salient_people"])
    }
    subjects = []
    authority_by_entity: dict[str, str] = {}
    for entity in source["i0_entity_registry"]:
        unit = units[entity["entity_id"]]
        authority_id = f"authority_subject_{len(subjects) + 1:02d}"
        authority_by_entity[entity["entity_id"]] = authority_id
        dynamic = entity["role"] == "dynamic_subject"
        subjects.append(
            {
                "schema_version": qwen.COVERAGE_AUTHORITY_SUBJECT_SCHEMA,
                "authority_id": authority_id,
                "entity_type": entity["entity_type"],
                "stable_reference": entity["stable_reference"],
                "i0_bbox_xyxy_1000": entity["i0_bbox_xyxy_1000"],
                "temporal_extent_bbox_xyxy_1000": entity[
                    "i0_bbox_xyxy_1000"
                ],
                "motion_role": "dynamic" if dynamic else "static_salient",
                "motion_component_types": (
                    [
                        component["component_type"]
                        for component in unit["source_motion_components"]
                    ]
                    if dynamic
                    else []
                ),
                "motion_evidence": unit["motion_evidence"],
                "confidence": "high",
            }
        )
    dynamic_entity = source["dynamic_units"][0]["entity_id"]
    inventory = qwen.validate_coverage_authority_inventory(
        {
            "schema_version": qwen.COVERAGE_AUTHORITY_INVENTORY_SCHEMA,
            "iid": source["iid"],
            "i0_subjects": subjects,
            "extra_dynamic_entities": [],
            "camera": {
                "schema_version": qwen.COVERAGE_AUTHORITY_CAMERA_SCHEMA,
                "dynamic": source["camera"]["dynamic"],
                "motion_class": source["camera"]["motion_class"],
                "motion_evidence": source["camera"]["motion_evidence"],
                "confidence": "high",
            },
            "all_i0_people_and_animals_enumerated": True,
            "all_dynamic_entities_enumerated": True,
            "uncertainty_codes": [],
            "confidence": "high",
        },
        expected_iid=source["iid"],
    )
    assignments = qwen.validate_coverage_authority_assignments(
        {
            "schema_version": qwen.COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA,
            "iid": source["iid"],
            "coverage_authority_inventory_sha256": contract.object_sha256(
                inventory
            ),
            "change_region_proposals_sha256": contract.object_sha256(
                proposals
            ),
            "allowed_owner_map_sha256": contract.object_sha256(
                qwen.build_coverage_authority_allowed_owner_map(
                    coverage_authority_inventory=inventory,
                    change_region_proposals=proposals,
                )
            ),
            "change_region_assignments": [
                {
                    "schema_version": qwen.CHANGE_REGION_ASSIGNMENT_SCHEMA,
                    "proposal_id": "proposal_01",
                    "assignment_kind": "entity",
                    "authority_entity_ids": [
                        authority_by_entity[dynamic_entity]
                    ],
                    "resolution_reason": (
                        "The active grid cell overlaps the independently "
                        "moving person's body across temporal checkpoints"
                    ),
                    "reject_reason_code": None,
                    "confidence": "high",
                }
            ],
            "all_change_regions_resolved": True,
            "uncertainty_codes": [],
            "confidence": "high",
        },
        expected_iid=source["iid"],
        coverage_authority_inventory=inventory,
        change_region_proposals=proposals,
    )
    authority = qwen.build_coverage_authority(
        coverage_authority_inventory=inventory,
        coverage_authority_assignments=assignments,
        change_region_proposals=proposals,
    )
    alignment = qwen.build_coverage_authority_alignment(
        coverage_authority=authority,
        change_region_proposals=proposals,
        i0_grounding=i0_grounding,
        primary=source,
        secondary=secondary_source,
        source_inventory_alignment=source_inventory_alignment,
    )
    return proposals, authority, alignment


def _plan(source: dict, *, shared_base: bool = True) -> dict:
    reference = "the walking person on the left"
    if shared_base:
        relation = "explicit_shared_base_with_novel_action"
        suppressed = False
        shared = "walk forward"
        clause = (
            "have the walking person on the left walk forward and raise the "
            "right hand to wave with an open palm"
        )
    else:
        relation = "replace"
        suppressed = True
        shared = None
        clause = (
            "have the walking person on the left stop walking, plant both feet, "
            "and raise the right hand to wave with an open palm"
        )
    target = {
        "schema_version": contract.TARGET_DYNAMIC_UNIT_SCHEMA,
        "unit_id": "unit_01",
        "entity_id": "entity_01",
        "stable_reference": reference,
        "target_action_signature": (
            "walk_forward_open_palm_wave" if shared_base else "stop_and_wave"
        ),
        "motion_relation": relation,
        "source_motion_suppressed": suppressed,
        "explicit_shared_base_motion": shared,
        "source_component_dispositions": [
            {
                "schema_version": contract.TARGET_COMPONENT_DISPOSITION_SCHEMA,
                "component_id": "component_01",
                "disposition": (
                    "explicit_shared_base" if shared_base else "suppress"
                ),
                "explicit_target_motion": (
                    "walk forward" if shared_base else None
                ),
            }
        ],
        "novel_target_motion": "raise the right hand to wave with an open palm",
        "target_clause": clause,
        "substantive_change": True,
        "starts_at_i0": True,
        "i0_executable": True,
        "complete_within_clip": True,
        "completion_time_seconds": 3.0,
        "ordered_stages": [
            "the walking person on the left raises the right forearm",
            "the walking person on the left opens the palm and waves",
        ],
        "interaction_entity_ids": [],
        "required_i0_entity_ids": ["entity_01"],
    }
    return {
        "schema_version": contract.TARGET_PLAN_SCHEMA,
        "iid": source["iid"],
        "source_census_sha256": contract.object_sha256(source),
        "dynamic_unit_targets": [target],
        "static_person_targets": [
            {
                "schema_version": contract.TARGET_STATIC_ENTITY_SCHEMA,
                "unit_id": "static_person_01",
                "entity_id": "entity_02",
                "entity_type": "animal",
                "stable_reference": "the seated dog on the right",
                "target_state": "remain_still",
                "target_clause": "have the seated dog on the right remain still",
            }
        ],
        "camera_target": {
            "schema_version": contract.TARGET_CAMERA_SCHEMA,
            "camera_id": "camera",
            "motion_relation": "preserve_static",
            "target_motion_class": "locked_off",
            "target_motion_signature": "locked_off",
            "target_motion_description": "locked off",
            "target_clause": "keep the camera locked off",
            "source_motion_suppressed": False,
            "substantive_change": False,
            "starts_at_i0": True,
            "i0_executable": True,
            "complete_within_clip": True,
            "completion_time_seconds": 3.2,
            "ordered_stages": ["keep the camera locked off for the full clip"],
        },
        "preservation": {
            "schema_version": contract.TARGET_PRESERVATION_SCHEMA,
            "preserve_identity": True,
            "preserve_appearance": True,
            "preserve_scene": True,
            "allow_new_entities": False,
            "allow_removed_entities": False,
        },
        "coverage": {
            "schema_version": contract.TARGET_COVERAGE_SCHEMA,
            "required_dynamic_unit_ids": ["unit_01"],
            "planned_changed_unit_ids": ["unit_01"],
            "missing_unit_ids": [],
            "extra_unit_ids": [],
            "required_static_person_ids": ["static_person_01"],
            "constrained_static_person_ids": ["static_person_01"],
            "camera_clause_present": True,
        },
        "i0_executable": True,
        "no_new_prerequisites": True,
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _critic(source: dict, plan: dict, compiled: dict) -> dict:
    return {
        "schema_version": contract.COVERAGE_CRITIC_SCHEMA,
        "iid": source["iid"],
        "source_census_sha256": contract.object_sha256(source),
        "target_plan_sha256": contract.object_sha256(plan),
        "instruction_sha256": compiled["instruction_sha256"],
        "required_dynamic_unit_ids": ["unit_01"],
        "plan_covered_dynamic_unit_ids": ["unit_01"],
        "instruction_covered_dynamic_unit_ids": ["unit_01"],
        "missing_unit_ids": [],
        "extra_unit_ids": [],
        "ambiguous_unit_ids": [],
        "per_unit_substantive_change": {"unit_01": True},
        "source_future_suppressed_or_explicit": {"unit_01": True},
        "camera_clause_present": True,
        "camera_target_valid": True,
        "required_static_person_ids": ["static_person_01"],
        "static_people_preserved": {"static_person_01": True},
        "i0_executable": True,
        "no_new_prerequisites": True,
        "no_unrequested_action": True,
        "verdict": "pass",
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _generation_row(
    source_path: Path,
    anchor_path: Path,
    *,
    shared_base: bool = True,
) -> dict:
    source = _source()
    i0_grounding = _i0_grounding(source)
    secondary_source = copy.deepcopy(source)
    source_inventory_alignment = contract.build_source_inventory_alignment(
        primary=source,
        secondary=secondary_source,
    )
    (
        change_region_proposals,
        coverage_authority,
        coverage_authority_alignment,
    ) = _coverage_artifacts(
        source,
        secondary_source,
        i0_grounding,
        source_inventory_alignment,
    )
    plan = _plan(source, shared_base=shared_base)
    compiled = instruction.compile_full_motion_instruction(source, plan)
    critic = _critic(source, plan, compiled)
    canonical_source, source_canonicalization = (
        contract.canonicalize_source_census_model_output(
            source, source["iid"]
        )
    )
    canonical_secondary, secondary_canonicalization = (
        contract.canonicalize_source_census_model_output(
            secondary_source, source["iid"]
        )
    )
    canonical_plan, target_canonicalization = (
        contract.canonicalize_target_plan_model_output(plan, source)
    )
    if (
        canonical_source != source
        or canonical_secondary != secondary_source
        or canonical_plan != plan
    ):
        raise AssertionError("canonical fixture unexpectedly changed")
    source_canonicalization_sha = contract.object_sha256(
        source_canonicalization
    )
    secondary_canonicalization_sha = contract.object_sha256(
        secondary_canonicalization
    )
    target_canonicalization_sha = contract.object_sha256(
        target_canonicalization
    )
    hard_gate = qwen.build_hard_gate(
        change_region_proposals=change_region_proposals,
        coverage_authority=coverage_authority,
        coverage_authority_alignment=coverage_authority_alignment,
        i0_grounding=i0_grounding,
        source_census=source,
        source_census_canonicalization=source_canonicalization,
        secondary_source_census=secondary_source,
        secondary_source_census_canonicalization=secondary_canonicalization,
        source_inventory_alignment=source_inventory_alignment,
        target_plan=plan,
        target_plan_canonicalization=target_canonicalization,
        compiled_instruction=compiled,
        coverage_critic=critic,
    )
    if hard_gate["decision"] != "pass":
        raise AssertionError(f"fixture hard gate rejected: {hard_gate}")
    motion_spec = {
        "schema_version": finalizer.MOTION_SPEC_SCHEMA,
        "change_region_proposals": change_region_proposals,
        "coverage_authority": coverage_authority,
        "i0_grounding": i0_grounding,
        "source_census": source,
        "secondary_source_census": secondary_source,
        "source_inventory_alignment": source_inventory_alignment,
        "coverage_authority_alignment": coverage_authority_alignment,
        "target_plan": plan,
        "compiled_instruction": compiled,
        "coverage_critic": critic,
        "full_motion_contract": contract.build_contract(
            source_census=source, target_plan=plan
        ),
        "qwen_result_digest": "1" * 64,
        "qwen_provenance_digest": "2" * 64,
    }
    selected_media = {
        "frame_count": 81,
        "fps": 25.0,
        "width": 832,
        "height": 480,
    }
    temporal = {
        "schema_version": finalizer.TEMPORAL_GEOMETRY_SCHEMA,
        "source_frame_count": 81,
        "source_frame_rate": "25/1",
        "source_timeline_span_seconds": 3.2,
        "target_frame_count": 81,
        "target_frame_rate": "25/1",
        "target_timeline_span_seconds": 3.2,
        "requires_exact_frame_count_and_rate_match": True,
    }
    source_sha = contract.object_sha256(source)
    secondary_sha = contract.object_sha256(secondary_source)
    alignment_sha = contract.object_sha256(source_inventory_alignment)
    authority_inventory = coverage_authority["inventory"]
    authority_assignments = coverage_authority["assignments"]
    authority_inventory_sha = contract.object_sha256(authority_inventory)
    authority_assignments_sha = contract.object_sha256(
        authority_assignments
    )
    receipt_digest = "3" * 64
    qwen_evidence = {
        "schema_version": finalizer.QWEN_EVIDENCE_SCHEMA,
        "record_schema_version": qwen.RECORD_SCHEMA,
        "input_digest": "4" * 64,
        "result_digest": motion_spec["qwen_result_digest"],
        "provenance_digest": motion_spec["qwen_provenance_digest"],
        "config_digest": "5" * 64,
        "run_config_digest": "6" * 64,
        "implementation_digest": "7" * 64,
        "visual_input_digest": "8" * 64,
        "media_verification": {"verified": True},
        "hard_gate": hard_gate,
        "change_region_proposals_digest": contract.object_sha256(
            change_region_proposals
        ),
        "coverage_authority_inventory_prompt_digest": "9" * 64,
        "coverage_authority_inventory_visual_input_digest": "a" * 64,
        "coverage_authority_inventory_digest": authority_inventory_sha,
        "coverage_authority_assignments_prompt_digest": "b" * 64,
        "coverage_authority_assignments_visual_input_digest": "c" * 64,
        "coverage_authority_assignments_digest": authority_assignments_sha,
        "coverage_authority_digest": contract.object_sha256(
            coverage_authority
        ),
        "coverage_authority_alignment_digest": contract.object_sha256(
            coverage_authority_alignment
        ),
        "i0_grounding_digest": contract.object_sha256(i0_grounding),
        "source_census_canonicalization": source_canonicalization,
        "source_census_canonicalization_digest": (
            source_canonicalization_sha
        ),
        "source_census_digest": source_sha,
        "secondary_source_census_canonicalization": (
            secondary_canonicalization
        ),
        "secondary_source_census_canonicalization_digest": (
            secondary_canonicalization_sha
        ),
        "secondary_source_census_digest": secondary_sha,
        "source_inventory_alignment_digest": alignment_sha,
        "target_plan_canonicalization": target_canonicalization,
        "target_plan_canonicalization_digest": target_canonicalization_sha,
        "target_plan_digest": contract.object_sha256(plan),
        "compiled_instruction_digest": contract.object_sha256(compiled),
        "full_motion_contract_digest": contract.object_sha256(
            motion_spec["full_motion_contract"]
        ),
        "coverage_critic_digest": contract.object_sha256(critic),
        "shard_index": 0,
        "num_shards": 8,
        "receipt_digest": receipt_digest,
        "receipt_sha256": "e" * 64,
        "output_sha256": "f" * 64,
        "model_path": "/models/Qwen3-VL-32B-Instruct",
        "model_revision": "revision",
        "transformers_version": "5.5.4",
    }
    # The frozen generation schema carries the complete Qwen-v6 record, not
    # only its two self-reported digests.  Populate every closed record key so
    # the finalizer/postcheck/select validators can independently project the
    # canonical result and recompute the complete provenance digest.
    qwen_record = {key: None for key in qwen._RECORD_KEYS}
    qwen_record.update(
        {
            "schema_version": qwen.RECORD_SCHEMA,
            "iid": source["iid"],
            "group_id": "group-001",
            "family": "motion_editing",
            "status": "ok",
            "error_type": None,
            "error": None,
            "input_digest": qwen_evidence["input_digest"],
            "config_digest": qwen_evidence["config_digest"],
            "run_config_digest": qwen_evidence["run_config_digest"],
            "implementation_digest": qwen_evidence[
                "implementation_digest"
            ],
            "model_path": qwen_evidence["model_path"],
            "model_revision": qwen_evidence["model_revision"],
            "transformers_version": qwen_evidence[
                "transformers_version"
            ],
            "shard_index": qwen_evidence["shard_index"],
            "num_shards": qwen_evidence["num_shards"],
            "failure_stage": None,
            "pipeline_stage": "coverage_critic",
            "pipeline_decision": "pass",
            "resolved_src_video": str(source_path),
            "resolved_anchor_image": str(anchor_path),
            "media_verification": qwen_evidence["media_verification"],
            "visual_input_digest": qwen_evidence["visual_input_digest"],
            "change_region_proposals": change_region_proposals,
            "change_region_proposals_digest": qwen_evidence[
                "change_region_proposals_digest"
            ],
            "coverage_authority_inventory_raw": json.dumps(
                authority_inventory,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "coverage_authority_inventory_prompt_digest": qwen_evidence[
                "coverage_authority_inventory_prompt_digest"
            ],
            "coverage_authority_inventory_visual_input_digest": qwen_evidence[
                "coverage_authority_inventory_visual_input_digest"
            ],
            "coverage_authority_inventory_validated_from": "original",
            "coverage_authority_inventory_digest": authority_inventory_sha,
            "coverage_authority_assignments_raw": json.dumps(
                authority_assignments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "coverage_authority_assignments_prompt_digest": qwen_evidence[
                "coverage_authority_assignments_prompt_digest"
            ],
            "coverage_authority_assignments_visual_input_digest": qwen_evidence[
                "coverage_authority_assignments_visual_input_digest"
            ],
            "coverage_authority_assignments_validated_from": "original",
            "coverage_authority_assignments_digest": (
                authority_assignments_sha
            ),
            "coverage_authority": coverage_authority,
            "coverage_authority_digest": qwen_evidence[
                "coverage_authority_digest"
            ],
            "i0_grounding": i0_grounding,
            "i0_grounding_digest": qwen_evidence["i0_grounding_digest"],
            "source_census": source,
            "source_census_digest": qwen_evidence[
                "source_census_digest"
            ],
            "source_census_canonicalization": source_canonicalization,
            "source_census_canonicalization_digest": qwen_evidence[
                "source_census_canonicalization_digest"
            ],
            "secondary_source_census": secondary_source,
            "secondary_source_census_digest": qwen_evidence[
                "secondary_source_census_digest"
            ],
            "secondary_source_census_canonicalization": (
                secondary_canonicalization
            ),
            "secondary_source_census_canonicalization_digest": qwen_evidence[
                "secondary_source_census_canonicalization_digest"
            ],
            "source_inventory_alignment": source_inventory_alignment,
            "source_inventory_alignment_digest": qwen_evidence[
                "source_inventory_alignment_digest"
            ],
            "coverage_authority_alignment": coverage_authority_alignment,
            "coverage_authority_alignment_digest": qwen_evidence[
                "coverage_authority_alignment_digest"
            ],
            "target_plan_raw": json.dumps(
                plan,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "target_plan_validated_from": "canonicalized_original",
            "target_plan": plan,
            "target_plan_digest": qwen_evidence["target_plan_digest"],
            "target_plan_canonicalization": target_canonicalization,
            "target_plan_canonicalization_digest": qwen_evidence[
                "target_plan_canonicalization_digest"
            ],
            "compiled_instruction": compiled,
            "compiled_instruction_digest": qwen_evidence[
                "compiled_instruction_digest"
            ],
            "full_motion_contract": motion_spec["full_motion_contract"],
            "full_motion_contract_digest": qwen_evidence[
                "full_motion_contract_digest"
            ],
            "coverage_critic": critic,
            "coverage_critic_digest": qwen_evidence[
                "coverage_critic_digest"
            ],
            "hard_gate": hard_gate,
        }
    )
    qwen_record["result_digest"] = _object_sha(
        qwen.qwen_result_payload(qwen_record)
    )
    qwen_record["provenance_digest"] = qwen.qwen_provenance_digest(
        qwen_record
    )
    motion_spec["qwen_result_digest"] = qwen_record["result_digest"]
    motion_spec["qwen_provenance_digest"] = qwen_record[
        "provenance_digest"
    ]
    qwen_evidence["result_digest"] = qwen_record["result_digest"]
    qwen_evidence["provenance_digest"] = qwen_record[
        "provenance_digest"
    ]
    qwen_evidence["qwen_record_payload"] = qwen_record
    row = {
        "schema_version": finalizer.GENERATION_SCHEMA,
        "iid": source["iid"],
        "group_id": "group-001",
        "family": "motion_editing",
        "source_video": str(source_path),
        "resolved_source_video": str(source_path),
        "source_video_sha256": _sha_bytes(source_path.read_bytes()),
        "anchor_image": str(anchor_path),
        "resolved_anchor_image": str(anchor_path),
        "anchor_sha256": _sha_bytes(anchor_path.read_bytes()),
        "selected_media_evidence": selected_media,
        "selected_media_evidence_sha256": _object_sha(selected_media),
        "strict_temporal_geometry": temporal,
        "edit_instruction": compiled["edit_instruction"],
        "edit_instruction_sha256": compiled["instruction_sha256"],
        "action_change_substantive": "yes",
        "manifest_role": "review_proposal",
        "human_review_status": "pending",
        "generation_authorized": False,
        "production_eligible": False,
        "approval": None,
        "authorization_interface_available": False,
        "annotation_source": "qwen3-vl-32b",
        "human_reviewed": False,
        "motion_spec": motion_spec,
        "motion_spec_sha256": _object_sha(motion_spec),
        "qwen_evidence": qwen_evidence,
        "full_motion_finalization": {
            "schema_version": finalizer.FINALIZATION_ROW_SCHEMA,
            "policy_version": finalizer.POLICY_VERSION,
            "candidate_rank": 1,
            "review_rank": 1,
            "selection_bucket": "primary",
            "dynamic_unit_count": 1,
            "target_action_signatures": [
                plan["dynamic_unit_targets"][0]["target_action_signature"]
            ],
            "family": "motion_editing",
            "required_canary": False,
            "qwen_shard_index": 0,
            "qwen_receipt_digest": receipt_digest,
        },
    }
    finalizer.validate_generation_row(row)
    return row


def _frame_evidence(first: str, last: str) -> list[dict]:
    return [
        {"frame_index": 0, "observation": first},
        {"frame_index": 80, "observation": last},
    ]


def _target_census() -> dict:
    return {
        "schema_version": postcheck.TARGET_CENSUS_SCHEMA,
        "single_continuous_shot": "yes",
        "artifact_level": "none",
        "motion_units": [
            {
                "observed_unit_id": "obs_01",
                "stable_reference": "the person on the left",
                "entity_type": "person",
                "observed_motion": "walks forward while raising a hand and waving",
                "frame_evidence": _frame_evidence(
                    "both arms are initially down",
                    "the right hand is raised with an open palm",
                ),
            }
        ],
        "static_salient_people": [
            {
                "observed_static_id": "static_obs_01",
                "stable_reference": "the seated dog on the right",
                "entity_type": "animal",
                "frame_evidence": _frame_evidence(
                    "the dog is seated in the same place",
                    "the dog remains seated in the same place",
                ),
            }
        ],
        "camera": {
            "motion_class": "locked_off",
            "motion_description": "the camera remains locked off",
            "frame_evidence": _frame_evidence(
                "the background begins aligned",
                "the background remains aligned",
            ),
        },
        "uncertainty_codes": [],
    }


def _judgment(*, shared_base: bool = True) -> dict:
    return {
        "schema_version": postcheck.CLAUSE_JUDGE_SCHEMA,
        "motion_unit_results": [
            {
                "unit_id": "unit_01",
                "motion_relation": (
                    "explicit_shared_base_with_novel_action"
                    if shared_base
                    else "replace"
                ),
                "census_match": "moving",
                "census_observed_id": "obs_01",
                "fulfilled": "yes",
                "source_future_handling_fulfilled": "yes",
                "explicit_shared_base_fulfilled": (
                    "yes" if shared_base else "not_applicable"
                ),
                "substantive_change_visible": "yes",
                "observed_target_motion": (
                    "the person walks forward and adds an open-palm wave"
                ),
                "frame_evidence": _frame_evidence(
                    "the person starts with the hand down",
                    "the person walks while waving with an open palm",
                ),
            }
        ],
        "static_entity_results": [
            {
                "static_id": "static_person_01",
                "census_match": "static",
                "census_observed_id": "static_obs_01",
                "remain_still": "yes",
                "frame_evidence": _frame_evidence(
                    "the dog is seated",
                    "the dog remains seated",
                ),
            }
        ],
        "camera_result": {
            "camera_id": "camera",
            "fulfilled": "yes",
            "observed_motion_class": "locked_off",
            "observed_motion_signature": "locked off",
            "source_camera_motion_suppressed": "not_applicable",
            "substantive_target_camera_change_visible": "not_applicable",
            "frame_evidence": _frame_evidence(
                "the camera starts locked",
                "the camera remains locked",
            ),
        },
        "preservation": {
            "identity": "pass",
            "appearance": "pass",
            "scene": "pass",
            "entity_inventory": "pass",
            "frame_evidence": [
                {
                    "source_frame_index": 0,
                    "target_frame_index": 0,
                    "observation": "the same person, dog, and park are visible",
                },
                {
                    "source_frame_index": 80,
                    "target_frame_index": 80,
                    "observation": "identity, appearance, and scene remain matched",
                },
            ],
        },
        "no_extra_actions": {
            "status": "pass",
            "observed_extra_actions": [],
            "frame_evidence": [],
        },
        "single_continuous_shot": "yes",
        "artifact_free": "yes",
        "uncertainty_codes": [],
        "decision": "pass",
    }


def _expected_visual_contract(source: dict, plan: dict) -> dict:
    return {
        "dynamic_unit_programs": [
            {
                "unit_id": target["unit_id"],
                "source_unit": source_unit,
                "target_unit": target,
                "compiled_clause": "bound dynamic clause",
            }
            for source_unit, target in zip(
                source["dynamic_units"],
                plan["dynamic_unit_targets"],
                strict=True,
            )
        ],
        "static_entity_programs": [
            {
                "unit_id": target["unit_id"],
                "source_entity": source_entity,
                "target_constraint": target,
                "compiled_clause": "bound static clause",
            }
            for source_entity, target in zip(
                source["static_salient_people"],
                plan["static_person_targets"],
                strict=True,
            )
        ],
        "camera_program": {
            "source_camera": source["camera"],
            "target_camera": plan["camera_target"],
            "compiled_clause": "bound camera clause",
        },
    }


def _temporal_policy() -> dict:
    return {
        "policy_version": "wan22-i2v-source-timebase-preserving-v1",
        "source": {
            "frame_count": 81,
            "frame_rate": "25/1",
            "duration_seconds": 3.24,
        },
        "target": {
            "frame_count": 81,
            "frame_rate": "25/1",
            "duration_seconds": 3.24,
        },
        "frame_count_equal": True,
        "frame_rate_equal": True,
        "duration_within_tolerance": True,
    }


def _write_generation_closure(root: Path) -> dict:
    source_path = root / "source.mp4"
    anchor_path = root / "anchor.png"
    source_path.write_bytes(b"source-video")
    anchor_path.write_bytes(b"lossless-anchor")
    row = _generation_row(source_path, anchor_path)
    manifest_path = root / "generation_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_sha = _sha_bytes(manifest_path.read_bytes())

    run_payload = {
        "schema_version": "motive-wan22-i2v-batch-run-v1",
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "row_count": 1,
        },
        "selected_inputs": [{"iid": row["iid"]}],
        "generation_parameters": {
            "frame_num": 81,
            "output_container_frame_rate": "25/1",
        },
    }
    run_contract = dict(run_payload)
    run_contract["contract_digest"] = _object_sha(run_payload)
    (root / "run_contract.json").write_text(
        json.dumps(run_contract, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    sample_dir = root / "samples" / row["iid"]
    sample_dir.mkdir(parents=True)
    target_path = sample_dir / "preview.mp4"
    original_anchor = sample_dir / "conditioning_anchor_original.png"
    frame0_png = sample_dir / "conditioning_frame0.png"
    frame0_float = sample_dir / "conditioning_frame0.npy"
    source_copy = sample_dir / "source_video.mp4"
    instruction_file = sample_dir / "edit_instruction.txt"
    result_path = sample_dir / "result.json"
    target_path.write_bytes(b"target-video")
    original_anchor.write_bytes(anchor_path.read_bytes())
    frame0_png.write_bytes(b"conditioning-png")
    frame0_float.write_bytes(b"float32")
    source_copy.write_bytes(source_path.read_bytes())
    instruction_file.write_bytes(row["edit_instruction"].encode("utf-8"))
    result_path.write_text("{}\n", encoding="utf-8")
    result_digest = "3" * 64
    generated = {
        "schema_version": postcheck.WAN_GENERATED_SCHEMA,
        "iid": row["iid"],
        "group_id": row["group_id"],
        "edit_instruction": row["edit_instruction"],
        "edit_instruction_sha256": row["edit_instruction_sha256"],
        "edit_instruction_file": str(instruction_file),
        "edit_instruction_file_sha256": row["edit_instruction_sha256"],
        "edit_instruction_file_bytes": instruction_file.stat().st_size,
        "source_video": str(source_copy),
        "source_video_sha256": row["source_video_sha256"],
        "source_video_bytes": source_copy.stat().st_size,
        "conditioning_anchor_original": str(original_anchor),
        "conditioning_anchor_original_sha256": row["anchor_sha256"],
        "conditioning_frame0_float32": str(frame0_float),
        "conditioning_frame0_float32_sha256": _sha_bytes(frame0_float.read_bytes()),
        "conditioning_frame0_png": str(frame0_png),
        "conditioning_frame0_png_sha256": _sha_bytes(frame0_png.read_bytes()),
        "target_preview_mp4": str(target_path),
        "target_preview_mp4_sha256": _sha_bytes(target_path.read_bytes()),
        "result_json": str(result_path),
        "result_digest": result_digest,
        "manifest_role": row["manifest_role"],
        "production_eligible": row["production_eligible"],
        "human_review_status": row["human_review_status"],
        "generation_authorized": row["generation_authorized"],
        "approval": row["approval"],
        "action_change_substantive": row["action_change_substantive"],
        "first_frame_policy": "wan22-i2v-strict-preencode-frame0-v1",
        "mp4_decode_pixel_equality_claimed": False,
        "temporal_policy": _temporal_policy(),
    }
    generated_path = root / "generated_manifest.jsonl"
    generated_path.write_text(
        json.dumps(generated, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    complete_payload = {
        "schema_version": postcheck.WAN_COMPLETE_SCHEMA,
        "contract_digest": run_contract["contract_digest"],
        "manifest_sha256": manifest_sha,
        "selected_sample_count": 1,
        "completed_sample_count": 1,
        "generated_manifest": generated_path.name,
        "generated_manifest_sha256": _sha_bytes(generated_path.read_bytes()),
        "temporal_policy": {},
        "sample_result_digests": [result_digest],
    }
    complete = dict(complete_payload)
    complete["complete_digest"] = _object_sha(complete_payload)
    (root / "run_complete.json").write_text(
        json.dumps(complete, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "row": row,
        "manifest_path": manifest_path,
        "manifest_sha": manifest_sha,
        "run_contract": run_contract,
        "generated": generated,
        "generated_path": generated_path,
        "source_path": source_path,
        "source_copy": source_copy,
        "instruction_file": instruction_file,
        "anchor_path": anchor_path,
        "sample_dir": sample_dir,
        "target_path": target_path,
        "original_anchor": original_anchor,
        "frame0_png": frame0_png,
        "frame0_float": frame0_float,
        "result_path": result_path,
    }


class _MockBackend:
    model_path = "/mock/Qwen3-VL-32B-Instruct"
    model_revision = "mock-revision"
    transformers_version = "mock-transformers"

    def __init__(self, *, malformed_census: bool = False) -> None:
        self.malformed_census = malformed_census
        self.census_calls = 0
        self.judge_calls = 0

    def generate_target_motion_census(self, **_: object) -> str:
        self.census_calls += 1
        value = {} if self.malformed_census else _target_census()
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def generate_full_motion_judgment(self, **_: object) -> str:
        self.judge_calls += 1
        return json.dumps(_judgment(), sort_keys=True, separators=(",", ":"))


def _args(fixture: dict, output: Path, *, resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=fixture["manifest_path"],
        generation_root=fixture["manifest_path"].parent,
        generated_manifest=fixture["generated_path"],
        output=output,
        model=Path("/mock/Qwen3-VL-32B-Instruct"),
        shard_index=0,
        num_shards=1,
        max_samples=None,
        nframes=13,
        max_pixels=1_500_000,
        max_new_tokens=3072,
        attn_implementation="auto",
        allow_download=False,
        resume=resume,
        ffprobe="ffprobe",
        ffmpeg="ffmpeg",
        frame0_max_mae=8.0,
        frame0_outlier_threshold=24,
        frame0_max_outlier_fraction=0.05,
    )


def _fake_media_validator(
    row: dict, *, generated_row: dict, generation_root: Path, **_: object
) -> dict:
    return {
        "schema_version": "mock-media-binding-v1",
        "source": {"path": row["resolved_source_video"]},
        "target": {"path": generated_row["target_preview_mp4"]},
        "generation_root": str(generation_root),
        "verified": True,
    }


class FullMotionContractBindingTests(unittest.TestCase):
    def test_normalization_binds_frozen_motion_spec_and_shared_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.mp4"
            anchor_path = root / "anchor.png"
            source_path.write_bytes(b"source")
            anchor_path.write_bytes(b"anchor")
            row = _generation_row(source_path, anchor_path)
            normalized = postcheck._normalize_contract(
                row, manifest_root=root
            )
            self.assertEqual(normalized["dynamic_ids"], ["unit_01"])
            self.assertEqual(normalized["static_ids"], ["static_person_01"])
            self.assertEqual(
                normalized["dynamic_units"][0]["motion_relation"],
                "explicit_shared_base_with_novel_action",
            )
            self.assertEqual(
                normalized["static_units"][0]["entity_type"], "animal"
            )
            self.assertEqual(
                normalized["instruction"], row["edit_instruction"]
            )
            binding = normalized["qwen_evidence_binding"]
            self.assertEqual(
                binding["schema_version"],
                postcheck.QWEN_EVIDENCE_BINDING_SCHEMA,
            )
            self.assertEqual(
                binding["secondary_source_census_digest"],
                contract.object_sha256(
                    row["motion_spec"]["secondary_source_census"]
                ),
            )
            self.assertEqual(
                normalized["i0_grounding_digest"],
                contract.object_sha256(row["motion_spec"]["i0_grounding"]),
            )
            self.assertEqual(
                binding["i0_grounding_digest"],
                row["qwen_evidence"]["i0_grounding_digest"],
            )
            expected_record_sha = _object_sha(
                row["qwen_evidence"]["qwen_record_payload"]
            )
            self.assertEqual(
                normalized["qwen_record_payload_sha256"],
                expected_record_sha,
            )
            self.assertEqual(
                binding["qwen_record_payload_sha256"],
                expected_record_sha,
            )
            for artifact, digest_field in (
                ("change_region_proposals", "change_region_proposals_digest"),
                ("coverage_authority", "coverage_authority_digest"),
                (
                    "coverage_authority_alignment",
                    "coverage_authority_alignment_digest",
                ),
            ):
                expected_digest = contract.object_sha256(
                    row["motion_spec"][artifact]
                )
                self.assertEqual(normalized[digest_field], expected_digest)
                self.assertEqual(binding[digest_field], expected_digest)
            for artifact, digest_field in (
                ("inventory", "coverage_authority_inventory_digest"),
                ("assignments", "coverage_authority_assignments_digest"),
            ):
                expected_digest = contract.object_sha256(
                    row["motion_spec"]["coverage_authority"][artifact]
                )
                self.assertEqual(normalized[digest_field], expected_digest)
                self.assertEqual(binding[digest_field], expected_digest)
            for field in (
                "source_census_canonicalization",
                "secondary_source_census_canonicalization",
                "target_plan_canonicalization",
            ):
                self.assertEqual(
                    binding[field],
                    row["qwen_evidence"][field],
                )
                self.assertEqual(
                    binding[f"{field}_digest"],
                    row["qwen_evidence"][f"{field}_digest"],
                )

    def test_qwen_v6_authority_receipts_and_v5_or_shadow_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.mp4"
            anchor_path = root / "anchor.png"
            source_path.write_bytes(b"source")
            anchor_path.write_bytes(b"anchor")
            row = _generation_row(source_path, anchor_path)

            cases: list[tuple[str, dict]] = []
            old_v5 = copy.deepcopy(row)
            old_v5["qwen_evidence"]["record_schema_version"] = (
                "goku-full-motion-qwen-record-v5"
            )
            old_v5["qwen_evidence"]["qwen_record_payload"][
                "schema_version"
            ] = "goku-full-motion-qwen-record-v5"
            cases.append(("pre-v6 Qwen record", old_v5))
            missing = copy.deepcopy(row)
            del missing["qwen_evidence"]
            cases.append(("missing evidence", missing))

            missing_record_payload = copy.deepcopy(row)
            del missing_record_payload["qwen_evidence"][
                "qwen_record_payload"
            ]
            cases.append(("missing Qwen record payload", missing_record_payload))

            shadow_record_payload = copy.deepcopy(row)
            shadow_record_payload["qwen_evidence"]["qwen_record_payload"][
                "shadow_result_digest"
            ] = shadow_record_payload["qwen_evidence"]["result_digest"]
            cases.append(("shadow Qwen record payload", shadow_record_payload))

            forged_reported_digests = copy.deepcopy(row)
            forged_reported_digests["motion_spec"]["qwen_result_digest"] = (
                "a" * 64
            )
            forged_reported_digests["motion_spec"][
                "qwen_provenance_digest"
            ] = "b" * 64
            forged_reported_digests["motion_spec_sha256"] = _object_sha(
                forged_reported_digests["motion_spec"]
            )
            forged_reported_digests["qwen_evidence"]["result_digest"] = (
                "a" * 64
            )
            forged_reported_digests["qwen_evidence"][
                "provenance_digest"
            ] = "b" * 64
            forged_reported_digests["qwen_evidence"][
                "qwen_record_payload"
            ]["result_digest"] = "a" * 64
            forged_reported_digests["qwen_evidence"][
                "qwen_record_payload"
            ]["provenance_digest"] = "b" * 64
            cases.append(
                ("forged self-reported result/provenance", forged_reported_digests)
            )

            self_redigested_payload = copy.deepcopy(row)
            payload = self_redigested_payload["qwen_evidence"][
                "qwen_record_payload"
            ]
            payload["compiled_instruction"] = copy.deepcopy(
                payload["compiled_instruction"]
            )
            payload["compiled_instruction"]["edit_instruction"] += (
                " and spin in place"
            )
            payload["result_digest"] = _object_sha(
                qwen.qwen_result_payload(payload)
            )
            payload["provenance_digest"] = qwen.qwen_provenance_digest(
                payload
            )
            self_redigested_payload["qwen_evidence"]["result_digest"] = (
                payload["result_digest"]
            )
            self_redigested_payload["qwen_evidence"][
                "provenance_digest"
            ] = payload["provenance_digest"]
            self_redigested_payload["motion_spec"]["qwen_result_digest"] = (
                payload["result_digest"]
            )
            self_redigested_payload["motion_spec"][
                "qwen_provenance_digest"
            ] = payload["provenance_digest"]
            self_redigested_payload["motion_spec_sha256"] = _object_sha(
                self_redigested_payload["motion_spec"]
            )
            cases.append(
                ("self-redigested Qwen payload tamper", self_redigested_payload)
            )

            old_generation = copy.deepcopy(row)
            old_generation["schema_version"] = (
                "motive-goku-full-motion-generation-v4"
            )
            cases.append(("old generation", old_generation))

            old_evidence = copy.deepcopy(row)
            old_evidence["qwen_evidence"]["schema_version"] = (
                "motive-goku-full-motion-qwen-evidence-v4"
            )
            cases.append(("old evidence", old_evidence))

            old_record = copy.deepcopy(row)
            old_record["qwen_evidence"]["record_schema_version"] = (
                "goku-full-motion-qwen-record-v4"
            )
            cases.append(("old record", old_record))

            old_spec = copy.deepcopy(row)
            old_spec["motion_spec"]["schema_version"] = (
                "motive-goku-full-motion-generation-spec-v4"
            )
            old_spec["motion_spec_sha256"] = _object_sha(
                old_spec["motion_spec"]
            )
            cases.append(("old motion spec", old_spec))

            old_gate = copy.deepcopy(row)
            old_gate["qwen_evidence"]["hard_gate"] = {
                "schema_version": "goku-full-motion-hard-gate-v4",
                "source_census_sha256": old_gate["qwen_evidence"][
                    "source_census_digest"
                ],
                "secondary_source_census_sha256": old_gate[
                    "qwen_evidence"
                ]["secondary_source_census_digest"],
                "source_inventory_alignment_sha256": old_gate[
                    "qwen_evidence"
                ]["source_inventory_alignment_digest"],
                "decision": "pass",
                "risk_codes": [],
            }
            cases.append(("old hard gate", old_gate))

            changed_receipt = copy.deepcopy(row)
            changed_receipt["qwen_evidence"][
                "secondary_source_census_canonicalization"
            ]["canonical_sha256"] = "0" * 64
            cases.append(("changed receipt", changed_receipt))

            changed_receipt_digest = copy.deepcopy(row)
            changed_receipt_digest["qwen_evidence"][
                "target_plan_canonicalization_digest"
            ] = "0" * 64
            cases.append(("changed receipt digest", changed_receipt_digest))

            changed_grounding = copy.deepcopy(row)
            changed_grounding["motion_spec"]["i0_grounding"]["subjects"][0][
                "viewer_left_extremity_state"
            ] = "viewer-left extremity stays below the waist at I0"
            changed_grounding["motion_spec_sha256"] = _object_sha(
                changed_grounding["motion_spec"]
            )
            cases.append(("changed exact-I0 grounding", changed_grounding))

            changed_grounding_digest = copy.deepcopy(row)
            changed_grounding_digest["qwen_evidence"][
                "i0_grounding_digest"
            ] = "0" * 64
            cases.append(
                ("changed exact-I0 grounding digest", changed_grounding_digest)
            )

            changed_gate_grounding_digest = copy.deepcopy(row)
            changed_gate_grounding_digest["qwen_evidence"]["hard_gate"][
                "i0_grounding_sha256"
            ] = "0" * 64
            cases.append(
                (
                    "changed hard-gate exact-I0 digest",
                    changed_gate_grounding_digest,
                )
            )

            shadow = copy.deepcopy(row)
            shadow_secondary = copy.deepcopy(
                shadow["motion_spec"]["secondary_source_census"]
            )
            shadow_secondary["scene_description"] = (
                "A different but structurally valid shadow description"
            )
            shadow["secondary_source_census"] = shadow_secondary
            cases.append(("top-level shadow", shadow))

            missing_authority = copy.deepcopy(row)
            del missing_authority["motion_spec"]["coverage_authority"]
            missing_authority["motion_spec_sha256"] = _object_sha(
                missing_authority["motion_spec"]
            )
            cases.append(("missing coverage authority", missing_authority))

            shadow_authority = copy.deepcopy(row)
            shadow_authority["motion_spec"][
                "coverage_authority_shadow"
            ] = copy.deepcopy(
                shadow_authority["motion_spec"]["coverage_authority"]
            )
            shadow_authority["motion_spec_sha256"] = _object_sha(
                shadow_authority["motion_spec"]
            )
            cases.append(("motion-spec authority shadow", shadow_authority))

            missing_inventory_digest = copy.deepcopy(row)
            del missing_inventory_digest["qwen_evidence"][
                "coverage_authority_inventory_digest"
            ]
            cases.append(("missing A0a inventory digest", missing_inventory_digest))

            shadow_assignments = copy.deepcopy(row)
            shadow_assignments["motion_spec"]["coverage_authority"][
                "assignments_shadow"
            ] = copy.deepcopy(
                shadow_assignments["motion_spec"]["coverage_authority"][
                    "assignments"
                ]
            )
            shadow_assignments["motion_spec_sha256"] = _object_sha(
                shadow_assignments["motion_spec"]
            )
            cases.append(("A0b assignments shadow", shadow_assignments))

            redigested_proposal = copy.deepcopy(row)
            proposal = redigested_proposal["motion_spec"][
                "change_region_proposals"
            ]
            proposal["regions"][0]["changed_pixel_count"] += 1
            proposal_digest = _object_sha(proposal)
            redigested_proposal["motion_spec_sha256"] = _object_sha(
                redigested_proposal["motion_spec"]
            )
            redigested_proposal["qwen_evidence"][
                "change_region_proposals_digest"
            ] = proposal_digest
            redigested_proposal["qwen_evidence"]["hard_gate"][
                "change_region_proposals_sha256"
            ] = proposal_digest
            cases.append(("redigested proposal tamper", redigested_proposal))

            redigested_alignment = copy.deepcopy(row)
            authority_alignment = redigested_alignment["motion_spec"][
                "coverage_authority_alignment"
            ]
            authority_alignment["all_authority_entities_aligned"] = False
            authority_alignment_digest = _object_sha(authority_alignment)
            redigested_alignment["motion_spec_sha256"] = _object_sha(
                redigested_alignment["motion_spec"]
            )
            redigested_alignment["qwen_evidence"][
                "coverage_authority_alignment_digest"
            ] = authority_alignment_digest
            redigested_alignment["qwen_evidence"]["hard_gate"][
                "coverage_authority_alignment_sha256"
            ] = authority_alignment_digest
            cases.append(("redigested alignment tamper", redigested_alignment))

            semantic_restatement = copy.deepcopy(row)
            target = semantic_restatement["motion_spec"]["target_plan"][
                "dynamic_unit_targets"
            ][0]
            target["novel_target_motion"] = "walk steadily forward"
            target["ordered_stages"] = ["walk steadily forward"]
            target["target_clause"] = (
                "have the walking person on the left walk steadily forward"
            )
            semantic_restatement["motion_spec_sha256"] = _object_sha(
                semantic_restatement["motion_spec"]
            )
            cases.append(("semantic novelty restatement", semantic_restatement))

            for label, candidate in cases:
                with self.subTest(label=label), self.assertRaisesRegex(
                    postcheck.GokuFullMotionPostcheckError,
                    "exact Qwen-v6 finalizer closure",
                ):
                    postcheck._normalize_contract(candidate, manifest_root=root)

    def test_postcheck_independently_rebuilds_a0_g_a1_a2_and_hard_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.mp4"
            anchor_path = root / "anchor.png"
            source_path.write_bytes(b"source")
            anchor_path.write_bytes(b"anchor")
            row = _generation_row(source_path, anchor_path)

            def bypass(value: dict) -> dict:
                return dict(value)

            with mock.patch.object(
                finalizer, "validate_generation_row", side_effect=bypass
            ):
                normalized = postcheck._normalize_contract(
                    row, manifest_root=root
                )
                self.assertEqual(
                    normalized["coverage_authority_alignment"],
                    qwen.build_coverage_authority_alignment(
                        coverage_authority=row["motion_spec"][
                            "coverage_authority"
                        ],
                        change_region_proposals=row["motion_spec"][
                            "change_region_proposals"
                        ],
                        i0_grounding=row["motion_spec"]["i0_grounding"],
                        primary=row["motion_spec"]["source_census"],
                        secondary=row["motion_spec"][
                            "secondary_source_census"
                        ],
                        source_inventory_alignment=row["motion_spec"][
                            "source_inventory_alignment"
                        ],
                    ),
                )

                wrong_digest = copy.deepcopy(row)
                wrong_digest["qwen_evidence"][
                    "coverage_authority_digest"
                ] = "0" * 64
                with self.assertRaisesRegex(
                    postcheck.GokuFullMotionPostcheckError,
                    "evidence digest binding",
                ):
                    postcheck._normalize_contract(
                        wrong_digest, manifest_root=root
                    )

                raw_text_tamper = copy.deepcopy(row)
                raw_record = raw_text_tamper["qwen_evidence"][
                    "qwen_record_payload"
                ]
                raw_record["coverage_authority_inventory_raw"] += "\n"
                with self.assertRaisesRegex(
                    postcheck.GokuFullMotionPostcheckError,
                    "result/provenance replay",
                ):
                    postcheck._normalize_contract(
                        raw_text_tamper, manifest_root=root
                    )

                wrong_gate = copy.deepcopy(row)
                wrong_gate["qwen_evidence"]["hard_gate"][
                    "coverage_authority_sha256"
                ] = "0" * 64
                with self.assertRaisesRegex(
                    postcheck.GokuFullMotionPostcheckError,
                    "rebuilt hard-gate",
                ):
                    postcheck._normalize_contract(wrong_gate, manifest_root=root)

                wrong_proposal = copy.deepcopy(row)
                proposal = wrong_proposal["motion_spec"][
                    "change_region_proposals"
                ]
                proposal["regions"][0]["changed_pixel_count"] += 1
                proposal_digest = _object_sha(proposal)
                wrong_proposal["motion_spec_sha256"] = _object_sha(
                    wrong_proposal["motion_spec"]
                )
                wrong_proposal["qwen_evidence"][
                    "change_region_proposals_digest"
                ] = proposal_digest
                wrong_proposal["qwen_evidence"]["hard_gate"][
                    "change_region_proposals_sha256"
                ] = proposal_digest
                with self.assertRaisesRegex(
                    postcheck.GokuFullMotionPostcheckError,
                    "A0 change-region proposals",
                ):
                    postcheck._normalize_contract(
                        wrong_proposal, manifest_root=root
                    )

                wrong_alignment = copy.deepcopy(row)
                alignment = wrong_alignment["motion_spec"][
                    "coverage_authority_alignment"
                ]
                alignment["all_authority_entities_aligned"] = False
                alignment_digest = _object_sha(alignment)
                wrong_alignment["motion_spec_sha256"] = _object_sha(
                    wrong_alignment["motion_spec"]
                )
                wrong_alignment["qwen_evidence"][
                    "coverage_authority_alignment_digest"
                ] = alignment_digest
                wrong_alignment["qwen_evidence"]["hard_gate"][
                    "coverage_authority_alignment_sha256"
                ] = alignment_digest
                with self.assertRaisesRegex(
                    postcheck.GokuFullMotionPostcheckError,
                    "A0a/A0b/G/A1/A2 alignment",
                ):
                    postcheck._normalize_contract(
                        wrong_alignment, manifest_root=root
                    )

                semantic_restatement = copy.deepcopy(row)
                target = semantic_restatement["motion_spec"]["target_plan"][
                    "dynamic_unit_targets"
                ][0]
                target["novel_target_motion"] = "walk steadily forward"
                target["ordered_stages"] = ["walk steadily forward"]
                target["target_clause"] = (
                    "have the walking person on the left walk steadily forward"
                )
                semantic_restatement["motion_spec_sha256"] = _object_sha(
                    semantic_restatement["motion_spec"]
                )
                with self.assertRaisesRegex(
                    postcheck.GokuFullMotionPostcheckError,
                    "semantic novelty",
                ):
                    postcheck._normalize_contract(
                        semantic_restatement, manifest_root=root
                    )

    def test_resume_record_recomputes_qwen_evidence_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.mp4"
            anchor_path = root / "anchor.png"
            source_path.write_bytes(b"source")
            anchor_path.write_bytes(b"anchor")
            row = _generation_row(source_path, anchor_path)
            normalized = postcheck._normalize_contract(row, manifest_root=root)
            config = {"config_digest": "a" * 64}
            record = {
                "schema_version": postcheck.POSTCHECK_SCHEMA,
                "iid": row["iid"],
                "status": "ok",
                "input_digest": _object_sha(row),
                **config,
                "qwen_evidence_binding": copy.deepcopy(
                    normalized["qwen_evidence_binding"]
                ),
                "change_region_proposals_digest": normalized[
                    "change_region_proposals_digest"
                ],
                "coverage_authority_inventory_digest": normalized[
                    "coverage_authority_inventory_digest"
                ],
                "coverage_authority_assignments_digest": normalized[
                    "coverage_authority_assignments_digest"
                ],
                "coverage_authority_digest": normalized[
                    "coverage_authority_digest"
                ],
                "coverage_authority_alignment_digest": normalized[
                    "coverage_authority_alignment_digest"
                ],
                "qwen_record_payload_sha256": normalized[
                    "qwen_record_payload_sha256"
                ],
            }
            record["result_digest"] = _object_sha(record)
            postcheck._validate_output_record(
                record,
                expected_row=row,
                config_binding=config,
            )

            old_postcheck = copy.deepcopy(record)
            old_postcheck["schema_version"] = (
                "motive-goku-full-motion-postcheck-v4"
            )
            del old_postcheck["result_digest"]
            old_postcheck["result_digest"] = _object_sha(old_postcheck)
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError,
                "schema differs",
            ):
                postcheck._validate_output_record(
                    old_postcheck,
                    expected_row=row,
                    config_binding=config,
                )

            tampered = copy.deepcopy(record)
            tampered["qwen_evidence_binding"]["receipt_digest"] = "b" * 64
            del tampered["result_digest"]
            tampered["result_digest"] = _object_sha(tampered)
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError,
                "Qwen-v6 evidence binding differs",
            ):
                postcheck._validate_output_record(
                    tampered,
                    expected_row=row,
                    config_binding=config,
                )

            tampered_receipt = copy.deepcopy(record)
            tampered_receipt["qwen_evidence_binding"][
                "source_census_canonicalization"
            ]["raw_sha256"] = "0" * 64
            del tampered_receipt["result_digest"]
            tampered_receipt["result_digest"] = _object_sha(
                tampered_receipt
            )
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError,
                "Qwen-v6 evidence binding differs",
            ):
                postcheck._validate_output_record(
                    tampered_receipt,
                    expected_row=row,
                    config_binding=config,
                )

            tampered_payload_sha = copy.deepcopy(record)
            tampered_payload_sha["qwen_record_payload_sha256"] = "c" * 64
            del tampered_payload_sha["result_digest"]
            tampered_payload_sha["result_digest"] = _object_sha(
                tampered_payload_sha
            )
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError,
                "qwen_record_payload_sha256 binding differs",
            ):
                postcheck._validate_output_record(
                    tampered_payload_sha,
                    expected_row=row,
                    config_binding=config,
                )

    def test_tampered_compiled_instruction_or_motion_spec_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.mp4"
            anchor_path = root / "anchor.png"
            source_path.write_bytes(b"source")
            anchor_path.write_bytes(b"anchor")
            row = _generation_row(source_path, anchor_path)

            bad_sha = copy.deepcopy(row)
            bad_sha["motion_spec"]["qwen_result_digest"] = "f" * 64
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError, "motion_spec SHA"
            ):
                postcheck._normalize_contract(bad_sha, manifest_root=root)

            bad_compiled = copy.deepcopy(row)
            bad_compiled["motion_spec"]["compiled_instruction"]["clauses"][0][
                "text_sha256"
            ] = "0" * 64
            bad_compiled["motion_spec_sha256"] = _object_sha(
                bad_compiled["motion_spec"]
            )
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError,
                "exact Qwen-v6 finalizer closure",
            ):
                postcheck._normalize_contract(
                    bad_compiled, manifest_root=root
                )


class FullMotionVisualSchemaTests(unittest.TestCase):
    def test_census_and_shared_base_clause_judgment_are_closed(self) -> None:
        census = postcheck.validate_target_census(_target_census())
        source = _source()
        plan = _plan(source)
        judgment = postcheck.validate_clause_judgment(
            _judgment(),
            expected_dynamic_units=plan["dynamic_unit_targets"],
            expected_static_ids=["static_person_01"],
        )
        aggregate = postcheck.aggregate_postcheck(
            census,
            judgment,
            expected_contract=_expected_visual_contract(source, plan),
        )
        self.assertEqual(aggregate["decision"], "pass")
        self.assertTrue(aggregate["all_source_futures_suppressed_or_explicit"])

    def test_unclear_missing_camera_static_motion_and_extra_action_reject(self) -> None:
        source = _source()
        plan = _plan(source)
        cases: list[tuple[str, dict, dict]] = []
        unclear_census = _target_census()
        unclear_census["uncertainty_codes"] = ["right_hand_occluded"]
        cases.append(("census uncertainty", unclear_census, _judgment()))
        camera = _judgment()
        camera["camera_result"]["fulfilled"] = "no"
        cases.append(("camera", _target_census(), camera))
        static = _judgment()
        static["static_entity_results"][0]["remain_still"] = "unclear"
        cases.append(("static", _target_census(), static))
        extra = _judgment()
        extra["no_extra_actions"] = {
            "status": "fail",
            "observed_extra_actions": ["the dog stands up"],
            "frame_evidence": _frame_evidence(
                "the dog begins seated", "the dog stands"
            ),
        }
        cases.append(("extra", _target_census(), extra))
        for name, census_raw, judgment_raw in cases:
            with self.subTest(name=name):
                census = postcheck.validate_target_census(census_raw)
                judgment = postcheck.validate_clause_judgment(
                    judgment_raw,
                    expected_dynamic_units=plan["dynamic_unit_targets"],
                    expected_static_ids=["static_person_01"],
                )
                self.assertEqual(
                    postcheck.aggregate_postcheck(
                        census,
                        judgment,
                        expected_contract=_expected_visual_contract(source, plan),
                    )["decision"],
                    "reject",
                )

    def test_shared_base_must_be_explicitly_fulfilled(self) -> None:
        source = _source()
        plan = _plan(source)
        judgment_raw = _judgment()
        judgment_raw["motion_unit_results"][0][
            "explicit_shared_base_fulfilled"
        ] = "no"
        judgment_raw["decision"] = "fail"
        judgment = postcheck.validate_clause_judgment(
            judgment_raw,
            expected_dynamic_units=plan["dynamic_unit_targets"],
            expected_static_ids=["static_person_01"],
        )
        aggregate = postcheck.aggregate_postcheck(
            _target_census(),
            judgment,
            expected_contract=_expected_visual_contract(source, plan),
        )
        self.assertIn(
            "explicit_shared_base_not_fulfilled:unit_01",
            aggregate["failure_codes"],
        )

    def test_expected_and_blind_census_inventory_closes_exactly(self) -> None:
        source = _source()
        plan = _plan(source)
        expected = _expected_visual_contract(source, plan)
        cases: list[tuple[str, dict, dict, str, bool]] = []

        missing_dynamic_census = _target_census()
        missing_dynamic_census["motion_units"] = []
        missing_dynamic_judge = _judgment()
        missing_dynamic_judge["motion_unit_results"][0].update(
            {"census_match": "missing", "census_observed_id": None}
        )
        cases.append(
            (
                "missing dynamic",
                missing_dynamic_census,
                missing_dynamic_judge,
                "dynamic_unit_missing_from_census:unit_01",
                False,
            )
        )

        extra_dynamic_census = _target_census()
        extra_dynamic_census["motion_units"].append(
            {
                "observed_unit_id": "obs_02",
                "stable_reference": "an unrequested cyclist in the background",
                "entity_type": "person",
                "observed_motion": "rides from right to left",
                "frame_evidence": _frame_evidence(
                    "the cyclist enters at the right edge",
                    "the cyclist reaches the left edge",
                ),
            }
        )
        cases.append(
            (
                "extra dynamic",
                extra_dynamic_census,
                _judgment(),
                "extra_motion_unit:obs_02",
                False,
            )
        )

        dynamic_became_static_census = _target_census()
        dynamic_became_static_census["motion_units"] = []
        dynamic_became_static_census["static_salient_people"].append(
            {
                "observed_static_id": "static_obs_02",
                "stable_reference": "the person on the left",
                "entity_type": "person",
                "frame_evidence": _frame_evidence(
                    "the person holds the initial pose",
                    "the person still holds the initial pose",
                ),
            }
        )
        dynamic_became_static_judge = _judgment()
        dynamic_became_static_judge["motion_unit_results"][0].update(
            {
                "census_match": "static",
                "census_observed_id": "static_obs_02",
            }
        )
        cases.append(
            (
                "dynamic became static",
                dynamic_became_static_census,
                dynamic_became_static_judge,
                "dynamic_unit_observed_static:unit_01",
                True,
            )
        )

        static_moved_census = _target_census()
        static_moved_census["static_salient_people"] = []
        static_moved_census["motion_units"].append(
            {
                "observed_unit_id": "obs_02",
                "stable_reference": "the dog on the right",
                "entity_type": "animal",
                "observed_motion": "stands and walks toward the person",
                "frame_evidence": _frame_evidence(
                    "the dog begins seated",
                    "the dog walks toward the person",
                ),
            }
        )
        static_moved_judge = _judgment()
        static_moved_judge["static_entity_results"][0].update(
            {"census_match": "moving", "census_observed_id": "obs_02"}
        )
        cases.append(
            (
                "static moved despite positive atomic claim",
                static_moved_census,
                static_moved_judge,
                "static_entity_moved:static_person_01",
                True,
            )
        )

        static_missing_census = _target_census()
        static_missing_census["static_salient_people"] = []
        static_missing_judge = _judgment()
        static_missing_judge["static_entity_results"][0].update(
            {"census_match": "missing", "census_observed_id": None}
        )
        cases.append(
            (
                "static disappeared despite positive atomic claim",
                static_missing_census,
                static_missing_judge,
                "static_entity_missing:static_person_01",
                False,
            )
        )

        extra_static_census = _target_census()
        extra_static_census["static_salient_people"].append(
            {
                "observed_static_id": "static_obs_02",
                "stable_reference": "an unexplained second dog",
                "entity_type": "animal",
                "frame_evidence": _frame_evidence(
                    "the second dog is visible by the tree",
                    "the second dog remains by the tree",
                ),
            }
        )
        cases.append(
            (
                "extra static",
                extra_static_census,
                _judgment(),
                "extra_static_entity:static_obs_02",
                False,
            )
        )

        wrong_dynamic_type_census = _target_census()
        wrong_dynamic_type_census["motion_units"][0]["entity_type"] = "animal"
        cases.append(
            (
                "judge maps expected person to an animal census unit",
                wrong_dynamic_type_census,
                _judgment(),
                "dynamic_unit_census_entity_type_mismatch:unit_01",
                True,
            )
        )

        for (
            name,
            census_raw,
            judgment_raw,
            expected_failure,
            inventory_closed,
        ) in cases:
            with self.subTest(name=name):
                census = postcheck.validate_target_census(census_raw)
                judgment = postcheck.validate_clause_judgment(
                    judgment_raw,
                    expected_dynamic_units=plan["dynamic_unit_targets"],
                    expected_static_ids=["static_person_01"],
                )
                aggregate = postcheck.aggregate_postcheck(
                    census,
                    judgment,
                    expected_contract=expected,
                )
                self.assertEqual(aggregate["decision"], "reject")
                self.assertIn(expected_failure, aggregate["failure_codes"])
                self.assertFalse(aggregate["all_expected_units_aligned"])
                self.assertIs(
                    aggregate["census_inventory_closed"], inventory_closed
                )

    def test_camera_census_and_judge_must_agree(self) -> None:
        source = _source()
        plan = _plan(source)
        judge = _judgment()
        judge["camera_result"]["observed_motion_class"] = "dynamic"
        aggregate = postcheck.aggregate_postcheck(
            postcheck.validate_target_census(_target_census()),
            postcheck.validate_clause_judgment(
                judge,
                expected_dynamic_units=plan["dynamic_unit_targets"],
                expected_static_ids=["static_person_01"],
            ),
            expected_contract=_expected_visual_contract(source, plan),
        )
        self.assertEqual(aggregate["decision"], "reject")
        self.assertIn(
            "camera_census_judge_conflict", aggregate["failure_codes"]
        )
        self.assertFalse(aggregate["camera_census_judge_consistent"])

    def test_dynamic_source_camera_requires_suppression_and_visible_change(self) -> None:
        source = _source()
        source["camera"].update(
            {
                "motion_class": "pan_left",
                "motion_signature": "slow_pan_left",
                "motion_description": "a slow pan left",
                "dynamic": True,
            }
        )
        plan = _plan(source)
        plan["camera_target"].update(
            {
                "motion_relation": "replace_motion",
                "target_motion_class": "dolly_in",
                "target_motion_signature": "steady_dolly_in",
                "target_motion_description": "a steady dolly toward the actors",
                "target_clause": "move steadily closer to the actors",
                "source_motion_suppressed": True,
                "substantive_change": True,
            }
        )
        census = _target_census()
        census["camera"].update(
            {
                "motion_class": "dynamic",
                "motion_description": "the camera dollies steadily forward",
            }
        )
        passing_judge = _judgment()
        passing_judge["camera_result"].update(
            {
                "observed_motion_class": "dynamic",
                "observed_motion_signature": "steady dolly in",
                "source_camera_motion_suppressed": "yes",
                "substantive_target_camera_change_visible": "yes",
            }
        )
        expected = _expected_visual_contract(source, plan)
        baseline = postcheck.aggregate_postcheck(
            postcheck.validate_target_census(census),
            postcheck.validate_clause_judgment(
                passing_judge,
                expected_dynamic_units=plan["dynamic_unit_targets"],
                expected_static_ids=["static_person_01"],
            ),
            expected_contract=expected,
        )
        self.assertEqual(baseline["decision"], "pass")

        cases = (
            (
                "source_camera_motion_suppressed",
                "no",
                "source_camera_motion_not_suppressed",
            ),
            (
                "substantive_target_camera_change_visible",
                "no",
                "substantive_target_camera_change_not_visible",
            ),
        )
        for field, value, failure in cases:
            with self.subTest(field=field):
                judge = copy.deepcopy(passing_judge)
                judge["camera_result"][field] = value
                aggregate = postcheck.aggregate_postcheck(
                    census,
                    postcheck.validate_clause_judgment(
                        judge,
                        expected_dynamic_units=plan["dynamic_unit_targets"],
                        expected_static_ids=["static_person_01"],
                    ),
                    expected_contract=expected,
                )
                self.assertEqual(aggregate["decision"], "reject")
                self.assertIn(failure, aggregate["failure_codes"])

    def test_each_clause_requires_two_distinct_frame_positions(self) -> None:
        bad = _judgment()
        bad["camera_result"]["frame_evidence"][1]["frame_index"] = 0
        source = _source()
        with self.assertRaisesRegex(
            postcheck.GokuFullMotionPostcheckError, "distinct frame"
        ):
            postcheck.validate_clause_judgment(
                bad,
                expected_dynamic_units=_plan(source)["dynamic_unit_targets"],
                expected_static_ids=["static_person_01"],
            )


class FullMotionMediaBindingTests(unittest.TestCase):
    def test_generated_manifest_completion_closure_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _write_generation_closure(Path(directory))
            rows, digest, complete, _ = postcheck._validate_generated_manifest(
                Path(directory),
                generated_manifest_path=fixture["generated_path"],
                generation_rows=[fixture["row"]],
                input_manifest_sha256=fixture["manifest_sha"],
                run_contract=fixture["run_contract"],
            )
            self.assertEqual([row["iid"] for row in rows], [fixture["row"]["iid"]])
            self.assertEqual(digest, _sha_bytes(fixture["generated_path"].read_bytes()))
            self.assertEqual(complete["completed_sample_count"], 1)

            tampered = json.loads(fixture["generated_path"].read_text())
            tampered["target_preview_mp4_sha256"] = "f" * 64
            fixture["generated_path"].write_text(json.dumps(tampered) + "\n")
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError,
                "generated-manifest SHA",
            ):
                postcheck._validate_generated_manifest(
                    Path(directory),
                    generated_manifest_path=fixture["generated_path"],
                    generation_rows=[fixture["row"]],
                    input_manifest_sha256=fixture["manifest_sha"],
                    run_contract=fixture["run_contract"],
                )

    def test_media_validator_enforces_81_frames_25_fps_and_i0_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _write_generation_closure(root)
            row = fixture["row"]
            normalized = postcheck._normalize_contract(row, manifest_root=root)
            conditioning_rgb = bytes([10, 20, 30, 40, 50, 60])
            target_rgb = bytes([11, 21, 31, 39, 49, 59])
            pixel_sha = _sha_bytes(conditioning_rgb)
            temporal = _temporal_policy()
            outputs = {
                "source_video": fixture["source_copy"].name,
                "source_video_sha256": row["source_video_sha256"],
                "source_video_bytes": fixture["source_copy"].stat().st_size,
                "edit_instruction_file": fixture["instruction_file"].name,
                "edit_instruction_file_sha256": row[
                    "edit_instruction_sha256"
                ],
                "edit_instruction_file_bytes": fixture[
                    "instruction_file"
                ].stat().st_size,
                "preview_mp4": fixture["target_path"].name,
                "preview_mp4_sha256": _sha_bytes(
                    fixture["target_path"].read_bytes()
                ),
                "preview_mp4_bytes": fixture["target_path"].stat().st_size,
                "conditioning_anchor_original": fixture[
                    "original_anchor"
                ].name,
                "conditioning_anchor_original_sha256": row["anchor_sha256"],
                "conditioning_frame0_png": fixture["frame0_png"].name,
                "conditioning_frame0_png_sha256": _sha_bytes(
                    fixture["frame0_png"].read_bytes()
                ),
            }
            result_payload = {
                "schema_version": "motive-wan22-i2v-sample-v1",
                "iid": row["iid"],
                "group_id": row["group_id"],
                "manifest_sha256": fixture["manifest_sha"],
                "manifest_row_digest": _object_sha(row),
                "contract_digest": fixture["run_contract"]["contract_digest"],
                "prompt": {
                    "field": "edit_instruction",
                    "text": row["edit_instruction"],
                    "sha256": row["edit_instruction_sha256"],
                },
                "inputs": {
                    "source_video_sha256": row["source_video_sha256"],
                    "source_video_resolved_path": str(fixture["source_path"]),
                    "source_video_committed_path": str(fixture["source_copy"]),
                    "anchor_sha256": row["anchor_sha256"],
                },
                "outputs": outputs,
                "first_frame_policy": {
                    "tensor_frame0_overridden_before_encoding": True,
                    "preencode_frame0_matches_png_pixels": True,
                    "mp4_codec_is_lossy": True,
                    "mp4_decode_pixel_equality_claimed": False,
                    "conditioning_tensor_shape": [3, 1, 2],
                    "preencode_frame0_pixel_sha256": pixel_sha,
                    "lossless_png_pixel_sha256": pixel_sha,
                },
                "temporal_policy": temporal,
            }
            result = dict(result_payload)
            result["result_digest"] = _object_sha(result_payload)
            fixture["result_path"].write_text(
                json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
            )
            generated = dict(fixture["generated"])
            generated.update(
                {
                    "result_digest": result["result_digest"],
                    "target_preview_mp4_sha256": outputs["preview_mp4_sha256"],
                    "conditioning_frame0_png_sha256": outputs[
                        "conditioning_frame0_png_sha256"
                    ],
                    "temporal_policy": temporal,
                }
            )

            def fake_probe(path: Path, **_: object) -> dict:
                return {
                    "codec": "h264",
                    "width": 2,
                    "height": 1,
                    "pixel_format": "yuv420p",
                    "frame_count": 81,
                    "avg_frame_rate": "25/1",
                    "r_frame_rate": "25/1",
                    "duration_seconds": 3.24,
                    "bytes": path.stat().st_size,
                }

            def fake_decode(path: Path, **_: object) -> bytes:
                return target_rgb if path.name == "preview.mp4" else conditioning_rgb

            binding = postcheck.validate_generated_sample(
                row,
                generated_row=generated,
                contract=normalized,
                manifest_path=fixture["manifest_path"],
                manifest_sha256=fixture["manifest_sha"],
                run_contract=fixture["run_contract"],
                run_contract_sha256="a" * 64,
                generation_root=root,
                probe_fn=fake_probe,
                decode_fn=fake_decode,
            )
            self.assertTrue(binding["verified"])
            self.assertLessEqual(
                binding["frame0_similarity"]["mean_absolute_error"], 1.0
            )

            fixture["instruction_file"].write_bytes(
                row["edit_instruction"].encode("utf-8") + b"\n"
            )
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError,
                "edit instruction closure",
            ):
                postcheck.validate_generated_sample(
                    row,
                    generated_row=generated,
                    contract=normalized,
                    manifest_path=fixture["manifest_path"],
                    manifest_sha256=fixture["manifest_sha"],
                    run_contract=fixture["run_contract"],
                    run_contract_sha256="a" * 64,
                    generation_root=root,
                    probe_fn=fake_probe,
                    decode_fn=fake_decode,
                )
            fixture["instruction_file"].write_bytes(
                row["edit_instruction"].encode("utf-8")
            )

            original_generated_source = generated["source_video"]
            generated["source_video"] = str(fixture["source_path"])
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError,
                "source video closure",
            ):
                postcheck.validate_generated_sample(
                    row,
                    generated_row=generated,
                    contract=normalized,
                    manifest_path=fixture["manifest_path"],
                    manifest_sha256=fixture["manifest_sha"],
                    run_contract=fixture["run_contract"],
                    run_contract_sha256="a" * 64,
                    generation_root=root,
                    probe_fn=fake_probe,
                    decode_fn=fake_decode,
                )
            generated["source_video"] = original_generated_source

            def bad_probe(path: Path, **_: object) -> dict:
                value = fake_probe(path)
                if path.name == "preview.mp4":
                    value["frame_count"] = 82
                return value

            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError, "exactly 81"
            ):
                postcheck.validate_generated_sample(
                    row,
                    generated_row=generated,
                    contract=normalized,
                    manifest_path=fixture["manifest_path"],
                    manifest_sha256=fixture["manifest_sha"],
                    run_contract=fixture["run_contract"],
                    run_contract_sha256="a" * 64,
                    generation_root=root,
                    probe_fn=bad_probe,
                    decode_fn=fake_decode,
                )


class FullMotionPostcheckResumeTests(unittest.TestCase):
    def test_mock_backend_atomic_receipt_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _write_generation_closure(Path(directory))
            output = Path(directory) / "postcheck.jsonl"
            backend = _MockBackend()
            self.assertEqual(
                postcheck.run_postcheck(
                    _args(fixture, output),
                    backend=backend,
                    media_validator=_fake_media_validator,
                ),
                0,
            )
            receipt_path = postcheck.shard_receipt_path(output)
            self.assertTrue(receipt_path.is_file())
            record = json.loads(output.read_text())
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(
                receipt["schema_version"], postcheck.SHARD_RECEIPT_SCHEMA
            )
            self.assertEqual(record["decision"], "pass")
            self.assertEqual(
                record["coverage_authority_alignment_digest"],
                record["qwen_evidence_binding"][
                    "coverage_authority_alignment_digest"
                ],
            )
            self.assertEqual((backend.census_calls, backend.judge_calls), (1, 1))

            old_receipt = copy.deepcopy(receipt)
            old_receipt["schema_version"] = (
                "motive-goku-full-motion-postcheck-shard-receipt-v4"
            )
            old_receipt["receipt_digest"] = postcheck._receipt_digest(
                old_receipt
            )
            receipt_path.write_text(
                json.dumps(old_receipt, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError,
                "receipt schema differs",
            ):
                postcheck.run_postcheck(
                    _args(fixture, output, resume=True),
                    media_validator=_fake_media_validator,
                )
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                postcheck.run_postcheck(
                    _args(fixture, output, resume=True),
                    media_validator=_fake_media_validator,
                ),
                0,
            )
            self.assertEqual((backend.census_calls, backend.judge_calls), (1, 1))

            record["eligible"] = False
            output.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                postcheck.GokuFullMotionPostcheckError, "receipt"
            ):
                postcheck.run_postcheck(
                    _args(fixture, output, resume=True),
                    backend=backend,
                    media_validator=_fake_media_validator,
                )

    def test_error_has_no_receipt_and_resume_retries_only_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _write_generation_closure(Path(directory))
            output = Path(directory) / "retry.jsonl"
            bad_backend = _MockBackend(malformed_census=True)
            self.assertEqual(
                postcheck.run_postcheck(
                    _args(fixture, output),
                    backend=bad_backend,
                    media_validator=_fake_media_validator,
                ),
                1,
            )
            self.assertFalse(postcheck.shard_receipt_path(output).exists())
            self.assertEqual(json.loads(output.read_text())["status"], "error")

            good_backend = _MockBackend()
            self.assertEqual(
                postcheck.run_postcheck(
                    _args(fixture, output, resume=True),
                    backend=good_backend,
                    media_validator=_fake_media_validator,
                ),
                0,
            )
            self.assertEqual(json.loads(output.read_text())["status"], "ok")
            self.assertTrue(postcheck.shard_receipt_path(output).is_file())


if __name__ == "__main__":
    unittest.main()
