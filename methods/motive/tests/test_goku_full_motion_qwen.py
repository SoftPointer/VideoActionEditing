from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from PIL import Image

from motive import goku_full_motion_contract as contract
from motive.goku_full_motion_instruction import compile_full_motion_instruction
from motive.goku_full_motion_qwen import (
    AUTHORITY_FRAME_INDICES,
    AUTHORITY_GRID_COLUMNS,
    AUTHORITY_GRID_ROWS,
    CHANGE_CELL_DELTA_PERCENTILE_MILLI,
    CHANGE_CELL_MIN_CHANGED_FRACTION_PPM,
    CHANGE_CELL_MIN_DELTA_AT_PERCENTILE_MILLI,
    CHANGE_REGION_ASSIGNMENT_SCHEMA,
    CHANGE_REGION_DELTA_THRESHOLD,
    CHANGE_REGION_PROPOSALS_SCHEMA,
    CHANGE_REGION_SCHEMA,
    COVERAGE_AUTHORITY_CAMERA_SCHEMA,
    COVERAGE_AUTHORITY_ASSIGNMENTS_PROMPT_SCHEMA,
    COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA,
    COVERAGE_AUTHORITY_ASSIGNMENTS_SYSTEM,
    COVERAGE_AUTHORITY_EXTRA_SCHEMA,
    COVERAGE_AUTHORITY_INVENTORY_PROMPT_SCHEMA,
    COVERAGE_AUTHORITY_INVENTORY_SCHEMA,
    COVERAGE_AUTHORITY_INVENTORY_SYSTEM,
    COVERAGE_AUTHORITY_SCHEMA,
    COVERAGE_AUTHORITY_SUBJECT_SCHEMA,
    COVERAGE_CRITIC_PROMPT_SCHEMA,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MOSAIC_COLUMNS,
    DEFAULT_NFRAMES,
    DEFAULT_TILE_WIDTH,
    GokuFullMotionQwenError,
    I0_GROUNDED_SUBJECT_SCHEMA,
    I0_GROUNDING_SCHEMA,
    HELD_CARRIED_OBJECT_CLOSURE_RULE,
    PASS_A_PROMPT,
    PASS_A_SYSTEM,
    PASS_A2_PROMPT,
    PASS_A2_SYSTEM,
    PASS_B_SYSTEM,
    PASS_C_SYSTEM,
    QWEN3_LOGICAL_SHARDS,
    SOURCE_CENSUS_PROMPT_SCHEMA,
    TARGET_PLAN_PROMPT_SCHEMA,
    TARGET_PLAN_SCHEMA_REPAIR_SYSTEM,
    _build_authority_grid_and_proposals,
    _build_grounded_temporal_zoom,
    _build_visuals,
    _coverage_authority_visual_digest,
    _generate_coverage_authority_pass,
    _generate_i0_grounding_pass,
    _generate_visual_pass,
    _parse_direct_object,
    _validate_schema_repair_ledger,
    _visual_digest,
    assigned_iids_for_shard,
    build_coverage_authority,
    build_coverage_authority_allowed_owner_map,
    build_coverage_authority_alignment,
    build_coverage_authority_assignments_prompt,
    build_coverage_authority_inventory_prompt,
    build_hard_gate,
    build_parser,
    build_target_plan_prompt,
    build_target_plan_schema_repair_prompt,
    canonicalize_coverage_authority_assignments_model_output,
    canonicalize_coverage_authority_inventory_model_output,
    qwen_provenance_digest,
    qwen_result_payload,
    run_audit,
    shard_receipt_path,
    target_plan_validated_raw,
    validate_change_region_proposals,
    validate_coverage_authority,
    validate_coverage_authority_allowed_owner_map,
    validate_coverage_authority_assignments,
    validate_coverage_authority_alignment,
    validate_coverage_authority_inventory,
    validate_i0_grounding,
    validate_source_census_i0_binding,
)
from motive.qwen_filter import _file_digest, _video_mosaic


def _evidence(description: str, start: int = 0, end: int = 80) -> dict:
    return {
        "schema_version": contract.MOTION_EVIDENCE_SCHEMA,
        "start_frame": start,
        "end_frame": end,
        "description": description,
    }


def _source_census(iid: str) -> dict:
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
        "scene_description": (
            "Two standing men are framed together in a yellow-lit studio"
        ),
        "i0_visible_entities": [
            "the blue-shirted man on viewer-left",
            "the tattooed man in a black sleeveless top on viewer-right",
        ],
        "i0_entity_registry": [
            {
                "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                "entity_id": "entity_01",
                "entity_type": "person",
                "stable_reference": "the blue-shirted man on viewer-left",
                "i0_bbox_xyxy_1000": [50, 350, 300, 900],
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
                "entity_type": "person",
                "stable_reference": (
                    "the tattooed man in a black sleeveless top on viewer-right"
                ),
                "i0_bbox_xyxy_1000": [700, 350, 950, 900],
                "viewer_region": "center_right",
                "region_ordinal": 1,
                "role": "dynamic_subject",
                "visible_at_i0": True,
                "reachable_at_i0": True,
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
                "stable_reference": "the blue-shirted man on viewer-left",
                "visible_at_i0": True,
                "independent_motion": True,
                "i0_state": (
                    "Standing at I0 with the hand on viewer-left near the "
                    "waist and the hand on viewer-right lowered"
                ),
                "source_action_signature": "raise_hand_into_peace_sign",
                "source_motion": (
                    "raises one hand from waist height and forms a peace sign"
                ),
                "source_motion_components": [
                    {
                        "schema_version": contract.SOURCE_MOTION_COMPONENT_SCHEMA,
                        "component_id": "component_01",
                        "component_type": "gesture",
                        "motion_signature": "raise_hand_into_peace_sign",
                        "motion_description": (
                            "raises one hand from waist height and forms a peace sign"
                        ),
                        "dependent_entity_ids": [],
                        "motion_evidence": [
                            _evidence(
                                "the blue-shirted man's hand rises and forms two fingers",
                                0,
                                56,
                            )
                        ],
                    }
                ],
                "motion_evidence": [
                    _evidence(
                        "the blue-shirted man's hand rises and forms two fingers",
                        0,
                        56,
                    )
                ],
                "confidence": "high",
            },
            {
                "schema_version": contract.SOURCE_DYNAMIC_UNIT_SCHEMA,
                "unit_id": "unit_02",
                "entity_id": "entity_02",
                "entity_type": "person",
                "stable_reference": (
                    "the tattooed man in a black sleeveless top on viewer-right"
                ),
                "visible_at_i0": True,
                "independent_motion": True,
                "i0_state": (
                    "Standing at I0 with the black-gloved hand on viewer-left "
                    "near the waist and the other hand on viewer-right lowered"
                ),
                "source_action_signature": "raise_gloved_hand_gesture",
                "source_motion": (
                    "raises his gloved hand from his waist into a hand sign"
                ),
                "source_motion_components": [
                    {
                        "schema_version": contract.SOURCE_MOTION_COMPONENT_SCHEMA,
                        "component_id": "component_01",
                        "component_type": "gesture",
                        "motion_signature": "raise_gloved_hand_gesture",
                        "motion_description": (
                            "raises his gloved hand from his waist into a hand sign"
                        ),
                        "dependent_entity_ids": [],
                        "motion_evidence": [
                            _evidence(
                                "the right man's black-gloved hand rises to chest height",
                                8,
                                64,
                            )
                        ],
                    }
                ],
                "motion_evidence": [
                    _evidence(
                        "the right man's black-gloved hand rises to chest height",
                        8,
                        64,
                    )
                ],
                "confidence": "high",
            },
        ],
        "static_salient_people": [],
        "camera": {
            "schema_version": contract.SOURCE_CAMERA_SCHEMA,
            "camera_id": "camera",
            "motion_class": "locked_off",
            "motion_signature": "locked_off",
            "motion_description": "locked off",
            "dynamic": False,
            "motion_evidence": [
                _evidence("the studio background remains aligned", 0, 80)
            ],
            "confidence": "high",
        },
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _i0_grounding(iid: str) -> dict:
    return {
        "schema_version": I0_GROUNDING_SCHEMA,
        "iid": iid,
        "subjects": [
            {
                "schema_version": I0_GROUNDED_SUBJECT_SCHEMA,
                "subject_id": "entity_01",
                "entity_type": "person",
                "stable_reference": "the blue-shirted man on viewer-left",
                "i0_bbox_xyxy_1000": [50, 350, 300, 900],
                "i0_state": (
                    "Standing at I0 with the hand on viewer-left near the "
                    "waist and the hand on viewer-right lowered"
                ),
                "viewer_left_extremity_height": "waistband",
                "viewer_left_extremity_state": (
                    "bare hand on viewer-left rests at the waistband"
                ),
                "viewer_right_extremity_height": "below_waist",
                "viewer_right_extremity_state": (
                    "bare hand on viewer-right hangs below the waist"
                ),
                "confidence": "high",
            },
            {
                "schema_version": I0_GROUNDED_SUBJECT_SCHEMA,
                "subject_id": "entity_02",
                "entity_type": "person",
                "stable_reference": (
                    "the tattooed man in a black sleeveless top on viewer-right"
                ),
                "i0_bbox_xyxy_1000": [700, 350, 950, 900],
                "i0_state": (
                    "Standing at I0 with the black-gloved hand on viewer-left "
                    "near the waist and the other hand on viewer-right lowered"
                ),
                "viewer_left_extremity_height": "waistband",
                "viewer_left_extremity_state": (
                    "black-gloved hand on viewer-left rests at the waistband"
                ),
                "viewer_right_extremity_height": "below_waist",
                "viewer_right_extremity_state": (
                    "bare hand on viewer-right hangs below the waist"
                ),
                "confidence": "high",
            },
        ],
        "all_visible_people_and_animals_enumerated": True,
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _change_region_proposals(iid: str) -> dict:
    return {
        "schema_version": CHANGE_REGION_PROPOSALS_SCHEMA,
        "iid": iid,
        "frame_indices": list(AUTHORITY_FRAME_INDICES),
        "grid_rows": AUTHORITY_GRID_ROWS,
        "grid_columns": AUTHORITY_GRID_COLUMNS,
        "delta_threshold": CHANGE_REGION_DELTA_THRESHOLD,
        "minimum_changed_fraction_ppm": (
            CHANGE_CELL_MIN_CHANGED_FRACTION_PPM
        ),
        "delta_percentile_milli": CHANGE_CELL_DELTA_PERCENTILE_MILLI,
        "minimum_delta_at_percentile_milli": (
            CHANGE_CELL_MIN_DELTA_AT_PERCENTILE_MILLI
        ),
        "regions": [
            {
                "schema_version": CHANGE_REGION_SCHEMA,
                "proposal_id": "proposal_01",
                "cell_row": 3,
                "cell_column": 1,
                "bbox_xyxy_1000": [0, 500, 250, 750],
                "changed_pixel_count": 10,
                "bbox_area_pixels": 100,
                "changed_fraction_ppm": 100_000,
                "delta_at_percentile_milli": 30_000,
            },
            {
                "schema_version": CHANGE_REGION_SCHEMA,
                "proposal_id": "proposal_02",
                "cell_row": 3,
                "cell_column": 4,
                "bbox_xyxy_1000": [750, 500, 1000, 750],
                "changed_pixel_count": 10,
                "bbox_area_pixels": 100,
                "changed_fraction_ppm": 100_000,
                "delta_at_percentile_milli": 30_000,
            },
        ],
        "active_cell_count": 2,
        "global_changed_fraction_ppm": 100_000,
        "all_active_cells_emitted": True,
    }


def _coverage_inventory(iid: str) -> dict:
    return {
        "schema_version": COVERAGE_AUTHORITY_INVENTORY_SCHEMA,
        "iid": iid,
        "i0_subjects": [
            {
                "schema_version": COVERAGE_AUTHORITY_SUBJECT_SCHEMA,
                "authority_id": "authority_subject_01",
                "entity_type": "person",
                "stable_reference": "the blue-shirted man on viewer-left",
                "i0_bbox_xyxy_1000": [50, 350, 300, 900],
                "temporal_extent_bbox_xyxy_1000": [50, 150, 350, 900],
                "motion_role": "dynamic",
                "motion_component_types": ["gesture"],
                "motion_evidence": [
                    _evidence("the left person's hand rises", 0, 56)
                ],
                "confidence": "high",
            },
            {
                "schema_version": COVERAGE_AUTHORITY_SUBJECT_SCHEMA,
                "authority_id": "authority_subject_02",
                "entity_type": "person",
                "stable_reference": (
                    "the tattooed man in a black sleeveless top on viewer-right"
                ),
                "i0_bbox_xyxy_1000": [700, 350, 950, 900],
                "temporal_extent_bbox_xyxy_1000": [650, 150, 950, 900],
                "motion_role": "dynamic",
                "motion_component_types": ["gesture"],
                "motion_evidence": [
                    _evidence("the right person's gloved hand rises", 8, 64)
                ],
                "confidence": "high",
            },
        ],
        "extra_dynamic_entities": [],
        "camera": {
            "schema_version": COVERAGE_AUTHORITY_CAMERA_SCHEMA,
            "dynamic": False,
            "motion_class": "locked_off",
            "motion_evidence": [
                _evidence("the studio background remains aligned", 0, 80)
            ],
            "confidence": "high",
        },
        "all_i0_people_and_animals_enumerated": True,
        "all_dynamic_entities_enumerated": True,
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _coverage_assignments(iid: str, proposals: dict, inventory: dict) -> dict:
    assignments = []
    for region in proposals["regions"]:
        bbox = region["bbox_xyxy_1000"]
        subject_id = (
            "authority_subject_01"
            if bbox[0] + bbox[2] < 1000
            else "authority_subject_02"
        )
        assignments.append(
            {
                "schema_version": CHANGE_REGION_ASSIGNMENT_SCHEMA,
                "proposal_id": region["proposal_id"],
                "assignment_kind": "entity",
                "authority_entity_ids": [subject_id],
                "resolution_reason": (
                    "The region follows the corresponding person's visible "
                    "hand gesture across checkpoints"
                ),
                "reject_reason_code": None,
                "confidence": "high",
            }
        )
    return {
        "schema_version": COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA,
        "iid": iid,
        "coverage_authority_inventory_sha256": contract.object_sha256(
            inventory
        ),
        "change_region_proposals_sha256": contract.object_sha256(proposals),
        "allowed_owner_map_sha256": contract.object_sha256(
            build_coverage_authority_allowed_owner_map(
                coverage_authority_inventory=inventory,
                change_region_proposals=proposals,
            )
        ),
        "change_region_assignments": assignments,
        "all_change_regions_resolved": True,
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _coverage_authority(iid: str, proposals: dict) -> dict:
    inventory = _coverage_inventory(iid)
    assignments = _coverage_assignments(iid, proposals, inventory)
    return build_coverage_authority(
        coverage_authority_inventory=inventory,
        coverage_authority_assignments=assignments,
        change_region_proposals=proposals,
    )


def _target_unit(
    *, unit_id: str, reference: str, signature: str, hand: str
) -> dict:
    novel = f"raise {hand} and perform an open-palm wave"
    return {
        "schema_version": contract.TARGET_DYNAMIC_UNIT_SCHEMA,
        "unit_id": unit_id,
        "entity_id": "entity_01" if unit_id == "unit_01" else "entity_02",
        "stable_reference": reference,
        "target_action_signature": signature,
        "motion_relation": "replace",
        "source_motion_suppressed": True,
        "explicit_shared_base_motion": None,
        "source_component_dispositions": [
            {
                "schema_version": contract.TARGET_COMPONENT_DISPOSITION_SCHEMA,
                "component_id": "component_01",
                "disposition": "suppress",
                "explicit_target_motion": None,
            }
        ],
        "novel_target_motion": novel,
        "target_clause": f"have {reference} {novel}",
        "substantive_change": True,
        "starts_at_i0": True,
        "i0_executable": True,
        "complete_within_clip": True,
        "completion_time_seconds": 3.0,
        "ordered_stages": [
            f"{reference} raises {hand} from the exact initial position",
            f"{reference} opens the palm and waves",
        ],
        "interaction_entity_ids": [],
        "required_i0_entity_ids": [
            "entity_01" if unit_id == "unit_01" else "entity_02"
        ],
    }


def _target_plan(source: dict) -> dict:
    ids = [item["unit_id"] for item in source["dynamic_units"]]
    return {
        "schema_version": contract.TARGET_PLAN_SCHEMA,
        "iid": source["iid"],
        "source_census_sha256": contract.object_sha256(source),
        "dynamic_unit_targets": [
            _target_unit(
                unit_id="unit_01",
                reference="the blue-shirted man on viewer-left",
                signature="raise_bare_hand_open_palm_wave",
                hand="his bare hand",
            ),
            _target_unit(
                unit_id="unit_02",
                reference=(
                    "the tattooed man in a black sleeveless top on viewer-right"
                ),
                signature="raise_gloved_hand_open_palm_wave",
                hand="his gloved hand",
            ),
        ],
        "static_person_targets": [],
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
            "required_dynamic_unit_ids": ids,
            "planned_changed_unit_ids": ids,
            "missing_unit_ids": [],
            "extra_unit_ids": [],
            "required_static_person_ids": [],
            "constrained_static_person_ids": [],
            "camera_clause_present": True,
        },
        "i0_executable": True,
        "no_new_prerequisites": True,
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _critic(source: dict, plan: dict, compiled: dict) -> dict:
    ids = [item["unit_id"] for item in source["dynamic_units"]]
    return {
        "schema_version": contract.COVERAGE_CRITIC_SCHEMA,
        "iid": source["iid"],
        "source_census_sha256": contract.object_sha256(source),
        "target_plan_sha256": contract.object_sha256(plan),
        "instruction_sha256": compiled["instruction_sha256"],
        "required_dynamic_unit_ids": ids,
        "plan_covered_dynamic_unit_ids": ids,
        "instruction_covered_dynamic_unit_ids": ids,
        "missing_unit_ids": [],
        "extra_unit_ids": [],
        "ambiguous_unit_ids": [],
        "per_unit_substantive_change": {item: True for item in ids},
        "source_future_suppressed_or_explicit": {item: True for item in ids},
        "camera_clause_present": True,
        "camera_target_valid": True,
        "required_static_person_ids": [],
        "static_people_preserved": {},
        "i0_executable": True,
        "no_new_prerequisites": True,
        "no_unrequested_action": True,
        "verdict": "pass",
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _write_media(root: Path, iid: str) -> tuple[Path, Path]:
    source = root / f"{iid}.avi"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (64, 48),
    )
    if not writer.isOpened():  # pragma: no cover
        raise RuntimeError("test OpenCV cannot create MJPG")
    for index in range(81):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (5, 8), (25, 43), (190, 70, 20), -1)
        cv2.rectangle(frame, (39, 8), (59, 43), (25, 25, 25), -1)
        cv2.circle(frame, (18, 35 - min(index, 30) // 2), 3, (230, 230, 230), -1)
        cv2.circle(frame, (46, 36 - min(index, 24) // 2), 3, (10, 10, 10), -1)
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(source))
    ok, frame = capture.read()
    capture.release()
    if not ok:  # pragma: no cover
        raise RuntimeError("test OpenCV cannot decode frame zero")
    anchor = root / f"{iid}.png"
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(anchor)
    return source, anchor


def _input_row(root: Path, iid: str = "two-people-wave-001") -> dict:
    source, anchor = _write_media(root, iid)
    return {
        "schema_version": "test-prefilter-v1",
        "iid": iid,
        "group_id": f"group-{iid}",
        "family": "wave",
        "src_video": source.name,
        "resolved_src_video": str(source.resolve()),
        "source_caption": "POISON_SOURCE_CAPTION_MUST_NOT_ENTER_PROMPT",
        "edited_caption": "POISON_EDITED_CAPTION_MUST_NOT_ENTER_PROMPT",
        "prompt": "Change only the left person's peace sign to a wave.",
        "anchor_image": str(anchor.resolve()),
        "resolved_anchor_image": str(anchor.resolve()),
        "anchor_sha256": _file_digest(anchor),
        "source_video_sha256": _file_digest(source),
        "prefilter_score": 0.97,
        "media": {
            "width": 64,
            "height": 48,
            "frame_count": 81,
            "fps": 25.0,
            "duration_seconds": 3.24,
        },
        "motion": {"label": "dynamic_object"},
    }


class _FakeBackend:
    model_revision = "fake-qwen3-vl-32b-revision"
    transformers_version = "5.5.4"
    instances: list["_FakeBackend"] = []

    def __init__(self, **kwargs) -> None:
        self.model_path = kwargs["model_path"]
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict]] = []
        self.proposals: dict | None = None
        self.inventory: dict | None = None
        self.assignments: dict | None = None
        self.authority: dict | None = None
        self.grounding: dict | None = None
        self.source: dict | None = None
        self.plan: dict | None = None
        type(self).instances.append(self)

    def generate_coverage_authority_inventory(self, **kwargs):
        self.calls.append(("coverage_inventory", kwargs))
        iid = json.loads(
            kwargs["user"].split("Exact IID: ", 1)[1].split("\n", 1)[0]
        )
        self.inventory = _coverage_inventory(iid)
        return (
            json.dumps(self.inventory),
            kwargs["expected_visual_input_digest"],
        )

    def generate_coverage_authority_assignments(self, **kwargs):
        self.calls.append(("coverage_assignments", kwargs))
        iid = json.loads(
            kwargs["user"].split("Exact IID: ", 1)[1].split("\n", 1)[0]
        )
        inventory_text = kwargs["user"].split(
            "Validated A0a inventory JSON:\n", 1
        )[1].split("\nExact validated A0a inventory SHA-256:", 1)[0]
        self.inventory = json.loads(inventory_text)
        proposals_text = kwargs["user"].split(
            "Deterministic change-region proposals JSON:\n", 1
        )[1].split("\nExact change-region proposals SHA-256:", 1)[0]
        self.proposals = json.loads(proposals_text)
        self.assignments = _coverage_assignments(
            iid, self.proposals, self.inventory
        )
        return (
            json.dumps(self.assignments),
            kwargs["expected_visual_input_digest"],
        )

    def generate_i0_grounding(self, **kwargs):
        self.calls.append(("i0_grounding", kwargs))
        iid = json.loads(
            kwargs["user"].split("Exact IID: ", 1)[1].split("\n", 1)[0]
        )
        self.grounding = _i0_grounding(iid)
        return (
            json.dumps(self.grounding),
            kwargs["expected_visual_input_digest"],
        )

    def generate_source_census(self, **kwargs):
        self.calls.append(("source", kwargs))
        iid = json.loads(kwargs["user"].split("Exact IID: ", 1)[1].split("\n", 1)[0])
        self.source = _source_census(iid)
        return json.dumps(self.source), kwargs["expected_visual_input_digest"]

    def generate_secondary_source_census(self, **kwargs):
        self.calls.append(("secondary_source", kwargs))
        assert self.source is not None
        return json.dumps(self.source), kwargs["expected_visual_input_digest"]

    def generate_target_plan(self, **kwargs):
        self.calls.append(("target", kwargs))
        assert self.source is not None
        self.plan = _target_plan(self.source)
        return json.dumps(self.plan), kwargs["expected_visual_input_digest"]

    def generate_coverage_critic(self, **kwargs):
        self.calls.append(("critic", kwargs))
        assert self.source is not None and self.plan is not None
        compiled = compile_full_motion_instruction(self.source, self.plan)
        return (
            json.dumps(_critic(self.source, self.plan, compiled)),
            kwargs["expected_visual_input_digest"],
        )


class _VisualInputs(dict):
    def to(self, _device):
        return self


class _RecordingProcessor:
    def __init__(self) -> None:
        self.messages = None
        self.images = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "rendered"

    def __call__(self, *, images, **_kwargs):
        self.images = images
        return _VisualInputs(input_ids=np.asarray([[1]]))


class _RecordingVisualBackend:
    mode = "visual"
    max_new_tokens = 16

    def __init__(self) -> None:
        self.processor = _RecordingProcessor()
        self.model = SimpleNamespace(
            device="cpu",
            generate=lambda **_kwargs: np.asarray([[1, 2]]),
        )
        self.torch = SimpleNamespace(inference_mode=nullcontext)

    @staticmethod
    def _decode(_inputs, _generated, _processor):
        return '{"ok":true}'


class _MissingRightBackend(_FakeBackend):
    def generate_target_plan(self, **kwargs):
        self.calls.append(("target", kwargs))
        assert self.source is not None
        self.plan = _target_plan(self.source)
        self.plan["dynamic_unit_targets"].pop()
        return json.dumps(self.plan), kwargs["expected_visual_input_digest"]


class _SecondaryMissingRightBackend(_FakeBackend):
    def generate_secondary_source_census(self, **kwargs):
        self.calls.append(("secondary_source", kwargs))
        assert self.source is not None
        secondary = json.loads(json.dumps(self.source))
        secondary["i0_visible_entities"] = [
            "the blue-shirted man on viewer-left"
        ]
        secondary["i0_entity_registry"] = secondary["i0_entity_registry"][:1]
        secondary["dynamic_units"] = secondary["dynamic_units"][:1]
        return json.dumps(secondary), kwargs["expected_visual_input_digest"]


class _CoverageAuthorityMissingRightBackend(_FakeBackend):
    def generate_coverage_authority_inventory(self, **kwargs):
        raw, digest = super().generate_coverage_authority_inventory(**kwargs)
        inventory = json.loads(raw)
        inventory["i0_subjects"] = inventory["i0_subjects"][:1]
        self.inventory = inventory
        return json.dumps(inventory), digest

    def generate_coverage_authority_assignments(self, **kwargs):
        raw, digest = super().generate_coverage_authority_assignments(**kwargs)
        assignments = json.loads(raw)
        assignments["change_region_assignments"][1].update(
            {
                "assignment_kind": "reject_artifact",
                "authority_entity_ids": [],
                "resolution_reason": (
                    "Incorrectly dismisses right-side motion as compression"
                ),
                "reject_reason_code": "compression_noise",
            }
        )
        self.assignments = assignments
        return json.dumps(assignments), digest


class _StandaloneProseBackend(_FakeBackend):
    def generate_target_plan(self, **kwargs):
        self.calls.append(("target", kwargs))
        assert self.source is not None
        self.plan = _target_plan(self.source)
        self.plan["dynamic_unit_targets"][0]["novel_target_motion"] = (
            "his bare hand rises from waist height, opens fully, and waves "
            "twice"
        )
        self.plan["dynamic_unit_targets"][0]["target_clause"] = (
            "have the blue-shirted man raise an open bare hand and wave twice"
        )
        self.plan["dynamic_unit_targets"][1]["novel_target_motion"] = (
            "from I0, his gloved hand opens while rising and completes two "
            "side-to-side waves"
        )
        self.plan["dynamic_unit_targets"][1]["target_clause"] = (
            "have the tattooed man open his rising gloved hand and wave twice"
        )
        return json.dumps(self.plan), kwargs["expected_visual_input_digest"]


class _RedundantFieldDriftBackend(_FakeBackend):
    """Omit only canonicalizer-whitelisted redundant identity fields."""

    def generate_source_census(self, **kwargs):
        self.calls.append(("source", kwargs))
        iid = json.loads(kwargs["user"].split("Exact IID: ", 1)[1].split("\n", 1)[0])
        canonical = _source_census(iid)
        raw = json.loads(json.dumps(canonical))
        raw.pop("i0_visible_entities")
        raw["i0_entity_registry"][0]["viewer_region"] = "lower_right"
        raw["dynamic_units"][0].pop("stable_reference")
        self.source = canonical
        return json.dumps(raw), kwargs["expected_visual_input_digest"]

    def generate_secondary_source_census(self, **kwargs):
        self.calls.append(("secondary_source", kwargs))
        assert self.source is not None
        raw = json.loads(json.dumps(self.source))
        raw["dynamic_units"][1].pop("stable_reference")
        return json.dumps(raw), kwargs["expected_visual_input_digest"]

    def generate_target_plan(self, **kwargs):
        self.calls.append(("target", kwargs))
        assert self.source is not None
        canonical = _target_plan(self.source)
        raw = json.loads(json.dumps(canonical))
        raw["dynamic_unit_targets"][0].pop("stable_reference")
        self.plan = canonical
        return json.dumps(raw), kwargs["expected_visual_input_digest"]


class _A0aLongCameraEvidenceRepairBackend(_FakeBackend):
    """Reproduce the v11 >512-char evidence defect, then repair it."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inventory_attempts = 0

    def generate_coverage_authority_inventory(self, **kwargs):
        raw, digest = super().generate_coverage_authority_inventory(**kwargs)
        self.inventory_attempts += 1
        inventory = json.loads(raw)
        if self.inventory_attempts == 1:
            inventory["camera"]["motion_evidence"][0]["description"] = (
                "x" * 588
            )
        self.inventory = inventory
        return json.dumps(inventory), digest


class _A0aRepairStillInvalidBackend(_A0aLongCameraEvidenceRepairBackend):
    """The only repair changes the long evidence description to an empty one."""

    def generate_coverage_authority_inventory(self, **kwargs):
        raw, digest = super().generate_coverage_authority_inventory(**kwargs)
        if self.inventory_attempts == 2:
            inventory = json.loads(raw)
            inventory["camera"]["motion_evidence"][0]["description"] = ""
            self.inventory = inventory
            return json.dumps(inventory), digest
        return raw, digest


class _A0bDynamicCameraRepairBackend(_FakeBackend):
    """Reproduce v11's uncovered dynamic-camera assignment, then cover it."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.assignment_attempts = 0

    def generate_coverage_authority_inventory(self, **kwargs):
        raw, digest = super().generate_coverage_authority_inventory(**kwargs)
        inventory = json.loads(raw)
        inventory["camera"].update(
            {
                "dynamic": True,
                "motion_class": "pan_left",
                "motion_evidence": [
                    _evidence("the full background shifts right", 0, 80)
                ],
            }
        )
        self.inventory = inventory
        return json.dumps(inventory), digest

    def generate_coverage_authority_assignments(self, **kwargs):
        raw, digest = super().generate_coverage_authority_assignments(**kwargs)
        self.assignment_attempts += 1
        assignments = json.loads(raw)
        if self.assignment_attempts == 2:
            owner_counts: dict[str, int] = {}
            for assignment in assignments["change_region_assignments"]:
                for owner in assignment["authority_entity_ids"]:
                    owner_counts[owner] = owner_counts.get(owner, 0) + 1
            duplicate_owner = next(
                owner for owner, count in owner_counts.items() if count > 1
            )
            camera_row = next(
                assignment
                for assignment in assignments["change_region_assignments"]
                if assignment["authority_entity_ids"] == [duplicate_owner]
            )
            camera_row.update(
                {
                    "assignment_kind": "camera",
                    "authority_entity_ids": [],
                    "resolution_reason": (
                        "The fixed-cell background displacement is explained "
                        "by the validated leftward camera pan"
                    ),
                }
            )
        self.assignments = assignments
        return json.dumps(assignments), digest

    def generate_source_census(self, **kwargs):
        self.calls.append(("source", kwargs))
        iid = json.loads(
            kwargs["user"].split("Exact IID: ", 1)[1].split("\n", 1)[0]
        )
        self.source = _source_census(iid)
        self.source["camera"].update(
            {
                "motion_class": "pan_left",
                "motion_signature": "slow_pan_left",
                "motion_description": "slow pan left",
                "dynamic": True,
                "motion_evidence": [
                    _evidence("the full background shifts right", 0, 80)
                ],
            }
        )
        return json.dumps(self.source), kwargs["expected_visual_input_digest"]

    def generate_target_plan(self, **kwargs):
        self.calls.append(("target", kwargs))
        assert self.source is not None
        self.plan = _target_plan(self.source)
        self.plan["camera_target"].update(
            {
                "motion_relation": "replace_motion",
                "target_motion_class": "dolly_in",
                "target_motion_signature": "steady_dolly_in",
                "target_motion_description": (
                    "a steady forward dolly toward both actors"
                ),
                "target_clause": "move the camera steadily toward both actors",
                "source_motion_suppressed": True,
                "substantive_change": True,
                "ordered_stages": [
                    "begin moving forward from exact I0",
                    "continue the steady forward dolly through the end",
                ],
            }
        )
        return json.dumps(self.plan), kwargs["expected_visual_input_digest"]


class _A0aSelfNegatedHeadLabelBackend(_FakeBackend):
    """Emit a head label whose own only evidence explicitly denies motion."""

    def generate_coverage_authority_inventory(self, **kwargs):
        raw, digest = super().generate_coverage_authority_inventory(**kwargs)
        inventory = json.loads(raw)
        subject = inventory["i0_subjects"][0]
        subject["motion_component_types"] = ["gesture", "head_or_gaze"]
        subject["motion_evidence"][0]["description"] = (
            "the left person's hand rises while head orientation remains "
            "stable with only a slight smile variation"
        )
        self.inventory = inventory
        return json.dumps(inventory), digest


class _A0bExplicitCameraEnumMismatchBackend(_A0bDynamicCameraRepairBackend):
    """Explain a pan as camera-caused while emitting reject_artifact."""

    def generate_coverage_authority_assignments(self, **kwargs):
        raw, digest = super().generate_coverage_authority_assignments(**kwargs)
        assignments = json.loads(raw)
        owner_counts: dict[str, int] = {}
        for assignment in assignments["change_region_assignments"]:
            for owner in assignment["authority_entity_ids"]:
                owner_counts[owner] = owner_counts.get(owner, 0) + 1
        duplicate_owner = next(
            owner for owner, count in owner_counts.items() if count > 1
        )
        camera_row = next(
            assignment
            for assignment in assignments["change_region_assignments"]
            if assignment["authority_entity_ids"] == [duplicate_owner]
        )
        camera_row.update(
            {
                "assignment_kind": "reject_artifact",
                "authority_entity_ids": [],
                "resolution_reason": (
                    "The background displacement is due to camera pan-left "
                    "motion rather than an independently moving entity"
                ),
                "reject_reason_code": "background_nonsemantic_motion",
            }
        )
        self.assignments = assignments
        return json.dumps(assignments), digest


class _TargetPlanMissingUnitIdRepairBackend(_FakeBackend):
    """Reproduce canary PASS_B output with both dynamic unit IDs omitted."""

    def generate_target_plan(self, **kwargs):
        self.calls.append(("target", kwargs))
        assert self.source is not None
        canonical = _target_plan(self.source)
        raw = json.loads(json.dumps(canonical))
        for target in raw["dynamic_unit_targets"]:
            target.pop("unit_id")
        self.plan = canonical
        return json.dumps(raw), kwargs["expected_visual_input_digest"]

    def generate_target_plan_schema_repair(self, **kwargs):
        self.calls.append(("target_schema_repair", kwargs))
        assert self.source is not None
        assert kwargs["expected_visual_input_digest"] is None
        self.plan = _target_plan(self.source)
        return json.dumps(self.plan), None


class _TargetPlanRepairStillMissingUnitIdBackend(
    _TargetPlanMissingUnitIdRepairBackend
):
    def generate_target_plan_schema_repair(self, **kwargs):
        self.calls.append(("target_schema_repair", kwargs))
        assert self.source is not None
        assert kwargs["expected_visual_input_digest"] is None
        raw = _target_plan(self.source)
        for target in raw["dynamic_unit_targets"]:
            target.pop("unit_id")
        self.plan = raw
        return json.dumps(raw), None


class _TargetPlanRepairMutatesActionBackend(
    _TargetPlanMissingUnitIdRepairBackend
):
    """Return a contract-valid-looking repair that also changes an action."""

    def generate_target_plan_schema_repair(self, **kwargs):
        self.calls.append(("target_schema_repair", kwargs))
        assert self.source is not None
        assert kwargs["expected_visual_input_digest"] is None
        raw = _target_plan(self.source)
        raw["dynamic_unit_targets"][0]["novel_target_motion"] = (
            "raise the open hand overhead and trace one slow circle"
        )
        raw["dynamic_unit_targets"][0]["target_clause"] = (
            "raise the open hand overhead and trace one slow circle"
        )
        self.plan = raw
        return json.dumps(raw), None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _args(input_path: Path, output: Path, row: dict) -> argparse.Namespace:
    shard = int(hashlib.sha256(row["iid"].encode()).hexdigest()[:16], 16) % 8
    return argparse.Namespace(
        input=input_path,
        output=output,
        model="/fake/Qwen3-VL-32B-Instruct",
        root=input_path.parent,
        resume=False,
        allow_errors=False,
        max_samples=None,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        repair_attempts=0,
        nframes=DEFAULT_NFRAMES,
        max_pixels=DEFAULT_MAX_PIXELS,
        tile_width=DEFAULT_TILE_WIDTH,
        mosaic_columns=DEFAULT_MOSAIC_COLUMNS,
        shard_index=shard,
        num_shards=8,
        all_shards_sequential=False,
        sequential_shards=None,
        attn_implementation="sdpa",
        allow_download=False,
    )


class GokuFullMotionQwenTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeBackend.instances.clear()

    def test_video_mosaic_columns_are_configurable_and_default_to_three(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            source, _ = _write_media(Path(name), "mosaic-columns")
            default = _video_mosaic(str(source), nframes=8, tile_width=32)
            four_columns = _video_mosaic(
                str(source), nframes=8, tile_width=32, columns=4
            )
            self.assertEqual(default.width, 3 * 32)
            self.assertEqual(four_columns.width, 4 * 32)
            self.assertGreater(default.height, four_columns.height)
            with self.assertRaises(ValueError):
                _video_mosaic(
                    str(source), nframes=8, tile_width=32, columns=0
                )

    def test_visual_bundle_includes_temporal_and_motion_attention_views(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            source, anchor = _write_media(Path(name), "visual-checkpoints")
            i0, mosaic, triptych, lr_zoom, attention, digest = _build_visuals(
                source_path=source,
                anchor_path=anchor,
                nframes=8,
                max_pixels=DEFAULT_MAX_PIXELS,
                tile_width=32,
                mosaic_columns=4,
            )
            self.assertEqual(i0.size, (64, 48))
            self.assertGreater(mosaic.width, i0.width)
            self.assertEqual(triptych.size, (192, 80))
            self.assertEqual(lr_zoom.size, (117, 160))
            self.assertEqual(attention.size, (128, 80))
            self.assertGreater(np.asarray(attention)[32:, :64].sum(), 0)
            self.assertEqual(
                digest,
                _visual_digest(
                    (
                        ("exact_i0", i0),
                        ("source_mosaic", mosaic),
                        ("source_temporal_triptych", triptych),
                        ("source_temporal_lr_zoom", lr_zoom),
                        ("source_motion_attention", attention),
                    )
                ),
            )

    def test_authority_grid_and_change_proposals_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, anchor = _write_media(root, "authority-grid-001")
            with Image.open(anchor) as opened:
                exact_i0 = opened.convert("RGB").copy()
            first_grid, first_proposals = _build_authority_grid_and_proposals(
                source_path=source,
                exact_i0=exact_i0,
                iid="authority-grid-001",
                max_pixels=DEFAULT_MAX_PIXELS,
            )
            second_grid, second_proposals = _build_authority_grid_and_proposals(
                source_path=source,
                exact_i0=exact_i0,
                iid="authority-grid-001",
                max_pixels=DEFAULT_MAX_PIXELS,
            )
            self.assertEqual(first_proposals, second_proposals)
            self.assertEqual(
                contract.object_sha256(first_proposals),
                contract.object_sha256(second_proposals),
            )
            self.assertEqual(
                _visual_digest((("authority_grid", first_grid),)),
                _visual_digest((("authority_grid", second_grid),)),
            )
            self.assertEqual(
                first_proposals["frame_indices"], [0, 20, 40, 60, 80]
            )
            self.assertEqual(first_proposals["grid_rows"], 4)
            self.assertEqual(first_proposals["grid_columns"], 4)
            self.assertTrue(first_proposals["regions"])
            self.assertLessEqual(len(first_proposals["regions"]), 16)
            active_cells = {
                (item["cell_row"], item["cell_column"])
                for item in first_proposals["regions"]
            }
            self.assertEqual(
                len(active_cells), len(first_proposals["regions"])
            )
            for actor_bbox in ([50, 150, 350, 900], [650, 150, 950, 900]):
                self.assertTrue(
                    any(
                        min(region["bbox_xyxy_1000"][2], actor_bbox[2])
                        > max(region["bbox_xyxy_1000"][0], actor_bbox[0])
                        and min(region["bbox_xyxy_1000"][3], actor_bbox[3])
                        > max(region["bbox_xyxy_1000"][1], actor_bbox[1])
                        for region in first_proposals["regions"]
                    ),
                    "active-cell proposals must cover both moving canary actors",
                )

    def test_temporal_prompts_forbid_state_backfill_and_attention_evidence(
        self,
    ) -> None:
        self.assertIn(
            "Never copy a hand state visible at CM/CF backward into C0",
            " ".join(PASS_A_PROMPT.split()),
        )
        self.assertIn("Never infer C0 from a later checkpoint", PASS_A2_SYSTEM)
        self.assertIn("at or below the waistband", " ".join(PASS_A_SYSTEM.split()))
        self.assertIn("at or below the waistband", " ".join(PASS_A2_SYSTEM.split()))
        self.assertIn("finite motion verb", " ".join(PASS_A_PROMPT.split()))
        self.assertIn("Recheck every i0_state", " ".join(PASS_C_SYSTEM.split()))
        for system_prompt in (
            PASS_A_SYSTEM,
            PASS_A2_SYSTEM,
            PASS_B_SYSTEM,
            PASS_C_SYSTEM,
        ):
            self.assertIn("not motion evidence", system_prompt)
        self.assertNotIn(
            "motion-attention aid, and a validated\nblind source census are authoritative",
            PASS_B_SYSTEM,
        )

    def test_a1_a2_prompts_share_held_object_closure_rule(self) -> None:
        for prompt in (
            PASS_A_SYSTEM,
            PASS_A_PROMPT,
            PASS_A2_SYSTEM,
            PASS_A2_PROMPT,
        ):
            normalized = " ".join(prompt.split())
            self.assertEqual(prompt.count(HELD_CARRIED_OBJECT_CLOSURE_RULE), 1)
            self.assertIn("independent actor", prompt)
            self.assertIn("passive_interaction_object", prompt)
            self.assertIn("object_interaction component", prompt)
            self.assertIn("dependent_entity_ids", prompt)
            self.assertIn("object visibly travels with its holder", prompt)
            self.assertIn("contact and object remain spatially static", prompt)
            self.assertIn("evidence description must itself state", normalized)
            self.assertIn("grip is steady", normalized)
            self.assertNotIn("makes, breaks, or maintains", prompt)
            self.assertNotIn(
                "Any action-relevant physical contact with another", prompt
            )

    def test_a0_prompts_close_text_held_prop_owner_and_camera_rules(self) -> None:
        inventory_prompt = " ".join(
            COVERAGE_AUTHORITY_INVENTORY_SYSTEM.split()
        )
        assignments_prompt = " ".join(
            COVERAGE_AUTHORITY_ASSIGNMENTS_SYSTEM.split()
        )
        self.assertIn("at or below 512 Unicode characters", inventory_prompt)
        self.assertIn("at or below 512 Unicode characters", assignments_prompt)
        self.assertIn("object visibly travels with its holder", inventory_prompt)
        self.assertIn("Do not invent a separate", inventory_prompt)
        self.assertIn("constant grip does not suppress", inventory_prompt)
        self.assertIn("contact and object remain spatially static", inventory_prompt)
        self.assertIn("do not include object_interaction", inventory_prompt)
        self.assertIn("arm/hand gesture alone is not body_pose", inventory_prompt)
        self.assertIn("Human/animal head motion is head_or_gaze", inventory_prompt)
        for required in (
            "every supplied proposal_id occurs exactly once",
            "exact allowed-owner-map row",
            "covers every dynamic A0a authority",
            "dynamic camera has at least one camera assignment",
            "locked-off camera has none",
            "Background content does not make camera-caused displacement",
            "row MUST use assignment_kind=camera",
        ):
            self.assertIn(required, assignments_prompt)

    def test_source_prompts_keep_review_inventory_derived_and_components_dynamic(
        self,
    ) -> None:
        for prompt in (PASS_A_SYSTEM, PASS_A2_SYSTEM):
            normalized = " ".join(prompt.split())
            self.assertIn(
                "i0_visible_entities to the exact stable_reference strings",
                normalized,
            )
            self.assertIn("head_steady_forward", normalized)
            self.assertIn("hold_lead_rope_steady", normalized)
            self.assertIn("halter_attached_to_lead_rope", normalized)
        self.assertIn(
            "exact stable_reference values from the registry below",
            SOURCE_CENSUS_PROMPT_SCHEMA["i0_visible_entities"][0],
        )

    def test_i0_grounding_is_closed_viewer_relative_and_has_no_future(self) -> None:
        grounding = _i0_grounding("grounding-contract-001")
        self.assertEqual(
            validate_i0_grounding(
                grounding, expected_iid="grounding-contract-001"
            ),
            grounding,
        )
        anatomical = json.loads(json.dumps(grounding))
        anatomical["subjects"][0]["left_hand_height"] = "waistband"
        with self.assertRaisesRegex(GokuFullMotionQwenError, "keys differ"):
            validate_i0_grounding(
                anatomical, expected_iid="grounding-contract-001"
            )
        future = json.loads(json.dumps(grounding))
        future["subjects"][1]["i0_state"] += " then raises the glove later"
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "future/temporal language"
        ):
            validate_i0_grounding(
                future, expected_iid="grounding-contract-001"
            )

    def test_source_census_cannot_rewrite_i0_only_grounding(self) -> None:
        iid = "grounding-binding-001"
        grounding = _i0_grounding(iid)
        source = _source_census(iid)
        self.assertEqual(
            validate_source_census_i0_binding(source, grounding), source
        )
        changed = json.loads(json.dumps(source))
        changed["dynamic_units"][1]["i0_state"] = (
            "Standing at I0 with the black glove at chest height"
        )
        with self.assertRaisesRegex(GokuFullMotionQwenError, "i0_state differs"):
            validate_source_census_i0_binding(changed, grounding)

    def test_coverage_authority_is_closed_and_resolves_every_region(self) -> None:
        iid = "coverage-authority-contract-001"
        proposals = _change_region_proposals(iid)
        inventory = _coverage_inventory(iid)
        assignments = _coverage_assignments(iid, proposals, inventory)
        authority = _coverage_authority(iid, proposals)
        self.assertEqual(
            validate_change_region_proposals(proposals, expected_iid=iid),
            proposals,
        )
        self.assertEqual(
            validate_coverage_authority(
                authority,
                expected_iid=iid,
                change_region_proposals=proposals,
            ),
            authority,
        )
        self.assertEqual(
            validate_coverage_authority_inventory(
                inventory, expected_iid=iid
            ),
            inventory,
        )
        self.assertEqual(
            validate_coverage_authority_assignments(
                assignments,
                expected_iid=iid,
                coverage_authority_inventory=inventory,
                change_region_proposals=proposals,
            ),
            assignments,
        )

        missing = json.loads(json.dumps(authority))
        missing["assignments"]["change_region_assignments"].pop()
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "does not resolve every change proposal"
        ):
            validate_coverage_authority(
                missing,
                expected_iid=iid,
                change_region_proposals=proposals,
            )

        uncertain = json.loads(json.dumps(authority))
        uncertain["inventory"]["uncertainty_codes"] = [
            "right_actor_uncertain"
        ]
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "uncertainty_codes must be empty"
        ):
            validate_coverage_authority(
                uncertain,
                expected_iid=iid,
                change_region_proposals=proposals,
            )

        wrong_owner = json.loads(json.dumps(authority))
        wrong_owner["assignments"]["change_region_assignments"][1][
            "authority_entity_ids"
        ] = ["authority_subject_01"]
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "allowed-owner map"
        ):
            validate_coverage_authority(
                wrong_owner,
                expected_iid=iid,
                change_region_proposals=proposals,
            )

    def test_two_stage_coverage_prompts_are_blind_and_owner_map_bound(self) -> None:
        iid = "coverage-authority-prompt-001"
        row = {
            "iid": iid,
            "prompt": "POISON_LEGACY_EDIT",
            "source_caption": "POISON_SOURCE_CAPTION",
            "edited_caption": "POISON_EDITED_CAPTION",
            "media": {"frame_count": 81, "fps": 25.0},
        }
        proposals = _change_region_proposals(iid)
        inventory = _coverage_inventory(iid)
        inventory_prompt = build_coverage_authority_inventory_prompt(
            row=row, nframes=DEFAULT_NFRAMES
        )
        self.assertNotIn("POISON_", inventory_prompt)
        self.assertNotIn("target_plan", inventory_prompt)
        self.assertNotIn("source_census_sha256", inventory_prompt)
        self.assertNotIn("proposal_01", inventory_prompt)
        self.assertNotIn("change_region_proposals_sha256", inventory_prompt)

        owner_map = build_coverage_authority_allowed_owner_map(
            coverage_authority_inventory=inventory,
            change_region_proposals=proposals,
        )
        self.assertEqual(
            [
                row["allowed_dynamic_owner_ids"]
                for row in owner_map["proposal_owner_rows"]
            ],
            [["authority_subject_01"], ["authority_subject_02"]],
        )
        self.assertEqual(
            validate_coverage_authority_allowed_owner_map(
                owner_map,
                coverage_authority_inventory=inventory,
                change_region_proposals=proposals,
            ),
            owner_map,
        )
        assignments_prompt = build_coverage_authority_assignments_prompt(
            row=row,
            coverage_authority_inventory=inventory,
            change_region_proposals=proposals,
        )
        self.assertNotIn("POISON_", assignments_prompt)
        self.assertIn("Deterministic change-region proposals JSON", assignments_prompt)
        self.assertIn("Validated A0a inventory JSON", assignments_prompt)
        self.assertIn(contract.object_sha256(owner_map), assignments_prompt)
        self.assertIn(
            json.dumps(owner_map, sort_keys=True, separators=(",", ":")),
            assignments_prompt,
        )

        tampered = _coverage_assignments(iid, proposals, inventory)
        tampered["allowed_owner_map_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "allowed-owner-map digest differs"
        ):
            validate_coverage_authority_assignments(
                tampered,
                expected_iid=iid,
                coverage_authority_inventory=inventory,
                change_region_proposals=proposals,
            )

    def test_assignment_gate_requires_every_dynamic_owner_and_dynamic_camera(self) -> None:
        iid = "coverage-authority-full-coverage-001"
        proposals = _change_region_proposals(iid)
        proposals["regions"][1].update(
            {
                "cell_row": 4,
                "cell_column": 1,
                "bbox_xyxy_1000": [0, 750, 250, 1000],
            }
        )
        inventory = _coverage_inventory(iid)
        assignments = _coverage_assignments(iid, proposals, inventory)
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "dynamic authorities.*uncovered"
        ):
            validate_coverage_authority_assignments(
                assignments,
                expected_iid=iid,
                coverage_authority_inventory=inventory,
                change_region_proposals=proposals,
            )

        original_proposals = _change_region_proposals(iid)
        moving_camera_inventory = _coverage_inventory(iid)
        moving_camera_inventory["camera"].update(
            {"dynamic": True, "motion_class": "pan_left"}
        )
        no_camera = _coverage_assignments(
            iid, original_proposals, moving_camera_inventory
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "dynamic camera uncovered"
        ):
            validate_coverage_authority_assignments(
                no_camera,
                expected_iid=iid,
                coverage_authority_inventory=moving_camera_inventory,
                change_region_proposals=original_proposals,
            )

    def test_a0a_canonicalization_drops_only_self_negated_head_label(self) -> None:
        iid = "coverage-authority-self-negated-head-001"
        inventory = _coverage_inventory(iid)
        subject = inventory["i0_subjects"][0]
        subject["motion_component_types"] = ["gesture", "head_or_gaze"]
        subject["motion_evidence"][0]["description"] = (
            "the hand rises while head orientation remains stable with a "
            "slight smile variation"
        )
        canonical = canonicalize_coverage_authority_inventory_model_output(
            inventory, expected_iid=iid
        )
        self.assertEqual(
            canonical["i0_subjects"][0]["motion_component_types"],
            ["gesture"],
        )
        self.assertEqual(
            inventory["i0_subjects"][0]["motion_component_types"],
            ["gesture", "head_or_gaze"],
            "canonicalization must not mutate the raw response",
        )

        positive = json.loads(json.dumps(inventory))
        positive["i0_subjects"][0]["motion_evidence"][0][
            "description"
        ] += "; the subject then turns his head left"
        self.assertEqual(
            canonicalize_coverage_authority_inventory_model_output(
                positive, expected_iid=iid
            )["i0_subjects"][0]["motion_component_types"],
            ["gesture", "head_or_gaze"],
        )

        positive_eye_motion = json.loads(json.dumps(inventory))
        positive_eye_motion["i0_subjects"][0]["motion_evidence"][0][
            "description"
        ] += "; while the head remains stable, his eyes dart left"
        self.assertEqual(
            canonicalize_coverage_authority_inventory_model_output(
                positive_eye_motion, expected_iid=iid
            )["i0_subjects"][0]["motion_component_types"],
            ["gesture", "head_or_gaze"],
        )

        mixed = json.loads(json.dumps(inventory))
        mixed["i0_subjects"][0]["motion_evidence"].append(
            _evidence("the face remains visible throughout", 0, 80)
        )
        self.assertEqual(
            canonicalize_coverage_authority_inventory_model_output(
                mixed, expected_iid=iid
            )["i0_subjects"][0]["motion_component_types"],
            ["gesture", "head_or_gaze"],
            "every evidence row must explicitly negate head/gaze motion",
        )

    def test_a0b_canonicalization_requires_exact_camera_causality_and_class(
        self,
    ) -> None:
        iid = "coverage-authority-camera-enum-canonicalization-001"
        proposals = _change_region_proposals(iid)
        proposals["regions"].insert(
            0,
            {
                "schema_version": CHANGE_REGION_SCHEMA,
                "proposal_id": "proposal_01",
                "cell_row": 1,
                "cell_column": 2,
                "bbox_xyxy_1000": [250, 0, 500, 250],
                "changed_pixel_count": 10,
                "bbox_area_pixels": 100,
                "changed_fraction_ppm": 100_000,
                "delta_at_percentile_milli": 30_000,
            }
        )
        proposals["regions"][1]["proposal_id"] = "proposal_02"
        proposals["regions"][2]["proposal_id"] = "proposal_03"
        proposals["active_cell_count"] = 3
        inventory = _coverage_inventory(iid)
        inventory["i0_subjects"][0]["i0_bbox_xyxy_1000"] = [
            50,
            550,
            200,
            700,
        ]
        inventory["i0_subjects"][0][
            "temporal_extent_bbox_xyxy_1000"
        ] = [0, 500, 250, 750]
        inventory["i0_subjects"][1]["i0_bbox_xyxy_1000"] = [
            800,
            550,
            950,
            700,
        ]
        inventory["i0_subjects"][1][
            "temporal_extent_bbox_xyxy_1000"
        ] = [750, 500, 1000, 750]
        inventory["camera"].update(
            {"dynamic": True, "motion_class": "pan_left"}
        )
        assignments = _coverage_assignments(iid, proposals, inventory)
        assignments["change_region_assignments"][0].update(
            {
                "assignment_kind": "reject_artifact",
                "authority_entity_ids": [],
                "resolution_reason": (
                    "The fixed background displacement is due to camera "
                    "pan-left motion"
                ),
                "reject_reason_code": "background_nonsemantic_motion",
            }
        )
        canonical = canonicalize_coverage_authority_assignments_model_output(
            assignments,
            expected_iid=iid,
            coverage_authority_inventory=inventory,
            change_region_proposals=proposals,
        )
        self.assertEqual(
            canonical["change_region_assignments"][0]["assignment_kind"],
            "camera",
        )
        self.assertIsNone(
            canonical["change_region_assignments"][0]["reject_reason_code"]
        )
        self.assertEqual(
            assignments["change_region_assignments"][0]["assignment_kind"],
            "reject_artifact",
            "canonicalization must preserve the exact raw response",
        )

        dolly_inventory = json.loads(json.dumps(inventory))
        dolly_inventory["camera"]["motion_class"] = "dolly_in"
        for reason in (
            (
                "The change in the top-left region is due to camera dolly-in "
                "motion, causing background elements to appear to recede."
            ),
            (
                "The sky region shows minimal change, consistent with "
                "background nonsemantic motion due to camera dolly-in; no "
                "dynamic entity is present in this region."
            ),
        ):
            v13_positive = _coverage_assignments(
                iid, proposals, dolly_inventory
            )
            v13_positive["change_region_assignments"][0].update(
                {
                    "assignment_kind": "reject_artifact",
                    "authority_entity_ids": [],
                    "resolution_reason": reason,
                    "reject_reason_code": "background_nonsemantic_motion",
                }
            )
            self.assertEqual(
                canonicalize_coverage_authority_assignments_model_output(
                    v13_positive,
                    expected_iid=iid,
                    coverage_authority_inventory=dolly_inventory,
                    change_region_proposals=proposals,
                )["change_region_assignments"][0]["assignment_kind"],
                "camera",
            )

        for reason in (
            "The camera pan-left is visible but no causal attribution is made",
            "The displacement is due to camera dolly-in motion",
            "The displacement is due to camera shake; pan-left is absent",
            "The displacement is not due to camera pan-left motion",
            "The displacement is due to camera pan-left motion being absent",
        ):
            adversarial = json.loads(json.dumps(assignments))
            adversarial["change_region_assignments"][0][
                "resolution_reason"
            ] = reason
            with self.assertRaisesRegex(
                GokuFullMotionQwenError, "dynamic camera uncovered"
            ):
                canonicalize_coverage_authority_assignments_model_output(
                    adversarial,
                    expected_iid=iid,
                    coverage_authority_inventory=inventory,
                    change_region_proposals=proposals,
                )

    def test_dependent_assignment_requires_every_owner_in_allowed_map(self) -> None:
        iid = "coverage-authority-dependent-owner-001"
        proposals = _change_region_proposals(iid)
        inventory = _coverage_inventory(iid)
        assignments = _coverage_assignments(iid, proposals, inventory)
        assignments["change_region_assignments"][0].update(
            {
                "assignment_kind": "dependent_motion",
                "authority_entity_ids": [
                    "authority_subject_01",
                    "authority_subject_02",
                ],
            }
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "allowed-owner map"
        ):
            validate_coverage_authority_assignments(
                assignments,
                expected_iid=iid,
                coverage_authority_inventory=inventory,
                change_region_proposals=proposals,
            )

    def test_authority_component_alignment_requires_exact_component_sets(self) -> None:
        iid = "coverage-authority-component-subset-001"
        proposals = _change_region_proposals(iid)
        grounding = _i0_grounding(iid)
        primary = _source_census(iid)
        secondary = json.loads(json.dumps(primary))
        source_alignment = contract.build_source_inventory_alignment(
            primary=primary, secondary=secondary
        )

        surplus_inventory = _coverage_inventory(iid)
        surplus_inventory["i0_subjects"][0]["motion_component_types"] = [
            "gesture",
            "body_pose",
            "head_or_gaze",
            "object_interaction",
        ]
        surplus_assignments = _coverage_assignments(
            iid, proposals, surplus_inventory
        )
        surplus_authority = build_coverage_authority(
            coverage_authority_inventory=surplus_inventory,
            coverage_authority_assignments=surplus_assignments,
            change_region_proposals=proposals,
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "motion component set differs"
        ):
            build_coverage_authority_alignment(
                coverage_authority=surplus_authority,
                change_region_proposals=proposals,
                i0_grounding=grounding,
                primary=primary,
                secondary=secondary,
                source_inventory_alignment=source_alignment,
            )

        omitted_inventory = _coverage_inventory(iid)
        omitted_inventory["i0_subjects"][0]["motion_component_types"] = [
            "body_pose"
        ]
        omitted_assignments = _coverage_assignments(
            iid, proposals, omitted_inventory
        )
        omitted_authority = build_coverage_authority(
            coverage_authority_inventory=omitted_inventory,
            coverage_authority_assignments=omitted_assignments,
            change_region_proposals=proposals,
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "motion component set differs"
        ):
            build_coverage_authority_alignment(
                coverage_authority=omitted_authority,
                change_region_proposals=proposals,
                i0_grounding=grounding,
                primary=primary,
                secondary=secondary,
                source_inventory_alignment=source_alignment,
            )

    def test_hard_gate_rejects_coherent_all_pass_actor_omission(self) -> None:
        """A shared miss across I0/A1/A2/C is still a source-coverage miss.

        The complete fixture contains two independently moving people.  This
        regression deliberately constructs the currently accepted failure
        mode in which every model-authored artifact silently drops the second
        person and self-reports completeness.  A fail-closed gate must reject
        the record unless an independent visual coverage adjudicator proves
        that the declared census closes over the source pixels.
        """

        iid = "coherent-actor-omission-001"
        complete_source = _source_census(iid)
        self.assertEqual(len(complete_source["dynamic_units"]), 2)
        proposals = _change_region_proposals(iid)
        authority = _coverage_authority(iid, proposals)

        grounding = _i0_grounding(iid)
        grounding["subjects"] = grounding["subjects"][:1]
        declared_source = json.loads(json.dumps(complete_source))
        declared_source["i0_visible_entities"] = declared_source[
            "i0_visible_entities"
        ][:1]
        declared_source["i0_entity_registry"] = declared_source[
            "i0_entity_registry"
        ][:1]
        declared_source["dynamic_units"] = declared_source["dynamic_units"][:1]
        declared_source = validate_source_census_i0_binding(
            declared_source, grounding
        )
        secondary = json.loads(json.dumps(declared_source))
        alignment = contract.build_source_inventory_alignment(
            primary=declared_source,
            secondary=secondary,
        )

        plan = _target_plan(declared_source)
        plan["dynamic_unit_targets"] = plan["dynamic_unit_targets"][:1]
        plan["coverage"]["required_dynamic_unit_ids"] = ["unit_01"]
        plan["coverage"]["planned_changed_unit_ids"] = ["unit_01"]
        plan = contract.validate_target_plan(
            plan, source_census=declared_source
        )
        compiled = compile_full_motion_instruction(declared_source, plan)
        critic = _critic(declared_source, plan, compiled)
        critic = contract.validate_coverage_critic(
            critic,
            source_census=declared_source,
            target_plan=plan,
            compiled_instruction=compiled,
        )
        _, primary_receipt = contract.canonicalize_source_census_model_output(
            declared_source, iid
        )
        _, secondary_receipt = contract.canonicalize_source_census_model_output(
            secondary, iid
        )
        _, plan_receipt = contract.canonicalize_target_plan_model_output(
            plan, declared_source
        )
        missing_authority_gate = build_hard_gate(
            i0_grounding=grounding,
            source_census=declared_source,
            source_census_canonicalization=primary_receipt,
            secondary_source_census=secondary,
            secondary_source_census_canonicalization=secondary_receipt,
            source_inventory_alignment=alignment,
            target_plan=plan,
            target_plan_canonicalization=plan_receipt,
            compiled_instruction=compiled,
            coverage_critic=critic,
        )
        self.assertEqual(missing_authority_gate["decision"], "reject")
        self.assertIn(
            "coverage_authority:change_regions_not_strict",
            missing_authority_gate["risk_codes"],
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenError, "entity count differs"
        ):
            build_coverage_authority_alignment(
                coverage_authority=authority,
                change_region_proposals=proposals,
                i0_grounding=grounding,
                primary=declared_source,
                secondary=secondary,
                source_inventory_alignment=alignment,
            )

        gate = build_hard_gate(
            i0_grounding=grounding,
            source_census=declared_source,
            source_census_canonicalization=primary_receipt,
            secondary_source_census=secondary,
            secondary_source_census_canonicalization=secondary_receipt,
            source_inventory_alignment=alignment,
            target_plan=plan,
            target_plan_canonicalization=plan_receipt,
            compiled_instruction=compiled,
            coverage_critic=critic,
            change_region_proposals=proposals,
            coverage_authority=authority,
            coverage_authority_alignment={},
        )

        self.assertEqual(
            gate["decision"],
            "reject",
            "a self-consistent declared subset must not prove visual completeness",
        )
        self.assertIn(
            "coverage_authority:not_bound_to_grounding_and_source_censuses",
            gate["risk_codes"],
        )

    def test_hard_gate_rejects_missing_a0_artifacts(self) -> None:
        iid = "missing-a0-hard-gate-001"
        grounding = _i0_grounding(iid)
        source = _source_census(iid)
        secondary = json.loads(json.dumps(source))
        alignment = contract.build_source_inventory_alignment(
            primary=source, secondary=secondary
        )
        plan = contract.validate_target_plan(
            _target_plan(source), source_census=source
        )
        compiled = compile_full_motion_instruction(source, plan)
        critic = _critic(source, plan, compiled)
        _, primary_receipt = contract.canonicalize_source_census_model_output(
            source, iid
        )
        _, secondary_receipt = contract.canonicalize_source_census_model_output(
            secondary, iid
        )
        _, plan_receipt = contract.canonicalize_target_plan_model_output(
            plan, source
        )
        gate = build_hard_gate(
            i0_grounding=grounding,
            source_census=source,
            source_census_canonicalization=primary_receipt,
            secondary_source_census=secondary,
            secondary_source_census_canonicalization=secondary_receipt,
            source_inventory_alignment=alignment,
            target_plan=plan,
            target_plan_canonicalization=plan_receipt,
            compiled_instruction=compiled,
            coverage_critic=critic,
            change_region_proposals=None,
            coverage_authority=None,
            coverage_authority_alignment=None,
        )
        self.assertEqual(gate["decision"], "reject")
        self.assertEqual(
            gate["risk_codes"],
            ["coverage_authority:change_regions_not_strict"],
        )

    def test_target_validator_rejects_semantic_restatement_and_future_shortcut(
        self,
    ) -> None:
        """Every declared dynamic actor needs a genuinely novel absolute motion."""

        cases = {
            "same_action_paraphrase": {
                "signature": "lift_glove_into_hand_sign",
                "motion": (
                    "moves his black-gloved hand upward from his waist and "
                    "shapes it into a hand sign"
                ),
                "stages": [
                    "immediately lift the black-gloved hand upward from the waist",
                    "then shape the raised black-gloved hand into a hand sign",
                ],
            },
            "unobservable_future_shortcut": {
                "signature": "nod_while_keep_doing_action",
                "motion": "keep doing what he does while nodding twice",
                "stages": [
                    "immediately keep doing what he does",
                    "then nod twice while doing it",
                ],
            },
        }
        for label, mutation in cases.items():
            with self.subTest(label=label):
                source = _source_census(f"novelty-{label}-001")
                plan = _target_plan(source)
                target = plan["dynamic_unit_targets"][1]
                target["target_action_signature"] = mutation["signature"]
                target["novel_target_motion"] = mutation["motion"]
                target["target_clause"] = mutation["motion"]
                target["ordered_stages"] = mutation["stages"]
                with self.assertRaises(contract.GokuFullMotionContractError):
                    contract.validate_target_plan(plan, source_census=source)

    def test_i0_backend_receives_only_exact_initial_frame(self) -> None:
        image = Image.new("RGB", (8, 6), color=(10, 20, 30))
        digest = _visual_digest((("exact_i0_only", image),))
        backend = _RecordingVisualBackend()
        raw, returned_digest = _generate_i0_grounding_pass(
            backend,
            system="i0 system",
            prompt="i0 prompt",
            anchor_path=Path("anchor.png"),
            exact_i0=image,
            expected_visual_digest=digest,
        )
        self.assertEqual(raw, '{"ok":true}')
        self.assertEqual(returned_digest, digest)
        self.assertEqual(backend.processor.images, [image])
        rendered = json.dumps(
            backend.processor.messages, default=lambda _value: "IMAGE"
        )
        self.assertIn("INITIAL FRAME I0 ONLY", rendered)
        self.assertNotIn("mosaic", rendered.casefold())

    def test_coverage_authority_backend_receives_only_six_blind_images(
        self,
    ) -> None:
        images = [
            Image.new("RGB", (8, 6), color=(index, index + 1, index + 2))
            for index in range(6)
        ]
        digest = _coverage_authority_visual_digest(
            stage="a0a_inventory",
            exact_i0=images[0],
            source_mosaic=images[1],
            source_temporal_triptych=images[2],
            source_temporal_lr_zoom=images[3],
            source_motion_attention=images[4],
            source_authority_grid=images[5],
        )
        backend = _RecordingVisualBackend()
        raw, returned_digest = _generate_coverage_authority_pass(
            backend,
            custom_method="missing_custom_method",
            stage_label="A0a INVENTORY",
            system=COVERAGE_AUTHORITY_INVENTORY_SYSTEM,
            prompt="blind A0 prompt",
            source_path=Path("source.mp4"),
            anchor_path=Path("anchor.png"),
            nframes=DEFAULT_NFRAMES,
            max_pixels=DEFAULT_MAX_PIXELS,
            tile_width=DEFAULT_TILE_WIDTH,
            mosaic_columns=DEFAULT_MOSAIC_COLUMNS,
            exact_i0=images[0],
            source_mosaic=images[1],
            source_temporal_triptych=images[2],
            source_temporal_lr_zoom=images[3],
            source_motion_attention=images[4],
            source_authority_grid=images[5],
            expected_visual_digest=digest,
        )
        self.assertEqual(raw, '{"ok":true}')
        self.assertEqual(returned_digest, digest)
        self.assertEqual(backend.processor.images, images)
        rendered = json.dumps(
            backend.processor.messages, default=lambda _value: "IMAGE"
        )
        self.assertIn("FULL-FRAME 4x4 SPATIAL GRID", rendered)
        self.assertIn("another I0 grounding", rendered)
        self.assertNotIn("legacy caption, target plan", "blind A0 prompt")

    def test_grounded_temporal_zoom_digest_depends_on_fixed_i0_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, anchor = _write_media(root, "grounded-zoom-001")
            with Image.open(anchor) as opened:
                exact_i0 = opened.convert("RGB").copy()
            grounding = _i0_grounding("grounded-zoom-001")
            first = _build_grounded_temporal_zoom(
                source_path=source,
                exact_i0=exact_i0,
                i0_grounding=grounding,
                max_pixels=DEFAULT_MAX_PIXELS,
                tile_width=64,
            )
            changed = json.loads(json.dumps(grounding))
            changed["subjects"][0]["i0_bbox_xyxy_1000"] = [100, 300, 450, 950]
            second = _build_grounded_temporal_zoom(
                source_path=source,
                exact_i0=exact_i0,
                i0_grounding=changed,
                max_pixels=DEFAULT_MAX_PIXELS,
                tile_width=64,
            )
            self.assertNotEqual(
                _visual_digest((("grounded", first),)),
                _visual_digest((("grounded", second),)),
            )

    def test_real_visual_backend_receives_all_six_digest_bound_images(self) -> None:
        images = [
            Image.new("RGB", (8, 6), color=(index, index + 1, index + 2))
            for index in range(6)
        ]
        digest = _visual_digest(
            tuple(
                zip(
                    (
                        "exact_i0",
                        "source_mosaic",
                        "source_temporal_triptych",
                        "source_temporal_lr_zoom",
                        "source_motion_attention",
                        "source_grounded_temporal_zoom",
                    ),
                    images,
                )
            )
        )
        backend = _RecordingVisualBackend()
        raw, returned_digest = _generate_visual_pass(
            backend,
            custom_method="missing_custom_method",
            system="system",
            prompt="prompt",
            source_path=Path("source.mp4"),
            anchor_path=Path("anchor.png"),
            nframes=16,
            max_pixels=DEFAULT_MAX_PIXELS,
            tile_width=DEFAULT_TILE_WIDTH,
            mosaic_columns=DEFAULT_MOSAIC_COLUMNS,
            exact_i0=images[0],
            source_mosaic=images[1],
            source_temporal_triptych=images[2],
            source_temporal_lr_zoom=images[3],
            source_motion_attention=images[4],
            source_grounded_temporal_zoom=images[5],
            expected_visual_digest=digest,
        )
        self.assertEqual(raw, '{"ok":true}')
        self.assertEqual(returned_digest, digest)
        self.assertEqual(backend.processor.images, images)
        rendered_messages = json.dumps(
            backend.processor.messages, default=lambda _value: "IMAGE"
        )
        self.assertIn("temporal comparison C0 / CM / CF", rendered_messages)
        self.assertIn("LEFT / RIGHT temporal zoom", rendered_messages)
        self.assertIn("PIXEL-CHANGE ATTENTION", rendered_messages)
        self.assertIn("subject temporal rows", rendered_messages)

    def test_two_moving_people_override_single_actor_legacy_seed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root)
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            self.assertEqual(run_audit(args, backend_factory=_FakeBackend), 0)
            record = json.loads(output.read_text().strip())
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["pipeline_decision"], "pass")
            self.assertEqual(
                [item["unit_id"] for item in record["source_census"]["dynamic_units"]],
                ["unit_01", "unit_02"],
            )
            instruction = record["compiled_instruction"]["edit_instruction"]
            self.assertIn("blue-shirted man", instruction)
            self.assertIn("tattooed man", instruction)
            self.assertIn("gloved hand", instruction)
            self.assertIn("camera locked off", instruction)
            backend = _FakeBackend.instances[-1]
            self.assertEqual(
                [stage for stage, _ in backend.calls],
                [
                    "coverage_inventory",
                    "coverage_assignments",
                    "i0_grounding",
                    "source",
                    "secondary_source",
                    "target",
                    "critic",
                ],
            )
            self.assertNotEqual(
                backend.calls[3][1]["system"],
                backend.calls[4][1]["system"],
            )
            self.assertIn("blue-shirted man", backend.calls[4][1]["user"])
            self.assertNotIn(row["prompt"], backend.calls[4][1]["user"])
            self.assertNotIn(row["prompt"], backend.calls[0][1]["user"])
            self.assertNotIn(row["prompt"], backend.calls[1][1]["user"])
            self.assertNotIn(row["source_caption"], backend.calls[0][1]["user"])
            self.assertNotIn(row["edited_caption"], backend.calls[0][1]["user"])
            self.assertTrue(
                record["source_inventory_alignment"]["projections_equal"]
            )
            self.assertEqual(
                record["hard_gate"]["source_inventory_alignment_sha256"],
                record["source_inventory_alignment_digest"],
            )
            self.assertEqual(
                record["hard_gate"]["change_region_proposals_sha256"],
                record["change_region_proposals_digest"],
            )
            self.assertEqual(
                record["hard_gate"]["coverage_authority_sha256"],
                record["coverage_authority_digest"],
            )
            self.assertEqual(
                record["hard_gate"]["coverage_authority_inventory_sha256"],
                record["coverage_authority_inventory_digest"],
            )
            self.assertEqual(
                record["hard_gate"]["coverage_authority_assignments_sha256"],
                record["coverage_authority_assignments_digest"],
            )
            self.assertEqual(
                record["hard_gate"]["coverage_authority_alignment_sha256"],
                record["coverage_authority_alignment_digest"],
            )
            self.assertTrue(
                record["coverage_authority_alignment"][
                    "all_authority_entities_aligned"
                ]
            )
            for field in (
                "source_census_canonicalization",
                "secondary_source_census_canonicalization",
                "target_plan_canonicalization",
            ):
                receipt = record[field]
                self.assertFalse(receipt["semantic_repair"])
                self.assertEqual(receipt["changed_field_paths"], [])
                self.assertEqual(
                    record[f"{field}_digest"], contract.object_sha256(receipt)
                )
            self.assertEqual(
                record["hard_gate"]["source_census_canonicalization_sha256"],
                record["source_census_canonicalization_digest"],
            )
            self.assertEqual(
                record["hard_gate"][
                    "secondary_source_census_canonicalization_sha256"
                ],
                record["secondary_source_census_canonicalization_digest"],
            )
            self.assertEqual(
                record["hard_gate"]["target_plan_canonicalization_sha256"],
                record["target_plan_canonicalization_digest"],
            )
            self.assertEqual(record["generation"]["nframes"], 16)
            self.assertEqual(record["generation"]["tile_width"], 512)
            self.assertEqual(record["generation"]["mosaic_columns"], 4)
            self.assertEqual(
                record["generation"]["high_resolution_checkpoints"],
                [
                    "exact_i0_only_grounding",
                    "full_frame_temporal_triptych",
                    "overlapping_left_right_temporal_zoom",
                    "fixed_full_frame_4x4_f0_f20_f40_f60_f80_grid",
                    "fixed_bbox_subject_f0_f20_f40_f60_f80_zoom",
                ],
            )
            self.assertEqual(
                record["generation"]["motion_attention"],
                "deterministic_i0_to_midpoint_or_final_pixel_change",
            )
            self.assertEqual(
                record["generation"]["visual_input"],
                "blind_two_stage_coverage_authority_plus_i0_only_grounding_plus_"
                "dense_source_mosaic_plus_temporal_triptych_lr_zoom_motion_"
                "attention_authority_grid_and_grounded_actor_zoom",
            )
            self.assertEqual(record["generation"]["max_pixels"], 2_359_296)
            self.assertEqual(record["generation"]["max_new_tokens"], 6144)
            target_prompt = backend.calls[5][1]["user"]
            self.assertIn(row["prompt"], target_prompt)
            self.assertIn("untrusted", target_prompt)
            self.assertNotIn(row["source_caption"], target_prompt)
            self.assertNotIn(row["edited_caption"], target_prompt)
            self.assertTrue(shard_receipt_path(output).is_file())
            shard_receipt = json.loads(
                shard_receipt_path(output).read_text(encoding="utf-8")
            )
            self.assertIn(
                "qwen_filter",
                shard_receipt["run_config"]["implementation_bundle"],
            )
            self.assertEqual(shard_receipt["run_config"]["tile_width"], 512)
            self.assertEqual(
                shard_receipt["run_config"]["mosaic_columns"], 4
            )
            repair_policy = shard_receipt["run_config"][
                "schema_repair_policy"
            ]
            self.assertEqual(repair_policy["eligible_stages"], ["target_plan"])
            self.assertEqual(
                repair_policy["visual_modes"],
                {"target_plan": "text_only_no_visual_input"},
            )
            self.assertFalse(repair_policy["semantic_repair_allowed"])
            self.assertNotIn(
                "a0_schema_repair",
                shard_receipt["run_config"]["prompt_template_digests"],
            )
            self.assertEqual(
                shard_receipt["run_config"]["generation"][
                    "high_resolution_checkpoints"
                ],
                [
                    "exact_i0_only_grounding",
                    "full_frame_temporal_triptych",
                    "overlapping_left_right_temporal_zoom",
                    "fixed_full_frame_4x4_f0_f20_f40_f60_f80_grid",
                    "fixed_bbox_subject_f0_f20_f40_f60_f80_zoom",
                ],
            )
            self.assertEqual(
                backend.calls[0][1]["expected_visual_input_digest"],
                record["coverage_authority_inventory_visual_input_digest"],
            )
            self.assertEqual(
                backend.calls[1][1]["expected_visual_input_digest"],
                record["coverage_authority_assignments_visual_input_digest"],
            )
            self.assertNotEqual(
                record["coverage_authority_inventory_visual_input_digest"],
                record["coverage_authority_assignments_visual_input_digest"],
            )
            self.assertEqual(
                backend.calls[2][1]["expected_visual_input_digest"],
                record["i0_grounding_visual_input_digest"],
            )
            visual_digests = {
                call[1]["expected_visual_input_digest"]
                for call in backend.calls[3:]
            }
            self.assertEqual(visual_digests, {record["visual_input_digest"]})

    def test_a0a_long_evidence_rejects_without_any_model_repair(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="a0a-long-evidence-reject-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            args.allow_errors = True
            self.assertEqual(
                run_audit(
                    args,
                    backend_factory=_A0aLongCameraEvidenceRepairBackend,
                ),
                0,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")
            self.assertEqual(
                record["failure_stage"],
                "coverage_authority_inventory_validation",
            )
            self.assertIsNone(
                record["coverage_authority_inventory_validated_from"]
            )
            self.assertEqual(record["generation"]["schema_repair_attempts"], 0)
            self.assertNotIn("schema_repairs", record["generation"])
            self.assertIn(
                "description must be non-empty, trimmed, and at most 512 chars",
                record["error"],
            )
            self.assertEqual(
                len(
                    json.loads(record["coverage_authority_inventory_raw"])["camera"][
                        "motion_evidence"
                    ][0]["description"]
                ),
                588,
            )
            backend = _A0aLongCameraEvidenceRepairBackend.instances[-1]
            inventory_calls = [
                call for call in backend.calls if call[0] == "coverage_inventory"
            ]
            self.assertEqual(len(inventory_calls), 1)
            self.assertEqual(
                [stage for stage, _ in backend.calls],
                ["coverage_inventory"],
            )

    def test_a0b_uncovered_dynamic_camera_rejects_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="a0b-camera-reject-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            args.allow_errors = True
            self.assertEqual(
                run_audit(args, backend_factory=_A0bDynamicCameraRepairBackend),
                0,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")
            self.assertEqual(
                record["coverage_authority_inventory_validated_from"],
                "original",
            )
            self.assertIsNone(
                record["coverage_authority_assignments_validated_from"]
            )
            self.assertIn(
                "leaves dynamic camera uncovered", record["error"]
            )
            self.assertEqual(
                record["failure_stage"],
                "coverage_authority_assignments_validation",
            )
            self.assertEqual(record["generation"]["schema_repair_attempts"], 0)
            self.assertNotIn("schema_repairs", record["generation"])
            backend = _A0bDynamicCameraRepairBackend.instances[-1]
            assignment_calls = [
                call for call in backend.calls if call[0] == "coverage_assignments"
            ]
            self.assertEqual(len(assignment_calls), 1)

    def test_a0a_self_negated_head_label_is_canonicalized_without_repair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="a0a-self-negated-head-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            self.assertEqual(
                run_audit(
                    args, backend_factory=_A0aSelfNegatedHeadLabelBackend
                ),
                0,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "ok")
            self.assertEqual(
                record["coverage_authority_inventory_validated_from"],
                "canonicalized_original",
            )
            self.assertEqual(
                record["coverage_authority"]["inventory"]["i0_subjects"][0][
                    "motion_component_types"
                ],
                ["gesture"],
            )
            self.assertEqual(record["generation"]["schema_repair_attempts"], 0)
            self.assertNotIn("schema_repairs", record["generation"])

    def test_a0b_explicit_camera_enum_mismatch_is_canonicalized_without_repair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="a0b-camera-enum-mismatch-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            self.assertEqual(
                run_audit(
                    args, backend_factory=_A0bExplicitCameraEnumMismatchBackend
                ),
                0,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "ok")
            self.assertEqual(
                record["coverage_authority_assignments_validated_from"],
                "canonicalized_original",
            )
            self.assertGreaterEqual(
                sum(
                    assignment["assignment_kind"] == "camera"
                    for assignment in record["coverage_authority"]["assignments"][
                        "change_region_assignments"
                    ]
                ),
                1,
            )
            self.assertEqual(record["generation"]["schema_repair_attempts"], 0)
            self.assertNotIn("schema_repairs", record["generation"])

    def test_a0a_invalid_original_never_reaches_second_or_third_call(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="a0a-original-invalid-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            args.allow_errors = True
            self.assertEqual(
                run_audit(
                    args, backend_factory=_A0aRepairStillInvalidBackend
                ),
                0,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")
            self.assertEqual(
                record["failure_stage"],
                "coverage_authority_inventory_validation",
            )
            self.assertIsNone(
                record["coverage_authority_inventory_validated_from"]
            )
            self.assertEqual(record["generation"]["schema_repair_attempts"], 0)
            self.assertNotIn("schema_repairs", record["generation"])
            backend = _A0aRepairStillInvalidBackend.instances[-1]
            self.assertEqual(
                [
                    stage
                    for stage, _ in backend.calls
                    if stage == "coverage_inventory"
                ],
                ["coverage_inventory"],
            )

    def test_target_plan_missing_both_unit_ids_repairs_text_only_and_replays(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="target-plan-unit-id-repair-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            self.assertEqual(
                run_audit(
                    args,
                    backend_factory=_TargetPlanMissingUnitIdRepairBackend,
                ),
                0,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "ok")
            self.assertEqual(
                record["target_plan_validated_from"],
                "canonicalized_repair_1",
            )
            original = json.loads(record["target_plan_raw"])
            self.assertTrue(
                all(
                    "unit_id" not in target
                    for target in original["dynamic_unit_targets"]
                )
            )
            ledger = record["generation"]["schema_repairs"]
            transcript = ledger["target_plan"]
            self.assertEqual(transcript["outcome"], "valid")
            self.assertEqual(
                transcript["validated_from"], "canonicalized_repair_1"
            )
            self.assertEqual(transcript["original_raw"], record["target_plan_raw"])
            self.assertIsNone(transcript["repair_visual_input_digest"])
            self.assertEqual(
                transcript["source_census_digest"],
                record["source_census_digest"],
            )
            self.assertIn(
                "missing_nonredundant=['unit_id']",
                transcript["validator_error"],
            )
            repaired = json.loads(transcript["repair_raw"])
            self.assertEqual(
                [target["unit_id"] for target in repaired["dynamic_unit_targets"]],
                ["unit_01", "unit_02"],
            )
            self.assertEqual(
                json.loads(
                    target_plan_validated_raw(
                        record, source_census=record["source_census"]
                    )
                ),
                repaired,
            )
            malicious_record = json.loads(json.dumps(record))
            malicious_repair = json.loads(transcript["repair_raw"])
            malicious_repair["dynamic_unit_targets"][0][
                "novel_target_motion"
            ] = "raise the open hand overhead and trace one slow circle"
            malicious_record["generation"]["schema_repairs"]["target_plan"][
                "repair_raw"
            ] = json.dumps(malicious_repair)
            with self.assertRaisesRegex(
                GokuFullMotionQwenError,
                "changed fields beyond missing unit_id insertion",
            ):
                target_plan_validated_raw(
                    malicious_record,
                    source_census=malicious_record["source_census"],
                )

            # JSON number and boolean types must also remain exact.  Python's
            # ordinary dict equality would incorrectly accept 3.0 -> 3 and
            # True -> 1, even though those are distinct JSON encodings.
            for field, replacement in (
                ("completion_time_seconds", 3),
                ("substantive_change", 1),
            ):
                type_mutated_record = json.loads(json.dumps(record))
                type_mutated_repair = json.loads(transcript["repair_raw"])
                type_mutated_repair["dynamic_unit_targets"][0][field] = (
                    replacement
                )
                type_mutated_record["generation"]["schema_repairs"][
                    "target_plan"
                ]["repair_raw"] = json.dumps(type_mutated_repair)
                with self.assertRaisesRegex(
                    GokuFullMotionQwenError,
                    "changed fields beyond missing unit_id insertion",
                ):
                    target_plan_validated_raw(
                        type_mutated_record,
                        source_census=type_mutated_record["source_census"],
                    )
            bool_attempt_record = json.loads(json.dumps(record))
            bool_attempt_record["generation"]["schema_repairs"][
                "target_plan"
            ]["attempt"] = True
            with self.assertRaisesRegex(
                GokuFullMotionQwenError,
                "schema-repair transcript identity differs",
            ):
                target_plan_validated_raw(
                    bool_attempt_record,
                    source_census=bool_attempt_record["source_census"],
                )
            self.assertEqual(
                record["target_plan_canonicalization"]["raw_sha256"],
                contract.object_sha256(repaired),
            )
            backend = _TargetPlanMissingUnitIdRepairBackend.instances[-1]
            repair_calls = [
                call for call in backend.calls if call[0] == "target_schema_repair"
            ]
            self.assertEqual(len(repair_calls), 1)
            repair_kwargs = repair_calls[0][1]
            self.assertIsNone(repair_kwargs["expected_visual_input_digest"])
            self.assertEqual(
                repair_kwargs["system"], TARGET_PLAN_SCHEMA_REPAIR_SYSTEM
            )
            original_prompt = next(
                kwargs["user"] for stage, kwargs in backend.calls if stage == "target"
            )
            expected_repair_prompt = build_target_plan_schema_repair_prompt(
                original_prompt=original_prompt,
                original_raw=record["target_plan_raw"],
                validator_error=transcript["validator_error"],
                source_census_digest=record["source_census_digest"],
            )
            self.assertEqual(repair_kwargs["user"], expected_repair_prompt)
            for hidden in (
                "coverage_authority",
                "secondary_source_census",
                "coverage_critic",
                "i0_grounding",
            ):
                self.assertNotIn(hidden, expected_repair_prompt)

            # The downstream finalizer independently consumes the selected
            # repair raw rather than the rejected original PASS_B raw.
            from motive import goku_full_motion_finalize as finalizer

            semantic_fields = (
                "change_region_proposals",
                "coverage_authority",
                "i0_grounding",
                "source_census",
                "source_census_canonicalization",
                "secondary_source_census",
                "secondary_source_census_canonicalization",
                "source_inventory_alignment",
                "coverage_authority_alignment",
                "target_plan",
                "target_plan_canonicalization",
                "compiled_instruction",
                "full_motion_contract",
                "coverage_critic",
                "hard_gate",
            )
            semantic_objects = {field: record[field] for field in semantic_fields}
            evidence = dict(record)
            evidence["record_schema_version"] = record["schema_version"]
            generation_row = {
                "iid": record["iid"],
                "group_id": record["group_id"],
                "family": record["family"],
                "resolved_source_video": record["resolved_src_video"],
                "resolved_anchor_image": record["resolved_anchor_image"],
            }
            self.assertEqual(
                finalizer._validate_qwen_record_payload(
                    record,
                    row=generation_row,
                    evidence=evidence,
                    expected_result_payload=qwen_result_payload(record),
                    semantic_objects=semantic_objects,
                ),
                record,
            )
            malicious_record["provenance_digest"] = qwen_provenance_digest(
                malicious_record
            )
            malicious_evidence = dict(malicious_record)
            malicious_evidence["record_schema_version"] = malicious_record[
                "schema_version"
            ]
            with self.assertRaisesRegex(
                finalizer.GokuFullMotionFinalizeError,
                "changed fields beyond missing unit_id insertion",
            ):
                finalizer._validate_qwen_record_payload(
                    malicious_record,
                    row=generation_row,
                    evidence=malicious_evidence,
                    expected_result_payload=qwen_result_payload(record),
                    semantic_objects=semantic_objects,
                )

            shard_receipt_path(output).unlink()
            args.resume = True
            self.assertEqual(
                run_audit(
                    args,
                    backend_factory=_TargetPlanMissingUnitIdRepairBackend,
                ),
                0,
            )
            self.assertEqual(
                _TargetPlanMissingUnitIdRepairBackend.instances[-1].calls, []
            )

            # A self-consistent transcript/provenance forgery cannot replace
            # the exact canonicalization error replayed from the original raw.
            shard_receipt_path(output).unlink()
            forged = json.loads(output.read_text(encoding="utf-8"))
            forged_transcript = forged["generation"]["schema_repairs"][
                "target_plan"
            ]
            forged_transcript["validator_error"] += " FORGED"
            original_prompt = build_target_plan_prompt(
                row=row, source_census=forged["source_census"]
            )
            forged_prompt = build_target_plan_schema_repair_prompt(
                original_prompt=original_prompt,
                original_raw=forged_transcript["original_raw"],
                validator_error=forged_transcript["validator_error"],
                source_census_digest=forged["source_census_digest"],
            )
            forged_transcript["repair_prompt_digest"] = hashlib.sha256(
                (
                    TARGET_PLAN_SCHEMA_REPAIR_SYSTEM
                    + "\n"
                    + forged_prompt
                ).encode("utf-8")
            ).hexdigest()
            forged["provenance_digest"] = qwen_provenance_digest(forged)
            _write_jsonl(output, [forged])
            with self.assertRaisesRegex(
                GokuFullMotionQwenError,
                "target_plan original validator error does not replay",
            ):
                run_audit(
                    args,
                    backend_factory=_TargetPlanMissingUnitIdRepairBackend,
                )
            self.assertEqual(
                _TargetPlanMissingUnitIdRepairBackend.instances[-1].calls, []
            )

    def test_target_plan_repair_that_changes_action_is_rejected_once(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="target-plan-malicious-action-repair-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            args.allow_errors = True
            self.assertEqual(
                run_audit(
                    args,
                    backend_factory=_TargetPlanRepairMutatesActionBackend,
                ),
                0,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")
            self.assertEqual(
                record["failure_stage"],
                "target_plan_schema_repair_validation",
            )
            transcript = record["generation"]["schema_repairs"]["target_plan"]
            self.assertEqual(transcript["outcome"], "invalid")
            self.assertIn(
                "changed fields beyond missing unit_id insertion",
                transcript["repair_validation_error"],
            )
            original = json.loads(transcript["original_raw"])
            malicious = json.loads(transcript["repair_raw"])
            self.assertNotEqual(
                original["dynamic_unit_targets"][0]["novel_target_motion"],
                malicious["dynamic_unit_targets"][0]["novel_target_motion"],
            )
            backend = _TargetPlanRepairMutatesActionBackend.instances[-1]
            stages = [stage for stage, _ in backend.calls]
            self.assertEqual(stages.count("target"), 1)
            self.assertEqual(stages.count("target_schema_repair"), 1)
            self.assertNotIn("critic", stages)

    def test_target_plan_repair_still_missing_unit_ids_fails_once(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="target-plan-repair-invalid-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            args.allow_errors = True
            self.assertEqual(
                run_audit(
                    args,
                    backend_factory=_TargetPlanRepairStillMissingUnitIdBackend,
                ),
                0,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")
            self.assertEqual(
                record["failure_stage"],
                "target_plan_schema_repair_validation",
            )
            self.assertIsNone(record["target_plan_validated_from"])
            transcript = record["generation"]["schema_repairs"]["target_plan"]
            self.assertEqual(transcript["outcome"], "invalid")
            self.assertIsNone(transcript["repair_visual_input_digest"])
            self.assertIn(
                "changed fields beyond missing unit_id insertion",
                transcript["repair_validation_error"],
            )
            backend = _TargetPlanRepairStillMissingUnitIdBackend.instances[-1]
            self.assertEqual(
                [stage for stage, _ in backend.calls].count("target_schema_repair"),
                1,
            )
            self.assertNotIn("critic", [stage for stage, _ in backend.calls])

    def test_safe_redundancy_is_canonicalized_with_a_nonsemantic_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="redundancy-only-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            self.assertEqual(
                run_audit(args, backend_factory=_RedundantFieldDriftBackend), 0
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "ok")
            source_raw = json.loads(record["source_census_raw"])
            secondary_raw = json.loads(record["secondary_source_census_raw"])
            target_raw = json.loads(record["target_plan_raw"])
            self.assertNotIn("i0_visible_entities", source_raw)
            self.assertNotIn(
                "stable_reference", source_raw["dynamic_units"][0]
            )
            self.assertNotIn(
                "stable_reference", secondary_raw["dynamic_units"][1]
            )
            self.assertNotIn(
                "stable_reference", target_raw["dynamic_unit_targets"][0]
            )
            self.assertEqual(
                record["source_census"]["i0_visible_entities"],
                [
                    item["stable_reference"]
                    for item in record["source_census"]["i0_entity_registry"]
                ],
            )
            for field in (
                "source_census_canonicalization",
                "secondary_source_census_canonicalization",
                "target_plan_canonicalization",
            ):
                receipt = record[field]
                self.assertFalse(receipt["semantic_repair"])
                self.assertTrue(receipt["changed_field_paths"])
            compiled = record["compiled_instruction"]["edit_instruction"]
            self.assertIn("blue-shirted man", compiled)

    def test_secondary_census_actor_omission_fails_before_planning(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="secondary-omission-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            args.allow_errors = True
            self.assertEqual(
                run_audit(args, backend_factory=_SecondaryMissingRightBackend),
                0,
            )
            record = json.loads(output.read_text().strip())
            self.assertEqual(record["status"], "error")
            self.assertEqual(
                record["failure_stage"], "secondary_source_census_validation"
            )
            self.assertIn("registry differs from I0 grounding", record["error"])
            self.assertEqual(
                [stage for stage, _ in _FakeBackend.instances[-1].calls],
                [
                    "coverage_inventory",
                    "coverage_assignments",
                    "i0_grounding",
                    "source",
                    "secondary_source",
                ],
            )

    def test_coverage_authority_actor_omission_fails_before_planning(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="authority-omission-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            args.allow_errors = True
            self.assertEqual(
                run_audit(
                    args,
                    backend_factory=_CoverageAuthorityMissingRightBackend,
                ),
                0,
            )
            record = json.loads(output.read_text().strip())
            self.assertEqual(record["status"], "error")
            self.assertEqual(
                record["failure_stage"],
                "coverage_authority_assignments_validation",
            )
            self.assertIn("intersecting dynamic authority", record["error"])
            self.assertEqual(
                [stage for stage, _ in _FakeBackend.instances[-1].calls],
                [
                    "coverage_inventory",
                    "coverage_assignments",
                ],
            )

    def test_missing_right_person_target_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root)
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            args.allow_errors = True
            self.assertEqual(
                run_audit(args, backend_factory=_MissingRightBackend), 0
            )
            record = json.loads(output.read_text().strip())
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["failure_stage"], "target_plan_validation")
            self.assertIn("exactly one target", record["error"])

    def test_qwen_pipeline_compiles_noun_and_from_i0_standalone_prose(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="standalone-prose-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            self.assertEqual(
                run_audit(args, backend_factory=_StandaloneProseBackend), 0
            )
            record = json.loads(output.read_text().strip())
            compiled = record["compiled_instruction"]
            self.assertEqual(record["status"], "ok")
            self.assertEqual(
                compiled["entity_clauses"]["unit_01"],
                "Have the blue-shirted man on viewer-left perform this "
                "complete target motion: his bare hand rises from waist "
                "height, opens fully, and waves twice",
            )
            self.assertEqual(
                compiled["entity_clauses"]["unit_02"],
                "Have the tattooed man in a black sleeveless top on "
                "viewer-right perform this complete target motion: from I0, "
                "his gloved hand opens while rising and completes two "
                "side-to-side waves",
            )
            self.assertIn("Keep the camera locked off", compiled["edit_instruction"])
            self.assertEqual(
                compiled["instruction_sha256"],
                hashlib.sha256(
                    compiled["edit_instruction"].encode("utf-8")
                ).hexdigest(),
            )

    def test_resume_terminal_receipt_makes_no_visual_calls(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root)
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            self.assertEqual(run_audit(args, backend_factory=_FakeBackend), 0)
            args.resume = True
            self.assertEqual(run_audit(args, backend_factory=_FakeBackend), 0)
            self.assertEqual(_FakeBackend.instances[-1].calls, [])

    def test_resume_recomputes_raw_to_canonical_receipt_and_rejects_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = _input_row(root, iid="resume-canonicalization-001")
            input_path = root / "selected.jsonl"
            _write_jsonl(input_path, [row])
            output = root / "qwen.jsonl"
            args = _args(input_path, output, row)
            self.assertEqual(run_audit(args, backend_factory=_FakeBackend), 0)

            # Remove only the terminal wrapper so resume reaches the record's
            # own deterministic raw -> canonical -> receipt replay.
            shard_receipt_path(output).unlink()
            record = json.loads(output.read_text(encoding="utf-8"))
            record["source_census_canonicalization"]["semantic_repair"] = True
            _write_jsonl(output, [record])
            args.resume = True
            with self.assertRaisesRegex(
                contract.GokuFullMotionContractError,
                "canonicalization receipt differs",
            ):
                run_audit(args, backend_factory=_FakeBackend)
            self.assertEqual(_FakeBackend.instances[-1].calls, [])

    def test_prompt_schema_names_match_closed_contract(self) -> None:
        self.assertIn("i0_subjects", COVERAGE_AUTHORITY_INVENTORY_PROMPT_SCHEMA)
        self.assertIn(
            "extra_dynamic_entities", COVERAGE_AUTHORITY_INVENTORY_PROMPT_SCHEMA
        )
        self.assertIn("camera", COVERAGE_AUTHORITY_INVENTORY_PROMPT_SCHEMA)
        self.assertIn(
            "change_region_assignments",
            COVERAGE_AUTHORITY_ASSIGNMENTS_PROMPT_SCHEMA,
        )
        for field in (
            "all_i0_people_and_animals_enumerated",
            "all_dynamic_entities_enumerated",
        ):
            self.assertIn(field, COVERAGE_AUTHORITY_INVENTORY_PROMPT_SCHEMA)
        self.assertIn(
            "all_change_regions_resolved",
            COVERAGE_AUTHORITY_ASSIGNMENTS_PROMPT_SCHEMA,
        )
        self.assertIn(
            "allowed_owner_map_sha256",
            COVERAGE_AUTHORITY_ASSIGNMENTS_PROMPT_SCHEMA,
        )
        self.assertIn("motion_inventory_complete", SOURCE_CENSUS_PROMPT_SCHEMA)
        self.assertIn("dynamic_units", SOURCE_CENSUS_PROMPT_SCHEMA)
        self.assertIn("static_salient_people", SOURCE_CENSUS_PROMPT_SCHEMA)
        self.assertIn("i0_visible_entities", SOURCE_CENSUS_PROMPT_SCHEMA)
        self.assertIn("i0_entity_registry", SOURCE_CENSUS_PROMPT_SCHEMA)
        self.assertIn(
            "source_motion_components",
            SOURCE_CENSUS_PROMPT_SCHEMA["dynamic_units"][0],
        )
        self.assertIn("dynamic_unit_targets", TARGET_PLAN_PROMPT_SCHEMA)
        self.assertIn(
            "source_component_dispositions",
            TARGET_PLAN_PROMPT_SCHEMA["dynamic_unit_targets"][0],
        )
        self.assertIn("camera_target", TARGET_PLAN_PROMPT_SCHEMA)
        self.assertIn(
            "instruction_covered_dynamic_unit_ids",
            COVERAGE_CRITIC_PROMPT_SCHEMA,
        )

    def test_target_prompt_labels_single_actor_seed_untrusted(self) -> None:
        source = _source_census("two-people-wave-001")
        row = {"prompt": "Change only the left person's gesture."}
        prompt = build_target_plan_prompt(row=row, source_census=source)
        self.assertIn("untrusted_optional_legacy_action_seed", prompt)
        self.assertIn('["unit_01","unit_02"]', prompt)
        self.assertIn("dynamic_unit_targets", prompt)
        self.assertIn(
            "complete standalone prose", PASS_B_SYSTEM.replace("\n", " ")
        )
        self.assertIn("explicit label and colon", PASS_B_SYSTEM)
        self.assertIn("blue-shirted man", PASS_B_SYSTEM)
        self.assertIn("tattooed man", PASS_B_SYSTEM)
        self.assertIn("BOTH men", PASS_B_SYSTEM)
        self.assertIn("locked off", PASS_B_SYSTEM)
        pass_a_flat = PASS_A_SYSTEM.replace("\n", " ")
        self.assertIn("S0, the middle tiles S7/S8", pass_a_flat)
        self.assertIn("final tile S15", pass_a_flat)
        self.assertIn("empty component list", pass_a_flat)
        self.assertIn(
            "A component is present only when that semantic channel itself has positive temporal change",
            pass_a_flat,
        )
        self.assertIn(
            "wheel rotation implied by steady rolling",
            pass_a_flat,
        )
        pass_a2_flat = PASS_A2_SYSTEM.replace("\n", " ")
        self.assertIn(
            "Stable hands, fingers, torso, head, or gaze are not",
            pass_a2_flat,
        )
        self.assertIn(
            "wheel rotation implied by steady rolling into locomotion",
            pass_a2_flat,
        )
        self.assertIn(
            "Use each component_type at most once per unit",
            pass_a2_flat,
        )
        self.assertIn(
            "Put stable context in i0_state/source_motion",
            pass_a2_flat,
        )
        self.assertIn(
            "Use head_or_gaze, never articulation",
            pass_a_flat,
        )
        self.assertIn(
            "Use head_or_gaze, never articulation",
            pass_a2_flat,
        )
        self.assertIn(
            "Emit object_interaction for contact only when",
            pass_a2_flat,
        )
        self.assertIn(
            "a spatially static held/touched object is registry context",
            pass_a2_flat,
        )
        self.assertIn(
            "passive_interaction_object of entity_type vehicle",
            pass_a2_flat,
        )
        self.assertIn(
            "Every passive interaction object must be visibly reachable at exact I0",
            pass_a_flat,
        )
        self.assertIn(
            "source_action_signature must name positive temporal motion",
            pass_a_flat,
        )
        self.assertIn(
            "source_action_signature must name positive temporal motion",
            pass_a2_flat,
        )
        pass_b_flat = PASS_B_SYSTEM.replace("\n", " ")
        pass_c_flat = PASS_C_SYSTEM.replace("\n", " ")
        self.assertIn(
            "Never hide that base inside replace.novel_target_motion",
            pass_b_flat,
        )
        self.assertIn("the bike continues moving leftward", pass_b_flat)
        self.assertIn(
            "Reject a replace target that hides retained",
            pass_c_flat,
        )
        self.assertIn(
            'Never append a comparison such as "as in the source"',
            pass_b_flat,
        )
        self.assertIn(
            "translation and remaining in place are contradictory",
            pass_b_flat,
        )
        self.assertIn(
            'Reject every "as in the source"',
            pass_c_flat,
        )
        self.assertIn(
            'also described as happening "in place"',
            pass_c_flat,
        )

    def test_target_prompt_uses_grounded_crosscheck_and_qualitative_timing(
        self,
    ) -> None:
        source = _source_census("two-people-wave-001")
        prompt = build_target_plan_prompt(
            row={"prompt": "Change only the left person's gesture."},
            source_census=source,
        )
        target_hint = TARGET_PLAN_PROMPT_SCHEMA["dynamic_unit_targets"][0]
        camera_hint = TARGET_PLAN_PROMPT_SCHEMA["camera_target"]

        self.assertIn("unique actor paraphrase", target_hint["target_clause"])
        self.assertIn("does not need to repeat", PASS_B_SYSTEM)
        self.assertIn("stable_reference byte-for-byte", PASS_B_SYSTEM)
        self.assertIn("short, unique actor paraphrase", prompt)

        self.assertIn("completion_time_seconds is the only numeric", PASS_B_SYSTEM)
        self.assertIn("decoded frame numbers", PASS_B_SYSTEM)
        self.assertIn("self-converted frame-to-second times", PASS_B_SYSTEM)
        self.assertIn("sole numeric target time", prompt)
        self.assertIn("qualitative ordering only", target_hint["novel_target_motion"])
        self.assertIn(
            "qualitative ordering only",
            camera_hint["target_motion_description"],
        )

        self.assertNotIn(
            "literal stage beginning at I0", target_hint["ordered_stages"]
        )
        self.assertEqual(len(target_hint["ordered_stages"]), 3)
        self.assertEqual(len(camera_hint["ordered_stages"]), 2)
        self.assertIn("2 to 4 concrete", PASS_B_SYSTEM)
        self.assertIn("2 to 4 actual target", prompt)
        self.assertIn("Never copy", PASS_B_SYSTEM)
        self.assertIn('no ordered_stages entry may copy "replace with"', prompt)

    def test_direct_json_parser_rejects_markdown_and_duplicate_keys(self) -> None:
        with self.assertRaises(GokuFullMotionQwenError):
            _parse_direct_object("```json\n{}\n```", stage="test")
        with self.assertRaises(GokuFullMotionQwenError):
            _parse_direct_object('{"a":1,"a":2}', stage="test")

    def test_cli_freezes_eight_shards_and_zero_repairs(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["--input", "in.jsonl", "--output", "out.jsonl", "--model", "qwen"]
        )
        self.assertEqual(args.num_shards, QWEN3_LOGICAL_SHARDS)
        self.assertEqual(args.repair_attempts, 0)
        self.assertEqual(args.nframes, 16)
        self.assertEqual(args.tile_width, 512)
        self.assertEqual(args.mosaic_columns, 4)
        self.assertEqual(args.max_pixels, 2_359_296)
        self.assertEqual(args.max_new_tokens, 6144)
        repair_help = next(
            action.help
            for action in parser._actions
            if action.dest == "repair_attempts"
        )
        self.assertIn("A0a/A0b are original-only", repair_help)
        self.assertNotIn("visual schema retry", repair_help)

    def test_schema_repair_attempt_counts_require_json_integers(self) -> None:
        self.assertIsNone(
            _validate_schema_repair_ledger({"schema_repair_attempts": 0})
        )
        for malformed in (False, 0.0):
            with self.assertRaisesRegex(
                GokuFullMotionQwenError,
                "attempt count/ledger differ",
            ):
                _validate_schema_repair_ledger(
                    {"schema_repair_attempts": malformed}
                )

    def test_assignment_is_deterministic(self) -> None:
        rows = [{"iid": f"row-{index}"} for index in range(20)]
        shards = [
            assigned_iids_for_shard(
                rows, shard_index=index, num_shards=8, max_samples=None
            )
            for index in range(8)
        ]
        flattened = [iid for shard in shards for iid in shard]
        self.assertEqual(set(flattened), {row["iid"] for row in rows})
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == "__main__":
    unittest.main()
