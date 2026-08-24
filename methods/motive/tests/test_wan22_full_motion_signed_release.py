from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from methods.motive.tests.test_goku_full_motion_postcheck import (
    _critic,
    _plan,
    _source,
)
from motive import goku_full_motion_contract as contract
from motive import goku_full_motion_instruction as instruction
from motive import goku_full_motion_qwen as qwen
from motive import wan22_full_motion_signed_release as release
from motive import wan22_i2v_batch as batch


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))


def _write_media(root: Path) -> tuple[Path, Path]:
    source_path = root / "source.avi"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (64, 48),
    )
    if not writer.isOpened():  # pragma: no cover
        raise RuntimeError("test OpenCV cannot create MJPG")
    for frame_index in range(81):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = 90
        cv2.rectangle(
            frame,
            (6 + frame_index // 8, 10),
            (24 + frame_index // 8, 42),
            (190, 80, 30),
            -1,
        )
        cv2.circle(frame, (51, 35), 7, (60, 60, 220), -1)
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(source_path))
    ok, first = capture.read()
    capture.release()
    if not ok:  # pragma: no cover
        raise RuntimeError("test OpenCV cannot decode frame zero")
    anchor_path = root / "anchor.png"
    Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB)).save(anchor_path)
    return source_path.resolve(), anchor_path.resolve()


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
    """Build a closed blind-coverage fixture bound to this source census."""

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
    registry = {
        item["entity_id"]: item for item in source["i0_entity_registry"]
    }
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
    allowed_owner_map = qwen.build_coverage_authority_allowed_owner_map(
        coverage_authority_inventory=inventory,
        change_region_proposals=proposals,
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
                allowed_owner_map
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


def _row(
    *, index: int, source_path: Path, anchor_path: Path
) -> dict:
    iid = f"full-motion-{index:03d}"
    source = _source(iid)
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
    plan = _plan(source)
    compiled = instruction.compile_full_motion_instruction(source, plan)
    critic = _critic(source, plan, compiled)
    full_contract = contract.build_contract(
        source_census=source, target_plan=plan
    )
    canonical_source, source_canonicalization = (
        contract.canonicalize_source_census_model_output(source, iid)
    )
    canonical_secondary, secondary_canonicalization = (
        contract.canonicalize_source_census_model_output(
            secondary_source, iid
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
    receipt_digest = _sha(f"receipt-{index % 8}".encode())
    media_verification = {
        "exact_i0": True,
        "source_video_sha256": _sha(source_path.read_bytes()),
        "anchor_sha256": _sha(anchor_path.read_bytes()),
    }
    visual_digest = _sha(f"visual-{iid}".encode())
    qwen_record = {
        key: None for key in qwen._RECORD_KEYS
    }
    qwen_record.update(
        {
            "schema_version": qwen.RECORD_SCHEMA,
            "iid": iid,
            "group_id": f"group-{index:03d}",
            "family": "motion_editing",
            "status": "ok",
            "error_type": None,
            "error": None,
            "input_digest": _sha(f"input-{iid}".encode()),
            "config_digest": _sha(b"config"),
            "run_config_digest": _sha(b"run-config"),
            "implementation_digest": _sha(b"implementation"),
            "model_path": "/models/Qwen3-VL-32B-Instruct",
            "model_revision": "test-revision",
            "transformers_version": "5.5.4",
            "shard_index": index % 8,
            "num_shards": 8,
            "execution_manifest": str(source_path.parent / "candidates.jsonl"),
            "execution_manifest_sha256": _sha(b"fixture candidates"),
            "generation": {
                "do_sample": False,
                "max_new_tokens": qwen.DEFAULT_MAX_NEW_TOKENS,
                "schema_repair_attempts": 0,
                "visual_input": (
                    "blind_coverage_authority_plus_i0_only_grounding_plus_"
                    "dense_source_mosaic_plus_temporal_triptych_lr_zoom_motion_"
                    "attention_authority_grid_and_grounded_actor_zoom"
                ),
                "nframes": qwen.DEFAULT_NFRAMES,
                "max_pixels": qwen.DEFAULT_MAX_PIXELS,
                "tile_width": qwen.DEFAULT_TILE_WIDTH,
                "mosaic_columns": qwen.DEFAULT_MOSAIC_COLUMNS,
                "high_resolution_checkpoints": [
                    "exact_i0_only_grounding",
                    "full_frame_temporal_triptych",
                    "overlapping_left_right_temporal_zoom",
                    "fixed_full_frame_4x4_f0_f20_f40_f60_f80_grid",
                    "fixed_bbox_subject_f0_f20_f40_f60_f80_zoom",
                ],
                "motion_attention": (
                    "deterministic_i0_to_midpoint_or_final_pixel_change"
                ),
            },
            "failure_stage": None,
            "pipeline_stage": "coverage_critic",
            "pipeline_decision": "pass",
            "resolved_src_video": str(source_path),
            "resolved_anchor_image": str(anchor_path),
            "media_verification": media_verification,
            "visual_input_digest": visual_digest,
            "legacy_seed": {
                "role": "untrusted_optional_legacy_action_seed",
                "text": "fixture motion-edit seed",
                "sha256": _sha(b"fixture motion-edit seed"),
                "authoritative": False,
                "source_caption_used": False,
                "edited_caption_used": False,
                "old_target_video_used": False,
            },
            "change_region_proposals": change_region_proposals,
            "change_region_proposals_digest": contract.object_sha256(
                change_region_proposals
            ),
            "coverage_authority_inventory_prompt_digest": _sha(
                f"authority-inventory-prompt-{iid}".encode()
            ),
            "coverage_authority_inventory_visual_input_digest": _sha(
                f"authority-inventory-visual-{iid}".encode()
            ),
            "coverage_authority_inventory_raw": _canonical(
                coverage_authority["inventory"]
            ).decode(),
            "coverage_authority_inventory_validated_from": "original",
            "coverage_authority_inventory_digest": contract.object_sha256(
                coverage_authority["inventory"]
            ),
            "coverage_authority_assignments_prompt_digest": _sha(
                f"authority-assignments-prompt-{iid}".encode()
            ),
            "coverage_authority_assignments_visual_input_digest": _sha(
                f"authority-assignments-visual-{iid}".encode()
            ),
            "coverage_authority_assignments_raw": _canonical(
                coverage_authority["assignments"]
            ).decode(),
            "coverage_authority_assignments_validated_from": "original",
            "coverage_authority_assignments_digest": contract.object_sha256(
                coverage_authority["assignments"]
            ),
            "coverage_authority": coverage_authority,
            "coverage_authority_digest": contract.object_sha256(
                coverage_authority
            ),
            "i0_grounding_prompt_digest": _sha(
                f"grounding-prompt-{iid}".encode()
            ),
            "i0_grounding_visual_input_digest": _sha(
                f"grounding-visual-{iid}".encode()
            ),
            "i0_grounding_raw": _canonical(i0_grounding).decode(),
            "i0_grounding_validated_from": "original",
            "i0_grounding": i0_grounding,
            "i0_grounding_digest": contract.object_sha256(i0_grounding),
            "source_census_prompt_digest": _sha(
                f"source-prompt-{iid}".encode()
            ),
            "source_census_raw": _canonical(source).decode(),
            "source_census_validated_from": "canonicalized_original",
            "source_census": source,
            "source_census_digest": contract.object_sha256(source),
            "source_census_canonicalization": source_canonicalization,
            "source_census_canonicalization_digest": (
                source_canonicalization_sha
            ),
            "secondary_source_census_prompt_digest": _sha(
                f"secondary-prompt-{iid}".encode()
            ),
            "secondary_source_census_visual_input_digest": visual_digest,
            "secondary_source_census_raw": _canonical(
                secondary_source
            ).decode(),
            "secondary_source_census_validated_from": (
                "canonicalized_original"
            ),
            "secondary_source_census": secondary_source,
            "secondary_source_census_digest": contract.object_sha256(
                secondary_source
            ),
            "secondary_source_census_canonicalization": (
                secondary_canonicalization
            ),
            "secondary_source_census_canonicalization_digest": (
                secondary_canonicalization_sha
            ),
            "source_inventory_alignment": source_inventory_alignment,
            "source_inventory_alignment_digest": contract.object_sha256(
                source_inventory_alignment
            ),
            "coverage_authority_alignment": coverage_authority_alignment,
            "coverage_authority_alignment_digest": contract.object_sha256(
                coverage_authority_alignment
            ),
            "target_plan_prompt_digest": _sha(
                f"target-prompt-{iid}".encode()
            ),
            "target_plan_visual_input_digest": visual_digest,
            "target_plan_raw": _canonical(plan).decode(),
            "target_plan_validated_from": "canonicalized_original",
            "target_plan": plan,
            "target_plan_digest": contract.object_sha256(plan),
            "target_plan_canonicalization": target_canonicalization,
            "target_plan_canonicalization_digest": target_canonicalization_sha,
            "compiled_instruction": compiled,
            "compiled_instruction_digest": contract.object_sha256(compiled),
            "full_motion_contract": full_contract,
            "full_motion_contract_digest": contract.object_sha256(
                full_contract
            ),
            "coverage_critic_prompt_digest": _sha(
                f"critic-prompt-{iid}".encode()
            ),
            "coverage_critic_visual_input_digest": visual_digest,
            "coverage_critic_raw": _canonical(critic).decode(),
            "coverage_critic_validated_from": "original",
            "coverage_critic": critic,
            "coverage_critic_digest": contract.object_sha256(critic),
            "hard_gate": hard_gate,
        }
    )
    qwen_result = contract.object_sha256(
        qwen.qwen_result_payload(qwen_record)
    )
    qwen_record["result_digest"] = qwen_result
    qwen_provenance = qwen.qwen_provenance_digest(qwen_record)
    qwen_record["provenance_digest"] = qwen_provenance
    motion_spec = {
        "schema_version": release.MOTION_SPEC_SCHEMA,
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
        "full_motion_contract": full_contract,
        "qwen_result_digest": qwen_result,
        "qwen_provenance_digest": qwen_provenance,
    }
    media = {
        "width": 64,
        "height": 48,
        "frame_count": 81,
        "fps": 25.0,
        "duration_seconds": 3.24,
    }
    qwen_evidence = {
        "schema_version": release.QWEN_EVIDENCE_SCHEMA,
        "record_schema_version": qwen.RECORD_SCHEMA,
        "input_digest": _sha(f"input-{iid}".encode()),
        "result_digest": qwen_result,
        "provenance_digest": qwen_provenance,
        "config_digest": _sha(b"config"),
        "run_config_digest": _sha(b"run-config"),
        "implementation_digest": _sha(b"implementation"),
        "visual_input_digest": _sha(f"visual-{iid}".encode()),
        "media_verification": media_verification,
        "hard_gate": hard_gate,
        "change_region_proposals_digest": contract.object_sha256(
            change_region_proposals
        ),
        "coverage_authority_inventory_prompt_digest": qwen_record[
            "coverage_authority_inventory_prompt_digest"
        ],
        "coverage_authority_inventory_visual_input_digest": qwen_record[
            "coverage_authority_inventory_visual_input_digest"
        ],
        "coverage_authority_inventory_digest": contract.object_sha256(
            coverage_authority["inventory"]
        ),
        "coverage_authority_assignments_prompt_digest": qwen_record[
            "coverage_authority_assignments_prompt_digest"
        ],
        "coverage_authority_assignments_visual_input_digest": qwen_record[
            "coverage_authority_assignments_visual_input_digest"
        ],
        "coverage_authority_assignments_digest": contract.object_sha256(
            coverage_authority["assignments"]
        ),
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
        "source_census_digest": contract.object_sha256(source),
        "secondary_source_census_canonicalization": (
            secondary_canonicalization
        ),
        "secondary_source_census_canonicalization_digest": (
            secondary_canonicalization_sha
        ),
        "secondary_source_census_digest": contract.object_sha256(
            secondary_source
        ),
        "source_inventory_alignment_digest": contract.object_sha256(
            source_inventory_alignment
        ),
        "target_plan_canonicalization": target_canonicalization,
        "target_plan_canonicalization_digest": target_canonicalization_sha,
        "target_plan_digest": contract.object_sha256(plan),
        "compiled_instruction_digest": contract.object_sha256(compiled),
        "full_motion_contract_digest": contract.object_sha256(full_contract),
        "coverage_critic_digest": contract.object_sha256(critic),
        "shard_index": index % 8,
        "num_shards": 8,
        "receipt_digest": receipt_digest,
        "receipt_sha256": _sha(f"receipt-file-{index % 8}".encode()),
        "output_sha256": _sha(f"output-{index % 8}".encode()),
        "model_path": "/models/Qwen3-VL-32B-Instruct",
        "model_revision": "test-revision",
        "transformers_version": "5.5.4",
        "qwen_record_payload": qwen_record,
    }
    return {
        "schema_version": release.GENERATION_MANIFEST_SCHEMA,
        "iid": iid,
        "group_id": f"group-{index:03d}",
        "family": "motion_editing",
        "source_video": str(source_path),
        "resolved_source_video": str(source_path),
        "anchor_image": str(anchor_path),
        "resolved_anchor_image": str(anchor_path),
        "source_video_sha256": _sha(source_path.read_bytes()),
        "anchor_sha256": _sha(anchor_path.read_bytes()),
        "selected_media_evidence": media,
        "selected_media_evidence_sha256": contract.object_sha256(media),
        "strict_temporal_geometry": {
            "schema_version": release.TEMPORAL_GEOMETRY_SCHEMA,
            "source_frame_count": 81,
            "source_frame_rate": "25/1",
            "source_timeline_span_seconds": 3.2,
            "target_frame_count": 81,
            "target_frame_rate": "25/1",
            "target_timeline_span_seconds": 3.2,
            "requires_exact_frame_count_and_rate_match": True,
        },
        "edit_instruction": compiled["edit_instruction"],
        "edit_instruction_sha256": compiled["instruction_sha256"],
        "motion_spec": motion_spec,
        "motion_spec_sha256": contract.object_sha256(motion_spec),
        "qwen_evidence": qwen_evidence,
        "full_motion_finalization": {
            "schema_version": release.FINALIZATION_ROW_SCHEMA,
            "policy_version": release.FINALIZATION_POLICY,
            "candidate_rank": index + 1,
            "review_rank": index + 1,
            "selection_bucket": "primary",
            "dynamic_unit_count": 1,
            "target_action_signatures": [
                plan["dynamic_unit_targets"][0]["target_action_signature"]
            ],
            "family": "motion_editing",
            "required_canary": index == 0,
            "qwen_shard_index": index % 8,
            "qwen_receipt_digest": receipt_digest,
        },
        "action_change_substantive": "yes",
        "manifest_role": "review_proposal",
        "human_review_status": "pending",
        "generation_authorized": False,
        "production_eligible": False,
        "approval": None,
        "authorization_interface_available": False,
        "annotation_source": "qwen3-vl-32b",
        "human_reviewed": False,
    }


def _key(root: Path) -> tuple[Path, str, str]:
    key = root / "release_key"
    generated = subprocess.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(key),
        ],
        capture_output=True,
        check=False,
    )
    if generated.returncode != 0:  # pragma: no cover
        raise RuntimeError(generated.stderr.decode())
    public = " ".join((root / "release_key.pub").read_text().split()[:2])
    fingerprint_result = subprocess.run(
        ["ssh-keygen", "-lf", str(root / "release_key.pub")],
        capture_output=True,
        text=True,
        check=True,
    )
    fingerprint = fingerprint_result.stdout.split()[1]
    return key, public, fingerprint


class Wan22FullMotionSignedReleaseTests(unittest.TestCase):
    def _fixture(self, root: Path):
        source, anchor = _write_media(root)
        rows = [
            _row(index=index, source_path=source, anchor_path=anchor)
            for index in range(16)
        ]
        root_manifest = root / "root_manifest.jsonl"
        _write_jsonl(root_manifest, rows)
        key, public, fingerprint = _key(root)
        signed = root / "full_motion_release.json"
        with (
            mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
            mock.patch.object(
                release, "SIGNER_KEY_FINGERPRINT", fingerprint
            ),
        ):
            release.build_and_sign_release(
                root_manifest_path=root_manifest,
                output_path=signed,
                signing_key=key,
                release_id="unit-test-root-16",
                issued_at_utc="2026-08-01T00:00:00+00:00",
            )
        return rows, signed, public, fingerprint

    def _verify(
        self,
        *,
        signed: Path,
        manifest: Path,
        public: str,
        fingerprint: str,
    ):
        with (
            mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
            mock.patch.object(
                release, "SIGNER_KEY_FINGERPRINT", fingerprint
            ),
        ):
            return release.verify_signed_release(
                release_path=signed, manifest_path=manifest
            )

    def test_release_recomputes_v6_authority_and_receipt_closure(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, anchor = _write_media(root)
            rows = [
                _row(index=index, source_path=source, anchor_path=anchor)
                for index in range(8)
            ]
            manifest = root / "root.jsonl"
            _write_jsonl(manifest, rows)
            payload = release.build_release_payload(
                root_manifest_path=manifest,
                release_id="v6-coverage-authority-closure",
                issued_at_utc="2026-08-01T00:00:00+00:00",
                verify_media=False,
            )
            self.assertEqual(
                payload["schema_version"],
                "motive-wan22-full-motion-root-release-payload-v3",
            )
            old_payload = copy.deepcopy(payload)
            old_payload["schema_version"] = (
                "motive-wan22-full-motion-root-release-payload-v2"
            )
            with self.assertRaises(release.Wan22FullMotionReleaseError):
                release._validate_payload_shape(old_payload)
            changed_payload_receipt = copy.deepcopy(payload)
            changed_payload_receipt["row_authorizations"][0][
                "target_plan_canonicalization"
            ]["raw_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                release.Wan22FullMotionReleaseError,
                "target_plan_canonicalization digest differs",
            ):
                release._validate_payload_shape(changed_payload_receipt)
            for row, authorization in zip(
                rows, payload["row_authorizations"], strict=True
            ):
                evidence = row["qwen_evidence"]
                self.assertEqual(
                    authorization["primary_source_census_sha256"],
                    evidence["source_census_digest"],
                )
                self.assertEqual(
                    authorization["secondary_source_census_sha256"],
                    evidence["secondary_source_census_digest"],
                )
                self.assertEqual(
                    authorization["source_inventory_alignment_sha256"],
                    evidence["source_inventory_alignment_digest"],
                )
                self.assertEqual(
                    authorization["change_region_proposals_sha256"],
                    evidence["change_region_proposals_digest"],
                )
                self.assertEqual(
                    authorization["coverage_authority_sha256"],
                    evidence["coverage_authority_digest"],
                )
                self.assertEqual(
                    authorization["coverage_authority_inventory_sha256"],
                    evidence["coverage_authority_inventory_digest"],
                )
                self.assertEqual(
                    authorization["coverage_authority_assignments_sha256"],
                    evidence["coverage_authority_assignments_digest"],
                )
                self.assertEqual(
                    authorization["coverage_authority_alignment_sha256"],
                    evidence["coverage_authority_alignment_digest"],
                )
                self.assertEqual(
                    authorization["qwen_hard_gate_sha256"],
                    contract.object_sha256(evidence["hard_gate"]),
                )
                self.assertEqual(
                    evidence["i0_grounding_digest"],
                    contract.object_sha256(row["motion_spec"]["i0_grounding"]),
                )
                self.assertEqual(
                    evidence["hard_gate"]["i0_grounding_sha256"],
                    evidence["i0_grounding_digest"],
                )
                for field in (
                    "source_census_canonicalization",
                    "secondary_source_census_canonicalization",
                    "target_plan_canonicalization",
                ):
                    self.assertEqual(
                        authorization[field],
                        evidence[field],
                    )
                    self.assertEqual(
                        authorization[f"{field}_sha256"],
                        evidence[f"{field}_digest"],
                    )

            mutations: list[list[dict]] = []
            old_generation = copy.deepcopy(rows)
            old_generation[0]["schema_version"] = (
                "motive-goku-full-motion-generation-v5"
            )
            mutations.append(old_generation)

            old_evidence = copy.deepcopy(rows)
            old_evidence[0]["qwen_evidence"]["schema_version"] = (
                "motive-goku-full-motion-qwen-evidence-v5"
            )
            mutations.append(old_evidence)

            old_record = copy.deepcopy(rows)
            old_record[0]["qwen_evidence"]["record_schema_version"] = (
                "goku-full-motion-qwen-record-v5"
            )
            mutations.append(old_record)

            old_gate = copy.deepcopy(rows)
            old_gate[0]["qwen_evidence"]["hard_gate"] = {
                "schema_version": "goku-full-motion-hard-gate-v5",
                "source_census_sha256": old_gate[0]["qwen_evidence"][
                    "source_census_digest"
                ],
                "secondary_source_census_sha256": old_gate[0][
                    "qwen_evidence"
                ]["secondary_source_census_digest"],
                "source_inventory_alignment_sha256": old_gate[0][
                    "qwen_evidence"
                ]["source_inventory_alignment_digest"],
                "decision": "pass",
                "risk_codes": [],
            }
            mutations.append(old_gate)

            old_spec = copy.deepcopy(rows)
            old_spec[0]["motion_spec"]["schema_version"] = (
                "motive-goku-full-motion-generation-spec-v5"
            )
            old_spec[0]["motion_spec_sha256"] = contract.object_sha256(
                old_spec[0]["motion_spec"]
            )
            mutations.append(old_spec)

            wrong_secondary = copy.deepcopy(rows)
            wrong_secondary[0]["qwen_evidence"][
                "secondary_source_census_digest"
            ] = "0" * 64
            mutations.append(wrong_secondary)

            wrong_inventory_digest = copy.deepcopy(rows)
            wrong_inventory_digest[0]["qwen_evidence"][
                "coverage_authority_inventory_digest"
            ] = "0" * 64
            mutations.append(wrong_inventory_digest)

            tampered_assignments_raw = copy.deepcopy(rows)
            tampered_row = tampered_assignments_raw[0]
            tampered_record = tampered_row["qwen_evidence"][
                "qwen_record_payload"
            ]
            raw_assignments = json.loads(
                tampered_record["coverage_authority_assignments_raw"]
            )
            raw_assignments["change_region_assignments"][0][
                "resolution_reason"
            ] = "The same grid cell has a different raw-model explanation"
            tampered_record["coverage_authority_assignments_raw"] = (
                _canonical(raw_assignments).decode()
            )
            tampered_provenance = qwen.qwen_provenance_digest(
                tampered_record
            )
            tampered_record["provenance_digest"] = tampered_provenance
            tampered_row["qwen_evidence"]["provenance_digest"] = (
                tampered_provenance
            )
            tampered_row["motion_spec"]["qwen_provenance_digest"] = (
                tampered_provenance
            )
            tampered_row["motion_spec_sha256"] = contract.object_sha256(
                tampered_row["motion_spec"]
            )
            mutations.append(tampered_assignments_raw)

            wrong_receipt = copy.deepcopy(rows)
            wrong_receipt[0]["qwen_evidence"][
                "source_census_canonicalization"
            ]["receipt_sha256"] = "0" * 64
            mutations.append(wrong_receipt)

            wrong_receipt_digest = copy.deepcopy(rows)
            wrong_receipt_digest[0]["qwen_evidence"][
                "target_plan_canonicalization_digest"
            ] = "0" * 64
            mutations.append(wrong_receipt_digest)

            changed_grounding = copy.deepcopy(rows)
            changed_grounding[0]["motion_spec"]["i0_grounding"]["subjects"][0][
                "viewer_left_extremity_state"
            ] = "viewer-left extremity stays below the waist at I0"
            changed_grounding[0]["motion_spec_sha256"] = contract.object_sha256(
                changed_grounding[0]["motion_spec"]
            )
            mutations.append(changed_grounding)

            wrong_grounding_digest = copy.deepcopy(rows)
            wrong_grounding_digest[0]["qwen_evidence"][
                "i0_grounding_digest"
            ] = "0" * 64
            mutations.append(wrong_grounding_digest)

            wrong_gate_grounding_digest = copy.deepcopy(rows)
            wrong_gate_grounding_digest[0]["qwen_evidence"]["hard_gate"][
                "i0_grounding_sha256"
            ] = "0" * 64
            mutations.append(wrong_gate_grounding_digest)

            forged_result = copy.deepcopy(rows)
            forged_result_sha = "a" * 64
            forged_result[0]["motion_spec"]["qwen_result_digest"] = (
                forged_result_sha
            )
            forged_result[0]["qwen_evidence"]["result_digest"] = (
                forged_result_sha
            )
            forged_record = forged_result[0]["qwen_evidence"][
                "qwen_record_payload"
            ]
            forged_record["result_digest"] = forged_result_sha
            forged_provenance = qwen.qwen_provenance_digest(forged_record)
            forged_record["provenance_digest"] = forged_provenance
            forged_result[0]["qwen_evidence"]["provenance_digest"] = (
                forged_provenance
            )
            forged_result[0]["motion_spec"]["qwen_provenance_digest"] = (
                forged_provenance
            )
            forged_result[0]["motion_spec_sha256"] = contract.object_sha256(
                forged_result[0]["motion_spec"]
            )
            mutations.append(forged_result)

            forged_provenance_rows = copy.deepcopy(rows)
            forged_provenance_rows[0]["motion_spec"][
                "qwen_provenance_digest"
            ] = "b" * 64
            forged_provenance_rows[0]["qwen_evidence"][
                "provenance_digest"
            ] = "b" * 64
            forged_provenance_rows[0]["qwen_evidence"][
                "qwen_record_payload"
            ]["provenance_digest"] = "b" * 64
            forged_provenance_rows[0]["motion_spec_sha256"] = (
                contract.object_sha256(
                    forged_provenance_rows[0]["motion_spec"]
                )
            )
            mutations.append(forged_provenance_rows)

            redigested_alignment = copy.deepcopy(rows)
            changed_alignment = redigested_alignment[0]["motion_spec"][
                "source_inventory_alignment"
            ]
            changed_alignment["projections_equal"] = False
            changed_alignment_sha = contract.object_sha256(changed_alignment)
            redigested_alignment[0]["motion_spec_sha256"] = (
                contract.object_sha256(redigested_alignment[0]["motion_spec"])
            )
            redigested_alignment[0]["qwen_evidence"][
                "source_inventory_alignment_digest"
            ] = changed_alignment_sha
            redigested_alignment[0]["qwen_evidence"]["hard_gate"][
                "source_inventory_alignment_sha256"
            ] = changed_alignment_sha
            mutations.append(redigested_alignment)

            shadow_authority = copy.deepcopy(rows)
            shadow_authority[0]["motion_spec"][
                "coverage_authority_shadow"
            ] = shadow_authority[0]["motion_spec"]["coverage_authority"]
            shadow_authority[0]["motion_spec_sha256"] = contract.object_sha256(
                shadow_authority[0]["motion_spec"]
            )
            mutations.append(shadow_authority)

            redigested_proposal = copy.deepcopy(rows)
            proposal = redigested_proposal[0]["motion_spec"][
                "change_region_proposals"
            ]
            proposal["regions"][0]["changed_pixel_count"] += 1
            redigested_proposal[0]["motion_spec_sha256"] = (
                contract.object_sha256(redigested_proposal[0]["motion_spec"])
            )
            redigested_proposal[0]["qwen_evidence"][
                "change_region_proposals_digest"
            ] = contract.object_sha256(proposal)
            redigested_proposal[0]["qwen_evidence"]["hard_gate"][
                "change_region_proposals_sha256"
            ] = contract.object_sha256(proposal)
            mutations.append(redigested_proposal)

            redigested_authority_alignment = copy.deepcopy(rows)
            authority_alignment = redigested_authority_alignment[0][
                "motion_spec"
            ]["coverage_authority_alignment"]
            authority_alignment["all_authority_entities_aligned"] = False
            authority_alignment_sha = contract.object_sha256(
                authority_alignment
            )
            redigested_authority_alignment[0]["motion_spec_sha256"] = (
                contract.object_sha256(
                    redigested_authority_alignment[0]["motion_spec"]
                )
            )
            redigested_authority_alignment[0]["qwen_evidence"][
                "coverage_authority_alignment_digest"
            ] = authority_alignment_sha
            redigested_authority_alignment[0]["qwen_evidence"]["hard_gate"][
                "coverage_authority_alignment_sha256"
            ] = authority_alignment_sha
            mutations.append(redigested_authority_alignment)

            semantic_restatement = copy.deepcopy(rows)
            target = semantic_restatement[0]["motion_spec"]["target_plan"][
                "dynamic_unit_targets"
            ][0]
            target["novel_target_motion"] = "walk steadily forward"
            target["ordered_stages"] = ["walk steadily forward"]
            target["target_clause"] = (
                "have the walking person on the left walk steadily forward"
            )
            semantic_restatement[0]["motion_spec_sha256"] = (
                contract.object_sha256(semantic_restatement[0]["motion_spec"])
            )
            mutations.append(semantic_restatement)

            for index, mutated in enumerate(mutations):
                invalid = root / f"invalid_{index}.jsonl"
                _write_jsonl(invalid, mutated)
                with self.assertRaises(release.Wan22FullMotionReleaseError):
                    release.build_release_payload(
                        root_manifest_path=invalid,
                        release_id=f"invalid-{index}",
                        issued_at_utc="2026-08-01T00:00:00+00:00",
                        verify_media=False,
                    )

    def test_one_root_release_authorizes_two_contiguous_eight_row_shards(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows, signed, public, fingerprint = self._fixture(root)
            envelope = json.loads(signed.read_text())
            self.assertEqual(
                envelope["schema_version"],
                "motive-wan22-full-motion-signed-root-release-v3",
            )
            self.assertEqual(
                envelope["signature"]["namespace"],
                "motive-wan22-full-motion-root-release-v3",
            )
            legacy = root / "legacy_v1_release.json"
            legacy_envelope = copy.deepcopy(envelope)
            legacy_envelope["schema_version"] = (
                "motive-wan22-full-motion-signed-root-release-v1"
            )
            legacy.write_text(json.dumps(legacy_envelope) + "\n")
            legacy.chmod(0o400)
            legacy_manifest = root / "legacy_shard.jsonl"
            _write_jsonl(legacy_manifest, rows[:8])
            with self.assertRaisesRegex(
                release.Wan22FullMotionReleaseError,
                "release schema differs",
            ):
                self._verify(
                    signed=legacy,
                    manifest=legacy_manifest,
                    public=public,
                    fingerprint=fingerprint,
                )
            for shard_index, shard_rows in enumerate((rows[:8], rows[8:])):
                shard = root / f"shard_{shard_index}.jsonl"
                _write_jsonl(shard, shard_rows)
                verified = self._verify(
                    signed=signed,
                    manifest=shard,
                    public=public,
                    fingerprint=fingerprint,
                )
                self.assertEqual(verified["selected_row_count"], 8)
                self.assertEqual(
                    verified["release"]["root_row_start_zero_based"],
                    shard_index * 8,
                )
                self.assertEqual(
                    {
                        row["_authorization_mode"]
                        for row in verified["selected_rows"]
                    },
                    {release.AUTHORIZATION_MODE},
                )

    def test_real_cli_verify_accepts_only_each_contiguous_eight_row_slice(
        self,
    ) -> None:
        """Exercise the real CLI parser and verifier, not a controller mock."""

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows, signed, public, fingerprint = self._fixture(root)
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release, "SIGNER_KEY_FINGERPRINT", fingerprint
                ),
            ):
                for shard_index in range(2):
                    shard = root / f"cli_shard_{shard_index:03d}.jsonl"
                    _write_jsonl(
                        shard,
                        rows[shard_index * 8 : (shard_index + 1) * 8],
                    )
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        status = release.main(
                            [
                                "verify",
                                "--release",
                                str(signed),
                                "--manifest",
                                str(shard),
                            ]
                        )
                    self.assertEqual(status, 0)
                    binding = json.loads(stdout.getvalue())
                    self.assertEqual(
                        binding["root_manifest_sha256"],
                        _sha((root / "root_manifest.jsonl").read_bytes()),
                    )
                    self.assertEqual(binding["root_manifest_rows"], 16)
                    self.assertEqual(
                        binding["root_row_start_zero_based"], shard_index * 8
                    )
                    self.assertEqual(
                        binding["root_row_stop_exclusive"],
                        (shard_index + 1) * 8,
                    )

                with self.assertRaisesRegex(
                    release.Wan22FullMotionReleaseError,
                    "manifest must contain exactly 8 rows",
                ):
                    release.main(
                        [
                            "verify",
                            "--release",
                            str(signed),
                            "--manifest",
                            str(root / "root_manifest.jsonl"),
                        ]
                    )

    def test_noncontiguous_or_tampered_motion_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows, signed, public, fingerprint = self._fixture(root)
            invalid_manifests: list[list[dict]] = []
            invalid_manifests.append(rows[:4] + rows[8:12])
            instruction_tamper = json.loads(json.dumps(rows[:8]))
            instruction_tamper[0]["edit_instruction"] = "Do something else."
            invalid_manifests.append(instruction_tamper)
            motion_tamper = json.loads(json.dumps(rows[:8]))
            motion_tamper[0]["motion_spec"]["target_plan"][
                "dynamic_unit_targets"
            ][0]["target_action_signature"] = "forged_action"
            motion_tamper[0]["motion_spec_sha256"] = contract.object_sha256(
                motion_tamper[0]["motion_spec"]
            )
            invalid_manifests.append(motion_tamper)
            for index, invalid_rows in enumerate(invalid_manifests):
                manifest = root / f"invalid_{index}.jsonl"
                _write_jsonl(manifest, invalid_rows)
                with self.assertRaises(release.Wan22FullMotionReleaseError):
                    self._verify(
                        signed=signed,
                        manifest=manifest,
                        public=public,
                        fingerprint=fingerprint,
                    )

    def test_wan_loader_dispatches_new_schema_and_keeps_only_edit_instruction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows, signed, public, fingerprint = self._fixture(root)
            shard = root / "shard.jsonl"
            _write_jsonl(shard, rows[4:12])
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release, "SIGNER_KEY_FINGERPRINT", fingerprint
                ),
            ):
                loaded = batch.load_generation_manifest(
                    shard,
                    allow_pending_review=False,
                    max_samples=8,
                    signed_release_path=signed,
                )
            self.assertEqual(loaded["selected_row_count"], 8)
            for original, prepared in zip(rows[4:12], loaded["selected_rows"]):
                self.assertEqual(
                    prepared["edit_instruction"], original["edit_instruction"]
                )
                self.assertEqual(
                    prepared["_authorization_mode"], release.AUTHORIZATION_MODE
                )
                self.assertEqual(prepared["action_category"], "full_motion")
                self.assertEqual(
                    prepared["target_action_verb"],
                    "multi_entity_action_edit",
                )
                self.assertEqual(prepared["manifest_role"], "review_proposal")
                self.assertEqual(prepared["human_review_status"], "pending")
                self.assertFalse(prepared["generation_authorized"])
                self.assertFalse(prepared["production_eligible"])
                self.assertIsNone(prepared["approval"])
                self.assertNotIn("absolute_target_prompt", prepared)
                self.assertNotIn("edited_caption", prepared)

    def test_media_change_after_signing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows, signed, public, fingerprint = self._fixture(root)
            shard = root / "shard.jsonl"
            _write_jsonl(shard, rows[:8])
            source = Path(rows[0]["resolved_source_video"])
            source.write_bytes(source.read_bytes() + b"tamper")
            with self.assertRaises(release.Wan22FullMotionReleaseError):
                self._verify(
                    signed=signed,
                    manifest=shard,
                    public=public,
                    fingerprint=fingerprint,
                )

    def test_unfrozen_or_wrong_key_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, anchor = _write_media(root)
            manifest = root / "root.jsonl"
            _write_jsonl(
                manifest,
                [
                    _row(index=index, source_path=source, anchor_path=anchor)
                    for index in range(8)
                ],
            )
            key, _public, _fingerprint = _key(root)
            with (
                mock.patch.object(
                    release,
                    "SIGNER_PUBLIC_KEY",
                    "REPLACE_WITH_DEDICATED_FULL_MOTION_ED25519_PUBLIC_KEY",
                ),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    "REPLACE_WITH_DEDICATED_FULL_MOTION_KEY_FINGERPRINT",
                ),
                self.assertRaisesRegex(
                    release.Wan22FullMotionReleaseError,
                    "public key is not frozen",
                ),
            ):
                release.build_and_sign_release(
                    root_manifest_path=manifest,
                    output_path=root / "release.json",
                    signing_key=key,
                    release_id="placeholder-rejected",
                    issued_at_utc="2026-08-01T00:00:00+00:00",
                )

    def test_offline_prepare_and_sign_is_challenge_and_source_bound(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, anchor = _write_media(root)
            rows = [
                _row(index=index, source_path=source, anchor_path=anchor)
                for index in range(8)
            ]
            manifest = root / "root.jsonl"
            _write_jsonl(manifest, rows)
            challenge = _sha(b"offline-signing-challenge")
            request = root / "release_request.json"
            release.prepare_release_request(
                root_manifest_path=manifest,
                request_path=request,
                release_id="offline-unit-test",
                issued_at_utc="2026-08-01T00:00:00+00:00",
                challenge=challenge,
            )
            key, public, fingerprint = _key(root)
            signed = root / "offline_release.json"
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release, "SIGNER_KEY_FINGERPRINT", fingerprint
                ),
            ):
                release.sign_prepared_request(
                    request_path=request,
                    output_path=signed,
                    signing_key=key,
                    expected_challenge=challenge,
                )
            shard = root / "shard.jsonl"
            _write_jsonl(shard, rows)
            verified = self._verify(
                signed=signed,
                manifest=shard,
                public=public,
                fingerprint=fingerprint,
            )
            self.assertEqual(verified["selected_row_count"], 8)

            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release, "SIGNER_KEY_FINGERPRINT", fingerprint
                ),
                self.assertRaisesRegex(
                    release.Wan22FullMotionReleaseError,
                    "challenge differs",
                ),
            ):
                release.sign_prepared_request(
                    request_path=request,
                    output_path=root / "wrong_challenge.json",
                    signing_key=key,
                    expected_challenge="f" * 64,
                )


if __name__ == "__main__":
    unittest.main()
