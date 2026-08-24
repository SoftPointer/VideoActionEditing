from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from methods.motive.tests.test_goku_full_motion_postcheck import (
    _critic as _single_critic,
    _plan as _single_plan,
    _source as _single_source,
)
from methods.motive.tests.test_goku_full_motion_qwen import (
    _A0aSelfNegatedHeadLabelBackend,
    _args as _real_v4_args,
    _critic as _double_critic,
    _input_row as _real_v4_input_row,
    _source_census as _double_source,
    _target_plan as _double_plan,
    _write_jsonl as _write_real_v4_jsonl,
)
from motive import goku_full_motion_contract as contract
from motive import goku_full_motion_finalize as finalize
from motive import goku_full_motion_postcheck as postcheck
from motive import goku_full_motion_qwen as qwen
from motive import wan22_full_motion_signed_release as signed_release
from motive.goku_full_motion_instruction import compile_full_motion_instruction


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_sha(value: object) -> str:
    return _sha(_canonical(value))


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


class _FakeQwenApi:
    @staticmethod
    def assigned_iids_for_shard(
        rows, *, shard_index: int, num_shards: int, max_samples
    ):
        assigned = [
            str(row["iid"])
            for row in rows
            if int(hashlib.sha256(str(row["iid"]).encode()).hexdigest()[:16], 16)
            % num_shards
            == shard_index
        ]
        return assigned if max_samples is None else assigned[:max_samples]

    @staticmethod
    def shard_receipt_path(output: Path) -> Path:
        return output.with_name(f"{output.stem}.receipt.json")

    @staticmethod
    def qwen_result_payload(record):
        return {
            "change_region_proposals": record["change_region_proposals"],
            "coverage_authority": record["coverage_authority"],
            "i0_grounding": record["i0_grounding"],
            "source_census": record["source_census"],
            "source_census_canonicalization": record[
                "source_census_canonicalization"
            ],
            "secondary_source_census": record["secondary_source_census"],
            "secondary_source_census_canonicalization": record[
                "secondary_source_census_canonicalization"
            ],
            "source_inventory_alignment": record[
                "source_inventory_alignment"
            ],
            "coverage_authority_alignment": record[
                "coverage_authority_alignment"
            ],
            "target_plan": record["target_plan"],
            "target_plan_canonicalization": record[
                "target_plan_canonicalization"
            ],
            "compiled_instruction": record["compiled_instruction"],
            "full_motion_contract": record["full_motion_contract"],
            "coverage_critic": record["coverage_critic"],
            "hard_gate": record["hard_gate"],
            "pipeline_stage": record["pipeline_stage"],
            "pipeline_decision": record["pipeline_decision"],
        }

    @staticmethod
    def qwen_provenance_digest(record):
        return qwen.qwen_provenance_digest(record)

    @classmethod
    def validate_output_record(cls, record, *, selected_row, expected_bindings):
        if record.get("iid") != selected_row.get("iid"):
            raise ValueError("IID differs")
        if record.get("group_id") != selected_row.get("group_id"):
            raise ValueError("group differs")
        if record.get("family") != selected_row.get("family"):
            raise ValueError("family differs")
        if record.get("input_digest") != _object_sha(selected_row):
            raise ValueError("input digest differs")
        for field, expected in expected_bindings.items():
            if record.get(field) != expected:
                raise ValueError(f"{field} differs")
        grounding = qwen.validate_i0_grounding(
            record.get("i0_grounding"), expected_iid=str(record["iid"])
        )
        if record.get("i0_grounding_digest") != _object_sha(grounding):
            raise ValueError("I0 grounding digest differs")
        qwen.validate_source_census_i0_binding(
            record["source_census"], grounding
        )
        qwen.validate_source_census_i0_binding(
            record["secondary_source_census"], grounding
        )
        proposals = qwen.validate_change_region_proposals(
            record.get("change_region_proposals"),
            expected_iid=str(record["iid"]),
        )
        authority = qwen.validate_coverage_authority(
            record.get("coverage_authority"),
            expected_iid=str(record["iid"]),
            change_region_proposals=proposals,
        )
        alignment = qwen.validate_coverage_authority_alignment(
            record.get("coverage_authority_alignment"),
            coverage_authority=authority,
            change_region_proposals=proposals,
            i0_grounding=grounding,
            primary=record["source_census"],
            secondary=record["secondary_source_census"],
            source_inventory_alignment=record["source_inventory_alignment"],
        )
        for artifact, field in (
            (proposals, "change_region_proposals_digest"),
            (
                authority["inventory"],
                "coverage_authority_inventory_digest",
            ),
            (
                authority["assignments"],
                "coverage_authority_assignments_digest",
            ),
            (authority, "coverage_authority_digest"),
            (alignment, "coverage_authority_alignment_digest"),
        ):
            if record.get(field) != _object_sha(artifact):
                raise ValueError(f"{field} differs")
        if record.get("provenance_digest") != cls.qwen_provenance_digest(record):
            raise ValueError("provenance differs")
        if record.get("status") == "ok":
            if record.get("result_digest") != _object_sha(
                cls.qwen_result_payload(record)
            ):
                raise ValueError("result differs")
        elif record.get("status") != "error":
            raise ValueError("status differs")
        return dict(record)

    @classmethod
    def validate_shard_receipt(
        cls,
        receipt,
        *,
        output,
        input_path,
        input_sha256,
        root,
        assigned_iids,
        selected_by_iid,
        shard_index,
        num_shards,
        implementation_digest,
        config_digest,
        run_config_digest,
        run_config,
        backend,
    ):
        expected = {
            "schema_version": "goku-full-motion-qwen-shard-receipt-v2",
            "status": "complete",
            "execution_manifest": str(input_path),
            "execution_manifest_sha256": input_sha256,
            "root": str(root),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "assigned_iids": list(assigned_iids),
            "implementation_digest": implementation_digest,
            "config_digest": config_digest,
            "run_config_digest": run_config_digest,
            "run_config": dict(run_config),
            "model_path": backend.model_path,
            "model_revision": backend.model_revision,
            "transformers_version": backend.transformers_version,
        }
        for field, value in expected.items():
            if receipt.get(field) != value:
                raise ValueError(f"receipt {field} differs")
        raw = output.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        if [row["iid"] for row in rows] != list(assigned_iids):
            raise ValueError("receipt coverage differs")
        if receipt.get("output") != {
            "path": str(output.resolve()),
            "sha256": _sha(raw),
            "bytes": len(raw),
            "rows": len(rows),
        }:
            raise ValueError("receipt output differs")
        for record in rows:
            bindings = {
                "execution_manifest": str(input_path),
                "execution_manifest_sha256": input_sha256,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "implementation_digest": implementation_digest,
                "config_digest": config_digest,
                "run_config_digest": run_config_digest,
                "model_path": backend.model_path,
                "model_revision": backend.model_revision,
                "transformers_version": backend.transformers_version,
            }
            cls.validate_output_record(
                record,
                selected_row=selected_by_iid[record["iid"]],
                expected_bindings=bindings,
            )
        payload = dict(receipt)
        digest = payload.pop("receipt_digest", None)
        if digest != _object_sha(payload):
            raise ValueError("receipt digest differs")
        return dict(receipt)


def _candidate(index: int, *, family: str | None = None) -> dict:
    iid = f"iid{index:03d}"
    return {
        "schema_version": "candidate-v1",
        "iid": iid,
        "group_id": f"group-{index:03d}",
        "family": family or f"family-{index % 4}",
        "src_video": f"source/{iid}.mp4",
        "resolved_src_video": f"/source/{iid}.mp4",
        "anchor_image": f"anchor/{iid}.png",
        "resolved_anchor_image": f"/anchor/{iid}.png",
        "prompt": "legacy untrusted seed",
        "source_video_sha256": _sha(f"source-{iid}".encode()),
        "anchor_sha256": _sha(f"anchor-{iid}".encode()),
        "media": {
            "width": 832,
            "height": 480,
            "frame_count": 81,
            "fps": 25.0,
            "duration_seconds": 3.24,
        },
    }


def _i0_grounding_for_source(source: dict) -> dict:
    """Build a closed exact-I0 fixture bound to each person/animal registry row."""

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
    """Build small closed v6 two-stage authority fixtures."""

    registry = {
        item["entity_id"]: item for item in source["i0_entity_registry"]
    }
    units = {
        item["entity_id"]: item
        for item in (*source["dynamic_units"], *source["static_salient_people"])
    }
    cell_owners: list[tuple[int, str]] = []
    for unit in source["dynamic_units"]:
        bbox = registry[unit["entity_id"]]["i0_bbox_xyxy_1000"]
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        column = min(4, center_x * 4 // 1000 + 1)
        row = min(4, center_y * 4 // 1000 + 1)
        cell_owners.append(((row - 1) * 4 + column - 1, unit["entity_id"]))
    if len({ordinal for ordinal, _ in cell_owners}) != len(cell_owners):
        raise AssertionError("fixture dynamic entities need distinct authority cells")
    cell_owners.sort()
    regions = []
    for index, (ordinal, _entity_id) in enumerate(cell_owners, start=1):
        row, column = divmod(ordinal, 4)
        regions.append(
            {
                "schema_version": qwen.CHANGE_REGION_SCHEMA,
                "proposal_id": f"proposal_{index:02d}",
                "cell_row": row + 1,
                "cell_column": column + 1,
                "bbox_xyxy_1000": [
                    column * 250,
                    row * 250,
                    (column + 1) * 250,
                    (row + 1) * 250,
                ],
                "changed_pixel_count": 64,
                "bbox_area_pixels": 100,
                "changed_fraction_ppm": 640_000,
                "delta_at_percentile_milli": 20_000,
            }
        )
    proposals = {
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
        "regions": regions,
        "active_cell_count": len(regions),
        "global_changed_fraction_ppm": 40_000,
        "all_active_cells_emitted": True,
    }
    proposals = qwen.validate_change_region_proposals(
        proposals, expected_iid=source["iid"]
    )
    subjects = []
    authority_by_entity: dict[str, str] = {}
    for entity in source["i0_entity_registry"]:
        if entity["entity_type"] not in {"person", "animal"}:
            continue
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
                        item["component_type"]
                        for item in unit["source_motion_components"]
                    ]
                    if dynamic
                    else []
                ),
                "motion_evidence": unit["motion_evidence"],
                "confidence": "high",
            }
        )
    assignments = []
    for proposal, (_ordinal, entity_id) in zip(
        regions, cell_owners, strict=True
    ):
        assignments.append(
            {
                "schema_version": qwen.CHANGE_REGION_ASSIGNMENT_SCHEMA,
                "proposal_id": proposal["proposal_id"],
                "assignment_kind": "entity",
                "authority_entity_ids": [authority_by_entity[entity_id]],
                "resolution_reason": (
                    "The active grid cell overlaps this independently moving "
                    "subject throughout the temporal checkpoints"
                ),
                "reject_reason_code": None,
                "confidence": "high",
            }
        )
    inventory = {
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
    }
    inventory = qwen.validate_coverage_authority_inventory(
        inventory,
        expected_iid=source["iid"],
    )
    allowed_owner_map = qwen.build_coverage_authority_allowed_owner_map(
        coverage_authority_inventory=inventory,
        change_region_proposals=proposals,
    )
    assignment_record = {
        "schema_version": qwen.COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA,
        "iid": source["iid"],
        "coverage_authority_inventory_sha256": contract.object_sha256(
            inventory
        ),
        "change_region_proposals_sha256": contract.object_sha256(proposals),
        "allowed_owner_map_sha256": contract.object_sha256(
            allowed_owner_map
        ),
        "change_region_assignments": assignments,
        "all_change_regions_resolved": True,
        "uncertainty_codes": [],
        "confidence": "high",
    }
    assignment_record = qwen.validate_coverage_authority_assignments(
        assignment_record,
        expected_iid=source["iid"],
        coverage_authority_inventory=inventory,
        change_region_proposals=proposals,
    )
    authority = qwen.build_coverage_authority(
        coverage_authority_inventory=inventory,
        coverage_authority_assignments=assignment_record,
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


def _record(
    row: dict,
    *,
    shard_index: int,
    input_path: Path,
    input_sha: str,
    dynamic_count: int,
    rejected: bool,
) -> dict:
    iid = row["iid"]
    if dynamic_count == 1:
        source = _single_source(iid)
        target = _single_plan(source)
        target["dynamic_unit_targets"][0]["target_action_signature"] += (
            f"_{iid}"
        )
        compiled = compile_full_motion_instruction(source, target)
        coverage = _single_critic(source, target, compiled)
    elif dynamic_count == 2:
        source = _double_source(iid)
        target = _double_plan(source)
        for unit_target in target["dynamic_unit_targets"]:
            unit_target["target_action_signature"] += f"_{iid}"
        compiled = compile_full_motion_instruction(source, target)
        coverage = _double_critic(source, target, compiled)
    else:  # pragma: no cover - fixture contract
        raise ValueError(f"unsupported dynamic_count={dynamic_count}")
    secondary_source = copy.deepcopy(source)
    i0_grounding = _i0_grounding_for_source(source)
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
    full_motion_contract = contract.build_contract(
        source_census=source,
        target_plan=target,
    )
    canonical_source, source_canonicalization = (
        contract.canonicalize_source_census_model_output(source, iid)
    )
    canonical_secondary, secondary_canonicalization = (
        contract.canonicalize_source_census_model_output(
            secondary_source, iid
        )
    )
    canonical_target, target_canonicalization = (
        contract.canonicalize_target_plan_model_output(target, source)
    )
    if (
        canonical_source != source
        or canonical_secondary != secondary_source
        or canonical_target != target
    ):
        raise AssertionError("canonical fixture unexpectedly changed")
    gate = qwen.build_hard_gate(
        change_region_proposals=change_region_proposals,
        coverage_authority=coverage_authority,
        coverage_authority_alignment=coverage_authority_alignment,
        i0_grounding=i0_grounding,
        source_census=source,
        source_census_canonicalization=source_canonicalization,
        secondary_source_census=secondary_source,
        secondary_source_census_canonicalization=(
            secondary_canonicalization
        ),
        source_inventory_alignment=source_inventory_alignment,
        target_plan=target,
        target_plan_canonicalization=target_canonicalization,
        compiled_instruction=compiled,
        coverage_critic=coverage,
    )
    config_digest = _sha(f"config-{shard_index}".encode())
    record_visual_digest = _sha(f"visual-{iid}".encode())
    record = {
        "schema_version": qwen.RECORD_SCHEMA,
        "iid": iid,
        "group_id": row["group_id"],
        "family": row["family"],
        "status": "error" if rejected else "ok",
        "error_type": "FixtureRejected" if rejected else None,
        "error": "fixture rejection" if rejected else None,
        "input_digest": _object_sha(row),
        "config_digest": config_digest,
        "run_config_digest": _sha(b"run-config"),
        "implementation_digest": _sha(b"implementation"),
        "execution_manifest": str(input_path),
        "execution_manifest_sha256": input_sha,
        "shard_index": shard_index,
        "num_shards": 8,
        "model_path": "/models/Qwen3-VL-32B-Instruct",
        "model_revision": "revision",
        "transformers_version": "5.5.4",
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
        "failure_stage": "fixture_reject" if rejected else None,
        "resolved_src_video": row["resolved_src_video"],
        "resolved_anchor_image": row["resolved_anchor_image"],
        "visual_input_digest": record_visual_digest,
        "media_verification": {
            "exact_i0": True,
            "source_video_sha256": row["source_video_sha256"],
            "anchor_sha256": row["anchor_sha256"],
        },
        "legacy_seed": {
            "role": "untrusted_optional_legacy_action_seed",
            "text": row["prompt"],
            "sha256": _sha(row["prompt"].encode()),
            "authoritative": False,
            "source_caption_used": False,
            "edited_caption_used": False,
            "old_target_video_used": False,
        },
        "change_region_proposals": change_region_proposals,
        "change_region_proposals_digest": _object_sha(
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
        "coverage_authority_inventory_digest": _object_sha(
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
        "coverage_authority_assignments_digest": _object_sha(
            coverage_authority["assignments"]
        ),
        "coverage_authority": coverage_authority,
        "coverage_authority_digest": _object_sha(coverage_authority),
        "i0_grounding": i0_grounding,
        "i0_grounding_prompt_digest": _sha(
            f"grounding-prompt-{iid}".encode()
        ),
        "i0_grounding_visual_input_digest": _sha(
            f"grounding-visual-{iid}".encode()
        ),
        "i0_grounding_raw": _canonical(i0_grounding).decode(),
        "i0_grounding_validated_from": "original",
        "i0_grounding_digest": _object_sha(i0_grounding),
        "source_census": source,
        "source_census_prompt_digest": _sha(
            f"source-prompt-{iid}".encode()
        ),
        "source_census_raw": _canonical(source).decode(),
        "source_census_validated_from": "canonicalized_original",
        "source_census_canonicalization": source_canonicalization,
        "source_census_canonicalization_digest": _object_sha(
            source_canonicalization
        ),
        "source_census_digest": _object_sha(source),
        "secondary_source_census": secondary_source,
        "secondary_source_census_prompt_digest": _sha(
            f"secondary-prompt-{iid}".encode()
        ),
        "secondary_source_census_visual_input_digest": record_visual_digest,
        "secondary_source_census_raw": _canonical(
            secondary_source
        ).decode(),
        "secondary_source_census_validated_from": "canonicalized_original",
        "secondary_source_census_canonicalization": (
            secondary_canonicalization
        ),
        "secondary_source_census_canonicalization_digest": _object_sha(
            secondary_canonicalization
        ),
        "secondary_source_census_digest": _object_sha(secondary_source),
        "source_inventory_alignment": source_inventory_alignment,
        "source_inventory_alignment_digest": _object_sha(
            source_inventory_alignment
        ),
        "coverage_authority_alignment": coverage_authority_alignment,
        "coverage_authority_alignment_digest": _object_sha(
            coverage_authority_alignment
        ),
        "target_plan": target,
        "target_plan_prompt_digest": _sha(
            f"target-prompt-{iid}".encode()
        ),
        "target_plan_visual_input_digest": record_visual_digest,
        "target_plan_raw": _canonical(target).decode(),
        "target_plan_validated_from": "canonicalized_original",
        "target_plan_canonicalization": target_canonicalization,
        "target_plan_canonicalization_digest": _object_sha(
            target_canonicalization
        ),
        "target_plan_digest": _object_sha(target),
        "compiled_instruction": compiled,
        "compiled_instruction_digest": _object_sha(compiled),
        "full_motion_contract": full_motion_contract,
        "full_motion_contract_digest": "",
        "coverage_critic": coverage,
        "coverage_critic_prompt_digest": _sha(
            f"critic-prompt-{iid}".encode()
        ),
        "coverage_critic_visual_input_digest": record_visual_digest,
        "coverage_critic_raw": _canonical(coverage).decode(),
        "coverage_critic_validated_from": "original",
        "coverage_critic_digest": "",
        "hard_gate": gate,
        "pipeline_stage": "coverage_critic",
        "pipeline_decision": "pass",
        "result_digest": None,
        "provenance_digest": None,
    }
    record["full_motion_contract_digest"] = _object_sha(
        record["full_motion_contract"]
    )
    record["coverage_critic_digest"] = _object_sha(record["coverage_critic"])
    if not rejected:
        record["result_digest"] = _object_sha(
            _FakeQwenApi.qwen_result_payload(record)
        )
    record["provenance_digest"] = _FakeQwenApi.qwen_provenance_digest(record)
    return record


def _make_qwen_run(
    root: Path,
    *,
    count: int = 12,
    multi_indices: set[int] | None = None,
    rejected_indices: set[int] | None = None,
    families: list[str] | None = None,
) -> tuple[Path, list[Path], list[dict]]:
    multi = multi_indices or set()
    rejected = rejected_indices or set()
    rows = [
        _candidate(index, family=families[index] if families else None)
        for index in range(count)
    ]
    candidate_path = root / "candidates.jsonl"
    candidate_raw = _jsonl(rows)
    candidate_path.write_bytes(candidate_raw)
    outputs: list[Path] = []
    for shard_index in range(8):
        output = root / f"qwen_shard_{shard_index:03d}.jsonl"
        assigned = _FakeQwenApi.assigned_iids_for_shard(
            rows,
            shard_index=shard_index,
            num_shards=8,
            max_samples=None,
        )
        by_iid = {row["iid"]: (index, row) for index, row in enumerate(rows)}
        records = []
        for iid in assigned:
            index, row = by_iid[iid]
            records.append(
                _record(
                    row,
                    shard_index=shard_index,
                    input_path=candidate_path.resolve(),
                    input_sha=_sha(candidate_raw),
                    dynamic_count=2 if index in multi else 1,
                    rejected=index in rejected,
                )
            )
        output_raw = _jsonl(records)
        output.write_bytes(output_raw)
        receipt = {
            "schema_version": "goku-full-motion-qwen-shard-receipt-v2",
            "status": "complete",
            "execution_manifest": str(candidate_path.resolve()),
            "execution_manifest_sha256": _sha(candidate_raw),
            "root": str(root.resolve()),
            "shard_index": shard_index,
            "num_shards": 8,
            "assigned_iids": assigned,
            "implementation_digest": _sha(b"implementation"),
            "config_digest": _sha(f"config-{shard_index}".encode()),
            "run_config_digest": _sha(b"run-config"),
            "run_config": {"max_samples": None, "num_shards": 8},
            "model_path": "/models/Qwen3-VL-32B-Instruct",
            "model_revision": "revision",
            "transformers_version": "5.5.4",
            "output": {
                "path": str(output.resolve()),
                "sha256": _sha(output_raw),
                "bytes": len(output_raw),
                "rows": len(records),
            },
        }
        receipt["receipt_digest"] = _object_sha(receipt)
        _FakeQwenApi.shard_receipt_path(output).write_bytes(
            _canonical(receipt) + b"\n"
        )
        outputs.append(output)
    return candidate_path, outputs, rows


def _run(
    *,
    candidate: Path,
    outputs: list[Path],
    output_dir: Path,
    required_iid: str,
    primary_size: int = 6,
    reserve_size: int = 2,
    min_multi: int = 3,
    family_cap: int = 32,
    signature_cap: int = 32,
):
    with mock.patch.object(
        finalize, "_load_qwen_api", return_value=_FakeQwenApi
    ):
        return finalize.finalize_full_motion(
            candidate_manifest=candidate,
            qwen_outputs=outputs,
            output_dir=output_dir,
            primary_size=primary_size,
            reserve_size=reserve_size,
            min_primary_multi_dynamic=min_multi,
            target_signature_cap=signature_cap,
            family_cap=family_cap,
            required_iids=[required_iid],
        )


class GokuFullMotionFinalizeTests(unittest.TestCase):
    def test_quota_canary_shape_and_hash_closure(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate, outputs, _ = _make_qwen_run(
                root, multi_indices={4, 5, 6, 7, 8}
            )
            summary = _run(
                candidate=candidate,
                outputs=outputs,
                output_dir=root / "final",
                required_iid="iid005",
            )
            self.assertGreaterEqual(summary["counts"]["primary_multi_dynamic_rows"], 3)
            primary = [
                json.loads(line)
                for line in (root / "final" / "primary_6.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(primary), 6)
            self.assertIn("iid005", [row["iid"] for row in primary])
            self.assertTrue(
                next(row for row in primary if row["iid"] == "iid005")[
                    "full_motion_finalization"
                ]["required_canary"]
            )
            for row in primary:
                finalize.validate_generation_row(row)
                self.assertEqual(row["strict_temporal_geometry"]["target_frame_count"], 81)
                self.assertEqual(row["strict_temporal_geometry"]["target_frame_rate"], "25/1")
                self.assertFalse(row["human_reviewed"])
                self.assertEqual(row["human_review_status"], "pending")
            done = json.loads((root / "final" / "done.json").read_text())
            done_payload = dict(done)
            digest = done_payload.pop("done_digest")
            self.assertEqual(digest, _object_sha(done_payload))
            for artifact, metadata in done["artifacts"].items():
                self.assertEqual(
                    metadata["sha256"], _sha((root / "final" / artifact).read_bytes())
                )

    def test_missing_terminal_receipt_fails(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate, outputs, _ = _make_qwen_run(root, multi_indices={0, 1, 2})
            _FakeQwenApi.shard_receipt_path(outputs[3]).unlink()
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "terminal receipt"
            ):
                _run(
                    candidate=candidate,
                    outputs=outputs,
                    output_dir=root / "final",
                    required_iid="iid000",
                )
            self.assertFalse((root / "final").exists())

    def test_tampered_qwen_output_fails_receipt_validation(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate, outputs, _ = _make_qwen_run(root, multi_indices={0, 1, 2})
            outputs[0].write_bytes(outputs[0].read_bytes() + b"\n")
            with self.assertRaises(finalize.GokuFullMotionFinalizeError):
                _run(
                    candidate=candidate,
                    outputs=outputs,
                    output_dir=root / "final",
                    required_iid="iid000",
                )

    def test_rejected_canary_fails_hard(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate, outputs, _ = _make_qwen_run(
                root,
                multi_indices={0, 1, 2, 3},
                rejected_indices={2},
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError,
                "required canary did not hard-pass",
            ):
                _run(
                    candidate=candidate,
                    outputs=outputs,
                    output_dir=root / "final",
                    required_iid="iid002",
                )

    def test_multi_dynamic_quota_never_silently_downgrades(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate, outputs, _ = _make_qwen_run(root, multi_indices={0, 1})
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError,
                "multi-dynamic quota cannot be satisfied",
            ):
                _run(
                    candidate=candidate,
                    outputs=outputs,
                    output_dir=root / "final",
                    required_iid="iid000",
                )

    def test_selection_and_artifacts_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate, outputs, _ = _make_qwen_run(
                root, multi_indices={2, 3, 5, 7, 9}
            )
            _run(
                candidate=candidate,
                outputs=outputs,
                output_dir=root / "first",
                required_iid="iid005",
            )
            _run(
                candidate=candidate,
                outputs=list(reversed(outputs)),
                output_dir=root / "second",
                required_iid="iid005",
            )
            for artifact in (
                "primary_6.jsonl",
                "reserve_2.jsonl",
                "review_candidates.jsonl",
                "summary.json",
                "done.json",
            ):
                self.assertEqual(
                    (root / "first" / artifact).read_bytes(),
                    (root / "second" / artifact).read_bytes(),
                )

    def test_family_cap_is_enforced_across_primary_and_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            families = ["same"] * 4 + [f"other-{index}" for index in range(8)]
            candidate, outputs, _ = _make_qwen_run(
                root,
                multi_indices={0, 4, 5},
                families=families,
            )
            summary = _run(
                candidate=candidate,
                outputs=outputs,
                output_dir=root / "final",
                required_iid="iid000",
                primary_size=4,
                reserve_size=2,
                min_multi=2,
                family_cap=1,
            )
            self.assertLessEqual(summary["diversity"]["family_counts"]["same"], 1)

    def test_real_v6_qwen_records_finalize_and_release_with_receipt_closure(
        self,
    ) -> None:
        """Exercise real Qwen v6 validators, not the selection-only fake."""

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows = [
                _real_v4_input_row(root, iid=f"real-v6-{index:03d}")
                for index in range(8)
            ]
            candidate = root / "candidates.jsonl"
            _write_real_v4_jsonl(candidate, rows)
            outputs: list[Path] = []
            _A0aSelfNegatedHeadLabelBackend.instances.clear()
            for shard_index in range(8):
                output = root / f"qwen_shard_{shard_index:03d}.jsonl"
                args = _real_v4_args(candidate, output, rows[0])
                args.shard_index = shard_index
                self.assertEqual(
                    qwen.run_audit(
                        args,
                        backend_factory=_A0aSelfNegatedHeadLabelBackend,
                    ),
                    0,
                )
                outputs.append(output)

            pool = root / "final"
            finalize.finalize_full_motion(
                candidate_manifest=candidate,
                qwen_outputs=outputs,
                output_dir=pool,
                primary_size=8,
                reserve_size=0,
                min_primary_multi_dynamic=8,
                target_signature_cap=32,
                family_cap=32,
                required_iids=[rows[0]["iid"]],
            )
            primary = pool / "primary_8.jsonl"
            finalized_rows = [
                json.loads(line) for line in primary.read_text().splitlines()
            ]
            self.assertEqual(len(finalized_rows), 8)
            for row in finalized_rows:
                validated = finalize.validate_generation_row(row)
                normalized = postcheck._normalize_contract(
                    row, manifest_root=pool
                )
                spec = validated["motion_spec"]
                evidence = validated["qwen_evidence"]
                self.assertEqual(
                    evidence["qwen_record_payload"][
                        "coverage_authority_inventory_validated_from"
                    ],
                    "canonicalized_original",
                )
                self.assertEqual(
                    normalized["coverage_authority"],
                    spec["coverage_authority"],
                )
                self.assertEqual(
                    evidence["i0_grounding_digest"],
                    contract.object_sha256(spec["i0_grounding"]),
                )
                self.assertEqual(
                    evidence["source_census_digest"],
                    contract.object_sha256(spec["source_census"]),
                )
                self.assertEqual(
                    evidence["secondary_source_census_digest"],
                    contract.object_sha256(spec["secondary_source_census"]),
                )
                self.assertEqual(
                    evidence["source_inventory_alignment_digest"],
                    contract.object_sha256(spec["source_inventory_alignment"]),
                )
                self.assertEqual(
                    evidence["change_region_proposals_digest"],
                    contract.object_sha256(spec["change_region_proposals"]),
                )
                self.assertEqual(
                    evidence["coverage_authority_digest"],
                    contract.object_sha256(spec["coverage_authority"]),
                )
                self.assertEqual(
                    evidence["coverage_authority_inventory_digest"],
                    contract.object_sha256(
                        spec["coverage_authority"]["inventory"]
                    ),
                )
                self.assertEqual(
                    evidence["coverage_authority_assignments_digest"],
                    contract.object_sha256(
                        spec["coverage_authority"]["assignments"]
                    ),
                )
                self.assertEqual(
                    evidence["coverage_authority_alignment_digest"],
                    contract.object_sha256(
                        spec["coverage_authority_alignment"]
                    ),
                )

            payload = signed_release.build_release_payload(
                root_manifest_path=primary,
                release_id="real-v6-cross-module",
                issued_at_utc="2026-08-01T00:00:00+00:00",
                verify_media=False,
            )
            self.assertEqual(len(payload["row_authorizations"]), 8)
            for row, authorization in zip(
                finalized_rows,
                payload["row_authorizations"],
                strict=True,
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
                    authorization[
                        "source_census_canonicalization_sha256"
                    ],
                    evidence["source_census_canonicalization_digest"],
                )
                self.assertEqual(
                    authorization[
                        "secondary_source_census_canonicalization_sha256"
                    ],
                    evidence[
                        "secondary_source_census_canonicalization_digest"
                    ],
                )
                self.assertEqual(
                    authorization[
                        "target_plan_canonicalization_sha256"
                    ],
                    evidence["target_plan_canonicalization_digest"],
                )

    def test_generation_row_rejects_v5_and_authority_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate, outputs, _ = _make_qwen_run(
                root,
                count=8,
                multi_indices=set(range(8)),
            )
            _run(
                candidate=candidate,
                outputs=outputs,
                output_dir=root / "final",
                required_iid="iid000",
                primary_size=6,
                reserve_size=2,
                min_multi=6,
            )
            row = json.loads(
                (root / "final" / "primary_6.jsonl")
                .read_text()
                .splitlines()[0]
            )

            old_generation = copy.deepcopy(row)
            old_generation["schema_version"] = (
                "motive-goku-full-motion-generation-v5"
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "generation row schema"
            ):
                finalize.validate_generation_row(old_generation)

            old_evidence = copy.deepcopy(row)
            old_evidence["qwen_evidence"]["schema_version"] = (
                "motive-goku-full-motion-qwen-evidence-v5"
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "evidence schema"
            ):
                finalize.validate_generation_row(old_evidence)

            old_record = copy.deepcopy(row)
            old_record["qwen_evidence"]["record_schema_version"] = (
                "goku-full-motion-qwen-record-v5"
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "v6|provenance"
            ):
                finalize.validate_generation_row(old_record)

            forged_result = copy.deepcopy(row)
            forged_result_sha = "a" * 64
            forged_result["motion_spec"]["qwen_result_digest"] = (
                forged_result_sha
            )
            forged_result["qwen_evidence"]["result_digest"] = (
                forged_result_sha
            )
            forged_record = forged_result["qwen_evidence"][
                "qwen_record_payload"
            ]
            forged_record["result_digest"] = forged_result_sha
            forged_provenance = qwen.qwen_provenance_digest(forged_record)
            forged_record["provenance_digest"] = forged_provenance
            forged_result["qwen_evidence"]["provenance_digest"] = (
                forged_provenance
            )
            forged_result["motion_spec"]["qwen_provenance_digest"] = (
                forged_provenance
            )
            forged_result["motion_spec_sha256"] = _object_sha(
                forged_result["motion_spec"]
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError,
                "result digest",
            ):
                finalize.validate_generation_row(forged_result)

            forged_provenance_row = copy.deepcopy(row)
            forged_provenance_row["motion_spec"][
                "qwen_provenance_digest"
            ] = "b" * 64
            forged_provenance_row["qwen_evidence"][
                "provenance_digest"
            ] = "b" * 64
            forged_provenance_row["qwen_evidence"][
                "qwen_record_payload"
            ]["provenance_digest"] = "b" * 64
            forged_provenance_row["motion_spec_sha256"] = _object_sha(
                forged_provenance_row["motion_spec"]
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError,
                "provenance digest",
            ):
                finalize.validate_generation_row(forged_provenance_row)

            old_gate = copy.deepcopy(row)
            old_gate["qwen_evidence"]["hard_gate"] = {
                "schema_version": "goku-full-motion-hard-gate-v5",
                "source_census_sha256": row["qwen_evidence"][
                    "source_census_digest"
                ],
                "secondary_source_census_sha256": row["qwen_evidence"][
                    "secondary_source_census_digest"
                ],
                "source_inventory_alignment_sha256": row["qwen_evidence"][
                    "source_inventory_alignment_digest"
                ],
                "decision": "pass",
                "risk_codes": [],
            }
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "v6|binding|digest|closure"
            ):
                finalize.validate_generation_row(old_gate)

            wrong_secondary_digest = copy.deepcopy(row)
            wrong_secondary_digest["qwen_evidence"][
                "secondary_source_census_digest"
            ] = "0" * 64
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "v6|binding|digest|closure"
            ):
                finalize.validate_generation_row(wrong_secondary_digest)

            wrong_receipt = copy.deepcopy(row)
            wrong_receipt["qwen_evidence"][
                "target_plan_canonicalization"
            ]["canonical_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError,
                "canonical artifact binding",
            ):
                finalize.validate_generation_row(wrong_receipt)

            wrong_receipt_digest = copy.deepcopy(row)
            wrong_receipt_digest["qwen_evidence"][
                "source_census_canonicalization_digest"
            ] = "0" * 64
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "v6|binding|digest|closure"
            ):
                finalize.validate_generation_row(wrong_receipt_digest)

            changed_grounding = copy.deepcopy(row)
            changed_grounding["motion_spec"]["i0_grounding"]["subjects"][0][
                "viewer_left_extremity_state"
            ] = "viewer-left extremity stays below the waist at I0"
            changed_grounding["motion_spec_sha256"] = _object_sha(
                changed_grounding["motion_spec"]
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "v6|binding|digest|closure"
            ):
                finalize.validate_generation_row(changed_grounding)

            changed_grounding_digest = copy.deepcopy(row)
            changed_grounding_digest["qwen_evidence"][
                "i0_grounding_digest"
            ] = "0" * 64
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "v6|binding|digest|closure"
            ):
                finalize.validate_generation_row(changed_grounding_digest)

            changed_gate_grounding_digest = copy.deepcopy(row)
            changed_gate_grounding_digest["qwen_evidence"]["hard_gate"][
                "i0_grounding_sha256"
            ] = "0" * 64
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "v6|binding|digest|closure"
            ):
                finalize.validate_generation_row(changed_gate_grounding_digest)

            old_motion_spec = copy.deepcopy(row)
            old_motion_spec["motion_spec"]["schema_version"] = (
                "motive-goku-full-motion-generation-spec-v5"
            )
            old_motion_spec["motion_spec_sha256"] = _object_sha(
                old_motion_spec["motion_spec"]
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "motion_spec schema"
            ):
                finalize.validate_generation_row(old_motion_spec)

            shadow = copy.deepcopy(row)
            shadow["motion_spec"]["coverage_authority_shadow"] = shadow[
                "motion_spec"
            ]["coverage_authority"]
            shadow["motion_spec_sha256"] = _object_sha(shadow["motion_spec"])
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "closed schema"
            ):
                finalize.validate_generation_row(shadow)

            wrong_inventory_digest = copy.deepcopy(row)
            wrong_inventory_digest["qwen_evidence"][
                "coverage_authority_inventory_digest"
            ] = "0" * 64
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError,
                "v6|binding|digest|closure",
            ):
                finalize.validate_generation_row(wrong_inventory_digest)

            mismatched_inventory_prompt = copy.deepcopy(row)
            prompt_record = mismatched_inventory_prompt["qwen_evidence"][
                "qwen_record_payload"
            ]
            prompt_record[
                "coverage_authority_inventory_prompt_digest"
            ] = "1" * 64
            prompt_provenance = qwen.qwen_provenance_digest(prompt_record)
            prompt_record["provenance_digest"] = prompt_provenance
            mismatched_inventory_prompt["qwen_evidence"][
                "provenance_digest"
            ] = prompt_provenance
            mismatched_inventory_prompt["motion_spec"][
                "qwen_provenance_digest"
            ] = prompt_provenance
            mismatched_inventory_prompt["motion_spec_sha256"] = _object_sha(
                mismatched_inventory_prompt["motion_spec"]
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError,
                "artifact digest binding",
            ):
                finalize.validate_generation_row(mismatched_inventory_prompt)

            tampered_assignments_raw = copy.deepcopy(row)
            assignment_record = tampered_assignments_raw["qwen_evidence"][
                "qwen_record_payload"
            ]
            raw_assignments = json.loads(
                assignment_record["coverage_authority_assignments_raw"]
            )
            raw_assignments["change_region_assignments"][0][
                "resolution_reason"
            ] = "The same grid cell has a different raw-model explanation"
            assignment_record["coverage_authority_assignments_raw"] = (
                _canonical(raw_assignments).decode()
            )
            assignments_provenance = qwen.qwen_provenance_digest(
                assignment_record
            )
            assignment_record["provenance_digest"] = assignments_provenance
            tampered_assignments_raw["qwen_evidence"][
                "provenance_digest"
            ] = assignments_provenance
            tampered_assignments_raw["motion_spec"][
                "qwen_provenance_digest"
            ] = assignments_provenance
            tampered_assignments_raw["motion_spec_sha256"] = _object_sha(
                tampered_assignments_raw["motion_spec"]
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError,
                "two-stage A0 raw/object binding",
            ):
                finalize.validate_generation_row(tampered_assignments_raw)

            tampered_proposal = copy.deepcopy(row)
            tampered_proposal["motion_spec"]["change_region_proposals"][
                "regions"
            ][0]["changed_pixel_count"] += 1
            tampered_proposal["motion_spec_sha256"] = _object_sha(
                tampered_proposal["motion_spec"]
            )
            tampered_proposal["qwen_evidence"][
                "change_region_proposals_digest"
            ] = _object_sha(
                tampered_proposal["motion_spec"]["change_region_proposals"]
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "v6|binding|digest|closure"
            ):
                finalize.validate_generation_row(tampered_proposal)

            tampered_alignment = copy.deepcopy(row)
            tampered_alignment["motion_spec"][
                "coverage_authority_alignment"
            ]["all_authority_entities_aligned"] = False
            tampered_alignment["motion_spec_sha256"] = _object_sha(
                tampered_alignment["motion_spec"]
            )
            tampered_alignment["qwen_evidence"][
                "coverage_authority_alignment_digest"
            ] = _object_sha(
                tampered_alignment["motion_spec"][
                    "coverage_authority_alignment"
                ]
            )
            with self.assertRaisesRegex(
                finalize.GokuFullMotionFinalizeError, "v6|binding|digest|closure"
            ):
                finalize.validate_generation_row(tampered_alignment)


if __name__ == "__main__":
    unittest.main()
