"""Fail-closed gate between the real Qwen smoke and the full annotation run.

The smoke is not accepted merely because eight processes exited.  This module
revalidates every terminal Qwen shard receipt and output row against the exact
input bytes, requires a useful overall hard-pass rate, and treats the known
two-person example as a semantic canary: both independently moving people and
the camera must be represented by the deterministic compiled instruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

from . import goku_full_motion_qwen as _qwen
from .goku_full_motion_contract import (
    GokuFullMotionContractError,
    object_sha256,
    validate_source_census,
)
from .goku_full_motion_instruction import validate_compiled_instruction
from .goku_full_motion_qwen import (
    QWEN3_LOGICAL_SHARDS,
    SHARD_RECEIPT_SCHEMA,
    _receipt_digest,
    _strict_jsonl,
    assigned_iids_for_shard,
    validate_output_record,
)
from .qwen_filter import _file_digest


SCHEMA_VERSION = "motive-goku-full-motion-qwen-smoke-gate-v6"
FAILURE_SCHEMA_VERSION = "motive-goku-full-motion-qwen-smoke-gate-failure-v1"
CANARY_ORACLE_SCHEMA = "motive-goku-full-motion-canary-oracle-v2"
DEFAULT_TWO_PERSON_CANARY_IID = "1dbe39537c984690"
DEFAULT_TWO_PERSON_SOURCE_SHA256 = (
    "7e03995017a9c5a5f3522712cfd72ee6a42467524435f9b45f3e528331b07e19"
)
DEFAULT_TWO_PERSON_ANCHOR_SHA256 = (
    "88734f907bf9629196207683b515760a094bfc34a38682722ada977e671f9daf"
)
CANARY_MIN_BBOX_IOU_MILLI = 350
REQUIRED_RECORD_SCHEMA = "goku-full-motion-qwen-record-v6"
REQUIRED_HARD_GATE_SCHEMA = "goku-full-motion-hard-gate-v6"
REQUIRED_PROVENANCE_SCHEMA = "goku-full-motion-qwen-provenance-v6"
REQUIRED_SOURCE_ALIGNMENT_SCHEMA = (
    "motive-goku-full-motion-source-inventory-alignment-v4"
)


class FullMotionSmokeGateError(RuntimeError):
    """The real visual smoke does not authorize the full annotation run."""


def _require_v6_qwen_lineage() -> None:
    """Freeze the smoke authority to the exact two-stage Qwen v6 contract."""

    expected = {
        "record": REQUIRED_RECORD_SCHEMA,
        "hard_gate": REQUIRED_HARD_GATE_SCHEMA,
        "provenance": REQUIRED_PROVENANCE_SCHEMA,
        "source_inventory_alignment": REQUIRED_SOURCE_ALIGNMENT_SCHEMA,
    }
    actual = {
        "record": _qwen.RECORD_SCHEMA,
        "hard_gate": _qwen.HARD_GATE_SCHEMA,
        "provenance": _qwen.PROVENANCE_SCHEMA,
        "source_inventory_alignment": _qwen.SOURCE_INVENTORY_ALIGNMENT_SCHEMA,
    }
    if actual != expected:
        raise FullMotionSmokeGateError(
            "frozen Qwen v6/source-alignment-v4 lineage differs"
        )


def _strict_object(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FullMotionSmokeGateError(f"{context} is not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                FullMotionSmokeGateError(f"non-finite JSON in {context}: {value}")
            ),
            object_pairs_hook=_closed_pairs,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, FullMotionSmokeGateError):
            raise
        raise FullMotionSmokeGateError(f"invalid JSON in {context}: {error}") from error
    if not isinstance(value, dict):
        raise FullMotionSmokeGateError(f"{context} is not an object")
    return value


def _closed_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FullMotionSmokeGateError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _atomic_new(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_failure_receipt(
    *,
    output: str | Path,
    input_path: str | Path,
    qwen_root: str | Path,
    canary_iid: str,
    error: Exception,
) -> dict[str, Any]:
    """Publish a terminal, explicitly non-authorizing smoke failure.

    This receipt intentionally makes no assertion that either input path was
    valid or completely inspected.  It only closes the attempted invocation
    and the gate's failure.  The pass receipt retains a disjoint schema and is
    still the sole object accepted by the full-run handoff.
    """

    receipt = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "status": "fail",
        "authorizes_full_run": False,
        "input_path": str(Path(input_path).expanduser()),
        "qwen_root": str(Path(qwen_root).expanduser()),
        "canary_iid": str(canary_iid),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    receipt["failure_digest"] = object_sha256(receipt)
    _atomic_new(Path(output), receipt)
    return receipt


def _built_in_canary_oracle(
    *, canary_iid: str, selected_row: Mapping[str, Any]
) -> dict[str, Any]:
    if canary_iid != DEFAULT_TWO_PERSON_CANARY_IID:
        raise FullMotionSmokeGateError(
            "a non-default canary requires an explicit oracle JSON"
        )
    return {
        "schema_version": CANARY_ORACLE_SCHEMA,
        "iid": canary_iid,
        "source_video_sha256": DEFAULT_TWO_PERSON_SOURCE_SHA256,
        "anchor_sha256": DEFAULT_TWO_PERSON_ANCHOR_SHA256,
        "expected_dynamic_entities": [
            {
                "oracle_id": "left_person",
                "entity_type": "person",
                "viewer_region": "center_left",
                "i0_bbox_xyxy_1000": [24, 173, 487, 1000],
                "required_motion_component_types": ["gesture"],
            },
            {
                "oracle_id": "right_person",
                "entity_type": "person",
                "viewer_region": "center_right",
                "i0_bbox_xyxy_1000": [479, 234, 945, 1000],
                "required_motion_component_types": ["gesture"],
            },
        ],
        "expected_camera": {"dynamic": False, "motion_class": "locked_off"},
    }


def _validate_canary_oracle(
    value: Mapping[str, Any],
    *,
    canary_iid: str,
    selected_row: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "iid",
        "source_video_sha256",
        "anchor_sha256",
        "expected_dynamic_entities",
        "expected_camera",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise FullMotionSmokeGateError("canary oracle is not a closed schema")
    if value.get("schema_version") != CANARY_ORACLE_SCHEMA:
        raise FullMotionSmokeGateError("canary oracle schema differs")
    if value.get("iid") != canary_iid:
        raise FullMotionSmokeGateError("canary oracle IID differs")
    if (
        value.get("source_video_sha256")
        != selected_row.get("source_video_sha256")
        or value.get("anchor_sha256") != selected_row.get("anchor_sha256")
    ):
        raise FullMotionSmokeGateError("canary oracle media binding differs")
    entities = value.get("expected_dynamic_entities")
    if not isinstance(entities, list) or len(entities) < 2:
        raise FullMotionSmokeGateError(
            "canary oracle must contain both expected moving actors"
        )
    seen_oracles: set[str] = set()
    seen_slots: set[tuple[str, str]] = set()
    for entity in entities:
        if not isinstance(entity, Mapping) or set(entity) != {
            "oracle_id",
            "entity_type",
            "viewer_region",
            "i0_bbox_xyxy_1000",
            "required_motion_component_types",
        }:
            raise FullMotionSmokeGateError(
                "canary oracle entity is not a closed schema"
            )
        oracle_id = entity.get("oracle_id")
        entity_type = entity.get("entity_type")
        region = entity.get("viewer_region")
        bbox = entity.get("i0_bbox_xyxy_1000")
        required_component_types = entity.get(
            "required_motion_component_types"
        )
        if (
            not isinstance(oracle_id, str)
            or not oracle_id
            or entity_type != "person"
            or region not in {"center_left", "center_right"}
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(coordinate) is not int for coordinate in bbox)
            or not (0 <= bbox[0] < bbox[2] <= 1000)
            or not (0 <= bbox[1] < bbox[3] <= 1000)
            or required_component_types != ["gesture"]
        ):
            raise FullMotionSmokeGateError(
                "canary oracle requires named left/right person actors with "
                "an explicit gesture component"
            )
        slot = (str(entity_type), str(region))
        if oracle_id in seen_oracles or slot in seen_slots:
            raise FullMotionSmokeGateError("canary oracle actor slots repeat")
        seen_oracles.add(oracle_id)
        seen_slots.add(slot)
    camera = value.get("expected_camera")
    if camera != {"dynamic": False, "motion_class": "locked_off"}:
        raise FullMotionSmokeGateError(
            "canary oracle must bind the known locked static camera"
        )
    return dict(value)


def _bbox_iou_milli(first: Sequence[int], second: Sequence[int]) -> int:
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return int(round(1000 * intersection / union)) if union else 0


def _match_canary_actors(
    *, census: Mapping[str, Any], oracle: Mapping[str, Any], label: str
) -> list[dict[str, Any]]:
    try:
        census = validate_source_census(census)
    except GokuFullMotionContractError as error:
        raise FullMotionSmokeGateError(
            f"{label} canary source census is invalid: {error}"
        ) from error
    registry = {
        str(item["entity_id"]): item for item in census["i0_entity_registry"]
    }
    units = list(census["dynamic_units"])
    expected = list(oracle["expected_dynamic_entities"])
    if census.get("camera", {}).get("dynamic") is not False or census.get(
        "camera", {}
    ).get("motion_class") != "locked_off":
        raise FullMotionSmokeGateError(
            f"{label} canary camera differs from the locked/static oracle"
        )
    if len(units) != len(expected):
        raise FullMotionSmokeGateError(
            f"{label} canary dynamic set does not exactly match the oracle"
        )
    matches: list[dict[str, Any]] = []
    used_units: set[str] = set()
    for expected_entity in expected:
        candidates: list[tuple[Mapping[str, Any], int]] = []
        for unit in units:
            entity = registry.get(str(unit["entity_id"]))
            components = unit.get("source_motion_components")
            component_types = (
                {
                    str(component.get("component_type"))
                    for component in components
                    if isinstance(component, Mapping)
                }
                if isinstance(components, list)
                else set()
            )
            required_component_types = set(
                expected_entity["required_motion_component_types"]
            )
            iou_milli = (
                0
                if entity is None
                else _bbox_iou_milli(
                    entity["i0_bbox_xyxy_1000"],
                    expected_entity["i0_bbox_xyxy_1000"],
                )
            )
            if (
                entity is not None
                and entity.get("role") == "dynamic_subject"
                and entity.get("entity_type") == expected_entity["entity_type"]
                and entity.get("viewer_region") == expected_entity["viewer_region"]
                and iou_milli >= CANARY_MIN_BBOX_IOU_MILLI
                and required_component_types.issubset(component_types)
            ):
                candidates.append((unit, iou_milli))
        if len(candidates) != 1:
            raise FullMotionSmokeGateError(
                f"{label} canary does not uniquely identify "
                f"{expected_entity['oracle_id']}"
            )
        unit, iou_milli = candidates[0]
        unit_id = str(unit["unit_id"])
        if unit_id in used_units:
            raise FullMotionSmokeGateError(
                f"{label} canary reuses one unit for two oracle actors"
            )
        used_units.add(unit_id)
        matches.append(
            {
                "oracle_id": str(expected_entity["oracle_id"]),
                "unit_id": unit_id,
                "entity_id": str(unit["entity_id"]),
                "bbox_iou_milli": iou_milli,
                "required_motion_component_types": list(
                    expected_entity["required_motion_component_types"]
                ),
                "matched_motion_component_types": sorted(
                    str(component["component_type"])
                    for component in unit["source_motion_components"]
                ),
            }
        )
    if used_units != {str(unit["unit_id"]) for unit in units}:
        raise FullMotionSmokeGateError(f"{label} canary has an extra dynamic unit")
    return matches


def _validate_runtime_closure(
    *,
    receipt: Mapping[str, Any],
    selected_path: Path,
    input_sha256: str,
    shard_index: int,
) -> dict[str, Any]:
    """Validate a shard runtime without trusting receipt-supplied digests.

    The producer's generic receipt validator accepts expected values from its
    caller.  A release gate must instead derive the immutable v6 values from
    its own code and the exact smoke manifest; otherwise a uniformly tampered
    set of receipts can form a self-consistent but false runtime claim.
    """

    _require_v6_qwen_lineage()
    run_config = receipt.get("run_config")
    if not isinstance(run_config, Mapping):
        raise FullMotionSmokeGateError("Qwen shard run config is not closed")
    computed_run_config_digest = object_sha256(run_config)
    if receipt.get("run_config_digest") != computed_run_config_digest:
        raise FullMotionSmokeGateError("Qwen shard run config digest differs")

    implementation_bundle = run_config.get("implementation_bundle")
    expected_implementation_bundle = _qwen._implementation_bundle()
    if (
        not isinstance(implementation_bundle, Mapping)
        or dict(implementation_bundle) != expected_implementation_bundle
        or receipt.get("implementation_digest")
        != object_sha256(expected_implementation_bundle)
    ):
        raise FullMotionSmokeGateError(
            "Qwen shard implementation bundle differs from gate code"
        )

    for field in ("model_path", "model_revision", "transformers_version"):
        if run_config.get(field) != receipt.get(field):
            raise FullMotionSmokeGateError(
                f"Qwen shard run config {field} differs from receipt"
            )
    expected_run_config = _qwen._build_run_config(
        args=argparse.Namespace(
            max_samples=None,
            num_shards=QWEN3_LOGICAL_SHARDS,
            max_new_tokens=_qwen.DEFAULT_MAX_NEW_TOKENS,
            nframes=_qwen.DEFAULT_NFRAMES,
            max_pixels=_qwen.DEFAULT_MAX_PIXELS,
            tile_width=_qwen.DEFAULT_TILE_WIDTH,
            mosaic_columns=_qwen.DEFAULT_MOSAIC_COLUMNS,
            attn_implementation="sdpa",
            allow_download=False,
        ),
        backend=argparse.Namespace(
            model_path=receipt.get("model_path"),
            model_revision=receipt.get("model_revision"),
            transformers_version=receipt.get("transformers_version"),
        ),
        implementation_bundle=expected_implementation_bundle,
    )
    if dict(run_config) != expected_run_config:
        raise FullMotionSmokeGateError("Qwen shard fixed v6 runtime differs")

    computed_config_digest = object_sha256(
        {
            "run_config_digest": computed_run_config_digest,
            "execution_manifest": str(selected_path),
            "execution_manifest_sha256": input_sha256,
            "root": str(selected_path.parent),
            "shard_index": shard_index,
            "num_shards": QWEN3_LOGICAL_SHARDS,
        }
    )
    if receipt.get("config_digest") != computed_config_digest:
        raise FullMotionSmokeGateError("Qwen shard config digest differs")
    return dict(run_config)


def _validate_v6_success_replay(
    *,
    record: Mapping[str, Any],
    selected_row: Mapping[str, Any],
    replayed_media: Mapping[str, Any],
    exact_i0: Any,
    mosaic: Any,
    temporal_triptych: Any,
    temporal_lr_zoom: Any,
    motion_attention: Any,
    authority_grid: Any,
    replayed_proposals: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently close A0, G, A1, and A2 for one hard-pass row."""

    _require_v6_qwen_lineage()
    iid = str(selected_row["iid"])
    if record.get("schema_version") != REQUIRED_RECORD_SCHEMA:
        raise FullMotionSmokeGateError("Qwen smoke record is not v6")

    proposals = _qwen.validate_change_region_proposals(
        record.get("change_region_proposals"), expected_iid=iid
    )
    if proposals != dict(replayed_proposals):
        raise FullMotionSmokeGateError(
            "Qwen smoke change-region proposals differ from media replay"
        )
    proposals_sha = object_sha256(proposals)
    if record.get("change_region_proposals_digest") != proposals_sha:
        raise FullMotionSmokeGateError(
            "Qwen smoke change-region proposal digest differs"
        )

    expected_inventory_prompt = _qwen.build_coverage_authority_inventory_prompt(
        row=selected_row, nframes=_qwen.DEFAULT_NFRAMES
    )
    inventory_prompt_sha = _qwen._text_digest(
        _qwen.COVERAGE_AUTHORITY_INVENTORY_SYSTEM,
        expected_inventory_prompt,
    )
    if (
        record.get("coverage_authority_inventory_prompt_digest")
        != inventory_prompt_sha
    ):
        raise FullMotionSmokeGateError(
            "Qwen smoke A0a inventory prompt digest differs"
        )
    inventory_visual_sha = _qwen._coverage_authority_visual_digest(
        stage="a0a_inventory",
        exact_i0=exact_i0,
        source_mosaic=mosaic,
        source_temporal_triptych=temporal_triptych,
        source_temporal_lr_zoom=temporal_lr_zoom,
        source_motion_attention=motion_attention,
        source_authority_grid=authority_grid,
    )
    if (
        record.get("coverage_authority_inventory_visual_input_digest")
        != inventory_visual_sha
    ):
        raise FullMotionSmokeGateError(
            "Qwen smoke A0a inventory visual digest differs from replay"
        )

    try:
        inventory = _qwen._replay_validated_a0_output(
            record=record,
            stage="coverage_authority_inventory",
            original_system=_qwen.COVERAGE_AUTHORITY_INVENTORY_SYSTEM,
            original_prompt=expected_inventory_prompt,
            expected_visual_input_digest=inventory_visual_sha,
            validator=lambda value: (
                _qwen.validate_coverage_authority_inventory(
                    value, expected_iid=iid
                )
            ),
            canonicalizer=lambda value: (
                _qwen.canonicalize_coverage_authority_inventory_model_output(
                    value, expected_iid=iid
                )
            ),
        )
    except Exception as error:
        raise FullMotionSmokeGateError(
            f"Qwen smoke A0a original/canonical replay differs: {error}"
        ) from error
    inventory_sha = object_sha256(inventory)
    if record.get("coverage_authority_inventory_digest") != inventory_sha:
        raise FullMotionSmokeGateError(
            "Qwen smoke A0a inventory raw/object binding differs"
        )

    expected_assignments_prompt = (
        _qwen.build_coverage_authority_assignments_prompt(
            row=selected_row,
            coverage_authority_inventory=inventory,
            change_region_proposals=proposals,
        )
    )
    assignments_prompt_sha = _qwen._text_digest(
        _qwen.COVERAGE_AUTHORITY_ASSIGNMENTS_SYSTEM,
        expected_assignments_prompt,
    )
    if (
        record.get("coverage_authority_assignments_prompt_digest")
        != assignments_prompt_sha
    ):
        raise FullMotionSmokeGateError(
            "Qwen smoke A0b assignments prompt digest differs"
        )
    assignments_visual_sha = _qwen._coverage_authority_visual_digest(
        stage="a0b_assignments",
        exact_i0=exact_i0,
        source_mosaic=mosaic,
        source_temporal_triptych=temporal_triptych,
        source_temporal_lr_zoom=temporal_lr_zoom,
        source_motion_attention=motion_attention,
        source_authority_grid=authority_grid,
    )
    if (
        record.get("coverage_authority_assignments_visual_input_digest")
        != assignments_visual_sha
    ):
        raise FullMotionSmokeGateError(
            "Qwen smoke A0b assignments visual digest differs from replay"
        )
    try:
        assignments = _qwen._replay_validated_a0_output(
            record=record,
            stage="coverage_authority_assignments",
            original_system=_qwen.COVERAGE_AUTHORITY_ASSIGNMENTS_SYSTEM,
            original_prompt=expected_assignments_prompt,
            expected_visual_input_digest=assignments_visual_sha,
            validator=lambda value: (
                _qwen.validate_coverage_authority_assignments(
                    value,
                    expected_iid=iid,
                    coverage_authority_inventory=inventory,
                    change_region_proposals=proposals,
                )
            ),
            canonicalizer=lambda value: (
                _qwen.canonicalize_coverage_authority_assignments_model_output(
                    value,
                    expected_iid=iid,
                    coverage_authority_inventory=inventory,
                    change_region_proposals=proposals,
                )
            ),
        )
    except Exception as error:
        raise FullMotionSmokeGateError(
            f"Qwen smoke A0b original/canonical replay differs: {error}"
        ) from error
    assignments_sha = object_sha256(assignments)
    if record.get("coverage_authority_assignments_digest") != assignments_sha:
        raise FullMotionSmokeGateError(
            "Qwen smoke A0b assignments raw/object binding differs"
        )
    authority = _qwen.validate_coverage_authority(
        record.get("coverage_authority"),
        expected_iid=iid,
        change_region_proposals=proposals,
    )
    expected_authority = _qwen.build_coverage_authority(
        coverage_authority_inventory=inventory,
        coverage_authority_assignments=assignments,
        change_region_proposals=proposals,
    )
    authority_sha = object_sha256(authority)
    if (
        authority != expected_authority
        or record.get("coverage_authority_digest") != authority_sha
    ):
        raise FullMotionSmokeGateError(
            "Qwen smoke A0 composite/digest differs"
        )

    grounding = _qwen.validate_i0_grounding(
        record.get("i0_grounding"), expected_iid=iid
    )
    census = _qwen.validate_source_census_i0_binding(
        validate_source_census(record.get("source_census")), grounding
    )
    secondary = _qwen.validate_source_census_i0_binding(
        validate_source_census(record.get("secondary_source_census")),
        grounding,
    )
    source_alignment = _qwen.validate_source_inventory_alignment(
        record.get("source_inventory_alignment"),
        primary=census,
        secondary=secondary,
    )
    if source_alignment.get("schema_version") != REQUIRED_SOURCE_ALIGNMENT_SCHEMA:
        raise FullMotionSmokeGateError(
            "Qwen smoke source inventory alignment is not v4"
        )
    source_alignment_sha = object_sha256(source_alignment)
    if record.get("source_inventory_alignment_digest") != source_alignment_sha:
        raise FullMotionSmokeGateError(
            "Qwen smoke source inventory alignment digest differs"
        )
    authority_alignment = _qwen.validate_coverage_authority_alignment(
        record.get("coverage_authority_alignment"),
        coverage_authority=authority,
        change_region_proposals=proposals,
        i0_grounding=grounding,
        primary=census,
        secondary=secondary,
        source_inventory_alignment=source_alignment,
    )
    authority_alignment_sha = object_sha256(authority_alignment)
    if (
        authority_alignment.get("schema_version")
        != _qwen.COVERAGE_AUTHORITY_ALIGNMENT_SCHEMA
        or record.get("coverage_authority_alignment_digest")
        != authority_alignment_sha
    ):
        raise FullMotionSmokeGateError(
            "Qwen smoke A0/G/A1/A2 alignment digest differs"
        )

    try:
        selected_target_raw = _qwen.target_plan_validated_raw(
            record, source_census=census
        )
        (
            _parsed_target_raw,
            replayed_target_plan,
            replayed_target_receipt,
        ) = _qwen._canonicalize_target_plan_raw(
            selected_target_raw,
            stage="smoke replay selected PASS_B target plan",
            source_census=census,
        )
    except Exception as error:
        raise FullMotionSmokeGateError(
            f"Qwen smoke PASS_B selected raw closure differs: {error}"
        ) from error
    if (
        replayed_target_plan != record.get("target_plan")
        or replayed_target_receipt
        != record.get("target_plan_canonicalization")
        or record.get("target_plan_digest")
        != object_sha256(replayed_target_plan)
        or record.get("target_plan_canonicalization_digest")
        != object_sha256(replayed_target_receipt)
    ):
        raise FullMotionSmokeGateError(
            "Qwen smoke PASS_B selected raw/object binding differs"
        )

    hard_gate = _qwen.build_hard_gate(
        i0_grounding=grounding,
        source_census=census,
        source_census_canonicalization=record[
            "source_census_canonicalization"
        ],
        secondary_source_census=secondary,
        secondary_source_census_canonicalization=record[
            "secondary_source_census_canonicalization"
        ],
        source_inventory_alignment=source_alignment,
        target_plan=record["target_plan"],
        target_plan_canonicalization=record[
            "target_plan_canonicalization"
        ],
        compiled_instruction=record["compiled_instruction"],
        coverage_critic=record["coverage_critic"],
        change_region_proposals=proposals,
        coverage_authority=authority,
        coverage_authority_alignment=authority_alignment,
    )
    if (
        hard_gate.get("schema_version") != REQUIRED_HARD_GATE_SCHEMA
        or hard_gate.get("decision") != "pass"
        or record.get("hard_gate") != hard_gate
    ):
        raise FullMotionSmokeGateError(
            "Qwen smoke v6 hard-gate replay differs"
        )
    result_sha = object_sha256(_qwen.qwen_result_payload(record))
    provenance_sha = _qwen.qwen_provenance_digest(record)
    if (
        record.get("result_digest") != result_sha
        or record.get("provenance_digest") != provenance_sha
    ):
        raise FullMotionSmokeGateError(
            "Qwen smoke v6 result/provenance digest differs"
        )

    return {
        "iid": iid,
        "record_schema_version": REQUIRED_RECORD_SCHEMA,
        "provenance_schema_version": REQUIRED_PROVENANCE_SCHEMA,
        "hard_gate_schema_version": REQUIRED_HARD_GATE_SCHEMA,
        "change_region_proposals_schema_version": (
            _qwen.CHANGE_REGION_PROPOSALS_SCHEMA
        ),
        "coverage_authority_schema_version": _qwen.COVERAGE_AUTHORITY_SCHEMA,
        "coverage_authority_inventory_schema_version": (
            _qwen.COVERAGE_AUTHORITY_INVENTORY_SCHEMA
        ),
        "coverage_authority_assignments_schema_version": (
            _qwen.COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA
        ),
        "source_inventory_alignment_schema_version": (
            REQUIRED_SOURCE_ALIGNMENT_SCHEMA
        ),
        "coverage_authority_alignment_schema_version": (
            _qwen.COVERAGE_AUTHORITY_ALIGNMENT_SCHEMA
        ),
        "media_verification_sha256": object_sha256(replayed_media),
        "change_region_proposals_sha256": proposals_sha,
        "coverage_authority_inventory_prompt_sha256": inventory_prompt_sha,
        "coverage_authority_inventory_visual_input_sha256": (
            inventory_visual_sha
        ),
        "coverage_authority_inventory_sha256": inventory_sha,
        "coverage_authority_assignments_prompt_sha256": (
            assignments_prompt_sha
        ),
        "coverage_authority_assignments_visual_input_sha256": (
            assignments_visual_sha
        ),
        "coverage_authority_assignments_sha256": assignments_sha,
        "coverage_authority_sha256": authority_sha,
        "i0_grounding_sha256": object_sha256(grounding),
        "primary_source_census_sha256": object_sha256(census),
        "secondary_source_census_sha256": object_sha256(secondary),
        "source_inventory_alignment_sha256": source_alignment_sha,
        "coverage_authority_alignment_sha256": authority_alignment_sha,
        "hard_gate_sha256": object_sha256(hard_gate),
        "result_sha256": result_sha,
        "provenance_sha256": provenance_sha,
    }


def gate_smoke(
    *,
    input_path: str | Path,
    qwen_root: str | Path,
    output: str | Path,
    canary_iid: str,
    canary_oracle: Mapping[str, Any] | None = None,
    minimum_hard_passes: int = 3,
    minimum_canary_dynamic_units: int = 2,
) -> dict[str, Any]:
    _require_v6_qwen_lineage()
    selected_path = Path(input_path).expanduser().resolve(strict=True)
    root = Path(qwen_root).expanduser().resolve(strict=True)
    if selected_path.is_symlink() or not selected_path.is_file():
        raise FullMotionSmokeGateError("smoke input is not a regular file")
    if root.is_symlink() or not root.is_dir():
        raise FullMotionSmokeGateError("Qwen smoke root is not a plain directory")
    if minimum_hard_passes <= 0 or minimum_canary_dynamic_units < 2:
        raise FullMotionSmokeGateError("smoke thresholds are invalid")
    selected_rows = _strict_jsonl(selected_path)
    if len(selected_rows) != 8:
        raise FullMotionSmokeGateError("the semantic smoke must contain exactly 8 rows")
    selected_by_iid = {str(row["iid"]): row for row in selected_rows}
    if canary_iid not in selected_by_iid:
        raise FullMotionSmokeGateError("required semantic canary is absent")
    oracle = _validate_canary_oracle(
        canary_oracle
        if canary_oracle is not None
        else _built_in_canary_oracle(
            canary_iid=canary_iid,
            selected_row=selected_by_iid[canary_iid],
        ),
        canary_iid=canary_iid,
        selected_row=selected_by_iid[canary_iid],
    )
    input_sha = _file_digest(selected_path)
    all_records: dict[str, dict[str, Any]] = {}
    semantic_bindings: dict[str, dict[str, Any]] = {}
    common: dict[str, Any] | None = None
    receipt_bindings: list[dict[str, Any]] = []
    for shard_index in range(QWEN3_LOGICAL_SHARDS):
        shard = root / f"qwen_shard_{shard_index:03d}.jsonl"
        receipt_path = root / f"qwen_shard_{shard_index:03d}.receipt.json"
        receipt = _strict_object(receipt_path, context=f"shard {shard_index} receipt")
        if receipt.get("schema_version") != SHARD_RECEIPT_SCHEMA:
            raise FullMotionSmokeGateError("Qwen shard receipt schema differs")
        if receipt.get("status") != "complete":
            raise FullMotionSmokeGateError("Qwen shard receipt is not terminal")
        if receipt.get("receipt_digest") != _receipt_digest(receipt):
            raise FullMotionSmokeGateError("Qwen shard receipt digest differs")
        assigned = assigned_iids_for_shard(
            selected_rows,
            shard_index=shard_index,
            num_shards=QWEN3_LOGICAL_SHARDS,
            max_samples=None,
        )
        if receipt.get("assigned_iids") != assigned:
            raise FullMotionSmokeGateError("Qwen shard IID assignment differs")
        if (
            receipt.get("execution_manifest") != str(selected_path)
            or receipt.get("execution_manifest_sha256") != input_sha
            or receipt.get("shard_index") != shard_index
            or receipt.get("num_shards") != QWEN3_LOGICAL_SHARDS
            or receipt.get("root") != str(selected_path.parent)
        ):
            raise FullMotionSmokeGateError("Qwen shard input binding differs")
        validated_run_config = _validate_runtime_closure(
            receipt=receipt,
            selected_path=selected_path,
            input_sha256=input_sha,
            shard_index=shard_index,
        )
        output_binding = receipt.get("output")
        if not isinstance(output_binding, Mapping):
            raise FullMotionSmokeGateError("Qwen shard output binding is absent")
        if (
            shard.is_symlink()
            or not shard.is_file()
            or output_binding.get("path") != str(shard)
            or output_binding.get("sha256") != _file_digest(shard)
            or output_binding.get("bytes") != shard.stat().st_size
        ):
            raise FullMotionSmokeGateError("Qwen shard output bytes differ")
        shard_rows = _strict_jsonl(shard, allow_empty=True)
        if [str(row.get("iid")) for row in shard_rows] != assigned:
            raise FullMotionSmokeGateError("Qwen shard row order differs")
        stable = {
            "implementation_digest": receipt.get("implementation_digest"),
            "run_config_digest": receipt.get("run_config_digest"),
            "run_config": validated_run_config,
            "model_path": receipt.get("model_path"),
            "model_revision": receipt.get("model_revision"),
            "transformers_version": receipt.get("transformers_version"),
        }
        if common is None:
            common = stable
        elif stable != common:
            raise FullMotionSmokeGateError("Qwen shard runtime/config differs")
        expected_bindings = {
            "execution_manifest": str(selected_path),
            "execution_manifest_sha256": input_sha,
            "shard_index": shard_index,
            "num_shards": QWEN3_LOGICAL_SHARDS,
            "implementation_digest": receipt["implementation_digest"],
            "config_digest": receipt["config_digest"],
            "run_config_digest": receipt["run_config_digest"],
            "model_path": receipt["model_path"],
            "model_revision": receipt["model_revision"],
            "transformers_version": receipt["transformers_version"],
        }
        for row in shard_rows:
            iid = str(row["iid"])
            selected_row = selected_by_iid[iid]
            try:
                validate_output_record(
                    row,
                    selected_row=selected_row,
                    expected_bindings=expected_bindings,
                )
            except Exception as error:
                raise FullMotionSmokeGateError(
                    f"Qwen smoke record {iid} fails v6 validation: {error}"
                ) from error
            source_path = _qwen._resolve_path(
                str(selected_row["resolved_src_video"]), selected_path.parent
            )
            anchor_path = _qwen._resolve_path(
                str(selected_row["resolved_anchor_image"]), selected_path.parent
            )
            try:
                replayed_media = _qwen.verify_exact_i0_binding(
                    source_path=source_path,
                    anchor_path=anchor_path,
                    source_sha256=str(selected_row["source_video_sha256"]),
                    anchor_sha256=str(selected_row["anchor_sha256"]),
                )
            except Exception as error:
                raise FullMotionSmokeGateError(
                    f"Qwen smoke exact media replay failed for {iid}: {error}"
                ) from error
            if row.get("media_verification") != replayed_media:
                raise FullMotionSmokeGateError(
                    "Qwen smoke exact media verification binding differs"
                )
            if row.get("status") != "ok":
                all_records[iid] = dict(row)
                continue
            (
                exact_i0,
                mosaic,
                temporal_triptych,
                temporal_lr_zoom,
                motion_attention,
                _,
            ) = _qwen._build_visuals(
                source_path=source_path,
                anchor_path=anchor_path,
                nframes=_qwen.DEFAULT_NFRAMES,
                max_pixels=_qwen.DEFAULT_MAX_PIXELS,
                tile_width=_qwen.DEFAULT_TILE_WIDTH,
                mosaic_columns=_qwen.DEFAULT_MOSAIC_COLUMNS,
            )
            authority_grid, replayed_proposals = (
                _qwen._build_authority_grid_and_proposals(
                    source_path=source_path,
                    exact_i0=exact_i0,
                    iid=iid,
                    max_pixels=_qwen.DEFAULT_MAX_PIXELS,
                )
            )
            expected_i0_visual_digest = _qwen._visual_digest(
                (("exact_i0_only", exact_i0),)
            )
            if (
                row.get("i0_grounding_visual_input_digest")
                != expected_i0_visual_digest
            ):
                raise FullMotionSmokeGateError(
                    "Qwen smoke exact-I0 grounding visual digest differs "
                    "from reconstruction"
                )
            try:
                semantic_bindings[iid] = _validate_v6_success_replay(
                    record=row,
                    selected_row=selected_row,
                    replayed_media=replayed_media,
                    exact_i0=exact_i0,
                    mosaic=mosaic,
                    temporal_triptych=temporal_triptych,
                    temporal_lr_zoom=temporal_lr_zoom,
                    motion_attention=motion_attention,
                    authority_grid=authority_grid,
                    replayed_proposals=replayed_proposals,
                )
            except FullMotionSmokeGateError:
                raise
            except Exception as error:
                raise FullMotionSmokeGateError(
                    f"Qwen smoke record {iid} fails independent "
                    f"A0/G/A1/A2 replay: {error}"
                ) from error
            grounded_temporal_zoom = _qwen._build_grounded_temporal_zoom(
                source_path=source_path,
                exact_i0=exact_i0,
                i0_grounding=row["i0_grounding"],
                max_pixels=_qwen.DEFAULT_MAX_PIXELS,
                tile_width=_qwen.DEFAULT_TILE_WIDTH,
            )
            expected_visual_digest = _qwen._visual_digest(
                (
                    ("exact_i0", exact_i0),
                    ("source_mosaic", mosaic),
                    ("source_temporal_triptych", temporal_triptych),
                    ("source_temporal_lr_zoom", temporal_lr_zoom),
                    ("source_motion_attention", motion_attention),
                    (
                        "source_grounded_temporal_zoom",
                        grounded_temporal_zoom,
                    ),
                )
            )
            if row.get("visual_input_digest") != expected_visual_digest:
                raise FullMotionSmokeGateError(
                    "Qwen smoke visual input digest differs from reconstruction"
                )
            all_records[iid] = dict(row)
        receipt_bindings.append(
            {
                "path": str(receipt_path),
                "sha256": _file_digest(receipt_path),
                "output_sha256": output_binding["sha256"],
            }
        )
    if set(all_records) != set(selected_by_iid):
        raise FullMotionSmokeGateError("smoke record coverage differs")
    hard_passes = [
        iid
        for iid, row in all_records.items()
        if row.get("status") == "ok"
        and row.get("pipeline_decision") == "pass"
        and isinstance(row.get("hard_gate"), Mapping)
        and row["hard_gate"].get("decision") == "pass"
    ]
    if set(semantic_bindings) != set(hard_passes):
        raise FullMotionSmokeGateError(
            "smoke v6 semantic replay coverage differs from hard passes"
        )
    if len(hard_passes) < minimum_hard_passes:
        raise FullMotionSmokeGateError(
            f"Qwen smoke hard-pass shortfall: {len(hard_passes)}/{minimum_hard_passes}"
        )
    canary = all_records[canary_iid]
    if canary_iid not in hard_passes:
        raise FullMotionSmokeGateError("two-person semantic canary did not hard-pass")
    census = canary["source_census"]
    secondary_census = canary["secondary_source_census"]
    plan = canary["target_plan"]
    compiled = validate_compiled_instruction(
        canary["compiled_instruction"],
        source_census=census,
        target_plan=plan,
    )
    primary_actor_matches = _match_canary_actors(
        census=census, oracle=oracle, label="primary"
    )
    secondary_actor_matches = _match_canary_actors(
        census=secondary_census, oracle=oracle, label="secondary"
    )
    dynamic_ids = [item["unit_id"] for item in primary_actor_matches]
    if len(dynamic_ids) < minimum_canary_dynamic_units:
        raise FullMotionSmokeGateError("canary oracle has too few moving people")
    target_ids = {
        str(item["unit_id"]) for item in plan["dynamic_unit_targets"]
    }
    if set(dynamic_ids) != target_ids:
        raise FullMotionSmokeGateError(
            "canary target does not cover both oracle-bound actors"
        )
    if set(compiled["entity_clauses"]) != {
        *dynamic_ids,
        *(str(unit["unit_id"]) for unit in census["static_salient_people"]),
    }:
        raise FullMotionSmokeGateError("canary compiled entity coverage differs")
    if not compiled.get("camera_clause"):
        raise FullMotionSmokeGateError("canary camera clause is missing")
    canary_binding = semantic_bindings[canary_iid]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "input": {
            "path": str(selected_path),
            "sha256": input_sha,
            "rows": len(selected_rows),
        },
        "qwen_root": str(root),
        "qwen_lineage": {
            "record": REQUIRED_RECORD_SCHEMA,
            "hard_gate": REQUIRED_HARD_GATE_SCHEMA,
            "provenance": REQUIRED_PROVENANCE_SCHEMA,
            "source_inventory_alignment": REQUIRED_SOURCE_ALIGNMENT_SCHEMA,
            "change_region_proposals": _qwen.CHANGE_REGION_PROPOSALS_SCHEMA,
            "coverage_authority": _qwen.COVERAGE_AUTHORITY_SCHEMA,
            "coverage_authority_inventory": (
                _qwen.COVERAGE_AUTHORITY_INVENTORY_SCHEMA
            ),
            "coverage_authority_assignments": (
                _qwen.COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA
            ),
            "coverage_authority_allowed_owner_map": (
                _qwen.COVERAGE_AUTHORITY_ALLOWED_OWNER_MAP_SCHEMA
            ),
            "coverage_authority_alignment": (
                _qwen.COVERAGE_AUTHORITY_ALIGNMENT_SCHEMA
            ),
        },
        "qwen_runtime": common,
        "receipt_bindings": receipt_bindings,
        "hard_passes": len(hard_passes),
        "hard_pass_iids": sorted(hard_passes),
        "hard_pass_bindings": [
            semantic_bindings[iid] for iid in sorted(hard_passes)
        ],
        "canary": {
            "iid": canary_iid,
            "oracle_sha256": object_sha256(oracle),
            "primary_actor_matches": primary_actor_matches,
            "secondary_actor_matches": secondary_actor_matches,
            "dynamic_unit_ids": dynamic_ids,
            "compiled_instruction_sha256": compiled["instruction_sha256"],
            "edit_instruction": compiled["edit_instruction"],
            "camera_clause": compiled["camera_clause"],
            "qwen_record_schema_version": canary["schema_version"],
            "qwen_provenance_schema_version": REQUIRED_PROVENANCE_SCHEMA,
            "qwen_i0_grounding_schema_version": canary["i0_grounding"][
                "schema_version"
            ],
            "qwen_i0_grounding_sha256": canary["i0_grounding_digest"],
            "qwen_i0_grounding_prompt_sha256": canary[
                "i0_grounding_prompt_digest"
            ],
            "qwen_i0_grounding_visual_input_sha256": canary[
                "i0_grounding_visual_input_digest"
            ],
            "qwen_hard_gate_schema_version": canary["hard_gate"][
                "schema_version"
            ],
            "change_region_proposals_schema_version": canary_binding[
                "change_region_proposals_schema_version"
            ],
            "coverage_authority_schema_version": canary_binding[
                "coverage_authority_schema_version"
            ],
            "coverage_authority_inventory_schema_version": canary_binding[
                "coverage_authority_inventory_schema_version"
            ],
            "coverage_authority_assignments_schema_version": canary_binding[
                "coverage_authority_assignments_schema_version"
            ],
            "source_inventory_alignment_schema_version": canary_binding[
                "source_inventory_alignment_schema_version"
            ],
            "coverage_authority_alignment_schema_version": canary_binding[
                "coverage_authority_alignment_schema_version"
            ],
            "change_region_proposals_sha256": canary_binding[
                "change_region_proposals_sha256"
            ],
            "coverage_authority_inventory_prompt_sha256": canary_binding[
                "coverage_authority_inventory_prompt_sha256"
            ],
            "coverage_authority_inventory_visual_input_sha256": canary_binding[
                "coverage_authority_inventory_visual_input_sha256"
            ],
            "coverage_authority_inventory_sha256": canary_binding[
                "coverage_authority_inventory_sha256"
            ],
            "coverage_authority_assignments_prompt_sha256": canary_binding[
                "coverage_authority_assignments_prompt_sha256"
            ],
            "coverage_authority_assignments_visual_input_sha256": canary_binding[
                "coverage_authority_assignments_visual_input_sha256"
            ],
            "coverage_authority_assignments_sha256": canary_binding[
                "coverage_authority_assignments_sha256"
            ],
            "coverage_authority_sha256": canary_binding[
                "coverage_authority_sha256"
            ],
            "source_inventory_alignment_sha256": canary_binding[
                "source_inventory_alignment_sha256"
            ],
            "coverage_authority_alignment_sha256": canary_binding[
                "coverage_authority_alignment_sha256"
            ],
            "qwen_hard_gate_sha256": canary_binding[
                "hard_gate_sha256"
            ],
            "source_census_canonicalization_sha256": canary[
                "source_census_canonicalization_digest"
            ],
            "secondary_source_census_canonicalization_sha256": canary[
                "secondary_source_census_canonicalization_digest"
            ],
            "target_plan_canonicalization_sha256": canary[
                "target_plan_canonicalization_digest"
            ],
            "qwen_result_digest": canary["result_digest"],
            "qwen_provenance_digest": canary["provenance_digest"],
        },
    }
    result["gate_digest"] = object_sha256(result)
    _atomic_new(Path(output), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--qwen-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--canary-iid", required=True)
    parser.add_argument("--canary-oracle-json", type=Path)
    parser.add_argument("--minimum-hard-passes", type=int, default=3)
    parser.add_argument("--minimum-canary-dynamic-units", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        canary_oracle = (
            None
            if args.canary_oracle_json is None
            else _strict_object(
                args.canary_oracle_json.expanduser().resolve(strict=True),
                context="canary oracle",
            )
        )
        result = gate_smoke(
            input_path=args.input,
            qwen_root=args.qwen_root,
            output=args.output,
            canary_iid=args.canary_iid,
            canary_oracle=canary_oracle,
            minimum_hard_passes=args.minimum_hard_passes,
            minimum_canary_dynamic_units=args.minimum_canary_dynamic_units,
        )
    except (FullMotionSmokeGateError, OSError, ValueError) as error:
        receipt = _write_failure_receipt(
            output=args.output,
            input_path=args.input,
            qwen_root=args.qwen_root,
            canary_iid=args.canary_iid,
            error=error,
        )
        print(
            "[goku-full-motion-smoke] "
            f"fail={receipt['failure_digest']} error={receipt['error']}",
            file=sys.stderr,
        )
        return 2
    print(
        "[goku-full-motion-smoke] "
        f"pass={result['hard_passes']}/8 canary={args.canary_iid}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
