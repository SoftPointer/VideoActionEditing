#!/usr/bin/env python3
"""Build the exact optimizer-free 4-cell/8-clip cross-anchor plan.

This module deliberately does not reinterpret the compact6 ``full_topup20``
population as the execution target.  It authenticates that larger registry,
rebuilds its deterministic authoring surface, and then selects exactly the
four seed cells preregistered by ``full30_action_minimal_cross_anchor_topup4``.
The resulting candidate envelopes are compatible with the frozen PAIR-v5
renderer, while this module (rather than PAIR's two-split bank validator)
owns the honest fit-only execution closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

import mosaic_event_population_authoring as population
import pair_v5_t2v_calibration_bank_spec as pair_contract


SELECTION_SCHEMA = "bernini-full30-action-minimal-cross-anchor-selection-v2"
ROOT_SCHEMA = "bernini-full30-action-minimal-cross-anchor-root-v2"
PLAN_SCHEMA = "bernini-full30-action-minimal-cross-anchor-plan-v2"
SELECTION_ID = "full30-minimal-cross-anchor-action-incomplete-topup4-v2"
PARENT_STAGE_ID = "full_topup20"
PLAN_ID = "full30-minimal-cross-anchor-action-incomplete-topup4-exact8-v2"
SELECTION_FILE_SHA256 = (
    "72a1d58ede5381f57d2fa8ef895a7e9d5c11b3872e87ecbc3e08fec0cc5ef38e"
)
PARENT_REGISTRY_FILE_SHA256 = (
    "71906510d162e6626338b5785fd1cf55b437de5ba77d9b9b122ad761694f8e62"
)
GEOMETRY_SOURCE_SHA256 = (
    "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
)
GROUP_LAYOUT = (
    ("sp4-a", [0, 1, 2, 3]),
    ("sp4-b", [4, 5, 6, 7]),
)
REQUIRED_BRANCH_ORDER = ("action", "incomplete")
DIAGNOSTIC_ONLY_BRANCHES = tuple(
    branch
    for branch in pair_contract.MACE_BRANCH_ORDER
    if branch not in REQUIRED_BRANCH_ORDER
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")

_SELECTION_FIELDS = {
    "schema_version",
    "selection_id",
    "governing_contract",
    "governing_preregistration",
    "parent_registry",
    "parent_registry_sha256",
    "geometry_source_video_sha256",
    "selection_frozen_before_generation",
    "selection_rule",
    "selected_seed_cells",
    "existing_cross_anchor_reuse",
    "formal_seed_cell_count",
    "branches_per_seed_cell",
    "formal_mp4_count",
    "diagnostic_only_rendered_branches_not_required",
    "same_state_noop_camera_appearance_and_wrong_control_forwards_still_required",
    "optimizer_updates",
    "generated_media_may_train_editor",
    "generated_media_may_be_editor_target",
    "generated_media_may_be_editor_condition",
    "representation_admission_still_requires_per_teacher_cell_same_state_six_sigma_sidecars",
    "physical_anchor_video_reuse_only_with_exact_intrinsic_identity",
    "sidecar_nuisance_evidence_and_receipt_reuse_forbidden",
    "generalization_claim_authorized",
}
_SELECTED_CELL_FIELDS = {
    "family_id",
    "identity_scene_id",
    "analysis_split",
    "seed",
    "required_branches",
}
_ROOT_FIELDS = {
    "schema_version",
    "selection",
    "parent_registry",
    "parent_stage_id",
    "sampling_contract",
    "semantic_input_closure",
    "artifact_use_contract",
    "geometry_contract",
    "analysis_split",
    "branch_order",
    "seed_cell_count",
    "candidate_count",
    "groups",
    "audit_requests",
    "execution_contract",
}
_ROOT_GROUP_FIELDS = {"group_id", "visible_gpus", "cells", "candidates"}
_ROOT_CELL_FIELDS = {
    "cell_id",
    "family_id",
    "identity_scene_id",
    "analysis_split",
    "seed",
    "actor_group_id",
    "scene_group_id",
    "action_group_id",
    "candidate_ids",
}
_PLAN_FIELDS = {
    "schema_version",
    "plan_id",
    "selection",
    "parent_registry",
    "root_spec",
    "analysis_split",
    "generation_invocation_count",
    "seed_cell_count",
    "branch_order",
    "tasks",
    "cell_proofs",
    "shards",
    "execution_contract",
    "plan_digest",
}


class MinimalCrossAnchorPlanError(RuntimeError):
    """Raised before ambiguous or broadened topup4 input is accepted."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise MinimalCrossAnchorPlanError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MinimalCrossAnchorPlanError(message)


def _closed(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    _require(set(value) == fields, f"{label} fields differ")
    return dict(value)


def _sha256(value: Any, label: str) -> str:
    _require(type(value) is str and _SHA256_RE.fullmatch(value) is not None, f"{label} differs")
    return value


def _safe_id(value: Any, label: str) -> str:
    _require(type(value) is str and _SAFE_ID_RE.fullmatch(value) is not None, f"{label} differs")
    return value


def _plain_file(path: str | Path, label: str) -> Path:
    value = Path(path)
    _require(
        value.is_absolute()
        and value.is_file()
        and not value.is_symlink()
        and value.resolve(strict=True) == value,
        f"{label} must be an absolute plain file",
    )
    return value


def _load_json(path: str | Path, label: str, expected_sha256: str) -> tuple[dict[str, Any], Path, str]:
    expected = _sha256(expected_sha256, f"{label} expected SHA-256")
    resolved = _plain_file(path, label)
    raw = resolved.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    _require(observed == expected, f"{label} file SHA-256 differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MinimalCrossAnchorPlanError(f"{label} is not JSON") from error
    _require(type(value) is dict, f"{label} must be an object")
    return dict(value), resolved, observed


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    _require(path.is_absolute() and path.parent.is_dir() and not path.parent.is_symlink(), "output parent differs")
    _require(not path.exists() and not path.is_symlink(), f"refusing output reuse: {path}")
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short output write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def validate_selection(value: Any) -> dict[str, Any]:
    selection = _closed(value, _SELECTION_FIELDS, "selection")
    _require(selection["schema_version"] == SELECTION_SCHEMA, "selection schema differs")
    _require(selection["selection_id"] == SELECTION_ID, "selection identity differs")
    _require(selection["governing_contract"] == "ACTION-BOX-20260814-R2", "governing contract differs")
    _require(selection["parent_registry_sha256"] == _sha256(selection["parent_registry_sha256"], "parent registry SHA-256"), "parent registry SHA differs")
    _require(selection["geometry_source_video_sha256"] == GEOMETRY_SOURCE_SHA256, "geometry source SHA-256 differs")
    _require(selection["selection_frozen_before_generation"] is True, "selection was not frozen before generation")
    _require(selection["formal_seed_cell_count"] == 4, "selection must contain four seed cells")
    _require(selection["branches_per_seed_cell"] == 2 and selection["formal_mp4_count"] == 8, "selection count differs")
    _require(
        selection["diagnostic_only_rendered_branches_not_required"]
        == list(DIAGNOSTIC_ONLY_BRANCHES),
        "diagnostic-only rendered branch list differs",
    )
    _require(
        selection[
            "same_state_noop_camera_appearance_and_wrong_control_forwards_still_required"
        ]
        is True,
        "same-state controls were weakened",
    )
    _require(selection["optimizer_updates"] == 0, "selection must remain optimizer-free")
    for field in (
        "generated_media_may_train_editor",
        "generated_media_may_be_editor_target",
        "generated_media_may_be_editor_condition",
        "generalization_claim_authorized",
    ):
        _require(selection[field] is False, f"selection exceeds authority: {field}")
    for field in (
        "representation_admission_still_requires_per_teacher_cell_same_state_six_sigma_sidecars",
        "physical_anchor_video_reuse_only_with_exact_intrinsic_identity",
        "sidecar_nuisance_evidence_and_receipt_reuse_forbidden",
    ):
        _require(selection[field] is True, f"selection safety gate differs: {field}")
    rows = selection["selected_seed_cells"]
    _require(type(rows) is list and len(rows) == 4, "selected seed cells differ")
    normalized = []
    keys: set[tuple[str, str, int]] = set()
    for raw in rows:
        row = _closed(raw, _SELECTED_CELL_FIELDS, "selected seed cell")
        family_id = _safe_id(row["family_id"], "family_id")
        identity_scene_id = _safe_id(row["identity_scene_id"], "identity_scene_id")
        _require(row["analysis_split"] == "fit", "topup4 is fit-only")
        _require(type(row["seed"]) is int and 0 <= row["seed"] < 2**63, "seed differs")
        _require(row["required_branches"] == list(REQUIRED_BRANCH_ORDER), "branch order differs")
        key = (family_id, identity_scene_id, row["seed"])
        _require(key not in keys, "selected seed cell is duplicated")
        keys.add(key)
        normalized.append({**row, "family_id": family_id, "identity_scene_id": identity_scene_id})
    _require(len({row["family_id"] for row in normalized}) == 4, "topup4 requires four distinct action events")
    return {**selection, "selected_seed_cells": normalized}


def load_selection(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], Path, str]:
    _require(
        expected_sha256 == SELECTION_FILE_SHA256,
        "selection file authority differs",
    )
    raw, resolved, observed = _load_json(path, "selection", expected_sha256)
    return validate_selection(raw), resolved, observed


def _selected_surface(selection: Mapping[str, Any], registry: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        bundle = population.build_stage_bundle(registry, stage_id=PARENT_STAGE_ID)
    except population.MosaicEventPopulationError as error:
        raise MinimalCrossAnchorPlanError(str(error)) from error
    desired = {
        (row["family_id"], row["identity_scene_id"], row["seed"]): row
        for row in selection["selected_seed_cells"]
    }
    cells = []
    for cell in bundle["authoring"]["cells"]:
        matching = [
            key for key in desired
            if key[0] == cell["action_family_id"]
            and key[1] in cell["iid"]
            and key[2] == cell["seed"]
        ]
        if matching:
            _require(len(matching) == 1, "selection matches multiple population cells")
            key = matching[0]
            cells.append({**cell, "identity_scene_id": key[1]})
    _require(len(cells) == 4, "parent registry does not contain the exact selected seed cells")
    _require(
        {(row["action_family_id"], row["identity_scene_id"], row["seed"]) for row in cells}
        == set(desired),
        "selected population closure differs",
    )
    requests_by_cell: dict[str, list[dict[str, Any]]] = {}
    for request in bundle["audit_requests"]["candidate_requests"]:
        requests_by_cell.setdefault(request["cell_id"], []).append(request)
    requests: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = f"cell-{cell['iid']}-s{cell['seed']}"
        rows = requests_by_cell.get(cell_id, [])
        rows = [
            row
            for row in rows
            if row["requested_semantic_branch"] in REQUIRED_BRANCH_ORDER
        ]
        _require([row["requested_semantic_branch"] for row in rows] == list(REQUIRED_BRANCH_ORDER), "audit request branch closure differs")
        requests.extend(rows)
    _require(len(requests) == 8, "selected audit request count differs")
    return cells, requests


def _pair_candidate(cell: Mapping[str, Any], branch: str) -> dict[str, Any]:
    caption = " ".join((cell["scene_caption"], cell["branch_descriptions"][branch], cell["camera_caption"]))
    candidate_id = f"topup4-{cell['iid']}-{branch}"
    candidate = {
        "candidate_id": _safe_id(candidate_id, "candidate_id"),
        "analysis_split": "fit",
        "action_family_id": cell["action_family_id"],
        "calibration_group_id": f"cell-{cell['iid']}-s{cell['seed']}",
        "prompt_group_id": f"{cell['actor_group_id']}--{cell['scene_group_id']}",
        "action_family_group_id": cell["action_group_id"],
        "actor_group_id": cell["actor_group_id"],
        "scene_group_id": cell["scene_group_id"],
        "action_group_id": cell["action_group_id"],
        "geometry_source_video": cell["geometry_source_video"],
        "geometry_source_video_sha256": GEOMETRY_SOURCE_SHA256,
        "geometry_contract": pair_contract.GEOMETRY_CONTRACT,
        "semantic_branch": branch,
        "full_t2v_caption": caption,
        "full_t2v_caption_utf8_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
        "caption_contract": pair_contract.CAPTION_CONTRACT,
        "seed": cell["seed"],
    }
    try:
        return pair_contract.validate_candidate(candidate)
    except pair_contract.PairT2VCalibrationSpecError as error:
        raise MinimalCrossAnchorPlanError(str(error)) from error


def build_root_value(
    *, selection: Mapping[str, Any], selection_path: Path, selection_sha256: str,
    registry: Mapping[str, Any], registry_path: Path, registry_sha256: str,
) -> dict[str, Any]:
    cells, audit_requests = _selected_surface(selection, registry)
    grouped_cells = (cells[:2], cells[2:])
    groups = []
    for (group_id, visible_gpus), group_cells in zip(GROUP_LAYOUT, grouped_cells):
        candidates = [
            _pair_candidate(cell, branch)
            for cell in group_cells
            for branch in REQUIRED_BRANCH_ORDER
        ]
        cell_rows = []
        for cell in group_cells:
            cell_id = f"cell-{cell['iid']}-s{cell['seed']}"
            cell_rows.append(
                {
                    "cell_id": cell_id,
                    "family_id": cell["action_family_id"],
                    "identity_scene_id": cell["identity_scene_id"],
                    "analysis_split": "fit",
                    "seed": cell["seed"],
                    "actor_group_id": cell["actor_group_id"],
                    "scene_group_id": cell["scene_group_id"],
                    "action_group_id": cell["action_group_id"],
                    "candidate_ids": [
                        row["candidate_id"]
                        for row in candidates
                        if row["calibration_group_id"] == cell_id
                    ],
                }
            )
        groups.append(
            {
                "group_id": group_id,
                "visible_gpus": visible_gpus,
                "cells": cell_rows,
                "candidates": candidates,
            }
        )
    return {
        "schema_version": ROOT_SCHEMA,
        "selection": {"path": str(selection_path), "file_sha256": selection_sha256, "selection_id": SELECTION_ID},
        "parent_registry": {"path": str(registry_path), "file_sha256": registry_sha256, "registry_id": registry["registry_id"]},
        "parent_stage_id": PARENT_STAGE_ID,
        "sampling_contract": pair_contract.SAMPLING_CONTRACT,
        "semantic_input_closure": pair_contract.SEMANTIC_INPUT_CLOSURE,
        "artifact_use_contract": pair_contract.ARTIFACT_USE_CONTRACT,
        "geometry_contract": pair_contract.GEOMETRY_CONTRACT,
        "analysis_split": "fit",
        "branch_order": list(REQUIRED_BRANCH_ORDER),
        "seed_cell_count": 4,
        "candidate_count": 8,
        "groups": groups,
        "audit_requests": audit_requests,
        "execution_contract": {
            "topology": "one_model_replica_world4_dp1_sp4",
            "groups_execute_strictly_serial": True,
            "candidate_order": "selection_order_then_mace_branch_order",
            "generated_media_role": "representation_authoring_evidence_only",
            "generated_media_is_editor_input_or_target": False,
            "independent_full81_review_required": True,
            "same_state_six_sigma_materialization_required_after_review": True,
            "diagnostic_only_rendered_branches_not_optimizer_required": list(
                DIAGNOSTIC_ONLY_BRANCHES
            ),
            "same_state_noop_camera_appearance_and_wrong_control_forwards_required": True,
            "optimizer_authorized": False,
        },
    }


def validate_root_value(value: Any) -> dict[str, Any]:
    root = _closed(value, _ROOT_FIELDS, "root spec")
    _require(root["schema_version"] == ROOT_SCHEMA, "root schema differs")
    _require(root["parent_stage_id"] == PARENT_STAGE_ID, "parent stage differs")
    _require(root["sampling_contract"] == pair_contract.SAMPLING_CONTRACT, "sampling contract differs")
    _require(root["semantic_input_closure"] == pair_contract.SEMANTIC_INPUT_CLOSURE, "semantic input closure differs")
    _require(root["artifact_use_contract"] == pair_contract.ARTIFACT_USE_CONTRACT, "artifact-use contract differs")
    _require(root["geometry_contract"] == pair_contract.GEOMETRY_CONTRACT, "geometry contract differs")
    _require(root["analysis_split"] == "fit", "root is not fit-only")
    _require(root["branch_order"] == list(REQUIRED_BRANCH_ORDER), "root branch order differs")
    _require(root["seed_cell_count"] == 4 and root["candidate_count"] == 8, "root count differs")
    groups = root["groups"]
    _require(type(groups) is list and len(groups) == 2, "root groups differ")
    seen_ids: set[str] = set()
    seen_cells: set[str] = set()
    normalized_groups = []
    for raw_group, (expected_id, expected_gpus) in zip(groups, GROUP_LAYOUT):
        group = _closed(raw_group, _ROOT_GROUP_FIELDS, "root group")
        _require(group["group_id"] == expected_id and group["visible_gpus"] == expected_gpus, "root group mapping differs")
        _require(type(group["cells"]) is list and len(group["cells"]) == 2, "root group cell count differs")
        candidates = []
        for raw_candidate in group["candidates"]:
            try:
                candidate = pair_contract.validate_candidate(raw_candidate)
            except pair_contract.PairT2VCalibrationSpecError as error:
                raise MinimalCrossAnchorPlanError(str(error)) from error
            _require(candidate["candidate_id"] not in seen_ids, "candidate ID is duplicated")
            _require(candidate["analysis_split"] == "fit", "candidate split differs")
            seen_ids.add(candidate["candidate_id"])
            candidates.append(candidate)
        _require(len(candidates) == 4, "root group candidate count differs")
        cells = []
        for raw_cell in group["cells"]:
            cell = _closed(raw_cell, _ROOT_CELL_FIELDS, "root cell")
            cell_id = _safe_id(cell["cell_id"], "cell_id")
            _require(cell_id not in seen_cells, "root cell is duplicated")
            seen_cells.add(cell_id)
            _require(cell["analysis_split"] == "fit", "root cell split differs")
            rows = [row for row in candidates if row["calibration_group_id"] == cell_id]
            _require([row["semantic_branch"] for row in rows] == list(REQUIRED_BRANCH_ORDER), "root cell branch closure differs")
            _require(cell["candidate_ids"] == [row["candidate_id"] for row in rows], "root cell candidate IDs differ")
            cells.append(cell)
        normalized_groups.append({**group, "cells": cells, "candidates": candidates})
    _require(len(seen_ids) == 8 and len(seen_cells) == 4, "root global count differs")
    audit_requests = root["audit_requests"]
    _require(type(audit_requests) is list and len(audit_requests) == 8, "audit request count differs")
    request_ids = []
    for request in audit_requests:
        _require(type(request) is dict, "audit request differs")
        unsigned = dict(request)
        digest = unsigned.pop("request_digest", None)
        _require(_sha256(digest, "audit request digest") == object_sha256(unsigned), "audit request digest differs")
        request_ids.append(request.get("candidate_id"))
    expected_request_ids = [
        candidate_id.replace("topup4-", "mosaic-full_topup20-v1-", 1)
        for group in normalized_groups
        for candidate_id in [row["candidate_id"] for row in group["candidates"]]
    ]
    _require(request_ids == expected_request_ids, "audit request/candidate order differs")
    execution = root["execution_contract"]
    _require(
        type(execution) is dict
        and execution.get("generated_media_is_editor_input_or_target") is False
        and execution.get("optimizer_authorized") is False
        and execution.get("independent_full81_review_required") is True
        and execution.get("same_state_six_sigma_materialization_required_after_review") is True,
        "root execution authority differs",
    )
    return {**root, "groups": normalized_groups}


def materialize_plan(
    *, selection_path: str | Path, expected_selection_sha256: str,
    registry_path: str | Path, expected_registry_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    selection, selection_resolved, selection_sha = load_selection(selection_path, expected_selection_sha256)
    _require(
        expected_registry_sha256 == PARENT_REGISTRY_FILE_SHA256,
        "parent registry file authority differs",
    )
    registry_raw, registry_resolved, registry_sha = _load_json(registry_path, "parent registry", expected_registry_sha256)
    _require(selection["parent_registry_sha256"] == registry_sha, "selection/registry SHA binding differs")
    try:
        registry = population.validate_registry(registry_raw)
    except population.MosaicEventPopulationError as error:
        raise MinimalCrossAnchorPlanError(str(error)) from error
    output = Path(output_dir)
    _require(
        output.is_absolute() and output != Path("/") and not output.exists() and not output.is_symlink()
        and output.parent.is_dir() and not output.parent.is_symlink(),
        "plan output must be a fresh absolute directory",
    )
    output.mkdir(mode=0o700)
    root_value = build_root_value(
        selection=selection,
        selection_path=selection_resolved,
        selection_sha256=selection_sha,
        registry=registry,
        registry_path=registry_resolved,
        registry_sha256=registry_sha,
    )
    validate_root_value(root_value)
    root_path = output / "full30-action-minimal-cross-anchor-topup4-root-v2.json"
    root_sha = _write_create_only(root_path, root_value)
    candidate_root = output / "candidate-plan"
    candidate_root.mkdir(mode=0o700)
    tasks = []
    cell_proofs = []
    shards = []
    for group in root_value["groups"]:
        group_dir = candidate_root / group["group_id"]
        group_dir.mkdir(mode=0o700)
        task_ids = []
        for ordinal, candidate in enumerate(group["candidates"]):
            envelope = {
                "schema_version": pair_contract.CANDIDATE_SCHEMA_VERSION,
                "root_spec_raw_sha256": root_sha,
                "group_id": group["group_id"],
                "visible_gpus": group["visible_gpus"],
                "ordinal": ordinal,
                "sampling_contract": pair_contract.SAMPLING_CONTRACT,
                "semantic_input_closure": pair_contract.SEMANTIC_INPUT_CLOSURE,
                "artifact_use_contract": pair_contract.ARTIFACT_USE_CONTRACT,
                "split_contract": pair_contract.SPLIT_CONTRACT,
                "candidate": candidate,
            }
            envelope_path = group_dir / f"{ordinal:04d}-{candidate['candidate_id']}.json"
            envelope_sha = _write_create_only(envelope_path, envelope)
            task = {
                "root_spec_path": str(root_path),
                "root_spec_sha256": root_sha,
                "candidate_spec_path": str(envelope_path),
                "candidate_spec_sha256": envelope_sha,
                "group_id": group["group_id"],
                "visible_gpus": group["visible_gpus"],
                "ordinal": ordinal,
                "candidate_id": candidate["candidate_id"],
                "analysis_split": "fit",
                "calibration_group_id": candidate["calibration_group_id"],
                "semantic_branch": candidate["semantic_branch"],
                "seed": candidate["seed"],
            }
            tasks.append(task)
            task_ids.append(candidate["candidate_id"])
        for cell in group["cells"]:
            cell_proofs.append(
                {
                    "group_id": group["group_id"],
                    "calibration_group_id": cell["cell_id"],
                    "analysis_split": "fit",
                    "seed": cell["seed"],
                    "candidate_ids": cell["candidate_ids"],
                    "branch_order": list(REQUIRED_BRANCH_ORDER),
                    "complete_ten_branch_cell": True,
                }
            )
        shards.append(
            {
                "shard_id": f"topup4-{group['group_id']}-fit",
                "group_id": group["group_id"],
                "visible_gpus": group["visible_gpus"],
                "candidate_ids": task_ids,
                "candidate_count": len(task_ids),
            }
        )
    execution_contract = dict(root_value["execution_contract"])
    plan_unsigned = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": PLAN_ID,
        "selection": root_value["selection"],
        "parent_registry": root_value["parent_registry"],
        "root_spec": {"path": str(root_path), "file_sha256": root_sha},
        "analysis_split": "fit",
        "generation_invocation_count": 8,
        "seed_cell_count": 4,
        "branch_order": list(REQUIRED_BRANCH_ORDER),
        "tasks": tasks,
        "cell_proofs": cell_proofs,
        "shards": shards,
        "execution_contract": execution_contract,
    }
    plan = {**plan_unsigned, "plan_digest": object_sha256(plan_unsigned)}
    plan_path = output / "full30-action-minimal-cross-anchor-topup4-plan-v2.json"
    plan_sha = _write_create_only(plan_path, plan)
    return {**plan, "_path": str(plan_path), "_file_sha256": plan_sha}


def load_plan(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], Path, str]:
    plan, resolved, observed = _load_json(path, "topup4 plan", expected_sha256)
    plan = _closed(plan, _PLAN_FIELDS, "topup4 plan")
    _require(plan["schema_version"] == PLAN_SCHEMA and plan["plan_id"] == PLAN_ID, "topup4 plan identity differs")
    unsigned = dict(plan)
    digest = unsigned.pop("plan_digest")
    _require(_sha256(digest, "plan digest") == object_sha256(unsigned), "plan digest differs")
    selection_ref = plan["selection"]
    registry_ref = plan["parent_registry"]
    selection, selection_path, selection_sha = load_selection(selection_ref["path"], selection_ref["file_sha256"])
    _require(
        registry_ref["file_sha256"] == PARENT_REGISTRY_FILE_SHA256,
        "parent registry file authority differs",
    )
    registry_raw, registry_path, registry_sha = _load_json(registry_ref["path"], "parent registry", registry_ref["file_sha256"])
    try:
        registry = population.validate_registry(registry_raw)
    except population.MosaicEventPopulationError as error:
        raise MinimalCrossAnchorPlanError(str(error)) from error
    root_ref = plan["root_spec"]
    root_raw, root_path, root_sha = _load_json(root_ref["path"], "topup4 root spec", root_ref["file_sha256"])
    root = validate_root_value(root_raw)
    expected_root = build_root_value(
        selection=selection, selection_path=selection_path, selection_sha256=selection_sha,
        registry=registry, registry_path=registry_path, registry_sha256=registry_sha,
    )
    _require(root == expected_root, "root spec is not the mechanical selection projection")
    tasks = plan["tasks"]
    _require(type(tasks) is list and len(tasks) == 8, "plan task count differs")
    expected_candidates = [row for group in root["groups"] for row in group["candidates"]]
    for task, candidate in zip(tasks, expected_candidates):
        _require(type(task) is dict, "plan task differs")
        envelope_path = _plain_file(task["candidate_spec_path"], "candidate envelope")
        _require(file_sha256(envelope_path) == task["candidate_spec_sha256"], "candidate envelope SHA differs")
        try:
            envelope = pair_contract.load_candidate_envelope(envelope_path, root_sha)
        except pair_contract.PairT2VCalibrationSpecError as error:
            raise MinimalCrossAnchorPlanError(str(error)) from error
        _require(envelope["candidate"] == candidate, "candidate envelope/root projection differs")
        _require(
            task["root_spec_path"] == str(root_path)
            and task["root_spec_sha256"] == root_sha
            and task["candidate_id"] == candidate["candidate_id"]
            and task["group_id"] == envelope["group_id"]
            and task["visible_gpus"] == envelope["visible_gpus"]
            and task["ordinal"] == envelope["ordinal"]
            and task["analysis_split"] == "fit"
            and task["calibration_group_id"] == candidate["calibration_group_id"]
            and task["semantic_branch"] == candidate["semantic_branch"]
            and task["seed"] == candidate["seed"],
            "plan task binding differs",
        )
    _require(plan["analysis_split"] == "fit", "plan split differs")
    _require(plan["generation_invocation_count"] == 8 and plan["seed_cell_count"] == 4, "plan count differs")
    _require(plan["branch_order"] == list(REQUIRED_BRANCH_ORDER), "plan branch order differs")
    _require(type(plan["cell_proofs"]) is list and len(plan["cell_proofs"]) == 4, "cell proofs differ")
    _require(type(plan["shards"]) is list and [row.get("candidate_count") for row in plan["shards"]] == [4, 4], "shard closure differs")
    _require(plan["execution_contract"] == root["execution_contract"], "execution contract differs")
    return plan, resolved, observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-plan")
    build.add_argument("--selection", required=True)
    build.add_argument("--expected-selection-sha256", required=True)
    build.add_argument("--parent-registry", required=True)
    build.add_argument("--expected-parent-registry-sha256", required=True)
    build.add_argument("--output-dir", required=True)
    validate = sub.add_parser("validate-plan")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-plan":
        result = materialize_plan(
            selection_path=args.selection,
            expected_selection_sha256=args.expected_selection_sha256,
            registry_path=args.parent_registry,
            expected_registry_sha256=args.expected_parent_registry_sha256,
            output_dir=args.output_dir,
        )
    else:
        result, _, _ = load_plan(args.plan, args.expected_plan_sha256)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MinimalCrossAnchorPlanError",
    "build_root_value",
    "canonical_json_bytes",
    "load_plan",
    "load_selection",
    "main",
    "materialize_plan",
    "object_sha256",
    "validate_root_value",
    "validate_selection",
]
