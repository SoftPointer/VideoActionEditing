#!/usr/bin/env python3
"""Fail-closed full-first8 block-22 Phi_v1 authority controller.

The controller has no unreviewed mode.  Before any model import it requires a
160/160 packet-bound external-review authority and the complete ten-branch
generation closure for the requested split.  It delegates frozen block-22,
exact40-index29 extraction to the existing SP4 materializer while observing
the already-computed camera/appearance Phi coordinates.  For the fit split it
then persists those exact [21,32] FP32 nuisance tensors and emits the 16-row
q0 operator-coordinate manifest required by O-stage projection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import os
from pathlib import Path
import shutil
import struct
import sys
from types import SimpleNamespace
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for root in (METHOD_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import generic_action_blind_review_authority_v1 as review_authority  # noqa: E402
import build_generic_action_phi_v1_authority_release_v1 as authority_release  # noqa: E402


AUTHORIZED_PLAN_SCHEMA = "bernini-phi-v1-external-authority-sp4-plan-v1"
PLAN_GAP_SCHEMA = "bernini-phi-v1-external-authority-plan-gap-v1"
OPERATOR_COORDINATE_SCHEMA = "bernini-phi-v1-operator-coordinate-manifest-v1"
CONTROLLER_RECEIPT_SCHEMA = "bernini-phi-v1-external-authority-controller-receipt-v1"
Q0_SHA256 = "71123fbde9571cfb5e2745eb8bee68c584e357e7b82dd12b48bab21047a99bbe"
FIT_OPERATOR_BRANCHES = ("action", "incomplete")
EXPECTED_SPLIT_CANDIDATES = 80
EXPECTED_SPLIT_CELLS = 8
EXPECTED_FIT_OPERATOR_ROWS = 16
RAW_COORDINATE_BYTES = 21 * 32 * 4
ALL_BRANCHES = review_authority.BRANCHES
SELECTED_BRANCHES = ("action", "noop", "reverse", "incomplete")

# Loaded only after validate_installed_closure succeeds.
manifests: Any = None
legacy_phi: Any = None


class PhiAuthorityControllerError(RuntimeError):
    """The review, generation, Phi, q0, or coordinate closure failed."""


def fail(message: str) -> NoReturn:
    raise PhiAuthorityControllerError(message)


def _activate_installed_closure(
    release_manifest: str | Path,
    expected_release_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Validate all overlay/base bytes before importing either base module."""

    try:
        installed = authority_release.validate_installed_closure(
            METHOD_ROOT, Path(release_manifest), expected_release_manifest_sha256,
        )
        review_authority._activate_installed_closure(
            release_manifest, expected_release_manifest_sha256,
        )
    except (
        authority_release.PhiAuthorityReleaseError,
        review_authority.BlindReviewAuthorityError,
    ) as error:
        raise PhiAuthorityControllerError(str(error)) from error
    global manifests, legacy_phi
    manifests = importlib.import_module("generic_action_manifest_v1")
    legacy_phi = importlib.import_module("materialize_phi_v1_sidecars_sp4")
    _require(tuple(legacy_phi.ALL_BRANCHES) == ALL_BRANCHES, "installed branch order differs")
    _require(tuple(legacy_phi.SELECTED_BRANCHES) == SELECTED_BRANCHES, "installed selected branch order differs")
    return installed


def _require_base_modules() -> tuple[Any, Any]:
    _require(manifests is not None and legacy_phi is not None, "installed release closure was not activated")
    return manifests, legacy_phi


def _require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    return review_authority.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return review_authority.object_sha256(value)


def file_sha256(path: str | Path) -> str:
    return review_authority.file_sha256(path)


def _write_json(path: Path, value: Mapping[str, Any], mode: int = 0o400) -> str:
    try:
        return review_authority._write_json(path, value, mode)
    except review_authority.BlindReviewAuthorityError as error:
        raise PhiAuthorityControllerError(str(error)) from error


def _load_json(path: str | Path, label: str, expected_sha256: Optional[str] = None, *, require_canonical: bool = True) -> tuple[dict[str, Any], Path, str]:
    try:
        return review_authority._load_json(path, label, expected_sha256, require_canonical=require_canonical)
    except review_authority.BlindReviewAuthorityError as error:
        raise PhiAuthorityControllerError(str(error)) from error


def _expected_cells(authoring: Mapping[str, Any], population: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    _, base_legacy_phi = _require_base_modules()
    cells = base_legacy_phi._expected_cells(authoring, population, split)
    _require(len(cells) == EXPECTED_SPLIT_CELLS, f"{split} must contain exactly eight seed cells")
    return cells


def _plan_gap(*, split: str, expected_ids: set[str], generation_ids: set[str], review_ids: set[str], unexpected_generation_ids: set[str]) -> dict[str, Any]:
    unsigned = {
        "schema_version": PLAN_GAP_SCHEMA, "analysis_split": split,
        "existing_core4_full_population_count": 80,
        "missing_reserve4_full_population_count_before_completion": 80,
        "full_first8_review_authority_required": 160,
        "expected_split_candidate_count": EXPECTED_SPLIT_CANDIDATES,
        "observed_generation_candidate_count": len(expected_ids & generation_ids),
        "observed_external_review_candidate_count": len(expected_ids & review_ids),
        "missing_generation_candidate_ids": sorted(expected_ids - generation_ids),
        "missing_external_review_candidate_ids": sorted(expected_ids - review_ids),
        "unexpected_generation_candidate_ids": sorted(unexpected_generation_ids),
        "phi_v1_materialization_authorized": False,
        "fit_operator_coordinate_manifest_authorized": False,
        "generated_rgb_or_latent_is_editor_input_or_target": False,
        "optimizer_authorized": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def build_authorized_plan(
    *, authoring_path: str | Path, population_path: str | Path,
    analysis_split: str, generation_roots: Sequence[str | Path],
    external_review_authority: str | Path,
    expected_external_review_authority_sha256: str,
    authority_release_manifest: str | Path,
    expected_authority_release_manifest_sha256: str,
    output: str | Path, gap_output: str | Path,
) -> Mapping[str, Any]:
    _activate_installed_closure(
        authority_release_manifest, expected_authority_release_manifest_sha256,
    )
    base_manifests, base_legacy_phi = _require_base_modules()
    _require(analysis_split in {"fit", "confirmation"}, "analysis split differs")
    authoring, _, _ = _load_json(authoring_path, "authoring registry", base_manifests.AUTHORING_SHA256, require_canonical=False)
    population, _, _ = _load_json(population_path, "population registry", base_manifests.POPULATION_SHA256, require_canonical=False)
    try:
        authority = review_authority.load_authority(
            external_review_authority, expected_external_review_authority_sha256,
            authority_release_manifest=authority_release_manifest,
            expected_authority_release_manifest_sha256=expected_authority_release_manifest_sha256,
            replay_packet=True,
        )
        generation = review_authority._scan_generation(generation_roots)
    except review_authority.BlindReviewAuthorityError as error:
        raise PhiAuthorityControllerError(str(error)) from error
    cells = _expected_cells(authoring, population, analysis_split)
    full_expected, _ = review_authority._population_context(authoring, population)
    full_expected_by_id = {row["candidate_id"]: row for row in full_expected}
    full_expected_ids = {row["candidate_id"] for row in full_expected}
    expected_ids = {candidate_id for cell in cells for candidate_id in cell["candidate_ids"].values()}
    authority_by_id = {row["candidate_id"]: row for row in authority["rows"] if row["analysis_split"] == analysis_split}
    generation_ids = set(generation)
    review_ids = set(authority_by_id)
    unexpected_generation_ids = generation_ids - full_expected_ids
    _require(len(expected_ids) == EXPECTED_SPLIT_CANDIDATES, "split candidate closure differs")
    if (expected_ids - generation_ids) or (expected_ids - review_ids) or unexpected_generation_ids:
        _write_json(Path(gap_output), _plan_gap(split=analysis_split, expected_ids=expected_ids, generation_ids=generation_ids, review_ids=review_ids, unexpected_generation_ids=unexpected_generation_ids))
    _require(not (expected_ids - generation_ids) and not (expected_ids - review_ids) and not unexpected_generation_ids, "generation/review closure is incomplete or contaminated; gap receipt written")
    plan_rows: list[dict[str, Any]] = []
    for cell in cells:
        generation_refs: dict[str, dict[str, str]] = {}
        review_refs: dict[str, dict[str, str]] = {}
        for branch in ALL_BRANCHES:
            candidate_id = cell["candidate_ids"][branch]
            registered = full_expected_by_id[candidate_id]
            receipt_path, generation_receipt = generation[candidate_id]
            candidate = generation_receipt["candidate"]
            _require(
                registered["source_iid"] == cell["source_iid"]
                and registered["analysis_split"] == analysis_split
                and registered["seed"] == cell["seed"]
                and registered["branch"] == branch
                and candidate["candidate_id"] == candidate_id
                and candidate["analysis_split"] == analysis_split
                and candidate["seed"] == cell["seed"]
                and candidate["semantic_branch"] == branch,
                "generation pinned cell binding differs",
            )
            review_row = authority_by_id[candidate_id]
            _require(review_row["branch"] == branch and review_row["source_iid"] == cell["source_iid"] and review_row["analysis_split"] == analysis_split and review_row["seed"] == cell["seed"], "external review pinned cell binding differs")
            _require(review_row["generation_receipt_file_sha256"] == generation_receipt["_file_sha256"] and review_row["media_sha256"] == generation_receipt["artifacts"]["mp4"]["sha256"], "external review/generation receipt file SHA differs")
            generation_refs[branch] = {"path": str(receipt_path), "file_sha256": generation_receipt["_file_sha256"]}
            review_refs[branch] = {"path": review_row["review_receipt_path"], "file_sha256": review_row["review_receipt_file_sha256"]}
        plan_rows.append({**cell, "generation_receipts": generation_refs, "review_receipts": review_refs})
    authority_path = review_authority._plain_file(external_review_authority, "external review authority")
    legacy_unsigned = {
        "schema_version": base_legacy_phi.PLAN_SCHEMA,
        "plan_id": f"generic-action-phi-v1-{analysis_split}-external-full160-r1-legacy-adapter",
        "analysis_split": analysis_split, "mode": "OFFICIAL_REVIEWED",
        "expected_seed_cells": EXPECTED_SPLIT_CELLS, "model_forwards": 82,
        "forward_accounting": "8_cells_x_5_prompt_pairs_x_2_plus_2_hook_parity",
        "phi_v1": {"block_index": 22, "teacher_exact40_index": 29, "p32_seed": base_manifests.P32_SEED, "nuisance_order": ["camera_only", "appearance_only_gram_schmidt_off_camera"]},
        "generated_media_is_optimizer_input_or_target": False, "rows": plan_rows,
    }
    legacy_plan = {**legacy_unsigned, "plan_digest": object_sha256(legacy_unsigned)}
    output_path = Path(output)
    legacy_path = output_path.with_name(f"{output_path.stem}.legacy-materializer.json")
    legacy_sha = _write_json(legacy_path, legacy_plan)
    unsigned = {
        "schema_version": AUTHORIZED_PLAN_SCHEMA,
        "legacy_materializer_schema": base_legacy_phi.PLAN_SCHEMA,
        "plan_id": f"generic-action-phi-v1-{analysis_split}-external-full160-r1",
        "analysis_split": analysis_split, "mode": "OFFICIAL_EXTERNAL_AUTHORITY_REVIEWED",
        "expected_seed_cells": EXPECTED_SPLIT_CELLS, "expected_candidate_reviews": EXPECTED_SPLIT_CANDIDATES,
        "model_forwards": 82,
        "forward_accounting": "8_cells_x_5_prompt_pairs_x_2_plus_2_hook_parity",
        "phi_v1": {"block_index": 22, "teacher_exact40_index": 29, "p32_seed": base_manifests.P32_SEED, "nuisance_order": ["camera_only", "appearance_only_gram_schmidt_off_camera"]},
        "legacy_materializer_plan": {"path": str(legacy_path), "file_sha256": legacy_sha, "plan_digest": legacy_plan["plan_digest"]},
        "external_review_authority": {"path": str(authority_path), "file_sha256": expected_external_review_authority_sha256, "authority_digest": authority["authority_digest"], "row_count": 160},
        "generated_media_is_optimizer_input_or_target": False,
        "rows": plan_rows,
    }
    plan = {**unsigned, "plan_digest": object_sha256(unsigned)}
    _write_json(output_path, plan)
    return plan


def validate_authorized_plan(
    path: str | Path, expected_sha256: str,
    *, authority_release_manifest: str | Path,
    expected_authority_release_manifest_sha256: str,
) -> tuple[Mapping[str, Any], Path, str, Mapping[str, Any]]:
    _activate_installed_closure(
        authority_release_manifest, expected_authority_release_manifest_sha256,
    )
    base_manifests, base_legacy_phi = _require_base_modules()
    plan, plan_path, observed_sha = _load_json(path, "authorized Phi plan", expected_sha256)
    required = {"schema_version", "legacy_materializer_schema", "plan_id", "analysis_split", "mode", "expected_seed_cells", "expected_candidate_reviews", "model_forwards", "forward_accounting", "phi_v1", "legacy_materializer_plan", "external_review_authority", "generated_media_is_optimizer_input_or_target", "rows", "plan_digest"}
    _require(set(plan) == required, "authorized Phi plan field closure differs")
    unsigned = dict(plan); declared = unsigned.pop("plan_digest")
    _require(declared == object_sha256(unsigned), "authorized Phi plan digest differs")
    split = plan["analysis_split"]
    exact_forward_accounting = "8_cells_x_5_prompt_pairs_x_2_plus_2_hook_parity"
    exact_phi = {"block_index": 22, "teacher_exact40_index": 29, "p32_seed": base_manifests.P32_SEED, "nuisance_order": ["camera_only", "appearance_only_gram_schmidt_off_camera"]}
    _require(
        split in {"fit", "confirmation"}
        and plan["schema_version"] == AUTHORIZED_PLAN_SCHEMA
        and plan["legacy_materializer_schema"] == base_legacy_phi.PLAN_SCHEMA
        and plan["plan_id"] == f"generic-action-phi-v1-{split}-external-full160-r1"
        and plan["mode"] == "OFFICIAL_EXTERNAL_AUTHORITY_REVIEWED"
        and plan["expected_seed_cells"] == 8
        and plan["expected_candidate_reviews"] == 80
        and plan["model_forwards"] == 82
        and plan["forward_accounting"] == exact_forward_accounting
        and plan["phi_v1"] == exact_phi
        and plan["generated_media_is_optimizer_input_or_target"] is False,
        "authorized Phi plan contract differs",
    )
    legacy_ref = plan["legacy_materializer_plan"]
    _require(set(legacy_ref) == {"path", "file_sha256", "plan_digest"}, "legacy materializer plan reference differs")
    legacy_plan, _, _ = _load_json(legacy_ref["path"], "legacy materializer plan", legacy_ref["file_sha256"])
    _require(set(legacy_plan) == {"schema_version", "plan_id", "analysis_split", "mode", "expected_seed_cells", "model_forwards", "forward_accounting", "phi_v1", "generated_media_is_optimizer_input_or_target", "rows", "plan_digest"}, "legacy materializer plan field closure differs")
    legacy_unsigned = dict(legacy_plan); legacy_digest = legacy_unsigned.pop("plan_digest", None)
    _require(
        legacy_digest == object_sha256(legacy_unsigned) == legacy_ref["plan_digest"]
        and legacy_plan["schema_version"] == base_legacy_phi.PLAN_SCHEMA
        and legacy_plan["plan_id"] == f"generic-action-phi-v1-{split}-external-full160-r1-legacy-adapter"
        and legacy_plan["analysis_split"] == split
        and legacy_plan["mode"] == "OFFICIAL_REVIEWED"
        and legacy_plan["expected_seed_cells"] == 8
        and legacy_plan["model_forwards"] == 82
        and legacy_plan["forward_accounting"] == exact_forward_accounting
        and legacy_plan["phi_v1"] == exact_phi
        and legacy_plan["generated_media_is_optimizer_input_or_target"] is False
        and legacy_plan["rows"] == plan["rows"],
        "legacy materializer plan binding differs",
    )
    authority_ref = plan["external_review_authority"]
    _require(set(authority_ref) == {"path", "file_sha256", "authority_digest", "row_count"} and authority_ref["row_count"] == 160, "plan review authority reference differs")
    try:
        authority = review_authority.load_authority(
            authority_ref["path"], authority_ref["file_sha256"],
            authority_release_manifest=authority_release_manifest,
            expected_authority_release_manifest_sha256=expected_authority_release_manifest_sha256,
            replay_packet=True,
        )
    except review_authority.BlindReviewAuthorityError as error:
        raise PhiAuthorityControllerError(str(error)) from error
    _require(authority["authority_digest"] == authority_ref["authority_digest"], "plan external authority digest differs")

    authoring, _, _ = _load_json(METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json", "installed authoring", base_manifests.AUTHORING_SHA256, require_canonical=False)
    population, _, _ = _load_json(METHOD_ROOT / "assets/mosaic_event_population_compact6_topup20_v1.json", "installed population", base_manifests.POPULATION_SHA256, require_canonical=False)
    expected_cells = _expected_cells(authoring, population, split)
    full_expected, _ = review_authority._population_context(authoring, population)
    registered_by_id = {row["candidate_id"]: row for row in full_expected}
    expected_ids = {candidate_id for cell in expected_cells for candidate_id in cell["candidate_ids"].values()}
    rows = plan["rows"]
    _require(type(rows) is list and len(rows) == len(expected_cells) == 8, "authorized Phi seed-cell closure differs")
    authority_by_id = {row["candidate_id"]: row for row in authority["rows"]}
    observed_ids: set[str] = set()
    for cell, expected_cell in zip(rows, expected_cells):
        _require(set(cell) == {"source_iid", "analysis_split", "seed", "candidate_ids", "generation_receipts", "review_receipts"}, "plan cell field closure differs")
        for field in ("source_iid", "analysis_split", "seed", "candidate_ids"):
            _require(cell[field] == expected_cell[field], f"plan pinned cell {field} differs")
        _require(set(cell["candidate_ids"]) == set(ALL_BRANCHES) and set(cell["generation_receipts"]) == set(ALL_BRANCHES) and set(cell["review_receipts"]) == set(ALL_BRANCHES), "plan branch closure differs")
        for branch in ALL_BRANCHES:
            candidate_id = cell["candidate_ids"][branch]
            registered = registered_by_id[candidate_id]
            _require(candidate_id not in observed_ids and registered["source_iid"] == cell["source_iid"] and registered["analysis_split"] == split and registered["seed"] == cell["seed"] and registered["branch"] == branch, "plan pinned candidate coordinate differs")
            observed_ids.add(candidate_id)
            generation_ref = cell["generation_receipts"][branch]
            _require(set(generation_ref) == {"path", "file_sha256"}, "plan generation reference closure differs")
            generation = base_legacy_phi._candidate_receipt(Path(generation_ref["path"]))
            candidate = generation["candidate"]
            authority_row = authority_by_id[candidate_id]
            _require(
                generation["_file_sha256"] == generation_ref["file_sha256"] == authority_row["generation_receipt_file_sha256"]
                and candidate["candidate_id"] == candidate_id
                and candidate["analysis_split"] == split
                and candidate["seed"] == cell["seed"]
                and candidate["semantic_branch"] == branch
                and candidate["calibration_group_id"] == f"cell-{cell['source_iid']}-s{cell['seed']}"
                and generation["root_spec_raw_sha256"] == registered["root_spec_raw_sha256"],
                "plan generation receipt/pinned authority differs",
            )
            review_ref = cell["review_receipts"][branch]
            _require(set(review_ref) == {"path", "file_sha256"}, "plan review reference closure differs")
            reviewed = base_manifests.validate_review_receipt(review_ref["path"], review_ref["file_sha256"])
            _require(
                reviewed["candidate_id"] == candidate_id
                and reviewed["branch"] == branch
                and reviewed["media_sha256"] == generation["artifacts"]["mp4"]["sha256"] == authority_row["media_sha256"]
                and authority_row["source_iid"] == cell["source_iid"]
                and authority_row["analysis_split"] == split
                and authority_row["seed"] == cell["seed"]
                and authority_row["branch"] == branch
                and authority_row["review_receipt_path"] == review_ref["path"]
                and authority_row["review_receipt_file_sha256"] == review_ref["file_sha256"],
                "plan review/generation full160 authority differs",
            )
    _require(observed_ids == expected_ids and len(observed_ids) == 80, "plan exact split80 candidate closure differs")
    return plan, plan_path, observed_sha, authority


def _q0_rows(q0_authority_path: str | Path) -> Mapping[str, Mapping[str, Any]]:
    value, _, _ = _load_json(q0_authority_path, "q0 authority", Q0_SHA256, require_canonical=False)
    _require(value.get("schema_version") == manifests.Q0_SCHEMA and value.get("generated_media_is_editor_input_or_target") is False and value.get("optimizer_authorized") is False, "q0 authority boundary differs")
    rows = value.get("rows")
    _require(type(rows) is list and len(rows) == 8 and len({row["iid"] for row in rows}) == 8, "q0 authority row closure differs")
    return {row["iid"]: row for row in rows}


def preflight_q0_sources(q0_authority_path: str | Path) -> Mapping[str, Mapping[str, Any]]:
    rows = _q0_rows(q0_authority_path)
    validated: dict[str, Mapping[str, Any]] = {}
    for iid, row in rows.items():
        source = review_authority._plain_file(row["q0_source_video_path"], f"q0 source {iid}")
        _require(file_sha256(source) == row["q0_source_video_sha256"], f"q0 source SHA differs: {iid}")
        try:
            exact81 = review_authority._probe_full81(source)
        except review_authority.BlindReviewAuthorityError as error:
            raise PhiAuthorityControllerError(str(error)) from error
        validated[iid] = {**row, "_exact81": exact81}
    return validated


def _binding(path: Path, sha256: str, *, normalization: str) -> dict[str, Any]:
    _require(path.stat().st_size == RAW_COORDINATE_BYTES and file_sha256(path) == sha256, "coordinate tensor byte binding differs")
    return {"path": str(path), "raw_sha256": sha256, "dtype": "float32", "byte_order": "little", "shape": [21, 32], "normalization": normalization}


def _validate_raw_coordinate(binding: Mapping[str, Any], label: str) -> tuple[float, ...]:
    _require(set(binding) == {"path", "raw_sha256", "dtype", "byte_order", "shape", "normalization"}, f"{label} binding closure differs")
    _require(binding["dtype"] == "float32" and binding["byte_order"] == "little" and binding["shape"] == [21, 32] and binding["normalization"] == "raw_phi_v1_phase0_zero_temporal_dc_before_nuisance_projection", f"{label} tensor contract differs")
    path = review_authority._plain_file(binding["path"], label)
    raw = path.read_bytes()
    _require(len(raw) == RAW_COORDINATE_BYTES and hashlib.sha256(raw).hexdigest() == binding["raw_sha256"], f"{label} tensor bytes differ")
    values = struct.unpack("<672f", raw)
    _require(all(math.isfinite(value) for value in values), f"{label} tensor is non-finite")
    _require(raw[: 32 * 4] == b"\x00" * (32 * 4), f"{label} phase zero differs")
    for channel in range(32):
        mean = sum(values[phase * 32 + channel] for phase in range(1, 21)) / 20
        _require(abs(mean) <= 2.0e-5, f"{label} temporal DC differs")
    _require(math.sqrt(sum(value * value for value in values)) > 1.0e-8, f"{label} coordinate degenerates")
    return values


def materialize_operator_coordinate_manifest(*, plan: Mapping[str, Any], authority: Mapping[str, Any], output_dir: str | Path, captured_cells: Sequence[tuple[Any, Any]], q0_authority_path: str | Path) -> Optional[Mapping[str, Any]]:
    if plan["analysis_split"] != "fit":
        return None
    _require(len(plan["rows"]) == len(captured_cells) == EXPECTED_SPLIT_CELLS, "fit nuisance capture closure differs")
    output = review_authority._plain_dir(output_dir, "Phi output")
    run_path = output / "phi-v1-sidecar-run-receipt.json"
    run, _, run_sha = _load_json(run_path, "legacy Phi run receipt")
    unsigned_run = dict(run); declared = unsigned_run.pop("receipt_digest", None)
    _require(declared == object_sha256(unsigned_run) and run.get("schema_version") == legacy_phi.RUN_SCHEMA and run.get("mode") == "OFFICIAL_REVIEWED" and run.get("model_forwards") == 82 and run.get("sidecar_count") == 32 and run.get("generated_media_is_optimizer_input_or_target") is False, "legacy Phi run receipt differs")
    sidecars: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    validated_p32: set[tuple[str, str]] = set()
    for ref in run["sidecars"]:
        receipt = manifests.validate_sidecar_receipt(ref["path"], ref["file_sha256"], require_admissible=True, validated_p32=validated_p32)
        sidecars[receipt["row_id"]] = (receipt, ref)
    _require(len(sidecars) == 32, "legacy Phi sidecar closure differs")
    q0 = preflight_q0_sources(q0_authority_path)
    coordinate_root = output / "operator-coordinates"
    _require(not coordinate_root.exists() and not coordinate_root.is_symlink(), "operator coordinate output already exists")
    coordinate_root.mkdir()
    rows: list[dict[str, Any]] = []
    cell_bindings: list[dict[str, Any]] = []
    try:
        for cell, captured in zip(plan["rows"], captured_cells):
            camera, appearance = captured
            cell_dir = coordinate_root / f"{cell['analysis_split']}__{cell['source_iid']}__s{cell['seed']}"
            cell_dir.mkdir()
            camera_path = cell_dir / "camera_only.raw-phi-v1.f32le"
            appearance_path = cell_dir / "appearance_only.raw-phi-v1.f32le"
            camera_sha = legacy_phi._save_f32le(camera_path, camera)
            appearance_sha = legacy_phi._save_f32le(appearance_path, appearance)
            camera_binding = _binding(camera_path, camera_sha, normalization="raw_phi_v1_phase0_zero_temporal_dc_before_nuisance_projection")
            appearance_binding = _binding(appearance_path, appearance_sha, normalization="raw_phi_v1_phase0_zero_temporal_dc_before_nuisance_projection")
            q0_row = q0[cell["source_iid"]]
            cell_bindings.append({"source_iid": cell["source_iid"], "seed": cell["seed"], "camera_coordinate": camera_binding, "appearance_coordinate": appearance_binding})
            for branch in FIT_OPERATOR_BRANCHES:
                row_id = f"gaav1:fit:{cell['source_iid']}:s{cell['seed']}:{branch}"
                sidecar, sidecar_ref = sidecars[row_id]
                nuisance = sidecar["nuisance_projection"]
                _require(nuisance["camera_raw_sha256"] == camera_sha and nuisance["appearance_raw_sha256"] == appearance_sha, f"captured nuisance bytes differ from sidecar: {row_id}")
                rows.append({
                    "row_id": row_id, "candidate_id": cell["candidate_ids"][branch],
                    "source_iid": cell["source_iid"], "analysis_split": "fit", "seed": cell["seed"], "branch": branch,
                    "q0_source": {"path": q0_row["q0_source_video_path"], "sha256": q0_row["q0_source_video_sha256"], "real_source_anchor": True, "exact81": q0_row["_exact81"]},
                    "quotient_tensor": sidecar["tensor"], "camera_coordinate": camera_binding,
                    "appearance_coordinate": appearance_binding,
                    "projection_contract": {"order": ["camera_only", "appearance_only_gram_schmidt_off_camera"], "same_projection_required_for_predicted_hidden_delta": True, "weighted_metric_mixing": False},
                    "phi_v1": {"block_index": 22, "teacher_exact40_index": 29, "p32_raw_sha256": sidecar["phi_v1"]["p32_raw_sha256"], "shape": [21, 32]},
                    "sidecar_receipt": {"path": sidecar_ref["path"], "file_sha256": sidecar_ref["file_sha256"], "receipt_digest": sidecar["receipt_digest"]},
                    "external_review_authority_digest": authority["authority_digest"],
                    "generated_rgb_or_latent_is_editor_input_or_target": False,
                    "operator_coordinate_eligible": True,
                })
        rows.sort(key=lambda row: row["row_id"])
        _require(len(rows) == EXPECTED_FIT_OPERATOR_ROWS and len({row["row_id"] for row in rows}) == EXPECTED_FIT_OPERATOR_ROWS, "fit operator row closure differs")
        manifest_unsigned = {
            "schema_version": OPERATOR_COORDINATE_SCHEMA,
            "manifest_id": "generic-action-first8-fit16-phi-v1-operator-coordinates-r1",
            "analysis_split": "fit", "row_count": EXPECTED_FIT_OPERATOR_ROWS,
            "seed_cell_count": EXPECTED_SPLIT_CELLS,
            "required_full_first8_review_count": 160,
            "observed_full_first8_review_count": authority["row_count"],
            "core4_only_operator_rows": 8, "full_first8_operator_rows": 16,
            "external_review_authority": {"path": plan["external_review_authority"]["path"], "file_sha256": plan["external_review_authority"]["file_sha256"], "authority_digest": authority["authority_digest"]},
            "legacy_phi_run_receipt": {"path": str(run_path), "file_sha256": run_sha, "receipt_digest": run["receipt_digest"]},
            "phi_v1": {"block_index": 22, "teacher_exact40_index": 29, "shape": [21, 32], "dtype": "float32", "byte_order": "little"},
            "projection_contract": {"order": ["camera_only", "appearance_only_gram_schmidt_off_camera"], "predicted_hidden_delta_must_use_identical_projection": True, "camera_and_appearance_are_separate_coordinates": True, "weighted_metric_mixing": False},
            "cell_coordinates": cell_bindings, "rows": rows,
            "generated_rgb_or_latent_is_editor_input_or_target": False,
            "q0_real_source_is_only_editor_anchor": True,
            "operator_coordinate_authorized": True,
            "optimizer_authorized_by_this_manifest_alone": False,
        }
        manifest = {**manifest_unsigned, "manifest_digest": object_sha256(manifest_unsigned)}
        _write_json(output / "phi-v1-operator-coordinate-manifest.json", manifest)
    except Exception:
        shutil.rmtree(coordinate_root, ignore_errors=True)
        raise
    return manifest


def validate_operator_coordinate_manifest(
    path: str | Path, expected_sha256: str,
    *, authority_release_manifest: str | Path,
    expected_authority_release_manifest_sha256: str,
) -> Mapping[str, Any]:
    _activate_installed_closure(
        authority_release_manifest, expected_authority_release_manifest_sha256,
    )
    base_manifests, base_legacy_phi = _require_base_modules()
    value, _, _ = _load_json(path, "operator coordinate manifest", expected_sha256)
    required = {"schema_version", "manifest_id", "analysis_split", "row_count", "seed_cell_count", "required_full_first8_review_count", "observed_full_first8_review_count", "core4_only_operator_rows", "full_first8_operator_rows", "external_review_authority", "legacy_phi_run_receipt", "phi_v1", "projection_contract", "cell_coordinates", "rows", "generated_rgb_or_latent_is_editor_input_or_target", "q0_real_source_is_only_editor_anchor", "operator_coordinate_authorized", "optimizer_authorized_by_this_manifest_alone", "manifest_digest"}
    _require(set(value) == required, "operator coordinate manifest field closure differs")
    unsigned = dict(value); declared = unsigned.pop("manifest_digest")
    _require(declared == object_sha256(unsigned), "operator coordinate manifest digest differs")
    _require(value["schema_version"] == OPERATOR_COORDINATE_SCHEMA and value["manifest_id"] == "generic-action-first8-fit16-phi-v1-operator-coordinates-r1" and value["analysis_split"] == "fit" and value["row_count"] == 16 and value["seed_cell_count"] == 8 and value["required_full_first8_review_count"] == value["observed_full_first8_review_count"] == 160 and value["core4_only_operator_rows"] == 8 and value["full_first8_operator_rows"] == 16, "operator coordinate population differs")
    _require(value["phi_v1"] == {"block_index": 22, "teacher_exact40_index": 29, "shape": [21, 32], "dtype": "float32", "byte_order": "little"}, "operator Phi contract differs")
    expected_projection = {"order": ["camera_only", "appearance_only_gram_schmidt_off_camera"], "predicted_hidden_delta_must_use_identical_projection": True, "camera_and_appearance_are_separate_coordinates": True, "weighted_metric_mixing": False}
    _require(value["projection_contract"] == expected_projection and value["generated_rgb_or_latent_is_editor_input_or_target"] is False and value["q0_real_source_is_only_editor_anchor"] is True and value["operator_coordinate_authorized"] is True and value["optimizer_authorized_by_this_manifest_alone"] is False, "operator authority boundary differs")
    authority_ref = value["external_review_authority"]
    _require(set(authority_ref) == {"path", "file_sha256", "authority_digest"}, "operator review authority reference differs")
    try:
        authority = review_authority.load_authority(
            authority_ref["path"], authority_ref["file_sha256"],
            authority_release_manifest=authority_release_manifest,
            expected_authority_release_manifest_sha256=expected_authority_release_manifest_sha256,
            replay_packet=True,
        )
    except review_authority.BlindReviewAuthorityError as error:
        raise PhiAuthorityControllerError(str(error)) from error
    _require(authority["authority_digest"] == authority_ref["authority_digest"] and authority["row_count"] == 160, "operator external review authority differs")
    run_ref = value["legacy_phi_run_receipt"]
    _require(set(run_ref) == {"path", "file_sha256", "receipt_digest"}, "operator run reference differs")
    run, _, _ = _load_json(run_ref["path"], "legacy Phi run receipt", run_ref["file_sha256"])
    run_unsigned = dict(run); run_digest = run_unsigned.pop("receipt_digest", None)
    _require(run_digest == object_sha256(run_unsigned) == run_ref["receipt_digest"] and run.get("schema_version") == base_legacy_phi.RUN_SCHEMA and run.get("model_forwards") == 82 and run.get("sidecar_count") == 32, "operator legacy run differs")
    cell_rows = value["cell_coordinates"]
    rows = value["rows"]
    _require(type(cell_rows) is list and len(cell_rows) == 8 and type(rows) is list and len(rows) == 16, "operator row closure differs")
    cells: dict[tuple[str, int], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for cell in cell_rows:
        _require(set(cell) == {"source_iid", "seed", "camera_coordinate", "appearance_coordinate"}, "operator cell coordinate closure differs")
        key = (cell["source_iid"], cell["seed"])
        _require(key not in cells, "operator cell coordinate duplicated")
        camera_values = _validate_raw_coordinate(cell["camera_coordinate"], f"camera coordinate {key}")
        appearance_values = _validate_raw_coordinate(cell["appearance_coordinate"], f"appearance coordinate {key}")
        camera_norm = math.sqrt(sum(item * item for item in camera_values))
        dot = sum(left * right for left, right in zip(camera_values, appearance_values)) / camera_norm
        appearance_orth = tuple(right - dot * (left / camera_norm) for left, right in zip(camera_values, appearance_values))
        _require(math.sqrt(sum(item * item for item in appearance_orth)) > 1.0e-8, f"appearance coordinate degenerates after camera GS: {key}")
        cells[key] = (cell["camera_coordinate"], cell["appearance_coordinate"])
    expected_ids: set[str] = set()
    validated_p32: set[tuple[str, str]] = set()
    authority_by_candidate = {row["candidate_id"]: row for row in authority["rows"]}
    authoring, _, _ = _load_json(METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json", "installed authoring", base_manifests.AUTHORING_SHA256, require_canonical=False)
    population, _, _ = _load_json(METHOD_ROOT / "assets/mosaic_event_population_compact6_topup20_v1.json", "installed population", base_manifests.POPULATION_SHA256, require_canonical=False)
    expected_fit_cells = _expected_cells(authoring, population, "fit")
    expected_row_ids = {f"gaav1:fit:{cell['source_iid']}:s{cell['seed']}:{branch}" for cell in expected_fit_cells for branch in FIT_OPERATOR_BRANCHES}
    q0_registry = _q0_rows(METHOD_ROOT / "assets/action_source_q0_authority_first8_v1.json")
    q0_probe_cache: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_fields = {"row_id", "candidate_id", "source_iid", "analysis_split", "seed", "branch", "q0_source", "quotient_tensor", "camera_coordinate", "appearance_coordinate", "projection_contract", "phi_v1", "sidecar_receipt", "external_review_authority_digest", "generated_rgb_or_latent_is_editor_input_or_target", "operator_coordinate_eligible"}
        _require(set(row) == row_fields, "operator row field closure differs")
        key = (row["source_iid"], row["seed"])
        expected_id = f"gaav1:fit:{row['source_iid']}:s{row['seed']}:{row['branch']}"
        _require(row["row_id"] == expected_id and expected_id not in expected_ids and row["analysis_split"] == "fit" and row["branch"] in FIT_OPERATOR_BRANCHES, "operator row identity differs")
        expected_ids.add(expected_id)
        _require(key in cells and row["camera_coordinate"] == cells[key][0] and row["appearance_coordinate"] == cells[key][1], "operator row/cell nuisance binding differs")
        _require(row["projection_contract"] == {"order": ["camera_only", "appearance_only_gram_schmidt_off_camera"], "same_projection_required_for_predicted_hidden_delta": True, "weighted_metric_mixing": False}, "operator row projection differs")
        q0 = row["q0_source"]
        _require(set(q0) == {"path", "sha256", "real_source_anchor", "exact81"} and q0["real_source_anchor"] is True, "operator q0 source binding differs")
        registered_q0 = q0_registry[row["source_iid"]]
        _require(q0["path"] == registered_q0["q0_source_video_path"] and q0["sha256"] == registered_q0["q0_source_video_sha256"], "operator q0 registry binding differs")
        q0_path = review_authority._plain_file(q0["path"], "operator q0 source")
        _require(file_sha256(q0_path) == q0["sha256"], "operator q0 source SHA differs")
        try:
            if str(q0_path) not in q0_probe_cache:
                q0_probe_cache[str(q0_path)] = review_authority._probe_full81(q0_path)
            observed_q0 = q0_probe_cache[str(q0_path)]
            _require(observed_q0 == q0["exact81"], "operator q0 exact81 replay differs")
        except review_authority.BlindReviewAuthorityError as error:
            raise PhiAuthorityControllerError(str(error)) from error
        sidecar_ref = row["sidecar_receipt"]
        _require(set(sidecar_ref) == {"path", "file_sha256", "receipt_digest"}, "operator sidecar reference differs")
        sidecar = base_manifests.validate_sidecar_receipt(sidecar_ref["path"], sidecar_ref["file_sha256"], require_admissible=True, validated_p32=validated_p32)
        _require(sidecar["receipt_digest"] == sidecar_ref["receipt_digest"] and sidecar["row_id"] == row["row_id"] and sidecar["candidate_id"] == row["candidate_id"] and sidecar["tensor"] == row["quotient_tensor"], "operator sidecar binding differs")
        _require(sidecar["nuisance_projection"]["camera_raw_sha256"] == row["camera_coordinate"]["raw_sha256"] and sidecar["nuisance_projection"]["appearance_raw_sha256"] == row["appearance_coordinate"]["raw_sha256"], "operator sidecar nuisance hashes differ")
        authority_row = authority_by_candidate.get(row["candidate_id"])
        _require(authority_row is not None and authority_row["source_iid"] == row["source_iid"] and authority_row["analysis_split"] == "fit" and authority_row["seed"] == row["seed"] and authority_row["branch"] == row["branch"] and row["external_review_authority_digest"] == authority["authority_digest"], "operator row external review binding differs")
        _require(row["phi_v1"] == {"block_index": 22, "teacher_exact40_index": 29, "p32_raw_sha256": sidecar["phi_v1"]["p32_raw_sha256"], "shape": [21, 32]} and row["generated_rgb_or_latent_is_editor_input_or_target"] is False and row["operator_coordinate_eligible"] is True, "operator row authority differs")
    _require(expected_ids == expected_row_ids and set(FIT_OPERATOR_BRANCHES) == {row["branch"] for row in rows} and all(sum(row["source_iid"] == iid and row["seed"] == seed for row in rows) == 2 for iid, seed in cells), "operator exact16 pairing differs")
    return value


def run_sp4(args: argparse.Namespace) -> int:
    installed_release = _activate_installed_closure(
        args.authority_release_manifest,
        args.expected_authority_release_manifest_sha256,
    )
    _, base_legacy_phi = _require_base_modules()
    plan, plan_path, plan_sha, authority = validate_authorized_plan(
        args.plan, args.expected_plan_sha256,
        authority_release_manifest=args.authority_release_manifest,
        expected_authority_release_manifest_sha256=args.expected_authority_release_manifest_sha256,
    )
    _require(file_sha256(Path(__file__).resolve(strict=True)) == args.expected_controller_source_sha256, "controller source SHA-256 differs")
    controller_pin = next(row["sha256"] for row in installed_release["files"] if row["path"] == "tools/materialize_phi_v1_authority_controller_v1.py")
    _require(controller_pin == args.expected_controller_source_sha256, "release/controller source pin differs")
    preflight_q0_sources(args.q0_authority)
    captured: list[tuple[Any, Any]] = []
    call_count = 0
    original = base_legacy_phi._gram_schmidt_project

    def capture(raw: Any, camera: Any, appearance: Any) -> tuple[Any, dict[str, Any]]:
        nonlocal call_count
        cell_index, branch_index = divmod(call_count, len(base_legacy_phi.SELECTED_BRANCHES) - 1)
        _require(cell_index < EXPECTED_SPLIT_CELLS, "nuisance capture call count overflow")
        if branch_index == 0:
            captured.append((camera.detach().float().cpu().contiguous(), appearance.detach().float().cpu().contiguous()))
        else:
            _require(base_legacy_phi._raw_tensor_sha(captured[cell_index][0]) == base_legacy_phi._raw_tensor_sha(camera) and base_legacy_phi._raw_tensor_sha(captured[cell_index][1]) == base_legacy_phi._raw_tensor_sha(appearance), "nuisance coordinates changed within seed cell")
        call_count += 1
        return original(raw, camera, appearance)

    legacy_args = SimpleNamespace(**vars(args))
    legacy_args.allow_unreviewed_technical_only = False
    compatibility = plan["legacy_materializer_plan"]
    legacy_args.plan = compatibility["path"]
    legacy_args.expected_plan_sha256 = compatibility["file_sha256"]
    base_legacy_phi._gram_schmidt_project = capture
    try:
        result = base_legacy_phi.run_sp4(legacy_args)
    finally:
        base_legacy_phi._gram_schmidt_project = original
    _require(call_count == EXPECTED_SPLIT_CELLS * 3 and len(captured) == EXPECTED_SPLIT_CELLS, "nuisance capture closure differs")
    if int(os.environ.get("RANK", "0")) == 0:
        coordinate = materialize_operator_coordinate_manifest(plan=plan, authority=authority, output_dir=args.output_dir, captured_cells=captured, q0_authority_path=args.q0_authority)
        coordinate_path = Path(args.output_dir) / "phi-v1-operator-coordinate-manifest.json"
        receipt_unsigned = {
            "schema_version": CONTROLLER_RECEIPT_SCHEMA,
            "authorized_plan_path": str(plan_path), "authorized_plan_file_sha256": plan_sha,
            "authorized_plan_digest": plan["plan_digest"],
            "external_review_authority_digest": authority["authority_digest"],
            "analysis_split": plan["analysis_split"], "world_size": 4,
            "model_forwards": 82, "sidecar_count": 32,
            "operator_coordinate_manifest": None if coordinate is None else {"path": str(coordinate_path), "file_sha256": file_sha256(coordinate_path), "manifest_digest": coordinate["manifest_digest"], "row_count": 16},
            "generated_rgb_or_latent_is_editor_input_or_target": False,
            "training_performed": False, "optimizer_created": False,
        }
        receipt = {**receipt_unsigned, "receipt_digest": object_sha256(receipt_unsigned)}
        _write_json(Path(args.output_dir) / "phi-v1-authority-controller-receipt.json", receipt)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("build-plan")
    plan.add_argument("--authoring", required=True); plan.add_argument("--population", required=True)
    plan.add_argument("--analysis-split", choices=("fit", "confirmation"), required=True)
    plan.add_argument("--generation-root", action="append", required=True)
    plan.add_argument("--external-review-authority", required=True); plan.add_argument("--expected-external-review-authority-sha256", required=True)
    plan.add_argument("--authority-release-manifest", required=True); plan.add_argument("--expected-authority-release-manifest-sha256", required=True)
    plan.add_argument("--output", required=True); plan.add_argument("--gap-output", required=True)
    run = commands.add_parser("run-sp4")
    run.add_argument("--plan", required=True); run.add_argument("--expected-plan-sha256", required=True)
    run.add_argument("--expected-controller-source-sha256", required=True); run.add_argument("--q0-authority", required=True)
    run.add_argument("--authority-release-manifest", required=True); run.add_argument("--expected-authority-release-manifest-sha256", required=True)
    run.add_argument("--bernini-root", required=True); run.add_argument("--veomni-root", required=True)
    run.add_argument("--checkpoint", required=True); run.add_argument("--checkpoint-content-manifest", required=True)
    run.add_argument("--expected-bernini-commit", required=True); run.add_argument("--expected-veomni-commit", required=True)
    run.add_argument("--output-dir", required=True)
    audit = commands.add_parser("audit-plan")
    audit.add_argument("--plan", required=True); audit.add_argument("--expected-plan-sha256", required=True)
    audit.add_argument("--authority-release-manifest", required=True); audit.add_argument("--expected-authority-release-manifest-sha256", required=True)
    operator = commands.add_parser("audit-operator-manifest")
    operator.add_argument("--manifest", required=True); operator.add_argument("--expected-manifest-sha256", required=True)
    operator.add_argument("--authority-release-manifest", required=True); operator.add_argument("--expected-authority-release-manifest-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-plan":
        value = build_authorized_plan(authoring_path=args.authoring, population_path=args.population, analysis_split=args.analysis_split, generation_roots=args.generation_root, external_review_authority=args.external_review_authority, expected_external_review_authority_sha256=args.expected_external_review_authority_sha256, authority_release_manifest=args.authority_release_manifest, expected_authority_release_manifest_sha256=args.expected_authority_release_manifest_sha256, output=args.output, gap_output=args.gap_output)
        print(canonical_json_bytes(value).decode("ascii"), flush=True); return 0
    if args.command == "audit-plan":
        value, _, _, _ = validate_authorized_plan(args.plan, args.expected_plan_sha256, authority_release_manifest=args.authority_release_manifest, expected_authority_release_manifest_sha256=args.expected_authority_release_manifest_sha256)
        print(canonical_json_bytes(value).decode("ascii"), flush=True); return 0
    if args.command == "audit-operator-manifest":
        value = validate_operator_coordinate_manifest(args.manifest, args.expected_manifest_sha256, authority_release_manifest=args.authority_release_manifest, expected_authority_release_manifest_sha256=args.expected_authority_release_manifest_sha256)
        print(canonical_json_bytes(value).decode("ascii"), flush=True); return 0
    return run_sp4(args)


if __name__ == "__main__":
    raise SystemExit(main())
