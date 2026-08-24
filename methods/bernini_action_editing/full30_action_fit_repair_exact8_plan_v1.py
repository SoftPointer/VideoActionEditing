#!/usr/bin/env python3
"""Closed BOX-EXP-011 fit action/incomplete repair exact8 plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence

import pair_v5_t2v_calibration_bank_spec as bank_contract

PLAN_SCHEMA = "bernini-full30-action-fit-repair-exact8-plan-v1"
PLAN_ID = "BOX-EXP-011-fit-action-incomplete-repair-exact8-v1"
PLAN_FILENAME = "full30-action-fit-repair-exact8-plan-v1.json"
SEED1_SPEC_SHA256 = "2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab"
SEED2_SPEC_SHA256 = "0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e"
ADMISSION_BRANCH_ORDER = ("action", "incomplete")
MATERIALIZER_CONTROL_ORDER = (
    "noop", "camera_only", "appearance_only", "wrong_actor", "wrong_object",
    "generic_wrong_motion",
)
GROUP_LAYOUT = (("sp4-a", [0, 1, 2, 3]), ("sp4-b", [4, 5, 6, 7]))
SHA256_RE = re.compile(r"[0-9a-f]{64}")

LOCKED_BLIND_AUTHORITY: Mapping[str, Any] = {
    "reviewer_receipt_sha256": "4b1c7186d2daee7699e2711bffbf849813f69cf86c11875542c5a8978242d7c2",
    "sealed_key_sha256": "e120ba7ac6b9b7c437d5515f1c2d3aba2a5990bdcf45de2ad08452f7e7d3ae01",
    "unblind_verdict_sha256": "dc97cc334f23a4c9f3187c6db606ee098beebe6a409e0f2408eea7cd77a261c9",
    "action_media_pass": [2, 4],
    "incomplete_media_pass": [0, 4],
    "complete_pair_pass": [0, 4],
    "verdict": "FAIL_CLOSED__NO_COMPLETE_FIT_SEED_CELL_ADMITTED",
    "repair_trigger": "locked_blind_failure_categories_only",
}

PROMPTS: Mapping[str, Mapping[str, str]] = {
    "arms_action": {
        "text": (
            "A continuous medium portrait shows a single adult woman with both arms clearly "
            "raised in the original simple one-subject scene. In one clear ordered motion, "
            "the main woman lowers both raised arms, places both hands firmly on her hips, "
            "and holds the completed hands-on-hips pose through the end while the camera "
            "stays locked. Both hands must visibly contact and remain on the hips. The shot "
            "stays continuous, the illumination remains stable, and the final frame is "
            "temporally coherent."
        ),
        "utf8_sha256": "1a82aecdf6386a4f71800d30e663ae1c1dfda97bcfa14e64fda0fa53e11c872f",
    },
    "arms_incomplete": {
        "text": (
            "A continuous medium portrait shows a single adult woman with both arms clearly "
            "raised in the original simple one-subject scene. The main woman performs only "
            "the correct lowering prefix: she lowers both raised arms a short distance, "
            "stops with both hands held at mid-torso level, and holds that before-terminal "
            "pose through the end while the camera stays locked. Neither hand ever touches, "
            "reaches, or passes toward either hip; the hands-on-hips terminal pose must never "
            "appear. The shot stays continuous, the illumination remains stable, and the "
            "final frame is temporally coherent."
        ),
        "utf8_sha256": "e76d1aabd6ae3aac0c9076ef71584a6ef8b246f7048b37c89de88886d97a0773",
    },
    "reach_action": {
        "text": (
            "A continuous medium shot shows a single adult performer holding the left hand "
            "as a fist in the original simple one-subject scene. In one clear ordered motion, "
            "the performer relaxes the left fist, extends the left arm straight forward, "
            "visibly rotates the open palm flat and facing down, and holds the fully extended "
            "palm-down terminal pose through the end. The full arm and flat downward-facing "
            "palm remain visible. The shot stays continuous, the illumination remains stable, "
            "and the final frame is temporally coherent."
        ),
        "utf8_sha256": "4fe014a09ff2cdf863d2777deebcc6af17e869d5a7f1b336a48edc35b0fea0d7",
    },
    "reach_incomplete": {
        "text": (
            "A continuous medium shot shows a single adult performer holding the left hand "
            "as a fist in the original simple one-subject scene. The performer performs only "
            "the correct prefix: relax the left fist, begin one short forward extension, then "
            "stop and hold the hand near the torso through the end. The arm never reaches "
            "full extension, and the open palm never rotates into the flat downward-facing "
            "terminal pose. The shot stays continuous, the illumination remains stable, and "
            "the final frame is temporally coherent."
        ),
        "utf8_sha256": "87d95c49fafcf2339a6449400a87c3ee41e427a6f435637b3edadb72bbfb6f87",
    },
}
PROMPT_BUNDLE_SHA256 = "72f3e046966eeeacb18f74d05675a4dcec814498b50f982f6773e64a5605b4d0"

CELL_SPECS: tuple[Mapping[str, Any], ...] = (
    {"seed_slot": "seed1", "group_id": "sp4-a", "visible_gpus": [0, 1, 2, 3],
     "iid": "00435ad621c44fac", "seed": 2026080821, "family": "arms",
     "source_ids": {
         "action": "pair5-t2v-reserve4-v1-00435ad621c44fac-action",
         "incomplete": "pair5-t2v-reserve4-v1-00435ad621c44fac-incomplete"}},
    {"seed_slot": "seed1", "group_id": "sp4-b", "visible_gpus": [4, 5, 6, 7],
     "iid": "71ba57892bd043df", "seed": 2026080824, "family": "reach",
     "source_ids": {
         "action": "pair5-t2v-reserve4-v1-71ba57892bd043df-action",
         "incomplete": "pair5-t2v-reserve4-v1-71ba57892bd043df-incomplete"}},
    {"seed_slot": "seed2", "group_id": "sp4-a", "visible_gpus": [0, 1, 2, 3],
     "iid": "00435ad621c44fac", "seed": 2026080921, "family": "arms",
     "source_ids": {
         "action": "pair5-t2v-reserve4-seed2-00435ad621c44fac-action",
         "incomplete": "pair5-t2v-reserve4-seed2-00435ad621c44fac-incomplete"}},
    {"seed_slot": "seed2", "group_id": "sp4-b", "visible_gpus": [4, 5, 6, 7],
     "iid": "71ba57892bd043df", "seed": 2026080924, "family": "reach",
     "source_ids": {
         "action": "pair5-t2v-reserve4-seed2-71ba57892bd043df-action",
         "incomplete": "pair5-t2v-reserve4-seed2-71ba57892bd043df-incomplete"}},
)


class FitRepairExact8PlanError(RuntimeError):
    """Raised before mutable prompts or widened repair data can pass."""


def fail(message: str) -> NoReturn:
    raise FitRepairExact8PlanError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise FitRepairExact8PlanError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def plain_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata, resolved = path.lstat(), path.resolve(strict=True)
    except OSError as error:
        raise FitRepairExact8PlanError(f"{label} is unavailable") from error
    require(
        resolved == path and stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain file",
    )
    return resolved


def load_json(
    value: str | Path, label: str, expected_sha256: Optional[str] = None,
) -> tuple[dict[str, Any], Path, str]:
    path = plain_file(value, label)
    raw, observed = path.read_bytes(), file_sha256(path)
    if expected_sha256 is not None:
        require(
            SHA256_RE.fullmatch(expected_sha256) is not None
            and observed == expected_sha256, f"{label} SHA-256 differs",
        )
    try:
        result = json.loads(
            raw, object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FitRepairExact8PlanError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FitRepairExact8PlanError(f"{label} is not valid JSON") from error
    require(type(result) is dict, f"{label} must be an object")
    require(raw == canonical_json_bytes(result) + b"\n", f"{label} is not canonical JSON")
    return result, path, observed


def write_create_only(path: Path, raw: bytes, label: str) -> str:
    require(
        path.is_absolute() and path.parent.is_dir() and not path.parent.is_symlink()
        and not path.exists() and not path.is_symlink(),
        f"{label} must be a fresh absolute path",
    )
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    observed = hashlib.sha256(raw).hexdigest()
    require(file_sha256(path) == observed, f"{label} write replay differs")
    return observed


def _assert_prompt_freeze() -> None:
    for name, prompt in PROMPTS.items():
        require(
            hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest()
            == prompt["utf8_sha256"], f"{name} prompt SHA differs",
        )
    require(object_sha256(PROMPTS) == PROMPT_BUNDLE_SHA256, "prompt bundle SHA differs")


def _slot_cells(slot: str) -> list[Mapping[str, Any]]:
    return [row for row in CELL_SPECS if row["seed_slot"] == slot]


def _new_candidate_id(cell: Mapping[str, Any], branch: str) -> str:
    return f"pair5-t2v-fit-repair-v1-{cell['seed_slot']}-{cell['iid']}-{branch}"


def repair_spec_value(source: Mapping[str, Any], slot: str) -> dict[str, Any]:
    _assert_prompt_freeze()
    require(slot in {"seed1", "seed2"}, "seed slot differs")
    repaired = copy.deepcopy(source)
    expected = {
        (cell["source_ids"][branch], branch): cell
        for cell in _slot_cells(slot) for branch in ADMISSION_BRANCH_ORDER
    }
    found: set[tuple[str, str]] = set()
    for group in repaired.get("groups", []):
        for candidate in group.get("candidates", []):
            key = (candidate.get("candidate_id"), candidate.get("semantic_branch"))
            if key not in expected:
                continue
            cell, branch = expected[key], key[1]
            require(
                group.get("group_id") == cell["group_id"]
                and group.get("visible_gpus") == cell["visible_gpus"]
                and candidate.get("analysis_split") == "fit"
                and candidate.get("seed") == cell["seed"],
                f"source fit cell binding differs: {key[0]}",
            )
            prompt = PROMPTS[f"{cell['family']}_{branch}"]
            candidate["candidate_id"] = _new_candidate_id(cell, branch)
            candidate["full_t2v_caption"] = prompt["text"]
            candidate["full_t2v_caption_utf8_sha256"] = prompt["utf8_sha256"]
            found.add(key)
    require(found == set(expected), "source fit repair candidate closure differs")
    try:
        return bank_contract.validate_root_spec(repaired)
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise FitRepairExact8PlanError(str(error)) from error


def _materialize_slot(
    slot: str, source_path: Path, source_sha: str,
    source: Mapping[str, Any], output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repaired = repair_spec_value(source, slot)
    spec_dir = output / "sealed-repair-specs"
    spec_dir.mkdir(mode=0o700, exist_ok=True)
    repaired_path = spec_dir / f"{slot}-fit-repair-root-spec-v1.json"
    repaired_raw = bank_contract.canonical_json_bytes(repaired) + b"\n"
    repaired_sha = write_create_only(repaired_path, repaired_raw, "repaired root spec")
    candidate_dir = output / f"{slot}-candidate-plan"
    try:
        manifest = bank_contract.materialize_plan(
            spec_path=repaired_path, expected_sha256=repaired_sha,
            output_dir=candidate_dir,
        )
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise FitRepairExact8PlanError(str(error)) from error
    manifest_path = candidate_dir / "manifest.json"
    expected_ids = {
        _new_candidate_id(cell, branch): (cell, branch)
        for cell in _slot_cells(slot) for branch in ADMISSION_BRANCH_ORDER
    }
    tasks: list[dict[str, Any]] = []
    for record in manifest["candidate_records"]:
        if record["candidate_id"] not in expected_ids:
            continue
        envelope_path = plain_file(record["path"], "repair candidate envelope")
        require(file_sha256(envelope_path) == record["sha256"], "envelope SHA differs")
        try:
            envelope = bank_contract.load_candidate_envelope(envelope_path, repaired_sha)
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise FitRepairExact8PlanError(str(error)) from error
        cell, branch = expected_ids[record["candidate_id"]]
        candidate = envelope["candidate"]
        tasks.append({
            "seed_slot": slot, "root_spec_path": str(repaired_path),
            "root_spec_sha256": repaired_sha,
            "candidate_spec_path": str(envelope_path),
            "candidate_spec_sha256": record["sha256"],
            "group_id": envelope["group_id"],
            "visible_gpus": list(envelope["visible_gpus"]),
            "ordinal": envelope["ordinal"], "candidate_id": candidate["candidate_id"],
            "analysis_split": candidate["analysis_split"],
            "calibration_group_id": candidate["calibration_group_id"],
            "semantic_branch": branch, "seed": candidate["seed"],
            "prompt_utf8_sha256": candidate["full_t2v_caption_utf8_sha256"],
            "source_geometry_video": candidate["geometry_source_video"],
            "source_geometry_video_sha256": candidate["geometry_source_video_sha256"],
            "_order": CELL_SPECS.index(cell),
        })
    require(len(tasks) == 4, f"{slot} repair must contain exactly four tasks")
    tasks.sort(key=lambda row: (
        row["_order"], ADMISSION_BRANCH_ORDER.index(row["semantic_branch"])
    ))
    for row in tasks:
        row.pop("_order")
    ref = {
        "seed_slot": slot,
        "source_root_spec_path": str(source_path),
        "source_root_spec_sha256": source_sha,
        "repaired_root_spec_path": str(repaired_path),
        "repaired_root_spec_sha256": repaired_sha,
        "candidate_plan_manifest_path": str(manifest_path),
        "candidate_plan_manifest_sha256": file_sha256(manifest_path),
        "candidate_plan_manifest_digest": manifest["manifest_digest"],
    }
    return ref, tasks


def _cells_and_shards(
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    for cell in CELL_SPECS:
        rows = [
            row for row in tasks
            if row["seed_slot"] == cell["seed_slot"] and row["group_id"] == cell["group_id"]
        ]
        require(
            len(rows) == 2
            and [row["semantic_branch"] for row in rows] == list(ADMISSION_BRANCH_ORDER)
            and len({row["seed"] for row in rows}) == 1
            and len({row["source_geometry_video"] for row in rows}) == 1
            and len({row["source_geometry_video_sha256"] for row in rows}) == 1,
            f"repair cell pair binding differs: {cell['seed_slot']}/{cell['group_id']}",
        )
        cells.append({
            "calibration_group_id": rows[0]["calibration_group_id"],
            "seed_slot": cell["seed_slot"], "group_id": cell["group_id"],
            "visible_gpus": list(cell["visible_gpus"]), "seed": cell["seed"],
            "iid": cell["iid"], "family": cell["family"],
            "candidate_ids": [row["candidate_id"] for row in rows],
            "branch_order": list(ADMISSION_BRANCH_ORDER),
            "same_source_geometry": True,
            "same_seed_and_official_gaussian_required": True,
        })
    shards: list[dict[str, Any]] = []
    for group_id, visible in GROUP_LAYOUT:
        selected = [dict(row) for row in tasks if row["group_id"] == group_id]
        require(
            len(selected) == 4
            and all(row["visible_gpus"] == visible for row in selected)
            and [row["semantic_branch"] for row in selected]
            == ["action", "incomplete", "action", "incomplete"],
            f"{group_id} exact4 shard differs",
        )
        shards.append({
            "shard_id": f"fit-repair-{group_id}-exact4-v1",
            "group_id": group_id, "visible_gpus": visible,
            "candidate_ids": [row["candidate_id"] for row in selected],
            "candidate_count": 4,
        })
    return cells, shards


def _fixed_header() -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA, "plan_id": PLAN_ID,
        "experiment_id": "BOX-EXP-011", "analysis_split": "fit",
        "repair_kind": "preregistered_data_authoring_repair_from_locked_blind_failures",
        "scientific_objective_changed": False, "admission_threshold_changed": False,
        "locked_blind_authority": dict(LOCKED_BLIND_AUTHORITY),
        "prompt_freeze": {
            "prompts": {key: dict(value) for key, value in PROMPTS.items()},
            "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
            "frozen_before_any_new_media": True,
        },
        "formal_candidate_count": 8, "comparator_cell_count": 4,
        "branch_order": list(ADMISSION_BRANCH_ORDER),
        "execution_contract": {
            "formal_dataset": "fit_action_incomplete_repair_exact8",
            "formal_generation_invocation_count": 8,
            "diagnostic_task_count": 0, "diagnostic_generation_allowed": False,
            "action_and_incomplete_each_require_independent_full81_pass": True,
            "complete_pair_pass_target": [4, 4],
            "num_frames": 81, "num_inference_steps": 40,
            "same_gaussian_action_incomplete_required_per_cell": True,
            "topology": "one_model_replica_world4_dp1_sp4",
            "shards_execute_strictly_serial": True,
            "candidates_serial_inside_shard": True,
            "generated_media_is_editor_input_or_target": False,
            "optimizer_created": False, "optimizer_authorized": False,
            "training_authorized": False,
        },
    }


def build_plan(
    *, seed1_spec: str | Path, expected_seed1_spec_sha256: str,
    seed2_spec: str | Path, expected_seed2_spec_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    require(
        expected_seed1_spec_sha256 == SEED1_SPEC_SHA256
        and expected_seed2_spec_sha256 == SEED2_SPEC_SHA256,
        "source reserve4 spec authority differs",
    )
    output = Path(output_dir)
    require(
        output.is_absolute() and output != Path("/") and not output.exists()
        and not output.is_symlink() and output.parent.is_dir()
        and not output.parent.is_symlink(),
        "plan output directory must be fresh and absolute",
    )
    paths = {
        "seed1": plain_file(seed1_spec, "seed1 root spec"),
        "seed2": plain_file(seed2_spec, "seed2 root spec"),
    }
    expected = {"seed1": SEED1_SPEC_SHA256, "seed2": SEED2_SPEC_SHA256}
    sources: dict[str, Mapping[str, Any]] = {}
    source_refs: list[dict[str, Any]] = []
    for slot in ("seed1", "seed2"):
        try:
            spec, observed = bank_contract.load_sealed_spec(paths[slot], expected[slot])
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise FitRepairExact8PlanError(str(error)) from error
        sources[slot] = spec
        source_refs.append(
            {"seed_slot": slot, "path": str(paths[slot]), "file_sha256": observed}
        )
    output.mkdir(mode=0o700)
    repaired_refs: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for slot in ("seed1", "seed2"):
        ref, rows = _materialize_slot(
            slot, paths[slot], expected[slot], sources[slot], output
        )
        repaired_refs.append(ref)
        tasks.extend(rows)
    tasks.sort(key=lambda row: (
        next(i for i, cell in enumerate(CELL_SPECS)
             if cell["seed_slot"] == row["seed_slot"] and cell["group_id"] == row["group_id"]),
        ADMISSION_BRANCH_ORDER.index(row["semantic_branch"]),
    ))
    cells, shards = _cells_and_shards(tasks)
    unsigned = {
        **_fixed_header(), "source_specs": source_refs,
        "repaired_specs": repaired_refs, "admission_tasks": tasks,
        "seed_cells": cells, "shards": shards,
    }
    value = {**unsigned, "plan_digest": object_sha256(unsigned)}
    plan_path = output / PLAN_FILENAME
    plan_sha = write_create_only(
        plan_path, canonical_json_bytes(value) + b"\n", "exact8 plan"
    )
    return {**value, "_path": str(plan_path), "_file_sha256": plan_sha}


def load_plan(
    path: str | Path, expected_sha256: str,
) -> tuple[dict[str, Any], Path, str]:
    value, resolved, observed = load_json(path, "fit repair exact8 plan", expected_sha256)
    unsigned = dict(value)
    declared = unsigned.pop("plan_digest", None)
    require(
        isinstance(declared, str) and SHA256_RE.fullmatch(declared) is not None
        and object_sha256(unsigned) == declared,
        "fit repair exact8 plan schema/digest differs",
    )
    fixed = _fixed_header()
    require(
        all(value.get(key) == expected for key, expected in fixed.items()),
        "fit repair exact8 fixed authority differs",
    )
    source_refs, repaired_refs, tasks = (
        value.get("source_specs"), value.get("repaired_specs"),
        value.get("admission_tasks"),
    )
    require(
        isinstance(source_refs, list) and isinstance(repaired_refs, list)
        and isinstance(tasks, list) and len(source_refs) == len(repaired_refs) == 2
        and len(tasks) == 8,
        "fit repair exact8 reference/task closure differs",
    )
    expected_source = {"seed1": SEED1_SPEC_SHA256, "seed2": SEED2_SPEC_SHA256}
    for source_ref, repaired_ref in zip(source_refs, repaired_refs):
        slot = source_ref.get("seed_slot")
        require(
            slot in expected_source and repaired_ref.get("seed_slot") == slot
            and source_ref.get("file_sha256") == expected_source[slot],
            "source spec reference differs",
        )
        source_path = plain_file(source_ref["path"], f"{slot} source spec")
        try:
            source, source_sha = bank_contract.load_sealed_spec(
                source_path, expected_source[slot]
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise FitRepairExact8PlanError(str(error)) from error
        repaired_path = plain_file(
            repaired_ref["repaired_root_spec_path"], f"{slot} repaired spec"
        )
        expected_raw = (
            bank_contract.canonical_json_bytes(repair_spec_value(source, slot)) + b"\n"
        )
        repaired_sha = hashlib.sha256(expected_raw).hexdigest()
        require(
            repaired_path.read_bytes() == expected_raw
            and repaired_ref.get("source_root_spec_path") == str(source_path)
            and repaired_ref.get("source_root_spec_sha256") == source_sha
            and repaired_ref.get("repaired_root_spec_sha256") == repaired_sha,
            f"{slot} repaired root replay differs",
        )
        manifest, manifest_path, manifest_sha = load_json(
            repaired_ref["candidate_plan_manifest_path"], f"{slot} candidate manifest",
            repaired_ref["candidate_plan_manifest_sha256"],
        )
        require(
            manifest.get("root_spec_raw_sha256") == repaired_sha
            and manifest.get("manifest_digest") == repaired_ref["candidate_plan_manifest_digest"]
            and str(manifest_path) == repaired_ref["candidate_plan_manifest_path"]
            and manifest_sha == repaired_ref["candidate_plan_manifest_sha256"],
            f"{slot} candidate manifest binding differs",
        )
    expected_ids = [
        _new_candidate_id(cell, branch)
        for cell in CELL_SPECS for branch in ADMISSION_BRANCH_ORDER
    ]
    require(
        [row.get("candidate_id") for row in tasks] == expected_ids
        and [row.get("semantic_branch") for row in tasks]
        == list(ADMISSION_BRANCH_ORDER) * 4,
        "fit repair exact8 task order differs",
    )
    for task in tasks:
        envelope_path = plain_file(task["candidate_spec_path"], "candidate envelope")
        require(
            file_sha256(envelope_path) == task.get("candidate_spec_sha256"),
            "candidate envelope SHA differs",
        )
        try:
            envelope = bank_contract.load_candidate_envelope(
                envelope_path, task["root_spec_sha256"]
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise FitRepairExact8PlanError(str(error)) from error
        candidate = envelope["candidate"]
        require(
            task.get("root_spec_path") == envelope_path.parents[2].joinpath(
                "sealed-repair-specs", f"{task['seed_slot']}-fit-repair-root-spec-v1.json"
            ).as_posix()
            and task.get("candidate_id") == candidate["candidate_id"]
            and task.get("group_id") == envelope["group_id"]
            and task.get("visible_gpus") == envelope["visible_gpus"]
            and task.get("ordinal") == envelope["ordinal"]
            and task.get("analysis_split") == candidate["analysis_split"] == "fit"
            and task.get("calibration_group_id") == candidate["calibration_group_id"]
            and task.get("semantic_branch") == candidate["semantic_branch"]
            and task.get("seed") == candidate["seed"]
            and task.get("prompt_utf8_sha256")
            == candidate["full_t2v_caption_utf8_sha256"]
            and task.get("source_geometry_video") == candidate["geometry_source_video"]
            and task.get("source_geometry_video_sha256")
            == candidate["geometry_source_video_sha256"],
            f"candidate task/envelope replay differs: {task.get('candidate_id')}",
        )
    cells, shards = _cells_and_shards(tasks)
    require(
        value.get("seed_cells") == cells and value.get("shards") == shards
        and "diagnostic_tasks" not in value,
        "fit repair exact8 cell/shard replay differs",
    )
    return value, resolved, observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    build.add_argument("--seed1-spec", required=True)
    build.add_argument("--expected-seed1-spec-sha256", required=True)
    build.add_argument("--seed2-spec", required=True)
    build.add_argument("--expected-seed2-spec-sha256", required=True)
    build.add_argument("--output-dir", required=True)
    validate = commands.add_parser("validate-plan")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-plan":
        value = build_plan(
            seed1_spec=args.seed1_spec,
            expected_seed1_spec_sha256=args.expected_seed1_spec_sha256,
            seed2_spec=args.seed2_spec,
            expected_seed2_spec_sha256=args.expected_seed2_spec_sha256,
            output_dir=args.output_dir,
        )
        result = {
            "plan_path": value["_path"], "plan_file_sha256": value["_file_sha256"],
            "formal_candidate_count": value["formal_candidate_count"],
            "diagnostic_task_count": value["execution_contract"]["diagnostic_task_count"],
            "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
        }
    else:
        value, resolved, observed = load_plan(args.plan, args.expected_plan_sha256)
        result = {
            "plan_path": str(resolved), "plan_file_sha256": observed,
            "plan_digest": value["plan_digest"],
            "formal_candidate_count": value["formal_candidate_count"],
            "diagnostic_task_count": value["execution_contract"]["diagnostic_task_count"],
            "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
        }
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADMISSION_BRANCH_ORDER", "CELL_SPECS", "FitRepairExact8PlanError",
    "LOCKED_BLIND_AUTHORITY", "MATERIALIZER_CONTROL_ORDER", "PLAN_SCHEMA",
    "PROMPTS", "PROMPT_BUNDLE_SHA256", "build_plan", "canonical_json_bytes",
    "load_plan", "object_sha256", "repair_spec_value",
]
