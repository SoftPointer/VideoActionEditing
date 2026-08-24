#!/usr/bin/env python3
"""Build the closed BOX-EXP-010 minimal cross-anchor topup4 exact8 plan.

The frozen selection contains four comparator cells.  This module projects
only the ordered ``action``/``incomplete`` pair for each cell from the sealed
authoring plan.  No diagnostic task is represented, rendered, or accepted by
the execution plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence

import full30_action_minimal_cross_anchor_topup4_plan_v1 as source_contract


PLAN_SCHEMA = "bernini-full30-action-topup4-exact8-plan-v1"
PLAN_ID = "BOX-EXP-010-minimal-cross-anchor-topup4-exact8-v1"
PLAN_FILENAME = "full30-action-topup4-exact8-plan-v1.json"
SELECTION_FILE_SHA256 = source_contract.SELECTION_FILE_SHA256
PARENT_REGISTRY_FILE_SHA256 = source_contract.PARENT_REGISTRY_FILE_SHA256
ADMISSION_BRANCH_ORDER = ("action", "incomplete")
MATERIALIZER_CONTROL_ORDER = (
    "noop",
    "camera_only",
    "appearance_only",
    "wrong_actor",
    "wrong_object",
    "generic_wrong_motion",
)
GROUP_LAYOUT = (("sp4-a", [0, 1, 2, 3]), ("sp4-b", [4, 5, 6, 7]))
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Topup4Exact8PlanError(RuntimeError):
    """Raised before a widened or mutable topup4 plan is admitted."""


def fail(message: str) -> NoReturn:
    raise Topup4Exact8PlanError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Topup4Exact8PlanError("value is not canonical finite JSON") from error


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
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise Topup4Exact8PlanError(f"{label} is unavailable") from error
    require(
        resolved == path
        and stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain file",
    )
    return resolved


def load_json(
    value: str | Path, label: str, expected_sha256: Optional[str] = None
) -> tuple[dict[str, Any], Path, str]:
    path = plain_file(value, label)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        require(
            SHA256_RE.fullmatch(expected_sha256) is not None
            and observed == expected_sha256,
            f"{label} SHA-256 differs",
        )
    try:
        result = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Topup4Exact8PlanError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Topup4Exact8PlanError(f"{label} is not valid JSON") from error
    require(type(result) is dict, f"{label} must be an object")
    require(raw == canonical_json_bytes(result) + b"\n", f"{label} is not canonical JSON")
    return result, path, observed


def write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    require(
        path.is_absolute()
        and path.parent.is_dir()
        and not path.parent.is_symlink()
        and not path.exists()
        and not path.is_symlink(),
        "plan output must be a fresh absolute path",
    )
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
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
    require(file_sha256(path) == observed, "plan write replay differs")
    return observed


def _cell_projection(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(tasks) == 8, "topup4 must contain exactly eight tasks")
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in tasks:
        cell_id = task.get("calibration_group_id")
        require(isinstance(cell_id, str), "task calibration cell differs")
        if cell_id in seen:
            continue
        seen.add(cell_id)
        rows = [dict(row) for row in tasks if row.get("calibration_group_id") == cell_id]
        require(
            len(rows) == 2
            and [row.get("semantic_branch") for row in rows]
            == list(ADMISSION_BRANCH_ORDER),
            f"cell {cell_id} is not one action/incomplete pair",
        )
        first, second = rows
        require(
            first.get("seed") == second.get("seed")
            and first.get("group_id") == second.get("group_id")
            and first.get("visible_gpus") == second.get("visible_gpus")
            and first.get("analysis_split") == second.get("analysis_split") == "fit",
            f"cell {cell_id} pair binding differs",
        )
        cells.append(
            {
                "calibration_group_id": cell_id,
                "group_id": first["group_id"],
                "visible_gpus": list(first["visible_gpus"]),
                "seed": first["seed"],
                "candidate_ids": [row["candidate_id"] for row in rows],
                "branch_order": list(ADMISSION_BRANCH_ORDER),
            }
        )
    require(len(cells) == 4, "topup4 must contain exactly four comparator cells")
    require(
        [row["group_id"] for row in cells] == ["sp4-a", "sp4-a", "sp4-b", "sp4-b"],
        "topup4 cell/group order differs",
    )
    return cells


def _shard_projection(
    tasks: Sequence[Mapping[str, Any]], cells: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for group_id, visible_gpus in GROUP_LAYOUT:
        selected = [dict(row) for row in tasks if row.get("group_id") == group_id]
        selected_cells = [
            row["calibration_group_id"] for row in cells if row["group_id"] == group_id
        ]
        require(
            len(selected) == 4
            and all(row.get("visible_gpus") == visible_gpus for row in selected)
            and [row.get("semantic_branch") for row in selected]
            == ["action", "incomplete", "action", "incomplete"],
            f"{group_id} exact4 shard differs",
        )
        shards.append(
            {
                "shard_id": f"topup4-{group_id}-exact4-v1",
                "group_id": group_id,
                "visible_gpus": visible_gpus,
                "calibration_group_ids": selected_cells,
                "candidate_ids": [row["candidate_id"] for row in selected],
                "candidate_count": 4,
            }
        )
    return shards


def _plan_value(
    *, source_plan: Mapping[str, Any], source_path: Path, source_sha256: str
) -> dict[str, Any]:
    tasks = [dict(row) for row in source_plan["tasks"]]
    cells = _cell_projection(tasks)
    shards = _shard_projection(tasks, cells)
    require(
        source_plan.get("generation_invocation_count") == 8
        and source_plan.get("seed_cell_count") == 4
        and source_plan.get("branch_order") == list(ADMISSION_BRANCH_ORDER)
        and source_plan.get("analysis_split") == "fit",
        "source exact8 authority differs",
    )
    unsigned = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": PLAN_ID,
        "experiment_id": "BOX-EXP-010",
        "analysis_split": "fit",
        "source_authoring_plan": {
            "path": str(source_path),
            "file_sha256": source_sha256,
            "plan_digest": source_plan["plan_digest"],
        },
        "selection": dict(source_plan["selection"]),
        "parent_registry": dict(source_plan["parent_registry"]),
        "root_spec": dict(source_plan["root_spec"]),
        "formal_candidate_count": 8,
        "comparator_cell_count": 4,
        "branch_order": list(ADMISSION_BRANCH_ORDER),
        "admission_tasks": tasks,
        "seed_cells": cells,
        "shards": shards,
        "execution_contract": {
            "formal_dataset": "minimal_cross_anchor_topup4_exact8",
            "formal_generation_invocation_count": 8,
            "diagnostic_task_count": 0,
            "diagnostic_generation_allowed": False,
            "diagnostic_generation_is_admission_prerequisite": False,
            "same_state_controls_source": "official_full30_action_psiout_materializer",
            "action_and_incomplete_each_require_independent_full81_pass": True,
            "topology": "one_model_replica_world4_dp1_sp4",
            "shards_execute_strictly_serial": True,
            "candidates_serial_inside_shard": True,
            "num_frames": 81,
            "num_inference_steps": 40,
            "same_gaussian_action_incomplete_required_per_cell": True,
            "generated_media_is_editor_input_or_target": False,
            "training_authorized": False,
            "optimizer_created": False,
            "optimizer_authorized": False,
        },
    }
    return {**unsigned, "plan_digest": object_sha256(unsigned)}


def build_plan(
    *, selection: str | Path, expected_selection_sha256: str,
    parent_registry: str | Path, expected_parent_registry_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    require(
        output.is_absolute()
        and output != Path("/")
        and not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.parent.is_symlink(),
        "plan output directory must be fresh and absolute",
    )
    output.mkdir(mode=0o700)
    source_dir = output / "sealed-authoring-plan"
    try:
        built = source_contract.materialize_plan(
            selection_path=selection,
            expected_selection_sha256=expected_selection_sha256,
            registry_path=parent_registry,
            expected_registry_sha256=expected_parent_registry_sha256,
            output_dir=source_dir,
        )
        source_plan, source_path, source_sha = source_contract.load_plan(
            built["_path"], built["_file_sha256"]
        )
    except source_contract.MinimalCrossAnchorPlanError as error:
        raise Topup4Exact8PlanError(str(error)) from error
    value = _plan_value(
        source_plan=source_plan, source_path=source_path, source_sha256=source_sha
    )
    plan_path = output / PLAN_FILENAME
    plan_sha = write_create_only(plan_path, value)
    return {**value, "_path": str(plan_path), "_file_sha256": plan_sha}


def load_plan(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], Path, str]:
    value, resolved, observed = load_json(path, "topup4 exact8 plan", expected_sha256)
    unsigned = dict(value)
    declared = unsigned.pop("plan_digest", None)
    require(
        value.get("schema_version") == PLAN_SCHEMA
        and value.get("plan_id") == PLAN_ID
        and isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and object_sha256(unsigned) == declared,
        "topup4 exact8 plan schema/digest differs",
    )
    ref = value.get("source_authoring_plan", {})
    try:
        source_plan, source_path, source_sha = source_contract.load_plan(
            ref["path"], ref["file_sha256"]
        )
    except (KeyError, source_contract.MinimalCrossAnchorPlanError) as error:
        raise Topup4Exact8PlanError("source authoring plan binding differs") from error
    require(
        ref.get("plan_digest") == source_plan["plan_digest"],
        "source authoring plan digest differs",
    )
    expected = _plan_value(
        source_plan=source_plan, source_path=source_path, source_sha256=source_sha
    )
    require(value == expected, "topup4 exact8 plan replay differs")
    require(
        "diagnostic_tasks" not in value
        and value["execution_contract"]["diagnostic_task_count"] == 0,
        "diagnostic task authority leaked into topup4 exact8",
    )
    return value, resolved, observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    build.add_argument("--selection", required=True)
    build.add_argument("--expected-selection-sha256", required=True)
    build.add_argument("--parent-registry", required=True)
    build.add_argument("--expected-parent-registry-sha256", required=True)
    build.add_argument("--output-dir", required=True)
    validate = commands.add_parser("validate-plan")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-plan":
        value = build_plan(
            selection=args.selection,
            expected_selection_sha256=args.expected_selection_sha256,
            parent_registry=args.parent_registry,
            expected_parent_registry_sha256=args.expected_parent_registry_sha256,
            output_dir=args.output_dir,
        )
        result = {
            "plan_path": value["_path"],
            "plan_file_sha256": value["_file_sha256"],
            "formal_candidate_count": value["formal_candidate_count"],
            "diagnostic_task_count": value["execution_contract"]["diagnostic_task_count"],
        }
    else:
        value, resolved, observed = load_plan(args.plan, args.expected_plan_sha256)
        result = {
            "plan_path": str(resolved),
            "plan_file_sha256": observed,
            "plan_digest": value["plan_digest"],
            "formal_candidate_count": value["formal_candidate_count"],
            "diagnostic_task_count": value["execution_contract"]["diagnostic_task_count"],
        }
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADMISSION_BRANCH_ORDER",
    "MATERIALIZER_CONTROL_ORDER",
    "PLAN_SCHEMA",
    "Topup4Exact8PlanError",
    "build_plan",
    "canonical_json_bytes",
    "load_plan",
    "object_sha256",
]
