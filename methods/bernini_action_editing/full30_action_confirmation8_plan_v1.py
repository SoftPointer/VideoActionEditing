#!/usr/bin/env python3
"""Build and replay the BOX-EXP-009 confirmation action-anchor exact8 plan.

The two registered confirmation IIDs are each rendered at two frozen seeds.
Only ``action`` and ``incomplete`` are admitted, giving four seed cells and
eight exact-81-frame clips.  The other eight PAIR-v5 branch *types* remain a
separate optional diagnostic lane.  They are never prerequisites for exact8,
never review-admission rows, and never inputs to q, a_min, or an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
for _root in (METHOD_ROOT, TOOLS_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import pair_v5_t2v_calibration_bank_spec as bank_contract  # noqa: E402
import build_pair_v5_t2v_seed2_bank as seed2_builder  # noqa: E402


PLAN_SCHEMA = "bernini-full30-action-confirmation8-plan-v1"
PLAN_FILENAME = "full30-action-confirmation8-plan-v1.json"
SPEC_AUTHORITIES = {
    "seed1": "2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab",
    "seed2": "0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e",
}
AUTHORING_REGISTRY_SHA256 = (
    "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"
)
RESERVE4_SELECTION_SHA256 = (
    "a4baa1aea27f6497ca2dd615cc09b2b90eee37173f506e60ae7d630c41886be6"
)
SEED_PREFIXES = {
    "seed1": "pair5-t2v-reserve4-v1-",
    "seed2": "pair5-t2v-reserve4-seed2-",
}
CONFIRMATION_CELL_REGISTRY = (
    {
        "seed_slot": "seed1",
        "group_id": "sp4-a",
        "iid": "0c6915018a5f4d9b",
        "seed": 2026080822,
        "action_family": "head-turn-smile",
    },
    {
        "seed_slot": "seed1",
        "group_id": "sp4-b",
        "iid": "33322eb8ec1e4703",
        "seed": 2026080823,
        "action_family": "peace-wave",
    },
    {
        "seed_slot": "seed2",
        "group_id": "sp4-a",
        "iid": "0c6915018a5f4d9b",
        "seed": 2026080922,
        "action_family": "head-turn-smile",
    },
    {
        "seed_slot": "seed2",
        "group_id": "sp4-b",
        "iid": "33322eb8ec1e4703",
        "seed": 2026080923,
        "action_family": "peace-wave",
    },
)
ADMISSION_BRANCH_ORDER = ("action", "incomplete")
DIAGNOSTIC_BRANCH_ORDER = tuple(
    branch
    for branch in bank_contract.MACE_BRANCH_ORDER
    if branch not in ADMISSION_BRANCH_ORDER
)
MATERIALIZER_CONTROL_ORDER = (
    "noop",
    "camera_only",
    "appearance_only",
    "wrong_actor",
    "wrong_object",
    "generic_wrong_motion",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Confirmation8PlanError(RuntimeError):
    """Raised before a mutable or over-broad confirmation plan is accepted."""


def fail(message: str) -> NoReturn:
    raise Confirmation8PlanError(message)


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
        raise Confirmation8PlanError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def plain_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise Confirmation8PlanError(f"{label} is unavailable") from error
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
                Confirmation8PlanError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Confirmation8PlanError(f"{label} is not valid UTF-8 JSON") from error
    require(type(result) is dict, f"{label} must be a JSON object")
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
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
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


def _sealed_specs(
    seed1_path: str | Path, seed2_path: str | Path
) -> dict[str, tuple[dict[str, Any], Path, str]]:
    result: dict[str, tuple[dict[str, Any], Path, str]] = {}
    for slot, path in (("seed1", seed1_path), ("seed2", seed2_path)):
        try:
            spec, observed = bank_contract.load_sealed_spec(
                path, SPEC_AUTHORITIES[slot]
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise Confirmation8PlanError(str(error)) from error
        result[slot] = (spec, plain_file(path, f"{slot} root spec"), observed)
    try:
        derived = seed2_builder.derive_seed2_spec(result["seed1"][0], "reserve4-v1")
    except seed2_builder.PairV5T2VSeed2Error as error:
        raise Confirmation8PlanError(str(error)) from error
    require(
        derived == result["seed2"][0],
        "seed2 is not the registered seed-only derivation of seed1",
    )
    return result


def _tasks_from_candidate_plan(
    *,
    slot: str,
    spec_path: Path,
    spec_sha256: str,
    candidate_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record in candidate_plan["candidate_records"]:
        envelope_path = plain_file(record["path"], "candidate envelope")
        require(
            file_sha256(envelope_path) == record["sha256"],
            "candidate envelope SHA-256 differs",
        )
        try:
            envelope = bank_contract.load_candidate_envelope(
                envelope_path, spec_sha256
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise Confirmation8PlanError(str(error)) from error
        candidate = envelope["candidate"]
        require(
            candidate["candidate_id"] == record["candidate_id"],
            "candidate plan identity differs",
        )
        if candidate["analysis_split"] != "confirmation":
            continue
        require(
            candidate["candidate_id"].startswith(SEED_PREFIXES[slot]),
            f"{slot} candidate prefix differs",
        )
        tasks.append(
            {
                "seed_slot": slot,
                "root_spec_path": str(spec_path),
                "root_spec_sha256": spec_sha256,
                "candidate_spec_path": str(envelope_path),
                "candidate_spec_sha256": record["sha256"],
                "group_id": envelope["group_id"],
                "visible_gpus": envelope["visible_gpus"],
                "ordinal": envelope["ordinal"],
                "candidate_id": candidate["candidate_id"],
                "analysis_split": candidate["analysis_split"],
                "calibration_group_id": candidate["calibration_group_id"],
                "semantic_branch": candidate["semantic_branch"],
                "seed": candidate["seed"],
            }
        )
    return tasks


def _expected_candidate_id(cell: Mapping[str, Any], branch: str) -> str:
    return f"{SEED_PREFIXES[str(cell['seed_slot'])]}{cell['iid']}-{branch}"


def partition_confirmation_tasks(
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the frozen 40-row source and project exact8/diagnostic lanes."""

    require(len(tasks) == 40, "confirmation source inventory must contain 40 rows")
    by_identity: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        candidate_id = task.get("candidate_id")
        require(
            isinstance(candidate_id, str) and candidate_id not in by_identity,
            "confirmation source candidate identity is absent or duplicated",
        )
        by_identity[candidate_id] = task

    expected_source_ids: list[str] = []
    admission: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for cell in CONFIRMATION_CELL_REGISTRY:
        source_ids = [
            _expected_candidate_id(cell, branch)
            for branch in bank_contract.MACE_BRANCH_ORDER
        ]
        expected_source_ids.extend(source_ids)
        rows = [by_identity.get(candidate_id) for candidate_id in source_ids]
        require(all(row is not None for row in rows), "confirmation cell is incomplete")
        typed_rows = [dict(row) for row in rows if row is not None]
        for row, branch in zip(typed_rows, bank_contract.MACE_BRANCH_ORDER):
            require(
                row.get("seed_slot") == cell["seed_slot"]
                and row.get("group_id") == cell["group_id"]
                and row.get("analysis_split") == "confirmation"
                and row.get("calibration_group_id")
                == f"cell-{cell['iid']}-s{cell['seed']}"
                and row.get("seed") == cell["seed"]
                and row.get("semantic_branch") == branch
                and row.get("candidate_id") == _expected_candidate_id(cell, branch),
                f"confirmation cell registry differs: {cell['seed_slot']}/{cell['iid']}/{branch}",
            )
        admission_rows = [
            next(row for row in typed_rows if row["semantic_branch"] == branch)
            for branch in ADMISSION_BRANCH_ORDER
        ]
        diagnostic_rows = [
            next(row for row in typed_rows if row["semantic_branch"] == branch)
            for branch in DIAGNOSTIC_BRANCH_ORDER
        ]
        admission.extend(admission_rows)
        diagnostics.extend(diagnostic_rows)
        cells.append(
            {
                **dict(cell),
                "calibration_group_id": f"cell-{cell['iid']}-s{cell['seed']}",
                "admission_candidate_ids": [
                    row["candidate_id"] for row in admission_rows
                ],
                "diagnostic_candidate_ids": [
                    row["candidate_id"] for row in diagnostic_rows
                ],
                "admission_branch_order": list(ADMISSION_BRANCH_ORDER),
                "diagnostic_branch_order": list(DIAGNOSTIC_BRANCH_ORDER),
            }
        )
    require(
        [row.get("candidate_id") for row in tasks] == expected_source_ids,
        "confirmation source inventory order or identity differs",
    )
    require(
        len(admission) == 8
        and len(diagnostics) == 32
        and not (
            {row["candidate_id"] for row in admission}
            & {row["candidate_id"] for row in diagnostics}
        ),
        "confirmation exact8/diagnostic partition differs",
    )
    return admission, diagnostics, cells


def _expected_shards(
    admission: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for cell in CONFIRMATION_CELL_REGISTRY:
        selected = [
            row
            for row in admission
            if row["seed_slot"] == cell["seed_slot"]
            and row["group_id"] == cell["group_id"]
        ]
        optional = [
            row
            for row in diagnostics
            if row["seed_slot"] == cell["seed_slot"]
            and row["group_id"] == cell["group_id"]
        ]
        require(
            [row["semantic_branch"] for row in selected]
            == list(ADMISSION_BRANCH_ORDER)
            and [row["semantic_branch"] for row in optional]
            == list(DIAGNOSTIC_BRANCH_ORDER),
            "confirmation shard partition differs",
        )
        rows.append(
            {
                "shard_id": (
                    f"{cell['seed_slot']}-{cell['group_id']}-confirmation-exact8"
                ),
                "seed_slot": cell["seed_slot"],
                "group_id": cell["group_id"],
                "visible_gpus": list(selected[0]["visible_gpus"]),
                "admission_candidate_ids": [
                    row["candidate_id"] for row in selected
                ],
                "admission_candidate_count": 2,
                "optional_diagnostic_candidate_ids": [
                    row["candidate_id"] for row in optional
                ],
                "optional_diagnostic_candidate_count": 8,
            }
        )
    return rows


def _plan_value(
    *, source_specs: Sequence[Mapping[str, Any]], source_tasks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    admission, diagnostics, cells = partition_confirmation_tasks(source_tasks)
    unsigned = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "BOX-EXP-009-confirmation-action-anchor-exact8-v1",
        "experiment_id": "BOX-EXP-009",
        "analysis_split": "confirmation",
        "source_specs": [dict(row) for row in source_specs],
        "source_candidate_count": 40,
        "seed_cell_count": 4,
        "admission_candidate_count": 8,
        "diagnostic_candidate_count": 32,
        "admission_branch_order": list(ADMISSION_BRANCH_ORDER),
        "diagnostic_branch_order": list(DIAGNOSTIC_BRANCH_ORDER),
        "materializer_control_order": list(MATERIALIZER_CONTROL_ORDER),
        "seed_cells": cells,
        "admission_tasks": admission,
        "diagnostic_tasks": diagnostics,
        "shards": _expected_shards(admission, diagnostics),
        "execution_contract": {
            "formal_dataset": "confirmation_action_anchor_exact8",
            "formal_generation_invocation_count": 8,
            "optional_diagnostic_generation_is_admission_prerequisite": False,
            "optional_diagnostic_generation_is_q_input": False,
            "optional_diagnostic_generation_is_a_min_input": False,
            "optional_diagnostic_generation_is_optimizer_input": False,
            "same_state_controls_source": "official_full30_action_psiout_materializer",
            "generated_noop_camera_appearance_wrong_controls_admitted": False,
            "action_and_incomplete_each_require_independent_full81_pass": True,
            "topology": "one_model_replica_world4_dp1_sp4",
            "candidates_serial_inside_shard": True,
            "num_frames": 81,
            "num_inference_steps": 40,
            "generated_media_is_editor_input_or_target": False,
            "training_authorized": False,
            "optimizer_created": False,
            "optimizer_authorized": False,
        },
    }
    return {**unsigned, "plan_digest": object_sha256(unsigned)}


def build_plan(
    *, seed1_spec: str | Path, seed2_spec: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    require(
        output.is_absolute()
        and output != Path("/")
        and not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.parent.is_symlink(),
        "plan output directory must be fresh, absolute, and canonically parented",
    )
    specs = _sealed_specs(seed1_spec, seed2_spec)
    output.mkdir(mode=0o700)
    source_specs: list[dict[str, Any]] = []
    source_tasks: list[dict[str, Any]] = []
    for slot in ("seed1", "seed2"):
        _, spec_path, spec_sha = specs[slot]
        plan_dir = output / f"{slot}-candidate-plan"
        try:
            candidate_plan = bank_contract.materialize_plan(
                spec_path=spec_path,
                expected_sha256=spec_sha,
                output_dir=plan_dir,
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise Confirmation8PlanError(str(error)) from error
        manifest_path = plan_dir / "manifest.json"
        source_specs.append(
            {
                "seed_slot": slot,
                "root_spec_path": str(spec_path),
                "root_spec_sha256": spec_sha,
                "candidate_plan_manifest_path": str(manifest_path),
                "candidate_plan_manifest_sha256": file_sha256(manifest_path),
                "candidate_plan_manifest_digest": candidate_plan["manifest_digest"],
            }
        )
        source_tasks.extend(
            _tasks_from_candidate_plan(
                slot=slot,
                spec_path=spec_path,
                spec_sha256=spec_sha,
                candidate_plan=candidate_plan,
            )
        )
    plan = _plan_value(source_specs=source_specs, source_tasks=source_tasks)
    plan_path = output / PLAN_FILENAME
    plan_sha = write_create_only(plan_path, plan)
    return {**plan, "_path": str(plan_path), "_file_sha256": plan_sha}


def load_plan(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], Path, str]:
    plan, resolved, observed = load_json(path, "confirmation8 plan", expected_sha256)
    unsigned = dict(plan)
    declared = unsigned.pop("plan_digest", None)
    require(
        plan.get("schema_version") == PLAN_SCHEMA
        and isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and object_sha256(unsigned) == declared,
        "confirmation8 plan schema/digest differs",
    )
    refs = plan.get("source_specs")
    require(
        isinstance(refs, list)
        and len(refs) == 2
        and [row.get("seed_slot") for row in refs if isinstance(row, Mapping)]
        == ["seed1", "seed2"],
        "confirmation8 source spec registry differs",
    )
    authorities = _sealed_specs(
        refs[0]["root_spec_path"], refs[1]["root_spec_path"]
    )
    source_tasks: list[dict[str, Any]] = []
    canonical_refs: list[dict[str, Any]] = []
    for ref in refs:
        slot = ref["seed_slot"]
        _, spec_path, spec_sha = authorities[slot]
        candidate_plan, manifest_path, manifest_sha = load_json(
            ref["candidate_plan_manifest_path"],
            f"{slot} candidate plan manifest",
            ref["candidate_plan_manifest_sha256"],
        )
        require(
            ref.get("root_spec_path") == str(spec_path)
            and ref.get("root_spec_sha256") == spec_sha
            and ref.get("candidate_plan_manifest_path") == str(manifest_path)
            and ref.get("candidate_plan_manifest_sha256") == manifest_sha
            and ref.get("candidate_plan_manifest_digest")
            == candidate_plan.get("manifest_digest"),
            f"{slot} source plan binding differs",
        )
        canonical_refs.append(dict(ref))
        source_tasks.extend(
            _tasks_from_candidate_plan(
                slot=slot,
                spec_path=spec_path,
                spec_sha256=spec_sha,
                candidate_plan=candidate_plan,
            )
        )
    expected = _plan_value(source_specs=canonical_refs, source_tasks=source_tasks)
    require(plan == expected, "confirmation8 plan replay differs")
    return plan, resolved, observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    build.add_argument("--seed1-spec", required=True)
    build.add_argument("--seed2-spec", required=True)
    build.add_argument("--split", choices=("confirmation",), required=True)
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
            seed2_spec=args.seed2_spec,
            output_dir=args.output_dir,
        )
        summary = {
            "plan_path": value["_path"],
            "plan_file_sha256": value["_file_sha256"],
            "admission_candidate_count": value["admission_candidate_count"],
            "diagnostic_candidate_count": value["diagnostic_candidate_count"],
        }
    else:
        value, resolved, observed = load_plan(
            args.plan, args.expected_plan_sha256
        )
        summary = {
            "plan_path": str(resolved),
            "plan_file_sha256": observed,
            "plan_digest": value["plan_digest"],
            "admission_candidate_count": value["admission_candidate_count"],
        }
    print(canonical_json_bytes(summary).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
