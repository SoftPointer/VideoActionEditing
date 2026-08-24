#!/usr/bin/env python3
"""Plan, run, and audit the sealed reserve4 PAIR-v5 media on one SP4 island.

This utility is deliberately upstream of ``Phi_v1``.  It renders complete
ten-branch cells from the two preregistered reserve4 specs, but generated RGB,
latents, and Gaussian tensors remain authoring evidence only.  Even a complete
generation audit does not authorize representation extraction until separate
blind full-81-frame reviews have been sealed, and it never makes generated
artifacts an editor input or target.

The only admitted confirmation sequence is::

  build-plan --split confirmation ...
  smoke-sp4 ...  # sealed first candidate, full native 40-step path
  run-sp4 --seed-slot seed1 --group-id sp4-a ...
  run-sp4 --seed-slot seed1 --group-id sp4-b ...
  run-sp4 --seed-slot seed2 --group-id sp4-a ...
  run-sp4 --seed-slot seed2 --group-id sp4-b ...
  audit ... --generation-root <each non-empty shard output>

Empty shards are absent from the sealed plan and therefore need not be run.
The smoke output is deleted and only its exact receipt may admit formal40.
Candidate generation is serial inside a shard so the 64-GiB holder never
silently becomes a two-replica data-parallel job.  Each torchrun rank uses a
private node-local cache root; NFS COMGR temporary storage is rejected.
Checkpoint deserialization is rank-serialized through an authenticated
node-local flock until each renderer is GPU-resident and host arenas are
trimmed.  A WORLD4 completion barrier precedes all source/tokenizer setup; T2V
keeps T5 on each rank GPU until renderer retirement, while the compile smoke
must byte-match the r10 MP4/Gaussian/latent authority.  After sampling, WORLD4
retires the renderer before rank zero loads the deferred decode-only VAE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pair_v5_t2v_calibration_bank_spec as bank_contract  # noqa: E402
import build_pair_v5_t2v_seed2_bank as seed2_builder  # noqa: E402


PLAN_SCHEMA = "bernini-reserve4-confirmation-generation-sp4-plan-v2"
GAP_SCHEMA = "bernini-reserve4-fixed-generation-gap-receipt-v1"
SHARD_RECEIPT_SCHEMA = "bernini-reserve4-fixed-generation-shard-receipt-v1"
AUDIT_RECEIPT_SCHEMA = "bernini-reserve4-fixed-generation-audit-receipt-v1"
COMPILE_SMOKE_SCHEMA = "bernini-generic-action-confirmation40-compile-smoke-v3"
RANK_EXEC = (
    METHOD_ROOT
    / "scripts/auh_generic_action_confirmation_data_prep_rank_exec_v1.sh"
)
PREPROCESSING_TOOL_SHA256 = {
    "tools/build_renderer_dataset.py": (
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
    ),
    "tools/materialize_vae.py": (
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
    ),
}
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
CONFIRMATION_CELL_REGISTRY = (
    {
        "seed_slot": "seed1",
        "group_id": "sp4-a",
        "iid": "0c6915018a5f4d9b",
        "seed": 2026080822,
    },
    {
        "seed_slot": "seed1",
        "group_id": "sp4-b",
        "iid": "33322eb8ec1e4703",
        "seed": 2026080823,
    },
    {
        "seed_slot": "seed2",
        "group_id": "sp4-a",
        "iid": "0c6915018a5f4d9b",
        "seed": 2026080922,
    },
    {
        "seed_slot": "seed2",
        "group_id": "sp4-b",
        "iid": "33322eb8ec1e4703",
        "seed": 2026080923,
    },
)
SEED_PREFIXES = {
    "seed1": "pair5-t2v-reserve4-v1-",
    "seed2": "pair5-t2v-reserve4-seed2-",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
EMPTY_FILE_SHA256 = hashlib.sha256(b"").hexdigest()


class Reserve4GenerationError(RuntimeError):
    """Raised before a mutable, partial, or over-authorized run can pass."""


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
        raise Reserve4GenerationError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Reserve4GenerationError(message)


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _plain_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Reserve4GenerationError(f"{label} is unavailable: {path}") from error
    _require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be a plain file",
    )
    return path.resolve(strict=True)


def _plain_dir(value: str | Path, label: str) -> Path:
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Reserve4GenerationError(f"{label} is unavailable: {path}") from error
    _require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be a plain directory",
    )
    return path.resolve(strict=True)


def _load_json(
    value: str | Path, label: str, expected_sha256: Optional[str] = None
) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(value, label)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require(
            SHA256_RE.fullmatch(expected_sha256) is not None
            and observed == expected_sha256,
            f"{label} SHA-256 differs",
        )
    try:
        result = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Reserve4GenerationError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Reserve4GenerationError(f"{label} is not valid UTF-8 JSON") from error
    _require(type(result) is dict, f"{label} must be a JSON object")
    return result, path, observed


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    _require(
        path.is_absolute() and path.parent.is_dir() and not path.parent.is_symlink(),
        f"output parent is unavailable: {path}",
    )
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
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
    _require(file_sha256(path) == observed, f"published bytes failed replay: {path}")
    return observed


def _sealed_specs(
    seed1_path: str | Path, seed2_path: str | Path
) -> dict[str, tuple[dict[str, Any], Path, str]]:
    results: dict[str, tuple[dict[str, Any], Path, str]] = {}
    for slot, path in (("seed1", seed1_path), ("seed2", seed2_path)):
        expected = SPEC_AUTHORITIES[slot]
        try:
            spec, observed = bank_contract.load_sealed_spec(path, expected)
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise Reserve4GenerationError(str(error)) from error
        resolved = _plain_file(path, f"reserve4 {slot} spec")
        _require(observed == expected, f"reserve4 {slot} authority differs")
        results[slot] = (spec, resolved, observed)
    try:
        derived = seed2_builder.derive_seed2_spec(results["seed1"][0], "reserve4-v1")
    except seed2_builder.PairV5T2VSeed2Error as error:
        raise Reserve4GenerationError(str(error)) from error
    _require(
        derived == results["seed2"][0],
        "seed2 is not the registered seed-only derivation of seed1",
    )
    return results


def _cell_proof(tasks: Sequence[Mapping[str, Any]], label: str) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for task in tasks:
        key = (
            str(task["seed_slot"]),
            str(task["group_id"]),
            str(task["calibration_group_id"]),
            int(task["seed"]),
        )
        cells.setdefault(key, []).append(task)
    proofs: list[dict[str, Any]] = []
    for key, rows in cells.items():
        branches = [str(row["semantic_branch"]) for row in rows]
        _require(
            branches == list(bank_contract.MACE_BRANCH_ORDER),
            f"{label} cell {key!r} is not one complete ordered ten-branch cell",
        )
        _require(
            len({row["analysis_split"] for row in rows}) == 1,
            f"{label} cell split differs",
        )
        proofs.append(
            {
                "seed_slot": key[0],
                "group_id": key[1],
                "calibration_group_id": key[2],
                "seed": key[3],
                "analysis_split": rows[0]["analysis_split"],
                "candidate_ids": [row["candidate_id"] for row in rows],
                "branch_order": branches,
                "complete_ten_branch_cell": True,
            }
        )
    return proofs


def _tasks_from_candidate_plan(
    *, slot: str, spec_path: Path, spec_sha256: str, plan: Mapping[str, Any], split: str
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record in plan["candidate_records"]:
        envelope_path = _plain_file(record["path"], "candidate envelope")
        _require(
            file_sha256(envelope_path) == record["sha256"],
            "candidate plan/envelope SHA-256 differs",
        )
        try:
            envelope = bank_contract.load_candidate_envelope(
                envelope_path, spec_sha256
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise Reserve4GenerationError(str(error)) from error
        candidate = envelope["candidate"]
        _require(
            candidate["candidate_id"] == record["candidate_id"],
            "candidate plan identity differs",
        )
        if candidate["analysis_split"] != split:
            continue
        _require(
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


def _validate_confirmation_scope(
    tasks: Sequence[Mapping[str, Any]], cell_proofs: Sequence[Mapping[str, Any]]
) -> None:
    """Bind the 40 rows to the two missing reserve IIDs and two sealed seeds."""

    _require(len(tasks) == 40 and len(cell_proofs) == 4, "confirmation40 size differs")
    observed = []
    for expected, proof in zip(CONFIRMATION_CELL_REGISTRY, cell_proofs):
        iid = str(expected["iid"])
        seed_slot = str(expected["seed_slot"])
        prefix = SEED_PREFIXES[seed_slot]
        candidate_ids = [
            f"{prefix}{iid}-{branch}" for branch in bank_contract.MACE_BRANCH_ORDER
        ]
        _require(
            proof.get("seed_slot") == seed_slot
            and proof.get("group_id") == expected["group_id"]
            and proof.get("calibration_group_id")
            == f"cell-{iid}-s{expected['seed']}"
            and proof.get("seed") == expected["seed"]
            and proof.get("analysis_split") == "confirmation"
            and proof.get("candidate_ids") == candidate_ids
            and proof.get("branch_order") == list(bank_contract.MACE_BRANCH_ORDER)
            and proof.get("complete_ten_branch_cell") is True,
            f"confirmation cell registry differs: {seed_slot}/{iid}",
        )
        observed.extend(candidate_ids)
    _require(
        [row.get("candidate_id") for row in tasks] == observed,
        "confirmation40 candidate order or identity differs",
    )


def build_plan(
    *, seed1_spec: str | Path, seed2_spec: str | Path, split: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    _require(split == "confirmation", "only the confirmation split is present")
    output = Path(output_dir)
    _require(
        output.is_absolute()
        and output != Path("/")
        and not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.parent.is_symlink(),
        "plan output must be a fresh absolute directory with a plain parent",
    )
    authorities = _sealed_specs(seed1_spec, seed2_spec)
    output.mkdir(mode=0o700)
    tasks: list[dict[str, Any]] = []
    source_plans: list[dict[str, Any]] = []
    for slot in ("seed1", "seed2"):
        _, spec_path, spec_sha = authorities[slot]
        candidate_plan_dir = output / f"{slot}-candidate-plan"
        try:
            candidate_plan = bank_contract.materialize_plan(
                spec_path=spec_path,
                expected_sha256=spec_sha,
                output_dir=candidate_plan_dir,
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise Reserve4GenerationError(str(error)) from error
        candidate_manifest_path = candidate_plan_dir / "manifest.json"
        source_plans.append(
            {
                "seed_slot": slot,
                "root_spec_path": str(spec_path),
                "root_spec_sha256": spec_sha,
                "candidate_plan_manifest_path": str(candidate_manifest_path),
                "candidate_plan_manifest_sha256": file_sha256(candidate_manifest_path),
                "candidate_plan_manifest_digest": candidate_plan["manifest_digest"],
            }
        )
        tasks.extend(
            _tasks_from_candidate_plan(
                slot=slot,
                spec_path=spec_path,
                spec_sha256=spec_sha,
                plan=candidate_plan,
                split=split,
            )
        )
    _require(len(tasks) == 40, f"{split} reserve4 scope must contain exactly 40 clips")
    cell_proofs = _cell_proof(tasks, "generation plan")
    _require(len(cell_proofs) == 4, f"{split} reserve4 scope must contain four seed cells")
    _validate_confirmation_scope(tasks, cell_proofs)
    shards = []
    for slot in ("seed1", "seed2"):
        for group_id, visible_gpus in bank_contract.GROUP_LAYOUT:
            candidate_ids = [
                row["candidate_id"]
                for row in tasks
                if row["seed_slot"] == slot and row["group_id"] == group_id
            ]
            if candidate_ids:
                shards.append(
                    {
                        "shard_id": f"{slot}-{group_id}-{split}",
                        "seed_slot": slot,
                        "group_id": group_id,
                        "visible_gpus": visible_gpus,
                        "candidate_ids": candidate_ids,
                        "candidate_count": len(candidate_ids),
                    }
                )
    _require(
        sum(row["candidate_count"] for row in shards) == 40,
        "generation shard coverage differs",
    )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": f"reserve4-{split}-two-seed-complete10-v2",
        "analysis_split": split,
        "source_specs": source_plans,
        "generation_invocation_count": 40,
        "seed_cell_count": 4,
        "branch_order": list(bank_contract.MACE_BRANCH_ORDER),
        "tasks": tasks,
        "cell_proofs": cell_proofs,
        "shards": shards,
        "execution_contract": {
            "topology": "one_model_replica_world4_dp1_sp4",
            "candidate_order": "sealed_spec_group_then_ordinal",
            "candidates_serial_inside_shard": True,
            "fit_first_does_not_reclassify_confirmation": True,
            "generated_media_role": "representation_authoring_evidence_only",
            "generated_media_is_editor_input_or_target": False,
            "generated_latent_or_gaussian_is_editor_input_or_target": False,
            "visual_review_required_before_phi_v1_extraction": True,
            "optimizer_authorized": False,
        },
    }
    plan = {**plan, "plan_digest": object_sha256(plan)}
    plan_path = output / "reserve4-fixed-generation-plan-v2.json"
    plan_sha = _write_create_only(plan_path, plan)
    gap = {
        "schema_version": GAP_SCHEMA,
        "plan_path": str(plan_path),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "analysis_split": split,
        "expected_candidate_count": 40,
        "observed_candidate_count": 0,
        "missing_candidate_ids": [row["candidate_id"] for row in tasks],
        "complete_ten_branch_seed_cells": 0,
        "independent_full81_review_count": 0,
        "phi_v1_extraction_authorized": False,
        "optimizer_authorized": False,
        "generated_media_is_editor_input_or_target": False,
    }
    gap = {**gap, "receipt_digest": object_sha256(gap)}
    _write_create_only(output / "reserve4-generation-gap-before-run-v1.json", gap)
    return {**plan, "_path": str(plan_path), "_file_sha256": plan_sha}


def load_plan(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], Path, str]:
    plan, plan_path, observed = _load_json(path, "reserve4 generation plan", expected_sha256)
    _require(plan.get("schema_version") == PLAN_SCHEMA, "generation plan schema differs")
    declared = plan.get("plan_digest")
    _require(SHA256_RE.fullmatch(str(declared)) is not None, "plan digest differs")
    unsigned = dict(plan)
    del unsigned["plan_digest"]
    _require(object_sha256(unsigned) == declared, "plan digest differs")
    split = plan.get("analysis_split")
    _require(split == "confirmation", "plan is not the confirmation split")
    source_specs = plan.get("source_specs")
    _require(type(source_specs) is list and len(source_specs) == 2, "plan source specs differ")
    refs = {row.get("seed_slot"): row for row in source_specs if type(row) is dict}
    _require(set(refs) == {"seed1", "seed2"}, "plan source slots differ")
    authorities = _sealed_specs(
        refs["seed1"]["root_spec_path"], refs["seed2"]["root_spec_path"]
    )
    expected_tasks: list[dict[str, Any]] = []
    for slot in ("seed1", "seed2"):
        _, spec_path, spec_sha = authorities[slot]
        ref = refs[slot]
        _require(
            ref.get("root_spec_path") == str(spec_path)
            and ref.get("root_spec_sha256") == spec_sha,
            "plan root spec binding differs",
        )
        candidate_manifest, manifest_path, manifest_sha = _load_json(
            ref["candidate_plan_manifest_path"], "candidate plan manifest",
            ref["candidate_plan_manifest_sha256"],
        )
        _require(
            manifest_sha == ref["candidate_plan_manifest_sha256"]
            and candidate_manifest.get("manifest_digest")
            == ref["candidate_plan_manifest_digest"],
            "candidate plan manifest binding differs",
        )
        expected_tasks.extend(
            _tasks_from_candidate_plan(
                slot=slot,
                spec_path=spec_path,
                spec_sha256=spec_sha,
                plan=candidate_manifest,
                split=split,
            )
        )
    _require(plan.get("tasks") == expected_tasks, "plan task bytes/order differ")
    _require(len(expected_tasks) == 40, "plan task count differs")
    expected_cells = _cell_proof(expected_tasks, "validated plan")
    _validate_confirmation_scope(expected_tasks, expected_cells)
    _require(plan.get("cell_proofs") == expected_cells, "plan cell proof differs")
    _require(plan.get("seed_cell_count") == 4, "plan seed-cell count differs")
    _require(plan.get("generation_invocation_count") == 40, "plan invocation count differs")
    contract = plan.get("execution_contract", {})
    _require(
        contract.get("topology") == "one_model_replica_world4_dp1_sp4"
        and contract.get("candidates_serial_inside_shard") is True
        and contract.get("generated_media_is_editor_input_or_target") is False
        and contract.get("generated_latent_or_gaussian_is_editor_input_or_target") is False
        and contract.get("visual_review_required_before_phi_v1_extraction") is True
        and contract.get("optimizer_authorized") is False,
        "plan execution/authority contract differs",
    )
    return plan, plan_path, observed


def _expected_interpretation() -> dict[str, Any]:
    return {
        "calibration_evidence_only": True,
        "event_qualified_from_generation_receipt": False,
        "action_success_not_implied": True,
        "training_performed": False,
        "parameter_update_performed": False,
        "optimizer_authorized": False,
        "t2v_media_as_rv2v_policy_candidate_forbidden": True,
        "donor_or_pseudo_target_use_forbidden": True,
    }


def _validate_candidate_receipt(
    task: Mapping[str, Any], receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    import infer_pair_v5_t2v_calibration_bank as renderer

    try:
        envelope = bank_contract.load_candidate_envelope(
            task["candidate_spec_path"], task["root_spec_sha256"]
        )
        receipt = renderer._load_pair_receipt(receipt_path)  # type: ignore[attr-defined]
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise Reserve4GenerationError(str(error)) from error
    candidate = envelope["candidate"]
    expected_visible = ",".join(str(item) for item in task["visible_gpus"])
    _require(
        receipt["root_spec_raw_sha256"] == task["root_spec_sha256"]
        and receipt["candidate_envelope_sha256"] == task["candidate_spec_sha256"]
        and receipt["candidate"] == candidate
        and receipt["group_id"] == task["group_id"]
        and receipt["visible_gpus"] == task["visible_gpus"]
        and receipt["ordinal"] == task["ordinal"]
        and receipt["runtime_topology"]
        == {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": expected_visible,
        },
        f"candidate receipt/spec binding differs: {task['candidate_id']}",
    )
    _require(
        receipt["sampling_contract"] == bank_contract.SAMPLING_CONTRACT
        and receipt["semantic_input_closure"] == bank_contract.SEMANTIC_INPUT_CLOSURE
        and receipt["artifact_use_contract"] == bank_contract.ARTIFACT_USE_CONTRACT
        and receipt["split_contract"] == bank_contract.SPLIT_CONTRACT
        and receipt["interpretation"] == _expected_interpretation(),
        f"candidate receipt exceeds generation-only authority: {task['candidate_id']}",
    )
    native_path = _plain_file(receipt["native_receipt_path"], "native receipt")
    try:
        native_receipt = renderer._load_json(native_path, "receipt-bound native receipt")  # type: ignore[attr-defined]
        _require(
            file_sha256(native_path) == receipt["native_receipt_sha256"],
            "native receipt SHA-256 differs",
        )
        native_artifacts = renderer._verify_native_receipt(  # type: ignore[attr-defined]
            native_receipt, candidate
        )
        expected_artifacts = {
            "mp4": native_artifacts["mp4"],
            "predecode_clean_latent": native_artifacts["predecode_clean_latent"],
            "official_initial_gaussian": native_artifacts["official_initial_gaussian"],
        }
        _require(
            native_artifacts["native_receipt_digest"] == receipt["native_receipt_digest"]
            and receipt["artifacts"] == expected_artifacts,
            "candidate/native artifact binding differs",
        )
        for name, artifact in receipt["artifacts"].items():
            renderer._verify_file_artifact(  # type: ignore[attr-defined]
                artifact, f"{task['candidate_id']} {name}"
            )
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise Reserve4GenerationError(str(error)) from error
    return receipt, candidate


def _gaussian_cell_proofs(
    validated: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for task, receipt in validated:
        key = (
            str(task["seed_slot"]),
            str(task["group_id"]),
            str(task["calibration_group_id"]),
            int(task["seed"]),
        )
        cells.setdefault(key, []).append((task, receipt))
    proofs: list[dict[str, Any]] = []
    identity_fields = (
        "raw_value_sha256", "content_sha256", "shape", "dtype", "stored_dtype",
        "generator_initial_seed",
    )
    for key, rows in cells.items():
        branches = [task["semantic_branch"] for task, _ in rows]
        _require(
            branches == list(bank_contract.MACE_BRANCH_ORDER),
            f"rendered cell {key!r} is incomplete or reordered",
        )
        gaussians = [receipt["artifacts"]["official_initial_gaussian"] for _, receipt in rows]
        identities = {
            object_sha256({field: artifact.get(field) for field in identity_fields})
            for artifact in gaussians
        }
        _require(len(identities) == 1, f"rendered cell {key!r} did not reuse one Gaussian")
        first = gaussians[0]
        proofs.append(
            {
                "seed_slot": key[0],
                "group_id": key[1],
                "calibration_group_id": key[2],
                "seed": key[3],
                "branch_order": branches,
                "official_gaussian_raw_value_sha256": first["raw_value_sha256"],
                "official_gaussian_content_sha256": first["content_sha256"],
                "all_ten_official_gaussian_tensor_values_byte_equal": True,
            }
        )
    return proofs


_ALLOWED_NODE_LOCAL_FILESYSTEMS = {"ext2/ext3", "xfs", "tmpfs"}
_SMOKE_ARTIFACT_NAMES = (
    "mp4",
    "official_initial_gaussian",
    "predecode_clean_latent",
)
_R10_SMOKE_ARTIFACT_AUTHORITY = {
    "mp4": {
        "file_sha256": "22d5b3d50d800b7d25debc1bf42c2ec3ea462ba84035f1afd42d021488239b70",
        "metadata_digest": "bfa65faebf57dc34645813141949379a2101f88f3b8b54b00fd8c9b1836aee0e",
    },
    "official_initial_gaussian": {
        "file_sha256": "fea764f90c2958265cd2d63df5a250f34c578bc347c42901415b275823c2b30a",
        "metadata_digest": "88482b29dd12d7d7e968a9d56317508326c3927ee8c78a7d5e8130ac8ba4c08c",
    },
    "predecode_clean_latent": {
        "file_sha256": "fb6a7330d4b604a4a032c5d208d5482be2e40850b8e7b5b6190cb83c17cbd1cd",
        "metadata_digest": "c942a0f27e087cf776ba69c2f504cb43c7e7f9b6ce50b864d3b8af4d485623c4",
    },
}


def _filesystem_type(path: Path) -> str:
    try:
        result = subprocess.run(
            ["stat", "-f", "-c", "%T", "--", str(path)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Reserve4GenerationError(
            "cannot identify the COMGR scratch filesystem"
        ) from error
    value = result.stdout.strip()
    _require(value in _ALLOWED_NODE_LOCAL_FILESYSTEMS, "COMGR scratch is not node-local")
    return value


def _runtime_binding(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    python = _plain_file(args.python, "Python executable")
    _require(os.access(python, os.X_OK), "Python executable is not executable")
    bernini_root = _plain_dir(args.bernini_root, "Bernini root")
    veomni_root = _plain_dir(args.veomni_root, "VeOmni root")
    checkpoint = _plain_dir(args.checkpoint, "checkpoint")
    checkpoint_manifest = _plain_file(
        args.checkpoint_content_manifest, "checkpoint content manifest"
    )
    _require(
        REVISION_RE.fullmatch(args.method_source_revision) is not None
        and SHA256_RE.fullmatch(args.method_source_archive_sha256) is not None,
        "method source revision/archive SHA differs",
    )
    _require(1024 <= args.master_port <= 65535, "master port differs")
    worker = _plain_file(
        METHOD_ROOT / "infer_pair_v5_t2v_calibration_bank.py",
        "generation worker",
    )
    rank_exec = _plain_file(RANK_EXEC, "per-rank cache wrapper")
    _require(os.access(rank_exec, os.X_OK), "per-rank cache wrapper is not executable")
    preprocessing: dict[str, str] = {}
    for relative, expected in PREPROCESSING_TOOL_SHA256.items():
        member = _plain_file(METHOD_ROOT / relative, f"release {relative}")
        observed = file_sha256(member)
        _require(observed == expected, f"release preprocessing identity differs: {relative}")
        preprocessing[relative] = observed
    scratch_value = os.environ.get("GADP_NODE_LOCAL_SCRATCH")
    expected_fstype = os.environ.get("GADP_NODE_LOCAL_SCRATCH_FSTYPE")
    _require(bool(scratch_value), "authenticated node-local scratch is absent")
    scratch = _plain_dir(str(scratch_value), "authenticated node-local scratch")
    observed_fstype = _filesystem_type(scratch)
    _require(
        expected_fstype == observed_fstype,
        "authenticated node-local scratch filesystem changed",
    )
    _require(
        os.environ.get("NATIVE_SERIALIZED_HOST_LOAD_REQUIRED") == "1",
        "serialized host checkpoint-load requirement is absent",
    )
    _require(
        os.environ.get("NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED") == "1",
        "T2V rank-GPU text-encoder residency requirement is absent",
    )
    load_lock_value = os.environ.get("NATIVE_V_AXIS_LOAD_LOCK")
    _require(bool(load_lock_value), "serialized host checkpoint-load lock is absent")
    load_lock = _plain_file(str(load_lock_value), "serialized host checkpoint-load lock")
    load_lock_metadata = load_lock.lstat()
    _require(
        load_lock.parent == scratch
        and load_lock_metadata.st_uid == os.geteuid()
        and load_lock_metadata.st_nlink == 1
        and stat.S_IMODE(load_lock_metadata.st_mode) == 0o400
        and load_lock_metadata.st_size == 0
        and file_sha256(load_lock) == EMPTY_FILE_SHA256,
        "serialized host checkpoint-load lock identity differs",
    )
    binding = {
        "method_root": str(METHOD_ROOT),
        "python": {"path": str(python), "sha256": file_sha256(python)},
        "bernini_root": str(bernini_root),
        "veomni_root": str(veomni_root),
        "checkpoint": str(checkpoint),
        "checkpoint_content_manifest": {
            "path": str(checkpoint_manifest),
            "sha256": file_sha256(checkpoint_manifest),
        },
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "generation_worker": {
            "path": str(worker),
            "sha256": file_sha256(worker),
        },
        "rank_cache_wrapper": {
            "path": str(rank_exec),
            "sha256": file_sha256(rank_exec),
        },
        "preprocessing_tools": preprocessing,
        "node_local_scratch": {
            "path": str(scratch),
            "filesystem_type": observed_fstype,
        },
        "serialized_host_checkpoint_load": {
            "required": True,
            "environment_variable": "NATIVE_V_AXIS_LOAD_LOCK",
            "path": str(load_lock),
            "sha256": EMPTY_FILE_SHA256,
            "mode": "0400",
            "parent_is_authenticated_node_local_scratch": True,
            "lock_held_through_model_to_rank_gpu_and_malloc_trim": True,
        },
        "t2v_text_encoder_rank_gpu_residency": {
            "required": True,
            "environment_variable": "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED",
            "official_model_sample_preserved": True,
            "exact_positional_cpu_offload_request_suppressed_once_per_rank": True,
            "all_other_to_requests_delegated": True,
            "text_encoder_retired_only_with_renderer": True,
        },
    }
    return binding, python, worker, rank_exec, scratch


def _candidate_command(
    args: argparse.Namespace,
    *,
    task: Mapping[str, Any],
    candidate_output: Path,
    python: Path,
    worker: Path,
    rank_exec: Path,
) -> list[str]:
    return [
        str(python),
        "-B",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=4",
        "--master_addr=127.0.0.1",
        f"--master_port={args.master_port}",
        "--no_python",
        str(rank_exec),
        str(worker),
        "--candidate-spec",
        str(task["candidate_spec_path"]),
        "--expected-root-spec-sha256",
        str(task["root_spec_sha256"]),
        "--output-dir",
        str(candidate_output),
        "--bernini-root",
        args.bernini_root,
        "--veomni-root",
        args.veomni_root,
        "--checkpoint",
        args.checkpoint,
        "--checkpoint-content-manifest",
        args.checkpoint_content_manifest,
        "--method-source-revision",
        args.method_source_revision,
        "--method-source-archive-sha256",
        args.method_source_archive_sha256,
    ]


def _candidate_environment(
    *,
    expected_visible: str,
    python: Path,
    scratch: Path,
    cache_token: str,
) -> dict[str, str]:
    _require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", cache_token) is not None,
        "rank cache token differs",
    )
    environment = dict(os.environ)
    load_lock = environment.get("NATIVE_V_AXIS_LOAD_LOCK")
    _require(
        environment.get("NATIVE_SERIALIZED_HOST_LOAD_REQUIRED") == "1"
        and bool(load_lock)
        and environment.get("NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED") == "1",
        "serialized host/T5 residency environment is absent",
    )
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MODELING_BACKEND": "hf",
            "ROCR_VISIBLE_DEVICES": expected_visible,
            "TMPDIR": str(scratch),
            "GADP_RANK_CACHE_TOKEN": cache_token,
            "GADP_RANK_PYTHON_BIN": str(python),
            "GADP_METHOD_ROOT": str(METHOD_ROOT),
            "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
            "NATIVE_V_AXIS_LOAD_LOCK": str(load_lock),
            "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED": "1",
        }
    )
    return environment


def _smoke_task_binding(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": task["candidate_id"],
        "seed_slot": task["seed_slot"],
        "group_id": task["group_id"],
        "visible_gpus": task["visible_gpus"],
        "analysis_split": task["analysis_split"],
        "ordinal": task["ordinal"],
        "candidate_spec_path": task["candidate_spec_path"],
        "candidate_spec_sha256": task["candidate_spec_sha256"],
        "root_spec_sha256": task["root_spec_sha256"],
    }


def _smoke_artifact_identities(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in _SMOKE_ARTIFACT_NAMES:
        artifact = receipt.get("artifacts", {}).get(name)
        _require(isinstance(artifact, Mapping), f"compile-smoke artifact differs: {name}")
        file_identity = artifact.get("sha256")
        _require(
            isinstance(file_identity, str)
            and SHA256_RE.fullmatch(file_identity) is not None,
            f"compile-smoke artifact SHA-256 differs: {name}",
        )
        metadata = dict(artifact)
        metadata.pop("path", None)
        rows.append(
            {
                "name": name,
                "file_sha256": file_identity,
                "metadata_digest": object_sha256(metadata),
            }
        )
    return rows


def _validate_r10_smoke_artifact_parity(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = [
        {"name": name, **_R10_SMOKE_ARTIFACT_AUTHORITY[name]}
        for name in _SMOKE_ARTIFACT_NAMES
    ]
    _require(
        [dict(row) for row in rows] == expected,
        "r11 compile smoke differs from the byte-exact r10 artifact authority",
    )


def load_compile_smoke_receipt(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], Path, str]:
    import infer_native_identity_generation_canary as native

    value, source, observed = _load_json(
        path, "compile-smoke receipt", expected_sha256
    )
    _require(
        source.read_bytes() == canonical_json_bytes(value) + b"\n",
        "compile-smoke receipt bytes are not canonical JSON",
    )
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    plan = value.get("plan")
    task = value.get("smoke_task")
    runtime = value.get("runtime")
    evidence = value.get("candidate_evidence")
    _require(
        set(value)
        == {
            "schema_version",
            "plan",
            "smoke_task",
            "runtime",
            "candidate_evidence",
            "world_size",
            "full_native_sampling_steps",
            "formal_candidate_count_at_gate",
            "disposable_output_deleted",
            "compile_smoke_passed",
            "training_performed",
            "optimizer_authorized",
            "receipt_digest",
        }
        and value.get("schema_version") == COMPILE_SMOKE_SCHEMA
        and declared == object_sha256(unsigned)
        and value.get("world_size") == 4
        and value.get("full_native_sampling_steps") == 40
        and value.get("formal_candidate_count_at_gate") == 0
        and value.get("disposable_output_deleted") is True
        and value.get("compile_smoke_passed") is True
        and value.get("training_performed") is False
        and value.get("optimizer_authorized") is False,
        "compile-smoke receipt schema/digest/authority differs",
    )
    _require(
        isinstance(plan, Mapping)
        and set(plan) == {"path", "file_sha256", "plan_digest"}
        and all(
            isinstance(plan.get(field), str)
            and SHA256_RE.fullmatch(str(plan.get(field))) is not None
            for field in ("file_sha256", "plan_digest")
        )
        and Path(str(plan.get("path"))).is_absolute(),
        "compile-smoke plan binding differs",
    )
    _require(
        isinstance(task, Mapping)
        and set(task)
        == {
            "candidate_id",
            "seed_slot",
            "group_id",
            "visible_gpus",
            "analysis_split",
            "ordinal",
            "candidate_spec_path",
            "candidate_spec_sha256",
            "root_spec_sha256",
        }
        and task.get("seed_slot") == "seed1"
        and task.get("group_id") == "sp4-a"
        and task.get("analysis_split") == "confirmation"
        and task.get("visible_gpus") == [0, 1, 2, 3]
        and type(task.get("ordinal")) is int
        and Path(str(task.get("candidate_spec_path"))).is_absolute()
        and all(
            isinstance(task.get(field), str)
            and SHA256_RE.fullmatch(str(task.get(field))) is not None
            for field in ("candidate_spec_sha256", "root_spec_sha256")
        ),
        "compile-smoke sealed first-task binding differs",
    )
    _require(
        isinstance(runtime, Mapping)
        and set(runtime)
        == {
            "method_root",
            "python",
            "bernini_root",
            "veomni_root",
            "checkpoint",
            "checkpoint_content_manifest",
            "method_source_revision",
            "method_source_archive_sha256",
            "generation_worker",
            "rank_cache_wrapper",
            "preprocessing_tools",
            "node_local_scratch",
            "serialized_host_checkpoint_load",
            "t2v_text_encoder_rank_gpu_residency",
        }
        and runtime.get("preprocessing_tools") == PREPROCESSING_TOOL_SHA256
        and isinstance(runtime.get("node_local_scratch"), Mapping)
        and set(runtime["node_local_scratch"]) == {"path", "filesystem_type"}
        and Path(str(runtime["node_local_scratch"].get("path"))).is_absolute()
        and runtime["node_local_scratch"].get("filesystem_type")
        in _ALLOWED_NODE_LOCAL_FILESYSTEMS
        and isinstance(runtime.get("serialized_host_checkpoint_load"), Mapping)
        and runtime["serialized_host_checkpoint_load"]
        == {
            "required": True,
            "environment_variable": "NATIVE_V_AXIS_LOAD_LOCK",
            "path": runtime["serialized_host_checkpoint_load"].get("path"),
            "sha256": EMPTY_FILE_SHA256,
            "mode": "0400",
            "parent_is_authenticated_node_local_scratch": True,
            "lock_held_through_model_to_rank_gpu_and_malloc_trim": True,
        }
        and Path(
            str(runtime["serialized_host_checkpoint_load"].get("path"))
        ).is_absolute()
        and Path(
            str(runtime["serialized_host_checkpoint_load"].get("path"))
        ).parent
        == Path(str(runtime["node_local_scratch"].get("path")))
        and runtime.get("t2v_text_encoder_rank_gpu_residency")
        == {
            "required": True,
            "environment_variable": "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED",
            "official_model_sample_preserved": True,
            "exact_positional_cpu_offload_request_suppressed_once_per_rank": True,
            "all_other_to_requests_delegated": True,
            "text_encoder_retired_only_with_renderer": True,
        }
        and all(
            isinstance(runtime.get(field), str)
            and Path(str(runtime.get(field))).is_absolute()
            for field in (
                "method_root",
                "bernini_root",
                "veomni_root",
                "checkpoint",
            )
        )
        and REVISION_RE.fullmatch(str(runtime.get("method_source_revision")))
        is not None
        and SHA256_RE.fullmatch(
            str(runtime.get("method_source_archive_sha256"))
        )
        is not None,
        "compile-smoke runtime binding differs",
    )
    for field in (
        "python",
        "checkpoint_content_manifest",
        "generation_worker",
        "rank_cache_wrapper",
    ):
        reference = runtime.get(field)
        _require(
            isinstance(reference, Mapping)
            and set(reference) == {"path", "sha256"}
            and Path(str(reference.get("path"))).is_absolute()
            and SHA256_RE.fullmatch(str(reference.get("sha256"))) is not None,
            f"compile-smoke runtime {field} binding differs",
        )
    _require(
        isinstance(evidence, Mapping)
        and set(evidence)
        == {
            "candidate_receipt_file_sha256",
            "candidate_receipt_digest",
            "native_receipt_file_sha256",
            "native_receipt_digest",
            "resource_lifecycle",
            "artifact_identities",
        }
        and all(
            SHA256_RE.fullmatch(str(evidence.get(field))) is not None
            for field in (
                "candidate_receipt_file_sha256",
                "candidate_receipt_digest",
                "native_receipt_file_sha256",
                "native_receipt_digest",
            )
        )
        and isinstance(evidence.get("artifact_identities"), list)
        and [row.get("name") for row in evidence["artifact_identities"]]
        == list(_SMOKE_ARTIFACT_NAMES),
        "compile-smoke candidate evidence differs",
    )
    try:
        native.validate_t2v_resource_lifecycle(
            evidence.get("resource_lifecycle"), require_serialized_load=True
        )
    except native.NativeIdentityCanaryError as error:
        raise Reserve4GenerationError(
            "compile-smoke did not prove WORLD4 load completion before sampling"
        ) from error
    for row in evidence["artifact_identities"]:
        _require(
            isinstance(row, Mapping)
            and set(row) == {"name", "file_sha256", "metadata_digest"}
            and SHA256_RE.fullmatch(str(row.get("file_sha256"))) is not None
            and SHA256_RE.fullmatch(str(row.get("metadata_digest"))) is not None,
            "compile-smoke artifact identity row differs",
        )
    _validate_r10_smoke_artifact_parity(evidence["artifact_identities"])
    return value, source, observed


def _validate_compile_smoke_for_runtime(
    args: argparse.Namespace,
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_sha: str,
    runtime: Mapping[str, Any],
) -> Mapping[str, Any]:
    receipt, _, _ = load_compile_smoke_receipt(
        args.compile_smoke_receipt, args.expected_compile_smoke_receipt_sha256
    )
    expected_plan = {
        "path": str(plan_path),
        "file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
    }
    _require(receipt["plan"] == expected_plan, "compile-smoke plan replay differs")
    _require(
        receipt["smoke_task"] == _smoke_task_binding(plan["tasks"][0]),
        "compile-smoke first-task replay differs",
    )
    _require(receipt["runtime"] == runtime, "compile-smoke runtime replay differs")
    return receipt


def _delete_disposable_smoke_root(root: Path, scratch: Path) -> None:
    try:
        resolved_root = root.resolve(strict=True)
        resolved_scratch = scratch.resolve(strict=True)
    except OSError as error:
        raise Reserve4GenerationError("disposable smoke root is unavailable") from error
    _require(
        root == resolved_root
        and not root.is_symlink()
        and root.is_dir()
        and resolved_root.parent == resolved_scratch
        and root.name.startswith("generic-action-compile-smoke."),
        "refusing unsafe disposable smoke cleanup",
    )
    shutil.rmtree(root)
    _require(not root.exists() and not root.is_symlink(), "smoke output cleanup failed")


def run_compile_smoke_sp4(args: argparse.Namespace) -> int:
    import infer_pair_v5_t2v_calibration_bank as renderer

    plan, plan_path, plan_sha = load_plan(args.plan, args.expected_plan_sha256)
    task = plan["tasks"][0]
    _require(
        task["seed_slot"] == "seed1"
        and task["group_id"] == "sp4-a"
        and task["analysis_split"] == "confirmation"
        and task["visible_gpus"] == [0, 1, 2, 3],
        "sealed compile-smoke candidate is not first confirmation/SP4-A task",
    )
    expected_visible = "0,1,2,3"
    _require(
        os.environ.get("ROCR_VISIBLE_DEVICES") == expected_visible,
        "compile-smoke ROCR_VISIBLE_DEVICES differs",
    )
    runtime, python, worker, rank_exec, scratch = _runtime_binding(args)
    receipt_output = Path(args.receipt_output)
    _require(
        receipt_output.is_absolute()
        and receipt_output.parent.is_dir()
        and not receipt_output.parent.is_symlink()
        and not receipt_output.exists()
        and not receipt_output.is_symlink(),
        "compile-smoke receipt output must be fresh",
    )
    smoke_root = Path(
        tempfile.mkdtemp(prefix="generic-action-compile-smoke.", dir=scratch)
    ).resolve(strict=True)
    candidate_output = smoke_root / str(task["candidate_id"])
    command = _candidate_command(
        args,
        task=task,
        candidate_output=candidate_output,
        python=python,
        worker=worker,
        rank_exec=rank_exec,
    )
    environment = _candidate_environment(
        expected_visible=expected_visible,
        python=python,
        scratch=scratch,
        cache_token="compile-smoke-" + object_sha256(_smoke_task_binding(task))[:20],
    )
    candidate_evidence: Optional[dict[str, Any]] = None
    try:
        subprocess.run(command, check=True, env=environment)
        candidate_receipt_path = (
            candidate_output / "pair-v5-t2v-calibration-receipt.json"
        )
        candidate_receipt, _ = _validate_candidate_receipt(
            task, candidate_receipt_path
        )
        native_receipt = renderer._load_json(  # type: ignore[attr-defined]
            _plain_file(candidate_receipt["native_receipt_path"], "native receipt"),
            "compile-smoke native receipt",
        )
        resource_lifecycle = renderer.native.validate_t2v_resource_lifecycle(  # type: ignore[attr-defined]
            native_receipt.get("resource_lifecycle"), require_serialized_load=True
        )
        candidate_evidence = {
            "candidate_receipt_file_sha256": file_sha256(candidate_receipt_path),
            "candidate_receipt_digest": candidate_receipt["receipt_digest"],
            "native_receipt_file_sha256": candidate_receipt["native_receipt_sha256"],
            "native_receipt_digest": candidate_receipt["native_receipt_digest"],
            "resource_lifecycle": resource_lifecycle,
            "artifact_identities": _smoke_artifact_identities(candidate_receipt),
        }
        _validate_r10_smoke_artifact_parity(
            candidate_evidence["artifact_identities"]
        )
    except subprocess.CalledProcessError as error:
        raise Reserve4GenerationError(
            "full native40 compile smoke failed; formal40 remains forbidden"
        ) from error
    finally:
        _delete_disposable_smoke_root(smoke_root, scratch)
    _require(candidate_evidence is not None, "compile-smoke evidence is absent")
    unsigned = {
        "schema_version": COMPILE_SMOKE_SCHEMA,
        "plan": {
            "path": str(plan_path),
            "file_sha256": plan_sha,
            "plan_digest": plan["plan_digest"],
        },
        "smoke_task": _smoke_task_binding(task),
        "runtime": runtime,
        "candidate_evidence": candidate_evidence,
        "world_size": 4,
        "full_native_sampling_steps": 40,
        "formal_candidate_count_at_gate": 0,
        "disposable_output_deleted": True,
        "compile_smoke_passed": True,
        "training_performed": False,
        "optimizer_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    receipt_sha = _write_create_only(receipt_output, receipt)
    load_compile_smoke_receipt(receipt_output, receipt_sha)
    return 0


def run_sp4(args: argparse.Namespace) -> int:
    plan, plan_path, plan_sha = load_plan(args.plan, args.expected_plan_sha256)
    matching = [
        row for row in plan["tasks"]
        if row["seed_slot"] == args.seed_slot and row["group_id"] == args.group_id
    ]
    _require(matching, "requested shard is empty or absent from the sealed plan")
    expected_visible = ",".join(str(item) for item in matching[0]["visible_gpus"])
    _require(
        all(
            row["visible_gpus"] == matching[0]["visible_gpus"]
            and row["analysis_split"] == plan["analysis_split"]
            for row in matching
        ),
        "requested shard topology/split differs",
    )
    _require(
        os.environ.get("ROCR_VISIBLE_DEVICES") == expected_visible,
        f"ROCR_VISIBLE_DEVICES must equal sealed shard mapping {expected_visible}",
    )
    runtime, python, worker, rank_exec, scratch = _runtime_binding(args)
    _validate_compile_smoke_for_runtime(
        args,
        plan=plan,
        plan_path=plan_path,
        plan_sha=plan_sha,
        runtime=runtime,
    )
    output = Path(args.output_dir)
    _require(
        output.is_absolute()
        and output != Path("/")
        and not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.parent.is_symlink(),
        "shard output must be a fresh absolute directory",
    )
    output.mkdir(mode=0o700)
    validated: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    receipt_rows: list[dict[str, Any]] = []
    for task in matching:
        candidate_output = output / task["candidate_id"]
        _require(
            not candidate_output.exists() and not candidate_output.is_symlink(),
            f"refusing candidate output reuse: {task['candidate_id']}",
        )
        command = _candidate_command(
            args,
            task=task,
            candidate_output=candidate_output,
            python=python,
            worker=worker,
            rank_exec=rank_exec,
        )
        environment = _candidate_environment(
            expected_visible=expected_visible,
            python=python,
            scratch=scratch,
            cache_token=(
                "formal-"
                + str(task["seed_slot"])
                + "-"
                + str(task["group_id"])
                + "-"
                + str(task["ordinal"])
            ),
        )
        try:
            subprocess.run(command, check=True, env=environment)
        except subprocess.CalledProcessError as error:
            raise Reserve4GenerationError(
                f"generation failed for {task['candidate_id']}; partial shard is non-authoritative"
            ) from error
        receipt_path = candidate_output / "pair-v5-t2v-calibration-receipt.json"
        receipt, _ = _validate_candidate_receipt(task, receipt_path)
        validated.append((task, receipt))
        receipt_rows.append(
            {
                "candidate_id": task["candidate_id"],
                "path": str(receipt_path),
                "file_sha256": file_sha256(receipt_path),
                "receipt_digest": receipt["receipt_digest"],
            }
        )
    gaussian_proofs = _gaussian_cell_proofs(validated)
    _require(
        len(gaussian_proofs) * 10 == len(matching),
        "shard ended without complete ten-branch cells",
    )
    shard_receipt = {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "plan_path": str(plan_path),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "analysis_split": plan["analysis_split"],
        "seed_slot": args.seed_slot,
        "group_id": args.group_id,
        "visible_gpus": matching[0]["visible_gpus"],
        "candidate_count": len(matching),
        "candidate_receipts": receipt_rows,
        "same_cell_gaussian_proofs": gaussian_proofs,
        "independent_full81_review_performed": False,
        "phi_v1_extraction_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "generated_media_is_editor_input_or_target": False,
    }
    shard_receipt = {
        **shard_receipt, "receipt_digest": object_sha256(shard_receipt)
    }
    _write_create_only(output / "reserve4-generation-shard-receipt-v1.json", shard_receipt)
    return 0


def audit_plan(
    *, plan_path: str | Path, expected_plan_sha256: str,
    generation_roots: Sequence[str | Path], output: str | Path,
    gap_output: str | Path,
) -> dict[str, Any]:
    plan, resolved_plan, plan_sha = load_plan(plan_path, expected_plan_sha256)
    receipts: dict[str, Path] = {}
    for root_value in generation_roots:
        root = _plain_dir(root_value, "generation root")
        for path in root.rglob("pair-v5-t2v-calibration-receipt.json"):
            candidate_id = path.parent.name
            _require(candidate_id not in receipts, f"duplicate generation receipt: {candidate_id}")
            receipts[candidate_id] = path.resolve(strict=True)
    expected_ids = [row["candidate_id"] for row in plan["tasks"]]
    unexpected = sorted(set(receipts) - set(expected_ids))
    _require(not unexpected, f"generation roots contain candidates outside plan: {unexpected}")
    validated: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    receipt_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in plan["tasks"]:
        path = receipts.get(task["candidate_id"])
        if path is None:
            missing.append(task["candidate_id"])
            continue
        receipt, _ = _validate_candidate_receipt(task, path)
        validated.append((task, receipt))
        receipt_rows.append(
            {
                "candidate_id": task["candidate_id"],
                "path": str(path),
                "file_sha256": file_sha256(path),
                "receipt_digest": receipt["receipt_digest"],
            }
        )
    complete_cells = len(validated) // 10 if len(validated) % 10 == 0 else 0
    gap = {
        "schema_version": GAP_SCHEMA,
        "plan_path": str(resolved_plan),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "analysis_split": plan["analysis_split"],
        "expected_candidate_count": len(expected_ids),
        "observed_candidate_count": len(validated),
        "missing_candidate_ids": missing,
        "complete_ten_branch_seed_cells": complete_cells,
        "independent_full81_review_count": 0,
        "phi_v1_extraction_authorized": False,
        "optimizer_authorized": False,
        "generated_media_is_editor_input_or_target": False,
    }
    gap = {**gap, "receipt_digest": object_sha256(gap)}
    _write_create_only(Path(gap_output), gap)
    _require(not missing and len(validated) == 40, "reserve4 generation closure is incomplete; gap receipt written")
    gaussian_proofs = _gaussian_cell_proofs(validated)
    _require(len(gaussian_proofs) == 4, "reserve4 generation cell closure differs")
    audit = {
        "schema_version": AUDIT_RECEIPT_SCHEMA,
        "plan_path": str(resolved_plan),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "analysis_split": plan["analysis_split"],
        "candidate_count": 40,
        "seed_cell_count": 4,
        "candidate_receipts": receipt_rows,
        "same_cell_gaussian_proofs": gaussian_proofs,
        "generation_complete": True,
        "independent_full81_review_performed": False,
        "visual_review_required_before_phi_v1_extraction": True,
        "phi_v1_extraction_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "generated_media_is_editor_input_or_target": False,
    }
    audit = {**audit, "receipt_digest": object_sha256(audit)}
    _write_create_only(Path(output), audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("build-plan")
    plan.add_argument("--seed1-spec", required=True)
    plan.add_argument("--seed2-spec", required=True)
    plan.add_argument("--split", choices=("confirmation",), required=True)
    plan.add_argument("--output-dir", required=True)

    def add_runtime_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--plan", required=True)
        command.add_argument("--expected-plan-sha256", required=True)
        command.add_argument("--python", required=True)
        command.add_argument("--bernini-root", required=True)
        command.add_argument("--veomni-root", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--checkpoint-content-manifest", required=True)
        command.add_argument("--method-source-revision", required=True)
        command.add_argument("--method-source-archive-sha256", required=True)
        command.add_argument("--master-port", type=int, required=True)

    smoke = commands.add_parser("smoke-sp4")
    add_runtime_args(smoke)
    smoke.add_argument("--receipt-output", required=True)
    run = commands.add_parser("run-sp4")
    add_runtime_args(run)
    run.add_argument("--seed-slot", choices=("seed1", "seed2"), required=True)
    run.add_argument("--group-id", choices=("sp4-a", "sp4-b"), required=True)
    run.add_argument("--compile-smoke-receipt", required=True)
    run.add_argument("--expected-compile-smoke-receipt-sha256", required=True)
    run.add_argument("--output-dir", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--plan", required=True)
    audit.add_argument("--expected-plan-sha256", required=True)
    audit.add_argument("--generation-root", action="append", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--gap-output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-plan":
        value = build_plan(
            seed1_spec=args.seed1_spec,
            seed2_spec=args.seed2_spec,
            split=args.split,
            output_dir=args.output_dir,
        )
        print(
            canonical_json_bytes(
                {
                    "plan_path": value["_path"],
                    "plan_file_sha256": value["_file_sha256"],
                    "candidate_count": value["generation_invocation_count"],
                    "shards": value["shards"],
                }
            ).decode("ascii"),
            flush=True,
        )
        return 0
    if args.command == "run-sp4":
        return run_sp4(args)
    if args.command == "smoke-sp4":
        return run_compile_smoke_sp4(args)
    value = audit_plan(
        plan_path=args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        generation_roots=args.generation_root,
        output=args.output,
        gap_output=args.gap_output,
    )
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
