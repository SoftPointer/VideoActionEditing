#!/usr/bin/env python3
"""Generate the no-diagnostic BOX-EXP-009 exact8 on 136141/gpu299.

The only executable population is four confirmation seed cells times the
ordered ``action``/``incomplete`` pair.  There is no diagnostic lane or CLI.

The renderer, WORLD4 resource lifecycle, 60/56-GiB host monitor, 10-ms
sampling, T5 rank-GPU residency, physical ``safe_open`` smoke evidence, and
r10-r13 tensor/MP4 parity are inherited from the frozen fit-r13 resource
contract.  This module accepts only its release-built, dedicated 136141
postimage; the fit-r13 preimage is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full30_action_confirmation8_136141_plan_v1 as plan_contract  # noqa: E402


RESOURCE_PREIMAGE_SHA256 = (
    "be722e4020040ba446f290f07378e870e2d3c1a4228ec997c3447770fcb53d5d"
)
RESOURCE_SPECIALIZED_SHA256 = (
    "be722e4020040ba446f290f07378e870e2d3c1a4228ec997c3447770fcb53d5d"
)
RESOURCE_SPECIALIZED_BASENAME = (
    "reserve4_fixed_generation_sp4_136141_confirmation8_specialized_v1.py"
)
RESOURCE_SPECIALIZED_SIZE = 166_064
HOLDER_JOB = "136141"
HOLDER_NODE = "auh7-1b-gpu-299"
SHARD_SCHEMA = "bernini-full30-action-confirmation8-136141-shard-v1"
AUDIT_SCHEMA = "bernini-full30-action-confirmation8-136141-audit-v1"
GAP_SCHEMA = "bernini-full30-action-confirmation8-136141-gap-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
IDENTITY_FIELDS = (
    "raw_value_sha256",
    "content_sha256",
    "shape",
    "dtype",
    "stored_dtype",
    "generator_initial_seed",
)


class Confirmation8GenerationError(RuntimeError):
    """Raised before a partial or over-authorized generation can pass."""


def fail(message: str) -> NoReturn:
    raise Confirmation8GenerationError(message)


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
        raise Confirmation8GenerationError(
            "value is not canonical finite JSON"
        ) from error


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
        raise Confirmation8GenerationError(f"{label} is unavailable") from error
    require(
        resolved == path
        and stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain file",
    )
    return resolved


def plain_dir(value: str | Path, label: str) -> Path:
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise Confirmation8GenerationError(f"{label} is unavailable") from error
    require(
        resolved == path
        and stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain directory",
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
                Confirmation8GenerationError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Confirmation8GenerationError(f"{label} is not valid JSON") from error
    require(type(result) is dict, f"{label} must be a JSON object")
    return result, path, observed


def write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    require(
        path.is_absolute()
        and path.parent.is_dir()
        and not path.parent.is_symlink()
        and not path.exists()
        and not path.is_symlink(),
        "receipt output must be a fresh absolute path",
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
    require(file_sha256(path) == observed, "receipt write replay differs")
    return observed


def load_resource_contract(value: str | Path) -> ModuleType:
    """Load only the release-built dedicated 136141 fit-r13 postimage."""

    path = plain_file(value, "136141 confirmation resource contract")
    raw = path.read_bytes()
    require(
        path.name == RESOURCE_SPECIALIZED_BASENAME
        and len(raw) == RESOURCE_SPECIALIZED_SIZE
        and hashlib.sha256(raw).hexdigest() == RESOURCE_SPECIALIZED_SHA256
        and hashlib.sha256(raw).hexdigest() == RESOURCE_PREIMAGE_SHA256
        and raw.count(b"136141") == 7
        and raw.count(b"136309") == 0
        and raw.count(b"auh7-1b-gpu-299") == 0,
        "136141 resource specialization identity differs",
    )
    module_name = "_bernini_full30_confirmation8_136141_resource_be722e40"
    require(module_name not in sys.modules, "resource specialization is preloaded")
    specification = importlib.util.spec_from_file_location(module_name, path)
    require(
        specification is not None and specification.loader is not None,
        "cannot load resource specialization",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    require(
        module.HOST_MEMORY_LIMIT_GIB == 60
        and module.HOST_MEMORY_SAFE_CEILING_GIB == 56
        and module.HOST_MEMORY_SAMPLE_INTERVAL_NS == 10_000_000
        and module.T2V_GPU_MEMORY_LIMIT_GIB == 52,
        "fit-r13 resource limits differ",
    )
    return module


def _formal_tasks(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tasks = plan.get("admission_tasks")
    require(
        isinstance(tasks, list)
        and len(tasks) == 8
        and plan.get("diagnostic_task_count") == 0
        and plan.get("diagnostic_generation_allowed") is False
        and "diagnostic_tasks" not in plan,
        "no-diagnostic formal task closure differs",
    )
    return tasks


def _runtime_binding(
    args: argparse.Namespace, resource: ModuleType
) -> tuple[Mapping[str, Any], Path, Path, Path, Path]:
    try:
        binding = resource._runtime_binding(args)
    except Exception as error:
        raise Confirmation8GenerationError(str(error)) from error
    require(
        isinstance(binding, tuple) and len(binding) == 5,
        "fit-r13 runtime binding differs",
    )
    return binding


def validate_resource_smoke(
    args: argparse.Namespace,
    resource: ModuleType,
    runtime: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        receipt, source, observed = resource.load_compile_smoke_receipt(
            args.resource_compile_smoke_receipt,
            args.expected_resource_compile_smoke_receipt_sha256,
        )
    except Exception as error:
        raise Confirmation8GenerationError(str(error)) from error
    require(
        source.read_bytes() == resource.canonical_json_bytes(receipt) + b"\n"
        and observed == args.expected_resource_compile_smoke_receipt_sha256
        and receipt.get("runtime") == runtime
        and receipt.get("formal_candidate_count_at_gate") == 0
        and receipt.get("physical_safetensors_safe_open_recomputation_required")
        is True
        and receipt.get("mp4_whole_file_sha256_parity_required") is True
        and receipt.get("gaussian_and_clean_tensor_identity_parity_required")
        is True
        and receipt.get("compile_smoke_passed") is True,
        "fit-r13 physical/r10 compile-smoke binding differs",
    )
    return receipt


def _validate_candidate_receipt(
    resource: ModuleType, task: Mapping[str, Any], receipt_path: Path
) -> Mapping[str, Any]:
    try:
        receipt, _ = resource._validate_candidate_receipt(task, receipt_path)
    except Exception as error:
        raise Confirmation8GenerationError(str(error)) from error
    mp4 = receipt.get("artifacts", {}).get("mp4", {})
    require(
        mp4.get("frame_count") == 81
        and receipt.get("candidate", {}).get("semantic_branch")
        == task["semantic_branch"],
        f"candidate is not an exact81 {task['semantic_branch']} clip",
    )
    return receipt


def _gaussian_pair_proof(
    tasks: Sequence[Mapping[str, Any]], receipts: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    require(
        [task["semantic_branch"] for task in tasks]
        == list(plan_contract.ADMISSION_BRANCH_ORDER)
        and len(receipts) == 2,
        "formal seed cell is not one ordered action/incomplete pair",
    )
    gaussians = [
        receipt["artifacts"]["official_initial_gaussian"] for receipt in receipts
    ]
    identities = [
        {field: artifact.get(field) for field in IDENTITY_FIELDS}
        for artifact in gaussians
    ]
    require(
        identities[0] == identities[1],
        "action/incomplete did not reuse one official Gaussian tensor",
    )
    return {
        "calibration_group_id": tasks[0]["calibration_group_id"],
        "seed": tasks[0]["seed"],
        "branch_order": list(plan_contract.ADMISSION_BRANCH_ORDER),
        "official_gaussian_identity": identities[0],
        "action_incomplete_official_gaussian_tensor_values_byte_equal": True,
    }


def run_shard(args: argparse.Namespace) -> int:
    plan, plan_path, plan_sha = plan_contract.load_plan(
        args.plan, args.expected_plan_sha256
    )
    lane_tasks = _formal_tasks(plan)
    matching = [
        task
        for task in lane_tasks
        if task["seed_slot"] == args.seed_slot
        and task["group_id"] == args.group_id
    ]
    expected_count = 2
    expected_branches = plan_contract.ADMISSION_BRANCH_ORDER
    require(
        len(matching) == expected_count
        and [task["semantic_branch"] for task in matching]
        == list(expected_branches),
        "sealed formal shard scope differs",
    )
    visible = ",".join(str(item) for item in matching[0]["visible_gpus"])
    require(
        all(task["visible_gpus"] == matching[0]["visible_gpus"] for task in matching)
        and os.environ.get("ROCR_VISIBLE_DEVICES") == visible,
        "sealed shard GPU mapping differs",
    )
    resource = load_resource_contract(args.resource_contract)
    runtime, python, worker, rank_exec, scratch = _runtime_binding(args, resource)
    smoke = validate_resource_smoke(args, resource, runtime)
    output = Path(args.output_dir)
    require(
        output.is_absolute()
        and output != Path("/")
        and not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.parent.is_symlink(),
        "shard output must be a fresh absolute directory",
    )
    output.mkdir(mode=0o700)
    receipt_rows: list[dict[str, Any]] = []
    candidate_receipts: list[Mapping[str, Any]] = []
    for task in matching:
        candidate_output = output / task["candidate_id"]
        try:
            resource.assert_live_host_cgroup_memory_monitor()
            command = resource._candidate_command(
                args,
                task=task,
                candidate_output=candidate_output,
                python=python,
                worker=worker,
                rank_exec=rank_exec,
            )
            environment = resource._candidate_environment(
                expected_visible=visible,
                python=python,
                scratch=scratch,
                cache_token=(
                    f"confirmation8-136141-{args.seed_slot}-"
                    f"{args.group_id}-{task['semantic_branch']}"
                ),
            )
            subprocess.run(command, check=True, env=environment)
            resource.assert_live_host_cgroup_memory_monitor()
        except subprocess.CalledProcessError as error:
            raise Confirmation8GenerationError(
                f"generation failed for {task['candidate_id']}"
            ) from error
        except Exception as error:
            if isinstance(error, Confirmation8GenerationError):
                raise
            raise Confirmation8GenerationError(str(error)) from error
        receipt_path = (
            candidate_output / "pair-v5-t2v-calibration-receipt.json"
        )
        receipt = _validate_candidate_receipt(resource, task, receipt_path)
        candidate_receipts.append(receipt)
        receipt_rows.append(
            {
                "candidate_id": task["candidate_id"],
                "semantic_branch": task["semantic_branch"],
                "path": str(receipt_path),
                "file_sha256": file_sha256(receipt_path),
                "receipt_digest": receipt["receipt_digest"],
                "full81_pass_pending_independent_review": True,
            }
        )
    unsigned: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA,
        "plan": {
            "path": str(plan_path),
            "file_sha256": plan_sha,
            "plan_digest": plan["plan_digest"],
        },
        "lane": "formal-action-incomplete-only",
        "seed_slot": args.seed_slot,
        "group_id": args.group_id,
        "visible_gpus": matching[0]["visible_gpus"],
        "candidate_count": len(matching),
        "branch_order": list(expected_branches),
        "candidate_receipts": receipt_rows,
        "resource_compile_smoke": {
            "path": str(plain_file(
                args.resource_compile_smoke_receipt,
                "resource compile-smoke receipt",
            )),
            "file_sha256": args.expected_resource_compile_smoke_receipt_sha256,
            "receipt_digest": smoke["receipt_digest"],
            "formal_candidate_count_at_gate": 0,
            "physical_safe_open_and_r10_parity_passed": True,
        },
        "independent_full81_review_performed": False,
        "q_input_authorized": False,
        "a_min_input_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
    }
    unsigned["action_incomplete_gaussian_pair_proof"] = _gaussian_pair_proof(
        matching, candidate_receipts
    )
    unsigned["counts_toward_confirmation_exact8"] = True
    unsigned["diagnostic_task_count"] = 0
    unsigned["diagnostic_generation_allowed"] = False
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    write_create_only(
        output / "confirmation8-136141-formal-shard-receipt-v1.json", receipt
    )
    return 0


def audit_exact8(
    *,
    plan_path: str | Path,
    expected_plan_sha256: str,
    generation_roots: Sequence[str | Path],
    output: str | Path,
    gap_output: str | Path,
) -> Mapping[str, Any]:
    plan, resolved, plan_sha = plan_contract.load_plan(
        plan_path, expected_plan_sha256
    )
    expected_tasks = list(_formal_tasks(plan))
    expected_ids = [task["candidate_id"] for task in expected_tasks]
    observed_paths: dict[str, Path] = {}
    for root_value in generation_roots:
        root = plain_dir(root_value, "formal exact8 generation root")
        for path in root.rglob("pair-v5-t2v-calibration-receipt.json"):
            candidate_id = path.parent.name
            require(
                candidate_id not in observed_paths,
                f"duplicate formal candidate receipt: {candidate_id}",
            )
            observed_paths[candidate_id] = path.resolve(strict=True)
    unexpected = sorted(set(observed_paths) - set(expected_ids))
    missing = [candidate_id for candidate_id in expected_ids if candidate_id not in observed_paths]
    gap_unsigned = {
        "schema_version": GAP_SCHEMA,
        "plan_path": str(resolved),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "expected_candidate_count": 8,
        "observed_candidate_count": len(observed_paths),
        "missing_candidate_ids": missing,
        "unexpected_candidate_ids": unexpected,
        "diagnostic_task_count": 0,
        "diagnostic_generation_allowed": False,
        "optimizer_authorized": False,
    }
    gap = {**gap_unsigned, "receipt_digest": object_sha256(gap_unsigned)}
    write_create_only(Path(gap_output), gap)
    require(
        not missing
        and not unexpected,
        "formal exact8 generation closure differs; gap receipt was written",
    )
    resource = load_resource_contract(
        METHOD_ROOT / "tools" / RESOURCE_SPECIALIZED_BASENAME
    )
    receipt_rows: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for task in expected_tasks:
        receipt_path = observed_paths[task["candidate_id"]]
        receipt = _validate_candidate_receipt(resource, task, receipt_path)
        by_cell.setdefault((task["seed_slot"], task["group_id"]), []).append(
            (task, receipt)
        )
        receipt_rows.append(
            {
                "candidate_id": task["candidate_id"],
                "semantic_branch": task["semantic_branch"],
                "path": str(receipt_path),
                "file_sha256": file_sha256(receipt_path),
                "receipt_digest": receipt["receipt_digest"],
                "frame_count": 81,
            }
        )
    gaussian_proofs = [
        _gaussian_pair_proof(
            [task for task, _ in rows], [receipt for _, receipt in rows]
        )
        for rows in by_cell.values()
    ]
    require(len(gaussian_proofs) == 4, "formal exact8 seed-cell closure differs")
    unsigned = {
        "schema_version": AUDIT_SCHEMA,
        "plan_path": str(resolved),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "dataset": "confirmation_action_anchor_exact8_136141",
        "candidate_count": 8,
        "seed_cell_count": 4,
        "branch_order_per_cell": list(plan_contract.ADMISSION_BRANCH_ORDER),
        "candidate_receipts": receipt_rows,
        "action_incomplete_gaussian_pair_proofs": gaussian_proofs,
        "all_candidates_exact81": True,
        "independent_full81_review_performed": False,
        "review_admission_authorized": False,
        "materializer_same_state_threshold_gate_present": False,
        "q_input_authorized": False,
        "a_min_input_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "diagnostic_task_count": 0,
        "diagnostic_generation_observed_or_allowed": False,
        "inference_steps_per_clip": 40,
        "optimizer_steps": 0,
    }
    audit = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    write_create_only(Path(output), audit)
    return audit


def add_runtime_args(command: argparse.ArgumentParser) -> None:
    command.add_argument("--plan", required=True)
    command.add_argument("--expected-plan-sha256", required=True)
    command.add_argument("--resource-contract", required=True)
    command.add_argument("--resource-compile-smoke-receipt", required=True)
    command.add_argument(
        "--expected-resource-compile-smoke-receipt-sha256", required=True
    )
    command.add_argument("--python", required=True)
    command.add_argument("--bernini-root", required=True)
    command.add_argument("--veomni-root", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--checkpoint-content-manifest", required=True)
    command.add_argument("--method-source-revision", required=True)
    command.add_argument("--method-source-archive-sha256", required=True)
    command.add_argument("--master-port", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-sp4")
    add_runtime_args(run)
    run.add_argument("--seed-slot", choices=("seed1", "seed2"), required=True)
    run.add_argument("--group-id", choices=("sp4-a", "sp4-b"), required=True)
    run.add_argument("--output-dir", required=True)
    audit = commands.add_parser("audit-exact8")
    audit.add_argument("--plan", required=True)
    audit.add_argument("--expected-plan-sha256", required=True)
    audit.add_argument("--generation-root", action="append", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--gap-output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-sp4":
        return run_shard(args)
    value = audit_exact8(
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
