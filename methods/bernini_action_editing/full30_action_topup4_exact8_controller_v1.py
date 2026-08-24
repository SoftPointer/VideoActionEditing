#!/usr/bin/env python3
"""Fail-closed controller for BOX-EXP-010 cross-anchor topup4 exact8.

This controller has no optimizer or training entrypoint.  It binds the exact8
generation audit, independent full-81 review of every action/incomplete clip,
the official same-state materializer threshold gate, and the terminal resource
gate on retained parent 136140/gpu215.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import sys
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
for _root in (METHOD_ROOT, TOOLS_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import full30_action_topup4_exact8_plan_v1 as plan_contract  # noqa: E402
import full30_action_topup4_exact8_generator_v1 as generator  # noqa: E402
import build_full30_action_topup4_exact8_release_v1 as release  # noqa: E402


CONTROLLER_PLAN_SCHEMA = "bernini-full30-action-topup4-exact8-controller-plan-v1"
REVIEW_SCHEMA = "bernini-full30-action-topup4-exact8-review-admission-v1"
MATERIALIZER_GATE_SCHEMA = (
    "bernini-full30-action-topup4-exact8-materializer-gate-v1"
)
TERMINAL_HOST_GATE_SCHEMA = (
    "bernini-full30-action-topup4-exact8-terminal-host-memory-gate-v1"
)
COMPLETION_SCHEMA = "bernini-full30-action-topup4-exact8-completion-v1"
HOLDER_JOB = "136140"
HOLDER_NODE = "auh7-1b-gpu-215"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FULL81_INDEX_SHA256 = hashlib.sha256(
    json.dumps(list(range(81)), separators=(",", ":")).encode("ascii")
).hexdigest()


class Topup4Exact8ControllerError(RuntimeError):
    """Raised before unbound or over-authorized exact8 state can pass."""


def fail(message: str) -> NoReturn:
    raise Topup4Exact8ControllerError(message)


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
        raise Topup4Exact8ControllerError(
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
        raise Topup4Exact8ControllerError(f"{label} is unavailable") from error
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
                Topup4Exact8ControllerError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Topup4Exact8ControllerError(f"{label} is not valid JSON") from error
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
        "controller output must be a fresh absolute path",
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
    require(file_sha256(path) == observed, "controller output replay differs")
    return observed


def validate_release_tree(
    *, method_root: str | Path, manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    root = Path(method_root)
    require(root.is_absolute(), "release method root must be absolute")
    try:
        root_metadata = root.lstat()
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise Topup4Exact8ControllerError("release method root is unavailable") from error
    require(
        resolved_root == root
        and stat.S_ISDIR(root_metadata.st_mode)
        and not stat.S_ISLNK(root_metadata.st_mode),
        "release method root must be one canonical directory",
    )
    manifest, resolved_manifest, observed_manifest = load_json(
        manifest_path, "release manifest", expected_manifest_sha256
    )
    try:
        release.validate_manifest(manifest)
    except release.Topup4Exact8ReleaseError as error:
        raise Topup4Exact8ControllerError(str(error)) from error
    require(
        observed_manifest == expected_manifest_sha256
        and resolved_manifest.parent != root,
        "release manifest must be separately pinned outside its member root",
    )
    expected_paths = {row["path"] for row in manifest["files"]}
    observed_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            continue
        require(path.is_file() and not path.is_symlink(), "release tree contains non-plain member")
        observed_paths.add(path.relative_to(root).as_posix())
    require(observed_paths == expected_paths, "release tree exact member closure differs")
    for row in manifest["files"]:
        member = plain_file(root / row["path"], f"release member {row['path']}")
        metadata = member.stat()
        require(
            metadata.st_size == row["size"]
            and stat.S_IMODE(metadata.st_mode) == row["mode"]
            and file_sha256(member) == row["sha256"],
            f"release member identity differs: {row['path']}",
        )
    source = (root / release.RESOURCE_SOURCE).read_bytes()
    specialized = (root / release.RESOURCE_SPECIALIZED).read_bytes()
    require(
        release.specialize_resource_bytes(source) == specialized,
        "release-bound 136140 resource specialization does not replay",
    )
    return manifest


def _controller_plan_value(
    *, method_root: Path, manifest_path: Path, manifest_sha256: str,
    manifest: Mapping[str, Any], exact8_plan_path: Path,
    exact8_plan_sha256: str, exact8_plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        "schema_version": CONTROLLER_PLAN_SCHEMA,
        "experiment_id": "BOX-EXP-010",
        "purpose": "author four missing cross-anchor comparator cells",
        "learning_target": "N/A; representation evidence only",
        "numeric_target": "N/A; q and a_min remain downstream materializer outputs",
        "dataset": "minimal_cross_anchor_topup4_exact8",
        "training_steps": 0,
        "baseline": "frozen Bernini T2V cross-anchor generation",
        "core_validation": (
            "all four action/incomplete same-Gaussian pairs pass all 81 frames; "
            "official same-state materializer controls pass thresholds"
        ),
        "release": {
            "method_root": str(method_root),
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": manifest_sha256,
            "manifest_digest": manifest["manifest_digest"],
            "resource_specialized_member": release.RESOURCE_SPECIALIZED,
            "resource_specialized_sha256": release.RESOURCE_SPECIALIZED_SHA256,
            "selection_member": release.SELECTION_MEMBER,
            "selection_sha256": release.SELECTION_SHA256,
        },
        "exact8_plan": {
            "path": str(exact8_plan_path),
            "file_sha256": exact8_plan_sha256,
            "plan_digest": exact8_plan["plan_digest"],
            "formal_candidate_count": 8,
            "comparator_cell_count": 4,
            "branch_order": list(plan_contract.ADMISSION_BRANCH_ORDER),
            "diagnostic_task_count": 0,
        },
        "holder": {"job_id": 136140, "node": HOLDER_NODE},
        "formal_shard_order": [row["shard_id"] for row in exact8_plan["shards"]],
        "authority": {
            "independent_full81_review_required_for_each_formal_clip": True,
            "same_gaussian_pair_proof_required_per_cell": True,
            "materializer_same_state_control_gate_required": True,
            "diagnostic_generation_allowed": False,
            "diagnostic_can_enter_q_a_min_or_optimizer": False,
            "optimizer_created": False,
            "optimizer_authorized": False,
            "training_authorized": False,
        },
    }
    return {**unsigned, "plan_digest": object_sha256(unsigned)}


def build_controller_plan(args: argparse.Namespace) -> Mapping[str, Any]:
    root = Path(args.method_root).resolve(strict=True)
    manifest_path = plain_file(args.release_manifest, "release manifest")
    manifest = validate_release_tree(
        method_root=root,
        manifest_path=manifest_path,
        expected_manifest_sha256=args.expected_release_manifest_sha256,
    )
    exact8_plan, exact8_path, exact8_sha = plan_contract.load_plan(
        args.exact8_plan, args.expected_exact8_plan_sha256
    )
    value = _controller_plan_value(
        method_root=root,
        manifest_path=manifest_path,
        manifest_sha256=args.expected_release_manifest_sha256,
        manifest=manifest,
        exact8_plan_path=exact8_path,
        exact8_plan_sha256=exact8_sha,
        exact8_plan=exact8_plan,
    )
    write_create_only(Path(args.output), value)
    return value


def load_controller_plan(
    path: str | Path, expected_sha256: str
) -> tuple[Mapping[str, Any], Path, str, Mapping[str, Any]]:
    value, resolved, observed = load_json(path, "controller plan", expected_sha256)
    unsigned = dict(value)
    declared = unsigned.pop("plan_digest", None)
    require(
        value.get("schema_version") == CONTROLLER_PLAN_SCHEMA
        and isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and object_sha256(unsigned) == declared,
        "controller plan schema/digest differs",
    )
    release_ref = value.get("release", {})
    exact8_ref = value.get("exact8_plan", {})
    manifest = validate_release_tree(
        method_root=release_ref["method_root"],
        manifest_path=release_ref["manifest_path"],
        expected_manifest_sha256=release_ref["manifest_file_sha256"],
    )
    exact8, exact8_path, exact8_sha = plan_contract.load_plan(
        exact8_ref["path"], exact8_ref["file_sha256"]
    )
    expected = _controller_plan_value(
        method_root=Path(release_ref["method_root"]),
        manifest_path=Path(release_ref["manifest_path"]),
        manifest_sha256=release_ref["manifest_file_sha256"],
        manifest=manifest,
        exact8_plan_path=exact8_path,
        exact8_plan_sha256=exact8_sha,
        exact8_plan=exact8,
    )
    require(value == expected, "controller plan replay differs")
    return value, resolved, observed, exact8


def validate_runtime_environment(
    controller_plan: str | Path, expected_plan_sha256: str
) -> Mapping[str, Any]:
    value, _, observed, _ = load_controller_plan(
        controller_plan, expected_plan_sha256
    )
    hostname = socket.gethostname().split(".", 1)[0]
    require(
        os.environ.get("SLURM_JOB_ID") == HOLDER_JOB
        and str(os.environ.get("SLURM_STEP_ID", "")).isdecimal()
        and hostname == HOLDER_NODE,
        "runtime is not the retained 136140/gpu215 numbered child",
    )
    forbidden = {
        "OPTIMIZER_STATE",
        "TRAINING_STEP",
        "FULL30_ACTION_OPTIMIZER",
        "ALLOW_DIAGNOSTIC_GENERATION",
        "ALLOW_DIAGNOSTIC_AS_Q",
        "ALLOW_DIAGNOSTIC_AS_A_MIN",
    }
    require(
        not any(os.environ.get(name) for name in forbidden),
        "optimizer/diagnostic authority environment is forbidden",
    )
    return {
        "controller_plan_digest": value["plan_digest"],
        "controller_plan_file_sha256": observed,
        "slurm_job_id": HOLDER_JOB,
        "slurm_step_id": os.environ["SLURM_STEP_ID"],
        "hostname": hostname,
        "runtime_authorized": True,
        "optimizer_authorized": False,
    }


def _validate_signed(value: Mapping[str, Any], schema: str, label: str) -> None:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    require(
        value.get("schema_version") == schema
        and isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and object_sha256(unsigned) == declared,
        f"{label} schema/digest differs",
    )


def validate_review_admission(
    value: Mapping[str, Any], exact8_plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    _validate_signed(value, REVIEW_SCHEMA, "review admission")
    expected_tasks = exact8_plan["admission_tasks"]
    expected_ids = [task["candidate_id"] for task in expected_tasks]
    rows = value.get("candidate_reviews")
    require(
        value.get("plan_digest") == exact8_plan["plan_digest"]
        and value.get("review_population") == "minimal_cross_anchor_topup4_exact8"
        and value.get("reviewer_independent_of_generator") is True
        and value.get("reviewer_independent_of_materializer") is True
        and value.get("candidate_count") == 8
        and value.get("diagnostic_candidate_count") == 0
        and isinstance(rows, list)
        and len(rows) == 8
        and [row.get("candidate_id") for row in rows] == expected_ids,
        "review admission population differs",
    )
    for row, task in zip(rows, expected_tasks):
        require(
            row.get("semantic_branch") == task["semantic_branch"]
            and row.get("frame_count") == 81
            and row.get("reviewed_frame_count") == 81
            and row.get("reviewed_frame_indices_sha256") == FULL81_INDEX_SHA256
            and row.get("all_81_frames_reviewed") is True
            and row.get("verdict") == "pass"
            and row.get("action_or_incomplete_pass") is True,
            f"full81 review failed: {task['candidate_id']}",
        )
    for cell in exact8_plan["seed_cells"]:
        cell_rows = [
            row
            for row, task in zip(rows, expected_tasks)
            if task["calibration_group_id"] == cell["calibration_group_id"]
        ]
        require(
            [row["semantic_branch"] for row in cell_rows]
            == list(plan_contract.ADMISSION_BRANCH_ORDER),
            "each comparator cell must pass action and incomplete independently",
        )
    return value


def validate_materializer_gate(
    value: Mapping[str, Any], exact8_plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    _validate_signed(value, MATERIALIZER_GATE_SCHEMA, "materializer gate")
    rows = value.get("seed_cell_gates")
    expected_cells = exact8_plan["seed_cells"]
    require(
        value.get("plan_digest") == exact8_plan["plan_digest"]
        and value.get("materializer") == "full30_action_psiout_materializer_v1"
        and value.get("same_real_source_state") is True
        and value.get("same_sigma") is True
        and value.get("same_noise") is True
        and value.get("official_frozen_noop_stopgrad") is True
        and value.get("all_threshold_gates_passed") is True
        and value.get("diagnostic_task_count") == 0
        and value.get("generated_media_used_by_optimizer") is False
        and value.get("optimizer_input_created") is False
        and isinstance(rows, list)
        and len(rows) == 4,
        "materializer authority or population differs",
    )
    expected_control_gates = {
        name: True for name in plan_contract.MATERIALIZER_CONTROL_ORDER
    }
    for row, cell in zip(rows, expected_cells):
        q_values = row.get("q_values")
        a_min = row.get("a_min")
        require(
            row.get("group_id") == cell["group_id"]
            and row.get("calibration_group_id") == cell["calibration_group_id"]
            and row.get("admitted_candidate_ids") == cell["candidate_ids"]
            and row.get("control_gates") == expected_control_gates
            and row.get("threshold_gate_passed") is True
            and isinstance(q_values, Mapping)
            and set(q_values) == {"action", "incomplete"}
            and all(
                type(item) in {int, float} and math.isfinite(item)
                for item in q_values.values()
            )
            and type(a_min) in {int, float}
            and math.isfinite(a_min)
            and a_min > 0,
            f"materializer comparator-cell gate differs: {cell['calibration_group_id']}",
        )
    return value


def validate_exact8_audit(
    value: Mapping[str, Any], exact8_plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    _validate_signed(value, generator.AUDIT_SCHEMA, "exact8 generation audit")
    expected_ids = [task["candidate_id"] for task in exact8_plan["admission_tasks"]]
    rows = value.get("candidate_receipts")
    require(
        value.get("plan_digest") == exact8_plan["plan_digest"]
        and value.get("candidate_count") == 8
        and value.get("comparator_cell_count") == 4
        and value.get("all_candidates_exact81") is True
        and isinstance(rows, list)
        and [row.get("candidate_id") for row in rows] == expected_ids
        and all(row.get("frame_count") == 81 for row in rows)
        and value.get("diagnostic_task_count") == 0
        and value.get("diagnostic_generation_observed_or_required") is False
        and value.get("optimizer_authorized") is False,
        "exact8 generation audit differs",
    )
    return value


def seal_terminal_host_gate(args: argparse.Namespace) -> Mapping[str, Any]:
    resource = generator.load_resource_contract(args.resource_contract)
    try:
        start, start_path, start_sha = resource.load_host_cgroup_memory_monitor_start(
            args.monitor_start_receipt,
            args.expected_monitor_start_receipt_sha256,
        )
        raw, packed_rows, metadata, _ = resource._journal_prefix(
            start, exact_terminal_size=True
        )
        rows = [resource._sample_row(row) for row in packed_rows]
        monitor_dead = not resource._process_identity_is_live(
            int(start["monitor_pid"]), int(start["monitor_proc_start_ticks"])
        )
    except Exception as error:
        raise Topup4Exact8ControllerError(str(error)) from error
    gaps = [
        int(right["monotonic_time_ns"]) - int(left["monotonic_time_ns"])
        for left, right in zip(rows, rows[1:])
    ]
    require(
        args.monitor_exit_status == 0
        and monitor_dead
        and len(rows) >= 2
        and rows[0] == start["initial_sample"]
        and [row["sequence"] for row in rows] == list(range(len(rows)))
        and rows[-1]["sample_kind"] == "stop_final"
        and all(row["sample_kind"] == "periodic" for row in rows[:-1])
        and all(0 < gap <= 100_000_000 for gap in gaps)
        and all(row["memory_current_bytes"] < 56 * 1024**3 for row in rows)
        and all(row["memory_max_bytes"] == 60 * 1024**3 for row in rows)
        and all(row["memory_events"] == {"oom": 0, "oom_kill": 0} for row in rows),
        "terminal exact8 host-memory/OOM/cadence gate failed",
    )
    unsigned = {
        "schema_version": TERMINAL_HOST_GATE_SCHEMA,
        "measurement_phase": "terminal_after_topup4_exact8",
        "formal_candidate_count_at_gate": 8,
        "monitor_start_receipt": {
            "path": str(start_path),
            "file_sha256": start_sha,
            "receipt_digest": start["receipt_digest"],
        },
        "sample_journal": {
            "path": start["sample_journal"]["path"],
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sample_count": len(rows),
        },
        "sample_interval_ns": 10_000_000,
        "maximum_observed_gap_ns": max(gaps),
        "maximum_allowed_gap_ns": 100_000_000,
        "host_memory_limit_gib": 60,
        "host_memory_safe_ceiling_gib": 56,
        "sampled_peak_memory_current_bytes": max(
            row["memory_current_bytes"] for row in rows
        ),
        "sampled_peak_strictly_below_56_gib": True,
        "all_samples_zero_oom_and_oom_kill": True,
        "monitor_exit_status": 0,
        "monitor_identity_dead_at_gate": True,
        "diagnostic_task_count": 0,
        "optimizer_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    write_create_only(Path(args.output), receipt)
    return receipt


def validate_terminal_host_gate(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _validate_signed(value, TERMINAL_HOST_GATE_SCHEMA, "terminal host gate")
    require(
        value.get("formal_candidate_count_at_gate") == 8
        and value.get("sample_interval_ns") == 10_000_000
        and value.get("maximum_observed_gap_ns") <= 100_000_000
        and value.get("host_memory_limit_gib") == 60
        and value.get("host_memory_safe_ceiling_gib") == 56
        and value.get("sampled_peak_strictly_below_56_gib") is True
        and value.get("all_samples_zero_oom_and_oom_kill") is True
        and value.get("monitor_exit_status") == 0
        and value.get("monitor_identity_dead_at_gate") is True
        and value.get("diagnostic_task_count") == 0
        and value.get("optimizer_authorized") is False,
        "terminal host gate authority differs",
    )
    return value


def seal_completion(args: argparse.Namespace) -> Mapping[str, Any]:
    controller_plan, plan_path, plan_sha, exact8_plan = load_controller_plan(
        args.controller_plan, args.expected_controller_plan_sha256
    )
    generation, generation_path, generation_sha = load_json(
        args.generation_audit,
        "exact8 generation audit",
        args.expected_generation_audit_sha256,
    )
    review, review_path, review_sha = load_json(
        args.review_admission,
        "independent full81 review admission",
        args.expected_review_admission_sha256,
    )
    materializer, materializer_path, materializer_sha = load_json(
        args.materializer_gate,
        "same-state materializer gate",
        args.expected_materializer_gate_sha256,
    )
    terminal, terminal_path, terminal_sha = load_json(
        args.terminal_host_gate,
        "terminal exact8 host gate",
        args.expected_terminal_host_gate_sha256,
    )
    validate_exact8_audit(generation, exact8_plan)
    validate_review_admission(review, exact8_plan)
    validate_materializer_gate(materializer, exact8_plan)
    validate_terminal_host_gate(terminal)
    unsigned = {
        "schema_version": COMPLETION_SCHEMA,
        "controller_plan": {
            "path": str(plan_path),
            "file_sha256": plan_sha,
            "plan_digest": controller_plan["plan_digest"],
        },
        "generation_audit": {
            "path": str(generation_path),
            "file_sha256": generation_sha,
            "receipt_digest": generation["receipt_digest"],
        },
        "independent_full81_review_admission": {
            "path": str(review_path),
            "file_sha256": review_sha,
            "receipt_digest": review["receipt_digest"],
            "action_and_incomplete_each_full81_pass": True,
        },
        "official_same_state_materializer_gate": {
            "path": str(materializer_path),
            "file_sha256": materializer_sha,
            "receipt_digest": materializer["receipt_digest"],
            "all_threshold_gates_passed": True,
        },
        "terminal_host_gate": {
            "path": str(terminal_path),
            "file_sha256": terminal_sha,
            "receipt_digest": terminal["receipt_digest"],
        },
        "dataset": "minimal_cross_anchor_topup4_exact8",
        "formal_candidate_count": 8,
        "comparator_cell_count": 4,
        "diagnostic_task_count": 0,
        "q_and_a_min_source": "official_same_state_materializer_only",
        "preoptimizer_cross_anchor_dataflow_gate_passed": True,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "training_performed": False,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    write_create_only(Path(args.output), receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--method-root", required=True)
    plan.add_argument("--release-manifest", required=True)
    plan.add_argument("--expected-release-manifest-sha256", required=True)
    plan.add_argument("--exact8-plan", required=True)
    plan.add_argument("--expected-exact8-plan-sha256", required=True)
    plan.add_argument("--output", required=True)
    validate = commands.add_parser("validate-plan")
    validate.add_argument("--controller-plan", required=True)
    validate.add_argument("--expected-controller-plan-sha256", required=True)
    runtime = commands.add_parser("validate-runtime")
    runtime.add_argument("--controller-plan", required=True)
    runtime.add_argument("--expected-controller-plan-sha256", required=True)
    terminal = commands.add_parser("seal-terminal-host-gate")
    terminal.add_argument("--resource-contract", required=True)
    terminal.add_argument("--monitor-start-receipt", required=True)
    terminal.add_argument("--expected-monitor-start-receipt-sha256", required=True)
    terminal.add_argument("--monitor-exit-status", type=int, required=True)
    terminal.add_argument("--output", required=True)
    complete = commands.add_parser("seal-completion")
    complete.add_argument("--controller-plan", required=True)
    complete.add_argument("--expected-controller-plan-sha256", required=True)
    complete.add_argument("--generation-audit", required=True)
    complete.add_argument("--expected-generation-audit-sha256", required=True)
    complete.add_argument("--review-admission", required=True)
    complete.add_argument("--expected-review-admission-sha256", required=True)
    complete.add_argument("--materializer-gate", required=True)
    complete.add_argument("--expected-materializer-gate-sha256", required=True)
    complete.add_argument("--terminal-host-gate", required=True)
    complete.add_argument("--expected-terminal-host-gate-sha256", required=True)
    complete.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        value = build_controller_plan(args)
    elif args.command == "validate-plan":
        value, _, _, _ = load_controller_plan(
            args.controller_plan, args.expected_controller_plan_sha256
        )
    elif args.command == "validate-runtime":
        value = validate_runtime_environment(
            args.controller_plan, args.expected_controller_plan_sha256
        )
    elif args.command == "seal-terminal-host-gate":
        value = seal_terminal_host_gate(args)
    else:
        value = seal_completion(args)
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
