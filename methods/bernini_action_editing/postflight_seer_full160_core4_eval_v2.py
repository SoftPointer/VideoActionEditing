#!/usr/bin/env python3
"""Afterok aggregation for the two-task SEER full160 core4 decode array."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


STAGE_ROOT = Path(__file__).resolve().parent
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))
if str(STAGE_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT / "tools"))

import bind_seer_full160_eval_source_v2 as binder  # noqa: E402


SCHEMA_VERSION = "bernini-seer-full160-core4-eval-master-binding-v2"
CORE4_SCHEMA_VERSION = "bernini-self-generated-action-lora-heldout-core4-receipt-v1"
CORE4_IIDS = (
    "99cde432839f4240",
    "6ea45d35943742bb",
    "311c82f83eca4a7f",
    "6d346c38cf504493",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NODE = re.compile(r"auh[0-9A-Za-z-]+\Z")
JOINT_SAFE_DEFINITION = (
    "requested_action_reaches_terminal_and_holds AND "
    "identity_scene_camera_inventory_preserved"
)
CORE4_DECISION_CONTRACT = {
    "human_full_video_review_required": True,
    "joint_safe_definition": JOINT_SAFE_DEFINITION,
    "latent_only_or_pipeline_only_result_is_success": False,
    "maximum_joint_safe_to_unsafe_regressions": 0,
    "minimum_base_fail_to_trained_joint_safe_flips": 2,
    "minimum_flip_actor_family_coverage": ["dog", "human"],
    "minimum_trained_joint_safe": 3,
    "training_completion_is_success": False,
    "unit": "paired_heldout_source_seed",
}
HELDOUT_RUNNER_RELATIVE = "run_self_generated_action_lora_heldout_core4_v1.py"


class PostflightError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return binder.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return binder.file_sha256(path)


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise PostflightError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PostflightError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise PostflightError(f"{label} must be a plain file")
    return path.resolve(strict=True)


def _executable_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value)
    if not requested.is_absolute():
        raise PostflightError(f"{label} must be absolute")
    try:
        resolved = requested.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise PostflightError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise PostflightError(f"{label} must resolve to an executable regular file")
    return resolved


def _directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise PostflightError(f"{label} must be absolute and non-root")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PostflightError(f"{label} is unavailable: {error}") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise PostflightError(f"{label} must be a plain directory")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PostflightError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PostflightError(f"{label} root must be an object")
    return value


def _verified_receipt(path: Path, *, label: str) -> dict[str, Any]:
    value = _read_json(_plain_file(path, label=label), label=label)
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha256(unsigned) != declared:
        raise PostflightError(f"{label} digest differs")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PostflightError(f"output must be fresh: {path}")
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise PostflightError("short master-binding write")
            view = view[count:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _task_completion(root: Path, *, array_job_id: int, task: int) -> dict[str, Any]:
    path = _plain_file(root / f"task-{task}.COMPLETE", label=f"task{task} completion")
    text = path.read_text(encoding="utf-8")
    match = re.fullmatch(
        rf"COMPLETE method=full160 array_job={array_job_id} array_task={task} node=(auh[0-9A-Za-z-]+)\n",
        text,
    )
    if match is None or _NODE.fullmatch(match.group(1)) is None:
        raise PostflightError(f"task{task} completion binding differs")
    return {"task": task, "node": match.group(1), "path": str(path), "sha256": file_sha256(path)}


def aggregate(
    *,
    output_root: Path,
    array_job_id: int,
    python_bin: Path,
    expected_spec_sha256: str,
    recovery_receipt: Path,
    expected_recovery_receipt_sha256: str,
    expected_recovery_receipt_digest: str,
) -> dict[str, Any]:
    try:
        external_recovery = binder._verify_recovery_binding(
            recovery_receipt,
            expected_sha256=expected_recovery_receipt_sha256,
            expected_digest=expected_recovery_receipt_digest,
        )
    except binder.SourceBindingError as error:
        raise PostflightError(str(error)) from error
    marker = _plain_file(output_root / ".array-job-id", label="array job marker")
    if marker.read_text(encoding="ascii") != f"{array_job_id}\n":
        raise PostflightError("array job marker differs")
    tasks = [
        _task_completion(output_root, array_job_id=array_job_id, task=task)
        for task in (0, 1)
    ]
    results_root = _directory(output_root / "results", label="shared results root")
    first_runtime = _directory(
        output_root / CORE4_IIDS[0] / "runtime-source",
        label="first case runtime source",
    )
    method_root = _directory(
        first_runtime / "methods" / "bernini_action_editing",
        label="runtime method root",
    )
    runner = _plain_file(
        method_root / "run_self_generated_action_lora_heldout_core4_v1.py",
        label="heldout core4 runner",
    )
    spec = _plain_file(
        method_root / "assets" / "self_generated_action_lora_heldout_core4_v1.json",
        label="heldout core4 spec",
    )
    if file_sha256(spec) != expected_spec_sha256:
        raise PostflightError("heldout core4 spec SHA differs")
    core4_path = results_root / "core4-master-receipt.json"
    if core4_path.exists() or core4_path.is_symlink():
        raise PostflightError("core4 master target must be fresh")

    case_rows: list[dict[str, Any]] = []
    overlay_identity: tuple[str, str, str] | None = None
    adapter_identity: tuple[str, str, str] | None = None
    recovery_identity: tuple[str, str, str, str, str, str] | None = None
    heldout_runner_sha256: str | None = None
    for iid in CORE4_IIDS:
        path = _plain_file(
            output_root / iid / "eval-execution-binding.json",
            label=f"{iid} case eval binding",
        )
        try:
            value = binder.verify_case_binding_receipt(path)
        except binder.SourceBindingError as error:
            raise PostflightError(str(error)) from error
        if value.get("iid") != iid:
            raise PostflightError(f"case eval IID differs: {iid}")
        source = value["source_binding"]
        trained = value["trained_adapter"]
        recovered = value["checkpoint_recovery"]
        source_path = _plain_file(
            source["path"], label=f"{iid} source binding receipt"
        )
        try:
            source_receipt = binder.verify_receipt(source_path)
        except binder.SourceBindingError as error:
            raise PostflightError(str(error)) from error
        if (
            source_receipt.get("receipt_digest") != source.get("receipt_digest")
            or source_receipt.get("inference_runtime_overlay", {}).get(
                "archive_sha256"
            )
            != source.get("overlay_archive_sha256")
            or source_receipt.get("inference_runtime_overlay", {}).get(
                "manifest_sha256"
            )
            != source.get("overlay_manifest_sha256")
            or source_receipt.get("inference_runtime_overlay", {}).get(
                "manifest_digest"
            )
            != source.get("overlay_manifest_digest")
        ):
            raise PostflightError(f"case/source overlay cross-bind differs: {iid}")
        overlay_files = source_receipt.get("inference_runtime_overlay", {}).get(
            "files"
        )
        if not isinstance(overlay_files, list):
            raise PostflightError(f"source overlay file closure differs: {iid}")
        runner_rows = [
            row
            for row in overlay_files
            if isinstance(row, Mapping)
            and row.get("path") == HELDOUT_RUNNER_RELATIVE
        ]
        if (
            len(runner_rows) != 1
            or not isinstance(runner_rows[0].get("sha256"), str)
            or _SHA256.fullmatch(runner_rows[0]["sha256"]) is None
        ):
            raise PostflightError(f"source overlay heldout runner row differs: {iid}")
        case_runner = _plain_file(
            output_root
            / iid
            / "runtime-source"
            / "methods"
            / "bernini_action_editing"
            / HELDOUT_RUNNER_RELATIVE,
            label=f"{iid} runtime heldout runner",
        )
        current_runner_sha = file_sha256(case_runner)
        if current_runner_sha != runner_rows[0]["sha256"]:
            raise PostflightError(f"runtime heldout runner/overlay cross-bind differs: {iid}")
        if heldout_runner_sha256 is None:
            heldout_runner_sha256 = current_runner_sha
        elif current_runner_sha != heldout_runner_sha256:
            raise PostflightError("core4 runtime heldout runner identity differs")
        current_overlay = (
            source["overlay_archive_sha256"],
            source["overlay_manifest_sha256"],
            source["overlay_manifest_digest"],
        )
        current_adapter = (
            trained["adapter_model_sha256"],
            trained["training_receipt_sha256"],
            trained["training_receipt_digest"],
        )
        current_recovery = (
            recovered["receipt_path"],
            recovered["sha256"],
            recovered["receipt_digest"],
            recovered["final_checkpoint_path"],
            recovered["final_training_receipt_digest"],
            recovered["final_adapter_model_sha256"],
        )
        if overlay_identity is None:
            overlay_identity = current_overlay
            adapter_identity = current_adapter
            recovery_identity = current_recovery
        elif (
            current_overlay != overlay_identity
            or current_adapter != adapter_identity
            or current_recovery != recovery_identity
        ):
            raise PostflightError("core4 overlay/adapter/recovery identity differs across cases")
        case_rows.append(
            {
                "iid": iid,
                "path": str(path),
                "sha256": file_sha256(path),
                "receipt_digest": value["receipt_digest"],
                "decoded_outputs_byte_identical": value[
                    "decoded_outputs_byte_identical"
                ],
                "pair_receipt_path": value["paired_receipt"]["path"],
                "pair_receipt_sha256": value["paired_receipt"]["sha256"],
                "pair_receipt_digest": value["paired_receipt"]["receipt_digest"],
                "recovery_receipt_sha256": recovered["sha256"],
                "recovery_receipt_digest": recovered["receipt_digest"],
            }
        )

    subprocess.run(
        [
            str(python_bin),
            "-B",
            str(runner),
            "--spec",
            str(spec),
            "--expected-spec-sha256",
            expected_spec_sha256,
            "verify-core4",
            "--output-root",
            str(results_root),
        ],
        check=True,
    )
    core4 = _verified_receipt(core4_path, label="core4 master receipt")
    if (
        core4.get("schema_version") != CORE4_SCHEMA_VERSION
        or core4.get("status")
        != "core4_decoded_pairs_complete_pending_blind_full_video_review"
        or core4.get("case_count") != 4
        or core4.get("dog_count") != 2
        or core4.get("human_count") != 2
        or core4.get("decision_contract") != CORE4_DECISION_CONTRACT
        or core4.get("training_completion_is_method_success") is not False
        or core4.get("decoded_generation_completion_is_method_success") is not False
        or core4.get("method_success_authorized") is not False
        or not isinstance(core4.get("pairs"), list)
        or len(core4["pairs"]) != 4
    ):
        raise PostflightError("core4 master receipt contract differs")
    pairs = {str(row.get("iid")): row for row in core4["pairs"] if isinstance(row, Mapping)}
    if set(pairs) != set(CORE4_IIDS):
        raise PostflightError("core4 master pair IID closure differs")
    for row in case_rows:
        pair = pairs[row["iid"]]
        if (
            pair.get("pair_receipt_path") != row["pair_receipt_path"]
            or pair.get("pair_receipt_sha256") != row["pair_receipt_sha256"]
            or pair.get("pair_receipt_digest") != row["pair_receipt_digest"]
        ):
            raise PostflightError("case binding/core4 pair cross-bind differs")
    assert (
        overlay_identity is not None
        and adapter_identity is not None
        and heldout_runner_sha256 is not None
        and recovery_identity is not None
    )
    if recovery_identity != (
        external_recovery["receipt_path"], external_recovery["sha256"],
        external_recovery["receipt_digest"], external_recovery["final_checkpoint_path"],
        external_recovery["final_training_receipt_digest"],
        external_recovery["final_adapter_model_sha256"],
    ):
        raise PostflightError("core4 recovery identity differs from external pin")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "full160_core4_execution_closed_pending_strict_paired_full_video_review"
        ),
        "array_job_id": array_job_id,
        "array_task_count": 2,
        "task_completions": tasks,
        "training_method_source": {
            "revision": binder.TRAINING_REVISION,
            "archive_sha256": binder.TRAINING_ARCHIVE_SHA256,
        },
        "inference_runtime_overlay": {
            "archive_sha256": overlay_identity[0],
            "manifest_sha256": overlay_identity[1],
            "manifest_digest": overlay_identity[2],
            "heldout_runner_sha256": heldout_runner_sha256,
            "is_training_method_archive": False,
        },
        "trained_adapter": {
            "adapter_model_sha256": adapter_identity[0],
            "training_receipt_sha256": adapter_identity[1],
            "training_receipt_digest": adapter_identity[2],
            "training_global_step": 160,
            "training_max_steps": 160,
        },
        "checkpoint_recovery": {
            "receipt_path": recovery_identity[0],
            "receipt_sha256": recovery_identity[1],
            "receipt_digest": recovery_identity[2],
            "final_checkpoint_path": recovery_identity[3],
            "final_training_receipt_digest": recovery_identity[4],
            "final_adapter_model_sha256": recovery_identity[5],
            "job_id": 135313,
            "slurm_job_success": False,
            "checkpoint_heldout_eligible": True,
        },
        "cases": case_rows,
        "decision_contract": dict(CORE4_DECISION_CONTRACT),
        "decoded_outputs_byte_identical_count": sum(
            int(row["decoded_outputs_byte_identical"]) for row in case_rows
        ),
        "core4_master_receipt": {
            "path": str(core4_path),
            "sha256": file_sha256(core4_path),
            "receipt_digest": core4["receipt_digest"],
        },
        "full_video_action_and_preservation_review_complete": False,
        "method_success_claimed": False,
        "method_success_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--array-job-id", required=True, type=int)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--recovery-receipt", required=True)
    parser.add_argument("--expected-recovery-receipt-sha256", required=True)
    parser.add_argument("--expected-recovery-receipt-digest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.array_job_id <= 0:
        raise PostflightError("array job ID must be positive")
    if _SHA256.fullmatch(args.expected_spec_sha256) is None:
        raise PostflightError("expected spec SHA differs")
    python_bin = _executable_file(args.python_bin, label="Python executable")
    output_root = _directory(args.output_root, label="eval output root")
    unsigned = aggregate(
        output_root=output_root,
        array_job_id=args.array_job_id,
        python_bin=python_bin,
        expected_spec_sha256=args.expected_spec_sha256,
        recovery_receipt=_plain_file(args.recovery_receipt, label="FM160 recovery receipt"),
        expected_recovery_receipt_sha256=args.expected_recovery_receipt_sha256,
        expected_recovery_receipt_digest=args.expected_recovery_receipt_digest,
    )
    result = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    output = Path(args.output)
    if not output.is_absolute() or output.parent.resolve(strict=True) != output_root:
        raise PostflightError("master binding output must be fresh under eval root")
    _write_create_only(output, result)
    print(canonical_json_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PostflightError, subprocess.CalledProcessError) as error:
        print(f"[seer-full160-core4-postflight] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
