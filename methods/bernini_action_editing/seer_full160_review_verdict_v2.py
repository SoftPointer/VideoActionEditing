#!/usr/bin/env python3
"""Strict human-review schema and deterministic SEER full160 verdict reducer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


BLANK_SCHEMA_VERSION = "bernini-seer-full160-strict-paired-full-video-review-v2"
VERDICT_SCHEMA_VERSION = "bernini-seer-full160-strict-verdict-v2"
CORE4_IIDS = (
    "99cde432839f4240",
    "6ea45d35943742bb",
    "311c82f83eca4a7f",
    "6d346c38cf504493",
)
FAMILIES = {
    "99cde432839f4240": "dog",
    "6ea45d35943742bb": "dog",
    "311c82f83eca4a7f": "human",
    "6d346c38cf504493": "human",
}
THRESHOLDS = {
    "minimum_trained_joint_safe": 3,
    "minimum_base_fail_to_trained_joint_safe_flips": 2,
    "minimum_flip_actor_family_coverage": ["dog", "human"],
    "maximum_joint_safe_to_unsafe_regressions": 0,
}
JOINT_SAFE_DEFINITION = (
    "requested_action_reaches_terminal_and_holds AND "
    "identity_scene_camera_inventory_preserved"
)
MASTER_DECISION_CONTRACT = {
    "joint_safe_definition": JOINT_SAFE_DEFINITION,
    **THRESHOLDS,
    "human_full_video_review_required": True,
    "latent_only_or_pipeline_only_result_is_success": False,
    "training_completion_is_success": False,
    "unit": "paired_heldout_source_seed",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NODE = re.compile(r"auh[0-9A-Za-z-]+\Z")
METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT / "tools"))
import recover_seer_fm160_job135313_v1 as recovery  # noqa: E402


class VerdictError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VerdictError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise VerdictError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise VerdictError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise VerdictError(f"{label} must be a plain file")
    return path.resolve(strict=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerdictError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise VerdictError(f"{label} root must be an object")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise VerdictError("review/verdict output must be absolute and fresh")
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _absolute_path_text(value: Any) -> bool:
    return isinstance(value, str) and Path(value).is_absolute() and Path(value) != Path("/")


def _sha256_text(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_master_binding(master: Mapping[str, Any]) -> None:
    if set(master) != {
        "schema_version", "status", "array_job_id", "array_task_count",
        "task_completions", "training_method_source",
        "inference_runtime_overlay", "trained_adapter", "cases",
        "checkpoint_recovery",
        "decision_contract", "decoded_outputs_byte_identical_count",
        "core4_master_receipt",
        "full_video_action_and_preservation_review_complete",
        "method_success_claimed", "method_success_authorized", "receipt_digest",
    }:
        raise VerdictError("full160 master root closure differs")
    candidate = dict(master)
    declared = candidate.pop("receipt_digest", None)
    if not _sha256_text(declared) or object_sha256(candidate) != declared:
        raise VerdictError("full160 master receipt digest differs")
    if (
        master.get("schema_version")
        != "bernini-seer-full160-core4-eval-master-binding-v2"
        or master.get("status")
        != "full160_core4_execution_closed_pending_strict_paired_full_video_review"
        or type(master.get("array_job_id")) is not int
        or master["array_job_id"] <= 0
        or master.get("array_task_count") != 2
        or master.get("decision_contract") != MASTER_DECISION_CONTRACT
        or master.get("full_video_action_and_preservation_review_complete") is not False
        or master.get("method_success_claimed") is not False
        or master.get("method_success_authorized") is not False
    ):
        raise VerdictError("full160 master execution contract differs")

    tasks = master.get("task_completions")
    if not isinstance(tasks, list) or len(tasks) != 2:
        raise VerdictError("full160 master task closure differs")
    for index, row in enumerate(tasks):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"task", "node", "path", "sha256"}
            or row.get("task") != index
            or not isinstance(row.get("node"), str)
            or _NODE.fullmatch(row["node"]) is None
            or not _absolute_path_text(row.get("path"))
            or not _sha256_text(row.get("sha256"))
        ):
            raise VerdictError("full160 master task closure differs")

    if master.get("training_method_source") != {
        "revision": "6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a",
        "archive_sha256": "ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822",
    }:
        raise VerdictError("full160 master training provenance differs")
    overlay = master.get("inference_runtime_overlay")
    if (
        not isinstance(overlay, Mapping)
        or set(overlay) != {
            "archive_sha256", "manifest_sha256", "manifest_digest",
            "heldout_runner_sha256", "is_training_method_archive",
        }
        or not all(
            _sha256_text(overlay.get(field))
            for field in (
                "archive_sha256", "manifest_sha256", "manifest_digest",
                "heldout_runner_sha256",
            )
        )
        or overlay.get("is_training_method_archive") is not False
        or overlay.get("archive_sha256")
        == master["training_method_source"]["archive_sha256"]
    ):
        raise VerdictError("full160 master overlay provenance differs")
    trained = master.get("trained_adapter")
    if (
        not isinstance(trained, Mapping)
        or set(trained) != {
            "adapter_model_sha256", "training_receipt_sha256",
            "training_receipt_digest", "training_global_step", "training_max_steps",
        }
        or not all(
            _sha256_text(trained.get(field))
            for field in (
                "adapter_model_sha256", "training_receipt_sha256",
                "training_receipt_digest",
            )
        )
        or trained.get("training_global_step") != 160
        or trained.get("training_max_steps") != 160
    ):
        raise VerdictError("full160 master adapter closure differs")
    recovered = master.get("checkpoint_recovery")
    if (
        not isinstance(recovered, Mapping)
        or set(recovered) != {
            "receipt_path", "receipt_sha256", "receipt_digest", "final_checkpoint_path",
            "final_training_receipt_digest", "final_adapter_model_sha256",
            "job_id", "slurm_job_success", "checkpoint_heldout_eligible",
        }
        or not all(_sha256_text(recovered.get(field)) for field in (
            "receipt_sha256", "receipt_digest", "final_training_receipt_digest",
            "final_adapter_model_sha256",
        ))
        or not _absolute_path_text(recovered.get("final_checkpoint_path"))
        or not _absolute_path_text(recovered.get("receipt_path"))
        or recovered.get("job_id") != 135313
        or recovered.get("slurm_job_success") is not False
        or recovered.get("checkpoint_heldout_eligible") is not True
        or recovered.get("final_training_receipt_digest")
        != trained.get("training_receipt_digest")
        or recovered.get("final_adapter_model_sha256")
        != trained.get("adapter_model_sha256")
    ):
        raise VerdictError("full160 master recovery closure differs")

    cases = master.get("cases")
    if not isinstance(cases, list) or len(cases) != len(CORE4_IIDS):
        raise VerdictError("full160 master case closure differs")
    identical_count = 0
    for expected_iid, row in zip(CORE4_IIDS, cases):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "iid", "path", "sha256", "receipt_digest",
                "decoded_outputs_byte_identical", "pair_receipt_path",
                "pair_receipt_sha256", "pair_receipt_digest",
                "recovery_receipt_sha256", "recovery_receipt_digest",
            }
            or row.get("iid") != expected_iid
            or not _absolute_path_text(row.get("path"))
            or not _absolute_path_text(row.get("pair_receipt_path"))
            or not all(
                _sha256_text(row.get(field))
                for field in (
                    "sha256", "receipt_digest", "pair_receipt_sha256",
                    "pair_receipt_digest",
                    "recovery_receipt_sha256", "recovery_receipt_digest",
                )
            )
            or row.get("recovery_receipt_sha256") != recovered["receipt_sha256"]
            or row.get("recovery_receipt_digest") != recovered["receipt_digest"]
            or type(row.get("decoded_outputs_byte_identical")) is not bool
        ):
            raise VerdictError("full160 master case closure differs")
        identical_count += int(row["decoded_outputs_byte_identical"])
    if master.get("decoded_outputs_byte_identical_count") != identical_count:
        raise VerdictError("full160 master identical-output count differs")

    core4 = master.get("core4_master_receipt")
    if (
        not isinstance(core4, Mapping)
        or set(core4) != {"path", "sha256", "receipt_digest"}
        or not _absolute_path_text(core4.get("path"))
        or not _sha256_text(core4.get("sha256"))
        or not _sha256_text(core4.get("receipt_digest"))
    ):
        raise VerdictError("full160 master core4 receipt reference differs")


def _verify_external_recovery(master: Mapping[str, Any], *, path: Path,
                              expected_sha256: str, expected_digest: str) -> None:
    path = _plain_file(path, label="FM160 recovery receipt")
    if file_sha256(path) != expected_sha256:
        raise VerdictError("FM160 recovery receipt SHA differs")
    try:
        value = recovery.verify_receipt(path)
    except recovery.RecoveryError as error:
        raise VerdictError(str(error)) from error
    row = master.get("checkpoint_recovery")
    if (
        value.get("receipt_digest") != expected_digest
        or not isinstance(row, Mapping)
        or row.get("receipt_path") != str(path)
        or row.get("receipt_sha256") != expected_sha256
        or row.get("receipt_digest") != expected_digest
    ):
        raise VerdictError("FM160 recovery/master cross-bind differs")


def build_blank_review(master_binding: Mapping[str, Any]) -> dict[str, Any]:
    _validate_master_binding(master_binding)
    cases = master_binding.get("cases")
    if not isinstance(cases, list) or [row.get("iid") for row in cases if isinstance(row, Mapping)] != list(CORE4_IIDS):
        raise VerdictError("full160 master case order/closure differs")
    rows = []
    for row in cases:
        iid = row["iid"]
        rows.append(
            {
                "iid": iid,
                "actor_family": FAMILIES[iid],
                "case_binding_sha256": row["sha256"],
                "frozen_base": {
                    "requested_action_reaches_terminal_and_holds": None,
                    "identity_scene_camera_inventory_preserved": None,
                    "joint_safe": None,
                },
                "trained_adapter": {
                    "requested_action_reaches_terminal_and_holds": None,
                    "identity_scene_camera_inventory_preserved": None,
                    "joint_safe": None,
                },
                "full_video_review_complete": False,
                "reviewer_notes": None,
            }
        )
    unsigned = {
        "schema_version": BLANK_SCHEMA_VERSION,
        "status": "strict_paired_full_video_review_pending",
        "master_binding_receipt_digest": master_binding["receipt_digest"],
        "checkpoint_recovery": dict(master_binding["checkpoint_recovery"]),
        "joint_safe_definition": JOINT_SAFE_DEFINITION,
        "decision_thresholds": dict(THRESHOLDS),
        "cases": rows,
        "all_four_cases_review_complete": False,
        "method_success_authorized": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def _validate_review_arm(value: Any, *, label: str) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "requested_action_reaches_terminal_and_holds",
        "identity_scene_camera_inventory_preserved",
        "joint_safe",
    }:
        raise VerdictError(f"{label} review closure differs")
    action = value["requested_action_reaches_terminal_and_holds"]
    preserved = value["identity_scene_camera_inventory_preserved"]
    joint = value["joint_safe"]
    if type(action) is not bool or type(preserved) is not bool or type(joint) is not bool:
        raise VerdictError(f"{label} review must contain booleans")
    if joint is not (action and preserved):
        raise VerdictError(f"{label} joint_safe is not the required AND")
    return joint


def reduce_review(review: Mapping[str, Any], master_binding: Mapping[str, Any]) -> dict[str, Any]:
    _validate_master_binding(master_binding)
    candidate = dict(review)
    declared = candidate.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha256(candidate) != declared:
        raise VerdictError("completed review receipt digest differs")
    if (
        set(review) != {
            "schema_version",
            "status",
            "master_binding_receipt_digest",
            "checkpoint_recovery",
            "joint_safe_definition",
            "decision_thresholds",
            "cases",
            "all_four_cases_review_complete",
            "method_success_authorized",
            "receipt_digest",
        }
        or review.get("schema_version") != BLANK_SCHEMA_VERSION
        or review.get("status") != "completed_strict_paired_full_video_review"
        or review.get("master_binding_receipt_digest") != master_binding.get("receipt_digest")
        or review.get("checkpoint_recovery") != master_binding.get("checkpoint_recovery")
        or review.get("joint_safe_definition") != JOINT_SAFE_DEFINITION
        or review.get("decision_thresholds") != THRESHOLDS
        or master_binding.get("decision_contract") != MASTER_DECISION_CONTRACT
        or review.get("all_four_cases_review_complete") is not True
        or review.get("method_success_authorized") is not False
    ):
        raise VerdictError("completed review root contract differs")
    master_cases = master_binding.get("cases")
    reviews = review.get("cases")
    if not isinstance(master_cases, list) or not isinstance(reviews, list) or len(reviews) != 4:
        raise VerdictError("completed review case closure differs")
    bound = {row["iid"]: row for row in master_cases}
    if [row.get("iid") for row in reviews if isinstance(row, Mapping)] != list(CORE4_IIDS):
        raise VerdictError("completed review IID order differs")

    trained_safe = 0
    flips: list[str] = []
    regressions: list[str] = []
    normalized = []
    for row in reviews:
        iid = row["iid"]
        if (
            set(row) != {
                "iid", "actor_family", "case_binding_sha256", "frozen_base",
                "trained_adapter", "full_video_review_complete", "reviewer_notes",
            }
            or row["actor_family"] != FAMILIES[iid]
            or row["case_binding_sha256"] != bound[iid]["sha256"]
            or row["full_video_review_complete"] is not True
            or row["reviewer_notes"] is not None and not isinstance(row["reviewer_notes"], str)
        ):
            raise VerdictError(f"completed review row differs: {iid}")
        base_safe = _validate_review_arm(row["frozen_base"], label=f"{iid} base")
        adapted_safe = _validate_review_arm(row["trained_adapter"], label=f"{iid} trained")
        identical = bound[iid].get("decoded_outputs_byte_identical")
        if type(identical) is not bool:
            raise VerdictError(f"master output identity evidence differs: {iid}")
        if identical and row["frozen_base"] != row["trained_adapter"]:
            raise VerdictError(
                f"identical decoded outputs require identical arm labels: {iid}"
            )
        trained_safe += int(adapted_safe)
        if not identical and not base_safe and adapted_safe:
            flips.append(iid)
        if not identical and base_safe and not adapted_safe:
            regressions.append(iid)
        normalized.append(
            {
                "iid": iid,
                "actor_family": FAMILIES[iid],
                "base_joint_safe": base_safe,
                "trained_joint_safe": adapted_safe,
                "base_fail_to_trained_joint_safe": (
                    not identical and not base_safe and adapted_safe
                ),
                "base_safe_to_trained_unsafe": (
                    not identical and base_safe and not adapted_safe
                ),
                "decoded_outputs_byte_identical": identical,
                "identical_output_forced_nonflip": identical,
            }
        )
    flip_families = sorted({FAMILIES[iid] for iid in flips})
    checks = {
        "minimum_trained_joint_safe_met": trained_safe >= THRESHOLDS["minimum_trained_joint_safe"],
        "minimum_base_fail_to_trained_joint_safe_flips_met": len(flips) >= THRESHOLDS["minimum_base_fail_to_trained_joint_safe_flips"],
        "minimum_flip_actor_family_coverage_met": set(flip_families) >= set(THRESHOLDS["minimum_flip_actor_family_coverage"]),
        "maximum_joint_safe_to_unsafe_regressions_met": len(regressions) <= THRESHOLDS["maximum_joint_safe_to_unsafe_regressions"],
    }
    go = all(checks.values())
    unsigned = {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "status": "GO" if go else "NO_GO",
        "master_binding_receipt_digest": master_binding["receipt_digest"],
        "checkpoint_recovery": dict(master_binding["checkpoint_recovery"]),
        "completed_review_receipt_digest": review["receipt_digest"],
        "decision_thresholds": dict(THRESHOLDS),
        "counts": {
            "trained_joint_safe": trained_safe,
            "base_fail_to_trained_joint_safe_flips": len(flips),
            "flip_actor_family_coverage": flip_families,
            "base_safe_to_trained_unsafe_regressions": len(regressions),
        },
        "checks": checks,
        "cases": normalized,
        "all_four_cases_review_complete": True,
        "unique_go_rule": "GO iff every registered threshold check is true; otherwise NO_GO",
        "method_success": go,
        "scientific_verdict_authorized": True,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    blank = sub.add_parser("blank")
    blank.add_argument("--master-binding", required=True)
    blank.add_argument("--output", required=True)
    reduce = sub.add_parser("reduce")
    reduce.add_argument("--master-binding", required=True)
    reduce.add_argument("--completed-review", required=True)
    reduce.add_argument("--output", required=True)
    for command in (blank, reduce):
        command.add_argument("--recovery-receipt", required=True)
        command.add_argument("--expected-recovery-receipt-sha256", required=True)
        command.add_argument("--expected-recovery-receipt-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    master_path = _plain_file(args.master_binding, label="master binding")
    master = _read_json(master_path, label="master binding")
    unsigned = dict(master)
    declared = unsigned.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha256(unsigned) != declared:
        raise VerdictError("master binding receipt digest differs")
    _verify_external_recovery(
        master,
        path=Path(args.recovery_receipt),
        expected_sha256=args.expected_recovery_receipt_sha256,
        expected_digest=args.expected_recovery_receipt_digest,
    )
    if args.command == "blank":
        result = build_blank_review(master)
    else:
        review = _read_json(
            _plain_file(args.completed_review, label="completed review"),
            label="completed review",
        )
        result = reduce_review(review, master)
    _write_create_only(Path(args.output), result)
    print(canonical_json_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerdictError as error:
        print(f"[seer-full160-verdict] ERROR: {error}", file=os.sys.stderr)
        raise SystemExit(2)
