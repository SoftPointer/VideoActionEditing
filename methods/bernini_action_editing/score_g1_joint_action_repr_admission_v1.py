#!/usr/bin/env python3
"""Conjunctive G1 admission for the joint ``M_flow + Delta H_middle`` ABI.

Input rows are deterministic receipts from
``evaluate_g1_action_repr_selectivity_v1.py``.  For every case and anchor kind,
correct must beat each preregistered control on its relevant atomic axes in
*both* flow and middle.  Modalities, axes, controls and cases are never
averaged.  Target and selfgen may close in the preregistered order via an
explicit ``target``, ``selfgen`` or ``both`` admission scope.  The scope is
sealed in the evidence manifest, repeated on the CLI, and recorded in the
replayable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


EVIDENCE_SCHEMA_VERSION = "bernini-g1-joint-action-repr-evidence-manifest-v2"
ADMISSION_SCHEMA_VERSION = "bernini-g1-joint-action-repr-admission-receipt-v2"
ANCHOR_KINDS = ("target", "selfgen")
ADMISSION_SCOPES = (*ANCHOR_KINDS, "both")
SPLITS = ("fit", "heldout")
MODALITIES = ("flow", "middle")
SCORE_AXES = (
    "action_identity",
    "action_presence",
    "onset",
    "ordered_transitions",
    "completion",
    "terminal_hold",
)
BRANCHES = (
    "correct",
    "zero_or_noop",
    "temporal_shuffle",
    "reverse",
    "incomplete",
    "wrong_action_energy_matched",
)
REQUIRED_COMPARISON_AXES: dict[str, tuple[str, ...]] = {
    "zero_or_noop": ("action_presence", "onset"),
    "temporal_shuffle": ("ordered_transitions",),
    "reverse": ("ordered_transitions",),
    "incomplete": ("completion", "terminal_hold"),
    "wrong_action_energy_matched": ("action_identity",),
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class G1JointAdmissionError(RuntimeError):
    """Raised when joint G1 evidence is structurally incomplete or altered."""


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            text = json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n"
        else:
            text = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
    except (TypeError, ValueError, UnicodeError) as error:
        raise G1JointAdmissionError("admission value is not finite canonical JSON") from error
    return text.encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_path(value: Path | str, *, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise G1JointAdmissionError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _read_json(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any], str]:
    resolved = _regular_path(path, label=label)
    payload = resolved.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise G1JointAdmissionError(f"{label} must be ASCII JSON") from error
    if not isinstance(value, dict):
        raise G1JointAdmissionError(f"{label} must be a JSON object")
    return resolved, value, hashlib.sha256(payload).hexdigest()


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise G1JointAdmissionError(f"{label} must be a sealed identifier")
    return value


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise G1JointAdmissionError(f"{label} must be lowercase SHA-256")
    return value


def _closed(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise G1JointAdmissionError(f"{label} field closure differs")
    return value


def _evaluator_module() -> Any:
    name = "evaluate_g1_action_repr_selectivity_v1_for_joint_admission"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().with_name("evaluate_g1_action_repr_selectivity_v1.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise G1JointAdmissionError("cannot load deterministic G1 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G1JointAdmissionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise G1JointAdmissionError(f"{label} must be finite in [0,1]")
    return result


def _validate_scores(value: Any, *, modality: str) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping) or set(value) != set(BRANCHES):
        raise G1JointAdmissionError(f"{modality} branch score closure differs")
    result: dict[str, dict[str, float]] = {}
    for branch in BRANCHES:
        axes = value[branch]
        if not isinstance(axes, Mapping) or set(axes) != set(SCORE_AXES):
            raise G1JointAdmissionError(f"{modality}.{branch} axis closure differs")
        result[branch] = {
            axis: _score(axes[axis], label=f"{modality}.{branch}.{axis}")
            for axis in SCORE_AXES
        }
    return result


def _modality_decision(scores: Mapping[str, Mapping[str, float]]) -> tuple[dict[str, Any], bool]:
    comparisons: dict[str, Any] = {}
    all_passed = True
    for control, axes in REQUIRED_COMPARISON_AXES.items():
        axis_rows: dict[str, Any] = {}
        for axis in axes:
            correct = float(scores["correct"][axis])
            negative = float(scores[control][axis])
            margin = correct - negative
            passed = margin > 0.0
            all_passed = all_passed and passed
            axis_rows[axis] = {
                "correct": correct,
                "control": negative,
                "margin": margin,
                "required": "strictly_positive",
                "passed": passed,
            }
        comparisons[control] = {
            "axes": axis_rows,
            "passed": all(item["passed"] for item in axis_rows.values()),
        }
    return comparisons, all_passed


def evaluate_manifest(
    path: Path | str,
    *,
    admission_scope: str = "both",
) -> dict[str, Any]:
    if admission_scope not in ADMISSION_SCOPES:
        raise G1JointAdmissionError(
            f"admission_scope must be one of {ADMISSION_SCOPES}"
        )
    required_anchor_kinds = (
        ANCHOR_KINDS if admission_scope == "both" else (admission_scope,)
    )
    manifest_path, manifest, manifest_sha = _read_json(path, label="joint G1 evidence manifest")
    row = _closed(
        manifest,
        {
            "schema_version",
            "experiment_id",
            "admission_scope",
            "expected_cases",
            "evaluations",
        },
        label="joint G1 evidence manifest",
    )
    if row["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise G1JointAdmissionError("joint evidence schema differs")
    if row["admission_scope"] != admission_scope:
        raise G1JointAdmissionError(
            "CLI admission scope differs from the manifest-sealed scope"
        )
    experiment_id = _identifier(row["experiment_id"], label="experiment_id")
    raw_cases = row["expected_cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise G1JointAdmissionError("expected_cases must be non-empty")
    cases: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(raw_cases):
        case = _closed(raw, {"case_id", "split", "action_family"}, label=f"expected_cases[{index}]")
        case_id = _identifier(case["case_id"], label="case_id")
        split = case["split"]
        family = _identifier(case["action_family"], label="action_family")
        if split not in SPLITS or case_id in cases:
            raise G1JointAdmissionError("expected case split/uniqueness differs")
        cases[case_id] = {"case_id": case_id, "split": split, "action_family": family}
    if {case["split"] for case in cases.values()} != set(SPLITS):
        raise G1JointAdmissionError("expected cases must cover fit and heldout")

    evaluations = row["evaluations"]
    if (
        not isinstance(evaluations, list)
        or len(evaluations) != len(required_anchor_kinds) * len(cases)
    ):
        raise G1JointAdmissionError(
            "evaluations must exactly cover every case in the sealed admission scope"
        )
    evaluator = _evaluator_module()
    evaluation_fields = {
        "case_id",
        "split",
        "action_family",
        "anchor_kind",
        "evaluation_receipt_path",
        "evaluation_receipt_sha256",
    }
    decisions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(evaluations):
        evidence = _closed(raw, evaluation_fields, label=f"evaluations[{index}]")
        case_id = _identifier(evidence["case_id"], label="evaluation case_id")
        anchor_kind = evidence["anchor_kind"]
        if case_id not in cases or anchor_kind not in required_anchor_kinds:
            raise G1JointAdmissionError("evaluation case/anchor differs")
        key = (case_id, anchor_kind)
        if key in seen:
            raise G1JointAdmissionError("duplicate evaluation case/anchor")
        seen.add(key)
        case = cases[case_id]
        if evidence["split"] != case["split"] or evidence["action_family"] != case["action_family"]:
            raise G1JointAdmissionError("evaluation metadata differs from expected case")
        receipt_path = _regular_path(evidence["evaluation_receipt_path"], label="evaluation receipt")
        receipt_sha = _sha(evidence["evaluation_receipt_sha256"], label="evaluation receipt SHA")
        if _sha256_file(receipt_path) != receipt_sha:
            raise G1JointAdmissionError("evaluation receipt SHA differs")
        try:
            receipt = evaluator.verify_evaluation_receipt(receipt_path)
        except Exception as error:
            raise G1JointAdmissionError("deterministic evaluation receipt failed replay") from error
        if (
            receipt.get("case_id") != case_id
            or receipt.get("action_family") != case["action_family"]
            or receipt.get("subject_anchor_kind") != anchor_kind
            or receipt.get("reference_anchor_kind") != "target"
            or (
                anchor_kind == "selfgen"
                and receipt.get("selfgen_uses_same_case_real_target_reference") is not True
            )
            or receipt.get("scoring_contract", {}).get("weighted_compensation_forbidden") is not True
            or receipt.get("scoring_contract", {}).get("total_energy_only") is not False
        ):
            raise G1JointAdmissionError("deterministic evaluation identity/contract differs")
        modality_scores = receipt.get("modality_scores")
        if not isinstance(modality_scores, Mapping) or set(modality_scores) != set(MODALITIES):
            raise G1JointAdmissionError("evaluation modality closure differs")
        modality_decisions: dict[str, Any] = {}
        modality_passes: dict[str, bool] = {}
        for modality in MODALITIES:
            scores = _validate_scores(modality_scores[modality], modality=modality)
            comparisons, passed = _modality_decision(scores)
            modality_decisions[modality] = {
                "comparisons": comparisons,
                "passed": passed,
            }
            modality_passes[modality] = passed
        joint_pass = all(modality_passes.values())
        decisions.append(
            {
                "case_id": case_id,
                "split": case["split"],
                "action_family": case["action_family"],
                "anchor_kind": anchor_kind,
                "evaluation_receipt_path": str(receipt_path),
                "evaluation_receipt_sha256": receipt_sha,
                "modality_decisions": modality_decisions,
                "joint_flow_and_middle_passed": joint_pass,
            }
        )
    expected_keys = {
        (case_id, anchor)
        for case_id in cases
        for anchor in required_anchor_kinds
    }
    if seen != expected_keys:
        raise G1JointAdmissionError(
            "evaluation coverage differs from the sealed admission scope"
        )
    decisions.sort(key=lambda item: (item["anchor_kind"], item["split"], item["case_id"]))

    anchor_decisions: dict[str, Any] = {}
    for anchor_kind in ANCHOR_KINDS:
        branch = [item for item in decisions if item["anchor_kind"] == anchor_kind]
        if anchor_kind not in required_anchor_kinds:
            anchor_decisions[anchor_kind] = {
                "status": "not_evaluated",
                "evaluated": False,
                "case_ids": [],
                "flow_g1_passed": None,
                "middle_g1_passed": None,
                "joint_g1_passed": None,
                "failed_flow_case_ids": [],
                "failed_middle_case_ids": [],
                "optimizer_g1_requirement_satisfied": None,
            }
            continue
        modality_results = {
            modality: all(item["modality_decisions"][modality]["passed"] for item in branch)
            for modality in MODALITIES
        }
        joint = len(branch) == len(cases) and all(modality_results.values()) and all(
            item["joint_flow_and_middle_passed"] for item in branch
        )
        anchor_decisions[anchor_kind] = {
            "status": "passed" if joint else "failed",
            "evaluated": True,
            "case_ids": sorted(item["case_id"] for item in branch),
            "flow_g1_passed": modality_results["flow"],
            "middle_g1_passed": modality_results["middle"],
            "joint_g1_passed": joint,
            "failed_flow_case_ids": sorted(
                item["case_id"] for item in branch if not item["modality_decisions"]["flow"]["passed"]
            ),
            "failed_middle_case_ids": sorted(
                item["case_id"] for item in branch if not item["modality_decisions"]["middle"]["passed"]
            ),
            "optimizer_g1_requirement_satisfied": joint,
        }
    scope_passed = all(
        anchor_decisions[anchor]["joint_g1_passed"] is True
        for anchor in required_anchor_kinds
    )
    all_anchor_kinds_passed = scope_passed if admission_scope == "both" else None
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "admission_scope": admission_scope,
        "scope_required_anchor_kinds": list(required_anchor_kinds),
        "evidence_manifest_path": str(manifest_path),
        "evidence_manifest_sha256": manifest_sha,
        "policy": {
            "abi": ["M_flow", "Delta_H_middle"],
            "admission_scope": admission_scope,
            "required_anchor_kinds": list(required_anchor_kinds),
            "required_branches": list(BRANCHES),
            "required_comparison_axes": {
                control: list(axes) for control, axes in REQUIRED_COMPARISON_AXES.items()
            },
            "margin_rule": "correct_minus_control_strictly_greater_than_zero",
            "modality_rule": "flow_pass_AND_middle_pass",
            "case_rule": "all_controls_and_required_axes_pass_without_averaging",
            "anchor_rule": "all_expected_fit_and_heldout_cases_pass",
            "target_and_selfgen_judged_separately": True,
            "selfgen_reference": "same_case_real_target_correct",
            "weighted_compensation_forbidden": True,
        },
        "cohort_decisions": decisions,
        "anchor_decisions": anchor_decisions,
        "g1_target_passed": anchor_decisions["target"]["joint_g1_passed"],
        "g1_selfgen_passed": anchor_decisions["selfgen"]["joint_g1_passed"],
        "g1_target_status": anchor_decisions["target"]["status"],
        "g1_selfgen_status": anchor_decisions["selfgen"]["status"],
        "g1_scope_passed": scope_passed,
        "g1_all_anchor_kinds_passed": all_anchor_kinds_passed,
        "weighted_or_scalar_compensation_used": False,
        "optimizer_creation_authorized_by_this_receipt": False,
        "stage_b_optimizer_still_requires": [
            "G0_integrity",
            "G1_relevant_anchor_kind_joint_flow_middle",
            "G2a_zero_init_noop",
            "information_firewall_implementation_and_tests",
        ],
    }


def _write_create_only(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise G1JointAdmissionError(f"refusing to overwrite admission output: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def score_and_publish(
    evidence_manifest: Path | str,
    output: Path | str,
    *,
    admission_scope: str = "both",
) -> dict[str, Any]:
    output_path = Path(output).expanduser().absolute()
    if output_path.suffix != ".json" or output_path.exists() or output_path.is_symlink():
        raise G1JointAdmissionError("admission output must be a new .json file")
    result = evaluate_manifest(
        evidence_manifest,
        admission_scope=admission_scope,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_create_only(output_path, _canonical_json_bytes(result, pretty=True))
    verify_admission_receipt(output_path)
    return result


def verify_admission_receipt(path: Path | str) -> dict[str, Any]:
    _, receipt, _ = _read_json(path, label="joint G1 admission receipt")
    if receipt.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        raise G1JointAdmissionError("joint admission receipt schema differs")
    scope = receipt.get("admission_scope")
    if scope not in ADMISSION_SCOPES:
        raise G1JointAdmissionError("joint admission receipt scope differs")
    replay = evaluate_manifest(
        receipt.get("evidence_manifest_path"),
        admission_scope=scope,
    )
    if _canonical_json_bytes(receipt) != _canonical_json_bytes(replay):
        raise G1JointAdmissionError("joint admission receipt does not replay exactly")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score joint flow+middle G1 admission without compensation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--evidence-manifest", required=True)
    score.add_argument("--output", required=True)
    score.add_argument(
        "--admission-scope",
        choices=ADMISSION_SCOPES,
        default="both",
        help="must exactly match admission_scope sealed in the evidence manifest",
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "score":
        result = score_and_publish(
            args.evidence_manifest,
            args.output,
            admission_scope=args.admission_scope,
        )
    else:
        result = verify_admission_receipt(args.receipt)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["g1_scope_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
