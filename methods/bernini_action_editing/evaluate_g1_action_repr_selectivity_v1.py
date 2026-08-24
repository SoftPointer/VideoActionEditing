#!/usr/bin/env python3
"""Deterministic frozen evaluator for the joint ``M_flow + Delta H_middle`` ABI.

Every candidate branch is compared with the *real-target correct* cache from
the same case.  In particular, a self-generated correct branch is never its
own reference.  Scores retain dense direction, spatial structure and temporal
phase order; total motion/residual energy is only one conjunctive ingredient.

The evaluator is read-only and has no learned parameters.  It emits six atomic
axes independently for flow and middle.  It does not average the modalities
and does not make an admission decision; the downstream G1 scorer requires
both modality-specific margins to pass.
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


SCHEMA_VERSION = "bernini-g1-joint-action-repr-evaluation-receipt-v1"
ALGORITHM_VERSION = "dense-phase-cosine-norm-conjunction-v1"
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
MODALITIES = ("flow", "middle")
PHASES = 21
EPSILON = 1.0e-12
_SHA256 = re.compile(r"[0-9a-f]{64}")


class G1DeterministicEvaluatorError(RuntimeError):
    """Raised when a deterministic G1 evaluation cannot be replayed."""


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
        raise G1DeterministicEvaluatorError("evaluation is not finite canonical JSON") from error
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
        raise G1DeterministicEvaluatorError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _read_json(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any], str]:
    resolved = _regular_path(path, label=label)
    payload = resolved.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise G1DeterministicEvaluatorError(f"{label} must be ASCII JSON") from error
    if not isinstance(value, dict):
        raise G1DeterministicEvaluatorError(f"{label} must be a JSON object")
    return resolved, value, hashlib.sha256(payload).hexdigest()


def _load_sibling(filename: str, module_name: str) -> Any:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise G1DeterministicEvaluatorError(f"cannot load sibling verifier: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _flow_module() -> Any:
    return _load_sibling(
        "materialize_g1_flow_control_cohort_v1.py",
        "materialize_g1_flow_control_cohort_v1_for_joint_eval",
    )


def _middle_module() -> Any:
    return _load_sibling(
        "materialize_g1_middle_control_cohort_v1.py",
        "materialize_g1_middle_control_cohort_v1_for_joint_eval",
    )


def _load_tensor_map(path: Path | str) -> dict[str, Any]:
    resolved = _regular_path(path, label="representation cache")
    try:
        from safetensors import safe_open
        with safe_open(str(resolved), framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(key).contiguous() for key in handle.keys()}
    except Exception as error:
        raise G1DeterministicEvaluatorError(f"cannot load representation cache: {resolved}") from error


def _branch_paths(receipt: Mapping[str, Any], *, modality: str) -> dict[str, str]:
    if modality == "flow":
        external = receipt["external_bundles"]
    elif modality == "middle":
        external = receipt["external_caches"]
    else:
        raise G1DeterministicEvaluatorError("unknown representation modality")
    generated = receipt["generated_controls"]
    return {
        "correct": external["correct"]["path"],
        "zero_or_noop": generated["zero_or_noop"]["path"],
        "temporal_shuffle": external["temporal_shuffle"]["path"],
        "reverse": external["reverse"]["path"],
        "incomplete": generated["incomplete"]["path"],
        "wrong_action_energy_matched": generated["wrong_action_energy_matched"]["path"],
    }


def _flow_phase_matrix(tensors: Mapping[str, Any]) -> Any:
    import torch

    if set(tensors) != {"backward_raw", "backward_camera_residual", "validity"}:
        raise G1DeterministicEvaluatorError("flow tensor closure differs")
    raw = tensors["backward_raw"].double()
    camera = tensors["backward_camera_residual"].double()
    validity = tensors["validity"].double()
    if (
        tuple(map(int, raw.shape[:2])) != (20, 2)
        or camera.shape != raw.shape
        or tuple(map(int, validity.shape)) != (20, 1, *map(int, raw.shape[-2:]))
        or not bool(torch.isfinite(raw).all().item())
        or not bool(torch.isfinite(camera).all().item())
        or not bool(torch.isfinite(validity).all().item())
    ):
        raise G1DeterministicEvaluatorError("flow geometry/finiteness differs")
    # Validity gates physical flow, but is also retained as one dense channel.
    # Thus a zero/noop control cannot masquerade as a valid static trajectory.
    dense = torch.cat((raw * validity, camera * validity, validity), dim=1)
    phases = dense.reshape(20, -1).contiguous()
    return torch.cat((torch.zeros_like(phases[:1]), phases), dim=0).contiguous()


def _middle_phase_matrix(tensors: Mapping[str, Any]) -> Any:
    import torch

    required = {f"middle_block_{index:02d}" for index in (6, 12, 18, 24)}
    if set(tensors) != required:
        raise G1DeterministicEvaluatorError("middle tensor closure differs")
    rows = []
    common_prefix: tuple[int, int, int] | None = None
    for key in sorted(required):
        tensor = tensors[key].double()
        if tensor.ndim != 4 or int(tensor.shape[1]) != PHASES or not bool(torch.isfinite(tensor).all().item()):
            raise G1DeterministicEvaluatorError(f"{key} middle geometry/finiteness differs")
        prefix = (int(tensor.shape[0]), int(tensor.shape[2]), int(tensor.shape[3]))
        if common_prefix is None:
            common_prefix = prefix
        elif prefix != common_prefix:
            raise G1DeterministicEvaluatorError("middle block geometry differs")
        rows.append(tensor.permute(1, 0, 2, 3).reshape(PHASES, -1))
    result = torch.cat(rows, dim=1).contiguous()
    if bool(result[0].any().item()):
        raise G1DeterministicEvaluatorError("middle phase zero is not exact zero")
    return result


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        raise G1DeterministicEvaluatorError("non-finite structured score")
    value = min(1.0, max(0.0, value))
    if abs(value) < 5.0e-15:
        return 0.0
    if abs(value - 1.0) < 5.0e-15:
        return 1.0
    return round(value, 12)


def _cosine01(candidate: Any, reference: Any) -> float:
    import torch

    left = candidate.reshape(-1).double()
    right = reference.reshape(-1).double()
    if left.numel() != right.numel() or left.numel() == 0:
        raise G1DeterministicEvaluatorError("structured cosine geometry differs")
    right_norm = float(torch.linalg.vector_norm(right).item())
    left_norm = float(torch.linalg.vector_norm(left).item())
    if right_norm <= EPSILON:
        return 1.0 if left_norm <= EPSILON else 0.0
    if left_norm <= EPSILON:
        return 0.0
    cosine = float(torch.dot(left, right).item()) / (left_norm * right_norm)
    return _bounded((max(-1.0, min(1.0, cosine)) + 1.0) * 0.5)


def _norm_match(candidate: Any, reference: Any) -> float:
    import torch

    left = float(torch.linalg.vector_norm(candidate.reshape(-1).double()).item())
    right = float(torch.linalg.vector_norm(reference.reshape(-1).double()).item())
    if right <= EPSILON:
        return 1.0 if left <= EPSILON else 0.0
    if left <= EPSILON:
        return 0.0
    return _bounded(min(left, right) / max(left, right))


def _phase_energy(value: Any) -> Any:
    import torch

    return torch.sqrt(value.double().square().mean(dim=1).clamp_min(0.0))


def _structured_pair(candidate: Any, reference: Any) -> float:
    return min(_cosine01(candidate, reference), _norm_match(candidate, reference))


def phase_structured_scores(candidate: Any, reference: Any) -> dict[str, float]:
    """Compute six non-weighted axes from two dense ``[21,D]`` phase paths."""

    if (
        candidate.ndim != 2
        or reference.ndim != 2
        or tuple(candidate.shape) != tuple(reference.shape)
        or int(candidate.shape[0]) != PHASES
        or int(candidate.shape[1]) <= 0
    ):
        raise G1DeterministicEvaluatorError("phase matrix geometry differs")
    candidate_energy = _phase_energy(candidate)
    reference_energy = _phase_energy(reference)
    candidate_delta = candidate[1:] - candidate[:-1]
    reference_delta = reference[1:] - reference[:-1]
    onset_slice = slice(1, 6)
    completion_slice = slice(PHASES - 5, PHASES)
    hold_slice = slice(PHASES - 3, PHASES)
    scores = {
        # Action identity requires both dense field direction and its ordered
        # transition signature; energy matching alone cannot raise this axis.
        "action_identity": min(
            _cosine01(candidate[1:], reference[1:]),
            _cosine01(candidate_delta, reference_delta),
            _cosine01(candidate_energy[1:], reference_energy[1:]),
        ),
        # Presence uses amplitude *and* the temporal distribution of activity.
        "action_presence": min(
            _norm_match(candidate[1:], reference[1:]),
            _cosine01(candidate_energy[1:], reference_energy[1:]),
        ),
        "onset": min(
            _structured_pair(candidate[onset_slice], reference[onset_slice]),
            _cosine01(candidate_energy[onset_slice], reference_energy[onset_slice]),
        ),
        "ordered_transitions": min(
            _structured_pair(candidate_delta, reference_delta),
            _cosine01(
                candidate_energy[1:] - candidate_energy[:-1],
                reference_energy[1:] - reference_energy[:-1],
            ),
        ),
        "completion": min(
            _structured_pair(candidate[completion_slice], reference[completion_slice]),
            _cosine01(candidate_energy[completion_slice], reference_energy[completion_slice]),
        ),
        "terminal_hold": min(
            _structured_pair(candidate[hold_slice], reference[hold_slice]),
            _cosine01(candidate_delta[-2:], reference_delta[-2:]),
            _cosine01(candidate_energy[hold_slice], reference_energy[hold_slice]),
        ),
    }
    return {axis: _bounded(float(scores[axis])) for axis in SCORE_AXES}


def _validate_pair_identity(
    target_flow: Mapping[str, Any],
    target_middle: Mapping[str, Any],
    subject_flow: Mapping[str, Any],
    subject_middle: Mapping[str, Any],
) -> tuple[str, str, str]:
    if target_flow.get("anchor_kind") != "target" or target_middle.get("anchor_kind") != "target":
        raise G1DeterministicEvaluatorError("reference must be real-target correct")
    case_ids = {
        target_flow.get("case_id"),
        target_middle.get("case_id"),
        subject_flow.get("case_id"),
        subject_middle.get("case_id"),
    }
    action_families = {
        target_flow.get("action_family"),
        target_middle.get("action_family"),
        subject_flow.get("action_family"),
        subject_middle.get("action_family"),
    }
    subject_anchor_kinds = {subject_flow.get("anchor_kind"), subject_middle.get("anchor_kind")}
    if len(case_ids) != 1 or len(action_families) != 1 or len(subject_anchor_kinds) != 1:
        raise G1DeterministicEvaluatorError("flow/middle target/subject identity differs")
    subject_anchor = next(iter(subject_anchor_kinds))
    if subject_anchor not in {"target", "selfgen"}:
        raise G1DeterministicEvaluatorError("subject anchor kind differs")
    return str(next(iter(case_ids))), str(next(iter(action_families))), str(subject_anchor)


def build_evaluation(
    *,
    target_flow_receipt: Path | str,
    target_middle_receipt: Path | str,
    subject_flow_receipt: Path | str,
    subject_middle_receipt: Path | str,
) -> dict[str, Any]:
    flow_module = _flow_module()
    middle_module = _middle_module()
    paths_and_receipts: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for name, path, verifier in (
        ("target_flow", target_flow_receipt, flow_module.verify_cohort_receipt),
        ("target_middle", target_middle_receipt, middle_module.verify_cohort_receipt),
        ("subject_flow", subject_flow_receipt, flow_module.verify_cohort_receipt),
        ("subject_middle", subject_middle_receipt, middle_module.verify_cohort_receipt),
    ):
        resolved, _, digest = _read_json(path, label=f"{name} control receipt")
        try:
            receipt = verifier(resolved)
        except Exception as error:
            raise G1DeterministicEvaluatorError(f"{name} control receipt did not replay") from error
        paths_and_receipts[name] = (resolved, receipt, digest)
    case_id, action_family, subject_anchor = _validate_pair_identity(
        paths_and_receipts["target_flow"][1],
        paths_and_receipts["target_middle"][1],
        paths_and_receipts["subject_flow"][1],
        paths_and_receipts["subject_middle"][1],
    )

    target_flow_paths = _branch_paths(paths_and_receipts["target_flow"][1], modality="flow")
    target_middle_paths = _branch_paths(paths_and_receipts["target_middle"][1], modality="middle")
    subject_flow_paths = _branch_paths(paths_and_receipts["subject_flow"][1], modality="flow")
    subject_middle_paths = _branch_paths(paths_and_receipts["subject_middle"][1], modality="middle")
    flow_reference = _flow_phase_matrix(_load_tensor_map(target_flow_paths["correct"]))
    middle_reference = _middle_phase_matrix(_load_tensor_map(target_middle_paths["correct"]))

    modality_scores: dict[str, dict[str, dict[str, float]]] = {"flow": {}, "middle": {}}
    for branch in BRANCHES:
        modality_scores["flow"][branch] = phase_structured_scores(
            _flow_phase_matrix(_load_tensor_map(subject_flow_paths[branch])),
            flow_reference,
        )
        modality_scores["middle"][branch] = phase_structured_scores(
            _middle_phase_matrix(_load_tensor_map(subject_middle_paths[branch])),
            middle_reference,
        )
    # A minimum is reported per axis only as an explicit conjunction.  The
    # admission scorer still checks both original modality margins separately.
    joint_axis_scores = {
        branch: {
            axis: min(
                modality_scores["flow"][branch][axis],
                modality_scores["middle"][branch][axis],
            )
            for axis in SCORE_AXES
        }
        for branch in BRANCHES
    }
    source_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "method_source_sha256": _sha256_file(source_path),
        "case_id": case_id,
        "action_family": action_family,
        "subject_anchor_kind": subject_anchor,
        "reference_anchor_kind": "target",
        "selfgen_uses_same_case_real_target_reference": subject_anchor == "selfgen",
        "control_receipts": {
            name: {"path": str(value[0]), "sha256": value[2]}
            for name, value in sorted(paths_and_receipts.items())
        },
        "score_axes": list(SCORE_AXES),
        "branch_order": list(BRANCHES),
        "modality_scores": modality_scores,
        "joint_axis_scores_by_minimum": joint_axis_scores,
        "scoring_contract": {
            "dense_spatial_phase_structure_retained": True,
            "temporal_differences_used": True,
            "total_energy_only": False,
            "score_components_combined_by_minimum_not_weighted_sum": True,
            "flow_and_middle_admission_margins_remain_separate": True,
            "weighted_compensation_forbidden": True,
            "higher_is_better": True,
            "score_range": [0.0, 1.0],
            "deterministic_cpu_float64_reduction": True,
        },
        "information_firewall": {
            "target_or_selfgen_video_accessed": False,
            "target_rgb_vae_clean_latent_or_absolute_hidden_accessed": False,
            "detached_flow_and_projected_middle_cache_only": True,
        },
        "training_authority": {
            "optimizer_created": False,
            "current_experiment_optimization_steps": 0,
            "generator_parameters_updated": False,
            "admission_decision_made_by_evaluator": False,
        },
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
        raise G1DeterministicEvaluatorError(f"refusing to overwrite evaluation output: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def evaluate_and_publish(
    *,
    target_flow_receipt: Path | str,
    target_middle_receipt: Path | str,
    subject_flow_receipt: Path | str,
    subject_middle_receipt: Path | str,
    output: Path | str,
) -> dict[str, Any]:
    output_path = Path(output).expanduser().absolute()
    if output_path.suffix != ".json" or output_path.exists() or output_path.is_symlink():
        raise G1DeterministicEvaluatorError("evaluation output must be a new .json file")
    result = build_evaluation(
        target_flow_receipt=target_flow_receipt,
        target_middle_receipt=target_middle_receipt,
        subject_flow_receipt=subject_flow_receipt,
        subject_middle_receipt=subject_middle_receipt,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_create_only(output_path, _canonical_json_bytes(result, pretty=True))
    verify_evaluation_receipt(output_path)
    return result


def verify_evaluation_receipt(path: Path | str) -> dict[str, Any]:
    _, receipt, _ = _read_json(path, label="G1 evaluation receipt")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise G1DeterministicEvaluatorError("evaluation receipt schema differs")
    controls = receipt.get("control_receipts")
    if not isinstance(controls, Mapping) or set(controls) != {
        "target_flow", "target_middle", "subject_flow", "subject_middle"
    }:
        raise G1DeterministicEvaluatorError("evaluation control receipt closure differs")
    replay = build_evaluation(
        target_flow_receipt=controls["target_flow"]["path"],
        target_middle_receipt=controls["target_middle"]["path"],
        subject_flow_receipt=controls["subject_flow"]["path"],
        subject_middle_receipt=controls["subject_middle"]["path"],
    )
    if _canonical_json_bytes(receipt) != _canonical_json_bytes(replay):
        raise G1DeterministicEvaluatorError("evaluation receipt does not replay exactly")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate joint M_flow + Delta-H-middle G1 selectivity.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--target-flow-receipt", required=True)
    evaluate.add_argument("--target-middle-receipt", required=True)
    evaluate.add_argument("--subject-flow-receipt", required=True)
    evaluate.add_argument("--subject-middle-receipt", required=True)
    evaluate.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_and_publish(
            target_flow_receipt=args.target_flow_receipt,
            target_middle_receipt=args.target_middle_receipt,
            subject_flow_receipt=args.subject_flow_receipt,
            subject_middle_receipt=args.subject_middle_receipt,
            output=args.output,
        )
    else:
        result = verify_evaluation_receipt(args.receipt)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
