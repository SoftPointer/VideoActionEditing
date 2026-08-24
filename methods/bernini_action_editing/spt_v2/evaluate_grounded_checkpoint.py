#!/usr/bin/env python3
"""Checkpoint-wide evaluation for the trusted SPT-v3 membership.

This is a read-only evaluator.  It evaluates every one of the 13 immutable
teacher-trust members exactly once, with one identical grounded-planner
checkpoint replicated over four data-parallel ranks.  The paired target is
accepted only by :func:`paired_teacher_plan`; the student entry point has only
the clean source and instruction tokens.

The report retains integer TP/FP/FN/TN sufficient statistics.  Micro metrics
are derived from their sums, macro metrics are explicit row means, and every
row remains available for auditing.  In particular, the Generate ceiling is
measured directly from the post-budget plan gates instead of trusting a model
diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
for _root in (HERE, METHOD_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import audit_teacher_cohort as cohort  # noqa: E402
import grounded_phase_planner as grounded  # noqa: E402
import motion_residual as motion  # noqa: E402
import phase_transport as spt  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_student as student  # noqa: E402


EVALUATION_SCHEMA = "bernini-spt-v3-trusted-checkpoint-evaluation-v2"
EXPECTED_TRUSTED_MEMBERS = 13
EXPECTED_WORLD_SIZE = 4
EVALUATOR_ULYSSES_SIZE = 1
TEACHER_FEATURE_CHANNELS = 64
GENERATE_BUDGET = 0.12
AUDITED_TEACHER_TEMPERATURE = 0.08
AUDITED_TEACHER_GENERATE_THRESHOLD = 0.35
AUDITED_TEACHER_TRANSPORT_MARGIN = 0.05
TRUSTED_MEMBERSHIP_DIGEST = (
    "2ce012cd25debd36a357fe041949b5d09ee8347ac7e46f8e2fdfa02f048ec507"
)
TRUSTED_MEMBERSHIP_FILE_SHA256 = (
    "da695b4068ad6f39c1f2d7d676d43c2def32591495a5e279f9e016d4086325c6"
)
TRUSTED_AUDIT_DIGEST = (
    "da6c7e57464741974866f1103d7681ef4c8063dd94161fb571803a0e4adc57af"
)
TRUSTED_AUDIT_FILE_SHA256 = (
    "1ef1ba2dbd39e05090de29f769f536378f5e86499c1cd22e3de7590dd155bca1"
)
CHANGE_THRESHOLD = 0.5
NOVELTY_THRESHOLD = 0.5
RANKING_SCORE_DOMAIN = "raw_float32_change_logit"
ACTION_NOOP_DELTA_SCORE_DOMAIN = (
    "raw_float32_action_change_logit_minus_noop_change_logit"
)
RANKING_SCORE_ENCODING = "python_float_hex_exact_float32_value"
RANKING_TIE_POLICY = "group_exact_equal_scores_before_metric_update"
TOPK_TIE_POLICY = "canonical_flat_yx_index_ascending"
SCREEN_THRESHOLDS = {
    "change_iou": 0.40,
    "change_precision": 0.50,
    "change_recall": 0.50,
    "minimum_change_ratio": 0.70,
    "maximum_change_ratio": 1.30,
    "conditional_tg_f1": 0.60,
    "offset_top1_accuracy": 0.35,
    "maximum_offset_mae": 1.00,
    "maximum_generate_fraction_per_phase": 0.12002,
    "maximum_noop_change_fraction": 0.01,
    "maximum_noop_soft_change_fraction": 0.05,
    "minimum_noop_zero_offset_top1_accuracy": 0.99,
    "maximum_noop_offset_mae": 0.01,
    "minimum_median_row_iou": 0.35,
    "minimum_row_iou": 0.25,
    "minimum_rows_passing_iou": 10,
}


class GroundedCheckpointEvaluationError(RuntimeError):
    """Raised before accepting an incomplete or unbound evaluation."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one grounded SPT-v3 checkpoint on all trusted rows"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--planner-checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--selected-membership", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--noop-instruction", default=motion.DEFAULT_NOOP_INSTRUCTION)
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.CHECKPOINT_TREE_SHA256,
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if type(args.seed) is not int or args.seed < 0:
        raise GroundedCheckpointEvaluationError("seed must be a non-negative integer")
    if (
        not isinstance(args.noop_instruction, str)
        or not args.noop_instruction.strip()
        or args.noop_instruction != args.noop_instruction.strip()
    ):
        raise GroundedCheckpointEvaluationError("noop instruction must be non-empty")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if re.fullmatch(r"[0-9a-fA-F]{40}", str(getattr(args, name))) is None:
            raise GroundedCheckpointEvaluationError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))) is None:
            raise GroundedCheckpointEvaluationError(
                f"{name} must be a lowercase SHA-256"
            )
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise GroundedCheckpointEvaluationError(
            "base checkpoint identity differs from audited Bernini-R 1.3B"
        )
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        raise GroundedCheckpointEvaluationError("output must be an absolute path")
    if output.exists() or output.is_symlink():
        raise GroundedCheckpointEvaluationError(f"refusing to overwrite output {output}")


def student_plans(
    planner: Any,
    source: Any,
    action_instruction_tokens: Any,
    noop_instruction_tokens: Any,
) -> tuple[spt.PhasePlan, spt.PhasePlan]:
    """Run the only deployable semantic inputs through the student."""

    return (
        student.student_plan(planner, source, action_instruction_tokens),
        student.student_plan(planner, source, noop_instruction_tokens),
    )


def paired_teacher_plan(
    source: Any,
    paired_target: Any,
    teacher_config: spt.PhaseTransportConfig,
) -> spt.PhasePlan:
    """The sole target-bearing function in the evaluation path."""

    return spt.build_oracle_plan(
        source,
        paired_target,
        teacher_config,
        feature_channels=TEACHER_FEATURE_CHANNELS,
    )


def _row_edit_instruction(raw_row: Mapping[str, Any]) -> str:
    """Read and validate only the message JSON, never either latent blob."""

    try:
        messages = legacy._parse_inputs(raw_row.get("inputs"))
        legacy._validate_message_contract(messages)
    except legacy.TrainingContractError as error:
        raise GroundedCheckpointEvaluationError(
            f"row instruction contract differs: {error}"
        ) from error
    instruction = messages[1]["text"]
    if not isinstance(instruction, str):  # guarded above; explicit for type safety
        raise GroundedCheckpointEvaluationError("row instruction is not text")
    return instruction


def text_only_token_batches(
    *,
    raw_row: Mapping[str, Any],
    tokenizer: Any,
    noop_instruction: str,
    prompt_cleaner: Any,
    system_prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce ``encode_renderer_messages`` without touching video data."""

    import torch

    if not isinstance(system_prompt, str) or not system_prompt:
        raise GroundedCheckpointEvaluationError("runtime mv2v system prompt is empty")
    action_instruction = _row_edit_instruction(raw_row)

    def _encode(instruction: str) -> dict[str, Any]:
        if not isinstance(instruction, str) or not instruction.strip() or "\x00" in instruction:
            raise GroundedCheckpointEvaluationError(
                "text-only instruction must be non-empty and contain no NUL"
            )
        cleaned = prompt_cleaner(instruction)
        if not isinstance(cleaned, str) or not cleaned.strip():
            raise GroundedCheckpointEvaluationError("Wan prompt cleaner returned empty text")
        encoded = tokenizer(
            system_prompt + cleaned,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        if (
            getattr(input_ids, "ndim", None) != 2
            or tuple(input_ids.shape) != tuple(attention_mask.shape)
            or int(input_ids.shape[0]) != 1
            or int(input_ids.shape[1]) <= 0
            or not bool(torch.isfinite(input_ids).all())
            or not bool(torch.isfinite(attention_mask).all())
        ):
            raise GroundedCheckpointEvaluationError(
                "tokenizer returned invalid text-only renderer tensors"
            )
        length = int(input_ids.shape[1])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # Official single-sample collation turns the pre-collate [1]
            # length tensor into [1,1].
            "t5_input_lens": torch.tensor([[length]], dtype=torch.long),
        }

    return _encode(action_instruction), _encode(noop_instruction)


def _binary_counts(predicted: Any, truth: Any, *, mask: Optional[Any] = None) -> dict[str, int]:
    import torch

    if tuple(predicted.shape) != tuple(truth.shape):
        raise GroundedCheckpointEvaluationError("binary prediction/truth shapes differ")
    predicted = predicted.bool()
    truth = truth.bool()
    selected = torch.ones_like(truth, dtype=torch.bool) if mask is None else mask.bool()
    if tuple(selected.shape) != tuple(truth.shape):
        raise GroundedCheckpointEvaluationError("binary metric mask shape differs")
    return {
        "tp": int((predicted & truth & selected).sum().item()),
        "fp": int((predicted & ~truth & selected).sum().item()),
        "fn": int((~predicted & truth & selected).sum().item()),
        "tn": int((~predicted & ~truth & selected).sum().item()),
    }


def metrics_from_counts(counts: Mapping[str, int]) -> dict[str, Any]:
    """Derive metrics with the same explicit empty-set rule as training."""

    try:
        tp, fp, fn, tn = (int(counts[name]) for name in ("tp", "fp", "fn", "tn"))
    except (KeyError, TypeError, ValueError) as error:
        raise GroundedCheckpointEvaluationError(
            f"invalid binary sufficient statistics: {error}"
        ) from error
    if min(tp, fp, fn, tn) < 0:
        raise GroundedCheckpointEvaluationError("binary counts cannot be negative")
    predicted_count = tp + fp
    target_count = tp + fn
    union = tp + fp + fn
    total = union + tn
    both_empty = predicted_count == 0 and target_count == 0
    precision = tp / predicted_count if predicted_count else float(both_empty)
    recall = tp / target_count if target_count else float(both_empty)
    iou = tp / union if union else float(both_empty)
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else float(both_empty)
    return {
        "raw_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "evaluated_cells": total,
        "predicted_positive_count": predicted_count,
        "target_positive_count": target_count,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "accuracy": (tp + tn) / total if total else 1.0,
        "predicted_fraction": predicted_count / total if total else 0.0,
        "target_fraction": target_count / total if total else 0.0,
        "change_ratio": (
            predicted_count / target_count
            if target_count
            else (1.0 if predicted_count == 0 else None)
        ),
    }


def _finite_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise GroundedCheckpointEvaluationError(
            "ranking score must be numeric"
        ) from error
    if not math.isfinite(score):
        raise GroundedCheckpointEvaluationError("ranking score must be finite")
    # Numeric thresholding treats signed zero as one tie.  Canonicalizing it
    # here also makes the hexadecimal receipt representation unique.
    return 0.0 if score == 0.0 else score


def raw_score_runs(scores: Any, truth: Any) -> list[list[Any]]:
    """Losslessly collapse raw float32 scores into exact tie groups.

    Each run is ``[score.hex(), positive_count, negative_count]`` and runs are
    ordered by decreasing numeric score.  This is an exact sufficient
    statistic for global threshold metrics: rows/ranks can be merged by score
    before computing PR-AUC or selecting a best-F1 threshold.  It deliberately
    does not average row metrics or break score ties by row/rank order.
    """

    import torch

    if tuple(scores.shape) != tuple(truth.shape):
        raise GroundedCheckpointEvaluationError(
            "ranking score/truth shapes differ"
        )
    score_tensor = scores.detach().float().cpu().reshape(-1)
    truth_tensor = truth.detach().bool().cpu().reshape(-1)
    if not bool(torch.isfinite(score_tensor).all()):
        raise GroundedCheckpointEvaluationError("ranking scores are non-finite")
    grouped: dict[float, list[int]] = {}
    for raw_score, positive in zip(score_tensor.tolist(), truth_tensor.tolist()):
        score = _finite_score(raw_score)
        counts = grouped.setdefault(score, [0, 0])
        counts[0 if bool(positive) else 1] += 1
    return [
        [score.hex(), int(grouped[score][0]), int(grouped[score][1])]
        for score in sorted(grouped, reverse=True)
    ]


def _decode_score_runs(runs: Sequence[Sequence[Any]]) -> list[tuple[float, int, int]]:
    decoded: list[tuple[float, int, int]] = []
    prior: Optional[float] = None
    for run in runs:
        if not isinstance(run, Sequence) or isinstance(run, (str, bytes)) or len(run) != 3:
            raise GroundedCheckpointEvaluationError(
                "ranking score run must contain score, positive count, negative count"
            )
        score_hex, positive, negative = run
        if not isinstance(score_hex, str):
            raise GroundedCheckpointEvaluationError(
                "ranking score run must use hexadecimal score encoding"
            )
        try:
            score = _finite_score(float.fromhex(score_hex))
        except (ValueError, OverflowError) as error:
            raise GroundedCheckpointEvaluationError(
                "ranking score hexadecimal encoding is invalid"
            ) from error
        if type(positive) is not int or type(negative) is not int:
            raise GroundedCheckpointEvaluationError(
                "ranking score-run counts must be integers"
            )
        if positive < 0 or negative < 0 or positive + negative <= 0:
            raise GroundedCheckpointEvaluationError(
                "ranking score-run counts must be non-negative and non-empty"
            )
        if prior is not None and not prior > score:
            raise GroundedCheckpointEvaluationError(
                "ranking score runs must be unique and strictly descending"
            )
        decoded.append((score, positive, negative))
        prior = score
    if not decoded:
        raise GroundedCheckpointEvaluationError("ranking score runs cannot be empty")
    return decoded


def merge_score_runs(
    run_groups: Sequence[Sequence[Sequence[Any]]],
) -> list[list[Any]]:
    """Merge exact score ties across every row/rank before deriving metrics."""

    grouped: dict[float, list[int]] = {}
    for runs in run_groups:
        for score, positive, negative in _decode_score_runs(runs):
            counts = grouped.setdefault(score, [0, 0])
            counts[0] += positive
            counts[1] += negative
    if not grouped:
        raise GroundedCheckpointEvaluationError("cannot merge zero ranking scores")
    return [
        [score.hex(), int(grouped[score][0]), int(grouped[score][1])]
        for score in sorted(grouped, reverse=True)
    ]


def ranking_metrics_from_score_runs(
    runs: Sequence[Sequence[Any]],
    *,
    retain_raw_runs: bool = True,
    score_domain: str = RANKING_SCORE_DOMAIN,
) -> dict[str, Any]:
    """Compute exact tie-aware average precision and the best F1 threshold."""

    decoded = _decode_score_runs(runs)
    positives = sum(item[1] for item in decoded)
    negatives = sum(item[2] for item in decoded)
    total = positives + negatives
    best_metrics = metrics_from_counts(
        {"tp": 0, "fp": 0, "fn": positives, "tn": negatives}
    )
    best_threshold_hex: Optional[str] = None
    best_mode = "predict_none_above_maximum_score"
    cumulative_tp = 0
    cumulative_fp = 0
    previous_recall = 0.0
    average_precision = 0.0
    for score, positive, negative in decoded:
        cumulative_tp += positive
        cumulative_fp += negative
        counts = {
            "tp": cumulative_tp,
            "fp": cumulative_fp,
            "fn": positives - cumulative_tp,
            "tn": negatives - cumulative_fp,
        }
        metrics = metrics_from_counts(counts)
        if positives:
            average_precision += (
                metrics["recall"] - previous_recall
            ) * metrics["precision"]
            previous_recall = metrics["recall"]
        # Runs are descending.  Strict improvement therefore makes the
        # highest (most conservative) threshold win exact F1 ties.
        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_threshold_hex = score.hex()
            best_mode = "score_greater_than_or_equal_to_threshold"
    result: dict[str, Any] = {
        "score_domain": score_domain,
        "score_encoding": RANKING_SCORE_ENCODING,
        "tie_policy": RANKING_TIE_POLICY,
        "raw_score_run_count": len(decoded),
        "cell_count": total,
        "positive_count": positives,
        "negative_count": negatives,
        "pr_auc": average_precision if positives else None,
        "pr_auc_available": positives > 0,
        "pr_auc_definition": (
            "stepwise_average_precision_after_exact_equal_score_tie_group"
        ),
        "best_f1": {
            "threshold_score_hex": best_threshold_hex,
            "prediction_rule": best_mode,
            "threshold_tie_policy": "highest_threshold_wins_equal_f1",
            **best_metrics,
        },
    }
    if retain_raw_runs:
        result["raw_score_runs_descending"] = [list(run) for run in runs]
    return result


def teacher_phase_cardinality_topk_statistics(
    change_logits: Any,
    teacher_change: Any,
) -> dict[str, Any]:
    """Measure saliency ranking with target-derived cardinality, never at inference.

    The teacher supplies only ``k`` independently for each sample/phase.  Raw
    student logits determine the ranking.  Exact score ties are resolved by
    ascending canonical flattened ``y * width + x`` index, which makes the
    diagnostic invariant to rank scheduling and explicit about its arbitrary
    boundary choice.
    """

    import torch

    if (
        getattr(change_logits, "ndim", None) != 5
        or int(change_logits.shape[1]) != 1
        or tuple(change_logits[:, 0].shape) != tuple(teacher_change.shape)
    ):
        raise GroundedCheckpointEvaluationError(
            "teacher-cardinality top-k score/truth shapes differ"
        )
    scores = change_logits[:, 0].detach().float().cpu()
    truth = teacher_change.detach().bool().cpu()
    if not bool(torch.isfinite(scores).all()):
        raise GroundedCheckpointEvaluationError(
            "teacher-cardinality top-k scores are non-finite"
        )
    totals = {name: 0 for name in ("tp", "fp", "fn", "tn")}
    phases: list[dict[str, Any]] = []
    batch, phase_count, height, width = map(int, scores.shape)
    cells = height * width
    for sample_index in range(batch):
        for phase_index in range(phase_count):
            phase_scores = [
                _finite_score(value)
                for value in scores[sample_index, phase_index].reshape(-1).tolist()
            ]
            phase_truth = [
                bool(value)
                for value in truth[sample_index, phase_index].reshape(-1).tolist()
            ]
            teacher_k = sum(phase_truth)
            order = sorted(
                range(cells), key=lambda index: (-phase_scores[index], index)
            )
            selected = order[:teacher_k]
            true_positive = sum(phase_truth[index] for index in selected)
            false_positive = teacher_k - true_positive
            false_negative = teacher_k - true_positive
            true_negative = cells - true_positive - false_positive - false_negative
            phase_counts = {
                "tp": int(true_positive),
                "fp": int(false_positive),
                "fn": int(false_negative),
                "tn": int(true_negative),
            }
            for name in totals:
                totals[name] += phase_counts[name]
            boundary_score: Optional[float] = (
                phase_scores[selected[-1]] if selected else None
            )
            if boundary_score is None:
                boundary_tie_total = 0
                boundary_tie_selected = 0
            else:
                boundary_tie_total = sum(
                    score == boundary_score for score in phase_scores
                )
                selected_set = set(selected)
                boundary_tie_selected = sum(
                    phase_scores[index] == boundary_score
                    for index in selected_set
                )
            phases.append(
                {
                    "sample_index": sample_index,
                    "phase_index": phase_index,
                    "height": height,
                    "width": width,
                    "teacher_cardinality": teacher_k,
                    "selected_cardinality": len(selected),
                    "raw_counts": phase_counts,
                    "boundary_score_hex": (
                        boundary_score.hex() if boundary_score is not None else None
                    ),
                    "boundary_tie_total": boundary_tie_total,
                    "boundary_tie_selected": boundary_tie_selected,
                }
            )
    return {
        "diagnostic_only": True,
        "deployable_student_metric": False,
        "paired_target_role": "teacher_change_cardinality_only",
        "student_score": RANKING_SCORE_DOMAIN,
        "selection_scope": "each_sample_each_phase_spatial_cells",
        "tie_policy": TOPK_TIE_POLICY,
        "flattened_spatial_index": "y_times_width_plus_x",
        "per_phase": phases,
        **metrics_from_counts(totals),
    }


def phase_mass_metrics_from_sufficient_statistics(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive global phase-mass errors/correlation from additive raw sums."""

    fields = (
        "count",
        "sum_predicted",
        "sum_teacher",
        "sum_absolute_error",
        "sum_squared_error",
        "sum_predicted_squared",
        "sum_teacher_squared",
        "sum_product",
    )
    try:
        count = int(raw["count"])
        values = {name: float(raw[name]) for name in fields[1:]}
    except (KeyError, TypeError, ValueError) as error:
        raise GroundedCheckpointEvaluationError(
            "phase-mass sufficient statistics are invalid"
        ) from error
    if count <= 0 or not all(math.isfinite(value) for value in values.values()):
        raise GroundedCheckpointEvaluationError(
            "phase-mass sufficient statistics are empty or non-finite"
        )
    covariance_sum = (
        values["sum_product"]
        - values["sum_predicted"] * values["sum_teacher"] / count
    )
    predicted_variance_sum = max(
        0.0,
        values["sum_predicted_squared"]
        - values["sum_predicted"] ** 2 / count,
    )
    teacher_variance_sum = max(
        0.0,
        values["sum_teacher_squared"]
        - values["sum_teacher"] ** 2 / count,
    )
    denominator = math.sqrt(predicted_variance_sum * teacher_variance_sum)
    correlation = covariance_sum / denominator if denominator > 0.0 else None
    return {
        "prediction": "spatial_mean_sigmoid_change_logits_per_sample_phase",
        "dedicated_learned_mass_head_exists": False,
        "diagnostic_only": True,
        "raw_sufficient_statistics": {
            "count": count,
            **values,
        },
        "mean_predicted_mass": values["sum_predicted"] / count,
        "mean_teacher_mass": values["sum_teacher"] / count,
        "mae": values["sum_absolute_error"] / count,
        "rmse": math.sqrt(values["sum_squared_error"] / count),
        "pearson_correlation": correlation,
        "pearson_correlation_available": correlation is not None,
    }


def phase_mass_statistics(change_logits: Any, teacher_change: Any) -> dict[str, Any]:
    import torch

    if (
        getattr(change_logits, "ndim", None) != 5
        or int(change_logits.shape[1]) != 1
        or tuple(change_logits[:, 0].shape) != tuple(teacher_change.shape)
    ):
        raise GroundedCheckpointEvaluationError("phase-mass score/truth shapes differ")
    logits = change_logits.float()
    if not bool(torch.isfinite(logits).all()):
        raise GroundedCheckpointEvaluationError("phase-mass logits are non-finite")
    predicted = torch.sigmoid(logits[:, 0]).mean(dim=(-2, -1)).double().cpu()
    target = teacher_change.float().mean(dim=(-2, -1)).double().cpu()
    difference = predicted - target
    raw = {
        "count": int(predicted.numel()),
        "sum_predicted": float(predicted.sum().item()),
        "sum_teacher": float(target.sum().item()),
        "sum_absolute_error": float(difference.abs().sum().item()),
        "sum_squared_error": float(difference.square().sum().item()),
        "sum_predicted_squared": float(predicted.square().sum().item()),
        "sum_teacher_squared": float(target.square().sum().item()),
        "sum_product": float((predicted * target).sum().item()),
    }
    return phase_mass_metrics_from_sufficient_statistics(raw)


def _required_grounded_diagnostics(plan: spt.PhasePlan, *, label: str) -> Mapping[str, Any]:
    required = {
        "architecture",
        "change_logits",
        "novelty_logits",
        "offset_candidate_logits",
        "offset_candidates",
    }
    diagnostics = plan.diagnostics
    if not isinstance(diagnostics, Mapping) or not required <= set(diagnostics):
        raise GroundedCheckpointEvaluationError(
            f"{label} grounded planner diagnostics are incomplete"
        )
    if diagnostics.get("architecture") != grounded.ARCHITECTURE_NAME:
        raise GroundedCheckpointEvaluationError(f"{label} is not grounded SPT-v3")
    return diagnostics


def _candidate_statistics(
    plan: spt.PhasePlan,
    teacher: spt.PhasePlan,
) -> dict[str, Any]:
    import torch

    diagnostics = _required_grounded_diagnostics(plan, label="action")
    logits = diagnostics["offset_candidate_logits"].float()
    candidates = diagnostics["offset_candidates"].float()
    expected_candidates = torch.tensor(
        grounded.candidate_lattice(), device=candidates.device, dtype=torch.float32
    )
    if (
        tuple(candidates.shape) != (125, 3)
        or not bool(torch.isfinite(candidates).all())
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.equal(candidates, expected_candidates))
    ):
        raise GroundedCheckpointEvaluationError(
            "student offset candidates are not the exact ordered 125 lattice"
        )
    expected_logits_shape = (
        int(teacher.offsets.shape[0]),
        125,
        *map(int, teacher.offsets.shape[2:]),
    )
    if tuple(logits.shape) != expected_logits_shape:
        raise GroundedCheckpointEvaluationError(
            "student candidate logits and teacher offsets have different geometry"
        )
    teacher_vectors = teacher.offsets.float().permute(0, 2, 3, 4, 1)
    matches = (
        teacher_vectors.unsqueeze(1)
        == candidates.view(1, 125, 1, 1, 1, 3)
    ).all(dim=-1)
    teacher_hard = teacher.gate_probs.float().argmax(dim=1)
    transport = teacher_hard == spt.GATE_TRANSPORT
    match_count = matches.sum(dim=1)
    if bool((transport & (match_count != 1)).any()):
        raise GroundedCheckpointEvaluationError(
            "teacher transport offset is outside the exact 125-candidate lattice"
        )
    target_index = matches.float().argmax(dim=1)
    predicted_index = logits.argmax(dim=1)
    predicted_vectors = candidates[predicted_index]
    plan_vectors = plan.offsets.float().permute(0, 2, 3, 4, 1)
    if not bool(torch.equal(plan_vectors, predicted_vectors)):
        raise GroundedCheckpointEvaluationError(
            "executed student offsets differ from candidate-logit top-1"
        )
    transport_cells = int(transport.sum().item())
    if transport_cells <= 0:
        raise GroundedCheckpointEvaluationError(
            "trusted row has no teacher-transport cell; offset metrics would be vacuous"
        )
    correct = int(((predicted_index == target_index) & transport).sum().item())
    absolute_axis_error_sum = float(
        ((predicted_vectors - teacher_vectors).abs() * transport.unsqueeze(-1)).sum().item()
    )
    axis_count = 3 * transport_cells
    return {
        "top1_correct": correct,
        "transport_cells": transport_cells,
        "top1_accuracy": correct / transport_cells if transport_cells else 1.0,
        "absolute_axis_error_sum": absolute_axis_error_sum,
        "axis_count": axis_count,
        "mae": absolute_axis_error_sum / axis_count if axis_count else 0.0,
    }


def _noop_offset_statistics(noop: spt.PhasePlan) -> dict[str, Any]:
    import torch

    diagnostics = _required_grounded_diagnostics(noop, label="noop")
    logits = diagnostics["offset_candidate_logits"].float()
    candidates = diagnostics["offset_candidates"].float()
    expected_candidates = torch.tensor(
        grounded.candidate_lattice(), device=candidates.device, dtype=torch.float32
    )
    if (
        tuple(candidates.shape) != (125, 3)
        or not bool(torch.isfinite(candidates).all())
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.equal(candidates, expected_candidates))
    ):
        raise GroundedCheckpointEvaluationError("noop candidate lattice differs")
    expected_logits_shape = (
        int(noop.offsets.shape[0]),
        125,
        *map(int, noop.offsets.shape[2:]),
    )
    if tuple(logits.shape) != expected_logits_shape:
        raise GroundedCheckpointEvaluationError("noop offset geometry differs")
    zero_index = grounded.candidate_lattice().index((0, 0, 0))
    predicted = logits.argmax(dim=1)
    predicted_vectors = candidates[predicted]
    noop_vectors = noop.offsets.float().permute(0, 2, 3, 4, 1)
    if not bool(torch.equal(noop_vectors, predicted_vectors)):
        raise GroundedCheckpointEvaluationError(
            "executed noop offsets differ from candidate-logit top-1"
        )
    cell_count = int(predicted.numel())
    correct = int((predicted == zero_index).sum().item())
    absolute_axis_error_sum = float(noop.offsets.float().abs().sum().item())
    axis_count = 3 * cell_count
    return {
        "zero_top1_correct": correct,
        "cells": cell_count,
        "zero_top1_accuracy": correct / cell_count if cell_count else 1.0,
        "absolute_axis_error_sum": absolute_axis_error_sum,
        "axis_count": axis_count,
        "mae_from_zero": absolute_axis_error_sum / axis_count if axis_count else 0.0,
    }


def evaluate_row_plans(
    *,
    row_index: int,
    iid: str,
    identity_sha256: str,
    source: Any,
    action: spt.PhasePlan,
    teacher: spt.PhasePlan,
    noop: spt.PhasePlan,
) -> dict[str, Any]:
    """Return JSON-domain sufficient statistics for one trusted row."""

    import torch

    if type(row_index) is not int or row_index < 0:
        raise GroundedCheckpointEvaluationError("row index must be non-negative")
    if not isinstance(iid, str) or not iid.strip():
        raise GroundedCheckpointEvaluationError("row IID must be non-empty")
    if re.fullmatch(r"[0-9a-f]{64}", identity_sha256) is None:
        raise GroundedCheckpointEvaluationError("row identity must be a SHA-256")
    for plan in (action, teacher, noop):
        plan.validate(source)
    action_diagnostics = _required_grounded_diagnostics(action, label="action")
    noop_diagnostics = _required_grounded_diagnostics(noop, label="noop")
    if action.provenance != "student" or noop.provenance != "student":
        raise GroundedCheckpointEvaluationError("student plan provenance differs")
    if teacher.provenance != "oracle_pair_proxy":
        raise GroundedCheckpointEvaluationError("teacher provenance differs")
    teacher_gates = teacher.gate_probs.float()
    if not bool(((teacher_gates == 0.0) | (teacher_gates == 1.0)).all()):
        raise GroundedCheckpointEvaluationError("teacher gates are not exactly one-hot")
    if not bool((teacher_gates.sum(dim=1) == 1.0).all()):
        raise GroundedCheckpointEvaluationError("teacher one-hot gates are invalid")
    teacher_generate_max = float(
        teacher_gates[:, spt.GATE_GENERATE].mean(dim=(-2, -1)).max().item()
    )
    if teacher_generate_max > GENERATE_BUDGET + 1.0e-6:
        raise GroundedCheckpointEvaluationError(
            "teacher violated its hardened per-phase Generate budget"
        )

    teacher_hard = teacher_gates.argmax(dim=1)
    teacher_change = teacher_hard != spt.GATE_PRESERVE
    action_change_logits = action_diagnostics["change_logits"].float()
    action_novelty_logits = action_diagnostics["novelty_logits"].float()
    noop_change_logits = noop_diagnostics["change_logits"].float()
    expected_binary_shape = (
        int(source.shape[0]),
        1,
        *map(int, source.shape[1:4]),
    )
    for label, value in (
        ("action change", action_change_logits),
        ("action novelty", action_novelty_logits),
        ("noop change", noop_change_logits),
    ):
        if tuple(value.shape) != expected_binary_shape or not bool(torch.isfinite(value).all()):
            raise GroundedCheckpointEvaluationError(f"{label} logits differ")

    action_change_head = torch.sigmoid(action_change_logits[:, 0]) >= CHANGE_THRESHOLD
    executed_action_change = (
        1.0 - action.gate_probs.float()[:, spt.GATE_PRESERVE]
    ) >= CHANGE_THRESHOLD
    hard_gate_argmax_action_change = (
        action.gate_probs.float().argmax(dim=1) != spt.GATE_PRESERVE
    )
    conditional_generate = (
        torch.sigmoid(action_novelty_logits[:, 0]) >= NOVELTY_THRESHOLD
    )
    teacher_generate = teacher_hard == spt.GATE_GENERATE
    noop_truth = torch.zeros_like(noop_change_logits[:, 0], dtype=torch.bool)
    noop_change_head = torch.sigmoid(noop_change_logits[:, 0]) >= CHANGE_THRESHOLD
    executed_noop_change = (
        1.0 - noop.gate_probs.float()[:, spt.GATE_PRESERVE]
    ) >= CHANGE_THRESHOLD
    hard_gate_argmax_noop_change = (
        noop.gate_probs.float().argmax(dim=1) != spt.GATE_PRESERVE
    )

    change_head = metrics_from_counts(
        _binary_counts(action_change_head, teacher_change)
    )
    executed_change = metrics_from_counts(
        _binary_counts(executed_action_change, teacher_change)
    )
    hard_gate_argmax_change = metrics_from_counts(
        _binary_counts(hard_gate_argmax_action_change, teacher_change)
    )
    conditional_tg = metrics_from_counts(
        _binary_counts(conditional_generate, teacher_generate, mask=teacher_change)
    )
    noop_change = metrics_from_counts(_binary_counts(noop_change_head, noop_truth))
    noop_executed_change = metrics_from_counts(
        _binary_counts(executed_noop_change, noop_truth)
    )
    noop_hard_gate_argmax_change = metrics_from_counts(
        _binary_counts(hard_gate_argmax_noop_change, noop_truth)
    )
    teacher_cardinality_topk_change = teacher_phase_cardinality_topk_statistics(
        action_change_logits,
        teacher_change,
    )
    change_ranking = ranking_metrics_from_score_runs(
        raw_score_runs(action_change_logits[:, 0], teacher_change)
    )
    action_noop_delta_ranking = ranking_metrics_from_score_runs(
        raw_score_runs(
            (action_change_logits - noop_change_logits)[:, 0],
            teacher_change,
        ),
        score_domain=ACTION_NOOP_DELTA_SCORE_DOMAIN,
    )
    phase_mass = phase_mass_statistics(action_change_logits, teacher_change)
    offset = _candidate_statistics(action, teacher)
    noop_offset = _noop_offset_statistics(noop)

    action_gates = action.gate_probs.float()
    noop_gates = noop.gate_probs.float()
    cell_count = int(action_gates[:, 0].numel())
    action_probability_sum = {
        "preserve": float(action_gates[:, spt.GATE_PRESERVE].sum().item()),
        "transport": float(action_gates[:, spt.GATE_TRANSPORT].sum().item()),
        "generate": float(action_gates[:, spt.GATE_GENERATE].sum().item()),
    }
    noop_probability_sum = {
        "preserve": float(noop_gates[:, spt.GATE_PRESERVE].sum().item()),
        "transport": float(noop_gates[:, spt.GATE_TRANSPORT].sum().item()),
        "generate": float(noop_gates[:, spt.GATE_GENERATE].sum().item()),
    }
    action_generate_phase = action_gates[:, spt.GATE_GENERATE].mean(dim=(-2, -1))
    noop_generate_phase = noop_gates[:, spt.GATE_GENERATE].mean(dim=(-2, -1))
    hard_gate_correct = int(
        (action_gates.argmax(dim=1) == teacher_hard).sum().item()
    )
    result = {
        "row_index": row_index,
        "iid": iid,
        "identity_sha256": identity_sha256,
        "cell_count": cell_count,
        "change_head": change_head,
        "executed_change": executed_change,
        "hard_gate_argmax_change": hard_gate_argmax_change,
        "conditional_tg_head": conditional_tg,
        "teacher_cardinality_topk_change": teacher_cardinality_topk_change,
        "change_ranking": change_ranking,
        "action_noop_delta_ranking": action_noop_delta_ranking,
        "phase_mass": phase_mass,
        "transport_offset": offset,
        "teacher": {
            "observed_max_generate_fraction_per_phase": teacher_generate_max,
        },
        "student_action": {
            "gate_probability_sum": action_probability_sum,
            "gate_fraction": {
                name: value / cell_count for name, value in action_probability_sum.items()
            },
            "hard_gate_correct": hard_gate_correct,
            "hard_gate_accuracy": hard_gate_correct / cell_count,
            "observed_max_generate_fraction_per_phase": float(
                action_generate_phase.max().item()
            ),
        },
        "noop": {
            "change_head": noop_change,
            "executed_change": noop_executed_change,
            "hard_gate_argmax_change": noop_hard_gate_argmax_change,
            "gate_probability_sum": noop_probability_sum,
            "gate_fraction": {
                name: value / cell_count for name, value in noop_probability_sum.items()
            },
            "observed_max_generate_fraction_per_phase": float(
                noop_generate_phase.max().item()
            ),
            "offset": noop_offset,
        },
    }
    numeric_leaves = (
        *action_probability_sum.values(),
        *noop_probability_sum.values(),
        result["student_action"]["observed_max_generate_fraction_per_phase"],
        result["noop"]["observed_max_generate_fraction_per_phase"],
        offset["absolute_axis_error_sum"],
        noop_offset["absolute_axis_error_sum"],
        phase_mass["mae"],
        phase_mass["rmse"],
        change_ranking["best_f1"]["f1"],
        action_noop_delta_ranking["best_f1"]["f1"],
    )
    if not all(math.isfinite(float(value)) for value in numeric_leaves):
        raise GroundedCheckpointEvaluationError("row metrics contain a non-finite value")
    return result


def _sum_binary_counts(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> dict[str, int]:
    totals = {name: 0 for name in ("tp", "fp", "fn", "tn")}
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        counts = value["raw_counts"]
        for name in totals:
            totals[name] += int(counts[name])
    return totals


def validate_complete_reports(
    rows: Sequence[Mapping[str, Any]],
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prove exact, duplicate-free checkpoint-wide trusted coverage."""

    if len(members) != EXPECTED_TRUSTED_MEMBERS:
        raise GroundedCheckpointEvaluationError(
            f"trusted membership must contain exactly {EXPECTED_TRUSTED_MEMBERS} rows"
        )
    expected = [
        (
            int(member["row_index"]),
            str(member["iid"]),
            str(member["identity_sha256"]),
        )
        for member in members
    ]
    if len({item[0] for item in expected}) != EXPECTED_TRUSTED_MEMBERS:
        raise GroundedCheckpointEvaluationError("trusted membership repeats a row")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["row_index"]))
    actual = [
        (
            int(row["row_index"]),
            str(row["iid"]),
            str(row["identity_sha256"]),
        )
        for row in ordered
    ]
    if actual != expected:
        raise GroundedCheckpointEvaluationError(
            "evaluation did not cover every trusted member exactly once"
        )
    return ordered


def _mean(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        if value is None or not math.isfinite(float(value)):
            raise GroundedCheckpointEvaluationError(
                f"macro metric {'.'.join(path)} is unavailable"
            )
        values.append(float(value))
    return sum(values) / len(values)


def _nullable_mean(
    rows: Sequence[Mapping[str, Any]], path: Sequence[str]
) -> dict[str, Any]:
    """Average available row values while making missingness auditable."""

    values: list[float] = []
    unavailable_count = 0
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        if value is None:
            unavailable_count += 1
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise GroundedCheckpointEvaluationError(
                f"macro metric {'.'.join(path)} is non-finite"
            )
        values.append(numeric)
    return {
        "mean": sum(values) / len(values) if values else None,
        "available_count": len(values),
        "unavailable_count": unavailable_count,
    }


def aggregate_reports(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise GroundedCheckpointEvaluationError("cannot aggregate zero rows")
    binary_paths = {
        "change_head": ("change_head",),
        "executed_change": ("executed_change",),
        "hard_gate_argmax_change": ("hard_gate_argmax_change",),
        "conditional_tg_head": ("conditional_tg_head",),
        "teacher_cardinality_topk_change": (
            "teacher_cardinality_topk_change",
        ),
        "noop_change_head": ("noop", "change_head"),
        "noop_executed_change": ("noop", "executed_change"),
        "noop_hard_gate_argmax_change": (
            "noop",
            "hard_gate_argmax_change",
        ),
    }
    micro = {
        label: metrics_from_counts(_sum_binary_counts(rows, path))
        for label, path in binary_paths.items()
    }
    ranking_paths = {
        "change_ranking": (("change_ranking",), RANKING_SCORE_DOMAIN),
        "action_noop_delta_ranking": (
            ("action_noop_delta_ranking",),
            ACTION_NOOP_DELTA_SCORE_DOMAIN,
        ),
    }
    for label, (path, score_domain) in ranking_paths.items():
        row_runs: list[Sequence[Sequence[Any]]] = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            row_runs.append(value["raw_score_runs_descending"])
        merged_runs = merge_score_runs(row_runs)
        micro[label] = ranking_metrics_from_score_runs(
            merged_runs,
            score_domain=score_domain,
        )
        micro[label]["aggregation"] = (
            "merge_exact_equal_raw_float32_score_runs_across_all_rows_then_derive"
        )
    phase_mass_fields = (
        "sum_predicted",
        "sum_teacher",
        "sum_absolute_error",
        "sum_squared_error",
        "sum_predicted_squared",
        "sum_teacher_squared",
        "sum_product",
    )
    phase_mass_raw: dict[str, Any] = {
        "count": sum(
            int(row["phase_mass"]["raw_sufficient_statistics"]["count"])
            for row in rows
        )
    }
    for name in phase_mass_fields:
        phase_mass_raw[name] = sum(
            float(row["phase_mass"]["raw_sufficient_statistics"][name])
            for row in rows
        )
    micro["phase_mass"] = phase_mass_metrics_from_sufficient_statistics(
        phase_mass_raw
    )
    micro["phase_mass"]["aggregation"] = (
        "sum_additive_raw_sufficient_statistics_across_all_rows_then_derive"
    )
    offset_correct = sum(int(row["transport_offset"]["top1_correct"]) for row in rows)
    offset_cells = sum(int(row["transport_offset"]["transport_cells"]) for row in rows)
    offset_error = sum(
        float(row["transport_offset"]["absolute_axis_error_sum"]) for row in rows
    )
    offset_axes = sum(int(row["transport_offset"]["axis_count"]) for row in rows)
    micro["transport_offset"] = {
        "top1_correct": offset_correct,
        "transport_cells": offset_cells,
        "top1_accuracy": offset_correct / offset_cells if offset_cells else 1.0,
        "absolute_axis_error_sum": offset_error,
        "axis_count": offset_axes,
        "mae": offset_error / offset_axes if offset_axes else 0.0,
    }
    noop_offset_correct = sum(int(row["noop"]["offset"]["zero_top1_correct"]) for row in rows)
    noop_offset_cells = sum(int(row["noop"]["offset"]["cells"]) for row in rows)
    noop_offset_error = sum(
        float(row["noop"]["offset"]["absolute_axis_error_sum"]) for row in rows
    )
    noop_offset_axes = sum(int(row["noop"]["offset"]["axis_count"]) for row in rows)
    micro["noop_offset"] = {
        "zero_top1_correct": noop_offset_correct,
        "cells": noop_offset_cells,
        "zero_top1_accuracy": (
            noop_offset_correct / noop_offset_cells if noop_offset_cells else 1.0
        ),
        "absolute_axis_error_sum": noop_offset_error,
        "axis_count": noop_offset_axes,
        "mae_from_zero": (
            noop_offset_error / noop_offset_axes if noop_offset_axes else 0.0
        ),
    }
    total_cells = sum(int(row["cell_count"]) for row in rows)
    action_gate_sum = {
        name: sum(float(row["student_action"]["gate_probability_sum"][name]) for row in rows)
        for name in ("preserve", "transport", "generate")
    }
    noop_gate_sum = {
        name: sum(float(row["noop"]["gate_probability_sum"][name]) for row in rows)
        for name in ("preserve", "transport", "generate")
    }
    micro["student_action"] = {
        "cell_count": total_cells,
        "gate_probability_sum": action_gate_sum,
        "gate_fraction": {name: value / total_cells for name, value in action_gate_sum.items()},
        "observed_max_generate_fraction_per_phase": max(
            float(row["student_action"]["observed_max_generate_fraction_per_phase"])
            for row in rows
        ),
        "hard_gate_correct": sum(int(row["student_action"]["hard_gate_correct"]) for row in rows),
    }
    micro["student_action"]["hard_gate_accuracy"] = (
        micro["student_action"]["hard_gate_correct"] / total_cells
    )
    micro["noop"] = {
        "cell_count": total_cells,
        "gate_probability_sum": noop_gate_sum,
        "gate_fraction": {name: value / total_cells for name, value in noop_gate_sum.items()},
        "observed_max_generate_fraction_per_phase": max(
            float(row["noop"]["observed_max_generate_fraction_per_phase"])
            for row in rows
        ),
    }

    binary_metric_names = (
        "precision",
        "recall",
        "iou",
        "f1",
        "accuracy",
        "predicted_fraction",
        "target_fraction",
        "change_ratio",
    )
    macro: dict[str, Any] = {}
    for label, path in binary_paths.items():
        macro[label] = {
            name: _mean(rows, (*path, name))
            for name in binary_metric_names
            if name != "change_ratio"
        }
        # A no-op target has zero positives by construction, so a
        # predicted/target change ratio is undefined even when prediction is
        # also empty.  Preserve explicit missingness instead of reporting 1.
        ratio = (
            {
                "mean": None,
                "available_count": 0,
                "unavailable_count": len(rows),
            }
            if label.startswith("noop_")
            else _nullable_mean(rows, (*path, "change_ratio"))
        )
        macro[label].update(
            {
                "change_ratio": ratio["mean"],
                "change_ratio_available_count": ratio["available_count"],
                "change_ratio_unavailable_count": ratio["unavailable_count"],
            }
        )
    for label, (path, _) in ranking_paths.items():
        pr_auc = _nullable_mean(rows, (*path, "pr_auc"))
        macro[label] = {
            "pr_auc": pr_auc["mean"],
            "pr_auc_available_count": pr_auc["available_count"],
            "pr_auc_unavailable_count": pr_auc["unavailable_count"],
            "best_f1": _mean(rows, (*path, "best_f1", "f1")),
        }
    phase_mass_correlation = _nullable_mean(
        rows, ("phase_mass", "pearson_correlation")
    )
    macro["phase_mass"] = {
        "mae": _mean(rows, ("phase_mass", "mae")),
        "rmse": _mean(rows, ("phase_mass", "rmse")),
        "pearson_correlation": phase_mass_correlation["mean"],
        "pearson_correlation_available_count": phase_mass_correlation[
            "available_count"
        ],
        "pearson_correlation_unavailable_count": phase_mass_correlation[
            "unavailable_count"
        ],
    }
    macro["transport_offset"] = {
        "top1_accuracy": _mean(rows, ("transport_offset", "top1_accuracy")),
        "mae": _mean(rows, ("transport_offset", "mae")),
    }
    macro["noop_offset"] = {
        "zero_top1_accuracy": _mean(rows, ("noop", "offset", "zero_top1_accuracy")),
        "mae_from_zero": _mean(rows, ("noop", "offset", "mae_from_zero")),
    }
    macro["student_action"] = {
        "observed_max_generate_fraction_per_phase": _mean(
            rows, ("student_action", "observed_max_generate_fraction_per_phase")
        )
    }
    macro["noop"] = {
        "observed_max_generate_fraction_per_phase": _mean(
            rows, ("noop", "observed_max_generate_fraction_per_phase")
        )
    }

    row_iou: dict[str, Any] = {}
    for label in ("change_head", "executed_change"):
        values = sorted(float(row[label]["iou"]) for row in rows)
        midpoint = len(values) // 2
        median = (
            values[midpoint]
            if len(values) % 2
            else 0.5 * (values[midpoint - 1] + values[midpoint])
        )
        row_iou[label] = {
            "values_sorted": values,
            "median": median,
            "threshold": SCREEN_THRESHOLDS["minimum_row_iou"],
            "rows_passing_threshold": sum(
                value >= SCREEN_THRESHOLDS["minimum_row_iou"] for value in values
            ),
        }

    head = micro["change_head"]
    executed = micro["executed_change"]
    executed_ratio = executed["change_ratio"]
    routing_checks = {
        "finite_and_contract_valid": True,
        "change_head_iou": head["iou"] >= SCREEN_THRESHOLDS["change_iou"],
        "change_head_precision": head["precision"] >= SCREEN_THRESHOLDS["change_precision"],
        "change_head_recall": head["recall"] >= SCREEN_THRESHOLDS["change_recall"],
        "executed_change_iou": executed["iou"] >= SCREEN_THRESHOLDS["change_iou"],
        "executed_change_precision": executed["precision"]
        >= SCREEN_THRESHOLDS["change_precision"],
        "executed_change_recall": executed["recall"]
        >= SCREEN_THRESHOLDS["change_recall"],
        "executed_change_ratio": executed_ratio is not None
        and SCREEN_THRESHOLDS["minimum_change_ratio"]
        <= executed_ratio
        <= SCREEN_THRESHOLDS["maximum_change_ratio"],
        "generate_budget": micro["student_action"]["observed_max_generate_fraction_per_phase"]
        <= SCREEN_THRESHOLDS["maximum_generate_fraction_per_phase"],
        "noop_change_head": micro["noop_change_head"]["predicted_fraction"]
        <= SCREEN_THRESHOLDS["maximum_noop_change_fraction"],
        "noop_executed_change": micro["noop_executed_change"]["predicted_fraction"]
        <= SCREEN_THRESHOLDS["maximum_noop_change_fraction"],
        "noop_soft_change": (
            micro["noop"]["gate_fraction"]["transport"]
            + micro["noop"]["gate_fraction"]["generate"]
        )
        <= SCREEN_THRESHOLDS["maximum_noop_soft_change_fraction"],
        "change_head_median_row_iou": row_iou["change_head"]["median"]
        >= SCREEN_THRESHOLDS["minimum_median_row_iou"],
        "executed_change_median_row_iou": row_iou["executed_change"]["median"]
        >= SCREEN_THRESHOLDS["minimum_median_row_iou"],
        "change_head_row_coverage": row_iou["change_head"]["rows_passing_threshold"]
        >= SCREEN_THRESHOLDS["minimum_rows_passing_iou"],
        "executed_change_row_coverage": row_iou["executed_change"][
            "rows_passing_threshold"
        ]
        >= SCREEN_THRESHOLDS["minimum_rows_passing_iou"],
    }
    joint_lora_checks = {
        **routing_checks,
        "conditional_tg_f1": micro["conditional_tg_head"]["f1"]
        >= SCREEN_THRESHOLDS["conditional_tg_f1"],
        "conditional_tg_nonvacuous": micro["conditional_tg_head"][
            "target_positive_count"
        ]
        > 0,
        "offset_top1_accuracy": micro["transport_offset"]["top1_accuracy"]
        >= SCREEN_THRESHOLDS["offset_top1_accuracy"],
        "offset_mae": micro["transport_offset"]["mae"]
        <= SCREEN_THRESHOLDS["maximum_offset_mae"],
        "noop_zero_offset_top1": micro["noop_offset"]["zero_top1_accuracy"]
        >= SCREEN_THRESHOLDS["minimum_noop_zero_offset_top1_accuracy"],
        "noop_offset_mae": micro["noop_offset"]["mae_from_zero"]
        <= SCREEN_THRESHOLDS["maximum_noop_offset_mae"],
    }
    routing_passed = all(routing_checks.values())
    joint_lora_passed = all(joint_lora_checks.values())
    routing_screen = {
        "thresholds": dict(SCREEN_THRESHOLDS),
        "checks": routing_checks,
        "passed": routing_passed,
        "checkpoint_eligible_for_routing_only": routing_passed,
    }
    joint_lora_screen = {
        "thresholds": dict(SCREEN_THRESHOLDS),
        "checks": joint_lora_checks,
        "passed": joint_lora_passed,
        "checkpoint_eligible_for_joint_lora": joint_lora_passed,
    }
    return {
        "row_count": len(rows),
        "micro": micro,
        "macro": macro,
        "row_iou": row_iou,
        "routing_screen": routing_screen,
        "joint_lora_screen": joint_lora_screen,
        # Historical consumers read ``screen``.  Keep it as the complete
        # joint-LoRA eligibility screen, not the weaker routing-only screen.
        "screen": dict(joint_lora_screen),
    }


def _atomic_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON file atomically with hard-link no-clobber."""

    if path.exists() or path.is_symlink():
        raise GroundedCheckpointEvaluationError(f"refusing to overwrite output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = legacy.canonical_json_bytes(value) + b"\n"
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # link(2) fails with EEXIST for files, directories, and symlinks.  It
        # therefore provides a real atomic no-overwrite publication primitive.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory_fd)
            except OSError as error:
                if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                    raise
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise GroundedCheckpointEvaluationError(
            f"refusing to overwrite output {path}"
        ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def runtime_conditioning_manifest(checkpoint: Path) -> dict[str, Any]:
    """Hash every base-checkpoint file that can alter evaluator conditioning."""

    requested: list[Path] = []
    for directory_name in ("tokenizer", "text_encoder"):
        directory = checkpoint / directory_name
        requested.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    requested.append(checkpoint / "vae" / "config.json")
    unique = sorted(set(requested), key=lambda path: str(path.relative_to(checkpoint)))
    if not unique:
        raise GroundedCheckpointEvaluationError(
            "base checkpoint conditioning manifest is empty"
        )
    entries = []
    for path in unique:
        if not path.is_file():
            raise GroundedCheckpointEvaluationError(
                f"base conditioning artifact is missing: {path}"
            )
        entries.append(
            {
                "relative_path": str(path.relative_to(checkpoint)),
                "size": int(path.stat().st_size),
                "sha256": legacy.file_sha256(path),
            }
        )
    value = {
        "scope": ["tokenizer/**", "text_encoder/**", "vae/config.json"],
        "scope_reason": (
            "exact files controlling token IDs, T5 embeddings, and latent normalization"
        ),
        "full_checkpoint_tree_recomputed": False,
        "files": entries,
    }
    value["manifest_digest"] = legacy.object_sha256(value)
    return value


def _read_checkpoint_contract(root_value: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    requested = Path(root_value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise GroundedCheckpointEvaluationError(
            "planner checkpoint must be an absolute non-symlink directory"
        )
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise GroundedCheckpointEvaluationError(
            f"planner checkpoint is unavailable: {error}"
        ) from error
    if not root.is_dir() or root.is_symlink():
        raise GroundedCheckpointEvaluationError("planner checkpoint must be a plain directory")
    try:
        receipt, config = student._load_resume(root, grounded.ARCHITECTURE_NAME)
    except (OSError, RuntimeError) as error:
        raise GroundedCheckpointEvaluationError(
            f"grounded planner checkpoint contract differs: {error}"
        ) from error
    immutable = receipt.get("immutable_contract")
    if (
        not isinstance(immutable, Mapping)
        or not isinstance(immutable.get("value"), Mapping)
        or immutable.get("digest") != legacy.object_sha256(immutable["value"])
    ):
        raise GroundedCheckpointEvaluationError("checkpoint immutable contract digest differs")
    immutable_value = immutable["value"]
    if (
        immutable_value.get("planner_architecture") != grounded.ARCHITECTURE_NAME
        or immutable_value.get("method") != student.GROUNDED_METHOD_NAME
        or immutable_value.get("planner_config") != config
        or immutable_value.get("target_used_by_student") is not False
        or immutable_value.get("target_used_by_training_teacher_only") is not True
    ):
        raise GroundedCheckpointEvaluationError(
            "checkpoint is not the target-isolated grounded SPT-v3 contract"
        )
    recorded_hashes = immutable_value.get("method_files_sha256")
    if not isinstance(recorded_hashes, Mapping) or recorded_hashes != student._method_hashes(
        grounded.ARCHITECTURE_NAME
    ):
        raise GroundedCheckpointEvaluationError(
            "runtime planner/training source differs from the checkpoint-bound source"
        )
    return root, receipt, config


def _normalize_teacher_config(value: Mapping[str, Any]) -> spt.PhaseTransportConfig:
    normalized = dict(value)
    for key in ("teacher_temporal_offsets", "teacher_spatial_offsets"):
        if isinstance(normalized.get(key), list):
            normalized[key] = tuple(normalized[key])
    try:
        config = spt.PhaseTransportConfig(**normalized)
        config.validate()
    except (TypeError, RuntimeError) as error:
        raise GroundedCheckpointEvaluationError(
            f"checkpoint teacher configuration differs: {error}"
        ) from error
    if (
        config.latent_channels != 64
        or config.teacher_temporal_offsets != grounded.TEMPORAL_CANDIDATES
        or config.teacher_spatial_offsets != grounded.SPATIAL_CANDIDATES
        or config.max_temporal_offset != 2.0
        or config.max_spatial_offset != 4.0
        or config.teacher_temperature != AUDITED_TEACHER_TEMPERATURE
        or config.teacher_generate_threshold
        != AUDITED_TEACHER_GENERATE_THRESHOLD
        or config.teacher_transport_margin != AUDITED_TEACHER_TRANSPORT_MARGIN
        or config.teacher_require_cycle is not True
        or config.teacher_allow_lossy_projection is not False
        or config.teacher_allow_unbounded_generate_ablation is not False
        or config.max_generate_fraction_per_phase != GENERATE_BUDGET
    ):
        raise GroundedCheckpointEvaluationError("checkpoint teacher is not the hardened oracle")
    return config


def _member_entries(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "row_index": int(member["row_index"]),
            "iid": str(member["iid"]),
            "identity_sha256": str(member["identity_sha256"]),
        }
        for member in value["members"]
    ]


def _validate_checkpoint_membership(
    receipt: Mapping[str, Any],
    selected_membership: Mapping[str, Any],
) -> list[dict[str, Any]]:
    members = _member_entries(selected_membership)
    if selected_membership.get("membership_digest") != TRUSTED_MEMBERSHIP_DIGEST:
        raise GroundedCheckpointEvaluationError(
            "membership is validly self-hashed but is not the audited trusted13 digest"
        )
    if len(members) != EXPECTED_TRUSTED_MEMBERS:
        raise GroundedCheckpointEvaluationError(
            f"selected membership has {len(members)} rows, expected {EXPECTED_TRUSTED_MEMBERS}"
        )
    immutable_value = receipt["immutable_contract"]["value"]
    training = immutable_value.get("training_membership")
    if not isinstance(training, Mapping):
        raise GroundedCheckpointEvaluationError("checkpoint training membership is unavailable")
    if (
        training.get("selection") != "teacher_trust_membership"
        or training.get("training_rows") != EXPECTED_TRUSTED_MEMBERS
        or training.get("selected_membership_digest")
        != selected_membership.get("membership_digest")
        or training.get("membership_sha256") != legacy.object_sha256(members)
        or training.get("members") != members
    ):
        raise GroundedCheckpointEvaluationError(
            "checkpoint was not trained on this exact trusted membership"
        )
    receipt_dataset = receipt.get("dataset")
    if not isinstance(receipt_dataset, Mapping) or receipt_dataset.get(
        "training_membership"
    ) != members:
        raise GroundedCheckpointEvaluationError(
            "checkpoint receipt membership differs from its immutable contract"
        )
    return members


def _assert_equal_across_ranks(value: Mapping[str, Any], *, label: str) -> None:
    import torch.distributed as dist

    gathered: list[Optional[dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(value))
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise GroundedCheckpointEvaluationError(f"{label} differs across ranks")


def _evaluation_method_hashes() -> dict[str, str]:
    paths = (
        HERE / "evaluate_grounded_checkpoint.py",
        HERE / "grounded_phase_planner.py",
        HERE / "phase_transport.py",
        HERE / "train_student.py",
        HERE / "audit_teacher_cohort.py",
        METHOD_ROOT / "train_lora.py",
        METHOD_ROOT / "motion_residual.py",
        HERE / "scripts" / "auh_evaluate_grounded_checkpoint.sbatch",
    )
    return {str(path.relative_to(METHOD_ROOT)): legacy.file_sha256(path) for path in paths}


def _committed_ranking_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replace large score runs by a canonical audit commitment for publication."""

    result = dict(value)
    try:
        runs = result.pop("raw_score_runs_descending")
    except KeyError as error:
        raise GroundedCheckpointEvaluationError(
            "ranking metrics lack raw score runs before receipt publication"
        ) from error
    decoded = _decode_score_runs(runs)
    run_count = len(decoded)
    cell_count = sum(positive + negative for _, positive, negative in decoded)
    if (
        int(result.get("raw_score_run_count", -1)) != run_count
        or int(result.get("cell_count", -1)) != cell_count
    ):
        raise GroundedCheckpointEvaluationError(
            "ranking score-run commitment counts differ from derived metrics"
        )
    result["raw_score_runs_embedded"] = False
    result["raw_score_runs_commitment"] = {
        "schema": "descending-python-float-hex-positive-negative-runs-v1",
        "sha256": legacy.object_sha256(runs),
        "run_count": run_count,
        "cell_count": cell_count,
        "availability": "aggregation_intermediate_not_embedded_in_receipt",
    }
    return result


def _receipt_publication_views(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create bounded receipt views after all exact score-run aggregation."""

    ranking_labels = ("change_ranking", "action_noop_delta_ranking")
    published_rows: list[dict[str, Any]] = []
    for row in rows:
        published = dict(row)
        for label in ranking_labels:
            published[label] = _committed_ranking_metrics(row[label])
        published_rows.append(published)
    published_aggregate = dict(aggregate)
    published_micro = dict(aggregate["micro"])
    for label in ranking_labels:
        published_micro[label] = _committed_ranking_metrics(
            aggregate["micro"][label]
        )
    published_aggregate["micro"] = published_micro
    return published_rows, published_aggregate


def build_evaluation_receipt(
    *,
    source_identity: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    membership_identity: Mapping[str, Any],
    distributed_identity: Mapping[str, Any],
    evaluation_inputs: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    published_rows, published_aggregate = _receipt_publication_views(
        rows,
        aggregate,
    )
    value: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA,
        "read_only_evaluation": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "student_semantic_inputs": ["source_video", "edit_instruction"],
        "student_target_argument_exists": False,
        "paired_target_use": "hardened_oracle_teacher_only",
        "external_mask_track_pose_flow": False,
        "metric_contract": {
            "change_prediction": "sigmoid(change_logits)>=0.5",
            "executed_change_prediction": "postbudget_(1-P)>=0.5",
            "hard_gate_argmax_change_prediction": "argmax(postbudget_PTG)!=preserve",
            "conditional_tg_domain": "teacher_change_cells_only",
            "conditional_generate_prediction": "sigmoid(novelty_logits)>=0.5",
            "teacher_cardinality_topk_change": {
                "purpose": "oracle-cardinality saliency-ranking diagnostic only",
                "student_score": RANKING_SCORE_DOMAIN,
                "paired_target_use": "integer_change_cardinality_per_sample_phase_only",
                "selection": "top_k_spatial_cells_independently_per_sample_phase",
                "tie_policy": TOPK_TIE_POLICY,
                "deployable_inference_metric": False,
                "eligibility_screen_input": False,
            },
            "change_ranking": {
                "score": RANKING_SCORE_DOMAIN,
                "truth": "teacher_change_cell",
                "pr_auc": "stepwise_average_precision_after_exact_score_tie_group",
                "best_f1_threshold": "highest_raw_score_threshold_wins_equal_f1",
            },
            "action_noop_delta_ranking": {
                "score": ACTION_NOOP_DELTA_SCORE_DOMAIN,
                "truth": "teacher_change_cell",
                "purpose": "measure_instruction-conditioned_change_separation",
                "pr_auc": "stepwise_average_precision_after_exact_score_tie_group",
            },
            "ranking_raw_sufficient_statistics": {
                "format": "descending_[python_float_hex,positive_count,negative_count]_runs",
                "tie_policy": RANKING_TIE_POLICY,
                "global_aggregation": "merge_equal_scores_across_rows_before_metric_update",
                "receipt_storage": (
                    "strip_runs_after_exact_aggregation_and_publish_counts_plus_"
                    "canonical_sha256_commitment"
                ),
            },
            "phase_mass": {
                "prediction": "spatial_mean_sigmoid_current_change_logits_per_sample_phase",
                "target": "spatial_mean_teacher_change_per_sample_phase",
                "dedicated_learned_mass_head_exists": False,
                "metrics": ["mae", "rmse", "pearson_correlation_when_nonconstant"],
                "global_aggregation": "sum_additive_raw_sufficient_statistics_then_derive",
            },
            "offset_domain": "teacher_transport_cells_only",
            "offset_top1": "argmax_exact_ordered_125_candidate_logits",
            "offset_mae": "mean_absolute_dt_dy_dx_on_teacher_transport_cells",
            "generate_max": "max_over_actual_postbudget_gate_G_spatial_mean_per_row_phase",
            "noop_target": "exact_preserve_and_zero_offset",
            "micro": "sum_raw_sufficient_statistics_then_derive",
            "macro": "unweighted_mean_of_row_metrics",
            "nullable_macro": "mean_available_rows_plus_available_and_unavailable_counts",
            "empty_set_rule": "both_empty_scores_one_otherwise_missing_side_scores_zero",
            "routing_screen": (
                "change_and_executed_routing_noop_and_generate_checks_without_"
                "conditional_tg_or_offset_checks"
            ),
            "joint_lora_screen": "routing_screen_plus_conditional_tg_and_offset_checks",
            "legacy_screen_alias": "full_joint_lora_screen",
        },
        "source": dict(source_identity),
        "evaluated_checkpoint": dict(checkpoint_identity),
        "dataset": dict(dataset_identity),
        "trusted_membership": dict(membership_identity),
        "distributed": dict(distributed_identity),
        "evaluation_inputs": dict(evaluation_inputs),
        "method_files_sha256": _evaluation_method_hashes(),
        "aggregate": published_aggregate,
        "rows": published_rows,
        "experimental_evaluation": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    value["evaluation_digest"] = legacy.object_sha256(value)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    rank_local_runtime_cache = student.configure_rank_local_runtime_cache()
    if not rank_local_runtime_cache:
        raise GroundedCheckpointEvaluationError(
            "formal four-rank evaluation requires rank-local runtime caches"
        )
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        base_checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
        distributed = legacy.distributed_contract()
    except legacy.TrainingContractError as error:
        raise GroundedCheckpointEvaluationError(str(error)) from error
    if distributed.world_size != EXPECTED_WORLD_SIZE:
        raise GroundedCheckpointEvaluationError(
            f"checkpoint-wide evaluation requires exactly {EXPECTED_WORLD_SIZE} ranks"
        )
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.training.data import SYSTEM_PROMPTS

    try:
        device, backend = legacy.initialise_distributed(distributed)
    except legacy.TrainingContractError as error:
        raise GroundedCheckpointEvaluationError(str(error)) from error
    from bernini.parallel import init_parallel_state

    init_parallel_state(ulysses_size=EVALUATOR_ULYSSES_SIZE)
    conditioning_holder: list[Optional[dict[str, Any]]] = [
        runtime_conditioning_manifest(base_checkpoint)
        if distributed.rank == 0
        else None
    ]
    dist.broadcast_object_list(conditioning_holder, src=0)
    conditioning_manifest = conditioning_holder[0]
    if not isinstance(conditioning_manifest, dict):
        raise GroundedCheckpointEvaluationError(
            "rank zero did not provide the base conditioning manifest"
        )
    planner_checkpoint, checkpoint_receipt, raw_planner_config = _read_checkpoint_contract(
        args.planner_checkpoint
    )
    artifact_hashes = {
        name: legacy.file_sha256(planner_checkpoint / name)
        for name in ("planner.safetensors", "planner_config.json", "receipt.json")
    }
    _assert_equal_across_ranks(
        {
            "planner_checkpoint": str(planner_checkpoint),
            "receipt_digest": checkpoint_receipt["receipt_digest"],
            "global_step": checkpoint_receipt["global_step"],
            "artifact_sha256": artifact_hashes,
        },
        label="evaluated checkpoint",
    )

    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = legacy.validate_preprocessed_dataset_summary(
        args.dataset_summary,
        dataset,
        allow_incomplete=args.allow_incomplete_dataset,
    )
    try:
        selected_membership, selected_rows = cohort.load_selected_membership(
            args.selected_membership,
            dataset=dataset,
            dataset_summary=dataset_summary,
            require_sufficient=True,
        )
    except (OSError, RuntimeError) as error:
        raise GroundedCheckpointEvaluationError(
            f"trusted membership differs: {error}"
        ) from error
    membership_path = Path(args.selected_membership).expanduser().resolve(strict=True)
    membership_file_sha256 = legacy.file_sha256(membership_path)
    if membership_file_sha256 != TRUSTED_MEMBERSHIP_FILE_SHA256:
        raise GroundedCheckpointEvaluationError(
            "selected membership file is not the byte-audited trusted13 artifact"
        )
    members = _validate_checkpoint_membership(checkpoint_receipt, selected_membership)
    if tuple(member["row_index"] for member in members) != selected_rows:
        raise GroundedCheckpointEvaluationError("trusted membership order differs")
    immutable_value = checkpoint_receipt["immutable_contract"]["value"]
    if (
        immutable_value.get("bernini_commit") != bernini_revision
        or immutable_value.get("veomni_commit") != veomni_revision
        or immutable_value.get("checkpoint_tree_sha256")
        != args.expected_checkpoint_tree_sha256
        or immutable_value.get("dataset_signature") != dataset.signature
        or immutable_value.get("dataset_summary_sha256") != dataset_summary["sha256"]
        or immutable_value.get("dataset_index_sha256") != dataset_summary["index_sha256"]
    ):
        raise GroundedCheckpointEvaluationError(
            "runtime source/checkpoint/dataset differs from training receipt"
        )
    _assert_equal_across_ranks(
        {
            "dataset_signature": dataset.signature,
            "dataset_summary_sha256": dataset_summary["sha256"],
            "membership_digest": selected_membership["membership_digest"],
            "members": members,
        },
        label="dataset and trusted membership",
    )

    try:
        planner_config = grounded.GroundedPhasePlannerConfig(**raw_planner_config)
        planner_config.validate()
    except (TypeError, RuntimeError) as error:
        raise GroundedCheckpointEvaluationError(
            f"grounded planner configuration differs: {error}"
        ) from error
    teacher_config = _normalize_teacher_config(immutable_value["teacher_config"])
    if immutable_value.get("teacher_feature_channels") != TEACHER_FEATURE_CHANNELS:
        raise GroundedCheckpointEvaluationError("teacher feature-channel contract differs")
    planner = grounded.GroundedPhasePlanner(planner_config).to(device)
    saved_state = load_file(
        str(planner_checkpoint / "planner.safetensors"), device=str(device)
    )
    if set(saved_state) != set(planner.state_dict()):
        raise GroundedCheckpointEvaluationError("planner checkpoint state-key scope differs")
    planner.load_state_dict(saved_state, strict=True)
    planner.requires_grad_(False)
    planner.eval()
    parameter_names = [name for name, _ in planner.named_parameters()]
    if (
        checkpoint_receipt.get("planner", {}).get("parameter_names") != parameter_names
        or checkpoint_receipt.get("planner", {}).get("parameter_names_sha256")
        != legacy.object_sha256(parameter_names)
    ):
        raise GroundedCheckpointEvaluationError("planner parameter-name receipt differs")

    renderer_config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(base_checkpoint),
    )
    renderer_config.dtype = torch.bfloat16
    renderer = BerniniRendererModel(renderer_config)
    renderer.requires_grad_(False)
    renderer.eval()
    renderer.t5_text_encoder.to(device)
    renderer.t5_text_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    mv2v_system_prompt = SYSTEM_PROMPTS.get("mv2v")
    if not isinstance(mv2v_system_prompt, str) or not mv2v_system_prompt:
        raise GroundedCheckpointEvaluationError(
            "runtime Bernini mv2v system prompt is unavailable"
        )
    vae_mean, vae_std, z_dim = legacy._vae_statistics(base_checkpoint)

    assigned_members = members[distributed.rank :: distributed.world_size]
    local_reports: list[dict[str, Any]] = []
    with torch.inference_mode():
        for ordinal, member in enumerate(assigned_members):
            row_index = int(member["row_index"])
            raw_row = dataset[row_index]
            legacy.seed_same_sample(
                legacy.step_seed(args.seed, distributed.rank + ordinal * distributed.world_size, row_index)
            )
            source, paired_target = student._clean_pair(
                raw_row, vae_mean, vae_std, z_dim, device
            )
            teacher_plan = paired_teacher_plan(source, paired_target, teacher_config)
            del paired_target
            action_batch, noop_batch = text_only_token_batches(
                raw_row=raw_row,
                tokenizer=tokenizer,
                noop_instruction=args.noop_instruction,
                prompt_cleaner=prompt_clean,
                system_prompt=mv2v_system_prompt,
            )
            action_tokens = student._embed_instruction(renderer, action_batch, device)
            noop_tokens = student._embed_instruction(renderer, noop_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                action_plan, noop_plan = student_plans(
                    planner, source, action_tokens, noop_tokens
                )
            local_reports.append(
                evaluate_row_plans(
                    row_index=row_index,
                    iid=str(member["iid"]),
                    identity_sha256=str(member["identity_sha256"]),
                    source=source,
                    action=action_plan,
                    teacher=teacher_plan,
                    noop=noop_plan,
                )
            )
            del source, action_tokens, noop_tokens, teacher_plan, action_plan, noop_plan

    gathered: list[Optional[list[dict[str, Any]]]] = [None] * distributed.world_size
    dist.all_gather_object(gathered, local_reports)
    reports = [
        report
        for rank_reports in gathered
        if rank_reports is not None
        for report in rank_reports
    ]
    reports = validate_complete_reports(reports, members)
    aggregate = aggregate_reports(reports)
    receipt = build_evaluation_receipt(
        source_identity={
            "declared_method_source_revision": args.method_source_revision.lower(),
            "declared_method_source_archive_sha256": args.method_source_archive_sha256,
            "revision_and_archive_declaration_verified_by_evaluator": False,
            "runtime_files_independently_hashed": True,
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
        },
        checkpoint_identity={
            "path": str(planner_checkpoint),
            "global_step": int(checkpoint_receipt["global_step"]),
            "training_receipt_schema": checkpoint_receipt["schema_version"],
            "training_receipt_digest": checkpoint_receipt["receipt_digest"],
            "immutable_contract_digest": checkpoint_receipt["immutable_contract"]["digest"],
            "artifact_sha256": artifact_hashes,
            "planner_config": asdict(planner_config),
        },
        dataset_identity={
            "path": str(dataset.root),
            "signature": dataset.signature,
            "summary": dict(dataset_summary),
            "base_checkpoint_path": str(base_checkpoint),
            "audited_base_checkpoint_tree_identifier": args.expected_checkpoint_tree_sha256,
            "audited_tree_identifier_recomputed_by_evaluator": False,
            "runtime_conditioning_manifest": conditioning_manifest,
        },
        membership_identity={
            "path": str(membership_path),
            "file_sha256": membership_file_sha256,
            "schema_version": selected_membership["schema_version"],
            "membership_digest": selected_membership["membership_digest"],
            "audited_membership_digest": TRUSTED_MEMBERSHIP_DIGEST,
            "audited_membership_file_sha256": TRUSTED_MEMBERSHIP_FILE_SHA256,
            "source_teacher_audit_digest": TRUSTED_AUDIT_DIGEST,
            "source_teacher_audit_file_sha256": TRUSTED_AUDIT_FILE_SHA256,
            "selected_count": len(members),
            "ordered_members": members,
            "exactly_once": True,
        },
        distributed_identity={
            "world_size": distributed.world_size,
            "data_parallel_size": distributed.world_size,
            "ulysses_size": EVALUATOR_ULYSSES_SIZE,
            "backend": backend,
            "rank_local_runtime_cache": rank_local_runtime_cache,
            "same_checkpoint_all_ranks": True,
            "rank_assignments": {
                str(rank): [
                    int(member["row_index"])
                    for member in members[rank :: distributed.world_size]
                ]
                for rank in range(distributed.world_size)
            },
            "all_members_exactly_once": True,
        },
        evaluation_inputs={
            "seed": int(args.seed),
            "noop_instruction": args.noop_instruction,
            "noop_instruction_utf8_sha256": hashlib.sha256(
                args.noop_instruction.encode("utf-8")
            ).hexdigest(),
            "mv2v_system_prompt": mv2v_system_prompt,
            "mv2v_system_prompt_utf8_sha256": hashlib.sha256(
                mv2v_system_prompt.encode("utf-8")
            ).hexdigest(),
            "prompt_cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenization": (
                "official_encode_renderer_messages_text_only_unpadded_untruncated"
            ),
            "teacher_config": asdict(teacher_config),
            "teacher_feature_channels": TEACHER_FEATURE_CHANNELS,
            "change_threshold": CHANGE_THRESHOLD,
            "novelty_threshold": NOVELTY_THRESHOLD,
        },
        rows=reports,
        aggregate=aggregate,
    )
    if distributed.rank == 0:
        output = Path(args.output).expanduser()
        _atomic_json_no_overwrite(output, receipt)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "evaluation_digest": receipt["evaluation_digest"],
                    "global_step": checkpoint_receipt["global_step"],
                    "rows": len(reports),
                    "screen_passed": receipt["aggregate"]["screen"]["passed"],
                    "micro": receipt["aggregate"]["micro"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
