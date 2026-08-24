#!/usr/bin/env python3
"""Contracts for strict checkpoint-wide grounded SPT-v3 evaluation."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SPT_ROOT = METHOD_ROOT / "spt_v2"
for root in (METHOD_ROOT, SPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_grounded_checkpoint as evaluation  # noqa: E402
import grounded_phase_planner as grounded  # noqa: E402
import phase_transport as spt  # noqa: E402


def _argv(output: str = "/tmp/grounded-evaluation.json") -> list[str]:
    return [
        "--bernini-root", "/bernini",
        "--veomni-root", "/veomni",
        "--checkpoint", "/base",
        "--planner-checkpoint", "/planner",
        "--preprocessed-parquet-dir", "/data",
        "--dataset-summary", "/data/summary.json",
        "--selected-membership", "/data/selected.json",
        "--output", output,
        "--method-source-revision", "1" * 40,
        "--method-source-archive-sha256", "2" * 64,
    ]


def _member(index: int) -> dict:
    return {
        "row_index": index,
        "iid": f"iid-{index:03d}",
        "identity_sha256": f"{index + 1:064x}",
    }


def _perfect_ranking(
    positive_count: int,
    negative_count: int,
    *,
    score_domain: str = evaluation.RANKING_SCORE_DOMAIN,
) -> dict:
    runs = []
    if positive_count:
        runs.append([(1.0).hex(), positive_count, 0])
    if negative_count:
        runs.append([(0.0).hex(), 0, negative_count])
    return evaluation.ranking_metrics_from_score_runs(
        runs,
        score_domain=score_domain,
    )


def _minimal_row(index: int, *, tp: int, fp: int, fn: int, tn: int, gmax: float) -> dict:
    counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    binary = evaluation.metrics_from_counts(counts)
    cell_count = tp + fp + fn + tn
    probability_sum = {
        "preserve": cell_count * (1.0 - gmax),
        "transport": 0.0,
        "generate": cell_count * gmax,
    }
    positive_count = tp + fn
    negative_count = fp + tn
    predicted_mass = (tp + fp) / cell_count
    teacher_mass = positive_count / cell_count
    mass_error = predicted_mass - teacher_mass
    phase_mass = evaluation.phase_mass_metrics_from_sufficient_statistics(
        {
            "count": 1,
            "sum_predicted": predicted_mass,
            "sum_teacher": teacher_mass,
            "sum_absolute_error": abs(mass_error),
            "sum_squared_error": mass_error**2,
            "sum_predicted_squared": predicted_mass**2,
            "sum_teacher_squared": teacher_mass**2,
            "sum_product": predicted_mass * teacher_mass,
        }
    )
    return {
        **_member(index),
        "cell_count": cell_count,
        "change_head": binary,
        "executed_change": binary,
        "hard_gate_argmax_change": binary,
        "conditional_tg_head": binary,
        "teacher_cardinality_topk_change": evaluation.metrics_from_counts(
            {
                "tp": positive_count,
                "fp": 0,
                "fn": 0,
                "tn": negative_count,
            }
        ),
        "change_ranking": _perfect_ranking(positive_count, negative_count),
        "action_noop_delta_ranking": _perfect_ranking(
            positive_count,
            negative_count,
            score_domain=evaluation.ACTION_NOOP_DELTA_SCORE_DOMAIN,
        ),
        "phase_mass": phase_mass,
        "transport_offset": {
            "top1_correct": tp,
            "transport_cells": tp + fn,
            "top1_accuracy": tp / (tp + fn),
            "absolute_axis_error_sum": float(fn * 3),
            "axis_count": (tp + fn) * 3,
            "mae": fn / (tp + fn),
        },
        "student_action": {
            "gate_probability_sum": probability_sum,
            "gate_fraction": {
                name: value / cell_count for name, value in probability_sum.items()
            },
            "hard_gate_correct": tp + tn,
            "hard_gate_accuracy": (tp + tn) / cell_count,
            "observed_max_generate_fraction_per_phase": gmax,
        },
        "noop": {
            "change_head": evaluation.metrics_from_counts(
                {"tp": 0, "fp": 0, "fn": 0, "tn": cell_count}
            ),
            "executed_change": evaluation.metrics_from_counts(
                {"tp": 0, "fp": 0, "fn": 0, "tn": cell_count}
            ),
            "hard_gate_argmax_change": evaluation.metrics_from_counts(
                {"tp": 0, "fp": 0, "fn": 0, "tn": cell_count}
            ),
            "gate_probability_sum": {
                "preserve": float(cell_count),
                "transport": 0.0,
                "generate": 0.0,
            },
            "gate_fraction": {"preserve": 1.0, "transport": 0.0, "generate": 0.0},
            "observed_max_generate_fraction_per_phase": 0.0,
            "offset": {
                "zero_top1_correct": cell_count,
                "cells": cell_count,
                "zero_top1_accuracy": 1.0,
                "absolute_axis_error_sum": 0.0,
                "axis_count": cell_count * 3,
                "mae_from_zero": 0.0,
            },
        },
    }


class GroundedCheckpointEvaluationPureTests(unittest.TestCase):
    def test_cli_and_constants_lock_full_trusted_four_rank_evaluation(self) -> None:
        args = evaluation.build_parser().parse_args(_argv())
        evaluation.validate_cli(args)
        self.assertEqual(evaluation.EXPECTED_TRUSTED_MEMBERS, 13)
        self.assertEqual(evaluation.EXPECTED_WORLD_SIZE, 4)
        self.assertEqual(evaluation.EVALUATOR_ULYSSES_SIZE, 1)
        self.assertEqual(
            evaluation.TRUSTED_MEMBERSHIP_DIGEST,
            "2ce012cd25debd36a357fe041949b5d09ee8347ac7e46f8e2fdfa02f048ec507",
        )
        self.assertFalse(hasattr(args, "prefix_rows"))
        self.assertFalse(hasattr(args, "row_range"))
        self.assertFalse(hasattr(args, "maximum_generate_fraction_per_phase"))

    def test_student_api_cannot_accept_target_and_main_deletes_target_first(self) -> None:
        self.assertEqual(
            list(inspect.signature(evaluation.student_plans).parameters),
            [
                "planner",
                "source",
                "action_instruction_tokens",
                "noop_instruction_tokens",
            ],
        )
        student_source = inspect.getsource(evaluation.student_plans)
        self.assertNotIn("target", student_source)
        main_source = inspect.getsource(evaluation.main)
        self.assertIn("del paired_target", main_source)
        self.assertLess(
            main_source.index("del paired_target"),
            main_source.index("student_plans("),
        )
        self.assertIn("paired_teacher_plan(source, paired_target", main_source)
        self.assertNotIn("_prepare_paired_batches", main_source)

    def test_teacher_config_is_pinned_to_the_membership_audit_oracle(self) -> None:
        value = evaluation.asdict(spt.PhaseTransportConfig())
        normalized = evaluation._normalize_teacher_config(value)
        self.assertEqual(normalized.teacher_temperature, 0.08)
        value["teacher_temperature"] = 0.081
        with self.assertRaisesRegex(
            evaluation.GroundedCheckpointEvaluationError, "hardened oracle"
        ):
            evaluation._normalize_teacher_config(value)

    def test_exact_13_coverage_rejects_missing_duplicate_or_wrong_identity(self) -> None:
        members = [_member(index) for index in range(13)]
        rows = [
            {
                **member,
                "payload": index,
            }
            for index, member in enumerate(reversed(members))
        ]
        ordered = evaluation.validate_complete_reports(rows, members)
        self.assertEqual([row["row_index"] for row in ordered], list(range(13)))
        with self.assertRaisesRegex(
            evaluation.GroundedCheckpointEvaluationError, "exactly once"
        ):
            evaluation.validate_complete_reports(rows[:-1], members)
        duplicated = list(rows)
        duplicated[-1] = dict(duplicated[0])
        with self.assertRaisesRegex(
            evaluation.GroundedCheckpointEvaluationError, "exactly once"
        ):
            evaluation.validate_complete_reports(duplicated, members)
        wrong = [dict(row) for row in rows]
        wrong[0]["iid"] = "wrong"
        with self.assertRaisesRegex(
            evaluation.GroundedCheckpointEvaluationError, "exactly once"
        ):
            evaluation.validate_complete_reports(wrong, members)

    def test_micro_uses_summed_raw_counts_macro_is_row_mean_and_gmax_is_real_max(self) -> None:
        rows = [
            _minimal_row(
                index,
                tp=1 if index == 0 else 9,
                fp=1,
                fn=0,
                tn=8 if index == 0 else 0,
                gmax=0.01 + index * 0.001,
            )
            for index in range(13)
        ]
        aggregate = evaluation.aggregate_reports(rows)
        micro = aggregate["micro"]["change_head"]
        self.assertEqual(micro["raw_counts"], {"tp": 109, "fp": 13, "fn": 0, "tn": 8})
        self.assertAlmostEqual(micro["precision"], 109 / 122)
        macro_precision = (0.5 + 12 * 0.9) / 13
        self.assertAlmostEqual(
            aggregate["macro"]["change_head"]["precision"], macro_precision
        )
        self.assertNotAlmostEqual(micro["precision"], macro_precision)
        self.assertAlmostEqual(
            aggregate["micro"]["student_action"][
                "observed_max_generate_fraction_per_phase"
            ],
            0.022,
        )

    def test_score_runs_merge_cross_row_ties_before_exact_global_pr_auc(self) -> None:
        rows = [
            _minimal_row(index, tp=1, fp=0, fn=0, tn=1, gmax=0.01)
            for index in range(2)
        ]
        rows[0]["change_ranking"] = evaluation.ranking_metrics_from_score_runs(
            [[(1.0).hex(), 1, 0], [(0.0).hex(), 0, 1]]
        )
        rows[1]["change_ranking"] = evaluation.ranking_metrics_from_score_runs(
            [[(1.0).hex(), 0, 1], [(0.0).hex(), 1, 0]]
        )
        aggregate = evaluation.aggregate_reports(rows)
        self.assertAlmostEqual(aggregate["macro"]["change_ranking"]["pr_auc"], 0.75)
        self.assertAlmostEqual(aggregate["micro"]["change_ranking"]["pr_auc"], 0.5)
        self.assertAlmostEqual(
            aggregate["micro"]["change_ranking"]["best_f1"]["f1"],
            2 / 3,
        )
        self.assertEqual(
            aggregate["micro"]["change_ranking"]["best_f1"][
                "threshold_score_hex"
            ],
            (0.0).hex(),
        )
        self.assertEqual(
            aggregate["micro"]["change_ranking"]["raw_score_runs_descending"],
            [[(1.0).hex(), 1, 1], [(0.0).hex(), 1, 1]],
        )

    def test_nullable_noop_change_ratio_is_reported_instead_of_aborting(self) -> None:
        rows = [
            _minimal_row(index, tp=5, fp=0, fn=0, tn=5, gmax=0.01)
            for index in range(13)
        ]
        for row in rows:
            row["noop"]["change_head"] = evaluation.metrics_from_counts(
                {"tp": 0, "fp": 1, "fn": 0, "tn": 9}
            )
        aggregate = evaluation.aggregate_reports(rows)
        macro = aggregate["macro"]["noop_change_head"]
        self.assertIsNone(macro["change_ratio"])
        self.assertEqual(macro["change_ratio_available_count"], 0)
        self.assertEqual(macro["change_ratio_unavailable_count"], 13)
        self.assertIsNone(aggregate["micro"]["noop_change_head"]["change_ratio"])

    def test_routing_screen_can_pass_while_joint_lora_screen_fails(self) -> None:
        rows = [
            _minimal_row(index, tp=5, fp=0, fn=0, tn=5, gmax=0.01)
            for index in range(13)
        ]
        for row in rows:
            row["conditional_tg_head"] = evaluation.metrics_from_counts(
                {"tp": 0, "fp": 0, "fn": 5, "tn": 5}
            )
            row["transport_offset"] = {
                "top1_correct": 0,
                "transport_cells": 5,
                "top1_accuracy": 0.0,
                "absolute_axis_error_sum": 30.0,
                "axis_count": 15,
                "mae": 2.0,
            }
        aggregate = evaluation.aggregate_reports(rows)
        self.assertTrue(aggregate["routing_screen"]["passed"])
        self.assertFalse(aggregate["joint_lora_screen"]["passed"])
        self.assertNotIn(
            "conditional_tg_f1", aggregate["routing_screen"]["checks"]
        )
        self.assertNotIn("offset_mae", aggregate["routing_screen"]["checks"])
        self.assertEqual(aggregate["screen"], aggregate["joint_lora_screen"])

    def test_atomic_json_refuses_to_overwrite_even_an_empty_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
            evaluation._atomic_json_no_overwrite(output, {"first": True})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"first": True})
            first_bytes = output.read_bytes()
            with self.assertRaisesRegex(
                evaluation.GroundedCheckpointEvaluationError, "overwrite"
            ):
                evaluation._atomic_json_no_overwrite(output, {"second": True})
            self.assertEqual(output.read_bytes(), first_bytes)

            empty = Path(directory) / "empty.json"
            empty.touch()
            with self.assertRaisesRegex(
                evaluation.GroundedCheckpointEvaluationError, "overwrite"
            ):
                evaluation._atomic_json_no_overwrite(empty, {"second": True})
            self.assertEqual(empty.read_bytes(), b"")

    def test_eligibility_fails_when_head_passes_but_postbudget_execution_fails(self) -> None:
        rows = [
            _minimal_row(index, tp=5, fp=0, fn=0, tn=5, gmax=0.01)
            for index in range(13)
        ]
        for row in rows:
            row["executed_change"] = evaluation.metrics_from_counts(
                {"tp": 0, "fp": 0, "fn": 5, "tn": 5}
            )
        aggregate = evaluation.aggregate_reports(rows)
        self.assertTrue(aggregate["screen"]["checks"]["change_head_iou"])
        self.assertFalse(aggregate["screen"]["checks"]["executed_change_iou"])
        self.assertFalse(aggregate["screen"]["passed"])

    def test_eligibility_rejects_diffuse_noop_mass_and_poor_row_coverage(self) -> None:
        rows = [
            _minimal_row(index, tp=5, fp=0, fn=0, tn=5, gmax=0.01)
            for index in range(13)
        ]
        for row in rows:
            cells = row["cell_count"]
            row["noop"]["gate_probability_sum"] = {
                "preserve": 0.90 * cells,
                "transport": 0.07 * cells,
                "generate": 0.03 * cells,
            }
            row["noop"]["gate_fraction"] = {
                "preserve": 0.90,
                "transport": 0.07,
                "generate": 0.03,
            }
        aggregate = evaluation.aggregate_reports(rows)
        self.assertFalse(aggregate["screen"]["checks"]["noop_soft_change"])

        rows = [
            _minimal_row(
                index,
                tp=5 if index < 9 else 1,
                fp=0 if index < 9 else 4,
                fn=0 if index < 9 else 4,
                tn=5 if index < 9 else 1,
                gmax=0.01,
            )
            for index in range(13)
        ]
        aggregate = evaluation.aggregate_reports(rows)
        self.assertEqual(
            aggregate["row_iou"]["executed_change"]["rows_passing_threshold"], 9
        )
        self.assertFalse(
            aggregate["screen"]["checks"]["executed_change_row_coverage"]
        )
        self.assertFalse(aggregate["screen"]["passed"])

    def test_receipt_is_hash_bound_and_read_only(self) -> None:
        rows = [
            _minimal_row(index, tp=1, fp=0, fn=0, tn=9, gmax=0.01)
            for index in range(13)
        ]
        aggregate = evaluation.aggregate_reports(rows)
        receipt = evaluation.build_evaluation_receipt(
            source_identity={"revision": "1" * 40},
            checkpoint_identity={"planner_sha256": "2" * 64},
            dataset_identity={"signature": "dataset"},
            membership_identity={"membership_digest": "3" * 64},
            distributed_identity={"world_size": 4},
            evaluation_inputs={"seed": 1, "noop_instruction": "copy"},
            rows=rows,
            aggregate=aggregate,
        )
        declared = receipt.pop("evaluation_digest")
        self.assertEqual(evaluation.legacy.object_sha256(receipt), declared)
        self.assertTrue(receipt["read_only_evaluation"])
        self.assertEqual(receipt["optimizer_steps"], 0)
        self.assertFalse(receipt["student_target_argument_exists"])
        self.assertEqual(len(receipt["rows"]), 13)
        self.assertFalse(
            receipt["metric_contract"]["teacher_cardinality_topk_change"][
                "deployable_inference_metric"
            ]
        )
        self.assertFalse(
            receipt["metric_contract"]["phase_mass"][
                "dedicated_learned_mass_head_exists"
            ]
        )
        self.assertEqual(
            receipt["metric_contract"]["legacy_screen_alias"],
            "full_joint_lora_screen",
        )
        def _contains_key(value, key: str) -> bool:
            if isinstance(value, dict):
                return key in value or any(
                    _contains_key(child, key) for child in value.values()
                )
            if isinstance(value, list):
                return any(_contains_key(child, key) for child in value)
            return False

        self.assertFalse(_contains_key(receipt, "raw_score_runs_descending"))
        self.assertTrue(
            receipt["rows"][0]["change_ranking"]["raw_score_runs_commitment"][
                "sha256"
            ]
        )
        self.assertFalse(
            receipt["aggregate"]["micro"]["action_noop_delta_ranking"][
                "raw_score_runs_embedded"
            ]
        )

    def test_main_contains_no_optimizer_or_training_step(self) -> None:
        source = inspect.getsource(evaluation.main)
        self.assertIn("distributed.world_size != EXPECTED_WORLD_SIZE", source)
        self.assertIn("members[distributed.rank :: distributed.world_size]", source)
        self.assertIn("configure_rank_local_runtime_cache", source)
        self.assertIn("torch.inference_mode", source)
        self.assertIn('"micro": receipt["aggregate"]["micro"]', source)
        self.assertNotIn('"micro": aggregate["micro"]', source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step", source)


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class GroundedCheckpointEvaluationTensorTests(unittest.TestCase):
    def test_raw_cell_pr_auc_and_best_f1_are_exact_and_tie_aware(self) -> None:
        scores = torch.tensor([0.9, 0.9, 0.1], dtype=torch.float32)
        truth = torch.tensor([True, False, True])
        metrics = evaluation.ranking_metrics_from_score_runs(
            evaluation.raw_score_runs(scores, truth)
        )
        self.assertAlmostEqual(metrics["pr_auc"], 7 / 12)
        self.assertAlmostEqual(metrics["best_f1"]["f1"], 0.8)
        self.assertEqual(
            metrics["best_f1"]["threshold_score_hex"],
            float(torch.tensor(0.1, dtype=torch.float32).item()).hex(),
        )
        self.assertEqual(metrics["tie_policy"], evaluation.RANKING_TIE_POLICY)

    def test_teacher_cardinality_topk_has_deterministic_canonical_ties(self) -> None:
        logits = torch.zeros(1, 1, 1, 1, 4)
        truth = torch.tensor([[[[True, False, False, True]]]])
        first = evaluation.teacher_phase_cardinality_topk_statistics(logits, truth)
        second = evaluation.teacher_phase_cardinality_topk_statistics(logits, truth)
        self.assertEqual(first, second)
        self.assertEqual(
            first["raw_counts"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
        )
        self.assertAlmostEqual(first["iou"], 1 / 3)
        self.assertEqual(first["tie_policy"], evaluation.TOPK_TIE_POLICY)
        self.assertEqual(first["per_phase"][0]["boundary_tie_total"], 4)
        self.assertEqual(first["per_phase"][0]["boundary_tie_selected"], 2)
        self.assertFalse(first["deployable_student_metric"])

    def test_text_only_tokenization_never_reads_either_latent_blob(self) -> None:
        class LatentGuardRow(dict):
            def get(self, key, default=None):
                if key == "video_vae_latents":
                    raise AssertionError("text encoder touched target-bearing latents")
                return super().get(key, default)

        messages = [
            {"type": "video", "has_loss": 0},
            {"type": "text", "text": "make the dog jump", "has_loss": 0},
            {"type": "video_gen", "has_loss": 1},
        ]

        class Tokenizer:
            def __call__(self, text, **kwargs):
                self.text = text
                self.kwargs = kwargs
                return SimpleNamespace(
                    input_ids=torch.tensor([[4, 5, 6]], dtype=torch.long),
                    attention_mask=torch.ones(1, 3, dtype=torch.long),
                )

        tokenizer = Tokenizer()
        action, noop = evaluation.text_only_token_batches(
            raw_row=LatentGuardRow(
                inputs=json.dumps(messages),
                video_vae_latents=object(),
            ),
            tokenizer=tokenizer,
            noop_instruction="Keep unchanged.",
            prompt_cleaner=lambda text: text.strip(),
            system_prompt="SYSTEM:",
        )
        self.assertEqual(tuple(action["input_ids"].shape), (1, 3))
        self.assertEqual(tuple(noop["t5_input_lens"].shape), (1, 1))
        self.assertEqual(tokenizer.kwargs["add_special_tokens"], True)

    def _plans(self):
        source = torch.zeros(1, 21, 1, 10, 8)
        teacher_gates = torch.zeros(1, 3, 21, 1, 10)
        teacher_gates[:, spt.GATE_PRESERVE, ..., :8] = 1.0
        teacher_gates[:, spt.GATE_TRANSPORT, ..., 8] = 1.0
        teacher_gates[:, spt.GATE_GENERATE, ..., 9] = 1.0
        teacher_offsets = torch.zeros(1, 3, 21, 1, 10)
        teacher_offsets[:, 2, ..., 8] = -2.0
        teacher = spt.PhasePlan(
            offsets=teacher_offsets,
            gate_probs=teacher_gates,
            provenance="oracle_pair_proxy",
        )

        change_logits = torch.full((1, 1, 21, 1, 10), -20.0)
        change_logits[..., 7] = 20.0
        change_logits[..., 8] = 20.0
        novelty_logits = torch.full_like(change_logits, -20.0)
        novelty_logits[..., 9] = 20.0
        action_gates = torch.zeros(1, 3, 21, 1, 10)
        action_gates[:, spt.GATE_PRESERVE] = 1.0
        action_gates[:, spt.GATE_PRESERVE, ..., 8] = 0.08
        action_gates[:, spt.GATE_TRANSPORT, ..., 8] = 0.92
        action_gates[:, spt.GATE_PRESERVE, ..., 9] = 0.10
        action_gates[:, spt.GATE_GENERATE, ..., 9] = 0.90
        candidates = torch.tensor(grounded.candidate_lattice(), dtype=torch.float32)
        zero_index = grounded.candidate_lattice().index((0, 0, 0))
        shifted_index = grounded.candidate_lattice().index((0, 0, -2))
        logits = torch.full((1, 125, 21, 1, 10), -20.0)
        logits[:, zero_index] = 20.0
        logits[:, zero_index, ..., 8] = -20.0
        logits[:, shifted_index, ..., 8] = 20.0
        action_offsets = torch.zeros_like(teacher_offsets)
        action_offsets[:, 2, ..., 8] = -2.0
        diagnostics = {
            "architecture": grounded.ARCHITECTURE_NAME,
            "change_logits": change_logits,
            "novelty_logits": novelty_logits,
            "offset_candidate_logits": logits,
            "offset_candidates": candidates,
        }
        action = spt.PhasePlan(
            offsets=action_offsets,
            gate_probs=action_gates,
            provenance="student",
            diagnostics=diagnostics,
        )

        noop_change = torch.full_like(change_logits, -20.0)
        noop_gates = torch.zeros_like(action_gates)
        noop_gates[:, spt.GATE_PRESERVE] = 1.0
        noop = spt.PhasePlan(
            offsets=torch.zeros_like(teacher_offsets),
            gate_probs=noop_gates,
            provenance="student",
            diagnostics={
                "architecture": grounded.ARCHITECTURE_NAME,
                "change_logits": noop_change,
                "novelty_logits": torch.zeros_like(noop_change),
                "offset_candidate_logits": torch.where(
                    torch.arange(125).view(1, 125, 1, 1, 1) == zero_index,
                    torch.tensor(20.0),
                    torch.tensor(-20.0),
                ).expand(1, 125, 21, 1, 10).clone(),
                "offset_candidates": candidates,
            },
        )
        return source, action, teacher, noop

    def test_tensor_metrics_preserve_raw_counts_conditional_tg_offsets_cap_and_noop(self) -> None:
        source, action, teacher, noop = self._plans()
        report = evaluation.evaluate_row_plans(
            row_index=0,
            iid="row-zero",
            identity_sha256="a" * 64,
            source=source,
            action=action,
            teacher=teacher,
            noop=noop,
        )
        self.assertEqual(
            report["change_head"]["raw_counts"],
            {"tp": 21, "fp": 21, "fn": 21, "tn": 147},
        )
        self.assertEqual(
            report["conditional_tg_head"]["raw_counts"],
            {"tp": 21, "fp": 0, "fn": 0, "tn": 21},
        )
        self.assertEqual(report["conditional_tg_head"]["f1"], 1.0)
        self.assertEqual(report["transport_offset"]["top1_correct"], 21)
        self.assertEqual(report["transport_offset"]["top1_accuracy"], 1.0)
        self.assertEqual(report["transport_offset"]["mae"], 0.0)
        self.assertAlmostEqual(
            report["student_action"]["observed_max_generate_fraction_per_phase"],
            0.09,
            places=6,
        )
        self.assertEqual(report["noop"]["change_head"]["predicted_fraction"], 0.0)
        self.assertEqual(report["noop"]["offset"]["zero_top1_accuracy"], 1.0)
        self.assertEqual(
            report["teacher_cardinality_topk_change"]["raw_counts"],
            {"tp": 21, "fp": 21, "fn": 21, "tn": 147},
        )
        self.assertAlmostEqual(report["change_ranking"]["pr_auc"], 0.35)
        self.assertAlmostEqual(
            report["action_noop_delta_ranking"]["pr_auc"], 0.35
        )
        self.assertLess(report["phase_mass"]["mae"], 1.0e-6)
        self.assertIsNone(report["phase_mass"]["pearson_correlation"])
        self.assertFalse(report["phase_mass"]["dedicated_learned_mass_head_exists"])

    def test_offset_metric_rejects_non_lattice_teacher_transport(self) -> None:
        source, action, teacher, noop = self._plans()
        teacher.offsets[:, 2, ..., 8] = -3.0
        with self.assertRaisesRegex(
            evaluation.GroundedCheckpointEvaluationError, "outside"
        ):
            evaluation.evaluate_row_plans(
                row_index=0,
                iid="row-zero",
                identity_sha256="a" * 64,
                source=source,
                action=action,
                teacher=teacher,
                noop=noop,
            )


if __name__ == "__main__":
    unittest.main()
