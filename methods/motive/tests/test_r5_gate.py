from __future__ import annotations

import unittest

import numpy as np

from motive.r5_gate import (
    GateInputError,
    R5GateThresholds,
    cross_content_retrieval,
    evaluate_r5_gate,
    false_activation_summary,
    macro_retrieval,
    paired_group_comparison,
)


ARMS = (
    "full",
    "text_only",
    "pairshuffle",
    "matched_random",
    "centroid",
    "source_shuffle",
    "prompt_shuffle",
)


def _contract(
    *,
    label_mode: str = "human",
    production_eligible: bool = True,
    split_version: str = "source-visual-cluster-v1",
    auxiliary_complete: bool = True,
) -> dict:
    auxiliary = {
        "direction_probe": auxiliary_complete,
        "speed_probe": auxiliary_complete,
        "phase_probe": auxiliary_complete,
        "camera_leakage": auxiliary_complete,
        "stability": auxiliary_complete,
        "pair_specificity": auxiliary_complete,
    }
    return {
        "data_seed": 7,
        "model_seeds": [2026, 2027],
        "dataset": {
            "label_mode": label_mode,
            "production_eligible": production_eligible,
            "split_version": split_version,
            "positive_count": 4,
            "positive_group_count": 4,
            "action_family_count": 2,
            "test_positive_group_count": 4,
            "negative_audit_count": 0,
        },
        "formal_auxiliary_checks_complete": auxiliary_complete,
        "formal_auxiliary_checks": auxiliary,
    }


def _rows(*, strong: bool = True) -> list[dict]:
    rows: list[dict] = []
    control_scores = {
        "centroid": 0.65,
        "pairshuffle": 0.60,
        "matched_random": 0.55,
        "text_only": 0.62,
        "source_shuffle": 0.58,
        "prompt_shuffle": 0.57,
    }
    for seed in (2026, 2027):
        for arm in ARMS:
            score = (
                (0.90 if strong else 0.60)
                if arm == "full"
                else control_scores.get(arm, 0.5)
            )
            for index in range(4):
                rows.append(
                    {
                        "iid": f"sample-{index}",
                        "split": "test",
                        "content_group_id": f"group-{index}",
                        "action_family": "walk" if index < 2 else "turn",
                        "label_role": "positive_delta",
                        "label_type": "positive",
                        "arm": arm,
                        "model_seed": seed,
                        "control_valid": True,
                        "actor_target_active": True,
                        "camera_target_active": True,
                        "actor_direction_cosine": score,
                        "actor_cross_content_ap": score,
                        "actor_cross_content_r1": score,
                        "actor_cross_content_r5": score,
                        "actor_log_magnitude_absolute_error": 1.0 - score,
                        "camera_log_magnitude_absolute_error": 1.0 - score,
                        "camera_direction_cosine": score,
                        "retrieval_valid": True,
                        "actor_predicted_log_magnitude": 0.3,
                    }
                )
    return rows


def _small_thresholds() -> R5GateThresholds:
    return R5GateThresholds(
        minimum_human_positives=4,
        minimum_positive_groups=4,
        minimum_action_families=2,
        minimum_test_positive_groups=4,
        minimum_model_seeds=2,
        minimum_positive_seed_directions=2,
        minimum_actor_cosine_gain=0.05,
        minimum_macro_map_gain=0.05,
        maximum_signflip_p=1.0,
        bootstrap_samples=500,
        signflip_samples=500,
    )


class RetrievalTests(unittest.TestCase):
    def test_retrieval_excludes_same_content_group(self) -> None:
        prediction = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        result = cross_content_retrieval(
            predicted_actor_direction=prediction,
            target_actor_direction=prediction,
            action_families=["walk", "walk", "turn", "turn"],
            content_group_ids=["a", "b", "c", "d"],
            iids=["0", "1", "2", "3"],
        )
        self.assertTrue(all(row["retrieval_valid"] for row in result))
        self.assertTrue(
            all(row["actor_cross_content_ap"] == 1.0 for row in result)
        )
        self.assertTrue(all(row["retrieval_positives"] == 1 for row in result))

    def test_query_without_cross_content_positive_is_invalid(self) -> None:
        result = cross_content_retrieval(
            predicted_actor_direction=np.eye(2, dtype=np.float32),
            target_actor_direction=np.eye(2, dtype=np.float32),
            action_families=["walk", "turn"],
            content_group_ids=["a", "b"],
            iids=["0", "1"],
        )
        self.assertFalse(result[0]["retrieval_valid"])
        self.assertIsNone(result[0]["actor_cross_content_ap"])

    def test_macro_is_by_family_not_query_frequency(self) -> None:
        rows = [
            {
                "action_family": "frequent",
                "label_role": "positive_delta",
                "control_valid": True,
                "retrieval_valid": True,
                "actor_cross_content_ap": value,
                "actor_cross_content_r1": value,
                "actor_cross_content_r5": value,
            }
            for value in (0.0, 0.0, 0.0)
        ]
        rows.append(
            {
                "action_family": "rare",
                "label_role": "positive_delta",
                "control_valid": True,
                "retrieval_valid": True,
                "actor_cross_content_ap": 1.0,
                "actor_cross_content_r1": 1.0,
                "actor_cross_content_r5": 1.0,
            }
        )
        summary = macro_retrieval(rows)
        self.assertEqual(summary["families"], 2)
        self.assertAlmostEqual(summary["macro_mAP"], 0.5)


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_unit_is_content_group(self) -> None:
        rows = _rows()
        comparison = paired_group_comparison(
            rows,
            treatment_arm="full",
            control_arm="centroid",
            metric="actor_direction_cosine",
            bootstrap_samples=200,
            signflip_samples=200,
            random_seed=9,
        )
        self.assertEqual(comparison["content_groups"], 4)
        self.assertEqual(comparison["model_seeds"], 2)
        self.assertEqual(comparison["paired_queries_across_seeds"], 8)
        self.assertAlmostEqual(comparison["mean_gain"], 0.25)
        self.assertGreater(comparison["group_bootstrap_ci"][0], 0.0)

    def test_negative_activation_is_reported_by_type(self) -> None:
        rows = [
            {
                "label_role": "negative_audit",
                "label_type": "static",
                "control_valid": True,
                "actor_predicted_log_magnitude": 0.1,
            },
            {
                "label_role": "negative_audit",
                "label_type": "static",
                "control_valid": True,
                "actor_predicted_log_magnitude": 0.5,
            },
            {
                "label_role": "negative_audit",
                "label_type": "endpoint",
                "control_valid": True,
                "actor_predicted_log_magnitude": 0.05,
            },
        ]
        summary = false_activation_summary(rows, activation_threshold=0.2)
        self.assertEqual(summary["negative_rows"], 3)
        self.assertEqual(summary["by_type"]["static"]["activation_rate"], 0.5)
        self.assertEqual(summary["by_type"]["endpoint"]["activation_rate"], 0.0)


class GateDecisionTests(unittest.TestCase):
    def test_strong_formal_evidence_can_pass_only_with_auxiliary_checks(self) -> None:
        result = evaluate_r5_gate(
            rows=_rows(strong=True),
            contract=_contract(),
            thresholds=_small_thresholds(),
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["production_decision"])

    def test_weak_formal_evidence_is_no_go(self) -> None:
        result = evaluate_r5_gate(
            rows=_rows(strong=False),
            contract=_contract(),
            thresholds=_small_thresholds(),
        )
        self.assertEqual(result["status"], "NO_GO")
        self.assertFalse(result["production_decision"])

    def test_legacy_qwen_is_always_insufficient(self) -> None:
        result = evaluate_r5_gate(
            rows=_rows(strong=True),
            contract=_contract(
                label_mode="strict_legacy_qwen",
                production_eligible=False,
                split_version="source-phash-near-cluster-v1",
            ),
            thresholds=_small_thresholds(),
        )
        self.assertEqual(result["status"], "INSUFFICIENT")
        self.assertFalse(result["production_decision"])

    def test_missing_auxiliary_checks_is_insufficient_not_pass(self) -> None:
        result = evaluate_r5_gate(
            rows=_rows(strong=True),
            contract=_contract(auxiliary_complete=False),
            thresholds=_small_thresholds(),
        )
        self.assertEqual(result["status"], "INSUFFICIENT")
        self.assertTrue(
            any(
                "auxiliary" in reason
                for reason in result["insufficiency_reasons"]
            )
        )

    def test_missing_arm_is_invalid_input(self) -> None:
        rows = [
            row for row in _rows() if row["arm"] != "pairshuffle"
        ]
        with self.assertRaisesRegex(GateInputError, "missing arms"):
            evaluate_r5_gate(
                rows=rows,
                contract=_contract(),
                thresholds=_small_thresholds(),
            )


if __name__ == "__main__":
    unittest.main()
