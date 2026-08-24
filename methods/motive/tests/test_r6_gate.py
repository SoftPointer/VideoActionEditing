from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from motive.r6_gate import (
    R6_QUERY_SCHEMA,
    R6GateInputError,
    R6PilotThresholds,
    compatibility_diagnostic,
    evaluate_r6_gate,
    hierarchical_paired_group_bootstrap,
    main,
)


def _contract() -> dict:
    digest = "a" * 64
    iids = sorted(
        [
            f"{split}-iid-{index}"
            for split in ("validation", "test")
            for index in range(20)
        ]
    )
    iid_digest = hashlib.sha256(
        json.dumps(
            iids,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "motive-r6-training-v1",
        "data_seed": 260108828,
        "model_seeds": [2026, 2027, 2028, 2029, 2030],
        "input_transform_fit_split": "train",
        "delta_transform_fit_split": "train",
        "delta_transform_fit_role": "positive_delta",
        "query_target_is_predictor_input": False,
        "failed_outcomes_update_delta_predictor": False,
        "compatibility_scales_conditioning_tokens": False,
        "source_snapshot": {
            "path": "/snapshot",
            "tree_sha256": "e" * 64,
            "source_files_manifest": "/snapshot/SOURCE_FILES.jsonl",
            "source_files_sha256": "f" * 64,
            "trainer_module_path": (
                "/snapshot/methods/motive/motive/"
                "train_source_aware_r6.py"
            ),
        },
        "runtime": {
            "python_version": "3.11",
            "numpy_version": "2.0",
            "torch_version": "2.5",
            "requested_device": "cpu",
            "deterministic_algorithms": True,
        },
        "claim_scope": {
            "generation_authorized": False,
            "generator_ready_tokens": False,
            "motion_token_export_authorized": False,
        },
        "dataset": {
            "action_family_source_verified": False,
            "row_count": 40,
            "iid_set_digest": iid_digest,
            "split_role_counts": {
                "validation:positive_delta": 10,
                "validation:failed_outcome_compatibility": 10,
                "test:positive_delta": 10,
                "test:failed_outcome_compatibility": 10,
            },
        },
        "evaluation": {
            "active_log_magnitude_threshold": 1e-4,
            "active_threshold_origin": "fixed-pre-registered-1e-4",
            "active_threshold_fit_scope": "none",
            "family_retrieval_gate_eligible": False,
        },
        "semantic_artifact": {
            "source_field": "instruction",
            "frozen_encoder": True,
            "target_derived_input": False,
            "label_derived_input": False,
            "provenance_digest": digest,
        },
        "reference_selector": {
            "selector_kind": (
                "prompt-to-observed-action-semantic-train-bank"
            ),
            "candidate_bank_split": "train",
            "candidate_bank_label_role": "positive_delta",
            "threshold_fit_split": "train",
            "threshold_fit_role": "positive_delta",
            "threshold_origin": "train-positive-self-alignment-q10",
            "threshold_quantile": 0.10,
            "similarity_threshold": 0.25,
            "query_target_used": False,
            "different_iid_enforced": True,
            "different_content_group_enforced": True,
            "different_subject_cluster_enforced": True,
            "oracle_action_family_used": False,
            "gate_eligible": True,
            "pair_ledger_sha256": "b" * 64,
            "pair_digest": "c" * 64,
            "reference_bank_provenance_digest": "d" * 64,
            "test_positive_coverage": {
                "eligible_queries": 10,
                "any_reference_fraction": 1.0,
                "full_reference_fraction": 1.0,
            },
            "reference_load": {
                "unique_reference_count": 10,
                "maximum_reference_fraction": 0.10,
            },
        },
    }


def _rows(*, oracle: bool = False, poor_compatibility: bool = False) -> list[dict]:
    arms = {
        "semantic_only": (0.60, 0.60),
        "independent_ref": (0.85, 0.85),
        "wrong_ref": (0.20, 0.20),
        "matched_random": (0.10, 0.10),
        "centroid": (0.20, 0.20),
        "source_shuffle": (0.10, 0.10),
        "semantic_shuffle": (0.10, 0.10),
    }
    output: list[dict] = []
    for seed in (2026, 2027, 2028, 2029, 2030):
        for split in ("validation", "test"):
            for index in range(20):
                positive = index < 10
                compatibility = (
                    0.5
                    if poor_compatibility
                    else 0.9
                    if positive
                    else 0.1
                )
                for arm, (cosine, average_precision) in arms.items():
                    row = {
                        "schema_version": R6_QUERY_SCHEMA,
                        "arm": arm,
                        "model_seed": seed,
                        "iid": f"{split}-iid-{index}",
                        "split": split,
                        "content_group_id": f"{split}-group-{index}",
                        "label_role": (
                            "positive_delta"
                            if positive
                            else "failed_outcome_compatibility"
                        ),
                        "control_valid": True,
                        "actor_cosine": cosine if positive else None,
                        "actor_cross_content_ap": (
                            average_precision if positive else None
                        ),
                        "compatibility_target": 1.0 if positive else 0.0,
                        "compatibility_probability": compatibility,
                        "synthetic_mismatched_positive_probability": (
                            0.2 if positive else None
                        ),
                        "oracle_diagnostic": False,
                        "gate_eligible": True,
                        "query_target_used_as_predictor_input": False,
                        "failed_outcome_used_as_noop": False,
                        "compatibility_scales_conditioning_tokens": False,
                    }
                    if arm == "independent_ref":
                        row.update(
                            {
                                "alternate_reference_available": True,
                                "alternate_reference_prediction_cosine": 0.95,
                            }
                        )
                    output.append(row)
                if oracle:
                    output.append(
                        {
                        "schema_version": R6_QUERY_SCHEMA,
                        "arm": "exact_target_oracle",
                        "model_seed": seed,
                        "iid": f"{split}-iid-{index}",
                        "split": split,
                        "content_group_id": f"{split}-group-{index}",
                        "label_role": (
                            "positive_delta"
                            if positive
                            else "failed_outcome_compatibility"
                        ),
                        "control_valid": True,
                        "actor_cosine": -1.0,
                        "actor_cross_content_ap": -1.0,
                        "compatibility_target": 1.0 if positive else 0.0,
                        "compatibility_probability": compatibility,
                        "oracle_diagnostic": True,
                        "gate_eligible": False,
                        }
                    )
    return output


def _thresholds() -> R6PilotThresholds:
    return R6PilotThresholds(
        bootstrap_samples=250,
        minimum_test_positive_groups=10,
        minimum_paired_test_groups=5,
    )


class R6GateTests(unittest.TestCase):
    def test_cli_defaults_to_contract_data_seed(self) -> None:
        rows = _rows()
        contract = _contract()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_query = root / "per_query.jsonl"
            contract_path = root / "contract.json"
            pair_ledger = root / "reference_pairs.jsonl"
            inferred_output = root / "gate_inferred.json"
            explicit_output = root / "gate_explicit.json"
            per_query.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            pair_ledger.write_text("", encoding="utf-8")
            contract["reference_selector"]["pair_ledger_sha256"] = (
                hashlib.sha256(b"").hexdigest()
            )
            contract_path.write_text(
                json.dumps(contract, sort_keys=True),
                encoding="utf-8",
            )
            common = [
                "--per-query",
                str(per_query),
                "--contract",
                str(contract_path),
                "--pair-ledger",
                str(pair_ledger),
                "--bootstrap-samples",
                "250",
            ]
            self.assertEqual(
                main(common + ["--output", str(inferred_output)]),
                0,
            )
            self.assertEqual(
                main(
                    common
                    + [
                        "--random-seed",
                        str(contract["data_seed"]),
                        "--output",
                        str(explicit_output),
                    ]
                ),
                0,
            )
            self.assertEqual(
                inferred_output.read_bytes(),
                explicit_output.read_bytes(),
            )

    def test_formal_gate_is_always_insufficient_but_pilot_can_go(self) -> None:
        result = evaluate_r6_gate(
            rows=_rows(),
            contract=_contract(),
            thresholds=_thresholds(),
            random_seed=7,
        )
        self.assertEqual(result["status"], "INSUFFICIENT")
        self.assertFalse(result["production_decision"])
        self.assertFalse(result["generation_authorized"])
        self.assertEqual(result["pilot_diagnostic"]["status"], "GO")
        self.assertGreaterEqual(
            result["compatibility_diagnostic"]["mean_auroc"],
            0.99,
        )
        direct = compatibility_diagnostic(_rows())
        self.assertEqual(
            direct["operating_threshold_fit_split"],
            "validation",
        )
        self.assertTrue(direct["both_classes_present_every_seed"])

    def test_oracle_rows_are_excluded_from_all_comparisons(self) -> None:
        without = evaluate_r6_gate(
            rows=_rows(oracle=False),
            contract=_contract(),
            thresholds=_thresholds(),
            random_seed=8,
        )
        with_oracle = evaluate_r6_gate(
            rows=_rows(oracle=True),
            contract=_contract(),
            thresholds=_thresholds(),
            random_seed=8,
        )
        self.assertEqual(without["comparisons"], with_oracle["comparisons"])
        self.assertGreater(
            with_oracle["oracle_diagnostic"]["row_count"],
            0,
        )
        self.assertTrue(
            with_oracle["oracle_diagnostic"][
                "excluded_from_all_gate_criteria"
            ]
        )

    def test_query_target_selector_leakage_fails_closed(self) -> None:
        contract = copy.deepcopy(_contract())
        contract["reference_selector"]["query_target_used"] = True
        with self.assertRaisesRegex(R6GateInputError, "query_target_used"):
            evaluate_r6_gate(
                rows=_rows(),
                contract=contract,
                thresholds=_thresholds(),
            )

    def test_poor_compatibility_prevents_pilot_go(self) -> None:
        result = evaluate_r6_gate(
            rows=_rows(poor_compatibility=True),
            contract=_contract(),
            thresholds=_thresholds(),
        )
        self.assertEqual(result["status"], "INSUFFICIENT")
        self.assertEqual(result["pilot_diagnostic"]["status"], "NO_GO")
        failed = {
            item["name"]
            for item in result["pilot_diagnostic"]["criteria"]
            if not item["passed"]
        }
        self.assertIn("compatibility_test_auroc", failed)
        self.assertIn("compatibility_test_failed_outcome_fpr", failed)

    def test_oracle_arm_cannot_enter_bootstrap(self) -> None:
        with self.assertRaisesRegex(ValueError, "oracle"):
            hierarchical_paired_group_bootstrap(
                _rows(oracle=True),
                treatment_arm="exact_target_oracle",
                control_arm="semantic_only",
                metric="actor_cosine",
            )

    def test_crossed_group_signflip_does_not_fake_zero_mean_gain(self) -> None:
        rows = _rows()
        for row in rows:
            if (
                row["arm"] == "independent_ref"
                and row["label_role"] == "positive_delta"
            ):
                index = int(str(row["iid"]).rsplit("-", 1)[-1])
                row["actor_cosine"] = 0.8 if index < 5 else 0.4
            elif (
                row["arm"] == "semantic_only"
                and row["label_role"] == "positive_delta"
            ):
                row["actor_cosine"] = 0.6
        result = hierarchical_paired_group_bootstrap(
            rows,
            treatment_arm="independent_ref",
            control_arm="semantic_only",
            metric="actor_cosine",
            bootstrap_samples=100,
            random_seed=99,
        )
        self.assertAlmostEqual(result["mean_gain"], 0.0, places=7)
        self.assertGreater(result["signflip_p"], 0.05)
        self.assertEqual(
            result["bootstrap_design"],
            "crossed-two-way-resample-shared-groups-and-model-seeds",
        )


if __name__ == "__main__":
    unittest.main()
