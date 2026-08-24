from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import train_action_repr_target_t0_canary_retry8_v1 as retry8


class Retry8SameRuntimeGateTests(unittest.TestCase):
    def test_revision_constants_and_source_pin_closure(self) -> None:
        self.assertIn("retry8", retry8.SCHEMA_VERSION)
        self.assertIn("stage_b_t0_retry8", retry8.EXPECTED_CANONICAL_OUTPUT_PATH)
        self.assertTrue(retry8.EXPECTED_ATTEMPT_CLAIM_PATH.endswith(
            ".single_update.retry8.attempt_claim.json"
        ))
        self.assertIn(
            "methods/bernini_action_editing/train_action_repr_target_t0_canary_retry8_v1.py",
            retry8.EXPECTED_SOURCE_PIN_PATHS,
        )
        self.assertIn(
            "methods/bernini_action_editing/train_action_repr_target_t0_canary_retry7_v1.py",
            retry8.EXPECTED_SOURCE_PIN_PATHS,
        )
        self.assertIn(
            "methods/bernini_action_editing/scripts/diagnose_stage_b_t0_batch_replay_v1.py",
            retry8.EXPECTED_SOURCE_PIN_PATHS,
        )

    def test_optimizer_core_receives_same_runtime_not_historical_digests(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        output = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        current_input = "1" * 64
        current_base = "2" * 64
        current_native = retry8.retry7.g2a.tensor_sha256(output)
        captured = {}

        def fake_core(**kwargs):
            captured.update(kwargs)
            return retry8.retry7.OneStepResult(
                step0_state={},
                step1_state={},
                facts={
                    "matched_source_owned_batch_sha256": current_input,
                    "native_step0_output_sha256": current_native,
                    "renderer_base_snapshot_digest_before": current_base,
                    "renderer_base_snapshot_digest_after": current_base,
                },
            )

        with (
            mock.patch.object(
                retry8.retry7.g2a_world4,
                "renderer_base_snapshot",
                return_value=type("Snapshot", (), {"digest": current_base})(),
            ),
            mock.patch.object(retry8.retry7, "_consensus"),
            mock.patch.object(retry8, "_ORIGINAL_RUN_ONE_STEP", side_effect=fake_core),
        ):
            result = retry8.run_one_step_optimizer_canary(
                model=model,
                forward_native=lambda: output.clone(),
                input_digest=lambda: current_input,
                routes={},
                feature_projection=torch.ones((1, 1)),
                hidden_width=1,
                middle_width=1,
                expected_input_digest="a" * 64,
                expected_base_digest="b" * 64,
                expected_native_output_digest="c" * 64,
            )

        self.assertEqual(captured["expected_input_digest"], current_input)
        self.assertEqual(captured["expected_base_digest"], current_base)
        self.assertEqual(captured["expected_native_output_digest"], current_native)
        facts = result.facts
        self.assertFalse(facts["matched_production_g2a_source_batch"])
        self.assertFalse(facts["matched_production_g2a_renderer_base"])
        self.assertFalse(facts["matched_production_g2a_native_output"])
        self.assertFalse(facts["cross_run_historical_match_required"])
        self.assertTrue(
            facts["same_runtime_g2a_gate"]["same_batch_used_by_optimizer_canary"]
        )
        self.assertTrue(
            facts["same_runtime_g2a_gate"][
                "route_off_and_six_zero_init_routes_exact_native_bits"
            ]
        )

    def test_receipt_validator_keeps_historical_mismatch_honest(self) -> None:
        historical_input = "a" * 64
        historical_base = "b" * 64
        historical_native = "c" * 64
        current_input = "d" * 64
        current_base = historical_base
        current_native = "e" * 64
        source_names = {
            "train_action_repr_target_t0_canary_retry8_v1.py",
            "action_repr_g2a_adapter_v1.py",
            "action_representation_joint_objective_v1.py",
            "audit_action_repr_g2a_world4_v1.py",
            "score_g1_joint_action_repr_admission_v1.py",
            "evaluate_g1_action_repr_selectivity_v1.py",
            "materialize_g1_flow_control_cohort_v1.py",
            "materialize_g1_middle_control_cohort_v1.py",
            "materialize_decoded_middle_action_repr_v1.py",
            "dense_flow_token_adapter_v1.py",
            "exact_local_video_materializer_v1.py",
            "train_lora.py",
            "train_self_generated_action_quotient_v1.py",
        }
        source_lock = {name: "f" * 64 for name in source_names}
        source_lock["train_action_repr_target_t0_canary_retry8_v1.py"] = (
            retry8.retry7.file_sha256(Path(retry8.__file__).resolve())
        )
        training = {
            "matched_source_owned_batch_sha256": current_input,
            "native_step0_output_sha256": current_native,
            "renderer_base_snapshot_digest_before": current_base,
            "renderer_base_snapshot_digest_after": current_base,
            "matched_production_g2a_source_batch": False,
            "matched_production_g2a_renderer_base": True,
            "matched_production_g2a_native_output": False,
            "cross_run_historical_match_required": False,
            "historical_production_g2a_reference": {
                "source_batch_sha256": historical_input,
                "renderer_base_sha256": historical_base,
                "native_output_sha256": historical_native,
                "source_batch_matches_same_runtime": False,
                "renderer_base_matches_same_runtime": True,
                "native_output_matches_same_runtime": False,
                "authenticated_reference_only": True,
            },
            "same_runtime_g2a_gate": {
                "source_batch_sha256": current_input,
                "renderer_base_sha256": current_base,
                "native_output_sha256": current_native,
                "pre_adapter_native_baseline_executed": True,
                "same_batch_used_by_optimizer_canary": True,
                "route_off_and_six_zero_init_routes_exact_native_bits": True,
                "batch_digest_stable_through_all_forwards": True,
            },
        }
        receipt = {
            "receipt_digest": "0" * 64,
            "upstream_authority": {
                "production_g2a_matched_native_batch_sha256": historical_input,
                "production_g2a_renderer_base_snapshot_digest": historical_base,
                "production_g2a_native_post_head_tensor_sha256": historical_native,
            },
            "training": training,
            "source_lock": source_lock,
        }
        projected = {}

        def capture(value):
            projected.update(value)
            return value

        with mock.patch.object(
            retry8, "_ORIGINAL_VALIDATE_RECEIPT", side_effect=capture
        ):
            returned = retry8.validate_t0_receipt(receipt)

        self.assertIs(returned, receipt)
        projected_training = projected["training"]
        self.assertTrue(projected_training["matched_production_g2a_source_batch"])
        self.assertEqual(
            projected_training["matched_source_owned_batch_sha256"], historical_input
        )
        self.assertNotIn("same_runtime_g2a_gate", projected_training)
        self.assertIn(
            "train_action_repr_target_t0_canary_retry7_v1.py",
            projected["source_lock"],
        )
        self.assertNotIn(
            "train_action_repr_target_t0_canary_retry8_v1.py",
            projected["source_lock"],
        )


if __name__ == "__main__":
    unittest.main()
