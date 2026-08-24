from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v15r2 online-anchor tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_online_anchor_attention_dynamic_static_v15r2 as method


WORKER = ROOT / "scripts/auh_train_online_anchor_dynamic_static_v15r2.sh"
LAUNCHER = ROOT / "scripts/auh_launch_online_anchor_dynamic_static_v15r2_job149363.sh"


class DynamicStaticV15R2Test(unittest.TestCase):
    def setUp(self):
        method._RUNTIME_AUDIT = method._empty_runtime_audit()
        method._RUNTIME_AUDIT["current_target_iid"] = "e03-v0"
        method._RUNTIME_AUDIT["current_target_event"] = "e03"

    def args(self, output: Path, *, steps: int = 8):
        return method.build_parser().parse_args(
            [
                "--bernini-root", "/tmp/bernini",
                "--veomni-root", "/tmp/veomni",
                "--checkpoint", "/tmp/Bernini-R-1.3B-Diffusers-ff4c5d4",
                "--pair-manifest", "/tmp/pairs.json",
                "--authoring", "/tmp/authoring.json",
                "--output", str(output),
                "--profile", "dynamic_static",
                "--route-operator", method.v15.ROUTE_OPERATOR,
                "--max-steps", str(steps),
                "--micro-records", "2",
                "--source-variant", "not_applicable",
                "--route-strength", "0.25",
                "--teacher-route-strength", "0.50",
                "--training-objective", method.v15.OBJECTIVE,
                "--training-interface", "first_phase_caption_i2v",
                "--paired-target-fm-weight", "0",
                "--real-source-manifest", "/tmp/real-source.json",
                "--real-source-manifest-sha256", "8" * 64,
                "--teacher-delta-mode", "raw",
                "--routed-teacher-mode", "same_action_route_only",
                "--source-reconstruction-weight", "0.025",
                "--replay-combine-mode", method.v15.REPLAY_COMBINE_MODE,
                "--source-reconstruction-prompt", "action",
                "--learning-rate", "1e-5",
                "--method-source-revision", "1" * 64,
                "--method-source-archive-sha256", "2" * 64,
            ]
        )

    def test_validation_is_fresh_v15r2_s8_or_s32(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method.validate_args(self.args(root / "fresh-v15r2-s8"))
            method.validate_args(self.args(root / "fresh-v15r2-s32", steps=32))
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(self.args(root / "fresh-v15r2-s2", steps=2))
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(self.args(root / "fresh-v15-s8"))

    def test_feasible_geometry_uses_original_pcgrad_unchanged(self):
        action = torch.tensor([1.0, 0.0])
        raw_norm = 4.886
        cosine = -0.926
        replay = torch.tensor(
            [raw_norm * cosine, raw_norm * math.sqrt(1.0 - cosine * cosine)]
        )
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = replay.clone()
        interaction = method.merge_component_gradients(
            [("x.lora_B.weight", parameter)],
            (action,),
            replay_combine_mode="action_priority_pcgrad_010",
            base_replay_scale=0.025,
        )
        self.assertTrue(torch.allclose(parameter.grad, torch.tensor([1.0, 0.1])))
        self.assertFalse(interaction["v15r2_collinear_fallback_applied"])
        self.assertEqual(method._RUNTIME_AUDIT["fallback_count"], 0)

    def test_near_collinear_gate_falls_back_to_primary_action_only(self):
        action = torch.tensor([1.0, 0.0])
        replay = torch.tensor([-0.99, math.sqrt(1.0 - 0.99**2)])
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = replay.clone()
        interaction = method.merge_component_gradients(
            [("x.lora_B.weight", parameter)],
            (action,),
            replay_combine_mode="action_priority_pcgrad_010",
            base_replay_scale=0.025,
        )
        self.assertTrue(torch.equal(parameter.grad, action))
        self.assertTrue(interaction["v15r2_collinear_fallback_applied"])
        self.assertEqual(interaction["effective_replay_scale"], 0.0)
        self.assertLess(
            interaction["processed_replay_retained_raw_norm_fraction"], 0.2
        )
        self.assertGreater(
            interaction["action_gradient_dot_combined_gradient_fp64"], 0.0
        )
        self.assertFalse(interaction["first_order_source_fm_preserved"])
        self.assertFalse(
            interaction["v15r2_source_preservation_claimed_for_update"]
        )
        self.assertEqual(method._RUNTIME_AUDIT["fallback_count"], 1)
        self.assertEqual(method._RUNTIME_AUDIT["fallback_steps"], [1])
        self.assertEqual(method._RUNTIME_AUDIT["fallback_target_iids"], {"e03-v0"})

    def test_fallback_does_not_swallow_diagnostic_or_other_failures(self):
        action = torch.tensor([1.0, 0.0])
        replay = torch.tensor([-0.99, math.sqrt(1.0 - 0.99**2)])
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = replay.clone()
        with self.assertRaisesRegex(
            method.base.OnlineAnchorTrainingError, "GRADIENT_DIAGNOSTIC_COMPLETE"
        ):
            method.merge_component_gradients(
                [("x.lora_B.weight", parameter)],
                (action,),
                replay_combine_mode="action_priority_pcgrad_010",
                base_replay_scale=0.025,
                diagnostic_only=True,
            )
        self.assertTrue(torch.equal(parameter.grad, replay))

    def test_receipt_discloses_aggregate_fallback_without_claim(self):
        method._RUNTIME_AUDIT.update(
            {
                "merge_call_count": 4,
                "fallback_count": 1,
                "fallback_steps": [4],
                "fallback_target_iids": {"e03-v0"},
                "fallback_target_events": {"e03"},
                "fallback_geometry": [
                    {
                        "step": 4,
                        "target_iid": "e03-v0",
                        "target_event": "e03",
                        "action_replay_cosine": -0.983,
                        "processed_replay_retained_raw_norm_fraction": 0.183,
                        "effective_replay_scale": 0.0,
                    }
                ],
            }
        )
        original = {
            "schema_version": method.v15.RECEIPT_SCHEMA,
            "scientific_claim_authorized": False,
            "training_contract": {"method": method.v15.METHOD},
        }
        with mock.patch.object(
            method, "_V15_CHECKPOINT_RECEIPT", return_value=original
        ):
            receipt = method.checkpoint_receipt(args=object())
        contract = receipt["training_contract"]
        summary = receipt["v15r2_collinear_fallback_summary"]
        self.assertEqual(receipt["schema_version"], method.RECEIPT_SCHEMA)
        self.assertEqual(contract["method"], method.METHOD)
        self.assertEqual(contract["near_collinear_fallback_count"], 1)
        self.assertFalse(contract["pcgrad_retained_raw_norm_floor_was_loosened"])
        self.assertFalse(contract["source_preservation_claimed"])
        self.assertFalse(summary["scientific_claim_authorized"])
        self.assertEqual(summary["fallback_steps"], [4])

    def test_worker_and_launcher_are_bound_fresh_and_multisample(self):
        worker = WORKER.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("expected_job=149363", worker)
        self.assertIn("expected_node=auh7-1b-gpu-312", worker)
        self.assertIn("train_online_anchor_attention_dynamic_static_v15r2.py", worker)
        self.assertIn("source-online-anchor-targetowned-qk-routed-teacher-v15r2-collinear-fallback-20260823", worker)
        self.assertIn("run_stage 8\nrun_stage 32", launcher)
        self.assertNotIn("run_stage 2", launcher)
        self.assertIn("near_collinear_fallback_count >= 1", launcher)


if __name__ == "__main__":
    unittest.main()
