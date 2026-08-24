from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v16r5 online-anchor tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r5 as method


class Full644DynamicStaticV16R5GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        method._V16R5_AUDIT = method._empty_v16r5_audit()
        method._PENDING_STEP_GEOMETRY = None
        method.parent._RUNTIME_AUDIT = method.parent._empty_runtime_audit()
        method.parent._ACTIVE_OPTIMIZER = None

    def tearDown(self) -> None:
        method._PENDING_STEP_GEOMETRY = None
        method.parent._ACTIVE_OPTIMIZER = None

    def test_exact_v16r4_s11_failure_geometry_is_admitted_without_attenuation(self):
        action_norm = 0.1808167925911539
        replay_norm = 0.19879941906698245
        cosine = -0.9455751902699228
        plan = method._direction_plan(
            action_sq=action_norm * action_norm,
            replay_sq=replay_norm * replay_norm,
            raw_dot=action_norm * replay_norm * cosine,
        )

        self.assertAlmostEqual(plan["cosine"], cosine, places=14)
        self.assertAlmostEqual(plan["q"], 0.9555751902699228, places=14)
        self.assertFalse(plan["q_cap_applied"])
        self.assertAlmostEqual(
            plan["action_normalized_margin"], 0.0964318076433002, places=14
        )
        self.assertAlmostEqual(plan["source_normalized_margin"], 0.01, places=14)
        self.assertGreater(
            plan["action_normalized_margin"],
            method.MIN_FORMAL_NORMALIZED_MARGIN,
        )
        self.assertGreater(
            plan["source_normalized_margin"],
            method.MIN_FORMAL_NORMALIZED_MARGIN,
        )
        self.assertGreater(plan["planned_action_descent_cosine"], 0.29)
        self.assertGreater(plan["planned_source_descent_cosine"], 0.03)
        self.assertEqual(plan["step_attenuation_gamma"], 1.0)
        self.assertFalse(plan["step_attenuation_applied"])

    def test_near_antipodal_q_is_capped_and_step_is_deterministically_attenuated(self):
        cosine = -0.995
        plan = method._direction_plan(
            action_sq=1.0,
            replay_sq=1.0,
            raw_dot=cosine,
        )
        self.assertAlmostEqual(plan["requested_q"], 1.005, places=14)
        self.assertEqual(plan["q"], 1.0)
        self.assertTrue(plan["q_cap_applied"])
        self.assertAlmostEqual(plan["antipodal_gap"], 0.005, places=14)
        self.assertAlmostEqual(plan["action_normalized_margin"], 0.005, places=14)
        self.assertAlmostEqual(plan["source_normalized_margin"], 0.005, places=14)
        self.assertAlmostEqual(
            plan["planned_action_descent_cosine"], math.sqrt(0.005 / 2.0)
        )
        self.assertAlmostEqual(
            plan["planned_source_descent_cosine"], math.sqrt(0.005 / 2.0)
        )
        self.assertAlmostEqual(
            plan["step_attenuation_gamma"], math.sqrt(0.005 / 0.01)
        )
        self.assertTrue(plan["step_attenuation_applied"])

    def test_antipodal_gap_gate_is_strict_and_fails_closed(self):
        cosine = -1.0 + 0.5 * method.MIN_ANTIPODAL_GAP
        with self.assertRaisesRegex(
            method.base.OnlineAnchorTrainingError,
            "antipodal gap is infeasible",
        ):
            method._direction_plan(
                action_sq=1.0,
                replay_sq=1.0,
                raw_dot=cosine,
            )
        with self.assertRaisesRegex(
            method.base.OnlineAnchorTrainingError,
            "antipodal gap is infeasible",
        ):
            method._direction_plan(
                action_sq=1.0,
                replay_sq=1.0,
                raw_dot=-1.0,
            )

    def test_receipt_records_geometry_and_attenuation_history(self):
        method._V16R5_AUDIT.update(
            {
                "formal_steps": [1],
                "actual_steps": [1],
                "attenuated_steps": [1],
                "minimum_antipodal_gap": 0.005,
                "minimum_action_normalized_margin": 0.005,
                "minimum_source_normalized_margin": 0.005,
                "minimum_actual_action_descent_cosine": 0.05,
                "minimum_actual_source_descent_cosine": 0.05,
                "last_formal": {
                    "v16r5_step_attenuation_gamma": math.sqrt(0.5)
                },
                "last_actual": {"step": 1},
            }
        )
        inherited = {
            "schema_version": method.parent.RECEIPT_SCHEMA,
            "global_step": 1,
            "training_contract": {"method": method.parent.METHOD},
            "v16r4_source_descent_summary": {
                "successful_update_count": 1,
                "optimizer": method.OPTIMIZER,
            },
            "v16r4_decoded_canary_contract": {
                "schema_version": method.parent.DECODED_CANARY_SCHEMA,
                "current_checkpoint_step": 1,
            },
        }
        with mock.patch.object(
            method, "_PARENT_CHECKPOINT_RECEIPT", return_value=inherited
        ):
            receipt = method.checkpoint_receipt(args=object())

        self.assertEqual(receipt["schema_version"], method.RECEIPT_SCHEMA)
        self.assertNotIn("v16r4_source_descent_summary", receipt)
        self.assertNotIn("v16r4_decoded_canary_contract", receipt)
        source = receipt["v16r5_source_descent_summary"]
        attenuation = source["near_antipodal_global_step_attenuation"]
        self.assertEqual(source["direction_policy"], method.DIRECTION_POLICY)
        self.assertEqual(source["q_min"], 0.01)
        self.assertEqual(source["q_max"], 1.0)
        self.assertEqual(attenuation["applied_steps"], [1])
        self.assertAlmostEqual(attenuation["current_step_gamma"], math.sqrt(0.5))
        self.assertAlmostEqual(
            attenuation["current_effective_active_coordinate_rms"],
            method.LEARNING_RATE * math.sqrt(0.5),
        )
        self.assertFalse(attenuation["changes_gradient_direction"])
        contract = receipt["training_contract"]
        self.assertEqual(contract["method"], method.METHOD)
        self.assertEqual(
            contract["optimizer_unattenuated_base_active_coordinate_rms"],
            method.LEARNING_RATE,
        )
        self.assertAlmostEqual(
            contract["optimizer_current_effective_active_coordinate_rms"],
            method.LEARNING_RATE * math.sqrt(0.5),
        )
        self.assertTrue(
            contract["optimizer_effective_active_coordinate_rms_is_step_dependent"]
        )
        self.assertFalse(contract["action_only_fallback_allowed"])
        self.assertFalse(contract["optimizer_retry_allowed"])
        self.assertFalse(contract["optimizer_state_reset_allowed"])

    def test_main_installs_only_temporary_parent_hooks(self):
        names = (
            "METHOD",
            "RECEIPT_SCHEMA",
            "OPTIMIZER",
            "OPTIMIZER_FAILURE_POLICY",
            "DECODED_CANARY_SCHEMA",
            "build_parser",
            "validate_args",
            "merge_component_gradients",
            "actual_optimizer_update_probe",
            "checkpoint_receipt",
            "_projected_optimizer_factory",
        )
        originals = {name: getattr(method.parent, name) for name in names}

        def observe(_argv):
            self.assertEqual(method.parent.METHOD, method.METHOD)
            self.assertIs(
                method.parent.merge_component_gradients,
                method.merge_component_gradients,
            )
            self.assertIs(
                method.parent.actual_optimizer_update_probe,
                method.actual_optimizer_update_probe,
            )
            self.assertIs(
                method.parent._projected_optimizer_factory,
                method._projected_optimizer_factory,
            )
            return 17

        with mock.patch.object(method.parent, "main", side_effect=observe):
            self.assertEqual(method.main([]), 17)
        for name, value in originals.items():
            self.assertIs(getattr(method.parent, name), value)

    def test_exact_s11_tensor_merge_preserves_both_directions(self):
        action_norm = 0.1808167925911539
        replay_norm = 0.19879941906698245
        cosine = -0.9455751902699228
        action = torch.tensor([action_norm, 0.0], dtype=torch.float64)
        replay = torch.tensor(
            [
                replay_norm * cosine,
                replay_norm * math.sqrt(1.0 - cosine * cosine),
            ],
            dtype=torch.float64,
        )
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        parameter.grad = replay.clone()
        values = method.merge_component_gradients(
            (("adapter", parameter),),
            (action,),
            replay_combine_mode=method.REPLAY_COMBINE_MODE,
            base_replay_scale=0.025,
        )
        self.assertAlmostEqual(values["correction_ratio_q"], 0.9555751902699228)
        self.assertGreater(
            values["v16r5_action_normalized_margin"],
            method.MIN_FORMAL_NORMALIZED_MARGIN,
        )
        self.assertGreater(
            values["v16r5_source_normalized_margin"],
            method.MIN_FORMAL_NORMALIZED_MARGIN,
        )
        self.assertGreater(
            values["v16r5_actual_merged_action_descent_cosine"],
            method.MIN_ACTUAL_DESCENT_COSINE,
        )
        self.assertGreater(
            values["v16r5_actual_merged_source_descent_cosine"],
            method.MIN_ACTUAL_DESCENT_COSINE,
        )
        self.assertFalse(values["v16r5_step_attenuation_applied"])

    def test_tensor_merge_fails_before_mutation_at_antipodal_boundary(self):
        cosine = -1.0 + 0.5 * method.MIN_ANTIPODAL_GAP
        action = torch.tensor([1.0, 0.0], dtype=torch.float64)
        replay = torch.tensor(
            [cosine, math.sqrt(1.0 - cosine * cosine)], dtype=torch.float64
        )
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        parameter.grad = replay.clone()
        before = parameter.grad.clone()
        with self.assertRaisesRegex(
            method.base.OnlineAnchorTrainingError,
            "antipodal gap is infeasible",
        ):
            method.merge_component_gradients(
                (("adapter", parameter),),
                (action,),
                replay_combine_mode=method.REPLAY_COMBINE_MODE,
                base_replay_scale=0.025,
            )
        self.assertTrue(torch.equal(parameter.grad, before))

    def test_optimizer_applies_only_the_registered_global_attenuation(self):
        cosine = -0.995
        action = torch.tensor([1.0, 0.0], dtype=torch.float64)
        replay = torch.tensor(
            [cosine, math.sqrt(1.0 - cosine * cosine)], dtype=torch.float64
        )
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        parameter.grad = replay.clone()
        method.merge_component_gradients(
            (("adapter", parameter),),
            (action,),
            replay_combine_mode=method.REPLAY_COMBINE_MODE,
            base_replay_scale=0.025,
        )
        before = (parameter.detach().clone(),)
        optimizer = method._make_attenuated_global_rms_sgd(
            (parameter,), lr=method.LEARNING_RATE
        )
        method.parent._ACTIVE_OPTIMIZER = optimizer
        optimizer.step()
        expected_gamma = math.sqrt(0.005 / 0.01)
        expected_delta = method.LEARNING_RATE * expected_gamma * math.sqrt(2.0)
        self.assertAlmostEqual(
            float((parameter.detach() - before[0]).norm().item()), expected_delta
        )
        values = method.actual_optimizer_update_probe(
            (("adapter", parameter),),
            before,
            (action,),
            (replay,),
            replay_combine_mode=method.REPLAY_COMBINE_MODE,
            step=1,
        )
        self.assertGreater(values["action_descent_cosine"], 1.0e-8)
        self.assertGreater(values["source_descent_cosine"], 1.0e-8)
        self.assertAlmostEqual(values["v16r5_step_attenuation_gamma"], expected_gamma)
        self.assertTrue(values["v16r5_step_attenuation_applied"])
        self.assertIsNone(method._PENDING_STEP_GEOMETRY)
        self.assertEqual(method._V16R5_AUDIT["actual_steps"], [1])


if __name__ == "__main__":
    unittest.main()
