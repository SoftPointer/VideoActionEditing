from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v16r6c tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r6c_two_sided_delta32 as c


def inherited_receipt(step: int = 32):
    return {
        "schema_version": "parent",
        "global_step": step,
        "complete": step == 32,
        "training_contract": {
            "method": "parent",
            "lora_scope": "all_30_blocks_attn1_attn2_qkvo",
            "lora_target_module_count": 240,
            "trainable_parameter_count": 188_743_680,
            "same_action_route_off_gradient_enabled": True,
            "same_action_student_delta_gradient_mode": c.GRADIENT_MODE,
            "single_continuous_fresh_from_base_exact644_run": True,
            "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
        },
        "v16_full644_summary": {
            "all_full644_rows_targeted_exactly_once": False,
        },
        "source_absorption_diagnostic": {
            "applicable": True,
            "micro_count": 2,
            "defined_micro_count": 2,
        },
    }


class V16R6CTwoSidedDeltaTest(unittest.TestCase):
    def test_toy_common_mode_cancels_and_route_dependent_gradient_remains(self):
        common = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        route = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
        teacher_action = torch.ones((1, 1, 21, 1, 1), dtype=torch.float64)
        teacher_source = torch.zeros_like(teacher_action)

        route_on = (common + route).expand_as(teacher_action)
        route_off_detached = common.detach().expand_as(teacher_action)
        action_side = c.base.real_source_routed_teacher_delta_loss(
            action_prediction=route_on,
            source_prediction=route_off_detached,
            teacher_action=teacher_action,
            teacher_source=teacher_source,
            mode="raw",
            name="toy action side",
        )
        action_side.backward()
        one_sided_common = common.grad.detach().clone()
        one_sided_route = route.grad.detach().clone()
        self.assertNotEqual(float(one_sided_common.item()), 0.0)
        self.assertNotEqual(float(one_sided_route.item()), 0.0)

        route_off_graph = common.expand_as(teacher_action)
        source_side = c.base.real_source_routed_teacher_delta_loss(
            action_prediction=route_on.detach(),
            source_prediction=route_off_graph,
            teacher_action=teacher_action,
            teacher_source=teacher_source,
            mode="raw",
            name="toy source side",
        )
        source_side.backward()

        self.assertEqual(float(common.grad.item()), 0.0)
        self.assertEqual(float(route.grad.item()), float(one_sided_route.item()))
        self.assertNotEqual(float(route.grad.item()), 0.0)

    def test_same_action_control_flow_selects_action_record_only_when_enabled(self):
        action = {"iid": "action"}
        paired = {"iid": "paired"}
        token = object()
        with mock.patch.object(
            c.base, "SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED", False
        ):
            self.assertFalse(
                c.base.requires_sequential_source_side_backward(
                    paired_action_loss=token, same_action_route_only=True
                )
            )
            with self.assertRaisesRegex(
                c.base.OnlineAnchorTrainingError, "was not enabled"
            ):
                c.base.sequential_source_side_record(
                    same_action_route_only=True,
                    action_record=action,
                    paired_source=paired,
                )
        with mock.patch.object(
            c.base, "SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED", True
        ):
            self.assertTrue(
                c.base.requires_sequential_source_side_backward(
                    paired_action_loss=token, same_action_route_only=True
                )
            )
            self.assertIs(
                c.base.sequential_source_side_record(
                    same_action_route_only=True,
                    action_record=action,
                    paired_source=paired,
                ),
                action,
            )
            self.assertIs(
                c.base.sequential_source_side_record(
                    same_action_route_only=False,
                    action_record=action,
                    paired_source=paired,
                ),
                paired,
            )

    def test_c_keeps_full_lora_lr_and_all_non_gradient_variables(self):
        self.assertEqual(c.LEARNING_RATE, 1e-6)
        self.assertEqual(c.base.LORA_SCOPE, "all_30_blocks_attn1_attn2_qkvo")
        self.assertEqual(c.base.LORA_TARGET_MODULE_COUNT, 240)
        self.assertEqual(c.base.LORA_TRAINABLE_TENSOR_COUNT, 480)
        self.assertEqual(c.base.LORA_PARAMETERS, 188_743_680)
        self.assertEqual(
            c.CHANGED_VARIABLE,
            "same_action_student_delta_gradient_estimator_only",
        )

    def test_validate_uses_v16r5_shadow_then_requires_debug32(self):
        seen = {}

        def observe(shadow):
            seen.update(vars(shadow))

        args = SimpleNamespace(
            max_steps=32,
            learning_rate=1e-6,
            output=Path("/tmp/train_v16r6c_probe"),
        )
        with mock.patch.object(c, "_PARENT_VALIDATE_ARGS", side_effect=observe):
            c.validate_args(args)
        self.assertEqual(seen["max_steps"], 644)
        self.assertEqual(seen["learning_rate"], 1e-6)
        with mock.patch.object(c, "_PARENT_VALIDATE_ARGS", return_value=None):
            with self.assertRaisesRegex(
                c.base.OnlineAnchorTrainingError, "max-steps=32"
            ):
                c.validate_args(SimpleNamespace(**{**vars(args), "max_steps": 31}))

    def test_receipt_binds_two_sided_execution_and_non_exact32_status(self):
        with mock.patch.object(
            c, "_PARENT_CHECKPOINT_RECEIPT", return_value=inherited_receipt()
        ):
            receipt = c.checkpoint_receipt(args=object())
        contract = receipt["training_contract"]
        self.assertFalse(receipt["complete"])
        self.assertFalse(receipt["exact644_training_complete"])
        self.assertEqual(
            contract["same_action_student_delta_jacobian"],
            "J_route_on_minus_J_route_off",
        )
        self.assertTrue(contract["same_action_route_off_recomputed_with_grad"])
        self.assertFalse(contract["simultaneous_two_30_block_graph_retention"])
        self.assertFalse(contract["learning_rate_changed_from_v16r5"])
        self.assertFalse(contract["lora_scope_changed_from_v16r5"])

    def test_main_patches_and_restores_only_gradient_estimator(self):
        old_enabled = c.base.SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED
        old_mode = c.base.SAME_ACTION_STUDENT_DELTA_GRADIENT_MODE
        old_scope = c.base.LORA_SCOPE
        old_lr = c.parent.LEARNING_RATE

        def observe(parent, argv):
            self.assertIs(parent, c.parent)
            self.assertTrue(c.base.SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED)
            self.assertEqual(
                c.base.SAME_ACTION_STUDENT_DELTA_GRADIENT_MODE,
                c.GRADIENT_MODE,
            )
            self.assertEqual(c.base.LORA_SCOPE, old_scope)
            self.assertEqual(c.parent.LEARNING_RATE, old_lr)
            return 29

        with mock.patch.object(c.debug, "run_v16r5_debug32", side_effect=observe):
            self.assertEqual(c.main([]), 29)
        self.assertEqual(
            c.base.SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED, old_enabled
        )
        self.assertEqual(c.base.SAME_ACTION_STUDENT_DELTA_GRADIENT_MODE, old_mode)
        self.assertEqual(c.base.LORA_SCOPE, old_scope)
        self.assertEqual(c.parent.LEARNING_RATE, old_lr)


if __name__ == "__main__":
    unittest.main()
