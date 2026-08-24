from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v16r6d tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r6d_absolute_anchor32 as d


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
            "same_action_route_off_gradient_enabled": False,
            "same_action_student_delta_gradient_mode": (
                d.LEGACY_DELTA_GRADIENT_MODE
            ),
            "same_action_route_off_absolute_anchor_enabled": True,
            "same_action_route_off_absolute_anchor_weight": (
                d.ABSOLUTE_ANCHOR_WEIGHT
            ),
            "same_action_route_off_absolute_anchor_mode": d.ABSOLUTE_ANCHOR_MODE,
            "single_continuous_fresh_from_base_exact644_run": True,
            "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
        },
        "v16_full644_summary": {
            "all_full644_rows_targeted_exactly_once": False,
        },
        "route_off_absolute_anchor_diagnostic": {
            "applicable": True,
            "mode": d.ABSOLUTE_ANCHOR_MODE,
            "micro_count": 2,
            "weight": d.ABSOLUTE_ANCHOR_WEIGHT,
            "student_delta_gradient_mode": d.LEGACY_DELTA_GRADIENT_MODE,
            "mean_fm": 0.5,
            "weighted_mean_fm": 0.0125,
        },
        "memory_gate": {
            "capture_phase": "parent",
            "passed": True,
        },
    }


class V16R6DAbsoluteAnchorTest(unittest.TestCase):
    def test_toy_anchor_restores_common_mode_at_zero_delta_error(self):
        common = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        route = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        teacher_action = torch.ones((1, 1, 21, 1, 1), dtype=torch.float64)
        teacher_source = torch.zeros_like(teacher_action)

        route_on = (common + route).expand_as(teacher_action)
        route_off_detached = common.detach().expand_as(teacher_action)
        delta = d.base.real_source_routed_teacher_delta_loss(
            action_prediction=route_on,
            source_prediction=route_off_detached,
            teacher_action=teacher_action,
            teacher_source=teacher_source,
            mode="raw",
            name="toy legacy delta",
        )
        delta.backward()
        self.assertEqual(float(common.grad.item()), 0.0)
        self.assertEqual(float(route.grad.item()), 0.0)

        anchor = d.base.same_action_route_off_absolute_anchor_loss(
            student_route_off_prediction=common.expand_as(teacher_source),
            frozen_route_off_teacher=teacher_source,
        )
        (d.ABSOLUTE_ANCHOR_WEIGHT * anchor).backward()
        self.assertGreater(float(common.grad.item()), 0.0)
        self.assertEqual(float(route.grad.item()), 0.0)
        # SGD subtracts this positive derivative, restoring common toward zero.
        self.assertLess(
            float((common - 0.1 * common.grad).detach().item()),
            float(common.detach().item()),
        )

    def test_toy_route_delta_gradient_is_retained_without_c_cancellation(self):
        common = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        route = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
        teacher_action = torch.ones((1, 1, 21, 1, 1), dtype=torch.float64)
        teacher_source = torch.zeros_like(teacher_action)

        delta = d.base.real_source_routed_teacher_delta_loss(
            action_prediction=(common + route).expand_as(teacher_action),
            source_prediction=common.detach().expand_as(teacher_action),
            teacher_action=teacher_action,
            teacher_source=teacher_source,
            mode="raw",
            name="toy retained route gradient",
        )
        delta.backward()
        route_before_anchor = route.grad.detach().clone()
        common_before_anchor = common.grad.detach().clone()
        self.assertNotEqual(float(route_before_anchor.item()), 0.0)
        # D does not run C's denominator loss; its independent absolute spring
        # cannot consume or cancel the route-dependent numerator gradient.
        anchor = d.base.same_action_route_off_absolute_anchor_loss(
            student_route_off_prediction=common.expand_as(teacher_source),
            frozen_route_off_teacher=teacher_source,
        )
        (d.ABSOLUTE_ANCHOR_WEIGHT * anchor).backward()
        self.assertEqual(float(route.grad.item()), float(route_before_anchor.item()))
        self.assertEqual(float(common.grad.item()), float(common_before_anchor.item()))

    def test_control_flow_requires_same_action_and_explicit_bounded_weight(self):
        with mock.patch.object(
            d.base, "SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED", False
        ):
            self.assertFalse(
                d.base.requires_same_action_route_off_absolute_anchor(
                    same_action_route_only=True
                )
            )
        with mock.patch.object(
            d.base, "SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED", True
        ), mock.patch.object(
            d.base,
            "SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT",
            d.ABSOLUTE_ANCHOR_WEIGHT,
        ):
            self.assertTrue(
                d.base.requires_same_action_route_off_absolute_anchor(
                    same_action_route_only=True
                )
            )
            with self.assertRaisesRegex(
                d.base.OnlineAnchorTrainingError, "contract differs"
            ):
                d.base.requires_same_action_route_off_absolute_anchor(
                    same_action_route_only=False
                )

    def test_d_keeps_full_lora_lr_and_legacy_delta_gradient(self):
        self.assertEqual(d.LEARNING_RATE, 1e-6)
        self.assertEqual(d.ABSOLUTE_ANCHOR_WEIGHT, 0.025)
        self.assertEqual(d.base.LORA_SCOPE, "all_30_blocks_attn1_attn2_qkvo")
        self.assertEqual(d.base.LORA_TARGET_MODULE_COUNT, 240)
        self.assertEqual(d.base.LORA_TRAINABLE_TENSOR_COUNT, 480)
        self.assertEqual(d.base.LORA_PARAMETERS, 188_743_680)
        self.assertFalse(d.base.SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED)
        self.assertEqual(
            d.CHANGED_VARIABLE,
            "same_state_route_off_absolute_common_mode_fm_anchor_only",
        )

    def test_validate_uses_v16r5_shadow_then_requires_debug32(self):
        seen = {}

        def observe(shadow):
            seen.update(vars(shadow))

        args = SimpleNamespace(
            max_steps=32,
            learning_rate=1e-6,
            output=Path("/tmp/train_v16r6d_probe"),
        )
        with mock.patch.object(d, "_PARENT_VALIDATE_ARGS", side_effect=observe):
            d.validate_args(args)
        self.assertEqual(seen["max_steps"], 644)
        self.assertEqual(seen["learning_rate"], 1e-6)
        with mock.patch.object(d, "_PARENT_VALIDATE_ARGS", return_value=None):
            with self.assertRaisesRegex(
                d.base.OnlineAnchorTrainingError, "max-steps=32"
            ):
                d.validate_args(SimpleNamespace(**{**vars(args), "max_steps": 31}))

    def test_receipt_binds_absolute_anchor_and_excludes_c(self):
        with mock.patch.object(
            d, "_PARENT_CHECKPOINT_RECEIPT", return_value=inherited_receipt()
        ):
            receipt = d.checkpoint_receipt(args=object())
        contract = receipt["training_contract"]
        d_contract = receipt["v16r6d_absolute_route_off_anchor_contract"]
        self.assertFalse(receipt["complete"])
        self.assertFalse(receipt["exact644_training_complete"])
        self.assertFalse(contract["same_action_route_off_gradient_enabled"])
        self.assertEqual(
            contract["same_action_student_delta_jacobian"],
            "J_route_on_only_legacy",
        )
        self.assertTrue(
            contract["absolute_common_mode_fm_preservation_objective_added"]
        )
        self.assertFalse(contract["decoded_source_preservation_claimed"])
        self.assertEqual(d_contract["weight"], 0.025)
        self.assertTrue(d_contract["teacher_detached"])
        self.assertFalse(d_contract["decoded_source_preservation_claimed"])
        self.assertEqual(
            contract["action_component_gradient_contains"],
            [
                "legacy_route_on_only_student_delta",
                "weighted_absolute_route_off_common_mode_anchor",
            ],
        )
        self.assertTrue(
            receipt["memory_gate"][
                "absolute_route_off_anchor_training_allocations_included"
            ]
        )

    def test_main_patches_and_restores_only_absolute_anchor(self):
        old_enabled = d.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED
        old_weight = d.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT
        old_mode = d.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_MODE
        old_c_enabled = d.base.SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED
        old_lr = d.parent.LEARNING_RATE

        def observe(parent, argv):
            self.assertIs(parent, d.parent)
            self.assertTrue(
                d.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED
            )
            self.assertEqual(
                d.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT, 0.025
            )
            self.assertEqual(
                d.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_MODE,
                d.ABSOLUTE_ANCHOR_MODE,
            )
            self.assertEqual(
                d.base.SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED, old_c_enabled
            )
            self.assertEqual(d.parent.LEARNING_RATE, old_lr)
            return 31

        with mock.patch.object(d.debug, "run_v16r5_debug32", side_effect=observe):
            self.assertEqual(d.main([]), 31)
        self.assertEqual(
            d.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED, old_enabled
        )
        self.assertEqual(
            d.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT, old_weight
        )
        self.assertEqual(
            d.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_MODE, old_mode
        )
        self.assertEqual(
            d.base.SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED, old_c_enabled
        )


if __name__ == "__main__":
    unittest.main()
