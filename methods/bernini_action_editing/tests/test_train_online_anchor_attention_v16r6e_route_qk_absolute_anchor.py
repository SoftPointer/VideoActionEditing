from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v16r6e tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r6e_route_qk_absolute_anchor32 as e


class _Affine:
    weight = object()


class _Renderer:
    def named_modules(self):
        yield "", self
        for block in range(30):
            for attention in ("attn1", "attn2"):
                for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                    yield (
                        f"diff_dec.transformer.blocks.{block}."
                        f"{attention}.{projection}",
                        _Affine(),
                    )


def inherited_receipt(step: int = 32):
    return {
        "schema_version": "parent",
        "global_step": step,
        "complete": step == 32,
        "training_contract": {
            "method": "parent",
            "lora_scope": e.LORA_SCOPE,
            "lora_target_module_count": e.LORA_TARGET_MODULE_COUNT,
            "lora_target_modules_sha256": e.TARGET_MODULES_SHA256,
            "trainable_parameter_count": e.LORA_PARAMETERS,
            "same_action_route_off_gradient_enabled": False,
            "same_action_student_delta_gradient_mode": (
                e.LEGACY_DELTA_GRADIENT_MODE
            ),
            "same_action_route_off_absolute_anchor_enabled": True,
            "same_action_route_off_absolute_anchor_weight": (
                e.ABSOLUTE_ANCHOR_WEIGHT
            ),
            "same_action_route_off_absolute_anchor_mode": e.ABSOLUTE_ANCHOR_MODE,
            "single_continuous_fresh_from_base_exact644_run": True,
            "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
        },
        "v16_full644_summary": {
            "all_full644_rows_targeted_exactly_once": False,
        },
        "route_off_absolute_anchor_diagnostic": {
            "applicable": True,
            "mode": e.ABSOLUTE_ANCHOR_MODE,
            "micro_count": 2,
            "weight": e.ABSOLUTE_ANCHOR_WEIGHT,
            "student_delta_gradient_mode": e.LEGACY_DELTA_GRADIENT_MODE,
            "mean_fm": 0.5,
            "weighted_mean_fm": 0.0125,
        },
        "memory_gate": {"capture_phase": "parent", "passed": True},
    }


class V16R6ERouteQKAbsoluteAnchorTest(unittest.TestCase):
    def test_scope_is_exactly_the_v16r6b_44_module_closure(self):
        names = e.select_route_attn1_qk_target_names(_Renderer())
        self.assertEqual(len(names), 44)
        self.assertEqual(e.base.legacy.object_sha256(list(names)), e.TARGET_MODULES_SHA256)
        self.assertEqual(e.LORA_TRAINABLE_TENSOR_COUNT, 88)
        self.assertEqual(e.LORA_PARAMETERS, 34_603_008)
        self.assertTrue(all(".attn1." in name for name in names))
        self.assertTrue(all(name.endswith((".to_q", ".to_k")) for name in names))

    def test_absolute_anchor_restores_route_off_common_mode(self):
        common = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        teacher = torch.zeros((1, 1, 21, 1, 1), dtype=torch.float64)
        anchor = e.base.same_action_route_off_absolute_anchor_loss(
            student_route_off_prediction=common.expand_as(teacher),
            frozen_route_off_teacher=teacher,
        )
        (e.ABSOLUTE_ANCHOR_WEIGHT * anchor).backward()
        self.assertGreater(float(common.grad.item()), 0.0)
        self.assertLess(
            float((common - 0.1 * common.grad).detach().item()),
            float(common.detach().item()),
        )

    def test_validate_uses_v16r5_shadow_then_requires_debug32(self):
        seen = {}

        def observe(shadow):
            seen.update(vars(shadow))

        args = SimpleNamespace(
            max_steps=32,
            learning_rate=1e-6,
            output=Path("/tmp/train_v16r6e_probe"),
        )
        with mock.patch.object(e, "_PARENT_VALIDATE_ARGS", side_effect=observe):
            e.validate_args(args)
        self.assertEqual(seen["max_steps"], 644)
        self.assertEqual(seen["learning_rate"], 1e-6)
        with mock.patch.object(e, "_PARENT_VALIDATE_ARGS", return_value=None):
            with self.assertRaisesRegex(e.base.OnlineAnchorTrainingError, "max-steps=32"):
                e.validate_args(SimpleNamespace(**{**vars(args), "max_steps": 31}))

    def test_receipt_binds_scope_and_anchor_as_one_composed_diagnostic(self):
        with mock.patch.object(
            e, "_PARENT_CHECKPOINT_RECEIPT", return_value=inherited_receipt()
        ):
            receipt = e.checkpoint_receipt(args=object())
        contract = receipt["training_contract"]
        scope = receipt["v16r6e_lora_scope_contract"]
        anchor = receipt["v16r6e_absolute_route_off_anchor_contract"]
        self.assertFalse(receipt["complete"])
        self.assertFalse(receipt["exact644_training_complete"])
        self.assertEqual(scope["target_module_count"], 44)
        self.assertEqual(scope["trainable_tensor_count"], 88)
        self.assertFalse(contract["lora_nonroute_blocks_trainable"])
        self.assertFalse(contract["lora_attn2_trainable"])
        self.assertTrue(contract["absolute_common_mode_fm_preservation_objective_added"])
        self.assertEqual(anchor["weight"], 0.025)
        self.assertTrue(anchor["teacher_detached"])
        self.assertEqual(
            anchor["sole_changed_training_variable_from_v16r6b"],
            e.CHANGED_VARIABLE,
        )
        self.assertTrue(
            receipt["memory_gate"][
                "absolute_route_off_anchor_training_allocations_included"
            ]
        )

    def test_main_installs_and_restores_both_scoped_changes(self):
        base_names = (
            "LORA_SCOPE",
            "LORA_TARGET_MODULE_COUNT",
            "LORA_TRAINABLE_TENSOR_COUNT",
            "LORA_PARAMETERS",
            "select_lora_target_names",
            "SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED",
            "SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT",
            "SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_MODE",
        )
        originals = {name: getattr(e.base, name) for name in base_names}

        def observe(parent, argv):
            self.assertIs(parent, e.parent)
            self.assertEqual(e.base.LORA_SCOPE, e.LORA_SCOPE)
            self.assertEqual(e.base.LORA_TARGET_MODULE_COUNT, 44)
            self.assertIs(
                e.base.select_lora_target_names,
                e.select_route_attn1_qk_target_names,
            )
            self.assertTrue(e.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED)
            self.assertEqual(e.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT, 0.025)
            self.assertEqual(
                e.base.SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_MODE,
                e.ABSOLUTE_ANCHOR_MODE,
            )
            return 37

        with mock.patch.object(e.debug, "run_v16r5_debug32", side_effect=observe):
            self.assertEqual(e.main([]), 37)
        for name, value in originals.items():
            self.assertIs(getattr(e.base, name), value)


if __name__ == "__main__":
    unittest.main()
