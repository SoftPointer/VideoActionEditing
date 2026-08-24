from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    import torch  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v16r6 online-anchor tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r6a_lr1e7_32 as a
import train_online_anchor_attention_full644_dynamic_static_v16r6b_route_qk32 as b
import train_online_anchor_attention_v16r6_debug_common as common


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


def inherited_receipt(
    *, step: int, scope: str, count: int, sha: str, params: int,
    epsilon: float = 1.0e-13,
):
    return {
        "schema_version": "parent",
        "global_step": step,
        "complete": step == 32,
        "scientific_claim_authorized": False,
        "training_contract": {
            "method": "parent",
            "lora_scope": scope,
            "lora_target_module_count": count,
            "lora_target_modules_sha256": sha,
            "trainable_parameter_count": params,
            "component_gradient_epsilon": epsilon,
            "single_continuous_fresh_from_base_exact644_run": True,
            "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
            "all_full644_rows_targeted_exactly_once": False,
        },
        "v16_full644_summary": {
            "manifest_row_count": 644,
            "target_prefix_row_count": step,
            "all_full644_rows_targeted_exactly_once": False,
        },
        "v16r5_source_descent_summary": {
            "optimizer": a.OPTIMIZER,
            "optimizer_current_effective_active_coordinate_rms": a.LEARNING_RATE,
        },
        "component_gradient_probes": {
            "action_objective": {"epsilon": epsilon},
            "raw_source_caption_trajectory_replay": {"epsilon": epsilon},
        },
    }


class V16R6BRouteQKScopeTest(unittest.TestCase):
    def test_exact_target_names_count_sha_and_forbidden_closure(self):
        names = b.select_route_attn1_qk_target_names(_Renderer())
        self.assertEqual(len(names), 44)
        self.assertEqual(len(set(names)), 44)
        self.assertEqual(
            b.base.legacy.object_sha256(list(names)), b.TARGET_MODULES_SHA256
        )
        self.assertEqual(
            names[:4],
            (
                "diff_dec.transformer.blocks.1.attn1.to_k",
                "diff_dec.transformer.blocks.1.attn1.to_q",
                "diff_dec.transformer.blocks.10.attn1.to_k",
                "diff_dec.transformer.blocks.10.attn1.to_q",
            ),
        )
        observed_blocks = set()
        for name in names:
            match = b.TARGET_PATTERN.fullmatch(name)
            self.assertIsNotNone(match)
            observed_blocks.add(int(match.group("block")))
            self.assertEqual(match.group("attention"), "attn1")
            self.assertIn(match.group("projection"), ("to_q", "to_k"))
            self.assertNotIn("attn2", name)
            self.assertNotIn("to_v", name)
            self.assertNotIn("to_out", name)
        self.assertEqual(observed_blocks, set(b.ROUTE_BLOCKS))
        self.assertTrue(observed_blocks.isdisjoint(set(b.base.CHECKPOINT_BLOCKS)))

    def test_parameter_and_tensor_counts_are_exact(self):
        per_rank256_square_projection = 2 * 1536 * 256
        self.assertEqual(b.LORA_TARGET_MODULE_COUNT, 22 * 2)
        self.assertEqual(b.LORA_TRAINABLE_TENSOR_COUNT, 2 * 44)
        self.assertEqual(
            b.LORA_PARAMETERS,
            b.LORA_TARGET_MODULE_COUNT * per_rank256_square_projection,
        )
        self.assertEqual(b.LORA_PARAMETERS, 34_603_008)
        self.assertEqual(b.LEARNING_RATE, 1.0e-6)

    def test_validate_uses_unchanged_v16r5_shadow_and_requires_debug32(self):
        seen = {}

        def observe(shadow):
            seen.update(vars(shadow))

        args = SimpleNamespace(
            max_steps=32,
            learning_rate=1.0e-6,
            output=Path("/tmp/train_v16r6b_probe"),
        )
        with mock.patch.object(b, "_PARENT_VALIDATE_ARGS", side_effect=observe):
            b.validate_args(args)
        self.assertEqual(seen["max_steps"], 644)
        self.assertEqual(seen["learning_rate"], 1.0e-6)
        self.assertIn("v16r5-contract-shadow", str(seen["output"]))
        with mock.patch.object(b, "_PARENT_VALIDATE_ARGS", return_value=None):
            with self.assertRaisesRegex(b.base.OnlineAnchorTrainingError, "max-steps=32"):
                b.validate_args(SimpleNamespace(**{**vars(args), "max_steps": 64}))

    def test_receipt_is_explicitly_non_exact644_and_binds_scope(self):
        inherited = inherited_receipt(
            step=32,
            scope=b.LORA_SCOPE,
            count=b.LORA_TARGET_MODULE_COUNT,
            sha=b.TARGET_MODULES_SHA256,
            params=b.LORA_PARAMETERS,
        )
        with mock.patch.object(
            b, "_PARENT_CHECKPOINT_RECEIPT", return_value=inherited
        ):
            receipt = b.checkpoint_receipt(args=object())
        contract = receipt["training_contract"]
        self.assertFalse(receipt["complete"])
        self.assertFalse(receipt["exact644_training_complete"])
        self.assertTrue(receipt["v16r6_debug_contract"]["debug_run_complete"])
        self.assertEqual(contract["lora_scope"], b.LORA_SCOPE)
        self.assertEqual(contract["lora_target_module_count"], 44)
        self.assertFalse(contract["lora_nonroute_blocks_trainable"])
        self.assertFalse(contract["lora_attn2_trainable"])
        self.assertFalse(contract["lora_value_or_output_trainable"])
        self.assertFalse(contract["learning_rate_changed_from_v16r5"])

    def test_main_temporarily_installs_only_scope_change(self):
        base_names = (
            "LORA_SCOPE",
            "LORA_TARGET_MODULE_COUNT",
            "LORA_TRAINABLE_TENSOR_COUNT",
            "LORA_PARAMETERS",
            "select_lora_target_names",
        )
        originals = {name: getattr(b.base, name) for name in base_names}

        def observe(parent, argv):
            self.assertIs(parent, b.parent)
            self.assertEqual(b.base.LORA_SCOPE, b.LORA_SCOPE)
            self.assertEqual(b.base.LORA_TARGET_MODULE_COUNT, 44)
            self.assertEqual(b.base.LORA_TRAINABLE_TENSOR_COUNT, 88)
            self.assertEqual(b.base.LORA_PARAMETERS, 34_603_008)
            self.assertIs(
                b.base.select_lora_target_names,
                b.select_route_attn1_qk_target_names,
            )
            self.assertEqual(parent.LEARNING_RATE, 1.0e-6)
            return 19

        with mock.patch.object(b.debug, "run_v16r5_debug32", side_effect=observe):
            self.assertEqual(b.main([]), 19)
        for name, value in originals.items():
            self.assertIs(getattr(b.base, name), value)


class V16R6AActiveRmsLearningRateTest(unittest.TestCase):
    def test_full_lora_target_set_count_sha_and_parameters_are_unchanged(self):
        names = tuple(a.base.legacy.select_attention_projection_names(_Renderer()))
        self.assertEqual(len(names), 240)
        self.assertEqual(
            a.base.legacy.object_sha256(list(names)),
            "d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a",
        )
        self.assertTrue(any("attn2" in name for name in names))
        self.assertTrue(any("to_v" in name for name in names))
        self.assertTrue(any("to_out.0" in name for name in names))
        self.assertTrue(any("blocks.0." in name for name in names))
        self.assertEqual(a.base.LORA_TARGET_MODULE_COUNT, 240)
        self.assertEqual(a.base.LORA_TRAINABLE_TENSOR_COUNT, 480)
        self.assertEqual(a.base.LORA_PARAMETERS, 188_743_680)

    def test_learning_rate_is_exactly_one_tenth_v16r5(self):
        self.assertEqual(a.LEARNING_RATE, 1.0e-7)
        self.assertEqual(a.V16R5_LEARNING_RATE, 1.0e-6)
        self.assertEqual(a.LEARNING_RATE / a.V16R5_LEARNING_RATE, 0.1)
        self.assertEqual(a.V16R5_COMPONENT_GRADIENT_EPSILON, 1e-12)
        self.assertEqual(a.COMPONENT_GRADIENT_EPSILON_SCALE, 0.1)
        self.assertEqual(a.COMPONENT_GRADIENT_EPSILON, 1e-13)

    def test_scale_equivalent_epsilon_avoids_pure_lr_false_negative(self):
        parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        parameter.grad = torch.tensor([5e-12], dtype=torch.float64)
        base_probe = a.base.component_gradient_probe(
            (("layer.lora_B.weight", parameter),), epsilon=1e-12
        )
        self.assertEqual(base_probe["epsilon_active_tensor_count"], 1)

        parameter.grad = torch.tensor([5e-13], dtype=torch.float64)
        fixed_probe = a.base.component_gradient_probe(
            (("layer.lora_B.weight", parameter),), epsilon=1e-12
        )
        scaled_probe = a.base.component_gradient_probe(
            (("layer.lora_B.weight", parameter),), epsilon=1e-13
        )
        self.assertEqual(fixed_probe["nonzero_tensor_count"], 1)
        self.assertEqual(fixed_probe["epsilon_active_tensor_count"], 0)
        self.assertEqual(scaled_probe["epsilon_active_tensor_count"], 1)

    def test_validate_uses_v16r5_shadow_then_requires_lr1e7_debug32(self):
        seen = {}

        def observe(shadow):
            seen.update(vars(shadow))
            self.assertEqual(a.parent.LEARNING_RATE, a.V16R5_LEARNING_RATE)

        args = SimpleNamespace(
            max_steps=32,
            learning_rate=1.0e-7,
            output=Path("/tmp/train_v16r6a_probe"),
        )
        with mock.patch.object(a, "_PARENT_VALIDATE_ARGS", side_effect=observe):
            a.validate_args(args)
        self.assertEqual(seen["max_steps"], 644)
        self.assertEqual(seen["learning_rate"], 1.0e-6)
        with mock.patch.object(a, "_PARENT_VALIDATE_ARGS", return_value=None):
            with self.assertRaisesRegex(a.base.OnlineAnchorTrainingError, "learning rate 1e-7"):
                a.validate_args(SimpleNamespace(**{**vars(args), "learning_rate": 1e-6}))

    def test_receipt_binds_lr_and_remains_non_exact644(self):
        inherited = inherited_receipt(
            step=32,
            scope="all_30_blocks_attn1_attn2_qkvo",
            count=240,
            sha="d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a",
            params=188_743_680,
        )
        with mock.patch.object(
            a, "_PARENT_CHECKPOINT_RECEIPT", return_value=inherited
        ):
            receipt = a.checkpoint_receipt(args=object())
        contract = receipt["training_contract"]
        self.assertFalse(receipt["complete"])
        self.assertFalse(receipt["terminal_full644_checkpoint"])
        self.assertEqual(contract["optimizer_scalar_learning_rate"], 1e-7)
        self.assertEqual(
            receipt["v16r6a_learning_rate_contract"]["ratio_to_v16r5"], 0.1
        )
        self.assertFalse(contract["lora_scope_changed_from_v16r5"])
        self.assertEqual(contract["lora_target_module_count"], 240)
        audit = receipt["v16r6a_scale_equivalent_gradient_audit"]
        self.assertEqual(audit["v16r5_absolute_epsilon"], 1e-12)
        self.assertEqual(audit["effective_absolute_epsilon"], 1e-13)
        self.assertFalse(audit["support_requirement_changed"])

    def test_main_temporarily_changes_lr_not_target_scope(self):
        original_selector = a.base.select_lora_target_names
        original_scope = a.base.LORA_SCOPE
        original_epsilon = a.base.COMPONENT_GRADIENT_EPSILON

        def observe(parent, argv):
            self.assertEqual(parent.LEARNING_RATE, 1e-7)
            self.assertEqual(parent.parent.LEARNING_RATE, 1e-7)
            self.assertEqual(parent.OPTIMIZER, a.OPTIMIZER)
            self.assertIs(a.base.select_lora_target_names, original_selector)
            self.assertEqual(a.base.LORA_SCOPE, original_scope)
            self.assertEqual(a.base.COMPONENT_GRADIENT_EPSILON, 1e-13)
            return 23

        with mock.patch.object(a.debug, "run_v16r5_debug32", side_effect=observe):
            self.assertEqual(a.main([]), 23)
        self.assertEqual(a.parent.LEARNING_RATE, a.V16R5_LEARNING_RATE)
        self.assertIs(a.base.select_lora_target_names, original_selector)
        self.assertEqual(a.base.COMPONENT_GRADIENT_EPSILON, original_epsilon)


class V16R6CommonReceiptTest(unittest.TestCase):
    def test_step_outside_debug_prefix_is_rejected(self):
        receipt = inherited_receipt(
            step=33,
            scope="scope",
            count=1,
            sha="0" * 64,
            params=1,
        )
        with self.assertRaisesRegex(ValueError, "outside 1..32"):
            common.decorate_debug_receipt(
                receipt,
                method="m",
                schema="s",
                variant="v",
                changed_variable="x",
            )


if __name__ == "__main__":
    unittest.main()
