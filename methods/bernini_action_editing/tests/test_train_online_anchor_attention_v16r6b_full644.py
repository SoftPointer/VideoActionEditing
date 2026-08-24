from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    import torch  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v16r6b exact644 tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r6b_route_qk_full644 as target


class _Affine:
    weight = object()


class _Renderer:
    def named_modules(self):
        yield "", self
        for block in range(30):
            for attention in ("attn1", "attn2"):
                for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                    yield f"diff_dec.transformer.blocks.{block}.{attention}.{projection}", _Affine()


class V16R6BExact644Test(unittest.TestCase):
    def test_scope_is_identical_to_winning_s32_variant(self):
        names = target.select_route_attn1_qk_target_names(_Renderer())
        self.assertEqual(len(names), 44)
        self.assertEqual(target.base.legacy.object_sha256(list(names)), target.TARGET_MODULES_SHA256)
        self.assertEqual(target.LORA_TRAINABLE_TENSOR_COUNT, 88)
        self.assertEqual(target.LORA_PARAMETERS, 34_603_008)

    def test_validation_preserves_parent_contract_and_requires_exact644(self):
        seen = {}

        def observe(shadow):
            seen.update(vars(shadow))

        args = SimpleNamespace(
            max_steps=644,
            learning_rate=1e-6,
            output=Path("/tmp/v16r6b-route-qk-exact644"),
            method_source_archive_sha256="a" * 64,
            method_source_revision=target.overlay_revision("a" * 64),
        )
        with mock.patch.object(target, "_PARENT_VALIDATE_ARGS", side_effect=observe):
            target.validate_args(args)
        self.assertEqual(seen["max_steps"], 644)
        self.assertIn("v16r5-contract-shadow", str(seen["output"]))
        with mock.patch.object(target, "_PARENT_VALIDATE_ARGS", return_value=None):
            with self.assertRaisesRegex(target.base.OnlineAnchorTrainingError, "max-steps=644"):
                target.validate_args(SimpleNamespace(**{**vars(args), "max_steps": 32}))

    def test_receipt_binds_exact_scope_and_fresh_initialization(self):
        inherited = {
            "schema_version": "parent",
            "global_step": 644,
            "complete": True,
            "exact644_training_complete": True,
            "training_contract": {
                "method": "parent",
                "lora_scope": target.LORA_SCOPE,
                "lora_target_module_count": target.LORA_TARGET_MODULE_COUNT,
                "lora_target_modules_sha256": target.TARGET_MODULES_SHA256,
                "trainable_parameter_count": target.LORA_PARAMETERS,
            },
        }
        with mock.patch.object(target, "_PARENT_CHECKPOINT_RECEIPT", return_value=inherited):
            receipt = target.checkpoint_receipt(
                args=SimpleNamespace(
                    method_source_archive_sha256="a" * 64,
                    method_source_revision=target.overlay_revision("a" * 64),
                )
            )
        contract = receipt["training_contract"]
        self.assertTrue(receipt["complete"])
        self.assertTrue(receipt["exact644_training_complete"])
        self.assertEqual(receipt["schema_version"], target.RECEIPT_SCHEMA)
        self.assertTrue(contract["fresh_from_frozen_base"])
        self.assertFalse(contract["s32_checkpoint_used_as_initialization"])
        self.assertEqual(receipt["v16r6b_full644_scope_contract"]["optimizer_step_budget"], 644)
        self.assertEqual(
            receipt["v16r6b_full644_scope_contract"]["trainer_overlay_sha256"],
            target.trainer_source_sha256(),
        )

    def test_main_installs_and_restores_only_promoted_scope(self):
        original_scope = target.base.LORA_SCOPE
        original_selector = target.base.select_lora_target_names

        def observe(_argv):
            self.assertEqual(target.parent.METHOD, target.METHOD)
            self.assertEqual(target.base.LORA_SCOPE, target.LORA_SCOPE)
            self.assertIs(target.base.select_lora_target_names, target.select_route_attn1_qk_target_names)
            return 29

        with mock.patch.object(target.parent, "main", side_effect=observe):
            self.assertEqual(target.main([]), 29)
        self.assertEqual(target.base.LORA_SCOPE, original_scope)
        self.assertIs(target.base.select_lora_target_names, original_selector)


if __name__ == "__main__":
    unittest.main()
