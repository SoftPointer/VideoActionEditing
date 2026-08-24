from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v16r3 online-anchor tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r3 as method


WORKER = ROOT / "scripts/auh_train_online_anchor_full644_dynamic_static_v16r3.sh"
CONTROLLER = ROOT / "scripts/auh_launch_online_anchor_full644_dynamic_static_v16r3.sh"


class Full644DynamicStaticV16R3Test(unittest.TestCase):
    def setUp(self) -> None:
        method._RUNTIME_AUDIT = method._empty_runtime_audit()
        method.v16r2._ACTIVE_MAX_GRAD_NORM = None

    def tearDown(self) -> None:
        method.v16r2._ACTIVE_MAX_GRAD_NORM = None

    def test_zero_rms_runtime_operator_contract(self):
        method._validate_zero_rms_operator()
        self.assertEqual(
            method.qk.QK_ONLY_ZERO_RMS_BACKWARD_POLICY,
            method.ZERO_RMS_POLICY,
        )

    def test_validation_keeps_v16_contract_and_requires_v16r3_namespace(self):
        good = SimpleNamespace(output="/tmp/fresh-v16r3-s644", max_grad_norm=10.0)
        with mock.patch.object(method, "_V16_VALIDATE_ARGS") as inherited:
            method.validate_args(good)
            inherited.assert_called_once_with(good)
            self.assertEqual(method.v16r2._ACTIVE_MAX_GRAD_NORM, 10.0)
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(
                    SimpleNamespace(
                        output="/tmp/fresh-v16r2-s644", max_grad_norm=10.0
                    )
                )
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(
                    SimpleNamespace(
                        output="/tmp/fresh-v16r3-s644", max_grad_norm=9.0
                    )
                )

    def test_s279_builder_binds_original_iid_seed_and_timestep_sequence(self):
        row = {"iid": method.S279_TARGET_IID, "family": method.S279_TARGET_FAMILY}

        def inherited(**kwargs):
            expected = next(
                item
                for item in method.S279_EXPECTED_CALLS
                if int(item["seed"]) == int(kwargs["seed"])
            )
            return (
                {"timestep": float(expected["timestep"])},
                {"timestep": float(expected["timestep"])},
            )

        with mock.patch.object(method, "_V16_BUILD_REAL_SOURCE", side_effect=inherited):
            for expected in method.S279_EXPECTED_CALLS:
                method.build_real_source_paired_records_full644_v16r3(
                    anchor_row=row,
                    real_sources={},
                    transform=object(),
                    mean=object(),
                    std=object(),
                    seed=int(expected["seed"]),
                )
        self.assertEqual(
            method._RUNTIME_AUDIT["s279_builder_calls"],
            list(method.S279_EXPECTED_CALLS),
        )

    def test_s279_builder_rejects_seed_or_timestep_drift_immediately(self):
        row = {"iid": method.S279_TARGET_IID, "family": method.S279_TARGET_FAMILY}
        with mock.patch.object(
            method,
            "_V16_BUILD_REAL_SOURCE",
            return_value=({"timestep": 999.0}, {"timestep": 999.0}),
        ):
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.build_real_source_paired_records_full644_v16r3(
                    anchor_row=row,
                    real_sources={},
                    transform=object(),
                    mean=object(),
                    std=object(),
                    seed=1656484053,
                )
        self.assertEqual(method._RUNTIME_AUDIT["s279_builder_calls"], [])

    @staticmethod
    def inherited_receipt(step: int):
        return {
            "schema_version": method.v16r2.RECEIPT_SCHEMA,
            "global_step": step,
            "training_contract": {"method": method.v16r2.METHOD},
            "anchor_cache": {
                "qk_only_zero_rms_backward_policy": method.ZERO_RMS_POLICY
            },
        }

    def test_receipt_before_s279_records_fixed_policy_without_false_coverage(self):
        with mock.patch.object(
            method,
            "_V16R2_CHECKPOINT_RECEIPT",
            return_value=self.inherited_receipt(256),
        ):
            receipt = method.checkpoint_receipt(args=object())
        summary = receipt["v16r3_zero_rms_backward_summary"]
        contract = receipt["training_contract"]
        self.assertEqual(receipt["schema_version"], method.RECEIPT_SCHEMA)
        self.assertEqual(contract["method"], method.METHOD)
        self.assertFalse(contract["s279_endpoint_canary_covered"])
        self.assertFalse(summary["s279_endpoint_canary"]["covered_by_checkpoint"])
        self.assertEqual(summary["s279_endpoint_canary"]["observed_calls"], [])
        self.assertFalse(summary["loss_scale_changed"])
        self.assertFalse(summary["seed_or_timestep_changed"])
        self.assertFalse(summary["sample_retry_or_skip"])
        self.assertFalse(summary["component_preallreduce_finite_gate_relaxed"])

    def test_receipt_after_s279_requires_exact_observed_canary(self):
        method._RUNTIME_AUDIT["s279_builder_calls"] = [
            dict(item) for item in method.S279_EXPECTED_CALLS
        ]
        with mock.patch.object(
            method,
            "_V16R2_CHECKPOINT_RECEIPT",
            return_value=self.inherited_receipt(359),
        ):
            receipt = method.checkpoint_receipt(args=object())
        canary = receipt["v16r3_zero_rms_backward_summary"]["s279_endpoint_canary"]
        self.assertTrue(canary["covered_by_checkpoint"])
        self.assertEqual(canary["observed_calls"], list(method.S279_EXPECTED_CALLS))

        method._RUNTIME_AUDIT["s279_builder_calls"] = [
            dict(method.S279_EXPECTED_CALLS[0])
        ]
        with mock.patch.object(
            method,
            "_V16R2_CHECKPOINT_RECEIPT",
            return_value=self.inherited_receipt(359),
        ):
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.checkpoint_receipt(args=object())

    def test_main_restores_all_outer_monkeypatches_on_failure(self):
        original_validate = method.v16r2.validate_args
        original_receipt = method.v16r2.checkpoint_receipt
        original_builder = method.v16.build_real_source_paired_records_full644_v16

        def observe(_argv):
            self.assertIs(method.v16r2.validate_args, method.validate_args)
            self.assertIs(method.v16r2.checkpoint_receipt, method.checkpoint_receipt)
            self.assertIs(
                method.v16.build_real_source_paired_records_full644_v16,
                method.build_real_source_paired_records_full644_v16r3,
            )
            raise RuntimeError("synthetic stop")

        with mock.patch.object(method.v16r2, "main", side_effect=observe):
            with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                method.main([])
        self.assertIs(method.v16r2.validate_args, original_validate)
        self.assertIs(method.v16r2.checkpoint_receipt, original_receipt)
        self.assertIs(
            method.v16.build_real_source_paired_records_full644_v16,
            original_builder,
        )

    def test_worker_controller_pin_fresh_exact644_and_s279_canary(self):
        worker = WORKER.read_text(encoding="utf-8")
        controller = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn(
            "train_online_anchor_attention_full644_dynamic_static_v16r3.py",
            worker,
        )
        self.assertIn("v16r3.method-source", worker)
        self.assertIn("--seed 2026082302", worker)
        self.assertIn("run_exact644", controller)
        self.assertIn(method.RECEIPT_SCHEMA, controller)
        self.assertIn(method.METHOD, controller)
        self.assertIn(method.ZERO_RMS_POLICY, controller)
        self.assertIn(method.S279_TARGET_IID, controller)
        self.assertIn("sample_retry_or_skip_for_v16r3 == false", controller)
        self.assertIn(
            "component_preallreduce_finite_gate_relaxed == false", controller
        )


if __name__ == "__main__":
    unittest.main()
