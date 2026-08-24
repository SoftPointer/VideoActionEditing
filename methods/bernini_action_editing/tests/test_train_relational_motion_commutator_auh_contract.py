from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import relational_commutator_objective as objective
import train_lora as legacy
import train_prior_tangent_lora as v5
import train_relational_motion_commutator_auh as trainer


class TrainerV7ContractTests(unittest.TestCase):
    @staticmethod
    def _argv() -> list[str]:
        sha1 = "1" * 40
        sha256 = "2" * 64
        return [
            "--bernini-root", "/bernini",
            "--veomni-root", "/veomni",
            "--checkpoint", "/checkpoint",
            "--preprocessed-parquet-dir", "/data",
            "--dataset-summary", "/summary.json",
            "--routing-jsonl", "/route.jsonl",
            "--output", "/output",
            "--method-source-revision", sha1,
            "--method-source-archive-sha256", sha256,
            "--inference-loader-parity-verified",
        ]

    def test_method_branch_and_projection_contracts_are_distinct_from_v6(self) -> None:
        self.assertIn("v7", trainer.METHOD_NAME)
        self.assertIn("v7", trainer.RECEIPT_SCHEMA)
        self.assertIn("v7", trainer.OPTIMIZER_SCHEMA)
        self.assertEqual(trainer.FORWARD_CELL_ORDER, objective.FORWARD_BRANCH_ORDER)
        self.assertEqual(len(trainer.FORWARD_CELL_ORDER), 7)
        self.assertEqual(len(objective.GRAPH_BRANCHES), 2)
        self.assertEqual(len(trainer.INFERENCE_FORWARD_ORDER), 5)
        self.assertTrue(trainer.MAIN_COMMUTATOR_CONFIG.temporal_smoothing)
        self.assertEqual(
            trainer.MAIN_COMMUTATOR_CONFIG.max_correction_increment_ratio, 0.25
        )
        self.assertEqual(
            trainer.MAIN_COMMUTATOR_CONFIG.correction_increment_rms_floor, 1.0e-3
        )
        self.assertEqual(trainer.METRICS_TIMING, "pre_optimizer_update")
        self.assertEqual(
            trainer.INFERENCE_RECEIPT_SCHEMA,
            "bernini-relational-motion-commutator-inference-receipt-v7",
        )

    def test_target_only_is_default_and_relational_opt_in_is_strict(self) -> None:
        parser = trainer.build_parser()
        args = parser.parse_args(self._argv())
        args.expected_bernini_commit = legacy.BERNINI_OFFICIAL_COMMIT
        args.expected_veomni_commit = legacy.VEOMNI_TESTED_COMMIT
        args.expected_checkpoint_tree_sha256 = legacy.CHECKPOINT_TREE_SHA256
        args.expected_routing_jsonl_sha256 = v5.STRICT_ROUTING_SHA256
        trainer.validate_cli(args)
        config = trainer.loss_config_from_args(args)
        self.assertEqual(args.teacher_mode, "target_only")
        self.assertEqual(config.relational_auxiliary_weight, 0.0)
        self.assertTrue(config.commutator_config.temporal_smoothing)

        args.inference_loader_parity_verified = False
        with self.assertRaisesRegex(trainer.RelationalCommutatorAUHError, "parity"):
            trainer.validate_cli(args)
        args.inference_loader_parity_verified = True

        args.relational_auxiliary_weight = 0.1
        with self.assertRaisesRegex(trainer.RelationalCommutatorAUHError, "target_only"):
            trainer.validate_cli(args)
        args.teacher_mode = "relational_auxiliary"
        trainer.validate_cli(args)
        args.relational_auxiliary_weight = 0.0
        with self.assertRaisesRegex(
            trainer.RelationalCommutatorAUHError, "positive"
        ):
            trainer.validate_cli(args)

    def test_cli_has_no_inference_oracle_argument(self) -> None:
        destinations = {action.dest for action in trainer.build_parser()._actions}
        forbidden = (
            "target_video",
            "mask",
            "flow",
            "pose",
            "trajectory",
            "anchor",
            "first_frame",
        )
        for destination in destinations:
            self.assertFalse(any(token in destination for token in forbidden))

    def test_seven_forward_cell_has_literal_seven_renderer_calls(self) -> None:
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_seven_forward_cell"
        )
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "renderer_velocity_prediction"
        ]
        self.assertEqual(len(calls), 7)
        source_segment = ast.get_source_segment(source, function)
        self.assertIsNotNone(source_segment)
        self.assertIn("adapted_noop_v.requires_grad", source_segment)
        self.assertIn("adapted_action_v.requires_grad", source_segment)
        self.assertIn("adapter_controller.disable_adapter()", source_segment)

    def test_immutable_contract_binds_raw_training_and_bounded_deployment(self) -> None:
        config = trainer.loss_config_from_args(
            SimpleNamespace(relational_auxiliary_weight=0.0)
        )
        args = SimpleNamespace(
            method_source_revision="1" * 40,
            method_source_archive_sha256="2" * 64,
            expected_bernini_commit=legacy.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=legacy.VEOMNI_TESTED_COMMIT,
            expected_checkpoint_tree_sha256=legacy.CHECKPOINT_TREE_SHA256,
            seed=7,
            weight_decay=0.0,
            max_grad_norm=1.0,
            teacher_mode="target_only",
        )

        class Dataset:
            signature = "dataset-signature"

        class Router:
            digest = "route-digest"
            file_sha256 = "3" * 64

        target_modules = [f"module.{index}" for index in range(46)]
        routes = [
            (
                index,
                SimpleNamespace(
                    iid=f"iid-{index}", tier="motion_only", full_target_weight=0.0
                ),
            )
            for index in range(359)
        ]
        immutable = trainer._immutable_contract(
            args=args,
            dataset=Dataset(),
            dataset_summary={"sha256": "4" * 64, "index_sha256": "5" * 64},
            router=Router(),
            eligible_routes=routes,
            target_modules=target_modules,
            checkpoint=Path("/checkpoint"),
            loss_config=config,
        )["value"]
        self.assertEqual(immutable["training_correction"], "raw_Ctheta")
        self.assertEqual(
            immutable["deployment_correction"],
            "temporal_smooth_then_hard_bound_Ctheta",
        )
        self.assertEqual(
            immutable["hard_bound_formula"],
            "max(kappa*frozen_increment_rms,absolute_floor)",
        )
        self.assertEqual(immutable["graph_forwards_per_candidate"], 2)
        self.assertEqual(immutable["inference_generator_forwards"], 0)
        self.assertEqual(immutable["metrics_timing"], "pre_optimizer_update")
        self.assertEqual(
            immutable["deployment_diagnostics"][
                "target_required_kappa_statistics"
            ],
            ["median", "p90", "max"],
        )
        parity = immutable["inference_loader_parity"]
        self.assertTrue(parity["verified"])
        self.assertEqual(parity["loader_module"], trainer.INFERENCE_LOADER_MODULE)
        self.assertEqual(parity["runner_module"], trainer.INFERENCE_RUNNER_MODULE)
        self.assertEqual(
            parity["finalizer_module"], trainer.INFERENCE_FINALIZER_MODULE
        )
        self.assertEqual(parity["contract_tests"], list(trainer.INFERENCE_PARITY_TESTS))

    def test_loss_metrics_exports_bound_scale_floor_and_required_kappa(self) -> None:
        class Scalar:
            def __init__(self, value: float):
                self.value = value

            def detach(self):
                return self

            def float(self):
                return self

            def mean(self):
                return self

            def cpu(self):
                return self

            def item(self):
                return self.value

            def all(self):
                return self

        scalar = Scalar(0.5)
        eligibility = SimpleNamespace(
            eligible=Scalar(1.0),
            centered_kernel_alignment=scalar,
            teacher=SimpleNamespace(off_diagonal_relational_rms=scalar),
            target=SimpleNamespace(off_diagonal_relational_rms=scalar),
            envelope_cosine=scalar,
            envelope_relative_error=scalar,
            teacher_target_energy_ratio=scalar,
            frequency_power_cosine=scalar,
        )
        result = SimpleNamespace(
            total=scalar,
            raw_target=scalar,
            noop_preservation=scalar,
            residual_temporal_jitter=scalar,
            relational_auxiliary=scalar,
            rho=1.0,
            diagnostics=SimpleNamespace(teacher_eligibility=eligibility),
        )
        cell = trainer.SevenForwardCellResult(
            weighted_loss=scalar,
            loss_result=result,
            inverse_sigma_weight=scalar,
        )
        detached = {
            "commutator_bound": {
                "mean_scale_active": 0.9,
                "saturated_fraction_active": 0.1,
                "target_saturated_fraction_active": 0.8,
                "target_bound_mean_scale_active": 0.3,
                "floor_dominated_fraction_active": 0.2,
                "target_floor_sufficient_fraction_active": 0.05,
                "target_required_kappa_median": 0.6,
                "target_required_kappa_p90": 1.2,
                "target_required_kappa_max": 2.4,
                "target_required_kappa_near_zero_threshold": 1.0e-6,
                "frozen_increment_near_zero_fraction_active": 0.25,
                "target_required_kappa_near_zero_proxy_fraction_active": 0.15,
                "target_required_kappa_exact_zero_unreachable_fraction_active": 0.1,
            }
        }
        with patch.object(
            objective, "detached_receipt_diagnostics", return_value=detached
        ):
            metrics = trainer._loss_metrics(cell)
        for key, expected in (
            ("target_bound_mean_scale_active", 0.3),
            ("floor_dominated_fraction_active", 0.2),
            ("target_floor_sufficient_fraction_active", 0.05),
            ("target_required_kappa_median", 0.6),
            ("target_required_kappa_p90", 1.2),
            ("target_required_kappa_max", 2.4),
            ("target_required_kappa_near_zero_threshold", 1.0e-6),
            ("frozen_increment_near_zero_fraction_active", 0.25),
            ("target_required_kappa_near_zero_proxy_fraction_active", 0.15),
            (
                "target_required_kappa_exact_zero_unreachable_fraction_active",
                0.1,
            ),
        ):
            self.assertEqual(metrics[key], expected)

    def test_step_record_and_receipt_label_metrics_as_pre_update(self) -> None:
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_source = ast.get_source_segment(source, main)
        self.assertIsNotNone(main_source)
        self.assertIn('"metrics_timing": METRICS_TIMING', main_source)
        receipt = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_build_receipt"
        )
        receipt_source = ast.get_source_segment(source, receipt)
        self.assertIsNotNone(receipt_source)
        self.assertIn('"metrics_timing": METRICS_TIMING', receipt_source)
        self.assertIn('"inference_loader_parity_pending": True', receipt_source)
        self.assertIn('"verified": False', receipt_source)
        self.assertIn(
            '"status": "pending_post_save_strict_reload"', receipt_source
        )

    def test_launcher_finalizes_only_post_save_formal_target_only_artifact(self) -> None:
        script = (
            METHOD_ROOT / "scripts/auh_train_relational_motion_commutator_v7.sbatch"
        ).read_text(encoding="utf-8")
        for required in (
            "finalize_relational_motion_commutator_checkpoint.py",
            "test_finalize_relational_motion_commutator_checkpoint.py",
            "test_infer_delta_lora_contract.py",
            '"${max_steps}" == 40',
            '"${save_every}" == 40',
            '"${teacher_mode}" == target_only',
            "checkpoint-00000040",
            ".inference_loader_parity_pending == false",
            ".artifact_validation.adapter_tensor_count == 92",
            'artifact_release=pending (non-formal arm)',
        ):
            self.assertIn(required, script)
        self.assertLess(
            script.index("train_relational_motion_commutator_auh.py"),
            script.rindex("finalize_relational_motion_commutator_checkpoint.py"),
        )


if __name__ == "__main__":
    unittest.main()
