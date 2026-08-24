from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v16r4 online-anchor tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r4 as method


class Full644DynamicStaticV16R4Test(unittest.TestCase):
    def setUp(self) -> None:
        method._RUNTIME_AUDIT = method._empty_runtime_audit()
        method._CANARY_BINDING = None
        method._ACTIVE_OPTIMIZER = None
        method.v16r3._RUNTIME_AUDIT = method.v16r3._empty_runtime_audit()
        method.v16._RUNTIME_AUDIT = method.v16._empty_runtime_audit()

    def tearDown(self) -> None:
        method._CANARY_BINDING = None
        method._ACTIVE_OPTIMIZER = None

    @staticmethod
    def write_canary(directory: Path) -> tuple[Path, str]:
        path = directory / "heldout8.jsonl"
        rows = [
            {
                "schema_version": method.DECODED_CANARY_INPUT_SCHEMA,
                "index": index,
                "iid": f"heldout-{index:02d}",
                "split": "test" if index < 5 else "validation",
                "source_video": f"/sealed/heldout/source-{index:02d}.mp4",
                "instruction": f"perform held-out action {index}",
                "seed": 2026 + index,
            }
            for index in range(method.DECODED_CANARY_CASE_COUNT)
        ]
        path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path, method.v16.file_sha256(path)

    def test_parser_adds_required_decoded_canary_binding(self):
        parser = method.build_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertTrue(actions["decoded_canary_manifest"].required)
        self.assertTrue(actions["decoded_canary_manifest_sha256"].required)

    def test_validation_shadows_only_old_mode_and_lr(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self.write_canary(Path(directory))
            args = SimpleNamespace(
                output="/tmp/fresh-v16r4-s644",
                max_grad_norm=method.MAX_GRAD_NORM,
                max_steps=method.v16.FULL644_ROWS,
                seed=method.SEED,
                gradient_diagnostic_only=False,
                replay_combine_mode=method.REPLAY_COMBINE_MODE,
                learning_rate=method.LEARNING_RATE,
                decoded_canary_manifest=str(path),
                decoded_canary_manifest_sha256=digest,
            )
            observed = []

            def inherited(shadow):
                observed.append(shadow)
                self.assertEqual(
                    shadow.replay_combine_mode, method.v15.REPLAY_COMBINE_MODE
                )
                self.assertEqual(shadow.learning_rate, 1.0e-5)
                self.assertEqual(shadow.output, args.output)

            with mock.patch.object(
                method, "_V16_VALIDATE_ARGS", side_effect=inherited
            ), mock.patch.object(method.v16r3, "_validate_zero_rms_operator"):
                method.validate_args(args)
            self.assertEqual(len(observed), 1)
            self.assertEqual(method._CANARY_BINDING["case_count"], 8)
            self.assertEqual(method._CANARY_BINDING["sha256"], digest)

    def test_validation_rejects_old_pcgrad_or_old_lr(self):
        base_args = {
            "output": "/tmp/fresh-v16r4-s644",
            "max_grad_norm": method.MAX_GRAD_NORM,
            "max_steps": method.v16.FULL644_ROWS,
            "seed": method.SEED,
            "gradient_diagnostic_only": False,
            "decoded_canary_manifest": "/does/not/matter",
            "decoded_canary_manifest_sha256": "0" * 64,
        }
        with mock.patch.object(method, "_V16_VALIDATE_ARGS"), mock.patch.object(
            method.v16r3, "_validate_zero_rms_operator"
        ):
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(
                    SimpleNamespace(
                        **base_args,
                        replay_combine_mode=method.v15.REPLAY_COMBINE_MODE,
                        learning_rate=method.LEARNING_RATE,
                    )
                )
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(
                    SimpleNamespace(
                        **base_args,
                        replay_combine_mode=method.REPLAY_COMBINE_MODE,
                        learning_rate=1.0e-5,
                    )
                )

    def test_canary_binding_rejects_wrong_digest_and_duplicate_iid(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self.write_canary(Path(directory))
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method._load_decoded_canary_binding(path, "0" * 64)

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["iid"] = rows[0]["iid"]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            duplicate_digest = method.v16.file_sha256(path)
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method._load_decoded_canary_binding(path, duplicate_digest)
            self.assertNotEqual(digest, duplicate_digest)

    def test_source_halfspace_merge_preserves_both_formal_directions(self):
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        parameter.grad = torch.tensor([-0.2, 1.0], dtype=torch.float32)
        action = (torch.tensor([1.0, 0.0], dtype=torch.float32),)
        values = method.merge_component_gradients(
            (("adapter", parameter),),
            action,
            replay_combine_mode=method.REPLAY_COMBINE_MODE,
            base_replay_scale=0.025,
        )
        self.assertGreater(
            values["action_gradient_dot_combined_gradient_fp64"], 0.0
        )
        self.assertGreaterEqual(
            values["raw_replay_gradient_dot_combined_gradient_fp64"], -1.0e-8
        )
        self.assertTrue(values["v16r4_source_descent_required"])
        self.assertFalse(values["v16r4_action_only_fallback_allowed"])
        self.assertEqual(method._RUNTIME_AUDIT["formal_merge_count"], 1)

    def test_merge_rejects_pcgrad_without_calling_any_fallback(self):
        with mock.patch.object(method, "_BASE_MERGE_COMPONENT_GRADIENTS") as merge:
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.merge_component_gradients(
                    (),
                    (),
                    replay_combine_mode=method.v15.REPLAY_COMBINE_MODE,
                    base_replay_scale=0.025,
                )
            merge.assert_not_called()

    def test_mixed_rank_gate_fails_closed_on_every_rank(self):
        def mixed(counts, op=None):
            del op
            counts.copy_(torch.tensor([3, 1], dtype=counts.dtype))

        with mock.patch.object(torch.distributed, "is_available", return_value=True), \
             mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(torch.distributed, "get_world_size", return_value=4), \
             mock.patch.object(torch.distributed, "all_reduce", side_effect=mixed):
            with self.assertRaisesRegex(
                method.base.OnlineAnchorTrainingError,
                "differs across ranks",
            ):
                method._collective_pass_or_failure(
                    True,
                    device=torch.device("cpu"),
                    phase="synthetic",
                )

    def test_projected_optimizer_actual_displacement_descends_both(self):
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        action = (torch.tensor([-1.0, 1.0]),)
        replay = (torch.tensor([3.0, 1.0]),)
        parameter.grad = replay[0].clone()
        method.merge_component_gradients(
            (("adapter", parameter),),
            action,
            replay_combine_mode=method.REPLAY_COMBINE_MODE,
            base_replay_scale=0.025,
        )
        before = (parameter.detach().clone(),)
        optimizer = method._make_global_rms_projected_sgd(
            (parameter,), lr=method.LEARNING_RATE
        )
        method._ACTIVE_OPTIMIZER = optimizer
        optimizer.step()
        values = method.actual_optimizer_update_probe(
            (("adapter", parameter),),
            before,
            action,
            replay,
            replay_combine_mode=method.REPLAY_COMBINE_MODE,
            step=1,
        )
        self.assertTrue(values["action_descent_passed"])
        self.assertTrue(values["source_descent_passed"])
        self.assertEqual(values["v16r4_optimizer"], method.OPTIMIZER)
        self.assertLess(
            values["v16r4_actual_vs_planned_delta_l2_relative_error"],
            1.0e-3,
        )
        self.assertEqual(values["v16r4_probe_retry_count"], 0)
        self.assertFalse(values["v16r4_optimizer_state_reset"])
        self.assertEqual(method._RUNTIME_AUDIT["actual_update_steps"], [1])

    def test_real_direction_distortion_adam_fails_where_projected_sgd_passes(self):
        action = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        replay = torch.tensor([3.0, 1.0], dtype=torch.float32)
        merge_parameter = torch.nn.Parameter(torch.zeros_like(action))
        merge_parameter.grad = replay.clone()
        formal = method.merge_component_gradients(
            (("adapter", merge_parameter),),
            (action,),
            replay_combine_mode=method.REPLAY_COMBINE_MODE,
            base_replay_scale=0.025,
        )
        combined = merge_parameter.grad.detach().clone()
        self.assertGreater(
            formal["action_gradient_dot_combined_gradient_fp64"], 0.0
        )
        self.assertGreater(
            formal["raw_replay_gradient_dot_combined_gradient_fp64"], 0.0
        )

        adam_parameter = torch.nn.Parameter(torch.zeros_like(action))
        adam_before = (adam_parameter.detach().clone(),)
        adam_parameter.grad = combined.clone()
        adam = torch.optim.AdamW(
            (adam_parameter,), lr=method.LEARNING_RATE, weight_decay=0.0
        )
        adam.step()
        adam_delta = adam_parameter.detach() - adam_before[0]
        self.assertGreater(float(-(action * adam_delta).sum().item()), 0.0)
        self.assertLess(float(-(replay * adam_delta).sum().item()), 0.0)
        with self.assertRaisesRegex(
            method.base.OnlineAnchorTrainingError,
            "required source-descent half-space",
        ):
            method._BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE(
                (("adapter", adam_parameter),),
                adam_before,
                (action,),
                (replay,),
                replay_combine_mode=method.REPLAY_COMBINE_MODE,
                step=1,
            )

        projected_parameter = torch.nn.Parameter(torch.zeros_like(action))
        projected_before = (projected_parameter.detach().clone(),)
        projected_parameter.grad = combined.clone()
        projected = method._make_global_rms_projected_sgd(
            (projected_parameter,), lr=method.LEARNING_RATE
        )
        projected.step()
        projected_values = method._BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE(
            (("adapter", projected_parameter),),
            projected_before,
            (action,),
            (replay,),
            replay_combine_mode=method.REPLAY_COMBINE_MODE,
            step=1,
        )
        self.assertTrue(projected_values["action_descent_passed"])
        self.assertTrue(projected_values["source_descent_passed"])
        self.assertEqual(len(projected.state), 0)
        self.assertAlmostEqual(
            projected._v16r4_last_step[
                "planned_delta_theta_l2_norm_fp64"
            ],
            method.LEARNING_RATE * (2.0 ** 0.5),
        )

    def test_observed_v16r3_s1_geometry_forecasts_safe_nontrivial_step(self):
        forecast = method._s1_projected_update_forecast()
        observation = forecast["v16r3_observation"]
        self.assertFalse(observation["formal_first_order_source_fm_preserved"])
        self.assertLess(observation["adamw_source_descent_fp64"], 0.0)
        self.assertFalse(observation["adamw_source_descent_passed"])
        self.assertGreater(
            forecast["source_halfspace_action_inner_product_forecast"], 0.0
        )
        self.assertGreater(
            forecast["source_halfspace_source_inner_product_forecast"], 0.0
        )
        self.assertGreater(forecast["projected_sgd_action_descent_forecast"], 0.0)
        self.assertGreater(forecast["projected_sgd_source_descent_forecast"], 0.0)
        self.assertGreater(
            forecast["projected_sgd_planned_delta_l2_forecast"], 9.0e-3
        )
        self.assertGreater(
            forecast["v16r3_adamw_to_projected_delta_l2_ratio"], 9.0
        )

    def test_actual_source_ascent_fails_once_without_retry(self):
        parameter = torch.nn.Parameter(torch.tensor([-0.1, 0.1]))
        before = (torch.zeros_like(parameter),)
        action = (torch.tensor([1.0, 0.0]),)
        replay = (torch.tensor([0.0, 1.0]),)
        authority = method._BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE
        with mock.patch.object(
            method,
            "_BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE",
            wraps=authority,
        ) as probe:
            with self.assertRaisesRegex(
                method.base.OnlineAnchorTrainingError,
                "required source-descent half-space",
            ):
                method.actual_optimizer_update_probe(
                    (("adapter", parameter),),
                    before,
                    action,
                    replay,
                    replay_combine_mode=method.REPLAY_COMBINE_MODE,
                    step=1,
                )
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(method._RUNTIME_AUDIT["actual_update_steps"], [])
        self.assertEqual(method._RUNTIME_AUDIT["failed_actual_probe_steps"], [1])

    @staticmethod
    def inherited_receipt(step: int):
        return {
            "schema_version": method.v16.RECEIPT_SCHEMA,
            "global_step": step,
            "training_contract": {"method": method.v16.METHOD},
            "component_gradient_probes": {
                "interaction": {
                    "replay_combine_mode": method.REPLAY_COMBINE_MODE,
                    "first_order_source_fm_preserved": True,
                }
            },
            "actual_optimizer_update_probe": {
                "replay_combine_mode": method.REPLAY_COMBINE_MODE,
                "optimizer_semantics_observed_not_modified": True,
                "action_descent_passed": True,
                "source_descent_required": True,
                "source_descent_passed": True,
                "v16r4_optimizer": method.OPTIMIZER,
            },
            "anchor_cache": {
                "qk_only_zero_rms_backward_policy": method.v16r3.ZERO_RMS_POLICY
            },
        }

    def prepare_receipt_state(self, step: int) -> None:
        optimizer_step = {
            "schema_version": "bernini-global-rms-projected-sgd-step-v1",
            "step": step,
            "optimizer": method.OPTIMIZER,
        }
        method._RUNTIME_AUDIT.update(
            {
                "formal_merge_count": step,
                "actual_update_steps": list(range(1, step + 1)),
                "failed_actual_probe_steps": [],
                "optimizer_step_count": step,
                "last_optimizer_step": optimizer_step,
            }
        )
        method._ACTIVE_OPTIMIZER = SimpleNamespace(
            _v16r4_step_count=step,
            _v16r4_last_step=optimizer_step,
            state={},
        )
        method._CANARY_BINDING = {
            "path": "/sealed/heldout8.jsonl",
            "sha256": "a" * 64,
            "case_count": 8,
            "iids": tuple(f"heldout-{index:02d}" for index in range(8)),
            "seeds": tuple(range(8)),
            "iids_sha256": "b" * 64,
        }
        method.v16._RUNTIME_AUDIT["manifest_iids"] = tuple(
            f"train-{index:03d}" for index in range(method.v16.FULL644_ROWS)
        )

    def test_receipt_binds_source_descent_and_automatic_canary(self):
        self.prepare_receipt_state(8)
        with mock.patch.object(
            method,
            "_V16_CHECKPOINT_RECEIPT",
            return_value=self.inherited_receipt(8),
        ):
            receipt = method.checkpoint_receipt(args=object())
        contract = receipt["training_contract"]
        source = receipt["v16r4_source_descent_summary"]
        canary = receipt["v16r4_decoded_canary_contract"]
        self.assertEqual(receipt["schema_version"], method.RECEIPT_SCHEMA)
        self.assertEqual(contract["method"], method.METHOD)
        self.assertEqual(contract["optimizer_scalar_learning_rate"], 1.0e-6)
        self.assertEqual(contract["seed"], method.SEED)
        self.assertEqual(contract["optimizer"], method.OPTIMIZER)
        self.assertFalse(contract["optimizer_coordinatewise_preconditioner"])
        self.assertTrue(contract["source_gradient_preservation_enforced"])
        self.assertFalse(contract["action_only_fallback_allowed"])
        self.assertFalse(contract["optimizer_state_reset_allowed"])
        self.assertEqual(source["successful_update_count"], 8)
        self.assertTrue(canary["current_checkpoint_requires_decoded_canary"])
        self.assertFalse(canary["per_sample_manual_review_required"])
        self.assertFalse(
            canary["checkpoint_promotion_eligible_from_training_receipt_alone"]
        )

    def test_receipt_requires_exact_s279_zero_rms_canary(self):
        self.prepare_receipt_state(359)
        method.v16r3._RUNTIME_AUDIT["s279_builder_calls"] = [
            dict(item) for item in method.v16r3.S279_EXPECTED_CALLS
        ]
        with mock.patch.object(
            method,
            "_V16_CHECKPOINT_RECEIPT",
            return_value=self.inherited_receipt(359),
        ):
            receipt = method.checkpoint_receipt(args=object())
        self.assertTrue(
            receipt["v16r3_zero_rms_backward_summary"]["s279_endpoint_canary"][
                "covered_by_checkpoint"
            ]
        )

        method.v16r3._RUNTIME_AUDIT["s279_builder_calls"] = []
        with mock.patch.object(
            method,
            "_V16_CHECKPOINT_RECEIPT",
            return_value=self.inherited_receipt(359),
        ):
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.checkpoint_receipt(args=object())

    def test_main_installs_no_fallback_path_and_restores_patches(self):
        originals = {
            "parser": method.v16.build_parser,
            "validate": method.v16.validate_args,
            "receipt": method.v16.checkpoint_receipt,
            "builder": method.v16.build_real_source_paired_records_full644_v16,
            "receipt_parent": method.v16._R2_CHECKPOINT_RECEIPT,
            "merge": method.r2.merge_component_gradients,
            "probe": method.base.actual_optimizer_update_probe,
            "adamw": torch.optim.AdamW,
        }

        def observe(_argv):
            self.assertIs(method.v16.build_parser, method.build_parser)
            self.assertIs(method.v16.validate_args, method.validate_args)
            self.assertIs(method.v16.checkpoint_receipt, method.checkpoint_receipt)
            self.assertIs(
                method.v16.build_real_source_paired_records_full644_v16,
                method.build_real_source_paired_records_full644_dynamic_static_v16r4,
            )
            self.assertIs(
                method.v16._R2_CHECKPOINT_RECEIPT,
                method._V15_CHECKPOINT_RECEIPT,
            )
            self.assertIs(method.r2.merge_component_gradients, method.merge_component_gradients)
            self.assertIs(
                method.base.actual_optimizer_update_probe,
                method.actual_optimizer_update_probe,
            )
            self.assertIsNot(torch.optim.AdamW, originals["adamw"])
            raise RuntimeError("synthetic stop")

        with mock.patch.object(method.v16, "main", side_effect=observe):
            with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                method.main([])
        self.assertIs(method.v16.build_parser, originals["parser"])
        self.assertIs(method.v16.validate_args, originals["validate"])
        self.assertIs(method.v16.checkpoint_receipt, originals["receipt"])
        self.assertIs(
            method.v16.build_real_source_paired_records_full644_v16,
            originals["builder"],
        )
        self.assertIs(method.v16._R2_CHECKPOINT_RECEIPT, originals["receipt_parent"])
        self.assertIs(method.r2.merge_component_gradients, originals["merge"])
        self.assertIs(method.base.actual_optimizer_update_probe, originals["probe"])
        self.assertIs(torch.optim.AdamW, originals["adamw"])


if __name__ == "__main__":
    unittest.main()
