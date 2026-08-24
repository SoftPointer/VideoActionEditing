#!/usr/bin/env python3
"""Contract and small-tensor tests for the SPT-v2 joint LoRA trainer."""

from __future__ import annotations

from dataclasses import asdict
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SPT_ROOT = METHOD_ROOT / "spt_v2"
for root in (METHOD_ROOT, SPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import phase_query_planner as phase_query  # noqa: E402
import phase_transport as spt  # noqa: E402
import train_joint_lora as joint  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_student as planner_train  # noqa: E402


class JointPureContractTests(unittest.TestCase):
    def _argv(self) -> list[str]:
        return [
            "--bernini-root", "/bernini",
            "--veomni-root", "/veomni",
            "--checkpoint", "/checkpoint",
            "--preprocessed-parquet-dir", "/data",
            "--dataset-summary", "/summary.json",
            "--planner-checkpoint", "/planner",
            "--output", "/output",
            "--method-source-revision", "1" * 40,
            "--method-source-archive-sha256", "2" * 64,
        ]

    def test_parser_is_fixed_to_conservative_joint_arm(self) -> None:
        parser = joint.build_parser()
        args = parser.parse_args(self._argv())
        joint.validate_cli(args)
        self.assertEqual(args.lora_scope, "cross_q_out")
        self.assertEqual(args.integration_steps, 40)
        self.assertEqual(args.integration_flow_shift, 5.0)
        self.assertEqual(args.max_oracle_generate_fraction, 0.12)
        self.assertIsNone(args.train_prefix_rows)
        self.assertIsNone(args.selected_membership)
        self.assertFalse(hasattr(args, "full_target_loss_weight"))
        self.assertFalse(hasattr(args, "init_adapter_checkpoint"))

    def test_selected_membership_is_exclusive_with_prefix(self) -> None:
        parser = joint.build_parser()
        args = parser.parse_args(
            self._argv()
            + [
                "--selected-membership",
                "/trusted.json",
                "--train-prefix-rows",
                "8",
            ]
        )
        with self.assertRaisesRegex(joint.JointTrainingError, "mutually exclusive"):
            joint.validate_cli(args)

    def test_generate_budget_cannot_be_relaxed_past_point_twelve(self) -> None:
        parser = joint.build_parser()
        for value in ("0", "-0.1", "0.120001", "0.15", "0.5"):
            args = parser.parse_args(
                self._argv() + ["--max-oracle-generate-fraction", value]
            )
            with self.assertRaisesRegex(joint.JointTrainingError, "0,0.12"):
                joint.validate_cli(args)

    def test_student_api_has_no_target_or_oracle_argument(self) -> None:
        self.assertEqual(
            list(inspect.signature(joint.student_plan).parameters),
            ["planner", "source", "raw_instruction_tokens"],
        )
        calls = []

        def planner(source, tokens):
            calls.append((source, tokens))
            return "plan"

        self.assertEqual(joint.student_plan(planner, "S", "raw"), "plan")
        self.assertEqual(calls, [("S", "raw")])

    def test_planner_bundle_is_hash_bound_phase_query_v2(self) -> None:
        config = asdict(phase_query.PhaseQueryPlannerConfig())
        receipt = {
            "schema_version": planner_train.RECEIPT_SCHEMA,
            "global_step": 17,
            "immutable_contract": {"value": {"planner_config": config}},
            "planner": {"architecture": phase_query.ARCHITECTURE_NAME},
            "supervision": {
                "student_api": ["source", "instruction_tokens"],
                "student_target_argument_exists": False,
            },
        }
        receipt["receipt_digest"] = legacy.object_sha256(receipt)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "planner_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            (root / "receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            (root / "planner.safetensors").write_bytes(b"strict-placeholder")
            bundle = joint.inspect_planner_bundle(root)
            self.assertEqual(bundle.identity["architecture"], "phase_query_v2")
            self.assertEqual(bundle.identity["global_step"], 17)
            self.assertRegex(bundle.identity["identity_digest"], r"^[0-9a-f]{64}$")

            bad = dict(receipt)
            bad["supervision"] = dict(receipt["supervision"])
            bad["supervision"]["student_target_argument_exists"] = True
            bad.pop("receipt_digest")
            bad["receipt_digest"] = legacy.object_sha256(bad)
            (root / "receipt.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(joint.JointTrainingError, "forbid target"):
                joint.inspect_planner_bundle(root)

    def test_source_contains_executor_and_forbids_full_target_repaint(self) -> None:
        main_source = inspect.getsource(joint.main)
        self.assertIn("plan=predicted_plan", main_source)
        self.assertIn("plan=oracle_plan", main_source)
        self.assertIn("execute_packed_velocity", main_source)
        self.assertLess(
            main_source.index("plan=predicted_plan"),
            main_source.index("plan=oracle_plan"),
        )
        loss_source = inspect.getsource(joint.compute_joint_loss)
        self.assertIn("generate_loss = _weighted_cell_mse", loss_source)
        self.assertIn("non_generate", loss_source)
        self.assertIn("ordinary_full_target_loss", loss_source)
        self.assertNotIn("torch.mean((action_prediction - action_target)", loss_source)

    def test_preflight_occurs_before_bernini_model_construction(self) -> None:
        source = inspect.getsource(joint.main)
        self.assertLess(
            source.index("preflight_oracle_budget("),
            source.index("base_model = BerniniRendererModel(config)"),
        )
        preflight_source = inspect.getsource(joint.preflight_oracle_budget)
        self.assertIn('members = training_membership.get("members")', preflight_source)
        self.assertIn('row_index = int(member["row_index"])', preflight_source)
        self.assertNotIn("for row_index in range(rows)", preflight_source)


try:
    import torch
except ImportError:  # pragma: no cover - local contract-only environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable in the local contract environment")
class JointTensorTests(unittest.TestCase):
    def _plan(self, gates):
        offsets = torch.zeros(1, 3, 21, 1, 3)
        return spt.PhasePlan(offsets, gates, "oracle_pair_proxy")

    def test_executor_loss_balances_classes_not_cell_counts(self) -> None:
        target = torch.zeros(1, 63, 4)
        prediction = torch.ones_like(target)
        # Preserve occupies 61 cells with MSE 1; the two minority cells have
        # MSE 4 and 9.  The required answer is the mean of three class means.
        prediction[:, -2] = 2.0
        prediction[:, -1] = 3.0
        gates = torch.zeros(1, 3, 21, 1, 3)
        flat = gates.reshape(1, 3, -1)
        flat[:, 0, :-2] = 1.0
        flat[:, 1, -2] = 1.0
        flat[:, 2, -1] = 1.0
        loss, parts = joint.class_balanced_executor_loss(
            prediction, target, gates
        )
        self.assertAlmostEqual(float(loss), 14.0 / 3.0, places=5)
        self.assertAlmostEqual(float(parts["executor_preserve"]), 1.0)
        self.assertAlmostEqual(float(parts["executor_transport"]), 4.0)
        self.assertAlmostEqual(float(parts["executor_generate"]), 9.0)

    def test_generate_budget_fails_closed(self) -> None:
        gates = torch.zeros(1, 3, 21, 1, 3)
        gates[:, 0] = 1.0
        gates[:, 0, :, :, :9 // 3] = 0.0
        gates[:, 2, :, :, :9 // 3] = 1.0
        plan = self._plan(gates)
        plan.validate(torch.zeros(1, 21, 1, 3, 64))
        with self.assertRaisesRegex(joint.JointTrainingError, "budget exceeded"):
            joint.enforce_oracle_generate_budget(plan, maximum=0.12, iid="row")

    def test_student_plan_really_controls_executor_velocity(self) -> None:
        source = torch.randn(1, 21, 1, 2, 64)
        source_packed = spt.video_to_packed(source)
        noisy = torch.randn_like(source_packed)
        base = torch.randn_like(source_packed, requires_grad=True)
        offsets = torch.zeros(1, 3, 21, 1, 2)
        gates = torch.zeros_like(offsets)
        gates[:, spt.GATE_GENERATE] = 1.0
        plan = spt.PhasePlan(offsets, gates, "student")
        actual = spt.execute_packed_velocity(
            source_packed=source_packed,
            noisy_packed=noisy,
            base_velocity_packed=base,
            sigma=torch.tensor([0.5]),
            height=1,
            width=2,
            plan=plan,
        )
        self.assertTrue(torch.allclose(actual, base.float(), atol=1e-5, rtol=1e-5))
        actual.square().mean().backward()
        self.assertIsNotNone(base.grad)
        self.assertGreater(float(base.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
