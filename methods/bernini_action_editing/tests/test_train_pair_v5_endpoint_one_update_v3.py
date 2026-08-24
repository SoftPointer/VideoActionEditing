from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = METHOD_ROOT / "train_pair_v5_endpoint_one_update_v3.py"
LAUNCHER_PATH = (
    METHOD_ROOT / "scripts/auh_train_pair_v5_endpoint_one_update_v3.sbatch"
)
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_pair_v5_endpoint_one_update_v3 as trainer  # noqa: E402


class EndpointOneUpdateTrainerV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = TRAINER_PATH.read_text(encoding="utf-8")
        cls.launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    def test_isolated_from_v2_calibration_selector_and_trainer(self) -> None:
        for forbidden in (
            "pair_v5_action_energy_calibration",
            "pair_v5_safe_pareto",
            "train_pair_v5_native_flow_dpo_v2",
            "train_pair_v5_action_preference",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_ast_contains_exactly_one_optimizer_step_call(self) -> None:
        tree = ast.parse(self.source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "optimizer"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("--max-steps", self.source)
        self.assertNotIn("--max-schedule-steps", self.source)

    def test_cio_is_optional_and_loader_returns_three_values(self) -> None:
        result = trainer._load_optional_frozen_cio(None, None)
        self.assertEqual(len(result), 3)
        self.assertIsNone(result[0])
        self.assertFalse(result[1]["loaded"])
        parser = trainer.build_parser()
        option_names = {action.dest for action in parser._actions}
        self.assertNotIn("frozen_cio_adapter", option_names)
        self.assertIn("expected_parent_policy_digest", option_names)

    def test_frozen_reference_uses_active_parent_action_lora(self) -> None:
        signature = inspect.signature(trainer._route_stack)
        self.assertNotIn("action_enabled", signature.parameters)
        route_source = inspect.getsource(trainer._route_stack)
        self.assertIn("enabled=True", route_source)
        reference_source = inspect.getsource(
            trainer._reference_and_student_predictions
        )
        self.assertIn("reference", reference_source)
        self.assertIn("student", reference_source)
        self.assertIn("before != after", reference_source)

    def test_dp2_assignment_uses_two_independent_pair_indices(self) -> None:
        dp0 = trainer.assigned_pair_indices(
            pair_count=4, dp_rank=0, accumulation_steps=2
        )
        dp1 = trainer.assigned_pair_indices(
            pair_count=4, dp_rank=1, accumulation_steps=2
        )
        self.assertEqual(dp0, (0, 2))
        self.assertEqual(dp1, (1, 3))
        self.assertTrue(set(dp0).isdisjoint(dp1))

    def test_fresh_noise_is_bound_to_pair_arm_and_accumulation(self) -> None:
        values = {
            trainer.fresh_noise_seed(
                base_seed=7,
                manifest_digest="a" * 64,
                pair_digest=(hex(index + 1)[2:] * 64)[:64],
                dp_rank=index % 2,
                accumulation_index=index,
            )
            for index in range(4)
        }
        self.assertEqual(len(values), 4)

    def test_preflight_authorization_precedes_model_and_optimizer(self) -> None:
        manifest_gate = self.source.index("preflight_inputs(args)")
        source_activation = self.source.index("legacy.activate_source_trees(")
        model_construction = self.source.index("renderer = BerniniRendererModel(config)")
        optimizer_construction = self.source.index("optimizer = torch.optim.AdamW(")
        self.assertLess(manifest_gate, source_activation)
        self.assertLess(source_activation, model_construction)
        self.assertLess(model_construction, optimizer_construction)

    def test_cli_contract_is_exact_one_update(self) -> None:
        namespace = argparse.Namespace(
            ack_experimental_no_action_success_claim=True,
            expected_generation_round=0,
            expected_parent_policy_digest="a" * 64,
            expected_manifest_sha256="b" * 64,
            expected_checkpoint_tree_sha256="c" * 64,
            method_source_archive_sha256="d" * 64,
            sigma_index=20,
            gradient_accumulation_steps=2,
            learning_rate=1.0e-6,
            beta=1000.0,
            max_grad_norm=1.0,
            seed=9,
        )
        contract = trainer.validate_cli(namespace)
        self.assertEqual(contract["optimizer_update_count"], 1)
        self.assertEqual(contract["optimizer_step_index"], 0)

    def test_launcher_is_world8_dp2_sp4_and_has_no_cio_override(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.launcher)
        self.assertIn("--nproc_per_node=8", self.launcher)
        self.assertIn("topology=WORLD8/DP2xSP4", self.launcher)
        self.assertIn("optimizer_updates=1", self.launcher)
        self.assertNotIn("MAX_STEPS", self.launcher)
        self.assertNotIn("CIO_ADAPTER", self.launcher)

    def test_launcher_audits_fresh_next_round_requirement(self) -> None:
        self.assertIn(
            "receipt['static_rollout_reused_for_multiple_steps'] is False",
            self.launcher,
        )
        self.assertIn(
            "receipt['fresh_next_round_rollout_required'] is True",
            self.launcher,
        )
        self.assertIn("PAIR_V5_ENDPOINT_V3_GENERATION_ROUND", self.launcher)
        self.assertIn("PAIR_V5_ENDPOINT_V3_PARENT_POLICY_DIGEST", self.launcher)


if __name__ == "__main__":
    unittest.main()
