from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import source_self_runtime as runtime
import train_preservation_residual_v1 as method


class PreservationResidualContractTests(unittest.TestCase):
    def test_registered_sigma_plan_covers_exact40_once(self) -> None:
        rows = method.training_coordinates(40)
        self.assertEqual([row.schedule_index for row in rows], list(range(40)))
        self.assertEqual(
            [row.optimizer_step_zero_based for row in rows], list(range(40))
        )
        self.assertTrue(all(method.validate_coordinate(row) is row for row in rows))

    def test_exact20_replicate_uses_the_registered_exact40_prefix(self) -> None:
        prefix = method.training_coordinates(20)
        full = method.training_coordinates(40)
        self.assertEqual(prefix, full[:20])
        self.assertEqual([row.schedule_index for row in prefix], list(range(20)))
        plan = method.fixed_plan_receipt(20)
        self.assertEqual(plan["training_schedule_indices"], list(range(20)))
        self.assertFalse(plan["formal_exact40_complete"])

    def test_continuous_run_registers_loadable_zero_twenty_forty(self) -> None:
        self.assertEqual(method.CHECKPOINT_INTERVAL, 20)
        self.assertEqual(method.LOADABLE_CHECKPOINT_STEPS, (0, 20, 40))
        parser = method.build_parser()
        checkpoint = next(
            item for item in parser._actions if item.dest == "checkpoint_output_root"
        )
        self.assertIsNone(checkpoint.default)
        source = (ROOT / "train_preservation_residual_v1.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("publish_cadence_checkpoint", source)
        self.assertIn('"continuous_trajectory": True', source)

    def test_optimizer_step_count_rejects_unregistered_values(self) -> None:
        for value in (True, 0, 4, 19, 21, 39, 41, 80, 1.5, "40"):
            with self.subTest(value=value):
                with self.assertRaises(method.SourceNoisedCarrierTrainingError):
                    method.training_coordinates(value)

    def test_fixed_plan_is_preservation_only(self) -> None:
        plan = method.fixed_plan_receipt(40)
        self.assertEqual(plan["optimizer_steps"], 40)
        self.assertEqual(
            plan["schedule_policy"],
            "one_update_at_each_exact40_positive_coordinate",
        )
        self.assertIs(plan["synthetic_target_consumed"], False)
        self.assertIs(plan["action_reward_consumed"], False)
        self.assertIn("z_clean", plan["source_condition_equation"])
        self.assertNotIn("z_style", str(plan))

    def test_exact_noop_prompt_is_pinned(self) -> None:
        self.assertTrue(
            method.EXACT_NOOP_INSTRUCTION.startswith(
                "Keep the source video exactly unchanged"
            )
        )
        self.assertIn("action", method.EXACT_NOOP_INSTRUCTION)
        self.assertNotIn("target", method.EXACT_NOOP_INSTRUCTION.lower())

    def test_world8_topology_parallelizes_two_sources_over_dp2(self) -> None:
        topology = method.register_preservation_topology(runtime)
        self.assertEqual(
            (topology.world_size, topology.dp_size, topology.sp_size), (8, 2, 4)
        )
        self.assertEqual(topology.sp_group_ranks, runtime.SP_GROUP_RANKS)
        self.assertEqual(topology.dp_group_ranks, runtime.DP_GROUP_RANKS)

    def test_optimizer_gate_accepts_one_complete_dp_owned_arm(self) -> None:
        self.assertTrue(
            method.authorize_optimizer_step(
                completed_control_arms=(0,), completed_backward_arms=(0,)
            )
        )
        self.assertTrue(
            method.authorize_optimizer_step(
                completed_control_arms=(1,), completed_backward_arms=(1,)
            )
        )
        with self.assertRaises(method.SourceNoisedCarrierTrainingError):
            method.authorize_optimizer_step(
                completed_control_arms=(0,), completed_backward_arms=(1,)
            )

    def test_parser_registers_only_rank8_main_and_rank2_variant(self) -> None:
        parser = method.build_parser()
        action = next(item for item in parser._actions if item.dest == "adapter_rank")
        self.assertEqual(tuple(action.choices), (2, 8))
        self.assertEqual(action.default, 8)
        steps = next(item for item in parser._actions if item.dest == "optimizer_steps")
        self.assertEqual(tuple(steps.choices), (20, 40))

    def test_private_role_route_admits_sp4_without_changing_legacy_role(self) -> None:
        source = (ROOT / "preservation_source_role_v1.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("if size not in {1, 2, 4}", source)
        self.assertIn("bernini-preservation-source-role-adapter-v1", source)
        trainer = (ROOT / "train_preservation_residual_v1.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("import preservation_source_role_v1 as role", trainer)
        self.assertIn("for loading_rank in range(topology.world_size)", trainer)
        self.assertIn('"rank_serialized_cpu_checkpoint_load": True', trainer)

    def test_source_load_is_model_free(self) -> None:
        module_name = "preservation_residual_model_free_import"
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / "train_preservation_residual_v1.py"
        )
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = loaded
        before = set(sys.modules)
        try:
            spec.loader.exec_module(loaded)
        finally:
            sys.modules.pop(module_name, None)
        imported = set(sys.modules) - before
        self.assertNotIn("torch", imported)


if __name__ == "__main__":
    unittest.main()
