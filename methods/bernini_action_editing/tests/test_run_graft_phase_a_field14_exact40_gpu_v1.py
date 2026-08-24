#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = METHOD_ROOT / "run_graft_phase_a_field14_exact40_gpu_v1.py"
CORE_PATH = METHOD_ROOT / "graft_phase_a_field14_exact40_v1.py"
PLAN_PATH = METHOD_ROOT / "assets/graft_phase_a_field14_exact40_world8_plan_v1.json"
SHORT_RUNNER_PATH = METHOD_ROOT / "run_graft_phase_a_a_lite_short_gpu_v1.py"
SHORT_TRAINER_PATH = METHOD_ROOT / "train_graft_phase_a_a_lite_short_v1.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


class Field14GPURunnerStaticTests(unittest.TestCase):
    def test_existing_short_sources_remain_exactly_frozen(self) -> None:
        self.assertEqual(
            _sha(SHORT_RUNNER_PATH),
            "4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4",
        )
        self.assertEqual(
            _sha(SHORT_TRAINER_PATH),
            "73e39048bb8836fef33516eb1aae4cbc3f9fa4ecefcfb5d2695925bcb150f7bb",
        )

    def test_plan_is_canonical_and_binds_new_runtime_bytes(self) -> None:
        raw = PLAN_PATH.read_bytes()
        plan = json.loads(raw.decode("ascii"))
        canonical = json.dumps(
            plan,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii") + b"\n"
        self.assertEqual(raw, canonical)
        self.assertEqual(plan["runtime"]["field14_core_sha256"], _sha(CORE_PATH))
        self.assertEqual(plan["runtime"]["field14_runner_sha256"], _sha(RUNNER_PATH))
        self.assertEqual(
            plan["runtime"]["short_runner_sha256"], _sha(SHORT_RUNNER_PATH)
        )
        self.assertEqual(
            plan["runtime"]["field14_source_commit"],
            "f9ef982e976ad19ed81ed075d33c9221952945e4",
        )
        self.assertEqual(
            plan["runtime"]["short_source_commit"],
            "a884d357a6c0742f751be48d226ba72c952bae76",
        )
        self.assertEqual(plan["exact40_indices"], list(range(40)))
        self.assertEqual(plan["inactive_indices"], list(range(26)))
        self.assertEqual(plan["active_indices"], list(range(26, 40)))
        self.assertEqual(plan["field_roles"], [
            "source_noop_target_velocity",
            "correct_atlas_noop_velocity",
            "wrong_atlas_noop_velocity",
            "dropped_atlas_noop_velocity",
            "correct_atlas_action_velocity",
            "dropped_atlas_action_velocity",
        ])
        self.assertTrue(plan["afterok_is_queue_gate_only"])
        self.assertFalse(plan["inherits_weights_from_dependency"])
        self.assertTrue(plan["no_checkpoint"])
        self.assertTrue(all(value is False for value in plan["authority"].values()))

    def test_live_order_is_short_then_no_grad_exact40(self) -> None:
        source = _function_source(RUNNER_PATH, "_run_official_gpu")
        short_position = source.index("execute_authenticated_short_run")
        sweep_position = source.index("execute_exact40_sweep")
        self.assertLess(short_position, sweep_position)
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("trainable_before_sweep", source)
        self.assertIn("trainable_after_sweep", source)
        self.assertIn("if trainable_before_sweep != trainable_after_sweep", source)
        self.assertIn("base_before != base_after", source)
        self.assertIn("canonical_json_bytes(dict(assembled))", source)
        self.assertIn("short_result_plain = json.loads(", source)
        self.assertIn("exact40_result_plain = json.loads(", source)
        self.assertIn("pickle.dumps(local_packet", source)

    def test_full_preinstall_baseline_and_per_index_release_are_explicit(self) -> None:
        baseline = _function_source(RUNNER_PATH, "_capture_full_preinstall_baseline")
        measure_source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("for index in field14.INACTIVE_INDICES", baseline)
        self.assertIn('purpose="adapter-off-parity"', baseline)
        self.assertIn("torch.cuda.empty_cache()", baseline)
        self.assertIn("correct_wrong_drop_negative_raw_byte_exact", measure_source)
        self.assertIn("correct_wrong_drop_noop_raw_byte_exact", measure_source)
        self.assertIn("correct_drop_action_raw_byte_exact", measure_source)
        self.assertIn("all_same_condition_raw_equal_preinstall", measure_source)
        self.assertIn("def release_index", measure_source)

    def test_no_checkpoint_output_or_optimizer_surface_is_added(self) -> None:
        runner_source = RUNNER_PATH.read_text(encoding="utf-8")
        core_source = CORE_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "torch.save",
            "torch.optim",
            'add_argument("--output',
            "write_text(",
            "write_bytes(",
        ):
            self.assertNotIn(forbidden, runner_source)
            self.assertNotIn(forbidden, core_source)
        self.assertNotIn("optimizer.step", runner_source)
        self.assertIn("checkpoint_written\": False", runner_source)
        self.assertIn("checkpoint_payload_returned\": False", runner_source)
        self.assertIn("publication_performed\": False", runner_source)

    def test_resource_plan_is_exact_world8_48h(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="ascii"))
        self.assertEqual(
            plan["resources"],
            {
                "cpus_per_task": 64,
                "gpus": 8,
                "memory_gib": 256,
                "nodes": 1,
                "ntasks": 1,
                "time_limit_hours": 48,
            },
        )
        self.assertEqual(
            plan["topology"],
            {
                "allocation": "single-node-8xMI210",
                "dp_size": 2,
                "sp_size": 4,
                "world_size": 8,
            },
        )


if __name__ == "__main__":
    unittest.main()
