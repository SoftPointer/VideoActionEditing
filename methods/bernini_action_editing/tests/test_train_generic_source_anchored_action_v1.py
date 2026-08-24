from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = METHOD_ROOT / "train_generic_source_anchored_action_v1.py"
SOURCE = RUNNER_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: F401
    import train_generic_source_anchored_action_v1 as runner

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    runner = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class GenericSourceAnchoredRunnerContractTests(unittest.TestCase):
    def test_r_profiles_are_the_only_current_executable_profiles(self) -> None:
        for fragment in (
            '"smoke-r"',
            '"stage-r64"',
            '"smoke-p"',
            '"smoke-o"',
            '"resume-po40"',
            '"action-only40"',
            'IMPLEMENTED_EXECUTION_PROFILES = ("smoke-r", "stage-r64")',
            "execution profile does not have an implemented runtime",
        ):
            self.assertIn(fragment, SOURCE)
        self.assertIn('args.experiment != "joint_source_anchored_v1"', SOURCE)

    def test_world4_is_one_shared_dp1_sp4_model(self) -> None:
        for fragment in (
            "runtime.parallel_topology(core.TOPOLOGY)",
            "distributed.world_size != core.WORLD_SIZE",
            "distributed.topology.dp_size != core.DP_SIZE",
            "distributed.topology.sp_size != core.SP_SIZE",
            '"one_shared_model": True',
            '"same_logical_row_on_all_ranks": True',
            '"rank_action_family_partition": False',
            '"gradient_sync": "SP4_mean"',
        ):
            self.assertIn(fragment, SOURCE)
        self.assertNotIn("DP2", SOURCE)
        self.assertNotIn("dog", SOURCE.casefold())
        self.assertNotIn("human", SOURCE.casefold())

    def test_r_uses_real_source_index0_same_noise_flow_matching(self) -> None:
        for fragment in (
            "source_data.PinnedPhysicalSourceOnlyPosteriorStore(",
            'memory_input_kind="same_noise_forward_noised_source"',
            "visual.no_op_flow_matching_loss(",
            "target_velocity=packed.target_velocity",
            '"same_epsilon_and_sigma_target_memory": True',
            '"synthetic_target_index1_bytes_read": False',
            '"generated_media_read": False',
            '"action_family_used_for_routing": False',
        ):
            self.assertIn(fragment, SOURCE)
        for forbidden in (
            "pyarrow",
            "pq.read",
            "posterior_list",
            "generated_video_path",
            "generated_latent",
            "teacher_velocity",
        ):
            self.assertNotIn(forbidden, SOURCE)

    def test_r64_is_one_canonical_pass_with_registered_sigma_map(self) -> None:
        for fragment in (
            'manifest.rows_for_split("train")',
            'core.STAGE_UPDATES["R"]',
            "object_sha256(row.receipt())",
            'schedule = core.fixed_sigma_schedule("R")',
            'if profile == "smoke-r":',
            "schedule = schedule[:1]",
            "zip(coordinates, selected_rows)",
        ):
            self.assertIn(fragment, SOURCE)

    def test_checkpoint_is_resume_bound_but_not_an_action_result(self) -> None:
        for fragment in (
            'R_CHECKPOINT_NAME = "stage_r_composite_checkpoint.pt"',
            'R_MIDPOINT_CHECKPOINT_NAME = "stage_r_u032_composite_checkpoint.pt"',
            'step_zero_based + 1 == 32',
            '"stage_update": 32',
            '"component_state": _component_cpu_state(composite)',
            '"optimizer_state": optimizer.state_dict()',
            '"resume_po40_authorized": args.execution_profile == "stage-r64"',
            '"complete_action_result": False',
            '"planner_updates": 0',
            '"operator_updates": 0',
            '["R"] if args.execution_profile == "stage-r64" else []',
            '[] if args.execution_profile == "stage-r64" else ["R"]',
            '"completed_stage_updates": {"R": len(coordinates)}',
        ):
            self.assertIn(fragment, SOURCE)

    def test_pair_invariants_bind_initial_po_and_defer_unavailable_manifests(self) -> None:
        for fragment in (
            '"status": "partial_r_only_action_manifest_fields_deferred"',
            '"representation_manifest_sha256": None',
            '"source_pair_manifest_sha256": None',
            '"action_row_order_sha256": None',
            '"stage_r_source_row_order_sha256": source_row_order_sha256',
            '"planner_initial_sha256": initial_component_sha256["P"]',
            '"operator_initial_sha256": initial_component_sha256["O"]',
            '"o_sigma_mapping": list(core.fixed_sigma_schedule("O"))',
        ):
            self.assertIn(fragment, SOURCE)

    def test_memory_gate_is_strict_and_reports_every_rank(self) -> None:
        for fragment in (
            "float(item[\"gpu_peak_reserved_gib\"]) >= gpu_limit_gib",
            "float(item[\"host_peak_rss_gib\"]) >= host_limit_gib",
            "float(item[\"host_cgroup_current_gib\"]) >= host_limit_gib",
            "float(item[\"host_cgroup_peak_gib\"]) >= host_limit_gib",
            '"gpu_peak_reserved_gib_by_rank": gpu_peak_reserved_by_rank',
            '"host_peak_rss_gib_by_rank": host_peak_rss_by_rank',
            '"host_cgroup_peak_gib_by_rank": host_cgroup_peak_by_rank',
            'resource_milestones["model_load"]',
            'resource_milestones["first_forward"]',
            'resource_milestones["first_backward"]',
            'resource_milestones["first_optimizer_step"]',
        ):
            self.assertIn(fragment, SOURCE)

    def test_rank0_transactions_are_symmetric_and_output_is_private(self) -> None:
        for fragment in (
            "def _rank0_call(",
            "dist.broadcast_object_list(box, src=0, group=world_group)",
            'label="private output creation"',
            "os.mkdir(output, mode=0o700)",
            "os.chmod(output, 0o700)",
            'label="Stage R u032 checkpoint write"',
            'label="final Stage R checkpoint write"',
            'label="final Stage R receipt publication"',
        ):
            self.assertIn(fragment, SOURCE)

    def test_terminal_toctou_and_real_gradient_gates_are_explicit(self) -> None:
        for fragment in (
            "def terminal_toctou_audit()",
            "file_sha256(source_manifest_path)",
            "file_sha256(checkpoint_manifest_path)",
            "terminal_checkpoint_content_identity = native_r.validate_checkpoint_content(",
            "checkpoint content changed during Stage R",
            "legacy.validate_source_trees(",
            "native_r.audit_packed_sp_sources(",
            'label="loaded frozen base transformer"',
            'label="terminal frozen base transformer"',
            "frozen base transformer changed during Stage R",
            'label="terminal TOCTOU audit"',
            "component_gradient_norms = _carrier_gradient_group_norms(active)",
            'component_gradient_norms["output"] <= 0.0',
            '"real_output_gradient_positive": True',
        ):
            self.assertIn(fragment, SOURCE)

    def test_history_precedes_complete_receipt_and_sp_route_is_normalized(self) -> None:
        publication = SOURCE[SOURCE.index("def publish_receipts()") :]
        self.assertLess(
            publication.index('output / "history.json"'),
            publication.index('output / "run_receipt.json"'),
        )
        for fragment in (
            "def _logical_route_receipt(",
            'if key not in {"rank", "sp_rank", "route_receipt"}',
            '"route_receipts_by_rank"',
            'item["route_receipt"]["sequence_parallel_rank"]',
        ):
            self.assertIn(fragment, SOURCE)

    def test_inactive_po_bytes_are_checked_after_every_r_step(self) -> None:
        for fragment in (
            'inactive_before = core.frozen_inactive_snapshot(composite, "R")',
            "optimizer_controller.assert_inactive_unchanged(inactive_before)",
            'final_component_sha256["P"] != initial_component_sha256["P"]',
            'final_component_sha256["O"] != initial_component_sha256["O"]',
            'final_component_sha256["R"] == initial_component_sha256["R"]',
        ):
            self.assertIn(fragment, SOURCE)

    def test_runner_has_no_remote_or_parent_job_mutation(self) -> None:
        lowered = SOURCE.casefold()
        for forbidden in (
            "subprocess",
            "ssh ",
            "scancel",
            "scontrol",
            "slurm_job_id",
            "os.kill",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertIn('"parent_allocation_released": False', SOURCE)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class GenericSourceAnchoredRunnerDynamicTests(unittest.TestCase):
    def test_sp_rank_route_receipts_share_one_logical_projection(self) -> None:
        rows = []
        for rank in range(4):
            value = {
                "total_tokens": 40,
                "condition_tokens": 19,
                "target_tokens": 21,
                "sequence_parallel_rank": rank,
                "sequence_parallel_size": 4,
                "enabled": True,
                "memory_digest": "a" * 64,
                "query_rows": "local_target_suffix_only",
                "key_value_rows": (
                    "independent_registered_source_visual_memory_only"
                ),
            }
            rows.append(
                {**value, "digest": runner.object_sha256(value)}
            )
        logical = [runner._logical_route_receipt(row) for row in rows]
        self.assertEqual(logical, [logical[0]] * 4)
        self.assertEqual(
            [row["sequence_parallel_rank"] for row in rows], list(range(4))
        )
        hostile = dict(rows[0])
        hostile["extra"] = True
        with self.assertRaises(runner.GenericSourceAnchoredTrainingError):
            runner._logical_route_receipt(hostile)


if __name__ == "__main__":
    unittest.main()
