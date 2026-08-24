#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import fields
import hashlib
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import train_graft_phase_a_active14_transaction_v1 as active14
    import run_graft_phase_a_active14_transaction_gpu_v1 as runner

    RUNTIME_AVAILABLE = True
except (ImportError, OSError):
    active14 = None  # type: ignore[assignment]
    runner = None  # type: ignore[assignment]
    RUNTIME_AVAILABLE = False


@unittest.skipUnless(RUNTIME_AVAILABLE, "active14 torch runtime is required")
class Active14RunnerContractTests(unittest.TestCase):
    def test_core_pin_and_fixed_schedule_are_exact(self) -> None:
        self.assertEqual(
            runner.PINNED_ACTIVE14_CORE_SHA256,
            hashlib.sha256(Path(active14.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(active14.ACTIVE_INDICES, tuple(range(26, 40)))
        self.assertEqual(runner.PLAN_SCHEMA_VERSION, "bernini-graft-phase-a-active14-world8-plan-v1")
        self.assertEqual(
            runner.RUNTIME_CLOSURE_SCHEMA_VERSION,
            "bernini-graft-phase-a-active14-runtime-python-closure-v1",
        )

    def test_parser_inherits_field14_and_adds_fail_closed_active14_inputs(self) -> None:
        parser = runner.build_parser()
        destinations = {action.dest for action in parser._actions}  # noqa: SLF001
        inherited = {
            "plan_path",
            "expected_plan_sha256",
            "expected_field14_core_sha256",
            "expected_field14_runner_sha256",
            "expected_field14_source_commit",
            "ack_exact40_no_grad_diagnostic_no_checkpoint_no_scientific_claim",
        }
        added = {
            "active14_plan_path",
            "expected_active14_plan_sha256",
            "upstream_field14_receipt_path",
            "expected_upstream_field14_receipt_sha256",
            "expected_upstream_field14_job_id",
            "expected_active14_core_sha256",
            "expected_active14_runner_sha256",
            "ack_active14_fresh_optimizer_14_update_transaction_no_checkpoint_no_scientific_claim",
        }
        self.assertTrue(inherited <= destinations)
        self.assertTrue(added <= destinations)

    def test_downstream_api_is_two_phase_and_context_is_complete(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(runner.replay_active14_for_downstream).parameters),
            ("args", "routing", "prepare", "finalize"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(runner.Active14ContinuationContext)),
            (
                "topology",
                "backend",
                "renderer",
                "diffusion",
                "transformer",
                "handle",
                "bindings",
                "schedule",
                "fit",
                "confirmation",
                "negative_condition",
                "noop_condition",
                "action_condition",
                "device",
                "local_rank",
                "bernini_revision",
                "checkpoint_root",
                "checkpoint_content_identity",
                "checkpoint_manifest_sha256",
                "short_receipt",
                "field14_receipt",
                "active14_precommit_receipt",
                "trainable_final_digest",
                "frozen_base_final_digest",
            ),
        )

    @staticmethod
    def _packet(rank: int):
        family = "dog" if rank < 4 else "human"
        arm = rank // 4
        commit = active14.seal_mapping(
            {
                "schema_version": active14.SCHEMA_VERSION,
                "family": family,
                "all_fourteen_updates_completed": True,
                "transaction_committed_in_memory": True,
                "optimizer_contract": {"schedule_indices": list(range(26, 40))},
                "initial_trainable_digest": str(arm + 1) * 64,
                "final_trainable_digest": str(arm + 3) * 64,
                "final_frozen_base_digest": "9" * 64,
                "checkpoint_written": False,
                "publication_performed": False,
                **{name: False for name in active14.AUTHORITY_FIELDS},
            }
        )
        outer = active14.seal_mapping(
            {
                "schema_version": "cpu-active14-outer-v1",
                "active14_commit_receipt_digest": commit["digest"],
                "checkpoint_written": False,
                "publication_performed": False,
                **{name: False for name in active14.AUTHORITY_FIELDS},
            }
        )
        return {
            "global_rank": rank,
            "active14_commit_receipt_digest": commit["digest"],
            "active14_commit_receipt": commit,
            "transaction_receipt_digest": outer["digest"],
            "transaction_receipt": outer,
            "checkpoint_written": False,
            "publication_performed": False,
            **{name: False for name in active14.AUTHORITY_FIELDS},
        }

    def test_world8_assembly_requires_all_ranks_and_sp4_consensus(self) -> None:
        result = runner._assemble_world8_packets(  # noqa: SLF001
            [self._packet(rank) for rank in range(8)]
        )
        self.assertTrue(result["all_eight_active14_transactions_completed"])
        self.assertTrue(result["both_sp4_arms_parameter_consensus"])
        self.assertFalse(result["checkpoint_written"])
        attacked = [self._packet(rank) for rank in range(8)]
        attacked[3]["active14_commit_receipt"] = self._packet(4)[
            "active14_commit_receipt"
        ]
        attacked[3]["active14_commit_receipt_digest"] = attacked[3][
            "active14_commit_receipt"
        ]["digest"]
        with self.assertRaises(runner.Active14GPUError):
            runner._assemble_world8_packets(attacked)  # noqa: SLF001

    def test_source_has_no_checkpoint_writer_and_reruns_all_stages(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("torch.save", source)
        self.assertIn("execute_authenticated_short_run", source)
        self.assertIn("execute_exact40_sweep", source)
        self.assertIn("execute_active14_transaction", source)
        self.assertIn("prepare_bridge", source)
        self.assertIn("finalize_bridge", source)
        self.assertIn("weights_inherited_from_dependency_job", source)


if __name__ == "__main__":
    unittest.main()
