from __future__ import annotations

from pathlib import Path
import ast
import unittest

from methods.bernini_action_editing import (
    clean_source_visual_context_pair_controller_v1 as pair,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER = METHOD_ROOT / "train_clean_source_visual_context_stage_b_v1.py"
COMMON = METHOD_ROOT / "scripts/auh_train_clean_source_visual_context_stage_b_holder_v1.sh"


class CleanSourceVisualContextBackwardPreflightTests(unittest.TestCase):
    def test_scope_is_real_backward_but_has_no_optimizer_or_checkpoint_path(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        helper = source[
            source.index("def run_backward_feasibility_microbatches(") :
            source.index("def _atomic_adapter_safetensors(")
        ]
        branch = source[
            source.index('if args.execution_scope == "backward-feasibility-preflight":', source.index("def main(")) :
            source.index('if admission is None:', source.index("def main("))
        ]
        self.assertIn("for coordinate in coordinates:", helper)
        self.assertIn("scaled_loss.backward()", helper)
        self.assertIn("runtime.synchronize_gradients(trainable, parallel)", helper)
        self.assertIn("stage_b_gradients_digest(trainable)", helper)
        self.assertIn("component_norms = grouped_gradient_norms(trainable)", helper)
        self.assertNotIn("torch.optim", helper + branch)
        self.assertNotIn("optimizer.step(", helper + branch)
        self.assertNotIn("CheckpointCoordinator", helper + branch)
        self.assertIn('"optimizer_constructed": False', branch)
        self.assertIn('"checkpoint_root_created": False', branch)
        self.assertIn('"parameters_changed": False', branch)

    def test_parameter_digest_must_be_identical_before_and_after_backward(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        helper = source[
            source.index("def run_backward_feasibility_microbatches(") :
            source.index("def _atomic_adapter_safetensors(")
        ]
        self.assertIn('if after != before:', helper)
        self.assertIn('fail("backward feasibility changed adapter parameters")', helper)
        self.assertIn('"sha256_before": before', helper)
        self.assertIn('"sha256_after": after', helper)
        self.assertNotIn("parameter.data", helper)
        self.assertIn("parameter.grad = None", helper)

    def test_upstream_pair_gate_allows_only_method_release_rebinding(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        fields = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id == "_BACKWARD_UPSTREAM_INVARIANT_FIELDS"
                for target in node.targets
            ):
                fields = ast.literal_eval(node.value)
        self.assertIsNotNone(fields)
        self.assertIn("source_only_manifest_digest", fields)
        self.assertIn("initial_parameter_digest", fields)
        self.assertIn("exact40_schedule_sha256", fields)
        self.assertNotIn("method_source_revision", fields)
        gate = source[
            source.index("def validate_backward_upstream_pair_invariants(") :
            source.index("def audit_packed_sp_sources(")
        ]
        self.assertIn("if current_projection != upstream_projection:", gate)
        self.assertIn("exact_except_new_method_release_identity", gate)
        self.assertIn("method_source_revision", gate)

    def test_pair_receipt_requires_both_arms_and_no_update(self) -> None:
        shared = {"initial_parameter_digest": "a" * 64}

        def arm(kind: str) -> dict:
            return {
                "complete": True,
                "memory_input_kind": kind,
                "pair_invariants": shared,
                "upstream_structural_pair": {"receipt_digest": "b" * 64},
                "authority": {
                    "four_microbatch_forward_executed": True,
                    "four_microbatch_backward_executed": True,
                    "dp2_sp4_gradient_sync_executed": True,
                    "optimizer_constructed": False,
                    "optimizer_step_count": 0,
                    "parameters_changed": False,
                    "checkpoint_written": False,
                },
                "backward_feasibility": {
                    "microbatches_per_dp_arm": 4,
                    "logical_records": 8,
                    "parameters": {
                        "unchanged": True,
                        "sha256_before": "a" * 64,
                        "sha256_after": "a" * 64,
                    },
                    "gradient_sync": {
                        "finite_all_parameters_world8": True,
                        "identical_full_gradient_digest_world8": True,
                    },
                },
            }

        clean = arm("clean_source")
        noised = arm("same_noise_forward_noised_source")
        self.assertEqual(pair._assert_backward_pair_receipts(clean, noised), shared)
        noised["authority"]["parameters_changed"] = True
        with self.assertRaisesRegex(pair.CleanSourceVisualPairError, "closure differs"):
            pair._assert_backward_pair_receipts(clean, noised)

    def test_holders_bind_registered_pair_and_never_release_parents(self) -> None:
        common = COMMON.read_text(encoding="utf-8")
        controller = (
            METHOD_ROOT / "clean_source_visual_context_pair_controller_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"holder_job": 135980', controller)
        self.assertIn('"holder_job": 135981', controller)
        self.assertIn(
            "136140:auh7-1b-gpu-215:clean_source", common
        )
        self.assertIn(
            "136140:auh7-1b-gpu-215:same_noise_forward_noised_source", common
        )
        self.assertIn('"backward-feasibility-preflight"', controller)
        self.assertIn('environment["CSVC_HOLDER_JOB"]', controller)
        self.assertIn("CSVC_PREFLIGHT_PAIR_RECEIPT", common)
        self.assertIn("backward_microbatches_per_dp_arm=4", common)
        self.assertNotIn("scancel", (common + controller).lower())
        self.assertNotIn("scontrol release", (common + controller).lower())


if __name__ == "__main__":
    unittest.main()
