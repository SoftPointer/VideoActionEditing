from __future__ import annotations

from pathlib import Path
import ast
import sys
from types import SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_pair_v6_scaid as trainer

try:
    import torch
except ModuleNotFoundError:
    torch = None


class PairV6SCAIDTrainerTests(unittest.TestCase):
    def test_exact40_partition_has_38_updates_and_two_zero_anchors(self) -> None:
        self.assertEqual([trainer.exact40_schedule_index(i) for i in range(40)], list(range(40)))
        self.assertEqual(trainer.expected_optimizer_updates(40), 38)
        self.assertEqual(trainer.expected_optimizer_updates(80), 76)
        with self.assertRaises(trainer.PairV6SCAIDTrainingError):
            trainer.expected_optimizer_updates(41)

    def test_runner_calls_authoritative_gate_and_leaf_vjp_core(self) -> None:
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        self.assertIn("load_authoritative_v3_authorization", calls)
        self.assertIn("run_scaid_cell", calls)
        self.assertIn("replay_native_student_vjp", calls)
        self.assertIn("native_runtime._build_pack", source)
        self.assertIn("leaf_vjp_mode=True", source)
        self.assertIn("native_student_component_serial_vjp_replay", source)
        self.assertIn("target_tail_equality_receipt", source)
        self.assertIn("aggregate_native_guidance_components(components)", source)
        self.assertIn("raw_caption_by_branch=spec.raw_caption_by_branch", source)
        self.assertIn(
            "expected_raw_caption_bank_sha256=spec.raw_caption_bank_sha256",
            source,
        )
        self.assertIn("request.prompt != self.task_prompts.get(request.branch)", source)
        self.assertNotIn("proposal_video", source)
        self.assertNotIn("target_video", source)

    def test_each_dp_arm_is_fixed_to_its_authoritative_event(self) -> None:
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        self.assertIn("event_index = contract.arm_index", source)
        self.assertIn("len({event.action_family for event in events}) != DP_SIZE", source)
        self.assertIn("len(rows) != DP_SIZE", source)

    @unittest.skipIf(torch is None, "torch runtime is required")
    def test_encoded_correct_wrong_native_geometry_must_match_exactly(self) -> None:
        source = torch.zeros(1, 16, 21, 4, 6)
        refs = tuple(torch.zeros(1, 16, 1, 4, 6) for _ in range(4))
        receipt = trainer.validate_native_condition_geometry(
            source, refs, source.clone(), tuple(value.clone() for value in refs)
        )
        self.assertTrue(receipt["native_pack_geometry_compatible"])
        with self.assertRaisesRegex(trainer.PairV6SCAIDTrainingError, "geometry differs"):
            trainer.validate_native_condition_geometry(
                source,
                refs,
                torch.zeros(1, 16, 21, 4, 8),
                tuple(torch.zeros(1, 16, 1, 4, 8) for _ in range(4)),
            )
        bad_refs = list(refs)
        bad_refs[0] = bad_refs[0].clone()
        bad_refs[0].reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(
            trainer.PairV6SCAIDTrainingError, "detached finite FP32"
        ):
            trainer.validate_native_condition_geometry(
                source, tuple(bad_refs), source.clone(), refs
            )

    def test_low_sigma_branch_never_calls_optimizer_step(self) -> None:
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        low_if = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "LOW_SIGMA_INDICES" in (ast.get_source_segment(source, node.test) or "")
        ]
        self.assertEqual(len(low_if), 1)
        body = "\n".join(ast.get_source_segment(source, node) or "" for node in low_if[0].body)
        orelse = "\n".join(ast.get_source_segment(source, node) or "" for node in low_if[0].orelse)
        self.assertNotIn("optimizer.step", body)
        self.assertIn("optimizer.step", orelse)

    def test_target_tail_content_equality_is_fail_closed(self) -> None:
        digest = "7" * 64
        receipt = trainer.target_tail_equality_receipt(
            SimpleNamespace(target_tail_sha256=digest),
            SimpleNamespace(
                target_tail_sha256_by_source_role={
                    "correct": digest,
                    "wrong": digest,
                }
            ),
        )
        self.assertTrue(receipt["same_x_sigma_target_tail_content_all_paths"])
        with self.assertRaisesRegex(trainer.PairV6SCAIDTrainingError, "target-tail"):
            trainer.target_tail_equality_receipt(
                SimpleNamespace(target_tail_sha256=digest),
                SimpleNamespace(
                    target_tail_sha256_by_source_role={
                        "correct": digest,
                        "wrong": "8" * 64,
                    }
                ),
            )

    def test_prompts_are_rebuilt_once_from_the_same_raw_caption(self) -> None:
        captions = {
            branch: f"A sealed raw caption for {branch}."
            for branch in trainer.scaid.BRANCH_ORDER
        }
        t2v, rv2v, receipt = trainer.build_task_prompt_registry(
            captions, prompt_cleaner=lambda value: value
        )
        t2v_prefix = trainer.native_infer.TASK_SYSTEM_PROMPTS[
            trainer.native_infer.ARM_TRAINING_TASK_NAMES["t2v"]
        ]
        rv2v_prefix = trainer.native_infer.TASK_SYSTEM_PROMPTS[
            trainer.native_infer.ARM_TRAINING_TASK_NAMES["rv2v"]
        ]
        self.assertEqual(receipt["raw_caption_bank_sha256"], trainer.object_sha256(captions))
        for branch, raw in captions.items():
            self.assertEqual(
                t2v[branch],
                t2v_prefix + trainer.native_infer.TASK_BINDING_CLAUSES["t2v"] + raw,
            )
            self.assertEqual(
                rv2v[branch],
                rv2v_prefix + trainer.native_infer.TASK_BINDING_CLAUSES["rv2v"] + raw,
            )
            self.assertNotEqual(t2v[branch], rv2v[branch])
            self.assertEqual(t2v[branch].count(t2v_prefix), 1)
            self.assertEqual(rv2v[branch].count(rv2v_prefix), 1)

        prefixed = dict(captions)
        prefixed[trainer.scaid.BRANCH_ORDER[0]] = t2v_prefix + "already wrapped"
        with self.assertRaisesRegex(trainer.PairV6SCAIDTrainingError, "double-wrap"):
            trainer.build_task_prompt_registry(
                prefixed, prompt_cleaner=lambda value: value
            )

    def test_component_serial_replay_uses_official_native_cfg_coefficients(self) -> None:
        self.assertEqual(
            dict(trainer.scaid.NATIVE_GUIDANCE_COMPONENTS),
            trainer.native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS,
        )

    def test_checkpoint_content_and_run_receipt_are_sealed(self) -> None:
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        parser = trainer.build_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertTrue(actions["checkpoint_content_manifest"].required)
        self.assertEqual(
            actions["expected_checkpoint_content_manifest_sha256"].default,
            trainer.source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256,
        )
        self.assertIn("source_audit.validate_checkpoint_content", source)
        self.assertIn("legacy.CHECKPOINT_TREE_SHA256", source)
        self.assertIn('(\"learning_rate\", \"max_grad_norm\")', source)
        self.assertIn('"receipt_digest": object_sha256(receipt_value)', source)
        self.assertIn('"artifacts": artifacts', source)
        self.assertIn('"archive_sha256": args.method_source_archive_sha256', source)
        self.assertIn('"training_config": {', source)
        self.assertIn('"residual_survival_receipt": dict(', source)
        self.assertIn('"loss_components": {', source)
        self.assertIn('"noise_seed": seed_value', source)

    def test_preflight_validates_source_commits_before_authorization(self) -> None:
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        preflight = source[source.index("def preflight(") : source.index("def _broadcast_sp")]
        self.assertLess(
            preflight.index("legacy.validate_source_trees"),
            preflight.index("scaid.load_authoritative_v3_authorization"),
        )

    def test_single_cell_high_sigma_smoke_is_explicit_and_nonzero_update(self) -> None:
        parser = trainer.build_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertIn("smoke_high_sigma_only", actions)
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        self.assertIn("action_adapter.HIGH_SIGMA_INDICES[0]", source)
        self.assertIn('"single_cell_high_sigma_smoke"', source)
        self.assertIn("expected_updates = (", source)


class PairV6SCAIDLauncherTests(unittest.TestCase):
    def test_launcher_is_eight_gpu_torchrun_dp2sp4_exact40(self) -> None:
        path = METHOD_ROOT / "scripts" / "auh_train_pair_v6_scaid_dp2sp4.sbatch"
        text = path.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn("--nproc_per_node=8", text)
        self.assertIn("topology=DP2xSP4", text)
        self.assertIn("schedule_steps % 40", text)
        self.assertIn("PAIR_V6_CAGD_V3_EVIDENCE_SHA256", text)
        self.assertIn("BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256", text)
        self.assertIn("--expected-checkpoint-content-manifest-sha256", text)
        self.assertIn("PAIR_V6_SCAID_HIGH_SIGMA_SMOKE_ONLY", text)
        self.assertIn("PAIR_V6_SCAID_HIGH_SIGMA_SMOKE_RECEIPT_SHA256", text)
        self.assertIn("--smoke-high-sigma-only", text)
        self.assertIn("--high-sigma-smoke-receipt", text)
        self.assertIn("train_pair_v6_scaid.py", text)
        self.assertIn("finalize_pair_v5_t2v_cagd_v3.py", text)
        self.assertIn("train_pair_v5_action_preference.py", text)
        self.assertNotIn("--mask", text)
        self.assertNotIn("--flow", text)
        self.assertNotIn("--pose", text)


if __name__ == "__main__":
    unittest.main()
