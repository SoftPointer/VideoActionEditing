from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_infer_fitq_official_runtime_scan.sbatch"


class AUHFITQOfficialRuntimeLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL)
        cls.runner = cls.source.split("runner_args=(", 1)[1].split(
            "audit_receipt()", 1
        )[0]

    def test_bash_and_embedded_python_parse(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.python_blocks), 6)
        for index, block in enumerate(self.python_blocks):
            with self.subTest(index=index):
                ast.parse(block)

    def test_uses_all_eight_gpus_as_two_legal_world4_groups(self) -> None:
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --gres=gpu:mi210:8",
            "#SBATCH --mem=256G",
        ):
            self.assertIn(directive, self.source)
        self.assertIn("topology=DP2xWORLD4/Ulysses4 allocated_gpus=8 groups=2", self.source)
        self.assertEqual(self.runner.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.source)
        self.assertEqual(self.runner.count("launch_group seed-"), 2)
        self.assertNotRegex(self.source, r"(?m)^\s*srun(?:\s|$)")

    def test_launches_read_only_fitq_runtime_with_exact81_context(self) -> None:
        self.assertIn(
            '"${method_root}/infer_fitq_official_runtime_scan.py"', self.runner
        )
        self.assertIn('--output-statistics-dir "${statistics_dir}"', self.runner)
        self.assertIn("--num-frames 81", self.runner)
        self.assertIn('--sigmas "${sigma_values[@]}"', self.runner)
        self.assertIn('--bridge-fractions "${bridge_fraction_values[@]}"', self.runner)
        self.assertNotRegex(self.runner, r'"\$\{method_root\}/train_[^"\n]+\.py"')
        for forbidden in (
            "--max-steps",
            "--learning-rate",
            "--optimizer",
            "--lora",
            "--adapter",
            "--save-checkpoint",
            "--num-frames 41",
        ):
            self.assertNotIn(forbidden, self.runner)

    def test_real_forward_count_and_hook_provenance_are_audited(self) -> None:
        for fragment in (
            "grid_forwards=84 duplicate_forwards=1 hook_off_references=1 actual_forwards_per_rank=86",
            'receipt.get("read_only_forward_hooks_present") is True',
            'receipt.get("forward_callback_present") is True',
            'receipt.get("custom_forward_core_present") is False',
            'receipt.get("custom_analysis_core_present") is True',
            'field.get("forwards_per_rank") == 86',
            'observation.get("observed_hooked_forwards_per_rank") == 85',
            'observation.get("total_official_forwards_per_rank") == 86',
            'observation.get("hook_on_off_field_parity", {}).get("byte_exact_equal") is True',
            'duplicate.get("block0_same_state_exact", {}).get("all") is True',
            'row.get("actual_runtime_evidence_digest") == actual_runtime_digest',
            'canonical_sha(field) == distributed.get("truthful_field_evidence_digest")',
            '== actual_runtime_digest, "actual hooked-runtime evidence digest differs"',
            'shadow.get("is_runtime_provenance") is False',
            'shadow.get("emitted_as_receipt") is False',
        ):
            self.assertIn(fragment, self.source)

    def test_controls_are_exact_noop_and_incomplete_branches(self) -> None:
        for fragment in (
            'ids != ["semantic_noop", "incomplete_approach_without_pickup"]',
            "FITQ A0 requires exact K=2 hard negatives",
            "digests[0] != noop_sha",
            "same-state action/no-op/incomplete parity failed",
        ):
            self.assertIn(fragment, self.source)

    def test_statistics_and_non_authorization_are_fail_closed(self) -> None:
        for fragment in (
            'observation.get("analysis_statistics") == "phase_head_mean_second_moment"',
            'artifact.get("mean_shape") == [121, 1, 21, 1, 1536]',
            'artifact.get("second_moment_shape") == [121, 1, 21, 1, 1536]',
            'artifact.get("contains_model_weights") is False',
            'artifact.get("is_checkpoint") is False',
            'observation.get("proposal_bank_status") == "insufficient_bank"',
            'observation.get("decision_scope") == "engineering_N0_like_diagnostic_only"',
            'observation.get("fitq_go_authorized") is False',
            'receipt.get("optimizer_update") == "null"',
            '"fitq_stage1_authorized": False',
            '"scientific_claim_authorized": False',
        ):
            self.assertIn(fragment, self.source)

    def test_checkpoint_is_rehashed_after_both_groups(self) -> None:
        for fragment in (
            "BERNINI_FITQ_POST_CHECKPOINT_BYTE_IDENTITY_OK",
            "post-FITQ checkpoint file set differs",
            "post-FITQ checkpoint content differs",
            '"verified_file_count": len(verified)',
            'field.get("checkpoint_content_identity") == post_checkpoint_identity',
            "model_weights_written=false",
        ):
            self.assertIn(fragment, self.source)

    def test_launcher_does_not_submit_or_mutate_git(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(
            self.source,
            r"(?m)^\s*git\s+(?:add|commit|push|reset|clean|checkout|switch)\b",
        )


if __name__ == "__main__":
    unittest.main()
