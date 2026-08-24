#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_infer_caper_native_counterfactual_siblings_all8.sbatch"
)


class AUHCaperNativeCounterfactualSiblingLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax_and_one_node_dual_world4_sp4(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn("#SBATCH --cpus-per-task=32", self.source)
        self.assertIn("#SBATCH --mem=256G", self.source)
        self.assertIn("--nproc_per_node=4", self.source)
        self.assertIn('unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL', self.source)
        self.assertIn('export ROCR_VISIBLE_DEVICES="${visible}"', self.source)
        self.assertIn('launch_group "${left}" 0,1,2,3', self.source)
        self.assertIn('launch_group "${right}" 4,5,6,7', self.source)
        left = self.source.index('launch_group "${left}" 0,1,2,3')
        right = self.source.index('launch_group "${right}" 4,5,6,7')
        wait = self.source.index('wait "${left_pid}"', right)
        self.assertLess(left, right)
        self.assertLess(right, wait)

    def test_exactly_one_public_fit_source_k4_and_no_seed_filter(self) -> None:
        cells = re.findall(
            r"\bfit-([0-9a-f]{16})-s(202608180[1-4])\b", self.source
        )
        self.assertEqual(
            set(cells),
            {
                ("7b88a1ca1f804f41", str(seed))
                for seed in range(2026081801, 2026081805)
            },
        )
        self.assertIn("source_id=\"7b88a1ca1f804f41\"", self.source)
        self.assertIn("arms=(target noop incomplete phase-order-violation)", self.source)
        self.assertIn('for ((index=0; index<${#cells[@]}; index+=2))', self.source)
        self.assertNotIn("CELL_FILTER", self.source)
        self.assertNotIn("BEST_SEED", self.source)
        self.assertNotIn("REPLACEMENT_SEED", self.source)
        self.assertNotIn("a66e6818e4144928", self.source)
        self.assertNotIn("CAPER_NATIVE_KSEED_ENABLE_LOCKBOX", self.source)

    def test_each_seed_gets_truthful_attempt_receipt_and_failures_survive(self) -> None:
        self.assertIn("write_attempt_receipt()", self.source)
        self.assertIn(
            'write_attempt_receipt "${left}" "${left_status}"', self.source
        )
        self.assertIn(
            'write_attempt_receipt "${right}" "${right_status}"', self.source
        )
        self.assertIn(
            '"attempt_status": "completed_success" if attempt_success else "completed_failure"',
            self.source,
        )
        self.assertIn(
            '"cell_process_attempt_recorded_even_on_failure": True', self.source
        )
        self.assertIn('"all_four_arm_outcomes_closed": attempt_success', self.source)
        self.assertIn(
            '"unobserved_or_incomplete_arm_outcomes_possible": not attempt_success',
            self.source,
        )
        self.assertNotIn(
            '"seed_and_all_arms_attempt_recorded_even_on_failure": True', self.source
        )
        self.assertIn('"seed_discarded": False', self.source)
        self.assertIn('"retry_or_replacement_seed_authorized": False', self.source)
        self.assertIn(
            'attempt_paths = sorted((root / "attempts").glob("*.json")', self.source
        )
        self.assertIn('"failed_attempts": failed_attempts', self.source)
        self.assertIn(
            '"all_four_arm_outcomes_closed": attempt.get("all_four_arm_outcomes_closed")',
            self.source,
        )
        first_attempt_writer = self.source.index("write_attempt_receipt()")
        root_aggregate = self.source.index(
            'attempt_paths = sorted((root / "attempts").glob("*.json")'
        )
        self.assertLess(first_attempt_writer, root_aggregate)

    def test_root_receipt_requires_complete_16_rollout_population(self) -> None:
        self.assertIn(
            '"population_decision": "PASS_COMPLETE" if population_complete else "NO_GO_INCOMPLETE_OR_FAILED_ATTEMPTS"',
            self.source,
        )
        self.assertIn('"expected_rollout_count": 16', self.source)
        self.assertIn('"complete_rollout_count": len(children) * 4', self.source)
        self.assertIn('"required_output_contract_exact81": True', self.source)
        self.assertIn(
            '"all_completed_outputs_verified_exact81": population_complete',
            self.source,
        )
        self.assertIn(
            '"all_completed_trajectories_verified_exact40_shift5": population_complete',
            self.source,
        )
        self.assertIn('len(attempts) == 4', self.source)
        self.assertIn('len(failed_attempts) == 0', self.source)
        self.assertIn('len(children) == 4', self.source)
        self.assertIn(
            'cross_k_invariants["each_cell_uses_one_source_latent_for_all_four_arms"] is True',
            self.source,
        )
        self.assertIn('"cross_seed_source_latent_byte_identity_required": False', self.source)
        self.assertIn('"observed_distinct_source_latent_raw_sha256_count"', self.source)
        self.assertIn('"same_checkpoint_all_k4_cells"', self.source)
        self.assertIn('"same_live_exact40_schedule_all_k4_cells"', self.source)
        self.assertIn('target = root / "population-receipt.json"', self.source)
        self.assertIn("raise SystemExit(0 if population_complete else 3)", self.source)

    def test_postflight_reads_every_exact81_latent_gaussian_and_arm_receipt(self) -> None:
        self.assertIn("import av", self.source)
        self.assertIn("from safetensors import safe_open", self.source)
        self.assertIn("frames != 81", self.source)
        self.assertIn("abs(fps - 25.0)", self.source)
        self.assertIn('tensor_raw_sha(clean, "normalized_clean_latent"', self.source)
        self.assertIn('tensor_raw_sha(noise, "official_initial_gaussian"', self.source)
        self.assertIn('f"{arm}.receipt.json"', self.source)
        self.assertIn('f"{arm}.mp4"', self.source)
        self.assertIn('f"{arm}.normalized-clean-latent.safetensors"', self.source)
        self.assertIn('f"{arm}.official-initial-gaussian.safetensors"', self.source)
        self.assertIn("len(gaussian_hashes) != 1", self.source)
        self.assertIn(
            'sampling.get("official_gaussian_captured_by_read_only_observer") is not True',
            self.source,
        )
        self.assertIn(
            'sampling.get("external_initial_noise_injection") is not False',
            self.source,
        )
        self.assertIn('shared.get("shared_high_sigma_prefix") is not False', self.source)
        self.assertIn(
            'trace.get("denoiser_or_scheduler_field_hook_installed") is not False',
            self.source,
        )
        self.assertIn('trace.get("initial_noise_observer_read_only") is not True', self.source)
        self.assertIn('trace.get("complete_independent_stock_steps") != 40', self.source)

    def test_source_archive_registry_and_exact_five_new_files_are_closed(self) -> None:
        required_new = (
            "infer_caper_native_counterfactual_siblings_v1.py",
            "caper_native_counterfactual_siblings_sit_v1.json",
            "auh_infer_caper_native_counterfactual_siblings_all8.sbatch",
            "test_infer_caper_native_counterfactual_siblings_v1.py",
            "test_auh_infer_caper_native_counterfactual_siblings_all8_launcher.py",
        )
        for name in required_new:
            self.assertIn(name, self.source)
        for dependency in (
            "infer_caper_native_kseed_population_v1.py",
            "infer_t2v_v2v_branch_homotopy_canary.py",
            "infer_native_identity_generation_canary.py",
            "tri_branch_unipc.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_source_value_residual_oracle.py",
        ):
            self.assertIn(dependency, self.source)
        self.assertIn(
            "09b1bdc13a91ded9fe853f9582499da310995c0bc4d48e50599f52cb0de154e0",
            self.source,
        )
        self.assertIn('sha256sum "${source_archive}"', self.source)
        self.assertIn('sha256sum "${registry}"', self.source)
        self.assertIn("git get-tar-commit-id", self.source)
        self.assertIn('git -C "${repo_root}" diff --quiet', self.source)
        self.assertIn('git -C "${repo_root}" diff --cached --quiet', self.source)
        self.assertIn('"training_performed": False', self.source)
        self.assertIn('"optimizer_created": False', self.source)
        self.assertIn('"parameter_update": False', self.source)
        self.assertIn('"preference_admission_performed": False', self.source)


if __name__ == "__main__":
    unittest.main()
