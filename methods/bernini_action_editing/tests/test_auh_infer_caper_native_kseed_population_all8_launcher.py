#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_infer_caper_native_kseed_population_all8.sbatch"
)


class AUHCaperNativeKSeedPopulationLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax_and_all8_world4_sp4_contract(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn("#SBATCH --cpus-per-task=32", self.source)
        self.assertIn("#SBATCH --mem=256G", self.source)
        self.assertIn("--nproc_per_node=4", self.source)
        self.assertIn('launch_group "${left}" 0,1,2,3', self.source)
        self.assertIn('launch_group "${right}" 4,5,6,7', self.source)
        left = self.source.index('launch_group "${left}" 0,1,2,3')
        right = self.source.index('launch_group "${right}" 4,5,6,7')
        wait = self.source.index('wait "${left_pid}"', right)
        self.assertLess(left, right)
        self.assertLess(right, wait)

    def test_fit_is_default_and_lockbox_is_explicit_second_stage(self) -> None:
        self.assertIn(
            'population_phase="${CAPER_NATIVE_KSEED_PHASE:-fit}"', self.source
        )
        self.assertIn(
            '"${CAPER_NATIVE_KSEED_ENABLE_LOCKBOX:-}" == "thresholds-frozen-open-lockbox-v1"',
            self.source,
        )
        self.assertIn("CAPER_NATIVE_KSEED_THRESHOLD_FREEZE_RECEIPT", self.source)
        self.assertIn("CAPER_NATIVE_KSEED_THRESHOLD_FREEZE_SHA256", self.source)
        self.assertIn("CAPER_NATIVE_KSEED_FIT_POPULATION_RECEIPT", self.source)
        self.assertIn(
            "threshold freeze does not bind supplied fit master receipt", self.source
        )
        self.assertIn("--lockbox-second-stage-enabled", self.source)
        self.assertIn("--threshold-freeze-receipt", self.source)
        self.assertIn("--expected-threshold-freeze-sha256", self.source)
        fit_branch = self.source.index('if [[ "${population_phase}" == fit ]]')
        lockbox_switch = self.source.index("thresholds-frozen-open-lockbox-v1")
        source_hash_loop = self.source.index('for index in "${!source_paths[@]}"')
        self.assertLess(fit_branch, lockbox_switch)
        self.assertLess(lockbox_switch, source_hash_loop)

    def test_complete_fit_and_lockbox_cell_banks_are_literal_and_unfiltered(self) -> None:
        fit_cells = set(
            re.findall(r"\bfit-([0-9a-f]{16})-s(202608180[1-4])\b", self.source)
        )
        lockbox_cells = set(
            re.findall(r"\blockbox-([0-9a-f]{16})-s(202608190[1-2])\b", self.source)
        )
        self.assertEqual(
            fit_cells,
            {
                (source_id, str(seed))
                for source_id in (
                    "7b88a1ca1f804f41",
                    "841b5e0080a1441d",
                    "402059390cdb4f50",
                    "3be4072a63144b8f",
                )
                for seed in range(2026081801, 2026081805)
            },
        )
        self.assertEqual(
            lockbox_cells,
            {
                (source_id, str(seed))
                for source_id in ("6a7ebea80ba64f18", "ac87fea937864bd5")
                for seed in range(2026081901, 2026081903)
            },
        )
        self.assertIn('cells=("${fit_cells[@]}")', self.source)
        self.assertIn('cells=("${lockbox_cells[@]}")', self.source)
        self.assertNotIn("CELL_FILTER", self.source)
        self.assertNotIn("BEST_SEED", self.source)
        self.assertNotIn("a66e6818e4144928", self.source)

    def test_every_cell_gets_attempt_receipt_and_failure_is_no_go(self) -> None:
        self.assertIn("write_attempt_receipt()", self.source)
        self.assertIn(
            'write_attempt_receipt "${left}" "${left_status}"', self.source
        )
        self.assertIn(
            'write_attempt_receipt "${right}" "${right_status}"', self.source
        )
        self.assertIn('"attempt_status": "completed_success" if attempt_success else "completed_failure"', self.source)
        self.assertIn('"seed_attempt_recorded_even_on_failure": True', self.source)
        self.assertIn('"seed_discarded": False', self.source)
        self.assertIn('"retry_or_replacement_seed_authorized": False', self.source)
        self.assertIn(
            '"population_decision": "PASS_COMPLETE" if population_complete else "NO_GO_INCOMPLETE_OR_FAILED_ATTEMPTS"',
            self.source,
        )
        self.assertIn('"every_registered_seed_attempt_recorded"', self.source)
        self.assertIn('"failed_attempts": failed_attempts', self.source)
        first_write = self.source.index("write_attempt_receipt()")
        aggregator = self.source.index('attempt_paths = sorted((root / "attempts")')
        self.assertLess(first_write, aggregator)

    def test_native_only_exact81_exact40_and_real_artifact_postflight(self) -> None:
        self.assertIn(
            'native_arm="native-source-video-only-v2v-endpoint"', self.source
        )
        self.assertNotIn("pure-target-only-t2v-endpoint.mp4", self.source)
        self.assertNotIn("t2v-v2v-branch-homotopy-095-075.mp4", self.source)
        self.assertIn("import av", self.source)
        self.assertIn("from safetensors import safe_open", self.source)
        self.assertIn("frames != 81", self.source)
        self.assertIn("abs(fps - 25.0)", self.source)
        self.assertIn('tensor_raw_sha(clean, "normalized_clean_latent"', self.source)
        self.assertIn('tensor_raw_sha(noise, "official_initial_gaussian"', self.source)
        self.assertIn(
            'sampling.get("official_gaussian_is_sampler_initial_noise") is not True',
            self.source,
        )
        self.assertIn(
            'sampling.get("official_gaussian_captured_by_read_only_observer") is not True',
            self.source,
        )
        self.assertIn(
            'sampling.get("observer_injects_or_replaces_noise") is not False',
            self.source,
        )
        self.assertIn('sampling.get("external_initial_noise_injection") is not False', self.source)

    def test_registry_source_closure_and_no_update_are_fail_closed(self) -> None:
        required = (
            "infer_caper_native_kseed_population_v1.py",
            "caper_native_kseed_population_sit_v1.json",
            "infer_t2v_v2v_branch_homotopy_canary.py",
            "infer_native_identity_generation_canary.py",
            "tri_branch_unipc.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_source_value_residual_oracle.py",
            "test_infer_caper_native_kseed_population_v1.py",
            "test_auh_infer_caper_native_kseed_population_all8_launcher.py",
        )
        for name in required:
            self.assertIn(name, self.source)
        self.assertIn(
            "f700eba0d7097943a1c26d7663815a7bdfbf5cd8786c07da2189f716921a7da4",
            self.source,
        )
        self.assertIn('sha256sum "${source_archive}"', self.source)
        self.assertIn('sha256sum "${registry}"', self.source)
        self.assertIn("git get-tar-commit-id", self.source)
        self.assertIn('git -C "${repo_root}" diff --quiet', self.source)
        self.assertIn('git -C "${repo_root}" diff --cached --quiet', self.source)
        self.assertIn('row.get("training_performed") is not False', self.source)
        self.assertIn('row.get("optimizer_created") is not False', self.source)
        self.assertIn('row.get("parameter_update") is not False', self.source)
        self.assertIn('"training_performed": False', self.source)
        self.assertIn('"optimizer_created": False', self.source)
        self.assertIn('"parameter_update": False', self.source)


if __name__ == "__main__":
    unittest.main()
