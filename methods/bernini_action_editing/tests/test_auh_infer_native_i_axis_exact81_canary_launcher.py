from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts/auh_infer_native_i_axis_exact81_canary_dual4.sbatch"


class NativeIAxisDualSP4LauncherTests(unittest.TestCase):
    def test_shell_syntax_and_all8_request(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", source)
        self.assertIn("#SBATCH --time=36:00:00", source)
        self.assertIn("launch_group dog 0,1,2,3", source)
        self.assertIn("launch_group human 4,5,6,7", source)
        self.assertIn('export ROCR_VISIBLE_DEVICES="${visible_gpus}"', source)
        self.assertIn("--nproc_per_node=4", source)
        self.assertNotIn("--nproc_per_node=8", source)

    def test_two_groups_are_concurrent_and_each_runs_all_arms_two_seeds(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        dog_launch = source.index("launch_group dog 0,1,2,3")
        human_launch = source.index("launch_group human 4,5,6,7")
        first_wait = source.index('wait "${dog_pid}"')
        self.assertLess(dog_launch, first_wait)
        self.assertLess(human_launch, first_wait)
        self.assertIn("N-C,N-W,G-C,G-W,G-P,G-D,G-S", source)
        self.assertIn("candidates=28", source)
        self.assertIn('len(candidates) == 14', source)
        self.assertIn('len(seeds) == 2', source)

    def test_launcher_runs_contract_tests_before_gpu_decode(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('running_launcher_path="$(realpath -e -- "$0")"', source)
        self.assertIn('repo_launcher_path="${method_root}/scripts/', source)
        self.assertIn(
            "running Slurm launcher differs from repository launcher", source
        )
        self.assertNotIn("launcher is outside repository method root", source)
        for test_name in (
            "test_native_i_axis_guidance.py",
            "test_infer_native_i_axis_exact81_canary.py",
            "test_auh_infer_native_i_axis_exact81_canary_launcher.py",
        ):
            self.assertIn(test_name, source)
        self.assertLess(
            source.index("for test_file in"),
            source.index("launch_group dog 0,1,2,3"),
        )

    def test_parent_receipt_enforces_exact40_parity_and_artifacts(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('[33, 34, 35, 36, 37]', source)
        self.assertIn('steps[index]["final_native_parity"]', source)
        self.assertIn('steps[index]["scheduler_received_original_model_output_object"]', source)
        self.assertIn('row["native_formula_exact_parity"]', source)
        self.assertIn('row["original_scheduler_call_count"] == 1', source)
        self.assertIn('native_velocity_raw_sha256', source)
        self.assertIn('executed_velocity_raw_sha256', source)
        self.assertIn('normalized_clean_latent', source)
        self.assertIn('os.O_EXCL', source)
        self.assertIn('all8-receipt.json', source)

    def test_no_training_or_scoring_command_is_launched(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('"training_performed": False', source)
        self.assertIn('"optimizer": None', source)
        self.assertIn('"ranking_performed": False', source)
        self.assertIn('"best_arm_selected": False', source)
        self.assertNotIn("train_lora.py \\", source)
        self.assertNotIn("torchrun --nproc_per_node=8", source)


if __name__ == "__main__":
    unittest.main()
