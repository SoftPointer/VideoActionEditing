from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "methods/bernini_action_editing/scripts/auh_stage_b_t0_single_update_20260824_retry8.sh"
)


class Retry8LauncherStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_fresh_revision_paths_and_permanent_claim(self) -> None:
        self.assertIn("source_stage_b_t0_retry8", self.source)
        self.assertIn("stage_b_t0_retry8/target_t0/0be6494dfac3", self.source)
        self.assertIn(".single_update.retry8.attempt_claim.json", self.source)
        self.assertIn("use retry9", self.source)
        self.assertNotIn("rm -", self.source)

    def test_world4_mi210_contract(self) -> None:
        self.assertIn("--gres=gpu:mi210:4", self.source)
        self.assertIn("--nproc_per_node=4", self.source)
        self.assertIn("ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_COUNT=4", self.source)
        self.assertIn("torch.cuda.device_count()", self.source)
        self.assertIn("HIP_VISIBLE_DEVICES", self.source)
        self.assertIn("CUDA_VISIBLE_DEVICES", self.source)

    def test_real_runner_and_one_step_validation(self) -> None:
        self.assertIn("train_action_repr_target_t0_canary_retry8_v1.py", self.source)
        self.assertIn("validate_published_t0_output", self.source)
        self.assertIn(".optimization_steps == 1", self.source)
        self.assertIn(".parameter_updates > 0", self.source)
        self.assertIn("SINGLE_UPDATE_PASS", self.source)

    def test_same_runtime_g2a_gate_is_required(self) -> None:
        self.assertIn("batch_replay_diagnostic", self.source)
        self.assertIn("source_posterior_matches_historical == false", self.source)
        self.assertIn("matched_native_batch_matches_historical == false", self.source)
        self.assertIn("same_runtime_g2a_gate", self.source)
        self.assertIn("route_off_and_six_zero_init_routes_exact_native_bits", self.source)
        self.assertIn("cross_run_historical_match_required == false", self.source)

    def test_no_target_media_or_decode_interface(self) -> None:
        self.assertNotIn("--target-video", self.source)
        self.assertNotIn("--anchor-video", self.source)
        self.assertIn(".decoded_video_generated == false", self.source)
        self.assertIn(".ours_model_claimed == false", self.source)
        self.assertIn(".quality_success_claimed == false", self.source)

    def test_cache_and_timeout_are_explicit(self) -> None:
        for name in (
            "XDG_CACHE_HOME",
            "MIOPEN_USER_DB_PATH",
            "MIOPEN_CUSTOM_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
        ):
            self.assertIn(name, self.source)
        self.assertIn("--kill-after=60s 60m", self.source)

    def test_preflight_launch_worker_status_modes_exist(self) -> None:
        for mode in ("preflight)", "launch)", "worker)", "status)"):
            self.assertIn(mode, self.source)
        self.assertIn("FRESH_UNCLAIMED", self.source)
        self.assertIn("COMPLETED_VALID", self.source)
        self.assertIn("CLAIMED_INCOMPLETE", self.source)


if __name__ == "__main__":
    unittest.main()
