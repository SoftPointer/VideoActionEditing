#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/auh_infer_native_branch_homotopy_dual4.sbatch"


class NativeBranchHomotopyLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_requests_one_full_eight_gpu_node_and_runs_two_waves(self) -> None:
        self.assertIn("#SBATCH --nodes=1", self.source)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn("wave1-fit-compatibility", self.source)
        self.assertIn("wave2-heldout-confirmation", self.source)
        self.assertGreaterEqual(self.source.count("run_wave"), 3)
        self.assertIn('launch_group "${dog_cell}" 0,1,2,3', self.source)
        self.assertIn('launch_group "${human_cell}" 4,5,6,7', self.source)
        self.assertEqual(self.source.count("--nproc_per_node=4"), 1)

    def test_all_four_predeclared_cells_are_postflighted_without_aggregation(self) -> None:
        for cell in (
            "fit-dog-7b88",
            "fit-human-a35b",
            "confirmation-dog-841b",
            "confirmation-human-a66e",
        ):
            self.assertIn(cell, self.source)
        self.assertIn('"fit_and_confirmation_never_aggregated": True', self.source)
        self.assertIn('"single_example_conclusion_authorized": False', self.source)
        self.assertIn('"two_sequential_waves": True', self.source)

    def test_source_closure_contains_runner_core_patch_registry_tests_launcher(self) -> None:
        for name in (
            "infer_native_branch_homotopy_canary.py",
            "native_branch_homotopy_v1.py",
            "native_branch_homotopy_runtime_v1.py",
            "source_self_native_ref_contrastive_v3.py",
            "test_native_branch_homotopy_v1.py",
            "test_native_branch_homotopy_runtime_v1.py",
            "test_infer_native_branch_homotopy_canary.py",
            "test_auh_native_branch_homotopy_launcher.py",
            "auh_infer_native_branch_homotopy_dual4.sbatch",
            "wrong_family_prompt_swap_pilot_registry_v1.json",
        ):
            self.assertIn(name, self.source)
        self.assertIn('closure=("${repo_closure[@]}" "${registry}")', self.source)
        self.assertIn("runtime-source-closure-sha256", self.source)

    def test_exact81_exact40_apg_and_schedule_are_postflighted(self) -> None:
        self.assertIn('sampling.get("frame_count") != 81', self.source)
        self.assertIn('sampling.get("num_inference_steps") != 40', self.source)
        self.assertIn('sampling.get("flow_shift_from_renderer_config") != 5.0', self.source)
        self.assertIn('sampling.get("eta") != 0.5', self.source)
        self.assertIn('sampling.get("norm_thresholds") != [50.0, 50.0]', self.source)
        self.assertIn('sampling.get("momentum") != 0.0', self.source)
        self.assertIn(
            "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2",
            self.source,
        )
        self.assertIn('sampling.get("high_endpoint_step_indices") != list(range(15))', self.source)
        self.assertIn('sampling.get("transition_step_indices") != list(range(15, 31))', self.source)
        self.assertIn('sampling.get("low_endpoint_step_indices") != list(range(31, 40))', self.source)

    def test_true_endpoints_and_homotopy_trace_are_postflighted(self) -> None:
        self.assertIn("set(traces) != set(arms)", self.source)
        self.assertNotIn("list(traces) != arms", self.source)
        self.assertIn('low.get("guidance_mode") != "v2v_apg"', self.source)
        self.assertIn('high.get("guidance_mode") != "r2v_apg"', self.source)
        self.assertIn('hom.get("transformer_forwards") != 200', self.source)
        self.assertIn('hom.get("low_vi_forwards") != 80', self.source)
        self.assertIn('hom.get("high_r2v4_forwards") != 120', self.source)
        self.assertIn('hom.get("patch_vae_latent_calls") != 400', self.source)
        self.assertIn('hom.get("original_scheduler_calls") != 40', self.source)
        self.assertIn('hom.get("low_official_apg_exact_parity_all_steps") is not True', self.source)
        self.assertIn('hom.get("scheduler_mutation_surface") != "model_output_argument_only"', self.source)

    def test_mode_native_prompt_cochange_and_frozen_boundary_are_explicit(self) -> None:
        self.assertIn(
            'prompts.get("positive_task_prefix_and_visual_regime_change_together") is not True',
            self.source,
        )
        self.assertIn(
            'prompts.get("shared_vr2v_positive_embedding_across_endpoints") is not False',
            self.source,
        )
        self.assertIn('row.get("training_performed") is not False', self.source)
        self.assertIn('row.get("optimizer_created") is not False', self.source)
        self.assertIn('row.get("parameter_update") is not False', self.source)

    def test_postflight_reopens_videos_clean_latents_and_official_gaussians(self) -> None:
        self.assertIn("decoded_video_contract", self.source)
        self.assertIn("safetensor_raw_sha", self.source)
        self.assertIn("normalized_clean_latent", self.source)
        self.assertIn("official_initial_gaussian", self.source)
        self.assertIn("live_output_postflight", self.source)
        self.assertIn("live_clean_latent_postflight", self.source)
        self.assertIn("live_initial_gaussian_postflight", self.source)


if __name__ == "__main__":
    unittest.main()
