#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts/auh_infer_t2v_v2v_branch_homotopy_dual4.sbatch"


class T2VV2VBranchHomotopyLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_requests_one_full_eight_gpu_node_and_two_dual4_waves(self) -> None:
        self.assertIn("#SBATCH --nodes=1", self.source)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn("wave1-fit-compatibility", self.source)
        self.assertIn("wave2-heldout-confirmation", self.source)
        self.assertIn('launch_group "${dog_cell}" 0,1,2,3', self.source)
        self.assertIn('launch_group "${human_cell}" 4,5,6,7', self.source)
        self.assertEqual(self.source.count("--nproc_per_node=4"), 1)

    def test_all_four_cells_and_three_arms_are_fixed(self) -> None:
        for cell in (
            "fit-dog-7b88",
            "fit-human-a35b",
            "confirmation-dog-841b",
            "confirmation-human-a66e",
        ):
            self.assertIn(cell, self.source)
        for arm in (
            "native-source-video-only-v2v-endpoint",
            "pure-target-only-t2v-endpoint",
            "t2v-v2v-branch-homotopy-095-075",
        ):
            self.assertIn(arm, self.source)
        self.assertIn('"fit_and_confirmation_never_aggregated": True', self.source)
        self.assertIn('"single_example_conclusion_authorized": False', self.source)

    def test_clean_hash_bound_source_checkpoint_and_registry_closure(self) -> None:
        for name in (
            "infer_t2v_v2v_branch_homotopy_canary.py",
            "t2v_v2v_branch_homotopy_v1.py",
            "t2v_v2v_branch_homotopy_runtime_v1.py",
            "t2v_v2v_branch_homotopy_core4_v1.json",
            "test_t2v_v2v_branch_homotopy_v1.py",
            "test_t2v_v2v_branch_homotopy_runtime_v1.py",
            "test_infer_t2v_v2v_branch_homotopy_canary.py",
            "test_auh_t2v_v2v_branch_homotopy_launcher.py",
            "auh_infer_t2v_v2v_branch_homotopy_dual4.sbatch",
        ):
            self.assertIn(name, self.source)
        self.assertIn("git get-tar-commit-id", self.source)
        self.assertIn("git -C \"${repo_root}\" diff --quiet", self.source)
        self.assertIn("git -C \"${repo_root}\" diff --cached --quiet", self.source)
        self.assertIn("checkpoint_manifest", self.source)
        self.assertIn("source_closure_sha256", self.source)
        self.assertIn(
            "dc0088fd3e43b7667a0f2bce7bb55e867553897bdddc8fd737d589b62fd84e43",
            self.source,
        )

    def test_postflight_locks_exact81_exact40_and_homotopy_regions(self) -> None:
        self.assertIn('sampling.get("frame_count") != 81', self.source)
        self.assertIn('sampling.get("num_inference_steps") != 40', self.source)
        self.assertIn('sampling.get("flow_shift_from_renderer_config") != 5.0', self.source)
        self.assertIn('sampling.get("omega_text") != 4.0', self.source)
        self.assertIn('sampling.get("eta") != 0.5', self.source)
        self.assertIn('sampling.get("norm_threshold") != 50.0', self.source)
        self.assertIn('sampling.get("momentum") != 0.0', self.source)
        self.assertIn('sampling.get("high_endpoint_step_indices") != list(range(9))', self.source)
        self.assertIn('sampling.get("transition_step_indices") != list(range(9, 26))', self.source)
        self.assertIn('sampling.get("low_endpoint_step_indices") != list(range(26, 40))', self.source)
        self.assertIn(
            "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2",
            self.source,
        )

    def test_postflight_enforces_source_absent_t2v_and_source_only_v2v(self) -> None:
        self.assertIn("set(traces) != set(arms)", self.source)
        self.assertNotIn("list(traces) != arms", self.source)
        self.assertIn('conditions.get("pure_t2v_full_source_video_count") != 0', self.source)
        self.assertIn('conditions.get("pure_t2v_source_reference_count") != 0', self.source)
        self.assertIn('conditions.get("pure_t2v_visual_conditions_all_none") is not True', self.source)
        self.assertIn('conditions.get("source_v2v_full_source_video_count") != 1', self.source)
        self.assertIn('conditions.get("source_v2v_source_reference_count") != 0', self.source)
        self.assertIn('high.get("guidance_mode") != "t2v_apg"', self.source)
        self.assertIn('low.get("guidance_mode") != "v2v_apg"', self.source)

    def test_postflight_enforces_prompt_shapes_vendor_hash_and_no_updates(self) -> None:
        self.assertIn('prompts.get("embedding_shape") != [1, 512, 4096]', self.source)
        self.assertIn('prompts.get("same_negative_embedding_object_all_branches") is not True', self.source)
        self.assertIn('revisions.get("wan_diffusion_sha256")', self.source)
        self.assertIn('row.get("training_performed") is not False', self.source)
        self.assertIn('row.get("optimizer_created") is not False', self.source)
        self.assertIn('row.get("parameter_update") is not False', self.source)
        self.assertIn('hom.get("optimizer_created") is not False', self.source)
        self.assertIn('hom.get("parameters_updated") is not False', self.source)
        self.assertIn('freeze.get("base_frozen") is not True', self.source)
        self.assertIn(
            'freeze.get("exact_parameter_and_buffer_bytes_hashed") is not True',
            self.source,
        )

    def test_postflight_reopens_media_clean_latents_and_official_gaussians(self) -> None:
        self.assertIn("video_contract", self.source)
        self.assertIn("tensor_raw_sha", self.source)
        self.assertIn("normalized_clean_latent", self.source)
        self.assertIn("official_initial_gaussian", self.source)
        self.assertIn("live_output_postflight", self.source)
        self.assertIn("live_clean_latent_postflight", self.source)
        self.assertIn("live_initial_gaussian_postflight", self.source)

    def test_high_uses_native_apg_but_trajectory_equality_is_not_claimed(self) -> None:
        self.assertIn(
            'disclosure.get("homotopy_high_same_state_target_only_t2v_apg_contract_certified") is not True',
            self.source,
        )
        self.assertIn(
            'branch_apg.get("function") != "bernini.models.wan_diffusion.normalized_guidance"',
            self.source,
        )
        self.assertIn(
            '"standalone_and_homotopy_trajectory_equality_claim_authorized": False',
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
