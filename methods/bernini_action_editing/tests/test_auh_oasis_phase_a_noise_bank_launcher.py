from __future__ import annotations

from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_infer_oasis_phase_a_noise_bank_dual4.sbatch"
RUNTIME = METHOD_ROOT / "infer_oasis_phase_a_noise_bank.py"


class AUHOASISNoiseBankLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")

    def test_all_eight_gpus_are_two_concurrent_independent_sp4_groups(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.launcher)
        self.assertEqual(self.launcher.count("--nproc_per_node=4"), 1)
        self.assertIn('launch_family dog_sit_hold "0,1,2,3"', self.launcher)
        self.assertIn('launch_family human_stand_hold "4,5,6,7"', self.launcher)
        self.assertIn("dog_pid=$!", self.launcher)
        self.assertIn("human_pid=$!", self.launcher)
        self.assertIn('wait "${dog_pid}"', self.launcher)
        self.assertIn('wait "${human_pid}"', self.launcher)
        self.assertIn(
            "two_concurrent_independent_WORLD4_Ulysses_SP4_on_all8",
            self.launcher,
        )
        self.assertNotIn("--nproc_per_node=8", self.launcher)

    def test_family_and_combined_cardinality_are_exact(self) -> None:
        self.assertIn('receipt.get("rollout_count") != 12', self.launcher)
        self.assertIn('receipt.get("source_cell_count") != 2', self.launcher)
        self.assertIn("len(all_candidate_ids) != 24", self.launcher)
        self.assertIn("len(all_triplet_audit_digests) != 8", self.launcher)
        self.assertIn('"source_cell_count": 4', self.launcher)
        self.assertIn('"candidate_count": 24', self.launcher)
        self.assertIn('"matched_triplet_count": 8', self.launcher)

    def test_source_instruction_seed_are_matched_across_rho(self) -> None:
        for token in (
            "core.validate_matched_rollout_triplet",
            "source_video_sha256=cell.source_video_sha256",
            "edit_instruction_sha256=cell.edit_instruction_sha256",
            'source_conditioning_digest=source_row["source_conditioning_digest"]',
            'source_frame_set_digest=source_row["source_frame_set_digest"]',
            "seed=seed",
            "matched source/instruction/seed audit differs",
            '"candidate_id_set_digest"',
            '"matched_triplet_audit_set_digest"',
        ):
            self.assertIn(token, self.launcher)

    def test_native_guidance_is_fixed_and_only_randn_is_interposed(self) -> None:
        for token in (
            "guidance=fixed-native-rv2v",
            '"bernini.models.wan_diffusion.randn_tensor"',
            'boundary.get("sample_one_step_replaced") is False',
            'boundary.get("native_cfg_or_apg_replaced") is False',
            'boundary.get("native_scheduler_replaced") is False',
            '"fixed_native_rv2v_no_ablation"',
        ):
            self.assertIn(token, self.launcher)
        self.assertIn('"sample_one_step_replaced": False', self.runtime)
        self.assertNotIn('setattr(wan_diffusion_module, "sample_one_step"', self.runtime)

    def test_operator_receipt_matches_the_actual_pure_operator_schema(self) -> None:
        for token in (
            'operator_receipt.get("trainer_integration_executed") is False',
            'operator_receipt.get("operator_self_registers_sampler_hook") is False',
            'operator_receipt.get("operator_self_registers_launcher") is False',
            'operator_receipt.get("ablation_only") is True',
            'operator_receipt.get("scientific_claim_authorized") is False',
            'runtime_binding.get("inference_integration_executed") is True',
        ):
            self.assertIn(token, self.launcher)
        for stale in (
            'operator_receipt.get("trainer_integration")',
            'operator_receipt.get("inference_integration")',
            'operator_receipt.get("launcher_registration")',
        ):
            self.assertNotIn(stale, self.launcher)
        self.assertIn(
            'information_flow.get("text_input") == "complete_action_caption"',
            self.launcher,
        )
        self.assertIn('"text_input": "complete_action_caption"', self.runtime)

    def test_archive_manifest_checkpoint_and_completion_are_hash_closed(self) -> None:
        for token in (
            "OASIS_PHASE_A_SOURCE_ARCHIVE_SHA256",
            "OASIS_PHASE_A_SOURCE_REVISION",
            "OASIS_PHASE_A_MANIFEST_SHA256",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
            "BERNINI_CHECKPOINT_TREE_SHA256",
            "git get-tar-commit-id",
            "scratch source archive hash differs after copy",
            "read-only extracted method tree changed",
            'receipt.get("method_source_revision") != source_revision',
            'receipt.get("method_source_archive_sha256") != source_archive_sha',
            'checkpoint.get("tree_sha256") == checkpoint_tree_sha',
            'checkpoint.get("content_manifest_file_sha256")',
            '"family_receipt_file_sha256"',
            '"method_source_archive_sha256"',
            '"checkpoint_content_manifest_file_sha256"',
        ):
            self.assertIn(token, self.launcher)
        self.assertIn(
            '"methods/bernini_action_editing/oasis_phase_a_core.py"',
            self.launcher,
        )
        self.assertIn("for test_file in", self.launcher)
        self.assertIn('"${method_root}/tests/${test_file}"', self.launcher)
        self.assertNotIn('"${python_bin}" -B -m unittest \\', self.launcher)

    def test_exact81_artifacts_and_rho_parent_bindings_are_audited(self) -> None:
        for token in (
            'native_sampling.get("num_frames") == 81',
            'native_sampling.get("num_inference_steps") == 40',
            'check_file(baseline, "file_sha256")',
            'check_file(injected, "file_sha256")',
            'check_file(endpoint, "sha256")',
            'check_file(endpoint["normalized_clean_latent"], "sha256")',
            'row.get("rho_zero_exact_native_object_forwarded") is True',
            'row.get("active_noise_parent_matches_official_control") is True',
            "matched parent Gaussian differs",
        ):
            self.assertIn(token, self.launcher)

    def test_launcher_has_no_action_scorer_training_selection_or_success_authority(self) -> None:
        for token in (
            '"endpoint_scoring_performed": False',
            '"endpoint_selection_performed": False',
            '"external_action_scorer_consumed": False',
            '"optimizer_or_training_authorized": False',
            '"training_performed": False',
            '"scientific_action_editing_success_claim": False',
        ):
            self.assertIn(token, self.launcher)
        for forbidden in (
            "mace_candidate_action_energy",
            "pair_v5_native_rv2v_action_score",
            "optimizer.step",
            "backward()",
            "--learning-rate",
        ):
            self.assertNotIn(forbidden, self.launcher)
            self.assertNotIn(forbidden, self.runtime)


if __name__ == "__main__":
    unittest.main()
