from __future__ import annotations

from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_preflight_oasis_phase_a_dp2sp4.sbatch"
RUNTIME = METHOD_ROOT / "infer_oasis_phase_a_frozen_controller.py"


class OASISPhaseALauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")

    def test_all_eight_gpus_and_world8_dp2sp4_are_explicit(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.launcher)
        self.assertIn("--nproc_per_node=8", self.launcher)
        self.assertIn("topology=WORLD8/DP2xSP4", self.launcher)
        self.assertNotIn("--nproc_per_node=4", self.launcher)
        self.assertIn('"dp_size": 2', self.runtime)
        self.assertIn('"sequence_parallel_size": 4', self.runtime)

    def test_launcher_is_honestly_preflight_only(self) -> None:
        self.assertIn("--preflight-only", self.launcher)
        self.assertIn("OASIS_ACK_BACKEND_NOT_EXECUTED", self.launcher)
        self.assertIn("controller_runtime=false rollouts=0", self.launcher)
        self.assertIn('"frozen_oracle_execution_authorized": False', self.runtime)
        self.assertIn('"controller_tensor_core_executed": False', self.runtime)
        self.assertIn('"model_forward_count": 0', self.runtime)
        self.assertIn('"native_scheduler_step_count": 0', self.runtime)
        self.assertNotIn("ACTION_EDITING_SUCCESS", self.launcher)
        self.assertIn('"scientific_action_editing_success_claim": False', self.runtime)

    def test_no_training_or_checkpoint_artifact_is_exposed(self) -> None:
        for forbidden in (
            "optimizer.step",
            "torch.optim",
            "adapter.safetensors",
            "optimizer.pt",
            "--learning-rate",
            "backward()",
        ):
            self.assertNotIn(forbidden, self.launcher)
            self.assertNotIn(forbidden, self.runtime)
        self.assertIn('"training_performed": False', self.runtime)
        self.assertIn('"parameter_mutation_performed": False', self.runtime)

    def test_exact81_exact40_and_tail_base_off_are_bound(self) -> None:
        self.assertIn("exact81=true exact40=true low38_39=base-off", self.launcher)
        self.assertIn('"exact81": True', self.runtime)
        self.assertIn('"exact40": True', self.runtime)
        self.assertIn('"low_sigma_exact_base_off_indices": [38, 39]', self.runtime)
        self.assertIn("final_clean_probe=true arbitrary_state_oracle=NO-GO", self.launcher)
        self.assertIn(
            '"candidate_score_coordinate": "final_clean_candidate_registered_renoise"',
            self.runtime,
        )
        self.assertIn('"arbitrary_native_state_velocity_norm_oracle": "NO-GO"', self.runtime)

    def test_source_archive_and_all_authorities_are_hash_bound(self) -> None:
        for token in (
            "OASIS_PHASE_A_SOURCE_ARCHIVE_SHA256",
            "OASIS_PHASE_A_SOURCE_REVISION",
            "OASIS_PHASE_A_MANIFEST_SHA256",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
            "BERNINI_CHECKPOINT_TREE_SHA256",
            "git get-tar-commit-id",
            "read-only extracted method tree changed",
        ):
            self.assertIn(token, self.launcher)
        self.assertIn("load_dedicated_scalar_calibration_evidence", self.runtime)
        self.assertIn("dedicated T2V scalar calibration prerequisite failed", self.runtime)
        self.assertNotIn("validate_pair_v5_cagd_evidence_v3", self.runtime)
        self.assertNotIn("finalize_pair_v5_t2v_cagd_v3.py", self.launcher)
        self.assertIn("validate_checkpoint_content", self.runtime)

    def test_candidate_backend_boundary_does_not_claim_old_one_step_replay(self) -> None:
        self.assertIn(
            "independent_full_exact40_candidate_grid_from_same_registered_gaussian",
            self.runtime,
        )
        self.assertIn("candidate_owned_registered_epsilon_sigma_renoise_probe", self.runtime)
        self.assertNotIn("reversible_candidate_local_clone_restore", self.runtime)
        self.assertNotIn("same_object_candidate_x_next", self.runtime)

    def test_wrongref_and_privileged_inputs_cannot_authorize(self) -> None:
        self.assertIn('"wrongref_proxy_used_for_authorization": False', self.runtime)
        self.assertIn('"target_or_paired_target_consumed": False', self.runtime)
        self.assertIn('"t2v_media_or_latent_consumed_by_rv2v": False', self.runtime)
        self.assertIn('"mask_flow_pose_track_consumed": False', self.runtime)

    def test_receipt_artifact_closure_cannot_be_mistaken_for_oracle_output(self) -> None:
        self.assertIn('{"preflight-receipt.json"}', self.launcher)
        self.assertIn('"artifact_kind": "preflight_receipt_not_oracle_result"', self.runtime)
        self.assertIn('"frozen_oracle_rollouts_executed": 0', self.runtime)
        self.assertIn('"scientific_action_editing_success_claim": False', self.runtime)


if __name__ == "__main__":
    unittest.main()
