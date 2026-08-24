from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
COMPUTE = (
    METHOD_ROOT
    / "scripts/auh_infer_native_action_scaffold_identity_rebinding_dual4.sbatch"
)
SUBMITTER = (
    METHOD_ROOT
    / "scripts/auh_submit_native_action_scaffold_identity_rebinding_chain.sh"
)


class AUHRoleRebindingLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compute = COMPUTE.read_text(encoding="utf-8")
        cls.submitter = SUBMITTER.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        for path in (COMPUTE, SUBMITTER):
            result = subprocess.run(
                ["bash", "-n", str(path)], check=False,
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_compute_occupies_all_eight_gpus_as_two_world4_groups(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.compute)
        self.assertIn('group_a_visible_gpus="0,1,2,3"', self.compute)
        self.assertIn('group_b_visible_gpus="4,5,6,7"', self.compute)
        self.assertIn("--nproc_per_node=4", self.compute)
        self.assertEqual(self.compute.count("launch_group group-"), 2)
        self.assertIn('export ROCR_VISIBLE_DEVICES="${visible_gpus}"', self.compute)
        self.assertNotIn("srun ", self.compute)

    def test_fixed_four_four_arm_partition_is_exact(self) -> None:
        group_a = (
            "source-video-source-refs action-donor-source-refs "
            "\\\n  noop-donor-source-refs reverse-donor-source-refs"
        )
        group_b = (
            "action-donor-only source-action-source-refs "
            "\\\n  action-source-source-refs action-donor-wrong-refs"
        )
        self.assertIn(group_a, self.compute)
        self.assertIn(group_b, self.compute)
        for arm in (
            "source-video-source-refs", "action-donor-source-refs",
            "noop-donor-source-refs", "reverse-donor-source-refs",
            "action-donor-only", "source-action-source-refs",
            "action-source-source-refs", "action-donor-wrong-refs",
        ):
            self.assertIn(f'"{arm}"', self.compute)

    def test_both_groups_bind_same_sp4_a_cell_and_cross_group_gaussian(self) -> None:
        self.assertIn("--factor-execution-group sp4-a", self.compute)
        self.assertIn('"fixed_execution_group") != "sp4-a"', self.compute)
        self.assertIn('"proposal cell": lambda r:', self.compute)
        self.assertIn('"target Gaussian": lambda r:', self.compute)
        self.assertIn('"same_official_target_gaussian_all_eight_arms": True', self.compute)
        self.assertIn('noise.get("original_return_tensor_forwarded_by_identity") is not True', self.compute)

    def test_exact81_source_refs_and_frozen_no_privileged_inputs_are_sealed(self) -> None:
        self.assertIn('cdf_dog_source_sha256="5ed911f66fea3ed', self.compute)
        self.assertIn('cdf_dog_wrong_source_sha256="da7e3efa6f4fabac', self.compute)
        self.assertIn('wrong_source_video="${ROLE_REBINDING_WRONG_SOURCE_VIDEO:', self.compute)
        self.assertIn('--wrong-source-video "${wrong_source_video}"', self.compute)
        self.assertIn('receipt.get("wrong_source", {}).get("reference_indices") != [0, 27, 53, 80]', self.compute)
        self.assertIn('receipt.get("wrong_source", {}).get("paired_target_accessed") is not False', self.compute)
        self.assertIn('receipt.get("wrong_source", {}).get("paired_parquet_accessed") is not False', self.compute)
        self.assertIn('receipt.get("wrong_source", {}).get("precomputed_latent_accessed") is not False', self.compute)
        self.assertIn('"wrong_reference_arm_vi_source_ids": [1.0, 2.0, 3.0, 4.0, 5.0]', self.compute)
        self.assertIn('"two_video_correct_reference_native_vi_source_id_interpolation": [', self.compute)
        self.assertIn('"conditioning_source_id_extrapolation_used": False', self.compute)
        self.assertIn('receipt.get("source", {}).get("reference_indices") != [0, 27, 53, 80]', self.compute)
        self.assertIn('"frame_count": 81', self.compute)
        self.assertIn('"fps": 25', self.compute)
        self.assertIn('"proposal_mp4_consumed": False', self.compute)
        self.assertIn('"mask_flow_pose_track_trajectory": False', self.compute)
        self.assertIn('"training_or_optimization": False', self.compute)
        self.assertIn('"quality_claim": False', self.compute)

    def test_archive_is_hash_bound_read_only_and_contains_complete_closure(self) -> None:
        self.assertIn('git get-tar-commit-id <"${source_archive}"', self.compute)
        self.assertIn('find "${method_root}" -type f -exec chmod a-w', self.compute)
        for member in (
            "infer_native_action_scaffold_identity_rebinding_canary.py",
            "infer_native_multivideo_motion_donor_oracle.py",
            "infer_native_identity_generation_canary.py",
            "dmiq_t2v_factorial_bank.py",
            "infer_fitq_owner_prompt_cross_query_micro.py",
            COMPUTE.name,
            SUBMITTER.name,
        ):
            self.assertIn(member, self.compute)
        self.assertIn("executed launcher differs from archive", self.compute)

    def test_compute_does_not_submit_and_publishes_sealed_stage_receipt(self) -> None:
        self.assertNotIn("sbatch ", self.compute)
        self.assertIn('"complete": True', self.compute)
        self.assertIn('stage["receipt_digest"] = hashlib.sha256', self.compute)
        self.assertIn("os.replace(temporary, final)", self.compute)
        self.assertIn("os.fsync(descriptor)", self.compute)
        self.assertIn('stage.receipt.json', self.compute)

    def test_submitter_is_one_step_then_afterok_exact40(self) -> None:
        self.assertEqual(self.submitter.count("sbatch \\"), 2)
        self.assertIn("ROLE_REBINDING_NUM_INFERENCE_STEPS=1", self.submitter)
        self.assertIn("ROLE_REBINDING_NUM_INFERENCE_STEPS=40", self.submitter)
        self.assertIn('--dependency="afterok:${canary_job}"', self.submitter)
        self.assertIn("canary-step1", self.submitter)
        self.assertIn("exact40", self.submitter)
        self.assertIn("refusing reused output root", self.submitter)
        self.assertIn("compute launcher differs from archive", self.submitter)
        self.assertIn("submitter differs from archive", self.submitter)
        self.assertIn("ROLE_REBINDING_WRONG_SOURCE_VIDEO", self.submitter)


if __name__ == "__main__":
    unittest.main()
