from __future__ import annotations

from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_train_pair_v5_t2v_guidance_dp2sp4.sbatch"
)


class PairV5T2VGuidanceLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_requests_all_eight_gpus_and_runs_world8(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.text)
        self.assertIn("--nproc_per_node=8", self.text)
        self.assertIn("topology=DP2xUlysses-SP4", self.text)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=0,1,2,3", self.text)

    def test_exact81_exact40_and_low_gate_trainer_are_sealed(self) -> None:
        self.assertIn("schedule_steps >= 40 && schedule_steps % 40 == 0", self.text)
        self.assertIn("train_pair_v5_t2v_guidance_distill.py", self.text)
        self.assertIn("--ack-experimental-no-action-success-claim", self.text)
        self.assertIn("PAIR_V5_SAME_STATE_T2V_GUIDANCE_DP2SP4_STRONG_AUDIT_OK", self.text)

    def test_launcher_declares_no_rv2v_pseudo_target_or_privileged_carrier(self) -> None:
        self.assertIn("pure_t2v_role=same_coordinate_field_query_only", self.text)
        for declaration in (
            "rv2v_target=false",
            "rv2v_input=false",
            "rv2v_noise=false",
            "donor=false",
            "mask=false",
            "flow=false",
            "pose=false",
            "track=false",
        ):
            self.assertIn(declaration, self.text)

    def test_source_and_manifest_are_content_addressed_before_torchrun(self) -> None:
        self.assertIn('sha256sum "${source_archive}"', self.text)
        self.assertIn('sha256sum "${event_manifest}"', self.text)
        self.assertIn('sha256sum "${cagd_evidence}"', self.text)
        self.assertIn('sha256sum "${scorer_group_a}"', self.text)
        self.assertIn('sha256sum "${scorer_group_b}"', self.text)
        self.assertIn('sha256sum "${checkpoint_manifest}"', self.text)
        self.assertIn('git get-tar-commit-id <"${source_archive}"', self.text)
        self.assertLess(
            self.text.index('sha256sum "${event_manifest}"'),
            self.text.index("--nproc_per_node=8"),
        )

    def test_recomputed_v3_authorization_and_checkpoint_content_are_required(self) -> None:
        for flag in (
            "--cagd-validator-evidence",
            "--expected-cagd-validator-evidence-sha256",
            "--scorer-group-receipt",
            "--expected-scorer-group-receipt-sha256",
            "--checkpoint-content-manifest",
            "--expected-checkpoint-content-manifest-sha256",
        ):
            self.assertIn(flag, self.text)


if __name__ == "__main__":
    unittest.main()
