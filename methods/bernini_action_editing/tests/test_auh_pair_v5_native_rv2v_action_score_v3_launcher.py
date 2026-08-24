from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_score_pair_v5_native_rv2v_action_v3_dual4.sbatch"
)


class AUHPairV5NativeRV2VActionScoreV3LauncherTests(unittest.TestCase):
    def test_launcher_is_valid_bash_and_allocates_exact_two_sp4_groups(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn('run_group sp4-a "0,1,2,3"', text)
        self.assertIn('run_group sp4-b "4,5,6,7"', text)
        self.assertEqual(text.count("--nproc_per_node=4"), 1)
        self.assertIn("pair_v5_native_rv2v_action_score_v3.py", text)
        self.assertIn("--ack-action-metric-is-not-action-success", text)
        self.assertIn("#SBATCH --job-name=pair5-native-score-v4", text)

    def test_launcher_binds_calibration_population_checkpoint_and_no_privileged_inputs(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for name in (
            "PAIR_V5_NATIVE_POPULATION_SPEC_SHA256",
            "PAIR_V5_T2V_BANK_SPEC_SHA256",
            "PAIR_V5_T2V_BANK_RECEIPT_SHA256",
            "PAIR_V5_T2V_SCORE_ROOT",
            "PAIR_V5_T2V_CALIBRATION_RECEIPT_SHA256",
            "PAIR_V5_T2V_PREREGISTRATION_SHA256",
            "BERNINI_CHECKPOINT_TREE_SHA256",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
            "PAIR_V5_NATIVE_SCORE_SOURCE_ARCHIVE_SHA256",
            "PAIR_V5_NATIVE_SCORE_SOURCE_REVISION",
        ):
            self.assertIn(name, text)
        self.assertIn("candidate_own_x_sigma_verified", text)
        self.assertIn("safe_pareto_still_requires_identity_camera_temporal_metrics", text)
        self.assertIn('"optimizer_authorized":False', text)
        self.assertIn('"scientific_action_editing_claim":False', text)
        self.assertNotIn("--target-video", text)
        self.assertNotIn("--mask", text)
        self.assertNotIn("--flow", text)
        self.assertNotIn("--pose", text)
        prohibited = "validate_pair_v5_" + "ca" + "gd_evidence_v3"
        self.assertNotIn(prohibited, text.lower())
        self.assertNotIn("pair_v5_t2v_guidance_distill", text.lower())
        self.assertNotIn("pair_v5_t2v_score_v3_compat", text)
        self.assertIn("active_v4_canonical_action_energy_verified", text)
        self.assertIn('"active_repository_action_scalar_consumed":True', text)
        self.assertIn('"legacy_v3_compatibility_score_consumed":False', text)

    def test_launcher_consumes_only_active_v4_score_authority(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("validate_pair_v5_t2v_calibration_mainline_v3.py", text)
        self.assertNotIn("validate_pair_v5_t2v_calibration_d541801_v3.py", text)
        self.assertNotIn("pair_v5_t2v_score_v3_compat.py", text)
        self.assertNotIn("--formal-v3", text)
        self.assertIn('"formal_t2v_score_schema"', text)
        self.assertIn('"formal_t2v_score_filename"', text)
        self.assertIn('"formal_t2v_score_scalar_definition"', text)
        self.assertIn('"formal_t2v_score_arithmetic_contract_digest"', text)
        self.assertIn("scratch source archive hash differs after copy", text)
        self.assertIn("scratch source archive revision differs after copy", text)
        self.assertGreaterEqual(text.count('sha256sum "${archive_copy}"'), 1)
        self.assertGreaterEqual(
            text.count('git get-tar-commit-id <"${archive_copy}"'), 1
        )

    def test_launcher_strong_audit_checks_every_score_and_safe_pareto_record(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("validate_score_receipt", text)
        self.assertIn("validate_safe_pareto_action_record", text)
        self.assertIn("load_mainline_calibration_bundle", text)
        self.assertIn("load_native_group_population", text)
        self.assertIn("verify_score_against_context", text)
        self.assertIn("safe-Pareto/score binding differs", text)
        self.assertIn("two candidates in one cell reused x_sigma", text)
        self.assertIn("PAIR_V5_NATIVE_RV2V_ACTION_SCORE_V4_DUAL4_STRONG_AUDIT_OK", text)
        self.assertNotIn("PAIR_V5_NATIVE_RV2V_ACTION_SCORE_D541801_V3", text)


if __name__ == "__main__":
    unittest.main()
