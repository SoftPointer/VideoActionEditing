from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts/auh_score_pair_v5_source_bound_preservation_v1_dual4.sbatch"
SCORER = METHOD_ROOT / "score_pair_v5_source_bound_preservation_v1.py"
CONTRACT = METHOD_ROOT / "pair_v5_source_bound_preservation_evaluator_v1.py"


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text()
        cls.scorer = SCORER.read_text()
        cls.contract = CONTRACT.read_text()

    def test_shell_syntax_and_disjoint_eight_gpu_execution(self) -> None:
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.launcher)
        self.assertIn('run_group sp4-a "0,1,2,3"', self.launcher)
        self.assertIn('run_group sp4-b "4,5,6,7"', self.launcher)
        self.assertIn("concurrent_disjoint_groups", self.launcher)

    def test_launcher_writes_and_rereads_durable_root_receipt(self) -> None:
        for token in (
            "pair-v5-source-bound-preservation-root-v1.json", "contract.make_root_receipt(",
            'temporary_path.open("xb")', "os.fsync(handle.fileno())", "os.O_DIRECTORY",
            "os.chmod(temporary_path,0o400)", "os.link(temporary_path,completion_path)",
            "os.unlink(temporary_path)", "contract.validate_root_receipt(",
            "if eligible != 8", "candidate_receipt_file_sha256_by_id",
            "group_receipt_file_sha256_by_id", "PAIR_V5_SOURCE_BOUND_STRONG_AUDIT_OK",
        ):
            self.assertIn(token, self.launcher)
        self.assertIn('"exploratory_dev_only": True', self.contract)
        self.assertIn('"action_score_dependency": False', self.contract)

    def test_runtime_has_no_privileged_visual_condition_cli(self) -> None:
        invocation = self.launcher.split('"${method_root}/score_pair_v5_source_bound_preservation_v1.py"', 1)[1].split("\n)", 1)[0]
        for forbidden in ("--target", "--proposal", "--donor", "--caption", "--prompt", "--mask", "--flow", "--pose", "--track", "--trajectory"):
            self.assertNotIn(forbidden, invocation)

    def test_launcher_preserves_virtualenv_invocation_path(self) -> None:
        self.assertNotIn('python_bin="$(realpath -e -- "${python_bin}")"', self.launcher)
        self.assertIn('python_bin_target="$(realpath -e -- "${python_bin}")"', self.launcher)
        self.assertIn('"${python_bin}" -B -', self.launcher)
        self.assertIn('"expected_runtime": spec["runtime_versions"]', self.launcher)
        self.assertIn('"observed_runtime": observed_runtime', self.launcher)

    def test_contract_eligibility_is_evidence_only_and_no_fake_noop(self) -> None:
        self.assertIn('"eligibility_policy": "provenance_decode_model_and_metric_evidence_valid_only"', self.contract)
        self.assertIn('"eligible_for_downstream_calibration": True', self.contract)
        self.assertIn("source_self_similarity_upper_bound", self.contract)
        self.assertNotIn('"probe_valid"', self.contract)
        self.assertNotIn('"no_op_source_self_similarity"', self.contract)

    def test_scorer_has_full_native_and_official_processor_verification(self) -> None:
        for token in (
            "expected_envelope_sha", "_PAIR_FIELDS", "_NATIVE_FIELDS",
            "native input/privileged-condition closure differs", "native checkpoint content seal differs",
            "native conditioning/source-id closure differs", "native sampling/guidance closure differs",
            "native latent/token geometry differs", "native model was not frozen adapter-free",
            "escaped candidate directory", "PAIR/native artifact binding differs",
            "AutoImageProcessor.from_pretrained", 'use_fast=False', '"BitImageProcessor"',
            "preprocessor_golden_output_sha256", "spatial_layout_viewpoint_similarity",
            "temporal_translation_agreement",
        ):
            self.assertIn(token, self.scorer)


if __name__ == "__main__":
    unittest.main()
