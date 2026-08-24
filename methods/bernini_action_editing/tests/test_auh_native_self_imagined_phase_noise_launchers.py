from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
COMPUTE = METHOD_ROOT / "scripts/auh_infer_native_self_imagined_phase_noise_dual4.sbatch"
SUBMIT = METHOD_ROOT / "scripts/auh_submit_native_self_imagined_phase_noise_chain.sh"


class NativeSelfImaginedPhaseNoiseLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compute = COMPUTE.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        for path in (COMPUTE, SUBMIT):
            result = subprocess.run(
                ["bash", "-n", str(path)], check=False,
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")

    def test_compute_uses_all_eight_gpus_as_two_isolated_world4_groups(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.compute)
        self.assertIn('sp4_a_visible_gpus="0,1,2,3"', self.compute)
        self.assertIn('sp4_b_visible_gpus="4,5,6,7"', self.compute)
        self.assertEqual(self.compute.count("--nproc_per_node=4"), 1)
        self.assertIn(
            'launch_group arms-a "${sp4_a_visible_gpus}" "${sp4_a_master_port}"',
            self.compute,
        )
        self.assertIn(
            'launch_group arms-b "${sp4_b_visible_gpus}" "${sp4_b_master_port}"',
            self.compute,
        )
        self.assertIn("sp4_b_master_port=$((sp4_a_master_port + 1))", self.compute)
        self.assertEqual(self.compute.count('wait "${sp4_'), 2)
        self.assertIn("ROCR_VISIBLE_DEVICES", self.compute)
        for cache_name in (
            "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME",
            "PYTHONPYCACHEPREFIX", "TMPDIR", "TORCHELASTIC_ERROR_FILE",
        ):
            self.assertIn(cache_name, self.compute)

    def test_arm_split_is_complete_and_binds_one_shared_proposal_cell(self) -> None:
        self.assertIn("action-phi noop-phi reverse-phi", self.compute)
        self.assertIn("matched-gaussian action-phi-source-dc-rho02", self.compute)
        self.assertEqual(self.compute.count('--execution-group "${execution_group}"'), 1)
        self.assertIn("same_registered_proposal_latents_across_groups", self.compute)
        self.assertIn("same_baseline_gaussian_across_groups", self.compute)

    def test_submitter_stops_after_one_step_with_exact40_gate_closed(self) -> None:
        self.assertIn("PHASE_NOISE_NUM_INFERENCE_STEPS=1", self.submit)
        self.assertNotIn("PHASE_NOISE_NUM_INFERENCE_STEPS=40", self.submit)
        self.assertNotIn('--dependency="afterok:${canary_job}"', self.submit)
        self.assertIn('canary_output="${output_root}/canary-step1"', self.submit)
        self.assertIn("exact40\\tNOT_SUBMITTED\\tSCIENTIFIC_GATE_CLOSED", self.submit)
        self.assertIn("exact40=NOT_SUBMITTED scientific_gate=CLOSED", self.submit)
        self.assertEqual(self.submit.count("  --parsable \\\n"), 1)

    def test_compute_never_nests_slurm_or_world8(self) -> None:
        executable = [
            line.strip() for line in self.compute.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(any(re.match(r"^(srun|sbatch)(\s|$)", line) for line in executable))
        self.assertNotIn("--nproc_per_node=8", self.compute)
        self.assertIn(
            '[[ "${num_inference_steps}" == "1" || "${num_inference_steps}" == "40" ]]',
            self.compute,
        )

    def test_runtime_invocation_has_no_hidden_supervision_or_external_noise(self) -> None:
        invocation = self.compute.split('"${runtime_path}" \\', 1)[1].split("\n)", 1)[0]
        for required in (
            "--source-video", "--factor-manifest", "--factor-bank-receipt",
            "--bank-output-root", "--execution-group", "--condition-mode",
            "--arms", "--num-inference-steps", "--runtime-source-revision",
            "--runtime-source-archive-sha256", "--launcher-source-sha256",
        ):
            self.assertIn(required, invocation)
        for forbidden in (
            "--target", "--mask", "--flow", "--pose", "--track",
            "--trajectory", "--donor-video", "--proposal-video",
            "--initial-noise", "--optimizer", "--learning-rate", "--lora",
        ):
            self.assertNotIn(forbidden, invocation)
        self.assertIn("proposal_mp4=false", self.compute)
        self.assertIn("injection=true observer=false", self.compute)

    def test_archive_closure_and_transactional_postflight(self) -> None:
        for filename in (
            "infer_native_self_imagined_phase_noise_canary.py",
            "self_imagined_phase_noise.py", "source_spectral_bridge.py",
            "dmiq_t2v_factorial_bank.py",
            "infer_fitq_owner_prompt_cross_query_micro.py",
            "infer_native_multivideo_motion_donor_oracle.py",
            "infer_native_identity_generation_canary.py", "tri_branch_unipc.py",
            "infer_lora.py", "train_lora.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_source_value_residual_oracle.py",
            "auh_infer_native_self_imagined_phase_noise_dual4.sbatch",
            "auh_submit_native_self_imagined_phase_noise_chain.sh",
        ):
            self.assertIn(filename, self.compute)
        for token in (
            "archive member escaped repository",
            "Bernini method closure contains a link or device",
            'find "${method_root}" -type f -exec chmod a-w',
            "executed launcher differs from archive",
            "stage.receipt.json", 'prefix=".stage.receipt."',
            "os.replace(temporary, stage_path)", "os.fsync(descriptor)",
            "all_artifact_file_sha256_recomputed",
            "sealed receipt digest differs",
            "partial group output remains",
        ):
            self.assertIn(token, self.compute)

    def test_compute_supports_only_native_r2v5_or_rv2v4(self) -> None:
        self.assertIn(
            '[[ "${condition_mode}" == "r2v5" || "${condition_mode}" == "rv2v4" ]]',
            self.compute,
        )
        self.assertIn('--condition-mode "${condition_mode}"', self.compute)
        self.assertIn("PHASE_NOISE_CONDITION_MODE", self.submit)

    def test_embedded_python_is_valid(self) -> None:
        snippets = re.findall(r"<<'PY'\n(.*?)\nPY", self.compute, flags=re.S)
        self.assertEqual(len(snippets), 2)
        for snippet in snippets:
            ast.parse(snippet)


if __name__ == "__main__":
    unittest.main()
