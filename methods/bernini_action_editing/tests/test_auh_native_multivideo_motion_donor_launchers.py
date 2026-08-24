from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
COMPUTE = (
    METHOD_ROOT
    / "scripts/auh_infer_native_multivideo_motion_donor_oracle_dual4.sbatch"
)
SUBMIT = (
    METHOD_ROOT
    / "scripts/auh_submit_native_multivideo_motion_donor_oracle_chain.sh"
)


class NativeMotionDonorLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compute = COMPUTE.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        for path in (COMPUTE, SUBMIT):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")

    def test_compute_is_exact_dual_world4_on_all_eight_gpus(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.compute)
        self.assertIn('sp4_a_visible_gpus="0,1,2,3"', self.compute)
        self.assertIn('sp4_b_visible_gpus="4,5,6,7"', self.compute)
        self.assertEqual(self.compute.count("--nproc_per_node=4"), 1)
        self.assertIn(
            'launch_group sp4-a "${sp4_a_visible_gpus}" "${sp4_a_master_port}"',
            self.compute,
        )
        self.assertIn(
            'launch_group sp4-b "${sp4_b_visible_gpus}" "${sp4_b_master_port}"',
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

    def test_login_submitter_chains_canary_then_exact40_afterok(self) -> None:
        self.assertIn("MOTION_DONOR_NUM_INFERENCE_STEPS=1", self.submit)
        self.assertIn("MOTION_DONOR_NUM_INFERENCE_STEPS=40", self.submit)
        self.assertIn('--dependency="afterok:${canary_job}"', self.submit)
        self.assertIn('canary_output="${output_root}/canary-step1"', self.submit)
        self.assertIn('exact40_output="${output_root}/exact40"', self.submit)
        self.assertLess(
            self.submit.index("MOTION_DONOR_NUM_INFERENCE_STEPS=1"),
            self.submit.index("MOTION_DONOR_NUM_INFERENCE_STEPS=40"),
        )
        self.assertEqual(self.submit.count("  --parsable \\\n"), 2)
        self.assertIn("afterok:%s", self.submit)

    def test_compute_never_submits_or_nests_slurm(self) -> None:
        executable_lines = [
            line.strip() for line in self.compute.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(any(re.match(r"^(srun|sbatch)(\s|$)", line) for line in executable_lines))
        self.assertNotIn("--nproc_per_node=8", self.compute)
        self.assertIn('num_inference_steps="${MOTION_DONOR_NUM_INFERENCE_STEPS:', self.compute)
        self.assertIn('[[ "${num_inference_steps}" == "1" || "${num_inference_steps}" == "40" ]]', self.compute)

    def test_archive_and_registered_input_closure_is_explicit(self) -> None:
        for token in (
            "MOTION_DONOR_SOURCE_ARCHIVE",
            "MOTION_DONOR_SOURCE_ARCHIVE_SHA256",
            "MOTION_DONOR_SOURCE_REVISION",
            "MOTION_DONOR_FACTOR_MANIFEST_SHA256",
            "MOTION_DONOR_BANK_RECEIPT_SHA256",
            "MOTION_DONOR_BANK_OUTPUT_ROOT",
            "MOTION_DONOR_SOURCE_VIDEO",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST",
            "git get-tar-commit-id",
            "archive member escaped repository",
            "Bernini method closure contains a link or device",
            "methods/bernini_action_editing\n",
            'find "${method_root}" -type f -exec chmod a-w',
            "executed launcher differs from source archive",
            "cdf_dog_source_sha256",
        ):
            self.assertIn(token, self.compute)
        for filename in (
            "infer_native_multivideo_motion_donor_oracle.py",
            "infer_fitq_owner_prompt_cross_query_micro.py",
            "dmiq_t2v_factorial_bank.py",
            "infer_native_identity_generation_canary.py",
            "tri_branch_unipc.py",
            "infer_lora.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_source_value_residual_oracle.py",
            "auh_infer_native_multivideo_motion_donor_oracle_dual4.sbatch",
            "auh_submit_native_multivideo_motion_donor_oracle_chain.sh",
        ):
            self.assertIn(filename, self.compute)
        self.assertIn("compute launcher differs from source archive", self.submit)
        self.assertIn("submitter differs from source archive", self.submit)

    def test_runtime_invocation_is_frozen_registered_donor_only(self) -> None:
        invocation = self.compute.split('"${runtime_path}" \\', 1)[1].split("\n)", 1)[0]
        for option in (
            "--source-video",
            "--factor-manifest",
            "--factor-bank-receipt",
            "--bank-output-root",
            "--execution-group",
            "--num-inference-steps",
            "--runtime-source-revision",
            "--runtime-source-archive-sha256",
            "--launcher-source-sha256",
        ):
            self.assertIn(option, invocation)
        for option in (
            "--donor-video", "--donor-latent", "--target", "--mask",
            "--flow", "--pose", "--track", "--trajectory", "--reference",
            "--first-frame", "--initial-noise", "--optimizer",
            "--learning-rate", "--lora",
        ):
            self.assertNotIn(option, invocation)
        self.assertIn("donor_mp4=false", self.compute)
        self.assertIn("target=false", self.compute)

    def test_postflight_requires_both_sealed_receipts_and_parity(self) -> None:
        for token in (
            'for group in ("sp4-a", "sp4-b")',
            '"O0", "Z0", "D-action", "D-noop", "D-reverse"',
            '"D-duplicate-source", "D-action-order-swap", "A-source-v2v-apg"',
            'get("same_target_gaussian_all_arms") is not True',
            'get("byte_exact_fp32") is not True',
            'get("donor_mp4_consumed") is not False',
            'get("scientific_claim_authorized") is not False',
            '"engineering_oom_callpath_canary"',
            '"matched_exact40_qualitative_causal_pilot"',
            'set(value) != expected_keyset',
            'all_artifact_file_sha256_recomputed',
            'stage.receipt.json',
            'file_sha256(path) != expected_sha',
            'actual_files != expected_files',
        ):
            self.assertIn(token, self.compute)

    def test_group_and_stage_outputs_are_transactional(self) -> None:
        runtime = (
            METHOD_ROOT / "infer_native_multivideo_motion_donor_oracle.py"
        ).read_text(encoding="utf-8")
        combined = self.compute + runtime
        for token in (
            "BERNINI_OUTPUT_TRANSACTION_ID",
            ".partial-",
            "_commit_output_transaction",
            "os.replace(staging, final)",
            'prefix=".stage.receipt."',
            "os.replace(temporary, stage_path)",
            "os.fsync(descriptor)",
        ):
            self.assertIn(token, combined)

    def test_embedded_python_is_valid(self) -> None:
        snippets = re.findall(r"<<'PY'\n(.*?)\nPY", self.compute, flags=re.S)
        self.assertEqual(len(snippets), 2)
        for snippet in snippets:
            ast.parse(snippet)


if __name__ == "__main__":
    unittest.main()
