from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SBATCH = METHOD_ROOT / "scripts" / "auh_train_source_self_role_repaint_one_step.sbatch"
SUBMIT = METHOD_ROOT / "scripts" / "auh_submit_source_self_role_repaint_one_step.sh"


class SourceSelfLauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sbatch = SBATCH.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        for path in (SBATCH, SUBMIT):
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_single_node_world8_and_one_step_only(self) -> None:
        for fragment in (
            "#SBATCH --nodes=1",
            "#SBATCH --gres=gpu:mi210:8",
            "--nproc_per_node=8",
            "--mode engineering-canary",
            "--rho 0",
            "--adapter-block-scope early-mid-0-22",
            "ONE_STEP_ONLY_SCIENTIFIC_GATE_CLOSED",
        ):
            self.assertIn(fragment, self.sbatch)
        self.assertNotIn("afterok-c0", self.sbatch)
        self.assertNotIn("afterok:", self.submit)
        self.assertIn("long_training_job=NOT_SUBMITTED", self.submit)
        self.assertIn("scientific_gate=CLOSED", self.submit)

    def test_materialization_precedes_training_and_is_independent_rgb(self) -> None:
        materialize = self.sbatch.rindex("materialize_source_self_role_repaint.py")
        training = self.sbatch.rindex("train_source_self_role_repaint.py")
        self.assertLess(materialize, training)
        self.assertIn("--expected-spec-sha256", self.sbatch)
        self.assertIn("--checkpoint-content-manifest", self.sbatch)
        self.assertIn("references_from_video_posterior_slice", self.sbatch)
        self.assertIn("independent_vae_encode_calls_per_row", self.sbatch)
        self.assertIn("all_six_calls_share_one_pinned_vae_identity", self.sbatch)
        self.assertIn("paired_dataset_accessed", self.sbatch)
        self.assertIn("conditional_base_rho_hex", self.sbatch)
        self.assertIn("source_self_runtime.py", self.sbatch)
        self.assertNotIn("train_ramp_c0.py", self.sbatch)

    def test_submitter_submits_exactly_one_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "sbatch"
            fake.write_text("#!/usr/bin/env bash\nprintf '123456\\n'\n", encoding="utf-8")
            fake.chmod(0o755)
            launcher = Path(directory) / "launcher.sbatch"
            launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            launcher.chmod(0o755)
            output = Path(directory) / "fresh-output"
            result = subprocess.run(
                ["bash", str(SUBMIT)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    "PATH": f"{directory}:/usr/bin:/bin",
                    "BERNINI_SSR_LAUNCHER": str(launcher),
                    "BERNINI_SSR_RUN_ROOT": str(output),
                },
            )
        self.assertIn("one_step_job=123456", result.stdout)
        self.assertIn("long_training_job=NOT_SUBMITTED", result.stdout)


if __name__ == "__main__":
    unittest.main()
