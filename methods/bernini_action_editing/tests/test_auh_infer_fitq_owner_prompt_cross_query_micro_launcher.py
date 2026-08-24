from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_infer_fitq_owner_prompt_cross_query_micro_dual4.sbatch"
)


class OwnerPromptCrossQueryLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_dual_world4_topology(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn('sp4_a_visible_gpus="0,1,2,3"', self.source)
        self.assertIn('sp4_b_visible_gpus="4,5,6,7"', self.source)
        self.assertEqual(self.source.count("--nproc_per_node=4"), 1)
        self.assertIn(
            'launch_group sp4-a "${sp4_a_visible_gpus}" "${sp4_a_master_port}"',
            self.source,
        )
        self.assertIn(
            'launch_group sp4-b "${sp4_b_visible_gpus}" "${sp4_b_master_port}"',
            self.source,
        )
        self.assertIn("sp4_b_master_port=$((sp4_a_master_port + 1))", self.source)
        self.assertEqual(self.source.count("wait \"${sp4_"), 2)

    def test_archive_and_input_provenance_are_fail_closed(self) -> None:
        for token in (
            "FITQ_CROSS_SOURCE_ARCHIVE",
            "FITQ_CROSS_SOURCE_ARCHIVE_SHA256",
            "FITQ_CROSS_SOURCE_REVISION",
            "FITQ_CROSS_FACTOR_MANIFEST_SHA256",
            "FITQ_CROSS_BANK_RECEIPT_SHA256",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST",
            "git get-tar-commit-id",
            "archive member escaped repository",
            "archive contains a link or device",
            "find \"${method_root}\" -type f -exec chmod a-w",
            "executed launcher differs from source archive",
        ):
            self.assertIn(token, self.source)
        required_files = (
            "infer_fitq_owner_prompt_cross_query_micro.py",
            "dmiq_t2v_factorial_bank.py",
            "infer_native_identity_generation_canary.py",
            "inference_sigma_strata.py",
            "internal_temporal_quotient_observer.py",
            "tri_branch_unipc.py",
        )
        for filename in required_files:
            self.assertIn(filename, self.source)

    def test_runtime_invocation_has_no_training_or_privileged_condition(self) -> None:
        invocation = self.source.split('"${runtime_path}" \\', 1)[1].split(
            "\n)", 1
        )[0]
        expected = (
            "--factor-manifest",
            "--factor-bank-receipt",
            "--bank-output-root",
            "--execution-group",
            "--runtime-source-revision",
            "--runtime-source-archive-sha256",
            "--launcher-source-sha256",
        )
        for option in expected:
            self.assertIn(option, invocation)
        forbidden = (
            "--target", "--mask", "--flow", "--pose", "--track",
            "--trajectory", "--reference", "--first-frame",
            "--initial-noise", "--optimizer", "--learning-rate", "--lora",
        )
        for option in forbidden:
            self.assertNotIn(option, invocation)
        self.assertNotIn("sbatch ", self.source)
        self.assertNotIn("srun ", self.source)
        self.assertNotIn("--nproc_per_node=8", self.source)

    def test_embedded_python_is_valid(self) -> None:
        snippets = re.findall(r"<<'PY'\n(.*?)\nPY", self.source, flags=re.S)
        self.assertEqual(len(snippets), 1)
        ast.parse(snippets[0])


if __name__ == "__main__":
    unittest.main()
