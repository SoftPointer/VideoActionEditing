from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT / "scripts" / "auh_materialize_appearance_identity_orbit_v2.sbatch"
)


class AUHAppearanceIdentityOrbitMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL)

    def test_shell_and_embedded_python_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.python_blocks), 1)
        ast.parse(self.python_blocks[0])

    def test_one_gpu_pinned_vae_materialization(self) -> None:
        for fragment in (
            "#SBATCH --gres=gpu:mi210:1",
            "--expected-spec-sha256",
            "--checkpoint-content-manifest",
            "--device cuda:0",
            "expected_checkpoint_manifest_sha256=",
            "archive differs from commit subtree",
        ):
            self.assertIn(fragment, self.source)

    def test_miopen_and_compiler_caches_are_job_private_and_writable(self) -> None:
        for fragment in (
            '"${task_scratch}/miopen-user"',
            '"${task_scratch}/miopen-custom"',
            'export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"',
            'export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"',
            'export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"',
            'export TRITON_CACHE_DIR="${task_scratch}/triton"',
            'export TMPDIR="${task_scratch}/tmp"',
        ):
            self.assertIn(fragment, self.source)

    def test_output_is_create_only_receipt_bound_and_two_row(self) -> None:
        for fragment in (
            "output is create-only",
            "$'dataset.parquet\\nreceipt.json'",
            "chmod a-w",
            "dataset receipt digest differs",
            'receipt.get("scientific_use_authorized") is not True',
            'receipt.get("direct_action_edit_claim_authorized") is not False',
            "parquet.metadata.num_rows != 2",
            '"bernini-appearance-counterfactual-identity-orbit-dataset-receipt-v3"',
            'encoding.get("reference_count") != 4',
            'encoding.get("reference_rgb_indices") != [0, 27, 53, 80]',
            'encoding.get("independent_rgb_reference_encode_calls_per_row") != 12',
            'encoding.get("independent_vae_encode_calls_per_row") != 15',
            '!= "one_video_plus_four_rgb_refs"',
            "required_posterior_columns",
            "actual_posterior_columns != required_posterior_columns",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn(
            "bernini-appearance-counterfactual-identity-orbit-dataset-receipt-v2",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
