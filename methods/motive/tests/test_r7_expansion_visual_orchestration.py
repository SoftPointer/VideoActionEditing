from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "auh_r7_expansion_visual_features.sbatch"
)


class R7ExpansionVisualOrchestrationTests(unittest.TestCase):
    def test_script_is_eight_gpu_dino_only_and_fail_closed(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        required = (
            "#SBATCH --gres=gpu:mi210:8",
            "torch.distributed.run",
            "--nproc_per_node=8",
            "expected 8 visible GPUs",
            "refusing partial shard set",
            "r7_expansion_visual_features extract",
            "r7_expansion_visual_features finalize",
            'done.get("split_ready") is not True',
            "PYTHONNOUSERSITE=1",
            "PYTHONDONTWRITEBYTECODE=1",
            "MOTIVE_R7_VISUAL_CANDIDATES_SHA256",
            "MOTIVE_DINOV2_REVISION",
        )
        for marker in required:
            self.assertIn(marker, text)
        self.assertEqual(text.count("--expected-tree-sha256"), 2)
        self.assertNotIn("--cotracker-checkpoint", text)
        self.assertNotIn("r7_temporal_teacher", text)
        self.assertNotIn("srun --nodes", text)

    def test_script_has_valid_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
