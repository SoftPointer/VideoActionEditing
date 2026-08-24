from __future__ import annotations

from pathlib import Path
import stat
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_infer_stage_a_source_edge_confirmation_four_holder_v1.sh"
)


class FourHolderConfirmationLauncherTests(unittest.TestCase):
    def test_shell_is_valid_executable_and_resources_are_exact(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(LAUNCHER.stat().st_mode & stat.S_IXUSR)
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--gres=gpu:mi210:4", text)
        self.assertIn("--mem=60G", text)
        self.assertIn("--nproc_per_node=4", text)
        self.assertIn("TresPerNode=gres/gpu:mi210:8", text)
        self.assertNotIn("--gres=gpu:mi210:8 \\", text)

    def test_fixed_four_holder_to_sentinel_mapping(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for value in (
            "animal-dog-pick",
            "human-runner-jump",
            "hand-object-blueprint-roll",
            "emitter-fireworks-explode",
            "136007",
            "136008",
            "136009",
            "136010",
            "auh7-1b-gpu-215",
            "auh7-1b-gpu-261",
            "auh7-1b-gpu-262",
            "auh7-1b-gpu-228",
        ):
            self.assertIn(value, text)
        self.assertIn("for index in 0 1 2 3", text)
        self.assertIn('child_pids[$index]=$!', text)

    def test_manifest_is_the_only_cell_authority_and_parents_are_retained(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--confirmation-manifest", text)
        self.assertIn("--expected-confirmation-manifest-sha256", text)
        self.assertNotIn("--schedule-indices", text)
        self.assertNotIn("--block-bands", text)
        self.assertNotIn("scancel", text)
        self.assertNotIn("sbatch", text)
        self.assertIn("parents_retained=true", text)
        self.assertIn("stage_b_admission=false", text)
        self.assertIn(
            "build_stage_a_source_edge_confirmation_review_html_v1.py", text
        )
        self.assertGreaterEqual(
            text.count("tests/test_schedule_block_source_edge_ablation_v2.py"), 1
        )
        self.assertGreaterEqual(
            text.count("tests/test_infer_schedule_block_source_edge_localization_v2.py"),
            1,
        )
        self.assertIn("outputs=56", text)


if __name__ == "__main__":
    unittest.main()
