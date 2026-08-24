from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "auh_r9_representation_controller.sh"
)


class R9RepresentationOrchestrationTests(unittest.TestCase):
    def test_controller_is_hash_bound_instruction_only_and_stage_gated(
        self,
    ) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        required = (
            "MOTIVE_SOURCE_TREE_SHA256",
            "MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256",
            "MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256",
            "MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256",
            "MOTIVE_R7_VISUAL_CANDIDATES_SHA256",
            "MOTIVE_R7_SCREEN_INPUTS_RECEIPT_SHA256",
            "screen_inputs_v14",
            "screen_inputs_v14.receipt.txt",
            "action_source_snapshot.py",
            "r7_artifact_permissions",
            "assert_sealed_tree",
            "instruction_model_registry",
            "availability_before_representation_gate.json",
            "r9_automated_representation_search",
            "validate_published_search",
            "r7_candidate_temporal_screen",
            "--resume",
            'exec 9>"${run_root}/.controller.lock"',
            "flock -n 9",
        )
        for marker in required:
            self.assertIn(marker, text)
        self.assertNotIn("vace", text.lower())
        self.assertNotIn("mask", text.lower())
        self.assertNotIn("sbatch ", text)
        self.assertNotIn("--representation-gate-passed", text)
        self.assertNotIn(
            'visual_features_final="${parent_run}/expansion/visual_features_v1/final"',
            text,
        )

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
