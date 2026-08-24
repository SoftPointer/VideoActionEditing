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
    / "auh_r7_candidate_temporal_screen.sh"
)


class R7CandidateTemporalScreenOrchestrationTests(unittest.TestCase):
    def test_cpu_screen_is_frozen_hash_bound_and_single_writer(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        required = (
            "MOTIVE_SOURCE_TREE_SHA256",
            "MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256",
            "MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256",
            "MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256",
            "MOTIVE_R7_VISUAL_CANDIDATES_SHA256",
            'seed="${MOTIVE_R7_SCREEN_SEED:-260108835}"',
            "action_source_snapshot.py",
            "expected-tree-sha256",
            "imported screen module is not frozen",
            'exec 9>"${output_dir}.screen.lock"',
            "flock -n 9",
            "-m motive.r7_candidate_temporal_screen",
            "--expected-candidate-manifest-done-sha256",
            "--expected-track-cache-done-sha256",
            "--expected-visual-features-done-sha256",
            "--expected-visual-candidates-sha256",
        )
        for marker in required:
            self.assertIn(marker, text)
        self.assertNotIn("#SBATCH", text)
        self.assertNotIn("sbatch ", text)
        self.assertNotIn("MOTIVE_R7_DATA_SEED", text)
        self.assertNotIn("--no-rehash", text)
        self.assertNotIn("--no-verify", text)

    def test_script_has_valid_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
