from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO_ROOT / "methods" / "motive" / "scripts"


class R7VisualGraphOrchestrationScriptsTest(unittest.TestCase):
    def _script(self, name: str) -> str:
        path = SCRIPT_ROOT / name
        subprocess.run(["bash", "-n", str(path)], check=True)
        return path.read_text(encoding="utf-8")

    def test_graph_input_is_cpu_only_and_strictly_revalidates(self) -> None:
        text = self._script("auh_r7_visual_graph_input.sbatch")
        self.assertNotIn("#SBATCH --gres=", text)
        self.assertIn("MOTIVE_R7_CANDIDATE_FEATURE_DONE_SHA256", text)
        self.assertIn("MOTIVE_R7_ANCHOR_FEATURE_DONE_SHA256", text)
        self.assertIn("MOTIVE_R7_GRAPH_INPUT_EXPECTED_ASSETS", text)
        self.assertIn("-m motive.r7_visual_graph_input", text)
        self.assertIn("--resume", text)
        self.assertIn("training_authorized", text)
        self.assertIn("refusing existing output", text)
        self.assertGreaterEqual(text.count("action_source_snapshot.py"), 2)

    def test_indexed_graph_is_cpu_only_and_fail_closed(self) -> None:
        text = self._script("auh_r7_indexed_visual_graph.sbatch")
        self.assertNotIn("#SBATCH --gres=", text)
        self.assertIn("MOTIVE_R7_GRAPH_INPUT_DIGEST", text)
        self.assertIn("MOTIVE_R7_DINO_EDGE_DONE_SHA256", text)
        self.assertIn("-m motive.r7_indexed_visual_graph_io", text)
        self.assertIn("--resume", text)
        self.assertIn("thresholds_human_calibrated", text)
        self.assertIn("formal_split", text)
        self.assertIn("training_authorized", text)
        self.assertIn("refusing existing output", text)
        self.assertGreaterEqual(text.count("action_source_snapshot.py"), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
