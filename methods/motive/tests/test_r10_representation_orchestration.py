from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "auh_r10a_representation_controller.sh"
)


class R10RepresentationOrchestrationTests(unittest.TestCase):
    def test_controller_is_hash_bound_and_training_closed(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        required = (
            "MOTIVE_SOURCE_TREE_SHA256",
            "MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256",
            "MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256",
            "MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256",
            "MOTIVE_R7_VISUAL_CANDIDATES_SHA256",
            "MOTIVE_R7_SCREEN_INPUTS_RECEIPT_SHA256",
            "MOTIVE_R10A_ATTEMPT",
            "MOTIVE_R10A_JOB_NAME",
            "MOTIVE_R10A_ATTEMPT_RECEIPT",
            "MOTIVE_R10A_EXPERIMENT_ROOT",
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "screen_inputs_v14",
            "action_source_snapshot.py",
            "assert_sealed_tree",
            "instruction_model_registry",
            "r10_dynamic_dino_representation_search",
            "validate_published_search",
            '--source-tree-sha256 "${source_tree_sha256}"',
            "--source-tree-verified-by-controller",
            "--repeats 2",
            "--folds 3",
            'exec 9>"${run_root}/.controller.lock"',
            "flock -n 9",
            'decision["renderer_probe_authorized"] is False',
            'decision["editor_training_authorized"] is False',
            "motive-r10a-job-attempt-receipt-v1",
            "motive-r10a-artifact-producer-v1",
            "search_seed_${seed}.producer.json",
            "artifact_created=false",
            "producer_mode=validate",
            "os.link(temporary, producer_receipt",
            "validate_observed()",
            "require_attempt_receipt_path",
            "require_experiment_path",
        )
        for marker in required:
            self.assertIn(marker, text)
        self.assertNotIn("vace", text.lower())
        self.assertNotIn("mask", text.lower())
        self.assertNotIn("sbatch ", text)

    def test_script_has_valid_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_embedded_python_blocks_parse(self) -> None:
        blocks = re.findall(
            r"<<'PY'\n(.*?)\nPY",
            SCRIPT.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
        self.assertTrue(blocks)
        for block in blocks:
            ast.parse(block)


if __name__ == "__main__":
    unittest.main()
