from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "auh_submit_r9_two_seed_remote.sh",
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "retry_submit_auh_r9.sh",
)


class R9SubmissionScriptTests(unittest.TestCase):
    def test_scripts_have_valid_bash_syntax(self) -> None:
        for script in SCRIPTS:
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"{script}: {completed.stderr}",
            )

    def test_remote_submission_is_two_seed_and_training_closed(self) -> None:
        text = SCRIPTS[0].read_text(encoding="utf-8")
        self.assertIn("seeds=(260108835 260108836)", text)
        self.assertIn("--nodes=1", text)
        self.assertIn("--gres=gpu:mi210:1", text)
        self.assertIn('"maximum_concurrent_nodes": 2', text)
        self.assertIn('"gpus_per_job": 1', text)
        self.assertIn('"gpu_compute_expected": False', text)
        self.assertIn('"gpu_allocation_reason"', text)
        self.assertIn('"reason_for_single_gpu"', text)
        self.assertIn(
            "MOTIVE_R7_SCREEN_INPUTS_RECEIPT_SHA256",
            text,
        )
        self.assertIn("screen_inputs_v14", text)
        self.assertIn("assert_sealed_tree", text)
        self.assertIn(
            "test_r9_automated_representation_search.py",
            text,
        )
        self.assertIn('"renderer_training_submitted": False', text)
        self.assertNotIn("vace", text.lower())

    def test_retry_is_bounded_and_receipt_idempotent(self) -> None:
        text = SCRIPTS[1].read_text(encoding="utf-8")
        self.assertIn("MOTIVE_R9_CONNECT_ATTEMPTS", text)
        self.assertIn("ConnectTimeout=10", text)
        self.assertIn("source_snapshot.${archive_sha256}.tar.gz", text)
        self.assertIn("remote_archive_sha256", text)
        self.assertIn("--delay-directory-restore", text)
        self.assertIn("--expected-tree-sha256", text)
        self.assertIn("submit_once", text)
        self.assertIn("submission completed", text)


if __name__ == "__main__":
    unittest.main()
