#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_run_starc_core4_critic_pilot_v1.sbatch"


class AUHSTARCCriticPilotLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_source_archive_master_and_running_launcher_are_hash_bound(self) -> None:
        self.assertIn("git get-tar-commit-id", self.source)
        self.assertIn("source_archive_sha256", self.source)
        self.assertIn("master_manifest_sha256", self.source)
        self.assertIn('${BASH_SOURCE[0]}', self.source)
        self.assertIn('sha256sum "${launcher}"', self.source)

    def test_exact_two_stage_runner_command_is_used(self) -> None:
        self.assertIn('"${runner}" fit-evaluate', self.source)
        self.assertIn("--expected-master-sha256", self.source)
        self.assertIn("--device cuda:0", self.source)
        self.assertIn("fit_steps=200", self.source)
        self.assertIn("confirmation_once=true", self.source)

    def test_editor_or_bernini_training_is_not_invoked(self) -> None:
        self.assertIn("editor_loaded=false", self.source)
        self.assertNotIn("train_lora.py", self.source)
        self.assertNotIn("train_joint_lora.py", self.source)
        self.assertNotIn("torch.distributed.run", self.source)
        self.assertNotIn("deepspeed", self.source)

    def test_output_closure_is_exactly_five_files(self) -> None:
        for name in (
            "starc-core4-critic-config-v1.json",
            "starc-core4-critic-fit-trace-v1.json",
            "starc-core4-critic-final-step-0200.safetensors",
            "starc-core4-critic-checkpoint-receipt-v1.json",
            "starc-core4-heldout-provisional-gate-receipt-v1.json",
        ):
            self.assertIn(name, self.source)
        self.assertIn("output_file_count", self.source)
        self.assertIn("output_file_count == 5", self.source)


if __name__ == "__main__":
    unittest.main()
