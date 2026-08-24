from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "auh_r7_human_audit_sample.sbatch"
)


class R7HumanAuditOrchestrationTests(unittest.TestCase):
    def test_cpu_only_script_is_preregistered_and_hash_bound(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres=", text)
        self.assertIn("MOTIVE_R7_HUMAN_AUDIT_PREREGISTRATION", text)
        self.assertIn(
            "MOTIVE_R7_HUMAN_AUDIT_IMPLEMENTATION_BUNDLE_DIGEST",
            text,
        )
        self.assertIn("MOTIVE_R7_HUMAN_AUDIT_SOURCE_DIGEST", text)
        self.assertIn(
            "MOTIVE_R7_HUMAN_AUDIT_SOURCE_INPUT_SHA256",
            text,
        )
        self.assertIn("--expected-implementation-bundle-digest", text)
        self.assertIn("--expected-source-artifact-digest", text)
        self.assertIn("--expected-source-input-sha256", text)
        self.assertGreaterEqual(text.count("--resume"), 2)
        self.assertIn("formal_policy_locked", text)
        self.assertIn("formal_gate_input_eligible", text)
        self.assertIn("training_authorized", text)
        self.assertIn("O_EXCL", text)
        self.assertIn("receipt must be external to sample", text)
        self.assertGreaterEqual(
            text.count("action_source_snapshot.py"),
            2,
        )
        receipt_publish = text.index("flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL")
        self.assertLess(text.rfind("action_source_snapshot.py"), receipt_publish)
        self.assertLess(text.rfind("sha256sum"), receipt_publish)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
