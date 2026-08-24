#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_infer_native_role_reversed_owner_identity_rebinding_dual4.sbatch"
)


class RoleReversedOwnerLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all8_is_two_isolated_sp4_groups(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.text)
        self.assertIn("#SBATCH --qos=gtqos", self.text)
        self.assertIn("launch_group dog 0,1,2,3", self.text)
        self.assertIn("launch_group human 4,5,6,7", self.text)
        self.assertGreaterEqual(self.text.count("--nproc_per_node=4"), 1)
        self.assertIn("two_concurrent_world4_sp4_groups_on_one_8gpu_node", self.text)

    def test_exact81_exact40_only(self) -> None:
        self.assertIn("--num-inference-steps 40", self.text)
        self.assertNotIn("--num-inference-steps 1", self.text)
        self.assertIn('"exact81": True', self.text)
        self.assertIn('"exact40": True', self.text)
        self.assertIn('"training_performed": False', self.text)
        self.assertIn('"pseudo_target_distillation_performed": False', self.text)

    def test_pending_owner_bytes_are_bound_without_fake_authority(self) -> None:
        for name in (
            "ROLE_REVERSED_REGISTRY_SHA256",
            "ROLE_REVERSED_OWNER_MASTER_SHA256",
            "ROLE_REVERSED_SOURCE_ARCHIVE_SHA256",
            "ROLE_REVERSED_AUDIT_SIDECAR_SHA256",
            "ROLE_REVERSED_AUDIT_PUBLIC_KEY_SHA256",
        ):
            self.assertIn(name, self.text)
        self.assertIn("test_materialize_self_imagined_owner_core2_v1.py", self.text)
        self.assertIn("--audit-sidecar", self.text)
        self.assertIn("--audit-evidence", self.text)
        self.assertIn("--audit-public-key", self.text)
        self.assertIn(
            "--experimental-owner-primal-ack frozen_role_reversed_owner_primal_diagnostic_only",
            self.text,
        )

    def test_launcher_binds_repo_and_output_transactions(self) -> None:
        self.assertIn("git get-tar-commit-id", self.text)
        self.assertIn("runtime_source_archive_sha256", self.text)
        self.assertIn("BERNINI_OUTPUT_TRANSACTION_ID", self.text)
        self.assertIn("all8-receipt.json", self.text)
        self.assertIn("os.fsync", self.text)


if __name__ == "__main__":
    unittest.main()
