from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/auh_infer_frozen_base_existing_allocation_v1.sh"


class FrozenBaseExistingAllocationLauncherTests(unittest.TestCase):
    def test_launcher_is_source_only_base_control(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--base-only", text)
        self.assertIn("--source-video", text)
        self.assertIn("--instruction", text)
        self.assertNotIn("--adapter-checkpoint", text)
        for forbidden in (
            "--target-video",
            "--mask",
            "--pose",
            "--trajectory",
            "--reference-video",
            "--shared-i0",
        ):
            self.assertNotIn(forbidden, text)

    def test_launcher_requires_exactly_four_visible_devices_and_sealed_source(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("torch.cuda.device_count()", text)
        self.assertIn("torch.cuda.mem_get_info", text)
        self.assertIn('count != 4', text)
        self.assertIn('row["used_bytes"] > 1024**3', text)
        self.assertIn("srun did not expose four unoccupied GPUs", text)
        self.assertIn("git get-tar-commit-id", text)
        self.assertIn("source_archive_sha256", text)
        self.assertIn("test_infer_lora_contract.py", text)
        self.assertIn("receipt digest differs", text)
        self.assertIn("scientific_claim_authorized", text)


if __name__ == "__main__":
    unittest.main()
