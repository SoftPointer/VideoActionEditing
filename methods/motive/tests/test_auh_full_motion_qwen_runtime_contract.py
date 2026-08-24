from __future__ import annotations

from pathlib import Path
import unittest


SCRIPTS = (
    "auh_full_motion_qwen_smoke_existing_job.sh",
    "auh_full_motion_qwen_full_existing_job.sh",
    "auh_full_motion_qwen_distributed_existing_job.sh",
)


class FullMotionQwenRuntimeContractTests(unittest.TestCase):
    def test_all_launchers_freeze_the_current_visual_runtime(self) -> None:
        scripts_root = Path(__file__).resolve().parents[1] / "scripts"
        for filename in SCRIPTS:
            with self.subTest(script=filename):
                text = (scripts_root / filename).read_text(encoding="utf-8")
                for marker in (
                    "--nframes 16",
                    "--tile-width 512",
                    "--mosaic-columns 4",
                    "--max-pixels 2359296",
                    "--max-new-tokens 6144",
                    "--repair-attempts",
                ):
                    if marker == "--repair-attempts":
                        # The CLI freezes repairs at zero by default; launchers
                        # must not opt into any repair path.
                        self.assertNotIn(marker, text)
                    else:
                        self.assertIn(marker, text)
                self.assertNotIn("--nframes 24", text)
                self.assertNotIn("--max-pixels 1179648", text)
                self.assertNotIn("--max-new-tokens 4096", text)


if __name__ == "__main__":
    unittest.main()
