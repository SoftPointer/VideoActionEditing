#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_infer_saic_fixed_r2v_visual_i0_pair_all8_v2.sbatch"
)


class FixedR2VAll8LauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_requests_all8_without_hold_or_dependency(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn("#SBATCH --nodes=1", self.source)
        self.assertIn("#SBATCH --ntasks=1", self.source)
        self.assertNotIn("#SBATCH --hold", self.source)
        self.assertNotIn("#SBATCH --dependency", self.source)
        self.assertNotIn("scontrol hold", self.source)
        self.assertNotIn("--hold ", self.source)

    def test_only_unique_minimal_pair_is_executed(self) -> None:
        self.assertIn("readonly arms=(R00 R11)", self.source)
        self.assertIn('for arm in "${arms[@]}"; do run_wave "$arm"; done', self.source)
        self.assertNotRegex(self.source, r"run_wave\s+[\"']?R10")
        self.assertNotRegex(self.source, r"run_wave\s+[\"']?R01")
        self.assertIn('"R10": "high/low sigma boundary is not uniquely specified"', self.source)
        self.assertIn('"R01": "high/low sigma boundary is not uniquely specified"', self.source)

    def test_reuses_exact_job132387_source_coordinates_read_only(self) -> None:
        self.assertIn("ssft-t0-i0-ec4bfb6-j6", self.source)
        self.assertIn(
            "e3560e77546d3936f7e7231d5aceb78b8f29ea379b98748c26e4b37a5f277d7a",
            self.source,
        )
        self.assertIn(
            "9cab2ac419833a3d8451b24c578c3ca3b466341f7bc17ea40e4067c8ebf6b7e9",
            self.source,
        )
        self.assertIn("reused_from_slurm_job_id", self.source)
        self.assertNotIn("cp -- \"$dog_source_clean\"", self.source)
        self.assertNotIn("cp -- \"$human_source_clean\"", self.source)

    def test_frame0_is_once_per_source_and_shared_by_both_arms(self) -> None:
        self.assertEqual(self.source.count("run_frame0 dog "), 1)
        self.assertEqual(self.source.count("run_frame0 human "), 1)
        self.assertIn("materialize_saic_frame0_latent_v1.py", self.source)
        self.assertIn(
            "995ac093c20fd07ef7018bdf942a1d7acb971c357effa53d4fddf5ac9e4e1f14",
            self.source,
        )
        self.assertIn(
            "a6ce585d73bfa7a5e607356c9c06c25ec4f5e1d6f8c3b5e2c83d685b6910c6e1",
            self.source,
        )
        self.assertIn("--reference-frame0-latent \"$frame0\"", self.source)
        self.assertIn("matched candidate-zero noise differs", self.source)

    def test_fresh_output_and_zero_authority_are_fail_closed(self) -> None:
        self.assertIn('[[ ! -e "$output_root" && ! -L "$output_root" ]]', self.source)
        self.assertIn("os.O_EXCL", self.source)
        self.assertIn('"selection": False', self.source)
        self.assertIn('"training": False', self.source)
        self.assertIn('"optimizer": False', self.source)
        self.assertIn('"production": False', self.source)
        self.assertIn("find \"$output_root\" -type d -exec chmod 0500", self.source)


if __name__ == "__main__":
    unittest.main()
