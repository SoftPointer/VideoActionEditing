from __future__ import annotations

from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_stage_b_t0_paired_decode_20260824_v7.sh"
)


class PairedDecodeV7LauncherStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_real_world4_gpu_without_cpu_fallback(self) -> None:
        for token in (
            "--gres=gpu:mi210:4",
            "--nproc_per_node=4",
            "--mem=0",
            "--num-inference-steps 40",
            "--source-onset-policy hard1_every_step",
            "no_cpu_fallback=true",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("--nproc_per_node=1", self.source)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=-1", self.source)

    def test_fresh_v7_source_output_and_logs(self) -> None:
        self.assertIn("source_stage_b_t0_decode_v7_paired", self.source)
        self.assertIn("matched_decode_v7_paired/0be6494dfac3", self.source)
        self.assertIn("logs/stage_b_t0_retry8/matched_decode_v7_paired", self.source)
        self.assertNotIn("matched_decode_v6/0be6494dfac3", self.source)

    def test_hash_locks_paired_and_single_dependencies(self) -> None:
        for digest in (
            "edd5d6e9d37116c81218525061ce89f31f1dcc87ef801c381e8e66c0e8ecd2c4",
            "52ba3e013925b011e5fd0165e099eeaed23525b724ca308871a41a3b75c62d0b",
            "c583ef9a74338aa62b3fc74d4fdbedcf6f8cdeb5b9612d626fe9c9addc6d44da",
            "908e12db0476987ed12b4e8e2328000a278bb65790f5ce710ba6e46318f37e4f",
        ):
            self.assertIn(digest, self.source)

    def test_binds_t0_g2a_and_both_adapter_states(self) -> None:
        for digest in (
            "f94c6ae79e2e377f875c137b5be45a6040ddf862366736abb092896910167a83",
            "7ea0ab20709d942ca51a3062f2306407be8f9d0f4445926dca57af9b83fc3f09",
            "2e10edd015abdc0ce077a59ba1e6ce45f79df8f2c2805ad64971a4de055ddee8",
            "91f06e92837dadf8229ca1f2e5a26e512e8bbc26ddac8e1057bc832fa93ea44c",
        ):
            self.assertIn(digest, self.source)

    def test_validates_receipt_and_every_video_with_pyav(self) -> None:
        self.assertIn("validate_paired_receipt", self.source)
        self.assertIn("validate_video_artifact", self.source)
        self.assertIn("for cell in receipt['cells']", self.source)
        self.assertNotIn("ffprobe", self.source)

    def test_gate_failure_is_preserved_and_blocks_comparison(self) -> None:
        self.assertIn('if [ "$rc" -eq 4 ]', self.source)
        self.assertIn("paired latent determinism gate failed", self.source)
        self.assertIn("coordinate is permanently consumed", self.source)
        self.assertIn("preserve v7 evidence", self.source)


if __name__ == "__main__":
    unittest.main()
