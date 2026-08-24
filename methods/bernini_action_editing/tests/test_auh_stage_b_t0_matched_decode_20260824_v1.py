from __future__ import annotations

from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_stage_b_t0_matched_decode_20260824_v1.sh"
)


class MatchedDecodeLauncherStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_requests_real_four_gpu_step_and_world4(self) -> None:
        self.assertIn("--gres=gpu:mi210:4", self.source)
        self.assertIn("--nproc_per_node=4", self.source)
        self.assertIn("--mem=0", self.source)
        self.assertIn("requires exactly four visible GPUs", self.source)

    def test_uses_native_forty_step_matched_cell(self) -> None:
        self.assertIn("--num-inference-steps 40", self.source)
        self.assertIn("--source-onset-policy hard1_every_step", self.source)
        self.assertIn("seed=2026081908", self.source)

    def test_binds_retry8_and_production_g2a_receipts(self) -> None:
        self.assertIn(
            "f94c6ae79e2e377f875c137b5be45a6040ddf862366736abb092896910167a83",
            self.source,
        )
        self.assertIn(
            "7ea0ab20709d942ca51a3062f2306407be8f9d0f4445926dca57af9b83fc3f09",
            self.source,
        )

    def test_binds_both_adapter_states(self) -> None:
        self.assertIn(
            "2e10edd015abdc0ce077a59ba1e6ce45f79df8f2c2805ad64971a4de055ddee8",
            self.source,
        )
        self.assertIn(
            "91f06e92837dadf8229ca1f2e5a26e512e8bbc26ddac8e1057bc832fa93ea44c",
            self.source,
        )

    def test_route_registry_contains_all_required_controls(self) -> None:
        self.assertIn(
            "route_off|zero|correct|temporal_shuffle|reverse|incomplete|wrong_action",
            self.source,
        )

    def test_output_is_create_once_and_failure_requires_new_revision(self) -> None:
        self.assertIn("decode coordinate is permanently consumed", self.source)
        self.assertIn("create-only decode log", self.source)
        self.assertIn("preserve output/logs and use a new revision", self.source)

    def test_video_is_decoded_and_geometry_validated(self) -> None:
        self.assertIn("ffprobe", self.source)
        self.assertIn("nb_read_frames == \"81\"", self.source)
        self.assertIn("r_frame_rate == \"25/1\"", self.source)

    def test_no_cpu_training_fallback(self) -> None:
        self.assertNotIn("--nproc_per_node=1", self.source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=-1", self.source)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=-1", self.source)


if __name__ == "__main__":
    unittest.main()
