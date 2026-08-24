from __future__ import annotations

from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_stage_b_t0_matched_decode_20260824_v2.sh"
)


class MatchedDecodeV2LauncherStaticTests(unittest.TestCase):
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

    def test_v2_uses_fresh_source_output_and_log_coordinates(self) -> None:
        self.assertIn("source_stage_b_t0_decode_v2", self.source)
        self.assertIn("matched_decode_v2/0be6494dfac3", self.source)
        self.assertIn("logs/stage_b_t0_retry8/matched_decode_v2", self.source)
        self.assertNotIn("source_stage_b_t0_decode_v1", self.source)
        self.assertNotIn("matched_decode_v1/0be6494dfac3", self.source)

    def test_binds_retry8_g2a_and_both_states(self) -> None:
        for digest in (
            "f94c6ae79e2e377f875c137b5be45a6040ddf862366736abb092896910167a83",
            "7ea0ab20709d942ca51a3062f2306407be8f9d0f4445926dca57af9b83fc3f09",
            "2e10edd015abdc0ce077a59ba1e6ce45f79df8f2c2805ad64971a4de055ddee8",
            "91f06e92837dadf8229ca1f2e5a26e512e8bbc26ddac8e1057bc832fa93ea44c",
        ):
            self.assertIn(digest, self.source)

    def test_route_registry_contains_all_required_controls(self) -> None:
        self.assertIn(
            "route_off|zero|correct|temporal_shuffle|reverse|incomplete|wrong_action",
            self.source,
        )

    def test_pyav_replaces_missing_compute_ffprobe(self) -> None:
        self.assertIn("import av", self.source)
        self.assertIn("validate_video_artifact", self.source)
        self.assertNotIn("ffprobe", self.source)

    def test_output_is_create_once_and_failure_requires_new_revision(self) -> None:
        self.assertIn("decode coordinate is permanently consumed", self.source)
        self.assertIn("create-only decode log", self.source)
        self.assertIn("preserve output/logs and use a new revision", self.source)

    def test_no_cpu_fallback(self) -> None:
        self.assertNotIn("--nproc_per_node=1", self.source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=-1", self.source)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=-1", self.source)


if __name__ == "__main__":
    unittest.main()
