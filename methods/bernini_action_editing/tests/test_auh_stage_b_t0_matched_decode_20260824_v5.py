from __future__ import annotations

from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_stage_b_t0_matched_decode_20260824_v5.sh"
)


class MatchedDecodeV5LauncherStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_world4_real_gpu_and_native_matched_cell(self) -> None:
        for token in (
            "--gres=gpu:mi210:4",
            "--nproc_per_node=4",
            "--mem=0",
            "--num-inference-steps 40",
            "--source-onset-policy hard1_every_step",
            "seed=2026081908",
        ):
            self.assertIn(token, self.source)

    def test_v5_has_fresh_immutable_coordinates(self) -> None:
        self.assertIn("source_stage_b_t0_decode_v5", self.source)
        self.assertIn("matched_decode_v5/0be6494dfac3", self.source)
        self.assertIn("logs/stage_b_t0_retry8/matched_decode_v5", self.source)
        self.assertNotIn("source_stage_b_t0_decode_v4", self.source)
        self.assertNotIn("matched_decode_v4/0be6494dfac3", self.source)

    def test_binds_fixed_parser_runner_and_test(self) -> None:
        self.assertIn(
            "667bd111855c8f45bc76ddfbea476aeef0ba081253c02ae3e0f5f5398a775405",
            self.source,
        )
        self.assertIn(
            "8093b9f9a337897486fdca88fbed403185614e651f57e3ba3c3486cf7460c0db",
            self.source,
        )
        self.assertIn("argparse_abbrev=false", self.source)
        self.assertIn("inference_audit=frozen", self.source)
        self.assertIn("receipt_base_identity=true", self.source)

    def test_binds_retry8_g2a_and_both_states(self) -> None:
        for digest in (
            "f94c6ae79e2e377f875c137b5be45a6040ddf862366736abb092896910167a83",
            "7ea0ab20709d942ca51a3062f2306407be8f9d0f4445926dca57af9b83fc3f09",
            "2e10edd015abdc0ce077a59ba1e6ce45f79df8f2c2805ad64971a4de055ddee8",
            "91f06e92837dadf8229ca1f2e5a26e512e8bbc26ddac8e1057bc832fa93ea44c",
        ):
            self.assertIn(digest, self.source)

    def test_route_registry_contains_all_controls(self) -> None:
        self.assertIn(
            "route_off|zero|correct|temporal_shuffle|reverse|incomplete|wrong_action",
            self.source,
        )

    def test_pyav_validation_and_no_missing_binary_dependency(self) -> None:
        self.assertIn("import av", self.source)
        self.assertIn("validate_video_artifact", self.source)
        self.assertNotIn("ffprobe", self.source)

    def test_create_once_and_no_cpu_fallback(self) -> None:
        self.assertIn("decode coordinate is permanently consumed", self.source)
        self.assertIn("preserve output/logs and use a new revision", self.source)
        self.assertNotIn("--nproc_per_node=1", self.source)
        self.assertNotIn("ROCR_VISIBLE_DEVICES=-1", self.source)


if __name__ == "__main__":
    unittest.main()
