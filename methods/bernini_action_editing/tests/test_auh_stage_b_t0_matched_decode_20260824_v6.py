from __future__ import annotations

from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_stage_b_t0_matched_decode_20260824_v6.sh"
)


class MatchedDecodeV6LauncherStaticTests(unittest.TestCase):
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

    def test_v6_has_fresh_immutable_coordinates(self) -> None:
        self.assertIn("source_stage_b_t0_decode_v6", self.source)
        self.assertIn("matched_decode_v6/0be6494dfac3", self.source)
        self.assertIn("logs/stage_b_t0_retry8/matched_decode_v6", self.source)
        self.assertNotIn("source_stage_b_t0_decode_v5", self.source)
        self.assertNotIn("matched_decode_v5/0be6494dfac3", self.source)

    def test_binds_fixed_parser_runner_and_test(self) -> None:
        self.assertIn(
            "c583ef9a74338aa62b3fc74d4fdbedcf6f8cdeb5b9612d626fe9c9addc6d44da",
            self.source,
        )
        self.assertIn(
            "908e12db0476987ed12b4e8e2328000a278bb65790f5ce710ba6e46318f37e4f",
            self.source,
        )
        self.assertIn("argparse_abbrev=false", self.source)
        self.assertIn("inference_audit=frozen", self.source)
        self.assertIn("receipt_base_identity=true", self.source)
        self.assertIn("strict_deterministic=true", self.source)
        self.assertIn("decoded_rgb_digest=true", self.source)

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
