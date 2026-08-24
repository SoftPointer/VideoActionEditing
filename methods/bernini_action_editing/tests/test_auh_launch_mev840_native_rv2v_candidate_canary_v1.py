from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "auh_launch_mev840_native_rv2v_candidate_canary_v1.sh"


class NativeRV2VCandidateCanaryLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_generation_surface_is_native_source_and_prompt_only(self) -> None:
        self.assertIn("--arms rv2v", self.text)
        self.assertIn("--num-inference-steps 40", self.text)
        self.assertIn("--source-video", self.text)
        self.assertIn("--action-prompt", self.text)
        self.assertNotRegex(self.text, r"--target(?:-|\s)")
        self.assertNotIn("real_target", self.text)
        self.assertNotIn("target_action_oracle", self.text)
        self.assertNotIn("activity25", self.text)
        self.assertNotIn("anchor_qk", self.text)
        self.assertNotIn("transport-strength", self.text)

    def test_exact_two_seed_node_assignment(self) -> None:
        self.assertIn(
            "2027) readonly expected_job=143808 expected_node=auh7-1b-gpu-292",
            self.text,
        )
        self.assertIn(
            "2028) readonly expected_job=147873 expected_node=auh7-1b-gpu-284",
            self.text,
        )
        self.assertEqual(
            re.findall(r"--seed \"\$\{seed\}\"", self.text),
            ['--seed "${seed}"'],
        )

    def test_runtime_archive_and_authorities_are_content_pinned(self) -> None:
        for digest in (
            "46ae7529d640a197006ab8d7d17c23ac81925dabd7fa1caf4b0bb261197e8115",
            "e104031526236f16e94a4753c31ad8048b1a65345b1913212c35e421fcad48ae",
            "4e78a935b2485e3f8c2c94aa5524a82ed25aa0b93aaf58dd81476dc5c9b48044",
            "ac22e19ffd109a2d6b85c32c64463b0be8373792",
            "bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42",
            "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
            "a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646",
            "effdf094385a4f2486391efc008150b7436a8137c1d5766864a678ed6e0c749f",
        ):
            self.assertIn(digest, self.text)
        self.assertIn("value.get(\"file_count\") != 19", self.text)
        self.assertIn("if seen != expected", self.text)
        self.assertIn("NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1", self.text)
        self.assertIn("NATIVE_V_AXIS_LOAD_LOCK", self.text)
        self.assertIn("generator_action_json_read\":False", self.text)
        self.assertIn("generator_target_video_read\":False", self.text)


if __name__ == "__main__":
    unittest.main()
