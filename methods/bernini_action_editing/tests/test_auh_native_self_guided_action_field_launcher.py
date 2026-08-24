#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/auh_infer_native_self_guided_action_field_dual4.sbatch"


class ActionFieldLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_requests_one_full_eight_gpu_node(self) -> None:
        self.assertIn("#SBATCH --nodes=1", self.source)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn("launch_group dog 0,1,2,3", self.source)
        self.assertIn("launch_group human 4,5,6,7", self.source)
        self.assertEqual(self.source.count("--nproc_per_node=4"), 1)

    def test_exact81_frozen_receipt_contract(self) -> None:
        self.assertIn('"exact81": True', self.source)
        self.assertIn('"exact40": True', self.source)
        self.assertIn('"guidance_mode": "v2v_apg"', self.source)
        self.assertIn('"training_performed": False', self.source)
        self.assertIn('"parameter_update": False', self.source)
        self.assertIn('"generated_owner_media_consumed": False', self.source)
        self.assertIn('"source_rich_initial_noise": False', self.source)

    def test_closure_includes_runtime_core_registry_tests_and_launcher(self) -> None:
        for name in (
            "infer_native_self_guided_action_field_canary.py",
            "self_guided_action_field_v1.py",
            "test_self_guided_action_field_v1.py",
            "test_infer_native_self_guided_action_field_canary.py",
            "test_auh_native_self_guided_action_field_launcher.py",
            "auh_infer_native_self_guided_action_field_dual4.sbatch",
        ):
            self.assertIn(name, self.source)
        self.assertIn('closure=("${repo_closure[@]}" "${registry}")', self.source)

    def test_two_children_must_complete_before_master_receipt(self) -> None:
        self.assertIn('wait "${dog_pid}"', self.source)
        self.assertIn('wait "${human_pid}"', self.source)
        self.assertIn('[[ "${dog_status}" == 0 ]]', self.source)
        self.assertIn('[[ "${human_status}" == 0 ]]', self.source)
        self.assertIn('for cell in ("dog", "human"):', self.source)

    def test_postflight_reopens_media_latents_and_gaussians(self) -> None:
        self.assertIn("decoded_video_contract", self.source)
        self.assertIn("decoded_frame_count", self.source)
        self.assertIn("safetensor_raw_sha", self.source)
        self.assertIn("normalized_clean_latent", self.source)
        self.assertIn("official_initial_gaussian", self.source)
        self.assertIn("live_output_postflight", self.source)
        self.assertIn("live_clean_latent_postflight", self.source)
        self.assertIn("live_initial_gaussian_postflight", self.source)
        self.assertIn("action_field_trace_digest", self.source)
        self.assertIn("freeze_certificate_digest", self.source)
        self.assertIn('field.get("native_rv2v_forwards") != 80', self.source)
        self.assertIn('field.get("frozen_t2v_teacher_forwards") != 120', self.source)
        self.assertIn('step.get("native_official_apg_exact_parity") is not True', self.source)


if __name__ == "__main__":
    unittest.main()
