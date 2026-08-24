from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_infer_native_v_axis_exact81_single_holder_v1.sh"
)


class NativeVAxisSingleHolderLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_retained_holder_and_no_release(self) -> None:
        self.assertIn("readonly holder_job=135412", self.source)
        self.assertIn("readonly holder_node=auh7-1b-gpu-293", self.source)
        self.assertIn("parent_not_released=true", self.source)
        self.assertIn("parent_released\": False", self.source)
        self.assertNotIn("scancel", self.source.lower())

    def test_one_numbered_all8_child_serially_reuses_one_sp4_group(self) -> None:
        self.assertIn('--jobid="${holder_job}"', self.source)
        self.assertIn("--gres=gpu:mi210:8", self.source)
        self.assertIn("launch_group dog 0,1,2,3", self.source)
        self.assertIn("launch_group human 0,1,2,3", self.source)
        dog_launch = self.source.index("launch_group dog 0,1,2,3")
        human_launch = self.source.index("launch_group human 0,1,2,3")
        self.assertLess(dog_launch, human_launch)
        self.assertNotIn('logs/dog.log" 2>&1 &', self.source)
        self.assertNotIn('logs/human.log" 2>&1 &', self.source)
        self.assertIn("--nproc_per_node=4", self.source)

    def test_three_arms_two_seeds_exact40_exact81_and_no_selection(self) -> None:
        self.assertIn("arms=V-on,V-off,wrong-V seeds=2 exact40 exact81", self.source)
        self.assertIn('arms = ["V-on", "V-off", "wrong-V"]', self.source)
        self.assertIn('len(candidates) == 6', self.source)
        self.assertIn('len(outputs) == 6', self.source)
        self.assertIn('"generated_video_count": 12', self.source)
        self.assertIn('interpretation.get("feature_scorer_consumed") is False', self.source)
        self.assertIn('interpretation.get("ranking_performed") is False', self.source)
        self.assertIn('interpretation.get("best_arm_selected") is False', self.source)

    def test_shared_rank_load_lock_and_resource_receipt_gate(self) -> None:
        self.assertIn('load_lock="${task_scratch}/renderer-load.lock"', self.source)
        self.assertIn('export NATIVE_V_AXIS_LOAD_LOCK="${load_lock}"', self.source)
        self.assertIn("rank_serialized_checkpoint_deserialize", self.source)
        self.assertIn("vae_instantiated_on_rank_zero_only", self.source)
        self.assertIn("text_encoder_retired_before_vae_and_sampling", self.source)
        self.assertIn(
            "sampling_model_destroyed_without_cpu_offload_before_rank_zero_decode",
            self.source,
        )

    def test_closure_includes_runtime_core_spec_launcher_and_tests_run_first(self) -> None:
        for name in (
            "infer_native_v_axis_exact81_probe_v1.py",
            "native_v_axis_guidance_v1.py",
            "native_v_axis_exact81_core2_v1.json",
            "auh_infer_native_v_axis_exact81_single_holder_v1.sh",
        ):
            self.assertIn(name, self.source)
        test_loop = self.source.index("for test_file in")
        launch = self.source.index("launch_group dog 0,1,2,3")
        self.assertLess(test_loop, launch)


if __name__ == "__main__":
    unittest.main()
