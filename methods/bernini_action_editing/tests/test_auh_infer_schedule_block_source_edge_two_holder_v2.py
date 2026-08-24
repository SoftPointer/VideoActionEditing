from __future__ import annotations

from pathlib import Path
import re
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = METHOD_ROOT / "scripts/auh_infer_schedule_block_source_edge_two_holder_v2.sh"


class StageATwoHolderLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_default_is_minimal_s16_early_pilot(self) -> None:
        self.assertIn('schedules="${STAGE_A_SCHEDULE_INDICES:-16}"', self.source)
        self.assertIn('bands="${STAGE_A_BLOCK_BANDS:-early}"', self.source)
        self.assertIn('envs/vace/bin/python3.12}', self.source)
        self.assertIn('expected_count = 8 + len(schedules.split(",")) * len(bands.split(",")) * 6', self.source)
        self.assertIn('candidates=%s schedules=%s bands=%s', self.source)

    def test_two_independent_world4_children_run_in_parallel(self) -> None:
        self.assertIn('--gres=gpu:mi210:4', self.source)
        self.assertIn('--nproc_per_node=4', self.source)
        self.assertIn('launch_child dog', self.source)
        self.assertIn('launch_child human', self.source)
        self.assertIn('>"${run_root}/logs/dog.log" 2>&1 &', self.source)
        self.assertIn('>"${run_root}/logs/human.log" 2>&1 &', self.source)
        self.assertIn('pilot requires two independent retained holders', self.source)

    def test_parent_jobs_are_never_released(self) -> None:
        self.assertIsNone(re.search(r"(?m)^\s*scancel\b", self.source))
        self.assertIsNone(re.search(r"(?m)^\s*scontrol\s+release\b", self.source))
        self.assertIsNone(re.search(r"(?m)^\s*sbatch\b", self.source))
        self.assertIn("parents_not_released=true", self.source)
        self.assertIn("parents_retained=true", self.source)

    def test_fresh_outputs_and_fail_closed_receipts_are_required(self) -> None:
        self.assertIn("run root must be fresh canonical", self.source)
        self.assertIn("family output must be fresh", self.source)
        self.assertIn("decoded Stage-A child receipt gate failed", self.source)
        self.assertIn('sampling.get("source_on_native_parity_bit_exact") is True', self.source)
        self.assertIn('interpretation.get("training_performed") is False', self.source)
        self.assertIn('interpretation.get("optimizer_present") is False', self.source)
        self.assertIn('observed_sha == artifact.get("sha256")', self.source)
        self.assertIn('is not an eight-MI210 holder', self.source)
        self.assertIn('TresPerNode=gres/gpu:mi210:8', self.source)

    def test_runtime_and_tests_are_in_sealed_closure(self) -> None:
        for name in (
            "infer_schedule_block_source_edge_localization_v2.py",
            "schedule_block_source_edge_ablation_v2.py",
            "schedule_block_causal_policy_v1.py",
            "source_kv_replay.py",
            "pair_v5_t2v_calibration_first8_authoring_v1.json",
            "test_schedule_block_source_edge_ablation_v2",
            "test_infer_schedule_block_source_edge_localization_v2",
        ):
            self.assertIn(name, self.source)
        self.assertNotIn('-B -m unittest', self.source)
        self.assertIn('"${method_root}/tests/${test_file}"', self.source)


if __name__ == "__main__":
    unittest.main()
