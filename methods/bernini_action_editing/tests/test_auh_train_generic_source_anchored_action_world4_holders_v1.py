from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = METHOD_ROOT / "scripts"
COMMON = SCRIPT_ROOT / "auh_train_generic_source_anchored_action_world4_holder_v1.sh"
RANK = SCRIPT_ROOT / "auh_generic_source_anchored_action_rank_exec_v1.sh"
MAIN = SCRIPT_ROOT / "auh_train_generic_source_anchored_action_main_136309_v1.sh"
CONTROL = (
    SCRIPT_ROOT
    / "auh_train_generic_source_anchored_action_action_only_136141_v1.sh"
)
STAGE_R = SCRIPT_ROOT / "auh_train_generic_source_anchored_action_stage_r_136309_v1.sh"
SMOKE_R = SCRIPT_ROOT / "auh_smoke_generic_source_anchored_action_r_136309_v1.sh"


class GenericActionWorld4HolderLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.common = COMMON.read_text(encoding="utf-8")
        cls.rank = RANK.read_text(encoding="utf-8")
        cls.main = MAIN.read_text(encoding="utf-8")
        cls.control = CONTROL.read_text(encoding="utf-8")
        cls.stage_r = STAGE_R.read_text(encoding="utf-8")
        cls.smoke_r = SMOKE_R.read_text(encoding="utf-8")

    def test_bound_wrappers_select_only_experiment_not_action_rows(self) -> None:
        self.assertIn("GSA_HOLDER_JOB=136309", self.main)
        self.assertIn("GSA_HOLDER_NODE=auh7-1b-gpu-280", self.main)
        self.assertIn("GSA_EXECUTION_PROFILE=resume-po40", self.main)
        self.assertIn("GSA_HOLDER_JOB=136141", self.control)
        self.assertIn("GSA_HOLDER_NODE=auh7-1b-gpu-299", self.control)
        self.assertIn("GSA_EXECUTION_PROFILE=action-only40", self.control)
        self.assertIn(
            "GSA_CARRIER_POLICY=not_installed_or_exact_zero_frozen",
            self.control,
        )
        self.assertNotIn('case "${local_rank}"', self.rank)
        self.assertNotIn('case "${global_rank}"', self.rank)
        self.assertIn("GSA_ARM_ID=joint_stage_r64", self.stage_r)
        self.assertIn("GSA_EXECUTION_PROFILE=stage-r64", self.stage_r)
        self.assertIn("GSA_ARM_ID=smoke_r", self.smoke_r)
        self.assertIn("GSA_EXECUTION_PROFILE=smoke-r", self.smoke_r)

    def test_world4_child_uses_one_same_xgmi_island(self) -> None:
        for fragment in (
            "--nproc_per_node=4",
            "--parallel-topology world4-dp1-sp4",
            "--cpus-per-task=32",
            "--mem=60G",
            "--gpus-per-task=4",
            "--gpu-bind=none",
            "--gres-flags=enforce-binding",
            "0,1,2,3|4,5,6,7",
            '"${xgmi_count}" == 24',
            '"${numa_zero_count}" == 4',
            '"${numa_one_count}" == 4',
            '"${visible_count}" == 4',
        ):
            self.assertIn(fragment, self.common)
        self.assertIn('"${world_size}" == 4', self.rank)

    def test_holder_admission_is_idle_twice_and_parent_safe(self) -> None:
        self.assertEqual(self.common.count("assert_idle\n"), 2)
        self.assertIn("sleep 2", self.common)
        self.assertIn("holder already has a numbered child", self.common)
        self.assertIn("holder acquired a numbered child before run-root creation", self.common)
        self.assertIn("child is not the holder's only numbered step", self.common)
        self.assertIn("--immediate=5", self.common)
        self.assertIn('"${holder_job}.${SLURM_STEP_ID}"', self.common)
        self.assertLess(
            self.common.index("holder acquired a numbered child before run-root creation"),
            self.common.index('mkdir -m 0700 "${run_root}"'),
        )
        self.assertLess(
            self.common.index('if [[ "${launcher_role}" == parent ]]'),
            self.common.index('if [[ "${launcher_role}" == child ]]'),
        )
        self.assertIn("parent run root must be fresh and canonical", self.common)
        self.assertIn(
            "child run root must be the canonical parent-created directory",
            self.common,
        )
        self.assertIn("parent_not_released=true", self.common)
        self.assertIn("kill -TERM \"${child_pid}\"", self.common)
        lowered = self.common.lower()
        self.assertNotIn("scancel", lowered)
        self.assertNotIn("scontrol release", lowered)
        self.assertNotIn("scontrol requeue", lowered)
        self.assertNotRegex(lowered, r"kill[^\n]*(136309|136141)")

    def test_memory_and_data_pins_are_fail_closed(self) -> None:
        self.assertIn("readonly gpu_memory_limit_gib=52", self.common)
        self.assertIn("readonly host_memory_limit_gib=60", self.common)
        self.assertIn("--gpu-memory-limit-gib", self.common)
        self.assertIn("--host-memory-limit-gib", self.common)
        self.assertIn(
            "128064fd335c4e48c567217c6e7bae43555a904875625c9d1e21178e6f7fcc3d",
            self.common,
        )
        self.assertIn("action manifest is absent or not canonical", self.common)
        self.assertIn("action manifest has no frozen SHA-256", self.common)
        self.assertIn("action-manifest validator rejected authority", self.common)

    def test_r_only_profiles_reject_action_inputs_and_cannot_claim_action_complete(self) -> None:
        self.assertIn("R-only profile must not consume action manifests", self.common)
        self.assertIn("complete_action_result=false", self.common)
        self.assertIn("controller.RETAIN_STAGE_COMPLETE", self.common)
        self.assertNotIn("controller.COMPLETE", self.common)

    def test_all_shell_sources_parse(self) -> None:
        for path in (COMMON, RANK, MAIN, CONTROL, STAGE_R, SMOKE_R):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
