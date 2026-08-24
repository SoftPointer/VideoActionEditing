from __future__ import annotations

from pathlib import Path
import re
import unittest


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
COMMON = SCRIPT_ROOT / "auh_train_clean_source_visual_context_stage_b_holder_v1.sh"
MAIN = SCRIPT_ROOT / "auh_train_clean_source_visual_context_main_holder_v1.sh"
NOISED = SCRIPT_ROOT / "auh_train_clean_source_visual_context_noised_holder_v1.sh"
PREFLIGHT_MAIN = (
    SCRIPT_ROOT / "auh_preflight_clean_source_visual_context_main_holder_v1.sh"
)
PREFLIGHT_NOISED = (
    SCRIPT_ROOT / "auh_preflight_clean_source_visual_context_noised_holder_v1.sh"
)
MATERIALIZE = (
    SCRIPT_ROOT
    / "auh_materialize_clean_source_visual_context_source_only_v3_holder_v1.sh"
)


class CleanSourceVisualContextHolderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.common = COMMON.read_text(encoding="utf-8")
        cls.main = MAIN.read_text(encoding="utf-8")
        cls.noised = NOISED.read_text(encoding="utf-8")
        cls.preflight_main = PREFLIGHT_MAIN.read_text(encoding="utf-8")
        cls.preflight_noised = PREFLIGHT_NOISED.read_text(encoding="utf-8")
        cls.materialize = MATERIALIZE.read_text(encoding="utf-8")

    def test_two_wrappers_bind_independent_holders_and_arms(self) -> None:
        self.assertIn("CSVC_HOLDER_JOB=135980", self.main)
        self.assertIn("CSVC_HOLDER_NODE=auh7-1b-gpu-239", self.main)
        self.assertIn("CSVC_MEMORY_INPUT_KIND=clean_source", self.main)
        self.assertIn("CSVC_EXECUTION_SCOPE=formal-exact80", self.main)
        self.assertIn("CSVC_HOLDER_JOB=135981", self.noised)
        self.assertIn("CSVC_HOLDER_NODE=auh7-1b-gpu-234", self.noised)
        self.assertIn(
            "CSVC_MEMORY_INPUT_KIND=same_noise_forward_noised_source",
            self.noised,
        )
        self.assertIn("CSVC_EXECUTION_SCOPE=formal-exact80", self.noised)
        self.assertNotIn("135407", self.main + self.noised + self.common)
        self.assertNotIn("135411", self.main + self.noised + self.common)

    def test_preflight_wrappers_use_new_holders_without_stage_a(self) -> None:
        self.assertIn("CSVC_HOLDER_JOB=135980", self.preflight_main)
        self.assertIn("CSVC_HOLDER_NODE=auh7-1b-gpu-239", self.preflight_main)
        self.assertIn(
            "CSVC_EXECUTION_SCOPE=structural-parity-preflight",
            self.preflight_main,
        )
        self.assertIn("CSVC_HOLDER_JOB=135981", self.preflight_noised)
        self.assertIn("CSVC_HOLDER_NODE=auh7-1b-gpu-234", self.preflight_noised)
        self.assertIn(
            "CSVC_EXECUTION_SCOPE=structural-parity-preflight",
            self.preflight_noised,
        )
        self.assertIn(
            "unset CSVC_STAGE_A_ADMISSION CSVC_STAGE_A_ADMISSION_SHA256",
            self.preflight_main,
        )
        self.assertIn("optimizer_constructed=false", self.common)
        self.assertIn("backward_executed", self.common)
        self.assertIn("controller.PREFLIGHT_COMPLETE", self.common)

    def test_common_holder_requires_admission_before_world8_runner(self) -> None:
        self.assertIn("CSVC_STAGE_A_ADMISSION", self.common)
        self.assertIn("CSVC_STAGE_A_ADMISSION_SHA256", self.common)
        self.assertIn("CSVC_METHOD_ARCHIVE", self.common)
        self.assertIn("CSVC_METHOD_MANIFEST", self.common)
        self.assertIn("--expected-stage-a-admission-sha256", self.common)
        self.assertIn("CSVC_EXPECTED_INITIAL_PARAMETER_DIGEST", self.common)
        self.assertIn("--expected-initial-parameter-digest", self.common)
        self.assertIn("--method-source-archive", self.common)
        self.assertIn("--method-source-manifest", self.common)
        self.assertIn("--nproc_per_node=8", self.common)
        self.assertIn("--parallel-topology world8-dp2-sp4", self.common)
        self.assertIn("--optimizer-steps 80", self.common)
        self.assertIn('--execution-scope "${execution_scope}"', self.common)
        self.assertIn(
            "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12",
            self.common + self.materialize,
        )
        self.assertNotIn("/envs/vace/bin/python\n", self.common + self.materialize)
        self.assertIn('readonly rank_exec_source="${method_root}/scripts/', self.common)
        self.assertIn('cp -- "${rank_exec_source}" "${rank_exec}"', self.common)
        self.assertNotIn('chmod 0500 "${rank_exec_source}"', self.common)

    def test_all_five_checkpoints_are_required_and_parent_is_retained(self) -> None:
        self.assertIn(
            "for step in 00000000 00000020 00000040 00000060 00000080",
            self.common,
        )
        self.assertIn("decode_chain_ready=true", self.common)
        self.assertIn("decoded_checkpoint_inference_executed=false", self.common)
        self.assertIn("html_review_generated=false", self.common)
        self.assertIn("review_complete=false", self.common)
        self.assertIn("controller.TRAINING_COMPLETE", self.common)
        self.assertNotIn("controller.COMPLETE", self.common)
        self.assertIn("parent_not_released=true", self.common)
        lowered = self.common.lower()
        self.assertNotIn("scancel", lowered)
        self.assertNotIn("scontrol release", lowered)
        self.assertNotRegex(lowered, r"kill[^\n]*(135980|135981)")

    def test_only_source_only_noop_runner_is_launched(self) -> None:
        self.assertIn(
            "train_clean_source_visual_context_stage_b_v1.py", self.common
        )
        self.assertNotIn("train_preservation_residual_v1.py", self.common)
        self.assertNotIn("train_source_noised_carrier_strata_v1.py", self.common)

    def test_source_only_materializer_is_cpu_only_reserve_first_and_retained(self) -> None:
        self.assertIn("holder_job=135980", self.materialize)
        self.assertIn("holder_node=auh7-1b-gpu-239", self.materialize)
        self.assertIn("build-manifest", self.materialize)
        self.assertIn("audit-manifest", self.materialize)
        self.assertIn("heldout_strict_single_actor=8", self.materialize)
        self.assertIn("optimizer_constructed=false", self.materialize)
        self.assertNotIn("--gres=gpu", self.materialize)
        self.assertNotIn("scancel", self.materialize.lower())


if __name__ == "__main__":
    unittest.main()
