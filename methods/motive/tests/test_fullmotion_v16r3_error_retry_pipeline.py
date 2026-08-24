from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE = REPO_ROOT / "tmp" / "launch_fullmotion_v16r3_error_retry_pipeline.sh"
ADAPTER = REPO_ROOT / "tmp" / "launch_fullmotion_v16r3_error_retry_persistent.sh"
GENERIC = REPO_ROOT / "tmp" / "launch_fullmotion_v16_full128_persistent.sh"
WAN = REPO_ROOT / "tmp" / "launch_fullmotion_v16_wan_stream.sh"
BATCH = REPO_ROOT / "methods" / "motive" / "motive" / "wan22_i2v_batch.py"


class FullMotionV16R3ErrorRetryPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = PIPELINE.read_text(encoding="utf-8")
        cls.adapter = ADAPTER.read_text(encoding="utf-8")
        cls.generic = GENERIC.read_text(encoding="utf-8")
        cls.wan = WAN.read_text(encoding="utf-8")
        cls.batch = BATCH.read_text(encoding="utf-8")

    def test_all_launchers_are_valid_bash(self) -> None:
        for path in (PIPELINE, ADAPTER, GENERIC):
            with self.subTest(path=path.name):
                completed = subprocess.run(
                    ["bash", "-n", str(path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_pipeline_waits_for_closed_r2_authority_before_exact_extraction(self) -> None:
        terminal_barrier = self.pipeline.index("terminal_count == 128")
        helper_call = self.pipeline.index('"${retry_helper}" "${helper_args[@]}"')
        handoff = self.pipeline.index(
            "validate_r2_watcher_terminal ||", helper_call
        )
        qwen_launch = self.pipeline.index('"${qwen_launcher}"', helper_call)
        self.assertLess(terminal_barrier, helper_call)
        self.assertLess(helper_call, handoff)
        self.assertLess(handoff, qwen_launch)
        self.assertIn("--expected-rows 128", self.pipeline)
        self.assertIn("--qwen-root \"${r2_qwen_root}\"", self.pipeline)
        self.assertIn("source_input_sha", self.pipeline)
        self.assertIn("expected_iids", self.pipeline)

    def test_retry_is_bound_to_uploaded_r3_snapshot_and_qwen3_vl_32b(self) -> None:
        expected_sha = (
            "ea6606c3131336d5dffbfc84b420311b83f8719d50bfef50c1fb62ebfecf8714"
        )
        self.assertIn("goku-full-motion128-source-v16r3-20260802T232100Z", self.adapter)
        self.assertIn(expected_sha, self.adapter)
        self.assertIn("immutable uploaded v16r3 snapshot", self.adapter)
        self.assertIn("Qwen3-VL-32B", self.generic)
        self.assertIn("MOTIVE_FULL_MOTION_QWEN_WORKERS", self.generic)
        self.assertIn("strided-dual4-first-four-nodes", self.generic)

    def test_only_error_manifest_and_new_qwen_root_feed_independent_wan(self) -> None:
        self.assertIn('MOTIVE_FULL_MOTION_QWEN_INPUT="${retry_input}"', self.pipeline)
        self.assertIn('MOTIVE_FULL_MOTION_QWEN_ROOT="${r3_qwen_root}"', self.pipeline)
        self.assertIn('MOTIVE_FULL_MOTION_WAN_ROOT="${r3_wan_root}"', self.pipeline)
        self.assertIn("qwen_qwen_errors_v16r3", self.pipeline)
        self.assertIn("wan_qwen_errors_v16r3", self.pipeline)
        self.assertIn(
            'r2_wan_root=${MOTIVE_FULL_MOTION_R2_WAN_ROOT:-${run}/wan_full128_v16r2}',
            self.pipeline,
        )
        self.assertIn("Qwen and Wan roots must be independent", self.wan)

    def test_resource_handoff_prevents_r2_expansion_race(self) -> None:
        self.assertIn("watcher_terminal.json", self.pipeline)
        self.assertIn("watch_contract.json", self.pipeline)
        self.assertIn("holder_numbered_steps_clear", self.pipeline)
        self.assertIn("batch/extern", self.pipeline)
        self.assertIn("unexpected holder step", self.pipeline)
        self.assertIn("assert_qwen_nodes_step_free", self.generic)
        self.assertIn("rocm-smi --showuse --showmemuse", self.generic)
        self.assertIn("double_idle_audit", self.wan)

    def test_create_only_disconnect_and_controller_recovery_are_explicit(self) -> None:
        for marker in (
            "MOTIVE_FULL_MOTION_R3_PIPELINE_RESUME",
            "pipeline_contract.json",
            "controller.lock",
            "flock -n",
            "trap '' HUP",
            "qwen_resumes",
            "MOTIVE_FULL_MOTION_R3_QWEN_RESUME=1",
            "MOTIVE_FULL_MOTION_WAN_RESUME=1",
            "pipeline_terminal.json",
        ):
            self.assertIn(marker, self.pipeline)
        self.assertIn("MOTIVE_FULL_MOTION_FULL128_RESUME", self.generic)
        self.assertIn("launch_contract.json", self.generic)
        self.assertIn("partial row cannot be resumed in-place", self.generic)
        for forbidden in ("rm -", "scancel", "scontrol cancel", "mv -f"):
            self.assertNotIn(forbidden, self.pipeline)

    def test_temporal_geometry_and_self_contained_sample_contract_are_retained(self) -> None:
        for marker in (
            '"frame_count": 81',
            '"fps": "25/1"',
            '"container_duration_seconds": 3.24',
            '"timeline_span_seconds": 3.2',
        ):
            self.assertIn(marker, self.pipeline)
        self.assertIn("--frame-num 81", self.wan)
        self.assertIn("--size '1280*720'", self.wan)
        self.assertIn("source_video", self.batch)
        self.assertIn("edit_instruction.txt", self.batch)
        self.assertIn("conditioning_frame0_float32.npy", self.batch)
        self.assertIn("preview.mp4", self.batch)


if __name__ == "__main__":
    unittest.main()
