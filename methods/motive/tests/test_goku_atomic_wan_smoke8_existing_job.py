from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "tmp" / "launch_goku_atomic_wan_smoke8_existing_job.sh"


def embedded_helper(text: str, marker: str) -> str:
    opening = f"<<'{marker}'"
    line_start = text.index(opening)
    start = text.index("\n", line_start) + 1
    end = text.index(f"\n{marker}\n", start)
    return text[start:end] + "\n"


class GokuAtomicWanSmoke8ExistingJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax_and_embedded_python(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        markers = re.findall(r"<<'([A-Z][A-Z0-9_]*)'", self.text)
        self.assertEqual(len(markers), 12)
        for marker in markers:
            with self.subTest(marker=marker):
                compile(embedded_helper(self.text, marker), marker, "exec")

    def test_exact_one_node_eight_mi210_binding(self) -> None:
        for marker in (
            '"NumNodes=1" "gres/gpu:mi210=8"',
            "${#nodes[@]} == 1",
            "--gpus-per-task=8 --gpu-bind=none",
            "--nproc_per_node=8",
            "--expected-world-size 8",
            "--expected-gpu-name-substring MI210",
        ):
            self.assertIn(marker, self.text)

    def test_serial_manifest_order_and_retry_ceiling(self) -> None:
        self.assertIn("len(lines) != 8", self.text)
        self.assertIn("strict_manifest_order_one_full_8gpu_sample_at_a_time", self.text)
        self.assertIn("while IFS=$'\\t' read -r ordinal iid", self.text)
        self.assertIn("(( max_attempts == 2 ))", self.text)
        self.assertIn('elif len(claims) >= maximum: print("exhausted")', self.text)
        # Generation is synchronous: the srun command is not backgrounded.
        runner = self.text.split("run_wan_attempt() {", 1)[1].split(
            "publish_sample() {", 1
        )[0]
        self.assertNotRegex(runner, r"srun[^\n]*&")

    def test_atomic_gate_and_private_planner_prompt_are_hash_bound(self) -> None:
        helper = embedded_helper(self.text, "PY_INPUT_CONTRACT")
        for marker in (
            'result.get("status") != "ok"',
            'sha(result_raw) != gate["result_sha256"]',
            'result["source_passed_sha256"] != fragment_sha',
            'passed["edit_instruction"]',
            '"wan_generation_prompt": passed["edit_instruction"]',
            '"primary_training_label": "atomic_action_instruction"',
            '"wan_generation_prompt_is_training_label": False',
        ):
            self.assertIn(marker, helper)

    def test_temporal_geometry_is_fixed_without_retiming(self) -> None:
        for marker in (
            "--frame-num 81",
            '"frame_count": 81, "frame_rate": "25/1", "retiming_allowed": False',
            'policy.get("source", {}).get("frame_count") != 81',
            'policy.get("target", {}).get("frame_count") != 81',
            'policy.get("source", {}).get("frame_rate") != "25/1"',
            'policy.get("target", {}).get("frame_rate") != "25/1"',
        ):
            self.assertIn(marker, self.text)

    def test_successful_sample_publishes_label_roles_and_media(self) -> None:
        sample = embedded_helper(self.text, "PY_SAMPLE")
        for marker in (
            '"source_video.mp4"',
            '"preview.mp4"',
            '"atomic_action_instruction.txt"',
            '"camera_instruction.txt"',
            '"preservation_instruction.txt"',
            '"full_edit_instruction.txt"',
            '"wan_generation_prompt.txt"',
            '"edit_instruction.txt"',
            '"primary_training_label_field": "atomic_action_instruction"',
            '"wan_generation_prompt_source": "planner_passed.edit_instruction"',
            '"private_generation_prompt_not_training_label"',
        ):
            self.assertIn(marker, sample)

    def test_controller_is_self_detached_and_children_drop_lock(self) -> None:
        for marker in (
            "nohup setsid -f env",
            "MOTIVE_ATOMIC_WAN_SMOKE_CONTROLLER_MODE=1",
            "trap '' HUP",
            'flock -n "${controller_lock_fd}"',
            "exec {controller_lock_fd}>&-",
            "controller_bootstrap.log",
        ):
            self.assertIn(marker, self.text)


if __name__ == "__main__":
    unittest.main()
