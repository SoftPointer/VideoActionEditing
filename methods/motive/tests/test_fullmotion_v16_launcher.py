from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "tmp" / "launch_fullmotion_v16_smoke.sh"
RETRY_LAUNCHER = (
    REPO_ROOT / "tmp" / "launch_fullmotion_v16_retry_subset.sh"
)


class FullMotionV16LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_fixed_run_snapshot_model_and_input_bindings(self) -> None:
        self.assertIn("fullmotion128_v16_20260802T221943Z", self.text)
        self.assertIn(
            "goku-full-motion128-source-v16-20260802T221943Z", self.text
        )
        self.assertIn(
            "/VLM/MEV-Annotation/checkpoints/Qwen3-VL-32B-Instruct",
            self.text,
        )
        self.assertIn(
            "prepare_smoke8_stratified/candidates.jsonl",
            self.text,
        )
        self.assertIn(
            "f9bd77773d220101a82c7be459a4b071b519d22890cac7a5ffd897eb719c7346",
            self.text,
        )
        self.assertIn("job_id=${MOTIVE_FULL_MOTION_JOB_ID:-118150}", self.text)
        self.assertIn("tree_sha=${MOTIVE_FULL_MOTION_TREE_SHA:-__V16_TREE_SHA__}", self.text)
        self.assertIn(
            '[[ "${tree_sha}" != "${unbound_tree_sha_sentinel}" ]]',
            self.text,
        )
        self.assertIn('digest_file "${smoke_input}"', self.text)
        self.assertIn("action_source_snapshot.py", self.text)
        self.assertIn("--expected-tree-sha256", self.text)

    def test_eight_rows_are_packed_two_per_qwen_node(self) -> None:
        self.assertIn("for node_index in 0 1 2 3; do", self.text)
        self.assertIn("--gpus-per-task=4", self.text)
        self.assertIn("--ntasks=2 --ntasks-per-node=2", self.text)
        self.assertIn("row_index=$((node_index * 2 + SLURM_LOCALID))", self.text)
        self.assertIn("visible_devices=0,1,2,3", self.text)
        self.assertIn("visible_devices=4,5,6,7", self.text)
        self.assertIn('ROCR_VISIBLE_DEVICES="${visible_devices}"', self.text)
        self.assertIn("unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES", self.text)
        self.assertNotIn('HIP_VISIBLE_DEVICES="${visible_devices}"', self.text)
        self.assertNotIn('CUDA_VISIBLE_DEVICES="${visible_devices}"', self.text)
        self.assertIn("qwen_nodes=(", self.text)
        self.assertIn("wan_free_nodes=(", self.text)
        self.assertIn(
            "-m motive.goku_full_motion_qwen_v16", self.text
        )
        self.assertIn('--row-index "${row_index}" --num-rows 8', self.text)
        self.assertIn("--max-new-tokens 4096", self.text)
        self.assertNotIn("--max-new-tokens 6144", self.text)
        self.assertIn("--allow-errors", self.text)
        self.assertIn("check_idle_node", self.text)
        self.assertIn("rocm-smi --showuse --showmemuse", self.text)
        self.assertIn("rocm-smi --showpids", self.text)
        self.assertIn("assert_holder_only_steps", self.text)
        self.assertIn("unexpected active Slurm step", self.text)
        self.assertIn("shared-step dual4 Qwen admission probe failed", self.text)
        self.assertIn("torch.cuda.device_count()", self.text)
        self.assertIn("torch.cuda.synchronize", self.text)
        self.assertIn('--job-name="v16-qwen-idle-${node}"', self.text)
        self.assertIn("--job-name=v16-qwen-probe-shared", self.text)
        self.assertIn('--job-name="v16-qwen-node-${node_index}"', self.text)

    def test_outputs_are_create_only_and_row_terminal(self) -> None:
        self.assertIn(
            'for path in "${output_root}" "${smoke_log}" "${smoke_pid_file}"',
            self.text,
        )
        self.assertIn("create-only v16 smoke target", self.text)
        self.assertIn(
            "terminal/${iid}.receipt.json", self.text
        )
        self.assertIn("rows/${iid}/result.json", self.text)
        self.assertIn("stream=%s/passed", self.text)

    def test_no_smoke_wide_pass_gate_or_holder_claim(self) -> None:
        forbidden = (
            "goku_full_motion_smoke_gate",
            "--minimum-hard-passes",
            "minimum_hard_passes",
            "pipeline_claim",
            "authorizes_full_run",
        )
        for item in forbidden:
            self.assertNotIn(item, self.text)
        self.assertIn("Do not compute or enforce a pass-count threshold", self.text)
        self.assertIn("passed/<iid>.jsonl", self.text)

    def test_unbound_snapshot_stops_before_remote_preflight(self) -> None:
        completed = subprocess.run(
            ["/bin/bash", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
            env={
                "HOME": "/tmp",
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "LANG": "C",
                # Do not inherit a caller-provided binding.
                "MOTIVE_FULL_MOTION_TREE_SHA": "",
            },
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "bind MOTIVE_FULL_MOTION_TREE_SHA", completed.stderr
        )
        self.assertNotIn("scontrol", completed.stderr)

class FullMotionV16RetrySubsetLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = RETRY_LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(RETRY_LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_requires_explicit_create_only_subset_bindings(self) -> None:
        for variable in (
            "MOTIVE_FULL_MOTION_TREE_SHA",
            "MOTIVE_FULL_MOTION_RETRY_INPUT",
            "MOTIVE_FULL_MOTION_RETRY_INPUT_SHA",
            "MOTIVE_FULL_MOTION_RETRY_OUTPUT_ROOT",
        ):
            self.assertIn(variable, self.text)
        self.assertIn("create-only retry target already exists", self.text)
        self.assertIn("retry input bytes differ", self.text)
        self.assertIn("--expected-tree-sha256", self.text)
        self.assertIn("retry output must not equal or live below qwen_smoke8_v16", self.text)

    def test_accepts_only_one_to_six_unique_rows(self) -> None:
        self.assertIn("retry subset must contain 1..6 rows", self.text)
        self.assertIn("if not 1 <= len(iids) <= 6", self.text)
        self.assertIn("retry subset IIDs are not unique", self.text)
        self.assertIn("needed_nodes=$(( (row_count + 1) / 2 ))", self.text)

    def test_uses_first_idle_nodes_and_single_shared_step_per_node(self) -> None:
        self.assertIn("load_busy_nodes", self.text)
        self.assertIn("candidate_nodes", self.text)
        self.assertIn('selected_nodes=("${idle_nodes[@]:0:${needed_nodes}}")', self.text)
        self.assertIn("check_idle_node", self.text)
        self.assertIn("rocm-smi --showpids", self.text)
        self.assertIn('--job-name="v16-retry-node-${node_slot}"', self.text)
        self.assertIn('--ntasks="${task_count}" --ntasks-per-node="${task_count}"', self.text)
        self.assertIn("--gpus-per-task=4 --gpu-bind=none", self.text)
        self.assertIn("--kill-on-bad-exit=0", self.text)
        self.assertNotIn("--kill-on-bad-exit=1", self.text)
        self.assertIn("row_index=$((first_row + SLURM_LOCALID))", self.text)
        self.assertIn("visible_devices=0,1,2,3", self.text)
        self.assertIn("visible_devices=4,5,6,7", self.text)
        self.assertIn("unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES", self.text)

    def test_independent_terminal_and_stream_outputs_without_gate(self) -> None:
        self.assertIn("terminal/${iid}.receipt.json", self.text)
        self.assertIn("rows/${iid}/result.json", self.text)
        self.assertIn("stream=%s/passed", self.text)
        self.assertIn("--allow-errors", self.text)
        self.assertIn("--max-new-tokens 4096", self.text)
        for forbidden in (
            "goku_full_motion_smoke_gate",
            "--minimum-hard-passes",
            "authorizes_full_run",
        ):
            self.assertNotIn(forbidden, self.text)

    def test_missing_bindings_stop_before_slurm(self) -> None:
        completed = subprocess.run(
            ["/bin/bash", str(RETRY_LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
            env={
                "HOME": "/tmp",
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "LANG": "C",
            },
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("set the explicit retry binding", completed.stderr)
        self.assertNotIn("scontrol", completed.stderr)


if __name__ == "__main__":
    unittest.main()
