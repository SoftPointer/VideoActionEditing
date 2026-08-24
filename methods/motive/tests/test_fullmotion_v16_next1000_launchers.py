from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE = REPO_ROOT / "tmp" / "launch_fullmotion_v16_next1000_pipeline.sh"
QWEN = REPO_ROOT / "tmp" / "launch_fullmotion_v16_full128_persistent.sh"
WAN = REPO_ROOT / "tmp" / "launch_fullmotion_v16_wan_stream.sh"


def embedded_helper(text: str, marker: str) -> str:
    opening = f"<<'{marker}'\n"
    start = text.index(opening) + len(opening)
    end = text.index(f"\n{marker}\n", start)
    return text[start:end]


class FullMotionV16Next1000LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = PIPELINE.read_text(encoding="utf-8")
        cls.qwen = QWEN.read_text(encoding="utf-8")
        cls.wan = WAN.read_text(encoding="utf-8")

    def test_launchers_are_valid_bash(self) -> None:
        for path in (PIPELINE, QWEN, WAN):
            with self.subTest(path=path.name):
                result = subprocess.run(
                    ["bash", "-n", str(path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_pipeline_embedded_python_helpers_compile(self) -> None:
        markers = (
            "PY_INPUT_CLOSURE",
            "PY_CONTRACT",
            "PY_LAUNCH_INDEX",
            "PY_QWEN_TERMINALS",
            "PY_TERMINAL",
        )
        for marker in markers:
            with self.subTest(marker=marker):
                compile(
                    embedded_helper(self.pipeline, marker),
                    f"next1000-{marker}",
                    "exec",
                )

    def test_default_1000_rows_close_as_eight_equal_strided_shards(self) -> None:
        self.assertIn("MOTIVE_FULL_MOTION_NEXT_EXPECTED_ROWS:-1000", self.pipeline)
        self.assertIn("MOTIVE_FULL_MOTION_NEXT_QWEN_WORKERS:-8", self.pipeline)
        self.assertIn(
            "shards = [len(range(worker, expected, workers)) for worker in range(workers)]",
            self.pipeline,
        )
        shards = [len(range(worker, 1000, 8)) for worker in range(8)]
        self.assertEqual(shards, [125] * 8)
        uneven = [len(range(worker, 1003, 8)) for worker in range(8)]
        self.assertEqual(sum(uneven), 1003)
        self.assertEqual(uneven, [126, 126, 126, 125, 125, 125, 125, 125])

    def test_contract_records_actual_shard_sizes(self) -> None:
        helper = embedded_helper(self.pipeline, "PY_CONTRACT")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            args = [
                sys.executable,
                "-c",
                helper,
                str(path),
                "0",
                "/run",
                "/snapshot",
                "a" * 64,
                "/input.jsonl",
                "b" * 64,
                "1000",
                "8",
                ",".join(["125"] * 8),
                "/run/qwen",
                "/run/wan",
                "/run/control",
                "118150",
                "/run/qwen-launcher.sh",
                "/run/wan-launcher.sh",
            ]
            result = subprocess.run(
                args, check=False, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["expected_rows"], 1000)
            self.assertEqual(value["qwen"]["worker_count"], 8)
            self.assertEqual(value["qwen"]["strided_shard_sizes"], [125] * 8)
            self.assertTrue(value["qwen"]["errors_are_terminal"])
            self.assertEqual(value["wan"]["initial_concurrency"], 4)
            self.assertEqual(value["wan"]["expanded_concurrency"], 8)

    def test_generic_qwen_has_no_128_row_ceiling(self) -> None:
        self.assertNotIn("expected_rows <= 128", self.qwen)
        self.assertNotIn("rows<=128", self.qwen)
        self.assertIn("worker_count <= 8", self.qwen)
        self.assertIn("worker_count <= expected_rows", self.qwen)
        self.assertIn('--num-rows "${expected_rows}"', self.qwen)
        self.assertIn(
            "row_index<expected_rows; row_index+=worker_count", self.qwen
        )

    def test_qwen_uses_two_four_gpu_workers_on_each_first_four_node(self) -> None:
        self.assertIn('"${allocated_nodes[0]}" "${allocated_nodes[1]}"', self.qwen)
        self.assertIn('"${allocated_nodes[2]}" "${allocated_nodes[3]}"', self.qwen)
        self.assertIn("needed_qwen_nodes=$(( (worker_count + 1) / 2 ))", self.qwen)
        self.assertIn('--ntasks="${tasks}" --ntasks-per-node="${tasks}"', self.qwen)
        self.assertIn("--gpus-per-task=4 --gpu-bind=none", self.qwen)
        self.assertIn(
            "worker_index=$((node_index * 2 + SLURM_LOCALID))", self.qwen
        )
        self.assertIn("visible_devices=0,1,2,3", self.qwen)
        self.assertIn("visible_devices=4,5,6,7", self.qwen)

    def test_every_worker_cryptographically_closes_its_actual_shard(self) -> None:
        helper = embedded_helper(self.qwen, "PY_WORKER_TERMINAL")
        compile(helper, "next1000-worker-terminal", "exec")
        self.assertIn("assigned = list(range(worker, expected, workers))", helper)
        self.assertIn("_validate_terminal_receipt", helper)
        self.assertIn("object_sha256(row)", helper)
        self.assertIn('counts = {"ok": 0, "error": 0}', helper)
        self.assertIn(
            "ok_count + error_count == terminal_count", self.qwen
        )

    def test_qwen_errors_are_terminal_and_do_not_abort_other_rows(self) -> None:
        self.assertIn("--allow-errors", self.qwen)
        self.assertIn("--kill-on-bad-exit=0", self.qwen)
        self.assertIn("counts['error']", self.qwen)
        self.assertIn('"errors_are_terminal": True', self.pipeline)
        self.assertIn("qwen_error_iids", self.wan)
        self.assertIn("pass dispatched independently", self.wan)

    def test_wan_starts_streaming_before_qwen_batch_closure(self) -> None:
        controller = self.pipeline.split("controller_main() {", 1)[1]
        qwen_launch = controller.index("launch_qwen 0")
        wan_launch = controller.index("launch_wan 0")
        wait_terminal = controller.index("watcher_terminal.json", wan_launch)
        self.assertLess(qwen_launch, wan_launch)
        self.assertLess(wan_launch, wait_terminal)
        self.assertNotIn("all Qwen rows pass", controller[:wan_launch])

    def test_wan_expands_only_after_terminal_step_and_double_idle_gate(self) -> None:
        self.assertIn(
            "MOTIVE_FULL_MOTION_WAN_EXPAND_AFTER_QWEN_TERMINAL=1",
            self.pipeline,
        )
        self.assertIn('dispatch_nodes=("${wan_nodes[@]}")', self.wan)
        self.assertIn("if (( all_qwen_terminal == 1", self.wan)
        self.assertIn("qwen_step_nodes_are_clear", self.wan)
        self.assertIn("expansion_idle_audit", self.wan)
        self.assertIn("for audit in 1 2", self.wan)
        self.assertIn(
            'dispatch_nodes=("${wan_nodes[@]}" "${qwen_nodes[@]}")', self.wan
        )
        self.assertIn(
            "all_qwen_terminal_steps_exited_double_gpu_idle", self.wan
        )

    def test_create_only_resume_and_hup_safety_are_single_owner(self) -> None:
        for marker in (
            "MOTIVE_FULL_MOTION_NEXT_RESUME",
            "create-only next target exists",
            "pipeline_contract.json",
            'exec {pipeline_lock_fd}>"${control_root}/controller.lock"',
            "another next pipeline controller is active",
            "trap '' HUP",
            "lock_is_held",
            "launch_qwen 1",
            "launch_wan 1",
            "pipeline_terminal.json",
        ):
            self.assertIn(marker, self.pipeline)
        self.assertIn('exec {qwen_lock_fd}>"${output_root}/controller.lock"', self.qwen)
        self.assertIn("another persistent Qwen controller is active", self.qwen)
        self.assertIn("watcher.lock", self.wan)
        self.assertIn("MOTIVE_FULL_MOTION_WAN_RESUME", self.wan)

    def test_launcher_does_not_submit_cancel_or_contact_remote_hosts(self) -> None:
        for forbidden in ("ssh ", "scp ", "sbatch", "scancel", "scontrol cancel"):
            self.assertNotIn(forbidden, self.pipeline)

    def test_missing_bindings_fail_before_any_slurm_query(self) -> None:
        result = subprocess.run(
            ["/bin/bash", str(PIPELINE)],
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
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing explicit binding", result.stderr)
        self.assertNotIn("scontrol", result.stderr)


if __name__ == "__main__":
    unittest.main()
