from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (
    REPO_ROOT / "tmp" / "launch_fullmotion_v16_full128_persistent.sh"
)


class FullMotionV16Full128LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    @classmethod
    def _embedded_helper(cls, marker: str) -> str:
        opening = f"<<'{marker}'\n"
        start = cls.text.index(opening) + len(opening)
        end = cls.text.index(f"\n{marker}\n", start)
        return cls.text[start:end]

    def test_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_explicit_immutable_and_create_only_bindings(self) -> None:
        for variable in (
            "MOTIVE_FULL_MOTION_TREE_SHA",
            "MOTIVE_FULL_MOTION_FULL128_INPUT",
            "MOTIVE_FULL_MOTION_FULL128_INPUT_SHA",
            "MOTIVE_FULL_MOTION_FULL128_OUTPUT_ROOT",
            "MOTIVE_FULL_MOTION_RETRY_ROOTS",
            "MOTIVE_FULL_MOTION_WAN_ROOTS",
        ):
            self.assertIn(variable, self.text)
        self.assertIn("action_source_snapshot.py", self.text)
        self.assertIn("--expected-tree-sha256", self.text)
        self.assertIn("full128 input bytes differ", self.text)
        self.assertIn("create-only full128 target already exists", self.text)
        self.assertIn("artifact parent must be a pre-created plain directory", self.text)
        self.assertIn('[[ "${job_id}" == 118150 ]]', self.text)
        self.assertIn("Qwen3-VL-32B-Instruct", self.text)

    def test_path_helper_accepts_siblings_and_rejects_any_overlap(self) -> None:
        helper = self._embedded_helper("PY_PATH_CONTRACT")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            output = run / "full128"
            log = run / "full128.log"
            pid = run / "full128.pid"
            smoke = run / "smoke"
            retry_a = run / "retry-a"
            retry_b = run / "retry-b"
            wan_a = run / "wan-a"
            wan_b = run / "wan-b"

            valid = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    helper,
                    str(output),
                    str(log),
                    str(pid),
                    str(smoke),
                    f"{retry_a}:{retry_b}",
                    f"{wan_a}:{wan_b}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)

            nested_smoke = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    helper,
                    str(smoke / "full128"),
                    str(log),
                    str(pid),
                    str(smoke),
                    str(retry_a),
                    str(wan_a),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(nested_smoke.returncode, 0)
            self.assertIn("overlaps protected smoke root", nested_smoke.stderr)

            ancestor_of_retry = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    helper,
                    str(run),
                    str(root / "other.log"),
                    str(root / "other.pid"),
                    str(smoke),
                    str(retry_a),
                    str(wan_a),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(ancestor_of_retry.returncode, 0)
            self.assertIn("overlaps protected", ancestor_of_retry.stderr)

    def test_input_helper_closes_exact_128_unique_iids(self) -> None:
        helper = self._embedded_helper("PY_INPUT_CONTRACT")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.jsonl"
            valid_rows = [{"iid": f"iid-{index:03d}"} for index in range(128)]
            valid_path.write_text(
                "".join(json.dumps(row) + "\n" for row in valid_rows),
                encoding="utf-8",
            )
            valid = subprocess.run(
                [sys.executable, "-c", helper, str(valid_path), "128"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(len(valid.stdout.splitlines()), 128)

            short_path = root / "short.jsonl"
            short_path.write_text(
                "".join(json.dumps(row) + "\n" for row in valid_rows[:-1]),
                encoding="utf-8",
            )
            short = subprocess.run(
                [sys.executable, "-c", helper, str(short_path), "128"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(short.returncode, 0)
            self.assertIn("exactly 128 rows", short.stderr)

            duplicate_path = root / "duplicate.jsonl"
            duplicate_rows = list(valid_rows)
            duplicate_rows[-1] = duplicate_rows[0]
            duplicate_path.write_text(
                "".join(json.dumps(row) + "\n" for row in duplicate_rows),
                encoding="utf-8",
            )
            duplicate = subprocess.run(
                [sys.executable, "-c", helper, str(duplicate_path), "128"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("IIDs are not unique", duplicate.stderr)

    def test_first_four_shared_steps_create_eight_persistent_workers(self) -> None:
        self.assertIn("qwen_nodes=(", self.text)
        self.assertIn('"${allocated_nodes[0]}" "${allocated_nodes[1]}"', self.text)
        self.assertIn("wan_nodes=(", self.text)
        self.assertIn("needed_qwen_nodes=$(( (worker_count + 1) / 2 ))", self.text)
        self.assertIn("node_index<needed_qwen_nodes", self.text)
        self.assertIn('--job-name="v16-full-node-${node_index}"', self.text)
        self.assertIn('--ntasks="${tasks}" --ntasks-per-node="${tasks}"', self.text)
        self.assertIn("--gpus-per-task=4 --gpu-bind=none", self.text)
        self.assertIn("--kill-on-bad-exit=0", self.text)
        self.assertIn(
            "worker_index=$((node_index * 2 + SLURM_LOCALID))", self.text
        )
        self.assertIn("visible_devices=0,1,2,3", self.text)
        self.assertIn("visible_devices=4,5,6,7", self.text)
        self.assertIn("unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES", self.text)
        self.assertIn(
            '--worker-index "${worker_index}" --num-workers "${worker_count}"',
            self.text,
        )
        self.assertIn('--num-rows "${expected_rows}"', self.text)
        self.assertEqual(
            self.text.count("-m motive.goku_full_motion_qwen_v16"), 1
        )

    def test_each_worker_must_close_its_dynamic_strided_terminal_set(self) -> None:
        self.assertIn(
            "row_index<expected_rows; row_index+=worker_count",
            self.text,
        )
        self.assertIn(
            "(expected_rows - 1 - worker_index) / worker_count + 1",
            self.text,
        )
        self.assertIn("terminal/${iid}.receipt.json", self.text)
        self.assertIn("rows/${iid}/result.json", self.text)
        self.assertIn("wait_for_worker_terminal", self.text)
        self.assertIn(
            "one or more persistent workers lack their assigned terminal IIDs",
            self.text,
        )

    def test_default_remains_full128_but_row_count_has_no_128_cap(self) -> None:
        self.assertIn(
            "MOTIVE_FULL_MOTION_EXPECTED_ROWS:-128", self.text
        )
        self.assertIn("MOTIVE_FULL_MOTION_QWEN_WORKERS:-8", self.text)
        self.assertIn("worker_count <= 8", self.text)
        self.assertIn("worker_count <= expected_rows", self.text)
        self.assertNotIn("expected_rows <= 128", self.text)
        self.assertNotIn("rows<=128", self.text)

    def test_worker_terminal_verification_validates_receipt_hash_closure(self) -> None:
        helper = self._embedded_helper("PY_WORKER_TERMINAL")
        compile(helper, "persistent-worker-terminal-helper", "exec")
        self.assertIn("_validate_terminal_receipt", helper)
        self.assertIn("object_sha256(row)", helper)
        self.assertIn('counts = {"ok": 0, "error": 0}', helper)
        self.assertIn("list(range(worker, expected, workers))", helper)

    def test_only_first_four_must_be_step_free_and_gpu_idle(self) -> None:
        self.assertIn("assert_qwen_nodes_step_free", self.text)
        self.assertIn("first-four Qwen node already has step", self.text)
        self.assertIn("Numbered steps are permitted on the four Wan nodes", self.text)
        self.assertIn("for node in \"${qwen_nodes[@]}\"; do", self.text)
        self.assertIn("rocm-smi --showuse --showmemuse", self.text)
        self.assertIn("rocm-smi --showpids", self.text)
        self.assertNotIn("assert_holder_only_steps", self.text)

    def test_streaming_is_per_iid_without_global_gate_or_wan_dispatch(self) -> None:
        self.assertIn("passed/<iid>.jsonl", self.text)
        self.assertIn("stream=%s/passed", self.text)
        self.assertIn("--allow-errors", self.text)
        self.assertIn("trap '' HUP", self.text)
        for forbidden in (
            "goku_full_motion_smoke_gate",
            "--minimum-hard-passes",
            "authorizes_full_run",
            "wan22_i2v_batch",
            "wan_dispatch",
        ):
            self.assertNotIn(forbidden, self.text)

    def test_resume_controller_is_single_owner_and_hup_safe(self) -> None:
        self.assertIn('exec {qwen_lock_fd}>"${output_root}/controller.lock"', self.text)
        self.assertIn('flock -n "${qwen_lock_fd}"', self.text)
        self.assertIn("another persistent Qwen controller is active", self.text)
        self.assertIn("trap '' HUP", self.text)
        lock = self.text.index('flock -n "${qwen_lock_fd}"')
        launch = self.text.index("controller_main()", lock)
        close = self.text.index('exec {qwen_lock_fd}>&-', launch)
        self.assertLess(lock, launch)
        self.assertLess(launch, close)

    def test_missing_explicit_bindings_stop_before_slurm(self) -> None:
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
            },
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("set the explicit full128 binding", completed.stderr)
        self.assertNotIn("scontrol", completed.stderr)


if __name__ == "__main__":
    unittest.main()
