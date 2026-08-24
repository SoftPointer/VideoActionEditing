from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tmp" / "supervise_goku_atomic1000_job123440_resume.sh"


class Atomic1000ResumeSupervisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_embedded_python_compiles(self) -> None:
        for marker in ("PY_CONTRACT", "PY_TERMINAL", "PY_PIDS"):
            opening = f"<<'{marker}'"
            start = self.text.index("\n", self.text.index(opening)) + 1
            end = self.text.index(f"\n{marker}\n", start)
            compile(self.text[start:end] + "\n", marker, "exec")

    def test_run_and_artifact_hashes_are_frozen(self) -> None:
        for marker in (
            "fullmotion_atomic1000_stream_v3_20260805T192958Z",
            "job_id=123440",
            "tree_sha=909cc1f0a51f12288f33dda7a2cf8642d117a81936733066f3c99b152060afad",
            "launcher_sha=13b7a928e7a1d22b1a3db1a770992d2c39127fbb22d0174a62996b6d30f6fc0b",
            "selected_sha=ed828b935526803c39ac9d679603b274f7d98ac081203b54f6b2b3ba07ff747a",
            "wan_launcher_sha=d992690f8bd0f738bba10ef6579466d8be5f8bc3a9395a6bacd01fa6f9fe1079",
        ):
            self.assertIn(marker, self.text)

    def test_resume_is_closed_and_exact(self) -> None:
        launch = self.text.split("launch_resume() {", 1)[1]
        self.assertIn("/usr/bin/env -i", launch)
        self.assertIn("MOTIVE_ATOMIC_RESUME=1", launch)
        self.assertIn("MOTIVE_ATOMIC_EXPECTED_ROWS=1235", launch)
        self.assertIn("MOTIVE_ATOMIC_MINIMUM_FINAL_SUCCESS=1000", launch)
        self.assertIn('MOTIVE_ATOMIC_WAN_LAUNCHER="${wan_launcher}"', launch)
        self.assertNotIn("sbatch", self.text)
        self.assertNotIn("scancel", self.text)

    def test_restart_requires_pid_step_and_both_lock_gates(self) -> None:
        loop = self.text.split("while true; do", 1)[1]
        order = [
            loop.index("recorded_live_pids"),
            loop.index("holder_only_steps"),
            loop.index("all_recovery_locks_free"),
            loop.index("quiescent_passes=$((quiescent_passes + 1))"),
            loop.index("launch_resume"),
        ]
        self.assertEqual(order, sorted(order))
        self.assertIn('flock -n "${controller_lock}" -c true', self.text)
        self.assertIn('flock -n "${lock}" -c true', self.text)
        self.assertIn('"${job_id}."[0-9]*) return 1', self.text)

    def test_audit_and_hup_safety(self) -> None:
        self.assertIn("trap '' HUP", self.text)
        self.assertIn("resume_supervisor_job${job_id}.audit.tsv", self.text)
        for event in (
            "supervisor_start",
            "recorded_process_alive",
            "numbered_steps_active",
            "recovery_lock_busy",
            "launch_resume",
            "launch_return",
            "terminal_complete",
        ):
            self.assertIn(event, self.text)


if __name__ == "__main__":
    unittest.main()
