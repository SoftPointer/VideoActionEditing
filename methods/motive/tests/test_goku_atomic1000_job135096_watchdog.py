from __future__ import annotations

import re
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
WATCHDOG = ROOT / "tmp" / "supervise_goku_atomic1000_job135096_resume.sh"


def embedded(text: str, marker: str) -> str:
    opening = f"<<'{marker}'"
    start = text.index("\n", text.index(opening)) + 1
    end = text.index(f"\n{marker}\n", start)
    return text[start:end] + "\n"


class Job135096WatchdogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WATCHDOG.read_text(encoding="utf-8")

    def test_shell_and_embedded_python_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(WATCHDOG)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        markers = re.findall(r"<<'([A-Z][A-Z0-9_]*)'", self.text)
        self.assertEqual(
            markers, ["PY_CONTRACT", "PY_TERMINAL", "PY_PIDS", "PY_FAILURE_AUDIT"]
        )
        for marker in markers:
            compile(embedded(self.text, marker), marker, "exec")

    def test_current_job_paths_and_hashes_are_bound(self) -> None:
        required = (
            "job_id=135096",
            "job_name=goku-a1k-r2-g4",
            "epoch_0001",
            "expected_rows=2000",
            "expected_epoch_success=128",
            "expected_global_target=1000",
            "681251a969eddca71eaa25402e9fce9f3ee19a4484f3a8dea83162cf7bcc4e06",
            "650a4e6d155de2bf0f3da8dbaa92f81afbab53ce83a23b8fba9faf672869dc6d",
            "4f907d99f064e3fac25e5fdd2ae8ce4e278cb74facaae03a58e85323ed53fe0a",
            "1857a1e0b29a29889141d55195e9cde842e0e385b9e923340de4b4317bf3ddcb",
        )
        for value in required:
            self.assertIn(value, self.text)

    def test_recovery_requires_two_quiescent_observations_and_receipt(self) -> None:
        self.assertIn("quiescent_passes=$((quiescent_passes + 1))", self.text)
        self.assertIn("if (( quiescent_passes < 2 ))", self.text)
        self.assertGreaterEqual(self.text.count("recorded_live_pids"), 4)
        self.assertGreaterEqual(self.text.count("holder_only_steps"), 4)
        self.assertGreaterEqual(self.text.count("all_recovery_locks_free"), 4)
        receipt = self.text.index("failure_receipt=$(write_failure_audit")
        resume = self.text.index("launch_resume || true")
        self.assertLess(receipt, resume)
        self.assertIn('MOTIVE_ATOMIC_RESUME=1', self.text)
        self.assertIn('"planned_action": (', embedded(self.text, "PY_FAILURE_AUDIT"))

    def test_watchdog_cannot_release_delete_submit_or_cancel(self) -> None:
        self.assertNotRegex(
            self.text,
            r"(?m)^\s*(?:scancel|sbatch|salloc|rm|unlink)(?:\s|$)",
        )
        self.assertIn('"wrote_holder_release": False', self.text)
        self.assertIn("partial terminal artifacts exist; no release or resume was attempted", self.text)
        self.assertNotRegex(
            self.text,
            r"(?:>|tee\s+)[^\n]*release_holder_",
        )

    def test_resume_geometry_is_loaded_from_validated_contract(self) -> None:
        contract = embedded(self.text, "PY_CONTRACT")
        self.assertIn('required_atomic_topup_successes', contract)
        self.assertIn('topup["smoke_batch_rows"]', contract)
        self.assertIn('planner.get("workers") != 8', contract)
        self.assertIn('atomic.get("workers") != 8', contract)
        self.assertIn('wan.get("launcher_sha256") != wan_launcher_sha', contract)
        self.assertIn('topup.get("module") != "motive.goku_atomic_topup"', contract)
        self.assertIn(
            'MOTIVE_ATOMIC_SMOKE_BATCH_ROWS="${smoke_batch_rows}"', self.text
        )
        self.assertIn(
            'MOTIVE_ATOMIC_FINAL_GUARD_REMAINING="${final_guard_remaining}"',
            self.text,
        )

    def test_job_identity_and_dead_pid_evidence_are_checked(self) -> None:
        self.assertIn("holder_identity_is_running()", self.text)
        self.assertGreaterEqual(self.text.count("holder_identity_is_running"), 4)
        audit = embedded(self.text, "PY_FAILURE_AUDIT")
        self.assertIn('"kind": kind', audit)
        self.assertIn('"alive": False', audit)
        self.assertIn('"recorded_process_receipts": recorded_process_receipts', audit)


if __name__ == "__main__":
    unittest.main()
