from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "auh_run_graft_phase_a_short_trajectory_world8_r3.sbatch"
SUBMITTER = ROOT / "scripts" / "auh_submit_graft_phase_a_short_trajectory_world8_r3.sh"
RUNNER = ROOT / "run_graft_phase_a_short_trajectory_diagnostic_gpu_v1.py"
PLAN = ROOT / "assets" / "graft_phase_a_short_trajectory_world8_plan_v1.json"
CLOSURE = ROOT / "assets" / "graft_phase_a_short_trajectory_runtime_closure_v1.json"

BEGIN = "# BEGIN GRAFT_SHORT_TRAJECTORY_R3_JOB_LOCAL_CACHE_V1"
END = "# END GRAFT_SHORT_TRAJECTORY_R3_JOB_LOCAL_CACHE_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GraftShortTrajectoryR3LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="ascii")
        start = cls.source.index(BEGIN)
        stop = cls.source.index(END, start) + len(END)
        cls.cache_fragment = cls.source[start:stop]
        failure_start = cls.source.index('task_scratch=""')
        failure_stop = cls.source.index("trap cleanup EXIT INT TERM HUP", failure_start)
        failure_stop += len("trap cleanup EXIT INT TERM HUP")
        cls.failure_fragment = cls.source[failure_start:failure_stop]
        cleanup_start = cls.source.index("cleanup_task_scratch_exact()")
        cleanup_stop = cls.source.index("\n\nfor path in", cleanup_start)
        cls.exact_cleanup_fragment = cls.source[cleanup_start:cleanup_stop]

    def run_cache_fragment(self, suffix: str) -> subprocess.CompletedProcess[str]:
        prelude = r'''
set -Eeuo pipefail
umask 077
fail() { echo "TEST_FAIL: $*" >&2; exit 2; }
SLURM_JOB_ID=990001
task_scratch="$(/usr/bin/mktemp -d "/tmp/graft-short-traj-r3-test-${SLURM_JOB_ID}.XXXXXX")"
task_scratch_identity="$(/usr/bin/stat -c '%d:%i' -- "${task_scratch}")"
readonly task_scratch task_scratch_identity
test_cleanup() {
  /usr/bin/chmod -R u+w -- "${task_scratch}" 2>/dev/null || true
  /usr/bin/rm -rf -- "${task_scratch}" 2>/dev/null || true
}
trap test_cleanup EXIT
'''
        environment = {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "MIOPEN_USER_DB_PATH": "/definitely/read-only/inherited-miopen-user",
            "MIOPEN_CUSTOM_CACHE_DIR": "/definitely/read-only/inherited-miopen-custom",
            "TORCH_EXTENSIONS_DIR": "/definitely/read-only/inherited-torch",
            "TRITON_CACHE_DIR": "/definitely/read-only/inherited-triton",
            "XDG_CACHE_HOME": "/definitely/read-only/inherited-xdg",
            "PYTHONPYCACHEPREFIX": "/definitely/read-only/inherited-pycache",
            "TMPDIR": "/definitely/read-only/inherited-tmp",
        }
        return subprocess.run(
            ["/bin/bash", "-p", "-c", prelude + self.cache_fragment + "\n" + suffix],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=environment,
            timeout=30,
        )

    def run_terminal_harness(self, body: str) -> tuple[subprocess.CompletedProcess[str], Path, tempfile.TemporaryDirectory[str]]:
        temporary = tempfile.TemporaryDirectory(prefix="graft-r3-terminal-")
        root = Path(temporary.name)
        output = root / "output"
        output.mkdir(mode=0o700)
        python_bin = Path(sys.executable).resolve(strict=True)
        prelude = f'''
set -Eeuo pipefail
umask 077
fail() {{ echo "TEST_FAIL: $*" >&2; exit 2; }}
python_bin={python_bin}
output_root={output}
SLURM_JOB_ID=990002
runner_sha256={'1' * 64}
plan_sha256={'2' * 64}
'''
        initialize = r'''
task_scratch="$(/usr/bin/mktemp -d "/tmp/graft-short-traj-r3-test-${SLURM_JOB_ID}.XXXXXX")"
task_scratch_identity="$(/usr/bin/stat -c '%d:%i' -- "${task_scratch}")"
readonly task_scratch task_scratch_identity
'''
        script = (
            prelude
            + self.failure_fragment
            + "\n"
            + initialize
            + self.cache_fragment
            + "\n"
            + self.exact_cleanup_fragment
            + "\n"
            + body
        )
        completed = subprocess.run(
            ["/bin/bash", "-p", "-c", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            timeout=30,
        )
        return completed, output, temporary

    def test_scientific_and_runtime_pins_are_unchanged(self) -> None:
        self.assertEqual(
            sha256(RUNNER),
            "fc2c61cc7d80c234cefa681a900457f31584ea9115bb1531fc63e90f8fc3b74b",
        )
        self.assertEqual(
            sha256(PLAN),
            "404861bb33bfd6a37a045b8c6227facc47a7b7f27cf465728bcb0c6bb4bf42cf",
        )
        self.assertEqual(
            sha256(CLOSURE),
            "3fc589fd57cb1cae41c0717590d0552d3e17b7b0ec7fce0d6969da5ce8a76372",
        )

    def test_launcher_syntax_and_ordered_failure_boundary(self) -> None:
        completed = subprocess.run(
            ["/bin/bash", "-n", str(LAUNCHER)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        trap_at = self.source.index("trap cleanup EXIT INT TERM HUP")
        mktemp_at = self.source.index('task_scratch="$(/usr/bin/mktemp')
        cache_at = self.source.index(BEGIN)
        torchrun_at = self.source.index('"${python_bin}" -I -B -m torch.distributed.run')
        cleanup_at = self.source.rindex("cleanup_task_scratch_exact\n")
        success_receipt_at = self.source.index('fd=os.open(output/"receipt.json"')
        self.assertLess(trap_at, mktemp_at)
        self.assertLess(mktemp_at, cache_at)
        self.assertLess(cache_at, torchrun_at)
        self.assertLess(torchrun_at, cleanup_at)
        self.assertLess(cleanup_at, success_receipt_at)
        self.assertTrue(self.source.rstrip().endswith("PY"))
        self.assertIn("os.fchmod(fd,0o444)\nexcept BaseException:", self.source)
        self.assertIn("os._exit(0)\nPY", self.source)

    def test_poisoned_inherited_caches_are_replaced_by_private_writable_paths(self) -> None:
        completed = self.run_cache_fragment(
            r'''
for path in "${runtime_cache_paths[@]}"; do
  [[ "${path}" == "${task_scratch}/"* && -d "${path}" && -w "${path}" && ! -L "${path}" ]]
done
[[ "${MIOPEN_USER_DB_PATH}" == "${task_scratch}/cache/miopen-user" ]]
[[ "${MIOPEN_CUSTOM_CACHE_DIR}" == "${task_scratch}/cache/miopen-custom" ]]
echo CACHE_OK
'''
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "CACHE_OK")

    def test_environment_escape_is_rejected(self) -> None:
        completed = self.run_cache_fragment(
            'MIOPEN_USER_DB_PATH=/tmp/escaped-miopen; validate_runtime_cache_paths\n'
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("MIOpen user DB path changed", completed.stderr)

    def test_same_owner_inode_replacement_is_rejected(self) -> None:
        completed = self.run_cache_fragment(
            r'''
/usr/bin/mv -- "${TRITON_CACHE_DIR}" "${TRITON_CACHE_DIR}.original"
/usr/bin/mkdir -- "${TRITON_CACHE_DIR}"
/usr/bin/chmod 700 -- "${TRITON_CACHE_DIR}"
validate_runtime_cache_paths
'''
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("runtime cache inode changed", completed.stderr)

    def test_symlink_replacement_is_rejected(self) -> None:
        completed = self.run_cache_fragment(
            r'''
/usr/bin/rm -rf -- "${XDG_CACHE_HOME}"
/usr/bin/ln -s -- /tmp "${XDG_CACHE_HOME}"
validate_runtime_cache_paths
'''
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("runtime cache is not an exact writable directory", completed.stderr)

    def test_cache_preflight_failure_retains_scratch_and_only_failure_receipt(self) -> None:
        scratch: Path | None = None
        completed, output, temporary = self.run_terminal_harness(
            'task_path="${task_scratch}"; /usr/bin/chmod 755 -- "${TRITON_CACHE_DIR}"; validate_runtime_cache_paths\n'
        )
        try:
            self.assertEqual(completed.returncode, 2)
            self.assertFalse((output / "receipt.json").exists())
            failure = output / "failure.receipt.json"
            self.assertTrue(failure.is_file())
            self.assertEqual(stat.S_IMODE(failure.stat().st_mode), 0o444)
            scratch = Path(completed.stderr.split("scratch=", 1)[1].splitlines()[0])
            self.assertTrue(scratch.is_dir())
        finally:
            if scratch is not None and scratch.is_dir():
                shutil.rmtree(scratch)
            temporary.cleanup()

    def test_simulated_runner_failure_retains_scratch_and_only_failure_receipt(self) -> None:
        scratch: Path | None = None
        completed, output, temporary = self.run_terminal_harness("exit 7\n")
        try:
            self.assertEqual(completed.returncode, 7)
            self.assertFalse((output / "receipt.json").exists())
            failure = output / "failure.receipt.json"
            self.assertTrue(failure.is_file())
            self.assertEqual(stat.S_IMODE(failure.stat().st_mode), 0o444)
            scratch = Path(completed.stderr.split("scratch=", 1)[1].splitlines()[0])
            self.assertTrue(scratch.is_dir())
        finally:
            if scratch is not None and scratch.is_dir():
                shutil.rmtree(scratch)
            temporary.cleanup()

    def test_simulated_success_cleans_scratch_before_only_success_receipt(self) -> None:
        completed, output, temporary = self.run_terminal_harness(
            r'''
saved_scratch="${task_scratch}"
cleanup_task_scratch_exact
/usr/bin/touch -- "${output_root}/receipt.json"
/usr/bin/chmod 0444 -- "${output_root}/receipt.json"
[[ ! -e "${saved_scratch}" && ! -L "${saved_scratch}" ]]
'''
        )
        try:
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "receipt.json").is_file())
            self.assertEqual(stat.S_IMODE((output / "receipt.json").stat().st_mode), 0o444)
            self.assertFalse((output / "failure.receipt.json").exists())
        finally:
            temporary.cleanup()

    def test_success_materializer_failure_has_no_0444_success_receipt(self) -> None:
        completed, output, temporary = self.run_terminal_harness(
            r'''
cleanup_task_scratch_exact
/usr/bin/touch -- "${output_root}/receipt.json"
/usr/bin/chmod 0600 -- "${output_root}/receipt.json"
exit 9
'''
        )
        try:
            self.assertEqual(completed.returncode, 9)
            partial = output / "receipt.json"
            self.assertTrue(partial.is_file())
            self.assertEqual(stat.S_IMODE(partial.stat().st_mode), 0o600)
            failure = output / "failure.receipt.json"
            self.assertTrue(failure.is_file())
            self.assertEqual(stat.S_IMODE(failure.stat().st_mode), 0o444)
        finally:
            temporary.cleanup()

    def test_r3_submitter_reserves_exactly_once_before_nonhold_sbatch(self) -> None:
        submitter = SUBMITTER.read_text(encoding="ascii")
        reservation = submitter.index(
            "fd=os.open(receipt_path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)"
        )
        sbatch = submitter.index("completed=subprocess.run(argv")
        finalize = submitter.index("os.fchmod(fd,0o444)")
        self.assertLess(reservation, sbatch)
        self.assertLess(sbatch, finalize)
        self.assertEqual(submitter.count("completed=subprocess.run(argv"), 1)
        self.assertNotIn("--dependency", submitter)
        self.assertNotIn('"--hold"', submitter)
        self.assertIn('"hold":False,"dependency":None', submitter)

    def test_r3_submitter_failure_is_exact_once_0600_reservation(self) -> None:
        self._exercise_r3_submitter(sbatch_success=False)

    def test_r3_submitter_success_is_same_inode_0444_nonhold_receipt(self) -> None:
        self._exercise_r3_submitter(sbatch_success=True)

    def _exercise_r3_submitter(self, *, sbatch_success: bool) -> None:
        submitter = SUBMITTER.read_text(encoding="ascii")
        with tempfile.TemporaryDirectory(prefix="graft-r3-submitter-") as raw:
            root = Path(raw)
            marker = root / "sbatch.calls"
            fake_sbatch = root / "sbatch"
            fake_sbatch.write_text(
                "#!/bin/bash\nprintf x >> "
                + str(marker)
                + ("\nprintf '314159;cluster\\n'\nexit 0\n" if sbatch_success else "\nexit 17\n"),
                encoding="ascii",
            )
            fake_sbatch.chmod(0o755)
            transformed = root / "submit.sh"
            transformed.write_text(
                submitter.replace("/usr/bin/sbatch", str(fake_sbatch)),
                encoding="ascii",
            )
            transformed.chmod(0o755)
            inputs: dict[str, Path] = {}
            for name in ("archive", "closure", "checkpoint_manifest", "plan", "terminal", "launcher"):
                path = root / name
                path.write_bytes((name + "\n").encode("ascii"))
                inputs[name] = path
            python_bin = Path(sys.executable).resolve(strict=True)
            output = root / "runs" / "fresh"
            output.parent.mkdir()
            environment = {
                "GRAFT_TRAJ_SOURCE_ARCHIVE": str(inputs["archive"]),
                "GRAFT_TRAJ_SOURCE_ARCHIVE_SHA256": sha256(inputs["archive"]),
                "GRAFT_TRAJ_RUNTIME_CLOSURE": str(inputs["closure"]),
                "GRAFT_TRAJ_RUNTIME_CLOSURE_SHA256": sha256(inputs["closure"]),
                "GRAFT_TRAJ_PYTHON_BIN": str(python_bin),
                "GRAFT_TRAJ_PYTHON_SHA256": sha256(python_bin),
                "BERNINI_OFFICIAL_ROOT": str(root / "bernini"),
                "BERNINI_VEOMNI_ROOT": str(root / "veomni"),
                "BERNINI_ACTION_CHECKPOINT": str(root / "checkpoint"),
                "BERNINI_CHECKPOINT_CONTENT_MANIFEST": str(inputs["checkpoint_manifest"]),
                "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256": sha256(inputs["checkpoint_manifest"]),
                "GRAFT_TRAJ_PLAN": str(inputs["plan"]),
                "GRAFT_TRAJ_PLAN_SHA256": sha256(inputs["plan"]),
                "GRAFT_TRAJ_TERMINAL_ADMISSION": str(inputs["terminal"]),
                "GRAFT_TRAJ_TERMINAL_ADMISSION_SHA256": sha256(inputs["terminal"]),
                "GRAFT_TRAJ_TERMINAL_MATERIALIZER_RUNTIME_SHA256": "1" * 64,
                "GRAFT_TRAJ_OUTPUT_ROOT": str(output),
                "GRAFT_TRAJ_LAUNCHER_SOURCE": str(inputs["launcher"]),
                "GRAFT_TRAJ_LAUNCHER_SHA256": sha256(inputs["launcher"]),
                "GRAFT_TRAJ_RUNNER_SHA256": "2" * 64,
            }
            completed = subprocess.run(
                ["/usr/bin/bash", "-p", str(transformed)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C", **environment},
                check=False,
                timeout=30,
            )
            receipt = Path(str(output) + ".submission.receipt.json")
            self.assertTrue(receipt.is_file())
            self.assertEqual(marker.read_text(encoding="ascii"), "x")
            if sbatch_success:
                self.assertEqual(completed.returncode, 0, completed.stderr.decode())
                self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o444)
                value = json.loads(receipt.read_text(encoding="ascii"))
                self.assertEqual(value["submitted_job"]["job_id"], "314159")
                self.assertEqual(value["request"]["job_name"], "graft-short-traj-r3")
                self.assertIsNone(value["request"]["dependency"])
                self.assertFalse(value["request"]["hold"])
                self.assertTrue(value["outputs"]["same_inode_retained_across_sbatch"])
            else:
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o600)
                self.assertEqual(receipt.stat().st_size, 0)
                retry = subprocess.run(
                    ["/usr/bin/bash", "-p", str(transformed)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C", **environment},
                    check=False,
                    timeout=30,
                )
                self.assertNotEqual(retry.returncode, 0)
                self.assertEqual(marker.read_text(encoding="ascii"), "x")


if __name__ == "__main__":
    unittest.main()
