from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_run_graft_phase_a_native_gpu_canary_dual4_all8_v1.sbatch"
)
SUBMIT = (
    METHOD_ROOT
    / "scripts/auh_submit_graft_phase_a_native_gpu_canary_dual4_all8_v1.sh"
)
LAUNCHER_SHA = "da85eb4db29216ebbaf66bce6d6094460a18e4e43089facbfde5530385abe587"
EXPORT_NAMES = (
    "GRAFT_PHASE_A_SOURCE_ARCHIVE",
    "GRAFT_PHASE_A_SOURCE_ARCHIVE_SHA256",
    "GRAFT_PHASE_A_RUNTIME_CLOSURE_MANIFEST",
    "GRAFT_PHASE_A_RUNTIME_CLOSURE_MANIFEST_SHA256",
    "GRAFT_PHASE_A_PYTHON_BIN",
    "GRAFT_PHASE_A_PYTHON_SHA256",
    "BERNINI_OFFICIAL_ROOT",
    "BERNINI_VEOMNI_ROOT",
    "BERNINI_ACTION_CHECKPOINT",
    "BERNINI_CHECKPOINT_CONTENT_MANIFEST",
    "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
    "GRAFT_PHASE_A_CELL_SPEC",
    "GRAFT_PHASE_A_CELL_SPEC_SHA256",
    "GRAFT_PHASE_A_OUTPUT_ROOT",
    "GRAFT_PHASE_A_LAUNCHER_SOURCE",
    "GRAFT_PHASE_A_LAUNCHER_SHA256",
    "GRAFT_PHASE_A_RUNNER_SHA256",
    "GRAFT_PHASE_A_CLOSURE_CORE_SHA256",
    "GRAFT_PHASE_A_REBINDER_SHA256",
)
SCHEDULER_ARGUMENTS = (
    "--parsable",
    f"--export={','.join(EXPORT_NAMES)}",
    "--partition=faculty",
    "--qos=bgqos",
    "--nodes=1",
    "--ntasks=1",
    "--cpus-per-task=32",
    "--mem=256G",
    "--gres=gpu:mi210:8",
    "--time=08:00:00",
    "--job-name=graft-pa-v1",
    "--exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def fake_sbatch_source(
    *,
    behavior: str,
    observation: Path,
    launcher: Path,
) -> bytes:
    python = Path(sys.executable).resolve(strict=True)
    return f'''#!{python}
import json
import os
from pathlib import Path
import sys
behavior = {behavior!r}
observation = Path({str(observation)!r})
launcher = Path({str(launcher)!r})
fd_path = Path(sys.argv[-1])
observation.write_text(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd(), "environment": dict(os.environ), "launcher_fd_sha256": __import__("hashlib").sha256(fd_path.read_bytes()).hexdigest()}}, sort_keys=True, separators=(",", ":")), encoding="ascii")
if behavior == "bad_exit":
    print("fake failure", file=sys.stderr)
    raise SystemExit(7)
if behavior == "bad_jobid":
    print("not-a-job-id")
    raise SystemExit(0)
if behavior == "replace_launcher":
    launcher.unlink()
    launcher.write_bytes(b"attacker")
    launcher.chmod(0o555)
print("765432;fakecluster")
'''.encode("ascii")


class Fixture:
    def __init__(self, root: Path, *, behavior: str = "success") -> None:
        self.root = root.resolve()
        self.inputs = self.root / "inputs"
        self.outputs = self.root / "outputs"
        self.inputs.mkdir()
        self.outputs.mkdir()
        self.observation = self.root / "sbatch-observation.json"

        self.launcher = self.inputs / "launcher.sbatch"
        self.launcher.write_bytes(LAUNCHER.read_bytes())
        self.launcher.chmod(0o555)
        assert sha(self.launcher.read_bytes()) == LAUNCHER_SHA

        self.archive = self.inputs / "runtime.tar"
        self.archive.write_bytes(b"archive")
        self.archive.chmod(0o444)
        self.closure = self.inputs / "closure.json"
        self.closure.write_bytes(b"closure\n")
        self.closure.chmod(0o444)
        self.cell = self.inputs / "cell.json"
        self.cell.write_bytes(b"cell\n")
        self.cell.chmod(0o444)
        self.checkpoint_manifest = self.inputs / "checkpoint.sha256"
        self.checkpoint_manifest.write_bytes(b"checkpoint manifest\n")
        self.checkpoint_manifest.chmod(0o444)
        self.bernini = self.root / "bernini"
        self.veomni = self.root / "veomni"
        self.checkpoint = self.root / "checkpoint"
        for directory in (self.bernini, self.veomni, self.checkpoint):
            directory.mkdir()
        self.output_root = self.outputs / "phase-a-canary"

        self.fake_sbatch = self.inputs / "sbatch"
        self.fake_sbatch.write_bytes(
            fake_sbatch_source(
                behavior=behavior,
                observation=self.observation,
                launcher=self.launcher,
            )
        )
        self.fake_sbatch.chmod(0o555)

        source = SUBMIT.read_text(encoding="utf-8")
        replacements = {
            "readonly required_sbatch_path=/usr/bin/sbatch": (
                f"readonly required_sbatch_path={self.fake_sbatch}"
            ),
            "readonly required_fd_root=/proc/self/fd": (
                "readonly required_fd_root=/dev/fd"
            ),
            "readonly required_fd_stat_identity=true": (
                "readonly required_fd_stat_identity=false"
            ),
            "readonly required_execute_sbatch_from_fd=true": (
                "readonly required_execute_sbatch_from_fd=false"
            ),
        }
        for old, new in replacements.items():
            if old not in source:
                raise AssertionError(f"fixture pin absent: {old}")
            source = source.replace(old, new)
        self.wrapper = self.inputs / "submit.sh"
        self.wrapper.write_text(source, encoding="utf-8")
        self.wrapper.chmod(0o444)

    def environment(self, **extra: str) -> dict[str, str]:
        python = Path(sys.executable).resolve(strict=True)
        values = {
            "GRAFT_PHASE_A_SOURCE_ARCHIVE": str(self.archive),
            "GRAFT_PHASE_A_SOURCE_ARCHIVE_SHA256": sha(self.archive.read_bytes()),
            "GRAFT_PHASE_A_RUNTIME_CLOSURE_MANIFEST": str(self.closure),
            "GRAFT_PHASE_A_RUNTIME_CLOSURE_MANIFEST_SHA256": sha(self.closure.read_bytes()),
            "GRAFT_PHASE_A_PYTHON_BIN": str(python),
            "GRAFT_PHASE_A_PYTHON_SHA256": sha(python.read_bytes()),
            "BERNINI_OFFICIAL_ROOT": str(self.bernini),
            "BERNINI_VEOMNI_ROOT": str(self.veomni),
            "BERNINI_ACTION_CHECKPOINT": str(self.checkpoint),
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST": str(self.checkpoint_manifest),
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256": sha(self.checkpoint_manifest.read_bytes()),
            "GRAFT_PHASE_A_CELL_SPEC": str(self.cell),
            "GRAFT_PHASE_A_CELL_SPEC_SHA256": sha(self.cell.read_bytes()),
            "GRAFT_PHASE_A_OUTPUT_ROOT": str(self.output_root),
            "GRAFT_PHASE_A_LAUNCHER_SOURCE": str(self.launcher),
            "GRAFT_PHASE_A_LAUNCHER_SHA256": LAUNCHER_SHA,
            "GRAFT_PHASE_A_RUNNER_SHA256": "1" * 64,
            "GRAFT_PHASE_A_CLOSURE_CORE_SHA256": "2" * 64,
            "GRAFT_PHASE_A_REBINDER_SHA256": "3" * 64,
        }
        values.update(extra)
        return values

    def run(
        self,
        *arguments: str,
        environment: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", "-p", str(self.wrapper), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.environment() if environment is None else environment,
            cwd=cwd,
            timeout=30,
        )

    @property
    def receipt(self) -> Path:
        return self.outputs / "phase-a-canary.submission.receipt.json"


class SubmitPhaseAAll8Tests(unittest.TestCase):
    def fixture(self, *, behavior: str = "success") -> Fixture:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Fixture(Path(temporary.name), behavior=behavior)

    def test_syntax_hardcode_no_args_and_exact_scheduler_contract(self) -> None:
        result = subprocess.run(
            ["/bin/bash", "-n", str(SUBMIT)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        source = SUBMIT.read_text(encoding="utf-8")
        self.assertTrue(source.startswith("#!/bin/bash -p\n"))
        self.assertIn(f"readonly required_launcher_sha256={LAUNCHER_SHA}", source)
        self.assertEqual(sha(LAUNCHER.read_bytes()), LAUNCHER_SHA)
        self.assertIn("readonly required_sbatch_path=/usr/bin/sbatch", source)
        self.assertIn('[[ "$#" -eq 0 ]]', source)
        self.assertIn("/usr/bin/env -i", source)
        self.assertNotIn("--export=ALL", source)
        for argument in SCHEDULER_ARGUMENTS:
            if argument.startswith("--export="):
                self.assertIn(
                    "readonly export_names_csv="
                    + argument[len("--export="):],
                    source,
                )
            else:
                self.assertIn(argument, source)

    def test_success_has_exact_child_env_argv_and_submission_not_job_receipt(self) -> None:
        fixture = self.fixture()
        result = fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        observation = json.loads(fixture.observation.read_bytes())
        self.assertEqual(observation["argv"][:-1], list(SCHEDULER_ARGUMENTS))
        self.assertTrue(observation["argv"][-1].startswith("/dev/fd/"))
        self.assertEqual(observation["launcher_fd_sha256"], LAUNCHER_SHA)
        expected_env = {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            **fixture.environment(),
        }
        self.assertEqual(observation["environment"], expected_env)
        raw = fixture.receipt.read_bytes()
        receipt = json.loads(raw)
        self.assertEqual(raw, canonical(receipt) + b"\n")
        self.assertEqual(stat.S_IMODE(fixture.receipt.stat().st_mode), 0o444)
        unsigned = dict(receipt)
        claimed = unsigned.pop("receipt_digest")
        self.assertEqual(claimed, sha(canonical(unsigned)))
        self.assertTrue(receipt["submission_success"])
        self.assertIsNone(receipt["job_success"])
        self.assertFalse(receipt["job_terminal_state_observed"])
        self.assertEqual(receipt["submitted_job"]["job_id"], "765432")
        self.assertEqual(receipt["export_contract"]["names"], list(EXPORT_NAMES))
        self.assertTrue(all(value is False for value in receipt["authority"].values()))
        self.assertTrue(
            receipt["failure_semantics"]["submission_success_is_not_job_success"]
        )

    def test_path_ld_sbatch_python_bash_and_function_poison_are_purged(self) -> None:
        fixture = self.fixture()
        marker = fixture.root / "bash-env-ran"
        bash_env = fixture.root / "bash-env"
        bash_env.write_text(f"/usr/bin/touch {marker}\n", encoding="utf-8")
        environment = fixture.environment(
            PATH="/attacker",
            LD_PRELOAD="/attacker.so",
            LD_LIBRARY_PATH="/attacker/lib",
            PYTHONPATH="/attacker/python",
            SBATCH_EXPORT="ALL",
            SBATCH_GRES="gpu:attacker:99",
            BASH_ENV=str(bash_env),
            **{"BASH_FUNC_exec%%": '() { /usr/bin/false; }'},
        )
        result = fixture.run(environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())
        observation = json.loads(fixture.observation.read_bytes())
        self.assertEqual(
            set(observation["environment"]),
            {"PATH", "LC_ALL", "LANG", *EXPORT_NAMES},
        )
        self.assertEqual(observation["argv"][:-1], list(SCHEDULER_ARGUMENTS))

    def test_hostile_submit_cwd_torch_package_and_site_shadows_do_not_execute(self) -> None:
        fixture = self.fixture()
        hostile = fixture.root / "hostile-submit-cwd"
        hostile.mkdir()
        markers = {
            "torch.py": fixture.root / "torch-py-ran",
            "site.py": fixture.root / "site-py-ran",
            "sitecustomize.py": fixture.root / "sitecustomize-ran",
        }
        for name, marker in markers.items():
            (hostile / name).write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
                encoding="utf-8",
            )
        torch_package = hostile / "torch"
        torch_package.mkdir()
        package_marker = fixture.root / "torch-package-ran"
        (torch_package / "__init__.py").write_text(
            f"from pathlib import Path\nPath({str(package_marker)!r}).write_text('ran')\n",
            encoding="utf-8",
        )

        result = fixture.run(cwd=hostile)
        self.assertEqual(result.returncode, 0, result.stderr)
        observation = json.loads(fixture.observation.read_bytes())
        self.assertEqual(observation["cwd"], str(hostile))
        self.assertTrue(fixture.receipt.is_file())
        for marker in (*markers.values(), package_marker):
            self.assertFalse(marker.exists(), marker)

        launcher = fixture.launcher.read_text(encoding="utf-8")
        cwd_boundary = launcher.index('builtin cd -- "${task_scratch}"')
        first_python = launcher.index('"${python_bin}" -I -S -B - "$1"')
        self.assertLess(cwd_boundary, first_python)
        self.assertIn(
            'exec "${python_bin}" -I -B -m torch.distributed.run', launcher
        )

    def test_extra_interface_and_arbitrary_argument_fail_before_sbatch(self) -> None:
        fixture = self.fixture()
        environment = fixture.environment(GRAFT_PHASE_A_ATTACKER_OVERRIDE="1")
        result = fixture.run(environment=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected Phase-A interface variable", result.stderr)
        self.assertFalse(fixture.observation.exists())
        result = fixture.run("--export=ALL")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("arbitrary arguments are forbidden", result.stderr)

    def test_nonprivileged_invocation_fails_closed(self) -> None:
        fixture = self.fixture()
        result = subprocess.run(
            ["/bin/bash", str(fixture.wrapper)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=fixture.environment(),
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Bash privileged mode is required", result.stderr)
        self.assertFalse(fixture.observation.exists())

    def test_launcher_path_replacement_after_fd_submission_fails(self) -> None:
        fixture = self.fixture(behavior="replace_launcher")
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("launcher path or fd identity changed", result.stderr)
        self.assertTrue(fixture.observation.exists())
        self.assertFalse(fixture.receipt.exists())

    def test_bad_sbatch_exit_and_bad_jobid_publish_no_receipt(self) -> None:
        for behavior, message in (
            ("bad_exit", "sbatch failed with exit 7"),
            ("bad_jobid", "sbatch parsable job ID differs"),
        ):
            fixture = self.fixture(behavior=behavior)
            result = fixture.run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
            self.assertFalse(fixture.receipt.exists())

    def test_preexisting_receipt_blocks_submission_and_is_preserved(self) -> None:
        fixture = self.fixture()
        fixture.receipt.write_bytes(b"preexisting")
        fixture.receipt.chmod(0o444)
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(fixture.receipt.read_bytes(), b"preexisting")
        self.assertFalse(fixture.observation.exists())

    def test_preexisting_output_root_blocks_submission_and_is_preserved(self) -> None:
        fixture = self.fixture()
        fixture.output_root.mkdir()
        sentinel = fixture.output_root / "sentinel"
        sentinel.write_bytes(b"owned")
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_bytes(), b"owned")
        self.assertFalse(fixture.observation.exists())

    def test_wrong_launcher_environment_sha_fails_before_sbatch(self) -> None:
        fixture = self.fixture()
        environment = fixture.environment()
        environment["GRAFT_PHASE_A_LAUNCHER_SHA256"] = "0" * 64
        result = fixture.run(environment=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("launcher SHA-256 differs from the wrapper hardcode", result.stderr)
        self.assertFalse(fixture.observation.exists())


if __name__ == "__main__":
    unittest.main()
