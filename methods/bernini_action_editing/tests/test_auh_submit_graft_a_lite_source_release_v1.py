from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SUBMIT = METHOD_ROOT / "scripts" / "auh_submit_graft_a_lite_source_release_v1.sh"

EXPORT_NAMES = (
    "GRAFT_A_LITE_SOURCE_ARCHIVE",
    "GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256",
    "GRAFT_A_LITE_PYTHON_BIN",
    "GRAFT_A_LITE_PYTHON_SHA256",
    "GRAFT_A_LITE_LAUNCHER_SOURCE",
    "GRAFT_A_LITE_LAUNCHER_SHA256",
    "GRAFT_A_LITE_OUTPUT_STEM",
)
SCHEDULER_ARGUMENTS = (
    "--parsable",
    f"--export={','.join(EXPORT_NAMES)}",
    "--partition=faculty",
    "--qos=bgqos",
    "--nodes=1",
    "--ntasks=1",
    "--cpus-per-task=8",
    "--mem=32G",
    "--gres=gpu:mi210:1",
    "--time=00:20:00",
    "--job-name=graft-a-lite-c4",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _object_sha(value: object) -> str:
    return _sha(_canonical(value))


def _fake_sbatch_source(
    *,
    behavior: str,
    observation: Path,
    launcher: Path,
    sbatch_path: Path,
    output_parent: Path,
) -> bytes:
    python = Path(sys.executable).resolve(strict=True)
    source = f'''#!{python}
import json
import os
from pathlib import Path
import stat
import sys

behavior = {behavior!r}
observation = Path({str(observation)!r})
launcher_path = Path({str(launcher)!r})
sbatch_path = Path({str(sbatch_path)!r})
output_parent = Path({str(output_parent)!r})

launcher_fd_path = Path(sys.argv[-1])
record = {{
    "argv": sys.argv[1:],
    "environment": dict(os.environ),
    "launcher_fd_bytes_hex": launcher_fd_path.read_bytes().hex(),
}}
observation.write_text(
    json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
    encoding="ascii",
)

if behavior == "bad_exit":
    print("fake failure", file=sys.stderr)
    raise SystemExit(7)
if behavior == "replace_launcher_path":
    launcher_path.unlink()
    launcher_path.write_bytes(b"attacker launcher")
    launcher_path.chmod(0o444)
elif behavior == "replace_sbatch_path":
    sbatch_path.unlink()
    sbatch_path.write_text("#!/bin/sh\\nexit 91\\n", encoding="ascii")
    sbatch_path.chmod(0o555)
elif behavior == "replace_output_parent":
    displaced = output_parent.with_name(output_parent.name + ".displaced")
    output_parent.rename(displaced)
    output_parent.mkdir()

if behavior == "bad_jobid":
    print("not-a-job-id")
else:
    print("765432;fakecluster")
'''
    return source.encode("ascii")


class _SubmitFixture:
    def __init__(
        self,
        root: Path,
        *,
        behavior: str = "success",
        portable_fake_transport: bool = True,
    ) -> None:
        self.root = root.resolve()
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.observation = self.root / "fake-sbatch-observation.json"

        self.archive = self.inputs / "builder-source.tar"
        self.archive.write_bytes(b"fake sealed builder archive")
        self.archive.chmod(0o444)

        self.launcher = self.inputs / "launcher.sbatch"
        self.launcher_bytes = b"#!/bin/bash -p\n# frozen fake launcher\n"
        self.launcher.write_bytes(self.launcher_bytes)
        self.launcher.chmod(0o444)

        self.output_stem = self.outputs / "canary4"
        self.fake_sbatch = self.inputs / "sbatch"
        self.fake_sbatch.write_bytes(
            _fake_sbatch_source(
                behavior=behavior,
                observation=self.observation,
                launcher=self.launcher,
                sbatch_path=self.fake_sbatch,
                output_parent=self.outputs,
            )
        )
        self.fake_sbatch.chmod(0o555)

        source = SUBMIT.read_text(encoding="utf-8")
        replacements = {
            "readonly required_sbatch_path=/usr/bin/sbatch": (
                f"readonly required_sbatch_path={self.fake_sbatch}"
            ),
        }
        if portable_fake_transport:
            replacements.update(
                {
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
            )
        for old, new in replacements.items():
            if old not in source:
                raise AssertionError(f"submit fixture pin is absent: {old}")
            source = source.replace(old, new)
        self.wrapper = self.inputs / "submit.sh"
        self.wrapper.write_text(source, encoding="utf-8")
        self.wrapper.chmod(0o444)

    def environment(self, **extra: str) -> dict[str, str]:
        python = Path(sys.executable).resolve(strict=True)
        values = {
            "GRAFT_A_LITE_SOURCE_ARCHIVE": str(self.archive),
            "GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256": _sha(self.archive.read_bytes()),
            "GRAFT_A_LITE_PYTHON_BIN": str(python),
            "GRAFT_A_LITE_PYTHON_SHA256": _sha(python.read_bytes()),
            "GRAFT_A_LITE_LAUNCHER_SOURCE": str(self.launcher),
            "GRAFT_A_LITE_LAUNCHER_SHA256": _sha(self.launcher.read_bytes()),
            "GRAFT_A_LITE_OUTPUT_STEM": str(self.output_stem),
        }
        values.update(extra)
        return values

    def run(
        self,
        *arguments: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", "-p", str(self.wrapper), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.environment() if environment is None else environment,
            timeout=30,
        )

    @property
    def receipt(self) -> Path:
        return self.outputs / "canary4.submission.receipt.json"


class AuhSubmitGraftALiteSourceReleaseTests(unittest.TestCase):
    def fixture(
        self,
        *,
        behavior: str = "success",
        portable_fake_transport: bool = True,
    ) -> tuple[tempfile.TemporaryDirectory[str], _SubmitFixture]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return temporary, _SubmitFixture(
            Path(temporary.name),
            behavior=behavior,
            portable_fake_transport=portable_fake_transport,
        )

    def test_bash_syntax_and_production_boundary_are_exact(self) -> None:
        completed = subprocess.run(
            ["/bin/bash", "-n", str(SUBMIT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        source = SUBMIT.read_text(encoding="utf-8")
        self.assertTrue(source.startswith("#!/bin/bash -p\n"))
        self.assertIn('case "$-" in', source)
        self.assertIn("readonly required_sbatch_path=/usr/bin/sbatch", source)
        self.assertIn("readonly required_fd_root=/proc/self/fd", source)
        self.assertIn("readonly required_fd_stat_identity=true", source)
        self.assertIn("readonly required_execute_sbatch_from_fd=true", source)
        self.assertIn("/usr/bin/env -i", source)
        self.assertIn('"${GRAFT_A_LITE_PYTHON_BIN}" -I -S -B -', source)
        self.assertIn('[[ "$#" -eq 0 ]]', source)
        self.assertNotIn("--export=ALL", source)
        self.assertEqual(source.count('"${GRAFT_A_LITE_PYTHON_BIN}" -I -S -B -'), 1)

    def test_success_uses_exact_argv_environment_and_canonical_receipt(self) -> None:
        _, fixture = self.fixture()
        completed = fixture.run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        observed = json.loads(fixture.observation.read_bytes())
        self.assertEqual(observed["argv"][:-1], list(SCHEDULER_ARGUMENTS))
        self.assertTrue(observed["argv"][-1].startswith("/dev/fd/"))
        self.assertEqual(
            bytes.fromhex(observed["launcher_fd_bytes_hex"]), fixture.launcher_bytes
        )
        expected_environment = {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            **fixture.environment(),
        }
        self.assertEqual(observed["environment"], expected_environment)

        self.assertTrue(fixture.receipt.is_file())
        self.assertEqual(stat.S_IMODE(fixture.receipt.stat().st_mode), 0o444)
        raw = fixture.receipt.read_bytes()
        receipt = json.loads(raw)
        self.assertEqual(_canonical(receipt) + b"\n", raw)
        unsigned = dict(receipt)
        declared = unsigned.pop("receipt_digest")
        self.assertEqual(declared, _object_sha(unsigned))
        self.assertTrue(receipt["submission_success"])
        self.assertIsNone(receipt["job_success"])
        self.assertFalse(receipt["job_terminal_state_observed"])
        self.assertEqual(receipt["submitted_job"]["job_id"], "765432")
        self.assertEqual(receipt["submitted_job"]["scheduler_cluster"], "fakecluster")
        self.assertEqual(receipt["export_contract"]["names"], list(EXPORT_NAMES))
        self.assertFalse(receipt["export_contract"]["contains_all"])
        self.assertTrue(all(value is False for value in receipt["authority"].values()))
        self.assertTrue(
            receipt["failure_semantics"]["submission_success_is_not_job_success"]
        )

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "production /proc/self/fd execution transport is Linux-only",
    )
    def test_linux_fake_uses_production_fd_execution_transport(self) -> None:
        _, fixture = self.fixture(portable_fake_transport=False)
        completed = fixture.run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(fixture.receipt.read_bytes())
        boundary = receipt["submission_boundary"]
        self.assertTrue(boundary["sbatch"]["executed_from_retained_fd"])
        self.assertTrue(
            boundary["sbatch"]["fd_transport_inode_identity_verified"]
        )
        self.assertTrue(
            boundary["launcher"]["fd_transport_inode_identity_verified"]
        )

    def test_path_sbatch_ld_bash_and_extra_scheduler_poison_are_purged(self) -> None:
        temporary, fixture = self.fixture()
        bash_marker = Path(temporary.name) / "bash-env-ran"
        exec_marker = Path(temporary.name) / "imported-exec-ran"
        bash_env = Path(temporary.name) / "bash-env"
        bash_env.write_text(f"/usr/bin/touch {str(bash_marker)!r}\n", encoding="utf-8")
        environment = fixture.environment(
            PATH="/attacker/path",
            SBATCH_EXPORT="ALL",
            SBATCH_PARTITION="attacker",
            SBATCH_GRES="gpu:attacker:99",
            LD_LIBRARY_PATH="/attacker/lib",
            LD_FAKE_POISON="present",
            BASH_ENV=str(bash_env),
            **{
                "BASH_FUNC_exec%%": (
                    f"() {{ /usr/bin/touch {str(exec_marker)!r}; "
                    'builtin exec "$@"; }'
                )
            },
        )
        completed = fixture.run(environment=environment)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        observed = json.loads(fixture.observation.read_bytes())
        self.assertFalse(bash_marker.exists())
        self.assertFalse(exec_marker.exists())
        self.assertEqual(set(observed["environment"]), {"PATH", "LC_ALL", "LANG", *EXPORT_NAMES})
        self.assertEqual(observed["argv"][:-1], list(SCHEDULER_ARGUMENTS))

    def test_extra_graft_interface_variable_fails_before_sbatch(self) -> None:
        _, fixture = self.fixture()
        environment = fixture.environment(GRAFT_A_LITE_ATTACKER_OVERRIDE="1")
        completed = fixture.run(environment=environment)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unexpected GRAFT interface variable", completed.stderr)
        self.assertFalse(fixture.observation.exists())
        self.assertFalse(fixture.receipt.exists())

    def test_arbitrary_argument_and_non_privileged_shell_fail_closed(self) -> None:
        _, fixture = self.fixture()
        completed = fixture.run("--export=ALL")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("arbitrary arguments are forbidden", completed.stderr)
        self.assertFalse(fixture.observation.exists())
        direct = subprocess.run(
            ["/bin/bash", str(fixture.wrapper)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=fixture.environment(),
            timeout=10,
        )
        self.assertNotEqual(direct.returncode, 0)
        self.assertIn("Bash privileged mode is required", direct.stderr)

    def test_launcher_path_replacement_is_detected_after_fd_submission(self) -> None:
        _, fixture = self.fixture(behavior="replace_launcher_path")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("launcher path or fd identity changed", completed.stderr)
        observed = json.loads(fixture.observation.read_bytes())
        self.assertEqual(
            bytes.fromhex(observed["launcher_fd_bytes_hex"]), fixture.launcher_bytes
        )
        self.assertEqual(fixture.launcher.read_bytes(), b"attacker launcher")
        self.assertFalse(fixture.receipt.exists())

    def test_sbatch_path_replacement_is_detected_after_fd_execution(self) -> None:
        _, fixture = self.fixture(behavior="replace_sbatch_path")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("sbatch path or fd identity changed", completed.stderr)
        self.assertTrue(fixture.observation.is_file())
        self.assertFalse(fixture.receipt.exists())

    def test_output_parent_replacement_is_detected_without_receipt(self) -> None:
        _, fixture = self.fixture(behavior="replace_output_parent")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("output parent path identity changed", completed.stderr)
        self.assertFalse(fixture.receipt.exists())
        displaced = fixture.outputs.with_name(fixture.outputs.name + ".displaced")
        self.assertFalse((displaced / fixture.receipt.name).exists())

    def test_bad_sbatch_exit_has_no_submission_receipt(self) -> None:
        _, fixture = self.fixture(behavior="bad_exit")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("sbatch failed with exit 7", completed.stderr)
        self.assertTrue(fixture.observation.is_file())
        self.assertFalse(fixture.receipt.exists())

    def test_bad_parsable_job_id_has_no_submission_receipt(self) -> None:
        _, fixture = self.fixture(behavior="bad_jobid")
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("sbatch parsable job ID differs", completed.stderr)
        self.assertTrue(fixture.observation.is_file())
        self.assertFalse(fixture.receipt.exists())

    def test_preexisting_receipt_blocks_sbatch_and_is_preserved(self) -> None:
        _, fixture = self.fixture()
        fixture.receipt.write_bytes(b"preexisting")
        fixture.receipt.chmod(0o444)
        completed = fixture.run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(fixture.receipt.read_bytes(), b"preexisting")
        self.assertFalse(fixture.observation.exists())

    def test_bad_launcher_hash_blocks_sbatch(self) -> None:
        _, fixture = self.fixture()
        environment = fixture.environment()
        environment["GRAFT_A_LITE_LAUNCHER_SHA256"] = "0" * 64
        completed = fixture.run(environment=environment)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("launcher SHA-256 differs", completed.stderr)
        self.assertFalse(fixture.observation.exists())
        self.assertFalse(fixture.receipt.exists())


if __name__ == "__main__":
    unittest.main()
