#!/usr/bin/env python3
"""AUH-only hostile closure for immutable SAIC formal-v2-r2 release inputs.

This suite is deliberately independent of the production submit path.  It may
be copied into a private AUH staging directory, but it must never be executed
on a workstation because several assertions query the authoritative Slurm
accounting service and the AUH-pinned static ffmpeg/Bash binaries.  The three
``__R2_FORMAL_*__`` hashes below are deliberately unresolved in this template:
the suite fails closed until one terminal retained-FD WORLD8 admission has
materialized the final gate, wrapper, and submitter bytes.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
import types
import unittest


PLACEHOLDER = re.compile(r"__R2_[A-Z0-9_]+__")
EXPECTED = {
    "materializer.py": "5585abb927206d0813caca4cec8dc10b846fe0ace704538bb12e0ad5cffe8b97",
    "base.sbatch": "12c1b2baaecfd479f65f9b5dbf0dbae17cd87196767e93c254fe2cffc895f29d",
    "effective.sbatch": "4227b4a00b7b2dea786457baad56b4fdcb4b476929e9619cb533a353b369f9f0",
    "gate.py": "877186f668f3ba89b9d887e81fbfa32a2d15b40f0e8b5f9c47b159bf88ad4151",
    "wrapper.sbatch": "4d5572f0c2da3efe84b87f5bf20db53facea8a853de04404388e8dd65b373f5d",
    "submitter.py": "2904b35bccf981f09ffd90b100bdb1c28bdbbd3ddc7130b1c2afd514767b4a0b",
    "guard.py": "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965",
    "probe_validator.py": (
        "3b5e78a0815fbfdee0404293ad800c640133deacc6b1bfafec12a02ca40ef85b"
    ),
}


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_private_dir(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    path.chmod(0o700)


def create_plain(path: Path, payload: bytes, mode: int = 0o444) -> Path:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        offset = 0
        while offset < len(payload):
            wrote = os.write(descriptor, payload[offset:])
            if wrote <= 0:
                raise AssertionError("short fixture write")
            offset += wrote
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    return path


class FormalV2HostileAUH(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        unresolved = {
            name: value for name, value in EXPECTED.items()
            if PLACEHOLDER.fullmatch(value) is not None
        }
        if unresolved:
            raise RuntimeError(
                "formal-r2 hostile template pins remain unresolved: "
                + ",".join(sorted(unresolved))
            )
        stage_value = os.environ.get("SAIC_FV2_HOSTILE_STAGE", "")
        if not stage_value:
            raise RuntimeError("SAIC_FV2_HOSTILE_STAGE is required")
        cls.stage = Path(stage_value)
        if not cls.stage.is_absolute() or cls.stage.resolve(strict=True) != cls.stage:
            raise RuntimeError("hostile stage is not canonical")
        cls.paths = {name: cls.stage / name for name in EXPECTED}
        for name, expected in EXPECTED.items():
            path = cls.paths[name]
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o444
                or sha_file(path) != expected
            ):
                raise RuntimeError(f"hostile input differs: {name}")
        cls.materializer = load(cls.paths["materializer.py"], "fv2_materializer_hostile")
        cls.gate = load(cls.paths["gate.py"], "fv2_gate_hostile")
        cls.submitter = load(cls.paths["submitter.py"], "fv2_submitter_hostile")
        cls.gate.ensure_r2_release_pins_resolved()
        cls.guard = cls.gate.load_guard(
            cls.paths["guard.py"], EXPECTED["guard.py"]
        )
        cls.canary = cls.gate.validate_canary(
            cls.guard,
            Path(cls.gate.CANARY_RECEIPT_PATH),
            Path(cls.gate.CANARY_SUBMISSION_PATH),
        )
        cls.probe_validator = cls.gate.load_probe_validator(
            cls.paths["probe_validator.py"], EXPECTED["probe_validator.py"]
        )
        cls.probe_binding = cls.gate.validate_compute_bash_probe_admission(
            cls.probe_validator,
            Path(cls.gate.COMPUTE_BASH_PROBE_ADMISSION_PATH),
        )
        cls.retained_canary = cls.gate.validate_retained_fd_canary(
            cls.guard,
            Path(cls.gate.RETAINED_FD_CANARY_ADMISSION_PATH),
            cls.probe_binding,
        )
        cls.compute_bash = cls.gate.validate_compute_bash()

    def test_01_exact_hashes_syntax_and_materialization(self) -> None:
        for name, expected in EXPECTED.items():
            self.assertEqual(sha_file(self.paths[name]), expected, name)
        for name in ("base.sbatch", "effective.sbatch", "wrapper.sbatch"):
            checked = subprocess.run(
                ["/usr/bin/bash", "-n", str(self.paths[name])],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
            self.assertEqual(checked.returncode, 0, (name, checked.stderr))
        base = self.paths["base.sbatch"].read_bytes()
        effective = self.paths["effective.sbatch"].read_bytes()
        derived = self.materializer.transform(base)
        self.assertEqual(derived, effective)
        self.assertEqual(sha_bytes(derived), EXPECTED["effective.sbatch"])
        self.assertNotIn(
            b'rendezvous_guard="${method_root}/saic_t2v_rendezvous_guard_v1.py"',
            effective,
        )
        self.assertNotIn(b'! -L "${rendezvous_guard}"', effective)
        self.assertEqual(
            effective.count(b'rendezvous_guard="${external_rendezvous_guard}"'), 1
        )
        self.assertEqual(
            effective.count(b"retained rendezvous guard bytes differ"), 1
        )
        self.assertEqual(
            effective.count(b'"saic-t2v-topup-rendezvous-dynamic-plan-v2"'), 1
        )
        gate_source = self.paths["gate.py"].read_text(encoding="ascii")
        wrapper_source = self.paths["wrapper.sbatch"].read_text(encoding="ascii")
        submitter_source = self.paths["submitter.py"].read_text(encoding="ascii")
        for anchor in (
            "validate_compute_bash_probe_admission",
            "validate_retained_fd_canary",
            "observe_retained_fd_canary_sacct",
            '"submit_line_sha256"',
            '"retained_wrapper_fd"',
            '"exact_submit_line"',
            '"three_independent_operational_proof_objects_required": True',
        ):
            self.assertIn(anchor, gate_source)
        for anchor in (
            'exec {gate_fd}<"${gate}"',
            'exec {effective_fd}<"${effective_launcher}"',
            'exec {guard_fd}<"${guard}"',
            'exec {probe_validator_fd}<"${probe_validator}"',
            '--retained-fd-canary-admission "${retained_fd_canary_admission}"',
            '--compute-bash-probe-admission "${compute_bash_probe_admission}"',
            'exec "${compute_bash}" "${effective_fd_path}"',
        ):
            self.assertIn(anchor, wrapper_source)
        for anchor in (
            "gate_module.validate_compute_bash_probe_admission",
            "gate_module.validate_retained_fd_canary",
            '"submitter_before_formal_sbatch"',
            '"hold": False',
            '"dependency": None',
        ):
            self.assertIn(anchor, submitter_source)

    def test_02_materializer_hostile_inputs_fail_closed(self) -> None:
        base = self.paths["base.sbatch"].read_bytes()
        first_anchor = self.materializer.REPLACEMENTS[0][0]
        with self.assertRaises(SystemExit):
            self.materializer.transform(base.replace(first_anchor, b"", 1))
        with self.assertRaises(SystemExit):
            self.materializer.transform(base + first_anchor)

        with tempfile.TemporaryDirectory(dir=self.stage) as tmp_value:
            tmp = Path(tmp_value)
            wrong_guard = create_plain(tmp / "wrong-guard.py", b"wrong\n")
            output = tmp / "effective.sbatch"
            with self.assertRaises(SystemExit):
                self.materializer.main(
                    [
                        "--base-launcher",
                        str(self.paths["base.sbatch"]),
                        "--guard-v2",
                        str(wrong_guard),
                        "--output",
                        str(output),
                    ]
                )
            target = tmp / "real-parent"
            target.mkdir(mode=0o700)
            linked = tmp / "linked-parent"
            linked.symlink_to(target, target_is_directory=True)
            with self.assertRaises(SystemExit):
                self.materializer.main(
                    [
                        "--base-launcher",
                        str(self.paths["base.sbatch"]),
                        "--guard-v2",
                        str(self.paths["guard.py"]),
                        "--output",
                        str(linked / "effective.sbatch"),
                    ]
                )

    def test_03_authoritative_sacct_static_ffmpeg_and_canary(self) -> None:
        observation = self.gate.observe_canary_sacct()
        self.assertEqual(observation["parsed_row"], self.gate.SACCT_PARSED_ROW)
        self.assertTrue(observation["exact_single_row"])
        self.assertEqual(observation["returncode"], 0)
        self.assertEqual(observation["stdout_sha256"], self.gate.SACCT_STDOUT_SHA256)
        self.submitter.validate_static_ffmpeg(Path(self.gate.STATIC_FFMPEG_PATH))
        self.assertEqual(self.canary["job_id"], "134393")
        self.assertEqual(
            self.canary["allocated_gpu_resource_required"], "gres/gpu:mi210=8"
        )
        self.assertEqual(self.probe_binding["slurm_job_id"], "134647")
        self.assertEqual(
            self.retained_canary["job_id"], self.gate.RETAINED_FD_CANARY_JOB_ID
        )
        self.assertEqual(
            self.retained_canary["probe_admission_binding"], self.probe_binding
        )
        self.assertEqual(self.compute_bash, self.probe_binding["compute_bash"])
        retained = self.gate.observe_retained_fd_canary_sacct(
            "hostile_after_retained_world8_terminal"
        )
        self.assertEqual(
            retained,
            self.gate.expected_retained_fd_canary_sacct_observation(
                "hostile_after_retained_world8_terminal"
            ),
        )
        self.assertTrue(retained["exact_submit_line"])
        self.assertGreaterEqual(retained["retained_wrapper_fd"], 3)

    def _run_submitter_fixture(self, root: Path) -> tuple[Path, Path, dict, dict]:
        submitter = load(self.paths["submitter.py"], "fv2_submitter_fixture")
        gate = self.gate
        guard = self.guard

        inputs = root / "inputs"
        runs = root / "runs"
        logs = root / "logs"
        bernini = root / "bernini"
        veomni = root / "veomni"
        checkpoint = root / "checkpoint"
        for directory in (inputs, runs, logs, bernini, veomni, checkpoint):
            exact_private_dir(directory)

        fixture_names = {
            "wrapper": "wrapper.sbatch",
            "base": "base.sbatch",
            "materializer": "materializer.py",
            "effective": "effective.sbatch",
            "gate": "gate.py",
            "guard": "guard.py",
            "probe_validator": "probe_validator.py",
        }
        fixture: dict[str, Path] = {}
        for label, name in fixture_names.items():
            fixture[label] = inputs / name
            shutil.copyfile(self.paths[name], fixture[label])
            fixture[label].chmod(0o444)

        source_manifest = create_plain(inputs / "source.json", b"{}\n")
        event_spec = create_plain(inputs / "events.json", b"[]\n")
        archive = create_plain(inputs / "source.tar", b"hostile-archive\n")
        checkpoint_manifest = create_plain(inputs / "checkpoint.sha256", b"fixture\n")
        output = runs / "formal-output"
        receipt = Path(str(output) + ".submission.receipt.json")
        capture = root / "fake-sbatch-capture.json"
        reader_evidence = root / "quick-start-reader.json"

        path_patches = {
            "FORMAL_ROOT": root,
            "FORMAL_RELEASE_ROOT": inputs,
            "EXPECTED_WRAPPER": fixture["wrapper"],
            "EXPECTED_BASE_LAUNCHER": fixture["base"],
            "EXPECTED_MATERIALIZER": fixture["materializer"],
            "EXPECTED_EFFECTIVE_LAUNCHER": fixture["effective"],
            "EXPECTED_GATE": fixture["gate"],
            "EXPECTED_GUARD_V2": fixture["guard"],
            "EXPECTED_PROBE_VALIDATOR": fixture["probe_validator"],
            "EXPECTED_RETAINED_FD_CANARY_ADMISSION": Path(
                gate.RETAINED_FD_CANARY_ADMISSION_PATH
            ),
            "EXPECTED_COMPUTE_BASH_PROBE_ADMISSION": Path(
                gate.COMPUTE_BASH_PROBE_ADMISSION_PATH
            ),
            "EXPECTED_COMPUTE_BASH": Path(gate.COMPUTE_BASH_PATH),
            "EXPECTED_SOURCE_ARCHIVE": archive,
            "EXPECTED_SOURCE_MANIFEST": source_manifest,
            "EXPECTED_EVENT_SPEC": event_spec,
            "EXPECTED_CHECKPOINT_MANIFEST": checkpoint_manifest,
            "EXPECTED_OUTPUT_ROOT": output,
            "EXPECTED_SUBMISSION_RECEIPT": receipt,
            "EXPECTED_SLURM_LOG_DIR": logs,
            "EXPECTED_BERNINI_ROOT": bernini,
            "EXPECTED_VEOMNI_ROOT": veomni,
            "EXPECTED_CHECKPOINT": checkpoint,
        }
        for name, value in path_patches.items():
            setattr(submitter, name, value)
        submitter.EXPECTED_WRAPPER_SHA256 = sha_file(fixture["wrapper"])
        submitter.EXPECTED_BASE_LAUNCHER_SHA256 = sha_file(fixture["base"])
        submitter.EXPECTED_MATERIALIZER_SHA256 = sha_file(fixture["materializer"])
        submitter.EXPECTED_EFFECTIVE_LAUNCHER_SHA256 = sha_file(fixture["effective"])
        submitter.EXPECTED_GATE_SHA256 = sha_file(fixture["gate"])
        submitter.EXPECTED_GUARD_V2_SHA256 = sha_file(fixture["guard"])
        submitter.EXPECTED_PROBE_VALIDATOR_SHA256 = sha_file(
            fixture["probe_validator"]
        )
        submitter.EXPECTED_RETAINED_FD_CANARY_ADMISSION_SHA256 = (
            gate.RETAINED_FD_CANARY_ADMISSION_SHA256
        )
        submitter.EXPECTED_RETAINED_FD_CANARY_ADMISSION_DIGEST = (
            gate.RETAINED_FD_CANARY_ADMISSION_DIGEST
        )
        submitter.EXPECTED_RETAINED_FD_CANARY_JOB_ID = (
            gate.RETAINED_FD_CANARY_JOB_ID
        )
        submitter.EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_SHA256 = (
            gate.COMPUTE_BASH_PROBE_ADMISSION_SHA256
        )
        submitter.EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_DIGEST = (
            gate.COMPUTE_BASH_PROBE_ADMISSION_DIGEST
        )
        submitter.EXPECTED_COMPUTE_BASH_SHA256 = gate.COMPUTE_BASH_SHA256
        submitter.EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256 = (
            gate.COMPUTE_BASH_VERSION_STDOUT_SHA256
        )
        submitter.EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE = (
            gate.COMPUTE_BASH_VERSION_FIRST_LINE
        )
        submitter.EXPECTED_SOURCE_ARCHIVE_SHA256 = sha_file(archive)
        submitter.EXPECTED_SOURCE_MANIFEST_SHA256 = sha_file(source_manifest)
        submitter.EXPECTED_EVENT_SPEC_SHA256 = sha_file(event_spec)
        submitter.EXPECTED_CHECKPOINT_MANIFEST_SHA256 = sha_file(checkpoint_manifest)
        submitter.EXPECTED_RUNTIME_SHA256 = sha_bytes(b"runtime")
        submitter.EXPECTED_ARCHIVED_GUARD_V1_SHA256 = sha_bytes(b"guard-v1")

        archived = {
            submitter.ARCHIVE_RUNTIME: b"runtime",
            submitter.ARCHIVE_GUARD_V1: b"guard-v1",
            submitter.ARCHIVE_BASE_LAUNCHER: fixture["base"].read_bytes(),
            submitter.ARCHIVE_SOURCE_MANIFEST: source_manifest.read_bytes(),
            submitter.ARCHIVE_EVENT_SPEC: event_spec.read_bytes(),
        }
        submitter.archive_payloads = lambda archive_path, revision: archived
        submitter.validate_materialization = lambda *args: None
        submitter.validate_static_ffmpeg = lambda *args: None
        submitter.exact_executable = (
            lambda value, expected_path, expected_sha, label: Path(value)
        )
        submitter.validate_canary = lambda gate_path, guard_path: (gate, self.canary)
        submitter.observe_canary_sacct = (
            lambda gate_module: gate.expected_submitter_sacct_precheck()
        )

        real_subprocess = submitter.subprocess

        def write_child_result(value: dict) -> None:
            payload = canonical(value) + b"\n"
            create_plain(reader_evidence, payload)

        def fake_run(command, **kwargs):
            descriptor = kwargs.get("pass_fds", (None,))[0]
            current = receipt.lstat()
            retained_path = Path(command[-1])
            retained = os.stat(retained_path)
            capture_value = {
                "command": list(command),
                "pass_fds": list(kwargs.get("pass_fds", ())),
                "reservation_mode": stat.S_IMODE(current.st_mode),
                "reservation_nlink": current.st_nlink,
                "retained_wrapper_sha256": sha_file(retained_path),
                "retained_wrapper_identity": [retained.st_dev, retained.st_ino],
                "logical_wrapper_identity": [
                    fixture["wrapper"].lstat().st_dev,
                    fixture["wrapper"].lstat().st_ino,
                ],
                "descriptor_matches_command": command[-1]
                == f"/proc/self/fd/{descriptor}",
            }
            create_plain(capture, canonical(capture_value) + b"\n")
            child = os.fork()
            if child == 0:
                try:
                    raw = guard.wait_ready_bytes(
                        receipt, label="quick-start formal submission receipt"
                    )
                    decoded = guard._decode_sealed(
                        raw,
                        schema_version=(
                            "saic-t2v-topup-r6-formal-v2-r2-submission-v1"
                        ),
                        exact_fields={
                            "schema_version", "status", "submission_success",
                            "job_success", "submitted_job", "request",
                            "submission_boundary", "inputs", "canary_admission",
                            "outputs", "authority", "threat_model", "receipt_digest",
                        },
                    )
                    write_child_result(
                        {
                            "status": "pass",
                            "observed_submission_status": decoded["status"],
                            "observed_mode": stat.S_IMODE(receipt.lstat().st_mode),
                        }
                    )
                    os._exit(0)
                except BaseException as error:
                    try:
                        write_child_result(
                            {"status": "fail", "error": type(error).__name__ + ":" + str(error)}
                        )
                    finally:
                        os._exit(1)
            return types.SimpleNamespace(returncode=0, stdout=b"999999\n", stderr=b"")

        submitter.subprocess = types.SimpleNamespace(
            run=fake_run,
            DEVNULL=real_subprocess.DEVNULL,
            PIPE=real_subprocess.PIPE,
        )

        arguments = [
            "--wrapper", str(fixture["wrapper"]),
            "--wrapper-sha256", submitter.EXPECTED_WRAPPER_SHA256,
            "--base-launcher", str(fixture["base"]),
            "--base-launcher-sha256", submitter.EXPECTED_BASE_LAUNCHER_SHA256,
            "--materializer", str(fixture["materializer"]),
            "--materializer-sha256", submitter.EXPECTED_MATERIALIZER_SHA256,
            "--effective-launcher", str(fixture["effective"]),
            "--effective-launcher-sha256", submitter.EXPECTED_EFFECTIVE_LAUNCHER_SHA256,
            "--gate", str(fixture["gate"]),
            "--gate-sha256", submitter.EXPECTED_GATE_SHA256,
            "--rendezvous-guard", str(fixture["guard"]),
            "--rendezvous-guard-sha256", submitter.EXPECTED_GUARD_V2_SHA256,
            "--probe-validator", str(fixture["probe_validator"]),
            "--probe-validator-sha256",
            submitter.EXPECTED_PROBE_VALIDATOR_SHA256,
            "--retained-fd-canary-admission",
            str(submitter.EXPECTED_RETAINED_FD_CANARY_ADMISSION),
            "--compute-bash-probe-admission",
            str(submitter.EXPECTED_COMPUTE_BASH_PROBE_ADMISSION),
            "--compute-bash-probe-admission-sha256",
            submitter.EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_SHA256,
            "--compute-bash-probe-admission-digest",
            submitter.EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_DIGEST,
            "--compute-bash", str(submitter.EXPECTED_COMPUTE_BASH),
            "--compute-bash-sha256", submitter.EXPECTED_COMPUTE_BASH_SHA256,
            "--compute-bash-version-stdout-sha256",
            submitter.EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256,
            "--compute-bash-version-first-line",
            submitter.EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE,
            "--source-archive", str(archive),
            "--source-archive-sha256", submitter.EXPECTED_SOURCE_ARCHIVE_SHA256,
            "--source-revision", submitter.EXPECTED_SOURCE_REVISION,
            "--source-manifest", str(source_manifest),
            "--source-manifest-sha256", submitter.EXPECTED_SOURCE_MANIFEST_SHA256,
            "--event-spec", str(event_spec),
            "--event-spec-sha256", submitter.EXPECTED_EVENT_SPEC_SHA256,
            "--checkpoint-manifest", str(checkpoint_manifest),
            "--checkpoint-manifest-sha256", submitter.EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "--python", str(submitter.EXPECTED_PYTHON),
            "--python-sha256", submitter.EXPECTED_PYTHON_SHA256,
            "--static-ffmpeg", str(submitter.EXPECTED_STATIC_FFMPEG),
            "--static-ffmpeg-sha256", submitter.EXPECTED_STATIC_FFMPEG_SHA256,
            "--bernini-root", str(bernini),
            "--veomni-root", str(veomni),
            "--checkpoint", str(checkpoint),
            "--canary-receipt", str(submitter.EXPECTED_CANARY_RECEIPT),
            "--canary-submission-receipt", str(submitter.EXPECTED_CANARY_SUBMISSION_RECEIPT),
            "--output-root", str(output),
            "--receipt", str(receipt),
            "--slurm-log-dir", str(logs),
        ]
        child = os.fork()
        if child == 0:
            submitter.main(arguments)
            os._exit(97)
        waited, status_value = os.waitpid(child, 0)
        self.assertEqual(waited, child)
        self.assertTrue(os.WIFEXITED(status_value), status_value)
        self.assertEqual(os.WEXITSTATUS(status_value), 0, status_value)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not reader_evidence.exists():
            time.sleep(0.02)
        self.assertTrue(reader_evidence.exists(), "quick-start reader did not finish")
        receipt_value = json.loads(receipt.read_text(encoding="ascii"))
        capture_value = json.loads(capture.read_text(encoding="ascii"))
        reader_value = json.loads(reader_evidence.read_text(encoding="ascii"))
        return receipt, reader_evidence, receipt_value, {
            "capture": capture_value,
            "reader": reader_value,
            "paths": path_patches,
        }

    def test_04_submitter_fast_start_receipt_and_exact_nested_gate_schema(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.stage) as tmp_value:
            root = Path(tmp_value)
            receipt, _, value, evidence = self._run_submitter_fixture(root)
            receipt_info = receipt.lstat()
            self.assertEqual(stat.S_IMODE(receipt_info.st_mode), 0o444)
            self.assertEqual(receipt_info.st_nlink, 1)
            unsigned = dict(value)
            claimed = unsigned.pop("receipt_digest")
            self.assertEqual(claimed, sha_bytes(canonical(unsigned)))
            self.assertEqual(evidence["capture"]["reservation_mode"], 0o600)
            self.assertEqual(evidence["capture"]["reservation_nlink"], 1)
            self.assertTrue(evidence["capture"]["descriptor_matches_command"])
            self.assertEqual(
                evidence["capture"]["retained_wrapper_sha256"],
                EXPECTED["wrapper.sbatch"],
            )
            self.assertEqual(evidence["reader"]["status"], "pass")
            self.assertEqual(evidence["reader"]["observed_submission_status"], "submitted")
            self.assertEqual(evidence["reader"]["observed_mode"], 0o444)
            self.assertEqual(value["request"]["hold"], False)
            self.assertIsNone(value["request"]["dependency"])
            self.assertEqual(
                value["submission_boundary"]["exact_job_export_names"],
                self.gate.EXPORT_NAMES,
            )
            self.assertEqual(len(self.gate.EXPORT_NAMES), len(set(self.gate.EXPORT_NAMES)))
            self.assertEqual(
                value["canary_admission"]["compute_bash_probe_admission"],
                self.probe_binding,
            )
            retained = value["canary_admission"]["retained_fd_world8"]
            self.assertEqual(
                retained["probe_admission_binding"], self.probe_binding
            )
            self.assertTrue(
                retained["submitter_sacct_observation"]["exact_submit_line"]
            )
            self.assertGreaterEqual(
                retained["submitter_sacct_observation"]["retained_wrapper_fd"],
                3,
            )
            for field in (
                "submit_line_sha256", "retained_wrapper_fd", "exact_submit_line",
            ):
                self.assertEqual(
                    retained["submitter_sacct_observation"][field],
                    retained["external_postflight_sacct_observation"][field],
                )

            inputs = value["inputs"]
            outputs = value["outputs"]
            gate = self.gate
            patched = {
                "FORMAL_WRAPPER": inputs["wrapper"],
                "FORMAL_BASE_LAUNCHER": inputs["base_launcher"],
                "FORMAL_BASE_LAUNCHER_SHA256": inputs["base_launcher_sha256"],
                "FORMAL_MATERIALIZER": inputs["materializer"],
                "FORMAL_MATERIALIZER_SHA256": inputs["materializer_sha256"],
                "FORMAL_EFFECTIVE_LAUNCHER": inputs["effective_launcher"],
                "FORMAL_EFFECTIVE_LAUNCHER_SHA256": inputs["effective_launcher_sha256"],
                "FORMAL_GATE": inputs["gate"],
                "FORMAL_GUARD_V2": inputs["rendezvous_guard"],
                "FORMAL_PROBE_VALIDATOR": inputs["probe_validator"],
                "FORMAL_SOURCE_ARCHIVE": inputs["source_archive"],
                "FORMAL_SOURCE_ARCHIVE_SHA256": inputs["source_archive_sha256"],
                "FORMAL_SOURCE_REVISION": inputs["source_revision"],
                "FORMAL_SOURCE_MANIFEST": inputs["source_manifest"],
                "FORMAL_SOURCE_MANIFEST_SHA256": inputs["source_manifest_sha256"],
                "FORMAL_EVENT_SPEC": inputs["event_spec"],
                "FORMAL_EVENT_SPEC_SHA256": inputs["event_spec_sha256"],
                "FORMAL_CHECKPOINT_MANIFEST": inputs["checkpoint_manifest"],
                "FORMAL_CHECKPOINT_MANIFEST_SHA256": inputs["checkpoint_manifest_sha256"],
                "FORMAL_PYTHON": inputs["python"],
                "FORMAL_PYTHON_SHA256": inputs["python_sha256"],
                "STATIC_FFMPEG_PATH": inputs["static_ffmpeg"],
                "STATIC_FFMPEG_SHA256": inputs["static_ffmpeg_sha256"],
                "STATIC_FFMPEG_VERSION_STDOUT_SHA256": inputs["static_ffmpeg_version_stdout_sha256"],
                "STATIC_FFMPEG_VERSION_FIRST_LINE": inputs["static_ffmpeg_version_first_line"],
                "FORMAL_BERNINI_ROOT": inputs["bernini_root"],
                "FORMAL_VEOMNI_ROOT": inputs["veomni_root"],
                "FORMAL_CHECKPOINT": inputs["checkpoint"],
                "RUNTIME_SHA256": inputs["generation_runtime_sha256"],
                "ARCHIVED_GUARD_V1_SHA256": inputs[
                    "archived_rendezvous_guard_v1_sha256"
                ],
            }
            original_globals = {name: getattr(gate, name) for name in patched}
            original_environ = dict(os.environ)
            try:
                for name, patched_value in patched.items():
                    setattr(gate, name, patched_value)
                export_values = {
                    "SAIC_T2V_FV2_BASE_LAUNCHER": inputs["base_launcher"],
                    "SAIC_T2V_FV2_BASE_LAUNCHER_SHA256": inputs["base_launcher_sha256"],
                    "SAIC_T2V_FV2_MATERIALIZER": inputs["materializer"],
                    "SAIC_T2V_FV2_MATERIALIZER_SHA256": inputs["materializer_sha256"],
                    "SAIC_T2V_FV2_EFFECTIVE_LAUNCHER": inputs["effective_launcher"],
                    "SAIC_T2V_FV2_EFFECTIVE_LAUNCHER_SHA256": inputs["effective_launcher_sha256"],
                    "SAIC_T2V_FV2_WRAPPER": inputs["wrapper"],
                    "SAIC_T2V_FV2_WRAPPER_SHA256": inputs["wrapper_sha256"],
                    "SAIC_T2V_FV2_GATE": inputs["gate"],
                    "SAIC_T2V_FV2_GATE_SHA256": inputs["gate_sha256"],
                    "SAIC_T2V_V4_EXTERNAL_RENDEZVOUS_GUARD": inputs["rendezvous_guard"],
                    "SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256": inputs["rendezvous_guard_sha256"],
                    "SAIC_T2V_FV2_CANARY_RECEIPT": str(
                        self.gate.CANARY_RECEIPT_PATH
                    ),
                    "SAIC_T2V_FV2_CANARY_SUBMISSION_RECEIPT": str(
                        self.gate.CANARY_SUBMISSION_PATH
                    ),
                    "SAIC_T2V_FV2_RETAINED_FD_CANARY_ADMISSION": inputs[
                        "retained_fd_canary_admission"
                    ],
                    "SAIC_T2V_FV2_PROBE_VALIDATOR": inputs["probe_validator"],
                    "SAIC_T2V_FV2_PROBE_VALIDATOR_SHA256": inputs[
                        "probe_validator_sha256"
                    ],
                    "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION": inputs[
                        "compute_bash_probe_admission"
                    ],
                    "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_SHA256": inputs[
                        "compute_bash_probe_admission_sha256"
                    ],
                    "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_DIGEST": inputs[
                        "compute_bash_probe_admission_digest"
                    ],
                    "SAIC_T2V_FV2_OWN_SUBMISSION_RECEIPT": outputs[
                        "submission_receipt"
                    ],
                    "SAIC_T2V_V3_SOURCE_ARCHIVE": inputs["source_archive"],
                    "SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256": inputs["source_archive_sha256"],
                    "SAIC_T2V_V3_SOURCE_REVISION": inputs["source_revision"],
                    "SAIC_T2V_V3_SOURCE_MANIFEST": inputs["source_manifest"],
                    "SAIC_T2V_V3_SOURCE_MANIFEST_SHA256": inputs["source_manifest_sha256"],
                    "SAIC_T2V_V3_EVENT_SPEC": inputs["event_spec"],
                    "SAIC_T2V_V3_EVENT_SPEC_SHA256": inputs["event_spec_sha256"],
                    "BERNINI_OFFICIAL_ROOT": inputs["bernini_root"],
                    "BERNINI_VEOMNI_ROOT": inputs["veomni_root"],
                    "BERNINI_ACTION_CHECKPOINT": inputs["checkpoint"],
                    "BERNINI_CHECKPOINT_CONTENT_MANIFEST": inputs["checkpoint_manifest"],
                    "SAIC_T2V_FV2_CHECKPOINT_MANIFEST_SHA256": inputs["checkpoint_manifest_sha256"],
                    "SAIC_T2V_V3_OUTPUT_ROOT": outputs["output_root"],
                    "SAIC_T2V_V3_PYTHON_BIN": inputs["python"],
                    "SAIC_T2V_FV2_PYTHON_SHA256": inputs["python_sha256"],
                    "SAIC_T2V_V3_STATIC_FFMPEG": inputs["static_ffmpeg"],
                    "SAIC_T2V_FV2_STATIC_FFMPEG_SHA256": inputs["static_ffmpeg_sha256"],
                    "SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_STDOUT_SHA256": inputs["static_ffmpeg_version_stdout_sha256"],
                    "SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_FIRST_LINE": inputs["static_ffmpeg_version_first_line"],
                    "SAIC_T2V_FV2_COMPUTE_BASH": inputs["compute_bash"],
                    "SAIC_T2V_FV2_COMPUTE_BASH_SHA256": inputs[
                        "compute_bash_sha256"
                    ],
                    "SAIC_T2V_FV2_COMPUTE_BASH_VERSION_STDOUT_SHA256": inputs[
                        "compute_bash_version_stdout_sha256"
                    ],
                    "SAIC_T2V_FV2_SLURM_LOG_DIR": outputs["slurm_log_dir"],
                }
                os.environ.clear()
                os.environ.update(export_values)
                accepted = gate.validate_own_submission(
                    self.guard,
                    receipt,
                    job_id="999999",
                    canary=self.canary,
                    probe_binding=self.probe_binding,
                    retained_canary=self.retained_canary,
                )
                self.assertEqual(accepted, value)

                tampered = copy.deepcopy(value)
                hostile_spec = str(root / "coordinated-hostile-event-spec.json")
                tampered["inputs"]["event_spec"] = hostile_spec
                tampered["outputs"]["submission_receipt"] = str(root / "tampered.json")
                unsigned = dict(tampered)
                unsigned.pop("receipt_digest")
                tampered["receipt_digest"] = sha_bytes(canonical(unsigned))
                tampered_path = create_plain(
                    root / "tampered.json", canonical(tampered) + b"\n"
                )
                os.environ["SAIC_T2V_V3_EVENT_SPEC"] = hostile_spec
                with self.assertRaisesRegex(
                    RuntimeError, "formal hard-pinned input environment differs"
                ):
                    gate.validate_own_submission(
                        self.guard,
                        tampered_path,
                        job_id="999999",
                        canary=self.canary,
                        probe_binding=self.probe_binding,
                        retained_canary=self.retained_canary,
                    )
            finally:
                for name, original in original_globals.items():
                    setattr(gate, name, original)
                os.environ.clear()
                os.environ.update(original_environ)

    def test_05_retained_fd_rejects_logical_leaf_swap(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.stage) as tmp_value:
            tmp = Path(tmp_value)
            logical = tmp / "wrapper.sbatch"
            shutil.copyfile(self.paths["wrapper.sbatch"], logical)
            logical.chmod(0o444)
            descriptor = os.open(logical, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                retained_before = os.fstat(descriptor)
                old = tmp / "wrapper-old.sbatch"
                logical.rename(old)
                shutil.copyfile(old, logical)
                logical.chmod(0o444)
                retained_after = os.fstat(descriptor)
                leaf = logical.lstat()
                self.assertEqual(
                    (retained_before.st_dev, retained_before.st_ino),
                    (retained_after.st_dev, retained_after.st_ino),
                )
                self.assertNotEqual(
                    (leaf.st_dev, leaf.st_ino),
                    (retained_after.st_dev, retained_after.st_ino),
                )
                self.assertEqual(
                    sha_file(Path(f"/proc/{os.getpid()}/fd/{descriptor}")),
                    EXPECTED["wrapper.sbatch"],
                )
            finally:
                os.close(descriptor)
        wrapper_source = self.paths["wrapper.sbatch"].read_text(encoding="utf-8")
        self.assertIn(
            '"$(stat -Lc \'%d:%i\' -- "${logical}")" == "$(stat -Lc \'%d:%i\' -- "${fd_path}")"',
            wrapper_source,
        )

    def test_06_forbidden_directory_symlink_rejected(self) -> None:
        terminal_original = json.loads(
            Path(self.gate.CANARY_RECEIPT_PATH).read_text(encoding="ascii")
        )
        submission_original = json.loads(
            Path(self.gate.CANARY_SUBMISSION_PATH).read_text(encoding="ascii")
        )
        with tempfile.TemporaryDirectory(dir=self.stage) as tmp_value:
            tmp = Path(tmp_value)
            job = tmp / "job-134393"
            job.mkdir(mode=0o700)
            terminal = job / "canary-receipt.json"
            submission = tmp / "submission-receipt.json"
            terminal_value = copy.deepcopy(terminal_original)
            submission_value = copy.deepcopy(submission_original)
            terminal_value["submission_receipt_path"] = str(submission)
            submission_value["outputs"]["job_output_root"] = str(job)
            submission_value["outputs"]["submission_receipt"] = str(submission)
            create_plain(terminal, canonical(terminal_value) + b"\n")
            create_plain(submission, canonical(submission_value) + b"\n")
            target = tmp / "forbidden-target"
            target.mkdir(mode=0o700)
            (job / "forbidden-attempts").symlink_to(target, target_is_directory=True)

            original_decode = self.gate.decode_fixed

            def staged_decode(module, path, **kwargs):
                if path == terminal:
                    return b"", terminal_value
                if path == submission:
                    return b"", submission_value
                raise AssertionError(path)

            self.gate.decode_fixed = staged_decode
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "pinned canary failure/scientific closure differs"
                ):
                    self.gate.validate_canary(
                        self.guard,
                        terminal,
                        submission,
                    )
            finally:
                self.gate.decode_fixed = original_decode


if __name__ == "__main__":
    unittest.main(verbosity=2)
