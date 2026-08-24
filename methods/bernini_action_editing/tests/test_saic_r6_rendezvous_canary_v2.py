#!/usr/bin/env python3
"""AUH-only hostile closure for the SAIC r6 rendezvous canary v2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import ModuleType
from unittest import mock
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = METHOD_ROOT / "saic_t2v_rendezvous_guard_v2.py"
LAUNCHER_PATH = (
    METHOD_ROOT / "scripts/auh_canary_saic_t2v_dynamic_rendezvous_all8_v2.sbatch"
)
SUBMITTER_PATH = METHOD_ROOT / "tools/submit_saic_r6_rendezvous_canary_v2.py"
RUNTIME_MEMBER = (
    "methods/bernini_action_editing/generate_saic_pure_t2v_event_bank_topup_v2.py"
)


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plain_copy(source: Path, target: Path, mode: int = 0o444) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.write_bytes(source.read_bytes())
    target.chmod(mode)
    return target


def write_sealed(module: ModuleType, path: Path, core: dict, mode: int = 0o444) -> dict:
    value = module.seal(core)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(module.canonical_json_bytes(value) + b"\n")
    path.chmod(mode)
    return value


def rewrite_sealed(module: ModuleType, path: Path, core: dict) -> dict:
    path.chmod(0o600)
    return write_sealed(module, path, core)


def unseal(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="ascii"))
    value.pop("receipt_digest")
    return value


def directory_identity(path: Path) -> str:
    info = path.lstat()
    return f"{info.st_dev}:{info.st_ino}"


class TerminalExit(BaseException):
    def __init__(self, code: int) -> None:
        self.code = code


class SubmitterFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.launcher = plain_copy(LAUNCHER_PATH, root / "inputs/launcher.sbatch")
        self.guard = plain_copy(GUARD_PATH, root / "inputs/guard.py")
        self.runtime_payload = b"def main(argv):\n    return 0\n"
        self.archive = root / "inputs/source.tar"
        with tarfile.open(self.archive, "w") as archive:
            staged = root / "runtime.py"
            staged.write_bytes(self.runtime_payload)
            archive.add(staged, arcname=RUNTIME_MEMBER)
        self.archive.chmod(0o444)
        self.python = Path(sys.executable).resolve(strict=True)
        self.output_parent = root / "fresh-output"
        self.output_parent.mkdir(mode=0o700)
        self.receipt = self.output_parent / "submission-receipt.json"
        self.launcher_sha = sha_file(self.launcher)
        self.guard_sha = sha_file(self.guard)
        self.archive_sha = sha_file(self.archive)
        self.python_sha = sha_file(self.python)

    def argv(self, launcher_sha: str | None = None) -> list[str]:
        return [
            "--launcher", str(self.launcher),
            "--launcher-sha256", launcher_sha or self.launcher_sha,
            "--archive", str(self.archive),
            "--archive-sha256", self.archive_sha,
            "--guard", str(self.guard),
            "--guard-sha256", self.guard_sha,
            "--python", str(self.python),
            "--python-sha256", self.python_sha,
            "--output-parent", str(self.output_parent),
            "--receipt", str(self.receipt),
        ]

    def patches(self, module: ModuleType):
        return mock.patch.multiple(
            module,
            EXPECTED_LAUNCHER_SHA256=self.launcher_sha,
            EXPECTED_GUARD_SHA256=self.guard_sha,
            EXPECTED_ARCHIVE_SHA256=self.archive_sha,
            EXPECTED_RUNTIME_SHA256=sha_bytes(self.runtime_payload),
        )


class SubmitterHostileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="saic-r6-cny2-submit-")
        self.root = Path(self.temporary.name).resolve()
        self.fixture = SubmitterFixture(self.root)
        self.module = load_module(f"submitter_{id(self)}", SUBMITTER_PATH)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def terminal_exit(code: int) -> None:
        raise TerminalExit(code)

    def test_fast_start_and_retained_fd_survive_path_replacement(self) -> None:
        original = self.fixture.launcher.read_bytes()

        def fake_sbatch(command, **kwargs):
            retained = kwargs["pass_fds"]
            self.assertEqual(len(retained), 1)
            self.assertEqual(command[-1], f"/proc/self/fd/{retained[0]}")
            os.lseek(retained[0], 0, os.SEEK_SET)
            self.assertEqual(os.read(retained[0], len(original) + 1), original)
            backup = self.fixture.launcher.with_name("launcher.original")
            self.fixture.launcher.rename(backup)
            self.fixture.launcher.write_bytes(b"#!/bin/sh\nexit 99\n")
            self.fixture.launcher.chmod(0o444)
            (self.fixture.output_parent / "job-97531").mkdir(mode=0o700)
            return subprocess.CompletedProcess(command, 0, b"97531\n", b"")

        with self.fixture.patches(self.module), mock.patch.object(
            self.module.subprocess, "run", side_effect=fake_sbatch
        ), mock.patch.object(
            self.module.os, "_exit", side_effect=self.terminal_exit
        ):
            with self.assertRaises(TerminalExit) as caught:
                self.module.main(self.fixture.argv())
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(stat.S_IMODE(self.fixture.receipt.stat().st_mode), 0o444)
        value = json.loads(self.fixture.receipt.read_text(encoding="ascii"))
        self.assertTrue(
            value["submission_boundary"]["launcher_submitted_from_retained_fd"]
        )
        self.assertEqual(value["submitted_job"]["job_id"], "97531")
        self.assertTrue((self.fixture.output_parent / "job-97531").is_dir())

    def test_sbatch_failure_retains_non_success_reservation(self) -> None:
        failed = subprocess.CompletedProcess([], 2, b"", b"scheduler failure")
        with self.fixture.patches(self.module), mock.patch.object(
            self.module.subprocess, "run", return_value=failed
        ):
            with self.assertRaises(SystemExit):
                self.module.main(self.fixture.argv())
        self.assertEqual(stat.S_IMODE(self.fixture.receipt.stat().st_mode), 0o600)
        self.assertNotIn(b'"status":"submitted"', self.fixture.receipt.read_bytes())

    def test_receipt_path_swap_after_sbatch_fails_closed(self) -> None:
        def fake_sbatch(command, **kwargs):
            moved = self.fixture.receipt.with_name("reservation.moved")
            self.fixture.receipt.rename(moved)
            self.fixture.receipt.write_bytes(b"replacement\n")
            self.fixture.receipt.chmod(0o600)
            return subprocess.CompletedProcess(command, 0, b"97532\n", b"")

        with self.fixture.patches(self.module), mock.patch.object(
            self.module.subprocess, "run", side_effect=fake_sbatch
        ):
            with self.assertRaises(SystemExit):
                self.module.main(self.fixture.argv())
        self.assertNotEqual(stat.S_IMODE(self.fixture.receipt.stat().st_mode), 0o444)

    def test_parent_swap_before_publication_fails_closed(self) -> None:
        completed = subprocess.CompletedProcess([], 0, b"97533\n", b"")
        original_fsync = self.module.fsync_directory
        calls = 0

        def swapping_fsync(path: Path) -> None:
            nonlocal calls
            original_fsync(path)
            calls += 1
            if calls == 2:
                moved = path.with_name("fresh-output-moved")
                path.rename(moved)
                path.mkdir(mode=0o700)

        with self.fixture.patches(self.module), mock.patch.object(
            self.module.subprocess, "run", return_value=completed
        ), mock.patch.object(
            self.module, "fsync_directory", side_effect=swapping_fsync
        ):
            with self.assertRaises((SystemExit, FileNotFoundError)):
                self.module.main(self.fixture.argv())
        moved_receipt = self.root / "fresh-output-moved/submission-receipt.json"
        self.assertEqual(stat.S_IMODE(moved_receipt.stat().st_mode), 0o600)

    def test_terminal_close_error_does_not_reverse_success(self) -> None:
        completed = subprocess.CompletedProcess([], 0, b"97534\n", b"")
        real_close = os.close

        def close_with_terminal_error(descriptor: int) -> None:
            try:
                observed = os.fstat(descriptor)
                target = self.fixture.receipt.lstat()
                is_terminal = (
                    (observed.st_dev, observed.st_ino)
                    == (target.st_dev, target.st_ino)
                    and stat.S_IMODE(observed.st_mode) == 0o444
                )
            except OSError:
                is_terminal = False
            real_close(descriptor)
            if is_terminal:
                raise OSError("injected post-publication close failure")

        with self.fixture.patches(self.module), mock.patch.object(
            self.module.subprocess, "run", return_value=completed
        ), mock.patch.object(
            self.module.os, "close", side_effect=close_with_terminal_error
        ), mock.patch.object(
            self.module.os, "_exit", side_effect=self.terminal_exit
        ):
            with self.assertRaises(TerminalExit) as caught:
                self.module.main(self.fixture.argv())
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(stat.S_IMODE(self.fixture.receipt.stat().st_mode), 0o444)

    def test_alternate_launcher_sha_is_rejected_before_sbatch(self) -> None:
        wrong = "0" * 64
        with mock.patch.object(self.module.subprocess, "run") as run:
            with self.assertRaises(SystemExit):
                self.module.main(self.fixture.argv(launcher_sha=wrong))
        run.assert_not_called()


class LauncherGateFixture:
    JOB = "97541"

    def __init__(self, root: Path, wait_seconds: float = 0.4) -> None:
        self.root = root
        self.launcher = plain_copy(LAUNCHER_PATH, root / "launcher.sbatch")
        guard_payload = GUARD_PATH.read_text(encoding="utf-8").replace(
            "WAIT_SECONDS = 90.0", f"WAIT_SECONDS = {wait_seconds!r}", 1
        ).encode("utf-8")
        self.guard = root / "guard-short-wait.py"
        self.guard.write_bytes(guard_payload)
        self.guard.chmod(0o444)
        self.guard_module = load_module(f"gate_guard_{id(self)}", self.guard)
        self.archive = root / "source.tar"
        self.archive.write_bytes(b"test archive bytes\n")
        self.archive.chmod(0o444)
        self.python = Path(sys.executable).resolve(strict=True)
        self.output_parent = root / "output"
        self.output_parent.mkdir(mode=0o700)
        self.receipt = self.output_parent / "submission-receipt.json"
        self.launcher_sha = sha_file(self.launcher)
        self.guard_sha = sha_file(self.guard)
        self.archive_sha = sha_file(self.archive)
        self.python_sha = sha_file(self.python)

    @staticmethod
    def expected_exports() -> list[str]:
        return [
            "SAIC_R6_CANARY_V2_SOURCE_ARCHIVE",
            "SAIC_R6_CANARY_V2_SOURCE_ARCHIVE_SHA256",
            "SAIC_R6_CANARY_V2_GUARD",
            "SAIC_R6_CANARY_V2_GUARD_SHA256",
            "SAIC_R6_CANARY_V2_PYTHON",
            "SAIC_R6_CANARY_V2_PYTHON_SHA256",
            "SAIC_R6_CANARY_V2_OUTPUT_PARENT",
            "SAIC_R6_CANARY_V2_LAUNCHER",
            "SAIC_R6_CANARY_V2_LAUNCHER_SHA256",
            "SAIC_R6_CANARY_V2_SUBMISSION_RECEIPT",
        ]

    def receipt_core(self, job: str | None = None) -> dict:
        job_value = job or self.JOB
        return {
            "schema_version": "saic-r6-rendezvous-canary-submission-v2",
            "status": "submitted",
            "submission_success": True,
            "job_success": None,
            "submitted_job": {
                "job_id": job_value,
                "cluster": None,
                "stdout_sha256": sha_bytes(f"{job_value}\n".encode("ascii")),
                "stderr_sha256": sha_bytes(b""),
            },
            "request": {
                "job_name": "saic-r6-rdzv-cny2", "partition": "faculty",
                "qos": "bgqos", "nodes": 1, "ntasks": 1,
                "cpus_per_task": 32, "memory": "64G",
                "walltime": "00:30:00",
                "gpu_resource_requested": "gpu:mi210:8",
                "world_topology": "two_concurrent_world4",
                "candidate_count": 60, "hold": False, "dependency": None,
                "scientific_generation": False,
            },
            "submission_boundary": {
                "environment_replaced": True,
                "exact_job_export_names": self.expected_exports(),
                "export_all": False,
                "reservation_created_before_sbatch": True,
                "same_inode_retained": True,
                "launcher_submitted_from_retained_fd": True,
                "reservation_device": 1,
                "reservation_inode": 2,
                "success_mode": "0444",
            },
            "inputs": {
                "launcher": str(self.launcher),
                "launcher_sha256": self.launcher_sha,
                "guard": str(self.guard),
                "guard_sha256": self.guard_sha,
                "source_archive": str(self.archive),
                "source_archive_sha256": self.archive_sha,
                "runtime_sha256": self.guard_module.EXPECTED_RUNTIME_SHA256,
                "python": str(self.python),
                "python_sha256": self.python_sha,
            },
            "outputs": {
                "output_parent": str(self.output_parent),
                "job_output_root": str(self.output_parent / f"job-{job_value}"),
                "submission_receipt": str(self.receipt),
                "fresh_before_submission": True,
            },
            "authority": {
                "scientific": False, "generation": False, "training": False,
                "publication": False, "formal_job_authorized": False,
            },
        }

    def publish_receipt(self, core: dict, mode: int = 0o444) -> None:
        write_sealed(self.guard_module, self.receipt, core, mode=mode)

    def env(self) -> dict[str, str]:
        return {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "SLURM_JOB_ID": self.JOB,
            "SLURMD_NODENAME": socket.gethostname(),
            "SLURM_TMPDIR": str(self.root / "deliberately-absent-scratch"),
            "SAIC_R6_CANARY_V2_SOURCE_ARCHIVE": str(self.archive),
            "SAIC_R6_CANARY_V2_SOURCE_ARCHIVE_SHA256": self.archive_sha,
            "SAIC_R6_CANARY_V2_GUARD": str(self.guard),
            "SAIC_R6_CANARY_V2_GUARD_SHA256": self.guard_sha,
            "SAIC_R6_CANARY_V2_PYTHON": str(self.python),
            "SAIC_R6_CANARY_V2_PYTHON_SHA256": self.python_sha,
            "SAIC_R6_CANARY_V2_OUTPUT_PARENT": str(self.output_parent),
            "SAIC_R6_CANARY_V2_LAUNCHER": str(self.launcher),
            "SAIC_R6_CANARY_V2_LAUNCHER_SHA256": self.launcher_sha,
            "SAIC_R6_CANARY_V2_SUBMISSION_RECEIPT": str(self.receipt),
        }

    def run(self, invoked: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["/usr/bin/bash", str(invoked or self.launcher)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
            env=self.env(),
        )


class LauncherSubmissionGateTests(unittest.TestCase):
    def make_fixture(self, wait_seconds: float = 0.4):
        temporary = tempfile.TemporaryDirectory(prefix="saic-r6-cny2-gate-")
        return temporary, LauncherGateFixture(Path(temporary.name).resolve(), wait_seconds)

    def test_same_inode_0600_delayed_publication_passes_before_output(self) -> None:
        temporary, fixture = self.make_fixture(wait_seconds=1.0)
        try:
            fixture.publish_receipt(fixture.receipt_core(), mode=0o600)

            def publish() -> None:
                time.sleep(0.1)
                fixture.receipt.chmod(0o444)

            thread = threading.Thread(target=publish)
            thread.start()
            completed = fixture.run()
            thread.join(timeout=2)
            self.assertEqual(completed.returncode, 2)
            self.assertIn(b"scratch parent differs", completed.stderr)
            self.assertFalse((fixture.output_parent / f"job-{fixture.JOB}").exists())
        finally:
            temporary.cleanup()

    def test_0600_timeout_fails_before_output(self) -> None:
        temporary, fixture = self.make_fixture(wait_seconds=0.15)
        try:
            fixture.publish_receipt(fixture.receipt_core(), mode=0o600)
            completed = fixture.run()
            self.assertEqual(completed.returncode, 1)
            self.assertIn(b"timed out waiting", completed.stderr)
            self.assertFalse((fixture.output_parent / f"job-{fixture.JOB}").exists())
        finally:
            temporary.cleanup()

    def test_wrong_job_and_resigned_exports_fail_before_output(self) -> None:
        for tamper in ("job", "exports"):
            with self.subTest(tamper=tamper):
                temporary, fixture = self.make_fixture()
                try:
                    core = fixture.receipt_core(
                        job="97542" if tamper == "job" else fixture.JOB
                    )
                    if tamper == "exports":
                        core["submission_boundary"]["exact_job_export_names"] = (
                            fixture.expected_exports()[:-1]
                        )
                    fixture.publish_receipt(core)
                    completed = fixture.run()
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(b"own canary submission receipt differs", completed.stderr)
                    self.assertFalse(
                        (fixture.output_parent / f"job-{fixture.JOB}").exists()
                    )
                finally:
                    temporary.cleanup()

    def test_invoked_script_bytes_mismatch_fails_before_receipt_or_output(self) -> None:
        temporary, fixture = self.make_fixture()
        try:
            fixture.publish_receipt(fixture.receipt_core())
            invoked = fixture.root / "different-spooled-script.sbatch"
            invoked.write_bytes(fixture.launcher.read_bytes() + b"\n")
            invoked.chmod(0o444)
            completed = fixture.run(invoked=invoked)
            self.assertEqual(completed.returncode, 2)
            self.assertIn(b"executed Slurm launcher differs", completed.stderr)
            self.assertFalse((fixture.output_parent / f"job-{fixture.JOB}").exists())
        finally:
            temporary.cleanup()


class TerminalFixture:
    JOB = "97551"

    def __init__(self, root: Path) -> None:
        self.base = root
        self.guard = plain_copy(GUARD_PATH, root / "inputs/guard.py")
        self.guard_module = load_module(f"terminal_guard_{id(self)}", self.guard)
        self.launcher = plain_copy(LAUNCHER_PATH, root / "inputs/launcher.sbatch")
        self.archive = root / "inputs/source.tar"
        self.archive.write_bytes(b"sealed source archive fixture\n")
        self.archive.chmod(0o444)
        self.python = Path(sys.executable).resolve(strict=True)
        self.output_parent = root / "output"
        self.output_parent.mkdir(mode=0o700)
        self.root = self.output_parent / f"job-{self.JOB}"
        self.logs = self.root / "logs"
        self.rendezvous = self.root / "rendezvous"
        self.claims = self.rendezvous / "port-claims"
        self.forbidden = self.root / "forbidden-attempts"
        for path in (self.root, self.logs, self.rendezvous, self.claims, self.forbidden):
            path.mkdir(mode=0o700)
        self.listener = self.logs / "legacy-arithmetic-port-listener.json"
        self.submission = self.output_parent / "submission-receipt.json"
        self.launcher_sha = sha_file(self.launcher)
        self.guard_sha = sha_file(self.guard)
        self.archive_sha = sha_file(self.archive)
        self.python_sha = sha_file(self.python)
        self.listener_pid = 45678
        self.chains: dict[tuple[str, int], dict[str, Path | int | str]] = {}
        self._build_logs_and_listener()
        self._build_success_lifecycles()
        self._build_collision()
        self._build_submission()

    @staticmethod
    def candidate_id(group: str, index: int) -> str:
        return f"canary-{group}-candidate-{index:02d}"

    def rdzv_id(self, group: str, index: int, ordinal: int) -> str:
        candidate = self.candidate_id(group, index)
        digest = sha_bytes(candidate.encode("ascii"))
        return f"saic-{self.JOB}-{group}-c{index:02d}-{digest[:16]}-l{ordinal:02d}"

    def identity(self, group: str, index: int, ordinal: int) -> dict:
        return {
            "slurm_job_id": self.JOB,
            "group_id": group,
            "candidate_index": index,
            "candidate_id": self.candidate_id(group, index),
            "launch_ordinal": ordinal,
            "rdzv_id": self.rdzv_id(group, index, ordinal),
        }

    def _build_logs_and_listener(self) -> None:
        for name in (
            "legacy-listener.stderr", "legacy-listener.stdout",
            "sp4-a.log", "sp4-b.log",
        ):
            path = self.logs / name
            path.write_bytes(b"")
            path.chmod(0o444)
        write_sealed(
            self.guard_module,
            self.listener,
            {
                "schema_version": "saic-r6-legacy-arithmetic-port-listener-v2",
                "status": "all_legacy_ports_listening_before_dynamic_canary",
                "slurm_job_id": self.JOB,
                "node": socket.gethostname(),
                "pid": self.listener_pid,
                "address": "127.0.0.1",
                "port_range_inclusive": [48730, 48789],
                "listener_count": 60,
                "listeners": [
                    {"port": port, "socket_inode": 100000 + port}
                    for port in range(48730, 48790)
                ],
                "receipt_publication_protocol": (
                    "open_first_inode_pinned_terminal_fchmod_v2"
                ),
                "authority": {
                    "scientific": False, "generation": False, "training": False,
                },
            },
        )

    def _build_success_lifecycles(self) -> None:
        number = 0
        for group in ("sp4-a", "sp4-b"):
            group_root = self.rendezvous / group
            group_root.mkdir(mode=0o700)
            for index in range(30):
                candidate = self.candidate_id(group, index)
                candidate_digest = sha_bytes(candidate.encode("ascii"))
                candidate_root = group_root / f"candidate-{index:02d}-{candidate_digest[:16]}"
                candidate_root.mkdir(mode=0o700)
                ordinal = 2 if group == "sp4-a" and index == 0 else 1
                life = candidate_root / f"launch-{ordinal:02d}"
                life.mkdir(mode=0o700)
                log = life / "torchrun.log"
                log.write_bytes(b"operational argparse help complete\n")
                log.chmod(0o444)
                port = 30000 + number
                number += 1
                identity = self.identity(group, index, ordinal)
                claim_path = self.claims / f"port-{port}.json"
                claim = write_sealed(
                    self.guard_module,
                    claim_path,
                    {
                        "schema_version": self.guard_module.CLAIM_SCHEMA_VERSION,
                        "status": "reserved_before_generation_runtime",
                        **identity,
                        "rdzv_backend": "c10d",
                        "rdzv_endpoint_request": "127.0.0.1:0",
                        "actual_master_addr": "127.0.0.1",
                        "actual_master_port": port,
                        "lifecycle_dir": str(life),
                        "lifecycle_dir_identity": directory_identity(life),
                        "admission_receipt_path": str(life / "admission.json"),
                        "torch_disable_share_rdzv_tcp_store": "0",
                        "shared_tcp_store_bootstrap": True,
                        "kernel_selected_free_port": True,
                        "port_claim_create_only_across_both_groups_for_this_job": True,
                        "generation_runtime_entered": False,
                        "scientific_spec_changed": False,
                        "authority": self.guard_module.AUTHORITY,
                    },
                )
                packets = []
                for rank in range(4):
                    packet = write_sealed(
                        self.guard_module,
                        life / f"rank-{rank}.json",
                        {
                            "schema_version": self.guard_module.RANK_SCHEMA_VERSION,
                            "status": "prepared_before_generation_runtime",
                            **identity,
                            "rdzv_backend": "c10d",
                            "rdzv_endpoint_request": "127.0.0.1:0",
                            "actual_master_addr": "127.0.0.1",
                            "actual_master_port": port,
                            "rank": rank,
                            "local_rank": rank,
                            "world_size": 4,
                            "local_world_size": 4,
                            "port_claim_receipt_digest": claim["receipt_digest"],
                            "runtime_sha256": self.guard_module.EXPECTED_RUNTIME_SHA256,
                            "torch_disable_share_rdzv_tcp_store": "0",
                            "shared_tcp_store_bootstrap": True,
                            "generation_runtime_entered_before_admission": False,
                            "scientific_spec_changed": False,
                            "authority": self.guard_module.AUTHORITY,
                        },
                    )
                    packets.append(packet)
                decision = write_sealed(
                    self.guard_module,
                    life / "admission.json",
                    {
                        "schema_version": self.guard_module.DECISION_SCHEMA_VERSION,
                        "status": "exact_world4_admitted_before_generation_runtime",
                        **identity,
                        "actual_master_addr": "127.0.0.1",
                        "actual_master_port": port,
                        "world_size": 4,
                        "rank_order": [0, 1, 2, 3],
                        "rank_packet_digests": [
                            packet["receipt_digest"] for packet in packets
                        ],
                        "port_claim_receipt_digest": claim["receipt_digest"],
                        "runtime_sha256": self.guard_module.EXPECTED_RUNTIME_SHA256,
                        "torch_disable_share_rdzv_tcp_store": "0",
                        "shared_tcp_store_bootstrap": True,
                        "all_four_ranks_admitted": True,
                        "generation_runtime_entry_authorized": True,
                        "scientific_spec_changed": False,
                        "authority": self.guard_module.AUTHORITY,
                    },
                )
                completion_path = life / "operational-completion.json"
                write_sealed(
                    self.guard_module,
                    completion_path,
                    {
                        "schema_version": (
                            "saic-r6-dynamic-rendezvous-operational-completion-v2"
                        ),
                        "status": (
                            "unchanged_runtime_import_and_help_completed_after_exact_world4_admission"
                        ),
                        **identity,
                        "actual_master_port": port,
                        "claim_path": str(claim_path),
                        "claim_receipt_digest": claim["receipt_digest"],
                        "rank_packet_digests": decision["rank_packet_digests"],
                        "admission_receipt_digest": decision["receipt_digest"],
                        "runtime_sha256": self.guard_module.EXPECTED_RUNTIME_SHA256,
                        "torch_disable_share_rdzv_tcp_store": "0",
                        "shared_tcp_store_bootstrap": True,
                        "receipt_publication_protocol": (
                            "open_first_inode_pinned_terminal_fchmod_v2"
                        ),
                        "scientific_generation_entered": False,
                        "scientific_output_created": False,
                        "authority": self.guard_module.AUTHORITY,
                    },
                )
                self.chains[(group, index)] = {
                    "life": life,
                    "port": port,
                    "claim": claim_path,
                    "completion": completion_path,
                }

    def _build_collision(self) -> None:
        current = self.chains[("sp4-a", 0)]
        prior = self.chains[("sp4-a", 1)]
        candidate_root = Path(str(current["life"])).parent
        collision_life = candidate_root / "launch-01"
        collision_life.mkdir(mode=0o700)
        log = collision_life / "torchrun.log"
        log.write_bytes(b"sealed same-job collision before runtime\n")
        log.chmod(0o444)
        prior_claim_path = Path(str(prior["claim"]))
        prior_claim = json.loads(prior_claim_path.read_text(encoding="ascii"))
        prior_admission = json.loads(
            (Path(str(prior["life"])) / "admission.json").read_text(encoding="ascii")
        )
        write_sealed(
            self.guard_module,
            collision_life / "collision.json",
            {
                "schema_version": self.guard_module.COLLISION_SCHEMA_VERSION,
                "status": "kernel_port_already_claimed_in_this_job_before_runtime",
                **self.identity("sp4-a", 0, 1),
                "actual_master_port": prior["port"],
                "existing_claim_receipt_digest": prior_claim["receipt_digest"],
                "existing_claim_sha256": sha_file(prior_claim_path),
                "existing_admission_receipt_path": prior_claim[
                    "admission_receipt_path"
                ],
                "existing_admission_receipt_digest": prior_admission[
                    "receipt_digest"
                ],
                "generation_runtime_entered": False,
                "candidate_output_reuse_authorized": False,
                "authority": self.guard_module.AUTHORITY,
            },
        )
        self.collision = collision_life / "collision.json"

    def _build_submission(self) -> None:
        gate = LauncherGateFixture.__new__(LauncherGateFixture)
        gate.JOB = self.JOB
        gate.launcher = self.launcher
        gate.guard = self.guard
        gate.archive = self.archive
        gate.python = self.python
        gate.output_parent = self.output_parent
        gate.receipt = self.submission
        gate.launcher_sha = self.launcher_sha
        gate.guard_sha = self.guard_sha
        gate.archive_sha = self.archive_sha
        gate.python_sha = self.python_sha
        gate.guard_module = self.guard_module
        write_sealed(self.guard_module, self.submission, gate.receipt_core())

    @staticmethod
    def terminal_source() -> str:
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        marker = 'spec = importlib.util.spec_from_file_location("canary_guard_terminal"'
        marker_at = source.index(marker)
        start = source.rfind("import hashlib\n", 0, marker_at)
        end = source.rfind("\nPY\n")
        if start < 0 or end <= marker_at:
            raise RuntimeError("terminal materializer source boundary differs")
        return source[start:end]

    def run_terminal(self) -> subprocess.CompletedProcess:
        command = [
            str(self.python), "-B", "-",
            str(self.guard), str(self.root), str(self.claims), str(self.listener),
            self.JOB, "node-test", self.guard_sha,
            str(self.archive), self.archive_sha,
            str(self.python), self.python_sha,
            str(self.launcher), self.launcher_sha,
            str(self.launcher), str(self.submission), str(self.listener_pid),
        ]
        environment = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "0",
        }
        return subprocess.run(
            command,
            input=self.terminal_source().encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env=environment,
        )

    def resign_full_candidate_chain(self) -> None:
        chain = self.chains[("sp4-a", 0)]
        life = Path(str(chain["life"]))
        identity = self.identity("sp4-a", 1, 2)
        claim_path = Path(str(chain["claim"]))
        claim_core = unseal(claim_path)
        claim_core.update(identity)
        claim = rewrite_sealed(self.guard_module, claim_path, claim_core)
        packets = []
        for rank in range(4):
            path = life / f"rank-{rank}.json"
            core = unseal(path)
            core.update(identity)
            core["port_claim_receipt_digest"] = claim["receipt_digest"]
            packets.append(rewrite_sealed(self.guard_module, path, core))
        admission_path = life / "admission.json"
        admission_core = unseal(admission_path)
        admission_core.update(identity)
        admission_core["port_claim_receipt_digest"] = claim["receipt_digest"]
        admission_core["rank_packet_digests"] = [
            packet["receipt_digest"] for packet in packets
        ]
        decision = rewrite_sealed(
            self.guard_module, admission_path, admission_core
        )
        completion_path = Path(str(chain["completion"]))
        completion_core = unseal(completion_path)
        completion_core.update(identity)
        completion_core["claim_receipt_digest"] = claim["receipt_digest"]
        completion_core["rank_packet_digests"] = decision["rank_packet_digests"]
        completion_core["admission_receipt_digest"] = decision["receipt_digest"]
        rewrite_sealed(self.guard_module, completion_path, completion_core)


class TerminalDeepAuditTests(unittest.TestCase):
    def make_fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix="saic-r6-cny2-terminal-")
        return temporary, TerminalFixture(Path(temporary.name).resolve())

    def assert_terminal_failure(self, fixture: TerminalFixture) -> None:
        completed = fixture.run_terminal()
        self.assertEqual(completed.returncode, 2, completed.stderr.decode("utf-8"))
        self.assertFalse((fixture.root / "canary-receipt.json").exists())
        failure = fixture.root / "canary-failure-receipt.json"
        self.assertTrue(failure.is_file())
        self.assertEqual(stat.S_IMODE(failure.stat().st_mode), 0o444)

    def test_terminal_exact60_positive_binds_submission_and_launcher(self) -> None:
        temporary, fixture = self.make_fixture()
        try:
            completed = fixture.run_terminal()
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8"))
            receipt = json.loads(
                (fixture.root / "canary-receipt.json").read_text(encoding="ascii")
            )
            self.assertEqual(receipt["candidate_count"], 60)
            self.assertEqual(receipt["rank_packet_count"], 240)
            self.assertEqual(receipt["collision_receipt_count"], 1)
            self.assertEqual(receipt["launcher_sha256"], fixture.launcher_sha)
            self.assertEqual(receipt["executed_launcher_sha256"], fixture.launcher_sha)
            self.assertEqual(receipt["submission_job_id"], fixture.JOB)
            self.assertEqual(receipt["submission_receipt_path"], str(fixture.submission))
            self.assertEqual(
                receipt["submission_receipt_sha256"], sha_file(fixture.submission)
            )
        finally:
            temporary.cleanup()

    def test_terminal_rejects_coordinated_candidate_chain_resign(self) -> None:
        temporary, fixture = self.make_fixture()
        try:
            fixture.resign_full_candidate_chain()
            self.assert_terminal_failure(fixture)
        finally:
            temporary.cleanup()

    def test_terminal_rejects_completion_link_swap_and_extra_claim(self) -> None:
        for tamper in ("completion", "extra-claim"):
            with self.subTest(tamper=tamper):
                temporary, fixture = self.make_fixture()
                try:
                    if tamper == "completion":
                        first = fixture.chains[("sp4-a", 0)]
                        other = fixture.chains[("sp4-a", 1)]
                        completion = Path(str(first["completion"]))
                        core = unseal(completion)
                        other_claim = json.loads(
                            Path(str(other["claim"])).read_text(encoding="ascii")
                        )
                        other_decision = json.loads(
                            (Path(str(other["life"])) / "admission.json").read_text(
                                encoding="ascii"
                            )
                        )
                        core["claim_path"] = str(other["claim"])
                        core["claim_receipt_digest"] = other_claim["receipt_digest"]
                        core["rank_packet_digests"] = other_decision[
                            "rank_packet_digests"
                        ]
                        core["admission_receipt_digest"] = other_decision[
                            "receipt_digest"
                        ]
                        rewrite_sealed(fixture.guard_module, completion, core)
                    else:
                        original = fixture.chains[("sp4-b", 29)]
                        extra_core = unseal(Path(str(original["claim"])))
                        extra_core["actual_master_port"] = 39999
                        write_sealed(
                            fixture.guard_module,
                            fixture.claims / "port-39999.json",
                            extra_core,
                        )
                    self.assert_terminal_failure(fixture)
                finally:
                    temporary.cleanup()

    def test_terminal_rejects_resigned_collision_listener_and_submission(self) -> None:
        for tamper in ("collision", "listener", "submission"):
            with self.subTest(tamper=tamper):
                temporary, fixture = self.make_fixture()
                try:
                    if tamper == "collision":
                        core = unseal(fixture.collision)
                        core["rdzv_id"] = fixture.rdzv_id("sp4-a", 0, 2)
                        rewrite_sealed(fixture.guard_module, fixture.collision, core)
                    elif tamper == "listener":
                        core = unseal(fixture.listener)
                        core["slurm_job_id"] = "97552"
                        rewrite_sealed(fixture.guard_module, fixture.listener, core)
                    else:
                        core = unseal(fixture.submission)
                        core["submitted_job"]["job_id"] = "97552"
                        rewrite_sealed(fixture.guard_module, fixture.submission, core)
                    self.assert_terminal_failure(fixture)
                finally:
                    temporary.cleanup()


class StaticEntryPointTests(unittest.TestCase):
    def test_shell_syntax_and_submitter_help(self) -> None:
        syntax = subprocess.run(
            ["/usr/bin/bash", "-n", str(LAUNCHER_PATH)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr.decode("utf-8"))
        help_result = subprocess.run(
            [str(Path(sys.executable).resolve()), "-B", str(SUBMITTER_PATH), "--help"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
            env={
                "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr.decode("utf-8"))
        self.assertIn(b"--launcher-sha256", help_result.stdout)


if __name__ == "__main__":
    unittest.main()
