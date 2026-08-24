#!/usr/bin/env python3
"""Hostile static tests for the exact5 step-429 replay recovery STOP file."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (
    METHOD_ROOT / "scripts"
    / "auh_recover_case01_source_bone_exact5_static_step429_job143808_node292_replay_only_v1.STOP.sh"
)
FORBIDDEN_STEP_PROGRAM = bytes((115, 114, 117, 110)).decode("ascii")


def embedded_python() -> str:
    source = CONTROLLER.read_text(encoding="utf-8")
    marker = "<<'PY'\n"
    if source.count(marker) != 1 or not source.endswith("\nPY\n"):
        raise RuntimeError("controller heredoc closure differs")
    return source.split(marker, 1)[1][:-4]


def load_embedded() -> types.ModuleType:
    module = types.ModuleType("_exact5_step429_replay_fixture")
    code = embedded_python()
    exec(compile(code, str(CONTROLLER) + ":embedded", "exec"), module.__dict__)
    return module


MODULE = load_embedded()


def finish_digest(value: dict, field: str) -> dict:
    result = dict(value)
    result[field] = MODULE.digest(result)
    return result


def package_fixture(payload_raw: bytes) -> tuple[dict, bytes]:
    payload_sha = hashlib.sha256(payload_raw).hexdigest()
    body = {
        "schema_version": "case01-source-bone-exact5-r64-materialization-v1",
        "status": "MATERIALIZED_NOT_SUBMITTED",
        "root": MODULE.ROOT,
        "holder_job_id": MODULE.JOB,
        "expected_node": MODULE.NODE,
        "campaign_mode": MODULE.CAMPAIGN,
        "selected_task_ids": list(MODULE.TASKS),
        "task_count": 5,
        "physical_release_file_count": 19,
        "production_identity_count": 18,
        "production_identity_decomposition": {
            "r5f_roles_with_exact5_wrapper_runner": 16,
            "additional_frozen_runner": 1,
            "additional_exact5_eval": 1,
        },
        "sealed_r5f_infer_lora_reused": True,
        "working_tree_infer_lora_read": False,
        "captured_materializer_sha256": "1" * 64,
        "input_root": "/sealed/input",
        "independent_audit": {
            "path": "/sealed/audit", "sha256": "2" * 64,
            "size": 2, "audit_digest": "3" * 64,
        },
        "plan": {
            "path": "/sealed/plan", "sha256": "4" * 64,
            "size": 4, "plan_digest": "5" * 64,
        },
        "launch": {
            "input": "/sealed/launch-input", "input_sha256": "6" * 64,
            "payload": "/sealed/launch-payload", "payload_sha256": "7" * 64,
            "payload_size": 7, "receipt": "/sealed/launch-receipt",
            "receipt_sha256": "8" * 64, "receipt_digest": "9" * 64,
            "release_digest": "a" * 64,
        },
        "cpu_admission": {
            "required_before_gpu_attempt": True,
            "static_probe": {
                "source": "/sealed/static.py", "source_sha256": "b" * 64,
                "payload": MODULE.PAYLOAD, "payload_sha256": payload_sha,
                "receipt": MODULE.RECEIPT, "executed": False,
            },
            "captured_root_fake_runner_probe": {"fixture": True},
        },
        "rank_cache_root": MODULE.CACHE,
        "fresh_outputs": True,
        "fresh_final": True,
        "fresh_runtime": True,
        "publication_final_internal_paths_pairwise_disjoint": True,
        "slurm_step_launched": False,
        "gpu_attempt_claimed": False,
        "retry_allowed_after_gpu_attempt": False,
        "artifacts_before_materialization_receipt": {
            "diagnostics/exact5_static_probe_payload_v1.sh": {
                "sha256": payload_sha, "size": len(payload_raw), "mode": 0o444,
            },
        },
    }
    package = finish_digest(body, "receipt_digest")
    return package, MODULE.canonical(package) + b"\n"


def attempt_fixture(package: dict, package_raw: bytes, payload_raw: bytes) -> dict:
    body = {
        "schema_version": "case01-source-bone-exact5-static-attempt-v1",
        "status": MODULE.ATTEMPT_STATUS,
        "holder_job_id": MODULE.JOB,
        "node": MODULE.NODE,
        "package_receipt_sha256": hashlib.sha256(package_raw).hexdigest(),
        "package_receipt_digest": package["receipt_digest"],
        "payload_path": MODULE.PAYLOAD,
        "payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
        "receipt_path": MODULE.RECEIPT,
        MODULE.SINGLE_ATTEMPT_KEY: True,
        "retry_allowed": False,
        "renderer_executed": False,
    }
    return finish_digest(body, "attempt_digest")


def receipt_fixture(package: dict) -> dict:
    body = {
        "schema_version": "case01-source-bone-exact5-static-probe-v1",
        "status": "PASS",
        "campaign_mode": MODULE.CAMPAIGN,
        "holder_job_id": MODULE.JOB,
        "expected_node": MODULE.NODE,
        "slurm_step_id": MODULE.STEP,
        "task_count": 5,
        "selected_task_ids": list(MODULE.TASKS),
        "release_file_count": 19,
        "launch_identity_count": 18,
        "plan_sha256": package["plan"]["sha256"],
        "plan_digest": package["plan"]["plan_digest"],
        "independent_audit_sha256": package["independent_audit"]["sha256"],
        "independent_audit_digest": package["independent_audit"]["audit_digest"],
        "checkpoint_manifest_sha256": MODULE.CHECKPOINT_MANIFEST_SHA256,
        "launch_receipt_sha256": package["launch"]["receipt_sha256"],
        "launch_receipt_digest": package["launch"]["receipt_digest"],
        "payload_sha256": package["launch"]["payload_sha256"],
        "ffprobe_path": MODULE.FFPROBE,
        "ffprobe_sha256": MODULE.FFPROBE_SHA256,
        "rank_cache_root": MODULE.CACHE,
        "production_outputs_fresh": True,
        "rank_cache_fresh": True,
        "pure_metadata_only": True,
        "torch_imported": False,
        "renderer_executed": False,
    }
    return finish_digest(body, "receipt_digest")


def dummy_identity() -> dict:
    return {
        "device": 1, "inode": 2, "uid": 2012, "gid": 2000,
        "mode": stat.S_IFREG | 0o400, "nlink": 1, "rdev": 0,
        "size": 3, "blocks": 1, "mtime_ns": 4, "ctime_ns": 5,
    }


def synthetic_recomputed_chain(changed: str | None = None) -> dict[str, bytes]:
    payload = b"sealed payload\n"
    if changed == "payload":
        payload = b"replacement payload\n"
    package_body = {
        "schema_version": "synthetic-package-v1",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_size": len(payload),
    }
    if changed == "package":
        package_body["replacement"] = True
    package = finish_digest(package_body, "receipt_digest")
    package_raw = MODULE.canonical(package) + b"\n"
    receipt_body = {
        "schema_version": "synthetic-receipt-v1", "status": "PASS",
        "package_sha256": hashlib.sha256(package_raw).hexdigest(),
        "package_digest": package["receipt_digest"],
    }
    if changed == "receipt":
        receipt_body["replacement"] = True
    receipt = finish_digest(receipt_body, "receipt_digest")
    receipt_raw = MODULE.canonical(receipt) + b"\n"
    attempt_body = {
        "schema_version": "synthetic-attempt-v1",
        "package_sha256": hashlib.sha256(package_raw).hexdigest(),
        "package_digest": package["receipt_digest"],
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "receipt_digest": receipt["receipt_digest"],
    }
    if changed == "attempt":
        attempt_body["replacement"] = True
    attempt = finish_digest(attempt_body, "attempt_digest")
    attempt_raw = MODULE.canonical(attempt) + b"\n"
    stdout = b"PASS " + receipt["receipt_digest"].encode("ascii") + b"\n"
    if changed == "stdout":
        stdout += b"replacement\n"
    stderr = b"replacement\n" if changed == "stderr" else b""
    return {
        "package": package_raw, "payload": payload, "attempt": attempt_raw,
        "receipt": receipt_raw, "stdout": stdout, "stderr": stderr,
    }


class StaticControllerTests(unittest.TestCase):
    def test_bash_syntax_and_embedded_normal_optimized_compile(self) -> None:
        observed = subprocess.run(
            ["/bin/bash", "-n", str(CONTROLLER)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(observed.returncode, 0, observed.stderr.decode("utf-8"))
        source = embedded_python()
        for optimize in (0, 2):
            compile(source, str(CONTROLLER), "exec", optimize=optimize)

    def test_ast_has_no_step_launcher_or_forbidden_process_api(self) -> None:
        controller_source = CONTROLLER.read_text(encoding="utf-8")
        python_source = embedded_python()
        self.assertNotIn(FORBIDDEN_STEP_PROGRAM, controller_source.lower())
        tree = ast.parse(python_source)
        imported = {
            alias.name
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        }
        called = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.append(node.func.attr)
        self.assertNotIn("subprocess", imported)
        self.assertNotIn("Popen", called)
        self.assertNotIn("system", called)
        self.assertEqual(called.count("fork"), 1)
        self.assertEqual(called.count("execve"), 1)
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
        self.assertIn('SACCT = "/usr/bin/sacct"', python_source)
        self.assertIn('FULL_STEP = JOB + "." + STEP', python_source)
        self.assertIn('STEP = "429"', python_source)

    def test_controller_is_trusted_stdin_and_exact_recovery_only(self) -> None:
        source = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn('[[ "$0" == /bin/bash && "$#" -eq 0', source)
        self.assertIn('exec -c "/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B -', source)
        self.assertIn('exec {SACCT_FD}<"$SACCT"', source)
        self.assertIn('"--jobs=" + FULL_STEP', source)
        self.assertIn('"COMPLETED", "0:0"', source)
        self.assertIn('ORIGINAL_CONTROLLER_SHA256 = "c997805c', source)
        self.assertIn('RECOVERY_REASON = "outer_postflight_rc1_', source)
        self.assertEqual(source.count("os.fchmod(descriptor, 0o400)"), 2)

    def test_exact_production_raw_sha_and_size_pins(self) -> None:
        self.assertEqual(
            {
                "package": (MODULE.PACKAGE_SHA256, MODULE.PACKAGE_SIZE),
                "payload": (MODULE.PAYLOAD_SHA256, MODULE.PAYLOAD_SIZE),
                "attempt": (MODULE.ATTEMPT_SHA256, MODULE.ATTEMPT_SIZE),
                "receipt": (MODULE.RECEIPT_SHA256, MODULE.RECEIPT_SIZE),
                "stdout": (MODULE.STDOUT_SHA256, MODULE.STDOUT_SIZE),
                "stderr": (MODULE.STDERR_SHA256, MODULE.STDERR_SIZE),
            },
            {
                "package": (
                    "0561608208e5a155028d4f8ec876b91a096189e7bd16bf71b8c72ee609e0433b",
                    12128,
                ),
                "payload": (
                    "79e064f1a3f77b36f12311d1e89b90747c97c5e756eb9e63314b5371182bfd11",
                    5257,
                ),
                "attempt": (
                    "969b43ddb0bd40646244e0624d5e0e728a49583ae4dfdab3f9f69d29caaafdff",
                    1000,
                ),
                "receipt": (
                    "b435fb39c481ac34732e754532f34b7e6c2eb679cf4b44352b50d1e52f3908cc",
                    1764,
                ),
                "stdout": (
                    "f8e320778e240b7ccb2ea03726e81231d602e0ed90c783fc6e7a18b0845f18d8",
                    91,
                ),
                "stderr": (
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    0,
                ),
            },
        )

    def test_each_replacement_rejected_after_full_chain_recomputation(self) -> None:
        baseline = synthetic_recomputed_chain()
        fixed_pins = {
            name: (hashlib.sha256(raw).hexdigest(), len(raw))
            for name, raw in baseline.items()
        }
        for changed in baseline:
            with self.subTest(changed=changed):
                recomputed = synthetic_recomputed_chain(changed)
                with self.assertRaises(RuntimeError):
                    for name, raw in recomputed.items():
                        sha256, size = fixed_pins[name]
                        MODULE.require_exact_raw(raw, sha256, size, name)

    def test_final_precreate_replay_covers_six_artifacts_and_both_parents(self) -> None:
        source = embedded_python()
        precreate = source[source.index("stdout_raw, stdout_identity = seal_log"):]
        precreate = precreate[:precreate.index("value = build_evidence")]
        for token in (
            "PACKAGE_SHA256, PACKAGE_SIZE", "PAYLOAD_SHA256, PAYLOAD_SIZE",
            "RECEIPT_SHA256, RECEIPT_SIZE", "ATTEMPT_SHA256, ATTEMPT_SIZE",
            "STDOUT_SHA256, STDOUT_SIZE", "STDERR_SHA256, STDERR_SIZE",
            "identity(evidence_parent_final) != identity(evidence_dir_before)",
            "identity(logs_parent_final) != identity(logs_dir_before)",
            "require_fresh_at(evidence_dir_fd",
        ):
            self.assertIn(token, precreate)

    def test_strict_json_rejects_duplicate_noncanonical_and_nonfinite(self) -> None:
        self.assertEqual(MODULE.strict_json(b'{"a":1}\n'), {"a": 1})
        for hostile in (
            b'{"a":1,"a":1}\n',
            b'{"a": 1}\n',
            b'{"a":NaN}\n',
            b'{"a":1}',
        ):
            with self.subTest(hostile=hostile), self.assertRaises(RuntimeError):
                MODULE.strict_json(hostile)

    def test_exact_package_attempt_receipt_and_step_are_fail_closed(self) -> None:
        payload_raw = b"#!/bin/bash -p\n"
        package, package_raw = package_fixture(payload_raw)
        MODULE.validate_package(package, package_raw, payload_raw)
        attempt = attempt_fixture(package, package_raw, payload_raw)
        MODULE.validate_attempt(attempt, package_raw, package, payload_raw)
        receipt = receipt_fixture(package)
        self.assertEqual(MODULE.validate_receipt(receipt, package), receipt["receipt_digest"])

        hostile_attempt = dict(attempt)
        hostile_attempt[MODULE.SINGLE_ATTEMPT_KEY] = False
        hostile_attempt.pop("attempt_digest")
        hostile_attempt = finish_digest(hostile_attempt, "attempt_digest")
        with self.assertRaises(RuntimeError):
            MODULE.validate_attempt(hostile_attempt, package_raw, package, payload_raw)

        hostile_receipt = dict(receipt)
        hostile_receipt["slurm_step_id"] = "430"
        hostile_receipt.pop("receipt_digest")
        hostile_receipt = finish_digest(hostile_receipt, "receipt_digest")
        with self.assertRaises(RuntimeError):
            MODULE.validate_receipt(hostile_receipt, package)

    def test_accounting_parser_requires_unique_exact_identity_completed_zero(self) -> None:
        line = (
            MODULE.FULL_STEP + "|" + MODULE.STEP_NAME
            + "|COMPLETED|0:0|" + MODULE.NODE + "\n"
        ).encode("ascii")
        self.assertEqual(
            MODULE.parse_accounting_output(line),
            {
                "JobIDRaw": MODULE.FULL_STEP, "JobName": MODULE.STEP_NAME,
                "State": "COMPLETED", "ExitCode": "0:0", "NodeList": MODULE.NODE,
            },
        )
        for hostile in (
            line + line,
            line.replace(b"COMPLETED", b"FAILED"),
            line.replace(b"0:0", b"1:0"),
            line.replace(MODULE.NODE.encode(), b"auh7-1b-gpu-293"),
            line.rstrip(b"\n") + b"|\n",
        ):
            with self.subTest(hostile=hostile), self.assertRaises(RuntimeError):
                MODULE.parse_accounting_output(hostile)

    def test_held_log_fchmod_seals_and_replays_exact_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            path = parent / "stdout.log"
            raw = b"sealed stdout\n"
            path.write_bytes(raw)
            path.chmod(0o600)
            descriptor = os.open(path, os.O_RDONLY)
            parent_descriptor = os.open(parent, os.O_RDONLY)
            old_uid, old_gid = MODULE.OWNER_UID, MODULE.OWNER_GID
            MODULE.OWNER_UID, MODULE.OWNER_GID = os.getuid(), os.getgid()
            try:
                before = MODULE.validate_log_before(
                    descriptor, parent_descriptor, str(path), raw,
                    hashlib.sha256(raw).hexdigest(), len(raw),
                )
                replayed, observed_identity = MODULE.seal_log(
                    descriptor, parent_descriptor, str(path), before, raw,
                )
                self.assertEqual(replayed, raw)
                self.assertEqual(stat.S_IMODE(os.fstat(descriptor).st_mode), 0o400)
                self.assertEqual(observed_identity, MODULE.identity(os.fstat(descriptor)))
            finally:
                MODULE.OWNER_UID, MODULE.OWNER_GID = old_uid, old_gid
                os.close(parent_descriptor)
                os.close(descriptor)

    def _evidence_fixture(self) -> dict:
        payload_raw = b"payload\n"
        package, package_raw = package_fixture(payload_raw)
        receipt = receipt_fixture(package)
        receipt_raw = MODULE.canonical(receipt) + b"\n"
        attempt = attempt_fixture(package, package_raw, payload_raw)
        attempt_raw = MODULE.canonical(attempt) + b"\n"
        value = MODULE.build_evidence(
            package_raw, package, dummy_identity(), payload_raw, dummy_identity(),
            receipt_raw, receipt, dummy_identity(), attempt_raw, attempt,
            dummy_identity(), b"stdout\n", dummy_identity(), b"", dummy_identity(),
        )
        self.assertEqual(set(value), MODULE.EVIDENCE_FIELDS)
        unsigned = dict(value)
        claimed = unsigned.pop("evidence_digest")
        self.assertEqual(claimed, MODULE.digest(unsigned))
        return value

    def test_create_only_final_0400_identity_bytes_and_second_create_rejected(self) -> None:
        value = self._evidence_fixture()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_descriptor = os.open(parent, os.O_RDONLY)
            old_path = MODULE.EVIDENCE
            old_uid, old_gid = MODULE.OWNER_UID, MODULE.OWNER_GID
            MODULE.EVIDENCE = str(parent / "evidence.json")
            MODULE.OWNER_UID, MODULE.OWNER_GID = os.getuid(), os.getgid()
            descriptor = None
            try:
                descriptor = MODULE.create_evidence(parent_descriptor, value)
                expected = MODULE.canonical(value) + b"\n"
                self.assertEqual(os.pread(descriptor, len(expected), 0), expected)
                self.assertEqual(stat.S_IMODE(os.fstat(descriptor).st_mode), 0o400)
                with self.assertRaises(FileExistsError):
                    MODULE.create_evidence(parent_descriptor, value)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                MODULE.EVIDENCE = old_path
                MODULE.OWNER_UID, MODULE.OWNER_GID = old_uid, old_gid
                os.close(parent_descriptor)

    def test_create_only_final_replay_rejects_commit_byte_tamper(self) -> None:
        value = self._evidence_fixture()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_descriptor = os.open(parent, os.O_RDONLY)
            old_path = MODULE.EVIDENCE
            old_uid, old_gid = MODULE.OWNER_UID, MODULE.OWNER_GID
            MODULE.EVIDENCE = str(parent / "tampered.json")
            MODULE.OWNER_UID, MODULE.OWNER_GID = os.getuid(), os.getgid()
            real_fchmod = os.fchmod

            def tampering_fchmod(descriptor, mode):
                real_fchmod(descriptor, mode)
                os.pwrite(descriptor, b"X", 0)

            try:
                with mock.patch.object(MODULE.os, "fchmod", tampering_fchmod):
                    with self.assertRaises(RuntimeError):
                        MODULE.create_evidence(parent_descriptor, value)
            finally:
                MODULE.EVIDENCE = old_path
                MODULE.OWNER_UID, MODULE.OWNER_GID = old_uid, old_gid
                os.close(parent_descriptor)


if __name__ == "__main__":
    unittest.main()
