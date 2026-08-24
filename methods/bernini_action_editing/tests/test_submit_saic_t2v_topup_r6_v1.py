#!/usr/bin/env python3
"""Hostile tests for the exact-once SAIC r6 full60 submitter."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
SUBMITTER_PATH = METHOD_ROOT / "tools/submit_saic_t2v_topup_r6_v1.py"
spec = importlib.util.spec_from_file_location("saic_r6_submitter", SUBMITTER_PATH)
assert spec is not None and spec.loader is not None
submitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submitter)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


class SubmitterFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.revision = "1" * 40
        self.runtime = (METHOD_ROOT / "generate_saic_pure_t2v_event_bank_topup_v2.py").read_bytes()
        self.launcher_bytes = b"#!/usr/bin/env bash\n#SBATCH --gres=gpu:mi210:8\n"
        self.guard_bytes = b"guard-bytes\n"
        self.source_manifest_bytes = b'{"source":"sealed"}\n'
        self.event_spec_bytes = b'{"event":"sealed"}\n'
        self.launcher = immutable(root / "launcher.sbatch", self.launcher_bytes)
        self.source_manifest = immutable(root / "source-manifest.json", self.source_manifest_bytes)
        self.event_spec = immutable(root / "event-spec.json", self.event_spec_bytes)
        self.checkpoint_manifest = immutable(root / "checkpoint.sha256", b"checkpoint\n")
        self.archive = root / "source.tar"
        archive_rows = {
            submitter.ARCHIVE_RUNTIME: self.runtime,
            submitter.ARCHIVE_GUARD: self.guard_bytes,
            submitter.ARCHIVE_LAUNCHER: self.launcher_bytes,
            submitter.ARCHIVE_SOURCE_MANIFEST: self.source_manifest_bytes,
            submitter.ARCHIVE_EVENT_SPEC: self.event_spec_bytes,
        }
        with tarfile.open(
            self.archive,
            "w",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": self.revision},
        ) as handle:
            for name, payload in archive_rows.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o444
                handle.addfile(info, io.BytesIO(payload))
        self.archive.chmod(0o444)
        self.python = Path(sys.executable).resolve(strict=True)
        self.ffmpeg = root / "ffmpeg"
        self.ffmpeg.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.ffmpeg.chmod(0o555)
        self.bernini = root / "bernini"
        self.veomni = root / "veomni"
        self.checkpoint = root / "checkpoint"
        self.output_parent = root / "runs"
        self.logs = root / "logs"
        for path in (self.bernini, self.veomni, self.checkpoint, self.output_parent, self.logs):
            path.mkdir()
        self.output = self.output_parent / "fresh-r6"
        self.receipt = Path(str(self.output) + ".submission.receipt.json")
        self.args = [
            "--launcher", str(self.launcher),
            "--launcher-sha256", digest_file(self.launcher),
            "--source-archive", str(self.archive),
            "--source-archive-sha256", digest_file(self.archive),
            "--source-revision", self.revision,
            "--source-manifest", str(self.source_manifest),
            "--source-manifest-sha256", digest_file(self.source_manifest),
            "--event-spec", str(self.event_spec),
            "--event-spec-sha256", digest_file(self.event_spec),
            "--rendezvous-guard-sha256", digest(self.guard_bytes),
            "--checkpoint-manifest", str(self.checkpoint_manifest),
            "--checkpoint-manifest-sha256", digest_file(self.checkpoint_manifest),
            "--python", str(self.python),
            "--python-sha256", digest_file(self.python),
            "--static-ffmpeg", str(self.ffmpeg),
            "--static-ffmpeg-sha256", digest_file(self.ffmpeg),
            "--bernini-root", str(self.bernini),
            "--veomni-root", str(self.veomni),
            "--checkpoint", str(self.checkpoint),
            "--output-root", str(self.output),
            "--receipt", str(self.receipt),
            "--slurm-log-dir", str(self.logs),
        ]


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix="saic-r6-submit-test-") as temporary:
        item = SubmitterFixture(Path(temporary).resolve())
        old_runtime = submitter.EXPECTED_RUNTIME_SHA256
        submitter.EXPECTED_RUNTIME_SHA256 = digest(item.runtime)
        try:
            yield item
        finally:
            submitter.EXPECTED_RUNTIME_SHA256 = old_runtime


class SubmitterTests(unittest.TestCase):
    def test_success_is_exact_once_nonhold_and_no_dependency(self) -> None:
        with fixture() as item:
            completed = SimpleNamespace(returncode=0, stdout=b"24680;auh\n", stderr=b"")
            def immediate_start(*_args, **_kwargs):
                item.output.mkdir()
                return completed

            with mock.patch.object(subprocess, "run", side_effect=immediate_start) as call:
                self.assertEqual(submitter.main(item.args), 0)
            command = call.call_args.args[0]
            self.assertEqual(command[0], "/usr/bin/sbatch")
            self.assertNotIn("--hold", command)
            self.assertFalse(any("dependency" in value.lower() for value in command))
            export = next(value for value in command if value.startswith("--export="))
            self.assertTrue(export.startswith("--export=NONE,"))
            self.assertNotIn("ALL", export)
            value = json.loads(item.receipt.read_text(encoding="ascii"))
            digest_value = value.pop("receipt_digest")
            self.assertEqual(digest_value, digest(submitter.canonical(value)))
            self.assertEqual(value["submitted_job"]["job_id"], "24680")
            self.assertFalse(value["request"]["hold"])
            self.assertIsNone(value["request"]["dependency"])
            info = item.receipt.lstat()
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o444)
            self.assertEqual(info.st_nlink, 1)
            with mock.patch.object(subprocess, "run") as second:
                with self.assertRaises(SystemExit):
                    submitter.main(item.args)
                second.assert_not_called()

    def test_sbatch_failure_retains_only_0600_non_success_reservation(self) -> None:
        with fixture() as item:
            completed = SimpleNamespace(returncode=1, stdout=b"", stderr=b"rejected\n")
            with mock.patch.object(subprocess, "run", return_value=completed) as call:
                with self.assertRaises(SystemExit):
                    submitter.main(item.args)
                call.assert_called_once()
            info = item.receipt.lstat()
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o600)
            value = json.loads(item.receipt.read_text(encoding="ascii"))
            self.assertFalse(value["submission_success"])
            self.assertEqual(value["status"], "reserved_before_sbatch_not_submission_success")
            self.assertFalse(item.output.exists())

    def test_precheck_tamper_never_calls_sbatch_or_reserves_receipt(self) -> None:
        with fixture() as item:
            item.launcher.chmod(0o644)
            with mock.patch.object(subprocess, "run") as call:
                with self.assertRaises(SystemExit):
                    submitter.main(item.args)
                call.assert_not_called()
            self.assertFalse(item.receipt.exists())

    def test_archive_member_resign_cannot_replace_frozen_launcher(self) -> None:
        with fixture() as item:
            replacement = immutable(item.root / "other-launcher.sbatch", b"different\n")
            index = item.args.index("--launcher") + 1
            item.args[index] = str(replacement)
            item.args[item.args.index("--launcher-sha256") + 1] = digest_file(replacement)
            with mock.patch.object(subprocess, "run") as call:
                with self.assertRaises(SystemExit):
                    submitter.main(item.args)
                call.assert_not_called()
            self.assertFalse(item.receipt.exists())

    def test_sbatch_pathname_swap_cannot_publish_false_success(self) -> None:
        with fixture() as item:
            moved = item.root / "reservation-moved-aside.json"
            completed = SimpleNamespace(returncode=0, stdout=b"24681\n", stderr=b"")

            def swap_reservation(*_args, **_kwargs):
                item.receipt.rename(moved)
                item.receipt.write_text("replacement\n", encoding="ascii")
                item.receipt.chmod(0o600)
                return completed

            with mock.patch.object(subprocess, "run", side_effect=swap_reservation) as call:
                with self.assertRaises(SystemExit):
                    submitter.main(item.args)
                call.assert_called_once()
            self.assertEqual(stat.S_IMODE(moved.lstat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(item.receipt.lstat().st_mode), 0o600)

    def test_archive_escape_is_rejected_before_required_name_normalization(self) -> None:
        with fixture() as item:
            item.archive.chmod(0o600)
            with tarfile.open(
                item.archive,
                "w",
                format=tarfile.PAX_FORMAT,
                pax_headers={"comment": item.revision},
            ) as handle:
                payload = item.launcher_bytes
                info = tarfile.TarInfo("../../" + submitter.ARCHIVE_LAUNCHER)
                info.size = len(payload)
                handle.addfile(info, io.BytesIO(payload))
            item.archive.chmod(0o444)
            item.args[item.args.index("--source-archive-sha256") + 1] = digest_file(item.archive)
            with mock.patch.object(subprocess, "run") as call:
                with self.assertRaises(SystemExit):
                    submitter.main(item.args)
                call.assert_not_called()
            self.assertFalse(item.receipt.exists())


if __name__ == "__main__":
    unittest.main()
