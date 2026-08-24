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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import full644_exploratory_matched_r5f_shared_ofd_probe_v1 as probe


class R5FSharedOFDProbeTests(unittest.TestCase):
    def _authority(self, root: Path) -> tuple[Path, bytes, int, dict[str, int]]:
        path = root / "authority.bin"
        raw = bytes(range(256)) * 257
        path.write_bytes(raw)
        path.chmod(0o444)
        descriptor = os.open(str(path), os.O_RDONLY)
        return path, raw, descriptor, probe.stat_identity(os.fstat(descriptor))

    def test_double_pread_preserves_offset_and_rejects_short_and_wrong_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, raw, descriptor, identity = self._authority(Path(temporary))
            self.addCleanup(os.close, descriptor)
            digest = hashlib.sha256(raw).hexdigest()
            os.lseek(descriptor, 13, os.SEEK_SET)
            result = probe.verify_double_pread(descriptor, identity, len(raw), digest)
            self.assertEqual(result["sha256"], [digest, digest])
            self.assertEqual(os.lseek(descriptor, 0, os.SEEK_CUR), 13)

            def short(fd: int, count: int, offset: int) -> bytes:
                return os.pread(fd, count - 2, offset)

            with self.assertRaises(probe.R5FSharedOFDProbeError) as caught:
                probe.verify_double_pread(
                    descriptor, identity, len(raw), digest, pread=short
                )
            self.assertEqual(caught.exception.code, "short-read")
            wrong = ("0" if digest[0] != "0" else "1") + digest[1:]
            with self.assertRaises(probe.R5FSharedOFDProbeError) as caught:
                probe.verify_double_pread(descriptor, identity, len(raw), wrong)
            self.assertEqual(caught.exception.code, "digest-mismatch")

    @unittest.skipUnless(hasattr(os, "fork"), "fork is required")
    def test_four_pread_workers_and_controlled_legacy_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, raw, descriptor, identity = self._authority(Path(temporary))
            self.addCleanup(os.close, descriptor)
            digest = hashlib.sha256(raw).hexdigest()
            safe = probe.run_pread_concurrency(
                descriptor, identity, len(raw), digest, 17
            )
            self.assertEqual(len(safe["workers"]), 4)
            self.assertEqual(safe["parent_offset_before"], 17)
            self.assertEqual(safe["parent_offset_after"], 17)
            control = os.open(str(path), os.O_RDONLY)
            try:
                legacy = probe.run_legacy_contention_control(
                    control, identity, len(raw), digest
                )
            finally:
                os.close(control)
            self.assertEqual(legacy["full_read_count"], 1)
            self.assertEqual(legacy["short_read_count"], 3)
            self.assertTrue(legacy["contention_detected"])

    def test_closed_reused_short_and_wrong_digest_are_all_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, raw, descriptor, identity = self._authority(Path(temporary))
            self.addCleanup(os.close, descriptor)
            rows = probe.exercise_hostile_rejections(
                descriptor, identity, len(raw), hashlib.sha256(raw).hexdigest()
            )
            self.assertEqual(
                [(row["case"], row["rejection_code"]) for row in rows],
                [
                    ("closed-fd", "fd-unavailable"),
                    ("reused-fd", "fd-not-regular"),
                    ("short-read", "short-read"),
                    ("wrong-digest", "digest-mismatch"),
                ],
            )

    def test_receipt_is_canonical_0400_nlink1_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            row = probe.write_receipt(path, {"schema_version": probe.SCHEMA, "status": "PASS"})
            raw = path.read_bytes()
            value = json.loads(raw)
            unsigned = dict(value)
            claim = unsigned.pop("receipt_digest")
            self.assertEqual(claim, probe.object_sha256(unsigned))
            self.assertEqual(raw, probe.canonical_bytes(value) + b"\n")
            self.assertEqual(row["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            self.assertEqual(path.stat().st_nlink, 1)
            with self.assertRaises(FileExistsError):
                probe.write_receipt(path, {"status": "PASS"})

    @unittest.skipUnless(
        sys.platform == "linux" and Path("/proc/self/fd").is_dir(),
        "public probe requires Linux /proc/self/fd",
    )
    def test_public_cli_under_isolated_normal_or_optimized_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "authority.bin"
            raw = bytes(range(251)) * 113
            source.write_bytes(raw)
            source.chmod(0o444)
            receipt = root / "receipt.json"
            command = [
                sys.executable,
                "-I",
                "-S",
                "-B",
                *(["-O"] if sys.flags.optimize else []),
                str(Path(probe.__file__).resolve()),
                "--source",
                str(source),
                "--expected-size",
                str(len(raw)),
                "--expected-sha256",
                hashlib.sha256(raw).hexdigest(),
                "--receipt",
                str(receipt),
            ]
            completed = subprocess.run(
                command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(completed.stderr, b"")
            reference = json.loads(completed.stdout)
            self.assertEqual(completed.stdout, probe.canonical_bytes(reference) + b"\n")
            value = json.loads(receipt.read_bytes())
            self.assertEqual(value["status"], "PASS")
            self.assertEqual(value["summary"]["worker_count"], 4)
            self.assertEqual(value["summary"]["pread_count"], 8)
            self.assertEqual(value["summary"]["hostile_rejection_count"], 4)
            self.assertTrue(value["summary"]["legacy_contention_detected"])
            self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)


if __name__ == "__main__":
    unittest.main()
