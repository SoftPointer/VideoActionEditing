#!/usr/bin/env python3
"""Hostile local tests for the inert physical15 AUH source stager."""

from __future__ import annotations

import base64
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    METHOD_ROOT / "scripts"
    / "auh_stage_case01_object_trajectory_exact5_source_once_v1.HOLD.py"
)
BOOTSTRAP = (
    METHOD_ROOT
    / "case01_object_trajectory_exact5_source_stager_remote_bootstrap_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "case01_object_trajectory_source_stager_hold_v1", SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("source stager import spec differs")
stager = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stager
SPEC.loader.exec_module(stager)

BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "case01_object_trajectory_source_stager_generic_bootstrap_v1",
    BOOTSTRAP,
)
if BOOTSTRAP_SPEC is None or BOOTSTRAP_SPEC.loader is None:
    raise RuntimeError("generic source stager bootstrap import spec differs")
generic_bootstrap = importlib.util.module_from_spec(BOOTSTRAP_SPEC)
sys.modules[BOOTSTRAP_SPEC.name] = generic_bootstrap
BOOTSTRAP_SPEC.loader.exec_module(generic_bootstrap)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stat_namespace(info, **changes):
    values = {
        name: getattr(info, name)
        for name in (
            "st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink",
            "st_rdev", "st_size", "st_blocks", "st_mtime_ns", "st_ctime_ns",
        )
    }
    values["st_flags"] = getattr(info, "st_flags", 0)
    values.update(changes)
    return SimpleNamespace(**values)


@dataclass
class Fixture:
    base: Path
    local: Path
    remote: Path
    target: Path
    receipt: Path
    local_terminal: Path
    specs: tuple[dict[str, object], ...]
    contents: dict[str, bytes]

    def remove_outputs(self) -> None:
        for candidate in tuple(self.remote.iterdir()):
            if candidate.is_symlink() or candidate.is_file():
                try:
                    candidate.chmod(0o600)
                except FileNotFoundError:
                    pass
                candidate.unlink(missing_ok=True)
                continue
            for directory, _children, _files in os.walk(
                candidate, topdown=True, followlinks=False,
            ):
                os.chmod(directory, 0o700)
            shutil.rmtree(candidate)


@contextmanager
def physical15_fixture():
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary).resolve()
        local = base / "local"
        remote = base / "remote"
        local.mkdir()
        remote.mkdir()
        rows: list[dict[str, object]] = []
        contents: dict[str, bytes] = {}
        for index, original in enumerate(stager.SOURCE_AUTHORITIES):
            relative = original["relative"]
            raw = f"physical15:{index:02d}:{relative}\n".encode("utf-8")
            path = local / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            path.chmod(0o644)
            contents[relative] = raw
            rows.append({
                "relative": relative,
                "sha256": sha(raw),
                "size": len(raw),
            })
        fixture = Fixture(
            base=base,
            local=local,
            remote=remote,
            target=remote / "physical15-target",
            receipt=remote / "physical15-target.receipt.json",
            local_terminal=base / "commit-terminal.json",
            specs=tuple(rows),
            contents=contents,
        )
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                stager, "SOURCE_AUTHORITIES", fixture.specs,
            ))
            stack.enter_context(mock.patch.object(
                stager, "LOCAL_SOURCE_ROOT", fixture.local,
            ))
            stack.enter_context(mock.patch.object(
                stager, "REMOTE_PARENT", fixture.remote,
            ))
            stack.enter_context(mock.patch.object(
                stager, "REMOTE_TARGET_ROOT", fixture.target,
            ))
            stack.enter_context(mock.patch.object(
                stager, "REMOTE_RECEIPT_PATH", fixture.receipt,
            ))
            stack.enter_context(mock.patch.object(
                stager, "LOCAL_COMMIT_TERMINAL_PATH", fixture.local_terminal,
            ))
            stack.enter_context(mock.patch.object(
                stager, "REMOTE_UID", os.geteuid(),
            ))
            stack.enter_context(mock.patch.object(
                stager, "REMOTE_GID", os.getegid(),
            ))
            try:
                yield fixture
            finally:
                fixture.remove_outputs()


@contextmanager
def held_sources(fixture: Fixture):
    held = []
    try:
        for row in fixture.specs:
            held.append(stager._open_local_authority(
                fixture.local / row["relative"],
                sha256=row["sha256"],
                size=row["size"],
            ))
        yield held
    finally:
        for authority in held:
            authority.close()


def payload_bytes(
    fixture: Fixture, *, bootstrap_sha256: str = "a" * 64,
    operation: str = stager.STAGE_OPERATION,
    commit_terminal: dict[str, object] | None = None,
) -> tuple[dict[str, object], bytes, str]:
    with held_sources(fixture) as held:
        manifest = stager._manifest_value()
        payload = stager._payload_value(
            manifest,
            bootstrap_sha256,
            held,
            operation=operation,
            commit_terminal=commit_terminal,
        )
        raw = stager.canonical(payload) + b"\n"
    return payload, raw, sha(raw)


def recovery_terminal(
    fixture: Fixture, *, bootstrap_sha256: str = "a" * 64,
) -> dict[str, object]:
    _payload, stage_raw, stage_sha = payload_bytes(
        fixture, bootstrap_sha256=bootstrap_sha256,
    )
    del _payload, stage_raw
    receipt_raw = fixture.receipt.read_bytes()
    receipt_identity = stager._identity(fixture.receipt.lstat())
    receipt_state = stager._receipt_reservation_state_value(
        receipt_raw, receipt_identity,
    )
    return stager._commit_terminal(
        stager._manifest_value(),
        request_payload_sha256=stage_sha,
        bootstrap_sha256=bootstrap_sha256,
        target_identity=stager._identity(fixture.target.lstat()),
        receipt_reservation_state=receipt_state,
        rename_result="returned_success",
        rename_classification="target_is_held_shadow",
        receipt_phase="sealed_0400_exact",
        receipt_authoritative=True,
        named_target_same_held_inode=True,
        recovery_admissible=True,
    )


def portable_noreplace(fixture: Fixture):
    def publish(
        parent_fd: int,
        shadow_name: str,
        target_name: str,
        shadow_fd: int,
        shadow_anchor: tuple[int, ...],
        reservation,
    ) -> None:
        self_identity = stager._inode_anchor(os.fstat(shadow_fd))
        if self_identity != shadow_anchor:
            raise stager.SourceStageError("portable held shadow differs")
        reservation.require_reserved(parent_fd)
        stager._absent_at(parent_fd, target_name, label="target root")
        shadow = fixture.remote / shadow_name
        target = fixture.remote / target_name
        # Darwin refuses to rename a 0555 directory in this test environment.
        # Production uses one ordinary same-parent POSIX rename; this portable
        # seam changes mode only around that same single syscall.
        shadow.chmod(0o700)
        try:
            os.rename(shadow, target)
        finally:
            if target.exists():
                target.chmod(0o555)
            elif shadow.exists():
                shadow.chmod(0o555)
        if not fixture.target.is_dir():
            raise stager.SourceStageError("portable publication differs")

    return publish


def target_snapshot(target: Path) -> dict[str, tuple[tuple[int, ...], bytes | None]]:
    snapshot: dict[str, tuple[tuple[int, ...], bytes | None]] = {}
    for directory, child_dirs, files in os.walk(target):
        directory_path = Path(directory)
        relative_directory = os.path.relpath(directory_path, target)
        snapshot[relative_directory] = (
            stager._identity(directory_path.lstat()), None,
        )
        for name in child_dirs:
            child = directory_path / name
            relative = os.path.relpath(child, target)
            snapshot[relative] = (stager._identity(child.lstat()), None)
        for name in files:
            child = directory_path / name
            relative = os.path.relpath(child, target)
            info = child.lstat()
            snapshot[relative] = (
                stager._identity(info),
                child.read_bytes() if stat.S_ISREG(info.st_mode) else None,
            )
    return snapshot


class DummyTransportAuthority:
    """A real retained fd with the minimal transport-authority test ABI."""

    def __init__(self, source_fd: int) -> None:
        self.descriptor = os.dup(source_fd)

    def replay(self) -> None:
        if self.descriptor < 0:
            raise RuntimeError("dummy transport authority is closed")
        os.fstat(self.descriptor)

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


class CloseFailureTransportAuthority(DummyTransportAuthority):
    def close(self) -> None:
        super().close()
        raise RuntimeError("injected close failure")


class HoldAndAuthorityTests(unittest.TestCase):
    def test_checked_in_hold_returns_88_before_all_action_seams(self) -> None:
        self.assertNotEqual(stager.CONTROLLER_STATE, stager.READY_STATE)
        with ExitStack() as stack:
            controller = stack.enter_context(mock.patch.object(stager, "controller"))
            opened = stack.enter_context(mock.patch.object(stager.os, "open"))
            mkdir = stack.enter_context(mock.patch.object(stager.os, "mkdir"))
            temporary = stack.enter_context(mock.patch.object(
                stager.tempfile, "TemporaryFile",
            ))
            popen = stack.enter_context(mock.patch.object(
                stager.subprocess, "Popen",
            ))
            self.assertEqual(stager.main(["even-invalid-argv"]), 88)
        controller.assert_not_called()
        opened.assert_not_called()
        mkdir.assert_not_called()
        temporary.assert_not_called()
        popen.assert_not_called()

    def test_final_pin_closure_and_corrupt_pin_refuses_before_io(self) -> None:
        dynamic = {
            "methods/bernini_action_editing/"
            "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py",
            "methods/bernini_action_editing/"
            "case01_object_trajectory_exact5_world4_probe_v1.py",
            "methods/bernini_action_editing/tools/"
            "build_case01_object_trajectory_exact5_source_snapshot_v1.py",
            "methods/bernini_action_editing/tools/"
            "materialize_case01_object_trajectory_exact5_r64_hold_package_v1.py",
        }
        self.assertEqual(stager.blocked_sources(), ())
        by_relative = {
            row["relative"]: row for row in stager.SOURCE_AUTHORITIES
        }
        for relative in dynamic:
            raw = (stager.LOCAL_SOURCE_ROOT / relative).read_bytes()
            self.assertEqual(by_relative[relative]["sha256"], sha(raw))
            self.assertEqual(by_relative[relative]["size"], len(raw))
        bootstrap_raw = BOOTSTRAP.read_bytes()
        self.assertEqual(stager.REMOTE_BOOTSTRAP_PATH, BOOTSTRAP)
        self.assertEqual(stager.REMOTE_BOOTSTRAP_SHA256, sha(bootstrap_raw))
        self.assertEqual(stager.REMOTE_BOOTSTRAP_SIZE, len(bootstrap_raw))
        corrupt = tuple(
            ({**row, "sha256": "__BLOCKED_CORRUPT_PIN__"}
             if row["relative"] == sorted(dynamic)[0] else dict(row))
            for row in stager.SOURCE_AUTHORITIES
        )
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                stager, "CONTROLLER_STATE", stager.READY_STATE,
            ))
            stack.enter_context(mock.patch.object(
                stager, "SOURCE_AUTHORITIES", corrupt,
            ))
            opened = stack.enter_context(mock.patch.object(stager.os, "open"))
            mkdir = stack.enter_context(mock.patch.object(stager.os, "mkdir"))
            temporary = stack.enter_context(mock.patch.object(
                stager.tempfile, "TemporaryFile",
            ))
            popen = stack.enter_context(mock.patch.object(
                stager.subprocess, "Popen",
            ))
            self.assertEqual(stager.main([]), 96)
        opened.assert_not_called()
        mkdir.assert_not_called()
        temporary.assert_not_called()
        popen.assert_not_called()

    def test_remote_loader_and_atomic_source_contract_survive_optimization(self) -> None:
        self.assertNotIn("assert ", stager.REMOTE_LOADER_SOURCE)
        compile(
            stager.REMOTE_LOADER_SOURCE,
            "<held-loader>",
            "exec",
            dont_inherit=True,
            optimize=2,
        )
        source = SCRIPT.read_text(encoding="utf-8")
        bootstrap_source = BOOTSTRAP.read_text(encoding="utf-8")
        protocol = (
            "posix_rename_same_parent_under_held_O_EXCL_"
            "receipt_reservation"
        )
        self.assertIn(protocol, source)
        self.assertIn(protocol, bootstrap_source)
        self.assertNotIn("renameat2", source)
        self.assertNotIn("renameat2", bootstrap_source)
        self.assertIn("os.rename(", source)
        self.assertIn("os.rename(", bootstrap_source)
        self.assertIn("RECEIPT_RESERVATION_MODE = 0o600", source)
        self.assertIn("RECEIPT_RESERVATION_MODE = 0o600", bootstrap_source)
        self.assertIn("'__file__':'/held/repo/methods/", source)
        self.assertIn("def _remote_dispatch(", bootstrap_source)
        self.assertNotIn("def controller(", bootstrap_source)
        self.assertNotIn("subprocess.Popen", bootstrap_source)
        self.assertNotIn("CONTROLLER_STATE", bootstrap_source)

    def test_nonempty_authority_replay_and_special_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "source.py"
            raw = b"nonempty-authority\n"
            path.write_bytes(raw)
            authority = stager._open_local_authority(
                path, sha256=sha(raw), size=len(raw),
            )
            try:
                authority.replay()
            finally:
                authority.close()

            hardlink = root / "hardlink.py"
            os.link(path, hardlink)
            with self.assertRaises(stager.SourceStageError):
                stager._open_local_authority(
                    path, sha256=sha(raw), size=len(raw),
                )
            hardlink.unlink()

            symlink = root / "symlink.py"
            symlink.symlink_to(path)
            with self.assertRaises(stager.SourceStageError):
                stager._open_local_authority(
                    symlink, sha256=sha(raw), size=len(raw),
                )

            if hasattr(os, "mkfifo"):
                fifo = root / "fifo"
                os.mkfifo(fifo)
                with self.assertRaises(stager.SourceStageError):
                    stager._open_local_authority(
                        fifo, sha256="0" * 64, size=1,
                    )


class PayloadTests(unittest.TestCase):
    def test_physical15_manifest_payload_and_held_framing_round_trip(self) -> None:
        with physical15_fixture() as fixture:
            payload, raw, claimed = payload_bytes(fixture)
            parsed, captured = stager._parse_payload(
                raw,
                claimed_sha256=claimed,
                held_bootstrap_sha256="a" * 64,
            )
            self.assertEqual(parsed, payload)
            self.assertEqual(captured, fixture.contents)
            self.assertEqual(len(parsed["files"]), 15)
            self.assertEqual(parsed["manifest"]["file_count"], 15)
            self.assertFalse(parsed["manifest"]["launch_allowed"])
            bootstrap = b"held bootstrap source\n"
            framed = stager._frame_payload(bootstrap, raw)
            self.assertEqual(
                stager._unframe_payload(framed), (bootstrap, raw),
            )

    def test_generic_bootstrap_accepts_controller_rows_and_rejects_row_hostiles(
        self,
    ) -> None:
        source = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertNotIn("SOURCE_AUTHORITIES", source)
        self.assertNotIn("FINAL_BUILDER_SHA256", source)
        self.assertNotIn("FINAL_MATERIALIZER_SHA256", source)
        for row in stager.SOURCE_AUTHORITIES:
            self.assertNotIn(row["sha256"], source)

        with physical15_fixture() as fixture:
            bootstrap_sha256 = sha(BOOTSTRAP.read_bytes())
            payload, raw, claimed = payload_bytes(
                fixture, bootstrap_sha256=bootstrap_sha256,
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    generic_bootstrap, "REMOTE_PARENT", fixture.remote,
                ))
                stack.enter_context(mock.patch.object(
                    generic_bootstrap, "REMOTE_TARGET_ROOT", fixture.target,
                ))
                stack.enter_context(mock.patch.object(
                    generic_bootstrap, "REMOTE_RECEIPT_PATH", fixture.receipt,
                ))
                stack.enter_context(mock.patch.object(
                    generic_bootstrap, "REMOTE_UID", os.geteuid(),
                ))
                stack.enter_context(mock.patch.object(
                    generic_bootstrap, "REMOTE_GID", os.getegid(),
                ))
                parsed, captured, rows = generic_bootstrap._parse_payload(
                    raw,
                    claimed_sha256=claimed,
                    held_bootstrap_sha256=bootstrap_sha256,
                )
                self.assertEqual(parsed, payload)
                self.assertEqual(captured, fixture.contents)
                self.assertEqual(rows, fixture.specs)

                def materialize_hostile(value):
                    value["manifest"].pop("manifest_digest", None)
                    value["manifest"]["manifest_digest"] = (
                        generic_bootstrap.object_digest(value["manifest"])
                    )
                    value.pop("authority_digest", None)
                    value["authority_digest"] = (
                        generic_bootstrap.object_digest(value)
                    )
                    hostile_raw = generic_bootstrap.canonical(value) + b"\n"
                    return hostile_raw, sha(hostile_raw)

                hostile_values = []
                reordered = json.loads(
                    stager.canonical(payload).decode("utf-8"),
                )
                for key in ("files",):
                    reordered["manifest"][key][0], reordered["manifest"][key][1] = (
                        reordered["manifest"][key][1],
                        reordered["manifest"][key][0],
                    )
                    reordered[key][0], reordered[key][1] = (
                        reordered[key][1], reordered[key][0],
                    )
                hostile_values.append(reordered)

                duplicate = json.loads(
                    stager.canonical(payload).decode("utf-8"),
                )
                duplicate["manifest"]["files"][1]["relative"] = (
                    duplicate["manifest"]["files"][0]["relative"]
                )
                duplicate["files"][1]["relative"] = (
                    duplicate["files"][0]["relative"]
                )
                hostile_values.append(duplicate)

                wrong_sha = json.loads(
                    stager.canonical(payload).decode("utf-8"),
                )
                wrong_sha["manifest"]["files"][0]["sha256"] = "f" * 64
                wrong_sha["files"][0]["sha256"] = "f" * 64
                hostile_values.append(wrong_sha)

                wrong_size = json.loads(
                    stager.canonical(payload).decode("utf-8"),
                )
                wrong_size["manifest"]["files"][0]["size"] += 1
                wrong_size["files"][0]["size"] += 1
                hostile_values.append(wrong_size)

                for hostile in hostile_values:
                    hostile_raw, hostile_sha = materialize_hostile(hostile)
                    with self.assertRaises(generic_bootstrap.SourceStageError):
                        generic_bootstrap._parse_payload(
                            hostile_raw,
                            claimed_sha256=hostile_sha,
                            held_bootstrap_sha256=bootstrap_sha256,
                        )

    def test_payload_tamper_noncanonical_duplicate_and_wrong_bootstrap_refused(self) -> None:
        with physical15_fixture() as fixture:
            payload, raw, claimed = payload_bytes(fixture)
            cases: list[tuple[bytes, str, str]] = [
                (raw + b" ", sha(raw + b" "), "a" * 64),
                (
                    b'{"schema_version":1,"schema_version":1}\n',
                    sha(b'{"schema_version":1,"schema_version":1}\n'),
                    "a" * 64,
                ),
                (raw, claimed, "b" * 64),
            ]
            changed = json.loads(stager.canonical(payload).decode("utf-8"))
            changed["files"][0]["content_b64"] = base64.b64encode(
                b"wrong bytes",
            ).decode("ascii")
            unsigned = dict(changed)
            unsigned.pop("authority_digest")
            changed["authority_digest"] = stager.object_digest(unsigned)
            changed_raw = stager.canonical(changed) + b"\n"
            cases.append((changed_raw, sha(changed_raw), "a" * 64))
            for hostile_raw, hostile_sha, bootstrap_sha in cases:
                with self.subTest(hostile_sha=hostile_sha):
                    with self.assertRaises(stager.SourceStageError):
                        stager._parse_payload(
                            hostile_raw,
                            claimed_sha256=hostile_sha,
                            held_bootstrap_sha256=bootstrap_sha,
                        )


class RemoteBootstrapTests(unittest.TestCase):
    def _assert_nonadmissible_reservation(self, fixture: Fixture) -> None:
        self.assertTrue(fixture.receipt.is_file())
        self.assertEqual(
            stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o600,
        )
        value = stager._strict_json(
            fixture.receipt.read_bytes(), label="test receipt reservation",
        )
        self.assertEqual(value["status"], "RESERVED_NOT_ADMISSION")
        self.assertFalse(value["receipt_is_admission"])

    def _run(self, fixture: Fixture) -> dict[str, object]:
        _payload, raw, claimed = payload_bytes(fixture)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                stager, "_validate_remote_runtime",
            ))
            stack.enter_context(mock.patch.object(
                stager,
                "_rename_under_receipt_reservation",
                side_effect=portable_noreplace(fixture),
            ))
            return stager._remote_bootstrap(raw, claimed, "a" * 64)

    def _recover(
        self,
        fixture: Fixture,
        terminal: dict[str, object] | None = None,
        *,
        bootstrap_sha256: str = "a" * 64,
    ) -> dict[str, object]:
        if terminal is None:
            terminal = recovery_terminal(
                fixture, bootstrap_sha256=bootstrap_sha256,
            )
        _payload, raw, claimed = payload_bytes(
            fixture,
            bootstrap_sha256=bootstrap_sha256,
            operation=stager.RECOVER_RECEIPT_OPERATION,
            commit_terminal=terminal,
        )
        with mock.patch.object(stager, "_validate_remote_runtime"):
            return stager._remote_bootstrap(
                raw, claimed, bootstrap_sha256,
            )

    def test_success_is_exact15_sealed_create_only_and_read_back(self) -> None:
        with physical15_fixture() as fixture:
            receipt = self._run(fixture)
            self.assertEqual(receipt["status"], "STAGED_RECEIPT_GATED")
            self.assertEqual(receipt["file_count"], 15)
            self.assertFalse(receipt["launch_allowed"])
            self.assertNotIn("target_replaced", receipt)
            self.assertEqual(receipt["operation"], stager.STAGE_OPERATION)
            self.assertEqual(
                receipt["target_observation"]["kind"],
                "live_posix_rename_under_held_receipt_reservation",
            )
            self.assertEqual(
                receipt["target_observation"]["historical_replacement_claim"],
                "not_made",
            )
            self.assertFalse(receipt["rename_noreplace"])
            self.assertTrue(receipt["receipt_is_admission"])
            self.assertTrue(receipt["receipt_is_consumption_gate"])
            self.assertTrue(receipt["ancestor_chain_nofollow"])
            unsigned = dict(receipt)
            claimed = unsigned.pop("receipt_digest")
            self.assertEqual(claimed, stager.object_digest(unsigned))

            self.assertEqual(
                fixture.receipt.read_bytes(), stager.canonical(receipt) + b"\n",
            )
            receipt_info = fixture.receipt.lstat()
            self.assertEqual(stat.S_IMODE(receipt_info.st_mode), 0o400)
            self.assertEqual(receipt_info.st_nlink, 1)

            observed_files: dict[str, bytes] = {}
            observed_directories = {"."}
            for directory, child_dirs, files in os.walk(fixture.target):
                relative_dir = os.path.relpath(directory, fixture.target)
                observed_directories.add(relative_dir)
                self.assertEqual(stat.S_IMODE(os.lstat(directory).st_mode), 0o555)
                for name in child_dirs:
                    relative = os.path.relpath(
                        Path(directory) / name, fixture.target,
                    )
                    observed_directories.add(relative)
                for name in files:
                    path = Path(directory) / name
                    relative = os.path.relpath(path, fixture.target)
                    info = path.lstat()
                    self.assertEqual(stat.S_IMODE(info.st_mode), 0o444)
                    self.assertEqual(info.st_nlink, 1)
                    observed_files[relative] = path.read_bytes()
            self.assertEqual(observed_files, fixture.contents)
            self.assertEqual(
                observed_directories,
                set(stager.expected_directories(fixture.specs)),
            )
            self.assertEqual(
                {entry.name for entry in fixture.remote.iterdir()},
                {fixture.target.name, fixture.receipt.name},
            )

    def test_committed_target_with_0600_reservation_recovers_same_inode(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_rename_under_receipt_reservation",
                    side_effect=portable_noreplace(fixture),
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_seal_reserved_receipt",
                    side_effect=stager.SourceStageError(
                        "injected post-rename receipt seal failure"
                    ),
                ))
                terminal = stager._remote_bootstrap(raw, claimed, "a" * 64)
            self.assertEqual(
                terminal["status"],
                "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
            )
            self.assertTrue(fixture.target.is_dir())
            self.assertTrue(fixture.receipt.is_file())
            self.assertEqual(
                stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o600,
            )
            reserved_anchor = stager._inode_anchor(fixture.receipt.lstat())
            reserved = stager._strict_json(
                fixture.receipt.read_bytes(), label="test reservation",
            )
            self.assertEqual(reserved["status"], "RESERVED_NOT_ADMISSION")
            self.assertFalse(reserved["receipt_is_admission"])
            before_target = target_snapshot(fixture.target)

            with ExitStack() as stack:
                build = stack.enter_context(mock.patch.object(
                    stager, "_build_shadow_at",
                ))
                rename = stack.enter_context(mock.patch.object(
                    stager, "_rename_under_receipt_reservation",
                ))
                mkdir = stack.enter_context(mock.patch.object(
                    stager.os, "mkdir",
                ))
                recovered = self._recover(fixture, terminal)
            build.assert_not_called()
            rename.assert_not_called()
            mkdir.assert_not_called()
            self.assertEqual(
                recovered["status"], "RECOVERED_RECEIPT_ONLY",
            )
            self.assertEqual(
                stager._inode_anchor(fixture.receipt.lstat()), reserved_anchor,
            )
            self.assertEqual(
                recovered["target_observation"]["historical_replacement_claim"],
                "not_made",
            )
            self.assertFalse(
                recovered["target_observation"]
                ["rename_noreplace_performed_this_operation"]
            )
            self.assertEqual(target_snapshot(fixture.target), before_target)
            receipt_before = (
                stager._identity(fixture.receipt.lstat()),
                fixture.receipt.read_bytes(),
            )
            with mock.patch.object(stager, "_seal_reserved_receipt") as write:
                verified = self._recover(fixture, terminal)
            write.assert_not_called()
            self.assertEqual(verified, recovered)
            self.assertEqual(
                (
                    stager._identity(fixture.receipt.lstat()),
                    fixture.receipt.read_bytes(),
                ),
                receipt_before,
            )
            self.assertEqual(target_snapshot(fixture.target), before_target)

    def test_postrename_partial_write_and_fchmod_failures_recover_same_inode(
        self,
    ) -> None:
        for seam in ("partial_pwrite", "precommit_fchmod"):
            with self.subTest(seam=seam), physical15_fixture() as fixture:
                _payload, raw, claimed = payload_bytes(fixture)
                real_seal = stager._seal_reserved_receipt
                real_pwrite = stager.os.pwrite
                real_fchmod = stager.os.fchmod

                def fail_during_seal(
                    parent_fd, reservation, value,
                    *, expected_prior_state=None,
                ):
                    if seam == "partial_pwrite":
                        write_count = 0

                        def partial_then_error(descriptor, content, offset):
                            nonlocal write_count
                            write_count += 1
                            if write_count == 1:
                                prefix = max(1, len(content) // 2)
                                return real_pwrite(
                                    descriptor, content[:prefix], offset,
                                )
                            raise OSError(errno.EIO, "injected receipt pwrite")

                        patcher = mock.patch.object(
                            stager.os, "pwrite", side_effect=partial_then_error,
                        )
                    else:
                        chmod_count = 0

                        def reservation_then_error(descriptor, mode):
                            nonlocal chmod_count
                            chmod_count += 1
                            if chmod_count == 2:
                                raise OSError(errno.EIO, "injected receipt fchmod")
                            return real_fchmod(descriptor, mode)

                        patcher = mock.patch.object(
                            stager.os, "fchmod", side_effect=reservation_then_error,
                        )
                    with patcher:
                        return real_seal(
                            parent_fd,
                            reservation,
                            value,
                            expected_prior_state=expected_prior_state,
                        )

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager, "_validate_remote_runtime",
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_rename_under_receipt_reservation",
                        side_effect=portable_noreplace(fixture),
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_seal_reserved_receipt",
                        side_effect=fail_during_seal,
                    ))
                    terminal = stager._remote_bootstrap(
                        raw, claimed, "a" * 64,
                    )

                self.assertEqual(
                    terminal["status"],
                    "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
                )
                self.assertEqual(terminal["receipt_phase"], "partial_0600")
                self.assertTrue(terminal["recovery_admissible"])
                self.assertEqual(
                    stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o600,
                )
                state = terminal["receipt_reservation_state"]
                self.assertEqual(state["size"], len(fixture.receipt.read_bytes()))
                self.assertEqual(state["sha256"], sha(fixture.receipt.read_bytes()))
                receipt_anchor = stager._inode_anchor(fixture.receipt.lstat())
                before_target = target_snapshot(fixture.target)

                recovered = self._recover(fixture, terminal)
                self.assertEqual(recovered["status"], "RECOVERED_RECEIPT_ONLY")
                self.assertEqual(
                    stager._inode_anchor(fixture.receipt.lstat()), receipt_anchor,
                )
                self.assertEqual(
                    stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o400,
                )
                self.assertEqual(target_snapshot(fixture.target), before_target)

    def test_parent_fsync_failure_after_rename_enters_recoverable_terminal(
        self,
    ) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            real_fsync = stager.os.fsync
            failed_after_commit = False

            def fsync_or_fail(descriptor):
                nonlocal failed_after_commit
                if (
                    not failed_after_commit
                    and fixture.target.is_dir()
                    and fixture.receipt.is_file()
                    and stat.S_IMODE(fixture.receipt.lstat().st_mode) == 0o600
                ):
                    failed_after_commit = True
                    raise OSError(errno.EIO, "injected post-rename parent fsync")
                return real_fsync(descriptor)

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_rename_under_receipt_reservation",
                    side_effect=portable_noreplace(fixture),
                ))
                stack.enter_context(mock.patch.object(
                    stager.os, "fsync", side_effect=fsync_or_fail,
                ))
                terminal = stager._remote_bootstrap(raw, claimed, "a" * 64)

            self.assertTrue(failed_after_commit)
            self.assertEqual(
                terminal["status"],
                "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
            )
            self.assertEqual(terminal["rename_result"], "returned_success")
            self.assertEqual(
                terminal["rename_classification"], "target_is_held_shadow",
            )
            self.assertEqual(terminal["receipt_phase"], "reserved_0600")
            self.assertTrue(terminal["recovery_admissible"])
            self._assert_nonadmissible_reservation(fixture)
            self.assertFalse(any(
                child.name.startswith("." + fixture.target.name + ".shadow-")
                for child in fixture.remote.iterdir()
            ))
            receipt_anchor = stager._inode_anchor(fixture.receipt.lstat())
            before_target = target_snapshot(fixture.target)

            recovered = self._recover(fixture, terminal)
            self.assertEqual(recovered["status"], "RECOVERED_RECEIPT_ONLY")
            self.assertEqual(
                stager._inode_anchor(fixture.receipt.lstat()), receipt_anchor,
            )
            self.assertEqual(target_snapshot(fixture.target), before_target)

    def test_recovery_rejects_mode000_and_never_rewrites_malformed_0400(
        self,
    ) -> None:
        for hostile in ("mode000", "malformed0400"):
            with self.subTest(hostile=hostile), physical15_fixture() as fixture:
                _payload, raw, claimed = payload_bytes(fixture)
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager, "_validate_remote_runtime",
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_rename_under_receipt_reservation",
                        side_effect=portable_noreplace(fixture),
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_seal_reserved_receipt",
                        side_effect=stager.SourceStageError(
                            "injected preseal failure",
                        ),
                    ))
                    terminal = stager._remote_bootstrap(
                        raw, claimed, "a" * 64,
                    )
                self.assertEqual(terminal["receipt_phase"], "reserved_0600")
                before_target = target_snapshot(fixture.target)
                if hostile == "mode000":
                    fixture.receipt.chmod(0o000)
                    before_receipt = stager._identity(fixture.receipt.lstat())
                else:
                    fixture.receipt.write_bytes(b"malformed-final-receipt\n")
                    fixture.receipt.chmod(0o400)
                    before_receipt = (
                        stager._identity(fixture.receipt.lstat()),
                        fixture.receipt.read_bytes(),
                    )
                with ExitStack() as stack:
                    seal = stack.enter_context(mock.patch.object(
                        stager, "_seal_reserved_receipt",
                    ))
                    truncate = stack.enter_context(mock.patch.object(
                        stager.os, "ftruncate",
                    ))
                    write = stack.enter_context(mock.patch.object(
                        stager.os, "pwrite",
                    ))
                    chmod = stack.enter_context(mock.patch.object(
                        stager.os, "fchmod",
                    ))
                    with self.assertRaises(stager.SourceStageError):
                        self._recover(fixture, terminal)
                seal.assert_not_called()
                truncate.assert_not_called()
                write.assert_not_called()
                chmod.assert_not_called()
                self.assertEqual(target_snapshot(fixture.target), before_target)
                if hostile == "mode000":
                    self.assertEqual(
                        stager._identity(fixture.receipt.lstat()), before_receipt,
                    )
                    self.assertEqual(
                        stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o000,
                    )
                else:
                    self.assertEqual(
                        (
                            stager._identity(fixture.receipt.lstat()),
                            fixture.receipt.read_bytes(),
                        ),
                        before_receipt,
                    )

    def test_recovery_mutation_failures_emit_fresh_same_inode_terminal(
        self,
    ) -> None:
        for seam in ("partial_pwrite", "pre0400_fsync", "pre0400_fchmod"):
            with self.subTest(seam=seam), physical15_fixture() as fixture:
                _payload, raw, claimed = payload_bytes(fixture)
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager, "_validate_remote_runtime",
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_rename_under_receipt_reservation",
                        side_effect=portable_noreplace(fixture),
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_seal_reserved_receipt",
                        side_effect=stager.SourceStageError(
                            "injected initial reservation terminal",
                        ),
                    ))
                    initial = stager._remote_bootstrap(
                        raw, claimed, "a" * 64,
                    )
                self.assertEqual(initial["receipt_phase"], "reserved_0600")
                original_anchor = stager._inode_anchor(fixture.receipt.lstat())
                before_target = target_snapshot(fixture.target)
                real_seal = stager._seal_reserved_receipt
                real_pwrite = stager.os.pwrite
                real_fsync = stager.os.fsync
                real_fchmod = stager.os.fchmod

                def fail_recovery_seal(
                    parent_fd, reservation, value,
                    *, expected_prior_state=None,
                ):
                    calls = 0

                    if seam == "partial_pwrite":
                        def operation(descriptor, content, offset):
                            nonlocal calls
                            calls += 1
                            if calls == 1:
                                prefix = max(1, len(content) // 2)
                                return real_pwrite(
                                    descriptor, content[:prefix], offset,
                                )
                            raise OSError(
                                errno.EIO, "injected recovery pwrite",
                            )
                        patcher = mock.patch.object(
                            stager.os, "pwrite", side_effect=operation,
                        )
                    elif seam == "pre0400_fsync":
                        def operation(descriptor):
                            nonlocal calls
                            calls += 1
                            if calls == 1:
                                raise OSError(
                                    errno.EIO, "injected recovery fsync",
                                )
                            return real_fsync(descriptor)
                        patcher = mock.patch.object(
                            stager.os, "fsync", side_effect=operation,
                        )
                    else:
                        def operation(descriptor, mode):
                            nonlocal calls
                            calls += 1
                            if calls == 2:
                                raise OSError(
                                    errno.EIO, "injected recovery fchmod",
                                )
                            return real_fchmod(descriptor, mode)
                        patcher = mock.patch.object(
                            stager.os, "fchmod", side_effect=operation,
                        )
                    with patcher:
                        return real_seal(
                            parent_fd,
                            reservation,
                            value,
                            expected_prior_state=expected_prior_state,
                        )

                with mock.patch.object(
                    stager,
                    "_seal_reserved_receipt",
                    side_effect=fail_recovery_seal,
                ):
                    refreshed = self._recover(fixture, initial)
                self.assertEqual(
                    refreshed["status"],
                    "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
                )
                self.assertEqual(refreshed["receipt_phase"], "partial_0600")
                self.assertTrue(refreshed["recovery_admissible"])
                self.assertNotEqual(
                    refreshed["terminal_digest"], initial["terminal_digest"],
                )
                self.assertEqual(
                    stager._inode_anchor(fixture.receipt.lstat()),
                    original_anchor,
                )
                self.assertEqual(target_snapshot(fixture.target), before_target)

                with self.assertRaises(stager.SourceStageError):
                    self._recover(fixture, initial)
                recovered = self._recover(fixture, refreshed)
                self.assertEqual(recovered["status"], "RECOVERED_RECEIPT_ONLY")
                self.assertEqual(
                    stager._inode_anchor(fixture.receipt.lstat()),
                    original_anchor,
                )
                self.assertEqual(target_snapshot(fixture.target), before_target)

    def test_postseal_replay_errors_return_bound_receipt_then_verify_only(
        self,
    ) -> None:
        for seam in ("receipt_replay", "target_check"):
            with self.subTest(seam=seam), physical15_fixture() as fixture:
                _payload, raw, claimed = payload_bytes(fixture)
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager, "_validate_remote_runtime",
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_rename_under_receipt_reservation",
                        side_effect=portable_noreplace(fixture),
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_seal_reserved_receipt",
                        side_effect=stager.SourceStageError(
                            "injected initial reservation terminal",
                        ),
                    ))
                    initial = stager._remote_bootstrap(
                        raw, claimed, "a" * 64,
                    )
                receipt_anchor = stager._inode_anchor(fixture.receipt.lstat())
                before_target = target_snapshot(fixture.target)

                if seam == "receipt_replay":
                    real_replay = stager.HeldReceipt.replay
                    calls = 0

                    def fail_first_replay(held, parent_fd):
                        nonlocal calls
                        calls += 1
                        if calls == 1:
                            raise stager.SourceStageError(
                                "injected postseal receipt replay",
                            )
                        return real_replay(held, parent_fd)

                    patcher = mock.patch.object(
                        stager.HeldReceipt,
                        "replay",
                        side_effect=fail_first_replay,
                        autospec=True,
                    )
                else:
                    real_assert_named = stager._assert_named_inode
                    target_checks = 0

                    def fail_second_target_check(*args, **kwargs):
                        nonlocal target_checks
                        if kwargs.get("label") == "recovery target root":
                            target_checks += 1
                            if target_checks == 2:
                                raise stager.SourceStageError(
                                    "injected postseal target check",
                                )
                        return real_assert_named(*args, **kwargs)

                    patcher = mock.patch.object(
                        stager,
                        "_assert_named_inode",
                        side_effect=fail_second_target_check,
                    )
                with patcher:
                    recovered = self._recover(fixture, initial)
                self.assertEqual(recovered["status"], "RECOVERED_RECEIPT_ONLY")
                self.assertEqual(
                    recovered["commit_terminal_digest"],
                    initial["terminal_digest"],
                )
                self.assertEqual(
                    stager._inode_anchor(fixture.receipt.lstat()),
                    receipt_anchor,
                )
                self.assertEqual(
                    stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o400,
                )
                self.assertEqual(target_snapshot(fixture.target), before_target)

                with mock.patch.object(stager, "_seal_reserved_receipt") as seal:
                    verified = self._recover(fixture, initial)
                seal.assert_not_called()
                self.assertEqual(verified, recovered)
                self.assertEqual(
                    stager._inode_anchor(fixture.receipt.lstat()),
                    receipt_anchor,
                )
                self.assertEqual(target_snapshot(fixture.target), before_target)

    def test_post_receipt_commit_terminal_recovers_by_exact_live_verify_only(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_rename_under_receipt_reservation",
                    side_effect=portable_noreplace(fixture),
                ))
                stack.enter_context(mock.patch.object(
                    stager.HeldReceipt,
                    "replay",
                    side_effect=stager.SourceStageError(
                        "injected post-receipt readback failure"
                    ),
                ))
                terminal = stager._remote_bootstrap(raw, claimed, "a" * 64)
            self.assertEqual(
                terminal["status"],
                "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
            )
            before_target = target_snapshot(fixture.target)
            before_receipt = (
                stager._identity(fixture.receipt.lstat()),
                fixture.receipt.read_bytes(),
            )
            with mock.patch.object(stager, "_seal_reserved_receipt") as write:
                receipt = self._recover(fixture, terminal)
            write.assert_not_called()
            self.assertEqual(receipt["operation"], stager.STAGE_OPERATION)
            self.assertEqual(receipt["status"], "STAGED_RECEIPT_GATED")
            self.assertEqual(target_snapshot(fixture.target), before_target)
            self.assertEqual(
                (
                    stager._identity(fixture.receipt.lstat()),
                    fixture.receipt.read_bytes(),
                ),
                before_receipt,
            )

    def test_recovery_refuses_target_and_receipt_hostiles_without_mutation(self) -> None:
        for mutation in ("tamper", "extra", "fifo"):
            with self.subTest(mutation=mutation), physical15_fixture() as fixture:
                self._run(fixture)
                terminal = recovery_terminal(fixture)
                fixture.receipt.chmod(0o600)
                fixture.receipt.unlink()
                first = fixture.target / fixture.specs[0]["relative"]
                if mutation == "tamper":
                    first.chmod(0o600)
                    first.write_bytes(b"x" * len(first.read_bytes()))
                    first.chmod(0o444)
                else:
                    fixture.target.chmod(0o700)
                    try:
                        if mutation == "extra":
                            extra = fixture.target / "extra"
                            extra.write_bytes(b"extra")
                            extra.chmod(0o444)
                        else:
                            if not hasattr(os, "mkfifo"):
                                raise unittest.SkipTest("mkfifo unavailable")
                            os.mkfifo(fixture.target / "fifo")
                    finally:
                        fixture.target.chmod(0o555)
                hostile_snapshot = target_snapshot(fixture.target)
                with self.assertRaises(stager.SourceStageError):
                    self._recover(fixture, terminal)
                self.assertFalse(fixture.receipt.exists())
                self.assertEqual(target_snapshot(fixture.target), hostile_snapshot)

        with physical15_fixture() as fixture:
            self._run(fixture)
            terminal = recovery_terminal(fixture)
            fixture.receipt.chmod(0o600)
            fixture.receipt.unlink()
            fixture.receipt.write_bytes(b"competitor\n")
            fixture.receipt.chmod(0o400)
            competitor = (
                stager._identity(fixture.receipt.lstat()),
                fixture.receipt.read_bytes(),
            )
            before_target = target_snapshot(fixture.target)
            with self.assertRaises(stager.SourceStageError):
                self._recover(fixture, terminal)
            self.assertEqual(
                (
                    stager._identity(fixture.receipt.lstat()),
                    fixture.receipt.read_bytes(),
                ),
                competitor,
            )
            self.assertEqual(target_snapshot(fixture.target), before_target)

        for replacement in ("target", "receipt"):
            with self.subTest(replacement=replacement), physical15_fixture() as fixture:
                self._run(fixture)
                terminal = recovery_terminal(fixture)
                before_target_bytes = {
                    key: raw for key, (_identity, raw)
                    in target_snapshot(fixture.target).items()
                }
                if replacement == "target":
                    original = fixture.remote / ".original-target"
                    fixture.target.chmod(0o700)
                    fixture.target.rename(original)
                    original.chmod(0o555)
                    shutil.copytree(original, fixture.target)
                    original_identity = stager._identity(original.lstat())
                    self.assertNotEqual(
                        stager._identity(fixture.target.lstat()),
                        original_identity,
                    )
                else:
                    original = fixture.remote / ".original-receipt"
                    fixture.receipt.rename(original)
                    fixture.receipt.write_bytes(original.read_bytes())
                    fixture.receipt.chmod(0o400)
                with self.assertRaises(stager.SourceStageError):
                    self._recover(fixture, terminal)
                after_target_bytes = {
                    key: raw for key, (_identity, raw)
                    in target_snapshot(fixture.target).items()
                }
                self.assertEqual(after_target_bytes, before_target_bytes)

    def test_preexisting_target_or_receipt_is_never_replaced(self) -> None:
        for collision in ("target", "receipt"):
            with self.subTest(collision=collision), physical15_fixture() as fixture:
                payload, raw, claimed = payload_bytes(fixture)
                del payload
                occupied = fixture.target if collision == "target" else fixture.receipt
                if collision == "target":
                    occupied.mkdir()
                    marker = occupied / "marker"
                    marker.write_bytes(b"preexisting")
                else:
                    occupied.write_bytes(b"preexisting")
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager, "_validate_remote_runtime",
                    ))
                    rename = stack.enter_context(mock.patch.object(
                        stager, "_rename_under_receipt_reservation",
                    ))
                    with self.assertRaises(stager.SourceStageError):
                        stager._remote_bootstrap(raw, claimed, "a" * 64)
                rename.assert_not_called()
                if collision == "target":
                    self.assertEqual(marker.read_bytes(), b"preexisting")
                    self.assertFalse(fixture.receipt.exists())
                else:
                    self.assertEqual(occupied.read_bytes(), b"preexisting")
                    self.assertFalse(fixture.target.exists())

    def test_symlink_parent_is_rejected_before_shadow_creation(self) -> None:
        with physical15_fixture() as fixture:
            alias = fixture.base / "remote-alias"
            alias.symlink_to(fixture.remote, target_is_directory=True)
            target = alias / fixture.target.name
            receipt = alias / fixture.receipt.name
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "REMOTE_PARENT", alias,
                ))
                stack.enter_context(mock.patch.object(
                    stager, "REMOTE_TARGET_ROOT", target,
                ))
                stack.enter_context(mock.patch.object(
                    stager, "REMOTE_RECEIPT_PATH", receipt,
                ))
                _payload, raw, claimed = payload_bytes(fixture)
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                rename = stack.enter_context(mock.patch.object(
                    stager, "_rename_under_receipt_reservation",
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._remote_bootstrap(raw, claimed, "a" * 64)
                rename.assert_not_called()
            self.assertEqual(tuple(fixture.remote.iterdir()), ())

    def test_partial_build_and_exact_tree_injections_leave_no_publication(self) -> None:
        mutations = (
            "write_failure", "missing", "tamper", "bad_mode", "extra",
            "symlink", "hardlink", "fifo",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), physical15_fixture() as fixture:
                _payload, raw, claimed = payload_bytes(fixture)
                real_build = stager._build_shadow_at
                real_write = stager._write_file_at
                calls = 0

                def failing_write(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 4:
                        raise stager.SourceStageError("injected write failure")
                    return real_write(*args, **kwargs)

                def mutating_build(shadow_fd, captured, creation_anchors):
                    sealed = real_build(
                        shadow_fd, captured, creation_anchors,
                    )
                    shadow_inode = os.fstat(shadow_fd).st_ino
                    root = next(
                        path for path in fixture.remote.iterdir()
                        if path.lstat().st_ino == shadow_inode
                    )
                    first = root / fixture.specs[0]["relative"]
                    root.chmod(0o700)
                    try:
                        if mutation == "missing":
                            first.parent.chmod(0o700)
                            try:
                                first.unlink()
                            finally:
                                first.parent.chmod(0o555)
                        elif mutation == "tamper":
                            first.chmod(0o600)
                            first.write_bytes(b"x" * len(first.read_bytes()))
                            first.chmod(0o444)
                        elif mutation == "bad_mode":
                            first.chmod(0o644)
                        elif mutation == "extra":
                            extra = root / "extra"
                            extra.write_bytes(b"extra")
                            extra.chmod(0o444)
                        elif mutation == "symlink":
                            (root / "linked").symlink_to("missing")
                        elif mutation == "hardlink":
                            os.link(first, root / "linked")
                        elif mutation == "fifo":
                            if not hasattr(os, "mkfifo"):
                                raise unittest.SkipTest("mkfifo unavailable")
                            os.mkfifo(root / "fifo")
                    finally:
                        root.chmod(0o555)
                    return sealed

                build_effect = (
                    real_build if mutation == "write_failure" else mutating_build
                )
                patches = [
                    mock.patch.object(stager, "_validate_remote_runtime"),
                    mock.patch.object(
                        stager, "_build_shadow_at", side_effect=build_effect,
                    ),
                    mock.patch.object(stager, "_rename_under_receipt_reservation"),
                ]
                if mutation == "write_failure":
                    patches.append(mock.patch.object(
                        stager, "_write_file_at", side_effect=failing_write,
                    ))
                with ExitStack() as stack:
                    active = [stack.enter_context(patch) for patch in patches]
                    with self.assertRaises(stager.SourceStageError):
                        stager._remote_bootstrap(raw, claimed, "a" * 64)
                active[2].assert_not_called()
                self.assertFalse(fixture.target.exists())
                if mutation in (
                    "missing", "extra", "symlink", "hardlink", "fifo",
                ):
                    leftovers = tuple(fixture.remote.iterdir())
                    self.assertEqual(len(leftovers), 2)
                    self.assertTrue(fixture.receipt.is_file())
                    self.assertEqual(
                        stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o600,
                    )
                    self.assertEqual(
                        len([
                            path for path in leftovers
                            if path.name.startswith(
                                ".physical15-target.shadow-"
                            )
                        ]),
                        1,
                    )
                else:
                    self.assertFalse(fixture.receipt.exists())
                    self.assertEqual(tuple(fixture.remote.iterdir()), ())

    def test_atomic_collision_keeps_competing_target_and_removes_shadow(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)

            def collision(
                _parent_fd, _shadow_name, _target_name, _shadow_fd,
                _shadow_anchor, _reservation,
            ):
                fixture.target.mkdir()
                (fixture.target / "marker").write_bytes(b"competitor")
                raise stager.SourceStageError("atomic publication target appeared")

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager, "_rename_under_receipt_reservation", side_effect=collision,
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._remote_bootstrap(raw, claimed, "a" * 64)
            self.assertEqual(
                (fixture.target / "marker").read_bytes(), b"competitor",
            )
            self.assertTrue(fixture.receipt.exists())
            self.assertEqual(
                stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o600,
            )
            self.assertEqual(len(tuple(fixture.remote.iterdir())), 3)

    def test_shadow_root_replacement_is_preserved_and_refused(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            real_build = stager._build_shadow_at
            replacement_marker = b"foreign-shadow-root"

            def replace_root(shadow_fd, captured, creation_anchors):
                sealed = real_build(shadow_fd, captured, creation_anchors)
                inode = os.fstat(shadow_fd).st_ino
                root = next(
                    path for path in fixture.remote.iterdir()
                    if path.lstat().st_ino == inode
                )
                original = fixture.remote / ".held-original-shadow-root"
                root.chmod(0o700)
                root.rename(original)
                original.chmod(0o555)
                root.mkdir(mode=0o700)
                marker = root / "replacement-marker"
                marker.write_bytes(replacement_marker)
                marker.chmod(0o444)
                root.chmod(0o555)
                return sealed

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager, "_build_shadow_at", side_effect=replace_root,
                ))
                rename = stack.enter_context(mock.patch.object(
                    stager, "_rename_under_receipt_reservation",
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._remote_bootstrap(raw, claimed, "a" * 64)
            rename.assert_not_called()
            replacement = next(
                path for path in fixture.remote.iterdir()
                if path.name.startswith(".physical15-target.shadow-")
            )
            self.assertEqual(
                (replacement / "replacement-marker").read_bytes(),
                replacement_marker,
            )
            self.assertFalse(fixture.target.exists())
            self._assert_nonadmissible_reservation(fixture)

    def test_child_byte_identical_replacement_is_preserved_and_refused(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            real_build = stager._build_shadow_at
            relative = fixture.specs[0]["relative"]

            def replace_child(shadow_fd, captured, creation_anchors):
                sealed = real_build(shadow_fd, captured, creation_anchors)
                inode = os.fstat(shadow_fd).st_ino
                root = next(
                    path for path in fixture.remote.iterdir()
                    if path.lstat().st_ino == inode
                )
                child = root / relative
                original = fixture.remote / ".held-original-child"
                child.parent.chmod(0o700)
                try:
                    child.rename(original)
                    child.write_bytes(fixture.contents[relative])
                    child.chmod(0o444)
                finally:
                    child.parent.chmod(0o555)
                return sealed

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager, "_build_shadow_at", side_effect=replace_child,
                ))
                rename = stack.enter_context(mock.patch.object(
                    stager, "_rename_under_receipt_reservation",
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._remote_bootstrap(raw, claimed, "a" * 64)
            rename.assert_not_called()
            replacement_root = next(
                path for path in fixture.remote.iterdir()
                if path.name.startswith(".physical15-target.shadow-")
            )
            replacement = replacement_root / relative
            self.assertEqual(replacement.read_bytes(), fixture.contents[relative])
            self.assertNotEqual(
                replacement.lstat().st_ino,
                (fixture.remote / ".held-original-child").lstat().st_ino,
            )
            self.assertFalse(fixture.target.exists())
            self._assert_nonadmissible_reservation(fixture)

    def test_child_move_away_is_retained_and_cleanup_refuses(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            real_build = stager._build_shadow_at
            relative = fixture.specs[0]["relative"]
            moved = fixture.remote / ".moved-owned-child"

            def move_child(shadow_fd, captured, creation_anchors):
                sealed = real_build(shadow_fd, captured, creation_anchors)
                inode = os.fstat(shadow_fd).st_ino
                root = next(
                    path for path in fixture.remote.iterdir()
                    if path.lstat().st_ino == inode
                )
                child = root / relative
                child.parent.chmod(0o700)
                try:
                    child.rename(moved)
                finally:
                    child.parent.chmod(0o555)
                return sealed

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager, "_build_shadow_at", side_effect=move_child,
                ))
                rename = stack.enter_context(mock.patch.object(
                    stager, "_rename_under_receipt_reservation",
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._remote_bootstrap(raw, claimed, "a" * 64)
            rename.assert_not_called()
            self.assertEqual(moved.read_bytes(), fixture.contents[relative])
            self.assertTrue(any(
                path.name.startswith(".physical15-target.shadow-")
                for path in fixture.remote.iterdir()
            ))
            self.assertFalse(fixture.target.exists())
            self._assert_nonadmissible_reservation(fixture)

    def test_mkdir_open_failures_never_claim_unknown_named_inodes(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_open_shadow_root",
                    side_effect=OSError(errno.EMFILE, "injected root open"),
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._remote_bootstrap(raw, claimed, "a" * 64)
            unknown = tuple(fixture.remote.iterdir())
            self.assertEqual(len(unknown), 2)
            shadow = next(
                path for path in unknown
                if path.name.startswith(".physical15-target.shadow-")
            )
            self.assertEqual(tuple(shadow.iterdir()), ())
            self.assertFalse(fixture.target.exists())
            self._assert_nonadmissible_reservation(fixture)

        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            original = fixture.remote / ".held-original-unopened-shadow"
            replacement_identity = None

            def replace_then_fail(parent_fd, name):
                nonlocal replacement_identity
                del parent_fd
                root = fixture.remote / name
                root.rename(original)
                root.mkdir(mode=0o700)
                replacement_identity = stager._identity(root.lstat())
                raise OSError(errno.EMFILE, "injected replacement open")

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_open_shadow_root",
                    side_effect=replace_then_fail,
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._remote_bootstrap(raw, claimed, "a" * 64)
            replacement = next(
                path for path in fixture.remote.iterdir()
                if path.name.startswith(".physical15-target.shadow-")
            )
            self.assertEqual(
                stager._identity(replacement.lstat()), replacement_identity,
            )
            self.assertEqual(tuple(replacement.iterdir()), ())
            self.assertTrue(original.is_dir())
            self.assertFalse(fixture.target.exists())
            self._assert_nonadmissible_reservation(fixture)

        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            original = fixture.remote / ".held-original-unopened-child"
            replacement_identity = None

            def replace_child_then_fail(parent_fd, name):
                nonlocal replacement_identity
                del parent_fd
                shadow = next(
                    path for path in fixture.remote.iterdir()
                    if path.name.startswith(".physical15-target.shadow-")
                )
                child = shadow / name
                child.rename(original)
                child.mkdir(mode=0o700)
                replacement_identity = stager._identity(child.lstat())
                raise OSError(errno.EMFILE, "injected child replacement open")

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_open_created_child_directory",
                    side_effect=replace_child_then_fail,
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._remote_bootstrap(raw, claimed, "a" * 64)
            shadow = next(
                path for path in fixture.remote.iterdir()
                if path.name.startswith(".physical15-target.shadow-")
            )
            replacement = shadow / "artifacts"
            self.assertEqual(
                stager._identity(replacement.lstat()), replacement_identity,
            )
            self.assertEqual(tuple(replacement.iterdir()), ())
            self.assertTrue(original.is_dir())
            self.assertFalse(fixture.target.exists())
            self._assert_nonadmissible_reservation(fixture)

    def test_descriptor_reserve_failure_occurs_before_mkdir(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_reserve_creation_descriptor",
                    side_effect=stager.SourceStageError(
                        "injected descriptor reserve failure"
                    ),
                ))
                mkdir = stack.enter_context(mock.patch.object(
                    stager.os, "mkdir", wraps=os.mkdir,
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._remote_bootstrap(raw, claimed, "a" * 64)
            mkdir.assert_not_called()
            self.assertEqual(tuple(fixture.remote.iterdir()), ())

    def test_post_mkdir_named_gate_failures_cleanup_held_root_and_child(self) -> None:
        for location in ("root", "child"):
            for number in (errno.ESTALE, errno.EIO):
                with self.subTest(
                    location=location, errno=number,
                ), physical15_fixture() as fixture:
                    _payload, raw, claimed = payload_bytes(fixture)
                    real_gate = stager._assert_created_directory_named
                    injected = False

                    def fail_named_gate(*args, label, **kwargs):
                        nonlocal injected
                        selected = (
                            label == "random shadow root creation"
                            if location == "root"
                            else label == "shadow child directory creation"
                        )
                        if selected and not injected:
                            injected = True
                            raise OSError(
                                number,
                                "injected post-mkdir named identity gate",
                            )
                        return real_gate(*args, label=label, **kwargs)

                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(
                            stager, "_validate_remote_runtime",
                        ))
                        stack.enter_context(mock.patch.object(
                            stager,
                            "_assert_created_directory_named",
                            side_effect=fail_named_gate,
                        ))
                        with self.assertRaises(OSError):
                            stager._remote_bootstrap(raw, claimed, "a" * 64)
                    self.assertTrue(injected)
                    self.assertEqual(tuple(fixture.remote.iterdir()), ())

    def test_post_mkdir_named_gate_replacements_are_preserved(self) -> None:
        for location in ("root", "child"):
            with self.subTest(location=location), physical15_fixture() as fixture:
                _payload, raw, claimed = payload_bytes(fixture)
                real_gate = stager._assert_created_directory_named
                injected = False
                moved = fixture.remote / (".held-created-" + location)

                def replace_before_named_gate(*args, label, **kwargs):
                    nonlocal injected
                    selected = (
                        label == "random shadow root creation"
                        if location == "root"
                        else label == "shadow child directory creation"
                    )
                    if selected and not injected:
                        injected = True
                        shadow = next(
                            path for path in fixture.remote.iterdir()
                            if path.name.startswith(
                                ".physical15-target.shadow-"
                            )
                        )
                        created = shadow if location == "root" else shadow / args[1]
                        created.rename(moved)
                        created.mkdir(mode=0o700)
                        marker = created / "foreign"
                        marker.write_bytes(b"foreign")
                    return real_gate(*args, label=label, **kwargs)

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager, "_validate_remote_runtime",
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_assert_created_directory_named",
                        side_effect=replace_before_named_gate,
                    ))
                    with self.assertRaises(stager.SourceStageError):
                        stager._remote_bootstrap(raw, claimed, "a" * 64)
                self.assertTrue(injected)
                replacement_root = next(
                    path for path in fixture.remote.iterdir()
                    if path.name.startswith(".physical15-target.shadow-")
                )
                replacement = (
                    replacement_root
                    if location == "root"
                    else replacement_root / "artifacts"
                )
                self.assertEqual((replacement / "foreign").read_bytes(), b"foreign")
                self.assertTrue(moved.is_dir())
                self.assertFalse(fixture.target.exists())
                self._assert_nonadmissible_reservation(fixture)

    def test_target_byte_identical_replacement_cannot_receive_false_receipt(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)

            def replace_after_publish(
                parent_fd, shadow_name, target_name, shadow_fd,
                shadow_anchor, reservation,
            ):
                portable_noreplace(fixture)(
                    parent_fd, shadow_name, target_name, shadow_fd,
                    shadow_anchor, reservation,
                )
                original = fixture.remote / ".held-original-target"
                fixture.target.chmod(0o700)
                fixture.target.rename(original)
                original.chmod(0o555)
                shutil.copytree(original, fixture.target)

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager, "_rename_under_receipt_reservation", side_effect=replace_after_publish,
                ))
                terminal = stager._remote_bootstrap(raw, claimed, "a" * 64)
            self.assertEqual(
                terminal["status"],
                "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
            )
            self.assertTrue(terminal["target_rename_commit_point_crossed"])
            self.assertFalse(terminal["zero_publication_claim"])
            self.assertFalse(terminal["recovery_admissible"])
            self.assertFalse(terminal["named_target_same_held_inode"])
            self.assertTrue(fixture.target.is_dir())
            self.assertTrue(fixture.receipt.exists())
            self.assertEqual(
                stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o600,
            )
            self.assertNotEqual(
                fixture.target.lstat().st_ino,
                (fixture.remote / ".held-original-target").lstat().st_ino,
            )

    def test_receipt_byte_identical_replacement_is_preserved_and_refused(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            real_seal = stager._seal_reserved_receipt

            def replace_receipt(
                parent_fd, reservation, value, *, expected_prior_state=None,
            ):
                held = real_seal(
                    parent_fd, reservation, value,
                    expected_prior_state=expected_prior_state,
                )
                original = fixture.remote / ".held-original-receipt"
                fixture.receipt.rename(original)
                fixture.receipt.write_bytes(original.read_bytes())
                fixture.receipt.chmod(0o400)
                return held

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_rename_under_receipt_reservation",
                    side_effect=portable_noreplace(fixture),
                ))
                stack.enter_context(mock.patch.object(
                    stager, "_seal_reserved_receipt", side_effect=replace_receipt,
                ))
                terminal = stager._remote_bootstrap(raw, claimed, "a" * 64)
            self.assertEqual(
                terminal["status"],
                "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
            )
            self.assertFalse(terminal["receipt_authoritative"])
            original = fixture.remote / ".held-original-receipt"
            self.assertEqual(fixture.receipt.read_bytes(), original.read_bytes())
            self.assertNotEqual(
                fixture.receipt.lstat().st_ino, original.lstat().st_ino,
            )
            self.assertTrue(fixture.target.is_dir())

    def test_failed_reservation_write_removes_its_owned_partial_inode(self) -> None:
        with physical15_fixture() as fixture:
            _payload, _raw, claimed = payload_bytes(fixture)
            parent_fd = os.open(
                fixture.remote,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager.os, "pwrite",
                        side_effect=stager.SourceStageError("injected receipt write"),
                    ))
                    with self.assertRaises(stager.SourceStageError):
                        stager._reserve_receipt(
                            parent_fd,
                            fixture.receipt.name,
                            manifest=stager._manifest_value(),
                            request_payload_sha256=claimed,
                            bootstrap_sha256="a" * 64,
                        )
            finally:
                os.close(parent_fd)
            self.assertFalse(fixture.receipt.exists())

    def test_held_ancestor_chain_detects_named_parent_replacement(self) -> None:
        with physical15_fixture() as fixture:
            descriptors, _parent_fd, anchors = stager._hold_directory_chain(
                fixture.remote,
            )
            moved = fixture.base / "remote-held-old"
            try:
                fixture.remote.rename(moved)
                fixture.remote.mkdir()
                with self.assertRaises(stager.SourceStageError):
                    stager._replay_directory_chain(
                        fixture.remote, descriptors, anchors,
                    )
            finally:
                for descriptor in reversed(descriptors):
                    os.close(descriptor)
                fixture.remote.rmdir()
                moved.rename(fixture.remote)


class ControllerIntegrationTests(unittest.TestCase):
    def test_local_commit_terminal_absent_tamper_and_replacement_refused(self) -> None:
        with physical15_fixture() as fixture:
            with ExitStack() as stack:
                execute = stack.enter_context(mock.patch.object(
                    stager, "_execute_remote",
                ))
                with self.assertRaises(FileNotFoundError):
                    stager.controller(
                        stager.RECOVER_RECEIPT_OPERATION,
                        fixture.local_terminal,
                    )
            execute.assert_not_called()

            synthetic_remote = {
                "schema_version": stager.TERMINAL_SCHEMA,
                "terminal_digest": "a" * 64,
            }
            held = stager._write_local_commit_terminal(
                fixture.local_terminal, synthetic_remote,
            )
            try:
                exact = held.replay()
            finally:
                held.close()
            self.assertEqual(
                exact["remote_commit_terminal"], synthetic_remote,
            )
            original = fixture.base / "original-terminal.json"
            fixture.local_terminal.rename(original)
            fixture.local_terminal.write_bytes(original.read_bytes())
            fixture.local_terminal.chmod(0o400)
            with self.assertRaises(stager.SourceStageError):
                stager._open_local_commit_terminal(fixture.local_terminal)
            self.assertEqual(
                fixture.local_terminal.read_bytes(), original.read_bytes(),
            )
            fixture.local_terminal.chmod(0o600)
            fixture.local_terminal.write_bytes(b"{}\n")
            fixture.local_terminal.chmod(0o400)
            with self.assertRaises(stager.SourceStageError):
                stager._open_local_commit_terminal(fixture.local_terminal)

    def test_postcommit_local_terminal_failures_are_permanent_hold(self) -> None:
        fixed_error = (
            "TARGET_RENAMED_LOCAL_TERMINAL_PERSISTENCE_FAILED_"
            "PERMANENT_HOLD"
        )
        for failure in ("competitor", "write", "seal"):
            with self.subTest(failure=failure), physical15_fixture() as fixture:
                _payload, raw, claimed = payload_bytes(
                    fixture,
                    bootstrap_sha256=stager.REMOTE_BOOTSTRAP_SHA256,
                )
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager, "_validate_remote_runtime",
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_rename_under_receipt_reservation",
                        side_effect=portable_noreplace(fixture),
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_seal_reserved_receipt",
                        side_effect=stager.SourceStageError(
                            "injected remote receipt failure"
                        ),
                    ))
                    terminal = stager._remote_bootstrap(
                        raw,
                        claimed,
                        stager.REMOTE_BOOTSTRAP_SHA256,
                    )
                self.assertEqual(
                    terminal["status"],
                    "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
                )
                self.assertTrue(terminal["recovery_admissible"])
                self.assertTrue(fixture.receipt.exists())
                self.assertEqual(
                    stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o600,
                )
                before_target = target_snapshot(fixture.target)
                competitor = None
                if failure == "competitor":
                    fixture.local_terminal.write_bytes(b"competitor\n")
                    fixture.local_terminal.chmod(0o400)
                    competitor = (
                        stager._identity(fixture.local_terminal.lstat()),
                        fixture.local_terminal.read_bytes(),
                    )

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager, "CONTROLLER_STATE", stager.READY_STATE,
                    ))
                    execute = stack.enter_context(mock.patch.object(
                        stager,
                        "_execute_remote",
                        return_value=stager.canonical(terminal) + b"\n",
                    ))
                    if failure == "write":
                        stack.enter_context(mock.patch.object(
                            stager.os,
                            "write",
                            side_effect=OSError(
                                errno.EIO, "injected local terminal write"
                            ),
                        ))
                    elif failure == "seal":
                        stack.enter_context(mock.patch.object(
                            stager.os,
                            "fchmod",
                            side_effect=OSError(
                                errno.EIO, "injected local terminal seal"
                            ),
                        ))
                    with self.assertRaises(stager.SourceStageError) as caught:
                        stager.controller()
                self.assertEqual(str(caught.exception), fixed_error)
                execute.assert_called_once()
                self.assertEqual(target_snapshot(fixture.target), before_target)
                self.assertTrue(fixture.receipt.exists())
                if competitor is None:
                    self.assertFalse(fixture.local_terminal.exists())
                else:
                    self.assertEqual(
                        (
                            stager._identity(fixture.local_terminal.lstat()),
                            fixture.local_terminal.read_bytes(),
                        ),
                        competitor,
                    )

    def test_ready_controller_uses_frozen_bootstrap_for_stage_and_recovery(self) -> None:
        with physical15_fixture() as fixture:
            calls = []

            def execute(payload_file, *, bootstrap_sha256, payload_sha256):
                payload_file.seek(0)
                bootstrap_raw, payload_raw = stager._unframe_payload(
                    payload_file.read(),
                )
                self.assertEqual(sha(bootstrap_raw), bootstrap_sha256)
                self.assertEqual(sha(payload_raw), payload_sha256)
                self.assertEqual(bootstrap_raw, BOOTSTRAP.read_bytes())
                payload_value = stager._strict_json(
                    payload_raw, label="test held payload",
                )
                calls.append((
                    bootstrap_sha256,
                    payload_sha256,
                    payload_value["operation"],
                ))
                held_scope = {
                    "__name__": "_held_source_stager_bootstrap",
                    "__file__": (
                        "/held/repo/methods/bernini_action_editing/"
                        "case01_object_trajectory_exact5_source_stager_"
                        "remote_bootstrap_v1.py"
                    ),
                }
                exec(
                    compile(
                        bootstrap_raw,
                        "<held-source-stager-bootstrap>",
                        "exec",
                        dont_inherit=True,
                    ),
                    held_scope,
                )
                held_scope.update({
                    "SOURCE_AUTHORITIES": fixture.specs,
                    "REMOTE_PARENT": fixture.remote,
                    "REMOTE_TARGET_ROOT": fixture.target,
                    "REMOTE_RECEIPT_PATH": fixture.receipt,
                    "REMOTE_UID": os.geteuid(),
                    "REMOTE_GID": os.getegid(),
                    "_validate_remote_runtime": lambda _manifest: None,
                    "_rename_under_receipt_reservation": portable_noreplace(fixture),
                })
                if payload_value["operation"] == stager.STAGE_OPERATION:
                    def fail_receipt(*_args, **_kwargs):
                        raise held_scope["SourceStageError"](
                            "injected controller commit receipt failure"
                        )
                    held_scope["_seal_reserved_receipt"] = fail_receipt
                self.assertIn("_remote_dispatch", held_scope)
                receipt = held_scope["_remote_dispatch"](
                    payload_raw, payload_sha256, bootstrap_sha256,
                )
                return stager.canonical(receipt) + b"\n"

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "CONTROLLER_STATE", stager.READY_STATE,
                ))
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_rename_under_receipt_reservation",
                    side_effect=portable_noreplace(fixture),
                ))
                stack.enter_context(mock.patch.object(
                    stager, "_execute_remote", side_effect=execute,
                ))
                with self.assertRaises(stager.CommitRecoveryRequired) as caught:
                    stager.controller()
                terminal_authority = caught.exception.terminal
                self.assertEqual(
                    fixture.local_terminal.read_bytes(),
                    stager.canonical(terminal_authority) + b"\n",
                )
                before_target = target_snapshot(fixture.target)
                recovered = stager.controller(
                    stager.RECOVER_RECEIPT_OPERATION,
                    fixture.local_terminal,
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(
                [call[2] for call in calls],
                [stager.STAGE_OPERATION, stager.RECOVER_RECEIPT_OPERATION],
            )
            self.assertEqual(
                terminal_authority["status"],
                "TARGET_RENAMED_LOCAL_TERMINAL_PERSISTED",
            )
            self.assertTrue(
                terminal_authority["remote_commit_terminal"]
                ["recovery_admissible"]
            )
            self.assertEqual(recovered["status"], "RECOVERED_RECEIPT_ONLY")
            self.assertEqual(target_snapshot(fixture.target), before_target)

    def test_refreshed_recovery_terminal_uses_explicit_create_only_path(
        self,
    ) -> None:
        with physical15_fixture() as fixture:
            bootstrap_sha256 = stager.REMOTE_BOOTSTRAP_SHA256
            _payload, raw, claimed = payload_bytes(
                fixture, bootstrap_sha256=bootstrap_sha256,
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_rename_under_receipt_reservation",
                    side_effect=portable_noreplace(fixture),
                ))
                stack.enter_context(mock.patch.object(
                    stager,
                    "_seal_reserved_receipt",
                    side_effect=stager.SourceStageError(
                        "injected initial terminal",
                    ),
                ))
                initial = stager._remote_bootstrap(
                    raw, claimed, bootstrap_sha256,
                )
            original_terminal = stager._write_local_commit_terminal(
                fixture.local_terminal, initial,
            )
            try:
                original_value = original_terminal.replay()
            finally:
                original_terminal.close()
            original_bytes = fixture.local_terminal.read_bytes()

            real_seal = stager._seal_reserved_receipt
            real_pwrite = stager.os.pwrite

            def partial_recovery_seal(
                parent_fd, reservation, value, *, expected_prior_state=None,
            ):
                calls = 0

                def partial_then_error(descriptor, content, offset):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        prefix = max(1, len(content) // 2)
                        return real_pwrite(
                            descriptor, content[:prefix], offset,
                        )
                    raise OSError(errno.EIO, "injected chained pwrite")

                with mock.patch.object(
                    stager.os, "pwrite", side_effect=partial_then_error,
                ):
                    return real_seal(
                        parent_fd,
                        reservation,
                        value,
                        expected_prior_state=expected_prior_state,
                    )

            with mock.patch.object(
                stager,
                "_seal_reserved_receipt",
                side_effect=partial_recovery_seal,
            ):
                refreshed = RemoteBootstrapTests()._recover(
                    fixture,
                    initial,
                    bootstrap_sha256=bootstrap_sha256,
                )
            self.assertEqual(
                refreshed["status"],
                "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
            )

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "CONTROLLER_STATE", stager.READY_STATE,
                ))
                execute = stack.enter_context(mock.patch.object(
                    stager,
                    "_execute_remote",
                    return_value=stager.canonical(refreshed) + b"\n",
                ))
                with self.assertRaises(stager.CommitRecoveryRequired) as caught:
                    stager.controller(
                        stager.RECOVER_RECEIPT_OPERATION,
                        fixture.local_terminal,
                    )
            execute.assert_called_once()
            derived_value = caught.exception.terminal
            derived_path = Path(derived_value["terminal_path"])
            self.assertEqual(
                derived_path,
                fixture.local_terminal.parent
                / (
                    fixture.local_terminal.name + "."
                    + refreshed["terminal_digest"]
                ),
            )
            self.assertEqual(fixture.local_terminal.read_bytes(), original_bytes)
            self.assertEqual(
                original_value["remote_commit_terminal"], initial,
            )
            self.assertEqual(stat.S_IMODE(derived_path.lstat().st_mode), 0o400)
            self.assertEqual(
                stager._strict_json(
                    derived_path.read_bytes(), label="derived local terminal",
                ),
                derived_value,
            )

            fixed_error = (
                "TARGET_RENAMED_LOCAL_TERMINAL_PERSISTENCE_FAILED_"
                "PERMANENT_HOLD"
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "CONTROLLER_STATE", stager.READY_STATE,
                ))
                execute = stack.enter_context(mock.patch.object(
                    stager,
                    "_execute_remote",
                    return_value=stager.canonical(refreshed) + b"\n",
                ))
                with self.assertRaises(stager.SourceStageError) as collision:
                    stager.controller(
                        stager.RECOVER_RECEIPT_OPERATION,
                        fixture.local_terminal,
                    )
            execute.assert_called_once()
            self.assertEqual(str(collision.exception), fixed_error)
            self.assertEqual(fixture.local_terminal.read_bytes(), original_bytes)
            self.assertEqual(
                stager._strict_json(
                    derived_path.read_bytes(), label="retained derived terminal",
                ),
                derived_value,
            )

            recovered_remote = RemoteBootstrapTests()._recover(
                fixture,
                refreshed,
                bootstrap_sha256=bootstrap_sha256,
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "CONTROLLER_STATE", stager.READY_STATE,
                ))
                execute = stack.enter_context(mock.patch.object(
                    stager,
                    "_execute_remote",
                    return_value=stager.canonical(recovered_remote) + b"\n",
                ))
                recovered = stager.controller(
                    stager.RECOVER_RECEIPT_OPERATION,
                    derived_path,
                )
            execute.assert_called_once()
            self.assertEqual(recovered, recovered_remote)
            self.assertEqual(recovered["status"], "RECOVERED_RECEIPT_ONLY")

    def test_cli_routes_recovery_and_commit_terminal_to_distinct_rc(self) -> None:
        recovered = {
            "schema_version": "synthetic-recovered",
            "status": "RECOVERED_RECEIPT_ONLY",
        }
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                stager, "CONTROLLER_STATE", stager.READY_STATE,
            ))
            controller = stack.enter_context(mock.patch.object(
                stager, "controller", return_value=recovered,
            ))
            printed = stack.enter_context(mock.patch("builtins.print"))
            self.assertEqual(stager.main([
                "recover-receipt", str(stager.LOCAL_COMMIT_TERMINAL_PATH),
            ]), 0)
        controller.assert_called_once_with(
            stager.RECOVER_RECEIPT_OPERATION,
            stager.LOCAL_COMMIT_TERMINAL_PATH,
        )
        printed.assert_called_once_with(
            stager.canonical(recovered).decode("utf-8")
        )

        terminal = {
            "schema_version": stager.TERMINAL_SCHEMA,
            "status": "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
        }
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                stager, "CONTROLLER_STATE", stager.READY_STATE,
            ))
            stack.enter_context(mock.patch.object(
                stager,
                "controller",
                side_effect=stager.CommitRecoveryRequired(terminal),
            ))
            printed = stack.enter_context(mock.patch("builtins.print"))
            self.assertEqual(stager.main([]), stager.COMMIT_RECOVERY_REQUIRED_RC)
        printed.assert_called_once_with(
            stager.canonical(terminal).decode("utf-8")
        )

    def test_single_posix_rename_applied_then_error_is_recoverable(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            real_rename = os.rename

            def applied_then_error(
                source, destination, *, src_dir_fd=None, dst_dir_fd=None,
            ):
                shadow = fixture.remote / source
                shadow.chmod(0o700)
                real_rename(
                    source,
                    destination,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                )
                fixture.target.chmod(0o555)
                raise OSError(errno.EIO, "injected applied-then-error")

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                renamed = stack.enter_context(mock.patch.object(
                    stager.os, "rename", side_effect=applied_then_error,
                ))
                terminal = stager._remote_bootstrap(raw, claimed, "a" * 64)
            renamed.assert_called_once()
            self.assertEqual(
                terminal["status"],
                "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
            )
            self.assertEqual(terminal["rename_attempt_count"], 1)
            self.assertEqual(terminal["rename_result"], "raised")
            self.assertEqual(
                terminal["rename_classification"],
                "applied_then_error_target_is_held_shadow",
            )
            self.assertTrue(terminal["recovery_admissible"])
            self.assertEqual(terminal["receipt_phase"], "reserved_0600")
            self.assertTrue(fixture.target.is_dir())
            self.assertEqual(
                stat.S_IMODE(fixture.receipt.lstat().st_mode), 0o600,
            )
            self.assertEqual(
                len([
                    path for path in fixture.remote.iterdir()
                    if path.name.startswith(".physical15-target.shadow-")
                ]),
                0,
            )
            target_before = target_snapshot(fixture.target)
            recovered = RemoteBootstrapTests()._recover(fixture, terminal)
            self.assertEqual(recovered["status"], "RECOVERED_RECEIPT_ONLY")
            self.assertEqual(target_snapshot(fixture.target), target_before)

    def test_single_posix_rename_not_applied_cleans_owned_names(self) -> None:
        with physical15_fixture() as fixture:
            _payload, raw, claimed = payload_bytes(fixture)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager, "_validate_remote_runtime",
                ))
                renamed = stack.enter_context(mock.patch.object(
                    stager.os,
                    "rename",
                    side_effect=OSError(errno.EIO, "injected not-applied"),
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._remote_bootstrap(raw, claimed, "a" * 64)
            renamed.assert_called_once()
            self.assertEqual(tuple(fixture.remote.iterdir()), ())


class TransportLifecycleTests(unittest.TestCase):
    def test_all_three_named_transport_authorities_are_in_argv(self) -> None:
        authorities = []
        for descriptor, path in zip(
            (11, 12, 13),
            (stager.SSH_PATH, stager.SSH_IDENTITY, stager.SSH_KNOWN_HOSTS),
        ):
            authority = mock.Mock()
            authority.descriptor = descriptor
            authority.path = path
            authorities.append(authority)
        command = stager._ssh_arguments("remote command", authorities)
        joined = "\0".join(command)
        self.assertEqual(command[0], str(stager.SSH_PATH))
        self.assertNotIn("/dev/fd/", joined)
        self.assertIn(f"IdentityFile={stager.SSH_IDENTITY}", command)
        self.assertIn(
            f"UserKnownHostsFile={stager.SSH_KNOWN_HOSTS}", command,
        )

    def test_bounded_stream_diagnostic_is_exact_and_truthful(self) -> None:
        raw = b"x" * (stager.TRANSPORT_DIAGNOSTIC_PREFIX_LIMIT + 17)
        value = stager._bounded_stream_diagnostic(raw)
        self.assertEqual(value["size"], len(raw))
        self.assertEqual(value["sha256"], sha(raw))
        self.assertEqual(value["prefix_size"], stager.TRANSPORT_DIAGNOSTIC_PREFIX_LIMIT)
        self.assertEqual(
            base64.b64decode(value["prefix_b64"], validate=True),
            raw[:stager.TRANSPORT_DIAGNOSTIC_PREFIX_LIMIT],
        )
        self.assertIs(value["truncated"], True)
        diagnostic = stager._transport_terminal_diagnostic(
            reason="terminal_contract", returncode=255,
            stdout=b"partial stdout\n", stderr=b"binary\x00stderr\n",
            streams_complete=True,
        )
        self.assertIs(diagnostic["remote_output_is_untrusted"], True)
        self.assertIs(diagnostic["prefix_may_contain_remote_echo_of_input"], True)
        unsigned = dict(diagnostic)
        claimed = unsigned.pop("diagnostic_digest")
        self.assertEqual(claimed, stager.object_digest(unsigned))

    @unittest.skipUnless(sys.platform == "darwin", "Darwin named transport gate")
    def test_terminal_failure_reports_bounded_exact_streams_without_retry(self) -> None:
        class FakeProcess:
            pid = 424_242
            returncode = 255
            stdout = None
            stderr = None

            @staticmethod
            def communicate(*, timeout):
                self.assertEqual(timeout, stager.TRANSPORT_TIMEOUT_SECONDS)
                return b"remote partial\n", b"ssh exact failure\x00\n"

            @staticmethod
            def poll():
                return 255

        with physical15_fixture() as fixture:
            payload = tempfile.TemporaryFile()
            try:
                with ExitStack() as stack:
                    popen = stack.enter_context(mock.patch.object(
                        stager.subprocess, "Popen", return_value=FakeProcess(),
                    ))
                    stack.enter_context(mock.patch.object(
                        stager.os, "getpgid", return_value=FakeProcess.pid,
                    ))
                    seal = stack.enter_context(mock.patch.object(
                        stager, "_seal_process_group",
                    ))
                    with self.assertRaises(stager.SourceStageError) as caught:
                        stager._execute_remote(
                            payload,
                            bootstrap_sha256="a" * 64,
                            payload_sha256="b" * 64,
                        )
                popen.assert_called_once()
                seal.assert_called_once()
                kwargs = popen.call_args.kwargs
                self.assertIs(kwargs["close_fds"], True)
                self.assertEqual(kwargs["pass_fds"], ())
                prefix = "single SSH staging terminal streams differ: "
                self.assertTrue(str(caught.exception).startswith(prefix))
                diagnostic = json.loads(str(caught.exception)[len(prefix):])
                self.assertEqual(diagnostic["returncode"], 255)
                self.assertIs(diagnostic["streams_complete"], True)
                self.assertEqual(
                    base64.b64decode(
                        diagnostic["stdout"]["prefix_b64"], validate=True,
                    ),
                    b"remote partial\n",
                )
                self.assertEqual(
                    base64.b64decode(
                        diagnostic["stderr"]["prefix_b64"], validate=True,
                    ),
                    b"ssh exact failure\x00\n",
                )
                self.assertFalse(fixture.target.exists())
                self.assertFalse(fixture.receipt.exists())
            finally:
                payload.close()

    def test_timeout_diagnostic_labels_partial_streams(self) -> None:
        diagnostic = stager._transport_terminal_diagnostic(
            reason="timeout_partial", returncode=None,
            stdout=b"partial", stderr=b"", streams_complete=False,
        )
        self.assertEqual(diagnostic["reason"], "timeout_partial")
        self.assertIsNone(diagnostic["returncode"])
        self.assertIs(diagnostic["streams_complete"], False)

    def test_rc0_stderr_and_wrong_stdout_newline_counts_all_refuse(self) -> None:
        cases = (
            (b"valid-looking\n", b"unexpected stderr\n", 1),
            (b"no newline", b"", 0),
            (b"two\nnewlines\n", b"", 2),
        )
        for stdout, stderr, newline_count in cases:
            with self.subTest(newline_count=newline_count, stderr=bool(stderr)):
                class FakeProcess:
                    pid = 454_545
                    returncode = 0
                    stdout = None
                    stderr = None

                    @staticmethod
                    def communicate(*, timeout):
                        return stdout, stderr

                    @staticmethod
                    def poll():
                        return 0

                payload = tempfile.TemporaryFile()
                transport = [
                    DummyTransportAuthority(payload.fileno()) for _ in range(3)
                ]
                try:
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(
                            stager, "_open_transport_authorities",
                            return_value=transport,
                        ))
                        stack.enter_context(mock.patch.object(
                            stager, "_validate_named_transport_authorities",
                        ))
                        stack.enter_context(mock.patch.object(
                            stager, "_ssh_arguments", return_value=["fake-ssh"],
                        ))
                        popen = stack.enter_context(mock.patch.object(
                            stager.subprocess, "Popen", return_value=FakeProcess(),
                        ))
                        stack.enter_context(mock.patch.object(
                            stager.os, "getpgid", return_value=FakeProcess.pid,
                        ))
                        stack.enter_context(mock.patch.object(
                            stager, "_seal_process_group",
                        ))
                        with self.assertRaises(stager.SourceStageError) as caught:
                            stager._execute_remote(
                                payload,
                                bootstrap_sha256="a" * 64,
                                payload_sha256="b" * 64,
                            )
                    popen.assert_called_once()
                    self.assertIn("terminal streams differ", str(caught.exception))
                    self.assertEqual(
                        [authority.descriptor for authority in transport],
                        [-1, -1, -1],
                    )
                finally:
                    for authority in transport:
                        authority.close()
                    payload.close()

    @unittest.skipUnless(sys.platform == "darwin", "Darwin named transport gate")
    def test_timeout_failure_reports_exact_partial_streams(self) -> None:
        class FakeProcess:
            pid = 434_343
            returncode = None
            stdout = None
            stderr = None

            @staticmethod
            def communicate(*, timeout):
                raise subprocess.TimeoutExpired(
                    cmd="held named ssh", timeout=timeout,
                    output=b"timeout stdout", stderr=b"timeout stderr\n",
                )

            @staticmethod
            def poll():
                return None

        with physical15_fixture() as fixture:
            payload = tempfile.TemporaryFile()
            try:
                with ExitStack() as stack:
                    popen = stack.enter_context(mock.patch.object(
                        stager.subprocess, "Popen", return_value=FakeProcess(),
                    ))
                    stack.enter_context(mock.patch.object(
                        stager.os, "getpgid", return_value=FakeProcess.pid,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "_seal_process_group",
                    ))
                    with self.assertRaises(stager.SourceStageError) as caught:
                        stager._execute_remote(
                            payload,
                            bootstrap_sha256="a" * 64,
                            payload_sha256="b" * 64,
                        )
                popen.assert_called_once()
                prefix = "single SSH staging attempt timed out: "
                self.assertTrue(str(caught.exception).startswith(prefix))
                diagnostic = json.loads(str(caught.exception)[len(prefix):])
                self.assertEqual(diagnostic["reason"], "timeout_partial")
                self.assertIsNone(diagnostic["returncode"])
                self.assertIs(diagnostic["streams_complete"], False)
                self.assertEqual(
                    base64.b64decode(
                        diagnostic["stdout"]["prefix_b64"], validate=True,
                    ),
                    b"timeout stdout",
                )
                self.assertEqual(
                    base64.b64decode(
                        diagnostic["stderr"]["prefix_b64"], validate=True,
                    ),
                    b"timeout stderr\n",
                )
                self.assertFalse(fixture.target.exists())
                self.assertFalse(fixture.receipt.exists())
            finally:
                payload.close()

    @unittest.skipUnless(sys.platform == "darwin", "Darwin sealed-system gate")
    def test_pinned_system_ssh_macho_signature_structure_and_hostiles(self) -> None:
        raw = stager.SSH_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), stager.SSH_SHA256)
        self.assertEqual(
            stager._macho_code_signature_ranges(raw),
            ((712_672, 24_256), (1_449_744, 24_384)),
        )

        def changed(offset, packed):
            mutated = bytearray(raw)
            mutated[offset:offset + len(packed)] = packed
            return bytes(mutated)

        x86_slice = stager.SSH_FAT_ARCHITECTURES[0][2]
        x86_signature = 712_672
        super_length = struct.unpack_from(">I", raw, x86_signature + 4)[0]
        code_directory = x86_signature + 52
        hostiles = {
            "fat_count": changed(4, struct.pack(">I", 1)),
            "fat_offset": changed(16, struct.pack(">I", x86_slice + 1)),
            "mach_cpu": changed(
                x86_slice + 4, struct.pack("<I", 0x0100_000C),
            ),
            "command_size": changed(
                x86_slice + 32 + 4, struct.pack("<I", 7),
            ),
            "missing_signature": changed(
                x86_slice + 3_200, struct.pack("<I", 0),
            ),
            "superblob_count": changed(
                x86_signature + 8, struct.pack(">I", 6),
            ),
            "superblob_child_offset": changed(
                x86_signature + 24, struct.pack(">I", 53),
            ),
            "code_directory_magic": changed(
                code_directory, struct.pack(">I", 0xFADE0B01),
            ),
            "identifier": changed(
                code_directory + 88, b"x",
            ),
            "hash_size": changed(code_directory + 36, b"\x14"),
            "platform": changed(code_directory + 38, b"\x00"),
            "page_size": changed(code_directory + 39, b"\x0d"),
            "nonzero_padding": changed(
                x86_signature + super_length, b"\x01",
            ),
        }
        for label, hostile in hostiles.items():
            with self.subTest(label=label):
                with self.assertRaises(stager.SourceStageError):
                    stager._macho_code_signature_ranges(hostile)

    @unittest.skipUnless(sys.platform == "darwin", "Darwin sealed-system gate")
    def test_system_ssh_metadata_and_read_only_authority_hostiles(self) -> None:
        authority = stager._open_local_authority(
            stager.SSH_PATH,
            sha256=stager.SSH_SHA256,
            size=stager.SSH_SIZE,
        )
        try:
            stager._validate_system_ssh_authority(authority)
            actual = os.fstat(authority.descriptor)
            unrestricted = stat_namespace(actual, st_flags=0)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(authority, "replay"))
                stack.enter_context(mock.patch.object(
                    stager.os, "fstat", return_value=unrestricted,
                ))
                stack.enter_context(mock.patch.object(
                    stager.os, "lstat", return_value=unrestricted,
                ))
                stack.enter_context(mock.patch.object(
                    stager, "_validate_read_only_filesystem",
                ))
                stack.enter_context(mock.patch.object(
                    stager, "_validate_system_parent_chain",
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._validate_system_ssh_authority(authority)

            for label, changes in (
                ("nlink", {"st_nlink": 2}),
                ("mode", {"st_mode": stat.S_IFREG | 0o775}),
                ("owner", {"st_uid": 501}),
            ):
                hostile_info = stat_namespace(actual, **changes)
                hostile_authority = SimpleNamespace(
                    path=stager.SSH_PATH,
                    descriptor=authority.descriptor,
                    identity=stager._identity(hostile_info),
                    raw=authority.raw,
                    sha256=authority.sha256,
                    replay=mock.Mock(),
                )
                with self.subTest(label=label), ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager.os, "fstat", return_value=hostile_info,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager.os, "lstat", return_value=hostile_info,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "_validate_read_only_filesystem",
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "_validate_system_parent_chain",
                    ))
                    with self.assertRaises(stager.SourceStageError):
                        stager._validate_system_ssh_authority(hostile_authority)

            with mock.patch.object(stager.sys, "platform", "linux"):
                with self.assertRaises(stager.SourceStageError):
                    stager._validate_system_ssh_authority(authority)

            read_write = SimpleNamespace(f_fsid=7, f_flag=0)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    stager.os, "fstatvfs", return_value=read_write,
                ))
                stack.enter_context(mock.patch.object(
                    stager.os, "statvfs", return_value=read_write,
                ))
                with self.assertRaises(stager.SourceStageError):
                    stager._validate_read_only_filesystem(
                        authority.descriptor, stager.SSH_PATH, actual.st_dev,
                    )

            real_lstat = stager.os.lstat

            def writable_usr(path):
                observed = real_lstat(path)
                if Path(path) != Path("/usr"):
                    return observed
                return stat_namespace(observed, st_mode=stat.S_IFDIR | 0o777)

            with mock.patch.object(stager.os, "lstat", side_effect=writable_usr):
                with self.assertRaises(stager.SourceStageError):
                    stager._validate_system_parent_chain(actual.st_dev)
        finally:
            authority.close()

    @unittest.skipUnless(sys.platform == "darwin", "Darwin named transport gate")
    def test_named_credential_authorities_and_parent_hostiles(self) -> None:
        transport = stager._open_transport_authorities()
        try:
            stager._validate_named_transport_authorities(transport)
            for authority, path, digest, size in (
                (
                    transport[1], stager.SSH_IDENTITY,
                    stager.SSH_IDENTITY_SHA256, stager.SSH_IDENTITY_SIZE,
                ),
                (
                    transport[2], stager.SSH_KNOWN_HOSTS,
                    stager.SSH_KNOWN_HOSTS_SHA256,
                    stager.SSH_KNOWN_HOSTS_SIZE,
                ),
            ):
                self.assertEqual(
                    stager._validate_named_credential_authority(
                        authority, path, digest, size,
                    ),
                    os.fstat(authority.descriptor).st_dev,
                )

            actual = os.fstat(transport[1].descriptor)
            for label, changes in (
                ("nlink", {"st_nlink": 2}),
                ("mode", {"st_mode": stat.S_IFREG | 0o640}),
                ("owner", {"st_uid": 0}),
            ):
                hostile_info = stat_namespace(actual, **changes)
                hostile = SimpleNamespace(
                    path=stager.SSH_IDENTITY,
                    descriptor=transport[1].descriptor,
                    identity=stager._identity(hostile_info),
                    raw=transport[1].raw,
                    sha256=transport[1].sha256,
                    replay=mock.Mock(),
                )
                with self.subTest(label=label), ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager.os, "fstat", return_value=hostile_info,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager.os, "lstat", return_value=hostile_info,
                    ))
                    with self.assertRaises(stager.SourceStageError):
                        stager._validate_named_credential_authority(
                            hostile, stager.SSH_IDENTITY,
                            stager.SSH_IDENTITY_SHA256,
                            stager.SSH_IDENTITY_SIZE,
                        )

            real_lstat = stager.os.lstat

            def writable_parent(path):
                observed = real_lstat(path)
                if Path(path) != stager.SSH_KNOWN_HOSTS.parent:
                    return observed
                return stat_namespace(observed, st_mode=stat.S_IFDIR | 0o775)

            with mock.patch.object(
                stager.os, "lstat", side_effect=writable_parent,
            ):
                with self.assertRaises(stager.SourceStageError):
                    stager._validate_credential_parent(
                        stager.SSH_KNOWN_HOSTS.parent, actual.st_dev,
                    )

            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                real_parent = root / "real"
                real_parent.mkdir(mode=0o755)
                alias = root / "alias"
                alias.symlink_to(real_parent, target_is_directory=True)
                with self.assertRaises(stager.SourceStageError):
                    stager._validate_credential_parent(alias, root.stat().st_dev)
        finally:
            stager._close_transport_authorities(transport)

    def test_named_credential_byte_identical_replacement_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "identity"
            raw = b"synthetic private credential\n"
            path.write_bytes(raw)
            path.chmod(0o600)
            authority = stager._open_local_authority(
                path, sha256=sha(raw), size=len(raw),
            )
            try:
                held_name = root / "held-original"
                path.rename(held_name)
                path.write_bytes(raw)
                path.chmod(0o600)
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager, "LOCAL_UID", os.geteuid(),
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "LOCAL_GID", os.getegid(),
                    ))
                    with self.assertRaises(stager.SourceStageError):
                        stager._validate_named_credential_authority(
                            authority, path, sha(raw), len(raw),
                        )
                self.assertEqual(path.read_bytes(), raw)
                self.assertEqual(held_name.read_bytes(), raw)
                self.assertNotEqual(path.stat().st_ino, held_name.stat().st_ino)
            finally:
                authority.close()

    @unittest.skipUnless(sys.platform == "darwin", "Darwin sealed-system gate")
    def test_named_system_ssh_executes_version_without_network(self) -> None:
        authority = stager._open_local_authority(
            stager.SSH_PATH,
            sha256=stager.SSH_SHA256,
            size=stager.SSH_SIZE,
        )
        process = None
        try:
            stager._validate_system_ssh_authority(authority)
            process = subprocess.Popen(
                [str(stager.SSH_PATH), "-V"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    "PATH": "/usr/bin:/bin", "HOME": "/var/empty",
                    "LANG": "C", "LC_ALL": "C",
                },
                start_new_session=True,
                close_fds=True,
            )
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0)
            self.assertEqual(stdout, b"")
            self.assertEqual(stderr, b"OpenSSH_9.7p1, LibreSSL 3.3.6\n")
            stager._validate_system_ssh_authority(authority)
        finally:
            if process is not None:
                stager._seal_process_group(process, process.pid)
            authority.close()

    @unittest.skipUnless(sys.platform == "darwin", "Darwin sealed-system gate")
    def test_named_exec_oserrors_close_authorities_and_never_retry(self) -> None:
        for injected_errno in (errno.ENOENT, errno.EACCES, errno.ENOEXEC):
            with self.subTest(injected_errno=injected_errno):
                with physical15_fixture() as fixture:
                    opened = []
                    real_open = stager._open_transport_authorities

                    def capture_open():
                        authorities = real_open()
                        opened.extend(authorities)
                        return authorities

                    def fail_exec(command, **kwargs):
                        self.assertEqual(command[0], "/usr/bin/ssh")
                        self.assertFalse(kwargs.get("shell", False))
                        self.assertIs(kwargs.get("close_fds"), True)
                        self.assertIn(
                            f"IdentityFile={stager.SSH_IDENTITY}", command,
                        )
                        self.assertIn(
                            f"UserKnownHostsFile={stager.SSH_KNOWN_HOSTS}",
                            command,
                        )
                        passed = kwargs["pass_fds"]
                        self.assertEqual(passed, ())
                        raise OSError(injected_errno, "injected exec failure")

                    payload = tempfile.TemporaryFile()
                    try:
                        with ExitStack() as stack:
                            stack.enter_context(mock.patch.object(
                                stager, "_open_transport_authorities",
                                side_effect=capture_open,
                            ))
                            popen = stack.enter_context(mock.patch.object(
                                stager.subprocess, "Popen", side_effect=fail_exec,
                            ))
                            with self.assertRaises(OSError) as caught:
                                stager._execute_remote(
                                    payload,
                                    bootstrap_sha256="a" * 64,
                                    payload_sha256="b" * 64,
                                )
                        self.assertEqual(caught.exception.errno, injected_errno)
                        popen.assert_called_once()
                        self.assertEqual(
                            [authority.descriptor for authority in opened],
                            [-1, -1, -1],
                        )
                        self.assertFalse(fixture.target.exists())
                        self.assertFalse(fixture.receipt.exists())
                    finally:
                        payload.close()

    def test_pre_spawn_failures_close_every_open_transport_authority(self) -> None:
        payload = tempfile.TemporaryFile()
        try:
            for seam in ("replay", "arguments"):
                with self.subTest(seam=seam):
                    authorities = [
                        DummyTransportAuthority(payload.fileno())
                        for _index in range(3)
                    ]
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(
                            stager, "_open_transport_authorities",
                            return_value=authorities,
                        ))
                        stack.enter_context(mock.patch.object(
                            stager, "_validate_named_transport_authorities",
                            side_effect=(
                                RuntimeError("injected transport replay failure")
                                if seam == "replay" else None
                            ),
                        ))
                        if seam == "arguments":
                            stack.enter_context(mock.patch.object(
                                stager, "_ssh_arguments",
                                side_effect=RuntimeError("injected argv failure"),
                            ))
                        popen = stack.enter_context(mock.patch.object(
                            stager.subprocess, "Popen",
                        ))
                        with self.assertRaises((RuntimeError, stager.SourceStageError)):
                            stager._execute_remote(
                                payload,
                                bootstrap_sha256="a" * 64,
                                payload_sha256="b" * 64,
                            )
                    popen.assert_not_called()
                    self.assertEqual(
                        [authority.descriptor for authority in authorities],
                        [-1, -1, -1],
                    )
        finally:
            payload.close()

    def test_partial_transport_open_failure_closes_prior_authorities(self) -> None:
        payload = tempfile.TemporaryFile()
        first = DummyTransportAuthority(payload.fileno())
        second = DummyTransportAuthority(payload.fileno())
        try:
            with mock.patch.object(
                stager, "_open_local_authority",
                side_effect=(first, second, RuntimeError("injected open failure")),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected open failure"):
                    stager._open_transport_authorities()
            self.assertEqual((first.descriptor, second.descriptor), (-1, -1))
        finally:
            first.close()
            second.close()
            payload.close()

    def test_transport_close_aggregates_without_skipping_later_fds(self) -> None:
        payload = tempfile.TemporaryFile()
        authorities = [
            CloseFailureTransportAuthority(payload.fileno()),
            DummyTransportAuthority(payload.fileno()),
            DummyTransportAuthority(payload.fileno()),
        ]
        try:
            with self.assertRaisesRegex(
                stager.SourceStageError, "authority close differs",
            ):
                stager._close_transport_authorities(authorities)
            self.assertEqual(
                [authority.descriptor for authority in authorities],
                [-1, -1, -1],
            )
        finally:
            for authority in authorities:
                try:
                    authority.close()
                except RuntimeError:
                    pass
            payload.close()

    def test_immediate_post_spawn_replay_failure_zeroes_process_group(self) -> None:
        with physical15_fixture() as fixture:
            ready = fixture.base / "post-spawn-child.pid"
            child_source = (
                "import os,signal,sys,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "open(sys.argv[1],'w').write(str(os.getpid()));"
                "time.sleep(60)"
            )
            leader_source = (
                "import os,subprocess,sys,time;"
                "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);"
                "deadline=time.monotonic()+5;"
                "\nwhile not os.path.exists(sys.argv[2]):\n"
                "  if time.monotonic()>deadline: raise RuntimeError('child not ready')\n"
                "  time.sleep(.01)\n"
            )
            command = [
                sys.executable, "-c", leader_source,
                child_source, str(ready),
            ]
            captured = []
            real_popen = subprocess.Popen

            def wait_for_descendant(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                captured.append(process)
                deadline = time.monotonic() + 5
                while not ready.exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("descendant did not become ready")
                    time.sleep(0.01)
                process.wait(timeout=5)
                return process

            payload = tempfile.TemporaryFile()
            authorities = [
                DummyTransportAuthority(payload.fileno()),
                DummyTransportAuthority(payload.fileno()),
                DummyTransportAuthority(payload.fileno()),
            ]
            try:
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager, "_open_transport_authorities",
                        return_value=authorities,
                    ))
                    validation = stack.enter_context(mock.patch.object(
                        stager, "_validate_named_transport_authorities",
                        side_effect=(
                            None,
                            None,
                            RuntimeError("injected post-spawn replay failure"),
                            RuntimeError("injected final replay failure"),
                        ),
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "_ssh_arguments", return_value=command,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "PROCESS_TERM_GRACE_SECONDS", 0.25,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "PROCESS_KILL_GRACE_SECONDS", 3.0,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager.subprocess, "Popen", side_effect=wait_for_descendant,
                    ))
                    with self.assertRaisesRegex(
                        stager.SourceStageError, "zero gate differs",
                    ):
                        stager._execute_remote(
                            payload,
                            bootstrap_sha256="a" * 64,
                            payload_sha256="b" * 64,
                        )
                self.assertEqual(len(captured), 1)
                self.assertEqual(validation.call_count, 4)
                process = captured[0]
                self.assertIsNotNone(process.poll())
                with self.assertRaises(ProcessLookupError):
                    os.killpg(process.pid, 0)
                child_pid = int(ready.read_text(encoding="ascii"))
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                self.assertEqual(
                    [authority.descriptor for authority in authorities],
                    [-1, -1, -1],
                )
                self.assertFalse(fixture.target.exists())
                self.assertFalse(fixture.receipt.exists())
            finally:
                payload.close()
                for process in captured:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)

    def test_unobservable_exited_leader_still_reaps_descendant_group(self) -> None:
        with physical15_fixture() as fixture:
            ready = fixture.base / "early-exit-child.pid"
            child_source = (
                "import os,signal,sys,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "open(sys.argv[1],'w').write(str(os.getpid()));"
                "time.sleep(60)"
            )
            leader_source = (
                "import os,subprocess,sys,time;"
                "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);"
                "deadline=time.monotonic()+5;"
                "\nwhile not os.path.exists(sys.argv[2]):\n"
                "  if time.monotonic()>deadline: raise RuntimeError('child not ready')\n"
                "  time.sleep(.01)\n"
            )
            command = [
                sys.executable, "-c", leader_source,
                child_source, str(ready),
            ]
            captured = []
            real_popen = subprocess.Popen

            def wait_for_descendant(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                captured.append(process)
                deadline = time.monotonic() + 5
                while not ready.exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("descendant did not become ready")
                    time.sleep(0.01)
                return process

            payload = tempfile.TemporaryFile()
            try:
                transport = [
                    DummyTransportAuthority(payload.fileno()) for _ in range(3)
                ]
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_open_transport_authorities",
                        return_value=transport,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_validate_named_transport_authorities",
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "_ssh_arguments", return_value=command,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager.os,
                        "getpgid",
                        side_effect=ProcessLookupError(errno.ESRCH, "injected"),
                    ))
                    stack.enter_context(mock.patch.object(
                        stager.subprocess, "Popen", side_effect=wait_for_descendant,
                    ))
                    with self.assertRaisesRegex(
                        stager.SourceStageError,
                        "process-group creation was not observable",
                    ):
                        stager._execute_remote(
                            payload,
                            bootstrap_sha256="a" * 64,
                            payload_sha256="b" * 64,
                        )
                self.assertEqual(len(captured), 1)
                leader = captured[0]
                self.assertIsNotNone(leader.poll())
                self.assertTrue(leader.stdout.closed)
                self.assertTrue(leader.stderr.closed)
                child_pid = int(ready.read_text(encoding="ascii"))
                with self.assertRaises(ProcessLookupError):
                    os.killpg(leader.pid, 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                self.assertFalse(fixture.target.exists())
                self.assertFalse(fixture.receipt.exists())
            finally:
                payload.close()
                for process in captured:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)

    def test_timeout_reaps_leader_and_sigterm_ignoring_process_group(self) -> None:
        with physical15_fixture() as fixture:
            ready = fixture.base / "orphan-ready.pid"
            child_source = (
                "import os,signal,sys,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "open(sys.argv[1],'w').write(str(os.getpid()));"
                "time.sleep(60)"
            )
            leader_source = (
                "import os,subprocess,sys,time;"
                "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);"
                "deadline=time.monotonic()+5;"
                "\nwhile not os.path.exists(sys.argv[2]):\n"
                "  if time.monotonic()>deadline: raise RuntimeError('child not ready')\n"
                "  time.sleep(.01)\n"
            )
            command = [
                sys.executable, "-c", leader_source,
                child_source, str(ready),
            ]
            captured = []
            real_popen = subprocess.Popen

            def capture_process(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                captured.append(process)
                return process

            payload = tempfile.TemporaryFile()
            started = time.monotonic()
            try:
                transport = [
                    DummyTransportAuthority(payload.fileno()) for _ in range(3)
                ]
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_open_transport_authorities",
                        return_value=transport,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager,
                        "_validate_named_transport_authorities",
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "_ssh_arguments", return_value=command,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "TRANSPORT_TIMEOUT_SECONDS", 0.3,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "PROCESS_TERM_GRACE_SECONDS", 0.25,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager, "PROCESS_KILL_GRACE_SECONDS", 3.0,
                    ))
                    stack.enter_context(mock.patch.object(
                        stager.subprocess, "Popen", side_effect=capture_process,
                    ))
                    with self.assertRaisesRegex(
                        stager.SourceStageError, "timed out",
                    ):
                        stager._execute_remote(
                            payload,
                            bootstrap_sha256="a" * 64,
                            payload_sha256="b" * 64,
                        )
                self.assertLess(time.monotonic() - started, 8.0)
                self.assertEqual(len(captured), 1)
                leader = captured[0]
                self.assertIsNotNone(leader.poll())
                self.assertTrue(leader.stdout.closed)
                self.assertTrue(leader.stderr.closed)
                with self.assertRaises(ProcessLookupError):
                    os.killpg(leader.pid, 0)
                child_pid = int(ready.read_text(encoding="ascii"))
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                self.assertFalse(fixture.target.exists())
                self.assertFalse(fixture.receipt.exists())
            finally:
                payload.close()
                for process in captured:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)


if __name__ == "__main__":
    unittest.main()
