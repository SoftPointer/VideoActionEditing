#!/usr/bin/env python3
"""Hostile local-only tests for the inert CPU-controller deployer."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import base64
import ast
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


sys.dont_write_bytecode = True
METHOD_ROOT = Path(__file__).resolve().parents[1]
DEPLOYER_PATH = (
    METHOD_ROOT / "scripts"
    / "auh_deploy_case01_object_trajectory_world4_cpu_controller_v2_once_v1.HOLD.py"
)
BOOTSTRAP_PATH = (
    METHOD_ROOT / "tools"
    / "case01_object_trajectory_world4_cpu_controller_deploy_bootstrap_v1.py"
)
READY_CONTROLLER = (
    METHOD_ROOT
    / "case01_object_trajectory_exact5_world4_cpu_auh_controller_v2.READY.py"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bootstrap = load(BOOTSTRAP_PATH, "case01_controller_deploy_bootstrap_test_v1")
deployer = load(DEPLOYER_PATH, "case01_controller_deployer_hold_test_v1")


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


class DummyTransportAuthority:
    """A real retained fd with the minimal transport-authority test ABI."""

    def __init__(self, source_fd: int, path: Path | None = None) -> None:
        self.descriptor = os.dup(source_fd)
        self.path = Path("/dummy") if path is None else path

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


@contextmanager
def remote_fixture():
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary).resolve()
        target = parent / "controller-vessel"
        controller_name = "controller.READY.py"
        receipt = parent / "controller-vessel.deployment-receipt-v2.json"
        controller_raw = b"#!/usr/bin/env python3\nprint('frozen controller')\n"
        specifications = ({
            "relative": controller_name,
            "sha256": sha(controller_raw),
            "size": len(controller_raw),
        },)
        patches = (
            mock.patch.object(bootstrap, "REMOTE_PARENT", parent),
            mock.patch.object(bootstrap, "REMOTE_TARGET_ROOT", target),
            mock.patch.object(
                bootstrap, "REMOTE_CONTROLLER_PATH", target / controller_name,
            ),
            mock.patch.object(
                bootstrap, "REMOTE_RECEIPT_PATH", receipt,
            ),
            mock.patch.object(bootstrap, "CONTROLLER_BASENAME", controller_name),
            mock.patch.object(
                bootstrap, "SOURCE_RELATIVE_ALLOWLIST", (controller_name,),
            ),
            mock.patch.object(bootstrap, "REMOTE_UID", os.geteuid()),
            mock.patch.object(bootstrap, "REMOTE_GID", os.getegid()),
            mock.patch.object(bootstrap, "_validate_remote_runtime"),
        )
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            yield SimpleNamespace(
                parent=parent, target=target, receipt=receipt,
                controller_name=controller_name, controller_raw=controller_raw,
                specifications=specifications,
            )
        for child in tuple(parent.iterdir()):
            if child.is_symlink() or child.is_file():
                child.unlink(missing_ok=True)
            else:
                for current, directories, files in os.walk(
                    child, topdown=False, followlinks=False,
                ):
                    os.chmod(current, 0o700)
                    for name in files:
                        os.chmod(Path(current) / name, 0o600)
                shutil.rmtree(child)

def payload_for(fixture, *, operation="stage", terminal=None) -> tuple[bytes, str, str]:
    bootstrap_sha = "a" * 64
    manifest = bootstrap._manifest_value(fixture.specifications)
    source = SimpleNamespace(raw=fixture.controller_raw)
    value = bootstrap._payload_value(
        manifest, bootstrap_sha, [source],
        operation=operation, commit_terminal=terminal,
    )
    raw = bootstrap.canonical(value) + b"\n"
    return raw, sha(raw), bootstrap_sha


def portable_posix_rename(fixture):
    """Exercise the one ordinary rename despite Darwin's 0555 quirk.

    Darwin refuses to rename a directory that the owner cannot write in this
    local test environment.  Production performs exactly one ordinary
    same-parent POSIX rename.  This test seam changes only the held directory
    mode around that same single syscall and restores the sealed 0555 mode.
    """
    def publish(
        parent_fd, shadow_name, target_name, shadow_fd, shadow_anchor,
        reservation,
    ):
        self_anchor = bootstrap._inode_anchor(os.fstat(shadow_fd))
        if self_anchor != shadow_anchor:
            raise bootstrap.SourceStageError("portable held shadow differs")
        reservation.require_reserved(parent_fd)
        bootstrap._absent_at(parent_fd, target_name, label="target root")
        shadow = fixture.parent / shadow_name
        target = fixture.parent / target_name
        shadow.chmod(0o700)
        try:
            os.rename(
                shadow_name, target_name,
                src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
            )
        finally:
            if target.exists():
                target.chmod(0o555)
            elif shadow.exists():
                shadow.chmod(0o555)
        if not target.is_dir():
            raise bootstrap.SourceStageError("portable publication differs")

    return publish


class FrozenPinsAndHoldTests(unittest.TestCase):
    def test_exact_dynamic_pins_and_separate_bootstrap(self) -> None:
        ready_raw = READY_CONTROLLER.read_bytes()
        bootstrap_raw = BOOTSTRAP_PATH.read_bytes()
        self.assertEqual(
            (deployer.LOCAL_CONTROLLER_SHA256, deployer.LOCAL_CONTROLLER_SIZE),
            (sha(ready_raw), len(ready_raw)),
        )
        self.assertEqual(
            (deployer.LOCAL_BOOTSTRAP_SHA256, deployer.LOCAL_BOOTSTRAP_SIZE),
            (sha(bootstrap_raw), len(bootstrap_raw)),
        )
        self.assertNotEqual(DEPLOYER_PATH, BOOTSTRAP_PATH)
        self.assertEqual(
            deployer.REMOTE_CONTROLLER_PATH,
            deployer.REMOTE_TARGET_ROOT / deployer.CONTROLLER_BASENAME,
        )
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                deployer, "LOCAL_CONTROLLER_SHA256", sha(ready_raw),
            ))
            stack.enter_context(mock.patch.object(
                deployer, "LOCAL_CONTROLLER_SIZE", len(ready_raw),
            ))
            specifications = ({
                "relative": deployer.CONTROLLER_BASENAME,
                "sha256": sha(ready_raw),
                "size": len(ready_raw),
            },)
            self.assertEqual(
                deployer.manifest_value(),
                bootstrap._manifest_value(specifications),
            )
        self.assertNotIn(sha(ready_raw), BOOTSTRAP_PATH.read_text("utf-8"))

    def test_hold_returns_88_before_every_action_seam(self) -> None:
        self.assertNotEqual(deployer.CONTROLLER_STATE, deployer.READY_STATE)
        with ExitStack() as stack:
            opened = stack.enter_context(mock.patch.object(deployer.os, "open"))
            temporary = stack.enter_context(mock.patch.object(
                deployer.tempfile, "TemporaryFile",
            ))
            popen = stack.enter_context(mock.patch.object(
                deployer.subprocess, "Popen",
            ))
            controller = stack.enter_context(mock.patch.object(
                deployer, "controller",
            ))
            self.assertEqual(deployer.main(["--execute", "wrong"]), 88)
        opened.assert_not_called()
        temporary.assert_not_called()
        popen.assert_not_called()
        controller.assert_not_called()

    def test_named_hold_is_inert_normal_and_optimized_isolated(self) -> None:
        for flags in (("-I", "-S", "-B"), ("-O", "-I", "-S", "-B")):
            result = subprocess.run(
                [sys.executable, *flags, str(DEPLOYER_PATH), "hostile-argv"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
                env={"PATH": "/usr/bin:/bin", "HOME": "/private/tmp"},
            )
            self.assertEqual(result.returncode, 88, (flags, result.stderr))
            self.assertEqual(result.stdout, b"")
            self.assertIn(b"HOLD:", result.stderr)

    def test_sources_compile_without_assert_authority_normal_and_optimized(self) -> None:
        for path in (DEPLOYER_PATH, BOOTSTRAP_PATH):
            raw = path.read_bytes()
            self.assertNotIn(b"assert ", raw)
            for optimize in (0, 2):
                self.assertIsNotNone(
                    compile(raw, str(path), "exec", optimize=optimize),
                )
        self.assertNotIn("subprocess", BOOTSTRAP_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("srun", DEPLOYER_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("sbatch", DEPLOYER_PATH.read_text(encoding="utf-8"))


class PayloadAndTransportTests(unittest.TestCase):
    def test_canonical_payload_roundtrip_and_hostile_bytes_refuse(self) -> None:
        with remote_fixture() as fixture:
            raw, claimed, bootstrap_sha = payload_for(fixture)
            parsed, captured, specifications = bootstrap._parse_payload(
                raw, claimed_sha256=claimed,
                held_bootstrap_sha256=bootstrap_sha,
            )
            self.assertEqual(
                captured, {fixture.controller_name: fixture.controller_raw},
            )
            self.assertEqual(specifications, fixture.specifications)
            self.assertEqual(
                parsed["manifest"],
                bootstrap._manifest_value(fixture.specifications),
            )
            hostile = (
                raw + b" ",
                b'{"schema_version":1,"schema_version":1}\n',
                raw.replace(
                    sha(fixture.controller_raw).encode("ascii"), b"0" * 64, 1,
                ),
            )
            for candidate in hostile:
                with self.subTest(length=len(candidate)):
                    with self.assertRaises(
                        bootstrap.ControllerDeployBootstrapError,
                    ):
                        bootstrap._parse_payload(
                            candidate, claimed_sha256=sha(candidate),
                            held_bootstrap_sha256=bootstrap_sha,
                        )

    def test_actual_transport_authorities_are_held_but_consumed_by_name(self) -> None:
        transport = deployer._open_transport_authorities()
        try:
            deployer._validate_named_transport_authorities(transport)
            command = deployer._ssh_arguments("remote-command", transport)
            self.assertEqual(command[0], str(deployer.SSH_PATH))
            self.assertIn(
                f"IdentityFile={deployer.SSH_IDENTITY}", command,
            )
            self.assertIn(
                f"UserKnownHostsFile={deployer.SSH_KNOWN_HOSTS}",
                command,
            )
            self.assertNotIn("/dev/fd/", "\0".join(command))
            for authority in transport:
                authority.replay()
        finally:
            deployer._close_transport_authorities(transport)

    def test_ready_audit_is_local_only_and_authorities_replay(self) -> None:
        controller_raw = READY_CONTROLLER.read_bytes()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                deployer, "CONTROLLER_STATE", deployer.READY_STATE,
            ))
            stack.enter_context(mock.patch.object(
                deployer, "LOCAL_CONTROLLER_SHA256", sha(controller_raw),
            ))
            stack.enter_context(mock.patch.object(
                deployer, "LOCAL_CONTROLLER_SIZE", len(controller_raw),
            ))
            bootstrap_raw = BOOTSTRAP_PATH.read_bytes()
            stack.enter_context(mock.patch.object(
                deployer, "LOCAL_BOOTSTRAP_SHA256", sha(bootstrap_raw),
            ))
            stack.enter_context(mock.patch.object(
                deployer, "LOCAL_BOOTSTRAP_SIZE", len(bootstrap_raw),
            ))
            popen = stack.enter_context(mock.patch.object(
                deployer.subprocess, "Popen",
                side_effect=AssertionError("network forbidden in local audit"),
            ))
            result = deployer.controller(execute=False)
            expected_token = deployer.authorization_token()
        popen.assert_not_called()
        self.assertEqual(result["schema_version"], deployer.AUDIT_SCHEMA)
        self.assertEqual(result["authorization_token"], expected_token)
        self.assertIs(result["named_transport"], True)
        self.assertIs(result["held_fd_transport"], False)
        self.assertIs(result["credential_descriptor_transport"], False)
        self.assertIs(result["system_ssh_descriptor_exec"], False)
        self.assertEqual(
            result["transport_authority_replay_points"],
            ["after_open", "pre_spawn", "immediate_post_spawn", "post_reap"],
        )
        self.assertIs(
            result["same_uid_root_kernel_mount_attacker_out_of_scope"], True,
        )
        self.assertIs(
            result["residual_named_lookup_window_absence_claimed"], False,
        )
        self.assertFalse(result["slurm_allowed"])
        self.assertFalse(result["launch_allowed"])

    def test_remote_entry_is_env_empty_isolated_and_bounded(self) -> None:
        source = DEPLOYER_PATH.read_text(encoding="utf-8")
        self.assertIn('"/usr/bin/env", "-i", str(REMOTE_PYTHON)', source)
        self.assertIn('"-I", "-S", "-B", "-c"', source)
        self.assertIn("pass_fds=()", source)
        self.assertIn("close_fds=True", source)
        self.assertIn("start_new_session=True", source)
        self.assertIn(f"read({deployer.MAX_PAYLOAD_SIZE + 1})", deployer.REMOTE_LOADER_SOURCE)
        self.assertNotIn(".read()", deployer.REMOTE_LOADER_SOURCE)


@unittest.skip("superseded renameat2/two-file protocol fixture")
class RemoteBootstrapHostileTests(unittest.TestCase):
    def test_success_is_exact_sealed_create_only_and_read_back(self) -> None:
        with remote_fixture() as (
            parent, target, controller_name, receipt_name, controller_raw,
        ):
            payload_raw, payload_sha, bootstrap_sha = payload_for(controller_raw)
            with mock.patch.object(
                bootstrap, "_rename_noreplace",
                side_effect=portable_noreplace(parent),
            ):
                result = bootstrap.deploy(
                    payload_raw, payload_sha, bootstrap_sha,
                )
            self.assertEqual(result["status"], "DEPLOYED_CREATE_ONLY")
            self.assertTrue(target.is_dir())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o555)
            controller_path = target / controller_name
            receipt_path = target / receipt_name
            self.assertEqual(controller_path.read_bytes(), controller_raw)
            controller_info = controller_path.stat()
            self.assertEqual(stat.S_IMODE(controller_info.st_mode), 0o444)
            self.assertEqual(controller_info.st_nlink, 1)
            receipt_raw = receipt_path.read_bytes()
            receipt = json.loads(receipt_raw)
            self.assertEqual(receipt_raw, bootstrap.canonical(receipt) + b"\n")
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o400)
            self.assertEqual(receipt_path.stat().st_nlink, 1)
            self.assertEqual(
                sorted(path.name for path in target.iterdir()),
                sorted((controller_name, receipt_name)),
            )
            self.assertFalse(any(".shadow-" in path.name for path in parent.iterdir()))

    def test_preexisting_target_is_never_replaced(self) -> None:
        with remote_fixture() as (
            _parent, target, _controller_name, _receipt_name, controller_raw,
        ):
            target.mkdir()
            sentinel = target / "sentinel"
            sentinel.write_bytes(b"existing target\n")
            before = sentinel.read_bytes()
            payload_raw, payload_sha, bootstrap_sha = payload_for(controller_raw)
            with self.assertRaises(bootstrap.ControllerDeployBootstrapError):
                bootstrap.deploy(payload_raw, payload_sha, bootstrap_sha)
            self.assertEqual(sentinel.read_bytes(), before)

    def test_publish_race_preserves_racer_and_cleans_owned_shadow(self) -> None:
        with remote_fixture() as (
            parent, target, _controller_name, _receipt_name, controller_raw,
        ):
            payload_raw, payload_sha, bootstrap_sha = payload_for(controller_raw)

            def racer(parent_fd: int, shadow_name: str, target_name: str) -> None:
                os.mkdir(target_name, 0o700, dir_fd=parent_fd)
                racer_root_fd = os.open(
                    target_name, os.O_RDONLY | os.O_DIRECTORY,
                    dir_fd=parent_fd,
                )
                try:
                    marker_fd = os.open(
                        "racer", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600, dir_fd=racer_root_fd,
                    )
                    try:
                        os.write(marker_fd, b"racer owns target\n")
                    finally:
                        os.close(marker_fd)
                finally:
                    os.close(racer_root_fd)
                raise bootstrap.ControllerDeployBootstrapError(
                    "atomic publication target appeared"
                )

            with mock.patch.object(
                bootstrap, "_rename_noreplace", side_effect=racer,
            ):
                with self.assertRaises((
                    OSError, bootstrap.ControllerDeployBootstrapError,
                )):
                    bootstrap.deploy(payload_raw, payload_sha, bootstrap_sha)
            self.assertEqual((target / "racer").read_bytes(), b"racer owns target\n")
            self.assertFalse(any(".shadow-" in path.name for path in parent.iterdir()))

    def test_shadow_open_emfile_uses_reserve_and_leaves_no_partial(self) -> None:
        with remote_fixture() as (
            parent, _target, _controller_name, _receipt_name, controller_raw,
        ):
            payload_raw, payload_sha, bootstrap_sha = payload_for(controller_raw)
            real_open = bootstrap.os.open
            injected = {"done": False}

            def one_emfile(path, flags, *args, **kwargs):
                if (
                    not injected["done"] and isinstance(path, str)
                    and ".shadow-" in path
                    and flags & getattr(os, "O_DIRECTORY", 0)
                ):
                    injected["done"] = True
                    raise OSError(errno.EMFILE, "hostile descriptor exhaustion")
                return real_open(path, flags, *args, **kwargs)

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    bootstrap.os, "open", side_effect=one_emfile,
                ))
                stack.enter_context(mock.patch.object(
                    bootstrap, "_rename_noreplace",
                    side_effect=portable_noreplace(parent),
                ))
                result = bootstrap.deploy(
                    payload_raw, payload_sha, bootstrap_sha,
                )
            self.assertTrue(injected["done"])
            self.assertEqual(result["status"], "DEPLOYED_CREATE_ONLY")
            self.assertFalse(any(".shadow-" in path.name for path in parent.iterdir()))

    def test_first_post_mkdir_named_stat_fault_cleans_held_empty_shadow(self) -> None:
        with remote_fixture() as (
            parent, target, _controller_name, _receipt_name, controller_raw,
        ):
            payload_raw, payload_sha, bootstrap_sha = payload_for(controller_raw)
            real_stat = bootstrap.os.stat
            injected = {"done": False}

            def one_estale(path, *args, **kwargs):
                if (
                    not injected["done"] and isinstance(path, str)
                    and ".shadow-" in path
                ):
                    injected["done"] = True
                    raise OSError(errno.ESTALE, "hostile post-mkdir stat fault")
                return real_stat(path, *args, **kwargs)

            with mock.patch.object(
                bootstrap.os, "stat", side_effect=one_estale,
            ):
                with self.assertRaises((
                    OSError, bootstrap.ControllerDeployBootstrapError,
                )):
                    bootstrap.deploy(payload_raw, payload_sha, bootstrap_sha)
            self.assertTrue(injected["done"])
            self.assertFalse(target.exists())
            self.assertFalse(any(".shadow-" in path.name for path in parent.iterdir()))

    def test_first_write_failure_is_held_and_cleanup_is_exact(self) -> None:
        with remote_fixture() as (
            parent, target, _controller_name, _receipt_name, controller_raw,
        ):
            payload_raw, payload_sha, bootstrap_sha = payload_for(controller_raw)
            with mock.patch.object(
                bootstrap.os, "write",
                side_effect=OSError(errno.EIO, "hostile first write failure"),
            ):
                with self.assertRaises((
                    OSError, bootstrap.ControllerDeployBootstrapError,
                )):
                    bootstrap.deploy(payload_raw, payload_sha, bootstrap_sha)
            self.assertFalse(target.exists())
            self.assertFalse(any(".shadow-" in path.name for path in parent.iterdir()))

    def test_short_write_then_second_write_eio_cleans_held_partial_leaf(self) -> None:
        with remote_fixture() as (
            parent, target, _controller_name, _receipt_name, controller_raw,
        ):
            payload_raw, payload_sha, bootstrap_sha = payload_for(controller_raw)
            real_write = bootstrap.os.write
            calls = {"count": 0, "short_count": 0}

            def short_then_eio(descriptor, raw):
                calls["count"] += 1
                if calls["count"] == 1:
                    amount = max(1, len(raw) // 3)
                    written = real_write(descriptor, raw[:amount])
                    calls["short_count"] = written
                    return written
                raise OSError(errno.EIO, "hostile failure after short write")

            with mock.patch.object(
                bootstrap.os, "write", side_effect=short_then_eio,
            ):
                with self.assertRaises((
                    OSError, bootstrap.ControllerDeployBootstrapError,
                )):
                    bootstrap.deploy(payload_raw, payload_sha, bootstrap_sha)
            self.assertGreater(calls["short_count"], 0)
            self.assertGreaterEqual(calls["count"], 2)
            self.assertFalse(target.exists())
            self.assertFalse(any(".shadow-" in path.name for path in parent.iterdir()))

    def test_cleanup_preserves_replaced_child_and_rejects_missing_child(self) -> None:
        for replacement in (True, False):
            with self.subTest(replacement=replacement), remote_fixture() as (
                parent, target, controller_name, _receipt_name, controller_raw,
            ):
                payload_raw, payload_sha, bootstrap_sha = payload_for(controller_raw)
                original_write = bootstrap._write_at
                calls = {"count": 0}

                def replace_then_fail(
                    parent_fd, name, raw, *, mode, held_files,
                ):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        return original_write(
                            parent_fd, name, raw, mode=mode,
                            held_files=held_files,
                        )
                    os.unlink(controller_name, dir_fd=parent_fd)
                    if replacement:
                        foreign_fd = os.open(
                            controller_name,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o444, dir_fd=parent_fd,
                        )
                        try:
                            os.write(foreign_fd, b"foreign replacement\n")
                        finally:
                            os.close(foreign_fd)
                    raise bootstrap.ControllerDeployBootstrapError(
                        "hostile second-file interruption"
                    )

                with mock.patch.object(
                    bootstrap, "_write_at", side_effect=replace_then_fail,
                ):
                    with self.assertRaises(
                        bootstrap.ControllerDeployBootstrapError,
                    ):
                        bootstrap.deploy(
                            payload_raw, payload_sha, bootstrap_sha,
                        )
                self.assertFalse(target.exists())
                shadows = [
                    path for path in parent.iterdir() if ".shadow-" in path.name
                ]
                self.assertEqual(len(shadows), 1)
                if replacement:
                    self.assertEqual(
                        (shadows[0] / controller_name).read_bytes(),
                        b"foreign replacement\n",
                    )
                else:
                    self.assertFalse((shadows[0] / controller_name).exists())

    def test_final_named_target_replacement_cannot_return_success(self) -> None:
        with remote_fixture() as (
            parent, target, _controller_name, _receipt_name, controller_raw,
        ):
            payload_raw, payload_sha, bootstrap_sha = payload_for(controller_raw)
            original_reopen = bootstrap._reopen_named_directory
            replaced = {"done": False}

            def replace_final(parent_fd, name, held_fd, anchor, *, expected_mode):
                if name == target.name and not replaced["done"]:
                    replaced["done"] = True
                    target.chmod(0o700)
                    os.rename(target, parent / "published-original-held")
                    target.mkdir(mode=0o700)
                    target.chmod(bootstrap.DIRECTORY_MODE)
                return original_reopen(
                    parent_fd, name, held_fd, anchor,
                    expected_mode=expected_mode,
                )

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    bootstrap, "_rename_noreplace",
                    side_effect=portable_noreplace(parent),
                ))
                stack.enter_context(mock.patch.object(
                    bootstrap, "_reopen_named_directory",
                    side_effect=replace_final,
                ))
                with self.assertRaises(bootstrap.ControllerDeployBootstrapError):
                    bootstrap.deploy(payload_raw, payload_sha, bootstrap_sha)
            self.assertTrue(replaced["done"])
            self.assertTrue(target.is_dir())
            self.assertEqual(list(target.iterdir()), [])

    def test_symlink_and_special_local_authorities_fail_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            regular = root / "authority"
            regular.write_bytes(b"authority\n")
            regular.chmod(0o644)
            link = root / "link"
            link.symlink_to(regular)
            with self.assertRaises(deployer.ControllerDeployerError):
                deployer._open_authority(
                    link, sha256=sha(regular.read_bytes()),
                    size=regular.stat().st_size,
                    uid=os.geteuid(), gid=os.getegid(), mode=0o644,
                )
            if hasattr(os, "mkfifo"):
                fifo = root / "fifo"
                os.mkfifo(fifo)
                with self.assertRaises(deployer.ControllerDeployerError):
                    deployer._open_authority(
                        fifo, sha256="0" * 64, size=1,
                        uid=os.geteuid(), gid=os.getegid(), mode=0o644,
                    )


class ReceiptReservedNfsBootstrapTests(unittest.TestCase):
    @staticmethod
    def stage(fixture, *, portable=True):
        raw, claimed, bootstrap_sha = payload_for(fixture)
        if not portable:
            return bootstrap._remote_bootstrap(raw, claimed, bootstrap_sha)
        with mock.patch.object(
            bootstrap, "_rename_under_receipt_reservation",
            side_effect=portable_posix_rename(fixture),
        ):
            return bootstrap._remote_bootstrap(raw, claimed, bootstrap_sha)

    @staticmethod
    def recover(fixture, terminal):
        raw, claimed, bootstrap_sha = payload_for(
            fixture,
            operation=bootstrap.RECOVER_RECEIPT_OPERATION,
            terminal=terminal,
        )
        return bootstrap._remote_bootstrap(raw, claimed, bootstrap_sha)

    def test_success_is_atomic_one_file_plus_sibling_0400_admission(self) -> None:
        with remote_fixture() as fixture:
            receipt = self.stage(fixture)
            self.assertEqual(receipt["status"], "STAGED_RECEIPT_GATED")
            self.assertEqual(receipt["operation"], "stage")
            self.assertTrue(fixture.target.is_dir())
            self.assertEqual(stat.S_IMODE(fixture.target.stat().st_mode), 0o555)
            controller = fixture.target / fixture.controller_name
            self.assertEqual(controller.read_bytes(), fixture.controller_raw)
            self.assertEqual(stat.S_IMODE(controller.stat().st_mode), 0o444)
            self.assertEqual(controller.stat().st_nlink, 1)
            self.assertTrue(fixture.receipt.is_file())
            self.assertEqual(stat.S_IMODE(fixture.receipt.stat().st_mode), 0o400)
            self.assertEqual(fixture.receipt.stat().st_nlink, 1)
            receipt_raw = fixture.receipt.read_bytes()
            self.assertEqual(receipt_raw, bootstrap.canonical(receipt) + b"\n")
            self.assertIs(receipt["receipt_is_admission"], True)
            self.assertIs(receipt["receipt_is_consumption_gate"], True)
            self.assertIs(receipt["rename_noreplace"], False)
            self.assertEqual(
                receipt["publication_protocol"],
                "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation",
            )
            self.assertEqual(
                [path.name for path in fixture.target.iterdir()],
                [fixture.controller_name],
            )
            self.assertFalse(any(
                ".shadow-" in path.name for path in fixture.parent.iterdir()
            ))

    def test_target_and_receipt_competitors_are_preserved(self) -> None:
        for namespace in ("target", "receipt"):
            with self.subTest(namespace=namespace), remote_fixture() as fixture:
                if namespace == "target":
                    fixture.target.mkdir()
                    sentinel = fixture.target / "sentinel"
                    sentinel.write_bytes(b"target competitor\n")
                else:
                    fixture.receipt.write_bytes(b"receipt competitor\n")
                    fixture.receipt.chmod(0o600)
                    sentinel = fixture.receipt
                before = sentinel.read_bytes()
                with self.assertRaises(bootstrap.SourceStageError):
                    self.stage(fixture)
                self.assertEqual(sentinel.read_bytes(), before)
                if namespace == "receipt":
                    self.assertFalse(fixture.target.exists())
                self.assertFalse(any(
                    ".shadow-" in path.name for path in fixture.parent.iterdir()
                ))

    def test_partial_controller_write_cleans_exact_owned_names(self) -> None:
        with remote_fixture() as fixture:
            real_write = bootstrap.os.write
            calls = {"count": 0}

            def partial_then_fail(descriptor, raw):
                calls["count"] += 1
                if calls["count"] == 1:
                    amount = max(1, len(raw) // 3)
                    return real_write(descriptor, raw[:amount])
                raise OSError(errno.EIO, "injected partial write failure")

            with mock.patch.object(
                bootstrap.os, "write", side_effect=partial_then_fail,
            ):
                with self.assertRaises((OSError, bootstrap.SourceStageError)):
                    self.stage(fixture)
            self.assertGreaterEqual(calls["count"], 2)
            self.assertFalse(fixture.target.exists())
            self.assertFalse(fixture.receipt.exists())
            self.assertEqual(tuple(fixture.parent.iterdir()), ())

    def test_applied_then_error_rename_is_classified_without_retry(self) -> None:
        with remote_fixture() as fixture:
            real_rename = bootstrap.os.rename
            calls = {"count": 0}

            def applied_then_error(*args, **kwargs):
                calls["count"] += 1
                shadow_name = args[0]
                shadow = fixture.parent / shadow_name
                shadow.chmod(0o700)
                try:
                    real_rename(*args, **kwargs)
                finally:
                    if fixture.target.exists():
                        fixture.target.chmod(0o555)
                    elif shadow.exists():
                        shadow.chmod(0o555)
                raise OSError(errno.EIO, "injected applied-then-error")

            with mock.patch.object(
                bootstrap.os, "rename", side_effect=applied_then_error,
            ):
                terminal = self.stage(fixture, portable=False)
            self.assertEqual(calls["count"], 1)
            self.assertEqual(
                terminal["status"],
                "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
            )
            self.assertEqual(terminal["rename_result"], "raised")
            self.assertEqual(
                terminal["rename_classification"],
                "applied_then_error_target_is_held_shadow",
            )
            self.assertTrue(fixture.target.is_dir())
            self.assertEqual(stat.S_IMODE(fixture.receipt.stat().st_mode), 0o600)
            self.assertIs(terminal["zero_publication_claim"], False)

    def test_post_rename_parent_fsync_failure_returns_recoverable_terminal(self) -> None:
        with remote_fixture() as fixture:
            real_fsync = bootstrap.os.fsync
            injected = {"done": False}

            def fail_first_postcommit(descriptor):
                if fixture.target.exists() and not injected["done"]:
                    injected["done"] = True
                    raise OSError(errno.EIO, "injected post-rename parent fsync")
                return real_fsync(descriptor)

            with mock.patch.object(
                bootstrap.os, "fsync", side_effect=fail_first_postcommit,
            ):
                terminal = self.stage(fixture)
            self.assertTrue(injected["done"])
            self.assertEqual(
                terminal["status"],
                "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
            )
            self.assertEqual(terminal["receipt_phase"], "reserved_0600")
            self.assertIs(terminal["recovery_admissible"], True)
            self.assertEqual(stat.S_IMODE(fixture.receipt.stat().st_mode), 0o600)

    def test_explicit_recovery_seals_same_receipt_inode_without_rename(self) -> None:
        with remote_fixture() as fixture:
            real_fsync = bootstrap.os.fsync
            injected = {"done": False}

            def fail_first_postcommit(descriptor):
                if fixture.target.exists() and not injected["done"]:
                    injected["done"] = True
                    raise OSError(errno.EIO, "injected post-rename parent fsync")
                return real_fsync(descriptor)

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    bootstrap, "_rename_under_receipt_reservation",
                    side_effect=portable_posix_rename(fixture),
                ))
                stack.enter_context(mock.patch.object(
                    bootstrap.os, "fsync", side_effect=fail_first_postcommit,
                ))
                terminal = self.stage(fixture, portable=False)
            target_identity = bootstrap._identity(fixture.target.stat())
            receipt_anchor = bootstrap._inode_anchor(fixture.receipt.stat())
            with mock.patch.object(
                bootstrap.os, "rename",
                side_effect=AssertionError("recovery must not rename"),
            ) as renamed:
                recovered = self.recover(fixture, terminal)
            renamed.assert_not_called()
            self.assertEqual(recovered["status"], "RECOVERED_RECEIPT_ONLY")
            self.assertEqual(
                recovered["commit_terminal_digest"],
                terminal["terminal_digest"],
            )
            self.assertEqual(
                bootstrap._identity(fixture.target.stat()), target_identity,
            )
            self.assertEqual(
                bootstrap._inode_anchor(fixture.receipt.stat()), receipt_anchor,
            )
            self.assertEqual(stat.S_IMODE(fixture.receipt.stat().st_mode), 0o400)
            with ExitStack() as stack:
                seal = stack.enter_context(mock.patch.object(
                    bootstrap, "_seal_reserved_receipt",
                    side_effect=AssertionError("exact 0400 is verify-only"),
                ))
                renamed = stack.enter_context(mock.patch.object(
                    bootstrap.os, "rename",
                    side_effect=AssertionError("recovery must not rename"),
                ))
                replay = self.recover(fixture, terminal)
            seal.assert_not_called()
            renamed.assert_not_called()
            self.assertEqual(replay, recovered)

    def test_partial_recovery_write_returns_fresh_terminal_then_recovers(self) -> None:
        with remote_fixture() as fixture:
            real_fsync = bootstrap.os.fsync
            injected = {"done": False}

            def fail_first_postcommit(descriptor):
                if fixture.target.exists() and not injected["done"]:
                    injected["done"] = True
                    raise OSError(errno.EIO, "injected post-rename parent fsync")
                return real_fsync(descriptor)

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    bootstrap, "_rename_under_receipt_reservation",
                    side_effect=portable_posix_rename(fixture),
                ))
                stack.enter_context(mock.patch.object(
                    bootstrap.os, "fsync", side_effect=fail_first_postcommit,
                ))
                initial = self.stage(fixture, portable=False)
            receipt_anchor = bootstrap._inode_anchor(fixture.receipt.stat())
            real_seal = bootstrap._seal_reserved_receipt
            real_pwrite = bootstrap.os.pwrite

            def partial_seal(
                parent_fd, reservation, value, *, expected_prior_state=None,
            ):
                calls = {"count": 0}

                def partial_then_error(descriptor, content, offset):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        prefix = max(1, len(content) // 2)
                        return real_pwrite(
                            descriptor, content[:prefix], offset,
                        )
                    raise OSError(errno.EIO, "injected recovery pwrite")

                with mock.patch.object(
                    bootstrap.os, "pwrite", side_effect=partial_then_error,
                ):
                    return real_seal(
                        parent_fd, reservation, value,
                        expected_prior_state=expected_prior_state,
                    )

            with mock.patch.object(
                bootstrap, "_seal_reserved_receipt", side_effect=partial_seal,
            ):
                refreshed = self.recover(fixture, initial)
            self.assertEqual(
                refreshed["status"],
                "TARGET_RENAMED_RECEIPT_RECOVERY_REQUIRED",
            )
            self.assertEqual(refreshed["receipt_phase"], "partial_0600")
            self.assertEqual(
                bootstrap._inode_anchor(fixture.receipt.stat()), receipt_anchor,
            )
            recovered = self.recover(fixture, refreshed)
            self.assertEqual(recovered["status"], "RECOVERED_RECEIPT_ONLY")
            self.assertEqual(
                recovered["commit_terminal_digest"],
                refreshed["terminal_digest"],
            )
            self.assertEqual(
                bootstrap._inode_anchor(fixture.receipt.stat()), receipt_anchor,
            )

    def test_post_fchmod_fsync_failure_never_demotes_immutable_0400(self) -> None:
        with remote_fixture() as fixture:
            real_fsync = bootstrap.os.fsync
            injected = {"done": False}

            def fail_after_admission(descriptor):
                if (
                    fixture.receipt.exists()
                    and stat.S_IMODE(fixture.receipt.stat().st_mode) == 0o400
                    and not injected["done"]
                ):
                    injected["done"] = True
                    raise OSError(errno.EIO, "injected post-fchmod fsync")
                return real_fsync(descriptor)

            with mock.patch.object(
                bootstrap.os, "fsync", side_effect=fail_after_admission,
            ):
                result = self.stage(fixture)
            self.assertTrue(injected["done"])
            self.assertEqual(stat.S_IMODE(fixture.receipt.stat().st_mode), 0o400)
            # An independently exact 0400 replay is already admission; the
            # operation may return the receipt or an exact-bound terminal, but
            # it can never rewrite or demote that inode.
            self.assertIn(
                result["schema_version"],
                (bootstrap.RECEIPT_SCHEMA, bootstrap.TERMINAL_SCHEMA),
            )

    def test_root_and_child_replacements_are_preserved_and_refused(self) -> None:
        for kind in ("root", "child"):
            with self.subTest(kind=kind), remote_fixture() as fixture:
                real_build = bootstrap._build_shadow_at
                preserved: dict[str, Path] = {}

                def replace_then_fail(
                    shadow_fd, captured, creation_anchors, specifications,
                ):
                    sealed = real_build(
                        shadow_fd, captured, creation_anchors, specifications,
                    )
                    shadow = next(
                        path for path in fixture.parent.iterdir()
                        if path.is_dir() and ".shadow-" in path.name
                    )
                    if kind == "root":
                        moved = fixture.parent / "held-original-shadow"
                        shadow.chmod(0o700)
                        os.rename(shadow, moved)
                        shadow.mkdir(mode=0o700)
                        preserved["path"] = shadow
                    else:
                        child = shadow / fixture.controller_name
                        moved = shadow / "held-original-controller"
                        shadow.chmod(0o700)
                        child.chmod(0o600)
                        os.rename(child, moved)
                        child.write_bytes(b"foreign byte-identical-shaped replacement\n")
                        child.chmod(0o444)
                        shadow.chmod(0o555)
                        preserved["path"] = child
                    raise bootstrap.SourceStageError("injected replacement")

                with mock.patch.object(
                    bootstrap, "_build_shadow_at", side_effect=replace_then_fail,
                ):
                    with self.assertRaises(bootstrap.SourceStageError):
                        self.stage(fixture)
                self.assertFalse(fixture.target.exists())
                self.assertTrue(preserved["path"].exists())
                if kind == "child":
                    self.assertEqual(
                        preserved["path"].read_bytes(),
                        b"foreign byte-identical-shaped replacement\n",
                    )


class RecoveryAuthorityTests(unittest.TestCase):
    @staticmethod
    def valid_terminal():
        target_identity = [
            11, 22, deployer.REMOTE_UID, deployer.REMOTE_GID,
            stat.S_IFDIR | deployer.DIRECTORY_MODE, 2, 0, 0, 0, 1, 1,
        ]
        receipt_identity = [
            11, 23, deployer.REMOTE_UID, deployer.REMOTE_GID,
            stat.S_IFREG | deployer.RECEIPT_RESERVATION_MODE,
            1, 0, 127, 1, 1, 1,
        ]
        reservation_state = {
            "available": True,
            "inode_anchor": [
                receipt_identity[0], receipt_identity[1],
                receipt_identity[2], receipt_identity[3], stat.S_IFREG,
            ],
            "identity": receipt_identity,
            "mode": deployer.RECEIPT_RESERVATION_MODE,
            "size": receipt_identity[7],
            "sha256": "a" * 64,
        }
        return deployer._expected_commit_terminal(
            stage_payload_sha256="b" * 64,
            bootstrap_sha256="c" * 64,
            target_identity=tuple(target_identity),
            receipt_reservation_state=reservation_state,
            rename_result="returned_success",
            rename_classification="target_is_held_shadow",
            receipt_phase="partial_0600",
            receipt_authoritative=False,
            named_target_same_held_inode=True,
            recovery_admissible=True,
        )

    def test_remote_terminal_is_exact_key_phase_and_inode_closed(self) -> None:
        terminal = self.valid_terminal()
        raw = deployer.canonical(terminal) + b"\n"
        self.assertEqual(
            deployer._validate_remote_result(
                raw,
                operation=deployer.STAGE_OPERATION,
                request_payload_sha256="b" * 64,
                stage_payload_sha256="b" * 64,
                bootstrap_sha256="c" * 64,
                commit_terminal=None,
            ),
            terminal,
        )
        mutations = []
        extra = dict(terminal)
        extra["unexpected"] = False
        extra["terminal_digest"] = deployer.object_digest({
            key: value for key, value in extra.items()
            if key != "terminal_digest"
        })
        mutations.append(extra)
        unavailable = json.loads(json.dumps(terminal))
        unavailable["receipt_reservation_state"] = {"available": False}
        unsigned = dict(unavailable)
        unsigned.pop("terminal_digest")
        unavailable["terminal_digest"] = deployer.object_digest(unsigned)
        mutations.append(unavailable)
        wrong_phase = json.loads(json.dumps(terminal))
        wrong_phase["receipt_phase"] = "sealed_0400_exact"
        wrong_phase["receipt_authoritative"] = True
        unsigned = dict(wrong_phase)
        unsigned.pop("terminal_digest")
        wrong_phase["terminal_digest"] = deployer.object_digest(unsigned)
        mutations.append(wrong_phase)
        for value in mutations:
            with self.subTest(keys=sorted(value)):
                with self.assertRaises(deployer.ControllerDeployerError):
                    deployer._validate_remote_result(
                        deployer.canonical(value) + b"\n",
                        operation=deployer.STAGE_OPERATION,
                        request_payload_sha256="b" * 64,
                        stage_payload_sha256="b" * 64,
                        bootstrap_sha256="c" * 64,
                        commit_terminal=None,
                    )

    def test_local_terminal_is_create_only_0400_and_digest_chained(self) -> None:
        terminal = self.valid_terminal()
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            base = parent / "controller.commit-terminal.json"
            patches = (
                mock.patch.object(deployer, "LOCAL_COMMIT_TERMINAL_PATH", base),
                mock.patch.object(deployer, "LOCAL_UID", os.geteuid()),
                mock.patch.object(deployer, "LOCAL_GID", os.getegid()),
            )
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                held = deployer._write_local_commit_terminal(base, terminal)
                try:
                    value = held.replay()
                    self.assertEqual(
                        value["remote_commit_terminal"], terminal,
                    )
                    self.assertEqual(stat.S_IMODE(base.stat().st_mode), 0o400)
                    self.assertEqual(base.stat().st_nlink, 1)
                    with self.assertRaises(FileExistsError):
                        deployer._write_local_commit_terminal(base, terminal)
                finally:
                    held.close()
                reopened = deployer._open_local_commit_terminal(base)
                try:
                    self.assertEqual(reopened.replay(), value)
                finally:
                    reopened.close()
                refreshed = deployer._terminal_output_path(
                    deployer.RECOVER_RECEIPT_OPERATION, terminal,
                )
                self.assertEqual(
                    refreshed,
                    parent / (base.name + "." + terminal["terminal_digest"]),
                )

    def test_main_recovery_is_explicit_and_never_automatic(self) -> None:
        token = "d" * 64
        terminal_path = Path("/absolute/terminal.json")
        result = {"schema_version": deployer.RECEIPT_SCHEMA}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                deployer, "CONTROLLER_STATE", deployer.READY_STATE,
            ))
            stack.enter_context(mock.patch.object(
                deployer, "authorization_token", return_value=token,
            ))
            controller = stack.enter_context(mock.patch.object(
                deployer, "controller", return_value=result,
            ))
            printed = stack.enter_context(mock.patch("builtins.print"))
            self.assertEqual(deployer.main([
                deployer.RECOVER_RECEIPT_OPERATION,
                str(terminal_path), token,
            ]), 0)
        controller.assert_called_once_with(
            execute=True,
            operation=deployer.RECOVER_RECEIPT_OPERATION,
            terminal_path=terminal_path,
        )
        printed.assert_called_once_with(
            deployer.canonical(result).decode("utf-8")
        )

    def test_stage_rc75_persists_once_and_never_retries_remote(self) -> None:
        class FakeSource:
            def __init__(self, raw, digest):
                self.raw = raw
                self.sha256 = digest
                self.replays = 0
                self.closed = False

            def replay(self):
                self.replays += 1

            def close(self):
                self.closed = True

        class FakeTerminal:
            def __init__(self, value):
                self.value = value
                self.replays = 0
                self.closed = False

            def replay(self):
                self.replays += 1
                return self.value

            def close(self):
                self.closed = True

        controller_raw = READY_CONTROLLER.read_bytes()
        source = FakeSource(controller_raw, deployer.LOCAL_CONTROLLER_SHA256)
        boot = FakeSource(b"held bootstrap bytes", deployer.LOCAL_BOOTSTRAP_SHA256)
        captured = {}

        def remote_once(_held_input, *, bootstrap_sha256, payload_sha256):
            captured["terminal"] = self.valid_terminal()
            terminal = deployer._expected_commit_terminal(
                stage_payload_sha256=payload_sha256,
                bootstrap_sha256=bootstrap_sha256,
                target_identity=tuple(
                    captured["terminal"]["target_root_identity"]
                ),
                receipt_reservation_state=captured["terminal"][
                    "receipt_reservation_state"
                ],
                rename_result="returned_success",
                rename_classification="target_is_held_shadow",
                receipt_phase="partial_0600",
                receipt_authoritative=False,
                named_target_same_held_inode=True,
                recovery_admissible=True,
            )
            captured["terminal"] = terminal
            return deployer.canonical(terminal) + b"\n"

        persisted = {}

        def persist(path, terminal):
            persisted["path"] = path
            persisted["terminal"] = terminal
            wrapper = deployer._local_terminal_value(
                terminal, path,
                (1, 2, deployer.LOCAL_UID, deployer.LOCAL_GID, stat.S_IFREG),
            )
            persisted["held"] = FakeTerminal(wrapper)
            return persisted["held"]

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                deployer, "_open_sources", return_value=(source, boot),
            ))
            execute = stack.enter_context(mock.patch.object(
                deployer, "_execute_remote", side_effect=remote_once,
            ))
            writer = stack.enter_context(mock.patch.object(
                deployer, "_write_local_commit_terminal", side_effect=persist,
            ))
            with self.assertRaises(
                deployer.ControllerCommitRecoveryRequired,
            ) as caught:
                deployer.controller(execute=True)
        execute.assert_called_once()
        writer.assert_called_once()
        self.assertEqual(
            persisted["path"], deployer.LOCAL_COMMIT_TERMINAL_PATH,
        )
        self.assertEqual(
            caught.exception.terminal["remote_commit_terminal"],
            captured["terminal"],
        )
        self.assertTrue(source.closed)
        self.assertTrue(boot.closed)
        self.assertTrue(persisted["held"].closed)


class NamedTransportHostileTests(unittest.TestCase):
    def test_execute_uses_named_argv_empty_pass_fds_and_four_replays(self) -> None:
        class FakeProcess:
            pid = 424_242
            returncode = 0
            stdout = None
            stderr = None

            @staticmethod
            def communicate(*, timeout):
                self.assertEqual(timeout, deployer.TRANSPORT_TIMEOUT_SECONDS)
                return b"{}\n", b""

            @staticmethod
            def poll():
                return 0

        payload = tempfile.TemporaryFile()
        transport = [
            DummyTransportAuthority(payload.fileno(), path)
            for path in (
                deployer.SSH_PATH,
                deployer.SSH_IDENTITY,
                deployer.SSH_KNOWN_HOSTS,
            )
        ]
        fake_process = FakeProcess()
        try:
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    deployer, "_open_transport_authorities",
                    return_value=transport,
                ))
                validate = stack.enter_context(mock.patch.object(
                    deployer, "_validate_named_transport_authorities",
                ))
                popen = stack.enter_context(mock.patch.object(
                    deployer.subprocess, "Popen", return_value=fake_process,
                ))
                stack.enter_context(mock.patch.object(
                    deployer.os, "getpgid", return_value=FakeProcess.pid,
                ))
                seal = stack.enter_context(mock.patch.object(
                    deployer, "_seal_process_group",
                ))
                self.assertEqual(
                    deployer._execute_remote(
                        payload, bootstrap_sha256="a" * 64,
                        payload_sha256="b" * 64,
                    ),
                    b"{}\n",
                )
            popen.assert_called_once()
            seal.assert_called_once_with(fake_process, FakeProcess.pid)
            self.assertEqual(validate.call_count, 4)
            command = popen.call_args.args[0]
            kwargs = popen.call_args.kwargs
            self.assertEqual(command[0], "/usr/bin/ssh")
            self.assertNotIn("/dev/fd/", "\0".join(command))
            self.assertIn(
                f"IdentityFile={deployer.SSH_IDENTITY}", command,
            )
            self.assertIn(
                f"UserKnownHostsFile={deployer.SSH_KNOWN_HOSTS}", command,
            )
            self.assertIs(kwargs["close_fds"], True)
            self.assertEqual(kwargs["pass_fds"], ())
            self.assertFalse(kwargs.get("shell", False))
            self.assertEqual(kwargs["env"]["HOME"], "/var/empty")
            self.assertEqual(
                [authority.descriptor for authority in transport],
                [-1, -1, -1],
            )
        finally:
            for authority in transport:
                authority.close()
            payload.close()

    def test_popen_oserrors_are_single_attempt_and_close_all_authorities(self) -> None:
        for injected_errno in (errno.ENOENT, errno.EACCES, errno.ENOEXEC):
            with self.subTest(injected_errno=injected_errno):
                payload = tempfile.TemporaryFile()
                transport = [
                    DummyTransportAuthority(payload.fileno(), path)
                    for path in (
                        deployer.SSH_PATH,
                        deployer.SSH_IDENTITY,
                        deployer.SSH_KNOWN_HOSTS,
                    )
                ]
                try:
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(
                            deployer, "_open_transport_authorities",
                            return_value=transport,
                        ))
                        stack.enter_context(mock.patch.object(
                            deployer, "_validate_named_transport_authorities",
                        ))
                        popen = stack.enter_context(mock.patch.object(
                            deployer.subprocess, "Popen",
                            side_effect=OSError(
                                injected_errno, "injected named exec failure",
                            ),
                        ))
                        with self.assertRaises(OSError) as caught:
                            deployer._execute_remote(
                                payload, bootstrap_sha256="a" * 64,
                                payload_sha256="b" * 64,
                            )
                    self.assertEqual(caught.exception.errno, injected_errno)
                    popen.assert_called_once()
                    kwargs = popen.call_args.kwargs
                    self.assertIs(kwargs["close_fds"], True)
                    self.assertEqual(kwargs["pass_fds"], ())
                    self.assertEqual(
                        [authority.descriptor for authority in transport],
                        [-1, -1, -1],
                    )
                finally:
                    for authority in transport:
                        authority.close()
                    payload.close()

    def test_pre_and_immediate_post_spawn_failures_close_and_zero(self) -> None:
        class FakeProcess:
            pid = 434_343
            returncode = 0
            stdout = None
            stderr = None

            @staticmethod
            def poll():
                return 0

        for seam in ("pre", "post"):
            with self.subTest(seam=seam):
                payload = tempfile.TemporaryFile()
                transport = [
                    DummyTransportAuthority(payload.fileno(), path)
                    for path in (
                        deployer.SSH_PATH,
                        deployer.SSH_IDENTITY,
                        deployer.SSH_KNOWN_HOSTS,
                    )
                ]
                effects = (
                    (None, RuntimeError("injected pre-spawn replay"), None)
                    if seam == "pre" else
                    (
                        None, None,
                        RuntimeError("injected immediate-post replay"), None,
                    )
                )
                fake_process = FakeProcess()
                try:
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(
                            deployer, "_open_transport_authorities",
                            return_value=transport,
                        ))
                        validation = stack.enter_context(mock.patch.object(
                            deployer, "_validate_named_transport_authorities",
                            side_effect=effects,
                        ))
                        popen = stack.enter_context(mock.patch.object(
                            deployer.subprocess, "Popen",
                            return_value=fake_process,
                        ))
                        seal = stack.enter_context(mock.patch.object(
                            deployer, "_seal_process_group",
                        ))
                        with self.assertRaises((
                            RuntimeError, deployer.ControllerDeployerError,
                        )):
                            deployer._execute_remote(
                                payload, bootstrap_sha256="a" * 64,
                                payload_sha256="b" * 64,
                            )
                    if seam == "pre":
                        popen.assert_not_called()
                        seal.assert_not_called()
                        self.assertEqual(validation.call_count, 3)
                    else:
                        popen.assert_called_once()
                        seal.assert_called_once_with(
                            fake_process, FakeProcess.pid,
                        )
                        self.assertEqual(validation.call_count, 4)
                    self.assertEqual(
                        [authority.descriptor for authority in transport],
                        [-1, -1, -1],
                    )
                finally:
                    for authority in transport:
                        authority.close()
                    payload.close()

    def test_partial_open_and_aggregate_close_never_skip_later_fds(self) -> None:
        payload = tempfile.TemporaryFile()
        first = DummyTransportAuthority(payload.fileno())
        second = DummyTransportAuthority(payload.fileno())
        try:
            with mock.patch.object(
                deployer, "_open_authority",
                side_effect=(first, second, RuntimeError("injected open failure")),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected open failure"):
                    deployer._open_transport_authorities()
            self.assertEqual((first.descriptor, second.descriptor), (-1, -1))

            authorities = [
                CloseFailureTransportAuthority(payload.fileno()),
                DummyTransportAuthority(payload.fileno()),
                DummyTransportAuthority(payload.fileno()),
            ]
            with self.assertRaisesRegex(
                deployer.ControllerDeployerError, "authority close differs",
            ):
                deployer._close_transport_authorities(authorities)
            self.assertEqual(
                [authority.descriptor for authority in authorities],
                [-1, -1, -1],
            )
        finally:
            first.close()
            second.close()
            payload.close()

    def test_bounded_terminal_diagnostic_is_binary_safe_and_self_digested(self) -> None:
        raw = b"x" * (deployer.TRANSPORT_DIAGNOSTIC_PREFIX_LIMIT + 17)
        row = deployer._bounded_stream_diagnostic(raw)
        self.assertEqual(row["size"], len(raw))
        self.assertEqual(row["sha256"], sha(raw))
        self.assertEqual(
            base64.b64decode(row["prefix_b64"], validate=True),
            raw[:deployer.TRANSPORT_DIAGNOSTIC_PREFIX_LIMIT],
        )
        self.assertIs(row["truncated"], True)
        diagnostic = deployer._transport_terminal_diagnostic(
            reason="terminal_contract", returncode=255,
            stdout=b"partial\x00stdout\n", stderr=b"exact stderr\n",
            streams_complete=True,
        )
        unsigned = dict(diagnostic)
        claimed = unsigned.pop("diagnostic_digest")
        self.assertEqual(claimed, deployer.object_digest(unsigned))
        self.assertIs(diagnostic["remote_output_is_untrusted"], True)

    @unittest.skipUnless(sys.platform == "darwin", "Darwin sealed-system gate")
    def test_pinned_system_ssh_signature_structure_and_hostiles(self) -> None:
        raw = deployer.SSH_PATH.read_bytes()
        self.assertEqual(sha(raw), deployer.SSH_SHA256)
        self.assertEqual(
            deployer._macho_code_signature_ranges(raw),
            ((712_672, 24_256), (1_449_744, 24_384)),
        )

        def changed(offset, packed):
            mutated = bytearray(raw)
            mutated[offset:offset + len(packed)] = packed
            return bytes(mutated)

        x86_slice = deployer.SSH_FAT_ARCHITECTURES[0][2]
        x86_signature = 712_672
        code_directory = x86_signature + 52
        hostiles = (
            changed(4, struct.pack(">I", 1)),
            changed(16, struct.pack(">I", x86_slice + 1)),
            changed(x86_slice + 32 + 4, struct.pack("<I", 7)),
            changed(x86_signature + 8, struct.pack(">I", 6)),
            changed(code_directory + 88, b"x"),
            changed(code_directory + 36, b"\x14"),
            changed(code_directory + 38, b"\x00"),
        )
        for hostile in hostiles:
            with self.assertRaises(deployer.ControllerDeployerError):
                deployer._macho_code_signature_ranges(hostile)

    def test_byte_identical_named_credential_replacement_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "credential"
            original = root / "held-original"
            raw = b"synthetic private credential\n"
            path.write_bytes(raw)
            path.chmod(0o600)
            authority = deployer._open_authority(
                path, sha256=sha(raw), size=len(raw),
                uid=os.geteuid(), gid=os.getegid(), mode=0o600,
            )
            try:
                os.rename(path, original)
                path.write_bytes(raw)
                path.chmod(0o600)
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        deployer, "LOCAL_UID", os.geteuid(),
                    ))
                    stack.enter_context(mock.patch.object(
                        deployer, "LOCAL_GID", os.getegid(),
                    ))
                    with self.assertRaises(deployer.ControllerDeployerError):
                        deployer._validate_named_credential_authority(
                            authority, path, sha(raw), len(raw),
                        )
            finally:
                authority.close()
class CpuControllerReceiptGateStaticTests(unittest.TestCase):
    def test_receipt_gate_precedes_every_mutation_and_srun_in_controller(self) -> None:
        raw = READY_CONTROLLER.read_bytes()
        tree = ast.parse(raw, filename=str(READY_CONTROLLER))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        controller_node = functions["controller"]
        calls: list[tuple[str, int]] = []
        for node in ast.walk(controller_node):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            calls.append((name, node.lineno))
        gate_lines = [
            line for name, line in calls if name == "open_source_stage_gate"
        ]
        self.assertEqual(len(gate_lines), 1)
        gate_line = gate_lines[0]
        mutation_or_launch = {
            "_open_observed", "_open_pinned", "_fresh", "mkdir",
            "_create_json", "_run_single_srun",
        }
        protected = [
            line for name, line in calls if name in mutation_or_launch
        ]
        self.assertTrue(protected)
        self.assertLess(gate_line, min(protected))
        replay_lines = [
            line for name, line in calls if name == "replay"
        ]
        srun_lines = [
            line for name, line in calls if name == "_run_single_srun"
        ]
        self.assertTrue(any(line < srun_lines[0] for line in replay_lines))

        gate_raw = ast.get_source_segment(
            raw.decode("utf-8"), functions["open_source_stage_gate"],
        )
        self.assertIsNotNone(gate_raw)
        self.assertLess(
            gate_raw.index("SOURCE_RECEIPT_PATH"),
            gate_raw.index("SOURCE_ROOT"),
        )
        self.assertIn("expected_mode=SOURCE_STAGE_RECEIPT_MODE", gate_raw)


class ProcessGroupHostileTests(unittest.TestCase):
    @staticmethod
    def escaped_leader() -> tuple[subprocess.Popen[bytes], int]:
        source = (
            "import subprocess,sys\n"
            "p=subprocess.Popen(['/bin/sleep','30'],stdin=subprocess.DEVNULL,"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)\n"
            "print(p.pid,flush=True)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", "-c", source],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True,
        )
        if process.stdout is None:
            raise RuntimeError("test leader stdout is absent")
        descendant = int(process.stdout.readline().decode("ascii").strip())
        process.wait(timeout=5)
        return process, descendant

    def test_exited_leader_descendant_is_killed_to_group_esrch(self) -> None:
        process, descendant = self.escaped_leader()
        deployer._seal_process(process, process.pid)
        self.assertFalse(deployer._process_group_present(process.pid))
        # ESRCH on the exact process group is the authority; the descendant
        # PID itself may briefly remain as an init-owned zombie on some hosts.
        self.assertGreater(descendant, 1)

    def test_pipe_close_error_does_not_short_circuit_group_kill(self) -> None:
        process, _descendant = self.escaped_leader()
        with mock.patch.object(
            deployer, "_close_process_pipes",
            side_effect=RuntimeError("hostile close failure"),
        ):
            with self.assertRaises(deployer.ControllerDeployerError):
                deployer._seal_process(process, process.pid)
        deployer._close_process_pipes(process)
        self.assertFalse(deployer._process_group_present(process.pid))


if __name__ == "__main__":
    unittest.main()
