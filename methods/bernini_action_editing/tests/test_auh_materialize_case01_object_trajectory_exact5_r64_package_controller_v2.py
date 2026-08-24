#!/usr/bin/env python3
"""Hostile/static tests for the receipt-first r64 package HOLD controller."""

from __future__ import annotations

import ast
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = (
    METHOD_ROOT / "scripts"
    / "auh_materialize_case01_object_trajectory_exact5_r64_package_controller_v2_once_v1.HOLD.py"
)
READY_PATH = (
    METHOD_ROOT / "scripts"
    / "auh_materialize_case01_object_trajectory_exact5_r64_package_controller_v2_once_v1.READY.py"
)
MATERIALIZER_PATH = (
    METHOD_ROOT / "tools"
    / "materialize_case01_object_trajectory_exact5_r64_hold_package_v1.py"
)
MATERIALIZER_SHA256 = (
    "31c0184c8187fe0224c92bcb425dd0ec27731e7197898bd552aef82f83fa49f9"
)
MATERIALIZER_SIZE = 88_833


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> types.ModuleType:
    name = "_test_r64_package_hold_controller_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = load(CONTROLLER_PATH)


class FakeHeld:
    def __init__(
        self, path: Path, raw: bytes, events: list[str], descriptor: int,
    ) -> None:
        self.path = path
        self.raw = raw
        self.events = events
        self.descriptor = descriptor
        self.held_identity = tuple(range(11))

    def replay(self) -> None:
        self.events.append("replay:" + self.path.name)

    def close(self) -> None:
        self.events.append("close:" + self.path.name)

    def row(self) -> dict[str, object]:
        return {
            "path": str(self.path), "sha256": sha256(self.raw),
            "size": len(self.raw), "identity": list(self.held_identity),
        }


class FakeDirectory:
    def __init__(self, path: Path, events: list[str]) -> None:
        self.path = path
        self.events = events
        self.descriptor = 90
        self.held_identity = tuple(range(11))

    def replay(self) -> None:
        self.events.append("replay:root")

    def close(self) -> None:
        self.events.append("close:root")

    def row(self) -> dict[str, object]:
        return {"path": str(self.path), "identity": list(self.held_identity)}


class PackageControllerTest(unittest.TestCase):
    def test_hold_gate_is_first_and_crosses_no_controller_action(self) -> None:
        touched: list[str] = []

        def forbidden(*_args, **_kwargs):
            touched.append("called")
            raise AssertionError("HOLD crossed first state gate")

        stderr = io.StringIO()
        with mock.patch.object(controller, "blocked_dynamic_pins", forbidden), \
             mock.patch.object(controller, "controller", forbidden), \
             redirect_stderr(stderr):
            result = controller.main(["--execute", "malicious"])
        self.assertEqual(result, 88)
        self.assertEqual(touched, [])
        self.assertIn("HOLD", stderr.getvalue())

    def test_final_exact35_dynamic_pins_are_exact_and_complete(self) -> None:
        self.assertEqual(
            controller.dynamic_pin_values(),
            {
                "snapshot_manifest_sha256":
                    "da9c070e012ff11ebf5c61115d9949a573d37a99701fb7e6b8d7b2a6d5eee8f9",
                "snapshot_manifest_size": 10_889,
                "snapshot_manifest_digest":
                    "9114102687bf58291d6b96eddebc15557c669faa51e51bfde9b26a1aa7040968",
                "snapshot_receipt_sha256":
                    "ea7089857a8593734603d544aa6f3c238ea06abfd2da6775de7fea2adf0ce2a4",
                "snapshot_receipt_size": 2_072,
                "snapshot_receipt_digest":
                    "49ea7163e41382dff45ba59b0eb4b9e35480dead8426c7399d74da80b2f110a3",
                "snapshot_root_identity": [
                    48, 6200596844122101067, 2012, 2000, 16749, 2, 0,
                    4096, 0, 1787356256218061495, 1787356256279241444,
                ],
            },
        )
        self.assertEqual(controller.blocked_dynamic_pins(), ())
        token = controller.authorization_token()
        self.assertRegex(token, r"[0-9a-f]{64}\Z")
        with mock.patch.object(controller, "SNAPSHOT_MANIFEST_SHA256", "0" * 64):
            self.assertNotEqual(controller.authorization_token(), token)

    def test_static_pins_state_gate_proc_fd_and_single_popen(self) -> None:
        raw = CONTROLLER_PATH.read_bytes()
        ready_raw = READY_PATH.read_bytes()
        materializer_raw = MATERIALIZER_PATH.read_bytes()
        tree = ast.parse(raw, filename=str(CONTROLLER_PATH))
        self.assertEqual(
            (sha256(materializer_raw), len(materializer_raw)),
            (MATERIALIZER_SHA256, MATERIALIZER_SIZE),
        )
        self.assertEqual(controller.MATERIALIZER_SHA256, MATERIALIZER_SHA256)
        self.assertEqual(controller.MATERIALIZER_SIZE, MATERIALIZER_SIZE)
        self.assertEqual(
            controller.CONTROLLER_STATE,
            "HOLD_PENDING_INDEPENDENT_REVIEW_AND_STATE_COPY",
        )
        hold_assignment = (
            b'CONTROLLER_STATE = "HOLD_PENDING_INDEPENDENT_REVIEW_AND_STATE_COPY"\n'
        )
        ready_assignment = (
            b'CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_ATTEMPT_R64_HOLD_PACKAGE"\n'
        )
        self.assertEqual(raw.count(hold_assignment), 1)
        self.assertNotIn(ready_assignment, raw)
        self.assertEqual(ready_raw.count(ready_assignment), 1)
        self.assertNotIn(hold_assignment, ready_raw)
        self.assertEqual(
            ready_raw, raw.replace(hold_assignment, ready_assignment, 1),
        )
        differing = [
            (left, right)
            for left, right in zip(
                raw.splitlines(keepends=True), ready_raw.splitlines(keepends=True),
            )
            if left != right
        ]
        self.assertEqual(differing, [(hold_assignment, ready_assignment)])
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        main = functions["main"]
        self.assertIsInstance(main.body[0], ast.If)
        first_names = {
            node.id for node in ast.walk(main.body[0].test)
            if isinstance(node, ast.Name)
        }
        self.assertEqual(first_names, {"CONTROLLER_STATE", "READY_STATE"})
        popen_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Popen"
        ]
        self.assertEqual(len(popen_calls), 1)
        source = raw.decode("utf-8")
        self.assertIn('f"/proc/self/fd/{runtime_fd}"', source)
        self.assertIn('f"/proc/self/fd/{materializer_fd}"', source)
        self.assertIn(
            'f"/proc/self/fd/{runtime_fd}", "-I", "-S", "-B",', source,
        )
        self.assertIn("pass_fds=(runtime.descriptor, materializer.descriptor)", source)
        self.assertIn("close_fds=True", source)
        self.assertIn("start_new_session=True", source)
        self.assertLess(
            source.index("snapshot_receipt = open_authority("),
            source.index("snapshot_root = open_directory("),
        )
        self.assertLess(
            source.index("create_immutable_json(ATTEMPT_PATH, attempt)"),
            source.index("stdout, stderr, returncode, observed_argv"),
        )
        self.assertNotIn("os.rename(", source)
        self.assertNotIn("os.replace(", source)
        self.assertNotIn("shell=True", source)
        for path, source_raw in ((CONTROLLER_PATH, raw), (READY_PATH, ready_raw)):
            for optimize in (0, 2):
                self.assertIsNotNone(
                    compile(source_raw, str(path), "exec", optimize=optimize)
                )
            self.assertTrue(source_raw.endswith(b"\n"))
            self.assertFalse(source_raw.endswith(b"\n\n"))
            self.assertNotIn(b"\r", source_raw)

    def test_strict_json_and_child_stdout_are_exact(self) -> None:
        self.assertEqual(
            controller.strict_json(b'{"a":1,"b":2}\n', label="valid"),
            {"a": 1, "b": 2},
        )
        child = {"a": 1, "b": 2}
        child_raw = json.dumps(child, sort_keys=True).encode() + b"\n"
        self.assertEqual(controller.strict_child_stdout(child_raw), child)
        for raw in (
            b'{"a":1,"a":2}\n', b'{"b":2, "a":1}\n',
            b'{"a":NaN}\n', b'{"a":1}',
        ):
            with self.subTest(raw=raw), self.assertRaises(
                controller.PackageControllerError
            ):
                controller.strict_json(raw, label="hostile")
        for raw in (
            b'{"a":1,"a":2}\n', b'{"a":1}\n{"b":2}\n',
            controller.canonical(child) + b"\n",
        ):
            with self.subTest(child_raw=raw), self.assertRaises(
                controller.PackageControllerError
            ):
                controller.strict_child_stdout(raw)

    def test_held_authority_rejects_links_and_named_replacement(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve()
            path = base / "authority.json"
            raw = b'{"authority":true}\n'
            path.write_bytes(raw); path.chmod(0o400)
            info = path.stat()
            held = controller.open_authority(
                path, expected_sha256=sha256(raw), expected_size=len(raw),
                expected_mode=0o400, expected_uid=info.st_uid,
                expected_gid=info.st_gid,
            )
            held.replay()
            replacement = base / "replacement.json"
            replacement.write_bytes(raw); replacement.chmod(0o400)
            os.replace(replacement, path)
            with self.assertRaisesRegex(
                controller.PackageControllerError, "held authority changed"
            ):
                held.replay()
            held.close()
            link = base / "link.json"; link.symlink_to(path)
            with self.assertRaises(controller.PackageControllerError):
                controller.open_authority(
                    link, expected_sha256=sha256(raw), expected_size=len(raw),
                    expected_mode=0o400, expected_uid=info.st_uid,
                    expected_gid=info.st_gid,
                )
            hardlink = base / "hardlink.json"; os.link(path, hardlink)
            with self.assertRaises(controller.PackageControllerError):
                controller.open_authority(
                    path, expected_sha256=sha256(raw), expected_size=len(raw),
                    expected_mode=0o400, expected_uid=info.st_uid,
                    expected_gid=info.st_gid,
                )

    def test_create_only_seals_and_postseal_error_never_demotes(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve(); info = base.stat()
            target = base / "attempt.json"
            with mock.patch.object(controller, "EXPERIMENTS", base), \
                 mock.patch.object(controller, "REMOTE_UID", info.st_uid), \
                 mock.patch.object(controller, "REMOTE_GID", info.st_gid):
                raw, anchor = controller.create_immutable_json(
                    target, {"schema_version": "test", "status": "CLAIMED"},
                )
                self.assertEqual(target.read_bytes(), raw)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)
                self.assertEqual(anchor, controller.inode_anchor(target.stat()))
                with self.assertRaises(FileExistsError):
                    controller.create_immutable_json(target, {"replacement": True})

        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve(); info = base.stat()
            target = base / "postseal.json"
            real_fsync = os.fsync
            calls = 0

            def fail_parent_fsync(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("hostile parent fsync after seal")
                real_fsync(descriptor)

            expected = controller.canonical({
                "schema_version": "test", "status": "CLAIMED",
            }) + b"\n"
            with mock.patch.object(controller, "EXPERIMENTS", base), \
                 mock.patch.object(controller, "REMOTE_UID", info.st_uid), \
                 mock.patch.object(controller, "REMOTE_GID", info.st_gid), \
                 mock.patch.object(controller.os, "fsync", fail_parent_fsync), \
                 self.assertRaisesRegex(OSError, "hostile parent fsync"):
                controller.create_immutable_json(
                    target, {"schema_version": "test", "status": "CLAIMED"},
                )
            self.assertEqual(target.read_bytes(), expected)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)

    def test_run_child_uses_held_proc_fds_once(self) -> None:
        events: list[str] = []
        runtime = FakeHeld(controller.VACE_PYTHON, b"python", events, 71)
        materializer = FakeHeld(controller.MATERIALIZER_PATH, b"source", events, 72)

        class Process:
            returncode = 0
            pid = 4242

            def communicate(self, timeout):
                self.timeout = timeout
                return b'{"status": "PASS"}\n', b""

            def poll(self):
                return self.returncode

        process = Process()
        with mock.patch.object(
            controller.subprocess, "Popen", return_value=process,
        ) as popen, mock.patch.object(
            controller, "_process_group_absent", return_value=True,
        ):
            stdout, stderr, returncode, argv = controller.run_one_materializer(
                runtime, materializer,
            )
        self.assertEqual((stdout, stderr, returncode), (
            b'{"status": "PASS"}\n', b"", 0,
        ))
        self.assertEqual(argv[0:5], [
            "/proc/self/fd/71", "-I", "-S", "-B", "/proc/self/fd/72",
        ])
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args, (argv,))
        kwargs = popen.call_args.kwargs
        self.assertEqual(kwargs["executable"], "/proc/self/fd/71")
        self.assertEqual(kwargs["pass_fds"], (71, 72))
        self.assertTrue(kwargs["close_fds"])
        self.assertTrue(kwargs["start_new_session"])

    def test_child_timeout_terminates_group_without_retry(self) -> None:
        events: list[str] = []
        runtime = FakeHeld(controller.VACE_PYTHON, b"python", events, 81)
        materializer = FakeHeld(controller.MATERIALIZER_PATH, b"source", events, 82)

        class Process:
            returncode = None
            pid = 5151
            calls = 0

            def communicate(self, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("materializer", timeout)
                self.returncode = -signal.SIGTERM
                return b"", b""

            def poll(self):
                return self.returncode

        process = Process()
        with mock.patch.object(
            controller.subprocess, "Popen", return_value=process,
        ) as popen, mock.patch.object(controller.os, "killpg") as killpg, \
             mock.patch.object(
                 controller, "_process_group_absent", return_value=True,
             ), \
             self.assertRaisesRegex(
                 controller.PackageControllerError, "timed out",
             ):
            controller.run_one_materializer(runtime, materializer)
        popen.assert_called_once()
        killpg.assert_called_once_with(5151, signal.SIGTERM)

    def _run_mocked_controller(self, *, child_error: Exception | None = None):
        events: list[str] = []
        receipt = FakeHeld(controller.SNAPSHOT_RECEIPT_PATH, b"receipt", events, 21)
        manifest = FakeHeld(controller.SNAPSHOT_MANIFEST_PATH, b"manifest", events, 22)
        materializer_held = FakeHeld(
            controller.MATERIALIZER_PATH, b"materializer", events, 23,
        )
        runtime = FakeHeld(controller.VACE_PYTHON, b"python", events, 24)
        self_held = FakeHeld(CONTROLLER_PATH, b"self", events, 25)
        root = FakeDirectory(controller.SNAPSHOT_ROOT, events)
        snapshot_bytes = {controller.MATERIALIZER_RELATIVE: b"materializer"}
        snapshot_evidence = {
            "sha256": controller.SNAPSHOT_MANIFEST_SHA256,
            "size": controller.SNAPSHOT_MANIFEST_SIZE,
            "manifest_digest": controller.SNAPSHOT_MANIFEST_DIGEST,
            "snapshot_publication_receipt": {
                "sha256": controller.SNAPSHOT_RECEIPT_SHA256,
                "size": controller.SNAPSHOT_RECEIPT_SIZE,
                "receipt_digest": controller.SNAPSHOT_RECEIPT_DIGEST,
            },
            "staging_receipt_authority": {
                "sha256": "0" * 64, "receipt_digest": "1" * 64,
            },
        }

        class Materializer:
            RELEASE_FILES = {}
            DIAGNOSTIC_FILES = {}
            SNAPSHOT_AUTHORITY_FILES = {}
            SOURCE_VIDEO = ("videos/exact_original.mp4", "2" * 64, 1)
            AUX_VIDEO = ("videos/bone_removed.mp4", "3" * 64, 1)
            calls = 0

            def preflight_snapshot(self, *_args, **_kwargs):
                self.calls += 1
                events.append("preflight")
                return snapshot_bytes, snapshot_evidence

        materializer = Materializer()
        held_by_path = {
            controller.SNAPSHOT_RECEIPT_PATH: receipt,
            controller.SNAPSHOT_MANIFEST_PATH: manifest,
            controller.MATERIALIZER_PATH: materializer_held,
        }

        def open_side_effect(path, **_kwargs):
            events.append("open:" + Path(path).name)
            return held_by_path[path]

        def directory_side_effect(path, **_kwargs):
            events.append("open:root")
            self.assertEqual(path, controller.SNAPSHOT_ROOT)
            return root

        created: list[tuple[Path, dict[str, object]]] = []

        def create_side_effect(path, value):
            events.append("create:" + path.name)
            copied = dict(value); created.append((path, copied))
            return controller.canonical(copied) + b"\n", [1, 2, 3, 4, 5]

        child_report = {"child": "report"}
        child_stdout = json.dumps(child_report, sort_keys=True).encode() + b"\n"

        def run_side_effect(*_args):
            events.append("child")
            if child_error is not None:
                raise child_error
            return (
                child_stdout, b"", 0,
                controller.materializer_argv(runtime.descriptor, materializer_held.descriptor),
            )

        def package_side_effect(child, bytes_value, evidence, module):
            events.append("validate:package")
            self.assertEqual(child, child_report)
            self.assertIs(bytes_value, snapshot_bytes)
            self.assertIs(evidence, snapshot_evidence)
            self.assertIs(module, materializer)
            return {"file_count": 39, "directory_count": 18}

        patches = (
            mock.patch.object(controller, "open_authority", side_effect=open_side_effect),
            mock.patch.object(controller, "validate_snapshot_receipt", return_value={}),
            mock.patch.object(controller, "open_directory", side_effect=directory_side_effect),
            mock.patch.object(controller, "validate_snapshot_manifest", return_value={}),
            mock.patch.object(controller, "load_materializer", return_value=materializer),
            mock.patch.object(controller, "open_runtime_authority", return_value=runtime),
            mock.patch.object(controller, "open_self_authority", return_value=self_held),
            mock.patch.object(controller, "require_fresh_outputs"),
            mock.patch.object(controller, "create_immutable_json", side_effect=create_side_effect),
            mock.patch.object(controller, "run_one_materializer", side_effect=run_side_effect),
            mock.patch.object(controller, "validate_package", side_effect=package_side_effect),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9], patches[10]:
            if child_error is None:
                result = controller.controller()
                return result, events, created, materializer
            with self.assertRaisesRegex(
                controller.PackageControllerError,
                "failed after the durable attempt claim",
            ):
                controller.controller()
        return None, events, created, materializer

    def test_receipt_root_manifest_order_attempt_then_one_child(self) -> None:
        result, events, created, materializer = self._run_mocked_controller()
        self.assertLess(
            events.index("open:" + controller.SNAPSHOT_RECEIPT_PATH.name),
            events.index("open:root"),
        )
        self.assertLess(
            events.index("open:root"),
            events.index("open:" + controller.SNAPSHOT_MANIFEST_PATH.name),
        )
        self.assertLess(
            events.index("create:" + controller.ATTEMPT_PATH.name),
            events.index("child"),
        )
        self.assertLess(events.index("child"), events.index("validate:package"))
        self.assertEqual(events.count("child"), 1)
        self.assertEqual(materializer.calls, 2)
        self.assertEqual([path for path, _value in created], [
            controller.ATTEMPT_PATH, controller.EVIDENCE_PATH,
        ])
        self.assertEqual(
            created[0][1]["status"],
            "ATTEMPT_CLAIMED_BEFORE_MATERIALIZER_CHILD",
        )
        self.assertEqual(
            created[1][1]["status"],
            "PASS_R64_HOLD_PACKAGE_RECEIPT_GATED",
        )
        self.assertFalse(result["launch_allowed"])
        self.assertFalse(result["retry_allowed"])

    def test_child_failure_leaves_only_attempt_and_never_retries(self) -> None:
        result, events, created, materializer = self._run_mocked_controller(
            child_error=RuntimeError("hostile child refusal")
        )
        self.assertIsNone(result)
        self.assertEqual(events.count("child"), 1)
        self.assertEqual(materializer.calls, 1)
        self.assertEqual([path for path, _value in created], [controller.ATTEMPT_PATH])
        self.assertNotIn("validate:package", events)


if __name__ == "__main__":
    unittest.main()
